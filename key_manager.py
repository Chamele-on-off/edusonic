import time
from datetime import datetime, timedelta
from typing import List, Dict
import json
from pathlib import Path

class APIKeyManager:
    def __init__(self, config_file: str = "api_keys.json"):
        self.config_file = Path(config_file)
        self.keys = self._load_keys()
        self.current_key_index = 0
        self.daily_limit = 40  # Лимит запросов на ключ в день
        self.reset_time = None
        
    def _load_keys(self) -> List[Dict]:
        """Загрузка ключей из файла конфигурации"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('keys', [])
            except Exception as e:
                print(f"❌ Ошибка загрузки ключей: {e}")
        
        # Если файла нет, используем ключ из config.py
        from config import get_api_key
        openrouter_key = get_api_key('openrouter')
        if openrouter_key:
            return [{
                'key': openrouter_key,
                'usage': 0,
                'last_reset': datetime.now().isoformat(),
                'name': 'default_key'
            }]
        return []
    
    def _save_keys(self):
        """Сохранение ключей в файл"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump({'keys': self.keys}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ Ошибка сохранения ключей: {e}")
    
    def add_key(self, api_key: str, name: str = "new_key"):
        """Добавление нового ключа"""
        self.keys.append({
            'key': api_key.strip(),
            'usage': 0,
            'last_reset': datetime.now().isoformat(),
            'name': name
        })
        self._save_keys()
        print(f"✅ Добавлен ключ: {name}")
    
    def get_next_key(self) -> str:
        """Получение следующего доступного ключа"""
        if not self.keys:
            raise Exception("❌ Нет доступных API ключей")
        
        # Проверяем сброс дневного лимита
        self._check_daily_reset()
        
        # Ищем ключ с доступными запросами
        for _ in range(len(self.keys)):
            key_data = self.keys[self.current_key_index]
            
            if key_data['usage'] < self.daily_limit:
                return key_data['key']
            
            # Переходим к следующему ключу
            self.current_key_index = (self.current_key_index + 1) % len(self.keys)
        
        # Если все ключи исчерпаны
        raise Exception("❌ Все API ключи исчерпали дневной лимит")
    
    def record_usage(self, api_key: str):
        """Запись использования ключа"""
        for key_data in self.keys:
            if key_data['key'] == api_key:
                key_data['usage'] += 1
                break
        self._save_keys()
    
    def _check_daily_reset(self):
        """Проверка и сброс дневного лимита"""
        now = datetime.now()
        
        for key_data in self.keys:
            last_reset = datetime.fromisoformat(key_data['last_reset'])
            
            # Если прошло больше 24 часов, сбрасываем счетчик
            if now - last_reset > timedelta(hours=24):
                key_data['usage'] = 0
                key_data['last_reset'] = now.isoformat()
                print(f"🔄 Сброс лимита для ключа: {key_data['name']}")
        
        self._save_keys()
    
    def get_usage_stats(self) -> Dict:
        """Получение статистики использования"""
        total_used = sum(key['usage'] for key in self.keys)
        total_available = len(self.keys) * self.daily_limit - total_used
        
        return {
            'total_keys': len(self.keys),
            'daily_limit_per_key': self.daily_limit,
            'total_used_today': total_used,
            'total_available_today': total_available,
            'keys': [
                {
                    'name': key['name'],
                    'used': key['usage'],
                    'available': self.daily_limit - key['usage'],
                    'last_reset': key['last_reset']
                }
                for key in self.keys
            ]
        }

# Глобальный экземпляр
key_manager = APIKeyManager()

def get_key_manager() -> APIKeyManager:
    return key_manager