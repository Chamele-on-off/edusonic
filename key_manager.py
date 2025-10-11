# key_manager.py
import time
from datetime import datetime, timedelta
from typing import List, Dict
from config import load_config, save_config

class APIKeyManager:
    def __init__(self):
        self.config = load_config()
        self.keys: List[Dict] = self._load_keys()
        self.current_key_index = 0
        self.daily_limit = 40  # лимит запросов в день на ключ
        self.reset_time = None
        self._setup_daily_reset()
    
    def _load_keys(self) -> List[Dict]:
        """Загружает ключи из конфигурации"""
        keys = []
        
        # Основной ключ из конфига
        main_key = self.config.get("openrouter", {}).get("api_key", "")
        if main_key:
            keys.append({
                "key": main_key,
                "requests_today": 0,
                "last_reset": datetime.now().date(),
                "total_requests": 0,
                "is_active": True
            })
        
        # Дополнительные ключи из расширенной конфигурации
        extra_keys = self.config.get("openrouter_keys", [])
        for key_data in extra_keys:
            if isinstance(key_data, dict) and key_data.get("key"):
                keys.append({
                    "key": key_data["key"],
                    "requests_today": key_data.get("requests_today", 0),
                    "last_reset": datetime.now().date(),
                    "total_requests": key_data.get("total_requests", 0),
                    "is_active": key_data.get("is_active", True)
                })
        
        return keys
    
    def _setup_daily_reset(self):
        """Настраивает ежедневный сброс счетчиков"""
        now = datetime.now()
        self.reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        print(f"🔧 Следующий сброс счетчиков в: {self.reset_time}")
    
    def _check_and_reset_daily_counts(self):
        """Проверяет и сбрасывает дневные счетчики если нужно"""
        now = datetime.now()
        if now >= self.reset_time:
            print("🔄 Сброс дневных счетчиков запросов")
            for key_data in self.keys:
                key_data["requests_today"] = 0
                key_data["last_reset"] = now.date()
            self._setup_daily_reset()
            self._save_keys()
    
    def get_current_key(self) -> str:
        """Возвращает текущий активный ключ"""
        if not self.keys:
            raise Exception("Нет доступных API ключей")
        
        self._check_and_reset_daily_counts()
        
        # Ищем ключ с доступными запросами
        for attempt in range(len(self.keys)):
            key_data = self.keys[self.current_key_index]
            
            if key_data["is_active"] and key_data["requests_today"] < self.daily_limit:
                return key_data["key"]
            
            # Переходим к следующему ключу
            self.current_key_index = (self.current_key_index + 1) % len(self.keys)
        
        # Если все ключи исчерпаны
        raise Exception("Все API ключи исчерпали дневной лимит")
    
    def record_request(self, success: bool = True):
        """Записывает использование текущего ключа"""
        if not self.keys:
            return
        
        key_data = self.keys[self.current_key_index]
        key_data["requests_today"] += 1
        key_data["total_requests"] += 1
        
        if not success:
            # Помечаем ключ как неактивный при ошибках
            key_data["is_active"] = False
            print(f"⚠️ Ключ {self._mask_key(key_data['key'])} помечен как неактивный")
        
        self._save_keys()
    
    def add_key(self, api_key: str):
        """Добавляет новый ключ"""
        new_key = {
            "key": api_key.strip(),
            "requests_today": 0,
            "last_reset": datetime.now().date(),
            "total_requests": 0,
            "is_active": True
        }
        
        self.keys.append(new_key)
        self._save_keys()
        print(f"✅ Добавлен новый API ключ: {self._mask_key(api_key)}")
    
    def _save_keys(self):
        """Сохраняет ключи в конфигурацию"""
        if "openrouter_keys" not in self.config:
            self.config["openrouter_keys"] = []
        
        # Обновляем конфигурацию
        self.config["openrouter_keys"] = [
            {
                "key": key_data["key"],
                "requests_today": key_data["requests_today"],
                "total_requests": key_data["total_requests"],
                "is_active": key_data["is_active"]
            }
            for key_data in self.keys
        ]
        
        save_config(self.config)
    
    def _mask_key(self, key: str) -> str:
        """Маскирует ключ для безопасного логирования"""
        if len(key) > 8:
            return key[:4] + "..." + key[-4:]
        return "***"
    
    def get_status(self) -> Dict:
        """Возвращает статус всех ключей"""
        self._check_and_reset_daily_counts()
        
        return {
            "total_keys": len(self.keys),
            "active_keys": sum(1 for k in self.keys if k["is_active"]),
            "current_key_index": self.current_key_index,
            "daily_limit": self.daily_limit,
            "keys": [
                {
                    "masked_key": self._mask_key(k["key"]),
                    "requests_today": k["requests_today"],
                    "remaining_today": self.daily_limit - k["requests_today"],
                    "total_requests": k["total_requests"],
                    "is_active": k["is_active"],
                    "last_reset": k["last_reset"].isoformat()
                }
                for k in self.keys
            ]
        }

# Глобальный экземпляр
key_manager = APIKeyManager()

def get_key_manager():
    return key_manager
