import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
from pathlib import Path
import threading
import re
from config import load_config, save_config
import hashlib

class APIKeyManager:
    def __init__(self, config_file: str = "api_keys.json"):
        self.config_file = Path(config_file)
        self.keys = self._load_keys()
        self.current_key_index = 0
        self.daily_limit = 40  # Стандартный лимит - 40 запросов в день
        self.extended_limit = 900  # Расширенный лимит - 900 запросов в день
        self.reset_time = "00:00"  # Время сброса счетчиков
        self.reset_check_interval = 60  # Проверка каждые 60 секунд
        self.model = "meta-llama/llama-3.3-70b-instruct:free"  # Текущая модель
        
        # Запускаем фоновую проверку сброса
        self._start_reset_checker()
        
    def _load_keys(self) -> List[Dict]:
        """Загрузка ключей из файла конфигурации"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    keys = data.get('keys', [])
                    
                    # Загружаем настройки лимитов и времени сброса
                    if 'settings' in data:
                        self.daily_limit = data['settings'].get('default_limit', 40)
                        self.extended_limit = data['settings'].get('extended_limit', 900)
                        self.reset_time = data['settings'].get('reset_time', '00:00')
                        self.model = data['settings'].get('model', 'meta-llama/llama-3.3-70b-instruct:free')
                    
                    # Обновляем структуру ключей если нужно
                    for key in keys:
                        if 'limit_type' not in key:
                            key['limit_type'] = 'standard'
                        if 'limit' not in key:
                            key['limit'] = self.daily_limit if key['limit_type'] == 'standard' else self.extended_limit
                        if 'is_active' not in key:
                            key['is_active'] = True
                        if 'created_at' not in key:
                            key['created_at'] = datetime.now().isoformat()
                        if 'last_reset' not in key:
                            key['last_reset'] = datetime.now().isoformat()
                        if 'total_requests' not in key:
                            key['total_requests'] = 0
                    
                    return keys
            except Exception as e:
                print(f"❌ Ошибка загрузки ключей: {e}")
                # Создаем пустой список при ошибке
                return []
        
        # Если файла нет, используем ключ из config.py
        from config import get_api_key
        openrouter_key = get_api_key('openrouter')
        if openrouter_key:
            return [{
                'key': openrouter_key,
                'usage': 0,
                'last_reset': datetime.now().isoformat(),
                'name': 'default_key',
                'limit_type': 'standard',
                'limit': self.daily_limit,
                'created_at': datetime.now().isoformat(),
                'is_active': True,
                'total_requests': 0,
                'key_hash': self._hash_key(openrouter_key)
            }]
        return []
    
    def _hash_key(self, key: str) -> str:
        """Хеширование ключа для безопасного хранения"""
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def _save_keys(self):
        """Сохранение ключей в файл"""
        try:
            data = {
                'keys': self.keys,
                'settings': {
                    'default_limit': self.daily_limit,
                    'extended_limit': self.extended_limit,
                    'reset_time': self.reset_time,
                    'model': self.model,
                    'last_updated': datetime.now().isoformat()
                }
            }
            
            # Создаем директорию если не существует
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ Ошибка сохранения ключей: {e}")
    
    def add_key(self, api_key: str, name: str = "new_key", limit_type: str = "standard"):
        """Добавление нового ключа"""
        api_key = api_key.strip()
        if not api_key:
            print(f"❌ Ключ пустой: {name}")
            return False
        
        # Проверяем, нет ли уже такого ключа
        key_hash = self._hash_key(api_key)
        for existing_key in self.keys:
            if existing_key.get('key_hash') == key_hash:
                print(f"⚠️ Ключ уже существует: {name}")
                return False
        
        limit = self.daily_limit if limit_type == "standard" else self.extended_limit
        
        new_key = {
            'key': api_key,
            'usage': 0,
            'last_reset': datetime.now().isoformat(),
            'name': name,
            'limit_type': limit_type,
            'limit': limit,
            'created_at': datetime.now().isoformat(),
            'is_active': True,
            'total_requests': 0,
            'key_hash': key_hash
        }
        
        self.keys.append(new_key)
        
        self._save_keys()
        print(f"✅ Добавлен ключ: {name} (лимит: {limit} запросов/день)")
        return True
    
    def get_next_key(self) -> str:
        """Получение следующего доступного ключа"""
        if not self.keys:
            raise Exception("❌ Нет доступных API ключей")
        
        # Проверяем активные ключи
        active_keys = [k for k in self.keys if k.get('is_active', True)]
        if not active_keys:
            raise Exception("❌ Нет активных API ключей")
        
        # Проверяем сброс дневного лимита
        self._check_daily_reset()
        
        # Ищем ключ с доступными запросами
        start_index = self.current_key_index
        
        for i in range(len(active_keys)):
            idx = (start_index + i) % len(active_keys)
            key_data = active_keys[idx]
            
            current_limit = key_data.get('limit', self.daily_limit)
            current_usage = key_data.get('usage', 0)
            
            if current_usage < current_limit:
                self.current_key_index = (idx + 1) % len(active_keys)
                return key_data['key']
        
        # Если все ключи исчерпаны
        raise Exception("❌ Все API ключи исчерпали дневной лимит")
    
    def record_usage(self, api_key: str):
        """Запись использования ключа"""
        for key_data in self.keys:
            if key_data['key'] == api_key:
                key_data['usage'] = key_data.get('usage', 0) + 1
                key_data['total_requests'] = key_data.get('total_requests', 0) + 1
                break
        self._save_keys()
    
    def _check_daily_reset(self, force: bool = False):
        """Проверка и сброс дневного лимита"""
        now = datetime.now()
        reset_hour, reset_minute = map(int, self.reset_time.split(':'))
        
        reset_performed = False
        
        for key_data in self.keys:
            last_reset_str = key_data.get('last_reset')
            if not last_reset_str:
                continue
                
            try:
                last_reset = datetime.fromisoformat(last_reset_str)
            except:
                # Если формат неверный, сбрасываем
                key_data['usage'] = 0
                key_data['last_reset'] = now.isoformat()
                continue
            
            # Определяем время следующего сброса
            next_reset = last_reset.replace(
                hour=reset_hour, 
                minute=reset_minute, 
                second=0, 
                microsecond=0
            )
            
            # Если время последнего сброса позже времени сброса сегодня, 
            # то следующий сброс - завтра
            if last_reset.time() >= next_reset.time():
                next_reset += timedelta(days=1)
            
            # Проверяем, наступило ли время сброса
            if force or now >= next_reset:
                old_usage = key_data.get('usage', 0)
                key_data['usage'] = 0
                key_data['last_reset'] = now.isoformat()
                reset_performed = True
                print(f"🔄 Сброс лимита для ключа {key_data['name']}: {old_usage} → 0")
        
        if force or reset_performed:
            self._save_keys()
    
    def _start_reset_checker(self):
        """Запуск фоновой проверки сброса счетчиков"""
        def check_reset_loop():
            while True:
                try:
                    # Проверяем сброс каждую минуту
                    current_time = datetime.now().strftime("%H:%M")
                    if current_time == self.reset_time:
                        self._check_daily_reset(force=True)
                        print(f"⏰ Автоматический сброс счетчиков в {self.reset_time}")
                    
                    time.sleep(self.reset_check_interval)
                except Exception as e:
                    print(f"❌ Ошибка в фоновой проверке сброса: {e}")
                    time.sleep(60)
        
        # Запускаем в отдельном потоке
        thread = threading.Thread(target=check_reset_loop, daemon=True)
        thread.start()
        print(f"🚀 Запущен фоновый проверщик сброса (проверка каждые {self.reset_check_interval} сек)")
    
    def get_usage_stats(self) -> Dict:
        """Получение статистики использования"""
        total_used = sum(key.get('usage', 0) for key in self.keys)
        total_limit = sum(key.get('limit', self.daily_limit) for key in self.keys)
        total_available = total_limit - total_used
        
        active_keys = [k for k in self.keys if k.get('is_active', True)]
        inactive_keys = [k for k in self.keys if not k.get('is_active', True)]
        
        # Время следующего сброса
        next_reset = self._get_next_reset_time()
        
        return {
            'total_keys': len(self.keys),
            'active_keys': len(active_keys),
            'inactive_keys': len(inactive_keys),
            'daily_limit_per_key': self.daily_limit,
            'extended_limit': self.extended_limit,
            'reset_time': self.reset_time,
            'next_reset': next_reset,
            'model': self.model,
            'total_used_today': total_used,
            'total_limit_today': total_limit,
            'total_available_today': total_available,
            'keys': [
                {
                    'name': key.get('name', 'Без имени'),
                    'key_preview': key['key'][:8] + '...' + key['key'][-4:] if len(key['key']) > 12 else key['key'],
                    'used': key.get('usage', 0),
                    'limit': key.get('limit', self.daily_limit),
                    'limit_type': key.get('limit_type', 'standard'),
                    'available': key.get('limit', self.daily_limit) - key.get('usage', 0),
                    'last_reset': key.get('last_reset', 'Неизвестно'),
                    'created_at': key.get('created_at', 'Неизвестно'),
                    'is_active': key.get('is_active', True),
                    'total_requests': key.get('total_requests', 0)
                }
                for key in self.keys
            ]
        }
    
    def _get_next_reset_time(self) -> str:
        """Получение времени следующего сброса"""
        now = datetime.now()
        reset_hour, reset_minute = map(int, self.reset_time.split(':'))
        
        next_reset = now.replace(hour=reset_hour, minute=reset_minute, second=0, microsecond=0)
        
        if now.time() >= next_reset.time():
            next_reset += timedelta(days=1)
        
        return next_reset.strftime("%Y-%m-%d %H:%M:%S")
    
    def set_key_limit(self, key_name: str, limit_type: str):
        """Установка типа лимита для ключа"""
        for key in self.keys:
            if key['name'] == key_name:
                old_limit = key.get('limit', self.daily_limit)
                key['limit_type'] = limit_type
                key['limit'] = self.extended_limit if limit_type == 'extended' else self.daily_limit
                self._save_keys()
                print(f"✅ Лимит ключа {key_name} изменен: {old_limit} → {key['limit']}")
                return True
        
        print(f"❌ Ключ {key_name} не найден")
        return False
    
    def set_reset_time(self, reset_time: str):
        """Установка времени сброса"""
        try:
            # Проверяем формат времени
            datetime.strptime(reset_time, "%H:%M")
            self.reset_time = reset_time
            self._save_keys()
            print(f"✅ Время сброса установлено: {reset_time}")
            return True
        except ValueError:
            print(f"❌ Неверный формат времени: {reset_time}. Используйте HH:MM")
            return False
    
    def delete_key(self, key_name: str):
        """Удаление ключа"""
        original_count = len(self.keys)
        self.keys = [k for k in self.keys if k['name'] != key_name]
        
        if len(self.keys) < original_count:
            self._save_keys()
            print(f"✅ Ключ {key_name} удален")
            return True
        else:
            print(f"❌ Ключ {key_name} не найден")
            return False
    
    def toggle_key_active(self, key_name: str, is_active: bool):
        """Включение/выключение ключа"""
        for key in self.keys:
            if key['name'] == key_name:
                old_status = key.get('is_active', True)
                key['is_active'] = is_active
                self._save_keys()
                status = "активирован" if is_active else "деактивирован"
                print(f"✅ Ключ {key_name} {status}")
                return True
        
        print(f"❌ Ключ {key_name} не найден")
        return False
    
    def import_keys_from_file(self, file_content: str):
        """Импорт ключей из текстового файла"""
        imported_count = 0
        
        lines = file_content.strip().split('\n')
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Формат: ключ, название, тип_лимита (опционально)
            parts = [p.strip() for p in line.split(',')]
            
            if len(parts) >= 1:
                api_key = parts[0]
                name = parts[1] if len(parts) > 1 else f"imported_key_{imported_count+1}"
                limit_type = parts[2] if len(parts) > 2 else "standard"
                
                if self.add_key(api_key, name, limit_type):
                    imported_count += 1
        
        print(f"✅ Импортировано ключей: {imported_count}")
        return imported_count
    
    def validate_model(self, model: str) -> bool:
        """Упрощенная валидация модели - разрешаем любые строки"""
        if not model or not isinstance(model, str):
            print(f"❌ Ошибка: Модель должна быть строкой")
            return False
        
        # Минимальная проверка: строка не пустая
        if len(model.strip()) < 3:
            print(f"❌ Ошибка: Название модели слишком короткое")
            return False
        
        # Проверяем наличие двоеточия (обычно формат model:free)
        if ':' not in model:
            print(f"⚠️ Предупреждение: Модель '{model}' не содержит двоеточия (обычно формат: model:free)")
            # Всё равно разрешаем - может быть кастомная модель
        
        print(f"✅ Модель принята: {model}")
        return True
    
    def set_model(self, model: str):
        """Установка модели по умолчанию - принимаем любую строку"""
        if not model or not isinstance(model, str):
            print(f"❌ Ошибка: Неверное название модели")
            return False
        
        # Просто устанавливаем модель
        self.model = model.strip()
        
        # Обновляем конфигурацию
        from config import load_config, save_config
        config = load_config()
        
        if 'openrouter' not in config:
            config['openrouter'] = {}
        
        config['openrouter']['model'] = self.model
        
        try:
            save_config(config)
            self._save_keys()
            print(f"✅ Модель установлена: {self.model}")
            return True
        except Exception as e:
            print(f"❌ Ошибка установки модели: {e}")
            return False
    
    def force_reset_all(self):
        """Принудительный сброс всех счетчиков"""
        for key in self.keys:
            old_usage = key.get('usage', 0)
            key['usage'] = 0
            key['last_reset'] = datetime.now().isoformat()
            print(f"🔄 Сброс ключа {key['name']}: {old_usage} → 0")
        
        self._save_keys()
        print(f"✅ Принудительный сброс всех ключей выполнен")
        return True
    
    def import_keys_from_text(self, keys_text: str):
        """Импорт ключей из текста (альтернативный метод)"""
        imported_count = 0
        
        lines = keys_text.strip().split('\n')
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Пробуем разные форматы
            parts = []
            
            # Формат 1: ключ, название, лимит
            if ',' in line:
                parts = [p.strip() for p in line.split(',')]
                api_key = parts[0] if len(parts) > 0 else ''
                name = parts[1] if len(parts) > 1 else f"key_{imported_count+1}"
                limit_type = parts[2] if len(parts) > 2 else "standard"
            
            # Формат 2: только ключ (определяем по префиксу или длине)
            elif 'sk-or-' in line or len(line) > 30:
                api_key = line
                name = f"imported_key_{imported_count+1}"
                limit_type = "standard"
            
            else:
                continue
            
            # Проверяем базовую валидность ключа
            if not api_key or len(api_key.strip()) < 20:
                print(f"⚠️ Пропускаем невалидный ключ в строке {line_num}")
                continue
            
            if self.add_key(api_key, name, limit_type):
                imported_count += 1
        
        print(f"✅ Импортировано ключей из текста: {imported_count}")
        return imported_count
    
    def get_known_models(self) -> List[str]:
        """Получение списка известных моделей (для справки)"""
        return [
            "meta-llama/llama-3.3-70b-instruct:free",
            "meta-llama/llama-3.2-3b-instruct:free",
            "meta-llama/llama-3.2-1b-instruct:free",
            "meta-llama/llama-3.1-8b-instruct:free",
            "qwen/qwen2.5-7b-instruct:free",
            "qwen/qwen2.5-14b-instruct:free",
            "google/gemma-2-2b-it:free",
            "google/gemma-2-9b-it:free",
            "microsoft/phi-3.5-mini-instruct:free",
            "deepseek/deepseek-chat-v3-0324:free"
        ]
    
    def test_model_connection(self, model: str, api_key: str = None) -> Dict:
        """Тестирование подключения к конкретной модели"""
        try:
            if not api_key:
                api_key = self.get_next_key()
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            test_data = {
                "model": model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=test_data,
                timeout=10
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "model": model,
                    "status": "available",
                    "message": "Модель доступна и работает"
                }
            elif response.status_code == 400:
                return {
                    "success": True,
                    "model": model,
                    "status": "invalid_model",
                    "message": "Модель не найдена или недоступна"
                }
            elif response.status_code == 401:
                return {
                    "success": True,
                    "model": model,
                    "status": "invalid_key",
                    "message": "Ключ API недействителен"
                }
            else:
                return {
                    "success": True,
                    "model": model,
                    "status": "error",
                    "message": f"Ошибка: {response.status_code}"
                }
                
        except requests.exceptions.Timeout:
            return {
                "success": True,
                "model": model,
                "status": "timeout",
                "message": "Таймаут подключения"
            }
        except Exception as e:
            return {
                "success": True,
                "model": model,
                "status": "connection_error",
                "message": f"Ошибка соединения: {str(e)}"
            }

# Глобальный экземпляр
key_manager = APIKeyManager()

def get_key_manager() -> APIKeyManager:
    return key_manager