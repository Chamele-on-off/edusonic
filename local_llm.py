import requests
import json
from typing import Optional, Dict
import time
import os
from config import get_local_llm_settings

class LocalLLM:
    def __init__(self, base_url: str = None):
        config = get_local_llm_settings()
        
        self.base_url = (base_url or 
                        os.getenv('OLLAMA_HOST') or 
                        config.get("base_url", "http://localhost:11434"))
            
        self.model = config.get("model", "qwen2.5:3b")
        # УВЕЛИЧИВАЕМ ТАЙМАУТ ДО 3 МИНУТ ДЛЯ ГЕНЕРАЦИИ ВОПРОСОВ
        self.timeout = 180  # 3 минуты для генерации вопросов
        self.max_retries = 2
        self.retry_delay = 3.0
        self.enabled = config.get("enabled", True)
        
        print(f"🔧 LocalLLM инициализирован с URL: {self.base_url}")
        print(f"⏱️ Таймаут установлен: {self.timeout} секунд")
        
    def is_available(self) -> bool:
        """Проверяет доступность локальной модели"""
        if not self.enabled:
            print("🔧 LocalLLM отключен в конфигурации")
            return False
            
        try:
            print(f"🔧 Проверка доступности Ollama по {self.base_url}...")
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                print("✅ Локальная модель доступна")
                return True
            else:
                print(f"❌ Локальная модель недоступна (статус: {response.status_code})")
                return False
        except requests.exceptions.ConnectTimeout:
            print(f"❌ Таймаут подключения к Ollama по {self.base_url}")
            return False
        except requests.exceptions.ConnectionError:
            print(f"❌ Ошибка подключения к Ollama по {self.base_url}")
            return False
        except Exception as e:
            print(f"❌ Локальная модель недоступна: {e}")
            return False
    
    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 1000) -> Optional[str]:
        """Генерация ответа через локальную модель"""
        if not self.enabled:
            print("🔧 LocalLLM отключен, пропускаем генерацию")
            return None
            
        for attempt in range(self.max_retries):
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                data = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "stream": False  # ← ОТКЛЮЧАЕМ ПОТОКОВУЮ ПЕРЕДАЧУ
                }
                
                print(f"🔧 Запрос к локальной модели {self.model} (попытка {attempt + 1})...")
                start_time = time.time()
                
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=data,
                    timeout=self.timeout  # Используем увеличенный таймаут
                )
                
                request_time = time.time() - start_time
                print(f"⏱️ Время запроса к локальной модели: {request_time:.2f}с")
                
                if response.status_code == 200:
                    result = response.json()
                    if 'message' in result and 'content' in result['message']:
                        content = result['message']['content']
                        print(f"✅ Локальная модель ответила ({len(content)} символов): {content[:100]}...")
                        return content
                    elif 'response' in result:
                        content = result['response']
                        print(f"✅ Локальная модель ответила ({len(content)} символов): {content[:100]}...")
                        return content
                    else:
                        print(f"❌ Неожиданный формат ответа: {result}")
                        # Пробуем извлечь ответ любым способом
                        if 'choices' in result and len(result['choices']) > 0:
                            if 'message' in result['choices'][0] and 'content' in result['choices'][0]['message']:
                                content = result['choices'][0]['message']['content']
                                print(f"✅ Извлечен ответ из choices: {content[:100]}...")
                                return content
                
                print(f"❌ Локальная модель вернула статус {response.status_code}")
                if attempt < self.max_retries - 1:
                    print(f"🔄 Повторная попытка через {self.retry_delay} сек...")
                    time.sleep(self.retry_delay)
                
            except requests.exceptions.Timeout:
                print(f"⏰ Таймаут локальной модели (попытка {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                    
            except Exception as e:
                print(f"❌ Ошибка локальной модели (попытка {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
        
        print("❌ Все попытки обращения к локальной модели завершились ошибкой")
        return None

    def get_status(self) -> Dict:
        """Получение статуса локальной модели"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_loaded = any(self.model in model.get('name', '') for model in models)
                
                return {
                    "available": True,
                    "model_loaded": model_loaded,
                    "models": [model.get('name') for model in models],
                    "base_url": self.base_url,
                    "current_model": self.model
                }
        except Exception as e:
            print(f"❌ Ошибка получения статуса локальной модели: {e}")
            
        return {
            "available": False,
            "model_loaded": False,
            "models": [],
            "base_url": self.base_url,
            "current_model": self.model
        }

# Глобальный экземпляр для использования
local_llm = LocalLLM()

def get_local_llm() -> LocalLLM:
    """Возвращает глобальный экземпляр LocalLLM"""
    return local_llm
