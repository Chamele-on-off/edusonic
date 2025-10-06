import requests
import json
from typing import Optional, Dict
import time
import os
from config import get_local_llm_settings

class LocalLLM:
    def __init__(self, base_url: str = None):
        config = get_local_llm_settings()
        
        # ИСПРАВЛЕНИЕ: Правильный URL для локального подключения
        self.base_url = (base_url or 
                        os.getenv('OLLAMA_HOST') or 
                        config.get("base_url", "http://localhost:11434"))
            
        # ИСПРАВЛЕНИЕ: Используем модель которая точно есть
        self.model = config.get("model", "llama3.2:3b")  # Изменено на llama3.2:3b
        self.timeout = config.get("timeout", 60)
        self.max_retries = config.get("max_retries", 2)
        self.retry_delay = config.get("retry_delay", 2.0)
        self.enabled = config.get("enabled", True)
        
        print(f"🔧 LocalLLM инициализирован с URL: {self.base_url}, модель: {self.model}")
        
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
                    "stream": False
                }
                
                print(f"🔧 Запрос к локальной модели {self.model} (попытка {attempt + 1})...")
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=data,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if 'message' in result and 'content' in result['message']:
                        content = result['message']['content']
                        print(f"✅ Локальная модель ответила: {content[:100]}...")
                        return content
                    elif 'response' in result:
                        content = result['response']
                        print(f"✅ Локальная модель ответила: {content[:100]}...")
                        return content
                    else:
                        print(f"❌ Неожиданный формат ответа: {result}")
                
                print(f"❌ Локальная модель вернула статус {response.status_code}")
                if attempt < self.max_retries - 1:
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
        
        return None

    def get_status(self) -> Dict:
        """Получение статуса локальной модели"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                
                # Проверяем загружена ли наша модель
                model_loaded = False
                available_models = []
                
                for model in models:
                    model_name = model.get('name', '')
                    available_models.append(model_name)
                    if self.model in model_name:
                        model_loaded = True
                
                return {
                    "available": True,
                    "model_loaded": model_loaded,
                    "current_model": self.model,
                    "available_models": available_models,
                    "base_url": self.base_url
                }
        except Exception as e:
            print(f"❌ Ошибка получения статуса локальной модели: {e}")
            
        return {
            "available": False,
            "model_loaded": False,
            "current_model": self.model,
            "available_models": [],
            "base_url": self.base_url
        }

    def pull_model(self, model_name: str = None) -> bool:
        """Загружает модель если она не загружена"""
        model_to_pull = model_name or self.model
        
        try:
            print(f"🔧 Попытка загрузить модель: {model_to_pull}")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"model": model_to_pull},
                timeout=300  # 5 минут для загрузки
            )
            
            if response.status_code == 200:
                print(f"✅ Модель {model_to_pull} успешно загружена")
                self.model = model_to_pull
                return True
            else:
                print(f"❌ Не удалось загрузить модель {model_to_pull}")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False

# Глобальный экземпляр для использования
local_llm = LocalLLM()

def get_local_llm() -> LocalLLM:
    """Возвращает глобальный экземпляр LocalLLM"""
    return local_llm
