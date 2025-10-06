# local_llm.py - УЛЬТРА-ЛЕГКОВЕСНАЯ ВЕРСИЯ
import requests
import json
from typing import Optional, Dict
import time
import os
from config import get_local_llm_settings

class LocalLLM:
    def __init__(self, base_url: str = None):
        config = get_local_llm_settings()
        
        # УЛЬТРА-ЛЕГКОВЕСНАЯ КОНФИГУРАЦИЯ
        self.base_url = (base_url or 
                        os.getenv('LOCAL_LLM_HOST') or 
                        config.get("base_url", "http://ollama:11434"))
            
        # САМАЯ ЛЕГКАЯ МОДЕЛЬ ДЛЯ 8GB RAM
        self.model = config.get("model", "qwen2.5:1.5b")  # Всего 0.9GB памяти!
        self.timeout = 10  # Уменьшен таймаут для скорости
        self.max_retries = 1  # Только одна попытка для скорости
        self.retry_delay = 1.0
        self.enabled = config.get("enabled", True)
        
        print(f"⚡ LocalLLM: {self.model} на {self.base_url} (таймаут: {self.timeout}с)")
        
    def is_available(self) -> bool:
        """СУПЕР-БЫСТРАЯ проверка доступности"""
        if not self.enabled:
            return False
            
        try:
            # УЛЬТРА-БЫСТРАЯ проверка (2 секунды максимум)
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 500) -> Optional[str]:
        """УЛЬТРА-БЫСТРАЯ генерация с приоритетом скорости"""
        if not self.enabled:
            return None
            
        try:
            # ОПТИМИЗИРОВАННЫЙ ПРОМПТ ДЛЯ СКОРОСТИ
            optimized_prompt = self._optimize_prompt(prompt, system_prompt)
            
            # МИНИМАЛЬНЫЙ JSON ДЛЯ СКОРОСТИ
            data = {
                "model": self.model,
                "prompt": optimized_prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.3,  # Низкая температура для консистентности
                    "top_k": 20,         # Ограничение для скорости
                    "top_p": 0.7,        # Ограничение для скорости
                    "repeat_penalty": 1.1
                }
            }
            
            print(f"⚡ Запрос к {self.model}: {prompt[:50]}...")
            start_time = time.time()
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '').strip()
                
                end_time = time.time()
                response_time = end_time - start_time
                
                print(f"✅ Ответ за {response_time:.2f}с: {response_text[:80]}...")
                return response_text
            else:
                print(f"❌ Ошибка {response.status_code} за {(time.time()-start_time):.2f}с")
                return None
                
        except requests.exceptions.Timeout:
            print(f"⏰ Таймаут локальной модели (> {self.timeout}с)")
            return None
        except Exception as e:
            print(f"❌ Ошибка локальной модели: {e}")
            return None

    def _optimize_prompt(self, prompt: str, system_prompt: str = "") -> str:
        """Оптимизация промпта для скорости"""
        if system_prompt:
            # КОРОТКИЙ СИСТЕМНЫЙ ПРОМПТ
            short_system = "Ты - учитель. Отвечай кратко и понятно. "
            return f"{short_system}\n\nВопрос: {prompt}\n\nОтвет:"
        else:
            return f"Вопрос: {prompt}\n\nОтвет:"

    def get_status(self) -> Dict:
        """Быстрая проверка статуса"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_loaded = any(self.model in model.get('name', '') for model in models)
                
                # БЫСТРАЯ ПРОВЕРКА РАБОТОСПОСОБНОСТИ
                test_response = self.generate("Привет", max_tokens=10)
                working = test_response is not None
                
                return {
                    "available": True,
                    "model_loaded": model_loaded,
                    "working": working,
                    "current_model": self.model,
                    "base_url": self.base_url,
                    "models": [model.get('name') for model in models[:3]]  # Только первые 3
                }
        except Exception as e:
            print(f"❌ Ошибка статуса: {e}")
            
        return {
            "available": False,
            "model_loaded": False,
            "working": False,
            "current_model": self.model,
            "base_url": self.base_url,
            "models": []
        }

    def set_model(self, model: str):
        """Смена модели"""
        self.model = model
        print(f"🔧 Установлена модель: {model}")

# Глобальный экземпляр
local_llm = LocalLLM()

def get_local_llm() -> LocalLLM:
    return local_llm