import requests
import json
from typing import Optional, Dict
from pathlib import Path
import time
from config import get_api_key, load_config, get_model_config

class LLMIntegration:
    def __init__(self, api_key: str = None, 
                 api_url: str = "https://openrouter.ai/api/v1/chat/completions",
                 cache_dir: str = "cache",
                 model: str = "meta-llama/llama-3.3-8b-instruct:free"):
        # Загружаем конфигурацию для получения API ключа
        config = load_config()
        openrouter_config = get_model_config("openrouter")
        
        self.api_key = api_key or openrouter_config.get("api_key", "")
        self.api_url = api_url or openrouter_config.get("api_url", "https://openrouter.ai/api/v1/chat/completions")
        self.model = model or openrouter_config.get("model", "meta-llama/llama-3.3-8b-instruct:free")
        self.cache_dir = Path(cache_dir)
        self.cache = self._load_cache()
        self.last_request_time = 0
        self.request_delay = 1.0  # Задержка между запросами для избежания 429
        self.max_retries = 3
        self.retry_delay = 2.0
        
    def _load_cache(self) -> Dict:
        """Загрузка кэша из файла"""
        cache_file = self.cache_dir / "llm_cache.json"
        try:
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Ошибка загрузки кэша: {e}")
        return {}

    def _save_cache(self):
        """Сохранение кэша в файл"""
        try:
            if not self.cache_dir.exists():
                self.cache_dir.mkdir(parents=True)
                
            cache_file = self.cache_dir / "llm_cache.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения кэша: {e}")

    def _process_content(self, content: str) -> str:
        """Обработка контента (удаление тегов thinking)"""
        return content.replace('<think>', '').replace('</think>', '')

    def _query_llm_api(self, prompt: str, context: str = "", subject: str = "") -> Optional[str]:
        """Запрос к LLM API через OpenRouter"""
        if not self.api_key:
            print("API ключ не установлен для LLM")
            return None
            
        # Добавляем задержку между запросами для избежания 429
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.request_delay:
            time.sleep(self.request_delay - time_since_last_request)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-site.com",  # Required by OpenRouter
            "X-Title": "AI Teacher"  # Required by OpenRouter
        }
        
        # Формируем промпт для учителя
        system_prompt = f"""Ты - профессиональный учитель и эксперт в предмете "{subject}". 
Твоя задача - давать четкие, понятные и структурированные ответы на вопросы учеников.
Отвечай кратко, но информативно, используя примеры если это уместно.
Объясняй сложные понятия простым языком.
Будь дружелюбным и поддерживающим учителем.
Отвечай на русском языке."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Контекст урока: {context}\n\nВопрос ученика: {prompt}"}
        ]

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 800,
            "stream": False
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                self.last_request_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result['choices'][0]['message']['content']
                    return self._process_content(answer.strip())
                elif response.status_code == 429:
                    print(f"Ошибка 429 (Rate Limit). Попытка {attempt + 1}/{self.max_retries}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))
                        continue
                    else:
                        print(f"Превышено количество попыток. Ошибка API: {response.status_code} - {response.text}")
                        return None
                else:
                    print(f"Ошибка API LLM: {response.status_code} - {response.text}")
                    return None
                    
            except requests.exceptions.Timeout:
                print(f"Таймаут запроса к LLM API. Попытка {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    print("Превышено количество попыток из-за таймаутов")
                    return None
            except Exception as e:
                print(f"Ошибка запроса к LLM API (попытка {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    return None
        
        return None

    def query(self, question: str, context: str = "", subject: str = "") -> Optional[str]:
        """Запрос к LLM API с гарантированным ответом"""
        if not question.strip():
            return None
            
        question_lower = question.lower().strip()
        
        # Проверка кэша
        cache_key = f"{subject}_{question_lower}" if subject else question_lower
        if cache_key in self.cache:
            print(f"Использую кэшированный ответ для: {question_lower}")
            return self.cache[cache_key]
        
        print(f"Запрос к LLM: {question} (предмет: {subject})")
        
        # Запрос к реальному LLM
        llm_response = self._query_llm_api(question, context, subject)
        
        if llm_response:
            print(f"Получен ответ от LLM: {llm_response[:100]}...")
            self.cache[cache_key] = llm_response
            self._save_cache()
            return llm_response
        
        # Fallback на локальные ответы если LLM недоступен
        print("LLM недоступен, использую fallback ответ")
        fallback_responses = [
            f"Интересный вопрос по {subject}! Давайте разберем его подробнее на следующем занятии.",
            f"По теме {subject} это важный аспект. Я подготовлю подробное объяснение к нашему следующему уроку.",
            f"Хороший вопрос! В контексте {subject} это требует детального изучения, которое мы проведем позже.",
            f"Записал ваш вопрос по {subject}. Вернемся к нему в подходящий момент урока.",
            f"В рамках {subject} этот вопрос очень важен. Давайте обсудим его дополнительно после текущего материала."
        ]
        
        answer = fallback_responses[hash(question_lower) % len(fallback_responses)]
        self.cache[cache_key] = answer
        self._save_cache()
        
        return answer

    def add_to_cache(self, question: str, answer: str, subject: str = ""):
        """Добавление ответа в кэш"""
        cache_key = f"{subject}_{question.lower()}" if subject else question.lower()
        self.cache[cache_key] = answer
        self._save_cache()

    def clear_cache(self):
        """Очистка кэша"""
        self.cache = {}
        self._save_cache()

    def get_cache_stats(self) -> Dict:
        """Получение статистики кэша"""
        return {
            "total_entries": len(self.cache),
            "subjects": list(set(key.split('_')[0] for key in self.cache.keys() if '_' in key))
        }

    def set_model(self, model: str):
        """Установка модели LLM"""
        available_models = {
            "llama": "meta-llama/llama-3.3-8b-instruct:free",
            "llama3": "meta-llama/llama-3.3-8b-instruct:free",
            "qwen": "qwen/qwen3-235b-a22b:free",
            "qwen-turbo": "qwen/qwen3-235b-a22b:free"
        }
        
        if model in available_models:
            self.model = available_models[model]
            print(f"Установлена модель: {self.model}")
        else:
            self.model = model
            print(f"Установлена кастомная модель: {self.model}")
            
        # Переключаем API ключ в зависимости от модели
        config = load_config()
        self.api_key = config.get("openrouter", {}).get("api_key", "")
