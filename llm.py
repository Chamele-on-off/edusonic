import requests
import json
from typing import Optional, Dict
from pathlib import Path
import time
from config import get_api_key, load_config  # Изменен импорт

class LLMIntegration:
    def __init__(self, api_key: str = None, 
                 api_url: str = "https://openrouter.ai/api/v1/chat/completions",
                 cache_dir: str = "cache",
                 model: str = "deepseek/deepseek-chat-v3-0324:free"):
        # Загружаем конфигурацию для получения API ключа
        config = load_config()
        self.api_key = api_key or config.get("llm", {}).get("api_key", "")
        self.api_url = api_url or config.get("llm", {}).get("api_url", "https://openrouter.ai/api/v1/chat/completions")
        self.model = model or config.get("llm", {}).get("model", "deepseek/deepseek-chat-v3-0324:free")
        self.cache_dir = Path(cache_dir)
        self.cache = self._load_cache()
        
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
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Формируем промпт для учителя
        system_prompt = f"""Ты - профессиональный учитель и эксперт в предмете "{subject}". 
Твоя задача - давать четкие, понятные и структурированные ответы на вопросы учеников.
Отвечай кратко, но информативно, используя примеры если это уместно.
Объясняй сложные понятия простым языком.
Будь дружелюбным и поддерживающим учителем."""

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Вопрос ученика: {prompt}\n\nКонтекст урока: {context}"}
            ],
            "temperature": 0.7,
            "max_tokens": 800,
            "stream": False
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content']
                return self._process_content(answer.strip())
            else:
                print(f"Ошибка API LLM: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Ошибка запроса к LLM API: {e}")
            return None

    def query(self, question: str, context: str = "", subject: str = "") -> Optional[str]:
        """Запрос к LLM API"""
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
            "deepseek": "deepseek/deepseek-chat-v3-0324:free",
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
        if "qwen" in model.lower():
            self.api_key = config.get("qwen", {}).get("api_key", "")
        else:
            self.api_key = config.get("llm", {}).get("api_key", "")