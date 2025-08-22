import requests
import json
from typing import Optional, Dict
from pathlib import Path
import time

class LLMIntegration:
    def __init__(self, api_key: str = None, 
                 api_url: str = "http://localhost:5001/api/ask",
                 cache_dir: str = "cache"):
        self.api_key = api_key
        self.api_url = api_url
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

    def query(self, question: str, context: str = "", subject: str = "") -> Optional[str]:
        """Запрос к LLM API"""
        if not question.strip():
            return None
            
        question_lower = question.lower().strip()
        
        # Проверка кэша
        cache_key = f"{subject}_{question_lower}" if subject else question_lower
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Здесь будет реальный вызов API
            # response = requests.post(
            #     self.api_url,
            #     json={
            #         'question': question,
            #         'context': context,
            #         'subject': subject,
            #         'api_key': self.api_key
            #     },
            #     timeout=10
            # )
            # 
            # if response.status_code == 200:
            #     result = response.json()
            #     answer = result.get('answer')
            #     if answer:
            #         self.cache[cache_key] = answer
            #         self._save_cache()
            #         return answer
            
            # Заглушка для демонстрации
            time.sleep(0.5)  # Имитация задержки API
            
            answers = [
                f"Интересный вопрос по {subject}! Давайте разберем его подробнее.",
                f"По теме {subject} это важный аспект. Объясню на следующем занятии.",
                f"Хороший вопрос! В контексте {subject} это требует детального изучения.",
                f"Записал ваш вопрос по {subject}. Вернемся к нему в подходящий момент.",
                f"В рамках {subject} этот вопрос очень важен. Обсудим его дополнительно."
            ]
            
            answer = answers[hash(question_lower) % len(answers)]
            self.cache[cache_key] = answer
            self._save_cache()
            
            return answer
            
        except Exception as e:
            print(f"Ошибка запроса к LLM: {e}")
            return "В настоящее время я не могу обработать ваш вопрос. Попробуйте позже."

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
