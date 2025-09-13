import requests
import json
from typing import Optional, Dict
from pathlib import Path
import time
from config import get_api_key, load_config, get_model_config
import re

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
        self.request_delay = 1.0
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

    def _clean_llm_response(self, content: str) -> str:
        """Очистка ответа LLM от форматирования и специальных символов"""
        if not content:
            return ""
            
        # Удаляем звездочки и другие маркеры форматирования
        content = re.sub(r'[\*\#\-\_]{2,}', '', content)
        content = re.sub(r'^\s*[\*\-\+]\s*', '', content, flags=re.MULTILINE)
        
        # Удаляем префиксы типа "Ответ:" или "AI:"
        prefixes = ["Ответ:", "AI:", "Ассистент:", "Assistant:", "**", "*"]
        for prefix in prefixes:
            if content.startswith(prefix):
                content = content[len(prefix):].strip()
        
        # Удаляем лишние пробелы и переносы строк
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n+', '\n', content)
        
        return content.strip()

    def _query_llm_api(self, prompt: str, context: str = "", subject: str = "", 
                       system_prompt: str = "", max_tokens: int = 1000) -> Optional[str]:
        """Запрос к LLM API через OpenRouter с улучшенной обработкой ошибок"""
        
        # Если нет API ключа, используем fallback ответ
        if not self.api_key:
            print("⚠️ API ключ не установлен, использую fallback ответ")
            return self._get_fallback_response(prompt, subject)
        
        # Добавляем задержку между запросами
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.request_delay:
            time.sleep(self.request_delay - time_since_last_request)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ai-teacher.com",
            "X-Title": "AI Teacher"
        }
        
        # Улучшенный промпт для учителя
        final_system_prompt = system_prompt or f"""Ты - профессиональный учитель и эксперт по предмету "{subject}". 
Твоя задача - давать четкие, понятные и информативные ответы на вопросы учеников.

Важные правила:
1. Отвечай максимально подробно и информативно
2. Объясняй сложные понятия простым языком
3. Приводи примеры если это уместно
4. Будь дружелюбным и поддерживающим
5. Отвечай на русском языке
6. Не используй форматирование markdown
7. Не говори общие фразы типа "расскажу подробнее" - сразу давай конкретный ответ
8. Если вопрос короткий, дай развернутый ответ
9. Структурируй ответ если это необходимо

Контекст: {context}"""

        messages = []
        
        # Добавляем системный промпт
        if final_system_prompt:
            messages.append({"role": "system", "content": final_system_prompt})
        
        # Добавляем контекст если есть
        if context and context.strip():
            messages.append({"role": "system", "content": f"Контекст разговора: {context}"})
        
        # Добавляем пользовательский запрос
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": max_tokens,
            "stream": False
        }

        print(f"🔧 Данные для запроса LLM:")
        print(f"   Модель: {self.model}")
        print(f"   Промпт: {prompt[:100]}...")
        if context:
            print(f"   Контекст: {context[:100]}...")
        print(f"   Предмет: {subject}")
        
        for attempt in range(self.max_retries):
            try:
                print(f"🔄 Попытка {attempt + 1}: Отправка запроса к {self.model}")
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                self.last_request_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if 'choices' in result and len(result['choices']) > 0:
                        answer = result['choices'][0]['message']['content']
                        processed_answer = self._clean_llm_response(answer)
                        print(f"✅ Получен ответ от LLM: {processed_answer[:100]}...")
                        return processed_answer
                    else:
                        print("❌ Неверный формат ответа от API")
                        return self._get_fallback_response(prompt, subject)
                        
                elif response.status_code == 429:
                    wait_time = self.retry_delay * (attempt + 1)
                    print(f"⏳ Ошибка 429 (Rate Limit). Ждем {wait_time} сек...")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    print(f"❌ Ошибка API: {response.status_code} - {response.text[:200]}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return self._get_fallback_response(prompt, subject)
                    
            except requests.exceptions.Timeout:
                print(f"⏰ Таймаут запроса. Попытка {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                return self._get_fallback_response(prompt, subject)
                
            except Exception as e:
                print(f"❌ Ошибка при запросе (попытка {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                return self._get_fallback_response(prompt, subject)
        
        return self._get_fallback_response(prompt, subject)

    def _get_fallback_response(self, prompt: str, subject: str = "") -> str:
        """Возвращает fallback ответ когда LLM недоступен"""
        prompt_lower = prompt.lower()
        
        # Определяем тип вопроса для более релевантного ответа
        if any(word in prompt_lower for word in ['что', 'как', 'почему', 'зачем']):
            return f"Интересный вопрос! По теме {subject if subject else 'этого'} есть много интересной информации. Давайте обсудим это подробнее!"
        
        if any(word in prompt_lower for word in ['объясни', 'расскажи', 'покажи']):
            return f"С удовольствием объясню! Это важный аспект {subject if subject else 'темы'}. Давайте разберем вместе."
        
        # Общий fallback ответ
        subjects = ["математика", "история", "обществознание", "физика", "химия"]
        subject_list = ", ".join(subjects[:3]) + " и другие"
        
        return f"Извините, возникли временные технические трудности. Но я готов помочь вам с учебой! У меня есть уроки по: {subject_list}. Что вас интересует?"

    def query(self, question: str, context: str = "", subject: str = "") -> Optional[str]:
        """Запрос к LLM API с гарантированным ответом"""
        if not question.strip():
            return None
            
        question_lower = question.lower().strip()
        
        # Проверка кэша
        cache_key = f"{subject}_{question_lower}" if subject else question_lower
        if cache_key in self.cache:
            print(f"💾 Использую кэшированный ответ для: {question_lower}")
            return self.cache[cache_key]
        
        print(f"📨 Запрос к LLM: '{question}' (предмет: {subject})")
        
        # Запрос к реальному LLM
        llm_response = self._query_llm_api(question, context, subject)
        
        if llm_response and llm_response.strip():
            print(f"✅ Ответ получен: {llm_response[:100]}...")
            self.cache[cache_key] = llm_response
            self._save_cache()
            return llm_response
        
        # Fallback на локальные ответы если LLM недоступен
        print("⚠️ LLM недоступен, использую fallback ответ")
        fallback_response = self._get_fallback_response(question, subject)
        self.cache[cache_key] = fallback_response
        self._save_cache()
        
        return fallback_response

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
            "deepseek": "deepseek/deepseek-chat-v3-0324:free"
        }
        
        if model in available_models:
            self.model = available_models[model]
            print(f"🔧 Установлена модель: {self.model}")
        else:
            self.model = model
            print(f"🔧 Установлена кастомная модель: {self.model}")
            
        # Всегда используем OpenRouter API ключ
        config = load_config()
        self.api_key = config.get("openrouter", {}).get("api_key", "")