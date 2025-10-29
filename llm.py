import requests
import json
from typing import Optional, Dict, Callable, List
from pathlib import Path
import time
from config import get_api_key, load_config, get_model_config, get_llm_priority, set_llm_priority
import re
from local_llm_manager import get_llm_manager
import queue
from key_manager import get_key_manager

def clean_text_for_speech(text: str) -> str:
    """Тщательная очистка текста для озвучивания"""
    if not text:
        return ""
    
    # Удаляем markdown разметку
    text = re.sub(r'[#\*\_\~`]', '', text)
    
    # Удаляем лишние пробелы и переносы
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Удаляем технические символы и специальные последовательности
    text = re.sub(r'\\n', ' ', text)
    text = re.sub(r'\\t', ' ', text)
    text = re.sub(r'\\r', ' ', text)
    
    # Удаляем английские и китайские символы (оставляем только кириллицу, латиницу, цифры и пунктуацию)
    text = re.sub(r'[^\u0400-\u04FFa-zA-Z0-9\s\.,!?;:()\-—]', '', text)
    
    # Удаляем множественные точки и запятые
    text = re.sub(r'[\.\,]{2,}', '.', text)
    
    # Восстанавливаем нормальную пунктуацию
    text = re.sub(r'\s+([\.,!?;:)])', r'\1', text)
    text = re.sub(r'([(\-])\s+', r'\1', text)
    
    # Удаляем пробелы в начале и конце
    text = text.strip()
    
    # Обеспечиваем, что предложения начинаются с заглавной буквы
    if text and len(text) > 1:
        text = text[0].upper() + text[1:]
    
    return text

class LLMIntegration:
    def __init__(self, api_key: str = None, 
                 api_url: str = "https://openrouter.ai/api/v1/chat/completions",
                 cache_dir: str = "cache",
                 model: str = "meta-llama/llama-3.3-8b-instruct:free"):
        
        config = load_config()
        openrouter_config = get_model_config("openrouter")
        
        # Используем менеджер ключей вместо одного ключа
        self.key_manager = get_key_manager()
        self.api_url = api_url or openrouter_config.get("api_url", "https://openrouter.ai/api/v1/chat/completions")
        self.model = model or openrouter_config.get("model", "meta-llama/llama-3.3-8b-instruct:free")
        self.cache_dir = Path(cache_dir)
        self.cache = self._load_cache()
        self.last_request_time = 0
        self.request_delay = 1.0
        self.max_retries = 3
        
        # УВЕЛИЧЕННЫЕ ТАЙМАУТЫ
        self.timeout = 120  # 2 минуты для локальной модели
        self.request_timeout = 60  # 1 минута для OpenRouter
        self.retry_delay = 5.0  # Увеличиваем задержку между попытками
        
        # Менеджер локальной LLM
        self.llm_manager = get_llm_manager()
        self.pending_requests = {}
        
        # Настройки приоритетов из конфигурации
        self.priority_mode = get_llm_priority()
        
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
        """Сохранение кэша в файле"""
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
                       system_prompt: str = "", max_tokens: int = 1000, 
                       room_id: str = "default", callback: Callable = None) -> Optional[str]:
        """УМНАЯ ЛОГИКА ПРИОРИТЕТОВ С FALLBACK"""
        
        print(f"🔧 [LLM] Запрос с приоритетом '{self.priority_mode}': {prompt[:100]}...")
        
        # ЛОГИКА ВЫБОРА МОДЕЛИ ПО ПРИОРИТЕТУ
        if self.priority_mode == "local_only":
            return self._handle_local_request(prompt, system_prompt, max_tokens, room_id, callback)
            
        elif self.priority_mode == "openrouter_only":
            return self._handle_openrouter_request(prompt, context, subject, system_prompt, max_tokens, room_id, callback)
            
        elif self.priority_mode == "openrouter_first":
            # Сначала пробуем OpenRouter
            response = self._handle_openrouter_request(prompt, context, subject, system_prompt, max_tokens, room_id, callback)
            if response or (callback is None and response is not None):
                return response
            # Fallback на локальную модель
            return self._handle_local_request(prompt, system_prompt, max_tokens, room_id, callback)
            
        else:  # local_first (по умолчанию)
            # Сначала пробуем локальную модель
            response = self._handle_local_request(prompt, system_prompt, max_tokens, room_id, callback)
            if response or (callback is None and response is not None):
                return response
            # Fallback на OpenRouter
            return self._handle_openrouter_request(prompt, context, subject, system_prompt, max_tokens, room_id, callback)
    
    def _handle_local_request(self, prompt: str, system_prompt: str, max_tokens: int,
                             room_id: str, callback: Callable) -> Optional[str]:
        """Обработка запроса через локальную модель"""
        if not self.llm_manager.local_llm.is_available():
            print("❌ [LLM] Локальная модель недоступна")
            return None
            
        print(f"⚡ [LLM] Использую локальную модель для комнаты {room_id}")
        
        # Асинхронный режим с callback
        if callback:
            request_id = self.llm_manager.submit_request(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                room_id=room_id
            )
            self.pending_requests[request_id] = callback
            return None
            
        # Синхронный режим с увеличенным таймаутом
        else:
            response_queue = queue.Queue()
            
            def sync_callback(req_id, response, r_id):
                response_queue.put(response)
            
            self.llm_manager.register_room_callback(room_id, sync_callback)
            
            request_id = self.llm_manager.submit_request(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                room_id=room_id
            )
            
            try:
                # УВЕЛИЧИВАЕМ ТАЙМАУТ ДО 120 СЕКУНД
                response = response_queue.get(timeout=120)
                self.llm_manager.unregister_room_callback(room_id)
                return response
            except queue.Empty:
                print(f"❌ [LLM] Таймаут локальной модели для комнаты {room_id}")
                self.llm_manager.unregister_room_callback(room_id)
                return None
    
    def _handle_openrouter_request(self, prompt: str, context: str, subject: str,
                                  system_prompt: str, max_tokens: int,
                                  room_id: str, callback: Callable) -> Optional[str]:
        """Обработка запроса через OpenRouter с использованием менеджера ключей"""
        try:
            # Получаем следующий доступный ключ
            try:
                api_key = self.key_manager.get_next_key()
            except Exception as e:
                print(f"❌ [LLM] {e}")
                return None if callback else self._get_fallback_response(prompt, subject)
            
            print(f"🔧 [LLM] Использую OpenRouter для комнаты {room_id} (ключ: {api_key[:8]}...)")
            
            # Добавляем задержку между запросами
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.request_delay:
                time.sleep(self.request_delay - time_since_last_request)
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://ai-teacher.com",
                "X-Title": "AI Teacher"
            }
            
            # Улучшенный промпт
            final_system_prompt = system_prompt or f"""Ты - профессиональный учитель и эксперт по предмету "{subject}". 
    Отвечай максимально подробно и информативно. Объясняй сложные понятия простым языком.
    Контекст: {context}"""
    
            messages = []
            
            if final_system_prompt:
                messages.append({"role": "system", "content": final_system_prompt})
            
            if context and context.strip():
                messages.append({"role": "system", "content": f"Контекст разговора: {context}"})
            
            messages.append({"role": "user", "content": prompt})
    
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": max_tokens,
                "stream": False
            }
    
            print(f"🔧 [LLM] Отправка запроса к OpenRouter для комнаты {room_id}")
            
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(
                        self.api_url,
                        headers=headers,
                        json=data,
                        timeout=self.request_timeout
                    )
                    
                    self.last_request_time = time.time()
                    
                    if response.status_code == 200:
                        # ЗАПИСЫВАЕМ ИСПОЛЬЗОВАНИЕ КЛЮЧА ПРИ УСПЕХЕ
                        self.key_manager.record_usage(api_key)
                        
                        result = response.json()
                        
                        if 'choices' in result and len(result['choices']) > 0:
                            answer = result['choices'][0]['message']['content']
                            # ОЧИСТКА ТЕКСТА ПЕРЕД ВОЗВРАТОМ
                            answer = clean_text_for_speech(answer)
                            processed_answer = self._clean_llm_response(answer)
                            
                            print(f"✅ [LLM] Ответ от OpenRouter получен и очищен: {processed_answer[:100]}...")
                            
                            # Если есть callback, вызываем его
                            if callback:
                                callback(processed_answer, room_id)
                                return None
                            else:
                                return processed_answer
                        else:
                            print("❌ [LLM] Неверный формат ответа от OpenRouter")
                            return self._get_fallback_response(prompt, subject)
                            
                    elif response.status_code == 429:
                        wait_time = self.retry_delay * (attempt + 1)
                        print(f"⏳ [LLM] Rate Limit. Ждем {wait_time} сек...")
                        time.sleep(wait_time)
                        continue
                        
                    elif response.status_code == 401:
                        print(f"❌ [LLM] Ошибка аутентификации OpenRouter (неверный API ключ)")
                        # Помечаем OpenRouter как недоступный для этого запроса
                        return None if callback else self._get_fallback_response(prompt, subject)
                        
                    else:
                        print(f"❌ [LLM] Ошибка OpenRouter: {response.status_code}")
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay)
                            continue
                        return self._get_fallback_response(prompt, subject)
                        
                except requests.exceptions.Timeout:
                    print(f"⏰ [LLM] Таймаут OpenRouter. Попытка {attempt + 1}/{self.max_retries}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return self._get_fallback_response(prompt, subject)
                    
                except Exception as e:
                    print(f"❌ [LLM] Ошибка OpenRouter (попытка {attempt + 1}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return self._get_fallback_response(prompt, subject)
            
        except Exception as e:
            print(f"❌ [LLM] Критическая ошибка OpenRouter: {e}")
            return self._get_fallback_response(prompt, subject)
        
        return self._get_fallback_response(prompt, subject)

    def _test_openrouter_connection(self) -> bool:
        """УПРОЩЕННАЯ проверка доступности OpenRouter API"""
        # Теперь считаем, что наличие ключа достаточно для доступности
        # Фактическую проверку соединения делаем только при реальных запросах
        return bool(self.key_manager.keys)

    def _get_fallback_response(self, prompt: str, subject: str = "") -> str:
        """Возвращает fallback ответ когда LLM недоступен"""
        prompt_lower = prompt.lower()
        
        # УЛУЧШЕННЫЕ FALLBACK ОТВЕТЫ
        if any(word in prompt_lower for word in ['привет', 'здравств', 'начать', 'старт']):
            return "Привет! Я ваш AI-учитель. Давайте выберем предмет для изучения - математика, история, обществознание или другой?"
        
        if any(word in prompt_lower for word in ['спасибо', 'благодар']):
            return "Пожалуйста! Рад был помочь. Есть еще вопросы?"
        
        if any(word in prompt_lower for word in ['как дела', 'настроен']):
            return "Всё отлично! Готов к интересному уроку. Какой предмет вас интересует?"
        
        # Для образовательных вопросов даем более полезный ответ
        if any(word in prompt_lower for word in ['что такое', 'объясни', 'расскажи', 'как работает']):
            return "Хороший вопрос! Давайте разберем эту тему подробнее. Мне нужно немного времени подумать..."
        
        # Общий fallback ответ
        return "Спасибо за вопрос! Я подумаю над ответом и скоро вернусь с подробным объяснением."

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
        
        # ДАЕМ БОЛЬШЕ ВРЕМЕНИ ДЛЯ ОБРАБОТКИ
        start_time = time.time()
        
        # Запрос к реальному LLM
        llm_response = self._query_llm_api(question, context, subject)
        
        total_time = time.time() - start_time
        print(f"⏱️ Общее время обработки: {total_time:.2f}с")
        
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

    def set_priority(self, priority: str):
        """Установка приоритета моделей"""
        valid_priorities = ["local_first", "openrouter_first", "local_only", "openrouter_only"]
        if priority in valid_priorities:
            self.priority_mode = priority
            print(f"🔧 [LLM] Приоритет изменен на: {priority}")
        else:
            print(f"❌ [LLM] Неверный приоритет: {priority}")

    def get_priority_status(self) -> Dict:
        """Получение статуса приоритетов"""
        local_available = self.llm_manager.local_llm.is_available()
        
        # УПРОЩЕННАЯ ПРОВЕРКА OPENROUTER - наличие ключа = доступен
        openrouter_available = bool(self.key_manager.keys)
        
        return {
            "current_priority": self.priority_mode,
            "local_available": local_available,
            "openrouter_available": openrouter_available,
            "local_status": self.llm_manager.local_llm.get_status(),
            "effective_priority": self._get_effective_priority()
        }
    
    def _get_effective_priority(self) -> str:
        """Фактический приоритет с учетом доступности - УПРОЩЕННАЯ ВЕРСИЯ"""
        local_available = self.llm_manager.local_llm.is_available()
        
        # УПРОЩЕННАЯ ПРОВЕРКА OPENROUTER - наличие ключа = доступен
        openrouter_available = bool(self.key_manager.keys)
        
        if self.priority_mode == "local_only":
            return "local_only"
        elif self.priority_mode == "openrouter_only":
            return "openrouter_only" if openrouter_available else "fallback"
        elif self.priority_mode == "local_first":
            return "local_first" if local_available else ("openrouter_only" if openrouter_available else "fallback")
        elif self.priority_mode == "openrouter_first":
            return "openrouter_first" if openrouter_available else ("local_only" if local_available else "fallback")
        else:
            return self.priority_mode

    def get_llm_status(self) -> Dict:
        """Получение статуса моделей с УПРОЩЕННОЙ проверкой OpenRouter"""
        local_available = self.llm_manager.local_llm.is_available()
        
        # УПРОЩЕННАЯ ПРОВЕРКА OPENROUTER - наличие ключа = доступен
        openrouter_available = bool(self.key_manager.keys)
        
        # ПРОВЕРЯЕМ РАБОТОСПОСОБНОСТЬ ЛОКАЛЬНОЙ МОДЕЛИ
        local_working = False
        if local_available:
            try:
                test_response = self.llm_manager.local_llm.generate("Тест", "Ты - учитель", 10)
                local_working = test_response is not None and len(test_response.strip()) > 5
            except:
                local_working = False
        
        return {
            "local_available": local_available,
            "local_working": local_working,
            "openrouter_available": openrouter_available,
            "current_priority": self.priority_mode,
            "effective_priority": self._get_effective_priority(),
            "local_status": self.llm_manager.local_llm.get_status(),
            "local_url": self.llm_manager.local_llm.base_url,
            "cache_stats": self.get_cache_stats()
        }

    def handle_llm_response(self, request_id: str, response: str, room_id: str):
        """Обработчик ответов от локальной LLM"""
        if request_id in self.pending_requests:
            callback = self.pending_requests.pop(request_id)
            try:
                callback(response, room_id)
            except Exception as e:
                print(f"❌ Ошибка в callback для запроса {request_id}: {e}")

    def get_visualization_data(self, topic: str, context: str = "") -> Dict:
        """Запрашивает у LLM структурированные данные для визуализации"""
        prompt = f"""
На основе темы урока предоставь дополнительные структурированные данные для визуализации.

ТЕМА УРОКА: {topic}
КОНТЕКСТ: {context}

ТРЕБОВАНИЯ:
1. НЕ повторяй текст урока - предоставь ДОПОЛНИТЕЛЬНУЮ информацию
2. Выдели 3-5 ключевых понятий или аспектов
3. Укажи связи между ними
4. Предложи тип визуализации: классификация, процесс, иерархия, сравнение
5. Добавь краткие пояснения для каждого элемента

ФОРМАТ ОТВЕТА (JSON):
{{
    "type": "тип_визуализации",
    "main_concept": "главное_понятие",
    "elements": [
        {{
            "name": "название_элемента",
            "description": "краткое_пояснение",
            "connections": ["связанный_элемент1", "связанный_элемент2"]
        }}
    ],
    "additional_info": "дополнительная_информация_для_углубления"
}}

Пример для темы "Фотосинтез":
{{
    "type": "process",
    "main_concept": "Фотосинтез",
    "elements": [
        {{
            "name": "Световая фаза",
            "description": "Поглощение света и расщепление воды",
            "connections": ["Темновая фаза"]
        }},
        {{
            "name": "Темновая фаза", 
            "description": "Синтез глюкозы из CO2",
            "connections": ["Световая фаза"]
        }}
    ],
    "additional_info": "Эффективность фотосинтеза зависит от интенсивности света и концентрации CO2"
}}

Верни ТОЛЬКО JSON без пояснений!
"""
        
        try:
            response = self._query_llm_api(
                prompt=prompt,
                context="",
                subject="general",
                system_prompt="""Ты предоставляешь структурированные данные для визуализации учебного материала.
                Строго соблюдай формат JSON.
                Предоставляй ДОПОЛНИТЕЛЬНУЮ информацию, не повторяй текст урока.
                Выделяй ключевые связи и структуры.""",
                max_tokens=800
            )
            
            if response:
                print(f"🔧 [Visualization] Сырой ответ от LLM: {response[:200]}...")
                
                # Пытаемся извлечь JSON из ответа
                json_data = self._extract_json_from_response(response)
                if json_data and self._validate_visualization_data(json_data):
                    print(f"✅ Получены структурированные данные для визуализации")
                    return json_data
                else:
                    print(f"⚠️ Не удалось извлечь JSON, используется fallback")
                    
        except Exception as e:
            print(f"❌ Ошибка получения данных визуализации: {e}")
        
        # Fallback данные
        return self._get_fallback_visualization_data(topic)

    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """Извлекает JSON из ответа LLM"""
        try:
            # Ищем JSON в ответе
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
        except Exception as e:
            print(f"❌ Ошибка парсинга JSON: {e}")
        
        return None

    def _validate_visualization_data(self, data: Dict) -> bool:
        """Проверяет валидность данных визуализации"""
        required_fields = ['type', 'main_concept', 'elements']
        return all(field in data for field in required_fields)

    def _get_fallback_visualization_data(self, topic: str) -> Dict:
        """Fallback данные для визуализации"""
        concepts = self._extract_key_concepts(topic)
        
        return {
            "type": "classification",
            "main_concept": concepts[0] if concepts else "Основное понятие",
            "elements": [
                {
                    "name": concepts[1] if len(concepts) > 1 else "Аспект 1",
                    "description": "Важная характеристика",
                    "connections": [concepts[0] if concepts else "Основное понятие"]
                },
                {
                    "name": concepts[2] if len(concepts) > 2 else "Аспект 2", 
                    "description": "Дополнительная информация",
                    "connections": [concepts[0] if concepts else "Основное понятие"]
                }
            ],
            "additional_info": "Изучите взаимосвязи между понятиями для лучшего понимания"
        }

    def _extract_key_concepts(self, text: str) -> List[str]:
        """Извлекает ключевые понятия из текста"""
        stop_words = {
            'это', 'которые', 'который', 'которых', 'включает', 'себя', 'основные', 
            'понятия', 'статистики', 'описательная', 'индуктивная', 'другими', 
            'словами', 'является', 'также', 'может', 'иметь', 'быть', 'очень'
        }
        
        words = re.findall(r'\b[А-Яа-я]{4,}\b', text.lower())
        filtered_words = [word for word in words if word not in stop_words]
        unique_concepts = list(dict.fromkeys(filtered_words))
        
        if len(unique_concepts) < 3:
            unique_concepts.extend(["Основное понятие", "Характеристика", "Пример"])
        
        return unique_concepts[:5]

    def generate_mermaid_from_data(self, viz_data: Dict) -> str:
        """Генерирует Mermaid код из структурированных данных"""
        try:
            viz_type = viz_data.get('type', 'classification')
            main_concept = viz_data.get('main_concept', 'Основное понятие')
            elements = viz_data.get('elements', [])
            
            if viz_type == 'process':
                return self._generate_process_mermaid(main_concept, elements)
            elif viz_type == 'hierarchy':
                return self._generate_hierarchy_mermaid(main_concept, elements)
            elif viz_type == 'comparison':
                return self._generate_comparison_mermaid(main_concept, elements)
            else:  # classification
                return self._generate_classification_mermaid(main_concept, elements)
                
        except Exception as e:
            print(f"❌ Ошибка генерации Mermaid: {e}")
            return self._generate_fallback_mermaid()

    def _generate_classification_mermaid(self, main_concept: str, elements: List[Dict]) -> str:
        """Генерация Mermaid для классификации"""
        mermaid_lines = ['flowchart TD']
        mermaid_lines.append(f'    A["{main_concept}"]')
        
        for i, element in enumerate(elements[:4]):  # Максимум 4 элемента
            node_id = chr(66 + i)  # B, C, D, E
            element_name = element.get('name', f'Элемент {i+1}')
            mermaid_lines.append(f'    A --> {node_id}["{element_name}"]')
            
            # Добавляем связи если есть
            connections = element.get('connections', [])
            for conn in connections[:2]:  # Максимум 2 связи на элемент
                if conn != main_concept:
                    mermaid_lines.append(f'    {node_id} --> F["{conn}"]')
        
        # Стили
        mermaid_lines.extend([
            '    ',
            '    style A fill:#4263EB,color:#fff,stroke:#333,stroke-width:2px',
            '    style B fill:#4cc9f0,color:#333,stroke:#333,stroke-width:2px',
            '    style C fill:#3a0ca3,color:#fff,stroke:#333,stroke-width:2px',
            '    style D fill:#f72585,color:#fff,stroke:#333,stroke-width:2px'
        ])
        
        return '\n'.join(mermaid_lines)

    def _generate_process_mermaid(self, main_concept: str, elements: List[Dict]) -> str:
        """Генерация Mermaid для процесса"""
        mermaid_lines = ['flowchart LR']
        mermaid_lines.append(f'    A["{main_concept}"]')
        
        prev_node = 'A'
        for i, element in enumerate(elements[:4]):
            node_id = chr(66 + i)
            element_name = element.get('name', f'Этап {i+1}')
            mermaid_lines.append(f'    {prev_node} --> {node_id}["{element_name}"]')
            prev_node = node_id
        
        return '\n'.join(mermaid_lines)

    def _generate_hierarchy_mermaid(self, main_concept: str, elements: List[Dict]) -> str:
        """Генерация Mermaid для иерархии"""
        mermaid_lines = ['flowchart TD']
        mermaid_lines.append(f'    A["{main_concept}"]')
        
        for i, element in enumerate(elements[:3]):
            node_id = chr(66 + i)
            element_name = element.get('name', f'Уровень {i+1}')
            mermaid_lines.append(f'    A --> {node_id}["{element_name}"]')
            
            # Подэлементы для иерархии
            sub_node = chr(69 + i)  # E, F, G
            mermaid_lines.append(f'    {node_id} --> {sub_node}["Подраздел {i+1}"]')
        
        return '\n'.join(mermaid_lines)

    def _generate_comparison_mermaid(self, main_concept: str, elements: List[Dict]) -> str:
        """Генерация Mermaid для сравнения"""
        mermaid_lines = ['flowchart TD']
        mermaid_lines.append(f'    A["{main_concept}"]')
        mermaid_lines.append('    A --> B["Сходства"]')
        mermaid_lines.append('    A --> C["Различия"]')
        
        for i, element in enumerate(elements[:2]):
            node_b = chr(68 + i)  # D, E
            node_c = chr(70 + i)  # F, G
            element_name = element.get('name', f'Аспект {i+1}')
            mermaid_lines.append(f'    B --> {node_b}["{element_name}"]')
            mermaid_lines.append(f'    C --> {node_c}["{element_name}"]')
        
        return '\n'.join(mermaid_lines)

    def _generate_fallback_mermaid(self) -> str:
        """Fallback Mermaid диаграмма"""
        return '''flowchart TD
    A["Основное понятие"] --> B["Характеристика 1"]
    A --> C["Характеристика 2"]
    B --> D["Конкретный пример"]
    C --> D
    
    style A fill:#4263EB,color:#fff,stroke:#333,stroke-width:2px
    style B fill:#4cc9f0,color:#333,stroke:#333,stroke-width:2px
    style C fill:#3a0ca3,color:#fff,stroke:#333,stroke-width:2px'''

    def generate_svg_from_data(self, viz_data: Dict) -> str:
        """Генерирует SVG из структурированных данных"""
        try:
            viz_type = viz_data.get('type', 'classification')
            main_concept = viz_data.get('main_concept', 'Основное понятие')
            elements = viz_data.get('elements', [])
            
            short_concept = main_concept[:20] + "..." if len(main_concept) > 20 else main_concept
            
            if viz_type == 'process':
                return self._generate_process_svg(short_concept, elements)
            elif viz_type == 'hierarchy':
                return self._generate_hierarchy_svg(short_concept, elements)
            else:  # classification and comparison
                return self._generate_classification_svg(short_concept, elements)
                
        except Exception as e:
            print(f"❌ Ошибка генерации SVG: {e}")
            return self._generate_fallback_svg()

    def _generate_classification_svg(self, main_concept: str, elements: List[Dict]) -> str:
        """Генерация SVG для классификации"""
        element_names = [elem.get('name', f'Элемент {i+1}') for i, elem in enumerate(elements[:3])]
        
        return f'''
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300">
            <defs>
                <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#f8f9fa"/>
                    <stop offset="100%" stop-color="#e9ecef"/>
                </linearGradient>
            </defs>
            
            <rect x="10" y="10" width="380" height="280" fill="url(#bg)" stroke="#dee2e6" stroke-width="2" rx="15"/>
            
            <rect x="50" y="30" width="300" height="40" fill="#4263EB" rx="10"/>
            <text x="200" y="55" text-anchor="middle" font-family="Arial" font-size="16" fill="white" font-weight="bold">
                {main_concept}
            </text>
            
            <rect x="150" y="100" width="100" height="40" fill="#4263EB" rx="8"/>
            <text x="200" y="125" text-anchor="middle" font-family="Arial" font-size="12" fill="white">
                Основное
            </text>
            
            <rect x="50" y="180" width="80" height="30" fill="#4cc9f0" rx="6"/>
            <text x="90" y="198" text-anchor="middle" font-family="Arial" font-size="10" fill="#333">
                {element_names[0] if len(element_names) > 0 else "Аспект 1"}
            </text>
            
            <rect x="160" y="180" width="80" height="30" fill="#3a0ca3" rx="6"/>
            <text x="200" y="198" text-anchor="middle" font-family="Arial" font-size="10" fill="white">
                {element_names[1] if len(element_names) > 1 else "Аспект 2"}
            </text>
            
            <rect x="270" y="180" width="80" height="30" fill="#f72585" rx="6"/>
            <text x="310" y="198" text-anchor="middle" font-family="Arial" font-size="10" fill="white">
                {element_names[2] if len(element_names) > 2 else "Аспект 3"}
            </text>
            
            <line x1="200" y1="140" x2="130" y2="180" stroke="#333" stroke-width="2"/>
            <line x1="200" y1="140" x2="200" y2="180" stroke="#333" stroke-width="2"/>
            <line x1="200" y1="140" x2="270" y2="180" stroke="#333" stroke-width="2"/>
        </svg>
        '''.strip()

    def _generate_process_svg(self, main_concept: str, elements: List[Dict]) -> str:
        """Генерация SVG для процесса"""
        element_names = [elem.get('name', f'Этап {i+1}') for i, elem in enumerate(elements[:4])]
        
        return f'''
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300">
            <defs>
                <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#f8f9fa"/>
                    <stop offset="100%" stop-color="#e9ecef"/>
                </linearGradient>
            </defs>
            
            <rect x="10" y="10" width="380" height="280" fill="url(#bg)" stroke="#dee2e6" stroke-width="2" rx="15"/>
            
            <text x="200" y="40" text-anchor="middle" font-family="Arial" font-size="16" fill="#333" font-weight="bold">
                {main_concept}
            </text>
            
            <rect x="50" y="80" width="60" height="40" fill="#4263EB" rx="8"/>
            <text x="80" y="105" text-anchor="middle" font-family="Arial" font-size="10" fill="white">
                {element_names[0] if len(element_names) > 0 else "Начало"}
            </text>
            
            <rect x="140" y="80" width="60" height="40" fill="#4cc9f0" rx="8"/>
            <text x="170" y="105" text-anchor="middle" font-family="Arial" font-size="10" fill="#333">
                {element_names[1] if len(element_names) > 1 else "Развитие"}
            </text>
            
            <rect x="230" y="80" width="60" height="40" fill="#3a0ca3" rx="8"/>
            <text x="260" y="105" text-anchor="middle" font-family="Arial" font-size="10" fill="white">
                {element_names[2] if len(element_names) > 2 else "Завершение"}
            </text>
            
            <rect x="320" y="80" width="60" height="40" fill="#f72585" rx="8"/>
            <text x="350" y="105" text-anchor="middle" font-family="Arial" font-size="10" fill="white">
                {element_names[3] if len(element_names) > 3 else "Результат"}
            </text>
            
            <line x1="110" y1="100" x2="140" y2="100" stroke="#333" stroke-width="2"/>
            <polygon points="140,100 135,95 135,105" fill="#333"/>
            
            <line x1="200" y1="100" x2="230" y2="100" stroke="#333" stroke-width="2"/>
            <polygon points="230,100 225,95 225,105" fill="#333"/>
            
            <line x1="290" y1="100" x2="320" y2="100" stroke="#333" stroke-width="2"/>
            <polygon points="320,100 315,95 315,105" fill="#333"/>
        </svg>
        '''.strip()

    def _generate_hierarchy_svg(self, main_concept: str, elements: List[Dict]) -> str:
        """Генерация SVG для иерархии"""
        return self._generate_classification_svg(main_concept, elements)

    def _generate_fallback_svg(self) -> str:
        """Fallback SVG"""
        return '''
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300">
            <rect x="10" y="10" width="380" height="280" fill="#f8f9fa" stroke="#dee2e6" stroke-width="2" rx="15"/>
            <text x="200" y="150" text-anchor="middle" font-family="Arial" font-size="16" fill="#666">
                Визуализация загружается...
            </text>
        </svg>
        '''.strip()

    def check_visualization_need(self, text: str) -> bool:
        """Проверяет, нужна ли визуализация для данного текста"""
        if not text or len(text.strip()) < 10:
            return False
            
        text_lower = text.lower()
        
        # Ключевые слова, указывающие на необходимость визуализации
        visualization_keywords = [
            'структура', 'схема', 'диаграмма', 'график', 'процесс', 
            'алгоритм', 'иерархия', 'взаимосвязь', 'соотношение',
            'таблица', 'классификация', 'этапы', 'стадии', 'система',
            'модель', 'цепочка', 'последовательность', 'отношение',
            'разделение', 'группировка', 'организация', 'архитектура',
            'понятие', 'определение', 'теория', 'концепция', 'принцип',
            'механизм', 'функция', 'свойство', 'характеристика'
        ]
        
        # Структурные индикаторы
        structure_indicators = [
            'состоит из', 'включает в себя', 'делится на', 'подразделяется',
            'можно разделить', 'выделяют', 'различают', 'существуют'
        ]
        
        # Проверяем наличие ключевых слов
        has_keywords = any(keyword in text_lower for keyword in visualization_keywords)
        
        # Проверяем наличие структурных индикаторов
        has_structure = any(indicator in text_lower for indicator in structure_indicators)
        
        # Проверяем длину текста (достаточно информативный)
        is_long_enough = len(text.split()) > 5
        
        return (has_keywords or has_structure) and is_long_enough

    def generate_visualization(self, topic: str, context: str = "") -> dict:
        """Генерация визуализаций через структурированные данные от LLM"""
        try:
            print(f"🎨 Генерация визуализаций для: {topic}")
            
            # Получаем структурированные данные от LLM
            viz_data = self.get_visualization_data(topic, context)
            print(f"🔧 Получены данные для визуализации: {viz_data.get('type', 'unknown')}")
            
            # Генерируем Mermaid из данных
            mermaid_code = self.generate_mermaid_from_data(viz_data)
            
            # Генерируем SVG из данных
            svg_code = self.generate_svg_from_data(viz_data)
            
            result = {
                "mermaid_code": mermaid_code,
                "svg_code": svg_code,
                "topic": topic,
                "viz_data": viz_data,  # Возвращаем также сырые данные
                "success": bool(mermaid_code and svg_code)
            }
            
            if result["success"]:
                print(f"✅ Визуализации сгенерированы на основе структурированных данных")
            else:
                print(f"❌ Не удалось сгенерировать визуализации")
                
            return result
            
        except Exception as e:
            print(f"❌ Ошибка генерации визуализаций: {e}")
            return {
                "mermaid_code": "",
                "svg_code": "",
                "topic": topic,
                "success": False,
                "error": str(e)
            }

    def test_connection(self) -> bool:
        """Тестирование подключения к API"""
        try:
            # Получаем ключ для тестирования
            api_key = self.key_manager.get_next_key()
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            test_data = {
                "model": self.model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=test_data,
                timeout=10
            )
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"❌ Ошибка тестирования подключения: {e}")
            return False

    def get_available_models(self) -> list:
        """Получение списка доступных моделей"""
        return [
            {"id": "llama", "name": "Llama 3.3 8B", "description": "Мощная и быстрая модель от Meta"},
            {"id": "qwen", "name": "Qwen 2.5 32B", "description": "Качественная модель от Alibaba"},
            {"id": "deepseek", "name": "DeepSeek Chat", "description": "Продвинутая модель для сложных задач"}
        ]

# Создаем глобальный экземпляр для использования в других модулях
llm_integration = LLMIntegration()

def get_llm_instance() -> LLMIntegration:
    """Возвращает глобальный экземпляр LLMIntegration"""
    return llm_integration

if __name__ == "__main__":
    # Тестирование модуля
    llm = LLMIntegration()
    
    print("🔧 Тестирование улучшенного LLM модуля...")
    
    # Тестирование генерации визуализации
    test_topic = "Статистика включает описательную статистику и индуктивную статистику"
    print(f"\n🔄 Генерация визуализации для: {test_topic}")
    viz_result = llm.generate_visualization(test_topic)
    
    if viz_result["success"]:
        print("✅ Визуализации успешно сгенерированы!")
        print(f"📊 Тип визуализации: {viz_result['viz_data'].get('type', 'unknown')}")
        print(f"📊 Основное понятие: {viz_result['viz_data'].get('main_concept', 'unknown')}")
        if viz_result["mermaid_code"]:
            print(f"📊 Mermaid код:")
            print(viz_result['mermaid_code'])
    else:
        print("❌ Не удалось сгенерировать визуализации")
    
    print("\n🎉 Тестирование завершено!")