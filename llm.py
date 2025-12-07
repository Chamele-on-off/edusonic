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
                 model: str = "meta-llama/llama-3.3-70b-instruct:free"):
        
        config = load_config()
        openrouter_config = get_model_config("openrouter")
        
        # 🔥 ИСПРАВЛЕНИЕ: Используем менеджер ключей вместо одного ключа
        self.key_manager = get_key_manager()  # Ключевое изменение!
        self.api_url = api_url or openrouter_config.get("api_url", "https://openrouter.ai/api/v1/chat/completions")
        self.model = model or openrouter_config.get("model", "meta-llama/llama-3.3-70b-instruct:free")
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
        
        print(f"🔧 [LLM] Инициализирован с приоритетом: {self.priority_mode}")
        print(f"🔧 [LLM] Используется менеджер ключей: {len(self.key_manager.keys)} ключей доступно")
        
    def _load_cache(self) -> Dict:
        """Загрузка кэша из файла"""
        cache_file = self.cache_dir / "llm_cache.json"
        try:
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"❌ Ошибка загрузки кэша: {e}")
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
            print(f"❌ Ошибка сохранения кэша: {e}")

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
        
        # Удаляем лишние пробелы и переносы
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n+', '\n', content)
        
        return content.strip()

    def _process_llm_response(self, response: str, is_svg: bool = False) -> str:
        """Обработка ответа LLM в зависимости от типа"""
        if is_svg:
            # Для SVG - минимальная очистка, сохраняем структуру
            print(f"🔧 [LLM] Обработка SVG ответа (без очистки речи)")
            return response.strip()
        else:
            # Для речи - полная очистка
            print(f"🔧 [LLM] Обработка текстового ответа (с очисткой речи)")
            clean1 = clean_text_for_speech(response)
            return self._clean_llm_response(clean1)

    def _query_llm_api(self, prompt: str, context: str = "", subject: str = "", 
                       system_prompt: str = "", max_tokens: int = 1000, 
                       room_id: str = "default", callback: Callable = None,
                       is_svg: bool = False) -> Optional[str]:
        """УМНАЯ ЛОГИКА ПРИОРИТЕТОВ С FALLBACK - БЕЗ БЛОКИРОВАНИЯ"""
        
        print(f"🔧 [LLM] Запрос с приоритетом '{self.priority_mode}': {prompt[:100]}...")
        print(f"🔧 [LLM] Тип запроса: {'SVG' if is_svg else 'Текст'}")
        
        # 🔥 КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: Используем асинхронный режим чтобы не блокировать систему
        if callback:
            # Асинхронный режим - запускаем в фоне
            import threading
            def async_query():
                try:
                    result = self._execute_llm_query(
                        prompt, context, subject, system_prompt, 
                        max_tokens, room_id, is_svg
                    )
                    if result:
                        callback(result, room_id)
                except Exception as e:
                    print(f"❌ [LLM] Ошибка асинхронного запроса: {e}")
                    # Всегда возвращаем fallback чтобы не блокировать систему
                    fallback = self._get_fallback_response(prompt, subject)
                    callback(fallback, room_id)
            
            # Запускаем в отдельном потоке
            thread = threading.Thread(target=async_query, daemon=True)
            thread.start()
            return None
        else:
            # Синхронный режим - с таймаутом и fallback
            try:
                return self._execute_llm_query(
                    prompt, context, subject, system_prompt,
                    max_tokens, room_id, is_svg
                )
            except Exception as e:
                print(f"❌ [LLM] Ошибка синхронного запроса: {e}")
                return self._get_fallback_response(prompt, subject)
    
    def _execute_llm_query(self, prompt: str, context: str, subject: str,
                          system_prompt: str, max_tokens: int,
                          room_id: str, is_svg: bool = False) -> Optional[str]:
        """Выполнение LLM запроса с безопасным fallback"""
        
        # ЛОГИКА ВЫБОРА МОДЕЛИ ПО ПРИОРИТЕТУ
        if self.priority_mode == "local_only":
            return self._handle_local_request_safe(prompt, system_prompt, max_tokens, is_svg)
            
        elif self.priority_mode == "openrouter_only":
            return self._handle_openrouter_request_safe(prompt, context, subject, system_prompt, max_tokens, is_svg)
            
        elif self.priority_mode == "openrouter_first":
            # Сначала пробуем OpenRouter
            response = self._handle_openrouter_request_safe(prompt, context, subject, system_prompt, max_tokens, is_svg)
            if response:
                return response
            # Fallback на локальную модель
            return self._handle_local_request_safe(prompt, system_prompt, max_tokens, is_svg)
            
        else:  # local_first (по умолчанию)
            # Сначала пробуем локальную модель
            response = self._handle_local_request_safe(prompt, system_prompt, max_tokens, is_svg)
            if response:
                return response
            # Fallback на OpenRouter
            return self._handle_openrouter_request_safe(prompt, context, subject, system_prompt, max_tokens, is_svg)
    
    def _handle_local_request_safe(self, prompt: str, system_prompt: str, max_tokens: int,
                                 is_svg: bool = False) -> Optional[str]:
        """Безопасная обработка локальной модели с таймаутом"""
        if not self.llm_manager.local_llm.is_available():
            print("❌ [LLM] Локальная модель недоступна")
            return None
            
        print(f"⚡ [LLM] Использую локальную модель")
        
        # Создаем отдельный поток для таймаута
        import threading
        result_queue = queue.Queue()
        
        def local_query():
            try:
                response = self.llm_manager.local_llm.generate(
                    prompt, 
                    system_prompt or "Ты - профессиональный учитель", 
                    max_tokens
                )
                if response:
                    processed = self._process_llm_response(response, is_svg)
                    result_queue.put(processed)
                else:
                    result_queue.put(None)
            except Exception as e:
                print(f"❌ [LLM] Ошибка локальной модели: {e}")
                result_queue.put(None)
        
        # Запускаем с таймаутом
        thread = threading.Thread(target=local_query, daemon=True)
        thread.start()
        thread.join(timeout=30)  # 30 секунд таймаут
        
        if thread.is_alive():
            print("⏰ [LLM] Таймаут локальной модели")
            return None
        
        try:
            return result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _handle_openrouter_request_safe(self, prompt: str, context: str, subject: str,
                                      system_prompt: str, max_tokens: int,
                                      is_svg: bool = False) -> Optional[str]:
        """Безопасная обработка OpenRouter с менеджером ключей"""
        try:
            # 🔥 ИСПРАВЛЕНИЕ: Используем менеджер ключей
            api_key = self.key_manager.get_next_key()
            print(f"🔧 [LLM] Использую OpenRouter (ключ: {api_key[:8]}...)")
            
        except Exception as e:
            print(f"❌ [LLM] Нет доступных ключей OpenRouter: {e}")
            return None
            
        try:
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
    
            print(f"🔧 [LLM] Отправка запроса к OpenRouter")
            
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
                        # 🔥 ЗАПИСЫВАЕМ ИСПОЛЬЗОВАНИЕ КЛЮЧА
                        self.key_manager.record_usage(api_key)
                        
                        result = response.json()
                        
                        if 'choices' in result and len(result['choices']) > 0:
                            answer = result['choices'][0]['message']['content']
                            
                            processed_answer = self._process_llm_response(answer, is_svg)
                            
                            print(f"✅ [LLM] Ответ от OpenRouter получен: {processed_answer[:100]}...")
                            return processed_answer
                        else:
                            print("❌ [LLM] Неверный формат ответа от OpenRouter")
                            return None
                            
                    elif response.status_code == 429:
                        wait_time = self.retry_delay * (attempt + 1)
                        print(f"⏳ [LLM] Rate Limit. Ждем {wait_time} сек...")
                        time.sleep(wait_time)
                        continue
                        
                    elif response.status_code == 401:
                        print(f"❌ [LLM] Ошибка аутентификации OpenRouter (неверный API ключ)")
                        # Помечаем OpenRouter как недоступный для этого запроса
                        return None
                        
                    else:
                        print(f"❌ [LLM] Ошибка OpenRouter: {response.status_code}")
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay)
                            continue
                        return None
                        
                except requests.exceptions.Timeout:
                    print(f"⏰ [LLM] Таймаут OpenRouter. Попытка {attempt + 1}/{self.max_retries}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return None
                    
                except Exception as e:
                    print(f"❌ [LLM] Ошибка OpenRouter (попытка {attempt + 1}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return None
            
        except Exception as e:
            print(f"❌ [LLM] Критическая ошибка OpenRouter: {e}")
            return None
        
        return None

    def _test_openrouter_connection(self) -> bool:
        """УПРОЩЕННАЯ проверка доступности OpenRouter API"""
        # Теперь считаем, что наличие ключа достаточно для доступности
        # Фактическую проверку соединения делаем только при реальных запросах
        return bool(self.key_manager.keys)

    def _get_fallback_response(self, prompt: str, subject: str = "") -> str:
        """Возвращает fallback ответ когда LLM недоступен - НИКОГДА НЕ ПАДАЕТ"""
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

    def query(self, question: str, context: str = "", subject: str = "") -> str:
        """Запрос к LLM API с ГАРАНТИРОВАННЫМ ответом - НЕ БЛОКИРУЕТ СИСТЕМУ"""
        if not question.strip():
            return ""
            
        question_lower = question.lower().strip()
        
        # Проверка кэша
        cache_key = f"{subject}_{question_lower}" if subject else question_lower
        if cache_key in self.cache:
            print(f"💾 Использую кэшированный ответ для: {question_lower}")
            return self.cache[cache_key]
        
        print(f"📨 Запрос к LLM: '{question}' (предмет: {subject})")
        
        # 🔥 ВАЖНО: Используем асинхронный запрос с таймаутом
        start_time = time.time()
        
        try:
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
            
        except Exception as e:
            print(f"❌ Критическая ошибка в LLM.query: {e}")
            # 🔥 КРИТИЧЕСКО ВАЖНО: Всегда возвращаем ответ, даже если ошибка
            return "Спасибо за вопрос! Я обрабатываю ваш запрос."

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
            "cache_stats": self.get_cache_stats(),
            "key_manager_stats": {
                "total_keys": len(self.key_manager.keys),
                "active_keys": len([k for k in self.key_manager.keys if k.get('is_active', True)])
            }
        }

    def handle_llm_response(self, request_id: str, response: str, room_id: str):
        """Обработчик ответов от локальной LLM"""
        if request_id in self.pending_requests:
            callback, is_svg = self.pending_requests.pop(request_id)
            try:
                # Обрабатываем ответ в зависимости от типа
                processed_response = self._process_llm_response(response, is_svg)
                callback(processed_response, room_id)
            except Exception as e:
                print(f"❌ Ошибка в callback для запроса {request_id}: {e}")

    def _extract_svg_code(self, response: str) -> str:
        """Извлекает чистый SVG код из ответа LLM - устойчиво к обёрткам"""
        if not response:
            return ""
        
        print(f"🔧 [DEBUG] Извлечение SVG из ответа длиной: {len(response)}")
        
        # 1. Удаляем всё до первого <svg (включая пояснения)
        svg_start = response.find('<svg')
        if svg_start == -1:
            print("❌ [DEBUG] Не найден тег <svg в ответе")
            return ""
        response = response[svg_start:]
        
        # 2. Удаляем всё после последнего </svg>
        svg_end = response.rfind('</svg>')
        if svg_end == -1:
            print("❌ [DEBUG] Не найден закрывающий тег </svg> в ответе")
            return ""
        response = response[:svg_end + len('</svg>')]
        
        # 3. Удаляем возможные markdown-блоки (```xml, ```svg, ```)
        response = re.sub(r'^```(xml|svg)?\s*', '', response, flags=re.MULTILINE)
        response = re.sub(r'```\s*$', '', response, flags=re.MULTILINE)
        response = response.strip()
        
        # 4. Убеждаемся, что есть xmlns (добавляем если нет)
        if 'xmlns=' not in response:
            response = response.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"', 1)
            print("🔧 [DEBUG] Добавлен xmlns в SVG")
        
        # 5. Удаляем лишние пробелы и переносы внутри тега
        response = re.sub(r'<svg\s+', '<svg ', response)
        
        print(f"✅ [DEBUG] SVG успешно извлечен, длина: {len(response)}")
        return response

    def _validate_svg(self, svg_code: str) -> bool:
        """Упрощенная валидация SVG - только базовые проверки"""
        if not svg_code or len(svg_code.strip()) < 50:  # Минимальная длина
            print(f"❌ [DEBUG] SVG слишком короткий: {len(svg_code)} символов")
            return False
        
        # Проверяем базовую структуру
        has_svg_open = '<svg' in svg_code
        has_svg_close = '</svg>' in svg_code
        has_proper_tags = any(tag in svg_code for tag in ['<rect', '<text', '<circle', '<path', '<line'])
        
        print(f"🔧 [DEBUG] Валидация SVG: open={has_svg_open}, close={has_svg_close}, tags={has_proper_tags}")
        
        return has_svg_open and has_svg_close and has_proper_tags

    def generate_infographic(self, topic: str, context: str = "") -> dict:
        """Генерация стильной инфографики в SVG формате"""
        
        # 🔥 УЛУЧШЕННЫЙ ПРОМПТ: Больше разнообразия и креативности
        prompt = f"""
Создай УНИКАЛЬНУЮ и КРЕАТИВНУЮ образовательную инфографику в формате SVG на тему: "{topic}"

ТРЕБОВАНИЯ К ИНФОГРАФИКЕ:
- Только чистый SVG код без пояснений
- УНИКАЛЬНЫЙ дизайн для каждой темы
- Информативная и понятная структура  
- Использование разнообразных фигур, текста, цветов
- Максимальная ширина: 600px, высота: 400px
- Четкая визуальная иерархия
- Баланс между визуальными элементами и текстом

ВАРИАНТЫ СТРУКТУРЫ (выбери наиболее подходящую):
1. Иерархическая схема - для процессов и структур
2. Сравнительная таблица - для сравнения понятий  
3. Временная шкала - для исторических событий
4. Круговая диаграмма - для пропорций и соотношений
5. Блок-схема - для алгоритмов и процессов
6. Концептуальная карта - для связей между понятиями
7. Инфографика с иконками - для визуального представления

ЭЛЕМЕНТЫ ДЛЯ ИСПОЛЬЗОВАНИЯ:
- Прямоугольники, круги, треугольники, стрелки
- Линии, кривые, пути
- Градиенты, тени, фильтры
- Текстовые блоки с заголовками
- Иконки и символы (если уместно)

ЦВЕТОВАЯ ПАЛИТРА (используй разные комбинации):
- Основные цвета: #4f46e5, #10b981, #f59e0b, #ef4444, #8b5cf6, #06b6d4, #84cc16, #f97316
- Фон: светлые оттенки (#f8fafc, #f1f5f9, #fef7ed, #f0fdf4)
- Текст: темные оттенки (#1f2937, #374151, #4b5563)

СТРУКТУРА ИНФОГРАФИКИ:
1. Привлекательный заголовок
2. Ключевые элементы/понятия  
3. Визуальные связи между элементами
4. Подписи и пояснения (если нужны)

ТЕМА: "{topic}"

Создай УНИКАЛЬНЫЙ дизайн, который лучше всего подходит для этой темы.
Не используй шаблонные решения - будь креативным!

Верни ТОЛЬКО SVG код без каких-либо пояснений, комментариев или markdown разметки.
Код должен начинаться с <svg и заканчиваться </svg>.
"""

        try:
            print(f"🎨 Генерация УНИКАЛЬНОЙ SVG инфографики для: {topic}")
            
            response = self._query_llm_api(
                prompt=prompt,
                context="",  # 🔥 УБИРАЕМ КОНТЕКСТ УРОКА - это решит проблему с передачей текста
                subject="general",
                system_prompt="""Ты - креативный дизайнер образовательной инфографики. 
Твоя задача - создавать УНИКАЛЬНЫЕ и РАЗНООБРАЗНЫЕ SVG инфографики для разных тем.

ОЧЕНЬ ВАЖНЫЕ ПРАВИЛА:
1. Возвращай ТОЛЬКО SVG код, без каких-либо пояснений
2. НЕ ИСПОЛЬЗУЙ markdown разметку (```xml, ```svg, ```)
3. НЕ добавляй комментарии, пояснения или текст вокруг SVG
4. Код должен начинаться с <svg и заканчиваться </svg>
5. Всегда включай xmlns="http://www.w3.org/2000/svg"
6. Создавай РАЗНЫЕ типы инфографик для разных тем
7. Будь КРЕАТИВНЫМ - избегай шаблонных решений
8. Используй разнообразные цвета, формы и структуры

Примеры разных типов инфографик:
- Иерархические схемы для структур
- Временные шкалы для процессов  
- Сравнительные таблицы для анализа
- Концептуальные карты для связей
- Круговые диаграммы для пропорций

Верни ТОЛЬКО SVG код. НИЧЕГО БОЛЬШЕ.""",
                max_tokens=2500,  # Увеличили для более сложных дизайнов
                is_svg=True
            )
            
            # 🔧 ДОБАВЛЕНО ОТЛАДОЧНОЕ ЛОГИРОВАНИЕ
            print(f"🔧 [DEBUG] RAW LLM RESPONSE:")
            print(f"'{response}'")
            print(f"🔧 [DEBUG] Response length: {len(response) if response else 0}")
            
            if response:
                # Очистка ответа (для SVG минимальная очистка)
                svg_code = self._extract_svg_code(response)
                
                # 🔧 ДОБАВЛЕНО ЛОГИ ДЛЯ ОТЛАДКИ
                print(f"🔧 [DEBUG] Extracted SVG code length: {len(svg_code)}")
                print(f"🔧 [DEBUG] SVG validation: {self._validate_svg(svg_code)}")
                
                if svg_code and self._validate_svg(svg_code):
                    print(f"✅ [DEBUG] SVG успешно извлечен и валидирован!")
                    return {
                        "success": True,
                        "svg_code": svg_code,
                        "topic": topic,
                        "type": "infographic"
                    }
                else:
                    print(f"❌ [DEBUG] Не удалось извлечь валидный SVG")
                    if svg_code:
                        print(f"🔧 [DEBUG] Extracted code (первые 500 символов): '{svg_code[:500]}...'")
                    else:
                        print(f"🔧 [DEBUG] SVG код пустой")
            
            print("❌ [DEBUG] Используем fallback SVG")
            return {
                "success": False,
                "svg_code": self._create_fallback_infographic(topic),
                "topic": topic,
                "type": "fallback"
            }
            
        except Exception as e:
            print(f"❌ Ошибка генерации инфографики: {e}")
            return {
                "success": False,
                "svg_code": self._create_fallback_infographic(topic),
                "topic": topic,
                "type": "error_fallback"
            }

    def _create_fallback_infographic(self, topic: str) -> str:
        """Создает простую SVG инфографику как fallback"""
        topic_short = topic[:50] + "..." if len(topic) > 50 else topic
        
        return f'''<svg width="600" height="400" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bgGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#4f46e5" />
      <stop offset="100%" stop-color="#7c3aed" />
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="4" dy="4" stdDeviation="8" flood-color="#000000" flood-opacity="0.3"/>
    </filter>
  </defs>
  
  <!-- Фон -->
  <rect width="100%" height="100%" fill="url(#bgGradient)" opacity="0.1"/>
  
  <!-- Основной контейнер -->
  <g filter="url(#shadow)">
    <rect x="50" y="50" width="500" height="300" rx="20" fill="white" stroke="#e5e7eb" stroke-width="2"/>
  </g>
  
  <!-- Заголовок -->
  <text x="300" y="100" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" font-weight="bold" fill="#1f2937">
    {topic_short}
  </text>
  
  <!-- Иконка -->
  <circle cx="300" cy="200" r="40" fill="#4f46e5" opacity="0.8"/>
  <text x="300" y="205" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="white">?</text>
  
  <!-- Подпись -->
  <text x="300" y="270" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#6b7280">
    Инфографика по теме
  </text>
  
  <!-- Декоративные элементы -->
  <circle cx="100" cy="100" r="15" fill="#10b981" opacity="0.6"/>
  <circle cx="500" cy="120" r="12" fill="#f59e0b" opacity="0.6"/>
  <circle cx="80" cy="280" r="10" fill="#ef4444" opacity="0.6"/>
  <circle cx="520" cy="260" r="8" fill="#8b5cf6" opacity="0.6"/>
</svg>'''

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

    def debug_infographic_generation(self, topic: str):
        """Отладочный метод для тестирования генерации инфографики"""
        print(f"🔧 DEBUG: Генерация инфографики для: {topic}")
        
        test_prompt = f"Создай простую SVG инфографику на тему: {topic}"
        
        response = self._query_llm_api(
            prompt=test_prompt,
            context="",
            subject="general", 
            system_prompt="Верни только SVG код без пояснений",
            max_tokens=1000,
            is_svg=True
        )
        
        print(f"🔧 DEBUG: Ответ LLM: {response}")
        
        if response:
            svg_code = self._extract_svg_code(response)
            print(f"🔧 DEBUG: Извлеченный SVG: {svg_code[:200]}...")
            
            return svg_code
        
        return None

# Создаем глобальный экземпляр для использования в других модулях
llm_integration = LLMIntegration()

def get_llm_instance() -> LLMIntegration:
    """Возвращает глобальный экземпляр LLMIntegration"""
    return llm_integration

if __name__ == "__main__":
    # Тестирование модуля
    llm = LLMIntegration()
    
    print("🔧 Тестирование улучшенного LLM модуля...")
    
    # Тестирование генерации инфографики
    test_topic = "Статистика включает описательную статистику и индуктивную статистику"
    print(f"\n🔄 Генерация инфографики для: {test_topic}")
    infographic_result = llm.generate_infographic(test_topic)
    
    if infographic_result["success"]:
        print("✅ Инфографика успешно сгенерирована!")
        print(f"📊 SVG код (первые 200 символов): {infographic_result['svg_code'][:200]}...")
    else:
        print("❌ Не удалось сгенерировать инфографику")
        print(f"📊 Использован fallback")
    
    print("\n🎉 Тестирование завершено!")
