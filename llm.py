import requests
import json
from typing import Optional, Dict, Callable
from pathlib import Path
import time
from config import get_api_key, load_config, get_model_config, get_llm_priority, set_llm_priority
import re
from local_llm_manager import get_llm_manager
import queue
from key_manager import get_key_manager

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
                            processed_answer = self._clean_llm_response(answer)
                            
                            print(f"✅ [LLM] Ответ от OpenRouter получен: {processed_answer[:100]}...")
                            
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

    def clean_and_validate_mermaid_code(self, code: str) -> str:
        """Очистка и валидация Mermaid кода"""
        if not code:
            return ""
        
        # Удаляем markdown обратные кавычки и все что между ними
        code = re.sub(r'```[\s\S]*?```', '', code)
        code = re.sub(r'`[^`]*`', '', code)
        code = re.sub(r'[\*\#\-\_]{2,}', '', code)
        
        # Удаляем комментарии Mermaid
        code = re.sub(r'%%.*', '', code)
        
        # Удаляем лишние пробелы и пустые строки
        code = '\n'.join([line.strip() for line in code.split('\n') if line.strip()])
        
        # Проверяем базовый синтаксис
        valid_starts = ['graph', 'flowchart', 'sequenceDiagram', 'classDiagram', 'stateDiagram', 'pie', 'gantt']
        
        # Если код не начинается с правильного типа, добавляем flowchart по умолчанию
        if not any(code.strip().startswith(start) for start in valid_starts):
            code = 'flowchart TD\n' + code
        
        # Убедимся, что есть базовые элементы
        lines = code.split('\n')
        if len(lines) < 2 or ('-->' not in code and '->' not in code):
            # Добавляем простую структуру если нет связей
            if len(lines) > 1:
                code = lines[0] + '\n' + 'A["Элемент A"] --> B["Элемент B"]'
            else:
                code = 'flowchart TD\nA["Элемент A"] --> B["Элемент B"]'
        
        return code.strip()

    def generate_mermaid_diagram(self, topic: str, context: str = "") -> str:
        """Генерация Mermaid диаграммы через LLM"""
        prompt = f"""
        Создай простую и понятную Mermaid.js диаграмму для объяснения темы: "{topic}".
        
        Контекст урока: {context}
        
        ТРЕБОВАНИЯ К ДИАГРАММЕ:
        1. Используй ТОЛЬКО корректный синтаксис Mermaid
        2. Максимум 8-10 элементов для наглядности
        3. Простые прямоугольники и стрелки
        4. Русские подписи в кавычках
        5. Логическая структура от общего к частному
        6. Избегай сложных конструкций
        
        ПРИМЕРЫ КОРРЕКТНОГО СИНТАКСИСА:
        flowchart TD
            A["Общее понятие"] --> B["Частный случай 1"]
            A --> C["Частный случай 2"]
            B --> D["Пример"]
            C --> D

        graph TD
            A[Старт] --> B[Процесс 1]
            B --> C[Процесс 2]
            C --> D[Результат]

        Тема для диаграмма: {topic}
        
        Верни ТОЛЬКО код Mermaid без каких-либо пояснений.
        Начни сразу с объявления типа диаграммы.
        """
        
        try:
            response = self._query_llm_api(
                prompt=prompt,
                context="",
                subject="general",
                system_prompt="""Ты - эксперт по созданию образовательных диаграмм. 
                Твоя задача - создавать ПРОСТЫЕ и ПОНЯТНЫЕ Mermaid диаграммы.
                ВАЖНЫЕ ПРАВИЛА:
                1. Всегда используй корректный синтаксис Mermaid
                2. Максимальная простота и наглядность
                3. Русские подписи в двойных кавычках
                4. Логические связи между элементами
                5. Избегай сложных конструкций
                
                Если не уверен в синтаксисе - используй простейшую структуру.""",
                max_tokens=500
            )
            
            if response:
                # Очищаем и проверяем синтаксис
                cleaned_code = self.clean_and_validate_mermaid_code(response)
                print(f"✅ Сгенерирован Mermaid код для: {topic}")
                return cleaned_code
            
        except Exception as e:
            print(f"❌ Ошибка генерации Mermaid кода: {e}")
        
        # Fallback - простая диаграмма по умолчанию
        return f'''flowchart TD
    A["{topic}"] --> B["Основной аспект 1"]
    A --> C["Основной аспект 2"]
    B --> D["Пример или свойство"]
    C --> D'''

    def generate_svg_diagram(self, topic: str, context: str = "") -> str:
        """Генерация простого SVG через LLM"""
        prompt = f"""
        Создай простой SVG код для визуализации: "{topic}".
        
        Контекст: {context}
        
        Используй только базовые элементы:
        - <rect> для прямоугольников и блоков
        - <circle> для кругов и узлов
        - <line> для линий и связей
        - <text> для текста и подписей
        - <path> для сложных форм
        
        Требования:
        - Размер: 400x300
        - Простая и понятная схема
        - Русские подписи
        - Минималистичный дизайн
        - Логическая структура
        - Цвета для различия элементов

        Верни ТОЛЬКО SVG код без пояснений.
        """
        
        try:
            svg_code = self._query_llm_api(
                prompt=prompt,
                context="",
                subject="general",
                system_prompt="Ты создаешь простые SVG схемы для образования. Используй минималистичный дизайн и четкую структуру.",
                max_tokens=1000
            )
            
            if svg_code:
                # Очищаем SVG код
                svg_code = re.sub(r'```(xml|svg)?\s*', '', svg_code)
                svg_code = re.sub(r'```\s*', '', svg_code)
                svg_code = svg_code.strip()
                
                # Проверяем валидность SVG
                if svg_code.startswith('<svg') and svg_code.endswith('</svg>'):
                    print(f"✅ Сгенерирован SVG код для: {topic}")
                    return svg_code
            
        except Exception as e:
            print(f"❌ Ошибка генерации SVG кода: {e}")
        
        return ""

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
        """Генерация обеих типов визуализаций (Mermaid и SVG)"""
        try:
            print(f"🎨 Генерация визуализаций для: {topic}")
            
            # Генерируем Mermaid диаграмму
            mermaid_code = self.generate_mermaid_diagram(topic, context)
            
            # Генерируем SVG схему
            svg_code = self.generate_svg_diagram(topic, context)
            
            result = {
                "mermaid_code": mermaid_code,
                "svg_code": svg_code,
                "topic": topic,
                "success": bool(mermaid_code or svg_code)
            }
            
            if result["success"]:
                print(f"✅ Визуализации сгенерированы для: {topic}")
            else:
                print(f"❌ Не удалось сгенерировать визуализации для: {topic}")
                
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
    
    print("🔧 Тестирование LLM модуля...")
    
    # Тестирование подключения
    connection_ok = llm.test_connection()
    print(f"📡 Подключение к API: {'✅ Успешно' if connection_ok else '❌ Ошибка'}")
    
    # Тестирование статуса моделей
    status = llm.get_llm_status()
    print(f"🔧 Статус моделей: {status}")
    
    # Тестовый запрос
    test_question = "Объясни, что такое фотосинтез"
    response = llm.query(test_question, subject="биология")
    
    print("\n🧪 Тестовый ответ:")
    print(response)
    
    # Тестирование проверки необходимости визуализации
    needs_viz = llm.check_visualization_need(test_question)
    print(f"\n🎨 Нужна ли визуализация для вопроса: {'✅ Да' if needs_viz else '❌ Нет'}")
    
    # Тестирование генерации визуализации
    if needs_viz:
        print("\n🔄 Генерация визуализации...")
        viz_result = llm.generate_visualization("Процесс фотосинтеза", "Фотосинтез - это процесс преобразования света в химическую энергию")
        
        if viz_result["success"]:
            print("✅ Визуализации успешно сгенерированы!")
            if viz_result["mermaid_code"]:
                print(f"📊 Mermaid код (первые 100 символов): {viz_result['mermaid_code'][:100]}...")
            if viz_result["svg_code"]:
                print(f"🖼️ SVG код (первые 100 символов): {viz_result['svg_code'][:100]}...")
        else:
            print("❌ Не удалось сгенерировать визуализации")
    
    # Тестирование статистики кэша
    cache_stats = llm.get_cache_stats()
    print(f"\n💾 Статистика кэша:")
    print(f"   Всего записей: {cache_stats['total_entries']}")
    print(f"   Предметы: {cache_stats['subjects']}")
    
    print("\n🎉 Тестирование завершено!")