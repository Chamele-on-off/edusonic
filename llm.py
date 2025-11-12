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

    def _parse_concepts_from_response(self, response: str) -> Dict:
        """Парсит концепты из ответа LLM - ИСПРАВЛЕННАЯ ВЕРСИЯ"""
        concepts = {
            "main_concept": "",
            "aspects": []
        }
        
        try:
            # Проверяем, что ответ не пустой и содержит нужные маркеры
            if not response or not any(marker in response for marker in ['ЦЕНТРАЛЬНОЕ ПОНЯТИЕ', 'АСПЕКТ']):
                print("❌ Ответ LLM не содержит нужных маркеров")
                return concepts
            
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                
                # Ищем центральное понятие
                if line.startswith('ЦЕНТРАЛЬНОЕ ПОНЯТИЕ:'):
                    main_concept = line.replace('ЦЕНТРАЛЬНОЕ ПОНЯТИЕ:', '').strip()
                    # Убираем лишние слова, оставляем только основное понятие
                    main_concept = re.sub(r'\s+АСПЕКТ\d+.*', '', main_concept)
                    concepts["main_concept"] = main_concept
                
                # Ищем аспекты и извлекаем только содержание аспекта (без метки АСПЕКТ1:)
                elif line.startswith('АСПЕКТ1:'):
                    aspect = line.replace('АСПЕКТ1:', '').strip()
                    if aspect and aspect != "Аспект 1":
                        concepts["aspects"].append(aspect)
                elif line.startswith('АСПЕКТ2:'):
                    aspect = line.replace('АСПЕКТ2:', '').strip()
                    if aspect and aspect != "Аспект 2":
                        concepts["aspects"].append(aspect)
                elif line.startswith('АСПЕКТ3:'):
                    aspect = line.replace('АСПЕКТ3:', '').strip()
                    if aspect and aspect != "Аспект 3":
                        concepts["aspects"].append(aspect)
                elif line.startswith('АСПЕКТ4:'):
                    aspect = line.replace('АСПЕКТ4:', '').strip()
                    if aspect and aspect != "Аспект 4":
                        concepts["aspects"].append(aspect)
            
            # Если центральное понятие содержит аспекты, очищаем его
            if concepts["main_concept"] and any(marker in concepts["main_concept"] for marker in ['АСПЕКТ1', 'АСПЕКТ2', 'АСПЕКТ3', 'АСПЕКТ4']):
                # Оставляем только текст до первого упоминания АСПЕКТ
                concepts["main_concept"] = re.split(r'\s+АСПЕКТ\d+', concepts["main_concept"])[0].strip()
            
            print(f"🔍 Извлеченные концепты: main='{concepts['main_concept']}', aspects={concepts['aspects']}")
            
        except Exception as e:
            print(f"❌ Ошибка парсинга концептов: {e}")
        
        return concepts

    def _generate_mermaid_from_concepts(self, concepts: Dict) -> str:
        """Генерация Mermaid из концептов - ИСПРАВЛЕННАЯ ВЕРСИЯ"""
        main_concept = concepts["main_concept"] or "Основное понятие"
        aspects = concepts["aspects"][:4]  # Берем до 4 аспектов
        
        # Если аспектов нет или они пустые, возвращаем простую диаграмму
        if not aspects:
            aspects = ["Аспект 1", "Аспект 2", "Аспект 3"]
        
        mermaid_lines = ['flowchart TD']
        
        # Центральное понятие (верхний блок)
        mermaid_lines.append(f'    A["{main_concept}"]')
        
        # Аспекты с нумерацией и содержанием (нижние блоки)
        for i, aspect in enumerate(aspects):
            node_id = chr(66 + i)  # B, C, D, E
            # Форматируем аспект с нумерацией и ограничиваем длину
            formatted_aspect = f"{i+1}) {aspect}"
            mermaid_lines.append(f'    A --> {node_id}["{formatted_aspect}"]')
        
        # Минимальные стили с увеличенными размерами блоков
        mermaid_lines.extend([
            '',
            '    style A fill:#4263EB,color:#fff,stroke-width:2px',
            '    style B fill:#4cc9f0,color:#333,stroke-width:2px',
            '    style C fill:#3a0ca3,color:#fff,stroke-width:2px',
            '    style D fill:#f72585,color:#fff,stroke-width:2px',
            '    style E fill:#7209b7,color:#fff,stroke-width:2px'
        ])
        
        return '\n'.join(mermaid_lines)

    def _generate_svg_from_concepts(self, concepts: Dict) -> str:
        """Генерация SVG из концептов - ИСПРАВЛЕННАЯ ВЕРСИЯ"""
        main_concept = concepts["main_concept"][:20] if concepts["main_concept"] else "Основное понятие"
        aspects = [aspect[:30] for aspect in concepts["aspects"][:4]]  # Ограничиваем длину
        
        # Если аспектов нет или они пустые, создаем простые аспекты
        if not aspects:
            aspects = ["Аспект 1", "Аспект 2", "Аспект 3"]
        
        # Создаем простой SVG с центральным понятием и аспектами
        aspect_positions = [
            (95, 170),   # Аспект 1
            (200, 170),  # Аспект 2  
            (305, 170),  # Аспект 3
            (200, 220)   # Аспект 4 (если есть)
        ]
        
        aspect_elements = []
        aspect_lines = []
        
        for i, aspect in enumerate(aspects):
            if i < len(aspect_positions):
                x, y = aspect_positions[i]
                
                # Форматируем аспект с нумерацией
                formatted_aspect = f"{i+1}) {aspect}"
                
                # Определяем цвет в зависимости от позиции
                colors = ["#4cc9f0", "#3a0ca3", "#f72585", "#7209b7"]
                color = colors[i] if i < len(colors) else "#4cc9f0"
                text_color = "white" if i in [1, 2, 3] else "#333"  # Белый текст для темных фонов
                
                # Увеличиваем высоту блока для текста
                rect_height = 45 if len(formatted_aspect) > 30 else 35
                text_y = y + 15 if len(formatted_aspect) > 30 else y + 18
                font_size = "9" if len(formatted_aspect) > 30 else "10"
                
                # Добавляем прямоугольник аспекта
                aspect_elements.append(f'''
                <rect x="{x-50}" y="{y}" width="100" height="{rect_height}" fill="{color}" rx="8"/>
                <text x="{x}" y="{text_y}" text-anchor="middle" font-family="Arial" font-size="{font_size}" fill="{text_color}">
                    {formatted_aspect}
                </text>
                ''')
                
                # Добавляем линию соединения
                aspect_lines.append(f'<line x1="200" y1="125" x2="{x}" y2="{y}" stroke="#333" stroke-width="2"/>')
        
        return f'''
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300">
            <defs>
                <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#f8f9fa"/>
                    <stop offset="100%" stop-color="#e9ecef"/>
                </linearGradient>
            </defs>
            
            <!-- Фон -->
            <rect x="10" y="10" width="380" height="280" fill="url(#bg)" stroke="#dee2e6" stroke-width="2" rx="15"/>
            
            <!-- Заголовок -->
            <text x="200" y="35" text-anchor="middle" font-family="Arial" font-size="14" fill="#333" font-weight="bold">
                Концептуальная карта
            </text>
            
            <!-- Основное понятие -->
            <rect x="80" y="70" width="240" height="50" fill="#4263EB" rx="10"/>
            <text x="200" y="100" text-anchor="middle" font-family="Arial" font-size="12" fill="white" font-weight="bold">
                {main_concept}
            </text>
            
            <!-- Линии соединения -->
            {''.join(aspect_lines)}
            
            <!-- Аспекты -->
            {''.join(aspect_elements)}
        </svg>
        '''.strip()

    def _generate_fallback_visualization(self, topic: str) -> dict:
        """Fallback визуализация - ТЕМАТИЧЕСКАЯ ВЕРСИЯ"""
        # Создаем тематические аспекты на основе темы
        topic_lower = topic.lower()
        
        if any(word in topic_lower for word in ['глобальн', 'проблем']):
            main_concept = "Глобальные проблемы"
            aspects = [
                "Экологические проблемы - загрязнение окружающей среды",
                "Демографические проблемы - рост населения Земли", 
                "Продовольственные проблемы - нехватка продуктов питания",
                "Энергетические проблемы - истощение ресурсов"
            ]
        elif any(word in topic_lower for word in ['экономич', 'рынок', 'финанс']):
            main_concept = "Экономическая сфера"
            aspects = [
                "Производство - создание товаров и услуг",
                "Распределение - движение товаров к потребителям", 
                "Обмен - купля-продажа товаров",
                "Потребление - использование товаров"
            ]
        elif any(word in topic_lower for word in ['общество', 'социал']):
            main_concept = "Общество"
            aspects = [
                "Социальная структура - взаимосвязанные элементы",
                "Социальные институты - формы организации", 
                "Культурные нормы - правила поведения",
                "Социальные взаимодействия - связи между людьми"
            ]
        elif any(word in topic_lower for word in ['политик', 'государств', 'власт']):
            main_concept = "Политическая сфера"
            aspects = [
                "Государство - политическая организация",
                "Политические партии - объединения граждан", 
                "Право - система норм и правил",
                "Власть - способность влиять на других"
            ]
        else:
            # Общий fallback
            main_concept = topic[:20] + "..." if len(topic) > 20 else topic
            aspects = [
                "Основной аспект - ключевой элемент темы",
                "Второстепенный аспект - дополнительный элемент", 
                "Структурный аспект - организация системы",
                "Функциональный аспект - назначение и работа"
            ]
        
        concepts = {
            "main_concept": main_concept,
            "aspects": aspects
        }
        
        return {
            "mermaid_code": self._generate_mermaid_from_concepts(concepts),
            "svg_code": self._generate_svg_from_concepts(concepts),
            "topic": topic,
            "concepts": concepts,
            "success": False
        }

    def check_visualization_need(self, text: str) -> bool:
        """Проверяет, нужна ли визуализация для данного текста"""
        if not text or len(text.strip()) < 10:
            return False
            
        text_lower = text.lower()
        
        # Ключевые слова, указывающие на необходимость визуализации
        visualization_keywords = [
            'структура', 'схема', 'диаграмма', 'процесс', 
            'алгоритм', 'иерархия', 'взаимосвязь', 'соотношение',
            'таблица', 'классификация', 'этапы', 'стадии', 'система',
            'модель', 'цепочка', 'последовательность'
        ]
        
        # Структурные индикаторы
        structure_indicators = [
            'состоит из', 'включает в себя', 'делится на', 'подразделяется',
            'можно разделить', 'выделяют', 'различают'
        ]
        
        # Проверяем наличие ключевых слов
        has_keywords = any(keyword in text_lower for keyword in visualization_keywords)
        
        # Проверяем наличие структурных индикаторов
        has_structure = any(indicator in text_lower for indicator in structure_indicators)
        
        # Проверяем длину текста (достаточно информативный)
        is_long_enough = len(text.split()) > 3
        
        return (has_keywords or has_structure) and is_long_enough

    def generate_visualization(self, topic: str, context: str = "") -> dict:
        """Генерация концептуальной карты на основе темы"""
        try:
            print(f"🎨 Генерация концептуальной карты для: {topic[:100]}...")
            
            # УВЕЛИЧИВАЕМ ТАЙМАУТ и токены для лучшего ответа
            prompt = f"""
На основе темы урока создай содержательную концептуальную карту с центральным понятием и развернутыми аспектами.

ТЕМА УРОКА: {topic}
КОНТЕКСТ: {context}

ТРЕБОВАНИЯ:
1. Выдели ОДНО центральное понятие (самый важный термин)
2. Определи 3-4 ключевых аспекта, которые непосредственно связаны с центральным понятием
3. Для каждого аспекта дай краткое пояснение через тире (что это такое)
4. Используй конкретные термины и их определения, соответствующие теме
5. Не добавляй лишних деталей или объяснений

Формат ответа должен быть СТРОГО таким:
ЦЕНТРАЛЬНОЕ ПОНЯТИЕ: [основное понятие]
АСПЕКТ1: [первый аспект] - [краткое пояснение]
АСПЕКТ2: [второй аспект] - [краткое пояснение] 
АСПЕКТ3: [третий аспект] - [краткое пояснение]
АСПЕКТ4: [четвертый аспект] - [краткое пояснение] (если есть)

Пример для "Глобальные проблемы человечества":
ЦЕНТРАЛЬНОЕ ПОНЯТИЕ: Глобальные проблемы
АСПЕКТ1: Экологические проблемы - загрязнение окружающей среды
АСПЕКТ2: Демографические проблемы - рост населения Земли
АСПЕКТ3: Продовольственные проблемы - нехватка продуктов питания
АСПЕКТ4: Энергетические проблемы - истощение ресурсов

Пример для "Экономическая сфера общества":
ЦЕНТРАЛЬНОЕ ПОНЯТИЕ: Экономическая сфера
АСПЕКТ1: Производство - создание товаров и услуг
АСПЕКТ2: Распределение - движение товаров к потребителям
АСПЕКТ3: Обмен - купля-продажа товаров и услуг
АСПЕКТ4: Потребление - использование товаров и услуг

Тема: {topic}

Верни ТОЛЬКО ответ в указанном формате без дополнительных комментариев.
"""
            
            response = self._query_llm_api(
                prompt=prompt,
                context=context,
                subject="general",
                system_prompt="""Ты создаешь содержательные концептуальные карты для обучения.
                Для каждого аспекта обязательно давай краткое пояснение через тире.
                Используй конкретные термины и их определения, соответствующие теме урока.
                Сосредоточься только на основном понятии и непосредственно связанных с ним аспектах.
                Не используй общие фразы вроде "Аспект 1", "Аспект 2" - всегда давай конкретные названия.
                Используй строго указанный формат ответа.""",
                max_tokens=500  # Увеличиваем количество токенов
            )
            
            if response:
                print(f"🔧 Получен ответ от LLM: {response[:200]}...")
                
                # Парсим ответ
                concepts = self._parse_concepts_from_response(response)
                
                # Проверяем, что парсинг прошел успешно и есть содержательные данные
                has_valid_concepts = (
                    concepts["main_concept"] and 
                    concepts["aspects"] and 
                    len(concepts["aspects"]) > 0 and
                    not all(aspect in ["Аспект 1", "Аспект 2", "Аспект 3", "Аспект 4"] for aspect in concepts["aspects"])
                )
                
                if has_valid_concepts:
                    print(f"✅ Извлечены валидные концепты: {concepts}")
                    
                    # Генерируем визуализацию
                    mermaid_code = self._generate_mermaid_from_concepts(concepts)
                    svg_code = self._generate_svg_from_concepts(concepts)
                    
                    return {
                        "mermaid_code": mermaid_code,
                        "svg_code": svg_code,
                        "topic": topic,
                        "concepts": concepts,
                        "success": True
                    }
                else:
                    print("❌ Ответ LLM не содержит валидных концептов, использую тематический fallback")
            
            # Fallback - если не удалось получить валидный ответ от LLM
            print("🔄 Использую тематический fallback")
            return self._generate_fallback_visualization(topic)
            
        except Exception as e:
            print(f"❌ Ошибка генерации визуализации: {e}")
            return self._generate_fallback_visualization(topic)

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
    
    # Тестирование генерации визуализации для разных тем
    test_topics = [
        "Глобальные проблемы человечества",
        "Экономическая сфера общества", 
        "Политическая система государства"
    ]
    
    for topic in test_topics:
        print(f"\n🔄 Генерация визуализации для: {topic}")
        viz_result = llm.generate_visualization(topic)
        
        if viz_result["success"]:
            print("✅ Визуализации успешно сгенерированы!")
            print(f"📊 Основное понятие: {viz_result['concepts'].get('main_concept', 'unknown')}")
            print(f"📊 Аспекты: {viz_result['concepts'].get('aspects', [])}")
        else:
            print("❌ Использован fallback")
            print(f"📊 Основное понятие: {viz_result['concepts'].get('main_concept', 'unknown')}")
            print(f"📊 Аспекты: {viz_result['concepts'].get('aspects', [])}")
    
    print("\n🎉 Тестирование завершено!")