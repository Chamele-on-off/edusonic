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

# 🔥 ИМПОРТ ДЛЯ ТЕХНИЧЕСКИХ ПРЕДМЕТОВ
try:
    from technical_subjects import (
        is_technical_subject, 
        clean_text_for_speech_technical,
        contains_formulas,
        get_subject_type
    )
    TECHNICAL_SUPPORT_ENABLED = True
except ImportError:
    TECHNICAL_SUPPORT_ENABLED = False
    print("⚠️ technical_subjects.py не найден, техническая поддержка отключена")

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
                 model: str = "google/gemma-3-12b-it:free"):
        
        config = load_config()
        openrouter_config = get_model_config("openrouter")
        
        # 🔥 ИСПРАВЛЕНИЕ: Используем менеджер ключей вместо одного ключа
        self.key_manager = get_key_manager()  # Ключевое изменение!
        self.api_url = api_url or openrouter_config.get("api_url", "https://openrouter.ai/api/v1/chat/completions")
        
        # Получаем модель из менеджера ключей, а не из конфига напрямую
        self.model = self.key_manager.model or model or openrouter_config.get("model", "meta-llama/llama-3.3-70b-instruct:free")
        
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
        print(f"🔧 [LLM] Текущая модель OpenRouter: {self.model}")
        
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

    def _process_llm_response(self, response: str, is_svg: bool = False, subject: str = "") -> str:
        """🔥 ОБНОВЛЕННЫЙ: Обработка ответа LLM в зависимости от типа и предмета"""
        if is_svg:
            # Для SVG - минимальная очистка, сохраняем структуру
            print(f"🔧 [LLM] Обработка SVG ответа (без очистки речи)")
            return response.strip()
        else:
            # 🔥 КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Умная очистка в зависимости от предмета
            if TECHNICAL_SUPPORT_ENABLED and subject:
                # Используем умную очистку для технических предметов
                cleaned = clean_text_for_speech_technical(response, subject)
                print(f"🔧 [LLM] Использована умная очистка для предмета: {subject}")
                # Дополнительная очистка от форматирования
                return self._clean_llm_response(cleaned)
            else:
                # Стандартная очистка для гуманитарных предметов
                print(f"🔧 [LLM] Обработка текстового ответа (стандартная очистка)")
                clean1 = clean_text_for_speech(response)
                return self._clean_llm_response(clean1)

    def _query_llm_api(self, prompt: str, context: str = "", subject: str = "", 
                       system_prompt: str = "", max_tokens: int = 1000, 
                       room_id: str = "default", callback: Callable = None,
                       is_svg: bool = False) -> Optional[str]:
        """🔥 ОБНОВЛЕННЫЙ: УМНАЯ ЛОГИКА ПРИОРИТЕТОВ С ПОДДЕРЖКОЙ ТЕХНИЧЕСКИХ ПРЕДМЕТОВ"""
        
        print(f"🔧 [LLM] Запрос с приоритетом '{self.priority_mode}': {prompt[:100]}...")
        print(f"🔧 [LLM] Тип запроса: {'SVG' if is_svg else 'Текст'}")
        print(f"🔧 [LLM] Используемая модель OpenRouter: {self.model}")
        
        # 🔥 ДОБАВЛЯЕМ ИНФОРМАЦИЮ О ПРЕДМЕТЕ В ЗАПРОСЫ
        is_technical = False
        if TECHNICAL_SUPPORT_ENABLED and subject:
            is_technical = is_technical_subject(subject)
            
            # Адаптируем системный промпт для технических предметов
            if is_technical and not is_svg:
                # Для технических предметов добавляем инструкции по формулам
                if "system" in system_prompt.lower():
                    system_prompt += "\n\nИСПОЛЬЗУЙ математические и научные обозначения: формулы, символы, единицы измерения."
                    system_prompt += "\nСОХРАНЯЙ формулы в читаемом формате (например, E=mc², F=ma, H₂O)."
                    system_prompt += "\nОБЪЯСНЯЙ сложные концепции с помощью примеров и вычислений."
                print(f"🔧 [LLM] Технический предмет: {subject}")
        
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
                        # 🔥 ОБРАБАТЫВАЕМ ОТВЕТ С УЧЕТОМ ПРЕДМЕТА
                        processed_result = self._process_llm_response(result, is_svg, subject)
                        callback(processed_result, room_id)
                    else:
                        # 🔥 ВСЕГДА ВОЗВРАЩАЕМ FALLBACK ДЛЯ АСИНХРОННЫХ ЗАПРОСОВ
                        fallback = self._get_fallback_response(prompt, subject)
                        callback(fallback, room_id)
                except Exception as e:
                    print(f"❌ [LLM] Ошибка асинхронного запроса: {e}")
                    # 🔥 ВСЕГДА ВОЗВРАЩАЕМ FALLBACK ЧТОБЫ НЕ БЛОКИРОВАТЬ СИСТЕМУ
                    fallback = self._get_fallback_response(prompt, subject)
                    callback(fallback, room_id)
            
            # Запускаем в отдельном потоке
            thread = threading.Thread(target=async_query, daemon=True)
            thread.start()
            return None
        else:
            # Синхронный режим - с таймаутом и fallback
            try:
                result = self._execute_llm_query(
                    prompt, context, subject, system_prompt,
                    max_tokens, room_id, is_svg
                )
                if result:
                    # 🔥 ОБРАБАТЫВАЕМ ОТВЕТ С УЧЕТОМ ПРЕДМЕТА
                    return self._process_llm_response(result, is_svg, subject)
                else:
                    return self._get_fallback_response(prompt, subject)
            except Exception as e:
                print(f"❌ [LLM] Ошибка синхронного запроса: {e}")
                return self._get_fallback_response(prompt, subject)
    
    def _execute_llm_query(self, prompt: str, context: str, subject: str,
                          system_prompt: str, max_tokens: int,
                          room_id: str, is_svg: bool = False) -> Optional[str]:
        """Выполнение LLM запроса с безопасным fallback"""
        
        print(f"🔧 [LLM] Запрос: {prompt[:100]}...")
        print(f"🔧 [LLM] Приоритет: {self.priority_mode}")
        print(f"🔧 [LLM] Используемая модель OpenRouter: {self.model}")
        
        # Проверяем доступность локальной модели
        local_available = self.llm_manager.local_llm.is_available()
        print(f"🔧 [LLM] Локальная модель доступна: {local_available}")
        
        # 🔥 АДАПТИРУЕМ ПРОМПТ ДЛЯ ТЕХНИЧЕСКИХ ПРЕДМЕТОВ
        adapted_system_prompt = system_prompt
        if TECHNICAL_SUPPORT_ENABLED and subject and is_technical_subject(subject):
            if not adapted_system_prompt:
                adapted_system_prompt = "Ты - профессиональный учитель по техническим предметам."
            
            # Добавляем технические инструкции
            adapted_system_prompt += "\n\n🔥 ДЛЯ ТЕХНИЧЕСКИХ ПРЕДМЕТОВ:"
            adapted_system_prompt += "\n- Используй математические и научные обозначения"
            adapted_system_prompt += "\n- Сохраняй формулы в правильном формате"
            adapted_system_prompt += "\n- Объясняй пошагово сложные концепции"
            adapted_system_prompt += "\n- Приводи конкретные вычисления и примеры"
            print(f"🔧 [LLM] Использован адаптированный промпт для технического предмета")
        
        if self.priority_mode == "local_only":
            print("🔧 [LLM] Режим: только локальная модель")
            if local_available:
                return self._handle_local_request_safe(prompt, adapted_system_prompt, max_tokens, is_svg)
            else:
                print("❌ [LLM] Локальная модель недоступна в режиме local_only")
                return None
                
        elif self.priority_mode == "openrouter_only":
            print("🔧 [LLM] Режим: только OpenRouter")
            response = self._handle_openrouter_request_safe(prompt, context, subject, adapted_system_prompt, max_tokens, is_svg)
            if response:
                return response
            else:
                print("⚠️ [LLM] OpenRouter не ответил, но режим 'openrouter_only' - возвращаем None")
                return None
                
        elif self.priority_mode == "openrouter_first":
            print("🔧 [LLM] Режим: сначала OpenRouter")
            response = self._handle_openrouter_request_safe(prompt, context, subject, adapted_system_prompt, max_tokens, is_svg)
            if response:
                print("✅ [LLM] Использую ответ от OpenRouter")
                return response
            
            print("⚠️ [LLM] OpenRouter не ответил, пробую локальную модель...")
            if local_available:
                response = self._handle_local_request_safe(prompt, adapted_system_prompt, max_tokens, is_svg)
                if response:
                    print("✅ [LLM] Использую ответ от локальной модели")
                    return response
            
            print("❌ [LLM] Ни одна модель не ответила")
            return None
            
        else:  # local_first (по умолчанию)
            print("🔧 [LLM] Режим: сначала локальная модель")
            if local_available:
                response = self._handle_local_request_safe(prompt, adapted_system_prompt, max_tokens, is_svg)
                if response:
                    print("✅ [LLM] Использую ответ от локальной модели")
                    return response
            
            print("⚠️ [LLM] Локальная модель не ответила, пробую OpenRouter...")
            response = self._handle_openrouter_request_safe(prompt, context, subject, adapted_system_prompt, max_tokens, is_svg)
            if response:
                print("✅ [LLM] Использую ответ от OpenRouter")
                return response
            
            print("❌ [LLM] Ни одна модель не ответила")
            return None
    
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
                    # 🔥 НЕ ОБРАБАТЫВАЕМ ТУТ - обработка будет в вызывающем коде с учетом предмета
                    result_queue.put(response)
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
        """Безопасная обработка OpenRouter с улучшенным fallback"""
        try:
            # 🔥 БЕЗОПАСНОЕ ПОЛУЧЕНИЕ КЛЮЧА
            try:
                api_key = self.key_manager.get_next_key()
                print(f"🔧 [LLM] Пробую OpenRouter: ключ {api_key[:8]}..., модель: {self.model}")
            except Exception as e:
                print(f"❌ [LLM] Нет доступных ключей OpenRouter: {e}")
                return None  # Fallback на локальную модель
            
            # 🔥 ПРОВЕРЯЕМ КЛЮЧ ПЕРЕД ИСПОЛЬЗОВАНИЕМ
            if not api_key or len(api_key.strip()) < 20:
                print(f"❌ [LLM] Неверный ключ API (слишком короткий)")
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
                    "HTTP-Referer": "https://ai.zindaki-edu.ru",
                    "X-Title": "https://ai.zindaki-edu.ru"
                }
                
                # 🔥 УПРОЩЕННЫЙ ПРОМПТ
                final_system_prompt = system_prompt or f"Ты - профессиональный учитель."
                
                messages = []
                
                if final_system_prompt:
                    messages.append({"role": "system", "content": final_system_prompt})
                
                if context and context.strip():
                    messages.append({"role": "user", "content": f"Контекст: {context}"})
                
                messages.append({"role": "user", "content": prompt})
        
                data = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": max_tokens,
                    "stream": False
                }
        
                print(f"🔧 [LLM] Отправляю запрос к модели OpenRouter: {self.model}")
                
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=data,
                    timeout=15  # Уменьшаем таймаут для быстрого fallback
                )
                
                self.last_request_time = time.time()
                
                if response.status_code == 200:
                    # 🔥 ЗАПИСЫВАЕМ ИСПОЛЬЗОВАНИЕ КЛЮЧА
                    self.key_manager.record_usage(api_key)
                    
                    result = response.json()
                    
                    if 'choices' in result and len(result['choices']) > 0:
                        answer = result['choices'][0]['message']['content']
                        
                        # 🔥 НЕ ОБРАБАТЫВАЕМ ТУТ - обработка будет в вызывающем коде
                        print(f"✅ [LLM] Успешный ответ от модели: {self.model}")
                        return answer
                    else:
                        print("❌ [LLM] Неверный формат ответа от OpenRouter")
                        return None
                        
                elif response.status_code == 400:
                    print(f"❌ [LLM] Модель {self.model} не найдена или недоступна (ошибка 400)")
                    # 🔥 ОШИБКА МОДЕЛИ - возвращаем None для fallback
                    return None
                    
                elif response.status_code == 401:
                    print(f"❌ [LLM] Ошибка аутентификации (неверный ключ)")
                    # Пробуем деактивировать ключ
                    try:
                        # Находим имя ключа для деактивации
                        for key_data in self.key_manager.keys:
                            if key_data['key'] == api_key:
                                self.key_manager.toggle_key_active(key_data['name'], False)
                                break
                    except:
                        pass
                    return None
                    
                elif response.status_code == 404:
                    print(f"❌ [LLM] Модель {self.model} не существует (ошибка 404)")
                    return None
                    
                elif response.status_code == 429:
                    print(f"⏳ [LLM] Rate limit для модели {self.model}")
                    return None
                    
                else:
                    print(f"❌ [LLM] Ошибка {response.status_code} для модели {self.model}")
                    return None
                    
            except requests.exceptions.Timeout:
                print(f"⏰ [LLM] Таймаут для модели {self.model}")
                return None
                
            except Exception as e:
                print(f"❌ [LLM] Ошибка при запросе к модели {self.model}: {e}")
                return None
                
        except Exception as e:
            print(f"❌ [LLM] Критическая ошибка при работе с моделью {self.model}: {e}")
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
        
        # 🔥 УЛУЧШЕННЫЕ FALLBACK ОТВЕТЫ С УЧЕТОМ ТИПА ПРЕДМЕТА
        if TECHNICAL_SUPPORT_ENABLED and subject:
            subject_type = get_subject_type(subject)
            
            if subject_type == "technical":
                # Fallback для технических предметов
                if any(word in prompt_lower for word in ['формул', 'уравнен', 'вычисл', 'рассчит', 'задач']):
                    return "Для решения этой задачи нужно применить соответствующую формулу. Давайте вспомним основные формулы по этой теме."
                elif any(word in prompt_lower for word in ['докаж', 'теорем', 'свойств', 'закон']):
                    return "Это требует доказательства или объяснения закона. Давайте разберем этот вопрос подробнее."
                elif any(word in prompt_lower for word in ['график', 'диаграмм', 'схем', 'чертеж']):
                    return "Для понимания этой темы полезно построить график или схему. Давайте визуализируем."
            
            elif subject_type == "natural_science":
                # Fallback для естественных наук
                if any(word in prompt_lower for word in ['эксперимент', 'опыт', 'наблюден', 'исследован']):
                    return "Этот вопрос лучше понять через эксперимент или наблюдение. Давайте рассмотрим практический пример."
                elif any(word in prompt_lower for word in ['процесс', 'явление', 'систем', 'взаимодейств']):
                    return "Это природное явление или процесс. Давайте разберем его механизм по шагам."
        
        # Базовые fallback ответы
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
        return "Спасибо за вопрос! Я обрабатываю ваш запрос."

    def query(self, question: str, context: str = "", subject: str = "") -> str:
        """🔥 ОБНОВЛЕННЫЙ: Запрос к LLM API с поддержкой технических предметов"""
        if not question.strip():
            return ""
            
        question_lower = question.lower().strip()
        
        # Проверка кэша
        cache_key = f"{subject}_{question_lower}" if subject else question_lower
        if cache_key in self.cache:
            print(f"💾 Использую кэшированный ответ для: {question_lower}")
            return self.cache[cache_key]
        
        print(f"📨 Запрос к LLM: '{question}' (предмет: {subject})")
        print(f"🔧 [LLM] Использую модель: {self.model}")
        
        # 🔥 ВАЖНО: Используем асинхронный запрос с таймаутом
        start_time = time.time()
        
        try:
            # Запрос к реальному LLM
            llm_response = self._query_llm_api(question, context, subject)
            
            total_time = time.time() - start_time
            print(f"⏱️ Общее время обработки: {total_time:.2f}с")
            
            if llm_response and llm_response.strip():
                # 🔥 ОБРАБАТЫВАЕМ ОТВЕТ С УЧЕТОМ ПРЕДМЕТА
                processed_response = self._process_llm_response(llm_response, subject=subject)
                print(f"✅ Ответ получен: {processed_response[:100]}...")
                self.cache[cache_key] = processed_response
                self._save_cache()
                return processed_response
            
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
        """Установка модели LLM - перенаправляем в key_manager"""
        print(f"🔧 [LLM] Установка модели через key_manager: {model}")
        return self.key_manager.set_model(model)

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
            "effective_priority": self._get_effective_priority(),
            "current_model": self.model
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
            "current_model": self.model,
            "effective_priority": self._get_effective_priority(),
            "local_status": self.llm_manager.local_llm.get_status(),
            "local_url": self.llm_manager.local_llm.base_url,
            "cache_stats": self.get_cache_stats(),
            "key_manager_stats": {
                "total_keys": len(self.key_manager.keys),
                "active_keys": len([k for k in self.key_manager.keys if k.get('is_active', True)]),
                "daily_limit": self.key_manager.daily_limit,
                "extended_limit": self.key_manager.extended_limit,
                "reset_time": self.key_manager.reset_time
            },
            "technical_support_enabled": TECHNICAL_SUPPORT_ENABLED
        }

    def handle_llm_response(self, request_id: str, response: str, room_id: str):
        """Обработчик ответов от локальной LLM"""
        if request_id in self.pending_requests:
            callback, is_svg = self.pending_requests.pop(request_id)
            try:
                # 🔥 ОБРАБАТЫВАЕМ ОТВЕТ В ЗАВИСИМОСТИ ОТ ТИПА (предмет передается через room_id)
                subject = room_id  # В этом контексте room_id может содержать информацию о предмете
                processed_response = self._process_llm_response(response, is_svg, subject)
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

    def generate_infographic(self, topic: str, context: str = "", subject: str = "") -> dict:
        """🔥 ОБНОВЛЕННЫЙ: Генерация стильной инфографики с поддержкой технических предметов"""
        
        # 🔥 ОПРЕДЕЛЯЕМ ТИП ИНФОГРАФИКИ В ЗАВИСИМОСТИ ОТ ПРЕДМЕТА
        is_technical = False
        if TECHNICAL_SUPPORT_ENABLED and subject:
            is_technical = is_technical_subject(subject)
        
        if is_technical:
            # 🔥 СПЕЦИАЛЬНЫЙ ПРОМПТ ДЛЯ ТЕХНИЧЕСКИХ ПРЕДМЕТОВ
            prompt = self._create_technical_infographic_prompt(topic, subject)
        else:
            # 🔥 УЛУЧШЕННЫЙ ПРОМПТ ДЛЯ ГУМАНИТАРНЫХ ПРЕДМЕТОВ
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
            print(f"🎨 Генерация {'ТЕХНИЧЕСКОЙ ' if is_technical else ''}SVG инфографики для: {topic}")
            
            # 🔥 АДАПТИРУЕМ СИСТЕМНЫЙ ПРОМПТ ДЛЯ ТИПА ИНФОГРАФИКИ
            system_prompt = """Ты - креативный дизайнер образовательной инфографики. 
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

Верни ТОЛЬКО SVG код. НИЧЕГО БОЛЬШЕ."""
            
            if is_technical:
                system_prompt += """

🔥 ДОПОЛНИТЕЛЬНО ДЛЯ ТЕХНИЧЕСКИХ ПРЕДМЕТОВ:
- Используй математические и научные символы если уместно
- Добавляй схемы, диаграммы, графики
- Включай формулы и обозначения в SVG
- Делай инфографику информативной и точной"""

            response = self._query_llm_api(
                prompt=prompt,
                context=context,
                subject=subject,
                system_prompt=system_prompt,
                max_tokens=2500,
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
                        "type": "technical_infographic" if is_technical else "infographic",
                        "subject": subject
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
                "svg_code": self._create_fallback_infographic(topic, is_technical, subject),
                "topic": topic,
                "type": "technical_fallback" if is_technical else "fallback",
                "subject": subject
            }
            
        except Exception as e:
            print(f"❌ Ошибка генерации инфографики: {e}")
            return {
                "success": False,
                "svg_code": self._create_fallback_infographic(topic, is_technical, subject),
                "topic": topic,
                "type": "error_fallback",
                "subject": subject
            }

    def _create_technical_infographic_prompt(self, topic: str, subject: str) -> str:
        """Создает промпт для технической инфографики"""
        subject_lower = subject.lower()
        
        if 'математика' in subject_lower or 'алгебра' in subject_lower or 'геометрия' in subject_lower:
            return f"""
            Создай МАТЕМАТИЧЕСКУЮ инфографику в формате SVG на тему: "{topic}"

            ТРЕБОВАНИЯ:
            - Только чистый SVG код без пояснений
            - Включай математические формулы и символы
            - Используй схемы, графики, диаграммы
            - Добавь примеры вычислений если уместно
            - Четкая логическая структура

            ЭЛЕМЕНТЫ:
            - Математические символы: ∑, ∫, √, π, ∞, ≠, ≈, ≤, ≥
            - Формулы в SVG тексте
            - Координатные оси для графиков
            - Геометрические фигуры
            - Блок-схемы для алгоритмов

            ЦВЕТОВАЯ ПАЛИТРА (научный стиль):
            - Основные: #2563eb (синий для формул), #059669 (зеленый для результатов)
            - Акценты: #dc2626 (красный для важного), #7c3aed (фиолетовый для определений)
            - Фон: #f8fafc или #f1f5f9

            РАЗМЕР: 600x400 пикселей

            ТЕМА: "{topic}"

            Верни ТОЛЬКО SVG код.
            Код должен начинаться с <svg и заканчиваться </svg>.
            """
        
        elif 'физика' in subject_lower:
            return f"""
            Создай ФИЗИЧЕСКУЮ инфографику в формате SVG на тему: "{topic}"

            ТРЕБОВАНИЯ:
            - Только чистый SVG код без пояснений
            - Включай физические формулы и законы
            - Добавь схемы экспериментов или явлений
            - Укажи единицы измерения
            - Покажи взаимосвязи между понятиями

            ЭЛЕМЕНТЫ:
            - Физические формулы: F=ma, E=mc² и т.д.
            - Схемы экспериментов
            - Графики зависимостей
            - Стрелки для сил и направлений
            - Обозначения физических величин

            ЦВЕТОВАЯ ПАЛИТРА:
            - Силы: #ef4444 (красный)
            - Энергия: #f59e0b (желтый)
            - Движение: #3b82f6 (синий)
            - Статика: #10b981 (зеленый)
            - Фон: #f8fafc

            РАЗМЕР: 600x400 пикселей

            ТЕМА: "{topic}"

            Верни ТОЛЬКО SVG код.
            """
        
        elif 'химия' in subject_lower:
            return f"""
            Создай ХИМИЧЕСКУЮ инфографику в формате SVG на тему: "{topic}"

            ТРЕБОВАНИЯ:
            - Только чистый SVG код без пояснений
            - Включай химические формулы и уравнения
            - Покажи структурные формулы веществ
            - Добавь схемы реакций
            - Укажи условия реакций

            ЭЛЕМЕНТЫ:
            - Химические формулы: H₂O, CO₂, NaCl
            - Структурные формулы молекул
            - Уравнения реакций с коэффициентами
            - Схемы химических процессов
            - Обозначения состояний веществ (г, ж, т, р-р)

            ЦВЕТОВАЯ ПАЛИТРА:
            - Металлы: #f59e0b (золотой)
            - Неметаллы: #3b82f6 (синий)
            - Газы: #a5b4fc (лавандовый)
            - Жидкости: #06b6d4 (бирюзовый)
            - Реакции: #ef4444 (красный для экзотермических)
            - Фон: #f8fafc

            РАЗМЕР: 600x400 пикселей

            ТЕМА: "{topic}"

            Верни ТОЛЬКО SVG код.
            """
        
        elif 'биология' in subject_lower:
            return f"""
            Создай БИОЛОГИЧЕСКУЮ инфографику в формате SVG на тему: "{topic}"

            ТРЕБОВАНИЯ:
            - Только чистый SVG код без пояснений
            - Включай схемы биологических процессов
            - Покажи взаимосвязи в экосистемах
            - Добавь классификации организмов
            - Используй научные термины

            ЭЛЕМЕНТЫ:
            - Схемы клеток и органов
            - Циклы развития организмов
            - Пищевые цепи и сети
            - Классификационные диаграммы
            - Процессы: фотосинтез, дыхание и т.д.

            ЦВЕТОВАЯ ПАЛИТРА:
            - Растения: #10b981 (зеленый)
            - Животные: #f59e0b (коричневый/желтый)
            - Микроорганизмы: #8b5cf6 (фиолетовый)
            - Процессы: #06b6d4 (бирюзовый)
            - Фон: #f0fdf4 (светло-зеленый)

            РАЗМЕР: 600x400 пикселей

            ТЕМА: "{topic}"

            Верни ТОЛЬКО SVG код.
            """
        
        else:
            # Общий промпт для других технических предметов
            return f"""
            Создай НАУЧНУЮ инфографику в формате SVG на тему: "{topic}"

            ПРЕДМЕТ: {subject}

            ТРЕБОВАНИЯ:
            - Только чистый SVG код без пояснений
            - Научный стиль с точностью
            - Включай схемы, диаграммы, графики
            - Используй профессиональную терминологию
            - Логическая структура информации

            ЭЛЕМЕНТЫ:
            - Схемы и диаграммы
            - Графики данных
            - Классификационные таблицы
            - Процессные карты
            - Сравнительные диаграммы

            ЦВЕТОВАЯ ПАЛИТРА (научный стиль):
            - Основные: #2563eb, #059669, #7c3aed
            - Акценты: #dc2626, #f59e0b
            - Фон: #f8fafc

            РАЗМЕР: 600x400 пикселей

            ТЕМА: "{topic}"

            Верни ТОЛЬКО SVG код.
            """

    def _create_fallback_infographic(self, topic: str, is_technical: bool = False, subject: str = "") -> str:
        """Создает простую SVG инфографику как fallback"""
        topic_short = topic[:50] + "..." if len(topic) > 50 else topic
        
        if is_technical:
            # Технический fallback
            subject_display = subject if subject else "технический предмет"
            icon_text = "Σ" if 'математика' in subject.lower() else "F" if 'физика' in subject.lower() else "H₂O" if 'химия' in subject.lower() else "DNA" if 'биология' in subject.lower() else "?"
            
            return f'''<svg width="600" height="400" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="techGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#2563eb" />
      <stop offset="100%" stop-color="#1d4ed8" />
    </linearGradient>
    <filter id="techShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="4" dy="4" stdDeviation="8" flood-color="#1e3a8a" flood-opacity="0.3"/>
    </filter>
  </defs>
  
  <!-- Фон -->
  <rect width="100%" height="100%" fill="url(#techGradient)" opacity="0.08"/>
  
  <!-- Основной контейнер -->
  <g filter="url(#techShadow)">
    <rect x="50" y="50" width="500" height="300" rx="15" fill="white" stroke="#d1d5db" stroke-width="2"/>
  </g>
  
  <!-- Заголовок -->
  <text x="300" y="100" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="bold" fill="#1f2937">
    {topic_short}
  </text>
  
  <!-- Предмет -->
  <text x="300" y="130" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#6b7280">
    {subject_display}
  </text>
  
  <!-- Научная иконка -->
  <g transform="translate(300, 220)">
    <rect x="-40" y="-40" width="80" height="80" rx="10" fill="#2563eb" opacity="0.9"/>
    <text x="0" y="10" text-anchor="middle" font-family="Arial, sans-serif" font-size="32" font-weight="bold" fill="white">
      {icon_text}
    </text>
  </g>
  
  <!-- Подпись -->
  <text x="300" y="290" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#4b5563">
    Научная инфографика
  </text>
  
  <!-- Декоративные элементы - научные символы -->
  <text x="100" y="100" font-family="Arial, sans-serif" font-size="16" fill="#2563eb" opacity="0.6">∫</text>
  <text x="500" y="120" font-family="Arial, sans-serif" font-size="18" fill="#10b981" opacity="0.6">∑</text>
  <text x="80" y="280" font-family="Arial, sans-serif" font-size="14" fill="#ef4444" opacity="0.6">π</text>
  <text x="520" y="260" font-family="Arial, sans-serif" font-size="20" fill="#8b5cf6" opacity="0.6">∞</text>
</svg>'''
        else:
            # Стандартный fallback
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
                "max_tokens": 5
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
        # Теперь получаем модели из key_manager
        known_models = self.key_manager.get_known_models()
        
        return [
            {"id": model, "name": model.split('/')[-1].split(':')[0], "description": f"Модель OpenRouter: {model}"}
            for model in known_models
        ]

    def debug_infographic_generation(self, topic: str, subject: str = ""):
        """Отладочный метод для тестирования генерации инфографики"""
        print(f"🔧 DEBUG: Генерация инфографики для: {topic}, предмет: {subject}")
        
        # Проверяем тип предмета
        is_technical = False
        if TECHNICAL_SUPPORT_ENABLED and subject:
            is_technical = is_technical_subject(subject)
        
        test_prompt = f"Создай {'техническую ' if is_technical else ''}SVG инфографику на тему: {topic}"
        
        response = self._query_llm_api(
            prompt=test_prompt,
            context="",
            subject=subject,
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

    def generate_formula_explanation(self, formula: str, subject: str = "") -> str:
        """🔥 НОВЫЙ МЕТОД: Генерация объяснения для формул (для технических предметов)"""
        if not formula or not subject:
            return ""
        
        prompt = f"""
        Объясни формулу или математическое выражение:
        
        ФОРМУЛА: {formula}
        ПРЕДМЕТ: {subject}
        
        Объясни:
        1. Что означает эта формула?
        2. Какие величины в ней используются?
        3. В каких случаях применяется?
        4. Как ее использовать для расчетов?
        5. Приведи простой пример расчета.
        
        Объяснение должно быть понятным для ученика.
        Используй русский язык для объяснений.
        """
        
        try:
            explanation = self.query(prompt, subject=subject)
            return explanation
        except Exception as e:
            print(f"❌ Ошибка генерации объяснения формулы: {e}")
            return f"Формула {formula} используется в {subject}. Для расчетов нужно подставить значения переменных."

    def adapt_for_technical_subject(self, text: str, subject: str) -> str:
        """🔥 НОВЫЙ МЕТОД: Адаптация текста для технического предмета"""
        if not TECHNICAL_SUPPORT_ENABLED or not is_technical_subject(subject):
            return text
        
        print(f"🔧 [LLM] Адаптация текста для технического предмета: {subject}")
        
        # Простая адаптация - добавление инструкций
        adapted = f"""
        🔬 ДЛЯ ТЕХНИЧЕСКОГО ПРЕДМЕТА "{subject}":
        
        {text}
        
        💡 ТЕХНИЧЕСКИЕ АСПЕКТЫ:
        - Используй точные формулировки
        - Приводи формулы и расчеты
        - Объясняй пошагово
        - Давай практические примеры
        """
        
        return adapted

# Создаем глобальный экземпляр для использования в других модулях
llm_integration = LLMIntegration()

def get_llm_instance() -> LLMIntegration:
    """Возвращает глобальный экземпляр LLMIntegration"""
    return llm_integration

if __name__ == "__main__":
    # Тестирование модуля
    llm = LLMIntegration()
    
    print("🔧 Тестирование улучшенного LLM модуля с поддержкой технических предметов...")
    
    # Тестирование определения технических предметов
    print(f"\n🔬 Техническая поддержка включена: {TECHNICAL_SUPPORT_ENABLED}")
    
    # Тестирование генерации инфографики для разных предметов
    test_cases = [
        ("Сумма углов треугольника равна 180 градусов", "математика"),
        ("Закон Ома: I = U/R", "физика"),
        ("Фотосинтез: CO₂ + H₂O → C₆H₁₂O₆ + O₂", "биология"),
        ("Квадратное уравнение: ax² + bx + c = 0", "алгебра"),
    ]
    
    for topic, subject in test_cases:
        print(f"\n🔄 Генерация инфографики для: {topic} ({subject})")
        infographic_result = llm.generate_infographic(topic, subject=subject)
        
        if infographic_result["success"]:
            print(f"✅ Инфографика успешно сгенерирована! Тип: {infographic_result['type']}")
            print(f"📊 SVG код (первые 200 символов): {infographic_result['svg_code'][:200]}...")
        else:
            print(f"❌ Не удалось сгенерировать инфографику")
            print(f"📊 Использован fallback")
    
    # Тестирование очистки текста
    print("\n🧪 Тестирование умной очистки текста:")
    test_texts = [
        ("Уравнение: E=mc², где E - энергия", "физика"),
        ("Молекула воды: H₂O состоит из 2 атомов водорода и 1 атома кислорода", "химия"),
        ("Интеграл ∫f(x)dx показывает площадь под кривой", "математика"),
        ("Великая французская революция началась в 1789 году", "история"),
    ]
    
    for text, subject in test_texts:
        print(f"\n📝 Исходный текст: {text}")
        print(f"📚 Предмет: {subject}")
        
        # Тестируем очистку
        cleaned = llm._process_llm_response(text, subject=subject)
        print(f"🧹 Очищенный текст: {cleaned[:100]}...")
    
    print("\n🎉 Тестирование завершено!")
