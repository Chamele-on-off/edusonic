import requests
import json
from typing import Optional, Dict
from pathlib import Path
import time
from config import get_api_key, load_config, get_model_config
import re
from local_llm import LocalLLM

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
        
        # УЛЬТРА-БЫСТРАЯ ЛОКАЛЬНАЯ МОДЕЛЬ
        self.local_llm = LocalLLM()
        self.use_local_first = True  # Флаг приоритета локальной модели
        
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
                       system_prompt: str = "", max_tokens: int = 1000) -> Optional[str]:
        """УЛУЧШЕННЫЙ ЗАПРОС С ПРИОРИТЕТОМ УЛЬТРА-БЫСТРОЙ ЛОКАЛЬНОЙ МОДЕЛИ"""
        
        # 1. УЛЬТРА-БЫСТРАЯ ЛОКАЛЬНАЯ МОДЕЛЬ (всегда сначала)
        if self.use_local_first and self.local_llm.is_available():
            print("⚡ Использую ультра-быструю локальную модель...")
            start_time = time.time()
            
            # ДАЕМ БОЛЬШЕ ВРЕМЕНИ ДЛЯ ОТВЕТА - 15 СЕКУНД
            try:
                local_response = self.local_llm.generate(prompt, system_prompt, max_tokens)
                response_time = time.time() - start_time
                
                if local_response and len(local_response.strip()) > 10:  # УВЕЛИЧИЛИ МИНИМАЛЬНУЮ ДЛИНУ
                    print(f"✅ Локальный ответ за {response_time:.2f}с: {local_response[:80]}...")
                    return local_response
                else:
                    print(f"❌ Локальная модель не дала содержательный ответ за {response_time:.2f}с")
                    # НЕ ПЕРЕКЛЮЧАЕМСЯ СРАЗУ, ПРОБУЕМ OPENROUTER
                    
            except Exception as e:
                print(f"❌ Ошибка локальной модели: {e}")
        
        # 2. Fallback на OpenRouter (оригинальная логика без изменений)
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

        print(f"🔧 Запрос к OpenRouter:")
        print(f"   Модель: {self.model}")
        print(f"   Промпт: {prompt[:100]}...")
        if context:
            print(f"   Контекст: {context[:100]}...")
        print(f"   Предмет: {subject}")
        
        for attempt in range(self.max_retries):
            try:
                print(f"🔄 Попытка {attempt + 1}: Отправка запроса к OpenRouter")
                start_time = time.time()
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=data,
                    timeout=30
                )
                response_time = time.time() - start_time
                
                self.last_request_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if 'choices' in result and len(result['choices']) > 0:
                        answer = result['choices'][0]['message']['content']
                        processed_answer = self._clean_llm_response(answer)
                        
                        # Проверяем, содержит ли ответ концепции для визуализации
                        visualization_concepts = ['диаграмм', 'схем', 'график', 'структур', 'процесс']
                        if any(concept in processed_answer.lower() for concept in visualization_concepts):
                            print("🎯 Ответ содержит концепции для визуализации")
                        
                        print(f"✅ Получен ответ от OpenRouter за {response_time:.2f}с: {processed_answer[:100]}...")
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

    def set_llm_priority(self, use_local_first: bool):
        """Установка приоритета моделей"""
        self.use_local_first = use_local_first
        status = "локальная модель" if use_local_first else "OpenRouter"
        print(f"🔧 Приоритет установлен на: {status}")

    def get_llm_status(self) -> Dict:
        """Получение статуса моделей"""
        local_available = self.local_llm.is_available()
        openrouter_available = bool(self.api_key)
        
        # ПРОВЕРЯЕМ РАБОТОСПОСОБНОСТЬ ЛОКАЛЬНОЙ МОДЕЛИ
        local_working = False
        if local_available:
            try:
                test_response = self.local_llm.generate("Тест", "Ты - учитель", 10)
                local_working = test_response is not None and len(test_response.strip()) > 5
            except:
                local_working = False
        
        return {
            "local_available": local_available,
            "local_working": local_working,
            "openrouter_available": openrouter_available,
            "current_priority": "local" if self.use_local_first else "openrouter",
            "local_status": self.local_llm.get_status(),
            "local_url": self.local_llm.base_url
        }

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

        Тема для диаграммы: {topic}
        
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
            headers = {
                "Authorization": f"Bearer {self.api_key}",
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
