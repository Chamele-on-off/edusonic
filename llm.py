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
        
        # НАСТРОЙКИ ДЛЯ DOCKER - ВАЖНО!
        self.local_llm_url = "http://host.docker.internal:11434/v1"  # Docker специальный адрес
        self.local_model = "llama3.2:3b"
        self.use_local_llm = True
        self.local_llm_enabled = True
        
        # Fallback на OpenRouter
        config = load_config()
        openrouter_config = get_model_config("openrouter")
        self.openrouter_api_key = openrouter_config.get("api_key", "")
        self.openrouter_url = openrouter_config.get("api_url", "https://openrouter.ai/api/v1/chat/completions")
        self.openrouter_model = openrouter_config.get("model", "meta-llama/llama-3.3-8b-instruct:free")
        self.openrouter_enabled = bool(self.openrouter_api_key)
        
        # Общие настройки
        self.cache_dir = Path(cache_dir)
        self.cache = self._load_cache()
        self.last_request_time = 0
        self.request_delay = 0.3
        self.max_retries = 2
        self.retry_delay = 1.0
        
        # Проверяем доступность локальной Llama при инициализации
        self._check_local_llama_availability()

    def _check_local_llama_availability(self):
        """Проверяет доступность локальной Llama при запуске"""
        try:
            print("🔍 Проверка доступности локальной Llama...")
            response = requests.get(f"{self.local_llm_url.replace('/v1', '')}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                models = models_data.get("models", [])
                print(f"📋 Найдено моделей: {len(models)}")
                
                local_model_exists = any(model.get("name") == self.local_model for model in models)
                
                if local_model_exists:
                    print(f"✅ Локальная Llama доступна, модель '{self.local_model}' найдена")
                    self.local_llm_enabled = True
                    
                    # Тестируем что модель действительно работает
                    test_result = self._test_local_model()
                    if not test_result:
                        print("⚠️ Модель найдена, но не отвечает")
                        self.local_llm_enabled = False
                else:
                    available_models = [model.get("name", "unknown") for model in models]
                    print(f"❌ Модель '{self.local_model}' не найдена. Доступные модели: {available_models}")
                    self.local_llm_enabled = False
            else:
                print(f"❌ Локальная Llama недоступна (код: {response.status_code})")
                self.local_llm_enabled = False
                
        except Exception as e:
            print(f"❌ Ошибка проверки локальной Llama: {e}")
            self.local_llm_enabled = False

    def _test_local_model(self):
        """Тестирует что локальная модель действительно работает"""
        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "model": self.local_model,
                "messages": [{"role": "user", "content": "Тест"}],
                "max_tokens": 10,
                "stream": False
            }
            
            response = requests.post(
                f"{self.local_llm_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=5
            )
            
            if response.status_code == 200:
                print("✅ Локальная модель работает корректно")
                return True
            else:
                print(f"❌ Локальная модель не отвечает (код: {response.status_code})")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка тестирования модели: {e}")
            return False

    def _load_cache(self) -> Dict:
        """Загрузка кэша из файла"""
        cache_file = self.cache_dir / "llm_cache.json"
        try:
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Ошибка загрузка кэша: {e}")
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

    def _try_local_llama(self, prompt: str, context: str, system_prompt: str, max_tokens: int) -> Optional[str]:
        """Запрос к локальной Llama"""
        if not self.local_llm_enabled:
            return None
            
        try:
            headers = {"Content-Type": "application/json"}
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if context:
                messages.append({"role": "system", "content": f"Контекст: {context}"})
            
            messages.append({"role": "user", "content": prompt})

            data = {
                "model": self.local_model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            print(f"🦙 Запрос к ЛОКАЛЬНОЙ Llama: {prompt[:80]}...")
            
            response = requests.post(
                f"{self.local_llm_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=8
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                print(f"✅ Локальная Llama ответила ({len(content)} символов)")
                return self._clean_llm_response(content)
            elif response.status_code == 404:
                print(f"❌ Модель '{self.local_model}' не найдена в локальной Llama")
                self.local_llm_enabled = False
                return None
            else:
                print(f"❌ Локальная Llama ошибка: {response.status_code} - {response.text[:100]}")
                return None
                
        except requests.exceptions.Timeout:
            print("⏰ Таймаут запроса к локальной Llama")
            return None
        except Exception as e:
            print(f"❌ Ошибка локальной Llama: {e}")
            return None

    def _try_openrouter(self, prompt: str, context: str, system_prompt: str, max_tokens: int) -> Optional[str]:
        """Запрос к OpenRouter (fallback)"""
        if not self.openrouter_enabled or not self.openrouter_api_key:
            return None
            
        try:
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://ai-teacher.com",
                "X-Title": "AI Teacher"
            }
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if context:
                messages.append({"role": "system", "content": f"Контекст: {context}"})
            
            messages.append({"role": "user", "content": prompt})

            data = {
                "model": self.openrouter_model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            print(f"🌐 Fallback на OpenRouter: {prompt[:80]}...")
            
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.request_delay:
                time.sleep(self.request_delay - time_since_last_request)
            
            response = requests.post(
                self.openrouter_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                print(f"✅ OpenRouter ответил")
                return self._clean_llm_response(content)
            else:
                print(f"❌ OpenRouter ошибка: {response.status_code} - {response.text[:200]}")
                return None
                
        except Exception as e:
            print(f"❌ Ошибка OpenRouter: {e}")
            return None

    def _query_llm_api(self, prompt: str, context: str = "", subject: str = "", 
                       system_prompt: str = "", max_tokens: int = 1000) -> Optional[str]:
        
        # ПЕРВЫЙ ПРИОРИТЕТ: Локальная Llama
        if self.local_llm_enabled:
            local_response = self._try_local_llama(prompt, context, system_prompt, max_tokens)
            if local_response:
                return local_response
        
        # ВТОРОЙ ПРИОРИТЕТ: OpenRouter (fallback)
        if self.openrouter_enabled:
            openrouter_response = self._try_openrouter(prompt, context, system_prompt, max_tokens)
            if openrouter_response:
                return openrouter_response
        
        # Финальный fallback
        return self._get_fallback_response(prompt, subject)

    def _get_fallback_response(self, prompt: str, subject: str = "") -> str:
        """Возвращает fallback ответ когда LLM недоступен"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['что', 'как', 'почему', 'зачем']):
            return f"Интересный вопрос! По теме {subject if subject else 'этого'} есть много интересной информации. Давайте обсудим это подробнее!"
        
        if any(word in prompt_lower for word in ['объясни', 'расскажи', 'покажи']):
            return f"С удовольствием объясню! Это важный аспект {subject if subject else 'темы'}. Давайте разберем вместе."
        
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
        
        llm_response = self._query_llm_api(question, context, subject)
        
        if llm_response and llm_response.strip():
            print(f"✅ Ответ получен: {llm_response[:100]}...")
            self.cache[cache_key] = llm_response
            self._save_cache()
            return llm_response
        
        print("⚠️ Все LLM провайдеры недоступны, использую fallback ответ")
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
            "subjects": list(set(key.split('_')[0] for key in self.cache.keys() if '_' in key)),
            "local_llm_enabled": self.local_llm_enabled,
            "openrouter_enabled": self.openrouter_enabled
        }

    def set_model(self, model: str):
        """Установка модели LLM"""
        if model in ["llama3.2:3b", "llama3.2:1b"]:
            self.local_model = model
            print(f"🔧 Установлена локальная модель: {self.local_model}")
        else:
            available_models = {
                "llama": "meta-llama/llama-3.3-8b-instruct:free",
                "llama3": "meta-llama/llama-3.3-8b-instruct:free",
                "qwen": "qwen/qwen3-235b-a22b:free",
                "deepseek": "deepseek/deepseek-chat-v3-0324:free"
            }
            
            if model in available_models:
                self.openrouter_model = available_models[model]
                print(f"🔧 Установлена OpenRouter модель: {self.openrouter_model}")
            else:
                self.openrouter_model = model
                print(f"🔧 Установлена кастомная OpenRouter модель: {self.openrouter_model}")

    def clean_and_validate_mermaid_code(self, code: str) -> str:
        """Очистка и валидация Mermaid кода"""
        if not code:
            return ""
        
        code = re.sub(r'```[\s\S]*?```', '', code)
        code = re.sub(r'`[^`]*`', '', code)
        code = re.sub(r'[\*\#\-\_]{2,}', '', code)
        code = re.sub(r'%%.*', '', code)
        code = '\n'.join([line.strip() for line in code.split('\n') if line.strip()])
        
        valid_starts = ['graph', 'flowchart', 'sequenceDiagram', 'classDiagram', 'stateDiagram', 'pie', 'gantt']
        
        if not any(code.strip().startswith(start) for start in valid_starts):
            code = 'flowchart TD\n' + code
        
        lines = code.split('\n')
        if len(lines) < 2 or ('-->' not in code and '->' not in code):
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
        # Тестируем локальную Llama
        if self.local_llm_enabled:
            try:
                response = requests.get(f"{self.local_llm_url.replace('/v1', '')}/api/tags", timeout=5)
                if response.status_code == 200:
                    print("✅ Локальная Llama подключена")
                    return True
            except Exception as e:
                print(f"❌ Локальная Llama недоступна: {e}")
        
        # Тестируем OpenRouter
        if self.openrouter_enabled:
            try:
                headers = {
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json"
                }
                
                test_data = {
                    "model": self.openrouter_model,
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 10
                }
                
                response = requests.post(
                    self.openrouter_url,
                    headers=headers,
                    json=test_data,
                    timeout=10
                )
                
                if response.status_code == 200:
                    print("✅ OpenRouter подключен")
                    return True
                else:
                    print(f"❌ OpenRouter ошибка: {response.status_code}")
                    return False
                    
            except Exception as e:
                print(f"❌ Ошибка подключения к OpenRouter: {e}")
                return False
        
        return False

    def get_available_models(self) -> list:
        """Получение списка доступных моделей"""
        models = []
        
        # Локальные модели
        if self.local_llm_enabled:
            models.append({"id": "llama3.2:3b", "name": "Llama 3.2 3B", "description": "Локальная быстрая модель", "provider": "local"})
            models.append({"id": "llama3.2:1b", "name": "Llama 3.2 1B", "description": "Локальная сверхбыстрая модель", "provider": "local"})
        
        # OpenRouter модели
        if self.openrouter_enabled:
            models.extend([
                {"id": "llama", "name": "Llama 3.3 8B", "description": "Мощная модель от Meta", "provider": "openrouter"},
                {"id": "qwen", "name": "Qwen 2.5 32B", "description": "Качественная модель от Alibaba", "provider": "openrouter"},
                {"id": "deepseek", "name": "DeepSeek Chat", "description": "Продвинутая модель для сложных задач", "provider": "openrouter"}
            ])
        
        return models

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
    print(f"📡 Подключение: {'✅ Успешно' if connection_ok else '❌ Ошибка'}")
    
    # Тестовый запрос
    test_question = "Объясни, что такое фотосинтез"
    response = llm.query(test_question, subject="биология")
    
    print("\n🧪 Тестовый ответ:")
    print(response)
    
    # Статистика
    cache_stats = llm.get_cache_stats()
    print(f"\n💾 Статистика: {cache_stats}")
    
    print("\n🎉 Тестирование завершено!")