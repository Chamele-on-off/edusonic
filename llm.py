import requests
import json
import time
import re
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import hashlib
from config import get_api_key, get_model_config, get_llm_priority, load_config
import threading
from collections import defaultdict
import random

class LLMIntegration:
    def __init__(self):
        self.config = load_config()
        self.openrouter_api_key = get_api_key('openrouter')
        self.openrouter_model = self.config.get('openrouter', {}).get('model', 'meta-llama/llama-3.3-8b-instruct:free')
        self.local_model_url = self.config.get('local_llm', {}).get('url', 'http://localhost:11434/api/generate')
        self.local_model_name = self.config.get('local_llm', {}).get('model', 'llama3.2:3b')
        self.priority_mode = get_llm_priority()
        self.cache_dir = Path("cache/llm_responses")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Кэш для быстрого доступа
        self.response_cache = {}
        self.cache_ttl = 3600  # 1 час
        
        # Статистика использования
        self.usage_stats = {
            'openrouter_requests': 0,
            'local_llm_requests': 0,
            'cache_hits': 0,
            'total_requests': 0,
            'errors': 0
        }
        
        # Очередь запросов для избежания перегрузки
        self.request_queue = []
        self.processing = False
        self.queue_lock = threading.Lock()
        
        # Таймауты и ограничения
        self.timeout_openrouter = 30
        self.timeout_local = 60
        self.max_retries = 3
        
        # Callback система для асинхронных ответов
        self.callbacks = defaultdict(list)
        
        print(f"🔧 LLMIntegration инициализирован с приоритетом: {self.priority_mode}")

    def _get_cache_key(self, prompt: str, context: str = "", subject: str = "") -> str:
        """Создает ключ кэша на основе промпта, контекста и предмета"""
        content = f"{prompt}|{context}|{subject}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """Загружает ответ из кэша"""
        # Сначала проверяем in-memory кэш
        if cache_key in self.response_cache:
            cached_data = self.response_cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_ttl:
                self.usage_stats['cache_hits'] += 1
                return cached_data['response']
            else:
                # Удаляем устаревший кэш
                del self.response_cache[cache_key]
        
        # Затем проверяем файловый кэш
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                if time.time() - cached_data['timestamp'] < self.cache_ttl:
                    # Сохраняем в memory кэш для быстрого доступа
                    self.response_cache[cache_key] = cached_data
                    self.usage_stats['cache_hits'] += 1
                    return cached_data['response']
                else:
                    # Удаляем устаревший файл
                    cache_file.unlink()
            except Exception as e:
                print(f"⚠️ Ошибка чтения кэша: {e}")
        
        return None

    def _save_to_cache(self, cache_key: str, response: str, subject: str = ""):
        """Сохраняет ответ в кэш"""
        cached_data = {
            'response': response,
            'timestamp': time.time(),
            'subject': subject
        }
        
        # Сохраняем в memory кэш
        self.response_cache[cache_key] = cached_data
        
        # Сохраняем в файловый кэш
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cached_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Ошибка сохранения в кэш: {e}")

    def _query_openrouter(self, prompt: str, system_prompt: str = "", max_tokens: int = 1000) -> Optional[str]:
        """Запрос к OpenRouter API"""
        if not self.openrouter_api_key:
            print("❌ OpenRouter API ключ не установлен")
            return None
        
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ai-teacher-app.com",
            "X-Title": "AI Teacher"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.openrouter_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        
        for attempt in range(self.max_retries):
            try:
                print(f"🌐 Запрос к OpenRouter (попытка {attempt + 1})...")
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=self.timeout_openrouter
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    self.usage_stats['openrouter_requests'] += 1
                    print(f"✅ Успешный ответ от OpenRouter: {len(content)} символов")
                    return content.strip()
                elif response.status_code == 429:
                    wait_time = (attempt + 1) * 5
                    print(f"⚠️ Rate limit, жду {wait_time} секунд...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Ошибка OpenRouter: {response.status_code} - {response.text}")
                    if attempt == self.max_retries - 1:
                        self.usage_stats['errors'] += 1
                    time.sleep(2)
                    
            except requests.exceptions.Timeout:
                print(f"❌ Таймаут OpenRouter (попытка {attempt + 1})")
                if attempt == self.max_retries - 1:
                    self.usage_stats['errors'] += 1
            except requests.exceptions.ConnectionError:
                print(f"❌ Ошибка подключения к OpenRouter (попытка {attempt + 1})")
                if attempt == self.max_retries - 1:
                    self.usage_stats['errors'] += 1
                time.sleep(3)
            except Exception as e:
                print(f"❌ Неожиданная ошибка OpenRouter: {e}")
                if attempt == self.max_retries - 1:
                    self.usage_stats['errors'] += 1
                time.sleep(2)
        
        return None

    def _query_local_llm(self, prompt: str, system_prompt: str = "", max_tokens: int = 1000) -> Optional[str]:
        """Запрос к локальной LLM (Ollama)"""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        data = {
            "model": self.local_model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": max_tokens
            }
        }
        
        for attempt in range(self.max_retries):
            try:
                print(f"💻 Запрос к локальной LLM (попытка {attempt + 1})...")
                response = requests.post(
                    self.local_model_url,
                    json=data,
                    timeout=self.timeout_local
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get('response', '').strip()
                    self.usage_stats['local_llm_requests'] += 1
                    print(f"✅ Успешный ответ от локальной LLM: {len(content)} символов")
                    return content
                else:
                    print(f"❌ Ошибка локальной LLM: {response.status_code} - {response.text}")
                    if attempt == self.max_retries - 1:
                        self.usage_stats['errors'] += 1
                    time.sleep(3)
                    
            except requests.exceptions.Timeout:
                print(f"❌ Таймаут локальной LLM (попытка {attempt + 1})")
                if attempt == self.max_retries - 1:
                    self.usage_stats['errors'] += 1
            except requests.exceptions.ConnectionError:
                print(f"❌ Ошибка подключения к локальной LLM (попытка {attempt + 1})")
                if attempt == self.max_retries - 1:
                    self.usage_stats['errors'] += 1
                time.sleep(5)
            except Exception as e:
                print(f"❌ Неожиданная ошибка локальной LLM: {e}")
                if attempt == self.max_retries - 1:
                    self.usage_stats['errors'] += 1
                time.sleep(2)
        
        return None

    def _test_openrouter_connection(self) -> bool:
        """Тестирование подключения к OpenRouter"""
        if not self.openrouter_api_key:
            return False
        
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        
        test_data = {
            "model": self.openrouter_model,
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 5
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=test_data,
                timeout=10
            )
            return response.status_code == 200
        except:
            return False

    def _test_local_llm_connection(self) -> bool:
        """Тестирование подключения к локальной LLM"""
        try:
            response = requests.get(
                self.local_model_url.replace('/api/generate', '/api/tags'),
                timeout=10
            )
            return response.status_code == 200
        except:
            return False

    def _query_llm_api(self, prompt: str, context: str = "", subject: str = "общее", 
                      system_prompt: str = "", max_tokens: int = 1000, 
                      room_id: str = None, callback: callable = None) -> Optional[str]:
        """Основной метод запроса к LLM с интеллектуальной маршрутизацией"""
        
        self.usage_stats['total_requests'] += 1
        
        # Создаем ключ кэша
        cache_key = self._get_cache_key(prompt, context, subject)
        
        # Пробуем загрузить из кэша
        cached_response = self._load_from_cache(cache_key)
        if cached_response:
            print(f"💾 Использован кэшированный ответ для предмета '{subject}'")
            
            # Если есть callback, вызываем его асинхронно
            if callback and room_id:
                threading.Thread(
                    target=callback,
                    args=(cached_response, room_id),
                    daemon=True
                ).start()
                return None
            
            return cached_response
        
        # Подготавливаем финальный промпт с контекстом
        final_prompt = prompt
        if context:
            final_prompt = f"Контекст: {context}\n\nВопрос: {prompt}"
        
        result = None
        
        # Интеллектуальная маршрутизация based on priority mode
        if self.priority_mode == "local_first":
            result = self._query_local_llm(final_prompt, system_prompt, max_tokens)
            if not result:
                result = self._query_openrouter(final_prompt, system_prompt, max_tokens)
                
        elif self.priority_mode == "openrouter_first":
            result = self._query_openrouter(final_prompt, system_prompt, max_tokens)
            if not result:
                result = self._query_local_llm(final_prompt, system_prompt, max_tokens)
                
        elif self.priority_mode == "local_only":
            result = self._query_local_llm(final_prompt, system_prompt, max_tokens)
            
        elif self.priority_mode == "openrouter_only":
            result = self._query_openrouter(final_prompt, system_prompt, max_tokens)
        
        # Если получили ответ, сохраняем в кэш
        if result:
            self._save_to_cache(cache_key, result, subject)
            
            # Если есть callback, вызываем его асинхронно
            if callback and room_id:
                threading.Thread(
                    target=callback,
                    args=(result, room_id),
                    daemon=True
                ).start()
                return None
        
        return result

    def query(self, question: str, context: str = "", subject: str = "общее") -> Optional[str]:
        """Публичный метод для запросов к LLM"""
        return self._query_llm_api(question, context, subject)

    def add_to_cache(self, question: str, answer: str, subject: str = ""):
        """Добавляет вопрос-ответ в кэш"""
        cache_key = self._get_cache_key(question, "", subject)
        self._save_to_cache(cache_key, answer, subject)

    def get_llm_status(self) -> Dict[str, Any]:
        """Возвращает статус LLM систем"""
        openrouter_ok = self._test_openrouter_connection()
        local_llm_ok = self._test_local_llm_connection()
        
        return {
            "openrouter": {
                "available": openrouter_ok,
                "model": self.openrouter_model,
                "api_key_set": bool(self.openrouter_api_key)
            },
            "local_llm": {
                "available": local_llm_ok,
                "model": self.local_model_name,
                "url": self.local_model_url
            },
            "priority_mode": self.priority_mode,
            "cache_stats": {
                "memory_cache_size": len(self.response_cache),
                "cache_hits": self.usage_stats['cache_hits'],
                "total_requests": self.usage_stats['total_requests']
            },
            "usage_stats": self.usage_stats.copy()
        }

    def set_priority(self, priority: str):
        """Устанавливает приоритет моделей"""
        valid_priorities = ["local_first", "openrouter_first", "local_only", "openrouter_only"]
        if priority in valid_priorities:
            self.priority_mode = priority
            print(f"🔧 Приоритет LLM изменен на: {priority}")
            return True
        return False

    def get_priority_status(self) -> Dict[str, Any]:
        """Возвращает статус приоритетов"""
        return {
            "current_priority": self.priority_mode,
            "available_priorities": ["local_first", "openrouter_first", "local_only", "openrouter_only"],
            "openrouter_available": self._test_openrouter_connection(),
            "local_llm_available": self._test_local_llm_connection()
        }

    def handle_llm_response(self, request_id: str, response: str, room_id: str):
        """Обработчик ответов от LLM для WebSocket коммуникации"""
        # Вызываем зарегистрированные callback'и
        if room_id in self.callbacks:
            for callback in self.callbacks[room_id]:
                try:
                    callback(request_id, response, room_id)
                except Exception as e:
                    print(f"❌ Ошибка в callback: {e}")

    def register_callback(self, room_id: str, callback: callable):
        """Регистрирует callback для комнаты"""
        self.callbacks[room_id].append(callback)

    def clear_callbacks(self, room_id: str):
        """Очищает callback'и для комнаты"""
        if room_id in self.callbacks:
            del self.callbacks[room_id]

    def generate_infographic(self, topic: str, context: str = "") -> dict:
        """Генерация стильной инфографики в SVG формате"""
        
        prompt = f"""
Создай стильную инфографику в формате SVG на тему: "{topic}"

КОНТЕКСТ: {context}

ТРЕБОВАНИЯ К ИНФОГРАФИКЕ:
- Только SVG код
- Стильный, современный дизайн
- Информативная и понятная структура
- Использование иконок, фигур, текста
- Цветовая палитра: приятные пастельные или яркие образовательные цвета
- Максимальная ширина: 600px
- Четкая визуальная иерархия
- Баланс между визуальными элементами и текстом

ЭЛЕМЕНТЫ ДЛЯ ИСПОЛЬЗОВАНИЯ:
- Прямоугольники с закругленными углами
- Стрелки для связей
- Иконки (простые геометрические формы)
- Текстовые блоки с заголовками
- Цветовые акценты для выделения ключевых моментов
- Простые диаграммы (круговые, линейные)

ВОЗМОЖНЫЕ ТИПЫ ВИЗУАЛИЗАЦИИ:
- Блок-схемы процессов
- Иерархические структуры  
- Сравнительные таблицы
- Временные линии
- Классификационные схемы
- Концептуальные карты

Верни ТОЛЬКО SVG код без каких-либо пояснений.
"""
    
        try:
            response = self._query_llm_api(
                prompt=prompt,
                context="",
                subject="general",
                system_prompt="""Ты - эксперт по созданию образовательной инфографики. 
Создавай чистый, семантически правильный SVG код.
Используй осмысленные id для элементов.
Соблюдай доступность (aria-label где нужно).
Создавай стильную и понятную инфографику.""",
                max_tokens=2000
            )
            
            if response:
                # Очистка ответа
                svg_code = self._extract_svg_code(response)
                if svg_code:
                    return {
                        "success": True,
                        "svg_code": svg_code,
                        "topic": topic,
                        "type": "infographic"
                    }
            
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

    def _extract_svg_code(self, response: str) -> str:
        """Извлекает чистый SVG код из ответа LLM"""
        # Удаляем markdown обрамление
        response = re.sub(r'```(xml|svg)?\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        response = response.strip()
        
        # Если ответ уже валидный SVG, возвращаем как есть
        if response.startswith('<svg') and response.endswith('</svg>'):
            return response
        
        # Ищем SVG теги в тексте
        svg_match = re.search(r'<svg[\s\S]*?</svg>', response)
        if svg_match:
            return svg_match.group(0)
        
        return ""

    def _create_fallback_infographic(self, topic: str) -> str:
        """Создает простую инфографику как fallback"""
        topic_short = topic[:30] + "..." if len(topic) > 30 else topic
        
        return f'''
<svg width="600" height="400" xmlns="http://www.w3.org/2000/svg">
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
</svg>
'''

    def clear_cache(self, older_than_hours: int = 24):
        """Очищает кэш старше указанного времени"""
        cutoff_time = time.time() - (older_than_hours * 3600)
        cleared_files = 0
        cleared_memory = 0
        
        # Очищаем файловый кэш
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                if cached_data.get('timestamp', 0) < cutoff_time:
                    cache_file.unlink()
                    cleared_files += 1
            except:
                pass
        
        # Очищаем memory кэш
        for cache_key in list(self.response_cache.keys()):
            if self.response_cache[cache_key].get('timestamp', 0) < cutoff_time:
                del self.response_cache[cache_key]
                cleared_memory += 1
        
        print(f"🗑️ Очищен кэш: {cleared_files} файлов, {cleared_memory} memory записей")
        return cleared_files + cleared_memory

    def get_detailed_stats(self) -> Dict[str, Any]:
        """Возвращает детальную статистику"""
        cache_files = list(self.cache_dir.glob("*.json"))
        
        return {
            "usage": self.usage_stats.copy(),
            "cache": {
                "memory_entries": len(self.response_cache),
                "file_entries": len(cache_files),
                "total_size_kb": sum(f.stat().st_size for f in cache_files) / 1024,
                "cache_hit_rate": self.usage_stats['cache_hits'] / max(1, self.usage_stats['total_requests'])
            },
            "connections": {
                "openrouter": self._test_openrouter_connection(),
                "local_llm": self._test_local_llm_connection()
            },
            "configuration": {
                "priority_mode": self.priority_mode,
                "openrouter_model": self.openrouter_model,
                "local_model": self.local_model_name
            }
        }

    def set_model(self, model_type: str, model_name: str):
        """Устанавливает модель для использования"""
        if model_type == "openrouter":
            self.openrouter_model = model_name
            print(f"🔧 Установлена OpenRouter модель: {model_name}")
        elif model_type == "local":
            self.local_model_name = model_name
            print(f"🔧 Установлена локальная модель: {model_name}")
        else:
            print(f"❌ Неизвестный тип модели: {model_type}")

# Глобальный экземпляр для использования в других модулях
_llm_instance = None

def get_llm_integration() -> LLMIntegration:
    """Возвращает глобальный экземпляр LLMIntegration"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMIntegration()
    return _llm_instance

# Тестирование модуля
if __name__ == "__main__":
    print("🧪 Тестирование LLMIntegration...")
    
    llm = LLMIntegration()
    
    # Тест статуса
    status = llm.get_llm_status()
    print(f"📊 Статус LLM: {status}")
    
    # Тест простого запроса
    test_response = llm.query("Привет! Как дела?")
    if test_response:
        print(f"✅ Тестовый запрос выполнен: {test_response[:100]}...")
    else:
        print("❌ Тестовый запрос не удался")
    
    # Тест инфографики
    infographic_result = llm.generate_infographic("Фотосинтез")
    if infographic_result and infographic_result.get("success"):
        print(f"✅ Инфографика сгенерирована: {len(infographic_result['svg_code'])} символов")
    else:
        print("❌ Генерация инфографики не удалась")
    
    print("✅ Тестирование завершено!")
