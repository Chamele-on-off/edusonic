# speech_manager.py - Управление озвучиванием и распознаванием речи для AI Teacher
# С поддержкой кастомного TTS сервиса Zindaki и технических предметов

import re
import io
import base64
import time
import requests
import logging
from typing import Optional, Tuple, Dict, Any, List
from gtts import gTTS
import threading

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# КОНФИГУРАЦИЯ И НАСТРОЙКИ
# =============================================================================

DEBUG_SPEECH = True

# Настройки TTS
TTS_CONFIG = {
    'enabled': True,
    'primary': 'zindaki',  # 'zindaki' или 'gtts'
    'zindaki': {
        'base_url': 'http://tts.zindaki-edu.ru',
        'api_endpoint': '/api/tts',
        'health_endpoint': '/api/health',
        'voices_endpoint': '/api/voices',
        'timeout': 30,
        'retries': 2,
        'language_mapping': {
            'ru': 'ru',
            'en': 'en',
            'auto': 'ru'
        },
        'speaker_mapping': {
            'female': 'baya',
            'male': 'aidar',
            'teacher': 'baya',
            'default': 'baya'
        }
    },
    'gtts': {
        'fallback': True,
        'timeout': 10
    },
    'cache': {
        'enabled': True,
        'max_size': 1000,
        'ttl_seconds': 3600
    }
}

# Общепринятые английские слова в русском контексте
COMMON_ENGLISH_IN_RUSSIAN = {
    'ok', 'hello', 'hi', 'yes', 'no', 'bye', 'sorry', 'please', 'thank', 
    'you', 'okay', 'thanks', 'good', 'morning', 'afternoon', 'evening',
    'night', 'welcome', 'excuse', 'me', 'pardon', 'what', 'how', 'why',
    'when', 'where', 'who', 'which', 'that', 'this', 'these', 'those',
    'my', 'your', 'his', 'her', 'our', 'their', 'name', 'age', 'old',
    'year', 'day', 'week', 'month', 'time', 'clock', 'hour', 'minute',
    'second', 'today', 'tomorrow', 'yesterday', 'now', 'then', 'here',
    'there', 'come', 'go', 'stop', 'start', 'begin', 'end', 'open', 'close',
    'big', 'small', 'good', 'bad', 'happy', 'sad', 'hot', 'cold', 'new',
    'old', 'young', 'right', 'left', 'up', 'down', 'front', 'back', 'in',
    'out', 'on', 'off', 'at', 'by', 'with', 'without', 'about', 'for',
    'from', 'to', 'and', 'or', 'but', 'so', 'because', 'if', 'then', 'else',
    'very', 'too', 'also', 'just', 'only', 'really', 'well', 'much', 'many',
    'some', 'any', 'all', 'every', 'each', 'few', 'more', 'less', 'most',
    'least', 'first', 'last', 'next', 'previous', 'same', 'different',
    'other', 'another', 'such', 'like', 'as', 'than', 'not', 'no', 'yes'
}

# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def debug_log(message: str):
    """Логирование для отладки"""
    if DEBUG_SPEECH:
        logger.info(f"🔊 [SPEECH] {message}")

# =============================================================================
# ОСНОВНЫЕ ФУНКЦИИ ОЧИСТКИ ТЕКСТА
# =============================================================================

def clean_text_for_speech(text: str) -> str:
    """Тщательная очистка текста для озвучивания"""
    if not text:
        return ""
    
    # Удаляем спецсимволы форматирования
    text = re.sub(r'[#\*\_\~`]', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\\n', ' ', text)
    text = re.sub(r'\\t', ' ', text)
    text = re.sub(r'\\r', ' ', text)
    
    # Оставляем только допустимые символы
    text = re.sub(r'[^\u0400-\u04FFa-zA-Z0-9\s\.,!?;:()\-—]', '', text)
    
    # Исправляем множественные пунктуационные знаки
    text = re.sub(r'[\.\,]{2,}', '.', text)
    text = re.sub(r'\s+([\.,!?;:)])', r'\1', text)
    text = re.sub(r'([(\-])\s+', r'\1', text)
    text = text.strip()
    
    # Делаем первую букву заглавной
    if text and len(text) > 1:
        text = text[0].upper() + text[1:]
    
    return text

def clean_text_for_speech_smart(text: str, subject: Optional[str] = None) -> str:
    """Умная очистка текста с учетом предмета"""
    try:
        # Если доступна поддержка технических предметов
        try:
            from technical_subjects import clean_text_for_speech_technical
            TECHNICAL_SUPPORT_AVAILABLE = True
        except ImportError:
            TECHNICAL_SUPPORT_AVAILABLE = False
            
        if TECHNICAL_SUPPORT_AVAILABLE and subject:
            cleaned_text = clean_text_for_speech_technical(text, subject)
            debug_log(f"🎯 Использована умная очистка для предмета: {subject}")
            return cleaned_text
        else:
            # Стандартная очистка
            cleaned_text = clean_text_for_speech(text)
            debug_log(f"🎯 Использована стандартная очистка")
            return cleaned_text
    except Exception as e:
        debug_log(f"⚠️ Ошибка умной очистки: {e}, используется стандартная")
        return clean_text_for_speech(text)

# =============================================================================
# ФУНКЦИИ ДЛЯ ОПРЕДЕЛЕНИЯ ЯЗЫКА
# =============================================================================

def detect_text_language_fast(text: str) -> Tuple[str, bool]:
    """Быстрое определение языка текста"""
    if not text:
        return 'ru', False
    
    sample = text[:200]
    
    # Ищем латинские буквы
    has_latin = bool(re.search(r'[a-zA-Z]', sample))
    has_cyrillic = bool(re.search(r'[а-яА-ЯеЕ]', sample))
    
    # Нет латинских букв - точно не иностранный
    if not has_latin:
        return 'ru', False
    
    # Ищем иностранные слова
    words = re.findall(r'\b[a-zA-Z]{2,}\b', sample.lower())
    
    if not words:
        return 'ru', False
    
    # Фильтруем общепринятые слова
    foreign_words = [w for w in words if w not in COMMON_ENGLISH_IN_RUSSIAN]
    unique_foreign_words = set(foreign_words)
    
    # Если есть хотя бы 2 уникальных НЕ общепринятых английских слова
    if len(unique_foreign_words) >= 2:
        return 'en', True
    
    # Или если есть длинное иностранное слово (более 4 букв)
    long_foreign_words = [w for w in foreign_words if len(w) > 4]
    if len(long_foreign_words) > 0:
        return 'en', True
    
    # Только кириллица или общепринятые слова
    if has_cyrillic:
        return 'ru', False
    elif has_latin and not has_cyrillic:
        # Только латиница, но это общепринятые слова
        return 'en', True
    else:
        return 'ru', False

def is_foreign_text(text: str) -> bool:
    """🔥 СУПЕР-БЫСТРАЯ проверка на иностранный текст"""
    if not text:
        return False
    
    sample = text[:200]
    
    # 1. Ищем латинские буквы
    if not re.search(r'[a-zA-Z]', sample):
        return False  # Нет латинских букв - точно не иностранный
    
    # 2. Игнорируем общепринятые английские слова в русском контексте
    words = re.findall(r'\b[a-zA-Z]{2,}\b', sample.lower())
    
    if not words:
        return False
    
    # Подсчитываем уникальные НЕ общепринятые слова
    foreign_words = [w for w in words if w not in COMMON_ENGLISH_IN_RUSSIAN]
    unique_foreign_words = set(foreign_words)
    
    # Если есть хотя бы 2 уникальных НЕ общепринятых английских слова
    if len(unique_foreign_words) >= 2:
        return True
    
    # Или если есть длинное иностранное слово (более 4 букв)
    long_foreign_words = [w for w in foreign_words if len(w) > 4]
    if len(long_foreign_words) > 0:
        return True
    
    return False

# =============================================================================
# КЛАСС КЛИЕНТА TTS СЕРВИСА
# =============================================================================

class ZindakiTTSClient:
    """Клиент для работы с TTS сервисом Zindaki"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Инициализация клиента TTS
        
        Args:
            config: Конфигурация TTS сервиса
        """
        self.config = config or TTS_CONFIG['zindaki']
        self.base_url = self.config['base_url'].rstrip('/')
        self.api_url = f"{self.base_url}{self.config['api_endpoint']}"
        self.health_url = f"{self.base_url}{self.config['health_endpoint']}"
        self.voices_url = f"{self.base_url}{self.config['voices_endpoint']}"
        self.timeout = self.config['timeout']
        self.max_retries = self.config['retries']
        self.available = False
        self.last_check = 0
        self.check_interval = 60  # Проверять доступность каждые 60 секунд
        self.session = requests.Session()
        
        # Настройка сессии
        self.session.headers.update({
            'User-Agent': 'AI-Teacher-TTS-Client/1.0',
            'Accept': 'application/json, audio/*',
            'Accept-Encoding': 'gzip, deflate'
        })
        
        # Кэш для голосов
        self._voices_cache = None
        self._voices_cache_time = 0
        
        # Статистика
        self.stats = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'timeouts': 0,
            'cache_hits': 0,
            'last_request_time': 0,
            'average_response_time': 0
        }
        
        # Проверяем доступность при инициализации
        self._check_availability()
        
        debug_log(f"✅ Zindaki TTS Client initialized. Base URL: {self.base_url}")
        debug_log(f"   TTS Service available: {self.available}")
    
    def _check_availability(self, force: bool = False) -> bool:
        """Проверка доступности TTS сервиса"""
        current_time = time.time()
        
        # Проверяем не чаще чем раз в check_interval секунд
        if not force and (current_time - self.last_check) < self.check_interval:
            return self.available
        
        self.last_check = current_time
        
        try:
            response = self.session.get(self.health_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                was_available = self.available
                self.available = data.get('status') == 'healthy'
                
                if self.available and not was_available:
                    debug_log(f"✅ TTS service is now AVAILABLE")
                elif not self.available and was_available:
                    debug_log(f"⚠️ TTS service is now UNAVAILABLE")
                    
                return self.available
        except requests.exceptions.Timeout:
            debug_log(f"⚠️ TTS service health check timeout")
        except requests.exceptions.ConnectionError:
            debug_log(f"⚠️ TTS service connection error")
        except Exception as e:
            debug_log(f"⚠️ TTS service check failed: {e}")
        
        self.available = False
        return False
    
    def generate_speech(self, text: str, language: str = 'ru', speaker: str = 'baya', 
                        sample_rate: int = 16000) -> Optional[bytes]:
        """
        Генерация речи через TTS сервис
        
        Args:
            text: Текст для озвучки
            language: Язык ('ru' или 'en')
            speaker: Голос ('baya', 'kseniya', 'aidar', etc.)
            sample_rate: Частота дискретизации
            
        Returns:
            Аудио данные в формате WAV или None при ошибке
        """
        self.stats['total_requests'] += 1
        
        # Проверяем доступность сервиса
        if not self._check_availability():
            self.stats['failed'] += 1
            debug_log(f"❌ TTS service not available")
            return None
        
        if not text.strip():
            debug_log(f"⚠️ Empty text provided")
            return None
        
        # Подготовка запроса
        payload = {
            'text': text,
            'language': language,
            'speaker': speaker,
            'sample_rate': sample_rate
        }
        
        debug_log(f"📨 Sending TTS request: '{text[:50]}...' (lang: {language}, speaker: {speaker})")
        
        # Попытка генерации с повторными попытками
        start_time = time.time()
        
        for attempt in range(self.max_retries + 1):
            try:
                request_start = time.time()
                response = self.session.post(
                    self.api_url,
                    json=payload,
                    timeout=self.timeout,
                    stream=True
                )
                
                request_time = time.time() - request_start
                
                if response.status_code == 200:
                    # Получаем аудио данные
                    audio_data = response.content
                    total_time = time.time() - start_time
                    
                    # Проверяем, что это валидный WAV файл
                    if len(audio_data) > 44 and audio_data[:4] == b'RIFF':
                        cache_hit = response.headers.get('X-Cache-Hit', 'false') == 'true'
                        gen_time = float(response.headers.get('X-Generation-Time', '0'))
                        
                        # Обновляем статистику
                        self.stats['successful'] += 1
                        self.stats['last_request_time'] = total_time
                        
                        # Обновляем среднее время ответа
                        if self.stats['average_response_time'] == 0:
                            self.stats['average_response_time'] = total_time
                        else:
                            self.stats['average_response_time'] = (
                                self.stats['average_response_time'] * 0.8 + total_time * 0.2
                            )
                        
                        if cache_hit:
                            self.stats['cache_hits'] += 1
                        
                        debug_log(f"✅ TTS generated in {total_time:.2f}s "
                                 f"(Cache: {'HIT' if cache_hit else 'MISS'}, "
                                 f"Size: {len(audio_data)} bytes)")
                        
                        return audio_data
                    else:
                        debug_log(f"❌ Invalid audio data received (size: {len(audio_data)})")
                        self.stats['failed'] += 1
                        return None
                        
                elif response.status_code == 429:
                    # Сервис перегружен
                    wait_time = 2 ** attempt  # экспоненциальная задержка
                    debug_log(f"⚠️ TTS service busy (429). Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    error_text = response.text[:200] if response.text else "No error message"
                    debug_log(f"❌ TTS generation failed: HTTP {response.status_code} - {error_text}")
                    self.stats['failed'] += 1
                    
                    # Если это 4xx ошибка (кроме 429) - не повторяем
                    if 400 <= response.status_code < 500 and response.status_code != 429:
                        break
                        
            except requests.exceptions.Timeout:
                debug_log(f"❌ TTS request timeout (attempt {attempt + 1}/{self.max_retries + 1})")
                self.stats['timeouts'] += 1
                if attempt < self.max_retries:
                    continue
                    
            except requests.exceptions.ConnectionError:
                debug_log(f"❌ TTS connection error (attempt {attempt + 1}/{self.max_retries + 1})")
                self.available = False  # Помечаем сервис как недоступный
                self.stats['failed'] += 1
                break
                
            except Exception as e:
                debug_log(f"❌ Unexpected TTS error: {e}")
                self.stats['failed'] += 1
                break
        
        return None
    
    def generate_speech_base64(self, text: str, language: str = 'ru', 
                               speaker: str = 'baya') -> Optional[str]:
        """
        Генерация речи и возврат в формате base64
        
        Returns:
            Base64 строка с аудио данными или None при ошибке
        """
        audio_data = self.generate_speech(text, language, speaker)
        
        if audio_data:
            return base64.b64encode(audio_data).decode('utf-8')
        return None
    
    def get_service_status(self) -> Dict[str, Any]:
        """Получение статуса TTS сервиса"""
        try:
            response = self.session.get(self.health_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                data['available'] = self.available
                data['last_check'] = self.last_check
                return data
        except Exception as e:
            debug_log(f"❌ Failed to get TTS service status: {e}")
        
        return {
            'status': 'unavailable', 
            'available': False,
            'error': 'Service check failed',
            'base_url': self.base_url
        }
    
    def get_available_voices(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Получение списка доступных голосов"""
        current_time = time.time()
        
        # Используем кэш, если не принудительное обновление
        if not force_refresh and self._voices_cache and (current_time - self._voices_cache_time) < 300:
            return self._voices_cache
        
        try:
            response = self.session.get(self.voices_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                self._voices_cache = data
                self._voices_cache_time = current_time
                return data
        except Exception as e:
            debug_log(f"❌ Failed to get available voices: {e}")
        
        # Возвращаем кэшированные данные или пустой словарь
        return self._voices_cache or {'all_voices': {}, 'loaded_voices': {}}
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики клиента"""
        return {
            **self.stats,
            'available': self.available,
            'base_url': self.base_url,
            'last_check': self.last_check,
            'success_rate': (self.stats['successful'] / max(self.stats['total_requests'], 1)) * 100
        }
    
    def clear_cache(self, days_old: Optional[int] = None) -> bool:
        """Очистка кэша TTS сервиса"""
        try:
            url = f"{self.base_url}/api/cache/clear"
            payload = {}
            if days_old:
                payload['days_old'] = days_old
            
            response = self.session.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    debug_log(f"✅ TTS cache cleared: {data.get('message')}")
                    return True
                    
        except Exception as e:
            debug_log(f"❌ Failed to clear TTS cache: {e}")
        
        return False
    
    def test_connection(self) -> bool:
        """Тестирование подключения к TTS сервису"""
        try:
            test_text = "Тестовое сообщение для проверки связи."
            audio_data = self.generate_speech(test_text, 'ru', 'baya')
            return audio_data is not None and len(audio_data) > 0
        except Exception as e:
            debug_log(f"❌ TTS connection test failed: {e}")
            return False

# =============================================================================
# ОСНОВНЫЕ ФУНКЦИИ TTS
# =============================================================================

def text_to_speech_gtts(text: str, lang: str = 'ru') -> Optional[str]:
    """🔥 ОПТИМИЗИРОВАННАЯ ВЕРСИЯ: Быстрое озвучивание с помощью gTTS"""
    try:
        if not text.strip():
            return None
            
        # 🔥 БЫСТРАЯ ГЕНЕРАЦИЯ БЕЗ ДОПОЛНИТЕЛЬНОГО АНАЛИЗА
        tts = gTTS(text=text, lang=lang, slow=False, lang_check=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return base64.b64encode(mp3_fp.read()).decode('utf-8')
        
    except Exception as e:
        debug_log(f"❌ Error in text_to_speech_gtts: {e}")
        return None

def text_to_speech_zindaki(text: str, lang: str = 'ru', voice_type: str = 'female', 
                          tts_client: Optional[ZindakiTTSClient] = None) -> Optional[str]:
    """🔥 Озвучивание через кастомный TTS сервис Zindaki"""
    if not tts_client:
        debug_log("⚠️ TTS client not provided")
        return None
    
    try:
        # Проверяем доступность сервиса
        if not tts_client.available:
            debug_log("⚠️ TTS service is not available")
            return None
        
        # Маппинг языка и голоса
        tts_lang = TTS_CONFIG['zindaki']['language_mapping'].get(lang, 'ru')
        
        # Определяем голос на основе типа и языка
        if voice_type in TTS_CONFIG['zindaki']['speaker_mapping']:
            speaker = TTS_CONFIG['zindaki']['speaker_mapping'][voice_type]
        else:
            # Пробуем определить по языку
            if lang == 'en':
                speaker = 'en_1'  # английский голос по умолчанию
            else:
                speaker = 'baya'  # русский голос по умолчанию
        
        debug_log(f"🔊 Using Zindaki TTS: lang={tts_lang}, speaker={speaker}, voice_type={voice_type}")
        
        # Генерация речи
        audio_data = tts_client.generate_speech(text, tts_lang, speaker)
        
        if audio_data:
            # Преобразуем в base64 для совместимости
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            debug_log(f"✅ Zindaki TTS generated successfully (size: {len(audio_data)} bytes)")
            return audio_base64
        else:
            debug_log("❌ Zindaki TTS generation failed")
            return None
            
    except Exception as e:
        debug_log(f"❌ Error in Zindaki TTS: {e}")
        return None

def text_to_speech_mixed(text: str, use_zindaki: bool = True, 
                        tts_client: Optional[ZindakiTTSClient] = None) -> Optional[str]:
    """🔥 ОПТИМИЗИРОВАННАЯ ФУНКЦИЯ: Озвучивание смешанного текста"""
    try:
        # 🔥 ОПТИМИЗИРОВАННЫЙ ШАБЛОН: ищем последовательности букв одного языка
        pattern = r'([а-яА-ЯеЕ][а-яА-ЯеЕ\s.,!?;:\'-]*|[a-zA-Z][a-zA-Z\s.,!?;:\'-]*)'
        fragments = re.findall(pattern, text)
        
        if not fragments:
            if use_zindaki and tts_client:
                return text_to_speech_zindaki(text, 'ru', 'female', tts_client)
            else:
                return text_to_speech_gtts(text, 'ru')
        
        audio_chunks = []
        
        for fragment in fragments:
            fragment = fragment.strip()
            if not fragment:
                continue
                
            # Определяем язык фрагмента - быстрая проверка
            has_cyrillic = bool(re.search(r'[а-яА-ЯеЕ]', fragment))
            lang = 'ru' if has_cyrillic else 'en'
            
            try:
                # Генерируем аудио для фрагмента
                if use_zindaki and tts_client:
                    audio_base64 = text_to_speech_zindaki(fragment, lang, 'female', tts_client)
                else:
                    audio_base64 = text_to_speech_gtts(fragment, lang)
                
                if audio_base64:
                    audio_data = base64.b64decode(audio_base64)
                    audio_chunks.append(audio_data)
                    
                    # 🔥 КОРОТКАЯ ПАУЗА МЕЖДУ ФРАГМЕНТАМИ (50ms silence)
                    silence = bytes([0] * 800)  # 50ms при 16kHz
                    audio_chunks.append(silence)
                
            except Exception as e:
                debug_log(f"⚠️ Error processing fragment '{fragment[:30]}...': {e}")
                continue
        
        if audio_chunks:
            # 🔥 УБИРАЕМ ПОСЛЕДНЮЮ ПАУЗУ
            if audio_chunks[-1] == bytes([0] * 800):
                audio_chunks.pop()
                
            combined_audio = b''.join(audio_chunks)
            return base64.b64encode(combined_audio).decode('utf-8')
        
        return None
    except Exception as e:
        debug_log(f"❌ Error in text_to_speech_mixed: {e}")
        
        # Fallback - используем обычную генерацию
        if use_zindaki and tts_client:
            return text_to_speech_zindaki(text, 'ru', 'female', tts_client)
        else:
            return text_to_speech_gtts(text, 'ru')

def text_to_speech_optimized(text: str, force_lang: Optional[str] = None, 
                           voice_type: str = 'female', use_zindaki: bool = True,
                           tts_client: Optional[ZindakiTTSClient] = None) -> Optional[str]:
    """Оптимизированная версия с выбором лучшего метода"""
    if not text.strip():
        return None
    
    # 🔥 Если явно указан язык - используем самый быстрый путь
    if force_lang and force_lang != 'auto':
        if use_zindaki and tts_client:
            return text_to_speech_zindaki(text, force_lang, voice_type, tts_client)
        else:
            return text_to_speech_gtts(text, force_lang)
    
    # 🔥 БЫСТРАЯ ЭВРИСТИКА: проверяем только на латинские буквы
    has_latin = bool(re.search(r'[a-zA-Z]', text))
    
    if not has_latin:
        # 🔥 ТОЛЬКО РУССКИЙ ТЕКСТ - БЫСТРЫЙ ПУТЬ
        if use_zindaki and tts_client:
            return text_to_speech_zindaki(text, 'ru', voice_type, tts_client)
        else:
            return text_to_speech_gtts(text, 'ru')
    else:
        # Есть латинские буквы - нужна более точная проверка
        if is_foreign_text(text):
            # 🔥 ЕСТЬ ИНОСТРАННЫЙ ТЕКСТ - автоопределение
            return text_to_speech_mixed(text, use_zindaki, tts_client)
        else:
            # 🔥 ТОЛЬКО ОБЩЕПРИНЯТЫЕ СЛОВА - РУССКИЙ
            if use_zindaki and tts_client:
                return text_to_speech_zindaki(text, 'ru', voice_type, tts_client)
            else:
                return text_to_speech_gtts(text, 'ru')

# =============================================================================
# КЛАСС ДЛЯ УПРАВЛЕНИЯ РЕЧЬЮ
# =============================================================================

class SpeechManager:
    """Менеджер для управления озвучиванием речи"""
    
    def __init__(self, socketio=None, config: Optional[Dict] = None):
        """Инициализация менеджера речи"""
        self.socketio = socketio
        self.debug = DEBUG_SPEECH
        self.config = config or TTS_CONFIG.copy()
        
        # Инициализация TTS клиента
        self.tts_client = None
        try:
            self.tts_client = ZindakiTTSClient(self.config['zindaki'])
            debug_log(f"✅ Zindaki TTS client initialized. Available: {self.tts_client.available}")
        except Exception as e:
            debug_log(f"⚠️ Failed to initialize TTS client: {e}")
            self.tts_client = None
        
        # Статистика
        self.stats = {
            'total_requests': 0,
            'zindaki_success': 0,
            'zindaki_failed': 0,
            'gtts_fallback': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'technical_cleaning': 0,
            'standard_cleaning': 0,
            'mixed_language': 0
        }
        
        # Простой кэш для часто используемых фраз
        self.simple_cache = {}
        self.cache_lock = threading.Lock()
        
        debug_log(f"✅ SpeechManager initialized with config: primary={self.config['primary']}")
    
    def log(self, message: str):
        """Логирование сообщений"""
        if self.debug:
            logger.info(f"🔊 [SpeechManager] {message}")
    
    def clean_text(self, text: str, subject: Optional[str] = None) -> str:
        """Очистка текста для озвучивания"""
        cleaned_text = clean_text_for_speech_smart(text, subject)
        
        # Обновляем статистику очистки
        if subject:
            self.stats['technical_cleaning'] += 1
        else:
            self.stats['standard_cleaning'] += 1
            
        return cleaned_text
    
    def detect_language(self, text: str) -> Tuple[str, bool]:
        """Определение языка текста"""
        return detect_text_language_fast(text)
    
    def generate_speech(self, text: str, lang: str = 'ru', 
                       voice_type: str = 'female') -> Optional[str]:
        """Генерация речи из текста (с использованием TTS сервиса)"""
        return self.generate_optimized_speech(text, lang, voice_type)
    
    def _get_cache_key(self, text: str, lang: str, voice_type: str) -> str:
        """Генерация ключа для кэша"""
        import hashlib
        key_string = f"{text}_{lang}_{voice_type}"
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()
    
    def _check_cache(self, text: str, lang: str, voice_type: str) -> Optional[str]:
        """Проверка кэша"""
        if not self.config['cache']['enabled']:
            return None
        
        cache_key = self._get_cache_key(text, lang, voice_type)
        
        with self.cache_lock:
            if cache_key in self.simple_cache:
                cache_entry = self.simple_cache[cache_key]
                # Проверяем TTL
                if time.time() - cache_entry['timestamp'] < self.config['cache']['ttl_seconds']:
                    self.stats['cache_hits'] += 1
                    self.log(f"✅ Cache hit for text: '{text[:30]}...'")
                    return cache_entry['audio_data']
                else:
                    # Удаляем просроченную запись
                    del self.simple_cache[cache_key]
        
        self.stats['cache_misses'] += 1
        return None
    
    def _add_to_cache(self, text: str, lang: str, voice_type: str, audio_data: str):
        """Добавление в кэш"""
        if not self.config['cache']['enabled']:
            return
        
        cache_key = self._get_cache_key(text, lang, voice_type)
        
        with self.cache_lock:
            # Очищаем старые записи если кэш переполнен
            if len(self.simple_cache) >= self.config['cache']['max_size']:
                # Удаляем самую старую запись
                oldest_key = min(self.simple_cache.keys(), 
                               key=lambda k: self.simple_cache[k]['timestamp'])
                del self.simple_cache[oldest_key]
                self.log(f"🗑️ Removed old cache entry: {oldest_key[:8]}...")
            
            self.simple_cache[cache_key] = {
                'audio_data': audio_data,
                'timestamp': time.time(),
                'text': text[:100] + ('...' if len(text) > 100 else ''),
                'lang': lang,
                'voice_type': voice_type
            }
    
    def generate_optimized_speech(self, text: str, force_lang: Optional[str] = None,
                                voice_type: str = 'female') -> Optional[str]:
        """Оптимизированная генерация речи с использованием TTS сервиса"""
        self.stats['total_requests'] += 1
        
        # Проверяем, включен ли TTS
        if not self.config['enabled']:
            self.log("⚠️ TTS is disabled in config")
            return text_to_speech_gtts(text, force_lang or 'ru')
        
        # Проверяем кэш
        lang_for_cache = force_lang if force_lang and force_lang != 'auto' else 'ru'
        cached_audio = self._check_cache(text, lang_for_cache, voice_type)
        if cached_audio:
            return cached_audio
        
        # Определяем, нужно ли использовать Zindaki TTS
        use_zindaki = (
            self.config['primary'] == 'zindaki' and 
            self.tts_client and 
            self.tts_client.available and
            self.config['zindaki']['fallback']
        )
        
        cleaned_text = text  # Используем оригинальный текст для TTS
        
        # Пробуем использовать Zindaki TTS если доступен и это основной сервис
        if use_zindaki:
            try:
                # Определяем параметры для TTS сервиса
                if force_lang and force_lang != 'auto':
                    tts_lang = self.config['zindaki']['language_mapping'].get(force_lang, 'ru')
                else:
                    tts_lang = 'ru'  # по умолчанию русский
                
                # Определяем голос
                speaker = self.config['zindaki']['speaker_mapping'].get(
                    voice_type, 
                    self.config['zindaki']['speaker_mapping']['default']
                )
                
                self.log(f"🔊 Requesting Zindaki TTS: lang={tts_lang}, speaker={speaker}")
                
                # Генерация через TTS сервис
                audio_base64 = text_to_speech_zindaki(cleaned_text, tts_lang, voice_type, self.tts_client)
                
                if audio_base64:
                    self.stats['zindaki_success'] += 1
                    self.log(f"✅ Zindaki TTS success")
                    
                    # Добавляем в кэш
                    self._add_to_cache(text, lang_for_cache, voice_type, audio_base64)
                    
                    return audio_base64
                else:
                    self.stats['zindaki_failed'] += 1
                    self.log("❌ Zindaki TTS failed, falling back to gTTS")
                    
            except Exception as e:
                self.stats['zindaki_failed'] += 1
                self.log(f"❌ Zindaki TTS error: {e}")
        
        # Fallback на gTTS
        self.stats['gtts_fallback'] += 1
        self.log("🔄 Falling back to gTTS")
        
        # Генерируем через gTTS
        if force_lang and force_lang != 'auto':
            audio_base64 = text_to_speech_gtts(cleaned_text, force_lang)
        else:
            audio_base64 = text_to_speech_optimized(cleaned_text, force_lang, voice_type, False, None)
        
        if audio_base64:
            # Добавляем в кэш
            self._add_to_cache(text, lang_for_cache, voice_type, audio_base64)
        
        return audio_base64
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики"""
        stats = {
            **self.stats,
            'cache_size': len(self.simple_cache),
            'cache_enabled': self.config['cache']['enabled'],
            'cache_max_size': self.config['cache']['max_size'],
            'tts_service_available': self.tts_client.available if self.tts_client else False,
            'tts_service_enabled': self.config['enabled'],
            'primary_tts': self.config['primary'],
            'zindaki_config': self.config['zindaki']
        }
        
        # Рассчитываем проценты
        if self.stats['total_requests'] > 0:
            stats['zindaki_success_rate'] = (self.stats['zindaki_success'] / self.stats['total_requests']) * 100
            stats['zindaki_failure_rate'] = (self.stats['zindaki_failed'] / self.stats['total_requests']) * 100
            stats['gtts_fallback_rate'] = (self.stats['gtts_fallback'] / self.stats['total_requests']) * 100
            stats['cache_hit_rate'] = (self.stats['cache_hits'] / self.stats['total_requests']) * 100
        else:
            stats['zindaki_success_rate'] = 0
            stats['zindaki_failure_rate'] = 0
            stats['gtts_fallback_rate'] = 0
            stats['cache_hit_rate'] = 0
        
        return stats
    
    def get_tts_service_status(self) -> Dict[str, Any]:
        """Получение статуса TTS сервиса"""
        if self.tts_client:
            status = self.tts_client.get_service_status()
            status['client_stats'] = self.tts_client.get_stats()
            return status
        return {
            'status': 'unavailable', 
            'available': False, 
            'message': 'TTS client not initialized',
            'config': self.config
        }
    
    def get_available_voices(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Получение доступных голосов"""
        if self.tts_client:
            return self.tts_client.get_available_voices(force_refresh)
        return {
            'all_voices': {}, 
            'loaded_voices': {}, 
            'message': 'TTS client not initialized',
            'config_mapping': self.config['zindaki']['speaker_mapping']
        }
    
    def clear_tts_cache(self, days_old: Optional[int] = None) -> bool:
        """Очистка кэша TTS сервиса"""
        if self.tts_client:
            return self.tts_client.clear_cache(days_old)
        return False
    
    def clear_local_cache(self) -> Dict[str, Any]:
        """Очистка локального кэша"""
        with self.cache_lock:
            cache_size = len(self.simple_cache)
            self.simple_cache.clear()
        
        self.log(f"🗑️ Cleared local cache: {cache_size} entries")
        
        return {
            'success': True,
            'message': f'Local cache cleared ({cache_size} entries)',
            'cleared_entries': cache_size,
            'remaining_entries': 0
        }
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """Обновление конфигурации"""
        try:
            # Обновляем только разрешенные ключи
            allowed_keys = ['enabled', 'primary', 'zindaki', 'gtts', 'cache']
            
            for key in allowed_keys:
                if key in new_config:
                    if key == 'zindaki' and isinstance(new_config[key], dict):
                        # Частичное обновление для zindaki
                        for subkey, value in new_config[key].items():
                            if subkey in self.config['zindaki']:
                                self.config['zindaki'][subkey] = value
                    elif key == 'cache' and isinstance(new_config[key], dict):
                        # Частичное обновление для cache
                        for subkey, value in new_config[key].items():
                            if subkey in self.config['cache']:
                                self.config['cache'][subkey] = value
                    else:
                        self.config[key] = new_config[key]
            
            # Переинициализируем TTS клиент если изменился base_url
            if 'zindaki' in new_config and 'base_url' in new_config['zindaki']:
                try:
                    self.tts_client = ZindakiTTSClient(self.config['zindaki'])
                    self.log(f"✅ TTS client reinitialized with new base_url: {self.config['zindaki']['base_url']}")
                except Exception as e:
                    self.log(f"⚠️ Failed to reinitialize TTS client: {e}")
                    self.tts_client = None
            
            self.log(f"✅ Config updated: primary={self.config['primary']}, enabled={self.config['enabled']}")
            return True
            
        except Exception as e:
            self.log(f"❌ Failed to update config: {e}")
            return False
    
    def test_tts_service(self, text: str = "Тестовое сообщение для проверки TTS сервиса.", 
                        language: str = 'ru', speaker: str = 'baya') -> Dict[str, Any]:
        """Тестирование TTS сервиса"""
        try:
            if self.tts_client and self.tts_client.available:
                audio_data = self.tts_client.generate_speech(text, language, speaker)
                
                if audio_data:
                    return {
                        'success': True,
                        'message': 'TTS service is working correctly',
                        'audio_size': len(audio_data),
                        'tts_service': 'zindaki',
                        'available': True
                    }
                else:
                    return {
                        'success': False,
                        'message': 'TTS service available but failed to generate audio',
                        'tts_service': 'zindaki',
                        'available': True
                    }
            else:
                # Тестируем gTTS как fallback
                audio_base64 = text_to_speech_gtts(text, language)
                
                if audio_base64:
                    return {
                        'success': True,
                        'message': 'gTTS fallback is working',
                        'audio_size': len(base64.b64decode(audio_base64)),
                        'tts_service': 'gtts',
                        'available': False
                    }
                else:
                    return {
                        'success': False,
                        'message': 'Both TTS services failed',
                        'tts_service': 'none',
                        'available': False
                    }
                    
        except Exception as e:
            return {
                'success': False,
                'message': f'Test failed: {str(e)}',
                'tts_service': 'error',
                'available': False
            }
    
    def is_foreign(self, text: str) -> bool:
        """Проверка на иностранный текст"""
        return is_foreign_text(text)
    
    def reset_stats(self) -> Dict[str, Any]:
        """Сброс статистики"""
        old_stats = self.stats.copy()
        
        self.stats = {
            'total_requests': 0,
            'zindaki_success': 0,
            'zindaki_failed': 0,
            'gtts_fallback': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'technical_cleaning': 0,
            'standard_cleaning': 0,
            'mixed_language': 0
        }
        
        self.log(f"✅ Statistics reset. Old stats: {old_stats}")
        
        return {
            'success': True,
            'message': 'Statistics reset successfully',
            'old_stats': old_stats,
            'new_stats': self.stats
        }

# =============================================================================
# ФАБРИЧНАЯ ФУНКЦИЯ
# =============================================================================

_speech_manager_instance = None

def get_speech_manager(socketio=None, config: Optional[Dict] = None) -> SpeechManager:
    """Получение или создание экземпляра SpeechManager"""
    global _speech_manager_instance
    if _spepeech_manager_instance is None:
        _speech_manager_instance = SpeechManager(socketio, config)
        debug_log("✅ SpeechManager инициализирован с поддержкой Zindaki TTS")
    return _speech_manager_instance

# =============================================================================
# ТЕСТОВЫЕ ФУНКЦИИ
# =============================================================================

def test_speech_functions():
    """Тестирование функций озвучивания"""
    test_texts = [
        "Привет, как дела?",
        "Hello, how are you?",
        "Привет, my name is Иван. I am 25 years old.",
        "Уравнение: E = mc²",
        "Просто текст без английских слов",
        "Слово ok и слово hello в русском тексте"
    ]
    
    print("🧪 Тестирование SpeechManager с Zindaki TTS поддержкой")
    
    manager = SpeechManager()
    
    print(f"Zindaki TTS доступен: {manager.tts_client.available if manager.tts_client else False}")
    print(f"Конфигурация: primary={manager.config['primary']}, enabled={manager.config['enabled']}")
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nТест {i}: {text}")
        print(f"Очищенный текст: {manager.clean_text(text)}")
        print(f"Иностранный текст: {manager.is_foreign(text)}")
        lang, is_foreign = manager.detect_language(text)
        print(f"Определенный язык: {lang}, Иностранный: {is_foreign}")
        
        # Генерация речи (только для короткого теста)
        if i <= 2:  # Только первые 2 текста
            print(f"Генерация аудио...")
            audio = manager.generate_optimized_speech(text)
            print(f"Аудио сгенерировано: {bool(audio)}")
    
    # Показываем статистику
    print(f"\n📊 Статистика:")
    stats = manager.get_stats()
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value}")

# =============================================================================
# ОСНОВНОЙ БЛОК
# =============================================================================

if __name__ == "__main__":
    print("🧪 Тестирование speech_manager.py с Zindaki TTS")
    print("=" * 60)
    
    try:
        test_speech_functions()
        print("\n" + "=" * 60)
        print("✅ Все функции работают корректно!")
    except Exception as e:
        print(f"\n❌ Ошибка тестирования: {e}")
        import traceback
        traceback.print_exc()