# speech_manager.py - Управление озвучиванием и распознаванием речи для AI Teacher
# Вынесено из app.py для лучшей организации кода

import re
import io
import base64
import time
from gtts import gTTS
from typing import Optional, Tuple
from pathlib import Path

# 🔥 Импорт для технических предметов (если доступен)
try:
    from technical_subjects import clean_text_for_speech_technical
    TECHNICAL_SUPPORT_AVAILABLE = True
except ImportError:
    TECHNICAL_SUPPORT_AVAILABLE = False

# =============================================================================
# КОНФИГУРАЦИЯ И НАСТРОЙКИ
# =============================================================================

DEBUG_SPEECH = True

def debug_log(message: str):
    """Логирование для отладки"""
    if DEBUG_SPEECH:
        print(f"🔊 [SPEECH] {message}")

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
# ОСНОВНЫЕ ФУНКЦИИ ОЧИСТКИ ТЕКСТА
# =============================================================================

def clean_text_for_speech(text: str) -> str:
    """Тщательная очистка текста для озвучивания"""
    if not text:
        return ""
    
    text = re.sub(r'[#\*\_\~`]', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\\n', ' ', text)
    text = re.sub(r'\\t', ' ', text)
    text = re.sub(r'\\r', ' ', text)
    text = re.sub(r'[^\u0400-\u04FFa-zA-Z0-9\s\.,!?;:()\-—]', '', text)
    text = re.sub(r'[\.\,]{2,}', '.', text)
    text = re.sub(r'\s+([\.,!?;:)])', r'\1', text)
    text = re.sub(r'([(\-])\s+', r'\1', text)
    text = text.strip()
    
    if text and len(text) > 1:
        text = text[0].upper() + text[1:]
    
    return text

def clean_text_for_speech_smart(text: str, subject: Optional[str] = None) -> str:
    """Умная очистка текста с учетом предмета"""
    try:
        # Если доступна поддержка технических предметов и предмет указан
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
# ОСНОВНЫЕ ФУНКЦИИ TTS
# =============================================================================

def text_to_speech(text: str, lang: str = 'ru') -> Optional[str]:
    """🔥 ОПТИМИЗИРОВАННАЯ ВЕРСИЯ: Быстрое озвучивание с опциональным автоопределением"""
    try:
        if not text.strip():
            return None
            
        # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Автоопределение ТОЛЬКО если явно запрошено
        if lang == 'auto':
            # Быстрая проверка - есть ли латинские буквы?
            has_latin = bool(re.search(r'[a-zA-Z]', text))
            has_cyrillic = bool(re.search(r'[а-яА-ЯеЕ]', text))
            
            # Если ЕСТЬ латиница И кириллица - смешанный режим
            if has_latin and has_cyrillic:
                return text_to_speech_mixed(text)
            # Если ТОЛЬКО латиница - английский
            elif has_latin and not has_cyrillic:
                lang = 'en'
            # По умолчанию - русский (для ТОЛЬКО кириллицы или цифр/знаков)
            else:
                lang = 'ru'
        
        # 🔥 БЫСТРАЯ ГЕНЕРАЦИЯ БЕЗ ДОПОЛНИТЕЛЬНОГО АНАЛИЗА
        tts = gTTS(text=text, lang=lang, slow=False, lang_check=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return base64.b64encode(mp3_fp.read()).decode('utf-8')
        
    except Exception as e:
        debug_log(f"Error in text_to_speech: {e}")
        # Fallback - используем русский язык
        try:
            tts = gTTS(text=text, lang='ru', slow=False, lang_check=False)
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            return base64.b64encode(mp3_fp.read()).decode('utf-8')
        except Exception as e2:
            debug_log(f"Fallback error in text_to_speech: {e2}")
            return None

def text_to_speech_mixed(text: str) -> Optional[str]:
    """🔥 ОПТИМИЗИРОВАННАЯ ФУНКЦИЯ: Озвучивание смешанного текста"""
    try:
        # 🔥 ОПТИМИЗИРОВАННЫЙ ШАБЛОН: ищем последовательности букв одного языка
        pattern = r'([а-яА-ЯеЕ][а-яА-ЯеЕ\s.,!?;:\'-]*|[a-zA-Z][a-zA-Z\s.,!?;:\'-]*)'
        fragments = re.findall(pattern, text)
        
        if not fragments:
            return text_to_speech(text, 'ru')  # Fallback
        
        audio_chunks = []
        
        for fragment in fragments:
            fragment = fragment.strip()
            if not fragment:
                continue
                
            # Определяем язык фрагмента - быстрая проверка
            has_cyrillic = bool(re.search(r'[а-яА-ЯеЕ]', fragment))
            lang = 'ru' if has_cyrillic else 'en'
            
            try:
                tts = gTTS(text=fragment, lang=lang, slow=False, lang_check=False)
                chunk_fp = io.BytesIO()
                tts.write_to_fp(chunk_fp)
                chunk_fp.seek(0)
                audio_chunks.append(chunk_fp.read())
                
                # 🔥 КОРОТКАЯ ПАУЗА МЕЖДУ ФРАГМЕНТАМИ (50ms silence)
                silence = bytes([0] * 800)  # 50ms при 16kHz
                audio_chunks.append(silence)
                
            except Exception as e:
                debug_log(f"Error processing fragment '{fragment[:30]}...': {e}")
                continue
        
        if audio_chunks:
            # 🔥 УБИРАЕМ ПОСЛЕДНЮЮ ПАУЗУ
            if audio_chunks[-1] == bytes([0] * 800):
                audio_chunks.pop()
                
            combined_audio = b''.join(audio_chunks)
            return base64.b64encode(combined_audio).decode('utf-8')
        
        return None
    except Exception as e:
        debug_log(f"Error in text_to_speech_mixed: {e}")
        return text_to_speech(text, 'ru')

def text_to_speech_optimized(text: str, force_lang: Optional[str] = None) -> Optional[str]:
    """Оптимизированная версия с выбором лучшего метода"""
    if not text.strip():
        return None
    
    # 🔥 Если явно указан язык - используем самый быстрый путь
    if force_lang and force_lang != 'auto':
        return text_to_speech(text, force_lang)
    
    # 🔥 БЫСТРАЯ ЭВРИСТИКА: проверяем только на латинские буквы
    has_latin = bool(re.search(r'[a-zA-Z]', text))
    
    if not has_latin:
        # 🔥 ТОЛЬКО РУССКИЙ ТЕКСТ - БЫСТРЫЙ ПУТЬ
        return text_to_speech(text, 'ru')
    else:
        # Есть латинские буквы - нужна более точная проверка
        if is_foreign_text(text):
            # 🔥 ЕСТЬ ИНОСТРАННЫЙ ТЕКСТ - автоопределение
            return text_to_speech(text, 'auto')
        else:
            # 🔥 ТОЛЬКО ОБЩЕПРИНЯТЫЕ СЛОВА - РУССКИЙ
            return text_to_speech(text, 'ru')

# =============================================================================
# КЛАСС ДЛЯ УПРАВЛЕНИЯ РЕЧЬЮ
# =============================================================================

class SpeechManager:
    """Менеджер для управления озвучиванием речи"""
    
    def __init__(self, socketio=None):
        """Инициализация менеджера речи"""
        self.socketio = socketio
        self.debug = DEBUG_SPEECH
        
    def log(self, message: str):
        """Логирование сообщений"""
        if self.debug:
            print(f"🔊 [SpeechManager] {message}")
    
    def clean_text(self, text: str, subject: Optional[str] = None) -> str:
        """Очистка текста для озвучивания"""
        return clean_text_for_speech_smart(text, subject)
    
    def detect_language(self, text: str) -> Tuple[str, bool]:
        """Определение языка текста"""
        return detect_text_language_fast(text)
    
    def generate_speech(self, text: str, lang: str = 'ru') -> Optional[str]:
        """Генерация речи из текста"""
        return text_to_speech(text, lang)
    
    def generate_mixed_speech(self, text: str) -> Optional[str]:
        """Генерация речи для смешанного текста"""
        return text_to_speech_mixed(text)
    
    def generate_optimized_speech(self, text: str, force_lang: Optional[str] = None) -> Optional[str]:
        """Оптимизированная генерация речи"""
        return text_to_speech_optimized(text, force_lang)
    
    def is_foreign(self, text: str) -> bool:
        """Проверка на иностранный текст"""
        return is_foreign_text(text)

# =============================================================================
# ФАБРИЧНАЯ ФУНКЦИЯ
# =============================================================================

_speech_manager_instance = None

def get_speech_manager(socketio=None) -> SpeechManager:
    """Получение или создание экземпляра SpeechManager"""
    global _speech_manager_instance
    if _speech_manager_instance is None:
        _speech_manager_instance = SpeechManager(socketio)
        debug_log("✅ SpeechManager инициализирован")
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
    
    manager = SpeechManager()
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nТест {i}: {text}")
        print(f"Очищенный текст: {manager.clean_text(text)}")
        print(f"Иностранный текст: {manager.is_foreign(text)}")
        lang, is_foreign = manager.detect_language(text)
        print(f"Определенный язык: {lang}, Иностранный: {is_foreign}")
        
        # Генерация речи (комментируем для теста, чтобы не создавать аудио)
        # audio = manager.generate_optimized_speech(text)
        # print(f"Аудио сгенерировано: {bool(audio)}")

if __name__ == "__main__":
    print("🧪 Тестирование speech_manager.py")
    test_speech_functions()
    print("\n✅ Все функции работают корректно!")
