# tts.py
import re
import io
import base64
from gtts import gTTS
from core import debug_log, TECHNICAL_SUPPORT_ENABLED

def is_foreign_text(text: str) -> bool:
    """
    Быстрая проверка, содержит ли текст иностранные (не общепринятые) слова.
    Используется для выбора языка озвучки.
    """
    if not text:
        return False
    
    # Быстрая проверка первых 200 символов
    sample = text[:200]
    
    # 1. Ищем латинские буквы
    if not re.search(r'[a-zA-Z]', sample):
        return False  # Нет латинских букв — точно не иностранный
    
    # 2. Игнорируем общепринятые английские слова в русском контексте
    common_english_in_russian = {
        'ok', 'hello', 'hi', 'yes', 'no', 'bye', 'sorry', 'please',
        'thank', 'thanks', 'okay', 'cool', 'wow', 'nice', 'good', 'bad',
        'start', 'stop', 'pause', 'play', 'next', 'back', 'menu'
    }
    
    # Извлекаем все слова из латиницы длиной >=2
    words = re.findall(r'\b[a-zA-Z]{2,}\b', sample.lower())
    if not words:
        return False
    
    # Фильтруем общепринятые
    foreign_words = [w for w in words if w not in common_english_in_russian]
    unique_foreign_words = set(foreign_words)
    
    # Считаем иностранным, если есть хотя бы 2 уникальных необщепринятых слова
    return len(unique_foreign_words) >= 2

def clean_text_for_speech(text: str) -> str:
    """
    Очистка текста для озвучки (гуманитарные предметы).
    Удаляет форматирование, оставляет только читаемый текст.
    """
    if not text:
        return ""
    
    # Удаляем markdown-разметку
    text = re.sub(r'[\*\_~`]{1,}', '', text)
    # Удаляем заголовки в стиле Markdown
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    # Удаляем списки
    text = re.sub(r'^[\-\*\+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    # Удаляем цитаты
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
    # Удаляем горизонтальные линии
    text = re.sub(r'^\-{3,}$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\*{3,}$', '', text, flags=re.MULTILINE)
    # Удаляем HTML-теги
    text = re.sub(r'<[^>]+>', '', text)
    # Нормализуем пробелы и переводы строк
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def clean_text_for_speech_technical(text: str, subject: str = "") -> str:
    """
    Очистка текста для озвучки технических предметов.
    Сохраняет ключевые символы: =, +, -, /, степени, греческие буквы.
    """
    if not text:
        return ""
    
    # Сохраняем технические символы и формулы
    # Удаляем только визуальное форматирование, но не содержание
    text = re.sub(r'[\*\_~`]{2,}', '', text)  # Удаляем только повторяющиеся маркеры
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\-\*\+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'<[^>]+>', '', text)
    # Нормализуем только лишние пробелы, но сохраняем структуру формул
    text = re.sub(r' {2,}', ' ', text)
    text = text.strip()
    
    return text

def text_to_speech(text: str, lang: str = 'ru') -> str:
    """
    Основная функция озвучки через gTTS.
    Возвращает base64-закодированный MP3.
    """
    if not text or not text.strip():
        return None
    
    try:
        # Используем gTTS без проверки языка для скорости
        tts = gTTS(text=text.strip(), lang=lang, slow=False, lang_check=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return base64.b64encode(mp3_fp.read()).decode('utf-8')
    except Exception as e:
        debug_log(f"Error in text_to_speech: {e}")
        # Fallback — используем русский язык
        try:
            tts = gTTS(text=text.strip(), lang='ru', slow=False, lang_check=False)
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            return base64.b64encode(mp3_fp.read()).decode('utf-8')
        except Exception as e2:
            debug_log(f"Fallback error in text_to_speech: {e2}")
            return None

def text_to_speech_mixed(text: str) -> str:
    """
    Озвучка смешанного текста (кириллица + латиница).
    Разбивает текст на фрагменты по языку и озвучивает по отдельности.
    """
    if not text or not text.strip():
        return text_to_speech(text, 'ru')  # Fallback
    
    # Паттерн для захвата последовательных блоков одного языка
    pattern = r'([а-яА-ЯёЁ][а-яА-ЯёЁ\s.,!?;:\-\'"()]*|[a-zA-Z][a-zA-Z\s.,!?;:\-\'"()]*)'
    fragments = re.findall(pattern, text)
    
    if not fragments:
        return text_to_speech(text, 'ru')
    
    audio_chunks = []
    for fragment in fragments:
        fragment = fragment.strip()
        if not fragment:
            continue
        
        # Определяем язык фрагмента
        has_cyrillic = bool(re.search(r'[а-яА-ЯёЁ]', fragment))
        lang = 'ru' if has_cyrillic else 'en'
        
        try:
            tts = gTTS(text=fragment, lang=lang, slow=False, lang_check=False)
            chunk_fp = io.BytesIO()
            tts.write_to_fp(chunk_fp)
            chunk_fp.seek(0)
            audio_chunks.append(chunk_fp.read())
            
            # Короткая пауза между фрагментами (50ms тишины при 16kHz)
            silence = bytes([0] * 800)
            audio_chunks.append(silence)
        except Exception as e:
            debug_log(f"Error processing fragment '{fragment[:30]}...': {e}")
            continue
    
    if audio_chunks:
        # Убираем последнюю паузу
        if len(audio_chunks) > 1 and audio_chunks[-1] == bytes([0] * 800):
            audio_chunks.pop()
        
        combined_audio = b''.join(audio_chunks)
        return base64.b64encode(combined_audio).decode('utf-8')
    
    return None

def smart_text_to_speech(text: str, subject: str = None) -> str:
    """
    Умная озвучка: автоматически выбирает режим в зависимости от текста и предмета.
    """
    if not text:
        return None
    
    # Очистка текста
    if TECHNICAL_SUPPORT_ENABLED and subject:
        from technical_subjects import is_technical_subject
        if is_technical_subject(subject):
            cleaned_text = clean_text_for_speech_technical(text, subject)
        else:
            cleaned_text = clean_text_for_speech(text)
    else:
        cleaned_text = clean_text_for_speech(text)
    
    if not cleaned_text:
        return None
    
    # Анализ языка
    has_latin = bool(re.search(r'[a-zA-Z]', cleaned_text))
    has_cyrillic = bool(re.search(r'[а-яА-ЯёЁ]', cleaned_text))
    
    # Смешанный режим
    if has_latin and has_cyrillic:
        return text_to_speech_mixed(cleaned_text)
    # Только латиница
    elif has_latin and not has_cyrillic:
        return text_to_speech(cleaned_text, 'en')
    # Только кириллица (или цифры/знаки)
    else:
        return text_to_speech(cleaned_text, 'ru')
