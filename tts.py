from gtts import gTTS
import base64
from io import BytesIO
import logging
from typing import List, Tuple
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleTTS:
    def __init__(self, cache_dir: str = "static/audio"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def text_to_phonemes(self, text: str) -> List[Tuple[str, float]]:
        """Упрощенная генерация фонем (заглушка)"""
        vowels = {'а', 'е', 'ё', 'и', 'о', 'у', 'ы', 'э', 'ю', 'я'}
        phonemes = []
        for char in text.lower():
            if char in vowels:
                phonemes.append((char, 0.3))  # Гласные длиннее
            elif char.isalpha():
                phonemes.append((char, 0.1))  # Согласные короче
        return phonemes[:50]  # Ограничиваем количество

    def synthesize(self, text: str, language: str = 'ru') -> dict:
        """Синтез речи с псевдо-фонемами"""
        try:
            # Генерация аудио в оперативной памяти
            tts = gTTS(text=text, lang=language, slow=False)
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            # Генерация упрощенных фонем
            phonemes = self.text_to_phonemes(text)
            
            return {
                'audio': base64.b64encode(audio_buffer.read()).decode('utf-8'),
                'phonemes': phonemes,
                'text': text
            }
        except Exception as e:
            logger.error(f"Ошибка синтеза речи: {str(e)}")
            return {
                'error': str(e),
                'phonemes': [('а', 0.2)] * 3  # Фолбэк
            }

# Пример использования:
if __name__ == "__main__":
    tts = SimpleTTS()
    result = tts.synthesize("Привет, это тест синтеза речи")
    print(f"Аудио (base64): {result['audio'][:30]}...")
    print(f"Фонемы: {result['phonemes']}")