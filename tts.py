import os
import base64
from gtts import gTTS
from io import BytesIO
import logging
from typing import List, Tuple, Dict
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
        
        # Соответствие фонем и параметров анимации
        self.phoneme_params = {
            'а': ('mouth_aa', 0.3), 'о': ('mouth_oo', 0.3),
            'у': ('mouth_uu', 0.3), 'и': ('mouth_ee', 0.3),
            'э': ('mouth_ee', 0.3), 'ы': ('mouth_aa', 0.3),
            'е': ('mouth_ee', 0.3), 'ё': ('mouth_oo', 0.3),
            'ю': ('mouth_uu', 0.3), 'я': ('mouth_aa', 0.3),
            'м': ('mouth_mm', 0.2), 'п': ('mouth_pp', 0.2),
            'б': ('mouth_bb', 0.2), 'ф': ('mouth_ff', 0.2),
            'в': ('mouth_vv', 0.2), 'ш': ('mouth_sh', 0.2),
            'ж': ('mouth_zh', 0.2), 'с': ('mouth_ss', 0.2),
            'з': ('mouth_zz', 0.2), 'р': ('mouth_rr', 0.2),
            'л': ('mouth_ll', 0.2), 'н': ('mouth_nn', 0.2),
            'т': ('mouth_tt', 0.2), 'д': ('mouth_dd', 0.2),
            'к': ('mouth_kk', 0.2), 'г': ('mouth_gg', 0.2),
            'х': ('mouth_hh', 0.2), 'ч': ('mouth_ch', 0.2),
            'щ': ('mouth_sh', 0.2), 'ц': ('mouth_ss', 0.2),
            'й': ('mouth_ee', 0.2)
        }

    def text_to_phonemes(self, text: str) -> List[Tuple[str, float]]:
        """Конвертирует текст в последовательность фонем"""
        phonemes = []
        for char in text.lower():
            if char in self.phoneme_params:
                phonemes.append(self.phoneme_params[char])
        return phonemes[:100]  # Ограничение на длину

    def synthesize(self, text: str, language: str = 'ru') -> Dict:
        """Синтезирует речь и возвращает аудио + фонемы"""
        try:
            # Проверка кэша
            cache_key = f"{language}_{hash(text)}"
            cache_file = self.cache_dir / f"{cache_key}.wav"
            
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    audio_data = f.read()
                phonemes = self.text_to_phonemes(text)
                logger.info(f"Using cached audio for: {text[:50]}...")
            else:
                # Генерация нового аудио
                tts = gTTS(text=text, lang=language, slow=False)
                audio_buffer = BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                audio_data = audio_buffer.read()
                
                # Сохранение в кэш
                with open(cache_file, 'wb') as f:
                    f.write(audio_data)
                
                phonemes = self.text_to_phonemes(text)
                logger.info(f"Generated new audio for: {text[:50]}...")

            return {
                'audio': base64.b64encode(audio_data).decode('utf-8'),
                'phonemes': phonemes,
                'text': text,
                'from_cache': cache_file.exists()
            }
        except Exception as e:
            logger.error(f"TTS error: {str(e)}")
            return {
                'error': str(e),
                'phonemes': [('mouth_neutral', 0.2)] * 3
            }

# Пример использования
if __name__ == "__main__":
    tts = SimpleTTS()
    sample_text = "Привет, это тест синтеза речи"
    result = tts.synthesize(sample_text)
    print(f"Audio length: {len(result['audio'])} bytes")
    print(f"Phonemes: {result['phonemes']}")
