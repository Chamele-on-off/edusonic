import os
import time
import base64
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from flask_socketio import emit
import torch
from TTS.api import TTS
from phonemizer import phonemize
from phonemizer.separator import Separator
import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tts.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TextToSpeech:
    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        output_dir: str = "static/audio"
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=2)
        self.tts = None
        self._load_model()

    def _load_model(self):
        """Загрузка TTS модели"""
        try:
            logger.info(f"Загрузка модели TTS {self.model_name}...")
            self.tts = TTS(model_name=self.model_name, progress_bar=False)
            logger.info("Модель TTS загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели TTS: {str(e)}")
            raise

    def generate_speech(self, text: str, language: str = "ru") -> dict:
        """Генерация речи из текста"""
        try:
            output_file = self.output_dir / f"tts_{int(time.time())}.wav"
            
            self.tts.tts_to_file(
                text=text,
                file_path=str(output_file),
                language=language
            )

            with open(output_file, 'rb') as f:
                audio_data = f.read()
            
            phonemes = self._extract_phonemes(text)
            os.remove(output_file)

            return {
                'audio': base64.b64encode(audio_data).decode('utf-8'),
                'phonemes': phonemes
            }
        except Exception as e:
            logger.error(f"Ошибка генерации речи: {str(e)}")
            return {'error': str(e)}

    def _extract_phonemes(self, text: str) -> list:
        """Извлечение фонем из текста"""
        try:
            phonemes = phonemize(
                text,
                language='ru',
                backend='espeak',
                separator=Separator(phone=' ', word='|', syllable='')
            )
            return [(p, 0.2) for p in phonemes.split()]  # Упрощенный тайминг
        except Exception as e:
            logger.error(f"Ошибка извлечения фонем: {str(e)}")
            return [('а', 0.2)] * len(text.split())

    def generate_stream(self, text: str, callback: callable):
        """Потоковая генерация речи"""
        self._executor.submit(
            self._generate_in_thread,
            text,
            callback
        )

    def _generate_in_thread(self, text: str, callback: callable):
        """Генерация в отдельном потоке"""
        try:
            result = self.generate_speech(text)
            callback(result)
        except Exception as e:
            logger.error(f"Ошибка в потоковой генерации: {str(e)}")
            callback({'error': str(e)})

    def shutdown(self):
        """Очистка ресурсов"""
        self._executor.shutdown()
        logger.info("TextToSpeech остановлен")