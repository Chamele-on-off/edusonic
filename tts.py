import os
import torch
import time
import logging
import base64
import numpy as np
import soundfile as sf
from typing import Tuple, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from TTS.api import TTS
from phonemizer import phonemize
from phonemizer.separator import Separator
import asyncio

# Настройка логирования
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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "static/audio",
        reference_voice: str = "static/reference/reference_voice.wav"
    ):
        self.model_name = model_name
        self.device = device
        self.output_dir = Path(output_dir)
        self.reference_voice = Path(reference_voice)
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # Создаем директории
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Проверяем референсный голос
        if not self.reference_voice.exists():
            logger.warning(f"Reference voice not found at {self.reference_voice}")
            self._generate_default_voice()
        
        self.tts = None
        self._load_model()
        self._warmup()

    def _load_model(self):
        """Загрузка TTS модели"""
        logger.info(f"Loading TTS model {self.model_name} on {self.device}...")
        
        try:
            self.tts = TTS(model_name=self.model_name, progress_bar=False).to(self.device)
            logger.info("TTS model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load TTS model: {str(e)}")
            raise

    def _warmup(self):
        """Прогрев модели тестовым запросом"""
        logger.info("Warming up TTS model...")
        try:
            self.generate_speech(
                text="Привет, это тестовое сообщение для прогрева модели.",
                voice_file=self.reference_voice,
                output_file="warmup.wav",
                language="ru"
            )
            logger.info("TTS model warmed up successfully")
        except Exception as e:
            logger.warning(f"Warmup failed: {str(e)}")

    def _generate_default_voice(self):
        """Генерация стандартного референсного голоса"""
        logger.info("Generating default reference voice...")
        try:
            self.reference_voice.parent.mkdir(parents=True, exist_ok=True)
            temp_tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            temp_tts.tts_to_file(
                text="Это стандартный референсный голос для системы.",
                file_path=str(self.reference_voice),
                speaker_wav=str(self.reference_voice),  # Будет создан новый голос
                language="ru"
            )
            logger.info(f"Default voice saved to {self.reference_voice}")
        except Exception as e:
            logger.error(f"Failed to generate default voice: {str(e)}")
            raise

    def _extract_phonemes(self, text: str) -> List[Tuple[str, float]]:
        """Извлечение фонем с таймингами (упрощенная версия)"""
        try:
            # Разбиваем текст на фонемы
            phonemes = phonemize(
                text,
                language='ru',
                backend='espeak',
                separator=Separator(phone=' ', word='|', syllable='')
            ).split()
            
            # Генерируем приблизительные тайминги (0.2 сек на фонему)
            phoneme_timings = []
            current_time = 0.0
            for phone in phonemes:
                duration = 0.2  # Базовая длительность
                # Корректировка для некоторых фонем
                if phone in ['а', 'о', 'у', 'и', 'э', 'ы']:
                    duration = 0.3
                elif phone in ['п', 'б', 'т', 'д', 'к', 'г']:
                    duration = 0.1
                
                phoneme_timings.append((phone, current_time))
                current_time += duration
            
            return [(phone, duration) for phone, duration in zip(phonemes, [0.2]*len(phonemes))]
        except Exception as e:
            logger.error(f"Phoneme extraction failed: {str(e)}")
            return [('а', 0.2)] * len(text.split())  # Fallback

    def generate_speech(
        self,
        text: str,
        voice_file: Optional[str] = None,
        output_file: Optional[str] = None,
        language: str = "ru",
        speed: float = 1.0,
        emotion: str = "neutral"
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Генерация аудио из текста с фонемами
        
        Args:
            text: Текст для озвучки
            voice_file: Путь к файлу с референсным голосом
            output_file: Имя выходного файла (без директории)
            language: Язык (ru/en/de/fr/etc)
            speed: Скорость воспроизведения (0.5-2.0)
            emotion: Эмоциональная окраска (neutral/happy/sad/angry)
            
        Returns:
            tuple: (путь к файлу, длительность в секундах, список фонем с таймингами)
        """
        start_time = time.time()
        voice_file = voice_file or self.reference_voice
        output_file = output_file or f"tts_{int(start_time)}.wav"
        output_path = str(self.output_dir / output_file)
        
        try:
            # Генерация аудио
            self.tts.tts_to_file(
                text=text,
                speaker_wav=str(voice_file),
                language=language,
                file_path=output_path,
                speed=speed,
                emotion=emotion
            )
            
            # Получаем длительность аудио
            duration = self._get_audio_duration(output_path)
            
            # Извлекаем фонемы
            phonemes = self._extract_phonemes(text)
            
            logger.info(f"Generated speech: {text[:50]}... (duration: {duration:.2f}s, phonemes: {len(phonemes)})")
            return output_path, duration, phonemes
            
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            raise

    async def generate_speech_async(self, *args, **kwargs):
        """Асинхронная версия генерации речи"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.generate_speech(*args, **kwargs)
        )

    def generate_with_phonemes(
        self,
        text: str,
        return_base64: bool = True
    ) -> Tuple[bytes, List[Tuple[str, float]]]:
        """
        Генерация аудио с фонемами (для WebSocket)
        
        Returns:
            tuple: (аудио в base64, список фонем с таймингами)
        """
        try:
            # Генерируем временный файл
            temp_file = f"temp_{int(time.time())}.wav"
            _, _, phonemes = self.generate_speech(text, output_file=temp_file)
            
            # Читаем аудио файл
            with open(self.output_dir / temp_file, 'rb') as f:
                audio_data = f.read()
            
            # Удаляем временный файл
            (self.output_dir / temp_file).unlink()
            
            if return_base64:
                return base64.b64encode(audio_data).decode('utf-8'), phonemes
            return audio_data, phonemes
            
        except Exception as e:
            logger.error(f"Error in generate_with_phonemes: {str(e)}")
            raise

    def _get_audio_duration(self, file_path: str) -> float:
        """Получение длительности аудиофайла"""
        try:
            with sf.SoundFile(file_path) as f:
                return len(f) / f.samplerate
        except Exception as e:
            logger.warning(f"Could not get duration: {str(e)}")
            return 0.0

    def cleanup(self, max_age_hours: int = 24):
        """Очистка сгенерированных файлов старше указанного возраста"""
        now = time.time()
        for f in self.output_dir.glob("*.wav"):
            if f.is_file() and (now - f.stat().st_mtime) > max_age_hours * 3600:
                try:
                    f.unlink()
                    logger.info(f"Deleted old file: {f.name}")
                except Exception as e:
                    logger.warning(f"Could not delete {f.name}: {str(e)}")

# Пример использования
if __name__ == "__main__":
    # Инициализация TTS
    tts = TextToSpeech()
    
    # Пример генерации речи с фонемами
    text = "Привет, это тестовое сообщение для проверки работы системы."
    audio_data, phonemes = tts.generate_with_phonemes(text)
    print(f"Generated audio (size: {len(audio_data)} bytes)")
    print("Phonemes:", phonemes)
