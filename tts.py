import os
import torch
import time
import logging
from typing import Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import soundfile as sf
from TTS.api import TTS
from scipy.io.wavfile import write as write_wav

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
        reference_voice: str = "static/reference_voice.wav"
    ):
        self.model_name = model_name
        self.device = device
        self.output_dir = Path(output_dir)
        self.reference_voice = reference_voice
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # Создаем директории
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Проверяем референсный голос
        if not Path(self.reference_voice).exists():
            logger.warning(f"Reference voice not found at {self.reference_voice}")
            self._generate_default_voice()
        
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
            # Используем английский для генерации, так как он более стабилен
            tts = TTS(model_name="tts_models/en/ljspeech/vits").to(self.device)
            tts.tts_to_file(
                text="This is a default reference voice for the system.",
                file_path=str(self.reference_voice)
            logger.info(f"Default voice saved to {self.reference_voice}")
        except Exception as e:
            logger.error(f"Failed to generate default voice: {str(e)}")
            raise

    def generate_speech(
        self,
        text: str,
        voice_file: Optional[str] = None,
        output_file: Optional[str] = None,
        language: str = "ru",
        speed: float = 1.0,
        emotion: str = "neutral"
    ) -> Tuple[str, float]:
        """
        Генерация аудио из текста
        
        Args:
            text: Текст для озвучки
            voice_file: Путь к файлу с референсным голосом
            output_file: Имя выходного файла (без директории)
            language: Язык (ru/en/de/fr/etc)
            speed: Скорость воспроизведения (0.5-2.0)
            emotion: Эмоциональная окраска (neutral/happy/sad/angry)
            
        Returns:
            tuple: (путь к файлу, длительность в секундах)
        """
        start_time = time.time()
        voice_file = voice_file or self.reference_voice
        output_file = output_file or f"tts_{int(start_time)}.wav"
        output_path = str(self.output_dir / output_file)
        
        try:
            # Генерация аудио
            self.tts.tts_to_file(
                text=text,
                speaker_wav=voice_file,
                language=language,
                file_path=output_path,
                speed=speed,
                emotion=emotion
            )
            
            # Получаем длительность аудио
            duration = self._get_audio_duration(output_path)
            
            logger.info(f"Generated speech: {text[:50]}... (duration: {duration:.2f}s)")
            return output_path, duration
            
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

    def _get_audio_duration(self, file_path: str) -> float:
        """Получение длительности аудиофайла"""
        try:
            with sf.SoundFile(file_path) as f:
                return len(f) / f.samplerate
        except Exception as e:
            logger.warning(f"Could not get duration: {str(e)}")
            return 0.0

    def generate_batch(self, texts: list, output_prefix: str = "batch") -> list:
        """
        Пакетная генерация аудио
        
        Args:
            texts: Список текстов для озвучки
            output_prefix: Префикс для имен файлов
            
        Returns:
            list: Список кортежей (путь к файлу, длительность)
        """
        results = []
        for i, text in enumerate(texts):
            try:
                output_file = f"{output_prefix}_{i}.wav"
                result = self.generate_speech(text, output_file=output_file)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch item {i}: {str(e)}")
                results.append((None, 0.0))
        return results

    def text_to_waveform(self, text: str, **kwargs) -> Tuple[np.ndarray, int]:
        """
        Генерация аудио в виде waveform (для стриминга)
        
        Returns:
            tuple: (audio_array, sample_rate)
        """
        try:
            # Создаем временный файл
            temp_file = f"temp_{time.time()}.wav"
            self.generate_speech(text, output_file=temp_file, **kwargs)
            
            # Читаем обратно в массив
            import soundfile as sf
            data, samplerate = sf.read(str(self.output_dir / temp_file))
            
            # Удаляем временный файл
            (self.output_dir / temp_file).unlink(missing_ok=True)
            
            return data, samplerate
            
        except Exception as e:
            logger.error(f"Error in waveform generation: {str(e)}")
            raise

    def cleanup(self):
        """Очистка сгенерированных файлов старше 24 часов"""
        now = time.time()
        for f in self.output_dir.glob("*.wav"):
            if f.is_file() and (now - f.stat().st_mtime) > 86400:  # 24 часа
                try:
                    f.unlink()
                    logger.info(f"Deleted old file: {f.name}")
                except Exception as e:
                    logger.warning(f"Could not delete {f.name}: {str(e)}")

# Пример использования
if __name__ == "__main__":
    # Инициализация TTS
    tts = TextToSpeech()
    
    # Пример генерации речи
    text = "Привет, это тестовое сообщение для проверки работы системы."
    audio_file, duration = tts.generate_speech(text)
    print(f"Generated audio: {audio_file}, duration: {duration:.2f}s")
    
    # Пример пакетной генерации
    texts = [
        "Первое сообщение для озвучки.",
        "Второе сообщение, более длинное и содержательное.",
        "Третье сообщение с важной информацией."
    ]
    results = tts.generate_batch(texts)
    for file, dur in results:
        print(f"- {file} ({dur:.2f}s)")
