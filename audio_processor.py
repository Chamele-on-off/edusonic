import os
import logging
import asyncio
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import whisper
from transformers import pipeline
import numpy as np
import torchaudio
import torch

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        self.model = None
        self.classifier = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._initialize_models()

    def _initialize_models(self):
        """Ленивая инициализация моделей для экономии памяти"""
        try:
            logger.info("Initializing Whisper model...")
            self.model = whisper.load_model("small", device="cuda" if torch.cuda.is_available() else "cpu")
            
            logger.info("Initializing question classifier...")
            self.classifier = pipeline(
                "text-classification",
                model="cointegrated/rubert-tiny2-cedr-emotion-detection",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise

    async def transcribe(self, audio_path: str) -> Optional[str]:
        """Транскрибация аудио в текст"""
        if not self.model:
            self._initialize_models()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                lambda: self.model.transcribe(audio_path, language="ru")
            )
            return result["text"].strip()
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            return None

    async def transcribe_raw_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Optional[str]:
        """Транскрибация сырого аудио (numpy array)"""
        if not self.model:
            self._initialize_models()

        try:
            # Конвертируем numpy array в torch tensor
            audio_tensor = torch.from_numpy(audio_data).float()
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Сохраняем во временный файл
            temp_file = "static/tmp/audio_temp.wav"
            torchaudio.save(temp_file, audio_tensor, sample_rate)
            
            # Транскрибируем
            result = await self.transcribe(temp_file)
            os.remove(temp_file)
            return result
        except Exception as e:
            logger.error(f"Raw audio transcription failed: {str(e)}")
            return None

    async def is_question(self, text: str) -> bool:
        """Определение, является ли текст вопросом"""
        if not text or not text.strip():
            return False
            
        # Простая проверка по знаку вопроса
        if text.endswith('?'):
            return True

        # Более сложный анализ с помощью модели
        if not self.classifier:
            self._initialize_models()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                lambda: self.classifier(text)
            )
            return result[0]['label'] == 'question'
        except Exception as e:
            logger.error(f"Question detection failed: {str(e)}")
            return False

    async def process_real_time_audio(self, audio_stream: bytes) -> Optional[str]:
        """Обработка аудио потока в реальном времени"""
        try:
            # Конвертируем байты в numpy array
            audio_data = np.frombuffer(audio_stream, dtype=np.float32)
            return await self.transcribe_raw_audio(audio_data)
        except Exception as e:
            logger.error(f"Real-time audio processing failed: {str(e)}")
            return None

# Пример использования
if __name__ == "__main__":
    async def main():
        processor = AudioProcessor()
        text = await processor.transcribe("test_audio.wav")
        print(f"Transcribed text: {text}")
        is_q = await processor.is_question(text)
        print(f"Is question: {is_q}")

    asyncio.run(main())
