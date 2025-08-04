import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor
import ffmpeg
import time
import asyncio
import subprocess

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wav2lip.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Wav2LipInference:
    def __init__(
        self,
        checkpoint_path: str = "wav2lip/pretrained/wav2lip.pth",
        face_detector_path: str = "wav2lip/pretrained/mobilenet.pth",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        tmp_dir: str = "static/tmp",
        img_size: int = 96,
        fps: float = 25.0
    ):
        self.device = device
        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(exist_ok=True, parents=True)
        self.img_size = img_size
        self.fps = fps
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # Проверка наличия моделей
        self.checkpoint_path = self._validate_path(checkpoint_path)
        self.face_detector_path = self._validate_path(face_detector_path)
        
        # Ленивая загрузка моделей
        self.model = None
        self.face_detector = None
        
        logger.info(f"Wav2Lip initialized (device: {device})")

    def _validate_path(self, path: str) -> Path:
        """Проверка наличия файла модели"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        return path

    def _load_model(self):
        """Загрузка модели Wav2Lip"""
        if self.model is not None:
            return
            
        try:
            from wav2lip.models import Wav2Lip
            logger.info("Loading Wav2Lip model...")
            self.model = Wav2Lip()
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model = self.model.to(self.device).eval()
            logger.info("Wav2Lip model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Wav2Lip model: {str(e)}")
            raise

    def _load_face_detector(self):
        """Загрузка детектора лиц"""
        if self.face_detector is not None:
            return
            
        try:
            from wav2lip.face_detection import FaceDetector
            logger.info("Loading face detector...")
            self.face_detector = FaceDetector(
                mobilenet_path=str(self.face_detector_path),
                device=self.device
            )
            logger.info("Face detector loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load face detector: {str(e)}")
            raise

    async def process(
        self,
        audio_path: str,
        face_path: str,
        output_path: Optional[str] = None,
        pads: Tuple[int, int, int, int] = (0, 10, 0, 0),
        resize_factor: int = 1,
        rotate: bool = False,
        crop: Tuple[int, int, int, int] = (0, -1, 0, -1)
    ) -> str:
        """
        Основной метод обработки видео с аудио
        
        Args:
            audio_path: Путь к аудиофайлу
            face_path: Путь к видео/изображению лица
            output_path: Выходной файл (если None - генерируется автоматически)
            pads: Отступы (верх, низ, лево, право)
            resize_factor: Фактор изменения размера
            rotate: Нужно ли вращать видео
            crop: Обрезка видео (x1,x2,y1,y2)
            
        Returns:
            Путь к выходному файлу
        """
        start_time = time.time()
        output_path = output_path or str(self.tmp_dir / f"output_{int(start_time)}.mp4")
        
        try:
            # Загружаем модели при первом вызове
            self._load_model()
            self._load_face_detector()
            
            # Подготовка входных данных
            face_frames = await self._prepare_face(face_path, resize_factor, rotate, crop)
            audio_data = await self._prepare_audio(audio_path)
            
            # Обработка кадров
            processed_frames = []
            async for frame in self._process_frames(face_frames, audio_data, pads):
                processed_frames.append(frame)
            
            # Сохранение результата
            await self._save_video(processed_frames, output_path, audio_path)
            
            duration = time.time() - start_time
            logger.info(f"Processing completed in {duration:.2f}s. Output: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise

    async def process_async(self, *args, **kwargs):
        """Асинхронная версия обработки"""
        return await self.process(*args, **kwargs)

    async def _prepare_face(self, face_path: str, resize_factor: int, rotate: bool, crop: Tuple[int, int, int, int]):
        """Подготовка видео/изображения лица"""
        from wav2lip.utils import read_frames
        
        frames = read_frames(
            face_path,
            resize_factor=resize_factor,
            rotate=rotate,
            crop=crop
        )
        
        # Обнаружение лица в первом кадре
        first_frame = next(frames)
        face = self.face_detector.detect(first_frame)
        
        if not face:
            raise ValueError("No face detected in the reference media")
        
        # Возвращаем асинхронный генератор
        async def frame_generator():
            yield first_frame
            for frame in frames:
                yield frame
                
        return frame_generator()

    async def _prepare_audio(self, audio_path: str) -> np.ndarray:
        """Подготовка аудиоданных"""
        try:
            # Используем ffmpeg для чтения аудио
            out, _ = (
                ffmpeg.input(audio_path)
                .output('pipe:', format='f32le', ac=1, ar=16000)
                .run(capture_stdout=True, quiet=True)
            )
            return np.frombuffer(out, np.float32)
        except Exception as e:
            logger.error(f"Audio preparation failed: {str(e)}")
            raise

    async def _process_frames(self, frames, audio, pads):
        """Обработка кадров с синхронизацией губ"""
        from wav2lip.audio import get_mel
        
        mel = get_mel(audio).T
        mel_chunks = []
        mel_idx_multiplier = 80. / self.fps
        i = 0
        
        async for frame in frames:
            face = self.face_detector.detect(frame)
            
            if not face:
                logger.warning(f"No face detected in frame {i}")
                i += 1
                continue
                
            face = self._get_smoothened_boxes([face], pads)[0]
            
            # Подготовка входных данных для модели
            face_img = self._prepare_face_image(frame, face)
            mel_chunk = mel[:, int(i * mel_idx_multiplier) : int((i + 1) * mel_idx_multiplier)]
            mel_chunks.append(mel_chunk)
            
            if len(mel_chunks) >= 16:  # Размер батча
                batch = self._create_batch(face_img, mel_chunks)
                
                # Предсказание
                with torch.no_grad():
                    pred = self.model(batch.to(self.device))
                
                # Постобработка и yield кадров
                for p in pred:
                    yield self._postprocess_frame(p, frame, face)
                
                mel_chunks = []
            
            i += 1

    def _get_smoothened_boxes(self, boxes, pads):
        """Сглаживание bounding boxes"""
        from wav2lip.utils import get_smoothened_boxes
        return get_smoothened_boxes(boxes, pads)

    def _prepare_face_image(self, frame, face_box):
        """Подготовка изображения лица для модели"""
        face = frame[
            int(face_box[1]):int(face_box[3]),
            int(face_box[0]):int(face_box[2])
        ]
        face = cv2.resize(face, (self.img_size, self.img_size))
        return torch.FloatTensor(face.transpose(2, 0, 1)).unsqueeze(0) / 255.

    def _create_batch(self, face_img, mel_chunks):
        """Создание батча для модели"""
        batch = []
        for mel in mel_chunks:
            if mel.shape[1] < 16:
                mel = np.pad(mel, ((0, 0), (0, 16 - mel.shape[1])))
            
            batch.append((face_img, torch.FloatTensor(mel).unsqueeze(0)))
        
        return torch.cat([torch.cat([f, m], dim=1) for f, m in batch])

    def _postprocess_frame(self, pred, original_frame, face_box):
        """Постобработка кадра"""
        pred = pred.cpu().numpy().transpose(1, 2, 0) * 255.
        pred = cv2.resize(pred.astype(np.uint8), 
                         (int(face_box[2] - face_box[0]), 
                         int(face_box[3] - face_box[1])))
        
        result = original_frame.copy()
        result[
            int(face_box[1]):int(face_box[3]),
            int(face_box[0]):int(face_box[2])
        ] = pred
        
        return result

    async def _save_video(self, frames, output_path, audio_path):
        """Сохранение видео с аудио"""
        try:
            if not frames:
                raise ValueError("No frames to save")
            
            # Сохраняем временное видео без звука
            tmp_video = str(self.tmp_dir / "temp_video.mp4")
            height, width = frames[0].shape[:2]
            
            (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='bgr24', 
                      s=f'{width}x{height}', r=self.fps)
                .output(tmp_video, pix_fmt='yuv420p')
                .overwrite_output()
                .run_async(pipe_stdin=True, quiet=True)
            ).stdin.write(b''.join([frame.tobytes() for frame in frames]))
            
            # Наложение аудио
            (
                ffmpeg
                .input(tmp_video)
                .input(audio_path)
                .output(output_path, c='copy', map='0:v:0', map='1:a:0', shortest=None)
                .overwrite_output()
                .run(quiet=True)
            )
            
            # Удаление временного файла
            os.unlink(tmp_video)
            
        except Exception as e:
            logger.error(f"Video saving failed: {str(e)}")
            raise

    def cleanup(self, max_age_hours: int = 24):
        """Очистка временных файлов"""
        now = time.time()
        for f in self.tmp_dir.glob('*'):
            if f.is_file() and (now - f.stat().st_mtime) > max_age_hours * 3600:
                try:
                    f.unlink()
                    logger.info(f"Deleted old file: {f.name}")
                except Exception as e:
                    logger.warning(f"Could not delete {f.name}: {str(e)}")

if __name__ == "__main__":
    async def main():
        wav2lip = Wav2LipInference()
        
        result = await wav2lip.process(
            audio_path="static/reference_voice.wav",
            face_path="static/reference_video.mp4",
            output_path="output.mp4"
        )
        
        print(f"Result saved to: {result}")

    asyncio.run(main())