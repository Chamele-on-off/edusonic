import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor
import ffmpeg

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
        tmp_dir: str = "static/tmp"
    ):
        self.device = device
        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(exist_ok=True, parents=True)
        
        # Проверка наличия моделей
        self.checkpoint_path = self._validate_path(checkpoint_path)
        self.face_detector_path = self._validate_path(face_detector_path)
        
        # Загрузка моделей
        self.model = self._load_model()
        self.face_detector = self._load_face_detector()
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        logger.info(f"Wav2Lip initialized on {device}")

    def _validate_path(self, path: str) -> Path:
        """Проверка наличия файла модели"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        return path

    def _load_model(self):
        """Загрузка модели Wav2Lip"""
        try:
            from wav2lip.models import Wav2Lip
            model = Wav2Lip()
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device).eval()
            logger.info("Wav2Lip model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load Wav2Lip model: {str(e)}")
            raise

    def _load_face_detector(self):
        """Загрузка детектора лиц"""
        try:
            from wav2lip.face_detection import FaceDetector
            detector = FaceDetector(
                mobilenet_path=self.face_detector_path,
                device=self.device
            )
            logger.info("Face detector loaded successfully")
            return detector
        except Exception as e:
            logger.error(f"Failed to load face detector: {str(e)}")
            raise

    def process(
        self,
        audio_path: str,
        face_path: str,
        output_path: Optional[str] = None,
        fps: float = 25.0,
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
            fps: Кадры в секунду
            pads: Отступы (верх, низ, лево, право)
            resize_factor: Фактор изменения размера
            rotate: Нужно ли вращать видео
            crop: Обрезка видео (x1,x2,y1,y2)
            
        Returns:
            Путь к выходному файлу
        """
        start_time = time.time()
        output_path = output_path or self.tmp_dir / f"output_{int(start_time)}.mp4"
        
        try:
            # Подготовка входных данных
            face_data = self._prepare_face(face_path, resize_factor, rotate, crop)
            audio_data = self._prepare_audio(audio_path)
            
            # Обработка кадров
            processed_frames = []
            for frame in self._process_frames(face_data, audio_data, pads):
                processed_frames.append(frame)
            
            # Сохранение результата
            self._save_video(processed_frames, str(output_path), fps, audio_path)
            
            duration = time.time() - start_time
            logger.info(f"Processing completed in {duration:.2f}s. Output: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise

    async def process_async(self, *args, **kwargs):
        """Асинхронная версия обработки"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.process(*args, **kwargs)
        )

    def _prepare_face(self, face_path: str, resize_factor: int, rotate: bool, crop: Tuple[int, int, int, int]):
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
        
        return itertools.chain([first_frame], frames)

    def _prepare_audio(self, audio_path: str) -> np.ndarray:
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

    def _process_frames(self, frames, audio, pads):
        """Обработка кадров с синхронизацией губ"""
        from wav2lip.utils import get_smoothened_boxes
        from wav2lip.audio import get_mel
        
        mel = get_mel(audio).T
        mel_idx_multiplier = 80. / fps
        
        for i, frame in enumerate(frames):
            face = self.face_detector.detect(frame)
            
            if not face:
                logger.warning(f"No face detected in frame {i}")
                continue
                
            face = get_smoothened_boxes([face], pads)[0]
            
            # Подготовка входных данных для модели
            img_batch = self._prepare_frame_batch(frame, face, mel, i, mel_idx_multiplier)
            
            # Предсказание
            with torch.no_grad():
                pred = self.model(img_batch.to(self.device))
            
            yield self._postprocess_frame(pred, frame, face)

    def _prepare_frame_batch(self, frame, face, mel, frame_idx, mel_idx_multiplier):
        """Подготовка батча для модели"""
        raise NotImplementedError("Frame batch preparation not implemented")

    def _postprocess_frame(self, pred, original_frame, face):
        """Постобработка кадра"""
        raise NotImplementedError("Frame postprocessing not implemented")

    def _save_video(self, frames, output_path, fps, audio_path):
        """Сохранение видео с аудио"""
        try:
            # Сохраняем временное видео без звука
            tmp_video = str(self.tmp_dir / "temp_video.mp4")
            height, width = frames[0].shape[:2]
            
            (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}', r=fps)
                .output(tmp_video, pix_fmt='yuv420p')
                .overwrite_output()
                .run(input=b''.join([frame.tobytes() for frame in frames]), quiet=True)
            )
            
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
    # Пример использования
    wav2lip = Wav2LipInference()
    
    result = wav2lip.process(
        audio_path="static/reference_voice.wav",
        face_path="static/reference_video.mp4",
        output_path="output.mp4"
    )
    
    print(f"Result saved to: {result}")
