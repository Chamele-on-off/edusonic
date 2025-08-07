import cv2
import numpy as np
import dlib
from pathlib import Path
import os
import logging
from typing import List, Tuple
import ffmpeg
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lip_sync_advanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedLipSync:
    def __init__(
        self,
        reference_video: str = "static/reference_video.mp4",
        mouth_frames_dir: str = "static/mouth_frames",
        output_dir: str = "static/tmp",
        fps: float = 25.0
    ):
        self.reference_video = Path(reference_video)
        self.mouth_frames_dir = Path(mouth_frames_dir)
        self.output_dir = Path(output_dir)
        self.fps = fps
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # Инициализация детектора лиц и предиктора ключевых точек
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Создаем директории
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mouth_frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Загружаем референсное видео и фреймы рта
        self.reference_frames = self._load_reference_video()
        self.mouth_frames = self._load_mouth_frames()
        
        logger.info("AdvancedLipSync initialized")

    def _load_reference_video(self) -> List[np.ndarray]:
        """Загрузка и детекция лица в референсном видео"""
        if not self.reference_video.exists():
            raise FileNotFoundError(f"Reference video not found: {self.reference_video}")
        
        cap = cv2.VideoCapture(str(self.reference_video))
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Конвертируем в grayscale для детекции
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            
            if len(faces) == 0:
                logger.warning("No face detected in reference frame")
                continue
                
            # Сохраняем кадр и информацию о лице
            frames.append({
                "frame": frame,
                "face": faces[0]  # Берем первое обнаруженное лицо
            })
        
        cap.release()
        
        if not frames:
            raise ValueError("No valid frames with faces found in reference video")
        
        return frames

    def _load_mouth_frames(self) -> dict:
        """Загрузка фреймов рта с альфа-каналом"""
        mouth_frames = {}
        for frame_file in self.mouth_frames_dir.glob("*.png"):
            phoneme = frame_file.stem
            frame = cv2.imread(str(frame_file), cv2.IMREAD_UNCHANGED)
            
            if frame.shape[2] == 4:  # Проверяем наличие альфа-канала
                mouth_frames[phoneme] = frame
            else:
                logger.warning(f"Mouth frame {frame_file} has no alpha channel")
        
        if not mouth_frames:
            logger.warning("No mouth frames found, generating default set")
            mouth_frames = self._generate_default_mouth_frames()
        
        return mouth_frames

    def _generate_default_mouth_frames(self) -> dict:
        """Генерация фреймов рта с альфа-каналом"""
        frames = {}
        for phoneme in ['neutral', 'aa', 'ee', 'oo', 'mm']:
            # Создаем изображение с альфа-каналом
            frame = np.zeros((100, 100, 4), dtype=np.uint8)
            
            # Задаем разные формы рта для разных фонем
            if phoneme == 'neutral':
                color = (0, 0, 255, 200)  # Красный с прозрачностью
                cv2.ellipse(frame, (50, 60), (30, 10), 0, 0, 360, color, -1)
            elif phoneme == 'aa':
                color = (0, 255, 0, 220)  # Зеленый
                cv2.ellipse(frame, (50, 50), (40, 20), 0, 0, 360, color, -1)
            elif phoneme == 'ee':
                color = (255, 0, 0, 210)  # Синий
                cv2.rectangle(frame, (30, 40), (70, 60), color, -1)
            elif phoneme == 'oo':
                color = (255, 255, 0, 230)  # Голубой
                cv2.circle(frame, (50, 50), 20, color, -1)
            elif phoneme == 'mm':
                color = (255, 0, 255, 240)  # Пурпурный
                cv2.rectangle(frame, (30, 45), (70, 55), color, -1)
            
            frames[phoneme] = frame
            cv2.imwrite(str(self.mouth_frames_dir / f"{phoneme}.png"), frame)
        
        return frames

    def _get_mouth_region(self, frame: np.ndarray, face: dlib.rectangle) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
        """Определение области рта с помощью ключевых точек"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shape = self.predictor(gray, face)
        
        # Точки рта (индексы 48-68 в 68-точечной модели)
        mouth_points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)])
        
        # Вычисляем ограничивающий прямоугольник
        x, y, w, h = cv2.boundingRect(mouth_points)
        
        # Увеличиваем область для лучшего покрытия
        padding = 15
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2*padding)
        h = min(frame.shape[0] - y, h + 2*padding)
        
        # Маска рта (выпуклая оболочка)
        hull = cv2.convexHull(mouth_points)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        
        return (x, y, w, h), mask

    def _blend_mouth_advanced(self, frame: np.ndarray, mouth_frame: np.ndarray, mouth_region: Tuple[int, int, int, int], mask: np.ndarray) -> np.ndarray:
        """Плавное наложение области рта с использованием альфа-канала"""
        x, y, w, h = mouth_region
        
        # Изменяем размер mouth_frame под область рта
        mouth_resized = cv2.resize(mouth_frame, (w, h))
        
        # Разделяем цвет и альфа-канал
        mouth_rgb = mouth_resized[:, :, :3]
        alpha = mouth_resized[:, :, 3] / 255.0
        alpha = cv2.merge([alpha, alpha, alpha])
        
        # Обрезаем маску под область рта
        mask_roi = mask[y:y+h, x:x+w]
        mask_roi = cv2.merge([mask_roi, mask_roi, mask_roi]) / 255.0
        
        # Комбинируем маски
        combined_mask = alpha * mask_roi
        
        # Наложение с плавным переходом
        result = frame.copy()
        roi = result[y:y+h, x:x+w]
        
        blended = (1.0 - combined_mask) * roi + combined_mask * mouth_rgb
        result[y:y+h, x:x+w] = blended.astype(np.uint8)
        
        return result

    def _get_phoneme_for_frame(self, audio_data: np.ndarray, frame_idx: int) -> str:
        """Определение фонемы (упрощенная версия)"""
        phonemes = ['neutral', 'aa', 'ee', 'oo', 'mm', 'neutral', 'aa', 'ee']
        return phonemes[frame_idx % len(phonemes)]

    async def generate_video(self, audio_path: str, output_path: Optional[str] = None) -> str:
        """Генерация видео с продвинутым блендингом"""
        start_time = time.time()
        output_path = output_path or str(self.output_dir / f"output_{int(start_time)}.mp4")
        
        try:
            # Получаем информацию об аудио
            audio_info = ffmpeg.probe(audio_path)
            duration = float(audio_info['format']['duration'])
            total_frames = int(duration * self.fps)
            
            # Создаем видео
            frame_size = (self.reference_frames[0]["frame"].shape[1], self.reference_frames[0]["frame"].shape[0])
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, frame_size)
            
            for i in range(total_frames):
                # Циклически используем кадры из референсного видео
                ref_data = self.reference_frames[i % len(self.reference_frames)]
                frame = ref_data["frame"].copy()
                face = ref_data["face"]
                
                # Определяем область рта
                mouth_region, mouth_mask = self._get_mouth_region(frame, face)
                
                # Определяем фонему
                phoneme = self._get_phoneme_for_frame(None, i)
                mouth_frame = self.mouth_frames.get(phoneme, self.mouth_frames['neutral'])
                
                # Накладываем область рта
                blended_frame = self._blend_mouth_advanced(frame, mouth_frame, mouth_region, mouth_mask)
                out.write(blended_frame)
            
            out.release()
            
            # Добавляем аудио к видео
            final_output = output_path.replace('.mp4', '_final.mp4')
            (
                ffmpeg
                .input(output_path)
                .input(audio_path)
                .output(final_output, c='copy', map='0:v:0', map='1:a:0', shortest=None)
                .overwrite_output()
                .run(quiet=True)
            )
            
            os.remove(output_path)
            logger.info(f"Video generated: {final_output} (duration: {duration:.2f}s)")
            return final_output
            
        except Exception as e:
            logger.error(f"Video generation failed: {str(e)}")
            raise

    async def process_async(self, audio_path: str) -> str:
        """Асинхронная версия обработки"""
        return await self.generate_video(audio_path)

# Пример использования
if __name__ == "__main__":
    async def main():
        lip_sync = AdvancedLipSync()
        result = await lip_sync.generate_video(
            audio_path="static/reference_voice.wav",
            output_path="output_advanced.mp4"
        )
        print(f"Result saved to: {result}")

    asyncio.run(main())
