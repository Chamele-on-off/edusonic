import os
import time
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from flask import send_file
from PIL import Image, ImageDraw
import cv2
import numpy as np
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AvatarGenerator:
    def __init__(self, frames_dir: str = "static/avatar/frames"):
        """
        Инициализация генератора аватара.
        
        :param frames_dir: Путь к папке с кадрами аватара
        """
        self.frames_dir = frames_dir
        self.frames = self._load_frames()
        self.current_frame_index = 0
        self.last_phoneme = None
        self.last_update = 0
        self.last_blink = time.time()
        self.blink_interval = random.uniform(3, 5)  # Случайный интервал моргания
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Соответствие фонем и групп кадров
        self.phoneme_groups = {
            'а': 'mouth_aa',
            'о': 'mouth_oo',
            'у': 'mouth_uu',
            'и': 'mouth_ee',
            'э': 'mouth_ee',
            'ы': 'mouth_aa',
            'е': 'mouth_ee',
            'ё': 'mouth_oo',
            'ю': 'mouth_uu',
            'я': 'mouth_aa',
            'м': 'mouth_mm',
            'п': 'mouth_pp',
            'б': 'mouth_bb',
            'ф': 'mouth_ff',
            'в': 'mouth_vv',
            'ш': 'mouth_sh',
            'ж': 'mouth_zh',
            'с': 'mouth_ss',
            'з': 'mouth_zz',
            'р': 'mouth_rr',
            'л': 'mouth_ll',
            'н': 'mouth_nn',
            'т': 'mouth_tt',
            'д': 'mouth_dd',
            'к': 'mouth_kk',
            'г': 'mouth_gg',
            'х': 'mouth_hh',
            'ч': 'mouth_ch',
            'щ': 'mouth_sh',
            'ц': 'mouth_ss',
            'й': 'mouth_ee'
        }

    def _load_frames(self) -> Dict[str, List[bytes]]:
        """
        Загружает все кадры аватара из папки в память.
        
        Возвращает словарь, где ключи - группы кадров (например, 'mouth_aa'),
        а значения - списки байтовых представлений изображений.
        """
        frames = {}
        
        if not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir, exist_ok=True)
            logger.warning(f"Папка с кадрами {self.frames_dir} не найдена, создана пустая")
            return frames

        # Сканируем все файлы в папке и группируем их
        for filename in os.listdir(self.frames_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Определяем группу кадра по имени файла
                group = self._get_frame_group(filename)
                
                try:
                    with open(os.path.join(self.frames_dir, filename), 'rb') as f:
                        frame_data = f.read()
                    
                    if group not in frames:
                        frames[group] = []
                    frames[group].append(frame_data)
                except Exception as e:
                    logger.error(f"Ошибка загрузки кадра {filename}: {str(e)}")

        # Если кадров нет, создаем два дефолтных
        if not frames:
            logger.warning("Не найдено кадров аватара, создаются дефолтные")
            frames['mouth_neutral'] = [self._generate_default_frame(eyes_open=True, mouth_open=False)]
            frames['mouth_open'] = [self._generate_default_frame(eyes_open=True, mouth_open=True)]
            frames['blink'] = [self._generate_default_frame(eyes_open=False, mouth_open=False)]

        logger.info(f"Загружено групп кадров: {len(frames)}")
        for group, group_frames in frames.items():
            logger.info(f"  Группа {group}: {len(group_frames)} кадров")
        
        return frames

    def _get_frame_group(self, filename: str) -> str:
        """
        Определяет группу кадра по его имени.
        
        Например:
        'mouth_aa_001.jpg' -> 'mouth_aa'
        'blink_05.jpg' -> 'blink'
        """
        # Удаляем расширение и номер
        base_name = os.path.splitext(filename)[0]
        if '_' in base_name:
            return '_'.join(base_name.split('_')[:-1])
        return base_name

    def _generate_default_frame(self, eyes_open: bool, mouth_open: bool) -> bytes:
        """
        Генерирует простой SVG-кадр как fallback.
        
        :param eyes_open: Открыты ли глаза
        :param mouth_open: Открыт ли рот
        :return: Байтовое представление PNG изображения
        """
        img = Image.new('RGB', (640, 480), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        # Голова
        draw.ellipse([(120, 50), (520, 430)], outline=(0, 0, 0), width=2, fill=(255, 255, 255))
        
        # Глаза
        eye_y = 180
        if eyes_open:
            draw.ellipse([(220, eye_y), (280, eye_y+60)], fill=(0, 0, 0))  # Левый глаз
            draw.ellipse([(360, eye_y), (420, eye_y+60)], fill=(0, 0, 0))  # Правый глаз
        else:
            draw.line([(220, eye_y+30), (280, eye_y+30)], fill=(0, 0, 0), width=2)
            draw.line([(360, eye_y+30), (420, eye_y+30)], fill=(0, 0, 0), width=2)
        
        # Рот
        mouth_y = 320
        if mouth_open:
            draw.ellipse([(270, mouth_y), (370, mouth_y+80)], fill=(0, 0, 0))  # Открытый рот
        else:
            draw.line([(270, mouth_y+40), (370, mouth_y+40)], fill=(0, 0, 0), width=2)
        
        # Сохраняем в bytes
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()

    def get_current_frame(self, phoneme: Optional[str] = None) -> bytes:
        """
        Возвращает текущий кадр аватара с учетом фонемы и времени.
        
        :param phoneme: Текущая фонема (например, 'а', 'о')
        :return: Байтовое представление изображения
        """
        now = time.time()
        
        # Обновляем текущую фонему
        if phoneme and now - self.last_update > 0.1:  # 100ms задержка
            self.last_phoneme = phoneme.lower()
            self.last_update = now
        
        # Проверяем, нужно ли моргать
        if now - self.last_blink > self.blink_interval:
            self.last_blink = now
            self.blink_interval = random.uniform(3, 5)  # Случайный интервал
            blink_frames = self.frames.get('blink', [])
            if blink_frames:
                return random.choice(blink_frames)
        
        # Выбираем кадры для текущей фонемы
        if self.last_phoneme:
            group_name = self.phoneme_groups.get(self.last_phoneme, 'mouth_neutral')
            group_frames = self.frames.get(group_name, [])
            
            if group_frames:
                # Плавная анимация - перебираем кадры по порядку
                self.current_frame_index = (self.current_frame_index + 1) % len(group_frames)
                return group_frames[self.current_frame_index]
        
        # Возвращаем нейтральный кадр
        neutral_frames = self.frames.get('mouth_neutral', [])
        if neutral_frames:
            return neutral_frames[0]
        
        # Fallback - первый доступный кадр
        for group_frames in self.frames.values():
            if group_frames:
                return group_frames[0]
        
        # Если вообще нет кадров (маловероятно)
        return self._generate_default_frame(eyes_open=True, mouth_open=False)

    def generate_video_stream(self):
        """
        Генератор для потоковой передачи видео.
        Используется для WebRTC трансляции.
        """
        while True:
            frame = self.get_current_frame(self.last_phoneme)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)  # ~30 FPS

# Глобальный экземпляр для использования в Flask
avatar_generator = AvatarGenerator()

def get_avatar_frame(phoneme: Optional[str] = None):
    """
    Flask endpoint для получения кадра аватара.
    
    :param phoneme: Текущая фонема для анимации рта
    :return: Response с изображением
    """
    frame = avatar_generator.get_current_frame(phoneme)
    return send_file(
        BytesIO(frame),
        mimetype='image/jpeg',  # Используем JPEG для эффективности
        as_attachment=False
    )
