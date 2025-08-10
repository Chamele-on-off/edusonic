import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from flask import send_file
import time

class AvatarGenerator:
    def __init__(self, frames_dir: str = "static/avatar/frames"):
        self.frames_dir = frames_dir
        self.frames = self._load_frames()
        self.current_frame = 0
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.last_phoneme = None
        self.last_update = 0
        
    def _load_frames(self) -> List[bytes]:
        """Загружает кадры аватара в память"""
        frames = []
        if not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir, exist_ok=True)
            # Генерируем дефолтные кадры если папка пуста
            frames.append(self._generate_default_frame(eyes_open=True, mouth_open=False))
            frames.append(self._generate_default_frame(eyes_open=True, mouth_open=True))
            return frames
            
        for frame_file in sorted(os.listdir(self.frames_dir)):
            if frame_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    with open(os.path.join(self.frames_dir, frame_file), 'rb') as f:
                        frames.append(f.read())
                except Exception as e:
                    print(f"Error loading frame {frame_file}: {str(e)}")
        
        if not frames:
            # Fallback - два простых кадра
            frames.append(self._generate_default_frame(eyes_open=True, mouth_open=False))
            frames.append(self._generate_default_frame(eyes_open=True, mouth_open=True))
        
        return frames
    
    def _generate_default_frame(self, eyes_open: bool, mouth_open: bool) -> bytes:
        """Генерация простого SVG-аватара как fallback"""
        from PIL import Image, ImageDraw
        
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
        """Возвращает текущий кадр с учетом фонемы"""
        now = time.time()
        
        # Простая логика анимации на основе фонем
        if phoneme and now - self.last_update > 0.1:  # 100ms delay
            self.last_phoneme = phoneme
            self.last_update = now
        
        # Если есть фонема, выбираем соответствующий кадр
        if self.last_phoneme:
            # Простая логика: чередуем два состояния рта
            frame_index = int(now * 10) % 2  # Меняем кадр каждые 100ms
            if frame_index < len(self.frames):
                return self.frames[frame_index]
        
        # Возвращаем первый кадр по умолчанию
        return self.frames[0] if self.frames else self._generate_default_frame(True, False)
    
    def generate_video_stream(self):
        """Генератор видео потока для WebRTC"""
        while True:
            frame = self.get_current_frame(self.last_phoneme)
            yield (b'--frame\r\n'
                   b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)  # 10 FPS

# Глобальный экземпляр генератора
avatar_generator = AvatarGenerator()

def get_avatar_frame(phoneme: Optional[str] = None):
    """Flask endpoint для получения кадра аватара"""
    frame = avatar_generator.get_current_frame(phoneme)
    return send_file(
        BytesIO(frame),
        mimetype='image/png',
        as_attachment=False
    )
