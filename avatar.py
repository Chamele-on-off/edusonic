# avatar.py
import os
import time
import random
import threading
from collections import defaultdict
from pathlib import Path
from flask_socketio import SocketIO

class AvatarAnimationManager:
    def __init__(self, socketio: SocketIO, frames_dir: Path):
        self.socketio = socketio
        self.frames_dir = frames_dir
        
        # Состояния анимации
        self.animation_running = defaultdict(bool)
        self.room_teacher_speaking = defaultdict(bool)
        
        # Соответствие букв кадрам анимации рта
        self.PHONEME_MAP = {
            'а': 'mouth_aa', 'о': 'mouth_oo', 'у': 'mouth_uu',
            'и': 'mouth_ee', 'э': 'mouth_ee', 'ы': 'mouth_aa',
            'е': 'mouth_ee', 'ё': 'mouth_oo', 'ю': 'mouth_uu',
            'я': 'mouth_aa', 'м': 'mouth_mm', 'п': 'mouth_pp',
            'б': 'mouth_bb', 'ф': 'mouth_ff', 'в': 'mouth_vv',
            'ш': 'mouth_sh', 'ж': 'mouth_zh', 'с': 'mouth_ss',
            'з': 'mouth_zz', 'р': 'mouth_rr', 'л': 'mouth_ll',
            'н': 'mouth_nn', 'т': 'mouth_tt', 'д': 'mouth_dd',
            'к': 'mouth_kk', 'г': 'mouth_gg', 'х': 'mouth_hh',
            'ч': 'mouth_ch', 'щ': 'mouth_sh', 'ц': 'mouth_ss',
            'й': 'mouth_ee'
        }

    def get_neutral_frames(self, avatar_name):
        """Возвращает список нейтральных кадров для аватара"""
        avatar_dir = self.frames_dir / avatar_name
        if not avatar_dir.exists():
            return []
        return sorted([f for f in os.listdir(avatar_dir) 
                      if f.startswith('mouth_neutral_')])

    def get_blink_frames(self, avatar_name):
        """Возвращает список кадров моргания для аватара"""
        avatar_dir = self.frames_dir / avatar_name
        if not avatar_dir.exists():
            return []
        return sorted([f for f in os.listdir(avatar_dir) 
                      if f.startswith('blink_')])

    def get_speech_frames(self, avatar_name, phoneme):
        """Возвращает список речевых кадров для указанной фонемы"""
        avatar_dir = self.frames_dir / avatar_name
        if not avatar_dir.exists():
            return []
        
        base_name = self.PHONEME_MAP.get(phoneme, 'mouth_aa')
        return [f for f in os.listdir(avatar_dir) 
                if f.startswith(base_name)]

    def set_teacher_speaking(self, room_id, speaking):
        """Устанавливает состояние речи учителя"""
        self.room_teacher_speaking[room_id] = speaking

    def animation_loop(self, room_id, avatar_name):
        """Основной цикл анимации для комнаты"""
        blink_counter = 0
        blink_frames = self.get_blink_frames(avatar_name)
        neutral_frames = self.get_neutral_frames(avatar_name)
        
        print(f"🚀 Анимация запущена для комнаты {room_id}, аватар: {avatar_name}")
        print(f"📁 Доступные кадры: нейтральные={len(neutral_frames)}, моргание={len(blink_frames)}")
        
        while self.animation_running[room_id]:
            try:
                # Анимация речи учителя
                if self.room_teacher_speaking[room_id]:
                    current_char = random.choice(list(self.PHONEME_MAP.keys()))
                    speech_frames = self.get_speech_frames(avatar_name, current_char)
                    if speech_frames:
                        frame = random.choice(speech_frames)
                        frame_path = f'/frames/{avatar_name}/{frame}'
                        self.socketio.emit('animation_frame', {'frame': frame_path}, room=room_id)
                        print(f"🎬 Отправлен кадр речи: {frame_path}")
                else:
                    # Анимация покоя
                    blink_counter += 1
                    if blink_counter >= 30 and blink_frames:
                        # Моргание
                        for frame in blink_frames:
                            frame_path = f'/frames/{avatar_name}/{frame}'
                            self.socketio.emit('animation_frame', {'frame': frame_path}, room=room_id)
                            time.sleep(0.1)
                        blink_counter = 0
                        print("👁️ Отправлены кадры моргания")
                    elif neutral_frames:
                        # Нейтральное состояние
                        frame = random.choice(neutral_frames)
                        frame_path = f'/frames/{avatar_name}/{frame}'
                        self.socketio.emit('animation_frame', {'frame': frame_path}, room=room_id)
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Ошибка в цикле анимации: {e}")
                time.sleep(0.5)  # Пауза при ошибке

    def start_animation(self, room_id, avatar_name):
        """Запускает анимацию для комнаты"""
        if not self.animation_running[room_id]:
            self.animation_running[room_id] = True
            # Принудительно устанавливаем начальное состояние
            self.room_teacher_speaking[room_id] = False
            
            # Проверяем существование аватара
            avatar_dir = self.frames_dir / avatar_name
            if not avatar_dir.exists():
                # Пытаемся найти любой доступный аватар
                avatars = [d for d in os.listdir(self.frames_dir) if (self.frames_dir / d).is_dir()]
                if avatars:
                    avatar_name = avatars[0]
                    print(f"⚠️  Аватар не найден, используем первый доступный: {avatar_name}")
                else:
                    print("❌ Нет доступных аватаров!")
                    return False
            
            thread = threading.Thread(target=self.animation_loop, args=(room_id, avatar_name))
            thread.daemon = True
            thread.start()
            print(f"✅ Анимация запущена для комнаты {room_id}")
            return True
        return False

    def stop_animation(self, room_id):
        """Останавливает анимацию для комнаты"""
        if self.animation_running[room_id]:
            self.animation_running[room_id] = False
            self.room_teacher_speaking[room_id] = False
            print(f"⏹️ Анимация остановлена для комнаты {room_id}")

    def is_animation_running(self, room_id):
        """Проверяет, запущена ли анимация"""
        return self.animation_running[room_id]
