import json
from pathlib import Path
from typing import List, Dict, Optional
import threading
import time

class LectureManager:
    def __init__(self, socketio):
        self.socketio = socketio
        self.lessons_dir = Path("lessons")
        self.current_lesson = None
        self.current_text_index = 0
        self.is_playing = False
        self.is_paused = False
        self.playback_thread = None
        
    def load_lesson(self, lesson_id: str, subject: str) -> Optional[Dict]:
        """Загружает урок по ID и предмету"""
        try:
            lesson_path = self.lessons_dir / f"{subject}_{lesson_id}.json"
            if not lesson_path.exists():
                # Попробуем найти по ID
                for lesson_file in self.lessons_dir.glob("*.json"):
                    if lesson_id in lesson_file.stem:
                        lesson_path = lesson_file
                        break
            
            if lesson_path.exists():
                with open(lesson_path, 'r', encoding='utf-8') as f:
                    lesson_data = json.load(f)
                    self.current_lesson = lesson_data
                    self.current_text_index = 0
                    return lesson_data
        except Exception as e:
            print(f"Ошибка загрузки урока: {e}")
        return None

    def get_lecture_texts(self) -> List[str]:
        """Возвращает тексты лекции"""
        if self.current_lesson:
            return self.current_lesson.get('lecture_texts', [])
        return []

    def get_current_text(self) -> Optional[str]:
        """Возвращает текущий текст лекции"""
        texts = self.get_lecture_texts()
        if texts and 0 <= self.current_text_index < len(texts):
            return texts[self.current_text_index]
        return None

    def get_next_text(self) -> Optional[str]:
        """Возвращает следующий текст лекции"""
        texts = self.get_lecture_texts()
        if texts and self.current_text_index + 1 < len(texts):
            self.current_text_index += 1
            return texts[self.current_text_index]
        return None

    def get_previous_text(self) -> Optional[str]:
        """Возвращает предыдущий текст лекции"""
        texts = self.get_lecture_texts()
        if texts and self.current_text_index - 1 >= 0:
            self.current_text_index -= 1
            return texts[self.current_text_index]
        return None

    def start_lecture(self, room_id: str, on_text_callback=None):
        """Начинает воспроизведение лекции"""
        if not self.current_lesson or self.is_playing:
            return False
        
        self.is_playing = True
        self.is_paused = False
        
        def playback_loop():
            texts = self.get_lecture_texts()
            
            for text_index, text in enumerate(texts):
                if not self.is_playing:
                    break
                    
                # Проверяем паузу
                while self.is_paused and self.is_playing:
                    time.sleep(0.5)
                    
                if not self.is_playing:
                    break
                
                self.current_text_index = text_index
                
                # Вызываем callback для озвучивания текста
                if on_text_callback:
                    on_text_callback(room_id, text)
                
                # Ждем перед следующим текстом
                text_duration = max(10, len(text) * 0.15)
                time.sleep(text_duration)
            
            self.is_playing = False
            if on_text_callback:
                on_text_callback(room_id, "Лекция завершена!")
        
        self.playback_thread = threading.Thread(target=playback_loop, daemon=True)
        self.playback_thread.start()
        return True

    def pause_lecture(self):
        """Приостанавливает лекцию"""
        self.is_paused = True

    def resume_lecture(self):
        """Возобновляет лекцию"""
        self.is_paused = False

    def stop_lecture(self):
        """Останавливает лекцию"""
        self.is_playing = False
        self.is_paused = False

    def skip_to_next(self):
        """Переходит к следующему тексту"""
        self.current_text_index += 1

    def skip_to_previous(self):
        """Переходит к предыдущему тексту"""
        self.current_text_index -= 1

    def get_lesson_info(self) -> Dict:
        """Возвращает информацию о текущем уроке"""
        if self.current_lesson:
            return {
                'title': self.current_lesson.get('title', ''),
                'subject': self.current_lesson.get('subject', ''),
                'total_texts': len(self.get_lecture_texts()),
                'current_index': self.current_text_index,
                'is_playing': self.is_playing,
                'is_paused': self.is_paused
            }
        return {}

    def get_available_lessons(self, subject: str = None) -> List[Dict]:
        """Возвращает список доступных уроков"""
        lessons = []
        try:
            for lesson_file in self.lessons_dir.glob("*.json"):
                try:
                    with open(lesson_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if subject and data.get('subject') != subject:
                            continue
                        lessons.append({
                            'id': data.get('id', ''),
                            'title': data.get('title', ''),
                            'subject': data.get('subject', ''),
                            'description': data.get('description', ''),
                            'text_count': len(data.get('lecture_texts', []))
                        })
                except:
                    continue
        except:
            pass
        return lessons
