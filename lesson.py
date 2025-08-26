import json
from pathlib import Path
import time
import threading
from typing import Dict, Optional, List
from flask_socketio import SocketIO

class LessonManager:
    def __init__(self, socketio: SocketIO):
        self.socketio = socketio
        self.current_lesson = None
        self.is_active = False
        self.current_phase_index = 0
        self.lessons_dir = Path("lessons")
        
    def start_lesson(self, lesson_data: Dict):
        """Запуск урока"""
        self.current_lesson = lesson_data
        self.is_active = True
        self.current_phase_index = 0
        
        # Запускаем выполнение урока в отдельном потоке
        threading.Thread(target=self._run_lesson, daemon=True).start()
    
    def _run_lesson(self):
        """Выполнение урока"""
        if not self.current_lesson:
            print("Ошибка: урок не загружен")
            return
            
        phases = self.current_lesson.get('phases', [])
        print(f"Начало выполнения {len(phases)} фаз урока: {self.current_lesson['title']}")
        
        for phase_index, phase in enumerate(phases):
            if not self.is_active:
                print(f"Урок прерван в фазе {phase_index}")
                break
                
            self.current_phase_index = phase_index
            
            # Отправляем информацию о фазе
            self.socketio.emit('lesson_phase', {
                'phase_index': phase_index,
                'total_phases': len(phases),
                'type': phase.get('type', 'explanation'),
                'content': phase.get('content', ''),
                'duration': phase.get('duration', 60)
            })
            
            print(f"Фаза {phase_index}: {phase.get('type', 'explanation')}")
            
            # Озвучиваем содержание фазы
            self._speak_phase_content(phase.get('content', ''))
            
            # Ждем продолжительность фазы
            phase_duration = phase.get('duration', 60)
            print(f"Ожидание фазы {phase_index}: {phase_duration} секунд")
            time.sleep(phase_duration)
        
        # Завершение урока
        self.is_active = False
        print("Урок завершен")
        self._speak_phase_content("Урок завершен! Отлично поработали!")
    
    def _speak_phase_content(self, text: str):
        """Озвучивание содержания фазы"""
        if not text.strip():
            return
            
        # Здесь мы эмулируем вызов speak_text из app.py
        # В реальной реализации это будет вызов через сокет
        print(f"Озвучивание: {text[:100]}...")
        
        # Эмуляция отправки через сокет
        self.socketio.emit('lesson_speech', {
            'text': text,
            'is_teacher': True
        })
    
    def stop_lesson(self):
        """Остановка урока"""
        self.is_active = False
        self.current_lesson = None
        self.current_phase_index = 0
    
    def pause_lesson(self):
        """Пауза урока"""
        self.is_active = False
    
    def resume_lesson(self):
        """Продолжение урока"""
        self.is_active = True
    
    def get_current_phase(self) -> Optional[Dict]:
        """Получение текущей фазы"""
        if not self.current_lesson or not self.is_active:
            return None
            
        phases = self.current_lesson.get('phases', [])
        if 0 <= self.current_phase_index < len(phases):
            return phases[self.current_phase_index]
        return None
    
    def get_lesson_progress(self) -> Dict:
        """Получение прогресса урока"""
        if not self.current_lesson:
            return {'active': False, 'progress': 0}
            
        phases = self.current_lesson.get('phases', [])
        if not phases:
            return {'active': self.is_active, 'progress': 0}
            
        progress = (self.current_phase_index / len(phases)) * 100
        return {
            'active': self.is_active,
            'progress': round(progress, 2),
            'current_phase': self.current_phase_index,
            'total_phases': len(phases),
            'lesson_title': self.current_lesson.get('title', '')
        }
    
    def is_lesson_active(self) -> bool:
        """Проверка активности урока"""
        return self.is_active
    
    def load_lesson_from_file(self, lesson_id: str) -> Optional[Dict]:
        """Загрузка урока из файла"""
        lesson_file = self.lessons_dir / f"{lesson_id}.json"
        try:
            if lesson_file.exists():
                with open(lesson_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Ошибка загрузки урока {lesson_id}: {e}")
        return None
    
    def get_available_lessons(self) -> List[Dict]:
        """Получение списка доступных уроков"""
        lessons = []
        try:
            if not self.lessons_dir.exists():
                self.lessons_dir.mkdir(parents=True)
                return lessons
                
            for lesson_file in self.lessons_dir.glob("*.json"):
                try:
                    with open(lesson_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        lessons.append({
                            'id': data.get('id', 'unknown'),
                            'title': data.get('title', 'Без названия'),
                            'subject': data.get('subject', 'other'),
                            'description': data.get('description', ''),
                            'difficulty': data.get('difficulty', 'medium'),
                            'duration': data.get('duration', 1800)
                        })
                except Exception as e:
                    print(f"Ошибка загрузки урока {lesson_file}: {e}")
        except Exception as e:
            print(f"Ошибка доступа к папке уроков: {e}")
        
        return lessons
    
    def get_lessons_by_subject(self, subject: str) -> List[Dict]:
        """Получение уроков по предмету"""
        all_lessons = self.get_available_lessons()
        return [lesson for lesson in all_lessons if lesson.get('subject') == subject]
