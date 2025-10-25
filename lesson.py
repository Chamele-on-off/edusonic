from pathlib import Path
import re
from typing import List, Optional
import threading
import time
from typing import Callable
import requests
import json

class LessonManager:
    def __init__(self, lessons_dir: str = "lessons"):
        self.lessons_dir = Path(lessons_dir)
        self.current_lessons = {}
        
        # НОВЫЕ ПОЛЯ ДЛЯ ТАЙМЕРА
        self.auto_continue_timer = None
        self.auto_continue_callback = None
        self.auto_continue_delay = 25  # 25 секунд
        self.timer_active = False
        self.room_id = None
        self.app_url = "http://localhost:5000"  # URL для вызова API
        
        self._ensure_lessons_dir()
    
    def _ensure_lessons_dir(self):
        """Создает папку lessons если ее нет"""
        if not self.lessons_dir.exists():
            self.lessons_dir.mkdir(parents=True)
            # Создаем демо-урок по обществознанию
            demo_lesson = self.lessons_dir / "social_general.txt"
            if not demo_lesson.exists():
                with open(demo_lesson, 'w', encoding='utf-8') as f:
                    f.write("Основы обществознания: подготовка к ЕГЭ.\n\nДобро пожаловать на демо-урок! Сегодня мы разберем фундаментальные понятия обществознания.\n\nОбщество - это сложная динамическая система, объединяющая людей, которые связаны совместной деятельностью, общими интересами и ценностями.\n\nГосударство - это политическая организация общества, обладающая суверенитетом и аппаратом управления.\n\nДемократия - это форма правления, при которой народ является источником власти.\n\nЭкономика - это хозяйственная деятельность общества, система производства и распределения товаров.\n\nКультура - это совокупность достижений человечества в духовной и материальной жизни.\n\nПраво - это система общеобязательных норм, охраняемых государством.\n\nСоциализация - это процесс усвоения индивидом социальных норм и ценностей.\n\nЛичность - это человек как носитель социальных качеств и сознательной деятельности.\n\nМораль - это система норм и принципов, регулирующих поведение людей.\n\nГлобализация - это процесс всемирной экономической, политической и культурной интеграции.")
    
    def load_lesson_content(self, lesson_file: str) -> List[str]:
        """Загружает содержание урока из текстового файла"""
        try:
            lesson_path = self.lessons_dir / lesson_file
            if not lesson_path.exists():
                return ["Урок не найден. Попробуйте выбрать другой урок."]
                
            with open(lesson_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Разбиваем на абзацы (по пустым строкам)
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            # Если абзацев нет, разбиваем на предложения
            if not paragraphs:
                sentences = re.split(r'(?<=[.!?])\s+', content)
                # Объединяем предложения в группы по 2-3 для плавного чтения
                current_paragraph = []
                paragraphs = []
                
                for sentence in sentences:
                    if sentence.strip():
                        current_paragraph.append(sentence.strip())
                        if len(current_paragraph) >= 2:  # Группируем по 2-3 предложения
                            paragraphs.append(' '.join(current_paragraph))
                            current_paragraph = []
                
                # Добавляем оставшиеся предложения
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
            
            return paragraphs if paragraphs else ["Содержание урока временно недоступно."]
            
        except Exception as e:
            print(f"Ошибка загрузки урока {lesson_file}: {e}")
            return ["Ошибка загрузки урока. Попробуйте позже."]
    
    def get_available_lessons(self) -> dict:
        """Возвращает список доступных уроков"""
        lessons = {}
        try:
            for lesson_file in self.lessons_dir.glob("*.txt"):
                subject = self._detect_subject(lesson_file.stem)
                
                if subject not in lessons:
                    lessons[subject] = []
                
                lessons[subject].append({
                    'id': lesson_file.stem,
                    'title': lesson_file.stem.replace('_', ' ').title(),
                    'file_path': lesson_file.name,
                    'type': 'text'
                })
        except Exception as e:
            print(f"Ошибка получения списка уроков: {e}")
        
        return lessons
    
    def _detect_subject(self, filename: str) -> str:
        """Определяет предмет по названию файла"""
        filename_lower = filename.lower()
        if any(word in filename_lower for word in ['math', 'математика', 'алгебра', 'геометрия']):
            return "математика"
        elif any(word in filename_lower for word in ['history', 'история', 'истор']):
            return "история"
        elif any(word in filename_lower for word in ['physics', 'физика', 'физ']):
            return "физика"
        elif any(word in filename_lower for word in ['chemistry', 'химия', 'хим']):
            return "химия"
        elif any(word in filename_lower for word in ['social', 'обществознание', 'общество']):
            return "обществознание"
        elif any(word in filename_lower for word in ['biology', 'биология', 'био']):
            return "биология"
        elif any(word in filename_lower for word in ['literature', 'литература', 'лит']):
            return "литература"
        elif any(word in filename_lower for word in ['russian', 'русский', 'язык']):
            return "русский язык"
        else:
            return "общее"

    # НОВЫЕ МЕТОДЫ ДЛЯ АВТОПРОДОЛЖЕНИЯ
    
    def set_auto_continue_callback(self, callback: Callable, room_id: str):
        """Устанавливает callback для автоматического продолжения"""
        self.auto_continue_callback = callback
        self.room_id = room_id
        print(f"🔧 Установлен callback авто-продолжения для комнаты {room_id}")
    
    def start_auto_continue_timer(self):
        """Запускает таймер автоматического продолжения"""
        if self.timer_active:
            self.stop_auto_continue_timer()
            
        self.timer_active = True
        print(f"⏰ Таймер авто-продолжения запущен для комнаты {self.room_id} ({self.auto_continue_delay} сек)")
        
        self.auto_continue_timer = threading.Timer(
            self.auto_continue_delay, 
            self._on_auto_continue_timeout
        )
        self.auto_continue_timer.daemon = True
        self.auto_continue_timer.start()
    
    def stop_auto_continue_timer(self):
        """Останавливает таймер автоматического продолжения"""
        if self.auto_continue_timer:
            self.auto_continue_timer.cancel()
            self.auto_continue_timer = None
        self.timer_active = False
        print(f"⏹️ Таймер авто-продолжения остановлен для комнаты {self.room_id}")
    
    def _on_auto_continue_timeout(self):
        """Вызывается при срабатывании таймера - ВЫЗЫВАЕТ API ДЛЯ РЕАЛЬНОГО ПРОДОЛЖЕНИЯ"""
        if not self.timer_active:
            return
            
        print(f"🔄 Таймер сработал, автоматическое продолжение для комнаты {self.room_id}")
        
        try:
            # ВЫЗЫВАЕМ API ДЛЯ РЕАЛЬНОГО ПРОДОЛЖЕНИЯ УРОКА
            response = requests.post(
                f"{self.app_url}/api/auto_continue/{self.room_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    print(f"✅ Автоматическое продолжение выполнено: {result.get('message', '')}")
                    
                    # Если урок завершен, останавливаем таймер
                    if result.get("lesson_finished"):
                        print("🏁 Урок завершен, останавливаем таймер")
                        self.stop_auto_continue_timer()
                else:
                    print(f"❌ Ошибка автоматического продолжения: {result.get('error', '')}")
                    self.stop_auto_continue_timer()
            else:
                print(f"❌ HTTP ошибка: {response.status_code}")
                self.stop_auto_continue_timer()
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка соединения при автоматическом продолжении: {e}")
            self.stop_auto_continue_timer()
        except Exception as e:
            print(f"❌ Неожиданная ошибка при автоматическом продолжении: {e}")
            self.stop_auto_continue_timer()
    
    def pause_auto_continue(self):
        """Временно приостанавливает таймер (при вопросе ученика)"""
        if self.timer_active and self.auto_continue_timer:
            self.auto_continue_timer.cancel()
            self.auto_continue_timer = None
            print(f"⏸️ Таймер приостановлен для комнаты {self.room_id}")
    
    def resume_auto_continue(self):
        """Возобновляет таймер после паузы"""
        if self.timer_active and not self.auto_continue_timer:
            self.start_auto_continue_timer()
            print(f"▶️ Таймер возобновлен для комнаты {self.room_id}")
    
    def is_timer_active(self) -> bool:
        """Проверяет, активен ли таймер"""
        return self.timer_active

# Создаем глобальный экземпляр менеджера уроков
lesson_manager = LessonManager()