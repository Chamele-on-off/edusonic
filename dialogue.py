import random
from typing import Dict, Optional, List
from difflib import SequenceMatcher
import json
from pathlib import Path
import time

class DialogueManager:
    def __init__(self, socketio):
        self.socketio = socketio
        self.dialogue_states = {
            "greeting": self._handle_greeting,
            "subject_selection": self._handle_subject_selection,
            "lesson_selection": self._handle_lesson_selection,
            "lesson_confirmation": self._handle_lesson_confirmation,
        }
        self.current_state = "greeting"
        self.current_subject = None
        self.selected_lesson = None
        self.lesson_started = False
        self.lessons_dir = Path("lessons")
        self._load_lessons()
        
        # База знаний для диалога
        self.qa_knowledge = {
            "привет": ["Привет! Рад тебя видеть!", "Здравствуй! Готов к уроку?"],
            "как дела": ["Отлично! А у тебя как?", "Прекрасно! Как твои успехи?"],
            "спасибо": ["Пожалуйста! Всегда рад помочь!", "Не за что! Ты отлично справляешься!"],
            "не понимаю": ["Ничего страшного! Давай разберем вместе.", "Это нормально! Объясню еще раз."],
            "повтори": ["Конечно, повторяю...", "Давай еще раз."]
        }

    def _load_lessons(self):
        """Загружает список доступных уроков"""
        self.lessons = {}
        try:
            if not self.lessons_dir.exists():
                self.lessons_dir.mkdir(parents=True)
                return
                
            for lesson_file in self.lessons_dir.glob("*.json"):
                try:
                    with open(lesson_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        subject = data.get('subject', 'other')
                        if subject not in self.lessons:
                            self.lessons[subject] = []
                        self.lessons[subject].append({
                            'id': data.get('id', 'unknown'),
                            'title': data.get('title', 'Без названия'),
                            'description': data.get('description', ''),
                            'phases': data.get('phases', [])
                        })
                except Exception as e:
                    print(f"Ошибка загрузки урока {lesson_file}: {e}")
        except Exception as e:
            print(f"Ошибка доступа к папке уроков: {e}")

    def _similarity(self, a: str, b: str) -> float:
        """Вычисление схожести строк"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def process_input(self, text: str) -> str:
        """Обработка входящего текста и генерация ответа"""
        if self.lesson_started:
            return None  # Урок начат, управление передано внешнему обработчику
            
        text_lower = text.lower().strip()
        
        # Проверка общих фраз
        for pattern, responses in self.qa_knowledge.items():
            if pattern in text_lower:
                return random.choice(responses)
        
        # Обработка по текущему состоянию
        handler = self.dialogue_states.get(self.current_state)
        if handler:
            response = handler(text_lower)
            if response:
                return response
        
        return "Давай продолжим выбор урока. Выбери предмет или урок из предложенных."

    def _handle_greeting(self, text: str) -> Optional[str]:
        if any(word in text for word in ["привет", "здравствуй", "начать", "старт"]):
            self.current_state = "subject_selection"
            subjects = list(self.lessons.keys())
            
            if not subjects:
                return "К сожалению, уроки еще не загружены."
                
            return (f"Привет! Давай начнём урок. Выбери предмет:\n" +
                    "\n".join(f"{i+1}) {subj}" for i, subj in enumerate(subjects)))
        return None

    def _handle_subject_selection(self, text: str) -> Optional[str]:
        subjects = list(self.lessons.keys())
        
        for i, subject in enumerate(subjects):
            if str(i+1) in text or subject.lower() in text.lower():
                self.current_subject = subject
                self.current_state = "lesson_selection"
                lessons = self.lessons[subject]
                return (f"Отлично! Выбран предмет: {subject}\n" +
                        "Выбери урок:\n" +
                        "\n".join(f"{i+1}) {lesson['title']}" 
                                for i, lesson in enumerate(lessons)))
        return None

    def _handle_lesson_selection(self, text: str) -> Optional[str]:
        if not self.current_subject:
            return "Давай сначала выберем предмет."
            
        lessons = self.lessons[self.current_subject]
        
        for i, lesson in enumerate(lessons):
            if str(i+1) in text or lesson['title'].lower() in text.lower():
                self.selected_lesson = lesson['id']
                self.current_state = "lesson_confirmation"
                return (f"Отличный выбор! Ты выбрал: {lesson['title']}\n" +
                        f"{lesson['description']}\n" +
                        "Скажи 'готов' чтобы начать урок.")
        return None

    def _handle_lesson_confirmation(self, text: str) -> Optional[str]:
        if "готов" in text.lower():
            self.lesson_started = True
            return "Отлично! Начинаем урок!"
        return None

    def get_selected_lesson(self) -> Optional[dict]:
        """Возвращает данные выбранного урока"""
        if not self.lesson_started or not self.selected_lesson or not self.current_subject:
            return None
            
        for lesson in self.lessons.get(self.current_subject, []):
            if lesson['id'] == self.selected_lesson:
                return lesson
        return None

    def is_lesson_started(self) -> bool:
        """Проверяет, начат ли урок"""
        return self.lesson_started

    def reset(self):
        """Сброс состояния диалога"""
        self.current_state = "greeting"
        self.current_subject = None
        self.selected_lesson = None
        self.lesson_started = False
