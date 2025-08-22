import random
from typing import Dict, Optional, List
from difflib import SequenceMatcher
import json
from pathlib import Path

class DialogueManager:
    def __init__(self):
        self.dialogue_states = {
            "greeting": self._handle_greeting,
            "subject_selection": self._handle_subject_selection,
            "lesson_selection": self._handle_lesson_selection,
            "lesson": self._handle_lesson,
            "practice": self._handle_practice,
            "qa": self._handle_qa,
            "farewell": self._handle_farewell
        }
        self.current_state = "greeting"
        self.current_subject = None
        self.selected_lesson = None
        self.lessons_dir = Path("lessons")
        self._load_lessons()

    def _load_lessons(self):
        """Загружает список доступных уроков"""
        self.lessons = {}
        for lesson_file in self.lessons_dir.glob("*.json"):
            with open(lesson_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                subject = data['subject']
                if subject not in self.lessons:
                    self.lessons[subject] = []
                self.lessons[subject].append({
                    'id': data['id'],
                    'title': data['title'],
                    'description': data['description']
                })

    def _similarity(self, a: str, b: str) -> float:
        """Вычисление схожести строк"""
        return SequenceMatcher(None, a, b).ratio()

    def _find_best_match(self, text: str, patterns: List[str], threshold: float = 0.6) -> Optional[str]:
        """Поиск наиболее подходящего вопроса в базе знаний"""
        text_lower = text.lower()
        best_match = None
        highest_similarity = 0.0
        
        for pattern in patterns:
            similarity = self._similarity(text_lower, pattern.lower())
            if similarity > highest_similarity and similarity >= threshold:
                highest_similarity = similarity
                best_match = pattern
                
        return best_match if highest_similarity >= threshold else None

    def process_input(self, text: str) -> str:
        """Обработка входящего текста и генерация ответа"""
        text_lower = text.lower().strip()
        
        # Обработка по текущему состоянию
        handler = self.dialogue_states.get(self.current_state)
        if handler:
            response = handler(text_lower)
            if response:
                return response
        
        # Fallback-ответ
        return random.choice([
            "Я не совсем понял. Можешь уточнить?",
            "Извини, я не уверен, что правильно тебя понял.",
            "Можешь сказать это другими словами?"
        ])

    def _handle_greeting(self, text: str) -> Optional[str]:
        if any(word in text for word in ["привет", "здравствуй", "добрый", "начать", "старт"]):
            self.current_state = "subject_selection"
            subjects = list(self.lessons.keys())
            return (f"Привет! Давай начнём наш урок. Какой предмет будем изучать?\n" +
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
                self.current_state = "lesson"
                return (f"Отличный выбор! Начинаем урок: {lesson['title']}\n" +
                        f"{lesson['description']}\n" +
                        "Скажи 'готов', когда будешь готов начать.")
        return None

    def _handle_lesson(self, text: str) -> Optional[str]:
        if "готов" in text.lower():
            return f"Начинаем урок {self.selected_lesson}! Сейчас я объясню материал."
        return None

    def _handle_practice(self, text: str) -> Optional[str]:
        if any(word in text for word in ["ответ", "думаю", "считаю"]):
            return "Интересный ответ! Давай проверим..." if random.random() > 0.3 else "Почти правильно, но давай уточним..."
        return None

    def _handle_qa(self, text: str) -> Optional[str]:
        if any(word in text for word in ["вопрос", "не знаю", "объясни"]):
            return "Хороший вопрос! Давай разберёмся..."
        return None

    def _handle_farewell(self, text: str) -> Optional[str]:
        if any(word in text for word in ["конец", "закончи", "хватит", "до свидан"]):
            return "Отлично поработали! До следующего урока!"
        return None

    def get_selected_lesson(self) -> Optional[str]:
        """Возвращает ID выбранного урока"""
        return self.selected_lesson

    def reset(self):
        """Сброс состояния диалога"""
        self.current_state = "greeting"
        self.current_subject = None
        self.selected_lesson = None
