import random
from typing import Dict, Optional, List
from difflib import SequenceMatcher
from pathlib import Path
import time
from knowledge.knowledge_base import KnowledgeBase
from llm import LLMIntegration
from lecture import LectureManager

class DialogueManager:
    def __init__(self, socketio):
        self.socketio = socketio
        self.lecture_manager = LectureManager(socketio)
        self.dialogue_states = {
            "greeting": self._handle_greeting,
            "subject_selection": self._handle_subject_selection,
            "lesson_selection": self._handle_lesson_selection,
            "lesson_confirmation": self._handle_lesson_confirmation,
            "lesson_active": self._handle_lesson_active
        }
        self.current_state = "greeting"
        self.current_subject = None
        self.selected_lesson = None
        self.lesson_started = False
        self.knowledge_base = None
        self.llm = LLMIntegration()
        
        # Локальные шаблоны для быстрого доступа
        self.local_patterns = {
            "привет": ["Привет! Какой предмет хочешь изучить?", "Здравствуй! Давай начнем урок. Просто скажи название предмета."],
            "как дела": ["Отлично! Готов помочь тебе с уроками. Какой предмет интересует?", "Прекрасно! Какой урок хочешь начать?"],
            "спасибо": ["Пожалуйста! Всегда рад помочь!", "Не за что! Ты отлично справляешься!"],
            "не понимаю": ["Ничего страшного! Давай разберем вместе.", "Это нормально! Объясню еще раз."],
            "повтори": ["Конечно, повторяю.", "Давай еще раз."],
            "скучно": ["Давай выберем интересный предмет! Что тебе нравится?", "Предлагаю сменим тему! Какой предмет хочешь?"],
            "трудно": ["Не переживай! Вместе разберемся.", "Сложности - это нормально! Я помогу."],
            "молодец": ["Спасибо! Стараюсь для вас", "Рад, что нравится!", "Вы тоже молодец!"],
            "хорошо": ["Отлично! Продолжаем!", "Супер! Двигаемся дальше!"],
            "не знаю": ["Ничего страшного! Сейчас разберемся.", "Это повод узнать новое!"],
            "начать": ["Отлично! Какой предмет хочешь изучать?", "Давай начнем! Просто назови предмет."],
            "урок": ["Какой урок хочешь начать?", "Отлично! Назови предмет для урока."],
            "математика": "LESSON_SELECTED:математика",
            "обществознание": "LESSON_SELECTED:обществознание",
            "русский": "LESSON_SELECTED:русский",
            "физика": "LESSON_SELECTED:физика",
            "химия": "LESSON_SELECTED:химия",
            "биология": "LESSON_SELECTED:биология",
            "история": "LESSON_SELECTED:история",
            "английский": "LESSON_SELECTED:английский",
            "информатика": "LESSON_SELECTED:информатика",
            "готов": "LESSON_START_SIGNAL",
            "да": "LESSON_START_SIGNAL",
            "поехали": "LESSON_START_SIGNAL",
            "начинаем": "LESSON_START_SIGNAL"
        }

    def _similarity(self, a: str, b: str) -> float:
        """Вычисление схожести строк"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def process_input(self, text: str) -> str:
        """Обработка входящего текста и генерация ответа"""
        text_lower = text.lower().strip()
        
        # Если урок уже начат, обрабатываем как вопрос во время урока
        if self.lesson_started:
            print("Урок уже начат, обрабатываем как вопрос")
            return self.handle_question_during_lesson(text)
        
        # 1. Быстрая проверка локальных шаблонов
        for pattern, responses in self.local_patterns.items():
            if pattern in text_lower:
                if responses == "LESSON_START_SIGNAL":
                    if self.current_state == "lesson_confirmation":
                        self.lesson_started = True
                        self.current_state = "lesson_active"
                        return "LESSON_START_SIGNAL"
                    else:
                        return "Сначала выбери предмет и урок."
                
                if responses.startswith("LESSON_SELECTED:"):
                    subject = responses.split(":")[1]
                    self.current_subject = subject
                    self.current_state = "lesson_selection"
                    return self._get_lesson_selection_message()
                
                if isinstance(responses, list):
                    return random.choice(responses)
                return responses
        
        # 2. Проверка диалоговых шаблонов из базы знаний
        if self.knowledge_base:
            dialogue_response = self.knowledge_base.get_dialogue_response(text_lower)
            if dialogue_response:
                return dialogue_response
        
        # 3. Обработка по текущему состоянию
        handler = self.dialogue_states.get(self.current_state)
        if handler:
            response = handler(text_lower)
            if response:
                return response
        
        # 4. Fallback
        fallbacks = {
            "greeting": ["Просто скажи название предмета, например 'математика' или 'обществознание'"],
            "subject_selection": ["Какой предмет хочешь изучать?"],
            "lesson_selection": ["Выбери урок из предложенных"],
            "lesson_confirmation": ["Скажи 'готов' чтобы начать"],
            "lesson_active": ["Задавайте вопросы по уроку!"]
        }
        
        return random.choice(fallbacks.get(self.current_state, ["Давай продолжим наш урок."]))

    def _handle_greeting(self, text: str) -> Optional[str]:
        greeting_words = ["привет", "здравствуй", "начать", "старт", "готов", "поехали"]
        
        for subject in self.lecture_manager.get_available_subjects():
            if subject in text:
                self.current_subject = subject
                self.current_state = "lesson_selection"
                return self._get_lesson_selection_message()
                
        if "урок" in text:
            self.current_state = "subject_selection"
            return "Отлично! Какой предмет хочешь изучать?"
            
        if any(word in text for word in greeting_words):
            self.current_state = "subject_selection"
            return "Привет! Просто скажи название предмета, например 'математика' или 'обществознание'"
            
        return None

    def _handle_subject_selection(self, text: str) -> Optional[str]:
        subjects = [lesson['subject'] for lesson in self.lecture_manager.get_available_lessons()]
        
        for subject in subjects:
            if subject.lower() in text.lower():
                self.current_subject = subject
                self.current_state = "lesson_selection"
                return self._get_lesson_selection_message()
                
        if any(word in text for word in ["назад", "вернуться", "сначала"]):
            self.current_state = "greeting"
            return "Хорошо, давай начнем сначала!"
            
        return None

    def _get_lesson_selection_message(self) -> str:
        """Формирует сообщение для выбора урока"""
        lessons = self.lecture_manager.get_available_lessons(self.current_subject)
        
        if not lessons:
            self.current_state = "subject_selection"
            return f"Для предмета '{self.current_subject}' нет доступных уроков."
        
        if len(lessons) == 1:
            self.selected_lesson = lessons[0]['id']
            self.current_state = "lesson_confirmation"
            return f"Отлично! Начинаем урок '{lessons[0]['title']}'. Скажи 'готов' чтобы начать!"
        
        lesson_list = " ".join([f"{i+1}) {lesson['title']}" for i, lesson in enumerate(lessons)])
        return f"Выбери урок: {lesson_list}. Скажи номер."

    def _handle_lesson_selection(self, text: str) -> Optional[str]:
        lessons = self.lecture_manager.get_available_lessons(self.current_subject)
        
        for i, lesson in enumerate(lessons):
            if str(i+1) in text or lesson['title'].lower() in text.lower():
                self.selected_lesson = lesson['id']
                self.current_state = "lesson_confirmation"
                return f"Выбран урок '{lesson['title']}'. Скажи 'готов' чтобы начать!"
        
        if any(word in text for word in ["назад", "вернуться", "другой предмет"]):
            self.current_state = "subject_selection"
            self.current_subject = None
            return "Хорошо, давай выберем другой предмет!"
            
        return None

    def _handle_lesson_confirmation(self, text: str) -> Optional[str]:
        ready_words = ["готов", "поехали", "начинаем", "старт", "давай", "начали", "да"]
        
        if any(word in text for word in ready_words):
            self.lesson_started = True
            self.current_state = "lesson_active"
            
            # Загружаем урок
            lesson_data = self.lecture_manager.load_lesson(self.selected_lesson, self.current_subject)
            if lesson_data:
                # Инициализируем базу знаний
                try:
                    self.knowledge_base = KnowledgeBase(self.current_subject)
                except:
                    self.knowledge_base = None
                
                return "LESSON_START_SIGNAL"
            else:
                return "Ошибка загрузки урока. Попробуй другой."
                
        if any(word in text for word in ["назад", "вернуться", "другой урок", "нет"]):
            self.current_state = "lesson_selection"
            self.selected_lesson = None
            return "Хорошо, давай выберем другой урок!"
            
        return None

    def _handle_lesson_active(self, text: str) -> Optional[str]:
        return self.handle_question_during_lesson(text)

    def handle_question_during_lesson(self, question: str) -> str:
        if not question.strip():
            return "Повтори, пожалуйста, вопрос."
            
        question_lower = question.lower().strip()
        
        # Команды управления лекцией
        control_commands = {
            "пауза": "Ставлю на паузу.",
            "продолжи": "Продолжаем!",
            "дальше": "Переходим дальше.",
            "стоп": "Останавливаю урок.",
            "повтори": "Повторяю."
        }
        
        for cmd, response in control_commands.items():
            if cmd in question_lower:
                return response
        
        # Локальные шаблоны
        for pattern, responses in self.local_patterns.items():
            if pattern in question_lower:
                if isinstance(responses, list):
                    return random.choice(responses)
                return responses
        
        # База знаний
        if self.knowledge_base:
            answer = self.knowledge_base.find_answer(question)
            if answer:
                return answer
        
        # LLM
        llm_response = self.llm.query(question, self.current_subject)
        if llm_response:
            if self.knowledge_base:
                self.knowledge_base.add_knowledge(question=question, answer=llm_response)
            return llm_response
        
        return "Интересный вопрос! Давайте обсудим его подробнее."

    def start_lecture(self, room_id: str, speak_callback):
        """Начинает лекцию"""
        return self.lecture_manager.start_lecture(room_id, speak_callback)

    def pause_lecture(self):
        """Приостанавливает лекцию"""
        self.lecture_manager.pause_lecture()

    def resume_lecture(self):
        """Возобновляет лекцию"""
        self.lecture_manager.resume_lecture()

    def stop_lecture(self):
        """Останавливает лекцию"""
        self.lecture_manager.stop_lecture()

    def get_selected_lesson(self) -> Optional[dict]:
        return self.lecture_manager.current_lesson

    def is_lesson_started(self) -> bool:
        return self.lesson_started

    def get_current_subject(self) -> Optional[str]:
        return self.current_subject

    def get_current_state(self) -> str:
        return self.current_state

    def reset(self):
        self.current_state = "greeting"
        self.current_subject = None
        self.selected_lesson = None
        self.lesson_started = False
        self.knowledge_base = None
        self.lecture_manager.stop_lecture()