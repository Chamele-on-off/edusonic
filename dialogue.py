import random
from typing import Dict, Optional, List
from difflib import SequenceMatcher
import json
from pathlib import Path
import time
import numpy as np
from knowledge.knowledge_base import KnowledgeBase
from llm import LLMIntegration

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
        self.knowledge_base = None
        self.llm = LLMIntegration()
        self._load_lessons()
        
        # Локальные шаблоны для быстрого доступа
        self.local_patterns = {
            "привет": ["Привет! Рад тебя видеть!", "Здравствуй! Готов к уроку?"],
            "как дела": ["Отлично! А у тебя как?", "Прекрасно! Как твои успехи?"],
            "спасибо": ["Пожалуйста! Всегда рад помочь!", "Не за что! Ты отлично справляешься!"],
            "не понимаю": ["Ничего страшного! Давай разберем вместе.", "Это нормально! Объясню еще раз."],
            "повтори": ["Конечно, повторяю...", "Давай еще раз."],
            "скучно": ["Давай сделаем урок интереснее!", "Предлагаю сменить активность!"],
            "трудно": ["Не переживай! Вместе разберемся.", "Сложности - это нормально! Я помогу."],
            "молодец": ["Спасибо! Стараюсь для вас", "Рад, что нравится!", "Вы тоже молодец!"],
            "хорошо": ["Отлично! Продолжаем!", "Супер! Двигаемся дальше!"],
            "не знаю": ["Ничего страшного! Сейчас разберемся.", "Это повод узнать новое!"]
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
                            'phases': data.get('phases', []),
                            'difficulty': data.get('difficulty', 'medium'),
                            'duration': data.get('duration', 1800)
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
            return None
            
        text_lower = text.lower().strip()
        
        # 1. Быстрая проверка локальных шаблонов
        for pattern, responses in self.local_patterns.items():
            if pattern in text_lower:
                return random.choice(responses)
        
        # 2. Проверка диалоговых шаблонов из базы знаний (если есть)
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
        
        # 4. Fallback с учетом состояния
        fallbacks = {
            "greeting": ["Давай начнем урок! Скажи 'привет'", "Готов начать занятие?"],
            "subject_selection": ["Выбери предмет из списка", "Какой предмет тебя интересует?"],
            "lesson_selection": ["Выбери урок из предложенных", "Какой урок хочешь пройти?"],
            "lesson_confirmation": ["Скажи 'готов' чтобы начать", "Жду твоего подтверждения"]
        }
        
        return random.choice(fallbacks.get(self.current_state, ["Давай продолжим наш урок."]))

    def _handle_greeting(self, text: str) -> Optional[str]:
        greeting_words = ["привет", "здравствуй", "начать", "старт", "готов", "поехали", "hello", "hi"]
        if any(word in text for word in greeting_words):
            self.current_state = "subject_selection"
            subjects = list(self.lessons.keys())
            
            if not subjects:
                return "К сожалению, уроки еще не загружены. Попробуй позже!"
                
            subject_list = "\n".join(f"{i+1}) {subj.capitalize()}" for i, subj in enumerate(subjects))
            return random.choice([
                f"Привет! Давай начнём урок. 🎓\nВыбери предмет:\n{subject_list}",
                f"Здравствуй! Рад начать наше занятие! 📚\nВыбери предмет:\n{subject_list}",
                f"Приветствую! Готов к увлекательному обучению! 🌟\nВыбери предмет:\n{subject_list}"
            ])
        return None

    def _handle_subject_selection(self, text: str) -> Optional[str]:
        subjects = list(self.lessons.keys())
        
        # Поиск по номеру
        for i, subject in enumerate(subjects):
            if str(i+1) in text:
                self.current_subject = subject
                self.current_state = "lesson_selection"
                return self._get_lesson_selection_message()
        
        # Поиск по названию
        for subject in subjects:
            if subject.lower() in text.lower():
                self.current_subject = subject
                self.current_state = "lesson_selection"
                return self._get_lesson_selection_message()
                
        # Возврат к приветствию
        if any(word in text for word in ["назад", "вернуться", "сначала"]):
            self.current_state = "greeting"
            return "Хорошо, давай начнем сначала! Скажи 'привет' чтобы продолжить."
            
        return None

    def _get_lesson_selection_message(self) -> str:
        """Формирует сообщение для выбора урока"""
        lessons = self.lessons[self.current_subject]
        lesson_list = "\n".join(
            f"{i+1}) {lesson['title']} "
            f"({'легкий' if lesson['difficulty'] == 'easy' else 'средний' if lesson['difficulty'] == 'medium' else 'сложный'}) - "
            f"{lesson['duration'] // 60} мин"
            for i, lesson in enumerate(lessons)
        )
        
        return random.choice([
            f"Отлично! Выбрал {self.current_subject.capitalize()}! 📖\nТеперь выбери урок:\n{lesson_list}",
            f"Прекрасный выбор! {self.current_subject.capitalize()} - это интересно! 🌟\nВыбери урок:\n{lesson_list}",
            f"Супер! {self.current_subject.capitalize()} - отличный предмет! 🎯\nКакой урок хочешь пройти?\n{lesson_list}"
        ])

    def _handle_lesson_selection(self, text: str) -> Optional[str]:
        if not self.current_subject:
            self.current_state = "subject_selection"
            return "Давай сначала выберем предмет. Какой тебе интересен?"
            
        lessons = self.lessons[self.current_subject]
        
        # Поиск по номеру
        for i, lesson in enumerate(lessons):
            if str(i+1) in text:
                self.selected_lesson = lesson['id']
                self.current_state = "lesson_confirmation"
                return self._get_lesson_confirmation_message(lesson)
        
        # Поиск по названию
        for lesson in lessons:
            if lesson['title'].lower() in text.lower():
                self.selected_lesson = lesson['id']
                self.current_state = "lesson_confirmation"
                return self._get_lesson_confirmation_message(lesson)
                
        # Возврат к выбору предмета
        if any(word in text for word in ["назад", "вернуться", "другой предмет"]):
            self.current_state = "subject_selection"
            self.current_subject = None
            return "Хорошо, давай выберем другой предмет!"
            
        return None

    def _get_lesson_confirmation_message(self, lesson: dict) -> str:
        """Формирует сообщение подтверждения выбора урока"""
        duration_min = lesson['duration'] // 60
        return random.choice([
            f"🎯 Отличный выбор! Ты выбрал: '{lesson['title']}'\n"
            f"⏱ Длительность: {duration_min} минут\n"
            f"📝 {lesson['description']}\n\n"
            f"Скажи 'готов' чтобы начать урок!",
            
            f"🌟 Прекрасно! Выбрал урок: '{lesson['title']}'\n"
            f"⏰ Время занятия: {duration_min} мин\n"
            f"📚 {lesson['description']}\n\n"
            f"Как будешь готов - скажи 'готов'!",
            
            f"🚀 Отлично! Начинаем: '{lesson['title']}'\n"
            f"🕐 Продолжительность: {duration_min} минут\n"
            f"📖 {lesson['description']}\n\n"
            f"Скажи 'готов' чтобы начать урок!"
        ])

    def _handle_lesson_confirmation(self, text: str) -> Optional[str]:
        ready_words = ["готов", "поехали", "начинаем", "старт", "давай", "начали", "погнали"]
        if any(word in text for word in ready_words):
            self.lesson_started = True
            # Инициализируем базу знаний для выбранного предмета
            self.knowledge_base = KnowledgeBase(self.current_subject)
            return random.choice([
                "🚀 Отлично! Начинаем урок! Сейчас я объясню материал...",
                "🎓 Поехали! Приступаем к изучению материала...",
                "📚 Отлично! Начинаем наш урок. Слушай внимательно...",
                "🌟 Прекрасно! Запускаем урок. Готовься узнавать новое..."
            ])
            
        # Возврат к выбору урока
        if any(word in text for word in ["назад", "вернуться", "другой урок"]):
            self.current_state = "lesson_selection"
            self.selected_lesson = None
            return "Хорошо, давай выберем другой урок!"
            
        return None

    def handle_question_during_lesson(self, question: str) -> str:
        """Обработка вопросов во время урока"""
        if not question.strip():
            return "Повтори, пожалуйста, вопрос. Я не расслышал."
            
        question_lower = question.lower().strip()
        
        # 1. Быстрая проверка локальных шаблонов
        for pattern, responses in self.local_patterns.items():
            if pattern in question_lower:
                return random.choice(responses)
        
        # 2. Проверка диалоговых шаблонов из базы знаний
        if self.knowledge_base:
            dialogue_response = self.knowledge_base.get_dialogue_response(question_lower)
            if dialogue_response:
                return dialogue_response
        
        # 3. Поиск в предметной базе знаний
        if self.knowledge_base:
            answer = self.knowledge_base.find_answer(question)
            if answer:
                return answer
        
        # 4. Запрос к LLM
        llm_response = self.llm.query(question, self.current_subject)
        if llm_response:
            # Сохраняем в кэш и базу знаний
            self.llm.add_to_cache(question, llm_response, self.current_subject)
            if self.knowledge_base:
                self.knowledge_base.add_knowledge(question=question, answer=llm_response)
            return llm_response
        
        # 5. Финальный fallback
        return random.choice([
            "Интересный вопрос! Давайте обсудим его подробнее.",
            "Хороший вопрос! Вернемся к нему в подходящий момент.",
            "Записал ваш вопрос. Обязательно разберем его.",
            "Это важный аспект! Обсудим его дополнительно."
        ])

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

    def get_current_subject(self) -> Optional[str]:
        """Возвращает текущий предмет"""
        return self.current_subject

    def get_current_state(self) -> str:
        """Возвращает текущее состояние диалога"""
        return self.current_state

    def reset(self):
        """Сброс состояния диалога"""
        self.current_state = "greeting"
        self.current_subject = None
        self.selected_lesson = None
        self.lesson_started = False
        self.knowledge_base = None

    def get_available_subjects(self) -> List[str]:
        """Возвращает список доступных предметов"""
        return list(self.lessons.keys())

    def get_lessons_for_subject(self, subject: str) -> List[dict]:
        """Возвращает уроки для указанного предмета"""
        return self.lessons.get(subject, [])

    def add_local_pattern(self, pattern: str, responses: List[str]):
        """Добавление локального шаблона"""
        self.local_patterns[pattern.lower()] = responses

    def get_dialogue_stats(self) -> Dict:
        """Получение статистики диалога"""
        if self.knowledge_base:
            return self.knowledge_base.get_stats()
        return {
            "local_patterns": len(self.local_patterns),
            "subjects_available": len(self.lessons),
            "current_state": self.current_state,
            "lesson_started": self.lesson_started
        }
