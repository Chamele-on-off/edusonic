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
            "lesson_active": self._handle_lesson_active
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
            "привет": ["Привет! Какой предмет хочешь изучить?", "Здравствуй! Давай начнем урок. Просто скажи название предмета."],
            "как дела": ["Отлично! Готов помочь тебе с уроками. Какой предмет интересует?", "Прекрасно! Какой урок хочешь начать?"],
            "спасибо": ["Пожалуйста! Всегда рад помочь!", "Не за что! Ты отлично справляешься!"],
            "не понимаю": ["Ничего страшного! Давай разберем вместе.", "Это нормально! Объясню еще раз."],
            "повтори": ["Конечно, повторяю.", "Давай еще раз."],
            "скучно": ["Давай выберем интересный предмет! Что тебе нравится?", "Предлагаю сменить тему! Какой предмет хочешь?"],
            "трудно": ["Не переживай! Вместе разберемся.", "Сложности - это нормально! Я помогу."],
            "молодец": ["Спасибо! Стараюсь для вас", "Рад, что нравится!", "Вы тоже молодец!"],
            "хорошо": ["Отлично! Продолжаем!", "Супер! Двигаемся дальше!"],
            "не знаю": ["Ничего страшного! Сейчас разберемся.", "Это повод узнать новое!"],
            "начать": ["Отлично! Какой предмет хочешь изучать?", "Давай начнем! Просто назови предмет."],
            "урок": ["Какой урок хочешь начать?", "Отлично! Назови предмет для урока."],
            "математика": ["Математика - отличный выбор! Начинаем урок по математике.", "Отлично! Запускаю урок математики."],
            "обществознание": ["Обществознание - интересный предмет! Начинаем урок.", "Хорошо! Запускаю урок обществознания."],
            "русский": ["Русский язык - важно знать! Начинаем урок.", "Отлично! Запускаю урок русского языка."],
            "физика": ["Физика - увлекательная наука! Начинаем урок.", "Хорошо! Запускаю урок физики."],
            "химия": ["Химия - это интересно! Начинаем урок.", "Отлично! Запускаю урок химии."],
            "биология": ["Биология - изучаем живую природу! Начинаем урок.", "Хорошо! Запускаю урок биологии."],
            "история": ["История - познаем прошлое! Начинаем урок.", "Отлично! Запускаю урок истории."],
            "английский": ["Английский язык - полезно знать! Начинаем урок.", "Хорошо! Запускаю урок английского."],
            "информатика": ["Информатика - современный предмет! Начинаем урок.", "Отлично! Запускаю урок информатики."],
            "готов": ["Отлично! Начинаем урок!", "Супер! Приступаем к занятию!"],
            "да": ["Хорошо, продолжаем!", "Отлично! Двигаемся дальше!"]
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
        text_lower = text.lower().strip()
        
        # Сначала обрабатываем текущее состояние, если это подтверждение урока
        # Это позволяет вернуть LESSON_START_SIGNAL до проверки lesson_started
        if self.current_state == "lesson_confirmation":
            print(f"Обработка подтверждения урока: '{text_lower}'")
            handler = self.dialogue_states.get(self.current_state)
            if handler:
                response = handler(text_lower)
                if response == "LESSON_START_SIGNAL":
                    print("ВОЗВРАЩАЕМ СИГНАЛ НАЧАЛА УРОКА")
                    return response  # Возвращаем сигнал немедленно
        
        # Если урок уже начат, обрабатываем как вопрос во время урока
        if self.lesson_started:
            print("Урок уже начат, обрабатываем как вопрос")
            return self.handle_question_during_lesson(text)
        
        # 1. Быстрая проверка локальных шаблонов (приоритет для предметов)
        for pattern, responses in self.local_patterns.items():
            if pattern in text_lower:
                # Если это название предмета, сразу переходим к выбору урока
                if pattern in ["математика", "обществознание", "русский", "физика", 
                              "химия", "биология", "история", "английский", "информатика"]:
                    self.current_subject = pattern
                    self.current_state = "lesson_selection"
                    return self._get_lesson_selection_message()
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
            "greeting": ["Просто скажи название предмета, например 'математика' или 'обществознание'", "Какой предмет тебя интересует? Просто назови его."],
            "subject_selection": ["Просто скажи название предмета", "Какой предмет хочешь изучать?"],
            "lesson_selection": ["Выбери урок из предложенных", "Какой урок хочешь пройти?"],
            "lesson_confirmation": ["Скажи 'готов' чтобы начать", "Жду твоего подтверждения"],
            "lesson_active": ["Задавайте вопросы по уроку!", "Что вас интересует из пройденного материала?"]
        }
        
        return random.choice(fallbacks.get(self.current_state, ["Давай продолжим наш урок."]))

    def _handle_greeting(self, text: str) -> Optional[str]:
        greeting_words = ["привет", "здравствуй", "начать", "старт", "готов", "поехали", "hello", "hi"]
        subject_words = ["математика", "обществознание", "русский", "физика", "химия", 
                        "биология", "история", "английский", "информатика", "урок"]
        
        # Если сразу назван предмет
        for subject in self.lessons.keys():
            if subject in text:
                self.current_subject = subject
                self.current_state = "lesson_selection"
                return self._get_lesson_selection_message()
                
        # Если названо общее слово "урок"
        if "урок" in text:
            self.current_state = "subject_selection"
            return "Отлично! Какой предмет хочешь изучать?"
            
        # Обычное приветствие
        if any(word in text for word in greeting_words):
            self.current_state = "subject_selection"
            return "Привет! Просто скажи название предмета, который хочешь изучать, например 'математика' или 'обществознание'"
            
        return None

    def _handle_subject_selection(self, text: str) -> Optional[str]:
        subjects = list(self.lessons.keys())
        
        # Поиск по названию предмета
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
        lessons = self.lessons.get(self.current_subject, [])
        
        if not lessons:
            self.current_state = "subject_selection"
            return f"К сожалению, для предмета '{self.current_subject}' нет доступных уроков. Выбери другой предмет."
        
        # Если есть только один урок, сразу его выбираем
        if len(lessons) == 1:
            self.selected_lesson = lessons[0]['id']
            self.current_state = "lesson_confirmation"
            return f"Отлично! Начинаем урок '{lessons[0]['title']}'. Скажи 'готов' чтобы начать!"
        
        # Если несколько уроков, предлагаем выбор
        lesson_list = "\n".join(
            f"{i+1}) {lesson['title']}"
            for i, lesson in enumerate(lessons)
        )
        
        return f"Отлично! Выбрал {self.current_subject.capitalize()}!\nТеперь выбери урок:\n{lesson_list}\nПросто скажи номер урока."

    def _handle_lesson_selection(self, text: str) -> Optional[str]:
        if not self.current_subject:
            self.current_state = "subject_selection"
            return "Давай сначала выберем предмет. Какой тебе интересен?"
            
        lessons = self.lessons.get(self.current_subject, [])
        
        if not lessons:
            self.current_state = "subject_selection"
            return f"Для предмета '{self.current_subject}' нет уроков. Выбери другой предмет."
        
        # Поиск по номеру
        for i, lesson in enumerate(lessons):
            if str(i+1) in text:
                self.selected_lesson = lesson['id']
                self.current_state = "lesson_confirmation"
                return f"Отлично! Ты выбрал: '{lesson['title']}'. Скажи 'готов' чтобы начать урок!"
        
        # Поиск по названию
        for lesson in lessons:
            if lesson['title'].lower() in text.lower():
                self.selected_lesson = lesson['id']
                self.current_state = "lesson_confirmation"
                return f"Отлично! Ты выбрал: '{lesson['title']}'. Скажи 'готов' чтобы начать урок!"
                
        # Возврат к выбору предмета
        if any(word in text for word in ["назад", "вернуться", "другой предмет"]):
            self.current_state = "subject_selection"
            self.current_subject = None
            return "Хорошо, давай выберем другой предмет!"
            
        return None

    def _handle_lesson_confirmation(self, text: str) -> Optional[str]:
        ready_words = ["готов", "поехали", "начинаем", "старт", "давай", "начали", "погнали", "yes", "да"]
        if any(word in text for word in ready_words):
            print("ПОЛЬЗОВАТЕЛЬ СКАЗАЛ 'ГОТОВ' - УСТАНАВЛИВАЕМ ФЛАГ И ВОЗВРАЩАЕМ СИГНАЛ")
            self.lesson_started = True
            self.current_state = "lesson_active"
            
            # Инициализируем базу знаний для выбранного предмета
            try:
                self.knowledge_base = KnowledgeBase(self.current_subject)
                print(f"База знаний инициализирована для предмета: {self.current_subject}")
                if self.knowledge_base:
                    print(f"Доступные термины: {list(self.knowledge_base.data['terms'].keys())}")
            except Exception as e:
                print(f"Ошибка инициализации базы знаний: {e}")
                self.knowledge_base = None
            
            print(f"Урок готов к запуску: {self.is_lesson_ready_to_start()}")
            
            # Возвращаем сигнал для запуска урока
            return "LESSON_START_SIGNAL"
                
        # Возврат к выбору урока
        if any(word in text for word in ["назад", "вернуться", "другой урок", "нет"]):
            self.current_state = "lesson_selection"
            self.selected_lesson = None
            return "Хорошо, давай выберем другой урок!"
            
        return None

    def _handle_lesson_active(self, text: str) -> Optional[str]:
        """Обработка ввода во время активного урока"""
        # Во время урока все вопросы обрабатываются как вопросы по материалу
        return self.handle_question_during_lesson(text)

    def is_lesson_ready_to_start(self) -> bool:
        """Проверяет, готов ли урок к запуску"""
        return (self.lesson_started and 
                self.current_subject is not None and 
                self.selected_lesson is not None and
                self.get_selected_lesson() is not None)

    def handle_question_during_lesson(self, question: str) -> str:
        """Обработка вопросов во время урока"""
        if not question.strip():
            return "Повтори, пожалуйста, вопрос. Я не расслышал."
            
        question_lower = question.lower().strip()
        print(f"Обработка вопроса во время урока: '{question_lower}'")
        print(f"Текущий предмет: {self.current_subject}")
        print(f"База знаний доступна: {self.knowledge_base is not None}")
        
        if self.knowledge_base:
            print(f"Доступные термины в базе: {list(self.knowledge_base.data['terms'].keys())}")
        
        # 1. Быстрая проверка локальных шаблонов
        for pattern, responses in self.local_patterns.items():
            if pattern in question_lower:
                print(f"Найден локальный шаблон: {pattern}")
                return random.choice(responses)
        
        # 2. Проверка диалоговых шаблонов из базы знаний
        if self.knowledge_base:
            dialogue_response = self.knowledge_base.get_dialogue_response(question_lower)
            if dialogue_response:
                print(f"Найден диалоговый шаблон: {dialogue_response}")
                return dialogue_response
        
        # 3. Поиск в предметной базе знаний (ПРИОРИТЕТ!)
        if self.knowledge_base:
            answer = self.knowledge_base.find_answer(question)
            if answer:
                print(f"Ответ найден в базе знаний: {answer}")
                return answer
        
        # 4. Запрос к LLM (только если не нашли в базе знаний)
        print("Ответ не найден в базе знаний, обращаюсь к LLM")
        llm_response = self.llm.query(question, self.current_subject)
        if llm_response:
            # Сохраняем в кэш и базу знаний для будущего использования
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
        if not self.current_subject or not self.selected_lesson:
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