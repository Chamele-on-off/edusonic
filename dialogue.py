import random
from typing import Dict, Optional, List
from difflib import SequenceMatcher
import json
from pathlib import Path
import time
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
        self.selected_lesson_id = None
        self.selected_lesson_data = None
        self.lesson_started = False
        self.lessons_dir = Path("lessons")
        self.knowledge_base = None
        self.llm = LLMIntegration()
        self.available_lessons = self._load_lessons()
        
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

    def _load_lessons(self) -> Dict[str, List[Dict]]:
        """Загружает все доступные уроки"""
        lessons_by_subject = {}
        try:
            if not self.lessons_dir.exists():
                self.lessons_dir.mkdir(parents=True)
                print("Папка lessons создана")
                return lessons_by_subject
                
            for lesson_file in self.lessons_dir.glob("*.json"):
                try:
                    with open(lesson_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        subject = data.get('subject', 'other').lower()
                        lesson_id = data.get('id', 'unknown')
                        
                        if subject not in lessons_by_subject:
                            lessons_by_subject[subject] = []
                            
                        lessons_by_subject[subject].append({
                            'id': lesson_id,
                            'title': data.get('title', 'Без названия'),
                            'description': data.get('description', ''),
                            'lecture_texts': data.get('lecture_texts', []),
                            'file_path': str(lesson_file),
                            'difficulty': data.get('difficulty', 'medium'),
                            'duration': data.get('duration', 1800)
                        })
                        
                        print(f"Загружен урок: {subject}/{lesson_id}")
                        
                except Exception as e:
                    print(f"Ошибка загрузки урока {lesson_file}: {e}")
                    
        except Exception as e:
            print(f"Ошибка доступа к папке уроков: {e}")
            
        print(f"Загружено уроков: {sum(len(lessons) for lessons in lessons_by_subject.values())}")
        return lessons_by_subject

    def _load_lesson_data(self, subject: str, lesson_id: str) -> Optional[Dict]:
        """Загружает данные конкретного урока"""
        try:
            # Пробуем найти по subject_lesson_id.json
            lesson_filename = f"{subject}_{lesson_id}.json"
            lesson_path = self.lessons_dir / lesson_filename
            
            if not lesson_path.exists():
                # Ищем любой файл, содержащий lesson_id в названии
                for lesson_file in self.lessons_dir.glob(f"*{lesson_id}*.json"):
                    lesson_path = lesson_file
                    break
            
            if lesson_path.exists():
                with open(lesson_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
                    
        except Exception as e:
            print(f"Ошибка загрузки урока {lesson_id}: {e}")
            
        return None

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
                # Если это сигнал начала урока
                if responses == "LESSON_START_SIGNAL":
                    if self.current_state == "lesson_confirmation" and self.selected_lesson_id:
                        self.lesson_started = True
                        self.current_state = "lesson_active"
                        return "LESSON_START_SIGNAL"
                    else:
                        return "Сначала выбери предмет и урок."
                
                # Если это выбор предмета
                if responses.startswith("LESSON_SELECTED:"):
                    subject = responses.split(":")[1]
                    self.current_subject = subject
                    self.current_state = "lesson_selection"
                    return self._get_lesson_selection_message()
                
                # Для остальных шаблонов
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
        
        # Проверяем предметы
        for subject in self.available_lessons.keys():
            if subject in text.lower():
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
        # Проверяем предметы
        for subject in self.available_lessons.keys():
            if subject in text.lower():
                self.current_subject = subject
                self.current_state = "lesson_selection"
                return self._get_lesson_selection_message()
                
        if any(word in text for word in ["назад", "вернуться", "сначала"]):
            self.current_state = "greeting"
            return "Хорошо, давай начнем сначала!"
            
        return None

    def _get_lesson_selection_message(self) -> str:
        """Формирует сообщение для выбора урока"""
        if not self.current_subject or self.current_subject not in self.available_lessons:
            self.current_state = "subject_selection"
            return "Сначала выбери предмет."
            
        lessons = self.available_lessons[self.current_subject]
        
        if not lessons:
            self.current_state = "subject_selection"
            return f"Для предмета '{self.current_subject}' нет доступных уроков. Выбери другой предмет."
        
        # Если есть только один урок, сразу его выбираем
        if len(lessons) == 1:
            self.selected_lesson_id = lessons[0]['id']
            self.current_state = "lesson_confirmation"
            return f"Отлично! Начинаем урок '{lessons[0]['title']}'. Скажи 'готов' чтобы начать!"
        
        # Если несколько уроков, предлагаем выбор
        lesson_list = "\n".join(
            f"{i+1}) {lesson['title']} - {lesson['description']}"
            for i, lesson in enumerate(lessons[:5])  # Ограничиваем до 5 уроков
        )
        
        return f"Отлично! Выбрал {self.current_subject}!\nТеперь выбери урок:\n{lesson_list}\nПросто скажи номер урока."

    def _handle_lesson_selection(self, text: str) -> Optional[str]:
        if not self.current_subject or self.current_subject not in self.available_lessons:
            self.current_state = "subject_selection"
            return "Давай сначала выберем предмет."
            
        lessons = self.available_lessons[self.current_subject]
        
        # Поиск по номеру
        for i, lesson in enumerate(lessons):
            if str(i+1) in text:
                self.selected_lesson_id = lesson['id']
                self.current_state = "lesson_confirmation"
                return f"Отлично! Ты выбрал: '{lesson['title']}'. Скажи 'готов' чтобы начать урок!"
        
        # Поиск по названию
        for lesson in lessons:
            if lesson['title'].lower() in text.lower():
                self.selected_lesson_id = lesson['id']
                self.current_state = "lesson_confirmation"
                return f"Отлично! Ты выбрал: '{lesson['title']}'. Скажи 'готов' чтобы начать урок!"
                
        # Возврат к выбору предмета
        if any(word in text for word in ["назад", "вернуться", "другой предмет"]):
            self.current_state = "subject_selection"
            self.current_subject = None
            return "Хорошо, давай выберем другой предмет!"
            
        return None

    def _handle_lesson_confirmation(self, text: str) -> Optional[str]:
        ready_words = ["готов", "поехали", "начинаем", "старт", "давай", "начали", "да"]
        
        if any(word in text for word in ready_words):
            if self.selected_lesson_id and self.current_subject:
                # Загружаем данные урока
                self.selected_lesson_data = self._load_lesson_data(self.current_subject, self.selected_lesson_id)
                
                if self.selected_lesson_data:
                    self.lesson_started = True
                    self.current_state = "lesson_active"
                    
                    # Инициализируем базу знаний
                    try:
                        self.knowledge_base = KnowledgeBase(self.current_subject)
                        print(f"База знаний инициализирована для предмета: {self.current_subject}")
                    except Exception as e:
                        print(f"Ошибка инициализации базы знаний: {e}")
                        self.knowledge_base = None
                    
                    return "LESSON_START_SIGNAL"
                else:
                    return "Не удалось загрузить урок. Попробуй другой."
            else:
                return "Сначала выбери урок."
                
        # Возврат к выбору урока
        if any(word in text for word in ["назад", "вернуться", "другой урок", "нет"]):
            self.current_state = "lesson_selection"
            self.selected_lesson_id = None
            return "Хорошо, давай выберем другой урок!"
            
        return None

    def _handle_lesson_active(self, text: str) -> Optional[str]:
        """Обработка ввода во время активного урока"""
        return self.handle_question_during_lesson(text)

    def handle_question_during_lesson(self, question: str) -> str:
        """Обработка вопросов во время урока"""
        if not question.strip():
            return "Повтори, пожалуйста, вопрос. Я не расслышал."
            
        question_lower = question.lower().strip()
        
        # Команды управления уроком
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

    def get_selected_lesson(self) -> Optional[dict]:
        """Возвращает данные выбранного урока"""
        return self.selected_lesson_data

    def is_lesson_started(self) -> bool:
        """Проверяет, начат ли урок"""
        return self.lesson_started

    def get_current_subject(self) -> Optional[str]:
        """Возвращает текущий предмет"""
        return self.current_subject

    def get_current_state(self) -> str:
        """Возвращает текущее состояние диалога"""
        return self.current_state

    def get_lecture_texts(self) -> List[str]:
        """Возвращает тексты лекции"""
        if self.selected_lesson_data:
            return self.selected_lesson_data.get('lecture_texts', [])
        return []

    def get_available_lessons(self, subject: str = None) -> List[Dict]:
        """Возвращает список доступных уроков"""
        if subject:
            return self.available_lessons.get(subject, [])
        else:
            all_lessons = []
            for lessons in self.available_lessons.values():
                all_lessons.extend(lessons)
            return all_lessons

    def reset(self):
        """Сброс состояния диалога"""
        self.current_state = "greeting"
        self.current_subject = None
        self.selected_lesson_id = None
        self.selected_lesson_data = None
        self.lesson_started = False
        self.knowledge_base = None
