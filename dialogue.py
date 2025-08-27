import random
from typing import Dict, Optional, List
from difflib import SequenceMatcher
import json
from pathlib import Path
import time
import re
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
            "lesson_reading": self._handle_lesson_reading
        }
        self.current_state = "greeting"
        self.current_subject = None
        self.selected_lesson = None
        self.lesson_started = False
        self.lesson_content = []
        self.current_paragraph = 0
        self.lessons_dir = Path("lessons")
        self.knowledge_base = None
        self.llm = LLMIntegration()
        self._load_lessons()
        
        # Локальные шаблоны для быстрого доступа
        self.local_patterns = {
            "привет": ["Привет. Рад вас видеть.", "Здравствуйте. Готовы к уроку?"],
            "как дела": ["Все хорошо. Продолжим урок.", "Нормально. Как ваши успехи?"],
            "спасибо": ["Пожалуйста.", "Всегда рад помочь."],
            "не понимаю": ["Давайте разберем еще раз.", "Объясню по-другому."],
            "повтори": ["Повторяю.", "Скажу еще раз."],
            "скучно": ["Давайте сменим активность.", "Предложу другой подход."],
            "трудно": ["Разберемся вместе.", "Сложности это нормально. Я помогу."],
            "молодец": ["Спасибо.", "Рад, что нравится."],
            "хорошо": ["Продолжаем.", "Двигаемся дальше."],
            "не знаю": ["Сейчас разберемся.", "Это повод узнать новое."],
            "записал": ["Хорошо, продолжаем.", "Отлично, идем дальше."],
            "дальше": ["Переходим к следующей части.", "Продолжаем урок."],
            "стоп": ["Останавливаю урок.", "Прерываю чтение."]
        }

    def _load_lessons(self):
        """Загружает список доступных уроков"""
        self.lessons = {}
        try:
            if not self.lessons_dir.exists():
                self.lessons_dir.mkdir(parents=True)
                return
                
            # Загрузка текстовых файлов уроков
            for lesson_file in self.lessons_dir.glob("*.txt"):
                try:
                    subject = "обществознание"  # По умолчанию
                    if "math" in lesson_file.stem:
                        subject = "математика"
                    elif "history" in lesson_file.stem:
                        subject = "история"
                    
                    if subject not in self.lessons:
                        self.lessons[subject] = []
                    
                    self.lessons[subject].append({
                        'id': lesson_file.stem,
                        'title': lesson_file.stem.replace('_', ' ').title(),
                        'description': f"Текстовый урок по {subject}",
                        'file_path': lesson_file,
                        'type': 'text'
                    })
                except Exception as e:
                    print(f"Ошибка загрузки урока {lesson_file}: {e}")
                    
        except Exception as e:
            print(f"Ошибка доступа к папке уроков: {e}")

    def _load_lesson_content(self, lesson_file: Path) -> List[str]:
        """Загружает содержание урока из текстового файла"""
        try:
            with open(lesson_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Разбиваем на абзацы по точкам, но сохраняем структуру
                paragraphs = re.split(r'(?<=[.!?])\s+', content)
                return [p.strip() for p in paragraphs if p.strip()]
        except Exception as e:
            print(f"Ошибка загрузки содержания урока: {e}")
            return ["Содержание урока временно недоступно."]

    def _similarity(self, a: str, b: str) -> float:
        """Вычисление схожести строк"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def process_input(self, text: str) -> str:
        """Обработка входящего текста и генерация ответа"""
        if self.lesson_started and self.current_state != "lesson_reading":
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
            "greeting": ["Давайте начнем урок. Скажите привет.", "Готовы начать занятие?"],
            "subject_selection": ["Выберите предмет из списка.", "Какой предмет вас интересует?"],
            "lesson_selection": ["Выберите урок из предложенных.", "Какой урок хотите пройти?"],
            "lesson_confirmation": ["Скажите готов чтобы начать.", "Жду вашего подтверждения."],
            "lesson_reading": ["Продолжаем урок.", "Слушайте внимательно."]
        }
        
        return random.choice(fallbacks.get(self.current_state, ["Продолжим наш урок."]))

    def _handle_greeting(self, text: str) -> Optional[str]:
        greeting_words = ["привет", "здравствуй", "начать", "старт", "готов", "поехали"]
        if any(word in text for word in greeting_words):
            self.current_state = "subject_selection"
            subjects = list(self.lessons.keys())
            
            if not subjects:
                return "Уроки еще не загружены. Попробуйте позже."
                
            subject_list = "\n".join(f"{i+1}) {subj.capitalize()}" for i, subj in enumerate(subjects))
            return f"Здравствуйте. Выберите предмет:\n{subject_list}"
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
            return "Хорошо, начнем сначала. Скажите привет чтобы продолжить."
            
        return None

    def _get_lesson_selection_message(self) -> str:
        """Формирует сообщение для выбора урока"""
        lessons = self.lessons[self.current_subject]
        lesson_list = "\n".join(f"{i+1}) {lesson['title']}" for i, lesson in enumerate(lessons))
        
        return f"Выбран предмет: {self.current_subject.capitalize()}.\nВыберите урок:\n{lesson_list}"

    def _handle_lesson_selection(self, text: str) -> Optional[str]:
        if not self.current_subject:
            self.current_state = "subject_selection"
            return "Сначала выберите предмет."
            
        lessons = self.lessons[self.current_subject]
        
        # Поиск по номеру
        for i, lesson in enumerate(lessons):
            if str(i+1) in text:
                self.selected_lesson = lesson
                self.current_state = "lesson_confirmation"
                return self._get_lesson_confirmation_message(lesson)
        
        # Поиск по названию
        for lesson in lessons:
            if lesson['title'].lower() in text.lower():
                self.selected_lesson = lesson
                self.current_state = "lesson_confirmation"
                return self._get_lesson_confirmation_message(lesson)
                
        # Возврат к выбору предмета
        if any(word in text for word in ["назад", "вернуться", "другой предмет"]):
            self.current_state = "subject_selection"
            self.current_subject = None
            return "Хорошо, выберем другой предмет."
            
        return None

    def _get_lesson_confirmation_message(self, lesson: dict) -> str:
        """Формирует сообщение подтверждения выбора урока"""
        return f"Выбран урок: {lesson['title']}.\nСкажите готов чтобы начать чтение урока."

    def _handle_lesson_confirmation(self, text: str) -> Optional[str]:
        ready_words = ["готов", "поехали", "начинаем", "старт", "давай", "начали"]
        if any(word in text for word in ready_words) and self.selected_lesson:
            self.lesson_started = True
            self.current_state = "lesson_reading"
            self.current_paragraph = 0
            
            # Загружаем содержание урока
            self.lesson_content = self._load_lesson_content(self.selected_lesson['file_path'])
            
            # Инициализируем базу знаний для выбранного предмета
            self.knowledge_base = KnowledgeBase(self.current_subject)
            
            if self.lesson_content:
                return "Начинаем чтение урока. Скажите записал когда будете готовы продолжить."
            else:
                return "Ошибка загрузки урока. Попробуйте выбрать другой урок."
                
        # Возврат к выбору урока
        if any(word in text for word in ["назад", "вернуться", "другой урок"]):
            self.current_state = "lesson_selection"
            self.selected_lesson = None
            return "Хорошо, выберем другой урок."
            
        return None

    def _handle_lesson_reading(self, text: str) -> Optional[str]:
        """Обработка во время чтения урока"""
        if "записал" in text.lower() or "дальше" in text.lower():
            return self._get_next_paragraph()
            
        if "стоп" in text.lower() or "останови" in text.lower():
            self.lesson_started = False
            self.current_state = "greeting"
            return "Урок остановлен. Скажите привет чтобы начать заново."
            
        # Если это не команда управления чтением, обрабатываем как вопрос
        return None

    def _get_next_paragraph(self) -> Optional[str]:
        """Возвращает следующий абзац урока"""
        if self.current_paragraph < len(self.lesson_content):
            paragraph = self.lesson_content[self.current_paragraph]
            self.current_paragraph += 1
            return paragraph
        else:
            self.lesson_started = False
            self.current_state = "greeting"
            return "Урок завершен. Скажите привет чтобы начать новый урок."

    def handle_question_during_lesson(self, question: str) -> str:
        """Обработка вопросов во время урока"""
        if not question.strip():
            return "Повторите вопрос пожалуйста."
            
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
        return "Интересный вопрос. Давайте обсудим его после завершения текущего материала."

    def get_selected_lesson(self) -> Optional[dict]:
        """Возвращает данные выбранного урока"""
        return self.selected_lesson

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
        self.lesson_content = []
        self.current_paragraph = 0
        self.knowledge_base = None

    def get_available_subjects(self) -> List[str]:
        """Возвращает список доступных предметов"""
        return list(self.lessons.keys())

    def get_lessons_for_subject(self, subject: str) -> List[dict]:
        """Возвращает уроки для указанного предмета"""
        return self.lessons.get(subject, [])