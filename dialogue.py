import json
from pathlib import Path
from typing import Dict, Optional, List
from difflib import SequenceMatcher
import random
import re
from knowledge.knowledge_base import KnowledgeBase
from llm import LLMIntegration
from config import get_llm_mode, get_initialization_mode

class DialogueManager:
    def __init__(self, socketio):
        self.socketio = socketio
        self.dialogue_states = {
            "greeting": self._handle_greeting,
            "subject_selection": self._handle_subject_selection,
            "lesson_reading": self._handle_lesson_reading,
            "demo_subject_selection": self._handle_demo_subject_selection,
            "topic_processing": self._handle_topic_processing
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
        self.conversation_counter = 0
        self.llm_query_mode = get_llm_mode()  # Загружаем режим из конфига
        self.initialization_mode = get_initialization_mode()  # Режим инициализации
        self.requested_topic = None
        self._load_lessons()
        self.demo_lessons_available = self._load_demo_lessons()
        
        # Расширенные локальные шаблоны для более естественного общения
        self.local_patterns = {
            "привет": ["Привет! Рад вас видеть.", "Здравствуйте! Готовы к уроку?"],
            "как дела": ["Все прекрасно! Готов помочь.", "Отлично! А у вас?"],
            "спасибо": ["Всегда пожалуйста!", "Не стоит благодарности!"],
            "не понимаю": ["Давайте разберем еще раз.", "Объясню по-другому."],
            "повтори": ["Конечно, повторяю.", "Скажу еще раз."],
            "скучно": ["Давайте сделаем интереснее!", "Предлагаю сменить активность!"],
            "трудно": ["Не переживайте! Я помогу.", "Вместе справимся!"],
            "молодец": ["Спасибо! Вы тоже молодец!"],
            "хорошо": ["Прекрасно! Продолжаем.", "Отлично! Двигаемся дальше."],
            "не знаю": ["Это нормально! Сейчас разберемся.", "Отличный повод узнать!"],
            "стоп": ["Останавливаю урок.", "Прерываю чтение."],
            "кто ты": ["Я ваш AI-учитель! Готов помочь."],
            "что умеешь": ["Могу проводить уроки и отвечать на вопросы!"],
            "расскажи о себе": ["Я цифровой преподаватель для вашего обучения!"]
        }

    def _load_demo_lessons(self):
        """Загрузка демо-уроков из папки lessons/demo"""
        demo_lessons = {}
        demo_dir = self.lessons_dir / "demo"
        
        if not demo_dir.exists():
            demo_dir.mkdir(parents=True)
            self._create_default_demo_lessons(demo_dir)
        
        try:
            for lesson_file in demo_dir.glob("*.txt"):
                subject = self._detect_subject(lesson_file.stem)
                
                if subject not in demo_lessons:
                    demo_lessons[subject] = []
                
                demo_lessons[subject].append({
                    'id': f"demo/{lesson_file.stem}",
                    'title': lesson_file.stem.replace('_', ' ').title(),
                    'description': f"Демо-урок по {subject}",
                    'file_path': lesson_file,
                    'type': 'text',
                    'is_demo': True
                })
        except Exception as e:
            print(f"Ошибка загрузки демо-уроков: {e}")
        
        return demo_lessons

    def _create_default_demo_lessons(self, demo_dir):
        """Создание демо-уроков по умолчанию"""
        demo_topics = {
            "математика": "Введение в алгебру: основы уравнений.",
            "история": "Древний Рим: от республики к империи.",
            "физика": "Законы Ньютона: основы механики.",
            "биология": "Клеточное строение организмов."
        }
        
        for subject, content in demo_topics.items():
            filename = f"demo_{subject.lower()}.txt"
            with open(demo_dir / filename, 'w', encoding='utf-8') as f:
                f.write(f"{content}\n\nЭто демо-урок по предмету '{subject}'.")

    def _load_lessons(self):
        """Загружает список доступных уроков"""
        self.lessons = {}
        try:
            if not self.lessons_dir.exists():
                self.lessons_dir.mkdir(parents=True)
                # Создаем демо-урок по обществознанию, если его нет
                demo_lesson = self.lessons_dir / "social_general.txt"
                if not demo_lesson.exists():
                    with open(demo_lesson, 'w', encoding='utf-8') as f:
                        f.write("Основы обществознания: подготовка к ЕГЭ.\n\nДобро пожаловать на демо-урок! Сегодня мы разберем фундаментальные понятия обществознания.\n\nОбщество - это сложная динамическая система, объединяющая людей, которые связаны совместной деятельностью, общими интересами и ценностями.\n\nГосударство - это политическая организация общества, обладающая суверенитетом и аппаратом управления.\n\nДемократия - это форма правления, при которой народ является источником власти.\n\nЭкономика - это хозяйственная деятельность общества, система производства и распределения товаров.\n\nКультура - это совокупность достижений человечества в духовной и материальной жизни.\n\nПраво - это система общеобязательных норм, охраняемых государством.\n\nСоциализация - это процесс усвоения индивидом социальных норм и ценностей.\n\nЛичность - это человек как носитель социальных качеств и сознательной деятельности.\n\nМораль - это система норм и принципов, регулирующих поведение людей.\n\nГлобализация - это процесс всемирной экономической, политической и культурной интеграции.")
                return
                
            # Загрузка текстовых файлов уроков
            for lesson_file in self.lessons_dir.glob("*.txt"):
                try:
                    subject = self._detect_subject(lesson_file.stem)
                    
                    if subject not in self.lessons:
                        self.lessons[subject] = []
                    
                    self.lessons[subject].append({
                        'id': lesson_file.stem,
                        'title': lesson_file.stem.replace('_', ' ').title(),
                        'description': f"Интересный урок по {subject}",
                        'file_path': lesson_file,
                        'type': 'text',
                        'is_demo': 'demo' in lesson_file.stem.lower() or 'general' in lesson_file.stem.lower()
                    })
                except Exception as e:
                    print(f"Ошибка загрузки урока {lesson_file}: {e}")
                    
        except Exception as e:
            print(f"Ошибка доступа к папке уроков: {e}")

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

    def _load_lesson_content(self, lesson_file: Path) -> List[str]:
        """Загружает содержание урока из текстового файла"""
        try:
            with open(lesson_file, 'r', encoding='utf-8') as f:
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
                
                return paragraphs if paragraphs else ["Содержание урока временно недоступно. Давайте поговорим на эту тему!"]
        except Exception as e:
            print(f"Ошибка загрузки содержания урока: {e}")
            return ["Содержание урока временно недоступно. Давайте поговорим на эту тему!"]

    def _similarity(self, a: str, b: str) -> float:
        """Вычисление схожести строк"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def process_input(self, text: str) -> Optional[str]:
        """Обработка входящего текста и генерация ответа"""
        text_lower = text.lower().strip()
        
        # Обработка в зависимости от режима инициализации
        if self.initialization_mode == "demo":
            return self._process_demo_mode(text_lower)
        else:
            return self._process_normal_mode(text_lower)

    def _process_normal_mode(self, text_lower: str) -> Optional[str]:
        """Обработка в нормальном режиме"""
        if self.lesson_started:
            if any(word in text_lower for word in ["стоп", "останови", "хватит", "закончи"]):
                return self._handle_lesson_reading(text_lower)
            return None
            
        self.conversation_counter += 1
        
        # 1. Если пользователь называет предмет - автоматически начинаем урок
        available_subjects = self.get_available_subjects()
        for subject in available_subjects:
            if subject.lower() in text_lower:
                return self._handle_subject_selection_direct(subject)
        
        # 2. Если есть база знаний по предмету, проверяем там
        if self.knowledge_base:
            knowledge_response = self.knowledge_base.get_dialogue_response(text_lower)
            if knowledge_response and not knowledge_response.startswith("Интересный вопрос!"):
                return self._limit_response_length(knowledge_response)
        
        # 3. Быстрая проверка локальных шаблонов
        for pattern, responses in self.local_patterns.items():
            if pattern in text_lower:
                return random.choice(responses)
        
        # 4. Проверка диалоговых шаблонов из базы знаний
        if self.knowledge_base:
            dialogue_response = self.knowledge_base.get_dialogue_response(text_lower)
            if dialogue_response:
                return self._limit_response_length(dialogue_response)
        
        # 5. Обработка по текущему состоянию
        handler = self.dialogue_states.get(self.current_state)
        if handler:
            response = handler(text_lower)
            if response:
                return self._limit_response_length(response)
        
        # 6. Fallback с учетом состояния и счетчика разговора
        fallbacks = {
            "greeting": [
                "Привет! Давайте познакомимся. Какой предмет вас интересует?",
                "Здравствуйте! Я готов помочь с обучением. О чем хотите узнать?",
                "Рад вас видеть! Давайте выберем интересную тему для урока."
            ],
            "subject_selection": [
                "У меня есть уроки по разным предметов. Что вас интересует?",
                "Могу предложить: обществознание, математика, история. Что выбираете?",
                "Какой предмет хотите изучить? Выбирайте - я найду подходящий урок!"
            ],
            "lesson_reading": [
                "Продолжаем наш увлекательный урок.",
                "Слушайте внимательно, это интересно!",
                "Продолжаем изучение материала."
            ]
        }
        
        fallback_responses = fallbacks.get(self.current_state, ["Продолжим наш урок."])
        response = random.choice(fallback_responses)
        
        # После 3-х реплик без прогресса - мягко направляем к выбору предмета
        if self.conversation_counter >= 3 and self.current_state == "greeting":
            response += " Кстати, какой предмет вас интересует?"
            
        return response

    def _process_demo_mode(self, text_lower: str) -> Optional[str]:
        """Обработка в демо-режиме"""
        if self.lesson_started:
            return None
            
        self.conversation_counter += 1
        
        # 1. Приветствие и предложение выбора
        if self.current_state == "greeting":
            if any(word in text_lower for word in ["привет", "здравствуй", "начать", "старт", "готов"]):
                self.current_state = "demo_subject_selection"
                return self._get_demo_greeting_response()
        
        # 2. Выбор предмета или темы
        elif self.current_state == "demo_subject_selection":
            # Проверяем, выбрал ли пользователь существующий предмет
            for subject in self.demo_lessons_available.keys():
                if subject.lower() in text_lower:
                    return self._handle_subject_selection_direct(subject)
            
            # Если пользователь называет тему, а не предмет
            if self._is_topic_request(text_lower):
                self.requested_topic = text_lower
                self.current_state = "topic_processing"
                return self._handle_topic_request(text_lower)
            
            # Если это не выбор предмета/темы, предлагаем варианты
            return self._get_demo_options_response()
        
        return None

    def _limit_response_length(self, response: str, max_sentences: int = 2) -> str:
        """Ограничивает длину ответа до 1-3 предложений для быстродействия"""
        if not response:
            return response
            
        # Разбиваем на предложения
        sentences = re.split(r'(?<=[.!?])\s+', response)
        
        # Берем только первые max_sentences предложений
        if len(sentences) > max_sentences:
            return ' '.join(sentences[:max_sentences]).strip()
        
        return response

    def _get_demo_greeting_response(self) -> str:
        """Приветственное сообщение для демо-режима"""
        subjects = list(self.demo_lessons_available.keys())
        subject_list = ", ".join([subj.capitalize() for subj in subjects])
        
        return self._limit_response_length(
            f"Привет! Это демо-режим обучения. "
            f"У меня есть демо-уроки по: {subject_list}. "
            f"Также ты можешь предложить свою тему! "
            f"Что тебя интересует?"
        )

    def _get_demo_options_response(self) -> str:
        """Сообщение с вариантами выбора для демо-режима"""
        subjects = list(self.demo_lessons_available.keys())
        subject_list = ", ".join([subj.capitalize() for subj in subjects])
        
        return self._limit_response_length(
            f"На выбор есть: {subject_list}. "
            f"Или назови тему, которую хочешь изучить!"
        )

    def _is_topic_request(self, text: str) -> bool:
        """Определяет, является ли запрос темой для изучения"""
        # Исключаем названия предметов
        subjects = list(self.demo_lessons_available.keys())
        for subject in subjects:
            if subject.lower() in text.lower():
                return False
        
        # Запросы, которые скорее всего являются темами
        topic_indicators = [
            "расскажи о", "что такое", "объясни", "урок про", 
            "тема", "изучим", "хочу узнать о", "про "
        ]
        
        return any(indicator in text.lower() for indicator in topic_indicators)

    def _handle_topic_request(self, topic: str) -> str:
        """Обработка запроса темы"""
        return self._limit_response_length(
            f"Отличная тема! '{topic}' - это интересно. "
            f"Сейчас подготовлю для тебя специальный урок."
        )

    def generate_topic_lesson(self, topic: str) -> Optional[str]:
        """Генерация урока по теме через LLM"""
        try:
            # Используем LLM для генерации краткого урока
            prompt = f"""Создай краткий образовательный урок на тему "{topic}". 
Максимум 3-4 предложения. Формат: понятное объяснение для ученика.
Пиши на русском языке. Будь лаконичным."""

            lesson_content = self.llm.query(prompt, "", "общее")
            
            if lesson_content:
                # Сохраняем сгенерированный урок
                lesson_id = f"demo/generated_{hash(topic) % 1000}"
                demo_dir = self.lessons_dir / "demo"
                lesson_file = demo_dir / f"generated_{hash(topic) % 1000}.txt"
                
                with open(lesson_file, 'w', encoding='utf-8') as f:
                    f.write(lesson_content)
                
                # Загружаем урок в систему
                subject = self._detect_subject(topic) or "общее"
                self.selected_lesson = {
                    'id': lesson_id,
                    'title': f"Урок по теме: {topic}",
                    'file_path': lesson_file,
                    'is_demo': True
                }
                
                self.lesson_started = True
                self.current_state = "lesson_reading"
                self.current_paragraph = 0
                self.lesson_content = self._load_lesson_content(lesson_file)
                self.knowledge_base = KnowledgeBase(subject)
                
                return None
            else:
                return "Извините, не удалось создать урок. Попробуйте другой предмет."
                
        except Exception as e:
            print(f"Ошибка генерации урока: {e}")
            return "Произошла ошибка. Попробуйте еще раз."

    def _handle_subject_selection_direct(self, subject: str) -> Optional[str]:
        """Прямая обработка выбора предмета"""
        self.current_subject = subject
        
        # Выбираем урок в зависимости от режима
        if self.initialization_mode == "demo":
            lessons = self.demo_lessons_available.get(subject, [])
        else:
            lessons = self.lessons.get(subject, [])
            demo_lessons = [l for l in lessons if l.get('is_demo', False)]
            
            if demo_lessons:
                self.selected_lesson = demo_lessons[0]
            elif lessons:
                self.selected_lesson = lessons[0]
            else:
                # Создаем временный урок
                self.selected_lesson = {
                    'id': f"demo_{subject}",
                    'title': f"Демо-урок по {subject}",
                    'file_path': self.lessons_dir / f"demo_{subject}.txt",
                    'is_demo': True
                }
        
        self.lesson_started = True
        self.current_state = "lesson_reading"
        self.current_paragraph = 0
        self.lesson_content = self._load_lesson_content(self.selected_lesson['file_path'])
        self.knowledge_base = KnowledgeBase(self.current_subject)
        
        return None

    def _handle_greeting(self, text: str) -> Optional[str]:
        greeting_words = ["привет", "здравствуй", 'начать', "старт", " готов", "поехали", "давай", "началом"]
        if any(word in text for word in greeting_words):
            self.current_state = "subject_selection"
            subjects = self.get_available_subjects()
            
            if not subjects:
                return "К сожалению, уроки еще не загружены. Попробуйте позже."
                
            subject_list = ", ".join([subj.capitalize() for subj in subjects])
            return f"Отлично! Давайте выберем предмет для урока. У меня есть: {subject_list}. Что вас интересует?"
        return None

    def _handle_subject_selection(self, text: str) -> Optional[str]:
        subjects = self.get_available_subjects()
        
        # Поиск по названию предмета
        for subject in subjects:
            if subject.lower() in text.lower():
                return self._handle_subject_selection_direct(subject)
                
        # Возврат к приветствию
        if any(word in text for word in ["назад", "вернуться", "сначала"]):
            self.current_state = "greeting"
            return "Хорошо, начнем сначала. Скажите привет чтобы продолжить."
            
        # Если пользователь просто говорит "да" или соглашается
        if any(word in text for word in ["да", "ага", 'угу', "ладно", "хорошо"]):
            return "Отлично! Какой предмет вас заинтересовал?"
            
        return None

    def _handle_lesson_reading(self, text: str) -> Optional[str]:
        """Обработка во время чтения урока"""
        if any(word in text for word in ["стоп", "останови", "хватит", "закончи"]):
            self.lesson_started = False
            self.current_state = "greeting"
            self.conversation_counter = 0
            self.knowledge_base = None
            return "Урок остановлен. Скажите 'привет' когда захотите продолжить."
            
        return None

    def _handle_demo_subject_selection(self, text: str) -> Optional[str]:
        """Обработка выбора в демо-режиме"""
        # Проверяем выбор существующего предмета
        for subject in self.demo_lessons_available.keys():
            if subject.lower() in text.lower():
                return self._handle_subject_selection_direct(subject)
        
        # Проверяем запрос темы
        if self._is_topic_request(text):
            self.requested_topic = text
            self.current_state = "topic_processing"
            return self._handle_topic_request(text)
        
        return self._get_demo_options_response()

    def _handle_topic_processing(self, text: str) -> Optional[str]:
        """Обработка состояния генерации темы"""
        # В этом состоянии просто ждем завершения генерации
        return "Создаю урок для вас. Один момент..."

    def _get_next_paragraph(self) -> Optional[str]:
        """Возвращает следующий абзац урока"""
        if self.current_paragraph < len(self.lesson_content):
            paragraph = self.lesson_content[self.current_paragraph]
            self.current_paragraph += 1
            return paragraph
        else:
            self.lesson_started = False
            self.current_state = "greeting"
            self.conversation_counter = 0
            self.knowledge_base = None
            return "Урок завершен! Скажите 'привет' чтобы начать новый урок."

    def handle_question_during_lesson(self, question: str) -> str:
        """Обработка вопросов во время урока с учетом выбранного режима"""
        if not question.strip():
            return "Повторите вопрос пожалуйста."
            
        question_lower = question.lower().strip()
        
        # Ограничиваем ответ для быстродействия
        def limited_response(response):
            return self._limit_response_length(response, max_sentences=2)
        
        # Режим "LLM в первую очередь"
        if self.llm_query_mode == "llm_first":
            # 1. Сначала пробуем запрос к LLM
            current_context = ""
            if self.lesson_content and self.current_paragraph > 0:
                context_start = max(0, self.current_paragraph - 2)
                current_context = " ".join(self.lesson_content[context_start:self.current_paragraph])
            
            llm_response = self.llm.query(question, current_context, self.current_subject)
            if llm_response and not llm_response.startswith("Интересный вопрос!"):
                self.llm.add_to_cache(question, llm_response, self.current_subject)
                if self.knowledge_base:
                    self.knowledge_base.add_llm_answer(question, llm_response)
                    self.knowledge_base.add_knowledge(question=question, answer=llm_response)
                return limited_response(llm_response)
            
            # 2. Если LLM не дал ответ, проверяем базу знаний
            if self.knowledge_base:
                knowledge_response = self.knowledge_base.get_dialogue_response(question_lower)
                if knowledge_response and not knowledge_response.startswith("Интересный вопрос!"):
                    return limited_response(knowledge_response)
            
            # 3. Проверяем базу ответов LLM
            if self.knowledge_base:
                llm_answer = self.knowledge_base.find_llm_answer(question, threshold=0.8)
                if llm_answer:
                    return limited_response(llm_answer)
            
            # 4. Финальный fallback
            return "Не удалось найти ответ. Давайте продолжим урок."
        
        # Традиционный режим
        else:
            # 1. Сначала проверяем базу знаний по предмету
            if self.knowledge_base:
                knowledge_response = self.knowledge_base.get_dialogue_response(question_lower)
                if knowledge_response and not knowledge_response.startswith("Интересный вопрос!"):
                    return limited_response(knowledge_response)
            
            # 2. Быстрая проверка локальных шаблонов
            for pattern, responses in self.local_patterns.items():
                if pattern in question_lower:
                    return random.choice(responses)
            
            # 3. Проверка диалоговых шаблонов из базы знаний
            if self.knowledge_base:
                dialogue_response = self.knowledge_base.get_dialogue_response(question_lower)
                if dialogue_response:
                    return limited_response(dialogue_response)
            
            # 4. Поиск в предметной базе знаний
            if self.knowledge_base:
                answer = self.knowledge_base.find_answer(question, threshold=0.5)
                if answer and not answer.startswith("Интересный вопрос!"):
                    return limited_response(answer)
            
            # 5. Проверяем базу ответов LLM
            if self.knowledge_base:
                llm_answer = self.knowledge_base.find_llm_answer(question, threshold=0.8)
                if llm_answer:
                    return limited_response(llm_answer)
            
            # 6. Запрос к LLM с контекстом
            current_context = ""
            if self.lesson_content and self.current_paragraph > 0:
                context_start = max(0, self.current_paragraph - 2)
                current_context = " ".join(self.lesson_content[context_start:self.current_paragraph])
            
            llm_response = self.llm.query(question, current_context, self.current_subject)
            if llm_response:
                self.llm.add_to_cache(question, llm_response, self.current_subject)
                if self.knowledge_base:
                    self.knowledge_base.add_llm_answer(question, llm_response)
                    self.knowledge_base.add_knowledge(question=question, answer=llm_response)
                return limited_response(llm_response)
            
            # 7. Финальный fallback
            return "Интересный вопрос! Обсудим после урока."

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
        self.conversation_counter = 0

    def get_available_subjects(self) -> List[str]:
        """Возвращает список доступных предметов"""
        subjects = list(self.lessons.keys())
        if "обществознание" not in subjects:
            subjects.append("обществознание")
        return subjects

    def get_lessons_for_subject(self, subject: str) -> List[dict]:
        """Возвращает уроки для указанного предмета"""
        return self.lessons.get(subject, [])

    def set_llm_model(self, model: str):
        """Установка модели LLM"""
        self.llm.set_model(model)
        print(f"Установлена модель LLM: {model}")

    def set_llm_mode(self, mode: str):
        """Установка режима запросов к LLM"""
        if mode in ["traditional", "llm_first"]:
            self.llm_query_mode = mode
            print(f"Установлен режим LLM: {mode}")

    def set_initialization_mode(self, mode: str):
        """Установка режима инициализации"""
        if mode in
