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
        self.conversation_counter = 0
        self._load_lessons()
        
        # Расширенные локальные шаблоны для более естественного общения
        self.local_patterns = {
            "привет": ["Привет! Рад вас видеть. Как ваше настроение?", "Здравствуйте! Готовы к интересному уроку?"],
            "как дела": ["Все прекрасно! Готов помочь вам с обучением.", "Отлично! А как ваши успехи в учебе?"],
            "спасибо": ["Всегда пожалуйста! Рад был помочь.", "Не стоит благодарности! Это моя работа."],
            "не понимаю": ["Давайте разберем этот момент еще раз вместе.", "Хорошо, объясню по-другому, чтобы было понятнее."],
            "повтори": ["Конечно, повторяю для вас...", "С удовольствием скажу еще раз."],
            "скучно": ["Давайте сделаем урок более интересным! Может, викторину?", "Понимаю. Предлагаю сменить активность!"],
            "трудно": ["Не переживайте! Сложности - это нормально. Я помогу разобраться.", "Вместе мы обязательно справимся!"],
            "молодец": ["Спасибо! Стараюсь для вас.", "Вы тоже молодец, что так активно участвуете!"],
            "хорошо": ["Прекрасно! Продолжаем наш урок.", "Отлично! Двигаемся дальше."],
            "не знаю": ["Это нормально не знать! Сейчас вместе разберемся.", "Отличный повод узнать что-то новое!"],
            "записал": ["Супер! Записали - значит запомнили. Продолжаем!", "Отлично! Идем дальше."],
            "дальше": ["Переходим к следующей интересной части.", "Продолжаем наш увлекательный урок."],
            "стоп": ["Останавливаю урок. Скажите 'привет', когда будете готовы продолжить.", "Прерываю чтение. Жду вашей команды."],
            "кто ты": ["Я ваш виртуальный учитель с искусственным интеллектом! Готов помочь с обучением.", 
                      "AI-учитель, который сделает ваше обучение интересным и эффективным."],
            "что умеешь": ["Я могу проводить уроки, отвечать на вопросы, объяснять сложные темы и делать обучение увлекательным!", 
                          "Умею преподавать разные предметы, отвечать на ваши вопросы и адаптироваться под ваш уровень."],
            "расскажи о себе": ["Я цифровой преподаватель, созданный чтобы сделать образование доступным и интересным для всех!", 
                               "Моя задача - помочь вам учиться с удовольствием и пониманием."]
        }

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
                        f.write("Основы обществознания: подготовка к ЕГЭ.\n\nДобро пожаловать на демо-урок! Сегодня мы разберем фундаментальные понятия обществознания.\n\nОбщество - это сложная динамическая система, объединяющая людей, которые связаны совместной деятельностью, общими интересами и ценностями.\n\nГосударство - это политическая организация общества, обладающая суверенитетом и аппаратом управления.\n\nДемократия - форма правления, при которой народ является источником власти.\n\nЭкономика - хозяйственная деятельность общества, система производства и распределения товаров.\n\nКультура - совокупность достижений человечества в духовной и материальной жизни.\n\nПраво - система общеобязательных норм, охраняемых государством.\n\nСоциализация - процесс усвоения индивидом социальных норм и ценностей.\n\nЛичность - человек как носитель социальных качеств и сознательной деятельности.\n\nМораль - система норм и принципов, регулирующих поведение людей.\n\nГлобализация - процесс всемирной экономической, политической и культурной интеграции.")
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
                # Разбиваем на предложения для более естественного чтения
                sentences = re.split(r'(?<=[.!?])\s+', content)
                # Объединяем предложения в группы по 2-4 для плавного чтения
                paragraphs = []
                current_paragraph = []
                
                for sentence in sentences:
                    if sentence.strip():
                        current_paragraph.append(sentence.strip())
                        if len(current_paragraph) >= 2:  # Группируем по 2-4 предложения
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

    def process_input(self, text: str) -> str:
        """Обработка входящего текста и генерация ответа"""
        if self.lesson_started and self.current_state != "lesson_reading":
            return None
            
        text_lower = text.lower().strip()
        self.conversation_counter += 1
        
        # 1. Если есть база знаний по предмету, сначала проверяем там
        if self.knowledge_base:
            knowledge_response = self.knowledge_base.get_dialogue_response(text_lower)
            if knowledge_response and not knowledge_response.startswith("Интересный вопрос!"):
                return knowledge_response
        
        # 2. Быстрая проверка локальных шаблонов
        for pattern, responses in self.local_patterns.items():
            if pattern in text_lower:
                return random.choice(responses)
        
        # 3. Проверка диалоговых шаблонов из базы знаний (если есть)
        if self.knowledge_base:
            dialogue_response = self.knowledge_base.get_dialogue_response(text_lower)
            if dialogue_response:
                return dialogue_response
        
        # 4. Если пользователь называет предмет - автоматически начинаем урок
        available_subjects = self.get_available_subjects()
        for subject in available_subjects:
            if subject.lower() in text_lower:
                self.current_subject = subject
                # Автоматически выбираем первый доступный урок (демо-урок если есть)
                lessons = self.lessons.get(subject, [])
                demo_lessons = [l for l in lessons if l.get('is_demo', False)]
                
                if demo_lessons:
                    self.selected_lesson = demo_lessons[0]
                elif lessons:
                    self.selected_lesson = lessons[0]
                else:
                    # Создаем временный урок, если нет доступных
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
                
                # Возвращаем None, чтобы сразу начать чтение урока
                return None
        
        # 5. Обработка по текущему состоянию
        handler = self.dialogue_states.get(self.current_state)
        if handler:
            response = handler(text_lower)
            if response:
                return response
        
        # 6. Fallback с учетом состояния и счетчика разговора
        fallbacks = {
            "greeting": [
                "Привет! Давайте познакомимся. Какой предмет вас интересует?",
                "Здравствуйте! Я готов помочь с обучением. О чем хотите узнать?",
                "Рад вас видеть! Давайте выберем интересную тему для урока."
            ],
            "subject_selection": [
                "У меня есть уроки по разным предметам. Что вас интересует?",
                "Могу предложить: обществознание, математика, история. Что выбираете?",
                "Какой предмет хотите изучить? Выбирайте - я найду подходящий урок!"
            ],
            "lesson_reading": [
                "Продолжаем наш увлекательный урок.",
                "Слушайте внимательно, это интересно!",
                "Продолжаем изучение материала."
            ]
        }
        
        # Добавляем вариативность в ответы
        fallback_responses = fallbacks.get(self.current_state, ["Продолжим наш урок."])
        response = random.choice(fallback_responses)
        
        # После 3-х реплик без прогресса - мягко направляем к выбору предмета
        if self.conversation_counter >= 3 and self.current_state == "greeting":
            response += " Кстати, какой предмет вас интересует?"
            
        return response

    def _handle_greeting(self, text: str) -> Optional[str]:
        greeting_words = ["привет", "здравствуй", "начать", "старт", "готов", "поехали", "давай", "началом"]
        if any(word in text for word in greeting_words):
            self.current_state = "subject_selection"
            subjects = self.get_available_subjects()
            
            if not subjects:
                return "К сожалению, уроки еще не загружены. Попробуйте позже."
                
            subject_list = ", ".join([subj.capitalize() for subj in subjects])
            return f"Отлично! Давайте выберем предмет для урока. У меня есть: {subject_list}. Что вас интересует? Можете просто сказать название предмета!"
        return None

    def _handle_subject_selection(self, text: str) -> Optional[str]:
        subjects = self.get_available_subjects()
        
        # Поиск по названию предмета
        for subject in subjects:
            if subject.lower() in text.lower():
                self.current_subject = subject
                # Автоматически начинаем урок при выборе предмета
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
                
                # Возвращаем None, чтобы сразу начать чтение урока
                return None
                
        # Возврат к приветствию
        if any(word in text for word in ["назад", "вернуться", "сначала"]):
            self.current_state = "greeting"
            return "Хорошо, начнем сначала. Скажите привет чтобы продолжить."
            
        # Если пользователь просто говорит "да" или соглашается
        if any(word in text for word in ["да", "ага", "угу", "ладно", "хорошо"]):
            return "Отлично! Какой предмет вас заинтересовал? Назовите его пожалуйста."
            
        return None

    def _handle_lesson_reading(self, text: str) -> Optional[str]:
        """Обработка во время чтения урока"""
        if "записал" in text.lower() or "дальше" in text.lower() or "продолжай" in text.lower():
            # Возвращаем None, так как чтение будет обработано в app.py
            return None
            
        if "стоп" in text.lower() or "останови" in text.lower() or "хватит" in text.lower():
            self.lesson_started = False
            self.current_state = "greeting"
            self.conversation_counter = 0
            self.knowledge_base = None
            return "Урок остановлен. Скажите 'привет' когда захотите продолжить или выбрать новый урок."
            
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
            self.conversation_counter = 0
            self.knowledge_base = None
            return "Урок завершен! Было очень интересно. Скажите 'привет' чтобы начать новый увлекательный урок."

    def handle_question_during_lesson(self, question: str) -> str:
        """Обработка вопросов во время урока"""
        if not question.strip():
            return "Повторите вопрос пожалуйста, я не расслышал."
            
        question_lower = question.lower().strip()
        
        # 1. Сначала проверяем базу знаний по предмету
        if self.knowledge_base:
            knowledge_response = self.knowledge_base.get_dialogue_response(question_lower)
            if knowledge_response and not knowledge_response.startswith("Интересный вопрос!"):
                return knowledge_response
        
        # 2. Быстрая проверка локальных шаблонов
        for pattern, responses in self.local_patterns.items():
            if pattern in question_lower:
                return random.choice(responses)
        
        # 3. Проверка диалоговых шаблонов из базы знаний
        if self.knowledge_base:
            dialogue_response = self.knowledge_base.get_dialogue_response(question_lower)
            if dialogue_response:
                return dialogue_response
        
        # 4. Поиск в предметной базе знаний
        if self.knowledge_base:
            answer = self.knowledge_base.find_answer(question)
            if answer and not answer.startswith("Интересный вопрос!"):
                return answer
        
        # 5. Запрос к LLM
        llm_response = self.llm.query(question, self.current_subject)
        if llm_response:
            # Сохраняем в кэш и базу знаний
            self.llm.add_to_cache(question, llm_response, self.current_subject)
            if self.knowledge_base:
                self.knowledge_base.add_knowledge(question=question, answer=llm_response)
            return llm_response
        
        # 6. Финальный fallback
        return "Интересный вопрос! Давайте обсудим его после завершения текущего материала, чтобы не отвлекаться."

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
        # Всегда добавляем обществознание, даже если нет уроков
        if "обществознание" not in subjects:
            subjects.append("обществознание")
        return subjects

    def get_lessons_for_subject(self, subject: str) -> List[dict]:
        """Возвращает уроки для указанного предмета"""
        return self.lessons.get(subject, [])