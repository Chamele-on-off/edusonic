import json
from pathlib import Path
from typing import Dict, Optional, List
from difflib import SequenceMatcher
import random
import re
from knowledge.knowledge_base import KnowledgeBase
from llm import LLMIntegration
from config import get_llm_mode, get_system_mode
import os

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
        self.llm_query_mode = get_llm_mode()  # Загружаем режим из конфига
        self.system_mode = get_system_mode()  # Загружаем режим системы из конфига
        self.conversation_history = []  # Для сохранения контекста в демо-режиме
        self.demo_lessons_dir = self.lessons_dir / "demo"  # Папка для демо-уроков
        self._load_lessons()
        self._ensure_demo_lessons_dir()
        
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
            "стоп": ["Останавливаю урок. Скажите 'привет', когда будете готовы продолжить.", "Прерываю чтение. Жду вашей команды."],
            "кто ты": ["Я ваш виртуальный учитель с искусственным интеллектом! Готов помочь с обучением.", 
                      "AI-учитель, который сделает ваше обучение интересным и эффективным."],
            "что умеешь": ["Я могу проводить уроки, отвечать на вопросы, объяснять сложные темы и делать обучение увлекательным!", 
                          "Умею преподавать разные предметы, отвечать на ваши вопросы и адаптироваться под ваш уровень."],
            "расскажи о себе": ["Я цифровой преподаватель, созданный чтобы сделать образование доступным и интересным для всех!", 
                               "Моя задача - помочь вам учиться с удовольствием и пониманием."]
        }

    def _ensure_demo_lessons_dir(self):
        """Создает папку demo если ее нет"""
        os.makedirs(self.demo_lessons_dir, exist_ok=True)

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
        
        # Добавляем реплику пользователя в историю (для демо-режима)
        if self.system_mode == "demo":
            self.conversation_history.append({"role": "user", "content": text})
            # Ограничиваем длину истории, чтобы не перегружать LLM
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
        
        # 1. Если пользователь называет предмет - автоматически начинаем урок
        # Логика отличается в зависимости от режима
        available_subjects = self.get_available_subjects()
        subject_detected = None
        for subject in available_subjects:
            if subject.lower() in text_lower:
                subject_detected = subject
                break
                
        if subject_detected:
            if self.system_mode == "normal":
                return self._handle_subject_selection_direct(subject_detected)
            else:
                # В демо-режиме ищем урок в демо-папке
                return self._handle_demo_subject_selection(subject_detected, text)
        
        # 2. Если это демо-режим и пользователь называет тему, а не предмет
        if self.system_mode == "demo" and not self.lesson_started:
            # Эвристика: если фраза короткая и не похожа на вопрос, возможно, это тема
            if len(text_lower.split()) < 5 and not any(word in text_lower for word in ['что', 'как', 'почему', 'зачем', '?']):
                # Проверяем через LLM, является ли это темой для урока
                return self._handle_potential_topic_selection(text)
        
        # 3. Если урок уже начат, обрабатываем как вопрос/команду
        if self.lesson_started:
            if any(word in text_lower for word in ["стоп", "останови", "хватит", "закончи"]):
                return self._handle_lesson_reading(text_lower)
            return None
            
        self.conversation_counter += 1
        
        # 4. Если есть база знаний по предмету, проверяем там
        if self.knowledge_base:
            knowledge_response = self.knowledge_base.get_dialogue_response(text_lower)
            if knowledge_response and not knowledge_response.startswith("Интересный вопрос!"):
                return knowledge_response
        
        # 5. Быстрая проверка локальных шаблонов
        for pattern, responses in self.local_patterns.items():
            if pattern in text_lower:
                return random.choice(responses)
        
        # 6. Проверка диалоговых шаблонов из базы знаний
        if self.knowledge_base:
            dialogue_response = self.knowledge_base.get_dialogue_response(text_lower)
            if dialogue_response:
                return dialogue_response
        
        # 7. Обработка по текущему состоянию
        handler = self.dialogue_states.get(self.current_state)
        if handler:
            response = handler(text_lower)
            if response:
                return response
        
        # 8. Fallback с учетом состояния и счетчика разговора
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
        
        # Добавляем вариативность в ответы
        fallback_responses = fallbacks.get(self.current_state, ["Продолжим наш урок."])
        response = random.choice(fallback_responses)
        
        # После 3-х реплик без прогресса - мягко направляем к выбору предмета
        if self.conversation_counter >= 3 and self.current_state == "greeting":
            if self.system_mode == "demo":
                demo_lessons = self._get_demo_lessons()
                subject_list = ", ".join([subj.capitalize() for subj in demo_lessons.keys()])
                response = f"Давайте выберем, что хочешь изучить! На выбор есть такие предметы: {subject_list}. Также можешь сказать свою тему, которая тебя интересует."
            else:
                response += " Кстати, какой предмет вас интересует?"
            
        return response

    def _handle_demo_subject_selection(self, subject: str, user_input: str) -> Optional[str]:
        """Обработка выбора предмета в демо-режиме"""
        self.current_subject = subject
        lessons = self._get_demo_lessons().get(subject, [])
        
        if lessons:
            # Выбираем первый демо-урок по предмету
            self.selected_lesson = lessons[0]
            self._start_lesson()
            return None
        else:
            # В демо-режиме нет урока по этому предмету
            # Можно предложить создать его или выбрать другой
            return f"В демо-режиме пока нет урока по предмету '{subject}'. Можешь выбрать другой предмет или назвать конкретную тему."

    def _handle_potential_topic_selection(self, topic: str) -> Optional[str]:
        """Обработка потенциальной темы через LLM в демо-режиме"""
        # Используем LLM, чтобы проверить, является ли ввод темой для урока
        prompt = f"""
        Пользователь сказал: "{topic}". 
        Может ли это быть темой для образовательного урока? Ответь только "ДА" или "НЕТ".
        Учти, что тема должна быть достаточно общей для урока (например, "квадратные уравнения", "Вторая мировая война", "фотосинтез").
        """
        
        # Делаем запрос к LLM БЕЗ контекста истории, т.к. это изолированный вопрос
        llm_response = self.llm.query(prompt, "", "общее")
        
        if llm_response and "ДА" in llm_response.upper():
            # Генерируем урок по теме
            return self._generate_lesson_from_topic(topic)
        else:
            # Если LLM решила, что это не тема, продолжаем диалог как обычно
            return None

    def _generate_lesson_from_topic(self, topic: str) -> str:
        """Генерация урока по теме с помощью LLM"""
        # Создаем промпт для генерации урока
        lesson_prompt = f"""
        Сгенерируй содержание образовательного урока на тему: "{topic}".
        Формат: текст, разбитый на абзацы (разделяй абзацы двумя переносами строки).
        Будь информативным и понятным. Первая строка - заголовок урока.
        """
        
        # Добавляем в промпт последние реплики для контекста
        context = "Контекст разговора:\n"
        for msg in self.conversation_history[-3:]:  # Берем последние 3 реплики
            role = "Ученик" if msg["role"] == "user" else "Учитель"
            context += f"{role}: {msg['content']}\n"
        
        full_prompt = context + "\n" + lesson_prompt
        
        # Делаем запрос к LLM
        llm_response = self.llm.query(full_prompt, "", "общее")
        
        if llm_response:
            # Сохраняем сгенерированный урок в демо-папку
            filename = f"demo_{topic.lower().replace(' ', '_')}.txt"
            lesson_path = self.demo_lessons_dir / filename
            
            try:
                with open(lesson_path, 'w', encoding='utf-8') as f:
                    f.write(llm_response)
                
                # Создаем запись об уроке и начинаем его
                self.selected_lesson = {
                    'id': f"demo_{topic}",
                    'title': topic,
                    'file_path': lesson_path,
                    'is_demo': True,
                    'is_generated': True  # Новый флаг
                }
                
                self._start_lesson()
                return None # Начинаем урок без дополнительного сообщения
                
            except Exception as e:
                print(f"Ошибка сохранения урока: {e}")
                return "Извините, не удалось создать урок по этой теме. Попробуйте другую тему или предмет."
        else:
            return "Извините, не удалось создать урок по этой теме. Попробуйте другую тему или предмет."

    def _start_lesson(self):
        """Общая функция для начала урока"""
        self.lesson_started = True
        self.current_state = "lesson_reading"
        self.current_paragraph = 0
        self.lesson_content = self._load_lesson_content(self.selected_lesson['file_path'])
        self.knowledge_base = KnowledgeBase(self.current_subject)

    def _get_demo_lessons(self) -> dict:
        """Получает только демо-уроки из папки demo"""
        demo_lessons = {}
        try:
            for lesson_file in self.demo_lessons_dir.glob("*.txt"):
                subject = self._detect_subject(lesson_file.stem)
                
                if subject not in demo_lessons:
                    demo_lessons[subject] = []
                
                demo_lessons[subject].append({
                    'id': lesson_file.stem,
                    'title': lesson_file.stem.replace('_', ' ').title(),
                    'description': f"Демо-урок по {subject}",
                    'file_path': lesson_file,
                    'type': 'text',
                    'is_demo': True
                })
        except Exception as e:
            print(f"Ошибка загрузки демо-уроков: {e}")
        
        return demo_lessons

    def _handle_subject_selection_direct(self, subject: str) -> Optional[str]:
        """Прямая обработка выбора предмета (без поиска в базе знаний)"""
        self.current_subject = subject
        # Автоматически выбираем первый доступный урок (демо-урок если есть)
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
        
        self._start_lesson()
        # Возвращаем None, чтобы сразу начать чтение урока без подтверждения
        return None

    def _handle_greeting(self, text: str) -> Optional[str]:
        greeting_words = ["привет", "здравствуй", 'начать', "старт", " готов", "поехали", "давай", "началом"]
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
                if self.system_mode == "normal":
                    return self._handle_subject_selection_direct(subject)
                else:
                    return self._handle_demo_subject_selection(subject, text)
                
        # Возврат к приветствию
        if any(word in text for word in ["назад", "вернуться", "сначала"]):
            self.current_state = "greeting"
            return "Хорошо, начнем сначала. Скажите привет чтобы продолжить."
            
        # Если пользователь просто говорит "да" или соглашается
        if any(word in text for word in ["да", "ага", 'угу', "ладно", "хорошо"]):
            return "Отлично! Какой предмет вас заинтересовал? Назовите его пожалуйста."
            
        return None

    def _handle_lesson_reading(self, text: str) -> Optional[str]:
        """Обработка во время чтения урока"""
        if any(word in text for word in ["стоп", "останови", "хватит", "закончи"]):
            self.lesson_started = False
            self.current_state = "greeting"
            self.conversation_counter = 0
            self.knowledge_base = None
            if self.system_mode == "demo":
                self.conversation_history = []
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
            if self.system_mode == "demo":
                self.conversation_history = []
            return "Урок завершен! Было очень интересно. Скажите 'привет' чтобы начать новый увлекательный урок."

    def handle_question_during_lesson(self, question: str) -> str:
        """Обработка вопросов во время урока с учетом выбранного режима"""
        if not question.strip():
            return "Повторите вопрос пожалуйста, я не расслышал."
            
        question_lower = question.lower().strip()
        
        # В ДЕМО-РЕЖИМЕ добавляем контекст к запросу LLM
        context = ""
        if self.system_mode == "demo" and self.lesson_content:
            # Берем текущий и предыдущий абзацы для контекста урока
            context_start = max(0, self.current_paragraph - 2)
            context = " ".join(self.lesson_content[context_start:self.current_paragraph])
            
            # Добавляем историю диалога для контекста
            if self.conversation_history:
                context += "\n\nКонтекст диалога:\n"
                for msg in self.conversation_history[-5:]:  # Берем последние 5 реплик
                    role = "Ученик" if msg["role"] == "user" else "Учитель"
                    context += f"{role}: {msg['content']}\n"
        
        # Режим "LLM в первую очередь"
        if self.llm_query_mode == "llm_first":
            print(f"🔀 Режим llm_first: Обработка вопроса '{question}'")
            
            # 1. Сначала пробуем запрос к LLM
            llm_response = self.llm.query(question, context, self.current_subject)
            if llm_response and not llm_response.startswith("Интересный вопрос!"):
                # Сохраняем ответ и возвращаем его
                self.llm.add_to_cache(question, llm_response, self.current_subject)
                if self.knowledge_base:
                    # Сохраняем ответ в базу знаний для будущего использования
                    self.knowledge_base.add_llm_answer(question, llm_response)
                    self.knowledge_base.add_knowledge(question=question, answer=llm_response)
                
                # В ДЕМО-РЕЖИМЕ добавляем ответ учителя в историю
                if self.system_mode == "demo":
                    self.conversation_history.append({"role": "assistant", "content": llm_response})
                
                print(f"✅ Ответ получен от LLM (режим llm_first): {llm_response[:100]}...")
                return llm_response
            
            # 2. Если LLM не дал ответ, проверяем базу знаний
            if self.knowledge_base:
                knowledge_response = self.knowledge_base.get_dialogue_response(question_lower)
                if knowledge_response and not knowledge_response.startswith("Интересный вопрос!"):
                    print(f"📚 Ответ найден в базе знаний после неудачи LLM: {knowledge_response[:100]}...")
                    
                    # В ДЕМО-РЕЖИМЕ добавляем ответ учителя в историю
                    if self.system_mode == "demo":
                        self.conversation_history.append({"role": "assistant", "content": knowledge_response})
                    
                    return knowledge_response
            
            # 3. Проверяем базу ответов LLM
            if self.knowledge_base:
                llm_answer = self.knowledge_base.find_llm_answer(question, threshold=0.8)
                if llm_answer:
                    print(f"💾 Использован сохраненный ответ LLM: {llm_answer[:100]}...")
                    
                    # В ДЕМО-РЕЖИМЕ добавляем ответ учителя в историю
                    if self.system_mode == "demo":
                        self.conversation_history.append({"role": "assistant", "content": llm_answer})
                    
                    return llm_answer
            
            # 4. Финальный fallback
            response = "Извините, не удалось найти ответ на ваш вопрос. Давайте продолжим урок."
            
            # В ДЕМО-РЕЖИМЕ добавляем ответ учителя в историю
            if self.system_mode == "demo":
                self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
        
        # Традиционный режим (оригинальная логика)
        else:
            print(f"🔀 Режим traditional: Обработка вопроса '{question}'")
            
            # 1. Сначала проверяем базу знаний по предмету
            if self.knowledge_base:
                knowledge_response = self.knowledge_base.get_dialogue_response(question_lower)
                if knowledge_response and not knowledge_response.startswith("Интересный вопрос!"):
                    
                    # В ДЕМО-РЕЖИМЕ добавляем ответ учителя в историю
                    if self.system_mode == "demo":
                        self.conversation_history.append({"role": "assistant", "content": knowledge_response})
                    
                    return knowledge_response
            
            # 2. Быстрая проверка локальных шаблонов
            for pattern, responses in self.local_patterns.items():
                if pattern in question_lower:
                    response = random.choice(responses)
                    
                    # В ДЕМО-РЕЖИМЕ добавляем ответ учителя в историю
                    if self.system_mode == "demo":
                        self.conversation_history.append({"role": "assistant", "content": response})
                    
                    return response
            
            # 3. Проверка диалоговых шаблонов из базы знаний
            if self.knowledge_base:
                dialogue_response = self.knowledge_base.get_dialogue_response(question_lower)
                if dialogue_response:
                    
                    # В ДЕМО-РЕЖИМЕ добавляем ответ учителя в историю
                    if self.system_mode == "demo":
                        self.conversation_history.append({"role": "assistant", "content": dialogue_response})
                    
                    return dialogue_response
            
            # 4. Поиск в предметной базе знаний с повышенным порогом схожести
            if self.knowledge_base:
                answer = self.knowledge_base.find_answer(question, threshold=0.5)
                if answer and not answer.startswith("Интересный вопрос!"):
                    
                    # В ДЕМО-РЕЖИМЕ добавляем ответ учителя в историю
                    if self.system_mode == "demo":
                        self.conversation_history.append({"role": "assistant", "content": answer})
                    
                    return answer
            
            # 5. Проверяем базу ответов LLM с высокой точностью (порог 0.8)
            if self.knowledge_base:
                llm_answer = self.knowledge_base.find_llm_answer(question, threshold=0.8)
                if llm_answer:
                    print(f"💾 Использован сохраненный ответ LLM для вопроса: {question}")
                    
                    # В ДЕМО-РЕЖИМЕ добавляем ответ учителя в историю
                    if self.system_mode == "demo":
                        self.conversation_history.append({"role": "assistant", "content": llm_answer})
                    
                    return llm_answer
            
            # 6. Запрос к LLM с контекстом текущего урока
            llm_response = self.llm.query(question, context, self.current_subject)
            if llm_response:
                # Сохраняем в кэш LLM и базу знаний ответов
                self.llm.add_to_cache(question, llm_response, self.current_subject)
                if self.knowledge_base:
                    # Сохраняем ответ в базу знаний для будущего использования
                    self.knowledge_base.add_llm_answer(question, llm_response)
                    self.knowledge_base.add_knowledge(question=question, answer=llm_response)
                
                # В ДЕМО-РЕЖИМЕ добавляем ответ учителя в историю
                if self.system_mode == "demo":
                    self.conversation_history.append({"role": "assistant", "content": llm_response})
                
                return llm_response
            
            # 7. Финальный fallback
            response = "Интересный вопрос! Давайте обсудим его после завершения текущего материала, чтобы не отвлекаться."
            
            # В ДЕМО-РЕЖИМЕ добавляем ответ учителя в историю
            if self.system_mode == "demo":
                self.conversation_history.append({"role": "assistant", "content": response})
            
            return response

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
        if self.system_mode == "demo":
            self.conversation_history = []

    def get_available_subjects(self) -> List[str]:
        """Возвращает список доступных предметов"""
        if self.system_mode == "demo":
            demo_lessons = self._get_demo_lessons()
            subjects = list(demo_lessons.keys())
        else:
            subjects = list(self.lessons.keys())
        
        # Всегда добавляем обществознание, даже если нет уроков
        if "обществознание" not in subjects:
            subjects.append("обществознание")
        return subjects

    def get_lessons_for_subject(self, subject: str) -> List[dict]:
        """Возвращает уроки для указанного предмета"""
        if self.system_mode == "demo":
            return self._get_demo_lessons().get(subject, [])
        else:
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

    def set_system_mode(self, mode: str):
        """Установка режима работы системы"""
        if mode in ["normal", "demo"]:
            self.system_mode = mode
            print(f"Установлен режим системы: {mode}")
            # При смене режима сбрасываем историю диалога
            if mode == "normal":
                self.conversation_history = []

    def get_knowledge_stats(self) -> Optional[Dict]:
        """Получение статистики базы знаний"""
        if self.knowledge_base:
            return self.knowledge_base.get_stats()
        return None
