import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from difflib import SequenceMatcher
import random
import re
from knowledge.knowledge_base import KnowledgeBase
from llm import LLMIntegration
from config import get_llm_mode, get_dialogue_settings
import time
import threading
from practice_manager import PracticeManager

class DialogueManager:
    def __init__(self, socketio):
        self.socketio = socketio
        self.dialogue_states = {
            "greeting": self._handle_greeting,
            "subject_selection": self._handle_subject_selection,
            "lesson_reading": self._handle_lesson_reading,
            "practice_session": self._handle_practice_session
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
        self.llm_query_mode = get_llm_mode()
        self.dialogue_settings = get_dialogue_settings()
        self.conversation_history = []
        self.dialogue_knowledge = self._load_dialogue_knowledge()
        self.conversation_context = []
        self.room_id = None
        
        # Менеджер практики
        self.practice_manager = PracticeManager(self.llm)
        
        # Новые поля для практики
        self.practice_active = False
        self.current_question_index = 0
        self.current_expected_answer = ""
        self.waiting_for_answer = False  # Флаг ожидания ответа ученика
        self.current_practice_question = None  # Текущий вопрос практики
        
        # Новые поля для улучшенного диалога
        self.last_subject_prompt_time = 0
        self.subject_prompt_cooldown = 30
        self.subject_prompt_variants = [
            "Давайте выберем предмет для урока! У меня есть: {subjects}. Что вас интересует?",
            "Какой предмет хотите изучить сегодня? Доступно: {subjects}.",
            "Сказать, какие предметы я преподаю? Или может ты хочешь изучить что-то определенное? У меня есть: {subjects}.",
            "Что будем изучать? Выбирайте из: {subjects}.",
            "Готов начать урок! Какой предмет вас интересует? У меня есть: {subjects}."
        ]
        
        # НОВЫЕ ПОЛЯ ДЛЯ ВИЗУАЛИЗАЦИИ - УЛУЧШЕННАЯ ЛОГИКА
        self.visualization_enabled = True  # Флаг визуализации
        self.last_visualization_time = 0
        self.visualization_cooldown = 5  # Уменьшена пауза между визуализациями (сек)
        self.visualization_counter = 0  # Счетчик для чередования визуализаций
        self.paragraphs_since_last_viz = 0  # Счетчик абзацев с последней визуализации
        self.viz_paragraph_interval = 2  # Генерировать визуализацию каждые N абзацев
        
        self._load_lessons()
        
        # Расширенные локальные шаблоны
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

    def _load_dialogue_knowledge(self) -> Dict:
        """Загрузка расширенной базы диалоговых шаблонов"""
        try:
            dialogue_path = Path("knowledge/dialogue_knowledge.json")
            if dialogue_path.exists():
                with open(dialogue_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Ошибка загрузки диалоговых шаблонов: {e}")
        
        return self._get_default_dialogue_patterns()

    def _get_default_dialogue_patterns(self) -> Dict:
        """Возвращает базовые диалоговые шаблоны по умолчанию"""
        return {
            "greeting_patterns": {
                "привет": ["Привет! Рад тебя видеть!", "Здравствуй! Готов к учебе?"],
                "здравствуй": ["Привет! Как настроение?", "Здравствуй! Что будем изучать?"]
            },
            "mood_patterns": {
                "как дела": ["Отлично! А у тебя как?", "Прекрасно! Готов к уроку."]
            },
            "learning_patterns": {
                "хочу учиться": ["Отлично! Какой предмет тебя интересует?", "Супер! Давай выберем тему!"]
            },
            "subject_questions": {
                "что преподаешь": ["У меня есть уроки по разным предметам! Что хочешь изучить?"]
            },
            "metadata": {
                "version": "1.0",
                "type": "default_dialogue_patterns"
            }
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
            print(f"📖 Загрузка урока из файла: {lesson_file}")
            
            if not lesson_file.exists():
                print(f"❌ Файл урока не существует: {lesson_file}")
                return ["Файл урока не найден. Попробуйте другой урок."]
                
            with open(lesson_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"✅ Файл прочитан, длина: {len(content)} символов")
            
            # Разбиваем на абзацы (по пустым строкам)
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            # Если абзацев нет, разбиваем на предложения
            if not paragraphs:
                print("⚠️ Нет абзацев, разбиваем на предложения")
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
            
            print(f"✅ Урок разбит на {len(paragraphs)} абзацев")
            
            if not paragraphs:
                print("❌ Не удалось разбить урок на абзацы")
                return ["Содержание урока временно недоступно. Давайте поговорим на эту тему!"]
                
            return paragraphs
            
        except Exception as e:
            print(f"❌ Ошибка загрузки содержания урока: {e}")
            return ["Ошибка загрузки урока. Попробуйте позже."]

    def _similarity(self, a: str, b: str) -> float:
        """Вычисление схожести строк"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def _add_to_conversation_history(self, text: str, is_user: bool = True):
        """Добавляет реплику в историю диалога"""
        self.conversation_history.append({
            "text": text,
            "is_user": is_user,
            "timestamp": time.time()
        })
        
        # Ограничиваем размер истории
        max_history = self.dialogue_settings.get("context_window", 10)
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]
            
        # Обновляем контекст (последние 3 реплики пользователя)
        if is_user:
            user_messages = [msg['text'] for msg in self.conversation_history if msg['is_user']]
            self.conversation_context = user_messages[-3:] if len(user_messages) > 3 else user_messages

    def _get_conversation_context(self) -> str:
        """Возвращает контекст диалога для LLM"""
        if not self.conversation_history:
            return ""
            
        context = []
        for msg in self.conversation_history[-6:]:  # Последние 6 реплик
            speaker = "Ученик" if msg["is_user"] else "Учитель"
            context.append(f"{speaker}: {msg['text']}")
        
        return "\n".join(context)

    def _limit_response_length(self, response: str, max_sentences: int = 3) -> str:
        """Ограничивает длину ответа количеством предложений"""
        if not response:
            return response
            
        sentences = re.split(r'(?<=[.!?])\s+', response)
        if len(sentences) > max_sentences:
            return ' '.join(sentences[:max_sentences])
        return response

    def _get_dialogue_response(self, text: str) -> Optional[str]:
        """Поиск ответа в диалоговых шаблонов с учетом контекста"""
        text_lower = text.lower().strip()
        
        # 1. Поиск точного совпадения в расширенной базе
        for category, patterns in self.dialogue_knowledge.items():
            if category.endswith('_patterns') and isinstance(patterns, dict):
                for pattern, responses in patterns.items():
                    if pattern in text_lower and responses:
                        return random.choice(responses)
        
        # 2. Контекстный поиск (есть есть история разговора)
        if self.conversation_context:
            last_user_messages = ' '.join(self.conversation_context).lower()
            
            # Поиск контекстных паттернов
            contextual_patterns = self.dialogue_knowledge.get('contextual_patterns', {})
            for pattern, responses in contextual_patterns.items():
                if pattern in last_user_messages and responses:
                    return random.choice(responses)
        
        # 3. Поиск в локальных шаблонов (fallback)
        for pattern, responses in self.local_patterns.items():
            if pattern in text_lower:
                return random.choice(responses)
        
        return None

    def _handle_llm_dialogue(self, text: str) -> Optional[str]:
        """Гарантированная обработка диалога через LLM с контекста"""
        try:
            # Собираем контекст диалога
            context = self._get_conversation_context()
            
            # Формируем промпт в зависимости от состояния
            if self.current_state == "greeting":
                system_prompt = self.dialogue_settings.get("subject_selection_prompt", 
                    "Ты - дружелюбный учитель. Помоги ученику выбрать предмет для изучения. Будь кратким и понятным. Отвечай на русском языке.")
            else:
                system_prompt = f"Ты - учитель по предмету {self.current_subject}. Отвечай кратко и понятно, максимум 2-3 предложения. Отвечай на русском языке."
            
            # ГАРАНТИРОВАННЫЙ запрос к LLM - даже если API ключа нет!
            llm_response = self.llm._query_llm_api(
                prompt=text,
                context=context,
                subject=self.current_subject or "общее",
                system_prompt=system_prompt,
                max_tokens=150
            )
            
            if llm_response:
                # Ограничиваем длину ответа
                limited_response = self._limit_response_length(
                    llm_response, 
                    self.dialogue_settings.get("max_response_length", 3)
                )
                
                return limited_response
                
        except Exception as e:
            print(f"Ошибка запроса к LLM для диалога: {e}")
            # При ошибке возвращаем естественный ответ, а не сообщение об ошибке
        
        # Fallback если LLM не ответил - предлагаем выбор предмета
        return self._get_subject_selection_prompt()

    def _get_subject_selection_prompt(self) -> Optional[str]:
        """Возвращает предложение выбора предмета с учетом кд"""
        current_time = time.time()
        if current_time - self.last_subject_prompt_time < self.subject_prompt_cooldown:
            return None  # Не предлагать выбор слишком часто
        
        self.last_subject_prompt_time = current_time
        subjects = self.get_available_subjects()
        
        if not subjects:
            return "К сожалению, уроки еще не загружены. Попробуйте позже."
        
        subject_list = ", ".join([subj.capitalize() for subj in subjects[:4]])
        if len(subjects) > 4:
            subject_list += " и другие"
        
        # Выбираем случайную фразу из вариантов
        prompt_template = random.choice(self.subject_prompt_variants)
        return prompt_template.format(subjects=subject_list)

    def _add_subject_suggestion(self, original_response: str) -> str:
        """Добавляет предложение выбора предмета к любому ответу ДО начала урока"""
        
        # НИКОГДА не добавляем предложение выбора во время урока
        if self.lesson_started:
            return original_response
        
        # Если ответ уже содержит предложение о выборе предмета, не дублируем
        if any(word in original_response.lower() for word in ['предмет', 'урок', 'выберем', 'изучать', 'интересует']):
            return original_response
        
        # Получаем предложение выбора (с учетом кд)
        subject_prompt = self._get_subject_selection_prompt()
        if not subject_prompt:
            return original_response
        
        # Ограничиваем общую длину ответа
        max_length = 500
        if len(original_response) + len(subject_prompt) > max_length:
            shortened_response = original_response[:max_length - len(subject_prompt) - 3] + "..."
            return shortened_response + " " + subject_prompt
        
        return original_response + " " + subject_prompt

    def _get_contextual_fallback(self) -> str:
        """Возвращает контекстно-зависимый ответ когда ничего не найдено"""
        if not self.conversation_history:
            return "Привет! Я ваш виртуальный учитель. Давайте познакомимся и выберем интересный урок вместе!"
        
        # Анализ контекста разговора
        user_messages = [msg['text'] for msg in self.conversation_history if msg['is_user']]
        last_user_message = user_messages[-1].lower() if user_messages else ""
        
        # Определяем тему разговора по последним сообщениям
        if any(word in last_user_message for word in ['имя', 'зовут', 'меня']):
            return "Приятно познакомиться! Теперь давайте выберем предмет для изучения. Что вас интересует?"
        
        if any(word in last_user_message for word in ['дела', 'настроение', 'чувств']):
            return "Рад это слышать! Так какой предмет хотите изучить сегодня?"
        
        if any(word in last_user_message for word in ['предмет', 'урок', 'учеба', 'изучать']):
            subjects = self.get_available_subjects()
            subject_list = ", ".join([s.capitalize() for s in subjects[:3]]) + " и другие"
            return f"Отлично! У меня есть: {subject_list}. Что выбираете?"
        
        # Стандартный ответ с напоминанием о выборе
        prompt = self._get_subject_selection_prompt()
        return prompt if prompt else "Давайте выберем предмет для изучения. Что вас интересует?"

    def _should_save_to_knowledge_base(self, text: str) -> bool:
        """Определяет, нужно ли сохранять фразу в базу знаний"""
        text_lower = text.lower()
        
        # Исключаем фразы для генерации уроков
        generation_patterns = [
            'давай изучим', 'хочу изучить', 'урок по', 'изучим', 
            'расскажи про', 'хочу узнать про', 'объясни тему',
            'создай урок', 'сгенерируй урок', 'научи меня'
        ]
        
        if any(pattern in text_lower for pattern in generation_patterns):
            return False
        
        # Исключаем команды выбора предметов
        available_subjects = self.get_available_subjects()
        for subject in available_subjects:
            if subject.lower() in text_lower and len(subject) > 3:
                return False
        
        return True

    def generate_lesson_on_demand(self, topic: str) -> Optional[dict]:
        """Генерирует урок по запрошенной теме с помощью LLM"""
        try:
            print(f"🎯 Генерация урока по теме: {topic}")
            
            # Формируем промпт для генерации урока
            system_prompt = """Ты - эксперт по созданию образовательных материалов. 
Создай структурированный урок по заданной теме. Урок должен быть:
1. Информативным и точным
2. Разделен на логические абзацы (разделяй пустыми строками)
3. Адаптирован для учеников
4. На русском языке
5. Содержать практические примеры если уместно

ВАЖНО: Разделяй абзацы ДВУМЯ переводами строки (\\n\\n) для правильного отображения.
Возвращай только текст урока без дополнительных комментариев."""

            # Запрос к LLM с увеличенным количеством токенов
            lesson_content = self.llm._query_llm_api(
                prompt=f"Создай подробный образовательный урок на тему: '{topic}'. Урок должен быть понятным и структурированным.",
                context="",
                subject="общее",
                system_prompt=system_prompt,
                max_tokens=2500  # Увеличиваем лимит токенов
            )
            
            if not lesson_content:
                print("❌ Ошибка: LLM не вернул содержание урока")
                return None
            
            print(f"✅ Получен контент урока, длина: {len(lesson_content)} символов")
            
            # Убедимся, что есть правильное разделение на абзацы
            # Если в ответе нет двойных переводов строк, добавим их после предложений
            if '\n\n' not in lesson_content:
                print("⚠️ В ответе нет двойных переводов строк, добавляем...")
                # Разбиваем на предложения и добавляем двойные переводы строк
                sentences = re.split(r'(?<=[.!?])\s+', lesson_content)
                lesson_content = '\n\n'.join(sentences)
            
            # Создаем файл урока
            lesson_id = f"generated_{topic.lower().replace(' ', '_')}_{int(time.time())}"
            filename = f"{lesson_id}.txt"
            lesson_path = self.lessons_dir / filename
            
            # Записываем контент в файл
            with open(lesson_path, 'w', encoding='utf-8') as f:
                f.write(f"Урок по теме: {topic}\n\n")
                f.write(lesson_content)
            
            print(f"✅ Файл урока создан: {lesson_path}")
            
            # Добавляем в список уроков
            subject = "общее"
            lesson_data = {
                'id': lesson_id,
                'title': f"Урок по теме: {topic}",
                'file_path': lesson_path,
                'type': 'text',
                'is_generated': True
            }
            
            if subject not in self.lessons:
                self.lessons[subject] = []
            self.lessons[subject].append(lesson_data)
            
            print(f"✅ Урок успешно сгенерирован и добавлен в список: {lesson_id}")
            return lesson_data
            
        except Exception as e:
            print(f"❌ Ошибка генерации урока: {e}")
            return None

    def _check_for_lesson_generation_intent(self, text_lower: str) -> bool:
        """
        Проверяет, хочет ли пользователь сгенерировать новый урок по теме.
        Возвращает True, если урок был успешно сгенерирован.
        """
        # Сначала проверяем, не запрашивает ли пользователь существующий предмет
        available_subjects = self.get_available_subjects()
        for subject in available_subjects:
            if subject.lower() in text_lower and len(subject) > 3:
                print(f"Обнаружен существующий предмет: {subject}, пропускаем генерацию")
                return False
        
        # Шаблоны фраз, которые означают "создай урок"
        generation_patterns = [
            r'хочу изучить (.+)',
            r'можешь рассказать про (.+)', 
            r'урок по (.+)',
            r'изучим (.+)',
            r'расскажи про (.+)',
            r'хочу узнать про (.+)',
            r'объясни тему (.+)',
            r'создай урок про (.+)',
            r'сгенерируй урок о (.+)',
            r'научи меня (.+)'
        ]
        
        for pattern in generation_patterns:
            match = re.search(pattern, text_lower)
            if match:
                topic = match.group(1).strip()
                # Удаляем возможные вопросительные слова в конце
                topic = re.sub(r'[.?]$', '', topic)
                if topic and len(topic) > 2:
                    print(f"Обнаружен запрос на генерацию урока по теме: '{topic}'")
                    # Пытаемся сгенерировать урок
                    generated_lesson = self.generate_lesson_on_demand(topic)
                    if generated_lesson:
                        print(f"Урок успешно сгенерирован: {generated_lesson['id']}")
                        # ВАЖНО: Сразу начинаем сгенерированный урок!
                        self._start_generated_lesson(generated_lesson)
                        return True
                    else:
                        print("❌ Не удалось сгенерировать урок")
        return False

    def _start_generated_lesson(self, lesson_data: dict):
        """Начинает сгенерированный урок"""
        try:
            print(f"🚀 Начинаем сгенерированный урок: {lesson_data['title']}")
            
            self.current_subject = "общее"
            self.selected_lesson = lesson_data
            self.lesson_started = True
            self.current_state = "lesson_reading"
            self.current_paragraph = 0
            
            # ВКЛЮЧАЕМ АВТОМАТИЧЕСКУЮ ВИЗУАЛИЗАЦИЮ ПРИ СТАРТЕ УРОКА
            self.enable_visualization()
            
            # Загружаем содержание урока
            print(f"📖 Загрузка содержания урока из: {lesson_data['file_path']}")
            self.lesson_content = self._load_lesson_content(lesson_data['file_path'])
            
            if not self.lesson_content:
                print("❌ Не удалось загрузить содержание урока")
                return
            
            print(f"✅ Урок загружен, количество абзацев: {len(self.lesson_content)}")
            
            # Инициализируем базу знаний
            self.knowledge_base = KnowledgeBase(self.current_subject)
            
            # Очищаем историю диалога при начале урока
            self.conversation_history = []
            self.conversation_context = []
            
            print(f"🎉 Сгенерированный урок '{lesson_data['title']}' успешно начат!")
            
        except Exception as e:
            print(f"❌ Ошибка начала сгенерированного урока: {e}")
            self.lesson_started = False

    # НОВЫЕ МЕТОДЫ ДЛЯ ВИЗУАЛИЗАЦИИ - УЛУЧШЕННАЯ ЛОГИКА
    def _should_generate_visualization(self, text: str) -> bool:
        """Определяет, нужно ли генерировать визуализацию для текста - УЛУЧШЕННАЯ ВЕРСИЯ"""
        # Проверяем кд
        current_time = time.time()
        if current_time - self.last_visualization_time < self.visualization_cooldown:
            return False
            
        # УВЕЛИЧИВАЕМ ЧАСТОТУ ГЕНЕРАЦИИ - генерируем для КАЖДОГО 2-го абзаца
        self.paragraphs_since_last_viz += 1
        if self.paragraphs_since_last_viz < self.viz_paragraph_interval:
            return False
            
        # СБРАСЫВАЕМ счетчик при генерации
        self.paragraphs_since_last_viz = 0
        
        # Более широкий список триггеров для автоматической генерации
        visualization_triggers = [
            'структура', 'схема', 'диаграмма', 'график', 'процесс', 
            'алгоритм', 'иерархия', 'взаимосвязь', 'соотношение',
            'таблица', 'классификация', 'этапы', 'стадии', 'система',
            'модель', 'цепочка', 'последовательность', 'отношение',
            'разделение', 'группировка', 'организация', 'архитектура',
            # Добавляем общие образовательные термины
            'понятие', 'определение', 'теория', 'концепция', 'принцип',
            'механизм', 'функция', 'свойство', 'характеристика',
            # Экономические термины
            'спрос', 'предложение', 'рынок', 'экономика', 'производство',
            'потребление', 'распределение', 'обмен', 'цена', 'стоимость',
            # Математические термины
            'формула', 'уравнение', 'функция', 'график', 'координаты',
            'прямая', 'кривая', 'площадь', 'объем', 'угол',
            # Исторические термины
            'период', 'эпоха', 'династия', 'хронология', 'событие',
            'война', 'революция', 'реформа', 'закон', 'документ'
        ]
        
        text_lower = text.lower()
        
        # Проверяем наличие триггерных слов
        has_trigger = any(trigger in text_lower for trigger in visualization_triggers)
        
        # Проверяем длину текста (достаточно информативный текст)
        is_long_enough = len(text.split()) > 3
        
        # Проверяем наличие структурных индикаторов
        has_structure_indicators = any(indicator in text_lower for indicator in 
                                     ['состоит из', 'включает в себя', 'делится на', 'подразделяется',
                                      'можно разделить', 'выделяют', 'различают', 'существуют',
                                      'с одной стороны', 'с другой стороны', 'во-первых', 'во-вторых'])
        
        # Генерируем для каждого 2-го абзаца ИЛИ если есть явные триггеры
        should_generate = (has_trigger or has_structure_indicators or 
                          self.visualization_counter % 2 == 0) and is_long_enough and self.visualization_enabled
        
        if should_generate:
            print(f"🎯 Генерация визуализации для: {text[:100]}...")
            print(f"   Триггеры: {has_trigger}, Структура: {has_structure_indicators}, Counter: {self.visualization_counter}")
        
        return should_generate

    def _generate_visualization(self, text: str, context: str = ""):
        """Генерация визуализации для текста - УЛУЧШЕННАЯ ВЕРСИЯ"""
        if not self.visualization_enabled:
            return
        
        if self._should_generate_visualization(text):
            try:
                # Обновляем время последней визуализации
                self.last_visualization_time = time.time()
                self.visualization_counter += 1
                
                # ОТПРАВЛЯЕМ ЗАПРОС НА ГЕНЕРАЦИЮ ВИЗУАЛИЗАЦИИ - НЕМЕДЛЕННАЯ ГЕНЕРАЦИЯ
                if self.room_id and self.socketio:
                    print(f"📊 Отправка запроса на генерацию визуализации для: {text[:100]}...")
                    
                    # Используем прямой запрос вместо очереди для немедленной генерации
                    self.socketio.emit('generate_visualization', {
                        'room_id': self.room_id,
                        'topic': text,
                        'context': context
                    }, room=self.room_id)
                    
            except Exception as e:
                print(f"❌ Ошибка запроса визуализации: {e}")

    def enable_visualization(self):
        """Включение автоматической визуализации"""
        self.visualization_enabled = True
        print("✅ Автоматическая визуализация включена")

    def disable_visualization(self):
        """Выключение автоматической визуализации"""
        self.visualization_enabled = False
        print("❌ Автоматическая визуализация выключена")

    def process_input(self, text: str) -> Optional[str]:
        """Обработка входящего текста и генерация ответа с гарантированным результатом"""
        text_lower = text.lower().strip()
        
        # УЛУЧШЕННАЯ ОБРАБОТКА КОМАНД УПРАВЛЕНИЯ (всегда приоритет)
        continue_commands = ["продолжай", "продолжить", "дальше", "следующий", "вперед", "давай дальше"]
        recorded_commands = ["записал", "понял", "ясно", "ага", "угу", "хорошо", "ок", "ладно", "ясно"]
        
        # Если урок начат и это команда продолжения
        if self.lesson_started and any(cmd in text_lower for cmd in continue_commands + recorded_commands):
            next_paragraph = self._get_next_paragraph()
            if next_paragraph:
                return next_paragraph
            else:
                return "Урок завершен. Переходим к практике."
        
        # Добавляем в историю диалога ВСЕГДА
        self._add_to_conversation_history(text, is_user=True)
        
        # 1. Если урок уже начат - используем стандартную логику (ПЕРВОЕ условие!)
        if self.lesson_started:
            handler = self.dialogue_states.get(self.current_state)
            if handler:
                response = handler(text_lower)
                if response:
                    self._add_to_conversation_history(response, is_user=False)
                    return response
            return None
        
        # 2. ПРОВЕРКА НА КОМАНДУ ГЕНЕРАЦИИ УРОКА (только если урок не начат)
        generated_lesson = self._check_for_lesson_generation_intent(text_lower)
        if generated_lesson:
            # Урок уже начат в _start_generated_lesson, возвращаем None для чтения первого абзаца
            return None
        
        # 3. Если пользователь называет предмет - автоматически начинаем урок
        available_subjects = self.get_available_subjects()
        for subject in available_subjects:
            if subject.lower() in text_lower and len(subject) > 3:
                print(f"Обнаружен выбор предмета: {subject}")
                return self._handle_subject_selection_direct(subject)
        
        # 4. Если ждем ответ на вопрос практики - обрабатываем как ответ
        if self.practice_active and self.waiting_for_answer:
            return self._handle_practice_answer(text)
        
        # 5. Поиск в диалоговых шаблонах (до выбора урока)
        dialogue_response = self._get_dialogue_response(text_lower)
        if dialogue_response:
            # ДОБАВЛЯЕМ предложение выбора предмета к любому ответу
            final_response = self._add_subject_suggestion(dialogue_response)
            if final_response:
                self._add_to_conversation_history(final_response, is_user=False)
                return final_response
        
        # 6. ГАРАНТИРОВАННЫЙ запрос к LLM если не найден в шаблонах
        llm_response = self._handle_llm_dialogue(text)
        if llm_response:
            # ДОБАВЛЯЕМ предложение выбора предмета к ответу LLM
            final_response = self._add_subject_suggestion(llm_response)
            if final_response:
                self._add_to_conversation_history(final_response, is_user=False)
                return final_response
        
        # 7. Финальный fallback с предложением выбора предмета
        fallback_response = self._get_contextual_fallback()
        if fallback_response:
            self._add_to_conversation_history(fallback_response, is_user=False)
            return fallback_response
        
        return None

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
        
        self.lesson_started = True
        self.current_state = "lesson_reading"
        self.current_paragraph = 0
        self.lesson_content = self._load_lesson_content(self.selected_lesson['file_path'])
        self.knowledge_base = KnowledgeBase(self.current_subject)
        
        # ВКЛЮЧАЕМ АВТОМАТИЧЕСКУЮ ВИЗУАЛИЗАЦИЮ ПРИ СТАРТЕ УРОКА
        self.enable_visualization()
        
        # Очищаем историю диалога при начале урока
        self.conversation_history = []
        self.conversation_context = []
        
        # Возвращаем None, чтобы сразу начать чтение урока без подтверждения
        return None

    def _handle_greeting(self, text: str) -> Optional[str]:
        greeting_words = ["привет", "здравствуй", 'начать', "старт", " готов", "поехали", "давай", "началом"]
        if any(word in text for word in greeting_words):
            self.current_state = "subject_selection"
            prompt = self._get_subject_selection_prompt()
            return prompt if prompt else "Давайте выберем предмет для изучения. Что вас интересует?"
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
            prompt = self._get_subject_selection_prompt()
            return prompt if prompt else "Отлично! Какой предмет вас заинтересовал? Назовите его пожалуйста."
            
        return None

    def _handle_lesson_reading(self, text: str) -> Optional[str]:
        """Обработка во время чтения урока"""
        if any(word in text for word in ["стоп", "останови", "хватит", "закончи"]):
            self.lesson_started = False
            self.current_state = "greeting"
            self.conversation_counter = 0
            self.knowledge_base = None
            self.conversation_history = []
            self.conversation_context = []
            return "Урок остановлен. Скажите 'привет' когда захотите продолжить или выбрать новый урок."
            
        # Если это не команда управления чтением, обрабатываем как вопрос
        return None

    def _handle_practice_session(self, text: str) -> Optional[str]:
        """Обработка во время фазы практики"""
        if any(word in text for word in ["стоп", "останови", "хватит", "закончи"]):
            self.practice_active = False
            self.waiting_for_answer = False
            self.current_state = "greeting"
            self.conversation_counter = 0
            self.conversation_history = []
            self.conversation_context = []
            
            # Отправляем событие завершения практики
            if self.room_id:
                self.socketio.emit('practice_ended', {'room_id': self.room_id})
            
            return "Практика остановлена. Скажите 'привет' когда захотите продолжить или выбрать новый урок."
            
        # Если ждем ответ, обрабатываем как ответ на вопрос практики
        if self.waiting_for_answer:
            return self._handle_practice_answer(text)
            
        return None

    def _get_next_paragraph(self) -> Optional[str]:
        """Возвращает следующий абзац урока с возможной визуализацией"""
        print(f"📄 Получение следующего абзаца: текущий {self.current_paragraph}, всего {len(self.lesson_content)}")
        
        if self.current_paragraph < len(self.lesson_content):
            paragraph = self.lesson_content[self.current_paragraph]
            self.current_paragraph += 1
            
            # АВТОМАТИЧЕСКАЯ ГЕНЕРАЦИЯ ВИЗУАЛИЗАЦИИ ДЛЯ КАЖДОГО АБЗАЦА
            if self.visualization_enabled:
                context = " ".join(self.lesson_content[:self.current_paragraph])
                self._generate_visualization(paragraph, context)
            
            print(f"✅ Возвращаем абзац {self.current_paragraph}: {paragraph[:100]}...")
            return paragraph
        else:
            # Урок завершен - запускаем практику
            print("🏁 Урок завершен, запускаем практику")
            practice_message = self._start_practice_session()
            return practice_message

    def _start_practice_session(self) -> str:
        """Запускает фазу практики с последовательной генерацией вопросов"""
        self.lesson_started = False
        self.current_state = "practice_session"
        self.practice_active = True
        self.waiting_for_answer = False
        self.current_question_index = 0
        
        print("=== ЗАПУСК ФАЗЫ ПРАКТИКИ С ПОСЛЕДОВАТЕЛЬНОЙ ГЕНЕРАЦИЕЙ ===")
        print(f"practice_active: {self.practice_active}, waiting_for_answer: {self.waiting_for_answer}")
        
        # Инициализируем контекст для генерации вопросов
        lesson_context = " ".join(self.lesson_content)
        self.practice_manager.initialize_practice_generation(lesson_context, self.current_subject)
        
        # Сохраняем практику в TXT файл
        if self.selected_lesson:
            lesson_id = self.selected_lesson.get('id', 'unknown')
            # Создаем базовую структуру практики для сохранения
            practice_data = {
                'questions': [],
                'lesson_id': lesson_id,
                'subject': self.current_subject
            }
            self.practice_manager.save_practice_to_txt(lesson_id, practice_data)
        
        # Отправляем событие начала практики
        if self.room_id:
            self.socketio.emit('practice_started', {'room_id': self.room_id})
        
        # Генерируем и возвращаем ПЕРВЫЙ вопрос
        first_question = self._generate_next_practice_question()
        if first_question:
            print(f"Начинаем практику с вопроса: {first_question}")
            print(f"Установлен waiting_for_answer: {self.waiting_for_answer}")
            return f"Отлично! Переходим к практике. Первый вопрос: {first_question}"
        else:
            print("Не удалось сгенерировать первый вопрос практики")
            self.practice_active = False
            return "Практические задания временно недоступны."

    def _generate_next_practice_question(self) -> Optional[str]:
        """Генерирует следующий вопрос практики через LLM"""
        try:
            # Генерируем вопрос на основе контекста урока
            question = self.practice_manager.generate_single_question()
            if question:
                self.current_practice_question = {
                    "id": self.current_question_index + 1,
                    "question": question,
                    "answer": ""  # Ответ будет сгенерирован при проверке
                }
                self.current_question_index += 1
                # ВАЖНО: Устанавливаем флаг ожидания ответа!
                self.waiting_for_answer = True
                print(f"✅ Сгенерирован вопрос {self.current_question_index}: {question}")
                print(f"📊 Установлен waiting_for_answer: {self.waiting_for_answer}")
                return question
            else:
                print("❌ Не удалось сгенерировать вопрос")
                return None
        except Exception as e:
            print(f"❌ Ошибка генерации вопроса практики: {e}")
            return None

    def _evaluate_and_generate_next(self, student_answer: str) -> str:
        """Оценивает ответ и генерирует следующий вопрос (новый метод)"""
        print(f"🔍 Обработка ответа: '{student_answer}'")
        print(f"📊 Состояние: practice_active={self.practice_active}, waiting_for_answer={self.waiting_for_answer}")
        
        if not self.practice_active:
            print("❌ Практика не активна")
            return "Практика не активна."
        
        print(f"🎯 Оценка ответа и генерация следующего вопроса...")
        
        # Получаем текущий вопрос
        current_question = self.current_practice_question
        if not current_question:
            self._end_practice_session()
            return "Практика завершена."
        
        # Оцениваем ответ ученика и получаем обратную связь
        evaluation = self.practice_manager.evaluate_single_answer(
            student_answer, 
            current_question["question"]
        )
        
        print(f"📝 Оценка: {evaluation}")
        
        # Проверяем, нужно ли генерировать следующий вопрос
        if self.current_question_index < 5:  # Всего 5 вопросов
            next_question = self._generate_next_practice_question()
            if next_question:
                response = f"{evaluation}. Следующий вопрос: {next_question}"
                print(f"➡️ Следующий вопрос сгенерирован: {next_question}")
                print(f"📊 Установлен waiting_for_answer: {self.waiting_for_answer}")
                return response
            else:
                # Не удалось сгенерировать следующий вопрос
                print("❌ Не удалось сгенерировать следующий вопрос")
                self._end_practice_session()
                return f"{evaluation}. Практика завершена - возникли трудности с генерацией вопросов."
        else:
            # Достигли лимита вопросов
            print("🏁 Достигнут лимит вопросов (5)")
            self._end_practice_session()
            return f"{evaluation}. Практика завершена! Вы ответили на все 5 вопросов. Отличная работа!"

    def _handle_practice_answer(self, text: str) -> str:
        """Обработчик ответов во время практики"""
        return self._evaluate_and_generate_next(text)

    def _end_practice_session(self):
        """Корректно завершает сессию практики"""
        self.practice_active = False
        self.waiting_for_answer = False
        self.current_state = "greeting"
        self.practice_manager.reset()
        
        # ВАЖНО: Сбрасываем состояние урока, чтобы можно было начать новый
        self.lesson_started = False
        self.selected_lesson = None
        self.current_subject = None
        self.lesson_content = []
        self.current_paragraph = 0
        
        if self.room_id:
            self.socketio.emit('practice_ended', {'room_id': self.room_id})
        print("=== 🏁 ПРАКТИКА ЗАВЕРШЕНА ===")

    def handle_question_during_lesson(self, question: str) -> str:
        """Обработка вопросов во время урока с учетом выбранного режима"""
        if not question.strip():
            return "Повторите вопрос пожалуйста, я не расслышал."
            
        question_lower = question.lower().strip()
        
        # АВТОМАТИЧЕСКАЯ ГЕНЕРАЦИЯ ВИЗУАЛИЗАЦИИ ДЛЯ ВОПРОСОВ
        if self.visualization_enabled:
            context = " ".join(self.lesson_content[max(0, self.current_paragraph-2):self.current_paragraph])
            self._generate_visualization(question, context)
        
        # НЕМЕДЛЕННАЯ обработка вопроса (без асинхронности)
        print(f"Немедленная обработка вопроса: '{question}'")
        final_response = None
        
        # Режим "LLM в первую очередь"
        if self.llm_query_mode == "llm_first":
            print(f"🔀 Режим llm_first: Обработка вопроса '{question}'")
            
            # 1. Сначала пробуем запрос к LLM
            current_context = ""
            if self.lesson_content and self.current_paragraph > 0:
                # Берем текущий и предыдущий абзацы для контекста
                context_start = max(0, self.current_paragraph - 2)
                current_context = " ".join(self.lesson_content[context_start:self.current_paragraph])
            
            llm_response = self.llm.query(question, current_context, self.current_subject)
            if llm_response and not llm_response.startswith("Интересный вопрос!"):
                # Сохраняем ответ и возвращаем его
                self.llm.add_to_cache(question, llm_response, self.current_subject)
                if self.knowledge_base and self._should_save_to_knowledge_base(question):
                    # Сохраняем ответ в базу знаний для будущего использования
                    self.knowledge_base.add_llm_answer(question, llm_response)
                    self.knowledge_base.add_knowledge(question=question, answer=llm_response)
                    # Сохраняем в диалоговую базу
                    self.knowledge_base.add_to_dialogue_knowledge(question, llm_response)
                print(f"✅ Ответ получен от LLM (режим llm_first): {llm_response[:100]}...")
                final_response = llm_response
            
            # 2. Если LLM не дал ответ, проверяем базу знаний
            if not final_response and self.knowledge_base:
                knowledge_response = self.knowledge_base.get_dialogue_response(question_lower)
                if knowledge_response and not knowledge_response.startswith("Интересный вопрос!"):
                    print(f"📚 Ответ найден в базе знаний после неудачи LLM: {knowledge_response[:100]}...")
                    final_response = knowledge_response
            
            # 3. Проверяем базу ответов LLM
            if not final_response and self.knowledge_base:
                llm_answer = self.knowledge_base.find_llm_answer(question, threshold=0.8)
                if llm_answer:
                    print(f"💾 Использован сохраненный ответ LLM: {llm_answer[:100]}...")
                    final_response = llm_answer
        
        # Традиционный режим (оригинальная логика)
        else:
            print(f"🔀 Режим traditional: Обработка вопроса '{question}'")
            
            # 1. Сначала проверяем базу знаний по предмету
            if self.knowledge_base:
                knowledge_response = self.knowledge_base.get_dialogue_response(question_lower)
                if knowledge_response and not knowledge_response.startswith("Интересный вопрос!"):
                    final_response = knowledge_response
            
            # 2. Быстрая проверка локальных шаблонов
            if not final_response:
                for pattern, responses in self.local_patterns.items():
                    if pattern in question_lower:
                        final_response = random.choice(responses)
                        break
            
            # 3. Проверка диалоговых шаблонов из базы знаний
            if not final_response and self.knowledge_base:
                dialogue_response = self.knowledge_base.get_dialogue_response(question_lower)
                if dialogue_response:
                    final_response = dialogue_response
            
            # 4. Поиск в предметной базе знаний с повышенным порогом схожести
            if not final_response and self.knowledge_base:
                answer = self.knowledge_base.find_answer(question, threshold=0.5)
                if answer and not answer.startswith("Интересный вопрос!"):
                    final_response = answer
            
            # 5. Проверяем базу ответов LLM с высокой точностью (порог 0.8)
            if not final_response and self.knowledge_base:
                llm_answer = self.knowledge_base.find_llm_answer(question, threshold=0.8)
                if llm_answer:
                    print(f"💾 Использован сохраненный ответ LLM для вопроса: {question}")
                    final_response = llm_answer
            
            # 6. Запрос к LLM с контекстом текущего урока
            if not final_response:
                current_context = ""
                if self.lesson_content and self.current_paragraph > 0:
                    # Берем текущий и предыдущий абзацы для контекста
                    context_start = max(0, self.current_paragraph - 2)
                    current_context = " ".join(self.lesson_content[context_start:self.current_paragraph])
                
                llm_response = self.llm.query(question, current_context, self.current_subject)
                if llm_response:
                    # Сохраняем в кэш LLM и базу знаний ответов
                    self.llm.add_to_cache(question, llm_response, self.current_subject)
                    if self.knowledge_base and self._should_save_to_knowledge_base(question):
                        # Сохраняем ответ в базу знаний для будущего использования
                        self.knowledge_base.add_llm_answer(question, llm_response)
                        self.knowledge_base.add_knowledge(question=question, answer=llm_response)
                        # Сохраняем в диалоговую базу
                        self.knowledge_base.add_to_dialogue_knowledge(question, llm_response)
                    final_response = llm_response
        
        # 7. Финальный fallback
        if not final_response:
            final_response = "Интересный вопрос! Давайте обсудим его после завершения текущего материала, чтобы не отвлекаться."
        
        # Возвращаем ответ сразу для озвучивания
        return final_response

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
        self.conversation_history = []
        self.conversation_context = []
        self.practice_active = False
        self.waiting_for_answer = False
        self.current_question_index = 0

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

    def set_llm_model(self, model: str):
        """Установка модели LLM"""
        self.llm.set_model(model)
        print(f"Установлена модель LLM: {model}")

    def set_llm_mode(self, mode: str):
        """Установка режима запросов к LLM"""
        if mode in ["traditional", "llm_first"]:
            self.llm_query_mode = mode
            print(f"Установлен режим LLM: {mode}")

    def get_knowledge_stats(self) -> Optional[Dict]:
        """Получение статистики базы знаний"""
        if self.knowledge_base:
            return self.knowledge_base.get_stats()
        return None