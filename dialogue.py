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
from infographic_generator import infographic_generator

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
        self.waiting_for_answer = False
        self.current_practice_question = None
        self.max_questions = 5  # Лимит вопросов для практики
        
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
        
        # НОВЫЕ ПОЛЯ ДЛЯ РЕЖИМА УЧЕНИКА
        self.is_student_mode = False
        self.auto_selected_subject = None
        self.student_conversation_count = 0
        self.student_lesson_started = False
        self.student_subject_prompted = False
        
        # НОВЫЕ ПОЛЯ ДЛЯ ДАННЫХ УЧЕНИКА (для промтов)
        self.student_data = {}
        
        # НОВЫЕ ПОЛЯ ДЛЯ ИНФОГРАФИКИ
        self.visualization_enabled = True
        self.last_visualization_time = 0
        self.visualization_cooldown = 5
        self.visualization_counter = 0
        self.paragraphs_since_last_viz = 0
        self.viz_paragraph_interval = 2
        
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
        """Загружает содержание урока из текстового файла с улучшенной очисткой"""
        try:
            print(f"📖 Загрузка урока из файла: {lesson_file}")
            
            if not lesson_file.exists():
                print(f"❌ Файл урока не существует: {lesson_file}")
                return ["Файл урока не найден. Попробуйте другой урок."]
                
            with open(lesson_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"✅ Файл прочитан, длина: {len(content)} символов")
            
            # УЛУЧШЕННАЯ ОЧИСТКА СОДЕРЖАНИЯ
            content = self._clean_lesson_content(content)
            
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
                        if len(current_paragraph) >= 2:
                            paragraphs.append(' '.join(current_paragraph))
                            current_paragraph = []
                
                # Добавляем оставшиеся предложения
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
            
            print(f"✅ Урок разбит на {len(paragraphs)} абзацев")
            
            if not paragraphs:
                print("❌ Не удалось разбить урок на абзацев")
                return ["Содержание урока временно недоступно. Давайте поговорим на эту тему!"]
                
            return paragraphs
            
        except Exception as e:
            print(f"❌ Ошибка загрузки содержания урока: {e}")
            return ["Ошибка загрузки урока. Попробуйте позже."]

    def _clean_lesson_content(self, content: str) -> str:
        """Очистка содержания урока от лишнего форматирования"""
        if not content:
            return content
        
        # Удаляем маркеры форматирования
        content = re.sub(r'[\*\#]{1,}', '', content)  # Удаляем одиночные * и #
        content = re.sub(r'\-\-\-+', '', content)  # Удаляем разделители ---
        content = re.sub(r'\+\+\+', '', content)  # Удаляем +++
        
        # Удаляем HTML-теги если есть
        content = re.sub(r'<[^>]+>', '', content)
        
        # Нормализуем переводы строк
        content = re.sub(r'\r\n', '\n', content)
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        # Удаляем начальные/конечные пробелы
        content = content.strip()
        
        return content

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
        for msg in self.conversation_history[-6:]:
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

    def _handle_llm_dialogue(self, text: str, room_id: str = None) -> Optional[str]:
        """Гарантированная обработка диалога через LLM с контекстом"""
        try:
            # Собираем контекст диалога
            context = self._get_conversation_context()
            
            # 🔥 ОБНОВЛЕННЫЙ ПРОМТ: Добавляем данные ученика если есть
            age = self.student_data.get('age', '12')
            level = self.student_data.get('level', '5')
            name = self.student_data.get('name', 'ученик')
            
            # Формируем промпт в зависимости от состояния и наличия данных ученика
            if self.current_state == "greeting":
                if self.student_data:
                    # Персонализированный промпт для ученика
                    system_prompt = f"""Ты - дружелюбный учитель для ученика {age} лет, {level} класс.

ОСОБЕННОСТИ УЧЕНИКА:
- Имя: {name}
- Возраст: {age} лет  
- Уровень: {level} класс
- Предмет: {self.current_subject or 'не выбран'}

СТИЛЬ ОБЩЕНИЯ:
- Обращайся на "ты"
- Используй язык, понятный для {age}-летнего
- Будь поддерживающим и терпеливым
- Объясняй сложные вещи простыми словами
- Используй примеры, релевантные для этого возраста

ОТВЕТЫ ДОЛЖНЫ БЫТЬ:
- Краткими (2-3 предложения максимум)
- Понятными для {age}-летнего
- Конкретными и полезными
- На русском языке

Помоги ученику выбрать предмет для изучения. Будь кратким и понятным."""
                else:
                    # Стандартный промпт для обычного пользователя
                    system_prompt = self.dialogue_settings.get("subject_selection_prompt", 
                        "Ты - дружелюбный учитель. Помоги ученику выбрать предмет для изучения. Будь кратким и понятным. Отвечай на русском языке.")
            else:
                if self.student_data:
                    # Персонализированный промпт для урока
                    system_prompt = f"""Ты - учитель по предмету {self.current_subject} для ученика {age} лет, {level} класс.

ОСОБЕННОСТИ УЧЕНИКА:
- Имя: {name}
- Возраст: {age} лет
- Уровень: {level} класс

СТИЛЬ ОБЩЕНИЯ:
- Обращайся на "ты" 
- Используй язык, понятный для {age}-летнего
- Объясняй сложные понятия простыми словами
- Адаптируй сложность объяснений под возраст ученика

ОТВЕТЫ ДОЛЖНЫ БЫТЬ:
- Краткими (2-3 предложения максимум)
- Понятными для {age}-летнего
- Конкретными и полезными
- На русском языке"""
                else:
                    # Стандартный промпт для урока
                    system_prompt = f"Ты - учитель по предмету {self.current_subject}. Отвечай кратко и понятно, максимум 2-3 предложения. Отвечай на русском языке."
            
            # АСИНХРОННЫЙ запрос к локальной модели
            if room_id and self.socketio:
                # Используем асинхронный режим с callback
                def llm_callback(response, r_id):
                    if response:
                        limited_response = self._limit_response_length(
                            response, 
                            self.dialogue_settings.get("max_response_length", 3)
                        )
                        
                        # Отправляем ответ через WebSocket
                        self.socketio.emit('llm_dialogue_response', {
                            'room_id': r_id,
                            'response': limited_response,
                            'original_text': text
                        }, room=r_id)
                
                # Асинхронный запрос - не блокируем основной поток
                self.llm._query_llm_api(
                    prompt=text,
                    context=context,
                    subject=self.current_subject or "общее",
                    system_prompt=system_prompt,
                    max_tokens=150,
                    room_id=room_id,
                    callback=llm_callback
                )
                
                return None  # Ответ придет асинхронно
                
            else:
                # Синхронный режим для обратной совместимости
                llm_response = self.llm._query_llm_api(
                    prompt=text,
                    context=context,
                    subject=self.current_subject or "общее",
                    system_prompt=system_prompt,
                    max_tokens=150
                )
                
                if llm_response:
                    limited_response = self._limit_response_length(
                        llm_response, 
                        self.dialogue_settings.get("max_response_length", 3)
                    )
                    return limited_response
                    
        except Exception as e:
            print(f"Ошибка запроса к LLM для диалога: {e}")
        
        return self._get_subject_selection_prompt()

    def _get_subject_selection_prompt(self) -> Optional[str]:
        """Возвращает предложение выбора предмета с учетом кд"""
        current_time = time.time()
        if current_time - self.last_subject_prompt_time < self.subject_prompt_cooldown:
            return None
        
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
        
        # В режиме ученика не предлагаем выбор предмета
        if self.is_student_mode and self.auto_selected_subject:
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
            if self.student_data:
                name = self.student_data.get('name', 'ученик')
                return f"Привет, {name}! Я твой виртуальный учитель. Давайте познакомимся и выберем интересный урок вместе!"
            else:
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
            
            # 🔥 ОБНОВЛЕННЫЙ ПРОМТ: Добавляем данные ученика для адаптации сложности
            age = self.student_data.get('age', '12')
            level = self.student_data.get('level', '5')
            name = self.student_data.get('name', 'ученик')
            
            # Формируем промпт для генерации урока с учетом возраста
            system_prompt = f"""Ты - эксперт по созданию образовательных материалов.

ВАЖНЫЕ ПАРАМЕТРЫ УЧЕНИКА:
- Имя: {name}
- Возраст: {age} лет
- Уровень образования: {level} класс
- Предмет: {self.current_subject or 'общее'}

Создай структурированный урок по заданной теме. Урок должен быть:

1. АДАПТИРОВАН ПОД ВОЗРАСТ {age} ЛЕТ:
   - Используй язык и примеры, понятные для {age}-летнего
   - Сложность материала должна соответствовать {level} классу
   - Длина предложений и абзацев должна подходить для этого возраста
   - Используй примеры и аналогии, релевантные для ученика {age} лет

2. СОДЕРЖАТЕЛЬНЫЕ ТРЕБОВАНИЯ:
   - Информативным и точным
   - Разделен на логические абзацы (разделяй пустыми строками)
   - Содержать практические примеры если уместно
   - Быть увлекательным и интересным

3. ФОРМАТИРОВАНИЕ:
   - Разделяй абзацы ДВУМЯ переводами строки (\\n\\n)
   - Используй подходящий для возраста стиль изложения
   - Объясняй сложные понятия простыми словами

Тема урока: '{topic}'

Возвращай только текст урока без дополнительных комментариев."""

            # Запрос к LLM с увеличенным количеством токенов
            lesson_content = self.llm._query_llm_api(
                prompt=f"Создай подробный образовательный урок на тему: '{topic}'. Урок должен быть понятным и структурированным.",
                context="",
                subject=self.current_subject or "общее",
                system_prompt=system_prompt,
                max_tokens=2500
            )
            
            if not lesson_content:
                print("❌ Ошибка: LLM не вернул содержание урока")
                return None
            
            print(f"✅ Получен контент урока, длина: {len(lesson_content)} символов")
            
            # Убедимся, что есть правильное разделение на абзацы
            if '\n\n' not in lesson_content:
                print("⚠️ В ответе нет двойных переводов строк, добавляем...")
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
            subject = self.current_subject or "общее"
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
        Возвращает True, если урок был успешно сгенерирован И СОХРАНЕН.
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
                topic = re.sub(r'[.?]$', '', topic)
                if topic and len(topic) > 2:
                    print(f"🎯 Обнаружен запрос на генерацию урока по теме: '{topic}'")
                    
                    # ГЕНЕРИРУЕМ И СОХРАНЯЕМ УРОК
                    generated_lesson = self.generate_lesson_on_demand(topic)
                    if generated_lesson:
                        print(f"✅ Урок успешно сгенерирован и сохранен: {generated_lesson['id']}")
                        
                        # КРИТИЧЕСКИ ВАЖНО: Добавляем урок в список доступных уроков
                        subject = self.current_subject or "общее"
                        if subject not in self.lessons:
                            self.lessons[subject] = []
                        
                        # Проверяем, нет ли уже такого урока
                        lesson_exists = any(lesson['id'] == generated_lesson['id'] for lesson in self.lessons[subject])
                        if not lesson_exists:
                            self.lessons[subject].append(generated_lesson)
                            print(f"✅ Урок добавлен в список уроков по предмету: {subject}")
                        
                        # НАЧИНАЕМ УРОК - это запускает отображение
                        self._start_generated_lesson(generated_lesson)
                        return True
                    else:
                        print("❌ Не удалось сгенерировать урок")
        return False

    def _start_generated_lesson(self, lesson_data: dict):
        """Начинает сгенерированный урок с гарантированным отображением"""
        try:
            print(f"🚀 НАЧИНАЕМ сгенерированный урок: {lesson_data['title']}")
            
            self.current_subject = self.current_subject or "общее"
            self.selected_lesson = lesson_data
            self.lesson_started = True
            self.current_state = "lesson_reading"
            self.current_paragraph = 0
            
            # ВКЛЮЧАЕМ АВТОМАТИЧЕСКУЮ ВИЗУАЛИЗАЦИЮ
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
            
            # КРИТИЧЕСКИ ВАЖНО: Уведомляем клиент о начале урока
            if self.room_id and self.socketio:
                self.socketio.emit('lesson_started', {
                    'lesson_id': lesson_data['id'],
                    'title': lesson_data['title'],
                    'subject': self.current_subject,
                    'is_generated': True
                }, room=self.room_id)
                print(f"📢 Уведомление о начале урока отправлено в комнату {self.room_id}")
            
            print(f"🎉 Сгенерированный урок '{lesson_data['title']}' успешно начат и отображается!")
            
        except Exception as e:
            print(f"❌ Ошибка начала сгенерированного урока: {e}")
            self.lesson_started = False

    def _has_visualization_triggers(self, text: str) -> bool:
        """Проверяет наличие триггеров для инфографики"""
        text_lower = text.lower()
        
        visualization_triggers = [
            'структура', 'схема', 'диаграмма', 'график', 'процесс', 
            'алгоритм', 'иерархия', 'взаимосвязь', 'соотношение',
            'таблица', 'классификация', 'этапы', 'стадии', 'система'
        ]
        
        structure_indicators = [
            'состоит из', 'включает в себя', 'делится на', 'подразделяется',
            'можно разделить', 'выделяют', 'различают', 'существуют'
        ]
        
        has_trigger = any(trigger in text_lower for trigger in visualization_triggers)
        has_structure = any(indicator in text_lower for indicator in structure_indicators)
        
        return has_trigger or has_structure

    def _generate_visualization(self, text: str, context: str = ""):
        """Генерация инфографики для текста"""
        if not self.visualization_enabled or not text.strip():
            return
    
        current_time = time.time()
        if current_time - self.last_visualization_time < self.visualization_cooldown:
            return
        
        self.paragraphs_since_last_viz += 1
        
        should_generate = (self.paragraphs_since_last_viz >= self.viz_paragraph_interval or 
                          self._has_visualization_triggers(text))
        
        if should_generate:
            try:
                self.last_visualization_time = current_time
                self.paragraphs_since_last_viz = 0
                self.visualization_counter += 1
                
                print(f"🎨 Генерация инфографики для: {text[:100]}...")
                
                if self.room_id and self.socketio:
                    # Используем новый генератор инфографики
                    result = infographic_generator.generate_infographic(text, context)
                    
                    if result and result.get("success"):
                        self.socketio.emit('infographic_generated', {
                            'room_id': self.room_id,
                            'topic': text[:100],
                            'svg_code': result['svg_code'],
                            'style': result['style'],
                            'timestamp': time.time()
                        }, room=self.room_id)
                        print(f"✅ Инфографика отправлена в комнату {self.room_id}")
                    
            except Exception as e:
                print(f"❌ Ошибка генерации инфографики: {e}")

    def enable_visualization(self):
        """Включение автоматической инфографики"""
        self.visualization_enabled = True
        print("✅ Автоматическая инфографика включена")

    def disable_visualization(self):
        """Выключение автоматической инфографики"""
        self.visualization_enabled = False
        print("❌ Автоматическая инфографика выключена")

    def process_input(self, text: str) -> Optional[str]:
        """Обработка входящего текста и генерация ответа с гарантированным результатом"""
        text_lower = text.lower().strip()
        
        # РАСШИРЕННЫЙ СПИСОК КОМАНД ПРОДОЛЖЕНИЯ - РАБОТАЕТ ЛЮБАЯ ИЗ НИХ В ЛЮБОЙ ПОСЛЕДОВАТЕЛЬНОСТИ
        continue_commands = [
            "продолжай", "продолжить", "дальше", "следующий", "вперед", "давай дальше",
            "записал", "понял", "ясно", "ага", "угу", "хорошо", "ок", " ладно", "ясно",
            "готов", "можно дальше", "следующая часть", "продолжаем", "всё", "все"
        ]

        if self.lesson_started and any(cmd in text_lower for cmd in continue_commands):
            next_paragraph = self._get_next_paragraph()
            if next_paragraph:
                print(f"✅ Команда продолжения обработана: '{text_lower}' -> следующий абзац")
                return next_paragraph
            else:
                print("🏁 Урок завершен по команде продолжения")
                return "Урок завершен. Переходим к практике."
        
        self._add_to_conversation_history(text, is_user=True)
        
        # ОСОБАЯ ЛОГИКА ДЛЯ РЕЖИМА УЧЕНИКА
        if self.is_student_mode and self.auto_selected_subject and not self.lesson_started:
            return self._handle_student_mode_input(text, text_lower)
        
        if self.lesson_started:
            handler = self.dialogue_states.get(self.current_state)
            if handler:
                response = handler(text_lower)
                if response:
                    self._add_to_conversation_history(response, is_user=False)
                    return response
            return None
        
        generated_lesson = self._check_for_lesson_generation_intent(text_lower)
        if generated_lesson:
            return None
        
        available_subjects = self.get_available_subjects()
        for subject in available_subjects:
            if subject.lower() in text_lower and len(subject) > 3:
                print(f"Обнаружен выбор предмета: {subject}")
                return self._handle_subject_selection_direct(subject)
        
        if self.practice_active and self.waiting_for_answer:
            return self._handle_practice_answer(text)
        
        dialogue_response = self._get_dialogue_response(text_lower)
        if dialogue_response:
            final_response = self._add_subject_suggestion(dialogue_response)
            if final_response:
                self._add_to_conversation_history(final_response, is_user=False)
                return final_response
        
        llm_response = self._handle_llm_dialogue(text)
        if llm_response:
            final_response = self._add_subject_suggestion(llm_response)
            if final_response:
                self._add_to_conversation_history(final_response, is_user=False)
                return final_response
        
        fallback_response = self._get_contextual_fallback()
        if fallback_response:
            self._add_to_conversation_history(fallback_response, is_user=False)
            return fallback_response
        
        return None

    def _handle_student_mode_input(self, text: str, text_lower: str) -> Optional[str]:
        """Обработка ввода в режиме ученика"""
        self.student_conversation_count += 1
        print(f"🎓 Режим ученика: счетчик разговора {self.student_conversation_count}, предмет: {self.auto_selected_subject}")
        
        # Проверяем, не хочет ли ученик изучить конкретную тему по выбранному предмету
        if self._check_for_specific_topic_request(text_lower):
            print(f"🎯 Ученик запросил конкретную тему по предмету {self.auto_selected_subject}")
            return None  # Позволяем существующей логике сгенерировать урок
        
        # После 2-3 фраз диалога автоматически предлагаем начать урок
        if self.student_conversation_count >= 2 and not self.student_subject_prompted:
            self.student_subject_prompted = True
            prompt = self._get_student_subject_prompt()
            if prompt:
                self._add_to_conversation_history(prompt, is_user=False)
                return prompt
        
        # Если ученик соглашается начать урок
        if any(word in text_lower for word in ['да', 'ага', 'угу', 'ладно', 'хорошо', 'начать', 'начнем', 'поехали']):
            return self._start_student_lesson()
        
        # Обычная обработка диалога
        dialogue_response = self._get_dialogue_response(text_lower)
        if dialogue_response:
            self._add_to_conversation_history(dialogue_response, is_user=False)
            return dialogue_response
        
        return None

    def _check_for_specific_topic_request(self, text_lower: str) -> bool:
        """Проверяет, запрашивает ли ученик конкретную тему по выбранному предмету"""
        topic_patterns = [
            r'хочу изучить (.+)',
            r'можешь рассказать про (.+)', 
            r'урок по (.+)',
            r'изучим (.+)',
            r'расскажи про (.+)',
            r'хочу узнать про (.+)',
            r'объясни тему (.+)'
        ]
        
        for pattern in topic_patterns:
            match = re.search(pattern, text_lower)
            if match:
                topic = match.group(1).strip()
                if topic and len(topic) > 2:
                    print(f"🎯 Ученик запросил тему '{topic}' по предмету {self.auto_selected_subject}")
                    return True
        return False

    def _get_student_subject_prompt(self) -> str:
        """Возвращает предложение начать урок по выбранному предмету"""
        prompts = [
            f"Отлично! Давайте начнем урок по {self.auto_selected_subject}. Готовы?",
            f"Прекрасно! Приступаем к уроку по {self.auto_selected_subject}. Начинаем?",
            f"Замечательно! Начнем наш урок по {self.auto_selected_subject}?",
            f"Отлично познакомились! Готовы начать урок по {self.auto_selected_subject}?",
            f"Рад нашему знакомству! Приступим к уроку по {self.auto_selected_subject}?"
        ]
        return random.choice(prompts)

    def _start_student_lesson(self) -> str:
        """Начинает урок для ученика по выбранному предмету"""
        print(f"🚀 Начинаем урок для ученика по предмету: {self.auto_selected_subject}")
        
        # Используем существующую логику выбора предмета
        response = self._handle_subject_selection_direct(self.auto_selected_subject)
        
        if response is None:
            # Успешно начали урок
            start_message = f"Отлично! Начинаем урок по {self.auto_selected_subject}. {self._get_next_paragraph()}"
            self._add_to_conversation_history(start_message, is_user=False)
            return start_message
        
        return response

    def _handle_subject_selection_direct(self, subject: str) -> Optional[str]:
        """Прямая обработка выбора предмета"""
        self.current_subject = subject
        lessons = self.lessons.get(subject, [])
        demo_lessons = [l for l in lessons if l.get('is_demo', False)]
        
        if demo_lessons:
            self.selected_lesson = demo_lessons[0]
        elif lessons:
            self.selected_lesson = lessons[0]
        else:
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
        
        self.enable_visualization()
        
        self.conversation_history = []
        self.conversation_context = []
        
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
        
        for subject in subjects:
            if subject.lower() in text.lower():
                return self._handle_subject_selection_direct(subject)
                
        if any(word in text for word in ["назад", "вернуться", "сначала"]):
            self.current_state = "greeting"
            return "Хорошо, начнем сначала. Скажите привет чтобы продолжить."
            
        if any(word in text for word in ["да", "ага", 'угу', "ладно", "хорошо"]):
            prompt = self._get_subject_selection_prompt()
            return prompt if prompt else "Отлично! Какой предмет вас заинтересовал? Назовите его пожалуйста."
            
        return None

    def _handle_lesson_reading(self, text: str) -> Optional[str]:
        if any(word in text for word in ["стоп", "останови", "хватит", "закончи"]):
            self.lesson_started = False
            self.current_state = "greeting"
            self.conversation_counter = 0
            self.knowledge_base = None
            self.conversation_history = []
            self.conversation_context = []
            return "Урок остановлен. Скажите 'привет' когда захотите продолжить или выбрать новый урок."
            
        return None

    def _handle_practice_session(self, text: str) -> Optional[str]:
        if any(word in text for word in ["стоп", "останови", "хватит", "закончи"]):
            self.practice_active = False
            self.waiting_for_answer = False
            self.current_state = "greeting"
            self.conversation_counter = 0
            self.conversation_history = []
            self.conversation_context = []
            
            if self.room_id:
                self.socketio.emit('practice_ended', {'room_id': self.room_id})
            
            return "Практика остановлена. Скажите 'привет' когда захотите продолжить или выбрать новый урок."
            
        if self.waiting_for_answer:
            return self._handle_practice_answer(text)
            
        return None

    def _get_next_paragraph(self) -> Optional[str]:
        print(f"📄 Получение следующего абзаца: текущий {self.current_paragraph}, всего {len(self.lesson_content)}")
        
        if self.current_paragraph < len(self.lesson_content):
            paragraph = self.lesson_content[self.current_paragraph]
            self.current_paragraph += 1
            
            if (self.visualization_enabled and paragraph and 
                len(paragraph.strip()) > 10 and self.room_id):
                
                def delayed_visualization():
                    time.sleep(0.5)
                    context = " ".join(self.lesson_content[max(0, self.current_paragraph-2):self.current_paragraph])
                    self._generate_visualization(paragraph, context)
                
                threading.Thread(target=delayed_visualization, daemon=True).start()
            
            print(f"✅ Возвращаем абзац {self.current_paragraph}: {paragraph[:100]}...")
            return paragraph
        else:
            print("🏁 Урок завершен, запускаем практику")
            practice_message = self._start_practice_session()
            return practice_message

    def _start_practice_session(self) -> str:
        """Запускает фазу практики с асинхронной генерацией вопросов"""
        self.lesson_started = False
        self.current_state = "practice_session"
        self.practice_active = True
        self.waiting_for_answer = False
        self.current_question_index = 0  # СБРАСЫВАЕМ СЧЕТЧИК ВОПРОСОВ
        
        print("=== ЗАПУСК ФАЗЫ ПРАКТИКИ ===")
        print(f"practice_active: {self.practice_active}, waiting_for_answer: {self.waiting_for_answer}")
        
        # 🔥 ОБНОВЛЕНИЕ: Передаем данные ученика в менеджер практики
        if hasattr(self.practice_manager, 'student_data'):
            self.practice_manager.student_data = self.student_data
        
        # Инициализируем менеджер практики с асинхронной генерацией
        lesson_context = " ".join(self.lesson_content)
        self.practice_manager.initialize_practice_generation(lesson_context, self.current_subject)
        
        # Уведомляем клиентов о начале практики
        if self.room_id:
            self.socketio.emit('practice_started', {'room_id': self.room_id})
        
        # ПОЛУЧАЕМ ПЕРВЫЙ ВОПРОС ИЗ ОЧЕРЕДИ
        print("🔄 Получение первого вопроса практики...")
        first_question = self.practice_manager.get_next_question()
        
        if first_question:
            print(f"✅ Первый вопрос получен: {first_question}")
            self.waiting_for_answer = True
            self.current_practice_question = {
                "id": 1,
                "question": first_question,
                "answer": ""
            }
            print(f"📊 Установлен waiting_for_answer: {self.waiting_for_answer}")
            return f"Отлично! Переходим к практике. Первый вопрос: {first_question}"
        else:
            print("❌ Не удалось получить первый вопрос практики")
            self.practice_active = False
            return "Практические задания временно недоступны. Давайте продолжим урок или выберем другую тему."

    def _evaluate_and_generate_next(self, student_answer: str) -> str:
        """Оценивает ответ и возвращает следующий вопрос с асинхронной генерацией"""
        print(f"🔍 Обработка ответа: '{student_answer}'")
        print(f"📊 Состояние: practice_active={self.practice_active}, waiting_for_answer={self.waiting_for_answer}")
        
        if not self.practice_active:
            print("❌ Практика не активна")
            return "Практика не активна."
        
        # ПРОВЕРЯЕМ, НЕ ЯВЛЯЕТСЯ ЛИ ОТВЕТ КОМАНДОЙ
        if any(cmd in student_answer.lower() for cmd in ['продолжай', 'дальше', 'следующий']):
            print(f"🔇 Игнорирую команду вместо ответа: {student_answer}")
            next_question = self.practice_manager.get_next_question()
            if next_question:
                return f"Это похоже на команду. Пожалуйста, дайте ответ на вопрос. Следующий вопрос: {next_question}"
            else:
                self._end_practice_session()
                return "Практика завершена."
        
        print(f"🎯 Оценка ответа и получение следующего вопроса...")
        
        current_question = self.current_practice_question
        if not current_question:
            print("❌ Нет текущего вопроса практики")
            self._end_practice_session()
            return "Практика завершена."
        
        # УВЕЛИЧИВАЕМ СЧЕТЧИК ОТВЕТОВ - ВАЖНОЕ ИЗМЕНЕНИЕ
        self.current_question_index += 1
        print(f"📊 Текущий номер вопроса: {self.current_question_index}/{self.max_questions}")
        
        # ПРОВЕРЯЕМ ЛИМИТ ВОПРОСОВ - ВАЖНОЕ ИЗМЕНЕНИЕ
        if self.current_question_index >= self.max_questions:
            print(f"🏁 Достигнут лимит вопросов: {self.current_question_index}/{self.max_questions}")
            self._end_practice_session()
            return "Отлично! Вы ответили на все вопросы практики. Урок завершен!"
        
        # ИСПОЛЬЗУЕМ НОВЫЙ МЕТОД: оценка + следующий вопрос
        feedback, next_question = self.practice_manager.evaluate_and_continue(
            student_answer, 
            current_question["question"]
        )
        
        # УЛУЧШЕННЫЙ FALLBACK ДЛЯ ПРАКТИКИ
        if not feedback or "Хороший вопрос! Давайте разберем эту тему подробнее" in feedback:
            feedback = "Спасибо за ответ! Переходим к следующему вопросу."
        
        if next_question:
            # Обновляем текущий вопрос
            self.current_practice_question = {
                "id": self.current_question_index + 1,
                "question": next_question,
                "answer": ""
            }
            self.waiting_for_answer = True
            
            response = f"{feedback}. Следующий вопрос: {next_question}"
            print(f"➡️ Следующий вопрос получен: {next_question[:80]}...")
            print(f"📊 Вопросов задано: {self.current_question_index}/{self.max_questions}")
            print(f"📊 Установлен waiting_for_answer: {self.waiting_for_answer}")
            return response
        else:
            print("❌ Не удалось получить следующий вопрос")
            self._end_practice_session()
            return f"{feedback}. Практика завершена!"

    def _handle_practice_answer(self, text: str) -> str:
        """Обработка ответа ученика во время практики"""
        return self._evaluate_and_generate_next(text)

    def _end_practice_session(self):
        """Завершает сессию практики"""
        self.practice_active = False
        self.waiting_for_answer = False
        self.current_state = "greeting"
        self.current_question_index = 0  # СБРАСЫВАЕМ СЧЕТЧИК
        self.practice_manager.stop_async_generation()
        
        self.lesson_started = False
        self.selected_lesson = None
        self.current_subject = None
        self.lesson_content = []
        self.current_paragraph = 0
        
        if self.room_id:
            self.socketio.emit('practice_ended', {'room_id': self.room_id})
        print("=== 🏁 ПРАКТИКА ЗАВЕРШЕНА ===")

    def handle_question_during_lesson(self, question: str) -> str:
        """Обработка вопросов ученика во время урока"""
        if not question.strip():
            return "Повторите вопрос пожалуйста, я не расслышал."
            
        question_lower = question.lower().strip()
        
        if self.visualization_enabled:
            context = " ".join(self.lesson_content[max(0, self.current_paragraph-2):self.current_paragraph])
            self._generate_visualization(question, context)
        
        print(f"Немедленная обработка вопроса: '{question}'")
        final_response = None
        
        if self.llm_query_mode == "llm_first":
            print(f"🔀 Режим llm_first: Обработка вопроса '{question}'")
            
            current_context = ""
            if self.lesson_content and self.current_paragraph > 0:
                context_start = max(0, self.current_paragraph - 2)
                current_context = " ".join(self.lesson_content[context_start:self.current_paragraph])
            
            llm_response = self.llm.query(question, current_context, self.current_subject)
            if llm_response and not llm_response.startswith("Интересный вопрос!"):
                self.llm.add_to_cache(question, llm_response, self.current_subject)
                if self.knowledge_base and self._should_save_to_knowledge_base(question):
                    self.knowledge_base.add_llm_answer(question, llm_response)
                    self.knowledge_base.add_knowledge(question=question, answer=llm_response)
                    self.knowledge_base.add_to_dialogue_knowledge(question, llm_response)
                print(f"✅ Ответ получен от LLM (режим llm_first): {llm_response[:100]}...")
                final_response = llm_response
            
            if not final_response and self.knowledge_base:
                knowledge_response = self.knowledge_base.get_dialogue_response(question_lower)
                if knowledge_response and not knowledge_response.startswith("Интересный вопрос!"):
                    print(f"📚 Ответ найден в базе знаний после неудачи LLM: {knowledge_response[:100]}...")
                    final_response = knowledge_response
            
            if not final_response and self.knowledge_base:
                llm_answer = self.knowledge_base.find_llm_answer(question, threshold=0.8)
                if llm_answer:
                    print(f"💾 Использован сохраненный ответ LLM: {llm_answer[:100]}...")
                    final_response = llm_answer
        
        else:
            print(f"🔀 Режим traditional: Обработка вопроса '{question}'")
            
            if self.knowledge_base:
                knowledge_response = self.knowledge_base.get_dialogue_response(question_lower)
                if knowledge_response and not knowledge_response.startswith("Интересный вопрос!"):
                    final_response = knowledge_response
            
            if not final_response:
                for pattern, responses in self.local_patterns.items():
                    if pattern in question_lower:
                        final_response = random.choice(responses)
                        break
            
            if not final_response and self.knowledge_base:
                dialogue_response = self.knowledge_base.get_dialogue_response(question_lower)
                if dialogue_response:
                    final_response = dialogue_response
            
            if not final_response and self.knowledge_base:
                answer = self.knowledge_base.find_answer(question, threshold=0.5)
                if answer and not answer.startswith("Интересный вопрос!"):
                    final_response = answer
            
            if not final_response and self.knowledge_base:
                llm_answer = self.knowledge_base.find_llm_answer(question, threshold=0.8)
                if llm_answer:
                    print(f"💾 Использован сохраненный ответ LLM для вопроса: {question}")
                    final_response = llm_answer
            
            if not final_response:
                current_context = ""
                if self.lesson_content and self.current_paragraph > 0:
                    context_start = max(0, self.current_paragraph - 2)
                    current_context = " ".join(self.lesson_content[context_start:self.current_paragraph])
                
                llm_response = self.llm.query(question, current_context, self.current_subject)
                if llm_response:
                    self.llm.add_to_cache(question, llm_response, self.current_subject)
                    if self.knowledge_base and self._should_save_to_knowledge_base(question):
                        self.knowledge_base.add_llm_answer(question, llm_response)
                        self.knowledge_base.add_knowledge(question=question, answer=llm_response)
                        self.knowledge_base.add_to_dialogue_knowledge(question, llm_response)
                    final_response = llm_response
        
        if not final_response:
            final_response = "Интересный вопрос! Давайте обсудим его после завершения текущего материала, чтобы не отвлекаться."
        
        return final_response

    def get_selected_lesson(self) -> Optional[dict]:
        return self.selected_lesson

    def is_lesson_started(self) -> bool:
        return self.lesson_started

    def get_current_subject(self) -> Optional[str]:
        return self.current_subject

    def get_current_state(self) -> str:
        return self.current_state

    def reset(self):
        """Полный сброс диалог менеджера"""
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
        self.practice_manager.reset()
        
        # Сброс режима ученика
        self.is_student_mode = False
        self.auto_selected_subject = None
        self.student_conversation_count = 0
        self.student_lesson_started = False
        self.student_subject_prompted = False

    def get_available_subjects(self) -> List[str]:
        subjects = list(self.lessons.keys())
        if "обществознание" not in subjects:
            subjects.append("обществознание")
        return subjects

    def get_lessons_for_subject(self, subject: str) -> List[dict]:
        return self.lessons.get(subject, [])

    def set_llm_model(self, model: str):
        self.llm.set_model(model)
        print(f"Установлена модель LLM: {model}")

    def set_llm_mode(self, mode: str):
        if mode in ["traditional", "llm_first"]:
            self.llm_query_mode = mode
            print(f"Установлен режим LLM: {mode}")

    def get_knowledge_stats(self) -> Optional[Dict]:
        if self.knowledge_base:
            return self.knowledge_base.get_stats()
        return None

    def set_room_id(self, room_id: str):
        """Установка ID комнаты для WebSocket коммуникации"""
        self.room_id = room_id
        print(f"🔧 Установлен room_id для DialogueManager: {room_id}")

    def set_student_mode(self, subject: str):
        """Устанавливает режим ученика с автоматическим выбором предмета"""
        self.is_student_mode = True
        self.auto_selected_subject = subject
        self.student_conversation_count = 0
        self.student_lesson_started = False
        self.student_subject_prompted = False
        print(f"🎓 Установлен режим ученика с предметом: {subject}")

    def get_practice_status(self) -> Dict:
        """Возвращает статус практики"""
        return {
            "practice_active": self.practice_active,
            "waiting_for_answer": self.waiting_for_answer,
            "current_question": self.current_practice_question,
            "question_index": self.current_question_index,
            "max_questions": self.max_questions,
            "questions_asked": len(self.practice_manager.generated_questions) if hasattr(self.practice_manager, 'generated_questions') else 0
        }

    def force_start_practice(self, lesson_context: str, subject: str) -> str:
        """Принудительно запускает практику (для тестирования)"""
        try:
            self.lesson_started = False
            self.current_state = "practice_session"
            self.practice_active = True
            self.waiting_for_answer = False
            self.current_question_index = 0
            
            print("=== ПРИНУДИТЕЛЬНЫЙ ЗАПУСК ПРАКТИКИ ===")
            
            # Инициализируем менеджер практики
            self.practice_manager.initialize_practice_generation(lesson_context, subject)
            
            # Получаем первый вопрос
            first_question = self.practice_manager.get_next_question()
            
            if first_question:
                self.waiting_for_answer = True
                self.current_practice_question = {
                    "id": 1,
                    "question": first_question,
                    "answer": ""
                }
                return f"Практика запущена. Первый вопрос: {first_question}"
            else:
                self.practice_active = False
                return "Не удалось запустить практику"
                
        except Exception as e:
            print(f"❌ Ошибка принудительного запуска практики: {e}")
            return f"Ошибка запуска практики: {e}"

    def skip_to_practice(self):
        """Пропускает урок и сразу переходит к практике (для тестирования)"""
        if not self.lesson_started or not self.lesson_content:
            return "Сначала нужно начать урок"
        
        print("=== ПРОПУСК К ПРАКТИКЕ ===")
        practice_message = self._start_practice_session()
        return practice_message

    def get_visualization_status(self) -> Dict:
        """Возвращает статус инфографики"""
        return {
            "visualization_enabled": self.visualization_enabled,
            "visualization_counter": self.visualization_counter,
            "last_visualization_time": self.last_visualization_time,
            "paragraphs_since_last_viz": self.paragraphs_since_last_viz
        }

    def force_visualization(self, text: str) -> bool:
        """Принудительно генерирует инфографику для текста"""
        try:
            if not self.room_id:
                print("❌ Нет room_id для отправки инфографики")
                return False
            
            context = " ".join(self.lesson_content[max(0, self.current_paragraph-2):self.current_paragraph]) if self.lesson_content else ""
            self._generate_visualization(text, context)
            return True
            
        except Exception as e:
            print(f"❌ Ошибка принудительной генерации инфографики: {e}")
            return False

    def get_conversation_stats(self) -> Dict:
        """Возвращает статистику диалога"""
        user_messages = [msg for msg in self.conversation_history if msg['is_user']]
        teacher_messages = [msg for msg in self.conversation_history if not msg['is_user']]
        
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": len(user_messages),
            "teacher_messages": len(teacher_messages),
            "current_state": self.current_state,
            "lesson_started": self.lesson_started,
            "current_paragraph": self.current_paragraph,
            "total_paragraphs": len(self.lesson_content),
            "conversation_context": self.conversation_context
        }

    def debug_info(self) -> Dict:
        """Возвращает отладочную информацию"""
        practice_stats = self.practice_manager.get_practice_stats() if hasattr(self.practice_manager, 'get_practice_stats') else {}
        
        return {
            "current_state": self.current_state,
            "current_subject": self.current_subject,
            "lesson_started": self.lesson_started,
            "practice_active": self.practice_active,
            "waiting_for_answer": self.waiting_for_answer,
            "current_paragraph": self.current_paragraph,
            "total_paragraphs": len(self.lesson_content),
            "available_subjects": self.get_available_subjects(),
            "llm_mode": self.llm_query_mode,
            "visualization_enabled": self.visualization_enabled,
            "conversation_history_length": len(self.conversation_history),
            "practice_stats": practice_stats,
            "current_practice_question": self.current_practice_question,
            "room_id": self.room_id,
            "questions_asked": len(self.practice_manager.generated_questions) if hasattr(self.practice_manager, 'generated_questions') else 0,
            "max_questions": self.max_questions,
            # Информация о режиме ученика
            "is_student_mode": self.is_student_mode,
            "auto_selected_subject": self.auto_selected_subject,
            "student_conversation_count": self.student_conversation_count,
            "student_lesson_started": self.student_lesson_started,
            "student_subject_prompted": self.student_subject_prompted
        }

    def add_custom_lesson(self, subject: str, title: str, content: str) -> bool:
        """Добавляет пользовательский урок"""
        try:
            # Создаем имя файла
            filename = f"{subject}_{title.lower().replace(' ', '_')}.txt"
            lesson_path = self.lessons_dir / filename
            
            # Записываем контент
            with open(lesson_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Добавляем в список уроков
            lesson_data = {
                'id': filename.replace('.txt', ''),
                'title': title,
                'file_path': lesson_path,
                'type': 'text',
                'is_custom': True
            }
            
            if subject not in self.lessons:
                self.lessons[subject] = []
            
            self.lessons[subject].append(lesson_data)
            
            print(f"✅ Пользовательский урок добавлен: {title} ({subject})")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка добавления пользовательского урока: {e}")
            return False

    def list_all_lessons(self) -> Dict[str, List[Dict]]:
        """Возвращает все доступные уроки по предметам"""
        return self.lessons

    def get_lesson_progress(self) -> Dict:
        """Возвращает прогресс по текущему уроку"""
        if not self.lesson_started or not self.lesson_content:
            return {"error": "Урок не начат"}
        
        progress_percent = (self.current_paragraph / len(self.lesson_content)) * 100 if self.lesson_content else 0
        
        return {
            "current_paragraph": self.current_paragraph,
            "total_paragraphs": len(self.lesson_content),
            "progress_percent": round(progress_percent, 1),
            "remaining_paragraphs": len(self.lesson_content) - self.current_paragraph,
            "lesson_title": self.selected_lesson['title'] if self.selected_lesson else "Неизвестно",
            "subject": self.current_subject
        }

    def continue_lesson(self) -> Optional[str]:
        """Продолжает урок с текущей позиции"""
        if not self.lesson_started:
            return "Урок не начат. Скажите 'привет' чтобы начать."
        
        return self._get_next_paragraph()

    def restart_lesson(self) -> str:
        """Перезапускает текущий урок"""
        if not self.selected_lesson:
            return "Нет активного урока для перезапуска."
        
        try:
            self.current_paragraph = 0
            self.lesson_content = self._load_lesson_content(self.selected_lesson['file_path'])
            
            if not self.lesson_content:
                return "Ошибка загрузки урока."
            
            first_paragraph = self.lesson_content[0] if self.lesson_content else ""
            return f"Урок перезапущен. {first_paragraph}"
            
        except Exception as e:
            print(f"❌ Ошибка перезапуска урока: {e}")
            return "Ошибка перезапуска урока."

    def set_max_practice_questions(self, max_questions: int):
        """Устанавливает максимальное количество вопросов в практике"""
        if 1 <= max_questions <= 20:
            self.max_questions = max_questions
            self.practice_manager.max_questions = max_questions
            print(f"🔧 Максимальное количество вопросов установлено: {max_questions}")
        else:
            print("❌ Некорректное количество вопросов. Должно быть от 1 до 20.")

    def get_system_status(self) -> Dict:
        """Возвращает общий статус системы"""
        llm_status = self.llm.get_llm_status() if hasattr(self.llm, 'get_llm_status') else {}
        knowledge_stats = self.get_knowledge_stats() or {}
        practice_stats = self.practice_manager.get_practice_stats() if hasattr(self.practice_manager, 'get_practice_stats') else {}
        
        return {
            "dialogue_manager": {
                "current_state": self.current_state,
                "lesson_started": self.lesson_started,
                "practice_active": self.practice_active,
                "current_subject": self.current_subject,
                "conversation_history_length": len(self.conversation_history),
                "questions_asked": len(self.practice_manager.generated_questions) if hasattr(self.practice_manager, 'generated_questions') else 0,
                "max_questions": self.max_questions,
                "is_student_mode": self.is_student_mode,
                "auto_selected_subject": self.auto_selected_subject,
                "student_conversation_count": self.student_conversation_count
            },
            "llm": llm_status,
            "knowledge_base": knowledge_stats,
            "practice": practice_stats,
            "visualization": self.get_visualization_status(),
            "lessons": {
                "available_subjects": self.get_available_subjects(),
                "total_lessons": sum(len(lessons) for lessons in self.lessons.values())
            }
        }

    def export_conversation_history(self) -> List[Dict]:
        """Экспортирует историю диалога"""
        return self.conversation_history.copy()

    def clear_conversation_history(self):
        """Очищает историю диалога"""
        self.conversation_history = []
        self.conversation_context = []
        print("🗑️ История диалога очищена")

    def simulate_student_answer(self, answer: str) -> str:
        """Симулирует ответ студента (для тестирования)"""
        if not self.practice_active or not self.waiting_for_answer:
            return "Практика не активна или система не ожидает ответа"
        
        return self._evaluate_and_generate_next(answer)

    def get_available_commands(self) -> Dict[str, str]:
        """Возвращает список доступных команд"""
        return {
            "привет": "Начать диалог",
            "продолжай": "Продолжить урок",
            "стоп": "Остановить урок/практику",
            "какой предмет": "Показать доступные предметы",
            "практика": "Перейти к практике (если урок завершен)",
            "статус": "Показать статус системы",
            "помощь": "Показать эту справку"
        }


# Создаем глобальный экземпляр для тестирования
if __name__ == "__main__":
    # Тестирование базовой функциональности
    dm = DialogueManager(None)
    
    print("🧪 Тестирование DialogueManager...")
    
    # Тест доступных предметов
    subjects = dm.get_available_subjects()
    print(f"📚 Доступные предметы: {subjects}")
    
    # Тест обработки приветствия
    response = dm.process_input("привет")
    print(f"👋 Ответ на приветствие: {response}")
    
    # Тест статуса системы
    status = dm.get_system_status()
    print(f"📊 Статус системы: {status}")
    
    print("✅ Тестирование завершено!")