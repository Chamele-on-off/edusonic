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
        
        # 🔥 НОВАЯ СТРУКТУРА ПАПОК ПО КЛАССАМ
        self.lessons_dir = Path("lessons")
        self.demo_lessons_dir = self.lessons_dir / "demo"
        self.students_base_dir = self.lessons_dir / "students"
        self.generated_lessons_dir = self.lessons_dir / "generated"
        
        # Создаем папки если их нет
        for folder in [self.demo_lessons_dir, self.students_base_dir, self.generated_lessons_dir]:
            folder.mkdir(parents=True, exist_ok=True)
        
        # 🔥 НОВЫЕ ПОЛЯ ДЛЯ ПРОГРЕССА УЧЕНИКА
        self.student_progress = {}  # {"subject": {"completed_lessons": [], "current_lesson": id, "total_lessons": X}}
        self.last_progress_save = 0
        self.progress_dir = Path("students_progress")
        self.progress_dir.mkdir(exist_ok=True)
        
        # 🔥 ПРОСТЫЕ ДАННЫЕ УЧЕНИКА
        self.student_data = {}
        self.has_student_data = False
        
        # 🔥 НОВЫЕ ПОЛЯ ДЛЯ СТРУКТУРЫ ПО КЛАССАМ
        self.lessons = {}  # Все уроки по предметам
        self.lessons_by_class = {}  # Уроки по классам {"5": {"математика": [lessons...], ...}}
        self.available_classes = ["5", "6", "7", "8", "9", "10", "11"]  # Поддерживаемые классы
        
        # НОВЫЕ ПОЛЯ ДЛЯ ВИЗУАЛИЗАЦИИ - ТОЛЬКО SVG
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
        """🔥 ОБНОВЛЕННАЯ ЗАГРУЗКА УРОКОВ ПО КЛАССАМ"""
        self.lessons = {}
        self.lessons_by_class = {}
        
        try:
            # 1. Загружаем демо-уроки
            self._load_lessons_from_dir(self.demo_lessons_dir, "demo")
            
            # 2. Загружаем уроки для учеников ПО КЛАССАМ
            if self.students_base_dir.exists():
                for class_dir in self.students_base_dir.iterdir():
                    if class_dir.is_dir() and "_class" in class_dir.name:
                        class_level = class_dir.name.replace("_class", "")
                        if class_level in self.available_classes:
                            self._load_student_lessons_by_class(class_dir, class_level)
            
            # 3. Загружаем сгенерированные уроки
            self._load_lessons_from_dir(self.generated_lessons_dir, "generated")
            
            # 4. 🔥 ИСПРАВЛЕННАЯ ЗАГРУЗКА LEGACY УРОКОВ (рекурсивно)
            self._load_legacy_lessons()
            
            print(f"✅ Уроки загружены: {sum(len(v) for v in self.lessons.values())} уроков")
            print(f"✅ Классы с уроками: {list(self.lessons_by_class.keys())}")
                    
        except Exception as e:
            print(f"Ошибка доступа к папке уроков: {e}")

    def _load_legacy_lessons(self):
        """Исправленная загрузка старых уроков (рекурсивный поиск)"""
        try:
            # 🔥 РЕКУРСИВНЫЙ ПОИСК ВСЕХ TXT ФАЙЛОВ В lessons/
            for lesson_file in self.lessons_dir.glob("**/*.txt"):
                if not lesson_file.is_file():
                    continue
                    
                # Пропускаем уже загруженные из других папок
                if (lesson_file.parent == self.demo_lessons_dir or 
                    lesson_file.parent == self.generated_lessons_dir or
                    "students" in str(lesson_file.parent)):
                    continue
                
                print(f"📂 Загрузка legacy урока: {lesson_file}")
                
                try:
                    subject = self._detect_subject(lesson_file.stem)
                    lesson_number = self._extract_lesson_number(lesson_file.stem)
                    lesson_title = self._format_lesson_title(lesson_file.stem)
                    
                    # Определяем класс из пути (если есть)
                    class_level = "general"
                    if "_class" in str(lesson_file):
                        for part in lesson_file.parts:
                            if part.endswith("_class"):
                                class_level = part.replace("_class", "")
                                break
                    
                    lesson_data = {
                        'id': f"legacy_{lesson_file.stem}",
                        'title': lesson_title,
                        'file_path': lesson_file,
                        'type': 'legacy',
                        'subject': subject,
                        'class_level': class_level,
                        'lesson_number': lesson_number,
                        'full_path': str(lesson_file.relative_to(self.lessons_dir))
                    }
                    
                    if subject not in self.lessons:
                        self.lessons[subject] = []
                    self.lessons[subject].append(lesson_data)
                    
                    # Добавляем в структуру по классам если класс определен
                    if class_level != "general":
                        if class_level not in self.lessons_by_class:
                            self.lessons_by_class[class_level] = {}
                        if subject not in self.lessons_by_class[class_level]:
                            self.lessons_by_class[class_level][subject] = []
                        self.lessons_by_class[class_level][subject].append(lesson_data)
                        
                except Exception as e:
                    print(f"Ошибка загрузки legacy урока {lesson_file}: {e}")
                    
        except Exception as e:
            print(f"Ошибка при поиске legacy уроков: {e}")

    def _load_student_lessons_by_class(self, class_dir: Path, class_level: str):
        """Загружает уроки для конкретного класса"""
        if class_level not in self.lessons_by_class:
            self.lessons_by_class[class_level] = {}
        
        print(f"📂 Загрузка уроков для класса {class_level}...")
        
        # Предметы по классам
        subjects_by_class = {
            "5": ["математика", "география", "биология", "русский", "литература", 
                  "английский", "французский", "история", "информатика"],
            "6": ["математика", "география", "биология", "русский", "литература",
                  "английский", "французский", "история", "обществознание", "информатика"],
            "7": ["алгебра", "геометрия", "физика", "география", "биология", "русский",
                  "литература", "математика", "английский", "французский", "история",
                  "обществознание", "информатика"],
            "8": ["алгебра", "геометрия", "физика", "география", "биология", "русский",
                  "литература", "математика", "английский", "французский", "история",
                  "обществознание", "информатика", "химия"],
            "9": ["алгебра", "геометрия", "физика", "география", "биология", "русский",
                  "литература", "математика", "английский", "французский", "история",
                  "обществознание", "информатика", "химия"],
            "10": ["алгебра", "геометрия", "физика", "география", "биология", "русский",
                   "литература", "математика", "английский", "французский", "история",
                   "обществознание", "информатика", "химия"],
            "11": ["алгебра", "геометрия", "физика", "география", "биология", "русский",
                   "литература", "математика", "английский", "французский", "история",
                   "обществознание", "информатика", "химия"]
        }
        
        subjects = subjects_by_class.get(class_level, [])
        for subject in subjects:
            subject_dir = class_dir / subject
            if subject_dir.exists() and subject_dir.is_dir():
                self._load_lessons_from_subject_dir(subject_dir, subject, class_level, "student")
        
        print(f"✅ Класс {class_level}: {sum(len(v) for v in self.lessons_by_class[class_level].values())} уроков")

    def _load_lessons_from_subject_dir(self, subject_dir: Path, subject: str, class_level: str, lesson_type: str):
        """Загружает уроки из папки предмета"""
        for lesson_file in subject_dir.glob("*.txt"):
            try:
                # Извлекаем номер урока из имени файла
                lesson_number = self._extract_lesson_number(lesson_file.stem)
                lesson_title = self._format_lesson_title(lesson_file.stem)
                
                # Создаем уникальный ID урока
                lesson_id = f"{class_level}_{subject}_{lesson_file.stem}"
                
                lesson_data = {
                    'id': lesson_id,
                    'title': lesson_title,
                    'file_path': lesson_file,
                    'type': lesson_type,
                    'subject': subject,
                    'class_level': class_level,
                    'lesson_number': lesson_number,
                    'full_path': f"{class_level}_class/{subject}/{lesson_file.name}"
                }
                
                # Добавляем в общий список по предметам
                if subject not in self.lessons:
                    self.lessons[subject] = []
                self.lessons[subject].append(lesson_data)
                
                # Добавляем в список по классам
                if subject not in self.lessons_by_class[class_level]:
                    self.lessons_by_class[class_level][subject] = []
                self.lessons_by_class[class_level][subject].append(lesson_data)
                
            except Exception as e:
                print(f"Ошибка загрузки урока {lesson_file}: {e}")

    def _load_lessons_from_dir(self, dir_path: Path, lesson_type: str):
        """Загружает уроки из директории (для демо, сгенерированных)"""
        if not dir_path.exists():
            return
        
        for lesson_file in dir_path.glob("*.txt"):
            try:
                subject = self._detect_subject(lesson_file.stem)
                lesson_number = self._extract_lesson_number(lesson_file.stem)
                lesson_title = self._format_lesson_title(lesson_file.stem)
                
                lesson_data = {
                    'id': f"{lesson_type}_{lesson_file.stem}",
                    'title': lesson_title,
                    'file_path': lesson_file,
                    'type': lesson_type,
                    'subject': subject,
                    'class_level': "general",  # Общие уроки
                    'lesson_number': lesson_number,
                    'full_path': f"{dir_path.name}/{lesson_file.name}"
                }
                
                if subject not in self.lessons:
                    self.lessons[subject] = []
                self.lessons[subject].append(lesson_data)
                
            except Exception as e:
                print(f"Ошибка загрузки урока {lesson_file}: {e}")

    def _extract_lesson_number(self, filename: str) -> int:
        """Извлекает номер урока из имени файла"""
        # Форматы: lesson_1_algebra.txt, урок_1.txt, 1_алгебра.txt
        match = re.search(r'(?:lesson|урок)[_\s]*(\d+)', filename.lower())
        if match:
            return int(match.group(1))
        
        # Пробуем найти число в начале
        match = re.search(r'^(\d+)', filename)
        if match:
            return int(match.group(1))
        
        return 999  # По умолчанию

    def _format_lesson_title(self, filename: str) -> str:
        """Форматирует название урока"""
        # Удаляем расширение и разделители
        name = filename.replace('.txt', '')
        
        # Преобразуем lesson_1_algebra → Урок 1: Алгебра
        if name.startswith('lesson_'):
            parts = name[7:].split('_', 1)
            if len(parts) == 2 and parts[0].isdigit():
                return f"Урок {parts[0]}: {parts[1].replace('_', ' ').title()}"
        
        # Преобразуем урок_1_алгебра → Урок 1: Алгебра
        if name.startswith('урок_'):
            parts = name[5:].split('_', 1)
            if len(parts) == 2 and parts[0].isdigit():
                return f"Урок {parts[0]}: {parts[1].replace('_', ' ').title()}"
        
        # Просто делаем читабельным
        return name.replace('_', ' ').title()

    def _detect_subject(self, filename: str) -> str:
        """Определяет предмет по названию файла"""
        filename_lower = filename.lower()
        
        # Карта английских названий предметов
        subject_map = {
            'math': 'математика',
            'mathematics': 'математика',
            'algebra': 'алгебра',
            'geometry': 'геометрия',
            'physics': 'физика', 
            'chemistry': 'химия',
            'biology': 'биология',
            'history': 'история',
            'social': 'обществознание',
            'literature': 'литература',
            'russian': 'русский язык',
            'english': 'английский язык',
            'french': 'французский язык',
            'geography': 'география',
            'informatics': 'информатика'
        }
        
        for eng, rus in subject_map.items():
            if eng in filename_lower:
                return rus
        
        # Дополнительные проверки для русского языка
        if any(word in filename_lower for word in ['математика', 'алгебра', 'геометрия']):
            return "математика" if 'алгебра' not in filename_lower and 'геометрия' not in filename_lower else "алгебра" if 'алгебра' in filename_lower else "геометрия"
        elif any(word in filename_lower for word in ['физика', 'физ']):
            return "физика"
        elif any(word in filename_lower for word in ['химия', 'хим']):
            return "химия"
        elif any(word in filename_lower for word in ['история', 'истор']):
            return "история"
        elif any(word in filename_lower for word in ['обществознание', 'общество']):
            return "обществознание"
        elif any(word in filename_lower for word in ['биология', 'био']):
            return "биология"
        elif any(word in filename_lower for word in ['литература', 'лит']):
            return "литература"
        elif any(word in filename_lower for word in ['русский', 'язык']):
            return "русский язык"
        elif any(word in filename_lower for word in ['английский']):
            return "английский язык"
        elif any(word in filename_lower for word in ['французский']):
            return "французский язык"
        elif any(word in filename_lower for word in ['география']):
            return "география"
        elif any(word in filename_lower for word in ['информатика']):
            return "информатика"
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

    def _add_personalization(self, response: str) -> str:
        """Добавляет персонализацию (имя ученика) к ответам"""
        if not self.has_student_data or not response:
            return response
        
        student_name = self.student_data.get('name', '').strip()
        if not student_name:
            return response
        
        # 🔥 ПЕРСОНАЛИЗИРУЕМ ТОЛЬКО ПЕРВОЕ ПРЕДЛОЖЕНИЕ В ОТВЕТЕ
        sentences = re.split(r'(?<=[.!?])\s+', response)
        if sentences:
            first_sentence = sentences[0]
            
            # Не персонализируем если уже есть обращение
            if student_name.lower() not in first_sentence.lower():
                # Добавляем имя в начале первого предложения
                personalized_first = f"{student_name}, {first_sentence[0].lower() + first_sentence[1:]}"
                sentences[0] = personalized_first
                
                return ' '.join(sentences)
        
        return response

    def _handle_llm_dialogue(self, text: str, room_id: str = None) -> Optional[str]:
        """Гарантированная обработка диалога через LLM с ПЕРСОНАЛИЗАЦИЕЙ"""
        try:
            # Собираем контекст диалога
            context = self._get_conversation_context()
            
            # 🔥 УЛУЧШЕННЫЙ ПРОМПТ С ДАННЫМИ УЧЕНИКА
            age = self.student_data.get('age', '12')
            level = self.student_data.get('education_level', '5')
            name = self.student_data.get('name', 'ученик')
            
            # ВСЕГДА передаем данные ученика в LLM
            system_prompt = f"""Ты - дружелюбный учитель для ученика {age} лет, {level} класс.

ОСОБЕННОСТИ УЧЕНИКА:
- Имя: {name}
- Возраст: {age} лет  
- Уровень: {level} класс
- Предмет: {self.current_subject or 'не выбран'}

СТИЛЬ ОБЩЕНИЯ:
- ОБРАЩАЙСЯ К УЧЕНИКУ ПО ИМЕНИ "{name}"
- Используй язык, понятный для {age}-летнего
- Будь поддерживающим и терпеливым
- Объясняй сложные вещи простыми словами
- Используй примеры, релевантные для этого возраста

ОТВЕТЫ ДОЛЖНЫ БЫТЬ:
- Краткими (2-3 предложения максимум)
- Понятными для {age}-летнего
- Конкретными и полезными
- На русском языке
- ОБРАЩАЙСЯ К УЧЕНИКУ ПО ИМЕНИ

Контекст разговора: {context}"""

            # АСИНХРОННЫЙ запрос к локальной модели
            if room_id and self.socketio:
                # Используем асинхронный режим с callback
                def llm_callback(response, r_id):
                    if response:
                        limited_response = self._limit_response_length(
                            response, 
                            self.dialogue_settings.get("max_response_length", 3)
                        )
                        
                        # 🔥 ПЕРСОНАЛИЗИРУЕМ ОТВЕТ ОТ LLM
                        personalized_response = self._add_personalization(limited_response)
                        
                        # Отправляем ответ через WebSocket
                        self.socketio.emit('llm_dialogue_response', {
                            'room_id': r_id,
                            'response': personalized_response,
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
                    # 🔥 ПЕРСОНАЛИЗИРУЕМ ОТВЕТ
                    return self._add_personalization(limited_response)
                    
        except Exception as e:
            print(f"Ошибка запроса к LLM для диалога: {e}")
        
        return self._get_subject_selection_prompt()

    def _get_subject_selection_prompt(self) -> Optional[str]:
        """Возвращает предложение выбора предмета с учетом кд"""
        current_time = time.time()
        if current_time - self.last_subject_prompt_time < self.subject_prompt_cooldown:
            return None
        
        self.last_subject_prompt_time = current_time
        
        # 🔥 НЕ предлагаем выбор предмета если уже есть предмет
        if self.current_subject and self.has_student_data:
            return None  # Уже есть предмет, не предлагаем выбор
        
        # 🔥 ДЛЯ УЧЕНИКА: показываем предметы его класса
        if self.has_student_data and self.student_data.get('education_level'):
            student_class = self.student_data['education_level']
            if student_class in self.lessons_by_class:
                subjects = list(self.lessons_by_class[student_class].keys())
                if subjects:
                    subject_list = ", ".join([s.capitalize() for s in subjects[:4]])
                    if len(subjects) > 4:
                        subject_list += " и другие"
                    
                    prompt_template = random.choice(self.subject_prompt_variants)
                    return prompt_template.format(subjects=subject_list)
        
        # Для обычных пользователей
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
        
        # 🔥 НЕ предлагаем выбор предмета если уже есть предмет
        if self.current_subject and self.has_student_data:
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
            # 🔥 ПЕРСОНАЛИЗИРОВАННОЕ ПРИВЕТСТВИЕ ДЛЯ УЧЕНИКА
            if self.has_student_data and self.current_subject:
                student_name = self.student_data.get('name', 'ученик')
                subject = self.current_subject
                
                # Если есть предмет - предлагаем начать урок по нему
                return f"Привет, {student_name}! Я вижу, ты хочешь изучать {subject}. Готов начать урок? Скажи 'готов' чтобы начать!"
            
            if self.has_student_data:
                student_name = self.student_data.get('name', 'ученик')
                # 🔥 ПЕРСОНАЛИЗИРОВАННОЕ ПРИВЕТСТВИЕ ДЛЯ УЧЕНИКА
                age = self.student_data.get('age', '12')
                level = self.student_data.get('education_level', '5')
                
                greeting_variants = [
                    f"Привет, {student_name}! Я твой виртуальный учитель. Рад видеть тебя! Ты в {level} классе, это отлично!",
                    f"Здравствуй, {student_name}! Я твой AI-репетитор. Готов помочь тебе с учебой в {level} классе!",
                    f"Привет, {student_name}! Я твой цифровой преподаватель. Рад видеть ученика {level} класса!",
                    f"Здравствуй, {student_name}! Я твой персональный учитель. {level} класс - это интересное время для учебы!"
                ]
                return random.choice(greeting_variants)
            else:
                return "Привет! Я ваш виртуальный учитель. Давайте познакомимся и выберем интересный урок вместе!"
        
        # Анализ контекст разговора
        user_messages = [msg['text'] for msg in self.conversation_history if msg['is_user']]
        last_user_message = user_messages[-1].lower() if user_messages else ""
        
        # Определяем тему разговора по последним сообщениям
        if any(word in last_user_message for word in ['имя', 'зовут', 'меня']):
            if self.has_student_data:
                student_name = self.student_data.get('name', 'ученик')
                return f"Приятно познакомиться, {student_name}! Теперь давайте выберем предмет для изучения. Что вас интересует?"
            else:
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
            level = self.student_data.get('education_level', '5')
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
            
            # Создаем файл урока в папке generated
            lesson_id = f"generated_{topic.lower().replace(' ', '_')}_{int(time.time())}"
            filename = f"{lesson_id}.txt"
            lesson_path = self.generated_lessons_dir / filename
            
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
                'type': 'generated',
                'subject': subject,
                'class_level': self.student_data.get('education_level', 'general') if self.has_student_data else 'general',
                'lesson_number': 999,
                'full_path': f"generated/{filename}"
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
        """Проверяет наличие триггеров для визуализации - ТОЛЬКО SVG"""
        text_lower = text.lower()
        
        visualization_triggers = [
            'структура', 'схема', 'диаграмма', 'график', 'процесс', 
            'алгоритм', 'иерархия', 'взаимосвязь', 'соотношение',
            'таблица', 'классификация', 'этапы', 'стадии', 'система',
            'сравнение', 'типы', 'виды', 'формы', 'принципы', 'компоненты'
        ]
        
        structure_indicators = [
            'состоит из', 'включает в себя', 'делится на', 'подразделяется',
            'можно разделить', 'выделяют', 'различают', 'существуют',
            'основные элементы', 'ключевые аспекты'
        ]
        
        has_trigger = any(trigger in text_lower for trigger in visualization_triggers)
        has_structure = any(indicator in text_lower for indicator in structure_indicators)
        
        return has_trigger or has_structure

    def _generate_visualization(self, text: str, context: str = ""):
        """Генерация SVG инфографики для текста - ТОЛЬКО SVG"""
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
                
                print(f"🎨 Генерация SVG инфографики для: {text[:100]}...")
                
                if self.room_id and self.socketio:
                    # Генерируем SVG инфографику через LLM
                    infographic_result = self.llm.generate_infographic(text, context)
                    
                    if infographic_result and infographic_result.get("success"):
                        self.socketio.emit('visualization_generated', {
                            'room_id': self.room_id,
                            'topic': text[:100],
                            'svg_code': infographic_result['svg_code'],
                            'timestamp': time.time(),
                            'type': 'infographic'
                        }, room=self.room_id)
                        print(f"✅ SVG инфографика отправлена в комнату {self.room_id}")
                    else:
                        # Fallback - простая SVG схема
                        fallback_svg = self._create_fallback_infographic(text)
                        self.socketio.emit('visualization_generated', {
                            'room_id': self.room_id,
                            'topic': text[:100],
                            'svg_code': fallback_svg,
                            'timestamp': time.time(),
                            'type': 'fallback'
                        }, room=self.room_id)
                        print(f"✅ Fallback SVG отправлена в комнату {self.room_id}")
                    
            except Exception as e:
                print(f"❌ Ошибка генерации SVG инфографики: {e}")
                # Fallback при ошибке
                if self.room_id and self.socketio:
                    fallback_svg = self._create_fallback_infographic(text)
                    self.socketio.emit('visualization_generated', {
                        'room_id': self.room_id,
                        'topic': text[:100],
                        'svg_code': fallback_svg,
                        'timestamp': time.time(),
                        'type': 'error_fallback'
                    }, room=self.room_id)

    def _create_fallback_infographic(self, text: str) -> str:
        """Создает простую SVG инфографику как fallback"""
        topic_short = text[:50] + "..." if len(text) > 50 else text
        
        return f'''
<svg width="600" height="400" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bgGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#4f46e5" />
      <stop offset="100%" stop-color="#7c3aed" />
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="4" dy="4" stdDeviation="8" flood-color="#000000" flood-opacity="0.3"/>
    </filter>
  </defs>
  
  <!-- Фон -->
  <rect width="100%" height="100%" fill="url(#bgGradient)" opacity="0.1"/>
  
  <!-- Основной контейнер -->
  <g filter="url(#shadow)">
    <rect x="50" y="50" width="500" height="300" rx="20" fill="white" stroke="#e5e7eb" stroke-width="2"/>
  </g>
  
  <!-- Заголовок -->
  <text x="300" y="100" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" font-weight="bold" fill="#1f2937">
    {topic_short}
  </text>
  
  <!-- Иконка -->
  <circle cx="300" cy="200" r="40" fill="#4f46e5" opacity="0.8"/>
  <text x="300" y="205" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="white">?</text>
  
  <!-- Подпись -->
  <text x="300" y="270" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#6b7280">
    Инфографика по теме
  </text>
  
  <!-- Декоративные элементы -->
  <circle cx="100" cy="100" r="15" fill="#10b981" opacity="0.6"/>
  <circle cx="500" cy="120" r="12" fill="#f59e0b" opacity="0.6"/>
  <circle cx="80" cy="280" r="10" fill="#ef4444" opacity="0.6"/>
  <circle cx="520" cy="260" r="8" fill="#8b5cf6" opacity="0.6"/>
</svg>
'''

    def enable_visualization(self):
        """Включение автоматической визуализации - ТОЛЬКО SVG"""
        self.visualization_enabled = True
        print("✅ Автоматическая SVG визуализация включена")

    def disable_visualization(self):
        """Выключение автоматической визуализации"""
        self.visualization_enabled = False
        print("❌ Автоматическая визуализация выключена")

    def process_input(self, text: str) -> Optional[str]:
        """🔥 ИСПРАВЛЕННАЯ ОБРАБОТКА ВХОДЯЩЕГО ТЕКСТА С УЧЕТОМ ДАННЫХ УЧЕНИКА"""
        text_lower = text.lower().strip()
        
        # 🔥 ИСПРАВЛЕНИЕ: ЕСЛИ УРОК ВЫБРАН И ГОВОРИМ "НАЧАТЬ УРОК" - НАЧИНАЕМ ЕГО!
        if (not self.lesson_started and 
            self.selected_lesson and 
            self.current_subject and
            any(cmd in text_lower for cmd in ['начать урок', 'начнем урок', 'начни урок', 'старт урока', 'приступаем', 'давай начнем', 'готов'])):
            
            print(f"🚀 КОМАНДА НАЧАЛА УРОКА: '{text_lower}', предмет: {self.current_subject}")
            return self._force_start_lesson()
        
        # РАСШИРЕННЫЙ СПИСОК КОМАНД ПРОДОЛЖЕНИЯ
        continue_commands = [
            "продолжай", "продолжить", "дальше", "следующий", "вперед", "давай дальше",
            "записал", "понял", "ясно", "ага", "угу", "хорошо", "ок", " ладно", "ясно",
            "готов", "можно дальше", "слушаю", "понятно", "ясно", "следующий вопрос"
        ]

        if self.lesson_started and any(cmd in text_lower for cmd in continue_commands):
            next_paragraph = self._get_next_paragraph()
            if next_paragraph:
                print(f"✅ Команда продолжения обработана: '{text_lower}' -> следующий абзац")
                return next_paragraph
            else:
                print("🏁 Урок завершен по команде продолжения")
                # 🔥 ОТМЕЧАЕМ УРОК КАК ЗАВЕРШЕННЫЙ
                if self.selected_lesson and self.has_student_data:
                    self.mark_lesson_completed(self.selected_lesson)
                return "Урок завершен. Переходим к практике."
        
        self._add_to_conversation_history(text, is_user=True)
        
        # 🔥 ИСПРАВЛЕНИЕ: УСЛОВИЯ ДЛЯ НАЧАЛА УРОКА ТОЛЬКО ПО ЯВНОМУ СОГЛАСИЮ
        if self.has_student_data and self.current_subject and not self.lesson_started:
            user_messages = [msg for msg in self.conversation_history if msg['is_user']]
            
            # 🔥 КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: Только после 3+ сообщений И явного согласия
            if len(user_messages) >= 3 and not self.lesson_started:
                student_name = self.student_data.get('name', 'ученик')
                
                # 🔥 ЯВНЫЕ КОМАНДЫ ДЛЯ НАЧАЛА УРОКА
                explicit_start_commands = [
                    'начать урок', 'начнем урок', 'поехали', 'готов к уроку', 
                    'давай начнем', 'приступаем', 'начинаем урок', 'старт урока',
                    'начни урок', 'запусти урок', 'хочу урок'
                ]
                
                if any(cmd in text_lower for cmd in explicit_start_commands):
                    # 🔥 ЕСЛИ УЖЕ ЕСТЬ ВЫБРАННЫЙ УРОК - НАЧИНАЕМ ЕГО!
                    if self.selected_lesson:
                        return self._force_start_lesson()
                    else:
                        # 🔥 НОВОЕ: ПРЕДЛАГАЕМ СЛЕДУЮЩИЙ УРОК ИЛИ ВЫБОР
                        return self._suggest_next_or_select_lesson()
                else:
                    # 🔥 ТОЛЬКО ЕСЛИ УЧЕНИК ЯВНО ВЫРАЗИЛ ИНТЕРЕС К УРОКУ
                    lesson_interest_words = ['урок', 'занятие', 'обучение', 'изучение', 'предмет', 'научи']
                    if any(word in text_lower for word in lesson_interest_words):
                        # 🔥 НОВОЕ: ПРЕДЛАГАЕМ ВЫБРАТЬ ИЛИ НАЧАТЬ СЛЕДУЮЩИЙ УРОК
                        return self._offer_lesson_options()
                    # 🔥 ИНАЧЕ - продолжаем обычный диалог без навязывания урока
        
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
            # 🔥 ПЕРСОНАЛИЗИРОВАННЫЙ ОТВЕТ
            personalized_response = self._add_personalization(dialogue_response)
            final_response = self._add_subject_suggestion(personalized_response)
            if final_response:
                self._add_to_conversation_history(final_response, is_user=False)
                return final_response
        
        llm_response = self._handle_llm_dialogue(text)
        if llm_response:
            # 🔥 ОТВЕТ УЖЕ ПЕРСОНАЛИЗИРОВАН В _handle_llm_dialogue
            final_response = self._add_subject_suggestion(llm_response)
            if final_response:
                self._add_to_conversation_history(final_response, is_user=False)
                return final_response
        
        fallback_response = self._get_contextual_fallback()
        if fallback_response:
            # 🔥 ПЕРСОНАЛИЗИРОВАННЫЙ FALLBACK
            personalized_fallback = self._add_personalization(fallback_response)
            self._add_to_conversation_history(personalized_fallback, is_user=False)
            return personalized_fallback
        
        return None

    def _suggest_next_or_select_lesson(self) -> str:
        """Предлагает следующий урок или выбор урока"""
        if not self.has_student_data or not self.current_subject:
            return "Давайте выберем предмет для изучения!"
        
        student_name = self.student_data.get('name', 'ученик')
        
        # 🔥 ПОЛУЧАЕМ СЛЕДУЮЩИЙ УРОК ПО ПРЕДМЕТУ
        next_lesson = self.get_next_lesson_for_student(self.current_subject)
        
        if next_lesson:
            # Есть следующий урок
            progress = self.get_student_progress(self.current_subject)
            completed_count = len(progress.get('completed_lessons', []))
            total_lessons = len(self.get_lessons_for_student_subject(self.current_subject))
            
            response = f"{student_name}, отлично! "
            response += f"Твой прогресс по {self.current_subject}: {completed_count}/{total_lessons} уроков. "
            response += f"Следующий урок: '{next_lesson['title']}'. Хочешь начать его?"
            
            # Сохраняем следующий урок как выбранный
            self.selected_lesson = next_lesson
            
            return response
        else:
            # Все уроки завершены
            return f"{student_name}, ты уже завершил все уроки по {self.current_subject}! Хочешь повторить какой-то урок или выбрать другой предмет?"

    def _offer_lesson_options(self) -> str:
        """Предлагает варианты работы с уроками"""
        if not self.has_student_data or not self.current_subject:
            return "Давайте выберем предмет для изучения!"
        
        student_name = self.student_data.get('name', 'ученик')
        
        # 🔥 ПРЕДЛАГАЕМ ВЫБОР:
        options = [
            f"Хочешь начать следующий урок по {self.current_subject}?",
            f"Или выбрать конкретный урок по {self.current_subject}?",
            f"Может быть, повторить пройденный материал по {self.current_subject}?"
        ]
        
        return f"{student_name}, {random.choice(options)}"

    def _start_lesson_for_student(self) -> str:
        """Начинает урок для ученика по выбранному предмету"""
        print(f"🚀 Начинаем урок для ученика по предмету: {self.current_subject}")
        
        # 🔥 НОВОЕ: ПРОВЕРЯЕМ, ЕСТЬ ЛИ ВЫБРАННЫЙ УРОК
        if not self.selected_lesson:
            # Если урок не выбран, предлагаем следующий
            return self._suggest_next_or_select_lesson()
        
        # Используем исправленную логику выбора предмета
        response = self._handle_subject_selection_direct(self.current_subject)
        
        if response is None:
            # Успешно начали урок
            student_name = self.student_data.get('name', '')
            name_greeting = f"{student_name}, " if student_name else ""
            
            # Получаем первый абзац
            first_paragraph = self._get_next_paragraph()
            if first_paragraph:
                start_message = f"{name_greeting}Отлично! Начинаем урок по {self.current_subject}. {first_paragraph}"
            else:
                start_message = f"{name_greeting}Отлично! Начинаем урок по {self.current_subject}."
            
            self._add_to_conversation_history(start_message, is_user=False)
            
            # 🔥 ГАРАНТИЯ: проверяем, что клиент знает о начале урока
            if self.room_id and self.socketio and self.selected_lesson:
                time.sleep(0.5)  # Небольшая задержка для надежности
                self.socketio.emit('lesson_started', {
                    'lesson_id': self.selected_lesson['id'],
                    'title': self.selected_lesson['title'],
                    'subject': self.current_subject,
                    'class_level': self.selected_lesson.get('class_level', 'general'),
                    'lesson_number': self.selected_lesson.get('lesson_number'),
                    'is_student_lesson': True
                }, room=self.room_id)
                print(f"📢 Уведомление 'lesson_started' отправлено в комнату {self.room_id}")
            
            return start_message
        
        return response

    def _handle_subject_selection_direct(self, subject: str) -> Optional[str]:
        """🔥 ИСПРАВЛЕННАЯ ЛОГИКА ВЫБОРА ПРЕДМЕТА ДЛЯ ВСЕХ КОМНАТ"""
        self.current_subject = subject
        
        print(f"🎯 Выбор предмета: {subject}, данные ученика: {self.has_student_data}")
        
        # 🔥 НОВОЕ: ДЛЯ УЧЕНИКА - ПРЕДЛАГАЕМ СЛЕДУЮЩИЙ УРОК ИЛИ ВЫБОР
        if self.has_student_data:
            student_name = self.student_data.get('name', 'ученик')
            age = self.student_data.get('age', '12')
            level = self.student_data.get('education_level', '5')
            
            # 🔥 ПРОВЕРЯЕМ, ЕСТЬ ЛИ УРОКИ ДЛЯ ЭТОГО КЛАССА
            if level not in self.lessons_by_class or subject not in self.lessons_by_class[level]:
                return f"{student_name}, у меня пока нет уроков по {subject} для {level} класса. Давай выберем другой предмет?"
            
            # 🔥 ПЕРСОНАЛИЗИРОВАННЫЙ ОТВЕТ О ВЫБОРЕ ПРЕДМЕТА
            subject_responses = [
                f"Отлично, {student_name}! {subject} - это интересный предмет для {level} класса.",
                f"Прекрасный выбор, {student_name}! {subject} действительно увлекателен.",
                f"Здорово, {student_name}! Я хорошо знаю {subject} для твоего уровня.",
                f"Отлично, {student_name}! {subject} подходит для {level} класса."
            ]
            
            # 🔥 ДОБАВЛЯЕМ ИНФОРМАЦИЮ О ПРОГРЕССЕ
            progress = self.get_student_progress(subject)
            completed_count = len(progress.get('completed_lessons', []))
            total_lessons = len(self.get_lessons_for_student_subject(subject))
            
            response = f"{random.choice(subject_responses)} "
            
            if completed_count > 0:
                response += f"Ты уже завершил {completed_count} из {total_lessons} уроков. "
            
            # 🔥 ПРЕДЛАГАЕМ СЛЕДУЮЩИЙ УРОК
            next_lesson = self.get_next_lesson_for_student(subject)
            if next_lesson:
                response += f"Следующий урок: '{next_lesson['title']}'. Хочешь начать его?"
                self.selected_lesson = next_lesson
            else:
                response += f"Ты уже завершил все уроки по {subject}! Хочешь повторить или выбрать другой предмет?"
            
            return response
        
        # 🔥 ДЛЯ ОБЫЧНЫХ ПОЛЬЗОВАТЕЛЕЙ: выбираем урок, но не начинаем автоматически
        available_lessons = self._get_available_lessons(subject)
        
        if available_lessons:
            self.selected_lesson = available_lessons[0]
            print(f"✅ Выбран существующий урок: {self.selected_lesson['title']}")
            
            # 🔥 НЕ НАЧИНАЕМ УРОК, ПРОСТО ИНФОРМИРУЕМ
            return f"Отлично! Я выбрал урок '{self.selected_lesson['title']}' по предмету {subject}. Когда будете готовы, скажите 'начать урок'!"
        else:
            # 🔥 КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Если уроков нет - создаем новый
            print(f"⚠️ Урок по предмету '{subject}' не найден. Генерация урока 'на лету'...")
            
            # Генерируем урок по предмету
            generated_lesson = self.generate_lesson_on_demand(f"Введение в {subject}")
            
            if generated_lesson:
                self.selected_lesson = generated_lesson
                print(f"✅ Сгенерирован новый урок: {generated_lesson['title']}")
                
                # 🔥 НЕ НАЧИНАЕМ УРОК, ПРОСТО ИНФОРМИРУЕМ
                return f"Я создал для вас урок по теме '{subject}'. Когда будете готовы, скажите 'начать урок'!"
            else:
                # Fallback на демо-урок, если генерация не удалась
                print("❌ Генерация не удалась, создаем демо-урок")
                self.selected_lesson = self._create_demo_lesson(subject)
                
                return f"Я подготовил демо-урок по предмету {subject}. Когда будете готовы, скажите 'начать урок'!"

    def _force_start_lesson(self) -> str:
        """🔥 НОВЫЙ МЕТОД: Принудительно начинает выбранный урок"""
        if not self.selected_lesson:
            return "Сначала выберите урок!"
        
        if not self.current_subject:
            self.current_subject = self.selected_lesson.get('subject', 'общее')
        
        print(f"🚀 ПРИНУДИТЕЛЬНЫЙ СТАРТ УРОКА: {self.selected_lesson['title']}")
        
        # Начинаем урок
        self.lesson_started = True
        self.current_state = "lesson_reading"
        self.current_paragraph = 0
        
        # Загружаем содержание
        try:
            self.lesson_content = self._load_lesson_content(self.selected_lesson['file_path'])
            if not self.lesson_content:
                self.lesson_started = False
                return "Ошибка загрузки урока. Попробуйте другой."
            
            # Инициализируем базу знаний
            if self.current_subject:
                from knowledge.knowledge_base import KnowledgeBase
                self.knowledge_base = KnowledgeBase(self.current_subject)
            
            # Очищаем историю
            self.conversation_history = []
            self.conversation_context = []
            
            # Получаем первый абзац
            first_paragraph = self._get_next_paragraph()
            
            if self.room_id and self.socketio:
                # Уведомляем клиент
                self.socketio.emit('lesson_started', {
                    'lesson_id': self.selected_lesson['id'],
                    'title': self.selected_lesson['title'],
                    'subject': self.current_subject,
                    'is_generated': self.selected_lesson.get('type') == 'generated'
                }, room=self.room_id)
            
            # Персонализированное начало
            if self.has_student_data:
                student_name = self.student_data.get('name', '')
                name_prefix = f"{student_name}, " if student_name else ""
                return f"{name_prefix}Отлично! Начинаем урок по {self.current_subject}. {first_paragraph}"
            else:
                return f"Отлично! Начинаем урок по {self.current_subject}. {first_paragraph}"
                
        except Exception as e:
            print(f"❌ Ошибка начала урока: {e}")
            self.lesson_started = False
            return f"Ошибка начала урока: {str(e)}"

    def _get_available_lessons(self, subject: str) -> List[dict]:
        """🔥 ВОЗВРАЩАЕТ УРОКИ В ЗАВИСИМОСТИ ОТ НАЛИЧИЯ ДАННЫХ УЧЕНИКА"""
        all_lessons = self.lessons.get(subject, [])
        
        if self.has_student_data:
            # Для ученика: только уроки его класса
            student_class = self.student_data.get('education_level', '5')
            if student_class in self.lessons_by_class and subject in self.lessons_by_class[student_class]:
                return self.lessons_by_class[student_class][subject]
            return []
        else:
            # Для обычного пользователя: только демо-уроки
            return [lesson for lesson in all_lessons if lesson['type'] == 'demo']

    def _create_demo_lesson(self, subject: str) -> dict:
        """Создает демо-урок для обычного пользователя"""
        lesson_id = f"demo_{subject}"
        filename = f"{lesson_id}.txt"
        lesson_path = self.demo_lessons_dir / filename
        
        # Простой демо-контент
        demo_content = f"""Демо-урок по предмету {subject}.

Это демонстрационный урок. В реальной системе здесь был бы полноценный учебный материал.

Для доступа ко всех функциям системы зарегистрируйтесь как ученик!"""
        
        with open(lesson_path, 'w', encoding='utf-8') as f:
            f.write(demo_content)
        
        lesson_data = {
            'id': lesson_id,
            'title': f"Демо-урок по {subject}",
            'file_path': lesson_path,
            'type': 'demo',
            'subject': subject,
            'class_level': "general",
            'lesson_number': 1,
            'full_path': f"demo/{filename}"
        }
        
        if subject not in self.lessons:
            self.lessons[subject] = []
        self.lessons[subject].append(lesson_data)
        
        return lesson_data

    def _handle_greeting(self, text: str) -> Optional[str]:
        greeting_words = ["привет", "здравствуй", 'начать', "старт", " готов", "поехали", "давай", "началом"]
        if any(word in text for word in greeting_words):
            self.current_state = "subject_selection"
            
            # 🔥 ПЕРСОНАЛИЗИРОВАННОЕ ПРИВЕТСТВИЕ ДЛЯ УЧЕНИКА
            if self.has_student_data:
                student_name = self.student_data.get('name', 'ученик')
                age = self.student_data.get('age', '12')
                level = self.student_data.get('education_level', '5')
                
                personalized_greetings = [
                    f"Привет, {student_name}! Я твой виртуальный учитель. Очень рад тебя видеть! Ты в {level} классе, тебе {age} лет - это прекрасный возраст для учебы!",
                    f"Здравствуй, {student_name}! Я твой AI-репетитор. Вижу, ты учишься в {level} классе. Готов помочь тебе с учебой!",
                    f"Привет, {student_name}! Я твой цифровой преподаватель. {age} лет - отличный возраст для новых открытий! Давай сделаем обучение интересным!",
                    f"Здравствуй, {student_name}! Я твой персональный учитель. Рад познакомиться с учеником {level} класса! Готов к увлекательному уроку?"
                ]
                return random.choice(personalized_greetings)
            else:
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
            
            # 🔥 ПЕРСОНАЛИЗИРОВАННОЕ СООБЩЕНИЕ ДЛЯ УЧЕНИКА
            if self.has_student_data:
                student_name = self.student_data.get('name', '')
                name_prefix = f"{student_name}, " if student_name else ""
                return f"{name_prefix}Урок остановлен. Скажи 'привет' когда захочешь продолжить."
            else:
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
            
            # 🔥 ПЕРСОНАЛИЗИРОВАННОЕ СООБЩЕНИЕ ДЛЯ УЧЕНИКА
            if self.has_student_data:
                student_name = self.student_data.get('name', '')
                name_prefix = f"{student_name}, " if student_name else ""
                return f"{name_prefix}Практика остановлена. Скажи 'привет' когда захочешь продолжить."
            else:
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
            # 🔥 ОТМЕЧАЕМ УРОК КАК ЗАВЕРШЕННЫЙ
            if self.selected_lesson and self.has_student_data:
                self.mark_lesson_completed(self.selected_lesson)
            
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
            
            # 🔥 ПЕРСОНАЛИЗИРОВАННОЕ СООБЩЕНИЕ ДЛЯ УЧЕНИКА
            if self.has_student_data:
                student_name = self.student_data.get('name', '')
                name_prefix = f"{student_name}, " if student_name else ""
                return f"{name_prefix}Отлично! Переходим к практике. Первый вопрос: {first_question}"
            else:
                return f"Отлично! Переходим к практике. Первый вопрос: {first_question}"
        else:
            print("❌ Не удалось получить первый вопрос практики")
            self.practice_active = False
            return "Практические задания временно недоступны. Давайте продолжим урок или выберем другую тему."

    def _handle_practice_answer(self, text: str) -> str:
        """Обработка ответа ученика во время практики"""
        return self._evaluate_and_generate_next(text)

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
                # 🔥 ПЕРСОНАЛИЗИРОВАННЫЙ ОТВЕТ
                if self.has_student_data:
                    student_name = self.student_data.get('name', '')
                    return f"{student_name}, это похоже на команду. Пожалуйста, дай ответ на вопрос. Следующий вопрос: {next_question}"
                else:
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
        
        # УВЕЛИЧИВАЕМ СЧЕТЧИК ОТВЕТОВ
        self.current_question_index += 1
        print(f"📊 Текущий номер вопроса: {self.current_question_index}/{self.max_questions}")
        
        # ПРОВЕРЯЕМ ЛИМИТ ВОПРОСОВ
        if self.current_question_index >= self.max_questions:
            print(f"🏁 Достигнут лимит вопросов: {self.current_question_index}/{self.max_questions}")
            self._end_practice_session()
            
            # 🔥 ПЕРСОНАЛИЗИРОВАННОЕ СООБЩЕНИЕ ДЛЯ УЧЕНИКА
            if self.has_student_data:
                student_name = self.student_data.get('name', '')
                name_prefix = f"{student_name}, " if student_name else ""
                return f"{name_prefix}Отлично! Ты ответил на все вопросы практики. Урок завершен!"
            else:
                return "Отлично! Вы ответили на все вопросы практики. Урок завершен!"
        
        # ИСПОЛЬЗУЕМ НОВЫЙ МЕТОД: оценка + следующий вопрос
        feedback, next_question = self.practice_manager.evaluate_and_continue(
            student_answer, 
            current_question["question"]
        )
        
        # 🔥 АДАПТИРУЕМ ОБРАТНУЮ СВЯЗЬ ДЛЯ УЧЕНИКА
        if self.has_student_data:
            student_name = self.student_data.get('name', '')
            if student_name and feedback:
                # Добавляем обращение по имени к фидбеку
                feedback = f"{student_name}, {feedback.lower()}"
        
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
        
        # 🔥 ПЕРСОНАЛИЗИРУЕМ ОТВЕТ НА ВОПРОС
        if self.has_student_data and final_response:
            final_response = self._add_personalization(final_response)
        
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
        
        # Сброс данных ученика
        self.student_data = {}
        self.has_student_data = False

    def get_available_subjects(self) -> List[str]:
        """🔥 ИСПРАВЛЕННАЯ ЛОГИКА: Возвращает доступные предметы"""
        # 🔥 ДЛЯ УЧЕНИКА: предметы его класса
        if self.has_student_data and self.student_data.get('education_level'):
            student_class = self.student_data['education_level']
            if student_class in self.lessons_by_class:
                return list(self.lessons_by_class[student_class].keys())
        
        # Для обычных пользователей
        subjects = list(self.lessons.keys())
        
        # Всегда добавляем обществознание
        if "обществознание" not in subjects:
            subjects.append("обществознание")
            
        return subjects

    def get_lessons_for_subject(self, subject: str) -> List[dict]:
        """🔥 ИСПРАВЛЕННАЯ ЛОГИКА: Возвращает уроки по предмету"""
        return self._get_available_lessons(subject)

    def get_lessons_for_student_subject(self, subject: str) -> List[dict]:
        """🔥 НОВЫЙ МЕТОД: Возвращает уроки по предмету для текущего ученика"""
        if not self.has_student_data:
            return []
        
        student_class = self.student_data.get('education_level', '5')
        if student_class in self.lessons_by_class and subject in self.lessons_by_class[student_class]:
            # Сортируем по номеру урока
            lessons = self.lessons_by_class[student_class][subject]
            return sorted(lessons, key=lambda x: x.get('lesson_number', 999))
        
        return []

    def get_next_lesson_for_student(self, subject: str) -> Optional[Dict]:
        """🔥 НОВЫЙ МЕТОД: Возвращает следующий незавершенный урок по предмету"""
        if not self.has_student_data:
            return None
        
        student_class = self.student_data.get('education_level', '5')
        if (student_class not in self.lessons_by_class or 
            subject not in self.lessons_by_class[student_class]):
            return None
        
        # Получаем прогресс ученика по предмету
        progress = self.get_student_progress(subject)
        completed_ids = progress.get('completed_lessons', [])
        
        # Ищем следующий незавершенный урок
        available_lessons = self.get_lessons_for_student_subject(subject)
        
        for lesson in available_lessons:
            if lesson['id'] not in completed_ids:
                return lesson
        
        return None  # Все уроки завершены

    def get_student_progress(self, subject: str = None) -> Dict:
        """🔥 НОВЫЙ МЕТОД: Возвращает прогресс ученика"""
        if not self.has_student_data:
            return {}
        
        student_id = self.student_data.get('student_id')
        if not student_id:
            return {}
        
        # 🔥 НОВОЕ: Загружаем прогресс из файла
        progress_file = self.progress_dir / f"{student_id}.json"
        try:
            if progress_file.exists():
                with open(progress_file, 'r', encoding='utf-8') as f:
                    all_progress = json.load(f)
                    
                    if subject:
                        return all_progress.get(subject, {
                            "completed_lessons": [],
                            "current_lesson": None,
                            "total_lessons": len(self.get_lessons_for_student_subject(subject)) if subject else 0,
                            "last_updated": 0
                        })
                    else:
                        return all_progress
        except Exception as e:
            print(f"Ошибка загрузки прогресса: {e}")
        
        # Возвращаем пустой прогресс если файла нет
        if subject:
            return {
                "completed_lessons": [],
                "current_lesson": None,
                "total_lessons": len(self.get_lessons_for_student_subject(subject)) if subject else 0,
                "last_updated": 0
            }
        else:
            return {}

    def save_student_progress(self, lesson_id: str, subject: str, completed: bool = True):
        """🔥 НОВЫЙ МЕТОД: Сохраняет прогресс ученика"""
        if not self.has_student_data:
            return
        
        student_id = self.student_data.get('student_id')
        if not student_id:
            return
        
        progress_file = self.progress_dir / f"{student_id}.json"
        
        # Загружаем существующий прогресс
        progress_data = {}
        if progress_file.exists():
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
            except:
                pass
        
        # Инициализируем прогресс по предмету если нужно
        if subject not in progress_data:
            progress_data[subject] = {
                "completed_lessons": [],
                "current_lesson": lesson_id,
                "total_lessons": len(self.get_lessons_for_student_subject(subject)) if subject else 0,
                "last_updated": time.time()
            }
        
        # Обновляем прогресс
        subject_progress = progress_data[subject]
        
        if completed and lesson_id not in subject_progress["completed_lessons"]:
            subject_progress["completed_lessons"].append(lesson_id)
            subject_progress["current_lesson"] = lesson_id
            subject_progress["last_updated"] = time.time()
            
            # Обновляем общее количество уроков
            subject_progress["total_lessons"] = len(self.get_lessons_for_student_subject(subject))
        
        # Сохраняем
        try:
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
            print(f"✅ Прогресс сохранен: {lesson_id} по предмету {subject}")
        except Exception as e:
            print(f"❌ Ошибка сохранения прогресса: {e}")

    def mark_lesson_completed(self, lesson_data: Dict):
        """🔥 НОВЫЙ МЕТОД: Помечает урок как завершенный"""
        if lesson_data and self.has_student_data:
            print(f"🎓 Отмечаем урок как завершенный: {lesson_data['title']}")
            self.save_student_progress(
                lesson_data['id'], 
                lesson_data['subject'], 
                completed=True
            )

    def get_available_subjects_for_student(self) -> Dict[str, List[Dict]]:
        """🔥 НОВЫЙ МЕТОД: Возвращает предметы и уроки для текущего ученика по его классу"""
        if not self.has_student_data:
            return {}
        
        student_class = self.student_data.get('education_level', '5')
        if student_class not in self.lessons_by_class:
            return {}
        
        result = {}
        for subject, lessons in self.lessons_by_class[student_class].items():
            # Сортируем уроки по номеру
            sorted_lessons = sorted(lessons, key=lambda x: x.get('lesson_number', 999))
            result[subject] = sorted_lessons
        
        return result

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

    def set_student_data(self, student_data: dict):
        """🔥 ПРОСТО УСТАНАВЛИВАЕМ ДАННЫЕ УЧЕНИКА"""
        self.student_data = student_data
        self.has_student_data = bool(student_data)
        
        if self.has_student_data:
            student_name = student_data.get('name', 'неизвестно')
            student_class = student_data.get('education_level', 'неизвестно')
            print(f"🎓 Установлены данные ученика: {student_name} ({student_class} класс)")
            
            # 🔥 НОВОЕ: Если в данных есть предмет - устанавливаем его как текущий
            if 'room_subject' in student_data:
                self.current_subject = student_data['room_subject']
                print(f"📚 Установлен предмет комнаты: {self.current_subject}")

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
        """Возвращает статус визуализации - ТОЛЬКО SVG"""
        return {
            "visualization_enabled": self.visualization_enabled,
            "visualization_counter": self.visualization_counter,
            "last_visualization_time": self.last_visualization_time,
            "paragraphs_since_last_viz": self.paragraphs_since_last_viz,
            "type": "svg_infographic"
        }

    def force_visualization(self, text: str) -> bool:
        """Принудительно генерирует SVG инфографику для текста"""
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
            "conversation_context": self.conversation_context,
            "has_student_data": self.has_student_data,
            "student_name": self.student_data.get('name', 'нет'),
            "student_class": self.student_data.get('education_level', 'нет')
        }

    def debug_info(self) -> Dict:
        """Возвращает отладочную информацию"""
        practice_stats = self.practice_manager.get_practice_stats() if hasattr(self.practice_manager, 'get_practice_stats') else {}
        
        # 🔥 НОВОЕ: Информация о прогрессе
        student_progress = {}
        if self.has_student_data and self.current_subject:
            student_progress = self.get_student_progress(self.current_subject)
        
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
            # Информация о данных ученика
            "has_student_data": self.has_student_data,
            "student_data": self.student_data,
            # 🔥 НОВОЕ: Информация о структуре уроков
            "available_classes": list(self.lessons_by_class.keys()),
            "student_class_lessons": self.get_available_subjects_for_student(),
            "student_progress": student_progress,
            "next_lesson": self.get_next_lesson_for_student(self.current_subject) if self.current_subject else None
        }

    def add_custom_lesson(self, subject: str, title: str, content: str) -> bool:
        """Добавляет пользовательский урок"""
        try:
            # Создаем имя файла
            filename = f"{subject}_{title.lower().replace(' ', '_')}.txt"
            
            # Сохраняем в соответствующую папку
            if self.has_student_data:
                student_class = self.student_data.get('education_level', '5')
                lesson_path = self.students_base_dir / f"{student_class}_class" / subject / filename
                lesson_path.parent.mkdir(parents=True, exist_ok=True)
                lesson_type = "student"
            else:
                lesson_path = self.demo_lessons_dir / filename
                lesson_type = "demo"
            
            # Записываем контент
            with open(lesson_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Добавляем в список уроков
            lesson_id = f"{lesson_type}_{filename.replace('.txt', '')}"
            lesson_data = {
                'id': lesson_id,
                'title': title,
                'file_path': lesson_path,
                'type': lesson_type,
                'subject': subject,
                'class_level': self.student_data.get('education_level', 'general') if self.has_student_data else 'general',
                'lesson_number': 999,
                'full_path': str(lesson_path.relative_to(self.lessons_dir))
            }
            
            if subject not in self.lessons:
                self.lessons[subject] = []
            
            self.lessons[subject].append(lesson_data)
            
            # 🔥 НОВОЕ: Добавляем в структуру по классам
            if self.has_student_data:
                student_class = self.student_data['education_level']
                if student_class not in self.lessons_by_class:
                    self.lessons_by_class[student_class] = {}
                if subject not in self.lessons_by_class[student_class]:
                    self.lessons_by_class[student_class][subject] = []
                self.lessons_by_class[student_class][subject].append(lesson_data)
            
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
        
        # 🔥 НОВОЕ: Информация о прогрессе ученика по предмету
        subject_progress = {}
        if self.has_student_data and self.current_subject:
            subject_progress = self.get_student_progress(self.current_subject)
        
        return {
            "current_paragraph": self.current_paragraph,
            "total_paragraphs": len(self.lesson_content),
            "progress_percent": round(progress_percent, 1),
            "remaining_paragraphs": len(self.lesson_content) - self.current_paragraph,
            "lesson_title": self.selected_lesson['title'] if self.selected_lesson else "Неизвестно",
            "subject": self.current_subject,
            "student_progress": subject_progress
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
        
        # 🔥 НОВОЕ: Прогресс ученика по всем предметам
        student_progress_all = {}
        if self.has_student_data:
            student_progress_all = self.get_student_progress()
        
        return {
            "dialogue_manager": {
                "current_state": self.current_state,
                "lesson_started": self.lesson_started,
                "practice_active": self.practice_active,
                "current_subject": self.current_subject,
                "conversation_history_length": len(self.conversation_history),
                "questions_asked": len(self.practice_manager.generated_questions) if hasattr(self.practice_manager, 'generated_questions') else 0,
                "max_questions": self.max_questions,
                "has_student_data": self.has_student_data,
                "student_data": self.student_data,
                "student_progress": student_progress_all
            },
            "llm": llm_status,
            "knowledge_base": knowledge_stats,
            "practice": practice_stats,
            "visualization": self.get_visualization_status(),
            "lessons": {
                "available_subjects": self.get_available_subjects(),
                "total_lessons": sum(len(lessons) for lessons in self.lessons.values()),
                "lessons_by_class": {k: {subj: len(lessons) for subj, lessons in v.items()} 
                                   for k, v in self.lessons_by_class.items()}
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
        base_commands = {
            "привет": "Начать диалог",
            "продолжай": "Продолжить урок",
            "стоп": "Остановить урок/практику",
            "какой предмет": "Показать доступные предметы",
            "практика": "Перейти к практике (если урок завершен)",
            "статус": "Показать статус системы",
            "помощь": "Показать эту справку"
        }
        
        # 🔥 НОВОЕ: Дополнительные команды для учеников
        if self.has_student_data:
            student_commands = {
                "мой прогресс": "Показать прогресс по предметам",
                "следующий урок": "Начать следующий урок по текущему предмету",
                "выбрать урок": "Выбрать конкретный урок",
                "мои уроки": "Показать все доступные уроки"
            }
            base_commands.update(student_commands)
        
        return base_commands

    def force_lesson_start_notification(self):
        """Принудительно отправляет уведомление о начале урока (для восстановления состояния)"""
        if self.lesson_started and self.selected_lesson and self.room_id and self.socketio:
            print(f"🔧 ПРИНУДИТЕЛЬНАЯ отправка lesson_started для комнаты {self.room_id}")
            
            self.socketio.emit('lesson_started', {
                'lesson_id': self.selected_lesson['id'],
                'title': self.selected_lesson['title'],
                'subject': self.current_subject,
                'class_level': self.selected_lesson.get('class_level', 'general'),
                'lesson_number': self.selected_lesson.get('lesson_number'),
                'is_student_lesson': True
            }, room=self.room_id)
            
            # Также отправляем текущий абзац если есть
            if self.lesson_content and self.current_paragraph > 0:
                current_paragraph = self.lesson_content[self.current_paragraph - 1]
                self.socketio.emit('speech_text', {
                    'text': f"Учитель: {current_paragraph}",
                    'sid': 'teacher',
                    'is_teacher': True
                }, room=self.room_id)
            
            return True
        return False

    def get_lessons_for_student_api(self) -> Dict:
        """🔥 НОВЫЙ МЕТОД: Возвращает уроки для API запроса (для личного кабинета)"""
        if not self.has_student_data:
            return {"success": False, "error": "Нет данных ученика"}
        
        student_class = self.student_data.get('education_level', '5')
        student_name = self.student_data.get('name', 'ученик')
        
        result = {
            "success": True,
            "student_name": student_name,
            "student_class": student_class,
            "subjects": []
        }
        
        if student_class in self.lessons_by_class:
            for subject, lessons in self.lessons_by_class[student_class].items():
                # Получаем прогресс по предмету
                progress = self.get_student_progress(subject)
                completed_ids = progress.get('completed_lessons', [])
                
                # Сортируем уроки
                sorted_lessons = sorted(lessons, key=lambda x: x.get('lesson_number', 999))
                
                subject_lessons = []
                for lesson in sorted_lessons:
                    is_completed = lesson['id'] in completed_ids
                    subject_lessons.append({
                        'id': lesson['id'],
                        'title': lesson['title'],
                        'subject': lesson['subject'],
                        'class_level': lesson.get('class_level', student_class),
                        'lesson_number': lesson.get('lesson_number'),
                        'completed': is_completed,
                        'file_path': str(lesson.get('file_path', '')),
                        'type': lesson.get('type', 'student')
                    })
                
                result["subjects"].append({
                    'subject': subject,
                    'lessons': subject_lessons,
                    'total_lessons': len(subject_lessons),
                    'completed_lessons': len([l for l in subject_lessons if l['completed']]),
                    'progress_percent': int((len([l for l in subject_lessons if l['completed']]) / len(subject_lessons)) * 100) if subject_lessons else 0
                })
        
        return result


# Создаем глобальный экземпляр для тестирования
if __name__ == "__main__":
    # Тестирование базовой функциональности
    dm = DialogueManager(None)
    
    print("🧪 Тестирование DialogueManager с новой структурой по классам...")
    
    # Тест доступных предметов
    subjects = dm.get_available_subjects()
    print(f"📚 Доступные предметы: {subjects}")
    
    # Тест загрузки уроков по классам
    print(f"📊 Уроки по классам: {list(dm.lessons_by_class.keys())}")
    
    # Тест обработки приветствия
    response = dm.process_input("привет")
    print(f"👋 Ответ на приветствие: {response}")
    
    # Тест статуса системы
    status = dm.get_system_status()
    print(f"📊 Статус системы: {status.keys()}")
    
    print("✅ Тестирование завершено!")
