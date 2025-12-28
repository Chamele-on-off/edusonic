# dialogue.py
# ИСПРАВЛЕННАЯ ВЕРСИЯ с полной ленивой инициализацией
# + ДОБАВЛЕНА ПОДДЕРЖКА СЛАЙДОВ ИЗОБРАЖЕНИЙ ДЛЯ УРОКОВ
# + ДОБАВЛЕНА ПОДДЕРЖКА ВЗРОСЛЫХ И УРОВНЕЙ CEFR

import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from difflib import SequenceMatcher
import random
import re
from knowledge.knowledge_base import KnowledgeBase
from llm import LLMIntegration
from config import get_llm_mode, get_dialogue_settings
import time
import threading
from practice_manager import PracticeManager

# 🔥 ИМПОРТ ДЛЯ ЯЗЫКОВОЙ ИНТЕГРАЦИИ
try:
    from language_integration import (
        LanguageIntegration, 
        is_language_subject, 
        get_language_settings,
        detect_cefr_level,
        get_cefr_level_config,
        create_bilingual_lesson_prompt_cefr,
        get_adult_study_modes,
        get_available_cefr_levels,
    )
    LANGUAGE_SUPPORT_ENABLED = True
except ImportError:
    LANGUAGE_SUPPORT_ENABLED = False
    print(f"🔥 [DIALOGUE] ⚠️ language_integration.py не найден, языковая поддержка отключена")

# 🔥 ИМПОРТ ДЛЯ ТЕХНИЧЕСКИХ ПРЕДМЕТОВ
try:
    from technical_subjects import (
        is_technical_subject,
        get_subject_type,
        generate_technical_practice_prompt,
        adapt_visualization_for_technical,
        TECHNICAL_SUBJECTS,
        NATURAL_SCIENCES
    )
    TECHNICAL_SUPPORT_ENABLED = True
except ImportError:
    TECHNICAL_SUPPORT_ENABLED = False
    print(f"🔥 [DIALOGUE] ⚠️ technical_subjects.py не найден, техническая поддержка отключена")

# 🔥 ИМПОРТ ПРОМПТОВ ДЛЯ ТЕХНИЧЕСКИХ ПРЕДМЕТОВ
try:
    from prompts.technical_prompts import (
        get_lesson_prompt,
        get_practice_prompt,
        TECHNICAL_LESSON_PROMPTS
    )
    TECHNICAL_PROMPTS_ENABLED = True
except ImportError:
    TECHNICAL_PROMPTS_ENABLED = False
    print(f"🔥 [DIALOGUE] ⚠️ technical_prompts.py не найден, промпты отключены")

def debug_log(message):
    """Логирование для отладки"""
    print(f"🔥 [DIALOGUE] {message}")

def generate_svg_code(topic: str, context: str = "") -> str:
    """Генерирует простую SVG инфографику как fallback"""
    topic_short = topic[:50] + "..." if len(topic) > 50 else topic
    
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
  <text x="300" y="205" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" fill="white">📊</text>
  
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

class DialogueManager:
    def __init__(self, socketio):
        self.socketio = socketio
        
        # 🔥 КРИТИЧЕСКО ВАЖНО: ПОЛНОСТЬЮ ЛЕНИВАЯ ИНИЦИАЛИЗАЦИЯ
        # НИКАКИХ операций с диском, парсинга файлов, инициализации LLM
        
        # Базовые поля состояния
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
        
        # 🔥 НОВОЕ: Список слайдов для текущего урока
        self.lesson_slides = []  # список URL или путей к файлам слайдов
        self.slides_enabled = True  # флаг включения слайдов
        
        # Пути к папкам (только создание объектов Path, без проверки существования)
        self.lessons_dir = Path("lessons")
        self.demo_lessons_dir = self.lessons_dir / "demo"
        self.students_base_dir = self.lessons_dir / "students"
        self.generated_lessons_dir = self.lessons_dir / "generated"
        
        # 🔥 ЛЕНИВЫЕ СТРУКТУРЫ ДАННЫХ
        self._student_progress = None
        self.last_progress_save = 0
        self.progress_dir = Path("students_progress")
        
        # 🔥 КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: ПОЛНОСТЬЮ ЛЕНИВАЯ ИНИЦИАЛИЗАЦИЯ
        self._llm = None
        self._llm_lock = threading.Lock()
        self._lessons_loaded = False
        self._lessons_lock = threading.Lock()
        self._dialogue_knowledge = None
        self._dialogue_knowledge_lock = threading.Lock()
        self.knowledge_base = None
        
        # Простые поля
        self.conversation_counter = 0
        self.llm_query_mode = get_llm_mode()
        self.dialogue_settings = get_dialogue_settings()
        self.conversation_history = []
        self.conversation_context = []
        self.room_id = None
        
        # Менеджер практики - ленивая инициализация
        self._practice_manager = None
        self._practice_manager_lock = threading.Lock()
        
        # Новые поля для практики
        self.practice_active = False
        self.current_question_index = 0
        self.current_expected_answer = ""
        self.waiting_for_answer = False
        self.current_practice_question = None
        self.max_questions = 5
        
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
        
        # Простые данные ученика
        self.student_data = {}
        self.has_student_data = False
        
        # 🔥 НОВЫЕ ПОЛЯ ДЛЯ ВЗРОСЛЫХ И ЯЗЫКОВЫХ УРОВНЕЙ
        self.is_adult_student = False
        self.adult_study_mode = None  # 'language' или 'anything'
        self.cefr_level = None  # A1, A2, B1, B2, C1, C2
        self.cefr_config = None
        
        # 🔥 ЛЕНИВЫЕ ДАННЫЕ УРОКОВ
        self._lessons = None
        self._lessons_by_class = None
        self.available_classes = ["5", "6", "7", "8", "9", "10", "11", "adult"]
        
        # Маппинг английских названий предметов
        self.subject_mapping = {
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
            'german': 'немецкий язык',
            'spanish': 'испанский язык',
            'geography': 'география',
            'informatics': 'информатика'
        }
        
        # Обратный маппинг
        self.subject_mapping_reverse = {v: k for k, v in self.subject_mapping.items()}
        
        # Поля для визуализации - ТОЛЬКО SVG
        self.visualization_enabled = True
        self.last_visualization_time = 0
        self.visualization_cooldown = 5
        self.visualization_counter = 0
        self.paragraphs_since_last_viz = 0
        self.viz_paragraph_interval = 2
        
        # Поля для языковой поддержки
        self.is_language_subject = False
        self.target_language = 'english'
        self.language_level = 'beginner'
        self.bilingual_ratio = 0.3
        self.language_practice_manager = None
        
        # 🔥 НОВЫЕ ПОЛЯ ДЛЯ ТЕХНИЧЕСКИХ ПРЕДМЕТОВ
        self.is_technical_subject = False
        self.subject_type = "general"  # "technical", "natural_science", "language", "humanitarian"
        self.technical_symbols_preserved = False
        
        # Локальные шаблоны (маленькие, можно оставить в памяти)
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
        
        debug_log(f"✅ DialogueManager создан за <1 мс (полностью ленивая инициализация)")

    # 🔥 ЛЕНИВЫЕ СВОЙСТВА
    @property
    def llm(self):
        """Ленивое свойство: Инициализирует LLM только при первом обращении"""
        if self._llm is None:
            with self._llm_lock:
                if self._llm is None:
                    debug_log("🔄 Ленивая инициализация LLMIntegration...")
                    self._llm = LLMIntegration()
                    debug_log("✅ LLMIntegration инициализирован")
        return self._llm

    @property
    def practice_manager(self):
        """Ленивое свойство: Инициализирует PracticeManager только при первом обращении"""
        if self._practice_manager is None:
            with self._practice_manager_lock:
                if self._practice_manager is None:
                    debug_log("🔄 Ленивая инициализация PracticeManager...")
                    self._practice_manager = PracticeManager(self.llm)
                    debug_log("✅ PracticeManager инициализирован")
        return self._practice_manager

    @property
    def dialogue_knowledge(self):
        """🔥 ЛЕНИВОЕ СВОЙСТВО: Загружает диалоговые шаблоны только при первом обращении"""
        if self._dialogue_knowledge is None:
            with self._dialogue_knowledge_lock:
                if self._dialogue_knowledge is None:
                    debug_log("🔄 Ленивая загрузка dialogue_knowledge...")
                    self._dialogue_knowledge = self._load_dialogue_knowledge()
                    debug_log(f"✅ dialogue_knowledge загружен: {len(self._dialogue_knowledge)} категорий")
        return self._dialogue_knowledge

    @property
    def lessons(self):
        """Ленивое свойство: Загружает уроки только при первом обращении"""
        if self._lessons is None:
            self._ensure_lessons_loaded()
        return self._lessons

    @property
    def lessons_by_class(self):
        """Ленивое свойство: Загружает уроки по классам только при первом обращении"""
        if self._lessons_by_class is None:
            self._ensure_lessons_loaded()
        return self._lessons_by_class

    def _ensure_lessons_loaded(self):
        """Ленивая загрузка: Загружает уроки только при первом обращении"""
        if not self._lessons_loaded:
            with self._lessons_lock:
                if not self._lessons_loaded:
                    debug_log("🔄 Ленивая загрузка уроков...")
                    self._load_lessons()
                    self._lessons_loaded = True
                    debug_log("✅ Уроки загружены")

    # 🔥 НОВЫЙ МЕТОД: Поиск слайдов для урока
    def _find_lesson_slides(self, lesson_path: Path) -> List[str]:
        """Ищет слайды (JPG/PNG/MP4/GIF) рядом с уроком: lesson_01.jpg, lesson_02.jpg и т.д."""
        if not lesson_path or not lesson_path.exists():
            debug_log(f"❌ Путь урока не существует или не указан: {lesson_path}")
            return []
    
        base_name = lesson_path.stem
        lesson_dir = lesson_path.parent
        debug_log(f"🔍 Поиск слайдов для урока: {base_name} в папке {lesson_dir}")
    
        slides = []
        idx = 1
        max_slides = 20
    
        while idx <= max_slides:
            found_slide = None
            # Основные шаблоны поиска
            patterns = [
                f"{base_name}_{idx:02d}",
                f"{base_name}_{idx}",
                f"{base_name}_slide_{idx:02d}",
                f"{base_name}_slide_{idx}",
                f"slide_{idx:02d}",
                f"slide_{idx}",
            ]
            extensions = ['.jpg', '.jpeg', '.png', '.gif', '.mp4']
        
            for pattern in patterns:
                if found_slide:
                    break
                for ext in extensions:
                    candidate = lesson_dir / (pattern + ext)
                    if candidate.exists():
                        found_slide = candidate
                        break
        
            if not found_slide:
                break  # больше слайдов нет
        
            try:
                # ✅ Сохраняем ТОЛЬКО относительный путь от lessons/
                if str(found_slide.resolve()).startswith(str(self.lessons_dir.resolve())):
                    rel_path = str(found_slide.relative_to(self.lessons_dir))
                    slides.append(rel_path)
                    debug_log(f"✅ Найден слайд {idx}: {found_slide.name} -> {rel_path}")
                else:
                    debug_log(f"⚠️ Слайд вне папки lessons, пропущен: {found_slide}")
                    # НЕ добавляем — это нарушает безопасность
            except Exception as e:
                debug_log(f"❌ Ошибка при вычислении относительного пути для {found_slide}: {e}")
        
            idx += 1  # ← всегда увеличиваем, чтобы избежать зацикливания
    
        debug_log(f"📊 Найдено слайдов для урока {base_name}: {len(slides)}")
        return slides

    def _load_dialogue_knowledge(self) -> Dict:
        """Загрузка расширенной базы диалоговых шаблонов"""
        try:
            dialogue_path = Path("knowledge/dialogue_knowledge.json")
            if dialogue_path.exists():
                debug_log(f"📂 Загрузка dialogue_knowledge из: {dialogue_path}")
                with open(dialogue_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    debug_log(f"✅ dialogue_knowledge загружен, размер: {len(str(data))} байт")
                    return data
            else:
                debug_log(f"⚠️ Файл {dialogue_path} не найден, использую шаблоны по умолчанию")
        except Exception as e:
            debug_log(f"❌ Ошибка загрузки диалоговых шаблонов: {e}")
        
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
                "что преподаешь": ["У меня есть уроки по разным предметы! Что хочешь изучить?"]
            },
            "metadata": {
                "version": "1.0",
                "type": "default_dialogue_patterns"
            }
        }

    def _load_lessons(self):
        """Загрузка уроков по классам ВКЛЮЧАЯ ВЗРОСЛЫХ"""
        self._lessons = {}
        self._lessons_by_class = {}
        
        try:
            # Создаем папки если их нет (только при первой загрузке)
            for folder in [self.demo_lessons_dir, self.students_base_dir, self.generated_lessons_dir]:
                folder.mkdir(parents=True, exist_ok=True)
            
            # 1. Загружаем демо-уроки
            self._load_lessons_from_dir(self.demo_lessons_dir, "demo")
            
            # 2. Загружаем уроки для учеников ПО КЛАССАМ (ВКЛЮЧАЯ adult)
            if self.students_base_dir.exists():
                for class_dir in self.students_base_dir.iterdir():
                    if class_dir.is_dir():
                        class_name = class_dir.name
                        
                        if "_class" in class_name:
                            # Школьные классы (5_class, 6_class и т.д.)
                            class_level = class_name.replace("_class", "")
                            if class_level in self.available_classes:
                                self._load_student_lessons_by_class(class_dir, class_level)
                        elif class_name == "adult_language":
                            # 🔥 НОВОЕ: Уроки для взрослых по языкам
                            self._load_adult_language_lessons(class_dir)
            
            # 3. Загружаем сгенерированные уроки
            self._load_lessons_from_dir(self.generated_lessons_dir, "generated")
            
            # 4. Загрузка LEGACY уроков (рекурсивно)
            self._load_legacy_lessons()
            
            debug_log(f"✅ Уроки загружены: {sum(len(v) for v in self._lessons.values())} уроков")
            debug_log(f"✅ Классы с уроками: {list(self._lessons_by_class.keys())}")
                    
        except Exception as e:
            debug_log(f"Ошибка доступа к папке уроков: {e}")

    def _load_adult_language_lessons(self, adult_lang_dir: Path):
        """🔥 НОВЫЙ МЕТОД: Загрузка уроков для взрослых по языкам и уровням"""
        debug_log(f"🎓 Загрузка уроков для взрослых из: {adult_lang_dir}")
        
        if not adult_lang_dir.exists():
            debug_log(f"⚠️ Папка для взрослых не существует: {adult_lang_dir}")
            return
        
        # Проходим по всем уровням CEFR
        for level_dir in adult_lang_dir.iterdir():
            if level_dir.is_dir() and "_english" in level_dir.name:
                level = level_dir.name.replace("_english", "")
                
                # Добавляем уровень в список доступных классов
                if "adult" not in self.lessons_by_class:
                    self._lessons_by_class["adult"] = {}
                
                subject = "английский язык"
                
                if subject not in self._lessons_by_class["adult"]:
                    self._lessons_by_class["adult"][subject] = []
                
                # Загружаем уроки для этого уровня
                for lesson_file in level_dir.glob("*.txt"):
                    try:
                        lesson_number = self._extract_lesson_number(lesson_file.stem)
                        lesson_title = self._format_lesson_title(lesson_file.stem)
                        
                        lesson_data = {
                            'id': f"adult_{level}_{lesson_file.stem}",
                            'title': lesson_title,
                            'file_path': lesson_file,
                            'type': 'adult_language',
                            'subject': subject,
                            'class_level': 'adult',
                            'lesson_number': lesson_number,
                            'full_path': f"students/adult_language/{level_dir.name}/{lesson_file.name}",
                            'cefr_level': level,
                            'target_language': 'english'
                        }
                        
                        # Добавляем в общий список
                        if subject not in self._lessons:
                            self._lessons[subject] = []
                        self._lessons[subject].append(lesson_data)
                        
                        # Добавляем в список для взрослых
                        self._lessons_by_class["adult"][subject].append(lesson_data)
                        
                        debug_log(f"🎓 Загружен урок для взрослых: {lesson_title} (уровень {level})")
                        
                    except Exception as e:
                        debug_log(f"Ошибка загрузки урока для взрослых {lesson_file}: {e}")
        
        debug_log(f"✅ Уроки для взрослых загружены")

    def _load_legacy_lessons(self):
        """Исправленная загрузка старых уроков (рекурсивный поиск)"""
        try:
            # РЕКУРСИВНЫЙ ПОИСК ВСЕХ TXT ФАЙЛОВ В lessons/
            for lesson_file in self.lessons_dir.glob("**/*.txt"):
                if not lesson_file.is_file():
                    continue
                    
                # Пропускаем уже загруженные из других папок
                if (lesson_file.parent == self.demo_lessons_dir or 
                    lesson_file.parent == self.generated_lessons_dir or
                    "students" in str(lesson_file.parent)):
                    continue
                
                debug_log(f"📂 Загрузка legacy урока: {lesson_file}")
                
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
                    
                    if subject not in self._lessons:
                        self._lessons[subject] = []
                    self._lessons[subject].append(lesson_data)
                    
                    # Добавляем в структуру по классам если класс определен
                    if class_level != "general":
                        if class_level not in self._lessons_by_class:
                            self._lessons_by_class[class_level] = {}
                        if subject not in self._lessons_by_class[class_level]:
                            self._lessons_by_class[class_level][subject] = []
                        self._lessons_by_class[class_level][subject].append(lesson_data)
                        
                except Exception as e:
                    debug_log(f"Ошибка загрузки legacy урока {lesson_file}: {e}")
                    
        except Exception as e:
            debug_log(f"Ошибка при поиске legacy уроков: {e}")

    def _load_student_lessons_by_class(self, class_dir: Path, class_level: str):
        """Загружает уроки для конкретного класса"""
        if class_level not in self._lessons_by_class:
            self._lessons_by_class[class_level] = {}
        
        debug_log(f"📂 Загрузка уроков для класса {class_level}...")
        
        # Предметы по классам
        subjects_by_class = {
            "5": ["математика", "география", "биология", "русский язык", "литература", 
                  "английский язык", "французский язык", "немецкий язык", "испанский язык", 
                  "история", "информатика"],
            "6": ["математика", "география", "биология", "русский язык", "литература",
                  "английский язык", "французский язык", "немецкий язык", "испанский язык",
                  "история", "обществознание", "информатика"],
            "7": ["алгебра", "геометрия", "физика", "география", "биология", "русский язык",
                  "литература", "английский язык", "французский язык", "немецкий язык", "испанский язык",
                  "история", "обществознание", "информатика"],
            "8": ["алгебра", "геометрия", "физика", "география", "биология", "русский язык",
                  "литература", "английский язык", "французский язык", "немецкий язык", "испанский язык",
                  "история", "обществознание", "информатика", "химия"],
            "9": ["алгебра", "геометрия", "физика", "география", "биология", "русский язык",
                  "литература", "английский язык", "французский язык", "немецкий язык", "испанский язык",
                  "история", "обществознание", "информатика", "химия"],
            "10": ["алгебра", "геометрия", "физика", "география", "биология", "русский язык",
                   "литература", "английский язык", "французский язык", "немецкий язык", "испанский язык",
                   "история", "обществознание", "информатика", "химия"],
            "11": ["алгебра", "геометрия", "физика", "география", "биология", "русский язык",
                   "литература", "английский язык", "французский язык", "немецкий язык", "испанский язык",
                   "история", "обществознание", "информатика", "химия"],
            "adult": ["английский язык"]  # 🔥 НОВОЕ: Для взрослых только английский
        }
        
        subjects = subjects_by_class.get(class_level, [])
        for subject in subjects:
            subject_dir = class_dir / subject
            if subject_dir.exists() and subject_dir.is_dir():
                self._load_lessons_from_subject_dir(subject_dir, subject, class_level, "student")
        
        debug_log(f"✅ Класс {class_level}: {sum(len(v) for v in self._lessons_by_class[class_level].values())} уроков")

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
                
                # 🔥 ДОПОЛНИТЕЛЬНЫЕ ПОЛЯ ДЛЯ ВЗРОСЛЫХ
                if class_level == "adult" and subject == "английский язык":
                    # Ищем уровень CEFR в имени папки
                    for parent in lesson_file.parents:
                        if "_english" in parent.name:
                            lesson_data['cefr_level'] = parent.name.replace("_english", "")
                            lesson_data['target_language'] = 'english'
                            break
                
                # Добавляем в общий список по предметам
                if subject not in self._lessons:
                    self._lessons[subject] = []
                self._lessons[subject].append(lesson_data)
                
                # Добавляем в список по классам
                if subject not in self._lessons_by_class[class_level]:
                    self._lessons_by_class[class_level][subject] = []
                self._lessons_by_class[class_level][subject].append(lesson_data)
                
                debug_log(f"✅ Загружен урок: {lesson_title} (класс {class_level}, предмет {subject})")
                
            except Exception as e:
                debug_log(f"Ошибка загрузки урока {lesson_file}: {e}")

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
                    'class_level': "general",
                    'lesson_number': lesson_number,
                    'full_path': f"{dir_path.name}/{lesson_file.name}"
                }
                
                if subject not in self._lessons:
                    self._lessons[subject] = []
                self._lessons[subject].append(lesson_data)
                
            except Exception as e:
                debug_log(f"Ошибка загрузки урока {lesson_file}: {e}")

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
        
        return 999

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
        
        # Используем маппинг из константа
        for eng, rus in self.subject_mapping.items():
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
        elif any(word in filename_lower for word in ['немецкий']):
            return "немецкий язык"
        elif any(word in filename_lower for word in ['испанский']):
            return "испанский язык"
        elif any(word in filename_lower for word in ['география']):
            return "география"
        elif any(word in filename_lower for word in ['информатика']):
            return "информатика"
        else:
            return "общее"

    def _load_lesson_content(self, lesson_file: Path) -> List[str]:
        """Загружает содержание урока из текстового файла с улучшенной очисткой"""
        try:
            debug_log(f"📖 Загрузка урока из файла: {lesson_file}")
            
            if not lesson_file.exists():
                debug_log(f"❌ Файл урока не существует: {lesson_file}")
                return ["Файл урока не найден. Попробуйте другой урок."]
                
            with open(lesson_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            debug_log(f"✅ Файл прочитан, длина: {len(content)} символов")
            
            # УЛУЧШЕННАЯ ОЧИСТКА СОДЕРЖАНИЯ
            content = self._clean_lesson_content(content)
            
            # Разбиваем на абзацы (по пустым строкам)
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            # Если абзацев нет, разбиваем на предложения
            if not paragraphs:
                debug_log("⚠️ Нет абзацев, разбиваем на предложения")
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
            
            debug_log(f"✅ Урок разбит на {len(paragraphs)} абзацев")
            
            if not paragraphs:
                debug_log("❌ Не удалось разбить урок на абзацев")
                return ["Содержание урока временно недоступно. Давайте поговорим на эту тему!"]
                
            return paragraphs
            
        except Exception as e:
            debug_log(f"❌ Ошибка загрузки содержания урока: {e}")
            return ["Ошибка загрузки урока. Попробуйте позже."]

    def _clean_lesson_content(self, content: str) -> str:
        """Очистка содержания урока от лишнего форматирования"""
        if not content:
            return content
        
        # 🔥 КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Умная очистка в зависимости от типа предмета
        if TECHNICAL_SUPPORT_ENABLED and self.current_subject:
            from technical_subjects import should_preserve_formatting
            if should_preserve_formatting(content, self.current_subject):
                debug_log(f"🎯 Технический предмет: сохраняем формулы в уроке")
                # Для технических предметов - минимальная очистка
                # Удаляем только маркеры форматирования
                content = re.sub(r'[#\*\_\~`]', '', content)
                # Сохраняем формулы и специальные символы
                content = re.sub(r'\n\s*\n', '\n\n', content)
                content = content.strip()
                return content
        
        # Для гуманитарных предметы - стандартная очистка
        # Удаляем маркеры форматирования
        content = re.sub(r'[\*\#]{1,}', '', content)
        content = re.sub(r'\-\-\-+', '', content)
        content = re.sub(r'\+\+\+', '', content)
        
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
        """Поиск ответа в диалоговых шаблонах с учетом контекста"""
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
        
        # Персонализируем только первое предложение в ответе
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
            # ЛЕНИВАЯ ИНИЦИАЛИЗАЦИЯ: инициализируем LLM при первом использовании
            _ = self.llm
            
            # Собираем контекст диалога
            context = self._get_conversation_context()
            
            # 🔥 УЛУЧШЕННЫЙ ПРОМПТ С ДАННЫМИ УЧЕНИКА И ТИПОМ ПРЕДМЕТА
            age = self.student_data.get('age', '12')
            level = self.student_data.get('education_level', '5')
            name = self.student_data.get('name', 'ученик')
            
            # 🔥 ОСОБЕННОСТИ ДЛЯ ВЗРОСЛЫХ
            adult_adjustment = ""
            if self.is_adult_student:
                adult_adjustment = f"\nВЗРОСЛЫЙ УЧЕНИК: Используй более сложную лексику, уважай жизненный опыт."
                if self.cefr_level:
                    adult_adjustment += f"\nУРОВЕНЬ ЯЗЫКА: {self.cefr_level} - {self.cefr_config.get('description', '')}"
                if self.adult_study_mode == 'anything':
                    adult_adjustment += "\nРЕЖИМ 'ИЗУЧАТЬ ЧТО УГОДНО': Отвечай на любые вопросы, обсуждай любые темы."
            
            # 🔥 АДАПТИРУЕМ ПРОМПТ ДЛЯ ТЕХНИЧЕСКИХ ПРЕДМЕТОВ
            subject_type_instructions = ""
            if TECHNICAL_SUPPORT_ENABLED and self.current_subject:
                if self.is_technical_subject:
                    subject_type_instructions = f"\nПРЕДМЕТ ТЕХНИЧЕСКИЙ: Используй формулы и научные обозначения! Сохраняй математические символы: =, +, -, ×, ÷, √, π, ∑, ∫ и т.д."
                elif self.subject_type == "natural_science":
                    subject_type_instructions = f"\nПРЕДМЕТ ЕСТЕСТВЕННОНАУЧНЫЙ: Используй научные термины, объясняй природные процессы!"
            
            system_prompt = f"""Ты - дружелюбный учитель для ученика {age} лет, {level} класс.

ОСОБЕННОСТИ УЧЕНИКА:
- Имя: {name}
- Возраст: {age} лет  
- Уровень: {level} класс
- Предмет: {self.current_subject or 'не выбран'}

{adult_adjustment}
{subject_type_instructions}

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
                def llm_callback(response, r_id):
                    if response:
                        limited_response = self._limit_response_length(
                            response, 
                            self.dialogue_settings.get("max_response_length", 3)
                        )
                        
                        # ПЕРСОНАЛИЗИРУЕМ ОТВЕТ ОТ LLM
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
                    # ПЕРСОНАЛИЗИРУЕМ ОТВЕТ
                    return self._add_personalization(limited_response)
                    
        except Exception as e:
            debug_log(f"Ошибка запроса к LLM для диалога: {e}")
        
        return self._get_subject_selection_prompt()

    def _get_subject_selection_prompt(self) -> Optional[str]:
        """Возвращает предложение выбора предмета с учетом кд"""
        current_time = time.time()
        if current_time - self.last_subject_prompt_time < self.subject_prompt_cooldown:
            return None
        
        self.last_subject_prompt_time = current_time
        
        # 🔥 ДЛЯ ВЗРОСЛОГО В РЕЖИМЕ "ИЗУЧАТЬ ЧТО УГОДНО"
        if self.is_adult_student and self.adult_study_mode == 'anything':
            return None  # Не предлагаем предметы в этом режиме
        
        # 🔥 ДЛЯ ВЗРОСЛОГО В РЕЖИМЕ "АНГЛИЙСКИЙ"
        if self.is_adult_student and self.adult_study_mode == 'language':
            return f"Добро пожаловать на урок английского языка уровня {self.cefr_level}! Скажите 'начать урок', чтобы начать."
        
        # 🔥 ДЛЯ УЧЕНИКА: показываем предметы его класса
        if self.has_student_data and self.student_data.get('education_level'):
            student_class = self.student_data.get('education_level')
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
        
        # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: НИКОГДА не добавляем предложение выбора во время урока ИЛИ практики
        if self.lesson_started or self.practice_active:
            return original_response
        
        # Если это взрослый в режиме "изучать что угодно" - не предлагаем предметы
        if self.is_adult_student and self.adult_study_mode == 'anything':
            return original_response
        
        # Если есть данные ученика и выбран предмет, не предлагаем выбор
        if self.has_student_data and self.current_subject:
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
            if self.has_student_data:
                name = self.student_data.get('name', 'ученик')
                age = self.student_data.get('age', '12')
                level = self.student_data.get('education_level', '5')
                
                # 🔥 СПЕЦИАЛЬНОЕ ПРИВЕТСТВИЕ ДЛЯ ВЗРОСЛЫХ
                if self.is_adult_student:
                    if self.adult_study_mode == 'anything':
                        return f"Здравствуйте, {name}! Я ваш AI-учитель. В режиме 'изучать что угодно' мы можем обсудить любую тему. О чем бы вы хотели поговорить?"
                    elif self.adult_study_mode == 'language':
                        return f"Здравствуйте, {name}! Добро пожаловать на урок английского языка уровня {self.cefr_level}. Готовы начать?"
                
                greeting_variants = [
                    f"Привет, {name}! Я твой виртуальный учитель. Рад видеть тебя! Ты в {level} классе, это отлично!",
                    f"Здравствуй, {name}! Я твой AI-репетитор. Вижу, ты учишься в {level} классе. Готов помочь тебе с учебой!",
                    f"Привет, {name}! Я твой цифровой преподаватель. {age} лет - отличный возраст для новых открытий! Давай сделаем обучение интересным!",
                    f"Здравствуй, {name}! Я твой персональный учитель. Рад познакомиться с учеником {level} класса! Готов к увлекательному уроку?"
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
                name = self.student_data.get('name', 'ученик')
                return f"Приятно познакомиться, {name}! Теперь давайте выберем предмет для изучения. Что вас интересует?"
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

    def generate_lesson_on_demand(self, topic: str, is_language: bool = False) -> Optional[dict]:
        """Генерирует урок по запрошенной теме с ПРОВЕРОЧНЫМИ ВОПРОСАМИ"""
        try:
            debug_log(f"🎯 Генерация урока по теме: {topic}")
            
            # ЛЕНИВАЯ ИНИЦИАЛИЗАЦИЯ: инициализируем LLM при первом использовании
            _ = self.llm
            
            # 🔥 ОБНОВЛЕННЫЙ: ПРОВЕРКА ДЛЯ ВЗРОСЛЫХ И CEFR
            use_cefr_prompt = False
            if (self.is_adult_student and 
                self.adult_study_mode == 'language' and 
                self.is_language_subject and
                self.cefr_level and
                LANGUAGE_SUPPORT_ENABLED):
                
                use_cefr_prompt = True
                debug_log(f"🎓 Используем CEFR промпт для взрослых: уровень {self.cefr_level}")
            
            # ОБНОВЛЕННЫЙ ПРОМПТ: Добавляем данные ученика и ТРЕБОВАНИЯ К ВОПРОСАМ
            age = self.student_data.get('age', '12')
            level = self.student_data.get('education_level', '5')
            name = self.student_data.get('name', 'ученик')
            
            # 🔥 УМНЫЙ ВЫБОР ПРОМПТА В ЗАВИСИМОСТИ ОТ ТИПА ПРЕДМЕТА
            system_prompt = ""
            
            # 1. ЕСЛИ ЭТО ЯЗЫКОВОЙ УРОК ДЛЯ ВЗРОСЛЫХ С CEFR
            if use_cefr_prompt and LANGUAGE_SUPPORT_ENABLED:
                system_prompt = create_bilingual_lesson_prompt_cefr(
                    topic=topic,
                    target_language=self.target_language,
                    cefr_level=self.cefr_level
                )
                debug_log(f"🎓 Используем CEFR промпт для уровня {self.cefr_level}")
            
            # 2. ЕСЛИ ЭТО ЯЗЫКОВОЙ УРОК (СТАРАЯ ЛОГИКА)
            elif is_language and LANGUAGE_SUPPORT_ENABLED:
                system_prompt = LanguageIntegration.create_bilingual_lesson_prompt(
                    topic=topic,
                    target_language=self.target_language,
                    level=self.language_level,
                    bilingual_ratio=self.bilingual_ratio
                )
                debug_log(f"🎯 Используем языковой промт для уровня {self.language_level}")
            
            # 3. ЕСЛИ ЭТО ТЕХНИЧЕСКИЙ ПРЕДМЕТ - ИСПОЛЬЗУЕМ ТЕХНИЧЕСКИЙ ПРОМПТ
            elif TECHNICAL_PROMPTS_ENABLED and self.current_subject and self.is_technical_subject:
                system_prompt = get_lesson_prompt(
                    subject=self.current_subject,
                    topic=topic,
                    level=level,
                    age=int(age) if age.isdigit() else 12
                )
                debug_log(f"🎯 Используем технический промпт для предмета: {self.current_subject}")
            
            # 4. ОБЩИЙ ПРОМПТ ДЛЯ ОСТАЛЬНЫХ ПРЕДМЕТОВ
            else:
                system_prompt = f"""Ты - эксперт по созданию образовательных материалов.

ВАЖНЫЕ ПАРАМЕТРЫ УЧЕНИКА:
- Имя: {name}
- Возраст: {age} лет
- Уровень образования: {level} класс
- Предмет: {self.current_subject or 'общее'}

🔥 ОЧЕНЬ ВАЖНОЕ ТРЕБОВАНИЕ:
- ВСТАВЬ 1-2 проверочных вопроса естественным образом в середине урока
- Вопросы должны проверять понимание ключевых понятий
- Вопросы должны быть частьи текста, а не выделены специально
- Используй естественные формулировки: "Как вы думаете...", "Попробуйте ответить...", "Как называется процесс..."

Примеры естественных вопросов:
- "Как вы думаете, почему это происходит?"
- "Как называется этот процесс?"
- "Что является результатом этой реакции?"
- "Можете назвать основные компоненты?"
- "Попробуйте объяснить это своими словами"

ВАЖНО: Вопросы должны органично вписываться в текст урока!

СОДЕРЖАТЕЛЬНЫЕ ТРЕБОВАНИЯ:
- Информативным и точным
- Разделен на логические абзацы (разделяй пустыми строками)
- Содержать практические примеры если уместно
- Быть увлекательным и интересным

ФОРМАТИРОВАНИЕ:
- Разделяй абзацы ДВУМЯ переводами строки (\\n\\n)
- Используй подходящий для возраста стиль изложения
- Объясняй сложные понятия простыми словами

Тема урока: '{topic}'

Возвращай только текст урока без дополнительных комментариев."""

            # Запрос к LLM с увеличенным количеством токенов
            lesson_content = self.llm._query_llm_api(
                prompt=f"Создай подробный образовательный урок на тему: '{topic}'. Урок должен быть понятным и структурированным, с проверочными вопросами.",
                context="",
                subject=self.current_subject or "общее",
                system_prompt=system_prompt,
                max_tokens=2500
            )
            
            if not lesson_content:
                debug_log("❌ Ошибка: LLM не вернул содержание урока")
                return None
            
            debug_log(f"✅ Получен контент урока, длина: {len(lesson_content)} символов")
            
            # 🔥 ДЛЯ ТЕХНИЧЕСКИХ ПРЕДМЕТОВ: проверяем наличие формул
            if TECHNICAL_SUPPORT_ENABLED and self.is_technical_subject:
                from technical_subjects import contains_formulas
                has_formulas = contains_formulas(lesson_content)
                if has_formulas:
                    debug_log("🎯 В сгенерированном уроке обнаружены формулы")
                    self.technical_symbols_preserved = True
            
            # Убедимся, что есть правильное разделение на абзацев
            if '\n\n' not in lesson_content:
                debug_log("⚠️ В ответе нет двойных переводов строк, добавляем...")
                sentences = re.split(r'(?<=[.!?])\s+', lesson_content)
                lesson_content = '\n\n'.join(sentences)
            
            # 🔥 ВЫБОР ПУТИ ДЛЯ СОХРАНЕНИЯ УРОКА
            lesson_id = ""
            lesson_path = None
            
            # Для взрослых с CEFR - сохраняем в специальную папку
            if use_cefr_prompt and self.is_adult_student:
                # Генерируем путь для взрослого урока
                import uuid
                lesson_id = f"adult_{self.cefr_level}_{topic.lower().replace(' ', '_')}_{int(time.time())}"
                level_dir = self.lessons_dir / "students" / "adult_language" / f"{self.cefr_level}_english"
                level_dir.mkdir(parents=True, exist_ok=True)
                
                # Находим следующий номер урока
                existing_lessons = list(level_dir.glob("lesson_*.txt"))
                lesson_number = len(existing_lessons) + 1
                
                filename = f"lesson_{lesson_number:02d}_{topic.lower().replace(' ', '_')[:30]}.txt"
                lesson_path = level_dir / filename
                
                debug_log(f"🎓 Сохраняем урок для взрослых: {lesson_path}")
            else:
                # Стандартное сохранение
                lesson_id = f"generated_{topic.lower().replace(' ', '_')}_{int(time.time())}"
                filename = f"{lesson_id}.txt"
                lesson_path = self.generated_lessons_dir / filename
            
            # Создаем папку если не существует
            lesson_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Записываем контент в файл
            with open(lesson_path, 'w', encoding='utf-8') as f:
                f.write(f"Урок по теме: {topic}\n\n")
                f.write(lesson_content)
            
            debug_log(f"✅ Файл урока создан: {lesson_path}")
            
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
                'full_path': str(lesson_path.relative_to(self.lessons_dir)),
                'is_language': is_language,
                'is_technical': self.is_technical_subject if TECHNICAL_SUPPORT_ENABLED else False,
                'subject_type': self.subject_type if TECHNICAL_SUPPORT_ENABLED else "general"
            }
            
            if is_language:
                lesson_data['target_language'] = self.target_language
                lesson_data['language_level'] = self.language_level
                lesson_data['bilingual_ratio'] = self.bilingual_ratio
            
            # 🔥 ДОПОЛНИТЕЛЬНЫЕ ПОЛЯ ДЛЯ ВЗРОСЛЫХ CEFR
            if use_cefr_prompt and self.is_adult_student:
                lesson_data['cefr_level'] = self.cefr_level
                lesson_data['target_language'] = self.target_language
                lesson_data['class_level'] = 'adult'
                lesson_data['type'] = 'adult_generated'
            
            if subject not in self.lessons:
                self.lessons[subject] = []
            self.lessons[subject].append(lesson_data)
            
            # Также добавляем в lessons_by_class для взрослых
            if self.is_adult_student and 'adult' not in self.lessons_by_class:
                self.lessons_by_class['adult'] = {}
            
            if self.is_adult_student and subject not in self.lessons_by_class['adult']:
                self.lessons_by_class['adult'][subject] = []
            
            if self.is_adult_student:
                self.lessons_by_class['adult'][subject].append(lesson_data)
            
            debug_log(f"✅ Урок успешно сгенерирован и добавлен в список: {lesson_id}")
            return lesson_data
            
        except Exception as e:
            debug_log(f"❌ Ошибка генерации урока: {e}")
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
                debug_log(f"Обнаружен существующий предмет: {subject}, пропускаем генерацию")
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
                    debug_log(f"🎯 Обнаружен запрос на генерацию урока по теме: '{topic}'")
                    
                    # 🔥 ПРОВЕРКА ДЛЯ ВЗРОСЛЫХ В РЕЖИМЕ "ИЗУЧАТЬ ЧТО УГОДНО"
                    if self.is_adult_student and self.adult_study_mode == 'anything':
                        # В этом режиме просто отвечаем на вопросы, не создаем структурированные уроки
                        debug_log("🎓 Взрослый в режиме 'изучать что угодно' - не создаем структурированный урок")
                        return False
                    
                    # ПРОВЕРЯЕМ, ЯЗЫКОВОЙ ЛИ ЭТО УРОК
                    is_language = False
                    if self.is_language_subject and self.target_language:
                        is_language = True
                        debug_log(f"🎯 Это языковой урок: {self.target_language}, уровень {self.language_level}")
                    
                    # ГЕНЕРИРУЕМ И СОХРАНЯЕМ УРОК
                    generated_lesson = self.generate_lesson_on_demand(topic, is_language=is_language)
                    if generated_lesson:
                        debug_log(f"✅ Урок успешно сгенерирован и сохранен: {generated_lesson['id']}")
                        
                        # КРИТИЧЕСКИ ВАЖНО: Добавляем урок в список доступных уроков
                        subject = self.current_subject or "общее"
                        if subject not in self.lessons:
                            self.lessons[subject] = []
                        
                        # Проверяем, нет ли уже такого урока
                        lesson_exists = any(lesson['id'] == generated_lesson['id'] for lesson in self.lessons[subject])
                        if not lesson_exists:
                            self.lessons[subject].append(generated_lesson)
                            debug_log(f"✅ Урок добавлен в список уроков по предмету: {subject}")
                        
                        # НАЧИНАЕМ УРОК - это запускает отображение
                        self._start_generated_lesson(generated_lesson)
                        return True
                    else:
                        debug_log("❌ Не удалось сгенерировать урок")
        return False

    def _start_generated_lesson(self, lesson_data: dict):
        """Начинает сгенерированный урок с гарантированным отображением"""
        try:
            debug_log(f"🚀 НАЧИНАЕМ сгенерированный урок: {lesson_data['title']}")
            
            self.current_subject = self.current_subject or "общее"
            self.selected_lesson = lesson_data
            self.lesson_started = True
            self.current_state = "lesson_reading"
            self.current_paragraph = 0
            
            # 🔥 УСТАНАВЛИВАЕМ ФЛАГИ ДЛЯ ТЕХНИЧЕСКИХ ПРЕДМЕТОВ
            if lesson_data.get('is_technical', False):
                self.is_technical_subject = True
                self.subject_type = lesson_data.get('subject_type', 'technical')
                debug_log(f"🎯 Начинаем технический урок: {self.current_subject}")
            
            # 🔥 УСТАНАВЛИВАЕМ ФЛАГИ ДЛЯ ВЗРОСЛЫХ И CEFR
            if lesson_data.get('cefr_level'):
                self.cefr_level = lesson_data.get('cefr_level')
                if LANGUAGE_SUPPORT_ENABLED:
                    self.cefr_config = get_cefr_level_config(self.cefr_level)
                debug_log(f"🎓 Начинаем урок для взрослых: уровень {self.cefr_level}")
            
            # ЕСЛИ ЭТО ЯЗЫКОВОЙ УРОК - УСТАНАВЛИВАЕМ НАСТРОЙКИ
            if lesson_data.get('is_language', False):
                self.is_language_subject = True
                self.target_language = lesson_data.get('target_language', 'english')
                self.language_level = lesson_data.get('language_level', 'beginner')
                self.bilingual_ratio = lesson_data.get('bilingual_ratio', 0.3)
                debug_log(f"🎯 Начинаем языковой урок: {self.target_language}, уровень {self.language_level}")
            
            # ВКЛЮЧАЕМ АВТОМАТИЧЕСКУЮ ВИЗУАЛИЗАЦИЮ
            self.enable_visualization()
            
            # Загружаем содержание урока
            debug_log(f"📖 Загрузка содержания урока из: {lesson_data['file_path']}")
            self.lesson_content = self._load_lesson_content(lesson_data['file_path'])
            
            if not self.lesson_content:
                debug_log("❌ Не удалось загрузить содержание урока")
                return
            
            debug_log(f"✅ Урок загружен, количество абзацев: {len(self.lesson_content)}")
            
            # 🔥 НОВОЕ: Поиск слайдов для урока
            self.lesson_slides = self._find_lesson_slides(lesson_data['file_path'])
            debug_log(f"✅ Найдено слайдов для урока: {len(self.lesson_slides)}")
            
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
                    'is_generated': True,
                    'is_language': lesson_data.get('is_language', False),
                    'target_language': self.target_language if lesson_data.get('is_language') else None,
                    'is_technical': lesson_data.get('is_technical', False),
                    'subject_type': lesson_data.get('subject_type', 'general'),
                    'cefr_level': self.cefr_level,
                    'is_adult': self.is_adult_student,
                    'slides_count': len(self.lesson_slides)  # 🔥 НОВОЕ: отправляем количество слайдов
                }, room=self.room_id)
                debug_log(f"📢 Уведомление о начале урока отправлено в комнату {self.room_id}")
            
            debug_log(f"🎉 Сгенерированный урок '{lesson_data['title']}' успешно начат и отображается!")
            
        except Exception as e:
            debug_log(f"❌ Ошибка начала сгенерированного урока: {e}")
            self.lesson_started = False

    def _has_visualization_triggers(self, text: str) -> bool:
        """Проверяет наличие триггеров для визуализации - ТОЛЬКО SVG"""
        text_lower = text.lower()
        
        # 🔥 УСОВЕРШЕНСТВОВАННЫЕ ТРИГГЕРЫ ДЛЯ ТЕХНИЧЕСКИХ ПРЕДМЕТОВ
        visualization_triggers = [
            'структура', 'схема', 'диаграмма', 'график', 'процесс', 
            'алгоритм', 'иерархия', 'взаимосвязь', 'соотношение',
            'таблица', 'классификация', 'этапы', 'стадии', 'система',
            'сравнение', 'типы', 'виды', 'формы', 'принципы', 'компоненты',
            # Технические триггеры
            'формула', 'уравнение', 'теорема', 'доказательство', 'вычисление',
            'эксперимент', 'реакция', 'процесс', 'механизм', 'модель',
            'диаграмма', 'график', 'координаты', 'ось', 'параметр'
        ]
        
        structure_indicators = [
            'состоит из', 'включает в себя', 'делится на', 'подразделяется',
            'можно разделить', 'выделяют', 'различают', 'существуют',
            'основные элементы', 'ключевые аспекты',
            # Технические индикаторы
            'формула имеет вид', 'уравнение записывается', 'теорема гласит',
            'согласно закону', 'из этого следует', 'отсюда получаем'
        ]
        
        # 🔥 СПЕЦИАЛЬНЫЕ ПРОВЕРКИ ДЛЯ ТЕХНИЧЕСКИХ ПРЕДМЕТОВ
        if TECHNICAL_SUPPORT_ENABLED and self.is_technical_subject:
            # Для технических предметов всегда генерируем визуализацию для формул
            formula_patterns = [
                r'\b[a-zA-Z]\s*=\s*',  # x = 
                r'\b[a-zA-Z]\s*\+\s*[a-zA-Z]',  # a + b
                r'\b[a-zA-Z]\s*/\s*[a-zA-Z]',  # a / b
                r'\b[a-zA-Z]\s*\^\s*\d',  # x^2
                r'\bH₂O|CO₂|NaCl|H₂SO₄\b',  # Химические формулы
                r'∑|∫|∂|∇|∞|√|≠|≈|≡|≤|≥|π|θ|α|β|γ',  # Математические символы
            ]
            
            for pattern in formula_patterns:
                if re.search(pattern, text):
                    debug_log(f"🎯 Технический триггер: формула обнаружена")
                    return True
        
        has_trigger = any(trigger in text_lower for trigger in visualization_triggers)
        has_structure = any(indicator in text_lower for indicator in structure_indicators)
        
        return has_trigger or has_structure

    def _generate_visualization(self, text: str, context: str = ""):
        """🔥 ИСПРАВЛЕННЫЙ МЕТОД: Генерация SVG инфографики для текста"""
        if not self.visualization_enabled or not text.strip() or not self.room_id:
            return
    
        # Ограничиваем частоту визуализаций
        current_time = time.time()
        if current_time - self.last_visualization_time < self.visualization_cooldown:
            return
        
        self.last_visualization_time = current_time
        self.visualization_counter += 1
        
        debug_log(f"🎨 Генерация визуализации №{self.visualization_counter} для: {text[:50]}...")
        
        # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Используем generate_infographic вместо query_llm
        try:
            result = self.llm.generate_infographic(text, context)
            
            svg_code = ""
            if result and result.get("success"):
                svg_code = result.get("svg_code", "")
                debug_log(f"✅ SVG код получен через generate_infographic, длина: {len(svg_code)} символов")
            else:
                # Fallback на базовую визуализацию
                svg_code = generate_svg_code(text, context)
                debug_log(f"⚠️ Используем fallback SVG, длина: {len(svg_code)} символов")
            
            # 🔥 Отправляем клиенту ОДИН РАЗ, через событие
            self.socketio.emit('visualization_generated', {
                'room_id': self.room_id,
                'topic': text[:100],
                'svg_code': svg_code,
                'timestamp': time.time(),
                'type': 'technical' if self.is_technical_subject else 'infographic',
                'subject_type': self.subject_type,
                'is_technical': self.is_technical_subject,
                'cefr_level': self.cefr_level,
                'is_adult': self.is_adult_student
            }, room=self.room_id)
            
            debug_log(f"✅ SVG визуализация отправлена в комнату {self.room_id}")
            
        except Exception as e:
            debug_log(f"❌ Ошибка генерации SVG визуализации: {e}")
            
            # Fallback при ошибке
            fallback_svg = generate_svg_code(text, "Ошибка генерации")
            self.socketio.emit('visualization_generated', {
                'room_id': self.room_id,
                'topic': text[:100],
                'svg_code': fallback_svg,
                'timestamp': time.time(),
                'type': 'error_fallback',
                'is_technical': self.is_technical_subject,
                'cefr_level': self.cefr_level
            }, room=self.room_id)

    def enable_visualization(self):
        """Включение автоматической визуализации - ТОЛЬКО SVG"""
        self.visualization_enabled = True
        debug_log("✅ Автоматическая SVG визуализация включена")

    def disable_visualization(self):
        """Выключение автоматической визуализации"""
        self.visualization_enabled = False
        debug_log("❌ Автоматическая визуализация выключена")

    def _start_student_lesson(self) -> str:
        """🔥 НОВЫЙ МЕТОД: Быстрый старт урока для ученика (аналог демо-комнаты)"""
        if not self.has_student_data or not self.current_subject:
            return "Нет данных ученика или предмета"
        
        if not self.selected_lesson:
            # Находим урок
            next_lesson = self.get_next_lesson_for_student(self.current_subject)
            if not next_lesson:
                return "Нет доступных уроков"
            self.selected_lesson = next_lesson
        
        debug_log(f"🚀 БЫСТРЫЙ СТАРТ УРОКА ДЛЯ УЧЕНИКА: {self.selected_lesson['title']}")
        
        # Начинаем урок ТОЧНО КАК В ДЕМО-КОМНАТАХ
        self.lesson_started = True
        self.current_state = "lesson_reading"
        self.current_paragraph = 0
        
        # 🔥 Загружаем содержание
        self.lesson_content = self._load_lesson_content(self.selected_lesson['file_path'])
        if not self.lesson_content:
            self.lesson_started = False
            return "Ошибка загрузки урока"
        
        # 🔥 НОВОЕ: Поиск слайдов для урока
        self.lesson_slides = self._find_lesson_slides(self.selected_lesson['file_path'])
        debug_log(f"✅ Найдено слайдов для урока: {len(self.lesson_slides)}")
        
        # Инициализируем базу знаний
        if self.current_subject:
            from knowledge.knowledge_base import KnowledgeBase
            self.knowledge_base = KnowledgeBase(self.current_subject)
        
        # Очищаем историю
        self.conversation_history = []
        self.conversation_context = []
        
        # Получаем первый абзац
        first_paragraph = self._get_next_paragraph()
        
        # Уведомляем клиент
        if self.room_id and self.socketio:
            lesson_data = {
                'lesson_id': self.selected_lesson['id'],
                'title': self.selected_lesson['title'],
                'subject': self.current_subject,
                'class_level': self.selected_lesson.get('class_level', 'general'),
                'lesson_number': self.selected_lesson.get('lesson_number'),
                'is_student_lesson': True,
                'is_technical': self.is_technical_subject,
                'subject_type': self.subject_type,
                'slides_count': len(self.lesson_slides)  # 🔥 НОВОЕ: отправляем количество слайдов
            }
            
            # 🔥 ДОБАВЛЯЕМ CEFR ДЛЯ ВЗРОСЛЫХ
            if self.is_adult_student and self.cefr_level:
                lesson_data['cefr_level'] = self.cefr_level
                lesson_data['is_adult'] = True
            
            self.socketio.emit('lesson_started', lesson_data, room=self.room_id)
        
        student_name = self.student_data.get('name', 'ученик')
        return f"{student_name}, начинаем урок по {self.current_subject}. {first_paragraph}"

    def process_input(self, text: str) -> Optional[str]:
        """🔥 ИСПРАВЛЕННАЯ ОБРАБОТКА ВХОДЯЩЕГО ТЕКСТА С УЧЕТОМ ДАННЫХ УЧЕНИКА"""
        # 🔥 КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: ЗАГРУЖАЕМ УРОКИ ТОЛЬКО ПРИ НЕОБХОДИМОСТИ
        # self._ensure_lessons_loaded() - НЕ ЗАГРУЖАЕМ ЗДЕСЬ!
        
        text_lower = text.lower().strip()
        
        # 🔥 ПРИОРИТЕТ: Обработка взрослых в режиме "изучать что угодно"
        if self.is_adult_student and self.adult_study_mode == 'anything':
            return self._handle_adult_anything_mode(text)
        
        # 🔥 ПРИОРИТЕТ: Обработка взрослых в режиме "язык"
        if self.is_adult_student and self.adult_study_mode == 'language' and not self.lesson_started:
            # Проверяем команды начала урока
            if any(cmd in text_lower for cmd in ['начать урок', 'начнем', 'старт', 'готов', 'приступаем']):
                return self._start_adult_language_lesson()
        
        # 🔥 ОПРЕДЕЛЯЕМ ТИП ПРЕДМЕТА ЕСЛИ ЕЩЕ НЕ ОПРЕДЕЛЕН
        if self.current_subject and not self.subject_type:
            self._determine_subject_type()
        
        # ПЕРВОЕ: Проверяем, не нужен ли языковой урок
        if self.current_subject and not self.lesson_started:
            # Используем language_integration если доступен
            if LANGUAGE_SUPPORT_ENABLED:
                is_language = is_language_subject(self.current_subject)
            else:
                is_language = self._detect_language_subject_local(self.current_subject)
                
            if is_language and not self.is_language_subject:
                self.is_language_subject = True
                if LANGUAGE_SUPPORT_ENABLED:
                    lang_settings = get_language_settings(
                        self.current_subject, 
                        int(self.student_data.get('age', 12)) if self.has_student_data else 12
                    )
                    self.target_language = lang_settings.get('target_language', 'english')
                    self.language_level = lang_settings.get('level', 'beginner')
                    self.bilingual_ratio = lang_settings.get('bilingual_ratio', 0.3)
                else:
                    self.target_language = self._extract_target_language_local(self.current_subject)
                    self._set_student_language_settings()
                debug_log(f"🎯 Обнаружен языковой предмет: {self.current_subject}, язык: {self.target_language}")
        
        # ВТОРОЕ: Проверяем запрос на языковой урок
        if (self.is_language_subject and 
            not self.lesson_started and
            self._check_for_language_lesson_generation(text_lower)):
            return None  # Урок будет сгенерирован
        
        # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: ЕДИНАЯ ЛОГИКА СТАРТА УРОКА ДЛЯ ВСЕХ
        start_patterns = [
            'начать урок', 'начнем урок', 'начни урок', 'старт урока', 'приступаем', 'давай начнем',
            'готов начать', 'готов к уроку', 'поехали', 'вперед', 'можно начать', 'стартуем'
        ]
        
        # Проверяем команду начала урока
        if (self.current_subject and not self.lesson_started and 
            any(pattern in text_lower for pattern in start_patterns)):
            
            debug_log(f"🎯 КОМАНДА НАЧАЛА УРОКА: '{text_lower}', subject={self.current_subject}, selected_lesson={self.selected_lesson is not None}")
            
            # Для учеников
            if self.has_student_data:
                # Если урок не выбран, находим его
                if not self.selected_lesson:
                    debug_log("⚠️ Урок не выбран, ищем следующий...")
                    next_lesson = self.get_next_lesson_for_student(self.current_subject)
                    if next_lesson:
                        self.selected_lesson = next_lesson
                        debug_log(f"✅ Найден и выбран урок: {next_lesson['title']}")
                    else:
                        # Пробуем любой урок
                        lessons = self.get_lessons_for_student_subject(self.current_subject)
                        if lessons:
                            self.selected_lesson = lessons[0]
                            debug_log(f"✅ Выбран первый урок: {lessons[0]['title']}")
                        else:
                            return "Нет доступных уроков. Попробуйте выбрать другой предмет."
                
                # Запускаем урок
                return self._start_student_lesson()
            else:
                # Для обычных пользователей (как в демо-комнатах)
                return self._force_start_lesson()
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ 1: Обработка первого входа ученика
        if (self.has_student_data and 
            self.current_subject and 
            not self.selected_lesson and
            not self.lesson_started):
            
            # Если это первое сообщение ученика в комнате
            if any(word in text_lower for word in ['привет', 'здравствуй', 'начать', 'старт', 'готов', 'здравствуйте', 'хай']):
                # НЕМЕДЛЕННОЕ ПРЕДЛОЖЕНИЕ УРОКОВ (КАК В ДЕМО)
                return self.auto_suggest_lessons_for_student()
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ 3: Обработка выбора урока по номеру
        lesson_patterns = [
            (r'урок\s*(\d+)', int),  # "урок 1"
            (r'(\d+)\s*урок', int),  # "1 урок"
            (r'первый', lambda x: 1),
            (r'второй', lambda x: 2),
            (r'третий', lambda x: 3),
            (r'четвертый', lambda x: 4),
            (r'пятый', lambda x: 5),
            (r'шестой', lambda x: 6),
            (r'седьмой', lambda x: 7),
            (r'восьмой', lambda x: 8),
            (r'девятый', lambda x: 9),
            (r'десятый', lambda x: 10),
            (r'последний', lambda x: -1),  # "последний урок"
            (r'любой', lambda x: 0),  # "любой урок"
            (r'lesson\s*(\d+)', int)  # "lesson 1"
        ]
        
        if self.has_student_data and self.current_subject and not self.lesson_started:
            for pattern, converter in lesson_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    try:
                        if pattern == r'последний':
                            lesson_num = -1
                        elif pattern == r'любой':
                            lesson_num = 0
                        else:
                            lesson_num = converter(match.group(1) if match.groups() else 1)
                        return self._select_lesson_by_number(lesson_num)
                    except:
                        pass
        
        # РАСШИРЕННЫЙ СПИСОК КОМАНД ПРОДОЛЖЕНИЯ
        continue_commands = [
            "продолжай", "продолжить", "дальше", "следующий", "вперед", "давай дальше",
            "записал", "понял", "ясно", "ага", "угу", "хорошо", "ок", " ладно", "ясно",
            "готов", "можно дальше", "слушаю", "понятно", "ясно", "следующий вопрос"
        ]

        if self.lesson_started and any(cmd in text_lower for cmd in continue_commands):
            next_paragraph = self._get_next_paragraph()
            if next_paragraph:
                debug_log(f"✅ Команда продолжения обработана: '{text_lower}' -> следующий абзац")
                return next_paragraph
            else:
                debug_log("🏁 Урок завершен по команде продолжения")
                # 🔥 ИСПРАВЛЕНИЕ: ОТМЕЧАЕМ УРОК КАК ЗАВЕРШЕННЫЙ, НО НЕ СБРАСЫВАЕМ КОНТЕКСТ
                if self.selected_lesson and self.has_student_data:
                    self.mark_lesson_completed(self.selected_lesson)
                # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Запускаем практику, но НЕ сбрасываем контекст
                practice_message = self._start_practice_session()
                return practice_message
        
        self._add_to_conversation_history(text, is_user=True)
        
        generated_lesson = self._check_for_lesson_generation_intent(text_lower)
        if generated_lesson:
            return None
        
        available_subjects = self.get_available_subjects()
        for subject in available_subjects:
            if subject.lower() in text_lower and len(subject) > 3:
                debug_log(f"Обнаружен выбор предмета: {subject}")
                return self._handle_subject_selection_direct(subject)
        
        if self.practice_active and self.waiting_for_answer:
            return self._handle_practice_answer(text)
        
        dialogue_response = self._get_dialogue_response(text_lower)
        if dialogue_response:
            # ПЕРСОНАЛИЗИРОВАННЫЙ ОТВЕТ
            personalized_response = self._add_personalization(dialogue_response)
            final_response = self._add_subject_suggestion(personalized_response)
            if final_response:
                self._add_to_conversation_history(final_response, is_user=False)
                return final_response
        
        llm_response = self._handle_llm_dialogue(text)
        if llm_response:
            # ОТВЕТ УЖЕ ПЕРСОНАЛИЗИРОВАН В _handle_llm_dialogue
            final_response = self._add_subject_suggestion(llm_response)
            if final_response:
                self._add_to_conversation_history(final_response, is_user=False)
                return final_response
        
        fallback_response = self._get_contextual_fallback()
        if fallback_response:
            # ПЕРСОНАЛИЗИРОВАННЫЙ FALLBACK
            personalized_fallback = self._add_personalization(fallback_response)
            self._add_to_conversation_history(personalized_fallback, is_user=False)
            return personalized_fallback
        
        return None

    def _handle_adult_anything_mode(self, text: str) -> Optional[str]:
        """🔥 НОВЫЙ МЕТОД: Обработка диалога для взрослых в режиме 'изучать что угодно'"""
        debug_log(f"🎓 Взрослый в режиме 'изучать что угодно': {text}")
        
        # Просто передаем вопрос в LLM без структурированных уроков
        self._add_to_conversation_history(text, is_user=True)
        
        # Используем специальный промпт для взрослых
        age = self.student_data.get('age', '30')
        name = self.student_data.get('name', 'студент')
        
        system_prompt = f"""Ты - опытный преподаватель для взрослого студента.

ОСОБЕННОСТИ УЧЕНИКА:
- Имя: {name}
- Возраст: {age} лет
- Уровень: взрослый студент
- Режим: 'изучать что угодно' - обсуждаем любые темы

СТИЛЬ ОБЩЕНИЯ:
- Уважай жизненный опыт взрослого человека
- Давай глубокие, содержательные ответы
- Обсуждай сложные темы
- Будь профессиональным, но дружелюбным
- Используй примеры из реальной жизни

ОТВЕТЫ ДОЛЖНЫ БЫТЬ:
- Информативными и полезными
- Соответствующими возрасту и опыту
- Содержательными (3-5 предложений)
- На русском языке

Контекст разговора: {self._get_conversation_context()}"""
        
        llm_response = self.llm._query_llm_api(
            prompt=text,
            context=self._get_conversation_context(),
            subject="общее",
            system_prompt=system_prompt,
            max_tokens=300
        )
        
        if llm_response:
            self._add_to_conversation_history(llm_response, is_user=False)
            return llm_response
        
        return "Я готов обсудить с вами любую тему. Что вас интересует?"

    def _start_adult_language_lesson(self) -> str:
        """🔥 НОВЫЙ МЕТОД: Начало языкового урока для взрослых"""
        debug_log(f"🎓 Начало языкового урока для взрослых (уровень: {self.cefr_level})")
        
        if not self.has_student_data or not self.current_subject:
            return "Нет данных ученика или предмета"
        
        # Находим уроки для уровня CEFR
        student_class = 'adult'
        subject = 'английский язык'
        
        if student_class in self.lessons_by_class and subject in self.lessons_by_class[student_class]:
            # Фильтруем уроки по уровню CEFR
            all_lessons = self.lessons_by_class[student_class][subject]
            if self.cefr_level:
                filtered_lessons = [lesson for lesson in all_lessons if lesson.get('cefr_level') == self.cefr_level]
                if filtered_lessons:
                    lessons = filtered_lessons
                else:
                    lessons = all_lessons
            else:
                lessons = all_lessons
            
            if lessons:
                # Находим следующий незавершенный урок
                progress = self.get_student_progress(subject)
                completed_ids = progress.get('completed_lessons', [])
                
                for lesson in sorted(lessons, key=lambda x: x.get('lesson_number', 999)):
                    if lesson['id'] not in completed_ids:
                        self.selected_lesson = lesson
                        break
                
                if not self.selected_lesson:
                    # Все уроки завершены, берем первый
                    self.selected_lesson = lessons[0]
                
                debug_log(f"🎓 Выбран урок для взрослых: {self.selected_lesson['title']}")
                
                # Запускаем урок
                return self._start_student_lesson()
            else:
                return f"У меня пока нет уроков английского языка для уровня {self.cefr_level}. Хотите, чтобы я создал урок на определенную тему?"
        
        return f"У меня пока нет уроков английского языка для уровня {self.cefr_level}. Скажите 'хочу изучить [тема]', чтобы создать урок."

    def _determine_subject_type(self):
        """Определяет тип текущего предмета"""
        if not self.current_subject:
            return
            
        if TECHNICAL_SUPPORT_ENABLED:
            self.subject_type = get_subject_type(self.current_subject)
            self.is_technical_subject = (self.subject_type in ["technical", "natural_science"])
            debug_log(f"🎯 Определен тип предмета: {self.current_subject} -> {self.subject_type}")
        else:
            # Без технической поддержки - определяем локально
            subject_lower = self.current_subject.lower()
            
            # Простая логика определения
            technical_keywords = ['математика', 'алгебра', 'геометрия', 'физика', 'химия', 
                                  'информатика', 'программирование', 'инженерия', 'технология']
            science_keywords = ['биология', 'география', 'астрономия', 'экология', 'геология']
            language_keywords = ['английский', 'французский', 'немецкий', 'испанский', 'китайский', 'язык']
            
            if any(keyword in subject_lower for keyword in technical_keywords):
                self.subject_type = "technical"
                self.is_technical_subject = True
            elif any(keyword in subject_lower for keyword in science_keywords):
                self.subject_type = "natural_science"
                self.is_technical_subject = True
            elif any(keyword in subject_lower for keyword in language_keywords):
                self.subject_type = "language"
                self.is_language_subject = True
            else:
                self.subject_type = "humanitarian"
                self.is_technical_subject = False
            
            debug_log(f"🎯 Локальное определение типа предмета: {self.subject_type}")

    def _detect_language_subject_local(self, subject: str) -> bool:
        """Локальная детекция языкового предмета"""
        language_subjects = [
            'английский язык', 'english', 'английский',
            'французский язык', 'french', 'французский',
            'немецкий язык', 'german', 'немецкий',
            'испанский язык', 'spanish', 'испанский',
            'китайский язык', 'chinese', 'китайский',
            'итальянский язык', 'italian', 'итальянский'
        ]
        
        subject_lower = subject.lower()
        for lang_subj in language_subjects:
            if lang_subj in subject_lower:
                return True
        
        return False

    def _extract_target_language_local(self, subject: str) -> str:
        """Локальное извлечение целевого языка"""
        subject_lower = subject.lower()
        
        if 'английский' in subject_lower or 'english' in subject_lower:
            return 'english'
        elif 'французский' in subject_lower or 'french' in subject_lower:
            return 'french'
        elif 'немецкий' in subject_lower or 'german' in subject_lower:
            return 'german'
        elif 'испанский' in subject_lower or 'spanish' in subject_lower:
            return 'spanish'
        elif 'китайский' in subject_lower or 'chinese' in subject_lower:
            return 'chinese'
        elif 'итальянский' in subject_lower or 'italian' in subject_lower:
            return 'italian'
        else:
            return 'english'

    def _set_student_language_settings(self):
        """Устанавливает настройки языка на основе данных ученика"""
        if not self.has_student_data:
            return
            
        # ОПРЕДЕЛЯЕМ УРОВЕНЬ ЯЗЫКА НА ОСНОВЕ ВОЗРАСТА
        age = int(self.student_data.get('age', 12))
        if age <= 10:
            self.language_level = 'beginner'
            self.bilingual_ratio = 0.3  # 30% иностранного
        elif age <= 14:
            self.language_level = 'intermediate'
            self.bilingual_ratio = 0.5  # 50% иностранного
        else:
            self.language_level = 'advanced'
            self.bilingual_ratio = 0.7  # 70% иностранного
            
        debug_log(f"🎯 Установлены языковые настройки: уровень {self.language_level}, {int(self.bilingual_ratio*100)}% иностранного")

    def _check_for_language_lesson_generation(self, text_lower: str) -> bool:
        """Проверяет, нужен ли языковой урок"""
        if not self.current_subject:
            return False
            
        # Проверяем, это языковой предмет
        if not self.is_language_subject:
            if LANGUAGE_SUPPORT_ENABLED:
                self.is_language_subject = is_language_subject(self.current_subject)
            else:
                self.is_language_subject = self._detect_language_subject_local(self.current_subject)
                
        if not self.is_language_subject:
            return False
            
        # Извлекаем целевой язык
        if LANGUAGE_SUPPORT_ENABLED:
            self.target_language = get_language_settings(
                self.current_subject, 
                int(self.student_data.get('age', 12)) if self.has_student_data else 12
            ).get('target_language', 'english')
        else:
            self.target_language = self._extract_target_language_local(self.current_subject)
        
        # Устанавливаем настройки языка для ученика
        if not self.has_student_data:
            self._set_student_language_settings()
        elif LANGUAGE_SUPPORT_ENABLED:
            lang_settings = get_language_settings(
                self.current_subject,
                int(self.student_data.get('age', 12))
            )
            self.language_level = lang_settings.get('level', 'beginner')
            self.bilingual_ratio = lang_settings.get('bilingual_ratio', 0.3)
        
        # Проверяем шаблоны для генерации языкового урока
        patterns = [
            r'хочу изучить (.+)',
            r'урок по (.+)',
            r'тема (.+)',
            r'изучаем (.+)',
            r'начнем с (.+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                topic = match.group(1).strip()
                topic = re.sub(r'[.?]$', '', topic)
                if topic and len(topic) > 2:
                    debug_log(f"🎯 Запрос на языковой урок по теме: '{topic}'")
                    
                    # ИСПОЛЬЗУЕМ ПРАВИЛЬНЫЙ ПРОМТ ДЛЯ ЯЗЫКОВ
                    is_language = True
                    
                    # Генерируем урок
                    generated_lesson = self.generate_lesson_on_demand(
                        topic, 
                        is_language=is_language
                    )
                    
                    if generated_lesson:
                        return True
                        
        return False

    def auto_suggest_lessons_for_student(self) -> str:
        """УЛУЧШЕННЫЙ: Автоматически предлагает уроки по предмету ученика"""
        if not self.has_student_data or not self.current_subject:
            return None
        
        student_name = self.student_data.get('name', 'ученик')
        subject = self.current_subject
        student_class = self.student_data.get('education_level', '5')
        
        debug_log(f"🔥 Автопредложение уроков для {student_name} ({student_class} класс), предмет: {subject}")
        
        # 🔥 ОСОБАЯ ЛОГИКА ДЛЯ ВЗРОСЛЫХ
        if self.is_adult_student:
            if self.adult_study_mode == 'anything':
                return f"{student_name}, привет! Вы находитесь в режиме 'изучать что угодно'. Можете задавать любые вопросы на любые темы!"
            elif self.adult_study_mode == 'language':
                return f"{student_name}, привет! Добро пожаловать на урок английского языка уровня {self.cefr_level}. Скажите 'начать урок', чтобы начать!"
        
        # Находим уроки
        lessons = self.get_lessons_for_student_subject(subject)
        
        if not lessons:
            return f"{student_name}, привет! К сожалению, у меня пока нет уроков по {subject} для {student_class} класса."
        
        # 🔥 КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: ВЫБИРАЕМ СЛЕДУЮЩИЙ УРОК СРАЗУ
        next_lesson = self.get_next_lesson_for_student(subject)
        if not next_lesson:
            next_lesson = lessons[0]  # Берем первый
        
        # Устанавливаем выбранный урок СРАЗУ
        self.selected_lesson = next_lesson
        debug_log(f"✅ Урок автоматически выбран: {next_lesson['title']}")
        
        # Формируем простое и понятное сообщение КАК В ДЕМО-КОМНАТАХ
        response = f"{student_name}, привет! Я твой виртуальный учитель по {subject}. "
        response += f"Твой следующий урок: '{next_lesson['title']}'. "
        response += "Скажи 'начать урок' или 'готов начать', чтобы начать!"
        
        return response

    def _select_lesson_by_number(self, lesson_number: int) -> str:
        """Выбирает урок по номеру и СОХРАНЯЕТ его"""
        if not self.current_subject:
            return "Сначала выбери предмет."
        
        lessons = self.get_lessons_for_student_subject(self.current_subject)
        
        if not lessons:
            # Если нет уроков для этого класса, пробуем общие уроки по предмету
            lessons = self.lessons.get(self.current_subject, [])
        
        if not lessons:
            return "Нет доступных уроков."
        
        # Сортируем уроки по номеру
        sorted_lessons = sorted(lessons, key=lambda x: x.get('lesson_number', 999))
        
        # Обработка специальных номеров
        if lesson_number == -1:  # "последний урок"
            selected_lesson = sorted_lessons[-1]
        elif lesson_number == 0:  # "любой урок"
            selected_lesson = random.choice(sorted_lessons)
        elif 1 <= lesson_number <= len(sorted_lessons):
            selected_lesson = sorted_lessons[lesson_number - 1]
        else:
            return f"У меня есть только {len(sorted_lessons)} уроков. " \
                   f"Пожалуйста, выбери от 1 до {len(sorted_lessons)}."
        
        # Устанавливаем выбранный урок
        self.selected_lesson = selected_lesson
        
        # 🔥 ОПРЕДЕЛЯЕМ ТИП ПРЕДМЕТА ДЛЯ ВЫБРАННОГО УРОКА
        if TECHNICAL_SUPPORT_ENABLED:
            self.subject_type = get_subject_type(selected_lesson.get('subject', ''))
            self.is_technical_subject = (self.subject_type in ["technical", "natural_science"])
            debug_log(f"🎯 Для выбранного урока определен тип: {self.subject_type}")
        
        # 🔥 ОПРЕДЕЛЯЕМ CEFR ДЛЯ ВЗРОСЛЫХ
        if self.is_adult_student and selected_lesson.get('cefr_level'):
            self.cefr_level = selected_lesson.get('cefr_level')
            if LANGUAGE_SUPPORT_ENABLED:
                self.cefr_config = get_cefr_level_config(self.cefr_level)
            debug_log(f"🎓 Для выбранного урока определен CEFR уровень: {self.cefr_level}")
        
        # ПЕРСОНАЛИЗИРОВАННЫЙ ОТВЕТ
        student_name = self.student_data.get('name', 'ученик')
        return f"Отлично, {student_name}! Выбран урок: '{selected_lesson['title']}'. " \
               f"Скажи 'начать урок', чтобы начать."

    def _suggest_next_or_select_lesson(self) -> str:
        """Предлагает следующий урок или выбор урока"""
        if not self.has_student_data or not self.current_subject:
            return "Давайте выберем предмет для изучения!"
        
        student_name = self.student_data.get('name', 'ученик')
        
        # 🔥 ОСОБАЯ ЛОГИКА ДЛЯ ВЗРОСЛЫХ
        if self.is_adult_student:
            if self.adult_study_mode == 'anything':
                return f"{student_name}, вы в режиме 'изучать что угодно'. Можете задавать любые вопросы!"
            elif self.adult_study_mode == 'language':
                # ПОЛУЧАЕМ СЛЕДУЮЩИЙ УРОК ПО ПРЕДМЕТУ
                next_lesson = self.get_next_lesson_for_student(self.current_subject)
                
                if next_lesson:
                    progress = self.get_student_progress(self.current_subject)
                    completed_count = len(progress.get('completed_lessons', []))
                    total_lessons = len(self.get_lessons_for_student_subject(self.current_subject))
                    
                    response = f"{student_name}, отлично! "
                    response += f"Твой прогресс по английскому языку уровня {self.cefr_level}: {completed_count}/{total_lessons} уроков. "
                    response += f"Следующий урок: '{next_lesson['title']}'. Хочешь начать его?"
                    
                    # КРИТИЧЕСКО ВАЖНО: Сохраняем следующий урок как выбранный
                    self.selected_lesson = next_lesson
                    
                    return response
                else:
                    return f"{student_name}, ты уже завершил все уроки английского языка уровня {self.cefr_level}! Хочешь повторить какой-то урок?"
        
        # ПОЛУЧАЕМ СЛЕДУЮЩИЙ УРОК ПО ПРЕДМЕТУ
        next_lesson = self.get_next_lesson_for_student(self.current_subject)
        
        if next_lesson:
            # Есть следующий урок
            progress = self.get_student_progress(self.current_subject)
            completed_count = len(progress.get('completed_lessons', []))
            total_lessons = len(self.get_lessons_for_student_subject(self.current_subject))
            
            response = f"{student_name}, отлично! "
            response += f"Твой прогресс по {self.current_subject}: {completed_count}/{total_lessons} уроков. "
            response += f"Следующий урок: '{next_lesson['title']}'. Хочешь начать его?"
            
            # КРИТИЧЕСКО ВАЖНО: Сохраняем следующий урок как выбранный
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
        
        # 🔥 ОСОБАЯ ЛОГИКА ДЛЯ ВЗРОСЛЫХ
        if self.is_adult_student:
            if self.adult_study_mode == 'anything':
                return f"{student_name}, вы можете задавать любые вопросы на любые темы!"
            elif self.adult_study_mode == 'language':
                return f"{student_name}, хотите продолжить изучение английского языка уровня {self.cefr_level}?"
        
        # ПРЕДЛАГАЕМ ВЫБОР:
        options = [
            f"Хочешь начать следующий урок по {self.current_subject}?",
            f"Или выбрать конкретный урок по {self.current_subject}?",
            f"Может быть, повторить пройденный материал по {self.current_subject}?"
        ]
        
        return f"{student_name}, {random.choice(options)}"

    def _start_lesson_for_student(self) -> str:
        """Начинает урок для ученика по выбранному предмету"""
        debug_log(f"🚀 Начинаем урок для ученика по предмету: {self.current_subject}")
        
        # НОВОЕ: ПРОВЕРЯЕМ, ЕСТЬ ЛИ ВЫБРАННЫЙ УРОК
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
            
            # ГАРАНТИЯ: проверяем, что клиент знает о начале урока
            if self.room_id and self.socketio and self.selected_lesson:
                time.sleep(0.5)  # Небольшая задержка для надежности
                
                lesson_data = {
                    'lesson_id': self.selected_lesson['id'],
                    'title': self.selected_lesson['title'],
                    'subject': self.current_subject,
                    'class_level': self.selected_lesson.get('class_level', 'general'),
                    'lesson_number': self.selected_lesson.get('lesson_number'),
                    'is_student_lesson': True,
                    'is_technical': self.is_technical_subject,
                    'subject_type': self.subject_type,
                    'slides_count': len(self.lesson_slides)  # 🔥 НОВОЕ: отправляем количество слайдов
                }
                
                # 🔥 ДОБАВЛЯЕМ CEFR ДЛЯ ВЗРОСЛЫХ
                if self.is_adult_student and self.cefr_level:
                    lesson_data['cefr_level'] = self.cefr_level
                    lesson_data['is_adult'] = True
                
                self.socketio.emit('lesson_started', lesson_data, room=self.room_id)
                debug_log(f"📢 Уведомление 'lesson_started' отправлено в комнату {self.room_id}")
            
            return start_message
        
        return response

    def start_lesson_for_student_with_ready(self):
        """НОВЫЙ МЕТОД: Явно начинает урок для ученика когда все готово"""
        if not self.has_student_data or not self.current_subject:
            debug_log("❌ Нет данных ученика или предмета")
            return None
        
        debug_log(f"🚀 Явный старт урока для ученика: {self.current_subject}")
        
        # 🔥 ОСОБАЯ ЛОГИКА ДЛЯ ВЗРОСЛЫХ В РЕЖИМЕ "ИЗУЧАТЬ ЧТО УГОДНО"
        if self.is_adult_student and self.adult_study_mode == 'anything':
            return f"Вы находитесь в режиме 'изучать что угодно'. Можете задавать любые вопросы на любые темы!"
        
        # 1. Получаем следующий урок
        next_lesson = self.get_next_lesson_for_student(self.current_subject)
        
        if not next_lesson:
            # Если нет следующего урока, берем первый
            available_lessons = self.get_lessons_for_student_subject(self.current_subject)
            if not available_lessons:
                return f"У меня нет уроков по {self.current_subject} для твоего класса."
            next_lesson = available_lessons[0]
        
        debug_log(f"🎯 Найден урок: {next_lesson['title']}")
        
        # 2. Устанавливаем урок
        self.selected_lesson = next_lesson
        self.lesson_started = True
        self.current_state = "lesson_reading"
        
        # 3. Определяем тип предмета
        self._determine_subject_type()
        
        # 4. Определяем CEFR для взрослых
        if self.is_adult_student and next_lesson.get('cefr_level'):
            self.cefr_level = next_lesson.get('cefr_level')
            if LANGUAGE_SUPPORT_ENABLED:
                self.cefr_config = get_cefr_level_config(self.cefr_level)
            debug_log(f"🎓 Определен CEFR уровень: {self.cefr_level}")
        
        # 5. Загружаем содержание
        self.lesson_content = self._load_lesson_content(next_lesson['file_path'])
        self.current_paragraph = 0
        
        # 🔥 НОВОЕ: Поиск слайдов для урока
        self.lesson_slides = self._find_lesson_slides(next_lesson['file_path'])
        debug_log(f"✅ Найдено слайдов для урока: {len(self.lesson_slides)}")
        
        if not self.lesson_content:
            return "Ошибка загрузки урока."
        
        # 6. Инициализируем базу знаний
        if self.current_subject:
            from knowledge.knowledge_base import KnowledgeBase
            self.knowledge_base = KnowledgeBase(self.current_subject)
        
        # 7. Очищаем историю
        self.conversation_history = []
        self.conversation_context = []
        
        # 8. Обновляем прогресс ученика
        student_id = self.student_data.get('student_id')
        if student_id:
            self.save_student_progress(
                next_lesson['id'],
                next_lesson['subject'],
                completed=False
            )
        
        # 9. Возвращаем первый абзац
        first_paragraph = self._get_next_paragraph()
        
        student_name = self.student_data.get('name', 'ученик')
        return f"{student_name}, начинаем урок по {self.current_subject}. {first_paragraph}"

    def _handle_subject_selection_direct(self, subject: str) -> Optional[str]:
        """🔥 ИСПРАВЛЕННАЯ ЛОГИКА ВЫБОРА ПРЕДМЕТА ДЛЯ ВСЕХ КОМНАТ"""
        self.current_subject = subject
        
        debug_log(f"🎯 Выбор предмета: {subject}, данные ученика: {self.has_student_data}")
        
        # 🔥 ОПРЕДЕЛЯЕМ ТИП ПРЕДМЕТА
        self._determine_subject_type()
        
        # 🔥 ДЛЯ ВЗРОСЛЫХ В РЕЖИМЕ "ИЗУЧАТЬ ЧТО УГОДНО" - просто устанавливаем предмет
        if self.is_adult_student and self.adult_study_mode == 'anything':
            student_name = self.student_data.get('name', 'студент')
            return f"Отлично, {student_name}! Теперь вы можете задавать любые вопросы на любые темы. О чем бы вы хотели поговорить?"
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: ДЛЯ УЧЕНИКА - СРАЗУ ВЫБИРАЕМ УРОК И УСТАНАВЛИВАЕМ ЕГО
        if self.has_student_data:
            student_name = self.student_data.get('name', 'ученик')
            level = self.student_data.get('education_level', '5')
            
            # 🔥 ОСОБАЯ ЛОГИКА ДЛЯ ВЗРОСЛЫХ
            if level == 'adult' and subject == 'английский язык':
                # Для взрослых английский язык
                if not self.cefr_level:
                    # Определяем уровень CEFR если не установлен
                    if LANGUAGE_SUPPORT_ENABLED:
                        age = int(self.student_data.get('age', 25))
                        self.cefr_level = detect_cefr_level(
                            age, 
                            self.student_data.get('language_level', 'B1'),
                            'adult'
                        )
                        self.cefr_config = get_cefr_level_config(self.cefr_level)
                    else:
                        self.cefr_level = 'B1'  # Дефолтный уровень
                    
                    debug_log(f"🎓 Определен CEFR уровень для взрослого: {self.cefr_level}")
            
            # ВАЖНО: Ищем уроки для этого класса и предмета
            # Пробуем разные варианты названия предмета
            subject_variants = [subject]
            if subject in self.subject_mapping_reverse:
                subject_variants.append(self.subject_mapping_reverse[subject])
            if 'язык' in subject:
                subject_variants.append(subject.replace('язык', '').strip())
            
            debug_log(f"🔥 Поиск уроков для класса {level}, предмет {subject}")
            debug_log(f"🔥 Варианты названия предмета: {subject_variants}")
            
            found_lessons = False
            found_subject_variant = None
            
            for subject_variant in subject_variants:
                if (level in self.lessons_by_class and 
                    subject_variant in self.lessons_by_class[level]):
                    found_lessons = True
                    found_subject_variant = subject_variant
                    debug_log(f"🔥 Найдены уроки по предмету '{subject_variant}' для класса {level}")
                    break
            
            if not found_lessons:
                # ИЩЕМ УРОКИ ЧЕРЕЗ ПРЯМОЙ ПОИСК
                debug_log(f"🔥 Не найдено уроков через lessons_by_class, пробуем прямой поиск")
                direct_lessons = self._find_lessons_directly(level, subject)
                
                if direct_lessons:
                    found_lessons = True
                    # Добавляем найденные уроки в кэш
                    if level not in self.lessons_by_class:
                        self.lessons_by_class[level] = {}
                    self.lessons_by_class[level][subject] = direct_lessons
            
            if not found_lessons:
                # 🔥 ДЛЯ ВЗРОСЛЫХ С АНГЛИЙСКИМ - предлагаем генерацию урока
                if level == 'adult' and subject == 'английский язык':
                    return f"{student_name}, у меня пока нет готовых уроков английского для уровня {self.cefr_level}. Хотите, чтобы я создал для вас урок на определенную тему?"
                
                return f"{student_name}, у меня пока нет уроков по {subject} для {level} класса."
            
            # КРИТИЧЕСКО ВАЖНО: НАХОДИМ СЛЕДУЮЩИЙ УРОК И УСТАНАВЛИВАЕМ ЕГО
            next_lesson = self.get_next_lesson_for_student(subject)
            if not next_lesson:
                # Берем первый урок
                if level in self.lessons_by_class and subject in self.lessons_by_class[level]:
                    available_lessons = self.lessons_by_class[level][subject]
                    if available_lessons:
                        next_lesson = available_lessons[0]
            
            if not next_lesson:
                return f"{student_name}, не удалось найти урок по {subject}."
            
            # КРИТИЧЕСКИ ВАЖНО: Устанавливаем урок!
            self.selected_lesson = next_lesson
            debug_log(f"✅ Урок установлен для ученика: {next_lesson['title']}")
            
            # 🔥 ОПРЕДЕЛЯЕМ CEFR ДЛЯ ВЗРОСЛЫХ ИЗ УРОКА
            if self.is_adult_student and next_lesson.get('cefr_level'):
                self.cefr_level = next_lesson.get('cefr_level')
                if LANGUAGE_SUPPORT_ENABLED:
                    self.cefr_config = get_cefr_level_config(self.cefr_level)
                debug_log(f"🎓 Установлен CEFR уровень из урока: {self.cefr_level}")
            
            # Теперь урок готов к запуску по команде 'начать урок'
            progress = self.get_student_progress(subject)
            completed_count = len(progress.get('completed_lessons', []))
            
            if level in self.lessons_by_class and subject in self.lessons_by_class[level]:
                total_lessons = len(self.lessons_by_class[level][subject])
            else:
                total_lessons = 0
            
            if completed_count > 0:
                # 🔥 ОСОБЫЙ ТЕКСТ ДЛЯ ВЗРОСЛЫХ С CEFR
                if self.is_adult_student and self.cefr_level:
                    return f"{student_name}, отлично! Ваш прогресс по английскому языку уровня {self.cefr_level}: {completed_count}/{total_lessons} уроков. Скажите 'начать урок', чтобы начать урок!"
                else:
                    return f"{student_name}, отлично! Твой прогресс по {subject}: {completed_count}/{total_lessons} уроков. Скажи 'начать урок', чтобы начать урок!"
            else:
                # 🔥 ОСОБЫЙ ТЕКСТ ДЛЯ ВЗРОСЛЫХ С CEFR
                if self.is_adult_student and self.cefr_level:
                    return f"{student_name}, отлично! Это будет ваш первый урок английского языка уровня {self.cefr_level}. Тема: '{next_lesson['title']}'. Скажите 'начать урок', чтобы начать урок!"
                else:
                    return f"{student_name}, отлично! Это будет твой первый урок по {subject}. Тема: '{next_lesson['title']}'. Скажи 'начать урок', чтобы начать урок!"
        
        # ДЛЯ ОБЫЧНЫХ ПОЛЬЗОВАТЕЛЕЙ: используем ту же логику диалога, что и в демо-комнатах
        available_lessons = self._get_available_lessons(subject)
        
        if available_lessons:
            self.selected_lesson = available_lessons[0]
            debug_log(f"✅ Выбран существующий урок: {self.selected_lesson['title']}")
            
            # ИСПРАВЛЕНИЕ: Возвращаем предложение начать урок, как в демо-комнатах
            return f"Отлично! Я выбрал урок '{self.selected_lesson['title']}' по предмету {subject}. Когда будете готовы, скажите 'начать урок'!"
        else:
            # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Если уроков нет - создаем новый
            debug_log(f"⚠️ Урок по предмету '{subject}' не найден. Генерация урока 'на лету'...")
            
            # Генерируем урок по предмету
            generated_lesson = self.generate_lesson_on_demand(f"Введение в {subject}")
            
            if generated_lesson:
                self.selected_lesson = generated_lesson
                debug_log(f"✅ Сгенерирован новый урок: {generated_lesson['title']}")
                
                # ВОЗВРАЩАЕМ ТО ЖЕ ПРЕДЛОЖЕНИЕ, ЧТО И В ДЕМО-КОМНАТАХ
                return f"Я создал для вас урок по теме '{subject}'. Когда будете готовы, скажите 'начать урок'!"
            else:
                # Fallback на демо-урок, если генерация не удалась
                debug_log("❌ Генерация не удалась, создаем демо-урок")
                self.selected_lesson = self._create_demo_lesson(subject)
                
                return f"Я подготовил демо-урок по предмету {subject}. Когда будете готовы, скажите 'начать урок'!"

    def _force_start_lesson(self) -> str:
        """НОВЫЙ МЕТОД: Принудительно начинает выбранный урок"""
        if not self.selected_lesson:
            return "Сначала выберите урок!"
        
        if not self.current_subject:
            self.current_subject = self.selected_lesson.get('subject', 'общее')
        
        debug_log(f"🚀 ПРИНУДИТЕЛЬНЫЙ СТАРТ УРОКА: {self.selected_lesson['title']}")
        
        # 🔥 ОПРЕДЕЛЯЕМ ТИП ПРЕДМЕТА ПЕРЕД НАЧАЛОМ
        self._determine_subject_type()
        
        # 🔥 ОПРЕДЕЛЯЕМ CEFR ДЛЯ ВЗРОСЛЫХ
        if self.selected_lesson.get('cefr_level'):
            self.cefr_level = self.selected_lesson.get('cefr_level')
            if LANGUAGE_SUPPORT_ENABLED:
                self.cefr_config = get_cefr_level_config(self.cefr_level)
            debug_log(f"🎓 Установлен CEFR уровень из урока: {self.cefr_level}")
        
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
            
            # 🔥 НОВОЕ: Поиск слайдов для урока
            self.lesson_slides = self._find_lesson_slides(self.selected_lesson['file_path'])
            debug_log(f"✅ Найдено слайдов для урока: {len(self.lesson_slides)}")
            
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
                lesson_data = {
                    'lesson_id': self.selected_lesson['id'],
                    'title': self.selected_lesson['title'],
                    'subject': self.current_subject,
                    'is_generated': self.selected_lesson.get('type') == 'generated',
                    'is_language': self.selected_lesson.get('is_language', False),
                    'target_language': self.target_language if self.selected_lesson.get('is_language') else None,
                    'is_technical': self.is_technical_subject,
                    'subject_type': self.subject_type,
                    'slides_count': len(self.lesson_slides)  # 🔥 НОВОЕ: отправляем количество слайдов
                }
                
                # 🔥 ДОБАВЛЯЕМ CEFR ДЛЯ ВЗРОСЛЫХ
                if self.cefr_level:
                    lesson_data['cefr_level'] = self.cefr_level
                    lesson_data['is_adult'] = self.is_adult_student
                
                self.socketio.emit('lesson_started', lesson_data, room=self.room_id)
            
            # Персонализированное начало
            if self.has_student_data:
                student_name = self.student_data.get('name', '')
                name_prefix = f"{student_name}, " if student_name else ""
                # 🔥 ОСОБЫЙ ТЕКСТ ДЛЯ ВЗРОСЛЫХ С CEFR
                if self.is_adult_student and self.cefr_level:
                    return f"{name_prefix}Отлично! Начинаем урок английского языка уровня {self.cefr_level}. {first_paragraph}"
                else:
                    return f"{name_prefix}Отлично! Начинаем урок по {self.current_subject}. {first_paragraph}"
            else:
                return f"Отлично! Начинаем урок по {self.current_subject}. {first_paragraph}"
                
        except Exception as e:
            debug_log(f"❌ Ошибка начала урока: {e}")
            self.lesson_started = False
            return f"Ошибка начала урока: {str(e)}"

    def _get_available_lessons(self, subject: str) -> List[dict]:
        """ВОЗВРАЩАЕТ УРОКИ В ЗАВИСИМОСТИ ОТ НАЛИЧИЯ ДАННЫХ УЧЕНИКА"""
        # 🔥 ЛЕНИВАЯ ЗАГРУЗКА: загружаем уроки только при первом обращении
        if self._lessons is None:
            self._ensure_lessons_loaded()
            
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
        
        # Создаем папку если не существует
        lesson_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Простой демо-контент
        demo_content = f"""Демо-урок по предмету {subject}.

Это демонстрационный урок. В реальной системе здесь был бы полноценный учебный материал.

Для доступа ко всех функции системы зарегистрируйтесь как ученик!"""
        
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
            
            # ПЕРСОНАЛИЗИРОВАННОЕ ПРИВЕТСТВИЕ ДЛЯ УЧЕНИКА
            if self.has_student_data:
                student_name = self.student_data.get('name', 'ученик')
                age = self.student_data.get('age', '12')
                level = self.student_data.get('education_level', '5')
                
                # 🔥 ОСОБОЕ ПРИВЕТСТВИЕ ДЛЯ ВЗРОСЛЫХ
                if self.is_adult_student:
                    if self.adult_study_mode == 'anything':
                        return f"Здравствуйте, {student_name}! Вы находитесь в режиме 'изучать что угодно'. Можете задавать любые вопросы на любые темы!"
                    elif self.adult_study_mode == 'language':
                        return f"Здравствуйте, {student_name}! Добро пожаловать на урок английского языка уровня {self.cefr_level}. Готовы начать?"
                
                personalized_greetings = [
                    f"Привет, {student_name}! Я твой виртуальный учитель. Очень рад тебя видеть! Ты в {level} классе, это прекрасный возраст для учебы!",
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
            
        if any(word in text for word in ["да", "ага", 'угу', " ладно", "хорошо"]):
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
            
            # 🔥 ОБНОВЛЕНИЕ: Сбрасываем контекст практики тоже
            if self.practice_active:
                self._end_practice_session()
            
            # 🔥 НОВОЕ: Сбрасываем слайды
            self.lesson_slides = []
            
            # 🔥 СБРАСЫВАЕМ CEFR ДЛЯ ВЗРОСЛЫХ (но не is_adult_student)
            self.cefr_level = None
            self.cefr_config = None
            
            # ПЕРСОНАЛИЗИРОВАННОЕ СООБЩЕНИЕ ДЛЯ УЧЕНИКА
            if self.has_student_data:
                student_name = self.student_data.get('name', '')
                name_prefix = f"{student_name}, " if student_name else ""
                return f"{name_prefix}Урок остановлен. Скажи 'привет' когда захочешь продолжить."
            else:
                return "Урок остановлен. Скажите 'привет' когда захотите продолжить или выбрать новый урок."
            
        return None

    def _handle_practice_session(self, text: str) -> Optional[str]:
        if any(word in text for word in ["стоп", "останови", "хватит", "закончи"]):
            self._end_practice_session()
            
            # ПЕРСОНАЛИЗИРОВАННОЕ СООБЩЕНИЕ ДЛЯ УЧЕНИКА
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
        """🔥 ОБНОВЛЕННЫЙ МЕТОД: Получает следующий абзац урока с ОТПРАВКОЙ СЛАЙДА"""
        debug_log(f"📄 Получение следующего абзаца: текущий {self.current_paragraph}, всего {len(self.lesson_content)}")
        
        if self.current_paragraph < len(self.lesson_content):
            paragraph = self.lesson_content[self.current_paragraph]
            self.current_paragraph += 1
            
            # 🔥 НОВОЕ: Отправка слайда если есть
            slide_url = None
            if self.slides_enabled and self.lesson_slides:
                # Слайд для текущего абзаца: 2-й абзац → слайд[0], 3-й → слайд[1] и т.д.
                slide_index = self.current_paragraph - 2  # т.к. слайды начинаются со 2-го абзаца
                if 0 <= slide_index < len(self.lesson_slides):
                    slide_url = f"/lesson_slide?path={self.lesson_slides[slide_index]}"
                    debug_log(f"🖼️ Отправка слайда {slide_index+1} для абзаца {self.current_paragraph}: {slide_url}")
                    
                    # Отправляем событие со слайдом
                    if self.room_id and self.socketio:
                        self.socketio.emit('lesson_slide', {
                            'room_id': self.room_id,
                            'slide_url': slide_url,
                            'slide_index': slide_index,
                            'slide_number': slide_index + 1,
                            'total_slides': len(self.lesson_slides),
                            'paragraph_index': self.current_paragraph - 1,
                            'has_slide': True
                        }, room=self.room_id)
            
            # 🔥 НОВОЕ: Отправляем абзац с информацией о слайде
            if self.room_id and self.socketio:
                paragraph_data = {
                    'text': paragraph,
                    'paragraph_index': self.current_paragraph - 1,
                    'total_paragraphs': len(self.lesson_content),
                    'slide_url': slide_url,
                    'has_slide': slide_url is not None
                }
                
                # 🔥 ДОБАВЛЯЕМ CEFR ДЛЯ ВЗРОСЛЫХ
                if self.cefr_level:
                    paragraph_data['cefr_level'] = self.cefr_level
                
                self.socketio.emit('lesson_paragraph', paragraph_data, room=self.room_id)
            
            # Генерация визуализации (оставляем как есть)
            if (self.visualization_enabled and paragraph and 
                len(paragraph.strip()) > 10 and self.room_id):
                
                def delayed_visualization():
                    time.sleep(0.5)
                    context = " ".join(self.lesson_content[max(0, self.current_paragraph-2):self.current_paragraph])
                    self._generate_visualization(paragraph, context)
                
                threading.Thread(target=delayed_visualization, daemon=True).start()
            
            debug_log(f"✅ Возвращаем абзац {self.current_paragraph}: {paragraph[:100]}...")
            return paragraph
        else:
            debug_log("🏁 Урок завершен, запускаем практику")
            # 🔥 ИСПРАВЛЕНИЕ: ОТМЕЧАЕМ УРОК КАК ЗАВЕРШЕННЫЙ, НЕ СБРАСЫВАЕМ КОНТЕКСТ
            if self.selected_lesson and self.has_student_data:
                self.mark_lesson_completed(self.selected_lesson)
            
            # 🔥 НОВОЕ: Отправляем событие о завершении урока
            if self.room_id and self.socketio:
                lesson_completed_data = {
                    'room_id': self.room_id,
                    'lesson_id': self.selected_lesson['id'] if self.selected_lesson else None,
                    'title': self.selected_lesson['title'] if self.selected_lesson else None,
                    'total_paragraphs': len(self.lesson_content),
                    'total_slides': len(self.lesson_slides)
                }
                
                # 🔥 ДОБАВЛЯЕМ CEFR ДЛЯ ВЗРОСЛЫХ
                
            
            # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Запускаем практику, но НЕ сбрасываем контекст
            practice_message = self._start_practice_session()
            return practice_message

    def _start_practice_session(self) -> str:
        """🔥 ИСПРАВЛЕННЫЙ МЕТОД: Запускает фазу практики НЕ сбрасывая контекст урока"""
        self.lesson_started = False  # Урок прочитан
        self.current_state = "practice_session"
        self.practice_active = True
        self.waiting_for_answer = False
        self.current_question_index = 0  # СБРАСЫВАЕМ СЧЕТЧИК ВОПРОСОВ
        
        debug_log("=== ЗАПУСК ФАЗЫ ПРАКТИКИ ===")
        debug_log(f"📊 КОНТЕКСТ СОХРАНЕН: subject={self.current_subject}, lesson={self.selected_lesson['title'] if self.selected_lesson else 'None'}")
        debug_log(f"practice_active: {self.practice_active}, waiting_for_answer: {self.waiting_for_answer}")
        
        # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: НЕ СБРАСЫВАЕМ КОНТЕКСТ УРОКА!
        # self.selected_lesson = None        # ← НЕ СБРАСЫВАЕМ!
        # self.current_subject = None        # ← НЕ СБРАСЫВАЕМ!
        # self.lesson_content = []           # ← НЕ СБРАСЫВАЕМ!
        # self.current_paragraph = 0         # ← НЕ СБРАСЫВАЕМ!
        # self.lesson_slides = []            # ← НЕ СБРАСЫВАЕМ!
        
        # 🔥 НЕ СБРАСЫВАЕМ CEFR ДЛЯ ВЗРОСЛЫХ
        # self.cefr_level = None            # ← НЕ СБРАСЫВАЕМ!
        # self.cefr_config = None           # ← НЕ СБРАСЫВАЕМ!
        
        # ОБНОВЛЕНИЕ: Передаем данные ученика в менеджер практики
        if hasattr(self.practice_manager, 'student_data'):
            self.practice_manager.student_data = self.student_data
        
        # 🔥 ПЕРЕДАЕМ CEFR ДЛЯ ВЗРОСЛЫХ В МЕНЕДЖЕР ПРАКТИКИ
        if hasattr(self.practice_manager, 'cefr_level'):
            self.practice_manager.cefr_level = self.cefr_level
        if hasattr(self.practice_manager, 'cefr_config'):
            self.practice_manager.cefr_config = self.cefr_config
        
        # 🔥 ИСПРАВЛЕНИЕ: УНИФИЦИРОВАННАЯ ЛОГИКА ПРАКТИКИ ДЛЯ ВСЕХ ПРЕДМЕТОВ
        lesson_context = " ".join(self.lesson_content)
        debug_log(f"🎯 Запускаем унифицированную практику для: {self.current_subject}")
        
        # 🔥 ИСПРАВЛЕНИЕ: Всегда используем стандартную инициализацию практики
        # Без разделения на технические/гуманитарные
        self.practice_manager.initialize_practice_generation(lesson_context, self.current_subject)
        
        # 🔥 Установка флагов для менеджера практики (для адаптации промптов)
        if hasattr(self.practice_manager, 'is_technical_subject'):
            self.practice_manager.is_technical_subject = self.is_technical_subject
        if hasattr(self.practice_manager, 'subject_type'):
            self.practice_manager.subject_type = self.subject_type
        if hasattr(self.practice_manager, 'student_data'):
            self.practice_manager.student_data = self.student_data
        if hasattr(self.practice_manager, 'is_adult_student'):
            self.practice_manager.is_adult_student = self.is_adult_student
        if hasattr(self.practice_manager, 'cefr_level'):
            self.practice_manager.cefr_level = self.cefr_level
        
        # Уведомляем клиентов о начале практики
        if self.room_id:
            practice_data = {
                'room_id': self.room_id,
                'is_technical': self.is_technical_subject,
                'subject_type': self.subject_type,
                'lesson_title': self.selected_lesson['title'] if self.selected_lesson else None
            }
            
            # 🔥 ДОБАВЛЯЕМ CEFR ДЛЯ ВЗРОСЛЫХ
            if self.cefr_level:
                practice_data['cefr_level'] = self.cefr_level
            
            self.socketio.emit('practice_started', practice_data)
        
        # 🔥 ИСПРАВЛЕНИЕ: ВСЕГДА используем get_next_question() для всех предметов
        debug_log("🔄 Получение первого вопроса практики через get_next_question()...")
        first_question = self.practice_manager.get_next_question()
        
        if first_question:
            debug_log(f"✅ Первый вопрос получен: {first_question}")
            self.waiting_for_answer = True
            self.current_practice_question = {
                "id": 1,
                "question": first_question,
                "answer": ""
            }
            debug_log(f"📊 Установлен waiting_for_answer: {self.waiting_for_answer}")
            
            # ПЕРСОНАЛИЗИРОВАННОЕ СООБЩЕНИЕ ДЛЯ УЧЕНИКА
            if self.has_student_data:
                student_name = self.student_data.get('name', '')
                name_prefix = f"{student_name}, " if student_name else ""
                # 🔥 ОСОБЫЙ ТЕКСТ ДЛЯ ВЗРОСЛЫХ С CEFR
                if self.is_adult_student and self.cefr_level:
                    return f"{name_prefix}Отлично! Переходим к практике английского языка уровня {self.cefr_level}. Первый вопрос: {first_question}"
                else:
                    return f"{name_prefix}Отлично! Переходим к практике. Первый вопрос: {first_question}"
            else:
                return f"Отлично! Переходим к практике. Первый вопрос: {first_question}"
        else:
            debug_log("❌ Не удалось получить первый вопрос практики")
            self.practice_active = False
            return "Практические задания временно недоступны. Давайте продолжим урок или выберем другую тему."

    def _handle_practice_answer(self, text: str) -> str:
        """Обработка ответа ученика во время практики"""
        return self._evaluate_and_generate_next(text)

    def _evaluate_and_generate_next(self, student_answer: str) -> str:
        """Оценивает ответ и возвращает следующий вопрос с асинхронной генерацией"""
        debug_log(f"🔍 Обработка ответа: '{student_answer}'")
        debug_log(f"📊 Состояние: practice_active={self.practice_active}, waiting_for_answer={self.waiting_for_answer}")
        
        if not self.practice_active:
            debug_log("❌ Практика не активна")
            return "Практика не активна."
        
        # ПРОВЕРЯЕМ, НЕ ЯВЛЯЕТСЯ ЛИ ОТВЕТ КОМАНДОЙ
        if any(cmd in student_answer.lower() for cmd in ['продолжай', 'дальше', 'следующий']):
            debug_log(f"🔇 Игнорирую команду вместо ответа: {student_answer}")
            next_question = self.practice_manager.get_next_question()
            if next_question:
                # ПЕРСОНАЛИЗИРОВАННЫЙ ОТВЕТ
                if self.has_student_data:
                    student_name = self.student_data.get('name', '')
                    return f"{student_name}, это похоже на команду. Пожалуйста, дай ответ на вопрос. Следующий вопрос: {next_question}"
                else:
                    return f"Это похоже на команду. Пожалуйста, дайте ответ на вопрос. Следующий вопрос: {next_question}"
            else:
                self._end_practice_session()
                return "Практика завершена."
        
        debug_log(f"🎯 Оценка ответа и получение следующего вопроса...")
        
        current_question = self.current_practice_question
        if not current_question:
            debug_log("❌ Нет текущего вопроса практики")
            self._end_practice_session()
            return "Практика завершена."
        
        # УВЕЛИЧИВАЕМ СЧЕТЧИК ОТВЕТОВ ПЕРЕД ПРОВЕРКОЙ ЛИМИТА
        self.current_question_index += 1
        debug_log(f"📊 Текущий номер вопроса: {self.current_question_index}/{self.max_questions}")
        
        # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: ПРОВЕРЯЕМ ЛИМИТ ВОПРОСОВ ДО ГЕНЕРАЦИИ СЛЕДУЮЩЕГО
        if self.current_question_index >= self.max_questions:
            debug_log(f"🏁 Достигнут лимит вопросов: {self.current_question_index}/{self.max_questions}")
            self._end_practice_session()
            
            # ПЕРСОНАЛИЗИРОВАННОЕ СООБЩЕНИЕ ДЛЯ УЧЕНИКА
            if self.has_student_data:
                student_name = self.student_data.get('name', '')
                name_prefix = f"{student_name}, " if student_name else ""
                # 🔥 ОСОБЫЙ ТЕКСТ ДЛЯ ВЗРОСЛЫХ С CEFR
                if self.is_adult_student and self.cefr_level:
                    return f"{name_prefix}Отлично! Вы ответили на все {self.max_questions} вопросов практики английского языка уровня {self.cefr_level}. Урок завершен!"
                else:
                    return f"{name_prefix}Отлично! Ты ответил на все {self.max_questions} вопросов практики. Урок завершен!"
            else:
                return f"Отлично! Вы ответили на все {self.max_questions} вопросов практики. Урок завершен!"
        
        # 🔥 ИСПРАВЛЕНИЕ: ВСЕГДА используем evaluate_and_continue для всех предметов
        feedback, next_question = self.practice_manager.evaluate_and_continue(
            student_answer, 
            current_question["question"]
        )
        
        # АДАПТИРУЕМ ОБРАТНУЮ СВЯЗЬ ДЛЯ УЧЕНИКА
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
            debug_log(f"➡️ Следующий вопрос получен: {next_question[:80]}...")
            debug_log(f"📊 Вопросов задано: {self.current_question_index}/{self.max_questions}")
            debug_log(f"📊 Установлен waiting_for_answer: {self.waiting_for_answer}")
            return response
        else:
            debug_log("❌ Не удалось получить следующий вопрос")
            self._end_practice_session()
            return f"{feedback}. Практика завершена!"

    def _end_practice_session(self):
        """🔥 СТАРАЯ ЛОГИКА ПРАКТИКИ: Завершает сессию практики и НЕ сбрасывает контекст"""
        self.practice_active = False
        self.waiting_for_answer = False
        self.current_state = "greeting"
        self.current_question_index = 0  # СБРАСЫВАЕМ СЧЕТЧИК
        
        # 🔥 ВАЖНО: В старой версии мы НЕ сбрасываем данные урока и предмета
        # Они остаются доступными для продолжения
        # self.lesson_started = False    # ← НЕ сбрасываем!
        # self.selected_lesson = None    # ← НЕ сбрасываем!
        # self.current_subject = None    # ← НЕ сбрасываем!
        # self.lesson_content = []       # ← НЕ сбрасываем!
        # self.current_paragraph = 0     # ← НЕ сбрасываем!
        # self.lesson_slides = []        # ← НЕ сбрасываем!
        
        # 🔥 НЕ СБРАСЫВАЕМ CEFR ДЛЯ ВЗРОСЛЫХ
        # self.cefr_level = None        # ← НЕ сбрасываем!
        # self.cefr_config = None       # ← НЕ сбрасываем!
        
        # Останавливаем генерацию вопросов
        if hasattr(self.practice_manager, 'stop_async_generation'):
            self.practice_manager.stop_async_generation()
        
        if self.room_id:
            self.socketio.emit('practice_ended', {'room_id': self.room_id})
        debug_log("=== 🏁 ПРАКТИКА ЗАВЕРШЕНА (СТАРАЯ ЛОГИКА) ===")

    def handle_question_during_lesson(self, question: str) -> str:
        """ОБНОВЛЕННЫЙ: Обработка вопросов ученика во время урока с КОНТЕКСТУАЛЬНЫМ АНАЛИЗОМ"""
        if not question.strip():
            return "Повторите вопрос пожалуйста, я не расслышал."
            
        question_lower = question.lower().strip()
        
        if self.visualization_enabled:
            context = " ".join(self.lesson_content[max(0, self.current_paragraph-2):self.current_paragraph])
            self._generate_visualization(question, context)
        
        debug_log(f"Немедленная обработка вопроса: '{question}'")
        
        # КРИТИЧЕСКОЕ ОБНОВЛЕНИЕ: ВСЕГДА передаем контекст урока
        current_context = ""
        if self.lesson_content and self.current_paragraph > 0:
            # Берем последние 2-3 абзацы как контекст
            context_start = max(0, self.current_paragraph - 3)
            current_context = " ".join(self.lesson_content[context_start:self.current_paragraph])
        
        # 🔥 УНИВЕРСАЛЬНЫЙ ПРОМПТ ДЛЯ ВСЕХ ОТВЕТОВ С УЧЕТОМ ТИПА ПРЕДМЕТА
        technical_instructions = ""
        if TECHNICAL_SUPPORT_ENABLED and self.is_technical_subject:
            technical_instructions = f"\nПРЕДМЕТ ТЕХНИЧЕСКИЙ: Используй формулы и научные обозначения! Сохраняй математические символы и объясняй их."
        
        # 🔥 ДОБАВЛЕНИЕ ДЛЯ ВЗРОСЛЫХ И CEFR
        adult_instructions = ""
        if self.is_adult_student:
            adult_instructions = f"\nВЗРОСЛЫЙ УЧЕНИК: Уважай жизненный опыт, давай глубокие ответы."
            if self.cefr_level:
                adult_instructions += f"\nУРОВЕНЬ ЯЗЫКА: {self.cefr_level} - адаптируй сложность ответа."
        
        universal_prompt = f"""
КОНТЕКСТ УРОКА (последние абзацы): {current_context}

ОТВЕТ УЧЕНИКА: {question}

{technical_instructions}
{adult_instructions}

ИНСТРУКЦИИ ДЛЯ УЧИТЕЛЯ:
1. Проанализируй контекст урока и ответ ученика
2. ЕСЛИ ответ ученика является ответом на вопрос из урока:
   - Оцени правильность ответа
   - Дай краткий комментарий
   - Скажи "Продолжим урок" для перехода к следующему абзацу
3. ЕСЛИ это НЕ ответ на вопрос, а обычный вопрос ученика:
   - Ответь на вопрос по теме урока
   - Используй контекст для более точного ответа
   - НЕ говори "Продолжим урок" в этом случае

Верни только ответ учителя, без пояснений о своей логике.
"""
        
        final_response = None
        
        if self.llm_query_mode == "llm_first":
            debug_log(f"🔀 Режим llm_first: Обработка вопроса '{question}' с контекстом")
            
            llm_response = self.llm.query(universal_prompt, current_context, self.current_subject)
            if llm_response and not llm_response.startswith("Интересный вопрос!"):
                self.llm.add_to_cache(question, llm_response, self.current_subject)
                if self.knowledge_base and self._should_save_to_knowledge_base(question):
                    self.knowledge_base.add_llm_answer(question, llm_response)
                    self.knowledge_base.add_knowledge(question=question, answer=llm_response)
                    self.knowledge_base.add_to_dialogue_knowledge(question, llm_response)
                debug_log(f"✅ Ответ получен от LLM (режим llm_first): {llm_response[:100]}...")
                final_response = llm_response
            
            if not final_response and self.knowledge_base:
                knowledge_response = self.knowledge_base.get_dialogue_response(question_lower)
                if knowledge_response and not knowledge_response.startswith("Интересный вопрос!"):
                    debug_log(f"📚 Ответ найден в базе знаний после неудачи LLM: {knowledge_response[:100]}...")
                    final_response = knowledge_response
            
            if not final_response and self.knowledge_base:
                llm_answer = self.knowledge_base.find_llm_answer(question, threshold=0.8)
                if llm_answer:
                    debug_log(f"💾 Использован сохраненный ответ LLM: {llm_answer[:100]}...")
                    final_response = llm_answer
        
        else:
            debug_log(f"🔀 Режим traditional: Обработка вопроса '{question}'")
            
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
                    debug_log(f"💾 Использован сохраненный ответ LLМ для вопроса: {question}")
                    final_response = llm_answer
            
            if not final_response:
                # Используем универсальный промпт
                llm_response = self.llm.query(universal_prompt, current_context, self.current_subject)
                if llm_response:
                    self.llm.add_to_cache(question, llm_response, self.current_subject)
                    if self.knowledge_base and self._should_save_to_knowledge_base(question):
                        self.knowledge_base.add_llm_answer(question, llm_response)
                        self.knowledge_base.add_knowledge(question=question, answer=llm_response)
                        self.knowledge_base.add_to_dialogue_knowledge(question, llm_response)
                    final_response = llm_response
        
        if not final_response:
            # УЛУЧШЕННЫЙ FALLBACK с учетом контекста
            if "Продолжим урок" in current_context:
                final_response = "Хорошо, продолжим урок."
            else:
                final_response = "Интересный вопрос! Давайте обсудим его после завершения текущего материала, чтобы не отвлекаться."
        
        # ПЕРСОНАЛИЗИРУЕМ ОТВЕТ НА ВОПРОС
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
        if self._practice_manager is not None:
            self.practice_manager.reset()
        
        # Сброс данных ученика
        self.student_data = {}
        self.has_student_data = False
        self.is_adult_student = False
        self.adult_study_mode = None
        
        # 🔥 СБРОС CEFR ДЛЯ ВЗРОСЛЫХ
        self.cefr_level = None
        self.cefr_config = None
        
        # 🔥 СБРОС НАСТРОЕК ТИПОВ ПРЕДМЕТОВ
        self.is_technical_subject = False
        self.subject_type = "general"
        self.technical_symbols_preserved = False
        
        # СБРОС ЯЗЫКОВЫХ НАСТРОЕК
        self.is_language_subject = False
        self.target_language = 'english'
        self.language_level = 'beginner'
        self.bilingual_ratio = 0.3
        
        # 🔥 НОВОЕ: Сброс слайдов
        self.lesson_slides = []
        self.slides_enabled = True

    def get_available_subjects(self) -> List[str]:
        """ИСПРАВЛЕННАЯ ЛОГИКА: Возвращает доступные предметы"""
        # 🔥 ВАЖНО: Загружаем уроки ТОЛЬКО при первом обращении
        # НЕ загружаем в set_student_data или __init__
        if self._lessons is None:
            self._ensure_lessons_loaded()
        
        # 🔥 ДЛЯ ВЗРОСЛЫХ В РЕЖИМЕ "ИЗУЧАТЬ ЧТО УГОДНО" - пустой список
        if self.is_adult_student and self.adult_study_mode == 'anything':
            return []
        
        # 🔥 ДЛЯ ВЗРОСЛЫХ В РЕЖИМЕ "АНГЛИЙСКИЙ" - только английский
        if self.is_adult_student and self.adult_study_mode == 'language':
            return ['английский язык']
        
        # ДЛЯ УЧЕНИКА: предметы его класса
        if self.has_student_data and self.student_data.get('education_level'):
            student_class = self.student_data.get('education_level')
            if student_class in self.lessons_by_class:
                return list(self.lessons_by_class[student_class].keys())
        
        # Для обычных пользователей
        subjects = list(self.lessons.keys())
        
        # Всегда добавляем обществознание
        if "обществознание" not in subjects:
            subjects.append("обществознание")
            
        return subjects

    def get_lessons_for_subject(self, subject: str) -> List[dict]:
        """ИСПРАВЛЕННАЯ ЛОГИКА: Возвращает уроки по предмету"""
        return self._get_available_lessons(subject)

    def get_lessons_for_student_subject(self, subject: str) -> List[dict]:
        """ИСПРАВЛЕННЫЙ МЕТОД: Возвращает уроки по предмету для текущего ученика"""
        if not self.has_student_data:
            return []
        
        student_class = self.student_data.get('education_level', '5')
        
        debug_log(f"🔥 Поиск уроков: класс {student_class}, предмет '{subject}'")
        
        # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: ЛЕНИВАЯ ЗАГРУЗКА УРОКОВ
        if self._lessons is None:
            self._ensure_lessons_loaded()
        
        # 🔥 ОСОБАЯ ЛОГИКА ДЛЯ ВЗРОСЛЫХ
        if self.is_adult_student:
            if self.adult_study_mode == 'anything':
                return []  # Нет уроков в этом режиме
            elif self.adult_study_mode == 'language' and subject == 'английский язык':
                # Для взрослых английский язык - фильтруем по уровню CEFR
                if student_class in self.lessons_by_class and subject in self.lessons_by_class[student_class]:
                    all_lessons = self.lessons_by_class[student_class][subject]
                    # Если есть CEFR уровень - фильтруем по нему
                    if self.cefr_level:
                        filtered_lessons = [lesson for lesson in all_lessons if lesson.get('cefr_level') == self.cefr_level]
                        if filtered_lessons:
                            debug_log(f"🎓 Найдено уроков для взрослых уровня {self.cefr_level}: {len(filtered_lessons)}")
                            return sorted(filtered_lessons, key=lambda x: x.get('lesson_number', 999))
                        else:
                            # Если нет уроков для этого уровня, возвращаем все
                            debug_log(f"🎓 Нет уроков для уровня {self.cefr_level}, возвращаем все")
                            return sorted(all_lessons, key=lambda x: x.get('lesson_number', 999))
                    else:
                        # Нет CEFR уровня - возвращаем все уроки английского для взрослых
                        return sorted(all_lessons, key=lambda x: x.get('lesson_number', 999))
        
        # Способ 1: Через lessons_by_class
        if student_class in self.lessons_by_class:
            # Пробуем разные варианты названия предмета
            subject_variants = [subject]
            
            # Если предмет содержит "язык", пробуем без него
            if 'язык' in subject:
                subject_variants.append(subject.replace('язык', '').strip())
            
            # Пробуем английское название
            if subject in self.subject_mapping_reverse:
                subject_variants.append(self.subject_mapping_reverse[subject])
            
            debug_log(f"🔥 Пробуем варианты: {subject_variants}")
            
            for subject_variant in subject_variants:
                if subject_variant in self.lessons_by_class[student_class]:
                    lessons = self.lessons_by_class[student_class][subject_variant]
                    debug_log(f"🔥 Найдено через lessons_by_class: {len(lessons)} уроков")
                    return sorted(lessons, key=lambda x: x.get('lesson_number', 999))
        
        # Способ 2: Прямой поиск в файловой системе
        debug_log(f"🔥 Прямой поиск в файловой системе...")
        direct_lessons = self._find_lessons_directly(student_class, subject)
        
        if direct_lessons:
            debug_log(f"🔥 Найдено прямым поиском: {len(direct_lessons)} уроков")
            
            # Добавляем найденные уроки в кэш
            if student_class not in self.lessons_by_class:
                self.lessons_by_class[student_class] = {}
            
            if subject not in self.lessons_by_class[student_class]:
                self.lessons_by_class[student_class][subject] = direct_lessons
            
            return sorted(direct_lessons, key=lambda x: x.get('lesson_number', 999))
        
        debug_log(f"🔥 Уроки не найдены ни одним способом")
        return []

    def _find_lessons_directly(self, class_level: str, subject: str) -> List[dict]:
        """Прямой поиск уроков в файловой системе"""
        lessons = []
        
        # 🔥 ОСОБАЯ ЛОГИКА ДЛЯ ВЗРОСЛЫХ
        if class_level == 'adult' and subject == 'английский язык':
            # Ищем уроки в adult_language/{level}_english/
            adult_lang_dir = self.students_base_dir / "adult_language"
            if adult_lang_dir.exists():
                # Если есть CEFR уровень - ищем только его папку
                if self.cefr_level:
                    level_dir = adult_lang_dir / f"{self.cefr_level}_english"
                    if level_dir.exists():
                        debug_log(f"🎓 Поиск уроков для взрослых уровня {self.cefr_level}: {level_dir}")
                        lesson_files = list(level_dir.glob("*.txt"))
                    else:
                        # Если папки уровня нет, ищем во всех уровнях
                        debug_log(f"🎓 Папки уровня {self.cefr_level} нет, ищем во всех уровнях")
                        lesson_files = []
                        for subdir in adult_lang_dir.iterdir():
                            if subdir.is_dir() and "_english" in subdir.name:
                                lesson_files.extend(subdir.glob("*.txt"))
                else:
                    # Нет CEFR уровня - ищем во всех уровнях
                    debug_log("🎓 Поиск уроков для взрослых во всех уровнях")
                    lesson_files = []
                    for subdir in adult_lang_dir.iterdir():
                        if subdir.is_dir() and "_english" in subdir.name:
                            lesson_files.extend(subdir.glob("*.txt"))
                
                # Обрабатываем найденные файлы
                for lesson_file in lesson_files:
                    try:
                        lesson_id = lesson_file.stem
                        lesson_number = self._extract_lesson_number(lesson_id)
                        lesson_title = self._format_lesson_title(lesson_id)
                        
                        # Извлекаем уровень CEFR из пути
                        cefr_level = "unknown"
                        for parent in lesson_file.parents:
                            if "_english" in parent.name:
                                cefr_level = parent.name.replace("_english", "")
                                break
                        
                        lesson_data = {
                            'id': f"adult_{cefr_level}_{lesson_id}",
                            'title': lesson_title,
                            'file_path': lesson_file,
                            'type': 'adult_language',
                            'subject': subject,
                            'class_level': class_level,
                            'lesson_number': lesson_number,
                            'full_path': str(lesson_file.relative_to(self.lessons_dir)),
                            'cefr_level': cefr_level,
                            'target_language': 'english'
                        }
                        
                        lessons.append(lesson_data)
                        debug_log(f"🎓 Добавлен урок для взрослых: {lesson_title} (уровень {cefr_level})")
                        
                    except Exception as e:
                        debug_log(f"🎓 Ошибка обработки файла {lesson_file}: {e}")
                
                return lessons
        
        # 🔥 ПРОВЕРЯЕМ СУЩЕСТВОВАНИЕ ПАПКИ КЛАССА
        class_dir = self.students_base_dir / f"{class_level}_class"
        if not class_dir.exists():
            debug_log(f"🔥 Папка класса не существует: {class_dir}")
            return lessons
        
        debug_log(f"🔥 Папка класса найдена: {class_dir}")
        
        # 🔥 ИЩЕМ ПАПКУ ПРЕДМЕТА
        # Вариант 1: Точное совпадение
        subject_dir = class_dir / subject
        if subject_dir.exists() and subject_dir.is_dir():
            debug_log(f"🔥 Найдена папка предмета (точное совпадение): {subject_dir}")
            lesson_files = list(subject_dir.glob("*.txt"))
            debug_log(f"🔥 Найдено файлов уроков: {len(lesson_files)}")
        else:
            # Вариант 2: Поиск по частичному совпадению
            debug_log(f"🔥 Ищем папку с частичным совпадением...")
            lesson_files = []
            
            for potential_subject_dir in class_dir.iterdir():
                if potential_subject_dir.is_dir():
                    dir_name_lower = potential_subject_dir.name.lower()
                    subject_lower = subject.lower()
                    
                    # Проверяем разные варианты совпадения
                    if (subject_lower in dir_name_lower or 
                        dir_name_lower in subject_lower or
                        any(word in dir_name_lower for word in subject_lower.split())):
                        
                        debug_log(f"🔥 Найдена подходящая папка: {potential_subject_dir}")
                        lesson_files.extend(potential_subject_dir.glob("*.txt"))
                        break
            
            debug_log(f"🔥 Найдено файлов уроков: {len(lesson_files)}")
        
        # 🔥 ПРЕОБРАЗУЕМ ФАЙЛЫ В СТРУКТУРУ УРОКОВ
        for lesson_file in lesson_files:
            try:
                lesson_id = lesson_file.stem
                lesson_number = self._extract_lesson_number(lesson_id)
                lesson_title = self._format_lesson_title(lesson_id)
                
                lesson_data = {
                    'id': f"{class_level}_{subject}_{lesson_id}",
                    'title': lesson_title,
                    'file_path': lesson_file,
                    'type': 'student',
                    'subject': subject,
                    'class_level': class_level,
                    'lesson_number': lesson_number,
                    'full_path': f"{class_level}_class/{subject}/{lesson_file.name}"
                }
                
                lessons.append(lesson_data)
                debug_log(f"🔥 Добавлен урок: {lesson_title}")
                
            except Exception as e:
                debug_log(f"🔥 Ошибка обработки файла {lesson_file}: {e}")
        
        return lessons

    def get_next_lesson_for_student(self, subject: str) -> Optional[Dict]:
        """ИСПРАВЛЕННЫЙ МЕТОД: Возвращает следующий незавершенный урок по предмету"""
        if not self.has_student_data:
            return None
        
        student_class = self.student_data.get('education_level', '5')
        
        # 🔥 ОСОБАЯ ЛОГИКА ДЛЯ ВЗРОСЛЫХ В РЕЖИМЕ "ИЗУЧАТЬ ЧТО УГОДНО"
        if self.is_adult_student and self.adult_study_mode == 'anything':
            return None  # Нет уроков в этом режиме
        
        # ИСПРАВЛЕНИЕ: Сначала получаем все уроки для этого класса и предмета
        lessons = self.get_lessons_for_student_subject(subject)
        
        if not lessons:
            # Если нет уроков для этого класса, возвращаем первый общий урок
            general_lessons = self.lessons.get(subject, [])
            if general_lessons:
                return general_lessons[0]
            return None
        
        # Получаем прогресс ученика по предмету
        progress = self.get_student_progress(subject)
        completed_ids = progress.get('completed_lessons', [])
        
        # Ищем следующий незавершенный урок
        for lesson in lessons:
            if lesson['id'] not in completed_ids:
                debug_log(f"🔥 Найден следующий урок: {lesson['title']}")
                return lesson
        
        debug_log(f"🔥 Все уроки по предмету '{subject}' завершены")
        return None  # Все уроки завершены

    def get_student_progress(self, subject: str = None) -> Dict:
        """НОВЫЙ МЕТОД: Возвращает прогресс ученика"""
        if not self.has_student_data:
            return {}
        
        student_id = self.student_data.get('student_id')
        if not student_id:
            return {}
        
        # 🔥 ОСОБАЯ ЛОГИКА ДЛЯ ВЗРОСЛЫХ В РЕЖИМЕ "ИЗУЧАТЬ ЧТО УГОДНО"
        if self.is_adult_student and self.adult_study_mode == 'anything':
            return {
                "completed_lessons": [],
                "current_lesson": None,
                "total_lessons": 0,
                "last_updated": 0,
                "study_mode": "anything",
                "has_structured_lessons": False
            }
        
        # НОВОЕ: Загружаем прогресс из файла
        progress_file = student_progress_dir / f"{student_id}.json"
        try:
            if progress_file.exists():
                with open(progress_file, 'r', encoding='utf-8') as f:
                    all_progress = json.load(f)
                    
                    if subject:
                        subject_progress = all_progress.get(subject, {
                            "completed_lessons": [],
                            "current_lesson": None,
                            "total_lessons": len(self.get_lessons_for_student_subject(subject)) if subject else 0,
                            "last_updated": 0
                        })
                        
                        # 🔥 ДОБАВЛЯЕМ CEFR ДЛЯ ВЗРОСЛЫХ
                        if self.is_adult_student and self.cefr_level:
                            subject_progress['cefr_level'] = self.cefr_level
                        
                        return subject_progress
                    else:
                        return all_progress
        except Exception as e:
            debug_log(f"Ошибка загрузки прогресс: {e}")
        
        # Возвращаем пустой прогресс если файла нет
        if subject:
            progress_data = {
                "completed_lessons": [],
                "current_lesson": None,
                "total_lessons": len(self.get_lessons_for_student_subject(subject)) if subject else 0,
                "last_updated": 0
            }
            
            # 🔥 ДОБАВЛЯЕМ CEFR ДЛЯ ВЗРОСЛЫХ
            if self.is_adult_student and self.cefr_level:
                progress_data['cefr_level'] = self.cefr_level
            
            return progress_data
        else:
            return {}

    def save_student_progress(self, lesson_id: str, subject: str, completed: bool = True):
        """НОВЫЙ МЕТОД: Сохраняет прогресс ученика"""
        if not self.has_student_data:
            return
        
        # 🔥 НЕ СОХРАНЯЕМ ПРОГРЕСС ДЛЯ ВЗРОСЛЫХ В РЕЖИМЕ "ИЗУЧАТЬ ЧТО УГОДНО"
        if self.is_adult_student and self.adult_study_mode == 'anything':
            debug_log("🎓 Взрослый в режиме 'изучать что угодно' - не сохраняем прогресс")
            return
        
        student_id = self.student_data.get('student_id')
        if not student_id:
            return
        
        progress_file = student_progress_dir / f"{student_id}.json"
        
        # Создаем папку если не существует
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        
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
        
        # 🔥 ДОБАВЛЯЕМ CEFR ДЛЯ ВЗРОСЛЫХ
        if self.is_adult_student and self.cefr_level:
            subject_progress['cefr_level'] = self.cefr_level
        
        # Сохраняем
        try:
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
            debug_log(f"✅ Прогресс сохранен: {lesson_id} по предмету {subject}")
        except Exception as e:
            debug_log(f"❌ Ошибка сохранения прогресс: {e}")

    def mark_lesson_completed(self, lesson_data: Dict):
        """НОВЫЙ МЕТОД: Помечает урок как завершенный"""
        if lesson_data and self.has_student_data:
            debug_log(f"🎓 Отмечаем урок как завершенный: {lesson_data['title']}")
            self.save_student_progress(
                lesson_data['id'], 
                lesson_data['subject'], 
                completed=True
            )

    def get_available_subjects_for_student(self) -> Dict[str, List[Dict]]:
        """НОВЫЙ МЕТОД: Возвращает предметы и уроки для текущего ученика по его классу"""
        if not self.has_student_data:
            return {}
        
        student_class = self.student_data.get('education_level', '5')
        
        # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: ЛЕНИВАЯ ЗАГРУЗКА
        if self._lessons is None:
            self._ensure_lessons_loaded()
            
        if student_class not in self.lessons_by_class:
            return {}
        
        result = {}
        for subject, lessons in self.lessons_by_class[student_class].items():
            # Сортируем уроки по номеру
            sorted_lessons = sorted(lessons, key=lambda x: x.get('lesson_number', 999))
            
            result[subject] = sorted_lessons
        
        return result

    def set_llm_model(self, model: str):
        _ = self.llm  # Инициализируем если еще не инициализирован
        self.llm.set_model(model)
        debug_log(f"Установлена модель LLM: {model}")

    def set_llm_mode(self, mode: str):
        if mode in ["traditional", "llm_first"]:
            self.llm_query_mode = mode
            debug_log(f"Установлен режим LLM: {mode}")

    def get_knowledge_stats(self) -> Optional[Dict]:
        if self.knowledge_base:
            return self.knowledge_base.get_stats()
        return None

    def set_room_id(self, room_id: str):
        """Установка ID комнаты для WebSocket коммуникации"""
        self.room_id = room_id
        debug_log(f"🔧 Установлен room_id для DialogueManager: {room_id}")

    def set_student_data(self, student_data: dict):
        """🔥 ИСПРАВЛЕННЫЙ МЕТОД: Устанавливаем данные ученика БЕЗ загрузки уроков"""
        self.student_data = student_data
        self.has_student_data = bool(student_data)
        
        # 🔥 ОПРЕДЕЛЯЕМ, ЭТО ВЗРОСЛЫЙ УЧЕНИК
        education_level = student_data.get('education_level', '')
        self.is_adult_student = (education_level == 'adult')
        
        if self.is_adult_student:
            self.adult_study_mode = student_data.get('study_mode', 'language')
            self.cefr_level = student_data.get('language_level', 'B1')
            
            if LANGUAGE_SUPPORT_ENABLED and self.cefr_level:
                self.cefr_config = get_cefr_level_config(self.cefr_level)
            
            debug_log(f"🎓 Установлены данные взрослого ученика: {student_data.get('name', 'неизвестно')}, режим: {self.adult_study_mode}, уровень CEFR: {self.cefr_level}")
        elif self.has_student_data:
            student_name = student_data.get('name', 'неизвестно')
            student_class = student_data.get('education_level', 'неизвестно')
            debug_log(f"🎓 Установлены данные ученика: {student_name} ({student_class} класс)")
            
        # 🔥 ВАЖНОЕ ИСПРАВЛЕНИЕ: НЕ загружаем уроки здесь!
        # Прогресс загружается лениво при вызове get_student_progress
        debug_log(f"📊 Прогресс ученика будет загружен при необходимости (лениво)")

    def get_practice_status(self) -> Dict:
        """Возвращает статус практики"""
        practice_status = {
            "practice_active": self.practice_active,
            "waiting_for_answer": self.waiting_for_answer,
            "current_question": self.current_practice_question,
            "question_index": self.current_question_index,
            "max_questions": self.max_questions,
            "questions_asked": len(self.practice_manager.generated_questions) if hasattr(self.practice_manager, 'generated_questions') else 0,
            "is_technical": self.is_technical_subject,
            "subject_type": self.subject_type
        }
        
        # 🔥 ДОБАВЛЯЕМ CEFR ДЛЯ ВЗРОСЛЫХ
        if self.cefr_level:
            practice_status['cefr_level'] = self.cefr_level
            practice_status['is_adult'] = self.is_adult_student
        
        return practice_status

    def force_start_practice(self, lesson_context: str, subject: str) -> str:
        """Принудительно запускает практику (для тестирования)"""
        try:
            self.lesson_started = False
            self.current_state = "practice_session"
            self.practice_active = True
            self.waiting_for_answer = False
            self.current_question_index = 0
            
            debug_log("=== ПРИНУДИТЕЛЬНЫЙ ЗАПУСК ПРАКТИКИ ===")
            
            # Определяем тип предмета
            self._determine_subject_type()
            
            # Инициализируем менеджер практики
            self.practice_manager.initialize_practice_generation(lesson_context, subject)
            
            # 🔥 ПЕРЕДАЕМ CEFR ДЛЯ ВЗРОСЛЫХ
            if hasattr(self.practice_manager, 'cefr_level'):
                self.practice_manager.cefr_level = self.cefr_level
            
            # Получаем первый вопрос
            debug_log("🔄 Получение первого вопроса практики...")
            first_question = self.practice_manager.get_next_question()
            
            if first_question:
                self.waiting_for_answer = True
                self.current_practice_question = {
                    "id": 1,
                    "question": first_question,
                    "answer": ""
                }
                # 🔥 ОСОБЫЙ ТЕКСТ ДЛЯ ВЗРОСЛЫХ С CEFR
                if self.is_adult_student and self.cefr_level:
                    return f"Практика английского языка уровня {self.cefr_level} запущена. Первый вопрос: {first_question}"
                else:
                    return f"Практика запущена. Первый вопрос: {first_question}"
            else:
                self.practice_active = False
                return "Не удалось запустить практику"
                
        except Exception as e:
            debug_log(f"❌ Ошибка принудительного запуска практики: {e}")
            return f"Ошибка запуска практики: {e}"

    def skip_to_practice(self):
        """Пропускает урок и сразу переходит к практике (для тестирования)"""
        if not self.lesson_started or not self.lesson_content:
            return "Сначала нужно начать урок"
        
        debug_log("=== ПРОПУСК К ПРАКТИКЕ ===")
        practice_message = self._start_practice_session()
        return practice_message

    def get_visualization_status(self) -> Dict:
        """Возвращает статус визуализации - ТОЛЬКО SVG"""
        viz_status = {
            "visualization_enabled": self.visualization_enabled,
            "visualization_counter": self.visualization_counter,
            "last_visualization_time": self.last_visualization_time,
            "paragraphs_since_last_viz": self.paragraphs_since_last_viz,
            "type": "svg_infographic",
            "is_technical": self.is_technical_subject,
            "subject_type": self.subject_type
        }
        
        # 🔥 ДОБАВЛЯЕМ CEFR ДЛЯ ВЗРОСЛЫХ
        if self.cefr_level:
            viz_status['cefr_level'] = self.cefr_level
            viz_status['is_adult'] = self.is_adult_student
        
        return viz_status

    def force_visualization(self, text: str) -> bool:
        """Принудительно генерирует SVG инфографику для текста"""
        try:
            if not self.room_id:
                debug_log("❌ Нет room_id для отправки инфографики")
                return False
            
            context = " ".join(self.lesson_content[max(0, self.current_paragraph-2):self.current_paragraph]) if self.lesson_content else ""
            self._generate_visualization(text, context)
            return True
            
        except Exception as e:
            debug_log(f"❌ Ошибка принудительной генерации инфографики: {e}")
            return False

    def get_conversation_stats(self) -> Dict:
        """Возвращает статистики диалога"""
        user_messages = [msg for msg in self.conversation_history if msg['is_user']]
        teacher_messages = [msg for msg in self.conversation_history if not msg['is_user']]
        
        stats = {
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
            "student_class": self.student_data.get('education_level', 'нет'),
            "subject_type": self.subject_type,
            "is_technical": self.is_technical_subject,
            "is_language": self.is_language_subject,
            "target_language": self.target_language if self.is_language_subject else None
        }
        
        # 🔥 ДОБАВЛЯЕМ ДАННЫЕ ВЗРОСЛЫХ
        if self.is_adult_student:
            stats['is_adult'] = True
            stats['adult_study_mode'] = self.adult_study_mode
            stats['cefr_level'] = self.cefr_level
        
        return stats

    def debug_info(self) -> Dict:
        """Возвращает отладочную информацию"""
        practice_stats = self.practice_manager.get_practice_stats() if hasattr(self.practice_manager, 'get_practice_stats') else {}
        
        # НОВОЕ: Информация о прогрессе
        student_progress = {}
        if self.has_student_data and self.current_subject:
            student_progress = self.get_student_progress(self.current_subject)
        
        debug_info = {
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
            # НОВОЕ: Информация о структуре уроков
            "available_classes": list(self.lessons_by_class.keys()) if self._lessons_by_class else [],
            "student_class_lessons": self.get_available_subjects_for_student(),
            "student_progress": student_progress,
            "next_lesson": self.get_next_lesson_for_student(self.current_subject) if self.current_subject else None,
            # НОВОЕ: Информация о языковой поддержке
            "is_language_subject": self.is_language_subject,
            "target_language": self.target_language,
            "language_level": self.language_level,
            "bilingual_ratio": self.bilingual_ratio,
            "language_support_enabled": LANGUAGE_SUPPORT_ENABLED,
            # 🔥 НОВОЕ: Информация о технической поддержке
            "is_technical_subject": self.is_technical_subject,
            "subject_type": self.subject_type,
            "technical_support_enabled": TECHNICAL_SUPPORT_ENABLED,
            "technical_prompts_enabled": TECHNICAL_PROMPTS_ENABLED,
            "technical_symbols_preserved": self.technical_symbols_preserved,
            # 🔥 НОВОЕ: Информация о слайдах
            "slides_enabled": self.slides_enabled,
            "lesson_slides_count": len(self.lesson_slides),
            "current_slide_index": self.current_paragraph - 2 if self.current_paragraph >= 2 else None
        }
        
        # 🔥 ДОБАВЛЯЕМ ИНФОРМАЦИЮ О ВЗРОСЛЫХ И CEFR
        if self.is_adult_student:
            debug_info['is_adult_student'] = True
            debug_info['adult_study_mode'] = self.adult_study_mode
            debug_info['cefr_level'] = self.cefr_level
            debug_info['cefr_config'] = self.cefr_config
            
            if LANGUAGE_SUPPORT_ENABLED:
                debug_info['available_cefr_levels'] = get_available_cefr_levels() if hasattr(get_available_cefr_levels, '__call__') else []
                debug_info['adult_study_modes'] = get_adult_study_modes() if hasattr(get_adult_study_modes, '__call__') else []
        
        return debug_info

    def export_conversation_history(self) -> List[Dict]:
        """Экспортирует историю диалога"""
        return self.conversation_history.copy()

    def clear_conversation_history(self):
        """Очищает историю диалога"""
        self.conversation_history = []
        self.conversation_context = []
        debug_log("🗑️ История диалога очищена")

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
        
        # НОВОЕ: Дополнительные команды для учеников
        if self.has_student_data:
            student_commands = {
                "мой прогресс": "Показать прогресс по предметам",
                "следующий урок": "Начать следующий урок по текущему предмету",
                "выбрать урок": "Выбрать конкретный урок",
                "мои уроки": "Показать все доступные уроки"
            }
            base_commands.update(student_commands)
        
        # 🔥 ДОПОЛНИТЕЛЬНЫЕ КОМАНДЫ ДЛЯ ВЗРОСЛЫХ
        if self.is_adult_student:
            adult_commands = {
                "сменить тему": "Сменить тему обсуждения (режим 'изучать что угодно')",
                "уровень языка": "Показать текущий уровень CEFR",
                "следующий урок английского": "Начать следующий урок английского"
            }
            base_commands.update(adult_commands)
        
        # 🔥 ДОПОЛНИТЕЛЬНЫЕ КОМАНДЫ ДЛЯ ТЕХНИЧЕСКИХ ПРЕДМЕТОВ
        if self.is_technical_subject:
            technical_commands = {
                "формула": "Показать формулы из урока",
                "график": "Показать графики/диаграммы",
                "пример": "Показать примеры решения"
            }
            base_commands.update(technical_commands)
        
        return base_commands

    def force_lesson_start_notification(self):
        """Принудительно отправляет уведомление о начале урока (для восстановления состояния)"""
        if self.lesson_started and self.selected_lesson and self.room_id and self.socketio:
            debug_log(f"🔧 ПРИНУДИТЕЛЬНАЯ отправка lesson_started для комнаты {self.room_id}")
            
            lesson_data = {
                'lesson_id': self.selected_lesson['id'],
                'title': self.selected_lesson['title'],
                'subject': self.current_subject,
                'class_level': self.selected_lesson.get('class_level', 'general'),
                'lesson_number': self.selected_lesson.get('lesson_number'),
                'is_student_lesson': True,
                'is_technical': self.is_technical_subject,
                'subject_type': self.subject_type,
                'slides_count': len(self.lesson_slides)  # 🔥 НОВОЕ: отправляем количество слайдов
            }
            
            # 🔥 ДОБАВЛЯЕМ CEFR ДЛЯ ВЗРОСЛЫХ
            if self.cefr_level:
                lesson_data['cefr_level'] = self.cefr_level
                lesson_data['is_adult'] = self.is_adult_student
            
            self.socketio.emit('lesson_started', lesson_data, room=self.room_id)
            
            # Также отправляем текущий абзац если есть
            if self.lesson_content and self.current_paragraph > 0:
                self.socketio.emit('speech_text', {
                    'text': f"Учитель: {self.lesson_content[self.current_paragraph - 1]}",
                    'sid': 'teacher',
                    'is_teacher': True
                }, room=self.room_id)
            
            # 🔥 НОВОЕ: Отправляем текущий слайд если есть
            if self.lesson_slides and self.current_paragraph >= 2:
                slide_index = self.current_paragraph - 2
                if 0 <= slide_index < len(self.lesson_slides):
                    self.socketio.emit('lesson_slide', {
                        'room_id': self.room_id,
                        'slide_url': self.lesson_slides[slide_index],
                        'slide_index': slide_index,
                        'slide_number': slide_index + 1,
                        'total_slides': len(self.lesson_slides),
                        'paragraph_index': self.current_paragraph - 1,
                        'has_slide': True
                    }, room=self.room_id)
            
            return True
        return False

    def get_lessons_for_student_api(self) -> Dict:
        """НОВЫЙ МЕТОД: Возвращает уроки для API запроса (для личного кабинета)"""
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
        
        # 🔥 ОСОБАЯ ЛОГИКА ДЛЯ ВЗРОСЛЫХ
        if self.is_adult_student:
            result['is_adult'] = True
            result['adult_study_mode'] = self.adult_study_mode
            result['cefr_level'] = self.cefr_level
            
            if self.adult_study_mode == 'anything':
                result['message'] = "Режим 'изучать что угодно' - нет структурированных уроков"
                return result
            elif self.adult_study_mode == 'language':
                # Только английский язык для взрослых
                subject = "английский язык"
                if student_class in self.lessons_by_class and subject in self.lessons_by_class[student_class]:
                    lessons = self.lessons_by_class[student_class][subject]
                    
                    # Фильтруем по уровню CEFR если установлен
                    if self.cefr_level:
                        filtered_lessons = [lesson for lesson in lessons if lesson.get('cefr_level') == self.cefr_level]
                        if filtered_lessons:
                            lessons = filtered_lessons
                    
                    # Получаем прогресс
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
                            'type': lesson.get('type', 'adult_language'),
                            'subject_type': 'language',
                            'cefr_level': lesson.get('cefr_level', self.cefr_level)
                        })
                    
                    result["subjects"].append({
                        'subject': subject,
                        'subject_type': 'language',
                        'lessons': subject_lessons,
                        'total_lessons': len(subject_lessons),
                        'completed_lessons': len([l for l in subject_lessons if l['completed']]),
                        'progress_percent': int((len([l for l in subject_lessons if l['completed']]) / len(subject_lessons)) * 100) if subject_lessons else 0,
                        'cefr_level': self.cefr_level
                    })
                
                return result
        
        if student_class in self.lessons_by_class:
            for subject, lessons in self.lessons_by_class[student_class].items():
                # Получаем прогресс по предмету
                progress = self.get_student_progress(subject)
                completed_ids = progress.get('completed_lessons', [])
                
                # Определяем тип предмета
                subject_type = "general"
                if TECHNICAL_SUPPORT_ENABLED:
                    subject_type = get_subject_type(subject)
                
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
                        'type': lesson.get('type', 'student'),
                        'subject_type': subject_type
                    })
                
                result["subjects"].append({
                    'subject': subject,
                    'subject_type': subject_type,
                    'lessons': subject_lessons,
                    'total_lessons': len(subject_lessons),
                    'completed_lessons': len([l for l in subject_lessons if l['completed']]),
                    'progress_percent': int((len([l for l in subject_lessons if l['completed']]) / len(subject_lessons)) * 100) if subject_lessons else 0
                })
        
        return result

    # 🔥 НОВЫЕ МЕТОДЫ ДЛЯ УПРАВЛЕНИЯ СЛАЙДАМИ
    def enable_slides(self):
        """Включение отображения слайдов"""
        self.slides_enabled = True
        debug_log("✅ Отображение слайдов включено")

    def disable_slides(self):
        """Выключение отображения слайдов"""
        self.slides_enabled = False
        debug_log("❌ Отображение слайдов выключено")

    def get_slides_status(self) -> Dict:
        """Возвращает статус слайдов для текущего урока"""
        slides_status = {
            "slides_enabled": self.slides_enabled,
            "slides_count": len(self.lesson_slides),
            "current_slide_index": self.current_paragraph - 2 if self.current_paragraph >= 2 else None,
            "lesson_has_slides": len(self.lesson_slides) > 0,
            "current_lesson": self.selected_lesson['title'] if self.selected_lesson else None
        }
        
        # 🔥 ДОБАВЛЯЕМ CEFR ДЛЯ ВЗРОСЛЫХ
        if self.cefr_level:
            slides_status['cefr_level'] = self.cefr_level
            slides_status['is_adult'] = self.is_adult_student
        
        return slides_status

    def force_show_slide(self, slide_index: int) -> bool:
        """Принудительно показывает слайд по индексу"""
        if not self.room_id or not self.socketio or slide_index < 0:
            return False
        
        if slide_index < len(self.lesson_slides):
            slide_url = self.lesson_slides[slide_index]
            self.socketio.emit('lesson_slide', {
                'room_id': self.room_id,
                'slide_url': slide_url,
                'slide_index': slide_index,
                'slide_number': slide_index + 1,
                'total_slides': len(self.lesson_slides),
                'paragraph_index': self.current_paragraph,
                'has_slide': True,
                'force_show': True
            }, room=self.room_id)
            debug_log(f"🔧 Принудительно показан слайд {slide_index + 1}: {slide_url}")
            return True
        
        return False

    # 🔥 НОВЫЕ МЕТОДЫ ДЛЯ РАБОТЫ С ВЗРОСЛЫМИ И CEFR
    def set_adult_study_mode(self, mode: str):
        """Установка режима обучения для взрослых"""
        if mode in ['language', 'anything']:
            self.adult_study_mode = mode
            debug_log(f"🎓 Установлен режим обучения для взрослых: {mode}")
        else:
            debug_log(f"⚠️ Неизвестный режим обучения для взрослых: {mode}")

    def set_cefr_level(self, level: str):
        """Установка уровня CEFR"""
        if level in ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']:
            self.cefr_level = level
            if LANGUAGE_SUPPORT_ENABLED:
                self.cefr_config = get_cefr_level_config(level)
            debug_log(f"🎓 Установлен уровень CEFR: {level}")
        else:
            debug_log(f"⚠️ Неизвестный уровень CEFR: {level}")

    def get_adult_info(self) -> Dict:
        """Возвращает информацию о настройках взрослого ученика"""
        if not self.is_adult_student:
            return {"is_adult": False}
        
        info = {
            "is_adult": True,
            "study_mode": self.adult_study_mode,
            "cefr_level": self.cefr_level,
            "has_structured_lessons": self.adult_study_mode == 'language'
        }
        
        if self.cefr_config:
            info['cefr_description'] = self.cefr_config.get('description', '')
            info['bilingual_ratio'] = self.cefr_config.get('bilingual_ratio', 0.5)
        
        return info

# Тестирование
if __name__ == "__main__":
    # Тестирование базовой функциональности
    dm = DialogueManager(None)
    
    debug_log("🧪 Тестирование DialogueManager с поддержкой взрослых и CEFR...")
    
    # Проверяем что создание быстрое
    debug_log("✅ DialogueManager создан мгновенно")
    
    # Тест метода поиска слайдов
    test_lesson_path = Path("lessons/demo/demo_physics.txt")
    if test_lesson_path.exists():
        slides = dm._find_lesson_slides(test_lesson_path)
        debug_log(f"📸 Тест поиска слайдов: найдено {len(slides)} слайдов")
    else:
        debug_log("⚠️ Тестовый файл урока не найден для проверки слайдов")
    
    # Тест доступных предметов (загрузит уроки лениво)
    subjects = dm.get_available_subjects()
    debug_log(f"📚 Доступные предметы: {len(subjects)} предметов")
    
    # Тест обработки приветствия
    response = dm.process_input("привет")
    debug_log(f"👋 Ответ на приветствие: {response[:100]}...")
    
    # Тест данных взрослого ученика
    adult_data = {
        'name': 'Иван',
        'education_level': 'adult',
        'age': '30',
        'student_id': 'test_adult_123',
        'study_mode': 'language',
        'language_level': 'B1'
    }
    
    dm.set_student_data(adult_data)
    debug_log(f"🎓 Установлены данные взрослого ученика: {dm.is_adult_student}, режим: {dm.adult_study_mode}, CEFR: {dm.cefr_level}")
    
    # Тест доступных предметов для взрослого
    adult_subjects = dm.get_available_subjects()
    debug_log(f"🎓 Доступные предметы для взрослого: {adult_subjects}")
    
    debug_log("✅ Тестирование завершено! Поддержка взрослых и CEFR добавлена.")
