"""
🔥 language_integration.py v2.0
ИНТЕГРАЦИЯ ЯЗЫКОВЫХ ФУНКЦИЙ И УРОВНЕЙ CEFR ДЛЯ ВЗРОСЛЫХ
Минимальные изменения, максимальная совместимость
Без нарушения рабочей логики системы
"""

import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

def debug_log(message: str):
    """Логирование для отладки"""
    print(f"🎯 [LANGUAGE] {message}")

class LanguageIntegration:
    """Класс для интеграции языковых функций в существующую систему"""
    
    # 🔥 КОНФИГУРАЦИЯ ЯЗЫКОВЫХ ПРЕДМЕТОВ (СУЩЕСТВУЮЩАЯ)
    LANGUAGE_SUBJECTS = {
        'english': {
            'russian_names': ['английский язык', 'английский'],
            'code': 'en',
            'levels': {
                'beginner': {'bilingual_ratio': 0.3, 'max_questions': 5},
                'intermediate': {'bilingual_ratio': 0.5, 'max_questions': 7},
                'advanced': {'bilingual_ratio': 0.7, 'max_questions': 10}
            }
        },
        'french': {
            'russian_names': ['французский язык', 'французский'],
            'code': 'fr',
            'levels': {
                'beginner': {'bilingual_ratio': 0.3, 'max_questions': 5},
                'intermediate': {'bilingual_ratio': 0.5, 'max_questions': 7},
                'advanced': {'bilingual_ratio': 0.7, 'max_questions': 10}
            }
        },
        'german': {
            'russian_names': ['немецкий язык', 'немецкий'],
            'code': 'de',
            'levels': {
                'beginner': {'bilingual_ratio': 0.3, 'max_questions': 5},
                'intermediate': {'bilingual_ratio': 0.5, 'max_questions': 7},
                'advanced': {'bilingual_ratio': 0.7, 'max_questions': 10}
            }
        },
        'spanish': {
            'russian_names': ['испанский язык', 'испанский'],
            'code': 'es',
            'levels': {
                'beginner': {'bilingual_ratio': 0.3, 'max_questions': 5},
                'intermediate': {'bilingual_ratio': 0.5, 'max_questions': 7},
                'advanced': {'bilingual_ratio': 0.7, 'max_questions': 10}
            }
        }
    }
    
    # 🔥 НОВАЯ КОНСТАНТА: УРОВНИ CEFR ДЛЯ ВЗРОСЛЫХ
    CEFR_LEVELS = {
        'A1': {
            'description': 'Начинающий',
            'prompt_adjustment': '''
            УРОВЕНЬ A1 (НАЧИНАЮЩИЙ):
            - Используй максимально простые предложения (3-5 слов)
            - 80% русский язык, 20% английский
            - Только базовая лексика (hello, my name is, thank you)
            - Повторяй ключевые фразы по 2-3 раза
            - Не используй сложные грамматические конструкции
            - Только Present Simple время
            - Каждое новое слово переводи и объясняй
            ''',
            'bilingual_ratio': 0.2,
            'max_questions': 5,
            'vocabulary_per_lesson': 8,
            'sentence_length': '3-5 слов'
        },
        'A2': {
            'description': 'Элементарный',
            'prompt_adjustment': '''
            УРОВЕНЬ A2 (ЭЛЕМЕНТАРНЫЙ):
            - Простые предложения (5-7 слов)
            - 70% русский, 30% английский
            - Базовая повседневная лексика (семья, работа, хобби)
            - Простые вопросы и ответы
            - Present Simple, Present Continuous, Past Simple
            - Давай упражнения на заполнение пропусков
            ''',
            'bilingual_ratio': 0.3,
            'max_questions': 6,
            'vocabulary_per_lesson': 12,
            'sentence_length': '5-7 слов'
        },
        'B1': {
            'description': 'Средний',
            'prompt_adjustment': '''
            УРОВЕНЬ B1 (СРЕДНИЙ):
            - Развернутые предложения (7-10 слов)
            - 50% русский, 50% английский
            - Широкая бытовая и учебная лексика
            - Объяснение грамматики на русском, практика на английском
            - Все основные времена (Present, Past, Future)
            - Диалоги и ситуации из реальной жизни
            - Упражнения на перефразирование
            ''',
            'bilingual_ratio': 0.5,
            'max_questions': 7,
            'vocabulary_per_lesson': 15,
            'sentence_length': '7-10 слов'
        },
        'B2': {
            'description': 'Выше среднего',
            'prompt_adjustment': '''
            УРОВЕНЬ B2 (ВЫШЕ СРЕДНЕГО):
            - Сложные предложения с придаточными
            - 30% русский, 70% английский
            - Абстрактные темы, аргументация, мнения
            - Нюансы грамматики (условные предложения, пассивный залог)
            - Идиомы и устойчивые выражения
            - Обсуждение новостей, фильмов, книг
            - Упражнения на эссе и дискуссии
            ''',
            'bilingual_ratio': 0.7,
            'max_questions': 8,
            'vocabulary_per_lesson': 18,
            'sentence_length': '10-15 слов'
        },
        'C1': {
            'description': 'Продвинутый',
            'prompt_adjustment': '''
            УРОВЕНЬ C1 (ПРОДВИНУТЫЙ):
            - Сложные тексты и дискуссии
            - 10% русский (только пояснения), 90% английский
            - Академическая и профессиональная лексика
            - Стилистические нюансы, синонимы, антонимы
            - Дебаты и презентации
            - Анализ сложных текстов
            - Упражнения на академическое письмо
            ''',
            'bilingual_ratio': 0.9,
            'max_questions': 9,
            'vocabulary_per_lesson': 20,
            'sentence_length': '15+ слов'
        },
        'C2': {
            'description': 'В совершенстве',
            'prompt_adjustment': '''
            УРОВЕНЬ C2 (В СОВЕРШЕНСТВЕ):
            - 100% английский язык
            - Сложнейшие тексты любой тематики
            - Нюансы, ирония, сарказм, юмор
            - Академическое письмо и исследования
            - Профессиональные дискуссии и переговоры
            - Литературный анализ
            - Упражнения на творческое письмо
            ''',
            'bilingual_ratio': 1.0,
            'max_questions': 10,
            'vocabulary_per_lesson': 25,
            'sentence_length': 'сложные конструкции'
        }
    }
    
    # 🔥 ТИПЫ ЯЗЫКОВЫХ УПРАЖНЕНИЙ ПО УРОВНЯМ (СУЩЕСТВУЮЩАЯ)
    LANGUAGE_EXERCISE_TYPES = {
        'beginner': [
            'vocabulary',        # Словарный запас
            'simple_translation', # Простой перевод
            'fill_blank',        # Заполнение пропусков
            'matching',          # Сопоставление
            'multiple_choice'    # Выбор правильного варианта
        ],
        'intermediate': [
            'grammar',           # Грамматика
            'dialogue',          # Диалоги
            'sentence_building', # Составление предложений
            'reading',           # Чтение с вопросами
            'translation'        # Перевод
        ],
        'advanced': [
            'composition',       # Сочинение
            'error_correction',  # Исправление ошибок
            'debate',           # Дискуссия
            'essay',            # Эссе
            'summary'           # Резюмирование
        ]
    }
    
    # 🔥 ТИПЫ УПРАЖНЕНИЙ ПО УРОВНЯМ CEFR
    CEFR_EXERCISE_TYPES = {
        'A1': [
            'vocabulary_matching',  # Сопоставление слов с картинками
            'fill_blank_simple',    # Заполнение пропусков в простых предложениях
            'multiple_choice_basic',# Выбор из 2-3 вариантов
            'repeat_after_me',      # Повторение за учителем
            'word_order'            # Расстановка слов в правильном порядке
        ],
        'A2': [
            'dialogue_completion',  # Завершение диалога
            'sentence_translation', # Перевод простых предложений
            'true_false',           # Верно/неверно
            'wh_questions',         # Вопросы who, what, where, when
            'fill_blank_dialogue'   # Заполнение пропусков в диалоге
        ],
        'B1': [
            'paragraph_translation',# Перевод абзацев
            'grammar_correction',   # Исправление грамматических ошибок
            'reading_comprehension',# Чтение с вопросами
            'role_play',            # Ролевые игры
            'sentence_transformation' # Преобразование предложений
        ],
        'B2': [
            'essay_writing',        # Написание эссе
            'debate_preparation',   # Подготовка к дебатам
            'article_summary',      # Резюме статьи
            'phrasal_verbs',        # Упражнения на фразовые глаголы
            'idiom_matching'        # Сопоставление идиом
        ],
        'C1': [
            'academic_writing',     # Академическое письмо
            'presentation_prep',    # Подготовка презентации
            'literary_analysis',    # Литературный анализ
            'negotiation_practice', # Практика переговоров
            'translation_complex'   # Перевод сложных текстов
        ],
        'C2': [
            'creative_writing',     # Творческое письмо
            'simultaneous_translation', # Устный перевод
            'philosophical_discussion', # Философские дискуссии
            'literary_criticism',   # Литературная критика
            'research_proposal'     # Написание исследовательского предложения
        ]
    }
    
    @staticmethod
    def is_language_subject(subject_name: str) -> bool:
        """Определяет, является ли предмет языковым"""
        if not subject_name:
            return False
        
        subject_lower = subject_name.lower()
        
        for lang_config in LanguageIntegration.LANGUAGE_SUBJECTS.values():
            for russian_name in lang_config['russian_names']:
                if russian_name in subject_lower:
                    return True
        
        # Также проверяем английские названия
        for lang_name in LanguageIntegration.LANGUAGE_SUBJECTS.keys():
            if lang_name in subject_lower:
                return True
        
        return False
    
    @staticmethod
    def get_target_language(subject_name: str) -> Optional[str]:
        """Извлекает целевой язык из названия предмета"""
        if not subject_name:
            return None
        
        subject_lower = subject_name.lower()
        
        for lang_name, lang_config in LanguageIntegration.LANGUAGE_SUBJECTS.items():
            # Проверяем русские названия
            for russian_name in lang_config['russian_names']:
                if russian_name in subject_lower:
                    return lang_name
            
            # Проверяем английское название
            if lang_name in subject_lower:
                return lang_name
        
        return None
    
    @staticmethod
    def get_language_code(language_name: str) -> str:
        """Возвращает код языка для TTS"""
        language_name = language_name.lower()
        
        for lang_name, lang_config in LanguageIntegration.LANGUAGE_SUBJECTS.items():
            if lang_name == language_name:
                return lang_config['code']
        
        return 'en'  # По умолчанию английский
    
    @staticmethod
    def analyze_student_level(age: int, self_assessment: str = '') -> Dict:
        """
        Анализирует и определяет уровень языка ученика
        
        Args:
            age: Возраст ученика
            self_assessment: Самооценка уровня (beginner, intermediate, advanced)
            
        Returns:
            Dict с уровнем и настройками
        """
        # 🔥 ОСНОВНОЙ АЛГОРИТМ ПО ВОЗРАСТУ
        if age <= 10:
            suggested_level = 'beginner'
        elif age <= 14:
            suggested_level = 'intermediate'
        else:
            suggested_level = 'advanced'
        
        # 🔥 УЧИТЫВАЕМ САМООЦЕНКУ УЧЕНИКА
        if self_assessment and self_assessment in ['beginner', 'intermediate', 'advanced']:
            suggested_level = self_assessment
            debug_log(f"🎯 Использована самооценка ученика: {self_assessment}")
        
        # 🔥 ПОЛУЧАЕМ НАСТРОЙКИ ДЛЯ УРОВНЯ
        bilingual_ratio = 0.3
        max_questions = 5
        
        # Ищем настройки для любого языка (все языки имеют одинаковую структуру)
        for lang_config in LanguageIntegration.LANGUAGE_SUBJECTS.values():
            if suggested_level in lang_config['levels']:
                level_config = lang_config['levels'][suggested_level]
                bilingual_ratio = level_config['bilingual_ratio']
                max_questions = level_config['max_questions']
                break
        
        return {
            'level': suggested_level,
            'bilingual_ratio': bilingual_ratio,
            'max_questions': max_questions,
            'exercise_types': LanguageIntegration.LANGUAGE_EXERCISE_TYPES.get(suggested_level, []),
            'description': f'Уровень {suggested_level} ({int(bilingual_ratio * 100)}% иностранного языка)'
        }
    
    @staticmethod
    def detect_cefr_level(age: int, self_assessment: str = '', education_level: str = '') -> str:
        """
        Автоматически определяет уровень CEFR на основе возраста, самооценки и уровня образования
        
        Args:
            age: Возраст ученика
            self_assessment: Самооценка уровня (A1, B2 и т.д.)
            education_level: Уровень образования (5, 6, ... adult)
            
        Returns:
            Уровень CEFR (A1, A2, B1, B2, C1, C2)
        """
        # 🔥 ЕСЛИ ЭТО ШКОЛЬНИК - используем упрощенную систему
        if education_level and education_level != 'adult':
            # Для школьников используем возраст
            if age <= 10:
                return 'A1'
            elif age <= 12:
                return 'A2'
            elif age <= 14:
                return 'B1'
            elif age <= 16:
                return 'B2'
            else:
                return 'C1'
        
        # 🔥 ЕСЛИ ЭТО ВЗРОСЛЫЙ (adult)
        # Проверяем самооценку
        if self_assessment:
            self_assessment_upper = self_assessment.upper()
            if self_assessment_upper in LanguageIntegration.CEFR_LEVELS:
                return self_assessment_upper
            
            # Маппинг русских/английских названий
            level_mapping = {
                'начинающий': 'A1',
                'начальный': 'A1',
                'beginner': 'A1',
                'элементарный': 'A2',
                'elementary': 'A2',
                'средний': 'B1',
                'intermediate': 'B1',
                'выше среднего': 'B2',
                'upper intermediate': 'B2',
                'продвинутый': 'C1',
                'advanced': 'C1',
                'профессиональный': 'C2',
                'proficient': 'C2',
                'в совершенстве': 'C2'
            }
            
            if self_assessment.lower() in level_mapping:
                return level_mapping[self_assessment.lower()]
        
        # 🔥 ДЕФОЛТНЫЕ ЗНАЧЕНИЯ ПО ВОЗРАСТУ ДЛЯ ВЗРОСЛЫХ
        if age < 18:
            return 'B1'
        elif age < 25:
            return 'B2'
        elif age < 40:
            return 'C1'
        else:
            return 'C1'  # Взрослые обычно имеют более высокий уровень
    
    @staticmethod
    def get_cefr_level_config(cefr_level: str) -> Dict:
        """
        Возвращает конфигурацию для уровня CEFR
        
        Args:
            cefr_level: Уровень CEFR (A1, A2, B1, B2, C1, C2)
            
        Returns:
            Конфигурация уровня
        """
        level = cefr_level.upper()
        default_config = {
            'description': 'Неизвестный уровень',
            'prompt_adjustment': '',
            'bilingual_ratio': 0.5,
            'max_questions': 5,
            'vocabulary_per_lesson': 10,
            'sentence_length': 'стандартная',
            'exercise_types': []
        }
        
        if level in LanguageIntegration.CEFR_LEVELS:
            config = LanguageIntegration.CEFR_LEVELS[level].copy()
            config['exercise_types'] = LanguageIntegration.CEFR_EXERCISE_TYPES.get(level, [])
            return config
        
        return default_config
    
    @staticmethod
    def create_bilingual_lesson_prompt(
        topic: str, 
        target_language: str = 'english',
        level: str = 'beginner',
        bilingual_ratio: float = 0.3
    ) -> str:
        """
        Создает промт для генерации билингвального языкового урока
        
        Args:
            topic: Тема урока
            target_language: Изучаемый язык
            level: Уровень ученика
            bilingual_ratio: Доля иностранного языка (0.0-1.0)
            
        Returns:
            Промт для LLM
        """
        # 🔥 РАСЧЕТ ПРОЦЕНТОВ
        foreign_percent = int(bilingual_ratio * 100)
        russian_percent = 100 - foreign_percent
        
        # 🔥 НАСТРОЙКИ ДЛЯ РАЗНЫХ УРОВНЕЙ
        level_settings = {
            'beginner': {
                'style': 'ОЧЕНЬ ПРОСТО и ПОСТЕПЕННО',
                'explanation': 'Объясняй полностью на русском, очень подробно',
                'examples': 'Простейшие примеры, максимум 3-4 слова',
                'grammar': 'Одно простое правило за урок',
                'questions': 'Очень простые вопросы, можно ответить одним словом',
                'transcription': 'не давай транскрипцию'
            },
            'intermediate': {
                'style': 'СБАЛАНСИРОВАННО и ПРАКТИЧНО',
                'explanation': 'Краткие объяснения на русском, больше практики',
                'examples': 'Развернутые примеры, естественные фразы',
                'grammar': '1-2 связанных правила',
                'questions': 'Вопросы требующие коротких предложений',
                'transcription': 'не давай транскрипцию'
            },
            'advanced': {
                'style': 'ИММЕРСИВНО и ЕСТЕСТВЕННО',
                'explanation': 'Минимум объяснений, максимум практики',
                'examples': 'Сложные конструкции, идиомы, сленг если уместно',
                'grammar': 'Нюансы и исключения',
                'questions': 'Вопросы для дискуссии, требующие развернутого ответа',
                'transcription': 'не давай транскрипцию'
            }
        }
        
        settings = level_settings.get(level, level_settings['beginner'])
        
        # 🔥 ФОРМИРУЕМ ПРОМТ
        prompt = f"""
        🔥 СОЗДАЙ БИЛИНГВАЛЬНЫЙ ЯЗЫКОВОЙ УРОК
        
        ===== ОСНОВНЫЕ ПАРАМЕТРЫ =====
        ТЕМА УРОКА: {topic}
        ИЗУЧАЕМЫЙ ЯЗЫК: {target_language.upper()}
        УРОВЕНЬ УЧЕНИКА: {level.upper()}
        СООТНОШЕНИЕ ЯЗЫКОВ: {russian_percent}% русский / {foreign_percent}% {target_language}
        СТИЛЬ: {settings['style']}
        
        ===== КРИТИЧЕСКИ ВАЖНЫЕ ТРЕБОВАНИЯ =====
        
        1. 🎯 СТРУКТУРА УРОКА:
           • Объяснения грамматики и правил: {settings['explanation']}
           • Примеры и упражнения: {foreign_percent}% на {target_language}
           • Новые слова: сначала на {target_language}, потом перевод
        
        2. 📝 ФОРМАТИРОВАНИЕ:
           • Новые слова: **house** [haʊs] (дом)
           • Примеры предложений: **I live in a big house.** (Я живу в большом доме.)
           • Грамматические правила: объясняй на русском, примеры на {target_language}
           • {settings['transcription']}
        
        3. ❓ ИНТЕРАКТИВНЫЕ ЭЛЕМЕНТЫ (ОБЯЗАТЕЛЬНО ВСТАВИТЬ В УРОК):
           • 2-3 проверочных вопроса НА {target_language.upper()} в середине урока
           • 1 упражнение на перевод (русский → {target_language})
           • 1 мини-диалог на {target_language} по теме
           • {settings['questions']}
        
        4. ✨ ТИПЫ УПРАЖНЕНИЙ ДЛЯ УРОВНЯ {level.upper()}:
           • Заполнение пропусков (fill in the blanks)
           • Составление предложений (sentence building)
           • Выбор правильного варианта (multiple choice)
           • Сопоставление (matching words with translations)
        
        5. 🎓 ЯЗЫКОВАЯ ПРАКТИКА:
           • {settings['examples']}
           • {settings['grammar']}
           • Естественные разговорные фразы
           • Практические, полезные выражения
        
        ===== ПРИМЕР СТРУКТУРЫ ДЛЯ УРОВНЯ {level.upper()} =====
        
        [РУССКИЙ] Введение в тему. Объяснение почему это важно.
        [{target_language.upper()}] New vocabulary: **word1** [транскрипция] (перевод), **word2** (перевод)
        [РУССКИЙ] Объяснение произношения и использования слов.
        [{target_language.upper()}] Example sentences: **Sentence1.** (Перевод1)
        [РУССКИЙ] Грамматическое правило или особенность.
        [{target_language.upper()}] ❓ Interactive question: Question here?
        [РУССКИЙ] Подсказка как ответить.
        [{target_language.upper()}] Mini-dialogue: A: Phrase1. B: Response1.
        [РУССКИЙ] Объяснение диалога.
        [{target_language.upper()}] Practice exercise: Complete/Fill/Translate...
        
        ===== ВАЖНЕЙШИЕ ПРАВИЛА =====
        
        🔥 НИКОГДА НЕ ДЕЛАЙ:
        • Резких переходов между языками
        • Слишком сложных конструкций для уровня {level}
        • Чисто теоретических объяснений без примеров
        
        🔥 ВСЕГДА ДЕЛАЙ:
        • Плавные, естественные переходы между русским и {target_language}
        • Практические, полезные примеры
        • Учет уровня {level} во всем
        
        ===== ИТОГОВАЯ ИНСТРУКЦИЯ =====
        
        Создай интересный, практичный и эффективный билингвальный урок.
        Ученик должен чувствовать прогресс и понимание.
        Урок должен мотивировать продолжать изучение языка.
        
        Верни ТОЛЬКО текст урока, без дополнительных комментариев.
        """
        
        return prompt
    
    @staticmethod
    def create_bilingual_lesson_prompt_cefr(
        topic: str, 
        target_language: str = 'english',
        cefr_level: str = 'A1',
        **kwargs
    ) -> str:
        """
        Создает промпт для генерации билингвального урока с учетом уровня CEFR
        
        Args:
            topic: Тема урока
            target_language: Изучаемый язык
            cefr_level: Уровень CEFR (A1, A2, B1, B2, C1, C2)
            
        Returns:
            Промпт для LLM
        """
        # Получаем конфигурацию уровня
        level_config = LanguageIntegration.get_cefr_level_config(cefr_level)
        
        bilingual_ratio = level_config.get('bilingual_ratio', 0.5)
        foreign_percent = int(bilingual_ratio * 100)
        russian_percent = 100 - foreign_percent
        
        # 🔥 ОСНОВНОЙ ПРОМПТ С УЧЕТОМ CEFR
        prompt = f"""
        🔥 СОЗДАЙ БИЛИНГВАЛЬНЫЙ ЯЗЫКОВОЙ УРОК ДЛЯ ВЗРОСЛЫХ (УРОВЕНЬ {cefr_level.upper()})
        
        ===== ОСНОВНЫЕ ПАРАМЕТРЫ =====
        ТЕМА УРОКА: {topic}
        ИЗУЧАЕМЫЙ ЯЗЫК: {target_language.upper()}
        УРОВЕНЬ CEFR: {cefr_level.upper()} ({level_config.get('description', '')})
        СООТНОШЕНИЕ ЯЗЫКОВ: {russian_percent}% русский / {foreign_percent}% {target_language}
        СЛОЖНОСТЬ: {level_config.get('sentence_length', 'стандартная')}
        
        ===== НАСТРОЙКИ УРОВНЯ {cefr_level.upper()} =====
        {level_config.get('prompt_adjustment', '')}
        
        ===== КРИТИЧЕСКИ ВАЖНЫЕ ТРЕБОВАНИЯ ДЛЯ УРОВНЯ {cefr_level.upper()} =====
        
        1. 🎯 СТРУКТУРА УРОКА:
           • Соотношение языков: строго {russian_percent}% русский, {foreign_percent}% {target_language}
           • Новые слова: не более {level_config.get('vocabulary_per_lesson', 10)} слов за урок
           • Длина предложений: {level_config.get('sentence_length', 'стандартная')}
           • Темп урока: медленный и повторяющиися для A1-A2, быстрее для B1+
        
        2. 📝 ФОРМАТИРОВАНИЕ:
           • Новые слова: **house** (дом) - НЕ давай транскрипцию в скобках []
           • Примеры: **I have a house.** (У меня есть дом.)
           • Грамматика: объясняй на русском, примеры на {target_language}
           • Для уровней A1-A2: используй МНОГО повторении
           • Для уровней B2-C2: добавляи идиомы и сложные конструкции
        
        3. ❓ ИНТЕРАКТИВНЫЕ ЭЛЕМЕНТЫ:
           • Вопросы на понимание: {level_config.get('max_questions', 5)} вопросов
           • Типы упражнении: {', '.join(level_config.get('exercise_types', ['базовые']))}
           • Практика: реалистичные ситуации для взрослых
        
        4. ✨ ПРАКТИЧЕСКАЯ ПОЛЕЗНОСТЬ:
           • Фразы, которые можно использовать СЕГОДНЯ
           • Ситуации из реальной жизни взрослых
           • Деловая лексика для уровней B1+
           • Культурные нюансы для уровней B2+
        
        5. 🎓 ОСОБЕННОСТИ ВЗРОСЛЫХ УЧЕНИКОВ:
           • Учитываи жизненныи опыт
           • Деловые/профессиональные темы для уровней B1+
           • Абстрактные понятия для уровней B2+
           • Критическое мышление для уровней C1-C2
        
        ===== ТИПЫ УПРАЖНЕНИЙ ДЛЯ УРОВНЯ {cefr_level.upper()} =====
        {', '.join(level_config.get('exercise_types', []))}
        
        ===== ПРИМЕР СТРУКТУРЫ ДЛЯ УРОВНЯ {cefr_level.upper()} =====
        
        {"[ТОЛЬКО ДЛЯ A1-A2]" if cefr_level in ['A1', 'A2'] else ""}
        [РУССКИЙ] Очень простое введение. "Сегодня мы научимся..."
        [{target_language.upper()}] New words: **word1** (перевод1), **word2** (перевод2)
        [РУССКИЙ] Повтори слова 2 раза. "Скажи со мной: word1, word1"
        [{target_language.upper()}] Simple sentence: **Sentence.** (Translation)
        [РУССКИЙ] Объяснение. "Это значит..."
        [{target_language.upper()}] ❓ Simple question: Question? (Ответ: одно слово)
        
        {"[ДЛЯ B1-B2]" if cefr_level in ['B1', 'B2'] else ""}
        [РУССКИЙ] Введение с контекстом. "В реальной жизни это используется..."
        [{target_language.upper()}] Vocabulary: **word1** (перевод), **word2** (перевод)
        [{target_language.upper()}] Examples: **Sentence1.** (Translation1) **Sentence2.** (Translation2)
        [РУССКИЙ] Краткое объяснение грамматики
        [{target_language.upper()}] Dialogue: A: ... B: ...
        [{target_language.upper()}] ❓ Comprehension questions: 1. ... 2. ...
        
        {"[ДЛЯ C1-C2]" if cefr_level in ['C1', 'C2'] else ""}
        [{target_language.upper()}] Introduction and context (90% на целевом языке)
        [{target_language.upper()}] Advanced vocabulary with nuances
        [{target_language.upper()}] Complex examples with cultural references
        [{target_language.upper()}] Grammar nuances and exceptions
        [{target_language.upper()}] Debate topic or essay question
        [РУССКИЙ] ТОЛЬКО если нужно объяснить очень сложную концепцию
        
        ===== ВАЖНЕИШИЕ ПРАВИЛА ДЛЯ ВЗРОСЛЫХ =====
        
        🔥 НИКОГДА НЕ ДЕЛАЙ С ВЗРОСЛЫМИ:
        • Детских примеров (игрушки, мультики и т.д.)
        • Слишком медленного темпа для уровней B1+
        • Очевидных объяснений для продвинутых уровней
        
        🔥 ВСЕГДА ДЕЛАЙ С ВЗРОСЛЫМИ:
        • Уважаи их жизненныи опыт
        • Даваи практические, полезные знания
        • Обсуждаи актуальные темы
        • Для уровней B1+ добавляи профессиональную лексику
        
        ===== ИТОГОВАЯ ИНСТРУКЦИЯ =====
        
        Создаи КАЧЕСТВЕННЫИ, ПРАКТИЧНЫИ урок для взрослого ученика уровня {cefr_level}.
        Урок должен быть полезным СЕГОДНЯ.
        Ученик должен почувствовать прогресс и мотивацию продолжать.
        
        Верни ТОЛЬКО текст урока, без дополнительных комментариев.
        """
        
        return prompt
    
    @staticmethod
    def create_language_exercise_prompt(
        exercise_type: str,
        topic: str,
        target_language: str,
        level: str,
        vocabulary: List[str] = None
    ) -> str:
        """
        Создает промт для генерации языкового упражнения
        
        Args:
            exercise_type: Тип упражнения
            topic: Тема упражнения
            target_language: Изучаемый язык
            level: Уровень ученика
            vocabulary: Список слов для упражнения (опционально)
            
        Returns:
            Промт для LLM
        """
        
        exercise_templates = {
            'vocabulary': f"""
            СОЗДАЙ УПРАЖНЕНИЕ НА СЛОВАРНЫЙ ЗАПАС
            
            ЯЗЫК: {target_language}
            УРОВЕНЬ: {level}
            ТЕМА: {topic}
            
            Создай упражнение "Сопоставление слов с переводами".
            
            ВКЛЮЧИ:
            1. 5-7 слов на {target_language} по теме "{topic}"
            2. Их переводы на русский в перемешанном порядке
            3. Инструкцию на русском
            
            ФОРМАТ:
            [Инструкция на русском]
            
            Сопоставьте слова на {target_language} с их переводом:
            
            1. Word1
            2. Word2
            3. Word3
            4. Word4
            5. Word5
            
            A. Перевод1
            B. Перевод2
            C. Перевод3
            D. Перевод4
            E. Перевод5
            
            ПРАВИЛЬНЫЕ ОТВЕТЫ: [не включай в упражнение]
            
            Слова должны быть полезными и соответствующими уровню {level}.
            """,
            
            'translation': f"""
            СОЗДАЙ УПРАЖНЕНИЕ НА ПЕРЕВОД
            
            ЯЗЫК: {target_language}
            УРОВЕНЬ: {level}
            ТЕМА: {topic}
            
            Создай упражнение "Переведите предложения на {target_language}".
            
            ВКЛЮЧИ:
            1. 3-5 простых предложений на русском по теме "{topic}"
            2. Инструкцию на русском
            3. Подсказки для сложных конструкций (для уровня {level})
            
            ФОРМАТ:
            [Инструкция на русском]
            
            Переведите на {target_language}:
            
            1. [Предложение на русском]
            2. [Предложение на русском]
            3. [Предисание на русском]
            
            ПРАВИЛЬНЫЕ ПЕРЕВОДЫ: [не включай в упражнение]
            
            Предложения должны быть понятными для уровня {level}.
            """,
            
            'fill_blank': f"""
            СОЗДАЙ УПРАЖНЕНИЕ "FILL IN THE BLANKS"
            
            ЯЗЫК: {target_language}
            УРОВЕНЬ: {level}
            ТЕМА: {topic}
            
            Создай упражнение "Заполните пропуски подходящими словами".
            
            ВКЛЮЧИ:
            1. Короткий текст на {target_language} с 3-5 пропусками
            2. Список слов для выбора (если нужно)
            3. Инструкцию на русском
            
            ФОРМАТ:
            [Инструкция на русском]
            
            Текст:
            [Текст с пропусками обозначенными как ______]
            
            Слова для выбора (если нужно):
            [word1, word2, word3]
            
            ПРАВИЛЬНЫЕ ОТВЕТЫ: [не включай в упражнение]
            
            Текст должен соответствовать уровню {level}.
            """,
            
            'dialogue': f"""
            СОЗДАЙ ДИАЛОГОВОЕ УПРАЖНЕНИЕ
            
            ЯЗЫК: {target_language}
            УРОВЕНЬ: {level}
            ТЕМА: {topic}
            
            Создай упражнение с диалогом.
            
            ВКЛЮЧИ:
            1. Короткий диалог (4-6 реплик) на {target_language}
            2. Вопросы на понимание диалога
            3. Задание "дополните диалог"
            4. Инструкцию на русском
            
            ФОРМАТ:
            [Инструкция на русском]
            
            Диалог:
            A: [Реплика 1]
            B: [Реплика 2]
            A: [Реплика 3]
            B: [Реплика 4]
            
            Вопросы:
            1. [Вопрос на понимание]
            2. [Вопрос на понимание]
            
            Задание:
            [Задание для ученика]
            
            ПРАВИЛЬНЫЕ ОТВЕТЫ: [не включай в упражнение]
            
            Диалог должен быть естественным для уровня {level}.
            """,
            
            'grammar': f"""
            СОЗДАЙ ГРАММАТИЧЕСКОЕ УПРАЖНЕНИЕ
            
            ЯЗЫК: {target_language}
            УРОВЕНЬ: {level}
            ТЕМА: {topic}
            
            Создай упражнение на грамматика.
            
            ВКЛЮЧИ:
            1. Краткое объяснение правила на русском
            2. 3-5 примеров на {target_language}
            3. Упражнение на применение правила
            4. Инструкцию на русском
            
            ФОРМАТ:
            [Инструкция на русском]
            
            Правило:
            [Объяснение правила на русском]
            
            Примеры:
            1. [Пример 1]
            2. [Пример 2]
            
            Упражнение:
            [Упражнение для ученика]
            
            ПРАВИЛЬНЫЕ ОТВЕТЫ: [не включай в упражнение]
            
            Правило должно быть понятным для уровня {level}.
            """
        }
        
        # 🔥 ИСПОЛЬЗУЕМ ОБЩИЙ ШАБЛОН ЕСЛИ ТИП НЕ НАЙДЕН
        default_template = f"""
        СОЗДАЙ ЯЗЫКОВОЕ УПРАЖНЕНИЕ
        
        ТИП: {exercise_type}
        ЯЗЫК: {target_language}
        УРОВЕНЬ: {level}
        ТЕМА: {topic}
        
        Создай упражнение по изучению {target_language}.
        Упражнение должно соответствовать уровню {level}.
        Включи инструкцию на русском.
        
        Верни упражнение в понятном формате.
        """
        
        return exercise_templates.get(exercise_type, default_template)
    
    @staticmethod
    def create_cefr_exercise_prompt(
        cefr_level: str,
        exercise_type: str,
        topic: str,
        target_language: str = 'english',
        vocabulary: List[str] = None
    ) -> str:
        """
        Создает промпт для генерации упражнения с учетом уровня CEFR
        
        Args:
            cefr_level: Уровень CEFR (A1, A2, B1, B2, C1, C2)
            exercise_type: Тип упражнения
            topic: Тема упражнения
            target_language: Изучаемый язык
            vocabulary: Список слов (опционально)
            
        Returns:
            Промпт для LLM
        """
        level_config = LanguageIntegration.get_cefr_level_config(cefr_level)
        
        # 🔥 ШАБЛОНЫ ДЛЯ РАЗНЫХ УРОВНЕЙ CEFR
        cefr_exercise_templates = {
            'A1': {
                'vocabulary_matching': f"""
                СОЗДАЙ УПРАЖНЕНИЕ ДЛЯ НАЧИНАЮЩИХ (A1)
                
                УРОВЕНЬ: A1 (Начинающий)
                ЯЗЫК: {target_language}
                ТЕМА: {topic}
                
                Создай упражнение "Сопоставь картинку со словом".
                
                ВКЛЮЧИ:
                1. 5 очень простых слов на {target_language} по теме "{topic}"
                2. Простые описания "картинок" на русском (например: "🍎 - красный фрукт")
                3. ОЧЕНЬ простую инструкцию на русском
                
                ФОРМАТ:
                [Инструкция: Соедини слово с картинкой]
                
                Слова:
                1. apple
                2. house
                3. car
                4. book
                5. cat
                
                Картинки:
                A. 🍎 - красный фрукт
                B. 🏠 - место где живут
                C. 🚗 - транспорт
                D. 📖 - можно читать
                E. 🐱 - домашнее животное
                
                ПРАВИЛЬНЫЕ ОТВЕТЫ: [не включай]
                
                Все должно быть ОЧЕНЬ ПРОСТО!
                """,
                
                'fill_blank_simple': f"""
                СОЗДАЙ ПРОСТОЕ УПРАЖНЕНИЕ "ЗАПОЛНИ ПРОПУСКИ" (A1)
                
                УРОВЕНЬ: A1
                ЯЗЫК: {target_language}
                ТЕМА: {topic}
                
                Создай ОЧЕНЬ простое упражнение с 3 пропусками.
                
                ВКЛЮЧИ:
                1. Простое предложение с пропусками
                2. Слова для выбора (max 3 слова)
                3. Простую инструкцию на русском
                
                ФОРМАТ:
                [Инструкция: Выбери правильное слово]
                
                I ___ a student. (am / is / are)
                My name ___ Anna. (am / is / are)
                I ___ 25 years old. (am / is / are)
                
                Слова: am, is, are
                
                ПРАВИЛЬНЫЕ ОТВЕТЫ: [не включай]
                
                Максимально просто!
                """
            },
            'B1': {
                'reading_comprehension': f"""
                СОЗДАЙ УПРАЖНЕНИЕ НА ЧТЕНИЕ С ВОПРОСАМИ (B1)
                
                УРОВЕНЬ: B1 (Средний)
                ЯЗЫК: {target_language}
                ТЕМА: {topic}
                
                Создай короткий текст (5-7 предложений) и 3 вопроса.
                
                ВКЛЮЧИ:
                1. Текст на {target_language} по теме "{topic}"
                2. 3 вопроса на понимание
                3. Инструкцию на русском
                
                ФОРМАТ:
                [Инструкция: Прочитай текст и ответь на вопросы]
                
                Текст:
                [Короткий текст на целевом языке]
                
                Вопросы:
                1. [Вопрос 1]
                2. [Вопрос 2]
                3. [Вопрос 3]
                
                ПРАВИЛЬНЫЕ ОТВЕТЫ: [не включай]
                
                Текст должен быть понятным для уровня B1.
                """
            },
            'C1': {
                'academic_writing': f"""
                СОЗДАЙ АКАДЕМИЧЕСКОЕ УПРАЖНЕНИЕ (C1)
                
                УРОВЕНЬ: C1 (Продвинутый)
                ЯЗЫК: {target_language}
                ТЕМА: {topic}
                
                Создай задание для академического письма.
                
                ВКЛЮЧИ:
                1. Тему для эссе или исследования
                2. Требования (объем, структура, стиль)
                3. Критерии оценки
                4. Инструкцию на {target_language}
                
                ФОРМАТ:
                [Instruction in {target_language}]
                
                Topic: [Complex topic for essay]
                
                Requirements:
                - Length: 300-500 words
                - Structure: Introduction, Body, Conclusion
                - Style: Academic, formal
                - Include: Arguments, examples, counterarguments
                
                Assessment Criteria:
                1. Content and arguments (40%)
                2. Structure and organization (30%)
                3. Language and vocabulary (20%)
                4. Grammar and accuracy (10%)
                
                CORRECT ANSWERS: [not included - this is a writing task]
                
                Make it challenging for C1 level.
                """
            }
        }
        
        # Ищем шаблон для уровня и типа
        level_templates = cefr_exercise_templates.get(cefr_level, {})
        
        if exercise_type in level_templates:
            return level_templates[exercise_type]
        
        # Fallback на общий шаблон для уровня
        default_cefr_template = f"""
        СОЗДАЙ УПРАЖНЕНИЕ ДЛЯ УРОВНЯ {cefr_level.upper()}
        
        УРОВЕНЬ: {cefr_level} ({level_config.get('description', '')})
        ЯЗЫК: {target_language}
        ТЕМА: {topic}
        ТИП УПРАЖНЕНИЯ: {exercise_type}
        
        Создай упражнение соответствующее уровню {cefr_level}.
        
        ОСОБЕННОСТИ УРОВНЯ {cefr_level.upper()}:
        - Сложность: {level_config.get('sentence_length', 'стандартная')}
        - Соотношение языков: {int((1 - level_config.get('bilingual_ratio', 0.5)) * 100)}% русский, {int(level_config.get('bilingual_ratio', 0.5) * 100)}% {target_language}
        - Типичные упражнения: {', '.join(level_config.get('exercise_types', []))}
        
        ВКЛЮЧИ:
        1. Упражнение соответствующее уровню {cefr_level}
        2. Инструкцию на {'русском' if cefr_level in ['A1', 'A2', 'B1'] else target_language}
        3. Четкие формулировки
        
        Создай качественное упражнение для взрослого ученика.
        """
        
        return default_cefr_template
    
    @staticmethod
    def split_text_by_language(text: str) -> List[Tuple[str, str]]:
        """
        Разделяет текст на фрагменты по языку
        
        Args:
            text: Исходный текст
            
        Returns:
            List of (fragment, language_code)
        """
        if not text:
            return []
        
        # 🔥 ШАБЛОН ДЛЯ ПОИСКА ПОСЛЕДОВАТЕЛЬНОСТЕЙ ОДНОГО ЯЗЫКА
        pattern = r'([а-яА-ЯёЁ][а-яА-ЯёЁ\s.,!?;:\'\"-]*|[a-zA-Z][a-zA-Z\s.,!?;:\'\"-]*)'
        
        fragments = re.findall(pattern, text)
        result = []
        
        for fragment in fragments:
            fragment = fragment.strip()
            if not fragment:
                continue
            
            # Определяем язык фрагмента
            has_cyrillic = bool(re.search(r'[а-яА-ЯёЁ]', fragment))
            has_latin = bool(re.search(r'[a-zA-Z]', fragment))
            
            if has_cyrillic and not has_latin:
                lang = 'ru'
            elif has_latin and not has_cyrillic:
                lang = 'en'  # По умолчанию английский, можно уточнить
            else:
                # Смешанный фрагмент - разбиваем дальше
                sub_fragments = LanguageIntegration._split_mixed_fragment(fragment)
                result.extend(sub_fragments)
                continue
            
            result.append((fragment, lang))
        
        return result
    
    @staticmethod
    def _split_mixed_fragment(fragment: str) -> List[Tuple[str, str]]:
        """Разбивает смешанные фрагменты на чистые"""
        result = []
        
        # 🔥 ИЩЕМ ПЕРЕКЛЮЧЕНИЯ МЕЖДУ ЯЗЫКАМИ
        current_lang = None
        current_text = []
        
        for char in fragment:
            is_cyrillic = bool(re.match(r'[а-яА-ЯёЁ]', char))
            is_latin = bool(re.match(r'[a-zA-Z]', char))
            
            char_lang = 'ru' if is_cyrillic else 'en' if is_latin else None
            
            if char_lang:
                if current_lang is None:
                    current_lang = char_lang
                    current_text.append(char)
                elif current_lang == char_lang:
                    current_text.append(char)
                else:
                    # Смена языка
                    if current_text:
                        result.append((''.join(current_text), current_lang))
                    current_lang = char_lang
                    current_text = [char]
            else:
                # Не-буква (пробел, знак препинания)
                current_text.append(char)
        
        # Добавляем последний фрагмент
        if current_text:
            result.append((''.join(current_text), current_lang or 'ru'))
        
        return result
    
    @staticmethod
    def validate_language_answer(
        student_answer: str,
        expected_answer: str,
        language: str,
        level: str
    ) -> Dict:
        """
        Валидация ответа ученика (упрощенная версия)
        Полная валидация будет через LLM
        
        Args:
            student_answer: Ответ ученика
            expected_answer: Ожидаемый ответ
            language: Язык упражнения
            level: Уровень ученика
            
        Returns:
            Dict с результатом валидации
        """
        student_lower = student_answer.strip().lower()
        expected_lower = expected_answer.strip().lower()
        
        # 🔥 ПРОСТАЯ ПРОВЕРКА (для демонстрации)
        # В реальной системе это делает LLM
        
        if student_lower == expected_lower:
            return {
                'correct': True,
                'score': 1.0,
                'feedback': 'Отлично! Правильный ответ.'
            }
        
        # Частичное совпадение
        if expected_lower in student_lower or student_lower in expected_lower:
            return {
                'correct': True,
                'score': 0.7,
                'feedback': 'Почти правильно! Можно было бы точнее.'
            }
        
        # Совпадение по ключевым словам
        student_words = set(re.findall(r'\b\w+\b', student_lower))
        expected_words = set(re.findall(r'\b\w+\b', expected_lower))
        common_words = student_words.intersection(expected_words)
        
        if common_words and len(common_words) >= len(expected_words) * 0.5:
            return {
                'correct': True,
                'score': 0.5,
                'feedback': 'Есть правильные элементы, но не полностью.'
            }
        
        return {
            'correct': False,
            'score': 0.0,
            'feedback': 'Попробуйте еще раз. Обратите внимание на правильный ответ.'
        }
    
    @staticmethod
    def get_adult_study_modes() -> List[Dict]:
        """
        Возвращает доступные режимы обучения для взрослых
        
        Returns:
            Список режимов обучения
        """
        return [
            {
                'id': 'language',
                'name': 'Английский язык',
                'description': 'Структурированное изучение английского по уровням CEFR',
                'has_lessons': True,
                'has_progress': True
            },
            {
                'id': 'anything',
                'name': 'Изучать что угодно',
                'description': 'Свободный диалог на любые темы без структурированных уроков',
                'has_lessons': False,
                'has_progress': False
            }
        ]
    
    @staticmethod
    def get_available_cefr_levels() -> List[Dict]:
        """
        Возвращает список доступных уровней CEFR
        
        Returns:
            Список уровней с описанием
        """
        levels = []
        for level_id, config in LanguageIntegration.CEFR_LEVELS.items():
            levels.append({
                'id': level_id,
                'name': f"{level_id} ({config['description']})",
                'description': config['description'],
                'bilingual_ratio': config['bilingual_ratio'],
                'foreign_percent': int(config['bilingual_ratio'] * 100),
                'russian_percent': 100 - int(config['bilingual_ratio'] * 100),
                'max_questions': config['max_questions'],
                'vocabulary_per_lesson': config['vocabulary_per_lesson'],
                'sentence_length': config['sentence_length']
            })
        return levels
    
    @staticmethod
    def generate_adult_lesson_path(cefr_level: str, lesson_number: int, topic: str = None) -> str:
        """
        Генерирует путь для урока взрослого студента
        
        Args:
            cefr_level: Уровень CEFR
            lesson_number: Номер урока
            topic: Тема урока (опционально)
            
        Returns:
            Путь к файлу урока
        """
        import re
        
        # Базовая структура
        base_path = f"students/adult_language/{cefr_level}_english"
        
        # Формируем имя файла
        if topic:
            # Очищаем тему для использования в имени файла
            topic_slug = re.sub(r'[^\w\s-]', '', topic.lower())
            topic_slug = re.sub(r'\s+', '_', topic_slug)
            topic_slug = topic_slug[:30]
            filename = f"lesson_{lesson_number:02d}_{topic_slug}.txt"
        else:
            filename = f"lesson_{lesson_number:02d}_general.txt"
        
        return f"{base_path}/{filename}"
    
    @staticmethod
    def should_use_cefr_prompt(education_level: str, study_mode: str = None) -> bool:
        """
        Определяет, нужно ли использовать промпты CEFR
        
        Args:
            education_level: Уровень образования (5, 6, ..., 'adult')
            study_mode: Режим обучения ('language' или 'anything')
            
        Returns:
            True если нужно использовать CEFR промпты
        """
        return (education_level == 'adult' and 
                study_mode == 'language')

# 🔥 УТИЛИТНЫЕ ФУНКЦИИ ДЛЯ БЫСТРОГО ДОСТУПА

def is_language_subject(subject_name: str) -> bool:
    """Проверка, является ли предмет языковым (удобная обертка)"""
    return LanguageIntegration.is_language_subject(subject_name)

def get_language_settings(subject_name: str, student_age: int = 12) -> Dict:
    """Получение настроек языка для предмета и ученика"""
    target_language = LanguageIntegration.get_target_language(subject_name)
    
    if not target_language:
        return {}
    
    level_info = LanguageIntegration.analyze_student_level(student_age)
    
    return {
        'target_language': target_language,
        'language_code': LanguageIntegration.get_language_code(target_language),
        'level': level_info['level'],
        'bilingual_ratio': level_info['bilingual_ratio'],
        'exercise_types': level_info['exercise_types'],
        'max_questions': level_info['max_questions']
    }

def detect_cefr_level(age: int, self_assessment: str = '', education_level: str = '') -> str:
    """Определение уровня CEFR (удобная обертка)"""
    return LanguageIntegration.detect_cefr_level(age, self_assessment, education_level)

def get_cefr_level_config(cefr_level: str) -> Dict:
    """Получение конфигурации уровня CEFR (удобная обертка)"""
    return LanguageIntegration.get_cefr_level_config(cefr_level)

def create_bilingual_lesson_prompt_cefr(topic: str, target_language: str = 'english', cefr_level: str = 'A1', **kwargs) -> str:
    """Создание промпта для урока с учетом CEFR (удобная обертка)"""
    return LanguageIntegration.create_bilingual_lesson_prompt_cefr(topic, target_language, cefr_level, **kwargs)

def get_adult_study_modes() -> List[Dict]:
    """Получение режимов обучения для взрослых (удобная обертка)"""
    return LanguageIntegration.get_adult_study_modes()

def get_available_cefr_levels() -> List[Dict]:
    """Получение доступных уровней CEFR (удобная обертка)"""
    return LanguageIntegration.get_available_cefr_levels()

# 🔥 ТЕСТИРОВАНИЕ
if __name__ == "__main__":
    print("🧪 Тестирование LanguageIntegration с поддержкой CEFR...")
    
    # Тест 1: Определение языкового предмета
    test_subjects = [
        "английский язык",
        "математика",
        "французский",
        "история",
        "english",
        "немецкий язык"
    ]
    
    print("\n📚 Определение языковых предметов:")
    for subject in test_subjects:
        is_lang = LanguageIntegration.is_language_subject(subject)
        target_lang = LanguageIntegration.get_target_language(subject)
        print(f"  {subject}: {'✅ Языковой' if is_lang else '❌ Не языковой'} ({target_lang})")
    
    # Тест 2: Определение уровня CEFR
    print("\n👤 Определение уровня CEFR:")
    test_cases = [
        (10, "beginner", "5"),
        (15, "intermediate", "8"),
        (25, "advanced", "adult"),
        (30, "B2", "adult"),
        (45, "средний", "adult"),
        (20, "", "adult")
    ]
    
    for age, self_assessment, education_level in test_cases:
        cefr_level = LanguageIntegration.detect_cefr_level(age, self_assessment, education_level)
        print(f"  Возраст {age}, оценка '{self_assessment}', образование '{education_level}': {cefr_level}")
    
    # Тест 3: Получение конфигурации уровня
    print("\n⚙️ Конфигурация уровней CEFR:")
    for level in ['A1', 'B1', 'C1']:
        config = LanguageIntegration.get_cefr_level_config(level)
        print(f"  {level}: {config.get('description')}, {int(config.get('bilingual_ratio', 0) * 100)}% иностранного")
    
    # Тест 4: Генерация промптов CEFR
    print("\n📝 Генерация промптов CEFR:")
    for level in ['A1', 'B2', 'C1']:
        prompt = LanguageIntegration.create_bilingual_lesson_prompt_cefr(
            topic="Знакомство",
            target_language="english",
            cefr_level=level
        )
        print(f"  Уровень {level}: промпт длиной {len(prompt)} символов")
        # Показать начало промпта
        print(f"    Начало: {prompt[:100]}...")
    
    # Тест 5: Режимы обучения для взрослых
    print("\n🎓 Режимы обучения для взрослых:")
    modes = LanguageIntegration.get_adult_study_modes()
    for mode in modes:
        print(f"  {mode['name']}: {mode['description']}")
    
    # Тест 6: Все уровни CEFR
    print("\n📊 Все уровни CEFR:")
    levels = LanguageIntegration.get_available_cefr_levels()
    for level in levels:
        print(f"  {level['id']}: {level['name']}, {level['russian_percent']}% русский, {level['foreign_percent']}% английский")
    
    print("\n✅ Тестирование завершено!")