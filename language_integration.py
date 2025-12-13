"""
🔥 language_integration.py
Интеграция языковых функций в систему AI-учителя
Минимальные изменения, максимальная совместимость
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
    
    # 🔥 КОНФИГУРАЦИЯ ЯЗЫКОВЫХ ПРЕДМЕТОВ
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
    
    # 🔥 ТИПЫ ЯЗЫКОВЫХ УПРАЖНЕНИЙ ПО УРОВНЯМ
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
            
            Создай упражнение на грамматику.
            
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

# 🔥 ТЕСТИРОВАНИЕ
if __name__ == "__main__":
    print("🧪 Тестирование LanguageIntegration...")
    
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
    
    # Тест 2: Анализ уровня ученика
    print("\n👤 Анализ уровня ученика:")
    for age in [8, 12, 16, 20]:
        level_info = LanguageIntegration.analyze_student_level(age)
        print(f"  Возраст {age}: {level_info['description']}")
    
    # Тест 3: Разделение текста по языкам
    print("\n🔤 Разделение текста по языкам:")
    test_text = "Hello, привет! My name is Иван. Я учу English."
    fragments = LanguageIntegration.split_text_by_language(test_text)
    for fragment, lang in fragments:
        print(f"  '{fragment[:20]}...' → {lang}")
    
    print("\n✅ Тестирование завершено!")
