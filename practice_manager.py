"""
🔥 practice_manager.py
Управление практическими занятиями для AI-учителя
Поддержка языковых, технических и гуманитарных предметов
Минимальные изменения для максимальной совместимости
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from difflib import SequenceMatcher
import time
import threading
import queue

# 🔥 ИМПОРТ ДЛЯ ТЕХНИЧЕСКИХ ПРЕДМЕТОВ
try:
    from technical_subjects import (
        is_technical_subject,
        is_natural_science,
        get_subject_type,
        contains_formulas
    )
    TECHNICAL_SUPPORT_ENABLED = True
except ImportError:
    TECHNICAL_SUPPORT_ENABLED = False
    print("⚠️ technical_subjects.py не найден, техническая поддержка отключена")

# 🔥 ИМПОРТ ПРОМПТОВ
try:
    from prompts.technical_prompts import (
        get_lesson_prompt,
        get_practice_prompt
    )
    TECHNICAL_PROMPTS_ENABLED = True
except ImportError:
    TECHNICAL_PROMPTS_ENABLED = False
    print("⚠️ technical_prompts.py не найден, промпты отключены")

def debug_log(message: str):
    """Логирование для отладки"""
    print(f"🔥 [PRACTICE] {message}")

class PracticeManager:
    def __init__(self, llm_integration):
        """Инициализация менеджера практики"""
        self.llm = llm_integration
        self.practice_dir = Path("materials/practice")
        self.current_lesson_context = ""
        self.current_lesson_summary = ""
        self.current_subject = ""
        self.current_subject_type = "general"  # 🔥 ТИП ПРЕДМЕТА
        
        # 🔥 ДАННЫЕ УЧЕНИКА ДЛЯ АДАПТАЦИИ
        self.student_data = {}
        self.is_language_subject = False
        self.is_technical_subject = False
        self.is_natural_science_subject = False
        self.target_language = None
        self.language_level = 'beginner'
        self.bilingual_ratio = 0.3
        
        # 🔥 МЕНЕДЖЕРЫ ДЛЯ РАЗНЫХ ТИПОВ ПРАКТИКИ
        self.current_practice_type = 'general'  # 'general', 'language', 'technical', 'natural_science'
        
        # 🔥 ТИПЫ ЯЗЫКОВЫХ УПРАЖНЕНИЙ ПО УРОВНЯМ
        self.language_exercise_types = {
            'beginner': ['vocabulary', 'simple_translation', 'fill_blank', 'matching', 'multiple_choice'],
            'intermediate': ['grammar', 'dialogue', 'sentence_building', 'reading', 'translation'],
            'advanced': ['composition', 'error_correction', 'debate', 'essay', 'summary']
        }
        
        # 🔥 ТИПЫ ТЕХНИЧЕСКИХ УПРАЖНЕНИЙ
        self.technical_exercise_types = {
            'математика': ['calculation', 'proof', 'graph', 'problem_solving', 'formula'],
            'физика': ['calculation', 'experiment', 'diagram', 'theory', 'application'],
            'химия': ['reaction', 'calculation', 'formula', 'experiment', 'analysis'],
            'биология': ['diagram', 'classification', 'process', 'experiment', 'analysis'],
            'информатика': ['algorithm', 'code', 'diagram', 'problem_solving', 'theory']
        }
        
        # 🔥 ОЧЕРЕДИ ДЛЯ АСИНХРОННОЙ ГЕНЕРАЦИИ
        self.question_queue = queue.Queue()  # Очередь готовых вопросов
        self.generated_questions = []        # История всех вопросов
        self.current_question_index = 0
        self.max_questions = 5  # ЖЕСТКИЙ ЛИМИТ 5 ВОПРОСОВ
        
        # 🔥 ФЛАГИ УПРАВЛЕНИЯ АСИНХРОННОЙ ГЕНЕРАЦИЕЙ
        self.generation_thread = None
        self.stop_generation = False
        self.generation_active = False
        
        # 🔥 СОЗДАЕМ ДИРЕКТОРИЮ ПРАКТИКИ
        self.practice_dir.mkdir(parents=True, exist_ok=True)
        
        debug_log("✅ PracticeManager инициализирован")

    def set_student_data(self, student_data: Dict):
        """Устанавливает данные ученика"""
        self.student_data = student_data or {}
        debug_log(f"👤 Установлены данные ученика: {student_data.get('name', 'неизвестно')}")

    def _is_language_subject(self, subject: str) -> bool:
        """Определяет, является ли предмет языковым"""
        language_subjects = [
            'английский язык', 'english', 'английский',
            'французский язык', 'french', 'французский',
            'немецкий язык', 'german', 'немецкий',
            'испанский язык', 'spanish', 'испанский',
            'китайский язык', 'chinese', 'китайский',
            'итальянский язык', 'italian', 'итальянский'
        ]
        
        if not subject:
            return False
            
        subject_lower = subject.lower()
        for lang_subj in language_subjects:
            if lang_subj in subject_lower:
                return True
        
        return False

    def _extract_target_language(self, subject: str) -> str:
        """Извлекает целевой язык из названия предмета"""
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
            return 'english'  # По умолчанию

    def _determine_subject_type(self, subject: str) -> str:
        """Определяет тип предмета"""
        if not subject:
            return "general"
        
        # 🔥 ПРОВЕРЯЕМ ЯЗЫКОВОЙ ПРЕДМЕТ
        if self._is_language_subject(subject):
            return "language"
        
        # 🔥 ПРОВЕРЯЕМ ТЕХНИЧЕСКИЙ ПРЕДМЕТ
        if TECHNICAL_SUPPORT_ENABLED:
            if is_technical_subject(subject):
                return "technical"
            elif is_natural_science(subject):
                return "natural_science"
        
        return "humanitarian"

    def _analyze_student_language_level(self, age: int = 12, self_assessment: str = '') -> Dict:
        """Анализирует и определяет уровень языка ученика"""
        # 🔥 ОСНОВНОЙ АЛГОРИТМ ПО ВОЗРАСТУ
        if age <= 10:
            suggested_level = 'beginner'
            bilingual_ratio = 0.3
        elif age <= 14:
            suggested_level = 'intermediate'
            bilingual_ratio = 0.5
        else:
            suggested_level = 'advanced'
            bilingual_ratio = 0.7
        
        # 🔥 УЧИТЫВАЕМ САМООЦЕНКУ УЧЕНИКА
        if self_assessment and self_assessment in ['beginner', 'intermediate', 'advanced']:
            suggested_level = self_assessment
            # Корректируем ratio под уровень
            if suggested_level == 'beginner':
                bilingual_ratio = 0.3
            elif suggested_level == 'intermediate':
                bilingual_ratio = 0.5
            else:
                bilingual_ratio = 0.7
            debug_log(f"🎯 Использована самооценка ученика: {self_assessment}")
        
        return {
            'level': suggested_level,
            'bilingual_ratio': bilingual_ratio,
            'max_questions': 5,
            'exercise_types': self.language_exercise_types.get(suggested_level, []),
            'description': f'Уровень {suggested_level} ({int(bilingual_ratio * 100)}% иностранного языка)'
        }

    def initialize_practice_generation(self, lesson_context: str, subject: str):
        """Инициализирует практику и ЗАРАНЕЕ начинает генерацию вопросов"""
        self.current_lesson_context = lesson_context
        self.current_subject = subject
        self.current_subject_type = self._determine_subject_type(subject)
        
        # 🔥 ОПРЕДЕЛЯЕМ ТИП ПРАКТИКИ
        self.is_language_subject = (self.current_subject_type == "language")
        self.is_technical_subject = (self.current_subject_type == "technical")
        self.is_natural_science_subject = (self.current_subject_type == "natural_science")
        
        debug_log(f"🎯 Инициализация практики. Тип предмета: {self.current_subject_type}")
        debug_log(f"   Языковой: {self.is_language_subject}")
        debug_log(f"   Технический: {self.is_technical_subject}")
        debug_log(f"   Естественная наука: {self.is_natural_science_subject}")
        
        if self.is_language_subject:
            # 🔥 ЯЗЫКОВАЯ ПРАКТИКА
            self.target_language = self._extract_target_language(subject)
            self.current_practice_type = 'language'
            
            # Определяем уровень ученика
            age = int(self.student_data.get('age', 12)) if self.student_data else 12
            self_assessment = self.student_data.get('language_level', '')
            level_info = self._analyze_student_language_level(age, self_assessment)
            
            self.language_level = level_info['level']
            self.bilingual_ratio = level_info['bilingual_ratio']
            
            debug_log(f"🎯 Языковая практика:")
            debug_log(f"   Язык: {self.target_language}")
            debug_log(f"   Уровень: {self.language_level}")
            debug_log(f"   Соотношение: {int(self.bilingual_ratio * 100)}% иностранного")
            
            self.current_lesson_summary = self._generate_language_lesson_summary(lesson_context)
        elif self.is_technical_subject or self.is_natural_science_subject:
            # 🔥 ТЕХНИЧЕСКАЯ ИЛИ ЕСТЕСТВЕННОНАУЧНАЯ ПРАКТИКА
            self.current_practice_type = 'technical'
            debug_log(f"🎯 Техническая/научная практика для предмета: {subject}")
            
            self.current_lesson_summary = self._generate_technical_lesson_summary(lesson_context)
        else:
            # 🔥 ОБЩАЯ ПРАКТИКА ДЛЯ ГУМАНИТАРНЫХ ПРЕДМЕТОВ
            self.current_practice_type = 'general'
            debug_log(f"🎯 Общая практика для предмета: {subject}")
            
            self.current_lesson_summary = self._generate_lesson_summary(lesson_context)
        
        # 🔥 СБРАСЫВАЕМ СОСТОЯНИЕ
        self.generated_questions = []
        self.current_question_index = 0
        
        # Очищаем очередь
        while not self.question_queue.empty():
            try:
                self.question_queue.get_nowait()
            except queue.Empty:
                break
        
        debug_log(f"📝 Максимальное количество вопросов: {self.max_questions}")
        
        # 🔥 ЛОГИРУЕМ ДАННЫЕ УЧЕНИКА ДЛЯ ОТЛАДКИ
        if self.student_data:
            age = self.student_data.get('age', 'неизвестен')
            level = self.student_data.get('level', 'неизвестен')
            debug_log(f"👤 Данные ученика: возраст {age} лет, {level} класс")
        
        # 🔥 ЗАПУСКАЕМ АСИНХРОННУЮ ГЕНЕРАЦИЮ ВОПРОСОВ СРАЗУ
        self._start_async_generation()
        
        # 🔥 ГЕНЕРИРУЕМ ПЕРВЫЙ ВОПРОС СИНХРОННО ДЛЯ НЕМЕДЛЕННОГО СТАРТА
        first_question = self._generate_first_question()
        if first_question:
            self.question_queue.put(first_question)
            debug_log(f"✅ Первый вопрос готов: {first_question[:80]}...")
        else:
            debug_log("⚠️ Не удалось сгенерировать первый вопрос")

    def _generate_first_question(self) -> Optional[str]:
        """Генерирует первый вопрос (синхронно)"""
        try:
            if self.is_language_subject:
                return self.generate_single_language_question()
            elif self.is_technical_subject or self.is_natural_science_subject:
                return self.generate_single_technical_question()
            else:
                return self.generate_single_question()
        except Exception as e:
            debug_log(f"❌ Ошибка генерации первого вопроса: {e}")
            return self._get_fallback_question()

    def _generate_language_lesson_summary(self, lesson_context: str) -> str:
        """Генерирует краткое содержание урока для языковой практики"""
        try:
            prompt = f"""
            Создай КРАТКОЕ содержание этого языкового урока для практических упражнений.
            
            ТЕКСТ УРОКА: {lesson_context[:1500]}
            ИЗУЧАЕМЫЙ ЯЗЫК: {self.target_language}
            УРОВЕНЬ УЧЕНИКА: {self.language_level}
            
            Выдели:
            1. Ключевые слова и фразы на {self.target_language}
            2. Основные грамматические правила
            3. Примеры диалогов или предложений
            4. Темы для практики
            
            Верни только краткое содержание (максимум 300 слов).
            Используй русский язык для объяснений.
            """
            
            summary = self.llm.query(
                question=prompt,
                context="",
                subject=f"{self.target_language} language"
            )
            return summary if summary and len(summary) > 50 else lesson_context[:500] + "..."
        except Exception as e:
            debug_log(f"❌ Ошибка генерации языкового содержания: {e}")
            return lesson_context[:500] + "..."

    def _generate_technical_lesson_summary(self, lesson_context: str) -> str:
        """Генерирует краткое содержание урока для технической практики"""
        try:
            prompt = f"""
            Создай КРАТКОЕ содержание этого урока для практических заданий.
            
            ТЕКСТ: {lesson_context[:1500]}
            ПРЕДМЕТ: {self.current_subject}
            ТИП ПРЕДМЕТА: {self.current_subject_type}
            
            🔥 ОСОБЫЕ ТРЕБОВАНИЯ ДЛЯ ТЕХНИЧЕСКОГО ПРЕДМЕТА:
            1. Сохрани все формулы и математические обозначения
            2. Выдели ключевые теоремы, законы или принципы
            3. Перечисли основные понятия с их определениями
            4. Включи примеры расчетов или решений
            5. Упомяни практические применения
            
            Верни только краткое содержание (максимум 300 слов).
            Сохраняй все технические обозначения!
            """
            
            summary = self.llm.query(
                question=prompt,
                context="",
                subject=self.current_subject
            )
            return summary if summary and len(summary) > 50 else lesson_context[:500] + "..."
        except Exception as e:
            debug_log(f"❌ Ошибка генерации технического содержания: {e}")
            return lesson_context[:500] + "..."

    def _generate_lesson_summary(self, lesson_context: str) -> str:
        """Генерирует краткое содержание урока для общей практики"""
        try:
            prompt = f"""
            Создай КРАТКОЕ содержание этого урока для практических вопросов.
            
            ТЕКСТ: {lesson_context[:1500]}
            ПРЕДМЕТ: {self.current_subject}
            
            Выдели только ключевые понятия и основные идеи.
            Верни только краткое содержание (максимум 300 слов).
            """
            
            summary = self.llm.query(
                question=prompt,
                context="",
                subject=self.current_subject
            )
            return summary if summary and len(summary) > 50 else lesson_context[:500] + "..."
        except Exception as e:
            debug_log(f"❌ Ошибка генерации краткого содержания: {e}")
            return lesson_context[:500] + "..."

    def _start_async_generation(self):
        """Запускает фоновую генерацию вопросов"""
        if self.generation_active:
            debug_log("⚠️ Фоновая генерация уже активна")
            return
            
        self.stop_generation = False
        self.generation_active = True
        
        def generate_questions_worker():
            debug_log("🔄 Фоновая генерация вопросов запущена...")
            
            generated_count = 0
            max_attempts = self.max_questions * 2  # Максимум попыток
            
            while (not self.stop_generation and 
                   self.generation_active and 
                   generated_count < self.max_questions - 1 and  # -1 потому что первый уже сгенерирован
                   generated_count < max_attempts):
                
                try:
                    # Проверяем лимит
                    if len(self.generated_questions) >= self.max_questions:
                        debug_log("🏁 Достигнут лимит вопросов в фоновой генерации")
                        break
                    
                    # Генерируем следующий вопрос в зависимости от типа
                    if self.is_language_subject:
                        next_question = self.generate_single_language_question()
                    elif self.is_technical_subject or self.is_natural_science_subject:
                        next_question = self.generate_single_technical_question()
                    else:
                        next_question = self.generate_single_question()
                    
                    if next_question:
                        self.question_queue.put(next_question)
                        generated_count += 1
                        debug_log(f"✅ Фоново сгенерирован вопрос {len(self.generated_questions)}/{self.max_questions}")
                    
                    # 🔥 ОПТИМИЗИРОВАННАЯ ПАУЗА
                    time.sleep(0.5)  # Уменьшенная пауза для скорости
                    
                except Exception as e:
                    debug_log(f"❌ Ошибка в фоновой генерации: {e}")
                    time.sleep(1)  # Пауза при ошибке
            
            debug_log(f"🏁 Фоновая генерация завершена. Сгенерировано: {generated_count} вопросов")
            self.generation_active = False
        
        # Запускаем в отдельном потоке
        self.generation_thread = threading.Thread(target=generate_questions_worker, daemon=True)
        self.generation_thread.start()
        debug_log("✅ Поток фоновой генерации запущен")

    def generate_single_question(self) -> Optional[str]:
        """Генерирует один ТЕСТОВЫЙ вопрос для гуманитарных предметов"""
        try:
            # 🔥 ПРОВЕРКА ЛИМИТА
            if len(self.generated_questions) >= self.max_questions:
                debug_log(f"🏁 Достигнут лимит вопросов: {len(self.generated_questions)}/{self.max_questions}")
                return None
            
            # 🔥 ПОЛУЧАЕМ ДАННЫЕ УЧЕНИКА ДЛЯ ПЕРСОНАЛИЗАЦИИ
            age = int(self.student_data.get('age', 12)) if self.student_data else 12
            level = self.student_data.get('level', '5')
            name = self.student_data.get('name', 'ученик')
            
            # 🔥 ОБНОВЛЕННЫЙ ПРОМТ ДЛЯ ГУМАНИТАРНЫХ ПРЕДМЕТОВ
            prompt = f"""
            СОЗДАЙ ТЕСТОВЫЙ ВОПРОС ДЛЯ ПРОВЕРКИ ПОНИМАНИЯ УРОКА.
            
            МАТЕРИАЛ УРОКА:
            {self.current_lesson_summary}
            
            ПРЕДМЕТ: {self.current_subject}
            
            ПАРАМЕТРЫ УЧЕНИКА:
            - Возраст: {age} лет
            - Уровень: {level} класс
            - Имя: {name}
            
            🔥 КРИТИЧЕСКИ ВАЖНЫЕ ТРЕБОВАНИЯ:
            1. ВОПРОС ДОЛЖЕН БЫТЬ ИСКЛЮЧИТЕЛЬНО ПО СОДЕРЖАНИЮ ЭТОГО УРОКА
            2. НЕ ВЫХОДИ ЗА ПРЕДЕЛЫ МАТЕРИАЛА УРОКА
            3. ВОПРОС НЕ ДОЛЖЕН ПОВТОРЯТЬСЯ С ПРЕДЫДУЩИМИ
            4. УЧИТЫВАЙ ВОЗРАСТ УЧЕНИКА ({age} лет)
            5. ВОПРОС ДОЛЖЕН БЫТЬ ПОНЯТНЫМ ДЛЯ ЭТОГО ВОЗРАСТА
            6. СОЗДАЙ 4 ВАРИАНТА ОТВЕТА: A, B, C, D
            7. НЕ ВКЛЮЧАЙ ПРАВИЛЬНЫЙ ОТВЕТ В ТЕКСТ ВОПРОСА
            
            🔥 ФОРМАТ:
            [Текст вопроса?]
            
            Варианты ответов:
            A) [Текст варианта A]
            B) [Текст варианта B]
            C) [Текст варианта C]
            D) [Текст варианта D]
            
            🔥 УЖЕ ЗАДАННЫЕ ВОПРОСЫ (НЕ ПОВТОРЯЙ!):
            {self._get_previous_questions_text()}
            
            🔥 ПРИМЕР ДЛЯ ВОЗРАСТА {age} ЛЕТ:
            Вопрос должен быть соответствующим сложности для {age} лет.
            
            Верни только вопрос в указанном формате.
            """
            
            llm_response = self.llm.query(
                question=prompt,
                context="",
                subject=self.current_subject
            )
            
            if llm_response and len(llm_response.strip()) > 50:
                question_text = llm_response.strip()
                
                # Проверяем формат тестового вопроса
                is_test_question = any(marker in question_text for marker in ["A)", "B)", "C)", "D)"])
                
                if self._is_question_unique(question_text):
                    question_data = {
                        "question": question_text,
                        "generated_at": time.time(),
                        "type": "test_question" if is_test_question else "general_question",
                        "subject": self.current_subject,
                        "subject_type": self.current_subject_type,
                        "student_age": age,
                        "student_level": level
                    }
                    self.generated_questions.append(question_data)
                    return question_text
            
            # 🔥 FALLBACK: Если не получился тестовый вопрос
            fallback = self._get_fallback_question_for_age(age, ensure_unique=True)
            debug_log(f"🔄 Использован fallback вопрос для возраста {age}: {fallback[:80]}...")
            return fallback
            
        except Exception as e:
            debug_log(f"❌ Ошибка генерации тестового вопроса: {e}")
            age = int(self.student_data.get('age', 12)) if self.student_data else 12
            fallback = self._get_fallback_question_for_age(age, ensure_unique=True)
            debug_log(f"🔄 Использован fallback после ошибки: {fallback[:80]}...")
            return fallback

    def generate_single_language_question(self) -> Optional[str]:
        """Генерирует один вопрос для языковой практики"""
        try:
            # 🔥 ПРОВЕРКА ЛИМИТА
            if len(self.generated_questions) >= self.max_questions:
                debug_log(f"🏁 Достигнут лимит языковых вопросов: {len(self.generated_questions)}/{self.max_questions}")
                return None
            
            # 🔥 ВЫБИРАЕМ ТИП УПРАЖНЕНИЯ ПО УРОВНЮ
            import random
            available_types = self.language_exercise_types.get(self.language_level, ['vocabulary', 'translation'])
            exercise_type = random.choice(available_types)
            
            # 🔥 ГЕНЕРИРУЕМ ПРОМТ ДЛЯ ЯЗЫКОВОГО УПРАЖНЕНИЯ
            prompt = self._create_language_exercise_prompt(exercise_type)
            
            llm_response = self.llm.query(
                question=prompt,
                context="",
                subject=f"{self.target_language} language"
            )
            
            if llm_response and len(llm_response.strip()) > 30:
                question_text = llm_response.strip()
                
                if self._is_question_unique(question_text):
                    question_data = {
                        "question": question_text,
                        "generated_at": time.time(),
                        "type": f"language_{exercise_type}",
                        "subject": self.current_subject,
                        "subject_type": self.current_subject_type,
                        "language": self.target_language,
                        "level": self.language_level,
                        "bilingual_ratio": self.bilingual_ratio,
                        "student_age": self.student_data.get('age', 12) if self.student_data else 12
                    }
                    self.generated_questions.append(question_data)
                    return question_text
            
            # 🔥 FALLBACK ДЛЯ ЯЗЫКОВОГО ВОПРОСА
            fallback = self._get_language_fallback_question(exercise_type, ensure_unique=True)
            debug_log(f"🔄 Использован языковой fallback: {fallback[:80]}...")
            return fallback
            
        except Exception as e:
            debug_log(f"❌ Ошибка генерации языкового вопроса: {e}")
            fallback = self._get_language_fallback_question('vocabulary', ensure_unique=True)
            return fallback

    def generate_single_technical_question(self) -> Optional[str]:
        """Генерирует один вопрос для технической или естественнонаучной практики"""
        try:
            # 🔥 ПРОВЕРКА ЛИМИТА
            if len(self.generated_questions) >= self.max_questions:
                debug_log(f"🏁 Достигнут лимит технических вопросов: {len(self.generated_questions)}/{self.max_questions}")
                return None
            
            # 🔥 ПОЛУЧАЕМ ДАННЫЕ УЧЕНИКА
            age = int(self.student_data.get('age', 12)) if self.student_data else 12
            level = self.student_data.get('level', '5')
            name = self.student_data.get('name', 'ученик')
            
            # 🔥 ВЫБИРАЕМ ТИП ТЕХНИЧЕСКОГО УПРАЖНЕНИЯ
            import random
            subject_lower = self.current_subject.lower()
            
            # Определяем типы упражнений для предмета
            exercise_types = []
            for key, types in self.technical_exercise_types.items():
                if key in subject_lower:
                    exercise_types = types
                    break
            
            if not exercise_types:
                # Общие типы для технических предметов
                exercise_types = ['calculation', 'problem', 'analysis', 'application', 'theory']
            
            exercise_type = random.choice(exercise_types)
            
            # 🔥 СОЗДАЕМ ПРОМТ ДЛЯ ТЕХНИЧЕСКОГО ВОПРОСА
            prompt = self._create_technical_exercise_prompt(exercise_type, age, level)
            
            llm_response = self.llm.query(
                question=prompt,
                context="",
                subject=self.current_subject
            )
            
            if llm_response and len(llm_response.strip()) > 50:
                question_text = llm_response.strip()
                
                # Проверяем наличие формул (должны быть для технических предметов)
                has_formulas = TECHNICAL_SUPPORT_ENABLED and contains_formulas(question_text)
                
                if self._is_question_unique(question_text):
                    question_data = {
                        "question": question_text,
                        "generated_at": time.time(),
                        "type": f"technical_{exercise_type}",
                        "subject": self.current_subject,
                        "subject_type": self.current_subject_type,
                        "has_formulas": has_formulas,
                        "student_age": age,
                        "student_level": level
                    }
                    self.generated_questions.append(question_data)
                    
                    debug_log(f"🔧 Технический вопрос создан. Формулы: {has_formulas}")
                    return question_text
            
            # 🔥 FALLBACK ДЛЯ ТЕХНИЧЕСКОГО ВОПРОСА
            fallback = self._get_technical_fallback_question(exercise_type, ensure_unique=True)
            debug_log(f"🔄 Использован технический fallback: {fallback[:80]}...")
            return fallback
            
        except Exception as e:
            debug_log(f"❌ Ошибка генерации технического вопроса: {e}")
            fallback = self._get_technical_fallback_question('calculation', ensure_unique=True)
            return fallback

    def _create_language_exercise_prompt(self, exercise_type: str) -> str:
        """Создает промт для языкового упражнения"""
        
        # 🔥 НАСТРОЙКИ ДЛЯ РАЗНЫХ УРОВНЕЙ
        level_settings = {
            'beginner': {
                'complexity': 'ОЧЕНЬ ПРОСТОЙ',
                'instructions': 'Инструкция должна быть максимально простой и понятной',
                'examples': 'Простейшие примеры, максимум 3-4 слова',
                'language_mix': f'{int((1 - self.bilingual_ratio) * 100)}% русского, {int(self.bilingual_ratio * 100)}% {self.target_language}',
                'hints': 'Добавь подсказки и примеры'
            },
            'intermediate': {
                'complexity': 'СРЕДНЕЙ СЛОЖНОСТИ',
                'instructions': 'Инструкция должна быть четкой и понятной',
                'examples': 'Развернутые примеры, естественные фразы',
                'language_mix': f'{int((1 - self.bilingual_ratio) * 100)}% русского, {int(self.bilingual_ratio * 100)}% {self.target_language}',
                'hints': 'Минимальные подсказки'
            },
            'advanced': {
                'complexity': 'СЛОЖНЫЙ',
                'instructions': 'Инструкция может быть на изучаемом языке',
                'examples': 'Сложные конструкции, идиомы, естественный язык',
                'language_mix': f'{int((1 - self.bilingual_ratio) * 100)}% русского, {int(self.bilingual_ratio * 100)}% {self.target_language}',
                'hints': 'Без подсказок'
            }
        }
        
        settings = level_settings.get(self.language_level, level_settings['beginner'])
        
        exercise_prompts = {
            'vocabulary': f"""
            СОЗДАЙ УПРАЖНЕНИЕ НА СЛОВАРНЫЙ ЗАПАС
            
            ИЗУЧАЕМЫЙ ЯЗЫК: {self.target_language}
            УРОВЕНЬ: {self.language_level} ({settings['complexity']})
            ТЕМА УРОКА: {self.current_lesson_summary[:300]}
            СООТНОШЕНИЕ ЯЗЫКОВ: {settings['language_mix']}
            
            Создай упражнение "Сопоставление слов с переводами".
            
            🔥 ТРЕБОВАНИЯ:
            1. {settings['instructions']}
            2. {settings['examples']}
            3. {settings['hints']}
            4. Включи транскрипцию для новых слов
            5. Слова должны быть полезными и практичными
            
            🔥 ФОРМАТ:
            [Инструкция на русском]
            
            Сопоставьте слова на {self.target_language} с их переводом:
            
            1. Word1 [транскрипция]
            2. Word2 [транскрипция]
            3. Word3 [транскрипция]
            4. Word4 [транскрипция]
            5. Word5 [транскрипция]
            
            A. Перевод1
            B. Перевод2
            C. Перевод3
            D. Перевод4
            E. Перевод5
            
            🔥 ПРИМЕР ДЛЯ УРОВНЯ {self.language_level}:
            {settings['examples']}
            
            Верни только упражнение.
            """,
            
            'translation': f"""
            СОЗДАЙ УПРАЖНЕНИЕ НА ПЕРЕВОД
            
            ИЗУЧАЕМЫЙ ЯЗЫК: {self.target_language}
            УРОВЕНЬ: {self.language_level} ({settings['complexity']})
            ТЕМА: {self.current_lesson_summary[:300]}
            
            Создай упражнение "Переведите предложения на {self.target_language}".
            
            🔥 ТРЕБОВАНИЯ:
            1. {settings['instructions']}
            2. {settings['examples']}
            3. {settings['hints']}
            4. Предложения должны соответствовать теме урока
            5. Учитывай уровень {self.language_level}
            
            🔥 ФОРМАТ:
            [Инструкция на русском]
            
            Переведите на {self.target_language}:
            
            1. [Предложение на русском]
            2. [Предложение на русском]
            3. [Предложение на русском]
            
            🔥 ПРИМЕР ДЛЯ УРОВНЯ {self.language_level}:
            {settings['examples']}
            
            Верни только упражнение.
            """
        }
        
        return exercise_prompts.get(exercise_type, exercise_prompts['vocabulary'])

    def _create_technical_exercise_prompt(self, exercise_type: str, age: int, level: str) -> str:
        """Создает промт для технического упражнения"""
        
        # 🔥 ИСПОЛЬЗУЕМ ТЕХНИЧЕСКИЕ ПРОМПТЫ ИЛИ СОЗДАЕМ ОБЩИЙ
        if TECHNICAL_PROMPTS_ENABLED:
            # Получаем промпт для практики из technical_prompts
            try:
                prompt = get_practice_prompt(
                    subject=self.current_subject,
                    topic=self.current_lesson_summary[:200],
                    level=level
                )
                
                # Адаптируем для конкретного типа упражнения
                prompt += f"\n\nСоздай задание типа '{exercise_type}'."
                prompt += f"\nУченик: {age} лет, {level} класс."
                prompt += "\nВключи формулы, вычисления, и практические применения."
                
                return prompt
            except Exception as e:
                debug_log(f"⚠️ Ошибка получения технического промпта: {e}")
        
        # 🔥 ОБЩИЙ ПРОМПТ ДЛЯ ТЕХНИЧЕСКИХ ПРЕДМЕТОВ
        subject_lower = self.current_subject.lower()
        
        if any(math_word in subject_lower for math_word in ['математика', 'алгебра', 'геометрия']):
            base_prompt = f"""
            СОЗДАЙ ЗАДАЧУ ПО МАТЕМАТИКЕ
            
            ТЕМА: {self.current_lesson_summary[:300]}
            ТИП УПРАЖНЕНИЯ: {exercise_type}
            УРОВЕНЬ: {level} класс
            ВОЗРАСТ: {age} лет
            
            🔥 ТРЕБОВАНИЯ:
            1. Включи конкретную математическую задачу
            2. Используй формулы и вычисления
            3. Добавь пошаговое решение
            4. Объясни математические понятия
            5. Свяжи с практическим применением
            
            🔥 ПРИМЕРЫ ТИПОВ:
            - calculation: "Вычислите: (15 + 7) × 3 - 12"
            - proof: "Докажите, что сумма углов треугольника равна 180°"
            - graph: "Постройте график функции y = 2x + 3"
            - problem_solving: "Решите задачу: На складе было 150 кг муки..."
            - formula: "Выведите формулу площади круга S = πr²"
            
            Создай задачу соответствующую типу '{exercise_type}'.
            """
            
        elif 'физика' in subject_lower:
            base_prompt = f"""
            СОЗДАЙ ЗАДАЧУ ПО ФИЗИКЕ
            
            ТЕМА: {self.current_lesson_summary[:300]}
            ТИП УПРАЖНЕНИЯ: {exercise_type}
            УРОВЕНЬ: {level} класс
            ВОЗРАСТ: {age} лет
            
            🔥 ТРЕБОВАНИЯ:
            1. Используй физические законы и формулы
            2. Включи единицы измерения
            3. Добавь расчеты с объяснениями
            4. Покажи практическое применение
            5. Объясни физический смысл
            
            🔥 ПРИМЕРЫ ТИПОВ:
            - calculation: "Рассчитайте скорость тела через 5 секунд..."
            - experiment: "Опишите эксперимент для проверки закона..."
            - diagram: "Нарисуйте схему сил, действующих на тело"
            - theory: "Объясните принцип работы..."
            - application: "Где в жизни применяется этот закон?"
            
            Создай задачу соответствующую типу '{exercise_type}'.
            """
            
        elif 'химия' in subject_lower:
            base_prompt = f"""
            СОЗДАЙ ЗАДАЧУ ПО ХИМИИ
            
            ТЕМА: {self.current_lesson_summary[:300]}
            ТИП УПРАЖНЕНИЯ: {exercise_type}
            УРОВЕНЬ: {level} класс
            ВОЗРАСТ: {age} лет
            
            🔥 ТРЕБОВАНИЯ:
            1. Используй химические формулы и уравнения
            2. Включи расчеты с молярными массами
            3. Добавь условия реакций
            4. Объясни химические процессы
            5. Упомяни технику безопасности
            
            🔥 ПРИМЕРЫ ТИПОВ:
            - reaction: "Напишите уравнение реакции горения метана"
            - calculation: "Рассчитайте массу продукта реакции..."
            - formula: "Определите молекулярную формулу вещества..."
            - experiment: "Опишите эксперимент по получению кислорода"
            - analysis: "Проанализируйте свойства вещества..."
            
            Создай задачу соответствующую типу '{exercise_type}'.
            """
            
        elif 'биология' in subject_lower:
            base_prompt = f"""
            СОЗДАЙ ЗАДАНИЕ ПО БИОЛОГИИ
            
            ТЕМА: {self.current_lesson_summary[:300]}
            ТИП УПРАЖНЕНИЯ: {exercise_type}
            УРОВЕНЬ: {level} класс
            ВОЗРАСТ: {age} лет
            
            🔥 ТРЕБОВАНИЯ:
            1. Используй научные термины
            2. Включи схемы или описания процессов
            3. Объясни биологические механизмы
            4. Покажи взаимосвязи в природе
            5. Свяжи с экологией и здоровьем
            
            🔥 ПРИМЕРЫ ТИПОВ:
            - diagram: "Нарисуйте схему строения клетки"
            - classification: "Классифицируйте растения по признакам"
            - process: "Опишите процесс фотосинтеза"
            - experiment: "Опишите опыт по изучение роста растений"
            - analysis: "Проанализируйте пищевую цепь в лесу"
            
            Создай задание соответствующее типу '{exercise_type}'.
            """
            
        else:
            # Общий промпт для остальных технических предметов
            base_prompt = f"""
            СОЗДАЙ ТЕХНИЧЕСКОЕ ЗАДАНИЕ
            
            ПРЕДМЕТ: {self.current_subject}
            ТЕМА: {self.current_lesson_summary[:300]}
            ТИП УПРАЖНЕНИЯ: {exercise_type}
            УРОВЕНЬ: {level} класс
            ВОЗРАСТ: {age} лет
            
            🔥 ОБЩИЕ ТРЕБОВАНИЯ ДЛЯ ТЕХНИЧЕСКИХ ПРЕДМЕТОВ:
            1. Используй точную терминологию
            2. Включи расчеты или формулы если применимо
            3. Добавь практические примеры
            4. Объясни принципы работы
            5. Покажи практическое применение
            
            Создай задание соответствующее типу '{exercise_type}'.
            """
        
        # Добавляем контекст предыдущих вопросов
        base_prompt += f"\n\n🔥 УЖЕ ЗАДАННЫЕ ВОПРОСЫ (НЕ ПОВТОРЯЙ!):"
        base_prompt += f"\n{self._get_previous_questions_text()}"
        
        return base_prompt

    def get_next_question(self, timeout: float = 2.0) -> Optional[str]:
        """Получает следующий вопрос из очереди (с ожиданием если нужно)"""
        try:
            # 🔥 ПРОВЕРКА ЛИМИТА
            if len(self.generated_questions) >= self.max_questions:
                debug_log(f"🏁 Достигнут лимит вопросов: {len(self.generated_questions)}/{self.max_questions}")
                return None
            
            # 🔥 ПЫТАЕМСЯ ВЗЯТЬ ВОПРОС ИЗ ОЧЕРЕДИ БЕЗ ОЖИДАНИЯ
            try:
                question = self.question_queue.get_nowait()
                debug_log(f"✅ Вопрос взят из очереди (в очереди еще: {self.question_queue.qsize()})")
                return question
            except queue.Empty:
                debug_log("⚠️ Очередь вопросов пуста, пробуем сгенерировать...")
            
            # 🔥 ЕСЛИ ОЧЕРЕДЬ ПУСТА, ГЕНЕРИРУЕМ СИНХРОННО
            if self.is_language_subject:
                question = self.generate_single_language_question()
            elif self.is_technical_subject or self.is_natural_science_subject:
                question = self.generate_single_technical_question()
            else:
                question = self.generate_single_question()
            
            if question:
                debug_log(f"✅ Синхронно сгенерирован вопрос: {question[:80]}...")
                return question
            
            # 🔥 ЕСЛИ СИНХРОННАЯ ГЕНЕРАЦИЯ НЕ УДАЛАСЬ, ИСПОЛЬЗУЕМ FALLBACK
            if self.is_language_subject:
                fallback = self._get_language_fallback_question('vocabulary', ensure_unique=True)
            elif self.is_technical_subject or self.is_natural_science_subject:
                fallback = self._get_technical_fallback_question('calculation', ensure_unique=True)
            else:
                age = int(self.student_data.get('age', 12)) if self.student_data else 12
                fallback = self._get_fallback_question_for_age(age, ensure_unique=True)
            
            debug_log(f"🔄 Использован fallback вопрос: {fallback[:80]}...")
            return fallback
            
        except Exception as e:
            debug_log(f"❌ Ошибка получения следующего вопроса: {e}")
            
            # 🔥 АВАРИЙНЫЙ FALLBACK
            if self.is_language_subject:
                fallback = "Составьте предложение на изучаемом языке."
            elif self.is_technical_subject or self.is_natural_science_subject:
                fallback = "Решите простую задачу по теме урока."
            else:
                fallback = "Расскажите об основном понятии из урока."
            
            debug_log(f"🔄 Использован аварийный fallback: {fallback}")
            return fallback

    def evaluate_and_continue(self, student_answer: str, current_question: str) -> Tuple[str, Optional[str]]:
        """Оценивает ответ и возвращает feedback + следующий вопрос"""
        try:
            # 1. ОЦЕНИВАЕМ ОТВЕТ (быстрая операция)
            if self.is_language_subject:
                feedback = self.evaluate_language_answer(student_answer, current_question)
            elif self.is_technical_subject or self.is_natural_science_subject:
                feedback = self.evaluate_technical_answer(student_answer, current_question)
            else:
                feedback = self.evaluate_single_answer(student_answer, current_question)
            
            # 2. ПОЛУЧАЕМ СЛЕДУЮЩИЙ ВОПРОС (параллельно)
            next_question = self.get_next_question()
            
            return feedback, next_question
            
        except Exception as e:
            debug_log(f"❌ Ошибка в evaluate_and_continue: {e}")
            
            # 🔥 АВАРИЙНЫЙ FALLBACK
            feedback = "Спасибо за ответ! Переходим к следующему вопросу."
            
            if self.is_language_subject:
                next_question = self._get_language_fallback_question('vocabulary', ensure_unique=True)
            elif self.is_technical_subject or self.is_natural_science_subject:
                next_question = self._get_technical_fallback_question('calculation', ensure_unique=True)
            else:
                age = int(self.student_data.get('age', 12)) if self.student_data else 12
                next_question = self._get_fallback_question_for_age(age, ensure_unique=True)
            
            return feedback, next_question

    def evaluate_single_answer(self, student_answer: str, question: str) -> str:
        """Оценивает ответ ученика через LLM"""
        try:
            if not student_answer or len(student_answer.strip()) < 2:
                return "Ответ слишком короткий. Пожалуйста, попробуйте ответить более развернуто."
            
            # 🔥 ПРОВЕРКА НА КОМАНДЫ
            command_words = ['продолжай', 'дальше', 'следующий', 'следующий вопрос', 'далее']
            student_answer_lower = student_answer.lower()
            if any(cmd in student_answer_lower for cmd in command_words):
                return "Это похоже на команду. Пожалуйста, дайте ответ на вопрос."
            
            # 🔥 ПОЛУЧАЕМ ДАННЫЕ УЧЕНИКА
            age = int(self.student_data.get('age', 12)) if self.student_data else 12
            level = self.student_data.get('level', '5')
            name = self.student_data.get('name', 'ученик')
            
            # 🔥 ПРОВЕРЯЕМ ТИП ВОПРОСА
            is_test_question = any(marker in question for marker in ["A)", "B)", "C)", "D)"])
            
            if is_test_question:
                # 🔥 ТЕСТОВЫЙ ВОПРОС
                correct_answer = self._generate_correct_answer_for_test(question)
                evaluation = self._evaluate_test_answer(student_answer, question, correct_answer, age)
            else:
                # 🔥 ОБЫЧНЫЙ ВОПРОС
                correct_answer = self._generate_correct_answer(question)
                evaluation = self._evaluate_general_answer(student_answer, question, correct_answer, age)
            
            # 🔥 ПРОВЕРКА НА НЕПОДХОДЯЩИЕ ОТВЕТЫ
            if not evaluation or any(phrase in evaluation for phrase in [
                "Хороший вопрос! Давайте разберем эту тему подробнее",
                "Мне нужно немного времени подумать",
                "Спасибо за вопрос! Я подумаю над ответом",
                "Давайте разберем эту тему"
            ]):
                if is_test_question:
                    evaluation = "Спасибо за ответ! Давайте проверим правильность."
                else:
                    evaluation = "Спасибо за ответ! Переходим к следующему вопросу."
            
            return evaluation
            
        except Exception as e:
            debug_log(f"❌ Ошибка оценки ответа: {e}")
            return "Спасибо за ответ! Переходим к следующему вопросу."

    def evaluate_language_answer(self, student_answer: str, question: str) -> str:
        """Оценивает ответ на языковое упражнение"""
        try:
            if not student_answer or len(student_answer.strip()) < 1:
                return "Ответ слишком короткий. Попробуйте ответить на изучаемом языке."
            
            # 🔥 ПРОВЕРКА НА КОМАНДЫ
            command_words = ['continue', 'next', 'дальше', 'следующий']
            student_answer_lower = student_answer.lower()
            if any(cmd in student_answer_lower for cmd in command_words):
                return "Пожалуйста, дайте ответ на упражнение."
            
            prompt = f"""
            Оцени ответ ученика на языковое упражнение.
            
            ЯЗЫК: {self.target_language}
            УРОВЕНЬ: {self.language_level}
            УПРАЖНЕНИЕ: {question}
            ОТВЕТ УЧЕНИКА: {student_answer}
            
            УЧЕНИК:
            - Возраст: {self.student_data.get('age', 12) if self.student_data else 12} лет
            - Уровень языка: {self.language_level}
            
            🔥 ТРЕБОВАНИЯ К ОЦЕНКЕ:
            1. Дай конструктивную обратную связь на русском
            2. Учитывай уровень {self.language_level}
            3. Если есть ошибки - мягко исправь их
            4. Похвали за правильные элементы
            5. Будь добрым и поддерживающим
            6. Не критикуй резко
            7. Максимум 3 предложения
            
            🔥 ПРИМЕР ДЛЯ УРОВНЯ {self.language_level}:
            - Beginner: "Молодец! Ты правильно использовал слово 'hello'. Попробуй добавить 'my name is'."
            - Intermediate: "Хорошая попытка! Обрати внимание на порядок слов в предложении."
            - Advanced: "Интересный ответ! Учти, что в этой ситуации лучше использовать Present Perfect."
            
            Верни только обратную связь.
            """
            
            feedback = self.llm.query(
                question=prompt,
                context="",
                subject=f"{self.target_language} language"
            )
            
            if feedback and len(feedback.strip()) > 20:
                return feedback
            
            return "Спасибо за ответ! Продолжайте практиковать язык."
            
        except Exception as e:
            debug_log(f"❌ Ошибка оценки языкового ответа: {e}")
            return "Спасибо за ответ! Это хорошая практика."

    def evaluate_technical_answer(self, student_answer: str, question: str) -> str:
        """Оценивает ответ на техническое упражнение"""
        try:
            if not student_answer or len(student_answer.strip()) < 1:
                return "Ответ слишком короткий. Пожалуйста, дайте более развернутый ответ."
            
            # 🔥 ПРОВЕРКА НА КОМАНДЫ
            command_words = ['продолжай', 'дальше', 'следующий', 'не знаю', 'не понимаю']
            student_answer_lower = student_answer.lower()
            if any(cmd in student_answer_lower for cmd in command_words):
                return "Пожалуйста, попробуйте решить задание или объясните свой подход."
            
            # 🔥 ПОЛУЧАЕМ ДАННЫЕ УЧЕНИКА
            age = int(self.student_data.get('age', 12)) if self.student_data else 12
            level = self.student_data.get('level', '5')
            
            prompt = f"""
            Оцени ответ ученика на техническое задание.
            
            ПРЕДМЕТ: {self.current_subject}
            ТИП ПРЕДМЕТА: {self.current_subject_type}
            ЗАДАНИЕ: {question}
            ОТВЕТ УЧЕНИКА: {student_answer}
            
            УЧЕНИК:
            - Возраст: {age} лет
            - Уровень: {level} класс
            
            КОНТЕКСТ УРОКА: {self.current_lesson_summary[:500]}
            
            🔥 ТРЕБОВАНИЯ К ОЦЕНКЕ ДЛЯ ТЕХНИЧЕСКОГО ПРЕДМЕТА:
            1. Проверь правильность расчетов и формул
            2. Оцени логику решения
            3. Учитывай возраст ученика ({age} лет)
            4. Если есть ошибки - покажи правильное решение
            5. Похвали за правильные элементы
            6. Будь поддерживающим, но точным
            7. Максимум 4 предложения
            
            🔥 ОСОБОЕ ВНИМАНИЕ:
            - Для математики: проверь вычисления
            - Для физики: проверь формулы и единицы измерения
            - Для химии: проверь уравнения реакций
            - Для биологии: проверь терминологию и схемы
            
            Верни только обратную связь.
            """
            
            feedback = self.llm.query(
                question=prompt,
                context="",
                subject=self.current_subject
            )
            
            if feedback and len(feedback.strip()) > 20:
                return feedback
            
            return "Спасибо за ответ! Проверьте свои вычисления и формулы."
            
        except Exception as e:
            debug_log(f"❌ Ошибка оценки технического ответа: {e}")
            return "Спасибо за ответ! Переходим к следующему заданию."

    def _evaluate_test_answer(self, student_answer: str, question: str, correct_answer: str, age: int) -> str:
        """Оценивает ответ на тестовый вопрос"""
        try:
            prompt = f"""
            Оцени ответ ученика на тестовый вопрос.
            
            ВОПРОС: {question}
            ПРАВИЛЬНЫЙ ОТВЕТ: {correct_answer}
            ОТВЕТ УЧЕНИКА: {student_answer}
            
            УЧЕНИК:
            - Возраст: {age} лет
            
            Контекст урока: {self.current_lesson_summary[:300]}
            
            🔥 ТРЕБОВАНИЯ:
            1. Оцени, насколько ответ правильный
            2. Учитывай возраст {age} лет
            3. Будь добрым и поддерживающим
            4. Если ответ неправильный - мягко объясни ошибку
            5. Если ответ правильный - похвали
            6. Не используй сложные термины
            7. Максимум 2 предложения
            
            🔥 ФОРМАТ ОЦЕНКИ:
            [Оценка ответа] [Объяснение если нужно] [Поддержка]
            
            Примеры:
            - "Правильно! Молодец!"
            - "Почти верно! Правильный ответ: B) Клетка."
            - "Не совсем. Обрати внимание на..."
            
            Верни только оценку.
            """
            
            evaluation = self.llm.query(
                question=prompt,
                context="",
                subject=self.current_subject
            )
            
            if evaluation and len(evaluation.strip()) > 10:
                return evaluation
            
            return f"Спасибо за ответ! Правильный ответ: {correct_answer}"
            
        except Exception as e:
            debug_log(f"❌ Ошибка оценки тестового ответа: {e}")
            return f"Спасибо за ответ! Правильный ответ: {correct_answer}"

    def _evaluate_general_answer(self, student_answer: str, question: str, correct_answer: str, age: int) -> str:
        """Оценивает ответ на общий вопрос"""
        try:
            prompt = f"""
            Оцени ответ ученика на вопрос.
            
            ВОПРОС: {question}
            ПРАВИЛЬНЫЙ ОТВЕТ: {correct_answer}
            ОТВЕТ УЧЕНИКА: {student_answer}
            
            УЧЕНИК:
            - Возраст: {age} лет
            
            Контекст урока: {self.current_lesson_summary[:300]}
            
            🔥 ТРЕБОВАНИЯ:
            1. Оцени содержание ответа
            2. Учитывай возраст {age} лет
            3. Будь добрым и поддерживающим
            4. Отметь правильные части ответа
            5. Мягко исправь ошибки
            6. Похвали за усилия
            7. Максимум 3 предложения
            
            🔥 НЕ ИСПОЛЬЗУЙ ФРАЗЫ:
            - "Хороший вопрос! Давайте разберем..."
            - "Мне нужно время подумать"
            - "Спасибо за вопрос"
            - "Давайте разберем подробнее"
            
            Вместо этого дай конкретную обратную связь.
            
            Верни только оценку.
            """
            
            evaluation = self.llm.query(
                question=prompt,
                context="",
                subject=self.current_subject
            )
            
            if evaluation and len(evaluation.strip()) > 20:
                return evaluation
            
            return f"Спасибо за ответ! Основная мысль: {correct_answer}"
            
        except Exception as e:
            debug_log(f"❌ Ошибка оценки общего ответа: {e}")
            return f"Спасибо за ответ! Основная мысль: {correct_answer}"

    def _generate_correct_answer_for_test(self, question: str) -> str:
        """Генерирует правильный ответ на тестовый вопрос"""
        try:
            prompt = f"""
            Этот тестовый вопрос был задан ученику:
            
            {question}
            
            Какой ПРАВИЛЬНЫЙ ответ на этот тестовый вопрос?
            Укажи только правильный вариант (A, B, C или D) и краткое объяснение.
            Объяснение должно быть понятным для ученика.
            
            Формат:
            [Правильный вариант]: [Краткое объяснение]
            
            Пример:
            A) Клетка: Потому что клетка - это основная единица живых организмов.
            """
            
            llm_response = self.llm.query(
                question=prompt,
                context="",
                subject=self.current_subject
            )
            return llm_response.strip() if llm_response else "Не удалось определить правильный ответ."
            
        except Exception as e:
            debug_log(f"❌ Ошибка генерации правильного ответа для теста: {e}")
            return "Правильный ответ не определен."

    def _generate_correct_answer(self, question: str) -> str:
        """Генерирует правильный ответ на обычный вопрос"""
        try:
            prompt = f"""
            Дай точный и краткий ответ на вопрос.
            
            ВОПРОС: {question}
            КОНТЕКСТ: {self.current_lesson_summary}
            ПРЕДМЕТ: {self.current_subject}
            
            Верни только краткий правильный ответ (1-2 предложения).
            Ответ должен быть понятным для ученика.
            """
            
            llm_response = self.llm.query(
                question=prompt,
                context="",
                subject=self.current_subject
            )
            return llm_response.strip() if llm_response else "Ответ не найден в материале урока."
            
        except Exception as e:
            debug_log(f"❌ Ошибка генерации правильного ответа: {e}")
            return "Не удалось сгенерировать ответ."

    def stop_async_generation(self):
        """Останавливает фоновую генерацию"""
        self.stop_generation = True
        self.generation_active = False
        if self.generation_thread and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=1.0)
            debug_log("✅ Фоновая генерация остановлена")

    def _get_previous_questions_text(self) -> str:
        """Возвращает текст уже заданных вопросов"""
        if not self.generated_questions:
            return "Вопросов еще не было. Это первый вопрос."
        
        questions_text = "Уже заданные вопросы:\n"
        for i, q_data in enumerate(self.generated_questions, 1):
            question_preview = q_data["question"][:80] + "..." if len(q_data["question"]) > 80 else q_data["question"]
            questions_text += f"{i}. {question_preview}\n"
        return questions_text

    def _is_question_unique(self, new_question: str, similarity_threshold: float = 0.7) -> bool:
        """Проверяет, является ли вопрос уникальным"""
        if not self.generated_questions:
            return True
        
        new_question_lower = new_question.lower()
        for existing_q in self.generated_questions:
            existing_question_lower = existing_q["question"].lower()
            similarity = SequenceMatcher(None, new_question_lower, existing_question_lower).ratio()
            
            if similarity > similarity_threshold:
                debug_log(f"⚠️ Вопрос похож на существующий (similarity: {similarity:.2f})")
                return False
        return True

    def _clean_question_text(self, text: str) -> str:
        """Очищает текст вопроса от лишних символов"""
        if not text:
            return ""
        text = re.sub(r'^\d+\.\s*', '', text.strip())
        text = re.sub(r'^[•\-]\s*', '', text)
        text = re.sub(r'^вопрос\s*\d*:*\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'["«»]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _get_fallback_question_for_age(self, age: int, ensure_unique: bool = False) -> str:
        """Возвращает fallback вопросы для конкретного возраста"""
        # Определяем возрастную группу
        if age <= 10:
            age_group = "young"
        elif age <= 14:
            age_group = "middle" 
        else:
            age_group = "old"
        
        # Возрастные вопросы для разных предметов
        subject_lower = self.current_subject.lower()
        
        # 🔥 СПЕЦИАЛЬНЫЕ ВОПРОСЫ ДЛЯ ТЕХНИЧЕСКИХ ПРЕДМЕТОВ
        if self.is_technical_subject or self.is_natural_science_subject:
            if any(math_word in subject_lower for math_word in ['математика', 'алгебра', 'геометрия']):
                age_questions = {
                    "young": [
                        "Посчитай, сколько будет 5 + 3?",
                        "Назови геометрические фигуры, которые ты знаешь.",
                        "Что такое цифра и число?",
                        "Как сравнить два числа?",
                        "Реши простую задачу на сложение."
                    ],
                    "middle": [
                        "Объясни основную концепцию из урока.",
                        "Реши простую задачу по теме урока.",
                        "Какие формулы мы сегодня изучали?",
                        "Приведи пример применения изученного материала.",
                        "В чем особенность изученного подхода?"
                    ],
                    "old": [
                        "Проанализируй изученную теорию.",
                        "Реши задачу средней сложности по теме.",
                        "Докажи или объясни основной принцип.",
                        "Каковы практические применения этой теории?",
                        "Сравни изученные методы или подходы."
                    ]
                }
            
            elif 'физика' in subject_lower:
                age_questions = {
                    "young": [
                        "Что такое сила и как она проявляется?",
                        "Назови простые физические явления.",
                        "Как измеряется скорость?",
                        "Что такое гравитация?",
                        "Какие приборы для измерений ты знаешь?"
                    ],
                    "middle": [
                        "Объясни физический закон из урока.",
                        "Реши простую расчетную задачу.",
                        "Какие формулы мы изучали?",
                        "Приведи пример физического явления.",
                        "В чем практическое значение изученного?"
                    ],
                    "old": [
                        "Проанализируй физический процесс.",
                        "Реши задачу с применением формул.",
                        "Объясни теорию своими словами.",
                        "Каковы практические применения?",
                        "Какие эксперименты можно провести?"
                    ]
                }
            
            elif 'химия' in subject_lower:
                age_questions = [
                    "Назови простое химическое вещество.",
                    "Что такое химическая реакция?",
                    "Приведи пример безопасного химического опыта.",
                    "Какие химические элементы ты знаешь?",
                    "Как химия применяется в жизни?"
                ]
            
            else:
                # Общие вопросы для технических предметов
                age_questions = [
                    "Объясни основной принцип из урока.",
                    "Приведи пример применения изученного.",
                    "Какие ключевые понятия мы сегодня изучили?",
                    "Как можно использовать эти знания на практике?",
                    "В чем важность изученной темы?"
                ]
            
            questions = age_questions.get(age_group, age_questions)
            
        elif any(subj in subject_lower for subj in ['обществознание', 'общество', 'социология']):
            age_questions = {
                "young": [
                    "Что такое семья и зачем она нужна?",
                    "Как нужно вести себя в школе?",
                    "Что такое дружба и почему она важна?",
                    "Какие правила поведения ты знаешь?",
                    "Что такое доброта и как ее проявлять?"
                ],
                "middle": [
                    "Что такое общество и каковы его основные элементы?",
                    "Объясни понятие 'социальный институт' и приведи примеры.",
                    "В чем разница между формальными и неформальными социальными нормами?",
                    "Какие функции выполняет государство в современном обществе?",
                    "Что такое гражданское общество и как оно взаимодействует с государством?"
                ],
                "old": [
                    "Проанализируй основные социальные институты и их функции в обществе.",
                    "В чем заключается сущность правового государства?",
                    "Каковы основные принципы гражданского общества?",
                    "Объясни взаимосвязь экономики, политики и социальной сферы.",
                    "Какие глобальные проблемы современного общества ты можешь назвать?"
                ]
            }
            questions = age_questions.get(age_group, age_questions["middle"])
            
        elif 'история' in subject_lower:
            age_questions = {
                "young": [
                    "Кто такие древние люди и как они жили?",
                    "Что такое Родина и почему ее нужно любить?",
                    "Какие праздники нашей страны ты знаешь?",
                    "Кто такие герои и почему их помнят?",
                    "Что такое музей и зачем туда ходят?"
                ],
                "middle": [
                    "Каковы были ключевые события изученного периода?",
                    "Как повлияли эти исторические события на современность?",
                    "В чем заключались основные причины исторических процессов, которые мы изучали?",
                    "Охарактеризуй ключевых исторических личностей этого периода.",
                    "Какие исторические закономерности можно проследить в изученном материале?"
                ],
                "old": [
                    "Проанализируй причинно-следственные связи исторических событий.",
                    "Сравни различные исторические периоды по ключевым параметрам.",
                    "Оцени влияние исторических процессов на современное общество.",
                    "Какие альтернативные исторические развития возможны были?",
                    "Как исторический контекст влияет на интерпретацию событий?"
                ]
            }
            questions = age_questions.get(age_group, age_questions["middle"])
            
        else:
            # Базовый набор вопросов для любого предмета
            general_questions = {
                "young": [
                    "Объясни самое главное, что мы сегодня узнали.",
                    "Что тебе больше всего понравилось в уроке?",
                    "Как можно использовать эти знания в жизни?",
                    "Что было самым интересным?",
                    "О чем бы ты хотел узнать больше?"
                ],
                "middle": [
                    "Объясни основную идею изученного материала.",
                    "В чем заключается главная мысль этого урока?",
                    "Какие ключевые понятия мы сегодня изучили?",
                    "Как можно применить эти знания на практике?",
                    "Почему эта тема важна для понимания предмета?"
                ],
                "old": [
                    "Проанализируйте основные концепции изученного материала.",
                    "Каковы практические применения этих знаний?",
                    "Сравните изученные понятия с ранее известными.",
                    "Какие перспективы развития этой темы вы видите?",
                    "Как эти знания связаны с другими областями науки?"
                ]
            }
            questions = general_questions.get(age_group, ["Расскажите об основном понятии из урока?"])
        
        if ensure_unique and self.generated_questions:
            # Ищем вопрос, которого еще не было
            existing_questions = [q["question"] for q in self.generated_questions]
            for question in questions:
                if question not in existing_questions:
                    return question
            
            # Если все вопросы уже использованы, берем из общего списка
            for general_q in general_questions.get(age_group, []):
                if general_q not in existing_questions:
                    return general_q
        
        # Возвращаем случайный вопрос
        import random
        return random.choice(questions)

    def _get_language_fallback_question(self, exercise_type: str, ensure_unique: bool = False) -> str:
        """Fallback вопросы для языковой практики"""
        fallback_questions = {
            'vocabulary': [
                f"Назови 3 слова на {self.target_language} по теме урока.",
                f"Как переводится слово 'привет' на {self.target_language}?",
                f"Составь простое предложение на {self.target_language}.",
                f"Какие слова на {self.target_language} ты запомнил из урока?"
            ],
            'translation': [
                f"Переведи на {self.target_language}: 'Меня зовут...'",
                f"Как сказать 'Спасибо' на {self.target_language}?",
                f"Переведи: 'Я учу {self.target_language}'.",
                f"Как будет 'До свидания' на {self.target_language}?"
            ],
            'grammar': [
                f"Составь вопрос на {self.target_language}.",
                f"Поставь глагол в правильную форму для {self.language_level} уровня.",
                f"Исправь ошибку в предложении на {self.target_language}.",
                f"Составь отрицательное предложение на {self.target_language}."
            ],
            'fill_blank': [
                f"Заполни пропуск: 'I ___ a student.' (am/is/are)",
                f"Вставь правильное слово: 'My name ___ John.'",
                f"Заполни пропуск в предложении на {self.target_language}.",
                f"Выбери правильный вариант для заполнения пропуска."
            ],
            'dialogue': [
                f"Составь простой диалог на {self.target_language}.",
                f"Как поздороваться на {self.target_language}?",
                f"Задай вопрос на {self.target_language}.",
                f"Ответь на вопрос на {self.target_language}."
            ]
        }
        
        questions = fallback_questions.get(exercise_type, fallback_questions['vocabulary'])
        
        if ensure_unique and self.generated_questions:
            existing_questions = [q["question"] for q in self.generated_questions]
            for question in questions:
                if question not in existing_questions:
                    return question
        
        import random
        return random.choice(questions)

    def _get_technical_fallback_question(self, exercise_type: str, ensure_unique: bool = False) -> str:
        """Fallback вопросы для технической практики"""
        
        # Вопросы по типам упражнений
        exercise_fallbacks = {
            'calculation': [
                "Решите простой пример по теме урока.",
                "Выполните расчет по изученной формуле.",
                "Рассчитайте значение по заданным параметрам.",
                "Решите задачу на применение изученного метода."
            ],
            'problem_solving': [
                "Решите практическую задачу по теме.",
                "Примените изученный метод к конкретной ситуации.",
                "Предложите решение для проблемы, связанной с темой.",
                "Как бы вы решили эту задачу на практике?"
            ],
            'analysis': [
                "Проанализируйте ситуацию с точки зрения изученной теории.",
                "Сравните два подхода или метода.",
                "Оцените эффективность изученного подхода.",
                "Какие выводы можно сделать из изученного материала?"
            ],
            'theory': [
                "Объясните основной принцип изученной теории.",
                "В чем суть изученного закона или правила?",
                "Каковы основные положения изученного материала?",
                "Как работает изученный механизм или процесс?"
            ],
            'application': [
                "Где можно применить изученные знания?",
                "Приведите пример практического использования.",
                "Как это работает в реальной жизни?",
                "Каковы практические следствия изученного?"
            ]
        }
        
        # Предметные вопросы
        subject_lower = self.current_subject.lower()
        
        if any(math_word in subject_lower for math_word in ['математика', 'алгебра', 'геометрия']):
            questions = [
                "Решите математический пример по теме урока.",
                "Примените изученную формулу к задаче.",
                "Докажите или объясните математическое утверждение.",
                "Постройте график или схему по теме.",
                "Решите задачу на логическое мышление."
            ]
        
        elif 'физика' in subject_lower:
            questions = [
                "Решите физическую задачу по изученному закону.",
                "Объясните физическое явление с точки зрения теории.",
                "Рассчитайте физическую величину по формуле.",
                "Опишите эксперимент по изученной теме.",
                "Где применяется изученный физический закон?"
            ]
        
        elif 'химия' in subject_lower:
            questions = [
                "Напишите химическое уравнение по теме.",
                "Объясните химический процесс или реакцию.",
                "Рассчитайте массу или количество вещества.",
                "Опишите химический эксперимент.",
                "Каковы практические применения этого вещества?"
            ]
        
        elif 'биология' in subject_lower:
            questions = [
                "Опишите биологический процесс или механизм.",
                "Объясните строение или функцию организма.",
                "Классифицируйте объекты по изученным признакам.",
                "Опишите биологический эксперимент или наблюдение.",
                "Каковы экологические последствия изученного?"
            ]
        
        else:
            # Общие технические вопросы
            questions = exercise_fallbacks.get(exercise_type, [
                "Решите задачу по теме урока.",
                "Объясните основной принцип изученного.",
                "Приведите пример практического применения.",
                "Проанализируйте ситуацию с точки зрения изученного.",
                "Какие выводы можно сделать из урока?"
            ])
        
        if ensure_unique and self.generated_questions:
            existing_questions = [q["question"] for q in self.generated_questions]
            for question in questions:
                if question not in existing_questions:
                    return question
        
        import random
        return random.choice(questions)

    def _get_fallback_question(self, ensure_unique: bool = False) -> str:
        """Старый метод для обратной совместимости"""
        age = int(self.student_data.get('age', 12)) if self.student_data else 12
        return self._get_fallback_question_for_age(age, ensure_unique)

    def has_more_questions(self) -> bool:
        """Проверяет, можно ли генерировать еще вопросы"""
        return len(self.generated_questions) < self.max_questions

    def get_generated_questions_count(self) -> int:
        """Возвращает количество сгенерированных вопросов"""
        return len(self.generated_questions)

    def reset(self):
        """Сброс состояния менеджера практики"""
        self.stop_async_generation()
        self.current_lesson_context = ""
        self.current_lesson_summary = ""
        self.current_subject = ""
        self.current_subject_type = "general"
        self.is_language_subject = False
        self.is_technical_subject = False
        self.is_natural_science_subject = False
        self.target_language = None
        self.language_level = 'beginner'
        self.bilingual_ratio = 0.3
        self.generated_questions = []
        self.current_question_index = 0
        self.current_practice_type = 'general'
        
        # Очищаем очередь
        while not self.question_queue.empty():
            try:
                self.question_queue.get_nowait()
            except queue.Empty:
                break
        
        debug_log("🔄 Менеджер практики сброшен")

    def get_current_question(self) -> Optional[str]:
        """Возвращает текущий вопрос"""
        if self.generated_questions and self.current_question_index < len(self.generated_questions):
            return self.generated_questions[self.current_question_index]["question"]
        elif self.generated_questions:
            return self.generated_questions[-1]["question"]
        return None

    def get_practice_stats(self) -> Dict:
        """Возвращает статистику по практике"""
        question_types = {}
        test_questions_count = 0
        language_questions_count = 0
        technical_questions_count = 0
        
        for q in self.generated_questions:
            q_type = q.get("type", "unknown")
            question_types[q_type] = question_types.get(q_type, 0) + 1
            
            if q_type == "test_question":
                test_questions_count += 1
            elif "language_" in q_type:
                language_questions_count += 1
            elif "technical_" in q_type:
                technical_questions_count += 1
        
        age = self.student_data.get('age', 'неизвестен') if self.student_data else 'неизвестен'
        level = self.student_data.get('level', 'неизвестен') if self.student_data else 'неизвестен'
        
        stats = {
            "total_questions": len(self.generated_questions),
            "max_questions": self.max_questions,
            "questions_in_queue": self.question_queue.qsize(),
            "generation_active": self.generation_active,
            "current_subject": self.current_subject,
            "current_subject_type": self.current_subject_type,
            "current_practice_type": self.current_practice_type,
            "has_more_questions": self.has_more_questions(),
            "question_types": question_types,
            "test_questions_count": test_questions_count,
            "language_questions_count": language_questions_count,
            "technical_questions_count": technical_questions_count,
            "lesson_summary_length": len(self.current_lesson_summary),
            "student_data": {
                "age": age,
                "level": level
            },
            "is_language_subject": self.is_language_subject,
            "is_technical_subject": self.is_technical_subject,
            "is_natural_science": self.is_natural_science_subject,
            "target_language": self.target_language if self.is_language_subject else None,
            "language_level": self.language_level if self.is_language_subject else None,
            "bilingual_ratio": self.bilingual_ratio if self.is_language_subject else None
        }
        
        if self.generated_questions:
            total = len(self.generated_questions)
            stats["test_questions_percentage"] = round((test_questions_count / total) * 100, 1)
            stats["language_questions_percentage"] = round((language_questions_count / total) * 100, 1)
            stats["technical_questions_percentage"] = round((technical_questions_count / total) * 100, 1)
        else:
            stats["test_questions_percentage"] = 0
            stats["language_questions_percentage"] = 0
            stats["technical_questions_percentage"] = 0
        
        return stats

    def save_practice_session(self, filename: str = None) -> bool:
        """Сохраняет сессию практики в файл"""
        try:
            if not filename:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                subject_slug = re.sub(r'[^\w]', '_', self.current_subject.lower())
                filename = f"practice_session_{subject_slug}_{timestamp}.json"
            
            filepath = self.practice_dir / filename
            
            session_data = {
                "timestamp": time.time(),
                "subject": self.current_subject,
                "subject_type": self.current_subject_type,
                "practice_type": self.current_practice_type,
                "lesson_summary": self.current_lesson_summary,
                "generated_questions": self.generated_questions,
                "student_data": self.student_data,
                "stats": self.get_practice_stats(),
                "is_language_subject": self.is_language_subject,
                "is_technical_subject": self.is_technical_subject,
                "is_natural_science": self.is_natural_science_subject,
                "target_language": self.target_language,
                "language_level": self.language_level
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            
            debug_log(f"💾 Сессия практики сохранена в {filepath}")
            return True
            
        except Exception as e:
            debug_log(f"❌ Ошибка сохранения сессии практики: {e}")
            return False

    def load_practice_session(self, filepath: str) -> bool:
        """Загружает сессию практики из файла"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            self.current_subject = session_data.get("subject", "")
            self.current_subject_type = session_data.get("subject_type", "general")
            self.current_practice_type = session_data.get("practice_type", "general")
            self.current_lesson_summary = session_data.get("lesson_summary", "")
            self.generated_questions = session_data.get("generated_questions", [])
            self.student_data = session_data.get("student_data", {})
            self.is_language_subject = session_data.get("is_language_subject", False)
            self.is_technical_subject = session_data.get("is_technical_subject", False)
            self.is_natural_science_subject = session_data.get("is_natural_science", False)
            self.target_language = session_data.get("target_language")
            self.language_level = session_data.get("language_level", "beginner")
            
            debug_log(f"📂 Сессия практики загружена из {filepath}")
            return True
            
        except Exception as e:
            debug_log(f"❌ Ошибка загрузки сессии практики: {e}")
            return False


# 🔥 КЛАСС ДЛЯ ТЕСТИРОВАНИЯ
class MockLLM:
    """Mock LLM для тестирования"""
    def query(self, question, context, subject):
        if "английский" in subject.lower():
            return """Тестовый вопрос по английскому языку.

Выберите правильный перевод слова "дом":
A) house
B) home  
C) building
D) apartment

Правильный ответ: A) house"""
        elif "математика" in subject.lower():
            return """Решите задачу по математике:

Найдите площадь прямоугольника со сторонами 5 см и 8 см.
Используйте формулу S = a × b.

Площадь: 40 см²"""
        else:
            return "Тестовый вопрос по предмету."


# 🔥 ТЕСТИРОВАНИЕ
if __name__ == "__main__":
    print("🧪 Тестирование PracticeManager с поддержкой технических предметов...")
    
    # Тестируем менеджер практики
    pm = PracticeManager(MockLLM())
    
    # Тест 1: Языковой предмет
    print("\n📚 Тест 1: Английский язык (языковой предмет)")
    pm.student_data = {'name': 'Ученик', 'age': '12', 'level': '5'}
    pm.initialize_practice_generation("Урок о базовых словах на английском: hello, goodbye, thank you", "английский язык")
    time.sleep(1)
    
    question1 = pm.get_next_question()
    print(f"📝 Языковой вопрос:\n{question1[:100]}...")
    
    # Тест 2: Технический предмет
    print("\n📚 Тест 2: Математика (технический предмет)")
    pm.reset()
    pm.student_data = {'name': 'Ученик', 'age': '14', 'level': '8'}
    pm.initialize_practice_generation("Урок о площади прямоугольника: S = a × b", "математика")
    time.sleep(1)
    
    question2 = pm.get_next_question()
    print(f"📝 Технический вопрос:\n{question2[:100]}...")
    
    # Тест 3: Гуманитарный предмет
    print("\n📚 Тест 3: История (гуманитарный предмет)")
    pm.reset()
    pm.student_data = {'name': 'Ученик', 'age': '15', 'level': '9'}
    pm.initialize_practice_generation("Урок о Древней Руси и князьях", "история")
    time.sleep(1)
    
    question3 = pm.get_next_question()
    print(f"📝 Гуманитарный вопрос:\n{question3[:100]}...")
    
    # Тест статистики
    stats = pm.get_practice_stats()
    print(f"\n📊 Статистика практики:")
    print(f"   Тип предмета: {stats['current_subject_type']}")
    print(f"   Тип практики: {stats['current_practice_type']}")
    print(f"   Всего вопросов: {stats['total_questions']}/{stats['max_questions']}")
    print(f"   Вопросов в очереди: {stats['questions_in_queue']}")
    
    # Тест оценки ответа
    print("\n📝 Тест оценки ответа:")
    feedback, next_q = pm.evaluate_and_continue("Площадь равна 40 см²", question2)
    print(f"   Feedback: {feedback}")
    
    print("\n✅ Тестирование PracticeManager завершено!")