import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
import time
import threading
import queue

def debug_log(message):
    """Логирование для отладки"""
    print(f"🔥 [PRACTICE] {message}")

class PracticeManager:
    def __init__(self, llm_integration):
        self.llm = llm_integration
        self.practice_dir = Path("materials/practice")
        self.current_lesson_context = ""
        self.current_lesson_summary = ""
        self.current_subject = ""
        
        # 🔥 НОВОЕ: Данные ученика для адаптации сложности
        self.student_data = {}
        self.is_language_subject = False
        self.target_language = None
        self.language_level = 'beginner'
        
        # 🔥 НОВОЕ: Менеджеры для разных типов практики
        self.language_practice_manager = None
        self.current_practice_type = 'general'  # 'general' или 'language'
        
        # ОЧЕРЕДИ ДЛЯ АСИНХРОННОЙ ГЕНЕРАЦИИ
        self.question_queue = queue.Queue()  # Очередь готовых вопросов
        self.generated_questions = []        # История всех вопросов
        self.current_question_index = 0
        self.max_questions = 5  # ЖЕСТКИЙ ЛИМИТ 5 ВОПРОСОВ
        
        # ФЛАГИ УПРАВЛЕНИЯ АСИНХРОННОЙ ГЕНЕРАЦИИ
        self.generation_thread = None
        self.stop_generation = False
        self.generation_active = False
        
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
            'китайский язык', 'chinese', 'китайский'
        ]
        
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
        else:
            return 'english'  # По умолчанию

    def initialize_practice_generation(self, lesson_context: str, subject: str):
        """Инициализирует практику и ЗАРАНЕЕ начинает генерацию вопросов"""
        self.current_lesson_context = lesson_context
        self.current_subject = subject
        
        # 🔥 НОВОЕ: Определяем тип практики
        self.is_language_subject = self._is_language_subject(subject)
        self.target_language = self._extract_target_language(subject) if self.is_language_subject else None
        
        # Определяем уровень ученика для языковой практики
        if self.is_language_subject and self.student_data:
            age = int(self.student_data.get('age', 12))
            if age <= 10:
                self.language_level = 'beginner'
            elif age <= 14:
                self.language_level = 'intermediate'
            else:
                self.language_level = 'advanced'
        
        debug_log(f"🎯 Инициализация практики. Предмет: {subject}, Языковой: {self.is_language_subject}")
        
        if self.is_language_subject:
            # Инициализируем языковую практику
            self.current_practice_type = 'language'
            
            # 🔥 ОТКЛЮЧАЕМ: Не используем LanguagePracticeManager пока он не готов
            # Вместо этого используем улучшенную версию существующего менеджера
            debug_log("🔧 Использую улучшенный режим для языковых предметов")
            
            self.current_lesson_summary = self._generate_language_lesson_summary(lesson_context)
            self.generated_questions = []
            self.current_question_index = 0
            
            # Очищаем очередь
            while not self.question_queue.empty():
                try:
                    self.question_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Запускаем асинхронную генерацию
            self._start_async_generation()
            
            # Первый вопрос синхронно
            first_question = self.generate_single_language_question()
            if first_question:
                self.question_queue.put(first_question)
                debug_log(f"✅ Первый языковой вопрос готов: {first_question[:80]}...")
        else:
            # Общая практика для неязыковых предметов
            self.current_practice_type = 'general'
            self.current_lesson_summary = self._generate_lesson_summary(lesson_context)
            self.generated_questions = []
            self.current_question_index = 0
            
            # Очищаем очередь
            while not self.question_queue.empty():
                try:
                    self.question_queue.get_nowait()
                except queue.Empty:
                    break
            
            debug_log(f"🎯 Инициализирована генерация практики для предмета: {subject}")
            debug_log(f"📝 Максимальное количество вопросов: {self.max_questions}")
            
            # 🔥 Логируем данные ученика для отладки
            if self.student_data:
                age = self.student_data.get('age', 'неизвестен')
                level = self.student_data.get('level', 'неизвестен')
                debug_log(f"👤 Данные ученика: возраст {age} лет, {level} класс")
            
            # ЗАПУСКАЕМ АСИНХРОННУЮ ГЕНЕРАЦИЮ ВОПРОСОВ СРАЗУ
            self._start_async_generation()
            
            # Генерируем ПЕРВЫЙ вопрос СИНХРОННО для немедленного старта
            first_question = self.generate_single_question()
            if first_question:
                self.question_queue.put(first_question)
                debug_log(f"✅ Первый вопрос готов: {first_question[:80]}...")

    def _generate_language_lesson_summary(self, lesson_context: str) -> str:
        """Генерирует краткое содержание урока для языковой практики"""
        try:
            prompt = f"""
            Создай КРАТКОЕ содержание этого языкового урока для практических упражнений.
            Выдели ключевые слова, фразы и грамматические конструкции.
            
            ТЕКСТ УРОКА: {lesson_context[:1500]}
            ЯЗЫК: {self.target_language}
            УРОВЕНЬ: {self.language_level}
            
            Верни только краткое содержание (максимум 300 слов).
            Включи:
            1. Ключевые слова и фразы
            2. Основные грамматические правила
            3. Примеры диалогов или предложений
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

    def _start_async_generation(self):
        """Запускает фоновую генерацию вопросов"""
        if self.generation_active:
            return
            
        self.stop_generation = False
        self.generation_active = True
        
        def generate_questions_worker():
            debug_log("🔄 Фоновая генерация вопросов запущена...")
            
            while (not self.stop_generation and 
                   self.generation_active and 
                   len(self.generated_questions) < self.max_questions - 1):  # -1 потому что первый уже сгенерирован
                
                try:
                    # ВАЖНОЕ ИЗМЕНЕНИЕ: Проверяем лимит перед генерацией
                    if len(self.generated_questions) >= self.max_questions:
                        debug_log("🏁 Достигнут лимит вопросов в фоновой генерации")
                        break
                    
                    # Генерируем следующий вопрос
                    if self.is_language_subject:
                        next_question = self.generate_single_language_question()
                    else:
                        next_question = self.generate_single_question()
                    
                    if next_question:
                        self.question_queue.put(next_question)
                        debug_log(f"✅ Фоново сгенерирован вопрос {len(self.generated_questions)}/{self.max_questions}")
                    
                    # Небольшая пауза между генерацией
                    time.sleep(1)
                    
                except Exception as e:
                    debug_log(f"❌ Ошибка в фоновой генерации: {e}")
                    time.sleep(2)  # Пауза при ошибке
            
            debug_log("🏁 Фоновая генерация вопросов завершена")
            self.generation_active = False
        
        # Запускаем в отдельном потоке
        self.generation_thread = threading.Thread(target=generate_questions_worker, daemon=True)
        self.generation_thread.start()

    def generate_single_question(self) -> Optional[str]:
        """Генерирует один ТЕСТОВЫЙ вопрос с вариантами ответов A, B, C, D (для неязыковых предметов)"""
        try:
            # ВАЖНОЕ ИЗМЕНЕНИЕ: Проверяем лимит вопросов
            if len(self.generated_questions) >= self.max_questions:
                debug_log(f"🏁 Достигнут лимит вопросов: {len(self.generated_questions)}/{self.max_questions}")
                return None
            
            # 🔥 ОБНОВЛЕННЫЙ ПРОМТ: Генерируем ТЕСТОВЫЙ вопрос с вариантами
            prompt = f"""
            СОЗДАЙ ТЕСТОВЫЙ ВОПРОС ДЛЯ ПРОВЕРКИ ПОНИМАНИЯ УРОКА.
            
            МАТЕРИАЛ УРОКА:
            {self.current_lesson_summary}
            
            ПРЕДМЕТ: {self.current_subject}
            
            ТРЕБОВАНИЯ К ВОПРОСУ:
            1. ВОПРОС ДОЛЖЕН БЫТЬ ИСКЛЮЧИТЕЛЬНО ПО СОДЕРЖАНИЮ ЭТОГО УРОКА
            2. НЕ ВЫХОДИ ЗА ПРЕДЕЛЫ МАТЕРИАЛА УРОКА  
            3. ВОПРОС НЕ ДОЛЖЕН ПОВТОРЯТЬСЯ С ПРЕДЫДУЩИМИ
            4. СОЗДАЙ 4 ВАРИАНТА ОТВЕТА: A, B, C, D
            5. НЕ ВКЛЮЧАЙ ПРАВИЛЬНЫЙ ОТВЕТ В ТЕКСТ ВОПРОСА
            6. ВОПРОС ДОЛЖЕН БЫТЬ ОДНОЗНАЧНЫМ
            7. ВОПРОС НЕ ДОЛЖЕН БЫТЬ ОЧЕНЬ СЛОЖНЫМ
            
            ФОРМАТ:
            [Текст вопроса?]
            
            Варианты ответов:
            A) [Текст варианта A]
            B) [Текст варианта B] 
            C) [Текст варианта C]
            D) [Текст варианта D]
            
            УЖЕ ЗАДАННЫЕ ВОПРОСЫ (НЕ ПОВТОРЯЙ!):
            {self._get_previous_questions_text()}
            
            Верни только вопрос в указанном формате.
            """
            
            llm_response = self.llm.query(
                question=prompt,
                context="",
                subject=self.current_subject
            )
            
            if llm_response and len(llm_response.strip()) > 50:
                # Проверяем, что есть варианты ответов
                if any(marker in llm_response for marker in ["A)", "B)", "C)", "D)"]):
                    question_text = llm_response.strip()
                    
                    if self._is_question_unique(question_text):
                        self.generated_questions.append({
                            "question": question_text,
                            "generated_at": time.time(),
                            "type": "test_question",
                            "subject": self.current_subject
                        })
                        return question_text
            
            # 🔥 FALLBACK: Если не получился тестовый вопрос - обычный вопрос
            fallback = self._get_fallback_question(ensure_unique=True)
            debug_log(f"🔄 Использован fallback вопрос: {fallback[:80]}...")
            return fallback
            
        except Exception as e:
            debug_log(f"❌ Ошибка генерации тестового вопроса: {e}")
            fallback = self._get_fallback_question(ensure_unique=True)
            debug_log(f"🔄 Использован fallback после ошибки: {fallback[:80]}...")
            return fallback

    def generate_single_language_question(self) -> Optional[str]:
        """Генерирует один вопрос для языковой практики"""
        try:
            if len(self.generated_questions) >= self.max_questions:
                return None
            
            # Выбираем тип языкового упражнения в зависимости от уровня
            exercise_types_by_level = {
                'beginner': ['vocabulary', 'simple_translation', 'fill_blank'],
                'intermediate': ['grammar', 'dialogue', 'sentence_building'],
                'advanced': ['composition', 'error_correction', 'debate']
            }
            
            available_types = exercise_types_by_level.get(self.language_level, ['vocabulary', 'translation'])
            
            import random
            exercise_type = random.choice(available_types)
            
            prompt = self._create_language_exercise_prompt(exercise_type)
            
            llm_response = self.llm.query(
                question=prompt,
                context="",
                subject=f"{self.target_language} language"
            )
            
            if llm_response and len(llm_response.strip()) > 30:
                question_text = llm_response.strip()
                
                if self._is_question_unique(question_text):
                    self.generated_questions.append({
                        "question": question_text,
                        "generated_at": time.time(),
                        "type": f"language_{exercise_type}",
                        "subject": self.current_subject,
                        "language": self.target_language,
                        "level": self.language_level
                    })
                    return question_text
            
            # Fallback для языкового вопроса
            fallback = self._get_language_fallback_question(exercise_type, ensure_unique=True)
            return fallback
            
        except Exception as e:
            debug_log(f"❌ Ошибка генерации языкового вопроса: {e}")
            return self._get_language_fallback_question('vocabulary', ensure_unique=True)

    def _create_language_exercise_prompt(self, exercise_type: str) -> str:
        """Создает промт для языкового упражнения"""
        prompts = {
            'vocabulary': f"""
            Создай упражнение на словарный запас для {self.target_language}.
            
            Уровень: {self.language_level}
            Тема урока: {self.current_lesson_summary[:200]}
            
            Создай упражнение одного из видов:
            1. Сопоставление слов с переводами
            2. Заполнение пропусков в предложениях
            3. Выбор правильного перевода
            
            Верни упражнение в формате:
            [Текст упражнения с четкой инструкцией]
            
            Используй смесь русского и {self.target_language} соответственно уровню {self.language_level}.
            """,
            
            'translation': f"""
            Создай упражнение на перевод для {self.target_language}.
            
            Уровень: {self.language_level}
            Тема: {self.current_lesson_summary[:200]}
            
            Создай 3 предложения для перевода с русского на {self.target_language}.
            Предложения должны соответствовать уровню {self.language_level}.
            
            Формат:
            Переведите на {self.target_language}:
            1. [Предложение на русском]
            2. [Предложение на русском]
            3. [Предложение на русском]
            """,
            
            'grammar': f"""
            Создай грамматическое упражнение для {self.target_language}.
            
            Уровень: {self.language_level}
            Тема урока: {self.current_lesson_summary[:200]}
            
            Создай упражнение на отработку грамматического правила.
            Включи инструкцию на русском и пример.
            
            Пример для beginner level:
            Поставьте глагол в правильную форму:
            I (to be) ___ a student. -> I am a student.
            """,
            
            'fill_blank': f"""
            Создай упражнение "fill in the blanks" для {self.target_language}.
            
            Уровень: {self.language_level}
            Тема: {self.current_lesson_summary[:200]}
            
            Создай текст с 3-5 пропусками.
            Пропуски должны быть ключевыми словами по теме.
            
            Формат:
            Заполните пропуски подходящими словами:
            [Текст с пропусками обозначенными как ______]
            """
        }
        
        return prompts.get(exercise_type, prompts['vocabulary'])

    def get_next_question(self, timeout: float = 10.0) -> Optional[str]:
        """Получает следующий вопрос из очереди (с ожиданием если нужно)"""
        try:
            # ВАЖНОЕ ИЗМЕНЕНИЕ: Проверяем лимит вопросов
            if len(self.generated_questions) >= self.max_questions:
                debug_log(f"🏁 Достигнут лимит вопросов: {len(self.generated_questions)}/{self.max_questions}")
                return None
            
            # Пытаемся взять вопрос из очереди без ожидания
            try:
                question = self.question_queue.get_nowait()
                debug_log(f"✅ Вопрос взят из очереди (в очереди еще: {self.question_queue.qsize()})")
                return question
            except queue.Empty:
                pass
            
            # Если очередь пуста, пробуем сгенерировать СИНХРОННО
            debug_log("⚠️ Очередь вопросов пуста, синхронная генерация...")
            
            if self.is_language_subject:
                question = self.generate_single_language_question()
            else:
                question = self.generate_single_question()
            
            if question:
                debug_log(f"✅ Синхронно сгенерирован вопрос: {question[:80]}...")
                return question
            else:
                # Если синхронная генерация не удалась, используем fallback
                if self.is_language_subject:
                    fallback = self._get_language_fallback_question('vocabulary', ensure_unique=True)
                else:
                    fallback = self._get_fallback_question(ensure_unique=True)
                
                debug_log(f"🔄 Использован fallback вопрос: {fallback[:80]}...")
                return fallback
                
        except Exception as e:
            debug_log(f"❌ Ошибка получения следующего вопроса: {e}")
            
            if self.is_language_subject:
                fallback = self._get_language_fallback_question('vocabulary', ensure_unique=True)
            else:
                fallback = self._get_fallback_question(ensure_unique=True)
            
            debug_log(f"🔄 Использован fallback после ошибки: {fallback[:80]}...")
            return fallback

    def evaluate_and_continue(self, student_answer: str, current_question: str) -> Tuple[str, Optional[str]]:
        """Оценивает ответ и возвращает feedback + следующий вопрос с учетом возраста"""
        try:
            # 1. Сначала оцениваем ответ (это быстро)
            if self.is_language_subject:
                feedback = self.evaluate_language_answer(student_answer, current_question)
            else:
                feedback = self.evaluate_single_answer(student_answer, current_question)
            
            # 2. Параллельно получаем следующий вопрос
            next_question = self.get_next_question()
            
            return feedback, next_question
            
        except Exception as e:
            debug_log(f"❌ Ошибка в evaluate_and_continue: {e}")
            # ИСПРАВЛЕННЫЙ FALLBACK ДЛЯ ПРАКТИКИ
            feedback = "Спасибо за ответ! Переходим к следующему вопросу."
            
            if self.is_language_subject:
                next_question = self._get_language_fallback_question('vocabulary', ensure_unique=True)
            else:
                age = self.student_data.get('age', '12')
                next_question = self._get_fallback_question_for_age(age, ensure_unique=True)
            
            return feedback, next_question

    def evaluate_single_answer(self, student_answer: str, question: str) -> str:
        """🔥 ОБНОВЛЕННЫЙ МЕТОД: Оценивает ответ ученика через LLM"""
        try:
            if not student_answer or len(student_answer.strip()) < 2:
                return "Ответ слишком короткий. Пожалуйста, попробуйте ответить более развернуто."
            
            command_words = ['продолжай', 'дальше', 'следующий']
            if any(cmd in student_answer.lower() for cmd in command_words):
                return "Это похоже на команду. Пожалуйста, дайте ответ на вопрос."
            
            # 🔥 ВАЖНО: Проверяем, это тестовый вопрос или обычный
            if any(marker in question for marker in ["A)", "B)", "C)", "D)"]):
                # 🔥 ЭТО ТЕСТОВЫЙ ВОПРОС - проверяем через LLM как обычно
                correct_answer = self._generate_correct_answer_for_test(question)
                evaluation = self._evaluate_with_llm_context(student_answer, question, correct_answer)
            else:
                # 🔥 ЭТО ОБЫЧНЫЙ ВОПРОС - используем старую логику
                correct_answer = self._generate_correct_answer(question)
                evaluation = self._evaluate_with_llm_context(student_answer, question, correct_answer)
            
            # ИСПРАВЛЕНИЕ: Заменяем неподходящие fallback ответы
            if not evaluation or any(phrase in evaluation for phrase in [
                "Хороший вопрос! Давайте разберем эту тему подробнее",
                "Мне нужно немного времени подумать",
                "Спасибо за вопрос! Я подумаю над ответом"
            ]):
                # Для тестовых вопросов даем более конкретный feedback
                if any(marker in question for marker in ["A)", "B)", "C)", "D)"]):
                    evaluation = "Спасибо за ответ! Давайте проверим правильность."
                else:
                    evaluation = "Спасибо за ответ! Переходим к следующему вопросу."
            
            return evaluation
            
        except Exception as e:
            debug_log(f"❌ Ошибка оценки ответа: {e}")
            # ИСПРАВЛЕННЫЙ FALLBACK
            return "Спасибо за ответ! Переходим к следующему вопросу."

    def evaluate_language_answer(self, student_answer: str, question: str) -> str:
        """Оценивает ответ на языковое упражнение"""
        try:
            if not student_answer or len(student_answer.strip()) < 1:
                return "Ответ слишком короткий. Попробуйте ответить на изучаемом языке."
            
            prompt = f"""
            Оцени ответ ученика на языковое упражнение.
            
            ЯЗЫК: {self.target_language}
            УРОВЕНЬ: {self.language_level}
            УПРАЖНЕНИЕ: {question}
            ОТВЕТ УЧЕНИКА: {student_answer}
            
            Дай конструктивную обратную связь на русском.
            Учитывай уровень {self.language_level}.
            Если есть ошибки - мягко исправь их.
            Похвали за правильные элементы.
            
            Верни только обратную связь (2-3 предложения).
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

    def _generate_correct_answer_for_test(self, question: str) -> Optional[str]:
        """Генерирует правильный ответ на тестовый вопрос"""
        try:
            # 🔥 ОБНОВЛЕННЫЙ ПРОМТ: Учитываем что это тестовый вопрос
            prompt = f"""
            Этот тестовый вопрос был задан ученику:
            
            {question}
            
            Какой ПРАВИЛЬНЫЙ ответ на этот тестовый вопрос?
            Укажи только правильный вариант (A, B, C или D) и краткое объяснение.
            """
            
            llm_response = self.llm.query(
                question=prompt,
                context="",
                subject=self.current_subject
            )
            return llm_response.strip() if llm_response else None
            
        except Exception as e:
            debug_log(f"❌ Ошибка генерации правильного ответа для теста: {e}")
            return None

    def _generate_correct_answer(self, question: str) -> Optional[str]:
        """Генерирует правильный ответ на обычный вопрос"""
        try:
            prompt = f"""
            Дай точный и краткий ответ на вопрос.
            
            ВОПРОС: {question}
            КОНТЕКСТ: {self.current_lesson_summary}
            ПРЕДМЕТ: {self.current_subject}
            
            Верни только краткий правильный ответ.
            """
            
            llm_response = self.llm.query(
                question=prompt,
                context="",
                subject=self.current_subject
            )
            return llm_response.strip() if llm_response else None
            
        except Exception as e:
            debug_log(f"❌ Ошибка генерации правильного ответа: {e}")
            return None

    def _evaluate_with_llm_context(self, student_answer: str, question: str, correct_answer: str) -> str:
        """🔥 ОБНОВЛЕННЫЙ МЕТОД: Оценивает ответ через LLM"""
        try:
            # 🔥 НОВОЕ: Получаем данные ученика для персонализации
            age = self.student_data.get('age', '12')
            level = self.student_data.get('level', '5')
            name = self.student_data.get('name', 'ученик')
            
            prompt = f"""
            Оцени ответ ученика на вопрос и дай обратную связь.

            ПАРАМЕТРЫ УЧЕНИКА:
            - Имя: {name}
            - Возраст: {age} лет
            - Уровень: {level} класс

            ВОПРОС: {question}
            ПРАВИЛЬНЫЙ ОТВЕТ: {correct_answer}
            ОТВЕТ УЧЕНИКА: {student_answer}

            Контекст урока: {self.current_lesson_summary[:300]}

            Дай добрую и поддерживающую обратную связь (2-3 предложения).
            Обращайся к ученику на "ты".
            Учитывай возраст {age} лет - будь терпеливым и понятным.

            НЕ ИСПОЛЬЗУЙ фразы:
            - "Хороший вопрос! Давайте разберем эту тему подробнее"
            - "Мне нужно немного времени подумать" 
            - "Спасибо за вопрос! Я подумаю над ответом"
            - "Давайте разберем эту тему подробнее"

            Вместо этого дай конкретную обратную связь по ответу, адаптированную для возраста {age} лет.
            """
            
            evaluation = self.llm.query(
                question=prompt,
                context="",
                subject=self.current_subject
            )
            
            # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА НА НЕПОДХОДЯЩИЕ ОТВЕТЫ
            if evaluation and any(phrase in evaluation for phrase in [
                "Хороший вопрос! Давайте разберем эту тему подробнее",
                "Мне нужно немного времени подумать",
                "Спасибо за вопрос! Я подумаю над ответом"
            ]):
                return f"Спасибо за ответ! Правильный ответ: {correct_answer}"
                
            return evaluation if evaluation else f"Спасибо за ответ! Правильный ответ: {correct_answer}"
            
        except Exception as e:
            debug_log(f"❌ Ошибка оценки через LLM: {e}")
            return f"Спасибо за ответ! Правильный ответ: {correct_answer}"

    def stop_async_generation(self):
        """Останавливает фоновую генерацию"""
        self.stop_generation = True
        self.generation_active = False
        if self.generation_thread and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=2.0)

    def _get_previous_questions_text(self) -> str:
        """Возвращает текст уже заданных вопросов"""
        if not self.generated_questions:
            return "Вопросов еще не было. Это первый вопрос."
        
        questions_text = "Уже заданные вопросы:\n"
        for i, q_data in enumerate(self.generated_questions, 1):
            # Берем только первые 100 символов вопроса для компактности
            question_preview = q_data["question"][:100] + "..." if len(q_data["question"]) > 100 else q_data["question"]
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

    def _generate_lesson_summary(self, lesson_context: str) -> str:
        """Генерирует краткое содержание урока"""
        try:
            prompt = f"""
            Создай КРАТКОЕ содержание этого урока для практических вопросов.
            Выдели только ключевые понятия и основные идеи.
            
            ТЕКСТ: {lesson_context[:1500]}
            ПРЕДМЕТ: {self.current_subject}
            
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

    def _get_fallback_question_for_age(self, age: int, ensure_unique: bool = False) -> str:
        """🔥 НОВЫЙ МЕТОД: Возвращает fallback вопросы для конкретного возраста"""
        # Определяем возрастную группу
        if age <= 10:
            age_group = "young"
        elif age <= 14:
            age_group = "middle" 
        else:
            age_group = "old"
        
        # Возрастные вопросы для разных предметов
        subject_lower = self.current_subject.lower()
        
        if 'обществознание' in subject_lower or 'общество' in subject_lower:
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
            
        elif 'математика' in subject_lower or 'алгебра' in subject_lower or 'геометрия' in subject_lower:
            age_questions = {
                "young": [
                    "Посчитай, сколько будет 5 + 3?",
                    "Что такое цифры и зачем они нужны?",
                    "Как сравнить числа 7 и 5?",
                    "Что такое геометрические фигуры?",
                    "Как решить простую задачу на сложение?"
                ],
                "middle": [
                    "Объясни основную концепцию, которую мы только что изучили.",
                    "Как применить изученный метод на практике?",
                    "В чем особенность этого математического подхода?",
                    "Какие существуют альтернативные способы решения этой задачи?",
                    "Почему этот математический принцип важен для понимания?"
                ],
                "old": [
                    "Докажи основную теорему изученного раздела.",
                    "Проанализируй применение этого математического метода в реальных задачах.",
                    "Каковы ограничения изученного алгоритма?",
                    "Сравни различные подходы к решению этой проблемы.",
                    "Какие практические применения имеет эта математическая концепция?"
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
            questions = general_questions.get(age_group, ["Расскажи об основном понятии из урока?"])
        
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
        self.is_language_subject = False
        self.target_language = None
        self.language_level = 'beginner'
        self.generated_questions = []
        self.current_question_index = 0
        while not self.question_queue.empty():
            try:
                self.question_queue.get_nowait()
            except queue.Empty:
                break
        debug_log("🔄 Менеджер практики сброшен")

    def get_current_question(self) -> Optional[str]:
        """Возвращает текущий вопрос"""
        if self.generated_questions:
            return self.generated_questions[-1]["question"]
        return None

    def get_practice_stats(self) -> Dict:
        """🔥 ОБНОВЛЕННЫЙ МЕТОД: Возвращает статистику по практике"""
        question_types = {}
        test_questions_count = 0
        language_questions_count = 0
        
        for q in self.generated_questions:
            q_type = q.get("type", "unknown")
            question_types[q_type] = question_types.get(q_type, 0) + 1
            
            if q_type == "test_question":
                test_questions_count += 1
            elif "language_" in q_type:
                language_questions_count += 1
        
        age = self.student_data.get('age', 'неизвестен') if self.student_data else 'неизвестен'
        level = self.student_data.get('level', 'неизвестен') if self.student_data else 'неизвестен'
        
        stats = {
            "total_questions": len(self.generated_questions),
            "max_questions": self.max_questions,
            "questions_in_queue": self.question_queue.qsize(),
            "generation_active": self.generation_active,
            "current_subject": self.current_subject,
            "has_more_questions": self.has_more_questions(),
            "question_types": question_types,
            "test_questions_count": test_questions_count,
            "language_questions_count": language_questions_count,
            "lesson_summary_length": len(self.current_lesson_summary),
            # 🔥 НОВОЕ: Статистика по адаптации
            "student_data": {
                "age": age,
                "level": level
            },
            "practice_type": "language" if self.is_language_subject else "general",
            "target_language": self.target_language if self.is_language_subject else None,
            "language_level": self.language_level if self.is_language_subject else None
        }
        
        if self.generated_questions:
            stats["test_questions_percentage"] = round((test_questions_count / len(self.generated_questions)) * 100, 1)
            stats["language_questions_percentage"] = round((language_questions_count / len(self.generated_questions)) * 100, 1)
        else:
            stats["test_questions_percentage"] = 0
            stats["language_questions_percentage"] = 0
        
        return stats


# Создаем глобальный экземпляр для тестирования
if __name__ == "__main__":
    # Тестирование базовой функциональности
    print("🧪 Тестирование PracticeManager с улучшенной языковой поддержкой...")
    
    # Создаем mock LLM для тестирования
    class MockLLM:
        def query(self, question, context, subject):
            return """Тестовый вопрос по английскому языку.

Выберите правильный перевод слова "дом":
A) house
B) home  
C) building
D) apartment

Правильный ответ: A) house"""
    
    # Тестируем менеджер практики
    pm = PracticeManager(MockLLM())
    
    # Тест с данными ученика
    pm.student_data = {
        'name': 'Тестовый ученик',
        'age': '12',
        'level': '5'
    }
    
    # Тест для языкового предмета
    print("\n📚 Тест для английского языка:")
    pm.initialize_practice_generation("Урок о базовых словах на английском: hello, goodbye, thank you", "английский язык")
    
    # Тест генерации вопроса
    question = pm.generate_single_language_question()
    print(f"📝 Сгенерированный языковой вопрос:\n{question}")
    
    # Тест статистики
    stats = pm.get_practice_stats()
    print(f"📊 Статистика практики: {stats}")
    
    # Сброс и тест для неязыкового предмета
    pm.reset()
    print("\n📚 Тест для математики:")
    pm.initialize_practice_generation("Урок о сложении и вычитании", "математика")
    
    question = pm.generate_single_question()
    print(f"📝 Сгенерированный вопрос по математике:\n{question}")
    
    stats = pm.get_practice_stats()
    print(f"📊 Статистика практики: {stats}")
    
    print("\n✅ Тестирование PracticeManager завершено!")
