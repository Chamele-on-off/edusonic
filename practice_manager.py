import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
import time
import threading
import queue

class PracticeManager:
    def __init__(self, llm_integration):
        self.llm = llm_integration
        self.practice_dir = Path("materials/practice")
        self.current_lesson_context = ""
        self.current_lesson_summary = ""
        self.current_subject = ""
        
        # ОЧЕРЕДИ ДЛЯ АСИНХРОННОЙ ГЕНЕРАЦИИ
        self.question_queue = queue.Queue()  # Очередь готовых вопросов
        self.generated_questions = []        # История всех вопросов
        self.current_question_index = 0
        self.max_questions = 5  # ЖЕСТКИЙ ЛИМИТ 5 ВОПРОСОВ
        
        # ФЛАГИ УПРАВЛЕНИЯ АСИНХРОННОЙ ГЕНЕРАЦИЕЙ
        self.generation_thread = None
        self.stop_generation = False
        self.generation_active = False
        
        self.practice_dir.mkdir(parents=True, exist_ok=True)

    def initialize_practice_generation(self, lesson_context: str, subject: str):
        """Инициализирует практику и ЗАРАНЕЕ начинает генерацию вопросов"""
        self.current_lesson_context = lesson_context
        self.current_subject = subject
        self.current_lesson_summary = self._generate_lesson_summary(lesson_context)
        self.generated_questions = []
        self.current_question_index = 0
        
        # Очищаем очередь от предыдущих вопросов
        while not self.question_queue.empty():
            try:
                self.question_queue.get_nowait()
            except queue.Empty:
                break
        
        print(f"🎯 Инициализирована генерация практики для предмета: {subject}")
        print(f"📝 Максимальное количество вопросов: {self.max_questions}")
        
        # ЗАПУСКАЕМ АСИНХРОННУЮ ГЕНЕРАЦИЮ ВОПРОСОВ СРАЗУ
        self._start_async_generation()
        
        # Генерируем ПЕРВЫЙ вопрос СИНХРОННО для немедленного старта
        first_question = self.generate_single_question()
        if first_question:
            self.question_queue.put(first_question)
            print(f"✅ Первый вопрос готов: {first_question[:80]}...")

    def _start_async_generation(self):
        """Запускает фоновую генерацию вопросов"""
        if self.generation_active:
            return
            
        self.stop_generation = False
        self.generation_active = True
        
        def generate_questions_worker():
            print("🔄 Фоновая генерация вопросов запущена...")
            
            while (not self.stop_generation and 
                   self.generation_active and 
                   len(self.generated_questions) < self.max_questions - 1):  # -1 потому что первый уже сгенерирован
                
                try:
                    # Проверяем лимит перед генерацией
                    if len(self.generated_questions) >= self.max_questions:
                        print("🏁 Достигнут лимит вопросов в фоновой генерации")
                        break
                    
                    # Генерируем следующий вопрос
                    next_question = self.generate_single_question()
                    
                    if next_question:
                        self.question_queue.put(next_question)
                        print(f"✅ Фоново сгенерирован вопрос {len(self.generated_questions)}/{self.max_questions}: {next_question[:80]}...")
                    
                    # Небольшая пауза между генерацией
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"❌ Ошибка в фоновой генерации: {e}")
                    time.sleep(2)  # Пауза при ошибке
            
            print("🏁 Фоновая генерация вопросов завершена")
            self.generation_active = False
        
        # Запускаем в отдельном потоке
        self.generation_thread = threading.Thread(target=generate_questions_worker, daemon=True)
        self.generation_thread.start()

    def get_next_question(self, timeout: float = 10.0) -> Optional[str]:
        """Получает следующий вопрос из очереди (с ожиданием если нужно)"""
        try:
            # ПРОВЕРЯЕМ ЛИМИТ ВОПРОСОВ
            if len(self.generated_questions) >= self.max_questions:
                print(f"🏁 Достигнут лимит вопросов: {len(self.generated_questions)}/{self.max_questions}")
                return None
            
            # Пытаемся взять вопрос из очереди без ожидания
            try:
                question = self.question_queue.get_nowait()
                print(f"✅ Вопрос взят из очереди (в очереди еще: {self.question_queue.qsize()})")
                return question
            except queue.Empty:
                pass
            
            # Если очередь пуста, пробуем сгенерировать СИНХРОННО
            print("⚠️ Очередь вопросов пуста, синхронная генерация...")
            question = self.generate_single_question()
            
            if question:
                print(f"✅ Синхронно сгенерирован вопрос: {question[:80]}...")
                return question
            else:
                # Если синхронная генерация не удалась, используем fallback
                fallback = self._get_fallback_question(ensure_unique=True)
                print(f"🔄 Использован fallback вопрос: {fallback[:80]}...")
                return fallback
                
        except Exception as e:
            print(f"❌ Ошибка получения следующего вопроса: {e}")
            fallback = self._get_fallback_question(ensure_unique=True)
            print(f"🔄 Использован fallback после ошибки: {fallback[:80]}...")
            return fallback

    def generate_single_question(self) -> Optional[str]:
        """Генерирует один вопрос (синхронно)"""
        try:
            # ПРОВЕРЯЕМ ЛИМИТ ВОПРОСОВ
            if len(self.generated_questions) >= self.max_questions:
                print(f"🏁 Достигнут лимит вопросов: {len(self.generated_questions)}/{self.max_questions}")
                return None
            
            # УЛУЧШЕННЫЙ промпт с историей вопросов
            previous_questions = self._get_previous_questions_text()
            
            prompt = f"""
            Создай ОДИН УНИКАЛЬНЫЙ учебный вопрос для проверки понимания темы.
            
            ПРЕДМЕТ: {self.current_subject}
            
            КРАТКОЕ СОДЕРЖАНИЕ УРОКА:
            {self.current_lesson_summary}
            
            УЖЕ ЗАДАННЫЕ ВОПРОСЫ (НЕ ПОВТОРЯЙ ИХ!):
            {previous_questions}
            
            ТРЕБОВАНИЯ:
            - Вопрос ДОЛЖЕН быть УНИКАЛЬНЫМ и не похожим на уже заданные
            - Проверяй понимание РАЗНЫХ аспектов материала
            - Вопрос должен требовать развернутого ответа
            - Будь конкретным и четким
            
            Верни ТОЛЬКО текст одного вопроса без нумерации и лишних слов.
            """
            
            # УВЕЛИЧИВАЕМ ТАЙМАУТ для LLM запроса
            llm_response = self.llm.query(
                question=prompt,
                context="",
                subject=self.current_subject
            )
            
            if llm_response and not llm_response.startswith("Спасибо за вопрос!"):
                question = self._clean_question_text(llm_response)
                
                if question and len(question.strip()) > 10 and self._is_question_unique(question):
                    # Сохраняем в историю
                    self.generated_questions.append({
                        "question": question,
                        "generated_at": time.time(),
                        "type": self._detect_question_type(question)
                    })
                    return question
            
            # Fallback если не удалось сгенерировать
            return self._get_fallback_question(ensure_unique=True)
                
        except Exception as e:
            print(f"❌ Ошибка генерации вопроса: {e}")
            return self._get_fallback_question(ensure_unique=True)

    def evaluate_and_continue(self, student_answer: str, current_question: str) -> Tuple[str, Optional[str]]:
        """Оценивает ответ и возвращает feedback + следующий вопрос"""
        try:
            # 1. Сначала оцениваем ответ (это быстро)
            feedback = self.evaluate_single_answer(student_answer, current_question)
            
            # 2. Параллельно получаем следующий вопрос
            next_question = self.get_next_question()
            
            return feedback, next_question
            
        except Exception as e:
            print(f"❌ Ошибка в evaluate_and_continue: {e}")
            # ИСПРАВЛЕННЫЙ FALLBACK ДЛЯ ПРАКТИКИ
            feedback = "Спасибо за ответ! Переходим к следующему вопросу."
            next_question = self._get_fallback_question(ensure_unique=True)
            return feedback, next_question

    def evaluate_single_answer(self, student_answer: str, question: str) -> str:
        """Оценивает ответ ученика (синхронно) с ИСПРАВЛЕННЫМИ FALLBACK"""
        try:
            if not student_answer or len(student_answer.strip()) < 2:
                return "Ответ слишком короткий. Пожалуйста, попробуйте ответить более развернуто."
            
            command_words = ['продолжай', 'дальше', 'следующий', 'стоп', 'останови']
            if any(cmd in student_answer.lower() for cmd in command_words):
                return "Это похоже на команду. Пожалуйста, дайте ответ на вопрос."
            
            correct_answer = self._generate_correct_answer(question)
            if not correct_answer:
                correct_answer = "Информация содержится в учебном материале."
            
            evaluation = self._evaluate_with_llm_context(question, student_answer, correct_answer)
            
            # ИСПРАВЛЕНИЕ: Заменяем неподходящие fallback ответы
            if not evaluation or any(phrase in evaluation for phrase in [
                "Хороший вопрос! Давайте разберем эту тему подробнее",
                "Мне нужно немного времени подумать",
                "Спасибо за вопрос! Я подумаю над ответом"
            ]):
                evaluation = f"Спасибо за ответ! Правильный ответ: {correct_answer}"
            
            return evaluation
            
        except Exception as e:
            print(f"❌ Ошибка оценки ответа: {e}")
            # ИСПРАВЛЕННЫЙ FALLBACK
            return "Спасибо за ответ! Переходим к следующему вопросу."

    def _generate_correct_answer(self, question: str) -> Optional[str]:
        try:
            prompt = f"""
            Дай точный и краткий ответ на вопрос.
            
            ВОПРОС: {question}
            КОНТЕКСТ: {self.current_lesson_summary}
            
            Требования:
            - Ответ должен быть точным и соответствовать материалу
            - Ответ должен быть кратким (1-2 предложения)
            - Без лишних объяснений
            
            Верни только ответ.
            """
            
            llm_response = self.llm.query(
                question=prompt,
                context="",
                subject=self.current_subject
            )
            return llm_response.strip() if llm_response else None
            
        except Exception as e:
            print(f"❌ Ошибка генерации правильного ответа: {e}")
            return None

    def _evaluate_with_llm_context(self, question: str, student_answer: str, correct_answer: str) -> str:
        try:
            prompt = f"""
            Оцени ответ ученика на вопрос и дай обратную связь.
            
            ВОПРОС: {question}
            ПРАВИЛЬНЫЙ ОТВЕТ: {correct_answer}
            ОТВЕТ УЧЕНИКА: {student_answer}
            
            Контекст урока: {self.current_lesson_summary[:300]}
            
            Дай добрую и поддерживающую обратную связь (2-3 предложения).
            Обращайся к ученику на "ты".
            
            НЕ ИСПОЛЬЗУЙ фразы:
            - "Хороший вопрос! Давайте разберем эту тему подробнее"
            - "Мне нужно немного времени подумать" 
            - "Спасибо за вопрос! Я подумаю над ответом"
            - "Давайте разберем эту тему подробнее"
            
            Вместо этого дай конкретную обратную связь по ответу.
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
            print(f"❌ Ошибка оценки через LLM: {e}")
            return f"Спасибо за ответ! Правильный ответ: {correct_answer}"

    def stop_async_generation(self):
        """Останавливает фоновую генерацию"""
        self.stop_generation = True
        self.generation_active = False
        if self.generation_thread and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=2.0)

    def _get_previous_questions_text(self) -> str:
        if not self.generated_questions:
            return "Вопросов еще не было. Это первый вопрос."
        
        questions_text = "Уже заданные вопросы:\n"
        for i, q_data in enumerate(self.generated_questions, 1):
            questions_text += f"{i}. {q_data['question']}\n"
        return questions_text

    def _is_question_unique(self, new_question: str, similarity_threshold: float = 0.7) -> bool:
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
        if not text:
            return ""
        text = re.sub(r'^\d+\.\s*', '', text.strip())
        text = re.sub(r'^[•\-]\s*', '', text)
        text = re.sub(r'^вопрос\s*\d*:*\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'["«»]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _generate_lesson_summary(self, lesson_context: str) -> str:
        try:
            prompt = f"""
            Создай КРАТКОЕ содержание этого урока для практических вопросов.
            Выдели только ключевые понятия и основные идеи.
            
            ТЕКСТ: {lesson_context[:1500]}
            
            Верни только краткое содержание (максимум 300 слов).
            """
            
            summary = self.llm.query(
                question=prompt,
                context="",
                subject=self.current_subject
            )
            return summary if summary and len(summary) > 50 else lesson_context[:500] + "..."
        except Exception as e:
            print(f"❌ Ошибка генерации краткого содержания: {e}")
            return lesson_context[:500] + "..."

    def _get_fallback_question(self, ensure_unique: bool = False) -> str:
        # БАЗОВЫЕ ВОПРОСЫ ДЛЯ РАЗНЫХ ПРЕДМЕТОВ
        subject_questions = {
            "обществознание": [
                "Что такое общество и каковы его основные элементы?",
                "Объясни понятие 'социальный институт' и приведи примеры.",
                "В чем разница между формальными и неформальными социальными нормами?",
                "Какие функции выполняет государство в современном обществе?",
                "Что такое гражданское общество и как оно взаимодействует с государством?"
            ],
            "математика": [
                "Объясни основную концепцию, которую мы только что изучили.",
                "Как применить изученный метод на практике?",
                "В чем особенность этого математического подхода?",
                "Какие существуют альтернативные способы решения этой задачи?",
                "Почему этот математический принцип важен для понимания?"
            ],
            "история": [
                "Каковы были ключевые события изученного периода?",
                "Как повлияли эти исторические события на современность?",
                "В чем заключались основные причины исторических процессов, которые мы изучали?",
                "Охарактеризуй ключевых исторических личностей этого периода.",
                "Какие исторические закономерности можно проследить в изученном материале?"
            ],
            "физика": [
                "Как работает основной физический принцип, который мы рассмотрели?",
                "Объясни физический смысл изученного явления.",
                "Где в повседневной жизни мы встречаемся с этим физическим законом?",
                "Какие практические применения имеет это физическое открытие?",
                "В чем заключается научная важность изученного физического явления?"
            ],
            "химия": [
                "Опиши основные химические процессы из урока.",
                "В чем особенность химических свойств изученных элементов?",
                "Как протекает химическая реакция, которую мы изучали?",
                "Какое практическое значение имеют эти химические процессы?",
                "Объясни взаимосвязь между строением и свойствами химических веществ."
            ],
            "биология": [
                "Каковы основные биологические процессы, которые мы изучили?",
                "Опиши строение и функции биологических структур из урока.",
                "Как взаимодействуют различные биологические системы?",
                "В чем биологическое значение изученных процессов?",
                "Какие адаптации организмов мы рассмотрели и в чем их смысл?"
            ],
            "литература": [
                "В чем основная идея или тема произведения, которое мы обсуждали?",
                "Охарактеризуй главных героев изученного произведения.",
                "Как автор раскрывает основные темы в произведении?",
                "В чем художественное своеобразие этого литературного произведения?",
                "Какие нравственные проблемы поднимает автор в произведении?"
            ],
            "русский язык": [
                "Объясни основное грамматическое правило, которое мы изучили.",
                "В чем особенности применения этого правила на практике?",
                "Какие исключения существуют из изученного правила?",
                "Как правильно использовать изученные языковые конструкции?",
                "Почему это грамматическое правило важно для правильной речи?"
            ]
        }
        
        # Базовый набор вопросов для любого предмета
        general_questions = [
            "Объясни основную идею изученного материала.",
            "В чем заключается главная мысль этого урока?",
            "Какие ключевые понятия мы сегодня изучили?",
            "Как можно применить эти знания на практике?",
            "Почему эта тема важна для понимания предмета?"
        ]
        
        # Получаем вопросы для текущего предмета или общие вопросы
        questions = subject_questions.get(self.current_subject, general_questions)
        
        if ensure_unique and self.generated_questions:
            # Ищем вопрос, которого еще не было
            existing_questions = [q["question"] for q in self.generated_questions]
            for question in questions:
                if question not in existing_questions:
                    return question
            
            # Если все вопросы уже использованы, берем из общего списка
            for general_q in general_questions:
                if general_q not in existing_questions:
                    return general_q
        
        # Возвращаем случайный вопрос
        import random
        return random.choice(questions)

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
        self.generated_questions = []
        self.current_question_index = 0
        while not self.question_queue.empty():
            try:
                self.question_queue.get_nowait()
            except queue.Empty:
                break
        print("🔄 Менеджер практики сброшен")

    def get_current_question(self) -> Optional[str]:
        """Возвращает текущий вопрос"""
        if self.generated_questions:
            return self.generated_questions[-1]["question"]
        return None

    def get_practice_stats(self) -> Dict:
        """Возвращает статистику по практике"""
        question_types = {}
        for q in self.generated_questions:
            q_type = q.get("type", "unknown")
            question_types[q_type] = question_types.get(q_type, 0) + 1
        
        return {
            "total_questions": len(self.generated_questions),
            "max_questions": self.max_questions,
            "questions_in_queue": self.question_queue.qsize(),
            "generation_active": self.generation_active,
            "current_subject": self.current_subject,
            "has_more_questions": self.has_more_questions(),
            "question_types": question_types,
            "lesson_summary_length": len(self.current_lesson_summary)
        }
