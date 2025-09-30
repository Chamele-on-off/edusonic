import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
import time

class PracticeManager:
    def __init__(self, llm_integration):
        self.llm = llm_integration
        self.practice_dir = Path("materials/practice")
        self.current_lesson_context = ""
        self.current_subject = ""
        self.generated_questions = []  # История сгенерированных вопросов
        self.current_question_index = 0
        self.max_questions = 5
        
        # Создаем директорию если не существует
        self.practice_dir.mkdir(parents=True, exist_ok=True)

    def initialize_practice_generation(self, lesson_context: str, subject: str):
        """Инициализирует генерацию практических вопросов"""
        self.current_lesson_context = lesson_context
        self.current_subject = subject
        self.generated_questions = []
        self.current_question_index = 0
        print(f"Инициализирована генерация практики для предмета: {subject}")

    def generate_single_question(self) -> Optional[str]:
        """Генерирует один вопрос на основе контекста урока с разными типами вопросов"""
        try:
            # Определяем тип вопроса случайным образом
            question_types = [
                "open_ended",  # Открытый вопрос
                "multiple_choice",  # Множественный выбор
                "true_false",  # Верно/неверно
                "comparison"  # Сравнение
            ]
            question_type = random.choice(question_types)
            
            if question_type == "multiple_choice":
                prompt = f"""
                На основе учебного материала создай вопрос с множественным выбором (4 варианта ответа).
                
                КОНТЕКСТ УРОКА:
                {self.current_lesson_context[:1000]}
                
                ТРЕБОВАНИЯ:
                - Вопрос должен проверять понимание ключевых понятий
                - Должно быть 4 варианта ответа (A, B, C, D)
                - Только один правильный ответ
                - Варианты должны быть правдоподобными
                - Укажи правильный ответ в конце в формате [ПРАВИЛЬНЫЙ: X]
                
                ПРЕДМЕТ: {self.current_subject}
                
                Верни только вопрос и варианты ответов.
                """
            elif question_type == "true_false":
                prompt = f"""
                На основе учебного материала создай вопрос типа "Верно/Неверно".
                
                КОНТЕКСТ УРОКА:
                {self.current_lesson_context[:1000]}
                
                ТРЕБОВАНИЯ:
                - Утверждение должно быть четким и проверяемым
                - Укажи правильный ответ в конце [ПРАВИЛЬНЫЙ: ВЕРНО] или [ПРАВИЛЬНЫЙ: НЕВЕРНО]
                
                ПРЕДМЕТ: {self.current_subject}
                
                Верни только утверждение.
                """
            else:  # open_ended или comparison
                prompt = f"""
                На основе учебного материала сгенерируй ОДИН практический вопрос для проверки понимания.
                
                КОНТЕКСТ УРОКА:
                {self.current_lesson_context[:1000]}
                
                ТРЕБОВАНИЯ К ВОПРОСУ:
                - Вопрос должен проверять понимание ключевых понятий из материала
                - Вопрос должен быть четким и понятным
                - Вопрос должен требовать развернутого ответа
                - Вопрос должен быть адаптирован для учеников
                - Только один вопрос, без нумерации
                
                ПРЕДМЕТ: {self.current_subject}
                
                Верни только текст вопроса без дополнительных комментариев.
                """
            
            llm_response = self.llm._query_llm_api(
                prompt=prompt,
                context="",
                subject=self.current_subject,
                system_prompt="Ты — помощник учителя. Создавай качественные вопросы для проверки понимания материала.",
                max_tokens=300
            )
            
            if llm_response:
                question = self._clean_question_text(llm_response)
                print(f"Сгенерирован вопрос типа {question_type}: {question}")
                
                # Сохраняем в историю
                self.generated_questions.append({
                    "question": question,
                    "type": question_type,
                    "generated_at": time.time()
                })
                
                return question
                
            return None
            
        except Exception as e:
            print(f"Ошибка генерации вопроса: {e}")
            return None

    def evaluate_single_answer(self, student_answer: str, question: str) -> str:
        """Оценивает один ответ ученика и генерирует обратную связь"""
        try:
            # Генерируем эталонный ответ для этого вопроса
            correct_answer = self._generate_correct_answer(question)
            
            if not correct_answer:
                correct_answer = "Информация содержится в учебном материале."
            
            # Оцениваем ответ через LLM
            evaluation = self._evaluate_with_llm_context(question, student_answer, correct_answer)
            return evaluation if evaluation else self._get_fallback_feedback(student_answer, correct_answer)
            
        except Exception as e:
            print(f"Ошибка оценки ответа: {e}")
            return "Спасибо за ответ! Переходим к следующему вопросу."

    def _generate_correct_answer(self, question: str) -> Optional[str]:
        """Генерирует правильный ответ на вопрос через LLM"""
        try:
            prompt = f"""
            Дай точный и краткий ответ на вопрос на основе учебного материала.
            
            ВОПРОС: {question}
            
            КОНТЕКСТ УРОКА:
            {self.current_lesson_context[:800]}
            
            ТРЕБОВАНИЯ:
            - Ответ должен быть точным и соответствовать материалу
            - Ответ должен быть кратким (1-2 предложения)
            - Ответ должен быть понятным для ученика
            - Не добавляй дополнительные объяснения или комментарии
            
            Верни только ответ без лишних слов.
            """
            
            llm_response = self.llm._query_llm_api(
                prompt=prompt,
                context="",
                subject=self.current_subject,
                system_prompt="Ты — эксперт по предмету. Дай точный и краткий ответ на вопрос.",
                max_tokens=150
            )
            
            return llm_response.strip() if llm_response else None
            
        except Exception as e:
            print(f"Ошибка генерации правильного ответа: {e}")
            return None

    def _clean_question_text(self, text: str) -> str:
        """Очищает текст вопроса от лишних символов"""
        if not text:
            return ""
        
        # Удаляем нумерацию и маркеры
        text = re.sub(r'^\d+\.\s*', '', text.strip())
        text = re.sub(r'^[•\-]\s*', '', text)
        text = re.sub(r'^вопрос\s*\d*:*\s*', '', text, flags=re.IGNORECASE)
        
        # Удаляем кавычки и лишние пробелы
        text = re.sub(r'["«»]', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def _evaluate_with_llm_context(self, question: str, student_answer: str, correct_answer: str) -> str:
        """Оценивает ответ через LLM с учетом контекста урока"""
        try:
            prompt = f"""
            Оцени ответ ученика на вопрос и дай обратную связь.
            
            ВОПРОС: {question}
            ПРАВИЛЬНЫЙ ОТВЕТ: {correct_answer}
            ОТВЕТ УЧЕНИКА: {student_answer}
            
            КОНТЕКСТ УРОКА (для справки):
            {self.current_lesson_context[:500]}
            
            ВАЖНЫЕ ПРАВИЛА:
            - Обращайся к ученику на "ты" (неформальное обращение)
            - Будь добрым и поддерживающим
            - Не используй обращение "ученик"
            - Объясняй ошибки понятным языком
            - Хвали за правильные ответы
            
            Твоя задача - дать добрую и поддерживающую обратную связь:
            
            ЕСЛИ ОТВЕТ ПРАВИЛЬНЫЙ:
            - Похвали ученика конкретно ("Молодец!", "Отлично!", "Правильно!")
            - Подтверди правильность ответа
            - Скажи что-то ободряющее
            
            ЕСЛИ ОТВЕТ ЧАСТИЧНО ПРАВИЛЬНЫЙ:
            - Отметь что было правильно ("Хорошо, что ты упомянул...")
            - Вежливо укажи на ошибки или неточности
            - Дай правильный ответ с объяснением
            - Поддержи ученика ("Не расстраивайся, ты близок к правильному ответу!")
            
            ЕСЛИ ОТВЕТ НЕПРАВИЛЬНЫЙ:
            - Не ругай, а поддержи ученика ("Не волнуйся, ошибки - это нормально!")
            - Объясни почему ответ неверный
            - Дай правильный ответ понятным языком
            - Ободри ученика ("Попробуй еще раз, у тебя все получится!")
            
            Будь добрым и поддерживающим учителем! 
            Максимум 2-3 предложения. Отвечай на русском языке.
            """
            
            system_prompt = f"""Ты - опытный и добрый учитель по предмету {self.current_subject}. 
            Твоя задача - помогать ученикам учиться на ошибках, а не ругать их.
            Обращайся к ученику на "ты" в дружеской манере."""
            
            evaluation = self.llm._query_llm_api(
                prompt=prompt,
                context="",
                subject=self.current_subject,
                system_prompt=system_prompt,
                max_tokens=300
            )
            
            # Пост-обработка для исправления обращений
            if evaluation:
                evaluation = evaluation.replace("Ученик", "ты")
                evaluation = evaluation.replace("ученик", "ты")
                evaluation = re.sub(r'[Оо]н\s', 'ты ', evaluation)
            
            return evaluation if evaluation else self._get_fallback_feedback(student_answer, correct_answer)
            
        except Exception as e:
            print(f"Ошибка оценки через LLM: {e}")
            return self._get_fallback_feedback(student_answer, correct_answer)

    def _get_fallback_feedback(self, student_answer: str, correct_answer: str) -> str:
        """Fallback обратная связь когда LLM недоступен"""
        return "Спасибо за ответ! Теперь перейдем к следующему вопросу."

    def has_more_questions(self) -> bool:
        """Проверяет, можно ли генерировать еще вопросы"""
        return len(self.generated_questions) < self.max_questions

    def get_generated_questions_count(self) -> int:
        """Возвращает количество сгенерированных вопросов"""
        return len(self.generated_questions)

    def reset(self):
        """Сброс состояния менеджера практики"""
        self.current_lesson_context = ""
        self.current_subject = ""
        self.generated_questions = []
        self.current_question_index = 0

    # Старые методы для обратной совместимости
    def load_practice(self, lesson_id: str) -> bool:
        """Загружает практические задания (для обратной совместимости)"""
        print(f"Загрузка практики для урока: {lesson_id}")
        return False

    def generate_practice(self, lesson_text: str, subject: str) -> bool:
        """Генерирует практические задания (для обратной совместимости)"""
        print(f"Генерация практики для предмета: {subject}")
        self.initialize_practice_generation(lesson_text, subject)
        return True

    def get_current_question(self) -> Optional[Dict]:
        """Возвращает текущий вопрос (для обратной совместимости)"""
        if self.generated_questions:
            return {
                "question": self.generated_questions[-1]["question"],
                "answer": ""
            }
        return None

    def move_to_next_question(self) -> bool:
        """Переходит к следующему вопросу (для обратной совместимости)"""
        return self.has_more_questions()

    def evaluate_answer_with_context(self, student_answer: str, question: str, correct_answer: str, context: str = "") -> str:
        """Оценивает ответ с контекстом (для обратной совместимости)"""
        return self.evaluate_single_answer(student_answer, question)