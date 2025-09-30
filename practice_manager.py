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
        """Генерирует один вопрос на основе контекста урока"""
        try:
            # Формируем промпт для генерации одного вопроса
            prompt = f"""
            На основе учебного материала сгенерируй ОДИН практический вопрос для проверки понимания.
            
            КОНТЕКСТ УРОКА:
            {self.current_lesson_context[:1000]}
            
            ТРЕБОВАНИЯ К ВОПРОСУ:
            - Вопрос должен проверять понимание ключевых понятий из материала
            - Вопрос должен быть четким и понятным
            - Вопрос должен требовать развернутого ответа (не просто "да/нет")
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
                max_tokens=200
            )
            
            if llm_response:
                # Очищаем ответ от лишних символов
                question = self._clean_question_text(llm_response)
                print(f"Сгенерирован вопрос: {question}")
                
                # Сохраняем в историю
                self.generated_questions.append({
                    "question": question,
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
            
            ВАЖНОЕ ПРАВИЛО: Всегда обращайся к ученику на "ты", а не "ученик" или "он".
            Например: "Ты близко подошел к правильному ответу" вместо "Ученик близко подошел"
            
            Твоя задача - дать добрую и поддерживающую обратную связь:
            
            ЕСЛИ ОТВЕТ ПРАВИЛЬНЫЙ:
            - Похвали ученика конкретно, обращаясь на "ты"
            - Подтверди правильность ответа
            - Скажи что-то ободряющее
            
            ЕСЛИ ОТВЕТ ЧАСТИЧНО ПРАВИЛЬНЫЙ:
            - Отметь что было правильно
            - Вежливо укажи на ошибки или неточности
            - Дай правильный ответ с объяснением
            
            ЕСЛИ ОТВЕТ НЕПРАВИЛЬНЫЙ:
            - Не ругай, а поддержи ученика
            - Объясни почему ответ неверный
            - Дай правильный ответ понятным языком
            
            Будь добрым и поддерживающим учителем! 
            Максимум 2-3 предложения. Отвечай на русском языке.
            Всегда обращайся к ученику на "ты".
            """
            
            system_prompt = f"""Ты - опытный и добрый учитель по предмету {self.current_subject}. 
            Твоя задача - помогать ученикам учиться на ошибках, а не ругать их.
            Всегда обращайся к ученику на "ты"."""
            
            evaluation = self.llm._query_llm_api(
                prompt=prompt,
                context="",
                subject=self.current_subject,
                system_prompt=system_prompt,
                max_tokens=300
            )
            
            # ДОПОЛНИТЕЛЬНАЯ ОБРАБОТКА: заменяем "ученик" на "ты"
            if evaluation:
                evaluation = self._fix_student_addressing(evaluation)
            
            return evaluation if evaluation else self._get_fallback_feedback(student_answer, correct_answer)
            
        except Exception as e:
            print(f"Ошибка оценки через LLM: {e}")
            return self._get_fallback_feedback(student_answer, correct_answer)

    def _fix_student_addressing(self, text: str) -> str:
        """Исправляет обращение к ученику с 'ученик' на 'ты'"""
        if not text:
            return text
        
        # Заменяем различные формы обращения
        replacements = {
            'ученик': 'ты',
            'ученица': 'ты', 
            'ученики': 'вы',
            'ученикам': 'вам',
            'учеников': 'вас',
            'он': 'ты',
            'она': 'ты',
            'его': 'твой',
            'её': 'твой',
            'ему': 'тебе',
            'ей': 'тебе'
        }
        
        result = text
        for old, new in replacements.items():
            # Заменяем с учетом регистра
            result = re.sub(r'\b' + re.escape(old) + r'\b', new, result, flags=re.IGNORECASE)
        
        return result

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

    def save_practice_to_txt(self, lesson_id: str, practice_data: dict):
        """Сохраняет практические задания в TXT файл"""
        try:
            txt_filename = f"{lesson_id}_practice.txt"
            txt_path = self.practice_dir / txt_filename
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"ПРАКТИЧЕСКИЕ ЗАДАНИЯ: {lesson_id}\n")
                f.write("=" * 50 + "\n\n")
                
                questions = practice_data.get('questions', [])
                for i, question_data in enumerate(questions, 1):
                    f.write(f"ВОПРОС {i}: {question_data.get('question', '')}\n\n")
                    
                    if 'correct_answer' in question_data:
                        f.write(f"Правильный ответ: {question_data['correct_answer']}\n")
                    
                    if 'explanation' in question_data and question_data['explanation']:
                        f.write(f"Объяснение: {question_data['explanation']}\n")
                    
                    f.write("\n" + "-" * 30 + "\n\n")
            
            print(f"✅ Практика сохранена в TXT: {txt_path}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка сохранения практики в TXT: {e}")
            return False

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
