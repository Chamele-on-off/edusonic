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
        self.generated_questions = []
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
        print(f"🎯 Инициализирована генерация практики для предмета: {subject}")

    def generate_single_question(self) -> Optional[str]:
        """Генерирует один вопрос на основе контекста урока"""
        try:
            # Проверяем лимит вопросов
            if len(self.generated_questions) >= self.max_questions:
                print("🏁 Достигнут лимит вопросов")
                return None
            
            print(f"🔄 Генерация вопроса {len(self.generated_questions) + 1}/{self.max_questions}...")
            
            # УПРОЩЕННЫЙ промпт для надежности
            prompt = f"""
            Создай один учебный вопрос для проверки понимания темы: {self.current_subject}
            
            Контекст: {self.current_lesson_context[:800]}
            
            Требования:
            - Один четкий вопрос
            - Проверяет понимание материала  
            - Требует развернутого ответа
            - Без нумерации и лишних символов
            
            Верни только текст вопроса.
            """
            
            # УПРОЩЕННАЯ логика с одной попыткой
            try:
                llm_response = self.llm.query(
                    question=prompt,
                    context="",
                    subject=self.current_subject
                )
                
                if llm_response and not llm_response.startswith("Спасибо за вопрос!"):
                    # Очищаем ответ
                    question = self._clean_question_text(llm_response)
                    
                    if question and len(question.strip()) > 10:
                        print(f"✅ Вопрос сгенерирован: {question[:100]}...")
                        
                        # Сохраняем в историю
                        self.generated_questions.append({
                            "question": question,
                            "generated_at": time.time()
                        })
                        
                        return question
                    
            except Exception as e:
                print(f"❌ Ошибка генерации вопроса: {e}")
            
            # Fallback если не удалось сгенерировать
            fallback_question = self._get_fallback_question()
            if fallback_question:
                self.generated_questions.append({
                    "question": fallback_question,
                    "generated_at": time.time()
                })
                return fallback_question
                
            return None
                
        except Exception as e:
            print(f"❌ Критическая ошибка генерации вопроса: {e}")
            return self._get_fallback_question()

    def evaluate_single_answer(self, student_answer: str, question: str) -> str:
        """Оценивает один ответ ученика и генерирует обратную связь"""
        try:
            # Проверяем валидность ответа ученика
            if not student_answer or len(student_answer.strip()) < 2:
                return "Ответ слишком короткий. Пожалуйста, попробуйте ответить более развернуто."
            
            # Проверяем, не является ли ответ командой
            command_words = ['продолжай', 'дальше', 'следующий', 'стоп', 'останови']
            if any(cmd in student_answer.lower() for cmd in command_words):
                return "Это похоже на команду. Пожалуйста, дайте ответ на вопрос."
            
            # Генерируем эталонный ответ для этого вопроса
            correct_answer = self._generate_correct_answer(question)
            
            if not correct_answer:
                correct_answer = "Информация содержится в учебном материале."
            
            # УПРОЩЕННАЯ оценка через LLM
            evaluation = self._evaluate_with_llm_context(question, student_answer, correct_answer)
            return evaluation if evaluation else self._get_fallback_feedback(student_answer, correct_answer)
            
        except Exception as e:
            print(f"❌ Ошибка оценки ответа: {e}")
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
            
            llm_response = self.llm.query(
                question=prompt,
                context="",
                subject=self.current_subject
            )
            
            return llm_response.strip() if llm_response else None
            
        except Exception as e:
            print(f"❌ Ошибка генерации правильного ответа: {e}")
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
            
            ВАЖНОЕ ПРАВИЛО: Всегда обращайся к ученику на "ты".
            
            Твоя задача - дать добрую и поддерживающую обратную связь:
            
            ЕСЛИ ОТВЕТ ПРАВИЛЬНЫЙ:
            - Похвали ученика конкретно
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
            """
            
            evaluation = self.llm.query(
                question=prompt,
                context="",
                subject=self.current_subject
            )
            
            return evaluation if evaluation else self._get_fallback_feedback(student_answer, correct_answer)
            
        except Exception as e:
            print(f"❌ Ошибка оценки через LLM: {e}")
            return self._get_fallback_feedback(student_answer, correct_answer)

    def _get_fallback_question(self) -> str:
        """Fallback вопрос когда не удается сгенерировать через LLM"""
        fallback_questions = {
            "обществознание": "Что такое общество и каковы его основные элементы?",
            "математика": "Объясни основную концепцию, которую мы только что изучили.",
            "история": "Каковы были ключевые события или личности в изученном периоде?",
            "физика": "Как работает основной принцип, который мы рассмотрели?",
            "химия": "Опиши основные химические процессы или элементы из урока.",
            "биология": "Каковы основные биологические процессы или структуры, которые мы изучили?",
            "литература": "В чем основная идея или тема произведения, которое мы обсуждали?",
            "русский язык": "Объясни основное грамматическое правило, которое мы изучили."
        }
        
        return fallback_questions.get(self.current_subject, "Расскажи основную идею изученного материала.")

    def _get_fallback_feedback(self, student_answer: str, correct_answer: str) -> str:
        """Fallback обратная связь когда LLM недоступен"""
        return f"Спасибо за ответ! Правильный ответ: {correct_answer}"

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
        print("🔄 Менеджер практики сброшен")

    def get_current_question(self) -> Optional[str]:
        """Возвращает текущий вопрос"""
        if self.generated_questions:
            return self.generated_questions[-1]["question"]
        return None

    def get_practice_stats(self) -> Dict:
        """Возвращает статистику по практике"""
        return {
            "total_questions": len(self.generated_questions),
            "current_subject": self.current_subject,
            "max_questions": self.max_questions,
            "has_more_questions": self.has_more_questions()
        }