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
        
        # НОВОЕ: используем готовое краткое содержание из lesson.py
        self.lesson_summary = ""
        
        # Создаем директорию если не существует
        self.practice_dir.mkdir(parents=True, exist_ok=True)

    def initialize_practice_generation(self, lesson_data: Dict, subject: str):
        """Инициализирует генерацию практических вопросов с готовым кратким содержанием"""
        self.current_lesson_context = lesson_data.get("original_content", "")
        self.current_subject = subject
        
        # НОВОЕ: используем готовое краткое содержание из lesson.py
        self.lesson_summary = lesson_data.get("summary", "")
        
        self.generated_questions = []
        self.current_question_index = 0
        
        print(f"🎯 Инициализирована генерация практики для предмета: {subject}")
        print(f"📝 Используется готовое краткое содержание: {len(self.lesson_summary)} символов")

    def generate_single_question(self) -> Optional[str]:
        """Генерирует один вопрос на основе готового краткого содержания"""
        try:
            # Проверяем лимит вопросов
            if len(self.generated_questions) >= self.max_questions:
                print("🏁 Достигнут лимит вопросов")
                return None
            
            print(f"🔄 Генерация вопроса {len(self.generated_questions) + 1}/{self.max_questions}...")
            
            # УЛУЧШЕННЫЙ ПРОМПТ с готовым кратким содержанием
            previous_questions_text = self._get_previous_questions_text()
            
            prompt = f"""
            Создай один учебный вопрос для проверки понимания темы: {self.current_subject}
            
            КРАТКОЕ СОДЕРЖАНИЕ УРОКА (уже готовое): {self.lesson_summary}
            
            УЖЕ ЗАДАННЫЕ ВОПРОСЫ (НЕ ПОВТОРЯЙ ИХ):
            {previous_questions_text}
            
            ТРЕБОВАНИЯ:
            - Один четкий и уникальный вопрос
            - НЕ повторяй вопросы из списка выше
            - Проверяет понимание РАЗНЫХ аспектов материала
            - Требует развернутого ответа
            - Будь разнообразным: используй разные типы вопросов
            - Без нумерации и лишних символов
            
            Верни только текст вопроса.
            """
            
            try:
                llm_response = self.llm.query(
                    question=prompt,
                    context="",
                    subject=self.current_subject
                )
                
                if llm_response and not llm_response.startswith("Спасибо за вопрос!"):
                    # Очищаем ответ
                    question = self._clean_question_text(llm_response)
                    
                    # Проверяем уникальность
                    if self._is_question_unique(question) and len(question.strip()) > 10:
                        print(f"✅ Уникальный вопрос сгенерирован: {question[:100]}...")
                        
                        # Сохраняем в историю
                        self.generated_questions.append({
                            "question": question,
                            "generated_at": time.time()
                        })
                        
                        return question
                    else:
                        print("⚠️ Вопрос слишком похож на предыдущие, пробуем снова...")
                        # Рекурсивно пробуем сгенерировать другой вопрос (максимум 2 попытки)
                        if len(self.generated_questions) < 2:
                            return self.generate_single_question()
                        else:
                            return self._get_diverse_fallback_question()
                    
            except Exception as e:
                print(f"❌ Ошибка генерации вопроса: {e}")
            
            # Fallback если не удалось сгенерировать
            fallback_question = self._get_diverse_fallback_question()
            if fallback_question:
                self.generated_questions.append({
                    "question": fallback_question,
                    "generated_at": time.time()
                })
                return fallback_question
                
            return None
                
        except Exception as e:
            print(f"❌ Критическая ошибка генерации вопроса: {e}")
            return self._get_diverse_fallback_question()

    def _get_previous_questions_text(self) -> str:
        """Возвращает текст предыдущих вопросов для промпта"""
        if not self.generated_questions:
            return "Пока нет заданных вопросов."
        
        questions_text = ""
        for i, q in enumerate(self.generated_questions[-3:]):  # Берем последние 3 вопроса
            questions_text += f"{i+1}. {q['question']}\n"
        
        return questions_text

    def _is_question_unique(self, new_question: str, similarity_threshold: float = 0.7) -> bool:
        """Проверяет, что новый вопрос не слишком похож на предыдущие"""
        if not self.generated_questions:
            return True
        
        new_question_lower = new_question.lower()
        
        for existing_q in self.generated_questions:
            existing_question_lower = existing_q['question'].lower()
            
            # Простая проверка по ключевым словам
            new_words = set(new_question_lower.split())
            existing_words = set(existing_question_lower.split())
            
            common_words = new_words.intersection(existing_words)
            if len(common_words) / max(len(new_words), 1) > 0.6:  # Если больше 60% общих слов
                return False
            
            # Более точная проверка схожести
            similarity = SequenceMatcher(None, new_question_lower, existing_question_lower).ratio()
            if similarity > similarity_threshold:
                return False
        
        return True

    def _get_diverse_fallback_question(self) -> str:
        """Fallback вопросы с разнообразием"""
        # Разные типы вопросов для разных индексов
        question_templates = [
            "Объясни основную концепцию, которую мы изучили, своими словами.",
            "Приведи практический пример применения этого материала.",
            "В чем заключается важность изученной темы?",
            "Сравни этот концепт с чем-то знакомым из повседневной жизни.",
            "Какие основные выводы можно сделать из изученного материала?",
            "Как бы ты объяснил эту тему другу, который ее не понимает?",
            "Что было самым интересным в этом материале и почему?",
            "Какие вопросы у тебя остались после изучения этой темы?",
            "Как этот материал связан с другими темами, которые мы изучали?",
            "Что бы ты хотел узнать дополнительно по этой теме?"
        ]
        
        # Выбираем вопрос на основе текущего индекса для разнообразия
        question_index = len(self.generated_questions) % len(question_templates)
        base_question = question_templates[question_index]
        
        # Добавляем контекст предмета
        subject_specific = {
            "обществознание": " в контексте общества",
            "математика": " в математическом контексте", 
            "история": " с исторической точки зрения",
            "физика": " с точки зрения физических законов",
            "химия": " в химическом контексте",
            "биология": " с биологической точки зрения",
            "литература": " на литературном примере",
            "русский язык": " в контексте русского языка"
        }
        
        subject_context = subject_specific.get(self.current_subject, "")
        return base_question + subject_context

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
            # ИСПОЛЬЗУЕМ КРАТКОЕ СОДЕРЖАНИЕ для скорости
            prompt = f"""
            Дай точный и краткий ответ на вопрос на основе учебного материала.
            
            ВОПРОС: {question}
            
            КРАТКОЕ СОДЕРЖАНИЕ УРОКА:
            {self.lesson_summary}
            
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
            # ИСПОЛЬЗУЕМ КРАТКОЕ СОДЕРЖАНИЕ для скорости
            prompt = f"""
            Оцени ответ ученика на вопрос и дай обратную связь.
            
            ВОПРОС: {question}
            ПРАВИЛЬНЫЙ ОТВЕТ: {correct_answer}
            ОТВЕТ УЧЕНИКА: {student_answer}
            
            КРАТКИЙ КОНТЕКСТ УРОКА (для справки):
            {self.lesson_summary}
            
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
        self.lesson_summary = ""  # НОВОЕ: сбрасываем краткое содержание
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
            "has_more_questions": self.has_more_questions(),
            "lesson_summary_length": len(self.lesson_summary)  # НОВОЕ: для диагностики
        }

# Создаем глобальный экземпляр для использования в других модулях
practice_manager = None

def get_practice_manager(llm_integration=None):
    """Возвращает глобальный экземпляр PracticeManager"""
    global practice_manager
    if practice_manager is None and llm_integration is not None:
        practice_manager = PracticeManager(llm_integration)
    return practice_manager

if __name__ == "__main__":
    # Тестирование модуля
    from llm import LLMIntegration
    
    print("🔧 Тестирование PracticeManager...")
    
    llm = LLMIntegration()
    pm = PracticeManager(llm)
    
    # Тестовые данные
    test_lesson_data = {
        "original_content": "Математика изучает числа, формы и пространственные отношения. Основные разделы: арифметика, алгебра, геометрия, математический анализ. Арифметика изучает числа и простые операции. Алгебра изучает уравнения и переменные. Геометрия изучает формы и пространство.",
        "summary": "Математика: числа, формы, пространство. Разделы: арифметика, алгебра, геометрия."
    }
    
    pm.initialize_practice_generation(test_lesson_data, "математика")
    
    # Генерация нескольких вопросов
    for i in range(3):
        question = pm.generate_single_question()
        if question:
            print(f"❓ Вопрос {i+1}: {question}")
        else:
            print("❌ Не удалось сгенерировать вопрос")
    
    print(f"📊 Статистика: {pm.get_practice_stats()}")
    print("🎉 Тестирование завершено!")