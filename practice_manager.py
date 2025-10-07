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
        print(f"🎯 Инициализирована генерация практики для предмета: {subject}")

    def generate_single_question(self) -> Optional[str]:
        """Генерирует один вопрос на основе контекста урока"""
        try:
            # Проверяем лимит вопросов
            if len(self.generated_questions) >= self.max_questions:
                print("🏁 Достигнут лимит вопросов")
                return None
            
            # УПРОЩЕННЫЙ промпт для надежности
            prompt = f"""
            Создай один учебный вопрос для проверки понимания темы: {self.current_subject}
            
            Контекст: {self.current_lesson_context[:500]}
            
            Требования:
            - Один четкий вопрос
            - Проверяет понимание материала  
            - Требует развернутого ответа
            - Без нумерации и лишних символов
            
            Верни только текст вопроса.
            """
            
            # СИНХРОННЫЙ запрос к LLM
            llm_response = self.llm.query(
                question=prompt,
                context="",
                subject=self.current_subject
            )
            
            if llm_response and not llm_response.startswith("Спасибо за вопрос!"):
                # Очищаем ответ
                question = self._clean_question_text(llm_response)
                
                if question and len(question.strip()) > 10:
                    print(f"✅ Сгенерирован вопрос: {question}")
                    
                    # Сохраняем в историю
                    self.generated_questions.append({
                        "question": question,
                        "generated_at": time.time()
                    })
                    
                    return question
            
            # Fallback если LLM не сработал
            return self._get_fallback_question()
                
        except Exception as e:
            print(f"❌ Ошибка генерации вопроса: {e}")
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
            
            # Оцениваем ответ через LLM
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
            
            evaluation = self.llm.query(
                question=prompt,
                context="",
                subject=self.current_subject
            )
            
            # ДОПОЛНИТЕЛЬНАЯ ОБРАБОТКА: заменяем "ученик" на "ты"
            if evaluation:
                evaluation = self._fix_student_addressing(evaluation)
            
            return evaluation if evaluation else self._get_fallback_feedback(student_answer, correct_answer)
            
        except Exception as e:
            print(f"❌ Ошибка оценки через LLM: {e}")
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
            'ей': 'тебе',
            'свой': 'твой',
            'свою': 'твою',
            'свое': 'твое',
            'свои': 'твои'
        }
        
        result = text
        for old, new in replacements.items():
            # Заменяем с учетом регистра
            result = re.sub(r'\b' + re.escape(old) + r'\b', new, result, flags=re.IGNORECASE)
        
        return result

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
        # Простая эвристическая оценка
        student_lower = student_answer.lower()
        correct_lower = correct_answer.lower()
        
        # Проверяем наличие ключевых слов из правильного ответа
        key_words = [word for word in correct_lower.split() if len(word) > 4]
        matches = sum(1 for word in key_words if word in student_lower)
        
        if matches >= len(key_words) * 0.7:
            return "Отличный ответ! Ты хорошо понял материал."
        elif matches >= len(key_words) * 0.4:
            return "Хорошая попытка! Ты уловил основные идеи, но можно добавить детали."
        else:
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

    def save_practice_to_txt(self, lesson_id: str, practice_data: dict):
        """Сохраняет практические задания в TXT файл"""
        try:
            txt_filename = f"{lesson_id}_practice.txt"
            txt_path = self.practice_dir / txt_filename
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"ПРАКТИЧЕСКИЕ ЗАДАНИЯ: {lesson_id}\n")
                f.write("=" * 50 + "\n\n")
                
                # Сохраняем сгенерированные вопросы
                for i, question_data in enumerate(self.generated_questions, 1):
                    f.write(f"ВОПРОС {i}: {question_data.get('question', '')}\n\n")
                    
                    # Генерируем и сохраняем правильный ответ
                    correct_answer = self._generate_correct_answer(question_data['question'])
                    if correct_answer:
                        f.write(f"Правильный ответ: {correct_answer}\n")
                    
                    f.write("\n" + "-" * 30 + "\n\n")
            
            print(f"✅ Практика сохранена в TXT: {txt_path}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка сохранения практики в TXT: {e}")
            return False

    def get_practice_stats(self) -> Dict:
        """Возвращает статистику по практике"""
        return {
            "total_questions": len(self.generated_questions),
            "current_subject": self.current_subject,
            "max_questions": self.max_questions,
            "has_more_questions": self.has_more_questions()
        }

    def validate_student_answer(self, answer: str) -> Tuple[bool, str]:
        """Проверяет валидность ответа ученика"""
        if not answer or not answer.strip():
            return False, "Ответ не может быть пустым"
        
        if len(answer.strip()) < 2:
            return False, "Ответ слишком короткий"
        
        # Проверяем на команды
        commands = ['продолжай', 'дальше', 'следующий', 'стоп', 'останови', 'закончи']
        if any(cmd in answer.lower() for cmd in commands):
            return False, "Это команда, а не ответ на вопрос"
        
        # Проверяем на шумовые паттерны
        noise_patterns = [
            r'^[а-я]*ммм[а-я]*$',
            r'^[а-я]*эээ[а-я]*$', 
            r'^[а-я]*ах[а-я]*$',
            r'^[а-я]*ох[а-я]*$',
        ]
        
        for pattern in noise_patterns:
            if re.match(pattern, answer.lower()):
                return False, "Ответ похож на случайный шум"
        
        return True, "Ответ валиден"

    # Старые методы для обратной совместимости
    def load_practice(self, lesson_id: str) -> bool:
        """Загружает практические задания (для обратной совместимости)"""
        print(f"📥 Загрузка практики для урока: {lesson_id}")
        return False

    def generate_practice(self, lesson_text: str, subject: str) -> bool:
        """Генерирует практические задания (для обратной совместимости)"""
        print(f"🔧 Генерация практики для предмета: {subject}")
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


# Тестирование модуля
if __name__ == "__main__":
    print("🧪 Тестирование PracticeManager...")
    
    # Создаем mock LLM для тестирования
    class MockLLM:
        def query(self, question, context, subject):
            if "вопрос" in question.lower():
                return "Что такое основные принципы демократии?"
            elif "ответ" in question.lower():
                return "Демократия - это форма правления, при которой власть принадлежит народу."
            elif "оцени" in question.lower():
                return "Ты правильно понял основные идеи! Демократия действительно предполагает народовластие."
            return "Тестовый ответ"
    
    # Тестируем
    llm = MockLLM()
    pm = PracticeManager(llm)
    
    # Инициализация
    test_context = "Демократия - это форма правления, при которой народ является источником власти. Основные принципы демократии включают разделение властей, верховенство закона, защиту прав человека и свободные выборы."
    pm.initialize_practice_generation(test_context, "обществознание")
    
    # Генерация вопроса
    question = pm.generate_single_question()
    print(f"📝 Сгенерированный вопрос: {question}")
    
    # Оценка ответа
    test_answer = "Демократия - это когда народ выбирает власть"
    feedback = pm.evaluate_single_answer(test_answer, question)
    print(f"📊 Обратная связь: {feedback}")
    
    # Статистика
    stats = pm.get_practice_stats()
    print(f"📈 Статистика: {stats}")
    
    # Тестирование валидации ответов
    test_answers = [
        "",
        "а",
        "продолжай",
        "ммм",
        "Демократия это народовластие"
    ]
    
    for answer in test_answers:
        is_valid, message = pm.validate_student_answer(answer)
        print(f"✅ Валидация '{answer}': {is_valid} - {message}")
    
    print("🎉 Тестирование завершено!")