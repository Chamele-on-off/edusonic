# practice_manager.py - ОБНОВЛЕННАЯ ВЕРСИЯ

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
        self.current_lesson_summary = ""  # НОВОЕ: краткое содержание для практики
        self.current_subject = ""
        self.generated_questions = []
        self.current_question_index = 0
        self.max_questions = 5
        
        # Создаем директорию если не существует
        self.practice_dir.mkdir(parents=True, exist_ok=True)

    def initialize_practice_generation(self, lesson_context: str, subject: str):
        """Инициализирует генерацию практических вопросов с созданием краткого содержания"""
        self.current_lesson_context = lesson_context
        self.current_subject = subject
        self.generated_questions = []
        self.current_question_index = 0
        
        # НОВОЕ: Создаем краткое содержание урока для практики
        self.current_lesson_summary = self._generate_lesson_summary(lesson_context)
        
        print(f"🎯 Инициализирована генерация практики для предмета: {subject}")
        print(f"📝 Создано краткое содержание урока для практики ({len(self.current_lesson_summary)} символов)")

    def _generate_lesson_summary(self, lesson_context: str) -> str:
        """Генерирует краткое содержание урока специально для практики"""
        try:
            # Используем LLM для создания краткого содержания
            prompt = f"""
            Создай КРАТКОЕ содержание этого урока для использования в практических вопросах.
            Выдели только ключевые понятия и основные идеи.
            
            ИСХОДНЫЙ ТЕКСТ:
            {lesson_context[:1500]}
            
            ТРЕБОВАНИЯ:
            - Только ключевые факты и понятия
            - Максимально кратко (не более 300 слов)
            - Структурировано по основным темам
            - Без подробных объяснений
            - Только на русском языке
            
            Верни только краткое содержание.
            """
            
            summary = self.llm.query(
                question=prompt,
                context="",
                subject=self.current_subject
            )
            
            if summary and len(summary) > 50:
                return summary
            else:
                # Fallback: берем первые 500 символов исходного текста
                return lesson_context[:500] + "..."
                
        except Exception as e:
            print(f"❌ Ошибка генерации краткого содержания: {e}")
            return lesson_context[:500] + "..."

    def generate_single_question(self) -> Optional[str]:
        """Генерирует один УНИКАЛЬНЫЙ вопрос на основе контекста урока"""
        try:
            # Проверяем лимит вопросов
            if len(self.generated_questions) >= self.max_questions:
                print("🏁 Достигнут лимит вопросов")
                return None
            
            print(f"🔄 Генерация вопроса {len(self.generated_questions) + 1}/{self.max_questions}...")
            
            # УЛУЧШЕННЫЙ промпт с историей вопросов и запретом дублирования
            previous_questions = self._get_previous_questions_text()
            
            prompt = f"""
            Создай ОДИН УНИКАЛЬНЫЙ учебный вопрос для проверки понимания темы.
            
            ПРЕДМЕТ: {self.current_subject}
            
            КРАТКОЕ СОДЕРЖАНИЕ УРОКА:
            {self.current_lesson_summary}
            
            УЖЕ ЗАДАННЫЕ ВОПРОСЫ (НЕ ПОВТОРЯЙ ИХ!):
            {previous_questions}
            
            КРИТИЧЕСКИ ВАЖНЫЕ ТРЕБОВАНИЯ:
            1. Вопрос ДОЛЖЕН быть УНИКАЛЬНЫМ и не похожим на уже заданные
            2. Проверяй понимание РАЗНЫХ аспектов материала
            3. Чередуй типы вопросов: фактические, объяснительные, сравнительные
            4. Вопрос должен требовать развернутого ответа
            5. Максимально разнообразь тематику вопросов
            
            ТИПЫ ВОПРОСОВ ДЛЯ РАЗНООБРАЗИЯ:
            - Фактический вопрос (кто, что, когда)
            - Объяснительный вопрос (почему, как)
            - Сравнительный вопрос (сравни, различи)
            - Прикладной вопрос (как применить)
            - Аналитический вопрос (проанализируй)
            
            Верни ТОЛЬКО текст одного вопроса без нумерации и лишних слов.
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
                    
                    # ПРОВЕРЯЕМ УНИКАЛЬНОСТЬ вопроса
                    if question and self._is_question_unique(question):
                        print(f"✅ Уникальный вопрос сгенерирован: {question[:100]}...")
                        
                        # Сохраняем в историю
                        self.generated_questions.append({
                            "question": question,
                            "generated_at": time.time(),
                            "type": self._detect_question_type(question)
                        })
                        
                        return question
                    else:
                        print("⚠️ Вопрос не уникален, пробуем сгенерировать другой...")
                        # Рекурсивно пробуем еще раз (максимум 2 попытки)
                        if len(self.generated_questions) < 3:  # Ограничиваем рекурсию
                            return self.generate_single_question()
                        else:
                            return self._get_fallback_question(ensure_unique=True)
                    
            except Exception as e:
                print(f"❌ Ошибка генерации вопроса: {e}")
            
            # Fallback если не удалось сгенерировать
            fallback_question = self._get_fallback_question(ensure_unique=True)
            if fallback_question:
                self.generated_questions.append({
                    "question": fallback_question,
                    "generated_at": time.time(),
                    "type": "fallback"
                })
                return fallback_question
                
            return None
                
        except Exception as e:
            print(f"❌ Критическая ошибка генерации вопроса: {e}")
            return self._get_fallback_question(ensure_unique=True)

    def _get_previous_questions_text(self) -> str:
        """Возвращает текст уже заданных вопросов для промпта"""
        if not self.generated_questions:
            return "Вопросов еще не было. Это первый вопрос."
        
        questions_text = "Уже заданные вопросы:\n"
        for i, q_data in enumerate(self.generated_questions, 1):
            questions_text += f"{i}. {q_data['question']}\n"
        
        return questions_text

    def _is_question_unique(self, new_question: str, similarity_threshold: float = 0.7) -> bool:
        """Проверяет, является ли вопрос уникальным по сравнению с уже заданными"""
        if not self.generated_questions:
            return True
        
        new_question_lower = new_question.lower()
        
        for existing_q in self.generated_questions:
            existing_question_lower = existing_q["question"].lower()
            
            # Проверяем схожесть с помощью SequenceMatcher
            similarity = SequenceMatcher(None, new_question_lower, existing_question_lower).ratio()
            
            # Проверяем ключевые слова (если много совпадений - вероятно дубликат)
            new_words = set(new_question_lower.split())
            existing_words = set(existing_question_lower.split())
            common_words = new_words.intersection(existing_words)
            
            word_similarity = len(common_words) / max(len(new_words), len(existing_words))
            
            # Если любой из показателей схожести превышает порог - вопрос не уникален
            if similarity > similarity_threshold or word_similarity > 0.6:
                print(f"⚠️ Обнаружен похожий вопрос: similarity={similarity:.2f}, word_similarity={word_similarity:.2f}")
                return False
        
        return True

    def _detect_question_type(self, question: str) -> str:
        """Определяет тип вопроса для разнообразия"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['почему', 'объясни', 'какова причина']):
            return "объяснительный"
        elif any(word in question_lower for word in ['сравни', 'различи', 'отличие']):
            return "сравнительный" 
        elif any(word in question_lower for word in ['как применить', 'пример', 'использовать']):
            return "прикладной"
        elif any(word in question_lower for word in ['проанализируй', 'оцени', 'как ты думаешь']):
            return "аналитический"
        else:
            return "фактический"

    def _get_fallback_question(self, ensure_unique: bool = False) -> str:
        """Fallback вопрос когда не удается сгенерировать через LLM"""
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
            "Почему эта тема важна для понимания предмета?",
            "Какие связи можно установить между изученными понятиями?",
            "В чем практическая ценность этого материала?",
            "Какие вопросы у тебя возникли при изучении этой темы?",
            "Как бы ты объяснил эту тему другому ученику?",
            "Что было самым интересным в этом материале?"
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

    # Остальные методы остаются без изменений...
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
            {self.current_lesson_summary}
            
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
            {self.current_lesson_summary}
            
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
        self.current_lesson_summary = ""
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
        question_types = {}
        for q in self.generated_questions:
            q_type = q.get("type", "unknown")
            question_types[q_type] = question_types.get(q_type, 0) + 1
        
        return {
            "total_questions": len(self.generated_questions),
            "current_subject": self.current_subject,
            "max_questions": self.max_questions,
            "has_more_questions": self.has_more_questions(),
            "question_types": question_types,
            "lesson_summary_length": len(self.current_lesson_summary)
        }