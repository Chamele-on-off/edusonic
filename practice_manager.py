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
        self.questions = []  # [{"id": 1, "question": "...", "answer": "..."}]
        self.current_question_index = 0
        self.current_lesson_text = ""
        self.current_subject = ""
        
        # Создаем директорию если не существует
        self.practice_dir.mkdir(parents=True, exist_ok=True)

    def load_practice(self, lesson_id: str) -> bool:
        """Загружает практические задания из JSON файла"""
        try:
            practice_file = self.practice_dir / f"{lesson_id}.json"
            
            if not practice_file.exists():
                print(f"Файл практики не найден: {practice_file}")
                return False
            
            # Загружаем практику из JSON
            with open(practice_file, 'r', encoding='utf-8') as f:
                practice_data = json.load(f)
            
            # Извлекаем вопросы из JSON структуры
            if 'questions' in practice_data and isinstance(practice_data['questions'], list):
                self.questions = []
                for i, q_data in enumerate(practice_data['questions'], 1):
                    if isinstance(q_data, dict):
                        self.questions.append({
                            "id": i,
                            "question": q_data.get('question', ''),
                            "answer": q_data.get('answer', '')
                        })
                    elif isinstance(q_data, str):
                        # Если вопросы в простом формате
                        self.questions.append({
                            "id": i,
                            "question": q_data,
                            "answer": "Ответ будет предоставлен учителем"
                        })
                
                self.current_question_index = 0
                print(f"Загружено {len(self.questions)} вопросов для практики из JSON")
                return len(self.questions) > 0
            else:
                print("Неверный формат файла практики: отсутствует список questions")
                return False
                
        except Exception as e:
            print(f"Ошибка загрузки практики: {e}")
            return False

    def generate_practice(self, lesson_text: str, subject: str) -> bool:
        """Генерирует практические задания на основе текста урока"""
        try:
            self.current_lesson_text = lesson_text
            self.current_subject = subject
            
            print(f"Генерация практики для предмета: {subject}")
            
            # Генерация вопросов через LLM с улучшенным промптом
            questions = self._generate_questions_improved(lesson_text, subject)
            if not questions or len(questions) == 0:
                print("Не удалось сгенерировать вопросы через LLM")
                # Создаем fallback вопросы на основе текста урока
                questions = self._create_fallback_questions(lesson_text)
            
            self.questions = questions
            self.current_question_index = 0
            
            print(f"Сгенерировано {len(self.questions)} вопросов для практики")
            return len(self.questions) > 0
            
        except Exception as e:
            print(f"Ошибка генерации практики: {e}")
            # Создаем минимальный fallback
            self.questions = self._create_fallback_questions(lesson_text)
            return len(self.questions) > 0

    def _generate_questions_improved(self, lesson_text: str, subject: str) -> List[Dict]:
        """Генерирует вопросы через LLM с улучшенным промптом"""
        prompt = f"""
        На основе следующего учебного материала создай 3-5 конкретных практических вопросов для проверки понимания.
        
        ТРЕБОВАНИЯ:
        - Вопросы должны быть КОНКРЕТНЫМИ и проверять понимание материала
        - Каждый вопрос должен начинаться с номера (1., 2., и т.д.)
        - Каждый вопрос должен быть на отдельной строке
        - Вопросы должны быть краткими и понятными
        - Включи вопросы разных типов: на определение, на понимание, на применение
        - Не добавляй ответы или дополнительные комментарии
        
        ПРЕДМЕТ: {subject}
        УЧЕБНЫЙ МАТЕРИАЛ:
        {lesson_text[:1000]}
        
        Пример хорошего вопроса: "1. Что такое демократия?"
        Пример плохого вопроса: "1. Расскажи о политической системе" (слишком общий)
        
        ВЕРНИ ТОЛЬКО ВОПРОСЫ В УКАЗАННОМ ФОРМАТЕ.
        """
        
        llm_response = self.llm._query_llm_api(
            prompt=prompt,
            context="",
            subject=subject,
            system_prompt="Ты — помощник учителя. Создавай КОНКРЕТНЫЕ вопросы для проверки понимания материала. Избегай общих вопросов.",
            max_tokens=800
        )
        
        if not llm_response:
            print("LLM не вернул вопросы")
            return []
        
        print(f"Сгенерированы вопросы: {llm_response[:200]}...")
        questions = self._parse_questions_improved(llm_response)
        
        # Генерируем ответы для вопросов
        if questions:
            questions_with_answers = self._generate_answers_for_questions(questions, lesson_text, subject)
            return questions_with_answers
        
        return []

    def _parse_questions_improved(self, content: str) -> List[Dict]:
        """Парсит вопросы из текста с улучшенной логикой"""
        questions = []
        
        # Разные стратегии разделения
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Ищем вопросы в формате "1. Текст вопроса"
            match = re.match(r'^(\d+)\.\s*(.+?)[.?]?$', line)
            if match:
                question_id = int(match.group(1))
                question_text = match.group(2).strip()
                
                if question_text and len(question_text) > 5:  # Минимальная длина вопроса
                    questions.append({
                        "id": question_id,
                        "question": question_text + "?",
                        "answer": ""  # Будет заполнено позже
                    })
        
        # Если не нашли в формате с номерами, пробуем другие стратегии
        if not questions:
            # Разделяем по точкам или знакам вопроса
            raw_questions = re.split(r'[.?]', content)
            for i, q_text in enumerate(raw_questions, 1):
                q_text = q_text.strip()
                if q_text and len(q_text) > 10:  # Более строгая проверка
                    # Убираем возможные номера в начале
                    q_text = re.sub(r'^\d+\.?\s*', '', q_text)
                    if q_text:
                        questions.append({
                            "id": i,
                            "question": q_text + "?",
                            "answer": ""
                        })
        
        # Ограничиваем количество вопросов
        questions = questions[:5]
        print(f"Распаршено вопросов: {len(questions)}")
        return questions

    def _generate_answers_for_questions(self, questions: List[Dict], lesson_text: str, subject: str) -> List[Dict]:
        """Генерирует ответы для вопросов"""
        questions_with_answers = []
        
        for i, question in enumerate(questions, 1):
            prompt = f"""
            Дай краткий и точный ответ на вопрос на основе учебного материала.
            
            ВОПРОС: {question['question']}
            
            УЧЕБНЫЙ МАТЕРИАЛ:
            {lesson_text[:800]}
            
            ТРЕБОВАНИЯ К ОТВЕТУ:
            - Ответ должен быть кратким (1-2 предложения)
            - Ответ должен быть точным и соответствовать материалу
            - Ответ должен быть понятным для ученика
            - Не добавляй дополнительные объяснения
            
            ВЕРНИ ТОЛЬКО ОТВЕТ БЕЗ ЛИШНИХ СЛОВ.
            """
            
            llm_response = self.llm._query_llm_api(
                prompt=prompt,
                context="",
                subject=subject,
                system_prompt="Ты — эксперт по предмету. Дай точный и краткий ответ на вопрос.",
                max_tokens=150
            )
            
            if llm_response:
                questions_with_answers.append({
                    "id": i,
                    "question": question['question'],
                    "answer": llm_response.strip()
                })
                print(f"Сгенерирован ответ для вопроса {i}")
            else:
                # Fallback ответ
                questions_with_answers.append({
                    "id": i,
                    "question": question['question'],
                    "answer": "Ответ основан на материале урока. Учитель объяснит подробнее."
                })
        
        return questions_with_answers

    def _create_fallback_questions(self, lesson_text: str) -> List[Dict]:
        """Создает простые вопросы на основе текста урока"""
        questions = []
        
        # Извлекаем ключевые предложения из текста
        sentences = re.split(r'[.!?]+', lesson_text)
        key_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:3]  # Берем 3 самых длинных предложения
        
        for i, sentence in enumerate(key_sentences, 1):
            if sentence:
                # Создаем простой вопрос на основе предложения
                words = sentence.split()
                if len(words) > 5:
                    # Берем ключевые слова из предложения
                    key_words = ' '.join(words[2:5])  # Пропускаем первые 2 слова
                    question = f"Что означает '{key_words}' в контексте урока?"
                    
                    questions.append({
                        "id": i,
                        "question": question,
                        "answer": sentence
                    })
        
        # Если не получилось создать вопросы, создаем минимальный набор
        if not questions:
            questions = [
                {
                    "id": 1,
                    "question": "Что было главной темой урока?",
                    "answer": "Основная тема урока была рассмотрена в материале."
                },
                {
                    "id": 2, 
                    "question": "Какие ключевые понятия ты запомнил?",
                    "answer": "Урок содержал важные понятия по теме."
                }
            ]
        
        return questions

    def get_question(self, index: int) -> Optional[Dict]:
        """Возвращает вопрос по индексу"""
        if 0 <= index < len(self.questions):
            return self.questions[index]
        return None

    def get_current_question(self) -> Optional[Dict]:
        """Возвращает текущий вопрос"""
        return self.get_question(self.current_question_index)

    def get_next_question(self) -> Optional[Dict]:
        """Возвращает следующий вопрос и увеличивает индекс"""
        if self.current_question_index < len(self.questions):
            question = self.questions[self.current_question_index]
            self.current_question_index += 1
            return question
        return None

    def evaluate_answer(self, student_answer: str, use_llm: bool = True) -> str:
        """Оценивает ответ ученика"""
        current_question = self.get_current_question()
        if not current_question:
            return "Вопрос не найден."
        
        correct_answer = current_question['answer']
        question_text = current_question['question']
        
        # Упрощенная оценка через сравнение
        return self._evaluate_with_similarity(student_answer, correct_answer)

    def evaluate_answer_with_context(self, student_answer: str, question: str, correct_answer: str, context: str = "") -> str:
        """Оценивает ответ ученика с учетом контекста урока"""
        return self._evaluate_with_similarity(student_answer, correct_answer)

    def _evaluate_with_similarity(self, student_answer: str, correct_answer: str) -> str:
        """Оценивает ответ через сравнение схожести"""
        similarity = self._calculate_similarity(student_answer, correct_answer)
        
        if similarity > 0.7:
            return "Правильно! Отличный ответ!"
        elif similarity > 0.5:
            return f"Почти верно! Правильный ответ: {correct_answer}"
        elif similarity > 0.3:
            return f"Есть правильные мысли. Полный ответ: {correct_answer}"
        else:
            return f"Попробуй еще раз. Правильно будет: {correct_answer}"

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Вычисляет схожесть двух текстов"""
        text1 = re.sub(r'\s+', ' ', text1.lower()).strip()
        text2 = re.sub(r'\s+', ' ', text2.lower()).strip()
        return SequenceMatcher(None, text1, text2).ratio()

    def move_to_next_question(self) -> bool:
        """Переходит к следующему вопросу (без увеличения индекса)"""
        return self.current_question_index < len(self.questions)

    def has_more_questions(self) -> bool:
        """Проверяет, есть ли еще вопросы"""
        return self.current_question_index < len(self.questions)

    def get_questions_count(self) -> int:
        """Возвращает количество вопросов"""
        return len(self.questions)

    def get_progress(self) -> Tuple[int, int]:
        """Возвращает текущий прогресс (текущий вопрос, всего вопросов)"""
        return (self.current_question_index + 1, len(self.questions))

    def reset(self):
        """Сброс состояния менеджера практики"""
        self.questions = []
        self.current_question_index = 0
        self.current_lesson_text = ""
        self.current_subject = ""

    def save_practice_for_lesson(self, lesson_id: str, questions: List[Dict], answers: List[Dict]) -> bool:
        """Сохраняет практику для конкретного урока"""
        try:
            practice_data = {
                "lesson_id": lesson_id,
                "questions": questions,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "question_count": len(questions)
            }
            
            practice_file = self.practice_dir / f"{lesson_id}.json"
            with open(practice_file, 'w', encoding='utf-8') as f:
                json.dump(practice_data, f, ensure_ascii=False, indent=2)
            
            print(f"Практика сохранена в: {practice_file}")
            return True
            
        except Exception as e:
            print(f"Ошибка сохранения практики: {e}")
            return False