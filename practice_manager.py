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
        """Загружает практические задания из раздельных файлов вопросов и ответов"""
        try:
            questions_file = self.practice_dir / f"{lesson_id}_questions.txt"
            answers_file = self.practice_dir / f"{lesson_id}_answers.txt"
            
            if not questions_file.exists() or not answers_file.exists():
                print(f"Файлы практики не найдены: {questions_file} или {answers_file}")
                return False
            
            # Загружаем вопросы
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions_content = f.read()
            questions = self._parse_questions(questions_content)
            
            # Загружаем ответы
            with open(answers_file, 'r', encoding='utf-8') as f:
                answers_content = f.read()
            answers = self._parse_answers(answers_content)
            
            # Объединяем вопросы и ответы
            self.questions = self._combine_questions_answers(questions, answers)
            self.current_question_index = 0
            
            print(f"Загружено {len(self.questions)} вопросов для практики")
            return len(self.questions) > 0
            
        except Exception as e:
            print(f"Ошибка загрузки практики: {e}")
            return False

    def generate_practice(self, lesson_text: str, subject: str) -> bool:
        """Генерирует практические задания на основе текста урока и сохраняет в раздельные файлы"""
        try:
            self.current_lesson_text = lesson_text
            self.current_subject = subject
            
            print(f"Генерация практики для предмета: {subject}")
            
            # 1. Генерация вопросов
            questions = self._generate_questions(lesson_text, subject)
            if not questions:
                print("Не удалось сгенерировать вопросы")
                return False
            
            # 2. Генерация ответов на основе вопросов и контекста урока
            answers = self._generate_answers(questions, lesson_text, subject)
            if not answers:
                print("Не удалось сгенерировать ответы")
                return False
            
            # 3. Объединяем вопросы и ответы
            self.questions = self._combine_questions_answers(questions, answers)
            self.current_question_index = 0
            
            # 4. Сохраняем в раздельные файлы
            success = self._save_practice_files(lesson_id="generated_practice", questions=questions, answers=answers)
            
            if success:
                print(f"Сгенерировано и сохранено {len(self.questions)} вопросов для практики")
            else:
                print("Ошибка сохранения практики в файлы")
                
            return success
            
        except Exception as e:
            print(f"Ошибка генерации практики: {e}")
            return False

    def _generate_questions(self, lesson_text: str, subject: str) -> List[Dict]:
        """Генерирует вопросы через LLM"""
        prompt = f"""
        На основе следующего учебного материала создай 5 практических вопросов для проверки понимания.
        
        ТРЕБОВАНИЯ К ФОРМАТУ:
        - Каждый вопрос должен начинаться с номера и точки: "1.", "2.", etc.
        - Каждый вопрос должен быть на отдельной строке
        - Разделяй вопросы ДВУМЯ переводами строки (пустой строкой)
        - Вопросы должны проверять ключевые понятия из материала
        - Вопросы должны быть краткими и понятными
        - Не добавляй ответы или дополнительные комментарии
        
        ПРЕДМЕТ: {subject}
        УЧЕБНЫЙ МАТЕРИАЛ:
        {lesson_text[:1500]}
        
        ВЕРНИ ТОЛЬКО ВОПРОСЫ В УКАЗАННОМ ФОРМАТЕ.
        """
        
        llm_response = self.llm._query_llm_api(
            prompt=prompt,
            context="",
            subject=subject,
            system_prompt="Ты — помощник учителя. Создавай качественные вопросы для проверки понимания материала. Строго следуй указанному формату.",
            max_tokens=1000
        )
        
        if not llm_response:
            print("LLM не вернул вопросы")
            return []
        
        print(f"Сгенерированы вопросы: {llm_response[:200]}...")
        return self._parse_questions(llm_response)

    def _generate_answers(self, questions: List[Dict], lesson_text: str, subject: str) -> List[Dict]:
        """Генерирует ответы на вопросы через LLM"""
        answers = []
        
        for i, question in enumerate(questions, 1):
            prompt = f"""
            На основе учебного материала дай точный и краткий ответ на вопрос.
            
            ВОПРОС {i}: {question['question']}
            
            УЧЕБНЫЙ МАТЕРИАЛ:
            {lesson_text[:1000]}
            
            ТРЕБОВАНИЯ:
            - Ответ должен быть кратким (1-2 предложения)
            - Ответ должен быть точным и соответствовать материалу
            - Не добавляй дополнительные объяснения или комментарии
            
            ВЕРНИ ТОЛЬКО ОТВЕТ БЕЗ ЛИШНИХ СЛОВ.
            """
            
            llm_response = self.llm._query_llm_api(
                prompt=prompt,
                context="",
                subject=subject,
                system_prompt="Ты — эксперт по предмету. Дай точный и краткий ответ на вопрос.",
                max_tokens=200
            )
            
            if llm_response:
                answers.append({
                    "id": i,
                    "answer": llm_response.strip()
                })
                print(f"Сгенерирован ответ для вопроса {i}")
            else:
                # Fallback ответ
                answers.append({
                    "id": i,
                    "answer": "Информация по этому вопросу содержится в учебном материале."
                })
        
        return answers

    def _parse_questions(self, content: str) -> List[Dict]:
        """Парсит вопросы из текста с улучшенным разделением по абзацам"""
        questions = []
        
        # Улучшенное разделение на вопросы (по двойным переводам строк)
        raw_questions = [q.strip() for q in content.strip().split('\n\n') if q.strip()]
        
        for i, question_block in enumerate(raw_questions, 1):
            if not question_block:
                continue
                
            # Очищаем от номеров и форматирования
            question_text = re.sub(r'^\d+\.\s*', '', question_block)
            question_text = question_text.strip()
            
            if question_text:
                questions.append({
                    "id": i,
                    "question": question_text
                })
        
        print(f"Распаршено вопросов: {len(questions)}")
        return questions

    def _parse_answers(self, content: str) -> List[Dict]:
        """Парсит ответы из текста"""
        answers = []
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Ищем ответы в формате "1. Текст ответа"
            match = re.match(r'^(\d+)\.\s*(.+)$', line)
            if match:
                answer_id = int(match.group(1))
                answer_text = match.group(2).strip()
                
                if answer_text:
                    answers.append({
                        "id": answer_id,
                        "answer": answer_text
                    })
        
        print(f"Распаршено ответов: {len(answers)}")
        return answers

    def _combine_questions_answers(self, questions: List[Dict], answers: List[Dict]) -> List[Dict]:
        """Объединяет вопросы и ответы по ID"""
        combined = []
        answers_dict = {answer["id"]: answer["answer"] for answer in answers}
        
        for question in questions:
            question_id = question["id"]
            answer = answers_dict.get(question_id, "Ответ не найден")
            
            combined.append({
                "id": question_id,
                "question": question["question"],
                "answer": answer
            })
        
        # Сортируем по ID
        combined.sort(key=lambda x: x["id"])
        return combined

    def _save_practice_files(self, lesson_id: str, questions: List[Dict], answers: List[Dict]) -> bool:
        """Сохраняет вопросы и ответы в раздельные файлы"""
        try:
            # Сохраняем вопросы
            questions_content = ""
            for q in questions:
                questions_content += f"{q['id']}. {q['question']}\n\n"
            
            questions_file = self.practice_dir / f"{lesson_id}_questions.txt"
            with open(questions_file, 'w', encoding='utf-8') as f:
                f.write(questions_content.strip())
            
            # Сохраняем ответы
            answers_content = ""
            for a in answers:
                answers_content += f"{a['id']}. {a['answer']}\n\n"
            
            answers_file = self.practice_dir / f"{lesson_id}_answers.txt"
            with open(answers_file, 'w', encoding='utf-8') as f:
                f.write(answers_content.strip())
            
            print(f"Практика сохранена в: {questions_file} и {answers_file}")
            return True
            
        except Exception as e:
            print(f"Ошибка сохранения практики: {e}")
            return False

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
        """Оценивает ответ ученика с приоритетом LLM и fallback на сравнение"""
        current_question = self.get_current_question()
        if not current_question:
            return "Вопрос не найден."
        
        correct_answer = current_question['answer']
        question_text = current_question['question']
        
        # Приоритет: оценка через LLM
        if use_llm and self._is_llm_available():
            return self._evaluate_with_llm(question_text, student_answer, correct_answer)
        else:
            # Fallback: сравнение ответов
            return self._evaluate_with_similarity(student_answer, correct_answer)

    def evaluate_answer_with_context(self, student_answer: str, question: str, correct_answer: str, context: str = "") -> str:
        """Оценивает ответ ученика с учетом контекста урока"""
        # Приоритет: оценка через LLM с контекстом
        if self._is_llm_available():
            return self._evaluate_with_llm_context(question, student_answer, correct_answer, context)
        else:
            # Fallback: сравнение ответов
            return self._evaluate_with_similarity(student_answer, correct_answer)

    def _is_llm_available(self) -> bool:
        """Проверяет доступность LLM"""
        try:
            return hasattr(self.llm, 'api_key') and bool(self.llm.api_key)
        except:
            return False

    def _evaluate_with_llm(self, question: str, student_answer: str, correct_answer: str) -> str:
        """Оценивает ответ через LLM с улучшенными промптами"""
        try:
            prompt = f"""
            ВОПРОС ДЛЯ УЧЕНИКА: {question}
            ПРАВИЛЬНЫЙ ОТВЕТ: {correct_answer}
            ОТВЕТ УЧЕНИКА: {student_answer}
            
            Твоя задача - дать обратную связь ученику. Будь добрым и поддерживающим учителем:
            
            ЕСЛИ ОТВЕТ ПРАВИЛЬНЫЙ:
            - Похвали ученика конкретно
            - Подтверди правильность ответа
            - Скажи что-то ободряющее
            
            ЕСЛИ ОТВЕТ ЧАСТИЧНО ПРАВИЛЬНЫЙ:
            - Отметь что было правильно
            - Вежливо укажи на ошибки
            - Дай правильный ответ
            - Объясни в чем была ошибка
            
            ЕСЛИ ОТВЕТ НЕПРАВИЛЬНЫЙ:
            - Не ругай, а поддержи ученика
            - Объясни почему ответ неверный
            - Дай правильный ответ понятным языком
            - Предложи попробовать еще раз в будущем
            
            ОБРАЩАЙСЯ К УЧЕНИКУ НА "ТЫ" И БУДЬ ДРУЖЕЛЮБНЫМ!
            Максимум 2-3 предложения. Отвечай на русском языке.
            """
            
            system_prompt = f"""Ты - опытный и добрый учитель по предмету {self.current_subject}. 
            Твоя задача - помогать ученикам учиться на ошибках, а не ругать их. 
            Всегда будь поддерживающим и терпеливым."""
            
            evaluation = self.llm._query_llm_api(
                prompt=prompt,
                context="",
                subject=self.current_subject,
                system_prompt=system_prompt,
                max_tokens=300
            )
            
            return evaluation if evaluation else self._get_fallback_feedback(student_answer, correct_answer)
            
        except Exception as e:
            print(f"Ошибка оценки через LLM: {e}")
            return self._get_fallback_feedback(student_answer, correct_answer)

    def _evaluate_with_llm_context(self, question: str, student_answer: str, correct_answer: str, context: str) -> str:
        """Оценивает ответ через LLM с учетом контекста урока"""
        try:
            prompt = f"""
            КОНТЕКСТ УРОКА: {context[:1000]}
            ВОПРОС: {question}
            ПРАВИЛЬНЫЙ ОТВЕТ: {correct_answer}
            ОТВЕТ УЧЕНИКА: {student_answer}
            
            Проанализируй ответ ученика с учетом контекста урока. Будь добрым и поддерживающим учителем:
            
            ЕСЛИ ОТВЕТ ПРАВИЛЬНЫЙ ИЛИ БЛИЗКИЙ К ПРАВИЛЬНОМУ:
            - Похвали ученика конкретно
            - Подтверди правильность ответа
            - Если ответ неполный, дополни его
            - Скажи что-то ободряющее
            
            ЕСЛИ ОТВЕТ ЧАСТИЧНО ПРАВИЛЬНЫЙ:
            - Отметь что было правильно
            - Вежливо укажи на ошибки или неточности
            - Дай правильный ответ с объяснением
            - Ссылайся на контекст урока если это уместно
            
            ЕСЛИ ОТВЕТ НЕПРАВИЛЬНЫЙ:
            - Не ругай, а поддержи ученика
            - Объясни почему ответ неверный с ссылкой на материал урока
            - Дай правильный ответ понятным языком
            - Предложи обратить внимание на конкретные аспекты темы
            
            ОБРАЩАЙСЯ К УЧЕНИКУ НА "ТЫ" И БУДЬ ДРУЖЕЛЮБНЫМ!
            Учитывай контекст урока при оценке.
            Максимум 2-3 предложения. Отвечай на русском языке.
            """
            
            system_prompt = f"""Ты - опытный и добрый учитель по предмету {self.current_subject}. 
            Твоя задача - помогать ученикам учиться на ошибках, используя контекст пройденного материала. 
            Всегда будь поддерживающим и терпеливым."""
            
            evaluation = self.llm._query_llm_api(
                prompt=prompt,
                context="",
                subject=self.current_subject,
                system_prompt=system_prompt,
                max_tokens=300
            )
            
            return evaluation if evaluation else self._get_fallback_feedback(student_answer, correct_answer)
            
        except Exception as e:
            print(f"Ошибка оценки через LLM с контекстом: {e}")
            return self._get_fallback_feedback(student_answer, correct_answer)

    def _get_fallback_feedback(self, student_answer: str, correct_answer: str) -> str:
        """Fallback обратная связь когда LLM недоступен"""
        similarity = self._calculate_similarity(student_answer, correct_answer)
        
        if similarity > 0.8:
            return "Отлично! Ты абсолютно прав! Это правильный ответ."
        elif similarity > 0.6:
            return f"Хорошо! Ты близок к правильному ответу. Полностью верно будет: {correct_answer}"
        elif similarity > 0.4:
            return f"Есть правильные мысли! Но точный ответ такой: {correct_answer}"
        else:
            return f"Попробуй еще раз! Правильный ответ: {correct_answer}. Не расстраивайся, ошибки - это часть обучения!"

    def _evaluate_with_similarity(self, student_answer: str, correct_answer: str) -> str:
        """Оценивает ответ через сравнение схожести"""
        similarity = self._calculate_similarity(student_answer, correct_answer)
        
        if similarity > 0.8:
            return f"Да, верно! {correct_answer}"
        elif similarity > 0.6:
            return f"Почти верно! Правильно: {correct_answer}"
        elif similarity > 0.4:
            return f"Частично верно. Полный ответ: {correct_answer}"
        else:
            return f"Не совсем. Правильный ответ: {correct_answer}"

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
        return self._save_practice_files(lesson_id, questions, answers)
