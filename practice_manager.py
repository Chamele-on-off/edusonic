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
        self.questions = []  # Не кешируем все вопросы сразу
        self.current_question_index = 0
        self.current_lesson_text = ""
        self.current_subject = ""
        self.current_lesson_id = ""  # Сохраняем ID урока для поточной загрузки
        
        # Создаем директорию если не существует
        self.practice_dir.mkdir(parents=True, exist_ok=True)

    def load_practice(self, lesson_id: str) -> bool:
        """Проверяет наличие файлов практики, но не загружает все вопросы сразу"""
        try:
            questions_file = self.practice_dir / f"{lesson_id}_questions.txt"
            answers_file = self.practice_dir / f"{lesson_id}_answers.txt"
            
            if not questions_file.exists() or not answers_file.exists():
                print(f"Файлы практики не найдены: {questions_file} или {answers_file}")
                return False
            
            self.current_lesson_id = lesson_id
            self.current_question_index = 0
            
            # Проверяем количество вопросов (только подсчет)
            with open(questions_file, 'r', encoding='utf-8') as f:
                content = f.read()
                question_count = len(re.findall(r'^\d+\.', content, re.MULTILINE))
            
            print(f"Файлы практики найдены. Вопросов: {question_count}")
            return question_count > 0
            
        except Exception as e:
            print(f"Ошибка проверки практики: {e}")
            return False

    def get_current_question(self) -> Optional[Dict]:
        """Загружает ТОЛЬКО текущий вопрос по требованию"""
        try:
            if not self.current_lesson_id:
                return None
            
            questions_file = self.practice_dir / f"{self.current_lesson_id}_questions.txt"
            answers_file = self.practice_dir / f"{self.current_lesson_id}_answers.txt"
            
            if not questions_file.exists() or not answers_file.exists():
                return None
            
            # Загружаем вопросы и находим нужный
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions_content = f.read()
            
            questions = self._parse_questions(questions_content)
            if self.current_question_index >= len(questions):
                return None
            
            current_question = questions[self.current_question_index]
            
            # Загружаем ответ для этого вопроса
            with open(answers_file, 'r', encoding='utf-8') as f:
                answers_content = f.read()
            
            answers = self._parse_answers(answers_content)
            answer_dict = {answer["id"]: answer["answer"] for answer in answers}
            
            question_data = {
                "id": current_question["id"],
                "question": current_question["question"],
                "answer": answer_dict.get(current_question["id"], "Ответ не найден")
            }
            
            print(f"Загружен вопрос {self.current_question_index + 1}: {question_data['question'][:50]}...")
            return question_data
            
        except Exception as e:
            print(f"Ошибка загрузки текущего вопроса: {e}")
            return None

    def generate_practice(self, lesson_text: str, subject: str) -> bool:
        """Генерирует практические задания и сохраняет в раздельные файлы"""
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
            
            # 3. Сохраняем в раздельные файлы
            lesson_id = f"practice_{int(time.time())}"
            success = self._save_practice_files(lesson_id, questions, answers)
            
            if success:
                self.current_lesson_id = lesson_id
                self.current_question_index = 0
                print(f"Сгенерировано и сохранено {len(questions)} вопросов для практики")
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
            system_prompt="Ты — помощник учителя. Создавай качественные вопросы для проверки понимания материала. Строго следуй указанному формату. Возвращай только нумерованный список вопросов.",
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
            
            ВОПРОС: {question['question']}
            
            УЧЕБНЫЙ МАТЕРИАЛ:
            {lesson_text[:1000]}
            
            ТРЕБОВАНИЯ:
            - Ответ должен быть кратким (1-2 предложения)
            - Ответ должен быть точным и соответствовать материалу
            - Не добавляй дополнительные объяснения или комментарии
            - Ответ должен быть готов к использованию в системе обучения
            
            ВЕРНИ ТОЛЬКО ОТВЕТ БЕЗ ЛИШНИХ СЛОВ.
            """
            
            llm_response = self.llm._query_llm_api(
                prompt=prompt,
                context="",
                subject=subject,
                system_prompt="Ты — эксперт по предмету. Дай точный и краткий ответ на вопрос. Не добавляй пояснений.",
                max_tokens=200
            )
            
            if llm_response:
                # Очищаем ответ от лишних фраз
                clean_answer = self._clean_answer(llm_response)
                answers.append({
                    "id": i,
                    "answer": clean_answer
                })
                print(f"Сгенерирован ответ для вопроса {i}: {clean_answer[:50]}...")
            else:
                answers.append({
                    "id": i,
                    "answer": "Информация по этому вопросу содержится в учебном материале."
                })
        
        return answers

    def _clean_answer(self, answer: str) -> str:
        """Очищает ответ от LLM от лишних фраз"""
        # Убираем фразы вроде "Ответ:", "Правильный ответ:" и т.д.
        patterns = [
            r'^(Ответ|Правильный ответ|Краткий ответ|Ответ на вопрос)[:\s]*',
            r'[.!?]\s*$'
        ]
        
        clean_answer = answer.strip()
        for pattern in patterns:
            clean_answer = re.sub(pattern, '', clean_answer, flags=re.IGNORECASE)
        
        return clean_answer.strip()

    def _parse_questions(self, content: str) -> List[Dict]:
        """Парсит вопросы из текста"""
        questions = []
        lines = content.strip().split('\n')
        
        current_id = 1
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Ищем вопросы в формате "1. Текст вопроса"
            match = re.match(r'^(\d+)\.\s*(.+)$', line)
            if match:
                question_id = int(match.group(1))
                question_text = match.group(2).strip()
                current_id = question_id
            else:
                # Если строка не начинается с цифры, но не пустая - это продолжение вопроса
                if questions and line:
                    last_question = questions[-1]
                    last_question["question"] += " " + line
                continue
                
            if question_text:
                questions.append({
                    "id": question_id,
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

    def _save_practice_files(self, lesson_id: str, questions: List[Dict], answers: List[Dict]) -> bool:
        """Сохраняет вопросы и ответы в раздельные файлы"""
        try:
            # Сохраняем вопросы
            questions_content = ""
            for q in sorted(questions, key=lambda x: x["id"]):
                questions_content += f"{q['id']}. {q['question']}\n\n"
            
            questions_file = self.practice_dir / f"{lesson_id}_questions.txt"
            with open(questions_file, 'w', encoding='utf-8') as f:
                f.write(questions_content.strip())
            
            # Сохраняем ответы
            answers_content = ""
            for a in sorted(answers, key=lambda x: x["id"]):
                answers_content += f"{a['id']}. {a['answer']}\n\n"
            
            answers_file = self.practice_dir / f"{lesson_id}_answers.txt"
            with open(answers_file, 'w', encoding='utf-8') as f:
                f.write(answers_content.strip())
            
            print(f"Практика сохранена в: {questions_file} и {answers_file}")
            return True
            
        except Exception as e:
            print(f"Ошибка сохранения практики: {e}")
            return False

    def evaluate_answer(self, student_answer: str, use_llm: bool = True) -> str:
        """Оценивает ответ ученика с улучшенными промптами"""
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

    def _is_llm_available(self) -> bool:
        """Проверяет доступность LLM"""
        try:
            return hasattr(self.llm, 'api_key') and bool(self.llm.api_key)
        except:
            return False

    def _evaluate_with_llm(self, question: str, student_answer: str, correct_answer: str) -> str:
        """Оценивает ответ через LLM с улучшенным промптом"""
        try:
            prompt = f"""
            Вопрос: {question}
            Правильный ответ: {correct_answer}
            Ответ ученика: {student_answer}
            
            Проанализируй ответ ученика и дай обратную связь. Будь поддерживающим учителем:
            
            - Если ответ правильный: похвали и подтверди правильность
            - Если ответ частично правильный: отметь что правильно, а что нужно дополнить
            - Если ответ неправильный: вежливо укажи на ошибку и объясни правильный ответ
            
            Будь кратким (2-3 предложения), дружелюбным и конструктивным.
            Обращайся к ученику на "вы" или используй нейтральные формулировки.
            """
            
            system_prompt = f"""Ты — опытный и доброжелательный учитель по предмету {self.current_subject}. 
            Твоя задача — дать полезную обратную связь, которая мотивирует ученика продолжать обучение.
            Избегай фраз "ответ ученика" - используй более личные обращения."""
            
            evaluation = self.llm._query_llm_api(
                prompt=prompt,
                context="",
                subject=self.current_subject,
                system_prompt=system_prompt,
                max_tokens=250
            )
            
            if evaluation:
                # Заменяем безличные формулировки на более персональные
                evaluation = evaluation.replace("Ответ ученика", "Ваш ответ")
                evaluation = evaluation.replace("ученик", "вы")
                evaluation = evaluation.replace("Ученик", "Вы")
                return evaluation
            else:
                return self._evaluate_with_similarity(student_answer, correct_answer)
            
        except Exception as e:
            print(f"Ошибка оценки через LLM: {e}")
            return self._evaluate_with_similarity(student_answer, correct_answer)

    def _evaluate_with_similarity(self, student_answer: str, correct_answer: str) -> str:
        """Оценивает ответ через сравнение схожести с улучшенными формулировками"""
        similarity = self._calculate_similarity(student_answer, correct_answer)
        
        if similarity > 0.8:
            return f"Правильно! {correct_answer}"
        elif similarity > 0.6:
            return f"Почти верно! Полный ответ: {correct_answer}"
        elif similarity > 0.4:
            return f"Есть неточности. Правильно: {correct_answer}"
        else:
            return f"Пока не совсем верно. Давайте разберем: {correct_answer}"

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Вычисляет схожесть двух текстов"""
        text1 = re.sub(r'\s+', ' ', text1.lower()).strip()
        text2 = re.sub(r'\s+', ' ', text2.lower()).strip()
        return SequenceMatcher(None, text1, text2).ratio()

    def move_to_next_question(self) -> bool:
        """Переходит к следующему вопросу"""
        self.current_question_index += 1
        return self.current_question_index < self._get_questions_count()

    def _get_questions_count(self) -> int:
        """Возвращает общее количество вопросов"""
        try:
            if not self.current_lesson_id:
                return 0
            
            questions_file = self.practice_dir / f"{self.current_lesson_id}_questions.txt"
            if not questions_file.exists():
                return 0
            
            with open(questions_file, 'r', encoding='utf-8') as f:
                content = f.read()
                return len(re.findall(r'^\d+\.', content, re.MULTILINE))
                
        except:
            return 0

    def has_more_questions(self) -> bool:
        """Проверяет, есть ли еще вопросы"""
        return self.current_question_index < self._get_questions_count()

    def get_questions_count(self) -> int:
        """Возвращает количество вопросов"""
        return self._get_questions_count()

    def get_progress(self) -> Tuple[int, int]:
        """Возвращает текущий прогресс (текущий вопрос, всего вопросов)"""
        total = self._get_questions_count()
        return (self.current_question_index + 1, total)

    def reset(self):
        """Сброс состояния менеджера практики"""
        self.questions = []
        self.current_question_index = 0
        self.current_lesson_text = ""
        self.current_subject = ""
        self.current_lesson_id = ""