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
        self.questions = []  # [{"question": "...", "answer": "...", "type": "text"}]
        self.current_question_index = 0
        self.current_lesson_text = ""
        self.current_subject = ""
        
        # Создаем директорию если не существует
        self.practice_dir.mkdir(parents=True, exist_ok=True)

    def load_practice(self, lesson_id: str) -> bool:
        """Загружает практические задания из текстового файла"""
        try:
            practice_file = self.practice_dir / f"{lesson_id}.txt"
            
            if not practice_file.exists():
                print(f"Файл практики не найден: {practice_file}")
                return False
            
            with open(practice_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.questions = self._parse_practice_content(content)
            self.current_question_index = 0
            
            print(f"Загружено {len(self.questions)} вопросов для практики из файла")
            return len(self.questions) > 0
            
        except Exception as e:
            print(f"Ошибка загрузки практики: {e}")
            return False

    def generate_practice(self, lesson_text: str, subject: str) -> bool:
        """Генерирует практические задания на основе текста урока и сохраняет в файл"""
        try:
            self.current_lesson_text = lesson_text
            self.current_subject = subject
            
            print(f"Генерация практики для предмета: {subject}")
            
            prompt = f"""
            На основе следующего учебного материала создай 5 практических вопросов для проверки понимания.
            
            ТРЕБОВАНИЯ К ФОРМАТУ:
            - Каждый вопрос должен начинаться с "Вопрос: "
            - После вопроса с новой строки должен быть "Ответ: "
            - Между вопросами должна быть пустая строка
            - Вопросы должны проверять ключевые понятия из материала
            - Ответы должны быть краткими и точными
            - Не добавляй никаких дополнительных комментариев
            
            ПРЕДМЕТ: {subject}
            УЧЕБНЫЙ МАТЕРИАЛ:
            {lesson_text[:2000]}
            
            ВЕРНИ ТОЛЬКО ВОПРОСЫ И ОТВЕТЫ В УКАЗАННОМ ФОРМАТЕ БЕЗ ЛИШНИХ СЛОВ.
            """
            
            llm_response = self.llm._query_llm_api(
                prompt=prompt,
                context="",
                subject=subject,
                system_prompt="Ты — помощник учителя. Создавай качественные вопросы для проверки понимания материала. Строго следуй указанному формату.",
                max_tokens=1500
            )
            
            if not llm_response:
                print("LLM не вернул ответ для генерации практики")
                return False
            
            print(f"Получен ответ LLM для практики: {llm_response[:200]}...")
            
            # Парсим вопросы из ответа
            self.questions = self._parse_practice_content(llm_response)
            
            if not self.questions:
                print("Не удалось распарсить вопросы из ответа LLM")
                return False
            
            # Сохраняем в файл
            success = self._save_practice_to_file(lesson_id="generated_practice", content=llm_response)
            
            if success:
                print(f"Сгенерировано и сохранено {len(self.questions)} вопросов для практики")
            else:
                print("Ошибка сохранения практики в файл")
                
            return success
            
        except Exception as e:
            print(f"Ошибка генерации практики: {e}")
            return False

    def _parse_practice_content(self, content: str) -> List[Dict]:
        """Парсит вопросы и ответы из текстового содержимого"""
        questions = []
        
        # Разделяем на блоки вопрос-ответ
        blocks = re.split(r'\n\s*\n', content.strip())
        
        for block in blocks:
            if not block.strip():
                continue
                
            # Ищем вопрос
            question_match = re.search(r'Вопрос:\s*(.+?)(?=\n|$)', block, re.IGNORECASE | re.DOTALL)
            if not question_match:
                # Пробуем другой формат
                question_match = re.search(r'^(.*?)\?', block)
                if not question_match:
                    continue
            
            # Ищем ответ
            answer_match = re.search(r'Ответ:\s*(.+?)(?=\n|$)', block, re.IGNORECASE | re.DOTALL)
            if not answer_match:
                # Пробуем найти ответ после вопроса
                lines = block.split('\n')
                if len(lines) > 1:
                    answer_match = re.match(r'^(.+)$', lines[1].strip()) if len(lines) > 1 else None
            
            if question_match and answer_match:
                question_text = question_match.group(1).strip()
                answer_text = answer_match.group(1).strip()
                
                # Убираем возможные маркеры в начале ответа
                answer_text = re.sub(r'^[-\*•]\s*', '', answer_text)
                
                if question_text and answer_text:
                    questions.append({
                        'question': question_text,
                        'answer': answer_text,
                        'type': 'text'
                    })
        
        print(f"Распаршено вопросов: {len(questions)}")
        return questions

    def _save_practice_to_file(self, lesson_id: str, content: str) -> bool:
        """Сохраняет практику в текстовый файл"""
        try:
            practice_file = self.practice_dir / f"{lesson_id}.txt"
            
            with open(practice_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Практика сохранена в: {practice_file}")
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

    def _is_llm_available(self) -> bool:
        """Проверяет доступность LLM"""
        try:
            # Простая проверка - есть ли API ключ
            return hasattr(self.llm, 'api_key') and bool(self.llm.api_key)
        except:
            return False

    def _evaluate_with_llm(self, question: str, student_answer: str, correct_answer: str) -> str:
        """Оценивает ответ через LLM"""
        try:
            prompt = f"""
            Вопрос: {question}
            Правильный ответ: {correct_answer}
            Ответ ученика: {student_answer}
            
            Проанализируй ответ ученика. Будь поддерживающим учителем:
            - Если ответ правильный или близкий к правильному - похвали
            - Если ответ неполный - дополни и объясни
            - Если ответ неправильный - вежливо исправь
            - Будь кратким (1-2 предложения)
            - Отвечай на русском языке
            """
            
            system_prompt = f"""Ты — опытный учитель по предмету {self.current_subject}. 
            Давай конструктивную обратную связь. Будь доброжелательным и поддерживающим."""
            
            evaluation = self.llm._query_llm_api(
                prompt=prompt,
                context="",
                subject=self.current_subject,
                system_prompt=system_prompt,
                max_tokens=200
            )
            
            return evaluation if evaluation else self._evaluate_with_similarity(student_answer, correct_answer)
            
        except Exception as e:
            print(f"Ошибка оценки через LLM: {e}")
            return self._evaluate_with_similarity(student_answer, correct_answer)

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
        # Приводим к нижнему регистру и убираем лишние пробелы
        text1 = re.sub(r'\s+', ' ', text1.lower()).strip()
        text2 = re.sub(r'\s+', ' ', text2.lower()).strip()
        
        # Используем SequenceMatcher для вычисления схожести
        return SequenceMatcher(None, text1, text2).ratio()

    def move_to_next_question(self) -> bool:
        """Переходит к следующему вопросу"""
        self.current_question_index += 1
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

    def save_practice_for_lesson(self, lesson_id: str, questions: List[Dict]) -> bool:
        """Сохраняет практику для конкретного урока"""
        try:
            content = ""
            for i, qa in enumerate(questions, 1):
                content += f"Вопрос: {qa['question']}\n"
                content += f"Ответ: {qa['answer']}\n\n"
            
            practice_file = self.practice_dir / f"{lesson_id}.txt"
            
            with open(practice_file, 'w', encoding='utf-8') as f:
                f.write(content.strip())
            
            print(f"Практика для урока {lesson_id} сохранена в: {practice_file}")
            return True
            
        except Exception as e:
            print(f"Ошибка сохранения практики для урока: {e}")
            return False