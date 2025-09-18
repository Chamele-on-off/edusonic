import json
from pathlib import Path
from typing import Dict, List, Optional
import re

class PracticeManager:
    def __init__(self, llm_integration):
        self.llm = llm_integration
        self.practice_dir = Path("materials/practice")
        self.questions = []
        self.current_question_index = 0
        
        # Создаем директорию если не существует
        self.practice_dir.mkdir(parents=True, exist_ok=True)

    def load_practice(self, lesson_id: str) -> bool:
        """Загружает практические задания из файла"""
        try:
            practice_file = self.practice_dir / f"{lesson_id}.json"
            
            if not practice_file.exists():
                print(f"Файл практики не найден: {practice_file}")
                return False
            
            with open(practice_file, 'r', encoding='utf-8') as f:
                practice_data = json.load(f)
            
            self.questions = practice_data.get('questions', [])
            self.current_question_index = 0
            
            print(f"Загружено {len(self.questions)} вопросов для практики из файла")
            return True
            
        except Exception as e:
            print(f"Ошибка загрузки практики: {e}")
            return False

    def generate_practice(self, lesson_text: str, subject: str) -> bool:
        """Генерирует практические задания на основе текста урока"""
        try:
            prompt = f"""
            На основе следующего учебного материала создай 3-5 практических вопросов для проверки понимания.
            Вопросы должны быть краткими, понятными и проверять ключевые моменты материала.
            Предмет: {subject}
            
            Верни ответ ТОЛЬКО в виде валидного JSON с следующей структурой:
            {{
                "questions": [
                    {{
                        "question": "текст вопроса",
                        "expected_answer": "короткий правильный ответ"
                    }}
                ]
            }}
            
            Учебный материал:
            {lesson_text[:3000]}  # Ограничиваем длину
            """
            
            system_prompt = """Ты — помощник учителя. Сгенерируй качественные вопросы для проверки понимания материала.
            Отвечай строго в формате JSON без дополнительных пояснений."""
            
            llm_response = self.llm._query_llm_api(
                prompt=prompt,
                context="",
                subject=subject,
                system_prompt=system_prompt,
                max_tokens=1500
            )
            
            if not llm_response:
                print("LLM не вернул ответ для генерации практики")
                return False
            
            # Пытаемся извлечь JSON из ответа
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if not json_match:
                print("Не удалось найти JSON в ответе LLM")
                return False
                
            practice_data = json.loads(json_match.group())
            self.questions = practice_data.get('questions', [])
            self.current_question_index = 0
            
            print(f"Сгенерировано {len(self.questions)} вопросов для практики")
            return True
            
        except Exception as e:
            print(f"Ошибка генерации практики: {e}")
            return False

    def get_question(self, index: int) -> Optional[Dict]:
        """Возвращает вопрос по индексу"""
        if 0 <= index < len(self.questions):
            return self.questions[index]
        return None

    def check_answer(self, question: str, student_answer: str, expected_answer: str, subject: str) -> str:
        """Проверяет ответ ученика через LLM"""
        try:
            prompt = f"""
            Проанализируй ответ ученика на вопрос.
            Вопрос: {question}
            Правильный ответ: {expected_answer}
            Ответ ученика: {student_answer}
            
            Проанализируй, является ли ответ ученика правильным по смыслу. 
            Если ответ неполный или неточный, объясни кратко и вежливо, в чем ошибка.
            Будь поддерживающим учителем. Отвечай на русском языке.
            """
            
            system_prompt = f"""Ты — опытный учитель по предмету {subject}. 
            Анализируй ответы учеников, давай конструктивную обратную связь.
            Если ответ правильный - похвали ученика.
            Если ответ неправильный - вежливо объясни ошибку и дай правильный ответ."""
            
            evaluation = self.llm._query_llm_api(
                prompt=prompt,
                context="",
                subject=subject,
                system_prompt=system_prompt,
                max_tokens=500
            )
            
            return evaluation if evaluation else "Спасибо за ответ! Давайте продолжим."
            
        except Exception as e:
            print(f"Ошибка проверки ответа: {e}")
            return "Спасибо за ответ! Давайте продолжим."

    def save_practice(self, lesson_id: str, practice_data: Dict) -> bool:
        """Сохраняет практические задания в файл"""
        try:
            practice_file = self.practice_dir / f"{lesson_id}.json"
            
            with open(practice_file, 'w', encoding='utf-8') as f:
                json.dump(practice_data, f, ensure_ascii=False, indent=2)
            
            print(f"Практические задания сохранены в: {practice_file}")
            return True
            
        except Exception as e:
            print(f"Ошибка сохранения практики: {e}")
            return False

    def get_questions_count(self) -> int:
        """Возвращает количество вопросов"""
        return len(self.questions)

    def has_more_questions(self) -> bool:
        """Проверяет, есть ли еще вопросы"""
        return self.current_question_index < len(self.questions)

    def reset(self):
        """Сброс состояния менеджера практики"""
        self.questions = []
        self.current_question_index = 0
