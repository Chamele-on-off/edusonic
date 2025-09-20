import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time

from llm import LLMIntegration
from knowledge.knowledge_base import KnowledgeBase

class PracticeManager:
    def __init__(self, socketio, dialogue_manager):
        self.socketio = socketio
        self.dialogue_manager = dialogue_manager
        self.llm = LLMIntegration()
        self.current_practice = None
        self.current_question_index = 0
        self.practice_questions = []
        self.is_practice_active = False
        self.current_subject = None
        self.lesson_content = []
        self.practice_dir = Path("materials/practice")
        self.practice_dir.mkdir(parents=True, exist_ok=True)
        
    def start_practice_session(self, subject: str, lesson_id: str, lesson_content: List[str]) -> bool:
        """Начинает сессию практики после урока"""
        self.current_subject = subject
        self.lesson_content = lesson_content
        self.is_practice_active = True
        self.current_question_index = 0
        
        # Пытаемся загрузить практику из файла
        practice_file = self.practice_dir / f"{lesson_id}.json"
        if practice_file.exists():
            try:
                with open(practice_file, 'r', encoding='utf-8') as f:
                    practice_data = json.load(f)
                self.practice_questions = practice_data.get("questions", [])
                print(f"Загружено {len(self.practice_questions)} вопросов из файла практики")
            except Exception as e:
                print(f"Ошибка загрузки файла практики: {e}")
                self.practice_questions = []
        else:
            print("Файл практики не найден, будет сгенерирована практика на лету")
            self.practice_questions = []
        
        # Если нет вопросов в файле, генерируем их через LLM
        if not self.practice_questions:
            self._generate_practice_questions(lesson_id, lesson_content)
        
        if self.practice_questions:
            self.current_practice = {
                "subject": subject,
                "lesson_id": lesson_id,
                "start_time": datetime.now().isoformat(),
                "total_questions": len(self.practice_questions),
                "correct_answers": 0
            }
            return True
        else:
            print("Не удалось создать вопросы для практики")
            return False
    
    def _generate_practice_questions(self, lesson_id: str, lesson_content: List[str]) -> None:
        """Генерирует вопросы для практики с помощью LLM"""
        try:
            print(f"Генерация практических вопросов для урока: {lesson_id}")
            
            # Формируем контент урока для промпта
            lesson_text = "\n".join(lesson_content[:10])  # Берем первые 10 абзацев
            
            system_prompt = """Ты - эксперт по созданию образовательных материалов. 
Создай 5-7 практических вопросов и заданий на основе предоставленного урока.
Формат вывода должен быть строго в JSON:

{
  "questions": [
    {
      "type": "multiple_choice|open_ended|true_false",
      "question": "текст вопроса",
      "options": ["вариант1", "вариант2", "вариант3", "вариант4"],  # только для multiple_choice
      "correct_answer": "правильный ответ или индекс варианта",
      "explanation": "объяснение правильного ответа"
    }
  ]
}

Типы вопросов:
- multiple_choice: вопросы с выбором из нескольких вариантов
- open_ended: открытые вопросы, требующие развернутого ответа
- true_false: вопросы на истинность/ложность

Вопросы должны быть:
1. Релевантными содержанию урока
2. Разнообразными по типам
3. Адаптированными для проверки понимания
4. С четкими правильными ответами
5. С полезными объяснениями"""

            # Запрос к LLM
            response = self.llm._query_llm_api(
                prompt=f"Создай практические вопросы на основе этого урока:\n\n{lesson_text}",
                context="",
                subject=self.current_subject,
                system_prompt=system_prompt,
                max_tokens=2000
            )
            
            if response:
                # Пытаемся извлечь JSON из ответа
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    practice_data = json.loads(json_match.group())
                    self.practice_questions = practice_data.get("questions", [])
                    
                    # Сохраняем сгенерированные вопросы для будущего использования
                    self._save_generated_practice(lesson_id, self.practice_questions)
                    
                    print(f"Сгенерировано {len(self.practice_questions)} вопросов для практики")
                else:
                    print("Не удалось извлечь JSON из ответа LLM")
                    # Fallback: создаем простые вопросы на основе контента
                    self._create_fallback_questions(lesson_content)
            else:
                print("LLM не вернул ответ для генерации вопросов")
                self._create_fallback_questions(lesson_content)
                
        except Exception as e:
            print(f"Ошибка генерации вопросов практики: {e}")
            self._create_fallback_questions(lesson_content)
    
    def _create_fallback_questions(self, lesson_content: List[str]) -> None:
        """Создает простые вопросы на основе контента урока (fallback)"""
        self.practice_questions = []
        
        # Берем ключевые предложения из первых 5 абзацев
        for i, paragraph in enumerate(lesson_content[:5]):
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences[:2]:  # Берем первые 2 предложения из каждого абзаца
                if len(sentence.split()) > 5:  # Только предложения достаточной длины
                    question = {
                        "type": "open_ended",
                        "question": f"Расскажите подробнее о: {sentence}",
                        "correct_answer": "Правильный ответ основан на содержании урока",
                        "explanation": "Это вопрос на проверку понимания материала урока."
                    }
                    self.practice_questions.append(question)
        
        if not self.practice_questions:
            # Минимальный fallback
            self.practice_questions = [
                {
                    "type": "open_ended",
                    "question": "Что вы запомнили из этого урока?",
                    "correct_answer": "Ответ должен содержать ключевые моменты урока",
                    "explanation": "Это общий вопрос на проверку понимания основного содержания."
                }
            ]
    
    def _save_generated_practice(self, lesson_id: str, questions: List[Dict]) -> None:
        """Сохраняет сгенерированные вопросы в файл"""
        try:
            practice_data = {
                "lesson_id": lesson_id,
                "subject": self.current_subject,
                "generated_date": datetime.now().isoformat(),
                "questions": questions,
                "metadata": {
                    "type": "llm_generated",
                    "total_questions": len(questions)
                }
            }
            
            practice_file = self.practice_dir / f"{lesson_id}.json"
            with open(practice_file, 'w', encoding='utf-8') as f:
                json.dump(practice_data, f, ensure_ascii=False, indent=2)
            
            print(f"Практические вопросы сохранены в: {practice_file}")
            
        except Exception as e:
            print(f"Ошибка сохранения практических вопросов: {e}")
    
    def get_next_question(self) -> Optional[Dict]:
        """Возвращает следующий вопрос практики"""
        if not self.is_practice_active or self.current_question_index >= len(self.practice_questions):
            return None
        
        question = self.practice_questions[self.current_question_index]
        return question
    
    def check_answer(self, user_answer: str, question: Dict) -> Tuple[bool, str]:
        """Проверяет ответ пользователя с помощью LLM"""
        try:
            if question["type"] == "multiple_choice":
                # Для вопросов с выбором проверяем точное соответствие
                correct_index = int(question["correct_answer"])
                user_choice = user_answer.strip().lower()
                
                # Пытаемся извлечь выбор пользователя (A, B, C, D или 1, 2, 3, 4)
                choice_map = {"a": 0, "b": 1, "c": 2, "d": 3, "1": 0, "2": 1, "3": 2, "4": 3}
                if user_choice in choice_map:
                    user_index = choice_map[user_choice]
                    is_correct = user_index == correct_index
                else:
                    # Если не удалось распознать выбор, используем LLM для проверки
                    return self._check_with_llm(user_answer, question)
                
                explanation = question.get("explanation", "Правильный ответ объясняется в материалах урока.")
                return is_correct, explanation
                
            elif question["type"] == "true_false":
                # Для вопросов истина/ложь
                correct_answer = question["correct_answer"].lower()
                user_answer_lower = user_answer.strip().lower()
                
                truth_map = {
                    "да": "true", "верно": "true", "правильно": "true", "истина": "true",
                    "нет": "false", "неверно": "false", "ложь": "false", "неправильно": "false"
                }
                
                normalized_user = truth_map.get(user_answer_lower, user_answer_lower)
                normalized_correct = truth_map.get(correct_answer, correct_answer)
                
                is_correct = normalized_user == normalized_correct
                explanation = question.get("explanation", "Правильный ответ объясняется в материалах урока.")
                return is_correct, explanation
                
            else:
                # Для открытых вопросов используем LLM для проверки
                return self._check_with_llm(user_answer, question)
                
        except Exception as e:
            print(f"Ошибка проверки ответа: {e}")
            # Fallback: считаем ответ неправильным
            return False, "Произошла ошибка при проверке ответа. Попробуйте ответить еще раз."
    
    def _check_with_llm(self, user_answer: str, question: Dict) -> Tuple[bool, str]:
        """Использует LLM для проверки открытых ответов"""
        try:
            prompt = f"""
            Вопрос: {question['question']}
            Правильный ответ: {question.get('correct_answer', 'Не указан')}
            Ответ ученика: {user_answer}
            
            Проанализируй ответ ученика и определи, является ли он правильным или близким к правильному.
            Учти возможные синонимы и разные формулировки правильной идеи.
            
            Верни ответ в формате JSON:
            {{
                "is_correct": true/false,
                "feedback": "конструктивная обратная связь с объяснением"
            }}
            """
            
            system_prompt = """Ты - опытный учитель, который проверяет ответы учеников.
            Будь справедливым и конструктивным. Учитывай, что ответ может быть правильным, 
            но сформулированным другими словами. Если ответ частично правильный, укажи это в обратной связи."""
            
            response = self.llm._query_llm_api(
                prompt=prompt,
                context="",
                subject=self.current_subject,
                system_prompt=system_prompt,
                max_tokens=500
            )
            
            if response:
                # Пытаемся извлечь JSON из ответа
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return result.get("is_correct", False), result.get("feedback", "Ответ проверен.")
            
            # Fallback если LLM не ответил
            user_lower = user_answer.lower()
            correct_lower = question.get('correct_answer', '').lower()
            
            # Простая проверка на совпадение ключевых слов
            keywords = correct_lower.split()
            matches = sum(1 for word in keywords if word in user_lower and len(word) > 3)
            
            if matches >= len(keywords) * 0.5:  # Хотя бы 50% ключевых слов
                return True, "Ответ в целом правильный, но мог бы быть более точным."
            else:
                return False, question.get("explanation", "Ответ не совсем точный. Обратитесь к материалу урока.")
                
        except Exception as e:
            print(f"Ошибка проверки ответа через LLM: {e}")
            return False, "Не удалось проверить ответ. Попробуйте ответить еще раз."
    
    def process_student_response(self, room_id: str, user_answer: str) -> None:
        """Обрабатывает ответ ученика во время практики"""
        if not self.is_practice_active:
            return
        
        current_question = self.get_next_question()
        if not current_question:
            return
        
        is_correct, feedback = self.check_answer(user_answer, current_question)
        
        # Обновляем статистику
        if is_correct:
            self.current_practice["correct_answers"] += 1
        
        # Отправляем feedback ученику
        if is_correct:
            success_messages = [
                "Правильно! Отличная работа!",
                "Верно! Вы хорошо усвоили материал!",
                "Правильный ответ! Так держать!",
                "Верно! Переходим к следующему вопросу.",
                "Правильно! Вы молодец!"
            ]
            message = f"{random.choice(success_messages)} {feedback}"
        else:
            correction_messages = [
                "Почти правильно, но есть ошибка.",
                "Не совсем верно. Давайте разберем:",
                "Есть небольшая неточность:",
                "Ответ требует уточнения:"
            ]
            message = f"{random.choice(correction_messages)} {feedback}"
        
        # Озвучиваем feedback
        self.socketio.emit('speech_text', {
            'text': f"Учитель: {message}",
            'sid': 'teacher',
            'is_teacher': True
        }, room=room_id)
        
        # Озвучиваем через TTS
        self.socketio.emit('generate_speech', {
            'room_id': room_id,
            'text': message,
            'voice': 'female',
            'is_teacher': True
        }, room=room_id)
        
        # Переходим к следующему вопросу или завершаем практику
        self.current_question_index += 1
        next_question = self.get_next_question()
        
        if next_question:
            # Задаем следующий вопрос
            self.ask_question(room_id, next_question)
        else:
            # Завершаем практику
            self.end_practice_session(room_id)
    
    def ask_question(self, room_id: str, question: Dict) -> None:
        """Задает вопрос ученику"""
        question_text = question["question"]
        
        if question["type"] == "multiple_choice":
            options = question.get("options", [])
            if options:
                options_text = "\n".join([f"{chr(65+i)}) {option}" for i, option in enumerate(options)])
                question_text = f"{question_text}\n\nВарианты ответов:\n{options_text}"
        
        # Отправляем вопрос
        self.socketio.emit('speech_text', {
            'text': f"Учитель: {question_text}",
            'sid': 'teacher',
            'is_teacher': True
        }, room=room_id)
        
        # Озвучиваем вопрос
        self.socketio.emit('generate_speech', {
            'room_id': room_id,
            'text': question_text,
            'voice': 'female',
            'is_teacher': True
        }, room=room_id)
        
        # Отправляем вопрос на виртуальную доску
        self.socketio.emit('practice_question', {
            'room_id': room_id,
            'question': question,
            'question_number': self.current_question_index + 1,
            'total_questions': len(self.practice_questions)
        }, room=room_id)
    
    def end_practice_session(self, room_id: str) -> None:
        """Завершает сессию практики и показывает результаты"""
        if not self.is_practice_active:
            return
        
        total = self.current_practice["total_questions"]
        correct = self.current_practice["correct_answers"]
        score = (correct / total) * 100 if total > 0 else 0
        
        if score >= 80:
            evaluation = "Отлично! Вы прекрасно усвоили материал!"
        elif score >= 60:
            evaluation = "Хорошо! Вы хорошо поняли основные моменты."
        elif score >= 40:
            evaluation = "Удовлетворительно. Рекомендуется повторить материал."
        else:
            evaluation = "Нужно повторить материал. Обратите внимание на основные понятия."
        
        result_message = (
            f"Практика завершена! Ваш результат: {correct} из {total} "
            f"({score:.1f}%). {evaluation}"
        )
        
        # Отправляем результаты
        self.socketio.emit('speech_text', {
            'text': f"Учитель: {result_message}",
            'sid': 'teacher',
            'is_teacher': True
        }, room=room_id)
        
        # Озвучиваем результаты
        self.socketio.emit('generate_speech', {
            'room_id': room_id,
            'text': result_message,
            'voice': 'female',
            'is_teacher': True
        }, room=room_id)
        
        # Отправляем результаты на доску
        self.socketio.emit('practice_results', {
            'room_id': room_id,
            'total_questions': total,
            'correct_answers': correct,
            'score': score,
            'evaluation': evaluation
        }, room=room_id)
        
        # Сбрасываем состояние практики
        self.is_practice_active = False
        self.current_practice = None
        self.practice_questions = []
        self.current_question_index = 0
        
        print(f"Практика завершена. Результат: {correct}/{total} ({score:.1f}%)")
    
    def skip_to_practice(self, room_id: str) -> None:
        """Немедленно начинает практику (для тестирования)"""
        if self.dialogue_manager.selected_lesson and self.dialogue_manager.lesson_content:
            lesson_data = self.dialogue_manager.selected_lesson
            success = self.start_practice_session(
                self.dialogue_manager.current_subject,
                lesson_data['id'],
                self.dialogue_manager.lesson_content
            )
            
            if success:
                first_question = self.get_next_question()
                if first_question:
                    self.ask_question(room_id, first_question)
                    return True
        return False