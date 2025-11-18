# student_dialogue.py
import json
from pathlib import Path
from typing import Dict, Optional, List
from difflib import SequenceMatcher
import random
import re
from knowledge.knowledge_base import KnowledgeBase
from llm import LLMIntegration
from config import get_llm_mode, get_dialogue_settings
import time
import threading
from practice_manager import PracticeManager
from dialogue import DialogueManager

class StudentDialogueManager(DialogueManager):
    def __init__(self, socketio, student_data):
        # Инициализируем родительский класс
        super().__init__(socketio)
        
        self.student_data = student_data
        self.is_student_mode = True
        self.auto_selected_subject = student_data.get('subject', 'общее')
        self.current_subject = self.auto_selected_subject  # Устанавливаем предмет сразу
        
        # Специфичные для ученика поля
        self.student_conversation_count = 0
        self.student_lesson_started = False
        self.student_subject_prompted = False
        
        # Адаптированные промты для ученика
        self.student_prompts = self._load_student_prompts()
        
        # Переопределяем локальные шаблоны для ученика
        self.local_patterns.update({
            "привет": self._get_age_appropriate_greeting(),
            "как дела": ["Отлично! А у тебя как настроение?", "Супер! Готов к интересному уроку?"],
            "спасибо": ["Всегда пожалуйста! Рад был помочь.", "Не стоит благодарности! Ты молодец!"],
            "не понимаю": ["Давай разберем этот момент еще раз вместе.", "Хорошо, объясню по-другому, чтобы было понятнее."],
            "повтори": ["Конечно, повторяю для тебя...", "С удовольствием скажу еще раз."],
            "скучно": ["Давай сделаем урок более интересным! Может, викторину?", "Понимаю. Предлагаю сменить активность!"],
            "трудно": ["Не переживай! Сложности - это нормально. Я помогу разобраться.", "Вместе мы обязательно справимся!"],
            "молодец": ["Спасибо! Ты тоже молодец, что так стараешься!", "Спасибо! Рад, что тебе нравится!"],
            "хорошо": ["Прекрасно! Продолжаем наш урок.", "Отлично! Двигаемся дальше."],
            "не знаю": ["Это нормально не знать! Сейчас вместе разберемся.", "Отличный повод узнать что-то новое!"],
            "стоп": ["Останавливаю урок. Скажи 'привет', когда будешь готов продолжить.", "Прерываю чтение. Жду твоей команды."],
            "кто ты": ["Я твой виртуальный учитель с искусственным интеллектом! Готов помочь с обучением.", 
                      "AI-учитель, который сделает твое обучение интересным и веселым."],
            "что умеешь": ["Я могу проводить уроки, отвечать на вопросы, объяснять сложные темы и делать обучение увлекательным!", 
                          "Умею преподавать разные предметы, отвечать на твои вопросы и адаптироваться под твой уровень."],
            "расскажи о себе": ["Я цифровой преподаватель, созданный чтобы сделать образование интересным и доступным!", 
                               "Моя задача - помочь тебе учиться с удовольствием и пониманием."]
        })

    def _load_student_prompts(self) -> Dict:
        """Загружает промты, адаптированные под возраст и уровень ученика"""
        age = int(self.student_data.get('age', 12))
        level = self.student_data.get('level', '5')
        subject = self.current_subject
        
        # Адаптированные приветствия по возрасту
        if age <= 8:
            greeting = "Привет! Я твой весёлый учитель. Давай узнаем что-то интересное вместе!"
            explanation_style = "простыми словами с картинками"
        elif age <= 12:
            greeting = "Привет! Я твой AI-репетитор. Готов к увлекательному уроку?"
            explanation_style = "понятными примерами и сравнениями"
        elif age <= 15:
            greeting = "Здравствуй! Я твой цифровой преподаватель. Начнём наше занятие?"
            explanation_style = "подробно, но доступно"
        else:
            greeting = "Здравствуй! Я твой персональный учитель. Готов углубиться в тему?"
            explanation_style = "углубленно и структурно"
        
        return {
            "greeting": greeting,
            "explanation_style": explanation_style,
            "age_group": self._get_age_group(age),
            "subject": subject
        }

    def _get_age_group(self, age: int) -> str:
        """Определяет возрастную группу"""
        if age <= 8: return "младшая_школа"
        elif age <= 12: return "средняя_школа"
        elif age <= 15: return "старшая_школа"
        else: return "студенты"

    def _get_age_appropriate_greeting(self) -> List[str]:
        """Возвращает приветствия, адаптированные по возрасту"""
        age = int(self.student_data.get('age', 12))
        
        if age <= 8:
            return [
                "Привет! Я твой весёлый учитель. Давай узнаем что-то интересное вместе!",
                "Здравствуй! Я твой помощник в учёбе. Готов к приключениям?",
                "Приветик! Я твой цифровой друг-учитель. Давай учиться весело!"
            ]
        elif age <= 12:
            return [
                "Привет! Я твой AI-репетитор. Готов к увлекательному уроку?",
                "Здравствуй! Я твой виртуальный учитель. Начнём наше путешествие в мир знаний?",
                "Привет! Я твой помощник в учёбе. Давай сделаем этот урок интересным!"
            ]
        elif age <= 15:
            return [
                "Здравствуй! Я твой цифровой преподаватель. Начнём наше занятие?",
                "Привет! Я твой персональный репетитор. Готов погрузиться в тему?",
                "Здравствуй! Я твой AI-учитель. Давай начнём наш урок продуктивно!"
            ]
        else:
            return [
                "Здравствуй! Я твой персональный учитель. Готов углубиться в тему?",
                "Привет! Я твой цифровой преподаватель. Начнём наше занятие?",
                "Здравствуй! Я твой AI-репетитор. Готов к продуктивной работе?"
            ]

    def _adapt_response_to_student(self, response: str) -> str:
        """Адаптирует ответ под уровень и возраст ученика"""
        if not response:
            return response
            
        age = int(self.student_data.get('age', 12))
        
        # Упрощаем язык для младших школьников
        if age <= 10:
            response = self._simplify_language(response)
            
        # Добавляем персонализированное обращение для младших
        if age <= 12 and not response.startswith(("Привет", "Здравствуй")):
            student_name = self.student_data.get('name', '')
            if student_name and len(response) < 100:
                response = f"{student_name}, {response.lower()}"
        
        return response

    def _simplify_language(self, text: str) -> str:
        """Упрощает язык для младших школьников"""
        # Замена сложных слов на простые
        replacements = {
            'осуществлять': 'делать',
            'воспринимать': 'понимать', 
            'преподаватель': 'учитель',
            'образовательный': 'учебный',
            'информационный': 'полезный',
            'деятельность': 'работа',
            'восприятие': 'понимание',
            'осознавать': 'понимать',
            'интеллектуальный': 'умный',
            'познавательный': 'интересный',
            'анализировать': 'разбирать',
            'синтезировать': 'собирать',
            'абстрактный': 'непонятный',
            'концептуальный': 'главный'
        }
        
        for complex_word, simple_word in replacements.items():
            text = text.replace(complex_word, simple_word)
            
        # Упрощаем длинные предложения
        sentences = re.split(r'(?<=[.!?])\s+', text)
        simplified_sentences = []
        
        for sentence in sentences:
            if len(sentence.split()) > 15:  # Если предложение слишком длинное
                # Разбиваем на более короткие
                words = sentence.split()
                mid_point = len(words) // 2
                part1 = ' '.join(words[:mid_point])
                part2 = ' '.join(words[mid_point:])
                simplified_sentences.extend([part1, part2])
            else:
                simplified_sentences.append(sentence)
                
        return ' '.join(simplified_sentences)

    def _handle_llm_dialogue(self, text: str, room_id: str = None) -> Optional[str]:
        """Гарантированная обработка диалога через LLM с контекстом ученика"""
        try:
            # Собираем контекст диалога
            context = self._get_conversation_context()
            
            # Формируем промпт с учетом данных ученика
            age = self.student_data.get('age', '12')
            level = self.student_data.get('level', '5')
            subject = self.current_subject
            
            system_prompt = f"""Ты - дружелюбный учитель для ученика {age} лет, {level} класс. 
Предмет: {subject}. Объясняй {self.student_prompts['explanation_style']}.
Будь кратким и понятным, максимум 2-3 предложения. Отвечай на русском языке."""
            
            # СИНХРОННЫЙ запрос к LLM
            llm_response = self.llm._query_llm_api(
                prompt=text,
                context=context,
                subject=self.current_subject,
                system_prompt=system_prompt,
                max_tokens=150
            )
            
            if llm_response:
                limited_response = self._limit_response_length(
                    llm_response, 
                    self.dialogue_settings.get("max_response_length", 3)
                )
                # Адаптируем ответ под ученика
                return self._adapt_response_to_student(limited_response)
            else:
                print("⚠️ LLM не вернул ответ для диалога ученика")
                
        except Exception as e:
            print(f"❌ Ошибка запроса к LLM для диалога ученика: {e}")
        
        # Fallback для ученика
        return self._get_student_lesson_prompt()

    def _get_student_lesson_prompt(self) -> Optional[str]:
        """Возвращает предложение начать урок для ученика"""
        current_time = time.time()
        if current_time - self.last_subject_prompt_time < self.subject_prompt_cooldown:
            return None
        
        self.last_subject_prompt_time = current_time
        
        # После 2-3 фраз диалога предлагаем начать урок
        if self.student_conversation_count >= 2 and not self.student_subject_prompted:
            self.student_subject_prompted = True
            prompts = [
                f"Отлично! Давайте начнем урок по {self.current_subject}. Готов?",
                f"Прекрасно! Приступаем к уроку по {self.current_subject}. Начинаем?",
                f"Замечательно! Начнем наш урок по {self.current_subject}?",
                f"Отлично познакомились! Готов начать урок по {self.current_subject}?",
                f"Рад нашему знакомству! Приступим к уроку по {self.current_subject}?"
            ]
            return random.choice(prompts)
        
        return None

    def process_input(self, text: str) -> Optional[str]:
        """Обработка входящего текста и генерация ответа для ученика"""
        text_lower = text.lower().strip()
        
        # Увеличиваем счетчик разговора
        self.student_conversation_count += 1
        print(f"🎓 Диалог ученика: счетчик {self.student_conversation_count}, предмет: {self.current_subject}")
        
        # Используем родительскую логику для команд продолжения
        continue_commands = [
            "продолжай", "продолжить", "дальше", "следующий", "вперед", "давай дальше",
            "записал", "понял", "ясно", "ага", "угу", "хорошо", "ок", "ладно", "ясно",
            "готов", "можно дальше", "следующая часть", "продолжаем", "всё", "все"
        ]

        if self.lesson_started and any(cmd in text_lower for cmd in continue_commands):
            next_paragraph = self._get_next_paragraph()
            if next_paragraph:
                print(f"✅ Команда продолжения обработана: '{text_lower}' -> следующий абзац")
                return next_paragraph
            else:
                print("🏁 Урок завершен по команде продолжения")
                return "Урок завершен. Переходим к практике."
        
        self._add_to_conversation_history(text, is_user=True)
        
        # ОСОБАЯ ЛОГИКА ДЛЯ РЕЖИМА УЧЕНИКА - быстро переходим к уроку
        if not self.lesson_started:
            return self._handle_student_mode_input(text, text_lower)
        
        # Для остальных случаев используем родительскую логику
        parent_response = super().process_input(text)
        if parent_response:
            return self._adapt_response_to_student(parent_response)
        return None

    def _handle_student_mode_input(self, text: str, text_lower: str) -> Optional[str]:
        """Обработка ввода в режиме ученика до начала урока"""
        
        # Проверяем, не хочет ли ученик изучить конкретную тему
        if self._check_for_specific_topic_request(text_lower):
            print(f"🎯 Ученик запросил конкретную тему по предмету {self.current_subject}")
            return None
        
        # После 2-3 фраз диалога автоматически предлагаем начать урок
        if self.student_conversation_count >= 2 and not self.student_subject_prompted:
            self.student_subject_prompted = True
            prompt = self._get_student_lesson_prompt()
            if prompt:
                self._add_to_conversation_history(prompt, is_user=False)
                return prompt
        
        # Если ученик соглашается начать урок
        if any(word in text_lower for word in ['да', 'ага', 'угу', 'ладно', 'хорошо', 'начать', 'начнем', 'поехали']):
            return self._start_student_lesson()
        
        # Обычная обработка диалога
        dialogue_response = self._get_dialogue_response(text_lower)
        if dialogue_response:
            adapted_response = self._adapt_response_to_student(dialogue_response)
            self._add_to_conversation_history(adapted_response, is_user=False)
            return adapted_response
        
        # Используем LLM для диалога с адаптацией
        llm_response = self._handle_llm_dialogue(text)
        if llm_response:
            self._add_to_conversation_history(llm_response, is_user=False)
            return llm_response
        
        return None

    def _check_for_specific_topic_request(self, text_lower: str) -> bool:
        """Проверяет, запрашивает ли ученик конкретную тему по выбранному предмету"""
        topic_patterns = [
            r'хочу изучить (.+)',
            r'можешь рассказать про (.+)', 
            r'урок по (.+)',
            r'изучим (.+)',
            r'расскажи про (.+)',
            r'хочу узнать про (.+)',
            r'объясни тему (.+)'
        ]
        
        for pattern in topic_patterns:
            match = re.search(pattern, text_lower)
            if match:
                topic = match.group(1).strip()
                if topic and len(topic) > 2:
                    print(f"🎯 Ученик запросил тему '{topic}' по предмету {self.current_subject}")
                    return True
        return False

    def _start_student_lesson(self) -> str:
        """Начинает урок для ученика по выбранному предмету"""
        print(f"🚀 Начинаем урок для ученика по предмету: {self.current_subject}")
        
        # Используем родительскую логику выбора предмета
        response = self._handle_subject_selection_direct(self.current_subject)
        
        if response is None:
            # Успешно начали урок
            student_name = self.student_data.get('name', '')
            greeting = f"{student_name}, " if student_name else ""
            start_message = f"{greeting}Отлично! Начинаем урок по {self.current_subject}. {self._get_next_paragraph()}"
            self._add_to_conversation_history(start_message, is_user=False)
            return start_message
        
        return response

    def generate_lesson_on_demand(self, topic: str) -> Optional[dict]:
        """Генерирует урок по запрошенной теме с помощью LLM с адаптацией под ученика"""
        try:
            print(f"🎯 Генерация урока для ученика по теме: {topic}")
            
            # Формируем промпт для генерации урока с учетом возраста
            age = self.student_data.get('age', '12')
            level = self.student_data.get('level', '5')
            
            system_prompt = f"""Ты - эксперт по созданию образовательных материалов для учеников {age} лет, {level} класс.
Создай структурированный урок по заданной теме. Урок должен быть:
1. Информативным и точным, но адаптированным под возраст {age} лет
2. Разделен на логические абзацы (разделяй пустыми строками)
3. Использовать простой и понятный язык
4. Содержать практические примеры если уместно
5. Быть увлекательным и интересным

ВАЖНО: Разделяй абзацы ДВУМЯ переводами строки (\\n\\n) для правильного отображения.
Используй {self.student_prompts['explanation_style']}.
Возвращай только текст урока без дополнительных комментариев."""

            # Запрос к LLM с увеличенным количеством токенов
            lesson_content = self.llm._query_llm_api(
                prompt=f"Создай подробный образовательный урок на тему: '{topic}'. Урок должен быть понятным и интересным для ученика.",
                context="",
                subject=self.current_subject,
                system_prompt=system_prompt,
                max_tokens=2500
            )
            
            if not lesson_content:
                print("❌ Ошибка: LLM не вернул содержание урока")
                return None
            
            print(f"✅ Получен контент урока, длина: {len(lesson_content)} символов")
            
            # Убедимся, что есть правильное разделение на абзацы
            if '\n\n' not in lesson_content:
                print("⚠️ В ответе нет двойных переводов строк, добавляем...")
                sentences = re.split(r'(?<=[.!?])\s+', lesson_content)
                lesson_content = '\n\n'.join(sentences)
            
            # Создаем файл урока
            lesson_id = f"student_{self.student_data.get('student_id', 'unknown')}_{topic.lower().replace(' ', '_')}_{int(time.time())}"
            filename = f"{lesson_id}.txt"
            lesson_path = self.lessons_dir / filename
            
            # Записываем контент в файл
            with open(lesson_path, 'w', encoding='utf-8') as f:
                f.write(f"Урок по теме: {topic}\n\n")
                f.write(lesson_content)
            
            print(f"✅ Файл урока создан: {lesson_path}")
            
            # Добавляем в список уроков
            lesson_data = {
                'id': lesson_id,
                'title': f"Урок по теме: {topic}",
                'file_path': lesson_path,
                'type': 'text',
                'is_generated': True,
                'student_specific': True
            }
            
            if self.current_subject not in self.lessons:
                self.lessons[self.current_subject] = []
            self.lessons[self.current_subject].append(lesson_data)
            
            print(f"✅ Урок успешно сгенерирован и добавлен в список: {lesson_id}")
            return lesson_data
            
        except Exception as e:
            print(f"❌ Ошибка генерации урока: {e}")
            return None

    def _evaluate_and_generate_next(self, student_answer: str) -> str:
        """Оценивает ответ и возвращает следующий вопрос с адаптацией для ученика"""
        print(f"🔍 Обработка ответа ученика: '{student_answer}'")
        
        if not self.practice_active:
            print("❌ Практика не активна")
            return "Практика не активна."
        
        # ПРОВЕРЯЕМ, НЕ ЯВЛЯЕТСЯ ЛИ ОТВЕТ КОМАНДОЙ
        if any(cmd in student_answer.lower() for cmd in ['продолжай', 'дальше', 'следующий']):
            print(f"🔇 Игнорирую команду вместо ответа: {student_answer}")
            next_question = self.practice_manager.get_next_question()
            if next_question:
                return f"Это похоже на команду. Пожалуйста, дай ответ на вопрос. Следующий вопрос: {next_question}"
            else:
                self._end_practice_session()
                return "Практика завершена."
        
        print(f"🎯 Оценка ответа и получение следующего вопроса...")
        
        current_question = self.current_practice_question
        if not current_question:
            print("❌ Нет текущего вопроса практики")
            self._end_practice_session()
            return "Практика завершена."
        
        # УВЕЛИЧИВАЕМ СЧЕТЧИК ОТВЕТОВ
        self.current_question_index += 1
        print(f"📊 Текущий номер вопроса: {self.current_question_index}/{self.max_questions}")
        
        # ПРОВЕРЯЕМ ЛИМИТ ВОПРОСОВ
        if self.current_question_index >= self.max_questions:
            print(f"🏁 Достигнут лимит вопросов: {self.current_question_index}/{self.max_questions}")
            self._end_practice_session()
            student_name = self.student_data.get('name', '')
            greeting = f"{student_name}, " if student_name else ""
            return f"{greeting}Отлично! Ты ответил на все вопросы практики. Урок завершен!"
        
        # ИСПОЛЬЗУЕМ НОВЫЙ МЕТОД: оценка + следующий вопрос
        feedback, next_question = self.practice_manager.evaluate_and_continue(
            student_answer, 
            current_question["question"]
        )
        
        # АДАПТИРУЕМ ОБРАТНУЮ СВЯЗЬ ДЛЯ УЧЕНИКА
        if not feedback or "Хороший вопрос! Давайте разберем эту тему подробнее" in feedback:
            feedback = "Спасибо за ответ! Переходим к следующему вопросу."
        else:
            # Упрощаем обратную связь для младших школьников
            age = int(self.student_data.get('age', 12))
            if age <= 10:
                feedback = self._simplify_language(feedback)
        
        if next_question:
            # Обновляем текущий вопрос
            self.current_practice_question = {
                "id": self.current_question_index + 1,
                "question": next_question,
                "answer": ""
            }
            self.waiting_for_answer = True
            
            response = f"{feedback}. Следующий вопрос: {next_question}"
            print(f"➡️ Следующий вопрос получен: {next_question[:80]}...")
            return response
        else:
            print("❌ Не удалось получить следующий вопрос")
            self._end_practice_session()
            return f"{feedback}. Практика завершена!"

    def handle_question_during_lesson(self, question: str) -> str:
        """Обработка вопросов ученика во время урока с адаптацией"""
        if not question.strip():
            return "Повтори вопрос пожалуйста, я не расслышал."
            
        question_lower = question.lower().strip()
        
        if self.visualization_enabled:
            context = " ".join(self.lesson_content[max(0, self.current_paragraph-2):self.current_paragraph])
            self._generate_visualization(question, context)
        
        print(f"Немедленная обработка вопроса ученика: '{question}'")
        
        # Используем родительскую логику, но адаптируем ответ
        parent_response = super().handle_question_during_lesson(question)
        if parent_response:
            return self._adapt_response_to_student(parent_response)
        
        return "Интересный вопрос! Давай обсудим его после завершения текущего материала."

    def reset(self):
        """Полный сброс диалог менеджера"""
        super().reset()
        
        # Сброс счетчиков ученика
        self.student_conversation_count = 0
        self.student_lesson_started = False
        self.student_subject_prompted = False

    def get_conversation_stats(self) -> Dict:
        """Возвращает статистику диалога"""
        stats = super().get_conversation_stats()
        stats.update({
            "student_conversation_count": self.student_conversation_count,
            "student_data": {
                "name": self.student_data.get('name'),
                "age": self.student_data.get('age'),
                "level": self.student_data.get('level')
            }
        })
        return stats

    def debug_info(self) -> Dict:
        """Возвращает отладочную информацию"""
        info = super().debug_info()
        info.update({
            "student_data": self.student_data,
            "student_conversation_count": self.student_conversation_count,
            "student_lesson_started": self.student_lesson_started,
            "student_subject_prompted": self.student_subject_prompted,
            "age_group": self.student_prompts.get('age_group', 'unknown')
        })
        return info

    def get_system_status(self) -> Dict:
        """Возвращает общий статус системы"""
        status = super().get_system_status()
        status["student_dialogue_manager"] = {
            "student_conversation_count": self.student_conversation_count,
            "student_data": {
                "name": self.student_data.get('name'),
                "age": self.student_data.get('age'),
                "level": self.student_data.get('level'),
                "subject": self.current_subject
            },
            "age_group": self.student_prompts.get('age_group', 'unknown')
        }
        return status


# Создаем глобальный экземпляр для тестирования
if __name__ == "__main__":
    # Тестирование базовой функциональности
    print("🧪 Тестирование StudentDialogueManager...")
    
    # Тестовые данные ученика
    test_student_data = {
        'name': 'Анна',
        'age': '12',
        'level': '5',
        'subject': 'математика',
        'student_id': 'test_student_123'
    }
    
    # Создаем экземпляр менеджера
    sdm = StudentDialogueManager(None, test_student_data)
    
    # Тест приветствия
    response = sdm.process_input("привет")
    print(f"👋 Ответ на приветствие: {response}")
    
    # Тест статуса системы
    status = sdm.get_system_status()
    print(f"📊 Статус системы: {status}")
    
    print("✅ Тестирование StudentDialogueManager завершено!")
