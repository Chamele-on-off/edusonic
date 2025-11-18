# student_dialogue.py
import json
from pathlib import Path
from typing import Dict, Optional, List
import random
import re
import time
import threading
from difflib import SequenceMatcher

from knowledge.knowledge_base import KnowledgeBase
from llm import LLMIntegration
from config import get_llm_mode, get_dialogue_settings
from practice_manager import PracticeManager

class StudentDialogueManager:
    def __init__(self, socketio, student_data):
        self.socketio = socketio
        self.student_data = student_data  # {name, age, level, subject, student_id}
        self.is_student_mode = True
        
        # Базовые настройки диалога
        self.dialogue_states = {
            "greeting": self._handle_greeting,
            "subject_selection": self._handle_subject_selection,
            "lesson_reading": self._handle_lesson_reading,
            "practice_session": self._handle_practice_session
        }
        self.current_state = "greeting"
        self.current_subject = student_data.get('subject', 'общее')
        self.selected_lesson = None
        self.lesson_started = False
        self.lesson_content = []
        self.current_paragraph = 0
        self.lessons_dir = Path("lessons")
        self.knowledge_base = None
        
        # 🔥 ВАЖНОЕ ИСПРАВЛЕНИЕ: Правильная инициализация LLM
        self.llm = LLMIntegration()
        self.llm_query_mode = get_llm_mode()
        
        # 🔥 ВАЖНОЕ ИСПРАВЛЕНИЕ: Правильная инициализация PracticeManager
        self.practice_manager = PracticeManager(self.llm)
        
        self.conversation_counter = 0
        self.dialogue_settings = get_dialogue_settings()
        self.conversation_history = []
        self.conversation_context = []
        self.room_id = None
        
        # Поля для практики
        self.practice_active = False
        self.current_question_index = 0
        self.waiting_for_answer = False
        self.current_practice_question = None
        self.max_questions = 5
        
        # Поля для ученика
        self.student_conversation_count = 0
        self.student_lesson_started = False
        self.student_subject_prompted = False
        
        # Поля для визуализации
        self.visualization_enabled = True
        self.last_visualization_time = 0
        self.visualization_cooldown = 5
        self.visualization_counter = 0
        self.paragraphs_since_last_viz = 0
        self.viz_paragraph_interval = 2
        
        # Адаптированные промты для ученика
        self.student_prompts = self._load_student_prompts()
        
        # Загрузка уроков и знаний
        self._load_lessons()
        self.dialogue_knowledge = self._load_dialogue_knowledge()
        self.local_patterns = self._get_student_patterns()
        
        print(f"🎓 StudentDialogueManager инициализирован для {student_data.get('name')} с предметом {self.current_subject}")

    def _load_student_prompts(self) -> Dict:
        """Загружает промты, адаптированные под возраст и уровень ученика"""
        age = int(self.student_data.get('age', 12))
        level = self.student_data.get('level', '5')
        subject = self.student_data.get('subject', 'общее')
        
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
            "subject_specific": self._get_subject_specific_prompts(subject)
        }

    def _get_age_group(self, age: int) -> str:
        """Определяет возрастную группу"""
        if age <= 8: return "младшая_школа"
        elif age <= 12: return "средняя_школа"
        elif age <= 15: return "старшая_школа"
        else: return "студенты"

    def _get_subject_specific_prompts(self, subject: str) -> Dict:
        """Возвращает предмет-специфичные промты"""
        subject_prompts = {
            "математика": {
                "difficulty": "с простыми примерами",
                "approach": "практико-ориентированный",
                "examples": "из реальной жизни"
            },
            "физика": {
                "difficulty": "с наглядными экспериментами", 
                "approach": "исследовательский",
                "examples": "физических явлений"
            },
            "химия": {
                "difficulty": "с безопасными опытами",
                "approach": "экспериментальный", 
                "examples": "химических реакций"
            },
            "биология": {
                "difficulty": "с интересными фактами",
                "approach": "познавательный",
                "examples": "из мира живой природы"
            },
            "история": {
                "difficulty": "с увлекательными историями",
                "approach": "повествовательный",
                "examples": "исторических событий"
            },
            "обществознание": {
                "difficulty": "с актуальными примерами",
                "approach": "дискуссионный", 
                "examples": "из современной жизни"
            },
            "литература": {
                "difficulty": "с цитатами и отрывками",
                "approach": "аналитический",
                "examples": "литературных произведений"
            },
            "русский язык": {
                "difficulty": "с практическими заданиями",
                "approach": "системный",
                "examples": "языковых конструкций"
            },
            "английский язык": {
                "difficulty": "с разговорными фразами", 
                "approach": "коммуникативный",
                "examples": "из повседневного общения"
            },
            "география": {
                "difficulty": "с картами и фотографиями",
                "approach": "исследовательский",
                "examples": "географических объектов"
            }
        }
        
        return subject_prompts.get(subject, {
            "difficulty": "с интересными примерами",
            "approach": "адаптивный",
            "examples": "из разных областей"
        })

    def _get_student_patterns(self):
        """Возвращает шаблоны, адаптированные для учеников"""
        return {
            "привет": self._get_age_appropriate_greeting(),
            "как дела": ["Отлично! А у тебя как настроение?", "Прекрасно! Готов к уроку?"],
            "спасибо": ["Всегда пожалуйста! Рад был помочь.", "Не стоит благодарности! Ты молодец!"],
            "не понимаю": ["Давай разберем этот момент еще раз вместе.", "Хорошо, объясню по-другому."],
            "повтори": ["Конечно, повторяю...", "С удовольствием скажу еще раз."],
            "скучно": ["Давай сделаем урок более интересным! Может, викторину?", "Понимаю. Предлагаю сменить активность!"],
            "трудно": ["Не переживай! Сложности - это нормально. Я помогу разобраться.", "Вместе мы обязательно справимся!"],
            "молодец": ["Спасибо! Ты тоже молодец, что так стараешься!", "Спасибо! Рад, что тебе нравится!"],
            "хорошо": ["Прекрасно! Продолжаем наш урок.", "Отлично! Двигаемся дальше."],
            "не знаю": ["Это нормально не знать! Сейчас вместе разберемся.", "Отличный повод узнать что-то новое!"],
            "стоп": ["Останавливаю урок. Скажи 'привет', когда будешь готов продолжить.", "Прерываю чтение."],
            "кто ты": ["Я твой виртуальный учитель с искусственным интеллектом! Готов помочь с обучением.", 
                      "AI-учитель, который сделает твое обучение интересным и веселым."],
            "что умеешь": ["Я могу проводить уроки, отвечать на вопросы, объяснять сложные темы и делать обучение увлекательным!", 
                          "Умею преподавать разные предметы, отвечать на твои вопросы и адаптироваться под твой уровень."],
            "расскажи о себе": ["Я цифровой преподаватель, созданный чтобы сделать образование интересным и доступным!", 
                               "Моя задача - помочь тебе учиться с удовольствием и пониманием."]
        }

    def _get_age_appropriate_greeting(self) -> List[str]:
        """Возвращает приветствия, адаптированные по возрасту"""
        age = int(self.student_data.get('age', 12))
        name = self.student_data.get('name', '')
        
        if age <= 8:
            return [
                f"Привет{', ' + name if name else ''}! Я твой весёлый учитель. Давай узнаем что-то интересное вместе!",
                f"Здравствуй{', ' + name if name else ''}! Я твой помощник в учёбе. Готов к приключениям?",
                f"Приветик{', ' + name if name else ''}! Я твой цифровой друг-учитель. Давай учиться весело!"
            ]
        elif age <= 12:
            return [
                f"Привет{', ' + name if name else ''}! Я твой AI-репетитор. Готов к увлекательному уроку?",
                f"Здравствуй{', ' + name if name else ''}! Я твой виртуальный учитель. Начнём наше путешествие в мир знаний?",
                f"Привет{', ' + name if name else ''}! Я твой помощник в учёбе. Давай сделаем этот урок интересным!"
            ]
        elif age <= 15:
            return [
                f"Здравствуй{', ' + name if name else ''}! Я твой цифровой преподаватель. Начнём наше занятие?",
                f"Привет{', ' + name if name else ''}! Я твой персональный репетитор. Готов погрузиться в тему?",
                f"Здравствуй{', ' + name if name else ''}! Я твой AI-учитель. Давай начнём наш урок продуктивно!"
            ]
        else:
            return [
                f"Здравствуй{', ' + name if name else ''}! Я твой персональный учитель. Готов углубиться в тему?",
                f"Привет{', ' + name if name else ''}! Я твой цифровой преподаватель. Начнём наше занятие?",
                f"Здравствуй{', ' + name if name else ''}! Я твой AI-репетитор. Готов к продуктивной работе?"
            ]

    def _load_dialogue_knowledge(self) -> Dict:
        """Загрузка диалоговых шаблонов"""
        try:
            dialogue_path = Path("knowledge/dialogue_knowledge.json")
            if dialogue_path.exists():
                with open(dialogue_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Ошибка загрузки диалоговых шаблонов: {e}")
        
        return {}

    def _load_lessons(self):
        """Загружает список доступных уроков"""
        self.lessons = {}
        try:
            if not self.lessons_dir.exists():
                self.lessons_dir.mkdir(parents=True)
                return
                
            for lesson_file in self.lessons_dir.glob("*.txt"):
                try:
                    subject = self._detect_subject(lesson_file.stem)
                    
                    if subject not in self.lessons:
                        self.lessons[subject] = []
                    
                    self.lessons[subject].append({
                        'id': lesson_file.stem,
                        'title': lesson_file.stem.replace('_', ' ').title(),
                        'file_path': lesson_file,
                        'type': 'text'
                    })
                except Exception as e:
                    print(f"Ошибка загрузки урока {lesson_file}: {e}")
                    
        except Exception as e:
            print(f"Ошибка доступа к папке уроков: {e}")

    def _detect_subject(self, filename: str) -> str:
        """Определяет предмет по названию файла"""
        filename_lower = filename.lower()
        subject_map = {
            'math': 'математика', 'математика': 'математика',
            'physics': 'физика', 'физика': 'физика',
            'chemistry': 'химия', 'химия': 'химия',
            'biology': 'биология', 'биология': 'биология',
            'history': 'история', 'история': 'история',
            'social': 'обществознание', 'обществознание': 'обществознание',
            'literature': 'литература', 'литература': 'литература',
            'russian': 'русский язык', 'русский': 'русский язык',
            'english': 'английский язык', 'английский': 'английский язык',
            'geography': 'география', 'география': 'география'
        }
        
        for key, value in subject_map.items():
            if key in filename_lower:
                return value
        return "общее"

    def _load_lesson_content(self, lesson_file: Path) -> List[str]:
        """Загружает содержание урока из текстового файла"""
        try:
            print(f"📖 Загрузка урока для ученика из файла: {lesson_file}")
            
            if not lesson_file.exists():
                print(f"❌ Файл урока не существует: {lesson_file}")
                return ["Урок временно недоступен. Давайте поговорим на эту тему!"]
                
            with open(lesson_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Очистка содержания
            content = self._clean_lesson_content(content)
            
            # Разбиваем на абзацы
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            # Если абзацев нет, разбиваем на предложения
            if not paragraphs:
                sentences = re.split(r'(?<=[.!?])\s+', content)
                current_paragraph = []
                paragraphs = []
                
                for sentence in sentences:
                    if sentence.strip():
                        current_paragraph.append(sentence.strip())
                        if len(current_paragraph) >= 2:
                            paragraphs.append(' '.join(current_paragraph))
                            current_paragraph = []
                
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
            
            print(f"✅ Урок для ученика загружен, абзацев: {len(paragraphs)}")
            return paragraphs if paragraphs else ["Содержание урока пусто."]
            
        except Exception as e:
            print(f"❌ Ошибка загрузки урока для ученика: {e}")
            return ["Ошибка загрузки урока."]

    def _clean_lesson_content(self, content: str) -> str:
        """Очистка содержания урока от лишнего форматирования"""
        if not content:
            return content
        
        # Удаляем маркеры форматирования
        content = re.sub(r'[\*\#]{1,}', '', content)
        content = re.sub(r'\-\-\-+', '', content)
        content = re.sub(r'\+\+\+', '', content)
        
        # Удаляем HTML-теги если есть
        content = re.sub(r'<[^>]+>', '', content)
        
        # Нормализуем переводы строк
        content = re.sub(r'\r\n', '\n', content)
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        # Удаляем начальные/конечные пробелы
        content = content.strip()
        
        return content

    def _add_to_conversation_history(self, text: str, is_user: bool = True):
        """Добавляет реплику в историю диалога"""
        self.conversation_history.append({
            "text": text,
            "is_user": is_user,
            "timestamp": time.time()
        })
        
        # Ограничиваем размер истории
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
            
        # Обновляем контекст
        if is_user:
            user_messages = [msg['text'] for msg in self.conversation_history if msg['is_user']]
            self.conversation_context = user_messages[-3:] if len(user_messages) > 3 else user_messages

    def _get_conversation_context(self) -> str:
        """Возвращает контекст диалога"""
        if not self.conversation_history:
            return ""
            
        context = []
        for msg in self.conversation_history[-6:]:
            speaker = "Ученик" if msg["is_user"] else "Учитель"
            context.append(f"{speaker}: {msg['text']}")
        
        return "\n".join(context)

    def _limit_response_length(self, response: str, max_sentences: int = 3) -> str:
        """Ограничивает длину ответа количеством предложений"""
        if not response:
            return response
            
        sentences = re.split(r'(?<=[.!?])\s+', response)
        if len(sentences) > max_sentences:
            return ' '.join(sentences[:max_sentences])
        return response

    def _get_dialogue_response(self, text: str) -> Optional[str]:
        """Поиск ответа в диалоговых шаблонов"""
        text_lower = text.lower().strip()
        
        # Поиск в локальных шаблонах
        for pattern, responses in self.local_patterns.items():
            if pattern in text_lower:
                if isinstance(responses, list):
                    return random.choice(responses)
                return responses
        
        return None

    def _handle_llm_dialogue(self, text: str) -> Optional[str]:
        """Обработка диалога через LLM"""
        try:
            # Собираем контекст диалога
            context = self._get_conversation_context()
            
            # Формируем промпт с учетом данных ученика
            age = self.student_data.get('age', '12')
            level = self.student_data.get('level', '5')
            subject = self.current_subject
            
            system_prompt = f"""Ты - дружелюбный учитель для ученика {age} лет, {level} класс. 
Предмет: {subject}. Объясняй {self.student_prompts['explanation_style']}.
Используй {self.student_prompts['subject_specific']['examples']}.
Будь кратким и понятным, максимум 2-3 предложения. Отвечай на русском языке."""
            
            # 🔥 ВАЖНОЕ ИСПРАВЛЕНИЕ: Простой синхронный запрос к LLM
            llm_response = self.llm._query_llm_api(
                prompt=text,
                context=context,
                subject=self.current_subject,
                system_prompt=system_prompt,
                max_tokens=150
            )
            
            if llm_response:
                limited_response = self._limit_response_length(llm_response, 3)
                # Адаптируем ответ под ученика
                return self._adapt_response_to_student(limited_response)
                    
        except Exception as e:
            print(f"Ошибка запроса к LLM для ученика: {e}")
        
        return None

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
            if len(sentence.split()) > 15:
                words = sentence.split()
                mid_point = len(words) // 2
                part1 = ' '.join(words[:mid_point])
                part2 = ' '.join(words[mid_point:])
                simplified_sentences.extend([part1, part2])
            else:
                simplified_sentences.append(sentence)
                
        return ' '.join(simplified_sentences)

    def _get_student_lesson_prompt(self) -> Optional[str]:
        """Возвращает предложение начать урок для ученика"""
        # После 2-3 фраз диалога предлагаем начать урок
        if self.student_conversation_count >= 2 and not self.student_subject_prompted:
            self.student_subject_prompted = True
            student_name = self.student_data.get('name', '')
            greeting = f"{student_name}, " if student_name else ""
            prompts = [
                f"{greeting}Давай начнем урок по {self.current_subject}. Готов?",
                f"{greeting}Прекрасно! Приступаем к уроку по {self.current_subject}. Начинаем?",
                f"{greeting}Замечательно! Начнем наш урок по {self.current_subject}?"
            ]
            return random.choice(prompts)
        
        return None

    def process_input(self, text: str) -> Optional[str]:
        """Основной метод обработки ввода ученика"""
        text_lower = text.lower().strip()
        
        # Увеличиваем счетчик разговора
        self.student_conversation_count += 1
        print(f"🎓 Диалог ученика: счетчик {self.student_conversation_count}, предмет: {self.current_subject}")
        
        # Команды продолжения урока
        continue_commands = [
            "продолжай", "дальше", "следующий", "вперед", "готов", 
            "понял", "ясно", "ага", "угу", "хорошо", "ок"
        ]

        if self.lesson_started and any(cmd in text_lower for cmd in continue_commands):
            next_paragraph = self._get_next_paragraph()
            if next_paragraph:
                print(f"✅ Команда продолжения обработана для ученика")
                return next_paragraph
            else:
                print("🏁 Урок завершен по команде продолжения")
                return "Урок завершен. Переходим к практике."

        # Добавляем в историю
        self._add_to_conversation_history(text, is_user=True)

        # Логика для режима ученика до начала урока
        if not self.lesson_started:
            return self._handle_student_pre_lesson(text, text_lower)

        # Обработка во время урока
        if self.lesson_started:
            return self._handle_lesson_interaction(text, text_lower)

        # Поиск в локальных шаблонах
        dialogue_response = self._get_dialogue_response(text_lower)
        if dialogue_response:
            adapted_response = self._adapt_response_to_student(dialogue_response)
            self._add_to_conversation_history(adapted_response, is_user=False)
            return adapted_response

        # Запрос к LLM
        llm_response = self._handle_llm_dialogue(text)
        if llm_response:
            self._add_to_conversation_history(llm_response, is_user=False)
            return llm_response

        # Fallback
        fallback = self._get_contextual_fallback()
        if fallback:
            self._add_to_conversation_history(fallback, is_user=False)
            return fallback
        
        return None

    def _handle_student_pre_lesson(self, text: str, text_lower: str) -> Optional[str]:
        """Обработка ввода до начала урока"""
        # Проверяем, не хочет ли ученик изучить конкретную тему
        if self._check_for_specific_topic_request(text_lower):
            print(f"🎯 Ученик запросил конкретную тему по предмету {self.current_subject}")
            return None

        # После 2-3 фраз предлагаем начать урок
        if self.student_conversation_count >= 2 and not self.student_subject_prompted:
            self.student_subject_prompted = True
            prompt = self._get_student_lesson_prompt()
            if prompt:
                self._add_to_conversation_history(prompt, is_user=False)
                return prompt

        # Если ученик соглашается начать урок
        if any(word in text_lower for word in ['да', 'ага', 'угу', 'ладно', 'хорошо', 'начать', 'начнем']):
            return self._start_student_lesson()

        # Поиск в шаблонах
        for pattern, responses in self.local_patterns.items():
            if pattern in text_lower:
                response = random.choice(responses) if isinstance(responses, list) else responses
                adapted_response = self._adapt_response_to_student(response)
                self._add_to_conversation_history(adapted_response, is_user=False)
                return adapted_response

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
                    return self._generate_lesson_on_demand(topic)
        return False

    def _generate_lesson_on_demand(self, topic: str) -> bool:
        """Генерирует урок по запрошенной теме"""
        try:
            print(f"🎯 Генерация урока для ученика по теме: {topic}")
            
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

            lesson_content = self.llm._query_llm_api(
                prompt=f"Создай подробный образовательный урок на тему: '{topic}'. Урок должен быть понятным и интересным для ученика.",
                context="",
                subject=self.current_subject,
                system_prompt=system_prompt,
                max_tokens=2500
            )
            
            if not lesson_content:
                print("❌ Ошибка: LLM не вернул содержание урока")
                return False
            
            # Создаем файл урока
            lesson_id = f"student_{self.student_data.get('student_id', 'unknown')}_{topic.lower().replace(' ', '_')}_{int(time.time())}"
            filename = f"{lesson_id}.txt"
            lesson_path = self.lessons_dir / filename
            
            # Записываем контент в файл
            with open(lesson_path, 'w', encoding='utf-8') as f:
                f.write(f"Урок по теме: {topic}\n\n")
                f.write(lesson_content)
            
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
            
            # Начинаем сгенерированный урок
            self._start_generated_lesson(lesson_data)
            return True
            
        except Exception as e:
            print(f"❌ Ошибка генерации урока для ученика: {e}")
            return False

    def _start_generated_lesson(self, lesson_data: dict):
        """Начинает сгенерированный урок"""
        try:
            print(f"🚀 Начинаем сгенерированный урок для ученика: {lesson_data['title']}")
            
            self.selected_lesson = lesson_data
            self.lesson_started = True
            self.current_state = "lesson_reading"
            self.current_paragraph = 0
            
            # Загружаем содержание урока
            self.lesson_content = self._load_lesson_content(lesson_data['file_path'])
            
            if not self.lesson_content:
                print("❌ Не удалось загрузить содержание урока")
                return
            
            # Инициализируем базу знаний
            self.knowledge_base = KnowledgeBase(self.current_subject)
            
            # Очищаем историю диалога
            self.conversation_history = []
            self.conversation_context = []
            
            print(f"🎉 Сгенерированный урок '{lesson_data['title']}' успешно начат!")
            
        except Exception as e:
            print(f"❌ Ошибка начала сгенерированного урока: {e}")
            self.lesson_started = False

    def _start_student_lesson(self) -> str:
        """Начинает урок для ученика"""
        print(f"🚀 Начинаем урок для ученика по предмету: {self.current_subject}")
        
        # Используем существующие уроки или создаем демо
        lessons = self.lessons.get(self.current_subject, [])
        if lessons:
            self.selected_lesson = lessons[0]
        else:
            self.selected_lesson = {
                'id': f"demo_{self.current_subject}",
                'title': f"Урок по {self.current_subject}",
                'file_path': self.lessons_dir / f"demo_{self.current_subject}.txt"
            }

        self.lesson_started = True
        self.lesson_content = self._load_lesson_content(self.selected_lesson['file_path'])
        self.knowledge_base = KnowledgeBase(self.current_subject)
        
        student_name = self.student_data.get('name', '')
        greeting = f"{student_name}, " if student_name else ""
        first_paragraph = self._get_next_paragraph()
        return f"{greeting}Отлично! Начинаем урок по {self.current_subject}. {first_paragraph}"

    def _get_next_paragraph(self) -> Optional[str]:
        """Возвращает следующий абзац урока"""
        print(f"📄 Получение следующего абзаца для ученика: текущий {self.current_paragraph}, всего {len(self.lesson_content)}")
        
        if self.current_paragraph < len(self.lesson_content):
            paragraph = self.lesson_content[self.current_paragraph]
            self.current_paragraph += 1
            
            # Генерация визуализации если включена
            if (self.visualization_enabled and paragraph and 
                len(paragraph.strip()) > 10 and self.room_id):
                
                def delayed_visualization():
                    time.sleep(0.5)
                    context = " ".join(self.lesson_content[max(0, self.current_paragraph-2):self.current_paragraph])
                    self._generate_visualization(paragraph, context)
                
                threading.Thread(target=delayed_visualization, daemon=True).start()
            
            print(f"✅ Возвращаем абзац {self.current_paragraph} для ученика")
            return paragraph
        else:
            # Урок завершен, начинаем практику
            print("🏁 Урок завершен для ученика, запускаем практику")
            return self._start_practice_session()

    def _start_practice_session(self) -> str:
        """Запускает практику"""
        self.lesson_started = False
        self.practice_active = True
        self.waiting_for_answer = True
        self.current_question_index = 0
        
        print("=== ЗАПУСК ФАЗЫ ПРАКТИКИ ДЛЯ УЧЕНИКА ===")
        
        # Инициализируем практику
        lesson_context = " ".join(self.lesson_content)
        self.practice_manager.initialize_practice_generation(lesson_context, self.current_subject)
        
        first_question = self.practice_manager.get_next_question()
        if first_question:
            self.current_practice_question = {
                "id": 1,
                "question": first_question,
                "answer": ""
            }
            student_name = self.student_data.get('name', '')
            greeting = f"{student_name}, " if student_name else ""
            return f"{greeting}Урок завершен! Переходим к практике. Первый вопрос: {first_question}"
        
        self.practice_active = False
        return "Урок завершен! Практические задания временно недоступны."

    def _handle_lesson_interaction(self, text: str, text_lower: str) -> Optional[str]:
        """Обработка взаимодействия во время урока"""
        # Команды остановки
        if any(word in text_lower for word in ["стоп", "останови", "хватит"]):
            self.lesson_started = False
            return "Урок остановлен. Скажи 'привет' чтобы продолжить."
        
        # Обработка вопросов во время урока
        return self.handle_question_during_lesson(text)

    def handle_question_during_lesson(self, question: str) -> str:
        """Обработка вопросов ученика во время урока"""
        if not question.strip():
            return "Повтори вопрос, пожалуйста."
            
        # Генерация визуализации если включена
        if self.visualization_enabled:
            context = " ".join(self.lesson_content[max(0, self.current_paragraph-2):self.current_paragraph])
            self._generate_visualization(question, context)
        
        print(f"🎓 Обработка вопроса ученика во время урока: '{question}'")
        
        # 🔥 ВАЖНОЕ ИСПРАВЛЕНИЕ: Используем ту же логику LLM что и в DialogueManager
        if self.llm_query_mode == "llm_first":
            # Сначала пробуем LLM
            current_context = ""
            if self.lesson_content and self.current_paragraph > 0:
                context_start = max(0, self.current_paragraph - 2)
                current_context = " ".join(self.lesson_content[context_start:self.current_paragraph])
            
            llm_response = self.llm.query(question, current_context, self.current_subject)
            if llm_response and not llm_response.startswith("Интересный вопрос!"):
                adapted_response = self._adapt_response_to_student(llm_response)
                return adapted_response
        
        # Fallback для ученика
        return "Интересный вопрос! Давай обсудим его после урока, чтобы не отвлекаться."

    def _evaluate_and_generate_next(self, student_answer: str) -> str:
        """Оценивает ответ и возвращает следующий вопрос"""
        print(f"🔍 Обработка ответа ученика: '{student_answer}'")
        
        if not self.practice_active:
            return "Практика не активна."
        
        # Пропускаем команды продолжения
        if any(cmd in student_answer.lower() for cmd in ['продолжай', 'дальше', 'следующий']):
            next_question = self.practice_manager.get_next_question()
            if next_question:
                return f"Это похоже на команду. Пожалуйста, дай ответ на вопрос. Следующий вопрос: {next_question}"
            else:
                self._end_practice_session()
                return "Практика завершена."
        
        current_question = self.current_practice_question
        if not current_question:
            self._end_practice_session()
            return "Практика завершена."
        
        # Увеличиваем счетчик вопросов
        self.current_question_index += 1
        
        # Проверяем лимит вопросов
        if self.current_question_index >= self.max_questions:
            self._end_practice_session()
            student_name = self.student_data.get('name', '')
            greeting = f"{student_name}, " if student_name else ""
            return f"{greeting}Отлично! Ты ответил на все вопросы практики. Урок завершен!"
        
        # Оценка ответа и получение следующего вопроса
        feedback, next_question = self.practice_manager.evaluate_and_continue(
            student_answer, 
            current_question["question"]
        )
        
        # Адаптируем обратную связь
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
            print(f"➡️ Следующий вопрос получен для ученика")
            return response
        else:
            self._end_practice_session()
            return f"{feedback}. Практика завершена!"

    def _handle_practice_answer(self, text: str) -> str:
        """Обработка ответа ученика во время практики"""
        return self._evaluate_and_generate_next(text)

    def _end_practice_session(self):
        """Завершает сессию практики"""
        self.practice_active = False
        self.waiting_for_answer = False
        self.current_state = "greeting"
        self.current_question_index = 0
        self.practice_manager.stop_async_generation()
        
        self.lesson_started = False
        self.selected_lesson = None
        self.lesson_content = []
        self.current_paragraph = 0
        
        if self.room_id:
            self.socketio.emit('practice_ended', {'room_id': self.room_id})
        print("=== 🏁 ПРАКТИКА ДЛЯ УЧЕНИКА ЗАВЕРШЕНА ===")

    def _generate_visualization(self, text: str, context: str = ""):
        """Генерация визуализации для текста"""
        if not self.visualization_enabled or not text.strip():
            return
    
        current_time = time.time()
        if current_time - self.last_visualization_time < self.visualization_cooldown:
            return
        
        self.paragraphs_since_last_viz += 1
        
        should_generate = (self.paragraphs_since_last_viz >= self.viz_paragraph_interval or 
                          self._has_visualization_triggers(text))
        
        if should_generate:
            try:
                self.last_visualization_time = current_time
                self.paragraphs_since_last_viz = 0
                self.visualization_counter += 1
                
                print(f"🎨 Генерация визуализации для ученика: {text[:100]}...")
                
                if self.room_id and self.socketio:
                    viz_result = self.llm.generate_visualization(text, context)
                    
                    if viz_result and viz_result.get("success"):
                        self.socketio.emit('visualization_generated', {
                            'room_id': self.room_id,
                            'topic': text[:100],
                            'mermaid_code': viz_result.get('mermaid_code', ''),
                            'svg_code': viz_result.get('svg_code', ''),
                            'timestamp': time.time()
                        }, room=self.room_id)
                        print(f"✅ Визуализация отправлена в комнату ученика {self.room_id}")
                    
            except Exception as e:
                print(f"❌ Ошибка генерации визуализации для ученика: {e}")

    def _has_visualization_triggers(self, text: str) -> bool:
        """Проверяет наличие триггеров для визуализации"""
        text_lower = text.lower()
        
        visualization_triggers = [
            'структура', 'схема', 'диаграмма', 'график', 'процесс', 
            'алгоритм', 'иерархия', 'взаимосвязь', 'соотношение',
            'таблица', 'классификация', 'этапы', 'стадии', 'система'
        ]
        
        structure_indicators = [
            'состоит из', 'включает в себя', 'делится на', 'подразделяется',
            'можно разделить', 'выделяют', 'различают', 'существуют'
        ]
        
        has_trigger = any(trigger in text_lower for trigger in visualization_triggers)
        has_structure = any(indicator in text_lower for indicator in structure_indicators)
        
        # Для младших школьников чаще используем визуализацию
        age = int(self.student_data.get('age', 12))
        if age <= 10:
            has_trigger = True
        
        is_long_enough = len(text.split()) > 3
        
        return (has_trigger or has_structure) and is_long_enough

    def _get_contextual_fallback(self) -> str:
        """Возвращает контекстно-зависимый ответ"""
        if not self.conversation_history:
            return self.student_prompts["greeting"]
        
        user_messages = [msg['text'] for msg in self.conversation_history if msg['is_user']]
        last_user_message = user_messages[-1].lower() if user_messages else ""
        
        if any(word in last_user_message for word in ['имя', 'зовут', 'меня']):
            student_name = self.student_data.get('name', 'друг')
            return f"Приятно познакомиться, {student_name}! Теперь давайте начнем наш урок по {self.current_subject}."
        
        if any(word in last_user_message for word in ['дела', 'настроение', 'чувств']):
            return "Рад это слышать! Так давайте начнем наш урок?"
        
        prompt = self._get_student_lesson_prompt()
        return prompt if prompt else "Давайте начнем наш урок. Готовы?"

    def _handle_greeting(self, text: str) -> Optional[str]:
        greeting_words = ["привет", "здравствуй", 'начать', "старт", " готов", "поехали", "давай"]
        if any(word in text for word in greeting_words):
            self.current_state = "subject_selection"
            return self.student_prompts["greeting"]
        return None

    def _handle_subject_selection(self, text: str) -> Optional[str]:
        # В режиме ученика предмет уже выбран, быстро переходим к уроку
        return self._start_student_lesson()

    def _handle_lesson_reading(self, text: str) -> Optional[str]:
        if any(word in text for word in ["стоп", "останови", "хватит"]):
            self.lesson_started = False
            self.current_state = "greeting"
            self.conversation_counter = 0
            self.knowledge_base = None
            self.conversation_history = []
            self.conversation_context = []
            return "Урок остановлен. Скажи 'привет' когда захочешь продолжить."
        return None

    def _handle_practice_session(self, text: str) -> Optional[str]:
        if any(word in text for word in ["стоп", "останови", "хватит"]):
            self.practice_active = False
            self.waiting_for_answer = False
            self.current_state = "greeting"
            self.conversation_counter = 0
            self.conversation_history = []
            self.conversation_context = []
            
            if self.room_id:
                self.socketio.emit('practice_ended', {'room_id': self.room_id})
            
            return "Практика остановлена. Скажи 'привет' когда захочешь продолжить."
            
        if self.waiting_for_answer:
            return self._handle_practice_answer(text)
            
        return None

    # 🔥 ВАЖНОЕ ИСПРАВЛЕНИЕ: Методы для работы с LLM
    def set_llm_mode(self, mode: str):
        """Устанавливает режим LLM"""
        if mode in ["traditional", "llm_first"]:
            self.llm_query_mode = mode
            self.llm.set_llm_mode(mode)
            print(f"🎓 Установлен режим LLM для ученика: {mode}")

    def set_llm_priority(self, priority: str):
        """Устанавливает приоритет моделей LLM"""
        if hasattr(self.llm, 'set_priority'):
            self.llm.set_priority(priority)
            print(f"🎓 Установлен приоритет LLM для ученика: {priority}")

    def set_room_id(self, room_id: str):
        """Устанавливает ID комнаты"""
        self.room_id = room_id
        print(f"🔧 Установлен room_id для StudentDialogueManager: {room_id}")

    # Методы для совместимости
    def get_current_subject(self) -> Optional[str]:
        return self.current_subject

    def is_lesson_started(self) -> bool:
        return self.lesson_started

    def get_selected_lesson(self) -> Optional[dict]:
        return self.selected_lesson

    def reset(self):
        """Сброс состояния"""
        self.current_state = "greeting"
        self.lesson_started = False
        self.lesson_content = []
        self.current_paragraph = 0
        self.conversation_history = []
        self.conversation_context = []
        self.practice_active = False
        self.waiting_for_answer = False
        self.student_conversation_count = 0
        self.student_subject_prompted = False

    def get_available_subjects(self) -> List[str]:
        subjects = list(self.lessons.keys())
        if self.current_subject not in subjects:
            subjects.append(self.current_subject)
        return subjects

    def get_practice_status(self) -> Dict:
        """Возвращает статус практики"""
        return {
            "practice_active": self.practice_active,
            "waiting_for_answer": self.waiting_for_answer,
            "current_question": self.current_practice_question,
            "question_index": self.current_question_index,
            "max_questions": self.max_questions,
            "questions_asked": len(self.practice_manager.generated_questions) if hasattr(self.practice_manager, 'generated_questions') else 0
        }

    def get_conversation_stats(self) -> Dict:
        """Возвращает статистику диалога"""
        user_messages = [msg for msg in self.conversation_history if msg['is_user']]
        teacher_messages = [msg for msg in self.conversation_history if not msg['is_user']]
        
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": len(user_messages),
            "teacher_messages": len(teacher_messages),
            "current_state": self.current_state,
            "lesson_started": self.lesson_started,
            "current_paragraph": self.current_paragraph,
            "total_paragraphs": len(self.lesson_content),
            "conversation_context": self.conversation_context,
            "student_conversation_count": self.student_conversation_count
        }

    def debug_info(self) -> Dict:
        """Возвращает отладочную информацию"""
        practice_stats = self.practice_manager.get_practice_stats() if hasattr(self.practice_manager, 'get_practice_stats') else {}
        
        return {
            "current_state": self.current_state,
            "current_subject": self.current_subject,
            "lesson_started": self.lesson_started,
            "practice_active": self.practice_active,
            "waiting_for_answer": self.waiting_for_answer,
            "current_paragraph": self.current_paragraph,
            "total_paragraphs": len(self.lesson_content),
            "llm_mode": self.llm_query_mode,
            "conversation_history_length": len(self.conversation_history),
            "practice_stats": practice_stats,
            "current_practice_question": self.current_practice_question,
            "room_id": self.room_id,
            "student_data": self.student_data,
            "student_conversation_count": self.student_conversation_count,
            "student_subject_prompted": self.student_subject_prompted
        }

    def get_system_status(self) -> Dict:
        """Возвращает общий статус системы"""
        llm_status = self.llm.get_llm_status() if hasattr(self.llm, 'get_llm_status') else {}
        practice_stats = self.practice_manager.get_practice_stats() if hasattr(self.practice_manager, 'get_practice_stats') else {}
        
        return {
            "student_dialogue_manager": {
                "current_state": self.current_state,
                "lesson_started": self.lesson_started,
                "practice_active": self.practice_active,
                "current_subject": self.current_subject,
                "conversation_history_length": len(self.conversation_history),
                "questions_asked": len(self.practice_manager.generated_questions) if hasattr(self.practice_manager, 'generated_questions') else 0,
                "max_questions": self.max_questions,
                "student_conversation_count": self.student_conversation_count,
                "student_data": {
                    "name": self.student_data.get('name'),
                    "age": self.student_data.get('age'),
                    "level": self.student_data.get('level')
                }
            },
            "llm": llm_status,
            "practice": practice_stats
        }


# Тестирование
if __name__ == "__main__":
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