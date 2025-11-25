# student_dialogue.py
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from difflib import SequenceMatcher
import random
import re
from knowledge.knowledge_base import KnowledgeBase
from llm import LLMIntegration
from config import get_llm_mode, get_dialogue_settings
import time
import threading
from practice_manager import PracticeManager
from visualization_manager import visualization_manager

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
        self.llm = LLMIntegration()
        self.conversation_counter = 0
        self.llm_query_mode = get_llm_mode()
        self.dialogue_settings = get_dialogue_settings()
        self.conversation_history = []
        self.dialogue_knowledge = self._load_dialogue_knowledge()
        self.conversation_context = []
        self.room_id = None
        
        # Менеджер практики
        self.practice_manager = PracticeManager(self.llm)
        
        # Поля для практики
        self.practice_active = False
        self.current_question_index = 0
        self.current_expected_answer = ""
        self.waiting_for_answer = False
        self.current_practice_question = None
        self.max_questions = 5
        
        # Поля для улучшенного диалога ученика
        self.last_subject_prompt_time = 0
        self.subject_prompt_cooldown = 30
        self.auto_selected_subject = self.current_subject
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
        
        # НОВАЯ СИСТЕМА ВИЗУАЛИЗАЦИИ
        self.visualization_manager = visualization_manager
        self.current_visualization_type = None
        self.generated_mindmap = None
        self.current_slide_index = 0
        self.slides = None
        
        # Адаптированные промты для ученика
        self.student_prompts = self._load_student_prompts()
        
        self._load_lessons()
        
        # Локальные шаблоны, адаптированные для учеников
        self.local_patterns = {
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
        }

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

    def _load_dialogue_knowledge(self) -> Dict:
        """Загрузка расширенной базы диалоговых шаблонов"""
        try:
            dialogue_path = Path("knowledge/dialogue_knowledge.json")
            if dialogue_path.exists():
                with open(dialogue_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Ошибка загрузки диалоговых шаблонов: {e}")
        
        return self._get_default_dialogue_patterns()

    def _get_default_dialogue_patterns(self) -> Dict:
        """Возвращает базовые диалоговые шаблоны по умолчанию"""
        return {
            "greeting_patterns": {
                "привет": self._get_age_appropriate_greeting(),
                "здравствуй": self._get_age_appropriate_greeting()
            },
            "mood_patterns": {
                "как дела": ["Отлично! А у тебя как?", "Прекрасно! Готов к уроку."]
            },
            "learning_patterns": {
                "хочу учиться": ["Отлично! Давай начнём наш урок!", "Супер! Приступаем к занятию!"]
            },
            "subject_questions": {
                "что преподаешь": [f"Я преподаю {self.current_subject}! Давай начнём урок."]
            },
            "metadata": {
                "version": "1.0",
                "type": "student_dialogue_patterns"
            }
        }

    def _load_lessons(self):
        """Загружает список доступных уроков"""
        self.lessons = {}
        try:
            if not self.lessons_dir.exists():
                self.lessons_dir.mkdir(parents=True)
                return
                
            # Загрузка текстовых файлов уроков
            for lesson_file in self.lessons_dir.glob("*.txt"):
                try:
                    subject = self._detect_subject(lesson_file.stem)
                    
                    if subject not in self.lessons:
                        self.lessons[subject] = []
                    
                    self.lessons[subject].append({
                        'id': lesson_file.stem,
                        'title': lesson_file.stem.replace('_', ' ').title(),
                        'description': f"Интересный урок по {subject}",
                        'file_path': lesson_file,
                        'type': 'text',
                        'is_demo': 'demo' in lesson_file.stem.lower() or 'general' in lesson_file.stem.lower()
                    })
                except Exception as e:
                    print(f"Ошибка загрузки урока {lesson_file}: {e}")
                    
        except Exception as e:
            print(f"Ошибка доступа к папке уроков: {e}")

    def _detect_subject(self, filename: str) -> str:
        """Определяет предмет по названию файла"""
        filename_lower = filename.lower()
        if any(word in filename_lower for word in ['math', 'математика', 'алгебра', 'геометрия']):
            return "математика"
        elif any(word in filename_lower for word in ['history', 'история', 'истор']):
            return "история"
        elif any(word in filename_lower for word in ['physics', 'физика', 'физ']):
            return "физика"
        elif any(word in filename_lower for word in ['chemistry', 'химия', 'хим']):
            return "химия"
        elif any(word in filename_lower for word in ['social', 'обществознание', 'общество']):
            return "обществознание"
        elif any(word in filename_lower for word in ['biology', 'биология', 'био']):
            return "биология"
        elif any(word in filename_lower for word in ['literature', 'литература', 'лит']):
            return "литература"
        elif any(word in filename_lower for word in ['russian', 'русский', 'язык']):
            return "русский язык"
        else:
            return "общее"

    def _load_lesson_content(self, lesson_file: Path) -> List[str]:
        """Загружает содержание урока из текстового файла с улучшенной очисткой"""
        try:
            print(f"📖 Загрузка урока из файла: {lesson_file}")
            
            if not lesson_file.exists():
                print(f"❌ Файл урока не существует: {lesson_file}")
                return ["Файл урока не найден. Попробуйте другой урок."]
                
            with open(lesson_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"✅ Файл прочитан, длина: {len(content)} символов")
            
            # УЛУЧШЕННАЯ ОЧИСТКА СОДЕРЖАНИЯ
            content = self._clean_lesson_content(content)
            
            # Разбиваем на абзацы (по пустым строкам)
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            # Если абзацев нет, разбиваем на предложения
            if not paragraphs:
                print("⚠️ Нет абзацев, разбиваем на предложения")
                sentences = re.split(r'(?<=[.!?])\s+', content)
                # Объединяем предложения в группы по 2-3 для плавного чтения
                current_paragraph = []
                paragraphs = []
                
                for sentence in sentences:
                    if sentence.strip():
                        current_paragraph.append(sentence.strip())
                        if len(current_paragraph) >= 2:
                            paragraphs.append(' '.join(current_paragraph))
                            current_paragraph = []
                
                # Добавляем оставшиеся предложения
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
            
            print(f"✅ Урок разбит на {len(paragraphs)} абзацев")
            
            if not paragraphs:
                print("❌ Не удалось разбить урок на абзацев")
                return ["Содержание урока временно недоступно. Давайте поговорим на эту тему!"]
                
            return paragraphs
            
        except Exception as e:
            print(f"❌ Ошибка загрузки содержания урока: {e}")
            return ["Ошибка загрузки урока. Попробуйте позже."]

    def _clean_lesson_content(self, content: str) -> str:
        """Очистка содержания урока от лишнего форматирования"""
        if not content:
            return content
        
        # Удаляем маркеры форматирования
        content = re.sub(r'[\*\#]{1,}', '', content)  # Удаляем одиночные * и #
        content = re.sub(r'\-\-\-+', '', content)  # Удаляем разделители ---
        content = re.sub(r'\+\+\+', '', content)  # Удаляем +++
        
        # Удаляем HTML-теги если есть
        content = re.sub(r'<[^>]+>', '', content)
        
        # Нормализуем переводы строк
        content = re.sub(r'\r\n', '\n', content)
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        # Удаляем начальные/конечные пробелы
        content = content.strip()
        
        return content

    def _similarity(self, a: str, b: str) -> float:
        """Вычисление схожести строк"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def _add_to_conversation_history(self, text: str, is_user: bool = True):
        """Добавляет реплику в историю диалога"""
        self.conversation_history.append({
            "text": text,
            "is_user": is_user,
            "timestamp": time.time()
        })
        
        # Ограничиваем размер истории
        max_history = self.dialogue_settings.get("context_window", 10)
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]
            
        # Обновляем контекст (последние 3 реплики пользователя)
        if is_user:
            user_messages = [msg['text'] for msg in self.conversation_history if msg['is_user']]
            self.conversation_context = user_messages[-3:] if len(user_messages) > 3 else user_messages

    def _get_conversation_context(self) -> str:
        """Возвращает контекст диалога для LLM"""
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
        """Поиск ответа в диалоговых шаблонов с учетом контекста"""
        text_lower = text.lower().strip()
        
        # 1. Поиск точного совпадения в расширенной базе
        for category, patterns in self.dialogue_knowledge.items():
            if category.endswith('_patterns') and isinstance(patterns, dict):
                for pattern, responses in patterns.items():
                    if pattern in text_lower and responses:
                        return random.choice(responses)
        
        # 2. Контекстный поиск (есть есть история разговора)
        if self.conversation_context:
            last_user_messages = ' '.join(self.conversation_context).lower()
            
            # Поиск контекстных паттернов
            contextual_patterns = self.dialogue_knowledge.get('contextual_patterns', {})
            for pattern, responses in contextual_patterns.items():
                if pattern in last_user_messages and responses:
                    return random.choice(responses)
        
        # 3. Поиск в локальных шаблонов (fallback)
        for pattern, responses in self.local_patterns.items():
            if pattern in text_lower:
                if isinstance(responses, list):
                    return random.choice(responses)
                else:
                    return responses
        
        return None

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
Используй {self.student_prompts['subject_specific']['examples']}.
Будь кратким и понятным, максимум 2-3 предложения. Отвечай на русском языке."""
            
            # АСИНХРОННЫЙ запрос к локальной модели
            if room_id and self.socketio:
                # Используем асинхронный режим с callback
                def llm_callback(response, r_id):
                    if response:
                        limited_response = self._limit_response_length(
                            response, 
                            self.dialogue_settings.get("max_response_length", 3)
                        )
                        
                        # Адаптируем ответ под ученика
                        adapted_response = self._adapt_response_to_student(limited_response)
                        
                        # Отправляем ответ через WebSocket
                        self.socketio.emit('llm_dialogue_response', {
                            'room_id': r_id,
                            'response': adapted_response,
                            'original_text': text
                        }, room=r_id)
                
                # Асинхронный запрос - не блокируем основной поток
                self.llm._query_llm_api(
                    prompt=text,
                    context=context,
                    subject=self.current_subject,
                    system_prompt=system_prompt,
                    max_tokens=150,
                    room_id=room_id,
                    callback=llm_callback
                )
                
                return None  # Ответ придет асинхронно
                
            else:
                # Синхронный режим для обратной совместимости
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
                    
        except Exception as e:
            print(f"Ошибка запроса к LLM для диалога: {e}")
        
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

    def _get_contextual_fallback(self) -> str:
        """Возвращает контекстно-зависимый ответ когда ничего не найдено"""
        if not self.conversation_history:
            return self.student_prompts["greeting"]
        
        # Анализ контекста разговора
        user_messages = [msg['text'] for msg in self.conversation_history if msg['is_user']]
        last_user_message = user_messages[-1].lower() if user_messages else ""
        
        # Определяем тему разговора по последним сообщениям
        if any(word in last_user_message for word in ['имя', 'зовут', 'меня']):
            student_name = self.student_data.get('name', 'друг')
            return f"Приятно познакомиться, {student_name}! Теперь давайте начнем наш урок по {self.current_subject}."
        
        if any(word in last_user_message for word in ['дела', 'настроение', 'чувств']):
            return "Рад это слышать! Так давайте начнем наш урок?"
        
        # Стандартный ответ с напоминанием о начале урока
        prompt = self._get_student_lesson_prompt()
        return prompt if prompt else "Давайте начнем наш урок. Готовы?"

    def _should_save_to_knowledge_base(self, text: str) -> bool:
        """Определяет, нужно ли сохранять фразу в базу знаний"""
        text_lower = text.lower()
        
        # Исключаем фразы для генерации уроков
        generation_patterns = [
            'давай изучим', 'хочу изучить', 'урок по', 'изучим', 
            'расскажи про', 'хочу узнать про', 'объясни тему',
            'создай урок', 'сгенерируй урок', 'научи меня'
        ]
        
        if any(pattern in text_lower for pattern in generation_patterns):
            return False
        
        return True

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

    def _check_for_lesson_generation_intent(self, text_lower: str) -> bool:
        """
        Проверяет, хочет ли ученик сгенерировать новый урок по теме.
        Возвращает True, если урок был успешно сгенерирован.
        """
        # Шаблоны фраз, которые означают "создай урок"
        generation_patterns = [
            r'хочу изучить (.+)',
            r'можешь рассказать про (.+)', 
            r'урок по (.+)',
            r'изучим (.+)',
            r'расскажи про (.+)',
            r'хочу узнать про (.+)',
            r'объясни тему (.+)',
            r'создай урок про (.+)',
            r'сгенерируй урок о (.+)',
            r'научи меня (.+)'
        ]
        
        for pattern in generation_patterns:
            match = re.search(pattern, text_lower)
            if match:
                topic = match.group(1).strip()
                topic = re.sub(r'[.?]$', '', topic)
                if topic and len(topic) > 2:
                    print(f"🎯 Ученик запросил тему по предмету {self.current_subject}: '{topic}'")
                    generated_lesson = self.generate_lesson_on_demand(topic)
                    if generated_lesson:
                        print(f"Урок успешно сгенерирован: {generated_lesson['id']}")
                        self._start_generated_lesson(generated_lesson)
                        return True
                    else:
                        print("❌ Не удалось сгенерировать урок")
        return False

    def _start_generated_lesson(self, lesson_data: dict):
        """Начинает сгенерированный урок"""
        try:
            print(f"🚀 Начинаем сгенерированный урок для ученика: {lesson_data['title']}")
            
            self.selected_lesson = lesson_data
            self.lesson_started = True
            self.current_state = "lesson_reading"
            self.current_paragraph = 0
            
            # ВКЛЮЧАЕМ АВТОМАТИЧЕСКУЮ ВИЗУАЛИЗАЦИЮ ПРИ СТАРТЕ УРОКА
            self.enable_visualization()
            
            # Загружаем содержание урока
            print(f"📖 Загрузка содержания урока из: {lesson_data['file_path']}")
            self.lesson_content = self._load_lesson_content(lesson_data['file_path'])
            
            if not self.lesson_content:
                print("❌ Не удалось загрузить содержание урока")
                return
            
            print(f"✅ Урок загружен, количество абзацев: {len(self.lesson_content)}")
            
            # Инициализируем базу знаний
            self.knowledge_base = KnowledgeBase(self.current_subject)
            
            # Очищаем историю диалога при начале урока
            self.conversation_history = []
            self.conversation_context = []
            
            # ИНИЦИАЛИЗАЦИЯ НОВОЙ СИСТЕМЫ ВИЗУАЛИЗАЦИИ
            self._start_lesson_visualization(
                lesson_data['id'],
                lesson_data['title'],
                self.lesson_content
            )
            
            print(f"🎉 Сгенерированный урок '{lesson_data['title']}' успешно начат!")
            
        except Exception as e:
            print(f"❌ Ошибка начала сгенерированного урока: {e}")
            self.lesson_started = False

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
            has_trigger = True  # Чаще показываем визуализации для младших
        
        is_long_enough = len(text.split()) > 3
        
        return (has_trigger or has_structure) and is_long_enough

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
                    # Используем улучшенную генерацию через структурированные данные
                    viz_result = self.llm.generate_visualization(text, context)
                    
                    if viz_result and viz_result.get("success"):
                        self.socketio.emit('visualization_generated', {
                            'room_id': self.room_id,
                            'topic': text[:100],
                            'mermaid_code': viz_result.get('mermaid_code', ''),
                            'svg_code': viz_result.get('svg_code', ''),
                            'timestamp': time.time()
                        }, room=self.room_id)
                        print(f"✅ Визуализация отправлена в комнату {self.room_id}")
                    
            except Exception as e:
                print(f"❌ Ошибка генерации визуализации: {e}")

    def enable_visualization(self):
        """Включение автоматической визуализации"""
        self.visualization_enabled = True
        print("✅ Автоматическая визуализация включена для ученика")

    def disable_visualization(self):
        """Выключение автоматической визуализации"""
        self.visualization_enabled = False
        print("❌ Автоматическая визуализация выключена")

    # НОВАЯ СИСТЕМА ВИЗУАЛИЗАЦИИ - МЕТОДЫ
    def _start_lesson_visualization_async(self, lesson_id: str, lesson_title: str, lesson_content: List[str]):
        """Асинхронно запускает визуализацию для урока"""
        try:
            print(f"🎨 Асинхронная инициализация визуализации для ученика: {lesson_title}")
            
            # Запускаем инициализацию визуализации (не блокируем основной поток)
            visualization_type = self.visualization_manager.initialize_lesson_visualization(
                lesson_id, 
                lesson_title, 
                " ".join(lesson_content), 
                self.room_id,
                self.socketio
            )
            
            self.current_visualization_type = visualization_type
            
            if visualization_type == "slides":
                self.slides = self.visualization_manager.get_lesson_slides(lesson_id)
                self.current_slide_index = 0
                print(f"🎨 Слайды загружены для ученика: {len(self.slides)} шт.")
            else:
                print("🎨 Запущена асинхронная генерация mind map для ученика")
                
            print(f"✅ Визуализация инициализирована для ученика (тип: {visualization_type})")
            
        except Exception as e:
            print(f"❌ Ошибка инициализации визуализации для ученика: {e}")

    def _start_lesson_visualization(self, lesson_id: str, lesson_title: str, lesson_content: List[str]):
        """Запускает систему визуализации для урока"""
        try:
            print(f"🎨 Инициализация визуализации для ученика: {lesson_id}")
            
            # ЗАПУСКАЕМ АСИНХРОННО - не блокируем начало урока
            self._start_lesson_visualization_async(lesson_id, lesson_title, lesson_content)
            
        except Exception as e:
            print(f"❌ Ошибка инициализации визуализации для ученика: {e}")
    
    def _handle_paragraph_visualization(self, paragraph_index: int):
        """Обрабатывает визуализацию для текущего абзаца ученика"""
        if not self.lesson_started or not self.selected_lesson:
            return
            
        if self.current_visualization_type == "slides":
            # Показываем следующий слайд для каждого нового абзаца
            if self.slides and paragraph_index < len(self.slides):
                self._send_slide(paragraph_index)
    
    def _send_slide(self, slide_index: int):
        """Отправляет слайд в комнату ученика"""
        if not self.slides or slide_index >= len(self.slides):
            return
            
        self.current_slide_index = slide_index
        slide_path = self.slides[slide_index]
        
        print(f"🖼️ Отправка слайда ученику {slide_index + 1}/{len(self.slides)}: {slide_path}")
        
        if self.room_id:
            self.socketio.emit('lesson_visualization', {
                'room_id': self.room_id,
                'type': 'slide',
                'data': {
                    'slide_path': slide_path,
                    'slide_number': slide_index + 1,
                    'total_slides': len(self.slides),
                    'timestamp': time.time()
                },
                'lesson_id': self.selected_lesson['id'],
                'lesson_title': self.selected_lesson.get('title', 'Урок')
            }, room=self.room_id)
    
    def send_current_visualization(self):
        """Принудительно отправляет текущую визуализацию ученику"""
        if not self.lesson_started or not self.selected_lesson or not self.room_id:
            return
            
        if self.current_visualization_type == "slides" and self.slides:
            self._send_slide(self.current_slide_index)
        elif self.current_visualization_type == "mindmap" and self.generated_mindmap:
            self.socketio.emit('lesson_visualization', {
                'room_id': self.room_id,
                'type': 'mindmap',
                'data': self.generated_mindmap,
                'lesson_id': self.selected_lesson['id'],
                'lesson_title': self.selected_lesson.get('title', 'Урок')
            }, room=self.room_id)

    def process_input(self, text: str) -> Optional[str]:
        """Обработка входящего текста и генерация ответа для ученика"""
        text_lower = text.lower().strip()
        
        # Увеличиваем счетчик разговора
        self.student_conversation_count += 1
        print(f"🎓 Диалог ученика: счетчик {self.student_conversation_count}, предмет: {self.current_subject}")
        
        # РАСШИРЕННЫЙ СПИСОК КОМАНД ПРОДОЛЖЕНИЯ
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
        
        if self.lesson_started:
            handler = self.dialogue_states.get(self.current_state)
            if handler:
                response = handler(text_lower)
                if response:
                    self._add_to_conversation_history(response, is_user=False)
                    return response
            return None
        
        generated_lesson = self._check_for_lesson_generation_intent(text_lower)
        if generated_lesson:
            return None
        
        if self.practice_active and self.waiting_for_answer:
            return self._handle_practice_answer(text)
        
        dialogue_response = self._get_dialogue_response(text_lower)
        if dialogue_response:
            final_response = self._adapt_response_to_student(dialogue_response)
            if final_response:
                self._add_to_conversation_history(final_response, is_user=False)
                return final_response
        
        llm_response = self._handle_llm_dialogue(text)
        if llm_response:
            final_response = self._adapt_response_to_student(llm_response)
            if final_response:
                self._add_to_conversation_history(final_response, is_user=False)
                return final_response
        
        fallback_response = self._get_contextual_fallback()
        if fallback_response:
            self._add_to_conversation_history(fallback_response, is_user=False)
            return fallback_response
        
        return None

    def _handle_student_mode_input(self, text: str, text_lower: str) -> Optional[str]:
        """Обработка ввода в режиме ученика до начала урока"""
        
        # Проверяем, не хочет ли ученик изучить конкретную тему по выбранному предмету
        if self._check_for_specific_topic_request(text_lower):
            print(f"🎯 Ученик запросил конкретную тему по предмету {self.current_subject}")
            return None  # Позволяем существующей логике сгенерировать урок
        
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
        
        # Используем существующую логику выбора предмета
        response = self._handle_subject_selection_direct(self.current_subject)
        
        if response is None:
            # Успешно начали урок
            student_name = self.student_data.get('name', '')
            greeting = f"{student_name}, " if student_name else ""
            start_message = f"{greeting}Отлично! Начинаем урок по {self.current_subject}. {self._get_next_paragraph()}"
            self._add_to_conversation_history(start_message, is_user=False)
            return start_message
        
        return response

    def _handle_subject_selection_direct(self, subject: str) -> Optional[str]:
        """Прямая обработка выбора предмета с инициализацией визуализации для ученика"""
        self.current_subject = subject
        lessons = self.lessons.get(subject, [])
        demo_lessons = [l for l in lessons if l.get('is_demo', False)]
        
        if demo_lessons:
            self.selected_lesson = demo_lessons[0]
        elif lessons:
            self.selected_lesson = lessons[0]
        else:
            self.selected_lesson = {
                'id': f"demo_{subject}",
                'title': f"Демо-урок по {subject}",
                'file_path': self.lessons_dir / f"demo_{subject}.txt",
                'is_demo': True
            }
        
        self.lesson_started = True
        self.current_state = "lesson_reading"
        self.current_paragraph = 0
        self.lesson_content = self._load_lesson_content(self.selected_lesson['file_path'])
        self.knowledge_base = KnowledgeBase(self.current_subject)
        
        # ИНИЦИАЛИЗАЦИЯ ВИЗУАЛИЗАЦИИ ПРИ СТАРТЕ УРОКА ДЛЯ УЧЕНИКА
        self._start_lesson_visualization(
            self.selected_lesson['id'],
            self.selected_lesson['title'],
            self.lesson_content
        )
        
        self.conversation_history = []
        self.conversation_context = []
        
        return None

    def _handle_greeting(self, text: str) -> Optional[str]:
        greeting_words = ["привет", "здравствуй", 'начать', "старт", " готов", "поехали", "давай", "началом"]
        if any(word in text for word in greeting_words):
            self.current_state = "subject_selection"
            prompt = self._get_student_lesson_prompt()
            return prompt if prompt else self.student_prompts["greeting"]
        return None

    def _handle_subject_selection(self, text: str) -> Optional[str]:
        # В режиме ученика предмет уже выбран, пропускаем этот шаг
        return self._start_student_lesson()

    def _handle_lesson_reading(self, text: str) -> Optional[str]:
        if any(word in text for word in ["стоп", "останови", "хватит", "закончи"]):
            self.lesson_started = False
            self.current_state = "greeting"
            self.conversation_counter = 0
            self.knowledge_base = None
            self.conversation_history = []
            self.conversation_context = []
            return "Урок остановлен. Скажи 'привет' когда захочешь продолжить."
            
        return None

    def _handle_practice_session(self, text: str) -> Optional[str]:
        if any(word in text for word in ["стоп", "останови", "хватит", "закончи"]):
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

    def _get_next_paragraph(self) -> Optional[str]:
        print(f"📄 Получение следующего абзаца ученика: текущий {self.current_paragraph}, всего {len(self.lesson_content)}")
        
        if self.current_paragraph < len(self.lesson_content):
            paragraph = self.lesson_content[self.current_paragraph]
            
            # ВИЗУАЛИЗАЦИЯ: Обрабатываем отображение для этого абзаца
            self._handle_paragraph_visualization(self.current_paragraph)
            
            self.current_paragraph += 1
            
            print(f"✅ Возвращаем абзац ученику {self.current_paragraph}: {paragraph[:100]}...")
            return paragraph
        else:
            print("🏁 Урок завершен для ученика, запускаем практику")
            practice_message = self._start_practice_session()
            return practice_message

    def _start_practice_session(self) -> str:
        """Запускает фазу практики с асинхронной генерацией вопросов"""
        self.lesson_started = False
        self.current_state = "practice_session"
        self.practice_active = True
        self.waiting_for_answer = False
        self.current_question_index = 0
        
        print("=== ЗАПУСК ФАЗЫ ПРАКТИКИ ДЛЯ УЧЕНИКА ===")
        
        # Инициализируем менеджер практики с асинхронной генерацией
        lesson_context = " ".join(self.lesson_content)
        self.practice_manager.initialize_practice_generation(lesson_context, self.current_subject)
        
        # Уведомляем клиентов о начале практики
        if self.room_id:
            self.socketio.emit('practice_started', {'room_id': self.room_id})
        
        # ПОЛУЧАЕМ ПЕРВЫЙ ВОПРОС ИЗ ОЧЕРЕДИ
        print("🔄 Получение первого вопроса практики...")
        first_question = self.practice_manager.get_next_question()
        
        if first_question:
            print(f"✅ Первый вопрос получен: {first_question}")
            self.waiting_for_answer = True
            self.current_practice_question = {
                "id": 1,
                "question": first_question,
                "answer": ""
            }
            student_name = self.student_data.get('name', '')
            greeting = f"{student_name}, " if student_name else ""
            return f"{greeting}Отлично! Переходим к практике. Первый вопрос: {first_question}"
        else:
            print("❌ Не удалось получить первый вопрос практики")
            self.practice_active = False
            return "Практические задания временно недоступны. Давайте продолжим урок или выберем другую тему."

    def _evaluate_and_generate_next(self, student_answer: str) -> str:
        """Оценивает ответ и возвращает следующий вопрос с асинхронной генерацией"""
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
        print("=== 🏁 ПРАКТИКА ЗАВЕРШЕНА ===")

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

    def get_selected_lesson(self) -> Optional[dict]:
        return self.selected_lesson

    def is_lesson_started(self) -> bool:
        return self.lesson_started

    def get_current_subject(self) -> Optional[str]:
        return self.current_subject

    def get_current_state(self) -> str:
        return self.current_state

    def reset(self):
        """Полный сброс диалог менеджера"""
        self.current_state = "greeting"
        self.selected_lesson = None
        self.lesson_started = False
        self.lesson_content = []
        self.current_paragraph = 0
        self.knowledge_base = None
        self.conversation_counter = 0
        self.conversation_history = []
        self.conversation_context = []
        self.practice_active = False
        self.waiting_for_answer = False
        self.current_question_index = 0
        self.practice_manager.reset()
        
        # Сброс счетчиков ученика
        self.student_conversation_count = 0
        self.student_lesson_started = False
        self.student_subject_prompted = False

        # Сброс визуализации
        self.current_visualization_type = None
        self.generated_mindmap = None
        self.current_slide_index = 0
        self.slides = None

    def get_available_subjects(self) -> List[str]:
        subjects = list(self.lessons.keys())
        if self.current_subject not in subjects:
            subjects.append(self.current_subject)
        return subjects

    def get_lessons_for_subject(self, subject: str) -> List[dict]:
        return self.lessons.get(subject, [])

    def set_llm_model(self, model: str):
        self.llm.set_model(model)
        print(f"Установлена модель LLM: {model}")

    def set_llm_mode(self, mode: str):
        if mode in ["traditional", "llm_first"]:
            self.llm_query_mode = mode
            print(f"Установлен режим LLM: {mode}")

    def get_knowledge_stats(self) -> Optional[Dict]:
        if self.knowledge_base:
            return self.knowledge_base.get_stats()
        return None

    def set_room_id(self, room_id: str):
        """Установка ID комнаты для WebSocket коммуникации"""
        self.room_id = room_id
        print(f"🔧 Установлен room_id для StudentDialogueManager: {room_id}")

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

    def get_visualization_status(self) -> Dict:
        """Возвращает статус визуализации"""
        return {
            "visualization_enabled": self.visualization_enabled,
            "visualization_counter": self.visualization_counter,
            "last_visualization_time": self.last_visualization_time,
            "paragraphs_since_last_viz": self.paragraphs_since_last_viz,
            "current_visualization_type": self.current_visualization_type,
            "current_slide_index": self.current_slide_index,
            "total_slides": len(self.slides) if self.slides else 0
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
            "available_subjects": self.get_available_subjects(),
            "llm_mode": self.llm_query_mode,
            "visualization_enabled": self.visualization_enabled,
            "conversation_history_length": len(self.conversation_history),
            "practice_stats": practice_stats,
            "current_practice_question": self.current_practice_question,
            "room_id": self.room_id,
            "questions_asked": len(self.practice_manager.generated_questions) if hasattr(self.practice_manager, 'generated_questions') else 0,
            "max_questions": self.max_questions,
            # Информация о ученике
            "student_data": self.student_data,
            "student_conversation_count": self.student_conversation_count,
            "student_lesson_started": self.student_lesson_started,
            "student_subject_prompted": self.student_subject_prompted,
            "age_group": self.student_prompts.get('age_group', 'unknown'),
            # Информация о визуализации
            "current_visualization_type": self.current_visualization_type,
            "has_mindmap": self.generated_mindmap is not None,
            "current_slide": self.current_slide_index,
            "total_slides": len(self.slides) if self.slides else 0
        }

    def get_system_status(self) -> Dict:
        """Возвращает общий статус системы"""
        llm_status = self.llm.get_llm_status() if hasattr(self.llm, 'get_llm_status') else {}
        knowledge_stats = self.get_knowledge_stats() or {}
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
            "knowledge_base": knowledge_stats,
            "practice": practice_stats,
            "visualization": self.get_visualization_status(),
            "lessons": {
                "available_subjects": self.get_available_subjects(),
                "total_lessons": sum(len(lessons) for lessons in self.lessons.values())
            }
        }

    def export_conversation_history(self) -> List[Dict]:
        """Экспортирует историю диалога"""
        return self.conversation_history.copy()

    def clear_conversation_history(self):
        """Очищает историю диалога"""
        self.conversation_history = []
        self.conversation_context = []
        print("🗑️ История диалога очищена")

    def simulate_student_answer(self, answer: str) -> str:
        """Симулирует ответ студента (для тестирования)"""
        if not self.practice_active or not self.waiting_for_answer:
            return "Практика не активна или система не ожидает ответа"
        
        return self._evaluate_and_generate_next(answer)

    def get_available_commands(self) -> Dict[str, str]:
        """Возвращает список доступных команд"""
        return {
            "привет": "Начать диалог",
            "продолжай": "Продолжить урок",
            "стоп": "Остановить урок/практику",
            "какой предмет": "Показать доступные предметы",
            "практика": "Перейти к практике (если урок завершен)",
            "статус": "Показать статус системы",
            "помощь": "Показать эту справку"
        }


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
