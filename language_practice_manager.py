import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
import time
import threading
import queue

class LanguagePracticeManager:
    def __init__(self, llm_integration, target_language='english', level='beginner'):
        self.llm = llm_integration
        self.target_language = target_language.lower()
        self.level = level.lower()
        self.practice_dir = Path("materials/practice/languages")
        
        # 🔥 СПЕЦИФИЧЕСКИЕ НАСТРОЙКИ ДЛЯ ЯЗЫКОВ
        self.language_config = self._get_language_config()
        self.current_lesson_topic = ""
        self.current_vocabulary = []
        self.current_grammar_rules = []
        
        # ОЧЕРЕДИ ДЛЯ АСИНХРОННОЙ ГЕНЕРАЦИИ
        self.exercise_queue = queue.Queue()
        self.generated_exercises = []
        self.current_exercise_index = 0
        self.max_exercises = 7  # Больше упражнений для языка
        
        # ФЛАГИ УПРАВЛЕНИЯ
        self.generation_thread = None
        self.stop_generation = False
        self.generation_active = False
        
        self.practice_dir.mkdir(parents=True, exist_ok=True)
        
        # 🔥 ЯЗЫКОВЫЕ ШАБЛОНЫ
        self.exercise_templates = {
            'vocabulary': self._generate_vocabulary_exercise,
            'translation': self._generate_translation_exercise,
            'grammar': self._generate_grammar_exercise,
            'dialogue': self._generate_dialogue_exercise,
            'listening': self._generate_listening_exercise,
            'pronunciation': self._generate_pronunciation_exercise,
            'fill_blank': self._generate_fill_blank_exercise,
            'sentence_building': self._generate_sentence_building_exercise
        }
        
        print(f"🎯 Инициализирован LanguagePracticeManager: {target_language}, уровень: {level}")
    
    def _get_language_config(self) -> Dict:
        """Конфигурация для разных языков и уровней"""
        configs = {
            'english': {
                'beginner': {
                    'exercise_types': ['vocabulary', 'translation', 'sentence_building'],
                    'sentence_complexity': 'simple',
                    'vocab_per_exercise': 5,
                    'bilingual_ratio': 0.3  # 30% английского
                },
                'intermediate': {
                    'exercise_types': ['grammar', 'dialogue', 'fill_blank'],
                    'sentence_complexity': 'medium',
                    'vocab_per_exercise': 7,
                    'bilingual_ratio': 0.5
                },
                'advanced': {
                    'exercise_types': ['listening', 'pronunciation', 'dialogue'],
                    'sentence_complexity': 'complex',
                    'vocab_per_exercise': 10,
                    'bilingual_ratio': 0.7
                }
            },
            'french': {
                'beginner': {
                    'exercise_types': ['vocabulary', 'translation'],
                    'sentence_complexity': 'simple',
                    'vocab_per_exercise': 5,
                    'bilingual_ratio': 0.3
                }
            }
            # Добавьте другие языки по аналогии
        }
        
        return configs.get(self.target_language, configs['english']).get(
            self.level, configs['english']['beginner']
        )
    
    def initialize_practice_generation(self, lesson_context: str, topic: str, vocabulary: List = None):
        """Инициализирует языковую практику"""
        self.current_lesson_topic = topic
        self.current_vocabulary = vocabulary or self._extract_vocabulary(lesson_context)
        self.current_grammar_rules = self._extract_grammar_rules(lesson_context)
        self.generated_exercises = []
        self.current_exercise_index = 0
        
        # Очищаем очередь
        while not self.exercise_queue.empty():
            try:
                self.exercise_queue.get_nowait()
            except queue.Empty:
                break
        
        print(f"🎯 Инициализирована языковая практика: {self.target_language}")
        print(f"📝 Тема: {topic}, Слов: {len(self.current_vocabulary)}")
        print(f"⚙️ Типы упражнений: {self.language_config['exercise_types']}")
        
        # Запускаем асинхронную генерацию
        self._start_async_generation()
        
        # Первое упражнение синхронно
        first_exercise = self.generate_single_exercise()
        if first_exercise:
            self.exercise_queue.put(first_exercise)
            print(f"✅ Первое упражнение готово: {first_exercise['type']}")
    
    def _start_async_generation(self):
        """Запускает фоновую генерацию упражнений"""
        if self.generation_active:
            return
            
        self.stop_generation = False
        self.generation_active = True
        
        def generate_exercises_worker():
            print("🔄 Фоновая генерация языковых упражнений запущена...")
            
            while (not self.stop_generation and 
                   self.generation_active and 
                   len(self.generated_exercises) < self.max_exercises - 1):
                
                try:
                    if len(self.generated_exercises) >= self.max_exercises:
                        print("🏁 Достигнут лимит упражнений")
                        break
                    
                    exercise = self.generate_single_exercise()
                    
                    if exercise:
                        self.exercise_queue.put(exercise)
                        print(f"✅ Фоново сгенерировано упражнение {len(self.generated_exercises)}: {exercise['type']}")
                    
                    time.sleep(1.5)  # Больше времени для языковых упражнений
                    
                except Exception as e:
                    print(f"❌ Ошибка в фоновой генерации: {e}")
                    time.sleep(2)
            
            print("🏁 Фоновая генерация упражнений завершена")
            self.generation_active = False
        
        self.generation_thread = threading.Thread(target=generate_exercises_worker, daemon=True)
        self.generation_thread.start()
    
    def generate_single_exercise(self) -> Optional[Dict]:
        """Генерирует одно языковое упражнение"""
        try:
            if len(self.generated_exercises) >= self.max_exercises:
                return None
            
            # Выбираем тип упражнения
            available_types = self.language_config['exercise_types']
            used_types = [e['type'] for e in self.generated_exercises]
            
            # Предпочитаем неиспользованные типы
            for ex_type in available_types:
                if ex_type not in used_types:
                    exercise_type = ex_type
                    break
            else:
                # Если все типы использованы, выбираем случайный
                import random
                exercise_type = random.choice(available_types)
            
            # Генерируем упражнение выбранного типа
            generator = self.exercise_templates.get(exercise_type)
            if not generator:
                generator = self._generate_vocabulary_exercise
            
            exercise = generator()
            
            if exercise:
                exercise['id'] = len(self.generated_exercises) + 1
                exercise['generated_at'] = time.time()
                self.generated_exercises.append(exercise)
            
            return exercise
            
        except Exception as e:
            print(f"❌ Ошибка генерации языкового упражнения: {e}")
            return self._get_fallback_exercise()
    
    def _generate_vocabulary_exercise(self) -> Dict:
        """Упражнение на словарный запас"""
        try:
            if not self.current_vocabulary:
                return self._get_fallback_exercise()
            
            # Выбираем слова для упражнения
            import random
            words_to_practice = random.sample(
                self.current_vocabulary, 
                min(self.language_config['vocab_per_exercise'], len(self.current_vocabulary))
            )
            
            prompt = f"""
            Создай упражнение на словарный запас для изучения {self.target_language}.
            
            Уровень: {self.level}
            Слова для отработки: {', '.join(words_to_practice)}
            Тема урока: {self.current_lesson_topic}
            
            Создай упражнение одного из типов:
            1. Сопоставление слов с переводами
            2. Заполнение пропусков в предложениях
            3. Составление слов из букв
            4. Выбор правильного перевода
            
            Формат упражнения должен быть понятным и интерактивным.
            Включи ответы для проверки.
            
            Верни упражнение в формате:
            ТИП: [тип упражнения]
            ВОПРОС: [текст упражнения]
            ВАРИАНТЫ: [если есть варианты ответов]
            ПРАВИЛЬНЫЙ_ОТВЕТ: [правильный ответ]
            ОБЪЯСНЕНИЕ: [краткое объяснение на русском]
            """
            
            llm_response = self.llm.query(
                question=prompt,
                context="",
                subject=f"{self.target_language} language"
            )
            
            if llm_response:
                return {
                    'type': 'vocabulary',
                    'content': llm_response,
                    'target_language': self.target_language,
                    'words': words_to_practice
                }
            
        except Exception as e:
            print(f"❌ Ошибка генерации словарного упражнения: {e}")
        
        return self._get_fallback_vocabulary_exercise()
    
    def _generate_translation_exercise(self) -> Dict:
        """Упражнение на перевод"""
        try:
            prompt = f"""
            Создай упражнение на перевод для изучения {self.target_language}.
            
            Уровень: {self.level}
            Тема: {self.current_lesson_topic}
            Направление перевода: с русского на {self.target_language}
            
            Создай 3-5 предложений для перевода, соответствующих уровню {self.level}.
            Включи подсказки для сложных конструкций.
            Укажи правильные переводы для проверки.
            
            Пример для уровня beginner:
            Исходное: "Я учу английский язык."
            Перевод: "I am learning English."
            
            Верни упражнение в формате:
            ТИП: translation
            ИНСТРУКЦИЯ: [инструкция на русском]
            ПРЕДЛОЖЕНИЯ: [список предложений для перевода]
            ПРАВИЛЬНЫЕ_ПЕРЕВОДЫ: [правильные переводы]
            ПОДСКАЗКИ: [грамматические подсказки]
            """
            
            llm_response = self.llm.query(
                question=prompt,
                context="",
                subject=f"{self.target_language} language"
            )
            
            if llm_response:
                return {
                    'type': 'translation',
                    'content': llm_response,
                    'direction': 'ru_to_target',
                    'difficulty': self.level
                }
                
        except Exception as e:
            print(f"❌ Ошибка генерации упражнения на перевод: {e}")
        
        return self._get_fallback_translation_exercise()
    
    def _generate_grammar_exercise(self) -> Dict:
        """Упражнение на грамматику"""
        try:
            grammar_rules = self.current_grammar_rules or ["present simple", "basic word order"]
            
            prompt = f"""
            Создай упражнение на грамматику {self.target_language}.
            
            Уровень: {self.level}
            Грамматическая тема: {grammar_rules[0] if grammar_rules else 'basic grammar'}
            Тема урока: {self.current_lesson_topic}
            
            Создай упражнение на отработку грамматического правила.
            Типы упражнений:
            1. Выбор правильной формы глагола
            2. Расстановка слов в правильном порядке
            3. Исправление ошибок в предложениях
            4. Заполнение пропусков правильными словами
            
            Включи объяснение правила на русском.
            Укажи правильные ответы.
            
            Верни упражнение в формате:
            ТИП: grammar
            ПРАВИЛО: [грамматическое правило на русском]
            УПРАЖНЕНИЕ: [текст упражнения]
            ПРАВИЛЬНЫЕ_ОТВЕТЫ: [правильные ответы]
            ОБЪЯСНЕНИЕ: [подробное объяснение ошибок]
            """
            
            llm_response = self.llm.query(
                question=prompt,
                context="",
                subject=f"{self.target_language} language"
            )
            
            if llm_response:
                return {
                    'type': 'grammar',
                    'content': llm_response,
                    'grammar_rule': grammar_rules[0] if grammar_rules else 'basic',
                    'difficulty': self.level
                }
                
        except Exception as e:
            print(f"❌ Ошибка генерации грамматического упражнения: {e}")
        
        return self._get_fallback_grammar_exercise()
    
    def _generate_dialogue_exercise(self) -> Dict:
        """Упражнение с диалогом"""
        try:
            prompt = f"""
            Создай упражнение с диалогом на {self.target_language}.
            
            Уровень: {self.level}
            Тема: {self.current_lesson_topic}
            Ситуация: повседневное общение
            
            Создай:
            1. Естественный диалог из 4-6 реплик
            2. Вопросы на понимание диалога
            3. Задание на заполнение пропусков в диалоге
            4. Ролевую игру для отработки
            
            Все на {self.target_language} с русскими пояснениями.
            
            Верни упражнение в формате:
            ТИП: dialogue
            ДИАЛОГ: [текст диалога]
            ВОПРОСЫ: [вопросы на понимание]
            ЗАДАНИЕ: [задание для студента]
            РОЛЕВАЯ_ИГРА: [инструкция для ролевой игры]
            """
            
            llm_response = self.llm.query(
                question=prompt,
                context="",
                subject=f"{self.target_language} language"
            )
            
            if llm_response:
                return {
                    'type': 'dialogue',
                    'content': llm_response,
                    'language': self.target_language,
                    'situation': 'everyday_conversation'
                }
                
        except Exception as e:
            print(f"❌ Ошибка генерации диалогового упражнения: {e}")
        
        return self._get_fallback_dialogue_exercise()
    
    def _generate_listening_exercise(self) -> Dict:
        """Упражнение на аудирование"""
        try:
            prompt = f"""
            Создай упражнение на аудирование для {self.target_language}.
            
            Уровень: {self.level}
            Тема: {self.current_lesson_topic}
            
            Создай:
            1. Текст для "прослушивания" (студент будет читать)
            2. Вопросы на понимание текста
            3. Упражнение "верно/неверно"
            4. Задание на заполнение пропусков
            
            Текст должен соответствовать уровню {self.level}.
            Включи "транскрипцию" с правильными ответами.
            
            Верни упражнение в формате:
            ТИП: listening
            ТЕКСТ: [текст для прослушивания на {self.target_language}]
            ВОПРОСЫ: [вопросы на понимание]
            ВЕРНО_НЕВЕРНО: [утверждения для проверки]
            ЗАПОЛНИ_ПРОПУСКИ: [текст с пропусками]
            ТРАНСКРИПЦИЯ: [полный текст с ответами]
            """
            
            llm_response = self.llm.query(
                question=prompt,
                context="",
                subject=f"{self.target_language} language"
            )
            
            if llm_response:
                return {
                    'type': 'listening',
                    'content': llm_response,
                    'language': self.target_language,
                    'difficulty': self.level
                }
                
        except Exception as e:
            print(f"❌ Ошибка генерации упражнения на аудирование: {e}")
        
        return self._get_fallback_listening_exercise()
    
    def _generate_pronunciation_exercise(self) -> Dict:
        """Упражнение на произношение"""
        try:
            prompt = f"""
            Создай упражнение на произношение для {self.target_language}.
            
            Уровень: {self.level}
            Тема: {self.current_lesson_topic}
            
            Создай упражнение, включающее:
            1. Слова и фразы с сложным произношением
            2. Транскрипцию в формате IPA или русскими буквами
            3. Мини-диалоги для отработки интонации
            4. Скороговорки или tongue twisters
            
            Сфокусируйся на типичных проблемах русскоговорящих учеников.
            
            Верни упражнение в формате:
            ТИП: pronunciation
            СЛОВА: [слова для отработки с транскрипцией]
            ФРАЗЫ: [фразы для произношения]
            СКОРОГОВОРКИ: [tongue twisters]
            СОВЕТЫ: [советы по произношению на русском]
            """
            
            llm_response = self.llm.query(
                question=prompt,
                context="",
                subject=f"{self.target_language} language"
            )
            
            if llm_response:
                return {
                    'type': 'pronunciation',
                    'content': llm_response,
                    'language': self.target_language,
                    'focus': 'typical_errors'
                }
                
        except Exception as e:
            print(f"❌ Ошибка генерации упражнения на произношение: {e}")
        
        return self._get_fallback_pronunciation_exercise()
    
    def _generate_fill_blank_exercise(self) -> Dict:
        """Упражнение "заполни пропуски" (fill in the blanks)"""
        try:
            prompt = f"""
            Создай упражнение "fill in the blanks" для {self.target_language}.
            
            Уровень: {self.level}
            Тема: {self.current_lesson_topic}
            
            Создай текст с пропущенными словами.
            Пропуски должны быть ключевыми словами по теме.
            Предоставьте банк слов для выбора или оставьте для самостоятельного заполнения.
            Укажите правильные ответы.
            
            Верни упражнение в формате:
            ТИП: fill_blank
            ТЕКСТ: [текст с пропусками обозначенными как ___]
            БАНК_СЛОВ: [слова для выбора, если нужно]
            ПРАВИЛЬНЫЕ_ОТВЕТЫ: [правильные слова для пропусков]
            """
            
            llm_response = self.llm.query(
                question=prompt,
                context="",
                subject=f"{self.target_language} language"
            )
            
            if llm_response:
                return {
                    'type': 'fill_blank',
                    'content': llm_response,
                    'language': self.target_language,
                    'difficulty': self.level
                }
                
        except Exception as e:
            print(f"❌ Ошибка генерации упражнения fill_blank: {e}")
        
        return self._get_fallback_fill_blank_exercise()
    
    def _generate_sentence_building_exercise(self) -> Dict:
        """Упражнение "составь предложение" (sentence building)"""
        try:
            prompt = f"""
            Создай упражнение "sentence building" для {self.target_language}.
            
            Уровень: {self.level}
            Тема: {self.current_lesson_topic}
            
            Создай упражнение, где нужно составить предложения из данных слов.
            Слова должны быть в перемешанном порядке.
            Включи 3-5 предложений разной сложности.
            Укажи правильный порядок слов.
            
            Верни упражнение в формате:
            ТИП: sentence_building
            СЛОВА_ДЛЯ_ПРЕДЛОЖЕНИЙ: [слова в перемешанном порядке]
            ПРАВИЛЬНЫЙ_ПОРЯДОК: [правильный порядок слов]
            ПОДСКАЗКИ: [грамматические подсказки]
            """
            
            llm_response = self.llm.query(
                question=prompt,
                context="",
                subject=f"{self.target_language} language"
            )
            
            if llm_response:
                return {
                    'type': 'sentence_building',
                    'content': llm_response,
                    'language': self.target_language,
                    'difficulty': self.level
                }
                
        except Exception as e:
            print(f"❌ Ошибка генерации упражнения sentence_building: {e}")
        
        return self._get_fallback_sentence_building_exercise()
    
    # 🔥 FALLBACK УПРАЖНЕНИЯ
    def _get_fallback_vocabulary_exercise(self) -> Dict:
        return {
            'type': 'vocabulary',
            'content': f"""ТИП: matching
ВОПРОС: Сопоставьте слова на {self.target_language} с их переводом на русский.
СЛОВА: {', '.join(self.current_vocabulary[:3]) if self.current_vocabulary else 'hello, goodbye, thank you'}
ПРАВИЛЬНЫЕ_ОТВЕТЫ: [сопоставьте самостоятельно]
ОБЪЯСНЕНИЕ: Практикуйте эти слова ежедневно.""",
            'is_fallback': True
        }
    
    def _get_fallback_translation_exercise(self) -> Dict:
        return {
            'type': 'translation',
            'content': f"""ТИП: translation
ИНСТРУКЦИЯ: Переведите предложения с русского на {self.target_language}.
ПРЕДЛОЖЕНИЯ: ["Я учу {self.target_language}.", "Меня зовут...", "Я живу в..."]
ПРАВИЛЬНЫЕ_ПЕРЕВОДЫ: [проверьте в словаре]
ПОДСКАЗКИ: Используйте простые времена.""",
            'is_fallback': True
        }
    
    # ... аналогичные fallback для других типов упражнений ...
    
    def _get_fallback_exercise(self) -> Dict:
        """Общее fallback упражнение"""
        import random
        fallback_types = [
            self._get_fallback_vocabulary_exercise,
            self._get_fallback_translation_exercise,
            self._get_fallback_grammar_exercise
        ]
        return random.choice(fallback_types)()
    
    def get_next_exercise(self, timeout: float = 10.0) -> Optional[Dict]:
        """Получает следующее упражнение"""
        try:
            if len(self.generated_exercises) >= self.max_exercises:
                print(f"🏁 Достигнут лимит упражнений: {len(self.generated_exercises)}/{self.max_exercises}")
                return None
            
            try:
                exercise = self.exercise_queue.get_nowait()
                print(f"✅ Упражнение взято из очереди: {exercise['type']}")
                return exercise
            except queue.Empty:
                pass
            
            # Синхронная генерация
            exercise = self.generate_single_exercise()
            
            if exercise:
                print(f"✅ Синхронно сгенерировано упражнение: {exercise['type']}")
                return exercise
            else:
                return self._get_fallback_exercise()
                
        except Exception as e:
            print(f"❌ Ошибка получения упражнения: {e}")
            return self._get_fallback_exercise()
    
    def evaluate_language_answer(self, student_answer: str, exercise: Dict) -> Tuple[str, bool]:
        """Оценивает ответ на языковое упражнение"""
        try:
            exercise_type = exercise.get('type', 'vocabulary')
            exercise_content = exercise.get('content', '')
            
            prompt = f"""
            Оцени ответ ученика на языковое упражнение.
            
            ТИП УПРАЖНЕНИЯ: {exercise_type}
            УПРАЖНЕНИЕ: {exercise_content}
            ОТВЕТ УЧЕНИКА: {student_answer}
            ЯЗЫК: {self.target_language}
            УРОВЕНЬ: {self.level}
            
            Проанализируй ответ и дай обратную связь:
            1. Правильность ответа (да/нет)
            2. Конструктивная обратная связь на русском
            3. Исправление ошибок, если есть
            4. Похвала за правильные элементы
            
            Верни в формате:
            ПРАВИЛЬНО: [true/false]
            ОБРАТНАЯ_СВЯЗЬ: [развернутая обратная связь на русском]
            ИСПРАВЛЕНИЯ: [исправленный вариант, если нужно]
            """
            
            llm_response = self.llm.query(
                question=prompt,
                context="",
                subject=f"{self.target_language} language"
            )
            
            if llm_response:
                # Парсим ответ
                is_correct = 'true' in llm_response.lower() or 'правильно' in llm_response.lower()
                
                # Извлекаем обратную связь
                feedback_match = re.search(r'ОБРАТНАЯ_СВЯЗЬ:\s*(.+)', llm_response, re.IGNORECASE | re.DOTALL)
                feedback = feedback_match.group(1).strip() if feedback_match else "Спасибо за ответ!"
                
                return feedback, is_correct
            
            return "Спасибо за ответ! Продолжайте практиковаться.", True
            
        except Exception as e:
            print(f"❌ Ошибка оценки языкового ответа: {e}")
            return "Спасибо за ответ! Это хорошая попытка.", True
    
    def _extract_vocabulary(self, lesson_context: str) -> List[str]:
        """Извлекает словарный запас из контекста урока"""
        try:
            prompt = f"""
            Извлеки ключевые слова и фразы из текста урока.
            
            ТЕКСТ: {lesson_context[:1000]}
            ЯЗЫК: {self.target_language}
            УРОВЕНЬ: {self.level}
            
            Верни только список слов/фраз на {self.target_language}, разделенных запятыми.
            Выбери 10-15 самых важных для изучения.
            """
            
            response = self.llm.query(
                question=prompt,
                context="",
                subject=f"{self.target_language} language"
            )
            
            if response:
                words = [w.strip() for w in response.split(',')]
                return words[:15]
            
        except Exception as e:
            print(f"❌ Ошибка извлечения словаря: {e}")
        
        # Fallback слова
        fallback_vocab = {
            'english': ['hello', 'goodbye', 'thank you', 'please', 'sorry', 
                       'yes', 'no', 'my name is', 'how are you', 'I am'],
            'french': ['bonjour', 'au revoir', 'merci', 's\'il vous plaît', 'pardon',
                      'oui', 'non', 'je m\'appelle', 'comment ça va', 'je suis']
        }
        
        return fallback_vocab.get(self.target_language, fallback_vocab['english'])
    
    def _extract_grammar_rules(self, lesson_context: str) -> List[str]:
        """Извлекает грамматические правила из контекста"""
        # Упрощенная реализация
        grammar_by_level = {
            'beginner': ['present simple', 'basic word order', 'pronouns'],
            'intermediate': ['past tense', 'future tense', 'modal verbs'],
            'advanced': ['conditionals', 'reported speech', 'passive voice']
        }
        
        return grammar_by_level.get(self.level, grammar_by_level['beginner'])
    
    def get_practice_stats(self) -> Dict:
        """Статистика языковой практики"""
        exercise_types = {}
        for ex in self.generated_exercises:
            ex_type = ex.get('type', 'unknown')
            exercise_types[ex_type] = exercise_types.get(ex_type, 0) + 1
        
        return {
            'total_exercises': len(self.generated_exercises),
            'max_exercises': self.max_exercises,
            'exercises_in_queue': self.exercise_queue.qsize(),
            'generation_active': self.generation_active,
            'target_language': self.target_language,
            'level': self.level,
            'exercise_types': exercise_types,
            'vocabulary_size': len(self.current_vocabulary),
            'current_topic': self.current_lesson_topic,
            'bilingual_ratio': self.language_config.get('bilingual_ratio', 0.3)
        }
    
    def reset(self):
        """Сброс состояния"""
        self.stop_async_generation()
        self.current_lesson_topic = ""
        self.current_vocabulary = []
        self.current_grammar_rules = []
        self.generated_exercises = []
        self.current_exercise_index = 0
        while not self.exercise_queue.empty():
            try:
                self.exercise_queue.get_nowait()
            except queue.Empty:
                break
        print(f"🔄 LanguagePracticeManager сброшен для {self.target_language}")
    
    def stop_async_generation(self):
        """Остановка фоновой генерации"""
        self.stop_generation = True
        self.generation_active = False
        if self.generation_thread and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=2.0)
