import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
import time
import threading
import queue

class PracticeManager:
    def __init__(self, llm_integration):
        self.llm = llm_integration
        self.practice_dir = Path("materials/practice")
        self.current_lesson_context = ""
        self.current_lesson_summary = ""
        self.current_subject = ""
        
        # 🔥 НОВОЕ: Данные ученика для адаптации сложности
        self.student_data = {}
        
        # ОЧЕРЕДИ ДЛЯ АСИНХРОННОЙ ГЕНЕРАЦИИ
        self.question_queue = queue.Queue()  # Очередь готовых вопросов
        self.generated_questions = []        # История всех вопросов
        self.current_question_index = 0
        self.max_questions = 5  # ЖЕСТКИЙ ЛИМИТ 5 ВОПРОСОВ
        
        # ФЛАГИ УПРАВЛЕНИЯ АСИНХРОННОЙ ГЕНЕРАЦИЕЙ
        self.generation_thread = None
        self.stop_generation = False
        self.generation_active = False
        
        self.practice_dir.mkdir(parents=True, exist_ok=True)

    def initialize_practice_generation(self, lesson_context: str, subject: str):
        """Инициализирует практику и ЗАРАНЕЕ начинает генерацию вопросов"""
        self.current_lesson_context = lesson_context
        self.current_subject = subject
        self.current_lesson_summary = self._generate_lesson_summary(lesson_context)
        self.generated_questions = []
        self.current_question_index = 0
        
        # Очищаем очередь от предыдущих вопросов
        while not self.question_queue.empty():
            try:
                self.question_queue.get_nowait()
            except queue.Empty:
                break
        
        print(f"🎯 Инициализирована генерация практики для предмета: {subject}")
        print(f"📝 Максимальное количество вопросов: {self.max_questions}")
        
        # 🔥 НОВОЕ: Логируем данные ученика для отладки
        if self.student_data:
            age = self.student_data.get('age', 'неизвестен')
            level = self.student_data.get('level', 'неизвестен')
            print(f"👤 Данные ученика: возраст {age} лет, {level} класс")
        
        # ЗАПУСКАЕМ АСИНХРОННУЮ ГЕНЕРАЦИЮ ВОПРОСОВ СРАЗУ
        self._start_async_generation()
        
        # Генерируем ПЕРВЫЙ вопрос СИНХРОННО для немедленного старта
        first_question = self.generate_single_question()
        if first_question:
            self.question_queue.put(first_question)
            print(f"✅ Первый вопрос готов: {first_question[:80]}...")

    def _start_async_generation(self):
        """Запускает фоновую генерацию вопросов"""
        if self.generation_active:
            return
            
        self.stop_generation = False
        self.generation_active = True
        
        def generate_questions_worker():
            print("🔄 Фоновая генерация вопросов запущена...")
            
            while (not self.stop_generation and 
                   self.generation_active and 
                   len(self.generated_questions) < self.max_questions - 1):  # -1 потому что первый уже сгенерирован
                
                try:
                    # ВАЖНОЕ ИЗМЕНЕНИЕ: Проверяем лимит перед генерацией
                    if len(self.generated_questions) >= self.max_questions:
                        print("🏁 Достигнут лимит вопросов в фоновой генерации")
                        break
                    
                    # Генерируем следующий вопрос
                    next_question = self.generate_single_question()
                    
                    if next_question:
                        self.question_queue.put(next_question)
                        print(f"✅ Фоново сгенерирован вопрос {len(self.generated_questions)}/{self.max_questions}: {next_question[:80]}...")
                    
                    # Небольшая пауза между генерацией
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"❌ Ошибка в фоновой генерации: {e}")
                    time.sleep(2)  # Пауза при ошибке
            
            print("🏁 Фоновая генерация вопросов завершена")
            self.generation_active = False
        
        # Запускаем в отдельном потоке
        self.generation_thread = threading.Thread(target=generate_questions_worker, daemon=True)
        self.generation_thread.start()

    def get_next_question(self, timeout: float = 10.0) -> Optional[str]:
        """Получает следующий вопрос из очереди (с ожиданием если нужно)"""
        try:
            # ВАЖНОЕ ИЗМЕНЕНИЕ: Проверяем лимит вопросов
            if len(self.generated_questions) >= self.max_questions:
                print(f"🏁 Достигнут лимит вопросов: {len(self.generated_questions)}/{self.max_questions}")
                return None
            
            # Пытаемся взять вопрос из очереди без ожидания
            try:
                question = self.question_queue.get_nowait()
                print(f"✅ Вопрос взят из очереди (в очереди еще: {self.question_queue.qsize()})")
                return question
            except queue.Empty:
                pass
            
            # Если очередь пуста, пробуем сгенерировать СИНХРОННО
            print("⚠️ Очередь вопросов пуста, синхронная генерация...")
            question = self.generate_single_question()
            
            if question:
                print(f"✅ Синхронно сгенерирован вопрос: {question[:80]}...")
                return question
            else:
                # Если синхронная генерация не удалась, используем fallback
                fallback = self._get_fallback_question(ensure_unique=True)
                print(f"🔄 Использован fallback вопрос: {fallback[:80]}...")
                return fallback
                
        except Exception as e:
            print(f"❌ Ошибка получения следующего вопроса: {e}")
            fallback = self._get_fallback_question(ensure_unique=True)
            print(f"🔄 Использован fallback после ошибки: {fallback[:80]}...")
            return fallback

    def generate_single_question(self) -> Optional[str]:
        """Генерирует один вопрос (синхронно) с учетом возраста ученика"""
        try:
            # ВАЖНОЕ ИЗМЕНЕНИЕ: Проверяем лимит вопросов
            if len(self.generated_questions) >= self.max_questions:
                print(f"🏁 Достигнут лимит вопросов: {len(self.generated_questions)}/{self.max_questions}")
                return None
            
            # 🔥 ОБНОВЛЕННЫЙ ПРОМТ: Добавляем данные ученика для адаптации сложности
            age = self.student_data.get('age', '12')
            level = self.student_data.get('level', '5')
            name = self.student_data.get('name', 'ученик')
            
            # УЛУЧШЕННЫЙ промт с историей вопросов и учетом возраста
            previous_questions = self._get_previous_questions_text()
            
            prompt = f"""
            Создай ОДИН УНИКАЛЬНЫЙ учебный вопрос для проверки понимания темы.

            ПАРАМЕТРЫ УЧЕНИКА:
            - Имя: {name}
            - Возраст: {age} лет
            - Уровень образования: {level} класс
            - Предмет: {self.current_subject}

            КРАТКОЕ СОДЕРЖАНИЕ УРОКА:
            {self.current_lesson_summary}

            УЖЕ ЗАДАННЫЕ ВОПРОСЫ (НЕ ПОВТОРЯЙ ИХ!):
            {previous_questions}

            ТРЕБОВАНИЯ К ВОПРОСУ:
            1. СООТВЕТСТВИЕ ВОЗРАСТУ {age} ЛЕТ:
               - Сложность вопроса должна быть адекватна для {age}-летнего
               - Формулировки должны быть понятны ученику {level} класса
               - Используй язык и примеры, релевантные для этого возраста

            2. ПЕДАГОГИЧЕСКИЕ ТРЕБОВАНИЯ:
               - Вопрос ДОЛЖЕН быть УНИКАЛЬНЫМ и не похожим на уже заданные
               - Проверяй понимание РАЗНЫХ аспектов материала
               - Вопрос должен требовать развернутого ответа
               - Будь конкретным и четким

            3. ФОРМАТ:
               - Только один вопрос
               - Без нумерации и лишних слов
               - Вопрос должен быть сформулирован ясно и однозначно

            Создай вопрос, который будет интересен и посилен для ученика {age} лет.
            """

            # УВЕЛИЧИВАЕМ ТАЙМАУТ для LLM запроса
            llm_response = self.llm.query(
                question=prompt,
                context="",
                subject=self.current_subject
            )
            
            if llm_response and not llm_response.startswith("Спасибо за вопрос!"):
                question = self._clean_question_text(llm_response)
                
                if question and len(question.strip()) > 10 and self._is_question_unique(question):
                    # Сохраняем в историю
                    self.generated_questions.append({
                        "question": question,
                        "generated_at": time.time(),
                        "type": "general",
                        "age_adapted": True  # 🔥 НОВОЕ: Отмечаем, что вопрос адаптирован по возрасту
                    })
                    return question
            
            # Fallback если не удалось сгенерировать
            return self._get_fallback_question_for_age(age, ensure_unique=True)
                
        except Exception as e:
            print(f"❌ Ошибка генерации вопроса: {e}")
            age = self.student_data.get('age', '12')
            return self._get_fallback_question_for_age(age, ensure_unique=True)

    def evaluate_and_continue(self, student_answer: str, current_question: str) -> Tuple[str, Optional[str]]:
        """Оценивает ответ и возвращает feedback + следующий вопрос с учетом возраста"""
        try:
            # 1. Сначала оцениваем ответ (это быстро)
            feedback = self.evaluate_single_answer(student_answer, current_question)
            
            # 2. Параллельно получаем следующий вопрос
            next_question = self.get_next_question()
            
            return feedback, next_question
            
        except Exception as e:
            print(f"❌ Ошибка в evaluate_and_continue: {e}")
            # ИСПРАВЛЕННЫЙ FALLBACK ДЛЯ ПРАКТИКИ
            feedback = "Спасибо за ответ! Переходим к следующему вопросу."
            age = self.student_data.get('age', '12')
            next_question = self._get_fallback_question_for_age(age, ensure_unique=True)
            return feedback, next_question

    def evaluate_single_answer(self, student_answer: str, question: str) -> str:
        """🔥 ОБНОВЛЕННЫЙ МЕТОД: Оценивает ответ ученика с учетом возраста"""
        try:
            if not student_answer or len(student_answer.strip()) < 2:
                return "Ответ слишком короткий. Пожалуйста, попробуйте ответить более развернуто."
            
            command_words = ['продолжай', 'дальше', 'следующий']
            if any(cmd in student_answer.lower() for cmd in command_words):
                return "Это похоже на команду. Пожалуйста, дайте ответ на вопрос."
            
            correct_answer = self._generate_correct_answer(question)
            if not correct_answer:
                correct_answer = "Информация содержится в учебном материале."
            
            evaluation = self._evaluate_with_llm_context(student_answer, question, correct_answer)
            
            # ИСПРАВЛЕНИЕ: Заменяем неподходящие fallback ответы
            if not evaluation or any(phrase in evaluation for phrase in [
                "Хороший вопрос! Давайте разберем эту тему подробнее",
                "Мне нужно немного времени подумать",
                "Спасибо за вопрос! Я подумаю над ответом"
            ]):
                evaluation = f"Спасибо за ответ! Правильный ответ: {correct_answer}"
            
            return evaluation
            
        except Exception as e:
            print(f"❌ Ошибка оценки ответа: {e}")
            # ИСПРАВЛЕННЫЙ FALLBACK
            return "Спасибо за ответ! Переходим к следующему вопросу."

    def _generate_correct_answer(self, question: str) -> Optional[str]:
        """Генерирует правильный ответ на вопрос с учетом возраста"""
        try:
            # 🔥 ОБНОВЛЕННЫЙ ПРОМТ: Учитываем возраст ученика
            age = self.student_data.get('age', '12')
            level = self.student_data.get('level', '5')
            
            prompt = f"""
            Дай точный и краткий ответ на вопрос, адаптированный для ученика {age} лет, {level} класс.

            ВОПРОС: {question}
            КОНТЕКСТ: {self.current_lesson_summary}

            Требования:
            - Ответ должен быть точным и соответствовать материалу
            - Ответ должен быть кратким (1-2 предложения)
            - Используй язык, понятный для {age}-летнего
            - Объясняй сложные понятия простыми словами
            - Без лишних объяснений

            Верни только ответ.
            """
            
            llm_response = self.llm.query(
                question=prompt,
                context="",
                subject=self.current_subject
            )
            return llm_response.strip() if llm_response else None
            
        except Exception as e:
            print(f"❌ Ошибка генерации правильного ответа: {e}")
            return None

    def _evaluate_with_llm_context(self, student_answer: str, question: str, correct_answer: str) -> str:
        """🔥 ОБНОВЛЕННЫЙ МЕТОД: Оценивает ответ через LLM с учетом возраста"""
        try:
            # 🔥 НОВОЕ: Получаем данные ученика для персонализации
            age = self.student_data.get('age', '12')
            level = self.student_data.get('level', '5')
            name = self.student_data.get('name', 'ученик')
            
            prompt = f"""
            Оцени ответ ученика на вопрос и дай обратную связь.

            ПАРАМЕТРЫ УЧЕНИКА:
            - Имя: {name}
            - Возраст: {age} лет
            - Уровень: {level} класс

            ВОПРОС: {question}
            ПРАВИЛЬНЫЙ ОТВЕТ: {correct_answer}
            ОТВЕТ УЧЕНИКА: {student_answer}

            Контекст урока: {self.current_lesson_summary[:300]}

            Дай добрую и поддерживающую обратную связь (2-3 предложения).
            Обращайся к ученику на "ты".
            Учитывай возраст {age} лет - будь терпеливым и понятным.

            НЕ ИСПОЛЬЗУЙ фразы:
            - "Хороший вопрос! Давайте разберем эту тему подробнее"
            - "Мне нужно немного времени подумать" 
            - "Спасибо за вопрос! Я подумаю над ответом"
            - "Давайте разберем эту тему подробнее"

            Вместо этого дай конкретную обратную связь по ответу, адаптированную для возраста {age} лет.
            """
            
            evaluation = self.llm.query(
                question=prompt,
                context="",
                subject=self.current_subject
            )
            
            # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА НА НЕПОДХОДЯЩИЕ ОТВЕТЫ
            if evaluation and any(phrase in evaluation for phrase in [
                "Хороший вопрос! Давайте разберем эту тему подробнее",
                "Мне нужно немного времени подумать",
                "Спасибо за вопрос! Я подумаю над ответом"
            ]):
                return f"Спасибо за ответ! Правильный ответ: {correct_answer}"
                
            return evaluation if evaluation else f"Спасибо за ответ! Правильный ответ: {correct_answer}"
            
        except Exception as e:
            print(f"❌ Ошибка оценки через LLM: {e}")
            return f"Спасибо за ответ! Правильный ответ: {correct_answer}"

    def stop_async_generation(self):
        """Останавливает фоновую генерацию"""
        self.stop_generation = True
        self.generation_active = False
        if self.generation_thread and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=2.0)

    def _get_previous_questions_text(self) -> str:
        """Возвращает текст уже заданных вопросов"""
        if not self.generated_questions:
            return "Вопросов еще не было. Это первый вопрос."
        
        questions_text = "Уже заданные вопросы:\n"
        for i, q_data in enumerate(self.generated_questions, 1):
            questions_text += f"{i}. {q_data['question']}\n"
        return questions_text

    def _is_question_unique(self, new_question: str, similarity_threshold: float = 0.7) -> bool:
        """Проверяет, является ли вопрос уникальным"""
        if not self.generated_questions:
            return True
        
        new_question_lower = new_question.lower()
        for existing_q in self.generated_questions:
            existing_question_lower = existing_q["question"].lower()
            similarity = SequenceMatcher(None, new_question_lower, existing_question_lower).ratio()
            
            if similarity > similarity_threshold:
                return False
        return True

    def _clean_question_text(self, text: str) -> str:
        """Очищает текст вопроса от лишних символов"""
        if not text:
            return ""
        text = re.sub(r'^\d+\.\s*', '', text.strip())
        text = re.sub(r'^[•\-]\s*', '', text)
        text = re.sub(r'^вопрос\s*\d*:*\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'["«»]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _generate_lesson_summary(self, lesson_context: str) -> str:
        """Генерирует краткое содержание урока"""
        try:
            prompt = f"""
            Создай КРАТКОЕ содержание этого урока для практических вопросов.
            Выдели только ключевые понятия и основные идеи.
            
            ТЕКСТ: {lesson_context[:1500]}
            
            Верни только краткое содержание (максимум 300 слов).
            """
            
            summary = self.llm.query(
                question=prompt,
                context="",
                subject=self.current_subject
            )
            return summary if summary and len(summary) > 50 else lesson_context[:500] + "..."
        except Exception as e:
            print(f"❌ Ошибка генерации краткого содержания: {e}")
            return lesson_context[:500] + "..."

    def _get_fallback_question_for_age(self, age: int, ensure_unique: bool = False) -> str:
        """🔥 НОВЫЙ МЕТОД: Возвращает fallback вопросы для конкретного возраста"""
        # Определяем возрастную группу
        if age <= 10:
            age_group = "young"
        elif age <= 14:
            age_group = "middle" 
        else:
            age_group = "old"
        
        # Возрастные вопросы для разных предметов
        age_questions = {
            "обществознание": {
                "young": [
                    "Что такое семья и зачем она нужна?",
                    "Как нужно вести себя в школе?",
                    "Что такое дружба и почему она важна?",
                    "Какие правила поведения ты знаешь?",
                    "Что такое доброта и как ее проявлять?"
                ],
                "middle": [
                    "Что такое общество и каковы его основные элементы?",
                    "Объясни понятие 'социальный институт' и приведи примеры.",
                    "В чем разница между формальными и неформальными социальными нормами?",
                    "Какие функции выполняет государство в современном обществе?",
                    "Что такое гражданское общество и как оно взаимодействует с государством?"
                ],
                "old": [
                    "Проанализируй основные социальные институты и их функции в обществе.",
                    "В чем заключается сущность правового государства?",
                    "Каковы основные принципы гражданского общества?",
                    "Объясни взаимосвязь экономики, политики и социальной сферы.",
                    "Какие глобальные проблемы современного общества ты можешь назвать?"
                ]
            },
            "математика": {
                "young": [
                    "Посчитай, сколько будет 5 + 3?",
                    "Что такое цифры и зачем они нужны?",
                    "Как сравнить числа 7 и 5?",
                    "Что такое геометрические фигуры?",
                    "Как решить простую задачу на сложение?"
                ],
                "middle": [
                    "Объясни основную концепцию, которую мы только что изучили.",
                    "Как применить изученный метод на практике?",
                    "В чем особенность этого математического подхода?",
                    "Какие существуют альтернативные способы решения этой задачи?",
                    "Почему этот математический принцип важен для понимания?"
                ],
                "old": [
                    "Докажи основную теорему изученного раздела.",
                    "Проанализируй применение этого математического метода в реальных задачах.",
                    "Каковы ограничения изученного алгоритма?",
                    "Сравни различные подходы к решению этой проблемы.",
                    "Какие практические применения имеет эта математическая концепция?"
                ]
            },
            "история": {
                "young": [
                    "Кто такие древние люди и как они жили?",
                    "Что такое Родина и почему ее нужно любить?",
                    "Какие праздники нашей страны ты знаешь?",
                    "Кто такие герои и почему их помнят?",
                    "Что такое музей и зачем туда ходят?"
                ],
                "middle": [
                    "Каковы были ключевые события изученного периода?",
                    "Как повлияли эти исторические события на современность?",
                    "В чем заключались основные причины исторических процессов, которые мы изучали?",
                    "Охарактеризуй ключевых исторических личностей этого периода.",
                    "Какие исторические закономерности можно проследить в изученном материале?"
                ],
                "old": [
                    "Проанализируй причинно-следственные связи исторических событий.",
                    "Сравни различные исторические периоды по ключевым параметрам.",
                    "Оцени влияние исторических процессов на современное общество.",
                    "Какие альтернативные исторические развития возможны были?",
                    "Как исторический контекст влияет на интерпретацию событий?"
                ]
            },
            "физика": {
                "young": [
                    "Что такое сила и как она действует?",
                    "Почему предметы падают на землю?",
                    "Что такое свет и тень?",
                    "Как работает магнит?",
                    "Почему летает воздушный шар?"
                ],
                "middle": [
                    "Как работает основной физический принцип, который мы рассмотрели?",
                    "Объясни физический смысл изученного явления.",
                    "Где в повседневной жизни мы встречаемся с этим физическим законом?",
                    "Какие практические применения имеет это физическое открытие?",
                    "В чем заключается научная важность изученного физического явления?"
                ],
                "old": [
                    "Выведи основное уравнение изученного физического закона.",
                    "Проанализируй границы применимости этой физической теории.",
                    "Сравни различные физические модели объяснения этого явления.",
                    "Какие экспериментальные подтверждения существуют для этой теории?",
                    "Как эта физическая концепция связана с другими разделами физики?"
                ]
            },
            "химия": {
                "young": [
                    "Что такое вещества и какие они бывают?",
                    "Почему вода важна для жизни?",
                    "Что такое воздух и из чего он состоит?",
                    "Как отличить твердое тело от жидкости?",
                    "Что такое раствор и как его получить?"
                ],
                "middle": [
                    "Опиши основные химические процессы из урока.",
                    "В чем особенность химических свойств изученных элементов?",
                    "Как протекает химическая реакция, которую мы изучали?",
                    "Какое практическое значение имеют эти химические процессы?",
                    "Объясни взаимосвязь между строением и свойствами химических веществ."
                ],
                "old": [
                    "Составь уравнение химической реакции и расставь коэффициенты.",
                    "Проанализируй механизм протекания этой химической реакции.",
                    "Сравни химические свойства изученных элементов и соединений.",
                    "Какие промышленные применения имеет этот химический процесс?",
                    "Объясни с позиций электронного строения свойства этого вещества."
                ]
            },
            "биология": {
                "young": [
                    "Что такое растения и животные?",
                    "Почему нужно мыть руки перед едой?",
                    "Как растут цветы?",
                    "Что такое семья животных?",
                    "Почему птицы улетают на юг?"
                ],
                "middle": [
                    "Каковы основные биологические процессы, которые мы изучили?",
                    "Опиши строение и функции биологических структур из урока.",
                    "Как взаимодействуют различные биологические системы?",
                    "В чем биологическое значение изученных процессов?",
                    "Какие адаптации организмов мы рассмотрели и в чем их смысл?"
                ],
                "old": [
                    "Проанализируй эволюционные предпосылки изученного биологического явления.",
                    "Сравни строение и функции различных биологических систем.",
                    "Каковы молекулярные механизмы этого биологического процесса?",
                    "Как это биологическое явление связано с экологией?",
                    "Какие практические применения в медицине имеет это биологическое знание?"
                ]
            },
            "литература": {
                "young": [
                    "О чем эта сказка/рассказ?",
                    "Кто главный герой и какой он?",
                    "Чему учит эта история?",
                    "Какие поступки героя тебе понравились?",
                    "Как бы ты поступил на месте героя?"
                ],
                "middle": [
                    "В чем основная идея или тема произведения, которое мы обсуждали?",
                    "Охарактеризуй главных героев изученного произведения.",
                    "Как автор раскрывает основные темы в произведении?",
                    "В чем художественное своеобразие этого литературного произведения?",
                    "Какие нравственные проблемы поднимает автор в произведении?"
                ],
                "old": [
                    "Проанализируйте художественные средства, использованные автором.",
                    "Сравните различные интерпретации этого литературного произведения.",
                    "Как исторический контекст влияет на понимание произведения?",
                    "Каковы философские аспекты поднятых в произведении проблем?",
                    "Как творчество этого автора связано с литературным направлением его эпохи?"
                ]
            },
            "русский язык": {
                "young": [
                    "Что такое буквы и звуки?",
                    "Как правильно писать свое имя?",
                    "Что такое предложение?",
                    "Как отличить гласные от согласных?",
                    "Зачем нужны знаки препинания?"
                ],
                "middle": [
                    "Объясни основное грамматическое правило, которое мы изучили.",
                    "В чем особенности применения этого правила на практике?",
                    "Какие исключения существуют из изученного правила?",
                    "Как правильно использовать изученные языковые конструкции?",
                    "Почему это грамматическое правило важно для правильной речи?"
                ],
                "old": [
                    "Проанализируйте стилистические особенности этого текста.",
                    "Сравните употребление этой грамматической конструкции в разных стилях речи.",
                    "Каковы исторические предпосылки этого языкового явления?",
                    "Как это правило связано с системой языка в целом?",
                    "Какие трудности возникают при практическом применении этого правила?"
                ]
            }
        }
        
        # Базовый набор вопросов для любого предмета
        general_questions = {
            "young": [
                "Объясни самое главное, что мы сегодня узнали.",
                "Что тебе больше всего понравилось в уроке?",
                "Как можно использовать эти знания в жизни?",
                "Что было самым интересным?",
                "О чем бы ты хотел узнать больше?"
            ],
            "middle": [
                "Объясни основную идею изученного материала.",
                "В чем заключается главная мысль этого урока?",
                "Какие ключевые понятия мы сегодня изучили?",
                "Как можно применить эти знания на практике?",
                "Почему эта тема важна для понимания предмета?"
            ],
            "old": [
                "Проанализируйте основные концепции изученного материала.",
                "Каковы практические применения этих знаний?",
                "Сравните изученные понятия с ранее известными.",
                "Какие перспективы развития этой темы вы видите?",
                "Как эти знания связаны с другими областями науки?"
            ]
        }
        
        # Получаем вопросы для текущего предмета и возраста
        subject_questions = age_questions.get(self.current_subject, {})
        questions = subject_questions.get(age_group, general_questions.get(age_group, ["Расскажи об основном понятии из урока?"]))
        
        if ensure_unique and self.generated_questions:
            # Ищем вопрос, которого еще не было
            existing_questions = [q["question"] for q in self.generated_questions]
            for question in questions:
                if question not in existing_questions:
                    return question
            
            # Если все вопросы уже использованы, берем из общего списка
            for general_q in general_questions.get(age_group, []):
                if general_q not in existing_questions:
                    return general_q
        
        # Возвращаем случайный вопрос
        import random
        return random.choice(questions)

    def _get_fallback_question(self, ensure_unique: bool = False) -> str:
        """Старый метод для обратной совместимости"""
        age = self.student_data.get('age', '12')
        return self._get_fallback_question_for_age(int(age), ensure_unique)

    def has_more_questions(self) -> bool:
        """Проверяет, можно ли генерировать еще вопросы"""
        return len(self.generated_questions) < self.max_questions

    def get_generated_questions_count(self) -> int:
        """Возвращает количество сгенерированных вопросов"""
        return len(self.generated_questions)

    def reset(self):
        """Сброс состояния менеджера практики"""
        self.stop_async_generation()
        self.current_lesson_context = ""
        self.current_lesson_summary = ""
        self.current_subject = ""
        self.generated_questions = []
        self.current_question_index = 0
        while not self.question_queue.empty():
            try:
                self.question_queue.get_nowait()
            except queue.Empty:
                break
        print("🔄 Менеджер практики сброшен")

    def get_current_question(self) -> Optional[str]:
        """Возвращает текущий вопрос"""
        if self.generated_questions:
            return self.generated_questions[-1]["question"]
        return None

    def get_practice_stats(self) -> Dict:
        """🔥 ОБНОВЛЕННЫЙ МЕТОД: Возвращает статистику по практике с учетом возраста"""
        question_types = {}
        age_adapted_count = 0
        
        for q in self.generated_questions:
            q_type = q.get("type", "unknown")
            question_types[q_type] = question_types.get(q_type, 0) + 1
            
            if q.get("age_adapted", False):
                age_adapted_count += 1
        
        age = self.student_data.get('age', 'неизвестен')
        level = self.student_data.get('level', 'неизвестен')
        
        return {
            "total_questions": len(self.generated_questions),
            "max_questions": self.max_questions,
            "questions_in_queue": self.question_queue.qsize(),
            "generation_active": self.generation_active,
            "current_subject": self.current_subject,
            "has_more_questions": self.has_more_questions(),
            "question_types": question_types,
            "lesson_summary_length": len(self.current_lesson_summary),
            # 🔥 НОВОЕ: Статистика по адаптации
            "student_data": {
                "age": age,
                "level": level
            },
            "age_adapted_questions": age_adapted_count,
            "adaptation_percentage": round((age_adapted_count / len(self.generated_questions)) * 100, 1) if self.generated_questions else 0
        }


# Создаем глобальный экземпляр для тестирования
if __name__ == "__main__":
    # Тестирование базовой функциональности
    print("🧪 Тестирование PracticeManager...")
    
    # Создаем mock LLM для тестирования
    class MockLLM:
        def query(self, question, context, subject):
            return "Тестовый вопрос: Объясни основную концепцию изученного материала?"
    
    # Тестируем менеджер практики
    pm = PracticeManager(MockLLM())
    
    # Тест с данными ученика
    pm.student_data = {
        'name': 'Тестовый ученик',
        'age': '12',
        'level': '5'
    }
    
    # Тест инициализации
    pm.initialize_practice_generation("Текст урока для тестирования", "математика")
    
    # Тест генерации вопроса
    question = pm.generate_single_question()
    print(f"📝 Сгенерированный вопрос: {question}")
    
    # Тест статистики
    stats = pm.get_practice_stats()
    print(f"📊 Статистика практики: {stats}")
    
    print("✅ Тестирование PracticeManager завершено!")