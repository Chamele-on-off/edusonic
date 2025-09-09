# demodialogue.py
import json
from pathlib import Path
from typing import Dict, Optional, List
import random
import time
from dialogue import DialogueManager
from llm import LLMIntegration

class DemoDialogueManager(DialogueManager):
    """
    Менеджер диалога для демо-режима.
    Общается через LLM, предлагает демо-уроки и генерирует контент на лету.
    """
    def __init__(self, socketio):
        # Не вызываем super().__init__(), чтобы не загружать стандартные уроки и базы знаний
        self.socketio = socketio
        self.current_state = "greeting"
        self.conversation_context = []  # Для сохранения контекста диалога
        self.llm = LLMIntegration()
        self.available_demo_subjects = self._discover_demo_subjects()
        self.current_topic = None
        self.generated_lesson_content = []
        self.current_paragraph = 0
        self.llm_query_mode = "llm_first"  # В демо-режиме всегда используем llm_first

    def _discover_demo_subjects(self) -> List[str]:
        """Обнаруживает все демо-уроки в папке lessons/demo"""
        demo_dir = Path("lessons/demo")
        subjects = set()
        try:
            if not demo_dir.exists():
                demo_dir.mkdir(parents=True)
                # Создаем пару демо-файлов, если папка пуста
                self._create_sample_demos(demo_dir)
            
            for lesson_file in demo_dir.glob("*.txt"):
                # Извлекаем тему из имени файла (например, "history_ww2" -> "history")
                subject = lesson_file.stem.split('_')[0]
                subjects.add(subject)
        except Exception as e:
            print(f"Ошибка при сканировании демо-уроков: {e}")
            subjects = {"history", "science", "art"}  # Fallback
        return list(subjects)

    def _create_sample_demos(self, demo_dir: Path):
        """Создает несколько примеров демо-уроков"""
        samples = {
            "history_ancients": "Древний мир: от первых цивилизаций до падения Рима.\n\nМесопотамия, Египет, Греция и Рим - основы современной цивилизации.",
            "science_basics": "Основы физики: движение, энергия, материя.\n\nВсе во вселенной подчиняется фундаментальным законам физики.",
            "art_renaissance": "Эпоха Возрождения: возрождение искусства и науки.\n\nЛеонардо да Винчи, Микеланджело и Рафаэль изменили мир искусства."
        }
        
        for filename, content in samples.items():
            with open(demo_dir / f"{filename}.txt", 'w', encoding='utf-8') as f:
                f.write(content)

    def process_input(self, text: str) -> Optional[str]:
        """Обработка ввода в демо-режиме с сохранением контекста"""
        text_lower = text.lower().strip()
        
        # Добавляем реплику пользователя в контекст
        self.conversation_context.append({"role": "user", "content": text})
        # Ограничиваем размер контекста для избежания переполнения
        if len(self.conversation_context) > 10:
            self.conversation_context = self.conversation_context[-10:]
        
        # Обработка состояний
        if self.current_state == "greeting":
            response = self._handle_greeting(text_lower)
        elif self.current_state == "subject_selection":
            response = self._handle_subject_selection(text_lower)
        elif self.current_state == "lesson_reading":
            response = self._handle_lesson_reading(text_lower)
        else:
            response = "Давайте продолжим наш разговор. Что вас интересует?"
        
        # Добавляем ответ системы в контекст
        if response:
            self.conversation_context.append({"role": "assistant", "content": response})
        
        return response

    def _handle_greeting(self, text: str) -> str:
        """Обработка приветствия в демо-режиме"""
        if any(word in text for word in ["привет", "здравств", "начать", "старт"]):
            self.current_state = "subject_selection"
            
            subjects_list = ", ".join([s.capitalize() for s in self.available_demo_subjects])
            return (f"Привет! Я ваш демо-учитель. Давайте исследуем мир знаний вместе! "
                   f"У меня есть демо-уроки по: {subjects_list}. "
                   f"Также вы можете предложить любую тему для изучения. Что вас интересует?")
        
        return "Приветствую! Я здесь, чтобы помочь вам узнать что-то новое. С чего начнем?"

    def _handle_subject_selection(self, text: str) -> Optional[str]:
        """Обработка выбора темы в демо-режиме"""
        # Проверяем, является ли ввод одним из доступных предметов
        for subject in self.available_demo_subjects:
            if subject in text.lower():
                self.current_topic = subject
                return self._start_demo_lesson(subject)
        
        # Если это не известный предмет, трактуем как произвольную тему
        self.current_topic = text
        return self._generate_lesson_for_topic(text)

    def _start_demo_lesson(self, subject: str) -> Optional[str]:
        """Начинает демо-урок по выбранному предмету"""
        demo_dir = Path("lessons/demo")
        lesson_files = list(demo_dir.glob(f"{subject}_*.txt"))
        
        if lesson_files:
            # Выбираем случайный урок по теме
            lesson_file = random.choice(lesson_files)
            with open(lesson_file, 'r', encoding='utf-8') as f:
                self.generated_lesson_content = [p.strip() for p in f.read().split('\n\n') if p.strip()]
            
            self.current_state = "lesson_reading"
            self.current_paragraph = 0
            
            title = lesson_file.stem.replace('_', ' ').title()
            return f"Отлично! Начинаем демо-урок: '{title}'. {self._get_next_paragraph()}"
        
        # Если файл не найден, генерируем урок
        return self._generate_lesson_for_topic(subject)

    def _generate_lesson_for_topic(self, topic: str) -> Optional[str]:
        """Генерирует урок по произвольной теме через LLM"""
        prompt = f"""
        Сгенерируй короткий учебный материал на тему: {topic}.
        Формат: несколько абзацев, разделенных пустой строкой.
        Первый абзац - введение, последующие - раскрытие темы.
        Будь информативным но лаконичным. Пиши на русском.
        """
        
        # Используем LLM для генерации контента
        generated_content = self.llm.query(prompt, "", "general")
        
        if generated_content:
            # Сохраняем сгенерированный контент
            self.generated_lesson_content = [p.strip() for p in generated_content.split('\n\n') if p.strip()]
            self.current_state = "lesson_reading"
            self.current_paragraph = 0
            
            # Сохраняем сгенерированный урок для будущего использования
            self._save_generated_lesson(topic, generated_content)
            
            return f"Отлично! Я подготовил материал по теме '{topic}'. {self._get_next_paragraph()}"
        
        return "Извините, не удалось создать материал по этой теме. Попробуйте другую тему."

    def _save_generated_lesson(self, topic: str, content: str):
        """Сохраняет сгенерированный урок для будущего использования"""
        try:
            # Создаем безопасное имя файла
            safe_topic = "".join(c if c.isalnum() else "_" for c in topic)
            filename = f"generated_{safe_topic}_{int(time.time())}.txt"
            demo_dir = Path("lessons/demo")
            
            with open(demo_dir / filename, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"Ошибка сохранения сгенерированного урока: {e}")

    def _get_next_paragraph(self) -> Optional[str]:
        """Возвращает следующий абзац урока"""
        if self.current_paragraph < len(self.generated_lesson_content):
            paragraph = self.generated_lesson_content[self.current_paragraph]
            self.current_paragraph += 1
            return paragraph
        return None

    def _handle_lesson_reading(self, text: str) -> Optional[str]:
        """Обработка ввода во время урока"""
        # Если пользователь хочет остановиться
        if any(word in text for word in ["стоп", "останови", "хватит", "закончи"]):
            self.current_state = "greeting"
            self.conversation_context = []
            return "Урок завершен. Хотите изучить что-то еще?"
        
        # Обработка вопросов во время урока с использованием контекста
        # Формируем промпт с историей диалога и текущим материалом
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversation_context[-4:]])
        current_content = self.generated_lesson_content[self.current_paragraph-1] if self.current_paragraph > 0 else ""
        
        prompt = f"""
        Контекст урока: {current_content}
        История диалога: {context}
        Вопрос ученика: {text}
        
        Ответь как учитель, основываясь на контексте урока. Будь краток и точен.
        """
        
        # Используем LLM для ответа на вопрос
        response = self.llm.query(prompt, "", self.current_topic or "general")
        return response

    def is_lesson_started(self) -> bool:
        return self.current_state == "lesson_reading"

    def handle_question_during_lesson(self, question: str) -> str:
        """Обработка вопросов во время урока - всегда через LLM с контекстом"""
        if not question.strip():
            return "Повторите вопрос пожалуйста, я не расслышал."
            
        # Формируем промпт с историей диалога
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversation_context[-6:]])
        current_content = self.generated_lesson_content[self.current_paragraph-1] if self.current_paragraph > 0 else ""
        
        prompt = f"""
        Контекст урока: {current_content}
        История диалога: {context}
        Вопрос ученика: {question}
        
        Ответь как учитель, основываясь на контексте урока. Будь краток и точен.
        """
        
        # Используем LLM для ответа на вопрос
        response = self.llm.query(prompt, "", self.current_topic or "general")
        return response or "Интересный вопрос! Давайте обсудим его подробнее."

    def set_llm_mode(self, mode: str):
        """В демо-режиме всегда используем llm_first"""
        self.llm_query_mode = "llm_first"
        print(f"В демо-режиме установлен режим LLM: llm_first (игнорируется запрос на {mode})")
