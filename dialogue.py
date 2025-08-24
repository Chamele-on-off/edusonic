import random
from typing import Dict, Optional, List
from difflib import SequenceMatcher
import json
from pathlib import Path

class SimpleDialogueManager:
    """
    Упрощенный менеджер диалогов для AI-учителя
    Только базовые приветствия и запуск уроков
    """
    
    def __init__(self):
        self.lessons_dir = Path("lessons")
        self._load_lessons()
        
        # Локальные шаблоны для быстрого ответа
        self.local_patterns = {
            "привет": ["Привет! Скажи 'начать урок' чтобы выбрать предмет.", "Здравствуй! Готов начать урок? Скажи 'начать урок'."],
            "как дела": ["Отлично! Готов помочь с уроками.", "Прекрасно! Давай начнем занятие."],
            "спасибо": ["Пожалуйста! Всегда рад помочь!", "Не за что! Ты отлично справляешься!"],
            "начать урок": ["Отлично! Давай выберем предмет для урока.", "Супер! Какой предмет хочешь изучать?"],
            "математика": ["Математика - отличный выбор! Начинаем урок.", "Запускаю урок математики."],
            "обществознание": ["Обществознание - интересно! Начинаем урок.", "Запускаю урок обществознания."],
            "русский": ["Русский язык - важно знать! Начинаем.", "Запускаю урок русского языка."],
            "физика": ["Физика - увлекательно! Начинаем урок.", "Запускаю урок физики."],
            "химия": ["Химия - это интересно! Начинаем.", "Запускаю урок химии."],
            "биология": ["Биология - изучаем природу! Начинаем.", "Запускаю урок биологии."],
            "история": ["История - познаем прошлое! Начинаем.", "Запускаю урок истории."],
            "английский": ["Английский - полезно знать! Начинаем.", "Запускаю урок английского."],
            "информатика": ["Информатика - современно! Начинаем.", "Запускаю урок информатики."],
            "готов": ["Отлично! Начинаем урок!", "Супер! Приступаем к занятию!"],
            "да": ["Хорошо, продолжаем!", "Отлично! Двигаемся дальше!"]
        }

    def _load_lessons(self):
        """Загружает список доступных уроков"""
        self.lessons = {}
        try:
            if not self.lessons_dir.exists():
                self.lessons_dir.mkdir(parents=True)
                return
                
            for lesson_file in self.lessons_dir.glob("*.json"):
                try:
                    with open(lesson_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        subject = data.get('subject', 'other').lower()
                        if subject not in self.lessons:
                            self.lessons[subject] = []
                        self.lessons[subject].append({
                            'id': data.get('id', lesson_file.stem),
                            'title': data.get('title', 'Без названия'),
                            'description': data.get('description', ''),
                            'duration': data.get('duration', 1800)
                        })
                except Exception as e:
                    print(f"Ошибка загрузки урока {lesson_file}: {e}")
        except Exception as e:
            print(f"Ошибка доступа к папке уроков: {e}")

    def _similarity(self, a: str, b: str) -> float:
        """Вычисление схожести строк"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def process_input(self, text: str) -> str:
        """
        Обработка входящего текста
        Возвращает ответ или None если обработка не требуется
        """
        if not text.strip():
            return "Повтори, пожалуйста, я не расслышал."
            
        text_lower = text.lower().strip()
        
        # 1. Быстрая проверка локальных шаблонов
        for pattern, responses in self.local_patterns.items():
            if pattern in text_lower:
                return random.choice(responses)
        
        # 2. Поиск предмета для начала урока
        subjects = list(self.lessons.keys())
        for subject in subjects:
            if subject in text_lower:
                lessons = self.lessons.get(subject, [])
                if lessons:
                    # Если есть только один урок по предмету
                    if len(lessons) == 1:
                        return f"Отлично! {subject.capitalize()}! Начинаем урок '{lessons[0]['title']}'. Скажи 'готов' чтобы начать!"
                    else:
                        # Если несколько уроков
                        lesson_list = "\n".join([f"{i+1}. {lesson['title']}" 
                                               for i, lesson in enumerate(lessons)])
                        return f"Отлично! {subject.capitalize()}! Выбери урок:\n{lesson_list}\nСкажи номер урока."
                else:
                    return f"К сожалению, для предмета '{subject}' нет уроков. Выбери другой предмет."
        
        # 3. Обработка выбора урока по номеру
        if text_lower.isdigit():
            lesson_num = int(text_lower)
            # Здесь можно реализовать логику выбора урока по номеру
            return f"Выбран урок номер {lesson_num}. Скажи 'готов' чтобы начать!"
        
        # 4. Fallback ответ
        fallbacks = [
            "Не совсем понял. Скажи 'начать урок' чтобы выбрать предмет.",
            "Давай начнем урок! Скажи 'начать урок'.",
            "Какой предмет тебя интересует? Скажи например 'математика' или 'обществознание'.",
            "Готов помочь с уроками! Скажи название предмета чтобы начать."
        ]
        
        return random.choice(fallbacks)

    def get_lessons_for_subject(self, subject: str) -> List[dict]:
        """Возвращает уроки для указанного предмета"""
        return self.lessons.get(subject.lower(), [])

    def get_all_lessons(self) -> List[dict]:
        """Возвращает все доступные уроки"""
        all_lessons = []
        for subject_lessons in self.lessons.values():
            all_lessons.extend(subject_lessons)
        return all_lessons

    def get_available_subjects(self) -> List[str]:
        """Возвращает список доступных предметов"""
        return list(self.lessons.keys())

# Создаем глобальный экземпляр для простоты
dialogue_manager = SimpleDialogueManager()
