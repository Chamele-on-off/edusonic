import random
from typing import Dict, Optional, List
from difflib import SequenceMatcher

class DialogueManager:
    def __init__(self):
        self.dialogue_states = {
            "greeting": self._handle_greeting,
            "lesson_selection": self._handle_lesson_selection,
            "lesson": self._handle_lesson,
            "practice": self._handle_practice,
            "qa": self._handle_qa,
            "farewell": self._handle_farewell
        }
        self.current_state = "greeting"
        self.current_subject = None
        self.current_topic = None
        
        # База знаний для диалога
        self.qa_knowledge = {
            # Приветствия
            "привет": ["Привет! Как твои дела?", "Здравствуй! Готов к уроку?"],
            "здравствуй": ["Приветствую! Как настроение?", "Здравствуй! Как твои успехи в учебе?"],
            
            # Состояние
            "как дела": ["Отлично! А у тебя как?", "Прекрасно, спасибо! Как твои успехи?"],
            "настроение": ["Я всегда в хорошем настроении, когда учу! А ты как?", "Отличное! Готов учить и учиться."],
            
            # Урок
            "начать урок": ["Отлично! Давай начнём.", "Хорошо, приступим к занятию."],
            "закончить урок": ["Хорошо, давай подведём итоги.", "Понял, завершаем занятие."],
            
            # Темы
            "обществознание": ["Отличный выбор! Какая тема тебя интересует: 1) Гражданское общество, 2) Права человека, 3) Экономика?", 
                             "По обществознанию у нас есть несколько тем. Выбери: 1) Социальные нормы, 2) Политика, 3) Культура"],
            "история": ["По истории можем обсудить: 1) Древний мир, 2) Средние века, 3) Новое время", 
                       "История - это интересно! Какая эпоха тебя интересует?"],
            
            # Общие
            "спасибо": ["Пожалуйста! Есть ещё вопросы?", "Всегда рад помочь! Что-то ещё?"],
            "не знаю": ["Ничего страшного, давай разберёмся вместе.", "Это нормально не знать, главное - научиться!"],
            "повтори": ["Конечно, повторяю...", "Давай ещё раз."]
        }

        # Fallback-сценарии
        self.fallback_responses = [
            "Я не совсем понял. Можешь уточнить?",
            "Извини, я не уверен, что правильно тебя понял. Переформулируй, пожалуйста.",
            "Можешь сказать это другими словами?",
            "Давай попробуем по-другому. О чём ты хотел спросить?",
            "Я пока не могу ответить на этот вопрос. Может, спросишь что-то другое?"
        ]

    def _similarity(self, a: str, b: str) -> float:
        """Вычисление схожести строк"""
        return SequenceMatcher(None, a, b).ratio()

    def _find_best_match(self, text: str, threshold: float = 0.6) -> Optional[str]:
        """Поиск наиболее подходящего вопроса в базе знаний"""
        text_lower = text.lower()
        best_match = None
        highest_similarity = 0.0
        
        for pattern in self.qa_knowledge.keys():
            similarity = self._similarity(text_lower, pattern)
            if similarity > highest_similarity and similarity >= threshold:
                highest_similarity = similarity
                best_match = pattern
                
        return best_match if highest_similarity >= threshold else None

    def process_input(self, text: str) -> str:
        """Обработка входящего текста и генерация ответа"""
        text_lower = text.lower().strip()
        
        # 1. Попробовать найти точный ответ по текущему состоянию
        handler = self.dialogue_states.get(self.current_state)
        if handler:
            response = handler(text_lower)
            if response:
                return response
        
        # 2. Попробовать найти похожий вопрос в базе знаний
        best_match = self._find_best_match(text_lower)
        if best_match:
            return random.choice(self.qa_knowledge[best_match])
        
        # 3. Fallback-ответ
        return random.choice(self.fallback_responses)

    def _handle_greeting(self, text: str) -> Optional[str]:
        if any(word in text for word in ["привет", "здравствуй", "добрый", "начать", "старт"]):
            self.current_state = "lesson_selection"
            return random.choice([
                "Привет! Давай начнём наш урок. Какой предмет будем изучать?",
                "Здравствуй! Сегодня можем позаниматься обществознанием или историей. Что выберешь?"
            ])
        return None

    def _handle_lesson_selection(self, text: str) -> Optional[str]:
        if "обществ" in text:
            self.current_subject = "обществознание"
            self.current_state = "lesson"
            return random.choice(self.qa_knowledge["обществознание"])
        elif "истори" in text:
            self.current_subject = "история"
            self.current_state = "lesson"
            return random.choice(self.qa_knowledge["история"])
        elif any(word in text for word in ["1", "2", "3", "перв", "втор", "трет"]):
            self.current_state = "lesson"
            return "Хороший выбор! Давай начнём урок. Сначала я расскажу теорию, потом обсудим примеры."
        return None

    def _handle_lesson(self, text: str) -> Optional[str]:
        if any(word in text for word in ["понял", "ясно", "дальше", "продолжи"]):
            return "Отлично! Давай перейдём к практике. Попробуй ответить на вопрос..."
        elif any(word in text for word in ["не пон", "не яс", "повтор"]):
            return "Хорошо, давай разберём ещё раз. Основная идея заключается в том, что..."
        return None

    def _handle_practice(self, text: str) -> Optional[str]:
        if any(word in text for word in ["ответ", "думаю", "считаю"]):
            return "Интересный ответ! Давай проверим... Да, всё верно!" if random.random() > 0.3 else "Почти правильно, но давай уточним..."
        return None

    def _handle_qa(self, text: str) -> Optional[str]:
        if any(word in text for word in ["вопрос", "не знаю", "объясни"]):
            return "Хороший вопрос! Давай разберёмся..."
        return None

    def _handle_farewell(self, text: str) -> Optional[str]:
        if any(word in text for word in ["конец", "закончи", "хватит", "до свидан"]):
            return "Отлично поработали! До следующего урока!"
        return None

    def reset(self):
        """Сброс состояния диалога"""
        self.current_state = "greeting"
        self.current_subject = None
        self.current_topic = None
