import json
from typing import Dict, List, Optional
import time
from datetime import datetime
from pathlib import Path
import re

from llm import LLMIntegration
from config import get_dialogue_settings, load_config


class LLMDialogueManager:
    def __init__(self):
        self.llm = LLMIntegration()
        self.dialogue_settings = get_dialogue_settings()
        self.conversation_history = []
        self.max_history = self.dialogue_settings.get("context_window", 5)
        self.subject_selection_prompt = self.dialogue_settings.get(
            "subject_selection_prompt", 
            "Ты - дружелюбный учитель. Помоги ученику выбрать предмет для изучения. "
            "Будь кратким и понятным, максимум 2-3 предложения. "
            "Цель - подвести ученика к выбору предмета для урока."
        )
        self.last_interaction_time = time.time()
        self.inactivity_timeout = 300  # 5 минут бездействия
        
    def _add_to_conversation_history(self, text: str, is_user: bool = True):
        """Добавляет реплику в историю диалога"""
        self.conversation_history.append({
            "text": text,
            "is_user": is_user,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat()
        })
        
        # Ограничиваем размер истории
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def _get_conversation_context(self) -> str:
        """Возвращает контекст диалога для LLM"""
        if not self.conversation_history:
            return ""
            
        context_lines = []
        for msg in self.conversation_history:
            speaker = "Ученик" if msg["is_user"] else "Учитель"
            context_lines.append(f"{speaker}: {msg['text']}")
        
        return "\n".join(context_lines)
    
    def _limit_response_length(self, response: str, max_sentences: int = 2) -> str:
        """Ограничивает длину ответа количеством предложений"""
        if not response:
            return response
            
        sentences = re.split(r'(?<=[.!?])\s+', response)
        if len(sentences) > max_sentences:
            return ' '.join(sentences[:max_sentences])
        return response
    
    def _detect_subject_intent(self, text: str) -> Optional[str]:
        """Определяет намерение выбора предмета в тексте"""
        text_lower = text.lower()
        
        subject_keywords = {
            "математика": ["математик", "матема", "алгебр", "геометри", "цифр", "числ", "уравнен"],
            "история": ["истори", "истор", "прошлое", "древн", "войн", "сражен", "цар", "император"],
            "физика": ["физик", "физ", "механи", "электри", "магнит", "теплот", "оптик", "атом"],
            "химия": ["хими", "хим", "веществ", "реакц", "элемент", "периодическ", "молекул", "атом"],
            "обществознание": ["обществ", "общест", "социум", "государств", "право", "экономик", "политик", "культур"],
            "биология": ["биолог", "био", "животн", "растен", "клетк", "организм", "ген", "эволюц"],
            "литература": ["литератур", "лит", "книг", "писатель", "поэт", "стих", "роман", "рассказ"],
            "русский язык": ["русск", "язык", "грамматик", "орфограф", "пунктуац", "слов", "предложен"]
        }
        
        for subject, keywords in subject_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return subject
        
        return None
    
    def _generate_subject_suggestion(self, detected_subject: str) -> str:
        """Генерирует предложение по выбранному предмету"""
        suggestions = {
            "математика": [
                "Отлично! Математика - это увлекательный мир чисел и закономерностей.",
                "Прекрасный выбор! Математика развивает логическое мышление.",
                "Математика! Отличный предмет для развития аналитических способностей."
            ],
            "история": [
                "История - это увлекательное путешествие в прошлое!",
                "Отлично! История помогает понять настоящее через прошлое.",
                "История! Замечательный выбор для изучения развития человечества."
            ],
            "физика": [
                "Физика - это наука о природе и её законах!",
                "Отлично! Физика объясняет, как устроен мир вокруг нас.",
                "Физика! Прекрасный выбор для любознательных умов."
            ],
            "химия": [
                "Химия - это магия превращения веществ!",
                "Отлично! Химия помогает понять состав всего вокруг.",
                "Химия! Увлекательный мир молекул и реакций."
            ],
            "обществознание": [
                "Обществознание - ключ к пониманию общества!",
                "Отлично! Обществознание помогает разобраться в социальных процессах.",
                "Обществознание! Важный предмет для современного человека."
            ],
            "биология": [
                "Биология - это наука о жизни во всём её разнообразии!",
                "Отлично! Биология раскрывает тайны живых организмов.",
                "Биология! Увлекательное изучение природы и человека."
            ],
            "литература": [
                "Литература - это искусство слова и мир воображения!",
                "Отлично! Литература развивает эмоциональный интеллект.",
                "Литература! Прекрасный выбор для ценителей прекрасного."
            ],
            "русский язык": [
                "Русский язык - это фундамент нашей культуры!",
                "Отлично! Грамотность открывает многие двери.",
                "Русский язык! Важная основа для эффективного общения."
            ]
        }
        
        import random
        if detected_subject in suggestions:
            return random.choice(suggestions[detected_subject])
        
        return "Интересный выбор! Давайте начнем урок."
    
    def process_input(self, text: str) -> Optional[str]:
        """Обработка входящего текста и генерация ответа через LLM"""
        if not text.strip():
            return "Не расслышал, повторите пожалуйста."
        
        # Обновляем время последнего взаимодействия
        self.last_interaction_time = time.time()
        
        # Добавляем пользовательский ввод в историю
        self._add_to_conversation_history(text, is_user=True)
        
        # Пытаемся определить намерение выбора предмета
        detected_subject = self._detect_subject_intent(text)
        
        # Если обнаружен явный выбор предмета, возвращаем None для начала урока
        if detected_subject and any(word in text.lower() for word in 
                                  ["хочу", "выбираю", "давай", "начнем", "урок", "занятие"]):
            return None
        
        # Получаем контекст диалога
        context = self._get_conversation_context()
        
        # Формируем промпт для LLM
        system_prompt = self.subject_selection_prompt
        
        # Если обнаружен предмет, добавляем это в промпт
        if detected_subject:
            system_prompt += f"\nУченик проявил интерес к предмету: {detected_subject}. " \
                           f"Поддержи этот интерес и мягко подведи к началу урока."
        
        # Запрос к LLM
        try:
            llm_response = self.llm._query_llm_api(
                prompt=text,
                context=context,
                subject="выбор предмета",
                system_prompt=system_prompt,
                max_tokens=100  # Ограничиваем длину ответа для быстрого диалога
            )
            
            if llm_response:
                # Ограничиваем длину ответа
                limited_response = self._limit_response_length(
                    llm_response, 
                    self.dialogue_settings.get("max_response_length", 2)
                )
                
                # Добавляем ответ в историю
                self._add_to_conversation_history(limited_response, is_user=False)
                
                # Если обнаружен предмет, добавляем предложение
                if detected_subject:
                    subject_suggestion = self._generate_subject_suggestion(detected_subject)
                    return f"{subject_suggestion} {limited_response}"
                
                return limited_response
                
        except Exception as e:
            print(f"Ошибка запроса к LLM для диалога: {e}")
        
        # Fallback ответ если LLM недоступен
        fallback_responses = [
            "Интересно! Давайте выберем предмет для урока. Что вас интересует?",
            "Понятно. Какой предмет хотели бы изучить?",
            "Хорошо! Давайте определимся с темой для нашего урока.",
            "Отлично! Какой предмет вас привлекает больше всего?",
            "Прекрасно! Выбор за вами - какой предмет изучаем?"
        ]
        
        import random
        return random.choice(fallback_responses)
    
    def get_conversation_history(self) -> List[Dict]:
        """Возвращает историю диалога"""
        return self.conversation_history.copy()
    
    def clear_history(self):
        """Очищает историю диалога"""
        self.conversation_history = []
    
    def is_inactive(self) -> bool:
        """Проверяет, был ли диалог неактивен слишком долго"""
        return time.time() - self.last_interaction_time > self.inactivity_timeout
    
    def reset_inactivity_timer(self):
        """Сбрасывает таймер неактивности"""
        self.last_interaction_time = time.time()
    
    def get_detected_subject(self, text: str) -> Optional[str]:
        """Возвращает обнаруженный предмет в тексте"""
        return self._detect_subject_intent(text)
