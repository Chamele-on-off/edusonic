import json
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime
from pathlib import Path
import re
import random

from llm import LLMIntegration
from config import get_dialogue_settings, load_config
from knowledge.knowledge_base import KnowledgeBase


class LLMDialogueManager:
    def __init__(self):
        self.llm = LLMIntegration()
        self.dialogue_settings = get_dialogue_settings()
        self.conversation_history = []
        self.max_history = self.dialogue_settings.get("context_window", 10)
        self.general_knowledge_base = KnowledgeBase("общее")
        self.last_interaction_time = time.time()
        self.inactivity_timeout = 300
        self.subject_suggested = False
        self.force_llm = True  # Принудительно использовать LLM для всех запросов
        
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
            return "Новый диалог. Ученик только что присоединился."
            
        context_lines = ["Предыдущий диалог:"]
        for msg in self.conversation_history[-6:]:  # Берем последние 6 сообщений
            speaker = "Ученик" if msg["is_user"] else "Учитель"
            context_lines.append(f"{speaker}: {msg['text']}")
        
        return "\n".join(context_lines)
    
    def _limit_response_length(self, response: str, max_sentences: int = 3) -> str:
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
            "математика": ["математик", "матема", "алгебр", "геометри", "цифр", "числ", "уравнен", "счет"],
            "история": ["истори", "истор", "прошлое", "древн", "войн", "сражен", "цар", "император", "историческ"],
            "физика": ["физик", "физ", "механи", "электри", "магнит", "теплот", "оптик", "атом", "физическ"],
            "химия": ["хими", "хим", "веществ", "реакц", "элемент", "периодическ", "молекул", "атом", "химическ"],
            "обществознание": ["обществ", "общест", "социум", "государств", "право", "экономик", "политик", "культур", "общество"],
            "биология": ["биолог", "био", "животн", "растен", "клетк", "организм", "ген", "эволюц", "биологическ"],
            "литература": ["литератур", "лит", "книг", "писатель", "поэт", "стих", "роман", "рассказ", "литературн"],
            "русский язык": ["русск", "язык", "грамматик", "орфограф", "пунктуац", "слов", "предложен", "русский"]
        }
        
        for subject, keywords in subject_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return subject
        
        return None
    
    def _is_greeting(self, text: str) -> bool:
        """Проверяет, является ли текст приветствием"""
        greetings = ["привет", "здравствуй", "здравствуйте", "добрый", "хай", "hello", "hi", 
                    "начать", "старт", "готов", "поехали", "давай", "началом", "здорова"]
        return any(greet in text.lower() for greet in greetings)
    
    def _is_subject_selection(self, text: str) -> bool:
        """Проверяет, является ли текст выбором предмета"""
        subject_words = ["урок", "предмет", "занятие", "изучать", "учить", "хочу", "выбираю", 
                        "математик", "истори", "физик", "хими", "обществ", "биолог", "литератур", "русск"]
        return any(word in text.lower() for word in subject_words)
    
    def _get_dialogue_response(self, text: str) -> Optional[str]:
        """Пытается получить ответ из базы знаний диалога"""
        try:
            # Проверяем общую базу знаний
            response = self.general_knowledge_base.get_dialogue_response(text)
            if response and not response.startswith("Интересный вопрос!"):
                return response
                
            # Проверяем сохраненные ответы LLM
            llm_answer = self.general_knowledge_base.find_llm_answer(text, threshold=0.6)
            if llm_answer:
                return llm_answer
                
        except Exception as e:
            print(f"Ошибка при поиске в базе знаний: {e}")
            
        return None
    
    def _query_llm_directly(self, text: str, context: str = "") -> Optional[str]:
        """Прямой запрос к LLM без проверки базы знаний"""
        try:
            # Определяем тип промпта
            if self._is_greeting(text):
                system_prompt = "Ты - приветливый учитель. Ответь на приветствие и спроси что интересует ученика."
            elif self._is_subject_selection(text):
                system_prompt = "Ты - учитель помогающий выбрать предмет. Ответь на вопрос и подведи к выбору урока."
            else:
                system_prompt = "Ты - helpful учитель. Ответь на вопрос ученика кратко и информативно."
            
            llm_response = self.llm._query_llm_api(
                prompt=text,
                context=context,
                subject="общее",
                system_prompt=system_prompt,
                max_tokens=200
            )
            
            if llm_response:
                # Сохраняем ответ в базу знаний для будущего использования
                try:
                    self.general_knowledge_base.add_llm_answer(text, llm_response)
                    self.general_knowledge_base.add_knowledge(question=text, answer=llm_response)
                except Exception as e:
                    print(f"Ошибка сохранения ответа в базу знаний: {e}")
                
                return llm_response
                
        except Exception as e:
            print(f"Ошибка запроса к LLM: {e}")
            
        return None
    
    def _should_suggest_subject(self, text: str) -> bool:
        """Определяет, нужно ли предлагать выбор предмета"""
        text_lower = text.lower()
        
        # Не предлагаем выбор предмета если:
        if any(word in text_lower for word in ["нет", "не хочу", "не сейчас", "потом", "позже", "не надо"]):
            return False
            
        if any(word in text_lower for word in ["урок", "предмет", "занятие", "изучать", "математик", "истори"]):
            return False
            
        # Предлагаем с вероятностью 30% после 2+ реплик
        return (len(self.conversation_history) >= 2 and 
                random.random() < 0.3 and
                not self.subject_suggested)
    
    def _generate_subject_suggestion(self) -> str:
        """Генерирует предложение выбрать предмет"""
        suggestions = [
            "Кстати, какой предмет тебя интересует?",
            "Давай выберем предмет для урока! Что хочешь изучать?",
            "Какой предмет тебе интересен?",
            "Что бы ты хотел изучить?",
            "Какой урок выберем?"
        ]
        return random.choice(suggestions)
    
    def process_input(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Обработка входящего текста
        Возвращает: (ответ, выбранный_предмет)
        """
        if not text.strip():
            return "Не расслышал, повторите пожалуйста.", None
        
        # Обновляем время последнего взаимодействия
        self.last_interaction_time = time.time()
        
        # Добавляем пользовательский ввод в историю
        self._add_to_conversation_history(text, is_user=True)
        
        # Получаем контекст диалога
        context = self._get_conversation_context()
        
        # ВСЕГДА используем LLM для генерации ответа
        response = self._query_llm_directly(text, context)
        
        # Если LLM не ответил, используем базу знаний как fallback
        if not response:
            response = self._get_dialogue_response(text)
        
        # Если все еще нет ответа, используем fallback
        if not response:
            fallbacks = [
                "Интересный вопрос! Что еще хочешь узнать?",
                "Понятно! Расскажи что тебя интересует?",
                "Хорошо! О чем еще хочешь поговорить?",
                "Я слушаю! Что тебя интересует?"
            ]
            response = random.choice(fallbacks)
        
        # Проверяем, был ли это выбор предмета
        detected_subject = self._detect_subject_intent(text)
        if detected_subject and any(word in text.lower() for word in 
                                  ["хочу", "выбираю", "давай", "начнем", "урок", "занятие"]):
            # Если пользователь явно выбрал предмет, возвращаем его
            self.subject_suggested = True
            self._add_to_conversation_history(response, is_user=False)
            return response, detected_subject
        
        # Добавляем мягкое предложение выбора предмета если нужно
        if self._should_suggest_subject(text):
            subject_suggestion = self._generate_subject_suggestion()
            response = f"{response} {subject_suggestion}"
            self.subject_suggested = True
        
        self._add_to_conversation_history(response, is_user=False)
        return response, None
    
    def get_conversation_history(self) -> List[Dict]:
        """Возвращает историю диалога"""
        return self.conversation_history.copy()
    
    def clear_history(self):
        """Очищает историю диалога"""
        self.conversation_history = []
        self.subject_suggested = False
    
    def is_inactive(self) -> bool:
        """Проверяет, был ли диалог неактивен слишком долго"""
        return time.time() - self.last_interaction_time > self.inactivity_timeout
    
    def reset_inactivity_timer(self):
        """Сбрасывает таймер неактивности"""
        self.last_interaction_time = time.time()
    
    def get_detected_subject(self, text: str) -> Optional[str]:
        """Возвращает обнаруженный предмет в тексте"""
        return self._detect_subject_intent(text)
