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

class DialogueManager:
    def __init__(self, socketio):
        self.socketio = socketio
        self.dialogue_states = {
            "greeting": self._handle_greeting,
            "subject_selection": self._handle_subject_selection,
            "lesson_reading": self._handle_lesson_reading,
            "practice_session": self._handle_practice_session
        }
        self.current_state = "greeting"
        self.current_subject = None
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
        
        # Новые поля для практики
        self.practice_active = False
        self.current_question_index = 0
        self.current_expected_answer = ""
        
        # Новые поля для улучшенного диалога
        self.last_subject_prompt_time = 0
        self.subject_prompt_cooldown = 30
        self.subject_prompt_variants = [
            "Давайте выберем предмет для урока! У меня есть: {subjects}. Что вас интересует?",
            "Какой предмет хотите изучить сегодня? Доступно: {subjects}.",
            "Сказать, какие предметы я преподаю? Или может ты хочешь изучить что-то определенное? У меня есть: {subjects}.",
            "Что будем изучать? Выбирайте из: {subjects}.",
            "Готов начать урок! Какой предмет вас интересует? У меня есть: {subjects}."
        ]
        
        self._load_lessons()
        
        # Расширенные локальные шаблоны
        self.local_patterns = {
            "привет": ["Привет! Рад вас видеть. Как ваше настроение?", "Здравствуйте! Готовы к интересному уроку?"],
            "как дела": ["Все прекрасно! Готов помочь вам с обучением.", "Отлично! А как ваши успехи в учебе?"],
            "спасибо": ["Всегда пожалуйста! Рад был помочь.", "Не стоит благодарности! Это моя работа."],
            "не понимаю": ["Давайте разберем этот момент еще раз вместе.", "Хорошо, объясню по-другому, чтобы было понятнее."],
            "повтори": ["Конечно, повторяю для вас...", "С удовольствием скажу еще раз."],
            "скучно": ["Давайте сделаем урок более интересным! Может, викторину?", "Понимаю. Предлагаю сменить активность!"],
            "трудно": ["Не переживайте! Сложности - это нормально. Я помогу разобраться.", "Вместе мы обязательно справимся!"],
            "молодец": ["Спасибо! Стараюсь для вас.", "Вы тоже молодец, что так активно участвуете!"],
            "хорошо": ["Прекрасно! Продолжаем наш урок.", "Отлично! Двигаемся дальше."],
            "не знаю": ["Это нормально не знать! Сейчас вместе разберемся.", "Отличный повод узнать что-то новое!"],
            "стоп": ["Останавливаю урок. Скажите 'привет', когда будете готовы продолжить.", "Прерываю чтение. Жду вашей команды."],
            "кто ты": ["Я ваш виртуальный учитель с искусственным интеллектом! Готов помочь с обучением.", 
                      "AI-учитель, который сделает ваше обучение интересным и эффективным."],
            "что умеешь": ["Я могу проводить уроки, отвечать на вопросы, объяснять сложные темы и делать обучение увлекательным!", 
                          "Умею преподавать разные предметы, отвечать на ваши вопросы и адаптироваться под ваш уровень."],
            "расскажи о себе": ["Я цифровой преподаватель, созданный чтобы сделать образование доступным и интересным для всех!", 
                               "Моя задача - помочь вам учиться с удовольствием и пониманием."]
        }

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
                "привет": ["Привет! Рад тебя видеть!", "Здравствуй! Готов к учебе?"],
                "здравствуй": ["Привет! Как настроение?", "Здравствуй! Что будем изучать?"]
            },
            "mood_patterns": {
                "как дела": ["Отлично! А у тебя как?", "Прекрасно! Готов к уроку."]
            },
            "learning_patterns": {
                "хочу учиться": ["Отлично! Какой предмет тебя интересует?", "Супер! Давай выберем тему!"]
            },
            "subject_questions": {
                "что преподаешь": ["У меня есть уроки по разным предметам! Что хочешь изучить?"]
            },
            "metadata": {
                "version": "1.0",
                "type": "default_dialogue_patterns"
            }
        }

    def _load_lessons(self):
        """Загружает список доступных уроков"""
        self.lessons = {}
        try:
            if not self.lessons_dir.exists():
                self.lessons_dir.mkdir(parents=True)
                # Создаем демо-урок по обществознанию, если его нет
                demo_lesson = self.lessons_dir / "social_general.txt"
                if not demo_lesson.exists():
                    with open(demo_lesson, 'w', encoding='utf-8') as f:
                        f.write("Основы обществознания: подготовка к ЕГЭ.\n\nДобро пожаловать на демо-урок! Сегодня мы разберем фундаментальные понятия обществознания.\n\nОбщество - это сложная динамическая система, объединяющая людей, которые связаны совместной деятельностью, общими интересами и ценностями.\n\nГосударство - это политическая организация общества, обладающая суверенитетом и аппаратом управления.\n\nДемократия - это форма правления, при которой народ является источником власти.\n\nЭкономика - это хозяйственная деятельность общества, система производства и распределения товаров.\n\nКультура - это совокупность достижений человечества в духовной и материальной жизни.\n\nПраво - это система общеобязательных норм, охраняемых государством.\n\nСоциализация - это процесс усвоения