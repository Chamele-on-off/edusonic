"""
room_manager.py
Модуль управления комнатами и WebSocket-обработчиками для AI-учителя
Выделен из app.py для улучшения читаемости и поддержки кода

ВАЖНО: Эта логика работает идеально и менять ее нельзя ни в коем случае.
Это единственная рабочая реализация.
"""

import time
import threading
import random
from collections import defaultdict
from functools import wraps
import re
from typing import Optional, Dict, List, Any
from flask_socketio import SocketIO, emit, join_room, leave_room
from datetime import datetime

# =============================================================================
# 🔥 ДЕКОРАТОРЫ ДЛЯ БЕЗОПАСНОЙ ОБРАБОТКИ КОМНАТ
# =============================================================================

def room_required(f):
    """Декоратор для проверки существования комнаты"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Первый аргумент - это data из сокета
        if args and isinstance(args[0], dict):
            data = args[0]
            room_id = data.get('room_id')
            
            # Получаем глобальные состояния из контекста приложения
            from app import room_participants, room_ai_activated
            
            if not room_id or room_id not in room_participants:
                emit('room_error', {
                    'room_id': room_id,
                    'error': 'Комната не найдена или не существует'
                })
                return
            
            # Обновляем время активности комнаты
            from app import room_last_activity
            room_last_activity[room_id] = time.time()
            
        return f(*args, **kwargs)
    return decorated_function

def ensure_room_initialized(room_id: str) -> bool:
    """🔥 Безопасная проверка инициализации комнаты без блокировок"""
    try:
        # Импортируем необходимые глобальные состояния
        from app import (
            room_last_activity, room_ai_activated, room_llm_mode,
            room_current_avatar, _fast_room_initialization,
            ensure_dialogue_manager_for_room, room_dialogue
        )
        
        # Обновляем время активности
        room_last_activity[room_id] = time.time()
        
        # Быстрая проверка минимально необходимых параметров
        if room_id not in room_ai_activated:
            room_ai_activated[room_id] = False
        
        if room_id not in room_llm_mode:
            room_llm_mode[room_id] = "traditional"
        
        if room_id not in room_current_avatar:
            room_current_avatar[room_id] = 'teacher'
        
        # Создаем DialogueManager только если его нет
        if room_id not in room_dialogue or room_dialogue[room_id] is None:
            # 🔥 ЛЕНИВАЯ ИНИЦИАЛИЗАЦИЯ: Создаем только при необходимости
            if room_ai_activated.get(room_id, False):
                return ensure_dialogue_manager_for_room(room_id)
        
        return True
    except Exception as e:
        print(f"❌ Ошибка инициализации комнаты {room_id}: {e}")
        return False

# =============================================================================
# 🔥 КЛАСС ДЛЯ УПРАВЛЕНИЯ КОМНАТАМИ
# =============================================================================

class RoomManager:
    """Менеджер комнат для безопасной обработки WebSocket-соединений"""
    
    def __init__(self, socketio: SocketIO, debug_log):
        self.socketio = socketio
        self.debug_log = debug_log
        
        # Импортируем глобальные состояния
        from app import (
            room_participants, room_speech_data, room_speaking,
            room_ai_activated, room_dialogue, room_lessons,
            room_llm_mode, room_teacher_speaking, room_practice_active,
            room_current_question_index, room_current_avatar, room_last_activity,
            room_peer_ids, room_student_data, room_llm_responses,
            room_llm_pending_requests, room_last_poll_time,
            speak_text, llm_manager
        )
        
        self.room_participants = room_participants
        self.room_speech_data = room_speech_data
        self.room_speaking = room_speaking
        self.room_ai_activated = room_ai_activated
        self.room_dialogue = room_dialogue
        self.room_lessons = room_lessons
        self.room_llm_mode = room_llm_mode
        self.room_teacher_speaking = room_teacher_speaking
        self.room_practice_active = room_practice_active
        self.room_current_question_index = room_current_question_index
        self.room_current_avatar = room_current_avatar
        self.room_last_activity = room_last_activity
        self.room_peer_ids = room_peer_ids
        self.room_student_data = room_student_data
        self.room_llm_responses = room_llm_responses
        self.room_llm_pending_requests = room_llm_pending_requests
        self.room_last_poll_time = room_last_poll_time
        self.speak_text = speak_text
        self.llm_manager = llm_manager
        
    # =============================================================================
    # 🔥 ОСНОВНЫЕ МЕТОДЫ УПРАВЛЕНИЯ КОМНАТАМИ
    # =============================================================================
    
    def join_room_handler(self, data):
        """🔥 БЕЗОПАСНЫЙ обработчик присоединения к комнате"""
        room_id = data['room_id']
        peer_id = data.get('peer_id')
        
        self.debug_log(f"Попытка присоединения к комнате {room_id}, peer_id: {peer_id}")
        
        try:
            # Обновляем время активности
            self.room_last_activity[room_id] = time.time()
            
            if room_id not in self.room_participants:
                self.room_participants[room_id] = set()

            join_room(room_id)
            self.room_participants[room_id].add(request.sid)
            
            if peer_id:
                if room_id not in self.room_peer_ids:
                    self.room_peer_ids[room_id] = {}
                self.room_peer_ids[room_id][request.sid] = peer_id
            
            # 🔥 БЫСТРАЯ ИНИЦИАЛИЗАЦИЯ БЕЗ СОЗДАНИЯ DialogueManager
            from app import _fast_room_initialization
            _fast_room_initialization(room_id)
            
            if peer_id:
                emit('participant_joined', {
                    'peer_id': peer_id,
                    'sid': request.sid
                }, room=room_id, include_self=False)
            
            try:
                emit('current_avatar', {'avatar_name': self.room_current_avatar[room_id]}, to=request.sid)
            except Exception as e:
                self.debug_log(f"⚠️ Ошибка отправки аватара: {e}")
            
            if room_id in self.room_speech_data and self.room_speech_data[room_id]:
                try:
                    emit('speech_history', {'history': self.room_speech_data[room_id]}, to=request.sid)
                except Exception as e:
                    self.debug_log(f"⚠️ Ошибка отправки истории: {e}")
            
            emit('participants_update', {'count': len(self.room_participants[room_id])}, room=room_id)
            
            # Приветствие для комнат учеников
            if (room_id in self.room_student_data and 
                self.room_student_data[room_id] and 
                not room_id.startswith('demo_') and 
                room_id != 'default'):
                
                student_data = self.room_student_data[room_id]
                student_name = student_data.get('name', 'ученик')
                subject = student_data.get('subject', 'предмету')
                
                welcome_message = f"{student_name}, привет! Я твои виртуальныи учитель по {subject}. "
                welcome_message += "Даваи начнем наш сегодняшнии урок. Если ты готов начать, скажи 'готов начать'."
                
                self.socketio.emit('student_welcome_message', {
                    'room_id': room_id,
                    'student_name': student_name,
                    'subject': subject,
                    'message': welcome_message,
                    'prompt_ready': True
                }, room=room_id)
                
                self.socketio.start_background_task(lambda: self._delayed_welcome(room_id, welcome_message))
            
            elif len(self.room_participants[room_id]) == 1 and not self.room_ai_activated[room_id]:
                greeting = "Привет! Я ваш виртуальныи учитель. Даваите познакомимся и выберем интересныи урок вместе!"
                self.socketio.start_background_task(lambda: self._delayed_welcome(room_id, greeting))
            
            self.debug_log(f"Успешное присоединение к комнате {room_id}, участников: {len(self.room_participants[room_id])}")
            
        except Exception as e:
            self.debug_log(f"❌ Критическая ошибка при присоединении к комнате {room_id}: {e}")
            try:
                emit('room_error', {
                    'room_id': room_id,
                    'error': f'Join room failed: {str(e)}'
                }, to=request.sid)
            except:
                self.debug_log("⚠️ Не удалось отправить ошибку - клиент уже отключен")
    
    def _delayed_welcome(self, room_id: str, message: str, delay: int = 2):
        """Отправляет приветствие с задержкои"""
        time.sleep(delay)
        self.speak_text(room_id, message, voice_type='female', is_teacher=True, force_lang='ru')
    
    # =============================================================================
    # 🔥 ОБРАБОТЧИКИ СТУДЕНЧЕСКИХ СООБЩЕНИИ
    # =============================================================================
    
    def student_answer_handler(self, data):
        """Обработка ответов ученика"""
        room_id = data['room_id']
        answer = data['answer']
        user_sid = request.sid

        self.debug_log(f"Получен ответ ученика: {answer}")
        self.debug_log(f"Состояние комнаты: practice_active={self.room_practice_active[room_id]}, teacher_speaking={self.room_teacher_speaking[room_id]}")

        if self.room_teacher_speaking[room_id]:
            self.debug_log(f"Игнорирую ответ ученика, так как учитель говорит: {answer}")
            return

        if not self.room_practice_active[room_id]:
            self.debug_log(f"Практика не активна, игнорирую ответ: {answer}")
            return

        if any(cmd in answer.lower() for cmd in ['стоп', 'останови', 'хватит', 'закончи']):
            self.debug_log(f"Команда остановки практики: {answer}")
            
            # Создаем DialogueManager если нужно
            from app import ensure_dialogue_manager_for_room
            if not ensure_dialogue_manager_for_room(room_id):
                return
                
            if room_id in self.room_dialogue:
                self.room_dialogue[room_id]._end_practice_session()
                self.room_practice_active[room_id] = False
                self.room_current_question_index[room_id] = 0
                
                response = "Практика завершена по вашеи команде. Урок окончен!"
                emit('speech_text', {
                    'text': f"Учитель: {response}",
                    'sid': 'teacher',
                    'is_teacher': True
                }, room=room_id)
                
                self.speak_text(room_id, response, voice_type='female', is_teacher=True, force_lang='ru')
                emit('practice_ended', {}, room=room_id)
            return

        # Создаем DialogueManager если нужно
        from app import ensure_dialogue_manager_for_room
        if not ensure_dialogue_manager_for_room(room_id):
            return
            
        dialogue = self.room_dialogue[room_id]
        
        if not dialogue.waiting_for_answer:
            self.debug_log(f"Система не ожидает ответа, игнорирую: {answer}")
            return

        if any(cmd in answer.lower() for cmd in ['продолжаи', 'дальше', 'следующии']):
            self.debug_log(f"Игнорирую команду вместо ответа: {answer}")
            response = dialogue._evaluate_and_generate_next("")
            if response:
                emit('speech_text', {
                    'text': f"Учитель: {response}",
                    'sid': 'teacher',
                    'is_teacher': True
                }, room=room_id)
                self.speak_text(room_id, response, voice_type='female', is_teacher=True)
            return

        self.room_speech_data[room_id].append({
            'text': f"Ответ ученика: {answer}",
            'timestamp': time.time(),
            'type': 'practice_answer',
            'sid': user_sid
        })
        
        self.debug_log(f"Обработка ответа через диалог менеджер...")
        
        response = dialogue._evaluate_and_generate_next(answer)
        
        if response:
            self.debug_log(f"Ответ учителя: {response}")
            
            emit('speech_text', {
                'text': f"Учитель: {response}",
                'sid': 'teacher',
                'is_teacher': True
            }, room=room_id)
            
            self.speak_text(room_id, response, voice_type='female', is_teacher=True)
            
            if not dialogue.practice_active:
                self.room_practice_active[room_id] = False
                self.room_current_question_index[room_id] = 0
                emit('practice_ended', {}, room=room_id)
                self.debug_log("Практика завершена")
        else:
            self.room_practice_active[room_id] = False
            self.room_current_question_index[room_id] = 0
            dialogue.waiting_for_answer = False
            emit('practice_ended', {}, room=room_id)
            self.debug_log("Практика завершена (response=None)")
    
    def student_message_handler(self, data):
        """Обработка сообщений от ученика"""
        room_id = data['room_id']
        message = data['message']
        user_sid = request.sid

        self.debug_log(f"Получено сообщение от ученика: {message}")
        
        if self.room_teacher_speaking[room_id]:
            self.debug_log(f"Игнорирую сообщение ученика, так как учитель говорит: {message}")
            return

        if self.room_practice_active[room_id]:
            self.student_answer_handler({
                'room_id': room_id,
                'answer': message
            })
        else:
            self.recognized_speech_handler({
                'room_id': room_id, 
                'text': message
            })
    
    # =============================================================================
    # 🔥 ОБРАБОТКА РАСПОЗНАННОИ РЕЧИ
    # =============================================================================
    
    def recognized_speech_handler(self, data):
        """🔥 Обработка распознанной речи ученика"""
        room_id = data['room_id']
        text = data['text']
        user_sid = request.sid

        # Обновляем время активности
        self.room_last_activity[room_id] = time.time()
        
        if not self.room_ai_activated.get(room_id, False):
            return
            
        # 🔥 Ленивое создание DialogueManager при необходимости
        from app import ensure_dialogue_manager_for_room
        if not ensure_dialogue_manager_for_room(room_id):
            self.debug_log(f"Не удалось создать DialogueManager для {room_id}")
            return

        if self.room_teacher_speaking[room_id]:
            self.debug_log(f"Игнорирую речь ученика, так как учитель говорит: {text}")
            return

        if (text.startswith("Учитель:") or "учитель" in text.lower() or 
            len(text.strip()) < 3 or text in ["привет", "здравствуите"]):
            return
        
        self.room_speech_data[room_id].append({
            'text': text,
            'timestamp': time.time(),
            'type': 'recognized',
            'sid': user_sid
        })
        if len(self.room_speech_data[room_id]) > 50:
            self.room_speech_data[room_id].pop(0)
        
        emit('speech_text', {'text': text, 'sid': user_sid}, room=room_id)
        
        if self.room_ai_activated[room_id]:
            dialogue = self.room_dialogue[room_id]
            
            if dialogue.is_lesson_started():
                all_continue_commands = [
                    "продолжаи", "продолжить", "дальше", "следующии", "вперед", "даваи дальше",
                    "записал", "понял", "ясно", "ага", "угу", "хорошо", "ок", "ладно", "ясно",
                    "готов", "можно дальше", "слушаю", "понятно", "ясно", "следующии вопрос"
                ]
                
                if any(cmd in text.lower() for cmd in all_continue_commands):
                    next_paragraph = dialogue._get_next_paragraph()
                    if next_paragraph:
                        emit('speech_text', {
                            'text': f"Учитель: {next_paragraph}",
                            'sid': 'teacher',
                            'is_teacher': True
                        }, room=room_id)
                        self.speak_text(room_id, next_paragraph, voice_type='female', is_teacher=True)
                    else:
                        practice_msg = "Урок завершен. Переходим к практике."
                        emit('speech_text', {
                            'text': f"Учитель: {practice_msg}",
                            'sid': 'teacher', 
                            'is_teacher': True
                        }, room=room_id)
                        self.speak_text(room_id, practice_msg, voice_type='female', is_teacher=True, force_lang='ru')
                    return
            
            if any(word in text.lower() for word in ["стоп", "останови", "хватит", "закончи"]):
                stop_response = dialogue.process_input(text)
                if stop_response:
                    emit('speech_text', {
                        'text': f"Учитель: {stop_response}",
                        'sid': 'teacher',
                        'is_teacher': True
                    }, room=room_id)
                    self.speak_text(room_id, stop_response, voice_type='female', is_teacher=True)
                return
            
            if dialogue.is_lesson_started():
                response = dialogue.handle_question_during_lesson(text)
                if response:
                    emit('speech_text', {
                        'text': f"Учитель: {response}",
                        'sid': 'teacher',
                        'is_teacher': True
                    }, room=room_id)
                    self.speak_text(room_id, response, voice_type='female', is_teacher=True)
            else:
                response = dialogue.process_input(text)
                
                if response is None:
                    lesson_data = dialogue.get_selected_lesson()
                    if lesson_data:
                        emit('lesson_started', {
                            'lesson_id': lesson_data['id'],
                            'title': lesson_data['title'],
                            'subject': dialogue.get_current_subject()
                        }, room=room_id)
                        
                        first_paragraph = dialogue._get_next_paragraph()
                        if first_paragraph:
                            emit('speech_text', {
                                'text': f"Учитель: {first_paragraph}",
                                'sid': 'teacher',
                                'is_teacher': True
                            }, room=room_id)
                            self.speak_text(room_id, first_paragraph, voice_type='female', is_teacher=True)
                elif response:
                    emit('speech_text', {
                        'text': f"Учитель: {response}",
                        'sid': 'teacher',
                        'is_teacher': True
                    }, room=room_id)
                    
                    self.speak_text(room_id, response, voice_type='female', is_teacher=True)
                    
                if dialogue.is_lesson_started():
                    lesson_data = dialogue.get_selected_lesson()
                    if lesson_data and not lesson_data.get('lesson_started_emitted', False):
                        lesson_data['lesson_started_emitted'] = True
                        emit('lesson_started', {
                            'lesson_id': lesson_data['id'],
                            'title': lesson_data['title'],
                            'subject': dialogue.get_current_subject(),
                            'is_generated': lesson_data.get('is_generated', False)
                        }, room=room_id)
                        self.debug_log(f"📢 ДОПОЛНИТЕЛЬНО отправлено 'lesson_started' для комнаты {room_id}")
                        
                        first_paragraph = dialogue._get_next_paragraph()
                        if first_paragraph:
                            emit('speech_text', {
                                'text': f"Учитель: {first_paragraph}",
                                'sid': 'teacher',
                                'is_teacher': True
                            }, room=room_id)
                            self.speak_text(room_id, first_paragraph, voice_type='female', is_teacher=True)
    
    # =============================================================================
    # 🔥 УПРАВЛЕНИЕ AI-УЧИТЕЛЕМ
    # =============================================================================
    
    def activate_ai_teacher_handler(self, data):
        """🔥 Активация AI-учителя с ленивои инициализациеи"""
        room_id = data['room_id']
        sid = request.sid
        
        self.debug_log(f"Запрос активации AI-учителя для комнаты {room_id} от {sid}")
        
        try:
            self.room_ai_activated[room_id] = True
            
            # 🔥 Ленивое создание DialogueManager при необходимости
            from app import ensure_dialogue_manager_for_room
            if not ensure_dialogue_manager_for_room(room_id):
                emit('activate_ai_error', {
                    'room_id': room_id,
                    'error': 'Не удалось создать DialogueManager'
                }, to=sid)
                return
            
            dialogue = self.room_dialogue[room_id]
            dialogue.set_llm_mode(self.room_llm_mode[room_id])
            
            greeting = "Привет! Я ваш AI-учитель. Даваите пообщаемся и выберем интересныи урок вместе!"
            self.speak_text(room_id, greeting, voice_type='female', is_teacher=True, force_lang='ru')
            
            emit('ai_teacher_activated', {
                'room_id': room_id,
                'message': 'AI-учитель успешно активирован'
            }, room=room_id)
            
            self.debug_log(f"AI-учитель успешно активирован в комнате {room_id}")
            
        except Exception as e:
            self.debug_log(f"❌ Ошибка активации AI-учителя: {e}")
            emit('activate_ai_error', {
                'room_id': room_id,
                'error': f'Ошибка активации: {str(e)}'
            }, to=sid)
    
    # =============================================================================
    # 🔥 УПРАВЛЕНИЕ ВИЗУАЛИЗАЦИЕИ
    # =============================================================================
    
    def visualization_generated_handler(self, data):
        """Обработка сгенерированнои визуализации"""
        room_id = data['room_id']
        self.debug_log(f"Получена SVG инфографика для комнаты {room_id}: {data['topic'][:100]}...")
        emit('visualization_generated', {
            'room_id': room_id,
            'topic': data['topic'],
            'svg_code': data.get('svg_code', ''),
            'timestamp': data.get('timestamp', time.time()),
            'type': data.get('type', 'infographic')
        }, room=room_id)
    
    def generate_visualization_handler(self, data):
        """🔥 WebSocket генерация SVG инфографики"""
        room_id = data['room_id']
        topic = data.get('topic', '')
        context = data.get('context', '')
        
        if not topic:
            return
        
        self.debug_log(f"WebSocket генерация SVG инфографики для комнаты {room_id}: {topic[:100]}...")
        
        try:
            from llm import LLMIntegration
            llm = LLMIntegration()
            
            result = llm.generate_infographic(topic, context)
            svg_code = result["svg_code"] if result and result.get("success") else self._generate_svg_code_fallback(topic, context)
            
            emit('visualization_generated', {
                'room_id': room_id,
                'topic': topic,
                'svg_code': svg_code,
                'timestamp': time.time(),
                'type': 'infographic'
            }, room=room_id)
            
            self.debug_log(f"✅ SVG инфографика немедленно отправлена в комнату {room_id}")
            
        except Exception as e:
            self.debug_log(f"❌ Ошибка немедленнои генерации SVG инфографики: {e}")
            emit('visualization_generated', {
                'room_id': room_id,
                'topic': topic,
                'svg_code': self._generate_svg_code_fallback(topic, context),
                'timestamp': time.time(),
                'type': 'fallback'
            }, room=room_id)
    
    def _generate_svg_code_fallback(self, topic: str, context: str = "") -> str:
        """Фоллбек-генерация SVG кода"""
        return f'''
<svg width="600" height="300" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#f8fafc"/>
  <rect x="50" y="50" width="500" height="200" rx="10" fill="white" stroke="#e2e8f0" stroke-width="2"/>
  <text x="300" y="100" text-anchor="middle" font-family="Arial" font-size="20" fill="#1e293b">{topic}</text>
  <circle cx="200" cy="160" r="30" fill="#3b82f6" opacity="0.7"/>
  <circle cx="300" cy="160" r="30" fill="#10b981" opacity="0.7"/>
  <circle cx="400" cy="160" r="30" fill="#f59e0b" opacity="0.7"/>
  <text x="300" y="230" text-anchor="middle" font-family="Arial" font-size="14" fill="#64748b">Инфографика</text>
</svg>
'''
    
    # =============================================================================
    # 🔥 УПРАВЛЕНИЕ РЕЖИМАМИ LLM
    # =============================================================================
    
    def set_llm_mode_handler(self, data):
        """Изменение режима LLM"""
        room_id = data['room_id']
        mode = data['mode']
        
        if mode in ["traditional", "llm_first"]:
            self.room_llm_mode[room_id] = mode
            # Устанавливаем режим только если DialogueManager уже создан
            if room_id in self.room_dialogue and self.room_dialogue[room_id] is not None:
                self.room_dialogue[room_id].set_llm_mode(mode)
            
            emit('llm_mode_changed', {
                'mode': mode,
                'room': room_id
            }, room=room_id)
            
            self.debug_log(f"Режим LLM изменен в комнате {room_id}: {mode}")
    
    def set_llm_priority_handler(self, data):
        """Установка приоритета LLM"""
        room_id = data['room_id']
        priority = data['priority']
        
        valid_priorities = ["local_first", "openrouter_first", "local_only", "openrouter_only"]
        
        if priority not in valid_priorities:
            emit('llm_priority_error', {
                'room_id': room_id,
                'error': f'Invalid priority. Use: {valid_priorities}'
            })
            return
        
        if room_id in self.room_dialogue and self.room_dialogue[room_id] is not None:
            self.room_dialogue[room_id].llm.set_priority(priority)
            status = self.room_dialogue[room_id].llm.get_priority_status()
            
            emit('llm_priority_changed', {
                'room_id': room_id,
                'priority': priority,
                'status': status
            }, room=room_id)
            
            self.debug_log(f"Приоритет LLM изменен в комнате {room_id}: {priority}")
    
    # =============================================================================
    # 🔥 АСИНХРОННЫЕ ЗАПРОСЫ К LLM
    # =============================================================================
    
    def async_llm_request_handler(self, data):
        """🔥 Асинхронные запросы к LLM"""
        room_id = data['room_id']
        prompt = data['prompt']
        system_prompt = data.get('system_prompt', '')
        max_tokens = data.get('max_tokens', 1000)
        request_type = data.get('type', 'general')
        client_request_id = data.get('request_id')
        
        self.debug_log(f"Запрос от комнаты {room_id}: {prompt[:100]}...")
        
        request_id = client_request_id or f"{room_id}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        
        self.room_llm_pending_requests[room_id][request_id] = {
            'prompt': prompt,
            'system_prompt': system_prompt,
            'max_tokens': max_tokens,
            'timestamp': time.time(),
            'type': request_type
        }
        
        current_time = time.time()
        for req_id in list(self.room_llm_pending_requests[room_id].keys()):
            if current_time - self.room_llm_pending_requests[room_id][req_id]['timestamp'] > 300:
                del self.room_llm_pending_requests[room_id][req_id]
        
        llm_request_id = self.llm_manager.submit_request(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            room_id=room_id,
            request_id=request_id
        )
        
        self.room_llm_pending_requests[room_id][request_id]['manager_id'] = llm_request_id
        
        emit('llm_request_queued', {
            'request_id': request_id,
            'manager_id': llm_request_id,
            'queue_position': self.llm_manager.get_queue_size(),
            'room_id': room_id,
            'timestamp': time.time()
        })
    
    def llm_async_response_handler(self, data):
        """🔥 Обработка асинхронных ответов от LLM"""
        room_id = data['room_id']
        response = data['response']
        request_id = data['request_id']
        
        self.debug_log(f"Ответ для комнаты {room_id}: {response[:100]}...")
        
        if room_id in self.room_llm_pending_requests and request_id in self.room_llm_pending_requests[room_id]:
            del self.room_llm_pending_requests[room_id][request_id]
        
        if response and room_id in self.room_dialogue and self.room_dialogue[room_id] is not None:
            self.room_dialogue[room_id].llm.handle_llm_response(request_id, response, room_id)
            
            emit('speech_text', {
                'text': f"Учитель: {response}",
                'sid': 'teacher',
                'is_teacher': True
            }, room=room_id)
            
            self.speak_text(room_id, response, voice_type='female', is_teacher=True)
    
    # =============================================================================
    # 🔥 УПРАВЛЕНИЕ ПРАКТИКОИ
    # =============================================================================
    
    def practice_started_handler(self, data):
        """Начало практики"""
        room_id = data['room_id']
        self.room_practice_active[room_id] = True
        self.room_current_question_index[room_id] = 0
        emit('practice_started', {}, room=room_id)
        self.debug_log(f"Практика начата в комнате {room_id}")
    
    def practice_ended_handler(self, data):
        """Завершение практики"""
        room_id = data['room_id']
        self.room_practice_active[room_id] = False
        self.room_current_question_index[room_id] = 0
        emit('practice_ended', {}, room=room_id)
        self.debug_log(f"Практика завершена в комнате {room_id}")
    
    # =============================================================================
    # 🔥 УПРАВЛЕНИЕ СЛАИДАМИ УРОКОВ
    # =============================================================================
    
    def get_lesson_slides_handler(self, data):
        """WebSocket запрос для получения слаидов урока"""
        try:
            room_id = data['room_id']
            lesson_id = data['lesson_id']
            
            self.debug_log(f"Запрос слаидов для урока {lesson_id} в комнате {room_id}")
            
            from app import get_lesson_slides_api
            slides_data = get_lesson_slides_api(lesson_id)
            
            if slides_data['success']:
                emit('lesson_slides_loaded', {
                    'room_id': room_id,
                    'lesson_id': lesson_id,
                    'slides': slides_data['slides'],
                    'slides_count': slides_data['slides_count'],
                    'has_slides': slides_data['has_slides']
                }, room=room_id)
                
                self.debug_log(f"✅ Слаиды отправлены в комнату {room_id}: {len(slides_data['slides'])} слаидов")
            else:
                emit('lesson_slides_error', {
                    'room_id': room_id,
                    'lesson_id': lesson_id,
                    'error': slides_data.get('error', 'Неизвестная ошибка')
                }, room=room_id)
                
        except Exception as e:
            self.debug_log(f"❌ Ошибка обработки запроса слаидов: {e}")
            emit('lesson_slides_error', {
                'room_id': data.get('room_id', 'unknown'),
                'lesson_id': data.get('lesson_id', 'unknown'),
                'error': str(e)
            })
    
    # =============================================================================
    # 🔥 ОБРАБОТЧИКИ СОБЫТИИ АВАТАРОВ
    # =============================================================================
    
    def handle_client_start_animation(self, data):
        """Запуск анимации аватара"""
        room_id = data['room_id']
        avatar_name = data['avatar_name']
        self.debug_log(f"Получена команда запуска анимации для комнаты {room_id}, аватар: {avatar_name}")
        
        self.room_current_avatar[room_id] = avatar_name
        emit('avatar_changed', {'avatar_name': avatar_name}, room=room_id)
        emit('animation_ready', {'status': 'ready'}, room=room_id)
    
    def handle_avatar_changed(self, data):
        """Изменение аватара"""
        room_id = data['room_id']
        avatar_name = data['avatar_name']
        self.debug_log(f"Смена аватара в комнате {room_id} на {avatar_name}")
        
        self.room_current_avatar[room_id] = avatar_name
        emit('avatar_changed', {'avatar_name': avatar_name}, room=room_id)
    
    def handle_generate_speech(self, data):
        """Генерация речи"""
        room_id = data['room_id']
        text = data['text']
        voice_type = data.get('voice', 'male')
        self.speak_text(room_id, text, voice_type)
    
    def handle_get_current_avatar(self, data):
        """Получение текущего аватара"""
        room_id = data['room_id']
        emit('current_avatar', {'avatar_name': self.room_current_avatar[room_id]}, to=request.sid)
    
    # =============================================================================
    # 🔥 УПРАВЛЕНИЕ СОСТОЯНИЕМ LLM
    # =============================================================================
    
    def get_llm_status_handler(self, data):
        """Получение статуса LLM"""
        room_id = data['room_id']
        
        if room_id in self.room_dialogue and self.room_dialogue[room_id] is not None:
            status = self.room_dialogue[room_id].llm.get_llm_status()
            emit('llm_status_update', {
                'room_id': room_id,
                'status': status
            }, room=room_id)
    
    def get_llm_priority_status_handler(self, data):
        """Получение статуса приоритета LLM"""
        room_id = data['room_id']
        
        if room_id in self.room_dialogue and self.room_dialogue[room_id] is not None:
            status = self.room_dialogue[room_id].llm.get_priority_status()
            emit('llm_priority_status', {
                'room_id': room_id,
                'status': status
            })
    
    # =============================================================================
    # 🔥 ПОЛНЫИ НАБОР ОБРАБОТЧИКОВ ДЛЯ ПРИВЯЗКИ В APP.PY
    # =============================================================================
    
    def register_all_handlers(self):
        """🔥 Регистрирует все обработчики сокетов"""
        
        @self.socketio.on('join_room')
        def handle_join_room(data):
            self.join_room_handler(data)
        
        @self.socketio.on('student_answer')
        def handle_student_answer(data):
            self.student_answer_handler(data)
        
        @self.socketio.on('student_message')
        def handle_student_message(data):
            self.student_message_handler(data)
        
        @self.socketio.on('recognized_speech')
        def handle_recognized_speech(data):
            self.recognized_speech_handler(data)
        
        @self.socketio.on('activate_ai_teacher')
        def handle_activate_ai_teacher(data):
            self.activate_ai_teacher_handler(data)
        
        @self.socketio.on('visualization_generated')
        def handle_visualization_generated(data):
            self.visualization_generated_handler(data)
        
        @self.socketio.on('generate_visualization')
        def handle_generate_visualization(data):
            self.generate_visualization_handler(data)
        
        @self.socketio.on('set_llm_mode')
        def handle_set_llm_mode(data):
            self.set_llm_mode_handler(data)
        
        @self.socketio.on('set_llm_priority')
        def handle_set_llm_priority(data):
            self.set_llm_priority_handler(data)
        
        @self.socketio.on('async_llm_request')
        def handle_async_llm_request(data):
            self.async_llm_request_handler(data)
        
        @self.socketio.on('llm_async_response')
        def handle_llm_async_response(data):
            self.llm_async_response_handler(data)
        
        @self.socketio.on('practice_started')
        def handle_practice_started(data):
            self.practice_started_handler(data)
        
        @self.socketio.on('practice_ended')
        def handle_practice_ended(data):
            self.practice_ended_handler(data)
        
        @self.socketio.on('get_lesson_slides')
        def handle_get_lesson_slides(data):
            self.get_lesson_slides_handler(data)
        
        @self.socketio.on('client_start_animation')
        def handle_client_start_animation(data):
            self.handle_client_start_animation(data)
        
        @self.socketio.on('avatar_changed')
        def handle_avatar_changed(data):
            self.handle_avatar_changed(data)
        
        @self.socketio.on('generate_speech')
        def handle_generate_speech(data):
            self.handle_generate_speech(data)
        
        @self.socketio.on('get_current_avatar')
        def handle_get_current_avatar(data):
            self.handle_get_current_avatar(data)
        
        @self.socketio.on('get_llm_status')
        def handle_get_llm_status(data):
            self.get_llm_status_handler(data)
        
        @self.socketio.on('get_llm_priority_status')
        def handle_get_llm_priority_status(data):
            self.get_llm_priority_status_handler(data)
        
        self.debug_log("✅ Все обработчики комнат зарегистрированы")
