# sockets.py
from core import (
    app, socketio, BASE_DIR, LESSONS_DIR, STUDENTS_DIR, STUDENT_PROGRESS_DIR,
    room_ai_activated, room_dialogue, dialogue_init_locks, room_current_avatar,
    debug_log, SLIDES_ENABLED, VISUALIZATION_ENABLED
)
from flask import session
from dialogue import DialogueManager
from pathlib import Path
import threading
import time
import json
import re
from collections import defaultdict

# Глобальные структуры состояния комнат
room_participants = defaultdict(set)
room_student_data = {}
room_last_activity = {}
MAX_ROOMS = 1000
ROOM_TIMEOUT = 3600  # 1 час бездействия

def _fast_room_initialization(room_id: str):
    """Быстрая инициализация комнаты без тяжёлых операций"""
    if room_id not in room_ai_activated:
        room_ai_activated[room_id] = False
    if room_id not in room_current_avatar:
        room_current_avatar[room_id] = 'teacher'
    room_last_activity[room_id] = time.time()
    debug_log(f"⚡ Быстрая инициализация комнаты: {room_id}")

def ensure_dialogue_manager_for_room(room_id: str) -> bool:
    """Ленивая инициализация DialogueManager для комнаты"""
    if room_dialogue[room_id] is not None:
        return True
    
    with dialogue_init_locks[room_id]:
        if room_dialogue[room_id] is not None:
            return True
        
        try:
            # Создаём DialogueManager только при необходимости
            dm = DialogueManager(socketio)
            dm.room_id = room_id
            dm.slides_enabled = SLIDES_ENABLED
            dm.visualization_enabled = VISUALIZATION_ENABLED
            
            # Устанавливаем аватар учителя
            if room_id in room_current_avatar:
                dm.teacher_avatar = room_current_avatar[room_id]
            
            room_dialogue[room_id] = dm
            debug_log(f"✅ Создан DialogueManager для комнаты: {room_id}")
            return True
        except Exception as e:
            debug_log(f"❌ Ошибка создания DialogueManager для {room_id}: {e}")
            return False

def cleanup_inactive_rooms():
    """Фоновая очистка неактивных комнат"""
    while True:
        try:
            current_time = time.time()
            rooms_to_remove = [
                room_id for room_id, last_time in room_last_activity.items()
                if current_time - last_time > ROOM_TIMEOUT
            ]
            for room_id in rooms_to_remove:
                if room_id in room_dialogue and room_dialogue[room_id] is not None:
                    del room_dialogue[room_id]
                if room_id in room_ai_activated:
                    del room_ai_activated[room_id]
                if room_id in room_current_avatar:
                    del room_current_avatar[room_id]
                if room_id in room_last_activity:
                    del room_last_activity[room_id]
                if room_id in room_student_data:
                    del room_student_data[room_id]
                debug_log(f"🗑️ Очищена неактивная комната: {room_id}")
        except Exception as e:
            debug_log(f"❌ Ошибка очистки комнат: {e}")
        time.sleep(600)  # Проверяем каждые 10 минут

# Запускаем фоновую очистку
cleanup_thread = threading.Thread(target=cleanup_inactive_rooms, daemon=True)
cleanup_thread.start()

# ==============================
# 🔌 Основные Socket.IO обработчики
# ==============================

@socketio.on('connect')
def handle_connect():
    """Обработка подключения клиента"""
    debug_log(f"🔌 Клиент подключился: {request.sid}")
    emit('connection_established', {'sid': request.sid})

@socketio.on('disconnect')
def handle_disconnect():
    """Обработка отключения клиента"""
    sid = request.sid
    rooms_left = []
    for room_id in list(room_participants.keys()):
        if sid in room_participants[room_id]:
            room_participants[room_id].remove(sid)
            rooms_left.append(room_id)
            if not room_participants[room_id]:
                debug_log(f"📭 Комната {room_id} стала пустой")
    debug_log(f"🔌 Клиент отключился: {sid}, покинул комнаты: {rooms_left}")

@socketio.on('join_room')
def handle_join_room(data):
    """Присоединение к комнате"""
    room_id = data.get('room_id')
    student_data = data.get('student_data')
    lesson_id = data.get('lesson_id')
    
    if not room_id:
        emit('error', {'message': 'Room ID required'})
        return
    
    _fast_room_initialization(room_id)
    join_room(room_id)
    room_participants[room_id].add(request.sid)
    room_last_activity[room_id] = time.time()
    
    if student_data:
        room_student_data[room_id] = student_data
        # Создаём DialogueManager если нужно (для учеников с данными)
        if lesson_id and lesson_id != 'next':
            ensure_dialogue_manager_for_room(room_id)
            dm = room_dialogue[room_id]
            if dm:
                dm.set_student_data(student_data)
    
    debug_log(f"🚪 Клиент {request.sid} присоединился к комнате: {room_id}")
    emit('room_joined', {
        'room_id': room_id,
        'participants': len(room_participants[room_id]),
        'ai_activated': room_ai_activated.get(room_id, False)
    })

@socketio.on('activate_ai_teacher')
def handle_activate_ai_teacher(data):
    """Активация AI-учителя в комнате"""
    room_id = data.get('room_id')
    avatar_name = data.get('avatar_name', 'teacher')
    
    if not room_id:
        emit('error', {'message': 'Room ID required'})
        return
    
    _fast_room_initialization(room_id)
    room_current_avatar[room_id] = avatar_name
    room_ai_activated[room_id] = True
    room_last_activity[room_id] = time.time()
    
    # ЛЕНИВАЯ ИНИЦИАЛИЗАЦИЯ: создаём DialogueManager только сейчас
    if not ensure_dialogue_manager_for_room(room_id):
        emit('error', {'message': 'Failed to initialize AI teacher'})
        return
    
    dm = room_dialogue[room_id]
    dm.teacher_avatar = avatar_name
    
    # Устанавливаем данные ученика, если есть
    if room_id in room_student_data:
        dm.set_student_data(room_student_data[room_id])
    
    # Приветствие
    greeting = "Здравствуйте! Я ваш AI-учитель. Готов помочь вам изучить новый материал или повторить пройденное."
    if dm.has_student_data and dm.student_data.get('name'):
        greeting = f"Здравствуйте, {dm.student_data['name']}! Я ваш AI-учитель. Готов помочь вам изучить новый материал."
    
    # Озвучиваем в фоне
    def speak_greeting():
        try:
            from app import text_to_speech  # Импортируем функцию озвучки
            audio_data = text_to_speech(greeting, 'ru')
            if audio_data:
                socketio.emit('speech_audio', {
                    'audio': audio_data,
                    'text': greeting,
                    'is_teacher': True,
                    'voice_type': 'female',
                    'subject': 'general'
                }, room=room_id)
        except Exception as e:
            debug_log(f"❌ Ошибка озвучки приветствия: {e}")
    
    # Запускаем озвучку в фоне
    socketio.start_background_task(speak_greeting)
    
    emit('ai_teacher_activated', {
        'room_id': room_id,
        'avatar': avatar_name,
        'greeting': greeting
    }, room=room_id)
    
    debug_log(f"🧠 AI-учитель активирован в комнате {room_id} с аватаром {avatar_name}")

@socketio.on('send_student_speech')
def handle_student_speech(data):
    """Обработка речи ученика"""
    room_id = data.get('room_id')
    text = data.get('text', '').strip()
    is_final = data.get('is_final', True)
    
    if not room_id or not text:
        return
    
    room_last_activity[room_id] = time.time()
    
    # Проверяем, активирован ли AI-учитель
    if not room_ai_activated.get(room_id, False):
        emit('error', {'message': 'AI teacher not activated'}, room=room_id)
        return
    
    # Получаем DialogueManager
    if not ensure_dialogue_manager_for_room(room_id):
        emit('error', {'message': 'Dialogue manager not ready'}, room=room_id)
        return
    
    dm = room_dialogue[room_id]
    
    # Отображаем речь ученика
    emit('student_speech_received', {
        'text': text,
        'timestamp': time.time(),
        'is_final': is_final
    }, room=room_id)
    
    # Обрабатываем только финальную речь
    if not is_final:
        return
    
    # Запускаем обработку в фоне
    def process_speech():
        try:
            response = dm.process_student_input(text, room_id)
            if response:
                # Озвучиваем ответ
                def speak_response():
                    try:
                        from app import text_to_speech
                        # Определяем язык (русский по умолчанию)
                        lang = 'ru'
                        if re.search(r'[a-zA-Z]', response):
                            lang = 'auto'
                        audio_data = text_to_speech(response, lang)
                        if audio_data:
                            socketio.emit('speech_audio', {
                                'audio': audio_data,
                                'text': response,
                                'is_teacher': True,
                                'voice_type': 'female',
                                'subject': dm.current_subject or 'general'
                            }, room=room_id)
                    except Exception as e:
                        debug_log(f"❌ Ошибка озвучки ответа: {e}")
                
                socketio.start_background_task(speak_response)
                
                # Отправляем текстовый ответ
                socketio.emit('teacher_response', {
                    'text': response,
                    'timestamp': time.time(),
                    'current_state': dm.current_state,
                    'is_technical': getattr(dm, 'is_technical_subject', False),
                    'subject_type': getattr(dm, 'subject_type', 'general')
                }, room=room_id)
        except Exception as e:
            debug_log(f"❌ Ошибка обработки речи ученика: {e}")
            socketio.emit('error', {
                'message': 'Ошибка обработки запроса'
            }, room=room_id)
    
    socketio.start_background_task(process_speech)

@socketio.on('request_lesson_start')
def handle_lesson_start_request(data):
    """Запрос на начало урока"""
    room_id = data.get('room_id')
    lesson_id = data.get('lesson_id')
    
    if not room_id or not lesson_id:
        emit('error', {'message': 'Room ID and lesson ID required'})
        return
    
    if not room_ai_activated.get(room_id, False):
        emit('error', {'message': 'AI teacher not activated'})
        return
    
    if not ensure_dialogue_manager_for_room(room_id):
        emit('error', {'message': 'Dialogue manager not ready'})
        return
    
    dm = room_dialogue[room_id]
    
    def start_lesson_task():
        try:
            # Ищем урок по ID
            lesson_file = None
            for lesson_dir in [LESSONS_DIR / "demo", LESSONS_DIR / "generated", LESSONS_DIR / "students"]:
                potential_file = lesson_dir / f"{lesson_id}.txt"
                if potential_file.exists():
                    lesson_file = potential_file
                    break
            
            if not lesson_file:
                # Проверяем в подпапках students
                for class_dir in (LESSONS_DIR / "students").glob("*_class"):
                    for subject_dir in class_dir.iterdir():
                        if subject_dir.is_dir():
                            potential_file = subject_dir / f"{lesson_id}.txt"
                            if potential_file.exists():
                                lesson_file = potential_file
                                break
                    if lesson_file:
                        break
            
            if not lesson_file or not lesson_file.exists():
                socketio.emit('error', {'message': 'Урок не найден'}, room=room_id)
                return
            
            # Начинаем урок
            result = dm._force_start_lesson(str(lesson_file))
            if "Ошибка" in result:
                socketio.emit('error', {'message': result}, room=room_id)
            else:
                socketio.emit('lesson_start_confirmed', {
                    'message': 'Урок начат',
                    'lesson_id': lesson_id
                }, room=room_id)
        except Exception as e:
            debug_log(f"❌ Ошибка начала урока: {e}")
            socketio.emit('error', {'message': f'Ошибка начала урока: {str(e)}'}, room=room_id)
    
    socketio.start_background_task(start_lesson_task)

@socketio.on('skip_to_practice')
def handle_skip_to_practice(data):
    """Пропуск к практике (для тестирования)"""
    room_id = data.get('room_id')
    if not room_id:
        return
    
    if not ensure_dialogue_manager_for_room(room_id):
        return
    
    dm = room_dialogue[room_id]
    result = dm.skip_to_practice()
    emit('practice_skipped', {'message': result}, room=room_id)

@socketio.on('force_visualization')
def handle_force_visualization(data):
    """Принудительная генерация визуализации"""
    room_id = data.get('room_id')
    text = data.get('text', '')
    if not room_id or not text:
        return
    
    if not ensure_dialogue_manager_for_room(room_id):
        return
    
    dm = room_dialogue[room_id]
    success = dm.force_visualization(text)
    emit('visualization_forced', {'success': success}, room=room_id)

@socketio.on('get_dialogue_status')
def handle_get_dialogue_status(data):
    """Получение статуса диалога"""
    room_id = data.get('room_id')
    if not room_id:
        return
    
    status = {
        'room_id': room_id,
        'ai_activated': room_ai_activated.get(room_id, False),
        'has_dialogue': room_dialogue[room_id] is not None,
        'participants': len(room_participants[room_id])
    }
    
    if room_dialogue[room_id] is not None:
        dm = room_dialogue[room_id]
        status.update(dm.get_dialogue_status())
    
    emit('dialogue_status', status, room=room_id)

@socketio.on('set_student_avatar')
def handle_set_student_avatar(data):
    """Установка аватара ученика"""
    room_id = data.get('room_id')
    avatar_name = data.get('avatar_name')
    if not room_id or not avatar_name:
        return
    
    if room_id in room_student_data:
        room_student_data[room_id]['preferred_avatar'] = avatar_name
    
    emit('student_avatar_set', {'avatar': avatar_name}, room=room_id)

# ==============================
# 📊 Служебные обработчики
# ==============================

@socketio.on('ping')
def handle_ping():
    emit('pong', {'timestamp': time.time()})

@socketio.on('error')
def handle_client_error(data):
    debug_log(f"🚨 Ошибка клиента: {data}")

debug_log("✅ Socket.IO обработчики зарегистрированы")
