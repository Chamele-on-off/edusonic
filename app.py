from flask import Flask, render_template, send_from_directory, jsonify, request, send_file
import os
from pathlib import Path
from flask_socketio import SocketIO, emit, join_room, leave_room
from gtts import gTTS
import io
import base64
import time
import threading
from collections import defaultdict
import random
from dialogue import DialogueManager
from config import update_api_key, get_api_key, load_config, get_model_config, get_llm_mode, set_llm_mode, get_llm_priority, set_llm_priority
import requests
import json
from datetime import datetime
from werkzeug.utils import secure_filename
import re
import tempfile
from local_llm_manager import get_llm_manager

# Настройка SocketIO с правильными таймаутами
app = Flask(__name__, static_folder='static')
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='threading',
    ping_timeout=60,           # Увеличенный таймаут ping
    ping_interval=25,          # Более частые ping
    max_http_buffer_size=1e8,  # Увеличенный размер буфера
    logger=True,               # Логирование для отладки
    engineio_logger=True,      # Логирование EngineIO
    always_connect=True        # Всегда пытаться подключиться
)

BASE_DIR = Path(__file__).parent
FRAMES_DIR = BASE_DIR / 'static' / 'avatar' / 'frames'
LESSONS_DIR = BASE_DIR / 'lessons'
MATERIALS_DIR = BASE_DIR / 'materials'
PRACTICE_DIR = BASE_DIR / 'materials' / 'practice'

# Создаем необходимые папки
for folder in [LESSONS_DIR, MATERIALS_DIR, PRACTICE_DIR]:
    os.makedirs(folder, exist_ok=True)

# Глобальные состояния
room_participants = defaultdict(set)
room_speech_data = defaultdict(list)
room_speaking = defaultdict(bool)
room_ai_activated = defaultdict(bool)
room_dialogue = defaultdict(lambda: DialogueManager(socketio))
room_lessons = defaultdict(dict)
room_llm_mode = defaultdict(lambda: get_llm_mode())
room_teacher_speaking = defaultdict(bool)
room_practice_active = defaultdict(bool)
room_current_question_index = defaultdict(int)
room_current_avatar = defaultdict(lambda: 'teacher')

# Кэш для визуализаций
diagram_cache = {}
# Очередь визуализаций для каждой комнаты
room_visualization_queue = defaultdict(list)
# Флаг активной визуализации для каждой комнаты
room_visualization_active = defaultdict(bool)

# Очереди ответов LLM для polling
room_llm_responses = defaultdict(list)
room_last_poll_time = defaultdict(lambda: 0)

# Менеджер локальной LLM
llm_manager = get_llm_manager()

def setup_llm_manager():
    """Настройка менеджера LLM"""
    # Запускаем менеджер
    llm_manager.start()
    
    # Регистрируем глобальный callback для обработки ответов
    def global_llm_callback(request_id, response, room_id):
        """Глобальный обработчик ответов от LLM"""
        print(f"🔧 [Global Callback] Получен ответ для комнаты {room_id}: {response[:100]}...")
        
        # Добавляем ответ в очередь для polling
        room_llm_responses[room_id].append({
            'request_id': request_id,
            'response': response,
            'timestamp': time.time()
        })
        
        # Ограничиваем размер очереди
        if len(room_llm_responses[room_id]) > 10:
            room_llm_responses[room_id].pop(0)
        
        # Отправляем ответ через WebSocket (если клиент подключен)
        try:
            socketio.emit('llm_async_response', {
                'request_id': request_id,
                'response': response,
                'room_id': room_id,
                'timestamp': time.time()
            }, room=room_id)
            print(f"✅ Ответ отправлен через WebSocket в комнату {room_id}")
        except Exception as e:
            print(f"⚠️ Не удалось отправить через WebSocket: {e}. Ответ сохранен для polling.")
    
    # Регистрируем глобальный callback
    llm_manager.register_room_callback('global', global_llm_callback)
    print("✅ LLM Manager настроен")

def reset_speaking_state(room_id, is_teacher=False):
    """Сбрасывает состояние речи для указанной комнаты"""
    room_speaking[room_id] = False
    if is_teacher:
        room_teacher_speaking[room_id] = False
    socketio.emit('speaking_state', {'speaking': False}, room=room_id)

def speak_text(room_id, text, voice_type='female', is_teacher=False, skip_history=False):
    """Озвучивает текст и добавляет его в историю"""
    if not text.strip():
        return
        
    # Устанавливаем флаг, что учитель начинает говорить
    if is_teacher:
        room_teacher_speaking[room_id] = True
        
    room_speaking[room_id] = True
    socketio.emit('speaking_state', {'speaking': True}, room=room_id)
    
    audio_data = text_to_speech(text, lang='ru')
    if audio_data:
        emit('speech_audio', {
            'audio': audio_data,
            'text': text,
            'timestamp': time.time(),
            'voice_type': voice_type,
            'is_teacher': is_teacher
        }, room=room_id)
        
        if not skip_history:
            room_speech_data[room_id].append({
                'text': text,
                'timestamp': time.time(),
                'type': 'generated',
                'voice_type': voice_type,
                'is_teacher': is_teacher
            })
            if len(room_speech_data[room_id]) > 50:
                room_speech_data[room_id].pop(0)
    
    # Длительность речи рассчитываем на основе длины текста
    speech_duration = max(2, len(text) * 0.1)
    threading.Timer(speech_duration, lambda: reset_speaking_state(room_id, is_teacher)).start()

@app.route('/')
def home():
    return render_template('teacher.html')

@app.route('/conference')
def conference():
    room_id = request.args.get('room', 'default')
    embed = request.args.get('embed', 'false') == 'true'
    return render_template('conference.html', room_id=room_id, embed=embed)

@app.route('/api/avatars')
def get_avatars():
    try:
        avatars = [d for d in os.listdir(FRAMES_DIR) if (FRAMES_DIR / d).is_dir()]
        return jsonify({"avatars": avatars})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/frames/<avatar_name>')
def get_frames(avatar_name):
    try:
        avatar_dir = FRAMES_DIR / avatar_name
        if not avatar_dir.exists():
            return jsonify({"error": "Avatar not found"}), 404
        
        # Поддерживаемые форматы изображений
        supported_formats = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
        frames = [f for f in os.listdir(avatar_dir) if f.lower().endswith(supported_formats)]
        
        # Сортируем кадры для правильной последовательности
        frames.sort()
        
        return jsonify({"frames": frames})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/frames/<avatar_name>/<path:filename>')
def serve_frame(avatar_name, filename):
    return send_from_directory(FRAMES_DIR / avatar_name, filename)

def text_to_speech(text, lang='ru'):
    """Преобразует текст в аудио (base64)"""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return base64.b64encode(mp3_fp.read()).decode('utf-8')
    except Exception as e:
        print(f"Error in text_to_speech: {e}")
        return None

@socketio.on('connect')
def handle_connect():
    print(f'✅ Client connected: {request.sid}')
    emit('connection_established', {'message': 'Connected to server', 'sid': request.sid})

@socketio.on('disconnect')
def handle_disconnect():
    print(f'❌ Client disconnected: {request.sid}')
    for room_id in list(room_participants.keys()):
        if request.sid in room_participants[room_id]:
            room_participants[room_id].remove(request.sid)
            emit('participant_left', {'sid': request.sid}, room=room_id)
            emit('participants_update', {'count': len(room_participants[room_id])}, room=room_id)

@socketio.on('join_room')
def handle_join_room(data):
    room_id = data['room_id']
    join_room(room_id)
    room_participants[room_id].add(request.sid)
    
    if room_id not in room_dialogue:
        room_dialogue[room_id] = DialogueManager(socketio)
        room_dialogue[room_id].room_id = room_id
    
    # Устанавливаем режим LLM для диалог менеджера комнаты
    room_dialogue[room_id].set_llm_mode(room_llm_mode[room_id])
    
    if len(room_participants[room_id]) == 1:
        greeting = "Привет! Я ваш виртуальный учитель. Давайте познакомимся и выберем интересный урок вместе!"
        speak_text(room_id, greeting, voice_type='female', is_teacher=True)
    
    emit('participants_update', {'count': len(room_participants[room_id])}, room=room_id)
    emit('new_participant', {'sid': request.sid}, room=room_id)
    
    # Отправляем текущий аватар комнаты новому участнику
    emit('current_avatar', {'avatar_name': room_current_avatar[room_id]}, to=request.sid)
    
    if len(room_participants[room_id]) == 2 and not room_ai_activated[room_id]:
        welcome_text = "Учитель с искусственным интеллектом активирован"
        speak_text(room_id, welcome_text, voice_type='female', is_teacher=True)
        emit('ai_teacher_available', {}, room=room_id)
    
    if room_speech_data[room_id]:
        emit('speech_history', {'history': room_speech_data[room_id]}, to=request.sid)

@socketio.on('get_current_avatar')
def handle_get_current_avatar(data):
    """Отправляет текущий аватар комнаты"""
    room_id = data['room_id']
    emit('current_avatar', {'avatar_name': room_current_avatar[room_id]}, to=request.sid)

@socketio.on('client_start_animation')
def handle_client_start_animation(data):
    """Обработчик команды запуска анимации от учителя"""
    room_id = data['room_id']
    avatar_name = data['avatar_name']
    print(f"Получена команда запуска анимации для комнаты {room_id}, аватар: {avatar_name}")
    
    # Сохраняем текущий аватар комнаты
    room_current_avatar[room_id] = avatar_name
    
    # Уведомляем всех клиентов в комнате о смене аватара
    emit('avatar_changed', {'avatar_name': avatar_name}, room=room_id)
    emit('animation_ready', {'status': 'ready'}, room=room_id)

@socketio.on('avatar_changed')
def handle_avatar_changed(data):
    """Обработчик смены аватара"""
    room_id = data['room_id']
    avatar_name = data['avatar_name']
    print(f"Смена аватара в комнате {room_id} на {avatar_name}")
    
    # Сохраняем новый аватар для комнаты
    room_current_avatar[room_id] = avatar_name
    
    # Пересылаем команду всем клиентам в комнате
    emit('avatar_changed', {'avatar_name': avatar_name}, room=room_id)

@socketio.on('generate_speech')
def handle_generate_speech(data):
    room_id = data['room_id']
    text = data['text']
    voice_type = data.get('voice', 'male')
    speak_text(room_id, text, voice_type)

@socketio.on('student_answer')
def handle_student_answer(data):
    """Обработчик ответов ученика во время практики с последовательной генерацией"""
    room_id = data['room_id']
    answer = data['answer']
    user_sid = request.sid

    print(f"📝 Получен ответ ученика: {answer}")
    print(f"📊 Состояние комнаты: practice_active={room_practice_active[room_id]}, teacher_speaking={room_teacher_speaking[room_id]}")

    # ИГНОРИРУЕМ ответ, если учитель говорит
    if room_teacher_speaking[room_id]:
        print(f"🔇 Игнорирую ответ ученика, так как учитель говорит: {answer}")
        return

    # Проверяем, активна ли практика
    if not room_practice_active[room_id]:
        print(f"🔇 Практика не активна, игнорирую ответ: {answer}")
        return

    # Проверяем, ожидает ли система ответа в диалог менеджере
    if room_id in room_dialogue:
        dialogue = room_dialogue[room_id]
        if not dialogue.waiting_for_answer:
            print(f"🔇 Система не ожидает ответа, игнорирую: {answer}")
            return

    # Добавляем ответ в историю
    room_speech_data[room_id].append({
        'text': f"Ответ ученика: {answer}",
        'timestamp': time.time(),
        'type': 'practice_answer',
        'sid': user_sid
    })
    
    # Обрабатываем ответ через диалог менеджер
    if room_id in room_dialogue:
        print(f"🔄 Обработка ответа через диалог менеджер...")
        
        # Сбрасываем флаг ожидания ПЕРЕД обработкой
        room_dialogue[room_id].waiting_for_answer = False
        
        # Используем новый метод для последовательной обработки
        response = room_dialogue[room_id]._evaluate_and_generate_next(answer)
        
        if response:
            print(f"🎯 Ответ учителя: {response}")
            
            # Отправляем ответ учителя
            emit('speech_text', {
                'text': f"Учитель: {response}",
                'sid': 'teacher',
                'is_teacher': True
            }, room=room_id)
            
            # Озвучиваем ответ
            speak_text(room_id, response, voice_type='female', is_teacher=True)
            
            # Проверяем, завершена ли практика
            if not room_dialogue[room_id].practice_active:
                room_practice_active[room_id] = False
                room_current_question_index[room_id] = 0
                emit('practice_ended', {}, room=room_id)
                print("🏁 Практика завершена")
        else:
            # Если response is None, практика завершена
            room_practice_active[room_id] = False
            room_current_question_index[room_id] = 0
            room_dialogue[room_id].waiting_for_answer = False
            emit('practice_ended', {}, room=room_id)
            print("🏁 Практика завершена (response=None)")

@socketio.on('student_message')
def handle_student_message(data):
    """Обработчик сообщений от ученика через текстовое поле"""
    room_id = data['room_id']
    message = data['message']
    user_sid = request.sid

    print(f"📝 Получено сообщение от ученика: {message}")
    
    # Если активна практика, обрабатываем как ответ
    if room_practice_active[room_id]:
        handle_student_answer({
            'room_id': room_id,
            'answer': message
        })
    else:
        # Иначе обрабатываем как обычную речь
        handle_recognized_speech({
            'room_id': room_id, 
            'text': message
        })

@socketio.on('recognized_speech')
def handle_recognized_speech(data):
    room_id = data['room_id']
    text = data['text']
    user_sid = request.sid

    # ИГНОРИРУЕМ распознанную речь, если учитель говорит
    if room_teacher_speaking[room_id]:
        print(f"Игнорирую речь ученика, так как учитель говорит: {text}")
        return

    # Игнорируем распознавание системных сообщений и короткие фразы
    if (text.startswith("Учитель:") or "учитель" in text.lower() or 
        len(text.strip()) < 3 or text in ["привет", "здравствуйте"]):
        return
    
    room_speech_data[room_id].append({
        'text': text,
        'timestamp': time.time(),
        'type': 'recognized',
        'sid': user_sid
    })
    if len(room_speech_data[room_id]) > 50:
        room_speech_data[room_id].pop(0)
    
    emit('speech_text', {'text': text, 'sid': user_sid}, room=room_id)
    
    if room_ai_activated[room_id]:
        dialogue = room_dialogue[room_id]
        
        # УЛУЧШЕННАЯ ОБРАБОТКА КОМАНД УПРАВЛЕНИЯ
        continue_commands = ["продолжай", "продолжить", "дальше", "следующий", "вперед", "давай дальше"]
        recorded_commands = ["записал", "понял", "ясно", "ага", "угу", "хорошо", "ок", "ладно", "ясно"]
        
        # Если урок начат и это команда продолжения
        if dialogue.is_lesson_started() and any(cmd in text.lower() for cmd in continue_commands + recorded_commands):
            # Получаем следующий абзац урока
            next_paragraph = dialogue._get_next_paragraph()
            if next_paragraph:
                # Отправляем текст
                emit('speech_text', {
                    'text': f"Учитель: {next_paragraph}",
                    'sid': 'teacher',
                    'is_teacher': True
                }, room=room_id)
                # Озвучиваем следующий абзац
                speak_text(room_id, next_paragraph, voice_type='female', is_teacher=True)
            return
        
        # Команды остановки
        if any(word in text.lower() for word in ["стоп", "останови", "хватит", "закончи"]):
            stop_response = dialogue.process_input(text)
            if stop_response:
                # Отправляем текст
                emit('speech_text', {
                    'text': f"Учитель: {stop_response}",
                    'sid': 'teacher',
                    'is_teacher': True
                }, room=room_id)
                # Озвучиваем ответ на остановку
                speak_text(room_id, stop_response, voice_type='female', is_teacher=True)
            return
        
        # Если урок уже начат, обрабатываем как вопрос/команду
        if dialogue.is_lesson_started():
            # Обработка вопросов во время чтения урока
            response = dialogue.handle_question_during_lesson(text)
            if response:
                # Отправляем текст
                emit('speech_text', {
                    'text': f"Учитель: {response}",
                    'sid': 'teacher',
                    'is_teacher': True
                }, room=room_id)
                # ОЗВУЧИВАЕМ ответ на вопрос (всегда!)
                speak_text(room_id, response, voice_type='female', is_teacher=True)
        else:
            # Обработка диалога выбора урока
            response = dialogue.process_input(text)
            
            # Если response None - это значит был выбран предмет и нужно начать урок
            if response is None:
                # Урок выбран, начинаем чтение
                lesson_data = dialogue.get_selected_lesson()
                if lesson_data:
                    emit('lesson_started', {
                        'lesson_id': lesson_data['id'],
                        'title': lesson_data['title'],
                        'subject': dialogue.get_current_subject()
                    }, room=room_id)
                    
                    # Немедленно начинаем чтение первого абзаца урока
                    first_paragraph = dialogue._get_next_paragraph()
                    if first_paragraph:
                        # Отправляем текст
                        emit('speech_text', {
                            'text': f"Учитель: {first_paragraph}",
                            'sid': 'teacher',
                            'is_teacher': True
                        }, room=room_id)
                        # Озвучиваем первый абзац
                        speak_text(room_id, first_paragraph, voice_type='female', is_teacher=True)
            elif response:
                # Отправляем текст
                emit('speech_text', {
                    'text': f"Учитель: {response}",
                    'sid': 'teacher',
                    'is_teacher': True
                }, room=room_id)
                
                # Озвучиваем ответ (всегда!)
                speak_text(room_id, response, voice_type='female', is_teacher=True)

@socketio.on('activate_ai_teacher')
def handle_activate_ai_teacher(data):
    room_id = data['room_id']
    room_ai_activated[room_id] = True
    room_dialogue[room_id] = DialogueManager(socketio)
    room_dialogue[room_id].room_id = room_id
    
    # Устанавливаем режим LLM для нового диалог менеджера
    room_dialogue[room_id].set_llm_mode(room_llm_mode[room_id])
    
    greeting = "Привет! Я ваш AI-учитель. Давайте пообщаемся и выберем интересный урок вместе!"
    speak_text(room_id, greeting, voice_type='female', is_teacher=True)
    
    emit('ai_teacher_activated', {}, room=room_id)

@socketio.on('set_llm_mode')
def handle_set_llm_mode(data):
    room_id = data['room_id']
    mode = data['mode']
    
    if mode in ["traditional", "llm_first"]:
        room_llm_mode[room_id] = mode
        if room_id in room_dialogue:
            room_dialogue[room_id].set_llm_mode(mode)
        
        emit('llm_mode_changed', {
            'mode': mode,
            'room': room_id
        }, room=room_id)
        
        print(f"Режим LLM изменен в комнате {room_id}: {mode}")

@socketio.on('llm_response_ready')
def handle_llm_response_ready(data):
    """Обработчик готовых ответов от LLM (для асинхронной обработка)"""
    room_id = data['room_id']
    question = data['question']
    answer = data['answer']
    
    print(f"Получен ответ LLM для комнаты {room_id}: {answer[:100]}...")
    
    # Сбрасываем состояние речи учителя, чтобы гарантированно озвучить ответ
    reset_speaking_state(room_id, is_teacher=True)
    room_teacher_speaking[room_id] = False
    room_speaking[room_id] = False
    
    # Небольшая задержка для гарантии сброса состояния
    time.sleep(0.5)
    
    # Отправляем ответ в комнату
    emit('speech_text', {
        'text': f"Учитель: {answer}",
        'sid': 'teacher',
        'is_teacher': True
    }, room=room_id)
    
    # Озвучиваем ответ
    speak_text(room_id, answer, voice_type='female', is_teacher=True)

@socketio.on('practice_started')
def handle_practice_started(data):
    """Обработчик начала фазы практики"""
    room_id = data['room_id']
    room_practice_active[room_id] = True
    room_current_question_index[room_id] = 0
    emit('practice_started', {}, room=room_id)
    print(f"Практика начата в комнате {room_id}")

@socketio.on('practice_ended')
def handle_practice_ended(data):
    """Обработчик завершения фазы практики"""
    room_id = data['room_id']
    room_practice_active[room_id] = False
    room_current_question_index[room_id] = 0
    emit('practice_ended', {}, room=room_id)
    print(f"Практика завершена в комнате {room_id}")

@socketio.on('visualization_generated')
def handle_visualization_generated(data):
    """Обработчик готовых визуализаций"""
    room_id = data['room_id']
    
    print(f"🎨 Получена готовая визуализация для комнаты {room_id}: {data['topic'][:100]}...")
    
    # Пересылаем всем клиентам в комнате
    emit('visualization_generated', {
        'room_id': room_id,
        'topic': data['topic'],
        'mermaid_code': data.get('mermaid_code', ''),
        'svg_code': data.get('svg_code', ''),
        'timestamp': data.get('timestamp', time.time())
    }, room=room_id)

# НОВЫЕ ФУНКЦИИ ДЛЯ POLLING ВИЗУАЛИЗАЦИИ
def add_visualization_to_queue(room_id, topic, context):
    """Добавляет визуализацию в очередь для комнаты"""
    if room_id not in room_visualization_queue:
        room_visualization_queue[room_id] = []
    
    # Генерируем визуализации сразу
    mermaid_code = generate_mermaid_code(topic, context)
    svg_code = generate_svg_code(topic, context)
    
    visualization_data = {
        'topic': topic,
        'context': context,
        'mermaid_code': mermaid_code,
        'svg_code': svg_code,
        'timestamp': time.time()
    }
    
    # Ограничиваем размер очереди
    if len(room_visualization_queue[room_id]) >= 5:
        room_visualization_queue[room_id].pop(0)
    
    room_visualization_queue[room_id].append(visualization_data)
    
    print(f"📊 Визуализация добавлена в очередь для комнаты {room_id}: {topic}")
    return True

# Polling endpoint для визуализации
@app.route('/api/poll_visualization', methods=['POST', 'GET'])
def poll_visualization():
    """Endpoint для polling визуализаций"""
    try:
        if request.method == 'POST':
            data = request.json
            room_id = data.get('room_id', 'default')
        else:
            room_id = request.args.get('room_id', 'default')
        
        if room_id in room_visualization_queue and room_visualization_queue[room_id]:
            visualization = room_visualization_queue[room_id].pop(0)
            
            # Отправляем через WebSocket для немедленного отображения
            if socketio and room_id:
                socketio.emit('visualization_generated', {
                    'room_id': room_id,
                    'topic': visualization['topic'],
                    'mermaid_code': visualization.get('mermaid_code', ''),
                    'svg_code': visualization.get('svg_code', ''),
                    'timestamp': visualization['timestamp']
                }, room=room_id)
            
            return jsonify({
                "success": True,
                "processed": True,
                "queue_length": len(room_visualization_queue.get(room_id, []))
            })
        
        return jsonify({
            "success": True,
            "processed": False,
            "queue_length": len(room_visualization_queue.get(room_id, []))
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/visualization_queue_status', methods=['GET'])
def visualization_queue_status():
    """Статус очереди визуализаций"""
    room_id = request.args.get('room_id', 'default')
    
    return jsonify({
        "success": True,
        "room_id": room_id,
        "queue_length": len(room_visualization_queue.get(room_id, [])),
        "active": room_visualization_active.get(room_id, False),
        "queue": room_visualization_queue.get(room_id, [])
    })

# Новые эндпоинты для визуализации
@app.route('/api/generate_diagram', methods=['POST'])
def generate_diagram():
    """Генерация диаграммы через LLM + Mermaid"""
    try:
        data = request.json
        topic = data.get('topic', '')
        context = data.get('context', '')
        room_id = data.get('room_id', 'default')
        
        if not topic:
            return jsonify({"success": False, "error": "Topic is required"})
        
        # Добавляем в очередь вместо немедленной генерации
        add_visualization_to_queue(room_id, topic, context)
        
        return jsonify({
            "success": True,
            "message": "Визуализация добавлена в очередь",
            "queue_position": len(room_visualization_queue.get(room_id, [])),
            "topic": topic
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

def generate_mermaid_code(topic: str, context: str = "") -> str:
    """Генерация Mermaid кода через LLM"""
    print(f"🔧 Генерация Mermaid для: {topic[:100]}...")
    
    prompt = f"""
    Создай простую и понятную Mermaid.js диаграмму для объяснения темы: "{topic}".
    
    Контекст: {context}
    
    ТРЕБОВАНИЯ:
    1. Используй ТОЛЬКО корректный синтаксис Mermaid
    2. Максимум 8-10 элементов для наглядности
    3. Простые прямоугольники и стрелки
    4. Русские подписи в двойных кавычках
    5. Логическая структура от общего к частному
    
    ПРИМЕР КОРРЕКТНОГО СИНТАКСИСА:
    flowchart TD
        A["Общее понятие"] --> B["Частный случай 1"]
        A --> C["Частный случай 2"]
        B --> D["Пример"]
        C --> D

    Тема для диаграммы: {topic}
    
    Верни ТОЛЬКО код Mermaid без каких-либо пояснений.
    Начни сразу с объявления типа диаграммы.
    """
    
    try:
        # Используем существующий LLM
        from llm import LLMIntegration
        llm = LLMIntegration()
        
        response = llm._query_llm_api(
            prompt=prompt,
            context="",
            subject="general",
            system_prompt="""Ты - эксперт по созданию образовательных диаграмм. 
            Создавай ПРОСТЫЕ и ПОНЯТНЫЕ Mermaid диаграммы.
            ВАЖНО: Всегда используй корректный синтаксис Mermaid.""",
            max_tokens=500
        )
        
        if response:
            # Очищаем и проверяем синтаксис
            cleaned_code = clean_mermaid_code(response)
            print(f"✅ Сгенерирован Mermaid код для: {topic[:50]}...")
            print(f"   Код: {cleaned_code[:100]}...")
            return cleaned_code
        else:
            print(f"❌ LLM не вернул ответ для Mermaid")
        
    except Exception as e:
        print(f"❌ Ошибка генерации Mermaid кода: {e}")
    
    # Fallback - простая диаграмма по умолчанию
    return f'''flowchart TD
    A["{topic}"] --> B["Основной аспект 1"]
    A --> C["Основной аспект 2"]
    B --> D["Пример или свойство"]
    C --> D'''

def generate_svg_code(topic: str, context: str = "") -> str:
    """Генерация простого SVG через LLM"""
    print(f"🔧 Генерация SVG для: {topic[:100]}...")
    
    prompt = f"""
    Создай простой SVG код для визуализации: "{topic}".
    
    Контекст: {context}
    
    Используй только базовые элементы:
    - <rect> для прямоугольников и блоков
    - <circle> для кругов и узлов
    - <line> для линий и связей
    - <text> для текста и подписей
    - <path> для сложных форм
    
    Требования:
    - Размер: 400x300
    - Простая и понятная схема
    - Русские подписи
    - Минималистичный дизайн
    - Логическая структура
    - Цвета для различия элементов

    Верни ТОЛЬКО SVG код без пояснений.
    """
    
    try:
        from llm import LLMIntegration
        llm = LLMIntegration()
        
        svg_code = llm._query_llm_api(
            prompt=prompt,
            context="",
            subject="general",
            system_prompt="Ты создаешь простые SVG схемы для образования. Используй минималистичный дизайн и четкую структуру.",
            max_tokens=1000
        )
        
        if svg_code:
            # Очищаем SVG код
            svg_code = re.sub(r'```(xml|svg)?\s*', '', svg_code)
            svg_code = re.sub(r'```\s*', '', svg_code)
            svg_code = svg_code.strip()
            
            # Проверяем валидность SVG
            if svg_code.startswith('<svg') and svg_code.endswith('</svg>'):
                print(f"✅ Сгенерирован SVG код для: {topic[:50]}...")
                print(f"   Длина кода: {len(svg_code)} символов")
                return svg_code
        else:
            print(f"❌ LLM не вернул ответ для SVG")
        
    except Exception as e:
        print(f"❌ Ошибка генерации SVG кода: {e}")
    
    return ""

def clean_mermaid_code(code: str) -> str:
    """Очистка Mermaid кода от лишних символов и проверка синтаксиса"""
    if not code:
        return ""
    
    # Удаляем markdown обратные кавычки
    code = re.sub(r'```mermaid\s*', '', code)
    code = re.sub(r'```\s*', '', code)
    
    # Удаляем лишние пробелы и комментарии
    code = re.sub(r'%%.*', '', code)  # Удаляем комментарии Mermaid
    code = '\n'.join([line for line in code.split('\n') if line.strip()])
    
    # Проверяем базовый синтаксис Mermaid
    valid_starts = ['graph', 'flowchart', 'sequenceDiagram', 'classDiagram', 'stateDiagram', 'pie', 'gantt', 'gitGraph']
    if not any(code.strip().startswith(start) for start in valid_starts):
        # Если не начинается с правильного типа, добавляем flowchart по умолчанию
        code = 'flowchart TD\n' + code
    
    # Убедимся, что есть хотя бы одна стрелка или связь
    if '-->' not in code and '->' not in code and '--' not in code:
        # Добавляем простую структуру если нет связей
        lines = code.split('\n')
        if len(lines) > 1:
            code = lines[0] + '\n' + 'A["Элемент A"] --> B["Элемент B"]'
        else:
            code = 'flowchart TD\nA["Элемент A"] --> B["Элемент B"]'
    
    return code.strip()

@socketio.on('generate_visualization')
def handle_generate_visualization(data):
    """Обработчик запроса генерации визуализации - НЕМЕДЛЕННАЯ ГЕНЕРАЦИЯ"""
    room_id = data['room_id']
    topic = data.get('topic', '')
    context = data.get('context', '')
    
    if not topic:
        return
    
    print(f"🎨 WebSocket генерация визуализации для комнаты {room_id}: {topic[:100]}...")
    
    # Используем polling подход вместо прямой отправки через WebSocket
    add_visualization_to_queue(room_id, topic, context)
    
    # Отправляем подтверждение
    emit('visualization_queued', {
        'room_id': room_id,
        'topic': topic,
        'queue_length': len(room_visualization_queue.get(room_id, []))
    }, room=room_id)

# НОВЫЕ ЭНДПОИНТЫ ДЛЯ УПРАВЛЕНИЯ ЛОКАЛЬНОЙ LLM
@app.route('/api/llm/priority', methods=['POST'])
def set_llm_priority():
    """Установка приоритета моделей LLM"""
    try:
        data = request.json
        priority = data.get('priority')
        
        if not priority:
            return jsonify({"success": False, "error": "Priority not specified"})
        
        # Сохраняем в конфигурацию
        success = set_llm_priority(priority)
        
        if success:
            # Обновляем для всех активных комнат
            for room_id in room_dialogue:
                room_dialogue[room_id].llm.set_priority(priority)
            
            return jsonify({
                "success": True,
                "message": f"Приоритет успешно изменен на '{priority}'",
                "priority": priority
            })
        else:
            return jsonify({"success": False, "error": "Failed to save priority"})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/llm/priority', methods=['GET'])
def get_llm_priority():
    """Получение текущего приоритета моделей LLM"""
    try:
        priority = get_llm_priority()
        return jsonify({
            "success": True,
            "priority": priority
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/llm/priority_status')
def get_llm_priority_status():
    """Получение статуса приоритетов"""
    room_id = request.args.get('room_id', 'default')
    
    if room_id in room_dialogue:
        status = room_dialogue[room_id].llm.get_priority_status()
        return jsonify({
            "success": True,
            "room": room_id,
            "status": status
        })
    
    return jsonify({"success": False, "error": "Room not found"})

@app.route('/api/llm/available_priorities')
def get_available_priorities():
    """Получение доступных приоритетов"""
    return jsonify({
        "success": True,
        "priorities": [
            {
                "id": "local_first",
                "name": "Локальная модель в первую очередь",
                "description": "Сначала локальная модель, затем OpenRouter как fallback"
            },
            {
                "id": "openrouter_first", 
                "name": "OpenRouter в первую очередь",
                "description": "Сначала OpenRouter, затем локальная модель как fallback"
            },
            {
                "id": "local_only",
                "name": "Только локальная модель",
                "description": "Использовать только локальную модель"
            },
            {
                "id": "openrouter_only",
                "name": "Только OpenRouter", 
                "description": "Использовать только OpenRouter"
            }
        ]
    })

@app.route('/api/llm/status')
def get_llm_status():
    """Получение статуса LLM моделей"""
    room_id = request.args.get('room_id', 'default')
    
    if room_id in room_dialogue:
        status = room_dialogue[room_id].llm.get_llm_status()
        return jsonify({
            "success": True,
            "room": room_id,
            "status": status
        })
    
    return jsonify({"success": False, "error": "Room not found"})

@app.route('/api/llm/local_status')
def get_local_llm_status():
    """Получение статуса локальной модели"""
    try:
        local_llm = llm_manager.local_llm
        status = local_llm.get_status()
        
        return jsonify({
            "success": True,
            "status": status
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/api/llm_manager/status')
def get_llm_manager_status():
    """Получение статуса менеджера LLM"""
    try:
        status = llm_manager.get_status()
        return jsonify({
            "success": True,
            "status": status
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@socketio.on('get_llm_status')
def handle_get_llm_status(data):
    """WebSocket обработчик получения статуса LLM"""
    room_id = data['room_id']
    
    if room_id in room_dialogue:
        status = room_dialogue[room_id].llm.get_llm_status()
        emit('llm_status_update', {
            'room_id': room_id,
            'status': status
        }, room=room_id)

@socketio.on('set_llm_priority')
def handle_set_llm_priority(data):
    """WebSocket установка приоритета"""
    room_id = data['room_id']
    priority = data['priority']
    
    valid_priorities = ["local_first", "openrouter_first", "local_only", "openrouter_only"]
    
    if priority not in valid_priorities:
        emit('llm_priority_error', {
            'room_id': room_id,
            'error': f'Invalid priority. Use: {valid_priorities}'
        })
        return
    
    if room_id in room_dialogue:
        room_dialogue[room_id].llm.set_priority(priority)
        status = room_dialogue[room_id].llm.get_priority_status()
        
        emit('llm_priority_changed', {
            'room_id': room_id,
            'priority': priority,
            'status': status
        }, room=room_id)
        
        print(f"🔧 Приоритет LLM изменен в комнате {room_id}: {priority}")

@socketio.on('get_llm_priority_status')
def handle_get_llm_priority_status(data):
    """WebSocket получение статуса приоритета"""
    room_id = data['room_id']
    
    if room_id in room_dialogue:
        status = room_dialogue[room_id].llm.get_priority_status()
        emit('llm_priority_status', {
            'room_id': room_id,
            'status': status
        })

# АСИНХРОННЫЕ ЗАПРОСЫ К LLM
@socketio.on('async_llm_request')
def handle_async_llm_request(data):
    """Обработчик асинхронных запросов к LLM"""
    room_id = data['room_id']
    prompt = data['prompt']
    system_prompt = data.get('system_prompt', '')
    max_tokens = data.get('max_tokens', 1000)
    
    print(f"📨 [Async LLM] Запрос от комнаты {room_id}")
    
    # Отправляем запрос в менеджер (не блокируем основной поток)
    request_id = llm_manager.submit_request(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        room_id=room_id
    )
    
    # Немедленно подтверждаем получение запроса
    emit('llm_request_queued', {
        'request_id': request_id,
        'queue_position': llm_manager.get_queue_size(),
        'room_id': room_id
    })

@socketio.on('llm_async_response')
def handle_llm_async_response(data):
    """Обработчик асинхронных ответов от LLM"""
    room_id = data['room_id']
    response = data['response']
    request_id = data['request_id']
    
    print(f"🔧 [Async LLM] Ответ для комнаты {room_id}: {response[:100]}...")
    
    # Обрабатываем ответ
    if response and room_id in room_dialogue:
        # Передаем ответ в диалог менеджер
        room_dialogue[room_id].llm.handle_llm_response(request_id, response, room_id)
        
        # Отправляем ответ учителя
        emit('speech_text', {
            'text': f"Учитель: {response}",
            'sid': 'teacher',
            'is_teacher': True
        }, room=room_id)
        
        # Озвучиваем ответ
        speak_text(room_id, response, voice_type='female', is_teacher=True)

# POLLING ДЛЯ LLM ОТВЕТОВ (резервный механизм)
@app.route('/api/llm/poll_response', methods=['POST'])
def poll_llm_response():
    """Polling endpoint для получения ответов от LLM"""
    try:
        data = request.json
        room_id = data.get('room_id', 'default')
        last_check = data.get('last_check', 0)
        
        current_time = time.time()
        room_last_poll_time[room_id] = current_time
        
        # Проверяем новые ответы в очереди
        if room_id in room_llm_responses and room_llm_responses[room_id]:
            # Фильтруем ответы, которые пришли после last_check
            new_responses = [
                resp for resp in room_llm_responses[room_id] 
                if resp['timestamp'] > last_check
            ]
            
            if new_responses:
                # Берем самый свежий ответ
                latest_response = new_responses[-1]
                
                return jsonify({
                    "success": True,
                    "has_response": True,
                    "response": latest_response['response'],
                    "request_id": latest_response['request_id'],
                    "timestamp": latest_response['timestamp'],
                    "total_responses": len(new_responses)
                })
        
        return jsonify({
            "success": True, 
            "has_response": False,
            "timestamp": current_time,
            "queue_size": len(room_llm_responses.get(room_id, []))
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/llm/clear_queue', methods=['POST'])
def clear_llm_queue():
    """Очистка очереди ответов LLM"""
    try:
        data = request.json
        room_id = data.get('room_id', 'default')
        
        if room_id in room_llm_responses:
            room_llm_responses[room_id].clear()
            
        return jsonify({
            "success": True,
            "message": f"Очередь ответов для комнаты {room_id} очищена"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/llm/model', methods=['POST'])
def set_llm_model():
    """Установка модели LLM для комнаты"""
    try:
        data = request.json
        model = data.get('model')
        room_id = data.get('room_id', 'default')
        
        if not model:
            return jsonify({"success": False, "error": "Model not specified"})
        
        if room_id in room_dialogue:
            room_dialogue[room_id].set_llm_model(model)
            return jsonify({"success": True, "model": model, "room": room_id})
        
        return jsonify({"success": False, "error": "Room not found"})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/llm/models', methods=['GET'])
def get_llm_models():
    """Получение списка доступных моделей LLM"""
    models = [
        {"id": "llama", "name": "Llama 3.3 8B", "description": "Мощная и быстрая модель от Meta", "provider": "openrouter"},
        {"id": "llama3", "name": "Llama 3.3 8B Instruct", "description": "Инструктивная версия Llama 3.3", "provider": "openrouter"},
        {"id": "qwen", "name": "Qwen 2.5 32B", "description": "Качественная модель от Alibaba", "provider": "openrouter"},
        {"id": "qwen-turbo", "name": "Qwen Coder", "description": "Специализированная модель для программирования", "provider": "openrouter"},
        {"id": "local_llama", "name": "Llama 3.2 3B (локальная)", "description": "Локальная модель Llama 3.2", "provider": "local"}
    ]
    return jsonify({"models": models})

@app.route('/api/llm/status', methods=['GET'])
def get_llm_status_old():
    """Получение статуса LLM для комнаты"""
    room_id = request.args.get('room_id', 'default')
    
    if room_id in room_dialogue:
        stats = room_dialogue[room_id].llm.get_cache_stats()
        return jsonify({
            "success": True,
            "room": room_id,
            "cache_stats": stats,
            "model": room_dialogue[room_id].llm.model
        })
    
    return jsonify({"success": False, "error": "Room not found"})

@app.route('/api/config/llm_mode', methods=['GET'])
def get_llm_mode_api():
    """Получение текущего режима работы LLM"""
    try:
        config = load_config()
        return jsonify({
            "success": True,
            "mode": config.get("llm_query_mode", {}).get("default_mode", "traditional"),
            "available_modes": config.get("llm_query_mode", {}).get("available_modes", ["traditional", "llm_first"])
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/config/llm_mode', methods=['POST'])
def set_llm_mode_api():
    """Установка режима работы LLM"""
    try:
        data = request.json
        mode = data.get('mode')
        
        if not mode:
            return jsonify({"success": False, "error": "Mode not specified"})
        
        if mode not in ["traditional", "llm_first"]:
            return jsonify({"success": False, "error": "Invalid mode. Use 'traditional' or 'llm_first'"})
        
        success = set_llm_mode(mode)
        
        if success:
            # Обновляем режим для всех активных комнат
            for room_id in room_llm_mode:
                room_llm_mode[room_id] = mode
                if room_id in room_dialogue:
                    room_dialogue[room_id].set_llm_mode(mode)
            
            return jsonify({
                "success": True,
                "message": f"Режим LLM успешно изменен на '{mode}'",
                "mode": mode
            })
        else:
            return jsonify({"success": False, "error": "Failed to save config"})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/knowledge/stats', methods=['GET'])
def get_knowledge_stats():
    """Получение статистики базы знаний для комнаты"""
    room_id = request.args.get('room_id', 'default')
    subject = request.args.get('subject', '')
    
    if room_id in room_dialogue:
        stats = room_dialogue[room_id].get_knowledge_stats()
        if stats:
            return jsonify({
                "success": True,
                "room": room_id,
                "subject": subject or stats.get("subject", "unknown"),
                "stats": stats
            })
    
    return jsonify({"success": False, "error": "Room not found"})

@app.route('/api/knowledge/search', methods=['GET'])
def search_knowledge():
    """Поиск в базе знаний"""
    room_id = request.args.get('room_id', 'default')
    query = request.args.get('query', '')
    max_results = int(request.args.get('max_results', 5))
    
    if not query:
        return jsonify({"success": False, "error": "Query parameter is required"})
    
    if room_id in room_dialogue and room_dialogue[room_id].knowledge_base:
        results = room_dialogue[room_id].knowledge_base.search_similar(query, max_results)
        return jsonify({
            "success": True,
            "room": room_id,
            "query": query,
            "results": results,
            "total_found": len(results)
        })
    
    return jsonify({"success": False, "error": "Room not found or no knowledge base"})

@app.route('/api/knowledge/llm_answers', methods=['GET'])
def get_llm_answers():
    """Получение списка ответов LLM для предмета"""
    room_id = request.args.get('room_id', 'default')
    subject = request.args.get('subject', '')
    
    if room_id in room_dialogue and room_dialogue[room_id].knowledge_base:
        answers = room_dialogue[room_id].knowledge_base.list_llm_answers()
        return jsonify({
            "success": True,
            "room": room_id,
            "subject": subject,
            "answers": answers,
            "total_answers": len(answers)
        })
    
    return jsonify({"success": False, "error": "Room not found or no knowledge base"})

@app.route('/api/lesson_content/<lesson_id>')
def get_lesson_content(lesson_id):
    """Получение содержания урока"""
    try:
        lesson_file = LESSONS_DIR / f"{lesson_id}.txt"
        if not lesson_file.exists():
            return jsonify({"error": "Lesson not found"}), 404
        
        with open(lesson_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Разбиваем на абзацы (улучшенная логика)
        paragraphs = []
        current_paragraph = []
        
        # Сначала разбиваем по двойным переводам строк
        if '\n\n' in content:
            raw_paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        else:
            # Если нет двойных переводов строк, разбиваем по одиночным
            raw_paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
        
        # Объединяем короткие абзацы в группы по 6 предложений
        for paragraph in raw_paragraphs:
            # Разбиваем на предложences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Если абзац уже содержит достаточно предложений, добавляем как есть
            if len(sentences) >= 6:
                paragraphs.append(' '.join(sentences))
                continue
                
            # Добавляем предложения в текущий абзац
            current_paragraph.extend(sentences)
            
            # Если накопилось достаточно предложений, создаем новый абзац
            if len(current_paragraph) >= 6:
                paragraphs.append(' '.join(current_paragraph[:6]))
                current_paragraph = current_paragraph[6:]
        
        # Добавляем оставшиеся предложения
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # Убираем \n\n из текста
        paragraphs = [p.replace('\n\n', ' ').replace('\n', ' ') for p in paragraphs]
        
        return jsonify({
            "success": True,
            "lesson_id": lesson_id,
            "content": paragraphs,
            "paragraph_count": len(paragraphs)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/lessons')
def get_available_lessons():
    """Получение списка доступных уроков"""
    try:
        lessons = {}
        for lesson_file in LESSONS_DIR.glob("*.txt"):
            subject = _detect_subject(lesson_file.stem)
            
            if subject not in lessons:
                lessons[subject] = []
            
            lessons[subject].append({
                'id': lesson_file.stem,
                'title': lesson_file.stem.replace('_', ' ').title(),
                'file_path': lesson_file.name,
                'type': 'text'
            })
        
        return jsonify({
            "success": True,
            "lessons": lessons
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/practice_content/<lesson_id>')
def get_practice_content(lesson_id):
    """Получение практических заданий для урока"""
    try:
        practice_file = PRACTICE_DIR / f"{lesson_id}.json"
        if not practice_file.exists():
            return jsonify({"error": "Практические задания не найдены", "success": False}), 404
        
        with open(practice_file, 'r', encoding='utf-8') as f:
            content = json.load(f)
        
        return jsonify({
            "success": True,
            'lesson_id': lesson_id,
            'content': content,
            'question_count': len(content.get('questions', []))
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/api/practice_files')
def get_practice_files():
    """Получение списка файлов практики"""
    try:
        practice_files = []
        for practice_file in PRACTICE_DIR.glob("*.json"):
            practice_files.append({
                'filename': practice_file.name,
                'size': practice_file.stat().st_size,
                'modified': datetime.fromtimestamp(practice_file.stat().st_mtime).isoformat()
            })
        
        return jsonify({
            "success": True,
            "files": practice_files
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/practice_txt_files')
def get_practice_txt_files():
    """Получение списка TXT файлов практики"""
    try:
        txt_files = []
        for txt_file in PRACTICE_DIR.glob("*.txt"):
            txt_files.append({
                'filename': txt_file.name,
                'size': txt_file.stat().st_size,
                'modified': datetime.fromtimestamp(txt_file.stat().st_mtime).isoformat()
            })
        
        return jsonify({
            "success": True,
            "files": txt_files,
            "total_files": len(txt_files)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload_practice', methods=['POST'])
def upload_practice():
    """Загрузка файла практики"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"})
        
        if file and file.filename.endswith('.json'):
            filename = secure_filename(file.filename)
            file.save(PRACTICE_DIR / filename)
            
            return jsonify({
                "success": True,
                "message": f"File {filename} uploaded successfully",
                "filename": filename
            })
        else:
            return jsonify({"success": False, "error": "Invalid file type. Only JSON allowed"})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/delete_practice/<filename>')
def delete_practice(filename):
    """Удаление файла практики"""
    try:
        practice_file = PRACTICE_DIR / filename
        if not practice_file.exists():
            return jsonify({"success": False, "error": "File not found"})
        
        practice_file.unlink()
        return jsonify({
            "success": True,
            "message": f"File {filename} deleted successfully"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

def _detect_subject(filename: str) -> str:
    """Определяет предмет по названию файла"""
    filename_lower = filename.lower()
    if any(word in filename_lower for word in ['math', 'математика', 'алгебра', 'геометрия']):
        return "математика"
    elif any(word in filename_lower for word in ['history', 'история', 'истор']):
        return "история"
    elif any(word in filename_lower for word in ['physics', 'физика', 'физ']):
        return "физика"
    elif any(word in filename_lower for word in ['chemistry', 'химия', 'хим']):
        return "химия"
    elif any(word in filename_lower for word in ['social', 'обществознание', 'общество']):
        return "обществознание"
    elif any(word in filename_lower for word in ['biology', 'биология', 'био']):
        return "биология"
    elif any(word in filename_lower for word in ['literature', 'литература', 'лит']):
        return "литература"
    elif any(word in filename_lower for word in ['russian', 'русский', 'язык']):
        return "русский язык"
    else:
        return "общее"

@app.route('/api/add_knowledge', methods=['POST'])
def add_knowledge():
    """Добавление знаний в базу"""
    try:
        data = request.json
        subject = data.get('subject', 'общее')
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({"success": False, "error": "Text is required"})
        
        # Создаем базу знаний для предмета если ее нет
        knowledge_file = MATERIALS_DIR / f"{subject}_knowledge.json"
        if knowledge_file.exists():
            with open(knowledge_file, 'r', encoding='utf-8') as f:
                knowledge_data = json.load(f)
        else:
            knowledge_data = {
                "terms": {},
                "questions": {},
                "examples": {},
                "metadata": {
                    "subject": subject,
                    "version": "1.0",
                    "last_updated": datetime.now().isoformat(),
                    "author": "AI Teacher System"
                }
            }
        
        # Парсим текст и добавляем в базу знаний
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines:
            if ' - ' in line:
                term, definition = line.split(' - ', 1)
                knowledge_data["terms"][term.strip().lower()] = definition.strip()
            elif line.endswith('?'):
                knowledge_data["questions"][line.strip().lower()] = "Ответ будет добавлен автоматически"
            else:
                # Просто добавляем как общую информацию
                if "general_info" not in knowledge_data:
                    knowledge_data["general_info"] = []
                knowledge_data["general_info"].append(line.strip())
        
        # Сохраняем обновленную базу знаний
        with open(knowledge_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_data, f, ensure_ascii=False, indent=2)
        
        return jsonify({"success": True, "subject": subject, "added_items": len(lines)})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/add_lesson', methods=['POST'])
def add_lesson():
    """Добавление нового урока"""
    try:
        data = request.json
        subject = data.get('subject', 'общее')
        title = data.get('title', '')
        content = data.get('content', '')
        
        if not title or not content:
            return jsonify({"success": False, "error": "Title and content are required"})
        
        # Создаем имя файла
        filename = f"{subject}_{title.lower().replace(' ', '_')}.txt"
        lesson_path = LESSONS_DIR / filename
        
        with open(lesson_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return jsonify({"success": True, "filename": filename, "subject": subject, "title": title})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/add_practice', methods=['POST'])
def add_practice():
    """Добавление практических заданий"""
    try:
        data = request.json
        lesson_id = data.get('lesson_id', '')
        practice_data = data.get('practice_data', {})
        
        if not lesson_id or not practice_data:
            return jsonify({"success": False, "error": "Lesson ID and practice data are required"})
        
        # Создаем файл практики
        practice_file = PRACTICE_DIR / f"{lesson_id}.json"
        
        with open(practice_file, 'w', encoding='utf-8') as f:
            json.dump(practice_data, f, ensure_ascii=False, indent=2)
        
        return jsonify({"success": True, "lesson_id": lesson_id, "question_count": len(practice_data.get('questions', []))})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/download_knowledge')
def download_knowledge():
    """Скачивание базы знаний"""
    subject = request.args.get('subject', 'обществознание')
    knowledge_file = MATERIALS_DIR / f"{subject}_knowledge.json"
    llm_answers_file = MATERIALS_DIR / f"{subject}_llm_answers.json"
    
    if not knowledge_file.exists() and not llm_answers_file.exists():
        return jsonify({"success": False, "error": f"База знаний для предмета '{subject}' не найдена"})
    
    # Создаем временный файл для скачивания
    import tempfile
    import zipfile
    
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    
    with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
        if knowledge_file.exists():
            zipf.write(knowledge_file, f"{subject}_knowledge.json")
        if llm_answers_file.exists():
            zipf.write(llm_answers_file, f"{subject}_llm_answers.json")
    
    temp_zip.close()
    
    return send_file(
        temp_zip.name,
        as_attachment=True,
        download_name=f"{subject}_knowledge_base.zip",
        mimetype='application/zip'
    )

@app.route('/api/download_lessons')
def download_lessons():
    """Скачивание всех уроков"""
    if not any(LESSONS_DIR.iterdir()):
        return jsonify({"success": False, "error": "Уроки не найдены"})
    
    # Создаем временный zip-файл
    import tempfile
    import zipfile
    
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    
    with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
        for lesson_file in LESSONS_DIR.glob("*.txt"):
            zipf.write(lesson_file, lesson_file.name)
    
    temp_zip.close()
    
    return send_file(
        temp_zip.name,
        as_attachment=True,
        download_name="ai_teacher_lessons.zip",
        mimetype='application/zip'
    )

@app.route('/api/download_practice')
def download_practice():
    """Скачивание всех практических заданий"""
    if not any(PRACTICE_DIR.iterdir()):
        return jsonify({"success": False, "error": "Практические задания не найдены"})
    
    # Создаем временный zip-файл
    import tempfile
    import zipfile
    
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    
    with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
        for practice_file in PRACTICE_DIR.glob("*.json"):
            zipf.write(practice_file, practice_file.name)
    
    temp_zip.close()
    
    return send_file(
        temp_zip.name,
        as_attachment=True,
        download_name="ai_teacher_practice.zip",
        mimetype='application/zip'
    )

@app.route('/api/download_practice_txt')
def download_practice_txt():
    """Скачивание практических заданий в текстовом формате"""
    try:
        # Ищем TXT файлы практики
        practice_txt_files = list(PRACTICE_DIR.glob("*.txt"))
        
        if not practice_txt_files:
            return jsonify({"success": False, "error": "TXT файлы практики не найдены"})
        
        # Создаем временный zip-файл
        import tempfile
        import zipfile
    
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        
        with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
            for txt_file in practice_txt_files:
                zipf.write(txt_file, txt_file.name)
        
        temp_zip.close()
        
        return send_file(
            temp_zip.name,
            as_attachment=True,
            download_name="ai_teacher_practice_txt.zip",
            mimetype='application/zip'
        )
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# Новые API эндпоинты для управления API ключами
@app.route('/api/config/keys', methods=['GET'])
def get_api_keys():
    """Получение текущих API ключей"""
    try:
        config = load_config()
        return jsonify({
            "success": True,
            "keys": {
                "openrouter": config.get("openrouter", {}).get("api_key", ""),
                "llm": config.get("llm", {}).get("api_key", "")
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/config/keys', methods=['POST'])
def set_api_key():
    """Установка API ключа"""
    try:
        data = request.json
        provider = data.get('provider')
        api_key = data.get('api_key')
        
        if not provider or not api_key:
            return jsonify({"success": False, "error": "Provider and API key are required"})
        
        if provider not in ['openrouter', 'llm']:
            return jsonify({"success": False, "error": "Invalid provider. Use 'openrouter' or 'llm'"})
        
        success = update_api_key(provider, api_key)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"API ключ для {provider} успешно обновлен",
                "provider": provider
            })
        else:
            return jsonify({"success": False, "error": "Failed to update API key"})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/config/test', methods=['POST'])
def test_api_key():
    """Тестирование API ключа"""
    try:
        data = request.json
        provider = data.get('provider')
        api_key = data.get('api_key')
        
        if not provider or not api_key:
            return jsonify({"success": False, "error": "Provider and API key are required"})
        
        # Тестируем ключ через простой запрос к OpenRouter
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-site.com",
            "X-Title": "AI Teacher"
        }
        
        test_data = {
            "model": "meta-llama/llama-3.3-8b-instruct:free",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 10
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            return jsonify({
                "success": True,
                "message": f"Ключ {provider} работает корректно",
                "valid": True
            })
        elif response.status_code == 401:
            return jsonify({
                "success": True,
                "message": f"Ключ {provider} неверный или неактивный",
                "valid": False
            })
        else:
            return jsonify({
                "success": True,
                "message": f"Ключ {provider} может быть неверным (код: {response.status_code})",
                "valid": False
            })
            
    except Exception as e:
        return jsonify({
            "success": True,
            "message": f"Ошибка проверки ключа: {str(e)}",
            "valid": False
        })

# Эндпоинт для принудительной генерации визуализации для тестирования
@app.route('/api/force_visualization', methods=['POST'])
def force_visualization():
    """Принудительная генерация визуализации для тестирования"""
    try:
        data = request.json
        room_id = data.get('room_id', 'default')
        topic = data.get('topic', 'Тестовая визуализация')
        context = data.get('context', 'Тестовый контекст')
        
        print(f"🔧 Принудительная генерация визуализации для комнаты {room_id}")
        
        # Добавляем в очередь
        add_visualization_to_queue(room_id, topic, context)
        
        return jsonify({
            "success": True,
            "message": f"Визуализация добавлена в очередь для комнаты {room_id}",
            "topic": topic,
            "queue_length": len(room_visualization_queue.get(room_id, []))
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# Эндпоинт для проверки здоровья системы
@app.route('/api/health')
def health_check():
    """Проверка здоровья системы"""
    try:
        # Проверяем доступность локальной модели
        local_status = llm_manager.local_llm.get_status()
        
        # Проверяем доступность OpenRouter
        openrouter_available = bool(get_api_key('openrouter'))
        
        # Проверяем доступность уроков
        lessons_available = any(LESSONS_DIR.iterdir())
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "local_llm": local_status,
                "openrouter": {"available": openrouter_available},
                "lessons": {"available": lessons_available, "count": len(list(LESSONS_DIR.glob("*.txt")))},
                "practice": {"available": any(PRACTICE_DIR.iterdir()), "count": len(list(PRACTICE_DIR.glob("*.json")))},
                "llm_manager": llm_manager.get_status()
            }
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    print("🚀 Запуск AI Teacher системы...")
    
    # Настраиваем менеджер LLM
    setup_llm_manager()
    
    print("🔧 Проверка конфигурации...")
    
    # Проверяем доступность локальной модели
    local_status = llm_manager.local_llm.get_status()
    print(f"🔧 Статус локальной модели: {local_status}")
    
    # Проверяем доступность OpenRouter
    openrouter_key = get_api_key('openrouter')
    print(f"🔧 OpenRouter API ключ: {'Установлен' if openrouter_key else 'Не установлен'}")
    
    # Проверяем уроки
    lessons_count = len(list(LESSONS_DIR.glob("*.txt")))
    print(f"📚 Доступно уроков: {lessons_count}")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)