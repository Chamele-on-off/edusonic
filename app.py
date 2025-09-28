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
from config import update_api_key, get_api_key, load_config, get_model_config, get_llm_mode, set_llm_mode
import requests
import json
from datetime import datetime
from werkzeug.utils import secure_filename
import re

app = Flask(__name__, static_folder='static')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

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
room_current_avatar = defaultdict(lambda: 'teacher')  # Храним текущий аватар для каждой комнаты

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
    print('Client connected:', request.sid)

@socketio.on('disconnect')
def handle_disconnect():
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
        
        # Если урок уже начат, обрабатываем как вопрос/команду
        if dialogue.is_lesson_started():
            # Сначала проверяем команды управления
            if any(word in text.lower() for word in ["записал", "дальше", 'продолжай', "следующий", "продолжить"]):
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
    """Обработчик готовых ответов от LLM (для асинхронной обработки)"""
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
        {"id": "qwen-turbo", "name": "Qwen Coder", "description": "Специализированная модель для программирования", "provider": "openrouter"}
    ]
    return jsonify({"models": models})

@app.route('/api/llm/status', methods=['GET'])
def get_llm_status():
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

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
