from flask import Flask, render_template, send_from_directory, jsonify, request
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

app = Flask(__name__, static_folder='static')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

BASE_DIR = Path(__file__).parent
FRAMES_DIR = BASE_DIR / 'static' / 'avatar' / 'frames'

# Глобальные состояния
animation_running = defaultdict(bool)
room_participants = defaultdict(set)
room_speech_data = defaultdict(list)
room_speaking = defaultdict(bool)
room_ai_activated = defaultdict(bool)
room_dialogue = defaultdict(lambda: DialogueManager(socketio))
room_lessons = defaultdict(dict)

# Соответствие букв кадрам анимации рта
PHONEME_MAP = {
    'а': 'mouth_aa', 'о': 'mouth_oo', 'у': 'mouth_uu',
    'и': 'mouth_ee', 'э': 'mouth_ee', 'ы': 'mouth_aa',
    'е': 'mouth_ee', 'ё': 'mouth_oo', 'ю': 'mouth_uu',
    'я': 'mouth_aa', 'м': 'mouth_mm', 'п': 'mouth_pp',
    'б': 'mouth_bb', 'ф': 'mouth_ff', 'в': 'mouth_vv',
    'ш': 'mouth_sh', 'ж': 'mouth_zh', 'с': 'mouth_ss',
    'з': 'mouth_zz', 'р': 'mouth_rr', 'л': 'mouth_ll',
    'н': 'mouth_nn', 'т': 'mouth_tt', 'д': 'mouth_dd',
    'к': 'mouth_kk', 'г': 'mouth_gg', 'х': 'mouth_hh',
    'ч': 'mouth_ch', 'щ': 'mouth_sh', 'ц': 'mouth_ss',
    'й': 'mouth_ee'
}

def reset_speaking_state(room_id):
    """Сбрасывает состояние речи для указанной комнаты"""
    room_speaking[room_id] = False
    socketio.emit('speaking_state', {'speaking': False}, room=room_id)

def speak_text(room_id, text, voice_type='female', is_teacher=False, skip_history=False):
    """Озвучивает текст с анимацией и добавляет его в историю"""
    if not text.strip():
        return
        
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
    
    speech_duration = max(2, len(text) * 0.1)
    threading.Timer(speech_duration, lambda: reset_speaking_state(room_id)).start()

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
        
        frames = [f for f in os.listdir(avatar_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
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

def get_neutral_frames(avatar_name):
    """Возвращает список нейтральных кадров для аватара"""
    return sorted([f for f in os.listdir(FRAMES_DIR / avatar_name) 
                  if f.startswith('mouth_neutral_')])

def get_blink_frames(avatar_name):
    """Возвращает список кадров моргания для аватара"""
    return sorted([f for f in os.listdir(FRAMES_DIR / avatar_name) 
                  if f.startswith('blink_')])

def get_speech_frames(avatar_name, phoneme):
    """Возвращает список речевых кадров для указанной фонемы"""
    base_name = PHONEME_MAP.get(phoneme, 'mouth_aa')
    return [f for f in os.listdir(FRAMES_DIR / avatar_name) 
            if f.startswith(base_name)]

def animation_loop(room_id, avatar_name):
    """Основной цикл анимации для комната"""
    blink_counter = 0
    blink_frames = get_blink_frames(avatar_name)
    neutral_frames = get_neutral_frames(avatar_name)
    
    while animation_running[room_id]:
        if room_speaking[room_id]:
            current_char = random.choice(list(PHONEME_MAP.keys()))
            speech_frames = get_speech_frames(avatar_name, current_char)
            if speech_frames:
                frame = random.choice(speech_frames)
                frame_path = f'/frames/{avatar_name}/{frame}'
                socketio.emit('animation_frame', {'frame': frame_path}, room=room_id)
        else:
            blink_counter += 1
            if blink_counter >= 30 and blink_frames:
                for frame in blink_frames:
                    frame_path = f'/frames/{avatar_name}/{frame}'
                    socketio.emit('animation_frame', {'frame': frame_path}, room=room_id)
                    time.sleep(0.1)
                blink_counter = 0
            elif neutral_frames:
                frame = random.choice(neutral_frames)
                frame_path = f'/frames/{avatar_name}/{frame}'
                socketio.emit('animation_frame', {'frame': frame_path}, room=room_id)
        
        time.sleep(0.1)

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
    
    if len(room_participants[room_id]) == 1:
        greeting = "Привет! Я ваш виртуальный учитель. Давайте познакомимся и выберем интересный урок вместе!"
        speak_text(room_id, greeting, voice_type='female', is_teacher=True)
    
    emit('participants_update', {'count': len(room_participants[room_id])}, room=room_id)
    emit('new_participant', {'sid': request.sid}, room=room_id)
    
    if len(room_participants[room_id]) == 2 and not room_ai_activated[room_id]:
        welcome_text = "Учитель с искусственным интеллектом активирован"
        speak_text(room_id, welcome_text, voice_type='female', is_teacher=True)
        emit('ai_teacher_available', {}, room=room_id)
    
    if room_speech_data[room_id]:
        emit('speech_history', {'history': room_speech_data[room_id]}, to=request.sid)

@socketio.on('start_animation')
def handle_start_animation(data):
    room_id = data['room_id']
    avatar_name = data['avatar_name']
    if not animation_running[room_id]:
        animation_running[room_id] = True
        threading.Thread(target=animation_loop, args=(room_id, avatar_name)).start()

@socketio.on('stop_animation')
def handle_stop_animation(data):
    room_id = data['room_id']
    animation_running[room_id] = False

@socketio.on('generate_speech')
def handle_generate_speech(data):
    room_id = data['room_id']
    text = data['text']
    voice_type = data.get('voice', 'male')
    speak_text(room_id, text, voice_type)

@socketio.on('recognized_speech')
def handle_recognized_speech(data):
    room_id = data['room_id']
    text = data['text']
    user_sid = request.sid
    
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
            if any(word in text.lower() for word in ["записал", "дальше", "продолжай", "следующий", "продолжить"]):
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
                # ОЗВУЧИВАЕМ ответ на вопрос
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
                
                # Озвучиваем ответ
                speak_text(room_id, response, voice_type='female', is_teacher=True)

@socketio.on('activate_ai_teacher')
def handle_activate_ai_teacher(data):
    room_id = data['room_id']
    room_ai_activated[room_id] = True
    room_dialogue[room_id] = DialogueManager(socketio)
    
    greeting = "Привет! Я ваш AI-учитель. Давайте пообщаемся и выберем интересный урок вместе!"
    speak_text(room_id, greeting, voice_type='female', is_teacher=True)
    
    emit('ai_teacher_activated', {}, room=room_id)

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
        {"id": "deepseek", "name": "DeepSeek R1", "description": "Мощная модель для образовательных целей"},
        {"id": "qwen", "name": "Qwen 2.5 32B", "description": "Качественная модель от Alibaba"},
        {"id": "qwen-turbo", "name": "Qwen Coder", "description": "Специализированная модель для программирования"}
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
    
    return jsonify({"success": False, "error": "Room not found or no knowledge base"})

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
        lesson_file = Path("lessons") / f"{lesson_id}.txt"
        if not lesson_file.exists():
            return jsonify({"error": "Lesson not found"}), 404
        
        with open(lesson_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Разбиваем на абзацы
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        return jsonify({
            "success": True,
            "lesson_id": lesson_id,
            "content": paragraphs,
            "paragraph_count": len(paragraphs)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/add_knowledge', methods=['POST'])
def add_knowledge():
    """Добавление знаний в базу"""
    try:
        data = request.json
        subject = data.get('subject', 'общее')
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({"success": False, "error": "Text is required"})
        
        # Здесь должна быть логика добавления в базу знаний
        # Пока просто возвращаем успех
        print(f"Добавление знаний по {subject}: {text[:100]}...")
        
        return jsonify({"success": True, "subject": subject})
        
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
        lesson_path = Path("lessons") / filename
        
        with open(lesson_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Добавлен урок: {title} по {subject}")
        
        return jsonify({"success": True, "filename": filename})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/download_knowledge')
def download_knowledge():
    """Скачивание базы знаний"""
    subject = request.args.get('subject', 'обществознание')
    # Здесь должна быть логика формирования файла базы знаний
    # Пока возвращаем заглушку
    return jsonify({"success": False, "error": "Not implemented yet"})

if __name__ == '__main__':
    # Создаем необходимые папки
    for folder in ['lessons', 'cache', 'materials']:
        os.makedirs(folder, exist_ok=True)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
