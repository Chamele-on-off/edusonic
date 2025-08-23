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
    'ш': 'mouth_sh', 'ж': 'mouth_zh', 'س': 'mouth_ss',
    'з': 'mouth_zz', 'р': 'mouth_rr', 'л': 'mouth_ll',
    'н': 'mouth_nn', 'т': 'mouth_tt', 'д': 'mouth_dd',
    'к': 'mouth_kk', 'г': 'mouth_gg', 'х': 'mouth_hh',
    'ч': 'mouth_ch', 'щ': 'mouth_sh', 'ц': 'mouth_ss',
    'й': 'mouth_ee'
}

def reset_speaking_state(room_id):
    room_speaking[room_id] = False
    socketio.emit('speaking_state', {'speaking': False}, room=room_id)

def speak_text(room_id, text, voice_type='female', is_teacher=False):
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

def start_lesson(room_id, lesson_data):
    """Запускает урок с передачей данных"""
    print(f"Запуск урока в комнате {room_id}: {lesson_data['title']}")
    
    room_lessons[room_id] = {
        'lesson_id': lesson_data['id'],
        'title': lesson_data['title'],
        'subject': lesson_data.get('subject', ''),
        'phases': lesson_data.get('phases', []),
        'current_phase': 0,
        'status': 'active',
        'start_time': time.time()
    }
    
    emit('lesson_started', {
        'lesson_id': lesson_data['id'],
        'title': lesson_data['title'],
        'subject': lesson_data.get('subject', '')
    }, room=room_id)
    
    # Запускаем выполнение урока в отдельном потоке
    threading.Thread(target=run_lesson_phases, args=(room_id,), daemon=True).start()

def run_lesson_phases(room_id):
    """Выполняет фазы урока"""
    if room_id not in room_lessons:
        print(f"Ошибка: урок не найден в комнате {room_id}")
        return
        
    lesson = room_lessons[room_id]
    phases = lesson['phases']
    
    print(f"Начало выполнения {len(phases)} фаз урока в комнате {room_id}")
    
    for phase_index, phase in enumerate(phases):
        if room_lessons[room_id]['status'] != 'active':
            print(f"Урок прерван в фазе {phase_index}")
            break
            
        # Обновляем текущую фазу
        room_lessons[room_id]['current_phase'] = phase_index
        
        # Отправляем информацию о фазе
        emit('lesson_phase', {
            'phase_index': phase_index,
            'total_phases': len(phases),
            'type': phase.get('type', 'explanation'),
            'content': phase.get('content', ''),
            'duration': phase.get('duration', 60)
        }, room=room_id)
        
        print(f"Фаза {phase_index}: {phase.get('type', 'explanation')}")
        
        # Озвучиваем содержание фазы
        speak_text(room_id, phase.get('content', ''), is_teacher=True)
        
        # Ждем продолжительность фазы
        phase_duration = phase.get('duration', 60)
        print(f"Ожидание фазы {phase_index}: {phase_duration} секунд")
        time.sleep(phase_duration)
    
    # Завершение урока
    if room_id in room_lessons:
        room_lessons[room_id]['status'] = 'completed'
        print(f"Урок завершен в комнате {room_id}")
        speak_text(room_id, "Урок завершен! Отлично поработали!", is_teacher=True)

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
    return sorted([f for f in os.listdir(FRAMES_DIR / avatar_name) 
                  if f.startswith('mouth_neutral_')])

def get_blink_frames(avatar_name):
    return sorted([f for f in os.listdir(FRAMES_DIR / avatar_name) 
                  if f.startswith('blink_')])

def get_speech_frames(avatar_name, phoneme):
    base_name = PHONEME_MAP.get(phoneme, 'mouth_aa')
    return [f for f in os.listdir(FRAMES_DIR / avatar_name) 
            if f.startswith(base_name)]

def animation_loop(room_id, avatar_name):
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
    
    # Приветствуем только первого участника
    if len(room_participants[room_id]) == 1:
        greeting = "Привет! Я твой виртуальный учитель. Просто скажи название предмета, например 'математика' или 'обществознание', чтобы начать урок."
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
        
        # Отладочная информация
        print(f"Текущее состояние диалога: {dialogue.get_current_state()}")
        print(f"Урок начат в диалоге: {dialogue.is_lesson_started()}")
        print(f"Урок активен в комнате: {room_id in room_lessons and room_lessons[room_id]['status'] == 'active'}")
        
        # Проверяем, активен ли урок в комнате
        lesson_active = room_id in room_lessons and room_lessons[room_id]['status'] == 'active'
        
        response = None
        
        if lesson_active:
            print("Обработка вопроса во время урока")
            response = dialogue.handle_question_during_lesson(text)
        else:
            print("Обработка диалога выбора урока")
            response = dialogue.process_input(text)
            
            # Проверяем специальный сигнал начала урока
            if response == "LESSON_START_SIGNAL":
                print("Получен сигнал начала урока")
                lesson_data = dialogue.get_selected_lesson()
                if lesson_data and room_id not in room_lessons:
                    print(f"Запуск урока: {lesson_data['title']}")
                    start_lesson(room_id, lesson_data)
                    # Не отправляем сигнал как обычное сообщение
                    return
                
            # Проверяем, был ли выбран и подтвержден урок ПОСЛЕ обработки ввода
            lesson_ready_after = dialogue.is_lesson_ready_to_start()
            lesson_data_after = dialogue.get_selected_lesson() if lesson_ready_after else None
            
            # Запускаем урок сразу, если он готов
            if lesson_data_after and room_id not in room_lessons:
                print(f"Запуск урока после обработки: {lesson_data_after['title']}")
                start_lesson(room_id, lesson_data_after)
        
        if response and response != "LESSON_START_SIGNAL":
            print(f"Ответ учителя: {response}")
            emit('speech_text', {
                'text': f"Учитель: {response}",
                'sid': 'teacher',
                'is_teacher': True
            }, room=room_id)
            
            speak_text(room_id, response, voice_type='female', is_teacher=True)

@socketio.on('activate_ai_teacher')
def handle_activate_ai_teacher(data):
    room_id = data['room_id']
    room_ai_activated[room_id] = True
    room_dialogue[room_id] = DialogueManager(socketio)
    
    greeting = "Привет! Я ваш AI-учитель. Просто скажите название предмета, чтобы начать урок."
    speak_text(room_id, greeting, voice_type='female', is_teacher=True)
    
    emit('ai_teacher_activated', {}, room=room_id)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)