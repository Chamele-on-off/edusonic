from flask import Flask, render_template, send_from_directory, jsonify, request
from flask_socketio import SocketIO, emit, join_room, leave_room
import os
from pathlib import Path
from gtts import gTTS
import io
import base64
import time
import uuid

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Конфигурация путей
BASE_DIR = Path(__file__).parent
FRAMES_DIR = BASE_DIR / 'static' / 'avatar' / 'frames'

# Хранилище для комнат
rooms = {}

class Room:
    def __init__(self, name):
        self.name = name
        self.teacher = None
        self.participants = []
        self.animation = {
            'avatar_name': None,
            'frames': [],
            'current_frame': 0,
            'is_playing': False,
            'fps': 15
        }
        self.audio_queue = []

@app.route('/')
def home():
    return render_template('teacher.html')

@app.route('/conference')
def conference():
    room = request.args.get('room', 'default')
    return render_template('conference.html', room=room)

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

@socketio.on('connect')
def handle_connect():
    print('Client connected:', request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    for room_name, room in rooms.items():
        if request.sid == room.teacher:
            room.teacher = None
            emit('teacher_disconnected', room_name=room_name)
        elif request.sid in room.participants:
            room.participants.remove(request.sid)
            emit('participant_left', {'sid': request.sid}, room=room_name)
    
    print('Client disconnected:', request.sid)

@socketio.on('join_room')
def handle_join_room(data):
    room_name = data['room']
    is_teacher = data.get('is_teacher', False)
    
    if room_name not in rooms:
        rooms[room_name] = Room(room_name)
    
    room = rooms[room_name]
    
    if is_teacher:
        room.teacher = request.sid
        join_room(room_name)
        emit('room_joined', {'is_teacher': True, 'room': room_name})
    else:
        room.participants.append(request.sid)
        join_room(room_name)
        emit('room_joined', {
            'is_teacher': False,
            'room': room_name,
            'animation_state': room.animation,
            'teacher_sid': room.teacher
        })
        
        # Уведомляем учителя о новом участнике
        if room.teacher:
            emit('new_participant', {'sid': request.sid}, room=room.teacher)

@socketio.on('start_animation')
def handle_start_animation(data):
    room_name = data['room']
    if room_name not in rooms:
        return
    
    room = rooms[room_name]
    room.animation.update({
        'avatar_name': data['avatar_name'],
        'frames': data['frames'],
        'current_frame': 0,
        'is_playing': True,
        'fps': data.get('fps', 15)
    })
    
    emit('animation_state', room.animation, room=room_name)

@socketio.on('stop_animation')
def handle_stop_animation(data):
    room_name = data['room']
    if room_name not in rooms:
        return
    
    room = rooms[room_name]
    room.animation['is_playing'] = False
    emit('animation_state', room.animation, room=room_name)

@socketio.on('next_frame')
def handle_next_frame(data):
    room_name = data['room']
    if room_name not in rooms:
        return
    
    room = rooms[room_name]
    if room.animation['is_playing'] and room.animation['frames']:
        room.animation['current_frame'] = (room.animation['current_frame'] + 1) % len(room.animation['frames'])
        emit('frame_update', {
            'frame_index': room.animation['current_frame'],
            'frame_url': f"/frames/{room.animation['avatar_name']}/{room.animation['frames'][room.animation['current_frame']]}"
        }, room=room_name)

@socketio.on('text_to_speech')
def handle_text_to_speech(data):
    room_name = data['room']
    text = data['text']
    
    if not text or room_name not in rooms:
        return
    
    # Создаем аудио с помощью gTTS (мужской голос, быстрая скорость)
    tts = gTTS(text=text, lang='ru', slow=False)
    tts.lang = 'ru'  # Явно указываем русский язык
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    
    # Отправляем base64-кодированный аудиопоток
    audio_base64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
    emit('audio_stream', {'audio': audio_base64}, room=room_name)

@socketio.on('toggle_mute')
def handle_toggle_mute(data):
    room_name = data['room']
    is_muted = data['is_muted']
    emit('participant_muted', {'sid': request.sid, 'is_muted': is_muted}, room=room_name)

@socketio.on('toggle_video')
def handle_toggle_video(data):
    room_name = data['room']
    is_video_off = data['is_video_off']
    emit('participant_video_off', {'sid': request.sid, 'is_video_off': is_video_off}, room=room_name)

@socketio.on('share_screen')
def handle_share_screen(data):
    room_name = data['room']
    is_sharing = data['is_sharing']
    emit('screen_shared', {'sid': request.sid, 'is_sharing': is_sharing}, room=room_name)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)