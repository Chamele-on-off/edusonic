from flask import Flask, render_template, send_from_directory, jsonify, request
import os
from pathlib import Path
from flask_socketio import SocketIO, emit
from gtts import gTTS
import io
import base64
import time
import threading
from collections import defaultdict

app = Flask(__name__, static_folder='static')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Конфигурация путей
BASE_DIR = Path(__file__).parent
FRAMES_DIR = BASE_DIR / 'static' / 'avatar' / 'frames'

# Глобальные переменные
animation_loops = {}
rooms = defaultdict(dict)

@app.route('/')
def home():
    return render_template('teacher.html')

@app.route('/conference')
def conference():
    return render_template('conference.html')

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
    tts = gTTS(text=text, lang=lang)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return base64.b64encode(mp3_fp.read()).decode('utf-8')

def animation_loop(room, frames):
    frame_index = 0
    while room in rooms and rooms[room].get('animation_running', False):
        frame_index = (frame_index + 1) % len(frames)
        frame_data = {
            'frame': f"/frames/{rooms[room]['avatar_name']}/{frames[frame_index]}",
            'index': frame_index,
            'total': len(frames)
        }
        socketio.emit('animation_frame', frame_data, room=room)
        time.sleep(0.1)  # 10 FPS

@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    for room in list(rooms.keys()):
        if sid in rooms[room]['participants']:
            rooms[room]['participants'].remove(sid)
            socketio.emit('participants_update', 
                         {'count': len(rooms[room]['participants'])}, 
                         room=room)
            break
    print(f'Client disconnected: {sid}')

@socketio.on('join_room')
def handle_join_room(data):
    room = data.get('room', 'default')
    sid = request.sid
    
    if room not in rooms:
        rooms[room] = {
            'participants': [],
            'animation_running': False,
            'avatar_name': '',
            'frames': []
        }
    
    rooms[room]['participants'].append(sid)
    emit('participants_update', 
         {'count': len(rooms[room]['participants'])}, 
         room=room)
    
    # Отправляем текущее состояние анимации новому участнику
    if rooms[room]['animation_running']:
        emit('animation_started', {
            'avatar_name': rooms[room]['avatar_name'],
            'frames': rooms[room]['frames']
        }, room=sid)
    
    print(f'Client {sid} joined room {room}')

@socketio.on('start_animation')
def handle_start_animation(data):
    room = data.get('room', 'default')
    if room in rooms:
        rooms[room]['animation_running'] = True
        rooms[room]['avatar_name'] = data['avatar_name']
        rooms[room]['frames'] = data['frames']
        
        # Запускаем поток анимации, если он еще не работает
        if room not in animation_loops:
            animation_loops[room] = threading.Thread(
                target=animation_loop,
                args=(room, data['frames'])
            )
            animation_loops[room].start()
        
        # Оповещаем всех участников
        emit('animation_started', {
            'avatar_name': data['avatar_name'],
            'frames': data['frames']
        }, room=room)

@socketio.on('stop_animation')
def handle_stop_animation(data):
    room = data.get('room', 'default')
    if room in rooms:
        rooms[room]['animation_running'] = False
        emit('animation_stopped', {}, room=room)

@socketio.on('generate_speech')
def handle_generate_speech(data):
    text = data['text']
    room = data.get('room', 'default')
    audio_data = text_to_speech(text)
    emit('speech_audio', {'audio': audio_data}, room=room)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)