from flask import Flask, render_template, send_from_directory, jsonify
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
animation_running = False
current_animation_frames = []
current_frame_index = 0
participants = defaultdict(dict)
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

def animation_loop(room):
    global current_frame_index, animation_running
    while animation_running and room in rooms:
        if current_animation_frames:
            current_frame_index = (current_frame_index + 1) % len(current_animation_frames)
            frame_data = {
                'frame': current_animation_frames[current_frame_index],
                'index': current_frame_index,
                'total': len(current_animation_frames)
            }
            socketio.emit('animation_frame', frame_data, room=room)
        time.sleep(0.1)  # 10 FPS

@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    for room, users in rooms.items():
        if sid in users:
            del users[sid]
            socketio.emit('participants_update', {'count': len(users)}, room=room)
            break
    print(f'Client disconnected: {sid}')

@socketio.on('join_room')
def handle_join_room(data):
    room = data.get('room', 'default')
    sid = request.sid
    rooms[room][sid] = {'sid': sid}
    emit('participants_update', {'count': len(rooms[room])}, room=room)
    print(f'Client {sid} joined room {room}')

@socketio.on('start_animation')
def handle_start_animation(data):
    global animation_running, current_animation_frames
    room = data.get('room', 'default')
    if not animation_running:
        current_animation_frames = data['frames']
        animation_running = True
        threading.Thread(target=animation_loop, args=(room,)).start()

@socketio.on('stop_animation')
def handle_stop_animation(data):
    global animation_running
    animation_running = False

@socketio.on('generate_speech')
def handle_generate_speech(data):
    text = data['text']
    room = data.get('room', 'default')
    audio_data = text_to_speech(text)
    emit('speech_audio', {'audio': audio_data}, room=room)

@socketio.on('user_media')
def handle_user_media(data):
    sid = request.sid
    room = data.get('room', 'default')
    if sid in rooms[room]:
        rooms[room][sid]['media'] = data['media']
        emit('user_media_update', {
            'sid': sid,
            'media': data['media']
        }, room=room, include_self=False)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)