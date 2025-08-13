from flask import Flask, render_template, send_from_directory, jsonify
from flask_socketio import SocketIO, emit
import os
from pathlib import Path
from gtts import gTTS
import io
import base64
import time

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Конфигурация путей
BASE_DIR = Path(__file__).parent
FRAMES_DIR = BASE_DIR / 'static' / 'avatar' / 'frames'

# Хранилище для текущего состояния анимации
current_animation = {
    'avatar_name': None,
    'frames': [],
    'current_frame': 0,
    'is_playing': False,
    'fps': 10
}

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

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('animation_state', current_animation)

@socketio.on('start_animation')
def handle_start_animation(data):
    current_animation.update({
        'avatar_name': data['avatar_name'],
        'frames': data['frames'],
        'current_frame': 0,
        'is_playing': True,
        'fps': data.get('fps', 10)
    })
    emit('animation_state', current_animation, broadcast=True)

@socketio.on('stop_animation')
def handle_stop_animation():
    current_animation['is_playing'] = False
    emit('animation_state', current_animation, broadcast=True)

@socketio.on('next_frame')
def handle_next_frame():
    if current_animation['is_playing'] and current_animation['frames']:
        current_animation['current_frame'] = (current_animation['current_frame'] + 1) % len(current_animation['frames'])
        emit('frame_update', {
            'frame_index': current_animation['current_frame'],
            'frame_url': f"/frames/{current_animation['avatar_name']}/{current_animation['frames'][current_animation['current_frame']]}"
        }, broadcast=True)

@socketio.on('text_to_speech')
def handle_text_to_speech(data):
    text = data['text']
    lang = data.get('lang', 'ru')
    
    # Создаем аудио с помощью gTTS
    tts = gTTS(text=text, lang=lang, slow=False)
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    
    # Отправляем base64-кодированный аудиопоток
    audio_base64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
    emit('audio_stream', {'audio': audio_base64}, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)