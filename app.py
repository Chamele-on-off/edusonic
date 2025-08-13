from flask import Flask, render_template, send_from_directory, jsonify
import os
from pathlib import Path
from flask_socketio import SocketIO, emit
from gtts import gTTS
import base64
import io

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Конфигурация путей
BASE_DIR = Path(__file__).parent
FRAMES_DIR = BASE_DIR / 'static' / 'avatar' / 'frames'

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

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_stream')
def handle_start_stream(data):
    # Пересылаем всем клиентам, кроме отправителя
    emit('stream_started', data, broadcast=True, include_self=False)

@socketio.on('stream_frame')
def handle_stream_frame(data):
    # Пересылаем кадр всем клиентам, кроме отправителя
    emit('new_frame', data, broadcast=True, include_self=False)

@socketio.on('text_to_speech')
def handle_text_to_speech(data):
    text = data['text']
    lang = data.get('lang', 'ru')
    
    # Генерируем аудио
    tts = gTTS(text=text, lang=lang)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    
    # Отправляем base64-кодированное аудио
    audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
    emit('new_audio', {'audio': audio_base64}, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)