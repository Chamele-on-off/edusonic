from flask import Flask, render_template, send_from_directory, jsonify, Response
import os
from pathlib import Path
import cv2
import numpy as np
from io import BytesIO
from gtts import gTTS
import threading
import time
from flask_socketio import SocketIO

app = Flask(__name__, static_folder='static')
socketio = SocketIO(app, cors_allowed_origins="*")

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

def generate_audio(text):
    tts = gTTS(text=text, lang='ru')
    audio_file = BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    return audio_file

def generate_video_frames(avatar_name):
    avatar_dir = FRAMES_DIR / avatar_name
    frames = [f for f in os.listdir(avatar_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    frames.sort()
    
    while True:
        for frame in frames:
            frame_path = str(avatar_dir / frame)
            img = cv2.imread(frame_path)
            if img is None:
                continue
                
            ret, buffer = cv2.imencode('.jpg', img)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.1)  # 10 FPS

@app.route('/video_feed/<avatar_name>')
def video_feed(avatar_name):
    return Response(generate_video_frames(avatar_name),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/audio_feed')
def audio_feed():
    text = "Добро пожаловать на конференцию. Это тестовое аудио сообщение."
    audio_file = generate_audio(text)
    
    def generate():
        while True:
            data = audio_file.read(1024)
            if not data:
                audio_file.seek(0)
                continue
            yield data
            time.sleep(0.1)
    
    return Response(generate(), mimetype='audio/mpeg')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)