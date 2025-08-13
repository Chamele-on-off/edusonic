from flask import Flask, render_template, jsonify, send_from_directory
import os
from pathlib import Path
import base64

app = Flask(__name__)

# Конфигурация путей
BASE_DIR = Path(__file__).parent
AVATAR_FRAMES_DIR = BASE_DIR / 'static' / 'avatar' / 'frames'

@app.route('/')
def home():
    return render_template('teacher.html')

@app.route('/api/avatar_frames')
def get_avatar_frames():
    try:
        # Проверяем существование директории
        if not AVATAR_FRAMES_DIR.exists():
            raise Exception(f"Directory not found: {AVATAR_FRAMES_DIR}")

        frames = {
            'mouth_neutral': [],
            'blink': [],
            'mouth_aa': [],
            'mouth_bb': []
        }

        # Сканируем директорию и проверяем существование файлов
        for anim_type in frames.keys():
            for ext in ['jpg', 'jpeg', 'png']:
                for i in range(1, 10):
                    frame_name = f"{anim_type}_{str(i).zfill(3)}.{ext}"
                    frame_path = AVATAR_FRAMES_DIR / frame_name
                    if frame_path.exists():
                        frames[anim_type].append(f"/avatar/frames/{frame_name}")

        return jsonify(frames)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/avatar/frames/<path:filename>')
def serve_frame(filename):
    return send_from_directory(AVATAR_FRAMES_DIR, filename)

@app.route('/api/avatar_frames_base64')
def get_frames_base64():
    try:
        frames = {}
        for anim_type in ['mouth_neutral', 'blink', 'mouth_aa']:
            frames[anim_type] = []
            for ext in ['jpg', 'jpeg', 'png']:
                for i in range(1, 3):
                    frame_name = f"{anim_type}_{str(i).zfill(3)}.{ext}"
                    frame_path = AVATAR_FRAMES_DIR / frame_name
                    if frame_path.exists():
                        with open(frame_path, 'rb') as f:
                            encoded = base64.b64encode(f.read()).decode('utf-8')
                            frames[anim_type].append(f"data:image/jpeg;base64,{encoded}")
        return jsonify(frames)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)