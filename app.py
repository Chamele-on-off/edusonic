from flask import Flask, render_template, jsonify, send_from_directory
import os
from pathlib import Path

app = Flask(__name__)

# Абсолютные пути к директориям
BASE_DIR = Path(__file__).parent
AVATAR_FRAMES_DIR = BASE_DIR / 'static' / 'avatar' / 'frames'

@app.route('/')
def home():
    return render_template('animation_test.html')

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
            # Добавьте другие группы
        }

        # Собираем файлы с разными расширениями
        extensions = ['.jpg', '.jpeg', '.png']
        for prefix in frames.keys():
            for ext in extensions:
                for i in range(1, 10):  # Проверяем до 9 кадров каждого типа
                    frame_name = f"{prefix}_{str(i).zfill(2)}{ext}"
                    frame_path = AVATAR_FRAMES_DIR / frame_name
                    if frame_path.exists():
                        frames[prefix].append(f"/static/avatar/frames/{frame_name}")

        # Проверяем, что есть хотя бы нейтральные кадры
        if not frames['mouth_neutral']:
            available_files = os.listdir(AVATAR_FRAMES_DIR)
            raise Exception(f"No neutral frames found. Available files: {available_files}")

        return jsonify(frames)

    except Exception as e:
        app.logger.error(f"Error loading frames: {str(e)}")
        return jsonify({"error": str(e), "available_files": os.listdir(AVATAR_FRAMES_DIR) if AVATAR_FRAMES_DIR.exists() else "Directory not found"}), 500

@app.route('/static/avatar/frames/<path:filename>')
def serve_frame(filename):
    return send_from_directory(AVATAR_FRAMES_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)