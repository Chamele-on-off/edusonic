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

@app.route('/api/avatar_frames/<avatar_name>')
def get_avatar_frames(avatar_name):
    try:
        avatar_dir = AVATAR_FRAMES_DIR / avatar_name
        if not avatar_dir.exists():
            return jsonify({"error": f"Avatar {avatar_name} not found"}), 404

        frames = {}
        # Собираем все файлы для аватара
        for filename in os.listdir(avatar_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                # Группируем по типу анимации (первая часть имени файла)
                anim_type = filename.split('_')[0]
                if anim_type not in frames:
                    frames[anim_type] = []
                frames[anim_type].append(f"/avatar/frames/{avatar_name}/{filename}")

        return jsonify(frames)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/avatar/frames/<avatar_name>/<path:filename>')
def serve_frame(avatar_name, filename):
    return send_from_directory(AVATAR_FRAMES_DIR / avatar_name, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)