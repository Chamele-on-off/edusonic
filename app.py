from flask import Flask, render_template, jsonify, send_from_directory
import os
from pathlib import Path

app = Flask(__name__)

# Конфигурация путей
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / 'static'
AVATAR_FRAMES_DIR = STATIC_DIR / 'avatar' / 'frames'

@app.route('/')
def home():
    return render_template('teacher.html')

@app.route('/api/avatar_frames')
def get_avatar_frames():
    frames = {
        'mouth_neutral': [],
        'blink': [],
        'mouth_aa': [],
        'mouth_bb': [],
        # Добавьте другие группы по аналогии
    }
    
    try:
        # Сканируем директорию с кадрами
        for filename in os.listdir(AVATAR_FRAMES_DIR):
            if filename.startswith('mouth_neutral_'):
                frames['mouth_neutral'].append(f'/static/avatar/frames/{filename}')
            elif filename.startswith('blink_'):
                frames['blink'].append(f'/static/avatar/frames/{filename}')
            elif filename.startswith('mouth_aa_'):
                frames['mouth_aa'].append(f'/static/avatar/frames/{filename}')
            elif filename.startswith('mouth_bb_'):
                frames['mouth_bb'].append(f'/static/avatar/frames/{filename}')
            # Добавьте другие условия для ваших файлов
            
        # Проверяем, что есть хотя бы нейтральные кадры
        if not frames['mouth_neutral']:
            raise Exception("No neutral mouth frames found")
            
        return jsonify(frames)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/avatar/frames/<path:filename>')
def serve_frames(filename):
    return send_from_directory(AVATAR_FRAMES_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)