from flask import Flask, render_template, jsonify, send_from_directory
import os
from pathlib import Path

app = Flask(__name__)

# Жестко прописываем доступные кадры для теста
TEST_FRAMES = {
    "mouth_neutral": [
        "/static/avatar/frames/mouth_neutral_001.jpg",
        "/static/avatar/frames/mouth_neutral_002.jpg"
    ],
    "blink": [
        "/static/avatar/frames/blink_01.jpg",
        "/static/avatar/frames/blink_02.jpg"
    ],
    "mouth_aa": [
        "/static/avatar/frames/mouth_aa_001.jpg",
        "/static/avatar/frames/mouth_aa_002.jpg"
    ]
}

@app.route('/')
def home():
    return render_template('teacher.html')

@app.route('/api/avatar_frames')
def get_avatar_frames():
    try:
        # Проверка существования файлов
        base_dir = Path(__file__).parent
        existing_frames = {k: [] for k in TEST_FRAMES.keys()}
        
        for anim_type, frames in TEST_FRAMES.items():
            for frame in frames:
                frame_path = base_dir / frame.lstrip('/')
                if frame_path.exists():
                    existing_frames[anim_type].append(frame)
                else:
                    print(f"File not found: {frame_path}")

        return jsonify(existing_frames)
    except Exception as e:
        return jsonify({"error": str(e), "test": "using hardcoded frames"}), 200

@app.route('/static/avatar/frames/<path:filename>')
def serve_frame(filename):
    return send_from_directory('static/avatar/frames', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)