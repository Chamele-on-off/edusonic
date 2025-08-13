from flask import Flask, render_template, send_from_directory
import os
from pathlib import Path

app = Flask(__name__)

# Конфигурация путей
BASE_DIR = Path(__file__).parent
FRAMES_DIR = BASE_DIR / 'static' / 'avatar' / 'frames'

@app.route('/')
def home():
    return render_template('teacher.html')

@app.route('/static/avatar/frames/<path:filename>')
def serve_frame(filename):
    return send_from_directory(FRAMES_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)