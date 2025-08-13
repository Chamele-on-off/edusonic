from flask import Flask, render_template, send_from_directory
from pathlib import Path

app = Flask(__name__, static_folder='static')

@app.route('/')
def home():
    return render_template('teacher.html')

@app.route('/frames/<path:filename>')
def serve_frame(filename):
    return send_from_directory('static/avatar/frames', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)