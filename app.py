from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)

# Разрешаем все возможные расширения изображений
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'webp'}

@app.route('/')
def animation_test():
    return render_template('teacher.html')

@app.route('/static/avatar/frames/<path:filename>')
def serve_frames(filename):
    return send_from_directory('static/avatar/frames', filename)

if __name__ == '__main__':
    app.run(debug=True)