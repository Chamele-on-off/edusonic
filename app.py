from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_socketio import SocketIO
from flask_cors import CORS
from llm import KnowledgeBase
from gtts import gTTS
import base64
from io import BytesIO
import logging
from pathlib import Path
import uuid
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

# Инициализация директорий
Path("static/audio").mkdir(parents=True, exist_ok=True)
Path("materials").mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(parents=True, exist_ok=True)
Path("static/lessons").mkdir(parents=True, exist_ok=True)  # Добавлено для уроков

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Инициализация Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET', 'dev-secret-key-123')
CORS(app)  # Включение CORS для всех доменов
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

class SimpleTTS:
    def __init__(self):
        self.phoneme_map = {
            'а': 0.3, 'о': 0.3, 'у': 0.3, 'и': 0.3, 'э': 0.3,
            'б': 0.1, 'в': 0.1, 'г': 0.1, 'д': 0.1, 'ж': 0.1
        }

    def text_to_phonemes(self, text: str) -> List[Tuple[str, float]]:
        """Генерация упрощенных фонем"""
        return [
            (char, self.phoneme_map.get(char.lower(), 0.15))
            for char in text if char.isalpha()
        ][:50]  # Ограничение количества

    def synthesize(self, text: str, lang: str = 'ru') -> dict:
        """Синтез речи с помощью gTTS"""
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            return {
                "audio": base64.b64encode(audio_buffer.read()).decode('utf-8'),
                "phonemes": self.text_to_phonemes(text),
                "text": text
            }
        except Exception as e:
            logger.error(f"TTS error: {str(e)}")
            return {
                "error": str(e),
                "phonemes": [('а', 0.2)] * 3
            }

# Инициализация компонентов
kb = KnowledgeBase()
tts = SimpleTTS()
executor = ThreadPoolExecutor(max_workers=4)

def init_demo_data():
    """Загрузка демо-материалов"""
    if not kb.find_similar("", "обществознание", 1):
        demo_materials = [
            ("обществознание", "Понятие общества", "Общество — это совокупность людей..."),
            ("обществознание", "Государство", "Государство — политическая организация..."),
        ]
        
        for subject, title, content in demo_materials:
            kb.add_material(subject, title, content)
        
        # Создаем демо-урок, если нет ни одного
        if not list(Path("static/lessons").glob("*.json")):
            demo_lesson = {
                "id": "demo-lesson",
                "title": "Демо-урок",
                "subject": "обществознание",
                "description": "Пример урока по обществознанию",
                "phases": []
            }
            with open(Path("static/lessons/demo-lesson.json"), 'w') as f:
                json.dump(demo_lesson, f, ensure_ascii=False, indent=2)

init_demo_data()

# API Endpoints
@app.route('/')
def home():
    return render_template('teacher.html')

@app.route('/static/<path:subpath>/<path:filename>')
def static_files(subpath, filename):
    """Обработка статических файлов (аватар, модели и т.д.)"""
    return send_from_directory(f'static/{subpath}', filename)

@app.route('/api/lessons')
def list_lessons():
    """Получение списка доступных уроков"""
    lessons = []
    try:
        for lesson_file in Path("static/lessons").glob("*.json"):
            try:
                with open(lesson_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    lessons.append({
                        'id': data['id'],
                        'title': data['title'],
                        'subject': data['subject'],
                        'description': data['description']
                    })
            except Exception as e:
                logger.error(f"Error loading lesson {lesson_file}: {str(e)}")
        return jsonify({"lessons": lessons})
    except Exception as e:
        logger.error(f"Error listing lessons: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/materials')
def get_materials():
    """Получение учебных материалов"""
    subject = request.args.get('subject', 'обществознание')
    return jsonify({
        "materials": kb.find_similar("", subject, 10)
    })

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Обработка вопросов"""
    data = request.json
    question = data.get('question', '')
    subject = data.get('subject', 'обществознание')
    
    materials = kb.find_similar(question, subject)
    response = kb.generate_response(question, materials)
    audio = tts.synthesize(response['text'])
    
    return jsonify({
        **response,
        "audio": audio['audio'],
        "phonemes": audio['phonemes']
    })

# WebSocket Handlers
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('ask_question')
def handle_question(data):
    session_id = data.get('session_id', str(uuid.uuid4()))
    
    def process():
        try:
            materials = kb.find_similar(
                data['question'], 
                data.get('subject', 'обществознание')
            )
            response = kb.generate_response(data['question'], materials)
            audio = tts.synthesize(response['text'])
            
            socketio.emit('ai_response', {
                "session_id": session_id,
                "text": response['text'],
                "audio": audio['audio'],
                "phonemes": audio['phonemes'],
                "materials": response['materials']
            }, room=request.sid)
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            socketio.emit('error', {
                "session_id": session_id,
                "message": str(e)
            }, room=request.sid)
    
    executor.submit(process)

if __name__ == '__main__':
    try:
        host = os.getenv('HOST', '0.0.0.0')
        port = int(os.getenv('PORT', 5000))
        logger.info(f"Starting server on {host}:{port}")
        socketio.run(app, host=host, port=port, debug=True)
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
    finally:
        executor.shutdown()
        kb.close()