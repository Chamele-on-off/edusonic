import os
import json
import uuid
import time
import logging
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
import base64
from io import BytesIO
from gtts import gTTS

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

# Конфигурация
BASE_DIR = Path(__file__).parent
CONFIG = {
    "materials_dir": "materials",
    "lessons_dir": "static/lessons",
    "avatar_frames_dir": "static/avatar/frames",
    "audio_cache": "static/audio",
    "max_workers": 4
}

# Создание необходимых директорий
for dir_path in [CONFIG["lessons_dir"], CONFIG["avatar_frames_dir"], CONFIG["audio_cache"], "logs", CONFIG["materials_dir"]]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# Инициализация Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET', 'dev-secret-key-123')
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")
executor = ThreadPoolExecutor(max_workers=CONFIG["max_workers"])

# Глобальные переменные для управления комнатами и сессиями
rooms = {}  # {room_id: {participants: [user_ids], teacher: user_id}}
active_sessions = {}  # {session_id: {room_id, lesson_id, participants}}

class KnowledgeBase:
    """База знаний для учебных материалов"""
    def __init__(self, data_dir: str = CONFIG["materials_dir"]):
        self.data_dir = Path(data_dir)
        self.data_file = self.data_dir / "materials.json"
        self._init_data()

    def _init_data(self):
        if not self.data_file.exists():
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump({"materials": []}, f, ensure_ascii=False, indent=2)

    def add_material(self, subject: str, title: str, content: str):
        with open(self.data_file, 'r+', encoding='utf-8') as f:
            data = json.load(f)
            data["materials"].append({
                "id": str(uuid.uuid4()),
                "subject": subject,
                "title": title,
                "content": content
            })
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Добавлен материал: {title} ({subject})")

    def find_materials(self, query: str, subject: str, limit: int = 3) -> List[Dict]:
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        materials = [m for m in data["materials"] if m["subject"].lower() == subject.lower()]
        
        if not query:
            return materials[:limit]
        
        # Простой поиск по ключевым словам
        query_words = set(query.lower().split())
        for m in materials:
            content_words = set(m["content"].lower().split())
            m["score"] = len(query_words & content_words)
        
        return sorted(materials, key=lambda x: x.get("score", 0), reverse=True)[:limit]

class TTSEngine:
    """Синтезатор речи с кэшированием"""
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def synthesize(self, text: str, lang: str = 'ru') -> Dict:
        cache_key = f"{lang}_{hash(text)}"
        cache_file = self.cache_dir / f"{cache_key}.mp3"
        
        try:
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    audio_data = f.read()
                logger.info(f"Using cached audio for: {text[:50]}...")
            else:
                tts = gTTS(text=text, lang=lang, slow=False)
                audio_buffer = BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_data = audio_buffer.getvalue()
                with open(cache_file, 'wb') as f:
                    f.write(audio_data)
                logger.info(f"Generated new audio for: {text[:50]}...")

            return {
                "audio": base64.b64encode(audio_data).decode('utf-8'),
                "text": text
            }
        except Exception as e:
            logger.error(f"TTS error: {str(e)}")
            return {"error": str(e)}

# Инициализация компонентов
knowledge_base = KnowledgeBase()
tts_engine = TTSEngine(CONFIG["audio_cache"])

@app.route('/')
def home():
    return render_template('teacher.html')

@app.route('/api/lessons')
def list_lessons():
    lessons = []
    for lesson_file in Path(CONFIG["lessons_dir"]).glob("*.json"):
        try:
            with open(lesson_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                lessons.append({
                    'id': data.get('id', lesson_file.stem),
                    'title': data.get('title', 'Без названия'),
                    'subject': data.get('subject', 'не указан'),
                    'description': data.get('description', '')
                })
        except Exception as e:
            logger.error(f"Ошибка загрузки урока {lesson_file}: {str(e)}")
    return jsonify({"lessons": lessons})

@app.route('/api/upload_lesson', methods=['POST'])
def upload_lesson():
    if 'lesson' not in request.files:
        return jsonify({"error": "Файл не загружен"}), 400
    
    file = request.files['lesson']
    if file.filename == '':
        return jsonify({"error": "Пустое имя файла"}), 400
    
    try:
        lesson_id = request.form.get('lesson_id', str(uuid.uuid4()))
        save_path = Path(CONFIG["lessons_dir"]) / f"{lesson_id}.json"
        file.save(save_path)
        return jsonify({"status": "success", "lesson_id": lesson_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/start_lesson', methods=['POST'])
def start_lesson():
    data = request.json
    lesson_id = data.get('lesson_id')
    room_id = data.get('room_id')
    
    if not lesson_id or not room_id:
        return jsonify({"error": "Не указан ID урока или комнаты"}), 400
    
    try:
        session_id = f"{lesson_id}_{room_id}_{int(time.time())}"
        active_sessions[session_id] = {
            "lesson_id": lesson_id,
            "room_id": room_id,
            "start_time": time.time()
        }
        return jsonify({
            "session_id": session_id,
            "status": "started"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stop_lesson', methods=['POST'])
def stop_lesson():
    session_id = request.json.get('session_id')
    if not session_id or session_id not in active_sessions:
        return jsonify({"error": "Неверный ID сессии"}), 400
    
    del active_sessions[session_id]
    return jsonify({"status": "stopped"})

@app.route('/api/add_material', methods=['POST'])
def add_material():
    try:
        data = request.json
        knowledge_base.add_material(
            subject=data['subject'],
            title=data['title'],
            content=data['content']
        )
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Socket.IO обработчики
@socketio.on('connect')
def handle_connect():
    logger.info(f"Клиент подключен: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    for room_id, room_data in rooms.items():
        if request.sid in room_data['participants']:
            room_data['participants'].remove(request.sid)
            emit('user_left', {'userId': request.sid}, room=room_id)
            logger.info(f"Пользователь {request.sid} вышел из комнаты {room_id}")

@socketio.on('create_room')
def handle_create_room(data):
    room_id = data['room']
    user_id = data['userId']
    
    if room_id not in rooms:
        rooms[room_id] = {
            'participants': [user_id],
            'teacher': user_id
        }
        emit('room_created', {'room': room_id}, room=request.sid)
        logger.info(f"Комната {room_id} создана пользователем {user_id}")
    else:
        emit('error', {'message': 'Комната уже существует'}, room=request.sid)

@socketio.on('join_room')
def handle_join_room(data):
    room_id = data['room']
    user_id = data['userId']
    
    if room_id in rooms:
        rooms[room_id]['participants'].append(user_id)
        emit('user_joined', {'userId': user_id, 'room': room_id}, room=room_id)
        emit('room_joined', {'room': room_id}, room=request.sid)
        logger.info(f"Пользователь {user_id} присоединился к комнате {room_id}")
    else:
        emit('error', {'message': 'Комната не найдена'}, room=request.sid)

@socketio.on('leave_room')
def handle_leave_room(data):
    room_id = data['room']
    user_id = data.get('userId', request.sid)
    
    if room_id in rooms and user_id in rooms[room_id]['participants']:
        rooms[room_id]['participants'].remove(user_id)
        emit('user_left', {'userId': user_id}, room=room_id)
        logger.info(f"Пользователь {user_id} вышел из комнаты {room_id}")

@socketio.on('ask_question')
def handle_question(data):
    def process():
        try:
            question = data.get('question', '')
            subject = data.get('subject', 'обществознание')
            session_id = data.get('session_id')
            
            # Поиск релевантных материалов
            materials = knowledge_base.find_materials(question, subject)
            response_text = "\n".join([m['content'] for m in materials])
            
            # Синтез речи
            tts_result = tts_engine.synthesize(response_text)
            
            # Отправка ответа
            emit('ai_response', {
                "text": response_text,
                "audio": tts_result["audio"],
                "materials": materials
            }, room=request.sid)
            
            logger.info(f"Обработан вопрос: {question[:50]}...")
        except Exception as e:
            logger.error(f"Ошибка обработки вопроса: {str(e)}")
            emit('error', {"message": str(e)}, room=request.sid)
    
    executor.submit(process)

if __name__ == '__main__':
    try:
        host = os.getenv('HOST', '0.0.0.0')
        port = int(os.getenv('PORT', 5000))
        logger.info(f"Сервер запущен на {host}:{port}")
        socketio.run(app, host=host, port=port, debug=True)
    except Exception as e:
        logger.error(f"Ошибка сервера: {str(e)}")
    finally:
        executor.shutdown()
        logger.info("Сервер остановлен")