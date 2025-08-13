import os
import json
import uuid
import time
import random
import logging
import base64
from pathlib import Path
from io import BytesIO
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, render_template, send_file
from flask_socketio import SocketIO, join_room, leave_room
from PIL import Image, ImageDraw
import numpy as np

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

# Глобальные переменные для хранения комнат и пользователей
rooms = {}
users = {}

class KnowledgeBase:
    """База знаний для ответов на вопросы (JSON-based)"""
    def __init__(self, data_dir: str = CONFIG["materials_dir"]):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.materials_file = self.data_dir / "materials.json"
        self._init_data()
    
    def _init_data(self):
        """Инициализирует данные, если файла нет"""
        if not self.materials_file.exists():
            with open(self.materials_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "materials": [
                        {
                            "id": "1",
                            "subject": "обществознание",
                            "title": "Понятие общества",
                            "content": "Общество — это совокупность людей, объединенных исторически сложившимися формами взаимодействия."
                        },
                        {
                            "id": "2",
                            "subject": "обществознание",
                            "title": "Государство",
                            "content": "Государство — политическая организация общества, обладающая суверенитетом."
                        }
                    ]
                }, f, ensure_ascii=False, indent=2)
    
    def _load_materials(self) -> List[Dict]:
        """Загружает материалы из JSON файла"""
        try:
            with open(self.materials_file, 'r', encoding='utf-8') as f:
                return json.load(f).get("materials", [])
        except Exception as e:
            logger.error(f"Ошибка загрузки материалов: {str(e)}")
            return []
    
    def _save_materials(self, materials: List[Dict]):
        """Сохраняет материалы в JSON файл"""
        with open(self.materials_file, 'w', encoding='utf-8') as f:
            json.dump({"materials": materials}, f, ensure_ascii=False, indent=2)
    
    def add_material(self, subject: str, title: str, content: str):
        """Добавляет учебный материал"""
        materials = self._load_materials()
        materials.append({
            "id": str(uuid.uuid4()),
            "subject": subject,
            "title": title,
            "content": content
        })
        self._save_materials(materials)
        logger.info(f"Добавлен материал: {title} ({subject})")
    
    def find_similar(self, query: str, subject: str, top_k: int = 3) -> List[Dict]:
        """Находит релевантные материалы (простая реализация)"""
        materials = self._load_materials()
        filtered = [m for m in materials if m.get("subject") == subject]
        
        if not query:
            return filtered[:top_k]
        
        # Простая фильтрация по вхождению слов запроса
        query_words = set(query.lower().split())
        for m in filtered:
            content_words = set(m["content"].lower().split())
            m["score"] = len(query_words & content_words)
        
        return sorted(filtered, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

class SimpleTTS:
    """Синтезатор речи с кэшированием"""
    def __init__(self, cache_dir: str = CONFIG["audio_cache"]):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.phoneme_map = {
            'а': ('mouth_aa', 0.3), 'о': ('mouth_oo', 0.3),
            'у': ('mouth_uu', 0.3), 'и': ('mouth_ee', 0.3),
            'э': ('mouth_ee', 0.3), 'ы': ('mouth_aa', 0.3),
            'е': ('mouth_ee', 0.3), 'ё': ('mouth_oo', 0.3),
            'ю': ('mouth_uu', 0.3), 'я': ('mouth_aa', 0.3),
            'м': ('mouth_mm', 0.15), 'п': ('mouth_pp', 0.15),
            'б': ('mouth_bb', 0.15), 'ф': ('mouth_ff', 0.15),
            'в': ('mouth_vv', 0.15), 'ш': ('mouth_sh', 0.15),
            'ж': ('mouth_zh', 0.15), 'с': ('mouth_ss', 0.15),
            'з': ('mouth_zz', 0.15), 'р': ('mouth_rr', 0.15),
            'л': ('mouth_ll', 0.15), 'н': ('mouth_nn', 0.15),
            'т': ('mouth_tt', 0.15), 'д': ('mouth_dd', 0.15),
            'к': ('mouth_kk', 0.15), 'г': ('mouth_gg', 0.15),
            'х': ('mouth_hh', 0.15), 'ч': ('mouth_ch', 0.15),
            'щ': ('mouth_sh', 0.15), 'ц': ('mouth_ss', 0.15),
            'й': ('mouth_ee', 0.15)
        }

    def text_to_phonemes(self, text: str) -> List[Tuple[str, float]]:
        """Конвертирует текст в последовательность фонем"""
        return [
            self.phoneme_map.get(char, ('mouth_neutral', 0.15))
            for char in text.lower() if char in self.phoneme_map
        ][:100]

    def synthesize(self, text: str, lang: str = 'ru') -> Dict:
        """Синтезирует речь и возвращает аудио + фонемы"""
        from gtts import gTTS
        
        cache_key = f"{lang}_{hash(text)}"
        cache_file = self.cache_dir / f"{cache_key}.wav"
        
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
                "phonemes": self.text_to_phonemes(text),
                "text": text
            }
        except Exception as e:
            logger.error(f"TTS error: {str(e)}")
            return {
                "error": str(e),
                "phonemes": [('mouth_neutral', 0.2)] * 3
            }

# Инициализация компонентов
knowledge_base = KnowledgeBase()
tts_engine = SimpleTTS()

# API Endpoints
@app.route('/')
def home():
    return render_template('teacher.html')

@app.route('/api/avatar_frames')
def get_avatar_frames():
    """Возвращает список доступных кадров аватара для фронтенда"""
    frames = {}
    frames_dir = Path(CONFIG["avatar_frames_dir"])
    
    for group in ['mouth_neutral', 'mouth_aa', 'mouth_oo', 'mouth_ee', 'blink']:
        frame_files = sorted(frames_dir.glob(f"{group}_*.jpg"))
        frames[group] = [f"/static/avatar/frames/{f.name}" for f in frame_files]
    
    # Если нет кадров, возвращаем пустой список
    if not any(frames.values()):
        frames = {
            'mouth_neutral': ['/static/avatar/frames/mouth_neutral_001.jpg'],
            'blink': ['/static/avatar/frames/mouth_neutral_001.jpg']
        }
    
    return jsonify(frames)

@app.route('/api/lessons')
def list_lessons():
    lessons = []
    for lesson_file in Path(CONFIG["lessons_dir"]).glob("*.json"):
        try:
            with open(lesson_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                lessons.append({
                    'id': data.get('id', os.path.splitext(lesson_file.name)[0]),
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
        return jsonify({
            "session_id": session_id,
            "status": "started"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

# WebSocket handlers
@socketio.on('connect')
def handle_connect():
    logger.info(f"Клиент подключен: {request.sid}")
    users[request.sid] = {
        'room': None,
        'role': None
    }

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Клиент отключен: {request.sid}")
    user = users.get(request.sid)
    if user and user['room']:
        leave_room(user['room'])
        socketio.emit('user_left', {
            'userId': request.sid,
            'room': user['room']
        }, room=user['room'])
        
        # Обновляем информацию о комнате
        if user['room'] in rooms:
            rooms[user['room']]['users'].remove(request.sid)
            if not rooms[user['room']]['users']:
                del rooms[user['room']]
    
    if request.sid in users:
        del users[request.sid]

@socketio.on('join_room')
def handle_join_room(data):
    room = data.get('room')
    role = data.get('role', 'student')
    
    if not room:
        return
    
    # Покидаем предыдущую комнату, если есть
    if users[request.sid]['room']:
        leave_room(users[request.sid]['room'])
        socketio.emit('user_left', {
            'userId': request.sid,
            'room': users[request.sid]['room']
        }, room=users[request.sid]['room'])
    
    # Присоединяемся к новой комнате
    join_room(room)
    users[request.sid] = {
        'room': room,
        'role': role
    }
    
    # Обновляем информацию о комнате
    if room not in rooms:
        rooms[room] = {
            'users': [],
            'teacher': None
        }
    
    rooms[room]['users'].append(request.sid)
    if role == 'teacher':
        rooms[room]['teacher'] = request.sid
    
    # Уведомляем всех в комнате о новом пользователе
    socketio.emit('user_joined', {
        'userId': request.sid,
        'room': room,
        'role': role
    }, room=room)
    
    # Отправляем подтверждение подключения
    socketio.emit('room_joined', {
        'room': room,
        'role': role,
        'users': [{
            'userId': uid,
            'role': users[uid]['role']
        } for uid in rooms[room]['users']]
    }, to=request.sid)

@socketio.on('leave_room')
def handle_leave_room():
    user = users.get(request.sid)
    if not user or not user['room']:
        return
    
    room = user['room']
    leave_room(room)
    socketio.emit('user_left', {
        'userId': request.sid,
        'room': room
    }, room=room)
    
    # Обновляем информацию о комнате
    if room in rooms:
        rooms[room]['users'].remove(request.sid)
        if request.sid == rooms[room]['teacher']:
            rooms[room]['teacher'] = None
        
        if not rooms[room]['users']:
            del rooms[room]
    
    users[request.sid]['room'] = None
    users[request.sid]['role'] = None

@socketio.on('offer')
def handle_offer(data):
    to = data.get('to')
    room = data.get('room')
    offer = data.get('offer')
    
    if not to or not room or not offer:
        return
    
    # Пересылаем оффер указанному пользователю
    socketio.emit('offer', {
        'from': request.sid,
        'offer': offer,
        'room': room
    }, to=to)

@socketio.on('answer')
def handle_answer(data):
    to = data.get('to')
    room = data.get('room')
    answer = data.get('answer')
    
    if not to or not room or not answer:
        return
    
    # Пересылаем ответ указанному пользователю
    socketio.emit('answer', {
        'from': request.sid,
        'answer': answer,
        'room': room
    }, to=to)

@socketio.on('ice_candidate')
def handle_ice_candidate(data):
    to = data.get('to')
    room = data.get('room')
    candidate = data.get('candidate')
    
    if not to or not room or not candidate:
        return
    
    # Пересылаем ICE кандидат указанному пользователю
    socketio.emit('ice_candidate', {
        'from': request.sid,
        'candidate': candidate,
        'room': room
    }, to=to)

@socketio.on('ask_question')
def handle_question(data):
    def process():
        try:
            question = data.get('question', '')
            subject = data.get('subject', 'обществознание')
            
            materials = knowledge_base.find_similar(question, subject)
            response_text = "\n".join([m['content'] for m in materials])
            
            tts_result = tts_engine.synthesize(response_text)
            
            socketio.emit('ai_response', {
                "text": response_text,
                "audio": tts_result["audio"],
                "phonemes": tts_result["phonemes"],
                "materials": materials
            }, room=request.sid)
        except Exception as e:
            logger.error(f"Ошибка обработки вопроса: {str(e)}")
            socketio.emit('error', {"message": str(e)}, room=request.sid)
    
    executor.submit(process)

@socketio.on('audio_data')
def handle_audio_data(data):
    def process():
        try:
            from speech_recognition import Recognizer, AudioData
            import wave
            
            audio_bytes = base64.b64decode(data['audio'])
            
            # Конвертируем в формат, понятный для speech_recognition
            with wave.open(BytesIO(audio_bytes), 'rb') as wav_file:
                audio_data = AudioData(
                    wav_file.readframes(wav_file.getnframes()),
                    wav_file.getframerate(),
                    wav_file.getsampwidth()
                )
            
            recognizer = Recognizer()
            text = recognizer.recognize_google(audio_data, language="ru-RU")
            
            socketio.emit('transcription', {
                "text": text,
                "room": data['room']
            }, room=request.sid)
            
        except Exception as e:
            logger.error(f"Ошибка транскрипции: {str(e)}")
    
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