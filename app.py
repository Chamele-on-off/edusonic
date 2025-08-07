import os
import json
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO
from pathlib import Path
from typing import Dict, Optional
from lesson_manager import LessonManager
from llm import LessonLLM
from tts import TextToSpeech
from lip_sync import AdvancedLipSync
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import uuid

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')

# Создаем директории
Path("static/audio").mkdir(parents=True, exist_ok=True)
Path("static/tmp").mkdir(parents=True, exist_ok=True)
Path("materials").mkdir(parents=True, exist_ok=True)
Path("static/reference").mkdir(parents=True, exist_ok=True)
Path("static/mouth_frames").mkdir(parents=True, exist_ok=True)

class AIServer:
    def __init__(self):
        self.llm = LessonLLM()
        self.tts = TextToSpeech()
        self.lip_sync = AdvancedLipSync(
            reference_video="static/reference/video.mp4",
            mouth_frames_dir="static/mouth_frames",
            output_dir="static/tmp"
        )
        self.lesson_manager = LessonManager()
        self.active_sessions: Dict[str, Dict] = {}
        self.active_conferences: Dict[str, Dict] = {}
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._load_materials()
        logger.info("AI Server initialized successfully")

    def _load_materials(self):
        try:
            materials_dir = Path("materials")
            if materials_dir.exists():
                self.llm.load_materials_from_dir(materials_dir)
                logger.info(f"Loaded materials from {materials_dir}")
            else:
                logger.warning("Materials directory not found, creating empty one")
                materials_dir.mkdir(parents=True)
        except Exception as e:
            logger.error(f"Failed to load materials: {str(e)}")

    async def start_lesson(self, lesson_id: str, room_id: str) -> str:
        try:
            def callback(message):
                socketio.emit('lesson_update', message, room=room_id)
            
            session_id = await self.lesson_manager.start_lesson(
                lesson_id=lesson_id,
                room_id=room_id,
                callback=callback
            )
            
            self.active_sessions[session_id] = {
                'lesson_id': lesson_id,
                'room_id': room_id,
                'status': 'running',
                'start_time': time.time(),
                'peer_id': f"ai_teacher_{room_id}"
            }
            
            logger.info(f"Lesson {lesson_id} started in room {room_id}, session_id: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start lesson: {str(e)}")
            raise

    async def connect_to_conference(self, room_id: str):
        try:
            if room_id in self.active_conferences:
                logger.info(f"Already connected to conference {room_id}")
                return True
                
            self.active_conferences[room_id] = {
                'peer_id': f"ai_teacher_{room_id}",
                'status': 'connected',
                'connected_at': time.time()
            }
            
            logger.info(f"Connected to conference {room_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to conference: {str(e)}")
            raise

    async def process_user_message(self, session_id: str, message: Dict):
        try:
            if session_id not in self.active_sessions:
                raise ValueError("Session not found")
                
            await self.lesson_manager.handle_user_response(session_id, message)
            logger.info(f"Processed user message in session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to process message: {str(e)}")
            raise

    async def generate_response(self, session_id: str, question: str) -> Dict:
        try:
            if session_id not in self.active_sessions:
                raise ValueError("Session not found")
                
            session = self.active_sessions[session_id]
            context = {
                'lesson': session['lesson_id'],
                'subject': self.lesson_manager._load_lesson_config(session['lesson_id']).subject,
                'current_phase': 'qa'
            }
            
            text_response = self.llm.generate_response(question, context)
            logger.info(f"Generated response for question: {question[:50]}...")
            
            audio_path, duration = await self.tts.generate_speech_async(text_response)
            video_path = await self.lip_sync.process_async(audio_path)
            
            audio_url = f"/static/audio/{Path(audio_path).name}"
            video_url = f"/static/tmp/{Path(video_path).name}"
            
            logger.info(f"Generated audio: {audio_url}, video: {video_url}")
            
            if session['room_id'] in self.active_conferences:
                socketio.emit('video_stream', {
                    'session_id': session_id,
                    'video_url': video_url,
                    'room_id': session['room_id']
                }, room=session['room_id'])
            
            return {
                'text': text_response,
                'audio_url': audio_url,
                'video_url': video_url,
                'duration': duration
            }
            
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise

    async def stop_lesson(self, session_id: str):
        try:
            if session_id in self.active_sessions:
                room_id = self.active_sessions[session_id]['room_id']
                await self.lesson_manager.stop_lesson(session_id)
                del self.active_sessions[session_id]
                
                if not any(s['room_id'] == room_id for s in self.active_sessions.values()):
                    await self.disconnect_from_conference(room_id)
                
                logger.info(f"Lesson {session_id} stopped successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to stop lesson: {str(e)}")
            raise

    async def disconnect_from_conference(self, room_id: str):
        try:
            if room_id in self.active_conferences:
                del self.active_conferences[room_id]
                logger.info(f"Disconnected from conference {room_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to disconnect from conference: {str(e)}")
            raise

ai_server = AIServer()

@app.route('/')
def index():
    return send_from_directory('static', 'teacher.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/api/lessons', methods=['GET'])
def list_lessons():
    try:
        lessons = ai_server.lesson_manager.list_available_lessons()
        return jsonify({'success': True, 'lessons': lessons})
    except Exception as e:
        logger.error(f"Failed to list lessons: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/start_lesson', methods=['POST'])
async def start_lesson():
    data = request.json
    try:
        if not data or 'lesson_id' not in data or 'room_id' not in data:
            raise ValueError("Missing lesson_id or room_id in request")
            
        await ai_server.connect_to_conference(data['room_id'])
        session_id = await ai_server.start_lesson(data['lesson_id'], data['room_id'])
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Lesson started successfully'
        })
    except Exception as e:
        logger.error(f"Error starting lesson: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/generate_response', methods=['POST'])
async def generate_response():
    data = request.json
    try:
        if not data or 'session_id' not in data or 'question' not in data:
            raise ValueError("Missing session_id or question in request")
            
        response = await ai_server.generate_response(data['session_id'], data['question'])
        return jsonify({'success': True, 'response': response})
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/stop_lesson', methods=['POST'])
async def stop_lesson():
    data = request.json
    try:
        if not data or 'session_id' not in data:
            raise ValueError("Missing session_id in request")
            
        success = await ai_server.stop_lesson(data['session_id'])
        return jsonify({
            'success': success,
            'message': 'Lesson stopped' if success else 'Lesson not found'
        })
    except Exception as e:
        logger.error(f"Error stopping lesson: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 400

@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('join_room')
async def handle_join_room(data):
    room_id = data.get('room_id', 'default_room')
    socketio.emit('system_message', {'text': f"New user joined room {room_id}"}, room=room_id)
    logger.info(f"User joined room {room_id}")

@socketio.on('user_message')
async def handle_user_message(data):
    try:
        if not data or 'session_id' not in data or 'text' not in data:
            raise ValueError("Missing session_id or text in message")
            
        await ai_server.process_user_message(
            session_id=data['session_id'],
            message={
                'type': 'question',
                'text': data['text'],
                'user_id': data.get('user_id', 'anonymous')
            }
        )
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        socketio.emit('error', {'message': str(e)}, room=request.sid)

@socketio.on('video_stream')
async def handle_video_stream(data):
    try:
        if not data or 'room_id' not in data or 'video_url' not in data:
            raise ValueError("Missing room_id or video_url in message")
            
        socketio.emit('video_update', {
            'video_url': data['video_url'],
            'peer_id': f"ai_teacher_{data['room_id']}"
        }, room=data['room_id'])
    except Exception as e:
        logger.error(f"Error processing video stream: {str(e)}")

def cleanup_temp_files():
    import glob
    for file in glob.glob('static/tmp/*') + glob.glob('static/audio/*'):
        try:
            if os.path.isfile(file):
                os.unlink(file)
        except Exception as e:
            logger.warning(f"Could not delete {file}: {str(e)}")

if __name__ == '__main__':
    cleanup_temp_files()
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
