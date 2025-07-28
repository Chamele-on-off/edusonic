import os
import json
import logging
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from pathlib import Path
from typing import Dict, Optional
from lesson_manager import LessonManager
from llm import LessonLLM
from tts import TextToSpeech
from inference import Wav2LipInference
import asyncio
from concurrent.futures import ThreadPoolExecutor

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

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')

class AIServer:
    def __init__(self):
        self.llm = LessonLLM()
        self.tts = TextToSpeech()
        self.wav2lip = Wav2LipInference()
        self.lesson_manager = LessonManager()
        self.active_sessions: Dict[str, Dict] = {}
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Загрузка материалов
        self._load_materials()
        
        logger.info("AI Server initialized")

    def _load_materials(self):
        """Загрузка учебных материалов"""
        try:
            materials_dir = Path("materials")
            if materials_dir.exists():
                self.llm.load_materials_from_dir(materials_dir)
                logger.info(f"Loaded materials from {materials_dir}")
            else:
                logger.warning("Materials directory not found")
        except Exception as e:
            logger.error(f"Failed to load materials: {str(e)}")

    async def start_lesson(self, lesson_id: str, room_id: str) -> str:
        """Запуск нового урока"""
        try:
            # Создаем callback для отправки сообщений через WebSocket
            def callback(message):
                socketio.emit('lesson_update', message, room=room_id)
            
            # Запускаем урок
            session_id = await self.lesson_manager.start_lesson(
                lesson_id=lesson_id,
                room_id=room_id,
                callback=callback
            )
            
            # Сохраняем сессию
            self.active_sessions[session_id] = {
                'lesson_id': lesson_id,
                'room_id': room_id,
                'status': 'running'
            }
            
            logger.info(f"Lesson {lesson_id} started in room {room_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start lesson: {str(e)}")
            raise

    async def process_user_message(self, session_id: str, message: Dict):
        """Обработка сообщения от пользователя"""
        try:
            if session_id not in self.active_sessions:
                raise ValueError("Session not found")
                
            await self.lesson_manager.handle_user_response(session_id, message)
            logger.info(f"Processed user message in session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to process message: {str(e)}")
            raise

    async def generate_response(self, session_id: str, question: str) -> Dict:
        """Генерация ответа на вопрос"""
        try:
            if session_id not in self.active_sessions:
                raise ValueError("Session not found")
                
            session = self.active_sessions[session_id]
            context = {
                'lesson': session['lesson_id'],
                'subject': 'math',  # В реальности нужно получать из урока
                'current_phase': 'qa'
            }
            
            # Генерация текстового ответа
            text_response = self.llm.generate_response(question, context)
            
            # Генерация аудио и видео
            audio_path, duration = await self.tts.generate_speech_async(text_response)
            video_path = await self.wav2lip.process_async(
                audio_path=audio_path,
                face_path="static/reference_video.mp4"
            )
            
            return {
                'text': text_response,
                'audio_url': f"/static/audio/{Path(audio_path).name}",
                'video_url': f"/static/tmp/{Path(video_path).name}",
                'duration': duration
            }
            
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise

# Инициализация сервера
ai_server = AIServer()

# HTTP Endpoints
@app.route('/api/lessons', methods=['GET'])
def list_lessons():
    """Получение списка доступных уроков"""
    return jsonify({
        'lessons': [
            {'id': 'math_01', 'title': 'Введение в алгебру', 'subject': 'math'},
            {'id': 'physics_01', 'title': 'Основы механики', 'subject': 'physics'}
        ]
    })

@app.route('/api/start_lesson', methods=['POST'])
async def start_lesson():
    """Запуск урока"""
    data = request.json
    try:
        session_id = await ai_server.start_lesson(
            lesson_id=data['lesson_id'],
            room_id=data['room_id']
        )
        return jsonify({'success': True, 'session_id': session_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/generate_response', methods=['POST'])
async def generate_response():
    """Генерация ответа на вопрос"""
    data = request.json
    try:
        response = await ai_server.generate_response(
            session_id=data['session_id'],
            question=data['question']
        )
        return jsonify({'success': True, 'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# WebSocket Handlers
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('join_room')
async def handle_join_room(data):
    room_id = data['room_id']
    socketio.emit('system_message', {'text': f"New user joined room {room_id}"}, room=room_id)
    logger.info(f"User joined room {room_id}")

@socketio.on('user_message')
async def handle_user_message(data):
    try:
        await ai_server.process_user_message(
            session_id=data['session_id'],
            message={
                'type': 'question',
                'text': data['text'],
                'user_id': data.get('user_id')
            }
        )
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        socketio.emit('error', {'message': str(e)}, room=request.sid)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
