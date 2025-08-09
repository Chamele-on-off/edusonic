import os
import logging
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from lesson_manager import LessonManager
from llm import LessonLLM
from tts import TextToSpeech
from webrtc_handler import WebRTCHandler
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import uuid
import time

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_teacher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Инициализация Flask и SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
socketio = SocketIO(app, cors_allowed_origins="*")

# Создание необходимых директорий
Path("static/audio").mkdir(parents=True, exist_ok=True)
Path("materials/lessons").mkdir(parents=True, exist_ok=True)
Path("static/reference").mkdir(parents=True, exist_ok=True)

class AITeacher:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.llm = LessonLLM()
        self.tts = TextToSpeech()
        self.webrtc = WebRTCHandler(socketio)
        self.lesson_manager = LessonManager(socketio)
        self.active_sessions = {}
        logger.info("AI Teacher initialized")

    def start_lesson(self, lesson_id: str, room_id: str) -> str:
        """Запуск нового урока с генерацией сессии"""
        session_id = f"{lesson_id}-{room_id}-{uuid.uuid4().hex[:8]}"
        
        self.active_sessions[session_id] = {
            'lesson_id': lesson_id,
            'room_id': room_id,
            'status': 'running',
            'start_time': time.time()
        }

        # Запуск в фоновом потоке
        self.executor.submit(
            self._run_lesson_session,
            session_id,
            lesson_id,
            room_id
        )
        
        return session_id

    def _run_lesson_session(self, session_id: str, lesson_id: str, room_id: str):
        """Основной цикл выполнения урока"""
        try:
            # Подключение к конференции
            self.webrtc.connect_to_conference(room_id)
            
            # Запуск урока через менеджер
            self.lesson_manager.start_lesson(lesson_id, room_id)
            
            logger.info(f"Lesson {lesson_id} started in session {session_id}")
            
        except Exception as e:
            logger.error(f"Lesson session {session_id} failed: {str(e)}")
            self.stop_lesson(session_id)

    def stop_lesson(self, session_id: str):
        """Корректная остановка урока"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['status'] = 'stopped'
            self.lesson_manager.stop_lesson(session_id)
            logger.info(f"Lesson {session_id} stopped")

    def process_user_message(self, session_id: str, message: dict) -> dict:
        """Обработка сообщений от пользователя (вопросы/ответы)"""
        if session_id not in self.active_sessions:
            return {'error': 'Session not found'}
        
        if message['type'] == 'question':
            return self._generate_ai_response(session_id, message['text'])
        elif message['type'] == 'answer':
            return self._validate_answer(session_id, message)

    def _generate_ai_response(self, session_id: str, question: str) -> dict:
        """Генерация ответа через LLM + TTS"""
        session = self.active_sessions.get(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        # Получаем контекст урока
        context = {
            'lesson': session['lesson_id'],
            'subject': self.lesson_manager.get_lesson_subject(session['lesson_id']),
            'current_phase': 'qa'
        }
        
        # Генерация текстового ответа
        text_response = self.llm.generate_response(question, context)
        
        # Синтез речи и фонем
        audio_data = self.tts.generate_speech(text_response)
        
        return {
            'type': 'ai_response',
            'text': text_response,
            'audio': audio_data['audio'],
            'phonemes': audio_data['phonemes'],
            'session_id': session_id
        }

    def _validate_answer(self, session_id: str, message: dict) -> dict:
        """Проверка ответа ученика (заглушка)"""
        return {
            'type': 'feedback',
            'is_correct': True,
            'explanation': 'Correct answer!',
            'session_id': session_id
        }

    def shutdown(self):
        """Корректное завершение работы"""
        self.executor.shutdown()
        self.webrtc.stop()
        self.tts.shutdown()
        logger.info("AI Teacher shutdown complete")

# Инициализация системы
ai_teacher = AITeacher()

# HTTP Endpoints
@app.route('/')
def index():
    return render_template('teacher.html')

@app.route('/api/lessons')
def list_lessons():
    try:
        lessons = ai_teacher.lesson_manager.list_available_lessons()
        return jsonify({'success': True, 'lessons': lessons})
    except Exception as e:
        logger.error(f"Failed to list lessons: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/start_lesson', methods=['POST'])
def start_lesson():
    try:
        data = request.get_json()
        session_id = ai_teacher.start_lesson(data['lesson_id'], data['room_id'])
        return jsonify({'success': True, 'session_id': session_id})
    except Exception as e:
        logger.error(f"Failed to start lesson: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/stop_lesson', methods=['POST'])
def stop_lesson():
    try:
        data = request.get_json()
        ai_teacher.stop_lesson(data['session_id'])
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Failed to stop lesson: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 400

# WebSocket Handlers
@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

@socketio.on('start_lesson')
def handle_start_lesson(data):
    try:
        session_id = ai_teacher.start_lesson(data['lesson_id'], data['room_id'])
        emit('session_started', {'session_id': session_id})
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('user_message')
def handle_user_message(data):
    try:
        response = ai_teacher.process_user_message(data['session_id'], data)
        emit('lesson_update', response)
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('stop_lesson')
def handle_stop_lesson(data):
    try:
        ai_teacher.stop_lesson(data['session_id'])
        emit('session_stopped', {})
    except Exception as e:
        emit('error', {'message': str(e)})

# Завершение работы
@app.teardown_appcontext
def shutdown_system(exception=None):
    ai_teacher.shutdown()

if __name__ == '__main__':
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        ai_teacher.shutdown()