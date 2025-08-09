import os
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import Dict, Optional
from lesson_manager import LessonManager
from llm import LessonLLM
from tts import TextToSpeech
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

app = FastAPI()

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

# Создаем директории
Path("static/audio").mkdir(parents=True, exist_ok=True)
Path("static/tmp").mkdir(parents=True, exist_ok=True)
Path("materials").mkdir(parents=True, exist_ok=True)
Path("static/reference").mkdir(parents=True, exist_ok=True)

class AIServer:
    def __init__(self):
        self.llm = LessonLLM()
        self.tts = TextToSpeech()
        self.lesson_manager = LessonManager()
        self.active_sessions: Dict[str, Dict] = {}
        self.active_conferences: Dict[str, Dict] = {}
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._load_materials()
        logger.info("AI Server initialized successfully")

    def _load_materials(self):
        """Загрузка учебных материалов"""
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

    async def start_lesson(self, lesson_id: str, room_id: str, websocket: WebSocket) -> str:
        """Запуск нового урока с подключением к конференции"""
        try:
            # Запускаем урок
            session_id = await self.lesson_manager.start_lesson(
                lesson_id=lesson_id,
                room_id=room_id,
                callback=lambda msg: self._send_ws_message(websocket, msg)
            )
            
            # Сохраняем сессию
            self.active_sessions[session_id] = {
                'lesson_id': lesson_id,
                'room_id': room_id,
                'status': 'running',
                'start_time': time.time(),
                'peer_id': f"ai_teacher_{room_id}",
                'websocket': websocket
            }
            
            logger.info(f"Lesson {lesson_id} started in room {room_id}, session_id: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start lesson: {str(e)}")
            raise

    async def _send_ws_message(self, websocket: WebSocket, message: Dict):
        """Отправка сообщения через WebSocket"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send WS message: {str(e)}")

    async def connect_to_conference(self, room_id: str):
        """Подключение к видеоконференции"""
        try:
            if room_id in self.active_conferences:
                logger.info(f"Already connected to conference {room_id}")
                return True
                
            peer_id = f"ai_teacher_{room_id}"
            self.active_conferences[room_id] = {
                'peer_id': peer_id,
                'status': 'connecting',
                'connected_at': time.time()
            }
            
            logger.info(f"Connected to conference {room_id} with peer_id {peer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to conference: {str(e)}")
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
        """Генерация ответа на вопрос с аудио и фонемами"""
        try:
            if session_id not in self.active_sessions:
                raise ValueError("Session not found")
                
            session = self.active_sessions[session_id]
            context = {
                'lesson': session['lesson_id'],
                'subject': self.lesson_manager._load_lesson_config(session['lesson_id']).subject,
                'current_phase': 'qa'
            }
            
            # Генерация текстового ответа
            text_response = self.llm.generate_response(question, context)
            logger.info(f"Generated response for question: {question[:50]}...")
            
            # Генерация аудио и фонем
            audio_data, phonemes = await self.tts.generate_with_phonemes(text_response)
            
            return {
                'type': 'audio_phonemes',
                'text': text_response,
                'audio': audio_data,
                'phonemes': phonemes
            }
            
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise

    async def stop_lesson(self, session_id: str):
        """Остановка урока"""
        try:
            if session_id in self.active_sessions:
                room_id = self.active_sessions[session_id]['room_id']
                await self.lesson_manager.stop_lesson(session_id)
                del self.active_sessions[session_id]
                
                # Отключаемся от конференции, если больше нет активных уроков
                active_in_room = any(
                    s['room_id'] == room_id 
                    for s in self.active_sessions.values()
                )
                if not active_in_room and room_id in self.active_conferences:
                    await self.disconnect_from_conference(room_id)
                
                logger.info(f"Lesson {session_id} stopped successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to stop lesson: {str(e)}")
            raise

# Инициализация сервера
ai_server = AIServer()

# HTTP Endpoints
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("teacher.html", {"request": request})

@app.get("/api/lessons", response_class=JSONResponse)
async def list_lessons():
    """Получение списка доступных уроков"""
    try:
        lessons = ai_server.lesson_manager.list_available_lessons()
        return JSONResponse({
            'success': True,
            'lessons': lessons
        })
    except Exception as e:
        logger.error(f"Failed to list lessons: {str(e)}")
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

@app.post("/api/start_lesson", response_class=JSONResponse)
async def start_lesson(data: dict):
    """Запуск урока"""
    try:
        if not data or 'lesson_id' not in data or 'room_id' not in data:
            raise ValueError("Missing lesson_id or room_id in request")
            
        # Подключаемся к конференции
        await ai_server.connect_to_conference(data['room_id'])
        
        return JSONResponse({
            'success': True,
            'message': 'Use WebSocket /ws/teach to start lesson'
        })
    except Exception as e:
        logger.error(f"Error starting lesson: {str(e)}")
        return JSONResponse({
            'success': False,
            'error': str(e)
        }, status_code=400)

@app.post("/api/generate_response", response_class=JSONResponse)
async def generate_response(data: dict):
    """Генерация ответа на вопрос"""
    try:
        if not data or 'session_id' not in data or 'question' not in data:
            raise ValueError("Missing session_id or question in request")
            
        response = await ai_server.generate_response(
            session_id=data['session_id'],
            question=data['question']
        )
        return JSONResponse({
            'success': True,
            'response': response
        })
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return JSONResponse({
            'success': False,
            'error': str(e)
        }, status_code=400)

@app.post("/api/stop_lesson", response_class=JSONResponse)
async def stop_lesson(data: dict):
    """Остановка урока"""
    try:
        if not data or 'session_id' not in data:
            raise ValueError("Missing session_id in request")
            
        success = await ai_server.stop_lesson(data['session_id'])
        return JSONResponse({
            'success': success,
            'message': 'Lesson stopped' if success else 'Lesson not found'
        })
    except Exception as e:
        logger.error(f"Error stopping lesson: {str(e)}")
        return JSONResponse({
            'success': False,
            'error': str(e)
        }, status_code=400)

# WebSocket Endpoints
@app.websocket("/ws/teach")
async def websocket_teach(websocket: WebSocket):
    await websocket.accept()
    session_id = None
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get('type') == 'start_lesson':
                # Запуск нового урока
                session_id = await ai_server.start_lesson(
                    lesson_id=data['lesson_id'],
                    room_id=data['room_id'],
                    websocket=websocket
                )
                await websocket.send_json({
                    'type': 'session_started',
                    'session_id': session_id
                })
                
            elif data.get('type') == 'user_message' and session_id:
                # Обработка сообщения от пользователя
                await ai_server.process_user_message(
                    session_id=session_id,
                    message={
                        'type': 'question',
                        'text': data['text'],
                        'user_id': data.get('user_id', 'anonymous')
                    }
                )
                
            elif data.get('type') == 'stop_lesson' and session_id:
                # Остановка урока
                await ai_server.stop_lesson(session_id)
                session_id = None
                await websocket.send_json({
                    'type': 'session_stopped'
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
        if session_id and session_id in ai_server.active_sessions:
            await ai_server.stop_lesson(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.send_json({
            'type': 'error',
            'message': str(e)
        })
        if session_id and session_id in ai_server.active_sessions:
            await ai_server.stop_lesson(session_id)

# Очистка временных файлов при запуске
def cleanup_temp_files():
    import glob
    for file in glob.glob('static/tmp/*') + glob.glob('static/audio/*'):
        try:
            if os.path.isfile(file):
                os.unlink(file)
        except Exception as e:
            logger.warning(f"Could not delete {file}: {str(e)}")

@app.on_event("startup")
async def startup_event():
    cleanup_temp_files()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
