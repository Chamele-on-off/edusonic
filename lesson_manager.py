import os
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Callable
from pathlib import Path
from datetime import timedelta
import sqlite3
import time
from flask_socketio import emit

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lesson_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LessonPhaseType(Enum):
    GREETING = "greeting"
    EXPLANATION = "explanation"
    PRACTICE = "practice"
    QA = "qa"
    FAREWELL = "farewell"

@dataclass
class LessonPhase:
    type: LessonPhaseType
    content: str
    duration: int
    options: Optional[Dict] = None

@dataclass
class LessonConfig:
    id: str
    title: str
    description: str
    subject: str
    difficulty: str
    phases: List[LessonPhase]
    materials: List[str]

class LessonManager:
    def __init__(self, socketio, db_path: str = "materials/materials.db"):
        self.socketio = socketio
        self._active_lessons: Dict[str, dict] = {}
        self.lessons_dir = Path("static/lessons")
        self.lessons_dir.mkdir(parents=True, exist_ok=True)
        self._init_db(db_path)

    def _init_db(self, db_path: str):
        """Инициализация базы данных"""
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lesson_materials (
                    id TEXT PRIMARY KEY,
                    subject TEXT,
                    content TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()

    def start_lesson(self, lesson_id: str, room_id: str) -> str:
        """Запуск нового урока"""
        lesson_config = self._load_lesson_config(lesson_id)
        session_id = f"{lesson_id}_{room_id}_{int(time.time())}"
        
        self._active_lessons[session_id] = {
            'config': lesson_config,
            'room_id': room_id,
            'status': 'running',
            'current_phase': None,
            'start_time': time.time()
        }

        self.socketio.start_background_task(
            self._run_lesson_session,
            session_id
        )

        logger.info(f"Урок {lesson_id} запущен в комнате {room_id}")
        return session_id

    def _run_lesson_session(self, session_id: str):
        """Фоновая задача для выполнения урока"""
        session = self._active_lessons.get(session_id)
        if not session:
            return

        try:
            for phase in session['config'].phases:
                if session['status'] != 'running':
                    break

                session['current_phase'] = phase.type.value
                self._send_phase_update(session_id, phase)

                time.sleep(phase.duration)

                if phase.type == LessonPhaseType.QA:
                    self._handle_qa_session(session_id, phase)
                elif phase.type == LessonPhaseType.PRACTICE:
                    self._handle_practice_session(session_id, phase)

            self.stop_lesson(session_id)
        except Exception as e:
            logger.error(f"Ошибка в уроке {session_id}: {str(e)}")

    def _send_phase_update(self, session_id: str, phase: LessonPhase):
        """Отправка обновления фазы урока"""
        emit('lesson_update', {
            'type': 'phase_start',
            'session_id': session_id,
            'phase': phase.type.value,
            'content': phase.content,
            'duration': phase.duration
        }, room=self._active_lessons[session_id]['room_id'])

    def _handle_qa_session(self, session_id: str, phase: LessonPhase):
        """Обработка сессии вопросов-ответов"""
        timeout = phase.options.get("timeout", 300) if phase.options else 300
        time.sleep(timeout)

    def _handle_practice_session(self, session_id: str, phase: LessonPhase):
        """Обработка практической сессии"""
        for exercise in phase.options.get("exercises", []):
            if self._active_lessons[session_id]['status'] != 'running':
                break

            emit('lesson_update', {
                'type': 'exercise',
                'session_id': session_id,
                'exercise': exercise,
                'time_limit': exercise.get("time_limit", 60)
            }, room=self._active_lessons[session_id]['room_id'])

            time.sleep(exercise.get("time_limit", 60))

    def process_user_message(self, session_id: str, message: dict):
        """Обработка сообщения от пользователя"""
        if message['type'] == 'question':
            self._process_question(session_id, message)
        elif message['type'] == 'answer':
            self._process_answer(session_id, message)

    def _process_question(self, session_id: str, message: dict):
        """Обработка вопроса пользователя"""
        session = self._active_lessons.get(session_id)
        if not session:
            return

        emit('lesson_update', {
            'type': 'user_question',
            'session_id': session_id,
            'question': message['text'],
            'user_id': message.get('user_id')
        }, room=session['room_id'])

    def _process_answer(self, session_id: str, message: dict):
        """Обработка ответа пользователя"""
        is_correct = self._check_answer_correctness(message)
        session = self._active_lessons.get(session_id)
        if not session:
            return

        emit('lesson_update', {
            'type': 'feedback',
            'session_id': session_id,
            'is_correct': is_correct,
            'explanation': "Правильно!" if is_correct else "Попробуйте еще раз"
        }, room=session['room_id'])

    def _check_answer_correctness(self, message: dict) -> bool:
        """Проверка правильности ответа (заглушка)"""
        return True

    def stop_lesson(self, session_id: str):
        """Остановка урока"""
        if session_id in self._active_lessons:
            self._active_lessons[session_id]['status'] = 'stopped'
            del self._active_lessons[session_id]
            logger.info(f"Урок {session_id} остановлен")

    def _load_lesson_config(self, lesson_id: str) -> LessonConfig:
        """Загрузка конфигурации урока"""
        lesson_file = self.lessons_dir / f"{lesson_id}.json"
        
        if not lesson_file.exists():
            raise FileNotFoundError(f"Файл урока {lesson_file} не найден")

        with open(lesson_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        phases = [
            LessonPhase(
                type=LessonPhaseType(phase["type"]),
                content=phase["content"],
                duration=phase["duration"],
                options=phase.get("options")
            ) for phase in data["phases"]
        ]

        return LessonConfig(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            subject=data["subject"],
            difficulty=data["difficulty"],
            phases=phases,
            materials=data.get("materials", [])
        )

    def list_available_lessons(self) -> List[dict]:
        """Список доступных уроков"""
        lessons = []
        for lesson_file in self.lessons_dir.glob("*.json"):
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
                logger.error(f"Ошибка загрузки урока {lesson_file}: {str(e)}")
        return lessons