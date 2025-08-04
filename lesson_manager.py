import os
import json
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Callable
from pathlib import Path
from datetime import timedelta
import sqlite3
from concurrent.futures import ThreadPoolExecutor

# Настройка логирования
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
    duration: int  # в секундах
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
    def __init__(self, db_path: str = "materials/materials.db"):
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._active_lessons: Dict[str, LessonSession] = {}
        self.lessons_dir = Path("static/lessons")
        self.lessons_dir.mkdir(parents=True, exist_ok=True)
        self._init_db(db_path)
        
    def _init_db(self, db_path: str):
        """Инициализация базы данных с материалами"""
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

    async def start_lesson(self, lesson_id: str, room_id: str, callback: Callable) -> str:
        """Запуск нового урока"""
        lesson_config = self._load_lesson_config(lesson_id)
        session = LessonSession(lesson_config, room_id, callback)
        self._active_lessons[session.session_id] = session
        
        # Запускаем урок в фоновом режиме
        asyncio.create_task(self._run_lesson_session(session))
        
        return session.session_id

    async def _run_lesson_session(self, session: 'LessonSession'):
        """Основной цикл выполнения урока"""
        try:
            logger.info(f"Starting lesson session {session.session_id}")
            
            for phase in session.lesson_config.phases:
                if not session.is_active:
                    break
                    
                await self._execute_phase(session, phase)
                
            logger.info(f"Lesson session {session.session_id} completed")
        except Exception as e:
            logger.error(f"Error in lesson session {session.session_id}: {str(e)}")
        finally:
            self._cleanup_session(session.session_id)

    async def _execute_phase(self, session: 'LessonSession', phase: LessonPhase):
        """Выполнение одной фазы урока"""
        logger.info(f"Executing phase {phase.type} for session {session.session_id}")
        
        # Отправляем контент фазы
        await session.callback({
            "type": "phase_start",
            "session_id": session.session_id,
            "phase": phase.type.value,
            "content": phase.content,
            "duration": phase.duration
        })
        
        # Ожидаем завершения фазы или действий пользователя
        await asyncio.sleep(phase.duration)
        
        # Обработка специальных фаз
        if phase.type == LessonPhaseType.QA:
            await self._handle_qa_session(session, phase)
        elif phase.type == LessonPhaseType.PRACTICE:
            await self._handle_practice_session(session, phase)

    async def _handle_qa_session(self, session: 'LessonSession', phase: LessonPhase):
        """Обработка сессии вопросов-ответов"""
        qa_timeout = phase.options.get("timeout", 300) if phase.options else 300
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < qa_timeout and session.is_active:
            await asyncio.sleep(1)

    async def _handle_practice_session(self, session: 'LessonSession', phase: LessonPhase):
        """Обработка практической сессии"""
        exercises = phase.options.get("exercises", []) if phase.options else []
        
        for exercise in exercises:
            if not session.is_active:
                break
                
            await session.callback({
                "type": "exercise",
                "session_id": session.session_id,
                "exercise": exercise,
                "time_limit": exercise.get("time_limit", 60)
            })
            
            # Ожидаем ответ или таймаут
            await asyncio.sleep(exercise.get("time_limit", 60))

    async def handle_user_response(self, session_id: str, response: Dict):
        """Обработка ответа от пользователя"""
        if session_id not in self._active_lessons:
            logger.warning(f"Session {session_id} not found")
            return
            
        session = self._active_lessons[session_id]
        
        if response["type"] == "answer":
            await self._process_user_answer(session, response)
        elif response["type"] == "question":
            await self._process_user_question(session, response)

    async def _process_user_answer(self, session: 'LessonSession', response: Dict):
        """Обработка ответа на упражнение"""
        is_correct = True  # В реальной реализации нужно добавить проверку
        
        feedback = {
            "type": "feedback",
            "session_id": session.session_id,
            "is_correct": is_correct,
            "correct_answer": None if is_correct else "Правильный ответ",
            "explanation": "Отличная работа!" if is_correct else "Попробуйте еще раз"
        }
        
        await session.callback(feedback)

    async def _process_user_question(self, session: 'LessonSession', response: Dict):
        """Обработка вопроса от пользователя"""
        question = response["text"]
        context = {
            "lesson": session.lesson_config.title,
            "subject": session.lesson_config.subject,
            "current_phase": session.current_phase
        }
        
        answer = await self._generate_answer(question, context)
        
        response = {
            "type": "answer",
            "session_id": session.session_id,
            "question": question,
            "answer": answer,
            "is_relevant": True
        }
        
        await session.callback(response)

    async def _generate_answer(self, question: str, context: Dict) -> str:
        """Генерация ответа на вопрос с учетом контекста"""
        return f"Ответ на вопрос '{question}' в контексте {context['subject']}"

    def _load_lesson_config(self, lesson_id: str) -> LessonConfig:
        """Загрузка конфигурации урока из файла"""
        lesson_file = self.lessons_dir / f"{lesson_id}.json"
        
        if not lesson_file.exists():
            logger.error(f"Lesson file {lesson_file} not found")
            raise FileNotFoundError(f"Lesson file {lesson_file} not found")
        
        try:
            with open(lesson_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            phases = [
                LessonPhase(
                    type=LessonPhaseType(phase["type"]),
                    content=phase["content"],
                    duration=phase["duration"],
                    options=phase.get("options", {})
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
        except Exception as e:
            logger.error(f"Error loading lesson config: {str(e)}")
            raise

    def _cleanup_session(self, session_id: str):
        """Очистка ресурсов сессии"""
        if session_id in self._active_lessons:
            del self._active_lessons[session_id]
            logger.info(f"Session {session_id} cleaned up")

    async def stop_lesson(self, session_id: str):
        """Принудительная остановка урока"""
        if session_id in self._active_lessons:
            self._active_lessons[session_id].is_active = False
            logger.info(f"Lesson {session_id} stopped by request")

    def list_available_lessons(self) -> List[Dict]:
        """Получение списка доступных уроков"""
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
                logger.error(f"Error reading lesson file {lesson_file}: {str(e)}")
        return lessons

class LessonSession:
    def __init__(self, lesson_config: LessonConfig, room_id: str, callback: Callable):
        self.session_id = f"{lesson_config.id}_{room_id}_{os.urandom(4).hex()}"
        self.lesson_config = lesson_config
        self.room_id = room_id
        self.callback = callback
        self.is_active = True
        self.current_phase = None
        self.start_time = asyncio.get_event_loop().time()
        
    @property
    def elapsed_time(self) -> timedelta:
        return timedelta(seconds=asyncio.get_event_loop().time() - self.start_time)

# Пример использования
async def example_callback(message: Dict):
    print("Received message:", message)

async def main():
    manager = LessonManager()
    
    # Запускаем урок
    session_id = await manager.start_lesson(
        lesson_id="math_01",
        room_id="room_123",
        callback=example_callback
    )
    
    # Эмуляция пользовательского ввода
    await asyncio.sleep(10)
    await manager.handle_user_response(
        session_id=session_id,
        response={
            "type": "question",
            "text": "Что такое алгебра?",
            "user_id": "student_1"
        }
    )
    
    await asyncio.sleep(5)
    await manager.stop_lesson(session_id)

if __name__ == "__main__":
    asyncio.run(main())
