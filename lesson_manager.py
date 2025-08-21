import json
import logging
from pathlib import Path
from typing import Dict, List
from knowledge.dialogue_knowledge import DialogueKnowledge
from knowledge.society_knowledge import SocietyKnowledge

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LessonManager:
    def __init__(self, socketio):
        self.socketio = socketio
        self.lessons_dir = Path("lessons")
        self.dialogue_kb = DialogueKnowledge()
        self.society_kb = SocietyKnowledge()
        self.active_lessons = {}

    def start_lesson(self, lesson_id: str, room_id: str):
        """Запуск урока по указанному ID"""
        try:
            lesson = self._load_lesson(lesson_id)
            session_id = f"{lesson_id}-{room_id}"
            
            self.active_lessons[session_id] = {
                "lesson": lesson,
                "room_id": room_id,
                "current_phase": 0,
                "is_active": True
            }
            
            self._run_lesson(session_id)
            return True
        except Exception as e:
            logger.error(f"Ошибка запуска урока: {str(e)}")
            return False

    def _run_lesson(self, session_id: str):
        """Основной цикл выполнения урока"""
        session = self.active_lessons.get(session_id)
        if not session:
            return

        lesson = session["lesson"]
        
        for phase in lesson["phases"]:
            if not session["is_active"]:
                break
                
            self._process_phase(session_id, phase)

    def _process_phase(self, session_id: str, phase: Dict):
        """Обработка одной фазы урока"""
        session = self.active_lessons[session_id]
        
        # Отправка информации о фазе
        self.socketio.emit('lesson_phase', {
            "type": phase["type"],
            "content": phase["content"],
            "duration": phase.get("duration", 60)
        }, room=session["room_id"])

        # Обработка QA фазы
        if phase["type"] == "qa":
            self._handle_qa_session(session_id, phase.get("duration", 120))

    def _handle_qa_session(self, session_id: str, duration: int):
        """Обработка сессии вопросов-ответов"""
        # Реализация таймера и обработки вопросов
        pass

    def handle_question(self, room_id: str, question: str) -> str:
        """Обработка вопроса от ученика"""
        # 1. Проверка общих шаблонов
        response = self.dialogue_kb.get_response(question)
        if response:
            return response
            
        # 2. Поиск в предметной базе
        response = self.society_kb.find_answer(question)
        if response:
            return response
            
        # 3. Fallback
        return "Я уточню этот вопрос и отвечу на следующем занятии."

    def stop_lesson(self, session_id: str):
        """Принудительная остановка урока"""
        if session_id in self.active_lessons:
            self.active_lessons[session_id]["is_active"] = False
            del self.active_lessons[session_id]

    def _load_lesson(self, lesson_id: str) -> Dict:
        """Загрузка урока из JSON файла"""
        path = self.lessons_dir / f"{lesson_id}.json"
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_lessons(self) -> List[Dict]:
        """Получение списка доступных уроков"""
        lessons = []
        for lesson_file in self.lessons_dir.glob("*.json"):
            with open(lesson_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                lessons.append({
                    "id": data["id"],
                    "title": data["title"],
                    "subject": data["subject"]
                })
        return lessons
