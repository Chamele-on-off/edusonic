# core.py
import os
import sys
from pathlib import Path
from flask import Flask
from flask_socketio import SocketIO
from collections import defaultdict
from threading import Lock, Semaphore
import logging

# ==============================
# 🧠 Базовые настройки и пути
# ==============================

BASE_DIR = Path(__file__).parent.resolve()
LESSONS_DIR = BASE_DIR / "lessons"
STUDENTS_DIR = BASE_DIR / "students"
STUDENT_PROGRESS_DIR = STUDENTS_DIR / "progress"
STATIC_DIR = BASE_DIR / "static"
FRAMES_DIR = STATIC_DIR / "frames"
CUSTOM_AVATARS_STUDENTS_DIR = STUDENTS_DIR / "avatars"
USERS_DIR = STUDENTS_DIR / "users"

LESSONS_DEMO_DIR = LESSONS_DIR / "demo"
LESSONS_STUDENTS_DIR = LESSONS_DIR / "students"
LESSONS_GENERATED_DIR = LESSONS_DIR / "generated"

# Убедимся, что все нужные папки существуют
for folder in [
    LESSONS_DIR, LESSONS_DEMO_DIR, LESSONS_STUDENTS_DIR, LESSONS_GENERATED_DIR,
    STUDENTS_DIR, STUDENT_PROGRESS_DIR, STATIC_DIR, FRAMES_DIR,
    CUSTOM_AVATARS_STUDENTS_DIR, USERS_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

# ==============================
# 🔌 Flask и SocketIO инициализация
# ==============================

app = Flask(__name__, static_folder='static')
app.secret_key = 'ai-teacher-secret-key-2024'

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=50 * 1024 * 1024
)

# ==============================
# 🌐 Глобальные переменные состояния комнат
# ==============================

dialogue_init_locks = defaultdict(Lock)
room_ai_activated = {}
room_dialogue = defaultdict(lambda: None)
room_current_avatar = defaultdict(lambda: 'teacher')
room_participants = defaultdict(set)
room_student_data = {}
room_last_activity = {}

# Семафоры и флаги
llm_semaphore = Semaphore(5)
VISUALIZATION_ENABLED = True
SLIDES_ENABLED = True
TECHNICAL_SUPPORT_ENABLED = True
TECHNICAL_PROMPTS_ENABLED = True
LANGUAGE_SUPPORT_ENABLED = False

# ==============================
# 🔒 Вспомогательные функции
# ==============================

def debug_log(message: str):
    import threading
    from datetime import datetime
    thread_name = threading.current_thread().name
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{thread_name}] {message}", flush=True)

def _fast_room_initialization(room_id: str):
    """Быстрая инициализация комнаты без тяжёлых операций"""
    if room_id not in room_ai_activated:
        room_ai_activated[room_id] = False
    if room_id not in room_current_avatar:
        room_current_avatar[room_id] = 'teacher'
    debug_log(f"⚡ Быстрая инициализация комнаты: {room_id}")

# ==============================
# 🛡️ Декораторы безопасности
# ==============================

def teacher_required(f):
    from functools import wraps
    from flask import request, jsonify, session

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('is_teacher'):
            return f(*args, **kwargs)
        return jsonify({"success": False, "error": "Требуется авторизация учителя"}), 403
    return decorated_function

debug_log("✅ Ядро приложения (core.py) инициализировано")
