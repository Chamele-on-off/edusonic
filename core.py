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

# Определяем корневую директорию проекта
BASE_DIR = Path(__file__).parent.resolve()

# Папки проекта
LESSONS_DIR = BASE_DIR / "lessons"
STUDENTS_DIR = BASE_DIR / "students"
STUDENT_PROGRESS_DIR = STUDENTS_DIR / "progress"
STATIC_DIR = BASE_DIR / "static"
FRAMES_DIR = STATIC_DIR / "frames"
CUSTOM_AVATARS_STUDENTS_DIR = STUDENTS_DIR / "avatars"

# Убедимся, что все нужные папки существуют
for folder in [
    LESSONS_DIR,
    STUDENTS_DIR,
    STUDENT_PROGRESS_DIR,
    STATIC_DIR,
    FRAMES_DIR,
    CUSTOM_AVATARS_STUDENTS_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

# ==============================
# 🔌 Flask и SocketIO инициализация
# ==============================

app = Flask(__name__, static_folder='static')
app.secret_key = 'ai-teacher-secret-key-2024'

# Отключаем встроенный логгер Flask для снижения шума
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Настройки SocketIO — threading для совместимости с блокирующими операциями (gTTS и т.д.)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=50 * 1024 * 1024  # 50 МБ
)

# ==============================
# 🌐 Глобальные переменные состояния комнат
# ==============================

# Блокировки для потокобезопасной инициализации DialogueManager на комнату
dialogue_init_locks = defaultdict(Lock)

# Флаги активации AI-учителя по комнатам
room_ai_activated = {}

# Хранение DialogueManager по комнатам (ленивая инициализация)
room_dialogue = defaultdict(lambda: None)

# Текущие аватары учителя по комнатам
room_current_avatar = defaultdict(lambda: 'teacher')

# Семафор для ограничения одновременных LLM-запросов (если используется)
llm_semaphore = Semaphore(5)  # максимум 5 параллельных запросов

# Флаг для отключения визуализации (для технических предметов и отладки)
VISUALIZATION_ENABLED = True

# Слайды включены по умолчанию
SLIDES_ENABLED = True

# ==============================
# 🔒 Вспомогательные функции
# ==============================

def debug_log(message: str):
    """Унифицированная функция логирования с префиксом времени и потока."""
    import threading
    from datetime import datetime
    thread_name = threading.current_thread().name
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{thread_name}] {message}", flush=True)

# ==============================
# 🛡️ Декораторы безопасности (если используются)
# ==============================

def teacher_required(f):
    """Декоратор для защиты админских эндпоинтов."""
    from functools import wraps
    from flask import request, jsonify

    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Временная заглушка: разрешаем всем (можно заменить на авторизацию)
        # В реальной системе — проверка токена или сессии
        return f(*args, **kwargs)
    return decorated_function

# ==============================
# 🚀 Финальная инициализация
# ==============================

debug_log("✅ Ядро приложения (core.py) инициализировано")
debug_log(f"📁 Базовая директория: {BASE_DIR}")
debug_log(f"📚 Папка уроков: {LESSONS_DIR}")
debug_log(f"👤 Папка учеников: {STUDENTS_DIR}")
debug_log(f"🖼️ Папка аватаров: {FRAMES_DIR}")
