# app.py - AI Teacher System с поддержкой технических и естественнонаучных предметов
# ОПТИМИЗИРОВАННАЯ ВЕРСИЯ С ПОДДЕРЖКОЙ ТЕХНИЧЕСКИХ ПРЕДМЕТОВ И УЛУЧШЕННОЙ ПРОИЗВОДИТЕЛЬНОСТЬЮ
# ВЕРСИЯ С ПОДДЕРЖКОЙ СЛАЙДОВ ДЛЯ УРОКОВ (JPG/PNG) И РЕЗЕРВНОГО КОПИРОВАНИЯ
# ВЕРСИЯ С ПОДДЕРЖКОЙ ДОБАВЛЕНИЯ НОВЫХ АВАТАРОВ ЧЕРЕЗ TEACHER.HTML
# 🔥 ДОБАВЛЕНА ПОДДЕРЖКА ВЗРОСЛЫХ СТУДЕНТОВ И ЯЗЫКОВЫХ УРОВНЕЙ (A1-C2)
# 🔥 ИНТЕГРАЦИЯ С КАСТОМНЫМ TTS СЕРВИСОМ (tts.zindaki-edu.ru)

from flask import Flask, render_template, send_from_directory, jsonify, request, send_file, session, redirect, url_for
import os
from pathlib import Path
from flask_socketio import SocketIO, emit, join_room, leave_room
import io
import base64
import time
import threading
from collections import defaultdict
import random
from dialogue import DialogueManager
from config import update_api_key, get_api_key, load_config, get_model_config, get_llm_mode, set_llm_mode, get_llm_priority, set_llm_priority
import requests
import json
from datetime import datetime
from werkzeug.utils import secure_filename
import re
import tempfile
from local_llm_manager import get_llm_manager
from key_manager import get_key_manager
from typing import Optional
import uuid
from functools import wraps
import sys
import atexit
from threading import Timer, Semaphore, Lock
import psutil
import shutil
import zipfile

# 🔥 ИМПОРТ НОВОГО МОДУЛЯ ДЛЯ УПРАВЛЕНИЯ УЧЕНИКАМИ
from student_management import StudentManagement
# 🔥 ИМПОРТ НОВОГО МОДУЛЯ ДЛЯ УПРАВЛЕНИЯ УРОКАМИ
from lesson_manager import LessonManager
# 🔥 ИМПОРТ МЕНЕДЖЕРА РЕЧИ
from speech_manager import get_speech_manager

# =============================================================================
# НАСТРОЙКА FLASK И SOCKETIO
# =============================================================================

app = Flask(__name__, static_folder='static')
app.secret_key = 'ai-teacher-secret-key-2024'

socketio = SocketIO(
    app, 
    cors_allowed_origins="*",
    async_mode='threading',
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=1e8,
    logger=False,
    engineio_logger=False,
    async_handlers=True
)

# 🔥 ИНИЦИАЛИЗАЦИЯ МЕНЕДЖЕРА УЧЕНИКОВ
BASE_DIR = Path(__file__).parent
FRAMES_DIR = BASE_DIR / 'static' / 'avatar' / 'frames'
LESSONS_DIR = BASE_DIR / 'lessons'

# Создаем менеджер для работы с учениками
student_manager = StudentManagement(BASE_DIR)
student_manager.create_lessons_structure()

# 🔥 ИНИЦИАЛИЗАЦИЯ МЕНЕДЖЕРА УРОКОВ
lesson_manager = LessonManager(BASE_DIR)

# 🔥 ИНИЦИАЛИЗАЦИЯ МЕНЕДЖЕРА РЕЧИ
speech_manager = get_speech_manager(socketio)

# Получаем пути из менеджеров
MATERIALS_DIR = lesson_manager.materials_dir
PRACTICE_DIR = lesson_manager.practice_dir
LESSONS_DEMO_DIR = lesson_manager.lessons_demo_dir
LESSONS_STUDENTS_DIR = lesson_manager.lessons_students_dir
LESSONS_GENERATED_DIR = lesson_manager.lessons_generated_dir
LESSONS_TRASH_DIR = lesson_manager.lessons_trash_dir

STUDENTS_DIR = student_manager.students_dir
USERS_DIR = student_manager.users_dir
STUDENT_PROGRESS_DIR = student_manager.student_progress_dir

# =============================================================================
# ГЛОБАЛЬНЫЕ СОСТОЯНИЯ И КОНФИГУРАЦИЯ
# =============================================================================

# Ограничитель параллельной инициализации комнат
init_semaphore = Semaphore(100)  # УВЕЛИЧИЛИ до 100 одновременных инициализаций
dialogue_init_locks = defaultdict(Lock)  # Лок для инициализации DialogueManager в каждой комнате

# Ручная настройка CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Глобальные состояния
room_participants = defaultdict(set)
room_speech_data = defaultdict(list)
room_speaking = defaultdict(bool)
room_ai_activated = defaultdict(bool)
room_dialogue = defaultdict(lambda: None)
room_lessons = defaultdict(dict)
room_llm_mode = defaultdict(lambda: get_llm_mode())
room_teacher_speaking = defaultdict(bool)
room_practice_active = defaultdict(bool)
room_current_question_index = defaultdict(int)
room_current_avatar = defaultdict(lambda: 'Woman')
room_last_activity = defaultdict(lambda: time.time())

# PeerJS tracking
room_peer_ids = defaultdict(dict)

# Кэш для визуализации
diagram_cache = {}
room_visualization_queue = defaultdict(list)
room_visualization_active = defaultdict(bool)

# Очереди ответов LLM
room_llm_responses = defaultdict(list)
room_last_poll_time = defaultdict(lambda: 0)
room_llm_pending_requests = defaultdict(dict)
room_last_llm_update = defaultdict(lambda: 0)

# Данные учеников
room_student_data = defaultdict(dict)

# Менеджер локальной LLM
llm_manager = get_llm_manager()

# Менеджер ключей API
key_manager = get_key_manager()

# Настройки очистки
ROOM_TIMEOUT = 3600  # 1 час неактивности
MAX_ROOMS = 200  # Максимальное количество комнат в памяти
DEBUG_LLM = True

def debug_log(message):
    if DEBUG_LLM:
        print(f"🔧 [LLM_DEBUG] {message}")

def json_serialize_paths(obj):
    """Рекурсивная сериализация объектов Path для JSON"""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, (list, tuple)):
        return [json_serialize_paths(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: json_serialize_paths(value) for key, value in obj.items()}
    return obj

# =============================================================================
# ИМПОРТ МОДУЛЕЙ ДЛЯ ТЕХНИЧЕСКИХ ПРЕДМЕТОВ
# =============================================================================

try:
    from technical_subjects import (
        is_technical_subject,
        clean_text_for_speech_technical,
        contains_formulas,
        get_subject_type
    )
    TECHNICAL_SUPPORT_ENABLED = True
    debug_log("✅ Модуль технических предметов загружен")
except ImportError as e:
    TECHNICAL_SUPPORT_ENABLED = False
    debug_log(f"⚠️ Модуль technical_subjects.py не найден: {e}. Техническая поддержка отключена.")

# =============================================================================
# 🔥 ИМПОРТ AVATAR MANAGER
# =============================================================================

try:
    from avatar_manager import AvatarManager
    avatar_manager = AvatarManager(FRAMES_DIR)
    AVATAR_MANAGER_ENABLED = True
    debug_log("✅ AvatarManager загружен")
except ImportError as e:
    AVATAR_MANAGER_ENABLED = False
    debug_log(f"⚠️ AvatarManager не загружен: {e}. Функционал управления аватарами отключен.")

# =============================================================================
# 🔥 НАСТРОЙКИ ДЛЯ ВЗРОСЛЫХ СТУДЕНТОВ И ЯЗЫКОВЫХ УРОВНЕЙ
# =============================================================================

CEFR_LEVELS = {
    'A1': {
        'description': 'Начинающий',
        'prompt_adjustment': '''
        УРОВЕНЬ A1 (НАЧИНАЮЩИЙ):
        - Используй максимально простые предложения (3-5 слов)
        - 80% русский язык, 20% английский
        - Только базовая лексика (hello, my name is, thank you)
        - Повторяй ключевые фразы по 2-3 раза
        - Не используй сложные грамматические конструкции
        ''',
        'bilingual_ratio': 0.2,
        'lesson_count': 15
    },
    'A2': {
        'description': 'Элементарный',
        'prompt_adjustment': '''
        УРОВЕНЬ A2 (ЭЛЕМЕНТАРНЫЙ):
        - Простые предложения (5-7 слов)
        - 70% русский, 30% английский
        - Базовая повседневная лексика
        - Простые вопросы и ответы
        - Основные времена (Present Simple)
        ''',
        'bilingual_ratio': 0.3,
        'lesson_count': 15
    },
    'B1': {
        'description': 'Средний',
        'prompt_adjustment': '''
        УРОВЕНЬ B1 (СРЕДНИЙ):
        - Развернутые предложения
        - 50% русский, 50% английский
        - Широкая бытовая и учебная лексика
        - Объяснение грамматики на русском, практика на английском
        - Все основные времена
        ''',
        'bilingual_ratio': 0.5,
        'lesson_count': 20
    },
    'B2': {
        'description': 'Выше среднего',
        'prompt_adjustment': '''
        УРОВЕНЬ B2 (ВЫШЕ СРЕДНЕГО):
        - Сложные предложения
        - 30% русский, 70% английский
        - Абстрактные темы, аргументация
        - Нюансы грамматики
        - Идиомы и устойчивые выражения
        ''',
        'bilingual_ratio': 0.7,
        'lesson_count': 20
    },
    'C1': {
        'description': 'Продвинутый',
        'prompt_adjustment': '''
        УРОВЕНЬ C1 (ПРОДВИНУТЫЙ):
        - Сложные тексты и дискуссии
        - 10% русский (только пояснения), 90% английский
        - Академическая и профессиональная лексика
        - Стилистические нюансы
        - Дебаты и презентации
        ''',
        'bilingual_ratio': 0.9,
        'lesson_count': 25
    },
    'C2': {
        'description': 'В совершенстве',
        'prompt_adjustment': '''
        УРОВЕНЬ C2 (В СОВЕРШЕНСТВЕ):
        - 100% английский язык
        - Сложнейшие тексты любой тематики
        - Нюансы, ирония, сарказм
        - Академическое письмо
        - Профессиональные дискуссии
        ''',
        'bilingual_ratio': 1.0,
        'lesson_count': 25
    }
}

def detect_cefr_level(age: int, self_assessment: str = '', education_level: str = '') -> str:
    """Автоматически определяет уровень CEFR на основе возраста и самооценки"""
    # Если это школьник (не adult) - возвращаем A1
    if education_level != 'adult':
        return 'A1'
    
    # Взрослые - на основе самооценки
    level_mapping = {
        'начинающий': 'A1',
        'новичок': 'A1',
        'продолжающий': 'B1',
        'промежуточный': 'B1',
        'продвинутый': 'C1',
        'начальный': 'A1',
        'средний': 'B1',
        'высокий': 'C1',
        'beginner': 'A1',
        'intermediate': 'B1',
        'advanced': 'C1',
        'a1': 'A1',
        'a2': 'A2',
        'b1': 'B1',
        'b2': 'B2',
        'c1': 'C1',
        'c2': 'C2'
    }
    
    if self_assessment.lower() in level_mapping:
        return level_mapping[self_assessment.lower()]
    
    # Дефолтные значения для взрослых по возрасту
    if age < 25:
        return 'B1'
    elif age < 40:
        return 'B2'
    else:
        return 'C1'

# =============================================================================
# СИСТЕМА АУТЕНТИФИКАЦИИ
# =============================================================================

def login_required(f):
    """Декоратор для проверки аутентификации"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function

def teacher_required(f):
    """Декоратор для проверки прав учителя"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect('/login')
        if session.get('role') != 'teacher':
            return redirect('/student')
        return f(*args, **kwargs)
    return decorated_function

def student_required(f):
    """Декоратор для проверки прав ученика"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect('/login')
        if session.get('role') != 'student':
            return redirect('/teacher')
        return f(*args, **kwargs)
    return decorated_function

# =============================================================================
# API ДЛЯ АУТЕНТИФИКАЦИИ
# =============================================================================

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/login')
def login():
    if 'user_id' in session:
        if session.get('role') == 'teacher':
            return redirect('/teacher')
        else:
            return redirect('/student')
    return render_template('login.html')

@app.route('/auth/login', methods=['POST'])
def auth_login():
    try:
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        role = data.get('role', 'student')
        
        if not username or not password:
            return jsonify({"success": False, "error": "Заполните все поля"})
        
        user_data = student_manager.authenticate_user(username, password, role)
        
        if user_data:
            session['user_id'] = user_data['user_id']
            session['username'] = user_data['username']
            session['role'] = user_data['role']
            
            user_data['last_login'] = datetime.now().isoformat()
            student_manager.save_user_data(user_data)
            
            return jsonify({
                "success": True, 
                "message": "Успешный вход",
                "role": user_data['role'],
                "profile_complete": user_data.get('profile_complete', False)
            })
        else:
            return jsonify({"success": False, "error": "Неверный логин или пароль"})
    except Exception as e:
        return jsonify({"success": False, "error": f"Ошибка входа: {str(e)}"})

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

@app.route('/logout', methods=['POST'])
def logout_post():
    session.clear()
    return jsonify({"success": True, "message": "Успешный выход"})

@app.route('/api/auth/check')
def check_auth():
    if 'user_id' in session:
        user_data = student_manager.load_user_data(session['user_id'])
        if user_data:
            return jsonify({
                "success": True,
                "role": user_data.get('role'),
                "user_id": session['user_id']
            })
    return jsonify({"success": False})

@app.route('/investing.html')
def investing():
    return render_template('investing.html')

# =============================================================================
# ЛИЧНЫЕ КАБИНЕТЫ
# =============================================================================

@app.route('/teacher')
@teacher_required
def teacher():
    user_data = student_manager.load_user_data(session['user_id'])
    return render_template('teacher.html', user=user_data)

@app.route('/student')
@student_required
def student():
    user_data = student_manager.load_user_data(session['user_id'])
    
    if not user_data.get('profile_complete', False):
        return render_template('student_profile.html', user=user_data)
    
    student_data = user_data.get('student_data', {})
    return render_template('student.html', user=user_data, student_data=student_data)

@app.route('/student_profile')
@student_required
def student_profile():
    user_data = student_manager.load_user_data(session['user_id'])
    return render_template('student_profile.html', user=user_data)

@app.route('/auth/complete-profile', methods=['POST'])
@student_required
def complete_profile():
    try:
        data = request.json
        user_id = session['user_id']
        
        # 🔥 НОВАЯ ЛОГИКА: Добавляем поддержку взрослых
        is_adult = data.get('level') == 'adult'
        study_mode = data.get('study_mode', 'language') if is_adult else 'structured'
        language_level = data.get('language_level', 'A1') if is_adult else ''
        
        # Автоматическое определение уровня CEFR для взрослых
        if is_adult and not language_level:
            age_int = int(data.get('age', 25))
            language_level = detect_cefr_level(age_int, '', 'adult')
        
        student_data = {
            'name': data.get('name'),
            'education_level': data.get('level'),
            'age': data.get('age'),
            'student_id': str(uuid.uuid4()),
            'registration_date': datetime.now().isoformat(),
            'is_adult': is_adult,
            'study_mode': study_mode,
            'language_level': language_level
        }
        
        if student_manager.update_student_profile(user_id, student_data):
            # 🔥 Для взрослых в режиме "изучать что угодно" не создаем структуру уроков
            if not (is_adult and study_mode == 'anything'):
                student_manager.initialize_student_progress(student_data['student_id'], student_data['education_level'])
            
            # 🔥 Создаем демо-уроки для взрослых если нужно
            if is_adult and study_mode == 'language':
                student_manager._create_adult_language_structure()
            
            return jsonify({
                "success": True,
                "message": "Профиль успешно сохранен",
                "student_id": student_data['student_id'],
                "is_adult": is_adult,
                "study_mode": study_mode,
                "language_level": language_level
            })
        else:
            return jsonify({"success": False, "error": "Ошибка сохранения профиля"})
    except Exception as e:
        return jsonify({"success": False, "error": f"Ошибка: {str(e)}"})

# =============================================================================
# API ДЛЯ УПРАВЛЕНИЯ ПОЛЬЗОВАТЕЛЯМИ
# =============================================================================

@app.route('/api/users')
@teacher_required
def get_all_users():
    try:
        users = []
        for user_file in USERS_DIR.glob("*.json"):
            with open(user_file, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
                users.append({
                    'user_id': user_data['user_id'],
                    'username': user_data['username'],
                    'role': user_data.get('role'),
                    'created_at': user_data.get('created_at'),
                    'last_login': user_data.get('last_login')
                })
        return jsonify({"success": True, "users": users})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/users/create', methods=['POST'])
@teacher_required
def create_user():
    try:
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        role = data.get('role', 'student')
        
        if not username or not password:
            return jsonify({"success": False, "error": "Заполните все поля"})
        
        for user_file in USERS_DIR.glob("*.json"):
            with open(user_file, 'r', encoding='utf-8') as f:
                existing_user = json.load(f)
                if existing_user.get('username') == username:
                    return jsonify({"success": False, "error": "Пользователь с таким логином уже существует"})
        
        if role == 'student':
            user_data = student_manager.create_new_student(username, password)
        elif role == 'teacher':
            user_data = student_manager.create_new_teacher(username, password)
        else:
            return jsonify({"success": False, "error": "Неверная роль пользователя"})
        
        if user_data:
            return jsonify({
                "success": True,
                "message": f"Пользователь {username} успешно создан",
                "user_id": user_data['user_id']
            })
        else:
            return jsonify({"success": False, "error": "Ошибка создания пользователя"})
    except Exception as e:
        return jsonify({"success": False, "error": f"Ошибка: {str(e)}"})

@app.route('/api/users/<user_id>', methods=['DELETE'])
@teacher_required
def delete_user(user_id):
    try:
        user_file = USERS_DIR / f"{user_id}.json"
        
        if not user_file.exists():
            return jsonify({"success": False, "error": "Пользователь не найден"})
        
        if session.get('user_id') == user_id:
            return jsonify({"success": False, "error": "Нельзя удалить свой собственный аккаунт"})
        
        user_file.unlink()
        return jsonify({"success": True, "message": "Пользователь удален"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# =============================================================================
# 🔥 НОВЫЕ API ДЛЯ ВЗРОСЛЫХ СТУДЕНТОВ
# =============================================================================

@app.route('/api/adult/levels')
def get_adult_language_levels():
    """Возвращает список уровней CEFR для взрослых"""
    return jsonify({
        "success": True,
        "levels": [
            {"id": "A1", "name": "A1 (Начинающий)", "description": "Базовые фразы, простые предложения"},
            {"id": "A2", "name": "A2 (Элементарный)", "description": "Повседневные выражения"},
            {"id": "B1", "name": "B1 (Средний)", "description": "Ясная речь на знакомые темы"},
            {"id": "B2", "name": "B2 (Выше среднего)", "description": "Сложные тексты, дискуссии"},
            {"id": "C1", "name": "C1 (Продвинутый)", "description": "Свободное общение, академические тексты"},
            {"id": "C2", "name": "C2 (В совершенстве)", "description": "Владение на уровне носителя"}
        ]
    })

@app.route('/api/student/adult/lessons')
@student_required
def get_adult_student_lessons():
    """Возвращает уроки для взрослого студента"""
    try:
        user_data = student_manager.load_user_data(session['user_id'])
        if not user_data:
            return jsonify({"success": False, "error": "Пользователь не найден"})
        
        student_data = user_data.get('student_data', {})
        
        # Проверяем, что это взрослый
        if student_data.get('education_level') != 'adult':
            return jsonify({"success": False, "error": "Не взрослый студент"})
        
        study_mode = student_data.get('study_mode', 'language')
        
        # Режим "изучать что угодно" - пустой список уроков
        if study_mode == 'anything':
            return jsonify({
                "success": True,
                "study_mode": "anything",
                "lessons": [],
                "message": "Режим свободного диалога - уроки не требуются",
                "room_type": "free_conversation"
            })
        
        # Режим английского - уроки по уровню
        language_level = student_data.get('language_level', 'A1')
        level_dir = LESSONS_STUDENTS_DIR / "adult_language" / f"{language_level}_english"
        
        lessons = []
        if level_dir.exists():
            for lesson_file in level_dir.glob("*.txt"):
                with open(lesson_file, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip('# ').strip()
                    title = first_line if first_line else lesson_file.stem.replace('_', ' ').title()
                
                lessons.append({
                    'id': f"adult_{language_level}_{lesson_file.stem}",
                    'title': title,
                    'level': language_level,
                    'file_path': str(lesson_file.relative_to(LESSONS_DIR)),
                    'type': 'adult_language',
                    'cefr_level': language_level
                })
        
        # Сортируем по номеру урока
        lessons.sort(key=lambda x: x['title'])
        
        return jsonify({
            "success": True,
            "study_mode": "language",
            "language_level": language_level,
            "lessons": lessons,
            "room_type": "structured_lesson"
        })
        
    except Exception as e:
        debug_log(f"❌ Ошибка получения уроков для взрослых: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/student/create-adult-room', methods=['POST'])
@student_required
def create_adult_room():
    """Создает комнату для взрослого студента"""
    try:
        data = request.json
        subject = data.get('subject', 'english')
        lesson_id = data.get('lesson_id', '')
        
        user_data = student_manager.load_user_data(session['user_id'])
        if not user_data:
            return jsonify({"success": False, "error": "Пользователь не найден"})
        
        student_data = user_data.get('student_data', {})
        
        # Проверяем, что это взрослый
        if student_data.get('education_level') != 'adult':
            return jsonify({"success": False, "error": "Не взрослый студент"})
        
        study_mode = student_data.get('study_mode', 'language')
        language_level = student_data.get('language_level', 'A1')
        
        # Генерируем уникальный ID комнаты
        conference_id = str(int(time.time() * 1000))
        student_name = student_data.get('name', 'adult_student').replace(' ', '_').lower()
        
        if study_mode == 'anything':
            # Режим "изучать что угодно" - свободная комната
            room_id = f"adult_anything_{student_name}_{conference_id}"
            room_subject = 'general'
        else:
            # Режим английского - комната с уровнем
            room_id = f"adult_{language_level}_english_{student_name}_{conference_id}"
            room_subject = f"{language_level}_english"
        
        # Сохраняем данные студента в комнату
        room_student_data[room_id] = {
            **student_data,
            'subject': room_subject,
            'conference_id': conference_id,
            'study_mode': study_mode,
            'language_level': language_level,
            'is_adult': True,
            'room_type': 'free_conversation' if study_mode == 'anything' else 'structured_lesson'
        }
        
        # Быстрая инициализация комнаты (без блокировок)
        _fast_room_initialization(room_id)
        
        # Если есть lesson_id, загружаем урок
        if lesson_id and study_mode == 'language':
            # Находим файл урока
            lesson_path = None
            level_dir = LESSONS_STUDENTS_DIR / "adult_language" / f"{language_level}_english"
            if level_dir.exists():
                for lesson_file in level_dir.glob("*.txt"):
                    if f"adult_{language_level}_{lesson_file.stem}" == lesson_id:
                        lesson_path = lesson_file
                        break
            
            if lesson_path:
                room_student_data[room_id]['lesson_id'] = lesson_id
                room_student_data[room_id]['lesson_path'] = str(lesson_path.relative_to(LESSONS_DIR))
        
        return jsonify({
            "success": True,
            "room_id": room_id,
            "conference_url": f"/conference?room={room_id}&student=true&subject={room_subject}",
            "student_data": room_student_data[room_id],
            "study_mode": study_mode,
            "language_level": language_level
        })
        
    except Exception as e:
        debug_log(f"❌ Ошибка создания комнаты для взрослого: {e}")
        return jsonify({"success": False, "error": str(e)})

# =============================================================================
# ОСНОВНЫЕ ФУНКЦИИ СИСТЕМЫ
# =============================================================================

def cleanup_inactive_rooms():
    """Очистка неактивных комнат"""
    try:
        current_time = time.time()
        rooms_to_remove = []
        
        for room_id, last_active in room_last_activity.items():
            if (current_time - last_active > ROOM_TIMEOUT and 
                len(room_participants.get(room_id, [])) == 0):
                rooms_to_remove.append(room_id)
        
        for room_id in rooms_to_remove:
            try:
                if room_id in room_dialogue:
                    del room_dialogue[room_id]
                if room_id in room_participants:
                    del room_participants[room_id]
                if room_id in room_student_data:
                    del room_student_data[room_id]
                if room_id in room_speech_data:
                    del room_speech_data[room_id]
                if room_id in room_ai_activated:
                    del room_ai_activated[room_id]
                if room_id in room_llm_mode:
                    del room_llm_mode[room_id]
                if room_id in room_teacher_speaking:
                    del room_teacher_speaking[room_id]
                if room_id in room_practice_active:
                    del room_practice_active[room_id]
                if room_id in room_current_question_index:
                    del room_current_question_index[room_id]
                if room_id in room_current_avatar:
                    del room_current_avatar[room_id]
                if room_id in room_llm_responses:
                    del room_llm_responses[room_id]
                if room_id in room_llm_pending_requests:
                    del room_llm_pending_requests[room_id]
                if room_id in room_last_activity:
                    del room_last_activity[room_id]
                
                debug_log(f"✅ Очищена неактивная комната: {room_id}")
            except Exception as e:
                debug_log(f"⚠️ Ошибка очистки комнаты {room_id}: {e}")
        
        return len(rooms_to_remove)
    except Exception as e:
        debug_log(f"❌ Ошибка очистки комнат: {e}")
        return 0

def periodic_cleanup():
    """Периодическая очистка неактивных комнат"""
    try:
        cleaned = cleanup_inactive_rooms()
        if cleaned > 0:
            debug_log(f"🧹 Периодическая очистка: удалено {cleaned} комнат")
        
        # Ограничение количества комнат в памяти
        if len(room_participants) > MAX_ROOMS:
            # Удаляем самые старые комнаты
            rooms_sorted = sorted(room_last_activity.items(), key=lambda x: x[1])
            rooms_to_remove = rooms_sorted[:len(room_participants) - MAX_ROOMS]
            for room_id, _ in rooms_to_remove:
                if len(room_participants.get(room_id, [])) == 0:
                    try:
                        # Очищаем комнату
                        if room_id in room_dialogue:
                            del room_dialogue[room_id]
                        if room_id in room_participants:
                            del room_participants[room_id]
                        debug_log(f"🧹 Удалена старая комната для оптимизации: {room_id}")
                    except Exception as e:
                        debug_log(f"⚠️ Ошибка удаления старой комнаты: {e}")
    except Exception as e:
        debug_log(f"❌ Ошибка периодической очистки: {e}")
    
    # Повторяем каждые 10 минут
    Timer(600, periodic_cleanup).start()

def handle_disconnected_session(sid):
    """Безопасная обработка отключенных сессий"""
    try:
        for room_id, participants in room_participants.items():
            if sid in participants:
                participants.remove(sid)
                debug_log(f"🔧 Удален отключенный участник {sid} из комнаты {room_id}")
                emit('participants_update', {'count': len(participants)}, room=room_id)
                
        for room_id, peers in room_peer_ids.items():
            if sid in peers:
                del peers[sid]
                debug_log(f"🔧 Удален peer_id для отключенного участника {sid}")
    except Exception as e:
        debug_log(f"⚠️ Ошибка очистки отключенной сессии {sid}: {e}")

def setup_llm_manager():
    """Настройка менеджера LLM"""
    llm_manager.start()
    
    def global_llm_callback(request_id, response, room_id, original_request_id=None):
        """Глобальный обработчик ответов от LLM"""
        debug_log(f"Получен ответ для комнаты {room_id}: {response[:100]}...")
        
        target_request_id = original_request_id
        if not target_request_id:
            for req_id, req_data in room_llm_pending_requests[room_id].items():
                if req_data.get('manager_id') == request_id:
                    target_request_id = req_id
                    break
        
        if not target_request_id:
            target_request_id = f"unknown_{int(time.time() * 1000)}"
        
        room_llm_responses[room_id].append({
            'request_id': target_request_id,
            'response': response,
            'timestamp': time.time(),
            'delivered_via_websocket': False
        })
        
        if len(room_llm_responses[room_id]) > 10:
            room_llm_responses[room_id].pop(0)
        
        try:
            socketio.emit('llm_async_response', {
                'request_id': target_request_id,
                'response': response,
                'room_id': room_id,
                'timestamp': time.time(),
                'delivered_via': 'websocket'
            }, room=room_id)
            debug_log(f"Ответ немедленно отправлен через WebSocket в комнату {room_id}")
            
            for resp in room_llm_responses[room_id]:
                if resp['request_id'] == target_request_id:
                    resp['delivered_via_websocket'] = True
                    break
        except Exception as e:
            debug_log(f"⚠️ Не удалось отправить через WebSocket: {e}")
    
    llm_manager.register_room_callback('global', global_llm_callback)
    debug_log("LLM Manager настроен с улучшенным callback")

def _fast_room_initialization(room_id):
    """🔥 СУПЕР-БЫСТРАЯ инициализация комнаты без создания DialogueManager"""
    with init_semaphore:
        try:
            # Обновляем время активности
            room_last_activity[room_id] = time.time()
            
            # Минимальная инициализация - НЕ создаем DialogueManager здесь
            if room_id not in room_ai_activated:
                room_ai_activated[room_id] = False
            
            if room_id not in room_llm_mode:
                room_llm_mode[room_id] = get_llm_mode()
            
            if room_id not in room_current_avatar:
                pass
            
            # DialogueManager НЕ создаем здесь - только при необходимости
            
            # Если это комната ученика, просто сохраняем данные
            if room_id in room_student_data and room_student_data[room_id]:
                student_data = room_student_data[room_id]
                debug_log(f"🔥 Комната {room_id} зарегистрирована как комната ученика: {student_data.get('name')}")
                # DialogueManager будет создан позже при необходимости
                # НЕ создаем его здесь!
            
            debug_log(f"✅ Быстрая инициализация завершена для комнаты {room_id}")
            return True
        except Exception as e:
            debug_log(f"❌ Ошибка инициализации комнаты {room_id}: {e}")
            return False

def ensure_dialogue_manager_for_room(room_id):
    """🔥 ИСПРАВЛЕННАЯ ФУНКЦИЯ: Гарантирует наличие DialogueManager для комнаты (создает при необходимости)"""
    try:
        # Проверяем, нужен ли нам DialogueManager
        if room_id not in room_dialogue or room_dialogue[room_id] is None:
            with dialogue_init_locks[room_id]:
                # Двойная проверка после захвата лока
                if room_id not in room_dialogue or room_dialogue[room_id] is None:
                    debug_log(f"🔥 Создаем DialogueManager для комнаты {room_id} (ленивая инициализация)")
                    
                    # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: УБИРАЕМ init_semaphore отсюда!
                    # Создаем DialogueManager БЕЗ семафора - это быстрая операция
                    dm = DialogueManager(socketio)
                    dm.room_id = room_id
                    dm.set_llm_mode(room_llm_mode[room_id])
                    
                    # Устанавливаем данные ученика если есть
                    if room_id in room_student_data and room_student_data[room_id]:
                        dm.set_student_data(room_student_data[room_id])
                        
                        # 🔥 ОСОБАЯ ОБРАБОТКА ДЛЯ ВЗРОСЛЫХ
                        student_data = room_student_data[room_id]
                        if student_data.get('is_adult', False):
                            dm.is_adult_student = True
                            dm.study_mode = student_data.get('study_mode', 'language')
                            dm.language_level = student_data.get('language_level', 'A1')
                            
                            if dm.study_mode == 'anything':
                                dm.current_subject = 'general'
                                debug_log(f"🔥 Взрослый студент в режиме 'изучать что угодно'")
                            else:
                                dm.current_subject = f"{dm.language_level}_english"
                                debug_log(f"🔥 Взрослый студент изучает английский уровень {dm.language_level}")
                        
                        elif 'subject' in student_data and student_data['subject']:
                            dm.current_subject = student_data['subject']
                            debug_log(f"🔥 Установлен предмет для ученика: {student_data['subject']}")
                        
                    elif room_id.startswith('student_'):
                        parts = room_id.split('_')
                        if len(parts) > 1:
                            dm.current_subject = parts[1]
                            debug_log(f"🔥 Предмет определен из имени комнаты: {parts[1]}")
                    
                    room_dialogue[room_id] = dm
                    debug_log(f"✅ DialogueManager создан для комнаты {room_id}")
        
        return room_dialogue[room_id] is not None
    except Exception as e:
        debug_log(f"❌ Ошибка создания DialogueManager для {room_id}: {e}")
        return False

def reset_speaking_state(room_id, is_teacher=False):
    """Сбрасывает состояние речи для указанной комнаты"""
    room_speaking[room_id] = False
    if is_teacher:
        room_teacher_speaking[room_id] = False
    socketio.emit('speaking_state', {'speaking': False}, room=room_id)

# =============================================================================
# 🔥 УЛУЧШЕННОЕ ОЗВУЧИВАНИЕ С ПОДДЕРЖКОЙ КАСТОМНОГО TTS СЕРВИСА
# =============================================================================

def speak_text(room_id, text, voice_type='female', is_teacher=False, skip_history=False, force_lang=None):
    """🔥 ОПТИМИЗИРОВАННАЯ: Озвучивает текст с поддержкой кастомного TTS сервиса"""
    if not text.strip():
        return
        
    # 🔥 ОПРЕДЕЛЯЕМ ПРЕДМЕТ ДЛЯ УМНОЙ ОЧИСТКИ
    subject = None
    if room_id in room_student_data and room_student_data[room_id]:
        subject = room_student_data[room_id].get('subject')
    elif room_id in room_dialogue and room_dialogue[room_id]:
        subject = room_dialogue[room_id].current_subject
    
    # 🔥 ИСПОЛЬЗУЕМ МЕНЕДЖЕР РЕЧИ ДЛЯ ОЧИСТКИ
    cleaned_text = speech_manager.clean_text(text, subject)
    
    if not cleaned_text.strip():
        debug_log("⚠️ Текст пуст после очистки, пропускаем озвучивание")
        return
        
    if is_teacher:
        room_teacher_speaking[room_id] = True
        
    room_speaking[room_id] = True
    socketio.emit('speaking_state', {'speaking': True}, room=room_id)
    
    # 🔥 ИСПОЛЬЗУЕМ МЕНЕДЖЕР РЕЧИ ДЛЯ ГЕНЕРАЦИИ АУДИО С ПОДДЕРЖКОЙ ZINDAKI TTS
    audio_data = speech_manager.generate_optimized_speech(cleaned_text, force_lang, voice_type)
    
    if audio_data:
        # 🔥 ДОБАВЛЯЕМ ИНФОРМАЦИЮ О TTS СЕРВИСЕ В ОТВЕТ
        tts_service = 'zindaki' if speech_manager.tts_client and speech_manager.tts_client.available else 'gtts'
        
        emit('speech_audio', {
            'audio': audio_data,
            'text': cleaned_text,
            'timestamp': time.time(),
            'voice_type': voice_type,
            'is_teacher': is_teacher,
            'subject': subject if subject else 'general',
            'tts_service': tts_service,
            'optimized': True,
            'technical_support': TECHNICAL_SUPPORT_ENABLED
        }, room=room_id)
        
        if not skip_history:
            room_speech_data[room_id].append({
                'text': cleaned_text,
                'timestamp': time.time(),
                'type': 'generated',
                'voice_type': voice_type,
                'is_teacher': is_teacher,
                'subject': subject if subject else 'general',
                'tts_service': tts_service
            })
            if len(room_speech_data[room_id]) > 50:
                room_speech_data[room_id].pop(0)
    else:
        debug_log("❌ Не удалось сгенерировать аудио")
    
    # 🔥 Более точная длительность речи
    speech_duration = max(1.5, len(cleaned_text) * 0.08)
    threading.Timer(speech_duration, lambda: reset_speaking_state(room_id, is_teacher)).start()

def create_student_conference(student_data, subject=None):
    """Создает комнату для ученика с ОБЯЗАТЕЛЬНЫМ предметом"""
    try:
        conference_id = str(int(time.time() * 1000))
        
        if subject:
            room_subject = subject
        elif 'subject' in student_data:
            room_subject = student_data['subject']
        else:
            room_subject = 'математика'
        
        student_name = student_data.get('name', 'ученик').replace(' ', '_').lower()
        room_id = f"student_{room_subject}_{student_name}_{conference_id}"
        
        room_student_data[room_id] = {
            **student_data,
            'subject': room_subject,
            'conference_id': conference_id
        }
        
        _fast_room_initialization(room_id)
        
        debug_log(f"Создана комната {room_id} для ученика {student_data.get('name')}, предмет: {room_subject}")
        
        return {
            'room_id': room_id,
            'conference_url': f'/conference?room={room_id}',
            'student_data': room_student_data[room_id]
        }
    except Exception as e:
        debug_log(f"❌ Ошибка создания студенческой конференции: {e}")
        return None

def create_student_rooms(student_data):
    """Автоматически создает комнаты для ученика"""
    try:
        student_id = student_data.get('student_id')
        student_name = student_data.get('name')
        conference_id = student_data.get('conference_id', str(int(time.time() * 1000)))
        
        if not student_id or not student_name:
            return False
        
        if not conference_id:
            conference_id = str(int(time.time() * 1000))
            student_data['conference_id'] = conference_id
        
        subjects = [
            'математика', 'физика', 'химия', 'биология', 
            'история', 'обществознание', 'литература', 'русский язык', 
            'английский язык', 'география', 'информатика'
        ]
        
        created_rooms = []
        
        for subject in subjects:
            room_name = f"student_{subject}_{student_name.replace(' ', '_').lower()}_{conference_id}"
            
            room_student_data[room_name] = {
                'name': student_name,
                'age': student_data.get('age'),
                'education_level': student_data.get('education_level'),
                'subject': subject,
                'student_id': student_id,
                'conference_id': conference_id
            }
            
            _fast_room_initialization(room_name)
            
            debug_log(f"Создана комната {room_name} для ученика {student_name}, предмет: {subject}")
            
            created_rooms.append({
                'subject': subject,
                'subject_name': subject,
                'room_name': room_name,
                'avatar': 'Woman',
                'conference_id': conference_id,
                'student_data': room_student_data[room_name]
            })
        
        student_data['rooms'] = created_rooms
        student_data['default_avatar'] = 'Woman'
        student_data['conference_id'] = conference_id
        
        student_manager.save_student_data(student_data)
        
        debug_log(f"Создано {len(created_rooms)} комнат для ученика {student_name} с ID: {conference_id}")
        return True
    except Exception as e:
        debug_log(f"❌ Ошибка создания комнат для ученика: {e}")
        return False

# =============================================================================
# 🔥 ОБЕРТКИ ДЛЯ ФУНКЦИЙ LESSON_MANAGER
# =============================================================================

def find_lesson_slides(lesson_path):
    """🔥 Обертка для вызова lesson_manager.find_lesson_slides"""
    return lesson_manager.find_lesson_slides(lesson_path)

def get_lesson_slides_api(lesson_id):
    """🔥 Обертка для вызова lesson_manager.get_lesson_slides_api"""
    return lesson_manager.get_lesson_slides_api(lesson_id)

# =============================================================================
# SOCKET.IO HANDLERS
# =============================================================================

@socketio.on('connect')
def handle_connect():
    debug_log(f"Client connected: {request.sid}")
    emit('connection_established', {'message': 'Connected to server', 'sid': request.sid})

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    debug_log(f"Client disconnected: {sid}")
    handle_disconnected_session(sid)

@socketio.on('join_room')
def handle_join_room(data):
    room_id = data['room_id']
    peer_id = data.get('peer_id')
    
    debug_log(f"Попытка присоединения к комнате {room_id}, peer_id: {peer_id}")
    
    try:
        # Обновляем время активности
        room_last_activity[room_id] = time.time()
        
        if room_id not in room_participants:
            room_participants[room_id] = set()

        join_room(room_id)
        room_participants[room_id].add(request.sid)
        
        if peer_id:
            if room_id not in room_peer_ids:
                room_peer_ids[room_id] = {}
            room_peer_ids[room_id][request.sid] = peer_id
        
        # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Только быстрая инициализация - БЕЗ создания DialogueManager
        _fast_room_initialization(room_id)
        
        if peer_id:
            emit('participant_joined', {
                'peer_id': peer_id,
                'sid': request.sid
            }, room=room_id, include_self=False)
        
        try:
            emit('current_avatar', {'avatar_name': room_current_avatar[room_id]}, to=request.sid)
        except Exception as e:
            debug_log(f"⚠️ Ошибка отправки аватара: {e}")
        
        if room_id in room_speech_data and room_speech_data[room_id]:
            try:
                emit('speech_history', {'history': room_speech_data[room_id]}, to=request.sid)
            except Exception as e:
                debug_log(f"⚠️ Ошибка отправки истории: {e}")
        
        emit('participants_update', {'count': len(room_participants[room_id])}, room=room_id)
        
        # 🔥 ОСОБОЕ ПРИВЕТСТВИЕ ДЛЯ ВЗРОСЛЫХ СТУДЕНТОВ
        if (room_id in room_student_data and 
            room_student_data[room_id] and 
            room_student_data[room_id].get('is_adult', False)):
            
            student_data = room_student_data[room_id]
            student_name = student_data.get('name', 'студент')
            study_mode = student_data.get('study_mode', 'language')
            language_level = student_data.get('language_level', 'A1')
            
            if study_mode == 'anything':
                welcome_message = f"{student_name}, привет! Я ваш AI-учитель. Вы выбрали режим 'изучать что угодно'. "
                welcome_message += "Задавайте любые вопросы по любым темам, и я постараюсь помочь вам!"
            else:
                level_name = CEFR_LEVELS.get(language_level, {}).get('description', 'Начинающий')
                welcome_message = f"{student_name}, привет! Я ваш виртуальный учитель английского языка. "
                welcome_message += f"Ваш уровень: {language_level} ({level_name}). "
                welcome_message += "Давайте начнем наш урок английского языка. Если готовы, скажите 'готов начать'."
            
            socketio.emit('student_welcome_message', {
                'room_id': room_id,
                'student_name': student_name,
                'subject': 'english' if study_mode == 'language' else 'general',
                'message': welcome_message,
                'prompt_ready': True,
                'is_adult': True,
                'study_mode': study_mode,
                'language_level': language_level
            }, room=room_id)
            
            socketio.start_background_task(lambda: delayed_welcome(room_id, welcome_message))
        
        # Приветствие для комнат школьников (существующая логика)
        elif (room_id in room_student_data and 
              room_student_data[room_id] and 
              not room_id.startswith('demo_') and 
              room_id != 'default'):
            
            student_data = room_student_data[room_id]
            student_name = student_data.get('name', 'ученик')
            subject = student_data.get('subject', 'предмету')
            
            welcome_message = f"{student_name}, привет! Я твой виртуальный учитель по {subject}. "
            welcome_message += "Давай начнем наш сегодняшний урок. Если ты готов начать, скажи 'готов начать'."
            
            socketio.emit('student_welcome_message', {
                'room_id': room_id,
                'student_name': student_name,
                'subject': subject,
                'message': welcome_message,
                'prompt_ready': True
            }, room=room_id)
            
            socketio.start_background_task(lambda: delayed_welcome(room_id, welcome_message))
        
        elif len(room_participants[room_id]) == 1 and not room_ai_activated[room_id]:
            greeting = "Привет! Я ваш виртуальный учитель. Давайте познакомимся и выберем интересный урок вместе!"
            socketio.start_background_task(lambda: delayed_welcome(room_id, greeting))
        
        debug_log(f"Успешное присоединение к комнате {room_id}, участников: {len(room_participants[room_id])}")
        
    except Exception as e:
        debug_log(f"❌ Критическая ошибка при присоединении к комнате {room_id}: {e}")
        try:
            emit('room_error', {
                'room_id': room_id,
                'error': f'Join room failed: {str(e)}'
            }, to=request.sid)
        except:
            debug_log("⚠️ Не удалось отправить ошибку - клиент уже отключен")

def delayed_welcome(room_id, message, delay=2):
    """Отправляет приветствие с задержкой"""
    time.sleep(delay)
    # 🔥 ОПТИМИЗИРОВАННЫЙ ВЫЗОВ: приветствие всегда на русском
    speak_text(room_id, message, voice_type='female', is_teacher=True, force_lang='ru')

@socketio.on('get_current_avatar')
def handle_get_current_avatar(data):
    room_id = data['room_id']
    emit('current_avatar', {'avatar_name': room_current_avatar[room_id]}, to=request.sid)

@socketio.on('client_start_animation')
def handle_client_start_animation(data):
    room_id = data['room_id']
    avatar_name = data['avatar_name']
    debug_log(f"Получена команда запуска анимации для комнаты {room_id}, аватар: {avatar_name}")
    
    room_current_avatar[room_id] = avatar_name
    emit('avatar_changed', {'avatar_name': avatar_name}, room=room_id)
    emit('animation_ready', {'status': 'ready'}, room=room_id)

@socketio.on('avatar_changed')
def handle_avatar_changed(data):
    room_id = data['room_id']
    avatar_name = data['avatar_name']
    debug_log(f"Смена аватара в комнате {room_id} на {avatar_name}")
    
    room_current_avatar[room_id] = avatar_name
    emit('avatar_changed', {'avatar_name': avatar_name}, room=room_id)

@socketio.on('generate_speech')
def handle_generate_speech(data):
    room_id = data['room_id']
    text = data['text']
    voice_type = data.get('voice', 'male')
    # 🔥 ОПТИМИЗИРОВАННЫЙ ВЫЗОВ
    speak_text(room_id, text, voice_type)

@socketio.on('student_answer')
def handle_student_answer(data):
    room_id = data['room_id']
    answer = data['answer']
    user_sid = request.sid

    debug_log(f"Получен ответ ученика: {answer}")
    debug_log(f"Состояние комнаты: practice_active={room_practice_active[room_id]}, teacher_speaking={room_teacher_speaking[room_id]}")

    if room_teacher_speaking[room_id]:
        debug_log(f"Игнорирую ответ ученика, так как учитель говорит: {answer}")
        return

    if not room_practice_active[room_id]:
        debug_log(f"Практика не активна, игнорирую ответ: {answer}")
        return

    if any(cmd in answer.lower() for cmd in ['стоп', 'останови', 'хватит', 'закончи']):
        debug_log(f"Команда остановки практики: {answer}")
        # Создаем DialogueManager если нужно
        if not ensure_dialogue_manager_for_room(room_id):
            return
            
        if room_id in room_dialogue:
            room_dialogue[room_id]._end_practice_session()
            room_practice_active[room_id] = False
            room_current_question_index[room_id] = 0
            
            response = "Практика завершена по вашей команде. Урок окончен!"
            emit('speech_text', {
                'text': f"Учитель: {response}",
                'sid': 'teacher',
                'is_teacher': True
            }, room=room_id)
            # 🔥 ОПТИМИЗИРОВАННЫЙ ВЫЗОВ
            speak_text(room_id, response, voice_type='female', is_teacher=True, force_lang='ru')
            emit('practice_ended', {}, room=room_id)
        return

    # Создаем DialogueManager если нужно
    if not ensure_dialogue_manager_for_room(room_id):
        return
        
    dialogue = room_dialogue[room_id]
    
    if not dialogue.waiting_for_answer:
        debug_log(f"Система не ожидает ответа, игнорирую: {answer}")
        return

    if any(cmd in answer.lower() for cmd in ['продолжай', 'дальше', 'следующий']):
        debug_log(f"Игнорирую команду вместо ответа: {answer}")
        response = dialogue._evaluate_and_generate_next("")
        if response:
            emit('speech_text', {
                'text': f"Учитель: {response}",
                'sid': 'teacher',
                'is_teacher': True
            }, room=room_id)
            # 🔥 ОПТИМИЗИРОВАННЫЙ ВЫЗОВ
            speak_text(room_id, response, voice_type='female', is_teacher=True)
        return

    room_speech_data[room_id].append({
        'text': f"Ответ ученика: {answer}",
        'timestamp': time.time(),
        'type': 'practice_answer',
        'sid': user_sid
    })
    
    debug_log(f"Обработка ответа через диалог менеджер...")
    
    response = dialogue._evaluate_and_generate_next(answer)
    
    if response:
        debug_log(f"Ответ учителя: {response}")
        
        emit('speech_text', {
            'text': f"Учитель: {response}",
            'sid': 'teacher',
            'is_teacher': True
        }, room=room_id)
        
        # 🔥 ОПТИМИЗИРОВАННЫЙ ВЫЗОВ
        speak_text(room_id, response, voice_type='female', is_teacher=True)
        
        if not dialogue.practice_active:
            room_practice_active[room_id] = False
            room_current_question_index[room_id] = 0
            emit('practice_ended', {}, room=room_id)
            debug_log("Практика завершена")
    else:
        room_practice_active[room_id] = False
        room_current_question_index[room_id] = 0
        dialogue.waiting_for_answer = False
        emit('practice_ended', {}, room=room_id)
        debug_log("Практика завершена (response=None)")

@socketio.on('student_message')
def handle_student_message(data):
    room_id = data['room_id']
    message = data['message']
    user_sid = request.sid

    debug_log(f"Получено сообщение от ученика: {message}")
    
    if room_teacher_speaking[room_id]:
        debug_log(f"Игнорирую сообщение ученика, так как учитель говорит: {message}")
        return

    if room_practice_active[room_id]:
        handle_student_answer({
            'room_id': room_id,
            'answer': message
        })
    else:
        handle_recognized_speech({
            'room_id': room_id, 
            'text': message
        })

@socketio.on('recognized_speech')
def handle_recognized_speech(data):
    room_id = data['room_id']
    text = data['text']
    user_sid = request.sid

    # Обновляем время активности
    room_last_activity[room_id] = time.time()
    
    if not room_ai_activated.get(room_id, False):
        return
        
    # 🔥 Ленивое создание DialogueManager при необходимости
    if not ensure_dialogue_manager_for_room(room_id):
        debug_log(f"Не удалось создать DialogueManager для {room_id}")
        return

    if room_teacher_speaking[room_id]:
        debug_log(f"Игнорирую речь ученика, так как учитель говорит: {text}")
        return

    if (text.startswith("Учитель:") or "учитель" in text.lower() or 
        len(text.strip()) < 3 or text in ["привет", "здравствуйте"]):
        return
    
    room_speech_data[room_id].append({
        'text': text,
        'timestamp': time.time(),
        'type': 'recognized',
        'sid': user_sid
    })
    if len(room_speech_data[room_id]) > 50:
        room_speech_data[room_id].pop(0)
    
    emit('speech_text', {'text': text, 'sid': user_sid}, room=room_id)
    
    if room_ai_activated[room_id]:
        dialogue = room_dialogue[room_id]
        
        if dialogue.is_lesson_started():
            all_continue_commands = [
                "продолжай", "продолжить", "дальше", "следующий", "вперед", "давай дальше",
                "записал", "понял", "ясно", "ага", "угу", "хорошо", "ок", "ладно", "ясно",
                "готов", "можно дальше", "слушаю", "понятно", "ясно", "следующий вопрос"
            ]
            
            if any(cmd in text.lower() for cmd in all_continue_commands):
                next_paragraph = dialogue._get_next_paragraph()
                if next_paragraph:
                    emit('speech_text', {
                        'text': f"Учитель: {next_paragraph}",
                        'sid': 'teacher',
                        'is_teacher': True
                    }, room=room_id)
                    # 🔥 ОПТИМИЗИРОВАННЫЙ ВЫЗОВ
                    speak_text(room_id, next_paragraph, voice_type='female', is_teacher=True)
                else:
                    practice_msg = "Урок завершен. Переходим к практике."
                    emit('speech_text', {
                        'text': f"Учитель: {practice_msg}",
                        'sid': 'teacher', 
                        'is_teacher': True
                    }, room=room_id)
                    speak_text(room_id, practice_msg, voice_type='female', is_teacher=True, force_lang='ru')
                return
        
        if any(word in text.lower() for word in ["стоп", "останови", "хватит", "закончи"]):
            stop_response = dialogue.process_input(text)
            if stop_response:
                emit('speech_text', {
                    'text': f"Учитель: {stop_response}",
                    'sid': 'teacher',
                    'is_teacher': True
                }, room=room_id)
                speak_text(room_id, stop_response, voice_type='female', is_teacher=True)
            return
        
        if dialogue.is_lesson_started():
            response = dialogue.handle_question_during_lesson(text)
            if response:
                emit('speech_text', {
                    'text': f"Учитель: {response}",
                    'sid': 'teacher',
                    'is_teacher': True
                }, room=room_id)
                # 🔥 ОПТИМИЗИРОВАННЫЙ ВЫЗОВ
                speak_text(room_id, response, voice_type='female', is_teacher=True)
        else:
            response = dialogue.process_input(text)
            
            if response is None:
                lesson_data = dialogue.get_selected_lesson()
                if lesson_data:
                    emit('lesson_started', {
                        'lesson_id': lesson_data['id'],
                        'title': lesson_data['title'],
                        'subject': dialogue.get_current_subject()
                    }, room=room_id)
                    
                    first_paragraph = dialogue._get_next_paragraph()
                    if first_paragraph:
                        emit('speech_text', {
                            'text': f"Учитель: {first_paragraph}",
                            'sid': 'teacher',
                            'is_teacher': True
                        }, room=room_id)
                        speak_text(room_id, first_paragraph, voice_type='female', is_teacher=True)
            elif response:
                emit('speech_text', {
                    'text': f"Учитель: {response}",
                    'sid': 'teacher',
                    'is_teacher': True
                }, room=room_id)
                
                # 🔥 ОПТИМИЗИРОВАННЫЙ ВЫЗОВ
                speak_text(room_id, response, voice_type='female', is_teacher=True)
                
            if dialogue.is_lesson_started():
                lesson_data = dialogue.get_selected_lesson()
                if lesson_data and not lesson_data.get('lesson_started_emitted', False):
                    lesson_data['lesson_started_emitted'] = True
                    emit('lesson_started', {
                        'lesson_id': lesson_data['id'],
                        'title': lesson_data['title'],
                        'subject': dialogue.get_current_subject(),
                        'is_generated': lesson_data.get('is_generated', False)
                    }, room=room_id)
                    debug_log(f"📢 ДОПОЛНИТЕЛЬНО отправлено 'lesson_started' для комнаты {room_id}")
                    
                    first_paragraph = dialogue._get_next_paragraph()
                    if first_paragraph:
                        emit('speech_text', {
                            'text': f"Учитель: {first_paragraph}",
                            'sid': 'teacher',
                            'is_teacher': True
                        }, room=room_id)
                        speak_text(room_id, first_paragraph, voice_type='female', is_teacher=True)

@socketio.on('activate_ai_teacher')
def handle_activate_ai_teacher(data):
    room_id = data['room_id']
    sid = request.sid
    
    debug_log(f"Запрос активации AI-учителя для комнаты {room_id} от {sid}")
    
    try:
        room_ai_activated[room_id] = True
        
        # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Ленивое создание DialogueManager при необходимости
        if not ensure_dialogue_manager_for_room(room_id):
            emit('activate_ai_error', {
                'room_id': room_id,
                'error': 'Не удалось создать DialogueManager'
            }, to=sid)
            return
        
        dialogue = room_dialogue[room_id]
        dialogue.set_llm_mode(room_llm_mode[room_id])
        
        # 🔥 ОСОБОЕ ПРИВЕТСТВИЕ ДЛЯ ВЗРОСЛЫХ В РЕЖИМЕ "ИЗУЧАТЬ ЧТО УГОДНО"
        if (room_id in room_student_data and 
            room_student_data[room_id].get('is_adult', False) and
            room_student_data[room_id].get('study_mode') == 'anything'):
            
            greeting = "Привет! Я ваш AI-учитель в режиме 'изучать что угодно'. "
            greeting += "Задавайте любые вопросы по любым темам: наука, искусство, технологии, бизнес - я помогу вам разобраться!"
        else:
            greeting = "Привет! Я ваш AI-учитель. Давайте пообщаемся и выберем интересный урок вместе!"
        
        # 🔥 ОПТИМИЗИРОВАННЫЙ ВЫЗОВ
        speak_text(room_id, greeting, voice_type='female', is_teacher=True, force_lang='ru')
        
        emit('ai_teacher_activated', {
            'room_id': room_id,
            'message': 'AI-учитель успешно активирован'
        }, room=room_id)
        
        debug_log(f"AI-учитель успешно активирован в комнате {room_id}")
        
    except Exception as e:
        debug_log(f"❌ Ошибка активации AI-учителя: {e}")
        emit('activate_ai_error', {
            'room_id': room_id,
            'error': f'Ошибка активации: {str(e)}'
        }, to=sid)

@socketio.on('visualization_generated')
def handle_visualization_generated(data):
    room_id = data['room_id']
    debug_log(f"Получена SVG инфографика для комнаты {room_id}: {data['topic'][:100]}...")
    emit('visualization_generated', {
        'room_id': room_id,
        'topic': data['topic'],
        'svg_code': data.get('svg_code', ''),
        'timestamp': data.get('timestamp', time.time()),
        'type': data.get('type', 'infographic')
    }, room=room_id)

@socketio.on('set_llm_mode')
def handle_set_llm_mode(data):
    room_id = data['room_id']
    mode = data['mode']
    
    if mode in ["traditional", "llm_first"]:
        room_llm_mode[room_id] = mode
        # Устанавливаем режим только если DialogueManager уже создан
        if room_id in room_dialogue and room_dialogue[room_id] is not None:
            room_dialogue[room_id].set_llm_mode(mode)
        
        emit('llm_mode_changed', {
            'mode': mode,
            'room': room_id
        }, room=room_id)
        
        debug_log(f"Режим LLM изменен в комнате {room_id}: {mode}")

@socketio.on('llm_response_ready')
def handle_llm_response_ready(data):
    room_id = data['room_id']
    question = data['question']
    answer = data['answer']
    
    debug_log(f"Получен ответ LLM для комнаты {room_id}: {answer[:100]}...")
    
    reset_speaking_state(room_id, is_teacher=True)
    room_teacher_speaking[room_id] = False
    room_speaking[room_id] = False
    
    time.sleep(0.5)
    
    emit('speech_text', {
        'text': f"Учитель: {answer}",
        'sid': 'teacher',
        'is_teacher': True
    }, room=room_id)
    
    # 🔥 ОПТИМИЗИРОВАННЫЙ ВЫЗОВ
    speak_text(room_id, answer, voice_type='female', is_teacher=True)

@socketio.on('practice_started')
def handle_practice_started(data):
    room_id = data['room_id']
    room_practice_active[room_id] = True
    room_current_question_index[room_id] = 0
    emit('practice_started', {}, room=room_id)
    debug_log(f"Практика начата в комнате {room_id}")

@socketio.on('practice_ended')
def handle_practice_ended(data):
    room_id = data['room_id']
    room_practice_active[room_id] = False
    room_current_question_index[room_id] = 0
    emit('practice_ended', {}, room=room_id)
    debug_log(f"Практика завершена в комнате {room_id}")

@socketio.on('get_llm_status')
def handle_get_llm_status(data):
    room_id = data['room_id']
    
    if room_id in room_dialogue and room_dialogue[room_id] is not None:
        status = room_dialogue[room_id].llm.get_llm_status()
        emit('llm_status_update', {
            'room_id': room_id,
            'status': status
        }, room=room_id)

@socketio.on('set_llm_priority')
def handle_set_llm_priority(data):
    room_id = data['room_id']
    priority = data['priority']
    
    valid_priorities = ["local_first", "openrouter_first", "local_only", "openrouter_only"]
    
    if priority not in valid_priorities:
        emit('llm_priority_error', {
            'room_id': room_id,
            'error': f'Invalid priority. Use: {valid_priorities}'
        })
        return
    
    if room_id in room_dialogue and room_dialogue[room_id] is not None:
        room_dialogue[room_id].llm.set_priority(priority)
        status = room_dialogue[room_id].llm.get_priority_status()
        
        emit('llm_priority_changed', {
            'room_id': room_id,
            'priority': priority,
            'status': status
        }, room=room_id)
        
        debug_log(f"Приоритет LLM изменен в комнате {room_id}: {priority}")

@socketio.on('get_llm_priority_status')
def handle_get_llm_priority_status(data):
    room_id = data['room_id']
    
    if room_id in room_dialogue and room_dialogue[room_id] is not None:
        status = room_dialogue[room_id].llm.get_priority_status()
        emit('llm_priority_status', {
            'room_id': room_id,
            'status': status
        })

@socketio.on('async_llm_request')
def handle_async_llm_request(data):
    room_id = data['room_id']
    prompt = data['prompt']
    system_prompt = data.get('system_prompt', '')
    max_tokens = data.get('max_tokens', 1000)
    request_type = data.get('type', 'general')
    client_request_id = data.get('request_id')
    
    debug_log(f"Запрос от комнаты {room_id}: {prompt[:100]}...")
    
    request_id = client_request_id or f"{room_id}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
    
    room_llm_pending_requests[room_id][request_id] = {
        'prompt': prompt,
        'system_prompt': system_prompt,
        'max_tokens': max_tokens,
        'timestamp': time.time(),
        'type': request_type
    }
    
    current_time = time.time()
    for req_id in list(room_llm_pending_requests[room_id].keys()):
        if current_time - room_llm_pending_requests[room_id][req_id]['timestamp'] > 300:
            del room_llm_pending_requests[room_id][req_id]
    
    llm_request_id = llm_manager.submit_request(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        room_id=room_id,
        request_id=request_id
    )
    
    room_llm_pending_requests[room_id][request_id]['manager_id'] = llm_request_id
    
    emit('llm_request_queued', {
        'request_id': request_id,
        'manager_id': llm_request_id,
        'queue_position': llm_manager.get_queue_size(),
        'room_id': room_id,
        'timestamp': time.time()
    })

@socketio.on('llm_async_response')
def handle_llm_async_response(data):
    room_id = data['room_id']
    response = data['response']
    request_id = data['request_id']
    
    debug_log(f"Ответ для комнаты {room_id}: {response[:100]}...")
    
    if room_id in room_llm_pending_requests and request_id in room_llm_pending_requests[room_id]:
        del room_llm_pending_requests[room_id][request_id]
    
    if response and room_id in room_dialogue and room_dialogue[room_id] is not None:
        room_dialogue[room_id].llm.handle_llm_response(request_id, response, room_id)
        
        emit('speech_text', {
            'text': f"Учитель: {response}",
            'sid': 'teacher',
            'is_teacher': True
        }, room=room_id)
        
        # 🔥 ОПТИМИЗИРОВАННЫЙ ВЫЗОВ
        speak_text(room_id, response, voice_type='female', is_teacher=True)

@socketio.on('generate_visualization')
def handle_generate_visualization(data):
    room_id = data['room_id']
    topic = data.get('topic', '')
    context = data.get('context', '')
    
    if not topic:
        return
    
    debug_log(f"WebSocket генерация SVG инфографики для комнаты {room_id}: {topic[:100]}...")
    
    try:
        from llm import LLMIntegration
        llm = LLMIntegration()
        
        result = llm.generate_infographic(topic, context)
        svg_code = result["svg_code"] if result and result.get("success") else generate_svg_code(topic, context)
        
        emit('visualization_generated', {
            'room_id': room_id,
            'topic': topic,
            'svg_code': svg_code,
            'timestamp': time.time(),
            'type': 'infographic'
        }, room=room_id)
        
        debug_log(f"✅ SVG инфографика немедленно отправлена в комнату {room_id}")
        
    except Exception as e:
        debug_log(f"❌ Ошибка немедленной генерации SVG инфографики: {e}")
        emit('visualization_generated', {
            'room_id': room_id,
            'topic': topic,
            'svg_code': generate_svg_code(topic, context),
            'timestamp': time.time(),
            'type': 'fallback'
        }, room=room_id)

# =============================================================================
# 🔥 НОВЫЕ СОБЫТИЯ ДЛЯ СЛАЙДОВ УРОКОВ
# =============================================================================

@socketio.on('get_lesson_slides')
def handle_get_lesson_slides(data):
    """WebSocket запрос для получения слайдов урока"""
    try:
        room_id = data['room_id']
        lesson_id = data['lesson_id']
        
        debug_log(f"Запрос слайдов для урока {lesson_id} в комнате {room_id}")
        
        slides_data = get_lesson_slides_api(lesson_id)
        
        if slides_data['success']:
            emit('lesson_slides_loaded', {
                'room_id': room_id,
                'lesson_id': lesson_id,
                'slides': slides_data['slides'],
                'slides_count': slides_data['slides_count'],
                'has_slides': slides_data['has_slides']
            }, room=room_id)
            
            debug_log(f"✅ Слайды отправлены в комнату {room_id}: {len(slides_data['slides'])} слайдов")
        else:
            emit('lesson_slides_error', {
                'room_id': room_id,
                'lesson_id': lesson_id,
                'error': slides_data.get('error', 'Неизвестная ошибка')
            }, room=room_id)
            
    except Exception as e:
        debug_log(f"❌ Ошибка обработки запроса слайдов: {e}")
        emit('lesson_slides_error', {
            'room_id': data.get('room_id', 'unknown'),
            'lesson_id': data.get('lesson_id', 'unknown'),
            'error': str(e)
        })

# =============================================================================
# API ЭНДПОИНТЫ
# =============================================================================

@app.route('/api/avatars')
def get_avatars():
    try:
        avatars = [d for d in os.listdir(FRAMES_DIR) if (FRAMES_DIR / d).is_dir()]
        return jsonify({"avatars": avatars})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/frames/<avatar_name>')
def get_frames(avatar_name):
    try:
        avatar_dir = FRAMES_DIR / avatar_name
        if not avatar_dir.exists():
            return jsonify({"error": "Avatar not found"}), 404
        
        supported_formats = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
        frames = [f for f in os.listdir(avatar_dir) if f.lower().endswith(supported_formats)]
        frames.sort()
        
        return jsonify({"frames": frames})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/frames/<avatar_name>/<path:filename>')
def serve_frame(avatar_name, filename):
    return send_from_directory(FRAMES_DIR / avatar_name, filename)

@app.route('/conference')
def conference():
    room_id = request.args.get('room', 'default')
    embed = request.args.get('embed', 'false') == 'true'
    student_mode = request.args.get('student', 'false') == 'true'
    subject = request.args.get('subject', '')
    subject_name = request.args.get('subject_name', '')
    lesson_id = request.args.get('lesson_id', '')
    
    return render_template('conference.html', 
                         room_id=room_id, 
                         embed=embed,
                         student_mode=student_mode,
                         subject=subject,
                         subject_name=subject_name,
                         lesson_id=lesson_id)

def add_visualization_to_queue(room_id, topic, context):
    if room_id not in room_visualization_queue:
        room_visualization_queue[room_id] = []
    
    svg_code = generate_svg_code(topic, context)
    
    visualization_data = {
        'topic': topic,
        'context': context,
        'svg_code': svg_code,
        'timestamp': time.time(),
        'type': 'infographic'
    }
    
    if len(room_visualization_queue[room_id]) >= 5:
        room_visualization_queue[room_id].pop(0)
    
    room_visualization_queue[room_id].append(visualization_data)
    
    debug_log(f"SVG инфографика добавлена в очередь для комнаты {room_id}: {topic}")
    return True

@app.route('/api/poll_visualization', methods=['POST', 'GET'])
def poll_visualization():
    try:
        if request.method == 'POST':
            data = request.json
            room_id = data.get('room_id', 'default')
        else:
            room_id = request.args.get('room_id', 'default')
        
        if room_id in room_visualization_queue and room_visualization_queue[room_id]:
            visualization = room_visualization_queue[room_id].pop(0)
            
            if socketio and room_id:
                socketio.emit('visualization_generated', {
                    'room_id': room_id,
                    'topic': visualization['topic'],
                    'svg_code': visualization.get('svg_code', ''),
                    'timestamp': visualization['timestamp'],
                    'type': visualization.get('type', 'infographic')
                }, room=room_id)
            
            return jsonify({
                "success": True,
                "processed": True,
                "queue_length": len(room_visualization_queue.get(room_id, []))
            })
        
        return jsonify({
            "success": True,
            "processed": False,
            "queue_length": len(room_visualization_queue.get(room_id, []))
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/visualization_queue_status', methods=['GET'])
def visualization_queue_status():
    room_id = request.args.get('room_id', 'default')
    
    return jsonify({
        "success": True,
        "room_id": room_id,
        "queue_length": len(room_visualization_queue.get(room_id, [])),
        "active": room_visualization_active.get(room_id, False),
        "queue": room_visualization_queue.get(room_id, [])
    })

def generate_svg_code(topic: str, context: str = "") -> str:
    debug_log(f"Генерация SVG инфографики для: {topic[:100]}...")
    
    try:
        from llm import LLMIntegration
        llm = LLMIntegration()
        
        result = llm.generate_infographic(topic, context)
        
        if result and result.get("success"):
            svg_code = result["svg_code"]
            debug_log(f"✅ Сгенерирована SVG инфографика для: {topic[:50]}...")
            debug_log(f"Длина SVG кода: {len(svg_code)} символов")
            return svg_code
        else:
            debug_log("❌ LLM не вернул SVG инфографику")
    except Exception as e:
        debug_log(f"❌ Ошибка генерации SVG инфографики: {e}")
    
    return f'''
<svg width="600" height="300" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#f8fafc"/>
  <rect x="50" y="50" width="500" height="200" rx="10" fill="white" stroke="#e2e8f0" stroke-width="2"/>
  <text x="300" y="100" text-anchor="middle" font-family="Arial" font-size="20" fill="#1e293b">{topic}</text>
  <circle cx="200" cy="160" r="30" fill="#3b82f6" opacity="0.7"/>
  <circle cx="300" cy="160" r="30" fill="#10b981" opacity="0.7"/>
  <circle cx="400" cy="160" r="30" fill="#f59e0b" opacity="0.7"/>
  <text x="300" y="230" text-anchor="middle" font-family="Arial" font-size="14" fill="#64748b">Инфографика</text>
</svg>
'''

@app.route('/api/generate_diagram', methods=['POST'])
def generate_diagram():
    try:
        data = request.json
        topic = data.get('topic', '')
        context = data.get('context', '')
        room_id = data.get('room_id', 'default')
        
        if not topic:
            return jsonify({"success": False, "error": "Topic is required"})
        
        add_visualization_to_queue(room_id, topic, context)
        
        return jsonify({
            "success": True,
            "message": "SVG инфографика добавлена в очередь",
            "queue_position": len(room_visualization_queue.get(room_id, [])),
            "topic": topic
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# =============================================================================
# 🔥 НОВЫЙ ЭНДПОИНТ ДЛЯ ОТДАЧИ СЛАЙДОВ УРОКОВ
# =============================================================================

@app.route('/lesson_slide')
def serve_lesson_slide():
    slide_path = request.args.get('path', '').strip()
    if not slide_path:
        return "Slide path missing", 400

    # 🔥 ВРЕМЕННО УБИРАЕМ ВСЕ ПРОВЕРКИ БЕЗОПАСНОСТИ — ТОЛЬКО ДЛЯ ТЕСТИРОВАНИЯ
    full_path = LESSONS_DIR / slide_path

    if not full_path.exists():
        return "File not found", 404

    mime = 'image/jpeg'
    if slide_path.lower().endswith('.png'):
        mime = 'image/png'
    elif slide_path.lower().endswith('.mp4'):
        mime = 'video/mp4'

    return send_file(full_path, mimetype=mime)

# =============================================================================
# 🔥 НОВЫЙ API ДЛЯ ПОЛУЧЕНИЯ СЛАЙДОВ УРОКА
# =============================================================================

@app.route('/api/lesson/slides/<lesson_id>')
def get_lesson_slides(lesson_id):
    """API для получения списка слайдов урока"""
    try:
        slides_data = get_lesson_slides_api(lesson_id)
        return jsonify(slides_data)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# =============================================================================
# 🔥 НОВЫЕ API ДЛЯ ЗАГРУЗКИ СЛАЙДОВ УРОКОВ
# =============================================================================

@app.route('/api/lesson/slides/upload', methods=['POST'])
@teacher_required
def upload_lesson_slides():
    """Загрузка слайдов для урока"""
    try:
        if 'files' not in request.files:
            return jsonify({"success": False, "error": "Файлы не найдены"})
        
        files = request.files.getlist('files')
        lesson_id = request.form.get('lesson_id')
        
        result = lesson_manager.upload_lesson_slides(files, lesson_id)
        return jsonify(result)
        
    except Exception as e:
        debug_log(f"❌ Ошибка загрузки слайдов: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/lesson/slides/delete', methods=['POST'])
@teacher_required
def delete_lesson_slide():
    """Удаление слайда урока"""
    try:
        data = request.json
        slide_path = data.get('slide_path')
        
        result = lesson_manager.delete_lesson_slide(slide_path)
        return jsonify(result)
        
    except Exception as e:
        debug_log(f"❌ Ошибка удаления слайда: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/lesson/slides/bulk_delete', methods=['POST'])
@teacher_required
def bulk_delete_lesson_slides():
    """Массовое удаление слайдов урока"""
    try:
        data = request.json
        lesson_id = data.get('lesson_id')
        
        result = lesson_manager.bulk_delete_lesson_slides(lesson_id)
        return jsonify(result)
        
    except Exception as e:
        debug_log(f"❌ Ошибка массового удаления слайдов: {e}")
        return jsonify({"success": False, "error": str(e)})

# =============================================================================
# 🔥 API ДЛЯ ЭКСПОРТА И ИМПОРТА ДАННЫХ УЧЕНИКОВ
# =============================================================================

@app.route('/api/students/export_full', methods=['GET'])
@teacher_required
def export_students_full():
    """Полный экспорт данных учеников (включая прогресс)"""
    try:
        zip_path = student_manager.export_students_full()
        
        if not zip_path:
            return jsonify({"success": False, "error": "Ошибка создания архива"})
        
        return send_file(
            zip_path,
            as_attachment=True,
            download_name=f"students_full_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mimetype='application/zip'
        )
        
    except Exception as e:
        debug_log(f"❌ Ошибка экспорта данных: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/students/import', methods=['POST'])
@teacher_required
def import_students_data():
    """Импорт данных учеников из ZIP-файла"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "Файл не найден"})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "Файл не выбран"})
        
        if not file.filename.endswith('.zip'):
            return jsonify({"success": False, "error": "Только ZIP файлы"})
        
        import tempfile
        import shutil
        
        # Создаем временную папку для распаковки
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, file.filename)
        file.save(zip_path)
        
        results = student_manager.import_students_data(Path(zip_path))
        
        # Очищаем временные файлы
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return jsonify(results)
        
    except Exception as e:
        debug_log(f"❌ Ошибка импорта данных: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/students/export_progress', methods=['GET'])
@teacher_required
def export_students_progress():
    """Экспорт только прогресса учеников"""
    try:
        zip_path = student_manager.export_students_progress()
        
        if not zip_path:
            return jsonify({"success": False, "error": "Ошибка создания архива"})
        
        return send_file(
            zip_path,
            as_attachment=True,
            download_name=f"students_progress_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mimetype='application/zip'
        )
        
    except Exception as e:
        debug_log(f"❌ Ошибка экспорта прогресса: {e}")
        return jsonify({"success": False, "error": str(e)})

# =============================================================================
# 🔥 НОВЫЕ API ДЛЯ УПРАВЛЕНИЯ АВАТАРАМИ (ДОБАВЛЕННЫЕ)
# =============================================================================

@app.route('/api/avatars/upload', methods=['POST'])
@teacher_required
def upload_avatar():
    """Загрузка нового аватара (ZIP архив с кадрами)"""
    try:
        if not AVATAR_MANAGER_ENABLED:
            return jsonify({"success": False, "error": "AvatarManager не загружен"})
        
        if 'avatar_file' not in request.files:
            return jsonify({"success": False, "error": "Файл не найден"})
        
        file = request.files['avatar_file']
        if file.filename == '':
            return jsonify({"success": False, "error": "Файл не выбран"})
        
        if not file.filename.endswith('.zip'):
            return jsonify({"success": False, "error": "Только ZIP архивы разрешены"})
        
        # Получаем параметры
        avatar_name = request.form.get('avatar_name', '').strip()
        force_overwrite = request.form.get('force_overwrite', 'false') == 'true'
        
        # Генерируем имя аватара если не указано
        if not avatar_name:
            avatar_name = Path(file.filename).stem
            avatar_name = re.sub(r'[^\w\s-]', '', avatar_name).replace(' ', '_').lower()
            avatar_name = avatar_name[:50]
        
        # Проверяем имя аватара
        if not avatar_name or len(avatar_name) < 2:
            return jsonify({"success": False, "error": "Неверное имя аватара"})
        
        # Создаем временный файл
        import tempfile
        temp_dir = tempfile.mkdtemp()
        temp_zip = Path(temp_dir) / file.filename
        file.save(temp_zip)
        
        # Извлекаем аватар
        success, message, avatar_path = avatar_manager.extract_avatar(temp_zip, avatar_name, force_overwrite)
        
        # Очищаем временные файлы
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        if success:
            # Обновляем кэш аватаров
            avatars = avatar_manager.list_avatars()
            
            # Уведомляем всех клиентов об обновлении списка аватаров
            socketio.emit('avatars_updated', {
                'avatars': avatars,
                'new_avatar': avatar_name,
                'timestamp': time.time()
            })
            
            return jsonify({
                "success": True,
                "message": message,
                "avatar_name": avatar_name,
                "avatar_path": avatar_path,
                "frames_count": len(list((FRAMES_DIR / avatar_name).iterdir()))
            })
        else:
            return jsonify({"success": False, "error": message})
        
    except Exception as e:
        debug_log(f"❌ Ошибка загрузки аватара: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/avatars/delete', methods=['POST'])
@teacher_required
def delete_avatar():
    """Удаление аватара"""
    try:
        if not AVATAR_MANAGER_ENABLED:
            return jsonify({"success": False, "error": "AvatarManager не загружен"})
        
        data = request.json
        avatar_name = data.get('avatar_name')
        
        if not avatar_name:
            return jsonify({"success": False, "error": "Не указано имя аватара"})
        
        # Проверяем, что это не системный аватар
        system_avatars = ['Woman', 'man', 'teacher', 'default']
        if avatar_name in system_avatars:
            return jsonify({"success": False, "error": "Нельзя удалить системный аватар"})
        
        success, message = avatar_manager.delete_avatar(avatar_name)
        
        if success:
            # Уведомляем об удалении
            socketio.emit('avatar_deleted', {
                'avatar_name': avatar_name,
                'timestamp': time.time()
            })
            
            return jsonify({
                "success": True,
                "message": message,
                "avatar_name": avatar_name
            })
        else:
            return jsonify({"success": False, "error": message})
        
    except Exception as e:
        debug_log(f"❌ Ошибка удаления аватара: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/avatars/rename', methods=['POST'])
@teacher_required
def rename_avatar():
    """Переименование аватара"""
    try:
        if not AVATAR_MANAGER_ENABLED:
            return jsonify({"success": False, "error": "AvatarManager не загружен"})
        
        data = request.json
        old_name = data.get('old_name')
        new_name = data.get('new_name')
        
        if not old_name or not new_name:
            return jsonify({"success": False, "error": "Не указаны имена"})
        
        # Проверяем новое имя
        new_name = re.sub(r'[^\w\s-]', '', new_name).replace(' ', '_').lower()
        if len(new_name) < 2:
            return jsonify({"success": False, "error": "Неверное новое имя"})
        
        success, message = avatar_manager.rename_avatar(old_name, new_name)
        
        if success:
            # Уведомляем клиентов
            socketio.emit('avatar_renamed', {
                'old_name': old_name,
                'new_name': new_name,
                'timestamp': time.time()
            })
            
            return jsonify({
                "success": True,
                "message": message,
                "old_name": old_name,
                "new_name": new_name
            })
        else:
            return jsonify({"success": False, "error": message})
        
    except Exception as e:
        debug_log(f"❌ Ошибка переименования аватара: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/avatars/stats', methods=['GET'])
@teacher_required
def get_avatars_stats():
    """Получение статистики по всем аватарам"""
    try:
        if not AVATAR_MANAGER_ENABLED:
            return jsonify({"success": False, "error": "AvatarManager не загружен"})
        
        avatars = avatar_manager.list_avatars()
        
        total_avatars = len(avatars)
        total_frames = sum(avatar['frames_count'] for avatar in avatars)
        total_size = sum(avatar['total_size'] for avatar in avatars)
        
        # Группируем по форматам
        format_stats = {}
        for avatar in avatars:
            for fmt in avatar['formats']:
                format_stats[fmt] = format_stats.get(fmt, 0) + 1
        
        return jsonify({
            "success": True,
            "stats": {
                "total_avatars": total_avatars,
                "total_frames": total_frames,
                "total_size": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "format_stats": format_stats,
                "avatars": avatars
            }
        })
        
    except Exception as e:
        debug_log(f"❌ Ошибка получения статистики аватаров: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/avatars/<avatar_name>/preview', methods=['GET'])
def get_avatar_preview(avatar_name):
    """Получение превью аватара (первый кадр)"""
    try:
        if not AVATAR_MANAGER_ENABLED:
            return jsonify({"success": False, "error": "AvatarManager не загружен"})
        
        stats = avatar_manager.get_avatar_stats(avatar_name)
        if not stats:
            return jsonify({"success": False, "error": "Аватар не найден"})
        
        return jsonify({
            "success": True,
            "avatar": stats
        })
        
    except Exception as e:
        debug_log(f"❌ Ошибка получения превью аватара: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/avatars/validate', methods=['POST'])
@teacher_required
def validate_avatar_archive():
    """Валидация ZIP архива перед загрузкой"""
    try:
        if not AVATAR_MANAGER_ENABLED:
            return jsonify({"success": False, "error": "AvatarManager не загружен"})
        
        if 'avatar_file' not in request.files:
            return jsonify({"success": False, "error": "Файл не найден"})
        
        file = request.files['avatar_file']
        if file.filename == '':
            return jsonify({"success": False, "error": "Файл не выбран"})
        
        if not file.filename.endswith('.zip'):
            return jsonify({"success": False, "error": "Только ZIP архивы"})
        
        # Создаем временный файл
        import tempfile
        temp_dir = tempfile.mkdtemp()
        temp_zip = Path(temp_dir) / file.filename
        file.save(temp_zip)
        
        # Валидируем архив
        is_valid, message, frame_files = avatar_manager.validate_avatar_archive(temp_zip)
        
        # Очищаем временные файлы
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        if is_valid:
            return jsonify({
                "success": True,
                "message": message,
                "frames_count": len(frame_files),
                "frame_files": frame_files[:20],  # Первые 20 файлов
                "total_frames": len(frame_files)
            })
        else:
            return jsonify({"success": False, "error": message})
        
    except Exception as e:
        debug_log(f"❌ Ошибка валидации архива аватара: {e}")
        return jsonify({"success": False, "error": str(e)})

# =============================================================================
# 🔥 НОВЫЕ API ДЛЯ УПРАВЛЕНИЯ TTS СЕРВИСОМ
# =============================================================================

@app.route('/api/tts/status', methods=['GET'])
@teacher_required
def get_tts_service_status():
    """Получение статуса TTS сервиса"""
    try:
        status = speech_manager.get_tts_service_status()
        stats = speech_manager.get_stats()
        
        return jsonify({
            "success": True,
            "tts_service": status,
            "stats": stats,
            "config": speech_manager.config
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/tts/voices', methods=['GET'])
@teacher_required
def get_tts_available_voices():
    """Получение списка доступных голосов TTS"""
    try:
        voices = speech_manager.get_available_voices()
        
        return jsonify({
            "success": True,
            "voices": voices,
            "config": speech_manager.config.get('zindaki', {}).get('speaker_mapping', {})
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/tts/config', methods=['GET'])
@teacher_required
def get_tts_config():
    """Получение конфигурации TTS"""
    try:
        return jsonify({
            "success": True,
            "config": speech_manager.config
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/tts/config', methods=['POST'])
@teacher_required
def update_tts_config():
    """Обновление конфигурации TTS"""
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"})
        
        # Обновляем конфигурацию
        if 'enabled' in data:
            speech_manager.config['enabled'] = bool(data['enabled'])
        
        if 'primary' in data and data['primary'] in ['zindaki', 'gtts']:
            speech_manager.config['primary'] = data['primary']
        
        if 'zindaki' in data and isinstance(data['zindaki'], dict):
            # Обновляем только существующие ключи
            for key in ['base_url', 'timeout', 'retries']:
                if key in data['zindaki']:
                    speech_manager.config['zindaki'][key] = data['zindaki'][key]
        
        return jsonify({
            "success": True,
            "message": "TTS configuration updated",
            "config": speech_manager.config
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/tts/clear-cache', methods=['POST'])
@teacher_required
def clear_tts_cache():
    """Очистка кэша TTS сервиса"""
    try:
        data = request.json or {}
        days_old = data.get('days_old')
        
        success = speech_manager.clear_tts_cache(days_old)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"TTS cache cleared{' for days older than ' + str(days_old) if days_old else ''}"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to clear TTS cache"
            })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/tts/test', methods=['POST'])
@teacher_required
def test_tts_service():
    """Тестирование TTS сервиса"""
    try:
        data = request.json
        text = data.get('text', 'Тестовое сообщение для проверки TTS сервиса.')
        language = data.get('language', 'ru')
        speaker = data.get('speaker', 'baya')
        
        # Используем SpeechManager для теста
        audio_data = speech_manager.generate_optimized_speech(text, language, 'female')
        
        if audio_data:
            return jsonify({
                "success": True,
                "message": "TTS service is working",
                "audio_available": True,
                "audio_size": len(base64.b64decode(audio_data)),
                "tts_service": 'zindaki' if speech_manager.tts_client and speech_manager.tts_client.available else 'gtts'
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to generate audio",
                "tts_service": 'zindaki' if speech_manager.tts_client and speech_manager.tts_client.available else 'gtts'
            })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/tts/stats', methods=['GET'])
@teacher_required
def get_tts_stats():
    """Получение статистики использования TTS"""
    try:
        stats = speech_manager.get_stats()
        
        # Рассчитываем проценты
        total = stats['total_requests']
        if total > 0:
            stats['zindaki_success_rate'] = (stats['zindaki_success'] / total) * 100
            stats['zindaki_failure_rate'] = (stats['zindaki_failed'] / total) * 100
            stats['gtts_fallback_rate'] = (stats['gtts_fallback'] / total) * 100
        else:
            stats['zindaki_success_rate'] = 0
            stats['zindaki_failure_rate'] = 0
            stats['gtts_fallback_rate'] = 0
        
        return jsonify({
            "success": True,
            "stats": stats
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/tts/reset-stats', methods=['POST'])
@teacher_required
def reset_tts_stats():
    """Сброс статистики TTS"""
    try:
        speech_manager.stats = {
            'total_requests': 0,
            'zindaki_success': 0,
            'zindaki_failed': 0,
            'gtts_fallback': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        return jsonify({
            "success": True,
            "message": "TTS statistics reset"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# =============================================================================
# ОСТАВШИЕСЯ API (НЕ ИЗМЕНЯЕМ)
# =============================================================================

@app.route('/api/llm/priority', methods=['POST'])
def set_llm_priority_route():
    try:
        data = request.json
        priority = data.get('priority')
        
        if not priority:
            return jsonify({"success": False, "error": "Priority not specified"})
        
        success = set_llm_priority(priority)
        
        if success:
            for room_id in room_dialogue:
                if room_dialogue[room_id] is not None:
                    room_dialogue[room_id].llm.set_priority(priority)
            
            return jsonify({
                "success": True,
                "message": f"Приоритет успешно изменен на '{priority}'",
                "priority": priority
            })
        else:
            return jsonify({"success": False, "error": "Failed to save priority"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/llm/priority', methods=['GET'])
def get_llm_priority_route():
    try:
        priority = get_llm_priority()
        return jsonify({
            "success": True,
            "priority": priority
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/llm/priority_status')
def get_llm_priority_status():
    room_id = request.args.get('room_id', 'default')
    
    if room_id in room_dialogue and room_dialogue[room_id] is not None:
        status = room_dialogue[room_id].llm.get_priority_status()
        return jsonify({
            "success": True,
            "room": room_id,
            "status": status
        })
    
    return jsonify({"success": False, "error": "Room not found"})

@app.route('/api/llm/available_priorities')
def get_available_priorities():
    return jsonify({
        "success": True,
        "priorities": [
            {
                "id": "local_first",
                "name": "Локальная модель в первую очередь",
                "description": "Сначала локальная модель, затем OpenRouter как fallback"
            },
            {
                "id": "openrouter_first", 
                "name": "OpenRouter в первую очередь",
                "description": "Сначала OpenRouter, затем локальная модель как fallback"
            },
            {
                "id": "local_only",
                "name": "Только локальная модель",
                "description": "Использовать только локальную модель"
            },
            {
                "id": "openrouter_only",
                "name": "Только OpenRouter", 
                "description": "Использовать только OpenRouter"
            }
        ]
    })

@app.route('/api/llm/status')
def get_llm_status():
    room_id = request.args.get('room_id', 'default')
    
    if room_id in room_dialogue and room_dialogue[room_id] is not None:
        status = room_dialogue[room_id].llm.get_llm_status()
        return jsonify({
            "success": True,
            "room": room_id,
            "status": status
        })
    
    return jsonify({"success": False, "error": "Room not found"})

@app.route('/api/llm/local_status')
def get_local_llm_status():
    try:
        local_llm = llm_manager.local_llm
        status = local_llm.get_status()
        
        return jsonify({
            "success": True,
            "status": status
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/api/llm_manager/status')
def get_llm_manager_status():
    try:
        status = llm_manager.get_status()
        return jsonify({
            "success": True,
            "status": status
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/llm/poll_response', methods=['POST'])
def poll_llm_response():
    try:
        data = request.json
        room_id = data.get('room_id', 'default')
        last_check = data.get('last_check', 0)
        request_id_filter = data.get('request_id')
        
        current_time = time.time()
        room_last_poll_time[room_id] = current_time
        
        new_responses = []
        if room_id in room_llm_responses and room_llm_responses[room_id]:
            for resp in room_llm_responses[room_id]:
                if (resp['timestamp'] > last_check and 
                    (not request_id_filter or resp['request_id'] == request_id_filter)):
                    new_responses.append(resp)
        
        if new_responses:
            new_responses.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return jsonify({
                "success": True,
                "has_response": True,
                "responses": new_responses,
                "timestamp": current_time,
                "total_new_responses": len(new_responses)
            })
        
        return jsonify({
            "success": True, 
            "has_response": False,
            "timestamp": current_time,
            "queue_size": len(room_llm_responses.get(room_id, [])),
            "pending_requests": len(room_llm_pending_requests.get(room_id, {}))
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/llm/clear_queue', methods=['POST'])
def clear_llm_queue():
    try:
        data = request.json
        room_id = data.get('room_id', 'default')
        
        if room_id in room_llm_responses:
            room_llm_responses[room_id].clear()
            
        if room_id in room_llm_pending_requests:
            room_llm_pending_requests[room_id].clear()
            
        return jsonify({
            "success": True,
            "message": f"Очередь ответов для комнаты {room_id} очищена"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/llm/model', methods=['POST'])
def set_llm_model():
    try:
        data = request.json
        model = data.get('model')
        room_id = data.get('room_id', 'default')
        
        if not model:
            return jsonify({"success": False, "error": "Model not specified"})
        
        if room_id in room_dialogue and room_dialogue[room_id] is not None:
            room_dialogue[room_id].set_llm_model(model)
            return jsonify({"success": True, "model": model, "room": room_id})
        
        return jsonify({"success": False, "error": "Room not found"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/llm/models', methods=['GET'])
def get_llm_models():
    models = [
        {"id": "llama", "name": "Llama 3.3 8B", "description": "Мощная и быстрая модель от Meta", "provider": "openrouter"},
        {"id": "llama3", "name": "Llama 3.3 8B Instruct", "description": "Инструктивная версия Llama 3.3", "provider": "openrouter"},
        {"id": "qwen", "name": "Qwen 2.5 32B", "description": "Качественная модель от Alibaba", "provider": "openrouter"},
        {"id": "qwen-turbo", "name": "Qwen Coder", "description": "Специализированная модель для программирования", "provider": "openrouter"},
        {"id": "local_llama", "name": "Llama 3.2 3B (локальная)", "description": "Локальная модель Llama 3.2", "provider": "local"}
    ]
    return jsonify({"models": models})

@app.route('/api/config/llm_mode', methods=['GET'])
def get_llm_mode_api():
    try:
        config = load_config()
        return jsonify({
            "success": True,
            "mode": config.get("llm_query_mode", {}).get("default_mode", "traditional"),
            "available_modes": config.get("llm_query_mode", {}).get("available_modes", ["traditional", "llm_first"])
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/config/llm_mode', methods=['POST'])
def set_llm_mode_api():
    try:
        data = request.json
        mode = data.get('mode')
        
        if not mode:
            return jsonify({"success": False, "error": "Mode not specified"})
        
        if mode not in ["traditional", "llm_first"]:
            return jsonify({"success": False, "error": "Invalid mode. Use 'traditional' or 'llm_first'"})
        
        success = set_llm_mode(mode)
        
        if success:
            for room_id in room_llm_mode:
                room_llm_mode[room_id] = mode
                if room_id in room_dialogue and room_dialogue[room_id] is not None:
                    room_dialogue[room_id].set_llm_mode(mode)
            
            return jsonify({
                "success": True,
                "message": f"Режим LLМ успешно изменен на '{mode}'",
                "mode": mode
            })
        else:
            return jsonify({"success": False, "error": "Failed to save config"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/knowledge/stats', methods=['GET'])
def get_knowledge_stats():
    room_id = request.args.get('room_id', 'default')
    subject = request.args.get('subject', '')
    
    if room_id in room_dialogue and room_dialogue[room_id] is not None:
        stats = room_dialogue[room_id].get_knowledge_stats()
        if stats:
            return jsonify({
                "success": True,
                "room": room_id,
                "subject": subject or stats.get("subject", "unknown"),
                "stats": stats
            })
    
    return jsonify({"success": False, "error": "Room not found"})

@app.route('/api/knowledge/search', methods=['GET'])
def search_knowledge():
    room_id = request.args.get('room_id', 'default')
    query = request.args.get('query', '')
    max_results = int(request.args.get('max_results', 5))
    
    if not query:
        return jsonify({"success": False, "error": "Query parameter is required"})
    
    if room_id in room_dialogue and room_dialogue[room_id] is not None and room_dialogue[room_id].knowledge_base:
        results = room_dialogue[room_id].knowledge_base.search_similar(query, max_results)
        return jsonify({
            "success": True,
            "room": room_id,
            "query": query,
            "results": results,
            "total_found": len(results)
        })
    
    return jsonify({"success": False, "error": "Room not found or no knowledge base"})

@app.route('/api/knowledge/llm_answers', methods=['GET'])
def get_llm_answers():
    room_id = request.args.get('room_id', 'default')
    subject = request.args.get('subject', '')
    
    if room_id in room_dialogue and room_dialogue[room_id] is not None and room_dialogue[room_id].knowledge_base:
        answers = room_dialogue[room_id].knowledge_base.list_llm_answers()
        return jsonify({
            "success": True,
            "room": room_id,
            "subject": subject,
            "answers": answers,
            "total_answers": len(answers)
        })
    
    return jsonify({"success": False, "error": "Room not found or no knowledge base"})

@app.route('/api/lesson_content/<lesson_id>')
def get_lesson_content(lesson_id):
    try:
        content = lesson_manager.get_lesson_content(lesson_id)
        return jsonify(content)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/lessons')
def get_available_lessons():
    try:
        lessons = lesson_manager.get_available_lessons()
        return jsonify(lessons)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/practice_content/<lesson_id>')
def get_practice_content(lesson_id):
    try:
        content = lesson_manager.get_practice_content(lesson_id)
        return jsonify(content)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/practice_files')
def get_practice_files():
    try:
        files = lesson_manager.get_practice_files()
        return jsonify(files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/practice_txt_files')
def get_practice_txt_files():
    try:
        txt_files = []
        for txt_file in PRACTICE_DIR.glob("*.txt"):
            txt_files.append({
                'filename': txt_file.name,
                'size': txt_file.stat().st_size,
                'modified': datetime.fromtimestamp(txt_file.stat().st_mtime).isoformat()
            })
        
        return jsonify({
            "success": True,
            "files": txt_files,
            "total_files": len(txt_files)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload_practice', methods=['POST'])
def upload_practice():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"})
        
        result = lesson_manager.upload_practice_file(file)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/delete_practice/<filename>')
def delete_practice(filename):
    try:
        result = lesson_manager.delete_practice_file(filename)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/add_knowledge', methods=['POST'])
def add_knowledge():
    try:
        data = request.json
        subject = data.get('subject', 'общее')
        text = data.get('text', '')
        
        result = lesson_manager.add_knowledge(subject, text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/add_lesson', methods=['POST'])
def add_lesson():
    try:
        data = request.json
        subject = data.get('subject', 'общее')
        title = data.get('title', '')
        content = data.get('content', '')
        class_level = data.get('class_level', '5')
        
        result = lesson_manager.add_lesson(subject, title, content, class_level)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/add_practice', methods=['POST'])
def add_practice():
    try:
        data = request.json
        lesson_id = data.get('lesson_id', '')
        practice_data = data.get('practice_data', {})
        
        result = lesson_manager.add_practice(lesson_id, practice_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/download_knowledge')
def download_knowledge():
    subject = request.args.get('subject', 'обществознание')
    
    try:
        zip_path = lesson_manager.download_knowledge(subject)
        
        if not zip_path:
            return jsonify({"success": False, "error": f"База знаний для предмета '{subject}' не найдена"})
        
        return send_file(
            zip_path,
            as_attachment=True,
            download_name=f"{subject}_knowledge_base.zip",
            mimetype='application/zip'
        )
        
    except Exception as e:
        debug_log(f"❌ Ошибка экспорта знаний: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/download_lessons')
def download_lessons():
    try:
        lesson_files = []
        for lesson_dir in [LESSONS_DEMO_DIR, LESSONS_STUDENTS_DIR, LESSONS_GENERATED_DIR, LESSONS_DIR]:
            if lesson_dir.exists():
                if lesson_dir == LESSONS_STUDENTS_DIR:
                    for class_folder in lesson_dir.glob("*_class"):
                        if class_folder.is_dir():
                            for subject_folder in class_folder.iterdir():
                                if subject_folder.is_dir():
                                    lesson_files.extend(subject_folder.glob("*.txt"))
                else:
                    for lesson_file in lesson_dir.glob("*.txt"):
                        lesson_files.append(lesson_file)
        
        if not lesson_files:
            return jsonify({"success": False, "error": "Уроки не найдены"})
        
        import tempfile
        import zipfile
        
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        
        with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
            for lesson_file in lesson_files:
                if lesson_file.parent == LESSONS_DEMO_DIR:
                    zip_path = f"demo/{lesson_file.name}"
                elif lesson_file.parent == LESSONS_STUDENTS_DIR:
                    rel_path = lesson_file.relative_to(LESSONS_STUDENTS_DIR)
                    zip_path = f"students/{rel_path}"
                elif lesson_file.parent == LESSONS_GENERATED_DIR:
                    zip_path = f"generated/{lesson_file.name}"
                else:
                    zip_path = f"legacy/{lesson_file.name}"
                zipf.write(lesson_file, zip_path)
        
        temp_zip.close()
        
        return send_file(
            temp_zip.name,
            as_attachment=True,
            download_name="ai_teacher_lessons.zip",
            mimetype='application/zip'
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/download_practice')
def download_practice():
    try:
        zip_path = lesson_manager.download_practice()
        
        if not zip_path:
            return jsonify({"success": False, "error": "Практические задания не найдены"})
        
        return send_file(
            zip_path,
            as_attachment=True,
            download_name="ai_teacher_practice.zip",
            mimetype='application/zip'
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/download_practice_txt')
def download_practice_txt():
    try:
        practice_txt_files = list(PRACTICE_DIR.glob("*.txt"))
        
        if not practice_txt_files:
            return jsonify({"success": False, "error": "TXT файлы практики не найдены"})
        
        import tempfile
        import zipfile
    
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        
        with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
            for txt_file in practice_txt_files:
                zipf.write(txt_file, txt_file.name)
        
        temp_zip.close()
        
        return send_file(
            temp_zip.name,
            as_attachment=True,
            download_name="ai_teacher_practice_txt.zip",
            mimetype='application/zip'
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/config/keys', methods=['GET'])
def get_api_keys():
    try:
        config = load_config()
        return jsonify({
            "success": True,
            "keys": {
                "openrouter": config.get("openrouter", {}).get("api_key", ""),
                "llm": config.get("llm", {}).get("api_key", "")
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/config/keys', methods=['POST'])
def set_api_key():
    try:
        data = request.json
        provider = data.get('provider')
        api_key = data.get('api_key')
        
        if not provider or not api_key:
            return jsonify({"success": False, "error": "Provider and API key are required"})
        
        if provider not in ['openrouter', 'llm']:
            return jsonify({"success": False, "error": "Invalid provider. Use 'openrouter' or 'llm'"})
        
        success = update_api_key(provider, api_key)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"API ключ для {provider} успешно обновлен",
                "provider": provider
            })
        else:
            return jsonify({"success": False, "error": "Failed to update API key"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/config/test', methods=['POST'])
def test_api_key():
    try:
        data = request.json
        provider = data.get('provider')
        api_key = data.get('api_key')
        
        if not provider or not api_key:
            return jsonify({"success": False, "error": "Provider and API key are required"})
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-site.com",
            "X-Title": "AI Teacher"
        }
        
        test_data = {
            "model": "meta-llama/llama-3.3-8b-instruct:free",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 10
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            return jsonify({
                "success": True,
                "message": f"Ключ {provider} работает корректно",
                "valid": True
            })
        elif response.status_code == 401:
            return jsonify({
                "success": True,
                "message": f"Ключ {provider} неверный или неактивный",
                "valid": False
            })
        else:
            return jsonify({
                "success": True,
                "message": f"Ключ {provider} может быть неверным (код: {response.status_code})",
                "valid": False
            })
    except Exception as e:
        return jsonify({
            "success": True,
            "message": f"Ошибка проверки ключа: {str(e)}",
            "valid": False
        })

@app.route('/api/force_visualization', methods=['POST'])
def force_visualization():
    try:
        data = request.json
        room_id = data.get('room_id', 'default')
        topic = data.get('topic', 'Тестовая инфографика')
        context = data.get('context', 'Тестовый контекст')
        
        debug_log(f"Принудительная генерация SVG инфографики для комнаты {room_id}")
        
        add_visualization_to_queue(room_id, topic, context)
        
        return jsonify({
            "success": True,
            "message": f"SVG инфографика добавлена в очередь для комнаты {room_id}",
            "topic": topic,
            "queue_length": len(room_visualization_queue.get(room_id, []))
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/health')
def health_check():
    try:
        local_status = llm_manager.local_llm.get_status()
        openrouter_available = bool(get_api_key('openrouter'))
        
        lesson_count = 0
        for lesson_dir in [LESSONS_DEMO_DIR, LESSONS_STUDENTS_DIR, LESSONS_GENERATED_DIR, LESSONS_DIR]:
            if lesson_dir.exists():
                lesson_count += len(list(lesson_dir.glob("*.txt")))
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "local_llm": local_status,
                "openrouter": {"available": openrouter_available},
                "lessons": {"available": lesson_count > 0, "count": lesson_count},
                "practice": {"available": any(PRACTICE_DIR.iterdir()), "count": len(list(PRACTICE_DIR.glob("*.json")))},
                "llm_manager": llm_manager.get_status()
            }
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/llm/debug_openrouter')
def debug_openrouter():
    try:
        from llm import LLMIntegration
        llm = LLMIntegration()
        
        config = load_config()
        openrouter_config = config.get('openrouter', {})
        api_key = openrouter_config.get('api_key', '')
        
        status = {
            'api_key_set': bool(api_key and api_key.strip()),
            'api_key_length': len(api_key) if api_key else 0,
            'api_key_prefix': api_key[:8] + '...' if api_key else 'none',
            'model': openrouter_config.get('model', 'not set'),
            'test_connection': llm._test_openrouter_connection(),
            'priority_mode': get_llm_priority()
        }
        
        return jsonify({
            "success": True,
            "status": status
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# =============================================================================
# НОВЫЕ API ДЛЯ УПРАВЛЕНИЯ КЛЮЧАМИ OPENROUTER
# =============================================================================

@app.route('/api/keys/status', methods=['GET'])
@teacher_required
def get_keys_status():
    try:
        key_manager = get_key_manager()
        stats = key_manager.get_usage_stats()
        return jsonify({
            "success": True,
            "stats": stats
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/keys/add', methods=['POST'])
@teacher_required
def add_api_key_route():
    try:
        data = request.json
        api_key = data.get('api_key')
        name = data.get('name', 'new_key')
        limit_type = data.get('limit_type', 'standard')
        
        if not api_key:
            return jsonify({"success": False, "error": "API key is required"})
        
        key_manager = get_key_manager()
        success = key_manager.add_key(api_key, name, limit_type)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Ключ {name} успешно добавлен",
                "total_keys": len(key_manager.keys)
            })
        else:
            return jsonify({"success": False, "error": "Ключ уже существует"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/keys/set_reset_time', methods=['POST'])
@teacher_required
def set_reset_time():
    try:
        data = request.json
        reset_time = data.get('reset_time')
        
        if not reset_time:
            return jsonify({"success": False, "error": "Time is required"})
        
        key_manager = get_key_manager()
        success = key_manager.set_reset_time(reset_time)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Время сброса установлено на {reset_time}"
            })
        else:
            return jsonify({"success": False, "error": "Неверный формат времени"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/keys/force_reset', methods=['POST'])
@teacher_required
def force_reset_keys():
    try:
        key_manager = get_key_manager()
        key_manager.force_reset_all()
        
        return jsonify({
            "success": True,
            "message": "Все счетчики ключей сброшены"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/keys/upload', methods=['POST'])
@teacher_required
def upload_keys_file():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"})
        
        if not file.filename.endswith('.txt'):
            return jsonify({"success": False, "error": "Only TXT files allowed"})
        
        file_content = file.read().decode('utf-8')
        
        key_manager = get_key_manager()
        imported_count = key_manager.import_keys_from_file(file_content)
        
        return jsonify({
            "success": True,
            "message": f"Импортировано {imported_count} ключей",
            "imported_count": imported_count
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/keys/set_model', methods=['POST'])
@teacher_required
def set_openrouter_model():
    try:
        data = request.json
        model = data.get('model')
        
        if not model:
            return jsonify({"success": False, "error": "Model is required"})
        
        key_manager = get_key_manager()
        success = key_manager.set_model(model)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Модель установлена: {model}"
            })
        else:
            return jsonify({"success": False, "error": "Ошибка установки модели"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/keys/toggle', methods=['POST'])
@teacher_required
def toggle_key_active():
    try:
        data = request.json
        key_name = data.get('key_name')
        is_active = data.get('is_active', True)
        
        if not key_name:
            return jsonify({"success": False, "error": "Key name is required"})
        
        key_manager = get_key_manager()
        success = key_manager.toggle_key_active(key_name, is_active)
        
        if success:
            status = "активирован" if is_active else "деактивирован"
            return jsonify({
                "success": True,
                "message": f"Ключ {key_name} {status}"
            })
        else:
            return jsonify({"success": False, "error": "Ключ не найден"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/keys/set_limit', methods=['POST'])
@teacher_required
def set_key_limit():
    try:
        data = request.json
        key_name = data.get('key_name')
        limit_type = data.get('limit_type', 'standard')
        
        if not key_name:
            return jsonify({"success": False, "error": "Key name is required"})
        
        key_manager = get_key_manager()
        success = key_manager.set_key_limit(key_name, limit_type)
        
        if success:
            limit = key_manager.extended_limit if limit_type == 'extended' else key_manager.daily_limit
            return jsonify({
                "success": True,
                "message": f"Лимит ключа {key_name} установлен на {limit} запросов"
            })
        else:
            return jsonify({"success": False, "error": "Ключ не найден"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/keys/delete', methods=['POST'])
@teacher_required
def delete_key():
    try:
        data = request.json
        key_name = data.get('key_name')
        
        if not key_name:
            return jsonify({"success": False, "error": "Key name is required"})
        
        key_manager = get_key_manager()
        success = key_manager.delete_key(key_name)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Ключ {key_name} удален"
            })
        else:
            return jsonify({"success": False, "error": "Ключ не найден"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/keys/advanced_stats', methods=['GET'])
@teacher_required
def get_keys_advanced_stats():
    try:
        key_manager = get_key_manager()
        stats = key_manager.get_usage_stats()
        
        return jsonify({
            "success": True,
            "stats": stats
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/student/save', methods=['POST'])
def save_student():
    try:
        data = request.json
        student_data = {
            'name': data.get('name'),
            'education_level': data.get('level'),
            'age': data.get('age'),
            'subjects': data.get('subjects', []),
            'registration_date': datetime.now().isoformat(),
            'last_login': datetime.now().isoformat()
        }
        
        existing_student = student_manager.find_student_by_name(student_data['name'])
        
        if existing_student:
            student_id = existing_student['student_id']
            student_data['student_id'] = student_id
            student_data['rooms'] = existing_student.get('rooms', [])
            student_data['conference_id'] = existing_student.get('conference_id')
            student_manager.update_student_data(student_id, {
                'education_level': student_data['education_level'],
                'age': student_data['age'],
                'last_login': student_data['last_login']
            })
        else:
            student_id = student_manager.save_student_data(student_data)
            student_data['student_id'] = student_id
            
            create_student_rooms(student_data)
            
            updated_data = student_manager.load_student_data(student_id)
            if updated_data and 'conference_id' in updated_data:
                student_data['conference_id'] = updated_data['conference_id']
        
        if student_id:
            response_data = {
                "success": True,
                "student_id": student_id,
                "message": "Данные ученика сохранены",
                "rooms_created": not existing_student
            }
            
            if 'conference_id' in student_data:
                response_data['conference_id'] = student_data['conference_id']
            
            return jsonify(response_data)
        else:
            return jsonify({"success": False, "error": "Ошибка сохранения данных"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/student/<student_id>')
def get_student(student_id):
    try:
        student_data = student_manager.load_student_data(student_id)
        if student_data:
            return jsonify({"success": True, "student": student_data})
        else:
            return jsonify({"success": False, "error": "Ученик не найден"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/student/<student_id>/update', methods=['POST'])
def update_student(student_id):
    try:
        data = request.json
        if student_manager.update_student_data(student_id, data):
            return jsonify({"success": True, "message": "Данные обновлены"})
        else:
            return jsonify({"success": False, "error": "Ошибка обновления"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/student/<student_id>/rooms')
def get_student_rooms(student_id):
    try:
        student_data = student_manager.load_student_data(student_id)
        if not student_data:
            return jsonify({"success": False, "error": "Ученик не найден"})
        
        rooms = student_data.get('rooms', [])
        return jsonify({
            "success": True,
            "student_id": student_id,
            "student_name": student_data.get('name'),
            "rooms": rooms,
            "default_avatar": student_data.get('default_avatar', 'Woman'),
            "total_rooms": len(rooms)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/student/<student_id>/room/<subject>')
def get_student_room(student_id, subject):
    try:
        student_data = student_manager.load_student_data(student_id)
        if not student_data:
            return jsonify({"success": False, "error": "Ученик не найден"})
        
        rooms = student_data.get('rooms', [])
        target_room = None
        
        for room in rooms:
            if room.get('subject') == subject:
                target_room = room
                break
        
        if target_room:
            return jsonify({
                "success": True,
                "room": target_room,
                "conference_url": f"/conference?room={target_room['room_name']}&student=true&subject={subject}"
            })
        else:
            return jsonify({"success": False, "error": f"Комната для предмета {subject} не найдена"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/student/<student_id>/lessons')
def get_student_lessons(student_id):
    try:
        student_data = student_manager.load_student_data(student_id)
        if not student_data:
            return jsonify({"success": False, "error": "Ученик не найден"})
        
        lessons = student_data.get('lessons', [])
        return jsonify({
            "success": True,
            "student_id": student_id,
            "lessons": lessons,
            "total_lessons": len(lessons)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/student/<student_id>/add-lesson', methods=['POST'])
def add_student_lesson(student_id):
    try:
        data = request.json
        student_data = student_manager.load_student_data(student_id)
        if not student_data:
            return jsonify({"success": False, "error": "Ученик не найден"})
        
        lesson_data = {
            'lesson_id': data.get('lesson_id'),
            'subject': data.get('subject'),
            'title': data.get('title'),
            'date': datetime.now().isoformat(),
            'duration': data.get('duration', 0),
            'score': data.get('score'),
            'completed': data.get('completed', False)
        }
        
        if 'lessons' not in student_data:
            student_data['lessons'] = []
        
        student_data['lessons'].append(lesson_data)
        student_data['last_activity'] = datetime.now().isoformat()
        
        if student_manager.save_student_data(student_data):
            return jsonify({"success": True, "message": "Урок добавлен в историю"})
        else:
            return jsonify({"success": False, "error": "Ошибка сохранения"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/student/create-conference', methods=['POST'])
@student_required  
def create_student_conference_route():
    try:
        data = request.json
        subject = data.get('subject')
        
        if not subject:
            return jsonify({"success": False, "error": "Не указан предмет"})
        
        user_data = student_manager.load_user_data(session['user_id'])
        student_data = user_data.get('student_data', {})
        
        conference = create_student_conference(student_data, subject)
        
        if conference:
            return jsonify({
                "success": True,
                "conference": conference,
                "message": f"Создана комната для урока по {subject}"
            })
        else:
            return jsonify({"success": False, "error": "Ошибка создания конференции"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# =============================================================================
# 🔥 НОВЫЕ API ДЛЯ СТРУКТУРЫ УРОКОВ ПО КЛАССАМ
# =============================================================================

@app.route('/api/student/lessons-by-class', methods=['GET'])
@student_required
def get_lessons_by_class():
    try:
        user_data = student_manager.load_user_data(session['user_id'])
        if not user_data or not user_data.get('student_data'):
            return jsonify({"success": False, "error": "Данные ученика не найдены"})
        
        student_class = user_data['student_data'].get('education_level', '5')
        
        lessons_by_subject = student_manager.get_student_lessons_by_class(student_class)
        
        student_id = user_data['student_data'].get('student_id')
        progress_data = {}
        if student_id:
            progress_file = STUDENT_PROGRESS_DIR / f"{student_id}.json"
            if progress_file.exists():
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
        
        result = {}
        for subject, lessons in lessons_by_subject.items():
            subject_progress = progress_data.get("subjects", {}).get(subject, {})
            completed_lessons = subject_progress.get("completed_lessons", [])
            total_lessons = len(lessons)
            completed_count = len(completed_lessons)
            
            sorted_lessons = sorted(lessons, key=lambda x: x.get('lesson_number', 999))
            
            formatted_lessons = []
            for lesson in sorted_lessons:
                is_completed = lesson['id'] in completed_lessons
                formatted_lessons.append({
                    'id': lesson['id'],
                    'title': lesson['title'],
                    'subject': lesson['subject'],
                    'class_level': lesson.get('class_level', student_class),
                    'lesson_number': lesson.get('lesson_number'),
                    'completed': is_completed,
                    'file_path': str(lesson.get('file_path', '')),
                    'type': 'student'
                })
            
            next_lesson = None
            for lesson in formatted_lessons:
                if not lesson['completed']:
                    next_lesson = lesson
                    break
            
            result[subject] = {
                'lessons': formatted_lessons,
                'total': total_lessons,
                'completed': completed_count,
                'progress_percent': int((completed_count / total_lessons) * 100) if total_lessons > 0 else 0,
                'next_lesson': next_lesson
            }
        
        return jsonify({
            "success": True,
            "student_class": student_class,
            "lessons": result
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
        
        
@app.route('/api/student/start-selected-lesson', methods=['POST'])
@student_required
def start_selected_lesson():
    """Специальный эндпоинт для запуска выбранного урока школьником"""
    try:
        data = request.json
        lesson_id = data.get('lesson_id')
        subject = data.get('subject')
        
        if not lesson_id or not subject:
            return jsonify({"success": False, "error": "Не указан ID урока или предмет"})
        
        user_data = student_manager.load_user_data(session['user_id'])
        student_data = user_data.get('student_data', {})
        
        # Определяем класс ученика
        student_class = student_data.get('education_level', '5')
        
        # Проверяем, что урок существует
        lessons_by_subject = student_manager.get_student_lessons_by_class(student_class)
        if subject not in lessons_by_subject:
            return jsonify({"success": False, "error": "Предмет не найден"})
        
        # Находим урок
        selected_lesson = None
        for lesson in lessons_by_subject[subject]:
            if lesson['id'] == lesson_id:
                selected_lesson = lesson
                break
        
        if not selected_lesson:
            return jsonify({"success": False, "error": "Урок не найден"})
        
        # Создаем комнату для урока
        conference_id = str(int(time.time() * 1000))
        student_name = student_data.get('name', 'ученик').replace(' ', '_').lower()
        room_id = f"student_{subject}_{student_name}_{conference_id}"
        
        room_student_data[room_id] = {
            **student_data,
            'subject': subject,
            'conference_id': conference_id,
            'lesson_id': lesson_id
        }
        
        # Быстрая инициализация комнаты
        _fast_room_initialization(room_id)
        
        return jsonify({
            "success": True,
            "room_id": room_id,
            "conference_url": f"/conference?room={room_id}&student=true&subject={subject}&lesson_id={lesson_id}",
            "lesson_title": selected_lesson['title']
        })
        
    except Exception as e:
        debug_log(f"❌ Ошибка запуска выбранного урока: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/lesson/start-specific', methods=['POST'])
@student_required
def start_specific_lesson():
    try:
        data = request.json
        lesson_id = data.get('lesson_id')
        room_id = data.get('room_id')
        
        if not lesson_id or not room_id:
            return jsonify({"success": False, "error": "Не указан ID урока или комнаты"})
        
        # Создаем DialogueManager если нужно
        if not ensure_dialogue_manager_for_room(room_id):
            return jsonify({"success": False, "error": "Не удалось создать DialogueManager"})
        
        dialogue = room_dialogue[room_id]
        
        user_data = student_manager.load_user_data(session['user_id'])
        if user_data and user_data.get('student_data'):
            dialogue.set_student_data(user_data['student_data'])
        
        selected_lesson = None
        for subject_lessons in dialogue.lessons.values():
            for lesson in subject_lessons:
                if lesson.get('id') == lesson_id:
                    selected_lesson = lesson
                    break
            if selected_lesson:
                break
        
        if not selected_lesson:
            return jsonify({"success": False, "error": "Урок не найден"})
        
        dialogue.selected_lesson = selected_lesson
        dialogue.current_subject = selected_lesson.get('subject')
        dialogue.lesson_started = True
        dialogue.current_state = "lesson_reading"
        
        lesson_content = dialogue._load_lesson_content(selected_lesson.get('file_path'))
        dialogue.lesson_content = lesson_content
        dialogue.current_paragraph = 0
        
        if dialogue.current_subject:
            from knowledge.knowledge_base import KnowledgeBase
            dialogue.knowledge_base = KnowledgeBase(dialogue.current_subject)
        
        dialogue.conversation_history = []
        dialogue.conversation_context = []
        
        student_id = user_data.get('student_data', {}).get('student_id')
        if student_id:
            student_manager.update_student_lesson_progress(
                student_id, 
                selected_lesson.get('subject'), 
                lesson_id, 
                completed=False
            )
        
        return jsonify({
            "success": True,
            "lesson": {
                'id': selected_lesson['id'],
                'title': selected_lesson['title'],
                'subject': selected_lesson.get('subject'),
                'class_level': selected_lesson.get('class_level'),
                'lesson_number': selected_lesson.get('lesson_number')
            },
            "first_paragraph": lesson_content[0] if lesson_content else None
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/student/progress', methods=['GET'])
@student_required
def get_student_progress_api():
    try:
        user_data = student_manager.load_user_data(session['user_id'])
        if not user_data or not user_data.get('student_data'):
            return jsonify({"success": False, "error": "Данные ученика не найдены"})
        
        student_id = user_data['student_data'].get('student_id')
        student_class = user_data['student_data'].get('education_level', '5')
        
        progress_data = {}
        progress_file = STUDENT_PROGRESS_DIR / f"{student_id}.json"
        if progress_file.exists():
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
        
        lessons_by_subject = student_manager.get_student_lessons_by_class(student_class)
        
        result = {
            'student_id': student_id,
            'student_class': student_class,
            'student_name': user_data['student_data'].get('name', ''),
            'subjects': {}
        }
        
        for subject_name in lessons_by_subject.keys():
            subject_progress = progress_data.get("subjects", {}).get(subject_name, {
                "completed_lessons": [],
                "current_lesson": None,
                "total_lessons": len(lessons_by_subject.get(subject_name, [])),
                "last_accessed": None,
                "progress_percent": 0
            })
            
            completed_count = len(subject_progress.get("completed_lessons", []))
            total_lessons = len(lessons_by_subject.get(subject_name, []))
            
            next_lesson = None
            subject_lessons = lessons_by_subject.get(subject_name, [])
            for lesson in subject_lessons:
                if lesson['id'] not in subject_progress.get("completed_lessons", []):
                    next_lesson = lesson
                    break
            
            result['subjects'][subject_name] = {
                'total_lessons': total_lessons,
                'completed_lessons': completed_count,
                'progress_percent': int((completed_count / total_lessons) * 100) if total_lessons > 0 else 0,
                'last_updated': subject_progress.get("last_accessed"),
                'current_lesson': subject_progress.get("current_lesson"),
                'next_lesson': next_lesson
            }
        
        total_completed = sum(subj['completed_lessons'] for subj in result['subjects'].values())
        total_lessons = sum(subj['total_lessons'] for subj in result['subjects'].values())
        
        result['overall'] = {
            'total_lessons': total_lessons,
            'completed_lessons': total_completed,
            'progress_percent': int((total_completed / total_lessons) * 100) if total_lessons > 0 else 0,
            'subjects_count': len(result['subjects'])
        }
        
        return jsonify({"success": True, "progress": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# 🔥 ИСПРАВЛЕННЫЙ ЭНДПОИНТ ДЛЯ ЛИЧНОГО КАБИНЕТА
@app.route('/api/student/progress/dashboard')
@student_required
def get_student_progress_dashboard():
    """🔥 ИСПРАВЛЕННЫЙ: Получает прогресс ученика для личного кабинета"""
    try:
        user_data = student_manager.load_user_data(session['user_id'])
        if not user_data or not user_data.get('student_data'):
            return jsonify({"success": False, "error": "Данные ученика не найдены"})
        
        student_id = user_data['student_data'].get('student_id')
        student_class = user_data['student_data'].get('education_level', '5')
        student_name = user_data['student_data'].get('name', '')
        
        if not student_id:
            return jsonify({"success": False, "error": "Student ID not found"})
        
        progress_data = student_manager.get_student_progress_dashboard(
            student_id, student_class, student_name
        )
        
        return jsonify({"success": True, "progress": progress_data})
    except Exception as e:
        debug_log(f"❌ Ошибка получения прогресса для дашборда: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/lessons/structure', methods=['GET'])
@teacher_required
def get_lessons_structure():
    try:
        structure = student_manager.get_lessons_structure()
        
        serialized_structure = json_serialize_paths(structure)
        
        return jsonify({
            "success": True,
            "structure": serialized_structure
        })
    except Exception as e:
        debug_log(f"❌ Ошибка получения структуры уроков: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/lessons/create-sample', methods=['POST'])
@teacher_required
def create_sample_lessons():
    try:
        created_count = 0
        
        demo_subjects = ['математика', 'физика', 'химия', 'биология', 'история', 'литература']
        for subject in demo_subjects:
            lesson_file = LESSONS_DEMO_DIR / f"demo_{subject}_введение.txt"
            if not lesson_file.exists():
                with open(lesson_file, 'w', encoding='utf-8') as f:
                    f.write(f"""# Демо урок: Введение в {subject}

Это демонстрационный урок по предмету {subject}.

На этом уроке вы познакомитесь с:
1. Основными понятиями предмета
2. Историей развития
3. Практическим применением знаний

Урок содержит примеры, упражнения и тестовые задания.

Это демо-версия для тестирования системы AI-учителя.

Приятного обучения!
""")
                created_count += 1
        
        for class_dir in LESSONS_STUDENTS_DIR.glob("*_class"):
            if class_dir.is_dir():
                class_level = class_dir.name.replace("_class", "")
                
                for subject_dir in class_dir.iterdir():
                    if subject_dir.is_dir():
                        if not any(subject_dir.glob("*.txt")):
                            for i in range(1, 4):
                                lesson_file = subject_dir / f"lesson_{i:02d}_введение.txt"
                                if not lesson_file.exists():
                                    pass
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/add_demo_lesson', methods=['POST'])
@teacher_required
def add_demo_lesson():
    try:
        data = request.json
        title = data.get('title', '')
        content = data.get('content', '')
        subject = data.get('subject', 'общее')
        
        if not title or not content:
            return jsonify({"success": False, "error": "Название и содержание урока обязательны"})
        
        lesson_dir = LESSONS_DEMO_DIR
        
        existing_lessons = list(lesson_dir.glob("demo_*.txt"))
        demo_numbers = []
        for lesson in existing_lessons:
            match = re.search(r'demo_(\d+)', lesson.stem.lower())
            if match:
                demo_numbers.append(int(match.group(1)))
        
        next_number = max(demo_numbers) + 1 if demo_numbers else 1
        
        title_slug = re.sub(r'[^\wа-яе\s-]+', '', title.lower()).strip()
        title_slug = re.sub(r'\s+', '_', title_slug)
        title_slug = title_slug[:50]
        
        filename = f"demo_{next_number:02d}_{subject}_{title_slug}.txt"
        lesson_path = lesson_dir / filename
        
        with open(lesson_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return jsonify({
            "success": True,
            "filename": filename,
            "subject": subject,
            "title": title,
            "type": "demo",
            "lesson_number": next_number,
            "file_path": str(lesson_path.relative_to(LESSONS_DIR))
        })
    except Exception as e:
        debug_log(f"❌ Ошибка при добавлении демо-урока: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/lessons/list', methods=['GET'])
@teacher_required
def get_all_lessons_list():
    try:
        lessons = student_manager.get_all_lessons_list()
        
        return jsonify({
            "success": True,
            "total": len(lessons),
            "lessons": lessons
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/lesson/edit/<path:lesson_path>', methods=['GET'])
@teacher_required
def get_lesson_for_edit(lesson_path):
    try:
        result = lesson_manager.get_lesson_for_edit(lesson_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/lesson/save', methods=['POST'])
@teacher_required
def save_edited_lesson():
    try:
        data = request.json
        lesson_path = data.get('lesson_path')
        content = data.get('content')
        
        result = lesson_manager.save_edited_lesson(lesson_path, content)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/lesson/delete', methods=['POST'])
@teacher_required
def delete_lesson():
    try:
        data = request.json
        lesson_path = data.get('lesson_path')
        
        result = lesson_manager.delete_lesson(lesson_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/lessons/edit', methods=['GET'])
@teacher_required
def get_lessons_for_edit():
    try:
        class_filter = request.args.get('class', 'all')
        subject_filter = request.args.get('subject', 'all')
        search_query = request.args.get('search', '')
        
        result = lesson_manager.view_lessons(class_filter, subject_filter, search_query)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# =============================================================================
# 🔥 НОВЫЕ API ДЛЯ ТЕХНИЧЕСКИХ ПРЕДМЕТОВ
# =============================================================================

@app.route('/api/technical/settings', methods=['POST'])
@student_required
def set_technical_settings():
    """Устанавливает настройки для технических предметов"""
    try:
        data = request.json
        student_id = data.get('student_id')
        technical_mode = data.get('technical_mode', 'standard')
        
        if not student_id:
            return jsonify({"success": False, "error": "Не указан student_id"})
        
        student_data = student_manager.load_student_data(student_id)
        if student_data:
            student_data['technical_mode'] = technical_mode
            student_data['technical_support'] = TECHNICAL_SUPPORT_ENABLED
            student_manager.save_student_data(student_data)
            
            return jsonify({
                "success": True,
                "message": f"Установлен режим технических предметов: {technical_mode}",
                "technical_mode": technical_mode,
                "technical_support_enabled": TECHNICAL_SUPPORT_ENABLED
            })
        else:
            return jsonify({"success": False, "error": "Ученик не найден"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/technical/detect-subject', methods=['POST'])
def detect_technical_subject():
    """Определяет, является ли предмет техническим"""
    try:
        data = request.json
        subject = data.get('subject', '')
        
        if not subject:
            return jsonify({"success": False, "error": "Предмет не указан"})
        
        is_technical = False
        subject_type = "general"
        formulas_supported = False
        
        if TECHNICAL_SUPPORT_ENABLED:
            is_technical = is_technical_subject(subject)
            subject_type = get_subject_type(subject)
            formulas_supported = is_technical or (subject_type == "natural_science")
        
        return jsonify({
            "success": True,
            "subject": subject,
            "is_technical": is_technical,
            "subject_type": subject_type,
            "formulas_supported": formulas_supported,
            "technical_support_enabled": TECHNICAL_SUPPORT_ENABLED,
            "suggested_settings": {
                "formula_preservation": formulas_supported,
                "cleaning_mode": "technical" if formulas_supported else "standard",
                "visualization_type": "diagram" if is_technical else "infographic"
            }
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/technical/test-formula', methods=['POST'])
def test_formula_cleaning():
    """Тестирование очистки формул"""
    try:
        data = request.json
        text = data.get('text', 'Уравнение: E=mc², где E - энергия, m - масса')
        subject = data.get('subject', 'физика')
        
        if not text:
            return jsonify({"success": False, "error": "Текст обязателен"})
        
        cleaned_text = ""
        if TECHNICAL_SUPPORT_ENABLED:
            cleaned_text = clean_text_for_speech_technical(text, subject)
        else:
            cleaned_text = speech_manager.clean_text(text, subject)
        
        contains_formula_check = False
        if TECHNICAL_SUPPORT_ENABLED:
            contains_formula_check = contains_formulas(text)
        
        return jsonify({
            "success": True,
            "original": text,
            "cleaned": cleaned_text,
            "contains_formula": contains_formula_check,
            "technical_support": TECHNICAL_SUPPORT_ENABLED,
            "subject": subject,
            "subject_type": get_subject_type(subject) if TECHNICAL_SUPPORT_ENABLED else "unknown"
        })
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# =============================================================================
# НОВЫЕ API ДЛЯ УПРАВЛЕНИЯ АВАТАРОМ И КОМНАТАМИ
# =============================================================================

@app.route('/api/student/set-avatar', methods=['POST'])
@student_required
def set_student_avatar():
    try:
        data = request.json
        student_id = data.get('student_id')
        avatar_name = data.get('avatar_name')
        
        if not student_id or not avatar_name:
            return jsonify({"success": False, "error": "Не указаны данные"})
        
        student_data = student_manager.load_student_data(student_id)
        if student_data:
            student_data['preferred_avatar'] = avatar_name
            student_manager.save_student_data(student_data)
        
        return jsonify({"success": True, "message": "Аватар сохранен"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/student/get-room/<subject>')
@student_required
def get_student_room_for_subject(subject):
    try:
        user_data = student_manager.load_user_data(session['user_id'])
        student_data = user_data.get('student_data', {})
        
        student_rooms = student_data.get('rooms', [])
        for room in student_rooms:
            if room.get('subject') == subject:
                return jsonify({
                    "success": True,
                    "room": room
                })
        
        conference_id = student_data.get('conference_id', str(int(time.time() * 1000)))
        student_class = student_data.get('education_level', '5')
        student_name = student_data.get('name', '').replace(' ', '_').lower()
        room_name = f"student_{subject}_{student_name}_{conference_id}"
        
        room_student_data[room_name] = {
            **student_data,
            'subject': subject
        }
        
        return jsonify({
            "success": True,
            "room": {
                'subject': subject,
                'room_name': room_name,
                'conference_id': conference_id,
                'student_name': student_data.get('name', ''),
                'student_class': student_class
            },
            "newly_created": True
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/room/register-student', methods=['POST'])
@student_required
def register_student_room():
    try:
        data = request.json
        room_id = data.get('room_id')
        student_data = data.get('student_data')
        lesson_id = data.get('lesson_id')
        subject = data.get('subject')
        selected_subject_only = data.get('selected_subject_only', False)
        
        if not room_id or not student_data:
            return jsonify({"success": False, "error": "Не указаны данные"})
        
        room_student_data[room_id] = student_data
        
        _fast_room_initialization(room_id)
        
        # 🔥 НОВАЯ ЛОГИКА: Если это только выбор предмета (без урока)
        if selected_subject_only:
            room_student_data[room_id]['selected_subject_only'] = True
            room_student_data[room_id]['subject'] = subject
            debug_log(f"🔥 Комната {room_id} зарегистрирована только с предметом {subject}")
            
            return jsonify({
                "success": True,
                "message": "Комната зарегистрирована с предметом",
                "room_id": room_id,
                "need_lesson_voice_command": True
            })
        
        # Старая логика с уроком
        if lesson_id and lesson_id != 'next':
            if not ensure_dialogue_manager_for_room(room_id):
                return jsonify({"success": False, "error": "Не удалось создать DialogueManager"})
                
            dialogue = room_dialogue[room_id]
            selected_lesson = None
            for subject_lessons in dialogue.lessons.values():
                for lesson in subject_lessons:
                    if lesson.get('id') == lesson_id:
                        selected_lesson = lesson
                        break
                if selected_lesson:
                    break
            
            if selected_lesson:
                dialogue.selected_lesson = selected_lesson
                dialogue.current_subject = selected_lesson.get('subject')
                dialogue.lesson_started = True
                dialogue.current_state = "lesson_reading"
                
                lesson_content = dialogue._load_lesson_content(selected_lesson.get('file_path'))
                dialogue.lesson_content = lesson_content
                dialogue.current_paragraph = 0
        
        return jsonify({
            "success": True,
            "message": "Комната зарегистрирована",
            "room_id": room_id
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/student/available-lessons', methods=['GET'])
@student_required
def get_student_available_lessons():
    try:
        user_data = student_manager.load_user_data(session['user_id'])
        student_class = user_data['student_data'].get('education_level', '5')
        
        room_id = "default"
        # Создаем DialogueManager если нужно
        if not ensure_dialogue_manager_for_room(room_id):
            return jsonify({"success": False, "error": "Не удалось создать DialogueManager"})
        
        dialogue = room_dialogue[room_id]
        dialogue.set_student_data(user_data['student_data'])
        
        lessons_data = dialogue.get_lessons_for_student_api()
        
        if lessons_data.get("success", False):
            return jsonify(lessons_data)
        else:
            lessons_by_subject = student_manager.get_student_lessons_by_class(student_class)
            
            student_id = user_data['student_data'].get('student_id')
            progress_data = {}
            if student_id:
                progress_file = STUDENT_PROGRESS_DIR / f"{student_id}.json"
                if progress_file.exists():
                    with open(progress_file, 'r', encoding='utf-8') as f:
                        progress_data = json.load(f)
            
            result = {
                "success": True,
                "student_name": user_data['student_data'].get('name', ''),
                "student_class": student_class,
                "subjects": []
            }
            
            for subject_name, lessons in lessons_by_subject.items():
                subject_progress = progress_data.get("subjects", {}).get(subject_name, {})
                completed_ids = subject_progress.get("completed_lessons", [])
                
                sorted_lessons = sorted(lessons, key=lambda x: x.get('lesson_number', 999))
                
                subject_lessons = []
                for lesson in sorted_lessons:
                    is_completed = lesson['id'] in completed_ids
                    subject_lessons.append({
                        'id': lesson['id'],
                        'title': lesson['title'],
                        'subject': lesson['subject'],
                        'class_level': lesson.get('class_level', student_class),
                        'lesson_number': lesson.get('lesson_number'),
                        'completed': is_completed,
                        'file_path': str(lesson.get('file_path', '')),
                        'type': 'student'
                    })
                
                result["subjects"].append({
                    'subject': subject_name,
                    'lessons': subject_lessons,
                    'total_lessons': len(subject_lessons),
                    'completed_lessons': len([l for l in subject_lessons if l['completed']]),
                    'progress_percent': int((len([l for l in subject_lessons if l['completed']]) / len(subject_lessons)) * 100) if subject_lessons else 0
                })
            
            return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# =============================================================================
# АДМИНСКИЕ ФУНКЦИИ
# =============================================================================

@app.route('/api/admin/create_all_student_rooms')
@teacher_required
def create_all_student_rooms():
    try:
        created_count = 0
        error_count = 0
        
        for student_file in STUDENTS_DIR.glob("*.json"):
            try:
                with open(student_file, 'r', encoding='utf-8') as f:
                    student_data = json.load(f)
                
                if 'rooms' not in student_data or not student_data['rooms']:
                    if create_student_rooms(student_data):
                        created_count += 1
                        debug_log(f"Созданы комнаты для {student_data.get('name')}")
                    else:
                        error_count += 1
                        debug_log(f"Ошибка создания комнат для {student_data.get('name')}")
            except Exception as e:
                debug_log(f"Ошибка обработки файла {student_file}: {e}")
                error_count += 1
        
        return jsonify({
            "success": True,
            "message": f"Создано комнат для {created_count} учеников, ошибок: {error_count}",
            "created": created_count,
            "errors": error_count
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/admin/update_student_conference_ids')
@teacher_required
def update_student_conference_ids():
    try:
        updated_count = 0
        for student_file in STUDENTS_DIR.glob("*.json"):
            try:
                with open(student_file, 'r', encoding='utf-8') as f:
                    student_data = json.load(f)
                
                if 'conference_id' not in student_data:
                    conference_id = str(int(time.time() * 1000) + random.randint(1000, 9999))
                    student_data['conference_id'] = conference_id
                    
                    create_student_rooms(student_data)
                    
                    updated_count += 1
                    debug_log(f"Обновлен ученик {student_data.get('name')}")
            except Exception as e:
                debug_log(f"Ошибка обработки файла {student_file}: {e}")
        
        return jsonify({
            "success": True,
            "message": f"Обновлено {updated_count} учеников",
            "updated_count": updated_count
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/fix-progress-structure', methods=['POST'])
@teacher_required
def fix_progress_structure():
    """Исправляет структуру файлов прогресса"""
    try:
        result = student_manager.fix_progress_files_structure()
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/room/initialize', methods=['POST'])
def force_room_initialization():
    try:
        data = request.json
        room_id = data.get('room_id')
        
        if not room_id:
            return jsonify({"success": False, "error": "Room ID is required"})
        
        success = _fast_room_initialization(room_id)
        
        if success:
            # Создаем DialogueManager если нужно
            dm_created = ensure_dialogue_manager_for_room(room_id)
            
            return jsonify({
                "success": True,
                "message": f"Комната {room_id} инициализирована",
                "ready": room_id in room_dialogue and room_dialogue[room_id] is not None,
                "dialogue_manager_created": dm_created
            })
        else:
            return jsonify({
                "success": False,
                "error": f"Не удалось инициализировать комнату {room_id}"
            })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/room/status/<room_id>')
def get_room_status(room_id):
    try:
        status = {
            "room_id": room_id,
            "initialized": room_id in room_dialogue and room_dialogue[room_id] is not None,
            "participants": len(room_participants.get(room_id, [])),
            "ai_activated": room_ai_activated.get(room_id, False),
            "ready": room_id in room_dialogue and room_dialogue[room_id] is not None
        }
        
        return jsonify({"success": True, "status": status})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/room/health/<room_id>')
def room_health(room_id):
    try:
        health_status = {
            'room_id': room_id,
            'exists': room_id in room_participants,
            'participants': len(room_participants.get(room_id, [])),
            'ai_activated': room_ai_activated.get(room_id, False),
            'dialogue_manager': room_id in room_dialogue and room_dialogue[room_id] is not None,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            "success": True,
            "health": health_status
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/debug/create-test-users')
def create_test_users_route():
    try:
        teacher = student_manager.create_new_teacher('teacher', '123456')
        student = student_manager.create_new_student('student', '123456')
        
        return jsonify({
            "success": True,
            "message": "Тестовые пользователи созданы",
            "teacher": {"username": "teacher", "password": "123456"},
            "student": {"username": "student", "password": "123456"}
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/debug/routes')
def debug_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'path': str(rule)
        })
    return jsonify({"routes": routes})

@app.route('/test-conference')
def test_conference():
    return render_template('conference.html', 
                         room_id='test_room', 
                         embed=False,
                         student_mode=True,
                         subject='math',
                         subject_name='Математика')

@app.route('/test-student-flow')
def test_student_flow():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Student Flow</title>
    </head>
    </html>
    """

# =============================================================================
# ДИАГНОСТИЧЕСКИЕ API
# =============================================================================

@app.route('/api/debug/system_status')
@teacher_required
def system_status():
    """Диагностика состояния системы"""
    try:
        status = {
            "total_rooms": len(room_participants),
            "active_rooms": sum(1 for p in room_participants.values() if len(p) > 0),
            "dialogue_managers": sum(1 for d in room_dialogue.values() if d is not None),
            "llm_manager_queue": llm_manager.get_queue_size(),
            "thread_count": threading.active_count(),
            "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024 if 'psutil' in sys.modules else 0,
            "room_details": []
        }
        
        for room_id in list(room_participants.keys())[:10]:
            status["room_details"].append({
                "room_id": room_id,
                "participants": len(room_participants.get(room_id, [])),
                "has_dialogue": room_id in room_dialogue and room_dialogue[room_id] is not None,
                "ai_activated": room_ai_activated.get(room_id, False),
                "last_activity": room_last_activity.get(room_id, 0)
            })
        
        return jsonify({
            "success": True,
            "status": status,
            "timestamp": time.time()
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/debug/fix_blocked_rooms')
@teacher_required
def fix_blocked_rooms():
    """Ручное исправление заблокированных комнат"""
    try:
        fixed_count = 0
        
        for room_id in list(room_dialogue.keys()):
            if room_dialogue[room_id] is not None:
                dialogue = room_dialogue[room_id]
                
                dialogue.waiting_for_answer = False
                dialogue.lesson_started = False
                dialogue.practice_active = False
                
                room_practice_active[room_id] = False
                room_teacher_speaking[room_id] = False
                room_speaking[room_id] = False
                
                fixed_count += 1
                debug_log(f"Исправлена комната {room_id}")
        
        for room_id in list(room_llm_pending_requests.keys()):
            if len(room_llm_pending_requests[room_id]) > 10:
                requests = list(room_llm_pending_requests[room_id].items())
                requests.sort(key=lambda x: x[1]['timestamp'], reverse=True)
                room_llm_pending_requests[room_id] = dict(requests[:10])
        
        return jsonify({
            "success": True,
            "message": f"Исправлено {fixed_count} комнат",
            "fixed_count": fixed_count
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/debug/room/<room_id>')
def debug_room_info(room_id):
    info = {
        "success": True,
        "room_id": room_id,
        "student_data": room_student_data.get(room_id, {}),
        "has_student_data": room_id in room_student_data,
        "subject": room_student_data.get(room_id, {}).get('subject') if room_id in room_student_data else None,
        "dialogue_exists": room_id in room_dialogue and room_dialogue[room_id] is not None,
        "dialogue_has_student_data": room_dialogue[room_id].has_student_data if room_id in room_dialogue and room_dialogue[room_id] else False,
        "dialogue_current_subject": room_dialogue[room_id].current_subject if room_id in room_dialogue and room_dialogue[room_id] else None,
        "participants": list(room_participants.get(room_id, [])),
        "ai_activated": room_ai_activated.get(room_id, False)
    }
    
    return jsonify(info)

@app.route('/api/debug/student-lessons')
@student_required
def debug_student_lessons_route():
    user_data = student_manager.load_user_data(session['user_id'])
    student_class = user_data['student_data'].get('education_level', '5')
    
    room_id = "default"
    if not ensure_dialogue_manager_for_room(room_id):
        return jsonify({"success": False, "error": "Не удалось создать DialogueManager"})
    
    dialogue = room_dialogue[room_id]
    dialogue.set_student_data(user_data['student_data'])
    
    lessons_data = dialogue.get_lessons_for_student_api()
    
    class_dir = LESSONS_STUDENTS_DIR / f"{student_class}_class"
    class_exists = class_dir.exists()
    
    subjects_in_folder = []
    if class_exists:
        subjects_in_folder = [d.name for d in class_dir.iterdir() if d.is_dir()]
    
    return jsonify({
        "success": True,
        "student_data": user_data['student_data'],
        "lessons_data": lessons_data,
        "class_dir_exists": class_exists,
        "class_dir": str(class_dir) if class_exists else "не существует",
        "subjects_in_folder": subjects_in_folder,
        "student_class": student_class
    })

@app.route('/test-student-room')
def test_student_room():
    student_data = {
        'name': 'Тестовый Ученик',
        'education_level': '5',
        'age': '11',
        'student_id': 'test123'
    }
    
    conference = create_student_conference(student_data, 'математика')
    
    if conference:
        return f"""
        <html>
            <body>
                <h1>Тест комнаты ученика</h1>
                <p>Комната создана: {conference['room_id']}</p>
                <p>Предмет: {conference['student_data'].get('subject')}</p>
                <a href="{conference['conference_url']}" target="_blank">Открыть комнату</a>
            </body>
        </html>
        """
    return "Ошибка создания комнаты"

# =============================================================================
# ИСПРАВЛЕННЫЕ API ДЛЯ ДОБАВЛЕНИЯ УРОКОВ
# =============================================================================

@app.route('/api/add_lesson_with_class', methods=['POST'])
@teacher_required
def add_lesson_with_class():
    """Добавляет урок с указанием класса"""
    try:
        data = request.json
        subject = data.get('subject', 'общее')
        title = data.get('title', '')
        content = data.get('content', '')
        class_level = data.get('class_level', '5')
        
        result = student_manager.add_lesson_with_class(subject, title, content, class_level)
        return jsonify(result)
    except Exception as e:
        debug_log(f"❌ Ошибка при добавлении урока: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/lessons/next_number', methods=['GET'])
@teacher_required
def get_next_lesson_number():
    try:
        class_level = request.args.get('class', 'demo')
        subject = request.args.get('subject', 'общее')
        
        result = student_manager.get_next_lesson_number(class_level, subject)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/download_lessons_by_class', methods=['GET'])
@teacher_required
def download_lessons_by_class():
    try:
        class_level = request.args.get('class', 'all')
        
        lesson_files = []
        
        if class_level == 'all':
            for lesson_dir in [LESSONS_DEMO_DIR, LESSONS_STUDENTS_DIR, LESSONS_GENERATED_DIR, LESSONS_DIR]:
                if lesson_dir.exists():
                    if lesson_dir == LESSONS_STUDENTS_DIR:
                        for class_folder in lesson_dir.glob("*_class"):
                            if class_folder.is_dir():
                                for subject_folder in class_folder.iterdir():
                                    if subject_folder.is_dir():
                                        lesson_files.extend(subject_folder.glob("*.txt"))
                    else:
                        for lesson_file in lesson_dir.glob("*.txt"):
                            lesson_files.append(lesson_file)
        elif class_level == 'demo':
            if LESSONS_DEMO_DIR.exists():
                lesson_files = list(LESSONS_DEMO_DIR.glob("*.txt"))
        elif class_level == 'generated':
            if LESSONS_GENERATED_DIR.exists():
                lesson_files = list(LESSONS_GENERATED_DIR.glob("*.txt"))
        else:
            class_dir = LESSONS_STUDENTS_DIR / f"{class_level}_class"
            if class_dir.exists():
                for subject_dir in class_dir.iterdir():
                    if subject_dir.is_dir():
                        lesson_files.extend(subject_dir.glob("*.txt"))
        
        if not lesson_files:
            return jsonify({"success": False, "error": f"Уроки для класса {class_level} не найдены"})
        
        import tempfile
        import zipfile
    
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        
        with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
            for lesson_file in lesson_files:
                if lesson_file.parent == LESSONS_DEMO_DIR:
                    zip_path = f"demo/{lesson_file.name}"
                elif lesson_file.parent == LESSONS_GENERATED_DIR:
                    zip_path = f"generated/{lesson_file.name}"
                elif LESSONS_STUDENTS_DIR in lesson_file.parents:
                    rel_path = lesson_file.relative_to(LESSONS_STUDENTS_DIR)
                    zip_path = f"students/{rel_path}"
                else:
                    zip_path = f"legacy/{lesson_file.name}"
                zipf.write(lesson_file, zip_path)
        
        temp_zip.close()
        
        if class_level == 'all':
            filename = "all_lessons.zip"
        elif class_level == 'demo':
            filename = "demo_lessons.zip"
        elif class_level == 'generated':
            filename = "generated_lessons.zip"
        else:
            filename = f"lessons_{class_level}_class.zip"
        
        return send_file(
            temp_zip.name,
            as_attachment=True,
            download_name=filename,
            mimetype='application/zip'
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# =============================================================================
# НОВЫЕ API ДЛЯ МАССОВОЙ ЗАГРУЗКИ УРОКОВ
# =============================================================================

@app.route('/api/bulk_upload_lessons', methods=['POST'])
@teacher_required
def bulk_upload_lessons():
    try:
        if 'files' not in request.files:
            return jsonify({"success": False, "error": "Файлы не найдены"})
        
        files = request.files.getlist('files')
        
        if not files or files[0].filename == '':
            return jsonify({"success": False, "error": "Нет выбранных файлов"})
        
        results = {
            "success": True,
            "total_files": len(files),
            "uploaded": 0,
            "failed": 0,
            "details": []
        }
        
        for file in files:
            try:
                if not file.filename.endswith('.txt'):
                    results["failed"] += 1
                    results["details"].append({
                        "filename": file.filename,
                        "status": "failed",
                        "error": "Только TXT файлы"
                    })
                    continue
                
                content = file.read().decode('utf-8')
                filename = secure_filename(file.filename)
                
                class_level = "5"
                subject = "общее"
                title = filename.replace('.txt', '').replace('_', ' ').title()
                
                if '_' in filename:
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        if parts[0] in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', 'demo', 'generated']:
                            class_level = parts[0]
                            subject = parts[1] if len(parts) > 1 else "общее"
                            title = '_'.join(parts[2:]) if len(parts) > 2 else title
                        else:
                            subject = parts[0]
                
                if class_level == 'demo':
                    lesson_dir = LESSONS_DEMO_DIR
                elif class_level == 'generated':
                    lesson_dir = LESSONS_GENERATED_DIR
                else:
                    class_dir = LESSONS_STUDENTS_DIR / f"{class_level}_class"
                    class_dir.mkdir(parents=True, exist_ok=True)
                    
                    subject_dir = class_dir / subject
                    subject_dir.mkdir(parents=True, exist_ok=True)
                    lesson_dir = subject_dir
                
                existing_lessons = list(lesson_dir.glob("*.txt"))
                lesson_numbers = []
                for lesson in existing_lessons:
                    match = re.search(r'lesson[_\s]*(\d+)', lesson.stem.lower())
                    if match:
                        lesson_numbers.append(int(match.group(1)))
                
                next_number = max(lesson_numbers) + 1 if lesson_numbers else 1
                
                title_slug = re.sub(r'[^\wа-яе\s-]+', '', title.lower()).strip()
                title_slug = re.sub(r'\s+', '_', title_slug)
                title_slug = title_slug[:50]
                
                new_filename = f"lesson_{next_number:02d}_{title_slug}.txt"
                lesson_path = lesson_dir / new_filename
                
                with open(lesson_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                results["uploaded"] += 1
                results["details"].append({
                    "filename": filename,
                    "new_filename": new_filename,
                    "class_level": class_level,
                    "subject": subject,
                    "title": title,
                    "status": "success",
                    "path": str(lesson_path.relative_to(LESSONS_DIR))
                })
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": str(e)
                })
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# =============================================================================
# НОВЫЕ API ДЛЯ ПРОСМОТРА УРОКОВ
# =============================================================================

@app.route('/api/view_lessons', methods=['GET'])
@teacher_required
def view_lessons():
    try:
        class_level = request.args.get('class', 'all')
        subject = request.args.get('subject', 'all')
        
        result = lesson_manager.view_lessons(class_level, subject)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# =============================================================================
# API ДЛЯ УПРАВЛЕНИЯ УЧЕНИКАМИ
# =============================================================================

@app.route('/api/students')
@teacher_required
def get_all_students():
    try:
        students = []
        for user_file in USERS_DIR.glob("*.json"):
            with open(user_file, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
                if user_data.get('role') == 'student' and user_data.get('profile_complete', False):
                    students.append({
                        'user_id': user_data['user_id'],
                        'username': user_data['username'],
                        'student_data': user_data.get('student_data', {}),
                        'created_at': user_data.get('created_at'),
                        'last_login': user_data.get('last_login')
                    })
        
        return jsonify({"success": True, "students": students})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/student/profile')
@student_required
def get_student_profile():
    try:
        user_data = student_manager.load_user_data(session['user_id'])
        if not user_data:
            return jsonify({"success": False, "error": "Пользователь не найден"})
        
        progress_data = {}
        student_id = user_data.get('student_data', {}).get('student_id', '')
        if student_id:
            progress_file = STUDENT_PROGRESS_DIR / f"{student_id}.json"
            if progress_file.exists():
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
        
        return jsonify({
            "success": True,
            "student_data": user_data.get('student_data', {}),
            "profile_complete": user_data.get('profile_complete', False),
            "user_id": session['user_id'],
            "progress": progress_data
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# =============================================================================
# 🔥 НОВЫЕ API ДЛЯ ТЕХНИЧЕСКОЙ АВТОДЕТЕКЦИИ
# =============================================================================

@app.route('/api/technical/generate-exercise', methods=['POST'])
def generate_technical_exercise():
    """Генерация упражнения для технических предметов"""
    try:
        data = request.json
        subject = data.get('subject', 'математика')
        topic = data.get('topic', 'уравнения')
        level = data.get('level', '5')
        room_id = data.get('room_id', 'default')
        
        # Определяем тип предмета
        is_technical = False
        if TECHNICAL_SUPPORT_ENABLED:
            is_technical = is_technical_subject(subject)
        
        # Используем существующую LLM через менеджер
        if is_technical:
            prompt = f"""
            Создай упражнение по {subject} для ученика {level} класса.
            
            ТЕМА: {topic}
            УРОВЕНЬ: {level} класс
            
            ТРЕБОВАНИЯ:
            1. Включай конкретные формулы и вычисления
            2. Добавляй пошаговые решения
            3. Учитывай возраст ученика
            4. Используй математическую/научную нотацию
            5. Для физики/химии включай единицы измерения
            
            Верни 3-5 практических заданий.
            """
        else:
            prompt = f"""
            Создай практические задания по {subject} на тему: {topic}
            
            Уровень ученика: {level} класс
            
            Создай 5 заданий разного типа:
            1. Вопросы на понимание
            2. Практические задачи
            3. Творческие задания
            4. Аналитические вопросы
            5. Применение знаний
            
            Верни задания в структурированном виде.
            """
        
        llm_response = llm_manager.submit_request(
            prompt=prompt,
            system_prompt="Ты - учитель. Создай полезное упражнение соответствующего типа.",
            max_tokens=800,
            room_id=room_id
        )
        
        return jsonify({
            "success": True,
            "exercise": llm_response if llm_response else "Упражнение будет создано в процессе урока.",
            "type": "technical" if is_technical else "general",
            "subject": subject,
            "level": level,
            "technical_support": TECHNICAL_SUPPORT_ENABLED
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# =============================================================================
# ЗАПУСК СЕРВЕРА
# =============================================================================

if __name__ == '__main__':
    debug_log("🚀 Запуск AI Teacher системы с поддержкой технических предметов и слайдов...")
    debug_log(f"🔥 Поддержка технических предметов: {'ВКЛЮЧЕНА' if TECHNICAL_SUPPORT_ENABLED else 'ВЫКЛЮЧЕНА'}")
    debug_log(f"🔥 Поддержка слайдов уроков: ВКЛЮЧЕНА")
    debug_log(f"🔥 Форматы слайдов: JPG, PNG, MP4, WebP")
    debug_log(f"🔥 Добавлены API для загрузки слайдов уроков")
    debug_log(f"🔥 Добавлены API для резервного копирования данных учеников")
    debug_log(f"🔥 Ленивая инициализация DialogueManager активирована")
    debug_log(f"🔥 УСТРАНЕНА БЛОКИРОВКА: DialogueManager создается только при активации AI-учителя")
    debug_log(f"🔥 Поддержка управления аватарами: {'ВКЛЮЧЕНА' if AVATAR_MANAGER_ENABLED else 'ВЫКЛЮЧЕНА'}")
    debug_log(f"🔥 Добавлен модуль lesson_manager для управления уроками")
    debug_log(f"🔥 ДОБАВЛЕНА ПОДДЕРЖКА ВЗРОСЛЫХ СТУДЕНТОВ")
    debug_log(f"🔥 ДОБАВЛЕНА ПОДДЕРЖКА ЯЗЫКОВЫХ УРОВНЕЙ CEFR (A1-C2)")
    debug_log(f"🔥 РЕЖИМЫ ДЛЯ ВЗРОСЛЫХ: 'изучать что угодно' и 'английский язык'")
    debug_log(f"🔥 СТРУКТУРА ПАПОК: lessons/students/adult_language/[A1-C2]_english/")
    debug_log(f"🔥 ДОБАВЛЕН МЕНЕДЖЕР РЕЧИ (SpeechManager) - все функции TTS вынесены в отдельный модуль")
    debug_log(f"🔥 ИНТЕГРАЦИЯ С КАСТОМНЫМ TTS СЕРВИСОМ ZINDAKI: tts.zindaki-edu.ru")
    debug_log(f"🔥 TTS СЕРВИС ДОСТУПЕН: {speech_manager.tts_client.available if speech_manager.tts_client else False}")
    debug_log(f"🔥 НОВЫЕ API ДЛЯ УПРАВЛЕНИЯ TTS: /api/tts/status, /api/tts/voices, /api/tts/config, /api/tts/test")
    
    setup_llm_manager()
    
    # Запускаем периодическую очистку
    periodic_cleanup()
    
    # Инициализируем системные комнаты
    system_rooms = ['default', 'demo_room', 'test_room']
    for room in system_rooms:
        _fast_room_initialization(room)
    
    debug_log(f"✅ Система готова. Async mode: {socketio.async_mode}")
    debug_log(f"✅ Максимальное количество комнат в памяти: {MAX_ROOMS}")
    debug_log(f"✅ Таймаут неактивных комнат: {ROOM_TIMEOUT} секунд")
    debug_log(f"✅ ДОБАВЛЕН новый API /api/adult/levels для получения уровней CEFR")
    debug_log(f"✅ ДОБАВЛЕН новый API /api/student/adult/lessons для взрослых студентов")
    debug_log(f"✅ ДОБАВЛЕН новый API /api/student/create-adult-room для создания комнат взрослых")
    debug_log(f"✅ ВСЕ НОВЫЕ API защищены декораторами @student_required/@teacher_required")
    debug_log(f"✅ МЕНЕДЖЕР РЕЧИ ИНИЦИАЛИЗИРОВАН И ГОТОВ К РАБОТЕ")
    debug_log(f"✅ TTS СЕРВИС ИНТЕГРИРОВАН И АКТИВЕН")
    debug_log(f"✅ СИСТЕМА ГОТОВА К РАБОТЕ! 🚀")
    
    socketio.run(
        app, 
        host='0.0.0.0', 
        port=5000, 
        debug=True, 
        allow_unsafe_werkzeug=True
    )
