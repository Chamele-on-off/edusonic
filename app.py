from flask import Flask, render_template, send_from_directory, jsonify, request, send_file, session, redirect, url_for
import os
from pathlib import Path
from flask_socketio import SocketIO, emit, join_room, leave_room
from gtts import gTTS
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

# Настройка Flask и SocketIO
app = Flask(__name__, static_folder='static')
app.secret_key = 'ai-teacher-secret-key-2024'  # Секретный ключ для сессий

socketio = SocketIO(
    app, 
    cors_allowed_origins="*",
    async_mode='threading',
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=1e8,
    logger=True,
    engineio_logger=True
)

# Убедитесь что это ПЕРВАЯ строка после создания socketio
socketio.async_mode = 'threading'

# Ручная настройка CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

BASE_DIR = Path(__file__).parent
FRAMES_DIR = BASE_DIR / 'static' / 'avatar' / 'frames'
LESSONS_DIR = BASE_DIR / 'lessons'
MATERIALS_DIR = BASE_DIR / 'materials'
PRACTICE_DIR = BASE_DIR / 'materials' / 'practice'
STUDENTS_DIR = BASE_DIR / "students_data"
USERS_DIR = BASE_DIR / "users_data"

# Создаем необходимые папки
for folder in [LESSONS_DIR, MATERIALS_DIR, PRACTICE_DIR, STUDENTS_DIR, USERS_DIR]:
    os.makedirs(folder, exist_ok=True)

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
room_current_avatar = defaultdict(lambda: 'teacher')

# PeerJS tracking
room_peer_ids = defaultdict(dict)  # room_id -> {socket_id: peer_id}

# Кэш для визуализаций
diagram_cache = {}
# Очереди визуализаций для каждой комнаты
room_visualization_queue = defaultdict(list)
# Флаг активной визуализации для каждой комната
room_visualization_active = defaultdict(bool)

# Очереди ответов LLM для polling
room_llm_responses = defaultdict(list)
room_last_poll_time = defaultdict(lambda: 0)

# Улучшенная система отслеживания запросов
room_llm_pending_requests = defaultdict(dict)
room_last_llm_update = defaultdict(lambda: 0)

# Менеджер локальной LLM
llm_manager = get_llm_manager()

# Менеджер ключей API
key_manager = get_key_manager()

# Данные учеников для комнат
room_student_data = defaultdict(dict)

# Отладочный режим
DEBUG_LLM = True

def debug_log(message):
    if DEBUG_LLM:
        print(f"🔧 [LLM_DEBUG] {message}")

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

def load_user_data(user_id):
    """Загрузка данных пользователя"""
    try:
        user_file = USERS_DIR / f"{user_id}.json"
        if user_file.exists():
            with open(user_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error loading user data: {e}")
        return None

def save_user_data(user_data):
    """Сохранение данных пользователя"""
    try:
        user_id = user_data['user_id']
        user_file = USERS_DIR / f"{user_id}.json"
        
        with open(user_file, 'w', encoding='utf-8') as f:
            json.dump(user_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving user data: {e}")
        return False

def authenticate_user(username, password, role):
    """Аутентификация пользователя - ТОЛЬКО существующие пользователи"""
    try:
        # Поиск пользователя по username и role
        for user_file in USERS_DIR.glob("*.json"):
            with open(user_file, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
                
                if (user_data.get('username') == username and 
                    user_data.get('role') == role and 
                    user_data.get('password') == password):
                    return user_data
        
        # Если пользователь не найден - ВОЗВРАЩАЕМ None
        return None
        
    except Exception as e:
        print(f"Authentication error: {e}")
        return None

def create_new_student(username, password):
    """Создание нового ученика"""
    try:
        user_id = str(uuid.uuid4())
        user_data = {
            'user_id': user_id,
            'username': username,
            'password': password,
            'role': 'student',
            'created_at': datetime.now().isoformat(),
            'last_login': datetime.now().isoformat(),
            'profile_complete': False,
            'student_data': None
        }
        
        if save_user_data(user_data):
            return user_data
        return None
    except Exception as e:
        print(f"Error creating student: {e}")
        return None

def create_new_teacher(username, password):
    """Создание нового учителя"""
    try:
        user_id = str(uuid.uuid4())
        user_data = {
            'user_id': user_id,
            'username': username,
            'password': password,
            'role': 'teacher',
            'created_at': datetime.now().isoformat(),
            'last_login': datetime.now().isoformat(),
            'profile_complete': True
        }
        
        if save_user_data(user_data):
            return user_data
        return None
    except Exception as e:
        print(f"Error creating teacher: {e}")
        return None

def update_student_profile(user_id, student_data):
    """Обновление профиля ученика"""
    try:
        user_data = load_user_data(user_id)
        if not user_data:
            return False
        
        user_data['student_data'] = student_data
        user_data['profile_complete'] = True
        user_data['profile_updated'] = datetime.now().isoformat()
        
        # Сохраняем также в отдельный файл ученика
        student_id = student_data.get('student_id')
        if student_id:
            save_student_data(student_data)
        
        return save_user_data(user_data)
    except Exception as e:
        print(f"Error updating student profile: {e}")
        return False

# =============================================================================
# МАРШРУТЫ АУТЕНТИФИКАЦИИ
# =============================================================================

@app.route('/')
def home():
    """Основная страница - лендинг"""
    return render_template('landing.html')

@app.route('/login')
def login():
    """Страница входа"""
    if 'user_id' in session:
        if session.get('role') == 'teacher':
            return redirect('/teacher')
        else:
            return redirect('/student')
    return render_template('login.html')

@app.route('/auth/login', methods=['POST'])
def auth_login():
    """Обработка входа"""
    try:
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        role = data.get('role', 'student')
        
        if not username or not password:
            return jsonify({"success": False, "error": "Заполните все поля"})
        
        user_data = authenticate_user(username, password, role)
        
        if user_data:
            session['user_id'] = user_data['user_id']
            session['username'] = user_data['username']
            session['role'] = user_data['role']
            
            user_data['last_login'] = datetime.now().isoformat()
            save_user_data(user_data)
            
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
    """Выход из системы"""
    session.clear()
    return redirect('/login')

@app.route('/logout', methods=['POST'])
def logout_post():
    """Выход из системы (POST запрос)"""
    session.clear()
    return jsonify({"success": True, "message": "Успешный выход"})

@app.route('/api/auth/check')
def check_auth():
    """Проверка статуса авторизации"""
    if 'user_id' in session:
        user_data = load_user_data(session['user_id'])
        if user_data:
            return jsonify({
                "success": True,
                "role": user_data.get('role'),
                "user_id": session['user_id']
            })
    return jsonify({"success": False})

@app.route('/investing.html')
def investing():
    """Страница для инвесторов"""
    return render_template('investing.html')

# =============================================================================
# ЛИЧНЫЕ КАБИНЕТЫ
# =============================================================================

@app.route('/teacher')
@teacher_required
def teacher():
    """Страница учителя (личный кабинет)"""
    user_data = load_user_data(session['user_id'])
    return render_template('teacher.html', user=user_data)

@app.route('/student')
@student_required
def student():
    """Страница ученика (личный кабинет)"""
    user_data = load_user_data(session['user_id'])
    
    if not user_data.get('profile_complete', False):
        return render_template('student_profile.html', user=user_data)
    
    student_data = user_data.get('student_data', {})
    return render_template('student.html', user=user_data, student_data=student_data)

@app.route('/auth/complete-profile', methods=['POST'])
@student_required
def complete_profile():
    """Завершение заполнения профиля ученика"""
    try:
        data = request.json
        user_id = session['user_id']
        
        student_data = {
            'name': data.get('name'),
            'education_level': data.get('level'),
            'age': data.get('age'),
            'student_id': str(uuid.uuid4()),
            'registration_date': datetime.now().isoformat()
        }
        
        if update_student_profile(user_id, student_data):
            create_student_rooms(student_data)
            
            return jsonify({
                "success": True,
                "message": "Профиль успешно сохранен",
                "student_id": student_data['student_id']
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
    """Получение списка всех пользователей"""
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
    """Создание нового пользователя (только для учителя)"""
    try:
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        role = data.get('role', 'student')
        
        if not username or not password:
            return jsonify({"success": False, "error": "Заполните все поля"})
        
        # Проверяем, не существует ли уже пользователь с таким именем
        for user_file in USERS_DIR.glob("*.json"):
            with open(user_file, 'r', encoding='utf-8') as f:
                existing_user = json.load(f)
                if existing_user.get('username') == username:
                    return jsonify({"success": False, "error": "Пользователь с таким логином уже существует"})
        
        # Создаем пользователя
        if role == 'student':
            user_data = create_new_student(username, password)
        elif role == 'teacher':
            user_data = create_new_teacher(username, password)
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
    """Удаление пользователя"""
    try:
        user_file = USERS_DIR / f"{user_id}.json"
        
        if not user_file.exists():
            return jsonify({"success": False, "error": "Пользователь не найден"})
        
        # Не позволяем удалить самого себя
        if session.get('user_id') == user_id:
            return jsonify({"success": False, "error": "Нельзя удалить свой собственный аккаунт"})
        
        user_file.unlink()
        return jsonify({"success": True, "message": "Пользователь удален"})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# =============================================================================
# API ДЛЯ УПРАВЛЕНИЯ УЧЕНИКАМИ
# =============================================================================

@app.route('/api/students')
@teacher_required
def get_all_students():
    """Получение списка всех учеников"""
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

@app.route('/api/students/export')
@teacher_required
def export_students_data():
    """Экспорт всех данных учеников"""
    try:
        import zipfile
        import tempfile
        
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        
        with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
            for user_file in USERS_DIR.glob("*.json"):
                with open(user_file, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)
                    if user_data.get('role') == 'student':
                        zipf.write(user_file, f"users/{user_file.name}")
            
            for student_file in STUDENTS_DIR.glob("*.json"):
                zipf.write(student_file, f"students/{student_file.name}")
        
        temp_zip.close()
        
        return send_file(
            temp_zip.name,
            as_attachment=True,
            download_name=f"students_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mimetype='application/zip'
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/student/<student_user_id>')
@teacher_required
def get_student_details(student_user_id):
    """Получение детальной информации об ученике"""
    try:
        user_data = load_user_data(student_user_id)
        if not user_data or user_data.get('role') != 'student':
            return jsonify({"success": False, "error": "Ученик не найден"})
        
        return jsonify({
            "success": True,
            "student": {
                'user_id': user_data['user_id'],
                'username': user_data['username'],
                'student_data': user_data.get('student_data', {}),
                'created_at': user_data.get('created_at'),
                'last_login': user_data.get('last_login'),
                'rooms': user_data.get('student_data', {}).get('rooms', [])
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/student/profile')
@student_required
def get_student_profile():
    """Получение профиля ученика"""
    try:
        user_data = load_user_data(session['user_id'])
        if not user_data:
            return jsonify({"success": False, "error": "Пользователь не найден"})
        
        return jsonify({
            "success": True,
            "student_data": user_data.get('student_data', {}),
            "profile_complete": user_data.get('profile_complete', False),
            "user_id": session['user_id']
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# =============================================================================
# ОСНОВНЫЕ ФУНКЦИИ СИСТЕМЫ
# =============================================================================

def handle_disconnected_session(sid):
    """Безопасная обработка отключенных сессий"""
    try:
        for room_id, participants in room_participants.items():
            if sid in participants:
                participants.remove(sid)
                print(f"🔧 Удален отключенный участник {sid} из комнаты {room_id}")
                emit('participants_update', {'count': len(participants)}, room=room_id)
                
        for room_id, peers in room_peer_ids.items():
            if sid in peers:
                del peers[sid]
                print(f"🔧 Удален peer_id для отключенного участника {sid}")
                
    except Exception as e:
        print(f"⚠️ Ошибка очистки отключенной сессии {sid}: {e}")

def setup_llm_manager():
    """Настройка менеджера LLM с улучшенным callback"""
    llm_manager.start()
    
    def global_llm_callback(request_id, response, room_id, original_request_id=None):
        """Глобальный обработчик ответов от LLM с немедленной доставкой"""
        debug_log(f"Получен ответ для комнаты {room_id}: {response[:100]}...")
        
        target_request_id = original_request_id
        if not target_request_id:
            for req_id, req_data in room_llm_pending_requests[room_id].items():
                if req_data.get('manager_id') == request_id:
                    target_request_id = req_id
                    break
        
        if not target_request_id:
            print(f"⚠️ Не найден исходный request_id для manager_id: {request_id}")
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
            print(f"⚠️ Не удалось отправить через WebSocket: {e}. Ответ сохранен для polling.")
    
    llm_manager.register_room_callback('global', global_llm_callback)
    debug_log("LLM Manager настроен с улучшенным callback")

def _fast_room_initialization(room_id):
    """Быстрая инициализация комнаты с гарантированным созданием DialogueManager"""
    try:
        is_student_room = '_' in room_id and room_id != 'default'
        
        if room_id not in room_dialogue or room_dialogue[room_id] is None:
            # ВСЕГДА используем единый DialogueManager
            room_dialogue[room_id] = DialogueManager(socketio)
            room_dialogue[room_id].room_id = room_id
            
            # Настраиваем режим ученика если нужно
            if is_student_room:
                student_data = room_student_data.get(room_id, {})
                subject = _extract_subject_from_room(room_id)
                if subject:
                    room_dialogue[room_id].set_student_mode(subject, student_data)
                    debug_log(f"Установлен режим ученика для комнаты {room_id}: {subject}")
            
            debug_log(f"Создан DialogueManager для комнаты {room_id}")
        
        if room_id not in room_current_avatar:
            room_current_avatar[room_id] = 'teacher' if not is_student_room else 'woman'
        
        # 🔥 ВАЖНОЕ ИСПРАВЛЕНИЕ: Убедимся, что режим ученика установлен правильно
        if is_student_room and room_dialogue[room_id]:
            if not getattr(room_dialogue[room_id], 'is_student_mode', False):
                subject = _extract_subject_from_room(room_id)
                if subject:
                    student_data = room_student_data.get(room_id, {})
                    room_dialogue[room_id].set_student_mode(subject, student_data)
                    debug_log(f"Установлен режим ученика для {room_id}: {subject}")
        
        if room_dialogue[room_id]:
            room_dialogue[room_id].set_llm_mode(room_llm_mode[room_id])
        
        debug_log(f"Быстрая инициализация завершена для {room_id}")
        return True
        
    except Exception as e:
        print(f"⚠️ Ошибка быстрой инициализации {room_id}: {e}")
        try:
            room_dialogue[room_id] = DialogueManager(socketio)
            room_dialogue[room_id].room_id = room_id
            room_dialogue[room_id].set_llm_mode(room_llm_mode[room_id])
            debug_log(f"Аварийно создан DialogueManager для {room_id}")
            return True
        except:
            return False

def _extract_subject_from_room(room_id: str) -> Optional[str]:
    """Извлекает предмет из названия комнаты для режима ученика"""
    if '_' in room_id:
        subject_part = room_id.split('_')[0].lower()
        subject_map = {
            'math': 'математика',
            'mathematics': 'математика',
            'physics': 'физика', 
            'chemistry': 'химия',
            'biology': 'биология',
            'history': 'история',
            'social': 'обществознание',
            'literature': 'литература',
            'russian': 'русский язык',
            'english': 'английский язык',
            'geography': 'география'
        }
        return subject_map.get(subject_part)
    return None

def clean_text_for_speech(text: str) -> str:
    """Тщательная очистка текста для озвучивания"""
    if not text:
        return ""
    
    text = re.sub(r'[#\*\_\~`]', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\\n', ' ', text)
    text = re.sub(r'\\t', ' ', text)
    text = re.sub(r'\\r', ' ', text)
    text = re.sub(r'[^\u0400-\u04FFa-zA-Z0-9\s\.,!?;:()\-—]', '', text)
    text = re.sub(r'[\.\,]{2,}', '.', text)
    text = re.sub(r'\s+([\.,!?;:)])', r'\1', text)
    text = re.sub(r'([(\-])\s+', r'\1', text)
    text = text.strip()
    
    if text and len(text) > 1:
        text = text[0].upper() + text[1:]
    
    return text

def reset_speaking_state(room_id, is_teacher=False):
    """Сбрасывает состояние речи для указанной комнаты"""
    room_speaking[room_id] = False
    if is_teacher:
        room_teacher_speaking[room_id] = False
    socketio.emit('speaking_state', {'speaking': False}, room=room_id)

def speak_text(room_id, text, voice_type='female', is_teacher=False, skip_history=False):
    """Озвучивает текст и добавляет его в историю"""
    if not text.strip():
        return
        
    cleaned_text = clean_text_for_speech(text)
    
    if not cleaned_text.strip():
        print(f"⚠️ Текст пуст после очистки: {text[:100]}...")
        return
        
    if is_teacher:
        room_teacher_speaking[room_id] = True
        
    room_speaking[room_id] = True
    socketio.emit('speaking_state', {'speaking': True}, room=room_id)
    
    audio_data = text_to_speech(cleaned_text, lang='ru')
    if audio_data:
        emit('speech_audio', {
            'audio': audio_data,
            'text': cleaned_text,
            'timestamp': time.time(),
            'voice_type': voice_type,
            'is_teacher': is_teacher
        }, room=room_id)
        
        if not skip_history:
            room_speech_data[room_id].append({
                'text': cleaned_text,
                'timestamp': time.time(),
                'type': 'generated',
                'voice_type': voice_type,
                'is_teacher': is_teacher
            })
            if len(room_speech_data[room_id]) > 50:
                room_speech_data[room_id].pop(0)
    
    speech_duration = max(2, len(cleaned_text) * 0.1)
    threading.Timer(speech_duration, lambda: reset_speaking_state(room_id, is_teacher)).start()

def text_to_speech(text, lang='ru'):
    """Преобразует текст в аудио (base64)"""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return base64.b64encode(mp3_fp.read()).decode('utf-8')
    except Exception as e:
        print(f"Error in text_to_speech: {e}")
        return None

def save_student_data(student_data):
    """Сохраняет данные ученика в JSON файл"""
    try:
        student_id = student_data.get('student_id')
        if not student_id:
            student_id = str(uuid.uuid4())
            student_data['student_id'] = student_id
        
        if 'conference_id' not in student_data:
            conference_id = str(int(time.time() * 1000))
            student_data['conference_id'] = conference_id
            debug_log(f"Создан идентификатор конференции для ученика {student_data.get('name')}: {conference_id}")
        
        student_data['last_updated'] = datetime.now().isoformat()
        
        filename = f"{student_id}.json"
        filepath = STUDENTS_DIR / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(student_data, f, ensure_ascii=False, indent=2)
        
        return student_id
    except Exception as e:
        print(f"Error saving student data: {e}")
        return None

def load_student_data(student_id):
    """Загружает данные ученика из JSON файла"""
    try:
        filename = f"{student_id}.json"
        filepath = STUDENTS_DIR / filename
        
        if not filepath.exists():
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading student data: {e}")
        return None

def find_student_by_name(name):
    """Находит ученика по имени"""
    try:
        for filepath in STUDENTS_DIR.glob("*.json"):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get('name', '').lower() == name.lower():
                    return data
        return None
    except Exception as e:
        print(f"Error finding student: {e}")
        return None

def update_student_data(student_id, updates):
    """Обновляет данные ученика"""
    try:
        current_data = load_student_data(student_id)
        if not current_data:
            return False
        
        current_data.update(updates)
        current_data['last_updated'] = datetime.now().isoformat()
        
        return save_student_data(current_data) is not None
    except Exception as e:
        print(f"Error updating student data: {e}")
        return False

def create_student_rooms(student_data):
    """Автоматически создает комнаты для ученика с единым идентификатором"""
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
            'math', 'physics', 'chemistry', 'biology', 
            'history', 'social', 'literature', 'russian', 
            'english', 'geography'
        ]
        
        created_rooms = []
        
        for subject in subjects:
            room_name = f"{subject}_{student_name.replace(' ', '_').lower()}_{conference_id}"
            
            room_student_data[room_name] = {
                'name': student_name,
                'age': student_data.get('age'),
                'level': student_data.get('education_level'),
                'subject': subject,
                'student_id': student_id,
                'conference_id': conference_id
            }
            
            # Инициализируем DialogueManager для комнаты
            _fast_room_initialization(room_name)
            
            debug_log(f"Создана комната {room_name} для ученика {student_name}")
            
            created_rooms.append({
                'subject': subject,
                'room_name': room_name,
                'avatar': 'woman',
                'conference_id': conference_id,
                'student_data': room_student_data[room_name]
            })
        
        student_data['rooms'] = created_rooms
        student_data['default_avatar'] = 'woman'
        student_data['conference_id'] = conference_id
        
        save_student_data(student_data)
        
        debug_log(f"Создано {len(created_rooms)} комнат для ученика {student_name} с ID: {conference_id}")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка создания комнат для ученика: {e}")
        return False

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
        if room_id not in room_participants:
            room_participants[room_id] = set()

        if room_id not in room_ai_activated:
            room_ai_activated[room_id] = False

        if room_id not in room_llm_mode:
            room_llm_mode[room_id] = get_llm_mode()

        _fast_room_initialization(room_id)

        join_room(room_id)
        room_participants[room_id].add(request.sid)
        
        if peer_id:
            if room_id not in room_peer_ids:
                room_peer_ids[room_id] = {}
            room_peer_ids[room_id][request.sid] = peer_id
        
        if room_id not in room_dialogue or room_dialogue[room_id] is None:
            debug_log(f"Экстренное создание DialogueManager для {room_id} после join")
            room_dialogue[room_id] = DialogueManager(socketio)
            room_dialogue[room_id].room_id = room_id
            room_dialogue[room_id].set_llm_mode(room_llm_mode[room_id])
        
        if peer_id:
            emit('participant_joined', {
                'peer_id': peer_id,
                'sid': request.sid
            }, room=room_id, include_self=False)
        
        try:
            emit('current_avatar', {'avatar_name': room_current_avatar[room_id]}, to=request.sid)
        except Exception as e:
            print(f"⚠️ Ошибка отправки аватара: {e}")
        
        if room_id in room_speech_data and room_speech_data[room_id]:
            try:
                emit('speech_history', {'history': room_speech_data[room_id]}, to=request.sid)
            except Exception as e:
                print(f"⚠️ Ошибка отправки истории: {e}")
        
        emit('participants_update', {'count': len(room_participants[room_id])}, room=room_id)
        
        if '_' in room_id and room_id != 'default' and len(room_participants[room_id]) == 1:
            debug_log(f"Запланирована автоматическая активация AI для комнаты ученика {room_id}")
            socketio.start_background_task(delayed_auto_activation, room_id)
        
        elif len(room_participants[room_id]) == 1 and not room_ai_activated[room_id]:
            greeting = "Привет! Я ваш виртуальный учитель. Давайте познакомимся и выберем интересный урок вместе!"
            speak_text(room_id, greeting, voice_type='female', is_teacher=True)
        
        debug_log(f"Успешное присоединение к комнате {room_id}, участников: {len(room_participants[room_id])}")
        
    except Exception as e:
        print(f"❌ Критическая ошибка при присоединении к комнате {room_id}: {e}")
        try:
            emit('room_error', {
                'room_id': room_id,
                'error': f'Join room failed: {str(e)}'
            }, to=request.sid)
        except:
            print("⚠️ Не удалось отправить ошибку - клиент уже отключен")

def delayed_auto_activation(room_id, delay=3):
    """Улучшенная отложенная автоматическая активация с повторными попытками"""
    time.sleep(delay)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            debug_log(f"Попытка автоматической активации {room_id} (попытка {attempt + 1})")
            
            room_exists = (
                room_id in room_participants or 
                room_id in room_dialogue or 
                room_id in room_current_avatar
            )
            
            if (room_exists and 
                room_id in room_participants and 
                len(room_participants[room_id]) > 0 and 
                not room_ai_activated.get(room_id, False)):
                
                debug_log(f"Запуск автоматической активации для {room_id}")
                
                room_ai_activated[room_id] = True
                _fast_room_initialization(room_id)
                
                socketio.emit('ai_teacher_activated', {
                    'room_id': room_id,
                    'message': 'AI-учитель автоматически активирован',
                    'is_student_mode': hasattr(room_dialogue.get(room_id), 'is_student_mode') and room_dialogue[room_id].is_student_mode if room_id in room_dialogue else False
                }, room=room_id)
                
                debug_log(f"Автоматическая активация успешна для {room_id}")
                break
                
            else:
                debug_log(f"Комната {room_id} не готова для автоматической активации (попытка {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    
        except Exception as e:
            print(f"⚠️ Ошибка автоматической активации {room_id} (попытка {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

@socketio.on('get_current_avatar')
def handle_get_current_avatar(data):
    """Отправляет текущий аватар комнаты"""
    room_id = data['room_id']
    emit('current_avatar', {'avatar_name': room_current_avatar[room_id]}, to=request.sid)

@socketio.on('client_start_animation')
def handle_client_start_animation(data):
    """Обработчик команды запуска анимации от учителя"""
    room_id = data['room_id']
    avatar_name = data['avatar_name']
    debug_log(f"Получена команда запуска анимации для комнаты {room_id}, аватар: {avatar_name}")
    
    room_current_avatar[room_id] = avatar_name
    emit('avatar_changed', {'avatar_name': avatar_name}, room=room_id)
    emit('animation_ready', {'status': 'ready'}, room=room_id)

@socketio.on('avatar_changed')
def handle_avatar_changed(data):
    """Обработчик смены аватара"""
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
    speak_text(room_id, text, voice_type)

@socketio.on('student_answer')
def handle_student_answer(data):
    """Обработчик ответов ученика во время практики с последовательной генерацией"""
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
            speak_text(room_id, response, voice_type='female', is_teacher=True)
            emit('practice_ended', {}, room=room_id)
        return

    if room_id in room_dialogue:
        dialogue = room_dialogue[room_id]
        if not dialogue.waiting_for_answer:
            debug_log(f"Система не ожидает ответа, игнорирую: {answer}")
            return

    if any(cmd in answer.lower() for cmd in ['продолжай', 'дальше', 'следующий']):
        debug_log(f"Игнорирую команду вместо ответа: {answer}")
        if room_id in room_dialogue:
            response = room_dialogue[room_id]._evaluate_and_generate_next("")
            if response:
                emit('speech_text', {
                    'text': f"Учитель: {response}",
                    'sid': 'teacher',
                    'is_teacher': True
                }, room=room_id)
                speak_text(room_id, response, voice_type='female', is_teacher=True)
        return

    room_speech_data[room_id].append({
        'text': f"Ответ ученика: {answer}",
        'timestamp': time.time(),
        'type': 'practice_answer',
        'sid': user_sid
    })
    
    if room_id in room_dialogue:
        debug_log(f"Обработка ответа через диалог менеджер...")
        
        response = room_dialogue[room_id]._evaluate_and_generate_next(answer)
        
        if response:
            debug_log(f"Ответ учителя: {response}")
            
            emit('speech_text', {
                'text': f"Учитель: {response}",
                'sid': 'teacher',
                'is_teacher': True
            }, room=room_id)
            
            speak_text(room_id, response, voice_type='female', is_teacher=True)
            
            if not room_dialogue[room_id].practice_active:
                room_practice_active[room_id] = False
                room_current_question_index[room_id] = 0
                emit('practice_ended', {}, room=room_id)
                debug_log("Практика завершена")
        else:
            room_practice_active[room_id] = False
            room_current_question_index[room_id] = 0
            room_dialogue[room_id].waiting_for_answer = False
            emit('practice_ended', {}, room=room_id)
            debug_log("Практика завершена (response=None)")

@socketio.on('student_message')
def handle_student_message(data):
    """Обработчик сообщений от ученика через текстовое поле"""
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

    if not room_ai_activated.get(room_id, False):
        return
        
    if room_id not in room_dialogue or room_dialogue[room_id] is None:
        debug_log(f"DialogueManager отсутствует для комнаты {room_id}, пытаемся создать...")
        if not _fast_room_initialization(room_id):
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
                    speak_text(room_id, next_paragraph, voice_type='female', is_teacher=True)
                else:
                    practice_msg = "Урок завершен. Переходим к практике."
                    emit('speech_text', {
                        'text': f"Учитель: {practice_msg}",
                        'sid': 'teacher', 
                        'is_teacher': True
                    }, room=room_id)
                    speak_text(room_id, practice_msg, voice_type='female', is_teacher=True)
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
                
                speak_text(room_id, response, voice_type='female', is_teacher=True)

@socketio.on('activate_ai_teacher')
def handle_activate_ai_teacher(data):
    room_id = data['room_id']
    sid = request.sid
    
    debug_log(f"Запрос активации AI-учителя для комнаты {room_id} от {sid}")
    
    try:
        room_ai_activated[room_id] = True
        
        if room_id not in room_dialogue or room_dialogue[room_id] is None:
            room_dialogue[room_id] = DialogueManager(socketio)
            room_dialogue[room_id].room_id = room_id
            debug_log(f"Создан DialogueManager при активации для {room_id}")
        
        room_dialogue[room_id].set_llm_mode(room_llm_mode[room_id])
        
        greeting = "Привет! Я ваш AI-учитель. Давайте пообщаемся и выберем интересный урок вместе!"
        speak_text(room_id, greeting, voice_type='female', is_teacher=True)
        
        emit('ai_teacher_activated', {
            'room_id': room_id,
            'message': 'AI-учитель успешно активирован'
        }, room=room_id)
        
        debug_log(f"AI-учитель успешно активирован в комнате {room_id}")
        
    except Exception as e:
        print(f"❌ Ошибка активации AI-учителя: {e}")
        emit('activate_ai_error', {
            'room_id': room_id,
            'error': f'Ошибка активации: {str(e)}'
        }, to=sid)

@socketio.on('set_llm_mode')
def handle_set_llm_mode(data):
    room_id = data['room_id']
    mode = data['mode']
    
    if mode in ["traditional", "llm_first"]:
        room_llm_mode[room_id] = mode
        if room_id in room_dialogue:
            room_dialogue[room_id].set_llm_mode(mode)
        
        emit('llm_mode_changed', {
            'mode': mode,
            'room': room_id
        }, room=room_id)
        
        debug_log(f"Режим LLM изменен в комнате {room_id}: {mode}")

@socketio.on('llm_response_ready')
def handle_llm_response_ready(data):
    """Обработчик готовых ответов от LLM (для асинхронной обработка)"""
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
    
    speak_text(room_id, answer, voice_type='female', is_teacher=True)

@socketio.on('practice_started')
def handle_practice_started(data):
    """Обработчик начала фазы практики"""
    room_id = data['room_id']
    room_practice_active[room_id] = True
    room_current_question_index[room_id] = 0
    emit('practice_started', {}, room=room_id)
    debug_log(f"Практика начата в комнате {room_id}")

@socketio.on('practice_ended')
def handle_practice_ended(data):
    """Обработчик завершения фазы практики"""
    room_id = data['room_id']
    room_practice_active[room_id] = False
    room_current_question_index[room_id] = 0
    emit('practice_ended', {}, room=room_id)
    debug_log(f"Практика завершена в комнате {room_id}")

@socketio.on('visualization_generated')
def handle_visualization_generated(data):
    """Обработчик готовых визуализаций - ТОЛЬКО SVG"""
    room_id = data['room_id']
    
    debug_log(f"Получена SVG инфографика для комнаты {room_id}: {data['topic'][:100]}...")
    
    emit('visualization_generated', {
        'room_id': room_id,
        'topic': data['topic'],
        'svg_code': data.get('svg_code', ''),
        'timestamp': data.get('timestamp', time.time()),
        'type': data.get('type', 'infographic')
    }, room=room_id)

@socketio.on('get_llm_status')
def handle_get_llm_status(data):
    """WebSocket обработчик получения статуса LLM"""
    room_id = data['room_id']
    
    if room_id in room_dialogue:
        status = room_dialogue[room_id].llm.get_llm_status()
        emit('llm_status_update', {
            'room_id': room_id,
            'status': status
        }, room=room_id)

@socketio.on('set_llm_priority')
def handle_set_llm_priority(data):
    """WebSocket установка приоритета"""
    room_id = data['room_id']
    priority = data['priority']
    
    valid_priorities = ["local_first", "openrouter_first", "local_only", "openrouter_only"]
    
    if priority not in valid_priorities:
        emit('llm_priority_error', {
            'room_id': room_id,
            'error': f'Invalid priority. Use: {valid_priorities}'
        })
        return
    
    if room_id in room_dialogue:
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
    """WebSocket получение статуса приоритета"""
    room_id = data['room_id']
    
    if room_id in room_dialogue:
        status = room_dialogue[room_id].llm.get_priority_status()
        emit('llm_priority_status', {
            'room_id': room_id,
            'status': status
        })

@socketio.on('async_llm_request')
def handle_async_llm_request(data):
    """Обработчик асинхронных запросов к LLM с улучшенным отслеживанием"""
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
    """Обработчик асинхронных ответов от LLM"""
    room_id = data['room_id']
    response = data['response']
    request_id = data['request_id']
    
    debug_log(f"Ответ для комнаты {room_id}: {response[:100]}...")
    
    if room_id in room_llm_pending_requests and request_id in room_llm_pending_requests[room_id]:
        del room_llm_pending_requests[room_id][request_id]
    
    if response and room_id in room_dialogue:
        room_dialogue[room_id].llm.handle_llm_response(request_id, response, room_id)
        
        emit('speech_text', {
            'text': f"Учитель: {response}",
            'sid': 'teacher',
            'is_teacher': True
        }, room=room_id)
        
        speak_text(room_id, response, voice_type='female', is_teacher=True)

@socketio.on('generate_visualization')
def handle_generate_visualization(data):
    """Обработчик запроса генерации инфографики - НЕМЕДЛЕННАЯ ГЕНЕРАЦИЯ SVG"""
    room_id = data['room_id']
    topic = data.get('topic', '')
    context = data.get('context', '')
    
    if not topic:
        return
    
    debug_log(f"WebSocket генерация SVG инфографики для комнаты {room_id}: {topic[:100]}...")
    
    # НЕМЕДЛЕННАЯ ГЕНЕРАЦИЯ И ОТПРАВКА SVG
    try:
        from llm import LLMIntegration
        llm = LLMIntegration()
        
        result = llm.generate_infographic(topic, context)
        svg_code = result["svg_code"] if result and result.get("success") else generate_svg_code(topic, context)
        
        # НЕМЕДЛЕННО отправляем инфографику
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
        # Fallback
        emit('visualization_generated', {
            'room_id': room_id,
            'topic': topic,
            'svg_code': generate_svg_code(topic, context),
            'timestamp': time.time(),
            'type': 'fallback'
        }, room=room_id)

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
    
    return render_template('conference.html', 
                         room_id=room_id, 
                         embed=embed,
                         student_mode=student_mode,
                         subject=subject,
                         subject_name=subject_name)

def add_visualization_to_queue(room_id, topic, context):
    """Добавляет инфографику в очередь для комнаты - ТОЛЬКО SVG"""
    if room_id not in room_visualization_queue:
        room_visualization_queue[room_id] = []
    
    # Генерируем только SVG инфографику
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
    """Endpoint для polling визуализаций - ТОЛЬКО SVG"""
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
    """Статус очереди визуализаций - ТОЛЬКО SVG"""
    room_id = request.args.get('room_id', 'default')
    
    return jsonify({
        "success": True,
        "room_id": room_id,
        "queue_length": len(room_visualization_queue.get(room_id, [])),
        "active": room_visualization_active.get(room_id, False),
        "queue": room_visualization_queue.get(room_id, [])
    })

def generate_svg_code(topic: str, context: str = "") -> str:
    """Генерация SVG инфографики через LLM"""
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
        print(f"❌ Ошибка генерации SVG инфографики: {e}")
    
    # Fallback - простая SVG схема
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
    """Генерация инфографики через LLM - ТОЛЬКО SVG"""
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

@app.route('/api/llm/priority', methods=['POST'])
def set_llm_priority_route():
    """Установка приоритета моделей LLM"""
    try:
        data = request.json
        priority = data.get('priority')
        
        if not priority:
            return jsonify({"success": False, "error": "Priority not specified"})
        
        success = set_llm_priority(priority)
        
        if success:
            for room_id in room_dialogue:
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
    """Получение текущего приоритета моделей LLM"""
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
    """Получение статуса приоритетов"""
    room_id = request.args.get('room_id', 'default')
    
    if room_id in room_dialogue:
        status = room_dialogue[room_id].llm.get_priority_status()
        return jsonify({
            "success": True,
            "room": room_id,
            "status": status
        })
    
    return jsonify({"success": False, "error": "Room not found"})

@app.route('/api/llm/available_priorities')
def get_available_priorities():
    """Получение доступных приоритетов"""
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
    """Получение статуса LLM моделей"""
    room_id = request.args.get('room_id', 'default')
    
    if room_id in room_dialogue:
        status = room_dialogue[room_id].llm.get_llm_status()
        return jsonify({
            "success": True,
            "room": room_id,
            "status": status
        })
    
    return jsonify({"success": False, "error": "Room not found"})

@app.route('/api/llm/local_status')
def get_local_llm_status():
    """Получение статуса локальной модели"""
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
    """Получение статуса менеджера LLM"""
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
    """Улучшенный polling endpoint для получения ответов от LLM"""
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
    """Очистка очереди ответов LLM"""
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
    """Установка модели LLM для комнаты"""
    try:
        data = request.json
        model = data.get('model')
        room_id = data.get('room_id', 'default')
        
        if not model:
            return jsonify({"success": False, "error": "Model not specified"})
        
        if room_id in room_dialogue:
            room_dialogue[room_id].set_llm_model(model)
            return jsonify({"success": True, "model": model, "room": room_id})
        
        return jsonify({"success": False, "error": "Room not found"})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/llm/models', methods=['GET'])
def get_llm_models():
    """Получение списка доступных моделей LLM"""
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
    """Получение текущего режима работы LLM"""
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
    """Установка режима работы LLM"""
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
                if room_id in room_dialogue:
                    room_dialogue[room_id].set_llm_mode(mode)
            
            return jsonify({
                "success": True,
                "message": f"Режим LLM успешно изменен на '{mode}'",
                "mode": mode
            })
        else:
            return jsonify({"success": False, "error": "Failed to save config"})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/knowledge/stats', methods=['GET'])
def get_knowledge_stats():
    """Получение статистики базы знаний для комнаты"""
    room_id = request.args.get('room_id', 'default')
    subject = request.args.get('subject', '')
    
    if room_id in room_dialogue:
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
    """Поиск в базе знаний"""
    room_id = request.args.get('room_id', 'default')
    query = request.args.get('query', '')
    max_results = int(request.args.get('max_results', 5))
    
    if not query:
        return jsonify({"success": False, "error": "Query parameter is required"})
    
    if room_id in room_dialogue and room_dialogue[room_id].knowledge_base:
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
    """Получение списка ответов LLM для предмета"""
    room_id = request.args.get('room_id', 'default')
    subject = request.args.get('subject', '')
    
    if room_id in room_dialogue and room_dialogue[room_id].knowledge_base:
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
    """Получение содержания урока"""
    try:
        lesson_file = LESSONS_DIR / f"{lesson_id}.txt"
        if not lesson_file.exists():
            return jsonify({"error": "Lesson not found"}), 404
        
        with open(lesson_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        paragraphs = []
        current_paragraph = []
        
        if '\n\n' in content:
            raw_paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        else:
            raw_paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
        
        for paragraph in raw_paragraphs:
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) >= 6:
                paragraphs.append(' '.join(sentences))
                continue
                
            current_paragraph.extend(sentences)
            
            if len(current_paragraph) >= 6:
                paragraphs.append(' '.join(current_paragraph[:6]))
                current_paragraph = current_paragraph[6:]
        
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        paragraphs = [p.replace('\n\n', ' ').replace('\n', ' ') for p in paragraphs]
        
        return jsonify({
            "success": True,
            "lesson_id": lesson_id,
            "content": paragraphs,
            "paragraph_count": len(paragraphs)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/lessons')
def get_available_lessons():
    """Получение списка доступных уроков"""
    try:
        lessons = {}
        for lesson_file in LESSONS_DIR.glob("*.txt"):
            subject = _detect_subject(lesson_file.stem)
            
            if subject not in lessons:
                lessons[subject] = []
            
            lessons[subject].append({
                'id': lesson_file.stem,
                'title': lesson_file.stem.replace('_', ' ').title(),
                'file_path': lesson_file.name,
                'type': 'text'
            })
        
        return jsonify({
            "success": True,
            "lessons": lessons
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _detect_subject(filename: str) -> str:
    """Определяет предмет по названию файла"""
    filename_lower = filename.lower()
    if any(word in filename_lower for word in ['math', 'математика', 'алгебра', 'геометрия']):
        return "математика"
    elif any(word in filename_lower for word in ['history', 'история', 'истор']):
        return "история"
    elif any(word in filename_lower for word in ['physics', 'физика', 'физ']):
        return "физика"
    elif any(word in filename_lower for word in ['chemistry', 'химия', 'хим']):
        return "химия"
    elif any(word in filename_lower for word in ['social', 'обществознание', 'общество']):
        return "обществознание"
    elif any(word in filename_lower for word in ['biology', 'биология', 'био']):
        return "биология"
    elif any(word in filename_lower for word in ['literature', 'литература', 'лит']):
        return "литература"
    elif any(word in filename_lower for word in ['russian', 'русский', 'язык']):
        return "русский язык"
    else:
        return "общее"

@app.route('/api/practice_content/<lesson_id>')
def get_practice_content(lesson_id):
    """Получение практических заданий для урока"""
    try:
        practice_file = PRACTICE_DIR / f"{lesson_id}.json"
        if not practice_file.exists():
            return jsonify({"error": "Практические задания не найдены", "success": False}), 404
        
        with open(practice_file, 'r', encoding='utf-8') as f:
            content = json.load(f)
        
        return jsonify({
            "success": True,
            'lesson_id': lesson_id,
            'content': content,
            'question_count': len(content.get('questions', []))
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/api/practice_files')
def get_practice_files():
    """Получение списка файлов практики"""
    try:
        practice_files = []
        for practice_file in PRACTICE_DIR.glob("*.json"):
            practice_files.append({
                'filename': practice_file.name,
                'size': practice_file.stat().st_size,
                'modified': datetime.fromtimestamp(practice_file.stat().st_mtime).isoformat()
            })
        
        return jsonify({
            "success": True,
            "files": practice_files
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/practice_txt_files')
def get_practice_txt_files():
    """Получение списка TXT файлов практики"""
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
    """Загрузка файла практики"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"})
        
        if file and file.filename.endswith('.json'):
            filename = secure_filename(file.filename)
            file.save(PRACTICE_DIR / filename)
            
            return jsonify({
                "success": True,
                "message": f"File {filename} uploaded successfully",
                "filename": filename
            })
        else:
            return jsonify({"success": False, "error": "Invalid file type. Only JSON allowed"})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/delete_practice/<filename>')
def delete_practice(filename):
    """Удаление файла практики"""
    try:
        practice_file = PRACTICE_DIR / filename
        if not practice_file.exists():
            return jsonify({"success": False, "error": "File not found"})
        
        practice_file.unlink()
        return jsonify({
            "success": True,
            "message": f"File {filename} deleted successfully"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/add_knowledge', methods=['POST'])
def add_knowledge():
    """Добавление знаний в базу"""
    try:
        data = request.json
        subject = data.get('subject', 'общее')
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({"success": False, "error": "Text is required"})
        
        knowledge_file = MATERIALS_DIR / f"{subject}_knowledge.json"
        if knowledge_file.exists():
            with open(knowledge_file, 'r', encoding='utf-8') as f:
                knowledge_data = json.load(f)
        else:
            knowledge_data = {
                "terms": {},
                "questions": {},
                "examples": {},
                "metadata": {
                    "subject": subject,
                    "version": "1.0",
                    "last_updated": datetime.now().isoformat(),
                    "author": "AI Teacher System"
                }
            }
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines:
            if ' - ' in line:
                term, definition = line.split(' - ', 1)
                knowledge_data["terms"][term.strip().lower()] = definition.strip()
            elif line.endswith('?'):
                knowledge_data["questions"][line.strip().lower()] = "Ответ будет добавлен автоматически"
            else:
                if "general_info" not in knowledge_data:
                    knowledge_data["general_info"] = []
                knowledge_data["general_info"].append(line.strip())
        
        with open(knowledge_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_data, f, ensure_ascii=False, indent=2)
        
        return jsonify({"success": True, "subject": subject, "added_items": len(lines)})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/add_lesson', methods=['POST'])
def add_lesson():
    """Добавление нового урока"""
    try:
        data = request.json
        subject = data.get('subject', 'общее')
        title = data.get('title', '')
        content = data.get('content', '')
        
        if not title or not content:
            return jsonify({"success": False, "error": "Title and content are required"})
        
        filename = f"{subject}_{title.lower().replace(' ', '_')}.txt"
        lesson_path = LESSONS_DIR / filename
        
        with open(lesson_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return jsonify({"success": True, "filename": filename, "subject": subject, "title": title})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/add_practice', methods=['POST'])
def add_practice():
    """Добавление практических заданий"""
    try:
        data = request.json
        lesson_id = data.get('lesson_id', '')
        practice_data = data.get('practice_data', {})
        
        if not lesson_id or not practice_data:
            return jsonify({"success": False, "error": "Lesson ID and practice data are required"})
        
        practice_file = PRACTICE_DIR / f"{lesson_id}.json"
        
        with open(practice_file, 'w', encoding='utf-8') as f:
            json.dump(practice_data, f, ensure_ascii=False, indent=2)
        
        return jsonify({"success": True, "lesson_id": lesson_id, "question_count": len(practice_data.get('questions', []))})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/download_knowledge')
def download_knowledge():
    """Скачивание базы знаний"""
    subject = request.args.get('subject', 'обществознание')
    knowledge_file = MATERIALS_DIR / f"{subject}_knowledge.json"
    llm_answers_file = MATERIALS_DIR / f"{subject}_llm_answers.json"
    
    if not knowledge_file.exists() and not llm_answers_file.exists():
        return jsonify({"success": False, "error": f"База знаний для предмета '{subject}' не найдена"})
    
    import tempfile
    import zipfile
    
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    
    with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
        if knowledge_file.exists():
            zipf.write(knowledge_file, f"{subject}_knowledge.json")
        if llm_answers_file.exists():
            zipf.write(llm_answers_file, f"{subject}_llm_answers.json")
    
    temp_zip.close()
    
    return send_file(
        temp_zip.name,
        as_attachment=True,
        download_name=f"{subject}_knowledge_base.zip",
        mimetype='application/zip'
    )

@app.route('/api/download_lessons')
def download_lessons():
    """Скачивание всех уроков"""
    if not any(LESSONS_DIR.iterdir()):
        return jsonify({"success": False, "error": "Уроки не найдены"})
    
    import tempfile
    import zipfile
    
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    
    with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
        for lesson_file in LESSONS_DIR.glob("*.txt"):
            zipf.write(lesson_file, lesson_file.name)
    
    temp_zip.close()
    
    return send_file(
        temp_zip.name,
        as_attachment=True,
        download_name="ai_teacher_lessons.zip",
        mimetype='application/zip'
    )

@app.route('/api/download_practice')
def download_practice():
    """Скачивание всех практических заданий"""
    if not any(PRACTICE_DIR.iterdir()):
        return jsonify({"success": False, "error": "Практические задания не найдены"})
    
    import tempfile
    import zipfile
    
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    
    with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
        for practice_file in PRACTICE_DIR.glob("*.json"):
            zipf.write(practice_file, practice_file.name)
    
    temp_zip.close()
    
    return send_file(
        temp_zip.name,
        as_attachment=True,
        download_name="ai_teacher_practice.zip",
        mimetype='application/zip'
    )

@app.route('/api/download_practice_txt')
def download_practice_txt():
    """Скачивание практических заданий в текстовом формате"""
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
    """Получение текущих API ключей"""
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
    """Установка API ключа"""
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
    """Тестирование API ключа"""
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
    """Принудительная генерация SVG инфографики для тестирования"""
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
    """Проверка здоровья системы"""
    try:
        local_status = llm_manager.local_llm.get_status()
        openrouter_available = bool(get_api_key('openrouter'))
        lessons_available = any(LESSONS_DIR.iterdir())
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "local_llm": local_status,
                "openrouter": {"available": openrouter_available},
                "lessons": {"available": lessons_available, "count": len(list(LESSONS_DIR.glob("*.txt")))},
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
    """Диагностика проблем с OpenRouter"""
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

@app.route('/api/keys/status', methods=['GET'])
def get_keys_status():
    """Получение статуса API ключей"""
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
def add_api_key_route():
    """Добавление нового API ключа"""
    try:
        data = request.json
        api_key = data.get('api_key')
        name = data.get('name', 'new_key')
        
        if not api_key:
            return jsonify({"success": False, "error": "API key is required"})
        
        key_manager = get_key_manager()
        key_manager.add_key(api_key, name)
        
        return jsonify({
            "success": True,
            "message": f"Ключ {name} успешно добавлен",
            "total_keys": len(key_manager.keys)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/student/save', methods=['POST'])
def save_student():
    """Сохранение данных ученика с автоматическим созданием комнат"""
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
        
        existing_student = find_student_by_name(student_data['name'])
        
        if existing_student:
            student_id = existing_student['student_id']
            student_data['student_id'] = student_id
            student_data['rooms'] = existing_student.get('rooms', [])
            student_data['conference_id'] = existing_student.get('conference_id')
            update_student_data(student_id, {
                'education_level': student_data['education_level'],
                'age': student_data['age'],
                'last_login': student_data['last_login']
            })
        else:
            student_id = save_student_data(student_data)
            student_data['student_id'] = student_id
            
            create_student_rooms(student_data)
            
            updated_data = load_student_data(student_id)
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
    """Получение данных ученика"""
    try:
        student_data = load_student_data(student_id)
        if student_data:
            return jsonify({"success": True, "student": student_data})
        else:
            return jsonify({"success": False, "error": "Ученик не найден"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/student/<student_id>/update', methods=['POST'])
def update_student(student_id):
    """Обновление данных ученика"""
    try:
        data = request.json
        if update_student_data(student_id, data):
            return jsonify({"success": True, "message": "Данные обновлены"})
        else:
            return jsonify({"success": False, "error": "Ошибка обновления"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/student/<student_id>/rooms')
def get_student_rooms(student_id):
    """Получение списка комнат ученика"""
    try:
        student_data = load_student_data(student_id)
        if not student_data:
            return jsonify({"success": False, "error": "Ученик не найден"})
        
        rooms = student_data.get('rooms', [])
        return jsonify({
            "success": True,
            "student_id": student_id,
            "student_name": student_data.get('name'),
            "rooms": rooms,
            "default_avatar": student_data.get('default_avatar', 'woman'),
            "total_rooms": len(rooms)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/student/<student_id>/room/<subject>')
def get_student_room(student_id, subject):
    """Получение конкретной комнаты ученика по предмету"""
    try:
        student_data = load_student_data(student_id)
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
    """Получение истории уроков ученика"""
    try:
        student_data = load_student_data(student_id)
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
    """Добавление урока в историю ученика"""
    try:
        data = request.json
        student_data = load_student_data(student_id)
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
        
        if save_student_data(student_data):
            return jsonify({"success": True, "message": "Урок добавлен в историю"})
        else:
            return jsonify({"success": False, "error": "Ошибка сохранения"})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/admin/create_all_student_rooms')
def create_all_student_rooms():
    """Админская функция для создания комнат всем существующим ученикам"""
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
def update_student_conference_ids():
    """Обновляет conference_id для всех существующих учеников"""
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

@app.route('/api/room/initialize', methods=['POST'])
def force_room_initialization():
    """Принудительная инициализация комнаты"""
    try:
        data = request.json
        room_id = data.get('room_id')
        
        if not room_id:
            return jsonify({"success": False, "error": "Room ID is required"})
        
        success = _fast_room_initialization(room_id)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Комната {room_id} инициализирована",
                "ready": room_id in room_dialogue and room_dialogue[room_id] is not None
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
    """Получение статуса комнаты"""
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
    """Проверка здоровья комнаты"""
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
    """Создание тестовых пользователей для отладки"""
    try:
        # Учитель
        teacher = create_new_teacher('teacher', '123456')
        # Ученик  
        student = create_new_student('student', '123456')
        
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
    """Отладочная информация о маршрутах"""
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
    """Тестовый маршрут для проверки открытия конференции"""
    return render_template('conference.html', 
                         room_id='test_room', 
                         embed=False,
                         student_mode=True,
                         subject='math',
                         subject_name='Математика')

@app.route('/test-student-flow')
def test_student_flow():
    """Тестовая страница для проверки потока ученика"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Student Flow</title>
    </head>
    <body>
        <h1>Тест открытия конференции</h1>
        <button onclick="testOpenConference()">Тест открытия конференции</button>
        <button onclick="testOpenInNewTab()">Тест открытия в новой вкладке</button>
        <button onclick="testOpenInSameTab()">Тест открытия в этой же вкладке</button>
        
        <script>
            function testOpenConference() {
                const url = '/conference?room=test_room&student=true&subject=math&subject_name=Математика';
                const newWindow = window.open(url, 'TestConference', 'width=1200,height=800');
                if (!newWindow) {
                    alert('Окно заблокировано! Разрешите всплывающие окна.');
                }
            }
            
            function testOpenInNewTab() {
                const url = '/conference?room=test_room&student=true&subject=math&subject_name=Математика';
                window.open(url, '_blank');
            }
            
            function testOpenInSameTab() {
                const url = '/conference?room=test_room&student=true&subject=math&subject_name=Математика';
                window.location.href = url;
            }
        </script>
    </body>
    </html>
    """

# =============================================================================
# ЗАПУСК СЕРВЕРА
# =============================================================================

if __name__ == '__main__':
    debug_log("Запуск ИСПРАВЛЕННОЙ AI Teacher системы с аутентификацией...")
    
    setup_llm_manager()
    
    debug_log("Проверка конфигурации SocketIO...")
    debug_log(f"Async mode: {socketio.async_mode}")
    debug_log(f"Server: {socketio.server}")
    
    debug_log("Предварительная инициализация системных комнат...")
    system_rooms = ['default']
    for room in system_rooms:
        _fast_room_initialization(room)
    
    socketio.run(
        app, 
        host='0.0.0.0', 
        port=5000, 
        debug=True, 
        allow_unsafe_werkzeug=True
    )