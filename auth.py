# auth.py - Аутентификация и авторизация для AI Teacher System

from flask import session, redirect, url_for, jsonify, request
from functools import wraps
import json
from pathlib import Path
from datetime import datetime
import uuid
from typing import Optional, Dict, Any

# Настройки путей
BASE_DIR = Path(__file__).parent
USERS_DIR = BASE_DIR / "users_data"

# Создаем папку если не существует
USERS_DIR.mkdir(exist_ok=True)

# =============================================================================
# ДЕКОРАТОРЫ ДЛЯ ПРОВЕРКИ АУТЕНТИФИКАЦИИ
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
# ФУНКЦИИ ДЛЯ РАБОТЫ С ПОЛЬЗОВАТЕЛЯМИ
# =============================================================================

def load_user_data(user_id: str) -> Optional[Dict[str, Any]]:
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

def save_user_data(user_data: Dict[str, Any]) -> bool:
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

def authenticate_user(username: str, password: str, role: str) -> Optional[Dict[str, Any]]:
    """Аутентификация пользователя"""
    try:
        for user_file in USERS_DIR.glob("*.json"):
            with open(user_file, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
                if (user_data.get('username') == username and 
                    user_data.get('role') == role and 
                    user_data.get('password') == password):
                    return user_data
        return None
    except Exception as e:
        print(f"Authentication error: {e}")
        return None

def create_new_student(username: str, password: str) -> Optional[Dict[str, Any]]:
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

def create_new_teacher(username: str, password: str) -> Optional[Dict[str, Any]]:
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

def update_student_profile(user_id: str, student_data: Dict[str, Any]) -> bool:
    """Обновление профиля ученика"""
    try:
        user_data = load_user_data(user_id)
        if not user_data:
            return False
        
        user_data['student_data'] = student_data
        user_data['profile_complete'] = True
        user_data['profile_updated'] = datetime.now().isoformat()
        
        return save_user_data(user_data)
    except Exception as e:
        print(f"Error updating student profile: {e}")
        return False

def get_all_users() -> Dict[str, Any]:
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
                    'last_login': user_data.get('last_login'),
                    'profile_complete': user_data.get('profile_complete', False)
                })
        return {"success": True, "users": users}
    except Exception as e:
        return {"success": False, "error": str(e)}

def create_user(username: str, password: str, role: str) -> Dict[str, Any]:
    """Создание нового пользователя"""
    try:
        if not username or not password:
            return {"success": False, "error": "Заполните все поля"}
        
        # Проверяем существование пользователя
        for user_file in USERS_DIR.glob("*.json"):
            with open(user_file, 'r', encoding='utf-8') as f:
                existing_user = json.load(f)
                if existing_user.get('username') == username:
                    return {"success": False, "error": "Пользователь с таким логином уже существует"}
        
        # Создаем пользователя
        if role == 'student':
            user_data = create_new_student(username, password)
        elif role == 'teacher':
            user_data = create_new_teacher(username, password)
        else:
            return {"success": False, "error": "Неверная роль пользователя"}
        
        if user_data:
            return {
                "success": True,
                "message": f"Пользователь {username} успешно создан",
                "user_id": user_data['user_id']
            }
        else:
            return {"success": False, "error": "Ошибка создания пользователя"}
    except Exception as e:
        return {"success": False, "error": f"Ошибка: {str(e)}"}

def delete_user(user_id: str, current_user_id: str) -> Dict[str, Any]:
    """Удаление пользователя"""
    try:
        user_file = USERS_DIR / f"{user_id}.json"
        
        if not user_file.exists():
            return {"success": False, "error": "Пользователь не найден"}
        
        if current_user_id == user_id:
            return {"success": False, "error": "Нельзя удалить свой собственный аккаунт"}
        
        user_file.unlink()
        return {"success": True, "message": "Пользователь удален"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def check_auth_status() -> Dict[str, Any]:
    """Проверка статуса аутентификации"""
    if 'user_id' in session:
        user_data = load_user_data(session['user_id'])
        if user_data:
            return {
                "success": True,
                "role": user_data.get('role'),
                "user_id": session['user_id'],
                "username": user_data.get('username'),
                "profile_complete": user_data.get('profile_complete', False)
            }
    return {"success": False}

def logout_user() -> Dict[str, Any]:
    """Выход пользователя"""
    session.clear()
    return {"success": True, "message": "Успешный выход"}

# =============================================================================
# API МАРШРУТЫ ДЛЯ АУТЕНТИФИКАЦИИ
# =============================================================================

def register_auth_routes(app):
    """Регистрация маршрутов аутентификации в приложении"""
    
    @app.route('/api/auth/check')
    def api_check_auth():
        return jsonify(check_auth_status())
    
    @app.route('/auth/login', methods=['POST'])
    def api_auth_login():
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
                
                # Обновляем время последнего входа
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
    
    @app.route('/logout', methods=['POST'])
    def api_logout():
        return jsonify(logout_user())
    
    @app.route('/api/users')
    @teacher_required
    def api_get_all_users():
        result = get_all_users()
        return jsonify(result)
    
    @app.route('/api/users/create', methods=['POST'])
    @teacher_required
    def api_create_user():
        try:
            data = request.json
            username = data.get('username', '').strip()
            password = data.get('password', '').strip()
            role = data.get('role', 'student')
            
            result = create_user(username, password, role)
            return jsonify(result)
        except Exception as e:
            return jsonify({"success": False, "error": f"Ошибка: {str(e)}"})
    
    @app.route('/api/users/<user_id>', methods=['DELETE'])
    @teacher_required
    def api_delete_user(user_id):
        current_user_id = session.get('user_id', '')
        result = delete_user(user_id, current_user_id)
        return jsonify(result)
    
    @app.route('/auth/complete-profile', methods=['POST'])
    @student_required
    def api_complete_profile():
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
                return jsonify({
                    "success": True,
                    "message": "Профиль успешно сохранен",
                    "student_id": student_data['student_id']
                })
            else:
                return jsonify({"success": False, "error": "Ошибка сохранения профиля"})
        except Exception as e:
            return jsonify({"success": False, "error": f"Ошибка: {str(e)}"})
    
    @app.route('/api/student/profile')
    @student_required
    def api_get_student_profile():
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
    
    return app
