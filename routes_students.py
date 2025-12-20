# routes_students.py
from core import (
    app, socketio, BASE_DIR, STUDENTS_DIR, STUDENT_PROGRESS_DIR,
    USERS_DIR, LESSONS_DIR, LESSONS_STUDENTS_DIR, debug_log, teacher_required
)
from flask import jsonify, request, send_file, session
from functools import wraps
import os
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime
import re

# ==============================
# 🔐 Вспомогательные функции авторизации
# ==============================

def student_required(f):
    """Декоратор для защиты студент-эндпоинтов"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"success": False, "error": "Не авторизован"}), 401
        user_data = load_user_data(session['user_id'])
        if not user_data or user_data.get('role') != 'student':
            return jsonify({"success": False, "error": "Требуется роль ученика"}), 403
        return f(*args, **kwargs)
    return decorated_function

# ==============================
# 📁 Вспомогательные функции работы с файлами
# ==============================

def load_user_data(user_id):
    """Загружает данные пользователя из USERS_DIR"""
    try:
        filepath = USERS_DIR / f"{user_id}.json"
        if not filepath.exists():
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        debug_log(f"Error loading user data {user_id}: {e}")
        return None

def save_user_data(user_data):
    """Сохраняет данные пользователя"""
    try:
        user_id = user_data.get('user_id')
        if not user_id:
            return False
        filepath = USERS_DIR / f"{user_id}.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(user_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        debug_log(f"Error saving user  {e}")
        return False

def load_student_data(student_id):
    """Загружает данные ученика из STUDENTS_DIR"""
    try:
        filepath = STUDENTS_DIR / f"{student_id}.json"
        if not filepath.exists():
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        debug_log(f"Error loading student data {student_id}: {e}")
        return None

def save_student_data(student_data):
    """Сохраняет данные ученика"""
    try:
        student_id = student_data.get('student_id')
        if not student_id:
            return None
        filepath = STUDENTS_DIR / f"{student_id}.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(student_data, f, ensure_ascii=False, indent=2)
        return student_id
    except Exception as e:
        debug_log(f"Error saving student  {e}")
        return None

def get_student_lessons_by_class(student_class: str):
    """Возвращает уроки для ученика по классу"""
    lessons_by_subject = {}
    class_dir = LESSONS_STUDENTS_DIR / f"{student_class}_class"
    if class_dir.exists() and class_dir.is_dir():
        for subject_dir in class_dir.iterdir():
            if subject_dir.is_dir():
                lessons = []
                for lesson_file in subject_dir.glob("*.txt"):
                    lesson_id = lesson_file.stem
                    lesson_number = extract_lesson_number(lesson_id)
                    title = format_lesson_title(lesson_id)
                    lessons.append({
                        'id': f"{student_class}_{subject_dir.name}_{lesson_id}",
                        'title': title,
                        'file_path': lesson_file,
                        'subject': subject_dir.name,
                        'class_level': student_class,
                        'lesson_number': lesson_number,
                        'type': 'student'
                    })
                if lessons:
                    lessons_by_subject[subject_dir.name] = sorted(lessons, key=lambda x: x['lesson_number'])
    return lessons_by_subject

def extract_lesson_number(filename: str) -> int:
    """Извлекает номер урока из имени файла"""
    match = re.search(r'(?:lesson|урок)[_\s]*(\d+)', filename.lower())
    if match:
        return int(match.group(1))
    match = re.search(r'^(\d+)', filename)
    if match:
        return int(match.group(1))
    return 999

def format_lesson_title(filename: str) -> str:
    """Форматирует заголовок урока из имени файла"""
    title = re.sub(r'(?:lesson|урок)[_\s]*\d+[_\s]*', '', filename, flags=re.IGNORECASE)
    title = title.replace('_', ' ').strip()
    return title.title() or filename

def initialize_student_progress(student_id: str, student_class: str):
    """Инициализирует файл прогресса ученика"""
    try:
        progress_file = STUDENT_PROGRESS_DIR / f"{student_id}.json"
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        
        lessons_by_subject = get_student_lessons_by_class(student_class)
        progress_data = {"subjects": {}}
        
        for subject, lessons in lessons_by_subject.items():
            lesson_ids = [lesson['id'] for lesson in lessons]
            progress_data["subjects"][subject] = {
                "completed_lessons": [],
                "current_lesson": None,
                "total_lessons": len(lesson_ids),
                "last_accessed": None
            }
        
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
        debug_log(f"✅ Создан начальный прогресс для ученика {student_id}")
    except Exception as e:
        debug_log(f"❌ Ошибка инициализации прогресса: {e}")

# ==============================
# 👤 Профиль ученика
# ==============================

@app.route('/api/student/profile')
@student_required
def get_student_profile():
    """Получение профиля ученика"""
    try:
        user_data = load_user_data(session['user_id'])
        if not user_
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

# ==============================
# 📊 Прогресс ученика
# ==============================

@app.route('/api/student/progress', methods=['GET'])
@student_required
def get_student_progress_api():
    """Получение прогресса ученика"""
    try:
        user_data = load_user_data(session['user_id'])
        if not user_data or not user_data.get('student_data'):
            return jsonify({"success": False, "error": "Данные ученика не найдены"})
        
        student_id = user_data['student_data'].get('student_id')
        student_class = user_data['student_data'].get('education_level', '5')
        if not student_id:
            return jsonify({"success": False, "error": "Student ID not found"})
        
        progress_file = STUDENT_PROGRESS_DIR / f"{student_id}.json"
        if not progress_file.exists():
            initialize_student_progress(student_id, student_class)
        
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)
        
        lessons_by_subject = get_student_lessons_by_class(student_class)
        result = {
            "student_id": student_id,
            "student_class": student_class,
            "student_name": user_data['student_data'].get('name', ''),
            "subjects": {}
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
            
            completed_count = len(completed_ids)
            total_lessons = len(lessons)
            
            # Ищем следующий урок
            next_lesson = None
            for lesson in subject_lessons:
                if not lesson['completed']:
                    next_lesson = lesson
                    break
            
            result['subjects'][subject_name] = {
                'total_lessons': total_lessons,
                'completed_lessons': completed_count,
                'progress_percent': int((completed_count / total_lessons) * 100) if total_lessons > 0 else 0,
                'last_updated': subject_progress.get("last_accessed"),
                'current_lesson': subject_progress.get("current_lesson"),
                'next_lesson': next_lesson,
                'lessons': subject_lessons
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
        debug_log(f"❌ Ошибка получения прогресса: {e}")
        return jsonify({"success": False, "error": str(e)})

# ==============================
# 👩‍🏫 Админка: управление учениками
# ==============================

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

@app.route('/api/student/<student_user_id>')
@teacher_required
def get_student_details(student_user_id):
    """Получение деталей ученика"""
    try:
        user_data = load_user_data(student_user_id)
        if not user_data or user_data.get('role') != 'student':
            return jsonify({"success": False, "error": "Ученик не найден"})
        
        progress_data = {}
        progress_file = STUDENT_PROGRESS_DIR / f"{user_data.get('student_data', {}).get('student_id', '')}.json"
        if progress_file.exists():
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
        
        return jsonify({
            "success": True,
            "student": {
                'user_id': user_data['user_id'],
                'username': user_data['username'],
                'student_data': user_data.get('student_data', {}),
                'created_at': user_data.get('created_at'),
                'last_login': user_data.get('last_login'),
                'rooms': user_data.get('student_data', {}).get('rooms', []),
                'progress': progress_data
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ==============================
# 📤 Экспорт данных
# ==============================

@app.route('/api/students/export_full', methods=['GET'])
@teacher_required
def export_students_full():
    """Полный экспорт данных учеников (включая прогресс)"""
    try:
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
            # 1. Экспорт пользователей
            for user_file in USERS_DIR.glob("*.json"):
                with open(user_file, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)
                if user_data.get('role') == 'student':
                    zipf.write(user_file, f"users/{user_file.name}")
            
            # 2. Экспорт данных учеников
            for student_file in STUDENTS_DIR.glob("*.json"):
                zipf.write(student_file, f"students/{student_file.name}")
            
            # 3. Экспорт прогресса
            for progress_file in STUDENT_PROGRESS_DIR.glob("*.json"):
                zipf.write(progress_file, f"progress/{progress_file.name}")
            
            # 4. Метаданные
            metadata = {
                "export_date": datetime.now().isoformat(),
                "total_users": len(list(USERS_DIR.glob("*.json"))),
                "total_students": len(list(STUDENTS_DIR.glob("*.json"))),
                "total_progress": len(list(STUDENT_PROGRESS_DIR.glob("*.json"))),
                "version": "1.0",
                "system": "AI Teacher System"
            }
            metadata_str = json.dumps(metadata, ensure_ascii=False, indent=2)
            metadata_path = tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w')
            metadata_path.write(metadata_str)
            metadata_path.close()
            zipf.write(metadata_path.name, "metadata.json")
            os.unlink(metadata_path.name)
        
        temp_zip.close()
        return send_file(
            temp_zip.name,
            as_attachment=True,
            download_name=f"students_full_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mimetype='application/zip'
        )
    except Exception as e:
        debug_log(f"❌ Ошибка экспорта данных: {e}")
        return jsonify({"success": False, "error": str(e)})

# ==============================
# 📥 Импорт данных
# ==============================

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
            return jsonify({"success": False, "error": "Требуется ZIP-файл"})
        
        # Распаковка
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, 'import.zip')
        file.save(file_path)
        
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        results = {
            "success": True,
            "imported": {"users": 0, "students": 0, "progress": 0},
            "errors": []
        }
        
        # Импорт пользователей
        users_dir = os.path.join(temp_dir, "users")
        if os.path.exists(users_dir):
            for user_file in os.listdir(users_dir):
                if user_file.endswith('.json'):
                    try:
                        src_path = os.path.join(users_dir, user_file)
                        dst_path = USERS_DIR / user_file
                        if dst_path.exists():
                            backup_path = USERS_DIR / f"{user_file}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            shutil.copy2(dst_path, backup_path)
                        shutil.copy2(src_path, dst_path)
                        results["imported"]["users"] += 1
                    except Exception as e:
                        results["errors"].append(f"Ошибка импорта пользователя {user_file}: {str(e)}")
        
        # Импорт данных учеников
        students_dir = os.path.join(temp_dir, "students")
        if os.path.exists(students_dir):
            for student_file in os.listdir(students_dir):
                if student_file.endswith('.json'):
                    try:
                        src_path = os.path.join(students_dir, student_file)
                        dst_path = STUDENTS_DIR / student_file
                        if dst_path.exists():
                            backup_path = STUDENTS_DIR / f"{student_file}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            shutil.copy2(dst_path, backup_path)
                        shutil.copy2(src_path, dst_path)
                        results["imported"]["students"] += 1
                    except Exception as e:
                        results["errors"].append(f"Ошибка импорта данных ученика {student_file}: {str(e)}")
        
        # Импорт прогресса
        progress_dir = os.path.join(temp_dir, "progress")
        if os.path.exists(progress_dir):
            for progress_file in os.listdir(progress_dir):
                if progress_file.endswith('.json'):
                    try:
                        src_path = os.path.join(progress_dir, progress_file)
                        dst_path = STUDENT_PROGRESS_DIR / progress_file
                        if dst_path.exists():
                            backup_path = STUDENT_PROGRESS_DIR / f"{progress_file}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            shutil.copy2(dst_path, backup_path)
                        shutil.copy2(src_path, dst_path)
                        results["imported"]["progress"] += 1
                    except Exception as e:
                        results["errors"].append(f"Ошибка импорта прогресса {progress_file}: {str(e)}")
        
        # Очистка
        shutil.rmtree(temp_dir, ignore_errors=True)
        return jsonify(results)
    except Exception as e:
        debug_log(f"❌ Ошибка импорта данных: {e}")
        return jsonify({"success": False, "error": str(e)})

# ==============================
# 🛠️ Служебные функции
# ==============================

@app.route('/api/debug/student-lessons')
@student_required
def debug_student_lessons_route():
    """Отладка: получение уроков ученика"""
    user_data = load_user_data(session['user_id'])
    student_class = user_data['student_data'].get('education_level', '5')
    result = get_student_lessons_by_class(student_class)
    return jsonify({"success": True, "lessons": result})

debug_log("✅ Роуты учеников зарегистрированы")
