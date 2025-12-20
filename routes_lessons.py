# routes_lessons.py
from core import (
    app, socketio, BASE_DIR, LESSONS_DIR, STUDENTS_DIR,
    STUDENT_PROGRESS_DIR, debug_log, teacher_required
)
from flask import jsonify, request, send_file
from pathlib import Path
import re
import json
import shutil
import tempfile
import zipfile
from datetime import datetime

# Пути к подпапкам уроков
LESSONS_DEMO_DIR = LESSONS_DIR / "demo"
LESSONS_STUDENTS_DIR = LESSONS_DIR / "students"
LESSONS_GENERATED_DIR = LESSONS_DIR / "generated"

# Убедимся, что папки существуют
for d in [LESSONS_DEMO_DIR, LESSONS_STUDENTS_DIR, LESSONS_GENERATED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def json_serialize_paths(obj):
    """Рекурсивно преобразует Path в строки для JSON"""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: json_serialize_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_serialize_paths(i) for i in obj]
    return obj

def get_student_lessons_by_class(student_class: str):
    """Возвращает уроки для ученика по классу (для student progress dashboard)"""
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

@app.route('/api/lessons')
def get_available_lessons():
    """Получение списка доступных уроков (для студента)"""
    try:
        lessons = {}
        lesson_dirs = [LESSONS_DEMO_DIR, LESSONS_STUDENTS_DIR, LESSONS_GENERATED_DIR, LESSONS_DIR]
        for lesson_dir in lesson_dirs:
            if not lesson_dir.exists():
                continue
            for lesson_file in lesson_dir.glob("*.txt"):
                try:
                    # Пропускаем файлы в папке students, если они не в структурированной папке
                    if lesson_dir == LESSONS_STUDENTS_DIR and "_class" not in str(lesson_file):
                        continue
                    lesson_id = lesson_file.stem
                    lesson_number = extract_lesson_number(lesson_id)
                    lesson_title = format_lesson_title(lesson_id)
                    
                    # Определяем предмет и класс
                    if "_class" in str(lesson_file):
                        parts = lesson_file.parts
                        class_level = next((p.replace("_class", "") for p in parts if p.endswith("_class")), "11")
                        subject = next((p for p in reversed(parts) if p != lesson_file.name and "_class" not in p), "общее")
                    else:
                        class_level = "demo"
                        subject = "общее"
                    
                    lesson_data = {
                        'id': f"{class_level}_{subject}_{lesson_id}",
                        'title': lesson_title,
                        'file_path': str(lesson_file.relative_to(LESSONS_DIR)),
                        'type': 'student' if class_level != 'demo' else 'demo',
                        'subject': subject,
                        'class_level': class_level,
                        'lesson_number': lesson_number
                    }
                    
                    if subject not in lessons:
                        lessons[subject] = []
                    lessons[subject].append(lesson_data)
                except Exception as e:
                    debug_log(f"Ошибка загрузки урока {lesson_file}: {e}")
        
        # Сортируем уроки по номеру
        for subject in lessons:
            lessons[subject] = sorted(lessons[subject], key=lambda x: x['lesson_number'])
        
        return jsonify({"success": True, "lessons": lessons})
    except Exception as e:
        debug_log(f"❌ Ошибка получения уроков: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/lessons/structure', methods=['GET'])
@teacher_required
def get_lessons_structure():
    """Получение структуры уроков для админки"""
    try:
        structure = {'demo': [], 'generated': [], 'students': {}}
        
        # Demo уроки
        for lesson_file in LESSONS_DEMO_DIR.glob("*.txt"):
            structure['demo'].append({
                'name': lesson_file.name,
                'size': lesson_file.stat().st_size,
                'modified': datetime.fromtimestamp(lesson_file.stat().st_mtime).isoformat(),
                'path': str(lesson_file.relative_to(LESSONS_DIR))
            })
        
        # Сгенерированные уроки
        for lesson_file in LESSONS_GENERATED_DIR.glob("*.txt"):
            structure['generated'].append({
                'name': lesson_file.name,
                'size': lesson_file.stat().st_size,
                'modified': datetime.fromtimestamp(lesson_file.stat().st_mtime).isoformat(),
                'path': str(lesson_file.relative_to(LESSONS_DIR))
            })
        
        # Уроки по классам
        for class_dir in sorted(LESSONS_STUDENTS_DIR.glob("*_class")):
            if class_dir.is_dir():
                class_name = class_dir.name
                structure['students'][class_name] = {}
                for subject_dir in sorted(class_dir.iterdir()):
                    if subject_dir.is_dir():
                        subject_name = subject_dir.name
                        lesson_files = []
                        for lesson_file in sorted(subject_dir.glob("*.txt")):
                            lesson_files.append({
                                'name': lesson_file.name,
                                'size': lesson_file.stat().st_size,
                                'modified': datetime.fromtimestamp(lesson_file.stat().st_mtime).isoformat(),
                                'path': str(lesson_file.relative_to(LESSONS_DIR))
                            })
                        structure['students'][class_name][subject_name] = lesson_files
        
        serialized_structure = json_serialize_paths(structure)
        return jsonify({"success": True, "structure": serialized_structure})
    except Exception as e:
        debug_log(f"❌ Ошибка получения структуры уроков: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/lesson/start-specific', methods=['POST'])
def start_specific_lesson():
    """Запуск конкретного урока по ID (вызывается из фронтенда)"""
    from sockets import ensure_dialogue_manager_for_room, room_dialogue
    try:
        data = request.json
        lesson_id = data.get('lesson_id')
        room_id = data.get('room_id')
        if not lesson_id or not room_id:
            return jsonify({"success": False, "error": "Не указан ID урока или комнаты"})
        
        # Находим файл урока
        possible_paths = []
        # Ищем во всех возможных папках
        for class_dir in LESSONS_STUDENTS_DIR.glob("*_class"):
            for subject_dir in class_dir.iterdir():
                if subject_dir.is_dir():
                    possible_paths.append(subject_dir / f"{lesson_id}.txt")
        possible_paths.extend([
            LESSONS_DEMO_DIR / f"{lesson_id}.txt",
            LESSONS_GENERATED_DIR / f"{lesson_id}.txt",
            LESSONS_DIR / f"{lesson_id}.txt"
        ])
        
        lesson_file = None
        for path in possible_paths:
            if path.exists():
                lesson_file = path
                break
        
        if not lesson_file:
            return jsonify({"success": False, "error": "Урок не найден"})
        
        # Создаем DialogueManager если нужно
        if not ensure_dialogue_manager_for_room(room_id):
            return jsonify({"success": False, "error": "Не удалось создать DialogueManager"})
        
        dialogue = room_dialogue[room_id]
        selected_lesson = {
            'id': lesson_id,
            'title': lesson_id.replace('_', ' ').title(),
            'file_path': lesson_file,
            'subject': 'общее'
        }
        
        # Устанавливаем выбранный урок
        dialogue.selected_lesson = selected_lesson
        dialogue.current_subject = selected_lesson.get('subject')
        dialogue.lesson_started = True
        dialogue.current_state = "lesson_reading"
        
        lesson_content = dialogue._load_lesson_content(selected_lesson.get('file_path'))
        dialogue.lesson_content = lesson_content
        dialogue.current_paragraph = 0
        
        first_paragraph = lesson_content[0] if lesson_content else ""
        
        return jsonify({
            "success": True,
            "lesson": selected_lesson,
            "first_paragraph": first_paragraph
        })
    except Exception as e:
        debug_log(f"❌ Ошибка запуска урока: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/lesson_slide')
def serve_lesson_slide():
    """Отдаёт слайд урока по запросу (безопасная версия)"""
    slide_path = request.args.get('path', '').strip()
    if not slide_path:
        return "Slide path missing", 400

    # Защита от path traversal
    if slide_path.startswith('/') or slide_path.startswith('\\') or '..' in slide_path:
        return "Invalid path", 400

    try:
        full_path = (LESSONS_DIR / slide_path).resolve()
        lessons_base = LESSONS_DIR.resolve()
    except Exception:
        return "Invalid path", 400

    if not str(full_path).startswith(str(lessons_base)):
        return "Access denied", 403

    if not full_path.exists():
        return "File not found", 404

    mime = 'image/jpeg'
    lower_name = slide_path.lower()
    if lower_name.endswith('.png'):
        mime = 'image/png'
    elif lower_name.endswith('.mp4'):
        mime = 'video/mp4'

    return send_file(full_path, mimetype=mime)

@app.route('/api/lessons/create-sample', methods=['POST'])
@teacher_required
def create_sample_lessons():
    """Создание демо-уроков для тестирования"""
    try:
        created_count = 0
        demo_subjects = ['математика', 'физика', 'химия', 'биология', 'история', 'литература']
        for subject in demo_subjects:
            lesson_file = LESSONS_DEMO_DIR / f"demo_{subject}.txt"
            if not lesson_file.exists():
                with open(lesson_file, 'w', encoding='utf-8') as f:
                    f.write(f"""# Демо-урок: {subject.title()}

Это демо-урок по предмету {subject}.
Урок содержит примеры, упражнения и тестовые задания.
Это демо-версия для тестирования системы AI-учителя.
Приятного обучения!""")
                created_count += 1
        
        # Создание уроков для классов
        for class_level in ['5', '8', '11']:
            class_dir = LESSONS_STUDENTS_DIR / f"{class_level}_class"
            class_dir.mkdir(exist_ok=True)
            for subject in ['математика', 'литература']:
                subject_dir = class_dir / subject
                subject_dir.mkdir(exist_ok=True)
                if not any(subject_dir.glob("*.txt")):
                    for i in range(1, 4):
                        lesson_file = subject_dir / f"lesson_{i:02d}_введение.txt"
                        if not lesson_file.exists():
                            with open(lesson_file, 'w', encoding='utf-8') as f:
                                f.write(f"""# Урок {i}: Введение в {subject}

Это примерный урок {i} по предмету {subject} для {class_level} класса.
Содержание урока:
1. Основные понятия
2. Примеры и упражнения
3. Практические задания
Этот урок был создан автоматически для демонстрации работы системы.""")
                            created_count += 1
        
        return jsonify({
            "success": True,
            "message": f"Создано {created_count} примерных уроков",
            "created_count": created_count
        })
    except Exception as e:
        debug_log(f"❌ Ошибка создания демо-уроков: {e}")
        return jsonify({"success": False, "error": str(e)})

# ========================================================================
# API для админки: управление уроками
# ========================================================================

@app.route('/api/lessons/list', methods=['GET'])
@teacher_required
def get_all_lessons_list():
    """Получение полного списка уроков для редактирования"""
    try:
        lessons = []
        # Demo уроки
        for lesson_file in LESSONS_DEMO_DIR.glob("*.txt"):
            lessons.append({
                'type': 'demo',
                'class': 'demo',
                'subject': 'demo',
                'name': lesson_file.name,
                'full_path': str(lesson_file.relative_to(LESSONS_DIR)),
                'size': lesson_file.stat().st_size,
                'modified': datetime.fromtimestamp(lesson_file.stat().st_mtime).isoformat(),
                'can_edit': True,
                'can_delete': True
            })
        # Сгенерированные уроки
        for lesson_file in LESSONS_GENERATED_DIR.glob("*.txt"):
            lessons.append({
                'type': 'generated',
                'class': 'generated',
                'subject': 'auto',
                'name': lesson_file.name,
                'full_path': str(lesson_file.relative_to(LESSONS_DIR)),
                'size': lesson_file.stat().st_size,
                'modified': datetime.fromtimestamp(lesson_file.stat().st_mtime).isoformat(),
                'can_edit': True,
                'can_delete': True
            })
        # Уроки по классам
        for class_dir in LESSONS_STUDENTS_DIR.glob("*_class"):
            if class_dir.is_dir():
                class_name = class_dir.name.replace("_class", "")
                for subject_dir in class_dir.iterdir():
                    if subject_dir.is_dir():
                        subject_name = subject_dir.name
                        for lesson_file in subject_dir.glob("*.txt"):
                            lessons.append({
                                'type': 'student',
                                'class': class_name,
                                'subject': subject_name,
                                'name': lesson_file.name,
                                'full_path': str(lesson_file.relative_to(LESSONS_DIR)),
                                'size': lesson_file.stat().st_size,
                                'modified': datetime.fromtimestamp(lesson_file.stat().st_mtime).isoformat(),
                                'can_edit': True,
                                'can_delete': True
                            })
        
        lessons.sort(key=lambda x: x['modified'], reverse=True)
        return jsonify({"success": True, "total": len(lessons), "lessons": lessons})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/lesson/edit/<path:lesson_path>', methods=['GET'])
@teacher_required
def get_lesson_for_edit(lesson_path):
    """Получение содержимого урока для редактирования"""
    try:
        full_path = LESSONS_DIR / lesson_path
        if not full_path.exists():
            return jsonify({"success": False, "error": "Урок не найден"})
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({
            "success": True,
            "lesson_path": lesson_path,
            "content": content,
            "size": len(content),
            "filename": full_path.name
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/lesson/save', methods=['POST'])
@teacher_required
def save_edited_lesson():
    """Сохранение отредактированного урока"""
    try:
        data = request.json
        lesson_path = data.get('lesson_path')
        content = data.get('content')
        if not lesson_path or content is None:
            return jsonify({"success": False, "error": "Не указаны данные урока"})
        full_path = LESSONS_DIR / lesson_path
        if not full_path.exists():
            return jsonify({"success": False, "error": "Урок не найден"})
        
        # Создаем резервную копию
        backup_path = full_path.with_suffix('.txt.backup')
        shutil.copy2(full_path, backup_path)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return jsonify({
            "success": True,
            "message": "Урок успешно сохранен",
            "backup_created": str(backup_path.name)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/lesson/delete', methods=['POST'])
@teacher_required
def delete_lesson():
    """Удаление урока"""
    try:
        data = request.json
        lesson_path = data.get('lesson_path')
        if not lesson_path:
            return jsonify({"success": False, "error": "Не указан путь к уроку"})
        full_path = LESSONS_DIR / lesson_path
        if full_path.exists():
            full_path.unlink()
            return jsonify({"success": True, "message": "Урок удалён"})
        else:
            return jsonify({"success": False, "error": "Урок не найден"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/download_lessons_by_class', methods=['GET'])
@teacher_required
def download_lessons_by_class():
    """Скачивание уроков по классу в ZIP"""
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
                        lesson_files.extend(lesson_dir.glob("*.txt"))
        elif class_level == 'demo':
            lesson_files = list(LESSONS_DEMO_DIR.glob("*.txt"))
        elif class_level == 'generated':
            lesson_files = list(LESSONS_GENERATED_DIR.glob("*.txt"))
        else:
            class_dir = LESSONS_STUDENTS_DIR / f"{class_level}_class"
            if class_dir.exists():
                for subject_dir in class_dir.iterdir():
                    if subject_dir.is_dir():
                        lesson_files.extend(subject_dir.glob("*.txt"))
        
        if not lesson_files:
            return jsonify({"success": False, "error": f"Уроки для класса {class_level} не найдены"})
        
        # Создаем ZIP
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
        
        # Определяем имя файла
        if class_level == 'all':
            filename = "all_lessons.zip"
        elif class_level == 'demo':
            filename = "demo_lessons.zip"
        elif class_level == 'generated':
            filename = "generated_lessons.zip"
        else:
            filename = f"{class_level}_class_lessons.zip"
        
        return send_file(
            temp_zip.name,
            as_attachment=True,
            download_name=filename,
            mimetype='application/zip'
        )
    except Exception as e:
        debug_log(f"❌ Ошибка скачивания уроков: {e}")
        return jsonify({"success": False, "error": str(e)})

debug_log("✅ Роуты уроков зарегистрированы")
