# lesson_manager.py - Управление уроками и материалами для AI Teacher System

import os
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import uuid

# =============================================================================
# НАСТРОЙКИ ПУТЕЙ
# =============================================================================

BASE_DIR = Path(__file__).parent
LESSONS_DIR = BASE_DIR / 'lessons'
MATERIALS_DIR = BASE_DIR / 'materials'
PRACTICE_DIR = BASE_DIR / 'materials' / 'practice'
STUDENT_PROGRESS_DIR = BASE_DIR / "students_progress"

# Создаем необходимые папки
for folder in [LESSONS_DIR, MATERIALS_DIR, PRACTICE_DIR, STUDENT_PROGRESS_DIR]:
    os.makedirs(folder, exist_ok=True)

# Создаем новую структуру папок для уроков
LESSONS_DEMO_DIR = LESSONS_DIR / "demo"
LESSONS_STUDENTS_DIR = LESSONS_DIR / "students" 
LESSONS_GENERATED_DIR = LESSONS_DIR / "generated"
LESSONS_TRASH_DIR = LESSONS_DIR / "trash"
for folder in [LESSONS_DEMO_DIR, LESSONS_STUDENTS_DIR, LESSONS_GENERATED_DIR, LESSONS_TRASH_DIR]:
    os.makedirs(folder, exist_ok=True)

# =============================================================================
# КЛАСС ДЛЯ УПРАВЛЕНИЯ УРОКАМИ
# =============================================================================

class LessonManager:
    """Менеджер для управления уроками и материалами"""
    
    def __init__(self):
        self.lessons_cache = {}
        self.progress_cache = {}
        self.create_lessons_structure()
    
    # =============================================================================
    # СТРУКТУРА УРОКОВ ПО КЛАССАМ
    # =============================================================================
    
    def create_lessons_structure(self):
        """Создает структуру папок для уроков по классам (1-11 классы)"""
        subjects_by_class = {
            "1": ["русский язык", "литературное чтение", "математика", "окружающий мир", 
                  "английский язык", "французский язык", "информатика"],
            "2": ["русский язык", "литературное чтение", "математика", "окружающий мир", 
                  "английский язык", "французский язык", "информатика"],
            "3": ["русский язык", "литературное чтение", "математика", "окружающий мир", 
                  "английский язык", "французский язык", "информатика"],
            "4": ["русский язык", "литературное чтение", "математика", "окружающий мир", 
                  "английский язык", "французский язык", "информатика"],
            "5": ["математика", "география", "биология", "русский язык", "литература", 
                  "английский язык", "французский язык", "история", "информатика"],
            "6": ["математика", "география", "биология", "русский язык", "литература", 
                  "английский язык", "французский язык", "история", "обществознание", "информатика"],
            "7": ["алгебра", "геометрия", "физика", "география", "биология", "русский язык", 
                  "литература", "английский язык", "французский язык", "история", "обществознание", "информатика"],
            "8": ["алгебра", "геометрия", "физика", "география", "биология", "русский язык", 
                  "литература", "английский язык", "французский язык", "история", "обществознание", "информатика", "химия"],
            "9": ["алгебра", "геометрия", "физика", "география", "биология", "русский язык", 
                  "литература", "английский язык", "французский язык", "история", "обществознание", "информатика", "химия"],
            "10": ["алгебра", "геометрия", "физика", "география", "биология", "русский язык", 
                   "литература", "английский язык", "французский язык", "история", "обществознание", "информатика", "химия"],
            "11": ["алгебра", "геометрия", "физика", "география", "биология", "русский язык", 
                   "литература", "английский язык", "французский язык", "история", "обществознание", "информатика", "химия"]
        }
        
        for class_level, subjects in subjects_by_class.items():
            class_dir = LESSONS_STUDENTS_DIR / f"{class_level}_class"
            class_dir.mkdir(parents=True, exist_ok=True)
            
            for subject in subjects:
                subject_dir = class_dir / subject
                subject_dir.mkdir(parents=True, exist_ok=True)
                
                if not any(subject_dir.glob("*.txt")):
                    for i in range(1, 4):
                        lesson_number = f"{i:02d}"
                        sample_lesson = subject_dir / f"lesson_{lesson_number}_introduction.txt"
                        if not sample_lesson.exists():
                            with open(sample_lesson, 'w', encoding='utf-8') as f:
                                f.write(f"""# Урок {i}: Введение в {subject}
                                
Добро пожаловать на урок {i} по предмету {subject}!

Этот урок предназначен для учеников {class_level} класса.

На этом уроке вы:
1. Познакомитесь с основными понятиями предмета
2. Узнаете интересные факты
3. Научитесь применять знания на практике

Этот урок был создан автоматически для демонстрации работы системы.

Желаем успехов в обучении!
""")
    
    # =============================================================================
    # РАБОТА С УРОКАМИ
    # =============================================================================
    
    def get_available_lessons(self) -> Dict[str, Any]:
        """Получение доступных уроков"""
        try:
            lessons = {}
            
            lesson_dirs = [LESSONS_DEMO_DIR, LESSONS_STUDENTS_DIR, LESSONS_GENERATED_DIR, LESSONS_DIR]
            
            for lesson_dir in lesson_dirs:
                if not lesson_dir.exists():
                    continue
                    
                for lesson_file in lesson_dir.glob("*.txt"):
                    try:
                        subject = self._detect_subject(lesson_file.stem)
                        
                        if subject not in lessons:
                            lessons[subject] = []
                        
                        if lesson_dir == LESSONS_DEMO_DIR:
                            lesson_type = "demo"
                        elif lesson_dir == LESSONS_STUDENTS_DIR:
                            lesson_type = "student" 
                        elif lesson_dir == LESSONS_GENERATED_DIR:
                            lesson_type = "generated"
                        else:
                            lesson_type = "legacy"
                        
                        lessons[subject].append({
                            'id': lesson_file.stem,
                            'title': lesson_file.stem.replace('_', ' ').title(),
                            'file_path': lesson_file.name,
                            'type': lesson_type,
                            'full_path': str(lesson_file)
                        })
                    except Exception as e:
                        print(f"Ошибка загрузки урока {lesson_file}: {e}")
            
            return {
                "success": True,
                "lessons": lessons
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _detect_subject(self, filename: str) -> str:
        """Определение предмета по названию файла"""
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
    
    def get_lesson_content(self, lesson_id: str) -> Dict[str, Any]:
        """Получение содержания урока"""
        try:
            possible_paths = []
            
            # Ищем урок в структуре по классам
            for class_dir in LESSONS_STUDENTS_DIR.glob("*_class"):
                lesson_file = class_dir / f"{lesson_id}.txt"
                if lesson_file.exists():
                    possible_paths.append(lesson_file)
                
                for subject_dir in class_dir.iterdir():
                    if subject_dir.is_dir():
                        lesson_file = subject_dir / f"{lesson_id}.txt"
                        if lesson_file.exists():
                            possible_paths.append(lesson_file)
            
            # Ищем в демо и сгенерированных уроках
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
                return {"error": "Lesson not found"}
            
            with open(lesson_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            paragraphs = self._split_into_paragraphs(content)
            
            return {
                "success": True,
                "lesson_id": lesson_id,
                "content": paragraphs,
                "paragraph_count": len(paragraphs),
                "file_path": str(lesson_file.relative_to(LESSONS_DIR))
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _split_into_paragraphs(self, content: str) -> List[str]:
        """Разделение текста на абзацы"""
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
        return paragraphs
    
    # =============================================================================
    # ПРОГРЕСС УЧЕНИКОВ
    # =============================================================================
    
    def initialize_student_progress(self, student_id: str, education_level: str) -> bool:
        """Инициализирует прогресс ученика"""
        try:
            progress_file = STUDENT_PROGRESS_DIR / f"{student_id}.json"
            
            progress_data = {
                "student_id": student_id,
                "education_level": education_level,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "subjects": {}
            }
            
            class_dir = LESSONS_STUDENTS_DIR / f"{education_level}_class"
            if class_dir.exists():
                for subject_dir in class_dir.iterdir():
                    if subject_dir.is_dir():
                        subject_name = subject_dir.name
                        lesson_files = list(subject_dir.glob("*.txt"))
                        
                        progress_data["subjects"][subject_name] = {
                            "completed_lessons": [],
                            "current_lesson": None,
                            "total_lessons": len(lesson_files),
                            "last_accessed": None,
                            "progress_percent": 0
                        }
            
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Инициализирован прогресс для ученика {student_id} ({education_level} класс)")
            return True
        except Exception as e:
            print(f"❌ Ошибка инициализации прогресса: {e}")
            return False
    
    def update_student_lesson_progress(self, student_id: str, subject: str, 
                                       lesson_id: str, completed: bool = True) -> bool:
        """Обновляет прогресс ученика по уроку"""
        try:
            progress_file = STUDENT_PROGRESS_DIR / f"{student_id}.json"
            
            if not progress_file.exists():
                # Если файл прогресса не существует, создаем его
                # Предполагаем, что у нас есть доступ к данным ученика
                return False
            
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            
            if subject not in progress_data["subjects"]:
                progress_data["subjects"][subject] = {
                    "completed_lessons": [],
                    "current_lesson": lesson_id,
                    "total_lessons": 0,
                    "last_accessed": datetime.now().isoformat(),
                    "progress_percent": 0
                }
            
            subject_progress = progress_data["subjects"][subject]
            
            if completed and lesson_id not in subject_progress["completed_lessons"]:
                subject_progress["completed_lessons"].append(lesson_id)
                subject_progress["current_lesson"] = lesson_id
                subject_progress["last_accessed"] = datetime.now().isoformat()
                
                # Пересчитываем общее количество уроков
                lesson_count = 0
                class_dir = LESSONS_STUDENTS_DIR / f"{progress_data.get('education_level', '5')}_class"
                if class_dir.exists():
                    subject_dir = class_dir / subject
                    if subject_dir.exists():
                        lesson_count = len(list(subject_dir.glob("*.txt")))
                
                subject_progress["total_lessons"] = lesson_count
            
            try:
                with open(progress_file, 'w', encoding='utf-8') as f:
                    json.dump(progress_data, f, ensure_ascii=False, indent=2)
                print(f"✅ Прогресс сохранен: {lesson_id} по предмету {subject}")
            except Exception as e:
                print(f"❌ Ошибка сохранения прогресса: {e}")
            
            return True
        except Exception as e:
            print(f"❌ Ошибка обновления прогресса: {e}")
            return False
    
    def get_student_lessons_by_class(self, student_class: str) -> Dict[str, List[Dict[str, Any]]]:
        """Получает уроки для конкретного класса"""
        try:
            lessons_by_subject = {}
            class_dir = LESSONS_STUDENTS_DIR / f"{student_class}_class"
            
            if not class_dir.exists():
                return lessons_by_subject
            
            for subject_dir in class_dir.iterdir():
                if subject_dir.is_dir():
                    subject_name = subject_dir.name
                    lessons_by_subject[subject_name] = []
                    
                    for lesson_file in subject_dir.glob("*.txt"):
                        lesson_name = lesson_file.stem
                        lesson_number = 0
                        
                        match = re.search(r'lesson[_\s]*(\d+)', lesson_name.lower())
                        if match:
                            lesson_number = int(match.group(1))
                        
                        lesson_data = {
                            'id': f"{student_class}_class_{subject_name}_{lesson_file.stem}",
                            'title': lesson_file.stem.replace('_', ' ').title(),
                            'file_path': str(lesson_file.relative_to(LESSONS_DIR)),
                            'subject': subject_name,
                            'class_level': student_class,
                            'lesson_number': lesson_number,
                            'full_path': f"students/{student_class}_class/{subject_name}/{lesson_file.name}"
                        }
                        
                        lessons_by_subject[subject_name].append(lesson_data)
                    
                    lessons_by_subject[subject_name].sort(key=lambda x: x['lesson_number'])
            
            return lessons_by_subject
        except Exception as e:
            print(f"❌ Ошибка получения уроков по классу: {e}")
            return {}
    
    def get_student_next_lesson(self, student_id: str, subject: str) -> Optional[Dict[str, Any]]:
        """Получает следующий урок для ученика по предмету"""
        try:
            progress_file = STUDENT_PROGRESS_DIR / f"{student_id}.json"
            
            if not progress_file.exists():
                return None
            
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            
            student_class = progress_data.get("education_level", "5")
            subject_progress = progress_data.get("subjects", {}).get(subject, {})
            completed_lessons = subject_progress.get("completed_lessons", [])
            
            lessons_by_subject = self.get_student_lessons_by_class(student_class)
            subject_lessons = lessons_by_subject.get(subject, [])
            
            if not subject_lessons:
                return None
            
            for lesson in subject_lessons:
                if lesson['id'] not in completed_lessons:
                    return lesson
            
            return subject_lessons[0] if subject_lessons else None
        except Exception as e:
            print(f"❌ Ошибка получения следующего урока: {e}")
            return None
    
    def get_student_progress(self, student_id: str) -> Dict[str, Any]:
        """Получение прогресса ученика"""
        try:
            progress_file = STUDENT_PROGRESS_DIR / f"{student_id}.json"
            
            if not progress_file.exists():
                return {"success": False, "error": "Прогресс не найден"}
            
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            
            student_class = progress_data.get("education_level", "5")
            lessons_by_subject = self.get_student_lessons_by_class(student_class)
            
            result = {
                'student_id': student_id,
                'student_class': student_class,
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
            
            return {"success": True, "progress": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # =============================================================================
    # ТЕХНИЧЕСКИЕ ПРЕДМЕТЫ И ФОРМУЛЫ
    # =============================================================================
    
    def detect_technical_subject(self, subject: str) -> Dict[str, Any]:
        """Быстрая проверка - является ли предмет техническим"""
        try:
            if not subject:
                return {"is_technical": False, "is_science": False, "subject_type": "general"}
            
            subject_lower = subject.lower()
            
            # 🔥 БЫСТРАЯ ПРОВЕРКА ТЕХНИЧЕСКИХ ПРЕДМЕТОВ
            technical_subjects = {
                'математика': 'math',
                'алгебра': 'math', 
                'геометрия': 'math',
                'физика': 'science',
                'химия': 'science',
                'биология': 'science',
                'информатика': 'tech',
                'программирование': 'tech',
                'компьютерные науки': 'tech',
                'инженерия': 'tech',
                'технология': 'tech',
                'астрономия': 'science',
                'статистика': 'math'
            }
            
            # 🔥 ОПТИМИЗИРОВАННЫЙ ПОИСК
            for tech_subj, subj_type in technical_subjects.items():
                if tech_subj in subject_lower:
                    is_science = subj_type == 'science'
                    return {
                        "is_technical": True,
                        "is_science": is_science,
                        "subject_type": subj_type,
                        "subject": subject,
                        "requires_formulas": subj_type in ['math', 'science'],
                        "requires_diagrams": subj_type in ['math', 'science', 'tech']
                    }
            
            return {"is_technical": False, "is_science": False, "subject_type": "general"}
        except Exception as e:
            print(f"⚠️ Ошибка определения технического предмета: {e}")
            return {"is_technical": False, "is_science": False, "subject_type": "general"}
    
    def extract_formulas_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Извлекает формулы из текста"""
        try:
            if not text:
                return []
            
            formulas = []
            
            # 🔥 ИЩЕМ ФОРМУЛЫ В ФОРМАТЕ $$
            latex_formulas = re.findall(r'\$\$(.*?)\$\$', text, re.DOTALL)
            formulas.extend([{"type": "latex", "formula": f.strip()} for f in latex_formulas])
            
            # 🔥 ИЩЕМ ФОРМУЛЫ В ФОРМАТЕ \( \)
            inline_formulas = re.findall(r'\\\((.*?)\\\)', text, re.DOTALL)
            formulas.extend([{"type": "inline", "formula": f.strip()} for f in inline_formulas])
            
            # 🔥 ИЩЕМ ПРОСТЫЕ ФОРМУЛЫ С РАВЕНСТВОМ
            simple_formulas = re.findall(r'([A-Za-zα-ωΑ-Ω0-9\+\-\*/=^_]+?=[A-Za-zα-ωΑ-Ω0-9\+\-\*/=^_]+)', text)
            formulas.extend([{"type": "simple", "formula": f.strip()} for f in simple_formulas if len(f) < 50])
            
            return formulas
        except Exception as e:
            print(f"⚠️ Ошибка извлечения формул: {e}")
            return []
    
    # =============================================================================
    # ПРАКТИЧЕСКИЕ ЗАДАНИЯ
    # =============================================================================
    
    def get_practice_content(self, lesson_id: str) -> Dict[str, Any]:
        """Получение практических заданий для урока"""
        try:
            practice_file = PRACTICE_DIR / f"{lesson_id}.json"
            if not practice_file.exists():
                return {"error": "Практические задания не найдены", "success": False}
            
            with open(practice_file, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            return {
                "success": True,
                'lesson_id': lesson_id,
                'content': content,
                'question_count': len(content.get('questions', []))
            }
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def get_practice_files(self) -> Dict[str, Any]:
        """Получение списка файлов практических заданий"""
        try:
            practice_files = []
            for practice_file in PRACTICE_DIR.glob("*.json"):
                practice_files.append({
                    'filename': practice_file.name,
                    'size': practice_file.stat().st_size,
                    'modified': datetime.fromtimestamp(practice_file.stat().st_mtime).isoformat()
                })
            
            return {
                "success": True,
                "files": practice_files
            }
        except Exception as e:
            return {"error": str(e)}
    
    def upload_practice(self, file) -> Dict[str, Any]:
        """Загрузка файла практических заданий"""
        try:
            if file.filename == '':
                return {"success": False, "error": "No file selected"}
            
            if file and file.filename.endswith('.json'):
                from werkzeug.utils import secure_filename
                filename = secure_filename(file.filename)
                file.save(PRACTICE_DIR / filename)
                
                return {
                    "success": True,
                    "message": f"File {filename} uploaded successfully",
                    "filename": filename
                }
            else:
                return {"success": False, "error": "Invalid file type. Only JSON allowed"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def delete_practice(self, filename: str) -> Dict[str, Any]:
        """Удаление файла практических заданий"""
        try:
            practice_file = PRACTICE_DIR / filename
            if not practice_file.exists():
                return {"success": False, "error": "File not found"}
            
            practice_file.unlink()
            return {
                "success": True,
                "message": f"File {filename} deleted successfully"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # =============================================================================
    # УПРАВЛЕНИЕ УРОКАМИ (CRUD)
    # =============================================================================
    
    def add_lesson(self, subject: str, title: str, content: str, 
                   class_level: str = '5') -> Dict[str, Any]:
        """Добавление нового урока"""
        try:
            if not title or not content:
                return {"success": False, "error": "Название и содержание урока обязательны"}
            
            if class_level == 'demo':
                lesson_dir = LESSONS_DEMO_DIR
            else:
                class_dir = LESSONS_STUDENTS_DIR / f"{class_level}_class"
                class_dir.mkdir(parents=True, exist_ok=True)
                
                subject_dir = class_dir / subject
                subject_dir.mkdir(parents=True, exist_ok=True)
                lesson_dir = subject_dir
            
            filename = f"lesson_{title.lower().replace(' ', '_')}.txt"
            lesson_path = lesson_dir / filename
            
            with open(lesson_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                "success": True, 
                "filename": filename, 
                "subject": subject, 
                "title": title, 
                "class_level": class_level
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def add_demo_lesson(self, title: str, content: str, subject: str = 'общее') -> Dict[str, Any]:
        """Добавление демо-урока"""
        try:
            if not title or not content:
                return {"success": False, "error": "Название и содержание урока обязательны"}
            
            lesson_dir = LESSONS_DEMO_DIR
            
            existing_lessons = list(lesson_dir.glob("demo_*.txt"))
            demo_numbers = []
            for lesson in existing_lessons:
                match = re.search(r'demo_(\d+)', lesson.stem.lower())
                if match:
                    demo_numbers.append(int(match.group(1)))
            
            next_number = max(demo_numbers) + 1 if demo_numbers else 1
            
            title_slug = re.sub(r'[^\wа-яё\s-]+', '', title.lower()).strip()
            title_slug = re.sub(r'\s+', '_', title_slug)
            title_slug = title_slug[:50]
            
            filename = f"demo_{next_number:02d}_{subject}_{title_slug}.txt"
            lesson_path = lesson_dir / filename
            
            with open(lesson_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                "success": True,
                "filename": filename,
                "subject": subject,
                "title": title,
                "type": "demo",
                "lesson_number": next_number,
                "file_path": str(lesson_path.relative_to(LESSONS_DIR))
            }
        except Exception as e:
            print(f"❌ Ошибка при добавлении демо-урока: {e}")
            return {"success": False, "error": str(e)}
    
    def add_lesson_with_class(self, subject: str, title: str, content: str, 
                              class_level: str = '5') -> Dict[str, Any]:
        """Добавление урока с указанием класса"""
        try:
            if not title or not content:
                return {"success": False, "error": "Название и содержание урока обязательны"}
            
            if not subject or subject == "Выберите класс сначала":
                return {"success": False, "error": "Выберите предмет"}
            
            if class_level == 'demo':
                lesson_dir = LESSONS_DEMO_DIR
            elif class_level == 'generated':
                lesson_dir = LESSONS_GENERATED_DIR
            else:
                if class_level not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']:
                    return {"success": False, "error": "Неверный класс"}
                
                class_dir = LESSONS_STUDENTS_DIR / f"{class_level}_class"
                class_dir.mkdir(parents=True, exist_ok=True)
                
                subject_dir = class_dir / subject
                subject_dir.mkdir(parents=True, exist_ok=True)
                lesson_dir = subject_dir
            
            existing_lessons = list(lesson_dir.glob("lesson_*.txt"))
            lesson_numbers = []
            for lesson in existing_lessons:
                match = re.search(r'lesson[_\s]*(\d+)', lesson.stem.lower())
                if match:
                    lesson_numbers.append(int(match.group(1)))
            
            next_number = max(lesson_numbers) + 1 if lesson_numbers else 1
            
            title_slug = re.sub(r'[^\wа-яё\s-]+', '', title.lower()).strip()
            title_slug = re.sub(r'\s+', '_', title_slug)
            title_slug = title_slug[:50]
            
            filename = f"lesson_{next_number:02d}_{title_slug}.txt"
            lesson_path = lesson_dir / filename
            
            with open(lesson_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                "success": True,
                "filename": filename,
                "subject": subject,
                "title": title,
                "class_level": class_level,
                "lesson_number": next_number,
                "file_path": str(lesson_path.relative_to(LESSONS_DIR))
            }
        except Exception as e:
            print(f"❌ Ошибка при добавлении урока: {e}")
            return {"success": False, "error": str(e)}
    
    def get_next_lesson_number(self, class_level: str, subject: str) -> Dict[str, Any]:
        """Получение следующего номера урока"""
        try:
            if not class_level or not subject or subject == "Выберите класс сначала":
                return {"success": False, "error": "Укажите класс и предмет"}
            
            if class_level == 'demo':
                lesson_dir = LESSONS_DEMO_DIR
            elif class_level == 'generated':
                lesson_dir = LESSONS_GENERATED_DIR
            else:
                class_dir = LESSONS_STUDENTS_DIR / f"{class_level}_class"
                if not class_dir.exists():
                    return {"success": False, "error": f"Класс {class_level} не найден"}
                
                lesson_dir = class_dir / subject
                if not lesson_dir.exists():
                    return {
                        "success": True,
                        "class_level": class_level,
                        "subject": subject,
                        "next_number": 1,
                        "total_lessons": 0
                    }
            
            existing_lessons = list(lesson_dir.glob("lesson_*.txt"))
            lesson_numbers = []
            for lesson in existing_lessons:
                match = re.search(r'lesson[_\s]*(\d+)', lesson.stem.lower())
                if match:
                    lesson_numbers.append(int(match.group(1)))
            
            next_number = max(lesson_numbers) + 1 if lesson_numbers else 1
            
            return {
                "success": True,
                "class_level": class_level,
                "subject": subject,
                "next_number": next_number,
                "total_lessons": len(existing_lessons)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def edit_lesson(self, lesson_path: str, content: str) -> Dict[str, Any]:
        """Редактирование урока"""
        try:
            full_path = LESSONS_DIR / lesson_path
            
            if not full_path.exists():
                return {"success": False, "error": "Урок не найден"}
            
            import shutil
            backup_path = full_path.with_suffix('.txt.backup')
            shutil.copy2(full_path, backup_path)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                "success": True,
                "message": "Урок успешно сохранен",
                "backup_created": str(backup_path.name)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def delete_lesson(self, lesson_path: str) -> Dict[str, Any]:
        """Удаление урока"""
        try:
            full_path = LESSONS_DIR / lesson_path
            
            if not full_path.exists():
                return {"success": False, "error": "Урок не найден"}
            
            if full_path.name.startswith('demo_') or 'generated' in str(full_path):
                trash_dir = LESSONS_DIR / 'trash'
                trash_dir.mkdir(exist_ok=True)
                backup_path = trash_dir / full_path.name
                import shutil
                shutil.move(full_path, backup_path)
                
                return {
                    "success": True,
                    "message": "Урок перемещен в корзину",
                    "backup_path": str(backup_path)
                }
            else:
                full_path.unlink()
                
                return {
                    "success": True,
                    "message": "Урок удален"
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_lesson_for_edit(self, lesson_path: str) -> Dict[str, Any]:
        """Получение урока для редактирования"""
        try:
            full_path = LESSONS_DIR / lesson_path
            
            if not full_path.exists():
                return {"success": False, "error": "Урок не найден"}
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                "success": True,
                "lesson_path": lesson_path,
                "content": content,
                "size": len(content),
                "filename": full_path.name
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # =============================================================================
    # СТРУКТУРА УРОКОВ
    # =============================================================================
    
    def get_lessons_structure(self) -> Dict[str, Any]:
        """Получение структуры уроков"""
        try:
            structure = {
                'demo': [],
                'generated': [],
                'students': {}
            }
            
            # Демо уроки
            demo_lessons = []
            for lesson_file in LESSONS_DEMO_DIR.glob("*.txt"):
                demo_lessons.append({
                    'name': lesson_file.name,
                    'size': lesson_file.stat().st_size,
                    'modified': datetime.fromtimestamp(lesson_file.stat().st_mtime).isoformat(),
                    'path': str(lesson_file)
                })
            structure['demo'] = demo_lessons
            
            # Сгенерированные уроки
            generated_lessons = []
            for lesson_file in LESSONS_GENERATED_DIR.glob("*.txt"):
                generated_lessons.append({
                    'name': lesson_file.name,
                    'size': lesson_file.stat().st_size,
                    'modified': datetime.fromtimestamp(lesson_file.stat().st_mtime).isoformat(),
                    'path': str(lesson_file)
                })
            structure['generated'] = generated_lessons
            
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
                                    'path': str(lesson_file)
                                })
                            structure['students'][class_name][subject_name] = lesson_files
            
            return {
                "success": True,
                "structure": structure
            }
        except Exception as e:
            print(f"❌ Ошибка получения структуры уроков: {e}")
            return {"success": False, "error": str(e)}
    
    def create_sample_lessons(self) -> Dict[str, Any]:
        """Создание примерных уроков"""
        try:
            created_count = 0
            
            # Создаем демо уроки
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
            
            # Создаем уроки для классов
            for class_dir in LESSONS_STUDENTS_DIR.glob("*_class"):
                if class_dir.is_dir():
                    class_level = class_dir.name.replace("_class", "")
                    
                    for subject_dir in class_dir.iterdir():
                        if subject_dir.is_dir():
                            if not any(subject_dir.glob("*.txt")):
                                for i in range(1, 4):
                                    lesson_file = subject_dir / f"lesson_{i:02d}_введение.txt"
                                    if not lesson_file.exists():
                                        with open(lesson_file, 'w', encoding='utf-8') as f:
                                            f.write(f"""# Урок {i}: Введение в {subject_dir.name}
                                            
Это примерный урок {i} по предмету {subject_dir.name} для {class_level} класса.

Содержание урока:
1. Основные понятия
2. Примеры и упражнения
3. Практические задания

Этот урок был создан автоматически для демонстрации работы системы.
""")
                                        created_count += 1
            
            return {
                "success": True,
                "message": f"Создано {created_count} примерных уроков",
                "created_count": created_count,
                "demo_created": len(demo_subjects)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # =============================================================================
    # ТЕХНИЧЕСКИЕ УПРАЖНЕНИЯ
    # =============================================================================
    
    def get_sample_technical_exercises(self, subject: str) -> Dict[str, Any]:
        """Получает примерные технические упражнения"""
        try:
            sample_exercises = {
                'математика': [
                    {
                        "problem": "Реши уравнение: 2x + 5 = 15",
                        "question": "Найди значение x",
                        "answer": "x = 5",
                        "explanation": "2x + 5 = 15 → 2x = 10 → x = 5"
                    },
                    {
                        "problem": "Найди площадь прямоугольника со сторонами 4 см и 6 см",
                        "question": "Какова площадь?",
                        "answer": "24 см²",
                        "explanation": "S = a * b = 4 * 6 = 24 см²"
                    }
                ],
                'физика': [
                    {
                        "problem": "Рассчитай скорость, если путь 100 метров пройден за 20 секунд",
                        "question": "Какова скорость?",
                        "answer": "5 м/с",
                        "explanation": "v = s / t = 100 / 20 = 5 м/с"
                    },
                    {
                        "problem": "Сила тяжести действует на тело массой 10 кг",
                        "question": "Найди силу тяжести",
                        "answer": "98 Н",
                        "explanation": "F = m * g = 10 * 9.8 = 98 Н"
                    }
                ],
                'химия': [
                    {
                        "problem": "Сколько молекул воды содержится в 18 граммах воды?",
                        "question": "Рассчитай количество молекул",
                        "answer": "6.02 × 10²³ молекул",
                        "explanation": "M(H₂O) = 18 г/моль, n = 1 моль, N = 6.02 × 10²³"
                    },
                    {
                        "problem": "Напиши уравнение реакции горения метана",
                        "question": "Какое уравнение?",
                        "answer": "CH₄ + 2O₂ → CO₂ + 2H₂O",
                        "explanation": "Метан реагирует с кислородом с образованием углекислого газа и воды"
                    }
                ],
                'информатика': [
                    {
                        "problem": "Напиши алгоритм для нахождения максимального числа в массиве",
                        "question": "Опиши алгоритм",
                        "answer": "1. Инициализируй max = первый элемент\n2. Для каждого элемента сравни с max\n3. Если элемент > max, обнови max\n4. Верни max",
                        "explanation": "Линейный поиск максимального элемента"
                    },
                    {
                        "problem": "Переведи число 42 из десятичной в двоичную систему",
                        "question": "Какое двоичное представление?",
                        "answer": "101010",
                        "explanation": "42 = 32 + 8 + 2 = 2⁵ + 2³ + 2¹ = 101010₂"
                    }
                ]
            }
            
            exercises = sample_exercises.get(subject, [])
            
            return {
                "success": True,
                "subject": subject,
                "exercises": exercises,
                "count": len(exercises),
                "is_technical": True
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

# Глобальный экземпляр LessonManager
_lesson_manager_instance = None

def get_lesson_manager():
    """Получение глобального экземпляра LessonManager"""
    global _lesson_manager_instance
    if _lesson_manager_instance is None:
        _lesson_manager_instance = LessonManager()
    return _lesson_manager_instance
