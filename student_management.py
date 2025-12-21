# student_management.py
# Модуль для управления учениками, их прогрессом и уроками по классам
# Вынесен из app.py для улучшения читаемости и поддержки кода

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil
import zipfile
import tempfile
import os

class StudentManagement:
    """Класс для управления учениками, их прогрессом и уроками"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.lessons_dir = base_dir / "lessons"
        self.students_dir = base_dir / "students_data"
        self.users_dir = base_dir / "users_data"
        self.student_progress_dir = base_dir / "students_progress"
        
        # Создаем структуру папок для уроков
        self.lessons_demo_dir = self.lessons_dir / "demo"
        self.lessons_students_dir = self.lessons_dir / "students"
        self.lessons_generated_dir = self.lessons_dir / "generated"
        self.lessons_trash_dir = self.lessons_dir / "trash"
        
        # Создаем необходимые папки
        self._create_directories()
        
        # Структура предметов по классам
        self.subjects_by_class = {
            "1": ["русскии язык", "литературное чтение", "математика", "окружающии мир", 
                  "англиискии язык", "французскии язык", "информатика"],
            "2": ["русскии язык", "литературное чтение", "математика", "окружающии мир", 
                  "англиискии язык", "французскии язык", "информатика"],
            "3": ["русскии язык", "литературное чтение", "математика", "окружающии мир", 
                  "англиискии язык", "французскии язык", "информатика"],
            "4": ["русскии язык", "литературное чтение", "математика", "окружающии мир", 
                  "англиискии язык", "французскии язык", "информатика"],
            "5": ["математика", "география", "биология", "русскии язык", "литература", 
                  "англиискии язык", "французскии язык", "история", "информатика"],
            "6": ["математика", "география", "биология", "русскии язык", "литература", 
                  "англиискии язык", "французскии язык", "история", "обществознание", "информатика"],
            "7": ["алгебра", "геометрия", "физика", "география", "биология", "русскии язык", 
                  "литература", "англиискии язык", "французскии язык", "история", "обществознание", "информатика"],
            "8": ["алгебра", "геометрия", "физика", "география", "биология", "русскии язык", 
                  "литература", "англиискии язык", "французскии язык", "история", "обществознание", 
                  "информатика", "химия"],
            "9": ["алгебра", "геометрия", "физика", "география", "биология", "русскии язык", 
                  "литература", "англиискии язык", "французскии язык", "история", "обществознание", 
                  "информатика", "химия"],
            "10": ["алгебра", "геометрия", "физика", "география", "биология", "русскии язык", 
                   "литература", "англиискии язык", "французскии язык", "история", "обществознание", 
                   "информатика", "химия"],
            "11": ["алгебра", "геометрия", "физика", "география", "биология", "русскии язык", 
                   "литература", "англиискии язык", "французскии язык", "история", "обществознание", 
                   "информатика", "химия"]
        }
    
    def _create_directories(self):
        """Создает необходимые директории"""
        directories = [
            self.students_dir, self.users_dir, self.student_progress_dir,
            self.lessons_demo_dir, self.lessons_students_dir, 
            self.lessons_generated_dir, self.lessons_trash_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def create_lessons_structure(self) -> None:
        """Создает структуру папок для уроков по классам (1-11 классы)"""
        for class_level, subjects in self.subjects_by_class.items():
            class_dir = self.lessons_students_dir / f"{class_level}_class"
            class_dir.mkdir(parents=True, exist_ok=True)
            
            for subject in subjects:
                subject_dir = class_dir / subject
                subject_dir.mkdir(parents=True, exist_ok=True)
                
                # Создаем демо-уроки, если их нет
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
    # МЕТОДЫ ДЛЯ РАБОТЫ С ДАННЫМИ УЧЕНИКОВ
    # =============================================================================
    
    def load_user_data(self, user_id: str) -> Optional[Dict]:
        """Загрузка данных пользователя"""
        try:
            user_file = self.users_dir / f"{user_id}.json"
            if user_file.exists():
                with open(user_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"Error loading user data: {e}")
            return None
    
    def save_user_data(self, user_data: Dict) -> bool:
        """Сохранение данных пользователя"""
        try:
            user_id = user_data['user_id']
            user_file = self.users_dir / f"{user_id}.json"
            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Error saving user data: {e}")
            return False
    
    def authenticate_user(self, username: str, password: str, role: str) -> Optional[Dict]:
        """Аутентификация пользователя"""
        try:
            for user_file in self.users_dir.glob("*.json"):
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
    
    def create_new_student(self, username: str, password: str) -> Optional[Dict]:
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
            if self.save_user_data(user_data):
                return user_data
            return None
        except Exception as e:
            print(f"Error creating student: {e}")
            return None
    
    def create_new_teacher(self, username: str, password: str) -> Optional[Dict]:
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
            if self.save_user_data(user_data):
                return user_data
            return None
        except Exception as e:
            print(f"Error creating teacher: {e}")
            return None
    
    def update_student_profile(self, user_id: str, student_data: Dict) -> bool:
        """Обновление профиля ученика"""
        try:
            user_data = self.load_user_data(user_id)
            if not user_data:
                return False
            
            user_data['student_data'] = student_data
            user_data['profile_complete'] = True
            user_data['profile_updated'] = datetime.now().isoformat()
            
            student_id = student_data.get('student_id')
            if student_id:
                self.save_student_data(student_data)
            
            return self.save_user_data(user_data)
        except Exception as e:
            print(f"Error updating student profile: {e}")
            return False
    
    # =============================================================================
    # МЕТОДЫ ДЛЯ РАБОТЫ С ДАННЫМИ УЧЕНИКОВ (JSON ФАИЛЫ)
    # =============================================================================
    
    def save_student_data(self, student_data: Dict) -> Optional[str]:
        """Сохраняет данные ученика в JSON фаил"""
        try:
            student_id = student_data.get('student_id')
            if not student_id:
                student_id = str(uuid.uuid4())
                student_data['student_id'] = student_id
            
            if 'conference_id' not in student_data:
                conference_id = str(int(datetime.now().timestamp() * 1000))
                student_data['conference_id'] = conference_id
            
            student_data['last_updated'] = datetime.now().isoformat()
            
            filename = f"{student_id}.json"
            filepath = self.students_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(student_data, f, ensure_ascii=False, indent=2)
            
            return student_id
        except Exception as e:
            print(f"Error saving student data: {e}")
            return None
    
    def load_student_data(self, student_id: str) -> Optional[Dict]:
        """Загружает данные ученика из JSON фаил"""
        try:
            filename = f"{student_id}.json"
            filepath = self.students_dir / filename
            
            if not filepath.exists():
                return None
            
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading student data: {e}")
            return None
    
    def find_student_by_name(self, name: str) -> Optional[Dict]:
        """Находит ученика по имени"""
        try:
            for filepath in self.students_dir.glob("*.json"):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('name', '').lower() == name.lower():
                        return data
            return None
        except Exception as e:
            print(f"Error finding student: {e}")
            return None
    
    def update_student_data(self, student_id: str, updates: Dict) -> bool:
        """Обновляет данные ученика"""
        try:
            current_data = self.load_student_data(student_id)
            if not current_data:
                return False
            
            current_data.update(updates)
            current_data['last_updated'] = datetime.now().isoformat()
            
            return self.save_student_data(current_data) is not None
        except Exception as e:
            print(f"Error updating student data: {e}")
            return False
    
    # =============================================================================
    # МЕТОДЫ ДЛЯ УПРАВЛЕНИЯ ПРОГРЕССОМ УЧЕНИКОВ
    # =============================================================================
    
    def initialize_student_progress(self, student_id: str, education_level: str) -> bool:
        """Инициализирует прогресс ученика"""
        try:
            progress_file = self.student_progress_dir / f"{student_id}.json"
            
            progress_data = {
                "student_id": student_id,
                "education_level": education_level,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "subjects": {}
            }
            
            class_dir = self.lessons_students_dir / f"{education_level}_class"
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
            print(f"❌ Ошибка инициализации прогресс: {e}")
            return False
    
    def update_student_lesson_progress(self, student_id: str, subject: str, 
                                     lesson_id: str, completed: bool = True) -> bool:
        """Обновляет прогресс ученика по уроку"""
        try:
            progress_file = self.student_progress_dir / f"{student_id}.json"
            
            if not progress_file.exists():
                student_data = self.load_student_data(student_id)
                if student_data:
                    self.initialize_student_progress(student_id, student_data.get('education_level', '5'))
                else:
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
                
                # Обновляем общее количество уроков
                lesson_count = 0
                class_dir = self.lessons_students_dir / f"{progress_data.get('education_level', '5')}_class"
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
            print(f"❌ Ошибка обновления прогресс: {e}")
            return False
    
    def get_student_lessons_by_class(self, student_class: str) -> Dict[str, List[Dict]]:
        """Получает уроки для конкретного класса"""
        try:
            lessons_by_subject = {}
            class_dir = self.lessons_students_dir / f"{student_class}_class"
            
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
                            'file_path': str(lesson_file.relative_to(self.lessons_dir)),
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
    
    def get_student_next_lesson(self, student_id: str, subject: str) -> Optional[Dict]:
        """Получает следующии урок для ученика по предмету"""
        try:
            progress_file = self.student_progress_dir / f"{student_id}.json"
            
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
    
    def get_student_progress_dashboard(self, student_id: str, student_class: str, 
                                      student_name: str = "") -> Dict:
        """🔥 Получает прогресс ученика для личного кабинета"""
        try:
            progress_file = self.student_progress_dir / f"{student_id}.json"
            if not progress_file.exists():
                # Создаем начальныи прогресс
                self.initialize_student_progress(student_id, student_class)
                return self.get_student_progress_dashboard(student_id, student_class, student_name)
            
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            
            lessons_by_subject = self.get_student_lessons_by_class(student_class)
            
            result = {
                'student_id': student_id,
                'student_class': student_class,
                'student_name': student_name,
                'subjects': {}
            }
            
            for subject_name, lessons in lessons_by_subject.items():
                subject_progress = progress_data.get("subjects", {}).get(subject_name, {
                    "completed_lessons": [],
                    "current_lesson": None,
                    "total_lessons": len(lessons),
                    "last_accessed": None,
                    "progress_percent": 0
                })
                
                completed_count = len(subject_progress.get("completed_lessons", []))
                total_lessons = len(lessons)
                
                # Ищем следующии урок
                next_lesson = None
                for lesson in lessons:
                    if lesson['id'] not in subject_progress.get("completed_lessons", []):
                        next_lesson = lesson
                        break
                
                result['subjects'][subject_name] = {
                    'total_lessons': total_lessons,
                    'completed_lessons': completed_count,
                    'progress_percent': int((completed_count / total_lessons) * 100) if total_lessons > 0 else 0,
                    'last_updated': subject_progress.get("last_accessed"),
                    'current_lesson': subject_progress.get("current_lesson"),
                    'next_lesson': next_lesson,
                    'has_lessons': total_lessons > 0
                }
            
            # Общая статистика
            total_completed = sum(subj['completed_lessons'] for subj in result['subjects'].values())
            total_lessons = sum(subj['total_lessons'] for subj in result['subjects'].values())
            
            result['overall'] = {
                'total_lessons': total_lessons,
                'completed_lessons': total_completed,
                'progress_percent': int((total_completed / total_lessons) * 100) if total_lessons > 0 else 0,
                'subjects_count': len(result['subjects'])
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Ошибка получения прогресса для дашборда: {e}")
            return {
                'student_id': student_id,
                'student_class': student_class,
                'student_name': student_name,
                'subjects': {},
                'overall': {'total_lessons': 0, 'completed_lessons': 0, 'progress_percent': 0, 'subjects_count': 0}
            }
    
    # =============================================================================
    # МЕТОДЫ ДЛЯ ЭКСПОРТА И ИМПОРТА ДАННЫХ
    # =============================================================================
    
    def export_students_full(self) -> Optional[Path]:
        """Полныи экспорт данных учеников (включая прогресс)"""
        try:
            temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            
            with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
                # 1. Экспорт пользователеи
                for user_file in self.users_dir.glob("*.json"):
                    with open(user_file, 'r', encoding='utf-8') as f:
                        user_data = json.load(f)
                        if user_data.get('role') == 'student':
                            zipf.write(user_file, f"users/{user_file.name}")
                
                # 2. Экспорт данных учеников
                for student_file in self.students_dir.glob("*.json"):
                    zipf.write(student_file, f"students/{student_file.name}")
                
                # 3. Экспорт прогресса
                for progress_file in self.student_progress_dir.glob("*.json"):
                    zipf.write(progress_file, f"progress/{progress_file.name}")
                
                # 4. Создаем фаил метаданных
                metadata = {
                    "export_date": datetime.now().isoformat(),
                    "total_users": len(list(self.users_dir.glob("*.json"))),
                    "total_students": len(list(self.students_dir.glob("*.json"))),
                    "total_progress": len(list(self.student_progress_dir.glob("*.json"))),
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
            return Path(temp_zip.name)
            
        except Exception as e:
            print(f"❌ Ошибка экспорта данных: {e}")
            return None
    
    def import_students_data(self, zip_file_path: Path) -> Dict:
        """Импорт данных учеников из ZIP-фаила"""
        try:
            import zipfile
            
            results = {
                "success": True,
                "imported": {
                    "users": 0,
                    "students": 0,
                    "progress": 0
                },
                "errors": []
            }
            
            # Создаем временную папку для распаковки
            temp_dir = tempfile.mkdtemp()
            
            # Распаковываем архив
            with zipfile.ZipFile(zip_file_path, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            # Импортируем пользователеи
            users_dir = os.path.join(temp_dir, "users")
            if os.path.exists(users_dir):
                for user_file in os.listdir(users_dir):
                    if user_file.endswith('.json'):
                        try:
                            src_path = os.path.join(users_dir, user_file)
                            dst_path = self.users_dir / user_file
                            
                            # Проверяем, существует ли уже такои пользователь
                            if dst_path.exists():
                                # Создаем резервную копию
                                backup_path = self.users_dir / f"{user_file}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                shutil.copy2(dst_path, backup_path)
                            
                            shutil.copy2(src_path, dst_path)
                            results["imported"]["users"] += 1
                        except Exception as e:
                            results["errors"].append(f"Ошибка импорта пользователя {user_file}: {str(e)}")
            
            # Импортируем данные учеников
            students_dir = os.path.join(temp_dir, "students")
            if os.path.exists(students_dir):
                for student_file in os.listdir(students_dir):
                    if student_file.endswith('.json'):
                        try:
                            src_path = os.path.join(students_dir, student_file)
                            dst_path = self.students_dir / student_file
                            
                            if dst_path.exists():
                                backup_path = self.students_dir / f"{student_file}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                shutil.copy2(dst_path, backup_path)
                            
                            shutil.copy2(src_path, dst_path)
                            results["imported"]["students"] += 1
                        except Exception as e:
                            results["errors"].append(f"Ошибка импорта данных ученика {student_file}: {str(e)}")
            
            # Импортируем прогресс
            progress_dir = os.path.join(temp_dir, "progress")
            if os.path.exists(progress_dir):
                for progress_file in os.listdir(progress_dir):
                    if progress_file.endswith('.json'):
                        try:
                            src_path = os.path.join(progress_dir, progress_file)
                            dst_path = self.student_progress_dir / progress_file
                            
                            if dst_path.exists():
                                backup_path = self.student_progress_dir / f"{progress_file}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                shutil.copy2(dst_path, backup_path)
                            
                            shutil.copy2(src_path, dst_path)
                            results["imported"]["progress"] += 1
                        except Exception as e:
                            results["errors"].append(f"Ошибка импорта прогресса {progress_file}: {str(e)}")
            
            # Очищаем временные фаилы
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            return results
            
        except Exception as e:
            print(f"❌ Ошибка импорта данных: {e}")
            return {
                "success": False,
                "error": str(e),
                "imported": {"users": 0, "students": 0, "progress": 0},
                "errors": [str(e)]
            }
    
    def export_students_progress(self) -> Optional[Path]:
        """Экспорт только прогресса учеников"""
        try:
            temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            
            with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
                for progress_file in self.student_progress_dir.glob("*.json"):
                    zipf.write(progress_file, progress_file.name)
            
            temp_zip.close()
            return Path(temp_zip.name)
            
        except Exception as e:
            print(f"❌ Ошибка экспорта прогресса: {e}")
            return None
    
    # =============================================================================
    # МЕТОДЫ ДЛЯ РАБОТЫ С УРОКАМИ
    # =============================================================================
    
    def get_lessons_structure(self) -> Dict:
        """Получает структуру всех уроков"""
        try:
            structure = {
                'demo': [],
                'generated': [],
                'students': {}
            }
            
            # Демо уроки
            demo_lessons = []
            for lesson_file in self.lessons_demo_dir.glob("*.txt"):
                demo_lessons.append({
                    'name': lesson_file.name,
                    'size': lesson_file.stat().st_size,
                    'modified': datetime.fromtimestamp(lesson_file.stat().st_mtime).isoformat(),
                    'path': str(lesson_file)
                })
            structure['demo'] = demo_lessons
            
            # Сгенерированные уроки
            generated_lessons = []
            for lesson_file in self.lessons_generated_dir.glob("*.txt"):
                generated_lessons.append({
                    'name': lesson_file.name,
                    'size': lesson_file.stat().st_size,
                    'modified': datetime.fromtimestamp(lesson_file.stat().st_mtime).isoformat(),
                    'path': str(lesson_file)
                })
            structure['generated'] = generated_lessons
            
            # Уроки по классам
            for class_dir in sorted(self.lessons_students_dir.glob("*_class")):
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
            
            return structure
        except Exception as e:
            print(f"❌ Ошибка получения структуры уроков: {e}")
            return {}
    
    def get_all_lessons_list(self) -> List[Dict]:
        """Получает список всех уроков"""
        try:
            lessons = []
            
            # Демо уроки
            for lesson_file in self.lessons_demo_dir.glob("*.txt"):
                lessons.append({
                    'type': 'demo',
                    'class': 'demo',
                    'subject': 'demo',
                    'name': lesson_file.name,
                    'full_path': str(lesson_file),
                    'size': lesson_file.stat().st_size,
                    'modified': datetime.fromtimestamp(lesson_file.stat().st_mtime).isoformat(),
                    'can_edit': True,
                    'can_delete': True
                })
            
            # Сгенерированные уроки
            for lesson_file in self.lessons_generated_dir.glob("*.txt"):
                lessons.append({
                    'type': 'generated',
                    'class': 'generated',
                    'subject': 'auto',
                    'name': lesson_file.name,
                    'full_path': str(lesson_file),
                    'size': lesson_file.stat().st_size,
                    'modified': datetime.fromtimestamp(lesson_file.stat().st_mtime).isoformat(),
                    'can_edit': True,
                    'can_delete': True
                })
            
            # Уроки по классам
            for class_dir in self.lessons_students_dir.glob("*_class"):
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
                                    'full_path': str(lesson_file),
                                    'size': lesson_file.stat().st_size,
                                    'modified': datetime.fromtimestamp(lesson_file.stat().st_mtime).isoformat(),
                                    'can_edit': True,
                                    'can_delete': True
                                })
            
            lessons.sort(key=lambda x: x['modified'], reverse=True)
            return lessons
        except Exception as e:
            print(f"❌ Ошибка получения списка уроков: {e}")
            return []
    
    def add_lesson_with_class(self, subject: str, title: str, content: str, 
                             class_level: str = '5') -> Dict:
        """Добавляет урок с указанием класса"""
        try:
            if not title or not content:
                return {"success": False, "error": "Название и содержание урока обязательны"}
            
            if not subject or subject == "Выберите класс сначала":
                return {"success": False, "error": "Выберите предмет"}
            
            if class_level == 'demo':
                lesson_dir = self.lessons_demo_dir
            elif class_level == 'generated':
                lesson_dir = self.lessons_generated_dir
            else:
                if class_level not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']:
                    return {"success": False, "error": "Неверныи класс"}
                
                class_dir = self.lessons_students_dir / f"{class_level}_class"
                class_dir.mkdir(parents=True, exist_ok=True)
                
                subject_dir = class_dir / subject
                subject_dir.mkdir(parents=True, exist_ok=True)
                lesson_dir = subject_dir
            
            # Определяем номер урока
            existing_lessons = list(lesson_dir.glob("lesson_*.txt"))
            lesson_numbers = []
            for lesson in existing_lessons:
                match = re.search(r'lesson[_\s]*(\d+)', lesson.stem.lower())
                if match:
                    lesson_numbers.append(int(match.group(1)))
            
            next_number = max(lesson_numbers) + 1 if lesson_numbers else 1
            
            # Создаем имя фаила
            title_slug = re.sub(r'[^\wа-яе\s-]+', '', title.lower()).strip()
            title_slug = re.sub(r'\s+', '_', title_slug)
            title_slug = title_slug[:50]
            
            filename = f"lesson_{next_number:02d}_{title_slug}.txt"
            lesson_path = lesson_dir / filename
            
            # Сохраняем урок
            with open(lesson_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                "success": True,
                "filename": filename,
                "subject": subject,
                "title": title,
                "class_level": class_level,
                "lesson_number": next_number,
                "file_path": str(lesson_path.relative_to(self.lessons_dir))
            }
        except Exception as e:
            print(f"❌ Ошибка при добавлении урока: {e}")
            return {"success": False, "error": str(e)}
    
    def get_next_lesson_number(self, class_level: str, subject: str) -> Dict:
        """Получает следующии номер урока для указанного класса и предмета"""
        try:
            if not class_level or not subject or subject == "Выберите класс сначала":
                return {"success": False, "error": "Укажите класс и предмет"}
            
            if class_level == 'demo':
                lesson_dir = self.lessons_demo_dir
            elif class_level == 'generated':
                lesson_dir = self.lessons_generated_dir
            else:
                class_dir = self.lessons_students_dir / f"{class_level}_class"
                if not class_dir.exists():
                    return {"success": False, "error": f"Класс {class_level} не наиден"}
                
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
            print(f"❌ Ошибка получения номера урока: {e}")
            return {"success": False, "error": str(e)}
