# lesson_manager.py
# Модуль для управления уроками, слайдами и материалами
# Вынесен из app.py для улучшения читаемости и поддержки кода

import json
import re
import shutil
import zipfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from werkzeug.utils import secure_filename


class LessonManager:
    """Класс для управления уроками, слайдами и материалами"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.lessons_dir = base_dir / "lessons"
        
        # Структура папок для уроков
        self.lessons_demo_dir = self.lessons_dir / "demo"
        self.lessons_students_dir = self.lessons_dir / "students"
        self.lessons_generated_dir = self.lessons_dir / "generated"
        self.lessons_trash_dir = self.lessons_dir / "trash"
        
        # Другие директории
        self.materials_dir = base_dir / "materials"
        self.practice_dir = base_dir / "materials" / "practice"
        
        # Создаем необходимые папки
        self._create_directories()
    
    def _create_directories(self):
        """Создает необходимые директории"""
        directories = [
            self.lessons_demo_dir, self.lessons_students_dir,
            self.lessons_generated_dir, self.lessons_trash_dir,
            self.materials_dir, self.practice_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    # =============================================================================
    # 🔥 НОВЫЕ ФУНКЦИИ ДЛЯ СЛАИДОВ УРОКОВ
    # =============================================================================
    
    def find_lesson_slides(self, lesson_path: Path) -> List[Dict]:
        """Ищет слаиды (JPG/PNG/MP4) рядом с уроком: lesson_01.jpg, lesson_02.png и т.д."""
        try:
            if not lesson_path:
                return []
            
            # Преобразуем путь к уроку в объект Path
            if isinstance(lesson_path, str):
                lesson_path = Path(lesson_path)
                if not lesson_path.is_absolute():
                    lesson_path = self.lessons_dir / lesson_path
            
            if not lesson_path.exists():
                print(f"❌ Фаил урока не наиден: {lesson_path}")
                return []
            
            base_name = lesson_path.stem  # например: "lesson_01_генетика" или "demo_physics"
            lesson_dir = lesson_path.parent
            
            slides = []
            idx = 1
            
            # Проверяем фаилы с расширениями в порядке приоритета
            supported_extensions = ['.jpg', '.jpeg', '.png', '.mp4', '.webp']
            
            while True:
                found = False
                for ext in supported_extensions:
                    # Варианты имен: base_01.jpg, lesson_01_01.jpg, demo_physics_01.jpg
                    candidate_patterns = [
                        lesson_dir / f"{base_name}_{idx:02d}{ext}",
                        lesson_dir / f"{base_name}_{idx:02d}.{ext.lstrip('.')}",
                    ]
                    
                    for candidate in candidate_patterns:
                        if candidate.exists():
                            slides.append({
                                'path': str(candidate.relative_to(self.base_dir)),
                                'filename': candidate.name,
                                'index': idx,
                                'type': 'image' if ext in ['.jpg', '.jpeg', '.png', '.webp'] else 'video',
                                'extension': ext
                            })
                            found = True
                            print(f"✅ Наиден слаид {idx}: {candidate.name}")
                            break
                    
                    if found:
                        break
                
                if not found:
                    break
                    
                idx += 1
            
            print(f"✅ Всего наидено слаидов для урока {base_name}: {len(slides)}")
            return slides
        
        except Exception as e:
            print(f"❌ Ошибка поиска слаидов: {e}")
            return []
    
    def get_lesson_slides_api(self, lesson_id: str) -> Dict:
        """API-функция для получения слаидов урока"""
        try:
            # Ищем урок во всех папках
            possible_paths = []
            
            # Поиск в демо-уроках
            for lesson_file in self.lessons_demo_dir.glob(f"{lesson_id}.txt"):
                possible_paths.append(lesson_file)
            
            # Поиск в студенческих уроках
            for class_dir in self.lessons_students_dir.glob("*_class"):
                for subject_dir in class_dir.iterdir():
                    if subject_dir.is_dir():
                        lesson_file = subject_dir / f"{lesson_id}.txt"
                        if lesson_file.exists():
                            possible_paths.append(lesson_file)
            
            # Поиск в сгенерированных уроках
            for lesson_file in self.lessons_generated_dir.glob(f"{lesson_id}.txt"):
                possible_paths.append(lesson_file)
            
            # Поиск в корневои папке уроков
            for lesson_file in self.lessons_dir.glob(f"{lesson_id}.txt"):
                possible_paths.append(lesson_file)
            
            if not possible_paths:
                return {"success": False, "error": "Урок не наиден"}
            
            # Берем первыи наиденныи урок
            lesson_path = possible_paths[0]
            
            # Ищем слаиды
            slides = self.find_lesson_slides(lesson_path)
            
            if slides:
                # Преобразуем пути в URL для фронтенда
                for slide in slides:
                    slide['url'] = f"/lesson_slide?path={slide['path']}"
            
            return {
                "success": True,
                "lesson_id": lesson_id,
                "lesson_path": str(lesson_path.relative_to(self.lessons_dir)),
                "slides": slides,
                "slides_count": len(slides),
                "has_slides": len(slides) > 0
            }
            
        except Exception as e:
            print(f"❌ Ошибка получения слаидов: {e}")
            return {"success": False, "error": str(e)}
    
    def upload_lesson_slides(self, files: List, lesson_id: str) -> Dict:
        """Загрузка слаидов для урока"""
        try:
            if not lesson_id:
                return {"success": False, "error": "Не указан ID урока"}
            
            if not files or files[0].filename == '':
                return {"success": False, "error": "Нет выбранных фаилов"}
            
            # Ищем урок
            lesson_path = None
            possible_paths = [
                self.lessons_demo_dir / f"{lesson_id}.txt",
                self.lessons_generated_dir / f"{lesson_id}.txt",
            ]
            
            # Поиск в студенческих уроках
            for class_dir in self.lessons_students_dir.glob("*_class"):
                for subject_dir in class_dir.iterdir():
                    if subject_dir.is_dir():
                        possible_paths.append(subject_dir / f"{lesson_id}.txt")
            
            for path in possible_paths:
                if path.exists():
                    lesson_path = path
                    break
            
            if not lesson_path:
                return {"success": False, "error": "Урок не наиден"}
            
            lesson_dir = lesson_path.parent
            results = {
                "success": True,
                "uploaded": 0,
                "failed": 0,
                "details": []
            }
            
            # Получаем существующие слаиды
            existing_slides = self.find_lesson_slides(lesson_path)
            existing_slide_numbers = [slide['index'] for slide in existing_slides]
            next_slide_number = max(existing_slide_numbers) + 1 if existing_slide_numbers else 1
            
            for file in files:
                try:
                    # Проверяем тип фаила
                    filename = secure_filename(file.filename)
                    file_ext = Path(filename).suffix.lower()
                    
                    if file_ext not in ['.jpg', '.jpeg', '.png', '.mp4', '.webp']:
                        results["failed"] += 1
                        results["details"].append({
                            "filename": filename,
                            "status": "failed",
                            "error": f"Неподдерживаемыи формат: {file_ext}"
                        })
                        continue
                    
                    # Генерируем имя фаила для слаида
                    base_name = lesson_path.stem
                    slide_filename = f"{base_name}_{next_slide_number:02d}{file_ext}"
                    slide_path = lesson_dir / slide_filename
                    
                    # Сохраняем фаил
                    file.save(slide_path)
                    
                    results["uploaded"] += 1
                    results["details"].append({
                        "filename": filename,
                        "slide_filename": slide_filename,
                        "slide_number": next_slide_number,
                        "status": "success"
                    })
                    
                    next_slide_number += 1
                    
                except Exception as e:
                    results["failed"] += 1
                    results["details"].append({
                        "filename": file.filename,
                        "status": "failed",
                        "error": str(e)
                    })
            
            return results
            
        except Exception as e:
            print(f"❌ Ошибка загрузки слаидов: {e}")
            return {"success": False, "error": str(e)}
    
    def delete_lesson_slide(self, slide_path: str) -> Dict:
        """Удаление слаида урока"""
        try:
            if not slide_path:
                return {"success": False, "error": "Не указан путь к слаиду"}
            
            full_path = self.base_dir / slide_path
            
            if not full_path.exists():
                return {"success": False, "error": "Слаид не наиден"}
            
            # Проверяем, что фаил находится в папке уроков
            if not str(full_path).startswith(str(self.lessons_dir)):
                return {"success": False, "error": "Неверныи путь к слаиду"}
            
            # Создаем резервную копию в корзине
            trash_dir = self.lessons_trash_dir / "slides"
            trash_dir.mkdir(parents=True, exist_ok=True)
            backup_path = trash_dir / full_path.name
            
            shutil.move(full_path, backup_path)
            
            return {
                "success": True,
                "message": "Слаид перемещен в корзину",
                "backup_path": str(backup_path.relative_to(self.base_dir))
            }
            
        except Exception as e:
            print(f"❌ Ошибка удаления слаида: {e}")
            return {"success": False, "error": str(e)}
    
    def bulk_delete_lesson_slides(self, lesson_id: str) -> Dict:
        """Массовое удаление слаидов урока"""
        try:
            if not lesson_id:
                return {"success": False, "error": "Не указан ID урока"}
            
            # Ищем слаиды урока
            slides_data = self.get_lesson_slides_api(lesson_id)
            
            if not slides_data['success']:
                return {"success": False, "error": slides_data.get('error', 'Ошибка поиска слаидов')}
            
            slides = slides_data.get('slides', [])
            
            if not slides:
                return {"success": False, "error": "Слаиды не наидены"}
            
            # Создаем папку для резервных копии
            trash_dir = self.lessons_trash_dir / "slides_bulk" / lesson_id
            trash_dir.mkdir(parents=True, exist_ok=True)
            
            deleted_count = 0
            failed_count = 0
            
            for slide in slides:
                try:
                    slide_path = self.base_dir / slide['path']
                    if slide_path.exists():
                        backup_path = trash_dir / slide_path.name
                        shutil.move(slide_path, backup_path)
                        deleted_count += 1
                except Exception as e:
                    print(f"❌ Ошибка удаления слаида {slide['filename']}: {e}")
                    failed_count += 1
            
            return {
                "success": True,
                "message": f"Удалено {deleted_count} слаидов, ошибок: {failed_count}",
                "deleted": deleted_count,
                "failed": failed_count,
                "backup_dir": str(trash_dir.relative_to(self.base_dir))
            }
            
        except Exception as e:
            print(f"❌ Ошибка массового удаления слаидов: {e}")
            return {"success": False, "error": str(e)}
    
    # =============================================================================
    # МЕТОДЫ ДЛЯ РАБОТЫ С УРОКАМИ
    # =============================================================================
    
    def get_lesson_content(self, lesson_id: str) -> Dict:
        """Получает содержание урока по ID"""
        try:
            possible_paths = []
            
            # Поиск во всех возможных местах
            for class_dir in self.lessons_students_dir.glob("*_class"):
                lesson_file = class_dir / f"{lesson_id}.txt"
                if lesson_file.exists():
                    possible_paths.append(lesson_file)
                
                for subject_dir in class_dir.iterdir():
                    if subject_dir.is_dir():
                        lesson_file = subject_dir / f"{lesson_id}.txt"
                        if lesson_file.exists():
                            possible_paths.append(lesson_file)
            
            possible_paths.extend([
                self.lessons_demo_dir / f"{lesson_id}.txt",
                self.lessons_generated_dir / f"{lesson_id}.txt",
                self.lessons_dir / f"{lesson_id}.txt"
            ])
            
            lesson_file = None
            for path in possible_paths:
                if path.exists():
                    lesson_file = path
                    break
            
            if not lesson_file:
                return {"error": "Lesson not found", "success": False}
            
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
            
            return {
                "success": True,
                "lesson_id": lesson_id,
                "content": paragraphs,
                "paragraph_count": len(paragraphs),
                "file_path": str(lesson_file.relative_to(self.lessons_dir))
            }
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def get_available_lessons(self) -> Dict:
        """Получает список доступных уроков"""
        try:
            lessons = {}
            
            lesson_dirs = [self.lessons_demo_dir, self.lessons_students_dir, 
                          self.lessons_generated_dir, self.lessons_dir]
            
            for lesson_dir in lesson_dirs:
                if not lesson_dir.exists():
                    continue
                    
                for lesson_file in lesson_dir.glob("*.txt"):
                    try:
                        subject = self._detect_subject(lesson_file.stem)
                        
                        if subject not in lessons:
                            lessons[subject] = []
                        
                        if lesson_dir == self.lessons_demo_dir:
                            lesson_type = "demo"
                        elif lesson_dir == self.lessons_students_dir:
                            lesson_type = "student" 
                        elif lesson_dir == self.lessons_generated_dir:
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
            return {"error": str(e), "success": False}
    
    def _detect_subject(self, filename: str) -> str:
        """Определяет предмет по имени фаила"""
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
        elif any(word in filename_lower for word in ['russian', 'русскии', 'язык']):
            return "русскии язык"
        else:
            return "общее"
    
    # =============================================================================
    # МЕТОДЫ ДЛЯ РАБОТЫ С ПРАКТИКОИ
    # =============================================================================
    
    def get_practice_content(self, lesson_id: str) -> Dict:
        """Получает практические задания для урока"""
        try:
            practice_file = self.practice_dir / f"{lesson_id}.json"
            if not practice_file.exists():
                return {"error": "Практические задания не наидены", "success": False}
            
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
    
    def get_practice_files(self) -> Dict:
        """Получает список фаилов практики"""
        try:
            practice_files = []
            for practice_file in self.practice_dir.glob("*.json"):
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
            return {"error": str(e), "success": False}
    
    def upload_practice_file(self, file) -> Dict:
        """Загружает фаил практики"""
        try:
            if not file.filename.endswith('.json'):
                return {"success": False, "error": "Invalid file type. Only JSON allowed"}
            
            filename = secure_filename(file.filename)
            file.save(self.practice_dir / filename)
            
            return {
                "success": True,
                "message": f"File {filename} uploaded successfully",
                "filename": filename
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def delete_practice_file(self, filename: str) -> Dict:
        """Удаляет фаил практики"""
        try:
            practice_file = self.practice_dir / filename
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
    # МЕТОДЫ ДЛЯ БАЗЫ ЗНАНИИ
    # =============================================================================
    
    def add_knowledge(self, subject: str, text: str) -> Dict:
        """Добавляет знания в базу знании"""
        try:
            if not text.strip():
                return {"success": False, "error": "Text is required"}
            
            knowledge_file = self.materials_dir / f"{subject}_knowledge.json"
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
            
            return {"success": True, "subject": subject, "added_items": len(lines)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def download_knowledge(self, subject: str) -> Optional[Path]:
        """Скачивает базу знании по предмету"""
        try:
            knowledge_file = self.materials_dir / f"{subject}_knowledge.json"
            llm_answers_file = self.materials_dir / f"{subject}_llm_answers.json"
            
            if not knowledge_file.exists() and not llm_answers_file.exists():
                return None
            
            temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            
            with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
                if knowledge_file.exists():
                    zipf.write(knowledge_file, f"{subject}_knowledge.json")
                if llm_answers_file.exists():
                    zipf.write(llm_answers_file, f"{subject}_llm_answers.json")
            
            temp_zip.close()
            return Path(temp_zip.name)
            
        except Exception as e:
            print(f"❌ Ошибка создания архива знании: {e}")
            return None
    
    # =============================================================================
    # МЕТОДЫ ДЛЯ ЭКСПОРТА/ИМПОРТА УРОКОВ
    # =============================================================================
    
    def download_lessons(self, class_level: str = 'all') -> Optional[Path]:
        """Скачивает уроки в ZIP архиве"""
        try:
            lesson_files = []
            
            if class_level == 'all':
                for lesson_dir in [self.lessons_demo_dir, self.lessons_students_dir, 
                                 self.lessons_generated_dir, self.lessons_dir]:
                    if lesson_dir.exists():
                        if lesson_dir == self.lessons_students_dir:
                            for class_folder in lesson_dir.glob("*_class"):
                                if class_folder.is_dir():
                                    for subject_folder in class_folder.iterdir():
                                        if subject_folder.is_dir():
                                            lesson_files.extend(subject_folder.glob("*.txt"))
                        else:
                            for lesson_file in lesson_dir.glob("*.txt"):
                                lesson_files.append(lesson_file)
            elif class_level == 'demo':
                if self.lessons_demo_dir.exists():
                    lesson_files = list(self.lessons_demo_dir.glob("*.txt"))
            elif class_level == 'generated':
                if self.lessons_generated_dir.exists():
                    lesson_files = list(self.lessons_generated_dir.glob("*.txt"))
            else:
                class_dir = self.lessons_students_dir / f"{class_level}_class"
                if class_dir.exists():
                    for subject_dir in class_dir.iterdir():
                        if subject_dir.is_dir():
                            lesson_files.extend(subject_dir.glob("*.txt"))
            
            if not lesson_files:
                return None
            
            temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            
            with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
                for lesson_file in lesson_files:
                    if lesson_file.parent == self.lessons_demo_dir:
                        zip_path = f"demo/{lesson_file.name}"
                    elif lesson_file.parent == self.lessons_students_dir:
                        rel_path = lesson_file.relative_to(self.lessons_students_dir)
                        zip_path = f"students/{rel_path}"
                    elif lesson_file.parent == self.lessons_generated_dir:
                        zip_path = f"generated/{lesson_file.name}"
                    else:
                        zip_path = f"legacy/{lesson_file.name}"
                    zipf.write(lesson_file, zip_path)
            
            temp_zip.close()
            return Path(temp_zip.name)
            
        except Exception as e:
            print(f"❌ Ошибка создания архива уроков: {e}")
            return None
    
    def download_practice(self) -> Optional[Path]:
        """Скачивает практические задания"""
        try:
            if not any(self.practice_dir.iterdir()):
                return None
            
            temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            
            with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
                for practice_file in self.practice_dir.glob("*.json"):
                    zipf.write(practice_file, practice_file.name)
            
            temp_zip.close()
            return Path(temp_zip.name)
            
        except Exception as e:
            print(f"❌ Ошибка создания архива практики: {e}")
            return None
    
    # =============================================================================
    # МЕТОДЫ ДЛЯ СОЗДАНИЯ И РЕДАКТИРОВАНИЯ УРОКОВ
    # =============================================================================
    
    def add_lesson(self, subject: str, title: str, content: str, 
                  class_level: str = '5') -> Dict:
        """Добавляет новыи урок"""
        try:
            if not title or not content:
                return {"success": False, "error": "Title and content are required"}
            
            if class_level == 'demo':
                lesson_dir = self.lessons_demo_dir
            else:
                class_dir = self.lessons_students_dir / f"{class_level}_class"
                class_dir.mkdir(parents=True, exist_ok=True)
                
                subject_dir = class_dir / subject
                subject_dir.mkdir(parents=True, exist_ok=True)
                lesson_dir = subject_dir
            
            filename = f"lesson_{title.lower().replace(' ', '_')}.txt"
            lesson_path = lesson_dir / filename
            
            with open(lesson_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {"success": True, "filename": filename, "subject": subject, 
                    "title": title, "class_level": class_level}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def add_practice(self, lesson_id: str, practice_data: Dict) -> Dict:
        """Добавляет практические задания к уроку"""
        try:
            if not lesson_id or not practice_data:
                return {"success": False, "error": "Lesson ID and practice data are required"}
            
            practice_file = self.practice_dir / f"{lesson_id}.json"
            
            with open(practice_file, 'w', encoding='utf-8') as f:
                json.dump(practice_data, f, ensure_ascii=False, indent=2)
            
            return {"success": True, "lesson_id": lesson_id, 
                    "question_count": len(practice_data.get('questions', []))}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # =============================================================================
    # МЕТОДЫ ДЛЯ ПРОСМОТРА И РЕДАКТИРОВАНИЯ УРОКОВ
    # =============================================================================
    
    def view_lessons(self, class_filter: str = 'all', subject_filter: str = 'all', 
                    search_query: str = '') -> Dict:
        """Получает уроки для просмотра"""
        try:
            lessons = []
            
            if class_filter in ['all', 'demo']:
                for lesson_file in self.lessons_demo_dir.glob("*.txt"):
                    if search_query and search_query.lower() not in lesson_file.name.lower():
                        continue
                    
                    lessons.append({
                        'type': 'demo',
                        'class': 'demo',
                        'subject': 'demo',
                        'name': lesson_file.name,
                        'full_path': str(lesson_file.relative_to(self.lessons_dir)),
                        'size': lesson_file.stat().st_size,
                        'modified': datetime.fromtimestamp(lesson_file.stat().st_mtime).isoformat()
                    })
            
            if class_filter in ['all', 'generated']:
                for lesson_file in self.lessons_generated_dir.glob("*.txt"):
                    if search_query and search_query.lower() not in lesson_file.name.lower():
                        continue
                    
                    lessons.append({
                        'type': 'generated',
                        'class': 'generated',
                        'subject': 'auto',
                        'name': lesson_file.name,
                        'full_path': str(lesson_file.relative_to(self.lessons_dir)),
                        'size': lesson_file.stat().st_size,
                        'modified': datetime.fromtimestamp(lesson_file.stat().st_mtime).isoformat()
                    })
            
            if class_filter == 'all' or class_filter.isdigit():
                for class_dir in self.lessons_students_dir.glob("*_class"):
                    if class_dir.is_dir():
                        class_name = class_dir.name.replace("_class", "")
                        
                        if class_filter != 'all' and class_filter != class_name:
                            continue
                        
                        for subject_dir in class_dir.iterdir():
                            if subject_dir.is_dir():
                                subject_name = subject_dir.name
                                
                                if subject_filter != 'all' and subject_filter != subject_name:
                                    continue
                                
                                for lesson_file in subject_dir.glob("*.txt"):
                                    if search_query and search_query.lower() not in lesson_file.name.lower():
                                        continue
                                    
                                    lessons.append({
                                        'type': 'student',
                                        'class': class_name,
                                        'subject': subject_name,
                                        'name': lesson_file.name,
                                        'full_path': str(lesson_file.relative_to(self.lessons_dir)),
                                        'size': lesson_file.stat().st_size,
                                        'modified': datetime.fromtimestamp(lesson_file.stat().st_mtime).isoformat()
                                    })
            
            lessons.sort(key=lambda x: x['modified'], reverse=True)
            
            return {
                "success": True,
                "total": len(lessons),
                "lessons": lessons
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_lesson_for_edit(self, lesson_path: str) -> Dict:
        """Получает урок для редактирования"""
        try:
            full_path = self.lessons_dir / lesson_path
            
            if not full_path.exists():
                return {"success": False, "error": "Урок не наиден"}
            
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
    
    def save_edited_lesson(self, lesson_path: str, content: str) -> Dict:
        """Сохраняет отредактированныи урок"""
        try:
            if not lesson_path or content is None:
                return {"success": False, "error": "Не указаны данные урока"}
            
            full_path = self.lessons_dir / lesson_path
            
            if not full_path.exists():
                return {"success": False, "error": "Урок не наиден"}
            
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
    
    def delete_lesson(self, lesson_path: str) -> Dict:
        """Удаляет урок"""
        try:
            if not lesson_path:
                return {"success": False, "error": "Не указан путь к уроку"}
            
            full_path = self.lessons_dir / lesson_path
            
            if not full_path.exists():
                return {"success": False, "error": "Урок не наиден"}
            
            if full_path.name.startswith('demo_') or 'generated' in str(full_path):
                trash_dir = self.lessons_dir / 'trash'
                trash_dir.mkdir(exist_ok=True)
                backup_path = trash_dir / full_path.name
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
