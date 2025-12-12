# student_manager.py - Управление учениками и их данными для AI Teacher System

import json
import uuid
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
import re

# =============================================================================
# НАСТРОЙКИ ПУТЕЙ
# =============================================================================

BASE_DIR = Path(__file__).parent
STUDENTS_DIR = BASE_DIR / "students_data"

# Создаем папку если не существует
STUDENTS_DIR.makedir(exist_ok=True)

# =============================================================================
# КЛАСС ДЛЯ УПРАВЛЕНИЯ УЧЕНИКАМИ
# =============================================================================

class StudentManager:
    """Менеджер для управления данными учеников"""
    
    def __init__(self):
        self.students_cache = {}
        self.student_rooms_cache = {}
    
    # =============================================================================
    # ОСНОВНЫЕ ОПЕРАЦИИ С ДАННЫМИ УЧЕНИКОВ
    # =============================================================================
    
    def save_student_data(self, student_data: Dict[str, Any]) -> Optional[str]:
        """Сохраняет данные ученика в JSON файл"""
        try:
            student_id = student_data.get('student_id')
            if not student_id:
                student_id = str(uuid.uuid4())
                student_data['student_id'] = student_id
            
            if 'conference_id' not in student_data:
                import time
                conference_id = str(int(time.time() * 1000))
                student_data['conference_id'] = conference_id
            
            student_data['last_updated'] = datetime.now().isoformat()
            
            filename = f"{student_id}.json"
            filepath = STUDENTS_DIR / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(student_data, f, ensure_ascii=False, indent=2)
            
            # Обновляем кэш
            self.students_cache[student_id] = student_data
            
            print(f"✅ Данные ученика сохранены: {student_id}")
            return student_id
        except Exception as e:
            print(f"❌ Ошибка сохранения данных ученика: {e}")
            return None
    
    def load_student_data(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Загружает данные ученика из JSON файла"""
        try:
            # Проверяем кэш
            if student_id in self.students_cache:
                return self.students_cache[student_id]
            
            filename = f"{student_id}.json"
            filepath = STUDENTS_DIR / filename
            
            if not filepath.exists():
                return None
            
            with open(filepath, 'r', encoding='utf-8') as f:
                student_data = json.load(f)
            
            # Сохраняем в кэш
            self.students_cache[student_id] = student_data
            
            return student_data
        except Exception as e:
            print(f"❌ Ошибка загрузки данных ученика: {e}")
            return None
    
    def find_student_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Находит ученика по имени"""
        try:
            # Проверяем кэш
            for student_data in self.students_cache.values():
                if student_data.get('name', '').lower() == name.lower():
                    return student_data
            
            # Ищем в файлах
            for filepath in STUDENTS_DIR.glob("*.json"):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('name', '').lower() == name.lower():
                        # Сохраняем в кэш
                        self.students_cache[data.get('student_id')] = data
                        return data
            return None
        except Exception as e:
            print(f"❌ Ошибка поиска ученика: {e}")
            return None
    
    def update_student_data(self, student_id: str, updates: Dict[str, Any]) -> bool:
        """Обновляет данные ученика"""
        try:
            current_data = self.load_student_data(student_id)
            if not current_data:
                return False
            
            current_data.update(updates)
            current_data['last_updated'] = datetime.now().isoformat()
            
            result = self.save_student_data(current_data) is not None
            if result:
                print(f"✅ Данные ученика обновлены: {student_id}")
            return result
        except Exception as e:
            print(f"❌ Ошибка обновления данных ученика: {e}")
            return False
    
    def get_all_students(self) -> List[Dict[str, Any]]:
        """Получает список всех учеников"""
        try:
            students = []
            for filepath in STUDENTS_DIR.glob("*.json"):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        student_data = json.load(f)
                        students.append({
                            'student_id': student_data.get('student_id'),
                            'name': student_data.get('name', 'Неизвестно'),
                            'age': student_data.get('age'),
                            'education_level': student_data.get('education_level'),
                            'conference_id': student_data.get('conference_id'),
                            'last_updated': student_data.get('last_updated'),
                            'rooms_count': len(student_data.get('rooms', [])),
                            'has_rooms': 'rooms' in student_data and bool(student_data['rooms'])
                        })
                except Exception as e:
                    print(f"⚠️ Ошибка загрузки файла ученика {filepath}: {e}")
                    continue
            
            # Сортируем по имени
            students.sort(key=lambda x: x.get('name', '').lower())
            return students
        except Exception as e:
            print(f"❌ Ошибка получения списка учеников: {e}")
            return []
    
    # =============================================================================
    # УПРАВЛЕНИЕ КОМНАТАМИ УЧЕНИКОВ
    # =============================================================================
    
    def create_student_rooms(self, student_data: Dict[str, Any]) -> bool:
        """Автоматически создает комнаты для ученика"""
        try:
            student_id = student_data.get('student_id')
            student_name = student_data.get('name')
            conference_id = student_data.get('conference_id')
            
            if not student_id or not student_name:
                return False
            
            if not conference_id:
                import time
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
                
                room_data = {
                    'name': student_name,
                    'age': student_data.get('age'),
                    'education_level': student_data.get('education_level'),
                    'subject': subject,
                    'student_id': student_id,
                    'conference_id': conference_id
                }
                
                # Сохраняем в кэш комнат
                if conference_id not in self.student_rooms_cache:
                    self.student_rooms_cache[conference_id] = {}
                self.student_rooms_cache[conference_id][room_name] = room_data
                
                created_rooms.append({
                    'subject': subject,
                    'subject_name': subject,
                    'room_name': room_name,
                    'avatar': 'woman',
                    'conference_id': conference_id,
                    'student_data': room_data
                })
            
            student_data['rooms'] = created_rooms
            student_data['default_avatar'] = 'woman'
            student_data['conference_id'] = conference_id
            
            # Сохраняем обновленные данные ученика
            self.save_student_data(student_data)
            
            print(f"✅ Создано {len(created_rooms)} комнат для ученика {student_name} с ID: {conference_id}")
            return True
        except Exception as e:
            print(f"❌ Ошибка создания комнат для ученика: {e}")
            return False
    
    def get_student_rooms(self, student_id: str) -> List[Dict[str, Any]]:
        """Получает комнаты ученика"""
        try:
            student_data = self.load_student_data(student_id)
            if not student_data:
                return []
            
            return student_data.get('rooms', [])
        except Exception as e:
            print(f"❌ Ошибка получения комнат ученика: {e}")
            return []
    
    def get_student_room_for_subject(self, student_id: str, subject: str) -> Optional[Dict[str, Any]]:
        """Получает комнату ученика для конкретного предмета"""
        try:
            rooms = self.get_student_rooms(student_id)
            for room in rooms:
                if room.get('subject') == subject:
                    return room
            return None
        except Exception as e:
            print(f"❌ Ошибка получения комнаты ученика: {e}")
            return None
    
    def create_student_room_for_subject(self, student_data: Dict[str, Any], subject: str) -> Optional[Dict[str, Any]]:
        """Создает комнату для ученика по предмету"""
        try:
            conference_id = student_data.get('conference_id')
            if not conference_id:
                import time
                conference_id = str(int(time.time() * 1000))
                student_data['conference_id'] = conference_id
            
            student_name = student_data.get('name', 'ученик').replace(' ', '_').lower()
            room_name = f"student_{subject}_{student_name}_{conference_id}"
            
            room_data = {
                'name': student_data.get('name'),
                'age': student_data.get('age'),
                'education_level': student_data.get('education_level'),
                'subject': subject,
                'student_id': student_data.get('student_id'),
                'conference_id': conference_id
            }
            
            # Сохраняем в кэш
            if conference_id not in self.student_rooms_cache:
                self.student_rooms_cache[conference_id] = {}
            self.student_rooms_cache[conference_id][room_name] = room_data
            
            room_info = {
                'subject': subject,
                'room_name': room_name,
                'conference_id': conference_id,
                'student_name': student_data.get('name', ''),
                'student_class': student_data.get('education_level', '5'),
                'student_data': room_data
            }
            
            # Обновляем данные ученика
            student_rooms = student_data.get('rooms', [])
            # Удаляем старую комнату для этого предмета если есть
            student_rooms = [r for r in student_rooms if r.get('subject') != subject]
            student_rooms.append(room_info)
            student_data['rooms'] = student_rooms
            
            self.save_student_data(student_data)
            
            print(f"✅ Создана комната {room_name} для ученика {student_data.get('name')}, предмет: {subject}")
            return room_info
        except Exception as e:
            print(f"❌ Ошибка создания комнаты ученика: {e}")
            return None
    
    # =============================================================================
    # УПРАВЛЕНИЕ УРОКАМИ УЧЕНИКОВ
    # =============================================================================
    
    def add_student_lesson(self, student_id: str, lesson_data: Dict[str, Any]) -> bool:
        """Добавляет урок в историю ученика"""
        try:
            student_data = self.load_student_data(student_id)
            if not student_data:
                return False
            
            if 'lessons' not in student_data:
                student_data['lessons'] = []
            
            # Проверяем, нет ли уже такого урока
            lesson_exists = any(
                l.get('lesson_id') == lesson_data.get('lesson_id') 
                for l in student_data['lessons']
            )
            
            if not lesson_exists:
                student_data['lessons'].append({
                    'lesson_id': lesson_data.get('lesson_id'),
                    'subject': lesson_data.get('subject'),
                    'title': lesson_data.get('title'),
                    'date': datetime.now().isoformat(),
                    'duration': lesson_data.get('duration', 0),
                    'score': lesson_data.get('score'),
                    'completed': lesson_data.get('completed', False)
                })
                
                student_data['last_activity'] = datetime.now().isoformat()
                
                return self.save_student_data(student_data) is not None
            
            return True
        except Exception as e:
            print(f"❌ Ошибка добавления урока ученику: {e}")
            return False
    
    def get_student_lessons(self, student_id: str) -> List[Dict[str, Any]]:
        """Получает историю уроков ученика"""
        try:
            student_data = self.load_student_data(student_id)
            if not student_data:
                return []
            
            return student_data.get('lessons', [])
        except Exception as e:
            print(f"❌ Ошибка получения уроков ученика: {e}")
            return []
    
    def get_student_lesson_stats(self, student_id: str) -> Dict[str, Any]:
        """Получает статистику уроков ученика"""
        try:
            lessons = self.get_student_lessons(student_id)
            
            total_lessons = len(lessons)
            completed_lessons = len([l for l in lessons if l.get('completed', False)])
            total_duration = sum(l.get('duration', 0) for l in lessons)
            
            # Группируем по предметам
            subjects = {}
            for lesson in lessons:
                subject = lesson.get('subject', 'Неизвестно')
                if subject not in subjects:
                    subjects[subject] = {
                        'total': 0,
                        'completed': 0,
                        'total_duration': 0,
                        'average_score': 0,
                        'scores': []
                    }
                
                subjects[subject]['total'] += 1
                if lesson.get('completed', False):
                    subjects[subject]['completed'] += 1
                
                subjects[subject]['total_duration'] += lesson.get('duration', 0)
                
                if lesson.get('score') is not None:
                    subjects[subject]['scores'].append(lesson.get('score'))
            
            # Рассчитываем средние баллы
            for subject, data in subjects.items():
                if data['scores']:
                    data['average_score'] = sum(data['scores']) / len(data['scores'])
            
            return {
                'total_lessons': total_lessons,
                'completed_lessons': completed_lessons,
                'completion_rate': (completed_lessons / total_lessons * 100) if total_lessons > 0 else 0,
                'total_duration_minutes': total_duration,
                'subjects': subjects
            }
        except Exception as e:
            print(f"❌ Ошибка получения статистики ученика: {e}")
            return {}
    
    # =============================================================================
    # ПОИСК И ФИЛЬТРАЦИЯ УЧЕНИКОВ
    # =============================================================================
    
    def search_students(self, query: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Поиск учеников по запросу и фильтрам"""
        try:
            all_students = self.get_all_students()
            
            if not query and not filters:
                return all_students
            
            filtered_students = []
            query_lower = query.lower() if query else ""
            
            for student in all_students:
                match = True
                
                # Поиск по тексту
                if query:
                    name_match = query_lower in student.get('name', '').lower()
                    class_match = query_lower in str(student.get('education_level', '')).lower()
                    id_match = query_lower in student.get('student_id', '').lower()
                    
                    if not (name_match or class_match or id_match):
                        match = False
                
                # Фильтры
                if match and filters:
                    if 'education_level' in filters and filters['education_level']:
                        if student.get('education_level') != filters['education_level']:
                            match = False
                    
                    if 'has_rooms' in filters and filters['has_rooms'] is not None:
                        has_rooms = student.get('has_rooms', False)
                        if filters['has_rooms'] and not has_rooms:
                            match = False
                        elif not filters['has_rooms'] and has_rooms:
                            match = False
                
                if match:
                    filtered_students.append(student)
            
            return filtered_students
        except Exception as e:
            print(f"❌ Ошибка поиска учеников: {e}")
            return []
    
    def get_students_by_class(self, class_level: str) -> List[Dict[str, Any]]:
        """Получает учеников по классу"""
        try:
            all_students = self.get_all_students()
            return [s for s in all_students if s.get('education_level') == class_level]
        except Exception as e:
            print(f"❌ Ошибка получения учеников по классу: {e}")
            return []
    
    # =============================================================================
    # ЭКСПОРТ И ИМПОРТ ДАННЫХ
    # =============================================================================
    
    def export_student_data(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Экспортирует данные ученика"""
        try:
            student_data = self.load_student_data(student_id)
            if not student_data:
                return None
            
            # Добавляем статистику
            stats = self.get_student_lesson_stats(student_id)
            
            export_data = {
                'student_info': {
                    'student_id': student_data.get('student_id'),
                    'name': student_data.get('name'),
                    'education_level': student_data.get('education_level'),
                    'age': student_data.get('age'),
                    'conference_id': student_data.get('conference_id'),
                    'created_at': student_data.get('registration_date'),
                    'last_updated': student_data.get('last_updated')
                },
                'rooms': student_data.get('rooms', []),
                'lessons': student_data.get('lessons', []),
                'statistics': stats,
                'export_date': datetime.now().isoformat(),
                'export_format': 'ai_teacher_v1'
            }
            
            return export_data
        except Exception as e:
            print(f"❌ Ошибка экспорта данных ученика: {e}")
            return None
    
    def export_all_students_data(self) -> Dict[str, Any]:
        """Экспортирует данные всех учеников"""
        try:
            all_students = self.get_all_students()
            
            export_data = {
                'total_students': len(all_students),
                'export_date': datetime.now().isoformat(),
                'export_format': 'ai_teacher_batch_v1',
                'students': []
            }
            
            for student in all_students:
                student_id = student.get('student_id')
                if student_id:
                    student_export = self.export_student_data(student_id)
                    if student_export:
                        export_data['students'].append(student_export)
            
            return export_data
        except Exception as e:
            print(f"❌ Ошибка экспорта всех данных учеников: {e}")
            return {'error': str(e), 'students': []}
    
    def import_student_data(self, import_data: Dict[str, Any]) -> Dict[str, Any]:
        """Импортирует данные ученика"""
        try:
            student_info = import_data.get('student_info', {})
            
            # Проверяем обязательные поля
            required_fields = ['name', 'education_level']
            for field in required_fields:
                if field not in student_info:
                    return {
                        'success': False,
                        'error': f'Отсутствует обязательное поле: {field}'
                    }
            
            # Ищем существующего ученика
            existing_student = self.find_student_by_name(student_info['name'])
            
            if existing_student:
                # Обновляем существующего ученика
                student_id = existing_student['student_id']
                
                # Обновляем основные данные
                updates = {
                    'education_level': student_info.get('education_level'),
                    'age': student_info.get('age'),
                    'last_updated': datetime.now().isoformat()
                }
                
                if self.update_student_data(student_id, updates):
                    # Добавляем уроки если есть
                    lessons = import_data.get('lessons', [])
                    for lesson in lessons:
                        self.add_student_lesson(student_id, lesson)
                    
                    return {
                        'success': True,
                        'message': f'Данные ученика {student_info["name"]} обновлены',
                        'student_id': student_id,
                        'updated': True
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Ошибка обновления данных ученика'
                    }
            else:
                # Создаем нового ученика
                import time
                student_data = {
                    'student_id': str(uuid.uuid4()),
                    'name': student_info['name'],
                    'education_level': student_info['education_level'],
                    'age': student_info.get('age'),
                    'registration_date': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat()
                }
                
                # Создаем conference_id если нет
                if 'conference_id' not in student_data:
                    student_data['conference_id'] = str(int(time.time() * 1000))
                
                # Сохраняем данные
                student_id = self.save_student_data(student_data)
                
                if student_id:
                    # Добавляем уроки если есть
                    lessons = import_data.get('lessons', [])
                    for lesson in lessons:
                        self.add_student_lesson(student_id, lesson)
                    
                    # Создаем комнаты
                    self.create_student_rooms(student_data)
                    
                    return {
                        'success': True,
                        'message': f'Создан новый ученик {student_info["name"]}',
                        'student_id': student_id,
                        'created': True
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Ошибка создания ученика'
                    }
        except Exception as e:
            return {
                'success': False,
                'error': f'Ошибка импорта данных: {str(e)}'
            }
    
    # =============================================================================
    # УПРАВЛЕНИЕ АВАТАРАМИ
    # =============================================================================
    
    def set_student_avatar(self, student_id: str, avatar_name: str) -> bool:
        """Устанавливает аватар для ученика"""
        try:
            student_data = self.load_student_data(student_id)
            if student_data:
                student_data['preferred_avatar'] = avatar_name
                return self.save_student_data(student_data) is not None
            return False
        except Exception as e:
            print(f"❌ Ошибка установки аватара ученика: {e}")
            return False
    
    def get_student_avatar(self, student_id: str) -> str:
        """Получает предпочитаемый аватар ученика"""
        try:
            student_data = self.load_student_data(student_id)
            if student_data:
                return student_data.get('preferred_avatar', 'woman')
            return 'woman'
        except Exception as e:
            print(f"❌ Ошибка получения аватара ученика: {e}")
            return 'woman'
    
    # =============================================================================
    # СИСТЕМНЫЕ ФУНКЦИИ
    # =============================================================================
    
    def cleanup_inactive_students(self, days_inactive: int = 90) -> Dict[str, Any]:
        """Очистка неактивных учеников"""
        try:
            from datetime import datetime, timedelta
            
            cutoff_date = datetime.now() - timedelta(days=days_inactive)
            cutoff_iso = cutoff_date.isoformat()
            
            inactive_students = []
            active_students = []
            
            for student_id, student_data in self.students_cache.items():
                last_updated = student_data.get('last_updated')
                if last_updated:
                    last_date = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                    if last_date < cutoff_date:
                        inactive_students.append(student_data)
                    else:
                        active_students.append(student_data)
            
            # Также проверяем файлы
            for filepath in STUDENTS_DIR.glob("*.json"):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        student_data = json.load(f)
                    
                    last_updated = student_data.get('last_updated')
                    if last_updated:
                        last_date = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                        if last_date < cutoff_date:
                            # Проверяем, нет ли уже в списке
                            if student_data.get('student_id') not in [s.get('student_id') for s in inactive_students]:
                                inactive_students.append(student_data)
                        else:
                            if student_data.get('student_id') not in [s.get('student_id') for s in active_students]:
                                active_students.append(student_data)
                except Exception as e:
                    print(f"⚠️ Ошибка обработки файла {filepath}: {e}")
                    continue
            
            return {
                'success': True,
                'total_students': len(inactive_students) + len(active_students),
                'inactive_students': len(inactive_students),
                'active_students': len(active_students),
                'inactive_list': [{'name': s.get('name'), 'student_id': s.get('student_id'), 
                                  'last_updated': s.get('last_updated')} for s in inactive_students]
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Получает системную статистику по ученикам"""
        try:
            all_students = self.get_all_students()
            
            # Статистика по классам
            class_stats = {}
            for student in all_students:
                class_level = student.get('education_level', 'Не указан')
                if class_level not in class_stats:
                    class_stats[class_level] = 0
                class_stats[class_level] += 1
            
            # Статистика по комнатам
            total_rooms = 0
            students_with_rooms = 0
            for student in all_students:
                rooms_count = student.get('rooms_count', 0)
                total_rooms += rooms_count
                if rooms_count > 0:
                    students_with_rooms += 1
            
            return {
                'success': True,
                'total_students': len(all_students),
                'students_with_rooms': students_with_rooms,
                'students_without_rooms': len(all_students) - students_with_rooms,
                'total_rooms': total_rooms,
                'average_rooms_per_student': total_rooms / len(all_students) if all_students else 0,
                'class_distribution': class_stats,
                'cache_size': len(self.students_cache),
                'rooms_cache_size': sum(len(v) for v in self.student_rooms_cache.values())
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def validate_student_data(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """Валидация данных ученика"""
        errors = []
        warnings = []
        
        # Проверка обязательных полей
        required_fields = ['name', 'education_level']
        for field in required_fields:
            if field not in student_data or not student_data[field]:
                errors.append(f'Отсутствует обязательное поле: {field}')
        
        # Проверка имени
        name = student_data.get('name', '').strip()
        if name:
            if len(name) < 2:
                errors.append('Имя слишком короткое (минимум 2 символа)')
            if len(name) > 50:
                warnings.append('Имя слишком длинное (рекомендуется до 50 символов)')
            
            # Проверка на допустимые символы
            if not re.match(r'^[a-zA-Zа-яА-ЯёЁ\s\-]+$', name):
                warnings.append('Имя содержит нестандартные символы')
        
        # Проверка возраста
        age = student_data.get('age')
        if age:
            try:
                age_int = int(age)
                if age_int < 5 or age_int > 18:
                    warnings.append(f'Возраст {age} выходит за обычные границы школьного возраста')
            except ValueError:
                warnings.append('Возраст должен быть числом')
        
        # Проверка класса
        class_level = student_data.get('education_level')
        if class_level:
            valid_classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
            if class_level not in valid_classes:
                warnings.append(f'Класс {class_level} не является стандартным школьным классом')
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'has_warnings': len(warnings) > 0
        }
    
    def fix_student_data_issues(self) -> Dict[str, Any]:
        """Исправление проблем с данными учеников"""
        try:
            issues_fixed = 0
            students_checked = 0
            
            for filepath in STUDENTS_DIR.glob("*.json"):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        student_data = json.load(f)
                    
                    students_checked += 1
                    needs_fix = False
                    
                    # Проверяем наличие обязательных полей
                    if 'student_id' not in student_data:
                        student_data['student_id'] = str(uuid.uuid4())
                        needs_fix = True
                    
                    if 'conference_id' not in student_data:
                        import time
                        student_data['conference_id'] = str(int(time.time() * 1000))
                        needs_fix = True
                    
                    if 'registration_date' not in student_data:
                        student_data['registration_date'] = datetime.now().isoformat()
                        needs_fix = True
                    
                    if 'last_updated' not in student_data:
                        student_data['last_updated'] = datetime.now().isoformat()
                        needs_fix = True
                    
                    # Проверяем формат дат
                    date_fields = ['registration_date', 'last_updated']
                    for field in date_fields:
                        if field in student_data:
                            try:
                                datetime.fromisoformat(student_data[field].replace('Z', '+00:00'))
                            except ValueError:
                                student_data[field] = datetime.now().isoformat()
                                needs_fix = True
                    
                    # Проверяем наличие комнат
                    if 'rooms' not in student_data:
                        student_data['rooms'] = []
                        needs_fix = True
                    
                    if needs_fix:
                        # Сохраняем исправленные данные
                        with open(filepath, 'w', encoding='utf-8') as f:
                            json.dump(student_data, f, ensure_ascii=False, indent=2)
                        
                        # Обновляем кэш
                        student_id = student_data.get('student_id')
                        if student_id:
                            self.students_cache[student_id] = student_data
                        
                        issues_fixed += 1
                        print(f"✅ Исправлены данные ученика: {student_data.get('name', 'Неизвестно')}")
                
                except Exception as e:
                    print(f"⚠️ Ошибка обработки файла {filepath}: {e}")
                    continue
            
            return {
                'success': True,
                'message': f'Проверено {students_checked} учеников, исправлено {issues_fixed} проблем',
                'students_checked': students_checked,
                'issues_fixed': issues_fixed
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Глобальный экземпляр StudentManager
_student_manager_instance = None

def get_student_manager():
    """Получение глобального экземпляра StudentManager"""
    global _student_manager_instance
    if _student_manager_instance is None:
        _student_manager_instance = StudentManager()
    return _student_manager_instance
