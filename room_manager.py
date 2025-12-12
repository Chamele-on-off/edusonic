# room_manager.py - Управление комнатами и состояниями для AI Teacher System

import time
import threading
from collections import defaultdict
from typing import Dict, Set, List, Optional, Any
import uuid
import random
from datetime import datetime

from dialogue import DialogueManager
from config import get_llm_mode

# =============================================================================
# ГЛОБАЛЬНЫЕ СОСТОЯНИЯ КОМНАТ
# =============================================================================

class RoomManager:
    """Менеджер для управления комнатами и их состояниями"""
    
    def __init__(self, socketio):
        self.socketio = socketio
        self.room_participants = defaultdict(set)
        self.room_speech_data = defaultdict(list)
        self.room_speaking = defaultdict(bool)
        self.room_ai_activated = defaultdict(bool)
        self.room_dialogue = defaultdict(lambda: None)
        self.room_lessons = defaultdict(dict)
        self.room_llm_mode = defaultdict(lambda: get_llm_mode())
        self.room_teacher_speaking = defaultdict(bool)
        self.room_practice_active = defaultdict(bool)
        self.room_current_question_index = defaultdict(int)
        self.room_current_avatar = defaultdict(lambda: 'woman')
        self.room_last_activity = defaultdict(lambda: time.time())
        
        # PeerJS tracking
        self.room_peer_ids = defaultdict(dict)
        
        # Данные учеников
        self.room_student_data = defaultdict(dict)
        
        # Очереди ответов LLM
        self.room_llm_responses = defaultdict(list)
        self.room_last_poll_time = defaultdict(lambda: 0)
        self.room_llm_pending_requests = defaultdict(dict)
        self.room_last_llm_update = defaultdict(lambda: 0)
        
        # Настройки очистки
        self.ROOM_TIMEOUT = 3600  # 1 час неактивности
        self.MAX_ROOMS = 100  # Максимальное количество комнат в памяти
        
        # Ограничитель параллельной инициализации комнат
        from threading import Semaphore
        self.init_semaphore = Semaphore(10)  # Не более 10 одновременных инициализаций
    
    def cleanup_inactive_rooms(self) -> int:
        """Очистка неактивных комнат"""
        try:
            current_time = time.time()
            rooms_to_remove = []
            
            for room_id, last_active in self.room_last_activity.items():
                if (current_time - last_active > self.ROOM_TIMEOUT and 
                    len(self.room_participants.get(room_id, [])) == 0):
                    rooms_to_remove.append(room_id)
            
            for room_id in rooms_to_remove:
                try:
                    self._clean_room_data(room_id)
                    print(f"✅ Очищена неактивная комната: {room_id}")
                except Exception as e:
                    print(f"⚠️ Ошибка очистки комнаты {room_id}: {e}")
            
            return len(rooms_to_remove)
        except Exception as e:
            print(f"❌ Ошибка очистки комнат: {e}")
            return 0
    
    def _clean_room_data(self, room_id: str):
        """Очистка всех данных комнаты"""
        if room_id in self.room_dialogue:
            del self.room_dialogue[room_id]
        if room_id in self.room_participants:
            del self.room_participants[room_id]
        if room_id in self.room_student_data:
            del self.room_student_data[room_id]
        if room_id in self.room_speech_data:
            del self.room_speech_data[room_id]
        if room_id in self.room_ai_activated:
            del self.room_ai_activated[room_id]
        if room_id in self.room_llm_mode:
            del self.room_llm_mode[room_id]
        if room_id in self.room_teacher_speaking:
            del self.room_teacher_speaking[room_id]
        if room_id in self.room_practice_active:
            del self.room_practice_active[room_id]
        if room_id in self.room_current_question_index:
            del self.room_current_question_index[room_id]
        if room_id in self.room_current_avatar:
            del self.room_current_avatar[room_id]
        if room_id in self.room_llm_responses:
            del self.room_llm_responses[room_id]
        if room_id in self.room_llm_pending_requests:
            del self.room_llm_pending_requests[room_id]
        if room_id in self.room_last_activity:
            del self.room_last_activity[room_id]
        if room_id in self.room_speaking:
            del self.room_speaking[room_id]
        if room_id in self.room_last_poll_time:
            del self.room_last_poll_time[room_id]
        if room_id in self.room_last_llm_update:
            del self.room_last_llm_update[room_id]
        if room_id in self.room_peer_ids:
            del self.room_peer_ids[room_id]
        if room_id in self.room_lessons:
            del self.room_lessons[room_id]
    
    def periodic_cleanup(self):
        """Периодическая очистка неактивных комнат"""
        try:
            cleaned = self.cleanup_inactive_rooms()
            if cleaned > 0:
                print(f"🧹 Периодическая очистка: удалено {cleaned} комнат")
            
            # Ограничение количества комнат в памяти
            if len(self.room_participants) > self.MAX_ROOMS:
                # Удаляем самые старые комнаты
                rooms_sorted = sorted(self.room_last_activity.items(), key=lambda x: x[1])
                rooms_to_remove = rooms_sorted[:len(self.room_participants) - self.MAX_ROOMS]
                for room_id, _ in rooms_to_remove:
                    if len(self.room_participants.get(room_id, [])) == 0:
                        try:
                            self._clean_room_data(room_id)
                            print(f"🧹 Удалена старая комната для оптимизации: {room_id}")
                        except Exception as e:
                            print(f"⚠️ Ошибка удаления старой комнаты: {e}")
        except Exception as e:
            print(f"❌ Ошибка периодической очистки: {e}")
        
        # Повторяем каждые 10 минут
        threading.Timer(600, self.periodic_cleanup).start()
    
    def handle_disconnected_session(self, sid: str):
        """Безопасная обработка отключенных сессий"""
        try:
            # Удаляем из комнат
            for room_id, participants in self.room_participants.items():
                if sid in participants:
                    participants.remove(sid)
                    print(f"🔧 Удален отключенный участник {sid} из комнаты {room_id}")
                    self.socketio.emit('participants_update', {'count': len(participants)}, room=room_id)
            
            # Удаляем peer IDs
            for room_id, peers in self.room_peer_ids.items():
                if sid in peers:
                    del peers[sid]
                    print(f"🔧 Удален peer_id для отключенного участника {sid}")
        except Exception as e:
            print(f"⚠️ Ошибка очистки отключенной сессии {sid}: {e}")
    
    # =============================================================================
    # ИНИЦИАЛИЗАЦИЯ КОМНАТ
    # =============================================================================
    
    def _fast_room_initialization(self, room_id: str) -> bool:
        """Быстрая инициализация комнаты с ограничением"""
        with self.init_semaphore:
            try:
                # Обновляем время активности
                self.room_last_activity[room_id] = time.time()
                
                # Минимальная инициализация
                if room_id not in self.room_ai_activated:
                    self.room_ai_activated[room_id] = False
                
                if room_id not in self.room_llm_mode:
                    self.room_llm_mode[room_id] = get_llm_mode()
                
                if room_id not in self.room_current_avatar:
                    self.room_current_avatar[room_id] = 'teacher'
                
                # Ленивая инициализация DialogueManager
                if room_id not in self.room_dialogue:
                    self.room_dialogue[room_id] = None
                
                # Если это комната ученика, устанавливаем данные
                if room_id in self.room_student_data and self.room_student_data[room_id]:
                    student_data = self.room_student_data[room_id]
                    print(f"🔥 Комната {room_id} является комнатой ученика: {student_data.get('name')}")
                    
                    # Создаем DialogueManager если нужно
                    if self.room_dialogue[room_id] is None:
                        self.room_dialogue[room_id] = DialogueManager(self.socketio)
                        self.room_dialogue[room_id].room_id = room_id
                    
                    self.room_dialogue[room_id].set_student_data(student_data)
                    
                    if 'subject' in student_data and student_data['subject']:
                        self.room_dialogue[room_id].current_subject = student_data['subject']
                        print(f"🔥 Установлен предмет для ученика: {student_data['subject']}")
                    elif room_id.startswith('student_'):
                        parts = room_id.split('_')
                        if len(parts) > 1:
                            self.room_dialogue[room_id].current_subject = parts[1]
                            print(f"🔥 Предмет определен из имени комнаты: {parts[1]}")
                
                # Устанавливаем режим LLM
                if self.room_dialogue[room_id] is not None:
                    self.room_dialogue[room_id].set_llm_mode(self.room_llm_mode[room_id])
                
                print(f"✅ Быстрая инициализация завершена для комнаты {room_id}")
                return True
            except Exception as e:
                print(f"❌ Ошибка инициализации комнаты {room_id}: {e}")
                return False
    
    def delayed_full_init(self, room_id: str):
        """Отложенная полная инициализация комнаты"""
        time.sleep(1)  # Даем время на установление соединения
        try:
            if room_id in self.room_dialogue and self.room_dialogue[room_id] is None:
                with self.init_semaphore:
                    self.room_dialogue[room_id] = DialogueManager(self.socketio)
                    self.room_dialogue[room_id].room_id = room_id
                    self.room_dialogue[room_id].set_llm_mode(self.room_llm_mode[room_id])
                    
                    # Устанавливаем данные ученика если есть
                    if room_id in self.room_student_data and self.room_student_data[room_id]:
                        self.room_dialogue[room_id].set_student_data(self.room_student_data[room_id])
                    
                    print(f"✅ Выполнена отложенная полная инициализация комнаты {room_id}")
        except Exception as e:
            print(f"❌ Ошибка отложенной инициализации {room_id}: {e}")
    
    def join_room(self, room_id: str, sid: str, peer_id: str = None) -> Dict[str, Any]:
        """Присоединение к комнате"""
        try:
            # Обновляем время активности
            self.room_last_activity[room_id] = time.time()
            
            if room_id not in self.room_participants:
                self.room_participants[room_id] = set()
            
            self.room_participants[room_id].add(sid)
            
            if peer_id:
                if room_id not in self.room_peer_ids:
                    self.room_peer_ids[room_id] = {}
                self.room_peer_ids[room_id][sid] = peer_id
            
            # Быстрая инициализация
            self._fast_room_initialization(room_id)
            
            response = {
                'success': True,
                'room_id': room_id,
                'participants_count': len(self.room_participants[room_id]),
                'avatar': self.room_current_avatar[room_id]
            }
            
            # Отложенная полная инициализация для первой сессии
            if len(self.room_participants[room_id]) == 1:
                threading.Thread(target=self.delayed_full_init, args=(room_id,)).start()
            
            print(f"✅ Успешное присоединение к комнате {room_id}, участников: {len(self.room_participants[room_id])}")
            return response
        except Exception as e:
            print(f"❌ Критическая ошибка при присоединении к комнате {room_id}: {e}")
            return {'success': False, 'error': str(e)}
    
    def leave_room(self, room_id: str, sid: str):
        """Выход из комнаты"""
        try:
            if room_id in self.room_participants and sid in self.room_participants[room_id]:
                self.room_participants[room_id].remove(sid)
                print(f"🔧 Участник {sid} вышел из комнаты {room_id}")
            
            # Удаляем peer ID
            if room_id in self.room_peer_ids and sid in self.room_peer_ids[room_id]:
                del self.room_peer_ids[room_id][sid]
        except Exception as e:
            print(f"⚠️ Ошибка выхода из комнаты: {e}")
    
    # =============================================================================
    # УПРАВЛЕНИЕ СОСТОЯНИЯМИ КОМНАТ
    # =============================================================================
    
    def reset_speaking_state(self, room_id: str, is_teacher: bool = False):
        """Сбрасывает состояние речи для указанной комнаты"""
        self.room_speaking[room_id] = False
        if is_teacher:
            self.room_teacher_speaking[room_id] = False
        self.socketio.emit('speaking_state', {'speaking': False}, room=room_id)
    
    def activate_ai_teacher(self, room_id: str) -> Dict[str, Any]:
        """Активация AI-учителя в комнате"""
        try:
            self.room_ai_activated[room_id] = True
            
            if room_id not in self.room_dialogue or self.room_dialogue[room_id] is None:
                self.room_dialogue[room_id] = DialogueManager(self.socketio)
                self.room_dialogue[room_id].room_id = room_id
                print(f"Создан DialogueManager при активации для {room_id}")
            
            self.room_dialogue[room_id].set_llm_mode(self.room_llm_mode[room_id])
            
            return {
                'success': True,
                'room_id': room_id,
                'message': 'AI-учитель успешно активирован'
            }
        except Exception as e:
            print(f"❌ Ошибка активации AI-учителя: {e}")
            return {'success': False, 'error': str(e)}
    
    def set_llm_mode(self, room_id: str, mode: str) -> bool:
        """Установка режима LLM для комнаты"""
        if mode in ["traditional", "llm_first"]:
            self.room_llm_mode[room_id] = mode
            if room_id in self.room_dialogue and self.room_dialogue[room_id]:
                self.room_dialogue[room_id].set_llm_mode(mode)
            
            print(f"Режим LLM изменен в комнате {room_id}: {mode}")
            return True
        return False
    
    def set_practice_active(self, room_id: str, active: bool):
        """Активация/деактивация практики"""
        self.room_practice_active[room_id] = active
        if not active:
            self.room_current_question_index[room_id] = 0
    
    def update_student_data(self, room_id: str, student_data: Dict[str, Any]):
        """Обновление данных ученика в комнате"""
        self.room_student_data[room_id] = student_data
        
        # Обновляем данные в DialogueManager если он существует
        if room_id in self.room_dialogue and self.room_dialogue[room_id]:
            self.room_dialogue[room_id].set_student_data(student_data)
    
    def add_speech_data(self, room_id: str, speech_data: Dict[str, Any]):
        """Добавление данных речи в историю комнаты"""
        self.room_speech_data[room_id].append(speech_data)
        if len(self.room_speech_data[room_id]) > 50:
            self.room_speech_data[room_id].pop(0)
    
    def get_speech_history(self, room_id: str) -> List[Dict[str, Any]]:
        """Получение истории речи комнаты"""
        return self.room_speech_data.get(room_id, [])
    
    def get_room_info(self, room_id: str) -> Dict[str, Any]:
        """Получение информации о комнате"""
        return {
            'room_id': room_id,
            'participants': len(self.room_participants.get(room_id, [])),
            'ai_activated': self.room_ai_activated.get(room_id, False),
            'practice_active': self.room_practice_active.get(room_id, False),
            'current_avatar': self.room_current_avatar.get(room_id, 'woman'),
            'has_dialogue': room_id in self.room_dialogue and self.room_dialogue[room_id] is not None,
            'has_student_data': room_id in self.room_student_data,
            'last_activity': self.room_last_activity.get(room_id, 0)
        }
    
    # =============================================================================
    # УПРАВЛЕНИЕ LLM ОТВЕТАМИ
    # =============================================================================
    
    def add_llm_response(self, room_id: str, request_id: str, response: str, 
                         delivered_via_websocket: bool = False) -> Dict[str, Any]:
        """Добавление ответа LLM в очередь комнаты"""
        try:
            response_data = {
                'request_id': request_id,
                'response': response,
                'timestamp': time.time(),
                'delivered_via_websocket': delivered_via_websocket
            }
            
            self.room_llm_responses[room_id].append(response_data)
            
            # Ограничиваем размер очереди
            if len(self.room_llm_responses[room_id]) > 10:
                self.room_llm_responses[room_id].pop(0)
            
            print(f"📨 Добавлен ответ LLM в комнату {room_id}, запрос: {request_id[:20]}...")
            
            return {
                'success': True,
                'response_id': request_id,
                'queue_size': len(self.room_llm_responses[room_id])
            }
        except Exception as e:
            print(f"❌ Ошибка добавления ответа LLM: {e}")
            return {'success': False, 'error': str(e)}
    
    def poll_llm_responses(self, room_id: str, last_check: float = 0, 
                           request_id_filter: str = None) -> Dict[str, Any]:
        """Опрос новых ответов LLM"""
        try:
            current_time = time.time()
            self.room_last_poll_time[room_id] = current_time
            
            new_responses = []
            if room_id in self.room_llm_responses and self.room_llm_responses[room_id]:
                for resp in self.room_llm_responses[room_id]:
                    if (resp['timestamp'] > last_check and 
                        (not request_id_filter or resp['request_id'] == request_id_filter)):
                        new_responses.append(resp)
            
            if new_responses:
                new_responses.sort(key=lambda x: x['timestamp'], reverse=True)
                
                return {
                    "success": True,
                    "has_response": True,
                    "responses": new_responses,
                    "timestamp": current_time,
                    "total_new_responses": len(new_responses)
                }
            
            return {
                "success": True, 
                "has_response": False,
                "timestamp": current_time,
                "queue_size": len(self.room_llm_responses.get(room_id, [])),
                "pending_requests": len(self.room_llm_pending_requests.get(room_id, {}))
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def clear_llm_queue(self, room_id: str) -> Dict[str, Any]:
        """Очистка очереди ответов LLM"""
        try:
            if room_id in self.room_llm_responses:
                self.room_llm_responses[room_id].clear()
            
            if room_id in self.room_llm_pending_requests:
                self.room_llm_pending_requests[room_id].clear()
            
            print(f"🧹 Очищена очередь LLM для комнаты {room_id}")
            
            return {
                "success": True,
                "message": f"Очередь ответов для комнаты {room_id} очищена"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def add_pending_request(self, room_id: str, request_id: str, request_data: Dict[str, Any]):
        """Добавление ожидающего запроса LLM"""
        if room_id not in self.room_llm_pending_requests:
            self.room_llm_pending_requests[room_id] = {}
        
        self.room_llm_pending_requests[room_id][request_id] = request_data
        
        # Очищаем старые запросы
        current_time = time.time()
        for req_id in list(self.room_llm_pending_requests[room_id].keys()):
            if current_time - self.room_llm_pending_requests[room_id][req_id]['timestamp'] > 300:
                del self.room_llm_pending_requests[room_id][req_id]
    
    def remove_pending_request(self, room_id: str, request_id: str):
        """Удаление ожидающего запроса LLM"""
        if room_id in self.room_llm_pending_requests and request_id in self.room_llm_pending_requests[room_id]:
            del self.room_llm_pending_requests[room_id][request_id]
    
    # =============================================================================
    # СОЗДАНИЕ КОМНАТ ДЛЯ УЧЕНИКОВ
    # =============================================================================
    
    def create_student_conference(self, student_data: Dict[str, Any], subject: str = None) -> Optional[Dict[str, Any]]:
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
            
            self.room_student_data[room_id] = {
                **student_data,
                'subject': room_subject,
                'conference_id': conference_id
            }
            
            self._fast_room_initialization(room_id)
            
            print(f"Создана комната {room_id} для ученика {student_data.get('name')}, предмет: {room_subject}")
            
            return {
                'room_id': room_id,
                'conference_url': f'/conference?room={room_id}',
                'student_data': self.room_student_data[room_id]
            }
        except Exception as e:
            print(f"❌ Ошибка создания студенческой конференции: {e}")
            return None
    
    def create_student_rooms(self, student_data: Dict[str, Any]) -> bool:
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
                
                self.room_student_data[room_name] = {
                    'name': student_name,
                    'age': student_data.get('age'),
                    'education_level': student_data.get('education_level'),
                    'subject': subject,
                    'student_id': student_id,
                    'conference_id': conference_id
                }
                
                self._fast_room_initialization(room_name)
                
                print(f"Создана комната {room_name} для ученика {student_name}, предмет: {subject}")
                
                created_rooms.append({
                    'subject': subject,
                    'subject_name': subject,
                    'room_name': room_name,
                    'avatar': 'woman',
                    'conference_id': conference_id,
                    'student_data': self.room_student_data[room_name]
                })
            
            student_data['rooms'] = created_rooms
            student_data['default_avatar'] = 'woman'
            student_data['conference_id'] = conference_id
            
            print(f"Создано {len(created_rooms)} комнат для ученика {student_name} с ID: {conference_id}")
            return True
        except Exception as e:
            print(f"❌ Ошибка создания комнат для ученика: {e}")
            return False
    
    # =============================================================================
    # СИСТЕМНАЯ ДИАГНОСТИКА
    # =============================================================================
    
    def get_system_status(self) -> Dict[str, Any]:
        """Диагностика состояния системы"""
        try:
            import psutil
            import sys
            
            status = {
                "total_rooms": len(self.room_participants),
                "active_rooms": sum(1 for p in self.room_participants.values() if len(p) > 0),
                "dialogue_managers": sum(1 for d in self.room_dialogue.values() if d is not None),
                "thread_count": threading.active_count(),
                "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024 if 'psutil' in sys.modules else 0,
                "room_details": []
            }
            
            for room_id in list(self.room_participants.keys())[:10]:
                status["room_details"].append({
                    "room_id": room_id,
                    "participants": len(self.room_participants.get(room_id, [])),
                    "has_dialogue": room_id in self.room_dialogue and self.room_dialogue[room_id] is not None,
                    "ai_activated": self.room_ai_activated.get(room_id, False),
                    "last_activity": self.room_last_activity.get(room_id, 0)
                })
            
            return {
                "success": True,
                "status": status,
                "timestamp": time.time()
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def fix_blocked_rooms(self) -> Dict[str, Any]:
        """Ручное исправление заблокированных комнат"""
        try:
            fixed_count = 0
            
            for room_id in list(self.room_dialogue.keys()):
                if self.room_dialogue[room_id] is not None:
                    dialogue = self.room_dialogue[room_id]
                    
                    dialogue.waiting_for_answer = False
                    dialogue.lesson_started = False
                    dialogue.practice_active = False
                    
                    self.room_practice_active[room_id] = False
                    self.room_teacher_speaking[room_id] = False
                    self.room_speaking[room_id] = False
                    
                    fixed_count += 1
                    print(f"Исправлена комната {room_id}")
            
            for room_id in list(self.room_llm_pending_requests.keys()):
                if len(self.room_llm_pending_requests[room_id]) > 10:
                    requests = list(self.room_llm_pending_requests[room_id].items())
                    requests.sort(key=lambda x: x[1]['timestamp'], reverse=True)
                    self.room_llm_pending_requests[room_id] = dict(requests[:10])
            
            return {
                "success": True,
                "message": f"Исправлено {fixed_count} комнат",
                "fixed_count": fixed_count
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_room_debug_info(self, room_id: str) -> Dict[str, Any]:
        """Получение отладочной информации о комнате"""
        info = {
            "success": True,
            "room_id": room_id,
            "student_data": self.room_student_data.get(room_id, {}),
            "has_student_data": room_id in self.room_student_data,
            "subject": self.room_student_data.get(room_id, {}).get('subject') if room_id in self.room_student_data else None,
            "dialogue_exists": room_id in self.room_dialogue and self.room_dialogue[room_id] is not None,
            "dialogue_has_student_data": self.room_dialogue[room_id].has_student_data if room_id in self.room_dialogue and self.room_dialogue[room_id] else False,
            "dialogue_current_subject": self.room_dialogue[room_id].current_subject if room_id in self.room_dialogue and self.room_dialogue[room_id] else None,
            "participants": list(self.room_participants.get(room_id, [])),
            "ai_activated": self.room_ai_activated.get(room_id, False),
            "practice_active": self.room_practice_active.get(room_id, False),
            "teacher_speaking": self.room_teacher_speaking.get(room_id, False),
            "speaking": self.room_speaking.get(room_id, False)
        }
        
        return info

# Глобальный экземпляр RoomManager
_room_manager_instance = None

def get_room_manager(socketio=None):
    """Получение глобального экземпляра RoomManager"""
    global _room_manager_instance
    if _room_manager_instance is None and socketio is not None:
        _room_manager_instance = RoomManager(socketio)
    return _room_manager_instance
