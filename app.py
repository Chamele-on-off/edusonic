# app.py - AI Teacher System с поддержкой технических и естественнонаучных предметов
# ОПТИМИЗИРОВАННАЯ ВЕРСИЯ С ФОРМУЛАМИ И ТЕХНИЧЕСКОЙ ПРАКТИКОЙ

from flask import Flask, render_template, send_from_directory, jsonify, request, send_file, session, redirect, url_for
import os
from pathlib import Path
from flask_socketio import SocketIO, emit, join_room, leave_room
from gtts import gTTS
import io
import base64
import time
import threading
import json
from datetime import datetime
import re
import uuid
import random

# Импорт модулей из новой структуры
from auth import login_required, teacher_required, student_required, register_auth_routes
from room_manager import get_room_manager
from lesson_manager import get_lesson_manager
from student_manager import get_student_manager
from config import update_api_key, get_api_key, load_config, get_model_config, get_llm_mode, set_llm_mode, get_llm_priority, set_llm_priority
from local_llm_manager import get_llm_manager
from key_manager import get_key_manager
from dialogue import DialogueManager

# Настройка Flask и SocketIO
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

# Инициализация менеджеров
room_manager = get_room_manager(socketio)
lesson_manager = get_lesson_manager()
student_manager = get_student_manager()
llm_manager = get_llm_manager()
key_manager = get_key_manager()

# Регистрация маршрутов аутентификации
app = register_auth_routes(app)

BASE_DIR = Path(__file__).parent
FRAMES_DIR = BASE_DIR / 'static' / 'avatar' / 'frames'

# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def debug_log(message):
    """Логирование для отладки"""
    DEBUG_LLM = True
    if DEBUG_LLM:
        print(f"🔧 [DEBUG] {message}")

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

def is_foreign_text(text):
    """🔥 СУПЕР-БЫСТРАЯ проверка на иностранный текст"""
    if not text:
        return False
    
    # Быстрая проверка первых 200 символов
    sample = text[:200]
    
    # 1. Ищем латинские буквы
    if not re.search(r'[a-zA-Z]', sample):
        return False  # Нет латинских букв - точно не иностранный
    
    # 2. Игнорируем общепринятые английские слова в русском контексте
    common_english_in_russian = ['ok', 'hello', 'hi', 'yes', 'no', 'bye', 'sorry', 'please', 'thank you', 'okay']
    words = re.findall(r'\b[a-zA-Z]{2,}\b', sample.lower())
    
    if not words:
        return False
    
    # Подсчитываем уникальные НЕ общепринятые слова
    foreign_words = [w for w in words if w not in common_english_in_russian]
    unique_foreign_words = set(foreign_words)
    
    # Если есть хотя бы 2 уникальных НЕ общепринятых английских слова
    if len(unique_foreign_words) >= 2:
        return True
    
    # Или если есть длинное иностранное слово (более 4 букв)
    long_foreign_words = [w for w in foreign_words if len(w) > 4]
    if len(long_foreign_words) > 0:
        return True
    
    return False

def text_to_speech(text, lang='ru'):
    """🔥 ОПТИМИЗИРОВАННАЯ ВЕРСИЯ: Быстрое озвучивание с опциональным автоопределением"""
    try:
        if not text.strip():
            return None
            
        # 🔥 ОЧИСТКА ФОРМУЛ ПЕРЕД ОЗВУЧИВАНИЕМ
        # Удаляем LaTeX формулы перед синтезом речи
        text_for_speech = re.sub(r'\$\$.*?\$\$', 'формула', text)
        text_for_speech = re.sub(r'\\\(.*?\\\)', 'формула', text_for_speech)
        
        # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Автоопределение ТОЛЬКО если явно запрошено
        if lang == 'auto':
            # Быстрая проверка - есть ли латинские буквы?
            has_latin = bool(re.search(r'[a-zA-Z]', text_for_speech))
            has_cyrillic = bool(re.search(r'[а-яА-ЯёЁ]', text_for_speech))
            
            # Если ЕСТЬ латиница И кириллица - смешанный режим
            if has_latin and has_cyrillic:
                return text_to_speech_mixed(text_for_speech)
            # Если ТОЛЬКО латиница - английский
            elif has_latin and not has_cyrillic:
                lang = 'en'
            # По умолчанию - русский (для ТОЛЬКО кириллицы или цифр/знаков)
            else:
                lang = 'ru'
        
        # 🔥 БЫСТРАЯ ГЕНЕРАЦИЯ БЕЗ ДОПОЛНИТЕЛЬНОГО АНАЛИЗА
        tts = gTTS(text=text_for_speech, lang=lang, slow=False, lang_check=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return base64.b64encode(mp3_fp.read()).decode('utf-8')
        
    except Exception as e:
        debug_log(f"Error in text_to_speech: {e}")
        # Fallback - используем русский язык
        try:
            tts = gTTS(text=text, lang='ru', slow=False, lang_check=False)
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            return base64.b64encode(mp3_fp.read()).decode('utf-8')
        except Exception as e2:
            debug_log(f"Fallback error in text_to_speech: {e2}")
            return None

def text_to_speech_mixed(text):
    """🔥 ОПТИМИЗИРОВАННАЯ ФУНКЦИЯ: Озвучивание смешанного текста"""
    try:
        # 🔥 ОПТИМИЗИРОВАННЫЙ ШАБЛОН: ищем последовательности букв одного языка
        # Разбиваем по границам языка (кириллица/латиница)
        pattern = r'([а-яА-ЯёЁ][а-яА-ЯёЁ\s.,!?;:\'-]*|[a-zA-Z][a-zA-Z\s.,!?;:\'-]*)'
        fragments = re.findall(pattern, text)
        
        if not fragments:
            return text_to_speech(text, 'ru')  # Fallback
        
        audio_chunks = []
        
        for fragment in fragments:
            fragment = fragment.strip()
            if not fragment:
                continue
                
            # Определяем язык фрагмента - быстрая проверка
            has_cyrillic = bool(re.search(r'[а-яА-ЯёЁ]', fragment))
            lang = 'ru' if has_cyrillic else 'en'
            
            try:
                tts = gTTS(text=fragment, lang=lang, slow=False, lang_check=False)
                chunk_fp = io.BytesIO()
                tts.write_to_fp(chunk_fp)
                chunk_fp.seek(0)
                audio_chunks.append(chunk_fp.read())
                
                # 🔥 КОРОТКАЯ ПАУЗА МЕЖДУ ФРАГМЕНТАМИ (50ms silence)
                silence = bytes([0] * 800)  # 50ms при 16kHz
                audio_chunks.append(silence)
                
            except Exception as e:
                debug_log(f"Error processing fragment '{fragment[:30]}...': {e}")
                continue
        
        if audio_chunks:
            # 🔥 УБИРАЕМ ПОСЛЕДНЮЮ ПАУЗУ
            if audio_chunks[-1] == bytes([0] * 800):
                audio_chunks.pop()
                
            combined_audio = b''.join(audio_chunks)
            return base64.b64encode(combined_audio).decode('utf-8')
        
        return None
    except Exception as e:
        debug_log(f"Error in text_to_speech_mixed: {e}")
        return text_to_speech(text, 'ru')

def speak_text(room_id, text, voice_type='female', is_teacher=False, skip_history=False, force_lang=None):
    """🔥 ОПТИМИЗИРОВАННАЯ: Озвучивает текст с оптимизацией производительности"""
    if not text.strip():
        return
        
    cleaned_text = clean_text_for_speech(text)
    
    if not cleaned_text.strip():
        return
        
    # Обновляем состояния в room_manager
    if is_teacher:
        room_manager.room_teacher_speaking[room_id] = True
        
    room_manager.room_speaking[room_id] = True
    socketio.emit('speaking_state', {'speaking': True}, room=room_id)
    
    # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ:
    # 1. Если force_lang указан - используем его (самый быстрый путь)
    # 2. Иначе определяем только если есть латинские буквы
    # 3. По умолчанию - русский (самый частый случай)
    
    audio_data = None
    
    if force_lang:
        # 🔥 ЯВНО УКАЗАН ЯЗЫК - САМЫЙ БЫСТРЫЙ ПУТЬ
        audio_data = text_to_speech(cleaned_text, lang=force_lang)
    else:
        # 🔥 БЫСТРАЯ ЭВРИСТИКА: проверяем только на латинские буквы
        # Это ДЕШЕВО - просто поиск по регулярке
        has_latin = bool(re.search(r'[a-zA-Z]', cleaned_text))
        
        if not has_latin:
            # 🔥 ТОЛЬКО РУССКИЙ ТЕКСТ - БЫСТРЫЙ ПУТЬ
            audio_data = text_to_speech(cleaned_text, lang='ru')
        else:
            # Есть латинские буквы - нужна более точная проверка
            if is_foreign_text(cleaned_text):
                # 🔥 ЕСТЬ ИНОСТРАННЫЙ ТЕКСТ - автоопределение
                audio_data = text_to_speech(cleaned_text, lang='auto')
            else:
                # 🔥 ТОЛЬКО ОБЩЕПРИНЯТЫЕ СЛОВА - РУССКИЙ
                audio_data = text_to_speech(cleaned_text, lang='ru')
    
    if audio_data:
        emit('speech_audio', {
            'audio': audio_data,
            'text': cleaned_text,
            'timestamp': time.time(),
            'voice_type': voice_type,
            'is_teacher': is_teacher,
            'optimized': True  # 🔥 Флаг что использовалась оптимизация
        }, room=room_id)
        
        if not skip_history:
            room_manager.add_speech_data(room_id, {
                'text': cleaned_text,
                'timestamp': time.time(),
                'type': 'generated',
                'voice_type': voice_type,
                'is_teacher': is_teacher
            })
    
    # 🔥 Более точная длительность речи
    speech_duration = max(1.5, len(cleaned_text) * 0.08)  # Уменьшили коэффициент
    threading.Timer(speech_duration, lambda: room_manager.reset_speaking_state(room_id, is_teacher)).start()

# =============================================================================
# ОСНОВНЫЕ МАРШРУТЫ
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

@app.route('/teacher')
@teacher_required
def teacher():
    from auth import load_user_data
    user_data = load_user_data(session['user_id'])
    return render_template('teacher.html', user=user_data)

@app.route('/student')
@student_required
def student():
    from auth import load_user_data
    user_data = load_user_data(session['user_id'])
    
    if not user_data.get('profile_complete', False):
        return render_template('student_profile.html', user=user_data)
    
    student_data = user_data.get('student_data', {})
    return render_template('student.html', user=user_data, student_data=student_data)

@app.route('/student_profile')
@student_required
def student_profile():
    from auth import load_user_data
    user_data = load_user_data(session['user_id'])
    return render_template('student_profile.html', user=user_data)

@app.route('/investing.html')
def investing():
    return render_template('investing.html')

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

# =============================================================================
# API МАРШРУТЫ ДЛЯ АВАТАРОВ И ФРЕЙМОВ
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

# =============================================================================
# API МАРШРУТЫ ДЛЯ УРОКОВ
# =============================================================================

@app.route('/api/lessons')
def get_available_lessons_api():
    result = lesson_manager.get_available_lessons()
    if "error" in result:
        return jsonify({"error": result["error"]}), 500
    return jsonify(result)

@app.route('/api/lesson_content/<lesson_id>')
def get_lesson_content_api(lesson_id):
    result = lesson_manager.get_lesson_content(lesson_id)
    if "error" in result:
        return jsonify({"error": result["error"]}), 404
    return jsonify(result)

@app.route('/api/practice_content/<lesson_id>')
def get_practice_content_api(lesson_id):
    result = lesson_manager.get_practice_content(lesson_id)
    if "error" in result:
        return jsonify({"error": result["error"], "success": False}), 404
    return jsonify(result)

@app.route('/api/practice_files')
def get_practice_files_api():
    result = lesson_manager.get_practice_files()
    if "error" in result:
        return jsonify({"error": result["error"]}), 500
    return jsonify(result)

@app.route('/api/upload_practice', methods=['POST'])
@teacher_required
def upload_practice_api():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"})
        
        file = request.files['file']
        result = lesson_manager.upload_practice(file)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/delete_practice/<filename>')
@teacher_required
def delete_practice_api(filename):
    result = lesson_manager.delete_practice(filename)
    return jsonify(result)

# =============================================================================
# API МАРШРУТЫ ДЛЯ КОНФИГУРАЦИИ И LLM
# =============================================================================

@app.route('/api/config/keys', methods=['GET'])
@teacher_required
def get_api_keys_api():
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
@teacher_required
def set_api_key_api():
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
@teacher_required
def test_api_key_api():
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
        
        import requests
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
@teacher_required
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
            # Обновляем режим во всех активных комнатах
            for room_id in room_manager.room_llm_mode:
                room_manager.room_llm_mode[room_id] = mode
                if room_id in room_manager.room_dialogue and room_manager.room_dialogue[room_id]:
                    room_manager.room_dialogue[room_id].set_llm_mode(mode)
            
            return jsonify({
                "success": True,
                "message": f"Режим LLМ успешно изменен на '{mode}'",
                "mode": mode
            })
        else:
            return jsonify({"success": False, "error": "Failed to save config"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/llm/priority', methods=['GET'])
def get_llm_priority_api():
    try:
        priority = get_llm_priority()
        return jsonify({
            "success": True,
            "priority": priority
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/llm/priority', methods=['POST'])
@teacher_required
def set_llm_priority_api():
    try:
        data = request.json
        priority = data.get('priority')
        
        if not priority:
            return jsonify({"success": False, "error": "Priority not specified"})
        
        success = set_llm_priority(priority)
        
        if success:
            for room_id in room_manager.room_dialogue:
                room_manager.room_dialogue[room_id].llm.set_priority(priority)
            
            return jsonify({
                "success": True,
                "message": f"Приоритет успешно изменен на '{priority}'",
                "priority": priority
            })
        else:
            return jsonify({"success": False, "error": "Failed to save priority"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/llm/available_priorities')
def get_available_priorities_api():
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
def get_llm_status_api():
    room_id = request.args.get('room_id', 'default')
    
    if room_id in room_manager.room_dialogue:
        status = room_manager.room_dialogue[room_id].llm.get_llm_status()
        return jsonify({
            "success": True,
            "room": room_id,
            "status": status
        })
    
    return jsonify({"success": False, "error": "Room not found"})

@app.route('/api/llm/local_status')
def get_local_llm_status_api():
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
def get_llm_manager_status_api():
    try:
        status = llm_manager.get_status()
        return jsonify({
            "success": True,
            "status": status
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/llm/poll_response', methods=['POST'])
def poll_llm_response_api():
    try:
        data = request.json
        room_id = data.get('room_id', 'default')
        last_check = data.get('last_check', 0)
        request_id_filter = data.get('request_id')
        
        result = room_manager.poll_llm_responses(room_id, last_check, request_id_filter)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/llm/clear_queue', methods=['POST'])
def clear_llm_queue_api():
    try:
        data = request.json
        room_id = data.get('room_id', 'default')
        
        result = room_manager.clear_llm_queue(room_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# =============================================================================
# API МАРШРУТЫ ДЛЯ УЧЕНИКОВ
# =============================================================================

@app.route('/api/student/save', methods=['POST'])
@student_required
def save_student_api():
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
        
        from auth import load_user_data, save_user_data
        user_data = load_user_data(user_id)
        if not user_data:
            return jsonify({"success": False, "error": "Пользователь не найден"})
        
        user_data['student_data'] = student_data
        user_data['profile_complete'] = True
        user_data['profile_updated'] = datetime.now().isoformat()
        
        if save_user_data(user_data):
            # Инициализируем прогресс ученика
            lesson_manager.initialize_student_progress(student_data['student_id'], student_data['education_level'])
            
            return jsonify({
                "success": True,
                "message": "Профиль успешно сохранен",
                "student_id": student_data['student_id']
            })
        else:
            return jsonify({"success": False, "error": "Ошибка сохранения профиля"})
    except Exception as e:
        return jsonify({"success": False, "error": f"Ошибка: {str(e)}"})

@app.route('/api/student/<student_id>')
@teacher_required
def get_student_api(student_id):
    student_data = student_manager.load_student_data(student_id)
    if student_data:
        return jsonify({"success": True, "student": student_data})
    else:
        return jsonify({"success": False, "error": "Ученик не найден"})

@app.route('/api/student/<student_id>/rooms')
@teacher_required
def get_student_rooms_api(student_id):
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
            "default_avatar": student_data.get('default_avatar', 'woman'),
            "total_rooms": len(rooms)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/student/<student_id>/room/<subject>')
@teacher_required
def get_student_room_api(student_id, subject):
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

@app.route('/api/student/create-conference', methods=['POST'])
@student_required  
def create_student_conference_api():
    try:
        data = request.json
        subject = data.get('subject')
        
        if not subject:
            return jsonify({"success": False, "error": "Не указан предмет"})
        
        from auth import load_user_data
        user_data = load_user_data(session['user_id'])
        student_data = user_data.get('student_data', {})
        
        conference = room_manager.create_student_conference(student_data, subject)
        
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

@app.route('/api/student/lessons-by-class', methods=['GET'])
@student_required
def get_lessons_by_class_api():
    try:
        from auth import load_user_data
        user_data = load_user_data(session['user_id'])
        if not user_data or not user_data.get('student_data'):
            return jsonify({"success": False, "error": "Данные ученика не найдены"})
        
        student_class = user_data['student_data'].get('education_level', '5')
        
        lessons_by_subject = lesson_manager.get_student_lessons_by_class(student_class)
        
        student_id = user_data['student_data'].get('student_id')
        progress_data = {}
        if student_id:
            from pathlib import Path
            STUDENT_PROGRESS_DIR = Path(__file__).parent / "students_progress"
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

@app.route('/api/student/progress', methods=['GET'])
@student_required
def get_student_progress_api():
    try:
        from auth import load_user_data
        user_data = load_user_data(session['user_id'])
        if not user_data or not user_data.get('student_data'):
            return jsonify({"success": False, "error": "Данные ученика не найдены"})
        
        student_id = user_data['student_data'].get('student_id')
        result = lesson_manager.get_student_progress(student_id)
        
        if result.get("success"):
            result["progress"]["student_name"] = user_data['student_data'].get('name', '')
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# =============================================================================
# API МАРШРУТЫ ДЛЯ ТЕХНИЧЕСКИХ ПРЕДМЕТОВ
# =============================================================================

@app.route('/api/technical/detect-subject', methods=['POST'])
def detect_technical_subject_api():
    """Определяет, является ли предмет техническим"""
    try:
        data = request.json
        subject = data.get('subject', '')
        
        if not subject:
            return jsonify({"success": False, "error": "Предмет не указан"})
        
        tech_info = lesson_manager.detect_technical_subject(subject)
        
        return jsonify({
            "success": True,
            "subject": subject,
            "is_technical": tech_info["is_technical"],
            "is_science": tech_info["is_science"],
            "subject_type": tech_info["subject_type"],
            "requires_formulas": tech_info["requires_formulas"],
            "requires_diagrams": tech_info["requires_diagrams"]
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/technical/extract-formulas', methods=['POST'])
def extract_formulas_api():
    """Извлекает формулы из текста"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({"success": False, "error": "Текст не указан"})
        
        formulas = lesson_manager.extract_formulas_from_text(text)
        
        return jsonify({
            "success": True,
            "text_preview": text[:200] + '...' if len(text) > 200 else text,
            "formulas": formulas,
            "formula_count": len(formulas),
            "has_latex": any(f['type'] == 'latex' for f in formulas),
            "has_inline": any(f['type'] == 'inline' for f in formulas)
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/technical/sample-exercises', methods=['GET'])
def get_sample_technical_exercises_api():
    """Получает примерные технические упражнения"""
    try:
        subject = request.args.get('subject', 'математика')
        result = lesson_manager.get_sample_technical_exercises(subject)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# =============================================================================
# API МАРШРУТЫ ДЛЯ УПРАВЛЕНИЯ КЛЮЧАМИ OPENROUTER
# =============================================================================

@app.route('/api/keys/status', methods=['GET'])
@teacher_required
def get_keys_status_api():
    try:
        stats = key_manager.get_usage_stats()
        return jsonify({
            "success": True,
            "stats": stats
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/keys/add', methods=['POST'])
@teacher_required
def add_api_key_api():
    try:
        data = request.json
        api_key = data.get('api_key')
        name = data.get('name', 'new_key')
        limit_type = data.get('limit_type', 'standard')
        
        if not api_key:
            return jsonify({"success": False, "error": "API key is required"})
        
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

@app.route('/api/keys/force_reset', methods=['POST'])
@teacher_required
def force_reset_keys_api():
    try:
        key_manager.force_reset_all()
        
        return jsonify({
            "success": True,
            "message": "Все счетчики ключей сброшены"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# =============================================================================
# API МАРШРУТЫ ДЛЯ УПРАВЛЕНИЯ УРОКАМИ
# =============================================================================

@app.route('/api/lessons/structure', methods=['GET'])
@teacher_required
def get_lessons_structure_api():
    result = lesson_manager.get_lessons_structure()
    return jsonify(result)

@app.route('/api/lessons/create-sample', methods=['POST'])
@teacher_required
def create_sample_lessons_api():
    result = lesson_manager.create_sample_lessons()
    return jsonify(result)

@app.route('/api/add_demo_lesson', methods=['POST'])
@teacher_required
def add_demo_lesson_api():
    try:
        data = request.json
        title = data.get('title', '')
        content = data.get('content', '')
        subject = data.get('subject', 'общее')
        
        result = lesson_manager.add_demo_lesson(title, content, subject)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/lessons/list', methods=['GET'])
@teacher_required
def get_all_lessons_list_api():
    try:
        class_filter = request.args.get('class', 'all')
        subject_filter = request.args.get('subject', 'all')
        search_query = request.args.get('search', '')
        
        lessons = []
        
        from pathlib import Path
        BASE_DIR = Path(__file__).parent
        LESSONS_DIR = BASE_DIR / 'lessons'
        LESSONS_DEMO_DIR = LESSONS_DIR / "demo"
        LESSONS_STUDENTS_DIR = LESSONS_DIR / "students"
        LESSONS_GENERATED_DIR = LESSONS_DIR / "generated"
        
        if class_filter in ['all', 'demo']:
            for lesson_file in LESSONS_DEMO_DIR.glob("*.txt"):
                if search_query and search_query.lower() not in lesson_file.name.lower():
                    continue
                
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
        
        if class_filter in ['all', 'generated']:
            for lesson_file in LESSONS_GENERATED_DIR.glob("*.txt"):
                if search_query and search_query.lower() not in lesson_file.name.lower():
                    continue
                
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
        
        if class_filter == 'all' or class_filter.isdigit():
            for class_dir in LESSONS_STUDENTS_DIR.glob("*_class"):
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
                                    'full_path': str(lesson_file),
                                    'size': lesson_file.stat().st_size,
                                    'modified': datetime.fromtimestamp(lesson_file.stat().st_mtime).isoformat(),
                                    'can_edit': True,
                                    'can_delete': True
                                })
        
        lessons.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({
            "success": True,
            "total": len(lessons),
            "lessons": lessons
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/lesson/edit/<path:lesson_path>', methods=['GET'])
@teacher_required
def get_lesson_for_edit_api(lesson_path):
    result = lesson_manager.get_lesson_for_edit(lesson_path)
    return jsonify(result)

@app.route('/api/lesson/save', methods=['POST'])
@teacher_required
def save_edited_lesson_api():
    try:
        data = request.json
        lesson_path = data.get('lesson_path')
        content = data.get('content')
        
        result = lesson_manager.edit_lesson(lesson_path, content)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/lesson/delete', methods=['POST'])
@teacher_required
def delete_lesson_api():
    try:
        data = request.json
        lesson_path = data.get('lesson_path')
        
        result = lesson_manager.delete_lesson(lesson_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# =============================================================================
# SOCKET.IO ОБРАБОТЧИКИ
# =============================================================================

@socketio.on('connect')
def handle_connect():
    debug_log(f"Client connected: {request.sid}")
    emit('connection_established', {'message': 'Connected to server', 'sid': request.sid})

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    debug_log(f"Client disconnected: {sid}")
    room_manager.handle_disconnected_session(sid)

@socketio.on('join_room')
def handle_join_room(data):
    room_id = data['room_id']
    peer_id = data.get('peer_id')
    
    debug_log(f"Попытка присоединения к комнате {room_id}, peer_id: {peer_id}")
    
    try:
        result = room_manager.join_room(room_id, request.sid, peer_id)
        
        if not result['success']:
            emit('room_error', {
                'room_id': room_id,
                'error': result['error']
            }, to=request.sid)
            return
        
        join_room(room_id)
        
        if peer_id:
            emit('participant_joined', {
                'peer_id': peer_id,
                'sid': request.sid
            }, room=room_id, include_self=False)
        
        try:
            emit('current_avatar', {'avatar_name': room_manager.room_current_avatar[room_id]}, to=request.sid)
        except Exception as e:
            debug_log(f"⚠️ Ошибка отправки аватара: {e}")
        
        # Отправляем историю речи
        speech_history = room_manager.get_speech_history(room_id)
        if speech_history:
            try:
                emit('speech_history', {'history': speech_history}, to=request.sid)
            except Exception as e:
                debug_log(f"⚠️ Ошибка отправки истории: {e}")
        
        emit('participants_update', {'count': result['participants_count']}, room=room_id)
        
        # Приветствие для комнат учеников
        if (room_id in room_manager.room_student_data and 
            room_manager.room_student_data[room_id] and 
            not room_id.startswith('demo_') and 
            room_id != 'default'):
            
            student_data = room_manager.room_student_data[room_id]
            student_name = student_data.get('name', 'ученик')
            subject = student_data.get('subject', 'предмету')
            
            # 🔥 ПРОВЕРЯЕМ ТИП ПРЕДМЕТА
            tech_info = lesson_manager.detect_technical_subject(subject)
            
            if tech_info["is_technical"]:
                welcome_message = f"{student_name}, привет! Я твой виртуальный учитель по {subject}. "
                welcome_message += "Это технический предмет, поэтому будем работать с формулами и задачами. "
                welcome_message += "Если ты готов начать, скажи 'готов начать'."
            else:
                welcome_message = f"{student_name}, привет! Я твой виртуальный учитель по {subject}. "
                welcome_message += "Давай начнем наш сегодняшний урок. Если ты готов начать, скажи 'готов начать'."
            
            socketio.emit('student_welcome_message', {
                'room_id': room_id,
                'student_name': student_name,
                'subject': subject,
                'is_technical': tech_info["is_technical"],
                'is_science': tech_info["is_science"],
                'message': welcome_message,
                'prompt_ready': True
            }, room=room_id)
            
            threading.Thread(target=lambda: delayed_welcome(room_id, welcome_message)).start()
        
        elif result['participants_count'] == 1 and not room_manager.room_ai_activated[room_id]:
            greeting = "Привет! Я ваш виртуальный учитель. Давайте познакомимся и выберем интересный урок вместе!"
            threading.Thread(target=lambda: delayed_welcome(room_id, greeting)).start()
        
    except Exception as e:
        debug_log(f"❌ Ошибка при присоединении к комнате {room_id}: {e}")
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
    if room_id in room_manager.room_current_avatar:
        emit('current_avatar', {'avatar_name': room_manager.room_current_avatar[room_id]}, to=request.sid)

@socketio.on('client_start_animation')
def handle_client_start_animation(data):
    room_id = data['room_id']
    avatar_name = data['avatar_name']
    debug_log(f"Получена команда запуска анимации для комнаты {room_id}, аватар: {avatar_name}")
    
    room_manager.room_current_avatar[room_id] = avatar_name
    emit('avatar_changed', {'avatar_name': avatar_name}, room=room_id)
    emit('animation_ready', {'status': 'ready'}, room=room_id)

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
    debug_log(f"Состояние комнаты: practice_active={room_manager.room_practice_active.get(room_id, False)}, teacher_speaking={room_manager.room_teacher_speaking.get(room_id, False)}")

    if room_manager.room_teacher_speaking.get(room_id, False):
        debug_log(f"Игнорирую ответ ученика, так как учитель говорит: {answer}")
        return

    if not room_manager.room_practice_active.get(room_id, False):
        debug_log(f"Практика не активна, игнорирую ответ: {answer}")
        return

    if any(cmd in answer.lower() for cmd in ['стоп', 'останови', 'хватит', 'закончи']):
        debug_log(f"Команда остановки практики: {answer}")
        if room_id in room_manager.room_dialogue:
            room_manager.room_dialogue[room_id]._end_practice_session()
            room_manager.room_practice_active[room_id] = False
            room_manager.room_current_question_index[room_id] = 0
            
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

    if room_id in room_manager.room_dialogue:
        dialogue = room_manager.room_dialogue[room_id]
        if not dialogue.waiting_for_answer:
            debug_log(f"Система не ожидает ответа, игнорирую: {answer}")
            return

    if any(cmd in answer.lower() for cmd in ['продолжай', 'дальше', 'следующий']):
        debug_log(f"Игнорирую команду вместо ответа: {answer}")
        if room_id in room_manager.room_dialogue:
            response = room_manager.room_dialogue[room_id]._evaluate_and_generate_next("")
            if response:
                emit('speech_text', {
                    'text': f"Учитель: {response}",
                    'sid': 'teacher',
                    'is_teacher': True
                }, room=room_id)
                # 🔥 ОПТИМИЗИРОВАННЫЙ ВЫЗОВ
                speak_text(room_id, response, voice_type='female', is_teacher=True)
        return

    room_manager.add_speech_data(room_id, {
        'text': f"Ответ ученика: {answer}",
        'timestamp': time.time(),
        'type': 'practice_answer',
        'sid': user_sid
    })
    
    if room_id in room_manager.room_dialogue:
        debug_log(f"Обработка ответа через диалог менеджер...")
        
        response = room_manager.room_dialogue[room_id]._evaluate_and_generate_next(answer)
        
        if response:
            debug_log(f"Ответ учителя: {response}")
            
            emit('speech_text', {
                'text': f"Учитель: {response}",
                'sid': 'teacher',
                'is_teacher': True
            }, room=room_id)
            
            # 🔥 ОПТИМИЗИРОВАННЫЙ ВЫЗОВ
            speak_text(room_id, response, voice_type='female', is_teacher=True)
            
            if not room_manager.room_dialogue[room_id].practice_active:
                room_manager.room_practice_active[room_id] = False
                room_manager.room_current_question_index[room_id] = 0
                emit('practice_ended', {}, room=room_id)
                debug_log("Практика завершена")
        else:
            room_manager.room_practice_active[room_id] = False
            room_manager.room_current_question_index[room_id] = 0
            room_manager.room_dialogue[room_id].waiting_for_answer = False
            emit('practice_ended', {}, room=room_id)
            debug_log("Практика завершена (response=None)")

@socketio.on('student_message')
def handle_student_message(data):
    room_id = data['room_id']
    message = data['message']
    user_sid = request.sid

    debug_log(f"Получено сообщение от ученика: {message}")
    
    if room_manager.room_teacher_speaking.get(room_id, False):
        debug_log(f"Игнорирую сообщение ученика, так как учитель говорит: {message}")
        return

    if room_manager.room_practice_active.get(room_id, False):
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
    room_manager.room_last_activity[room_id] = time.time()
    
    if not room_manager.room_ai_activated.get(room_id, False):
        return
        
    if room_id not in room_manager.room_dialogue or room_manager.room_dialogue[room_id] is None:
        debug_log(f"DialogueManager отсутствует для комнаты {room_id}, пытаемся создать...")
        if not room_manager._fast_room_initialization(room_id):
            debug_log(f"Не удалось создать DialogueManager для {room_id}")
            return

    if room_manager.room_teacher_speaking.get(room_id, False):
        debug_log(f"Игнорирую речь ученика, так как учитель говорит: {text}")
        return

    if (text.startswith("Учитель:") or "учитель" in text.lower() or 
        len(text.strip()) < 3 or text in ["привет", "здравствуйте"]):
        return
    
    room_manager.add_speech_data(room_id, {
        'text': text,
        'timestamp': time.time(),
        'type': 'recognized',
        'sid': user_sid
    })
    
    emit('speech_text', {'text': text, 'sid': user_sid}, room=room_id)
    
    if room_manager.room_ai_activated[room_id]:
        dialogue = room_manager.room_dialogue[room_id]
        
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
        result = room_manager.activate_ai_teacher(room_id)
        
        if result['success']:
            greeting = "Привет! Я ваш AI-учитель. Давайте пообщаемся и выберем интересный урок вместе!"
            # 🔥 ОПТИМИЗИРОВАННЫЙ ВЫЗОВ
            speak_text(room_id, greeting, voice_type='female', is_teacher=True, force_lang='ru')
            
            emit('ai_teacher_activated', {
                'room_id': room_id,
                'message': 'AI-учитель успешно активирован'
            }, room=room_id)
            
            debug_log(f"AI-учитель успешно активирован в комнате {room_id}")
        else:
            emit('activate_ai_error', {
                'room_id': room_id,
                'error': result['error']
            }, to=sid)
        
    except Exception as e:
        debug_log(f"❌ Ошибка активации AI-учителя: {e}")
        emit('activate_ai_error', {
            'room_id': room_id,
            'error': f'Ошибка активации: {str(e)}'
        }, to=sid)

@socketio.on('set_llm_mode')
def handle_set_llm_mode(data):
    room_id = data['room_id']
    mode = data['mode']
    
    success = room_manager.set_llm_mode(room_id, mode)
    
    if success:
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
    
    room_manager.reset_speaking_state(room_id, is_teacher=True)
    
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
    room_manager.set_practice_active(room_id, True)
    emit('practice_started', {}, room=room_id)
    debug_log(f"Практика начата в комнате {room_id}")

@socketio.on('practice_ended')
def handle_practice_ended(data):
    room_id = data['room_id']
    room_manager.set_practice_active(room_id, False)
    emit('practice_ended', {}, room=room_id)
    debug_log(f"Практика завершена в комнате {room_id}")

@socketio.on('get_llm_status')
def handle_get_llm_status(data):
    room_id = data['room_id']
    
    if room_id in room_manager.room_dialogue:
        status = room_manager.room_dialogue[room_id].llm.get_llm_status()
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
    
    if room_id in room_manager.room_dialogue:
        room_manager.room_dialogue[room_id].llm.set_priority(priority)
        status = room_manager.room_dialogue[room_id].llm.get_priority_status()
        
        emit('llm_priority_changed', {
            'room_id': room_id,
            'priority': priority,
            'status': status
        }, room=room_id)
        
        debug_log(f"Приоритет LLM изменен в комнате {room_id}: {priority}")

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
    
    room_manager.add_pending_request(room_id, request_id, {
        'prompt': prompt,
        'system_prompt': system_prompt,
        'max_tokens': max_tokens,
        'timestamp': time.time(),
        'type': request_type
    })
    
    llm_request_id = llm_manager.submit_request(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        room_id=room_id,
        request_id=request_id
    )
    
    room_manager.room_llm_pending_requests[room_id][request_id]['manager_id'] = llm_request_id
    
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
    
    room_manager.remove_pending_request(room_id, request_id)
    
    if response and room_id in room_manager.room_dialogue:
        room_manager.room_dialogue[room_id].llm.handle_llm_response(request_id, response, room_id)
        
        emit('speech_text', {
            'text': f"Учитель: {response}",
            'sid': 'teacher',
            'is_teacher': True
        }, room=room_id)
        
        # 🔥 ОПТИМИЗИРОВАННЫЙ ВЫЗОВ
        speak_text(room_id, response, voice_type='female', is_teacher=True)

# =============================================================================
# ЗАПУСК СЕРВЕРА
# =============================================================================

if __name__ == '__main__':
    debug_log("🚀 Запуск AI Teacher системы с поддержкой технических предметов...")
    
    # Настройка менеджера LLM
    def global_llm_callback(request_id, response, room_id, original_request_id=None):
        """Глобальный обработчик ответов от LLM"""
        debug_log(f"Получен ответ для комнаты {room_id}: {response[:100]}...")
        
        target_request_id = original_request_id
        if not target_request_id:
            for req_id, req_data in room_manager.room_llm_pending_requests[room_id].items():
                if req_data.get('manager_id') == request_id:
                    target_request_id = req_id
                    break
        
        if not target_request_id:
            target_request_id = f"unknown_{int(time.time() * 1000)}"
        
        room_manager.add_llm_response(room_id, target_request_id, response, delivered_via_websocket=False)
        
        try:
            socketio.emit('llm_async_response', {
                'request_id': target_request_id,
                'response': response,
                'room_id': room_id,
                'timestamp': time.time(),
                'delivered_via': 'websocket'
            }, room=room_id)
            debug_log(f"Ответ немедленно отправлен через WebSocket в комнату {room_id}")
        except Exception as e:
            debug_log(f"⚠️ Не удалось отправить через WebSocket: {e}")
    
    llm_manager.start()
    llm_manager.register_room_callback('global', global_llm_callback)
    
    # Запускаем периодическую очистку
    room_manager.periodic_cleanup()
    
    # Инициализируем системные комнаты
    system_rooms = ['default', 'demo_room', 'test_room']
    for room in system_rooms:
        room_manager._fast_room_initialization(room)
    
    debug_log(f"✅ Система готова. Async mode: {socketio.async_mode}")
    debug_log(f"✅ Максимальное количество комнат в памяти: {room_manager.MAX_ROOMS}")
    debug_log(f"✅ Таймаут неактивных комнат: {room_manager.ROOM_TIMEOUT} секунд")
    debug_log(f"🔥 Добавлена поддержка технических предметов и формул")
    debug_log(f"🚀 ОПТИМИЗАЦИЯ: Включена быстрая проверка технических предметов")
    debug_log(f"📁 Модульная архитектура: auth.py, room_manager.py, lesson_manager.py, student_manager.py")
    
    socketio.run(
        app, 
        host='0.0.0.0', 
        port=5000, 
        debug=True, 
        allow_unsafe_werkzeug=True
    )