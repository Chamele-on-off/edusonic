from flask import Flask, render_template, send_from_directory, jsonify, request
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

app = Flask(__name__, static_folder='static')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

BASE_DIR = Path(__file__).parent
FRAMES_DIR = BASE_DIR / 'static' / 'avatar' / 'frames'

animation_running = defaultdict(bool)
current_animation_frames = defaultdict(list)
current_frame_index = defaultdict(int)
room_participants = defaultdict(set)
room_speech_data = defaultdict(list)
active_speakers = defaultdict(dict)

# Фонемы для анимации губ (обновленные по вашим файлам)
PHONEME_MAP = {
    'aa': ['а', 'я', 'да', 'ма'],  # Открытый рот для гласных
    'bb': ['б', 'п'],              # Губные звуки
    'ch': ['ч', 'щ'],              # Переднеязычные
    'dd': ['д', 'т'],              # Зубные
    'ee': ['е', 'э'],              # Средний открытый
    'ff': ['ф', 'в'],              # Губно-зубные
    'hh': ['х'],                   # Гортанные
    'kk': ['к', 'г'],              # Заднеязычные
    'll': ['л'],                   # Боковые
    'mm': ['м'],                   # Губные носовые
    'nn': ['н'],                   # Зубные носовые
    'oo': ['о', 'ё'],              # Огубленные гласные
    'pp': ['п', 'б'],              # Губные взрывные
    'rr': ['р'],                   # Дрожащие
    'ss': ['с', 'з'],              # Свистящие
    'sh': ['ш', 'ж'],              # Шипящие
    'tt': ['т', 'д'],              # Зубные взрывные
    'uu': ['у', 'ю'],              # Огубленные гласные
    'vv': ['в', 'ф'],              # Губно-зубные
    'zh': ['ж', 'ш'],              # Шипящие
    'zz': ['з', 'с'],              # Звонкие свистящие
    'neutral': [' ']               # Нейтральное положение
}

# Веса для разных вариантов фонем
PHONEME_WEIGHTS = {
    'aa': [0.3, 0.3, 0.2, 0.2],
    'bb': [0.25, 0.25, 0.2, 0.2, 0.1],
    'ch': [0.25, 0.25, 0.2, 0.2, 0.1],
    'ee': [0.25, 0.25, 0.2, 0.2, 0.1],
    'ss': [0.25, 0.25, 0.2, 0.2, 0.1],
    'zh': [0.25, 0.25, 0.2, 0.2, 0.1],
    'zz': [0.25, 0.25, 0.2, 0.2, 0.1],
    'default': [0.5, 0.3, 0.2]  # Для фонем с 3 вариантами
}

def get_phoneme_for_letter(letter):
    letter = letter.lower()
    for phoneme, letters in PHONEME_MAP.items():
        if letter in letters:
            return phoneme
    return 'neutral'

def get_random_frame_variant(avatar_name, phoneme):
    """Выбирает случайный вариант кадра для фонемы с учетом весов"""
    avatar_dir = FRAMES_DIR / avatar_name
    if not avatar_dir.exists():
        return None
    
    # Получаем все кадры для данной фонемы
    frames = [f for f in os.listdir(avatar_dir) if f.startswith(f'mouth_{phoneme}_')]
    
    if not frames:
        return None
    
    # Получаем веса для вариантов
    weights = PHONEME_WEIGHTS.get(phoneme, PHONEME_WEIGHTS['default'])
    weights = weights[:len(frames)]  # Обрезаем веса по количеству доступных кадров
    weights = [w/sum(weights) for w in weights]  # Нормализуем
    
    # Выбираем случайный кадр с учетом весов
    selected_frame = random.choices(frames, weights=weights, k=1)[0]
    return f'/frames/{avatar_name}/{selected_frame}'

@app.route('/')
def home():
    return render_template('teacher.html')

@app.route('/conference')
def conference():
    room_id = request.args.get('room', 'default')
    embed = request.args.get('embed', 'false') == 'true'
    return render_template('conference.html', room_id=room_id, embed=embed)

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
        
        frames = [f for f in os.listdir(avatar_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        return jsonify({"frames": frames})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/frames/<avatar_name>/<path:filename>')
def serve_frame(avatar_name, filename):
    return send_from_directory(FRAMES_DIR / avatar_name, filename)

def text_to_speech(text, lang='ru'):
    tts = gTTS(text=text, lang=lang)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return base64.b64encode(mp3_fp.read()).decode('utf-8')

def lip_sync_animation(room_id, avatar_name, text):
    """Создает анимацию губ для произносимого текста"""
    avatar_dir = FRAMES_DIR / avatar_name
    if not avatar_dir.exists():
        return
    
    # Создаем последовательность кадров для анимации губ
    lip_frames = []
    for char in text.lower():
        phoneme = get_phoneme_for_letter(char)
        frame = get_random_frame_variant(avatar_name, phoneme)
        if frame:
            lip_frames.append(frame)
        else:
            # Если не нашли подходящий кадр, используем нейтральный
            neutral_frame = get_random_frame_variant(avatar_name, 'neutral')
            if neutral_frame:
                lip_frames.append(neutral_frame)
    
    if lip_frames:
        # Добавляем несколько повторов для плавности
        lip_frames = lip_frames * 3
        current_animation_frames[room_id] = lip_frames
        current_frame_index[room_id] = 0

def blink_animation(room_id, avatar_name):
    """Анимация моргания"""
    avatar_dir = FRAMES_DIR / avatar_name
    if not avatar_dir.exists():
        return
    
    blink_frames = [f for f in os.listdir(avatar_dir) if f.startswith('blink_')]
    if blink_frames:
        blink_frames.sort()
        blink_frames = [f'/frames/{avatar_name}/{f}' for f in blink_frames]
        current_animation_frames[room_id] = blink_frames
        current_frame_index[room_id] = 0

def animation_loop(room_id, avatar_name):
    """Основной цикл анимации"""
    last_blink_time = time.time()
    blink_interval = random.uniform(3.0, 6.0)  # Случайный интервал между морганиями
    
    while animation_running[room_id]:
        # Проверяем, нужно ли моргнуть
        if time.time() - last_blink_time > blink_interval:
            blink_animation(room_id, avatar_name)
            last_blink_time = time.time()
            blink_interval = random.uniform(3.0, 6.0)
        
        if current_animation_frames[room_id]:
            current_frame_index[room_id] = (current_frame_index[room_id] + 1) % len(current_animation_frames[room_id])
            frame_data = {
                'frame': current_animation_frames[room_id][current_frame_index[room_id]],
                'index': current_frame_index[room_id],
                'total': len(current_animation_frames[room_id])
            }
            socketio.emit('animation_frame', frame_data, room=room_id)
        
        time.sleep(0.1)

@socketio.on('connect')
def handle_connect():
    print('Client connected:', request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    for room_id in list(room_participants.keys()):
        if request.sid in room_participants[room_id]:
            room_participants[room_id].remove(request.sid)
            emit('participant_left', {'sid': request.sid}, room=room_id)
            emit('participants_update', {'count': len(room_participants[room_id])}, room=room_id)
            
            if request.sid in active_speakers[room_id]:
                del active_speakers[room_id][request.sid]
                emit('active_speakers_update', {'speakers': list(active_speakers[room_id].values())}, room=room_id)

@socketio.on('join_room')
def handle_join_room(data):
    room_id = data['room_id']
    join_room(room_id)
    room_participants[room_id].add(request.sid)
    emit('participants_update', {'count': len(room_participants[room_id])}, room=room_id)
    emit('new_participant', {'sid': request.sid}, room=room_id)
    
    if room_speech_data[room_id]:
        emit('speech_history', {'history': room_speech_data[room_id]}, to=request.sid)

@socketio.on('start_animation')
def handle_start_animation(data):
    room_id = data['room_id']
    avatar_name = data.get('avatar_name', '')
    if not animation_running[room_id]:
        current_animation_frames[room_id] = data['frames']
        animation_running[room_id] = True
        threading.Thread(target=animation_loop, args=(room_id, avatar_name)).start()

@socketio.on('stop_animation')
def handle_stop_animation(data):
    room_id = data['room_id']
    animation_running[room_id] = False

@socketio.on('generate_speech')
def handle_generate_speech(data):
    room_id = data['room_id']
    text = data['text']
    avatar_name = data.get('avatar_name', '')
    
    audio_data = text_to_speech(text)
    lip_sync_animation(room_id, avatar_name, text)
    
    emit('speech_audio', {
        'audio': audio_data, 
        'text': text,
        'type': 'generated'
    }, room=room_id)
    
    room_speech_data[room_id].append({
        'text': text,
        'timestamp': time.time(),
        'type': 'generated'
    })
    if len(room_speech_data[room_id]) > 50:
        room_speech_data[room_id].pop(0)

@socketio.on('recognized_speech')
def handle_recognized_speech(data):
    room_id = data['room_id']
    text = data['text']
    is_human = data.get('is_human', False)
    
    if is_human:
        active_speakers[room_id][request.sid] = {
            'sid': request.sid,
            'name': f'Участник {len(active_speakers[room_id]) + 1}',
            'timestamp': time.time()
        }
        emit('active_speakers_update', {
            'speakers': list(active_speakers[room_id].values())
        }, room=room_id)
    
    emit('speech_text', {
        'text': text, 
        'sid': request.sid,
        'is_human': is_human
    }, room=room_id)
    
    room_speech_data[room_id].append({
        'text': text,
        'timestamp': time.time(),
        'type': 'recognized',
        'sid': request.sid,
        'is_human': is_human
    })
    if len(room_speech_data[room_id]) > 50:
        room_speech_data[room_id].pop(0)

@socketio.on('stream_audio')
def handle_stream_audio(data):
    room_id = data['room_id']
    emit('stream_audio', {
        'audio': data['audio'],
        'sid': request.sid
    }, room=room_id, include_self=False)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)