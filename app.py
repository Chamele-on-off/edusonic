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

app = Flask(__name__, static_folder='static')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

BASE_DIR = Path(__file__).parent
FRAMES_DIR = BASE_DIR / 'static' / 'avatar' / 'frames'

room_participants = defaultdict(set)
room_speech_data = defaultdict(list)
active_speeches = defaultdict(dict)

phoneme_mapping = {
    'aa': ['а', 'я'],
    'bb': ['б', 'п'],
    'ch': ['ч'],
    'dd': ['д', 'т'],
    'ee': ['е', 'э', 'и'],
    'ff': ['ф', 'в'],
    'hh': ['х', 'г'],
    'kk': ['к'],
    'll': ['л'],
    'nn': ['н'],
    'oo': ['о', 'ё'],
    'pp': ['п'],
    'rr': ['р'],
    'sh': ['ш', 'щ'],
    'ss': ['с', 'з'],
    'tt': ['т'],
    'uu': ['у', 'ю'],
    'vv': ['в'],
    'zh': ['ж'],
    'zz': ['з']
}

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

def get_phoneme_for_char(char):
    char_lower = char.lower()
    for phoneme, chars in phoneme_mapping.items():
        if char_lower in chars:
            return phoneme
    return None

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

@socketio.on('join_room')
def handle_join_room(data):
    room_id = data['room_id']
    join_room(room_id)
    room_participants[room_id].add(request.sid)
    emit('participants_update', {'count': len(room_participants[room_id])}, room=room_id)
    emit('new_participant', {'sid': request.sid}, room=room_id)
    
    if room_speech_data[room_id]:
        emit('speech_history', {'history': room_speech_data[room_id]}, to=request.sid)

@socketio.on('generate_speech')
def handle_generate_speech(data):
    room_id = data['room_id']
    text = data['text']
    lang = data.get('lang', 'ru')
    voice = data.get('voice', None)
    
    audio_data = text_to_speech(text, lang)
    speech_id = f"speech_{time.time()}"
    active_speeches[room_id][speech_id] = text
    
    emit('speech_audio', {
        'audio': audio_data,
        'text': text,
        'speech_id': speech_id,
        'phonemes': [get_phoneme_for_char(c) for c in text]
    }, room=room_id)
    
    room_speech_data[room_id].append({
        'text': text,
        'timestamp': time.time(),
        'type': 'generated',
        'speech_id': speech_id
    })
    if len(room_speech_data[room_id]) > 50:
        room_speech_data[room_id].pop(0)

@socketio.on('recognized_speech')
def handle_recognized_speech(data):
    room_id = data['room_id']
    text = data['text']
    
    # Игнорируем речь, которая воспроизводится с сервера
    for speech_id, speech_text in active_speeches[room_id].items():
        if speech_text.lower() in text.lower():
            return
    
    emit('speech_text', {
        'text': text,
        'sid': request.sid,
        'phonemes': [get_phoneme_for_char(c) for c in text]
    }, room=room_id)
    
    room_speech_data[room_id].append({
        'text': text,
        'timestamp': time.time(),
        'type': 'recognized',
        'sid': request.sid
    })
    if len(room_speech_data[room_id]) > 50:
        room_speech_data[room_id].pop(0)

@socketio.on('speech_ended')
def handle_speech_ended(data):
    room_id = data['room_id']
    speech_id = data['speech_id']
    if speech_id in active_speeches[room_id]:
        del active_speeches[room_id][speech_id]

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
