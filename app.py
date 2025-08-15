from flask import Flask, render_template, send_from_directory, jsonify, request
import os
from pathlib import Path
from flask_socketio import SocketIO, emit, join_room, leave_room
import time
import threading
from collections import defaultdict

app = Flask(__name__, static_folder='static')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

BASE_DIR = Path(__file__).parent
FRAMES_DIR = BASE_DIR / 'static' / 'avatar' / 'frames'

animation_running = defaultdict(bool)
current_animation_frames = defaultdict(list)
current_frame_index = defaultdict(int)
room_participants = defaultdict(set)
room_speech_data = defaultdict(list)
room_audio_data = defaultdict(dict)

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

def animation_loop(room_id):
    while animation_running[room_id]:
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
    if not animation_running[room_id]:
        current_animation_frames[room_id] = data['frames']
        animation_running[room_id] = True
        threading.Thread(target=animation_loop, args=(room_id,)).start()

@socketio.on('stop_animation')
def handle_stop_animation(data):
    room_id = data['room_id']
    animation_running[room_id] = False

@socketio.on('generate_speech')
def handle_generate_speech(data):
    room_id = data['room_id']
    text = data['text']
    lang = data.get('lang', 'ru-RU')
    voice_name = data.get('voice', None)
    
    emit('speech_request', {
        'text': text,
        'lang': lang,
        'voice': voice_name
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
    emit('speech_text', {'text': text, 'sid': request.sid}, room=room_id)
    
    room_speech_data[room_id].append({
        'text': text,
        'timestamp': time.time(),
        'type': 'recognized',
        'sid': request.sid
    })
    if len(room_speech_data[room_id]) > 50:
        room_speech_data[room_id].pop(0)

@socketio.on('audio_data')
def handle_audio_data(data):
    room_id = data['room_id']
    audio_blob = data['audio']
    room_audio_data[room_id][request.sid] = audio_blob
    emit('audio_stream', {'audio': audio_blob, 'sid': request.sid}, room=room_id, include_self=False)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)