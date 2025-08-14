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

# Состояние комнат
rooms = {
    'default': {
        'animation_running': False,
        'current_frames': [],
        'current_index': 0,
        'participants': set(),
        'current_phoneme': 'neutral'
    }
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
        
        frames = sorted([f for f in os.listdir(avatar_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        return jsonify({"frames": frames})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/frames/<avatar_name>/<path:filename>')
def serve_frame(avatar_name, filename):
    return send_from_directory(FRAMES_DIR / avatar_name, filename)

def animation_loop(room_id):
    while rooms[room_id]['animation_running']:
        if rooms[room_id]['current_frames']:
            rooms[room_id]['current_index'] = (rooms[room_id]['current_index'] + 1) % len(rooms[room_id]['current_frames'])
            
            frame_data = {
                'frame': rooms[room_id]['current_frames'][rooms[room_id]['current_index']],
                'phoneme': rooms[room_id]['current_phoneme'],
                'index': rooms[room_id]['current_index'],
                'total': len(rooms[room_id]['current_frames'])
            }
            socketio.emit('animation_frame', frame_data, room=room_id)
        time.sleep(0.1)

@socketio.on('connect')
def handle_connect():
    print('Client connected:', request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    for room_id, room_data in rooms.items():
        if request.sid in room_data['participants']:
            room_data['participants'].remove(request.sid)
            emit('participant_left', {'sid': request.sid}, room=room_id)
            emit('participants_update', {'count': len(room_data['participants'])}, room=room_id)

@socketio.on('join_room')
def handle_join_room(data):
    room_id = data['room_id']
    if room_id not in rooms:
        rooms[room_id] = {
            'animation_running': False,
            'current_frames': [],
            'current_index': 0,
            'participants': set(),
            'current_phoneme': 'neutral'
        }
    
    join_room(room_id)
    rooms[room_id]['participants'].add(request.sid)
    emit('participants_update', {'count': len(rooms[room_id]['participants'])}, room=room_id)
    emit('new_participant', {'sid': request.sid}, room=room_id)

@socketio.on('start_animation')
def handle_start_animation(data):
    room_id = data['room_id']
    if not rooms[room_id]['animation_running']:
        rooms[room_id]['current_frames'] = data['frames']
        rooms[room_id]['animation_running'] = True
        threading.Thread(target=animation_loop, args=(room_id,)).start()

@socketio.on('stop_animation')
def handle_stop_animation(data):
    room_id = data['room_id']
    rooms[room_id]['animation_running'] = False
    rooms[room_id]['current_phoneme'] = 'neutral'

@socketio.on('update_phoneme')
def handle_update_phoneme(data):
    room_id = data['room_id']
    phoneme = data['phoneme']
    rooms[room_id]['current_phoneme'] = phoneme

@socketio.on('generate_speech')
def handle_generate_speech(data):
    room_id = data['room_id']
    text = data['text']
    emit('speech_text', {
        'text': text,
        'type': 'avatar',
        'room_id': room_id
    }, room=room_id)

@socketio.on('recognized_speech')
def handle_recognized_speech(data):
    room_id = data['room_id']
    text = data['text']
    emit('speech_text', {
        'text': text,
        'sid': request.sid,
        'type': 'human',
        'room_id': room_id
    }, room=room_id)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)