from flask import Flask, render_template, jsonify
import os

app = Flask(__name__)

@app.route('/')
def animation_test():
    return render_template('animation_test.html')

@app.route('/api/avatar_frames')
def get_avatar_frames():
    frames = {
        'mouth_neutral': [],
        'blink': [],
        'mouth_aa': [],
        'mouth_bb': [],
        # Добавьте остальные группы по аналогии
    }
    
    frames_dir = os.path.join('static', 'avatar', 'frames')
    for filename in os.listdir(frames_dir):
        if filename.startswith('mouth_neutral_'):
            frames['mouth_neutral'].append(f'/static/avatar/frames/{filename}')
        elif filename.startswith('blink_'):
            frames['blink'].append(f'/static/avatar/frames/{filename}')
        elif filename.startswith('mouth_aa_'):
            frames['mouth_aa'].append(f'/static/avatar/frames/{filename}')
        elif filename.startswith('mouth_bb_'):
            frames['mouth_bb'].append(f'/static/avatar/frames/{filename}')
        # Добавьте обработку остальных префиксов
    
    return jsonify(frames)

if __name__ == '__main__':
    app.run(debug=True)