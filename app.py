import os
import json
import uuid
import time
import random
import logging
import base64
from pathlib import Path
from io import BytesIO
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, render_template, send_file
from flask_socketio import SocketIO
from PIL import Image, ImageDraw
import numpy as np
import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Конфигурация
BASE_DIR = Path(__file__).parent
CONFIG = {
    "materials_dir": "materials",
    "lessons_dir": "static/lessons",
    "avatar_frames_dir": "static/avatar/frames",
    "audio_cache": "static/audio",
    "max_workers": 4
}

# Создание необходимых директорий
for dir_path in [CONFIG["lessons_dir"], CONFIG["avatar_frames_dir"], CONFIG["audio_cache"], "logs", CONFIG["materials_dir"]]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# Инициализация Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET', 'dev-secret-key-123')
socketio = SocketIO(app, async_mode='threading')
executor = ThreadPoolExecutor(max_workers=CONFIG["max_workers"])

class AvatarGenerator:
    """Генератор 2D аватара с анимацией рта и морганием"""
    def __init__(self, frames_dir: str = CONFIG["avatar_frames_dir"]):
        self.frames_dir = frames_dir
        self.frames = self._load_frames()
        self.current_frame_index = 0
        self.last_phoneme = None
        self.last_update = 0
        self.last_blink = time.time()
        self.blink_interval = random.uniform(3, 5)
        
        self.phoneme_groups = {
            'а': 'mouth_aa', 'о': 'mouth_oo', 'у': 'mouth_uu',
            'и': 'mouth_ee', 'э': 'mouth_ee', 'ы': 'mouth_aa',
            'е': 'mouth_ee', 'ё': 'mouth_oo', 'ю': 'mouth_uu',
            'я': 'mouth_aa', 'м': 'mouth_mm', 'п': 'mouth_pp',
            'б': 'mouth_bb', 'ф': 'mouth_ff', 'в': 'mouth_vv',
            'ш': 'mouth_sh', 'ж': 'mouth_zh', 'с': 'mouth_ss',
            'з': 'mouth_zz', 'р': 'mouth_rr', 'л': 'mouth_ll',
            'н': 'mouth_nn', 'т': 'mouth_tt', 'д': 'mouth_dd',
            'к': 'mouth_kk', 'г': 'mouth_gg', 'х': 'mouth_hh',
            'ч': 'mouth_ch', 'щ': 'mouth_sh', 'ц': 'mouth_ss',
            'й': 'mouth_ee'
        }

    def _load_frames(self) -> Dict[str, List[bytes]]:
        """Загружает и группирует кадры анимации"""
        frames = {}
        
        if not os.path.exists(self.frames_dir):
            logger.warning(f"Директория {self.frames_dir} не найдена!")
            return frames

        for filename in sorted(os.listdir(self.frames_dir)):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    group = self._get_frame_group(filename)
                    with open(os.path.join(self.frames_dir, filename), 'rb') as f:
                        if group not in frames:
                            frames[group] = []
                        frames[group].append(f.read())
                except Exception as e:
                    logger.error(f"Ошибка загрузки {filename}: {str(e)}")

        if not frames:
            logger.warning("Созданы дефолтные кадры аватара")
            frames = {
                'mouth_neutral': [self._generate_default_frame(True, False)],
                'mouth_open': [self._generate_default_frame(True, True)],
                'blink': [self._generate_default_frame(False, False)]
            }

        logger.info(f"Загружено {sum(len(v) for v in frames.values())} кадров")
        return frames

    def _get_frame_group(self, filename: str) -> str:
        """Определяет группу кадра по имени файла"""
        base = os.path.splitext(filename)[0].rsplit('_', 1)[0]
        return base if base in self.phoneme_groups.values() else 'mouth_neutral'

    def _generate_default_frame(self, eyes_open: bool, mouth_open: bool) -> bytes:
        """Генерирует фолбек-кадр SVG"""
        img = Image.new('RGB', (640, 480), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        # Голова
        draw.ellipse([(120, 50), (520, 430)], outline=(0, 0, 0), width=2, fill=(255, 255, 255))
        
        # Глаза
        eye_y = 180
        if eyes_open:
            draw.ellipse([(220, eye_y), (280, eye_y+60)], fill=(0, 0, 0))
            draw.ellipse([(360, eye_y), (420, eye_y+60)], fill=(0, 0, 0))
        else:
            draw.line([(220, eye_y+30), (280, eye_y+30)], fill=(0, 0, 0), width=2)
            draw.line([(360, eye_y+30), (420, eye_y+30)], fill=(0, 0, 0), width=2)
        
        # Рот
        mouth_y = 320
        if mouth_open:
            draw.ellipse([(270, mouth_y), (370, mouth_y+80)], fill=(0, 0, 0))
        else:
            draw.line([(270, mouth_y+40), (370, mouth_y+40)], fill=(0, 0, 0), width=2)
        
        buf = BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

    def get_current_frame(self, phoneme: Optional[str] = None) -> bytes:
        """Возвращает текущий кадр с анимацией"""
        now = time.time()
        
        # Обновление фонемы
        if phoneme and now - self.last_update > 0.1:
            self.last_phoneme = phoneme.lower()
            self.last_update = now
        
        # Моргание
        if now - self.last_blink > self.blink_interval:
            self.last_blink = now
            self.blink_interval = random.uniform(3, 5)
            if 'blink' in self.frames:
                return random.choice(self.frames['blink'])
        
        # Анимация рта
        if self.last_phoneme:
            group = self.phoneme_groups.get(self.last_phoneme, 'mouth_neutral')
            if group in self.frames and self.frames[group]:
                self.current_frame_index = (self.current_frame_index + 1) % len(self.frames[group])
                return self.frames[group][self.current_frame_index]
        
        # Нейтральное состояние
        return self.frames.get('mouth_neutral', [self._generate_default_frame(True, False)])[0]

    def get_available_frames(self) -> Dict[str, List[str]]:
        """Возвращает список доступных кадров для фронтенда"""
        frames = {}
        for group in ['mouth_neutral', 'mouth_aa', 'mouth_oo', 'mouth_ee', 'blink']:
            frame_files = sorted(Path(self.frames_dir).glob(f"{group}_*.jpg"))
            frames[group] = [f"/static/avatar/frames/{f.name}" for f in frame_files]
        return frames

class KnowledgeBase:
    """База знаний для ответов на вопросы (JSON-based)"""
    def __init__(self, data_dir: str = CONFIG["materials_dir"]):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.materials_file = self.data_dir / "materials.json"
        self._init_data()
    
    def _init_data(self):
        """Инициализирует данные, если файла нет"""
        if not self.materials_file.exists():
            with open(self.materials_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "materials": [
                        {
                            "id": "1",
                            "subject": "обществознание",
                            "title": "Понятие общества",
                            "content": "Общество — это совокупность людей, объединенных исторически сложившимися формами взаимодействия."
                        },
                        {
                            "id": "2",
                            "subject": "обществознание",
                            "title": "Государство",
                            "content": "Государство — политическая организация общества, обладающая суверенитетом."
                        }
                    ]
                }, f, ensure_ascii=False, indent=2)
    
    def _load_materials(self) -> List[Dict]:
        """Загружает материалы из JSON файла"""
        try:
            with open(self.materials_file, 'r', encoding='utf-8') as f:
                return json.load(f).get("materials", [])
        except Exception as e:
            logger.error(f"Ошибка загрузки материалов: {str(e)}")
            return []
    
    def _save_materials(self, materials: List[Dict]):
        """Сохраняет материалы в JSON файл"""
        with open(self.materials_file, 'w', encoding='utf-8') as f:
            json.dump({"materials": materials}, f, ensure_ascii=False, indent=2)
    
    def add_material(self, subject: str, title: str, content: str):
        """Добавляет учебный материал"""
        materials = self._load_materials()
        materials.append({
            "id": str(uuid.uuid4()),
            "subject": subject,
            "title": title,
            "content": content
        })
        self._save_materials(materials)
        logger.info(f"Добавлен материал: {title} ({subject})")
    
    def find_similar(self, query: str, subject: str, top_k: int = 3) -> List[Dict]:
        """Находит релевантные материалы (простая реализация)"""
        materials = self._load_materials()
        filtered = [m for m in materials if m.get("subject") == subject]
        
        if not query:
            return filtered[:top_k]
        
        # Простая фильтрация по вхождению слов запроса
        query_words = set(query.lower().split())
        for m in filtered:
            content_words = set(m["content"].lower().split())
            m["score"] = len(query_words & content_words)
        
        return sorted(filtered, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

class SimpleTTS:
    """Синтезатор речи с кэшированием"""
    def __init__(self, cache_dir: str = CONFIG["audio_cache"]):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.phoneme_map = {
            'а': ('mouth_aa', 0.3), 'о': ('mouth_oo', 0.3),
            'у': ('mouth_uu', 0.3), 'и': ('mouth_ee', 0.3),
            'э': ('mouth_ee', 0.3), 'ы': ('mouth_aa', 0.3),
            'е': ('mouth_ee', 0.3), 'ё': ('mouth_oo', 0.3),
            'ю': ('mouth_uu', 0.3), 'я': ('mouth_aa', 0.3),
            'м': ('mouth_mm', 0.15), 'п': ('mouth_pp', 0.15),
            'б': ('mouth_bb', 0.15), 'ф': ('mouth_ff', 0.15),
            'в': ('mouth_vv', 0.15), 'ш': ('mouth_sh', 0.15),
            'ж': ('mouth_zh', 0.15), 'с': ('mouth_ss', 0.15),
            'з': ('mouth_zz', 0.15), 'р': ('mouth_rr', 0.15),
            'л': ('mouth_ll', 0.15), 'н': ('mouth_nn', 0.15),
            'т': ('mouth_tt', 0.15), 'д': ('mouth_dd', 0.15),
            'к': ('mouth_kk', 0.15), 'г': ('mouth_gg', 0.15),
            'х': ('mouth_hh', 0.15), 'ч': ('mouth_ch', 0.15),
            'щ': ('mouth_sh', 0.15), 'ц': ('mouth_ss', 0.15),
            'й': ('mouth_ee', 0.15)
        }

    def text_to_phonemes(self, text: str) -> List[Tuple[str, float]]:
        """Конвертирует текст в последовательность фонем"""
        return [
            self.phoneme_map.get(char, ('mouth_neutral', 0.15))
            for char in text.lower() if char in self.phoneme_map
        ][:100]

    def synthesize(self, text: str, lang: str = 'ru') -> Dict:
        """Синтезирует речь и возвращает аудио + фонемы"""
        from gtts import gTTS
        
        cache_key = f"{lang}_{hash(text)}"
        cache_file = self.cache_dir / f"{cache_key}.wav"
        
        try:
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    audio_data = f.read()
                logger.info(f"Using cached audio for: {text[:50]}...")
            else:
                tts = gTTS(text=text, lang=lang, slow=False)
                audio_buffer = BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_data = audio_buffer.getvalue()
                with open(cache_file, 'wb') as f:
                    f.write(audio_data)
                logger.info(f"Generated new audio for: {text[:50]}...")

            return {
                "audio": base64.b64encode(audio_data).decode('utf-8'),
                "phonemes": self.text_to_phonemes(text),
                "text": text
            }
        except Exception as e:
            logger.error(f"TTS error: {str(e)}")
            return {
                "error": str(e),
                "phonemes": [('mouth_neutral', 0.2)] * 3
            }

class WebRTCHandler:
    """Обработчик WebRTC соединений"""
    def __init__(self, socketio):
        self.socketio = socketio
        self.pcs = set()
        self.logger = logging.getLogger(__name__)
    
    def handle_offer(self, offer: dict, room: str):
        """Обработка WebRTC оффера в отдельном потоке"""
        executor.submit(self._async_handle_offer, offer, room)
    
    def _async_handle_offer(self, offer: dict, room: str):
        """Асинхронная обработка оффера"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._process_offer(offer, room))
        loop.close()
    
    async def _process_offer(self, offer: dict, room: str):
        """Основная логика обработки оффера"""
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            if pc.iceConnectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)

        try:
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
            )
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            self.socketio.emit('webrtc_answer', {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type
            }, room=room)
            
            self.logger.info(f"WebRTC соединение установлено для комнаты {room}")

        except Exception as e:
            self.logger.error(f"WebRTC ошибка: {str(e)}")
            self.socketio.emit('webrtc_error', {
                "error": str(e)
            }, room=room)

    async def cleanup(self):
        """Очистка соединений"""
        for pc in self.pcs:
            await pc.close()
        self.pcs.clear()
        self.logger.info("Все WebRTC соединения закрыты")

# Инициализация компонентов
avatar_generator = AvatarGenerator()
knowledge_base = KnowledgeBase()
tts_engine = SimpleTTS()
webrtc_handler = WebRTCHandler(socketio)

# API Endpoints
@app.route('/')
def home():
    return render_template('teacher.html')

@app.route('/api/avatar_frame')
def get_avatar_frame():
    phoneme = request.args.get('phoneme')
    frame = avatar_generator.get_current_frame(phoneme)
    return send_file(BytesIO(frame), mimetype='image/jpeg')

@app.route('/api/avatar_frames')
def get_avatar_frames():
    return jsonify(avatar_generator.get_available_frames())

@app.route('/api/lessons')
def list_lessons():
    lessons = []
    for lesson_file in Path(CONFIG["lessons_dir"]).glob("*.json"):
        try:
            with open(lesson_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                lessons.append({
                    'id': data.get('id', os.path.splitext(lesson_file.name)[0]),
                    'title': data.get('title', 'Без названия'),
                    'subject': data.get('subject', 'не указан'),
                    'description': data.get('description', '')
                })
        except Exception as e:
            logger.error(f"Ошибка загрузки урока {lesson_file}: {str(e)}")
    return jsonify({"lessons": lessons})

@app.route('/api/upload_lesson', methods=['POST'])
def upload_lesson():
    if 'lesson' not in request.files:
        return jsonify({"error": "Файл не загружен"}), 400
    
    file = request.files['lesson']
    if file.filename == '':
        return jsonify({"error": "Пустое имя файла"}), 400
    
    try:
        lesson_id = request.form.get('lesson_id', str(uuid.uuid4()))
        save_path = Path(CONFIG["lessons_dir"]) / f"{lesson_id}.json"
        file.save(save_path)
        return jsonify({"status": "success", "lesson_id": lesson_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/start_lesson', methods=['POST'])
def start_lesson():
    data = request.json
    lesson_id = data.get('lesson_id')
    room_id = data.get('room_id')
    
    if not lesson_id or not room_id:
        return jsonify({"error": "Не указан ID урока или комнаты"}), 400
    
    try:
        session_id = f"{lesson_id}_{room_id}_{int(time.time())}"
        return jsonify({
            "session_id": session_id,
            "status": "started"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/add_material', methods=['POST'])
def add_material():
    try:
        data = request.json
        knowledge_base.add_material(
            subject=data['subject'],
            title=data['title'],
            content=data['content']
        )
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# WebSocket handlers
@socketio.on('connect')
def handle_connect():
    logger.info(f"Клиент подключен: {request.sid}")

@socketio.on('webrtc_offer')
def handle_webrtc_offer(data):
    room = data.get('room')
    offer = data.get('offer')
    webrtc_handler.handle_offer(offer, room)

@socketio.on('ask_question')
def handle_question(data):
    def process():
        try:
            question = data.get('question', '')
            subject = data.get('subject', 'обществознание')
            
            materials = knowledge_base.find_similar(question, subject)
            response_text = "\n".join([m['content'] for m in materials])
            
            tts_result = tts_engine.synthesize(response_text)
            
            socketio.emit('ai_response', {
                "text": response_text,
                "audio": tts_result["audio"],
                "phonemes": tts_result["phonemes"],
                "materials": materials
            }, room=request.sid)
        except Exception as e:
            logger.error(f"Ошибка обработки вопроса: {str(e)}")
            socketio.emit('error', {"message": str(e)}, room=request.sid)
    
    executor.submit(process)

@socketio.on('audio_data')
def handle_audio_data(data):
    def process():
        try:
            from speech_recognition import Recognizer, AudioData
            import wave
            
            audio_bytes = base64.b64decode(data['audio'])
            
            # Конвертируем в формат, понятный для speech_recognition
            with wave.open(BytesIO(audio_bytes), 'rb') as wav_file:
                audio_data = AudioData(
                    wav_file.readframes(wav_file.getnframes()),
                    wav_file.getframerate(),
                    wav_file.getsampwidth()
                )
            
            recognizer = Recognizer()
            text = recognizer.recognize_google(audio_data, language="ru-RU")
            
            socketio.emit('transcription', {
                "text": text,
                "room": data['room']
            }, room=request.sid)
            
        except Exception as e:
            logger.error(f"Ошибка транскрипции: {str(e)}")
    
    executor.submit(process)

if __name__ == '__main__':
    try:
        host = os.getenv('HOST', '0.0.0.0')
        port = int(os.getenv('PORT', 5000))
        logger.info(f"Сервер запущен на {host}:{port}")
        socketio.run(app, host=host, port=port, debug=True)
    except Exception as e:
        logger.error(f"Ошибка сервера: {str(e)}")
    finally:
        executor.shutdown()
        asyncio.get_event_loop().run_until_complete(webrtc_handler.cleanup())
        logger.info("Сервер остановлен")