import os
import logging
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sqlite3
from pathlib import Path
import uuid
from concurrent.futures import ThreadPoolExecutor
from gtts import gTTS
import base64
from io import BytesIO
from typing import List, Dict, Tuple

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

# Инициализация Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET', 'dev-secret-key-123')
socketio = SocketIO(app, cors_allowed_origins="*")

# Создание директорий
Path("static/audio").mkdir(parents=True, exist_ok=True)
Path("materials").mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(parents=True, exist_ok=True)

class KnowledgeBase:
    def __init__(self, db_path: str = "materials/knowledge.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB модель
        self._init_db()

    def _init_db(self):
        """Инициализация структуры базы данных"""
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS materials (
            id INTEGER PRIMARY KEY,
            subject TEXT,
            title TEXT,
            content TEXT,
            embedding BLOB
        )
        """)
        self.conn.commit()

    def add_material(self, subject: str, title: str, content: str):
        """Добавление материала в базу знаний"""
        embedding = self.model.encode(content)
        self.conn.execute(
            "INSERT INTO materials (subject, title, content, embedding) VALUES (?, ?, ?, ?)",
            (subject, title, content, embedding.tobytes())
        )
        self.conn.commit()

    def find_similar(self, query: str, subject: str, top_k: int = 3) -> List[Dict]:
        """Поиск релевантных материалов"""
        query_embed = self.model.encode(query)
        cursor = self.conn.cursor()
        
        cursor.execute(
            "SELECT id, title, content, embedding FROM materials WHERE subject = ?",
            (subject,)
        )
        
        results = []
        for row in cursor.fetchall():
            embed = np.frombuffer(row[3], dtype=np.float32)
            similarity = cosine_similarity([query_embed], [embed])[0][0]
            results.append({
                'id': row[0],
                'title': row[1],
                'content': row[2],
                'score': float(similarity)
            })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]

    def generate_response(self, question: str, context: List[Dict]) -> Dict:
        """Генерация ответа на основе контекста"""
        if not context:
            return {
                'text': "Информация по данному вопросу не найдена.",
                'materials': []
            }
        
        return {
            'text': f"Вот что я нашел по вашему вопросу '{question}':\n\n" +
                   "\n\n".join(f"### {item['title']}\n{item['content']}" for item in context),
            'materials': context
        }

    def close(self):
        self.conn.close()

class SimpleTTS:
    def __init__(self, cache_dir: str = "static/audio"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def text_to_phonemes(self, text: str) -> List[Tuple[str, float]]:
        """Упрощенная генерация фонем"""
        vowels = {'а', 'е', 'ё', 'и', 'о', 'у', 'ы', 'э', 'ю', 'я'}
        phonemes = []
        for char in text.lower():
            if char in vowels:
                phonemes.append((char, 0.3))
            elif char.isalpha():
                phonemes.append((char, 0.1))
        return phonemes[:50]

    def synthesize(self, text: str, language: str = 'ru') -> dict:
        """Синтез речи с фонемами"""
        try:
            tts = gTTS(text=text, lang=language, slow=False)
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            return {
                'audio': base64.b64encode(audio_buffer.read()).decode('utf-8'),
                'phonemes': self.text_to_phonemes(text),
                'text': text
            }
        except Exception as e:
            logger.error(f"TTS error: {str(e)}")
            return {
                'error': str(e),
                'phonemes': [('а', 0.2)] * 3
            }

# Инициализация компонентов
executor = ThreadPoolExecutor(max_workers=4)
kb = KnowledgeBase()
tts = SimpleTTS()

def init_demo_data():
    """Инициализация демо-данных по обществознанию"""
    if not list(kb.find_similar("", "обществознание", 1)):
        demo_materials = [
            ("обществознание", "Понятие общества", "Общество — это совокупность людей, объединенных исторически сложившимися формами взаимодействия."),
            ("обществознание", "Типы экономических систем", "1. Традиционная 2. Командная 3. Рыночная 4. Смешанная. Россия имеет смешанную экономическую систему."),
            ("обществознание", "Разделение властей", "1. Законодательная (Федеральное Собрание) 2. Исполнительная (Правительство) 3. Судебная (суды)."),
            ("обществознание", "Социальная стратификация", "Деление общества на слои по доходам, власти, образованию. Основные классы: высший, средний, низший."),
            ("обществознание", "Политические режимы", "1. Демократия 2. Авторитаризм 3. Тоталитаризм. Россия — демократическое государство."),
            ("обществознание", "Конституция РФ", "Основной закон России. Принята 12 декабря 1993 года. Гарантирует права и свободы человека."),
            ("обществознание", "Глобализация", "Процесс worldwide интеграции в экономике, культуре и технологиях. Имеет как плюсы, так и минусы.")
        ]
        
        for subject, title, content in demo_materials:
            kb.add_material(subject, title, content)
        logger.info("Демо-данные по обществознанию загружены")

# Инициализация демо-данных при старте
init_demo_data()

# Web Routes
@app.route('/')
def home():
    return render_template('teacher.html')

@app.route('/api/materials', methods=['GET'])
def get_materials():
    subject = request.args.get('subject', 'обществознание')
    return jsonify({
        'materials': [m for m in kb.find_similar("", subject, 10) if m['score'] > 0.3]
    })

@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question', '')
    subject = data.get('subject', 'обществознание')
    
    materials = kb.find_similar(question, subject)
    response = kb.generate_response(question, materials)
    audio_data = tts.synthesize(response['text'])
    
    return jsonify({
        **response,
        'audio': audio_data['audio'],
        'phonemes': audio_data['phonemes']
    })

# WebSocket Handlers
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('ask_question')
def handle_question(data):
    session_id = data.get('session_id', str(uuid.uuid4()))
    
    def process():
        try:
            materials = kb.find_similar(data['question'], data.get('subject', 'обществознание'))
            response = kb.generate_response(data['question'], materials)
            audio = tts.synthesize(response['text'])
            
            socketio.emit('response', {
                'session_id': session_id,
                'text': response['text'],
                'audio': audio['audio'],
                'phonemes': audio['phonemes'],
                'materials': response['materials']
            }, room=request.sid)
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            socketio.emit('error', {
                'session_id': session_id,
                'message': str(e)
            }, room=request.sid)
    
    executor.submit(process)

if __name__ == '__main__':
    try:
        host = os.getenv('HOST', '0.0.0.0')
        port = int(os.getenv('PORT', 5000))
        logger.info(f"Starting server on {host}:{port}")
        socketio.run(app, host=host, port=port)
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
    finally:
        executor.shutdown()
        kb.close()