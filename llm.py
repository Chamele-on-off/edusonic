import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KnowledgeBase:
    def __init__(self, db_path: str = "materials.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._init_db()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB модель

    def _init_db(self):
        """Инициализация базы данных"""
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
        """Добавление учебного материала"""
        embedding = self.model.encode(content)
        self.conn.execute(
            "INSERT INTO materials (subject, title, content, embedding) VALUES (?, ?, ?, ?)",
            (subject, title, content, embedding.tobytes())
        )
        self.conn.commit()
        logger.info(f"Добавлен материал: {title} ({subject})")

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

    def generate_response(self, question: str, context: List[Dict]) -> str:
        """Генерация ответа на основе контекста"""
        if not context:
            return "Информация по данному вопросу не найдена."
        
        # Простое объединение лучших результатов
        combined = "\n".join(
            f"### {item['title']}\n{item['content']}" 
            for item in context
        )
        
        return f"""Вот что я нашел по вашему вопросу "{question}":\n\n{combined}"""

    def close(self):
        self.conn.close()

# Пример использования:
if __name__ == "__main__":
    kb = KnowledgeBase()
    
    # Добавление материалов (обычно делается один раз)
    kb.add_material(
        subject="математика",
        title="Квадратные уравнения",
        content="Квадратное уравнение имеет вид ax² + bx + c = 0..."
    )
    
    # Поиск информации
    results = kb.find_similar("Как решать уравнения второй степени?", "математика")
    answer = kb.generate_response("Квадратные уравнения", results)
    print(answer)