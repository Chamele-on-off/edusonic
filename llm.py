import sqlite3
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import joblib
from typing import List, Dict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KnowledgeBase:
    def __init__(self, db_path: str = "materials/knowledge.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.vectorizer = self._init_vectorizer()
        self._init_db()

    def _init_db(self):
        """Инициализация структуры базы данных"""
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS materials (
            id INTEGER PRIMARY KEY,
            subject TEXT,
            title TEXT,
            content TEXT
        )
        """)
        self.conn.commit()

    def _init_vectorizer(self):
        """Инициализация TF-IDF векторайзера"""
        vectorizer_path = Path("models/tfidf_vectorizer.joblib")
        vectorizer_path.parent.mkdir(exist_ok=True)
        
        if vectorizer_path.exists():
            logger.info("Загрузка существующего векторайзера")
            return joblib.load(vectorizer_path)
        
        logger.info("Создание нового TF-IDF векторайзера")
        vectorizer = TfidfVectorizer(max_features=5000)
        
        # Инициализация на демо-данных
        demo_texts = [
            "Общество — это совокупность людей",
            "Экономика изучает производство и потребление",
            "Право — система норм и правил поведения"
        ]
        vectorizer.fit(demo_texts)
        joblib.dump(vectorizer, vectorizer_path)
        return vectorizer

    def add_material(self, subject: str, title: str, content: str):
        """Добавление учебного материала"""
        self.conn.execute(
            "INSERT INTO materials (subject, title, content) VALUES (?, ?, ?)",
            (subject, title, content)
        )
        self.conn.commit()
        logger.info(f"Добавлен материал: {title} ({subject})")

    def find_similar(self, query: str, subject: str, top_k: int = 3) -> List[Dict]:
        """Поиск релевантных материалов с TF-IDF"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, title, content FROM materials WHERE subject = ?",
            (subject,)
        )
        materials = [
            {"id": row[0], "title": row[1], "content": row[2]}
            for row in cursor.fetchall()
        ]

        if not materials:
            return []

        # Векторизация контента и запроса
        contents = [m["content"] for m in materials]
        all_texts = contents + [query]
        tfidf_matrix = self.vectorizer.transform(all_texts)

        # Вычисление схожести
        query_vector = tfidf_matrix[-1]
        for i, material in enumerate(materials):
            material["score"] = cosine_similarity(
                query_vector, 
                tfidf_matrix[i]
            )[0][0]

        return sorted(materials, key=lambda x: x["score"], reverse=True)[:top_k]

    def generate_response(self, question: str, context: List[Dict]) -> Dict:
        """Формирование ответа на основе найденных материалов"""
        if not context:
            return {
                "text": "Информация по данному вопросу не найдена.",
                "materials": []
            }

        response_text = f"Ответ на вопрос '{question}':\n\n"
        response_text += "\n\n".join(
            f"### {item['title']}\n{item['content']}" 
            for item in context
        )
        
        return {
            "text": response_text,
            "materials": context
        }

    def close(self):
        self.conn.close()