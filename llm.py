import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KnowledgeBase:
    def __init__(self, data_dir: str = "materials"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.materials_file = self.data_dir / "materials.json"
        self.vectorizer_file = self.data_dir / "tfidf_vectorizer.joblib"
        self._init_data()
        self.vectorizer = self._init_vectorizer()

    def _init_data(self):
        """Инициализация данных, если файла нет"""
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
            logger.info("Создан новый файл материалов")

    def _init_vectorizer(self):
        """Инициализация TF-IDF векторайзера"""
        if self.vectorizer_file.exists():
            logger.info("Загрузка существующего векторайзера")
            return joblib.load(self.vectorizer_file)
        
        logger.info("Создание нового TF-IDF векторайзера")
        vectorizer = TfidfVectorizer(max_features=5000)
        
        # Инициализация на демо-данных
        demo_texts = [
            "Общество — это совокупность людей",
            "Экономика изучает производство и потребление",
            "Право — система норм и правил поведения"
        ]
        vectorizer.fit(demo_texts)
        joblib.dump(vectorizer, self.vectorizer_file)
        return vectorizer

    def _load_materials(self) -> List[Dict]:
        """Загрузка материалов из JSON файла"""
        try:
            with open(self.materials_file, 'r', encoding='utf-8') as f:
                return json.load(f).get("materials", [])
        except Exception as e:
            logger.error(f"Ошибка загрузки материалов: {str(e)}")
            return []

    def _save_materials(self, materials: List[Dict]):
        """Сохранение материалов в JSON файл"""
        with open(self.materials_file, 'w', encoding='utf-8') as f:
            json.dump({"materials": materials}, f, ensure_ascii=False, indent=2)

    def add_material(self, subject: str, title: str, content: str):
        """Добавление учебного материала"""
        materials = self._load_materials()
        materials.append({
            "id": str(len(materials) + 1),
            "subject": subject,
            "title": title,
            "content": content
        })
        self._save_materials(materials)
        logger.info(f"Добавлен материал: {title} ({subject})")

    def find_similar(self, query: str, subject: str, top_k: int = 3) -> List[Dict]:
        """Поиск релевантных материалов с TF-IDF"""
        materials = self._load_materials()
        filtered = [m for m in materials if m.get("subject") == subject]

        if not filtered:
            return []

        # Векторизация контента и запроса
        contents = [m["content"] for m in filtered]
        all_texts = contents + [query]
        tfidf_matrix = self.vectorizer.transform(all_texts)

        # Вычисление схожести
        query_vector = tfidf_matrix[-1]
        for i, material in enumerate(filtered):
            material["score"] = cosine_similarity(
                query_vector, 
                tfidf_matrix[i]
            )[0][0]

        return sorted(filtered, key=lambda x: x["score"], reverse=True)[:top_k]

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