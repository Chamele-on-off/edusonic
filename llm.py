import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class LLMIntegration:
    def __init__(self, knowledge_dir: str = "materials"):
        self.knowledge_dir = Path(knowledge_dir)
        self.vectorizer = TfidfVectorizer()
        self._init_vectorizer()

    def _init_vectorizer(self):
        # Инициализация на демо-данных
        demo_texts = ["общество", "государство", "экономика"]
        self.vectorizer.fit(demo_texts)

    def find_similar_question(self, question: str, knowledge: Dict, threshold: float = 0.5) -> Optional[str]:
        """Поиск похожего вопроса в базе знаний"""
        questions = list(knowledge["questions"].keys())
        if not questions:
            return None
            
        q_vec = self.vectorizer.transform([question])
        q_vecs = self.vectorizer.transform(questions)
        similarities = cosine_similarity(q_vec, q_vecs)
        max_idx = np.argmax(similarities)
        
        if similarities[0][max_idx] > threshold:
            return questions[max_idx]
        return None

    def process_unknown_question(self, question: str, subject: str) -> str:
        """Обработка неизвестного вопроса (заглушка для LLM)"""
        return f"Я записал ваш вопрос по теме {subject} и отвечу на следующем занятии."
