import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SocietyKnowledge:
    def __init__(self):
        self.knowledge_path = Path("materials/social_knowledge.json")
        self.vectorizer = TfidfVectorizer()
        self.data = self._load_knowledge()
        self._init_vectorizer()
        
    def _load_knowledge(self) -> Dict:
        try:
            with open(self.knowledge_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "terms": {},
                "questions": {},
                "examples": {}
            }

    def _init_vectorizer(self):
        texts = list(self.data["terms"].keys()) + list(self.data["questions"].keys())
        if texts:
            self.vectorizer.fit(texts)

    def find_answer(self, question: str, threshold: float = 0.6) -> Optional[str]:
        # Точное совпадение
        if question in self.data["questions"]:
            return self.data["questions"][question]
        
        # Поиск по схожести
        if self.data["questions"]:
            questions = list(self.data["questions"].keys())
            q_vec = self.vectorizer.transform([question])
            q_vecs = self.vectorizer.transform(questions)
            similarities = cosine_similarity(q_vec, q_vecs)
            max_idx = np.argmax(similarities)
            
            if similarities[0][max_idx] > threshold:
                return self.data["questions"][questions[max_idx]]
        
        return None

    def get_term(self, term: str) -> Optional[str]:
        return self.data["terms"].get(term)

    def add_knowledge(self, term: str, definition: str, question: str = None, answer: str = None):
        if term:
            self.data["terms"][term] = definition
        if question and answer:
            self.data["questions"][question] = answer
        
        with open(self.knowledge_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
