import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class KnowledgeBase:
    def __init__(self, subject: str = "general"):
        self.subject = subject
        self.knowledge_path = Path(f"materials/{subject}_knowledge.json")
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words=['и', 'в', 'на', 'с', 'по', 'для'])
        self.data = self._load_knowledge()
        self._init_vectorizer()
        
    def _load_knowledge(self) -> Dict:
        """Загрузка базы знаний из JSON файла"""
        try:
            if not self.knowledge_path.parent.exists():
                self.knowledge_path.parent.mkdir(parents=True)
                
            if self.knowledge_path.exists():
                with open(self.knowledge_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Ошибка загрузки базы знаний {self.subject}: {e}")
        
        # База по умолчанию
        return {
            "terms": {},
            "questions": {},
            "examples": {},
            "metadata": {
                "subject": self.subject,
                "version": "1.0",
                "last_updated": "2024-01-01"
            }
        }

    def _init_vectorizer(self):
        """Инициализация TF-IDF векторайзера"""
        texts = list(self.data["terms"].keys()) + list(self.data["questions"].keys())
        if texts:
            self.vectorizer.fit(texts)
        else:
            # Запасные данные для инициализации
            demo_texts = [
                "общество группа людей",
                "государство политическая организация", 
                "экономика хозяйственная деятельность",
                "право система норм",
                "культура духовные ценности"
            ]
            self.vectorizer.fit(demo_texts)

    def find_answer(self, question: str, threshold: float = 0.4) -> Optional[str]:
        """Поиск ответа на вопрос в базе знаний"""
        if not question.strip():
            return None
            
        question_lower = question.lower().strip()
        
        # 1. Точное совпадение
        if question_lower in self.data["questions"]:
            return self.data["questions"][question_lower]
        
        # 2. Поиск по терминам
        for term, definition in self.data["terms"].items():
            if term.lower() in question_lower:
                return definition
        
        # 3. Поиск по схожести (TF-IDF + косинусная схожесть)
        if self.data["questions"]:
            questions = list(self.data["questions"].keys())
            try:
                q_vec = self.vectorizer.transform([question_lower])
                q_vecs = self.vectorizer.transform(questions)
                similarities = cosine_similarity(q_vec, q_vecs)
                max_idx = np.argmax(similarities)
                
                if similarities[0][max_idx] > threshold:
                    return self.data["questions"][questions[max_idx]]
            except Exception as e:
                print(f"Ошибка поиска по схожести: {e}")
        
        return None

    def get_term(self, term: str) -> Optional[str]:
        """Получение определения термина"""
        return self.data["terms"].get(term.lower())

    def add_knowledge(self, term: str = None, definition: str = None, 
                     question: str = None, answer: str = None):
        """Добавление знаний в базу"""
        if term and definition:
            self.data["terms"][term.lower()] = definition
            
        if question and answer:
            self.data["questions"][question.lower()] = answer
            
        self._save_knowledge()

    def _save_knowledge(self):
        """Сохранение базы знаний в файл"""
        try:
            with open(self.knowledge_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения базы знаний: {e}")

    def get_stats(self) -> Dict:
        """Получение статистики базы знаний"""
        return {
            "subject": self.subject,
            "terms_count": len(self.data["terms"]),
            "questions_count": len(self.data["questions"]),
            "examples_count": len(self.data["examples"])
        }
