import json
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

class KnowledgeBase:
    def __init__(self, subject: str = "general"):
        self.subject = subject
        self.knowledge_path = Path(f"materials/{subject}_knowledge.json")
        self.dialogue_path = Path("materials/dialogue_knowledge.json")
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words=['и', 'в', 'на', 'с', 'по', 'для', 'что', 'это'])
        self.data = self._load_knowledge()
        self.dialogue_data = self._load_dialogue_knowledge()
        self._init_vectorizer()
        
    def _load_dialogue_knowledge(self) -> Dict:
        """Загрузка диалоговых шаблонов"""
        try:
            if not self.dialogue_path.parent.exists():
                self.dialogue_path.parent.mkdir(parents=True)
                
            if self.dialogue_path.exists():
                with open(self.dialogue_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Ошибка загрузки диалоговых шаблонов: {e}")
        
        # База по умолчанию
        return {
            "patterns": {
                "как дела": ["Все хорошо, продолжим урок", "Отлично, давайте заниматься"],
                "не понял": ["Давайте разберем еще раз", "Попробую объяснить по-другому"],
                "повтори": ["Повторяю...", "Еще раз:"],
                "спасибо": ["Пожалуйста!", "Всегда рад помочь!"]
            },
            "contexts": {
                "greeting": ["Привет!", "Здравствуйте!", "Рад вас видеть!"],
                "farewell": ["До свидания!", "Удачи!", "До следующего урока!"]
            },
            "metadata": {
                "version": "1.0",
                "type": "dialogue_patterns",
                "last_updated": "2024-01-01"
            }
        }

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
                "культура духовные ценности",
                "образование процесс обучения",
                "наука система знаний"
            ]
            self.vectorizer.fit(demo_texts)

    def get_dialogue_response(self, text: str) -> Optional[str]:
        """Получение ответа из диалоговых шаблонов"""
        if not text.strip():
            return None
            
        text_lower = text.lower().strip()
        
        # Поиск по шаблонам
        for pattern, responses in self.dialogue_data.get("patterns", {}).items():
            if pattern in text_lower and responses:
                return random.choice(responses)
        
        # Поиск по контекстам (если есть точное совпадение)
        for context, responses in self.dialogue_data.get("contexts", {}).items():
            if context in text_lower and responses:
                return random.choice(responses)
                
        return None

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

    def add_dialogue_pattern(self, pattern: str, responses: List[str]):
        """Добавление диалогового шаблона"""
        if pattern and responses:
            self.dialogue_data["patterns"][pattern.lower()] = responses
            self._save_dialogue_knowledge()

    def _save_knowledge(self):
        """Сохранение базы знаний в файл"""
        try:
            with open(self.knowledge_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения базы знаний: {e}")

    def _save_dialogue_knowledge(self):
        """Сохранение диалоговых шаблонов в файл"""
        try:
            with open(self.dialogue_path, 'w', encoding='utf-8') as f:
                json.dump(self.dialogue_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения диалоговых шаблонов: {e}")

    def get_stats(self) -> Dict:
        """Получение статистики базы знаний"""
        return {
            "subject": self.subject,
            "terms_count": len(self.data["terms"]),
            "questions_count": len(self.data["questions"]),
            "examples_count": len(self.data["examples"]),
            "dialogue_patterns": len(self.dialogue_data.get("patterns", {})),
            "dialogue_contexts": len(self.dialogue_data.get("contexts", {}))
        }

    def search_similar(self, query: str, max_results: int = 3) -> List[Dict]:
        """Поиск похожих вопросов и терминов"""
        results = []
        
        if not query.strip():
            return results
            
        query_lower = query.lower().strip()
        
        # Поиск по вопросам
        if self.data["questions"]:
            questions = list(self.data["questions"].keys())
            try:
                q_vec = self.vectorizer.transform([query_lower])
                q_vecs = self.vectorizer.transform(questions)
                similarities = cosine_similarity(q_vec, q_vecs)
                
                for i, similarity in enumerate(similarities[0]):
                    if similarity > 0.3:  # Порог схожести
                        results.append({
                            "type": "question",
                            "text": questions[i],
                            "answer": self.data["questions"][questions[i]],
                            "similarity": float(similarity)
                        })
            except Exception as e:
                print(f"Ошибка поиска похожих вопросов: {e}")
        
        # Поиск по терминам
        for term, definition in self.data["terms"].items():
            if query_lower in term.lower():
                results.append({
                    "type": "term",
                    "text": term,
                    "definition": definition,
                    "similarity": 1.0
                })
        
        # Сортировка по схожести
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:max_results]
