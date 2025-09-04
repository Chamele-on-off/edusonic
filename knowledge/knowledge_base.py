import json
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import string
import re
from datetime import datetime

class KnowledgeBase:
    def __init__(self, subject: str = "general"):
        self.subject = subject
        self.knowledge_path = Path(f"materials/{subject}_knowledge.json")
        self.llm_answers_path = Path(f"materials/{subject}_llm_answers.json")
        self.dialogue_path = Path("materials/dialogue_knowledge.json")
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words=['и', 'в', 'на', 'с', 'по', 'для', 'что', 'это'])
        self.llm_vectorizer = TfidfVectorizer(max_features=1000, stop_words=['и', 'в', 'на', 'с', 'по', 'для', 'что', 'это'])
        self.data = self._load_knowledge()
        self.llm_answers_data = self._load_llm_answers()
        self.dialogue_data = self._load_dialogue_knowledge()
        self._init_vectorizers()
        
    def _load_llm_answers(self) -> Dict:
        """Загрузка ответов LLM из файла"""
        try:
            if not self.llm_answers_path.parent.exists():
                self.llm_answers_path.parent.mkdir(parents=True)
                
            if self.llm_answers_path.exists():
                with open(self.llm_answers_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Ошибка загрузки ответов LLM: {e}")
        
        return {
            "answers": {},
            "metadata": {
                "subject": self.subject,
                "version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "total_answers": 0
            }
        }

    def _save_llm_answers(self):
        """Сохранение ответов LLM в файл"""
        try:
            with open(self.llm_answers_path, 'w', encoding='utf-8') as f:
                json.dump(self.llm_answers_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения ответов LLM: {e}")

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
        
        return {
            "patterns": {
                "как дела": ["Все хорошо, продолжим урок", "Отлично, давайте заниматься", "Прекрасно! А у вас?"],
                "не понял": ["Давайте разберем еще раз", "Попробую объяснить по-другому", "Хорошо, объясню подробнее"],
                "повтори": ["Конечно, повторяю...", "Еще раз:", "Повторяю для вас"],
                "спасибо": ["Пожалуйста!", "Всегда рад помочь!", "Рад был помочь!"],
                "скучно": ["Давайте сделаем урок интереснее!", "Предлагаю викторину!", "Сменим активность?"],
                "трудно": ["Не переживай! Вместе разберемся", "Сложности - это нормально!", "Я помогу тебе"],
                "молодец": ["Спасибо! Стараюсь для вас", "Рад, что нравится!", "Вы тоже молодец!"],
                "что такое": ["Сейчас объясню этот термин...", "Давайте разберем это понятие...", "Отличный вопрос! Это..."],
                "объясни": ["С удовольствием объясню", "Давайте разберем этот вопрос"],
                "расскажи": ["Расскажу подробнее", "Поделюсь информацией об этом"],
                "зачем учить": ["Это важно для понимания мира вокруг нас!", "Знания помогут в будущем!", "Это интересно и полезно!"],
                "почему важно": ["Это основа для многих других знаний!", "Помогает развивать мышление!", "Пригодится в жизни!"],
                "интересно": ["Рад, что вам интересно!", "Отлично! Продолжим изучать!", "Это действительно увлекательная тема!"],
                "понятно": ["Прекрасно, что поняли!", "Отлично усвоили материал!", "Так держать!"],
                "еще": ["Конечно, продолжим!", "Еще больше интересной информации!", "С удовольствием расскажу больше!"]
            },
            "contexts": {
                "greeting": ["Привет!", "Здравствуйте!", "Рад вас видеть!", "Добро пожаловать на урок!"],
                "farewell": ["До свидания!", "Удачи!", "До следующего урока!", "Хорошего дня!"],
                "encouragement": ["Отлично!", "Прекрасно!", "Так держать!", "Молодец!", "Замечательно!"],
                "question": ["Интересный вопрос!", "Хорошо, что спрашиваете!", "Отличный вопрос!", "Сейчас объясню!"],
                "explanation": ["Давайте разберем подробнее...", "Объясню этот момент...", "Вот как это работает..."],
                "transition": ["Теперь перейдем к...", "Следующая тема...", "А теперь давайте поговорим о..."]
            },
            "metadata": {
                "version": "2.0",
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
                print(f"Загружаю базу знаний из: {self.knowledge_path}")
                with open(self.knowledge_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"Успешно загружено {len(data.get('terms', {}))} терминов и {len(data.get('questions', {}))} вопросов по предмету {self.subject}")
                    return data
        except Exception as e:
            print(f"Ошибка загрузки базы знаний {self.subject}: {e}")
        
        print(f"База знаний для предмета '{self.subject}' не найдена. Создаю пустую базу.")
        return {
            "terms": {},
            "questions": {},
            "examples": {},
            "metadata": {
                "subject": self.subject,
                "version": "1.0",
                "last_updated": "2024-01-01",
                "note": "Автоматически созданная база"
            }
        }

    def _init_vectorizers(self):
        """Инициализация TF-IDF векторайзеров"""
        texts = list(self.data["terms"].keys()) + list(self.data["questions"].keys())
        if texts:
            self.vectorizer.fit(texts)
            print(f"Векторайзер инициализирован с {len(texts)} текстами для предмета {self.subject}")
        else:
            demo_texts = [
                "общество группа людей",
                "государство политическая организация", 
                "экономика хозяйственная деятельность",
                "право система норм",
                "культура духовные ценности"
            ]
            self.vectorizer.fit(demo_texts)
            print(f"Векторайзер инициализирован с демо-данными для предмета {self.subject}")
        
        llm_texts = list(self.llm_answers_data.get("answers", {}).keys())
        if llm_texts:
            self.llm_vectorizer.fit(llm_texts)
            print(f"LLM векторайзер инициализирован с {len(llm_texts)} ответами для предмета {self.subject}")
        else:
            self.llm_vectorizer.fit(["демо вопрос ответ"])
            print(f"LLM векторайзер инициализирован с демо-данными")

    def add_llm_answer(self, question: str, answer: str):
        """Добавление ответа LLM в базу"""
        clean_question = self._clean_text(question)
        if clean_question and answer:
            self.llm_answers_data["answers"][clean_question] = {
                "answer": answer,
                "timestamp": datetime.now().isoformat(),
                "subject": self.subject
            }
            self.llm_answers_data["metadata"]["total_answers"] = len(self.llm_answers_data["answers"])
            self.llm_answers_data["metadata"]["last_updated"] = datetime.now().isoformat()
            self._save_llm_answers()
            
            llm_texts = list(self.llm_answers_data["answers"].keys())
            if llm_texts:
                self.llm_vectorizer.fit(llm_texts)
                print(f"Добавлен ответ LLM в базу. Всего ответов: {len(self.llm_answers_data['answers'])}")
            
            return True
        return False

    def find_llm_answer(self, question: str, threshold: float = 0.4) -> Optional[str]:
        """Поиск похожего ответа в базе ответов LLM"""
        if not question.strip():
            return None
            
        clean_question = self._clean_text(question)
        answers = self.llm_answers_data.get("answers", {})
        
        if not answers:
            return None
        
        if clean_question in answers:
            print(f"Точное совпадение в базе ответов LLM: {clean_question}")
            return answers[clean_question]["answer"]
        
        try:
            questions = list(answers.keys())
            q_vec = self.llm_vectorizer.transform([clean_question])
            q_vecs = self.llm_vectorizer.transform(questions)
            similarities = cosine_similarity(q_vec, q_vecs)
            
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[0][best_match_idx]
            
            if best_similarity > threshold:
                best_question = questions[best_match_idx]
                print(f"Найдено похожий вопрос в базе LLM: {best_question} (схожесть: {best_similarity:.2f})")
                return answers[best_question]["answer"]
                
        except Exception as e:
            print(f"Ошибка поиска по схожести в базе LLM: {e}")
        
        return None

    def get_dialogue_response(self, text: str) -> Optional[str]:
        """Получение ответа из диалоговых шаблонов"""
        if not text.strip():
            return None
            
        text_lower = text.lower().strip()
        
        # Сначала проверяем, есть ли ответ в базе знаний предмета
        knowledge_answer = self.find_answer(text_lower)
        if knowledge_answer and not knowledge_answer.startswith("Интересный вопрос!"):
            print(f"Найден ответ в базе знаний: {knowledge_answer[:100]}...")
            return knowledge_answer
        
        # Затем проверяем базу ответов LLM
        llm_answer = self.find_llm_answer(text_lower)
        if llm_answer:
            print(f"Найден ответ в базе LLM: {llm_answer[:100]}...")
            return llm_answer
        
        # Затем проверяем диалоговые шаблоны
        for pattern, responses in self.dialogue_data.get("patterns", {}).items():
            if pattern in text_lower and responses:
                response = random.choice(responses)
                print(f"Найден диалоговый шаблон: {pattern} -> {response}")
                return response
        
        # Поиск по контекстам
        for context, responses in self.dialogue_data.get("contexts", {}).items():
            if context in text_lower and responses:
                response = random.choice(responses)
                print(f"Найден контекстный шаблон: {context} -> {response}")
                return response
                
        print(f"Ответ не найден для: {text_lower}")
        return None

    def _clean_text(self, text: str) -> str:
        """Очистка текста от знаков препинания и лишних пробелов"""
        if not text:
            return ""
            
        translator = str.maketrans('', '', string.punctuation + '«»„""')
        clean_text = text.translate(translator)
        
        return ' '.join(clean_text.lower().split())

    def find_answer(self, question: str, threshold: float = 0.3) -> Optional[str]:
        """Поиск ответа на вопрос в базе знаний"""
        if not question.strip():
            return None
            
        clean_question = self._clean_text(question)
        print(f"Поиск ответа для: '{clean_question}' в предмете {self.subject}")
        
        # 1. Точное совпадение с вопросами
        if clean_question in self.data["questions"]:
            print(f"Точное совпадение с вопросом: {clean_question}")
            return self.data["questions"][clean_question]
        
        # 2. Поиск по ключевым словам "что такое"
        if "что такое" in clean_question:
            term_part = clean_question.replace("что такое", "").strip()
            if term_part:
                print(f"Поиск термина после 'что такое': '{term_part}'")
                if term_part in self.data["terms"]:
                    print(f"Точное совпадение термина: {term_part}")
                    return self.data["terms"][term_part]
                
                for term, definition in self.data["terms"].items():
                    if term in term_part or term_part in term:
                        print(f"Частичное совпадение термина: {term}")
                        return definition
        
        # 3. Поиск по терминам (прямое вхождение)
        for term, definition in self.data["terms"].items():
            clean_term = self._clean_text(term)
            if clean_term and clean_term in clean_question:
                print(f"Термин найден в вопросе: {term}")
                return f"{term}: {definition}"
        
        # 4. Поиск по частичному совпадению терминов
        for term, definition in self.data["terms"].items():
            clean_term = self._clean_text(term)
            if clean_term:
                term_words = clean_term.split()
                if len(term_words) > 1:
                    matches = sum(1 for word in term_words if word in clean_question)
                    if matches >= max(1, len(term_words) - 1):
                        print(f"Частичное совпадение слов термина: {term}")
                        return f"{term}: {definition}"
        
        # 5. Поиск по схожести (TF-IDF + косинусная схожесть)
        if self.data["questions"]:
            questions = list(self.data["questions"].keys())
            try:
                q_vec = self.vectorizer.transform([clean_question])
                q_vecs = self.vectorizer.transform(questions)
                similarities = cosine_similarity(q_vec, q_vecs)
                
                best_match_idx = np.argmax(similarities)
                best_similarity = similarities[0][best_match_idx]
                
                if best_similarity > threshold:
                    best_question = questions[best_match_idx]
                    print(f"Найдено похожий вопрос по схожести: {best_question} (схожесть: {best_similarity:.2f})")
                    return self.data["questions"][best_question]
                    
            except Exception as e:
                print(f"Ошибка поиска по схожести: {e}")
        
        print("Ответ не найден в базе знаний")
        return "Интересный вопрос! Давайте обсудим его подробнее."

    def get_term(self, term: str) -> Optional[str]:
        """Получение определение термина"""
        term_lower = term.lower().strip()
        definition = self.data["terms"].get(term_lower)
        if definition:
            print(f"Найдено определение термина '{term}': {definition[:100]}...")
        else:
            print(f"Термин '{term}' не найден в базе знаний")
        return definition

    def add_knowledge(self, term: str = None, definition: str = None, 
                     question: str = None, answer: str = None):
        """Добавление знаний в базу"""
        if term and definition:
            self.data["terms"][term.lower()] = definition
            print(f"Добавлен термин: {term}")
            
        if question and answer:
            self.data["questions"][question.lower()] = answer
            print(f"Добавлен вопрос: {question}")
            
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
            print(f"База знаний сохранена в: {self.knowledge_path}")
        except Exception as e:
            print(f"Ошибка сохранения базы знаний: {e}")

    def _save_dialogue_knowledge(self):
        """Сохранение диалоговых шаблонов в файл"""
        try:
            with open(self.dialogue_path, 'w', encoding='utf-8') as f:
                json.dump(self.dialogue_data, f, ensure_ascii=False, indent=2)
            print(f"Диалоговые шаблоны сохранены в: {self.dialogue_path}")
        except Exception as e:
            print(f"Ошибка сохранения диалоговых шаблонов: {e}")

    def get_stats(self) -> Dict:
        """Получение статистики базы знаний"""
        return {
            "subject": self.subject,
            "terms_count": len(self.data["terms"]),
            "questions_count": len(self.data["questions"]),
            "examples_count": len(self.data["examples"]),
            "llm_answers_count": len(self.llm_answers_data.get("answers", {})),
            "dialogue_patterns": len(self.dialogue_data.get("patterns", {})),
            "dialogue_contexts": len(self.dialogue_data.get("contexts", {}))
        }

    def search_similar(self, query: str, max_results: int = 5) -> List[Dict]:
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
                    if similarity > 0.2:
                        results.append({
                            "type": "question",
                            "text": questions[i],
                            "answer": self.data["questions"][questions[i]],
                            "similarity": float(similarity)
                        })
            except Exception as e:
                print(f"Ошибка поиска похожих вопросов: {e}")
        
        # Поиск по ответам LLM
        llm_answers = self.llm_answers_data.get("answers", {})
        if llm_answers:
            llm_questions = list(llm_answers.keys())
            try:
                q_vec = self.llm_vectorizer.transform([query_lower])
                q_vecs = self.llm_vectorizer.transform(llm_questions)
                similarities = cosine_similarity(q_vec, q_vecs)
                
                for i, similarity in enumerate(similarities[0]):
                    if similarity > 0.3:
                        results.append({
                            "type": "llm_answer",
                            "text": llm_questions[i],
                            "answer": llm_answers[llm_questions[i]]["answer"],
                            "similarity": float(similarity),
                            "timestamp": llm_answers[llm_questions[i]]["timestamp"]
                        })
            except Exception as e:
                print(f"Ошибка поиска похожих ответов LLM: {e}")
        
        # Поиск по терминам
        for term, definition in self.data["terms"].items():
            if query_lower in term.lower() or any(word in term.lower() for word in query_lower.split()):
                results.append({
                    "type": "term",
                    "text": term,
                    "definition": definition,
                    "similarity": 0.8
                })
        
        # Сортировка по схожести
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:max_results]

    def list_terms(self) -> List[str]:
        """Возвращает список всех терминов в базе знаний"""
        return list(self.data["terms"].keys())

    def list_questions(self) -> List[str]:
        """Возвращает список всех вопросов в базе знаний"""
        return list(self.data["questions"].keys())

    def list_llm_answers(self) -> List[Dict]:
        """Возвращает список всех ответов LLM"""
        return [
            {"question": q, "answer": a["answer"], "timestamp": a["timestamp"]}
            for q, a in self.llm_answers_data.get("answers", {}).items()
        ]