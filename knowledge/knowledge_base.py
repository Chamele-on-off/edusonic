import json
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import string
import re

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
                "как дела": ["Все хорошо, продолжим урок", "Отлично, давайте заниматься", "Прекрасно! А у вас?"],
                "не понял": ["Давайте разберем еще раз", "Попробую объяснить по-другому", "Хорошо, объясню подробнее"],
                "повтори": ["Конечно, повторяю...", "Еще раз:", "Повторяю для вас"],
                "спасибо": ["Пожалуйста!", "Всегда рад помочь!", "Рад был помочь!"],
                "скучно": ["Давайте сделаем урок интереснее!", "Предлагаю викторину!", "Сменим активность?"],
                "трудно": ["Не переживай! Вместе разберемся", "Сложности - это нормально!", "Я помогу тебе"],
                "молодец": ["Спасибо! Стараюсь для вас", "Рад, что нравится!", "Вы тоже молодец!"],
                "что такое": ["Расскажу подробнее об этом понятии", "Объясню этот термин"],
                "объясни": ["С удовольствием объясню", "Давайте разберем этот вопрос"],
                "расскажи": ["Расскажу подробнее", "Поделюсь информацией об этом"],
                "зачем учить": ["Это важно для понимания мира вокруг нас!", "Знания помогут в будущем!", "Это интересно и полезно!"],
                "почему важно": ["Это основа для многих других знаний!", "Помогает развивать мышление!", "Пригодится в жизни!"]
            },
            "contexts": {
                "greeting": ["Привет!", "Здравствуйте!", "Рад вас видеть!"],
                "farewell": ["До свидания!", "Удачи!", "До следующего урока!"],
                "encouragement": ["Отлично!", "Прекрасно!", "Так держать!"],
                "question": ["Интересный вопрос!", "Хорошо, что спрашиваете!", "Отличный вопрос!"]
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
                with open(self.knowledge_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Ошибка загрузки базы знаний {self.subject}: {e}")
        
        # База по умолчанию
        default_knowledge = {
            "terms": {},
            "questions": {},
            "examples": {},
            "metadata": {
                "subject": self.subject,
                "version": "1.0",
                "last_updated": "2024-01-01"
            }
        }
        
        # Добавляем базовые знания для обществознания
        if self.subject == "обществознание":
            default_knowledge["terms"] = {
                "общество": "Группа людей, объединенных общей территорией, культурой и социальной структурой.",
                "государство": "Политическая организация общества, обладающая суверенитетом и аппаратом управления.",
                "демократия": "Форма правления, при которой народ является источником власти.",
                "экономика": "Хозяйственная деятельность общества, система производства и распределения товаров.",
                "культура": "Совокупность достижений человечества в духовной и материальной жизни.",
                "право": "Система общеобязательных норм, охраняемых государством.",
                "социализация": "Процесс усвоения индивидом социальных норм и ценностей.",
                "личность": "Человек как носитель социальных качеств и сознательной деятельности.",
                "мораль": "Система норм и принципов, регулирующих поведение людей.",
                "глобализация": "Процесс всемирной экономической, политической и культурной интеграции."
            }
            default_knowledge["questions"] = {
                "что такое общество": "Общество - это устойчивая группа людей, имеющая общую территорию, культуру, экономику и социальную структуру.",
                "какие функции у государства": "Основные функции государства: защита территории, управление экономикой, социальная защита, поддержание правопорядка.",
                "что изучает обществознание": "Обществознание изучает общество, социальные отношения, политику, экономику, право и духовную жизнь.",
                "зачем нужно обществознание": "Обществознание помогает понимать общественные процессы, права и обязанности гражданина, развивает критическое мышление.",
                "что такое демократия": "Демократия - это форма правления, где власть принадлежит народу и осуществляется через выборы и референдумы.",
                "как устроено общество": "Общество состоит из социальных институтов: семьи, государства, экономики, образования, религии и культуры.",
                "что такое право": "Право - система общеобязательных норм, установленных государством и охраняемых его принудительной силой.",
                "какие бывают экономические системы": "Основные экономические системы: традиционная, командная, рыночная и смешанная.",
                "что такое социальная стратификация": "Социальная стратификация - расслоение общества на группы по доходу, власти, образованию и престижу.",
                "как происходит социализация": "Социализация происходит через семью, школу, СМИ, группу сверстников и другие социальные институты."
            }
        
        return default_knowledge

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
                "наука система знаний",
                "математика наука о числах",
                "физика наука о природе",
                "химия наука о веществах",
                "история наука о прошлом",
                "литература искусство слова",
                "биология наука о жизни",
                "география наука о земле"
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

    def _clean_text(self, text: str) -> str:
        """Очистка текста от знаков препинания и лишних пробелов"""
        if not text:
            return ""
            
        # Удаляем знаки препинания
        translator = str.maketrans('', '', string.punctuation + '«»„""')
        clean_text = text.translate(translator)
        
        # Удаляем лишние пробелы и приводим к нижнему регистру
        return ' '.join(clean_text.lower().split())

    def find_answer(self, question: str, threshold: float = 0.2) -> Optional[str]:
        """Поиск ответа на вопрос в базе знаний"""
        if not question.strip():
            return None
            
        clean_question = self._clean_text(question)
        
        # 1. Точное совпадение с вопросами
        if clean_question in self.data["questions"]:
            return self.data["questions"][clean_question]
        
        # 2. Поиск по ключевым словам вопросов
        question_keywords = ["что такое", "определение", "объясни", "расскажи", "поясни", "какие", "что значит", "кто такой", "зачем", "почему"]
        if any(keyword in clean_question for keyword in question_keywords):
            # Ищем все термины в вопросе
            found_terms = []
            for term in self.data["terms"].keys():
                clean_term = self._clean_text(term)
                if clean_term and clean_term in clean_question:
                    found_terms.append((term, self.data["terms"][term]))
            
            if found_terms:
                # Возвращаем все найденные термины
                response = "В вашем вопросе упоминаются следующие понятия:\n"
                for term, definition in found_terms:
                    response += f"• {term}: {definition}\n"
                return response
        
        # 3. Поиск по терминам (прямое вхождение) - ищем ВСЕ термины
        found_terms = []
        for term, definition in self.data["terms"].items():
            clean_term = self._clean_text(term)
            if clean_term and clean_term in clean_question:
                found_terms.append((term, definition))
        
        if found_terms:
            response = "По вашему вопросу:\n"
            for term, definition in found_terms:
                response += f"• {term}: {definition}\n"
            return response
        
        # 4. Поиск по частичному совпадению терминов
        for term, definition in self.data["terms"].items():
            clean_term = self._clean_text(term)
            if clean_term:
                # Разбиваем термин на слова и проверяем наличие каждого слова в вопросе
                term_words = clean_term.split()
                if len(term_words) > 1:
                    matches = sum(1 for word in term_words if word in clean_question)
                    if matches >= max(1, len(term_words) - 1):
                        return f"{term}: {definition}"
        
        # 5. Поиск по схожести (TF-IDF + косинусная схожесть)
        if self.data["questions"]:
            questions = list(self.data["questions"].keys())
            try:
                q_vec = self.vectorizer.transform([clean_question])
                q_vecs = self.vectorizer.transform(questions)
                similarities = cosine_similarity(q_vec, q_vecs)
                
                # Находим несколько лучших совпадений
                best_matches = []
                for i, similarity in enumerate(similarities[0]):
                    if similarity > threshold:
                        best_matches.append((questions[i], similarity))
                
                # Сортируем по убыванию схожести
                best_matches.sort(key=lambda x: x[1], reverse=True)
                
                if best_matches:
                    # Возвращаем несколько лучших ответов
                    response = "Возможно, вы имели в виду:\n"
                    for question_text, similarity in best_matches[:2]:
                        answer = self.data["questions"][question_text]
                        response += f"• {answer}\n"
                    return response
                    
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