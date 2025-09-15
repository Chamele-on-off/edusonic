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
        self.extended_dialogue_path = Path("knowledge/dialogue_knowledge.json")
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words=['и', 'в', 'на', 'с', 'по', 'для', 'что', 'это'])
        self.llm_vectorizer = TfidfVectorizer(max_features=1000, stop_words=['и', 'в', 'на', 'с', 'по', 'для', 'что', 'это'])
        self.data = self._load_knowledge()
        self.llm_answers_data = self._load_llm_answers()
        self.dialogue_data = self._load_dialogue_knowledge()
        self.extended_dialogue_data = self._load_extended_dialogue_knowledge()
        self._init_vectorizers()
        
    def _load_extended_dialogue_knowledge(self) -> Dict:
        """Загрузка расширенной базы диалоговых шаблонов"""
        try:
            if self.extended_dialogue_path.exists():
                with open(self.extended_dialogue_path, 'r', encoding='utf-8') as f:
                    print(f"✅ Загружена расширенная база диалоговых шаблонов")
                    return json.load(f)
        except Exception as e:
            print(f"❌ Ошибка загрузки расширенных диалоговых шаблонов: {e}")
        
        print("⚠️ Расширенная база диалоговых шаблонов не найдена, используется базовая")
        return {}

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

    def add_to_dialogue_knowledge(self, question: str, answer: str) -> bool:
        """Добавляет вопрос-ответ в диалоговую базу знаний"""
        try:
            dialogue_path = Path("materials/dialogue_knowledge.json")
            if dialogue_path.exists():
                with open(dialogue_path, 'r', encoding='utf-8') as f:
                    dialogue_data = json.load(f)
            else:
                dialogue_data = {"patterns": {}, "metadata": {"version": "1.0", "last_updated": datetime.now().isoformat()}}
            
            # Создаем ключ для вопроса (ограничиваем длину)
            question_key = question.lower().strip()[:100]
            
            # Добавляем или обновляем ответ
            if question_key not in dialogue_data["patterns"]:
                dialogue_data["patterns"][question_key] = []
            
            # Добавляем ответ если его еще нет
            if answer not in dialogue_data["patterns"][question_key]:
                dialogue_data["patterns"][question_key].append(answer)
            
            dialogue_data["metadata"]["last_updated"] = datetime.now().isoformat()
            
            # Сохраняем
            with open(dialogue_path, 'w', encoding='utf-8') as f:
                json.dump(dialogue_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Ответ сохранен в диалоговую базу знаний: {question_key}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка сохранения в диалоговую базу: {e}")
            return False

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

    def find_llm_answer(self, question: str, threshold: float = 0.8) -> Optional[str]:
        """Поиск похожего ответа в базе ответов LLM с высокой точностью"""
        if not question.strip():
            return None
            
        clean_question = self._clean_text(question)
        answers = self.llm_answers_data.get("answers", {})
        
        if not answers:
            return None
        
        # 1. Точное совпадение (самый надежный способ)
        if clean_question in answers:
            print(f"💾 Точное совпадение в базе ответов LLM: {clean_question}")
            return answers[clean_question]["answer"]
        
        # 2. Поиск по очень высокой схожести (только для уверенных совпадений)
        try:
            questions = list(answers.keys())
            q_vec = self.llm_vectorizer.transform([clean_question])
            q_vecs = self.llm_vectorizer.transform(questions)
            similarities = cosine_similarity(q_vec, q_vecs)
            
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[0][best_match_idx]
            
            if best_similarity > threshold:
                best_question = questions[best_match_idx]
                print(f"💾 Найдено очень похожий вопрос в базе LLM: {best_question} (схожесть: {best_similarity:.3f})")
                return answers[best_question]["answer"]
                
        except Exception as e:
            print(f"Ошибка поиска по схожести в базе LLM: {e}")
        
        return None

    def get_dialogue_response(self, text: str, context: List[str] = None) -> Optional[str]:
        """Получение ответа на основе диалоговых шаблонов с учетом контекста"""
        if not text.strip():
            return None
            
        text_lower = text.lower().strip()
        
        # 1. Сначала проверяем расширенную базу диалоговых шаблонов
        if self.extended_dialogue_data:
            for category, patterns in self.extended_dialogue_data.items():
                if category.endswith('_patterns') and isinstance(patterns, dict):
                    for pattern, responses in patterns.items():
                        if pattern in text_lower and responses:
                            response = random.choice(responses)
                            print(f"📖 Найден в расширенных шаблонах: {pattern} -> {response}")
                            return response
        
        # 2. Проверяем базовую базу диалоговых шаблонов
        for pattern, responses in self.dialogue_data.get("patterns", {}).items():
            if pattern in text_lower and responses:
                response = random.choice(responses)
                print(f"📖 Найден в базовых шаблонах: {pattern} -> {response}")
                return response
        
        # 3. Контекстный поиск (если предоставлен контекст)
        if context:
            context_text = ' '.join(context).lower()
            contextual_patterns = self.extended_dialogue_data.get('contextual_patterns', {}) or self.dialogue_data.get('contexts', {})
            
            for pattern, responses in contextual_patterns.items():
                if pattern in context_text and responses:
                    response = random.choice(responses)
                    print(f"📖 Найден контекстный шаблон: {pattern} -> {response}")
                    return response
        
        # 4. Проверяем базу знаний предмета
        knowledge_response = self.find_answer(text_lower)
        if knowledge_response and not knowledge_response.startswith("Интересный вопрос!"):
            print(f"📚 Найден ответ в базе знаний: {knowledge_response[:100]}...")
            return knowledge_response
        
        # 5. Проверяем базу ответов LLM с высокой точностью
        llm_answer = self.find_llm_answer(text_lower, threshold=0.8)
        if llm_answer:
            print(f"💾 Найден ответ в базе LLM: {llm_answer[:100]}...")
            return llm_answer
            
        print(f"❌ Ответ не найден в диалоговых шаблонах для: {text_lower}")
        return None

    def _clean_text(self, text: str) -> str:
        """Очистка текста от знаков препинания и лишних пробелов"""
        if not text:
            return ""
            
        translator = str.maketrans('', '', string.punctuation + '«»„""')
        clean_text = text.translate(translator)
        
        return ' '.join(clean_text.lower().split())

    def _extract_term_from_question(self, question: str) -> Optional[str]:
        """Извлекает термин из вопроса типа 'что такое X'"""
        patterns = [
            r'что такое (.+?)\??$',
            r'что значит (.+?)\??$',
            r'определение (.+?)\??$',
            r'объясни (.+?)\??$',
            r'расскажи про (.+?)\??$',
            r'разъясни (.+?)\??$',
            r'кто такой (.+?)\??$',
            r'кто такая (.+?)\??$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question.lower())
            if match:
                term = match.group(1).strip()
                if term:
                    return term
        return None

    def _remove_question_words(self, text: str) -> str:
        """Удаляет вопросительные слова из текста"""
        question_words = [
            'что такое', 'что значит', 'объясни', 'расскажи про', 
            'разъясни', 'кто такой', 'кто такая', 'как работает',
            'что это', 'что означает'
        ]
        
        result = text.lower()
        for word in question_words:
            result = result.replace(word, '').strip()
        
        # Удаляем знаки вопроса и лишние пробелы
        result = result.replace('?', '').strip()
        result = re.sub(r'\s+', ' ', result)
        
        return result

    def find_answer(self, question: str, threshold: float = 0.5) -> Optional[str]:
        """Поиск ответа на вопрос в базе знаний"""
        if not question.strip():
            return None
            
        clean_question = self._clean_text(question)
        print(f"Поиск ответа для: '{clean_question}' в предмете {self.subject}")
        
        # 1. Точное совпадение с вопросами
        if clean_question in self.data["questions"]:
            print(f"Точное совпадение с вопросом: {clean_question}")
            return self.data["questions"][clean_question]
        
        # 2. Удаляем вопросительные слова и ищем чистый термин
        clean_term_query = self._remove_question_words(question)
        if clean_term_query and clean_term_query != clean_question:
            print(f"Очищенный запрос от вопросительных слов: '{clean_term_query}'")
            
            # Проверяем точное совпадение очищенного термина
            if clean_term_query in self.data["terms"]:
                print(f"Точное совпадение очищенного термина: {clean_term_query}")
                return self.data["terms"][clean_term_query]
            
            # Проверяем частичные совпадения очищенных терминов
            for term, definition in self.data["terms"].items():
                clean_term = self._clean_text(term)
                if clean_term_query == clean_term:
                    print(f"Точное совпадение после очистки: {term}")
                    return definition
        
        # 3. Извлечение термина из вопроса типа "что такое X"
        extracted_term = self._extract_term_from_question(question)
        if extracted_term:
            print(f"Извлечен термин из вопроса: '{extracted_term}'")
            
            # Проверяем точное совпадение термина
            if extracted_term in self.data["terms"]:
                print(f"Точное совпадение термина: {extracted_term}")
                return self.data["terms"][extracted_term]
            
            # Проверяем частичные совпадения терминов
            clean_extracted_term = self._clean_text(extracted_term)
            for term, definition in self.data["terms"].items():
                clean_term = self._clean_text(term)
                if clean_extracted_term == clean_term:
                    print(f"Точное совпадение после очистки: {term}")
                    return definition
        
        # 4. Поиск по терминам (прямое вхождение)
        for term, definition in self.data["terms"].items():
            clean_term = self._clean_text(term)
            if clean_term and clean_term in clean_question:
                print(f"Термин найден в вопросе: {term}")
                return f"{term}: {definition}"
        
        # 5. Поиск по частичному совпадению терминов (только для многословных терминов)
        for term, definition in self.data["terms"].items():
            clean_term = self._clean_text(term)
            if clean_term:
                term_words = clean_term.split()
                if len(term_words) > 1:
                    matches = sum(1 for word in term_words if word in clean_question)
                    # Требуем совпадения всех слов термина
                    if matches == len(term_words):
                        print(f"Полное совпадение слов термина: {term}")
                        return f"{term}: {definition}"
        
        # 6. Поиск по схожести (TF-IDF + косинусная схожесть) - только для вопросов
        if self.data["questions"]:
            questions = list(self.data["questions"].keys())
            try:
                q_vec = self.vectorizer.transform([clean_question])
                q_vecs = self.vectorizer.transform(questions)
                similarities = cosine_similarity(q_vec, q_vecs)
                
                # Находим лучшее совпадение
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
            "dialogue_contexts": len(self.dialogue_data.get("contexts", {})),
            "extended_dialogue_patterns": sum(len(patterns) for category, patterns in self.extended_dialogue_data.items() 
                                           if category.endswith('_patterns') and isinstance(patterns, dict))
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