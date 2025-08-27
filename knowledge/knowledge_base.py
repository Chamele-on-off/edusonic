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
                "общество": "Общество - это сложная динамическая система, объединяющая людей, которые связаны совместной деятельностью, общими интересами и ценностями. Общество характеризуется целостностью, саморегуляцией, открытостью и способностью к развитию.",
                "государство": "Государство - это политическая организация общества, обладающая суверенитетом, специальным аппаратом управления и принуждения, а также устанавливающая правовой порядок на определенной территории.",
                "демократия": "Демократия - это форма правления, при которой народ является источником власти. Основные принципы демократии: выборность органов власти, разделение властей, верховенство права и защита прав человека.",
                "экономика": "Экономика - это хозяйственная деятельность общества, а также совокупность отношений, складывающихся в системе производства, распределения, обмена и потребления товаров и услуг.",
                "культура": "Культура - это совокупность достижений человечества в производственной, общественной и духовной жизни. Культура включает в себя знания, искусство, мораль, законы, обычаи и другие способности и привычки, приобретенные человеком как членом общества.",
                "право": "Право - это система общеобязательных норм, установленных или санкционированных государством и охраняемых его принудительной силой. Право регулирует общественные отношения и обеспечивает порядок в обществе.",
                "социализация": "Социализация - это процесс усвоения индивидом социальных норм, ценностей, знаний и навыков, необходимых для успешного функционирования в обществе. Социализация происходит throughout всей жизни человека.",
                "личность": "Личность - это человек как носитель социальных качеств и сознательной деятельности. Личность формируется в процессе социализации и характеризуется уникальным сочетанием социально значимых черт.",
                "мораль": "Мораль - это система норм и принципов, регулирующих поведение людей с позиций добра и зла, справедливости и несправедливости. Мораль основывается на общечеловеческих ценностях и традициях.",
                "глобализация": "Глобализация - это процесс всемирной экономической, политической и культурной интеграции и унификации. Глобализация проявляется в расширении международной торговли, увеличении миграции населения и распространении информационных технологий."
            }
            default_knowledge["questions"] = {
                "что такое общество": "Общество - это сложная динамическая система, объединяющая людей, которые связаны совместной деятельностью, общими интересами и ценностями. Общество характеризуется целостностью, саморегуляцией, открытостью и способностью к развитию.",
                "что такое государство": "Государство - это политическая организация общества, обладающая суверенитетом, специальным аппаратом управления и принуждения, а также устанавливающая правовой порядок на определенной территории.",
                "что такое демократия": "Демократия - это форма правления, при которой народ является источником власти. Основные принципы демократии: выборность органов власти, разделение властей, верховенство права и защита прав человека.",
                "что такое экономика": "Экономика - это хозяйственная деятельность общества, а также совокупность отношений, складывающихся в системе производства, распределения, обмена и потребления товаров и услуг.",
                "что такое культура": "Культура - это совокупность достижений человечества в производственной, общественной и духовной жизни. Культура включает знания, искусство, мораль, законы, обычаи и другие способности человека.",
                "что такое право": "Право - это система общеобязательных норм, установленных или санкционированных государством и охраняемых его принудительной силой. Право регулирует общественные отношения.",
                "что такое социализация": "Социализация - это процесс усвоения индивидом социальных норм, ценностей, знаний и навыков, необходимых для успешного функционирования в обществе.",
                "что такое личность": "Личность - это человек как носитель социальных качеств и сознательной деятельности. Личность формируется в процессе социализации.",
                "что такое мораль": "Мораль - это система норм и принципов, регулирующих поведение людей с позиций добра и зла, справедливости и несправедливости.",
                "что такое глобализация": "Глобализация - это процесс всемирной экономической, политической и культурной интеграции и унификации.",
                "какие функции у государства": "Основные функции государства: защита территории и суверенитета, управление экономикой, социальная защита населения, поддержание правопорядка, осуществление правосудия и международное сотрудничество.",
                "что изучает обществознание": "Обществознание изучает общество, социальные отношения, политику, экономику, право и духовную жизнь. Это комплексная наука, помогающая понять законы развития общества и место человека в нем.",
                "зачем нужно обществознание": "Обществознание помогает понимать общественные процессы, права и обязанности гражданина, развивает критическое мышление и формирует активную гражданскую позицию. Знание обществознания необходимо для успешной социализации и участия в общественной жизни.",
                "как устроено общество": "Общество состоит из социальных институтов: семьи, государства, экономики, образования, религии и культуры. Эти институты взаимосвязаны и выполняют определенные функции в обществе. Общество также делится на социальные группы, слои и классы.",
                "какие бывают экономические системы": "Основные экономические системы: традиционная (основана на обычаях и традициях), командная (централизованное планирование), рыночная (спрос и предложение) и смешанная (сочетание рыночных механизмов и государственного регулирования).",
                "что такое социальная стратификация": "Социальная стратификация - это расслоение общества на группы по различным критериям: доход, власть, образование и престиж. Основные формы стратификации: рабство, касты, сословия и классы.",
                "как происходит социализация": "Социализация происходит через агентов социализации: семью, школу, СМИ, группу сверстников и другие социальные институты. Социализация включает два этапа: первичную (в детстве) и вторичную (в течение всей жизни)."
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
        
        # 2. Поиск по ключевым словам "что такое"
        if "что такое" in clean_question:
            # Извлекаем термин после "что такое"
            term = clean_question.replace("что такое", "").strip()
            if term in self.data["terms"]:
                return self.data["terms"][term]
            
            # Поиск похожих терминов
            for known_term in self.data["terms"].keys():
                if known_term in term or term in known_term:
                    return self.data["terms"][known_term]
        
        # 3. Поиск по терминам (прямое вхождение)
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
