import os
import json
from typing import List, Dict, Optional
from pathlib import Path
import sqlite3
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RAGDatabase:
    def __init__(self, db_path: str = "materials/materials.db"):
        self.db_path = db_path
        self._init_db()
        self.embedding_model = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        
    def _init_db(self):
        """Инициализация базы данных для RAG"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS materials (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT,
                    title TEXT,
                    content TEXT,
                    embedding BLOB
                )
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS materials_fts 
                USING fts5(subject, title, content)
            """)
            conn.commit()

    def _load_embedding_model(self):
        """Ленивая загрузка модели для эмбеддингов"""
        if self.embedding_model is None:
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            logger.info("Embedding model loaded")

    def add_material(self, subject: str, title: str, content: str):
        """Добавление материала в базу с индексированием"""
        self._load_embedding_model()
        embedding = self._get_embedding(content)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO materials (subject, title, content, embedding) VALUES (?, ?, ?, ?)",
                (subject, title, content, embedding.tobytes())
            )
            cursor.execute(
                "INSERT INTO materials_fts (subject, title, content) VALUES (?, ?, ?)",
                (subject, title, content)
            )
            conn.commit()

    def _get_embedding(self, text: str) -> np.ndarray:
        """Генерация embedding для текста"""
        self._load_embedding_model()
        return self.embedding_model.encode(text, convert_to_numpy=True)

    def search_relevant_materials(self, query: str, subject: str, top_k: int = 3) -> List[Dict]:
        """Поиск релевантных материалов по семантическому сходству"""
        self._load_embedding_model()
        query_embedding = self._get_embedding(query)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Полнотекстовый поиск по subject
            cursor.execute(
                "SELECT * FROM materials WHERE subject = ?",
                (subject,)
            )
            materials = cursor.fetchall()
            
            # Вычисление схожести
            results = []
            for row in materials:
                emb = np.frombuffer(row['embedding'], dtype=np.float32)
                similarity = F.cosine_similarity(
                    torch.from_numpy(query_embedding).unsqueeze(0),
                    torch.from_numpy(emb).unsqueeze(0)
                ).item()
                
                results.append({
                    'id': row['id'],
                    'subject': row['subject'],
                    'title': row['title'],
                    'content': row['content'],
                    'similarity': similarity
                })
            
            # Сортировка по убыванию схожести
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            return results[:top_k]

class LessonLLM:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rag = RAGDatabase()
        self.model = None
        self.tokenizer = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        logger.info(f"LLM initialized (device: {self.device})")

    def _load_model(self):
        """Загрузка модели и токенизатора"""
        if self.model is not None and self.tokenizer is not None:
            return
            
        logger.info(f"Loading model {self.model_name}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                trust_remote_code=True
            )
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def generate_response(
        self,
        question: str,
        context: Dict,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Генерация ответа с учетом контекста урока и предметных материалов
        
        Args:
            question: Вопрос от ученика
            context: {
                "lesson": "Название урока",
                "subject": "Предмет",
                "current_phase": "Текущая фаза урока"
            }
            max_length: Максимальная длина ответа
            temperature: Параметр креативности
            top_p: Параметр diversity
            
        Returns:
            Сгенерированный ответ
        """
        self._load_model()
        
        try:
            # Поиск релевантных материалов
            materials = self.rag.search_relevant_materials(question, context['subject'])
            context_str = self._build_context_string(context, materials)
            
            # Формирование промта
            prompt = self._build_prompt(question, context_str)
            
            # Генерация ответа
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return self._postprocess_response(response, prompt)
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "Извините, возникла ошибка при обработке вашего вопроса."

    def _build_context_string(self, context: Dict, materials: List[Dict]) -> str:
        """Формирование строки контекста из материалов"""
        context_parts = [
            f"Текущий урок: {context['lesson']}",
            f"Тема: {context['subject']}",
            f"Текущая фаза: {context['current_phase']}"
        ]
        
        if materials:
            context_parts.append("\nРелевантные материалы:")
            for i, material in enumerate(materials, 1):
                context_parts.append(
                    f"{i}. {material['title']} (схожесть: {material['similarity']:.2f}):\n"
                    f"{material['content'][:500]}..."
                )
        
        return "\n".join(context_parts)

    def _build_prompt(self, question: str, context: str) -> str:
        """Формирование промта для LLM"""
        return f"""Ты - учитель-ассистент. Ответь на вопрос ученика, используя предоставленный контекст.

Контекст:
{context}

Вопрос: {question}

Ответь подробно и понятно, используя только факты из контекста. Если информации недостаточно, скажи об этом.
Ответ:"""

    def _postprocess_response(self, response: str, prompt: str) -> str:
        """Постобработка сгенерированного ответа"""
        # Удаляем промт из ответа
        if prompt in response:
            response = response[len(prompt):]
        
        # Удаляем повторения
        sentences = response.split('. ')
        unique_sentences = []
        seen_sentences = set()
        
        for sent in sentences:
            clean_sent = sent.strip()
            if clean_sent and clean_sent not in seen_sentences:
                seen_sentences.add(clean_sent)
                unique_sentences.append(clean_sent)
        
        return '. '.join(unique_sentences) + ('' if not unique_sentences else '.')

    def load_materials_from_dir(self, dir_path: str):
        """Загрузка материалов уроков из директории"""
        dir_path = Path(dir_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory {dir_path} not found")
        
        for file_path in dir_path.glob("**/*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.rag.add_material(
                        subject=data['subject'],
                        title=data['title'],
                        content=data['content']
                    )
                logger.info(f"Loaded material from {file_path}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")

# Пример использования
if __name__ == "__main__":
    # Инициализация модели
    llm = LessonLLM()
    
    # Загрузка материалов (если нужно)
    llm.load_materials_from_dir("materials/math")
    
    # Пример генерации ответа
    context = {
        "lesson": "Введение в алгебру",
        "subject": "math",
        "current_phase": "explanation"
    }
    
    question = "Что такое квадратное уравнение?"
    response = llm.generate_response(question, context)
    
    print("Question:", question)
    print("Response:", response)