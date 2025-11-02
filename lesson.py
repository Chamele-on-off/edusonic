from pathlib import Path
import re
from typing import List, Optional, Dict
from llm import get_llm_instance  # Добавляем импорт LLM

class LessonManager:
    def __init__(self, lessons_dir: str = "lessons"):
        self.lessons_dir = Path(lessons_dir)
        self.current_lessons = {}
        self.llm = get_llm_instance()  # Добавляем LLM для генерации краткого содержания
        self._ensure_lessons_dir()
    
    def _ensure_lessons_dir(self):
        """Создает папку lessons если ее нет"""
        if not self.lessons_dir.exists():
            self.lessons_dir.mkdir(parents=True)
            # Создаем демо-урок по обществознанию
            demo_lesson = self.lessons_dir / "social_general.txt"
            if not demo_lesson.exists():
                with open(demo_lesson, 'w', encoding='utf-8') as f:
                    f.write("Основы обществознания: подготовка к ЕГЭ.\n\nДобро пожаловать на демо-урок! Сегодня мы разберем фундаментальные понятия обществознания.\n\nОбщество - это сложная динамическая система, объединяющая людей, которые связаны совместной деятельностью, общими интересами и ценностями.\n\nГосударство - это политическая организация общества, обладающая суверенитетом и аппаратом управления.\n\nДемократия - это форма правления, при которой народ является источником власти.\n\nЭкономика - это хозяйственная деятельность общества, система производства и распределения товаров.\n\nКультура - это совокупность достижений человечества в духовной и материальной жизни.\n\nПраво - это система общеобязательных норм, охраняемых государством.\n\nСоциализация - это процесс усвоения индивидом социальных норм и ценностей.\n\nЛичность - это человек как носитель социальных качеств и сознательной деятельности.\n\nМораль - это система норм и принципов, регулирующих поведение людей.\n\nГлобализация - это процесс всемирной экономической, политической и культурной интеграции.")
    
    def load_lesson_content(self, lesson_file: str) -> Dict:
        """Загружает содержание урока и генерирует краткое содержание"""
        try:
            lesson_path = self.lessons_dir / lesson_file
            if not lesson_path.exists():
                return {
                    "paragraphs": ["Урок не найден. Попробуйте выбрать другой урок."],
                    "summary": "Урок не найден"
                }
                
            with open(lesson_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Разбиваем на абзацы (по пустым строкам)
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            # Если абзацев нет, разбиваем на предложения
            if not paragraphs:
                sentences = re.split(r'(?<=[.!?])\s+', content)
                # Объединяем предложения в группы по 2-3 для плавного чтения
                current_paragraph = []
                paragraphs = []
                
                for sentence in sentences:
                    if sentence.strip():
                        current_paragraph.append(sentence.strip())
                        if len(current_paragraph) >= 2:  # Группируем по 2-3 предложения
                            paragraphs.append(' '.join(current_paragraph))
                            current_paragraph = []
                
                # Добавляем оставшиеся предложения
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
            
            # ГЕНЕРИРУЕМ КРАТКОЕ СОДЕРЖАНИЕ ДЛЯ ПРАКТИКИ
            lesson_summary = self._generate_lesson_summary(content, lesson_file)
            
            return {
                "paragraphs": paragraphs if paragraphs else ["Содержание урока временно недоступно."],
                "summary": lesson_summary,
                "original_content": content[:500]  # Сохраняем начало для fallback
            }
            
        except Exception as e:
            print(f"Ошибка загрузки урока {lesson_file}: {e}")
            return {
                "paragraphs": ["Ошибка загрузки урока. Попробуйте позже."],
                "summary": "Ошибка загрузки урока"
            }
    
    def _generate_lesson_summary(self, content: str, lesson_file: str) -> str:
        """Генерирует краткое содержание урока для практики"""
        try:
            # Если контент очень короткий, используем его как есть
            if len(content) < 200:
                return content
            
            # Определяем предмет из названия файла
            subject = self._detect_subject(lesson_file.replace('.txt', ''))
            
            prompt = f"""
            Создай очень краткое содержание урока для генерации практических вопросов.
            
            ПРЕДМЕТ: {subject}
            ПОЛНЫЙ ТЕКСТ УРОКА: {content[:1500]}...
            
            ТРЕБОВАНИЯ:
            - Очень кратко (максимум 150 слов)
            - Выдели только ключевые понятия и идеи
            - Убери вводные фразы и примеры
            - Сохрани основную суть для создания вопросов
            - Формат: простой текст без маркеров
            
            Верни только краткое содержание.
            """
            
            # Используем локальную модель для скорости
            summary = self.llm._query_llm_api(
                prompt=prompt,
                context="",
                subject=subject,
                system_prompt="Ты создаешь краткие содержания уроков для образовательных целей. Будь максимально лаконичным.",
                max_tokens=200
            )
            
            if summary and len(summary.strip()) > 50:
                print(f"✅ Сгенерировано краткое содержание урока: {len(summary)} символов")
                return summary.strip()
            else:
                # Fallback: используем начало контента
                fallback_summary = content[:300] + "..." if len(content) > 300 else content
                print(f"⚠️ Используется fallback содержание: {len(fallback_summary)} символов")
                return fallback_summary
                
        except Exception as e:
            print(f"❌ Ошибка генерации краткого содержания: {e}")
            # Fallback на начало контента
            return content[:250] + "..." if len(content) > 250 else content
    
    def get_available_lessons(self) -> dict:
        """Возвращает список доступных уроков"""
        lessons = {}
        try:
            for lesson_file in self.lessons_dir.glob("*.txt"):
                subject = self._detect_subject(lesson_file.stem)
                
                if subject not in lessons:
                    lessons[subject] = []
                
                lessons[subject].append({
                    'id': lesson_file.stem,
                    'title': lesson_file.stem.replace('_', ' ').title(),
                    'file_path': lesson_file.name,
                    'type': 'text'
                })
        except Exception as e:
            print(f"Ошибка получения списка уроков: {e}")
        
        return lessons
    
    def _detect_subject(self, filename: str) -> str:
        """Определяет предмет по названию файла"""
        filename_lower = filename.lower()
        if any(word in filename_lower for word in ['math', 'математика', 'алгебра', 'геометрия']):
            return "математика"
        elif any(word in filename_lower for word in ['history', 'история', 'истор']):
            return "история"
        elif any(word in filename_lower for word in ['physics', 'физика', 'физ']):
            return "физика"
        elif any(word in filename_lower for word in ['chemistry', 'химия', 'хим']):
            return "химия"
        elif any(word in filename_lower for word in ['social', 'обществознание', 'общество']):
            return "обществознание"
        elif any(word in filename_lower for word in ['biology', 'биология', 'био']):
            return "биология"
        elif any(word in filename_lower for word in ['literature', 'литература', 'лит']):
            return "литература"
        elif any(word in filename_lower for word in ['russian', 'русский', 'язык']):
            return "русский язык"
        else:
            return "общее"

# Создаем глобальный экземпляр менеджера уроков
lesson_manager = LessonManager()