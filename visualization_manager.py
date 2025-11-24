import os
import re
from pathlib import Path
from typing import Dict, Optional, List
import time
from llm import LLMIntegration

class VisualizationManager:
    def __init__(self, lessons_dir: str = "lessons"):
        self.lessons_dir = Path(lessons_dir)
        self.llm = LLMIntegration()
        
    def get_lesson_slides(self, lesson_id: str) -> Optional[List[str]]:
        """Проверяет наличие слайдов для урока и возвращает их список"""
        lesson_name = lesson_id.lower()
        slides_dir = self.lessons_dir / "slides"
        
        if not slides_dir.exists():
            return None
            
        # Ищем файлы: названиеурока_01.jpg, названиеурока_02.jpg и т.д.
        slide_pattern = f"{lesson_name}_*.jpg"
        slide_files = list(slides_dir.glob(slide_pattern))
        
        if not slide_files:
            # Пробуем другие расширения
            for ext in ['png', 'jpeg', 'webp']:
                slide_pattern = f"{lesson_name}_*.{ext}"
                slide_files.extend(list(slides_dir.glob(slide_pattern)))
        
        if slide_files:
            # Сортируем по номеру
            slide_files.sort(key=lambda x: self._extract_slide_number(x.name))
            return [str(slide) for slide in slide_files]
        
        return None
    
    def _extract_slide_number(self, filename: str) -> int:
        """Извлекает номер слайда из имени файла"""
        match = re.search(r'_(\d+)\.', filename)
        return int(match.group(1)) if match else 0
    
    def generate_lesson_mindmap(self, lesson_content: str, lesson_title: str) -> Dict:
        """Генерирует mind map всего урока один раз за урок"""
        try:
            prompt = f"""
            Создай структурную mind map (интеллект-карту) для урока на тему: "{lesson_title}"

            СОДЕРЖАНИЕ УРОКА:
            {lesson_content[:2000]}  # Ограничиваем длину для экономии токенов

            ТРЕБОВАНИЯ К MIND MAP:
            1. Основная тема в центре: "{lesson_title}"
            2. 3-5 основных разделов (первого уровня)
            3. 2-3 подраздела для каждого основного раздела (второго уровня)
            4. Максимальная глубина: 2 уровня
            5. Используй четкие и краткие формулировки
            6. Логическая структура от общего к частному

            Формат Mermaid:
            flowchart TD
                A["Основная тема"] --> B["Раздел 1"]
                A --> C["Раздел 2"] 
                B --> D["Подраздел 1.1"]
                B --> E["Подраздел 1.2"]
                C --> F["Подраздел 2.1"]

            Верни ТОЛЬКО код Mermaid без пояснений.
            """
            
            system_prompt = """Ты создаешь структурные mind maps для образовательных уроков. 
            Создавай четкие логические структуры с ограниченным количеством элементов для лучшей читаемости."""
            
            mermaid_code = self.llm._query_llm_api(
                prompt=prompt,
                context="",
                subject="general",
                system_prompt=system_prompt,
                max_tokens=500
            )
            
            if mermaid_code:
                cleaned_code = self._clean_mermaid_code(mermaid_code, lesson_title)
                return {
                    "type": "mindmap",
                    "mermaid_code": cleaned_code,
                    "lesson_title": lesson_title,
                    "timestamp": time.time()
                }
            
        except Exception as e:
            print(f"❌ Ошибка генерации mind map: {e}")
        
        # Fallback mind map
        return {
            "type": "mindmap",
            "mermaid_code": self._create_fallback_mindmap(lesson_title),
            "lesson_title": lesson_title,
            "timestamp": time.time()
        }
    
    def _clean_mermaid_code(self, code: str, lesson_title: str) -> str:
        """Очистка и валидация Mermaid кода"""
        if not code:
            return self._create_fallback_mindmap(lesson_title)
        
        # Удаляем markdown обратные кавычки
        code = re.sub(r'```[\s\S]*?```', '', code)
        code = re.sub(r'`', '', code)
        
        # Убеждаемся, что это корректный Mermaid
        if not code.strip().startswith(('flowchart', 'graph')):
            code = 'flowchart TD\n' + code
        
        # Добавляем основную тему если ее нет
        if f'["{lesson_title}"]' not in code and '["Основная тема"]' not in code:
            lines = code.split('\n')
            if len(lines) > 1:
                lines[0] = f'    A["{lesson_title}"]'
            code = '\n'.join(lines)
        
        return code.strip()
    
    def _create_fallback_mindmap(self, lesson_title: str) -> str:
        """Создает простую mind map по умолчанию"""
        return f'''flowchart TD
    A["{lesson_title}"] --> B["Основные понятия"]
    A --> C["Ключевые аспекты"]
    A --> D["Практическое применение"]
    B --> E["Определения"]
    B --> F["Примеры"]
    C --> G["Теория"]
    C --> H["Факты"]
    D --> I["Задачи"]
    D --> J["Упражнения"]'''
    
    def should_generate_visualization(self, lesson_id: str, paragraph_index: int) -> bool:
        """Определяет, когда генерировать визуализацию"""
        # Mind map генерируется только один раз за урок - в начале
        return paragraph_index == 0
    
    def get_visualization_type(self, lesson_id: str) -> str:
        """Определяет тип визуализации: слайды или mindmap"""
        slides = self.get_lesson_slides(lesson_id)
        return "slides" if slides else "mindmap"

# Глобальный экземпляр менеджера визуализации
visualization_manager = VisualizationManager()
