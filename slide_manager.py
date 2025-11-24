import os
from pathlib import Path
import re

class SlideManager:
    def __init__(self, lessons_dir="lessons"):
        self.lessons_dir = Path(lessons_dir)
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    
    def get_slides_for_lesson(self, lesson_name):
        """Возвращает список слайдов для урока или None если их нет"""
        try:
            # Ищем файлы: названиеурока_01.jpg, названиеурока_02.jpg и т.д.
            pattern = f"{lesson_name}_[0-9][0-9]{'.*'}"
            slides = []
            
            for ext in self.supported_formats:
                slide_files = list(self.lessons_dir.glob(f"*{pattern}{ext}"))
                slide_files.extend(list(self.lessons_dir.glob(f"*{pattern}")))
                
                for slide_file in slide_files:
                    if self._is_slide_for_lesson(slide_file, lesson_name):
                        slides.append(slide_file)
            
            # Сортируем по номеру
            slides.sort(key=self._extract_slide_number)
            
            return slides if slides else None
            
        except Exception as e:
            print(f"Ошибка поиска слайдов для {lesson_name}: {e}")
            return None
    
    def _is_slide_for_lesson(self, slide_file, lesson_name):
        """Проверяет, относится ли файл к уроку"""
        filename = slide_file.stem.lower()
        lesson_pattern = lesson_name.lower().replace(' ', '_')
        
        # Проверяем форматы: lesson_01, lesson_01_extra, lesson_01_something
        pattern = rf"^{re.escape(lesson_pattern)}_(\d{{2}})(?:_.*)?$"
        return re.match(pattern, filename) is not None
    
    def _extract_slide_number(self, slide_file):
        """Извлекает номер слайда из文件名"""
        filename = slide_file.stem.lower()
        match = re.search(r'_(\d{2})', filename)
        return int(match.group(1)) if match else 0
    
    def get_slide_url(self, slide_file):
        """Возвращает URL для доступа к слайду"""
        return f"/slides/{slide_file.name}"

# Глобальный экземпляр
slide_manager = SlideManager()
