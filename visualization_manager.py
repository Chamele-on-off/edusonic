import os
import re
from pathlib import Path
from typing import Dict, Optional, List
import time
import threading
from llm import LLMIntegration

class VisualizationManager:
    def __init__(self, lessons_dir: str = "lessons"):
        self.lessons_dir = Path(lessons_dir)
        self.llm = LLMIntegration()
        self.generated_mindmaps = {}  # Кэш сгенерированных mind maps
        
    def extract_lesson_base_name(self, lesson_id: str) -> str:
        """Извлекает базовое название урока из lesson_id"""
        base_name = lesson_id.replace('.txt', '')
        
        if '/' in base_name or '\\' in base_name:
            base_name = Path(base_name).stem
        
        print(f"🔍 Извлекаем базовое название из '{lesson_id}' -> '{base_name}'")
        return base_name
    
    def get_lesson_slides(self, lesson_id: str) -> Optional[List[str]]:
        """Проверяет наличие слайдов для урока и возвращает их список"""
        base_name = self.extract_lesson_base_name(lesson_id)
        slides_dir = self.lessons_dir / "slides"
        
        print(f"🔍 Поиск слайдов для урока: {base_name}")
        
        if not slides_dir.exists():
            print(f"❌ Папка слайдов не существует: {slides_dir}")
            return None
            
        found_slides = []
        
        for ext in ['jpg', 'jpeg', 'png', 'webp']:
            slide_pattern = f"{base_name}_*.{ext}"
            slide_files = list(slides_dir.glob(slide_pattern))
            
            if slide_files:
                print(f"✅ Найдены слайды {ext}: {[f.name for f in slide_files]}")
                found_slides.extend(slide_files)
        
        if found_slides:
            found_slides.sort(key=lambda x: self._extract_slide_number(x.name))
            slide_paths = [f"slides/{slide.name}" for slide in found_slides]
            print(f"🎯 Итоговый список слайдов: {slide_paths}")
            return slide_paths
        
        print(f"❌ Слайды для '{base_name}' не найдены")
        return None
    
    def _extract_slide_number(self, filename: str) -> int:
        """Извлекает номер слайда из имени файла"""
        match = re.search(r'_(\d+)\.', filename)
        if match:
            return int(match.group(1))
        
        match = re.search(r'-(\d+)\.', filename)
        if match:
            return int(match.group(1))
            
        return 0
    
    def generate_lesson_mindmap_async(self, lesson_content: str, lesson_title: str, room_id: str, callback=None):
        """Асинхронная генерация mind map всего урока"""
        def generate_thread():
            try:
                print(f"🎨 Начало асинхронной генерации mind map для урока: {lesson_title}")
                
                # Проверяем кэш
                cache_key = f"{lesson_title}_{hash(lesson_content[:1000])}"
                if cache_key in self.generated_mindmaps:
                    print(f"✅ Используем mind map из кэша: {lesson_title}")
                    mindmap = self.generated_mindmaps[cache_key]
                    if callback:
                        callback(mindmap, room_id)
                    return
                
                prompt = f"""
                Создай подробную и структурированную mind map (интеллект-карту) для урока на тему: "{lesson_title}"

                СОДЕРЖАНИЕ УРОКА:
                {lesson_content[:3000]}

                ТРЕБОВАНИЯ К MIND MAP:
                1. Основная тема в центре: "{lesson_title}"
                2. 4-6 основных разделов (первого уровня)
                3. 2-4 подраздела для каждого основного раздела (второго уровня)  
                4. При необходимости добавляй подразделы третьего уровня
                5. Используй четкие, конкретные формулировки
                6. Отрази ключевые концепции, определения, примеры и практическое применение
                7. Структура должна логически следовать из содержания урока
                8. Включай важные детали и специфические термины

                Формат Mermaid (версия 10.2.0):
                flowchart TD
                    A["{lesson_title}"] --> B["Ключевой раздел 1"]
                    B --> C["Важный аспект 1.1"]
                    B --> D["Важный аспект 1.2"]
                    C --> E["Конкретная деталь 1.1.1"]
                    A --> F["Ключевой раздел 2"]
                    F --> G["Важный аспект 2.1"]

                Создай максимально подробную и полезную mind map, которая поможет ученику понять структуру урока.
                Верни ТОЛЬКО код Mermaid без пояснений и обратных кавычек.
                """
                
                system_prompt = """Ты - эксперт по созданию образовательных mind maps. 
                Создавай подробные, структурированные и логичные интеллект-карты, которые действительно помогают в обучении.
                Не ограничивай себя искусственно - добавляй столько уровней и элементов, сколько нужно для полного охвата темы.
                Используй конкретные термины и понятия из урока."""
                
                mermaid_code = self.llm._query_llm_api(
                    prompt=prompt,
                    context="",
                    subject="general",
                    system_prompt=system_prompt,
                    max_tokens=800  # Увеличиваем для более детальных карт
                )
                
                if mermaid_code:
                    cleaned_code = self._clean_mermaid_code(mermaid_code, lesson_title)
                    mindmap = {
                        "type": "mindmap",
                        "mermaid_code": cleaned_code,
                        "lesson_title": lesson_title,
                        "timestamp": time.time()
                    }
                    
                    # Сохраняем в кэш
                    self.generated_mindmaps[cache_key] = mindmap
                    
                    print(f"✅ Mind map успешно сгенерирована для: {lesson_title}")
                    
                    if callback:
                        callback(mindmap, room_id)
                else:
                    print("❌ Не удалось сгенерировать mind map")
                    fallback = self._create_fallback_mindmap(lesson_title)
                    if callback:
                        callback(fallback, room_id)
                
            except Exception as e:
                print(f"❌ Ошибка генерации mind map: {e}")
                fallback = self._create_fallback_mindmap(lesson_title)
                if callback:
                    callback(fallback, room_id)
        
        # Запускаем в отдельном потоке
        thread = threading.Thread(target=generate_thread)
        thread.daemon = True
        thread.start()
        print(f"🔄 Запущен поток генерации mind map для: {lesson_title}")
    
    def generate_lesson_mindmap(self, lesson_content: str, lesson_title: str) -> Dict:
        """Синхронная версия для обратной совместимости"""
        prompt = f"""
        Создай структурную mind map для урока: "{lesson_title}"
        
        Содержание: {lesson_content[:2000]}
        
        Верни ТОЛЬКО код Mermaid.
        """
        
        try:
            mermaid_code = self.llm._query_llm_api(
                prompt=prompt,
                context="",
                subject="general",
                system_prompt="Создай mind map для урока.",
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
        
        # Удаляем пояснения и лишний текст
        lines = []
        for line in code.split('\n'):
            line = line.strip()
            if line and not line.startswith(('%%', '//', '#')) and 'mermaid' not in line.lower():
                lines.append(line)
        
        code = '\n'.join(lines)
        
        # Убеждаемся, что это корректный Mermaid
        if not code.strip().startswith(('flowchart', 'graph', 'mindmap')):
            code = 'flowchart TD\n' + code
        
        # Добавляем основную тему если ее нет
        if f'["{lesson_title}"]' not in code and '["Основная тема"]' not in code:
            lines = code.split('\n')
            if len(lines) > 1 and '-->' in lines[1]:
                # Вставляем основную тему перед первыми связями
                lines.insert(0, f'    A["{lesson_title}"]')
            code = '\n'.join(lines)
        
        return code.strip()
    
    def _create_fallback_mindmap(self, lesson_title: str) -> str:
        """Создает детальную mind map по умолчанию"""
        return f'''flowchart TD
    A["{lesson_title}"] --> B["Основные понятия и определения"]
    A --> C["Ключевые теории и принципы"]
    A --> D["Практическое применение"]
    A --> E["Важные примеры и случаи"]
    A --> F["Связи с другими темами"]
    
    B --> B1["Базовые определения"]
    B --> B2["Терминология"]
    B --> B3["Фундаментальные концепции"]
    
    C --> C1["Основные законы и правила"]
    C --> C2["Теоретические основы"]
    C --> C3["Методы и подходы"]
    
    D --> D1["Реальные примеры"]
    D --> D2["Практические задачи"]
    D --> D3["Применение в жизни"]
    
    E --> E1["Типичные случаи"]
    E --> E2["Особые ситуации"]
    E --> E3["Сравнительные примеры"]
    
    F --> F1["Смежные темы"]
    F --> F2["Предыдущие знания"]
    F --> F3["Будущее применение"]'''
    
    def get_visualization_type(self, lesson_id: str) -> str:
        """Определяет тип визуализации: слайды или mindmap"""
        slides = self.get_lesson_slides(lesson_id)
        return "slides" if slides else "mindmap"
    
    def initialize_lesson_visualization(self, lesson_id: str, lesson_title: str, lesson_content: str, room_id: str, socketio=None):
        """Инициализирует визуализацию для урока (не блокирующая)"""
        visualization_type = self.get_visualization_type(lesson_id)
        
        print(f"🎨 Инициализация визуализации типа: {visualization_type}")
        
        if visualization_type == "slides":
            # Для слайдов просто загружаем их список
            slides = self.get_lesson_slides(lesson_id)
            if slides and socketio:
                # Отправляем информацию о слайдах
                socketio.emit('lesson_visualization', {
                    'room_id': room_id,
                    'type': 'slides_info',
                    'data': {
                        'slides': slides,
                        'total_slides': len(slides),
                        'lesson_title': lesson_title
                    },
                    'lesson_id': lesson_id,
                    'lesson_title': lesson_title
                }, room=room_id)
                print(f"✅ Информация о слайдах отправлена: {len(slides)} слайдов")
                
        else:
            # Для mind map запускаем асинхронную генерацию
            def mindmap_callback(mindmap, r_id):
                if socketio:
                    socketio.emit('lesson_visualization', {
                        'room_id': r_id,
                        'type': 'mindmap',
                        'data': mindmap,
                        'lesson_id': lesson_id,
                        'lesson_title': lesson_title
                    }, room=r_id)
                    print(f"✅ Mind map отправлена в комнату {r_id}")
            
            self.generate_lesson_mindmap_async(
                lesson_content, 
                lesson_title, 
                room_id, 
                mindmap_callback
            )
            
        return visualization_type

# Глобальный экземпляр менеджера визуализации
visualization_manager = VisualizationManager()
