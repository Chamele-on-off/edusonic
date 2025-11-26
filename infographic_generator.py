import re
import json
import random
from typing import Dict, List, Optional
from llm import LLMIntegration

class InfographicGenerator:
    def __init__(self):
        self.llm = LLMIntegration()
        self.styles = self._get_styles()
        
    def _get_styles(self) -> Dict:
        """Возвращает стили для инфографики"""
        return {
            "modern": {
                "colors": ["#6366f1", "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b"],
                "font": "Arial, sans-serif",
                "background": "#f8fafc",
                "border_radius": "12px",
                "shadow": "0 4px 6px -1px rgba(0, 0, 0, 0.1)"
            },
            "minimal": {
                "colors": ["#1f2937", "#374151", "#4b5563", "#6b7280"],
                "font": "Georgia, serif", 
                "background": "#ffffff",
                "border_radius": "8px",
                "shadow": "0 1px 3px 0 rgba(0, 0, 0, 0.1)"
            },
            "colorful": {
                "colors": ["#ef4444", "#f97316", "#eab308", "#22c55e", "#3b82f6"],
                "font": "Comic Sans MS, cursive",
                "background": "#fef7ed",
                "border_radius": "16px",
                "shadow": "0 8px 15px -3px rgba(0, 0, 0, 0.1)"
            }
        }
    
    def generate_infographic(self, topic: str, context: str = "") -> Dict:
        """Генерирует инфографику для темы"""
        try:
            print(f"🎨 Генерация инфографики для: {topic[:100]}...")
            
            # Выбираем случайный стиль
            style_name = random.choice(list(self.styles.keys()))
            style = self.styles[style_name]
            
            # Генерируем структуру инфографики через LLM
            structure = self._generate_structure(topic, context)
            
            if not structure:
                return self._create_fallback_infographic(topic, style)
            
            # Создаем SVG на основе структуры
            svg_code = self._create_svg_from_structure(structure, style)
            
            return {
                "success": True,
                "topic": topic,
                "svg_code": svg_code,
                "style": style_name,
                "structure": structure,
                "timestamp": time.time()
            }
            
        except Exception as e:
            print(f"❌ Ошибка генерации инфографики: {e}")
            return self._create_fallback_infographic(topic, self.styles["modern"])
    
    def _generate_structure(self, topic: str, context: str) -> Optional[Dict]:
        """Генерирует структуру инфографики через LLM"""
        prompt = f"""
        Создай структуру инфографики для темы: "{topic}"
        
        Контекст: {context}
        
        Требования к структуре:
        - Максимум 6 ключевых элементов
        - Логическая группировка информации
        - Визуально привлекательная компоновка
        - Понятная иерархия
        
        Формат ответа (только JSON):
        {{
            "title": "Заголовок инфографики",
            "elements": [
                {{
                    "type": "header|key_point|fact|diagram|process",
                    "content": "Текст или данные",
                    "importance": 1-5
                }}
            ],
            "layout": "vertical|horizontal|grid|radial"
        }}
        
        Используй только русский язык.
        """
        
        try:
            response = self.llm._query_llm_api(
                prompt=prompt,
                context="",
                subject="general",
                system_prompt="Ты создаешь структуры для образовательной инфографики. Возвращай только валидный JSON.",
                max_tokens=800
            )
            
            if response:
                # Очищаем ответ от лишнего текста
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    return json.loads(json_str)
                    
        except Exception as e:
            print(f"❌ Ошибка генерации структуры: {e}")
            
        return None
    
    def _create_svg_from_structure(self, structure: Dict, style: Dict) -> str:
        """Создает SVG код на основе структуры"""
        try:
            elements = structure.get("elements", [])
            layout = structure.get("layout", "vertical")
            
            # Создаем SVG с базовой структурой
            svg_parts = [
                f'<svg width="400" height="300" xmlns="http://www.w3.org/2000/svg" style="background:{style["background"]}">',
                '<style>',
                f'.text {{ font-family: {style["font"]}; fill: #1f2937; }}',
                '.header { font-size: 16px; font-weight: bold; }',
                '.key-point { font-size: 12px; }',
                '.fact { font-size: 11px; fill: #4b5563; }',
                '</style>'
            ]
            
            # Добавляем заголовок
            title = structure.get("title", "Инфографика")
            svg_parts.append(f'<text x="200" y="30" text-anchor="middle" class="text header">{title}</text>')
            
            # Добавляем элементы в зависимости от layout
            if layout == "vertical":
                svg_parts.extend(self._create_vertical_layout(elements, style))
            elif layout == "horizontal":
                svg_parts.extend(self._create_horizontal_layout(elements, style))
            elif layout == "grid":
                svg_parts.extend(self._create_grid_layout(elements, style))
            else:
                svg_parts.extend(self._create_vertical_layout(elements, style))
            
            svg_parts.append('</svg>')
            return '\n'.join(svg_parts)
            
        except Exception as e:
            print(f"❌ Ошибка создания SVG: {e}")
            return self._create_fallback_svg(structure.get("title", "Инфографика"), style)
    
    def _create_vertical_layout(self, elements: List[Dict], style: Dict) -> List[str]:
        """Создает вертикальную компоновку"""
        svg_elements = []
        y_position = 60
        
        for i, element in enumerate(elements):
            color = style["colors"][i % len(style["colors"])]
            elem_type = element.get("type", "key_point")
            content = element.get("content", "")
            
            # Обрезаем длинный текст
            if len(content) > 80:
                content = content[:77] + "..."
            
            # Создаем визуальный элемент
            if elem_type == "header":
                svg_elements.extend([
                    f'<rect x="50" y="{y_position}" width="300" height="40" rx="{style["border_radius"]}" fill="{color}" opacity="0.8" style="filter: drop-shadow({style["shadow"]})"/>',
                    f'<text x="200" y="{y_position + 25}" text-anchor="middle" class="text header" fill="white">{content}</text>'
                ])
                y_position += 60
            else:
                svg_elements.extend([
                    f'<circle cx="80" cy="{y_position + 15}" r="8" fill="{color}"/>',
                    f'<text x="100" y="{y_position + 20}" class="text key-point">{content}</text>'
                ])
                y_position += 40
                
            # Ограничиваем высоту
            if y_position > 250:
                break
                
        return svg_elements
    
    def _create_horizontal_layout(self, elements: List[Dict], style: Dict) -> List[str]:
        """Создает горизонтальную компоновку"""
        svg_elements = []
        x_positions = [80, 160, 240, 320]
        y_position = 100
        
        for i, element in enumerate(elements[:4]):  # Максимум 4 элемента
            color = style["colors"][i % len(style["colors"])]
            content = element.get("content", "")
            
            if len(content) > 30:
                content = self._wrap_text(content, 25)
                
            svg_elements.extend([
                f'<rect x="{x_positions[i]-30}" y="{y_position-40}" width="60" height="60" rx="{style["border_radius"]}" fill="{color}" opacity="0.7" style="filter: drop-shadow({style["shadow"]})"/>',
                f'<text x="{x_positions[i]}" y="{y_position}" text-anchor="middle" class="text key-point" fill="white">{content}</text>'
            ])
            
        return svg_elements
    
    def _create_grid_layout(self, elements: List[Dict], style: Dict) -> List[str]:
        """Создает сеточную компоновку"""
        svg_elements = []
        positions = [(100, 80), (300, 80), (100, 180), (300, 180)]
        
        for i, element in enumerate(elements[:4]):
            color = style["colors"][i % len(style["colors"])]
            content = element.get("content", "")
            
            if len(content) > 40:
                content = content[:37] + "..."
                
            x, y = positions[i]
            svg_elements.extend([
                f'<circle cx="{x}" cy="{y}" r="35" fill="{color}" opacity="0.6"/>',
                f'<text x="{x}" y="{y}" text-anchor="middle" class="text key-point" fill="white">{content}</text>'
            ])
            
        return svg_elements
    
    def _wrap_text(self, text: str, max_length: int) -> str:
        """Разбивает текст на несколько строк"""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            if len(' '.join(current_line + [word])) <= max_length:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                
        if current_line:
            lines.append(' '.join(current_line))
            
        return '\n'.join(lines[:2])  # Максимум 2 строки
    
    def _create_fallback_infographic(self, topic: str, style: Dict) -> Dict:
        """Создает инфографику по умолчанию"""
        fallback_svg = self._create_fallback_svg(topic, style)
        
        return {
            "success": True,
            "topic": topic,
            "svg_code": fallback_svg,
            "style": "modern",
            "structure": {"title": topic, "elements": [], "layout": "vertical"},
            "timestamp": time.time()
        }
    
    def _create_fallback_svg(self, title: str, style: Dict) -> str:
        """Создает SVG по умолчанию"""
        return f'''
        <svg width="400" height="300" xmlns="http://www.w3.org/2000/svg" style="background:{style['background']}">
            <style>
                .text {{ font-family: {style['font']}; fill: #1f2937; }}
                .header {{ font-size: 18px; font-weight: bold; }}
            </style>
            <rect x="50" y="50" width="300" height="200" rx="{style['border_radius']}" fill="{style['colors'][0]}" opacity="0.1" style="filter: drop-shadow({style['shadow']})"/>
            <text x="200" y="100" text-anchor="middle" class="text header">{title}</text>
            <text x="200" y="140" text-anchor="middle" class="text" font-size="14px">Инфографика</text>
            <circle cx="200" cy="180" r="8" fill="{style['colors'][1]}"/>
            <circle cx="170" cy="180" r="8" fill="{style['colors'][2]}"/>
            <circle cx="230" cy="180" r="8" fill="{style['colors'][3]}"/>
        </svg>
        '''

# Глобальный экземпляр генератора
infographic_generator = InfographicGenerator()
