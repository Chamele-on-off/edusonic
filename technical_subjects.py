# technical_subjects.py
# Отдельный модуль для технических и естественнонаучных предметов

import re
from typing import Dict, List, Tuple, Optional

# =============================================================================
# КОНСТАНТЫ И НАСТРОЙКИ
# =============================================================================

# Списки технических и естественнонаучных предметов
TECHNICAL_SUBJECTS = [
    'математика', 'алгебра', 'геометрия', 'физика', 'химия', 
    'биология', 'информатика', 'программирование', 'инженерия'
]

NATURAL_SCIENCES = [
    'биология', 'география', 'астрономия', 'экология', 'геология'
]

# Специальные символы и формулы для технических предметов
TECHNICAL_SYMBOLS = {
    'математика': ['∑', '∫', '∂', '∇', '∞', '√', '≠', '≈', '≡', '≤', '≥', '→', 'π', 'θ', 'α', 'β', 'γ'],
    'физика': ['F', 'm', 'a', 'v', 't', 's', 'E', 'p', 'ρ', 'λ', 'ν', 'ω', 'τ', 'μ', 'ε', 'σ', 'Ω'],
    'химия': ['H₂O', 'CO₂', 'H₂SO₄', 'NaCl', 'C₆H₁₂O₆', '→', '⇌', '↑', '↓', 'Δ', '°C'],
    'биология': ['DNA', 'RNA', 'ATP', 'ADP', 'pH', 'O₂', 'CO₂', 'N₂', '→', '⇌']
}

# Паттерны для обнаружения формул
FORMULA_PATTERNS = [
    r'\b[a-zA-Z]\s*=\s*',  # x = 
    r'\b[a-zA-Z]\s*\+\s*[a-zA-Z]',  # a + b
    r'\b[a-zA-Z]\s*\*\s*[a-zA-Z]',  # a * b
    r'\b[a-zA-Z]\s*/\s*[a-zA-Z]',  # a / b
    r'\b[a-zA-Z]\s*\^\s*\d',  # x^2
    r'\bsqrt\s*\(',  # sqrt(
    r'\bsin\s*\(|cos\s*\(|tan\s*\(',  # sin(, cos(, tan(
    r'\bH₂O|CO₂|NaCl|H₂SO₄\b',  # Химические формулы
    r'→|⇌|↑|↓|Δ|°C|℃|℉',  # Химические символы
    r'∑|∫|∂|∇|∞|√|≠|≈|≡|≤|≥|π|θ|α|β|γ',  # Математические символы
    r'\d+\s*[+\-*/]\s*\d+',  # Арифметические выражения
    r'[a-zA-Z]_?\d+',  # Индексы: x_1, H2O
]

# Замены символов для произношения
SYMBOL_REPLACEMENTS = {
    '→': ' стремится к ',
    '=': ' равно ',
    '≠': ' не равно ',
    '≈': ' приблизительно равно ',
    '≤': ' меньше или равно ',
    '≥': ' больше или равно ',
    '∑': ' сумма ',
    '∫': ' интеграл ',
    '∂': ' частная производная ',
    '∇': ' набла ',
    '∞': ' бесконечность ',
    '√': ' корень квадратный из ',
    'π': ' пи ',
    'θ': ' тэта ',
    'α': ' альфа ',
    'β': ' бета ',
    'γ': ' гамма ',
    'Δ': ' дельта ',
    '°C': ' градусов Цельсия ',
    '℃': ' градусов Цельсия ',
    '℉': ' градусов Фаренгейта ',
    '↑': ' вверх ',
    '↓': ' вниз ',
    '⇌': ' обратимая реакция ',
    'H₂O': ' аш два о ',
    'CO₂': ' цэ о два ',
    'NaCl': ' натрий хлор ',
    'H₂SO₄': ' аш два эс о четыре ',
    'C₆H₁₂O₆': ' цэ шесть аш двенадцать о шесть ',
}

# =============================================================================
# ОСНОВНЫЕ ФУНКЦИИ
# =============================================================================

def is_technical_subject(subject: str) -> bool:
    """
    Определяет, является ли предмет техническим или естественнонаучным
    
    Args:
        subject: Название предмета
        
    Returns:
        bool: True если предмет технический или естественнонаучный
    """
    if not subject:
        return False
    
    subject_lower = subject.lower()
    
    # Проверяем технические предметы
    for tech_subj in TECHNICAL_SUBJECTS:
        if tech_subj in subject_lower:
            return True
    
    # Проверяем естественные науки
    for science_subj in NATURAL_SCIENCES:
        if science_subj in subject_lower:
            return True
    
    return False


def is_natural_science(subject: str) -> bool:
    """
    Определяет, является ли предмет естественной наукой
    
    Args:
        subject: Название предмета
        
    Returns:
        bool: True если предмет естественная наука
    """
    if not subject:
        return False
    
    subject_lower = subject.lower()
    for science_subj in NATURAL_SCIENCES:
        if science_subj in subject_lower:
            return True
    return False


def contains_formulas(text: str) -> bool:
    """
    Проверяет, содержит ли текст формулы или технические обозначения
    
    Args:
        text: Текст для проверки
        
    Returns:
        bool: True если содержит формулы
    """
    if not text:
        return False
    
    # Проверяем все паттерны формул
    for pattern in FORMULA_PATTERNS:
        if re.search(pattern, text):
            return True
    
    # Дополнительная проверка на специальные символы
    for symbol in SYMBOL_REPLACEMENTS.keys():
        if symbol in text and len(symbol) > 1:  # Проверяем только многосимвольные замены
            if symbol in text:
                return True
    
    return False


def should_preserve_formatting(text: str, subject: Optional[str] = None) -> bool:
    """
    Определяет, нужно ли сохранять форматирование для текста
    
    Args:
        text: Текст для анализа
        subject: Предмет (опционально)
        
    Returns:
        bool: True если нужно сохранять форматирование
    """
    if not text:
        return False
    
    # Проверяем предмет
    if subject and is_technical_subject(subject):
        return True
    
    # Проверяем формулы в тексте
    if contains_formulas(text):
        return True
    
    return False


def clean_text_for_speech_technical(text: str, subject: Optional[str] = None) -> str:
    """
    Умная очистка текста для озвучивания:
    - Для гуманитарных предметов: полная очистка
    - Для технических предметов: частичная очистка с сохранением формул
    
    Args:
        text: Текст для очистки
        subject: Предмет (опционально)
        
    Returns:
        str: Очищенный текст
    """
    if not text:
        return ""
    
    # Если это технический предмет ИЛИ есть формулы, очищаем аккуратно
    if should_preserve_formatting(text, subject):
        return _clean_text_preserving_formulas(text)
    else:
        # Полная очистка для гуманитарных предметов
        return _clean_text_completely(text)


def _clean_text_preserving_formulas(text: str) -> str:
    """
    Очистка текста с сохранением формул и специальных символов
    
    Args:
        text: Текст для очистки
        
    Returns:
        str: Очищенный текст с сохраненными формулами
    """
    if not text:
        return ""
    
    # Удаляем только маркеры форматирования, но не формулы
    text = re.sub(r'[#\*\_\~`]', '', text)
    
    # Удаляем лишние пробелы, но сохраняем специальные символы
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Сохраняем специальные символы для формул
    # Разрешаем: буквы, цифры, пробелы, знаки препинания, математические символы
    allowed_chars = r'\w\s\.,!?;:()\-—\+\*/=\^πθαβγ∑∫∂∇∞√≠≈≡≤≥→°C℃℉Δ↑↓⇌₂₃₄₅₆₇₈₉'
    text = re.sub(f'[^{allowed_chars}]', '', text)
    
    # Заменяем некоторые символы для лучшего произношения
    for symbol, replacement in SYMBOL_REPLACEMENTS.items():
        text = text.replace(symbol, replacement)
    
    # Нормализуем пробелы
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\s+([\.,!?;:)])', r'\1', text)
    text = re.sub(r'([(\-])\s+', r'\1', text)
    text = text.strip()
    
    # Обеспечиваем, что предложения начинаются с заглавной буквы
    if text and len(text) > 1:
        text = text[0].upper() + text[1:]
    
    return text


def _clean_text_completely(text: str) -> str:
    """
    Полная очистка текста (для гуманитарных предметов)
    
    Args:
        text: Текст для очистки
        
    Returns:
        str: Полностью очищенный текст
    """
    if not text:
        return ""
    
    # Удаляем маркеры форматирования
    text = re.sub(r'[#\*\_\~`]', '', text)
    
    # Удаляем лишние переносы строк и пробелы
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Удаляем escape-последовательности
    text = re.sub(r'\\n', ' ', text)
    text = re.sub(r'\\t', ' ', text)
    text = re.sub(r'\\r', ' ', text)
    
    # Оставляем только кириллицу, латиницу, цифры и основные знаки препинания
    text = re.sub(r'[^\u0400-\u04FFa-zA-Z0-9\s\.,!?;:()\-—]', '', text)
    
    # Нормализуем знаки препинания
    text = re.sub(r'[\.\,]{2,}', '.', text)
    text = re.sub(r'\s+([\.,!?;:)])', r'\1', text)
    text = re.sub(r'([(\-])\s+', r'\1', text)
    text = text.strip()
    
    # Делаем первую букву заглавной
    if text and len(text) > 1:
        text = text[0].upper() + text[1:]
    
    return text


def get_subject_type(subject: str) -> str:
    """
    Определяет тип предмета
    
    Args:
        subject: Название предмета
        
    Returns:
        str: Тип предмета: "technical", "natural_science", "language", "humanitarian"
    """
    if not subject:
        return "general"
    
    subject_lower = subject.lower()
    
    # Технические предметы
    for tech_subj in TECHNICAL_SUBJECTS:
        if tech_subj in subject_lower:
            return "technical"
    
    # Естественные науки
    for science_subj in NATURAL_SCIENCES:
        if science_subj in subject_lower:
            return "natural_science"
    
    # Языковые предметы
    language_keywords = [
        'английский', 'французский', 'немецкий', 'испанский', 
        'китайский', 'японский', 'корейский', 'итальянский',
        'язык', 'language', 'english', 'french', 'german', 'spanish'
    ]
    for keyword in language_keywords:
        if keyword in subject_lower:
            return "language"
    
    return "humanitarian"


def generate_technical_practice_prompt(subject: str, topic: str, level: str = "5", age: int = 12) -> str:
    """
    Генерирует промпт для практики по техническим предметам
    
    Args:
        subject: Название предмета
        topic: Тема урока
        level: Уровень класса
        age: Возраст ученика
        
    Returns:
        str: Промпт для LLM
    """
    subject_type = get_subject_type(subject)
    
    if subject_type == "technical":
        # Математика, физика, химия, информатика
        return f"""
        Создай практические задания по {subject} на тему: {topic}
        
        УРОВЕНЬ УЧЕНИКА: {level} класс, {age} лет
        
        ТИПЫ ЗАДАНИЙ:
        1. Решение задач с формулами
        2. Вычисление примеров
        3. Построение графиков/диаграмм
        4. Доказательство теорем (если применимо)
        5. Практические эксперименты (для физики/химии)
        
        ТРЕБОВАНИЯ:
        - Включай конкретные формулы и вычисления
        - Добавляй пошаговые решения
        - Учитывай возраст ученика
        - Используй математическую нотацию
        - Для физики/химии включай единицы измерения
        - Предоставляй правильные ответы для проверки
        
        СТРУКТУРА:
        Каждое задание должно содержать:
        1. Формулировку задания
        2. Подсказки (если нужно)
        3. Пример решения (для первого задания)
        4. Ожидаемый ответ
        
        Верни 3-5 практических заданий в формате JSON:
        {{
            "practice_type": "technical",
            "subject": "{subject}",
            "topic": "{topic}",
            "level": "{level}",
            "questions": [
                {{
                    "id": 1,
                    "type": "calculation",  # или "proof", "graph", "experiment"
                    "question": "текст вопроса",
                    "hint": "подсказка (опционально)",
                    "expected_answer": "правильный ответ",
                    "explanation": "пояснение решения"
                }}
            ]
        }}
        """
    
    elif subject_type == "natural_science":
        # Биология, география, астрономия
        return f"""
        Создай практические задания по {subject} на тему: {topic}
        
        УРОВЕНЬ УЧЕНИКА: {level} класс, {age} лет
        
        ТИПЫ ЗАДАНИЙ:
        1. Описание процессов в природе
        2. Схемы и диаграммы
        3. Классификация объектов
        4. Эксперименты и наблюдения
        5. Анализ данных
        
        ТРЕБОВАНИЯ:
        - Основано на научных фактах
        - Содержит схемы/иллюстрации мысленно
        - Включает практические наблюдения
        - Учитывает экологический аспект
        - Использует научную терминологию
        
        СТРУКТУРА:
        Каждое задание должно содержать:
        1. Научную проблему или вопрос
        2. Контекст или данные
        3. Задание для ученика
        4. Критерии оценки
        
        Верни 3-5 практических заданий в формате JSON.
        """
    
    else:
        # Гуманитарные предметы (стандартный промпт)
        return f"""
        Создай практические задания по {subject} на тему: {topic}
        
        УРОВЕНЬ УЧЕНИКА: {level} класс, {age} лет
        
        Создай 5 заданий разного типа:
        1. Вопросы на понимание
        2. Практические задачи
        3. Творческие задания
        4. Аналитические вопросы
        5. Применение знаний
        
        Верни задания в структурированном виде в формате JSON.
        """


def adapt_visualization_for_technical(text: str, subject: str) -> Dict:
    """
    Адаптирует визуализацию для технических предметов
    
    Args:
        text: Текст для визуализации
        subject: Предмет
        
    Returns:
        Dict: Настройки визуализации
    """
    subject_type = get_subject_type(subject)
    
    if subject_type in ["technical", "natural_science"]:
        return {
            "topic": text,
            "type": "technical_diagram",
            "subject": subject,
            "requirements": [
                "Используй математические символы если есть формулы",
                "Добавь формулы в читаемом формате",
                "Создай схему/диаграмму для наглядности",
                "Используй научную нотацию",
                "Сделай визуализацию понятной для учеников"
            ],
            "style": {
                "use_grid": True,
                "show_coordinates": subject_type == "technical",
                "add_labels": True,
                "color_scheme": "scientific"
            }
        }
    
    # Для гуманитарных предметов - стандартная SVG инфографика
    return {
        "topic": text,
        "type": "infographic",
        "subject": subject,
        "requirements": ["Создай понятную инфографику"],
        "style": {
            "use_grid": False,
            "show_coordinates": False,
            "add_labels": True,
            "color_scheme": "educational"
        }
    }


def extract_formulas_from_text(text: str) -> List[str]:
    """
    Извлекает формулы из текста
    
    Args:
        text: Текст для анализа
        
    Returns:
        List[str]: Список найденных формул
    """
    if not text:
        return []
    
    formulas = []
    
    # Ищем паттерны формул
    for pattern in FORMULA_PATTERNS:
        matches = re.findall(pattern, text)
        if matches:
            formulas.extend(matches)
    
    # Ищем химические формулы (H2O, CO2 и т.д.)
    chem_pattern = r'[A-Z][a-z]?\d*[A-Z][a-z]?\d*'
    chem_matches = re.findall(chem_pattern, text)
    formulas.extend(chem_matches)
    
    # Удаляем дубликаты
    formulas = list(set(formulas))
    
    return formulas


def analyze_technical_complexity(text: str, subject: str) -> Dict:
    """
    Анализирует сложность технического текста
    
    Args:
        text: Текст для анализа
        subject: Предмет
        
    Returns:
        Dict: Результаты анализа сложности
    """
    formulas = extract_formulas_from_text(text)
    
    # Подсчитываем формулы и символы
    formula_count = len(formulas)
    symbol_count = 0
    
    for symbol in SYMBOL_REPLACEMENTS.keys():
        if symbol in text:
            symbol_count += text.count(symbol)
    
    # Определяем уровень сложности
    complexity = "low"
    if formula_count > 5 or symbol_count > 10:
        complexity = "high"
    elif formula_count > 2 or symbol_count > 5:
        complexity = "medium"
    
    return {
        "subject": subject,
        "formula_count": formula_count,
        "symbol_count": symbol_count,
        "complexity": complexity,
        "has_formulas": formula_count > 0,
        "formulas": formulas[:10]  # Ограничиваем список
    }


def generate_technical_explanation(text: str, subject: str) -> str:
    """
    Генерирует объяснение для технического текста
    
    Args:
        text: Исходный текст
        subject: Предмет
        
    Returns:
        str: Упрощенное объяснение
    """
    subject_type = get_subject_type(subject)
    
    if subject_type == "technical":
        # Для математики, физики, химии
        formulas = extract_formulas_from_text(text)
        
        if formulas:
            explanation = f"Этот текст содержит {len(formulas)} формул(у). "
            explanation += "Давайте разберем основные концепции:\n\n"
            
            for i, formula in enumerate(formulas[:3], 1):
                explanation += f"{i}. Формула '{formula}' - "
                
                if '=' in formula:
                    explanation += "выражает равенство между величинами.\n"
                elif '+' in formula or '-' in formula or '*' in formula or '/' in formula:
                    explanation += "представляет математическую операцию.\n"
                elif '²' in formula or '^2' in formula:
                    explanation += "связана с квадратом величины.\n"
                elif '√' in formula:
                    explanation += "связана с квадратным корнем.\n"
                else:
                    explanation += "описывает зависимость между переменными.\n"
            
            if len(formulas) > 3:
                explanation += f"\nИ еще {len(formulas) - 3} формул(ы) для изучения.\n"
            
            explanation += "\nРекомендую обратить внимание на каждую формулу и понять её смысл."
            return explanation
    
    # Для остальных случаев возвращаем исходный текст
    return text


def format_technical_question(question: str, subject: str) -> str:
    """
    Форматирует вопрос для технических предметов
    
    Args:
        question: Вопрос
        subject: Предмет
        
    Returns:
        str: Отформатированный вопрос
    """
    subject_type = get_subject_type(subject)
    
    if subject_type == "technical":
        # Добавляем инструкции для технических вопросов
        formatted = f"Вопрос по {subject}:\n\n"
        formatted += f"{question}\n\n"
        formatted += "ВНИМАНИЕ: Если в ответе есть формулы, запиши их четко и понятно.\n"
        formatted += "Используй математическую нотацию где это необходимо.\n"
        formatted += "Если нужно вычисление - покажи все шаги решения."
        return formatted
    elif subject_type == "natural_science":
        formatted = f"Вопрос по {subject}:\n\n"
        formatted += f"{question}\n\n"
        formatted += "Опиши явление или процесс подробно, используя научные термины."
        return formatted
    else:
        return question


# =============================================================================
# ФУНКЦИИ ДЛЯ ГЕНЕРАЦИИ УРОКОВ
# =============================================================================

def generate_technical_lesson_prompt(subject: str, topic: str, level: str = "5", age: int = 12) -> str:
    """
    Генерирует промпт для создания урока по техническому предмету
    
    Args:
        subject: Название предмета
        topic: Тема урока
        level: Уровень класса
        age: Возраст ученика
        
    Returns:
        str: Промпт для LLM
    """
    subject_type = get_subject_type(subject)
    
    if subject_type == "technical":
        return f"""
        Создай урок по {subject} на тему: {topic}
        
        УРОВЕНЬ УЧЕНИЯ: {level} класс, {age} лет
        
        СТРУКТУРА УРОКА:
        1. ВВЕДЕНИЕ (что будем изучать, зачем это важно)
        2. ОСНОВНЫЕ ПОНЯТИЯ (определения, термины)
        3. ТЕОРИЯ И ФОРМУЛЫ (математические выражения, законы)
        4. ПРИМЕРЫ И РЕШЕНИЯ (конкретные задачи с пошаговым решением)
        5. ПРАКТИЧЕСКОЕ ПРИМЕНЕНИЕ (где используется в реальной жизни)
        6. ИТОГИ И ВЫВОДЫ (краткое повторение)
        
        ТРЕБОВАНИЯ К УРОКУ:
        - Используй точные научные формулировки
        - Включай формулы и математические выражения
        - Объясняй каждый шаг в примерах
        - Связывай теорию с практикой
        - Учитывай возрастные особенности
        - Добавляй интересные факты по теме
        
        ОСОБЫЕ УКАЗАНИЯ:
        - Формулы записывай в читаемом формате
        - Для физики указывай единицы измерения
        - Для химии используй химические уравнения
        - Для математики покажи вычислительные методы
        
        ДЛИНА УРОКА: примерно 800-1200 слов, разбитых на 6-8 абзацев
        
        Верни текст урока, готовый к озвучиванию.
        """
    
    elif subject_type == "natural_science":
        return f"""
        Создай урок по {subject} на тему: {topic}
        
        УРОВЕНЬ УЧЕНИЯ: {level} класс, {age} лет
        
        СТРУКТУРА УРОКА:
        1. ВВЕДЕНИЕ (предмет изучения, актуальность)
        2. ОСНОВНЫЕ ЯВЛЕНИЯ И ПРОЦЕССЫ
        3. НАУЧНЫЕ ФАКТЫ И ЗАКОНОМЕРНОСТИ
        4. СХЕМЫ И КЛАССИФИКАЦИИ (мысленные диаграммы)
        5. ПРИМЕРЫ В ПРИРОДЕ
        6. ЗНАЧЕНИЕ ДЛЯ ЧЕЛОВЕКА И ЭКОЛОГИИ
        
        ТРЕБОВАНИЯ:
        - Используй научную терминологию с пояснениями
        - Описывай процессы последовательно
        - Приводи конкретные примеры из природы
        - Объясняй причинно-следственные связи
        - Указывай экологический аспект
        
        ДЛИНА УРОКА: примерно 700-1000 слов
        
        Верни текст урока.
        """
    
    else:
        # Гуманитарные предметы
        return f"""
        Создай образовательный урок на тему: {topic}
        
        Предмет: {subject}
        Уровень ученика: {level} класс, {age} лет
        
        Урок должен быть:
        - Информативным и структурированным
        - Интересным и понятным
        - С примерами и пояснениями
        - Разбитым на логические разделы
        
        ДЛИНА: примерно 600-900 слов
        
        Верни только текст урока.
        """


def get_technical_teaching_strategy(subject: str, complexity: str) -> Dict:
    """
    Возвращает стратегию преподавания для технического предмета
    
    Args:
        subject: Предмет
        complexity: Уровень сложности ("low", "medium", "high")
        
    Returns:
        Dict: Стратегия преподавания
    """
    subject_type = get_subject_type(subject)
    
    strategies = {
        "technical": {
            "low": {
                "pace": "медленный",
                "examples": "много простых примеров",
                "formulas": "объяснять каждую формулу подробно",
                "practice": "много базовых упражнений",
                "visualization": "простые схемы и графики"
            },
            "medium": {
                "pace": "умеренный",
                "examples": "сбалансированное количество примеров",
                "formulas": "объяснять ключевые формулы",
                "practice": "разнообразные задания",
                "visualization": "детализированные диаграммы"
            },
            "high": {
                "pace": "интенсивный",
                "examples": "сложные и комплексные примеры",
                "formulas": "фокус на взаимосвязи формул",
                "practice": "задачи повышенной сложности",
                "visualization": "сложные схемы и модели"
            }
        },
        "natural_science": {
            "low": {
                "pace": "медленный",
                "examples": "яркие примеры из природы",
                "terminology": "минимальная, с пояснениями",
                "practice": "наблюдения и описания",
                "visualization": "простые схемы процессов"
            },
            "medium": {
                "pace": "умеренный",
                "examples": "детальные примеры",
                "terminology": "расширенная, с определениями",
                "practice": "классификация и анализ",
                "visualization": "детальные диаграммы"
            },
            "high": {
                "pace": "интенсивный",
                "examples": "комплексные системы",
                "terminology": "полная научная терминология",
                "practice": "исследовательские задачи",
                "visualization": "сложные системные схемы"
            }
        }
    }
    
    # Получаем стратегию по умолчанию
    default_strategy = {
        "pace": "умеренный",
        "examples": "достаточное количество",
        "formulas": "объяснять по мере необходимости",
        "practice": "стандартные задания",
        "visualization": "базовая инфографика"
    }
    
    if subject_type in strategies and complexity in strategies[subject_type]:
        return strategies[subject_type][complexity]
    
    return default_strategy


# =============================================================================
# ФУНКЦИИ ДЛЯ ОЦЕНКИ ОТВЕТОВ
# =============================================================================

def evaluate_technical_answer(student_answer: str, correct_answer: str, subject: str) -> Dict:
    """
    Оценивает ответ ученика по техническому предмету
    
    Args:
        student_answer: Ответ ученика
        correct_answer: Правильный ответ
        subject: Предмет
        
    Returns:
        Dict: Результаты оценки
    """
    subject_type = get_subject_type(subject)
    
    # Базовая оценка
    evaluation = {
        "correct": False,
        "score": 0,
        "feedback": "",
        "mistakes": [],
        "suggestions": []
    }
    
    # Приводим ответы к нижнему регистру для сравнения
    student_lower = student_answer.lower().strip()
    correct_lower = correct_answer.lower().strip()
    
    # Простое сравнение для начала
    if student_lower == correct_lower:
        evaluation["correct"] = True
        evaluation["score"] = 100
        evaluation["feedback"] = "Отличный ответ! Всё верно."
        return evaluation
    
    # Для технических предметов проверяем наличие ключевых элементов
    if subject_type in ["technical", "natural_science"]:
        # Извлекаем формулы из правильного ответа
        correct_formulas = extract_formulas_from_text(correct_answer)
        student_formulas = extract_formulas_from_text(student_answer)
        
        # Проверяем наличие ключевых формул
        missing_formulas = []
        for formula in correct_formulas:
            if formula not in student_answer:
                missing_formulas.append(formula)
        
        if missing_formulas:
            evaluation["mistakes"].append(f"Отсутствуют формулы: {', '.join(missing_formulas)}")
            evaluation["suggestions"].append("Включите все необходимые формулы в ответ.")
        
        # Проверяем наличие ключевых терминов
        key_terms = []
        if "математика" in subject.lower():
            key_terms = ["формула", "уравнение", "решение", "ответ", "вычислить"]
        elif "физика" in subject.lower():
            key_terms = ["закон", "сила", "энергия", "масса", "скорость", "формула"]
        elif "химия" in subject.lower():
            key_terms = ["реакция", "формула", "уравнение", "элемент", "вещество"]
        
        missing_terms = []
        for term in key_terms:
            if term in correct_lower and term not in student_lower:
                missing_terms.append(term)
        
        if missing_terms:
            evaluation["mistakes"].append(f"Отсутствуют ключевые термины: {', '.join(missing_terms)}")
        
        # Оценка на основе найденных ошибок
        if not evaluation["mistakes"]:
            # Если нет явных ошибок с формулами и терминами
            evaluation["score"] = 70
            evaluation["correct"] = True
            evaluation["feedback"] = "Ответ в целом верный, но можно было бы добавить больше деталей."
        else:
            evaluation["score"] = max(0, 100 - len(evaluation["mistakes"]) * 20)
            evaluation["feedback"] = "Есть ошибки. Обратите внимание на замечания выше."
    
    else:
        # Для гуманитарных предметов - более мягкая оценка
        # Проверяем совпадение по ключевым словам
        student_words = set(student_lower.split())
        correct_words = set(correct_lower.split())
        
        common_words = student_words.intersection(correct_words)
        similarity = len(common_words) / len(correct_words) if correct_words else 0
        
        evaluation["score"] = int(similarity * 100)
        evaluation["correct"] = evaluation["score"] >= 70
        
        if evaluation["correct"]:
            evaluation["feedback"] = f"Ответ содержит ключевые понятия ({evaluation['score']}%)."
        else:
            evaluation["feedback"] = f"Ответ частично верен ({evaluation['score']}%). Добавьте больше деталей."
    
    return evaluation


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def debug_technical_analysis(text: str, subject: str = "") -> Dict:
    """
    Отладочная функция для анализа текста
    
    Args:
        text: Текст для анализа
        subject: Предмет (опционально)
        
    Returns:
        Dict: Результаты анализа
    """
    return {
        "text_sample": text[:200] + "..." if len(text) > 200 else text,
        "text_length": len(text),
        "subject": subject,
        "subject_type": get_subject_type(subject) if subject else "unknown",
        "is_technical": is_technical_subject(subject) if subject else False,
        "contains_formulas": contains_formulas(text),
        "formulas_found": extract_formulas_from_text(text),
        "should_preserve_formatting": should_preserve_formatting(text, subject),
        "cleaned_technical": clean_text_for_speech_technical(text, subject)[:300] + "..." if len(text) > 300 else clean_text_for_speech_technical(text, subject),
        "cleaned_complete": _clean_text_completely(text)[:300] + "..." if len(text) > 300 else _clean_text_completely(text)
    }


def validate_subject_for_technical_support(subject: str) -> Tuple[bool, str]:
    """
    Проверяет, поддерживается ли предмет техническими функциями
    
    Args:
        subject: Название предмета
        
    Returns:
        Tuple[bool, str]: (поддерживается, сообщение)
    """
    if not subject:
        return False, "Предмет не указан"
    
    subject_lower = subject.lower()
    subject_type = get_subject_type(subject)
    
    if subject_type == "technical":
        return True, f"Полная техническая поддержка для {subject}"
    elif subject_type == "natural_science":
        return True, f"Поддержка естественных наук для {subject}"
    elif subject_type == "language":
        return False, f"Языковой предмет {subject} - используйте языковые модули"
    else:
        return False, f"Гуманитарный предмет {subject} - стандартная обработка"


# =============================================================================
# ТЕСТОВЫЕ ФУНКЦИИ (для отладки)
# =============================================================================

def test_technical_functions():
    """Запускает тесты функций модуля"""
    
    print("🧪 Тестирование technical_subjects.py")
    print("=" * 50)
    
    # Тест 1: Определение типа предмета
    test_subjects = [
        ("Математика", "technical"),
        ("Физика", "technical"),
        ("Химия", "technical"),
        ("Биология", "natural_science"),
        ("География", "natural_science"),
        ("Английский язык", "language"),
        ("Литература", "humanitarian"),
        ("История", "humanitarian"),
    ]
    
    print("1. Определение типа предмета:")
    for subject, expected in test_subjects:
        result = get_subject_type(subject)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {subject:20} -> {result:20} (ожидалось: {expected})")
    
    # Тест 2: Проверка формул
    print("\n2. Обнаружение формул:")
    test_texts = [
        ("E = mc²", True),
        ("Вода имеет формулу H₂O", True),
        ("Сумма квадратов a² + b² = c²", True),
        ("Сегодня хорошая погода", False),
        ("Квадратный корень из 16 равен √16 = 4", True),
    ]
    
    for text, expected in test_texts:
        result = contains_formulas(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{text[:30]}...' -> {result} (ожидалось: {expected})")
    
    # Тест 3: Очистка текста
    print("\n3. Очистка текста:")
    test_clean_texts = [
        ("F = ma (второй закон Ньютона)", "физика"),
        ("Уравнение x² - 5x + 6 = 0", "математика"),
        ("Сегодня мы изучаем творчество Пушкина", "литература"),
    ]
    
    for text, subject in test_clean_texts:
        cleaned = clean_text_for_speech_technical(text, subject)
        print(f"  Предмет: {subject}")
        print(f"    Исходный: {text}")
        print(f"    Очищенный: {cleaned[:60]}...")
        print()
    
    # Тест 4: Анализ сложности
    print("4. Анализ сложности текста:")
    complex_text = """
    Закон Ома: U = I * R, где U - напряжение (Вольт), 
    I - сила тока (Ампер), R - сопротивление (Ом).
    Мощность вычисляется как P = U * I = I² * R = U² / R.
    """
    
    analysis = analyze_technical_complexity(complex_text, "физика")
    print(f"  Текст содержит: {analysis['formula_count']} формул, {analysis['symbol_count']} символов")
    print(f"  Сложность: {analysis['complexity']}")
    print(f"  Формулы: {analysis['formulas']}")
    
    print("\n✅ Тестирование завершено")


# =============================================================================
# ТОЧКА ВХОДА (для тестирования модуля)
# =============================================================================

if __name__ == "__main__":
    # Запуск тестов если модуль выполняется напрямую
    test_technical_functions()
    
    # Дополнительные примеры использования
    print("\n📚 Примеры использования:")
    print("-" * 50)
    
    # Пример 1: Генерация промпта для урока
    prompt = generate_technical_lesson_prompt("физика", "Законы Ньютона", "9", 15)
    print("1. Промпт для урока физики:")
    print(prompt[:300], "...")
    
    # Пример 2: Генерация практики
    practice_prompt = generate_technical_practice_prompt("математика", "Квадратные уравнения", "8", 14)
    print("\n2. Промпт для практики:")
    print(practice_prompt[:300], "...")
    
    # Пример 3: Оценка ответа
    student_answer = "F равно m умножить на a"
    correct_answer = "F = m * a"
    evaluation = evaluate_technical_answer(student_answer, correct_answer, "физика")
    print("\n3. Оценка ответа:")
    print(f"   Ответ ученика: '{student_answer}'")
    print(f"   Правильный ответ: '{correct_answer}'")
    print(f"   Оценка: {evaluation}")
