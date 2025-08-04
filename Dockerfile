
FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование requirements.txt первым для кэширования
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование остальных файлов
COPY . .

# Создание необходимых директорий
RUN mkdir -p \
    /app/static/audio \
    /app/static/tmp \
    /app/static/lessons \
    /app/materials

# Загрузка моделей при сборке (опционально)
# RUN python -c "from llm import LessonLLM; LessonLLM()"
# RUN python -c "from tts import TextToSpeech; TextToSpeech()"

# Порт, который будет слушать приложение
EXPOSE 5000

# Команда запуска
CMD ["python", "app.py"]
