# Базовый образ с Python 3.10
FROM python:3.10-slim-bullseye

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    ffmpeg \
    espeak-ng \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем зависимости отдельно для лучшего кэширования
COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY . .

# Создаем необходимые директории
RUN mkdir -p \
    /app/static/audio \
    /app/static/models \
    /app/static/reference \
    /app/materials

# Устанавливаем переменные окружения для оптимизации
ENV CUDA_VISIBLE_DEVICES=-1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# Открываем порт для FastAPI
EXPOSE 5000

# Команда запуска (без GPU)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "2"]