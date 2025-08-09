FROM python:3.10-slim-bullseye

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    ffmpeg \
    espeak-ng \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Временный файл для корректного кэширования
COPY requirements.txt .

# Установка зависимостей с четким порядком
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.1.0 torchaudio==2.1.0 --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Создание директорий
RUN mkdir -p \
    /app/static/audio \
    /app/static/models \
    /app/static/reference \
    /app/materials

EXPOSE 5000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]