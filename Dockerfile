FROM python:3.9-slim-bullseye

RUN apt-get update && apt-get install -y \
    ffmpeg \
    espeak-ng \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Создаем необходимые директории
RUN mkdir -p \
    static/audio \
    static/tmp \
    static/lessons \
    static/reference \
    materials \
    cache

# Добавляем переменную окружения для Docker
ENV DOCKER_ENV=true
ENV OLLAMA_HOST=http://host.docker.internal:11434

EXPOSE 5000

CMD ["python", "app.py"]
