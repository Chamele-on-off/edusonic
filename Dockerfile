FROM python:3.9-slim

WORKDIR /app

# Установка системных зависимостей для dlib и ffmpeg
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    cmake \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Загрузка shape_predictor_68_face_landmarks.dat
RUN wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 \
    && bzip2 -d shape_predictor_68_face_landmarks.dat.bz2 \
    && mv shape_predictor_68_face_landmarks.dat /app/static/reference/

# Копирование и установка Python-зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["python", "app.py"]
