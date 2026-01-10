# Dockerfile
# Multi-stage production Dockerfile для Face Recognition Service
# Этап 1: Сборка зависимостей
FROM python:3.11-slim as builder

# Установка системных зависимостей для сборки
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создание виртуального окружения
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Копирование и установка Python зависимостей
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Этап 2: Runtime образ
FROM python:3.11-slim as runtime

# Установка только необходимых системных библиотек для runtime
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создание пользователя для приложения
RUN groupadd -r app && useradd -r -g app -d /home/app -s /bin/bash app

# Копирование виртуального окружения из builder этапа
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Установка рабочей директории
WORKDIR /app

# Создание необходимых директорий
RUN mkdir -p /app/logs /app/uploads /app/temp && \
    chown -R app:app /app

# Копирование исходного кода
COPY --chown=app:app . .

# Переключение на пользователя app
USER app

# Переменные окружения
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

# Проверка здоровья контейнера
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
CMD curl -f http://localhost:8000/health || exit 1

# Экспорт порта
EXPOSE 8000

# Команда запуска
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]