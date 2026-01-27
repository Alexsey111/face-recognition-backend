# =============================================================================
# Dockerfile для Face Recognition Service
# =============================================================================
# Поддержка GPU: Используйте nvidia/cuda base image
#   docker build --build-arg BASE_IMAGE=nvidia/cuda:12.2-cudnn8-devel ...
#   или docker-compose -f docker-compose.gpu.yml
# =============================================================================

# =============================================================================
# Stage 1: Builder - установка зависимостей
# =============================================================================
ARG PYTHON_VERSION=3.12
ARG BASE_IMAGE=python:${PYTHON_VERSION}-slim

FROM ${BASE_IMAGE} as builder

LABEL stage=builder

# Установка системных зависимостей для сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    cmake \
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
    wget \
    && rm -rf /var/lib/apt/lists/*

# Создание виртуального окружения
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Копирование только requirements для кэширования слоя
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Runtime - финальный образ
# =============================================================================
# Для GPU используйте: docker build --build-arg BASE_IMAGE=nvidia/cuda:12.2-cudnn8-devel ...
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE} as runtime

LABEL maintainer="your-team@example.com" \
      version="1.0.0" \
      description="Face Recognition Service API with GPU support"

# Labels для GPU (если используется nvidia/cuda)
ARG CUDA_VERSION=not-used
LABEL com.nvidia.cuda.version="${CUDA_VERSION}" \
      com.nvidia.cudnn.version="8"

# Установка только runtime библиотек
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    libv4l-0 \
    libxvidcore4 \
    libx264-160 \
    libjpeg62-turbo \
    libpng16-16 \
    libtiff5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создание непривилегированного пользователя
RUN groupadd -r app && \
    useradd -r -g app -d /home/app -s /sbin/nologin -c "Application user" app && \
    mkdir -p /home/app && \
    chown -R app:app /home/app

# Копирование виртуального окружения из builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Установка рабочей директории
WORKDIR /app

# Создание необходимых директорий с правильными правами
RUN mkdir -p /app/logs /app/uploads /app/temp /app/models && \
    chown -R app:app /app

# Копирование исходного кода
COPY --chown=app:app . .

# Переключение на пользователя app
USER app

# Переменные окружения
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    WORKERS=4

# Проверка здоровья контейнера
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Экспорт порта
EXPOSE 8000

# Entrypoint для гибкой конфигурации
ENTRYPOINT ["uvicorn", "app.main:app"]

# Аргументы по умолчанию (можно переопределить)
CMD ["--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
