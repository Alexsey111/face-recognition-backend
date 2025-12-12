# Решение проблем с виртуальной средой

## 🔍 Диагностика проблем

### Проверка версии Python
```bash
python --version
python3 --version
which python
which python3
```

### Проверка pip
```bash
pip --version
pip3 --version
python -m pip --version
```

### Проверка виртуальной среды
```bash
which python
echo $VIRTUAL_ENV
```

## 🛠️ Решения

### 1. Создание чистой виртуальной среды

```bash
# Удаление старой среды (если есть)
rm -rf venv/
rm -rf .venv/

# Создание новой среды
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Обновление pip
python -m pip install --upgrade pip setuptools wheel
```

### 2. Установка системных зависимостей (Linux)

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y \
    python3-dev \
    python3-pip \
    build-essential \
    libpq-dev \
    libmagic1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev

# CentOS/RHEL
sudo yum install -y \
    python3-devel \
    python3-pip \
    gcc \
    gcc-c++ \
    postgresql-devel \
    file-devel \
    mesa-libGL \
    glibc
```

### 3. Установка зависимостей проекта

```bash
# Установка production зависимостей
pip install -r requirements.txt

# Или используя pyproject.toml
pip install -e .
pip install -e .[dev]
```

### 4. Проверка установки

```bash
# Тест основных импортов
python -c "import fastapi; print('FastAPI OK')"
python -c "import cv2; print('OpenCV OK')"
python -c "import sqlalchemy; print('SQLAlchemy OK')"
python -c "import redis; print('Redis OK')"

# Запуск тестов
python -m pytest tests/ -v
```

## 🐳 Альтернатива: Docker

Если проблемы продолжаются, используйте Docker:

```bash
# Сборка и запуск в Docker
docker-compose up -d

# Или development режим
docker-compose -f docker-compose.dev.yml up -d

# Проверка логов
docker-compose logs -f face-recognition-api
```

## 🔧 Частые проблемы и решения

### Проблема: OpenCV не устанавливается
**Решение:**
```bash
pip install opencv-python-headless==4.8.1.78
```

### Проблема: psycopg2 ошибки
**Решение:**
```bash
sudo apt install libpq-dev
pip install psycopg2-binary
```

### Проблема: Python версия не подходит
**Решение:**
```bash
# Установка Python 3.11+
sudo apt install python3.11 python3.11-venv python3.11-dev
```

### Проблема: Конфликты зависимостей
**Решение:**
```bash
# Очистка кэша pip
pip cache purge

# Установка с флагами
pip install --no-cache-dir -r requirements.txt
```

### Проблема: Permission denied
**Решение:**
```bash
# Создание venv без sudo
python3 -m venv venv --without-pip
python3 -m ensurepip --default-pip
source venv/bin/activate
```

## 📋 Чек-лист настройки

- [ ] Проверить версию Python (должна быть 3.11+)
- [ ] Создать новую виртуальную среду
- [ ] Установить системные зависимости
- [ ] Обновить pip, setuptools, wheel
- [ ] Установить зависимости проекта
- [ ] Протестировать основные импорты
- [ ] Запустить тесты
- [ ] Проверить работу приложения

## 🚀 Быстрый старт

```bash
# Клонирование репозитория
git clone <repository-url>
cd face-recognition-service

# Создание и активация виртуальной среды
python3 -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Проверка
python -c "import fastapi; print('OK')"
python -m uvicorn app.main:app --reload
```

## ⚡ Использование Poetry (альтернатива)

```bash
# Установка Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Создание проекта
poetry install

# Активация среды
poetry shell

# Запуск
poetry run uvicorn app.main:app --reload
```