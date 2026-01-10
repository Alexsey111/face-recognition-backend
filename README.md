# Face Recognition Backend

Production-ready FastAPI backend for facial recognition with detection, embedding extraction, liveness detection and verification.

See docs/ for detailed guides: QUICKSTART, DEVELOPMENT, DEPLOYMENT, API, SECURITY.
# Face Recognition Service

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

Высокопроизводительный сервис распознавания лиц с современной архитектурой, построенный на FastAPI и оптимизированный для production использования.

## 🚀 Возможности

- **Распознавание лиц** - современные алгоритмы компьютерного зрения
- **Верификация** - проверка личности по эталонным изображениям
- **Liveness detection** - определение живой личности vs фотографии
- **Batch processing** - массовая обработка изображений
- **Высокая производительность** - оптимизирован для обработки тысяч запросов
- **Масштабируемость** - горизонтальное масштабирование через Docker/Kubernetes
- **Безопасность** - шифрование данных, JWT аутентификация, rate limiting
- **Мониторинг** - Prometheus метрики, Grafana дашборды, health checks
- **API документация** - автоматическая документация с OpenAPI/Swagger

## 📋 Содержание

- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [API документация](#api-документация)
- [Конфигурация](#конфигурация)
- [Deployment](#deployment)
- [Тестирование](#тестирование)
- [Мониторинг](#мониторинг)
- [Разработка](#разработка)
- [Лицензия](#лицензия)

## 🛠 Установка

### Системные требования

- **Python** 3.11+
- **Docker** 20.10+
- **Docker Compose** 2.0+
- **4GB RAM** минимум
- **2 CPU cores** минимум
- **20GB** свободного места

### Способы установки

#### 1. Docker (Рекомендуется)

```bash
# Клонирование репозитория
git clone <repository-url>
cd face-recognition-service

# Копирование конфигурации
cp .env.example .env

# Редактирование переменных окружения
nano .env

# Запуск в production режиме
docker-compose up -d

# Или в development режиме
docker-compose -f docker-compose.dev.yml up -d
```

#### 2. Локальная установка

```bash
# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Установка зависимостей
pip install -r requirements.txt

# Настройка переменных окружения
cp .env.example .env

# Инициализация базы данных
alembic upgrade head

# Настройка MinIO
python setup_minio.py

# Запуск приложения
uvicorn app.main:create_app --host 0.0.0.0 --port 8000 --reload
```

## 🚀 Быстрый старт

### 1. Проверка работоспособности

```bash
# Health check
curl http://localhost:8000/health

# Должен вернуть:
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0"
}
```

### 2. Загрузка эталонного изображения

```bash
curl -X POST "http://localhost:8000/api/v1/reference" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@reference_image.jpg" \
  -F "label=john_doe"
```

### 3. Верификация личности

```bash
curl -X POST "http://localhost:8000/api/v1/verify" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@test_image.jpg" \
  -F "reference_id=reference_uuid"
```

### 4. Liveness проверка

```bash
curl -X POST "http://localhost:8000/api/v1/liveness" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@test_image.jpg"
```

## 📚 API документация

Полная API документация доступна по адресу:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

Основные endpoints:

| Endpoint | Method | Описание |
|----------|--------|----------|
| `/health` | GET | Проверка состояния сервиса |
| `/api/v1/reference` | POST | Загрузка эталонного изображения |
| `/api/v1/reference/{id}` | GET | Получение эталонного изображения |
| `/api/v1/reference/{id}` | DELETE | Удаление эталонного изображения |
| `/api/v1/verify` | POST | Верификация личности |
| `/api/v1/liveness` | POST | Проверка живой личности |
| `/api/v1/admin/users` | GET | Управление пользователями |
| `/api/v1/admin/stats` | GET | Статистика использования |

Подробная документация: [API.md](API.md)

## ⚙️ Конфигурация

### Переменные окружения

Основные настройки в `.env` файле:

```bash
# База данных
DATABASE_URL=postgresql://face_user:password@localhost:5432/face_recognition_db
DB_PASSWORD=your_secure_password

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=your_redis_password

# MinIO/S3
S3_ENDPOINT_URL=http://localhost:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin123
S3_BUCKET_NAME=face-recognition

# Безопасность
JWT_SECRET_KEY=your-super-secret-jwt-key
ENCRYPTION_KEY=your-256-bit-encryption-key

# Настройки приложения
DEBUG=false
ENVIRONMENT=production
MAX_UPLOAD_SIZE=10485760  # 10MB
ALLOWED_IMAGE_FORMATS=JPEG,JPG,PNG,WEBP
```

Полный список переменных: [.env.example](.env.example)

### Настройка производительности

```bash
# Database
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# Redis
REDIS_CONNECTION_POOL_SIZE=10

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_BURST=10

# ML Service
ML_SERVICE_TIMEOUT=30
```

## 🚢 Deployment

### Production deployment

#### 1. Docker Swarm

```bash
# Инициализация swarm
docker swarm init

# Деплой stack
docker stack deploy -c docker-compose.yml face-recognition

# Масштабирование
docker service scale face-recognition_face-recognition-api=3
```

#### 2. Kubernetes

```bash
# Применение манифестов
kubectl apply -f k8s/

# Проверка статуса
kubectl get pods -l app=face-recognition
```

#### 3. Docker Compose (с Nginx)

```bash
# Запуск с reverse proxy
docker-compose -f docker-compose.yml -f docker-compose.nginx.yml up -d
```

### Environment-specific конфигурации

- **Development**: `docker-compose.dev.yml`
- **Testing**: `docker-compose.test.yml`
- **Production**: `docker-compose.yml`

## 🧪 Тестирование

### Запуск тестов

```bash
# Все тесты
docker-compose -f docker-compose.test.yml up --abort-on-container-exit

# Только unit тесты
docker-compose exec face-recognition-api-test pytest tests/unit -v

# Интеграционные тесты
docker-compose exec face-recognition-api-test pytest tests/integration -v

# Performance тесты
docker-compose -f docker-compose.test.yml up performance-test
```

### Оценка модели

```bash
# С синтетическими данными
python evaluate.py --synthetic --generate-plots

# С реальными данными
python evaluate.py --data-dir ./test_data --output ./results --generate-plots

# Анализ порогов
python evaluate.py --threshold-range 0.1,0.9 --num-points 50
```

### Тестовые данные

```bash
# Структура тестовых данных
test_data/
├── genuine/          # Genuine pairs (один человек)
├── impostor/         # Impostor pairs (разные люди)
└── liveness/         # Liveness тесты
    ├── live/         # Живые лица
    └── spoof/        # Фотографии/маски
```

## 📊 Мониторинг

### Health Checks

```bash
# Общий health check
curl http://localhost:8000/health

# Детальная информация
curl http://localhost:8000/health/detailed

# Готовность к трафику
curl http://localhost:8000/health/ready
```

### Метрики

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **API метрики**: http://localhost:8000/metrics

### Логирование

```bash
# Просмотр логов
docker-compose logs -f face-recognition-api

# Логи определенного сервиса
docker-compose logs -f postgres
docker-compose logs -f redis
docker-compose logs -f minio

# Поиск по логам
docker-compose logs face-recognition-api | grep ERROR
```

### Мониторинг базы данных

```bash
# pgAdmin (Development)
http://localhost:5050
# Email: admin@face-recognition.local
# Password: admin

# Redis Commander (Development)
http://localhost:8081
```

### MinIO Console

```bash
# MinIO Management Console
http://localhost:9001
# Access Key: minioadmin
# Secret Key: minioadmin123
```

## 🔧 Разработка

### Структура проекта

```
face-recognition-service/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI приложение
│   ├── config.py            # Конфигурация
│   ├── models/              # Pydantic модели
│   ├── routes/              # API endpoints
│   ├── services/            # Бизнес-логика
│   ├── middleware/          # Middleware компоненты
│   ├── utils/               # Утилиты
│   └── db/                  # Database слой
├── alembic/                 # Database миграции
├── tests/                   # Тесты
├── docs/                    # Документация
├── docker-compose*.yml      # Docker конфигурации
├── requirements*.txt        # Python зависимости
├── evaluate.py             # Оценка модели
└── setup_minio.py          # Настройка MinIO
```

### Разработка API

```bash
# Запуск в development режиме
docker-compose -f docker-compose.dev.yml up -d

# Автоматическая перезагрузка при изменении кода
# Hot reload включен в development конфигурации

# Отладка
docker-compose -f docker-compose.dev.yml exec face-recognition-api-dev bash
```

### Добавление новых endpoints

1. Создайте route в `app/routes/`
2. Добавьте схемы в `app/models/`
3. Добавьте бизнес-логику в `app/services/`
4. Обновите документацию в `API.md`

### Code Quality

```bash
# Форматирование кода
black app/ tests/
isort app/ tests/

# Линтинг
flake8 app/ tests/
mypy app/

# Безопасность
bandit -r app/
safety check
```

### Database миграции

```bash
# Создание новой миграции
alembic revision --autogenerate -m "Description"

# Применение миграций
alembic upgrade head

# Откат миграции
alembic downgrade -1
```

## 🔒 Безопасность

### Аутентификация

- **JWT токены** для API аутентификации
- **API ключи** для интеграций
- **OAuth2** поддержка (опционально)

### Авторизация

- **RBAC** (Role-Based Access Control)
- **Права доступа** на уровне ресурсов
- **Rate limiting** для предотвращения атак

### Шифрование

- **AES-256-GCM** для чувствительных данных
- **TLS/SSL** для транспортного уровня
- **Hashing** паролей с bcrypt

### Безопасность изображений

- **Валидация форматов** изображений
- **Проверка размера** файлов
- **Сканирование на вирусы** (опционально)
- **Анонимизация** метаданных

## 📈 Производительность

### Оптимизации

- **Асинхронная обработка** запросов
- **Connection pooling** для БД
- **Кэширование** в Redis
- **CDN** для статических файлов
- **Горизонтальное масштабирование**

### Бенчмарки

```
Benchmark Results (typical hardware):
- Image processing: ~50ms per image
- Face detection: ~20ms
- Embedding generation: ~30ms
- Verification: ~10ms
- Database queries: ~5ms
- Total response time: <100ms (95th percentile)
```

### Масштабирование

```bash
# Горизонтальное масштабирование API
docker-compose up -d --scale face-recognition-api=3

# Load balancing через Nginx
# Автоматическое распределение нагрузки
```

## 🆘 Troubleshooting

### Частые проблемы

#### 1. Ошибка подключения к БД

```bash
# Проверка статуса PostgreSQL
docker-compose logs postgres

# Проверка connectivity
docker-compose exec face-recognition-api python -c "
from app.db.database import check_database_connection
check_database_connection()
"
```

#### 2. Проблемы с MinIO

```bash
# Проверка MinIO health
curl http://localhost:9000/minio/health/live

# Пересоздание buckets
python setup_minio.py
```

#### 3. Ошибки памяти

```bash
# Мониторинг использования памяти
docker stats

# Увеличение лимитов
# В docker-compose.yml:
deploy:
  resources:
    limits:
      memory: 2G
```

#### 4. Проблемы с производительностью

```bash
# Профилирование
docker-compose exec face-recognition-api python -m cProfile -o profile.stats main.py

# Анализ метрик в Grafana
# Проверка bottleneck'ов
```

### Логи и отладка

```bash
# Включение debug режима
DEBUG=true ENVIRONMENT=development docker-compose up

# Подробное логирование
LOG_LEVEL=DEBUG docker-compose up

# Интерактивная отладка
docker-compose exec face-recognition-api python -m pdb app/main.py
```

## 📚 Дополнительные ресурсы

- [API документация](API.md)
- [Deployment Guide](docs/deployment.md)
- [Architecture Overview](docs/architecture.md)
- [Security Best Practices](docs/security.md)
- [Performance Tuning](docs/performance.md)

## 🤝 Вклад в проект

1. Fork репозитория
2. Создайте feature branch (`git checkout -b feature/amazing-feature`)
3. Commit изменения (`git commit -m 'Add amazing feature'`)
4. Push в branch (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

### Guidelines

- Следуйте PEP 8 для Python кода
- Добавляйте тесты для новой функциональности
- Обновляйте документацию
- Используйте meaningful commit messages

## 📄 Лицензия

Этот проект распространяется под лицензией MIT. См. файл [LICENSE](LICENSE) для деталей.

## 👥 Команда

- **Разработчик**: [Your Name]
- **DevOps**: [DevOps Engineer]
- **ML Engineer**: [ML Engineer]

## 📞 Поддержка

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: support@yourcompany.com

---

**⭐ Star этот проект, если он был полезен!**