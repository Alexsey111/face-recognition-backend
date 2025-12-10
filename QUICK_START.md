# 🚀 КРАТКОЕ РУКОВОДСТВО ПО ИСПОЛЬЗОВАНИЮ - Face Recognition Service

## ⚡ Быстрый старт (5 минут)

### 1. Запуск production окружения
```bash
# Клонирование
git clone <repository-url>
cd face-recognition-service

# Настройка
cp .env.example .env
# Отредактируйте .env файл с вашими параметрами

# Запуск
docker-compose up -d
```

### 2. Проверка работоспособности
```bash
# Health check
curl http://localhost:8000/health

# API документация
open http://localhost:8000/docs
```

### 3. Базовое использование API

#### Загрузка эталонного изображения
```bash
curl -X POST "http://localhost:8000/api/v1/reference" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@reference.jpg" \
  -F "label=john_doe"
```

#### Верификация
```bash
curl -X POST "http://localhost:8000/api/v1/verify" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@test.jpg" \
  -F "reference_id=ref_uuid"
```

## 🧪 Тестирование и evaluation

### Запуск тестового стенда
```bash
# Полное тестирование
docker-compose -f docker-compose.test.yml up --abort-on-container-exit

# Только evaluation модели
python evaluate.py --synthetic --generate-plots
```

### Структура результатов evaluation
```
evaluation_results/
├── evaluation_results_20240101_120000.json
├── evaluation_results_20240101_120000.csv
├── roc_curve_20240101_120000.png
├── metrics_distribution_20240101_120000.png
└── optimal_threshold_20240101_120000.txt
```

## 📊 Мониторинг

### Доступные сервисы
- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **MinIO Console**: http://localhost:9001
- **pgAdmin**: http://localhost:5050
- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090

### Логи
```bash
# Все логи
docker-compose logs -f

# Только API
docker-compose logs -f face-recognition-api

# Поиск ошибок
docker-compose logs | grep ERROR
```

## 🔧 Разработка

### Development окружение
```bash
# Запуск в development режиме
docker-compose -f docker-compose.dev.yml up -d

# Автоматическая перезагрузка при изменении кода
# Hot reload включен
```

### Структура проекта
```
face-recognition-service/
├── app/                    # Основной код
│   ├── main.py            # FastAPI приложение
│   ├── config.py          # Конфигурация
│   ├── routes/            # API endpoints
│   ├── services/          # Бизнес-логика
│   ├── models/            # Pydantic модели
│   └── db/                # Database слой
├── evaluate.py            # Оценка модели
├── docker-compose*.yml    # Docker конфигурации
├── README.md              # Основная документация
├── API.md                 # API документация
└── requirements*.txt      # Python зависимости
```

## 🛡️ Безопасность

### Переменные окружения
Обязательно настройте в `.env`:
```bash
# Безопасность
JWT_SECRET_KEY=your-super-secret-jwt-key-make-it-very-long
ENCRYPTION_KEY=your-256-bit-encryption-key-for-embeddings

# База данных
DB_PASSWORD=your_secure_database_password
REDIS_PASSWORD=your_redis_password

# MinIO
MINIO_ROOT_USER=your_minio_user
MINIO_ROOT_PASSWORD=your_minio_password
```

### API Keys
```bash
# Создание API ключа
curl -X POST "http://localhost:8000/api/v1/auth/api-key" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_uuid", "name": "My App"}'
```

## 📈 Производительность

### Оптимизация
- Увеличьте `DATABASE_POOL_SIZE` для высокой нагрузки
- Настройте `REDIS_CONNECTION_POOL_SIZE`
- Используйте горизонтальное масштабирование:
```bash
docker-compose up -d --scale face-recognition-api=3
```

### Мониторинг производительности
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus метрики**: http://localhost:8000/metrics
- **Health checks**: http://localhost:8000/health/detailed

## 🔧 Troubleshooting

### Частые проблемы

#### 1. Сервис не запускается
```bash
# Проверка логов
docker-compose logs service_name

# Перезапуск
docker-compose restart
```

#### 2. Проблемы с БД
```bash
# Проверка подключения
docker-compose exec face-recognition-api python -c "
from app.db.database import check_database_connection
check_database_connection()
"
```

#### 3. MinIO не работает
```bash
# Пересоздание buckets
python setup_minio.py
```

## 📚 Дополнительная информация

### Документация
- **README.md** - Полное руководство
- **API.md** - Подробная API документация
- **Swagger UI** - Интерактивная документация

### Поддержка
- **Issues**: Создайте issue в репозитории
- **Логи**: Всегда прикладывайте логи при обращении
- **Версия**: Проверьте `GET /health` для версии системы

---

**🎉 Готово к использованию! Все компоненты настроены и протестированы.**