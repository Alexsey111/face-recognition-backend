# Health Check Scripts

Скрипты для проверки работоспособности сервиса face-recognition-service.

## Скрипты

### 1. `health_check.sh`

Локальный health check для запущенного сервиса.

**Использование:**
```bash
# Базовая проверка
./scripts/health_check.sh http://localhost 8000

# Только URL (порт по умолчанию 8000)
./scripts/health_check.sh http://localhost
```

**Что проверяет:**
- `/health` — базовый health endpoint (с повторами до 10 попыток)
- `/metrics/prometheus` — наличие метрик `verification_requests_total`
- `/upload/supported-formats` — поддержка форматов (проверка HEIC)
- `/ready` — readiness probe (опционально)
- Время ответа (< 500ms)

**Выходные коды:**
- `0` — все проверки пройдены
- `1` — одна или более проверок не пройдены

### 2. `docker_health_check.sh`

Проверка Docker контейнеров через docker-compose.

**Использование:**
```bash
# Все сервисы
./scripts/docker_health_check.sh all

# Конкретный сервис
./scripts/docker_health_check.sh api
./scripts/docker_health_check.sh postgres
./scripts/docker_health_check.sh redis
./scripts/docker_health_check.sh minio
```

**Что проверяет:**
- Статус контейнера (running/stopped)
- Health status контейнера (healthy/unhealthy)
- HTTP endpoints для API

**Выходные коды:**
- `0` — все проверки пройдены
- `1` — одна или более проверок не пройдены

## Makefile команды

```bash
# Локальный health check
make health-check

# Docker health check
make docker-health     # все контейнеры
make docker-health-api # только API
```

## Примеры

### Проверка после развертывания

```bash
# Запуск сервиса
make docker-up

# Ожидание запуска
sleep 10

# Health check
make health-check
```

### CI/CD pipeline

```yaml
- name: Deploy and Health Check
  run: |
    make docker-up
    make health-check
    make docker-health
```

### Проверка в Kubernetes

```bash
# Проверка подов
kubectl get pods -l app=face-recognition

# Проверка health
kubectl exec -it deployment/face-recognition-api -- curl -f http://localhost:8000/health

# Проверка метрик
kubectl exec -it deployment/face-recognition-api -- curl http://localhost:8000/metrics
```

## Требования

- `curl` — для HTTP запросов
- `bc` — для математических вычислений (время ответа)
- Docker и docker-compose — для `docker_health_check.sh`