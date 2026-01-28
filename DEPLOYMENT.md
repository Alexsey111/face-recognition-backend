# Deployment

This document provides comprehensive deployment instructions for the Face Recognition Service.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Configuration](#environment-configuration)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [High Availability Deployment](#high-availability-deployment)
- [Security Configuration](#security-configuration)
- [Monitoring Setup](#monitoring-setup)
- [Backup and Recovery](#backup-and-recovery)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|----------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8+ GB |
| Storage | 20 GB | 50+ GB |
| Docker | 20.10+ | 23.0+ |
| Docker Compose | 2.0+ | 2.20+ |

### Required Services

| Service | Version | Purpose |
|---------|---------|---------|
| PostgreSQL | 15+ | Primary database |
| Redis | 7+ | Cache and sessions |
| MinIO | Latest | Object storage |

### Network Requirements

- Open ports: 80, 443, 8000
- DNS records for API and monitoring
- SSL certificates (Let's Encrypt or purchased)

## Environment Configuration

### Required Environment Variables

```bash
# Database
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=face_recognition
DATABASE_USER=postgres
DATABASE_PASSWORD=your_secure_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# MinIO/S3
S3_ENDPOINT=localhost:9000
S3_ACCESS_KEY=your_access_key
S3_SECRET_KEY=your_secret_key
S3_BUCKET=face-recognition
S3_SECURE=false

# Security
JWT_SECRET_KEY=your_jwt_secret_key
ENCRYPTION_KEY=your_encryption_key
SECRET_KEY=your_secret_key

# ML Configuration
LOCAL_ML_ENABLE_CUDA=false
LOCAL_ML_DEVICE=cpu

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Security-Sensitive Variables

Never commit these to version control:

```bash
# Generate secure keys
openssl rand -hex 32  # JWT_SECRET_KEY
openssl rand -hex 32  # ENCRYPTION_KEY
openssl rand -hex 64  # SECRET_KEY
```

### Using Environment Files

```bash
# Copy example environment
cp .env.example .env

# Edit with your values
nano .env

# Verify syntax
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('OK')"
```

## Docker Deployment

### Basic Docker Build

```bash
# Build the image
docker build -t face-recognition:latest .

# Tag for registry
docker tag face-recognition:latest ghcr.io/your-org/face-recognition:latest

# Push to registry
docker push ghcr.io/your-org/face-recognition:latest
```

### Docker Compose Deployment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f face-recognition

# Check status
docker-compose ps

# Stop all services
docker-compose down
```

### Production Docker Compose

```bash
# Use production configuration
docker-compose -f docker-compose.prod.yml up -d

# With monitoring
docker-compose -f docker-compose.prod.yml -f docker-compose.monitoring.yml up -d
```

### Docker Compose Configuration

#### Development (docker-compose.yml)

```yaml
services:
  face-recognition:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=development
      - DEBUG=true
    volumes:
      - ./app:/app/app
    env_file:
      - .env
    depends_on:
      - postgres
      - redis
      - minio

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: face_recognition
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass redis_password

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
```

#### Production (docker-compose.prod.yml)

```yaml
services:
  face-recognition:
    image: ghcr.io/your-org/face-recognition:latest
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          memory: 2G
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    restart_policy:
      condition: on-failure
      delay: 5s
      max_attempts: 3
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s

  postgres:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    restart: unless-stopped

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    restart: unless-stopped
```

### Docker Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ENVIRONMENT` | Yes | `development` | Application environment |
| `DATABASE_HOST` | Yes | - | PostgreSQL host |
| `DATABASE_PORT` | Yes | `5432` | PostgreSQL port |
| `DATABASE_NAME` | Yes | - | Database name |
| `DATABASE_USER` | Yes | - | Database user |
| `DATABASE_PASSWORD` | Yes | - | Database password |
| `REDIS_HOST` | Yes | - | Redis host |
| `REDIS_PORT` | Yes | `6379` | Redis port |
| `REDIS_PASSWORD` | Yes | - | Redis password |
| `S3_ENDPOINT` | Yes | - | MinIO/S3 endpoint |
| `S3_ACCESS_KEY` | Yes | - | S3 access key |
| `S3_SECRET_KEY` | Yes | - | S3 secret key |
| `S3_BUCKET` | Yes | - | S3 bucket name |
| `JWT_SECRET_KEY` | Yes | - | JWT secret key |
| `ENCRYPTION_KEY` | Yes | - | Encryption key |
| `SECRET_KEY` | Yes | - | Application secret key |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `LOCAL_ML_ENABLE_CUDA` | No | `false` | Enable CUDA |
| `LOCAL_ML_DEVICE` | No | `cpu` | ML device (cpu/cuda) |

## Kubernetes Deployment

### Prerequisites

- Kubernetes 1.25+
- Helm 3.10+
- kubectl configured
- Ingress controller (NGINX/Traefik)
- Cert-manager for TLS

### Helm Deployment

```bash
# Add Helm repository
helm repo add face-recognition https://your-helm-repo.com
helm repo update

# Install chart
helm install face-recognition face-recognition/face-recognition \
  --namespace face-recognition \
  --create-namespace \
  -f values-production.yaml
```

### Kubernetes Values (values-production.yaml)

```yaml
replicaCount: 3

image:
  repository: ghcr.io/your-org/face-recognition
  tag: latest
  pullPolicy: Always

service:
  type: ClusterIP
  port: 8000

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: api.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: face-recognition-tls
      hosts:
        - api.yourdomain.com

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80

env:
  - name: ENVIRONMENT
    value: production
  - name: LOG_LEVEL
    value: INFO
  - name: LOCAL_ML_ENABLE_CUDA
    value: "false"

postgresql:
  enabled: true
  postgresqlUsername: postgres
  postgresqlPassword: postgres
  postgresqlDatabase: face_recognition

redis:
  enabled: true
  auth:
    enabled: true
    password: redis_password

minio:
  enabled: true
  rootUser: minioadmin
  rootPassword: minioadmin
```

### Kubernetes Manifests

#### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: face-recognition
  namespace: face-recognition
spec:
  replicas: 3
  selector:
    matchLabels:
      app: face-recognition
  template:
    metadata:
      labels:
        app: face-recognition
    spec:
      containers:
        - name: face-recognition
          image: ghcr.io/your-org/face-recognition:latest
          ports:
            - containerPort: 8000
          envFrom:
            - secretRef:
                name: face-recognition-secret
          resources:
            limits:
              memory: 4Gi
              cpu: 2000m
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
```

#### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: face-recognition
  namespace: face-recognition
spec:
  selector:
    app: face-recognition
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP
```

#### Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: face-recognition
  namespace: face-recognition
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "30"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - api.yourdomain.com
      secretName: face-recognition-tls
  rules:
    - host: api.yourdomain.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: face-recognition
                port:
                  number: 80
```

#### Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: face-recognition-secret
  namespace: face-recognition
type: Opaque
stringData:
  DATABASE_HOST: postgres-host
  DATABASE_PORT: "5432"
  DATABASE_NAME: face_recognition
  DATABASE_USER: postgres
  DATABASE_PASSWORD: postgres-password
  REDIS_HOST: redis-host
  REDIS_PORT: "6379"
  REDIS_PASSWORD: redis-password
  S3_ENDPOINT: s3.amazonaws.com
  S3_ACCESS_KEY: your-access-key
  S3_SECRET_KEY: your-secret-key
  S3_BUCKET: face-recognition
  JWT_SECRET_KEY: your-jwt-secret
  ENCRYPTION_KEY: your-encryption-key
  SECRET_KEY: your-secret-key
```

## High Availability Deployment

### Architecture

```
                    ┌─────────────────────────────────────┐
                    │           Traefik (LB)               │
                    │    (port 80/443, SSL termination)    │
                    └──────────────┬──────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
    ┌──────▼──────┐         ┌──────▼──────┐         ┌──────▼──────┐
    │ face-api-1  │         │ face-api-2  │         │ face-api-3  │
    │ (replica)   │         │ (replica)   │         │ (replica)   │
    └──────┬──────┘         └──────┬──────┘         └──────┬──────┘
           │                       │                       │
           └───────────────────────┼───────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
   ┌────▼────┐               ┌────▼────┐               ┌────▼────┐
   │Postgres │               │  Redis  │               │  MinIO  │
   │Primary  │◄─────────────►│ Master  │               │ Storage │
   └────┬────┘               └────┬────┘               └─────────┘
        │                         │
   ┌────▼────┐               ┌────▼────┐
   │Postgres │               │  Redis  │
   │ Replica │               │ Replica │
   └─────────┘               └─────────┘

        ┌─────────────────────────────────────────────────────┐
        │              Monitoring Stack                        │
        │  Prometheus ←── Jaeger (tracing) ←── Grafana        │
        └─────────────────────────────────────────────────────┘
```

### High Availability Features

| Feature | Configuration |
|---------|---------------|
| **Horizontal Scaling** | 3 API replicas (configurable) |
| **Load Balancing** | Traefik with DNS round-robin |
| **Health Checks** | 30s interval, 3 retries |
| **Auto-restart** | on-failure, max 3 attempts |
| **Database Replication** | Primary + Replica |
| **Redis Replication** | Master + Replica |
| **SSL/TLS** | Let's Encrypt via Traefik |
| **Monitoring** | Prometheus + Grafana + Jaeger |

### Resource Limits

| Service | CPU | Memory |
|---------|-----|--------|
| face-api | 0.5-2 cores | 512MB-2GB |
| postgres | 2 cores | 2GB |
| redis | 1 core | 1GB |
| traefik | 0.5 cores | 256MB |
| prometheus | 1 core | 1GB |
| grafana | 0.5 cores | 512MB |

### High Availability Deployment

```bash
# Deploy with 3 API replicas
docker stack deploy -c docker-compose.ha.yml face-recognition

# Or with docker-compose
docker-compose -f docker-compose.ha.yml up -d
```

## Security Configuration

### SSL/TLS Setup

#### Using Let's Encrypt (Traefik)

```yaml
# docker-compose.ha.yml
services:
  traefik:
    image: traefik:v2.10
    command:
      - "--providers.docker"
      - "--providers.docker.exposedByDefault=false"
      - "--entrypoints.websecure.address=:443"
      - "--certificatesresolvers.letsencryptresolver.acme.httpchallenge=true"
      - "--certificatesresolvers.letsencryptresolver.acme.httpchallenge.entrypoint=web"
      - "--certificatesresolvers.letsencryptresolver.acme.storage=/letsencrypt/acme.json"
    volumes:
      - letsencrypt:/letsencrypt
    ports:
      - "443:443"
    environment:
      - ACME_EMAIL=your-email@example.com
```

#### Using Custom Certificates

```bash
# Generate self-signed certificate (development only)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes \
  -subj "/CN=localhost" -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"

# Use certificate in deployment
docker-compose -f docker-compose.prod.yml up -d \
  -e SSL_CERT=$(cat cert.pem) \
  -e SSL_KEY=$(cat key.pem)
```

### Firewall Configuration

```bash
# Allow only necessary ports
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 22/tcp

# Block all other ports
ufw default deny incoming
ufw default allow outgoing
```

### Security Headers

```python
# In app/middleware/security_headers.py
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response
```

## Monitoring Setup

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: face-recognition
    static_configs:
      - targets: ['face-recognition:8000']
    metrics_path: /metrics
    scrape_interval: 15s
```

### Alert Rules

```yaml
# alerts.yml
groups:
  - name: face-recognition-alerts
    rules:
      - alert: API Down
        expr: up == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Face Recognition API is down"

      - alert: High Error Rate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"

      - alert: High Latency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API latency detected"
```

### Grafana Dashboards

Import dashboards from `grafana/provisioning/dashboards/`:

```bash
# Deploy Grafana with pre-configured dashboards
docker-compose -f docker-compose.monitoring.yml up -d grafana
```

### Health Check Endpoints

| Endpoint | Description |
|----------|-------------|
| `/health` | Basic health check |
| `/health/detailed` | Detailed health with all services |
| `/health/ready` | Readiness probe |
| `/health/live` | Liveness probe |
| `/metrics` | Prometheus metrics |

### Health Check Scripts

Для проверки работоспособности сервиса доступны скрипты:

#### Локальный health check

```bash
# Проверка всех endpoints
make health-check

# Или напрямую
bash scripts/health_check.sh http://localhost 8000
```

Скрипт проверяет:
- `/health` — базовый health check с повторами
- `/metrics/prometheus` — наличие метрик
- `/upload/supported-formats` — поддерживаемые форматы
- Время ответа (< 500ms)

#### Docker health check

```bash
# Проверить все контейнеры
make docker-health

# Только API
make docker-health-api

# Напрямую
bash scripts/docker_health_check.sh all        # все сервисы
bash scripts/docker_health_check.sh api        # только API
bash scripts/docker_health_check.sh postgres   # PostgreSQL
bash scripts/docker_health_check.sh redis      # Redis
bash scripts/docker_health_check.sh minio      # MinIO
```

#### CI/CD интеграция

```bash
# В CI pipeline
- name: Health Check
  run: |
    make docker-up
    make health-check
    make docker-health
```

#### Автоматический healthcheck в Docker

В `docker-compose.prod.yml` уже настроен healthcheck:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

## Backup and Recovery

### Database Backup

```bash
# Schedule daily backups
0 2 * * * pg_dump -h localhost -U postgres -d face_recognition | gzip > /backup/face_recognition_$(date +\%Y\%m\%d).sql.gz

# Backup to S3
0 3 * * * aws s3 cp /backup/face_recognition_$(date +\%Y\%m\%d).sql.gz s3://your-backup-bucket/
```

### Point-in-Time Recovery

```bash
# Restore to specific point in time
pg_restore -h localhost -U postgres -d face_recognition \
  --verbose \
  --clean \
  --if-exists \
  --jobs=4 \
  backup.dump
```

### Object Storage Backup

```bash
# Sync MinIO buckets
mc mirror minio/face-recognition s3/backup/face-recognition

# Schedule daily sync
0 4 * * * mc mirror minio/face-recognition s3/backup/face-recognition
```

### Disaster Recovery Plan

1. **Identify the incident**
2. **Assess the impact**
3. **Activate recovery plan**
4. **Restore from backup**
5. **Verify data integrity**
6. **Resume operations**
7. **Document the incident**

## Troubleshooting

### Common Issues

#### Container won't start

```bash
# Check container logs
docker-compose logs face-recognition

# Check environment variables
docker-compose exec face-recognition env

# Verify database connection
docker-compose exec face-recognition python -c "import asyncio; asyncio.run(db.connect())"
```

#### Database connection issues

```bash
# Test database connection
docker-compose exec face-recognition python -c "
import asyncpg
import asyncio

async def test():
    conn = await asyncpg.connect(
        host='postgres',
        port=5432,
        user='postgres',
        password='postgres',
        database='face_recognition'
    )
    await conn.close()
    print('Database connection successful')

asyncio.run(test())
"
```

#### Memory issues

```bash
# Check container memory usage
docker stats

# Increase memory limit
docker-compose -f docker-compose.prod.yml up -d --scale face-recognition=2
```

#### Performance issues

```bash
# Check database queries
docker-compose exec face-recognition python -c "
import asyncio
from app.services.database_service import db

async def check_slow_queries():
    result = await db.execute('''
        SELECT query, duration_ms
        FROM pg_stat_statements
        ORDER BY total_time DESC
        LIMIT 10
    ''')
    print(result)

asyncio.run(check_slow_queries())
"
```

### Debug Mode

```bash
# Enable debug logging
LOG_LEVEL=DEBUG docker-compose up -d

# View detailed logs
docker-compose logs -f --tail=100 face-recognition
```

### Rollback Procedure

```bash
# Rollback to previous image
docker-compose -f docker-compose.prod.yml pull face-recognition
docker-compose -f docker-compose.prod.yml up -d face-recognition

# Rollback database migration
alembic downgrade -1
```

### Performance Tuning

```python
# In app/db/database.py
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
)
```

### Logging Configuration

```python
# In app/utils/logger.py
import logging
import sys

LOG_FORMAT = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(levelname)s %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json",
            "stream": sys.stdout,
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
}
```

### Security Scanning

```bash
# Run security scans
bandit -r app/
safety check -r requirements.txt
trivy image face-recognition:latest
```

### Performance Benchmarks

```bash
# Run load tests
locust -f tests/load/locustfile.py --users 100 --spawn-rate 10 --run-time 10m

# Run benchmarks
pytest tests/performance/ -v --benchmark-only
```

### Monitoring Alerts

| Alert | Threshold | Action |
|-------|-----------|--------|
| API Down | 2 min | Restart container |
| High Error Rate | 5% 5xx | Alert on-call |
| High Latency | P95 > 2s | Scale horizontally |
| Memory Usage | > 90% | Restart container |
| Disk Usage | > 80% | Clean up logs |
| CPU Usage | > 80% | Scale horizontally |

## Maintenance

### Rolling Updates

```bash
# Update without downtime
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d --no-deps face-recognition
```

### Database Migrations

```bash
# Run migrations
docker-compose exec face-recognition alembic upgrade head

# Check migration status
docker-compose exec face-recognition alembic current

# Create new migration
docker-compose exec face-recognition alembic revision --autogenerate -m "description"
```

### Log Rotation

```bash
# Configure log rotation
cat > /etc/logrotate.d/face-recognition <<EOF
/var/log/face-recognition/*.log {
    daily
    rotate 7
    compress
    delaycompress
    notifempty
    create 0640 root adm
}
EOF
```

### Certificate Renewal

```bash
# Check certificate expiry
openssl x509 -enddate -noout -in /etc/letsencrypt/live/yourdomain.com/fullchain.pem

# Renew certificates
certbot renew --quiet

# Reload Traefik
docker-compose exec traefik kill -HUP 1
```