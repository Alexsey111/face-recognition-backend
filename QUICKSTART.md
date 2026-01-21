# Quick Start Guide

Get the Face Recognition API running in 5 minutes!

## Prerequisites

- Docker & Docker Compose installed
- Python 3.12+ (for local development)
- 4GB RAM minimum
- Git

## Quick Start (Docker)

```bash
# 1. Clone repository
git clone https://github.com/Alexsey111/face-recognition-backend.git
cd face-recognition-backend/face-recognition-service

# 2. Create environment file
cp .env.example .env

# 3. Start all services
docker-compose up -d

# 4. Check health
curl http://localhost:8000/api/v1/health

# 5. Access API docs
open http://localhost:8000/docs
```

## Quick Start (Local Development)

```bash
# 1. Clone repository
git clone https://github.com/Alexsey111/face-recognition-backend.git
cd face-recognition-backend/face-recognition-service

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Start infrastructure (PostgreSQL, Redis, MinIO)
docker-compose up -d postgres redis minio

# 5. Run migrations
alembic upgrade head

# 6. Start API
uvicorn app.main:app --reload

# 7. Access at http://localhost:8000/docs
```

## First API Call

```bash
# Register user
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123!",
    "full_name": "Test User"
  }'

# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=SecurePass123!"

# Use the token from response for authenticated requests
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=SecurePass123!" | jq -r '.access_token')

# Check current user
curl -X GET http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer $TOKEN"
```

## Available Services

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:8000 | Main API endpoint |
| API Docs | http://localhost:8000/docs | Swagger UI |
| ReDoc | http://localhost:8000/redoc | ReDoc documentation |
| MinIO Console | http://localhost:9001 | MinIO web console |
| MinIO API | http://localhost:9000 | MinIO S3-compatible API |

## Default Credentials

### MinIO
- **Access Key**: minioadmin
- **Secret Key**: minioadmin
- **Console**: http://localhost:9001 (minioadmin / minioadmin)

### PostgreSQL
- **Host**: localhost:5432
- **User**: postgres
- **Password**: postgres
- **Database**: face_recognition

### Redis
- **Host**: localhost:6379
- **Password**: (none by default)

## Environment Variables

```bash
# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/face_recognition

# Redis
REDIS_URL=redis://localhost:6379/0

# MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=face-recognition

# Security
SECRET_KEY=your-secret-key-here
ENCRYPTION_KEY=your-32-byte-encryption-key!!

# JWT
JWT_SECRET_KEY=your-jwt-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=30

# App
ENVIRONMENT=development
DEBUG=true
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_auth.py

# Run integration tests only
pytest tests/integration/

# Run e2e tests
pytest tests/e2e/ -m e2e
```

## Docker Services

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Rebuild images
docker-compose build --no-cache
```

## Project Structure

```
face-recognition-service/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── db/
│   │   ├── database.py      # Database connection
│   │   ├── models.py        # SQLAlchemy models
│   │   └── crud.py         # Database operations
│   ├── routes/
│   │   ├── auth.py         # Authentication endpoints
│   │   ├── verify.py       # Face verification endpoints
│   │   ├── reference.py   # Reference management
│   │   ├── upload.py      # File upload endpoints
│   │   └── health.py      # Health check endpoints
│   ├── services/
│   │   ├── verify_service.py
│   │   ├── cache_service.py
│   │   ├── storage_service.py
│   │   └── ml_service.py
│   └── utils/
│       ├── security.py
│       ├── validators.py
│       └── helpers.py
├── tests/
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── e2e/               # End-to-end tests
├── alembic/               # Database migrations
├── scripts/               # Utility scripts
└── docs/                 # Documentation
```

## CI/CD Pipeline

The project includes GitHub Actions workflows for:

- **Lint & Format Check** - Black, isort, flake8, mypy
- **Security Scan** - Bandit, pip-audit, safety
- **Tests** - Unit and integration tests with coverage
- **Build** - Docker image build and push to GHCR
- **Deploy** - Staging and production deployments

## Troubleshooting

### Port already in use
```bash
# Find process using port 8000
lsof -i :8000
# Kill the process
kill <PID>
```

### Database connection failed
```bash
# Check if PostgreSQL is running
docker-compose ps postgres
# Restart PostgreSQL
docker-compose restart postgres
```

### MinIO connection issues
```bash
# Check MinIO status
docker-compose ps minio
# View MinIO logs
docker-compose logs minio
# Recreate MinIO bucket
mc mb local/face-recognition
```

## Next Steps

1. Read the [API Documentation](http://localhost:8000/docs)
2. Check out [examples](./docs/examples.md)
3. Learn about [deployment](./docs/deployment.md)
4. Contribute to the project!

## Support

- GitHub Issues: https://github.com/Alexsey111/face-recognition-backend/issues
- Documentation: ./docs/