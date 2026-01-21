# Development Guide

Complete guide for developers working on the Face Recognition API.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Style](#code-style)
- [Database Migrations](#database-migrations)
- [Debugging](#debugging)

## Environment Setup

### Prerequisites

```bash
# Required
- Python 3.11+
- Docker & Docker Compose
- Git
- PostgreSQL client (optional, for direct DB access)

# Infrastructure Services (via Docker Compose)
- PostgreSQL 15+ (database)
- Redis 7+ (caching)
- MinIO (object storage)

# Recommended
- VSCode with Python extension
- Postman or Insomnia for API testing
- DBeaver or pgAdmin for database management
```

### Initial Setup

```bash
# Clone repository
git clone <repository-url>
cd face-recognition-service

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Copy environment file
cp .env.example .env

# Configure environment variables in .env
# Edit .env with your database and service configurations

# Start infrastructure services (PostgreSQL, Redis, MinIO)
docker-compose up -d postgres redis minio

# Run database migrations
alembic upgrade head

# Create initial admin user (optional)
python scripts/create_admin.py

# Verify installation
pytest tests/ --co -q  # List all tests
```

### Development Tools

```bash
# Start development server with auto-reload and debug logging
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --log-level debug

# Run linting
ruff check .

# Format code
ruff format .

# Type checking
pyright app/
```

## Project Structure

```
face-recognition-service/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration management
│   ├── dependencies.py          # Dependency injection
│   ├── locustfile.py           # Load testing configuration
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── database.py         # Database connection (SQLAlchemy)
│   │   ├── models.py           # SQLAlchemy ORM models
│   │   └── crud.py             # Database operations
│   │
│   ├── models/                 # Pydantic schemas (data validation)
│   │   ├── __init__.py
│   │   ├── user.py             # User model
│   │   ├── reference.py         # Reference face model
│   │   ├── verification.py     # Verification result model
│   │   ├── webhook.py           # Webhook configuration
│   │   ├── face.py             # Face data model
│   │   ├── request.py           # Request models
│   │   └── response.py          # Response models
│   │
│   ├── routes/                 # API endpoints (FastAPI routers)
│   │   ├── __init__.py
│   │   ├── admin.py            # Admin endpoints
│   │   ├── auth.py             # Authentication (register, login)
│   │   ├── health.py           # Health checks
│   │   ├── liveness.py        # Liveness detection
│   │   ├── metrics.py          # Prometheus metrics
│   │   ├── reference.py       # Reference face management
│   │   ├── upload.py           # File upload endpoints
│   │   ├── verify.py           # Face verification
│   │   └── webhook.py          # Webhook endpoints
│   │
│   ├── services/               # Business logic layer
│   │   ├── __init__.py
│   │   ├── auth_service.py     # Authentication logic
│   │   ├── audit_service.py    # Audit logging
│   │   ├── cache_service.py    # Redis caching
│   │   ├── database_service.py # DB operations
│   │   ├── encryption_service.py # Data encryption
│   │   ├── liveness_service.py # Liveness detection
│   │   ├── ml_service.py       # ML model inference
│   │   ├── reference_service.py # Reference management
│   │   ├── session_service.py  # Session management
│   │   ├── storage_service.py  # MinIO/object storage
│   │   ├── validation_service.py # Input validation
│   │   └── verify_service.py   # Verification logic
│   │
│   ├── middleware/             # Custom middleware
│   │   ├── __init__.py
│   │   ├── auth.py             # JWT authentication
│   │   ├── cors.py             # CORS handling
│   │   ├── error_handler.py    # Global error handling
│   │   ├── logging.py          # Request logging
│   │   ├── metrics.py          # Metrics collection
│   │   ├── rate_limit.py       # Rate limiting
│   │   └── request_logging.py  # Detailed request logging
│   │
│   ├── utils/                  # Helper utilities
│   │   ├── __init__.py
│   │   ├── constants.py        # Application constants
│   │   ├── decorators.py       # Custom decorators
│   │   ├── exceptions.py       # Custom exceptions
│   │   ├── face_aligner.py    # Face alignment utilities
│   │   ├── face_alignment_utils.py
│   │   ├── file_utils.py       # File operations
│   │   ├── helpers.py          # General helpers
│   │   ├── lighting_analyzer.py # Lighting analysis
│   │   ├── logger.py           # Logger setup
│   │   ├── security.py         # Security utilities
│   │   ├── structured_logging.py # Structured logging
│   │   └── validators.py       # Custom validators
│   │
│   └── tasks/                  # Background tasks (Celery/Scheduler)
│       ├── __init__.py
│       ├── cleanup.py          # Cleanup tasks
│       └── scheduler.py        # Task scheduling
│
├── alembic/                    # Database migrations
│   ├── versions/               # Migration files
│   └── env.py                 # Alembic configuration
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py            # Pytest fixtures and configuration
│   ├── test_*.py              # Unit tests for modules
│   ├── unit/                   # Detailed unit tests
│   ├── integration/           # Integration tests
│   ├── e2e/                   # End-to-end tests
│   ├── performance/           # Load and stress tests
│   ├── security/              # Security vulnerability tests
│   └── manual/                # Manual test scenarios
│
├── scripts/                   # Utility scripts
│   ├── check_existing_indexes.py
│   ├── check_index_size.py
│   ├── delete_test_user.py
│   ├── fix_alembic_heads.py
│   ├── init_minio.py
│   ├── migrate_add_threshold_column.py
│   ├── monitor_index_usage.py
│   ├── test_metrics_endpoints.py
│   └── validate_index_performance.py
│
├── docs/                      # Additional documentation
│   ├── PERFORMANCE.md
│   └── webhook_integration.md
│
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI/CD pipeline
│
├── docker-compose*.yml        # Docker Compose files
│   ├── docker-compose.yml     # Main configuration
│   ├── docker-compose.dev.yml # Development environment
│   ├── docker-compose.prod.yml # Production
│   ├── docker-compose.test.yml # Testing
│   └── docker-compose.monitoring.yml # Monitoring stack
│
├── Dockerfile*                # Container images
├── Dockerfile.dev             # Development container
├── pyproject.toml            # Poetry configuration
├── requirements*.txt          # Dependencies
├── Makefile                  # Build automation
├── .env.example              # Environment template
├── .gitignore
├── README.md
└── alembic.ini
```

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `app/routes/` | API endpoints (auth, health, reference, upload, verify, webhook, admin, liveness, metrics) |
| `app/services/` | Business logic (ML, anti-spoofing, liveness, storage, auth, encryption) |
| `app/middleware/` | Authentication, CORS, error handling, rate limiting, metrics, logging |
| `app/utils/` | Face alignment, lighting analysis, validators, decorators, security |
| `app/tasks/` | Background tasks for cleanup and scheduling |
| `tests/` | Multi-level testing (unit, integration, e2e, performance, security, manual) |
| `scripts/` | Database maintenance, migration, and utility scripts |

## Development Workflow

### Branch Strategy

```
main           # Production-ready code
└── develop    # Development branch
    ├── feature/add-new-feature
    ├── fix/bug-description
    └── refactor/component-name
```

| Branch | Purpose | Protection Rules |
|--------|---------|-----------------|
| `main` | Production-ready code | Protected: PR required, reviews, tests |
| `develop` | Integration branch for features | Protected: PR required, tests pass |
| `feature/*` | New features | Merge to develop via PR |
| `fix/*` | Bug fixes | Merge to develop via PR |
| `refactor/*` | Code refactoring | Merge to develop via PR |
| `hotfix/*` | Urgent production fixes | Merge directly to main & develop |

### Creating a New Feature

```bash
# 1. Create feature branch from develop
git checkout develop
git pull origin develop
git checkout -b feature/my-new-feature

# 2. Implement feature with tests
# - Write tests first (TDD approach recommended)
# - Implement feature
# - Ensure all tests pass

# 3. Run quality checks
black app tests
isort app tests
flake8 app tests
pytest tests/ --cov=app

# 4. Commit changes
git add .
git commit -m "feat: add new feature description"

# 5. Push and create PR
git push origin feature/my-new-feature
# Create Pull Request on GitHub
```

### 1. Git Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: add your feature description"

# Push changes
git push origin feature/your-feature-name

# Create pull request for review
```

### 2. Branch Naming Convention

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feature/` | New features | `feature/add-face-embedding` |
| `fix/` | Bug fixes | `fix/login-timeout-issue` |
| `hotfix/` | Urgent production fixes | `hotfix/security-patch-v1.2.1` |
| `docs/` | Documentation updates | `docs/update-api-endpoints` |
| `refactor/` | Code refactoring | `refactor/auth-middleware` |
| `test/` | Test additions | `test/add-verification-tests` |

### 3. Commit Message Convention

Follow the Conventional Commits specification for all commit messages.

#### Message Structure

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

#### Type Definitions

| Type | Description | Example |
|------|-------------|---------|
| `feat` | New feature | `feat(auth): add JWT token refresh` |
| `fix` | Bug fix | `fix(upload): resolve file size limit` |
| `docs` | Documentation changes | `docs(API): update endpoint descriptions` |
| `test` | Test additions or modifications | `test(verify): add liveness detection tests` |
| `refactor` | Code restructuring | `refactor(ml): restructure face detection` |
| `perf` | Performance improvements | `perf(db): optimize verification queries` |
| `chore` | Maintenance tasks | `chore(deps): update dependencies` |
| `style` | Code style changes | `style(utils): format with black` |
| `build` | Build system changes | `build(docker): optimize image layers` |
| `ci` | CI/CD pipeline changes | `ci(github): add performance tests` |

#### Examples

```bash
# Feature
feat(reference): add batch reference upload endpoint

# Bug Fix
fix(auth): resolve token expiration issue

# Documentation
docs(readme): add quickstart guide

# Performance
perf(ml): optimize face detection inference time

# Refactoring
refactor(services): separate business logic from API

# Tests
test(webhook): add integration tests for callback

# Breaking Change
feat(auth): change token validation
BREAKING CHANGE: token format updated to JWT RS256
```

#### Scope Guidelines

| Scope | Usage |
|-------|-------|
| `auth` | Authentication, authorization, tokens |
| `upload` | File uploads, processing |
| `verify` | Face verification logic |
| `reference` | Reference face management |
| `ml` | Machine learning models |
| `db` | Database operations |
| `api` | API endpoints, routes |
| `utils` | Utility functions |
| `middleware` | Custom middleware |
| `deps` | Dependency updates |

#### Best Practices

1. **Use imperative mood**: "add feature" not "added feature"
2. **Keep subject line under 50 characters**
3. **Separate subject from body with blank line**
4. **Capitalize first letter of subject**
5. **No period at end of subject line**
6. **Body wraps at 72 characters**
7. **Reference issues when applicable**

```bash
# Good commit messages
feat(auth): implement OAuth2 login
fix(db): resolve connection pool leak
docs(README): add deployment instructions

# Avoid
Fixed the bug
update code
changes made
```

#### Automatic Changelog Generation

Commits are used to generate changelog automatically:

```bash
# Generate changelog from commits
git-changelog --output CHANGELOG.md
```

#### Git Hooks (Optional)

Enable commit message validation:

```bash
# Install commit message hook
cat > .git/hooks/commit-msg << 'EOF'
#!/bin/bash
python scripts/validate_commit_msg.py "$1"
EOF
chmod +x .git/hooks/commit-msg
```

```python
# scripts/validate_commit_msg.py
import re
import sys

pattern = r'^(feat|fix|docs|test|refactor|perf|chore|style|build|ci)(\([a-z]+\))?: .+'

if not re.match(pattern, sys.argv[1]):
    print("Invalid commit message format")
    print("Expected: <type>(<scope>): <description>")
    sys.exit(1)
```

### 4. Pull Request Process

1. **Before Creating PR**
   ```bash
   # Ensure tests pass
   pytest tests/ -v
   
   # Run linting
   ruff check .
   
   # Format code
   ruff format .
   
   # Type checking
   pyright app/
   ```

2. **PR Requirements**
   - [ ] All tests pass
   - [ ] No linting errors
   - [ ] Type hints complete
   - [ ] Documentation updated (if needed)
   - [ ] Tests added/updated for new functionality

3. **PR Description Template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update
   
   ## Testing
   - [ ] Unit tests added
   - [ ] Integration tests added
   - [ ] Manual testing performed
   
   ## Checklist
   - [ ] My code follows the style guidelines
   - [ ] I have performed a self-review
   - [ ] I have commented complex code
   - [ ] I have updated documentation
   - [ ] My changes generate no new warnings
   ```

4. **Code Review Process**
   - Assign reviewers
   - Address feedback
   - Squash commits if needed
   - Merge after approval

### 5. Development Cycle

```mermaid
graph LR
    A[Create Branch] --> B[Write Tests]
    B --> C[Implement Feature]
    C --> D[Run Quality Checks]
    D --> E[Local Testing]
    E --> F[Create PR]
    F --> G[Code Review]
    G --> H[Merge to Develop]
    H --> I[CI/CD Pipeline]
    I --> J[Deploy to Staging]
```

### 6. Release Process

```bash
# Create release branch from develop
git checkout develop
git pull origin develop
git checkout -b release/v1.0.0

# Update version numbers, changelog
# Run final tests
# Merge to main
git checkout main
git merge release/v1.0.0 --no-ff
git tag -a v1.0.0 -m "Release v1.0.0"

# Merge back to develop
git checkout develop
git merge release/v1.0.0 --no-ff

# Delete release branch
git branch -d release/v1.0.0
```

### 7. Hotfix Process

```bash
# Create hotfix branch from main
git checkout main
git pull origin main
git checkout -b hotfix/critical-fix

# Make fix and commit
git add .
git commit -m "fix: critical security patch"

# Merge to main
git checkout main
git merge hotfix/critical-fix --no-ff
git tag -a v1.0.1 -m "Hotfix v1.0.1"

# Merge to develop
git checkout develop
git merge hotfix/critical-fix --no-ff

# Delete hotfix branch
git branch -d hotfix/critical-fix
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_auth.py

# Run specific test class
pytest tests/test_auth.py::TestAuth

# Run specific test method
pytest tests/test_auth.py::TestAuth::test_login_success

# Run with markers
pytest -m unit          # Only unit tests
pytest -m integration   # Only integration tests
pytest -m "not slow"    # Exclude slow tests

# Run with verbose output
pytest -v -s

# Run with debugging
pytest --pdb           # Drop into debugger on failure

# Run and generate JUnit XML report
pytest --junitxml=report.xml

# Run and generate HTML report
pytest --html=report.html
```

### Test Structure

```bash
# Run all tests
pytest

# Run specific test types
pytest tests/unit/           # Unit tests
pytest tests/integration/    # Integration tests
pytest tests/e2e/           # End-to-end tests
pytest tests/performance/   # Load tests
pytest tests/security/      # Security tests
pytest tests/manual/        # Manual test scenarios
```

### Test Categories

| Type | Location | Purpose |
|------|----------|---------|
| Unit | `tests/unit/` | Test individual functions/classes |
| Integration | `tests/integration/` | Test component interactions |
| End-to-End | `tests/e2e/` | Test complete user flows |
| Performance | `tests/performance/` | Load and stress testing |
| Security | `tests/security/` | Vulnerability testing |

### Writing Tests

#### Test Structure Example

```python
import pytest
from fastapi import status

class TestFeature:
    """Test suite for feature"""

    def test_success_case(self, client, auth_headers):
        """Test successful operation"""
        response = client.post(
            "/api/v1/endpoint",
            headers=auth_headers,
            json={"key": "value"}
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["key"] == "value"

    def test_error_case(self, client, auth_headers):
        """Test error handling"""
        response = client.post(
            "/api/v1/endpoint",
            headers=auth_headers,
            json={"invalid": "data"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.parametrize("input_data,expected", [
        ({"a": 1}, {"result": 1}),
        ({"a": 2}, {"result": 2}),
    ])
    def test_multiple_cases(self, client, auth_headers, input_data, expected):
        """Test multiple scenarios with parametrize"""
        response = client.post(
            "/api/v1/endpoint",
            headers=auth_headers,
            json=input_data
        )
        assert response.json() == expected
```

#### Using Fixtures

```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    """Create test client"""
    with TestClient(app) as client:
        yield client

@pytest.fixture
def auth_headers(client):
    """Get authentication headers"""
    response = client.post("/auth/login", json={
        "email": "test@example.com",
        "password": "testpassword"
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def sample_image():
    """Load sample test image"""
    with open("tests/fixtures/face.jpg", "rb") as f:
        yield {"image": f}
```

#### Async Tests

```python
import pytest
from httpx import AsyncClient, ASGITransport

@pytest.fixture
async def async_client():
    """Create async test client"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

class TestAsyncFeature:
    async def test_async_operation(self, async_client, auth_headers):
        """Test async endpoint"""
        response = await async_client.post(
            "/api/v1/endpoint",
            headers=auth_headers,
            json={"key": "value"}
        )
        assert response.status_code == 200
```

#### Database Tests

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.database import Base, get_db

@pytest.fixture(scope="session")
def engine():
    return create_engine("sqlite:///:memory:")

@pytest.fixture(scope="session")
def tables(engine):
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)

@pytest.fixture
def db_session(engine, tables):
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

@pytest.fixture
def db_client(db_session):
    def override_get_db():
        yield db_session
    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.clear()
```

#### Mocking External Services

```python
from unittest.mock import patch
import pytest

class TestWithMocks:
    def test_with_storage_mock(self, client, auth_headers):
        """Test with mocked storage service"""
        with patch("app.services.storage_service.upload") as mock_upload:
            mock_upload.return_value = {"url": "https://example.com/image.jpg"}
            
            response = client.post(
                "/api/v1/upload",
                headers=auth_headers,
                files={"file": ("test.jpg", b"image data", "image/jpeg")}
            )
            
            assert response.status_code == 200
            mock_upload.assert_called_once()

    def test_with_ml_mock(self, client, auth_headers):
        """Test with mocked ML service"""
        with patch("app.services.ml_service.detect_faces") as mock_detect:
            mock_detect.return_value = [{"bbox": [0, 0, 100, 100]}]
            
            response = client.post(
                "/api/v1/verify",
                headers=auth_headers,
                json={"image_id": "test-image"}
            )
            
            assert response.status_code == 200
```

#### Test Naming Conventions

| Pattern | Example | Purpose |
|---------|---------|---------|
| `test_<method>_<scenario>` | `test_login_with_valid_credentials` | Clear test description |
| `test_<class>_<method>` | `test_UserModel_create` | Group related tests |
| Use descriptive names | `test_face_detection_returns_bounding_box` | Self-documenting |

#### Best Practices

```python
# ✅ Good
class TestAuthentication:
    def test_login_success_returns_token(self, client):
        response = client.post("/auth/login", json={
            "email": "user@example.com",
            "password": "validpassword"
        })
        assert response.status_code == 200
        assert "access_token" in response.json()

    def test_login_invalid_password_returns_401(self, client):
        response = client.post("/auth/login", json={
            "email": "user@example.com",
            "password": "wrongpassword"
        })
        assert response.status_code == 401

# ❌ Avoid
class TestAuth:
    def test_1(self):
        # Unclear test name
        pass
    
    def test_login(self):
        # Missing specific scenario
        pass
```

#### Test Coverage Goals

```bash
# Minimum coverage requirements
pytest --cov=app --cov-fail-under=80

# Per-module coverage
pytest --cov=app.services --cov-report=term-missing

# Generate detailed HTML report
pytest --cov=app --cov-report=html:htmlcov
```

### Test Configuration (pytest.ini)

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
filterwarnings =
    ignore::DeprecationWarning
    ignore::pytest.PytestUnraisableExceptionWarning
```

### Running Load Tests

```bash
# Install locust
pip install locust

# Run load test
locust -f tests/load/locustfile.py --host=http://localhost:8000

# Access web UI at http://localhost:8089
```

#### Quick Start

```bash
# Quick smoke test (10 users, 1 spawn rate)
locust -f tests/load/locustfile.py --users 10 --spawn-rate 1

# Load test (100 users, 10 spawn rate)
locust -f tests/load/locustfile.py --users 100 --spawn-rate 10

# Stress test (500 users, 50 spawn rate)
locust -f tests/load/locustfile.py --users 500 --spawn-rate 50

# Headless mode (no UI, CSV output)
locust -f tests/load/locustfile.py --users 100 --spawn-rate 10 --run-time 10m --csv=results

# Distributed (master)
locust -f tests/load/locustfile.py --master

# Distributed (worker)
locust -f tests/load/locustfile.py --worker --master-host=192.168.1.100
```

#### Load Test Configuration Example

```python
# tests/load/locustfile.py
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner

class FaceRecognitionUser(HttpUser):
    """Simulate real user behavior"""
    wait_time = between(1, 3)
    
    @task(3)
    def verify_face(self):
        """Verify face - most common operation"""
        self.client.post("/api/v1/verify", json={
            "reference_id": "test-ref-1",
            "image": "base64_encoded_image"
        })
    
    @task(2)
    def liveness_check(self):
        """Liveness detection"""
        self.client.post("/api/v1/liveness/check", json={
            "image": "base64_encoded_image"
        })
    
    @task(1)
    def get_reference(self):
        """Get reference faces"""
        self.client.get("/api/v1/reference", params={"limit": 10})
    
    def on_start(self):
        """Login on start"""
        response = self.client.post("/auth/login", json={
            "email": "loadtest@example.com",
            "password": "testpassword"
        })
        if response.status_code == 200:
            self.client.headers.update({
                "Authorization": f"Bearer {response.json()['access_token']}"
            })

@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Initialize distributed testing"""
    if isinstance(environment.runner, MasterRunner):
        print("Locust master initialized")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print summary on test stop"""
    print(f"Total requests: {environment.stats.total.num_requests}")
    print(f"Failed requests: {environment.stats.total.num_failures}")
```

#### Load Test Scenarios

```bash
# Quick smoke test
locust -f tests/load/locustfile.py --users 10 --spawn-rate 1 --host=http://localhost:8000

# Load test
locust -f tests/load/locustfile.py --users 100 --spawn-rate 10 --host=http://localhost:8000

# Stress test
locust -f tests/load/locustfile.py --users 500 --spawn-rate 50 --host=http://localhost:8000

# Run without UI (headless)
locust -f tests/load/locustfile.py --users 100 --spawn-rate 10 --host=http://localhost:8000 --run-time 10m --csv=results

# Distributed load test (master)
locust -f tests/load/locustfile.py --master --host=http://localhost:8000

# Distributed load test (worker)
locust -f tests/load/locustfile.py --worker --master-host=192.168.1.100
```

#### Load Test Targets

| Test Type | Users | Spawn Rate | Duration | Purpose |
|-----------|-------|------------|----------|---------|
| Smoke | 10 | 1 | 1 min | Verify system works |
| Load | 100 | 10 | 10 min | Normal operating conditions |
| Stress | 500 | 50 | 15 min | System breaking point |
| Spike | 1000 | 100 | 5 min | Sudden traffic surge |
| Soak | 50 | 5 | 1 hour | Long-term stability |

#### Performance Benchmarks

```python
# tests/performance/benchmarks.py
import pytest

class TestPerformanceBenchmarks:
    @pytest.mark.performance
    def test_verification_response_time(self, client, auth_headers):
        """Verify response time < 500ms"""
        import time
        start = time.time()
        response = client.post("/api/v1/verify", json={
            "reference_id": "test-ref",
            "image": "base64_image"
        })
        elapsed = time.time() - start
        assert response.status_code == 200
        assert elapsed < 0.5  # 500ms requirement

    @pytest.mark.performance
    def test_liveness_response_time(self, client, auth_headers):
        """Liveness check < 1s"""
        import time
        start = time.time()
        response = client.post("/api/v1/liveness/check", json={
            "image": "base64_image"
        })
        elapsed = time.time() - start
        assert response.status_code == 200
        assert elapsed < 1.0  # 1s requirement
```

## Code Style

### Python Style Guide

We follow PEP 8 with these modifications:

| Rule | Value |
|------|-------|
| Line length | 100 characters |
| Formatter | Black |
| Import sorting | isort |
| Type hints | Required for public functions |
| Docstrings | Google-style |

#### Example Function

```python
from typing import Optional, List
from datetime import datetime

def process_data(
    user_id: int,
    data: List[dict],
    timeout: Optional[int] = None
) -> dict:
    """
    Process user data with optional timeout.

    Args:
        user_id: The user identifier
        data: List of data dictionaries to process
        timeout: Optional timeout in seconds

    Returns:
        Dictionary containing processed results

    Raises:
        ValueError: If user_id is invalid
        TimeoutError: If processing exceeds timeout
    """
    if user_id <= 0:
        raise ValueError("Invalid user_id")

    # Implementation here
    return {"status": "success"}
```

### Code Quality Tools

```bash
# Format code
black app tests

# Sort imports
isort app tests

# Lint code
flake8 app tests --max-line-length=100

# Type checking
mypy app --ignore-missing-imports

# Security check
bandit -r app

# Dependency security
safety check

# Run all checks
black app tests && isort app tests && flake8 app tests && mypy app
```

### Pre-commit Hooks

Install and configure pre-commit hooks:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install
```

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.0.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: ["--max-line-length=100", "--extend-ignore=E203,W503"]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--max-diff-size=10M']

  - repo: https://github.com/pycqa/bandit
    rev: 1.7.6
    hooks:
      - id: bandit
        args: [-r, app/]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        args: [--ignore-missing-imports, app/]
```

### Code Organization

```
app/
├── routes/          # One file per endpoint group
├── services/        # Business logic separated by domain
├── models/          # Pydantic models for validation
└── utils/           # Shared utilities
```

### Best Practices

```python
# ✅ Good: Type hints and docstrings
def process_image(
    image: UploadFile,
    threshold: float = 0.6
) -> dict[str, Any]:
    """
    Process uploaded image for face recognition.

    Args:
        image: Uploaded image file
        threshold: Similarity threshold (0.0-1.0)

    Returns:
        Processing results with confidence score
    """
    # Implementation
    pass

# ❌ Avoid: Missing type hints
def process_image(image, threshold=0.6):
    pass
```

### Naming Conventions

| Component | Convention | Example |
|-----------|------------|---------|
| Variables | snake_case | `user_id`, `file_path` |
| Functions | snake_case | `get_user_data()` |
| Classes | PascalCase | `FaceVerificationService` |
| Constants | UPPER_SCASE | `MAX_FILE_SIZE` |
| Private methods | `_private_method()` | `_validate_input()` |
| Async functions | `async_` prefix | `async_get_data()` |

## Database Migrations

### Creating Migrations

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "add user verification table"

# Create empty migration
alembic revision -m "custom migration"

# Review generated migration in alembic/versions/
```

### Migration Examples

```python
# alembic/versions/xxxx_add_feature.py
"""add user verification table

Revision ID: abc123
Revises: xyz789
Create Date: 2026-01-21 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

def upgrade():
    """Apply migration"""
    op.create_table(
        'user_verifications',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_user_verifications_user_id', 'user_verifications', ['user_id'])

def downgrade():
    """Rollback migration"""
    op.drop_index('ix_user_verifications_user_id', table_name='user_verifications')
    op.drop_table('user_verifications')
```

### Running Migrations

```bash
# Upgrade to latest
alembic upgrade head

# Upgrade one version
alembic upgrade +1

# Downgrade one version
alembic downgrade -1

# Show current version
alembic current

# Show migration history
alembic history --verbose
```

### Migration Best Practices

1. Always write `upgrade()` and `downgrade()` functions
2. Test downgrade path before merging
3. Include meaningful migration messages
4. Keep migrations small and focused
5. Don't modify existing migrations after they're merged

### Troubleshooting Migrations

```bash
# Check for migration conflicts
alembic heads

# Fix migration heads
python scripts/fix_alembic_heads.py

# Show pending migrations
alembic pending_migrations
```

## Debugging

### Local Debugging

```bash
# Enable debug mode
uvicorn app.main:app --reload --log-level debug

# View detailed logs
LOG_LEVEL=DEBUG python -m uvicorn app.main:app
```

### Using VSCode

```bash
# Set breakpoint in code (click left margin)
# Press F5 or use Run → Start Debugging
# Trigger the endpoint
# Debugger will pause at breakpoint
```

### Using pdb

```python
# Add to code where you want to debug
import pdb; pdb.set_trace()

# Or use breakpoint() in Python 3.7+
breakpoint()
```

### Debugging Tests

```bash
# Run tests with debugger
pytest --pdb

# Debug specific test
pytest tests/test_auth.py::test_login_success --pdb
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)

# Use appropriate log levels
logger.debug("Detailed debug information")
logger.info("General informational message")
logger.warning("Warning message")
logger.error("Error occurred", exc_info=True)  # Include traceback
logger.critical("Critical error")
```

### Database Debugging

```bash
# Connect to local database
psql -h localhost -U your_user -d your_db

# View logs
docker-compose logs postgres

# Watch logs in real-time
docker-compose logs -f postgres
```

### Common Issues

#### Database Connection

```bash
# Check database status
docker-compose -f docker-compose.dev.yml ps postgres

# Test connection
psql -h localhost -U postgres -d face_recognition

# View logs
docker-compose -f docker-compose.dev.yml logs postgres
```

#### Dependency Issues

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Clear pip cache
pip cache purge

# Recreate virtual environment
rm -rf .venv
python -m venv .venv
pip install -r requirements.txt -r requirements-dev.txt
```

#### Test Failures

```bash
# Run test with verbose output
pytest -v --tb=short tests/

# Run single test file
pytest tests/test_auth.py -v --tb=long

# Check for missing fixtures
pytest tests/ --collect-only
```

### Common Error Solutions

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError` | Check PYTHONPATH, reinstall deps |
| `ConnectionRefused` | Start Docker services |
| `Migration conflict` | Run `alembic heads` check |
| `Test timeout` | Increase pytest timeout or check async issues |
| `TypeError` | Verify Pydantic model compatibility |

### Performance Profiling

#### Profiling API Endpoints

```python
import cProfile
import pstats
from io import StringIO

def profile_endpoint():
    profiler = cProfile.Profile()
    profiler.enable()

    # Your code here (e.g., make an API call)
    import requests
    response = requests.get("http://localhost:8000/api/v1/health")

    profiler.disable()
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    print(s.getvalue())

if __name__ == "__main__":
    profile_endpoint()
```

#### Using py-spy for Live Profiling

```bash
# Install py-spy
pip install py-spy

# Profile running process (need PID)
py-spy top --pid <process_id>

# Generate flame graph
py-spy record -o profile.svg --pid <process_id>

# Profile specific function
py-spy record -o profile.svg -- python -m myapp
```

#### Database Query Profiling

```python
from sqlalchemy import event
from sqlalchemy.engine import Engine
import logging
import time

logging.basicConfig()
logger = logging.getLogger("sqlalchemy.engine")
logger.setLevel(logging.INFO)

@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    conn.info.setdefault('query_start_time', []).append(time.time())

@event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total = time.time() - conn.info['query_start_time'].pop(-1)
    logger.info(f"Query took {total:.4f}s: {statement}")
```

#### Slow Query Detection

```python
# Log slow queries (> 100ms)
SLOW_QUERY_THRESHOLD = 0.1  # seconds

@event.listens_for(Engine, "after_cursor_execute")
def log_slow_queries(conn, cursor, statement, parameters, context, executemany):
    total = time.time() - conn.info['query_start_time'].pop(-1)
    if total > SLOW_QUERY_THRESHOLD:
        logger.warning(f"SLOW QUERY ({total:.4f}s): {statement}")
```

#### Memory Profiling

```python
from memory_profiler import profile

@profile  # Decorator to profile function
def memory_intensive_function():
    data = [i * 1000 for i in range(10000)]
    return sum(data)

# Run memory profiler on script
python -m memory_profiler myscript.py
```

#### Async Profiling

```python
import asyncio
import cProfile

async def profiled_async_operation():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your async code here
    await asyncio.sleep(1)
    result = await some_async_function()
    
    profiler.disable()
    return profiler

# Profile and print results
profiler = asyncio.run(profiled_async_operation())
import pstats
s = pstats.Stats(profiler).sort_stats('cumulative')
s.print_stats(10)
```

#### Benchmarking with timeit

```python
import timeit

# Simple benchmark
def my_function():
    # Code to benchmark
    return sum(range(1000))

# Run 1000 times
result = timeit.timeit(my_function, number=1000)
print(f"Average time: {result/1000*1000:.3f}ms")

# Compare implementations
setup = "data = list(range(10000))"
stmt1 = "sorted(data)"
stmt2 = "data.sort()"

t1 = timeit.timeit(stmt1, setup=setup, number=1000)
t2 = timeit.timeit(stmt2, setup=setup, number=1000)
print(f"sorted(): {t1:.4f}s, list.sort(): {t2:.4f}s")
```

#### Profiling Startup Time

```bash
# Measure import time
python -X importtime -c "from app.main import app" 2>&1 | grep -E "^(.*):.*import time"

# Full startup profiling
python -m cProfile -o startup.prof app/main.py
python -c "import pstats; p = pstats.Stats('startup.prof'); p.sort_stats('cumtime').print_stats(20)"
```

#### Performance Regression Testing

```python
# tests/performance/test_benchmarks.py
import pytest
import time

class TestPerformanceBenchmarks:
    @pytest.mark.performance
    def test_verification_response_time(self, client, auth_headers):
        """Verify response time < 500ms"""
        start = time.time()
        response = client.post("/api/v1/verify", json={
            "reference_id": "test-ref",
            "image": "base64_image"
        })
        elapsed = time.time() - start
        assert response.status_code == 200
        assert elapsed < 0.5, f"Response time {elapsed:.3f}s exceeds 500ms"

    @pytest.mark.performance
    def test_liveness_response_time(self, client, auth_headers):
        """Liveness check < 1s"""
        start = time.time()
        response = client.post("/api/v1/liveness/check", json={
            "image": "base64_image"
        })
        elapsed = time.time() - start
        assert response.status_code == 200
        assert elapsed < 1.0, f"Response time {elapsed:.3f}s exceeds 1s"
```

#### Monitoring in Development

```bash
# View application logs
tail -f logs/app.log

# Monitor API requests
curl http://localhost:8000/health

# Check metrics endpoint
curl http://localhost:8000/metrics
```

## Common Issues & Solutions

### Issue: Import errors

```bash
# Error: ModuleNotFoundError: No module named 'app'

# Solution: Ensure PYTHONPATH is set
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Or in VSCode settings.json
"python.envFile": "${workspaceFolder}/.env"
```

### Issue: Database connection errors

```bash
# Error: could not connect to server: Connection refused

# Solution: Check if services are running
docker-compose ps

# Restart services
docker-compose restart postgres

# Check database logs
docker-compose logs postgres

# Verify connection string in .env
# DATABASE_URL=postgresql://user:pass@localhost:5432/dbname
```

### Issue: Port already in use

```bash
# Error: [Errno 98] Address already in use

# Solution: Find and kill process
lsof -i :8000
kill -9 <PID>

# Or use a different port
uvicorn app.main:app --port 8001

# Find all processes on port
netstat -tlnp | grep 8000
```

### Issue: Migration conflicts

```bash
# Error: Multiple heads or branch point

# Solution: Reset to head and reapply
alembic downgrade base
alembic upgrade head

# Or merge branches
alembic merge heads -m "merge branches"

# Check current heads
alembic heads
```

### Issue: Docker build fails

```bash
# Error: Build fails with dependency issues

# Solution: Clear Docker cache and rebuild
docker-compose build --no-cache

# Or remove dangling images
docker image prune -a

# Check for out of space
df -h
```

### Issue: Token expired errors

```bash
# Error: 401 Unauthorized: Token has expired

# Solution: Check token expiration
# JWT tokens expire based on settings.JWT_EXPIRATION_HOURS
# Re-login to get new token

# Or increase expiration in .env
JWT_EXPIRATION_HOURS=24
```

### Issue: Memory out of limits

```bash
# Error: MemoryError or OOM killer

# Solution: Increase memory limits
# In docker-compose.yml:
# services:
#   api:
#     mem_limit: 4g

# Or use swap
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Issue: Redis connection refused

```bash
# Error: Error 111 while connecting to Redis

# Solution: Check Redis status
docker-compose ps redis

# Restart Redis
docker-compose restart redis

# Test connection
redis-cli ping
```

### Issue: MinIO/S3 bucket access denied

```bash
# Error: Access Denied or 403 Forbidden

# Solution: Check credentials in .env
# MINIO_ACCESS_KEY
# MINIO_SECRET_KEY

# Create bucket if not exists
mc mb myminio/face-images

# Or initialize with script
python scripts/init_minio.py
```

### Issue: Slow tests

```bash
# Tests taking too long to run

# Solution: Run in parallel
pytest -n auto  # Requires pytest-xdist

# Run only changed tests
pytest --co -q  # List tests
pytest tests/unit/  # Skip integration tests

# Use faster database for tests
# In pytest.ini: SQLALCHEMY_DATABASE_URL=sqlite:///:memory:
```

### Issue: Type checking errors

```bash
# Error: mypy or pyright finds errors

# Solution: Check specific errors
mypy app/ --verbose

# Ignore missing imports
mypy app/ --ignore-missing-imports

# Update type stubs
pip install --upgrade types-requests types-Pillow
```

### Issue: CORS errors in browser

```bash
# Error: Access to fetch blocked by CORS policy

# Solution: Configure CORS in .env
# CORS_ORIGINS=["http://localhost:3000"]

# Or in app/config.py
ALLOWED_ORIGINS = ["http://localhost:3000"]
```

### Issue: File upload size limit

```bash
# Error: 413 Payload Too Large

# Solution: Increase limit in .env
# MAX_UPLOAD_SIZE=10485760  # 10MB

# Or in app/config.py
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
```

### Issue: Git merge conflicts

```bash
# Error: Merge conflict in files

# Solution: View conflicts
git status
git diff --name-only --diff-filter=U

# Accept incoming changes
git checkout --ours <file>   # Keep our version
git checkout --theirs <file> # Keep their version

# Or use merge tool
git mergetool
```

### Issue: Alembic migration missing table

```bash
# Error: Table 'xxx' has no column 'yyy'

# Solution: Check if migration was applied
alembic current

# If not, apply migration
alembic upgrade head

# Or stamp to specific revision
alembic stamp <revision_id>
```

### Issue: JWT token validation fails

```bash
# Error: Could not validate credentials

# Solution: Check secret key matches
# settings.SECRET_KEY must be same for all instances

# Verify token format
import jwt
token = "your_token"
decoded = jwt.decode(token, key, algorithms=["HS256"])
```

### Quick Reference Commands

```bash
# Development shortcuts
alias dc="docker-compose"
alias dcu="docker-compose up -d"
alias dcd="docker-compose down"
alias dcl="docker-compose logs -f"

# Run tests
alias pt="pytest tests/ -v --tb=short"

# Run with coverage
alias ptc="pytest tests/ --cov=app --cov-report=term-missing"

# Format code
alias fmt="black app tests && isort app tests"

# Check types
alias type="mypy app --ignore-missing-imports"
```

## IDE Configuration

### VSCode Settings

Create `.vscode/settings.json`:

```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.linting.flake8Args": ["--max-line-length=100"],
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length=100"],
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "[python]": {
    "editor.rulers": [100]
  }
}
```

### VSCode Launch Configurations

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: FastAPI",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": [
        "app.main:app",
        "--reload",
        "--log-level",
        "debug"
      ],
      "jinja": true,
      "justMyCode": false,
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    },
    {
      "name": "Python: Pytest",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": [
        "tests/",
        "-v",
        "--log-cli-level=DEBUG"
      ],
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
```

### Recommended VSCode Extensions

```json
// .vscode/extensions.json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "njpwerner.autodocstring",
    "esbenp.prettier-vscode",
    "streetsidesoftware.code-spell-checker"
  ]
}
```

## Additional Resources

### Official Documentation

| Topic | Link |
|-------|------|
| FastAPI | https://fastapi.tiangolo.com |
| SQLAlchemy | https://docs.sqlalchemy.org |
| pytest | https://docs.pytest.org |
| Python Type Hints | https://docs.python.org/3/library/typing.html |
| Alembic | https://alembic.sqlalchemy.org |
| Pydantic | https://docs.pydantic.dev |
| Redis | https://redis.io/docs |
| MinIO | https://min.io/docs |

### API Documentation

See [API Documentation](API.md) for complete endpoint reference including:

- **Authentication** — Register, login, token refresh
- **Reference Management** — Create, list, delete reference faces
- **Face Verification** — Single and batch verification
- **Liveness Detection** — Anti-spoofing checks
- **File Upload** — Upload sessions and file handling
- **Webhooks** — Event notifications

### Architecture & Design

- [Architecture Guide](ARCHITECTURE.md) — System design and component overview
- [Security Guidelines](SECURITY.md) — Security best practices
- [Deployment Guide](DEPLOYMENT.md) — Production deployment steps

### Getting Help

1. **Check existing issues** — Search GitHub for similar problems
2. **Ask in team Slack** — `#face-recognition-dev` channel
3. **Create new issue** — Include detailed description:
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages and logs
   - Environment details
4. **Review this documentation** — Search DEVELOPMENT.md first

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines including:

- Code style requirements
- Pull request process
- Testing expectations
- Documentation updates

---

## Quick Reference

### Essential Commands

```bash
# Start development
docker-compose up -d
uvicorn app.main:app --reload

# Run tests
pytest tests/ -v --tb=short

# Code quality
black app tests && isort app tests && flake8 app tests

# Database
alembic upgrade head
alembic revision --autogenerate -m "description"

# Production deploy
docker-compose -f docker-compose.prod.yml up -d
```

### File Locations

| What | Where |
|------|-------|
| Main app | `app/main.py` |
| Config | `app/config.py` |
| Routes | `app/routes/` |
| Services | `app/services/` |
| Tests | `tests/` |
| Migrations | `alembic/versions/` |
| Scripts | `scripts/` |

---

**Last updated:** January 2026
**Maintained by:** NLP-Core-Team