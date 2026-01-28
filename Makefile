# Makefile для Face Recognition Service
# Удобные команды для разработки и развертывания

.PHONY: help setup install install-dev test lint format clean run dev docker-up docker-down docker-logs

# Переменные
PYTHON := python3.11
PIP := pip3.11
VENV := venv
PROJECT_NAME := face-recognition-service

# Цвета для вывода
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Показать справку по командам
	@echo "$(BLUE)Face Recognition Service - Доступные команды:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

setup: ## Настроить виртуальную среду
	@echo "$(BLUE)[INFO]$(NC) Настройка виртуальной среды..."
	@if [ -f "setup_venv.sh" ]; then \
		chmod +x setup_venv.sh; \
		./setup_venv.sh; \
	elif [ -f "setup_venv.bat" ]; then \
		setup_venv.bat; \
	else \
		echo "$(RED)[ERROR]$(NC) Скрипт настройки не найден"; \
		exit 1; \
	fi

install: ## Установить зависимости production
	@echo "$(BLUE)[INFO]$(NC) Установка зависимостей..."
	@$(PYTHON) -m pip install --upgrade pip setuptools wheel
	@$(PYTHON) -m pip install --no-cache-dir -r requirements.txt
	@echo "$(GREEN)[SUCCESS]$(NC) Зависимости установлены"

install-dev: ## Установить все зависимости (production + dev)
	@echo "$(BLUE)[INFO]$(NC) Установка всех зависимостей..."
	@$(PYTHON) -m pip install --upgrade pip setuptools wheel
	@$(PYTHON) -m pip install --no-cache-dir -r requirements.txt
	@$(PYTHON) -m pip install --no-cache-dir -r requirements-dev.txt
	@echo "$(GREEN)[SUCCESS]$(NC) Все зависимости установлены"

install-poetry: ## Установить Poetry и зависимости
	@echo "$(BLUE)[INFO]$(NC) Установка Poetry..."
	@curl -sSL https://install.python-poetry.org | $(PYTHON) -
	@echo "$(GREEN)[SUCCESS]$(NC) Poetry установлен"
	@echo "$(BLUE)[INFO]$(NC) Установка зависимостей через Poetry..."
	@poetry install
	@poetry install --with dev
	@echo "$(GREEN)[SUCCESS]$(NC) Зависимости установлены через Poetry"

dev: ## Активировать виртуальную среду и запустить в development режиме
	@echo "$(BLUE)[INFO]$(NC) Активация виртуальной среды и запуск в dev режиме..."
	@source $(VENV)/bin/activate && \
		export ENVIRONMENT=development && \
		export DEBUG=true && \
		uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run: ## Запустить приложение в production режиме
	@echo "$(BLUE)[INFO]$(NC) Запуск приложения..."
	@source $(VENV)/bin/activate && \
		uvicorn app.main:app --host 0.0.0.0 --port 8000

shell: ## Открыть Python shell с активированной средой
	@echo "$(BLUE)[INFO]$(NC) Открытие Python shell..."
	@source $(VENV)/bin/activate && $(PYTHON)

test: ## Запустить все тесты
	@echo "$(BLUE)[INFO]$(NC) Запуск тестов..."
	@source $(VENV)/bin/activate && pytest tests/ -v --cov=app --cov-report=term-missing

test-unit: ## Запустить только unit тесты
	@echo "$(BLUE)[INFO]$(NC) Запуск unit тестов..."
	@source $(VENV)/bin/activate && pytest tests/unit/ -v

test-integration: ## Запустить только integration тесты
	@echo "$(BLUE)[INFO]$(NC) Запуск integration тестов..."
	@source $(VENV)/bin/activate && pytest tests/integration/ -v

test-watch: ## Запустить тесты в watch режиме
	@echo "$(BLUE)[INFO]$(NC) Запуск тестов в watch режиме..."
	@source $(VENV)/bin/activate && pytest-watch tests/ -v

coverage: ## Создать отчет о покрытии тестами
	@echo "$(BLUE)[INFO]$(NC) Создание отчета о покрытии..."
	@source $(VENV)/bin/activate && pytest tests/ --cov=app --cov-report=html --cov-report=xml
	@echo "$(GREEN)[SUCCESS]$(NC) Отчет создан в htmlcov/index.html"

lint: ## Проверить код линтерами
	@echo "$(BLUE)[INFO]$(NC) Проверка кода линтерами..."
	@source $(VENV)/bin/activate && \
		flake8 app/ tests/ && \
		mypy app/ && \
		bandit -r app/

format: ## Форматировать код
	@echo "$(BLUE)[INFO]$(NC) Форматирование кода..."
	@source $(VENV)/bin/activate && \
		black app/ tests/ && \
		isort app/ tests/

format-check: ## Проверить форматирование кода
	@echo "$(BLUE)[INFO]$(NC) Проверка форматирования..."
	@source $(VENV)/bin/activate && \
		black --check app/ tests/ && \
		isort --check-only app/ tests/

security: ## Проверить безопасность зависимостей
	@echo "$(BLUE)[INFO]$(NC) Проверка безопасности..."
	@source $(VENV)/bin/activate && \
		$(PYTHON) -m safety check && \
		$(PYTHON) -m pip audit

clean: ## Очистить временные файлы
	@echo "$(BLUE)[INFO]$(NC) Очистка временных файлов..."
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf htmlcov/
	@rm -rf .coverage
	@rm -rf .pytest_cache/
	@rm -rf .mypy_cache/
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)[SUCCESS]$(NC) Временные файлы очищены"

clean-venv: ## Удалить виртуальную среду
	@echo "$(YELLOW)[WARNING]$(NC) Удаление виртуальной среды..."
	@rm -rf $(VENV)
	@echo "$(GREEN)[SUCCESS]$(NC) Виртуальная среда удалена"

# Docker команды
docker-up: ## Запустить через Docker Compose
	@echo "$(BLUE)[INFO]$(NC) Запуск через Docker Compose..."
	docker-compose up -d
	@echo "$(GREEN)[SUCCESS]$(NC) Сервисы запущены"

docker-up-dev: ## Запустить в development режиме через Docker
	@echo "$(BLUE)[INFO]$(NC) Запуск development режима через Docker..."
	docker-compose -f docker-compose.dev.yml up -d
	@echo "$(GREEN)[SUCCESS]$(NC) Development сервисы запущены"

docker-down: ## Остановить Docker сервисы
	@echo "$(BLUE)[INFO]$(NC) Остановка Docker сервисов..."
	docker-compose down
	@echo "$(GREEN)[SUCCESS]$(NC) Сервисы остановлены"

docker-logs: ## Показать логи Docker сервисов
	@echo "$(BLUE)[INFO]$(NC) Логи Docker сервисов..."
	docker-compose logs -f

docker-rebuild: ## Пересобрать Docker образы
	@echo "$(BLUE)[INFO]$(NC) Пересборка Docker образов..."
	docker-compose down
	docker-compose build --no-cache
	docker-compose up -d
	@echo "$(GREEN)[SUCCESS]$(NC) Docker образы пересобраны"

# База данных
db-migrate: ## Применить миграции БД
	@echo "$(BLUE)[INFO]$(NC) Применение миграций..."
	@source $(VENV)/bin/activate && alembic upgrade head

db-migrate-create: ## Создать новую миграцию
	@echo "$(BLUE)[INFO]$(NC) Создание новой миграции..."
	@source $(VENV)/bin/activate && alembic revision --autogenerate -m "$(MSG)"

# ==================== Database Indexes ====================

db-check-indexes: ## Проверить существующие индексы
	@echo "$(BLUE)🔍 Checking existing indexes...$(NC)"
	@source $(VENV)/bin/activate && python scripts/check_existing_indexes.py

db-validate-performance: ## Валидация производительности запросов
	@echo "$(BLUE)⏱️  Validating query performance...$(NC)"
	@source $(VENV)/bin/activate && python scripts/validate_index_performance.py

db-monitor-indexes: ## Мониторинг использования индексов
	@echo "$(BLUE)📊 Monitoring index usage...$(NC)"
	@source $(VENV)/bin/activate && python scripts/monitor_index_usage.py

db-index-sizes: ## Проверить размер индексов
	@echo "$(BLUE)📏 Checking index sizes...$(NC)"
	@source $(VENV)/bin/activate && python scripts/check_index_size.py

db-migrate-indexes: ## Применить миграцию с индексами
	@echo "$(BLUE)🚀 Applying index migration...$(NC)"
	@source $(VENV)/bin/activate && alembic upgrade head
	@echo "$(GREEN)✅ Migration complete!$(NC)"
	@echo ""
	@make db-check-indexes

db-downgrade: ## Откатить последнюю миграцию
	@echo "$(BLUE)[INFO]$(NC) Откат последней миграции..."
	@source $(VENV)/bin/activate && alembic downgrade -1

db-reset: ## Сбросить БД и применить все миграции
	@echo "$(YELLOW)[WARNING]$(NC) Сброс БД..."
	@source $(VENV)/bin/activate && \
		alembic downgrade base && \
		alembic upgrade head
	@echo "$(GREEN)[SUCCESS]$(NC) БД сброшена и миграции применены"

# Модель
download-model: ## Загрузить модель MiniFASNetV2
	@echo "$(BLUE)[INFO]$(NC) Загрузка модели MiniFASNetV2..."
	@source $(VENV)/bin/activate && python scripts/download_model.py

download-model-force: ## Перезагрузить модель MiniFASNetV2
	@echo "$(BLUE)[INFO]$(NC) Принудительная загрузка модели MiniFASNetV2..."
	@source $(VENV)/bin/activate && python scripts/download_model.py --force

# Мониторинг
health: ## Проверить health status сервиса
	@echo "$(BLUE)[INFO]$(NC) Проверка health status..."
	@curl -s http://localhost:8000/health | jq . || curl -s http://localhost:8000/health

health-check: ## Запустить полный health check скрипт
	@echo "$(BLUE)[INFO]$(NC) Запуск полного health check..."
	@bash scripts/health_check.sh http://localhost 8000

docker-health: ## Проверить health всех Docker контейнеров
	@echo "$(BLUE)[INFO]$(NC) Проверка Docker контейнеров..."
	@bash scripts/docker_health_check.sh all

docker-health-api: ## Проверить health API контейнера
	@echo "$(BLUE)[INFO]$(NC) Проверка API контейнера..."
	@bash scripts/docker_health_check.sh api

logs: ## Показать логи приложения
	@echo "$(BLUE)[INFO]$(NC) Логи приложения..."
	tail -f logs/app.log 2>/dev/null || echo "Логи не найдены"

status: ## Показать статус всех сервисов
	@echo "$(BLUE)[INFO]$(NC) Статус сервисов..."
	@echo "=== Python версия ==="
	@$(PYTHON) --version
	@echo "=== Активная среда ==="
	@echo $$VIRTUAL_ENV
	@echo "=== Зависимости ==="
	@source $(VENV)/bin/activate && $(PYTHON) -m pip list | head -20

# Утилиты
deps-check: ## Проверить устаревшие зависимости
	@echo "$(BLUE)[INFO]$(NC) Проверка устаревших зависимостей..."
	@source $(VENV)/bin/activate && $(PYTHON) -m pip list --outdated

deps-update: ## Обновить зависимости
	@echo "$(BLUE)[INFO]$(NC) Обновление зависимостей..."
	@source $(VENV)/bin/activate && $(PYTHON) -m pip install --upgrade -r requirements.txt

backup: ## Создать backup проекта
	@echo "$(BLUE)[INFO]$(NC) Создание backup..."
	@tar -czf backup-$(shell date +%Y%m%d-%H%M%S).tar.gz \
		--exclude=venv \
		--exclude=.git \
		--exclude=__pycache__ \
		--exclude=.pytest_cache \
		--exclude=htmlcov \
		--exclude=*.log \
		.
	@echo "$(GREEN)[SUCCESS]$(NC) Backup создан"

# Развертывание
deploy: ## Развернуть в production
	@echo "$(BLUE)[INFO]$(NC) Развертывание в production..."
	@echo "$(YELLOW)[WARNING]$(NC) Убедитесь, что настроены переменные окружения!"
	@source $(VENV)/bin/activate && \
		alembic upgrade head && \
		uvicorn app.main:app --host 0.0.0.0 --port 8000

# Справка по переменным окружения
env-help: ## Показать справку по переменным окружения
	@echo "$(BLUE)Переменные окружения:$(NC)"
	@echo ""
	@echo "$(GREEN)Обязательные:$(NC)"
	@echo "  DATABASE_URL - строка подключения к PostgreSQL"
	@echo "  REDIS_URL - строка подключения к Redis"
	@echo "  JWT_SECRET_KEY - секретный ключ для JWT"
	@echo "  ENCRYPTION_KEY - ключ шифрования (256 бит)"
	@echo ""
	@echo "$(GREEN)Опциональные:$(NC)"
	@echo "  DEBUG=false - режим отладки"
	@echo "  ENVIRONMENT=production - среда выполнения"
	@echo "  S3_ENDPOINT_URL - URL MinIO/S3"
	@echo "  LOG_LEVEL=INFO - уровень логирования"
	@echo ""
	@echo "Пример .env файла:"
	@echo "  cp .env.example .env"

# Все в одном
all: clean install-dev format lint test ## Полная очистка, установка, форматирование, проверка и тесты
	@echo "$(GREEN)[SUCCESS]$(NC) Все операции завершены успешно!"

# ==================== Metrics Validation ====================

validate-metrics: ## Запуск валидации метрик FAR/FRR/Liveness
	@echo "$(BLUE)🔍 Запуск валидации метрик FAR/FRR/Liveness...$(NC)"
	@source $(VENV)/bin/activate && python scripts/validate_metrics.py \
		--dataset tests/datasets/validation_dataset \
		--threshold $(shell python -c "from app.config import settings; print(settings.VERIFICATION_THRESHOLD)" 2>/dev/null || echo "0.6") \
		--output validation_results_$(shell date +%Y%m%d_%H%M%S).json

validate-metrics-threshold: ## Валидация с указанием порога
	@echo "$(BLUE)🔍 Запуск валидации метрик с порогом $(THRESHOLD)...$(NC)"
	@source $(VENV)/bin/activate && python scripts/validate_metrics.py \
		--dataset tests/datasets/validation_dataset \
		--threshold $(THRESHOLD) \
		--output validation_results_$(THRESHOLD)_$(shell date +%Y%m%d_%H%M%S).json

validate-quick: ## Быстрая валидация (с меньшим датасетом)
	@echo "$(BLUE)⚡ Быстрая валидация...$(NC)"
	@source $(VENV)/bin/activate && python scripts/validate_metrics.py \
		--dataset tests/datasets/quick_validation \
		--threshold 0.6 \
		--output validation_quick.json

validate-threshold-analysis: ## Анализ оптимального порога
	@echo "$(BLUE)📊 Анализ оптимального порога...$(NC)"
	@source $(VENV)/bin/activate && python scripts/validate_metrics.py \
		--dataset tests/datasets/validation_dataset \
		--analyze-thresholds \
		--output threshold_analysis.json

validate-rocauc: ## Построение ROC кривой и расчет AUC
	@echo "$(BLUE)📈 Построение ROC кривой...$(NC)"
	@source $(VENV)/bin/activate && python scripts/validate_metrics.py \
		--dataset tests/datasets/validation_dataset \
		--roc-curve \
		--output roc_curve.json

prepare-dataset: ## Создание структуры тестового датасета
	@echo "$(BLUE)📦 Создание структуры тестового датасета...$(NC)"
	@mkdir -p tests/datasets/validation_dataset/{genuine_pairs,impostor_pairs,live_faces,spoofed_faces}
	@mkdir -p tests/datasets/quick_validation/{genuine_pairs,impostor_pairs,live_faces,spoofed_faces}
	@echo "$(GREEN)✅ Структура создана.$(NC)"
	@echo "$(YELLOW)📝 Следуйте инструкциям в tests/datasets/README.md для заполнения данными$(NC)"

validate-compliance: ## Проверка compliance с 152-ФЗ
	@echo "$(BLUE)📋 Проверка compliance с 152-ФЗ...$(NC)"
	@source $(VENV)/bin/activate && python scripts/validate_metrics.py \
		--dataset tests/datasets/validation_dataset \
		--compliance-check \
		--output compliance_report.json

validate-report: ## Создание детального отчета
	@echo "$(BLUE)📊 Создание детального отчета...$(NC)"
	@source $(VENV)/bin/activate && python scripts/validate_metrics.py \
		--dataset tests/datasets/validation_dataset \
		--detailed-report \
		--output detailed_report.json

# ========================================
# Performance Testing
# ========================================

.PHONY: test-performance load-test-smoke load-test-standard load-test-stress load-test-soak test-all-performance

test-performance: ## Запустить performance тесты
	@echo "$(BLUE)🚀$(NC) Running performance tests..."
	@source $(VENV)/bin/activate && pytest tests/performance/test_performance_requirements.py -v -m performance

load-test-smoke: ## Smoke load test (5 users, 60s)
	@echo "$(BLUE)💨$(NC) Running smoke load test..."
	@locust \
		-f tests/performance/locustfile.py \
		--host=http://localhost:8000 \
		--users=5 \
		--spawn-rate=1 \
		--run-time=60s \
		--headless \
		--only-summary

load-test-standard: ## Standard load test
	@echo "$(BLUE)📊$(NC) Running standard load test..."
	@bash tests/performance/run_load_tests.sh

load-test-stress: ## Stress test (200 users, 300s)
	@echo "$(BLUE)💥$(NC) Running stress test..."
	@mkdir -p load_test_results
	@locust \
		-f tests/performance/locustfile.py \
		--host=http://localhost:8000 \
		--users=200 \
		--spawn-rate=10 \
		--run-time=300s \
		--headless \
		--only-summary \
		--html=load_test_results/stress_test.html

load-test-soak: ## Soak test (1 hour)
	@echo "$(BLUE)⏰$(NC) Running soak test (1 hour)..."
	@RUN_SOAK_TEST=true bash tests/performance/run_load_tests.sh

test-all-performance: test-performance load-test-standard ## Все performance тесты
	@echo "$(GREEN)✅$(NC) All performance tests completed"

# ========================================
# CI/CD
# ========================================

.PHONY: ci-lint ci-security ci-test ci-build ci-validate ci-all setup-pre-commit

ci-lint: ## Проверить код линтерами (CI)
	@echo "🔍 Running linters..."
	@source $(VENV)/bin/activate && \
		black --check app/ tests/ && \
		isort --check-only app/ tests/ && \
		flake8 app/ tests/ --max-line-length=120 && \
		mypy app/ --ignore-missing-imports

ci-security: ## Проверить безопасность (CI)
	@echo "🔒 Running security scans..."
	@source $(VENV)/bin/activate && \
		bandit -r app/ -f json -o bandit-report.json && \
		safety check --json > safety-report.json || true && \
		trivy fs --exit-code 0 .

ci-test: ## Запустить все тесты (CI)
	@echo "🧪 Running all tests..."
	@source $(VENV)/bin/activate && pytest tests/ -v --cov=app --cov-report=xml --cov-report=html

ci-build: ## Собрать Docker образ (CI)
	@echo "🏗️  Building Docker image..."
	@docker build -t face-recognition-api:$(shell git rev-parse --short HEAD) .

ci-validate: ## Валидировать метрики (CI)
	@echo "✅ Validating metrics..."
	@source $(VENV)/bin/activate && python scripts/validate_metrics.py \
		--dataset tests/datasets/validation_dataset \
		--output validation_results.json

ci-all: ci-lint ci-security ci-test ci-build ci-validate ## Все CI проверки
	@echo "✅ All CI checks passed!"

setup-pre-commit: ## Настроить pre-commit hooks
	@echo "📋 Setting up pre-commit hooks..."
	@source $(VENV)/bin/activate && \
		pip install pre-commit && \
		pre-commit install
	@echo "✅ Pre-commit hooks installed"

# По умолчанию показать справку
.DEFAULT_GOAL := help