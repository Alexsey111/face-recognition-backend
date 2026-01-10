"""
Тесты для health check endpoints.
Проверка работоспособности и мониторинга сервиса.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import status
import time
import psutil
import os

from app.main import create_test_app
from app import __version__


class TestHealthEndpoints:
    """Тесты основных health check endpoints"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        self.app = create_test_app()
        self.client = TestClient(self.app)
    
    def test_health_basic_success(self):
        """Тест базового health check - успешный ответ"""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Проверяем основные поля
        assert "success" in data
        assert "status" in data
        assert "version" in data
        assert "uptime" in data
        assert "services" in data
        assert "system_info" in data
        
        # Проверяем значения для Phase 2 (внешние сервисы недоступны)
        assert data["success"] is False  # Не все критичные сервисы доступны
        assert data["status"] == "degraded"  # Статус degraded когда внешние сервисы недоступны
        assert data["version"] == __version__
        assert isinstance(data["uptime"], (int, float))
        assert data["uptime"] >= 0
        
        # Проверяем структуру services
        services = data["services"]
        assert "api" in services
        assert "database" in services
        assert "redis" in services
        assert "storage" in services
        assert "ml_service" in services
        
        # Для Phase 2 внешние сервисы недоступны и возвращают ошибки подключения
        assert "database" in services
        assert "redis" in services
        assert "storage" in services
        assert "ml_service" in services
        assert services["api"] == "healthy"
        
        # Внешние сервисы должны быть недоступны (не "healthy")
        assert services["database"] != "healthy"
        assert services["redis"] != "healthy"
        assert services["storage"] != "healthy"
        assert services["ml_service"] != "healthy"
        
        # Проверяем system_info
        system_info = data["system_info"]
        assert "memory_percent" in system_info
        assert "cpu_count" in system_info
        assert "python_version" in system_info
        assert isinstance(system_info["memory_percent"], (int, float))
        assert 0 <= system_info["memory_percent"] <= 100
        assert system_info["cpu_count"] > 0
        assert isinstance(system_info["python_version"], str)
    
    def test_health_multiple_calls(self):
        """Тест множественных вызовов health check"""
        # Первый вызов
        response1 = self.client.get("/health")
        data1 = response1.json()
        
        # Ждем немного
        time.sleep(0.1)
        
        # Второй вызов
        response2 = self.client.get("/health")
        data2 = response2.json()
        
        # Uptime должен увеличиться
        assert data2["uptime"] > data1["uptime"]
        
        # Остальные поля должны быть те же
        assert data1["status"] == data2["status"]
        assert data1["version"] == data2["version"]
        assert data1["success"] == data2["success"]
    
    def test_health_no_auth_required(self):
        """Тест что health check не требует авторизации"""
        # Без заголовков авторизации
        response = self.client.get("/health")
        assert response.status_code == 200
        
        # С невалидными заголовками
        response = self.client.get("/health", headers={"Authorization": "Bearer invalid"})
        assert response.status_code == 200
        
        # С валидными заголовками
        response = self.client.get("/health", headers={"Authorization": "Bearer valid_token"})
        assert response.status_code == 200
    
    def test_health_with_query_params(self):
        """Тест health check с query параметрами (должны игнорироваться)"""
        response = self.client.get("/health?some=value&another=param")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestStatusEndpoints:
    """Тесты детальных status endpoints"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        self.app = create_test_app()
        self.client = TestClient(self.app)
    
    def test_status_detailed_success(self):
        """Тест детального status check"""
        response = self.client.get("/status")
        
        assert response.status_code == 200
        data = response.json()
        
        # Проверяем основные поля
        assert "success" in data
        assert "database_status" in data
        assert "redis_status" in data
        assert "storage_status" in data
        assert "ml_service_status" in data
        assert "last_heartbeat" in data
        
        # Для Phase 2 внешние сервисы недоступны, общий статус не успешный
        assert data["success"] is False  # Не все критичные сервисы доступны
        
        # Проверяем что статусы не "healthy" (сервисы недоступны)
        assert "healthy" not in data["database_status"]
        assert "healthy" not in data["redis_status"]
        assert "healthy" not in data["storage_status"]
        assert "healthy" not in data["ml_service_status"]
        
        # Статусы должны содержать информацию об ошибках
        assert "unhealthy" in data["database_status"] or "error" in data["database_status"].lower()
        assert "unhealthy" in data["redis_status"] or "error" in data["redis_status"].lower()
        assert "unhealthy" in data["storage_status"] or "error" in data["storage_status"].lower()
        assert "unhealthy" in data["ml_service_status"] or "error" in data["ml_service_status"].lower()
        
        # Проверяем формат timestamp
        assert isinstance(data["last_heartbeat"], str)
    
    def test_status_no_auth_required(self):
        """Тест что status check не требует авторизации"""
        response = self.client.get("/status")
        assert response.status_code == 200


class TestReadinessEndpoints:
    """Тесты readiness probe"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        self.app = create_test_app()
        self.client = TestClient(self.app)
    
    def test_readiness_success(self):
        """Тест readiness probe - успешный ответ"""
        response = self.client.get("/ready")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "success" in data
        assert "message" in data
        assert data["success"] is True
        assert data["message"] == "Service is ready"
    
    def test_readiness_no_auth_required(self):
        """Тест что readiness не требует авторизации"""
        response = self.client.get("/ready")
        assert response.status_code == 200


class TestLivenessEndpoints:
    """Тесты liveness probe"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        self.app = create_test_app()
        self.client = TestClient(self.app)
    
    def test_liveness_success(self):
        """Тест liveness probe - успешный ответ"""
        response = self.client.get("/live")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "success" in data
        assert "message" in data
        assert data["success"] is True
        assert data["message"] == "Service is alive"
    
    def test_liveness_no_auth_required(self):
        """Тест что liveness не требует авторизации"""
        response = self.client.get("/live")
        assert response.status_code == 200
    
    def test_liveness_simple_response(self):
        """Тест что liveness возвращает корректный ответ с BaseResponse полями"""
        response = self.client.get("/live")
        data = response.json()
        
        # BaseResponse содержит success, message, timestamp, request_id
        assert len(data) == 4
        assert set(data.keys()) == {"success", "message", "timestamp", "request_id"}
        
        # Проверяем обязательные поля
        assert data["success"] is True
        assert data["message"] == "Service is alive"
        assert "timestamp" in data
        assert "request_id" in data


class TestMetricsEndpoints:
    """Тесты metrics endpoint"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        self.app = create_test_app()
        self.client = TestClient(self.app)
    
    def test_metrics_success(self):
        """Тест получения метрик - успешный ответ"""
        response = self.client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        # Проверяем основные секции
        assert "timestamp" in data
        assert "system" in data
        assert "process" in data
        assert "application" in data
        
        # Проверяем system метрики
        system = data["system"]
        assert "cpu_percent" in system
        assert "memory_percent" in system
        assert "memory_available" in system
        assert "memory_total" in system
        assert "disk_percent" in system
        assert "disk_free" in system
        assert "disk_total" in system
        
        # Проверяем диапазоны значений
        assert 0 <= system["cpu_percent"] <= 100
        assert 0 <= system["memory_percent"] <= 100
        assert 0 <= system["disk_percent"] <= 100
        assert system["memory_available"] >= 0
        assert system["memory_total"] > 0
        assert system["disk_free"] >= 0
        assert system["disk_total"] > 0
        
        # Проверяем process метрики
        process = data["process"]
        assert "cpu_percent" in process
        assert "memory_percent" in process
        assert "memory_info" in process
        assert "num_threads" in process
        assert "create_time" in process
        assert "status" in process
        
        assert 0 <= process["cpu_percent"] <= 100
        assert 0 <= process["memory_percent"] <= 100
        assert process["num_threads"] > 0
        assert isinstance(process["create_time"], (int, float))
        assert process["create_time"] > 0
        
        # Проверяем application метрики
        application = data["application"]
        assert "uptime" in application
        assert "worker_id" in application
        assert "version" in application
        
        assert application["uptime"] >= 0
        assert isinstance(application["version"], str)
    
    def test_metrics_asynchronous_cpu_measurement(self):
        """Тест что CPU измеряется асинхронно (не блокирует)"""
        import asyncio
        import concurrent.futures
        
        # Запускаем несколько запросов одновременно
        responses = []
        
        def make_request():
            return self.client.get("/metrics")
        
        # Запускаем несколько запросов параллельно
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request) for _ in range(3)]
            responses = [future.result() for future in futures]
        
        # Все должны быть успешными
        assert len(responses) == 3
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "system" in data
            assert "cpu_percent" in data["system"]
    
    def test_metrics_no_auth_required(self):
        """Тест что metrics не требует авторизации"""
        response = self.client.get("/metrics")
        assert response.status_code == 200


class TestHealthCheckScenarios:
    """Тесты различных сценариев health check"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        self.app = create_test_app()
        self.client = TestClient(self.app)
    
    def test_health_endpoints_order(self):
        """Тест что все health endpoints отвечают корректно"""
        endpoints = [
            ("/health", 200),
            ("/status", 200),
            ("/ready", 200),
            ("/live", 200),
            ("/metrics", 200)
        ]
        
        for endpoint, expected_status in endpoints:
            response = self.client.get(endpoint)
            assert response.status_code == expected_status, \
                f"Endpoint {endpoint} должен вернуть {expected_status}, но вернул {response.status_code}"
    
    def test_health_consistency(self):
        """Тест консистентности данных между health endpoints"""
        # Получаем данные с разных endpoints
        health_response = self.client.get("/health").json()
        status_response = self.client.get("/status").json()
        
        # Проверяем согласованность версии
        assert health_response["version"] == __version__
        
        # Проверяем что статус API healthy
        assert health_response["services"]["api"] == "healthy"
    
    def test_health_performance(self):
        """Тест производительности health endpoints"""
        import time
        
        endpoints = ["/health", "/status", "/ready", "/live", "/metrics"]
        times = {}
        
        for endpoint in endpoints:
            start_time = time.time()
            response = self.client.get(endpoint)
            end_time = time.time()
            
            assert response.status_code == 200
            times[endpoint] = end_time - start_time
        
        # Health endpoints должны отвечать быстро (< 1 секунды)
        for endpoint, response_time in times.items():
            assert response_time < 1.0, \
                f"Endpoint {endpoint} отвечает слишком медленно: {response_time:.3f}s"
    
    def test_health_headers(self):
        """Тест заголовков ответов health endpoints"""
        response = self.client.get("/health")
        
        # Проверяем стандартные заголовки
        assert "content-type" in response.headers
        assert response.headers["content-type"] == "application/json"
        
        # Health не должен требовать особых заголовков авторизации
        assert "www-authenticate" not in response.headers
    
    def test_health_json_response(self):
        """Тест что все health endpoints возвращают корректный JSON"""
        endpoints = ["/health", "/status", "/ready", "/live", "/metrics"]
        
        for endpoint in endpoints:
            response = self.client.get(endpoint)
            assert response.status_code == 200
            
            # Проверяем что ответ - валидный JSON
            try:
                data = response.json()
                assert isinstance(data, dict), f"Endpoint {endpoint} должен возвращать объект"
            except Exception as e:
                pytest.fail(f"Endpoint {endpoint} возвращает некорректный JSON: {e}")


class TestHealthCheckErrorHandling:
    """Тесты обработки ошибок в health endpoints"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        self.app = create_test_app()
        self.client = TestClient(self.app)
    
    def test_health_invalid_method(self):
        """Тест health endpoint с неправильным HTTP методом"""
        # POST на health должен вернуть 405 Method Not Allowed
        response = self.client.post("/health")
        assert response.status_code == 405
    
    def test_health_invalid_path(self):
        """Тест несуществующего health path"""
        response = self.client.get("/health/nonexistent")
        assert response.status_code == 404
    
    @patch('psutil.Process')
    def test_health_psutil_error_handling(self, mock_process):
        """Тест обработки ошибок psutil в health check"""
        # Симулируем ошибку psutil
        mock_process.side_effect = Exception("psutil error")
        
        response = self.client.get("/health")
        # Должен вернуть ошибку 503 при проблемах с системной информацией
        assert response.status_code == 503
    
    def test_health_corrupted_response(self):
        """Тест корректной обработки поврежденных данных"""
        # Этот тест проверяет что health endpoint стабилен
        # даже при проблемах с системными данными
        
        # Делаем несколько запросов подряд
        for _ in range(5):
            response = self.client.get("/health")
            assert response.status_code in [200, 503]  # 503 допустим при системных проблемах


if __name__ == "__main__":
    # Запуск тестов напрямую
    pytest.main([__file__, "-v"])