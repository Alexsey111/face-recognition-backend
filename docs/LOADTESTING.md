# Load Testing Guide

## Overview

This guide describes how to run load tests for the Face Recognition Service using Locust and Docker Compose.

## Quick Start

### 1. Start the infrastructure

```bash
# Start all services (API + Locust master + 4 workers)
docker-compose -f docker-compose.loadtest.yml up -d

# Check status
docker-compose -f docker-compose.loadtest.yml ps
```

### 2. Access Locust Web UI

- **URL**: http://localhost:8089
- **Host**: http://face-recognition-api:8000

### 3. Run a test via Web UI

1. Open http://localhost:8089
2. Enter number of users (e.g., 100)
3. Enter spawn rate (e.g., 10 users/sec)
4. Click "Start Swarming"

### 4. Run headless test

```bash
# Smoke test (5 users, 60s)
make load-test-smoke

# Standard test
make load-test-standard

# Stress test (200 users, 300s)
make load-test-stress

# Soak test (1 hour)
make load-test-soak
```

## Docker Compose Services

| Service | Port | Description |
|---------|------|-------------|
| face-recognition-api | 8000 | Face Recognition API with GPU support |
| locust-master | 8089 | Locust master node (Web UI) |
| locust-worker | - | Worker nodes (4 replicas) |
| prometheus | 9090 | Metrics collection |
| grafana | 3001 | Visualization dashboard |

## Running Specific Tests

### Smoke Test (Quick validation)

```bash
docker-compose -f docker-compose.loadtest.yml run --rm locust-master \
  -f /mnt/locust/locustfile.py \
  --users 5 \
  --spawn-rate 1 \
  --run-time 60s \
  --headless \
  --host=http://face-recognition-api:8000
```

### Standard Load Test

```bash
docker-compose -f docker-compose.loadtest.yml run --rm locust-master \
  -f /mnt/locust/locustfile.py \
  --users 50 \
  --spawn-rate 5 \
  --run-time 300s \
  --headless \
  --host=http://face-recognition-api:8000
```

### Stress Test (Breaking point)

```bash
docker-compose -f docker-compose.loadtest.yml run --rm locust-master \
  -f /mnt/locust/locustfile.py \
  --users 200 \
  --spawn-rate 20 \
  --run-time 600s \
  --headless \
  --host=http://face-recognition-api:8000 \
  --html=/mnt/locust/results/stress_test.html
```

### Soak Test (Long duration)

```bash
RUN_SOAK_TEST=true \
docker-compose -f docker-compose.loadtest.yml run --rm locust-master \
  -f /mnt/locust/locustfile.py \
  --users 100 \
  --spawn-rate 10 \
  --run-time 3600s \
  --headless \
  --host=http://face-recognition-api:8000
```

## Monitoring with Grafana

1. Open Grafana: http://localhost:3001
2. Login: `admin` / `loadtest123`
3. Navigate to "Load Test Dashboard"

### Available Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| Requests/sec | Throughput | < 10 req/s |
| Success Rate | % of successful requests | < 95% |
| p50 Latency | Median response time | > 100ms |
| p95 Latency | 95th percentile | > 500ms |
| p99 Latency | 99th percentile | > 1000ms |
| Error Rate | % of failed requests | > 1% |

## Test Scenarios

### Verification Endpoint Test

```python
class FaceRecognitionLocust(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def verify_face(self):
        """Test verification endpoint"""
        files = {'image': open('test_image.jpg', 'rb')}
        self.client.post('/api/v1/verify', files=files)
    
    @task(1)
    def liveness_check(self):
        """Test liveness detection"""
        files = {'image': open('test_image.jpg', 'rb')}
        self.client.post('/api/v1/liveness', files=files)
```

### Custom Test Configuration

Edit `tests/performance/locustfile.py` to customize:

```python
HOST = "http://face-recognition-api:8000"
WEIGHT = 1
TEST_DATA_DIR = Path(__file__).parent / "test_data"
```

## Results Analysis

### Locust Statistics

```
 Type     | Name                             | # reqs  | # fails |    Avg |    Min |    Max |  RPS
----------|----------------------------------|---------|---------|--------|--------|--------|--------
 POST     | /api/v1/verify                   |  15234  |  12(0.08%)|    52ms |   12ms |  245ms |  25.4
 POST     | /api/v1/liveness                 |   7621  |   5(0.07%)|    18ms |    5ms |   89ms |  12.7
----------|----------------------------------|---------|---------|--------|--------|--------|--------
```

### Key Metrics

- **RPS (Requests per Second)**: Measures throughput
- **Avg Response Time**: Average latency
- **95th/99th Percentile**: Tail latency (important for SLA)
- **Failure Rate**: Should be < 0.1% for production

## Cleanup

```bash
# Stop all services
docker-compose -f docker-compose.loadtest.yml down

# Remove volumes (including Prometheus data)
docker-compose -f docker-compose.loadtest.yml down -v
```

## Troubleshooting

### GPU not available

```bash
# Check GPU in containers
docker-compose -f docker-compose.loadtest.yml exec face-recognition-api nvidia-smi

# If GPU not detected, ensure nvidia-docker is installed
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Locust workers not connecting

```bash
# Check worker logs
docker-compose -f docker-compose.loadtest.yml logs locust-worker

# Ensure network connectivity
docker-compose -f docker-compose.loadtest.yml exec locust-master ping locust-worker
```

### Out of memory

```bash
# Increase memory limit in docker-compose.loadtest.yml
# Or reduce number of concurrent users
```

## Performance Targets

| Scenario | Target RPS | p95 Latency | Error Rate |
|----------|------------|-------------|------------|
| Smoke    | 5+         | < 100ms     | < 0.1%     |
| Standard | 20+        | < 200ms     | < 0.1%     |
| Stress   | 50+        | < 500ms     | < 1.0%     |
| Soak     | 15+        | < 300ms     | < 0.5%     |
