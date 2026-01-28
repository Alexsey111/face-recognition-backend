# Face Recognition Service - High Availability Configuration

## Files Created

### Docker Compose
- `docker-compose.ha.yml` - High availability configuration
- `docker-compose.prod.yml` - Production configuration (updated)

### Monitoring
- `monitoring/prometheus.yml` - Prometheus scrape configuration
- `monitoring/rules/face-recognition-rules.yml` - Alert rules
- `monitoring/grafana/provisioning/datasources/prometheus.yml` - Grafana datasource

## Usage

### High Availability Deployment
```bash
# Deploy with 3 API replicas
docker stack deploy -c docker-compose.ha.yml face-recognition

# Or with docker-compose
docker-compose -f docker-compose.ha.yml up -d
```

### Access Points
- **API**: http://localhost:8000
- **Traefik Dashboard**: http://localhost:8080
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Jaeger**: http://localhost:16686

## Architecture

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

## High Availability Features

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

## Resource Limits

| Service | CPU | Memory |
|---------|-----|--------|
| face-api | 0.5-2 cores | 512MB-2GB |
| postgres | 2 cores | 2GB |
| redis | 1 core | 1GB |
| traefik | 0.5 cores | 256MB |
| prometheus | 1 core | 1GB |
| grafana | 0.5 cores | 512MB |

## Alert Rules

- **API Down**: API unavailable for >2min
- **High Error Rate**: >5% 5xx errors
- **High Latency**: P95 >2 seconds
- **Liveness Failure**: >10% failures
- **Verification Failure**: >5% errors
- **DB Pool Exhausted**: >90% connections
- **Redis Memory**: >90% used

# Face Recognition Service - Monitoring Configuration

## Files Created

### Docker Compose
- `docker-compose.ha.yml` - High availability configuration
- `docker-compose.prod.yml` - Production configuration (updated)

### Monitoring
- `monitoring/prometheus.yml` - Prometheus scrape configuration
- `monitoring/prometheus/prometheus.yml` - Full Prometheus config
- `monitoring/rules/biometric_alerts.yml` - FAR/FRR alert rules
- `monitoring/rules/face-recognition-rules.yml` - General alert rules
- `monitoring/grafana/provisioning/datasources/prometheus.yml` - Grafana datasource
- `monitoring/grafana/dashboards/biometric_metrics.json` - Biometric dashboard
- `monitoring/alertmanager/alertmanager.yml` - Alert routing

## Usage

### Start Monitoring Stack
```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

### Access Points
| Service | URL | Credentials |
|---------|-----|-------------|
| API | http://localhost:8000 | - |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin/admin |
| AlertManager | http://localhost:9093 | - |

## Prometheus Metrics

### Biometric Metrics
| Metric | Description | Threshold |
|--------|-------------|-----------|
| `false_accept_rate` | False Accept Rate (%) | < 0.1% |
| `false_reject_rate` | False Reject Rate (%) | < 3.0% |
| `equal_error_rate` | Equal Error Rate (%) | < 3.0% |
| `false_accept_total{severity}` | Counter by severity | - |
| `false_reject_total{severity}` | Counter by severity | - |

### Performance Metrics
| Metric | Description | Threshold |
|--------|-------------|-----------|
| `verification_duration_seconds` | Latency histogram | < 1s P95 |
| `verification_requests_total` | Total verifications | - |
| `verification_confidence_score` | Confidence histogram | - |

### Liveness Metrics
| Metric | Description | Threshold |
|--------|-------------|-----------|
| `liveness_checks_total{result}` | Pass/fail counts | - |
| `spoofing_attacks_detected{type}` | Blocked attacks | - |

## Grafana Dashboards

### Biometric Metrics Dashboard
**File:** `monitoring/grafana/dashboards/biometric_metrics.json`

Panels:
1. FAR/FRR Compliance (stat)
2. Equal Error Rate (gauge)
3. Overall Compliance Status
4. FAR/FRR Trend (24h)
5. False Accepts by Severity (stacked bars)
6. False Rejects by Severity (stacked bars)
7. Liveness Detection Rate
8. Spoofing Attacks Pie Chart
9. P95 Response Time
10. Verification Throughput
11. Confidence Distribution (heatmap)
12. GPU Utilization

### Import Dashboard
```bash
# Via API
curl -X POST -H "Content-Type: application/json" \
  --data-binary @monitoring/grafana/dashboards/biometric_metrics.json \
  http://admin:biometric123@localhost:3000/api/dashboards/db
```

## Alert Rules

### Critical Alerts
| Alert | Expression | Action |
|-------|------------|--------|
| `HighFAR` | `false_accept_rate > 0.5` for 5m | PagerDuty + Email |
| `CriticalEER` | `equal_error_rate > 5.0` for 5m | PagerDuty + Email |
| `HighSeverityFalseAccepts` | `rate(false_accept_total{severity="high"}[5m]) > 0.1` | PagerDuty |
| `SpoofingAttackDetected` | `increase(spoofing_attacks_detected[5m]) > 10` | PagerDuty |

### Warning Alerts
| Alert | Expression | Action |
|-------|------------|--------|
| `HighFRR` | `false_reject_rate > 3.0` for 5m | Email + Slack |
| `HighEER` | `equal_error_rate > 3.0` for 10m | Email + Slack |
| `LowLivenessDetectionRate` | `< 95%` for 5m | Email + Slack |
| `VerificationLatencyHigh` | `P95 > 1s` for 5m | Email |
| `GPUHighUtilization` | `> 90%` for 5m | Email |

### Recording Rules
```promql
# Pre-aggregated metrics
biometric:far:5m
biometric:frr:5m
biometric:eer:5m
biometric:compliance:overall
biometric:latency:p95:5m
biometric:throughput:5m
```

## AlertManager Routes

### Receivers
1. **critical** - PagerDuty + Email (critical severity)
2. **security** - Slack + Email (security team)
3. **operations** - Email (ops team)
4. **telegram** - Telegram bot

### Route Configuration
```yaml
route:
  - match:
      severity: critical
    receiver: 'critical'
  - match:
      team: security
    receiver: 'security'
  - match:
      team: infrastructure
    receiver: 'operations'
```

## Compliance (152-ФЗ)

### Requirements
| Metric | Requirement | Dashboard Panel |
|--------|-------------|-----------------|
| FAR | < 0.1% | Panel 1 (FAR) |
| FRR | < 1-3% | Panel 1 (FRR) |
| EER | < 3% | Panel 2 (EER) |

### Compliance Status Query
```promql
# Check overall compliance
biometric:compliance:far and biometric:compliance:frr
```

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │           Traefik (LB)               │
                    └──────────────┬──────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
    ┌──────▼──────┐         ┌──────▼──────┐         ┌──────▼──────┐
    │ face-api-1  │         │ face-api-2  │         │ face-api-3  │
    └──────┬──────┘         └──────┬──────┘         └──────┬──────┘
           │                       │                       │
           └───────────────────────┼───────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
   ┌────▼────┐               ┌────▼────┐               ┌────▼────┐
   │Postgres │               │  Redis  │               │  MinIO  │
   └─────────┘               └─────────┘               └─────────┘

        ┌─────────────────────────────────────────────────────┐
        │              Monitoring Stack                        │
        │  Prometheus ──► AlertManager ──► Grafana            │
        │       │                                              │
        │       ▼                                              │
        │  face-api:8000/metrics/prometheus                    │
        └─────────────────────────────────────────────────────┘
```

## High Availability Features

| Feature | Configuration |
|---------|---------------|
| **Horizontal Scaling** | 3 API replicas (configurable) |
| **Load Balancing** | Traefik with DNS round-robin |
| **Health Checks** | 30s interval, 3 retries |
| **Auto-restart** | on-failure, max 3 attempts |
| **Database Replication** | Primary + Replica |
| **Redis Replication** | Master + Replica |

## Resource Limits

| Service | CPU | Memory |
|---------|-----|--------|
| face-api | 0.5-2 cores | 512MB-2GB |
| postgres | 2 cores | 2GB |
| redis | 1 core | 1GB |
| traefik | 0.5 cores | 256MB |
| prometheus | 1 core | 1GB |
| grafana | 0.5 cores | 512MB |

## Environment Variables

```bash
# AlertManager
SMTP_HOST=smtp.example.com:25
PAGERDUTY_SERVICE_KEY=your_key
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
TELEGRAM_BOT_TOKEN=your_bot_token

# Grafana
GRAFANA_ADMIN_PASSWORD=secure_password

# Environment
ENVIRONMENT=production
```

## Health Checks

```bash
# Prometheus
curl http://localhost:9090/-/healthy

# AlertManager
curl http://localhost:9093/-/healthy

# Grafana
curl http://localhost:3000/api/health
```

## Troubleshooting

```bash
# Check service status
docker-compose -f docker-compose.monitoring.yml ps

# View logs
docker-compose -f docker-compose.monitoring.yml logs -f prometheus
docker-compose -f docker-compose.monitoring.yml logs -f grafana

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check AlertManager status
curl http://localhost:9093/api/v1/status
```