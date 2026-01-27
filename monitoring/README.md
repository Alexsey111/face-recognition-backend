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