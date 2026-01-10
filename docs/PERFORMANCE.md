Performance tuning and testing

1) Enable SQLAlchemy query logging for analysis:

export SQLALCHEMY_ECHO=true
# or set in config

2) Run Locust load test (headless):

pip install locust
locust -f locustfile.py --headless -u 100 -r 5 -t 5m

3) Profiling example (cProfile):

python -m cProfile -o perf.prof -m uvicorn app.main:app

4) Prometheus: add scrape config for /metrics endpoint

scrape_configs:
  - job_name: "face-recognition"
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics

Notes:
- Use Redis metrics at /api/v1/admin/system/health and /metrics counters for cache hit/miss
- Monitor DB connection gauge `database_connections_active` and tune pool_size accordingly
