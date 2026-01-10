# Deployment

See DEPLOYMENT checklist in docs and follow these steps for production deployment.

Basic Docker build:

```bash
docker build -t face-recognition:latest .
docker tag face-recognition:latest ghcr.io/<org>/face-recognition:latest
docker push ghcr.io/<org>/face-recognition:latest
```

Use Kubernetes or docker-compose with environment variables and secrets management.
