from fastapi import APIRouter, Response
try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
except Exception:
    def generate_latest():
        return b""

    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

router = APIRouter()


@router.get("/metrics")
def metrics_endpoint():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
