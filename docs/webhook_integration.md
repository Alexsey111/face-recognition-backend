# Webhook Integration Guide

## 📋 Overview

Webhook система Face Recognition Service позволяет получать real-time уведомления о событиях верификации и проверки живости. Система гарантирует доставку через механизм retry с exponential backoff и обеспечивает безопасность через HMAC-SHA256 подпись.

---

## 🚀 Quick Start

### 1. Configuration

Добавьте webhook переменные в `.env`:

```bash
# Webhook Configuration
WEBHOOK_URL=https://your-crm.com/api/webhook
WEBHOOK_SECRET=your-secret-key-min-32-chars
WEBHOOK_TIMEOUT=10
WEBHOOK_MAX_RETRIES=3
WEBHOOK_RETRY_DELAY=1
```

### 2. Create Webhook Endpoint

```python
# your_crm/api/webhooks.py
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import hmac
import hashlib
import json

router = APIRouter()

# Your webhook secret (must match WEBHOOK_SECRET in .env)
WEBHOOK_SECRET = "your-secret-key-min-32-chars"

class WebhookPayload(BaseModel):
    event_type: str
    event_id: str
    timestamp: str
    data: Dict[str, Any]
    signature: str

def verify_signature(payload: bytes, signature: str) -> bool:
    """Verify HMAC-SHA256 signature."""
    expected = hmac.new(
        WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, expected)

@router.post("/api/webhook")
async def receive_webhook(request: Request, payload: WebhookPayload):
    """Receive webhook from Face Recognition Service."""
    
    # Verify signature
    signature = request.headers.get("X-Signature-SHA256", "")
    body = await request.body()
    
    if not verify_signature(body, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    # Process webhook
    event_type = payload.event_type
    data = payload.data
    
    if event_type == "verification.completed":
        if data["success"]:
            print(f"✅ User {data['user_id']} verified")
            # Update user status in your CRM
        else:
            print(f"❌ Verification failed for user {data['user_id']}")
            # Handle failed verification
    
    elif event_type == "liveness.completed":
        if data["is_live"]:
            print(f"✅ User {data['user_id']} passed liveness check")
        else:
            print(f"⚠️ User {data['user_id']} failed liveness check")
    
    return {"status": "received"}
```

---

## 🔐 Security

### Signature Verification

```python
import hmac
import hashlib
import json

def verify_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify HMAC-SHA256 signature of webhook payload."""
    expected_signature = hmac.new(
        secret.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, f"sha256={expected_signature}")
```

### Retry Logic

| Attempt | Delay | Total Time |
|---------|-------|------------|
| 1 | 0s | 0s |
| 2 | 1s | 1s |
| 3 | 2s | 3s |
| 4 | 4s | 7s |
| 5 | 8s | 15s |

---

## 📊 Event Types

### Verification Events

| Event Type | Description |
|------------|-------------|
| `verification.started` | Verification process started |
| `verification.completed` | Verification completed |
| `verification.failed` | Verification failed |

### Liveness Events

| Event Type | Description |
|------------|-------------|
| `liveness.started` | Liveness check started |
| `liveness.completed` | Liveness check completed |
| `liveness.failed` | Liveness check failed |

### Webhook Events

| Event Type | Description |
|------------|-------------|
| `webhook.created` | Webhook configuration created |
| `webhook.updated` | Webhook configuration updated |
| `webhook.deleted` | Webhook configuration deleted |
| `webhook.test` | Test webhook sent |

---

## 📝 Webhook Payloads

### 1. Verification Completed Payload

```json
{
  "event_type": "verification.completed",
  "event_id": "evt_01h2x3y4z5a6b7c8d9e0f",
  "timestamp": "2026-01-27T05:47:06Z",
  "data": {
    "user_id": "user_12345",
    "session_id": "sess_01abc2def3",
    "success": true,
    "confidence": 0.95,
    "verification_score": 0.9234,
    "threshold_used": 0.75,
    "match_level": "high",
    "reference_id": "ref_001",
    "reference_label": "John Doe - ID Photo",
    "processing_time_ms": 156,
    "model_version": "1.0.0"
  },
  "signature": "sha256=abc123..."
}
```

### 2. Verification Failed Payload

```json
{
  "event_type": "verification.completed",
  "event_id": "evt_01h2x3y4z5a6b7c8d9e1f",
  "timestamp": "2026-01-27T05:48:00Z",
  "data": {
    "user_id": "user_12346",
    "session_id": "sess_01def2ghi3",
    "success": false,
    "confidence": 0.35,
    "verification_score": 0.4234,
    "threshold_used": 0.75,
    "match_level": "none",
    "failure_reason": "low_similarity",
    "processing_time_ms": 142,
    "model_version": "1.0.0"
  },
  "signature": "sha256=def456..."
}
```

### 3. Liveness Check Passed Payload

```json
{
  "event_type": "liveness.completed",
  "event_id": "evt_01h2x3y4z5a6b7c8d9e2f",
  "timestamp": "2026-01-27T05:49:00Z",
  "data": {
    "user_id": "user_12345",
    "session_id": "sess_01ghi3jkl4",
    "is_live": true,
    "liveness_score": 0.98,
    "liveness_confidence": "high",
    "checks_performed": {
      "blinking": true,
      "mouth_opening": true,
      "head_turn": true,
      "face_depth": true
    },
    "anti_spoofing_score": 0.95,
    "processing_time_ms": 320,
    "model_version": "1.0.0"
  },
  "signature": "sha256=ghi789..."
}
```

### 4. Liveness Check Failed Payload

```json
{
  "event_type": "liveness.completed",
  "event_id": "evt_01h2x3y4z5a6b7c8d9e3f",
  "timestamp": "2026-01-27T05:50:00Z",
  "data": {
    "user_id": "user_12346",
    "session_id": "sess_01jkl4mno5",
    "is_live": false,
    "liveness_score": 0.25,
    "liveness_confidence": "low",
    "checks_performed": {
      "blinking": true,
      "mouth_opening": true,
      "head_turn": false,
      "face_depth": false
    },
    "anti_spoofing_score": 0.15,
    "failure_reason": "photo_attack_detected",
    "processing_time_ms": 280,
    "model_version": "1.0.0"
  },
  "signature": "sha256=jkl012..."
}
```

### 5. Batch Processing Complete Payload

```json
{
  "event_type": "batch.completed",
  "event_id": "evt_01h2x3y4z5a6b7c8d9e4f",
  "timestamp": "2026-01-27T05:51:00Z",
  "data": {
    "batch_id": "batch_001",
    "total_images": 100,
    "processed_images": 100,
    "successful_verifications": 85,
    "failed_verifications": 15,
    "results": [
      {
        "image_id": "img_001",
        "user_id": "user_12345",
        "success": true,
        "confidence": 0.92
      }
    ],
    "processing_time_seconds": 45,
    "model_version": "1.0.0"
  },
  "signature": "sha256=mno345..."
}
```

### 6. Webhook Test Payload

```json
{
  "event_type": "webhook.test",
  "event_id": "evt_01h2x3y4z5a6b7c8d9e5f",
  "timestamp": "2026-01-27T05:52:00Z",
  "data": {
    "message": "Webhook test successful",
    "webhook_id": "wh_001",
    "timestamp": "2026-01-27T05:52:00Z"
  },
  "signature": "sha256=pqr678..."
}
```

### 7. Reference Created Payload

```json
{
  "event_type": "reference.created",
  "event_id": "evt_01h2x3y4z5a6b7c8d9e6f",
  "timestamp": "2026-01-27T05:53:00Z",
  "data": {
    "reference_id": "ref_001",
    "user_id": "user_12345",
    "label": "John Doe - ID Photo",
    "quality_score": 0.95,
    "face_detected": true,
    "processing_time_ms": 120,
    "model_version": "1.0.0"
  },
  "signature": "sha256=stu901..."
}
```

---

## 🔄 Webhook Configuration

### Create Webhook

```bash
curl -X POST "http://localhost:8000/api/v1/webhook" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-crm.com/api/webhook",
    "events": ["verification.completed", "liveness.completed"],
    "secret": "your-secret-key-min-32-chars",
    "is_active": true
  }'
```

### Response

```json
{
  "id": "wh_001",
  "url": "https://your-crm.com/api/webhook",
  "events": ["verification.completed", "liveness.completed"],
  "secret": "your-secret-key-min-32-chars",
  "is_active": true,
  "created_at": "2026-01-27T05:54:00Z",
  "updated_at": "2026-01-27T05:54:00Z"
}
```

### Test Webhook

```bash
curl -X POST "http://localhost:8000/api/v1/webhook/wh_001/test" \
  -H "Authorization: Bearer <token>"
```

### Update Webhook

```bash
curl -X PUT "http://localhost:8000/api/v1/webhook/wh_001" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-crm.com/api/webhook",
    "events": ["verification.completed", "liveness.completed", "batch.completed"],
    "is_active": true
  }'
```

### Delete Webhook

```bash
curl -X DELETE "http://localhost:8000/api/v1/webhook/wh_001" \
  -H "Authorization: Bearer <token>"
```

---

## 📋 Webhook Best Practices

### 1. Idempotency

Process each webhook only once:

```python
import redis
import json

redis_client = redis.Redis()

async def process_webhook(event_id: str, data: dict):
    # Check if event was already processed
    if redis_client.setnx(f"webhook:{event_id}:processing", 1):
        try:
            # Process webhook
            await process_event(data)
            # Mark as processed
            redis_client.setex(f"webhook:{event_id}:processed", 86400, json.dumps(data))
        finally:
            redis_client.delete(f"webhook:{event_id}:processing")
```

### 2. Signature Verification

Always verify the signature:

```python
import hmac
import hashlib

def verify_webhook(payload: bytes, signature: str, secret: str) -> bool:
    expected = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, f"sha256={expected}")
```

### 3. Error Handling

Handle webhook errors gracefully:

```python
import asyncio
import aiohttp

async def send_webhook(url: str, payload: dict, secret: str):
    headers = {
        "Content-Type": "application/json",
        "X-Signature-SHA256": generate_signature(payload, secret),
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status >= 500:
                    raise WebhookRetryableError(response.status)
                return await response.json()
    except asyncio.TimeoutError:
        raise WebhookRetryableError("timeout")
```

### 4. Logging

Log all webhooks for debugging:

```python
import structlog

logger = structlog.get_logger()

async def process_webhook(event_type: str, data: dict):
    logger.info(
        "webhook_received",
        event_type=event_type,
        event_id=data.get("event_id"),
        user_id=data.get("user_id"),
    )
```

### 5. Performance

Use async processing for high throughput:

```python
import asyncio
from aiohttp import ClientSession

async def process_webhooks_batch(webhooks: list):
    async with ClientSession() as session:
        tasks = [send_webhook(session, wh) for wh in webhooks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
```

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Webhooks not received | Check URL is accessible |
| Signature verification failed | Verify secret matches configuration |
| Webhooks delayed | Check retry logic and network |
| Duplicate webhooks | Implement idempotency |

### Debug Mode

```python
# Enable webhook debugging
import logging
logging.getLogger("app.services.webhook_service").setLevel(logging.DEBUG)
```

### Test Webhook Endpoint

```python
# Test your webhook endpoint
import requests

test_payload = {
    "event_type": "webhook.test",
    "event_id": "test_001",
    "timestamp": "2026-01-27T05:55:00Z",
    "data": {"message": "test"},
    "signature": "sha256=test"
}

response = requests.post(
    "https://your-crm.com/api/webhook",
    json=test_payload
)
print(response.status_code)
```