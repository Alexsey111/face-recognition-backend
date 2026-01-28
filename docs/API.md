# API Documentation

Complete API reference for Face Recognition Backend.

## Base URL

| Environment | URL |
|-------------|-----|
| Development | `http://localhost:8000` |
| Production | `https://api.yourdomain.com` |

## Authentication

All protected endpoints require JWT token in Authorization header:

```http
Authorization: Bearer <your_jwt_token>
```

### Token Expiration

| Token Type | Expiration |
|------------|------------|
| Access Token | 30 minutes |
| Refresh Token | 7 days |

## Endpoints

### Authentication

#### Register User

Create a new user account.

**Endpoint:** `POST /api/v1/auth/register`

**Request Body:**

```json
{
  "email": "user@example.com",
  "password": "SecurePass123!",
  "full_name": "John Doe"
}
```

**Validation Rules:**

| Field | Rule |
|-------|------|
| Email | Valid email format |
| Password | Minimum 8 characters, must contain uppercase, lowercase, digit, special char |
| Full Name | 2-100 characters |

**Response:** `201 Created`

```json
{
  "id": 1,
  "email": "user@example.com",
  "full_name": "John Doe",
  "is_active": true,
  "is_verified": false,
  "created_at": "2026-01-21T10:00:00Z"
}
```

**Error Responses:**

| Status | Description |
|--------|-------------|
| 400 Bad Request | Email already registered |
| 422 Unprocessable Entity | Validation error |

```json
{
  "detail": "Email already registered"
}
```

#### Login

Authenticate user and receive access token.

**Endpoint:** `POST /api/v1/auth/login`

**Content-Type:** `application/x-www-form-urlencoded`

**Request Body:**

```json
username=user@example.com&password=SecurePass123!
```

**Response:** `200 OK`

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

| Field | Type | Description |
|-------|------|-------------|
| access_token | string | JWT access token |
| token_type | string | Always "bearer" |
| expires_in | int | Token validity in seconds (1800 = 30 min) |

**Error Responses:**

| Status | Description |
|--------|-------------|
| 401 Unauthorized | Invalid credentials |

#### Get Current User

Get authenticated user's information.

**Endpoint:** `GET /api/v1/auth/me`

**Headers:** `Authorization: Bearer <token>`

**Response:** `200 OK`

```json
{
  "id": 1,
  "email": "user@example.com",
  "full_name": "John Doe",
  "is_active": true,
  "is_verified": true,
  "created_at": "2026-01-21T10:00:00Z",
  "has_reference": true,
  "reference_created_at": "2026-01-21T11:00:00Z"
}
```

| Field | Type | Description |
|-------|------|-------------|
| id | int | User ID |
| email | string | User email |
| full_name | string | User's full name |
| is_active | boolean | Account active status |
| is_verified | boolean | Email verified status |
| has_reference | boolean | Has reference face uploaded |
| reference_created_at | datetime | When reference was created (if exists) |

**Error Responses:

### Reference Management

#### Create/Update Reference

Set or update user's reference image.

**Endpoint:** `PUT /api/v1/reference`

**Headers:** `Authorization: Bearer <token>`

**Request Body:**

```json
{
  "file_key": "uploads/user_1/2026/01/21/abc123.jpg",
  "replace_existing": true
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| file_key | string | Path to uploaded file |
| replace_existing | boolean | Replace existing reference (default: true) |

**Response:** `200 OK`

```json
{
  "reference_id": "ref_xyz789",
  "user_id": 1,
  "embedding_version": "v1.0",
  "quality_score": 0.95,
  "created_at": "2026-01-21T10:30:00Z",
  "expires_at": null,
  "metadata": {
    "face_landmarks": 68,
    "pose_angle": 2.5,
    "brightness": 0.75
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| reference_id | string | Unique reference identifier |
| user_id | int | User ID |
| embedding_version | string | Version of embedding model used |
| quality_score | float | Image quality (0.0-1.0) |
| created_at | datetime | When reference was created |
| expires_at | datetime | Reference expiration (null = never) |
| metadata.face_landmarks | int | Number of detected landmarks |
| metadata.pose_angle | float | Face pose in degrees |
| metadata.brightness | float | Image brightness (0.0-1.0) |

**Error Responses:**

| Status | Description |
|--------|-------------|
| 400 Bad Request | Invalid file_key or low quality |
| 404 Not Found | File not found |

#### Get Reference

Retrieve current reference information.

**Endpoint:** `GET /api/v1/reference`

**Headers:** `Authorization: Bearer <token>`

**Response:** `200 OK`

```json
{
  "reference_id": "ref_xyz789",
  "has_reference": true,
  "created_at": "2026-01-21T10:30:00Z",
  "updated_at": "2026-01-21T10:30:00Z",
  "embedding_version": "v1.0",
  "quality_score": 0.95
}
```

**Response:** `404 Not Found` (if no reference exists)

```json
{
  "detail": "No reference image found for user"
}
```

#### Delete Reference

Remove user's reference image.

**Endpoint:** `DELETE /api/v1/reference`

**Headers:** `Authorization: Bearer <token>`

**Response:** `204 No Content`

**Error Responses:**

| Status | Description |
|--------|-------------|
| 404 Not Found | No reference exists |

### Face Verification

#### Verify Face

Compare uploaded image with reference.

**Endpoint:** `POST /api/v1/verify`

**Headers:** `Authorization: Bearer <token>`

**Request Body:**

```json
{
  "file_key": "uploads/user_1/2026/01/21/verify_abc.jpg",
  "threshold": 0.7,
  "return_details": true
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| file_key | string | - | File to verify (required) |
| threshold | float | 0.7 | Similarity threshold (0.0-1.0) |
| return_details | boolean | false | Include detailed analysis |

**Response:** `200 OK`

```json
{
  "verification_id": "ver_123456",
  "is_match": true,
  "similarity_score": 0.89,
  "confidence": 0.95,
  "threshold_used": 0.7,
  "processing_time_ms": 234,
  "timestamp": "2026-01-21T10:45:00Z",
  "details": {
    "face_detected": true,
    "face_count": 1,
    "quality_score": 0.92,
    "pose_angle": 3.2,
    "lighting_quality": 0.88,
    "embeddings_distance": 0.11
  }
}
```

**Match Decision Logic:**

| Field | Description |
|-------|-------------|
| is_match | true if similarity_score >= threshold |
| confidence | Model certainty (0.0-1.0) |
| similarity_score | Higher = better match |

**Error Responses:**

| Status | Description |
|--------|-------------|
| 400 Bad Request | No face detected or poor quality |
| 404 Not Found | No reference image exists |
| 422 Unprocessable Entity | Invalid parameters |

```json
{
  "detail": "No reference image found. Please create reference first.",
  "error_code": "NO_REFERENCE",
  "next_step": "POST /api/v1/reference"
}
```

#### Batch Verification

Verify image against multiple reference faces.

**Endpoint:** `POST /api/v1/verify/batch`

**Headers:** `Authorization: Bearer <token>`

**Request Body:**

```json
{
  "file_key": "uploads/user_1/2026/01/21/verify_abc.jpg",
  "reference_ids": ["ref_abc123", "ref_def456"],
  "threshold": 0.7
}
```

**Response:** `200 OK`

```json
{
  "results": [
    {
      "reference_id": "ref_abc123",
      "is_match": true,
      "similarity_score": 0.85,
      "confidence": "high"
    },
    {
      "reference_id": "ref_def456",
      "is_match": false,
      "similarity_score": 0.32,
      "confidence": "low"
    }
  ],
  "best_match": {
    "reference_id": "ref_abc123",
    "similarity_score": 0.85
  }
}
```

### Liveness Detection

#### Check Liveness

Verify image is from live person, not photo/video.

**Endpoint:** `POST /api/v1/liveness`

**Headers:** `Authorization: Bearer <token>`

**Request Body:**

```json
{
  "file_key": "uploads/user_1/2026/01/21/liveness_abc.jpg",
  "mode": "passive",
  "strict": true
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| file_key | string | - | Image to check (required) |
| mode | string | "passive" | "passive" or "active" |
| strict | boolean | false | Stricter checks |

**Response:** `200 OK`

```json
{
  "liveness_id": "live_789xyz",
  "is_live": true,
  "liveness_score": 0.94,
  "confidence": 0.97,
  "method": "passive",
  "timestamp": "2026-01-21T10:50:00Z",
  "checks": {
    "texture_analysis": "pass",
    "depth_detection": "pass",
    "micro_movements": "pass",
    "screen_detection": "pass",
    "print_detection": "pass"
  },
  "risk_factors": []
}
```

| Field | Type | Description |
|-------|------|-------------|
| liveness_id | string | Unique liveness check ID |
| is_live | boolean | Whether image is from live person |
| liveness_score | float | Liveness score (0.0-1.0) |
| confidence | float | Model confidence (0.0-1.0) |
| method | string | Detection method used |
| checks | object | Individual check results |
| risk_factors | array | Detected risk factors |

**Liveness Score Interpretation:**

| Score Range | Interpretation |
|-------------|----------------|
| 0.9 - 1.0 | Very likely live |
| 0.7 - 0.9 | Likely live |
| 0.5 - 0.7 | Uncertain |
| 0.0 - 0.5 | Likely spoof |

**Error Responses:**

| Status | Description |
|--------|-------------|
| 400 Bad Request | Poor quality or no face detected |

```json
{
  "detail": "Liveness check failed",
  "error_code": "LIVENESS_FAILED",
  "is_live": false,
  "liveness_score": 0.23,
  "risk_factors": [
    "screen_reflection_detected",
    "lack_of_depth_information"
  ],
  "suggestions": [
    "Use device camera directly",
    "Ensure good lighting",
    "Avoid photos of photos"
  ]
}
```

### Health & Monitoring

#### Health Check

Check API health status.

**Endpoint:** `GET /api/v1/health`

**Response:** `200 OK` (healthy)

```json
{
  "status": "healthy",
  "timestamp": "2026-01-21T10:55:00Z",
  "version": "1.0.0",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "storage": "healthy",
    "ml_models": "healthy"
  },
  "uptime_seconds": 86400
}
```

**Response:** `503 Service Unavailable` (unhealthy)

```json
{
  "status": "unhealthy",
  "timestamp": "2026-01-21T10:55:00Z",
  "services": {
    "database": "healthy",
    "redis": "unhealthy",
    "storage": "healthy",
    "ml_models": "healthy"
  },
  "errors": [
    "Redis connection failed"
  ]
}
```

| Service | Status Values |
|---------|---------------|
| database | healthy, unhealthy |
| redis | healthy, unhealthy |
| storage | healthy, unhealthy |
| ml_models | healthy, unhealthy, loading |

#### Metrics

Prometheus metrics endpoint.

**Endpoint:** `GET /metrics`

**Response:** `200 OK` (Prometheus format)

```prometheus
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="POST",endpoint="/api/v1/verify",status="200"} 1523

# HELP http_request_duration_seconds HTTP request latency
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{le="0.1"} 1234
http_request_duration_seconds_bucket{le="0.5"} 1450
```

**Available Metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| http_requests_total | counter | Total HTTP requests |
| http_request_duration_seconds | histogram | Request latency |
| verifications_total | counter | Total verifications |
| verifications_success_total | counter | Successful verifications |
| verifications_failed_total | counter | Failed verifications |
| liveness_checks_total | counter | Total liveness checks |
| active_users_gauge | gauge | Currently active users |
| database_connections_gauge | gauge | Open DB connections |
| model_inference_seconds | histogram | ML inference latency |

### Webhooks

#### Create Upload Session

Initialize file upload session with presigned URL.

**Endpoint:** `POST /api/v1/upload`

**Headers:** `Authorization: Bearer <token>`

**Request Body:** (Optional)

```json
{
  "purpose": "reference",
  "metadata": {
    "device": "iPhone 15",
    "app_version": "1.0.0"
  }
}
```

**Response:** `200 OK`

```json
{
  "session_id": "sess_abc123def456",
  "upload_url": "https://minio.example.com/uploads/...",
  "expires_at": "2026-01-21T10:15:00Z",
  "max_file_size": 10485760,
  "allowed_types": ["image/jpeg", "image/png", "image/heic"]
}
```

| Field | Type | Description |
|-------|------|-------------|
| session_id | string | Unique session identifier |
| upload_url | string | Presigned upload URL |
| expires_at | datetime | Session expiration time |
| max_file_size | int | Maximum file size in bytes |
| allowed_types | array | Allowed MIME types |

#### Upload File

Upload file to session.

**Endpoint:** `POST /api/v1/upload/{session_id}/file`

**Headers:**
- `Authorization: Bearer <token>`
- `Content-Type: multipart/form-data`

**Request Body:** (Multipart Form Data)

```
file: <binary data>
```

**Validation:**

| Rule | Value |
|------|-------|
| Max size | 10MB (10,485,760 bytes) |
| Allowed types | JPG, PNG, HEIC |
| Image must contain | Detectable face |
| Minimum resolution | 640x480 pixels |

**Response:** `200 OK`

```json
{
  "file_key": "uploads/user_1/2026/01/21/abc123.jpg",
  "file_size": 2048576,
  "mime_type": "image/jpeg",
  "dimensions": {
    "width": 1920,
    "height": 1080
  },
  "face_detected": true,
  "face_count": 1,
  "quality_score": 0.95
}
```

**Error Responses:**

| Status | Description |
|--------|-------------|
| 400 Bad Request | No face detected, multiple faces, poor quality |
| 413 Payload Too Large | File exceeds size limit |
| 415 Unsupported Media Type | Invalid file type |

```json
{
  "detail": "No face detected in image",
  "error_code": "NO_FACE_DETECTED",
  "suggestions": [
    "Ensure face is clearly visible",
    "Improve lighting conditions",
    "Remove obstructions (mask, glasses)"
  ]
}
```

### Webhooks

#### Register Webhook

Register a webhook URL for event notifications.

**Endpoint:** `POST /api/v1/webhook`

**Headers:** `Authorization: Bearer <token>`

**Request Body:**

```json
{
  "url": "https://yourdomain.com/webhook",
  "events": ["verification.completed", "liveness.failed"],
  "secret": "webhook_secret_key"
}
```

**Response:** `201 Created`

```json
{
  "webhook_id": "wh_abc123",
  "url": "https://yourdomain.com/webhook",
  "events": ["verification.completed", "liveness.failed"],
  "is_active": true,
  "created_at": "2026-01-21T10:00:00Z"
}
```

#### List Webhooks

Get all webhooks for authenticated user.

**Endpoint:** `GET /api/v1/webhook`

**Headers:** `Authorization: Bearer <token>`

**Response:** `200 OK`

```json
{
  "webhooks": [
    {
      "webhook_id": "wh_abc123",
      "url": "https://yourdomain.com/webhook",
      "events": ["verification.completed"],
      "is_active": true
    }
  ]
}
```

## Rate Limits

All endpoints are rate-limited:

| Endpoint Type | Limit | Window |
|---------------|-------|--------|
| Authentication | 5 requests | 1 minute |
| File Upload | 10 requests | 1 minute |
| Verification | 20 requests | 1 minute |
| General | 100 requests | 1 minute |

**Rate Limit Headers:**

```http
X-RateLimit-Limit: 20
X-RateLimit-Remaining: 15
X-RateLimit-Reset: 1642766400
```

**Error Response:** `429 Too Many Requests`

```json
{
  "detail": "Rate limit exceeded",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "retry_after": 45
}
```

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| VALIDATION_ERROR | 422 | Input validation failed |
| AUTHENTICATION_FAILED | 401 | Invalid credentials |
| UNAUTHORIZED | 401 | Missing or invalid token |
| FORBIDDEN | 403 | Insufficient permissions |
| NOT_FOUND | 404 | Resource not found |
| NO_FACE_DETECTED | 400 | No face in image |
| MULTIPLE_FACES | 400 | Multiple faces detected |
| POOR_IMAGE_QUALITY | 400 | Image quality too low |
| NO_REFERENCE | 404 | Reference image missing |
| LIVENESS_FAILED | 400 | Liveness check failed |
| RATE_LIMIT_EXCEEDED | 429 | Too many requests |
| INTERNAL_ERROR | 500 | Server error |

## Webhooks

### Configure Webhooks

Configure webhooks to receive verification results.

**Endpoint:** `POST /api/v1/webhooks`

**Headers:** `Authorization: Bearer <token>`

**Request Body:**

```json
{
  "url": "https://your-crm.com/webhooks/face-verification",
  "events": ["verification.completed", "reference.updated"],
  "secret": "your_webhook_secret"
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| url | string | Webhook endpoint URL |
| events | array | Event types to subscribe |
| secret | string | Secret for signature verification |

**Available Events:**

| Event | Description |
|-------|-------------|
| verification.completed | Verification finished |
| verification.failed | Verification failed |
| liveness.completed | Liveness check finished |
| liveness.failed | Liveness check failed |
| reference.created | Reference image created |
| reference.updated | Reference image updated |
| reference.deleted | Reference image deleted |

**Response:** `201 Created`

```json
{
  "webhook_id": "wh_abc123",
  "url": "https://your-crm.com/webhooks/face-verification",
  "events": ["verification.completed", "reference.updated"],
  "is_active": true,
  "created_at": "2026-01-21T10:00:00Z"
}
```

### Webhook Payload

When verification completes, POST request sent to your URL:

```json
{
  "event": "verification.completed",
  "timestamp": "2026-01-21T11:00:00Z",
  "data": {
    "verification_id": "ver_123456",
    "user_id": 1,
    "is_match": true,
    "similarity_score": 0.89,
    "confidence": 0.95
  },
  "signature": "sha256=abc123..."
}
```

### Verify Webhook Signature

```python
import hmac
import hashlib

def verify_webhook(payload, signature, secret):
    expected = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)
```

### Webhook Retries

- Webhooks are retried 3 times with exponential backoff
- Retry intervals: 1min, 5min, 30min
- Failed webhooks are logged and can be viewed in dashboard

## SDKs & Examples

### Python Example

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

# Login
response = requests.post(
    f"{BASE_URL}/auth/login",
    data={"username": "user@example.com", "password": "password"}
)
token = response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# Upload file
with open("face.jpg", "rb") as f:
    files = {"file": f}
    
    # Create session
    session_response = requests.post(
        f"{BASE_URL}/upload",
        headers=headers,
    )
    session_id = session_response.json()["session_id"]
    
    # Upload file
    upload_response = requests.post(
        f"{BASE_URL}/upload/{session_id}/file",
        headers=headers,
        files=files
    )
    file_key = upload_response.json()["file_key"]

# Verify face
verify_response = requests.post(
    f"{BASE_URL}/verify",
    headers=headers,
    json={"file_key": file_key}
)
result = verify_response.json()
print(f"Match: {result['is_match']}, Score: {result['similarity_score']}")
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');

const BASE_URL = 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: BASE_URL,
});

// Login
async function login() {
  const response = await api.post('/auth/login', null, {
    auth: { username: 'user@example.com', password: 'password' }
  });
  api.defaults.headers.common['Authorization'] = `Bearer ${response.data.access_token}`;
  return response.data;
}

// Upload file
async function uploadFile(filePath) {
  const FormData = require('form-data');
  const fs = require('fs');
  
  const form = new FormData();
  form.append('file', fs.createReadStream(filePath));
  
  const session = await api.post('/upload', form);
  const upload = await api.post(
    `/upload/${session.data.session_id}/file`,
    form
  );
  return upload.data.file_key;
}

// Verify face
async function verifyFace(fileKey) {
  const response = await api.post('/verify', { file_key: fileKey });
  return response.data;
}

// Usage
(async () => {
  await login();
  const fileKey = await uploadFile('./face.jpg');
  const result = await verifyFace(fileKey);
  console.log(`Match: ${result.is_match}, Score: ${result.similarity_score}`);
})();
```

### cURL Examples

```bash
# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=password"

# Create upload session
curl -X POST http://localhost:8000/api/v1/upload \
  -H "Authorization: Bearer YOUR_TOKEN"

# Upload file
curl -X POST http://localhost:8000/api/v1/upload/sess_abc123/file \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@face.jpg"

# Verify face
curl -X POST http://localhost:8000/api/v1/verify \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"file_key": "uploads/user_1/image.jpg"}'

# Check liveness
curl -X POST http://localhost:8000/api/v1/liveness \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"file_key": "uploads/user_1/liveness.jpg"}'

# Health check
curl http://localhost:8000/api/v1/health

# Get reference
curl http://localhost:8000/api/v1/reference \
  -H "Authorization: Bearer YOUR_TOKEN"

# Delete reference
curl -X DELETE http://localhost:8000/api/v1/reference \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Go Example

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
    "os"
)

const BASE_URL = "http://localhost:8000/api/v1"

func main() {
    // Login
    client := &http.Client{}
    req, _ := http.NewRequest("POST", BASE_URL+"/auth/login", 
        bytes.NewBufferString("username=user@example.com&password=password"))
    req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
    
    resp, _ := client.Do(req)
    defer resp.Body.Close()
    
    var loginResp map[string]interface{}
    json.NewDecoder(resp.Body).Decode(&loginResp)
    
    fmt.Println("Login successful, token:", loginResp["access_token"])
    
    // Verify Face with HEIC support
    verifyFace("uploads/user_1/image.jpg", loginResp["access_token"].(string))
}

func verifyFace(fileKey string, token string) {
    client := &http.Client{}
    body, _ := json.Marshal(map[string]string{"file_key": fileKey})
    req, _ := http.NewRequest("POST", BASE_URL+"/verify", bytes.NewBuffer(body))
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Authorization", "Bearer "+token)
    
    resp, _ := client.Do(req)
    defer resp.Body.Close()
    
    var verifyResp map[string]interface{}
    json.NewDecoder(resp.Body).Decode(&verifyResp)
    
    fmt.Printf("Match: %v, Score: %.2f\n", 
        verifyResp["is_match"], 
        verifyResp["similarity_score"])
}
```

---

## Image Format Support

### Supported Formats

The service supports the following image formats:

| Format | Extensions | Description | Auto-conversion |
|--------|------------|-------------|-----------------|
| JPEG   | .jpg, .jpeg | Standard format | - |
| PNG    | .png | Format with transparency | - |
| **HEIC** | **.heic, .heif** | **Apple format (iOS/macOS)** | **✓ Yes → JPEG** |
| WebP   | .webp | Google format | - |

### HEIC (High Efficiency Image Container)

HEIC is a modern image compression format used by Apple starting with iOS 11.

**Processing Features:**
- ✅ Automatic format detection via magic bytes
- ✅ Conversion to JPEG with quality preservation (95%)
- ✅ Transparency handling (RGB conversion with white background)
- ✅ HEIF variant support
- ⚠️ Slight processing time increase (~100-200ms)

### Upload Endpoints with HEIC Support

#### POST /api/v1/upload/validate

Validate uploaded image.

```bash
curl -X POST "http://localhost:8000/api/v1/upload/validate" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@photo.heic"
```

**Response:**

```json
{
  "status": "success",
  "message": "Изображение валидно",
  "image_info": {
    "width": 3024,
    "height": 4032,
    "format": "HEIC",
    "mime_type": "image/heic",
    "size_mb": 2.45
  },
  "converted_from_heic": true,
  "dimensions": "3024x4032"
}
```

#### GET /api/v1/upload/supported-formats

Get list of supported formats.

```bash
curl "http://localhost:8000/api/v1/upload/supported-formats" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response:**

```json
{
  "supported_extensions": [".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"],
  "max_file_size_mb": 10,
  "max_dimension": 4096,
  "min_dimension": 160,
  "formats": {
    "JPEG": {
      "extensions": [".jpg", ".jpeg"],
      "description": "Standard format"
    },
    "PNG": {
      "extensions": [".png"],
      "description": "Format with transparency support"
    },
    "HEIC": {
      "extensions": [".heic", ".heif"],
      "description": "Apple format (auto-converted to JPEG)"
    },
    "WebP": {
      "extensions": [".webp"],
      "description": "Google format"
    }
  }
}
```

#### POST /api/v1/upload/convert-heic

Convert HEIC/HEIF to JPEG.

```bash
curl -X POST "http://localhost:8000/api/v1/upload/convert-heic" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@photo.heic"
```

**Response:**

```json
{
  "status": "success",
  "message": "Конвертация завершена",
  "jpeg_size_bytes": 524288,
  "jpeg_size_mb": 0.5,
  "dimensions": "1920x1080",
  "jpeg_base64": "/9j/4AAQSkZJRgABAQEAYABgAAD..."
}
```

### Limitations

| Parameter | Value |
|-----------|-------|
| Maximum file size | 10 MB |
| Maximum resolution | 4096x4096 px |
| Minimum resolution | 160x160 px (for face recognition) |
| JPEG quality during conversion | 95% |

### Technical Details

#### Format Detection by Magic Bytes

| Format | Magic Bytes |
|--------|-------------|
| JPEG   | `FF D8 FF E0` or `FF D8 FF E1` |
| PNG    | `89 50 4E 47 0D 0A 1A 0A` |
| HEIC   | `ftypheic` or `ftypmif1` (offset 4) |
| HEIF   | `ftypheif` (offset 4) |
| WebP   | `RIFF....WEBP` |

#### HEIC → JPEG Conversion

```python
def convert_heic_to_jpeg(file_data: bytes, quality: int = 95) -> bytes:
    image = Image.open(io.BytesIO(file_data))
    # Transparency handling
    if image.mode in ('RGBA', 'LA', 'P'):
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        image = background
    # Save as JPEG
    return image_to_bytes(image, format='JPEG', quality=quality)
```

### Integration with Other Endpoints

HEIC files are automatically converted in the following endpoints:

- `POST /api/v1/upload/{session_id}/file` - File upload
- `POST /api/v1/verify` - Face verification
- `POST /api/v1/liveness` - Liveness check
- `POST /api/v1/reference` - Create reference

### Testing HEIC Support

```bash
# Validate HEIC file
curl -X POST "http://localhost:8000/api/v1/upload/validate" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@iphone_photo.heic"

# Convert HEIC to JPEG
curl -X POST "http://localhost:8000/api/v1/upload/convert-heic" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@iphone_photo.heic" \
  -o converted.jpg

# Use HEIC in verification
curl -X POST "http://localhost:8000/api/v1/verify" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"file_key": "path/to/heic/file.heic"}'
```
## Используемые модели машинного обучения

### Краткий обзор

Сервис использует следующие модели машинного обучения в едином pipeline обработки:

| Задача | Модель | Точность | Скорость (GPU) |
|--------|--------|----------|----------------|
| Face Detection | MTCNN | 95% precision | 10ms |
| Face Verification | FaceNet (Inception-ResNet-v1) | 99.65% (LFW) | 5ms |
| Anti-Spoofing | MiniFASNetV2 | 98.95% (ACER) | 2ms |

**Полная документация:** См. [MODELS.md](./MODELS.md)

### Pipeline обработки

```
Input Image → Face Detection (MTCNN) → Face Alignment → 
├─→ Anti-Spoofing (MiniFASNetV2) → is_live
└─→ Embedding Extraction (FaceNet) → 512-dim vector → Comparison
```

### Конфигурация моделей

Параметры моделей настраиваются через `config.py`:

```python
# Face Verification
VERIFICATION_THRESHOLD = 0.6  # Standard (FAR < 0.1%, FRR < 1-3%)
FACENET_VERSION = "1.0.0"
EMBEDDING_SIZE = 512

# Liveness Detection
LIVENESS_THRESHOLD = 0.5  # Balanced
MINIFASNET_VERSION = "2.0.1"

# Face Detection
FACE_DETECTOR = "mtcnn"  # or "retinaface"
MIN_FACE_SIZE = 20
```

### Метрики производительности

#### Latency breakdown (полный pipeline, 1024×1024 image)

| Этап | CPU (ms) | GPU (ms) |
|------|----------|----------|
| Image loading & decode | 10 | 10 |
| Face detection (MTCNN) | 100 | 10 |
| Face alignment | 5 | 5 |
| Liveness check (MiniFASNet) | 15 | 2 |
| Embedding extraction (FaceNet) | 50 | 5 |
| Comparison | 1 | 0.5 |
| **Total** | **~201ms** | **~53ms** |

#### Throughput

| Конфигурация | Запросов/сек |
|--------------|--------------|
| Single CPU | 5 |
| Single GPU | 18 |
| 10 pods (GPU) | ~170 |

### Модели в ответах API

#### Verification Response

```json
{
  "verification_id": "ver_123456",
  "is_match": true,
  "similarity_score": 0.89,
  "confidence": 0.95,
  "threshold_used": 0.7,
  "model_info": {
    "face_detector": "mtcnn",
    "embedder": "facenet",
    "liveness_enabled": true
  }
}
```

#### Liveness Response

```json
{
  "liveness_id": "live_789xyz",
  "is_live": true,
  "liveness_score": 0.94,
  "confidence": 0.97,
  "model": "MiniFASNetV2",
  "spoof_type": null
}
```

### Health Check

Эндпоинт `/api/v1/health` возвращает статус моделей:

```json
{
  "status": "healthy",
  "services": {
    "ml_models": "healthy"
  },
  "model_versions": {
    "facenet": "1.0.0",
    "minifasnet": "2.0.1",
    "mtcnn": "1.0.0"
  }
}
```

### Метрики Prometheus

```prometheus
# HELP model_inference_seconds Model inference latency
# TYPE model_inference_seconds histogram
model_inference_seconds_bucket{model="facenet",le="0.01"} 1523
model_inference_seconds_bucket{model="facenet",le="0.05"} 1890
model_inference_seconds_bucket{model="minifasnet",le="0.005"} 1450

# HELP verifications_total Total verifications
verifications_total{model="facenet"} 5234

# HELP liveness_checks_total Total liveness checks
liveness_checks_total{model="minifasnet"} 4123
```