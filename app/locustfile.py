# locustfile.py

"""
Load Testing для Face Recognition Service

Тестирует основные endpoints:
- Authentication (login/register)
- Reference upload
- Face verification
- Liveness check
- User stats

Запуск:
    # Web UI (интерактивный режим)
    locust -f locustfile.py --host http://localhost:8000

    # Headless (100 users, 10/sec spawn rate, 5 min)
    locust -f locustfile.py --headless -u 100 -r 10 -t 5m --host http://localhost:8000

    # С выводом в HTML
    locust -f locustfile.py --headless -u 100 -r 10 -t 5m --host http://localhost:8000 --html report.html

Performance targets:
    - p50 latency: < 500ms
    - p95 latency: < 1000ms
    - p99 latency: < 2000ms
    - Success rate: > 99%
    - Throughput: > 100 RPS
"""

import base64
import json
import os
import random
import time
from io import BytesIO
from typing import Any, Dict, Optional

import numpy as np
from locust import HttpUser, between, events, task
from locust.exception import StopUser
from PIL import Image

# ==================== Configuration ====================

# Test data paths
TEST_IMAGES_DIR = "tests/fixtures"
TEST_IMAGE_PATH = os.path.join(TEST_IMAGES_DIR, "test_face.jpg")

# Performance thresholds (milliseconds)
THRESHOLDS = {
    "auth_login": 500,
    "auth_register": 1000,
    "reference_upload": 2000,  # Slower due to ML processing
    "verify": 1500,  # Critical endpoint
    "liveness": 1500,
    "stats": 300,
}

# Test users configuration
TEST_USER_PREFIX = "loadtest_user_"
TEST_USER_PASSWORD = "LoadTest123!@#"


# ==================== Helper Functions ====================


def generate_test_image(size=(640, 480), format="JPEG") -> bytes:
    """
    Generate synthetic test image (for when real images unavailable)

    Args:
        size: Image dimensions (width, height)
        format: Image format (JPEG, PNG)

    Returns:
        bytes: Image data
    """
    # Create random RGB image
    img_array = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(img_array, mode="RGB")

    # Add some structure (fake face-like pattern)
    # This makes it more realistic for ML models
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)

    # Draw oval (fake face)
    center_x, center_y = size[0] // 2, size[1] // 2
    draw.ellipse(
        [center_x - 100, center_y - 120, center_x + 100, center_y + 120],
        fill=(220, 180, 150),
    )

    # Draw eyes
    draw.ellipse(
        [center_x - 60, center_y - 40, center_x - 30, center_y - 10], fill=(50, 50, 50)
    )
    draw.ellipse(
        [center_x + 30, center_y - 40, center_x + 60, center_y - 10], fill=(50, 50, 50)
    )

    # Convert to bytes
    buffer = BytesIO()
    img.save(buffer, format=format)
    return buffer.getvalue()


def load_test_image() -> bytes:
    """
    Load test image from file or generate if not exists

    Returns:
        bytes: Image data
    """
    if os.path.exists(TEST_IMAGE_PATH):
        with open(TEST_IMAGE_PATH, "rb") as f:
            return f.read()
    else:
        print(
            f"⚠️  Test image not found at {TEST_IMAGE_PATH}, generating synthetic image..."
        )
        return generate_test_image()


# ==================== Locust Events (Metrics Collection) ====================


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """
    Custom metrics collection on each request
    """
    # Check if request exceeds threshold
    threshold = THRESHOLDS.get(name, 1000)

    if response_time > threshold:
        print(
            f"⚠️  SLOW REQUEST: {name} took {response_time}ms (threshold: {threshold}ms)"
        )


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """
    Initialize test environment
    """
    print("=" * 80)
    print("🚀 LOAD TEST STARTING")
    print("=" * 80)
    print(f"Target host: {environment.host}")
    print(f"Test images dir: {TEST_IMAGES_DIR}")
    print(f"Performance thresholds: {THRESHOLDS}")
    print("=" * 80)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """
    Generate test summary
    """
    stats = environment.stats

    print("\n" + "=" * 80)
    print("📊 LOAD TEST SUMMARY")
    print("=" * 80)

    print(f"Total requests: {stats.total.num_requests}")
    print(f"Total failures: {stats.total.num_failures}")
    print(f"Success rate: {(1 - stats.total.fail_ratio) * 100:.2f}%")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"Median response time: {stats.total.median_response_time:.2f}ms")
    print(f"95th percentile: {stats.total.get_response_time_percentile(0.95):.2f}ms")
    print(f"99th percentile: {stats.total.get_response_time_percentile(0.99):.2f}ms")
    print(f"Max response time: {stats.total.max_response_time:.2f}ms")
    print(f"RPS: {stats.total.total_rps:.2f}")

    # Check if targets met
    print("\n" + "=" * 80)
    print("🎯 PERFORMANCE TARGET VALIDATION")
    print("=" * 80)

    p50 = stats.total.median_response_time
    p95 = stats.total.get_response_time_percentile(0.95)
    p99 = stats.total.get_response_time_percentile(0.99)
    success_rate = (1 - stats.total.fail_ratio) * 100
    rps = stats.total.total_rps

    results = []
    results.append(("p50 < 500ms", p50 < 500, f"{p50:.0f}ms"))
    results.append(("p95 < 1000ms", p95 < 1000, f"{p95:.0f}ms"))
    results.append(("p99 < 2000ms", p99 < 2000, f"{p99:.0f}ms"))
    results.append(("Success rate > 99%", success_rate > 99, f"{success_rate:.1f}%"))
    results.append(("RPS > 100", rps > 100, f"{rps:.1f}"))

    for target, passed, value in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {target} (actual: {value})")

    print("=" * 80)


# ==================== User Behavior Classes ====================


class FaceRecognitionUser(HttpUser):
    """
    Base user class for face recognition service
    Simulates realistic user behavior
    """

    # Wait time between tasks (1-3 seconds)
    wait_time = between(1, 3)

    # User state
    token: Optional[str] = None
    user_id: Optional[str] = None
    email: Optional[str] = None
    test_image: Optional[bytes] = None

    def on_start(self):
        """
        Called when user starts (login/register)
        """
        # Load test image once per user
        self.test_image = load_test_image()

        # Generate unique user credentials
        user_number = random.randint(1, 10000)
        self.email = f"{TEST_USER_PREFIX}{user_number}@example.com"

        # Try to login, if fails then register
        if not self.login():
            self.register()
            self.login()

        # Upload reference image (required for verification)
        self.upload_reference()

    def on_stop(self):
        """
        Called when user stops (cleanup)
        """
        # Optional: Logout or cleanup
        pass

    # ==================== Authentication ====================

    def register(self) -> bool:
        """
        Register new user

        Returns:
            bool: True if successful
        """
        with self.client.post(
            "/api/v1/auth/register",
            json={
                "email": self.email,
                "password": TEST_USER_PASSWORD,
                "phone": f"+7900{random.randint(1000000, 9999999)}",
            },
            catch_response=True,
            name="auth_register",
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
                return True
            elif (
                response.status_code == 400
                and "already exists" in response.text.lower()
            ):
                # User already exists, that's OK
                response.success()
                return True
            else:
                response.failure(f"Register failed: {response.status_code}")
                return False

    def login(self) -> bool:
        """
        Login user and get JWT token

        Returns:
            bool: True if successful
        """
        with self.client.post(
            "/api/v1/auth/login",
            json={"email": self.email, "password": TEST_USER_PASSWORD},
            catch_response=True,
            name="auth_login",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("access_token")
                self.user_id = data.get("user_id")

                if self.token:
                    response.success()
                    return True
                else:
                    response.failure("No token in response")
                    return False
            else:
                response.failure(f"Login failed: {response.status_code}")
                return False

    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get authorization headers

        Returns:
            dict: Headers with Bearer token
        """
        if not self.token:
            raise StopUser("No authentication token available")

        return {"Authorization": f"Bearer {self.token}"}

    # ==================== Reference Management ====================

    def upload_reference(self) -> bool:
        """
        Upload reference image

        Returns:
            bool: True if successful
        """
        with self.client.post(
            "/api/v1/reference/upload",
            files={"file": ("test_face.jpg", self.test_image, "image/jpeg")},
            headers=self.get_auth_headers(),
            catch_response=True,
            name="reference_upload",
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
                return True
            else:
                response.failure(f"Reference upload failed: {response.status_code}")
                return False

    # ==================== Tasks (Weighted) ====================

    @task(5)  # 50% of traffic - most common operation
    def verify_face(self):
        """
        Verify face against reference
        This is the CRITICAL endpoint for performance
        """
        with self.client.post(
            "/api/v1/verify",
            files={"file": ("verify_face.jpg", self.test_image, "image/jpeg")},
            headers=self.get_auth_headers(),
            catch_response=True,
            name="verify",
        ) as response:
            if response.status_code == 200:
                data = response.json()

                # Validate response structure
                if "is_match" in data and "similarity_score" in data:
                    response.success()
                else:
                    response.failure("Invalid response structure")
            elif response.status_code == 404:
                # No reference found - expected for some users
                response.success()
            else:
                response.failure(f"Verify failed: {response.status_code}")

    @task(3)  # 30% of traffic
    def check_liveness(self):
        """
        Check liveness (anti-spoofing)
        """
        with self.client.post(
            "/api/v1/liveness",
            files={"file": ("liveness_check.jpg", self.test_image, "image/jpeg")},
            headers=self.get_auth_headers(),
            catch_response=True,
            name="liveness",
        ) as response:
            if response.status_code == 200:
                data = response.json()

                if "is_live" in data:
                    response.success()
                else:
                    response.failure("Invalid liveness response")
            else:
                response.failure(f"Liveness check failed: {response.status_code}")

    @task(1)  # 10% of traffic
    def get_user_stats(self):
        """
        Get user statistics
        """
        with self.client.get(
            "/api/v1/user/stats",
            headers=self.get_auth_headers(),
            catch_response=True,
            name="stats",
        ) as response:
            if response.status_code == 200:
                data = response.json()

                if "total_verifications" in data:
                    response.success()
                else:
                    response.failure("Invalid stats response")
            else:
                response.failure(f"Stats request failed: {response.status_code}")

    @task(1)  # 10% of traffic - occasional reference update
    def update_reference(self):
        """
        Update reference image (simulate user re-enrolling)
        """
        # Generate slightly different image
        modified_image = generate_test_image(
            size=(640 + random.randint(-50, 50), 480 + random.randint(-50, 50))
        )

        with self.client.post(
            "/api/v1/reference/upload",
            files={"file": ("new_reference.jpg", modified_image, "image/jpeg")},
            headers=self.get_auth_headers(),
            catch_response=True,
            name="reference_upload",
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"Reference update failed: {response.status_code}")


# ==================== Specialized User Classes ====================


class AdminUser(HttpUser):
    """
    Admin user with different access patterns
    """

    wait_time = between(5, 10)  # Admins check less frequently
    token: Optional[str] = None

    def on_start(self):
        """Login as admin"""
        # Use predefined admin credentials
        with self.client.post(
            "/api/v1/auth/login",
            json={"email": "admin@example.com", "password": "AdminPassword123"},
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                self.token = response.json().get("access_token")
            else:
                raise StopUser("Admin login failed")

    def get_auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    @task(1)
    def get_system_stats(self):
        """Get system-wide statistics"""
        with self.client.get(
            "/api/v1/admin/stats",
            headers=self.get_auth_headers(),
            catch_response=True,
            name="admin_stats",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Admin stats failed: {response.status_code}")

    @task(1)
    def get_cache_metrics(self):
        """Get cache performance metrics"""
        with self.client.get(
            "/api/v1/metrics/cache",
            headers=self.get_auth_headers(),
            catch_response=True,
            name="cache_metrics",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Cache metrics failed: {response.status_code}")

    @task(1)
    def get_database_metrics(self):
        """Get database metrics"""
        with self.client.get(
            "/api/v1/metrics/database",
            headers=self.get_auth_headers(),
            catch_response=True,
            name="database_metrics",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Database metrics failed: {response.status_code}")


class HealthCheckUser(HttpUser):
    """
    Lightweight user that only checks health endpoint
    Simulates monitoring systems
    """

    wait_time = between(10, 30)

    @task(1)
    def check_health(self):
        """Health check endpoint"""
        with self.client.get("/health", catch_response=True, name="health") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")


# ==================== Custom Scenarios ====================


class SpikeTestUser(HttpUser):
    """
    User for spike testing - sends bursts of requests
    """

    wait_time = between(0.1, 0.5)  # Very fast
    token: Optional[str] = None
    test_image: Optional[bytes] = None

    def on_start(self):
        self.test_image = load_test_image()
        self.email = f"{TEST_USER_PREFIX}spike_{random.randint(1, 1000)}@example.com"

        # Quick auth
        self.client.post(
            "/api/v1/auth/register",
            json={"email": self.email, "password": TEST_USER_PASSWORD},
        )

        response = self.client.post(
            "/api/v1/auth/login",
            json={"email": self.email, "password": TEST_USER_PASSWORD},
        )

        if response.status_code == 200:
            self.token = response.json().get("access_token")

    @task(1)
    def rapid_verify(self):
        """Send verification requests rapidly"""
        if not self.token:
            return

        for _ in range(5):  # Burst of 5 requests
            self.client.post(
                "/api/v1/verify",
                files={"file": ("verify.jpg", self.test_image, "image/jpeg")},
                headers={"Authorization": f"Bearer {self.token}"},
                name="verify_spike",
            )
            time.sleep(0.05)  # 50ms between bursts
