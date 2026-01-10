from locust import HttpUser, task, between
import os

TEST_IMAGE = os.path.join(os.path.dirname(__file__), 'tests', 'fixtures', 'test_image.jpg')

class FaceRecognitionUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        # Attempt anonymous or test login, tolerate failures
        try:
            resp = self.client.post('/api/v1/auth/login', json={'email': 'test@example.com', 'password': 'password'})
            if resp.status_code == 200 and 'access_token' in resp.json():
                token = resp.json().get('access_token')
                self.token = token
            else:
                self.token = None
        except Exception:
            self.token = None

    @task(5)
    def verify_face(self):
        headers = {}
        if getattr(self, 'token', None):
            headers['Authorization'] = f"Bearer {self.token}"
        files = {}
        if os.path.exists(TEST_IMAGE):
            with open(TEST_IMAGE, 'rb') as f:
                files = {'file': f}
                self.client.post('/api/v1/verify', files=files, headers=headers, name='/api/v1/verify')
        else:
            # fallback to health endpoint if image missing
            self.client.get('/health')

    @task(1)
    def get_stats(self):
        headers = {}
        if getattr(self, 'token', None):
            headers['Authorization'] = f"Bearer {self.token}"
        self.client.get('/api/v1/admin/stats', headers=headers)
