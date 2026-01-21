# scripts/test_metrics_endpoints.py

"""
Скрипт для тестирования metrics endpoints
"""

import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000/api/v1/metrics"

# Test credentials
TEST_EMAIL = "test@example.com"
TEST_PASSWORD = "TestPass123!"  # с спецсимволом для валидации


def get_admin_token() -> str:
    """Получить admin токен (регистрирует пользователя если не существует)"""
    # Сначала пробуем зарегистрировать пользователя
    try:
        reg_response = requests.post(
            "http://localhost:8000/api/v1/auth/register",
            json={
                "email": TEST_EMAIL,
                "password": TEST_PASSWORD,
                "full_name": "Test Admin"
            }
        )
        if reg_response.status_code == 409:
            print("ℹ️  User already exists, trying login...")
        elif reg_response.status_code != 201:
            print(f"⚠️  Registration warning: {reg_response.status_code}")
    except Exception as e:
        print(f"⚠️  Registration attempt failed: {e}")
    
    # Теперь пробуем войти
    response = requests.post(
        "http://localhost:8000/api/v1/auth/login",
        json={
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD
        }
    )
    
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception(f"Failed to login: {response.status_code} - {response.text}")


def test_endpoint(name: str, url: str, headers: Dict[str, str] = None):
    """Тестировать endpoint"""
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"URL: {url}")
    print('='*80)
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(json.dumps(data, indent=2))
            print("✅ PASS")
        else:
            print(f"❌ FAIL: {response.text}")
    
    except Exception as e:
        print(f"❌ ERROR: {e}")


def main():
    """Main test function"""
    print("🧪 Testing Metrics Endpoints")
    
    # Get admin token
    print("\n🔐 Getting admin token...")
    try:
        token = get_admin_token()
        headers = {"Authorization": f"Bearer {token}"}
        print("✅ Admin token obtained")
    except Exception as e:
        print(f"❌ Failed to get admin token: {e}")
        return
    
    # Test public endpoints (no auth)
    test_endpoint("Health Check", f"{BASE_URL}/health")
    test_endpoint("Detailed Health Check", f"{BASE_URL}/health/detailed")
    
    # Test protected endpoints (admin auth)
    test_endpoint("System Metrics", f"{BASE_URL}/system", headers)
    test_endpoint("Cache Metrics", f"{BASE_URL}/cache", headers)
    test_endpoint("Database Metrics", f"{BASE_URL}/database", headers)
    test_endpoint("Application Metrics", f"{BASE_URL}/application?period_hours=24", headers)
    test_endpoint("Performance Metrics", f"{BASE_URL}/performance", headers)
    test_endpoint("Metrics Summary", f"{BASE_URL}/summary", headers)
    
    print("\n" + "="*80)
    print("✅ All tests complete!")
    print("="*80)


if __name__ == "__main__":
    main()
