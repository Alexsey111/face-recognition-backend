#!/usr/bin/env python3
"""
Простой тест для проверки health endpoint.
"""

import subprocess
import time
import sys
import os

def test_health_endpoint_simple():
    """Простой тест health endpoint."""
    try:
        # Запускаем сервер в subprocess
        print("Starting server...")
        server_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "app.main:app", 
            "--host", "127.0.0.1", 
            "--port", "8003",
            "--log-level", "error"
        ], cwd=os.getcwd())
        
        # Ждем запуска сервера
        print("Waiting for server to start...")
        time.sleep(5)
        
        # Проверяем health endpoint
        import requests
        try:
            response = requests.get("http://127.0.0.1:8003/health", timeout=10)
            if response.status_code == 200:
                print("Health endpoint working!")
                print(f"Response: {response.json()}")
                return True
            else:
                print(f"Health endpoint returned {response.status_code}")
                return False
        except Exception as e:
            print(f"Health endpoint test failed: {e}")
            return False
        finally:
            # Останавливаем сервер
            print("Stopping server...")
            server_process.terminate()
            server_process.wait(timeout=5)
            
    except Exception as e:
        print(f"Server test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Health Endpoint...")
    print("=" * 40)
    
    if test_health_endpoint_simple():
        print("Health endpoint test PASSED!")
    else:
        print("Health endpoint test FAILED!")
    
    print("=" * 40)