#!/usr/bin/env python3
"""
Тест для проверки базового функционирования сервера.
"""

import subprocess
import time
import sys
import os

def test_server_basic():
    """Тест базового функционирования сервера."""
    try:
        # Запускаем сервер
        print("Starting server...")
        server_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "app.main:app", 
            "--host", "127.0.0.1", 
            "--port", "8004",
            "--log-level", "critical"
        ], cwd=os.getcwd())
        
        # Ждем запуска
        print("Waiting for server to start...")
        time.sleep(5)
        
        # Проверяем разные endpoints
        import requests
        
        endpoints = [
            ("/", "Root"),
            ("/docs", "Swagger Docs"),
            ("/openapi.json", "OpenAPI Schema"),
            ("/health", "Health Check"),
            ("/api/v1/health", "API Health Check")
        ]
        
        for endpoint, name in endpoints:
            try:
                print(f"Testing {name} ({endpoint})...")
                response = requests.get(f"http://127.0.0.1:8004{endpoint}", timeout=5)
                print(f"  Status: {response.status_code}")
                if response.status_code == 200:
                    print(f"  Success: {name} working!")
                else:
                    print(f"  Error: {name} returned {response.status_code}")
            except Exception as e:
                print(f"  Failed: {name} - {e}")
        
        return True
        
    except Exception as e:
        print(f"Server test failed: {e}")
        return False
    finally:
        # Останавливаем сервер
        print("Stopping server...")
        try:
            server_process.terminate()
            server_process.wait(timeout=5)
        except:
            server_process.kill()

if __name__ == "__main__":
    print("Testing Basic Server Functionality...")
    print("=" * 50)
    
    test_server_basic()
    
    print("=" * 50)