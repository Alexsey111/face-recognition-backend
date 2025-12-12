#!/usr/bin/env python3
"""
Детальный тест для root endpoint.
"""

import subprocess
import time
import sys
import os

def test_root_endpoint():
    """Детальный тест root endpoint."""
    try:
        # Запускаем сервер
        print("Starting server...")
        server_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "app.main:app", 
            "--host", "127.0.0.1", 
            "--port", "8005",
            "--log-level", "info"
        ], cwd=os.getcwd())
        
        # Ждем запуска
        print("Waiting for server to start...")
        time.sleep(5)
        
        # Проверяем root endpoint с деталями
        import requests
        try:
            print("Testing root endpoint...")
            response = requests.get("http://127.0.0.1:8005/", timeout=10)
            print(f"Status Code: {response.status_code}")
            print(f"Headers: {dict(response.headers)}")
            print(f"Response Text: {response.text}")
            
            if response.status_code == 200:
                print("Root endpoint working!")
                print(f"Response JSON: {response.json()}")
                return True
            else:
                print(f"Root endpoint error: {response.status_code}")
                return False
        except Exception as e:
            print(f"Root endpoint failed: {e}")
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
    print("Testing Root Endpoint...")
    print("=" * 40)
    
    test_root_endpoint()
    
    print("=" * 40)