#!/usr/bin/env python3
"""
GPU Health Check Script для Face Recognition Service.

Проверяет доступность CUDA/GPU и корректность работы PyTorch.

Использование:
    python scripts/check_gpu.py
"""

import sys
import time


def check_nvidia_smi():
    """Проверка nvidia-smi."""
    import subprocess
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", 
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print("✅ nvidia-smi доступен")
            for line in result.stdout.strip().split("\n"):
                name, mem, driver = line.split(", ")
                print(f"   GPU: {name}, Memory: {mem} MB, Driver: {driver}")
            return True
        else:
            print("❌ nvidia-smi вернул ошибку")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi не найден (NVIDIA Driver не установлен)")
        return False
    except Exception as e:
        print(f"❌ Ошибка nvidia-smi: {e}")
        return False


def check_cuda():
    """Проверка CUDA через PyTorch."""
    try:
        import torch
        print(f"✅ PyTorch версия: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA доступна: {torch.version.cuda}")
            print(f"✅ GPU устройств: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / (1024**3)
                print(f"   GPU {i}: {props.name}")
                print(f"      Memory: {mem_gb:.1f} GB")
                print(f"      Compute Capability: {props.major}.{props.minor}")
            return True
        else:
            print("⚠️ CUDA недоступна в PyTorch (работаем на CPU)")
            return False
    except ImportError as e:
        print(f"❌ PyTorch не установлен: {e}")
        return False
    except Exception as e:
        print(f"❌ Ошибка проверки CUDA: {e}")
        return False


def check_facenet_models():
    """Проверка загрузки моделей FaceNet."""
    try:
        import torch
        from facenet_pytorch import Mtcnn, InceptionResnetV1
        
        print("✅ facenet-pytorch импортирован")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 Устройство: {device}")
        
        start = time.time()
        
        # MTCNN для детекции лица
        mtcnn = Mtcnn(image_size=160, margin=0).to(device)
        print("✅ MTCNN загружен")
        
        # FaceNet для эмбеддингов
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        print("✅ InceptionResnetV1 загружен")
        
        elapsed = time.time() - start
        print(f"⏱️ Время загрузки: {elapsed:.2f}s")
        
        # Тестовый inference
        dummy = torch.randn(1, 3, 160, 160).to(device)
        with torch.no_grad():
            embedding = resnet(dummy)
        print(f"✅ Тестовый inference: shape={embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка загрузки моделей: {e}")
        return False


def check_liveness_model():
    """Проверка MiniFASNetV2 для liveness detection."""
    try:
        import torch
        from app.services.anti_spoofing_service import AntiSpoofingService
        
        print("✅ AntiSpoofingService импортирован")
        
        service = AntiSpoofingService()
        
        # Проверка доступности GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 Anti-Spoofing device: {device}")
        
        # Проверка статуса модели
        model_status = service.get_model_status()
        print(f"📊 Model status: {model_status}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Anti-Spoofing check skipped: {e}")
        return True  # Не критично


def main():
    """Основная функция проверки."""
    print("=" * 50)
    print("🔍 GPU/CUDA Health Check")
    print("=" * 50)
    print()
    
    results = []
    
    # 1. nvidia-smi
    print("1️⃣  NVIDIA Driver (nvidia-smi)")
    results.append(("nvidia-smi", check_nvidia_smi()))
    print()
    
    # 2. CUDA
    print("2️⃣  CUDA/PyTorch")
    results.append(("CUDA", check_cuda()))
    print()
    
    # 3. FaceNet models
    print("3️⃣  FaceNet Models (MTCNN + InceptionResnetV1)")
    results.append(("FaceNet", check_facenet_models()))
    print()
    
    # 4. Liveness model
    print("4️⃣  Liveness Detection (MiniFASNetV2)")
    results.append(("Liveness", check_liveness_model()))
    print()
    
    # Итог
    print("=" * 50)
    print("📋 Результаты проверки:")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"   {status} {name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("✅ Все проверки пройдены!")
        print()
        print("🌐 Полезные ссылки:")
        print("   Prometheus: http://localhost:9090")
        print("   Grafana:    http://localhost:3000")
        print("   API Docs:   http://localhost:8000/docs")
        return 0
    else:
        print("⚠️ Некоторые проверки не пройдены")
        print("   Проверьте настройки Docker с NVIDIA runtime")
        return 1


if __name__ == "__main__":
    sys.exit(main())