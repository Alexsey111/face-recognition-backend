#!/bin/bash
# =============================================================================
# GPU Health Check Script
# =============================================================================
# Проверяет доступность CUDA/GPU в контейнере Face Recognition Service
#
# Использование в контейнере:
#   ./scripts/check_gpu.sh
#
# Или через docker exec:
#   docker exec face-recognition python scripts/check_gpu.py
# =============================================================================

set -e

echo "========================================"
echo "🔍 GPU/CUDA Health Check"
echo "========================================"
echo ""

# 1. Проверка nvidia-smi
echo "1️⃣  Checking nvidia-smi..."
if command -v nvidia-smi &> /dev/null; then
    echo "   ✅ nvidia-smi found"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version \
        --format=csv,noheader,nounits 2>/dev/null || echo "   ℹ️  nvidia-smi works but query failed"
else
    echo "   ❌ nvidia-smi not found"
fi
echo ""

# 2. Проверка CUDA driver
echo "2️⃣  Checking CUDA driver..."
if nvidia-smi &> /dev/null; then
    CUDA_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    echo "   ✅ CUDA Driver: $CUDA_DRIVER"
else
    echo "   ❌ CUDA driver not accessible"
fi
echo ""

# 3. Проверка CUDA version
echo "3️⃣  Checking CUDA version..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}')
    echo "   ✅ CUDA Toolkit: $CUDA_VERSION"
else
    echo "   ℹ️  nvcc not installed (CUDA Toolkit)"
fi
echo ""

# 4. Проверка через Python/PyTorch
echo "4️⃣  Checking PyTorch CUDA..."
python3 << 'PYTHON_EOF'
import sys
try:
    import torch
    print(f"   ✅ PyTorch version: {torch.__version__}")
    print(f"   ✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✅ CUDA version: {torch.version.cuda}")
        print(f"   ✅ GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"      GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("   ⚠️  CUDA not available in PyTorch")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ PyTorch CUDA check failed: {e}")
    sys.exit(1)
PYTHON_EOF

echo ""

# 5. Проверка FaceNet/PyTorch
echo "5️⃣  Checking FaceNet model loading..."
python3 << 'PYTHON_EOF'
import sys
import time
start = time.time()

try:
    from facenet_pytorch import Mtcnn, InceptionResnetV1
    import torch
    
    print("   ✅ facenet-pytorch imported successfully")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   📱 Device: {device}")
    
    # Load MTCNN (face detection)
    mtcnn = Mtcnn(image_size=160, margin=0).to(device)
    print("   ✅ MTCNN loaded")
    
    # Load FaceNet (embedding)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    print("   ✅ InceptionResnetV1 loaded")
    
    elapsed = time.time() - start
    print(f"   ⏱️  Model load time: {elapsed:.2f}s")
    
    # Test inference
    import numpy as np
    dummy = torch.randn(1, 3, 160, 160).to(device)
    with torch.no_grad():
        output = resnet(dummy)
    print(f"   ✅ Inference test passed (output shape: {output.shape})")
    
except Exception as e:
    print(f"   ❌ FaceNet loading failed: {e}")
    sys.exit(1)
PYTHON_EOF

echo ""
echo "========================================"
echo "✅ GPU Health Check PASSED"
echo "========================================"
echo ""
echo "📋 Quick commands:"
echo "   docker exec face-recognition nvidia-smi"
echo "   docker exec face-recognition python scripts/check_gpu.py"
echo ""
echo "🌐 Useful URLs:"
echo "   Prometheus: http://localhost:9090"
echo "   Grafana:    http://localhost:3000"
echo "   API Docs:   http://localhost:8000/docs"