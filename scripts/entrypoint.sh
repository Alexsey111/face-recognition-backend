#!/bin/bash
# =============================================================================
# Entrypoint Script для Face Recognition Service
# =============================================================================
# Автоматически определяет доступность GPU и настраивает окружение
#
# Environment variables:
#   LOCAL_ML_ENABLE_CUDA=true/false (auto-detect by default)
#   LOCAL_ML_DEVICE=cuda/cpu (auto-detect by default)
# =============================================================================

set -e

echo "========================================"
echo "🚀 Face Recognition Service Startup"
echo "========================================"
echo ""

# 1. Определение окружения
echo "1️⃣  Environment Detection"
ENVIRONMENT=${ENVIRONMENT:-production}
echo "   ENVIRONMENT: $ENVIRONMENT"

# 2. Проверка GPU
echo ""
echo "2️⃣  GPU Detection"

# Проверка nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "Unknown")
    echo "   ✅ NVIDIA GPU detected: $GPU_INFO"
    GPU_AVAILABLE=true
else
    echo "   ⚠️  NVIDIA GPU not detected (nvidia-smi not found)"
    GPU_AVAILABLE=false
fi

# Проверка CUDA через PyTorch
PYTHON_GPU_CHECK=$(python3 -c "
import torch
if torch.cuda.is_available():
    print(f'cuda:{torch.cuda.current_device()}', end='')
else:
    print('cpu', end='')
" 2>/dev/null || echo "unknown")

echo "   📱 PyTorch device: $PYTHON_GPU_CHECK"

# 3. Настройка переменных окружения для ML
echo ""
echo "3️⃣  ML Environment Configuration"

# Автоматическое определение CUDA
if [ "$LOCAL_ML_ENABLE_CUDA" != "false" ] && [ "$GPU_AVAILABLE" = "true" ]; then
    export LOCAL_ML_ENABLE_CUDA=true
    export LOCAL_ML_DEVICE=cuda
    export TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6;8.9;9.0"  # GPU architectures
    echo "   ✅ CUDA enabled"
else
    export LOCAL_ML_ENABLE_CUDA=false
    export LOCAL_ML_DEVICE=cpu
    echo "   ℹ️  Using CPU"
fi

echo "   LOCAL_ML_ENABLE_CUDA: $LOCAL_ML_ENABLE_CUDA"
echo "   LOCAL_ML_DEVICE: $LOCAL_ML_DEVICE"

# 4. Проверка моделей
echo ""
echo "4️⃣  Model Loading Check"

# Проверка наличия моделей
if [ -d "/app/models" ]; then
    MODEL_COUNT=$(find /app/models -name "*.pth" -o -name "*.pt" 2>/dev/null | wc -l)
    echo "   📦 Models found: $MODEL_COUNT"
else
    echo "   ℹ️  Models directory not found (will download on first run)"
fi

# 5. Запуск проверки GPU (опционально)
if [ "$GPU_HEALTH_CHECK" = "true" ]; then
    echo ""
    echo "5️⃣  Running GPU Health Check..."
    python3 /app/scripts/check_gpu.py || true
fi

# 6. Вывод информации о запуске
echo ""
echo "========================================"
echo "🌐 Starting Service"
echo "========================================"
echo ""
echo "   API Docs: http://localhost:8000/docs"
echo "   Health:   http://localhost:8000/health"
echo ""

# 7. Запуск uvicorn
exec uvicorn app.main:app "$@"