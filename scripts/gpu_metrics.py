"""
GPU Metrics Exporter для Prometheus.

Собирает метрики GPU и экспонирует их для Prometheus.

Использование:
    from scripts.gpu_metrics import get_gpu_metrics
    metrics = get_gpu_metrics()
"""

import time
from typing import Dict, Any, Optional
from prometheus_client import Gauge, Info, CollectorRegistry, generate_latest


class GPUMetricsCollector:
    """Collector для метрик GPU."""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self._setup_gauges()
    
    def _setup_gauges(self):
        """Настройка Prometheus gauge метрик."""
        # GPU Memory
        self.gpu_memory_total = Gauge(
            'gpu_memory_total_bytes',
            'Total GPU memory in bytes',
            ['gpu_id'],
            registry=self.registry
        )
        self.gpu_memory_used = Gauge(
            'gpu_memory_used_bytes',
            'Used GPU memory in bytes',
            ['gpu_id'],
            registry=self.registry
        )
        self.gpu_memory_free = Gauge(
            'gpu_memory_free_bytes',
            'Free GPU memory in bytes',
            ['gpu_id'],
            registry=self.registry
        )
        
        # GPU Utilization
        self.gpu_utilization = Gauge(
            'gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id'],
            registry=self.registry
        )
        self.gpu_memory_utilization = Gauge(
            'gpu_memory_utilization_percent',
            'GPU memory utilization percentage',
            ['gpu_id'],
            registry=self.registry
        )
        
        # GPU Temperature
        self.gpu_temperature = Gauge(
            'gpu_temperature_celsius',
            'GPU temperature in Celsius',
            ['gpu_id'],
            registry=self.registry
        )
        
        # GPU Info
        self.gpu_info = Info(
            'gpu_info',
            'GPU information',
            ['gpu_id', 'name', 'driver_version', 'cuda_version'],
            registry=self.registry
        )
        
        # Service metrics
        self.service_gpu_enabled = Gauge(
            'service_gpu_enabled',
            'Whether GPU is enabled for the service',
            registry=self.registry
        )
        self.service_device = Gauge(
            'service_device',
            'Device used by service (0=cpu, 1=gpu)',
            registry=self.registry
        )
    
    def collect_nvidia_smi(self) -> Optional[Dict[str, Any]]:
        """Сбор метрик через nvidia-smi."""
        import subprocess
        import re
        
        try:
            # Получить информацию о GPU
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,"
                 "utilization.gpu,utilization.memory,temperature.gpu,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0:
                return None
            
            gpus = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 9:
                    gpu = {
                        "id": parts[0],
                        "name": parts[1],
                        "memory_total_mb": int(parts[2]),
                        "memory_used_mb": int(parts[3]),
                        "memory_free_mb": int(parts[4]),
                        "utilization_percent": int(parts[5]),
                        "memory_utilization_percent": int(parts[6]),
                        "temperature_celsius": int(parts[7]),
                        "driver_version": parts[8],
                    }
                    gpus.append(gpu)
            
            return gpus if gpus else None
            
        except Exception as e:
            print(f"nvidia-smi error: {e}")
            return None
    
    def collect_torch(self) -> Dict[str, Any]:
        """Сбор метрик через PyTorch."""
        try:
            import torch
            
            info = {
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            }
            
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    info[f"gpu_{i}_name"] = props.name
                    info[f"gpu_{i}_memory_total"] = props.total_memory
                    info[f"gpu_{i}_compute_cap"] = f"{props.major}.{props.minor}"
            
            return info
            
        except Exception as e:
            return {"error": str(e)}
    
    def update(self):
        """Обновление всех метрик."""
        # PyTorch metrics
        torch_info = self.collect_torch()
        
        self.service_gpu_enabled.set(
            1 if torch_info.get("cuda_available") else 0
        )
        self.service_device.set(
            1 if torch_info.get("cuda_available") else 0
        )
        
        # nvidia-smi metrics
        gpus = self.collect_nvidia_smi()
        
        if gpus:
            for gpu in gpus:
                gpu_id = gpu["id"]
                
                self.gpu_memory_total.labels(gpu_id=gpu_id).set(
                    gpu["memory_total_mb"] * 1024 * 1024
                )
                self.gpu_memory_used.labels(gpu_id=gpu_id).set(
                    gpu["memory_used_mb"] * 1024 * 1024
                )
                self.gpu_memory_free.labels(gpu_id=gpu_id).set(
                    gpu["memory_free_mb"] * 1024 * 1024
                )
                self.gpu_utilization.labels(gpu_id=gpu_id).set(
                    gpu["utilization_percent"]
                )
                self.gpu_memory_utilization.labels(gpu_id=gpu_id).set(
                    gpu["memory_utilization_percent"]
                )
                self.gpu_temperature.labels(gpu_id=gpu_id).set(
                    gpu["temperature_celsius"]
                )
                
                self.gpu_info.labels(
                    gpu_id=gpu_id,
                    name=gpu["name"],
                    driver_version=gpu["driver_version"],
                    cuda_version=torch_info.get("cuda_version", "N/A")
                ).info({})
    
    def get_metrics(self) -> bytes:
        """Получение метрик в формате Prometheus."""
        self.update()
        return generate_latest(self.registry)


# Singleton instance
_metrics_collector: Optional[GPUMetricsCollector] = None


def get_metrics_collector() -> GPUMetricsCollector:
    """Получение singleton collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = GPUMetricsCollector()
    return _metrics_collector


def get_gpu_metrics() -> bytes:
    """Получение метрик GPU для Prometheus."""
    return get_metrics_collector().get_metrics()


def get_gpu_summary() -> Dict[str, Any]:
    """Получение краткой сводки GPU."""
    collector = get_metrics_collector()
    torch_info = collector.collect_torch()
    gpus = collector.collect_nvidia_smi()
    
    return {
        "cuda_available": torch_info.get("cuda_available", False),
        "device_count": torch_info.get("device_count", 0),
        "cuda_version": torch_info.get("cuda_version"),
        "gpus": gpus if gpus else [],
    }


if __name__ == "__main__":
    # Тестовый вывод
    print("GPU Metrics Summary:")
    print("=" * 50)
    
    summary = get_gpu_summary()
    
    print(f"CUDA Available: {summary['cuda_available']}")
    print(f"Device Count: {summary['device_count']}")
    print(f"CUDA Version: {summary['cuda_version']}")
    
    for gpu in summary['gpus']:
        print(f"\nGPU {gpu['id']}: {gpu['name']}")
        print(f"  Memory: {gpu['memory_used_mb']}/{gpu['memory_total_mb']} MB")
        print(f"  Utilization: {gpu['utilization_percent']}%")
        print(f"  Temperature: {gpu['temperature_celsius']}°C")