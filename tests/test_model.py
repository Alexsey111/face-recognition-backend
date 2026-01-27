import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image, ImageDraw  # ✅ Импортируем вместе
import io
from app.services.anti_spoofing_service import AntiSpoofingService
import asyncio

async def test_model():
    service = AntiSpoofingService()
    await service.initialize()
    
    print("=" * 60)
    print("Тест 1: Numpy array (80x80x3)")
    print("=" * 60)
    
    test_image_np = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
    
    try:
        pil_image = Image.fromarray(test_image_np)
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        
        result = await service.check_liveness(img_bytes)
        print(f"✅ Тест пройден!")
        print(f"   liveness_detected: {result.get('liveness_detected')}")
        print(f"   confidence: {result.get('confidence', 0):.4f}")
        print(f"   spoof_score: {result.get('spoof_score', 0):.4f}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    print("\n" + "=" * 60)
    print("Тест 2: Простое лицо")
    print("=" * 60)
    
    test_image = Image.new('RGB', (80, 80), color='beige')
    draw = ImageDraw.Draw(test_image)
    draw.ellipse([10, 10, 70, 70], fill='peachpuff', outline='black')
    draw.ellipse([25, 25, 35, 35], fill='black')
    draw.ellipse([45, 25, 55, 35], fill='black')
    draw.arc([30, 45, 50, 60], 0, 180, fill='black')
    
    img_bytes = io.BytesIO()
    test_image.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    
    try:
        result = await service.check_liveness(img_bytes)
        print(f"✅ Тест пройден!")
        print(f"   liveness_detected: {result.get('liveness_detected')}")
        print(f"   confidence: {result.get('confidence', 0):.4f}")
        print(f"   real_probability: {result.get('real_probability', 0):.4f}")
        print(f"   spoof_probability: {result.get('spoof_probability', 0):.4f}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    print("\n" + "=" * 60)
    print("Тест 3: Batch processing")
    print("=" * 60)
    
    batch_images = []
    for i in range(3):
        img = Image.new('RGB', (80, 80), color=f'rgb({100+i*30}, {150+i*20}, {200+i*10})')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        batch_images.append(img_bytes.getvalue())
    
    try:
        results = await service.batch_check_liveness(batch_images)
        print(f"✅ Batch тест: {len(results)} изображений")
        for i, result in enumerate(results):
            print(f"   [{i+1}] liveness={result.get('liveness_detected')}, conf={result.get('confidence', 0):.4f}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    print("\n" + "=" * 60)
    print("Тест 4: Статистика")
    print("=" * 60)
    
    stats = service.get_stats()
    print(f"✅ Проверок: {stats['total_checks']}")
    print(f"   Real: {stats['real_detected']} ({stats['real_rate']*100:.1f}%)")
    print(f"   Spoof: {stats['spoof_detected']} ({stats['spoof_rate']*100:.1f}%)")
    print(f"   Avg confidence: {stats['avg_confidence']:.4f}")
    print(f"   Avg inference: {stats['avg_inference_time']*1000:.1f}ms")

if __name__ == "__main__":
    asyncio.run(test_model())
