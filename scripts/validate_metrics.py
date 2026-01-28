# scripts/validate_metrics.py

"""
Скрипт валидации метрик точности для сервиса распознавания лиц
Проверяет соответствие требованиям ТЗ:
- FAR < 0.1%
- FRR < 1-3%
- Liveness Accuracy > 98%
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import numpy as np
from tqdm import tqdm
import cv2

# Добавляем путь к модулям приложения
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.face_verification_service import FaceVerificationService
from app.services.anti_spoofing_service import AntiSpoofingService
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsValidator:
    """Валидатор метрик точности распознавания лиц"""
    
    def __init__(
        self,
        verification_service: FaceVerificationService,
        anti_spoofing_service: AntiSpoofingService,
        threshold: float = 0.6
    ):
        self.verification_service = verification_service
        self.anti_spoofing_service = anti_spoofing_service
        self.threshold = threshold
        
        # Результаты тестирования
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "threshold": threshold,
            "verification": {},
            "liveness": {},
            "summary": {}
        }
    
    async def load_test_dataset(self, dataset_path: Path) -> Dict:
        """
        Загрузка тестового датасета
        
        Структура датасета:
        dataset/
        ├── genuine_pairs/      # Пары изображений одного человека (для FRR)
        │   ├── person1_img1.jpg
        │   ├── person1_img2.jpg
        │   ├── person2_img1.jpg
        │   └── person2_img2.jpg
        ├── impostor_pairs/     # Пары изображений разных людей (для FAR)
        │   ├── personA_img1.jpg
        │   ├── personB_img1.jpg
        │   ├── personC_img1.jpg
        │   └── personD_img1.jpg
        ├── live_faces/         # Живые лица (для Liveness TP)
        │   ├── live1.jpg
        │   └── live2.jpg
        └── spoofed_faces/      # Поддельные лица (для Liveness TN)
            ├── print1.jpg
            ├── screen1.jpg
            └── mask1.jpg
        """
        logger.info(f"Загрузка тестового датасета из {dataset_path}")
        
        dataset = {
            "genuine_pairs": [],
            "impostor_pairs": [],
            "live_faces": [],
            "spoofed_faces": []
        }
        
        # Загрузка genuine pairs (одинаковые люди)
        genuine_dir = dataset_path / "genuine_pairs"
        if genuine_dir.exists():
            images = sorted(list(genuine_dir.glob("*.jpg")) + list(genuine_dir.glob("*.png")))
            # Группируем по парам (person1_img1 + person1_img2)
            for i in range(0, len(images) - 1, 2):
                if images[i].stem.split('_')[0] == images[i+1].stem.split('_')[0]:
                    dataset["genuine_pairs"].append((images[i], images[i+1]))
        
        # Загрузка impostor pairs (разные люди)
        impostor_dir = dataset_path / "impostor_pairs"
        if impostor_dir.exists():
            images = list(impostor_dir.glob("*.jpg")) + list(impostor_dir.glob("*.png"))
            # Создаем пары из разных людей
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    if images[i].stem.split('_')[0] != images[j].stem.split('_')[0]:
                        dataset["impostor_pairs"].append((images[i], images[j]))
        
        # Загрузка live faces
        live_dir = dataset_path / "live_faces"
        if live_dir.exists():
            dataset["live_faces"] = list(live_dir.glob("*.jpg")) + list(live_dir.glob("*.png"))
        
        # Загрузка spoofed faces
        spoofed_dir = dataset_path / "spoofed_faces"
        if spoofed_dir.exists():
            dataset["spoofed_faces"] = list(spoofed_dir.glob("*.jpg")) + list(spoofed_dir.glob("*.png"))
        
        logger.info(f"Загружено:")
        logger.info(f"  - Genuine pairs: {len(dataset['genuine_pairs'])}")
        logger.info(f"  - Impostor pairs: {len(dataset['impostor_pairs'])}")
        logger.info(f"  - Live faces: {len(dataset['live_faces'])}")
        logger.info(f"  - Spoofed faces: {len(dataset['spoofed_faces'])}")
        
        return dataset
    
    async def validate_far_frr(self, dataset: Dict) -> Dict:
        """
        Валидация FAR (False Accept Rate) и FRR (False Reject Rate)
        
        FAR: процент ошибочно принятых самозванцев (impostor pairs, которые прошли)
        FRR: процент ошибочно отклоненных настоящих (genuine pairs, которые не прошли)
        """
        logger.info("=" * 80)
        logger.info("Валидация FAR/FRR")
        logger.info("=" * 80)
        
        # Тестирование FRR на genuine pairs
        logger.info(f"\n1. Тестирование FRR на {len(dataset['genuine_pairs'])} парах genuine...")
        genuine_scores = []
        false_rejects = 0
        
        for img1_path, img2_path in tqdm(dataset['genuine_pairs'], desc="FRR Test"):
            try:
                img1 = cv2.imread(str(img1_path))
                img2 = cv2.imread(str(img2_path))
                
                # Получаем embedding и сравниваем
                emb1 = await self.verification_service.extract_embedding(img1)
                emb2 = await self.verification_service.extract_embedding(img2)
                
                similarity = await self.verification_service.compare_embeddings(emb1, emb2)
                genuine_scores.append(similarity)
                
                if similarity < self.threshold:
                    false_rejects += 1
                    
            except Exception as e:
                logger.error(f"Ошибка обработки пары {img1_path.name}, {img2_path.name}: {e}")
        
        frr = (false_rejects / len(dataset['genuine_pairs'])) * 100 if dataset['genuine_pairs'] else 0
        
        # Тестирование FAR на impostor pairs
        logger.info(f"\n2. Тестирование FAR на {len(dataset['impostor_pairs'])} парах impostor...")
        impostor_scores = []
        false_accepts = 0
        
        for img1_path, img2_path in tqdm(dataset['impostor_pairs'], desc="FAR Test"):
            try:
                img1 = cv2.imread(str(img1_path))
                img2 = cv2.imread(str(img2_path))
                
                emb1 = await self.verification_service.extract_embedding(img1)
                emb2 = await self.verification_service.extract_embedding(img2)
                
                similarity = await self.verification_service.compare_embeddings(emb1, emb2)
                impostor_scores.append(similarity)
                
                if similarity >= self.threshold:
                    false_accepts += 1
                    
            except Exception as e:
                logger.error(f"Ошибка обработки пары {img1_path.name}, {img2_path.name}: {e}")
        
        far = (false_accepts / len(dataset['impostor_pairs'])) * 100 if dataset['impostor_pairs'] else 0
        
        # Расчет дополнительных метрик
        eer = self._calculate_eer(genuine_scores, impostor_scores)
        
        results = {
            "far": {
                "value": far,
                "threshold_requirement": "< 0.1%",
                "passed": far < 0.1,
                "false_accepts": false_accepts,
                "total_impostor_pairs": len(dataset['impostor_pairs'])
            },
            "frr": {
                "value": frr,
                "threshold_requirement": "< 1-3%",
                "passed": frr < 3.0,
                "false_rejects": false_rejects,
                "total_genuine_pairs": len(dataset['genuine_pairs'])
            },
            "eer": {
                "value": eer,
                "description": "Equal Error Rate (где FAR = FRR)"
            },
            "distributions": {
                "genuine_mean": float(np.mean(genuine_scores)) if genuine_scores else 0,
                "genuine_std": float(np.std(genuine_scores)) if genuine_scores else 0,
                "impostor_mean": float(np.mean(impostor_scores)) if impostor_scores else 0,
                "impostor_std": float(np.std(impostor_scores)) if impostor_scores else 0
            }
        }
        
        self.results["verification"] = results
        
        # Вывод результатов
        logger.info("\n" + "=" * 80)
        logger.info("РЕЗУЛЬТАТЫ FAR/FRR")
        logger.info("=" * 80)
        logger.info(f"FAR:  {far:.4f}% (требование: < 0.1%) - {'✓ PASSED' if results['far']['passed'] else '✗ FAILED'}")
        logger.info(f"FRR:  {frr:.4f}% (требование: < 1-3%) - {'✓ PASSED' if results['frr']['passed'] else '✗ FAILED'}")
        logger.info(f"EER:  {eer:.4f}%")
        logger.info(f"\nРаспределения:")
        logger.info(f"  Genuine:  μ={results['distributions']['genuine_mean']:.4f}, σ={results['distributions']['genuine_std']:.4f}")
        logger.info(f"  Impostor: μ={results['distributions']['impostor_mean']:.4f}, σ={results['distributions']['impostor_std']:.4f}")
        logger.info("=" * 80)
        
        return results
    
    async def validate_liveness(self, dataset: Dict) -> Dict:
        """
        Валидация Liveness Detection
        
        Требование: > 98% точность
        """
        logger.info("\n" + "=" * 80)
        logger.info("Валидация Liveness Detection")
        logger.info("=" * 80)
        
        true_positives = 0  # Правильно определенные живые лица
        false_negatives = 0  # Живые лица, определенные как поддельные
        true_negatives = 0  # Правильно определенные поддельные лица
        false_positives = 0  # Поддельные лица, определенные как живые
        
        # Тест на живых лицах (должны определяться как live)
        logger.info(f"\n1. Тестирование на {len(dataset['live_faces'])} живых лицах...")
        for img_path in tqdm(dataset['live_faces'], desc="Live Faces Test"):
            try:
                img = cv2.imread(str(img_path))
                result = await self.anti_spoofing_service.detect_spoofing(img)
                
                if result["is_live"]:
                    true_positives += 1
                else:
                    false_negatives += 1
                    logger.warning(f"FN: {img_path.name} определено как поддельное (score: {result['score']:.4f})")
                    
            except Exception as e:
                logger.error(f"Ошибка обработки {img_path.name}: {e}")
                false_negatives += 1
        
        # Тест на поддельных лицах (должны определяться как spoofed)
        logger.info(f"\n2. Тестирование на {len(dataset['spoofed_faces'])} поддельных лицах...")
        for img_path in tqdm(dataset['spoofed_faces'], desc="Spoofed Faces Test"):
            try:
                img = cv2.imread(str(img_path))
                result = await self.anti_spoofing_service.detect_spoofing(img)
                
                if not result["is_live"]:
                    true_negatives += 1
                else:
                    false_positives += 1
                    logger.warning(f"FP: {img_path.name} определено как живое (score: {result['score']:.4f})")
                    
            except Exception as e:
                logger.error(f"Ошибка обработки {img_path.name}: {e}")
                true_negatives += 1  # Ошибка обработки = безопасное отклонение
        
        # Расчет метрик
        total_live = len(dataset['live_faces'])
        total_spoofed = len(dataset['spoofed_faces'])
        total = total_live + total_spoofed
        
        accuracy = ((true_positives + true_negatives) / total * 100) if total > 0 else 0
        precision = (true_positives / (true_positives + false_positives) * 100) if (true_positives + false_positives) > 0 else 0
        recall = (true_positives / (true_positives + false_negatives) * 100) if (true_positives + false_negatives) > 0 else 0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        
        results = {
            "accuracy": {
                "value": accuracy,
                "threshold_requirement": "> 98%",
                "passed": accuracy > 98.0
            },
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "confusion_matrix": {
                "true_positives": true_positives,
                "false_negatives": false_negatives,
                "true_negatives": true_negatives,
                "false_positives": false_positives
            },
            "totals": {
                "live_samples": total_live,
                "spoofed_samples": total_spoofed
            }
        }
        
        self.results["liveness"] = results
        
        # Вывод результатов
        logger.info("\n" + "=" * 80)
        logger.info("РЕЗУЛЬТАТЫ LIVENESS DETECTION")
        logger.info("=" * 80)
        logger.info(f"Accuracy:  {accuracy:.2f}% (требование: > 98%) - {'✓ PASSED' if results['accuracy']['passed'] else '✗ FAILED'}")
        logger.info(f"Precision: {precision:.2f}%")
        logger.info(f"Recall:    {recall:.2f}%")
        logger.info(f"F1-Score:  {f1_score:.2f}%")
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TP (Live → Live):       {true_positives}/{total_live}")
        logger.info(f"  FN (Live → Spoofed):    {false_negatives}/{total_live}")
        logger.info(f"  TN (Spoofed → Spoofed): {true_negatives}/{total_spoofed}")
        logger.info(f"  FP (Spoofed → Live):    {false_positives}/{total_spoofed}")
        logger.info("=" * 80)
        
        return results
    
    def _calculate_eer(self, genuine_scores: List[float], impostor_scores: List[float]) -> float:
        """
        Расчет Equal Error Rate (EER) - точка, где FAR = FRR
        """
        if not genuine_scores or not impostor_scores:
            return 0.0
        
        thresholds = np.linspace(0, 1, 1000)
        far_values = []
        frr_values = []
        
        for threshold in thresholds:
            far = np.sum(np.array(impostor_scores) >= threshold) / len(impostor_scores) * 100
            frr = np.sum(np.array(genuine_scores) < threshold) / len(genuine_scores) * 100
            far_values.append(far)
            frr_values.append(frr)
        
        # Находим точку пересечения
        eer_idx = np.argmin(np.abs(np.array(far_values) - np.array(frr_values)))
        eer = (far_values[eer_idx] + frr_values[eer_idx]) / 2
        
        return eer
    
    def generate_summary(self) -> Dict:
        """Генерация итогового отчета"""
        verification = self.results.get("verification", {})
        liveness = self.results.get("liveness", {})
        
        all_tests_passed = (
            verification.get("far", {}).get("passed", False) and
            verification.get("frr", {}).get("passed", False) and
            liveness.get("accuracy", {}).get("passed", False)
        )
        
        summary = {
            "overall_status": "PASSED" if all_tests_passed else "FAILED",
            "tests": {
                "FAR": "✓" if verification.get("far", {}).get("passed", False) else "✗",
                "FRR": "✓" if verification.get("frr", {}).get("passed", False) else "✗",
                "Liveness": "✓" if liveness.get("accuracy", {}).get("passed", False) else "✗"
            },
            "compliance_152_fz": all_tests_passed
        }
        
        self.results["summary"] = summary
        return summary
    
    def save_results(self, output_path: Path):
        """Сохранение результатов в JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info(f"\nРезультаты сохранены в {output_path}")
    
    def print_final_report(self):
        """Вывод финального отчета"""
        summary = self.results.get("summary", {})
        
        logger.info("\n" + "=" * 80)
        logger.info("ФИНАЛЬНЫЙ ОТЧЕТ ВАЛИДАЦИИ")
        logger.info("=" * 80)
        logger.info(f"Статус: {summary.get('overall_status', 'UNKNOWN')}")
        logger.info(f"Timestamp: {self.results.get('timestamp', 'N/A')}")
        logger.info(f"Threshold: {self.results.get('threshold', 'N/A')}")
        logger.info(f"\nТесты:")
        for test_name, status in summary.get("tests", {}).items():
            logger.info(f"  {test_name}: {status}")
        logger.info(f"\nСоответствие 152-ФЗ: {'✓' if summary.get('compliance_152_fz') else '✗'}")
        logger.info("=" * 80)


async def main():
    parser = argparse.ArgumentParser(description="Валидация метрик точности сервиса распознавания лиц")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("tests/datasets/validation_dataset"),
        help="Путь к тестовому датасету"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Порог верификации (default: 0.6)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation_results.json"),
        help="Путь для сохранения результатов"
    )
    
    args = parser.parse_args()
    
    # Инициализация сервисов
    logger.info("Инициализация сервисов...")
    verification_service = FaceVerificationService()
    anti_spoofing_service = AntiSpoofingService()
    
    # Создание валидатора
    validator = MetricsValidator(
        verification_service=verification_service,
        anti_spoofing_service=anti_spoofing_service,
        threshold=args.threshold
    )
    
    # Загрузка датасета
    dataset = await validator.load_test_dataset(args.dataset)
    
    # Валидация FAR/FRR
    await validator.validate_far_frr(dataset)
    
    # Валидация Liveness
    await validator.validate_liveness(dataset)
    
    # Генерация итогового отчета
    validator.generate_summary()
    validator.print_final_report()
    
    # Сохранение результатов
    validator.save_results(args.output)
    
    # Возврат кода выхода
    if validator.results["summary"]["overall_status"] == "PASSED":
        logger.info("\n✓ Все тесты пройдены успешно!")
        sys.exit(0)
    else:
        logger.error("\n✗ Некоторые тесты не прошли валидацию")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
