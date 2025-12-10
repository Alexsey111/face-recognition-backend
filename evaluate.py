#!/usr/bin/env python3
"""
Face Recognition Model Evaluation Script

Полнофункциональный скрипт для оценки качества системы распознавания лиц.
Вычисляет метрики FAR (False Acceptance Rate), FRR (False Rejection Rate), 
ROC кривые, EER (Equal Error Rate) и генерирует отчеты в JSON/CSV форматах.

Usage:
    python evaluate.py --data-dir ./test_data --output ./results
    python evaluate.py --threshold 0.6 --generate-plots
"""

import argparse
import json
import csv
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Результаты оценки модели."""
    threshold: float
    far: float  # False Acceptance Rate
    frr: float  # False Rejection Rate
    tpr: float  # True Positive Rate (1 - FRR)
    fpr: float  # False Positive Rate (FAR)
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_pairs: int
    genuine_pairs: int
    impostor_pairs: int
    correct_predictions: int
    false_acceptances: int
    false_rejections: int

@dataclass
class ROCMetrics:
    """ROC метрики."""
    thresholds: List[float]
    tprs: List[float]
    fprs: List[float]
    auc_score: float
    eer: float
    eer_threshold: float

class FaceRecognitionEvaluator:
    """Оценщик системы распознавания лиц."""
    
    def __init__(self, threshold: float = 0.5):
        """
        Инициализация оценщика.
        
        Args:
            threshold: Порог принятия решения (0-1)
        """
        self.threshold = threshold
        self.results: List[EvaluationResult] = []
        self.roc_metrics: Optional[ROCMetrics] = None
        
    def evaluate_threshold(self, 
                          similarity_scores: np.ndarray, 
                          true_labels: np.ndarray,
                          threshold: float) -> EvaluationResult:
        """
        Оценка модели на заданном пороге.
        
        Args:
            similarity_scores: Массив оценок схожести (0-1)
            true_labels: Истинные метки (1 - genuine pair, 0 - impostor pair)
            threshold: Порог принятия решения
            
        Returns:
            EvaluationResult: Результаты оценки
        """
        # Предсказания на основе порога
        predictions = (similarity_scores >= threshold).astype(int)
        
        # Подсчет метрик
        total_pairs = len(similarity_scores)
        genuine_pairs = np.sum(true_labels == 1)
        impostor_pairs = np.sum(true_labels == 0)
        
        # TP, TN, FP, FN
        tp = np.sum((predictions == 1) & (true_labels == 1))  # True Positive
        tn = np.sum((predictions == 0) & (true_labels == 0))  # True Negative
        fp = np.sum((predictions == 1) & (true_labels == 0))  # False Positive
        fn = np.sum((predictions == 0) & (true_labels == 1))  # False Negative
        
        # FAR (False Acceptance Rate) = FP / (FP + TN) = FP / impostor_pairs
        far = fp / impostor_pairs if impostor_pairs > 0 else 0
        
        # FRR (False Rejection Rate) = FN / (FN + TP) = FN / genuine_pairs
        frr = fn / genuine_pairs if genuine_pairs > 0 else 0
        
        # TPR (True Positive Rate) = 1 - FRR
        tpr = tp / genuine_pairs if genuine_pairs > 0 else 0
        
        # FPR (False Positive Rate) = FAR
        fpr = far
        
        # Accuracy
        accuracy = (tp + tn) / total_pairs if total_pairs > 0 else 0
        
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall
        recall = tpr
        
        # F1-Score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return EvaluationResult(
            threshold=threshold,
            far=far,
            frr=frr,
            tpr=tpr,
            fpr=fpr,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            total_pairs=total_pairs,
            genuine_pairs=genuine_pairs,
            impostor_pairs=impostor_pairs,
            correct_predictions=tp + tn,
            false_acceptances=fp,
            false_rejections=fn
        )
    
    def compute_roc_metrics(self, 
                           similarity_scores: np.ndarray, 
                           true_labels: np.ndarray) -> ROCMetrics:
        """
        Вычисление ROC метрик.
        
        Args:
            similarity_scores: Массив оценок схожести
            true_labels: Истинные метки
            
        Returns:
            ROCMetrics: ROC метрики
        """
        # Вычисление ROC кривой
        fpr, tpr, thresholds = roc_curve(true_labels, similarity_scores)
        auc_score = auc(fpr, tpr)
        
        # Вычисление EER (Equal Error Rate)
        # EER = порог, где FAR = FRR
        eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
        eer = fpr[eer_idx]
        eer_threshold = thresholds[eer_idx]
        
        return ROCMetrics(
            thresholds=thresholds.tolist(),
            tprs=tpr.tolist(),
            fprs=fpr.tolist(),
            auc_score=auc_score,
            eer=eer,
            eer_threshold=eer_threshold
        )
    
    def evaluate_multiple_thresholds(self, 
                                   similarity_scores: np.ndarray, 
                                   true_labels: np.ndarray,
                                   threshold_range: Tuple[float, float] = (0.1, 0.9),
                                   num_points: int = 100) -> List[EvaluationResult]:
        """
        Оценка модели на множестве порогов.
        
        Args:
            similarity_scores: Массив оценок схожести
            true_labels: Истинные метки
            threshold_range: Диапазон порогов (min, max)
            num_points: Количество точек для анализа
            
        Returns:
            List[EvaluationResult]: Список результатов для всех порогов
        """
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_points)
        results = []
        
        for threshold in thresholds:
            result = self.evaluate_threshold(similarity_scores, true_labels, threshold)
            results.append(result)
        
        self.results = results
        return results
    
    def find_optimal_threshold(self, 
                             similarity_scores: np.ndarray, 
                             true_labels: np.ndarray,
                             metric: str = 'f1_score') -> EvaluationResult:
        """
        Поиск оптимального порога по выбранной метрике.
        
        Args:
            similarity_scores: Массив оценок схожести
            true_labels: Истинные метки
            metric: Метрика для оптимизации ('f1_score', 'accuracy', 'precision', 'recall')
            
        Returns:
            EvaluationResult: Оптимальный результат
        """
        results = self.evaluate_multiple_thresholds(similarity_scores, true_labels)
        
        if not results:
            raise ValueError("No results found")
        
        # Поиск лучшего порога
        best_result = max(results, key=lambda x: getattr(x, metric))
        
        # Установка оптимального порога
        self.threshold = best_result.threshold
        
        return best_result
    
    def generate_synthetic_data(self, 
                              num_genuine: int = 1000,
                              num_impostor: int = 1000,
                              seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерация синтетических данных для тестирования.
        
        Args:
            num_genuine: Количество genuine pairs
            num_impostor: Количество impostor pairs
            seed: Seed для воспроизводимости
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (similarity_scores, true_labels)
        """
        np.random.seed(seed)
        
        # Genuine pairs: высокие оценки схожести (0.6-1.0)
        genuine_scores = np.random.beta(2, 1, num_genuine) * 0.4 + 0.6
        
        # Impostor pairs: низкие оценки схожести (0.0-0.4)
        impostor_scores = np.random.beta(1, 2, num_impostor) * 0.4
        
        # Объединение данных
        similarity_scores = np.concatenate([genuine_scores, impostor_scores])
        true_labels = np.concatenate([np.ones(num_genuine), np.zeros(num_impostor)])
        
        # Перемешивание
        indices = np.random.permutation(len(similarity_scores))
        
        return similarity_scores[indices], true_labels[indices]
    
    def save_results_json(self, results: List[EvaluationResult], output_path: str):
        """
        Сохранение результатов в JSON файл.
        
        Args:
            results: Список результатов оценки
            output_path: Путь для сохранения
        """
        def convert_numpy_types(obj):
            """Конвертация numpy типов в Python типы для JSON сериализации."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        output_data = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'total_evaluations': len(results),
            'results': [convert_numpy_types(asdict(result)) for result in results]
        }
        
        if self.roc_metrics:
            roc_data = asdict(self.roc_metrics)
            output_data['roc_metrics'] = convert_numpy_types(roc_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
    
    def save_results_csv(self, results: List[EvaluationResult], output_path: str):
        """
        Сохранение результатов в CSV файл.
        
        Args:
            results: Список результатов оценки
            output_path: Путь для сохранения
        """
        df = pd.DataFrame([asdict(result) for result in results])
        df.to_csv(output_path, index=False)
        
        logger.info(f"Results saved to {output_path}")
    
    def plot_roc_curve(self, output_path: str, figsize: Tuple[int, int] = (10, 8)):
        """
        Построение ROC кривой.
        
        Args:
            output_path: Путь для сохранения графика
            figsize: Размер фигуры
        """
        if not self.roc_metrics:
            raise ValueError("ROC metrics not computed. Run evaluate_multiple_thresholds first.")
        
        plt.figure(figsize=figsize)
        plt.plot(self.roc_metrics.fprs, self.roc_metrics.tprs, 
                label=f'ROC Curve (AUC = {self.roc_metrics.auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate (FAR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve for Face Recognition System')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        # Добавление EER точки
        eer_point = (self.roc_metrics.eer, 1 - self.roc_metrics.eer)
        plt.plot(eer_point[0], eer_point[1], 'ro', markersize=8, 
                label=f'EER = {self.roc_metrics.eer:.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve saved to {output_path}")
    
    def plot_metrics_distribution(self, results: List[EvaluationResult], output_path: str):
        """
        Построение графиков распределения метрик.
        
        Args:
            results: Список результатов
            output_path: Путь для сохранения
        """
        df = pd.DataFrame([asdict(result) for result in results])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # FAR vs FRR
        axes[0, 0].plot(df['threshold'], df['far'], label='FAR', linewidth=2)
        axes[0, 0].plot(df['threshold'], df['frr'], label='FRR', linewidth=2)
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Rate')
        axes[0, 0].set_title('FAR vs FRR')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(df['threshold'], df['accuracy'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy vs Threshold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision, Recall, F1-Score
        axes[1, 0].plot(df['threshold'], df['precision'], label='Precision', linewidth=2)
        axes[1, 0].plot(df['threshold'], df['recall'], label='Recall', linewidth=2)
        axes[1, 0].plot(df['threshold'], df['f1_score'], label='F1-Score', linewidth=2)
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Precision, Recall, F1-Score vs Threshold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # EER Point
        best_result = max(results, key=lambda x: x.f1_score)
        axes[1, 1].scatter(best_result.far, best_result.tpr, color='red', s=100, 
                          label=f'Best F1 (Threshold={best_result.threshold:.3f})')
        axes[1, 1].plot(df['far'], df['tpr'], 'b-', alpha=0.7)
        axes[1, 1].set_xlabel('FAR')
        axes[1, 1].set_ylabel('TPR')
        axes[1, 1].set_title('TAR vs FAR (DET Curve)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Metrics distribution plots saved to {output_path}")
    
    def print_summary(self, results: List[EvaluationResult]):
        """
        Вывод краткого отчета в консоль.
        
        Args:
            results: Список результатов
        """
        if not results:
            print("No results to display")
            return
        
        print("\n" + "="*80)
        print("FACE RECOGNITION SYSTEM EVALUATION REPORT")
        print("="*80)
        
        # Общая информация
        first_result = results[0]
        print(f"Total pairs evaluated: {first_result.total_pairs:,}")
        print(f"Genuine pairs: {first_result.genuine_pairs:,}")
        print(f"Impostor pairs: {first_result.impostor_pairs:,}")
        
        # Лучшие метрики
        best_f1 = max(results, key=lambda x: x.f1_score)
        best_accuracy = max(results, key=lambda x: x.accuracy)
        best_precision = max(results, key=lambda x: x.precision)
        best_recall = max(results, key=lambda x: x.recall)
        
        print(f"\nBEST F1-SCORE: {best_f1.f1_score:.4f} (threshold: {best_f1.threshold:.3f})")
        print(f"BEST ACCURACY: {best_accuracy.accuracy:.4f} (threshold: {best_accuracy.threshold:.3f})")
        print(f"BEST PRECISION: {best_precision.precision:.4f} (threshold: {best_precision.threshold:.3f})")
        print(f"BEST RECALL: {best_recall.recall:.4f} (threshold: {best_recall.threshold:.3f})")
        
        # EER
        if self.roc_metrics:
            print(f"\nEQUAL ERROR RATE (EER): {self.roc_metrics.eer:.4f}")
            print(f"EER Threshold: {self.roc_metrics.eer_threshold:.3f}")
            print(f"AUC Score: {self.roc_metrics.auc_score:.4f}")
        
        print("\n" + "="*80)


def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(description='Face Recognition Model Evaluation')
    parser.add_argument('--data-dir', type=str, help='Directory with evaluation data')
    parser.add_argument('--output', type=str, default='./evaluation_results', 
                       help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Threshold for evaluation')
    parser.add_argument('--threshold-range', type=str, default='0.1,0.9',
                       help='Range for threshold analysis (min,max)')
    parser.add_argument('--num-points', type=int, default=100,
                       help='Number of threshold points for analysis')
    parser.add_argument('--generate-plots', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data for testing')
    parser.add_argument('--num-genuine', type=int, default=1000,
                       help='Number of genuine pairs for synthetic data')
    parser.add_argument('--num-impostor', type=int, default=1000,
                       help='Number of impostor pairs for synthetic data')
    
    args = parser.parse_args()
    
    # Создание выходной директории
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Инициализация оценщика
    evaluator = FaceRecognitionEvaluator(threshold=args.threshold)
    
    # Загрузка данных
    if args.synthetic:
        logger.info("Using synthetic data for evaluation")
        similarity_scores, true_labels = evaluator.generate_synthetic_data(
            num_genuine=args.num_genuine,
            num_impostor=args.num_impostor
        )
    elif args.data_dir:
        logger.info(f"Loading data from {args.data_dir}")
        # TODO: Реализовать загрузку реальных данных
        raise NotImplementedError("Real data loading not implemented yet")
    else:
        logger.error("Either --synthetic or --data-dir must be specified")
        sys.exit(1)
    
    # Оценка на множестве порогов
    threshold_range = tuple(map(float, args.threshold_range.split(',')))
    results = evaluator.evaluate_multiple_thresholds(
        similarity_scores, true_labels, 
        threshold_range=threshold_range,
        num_points=args.num_points
    )
    
    # Вычисление ROC метрик
    roc_metrics = evaluator.compute_roc_metrics(similarity_scores, true_labels)
    evaluator.roc_metrics = roc_metrics
    
    # Поиск оптимального порога
    optimal_result = evaluator.find_optimal_threshold(similarity_scores, true_labels)
    
    # Вывод отчета
    evaluator.print_summary(results)
    
    # Сохранение результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON отчет
    json_path = output_dir / f"evaluation_results_{timestamp}.json"
    evaluator.save_results_json(results, json_path)
    
    # CSV отчет
    csv_path = output_dir / f"evaluation_results_{timestamp}.csv"
    evaluator.save_results_csv(results, csv_path)
    
    # Генерация графиков
    if args.generate_plots:
        # ROC кривая
        roc_path = output_dir / f"roc_curve_{timestamp}.png"
        evaluator.plot_roc_curve(str(roc_path))
        
        # Распределение метрик
        metrics_path = output_dir / f"metrics_distribution_{timestamp}.png"
        evaluator.plot_metrics_distribution(results, str(metrics_path))
    
    # Сохранение оптимального порога
    optimal_path = output_dir / f"optimal_threshold_{timestamp}.txt"
    with open(optimal_path, 'w') as f:
        f.write(f"Optimal threshold: {optimal_result.threshold:.6f}\n")
        f.write(f"Best F1-Score: {optimal_result.f1_score:.6f}\n")
        f.write(f"EER: {roc_metrics.eer:.6f}\n")
        f.write(f"AUC: {roc_metrics.auc_score:.6f}\n")
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()