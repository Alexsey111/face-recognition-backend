#!/usr/bin/env python3
"""
Скрипт проверки соответствия метрик требованиям ТЗ
Поддерживает проверку FAR, FRR, Liveness, EER и других метрик
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

# ANSI color codes for terminal output (Windows-compatible)
class Colors:
    """Цветовые коды для терминала"""
    RESET = ""
    BOLD = ""
    GREEN = ""
    RED = ""
    YELLOW = ""
    BLUE = ""
    CYAN = ""
    # Enable colors only if terminal supports it
    @staticmethod
    def enable():
        import os
        if sys.platform != "win32" or os.environ.get("TERM"):
            Colors.RESET = "\033[0m"
            Colors.BOLD = "\033[1m"
            Colors.GREEN = "\033[92m"
            Colors.RED = "\033[91m"
            Colors.YELLOW = "\033[93m"
            Colors.BLUE = "\033[94m"
            Colors.CYAN = "\033[96m"


class Status(Enum):
    """Статус проверки"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ThresholdConfig:
    """Конфигурация порогового значения"""
    threshold: float
    operator: str  # "<", ">", "<=", ">="
    description: str = ""
    
    def check(self, value: float) -> bool:
        """Проверка значения относительно порога"""
        if self.operator == "<":
            return value < self.threshold
        elif self.operator == ">":
            return value > self.threshold
        elif self.operator == "<=":
            return value <= self.threshold
        elif self.operator == ">=":
            return value >= self.threshold
        return False


@dataclass
class MetricRequirement:
    """Требование к метрике"""
    name: str
    display_name: str
    category: str
    threshold: ThresholdConfig
    weight: float = 1.0  # Для взвешенной оценки
    enabled: bool = True
    critical: bool = False  # Критическое требование - провал = общий провал


@dataclass
class ComplianceResult:
    """Результат проверки соответствия"""
    metric_name: str
    display_name: str
    category: str
    value: float
    threshold: float
    operator: str
    status: Status
    message: str
    is_critical: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "metric_name": self.metric_name,
            "display_name": self.display_name,
            "category": self.category,
            "value": self.value,
            "threshold": self.threshold,
            "operator": self.operator,
            "status": self.status.value,
            "message": self.message,
            "is_critical": self.is_critical
        }


@dataclass
class ComplianceReport:
    """Полный отчёт о соответствии"""
    timestamp: str
    input_file: str
    spec_version: str
    total_metrics: int = 0
    passed: int = 0
    failed: int = 0
    warnings: int = 0
    skipped: int = 0
    overall_status: str = "unknown"
    results: List[Dict] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "input_file": self.input_file,
            "spec_version": self.spec_version,
            "total_metrics": self.total_metrics,
            "passed": self.passed,
            "failed": self.failed,
            "warnings": self.warnings,
            "skipped": self.skipped,
            "overall_status": self.overall_status,
            "results": self.results,
            "summary": self.summary
        }


class ComplianceChecker:
    """Проверка соответствия метрик требованиям ТЗ"""
    
    # Стандартные требования ТЗ (версия 1.0)
    DEFAULT_REQUIREMENTS = {
        "verification": {
            "far": MetricRequirement(
                name="far",
                display_name="False Accept Rate (FAR)",
                category="verification",
                threshold=ThresholdConfig(
                    threshold=0.1,
                    operator="<",
                    description="Максимальный процент ложных срабатываний"
                ),
                weight=1.5,
                critical=True
            ),
            "frr": MetricRequirement(
                name="frr",
                display_name="False Reject Rate (FRR)",
                category="verification",
                threshold=ThresholdConfig(
                    threshold=3.0,
                    operator="<",
                    description="Максимальный процент ложных отклонений"
                ),
                weight=1.5,
                critical=True
            ),
            "eer": MetricRequirement(
                name="eer",
                display_name="Equal Error Rate (EER)",
                category="verification",
                threshold=ThresholdConfig(
                    threshold=1.0,
                    operator="<",
                    description="Точка равенства FAR и FRR (ниже лучше)"
                ),
                weight=1.0,
                critical=False
            ),
            "fmr_01": MetricRequirement(
                name="fmr_01",
                display_name="FMR @ FMR=0.1%",
                category="verification",
                threshold=ThresholdConfig(
                    threshold=5.0,
                    operator="<",
                    description="FNMR при FMR=0.1%"
                ),
                weight=0.5,
                critical=False
            ),
            "fnmr_01": MetricRequirement(
                name="fnmr_01",
                display_name="FNMR @ FMR=0.1%",
                category="verification",
                threshold=ThresholdConfig(
                    threshold=5.0,
                    operator="<",
                    description="False Non-Match Rate при FMR=0.1%"
                ),
                weight=0.5,
                critical=False
            )
        },
        "liveness": {
            "accuracy": MetricRequirement(
                name="accuracy",
                display_name="Liveness Accuracy",
                category="liveness",
                threshold=ThresholdConfig(
                    threshold=98.0,
                    operator=">",
                    description="Минимальная точность детекции живого лица"
                ),
                weight=2.0,
                critical=True
            ),
            "precision": MetricRequirement(
                name="precision",
                display_name="Liveness Precision",
                category="liveness",
                threshold=ThresholdConfig(
                    threshold=97.0,
                    operator=">",
                    description="Precision для детекции живого лица"
                ),
                weight=1.0,
                critical=False
            ),
            "recall": MetricRequirement(
                name="recall",
                display_name="Liveness Recall",
                category="liveness",
                threshold=ThresholdConfig(
                    threshold=97.0,
                    operator=">",
                    description="Recall для детекции живого лица"
                ),
                weight=1.0,
                critical=False
            ),
            "f1_score": MetricRequirement(
                name="f1_score",
                display_name="Liveness F1-Score",
                category="liveness",
                threshold=ThresholdConfig(
                    threshold=97.0,
                    operator=">",
                    description="F1-Score для детекции живого лица"
                ),
                weight=1.0,
                critical=False
            ),
            "apcer": MetricRequirement(
                name="apcer",
                display_name="APCER",
                category="liveness",
                threshold=ThresholdConfig(
                    threshold=2.0,
                    operator="<",
                    description="Attack Presentation Classification Error Rate"
                ),
                weight=1.0,
                critical=False
            ),
            "bpcer": MetricRequirement(
                name="bpcer",
                display_name="BPCER",
                category="liveness",
                threshold=ThresholdConfig(
                    threshold=2.0,
                    operator="<",
                    description="Bona Fide Presentation Classification Error Rate"
                ),
                weight=1.0,
                critical=False
            )
        },
        "performance": {
            "inference_time_ms": MetricRequirement(
                name="inference_time_ms",
                display_name="Inference Time",
                category="performance",
                threshold=ThresholdConfig(
                    threshold=100.0,
                    operator="<",
                    description="Максимальное время инференса в мс"
                ),
                weight=1.0,
                critical=False
            ),
            "throughput_fps": MetricRequirement(
                name="throughput_fps",
                display_name="Throughput",
                category="performance",
                threshold=ThresholdConfig(
                    threshold=10.0,
                    operator=">",
                    description="Минимальная пропускная способность (FPS)"
                ),
                weight=1.0,
                critical=False
            )
        }
    }
    
    def __init__(
        self,
        spec_version: str = "1.0",
        requirements: Optional[Dict] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.spec_version = spec_version
        self.requirements = requirements or self.DEFAULT_REQUIREMENTS
        self.logger = logger or self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера"""
        logger = logging.getLogger("compliance_checker")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def _get_nested_value(self, data: Dict, path: str) -> Optional[float]:
        """Получение значения из вложенного словаря по пути"""
        keys = path.split("/")
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return float(value) if value is not None else None
    
    def _format_value(self, value: float, suffix: str = "%") -> str:
        """Форматирование значения для вывода"""
        if value is None:
            return "N/A"
        if abs(value) >= 1000:
            return f"{value:.2e}{suffix}"
        return f"{value:.4f}{suffix}"
    
    def check_metric(
        self,
        requirement: MetricRequirement,
        value: Optional[float]
    ) -> ComplianceResult:
        """Проверка одной метрики"""
        if value is None:
            return ComplianceResult(
                metric_name=requirement.name,
                display_name=requirement.display_name,
                category=requirement.category,
                value=0.0,
                threshold=requirement.threshold.threshold,
                operator=requirement.threshold.operator,
                status=Status.SKIPPED,
                message=f"Метрика '{requirement.name}' не найдена в результатах",
                is_critical=requirement.critical
            )
        
        passed = requirement.threshold.check(value)
        
        if passed:
            status = Status.PASSED
            message = f"[PASS] Requirement met"
        else:
            status = Status.FAILED
            relation = "<" if requirement.threshold.operator == "<" else ">"
            message = f"[FAIL] Requirement not met: {value:.4f} {relation} {requirement.threshold.threshold}"
        
        return ComplianceResult(
            metric_name=requirement.name,
            display_name=requirement.display_name,
            category=requirement.category,
            value=value,
            threshold=requirement.threshold.threshold,
            operator=requirement.threshold.operator,
            status=status,
            message=message,
            is_critical=requirement.critical
        )
    
    def check_compliance(
        self,
        results_file: Path,
        fail_on_violation: bool = True,
        output_format: str = "both"  # "text", "json", "both"
    ) -> ComplianceReport:
        """
        Проверка соответствия требованиям ТЗ
        
        Args:
            results_file: Путь к JSON-файлу с результатами
            fail_on_violation: Завершить с ошибкой при нарушении требований
            output_format: Формат вывода ("text", "json", "both")
        
        Returns:
            ComplianceReport с результатами проверки
        """
        self.logger.info(f"Загрузка результатов из {results_file}")
        
        # Загрузка данных
        with open(results_file) as f:
            data = json.load(f)
        
        report = ComplianceReport(
            timestamp=datetime.now().isoformat(),
            input_file=str(results_file),
            spec_version=self.spec_version
        )
        
        # Проверка всех метрик
        for category, requirements in self.requirements.items():
            for name, requirement in requirements.items():
                if not requirement.enabled:
                    continue
                
                # Поиск значения в данных (поддержка разных форматов)
                value = self._get_nested_value(data, f"{category}/{name}/value")
                if value is None:
                    value = self._get_nested_value(data, f"{category}/{name}")
                
                result = self.check_metric(requirement, value)
                report.results.append(result.to_dict())
        
        # Подсчёт результатов
        for result in report.results:
            if result["status"] == "passed":
                report.passed += 1
            elif result["status"] == "failed":
                report.failed += 1
                if result["is_critical"]:
                    report.overall_status = "failed"
            elif result["status"] == "warning":
                report.warnings += 1
            elif result["status"] == "skipped":
                report.skipped += 1
        
        report.total_metrics = len(report.results)
        
        # Определение общего статуса
        if report.failed > 0:
            # Проверяем, есть ли критические провалы
            critical_failures = [r for r in report.results 
                                if r["status"] == "failed" and r["is_critical"]]
            if critical_failures:
                report.overall_status = "failed"
            else:
                report.overall_status = "warning"
        elif report.warnings > 0:
            report.overall_status = "warning"
        else:
            report.overall_status = "passed"
        
        # Генерация summary по категориям
        categories = set(r["category"] for r in report.results)
        report.summary = {}
        for category in categories:
            cat_results = [r for r in report.results if r["category"] == category]
            cat_passed = sum(1 for r in cat_results if r["status"] == "passed")
            cat_failed = sum(1 for r in cat_results if r["status"] == "failed")
            report.summary[category] = {
                "total": len(cat_results),
                "passed": cat_passed,
                "failed": cat_failed,
                "status": "passed" if cat_failed == 0 else "warning" if cat_passed > 0 else "failed"
            }
        
        # Вывод результатов
        if output_format in ("text", "both"):
            self._print_text_report(report)
        
        if output_format in ("json", "both"):
            json_output = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
            print(f"\n{Colors.CYAN}=== JSON Report ==={Colors.RESET}")
            print(json_output)
        
        # Вывод финального результата
        self._print_final_status(report)
        
        # Завершение с ошибкой при необходимости
        if fail_on_violation and report.overall_status == "failed":
            self.logger.error("Проверка не пройдена: есть критические нарушения требований")
            sys.exit(1)
        
        return report
    
    def _print_text_report(self, report: ComplianceReport):
        """Вывод текстового отчёта"""
        print(f"\n{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        print(f"{Colors.BOLD}COMPLIANCE CHECK RESULTS{Colors.RESET}")
        print(f"{Colors.BOLD}Spec Version: {report.spec_version}{Colors.RESET}")
        print(f"{Colors.BOLD}Timestamp: {report.timestamp}{Colors.RESET}")
        print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        
        # Группировка по категориям
        categories = {}
        for result in report.results:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)
        
        for category, results in categories.items():
            print(f"\n{Colors.BLUE}{'-' * 40}{Colors.RESET}")
            print(f"{Colors.BOLD}{category.upper()}{Colors.RESET}")
            print(f"{Colors.BLUE}{'-' * 40}{Colors.RESET}")

            for result in results:
                # Выбор цвета по статусу
                if result["status"] == "passed":
                    color = Colors.GREEN
                    icon = "[PASS]"
                elif result["status"] == "failed":
                    color = Colors.RED
                    icon = "[FAIL]"
                elif result["status"] == "warning":
                    color = Colors.YELLOW
                    icon = "[WARN]"
                else:
                    color = Colors.BLUE
                    icon = "[SKIP]"

                # Форматирование строки
                status_str = f"{color}{icon} {result['status'].upper()}{Colors.RESET}"
                value_str = self._format_value(result["value"])
                threshold_str = self._format_value(result["threshold"])
                
                line = f"  {result['display_name']}:"
                print(f"{line:<40} {value_str} {result['operator']} {threshold_str} {status_str}")
                
                if result["status"] == "failed" and result.get("is_critical"):
                    print(f"    {Colors.RED}*** CRITICAL ***{Colors.RESET}")
        
        # Summary по категориям
        print(f"\n{Colors.BLUE}{'-' * 40}{Colors.RESET}")
        print(f"{Colors.BOLD}CATEGORY SUMMARY{Colors.RESET}")
        print(f"{Colors.BLUE}{'-' * 40}{Colors.RESET}")

        for category, summary in report.summary.items():
            if summary["status"] == "passed":
                color = Colors.GREEN
                icon = "[OK]"
            elif summary["status"] == "warning":
                color = Colors.YELLOW
                icon = "[WARN]"
            else:
                color = Colors.RED
                icon = "[FAIL]"

            print(f"  {category:<20} {summary['passed']}/{summary['total']} {color}{icon}{Colors.RESET}")
        
        # Общая статистика
        print(f"\n{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        print(f"{Colors.BOLD}TOTAL: {report.passed} passed, {report.failed} failed, {report.warnings} warnings, {report.skipped} skipped{Colors.RESET}")
    
    def _print_final_status(self, report: ComplianceReport):
        """Вывод финального статуса"""
        print(f"\n{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        
        if report.overall_status == "passed":
            print(f"{Colors.GREEN}{Colors.BOLD}[SUCCESS] ALL COMPLIANCE CHECKS PASSED!{Colors.RESET}")
        elif report.overall_status == "warning":
            print(f"{Colors.YELLOW}{Colors.BOLD}[WARNING] COMPLIANCE CHECKS PASSED WITH WARNINGS{Colors.RESET}")
        else:
            print(f"{Colors.RED}{Colors.BOLD}[FAILED] COMPLIANCE CHECKS FAILED!{Colors.RESET}")

        print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
    
    def save_report(self, report: ComplianceReport, output_path: Path):
        """Сохранение отчёта в JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        self.logger.info(f"Отчёт сохранён в {output_path}")


def load_requirements_from_file(config_path: Path) -> Dict:
    """Загрузка требований из конфигурационного файла"""
    with open(config_path) as f:
        return json.load(f)


def main():
    # Enable colors for terminal output
    Colors.enable()
    
    parser = argparse.ArgumentParser(
        description="Проверка соответствия метрик требованиям ТЗ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python check_compliance.py --results validation_results.json
  python check_compliance.py --results results.json --fail-on-violation
  python check_compliance.py --results results.json --output-format json
  python check_compliance.py --results results.json --config custom_requirements.json
        """
    )
    
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Путь к JSON-файлу с результатами метрик"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Путь к конфигурационному файлу с требованиями (JSON)"
    )
    
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        default=True,
        help="Завершить с ошибкой при нарушении требований (по умолчанию включено)"
    )
    
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Не завершать с ошибкой при нарушении требований"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["text", "json", "both"],
        default="both",
        help="Формат вывода (по умолчанию: both)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Путь для сохранения JSON-отчёта"
    )
    
    parser.add_argument(
        "--spec-version",
        type=str,
        default="1.0",
        help="Версия спецификации требований (по умолчанию: 1.0)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Подробный вывод"
    )
    
    args = parser.parse_args()
    
    # Настройка логгера
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)
    logger = logging.getLogger("compliance_checker")
    
    # Загрузка требований
    requirements = None
    if args.config:
        logger.info(f"Загрузка требований из {args.config}")
        requirements = load_requirements_from_file(args.config)
    
    # Создание checker
    checker = ComplianceChecker(
        spec_version=args.spec_version,
        requirements=requirements,
        logger=logger
    )
    
    # Проверка соответствия
    fail_on_violation = not args.no_fail
    report = checker.check_compliance(
        results_file=args.results,
        fail_on_violation=fail_on_violation,
        output_format=args.output_format
    )
    
    # Сохранение отчёта
    if args.output:
        checker.save_report(report, args.output)
    
    # Код выхода
    if report.overall_status == "passed":
        sys.exit(0)
    elif report.overall_status == "warning":
        sys.exit(0)  # Предупреждения не считаются ошибкой
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()