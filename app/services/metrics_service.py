"""Сервис для расчета и отслеживания метрик FAR/FRR"""
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
import numpy as np
from collections import deque
from dataclasses import dataclass, field

from app.services.cache_service import CacheService
from app.services.database_service import DatabaseService
from app.middleware.metrics import (
    false_accept_rate,
    false_reject_rate,
    equal_error_rate,
    false_accept_total,
    false_reject_total,
    record_verification_duration,
    record_verification_confidence,
)
from app.utils.logger import get_logger
from app.config import settings

logger = get_logger(__name__)


@dataclass
class MetricsConfig:
    """Конфигурация метрик."""
    WINDOW_SIZE: int = 1000
    UPDATE_INTERVAL: int = 60
    TARGET_FAR: float = 0.001  # < 0.1%
    TARGET_FRR: float = 0.03   # < 3%
    SEVERITY_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.1,
        'medium': 0.2,
    })


class MetricsService:
    """Сервис для расчета и мониторинга FAR/FRR метрик."""

    def __init__(
        self,
        cache_service: Optional[CacheService] = None,
        db_service: Optional[DatabaseService] = None,
        config: Optional[MetricsConfig] = None,
    ):
        self.config = config or MetricsConfig()
        self.cache = cache_service or CacheService()
        self.db = db_service or DatabaseService()

        # Буферы для скользящего окна
        self.genuine_scores = deque(maxlen=self.config.WINDOW_SIZE)
        self.impostor_scores = deque(maxlen=self.config.WINDOW_SIZE)

        # Счетчики (атомарные для thread-safety)
        self._genuine_total = 0
        self._genuine_rejected = 0
        self._impostor_total = 0
        self._impostor_accepted = 0
        self._lock = asyncio.Lock()

        # Фоновые задачи
        self._update_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Запуск фонового процесса обновления метрик."""
        if self._running:
            return

        self._running = True
        self._update_task = asyncio.create_task(self._update_metrics_loop())
        logger.info("MetricsService started")

    async def stop(self) -> None:
        """Graceful остановка фонового процесса."""
        if not self._running:
            return

        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await asyncio.wait_for(self._update_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Metrics update task did not stop in time")
            except asyncio.CancelledError:
                pass

        # Финальное обновление метрик
        await self._update_metrics()
        logger.info("MetricsService stopped")

    async def record_verification(
        self,
        similarity_score: float,
        is_genuine: bool,
        threshold: float,
        match_result: bool,
        processing_time_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Запись результата верификации.

        Args:
            similarity_score: Скор схожести [0, 1]
            is_genuine: True если это genuine pair (одинаковые люди)
            threshold: Используемый порог
            match_result: Результат верификации (match/no match)
            processing_time_ms: Время обработки для метрик

        Returns:
            Словарь с деталями записи
        """
        async with self._lock:
            if is_genuine:
                self.genuine_scores.append(similarity_score)
                self._genuine_total += 1
                is_rejected = not match_result

                if is_rejected:
                    self._genuine_rejected += 1
                    severity = self._calculate_severity(
                        similarity_score, threshold, is_false_reject=True
                    )
                    false_reject_total.labels(severity=severity).inc()
                    logger.warning(
                        f"False Reject: score={similarity_score:.3f}, "
                        f"threshold={threshold:.3f}, severity={severity}"
                    )
                    return {
                        'event': 'false_reject',
                        'severity': severity,
                        'score': similarity_score,
                        'threshold': threshold,
                    }
            else:
                self.impostor_scores.append(similarity_score)
                self._impostor_total += 1
                is_accepted = match_result

                if is_accepted:
                    self._impostor_accepted += 1
                    severity = self._calculate_severity(
                        similarity_score, threshold, is_false_reject=False
                    )
                    false_accept_total.labels(severity=severity).inc()
                    logger.error(
                        f"False Accept: score={similarity_score:.3f}, "
                        f"threshold={threshold:.3f}, severity={severity}"
                    )
                    return {
                        'event': 'false_accept',
                        'severity': severity,
                        'score': similarity_score,
                        'threshold': threshold,
                    }

            # Record processing metrics
            if processing_time_ms is not None:
                record_verification_duration("cosine", processing_time_ms / 1000.0)
            record_verification_confidence(
                "match" if match_result else "no_match",
                similarity_score
            )
        
            return {'event': 'verification_recorded', 'score': similarity_score}

    def _calculate_severity(
        self,
        score: float,
        threshold: float,
        is_false_reject: bool,
    ) -> str:
        """
        Определение серьезности ошибки.

        Returns:
            'low', 'medium', 'high'
        """
        distance = abs(score - threshold)
        low_threshold = self.config.SEVERITY_THRESHOLDS['low']
        medium_threshold = self.config.SEVERITY_THRESHOLDS['medium']

        if is_false_reject:
            # FRR: genuine pair отклонена
            if distance > medium_threshold:
                return 'high'
            elif distance > low_threshold:
                return 'medium'
            return 'low'
        else:
            # FAR: impostor accepted
            if score > threshold + medium_threshold:
                return 'high'
            elif score > threshold + low_threshold:
                return 'medium'
            return 'low'

    async def _update_metrics(self) -> None:
        """Обновление Prometheus метрик."""
        async with self._lock:
            genuine_total = self._genuine_total
            genuine_rejected = self._genuine_rejected
            impostor_total = self._impostor_total
            impostor_accepted = self._impostor_accepted

        # FAR: False Accept Rate
        far = (impostor_accepted / impostor_total * 100) if impostor_total > 0 else 0.0
        false_accept_rate.set(far)

        # FRR: False Reject Rate
        frr = (genuine_rejected / genuine_total * 100) if genuine_total > 0 else 0.0
        false_reject_rate.set(frr)

        # EER: Equal Error Rate
        eer = 0.0
        if len(self.genuine_scores) > 0 and len(self.impostor_scores) > 0:
            eer = self._calculate_eer(
                list(self.genuine_scores),
                list(self.impostor_scores)
            )
        equal_error_rate.set(eer)
        
        logger.debug(
            f"Metrics updated: FAR={far:.4f}%, FRR={frr:.4f}%, "
            f"EER={eer:.4f}%, window={len(self.genuine_scores) + len(self.impostor_scores)}"
        )
        
    def _calculate_eer(
        self,
        genuine_scores: List[float],
        impostor_scores: List[float],
    ) -> float:
        """
        Расчет Equal Error Rate (EER).

        EER - точка где FAR = FRR.
        """
        if not genuine_scores or not impostor_scores:
            return 0.0

        genuine_arr = np.array(genuine_scores)
        impostor_arr = np.array(impostor_scores)

        # Оптимизированный поиск EER
        thresholds = np.linspace(0, 1, 1001)
        far_values = []
        frr_values = []

        for threshold in thresholds:
            far = np.sum(impostor_arr >= threshold) / len(impostor_arr) * 100
            frr = np.sum(genuine_arr < threshold) / len(genuine_arr) * 100
            far_values.append(far)
            frr_values.append(frr)

        far_arr = np.array(far_values)
        frr_arr = np.array(frr_values)
        differences = np.abs(far_arr - frr_arr)
        eer_idx = np.argmin(differences)

        return (far_arr[eer_idx] + frr_arr[eer_idx]) / 2

    async def _update_metrics_loop(self) -> None:
        """Фоновый цикл обновления метрик."""
        while self._running:
            try:
                await asyncio.sleep(self.config.UPDATE_INTERVAL)
                await self._update_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics update loop: {e}")

    async def get_current_metrics(self) -> Dict:
        """
        Получение текущих метрик.

        Returns:
            Словарь с текущими метриками
        """
        async with self._lock:
            genuine_total = self._genuine_total
            genuine_rejected = self._genuine_rejected
            impostor_total = self._impostor_total
            impostor_accepted = self._impostor_accepted
            genuine_scores = list(self.genuine_scores)
            impostor_scores = list(self.impostor_scores)

        # Получаем значения метрик безопасным способом
        try:
            far = false_accept_rate._value.get() if hasattr(false_accept_rate, '_value') else 0.0
            frr = false_reject_rate._value.get() if hasattr(false_reject_rate, '_value') else 0.0
            eer = equal_error_rate._value.get() if hasattr(equal_error_rate, '_value') else 0.0
        except (AttributeError, ValueError):
            # Fallback к прямым переменным
            far = (impostor_accepted / impostor_total * 100) if impostor_total > 0 else 0.0
            frr = (genuine_rejected / genuine_total * 100) if genuine_total > 0 else 0.0
            eer = 0.0

        return {
            "far": round(far, 4),
            "frr": round(frr, 4),
            "eer": round(eer, 4),
            "genuine_count": genuine_total,
            "impostor_count": impostor_total,
            "compliance": {
                "far_compliant": far < self.config.TARGET_FAR * 100,
                "frr_compliant": frr < self.config.TARGET_FRR * 100,
                "overall_compliant": (
                    far < self.config.TARGET_FAR * 100 and
                    frr < self.config.TARGET_FRR * 100
                ),
            },
            "window_size": len(genuine_scores) + len(impostor_scores),
            "distributions": {
                "genuine_mean": round(float(np.mean(genuine_scores)), 4) if genuine_scores else 0,
                "genuine_std": round(float(np.std(genuine_scores)), 4) if genuine_scores else 0,
                "impostor_mean": round(float(np.mean(impostor_scores)), 4) if impostor_scores else 0,
                "impostor_std": round(float(np.std(impostor_scores)), 4) if impostor_scores else 0,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def get_historical_metrics(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Dict]:
        """
        Получение исторических метрик из БД.

        Args:
            start_date: Начальная дата
            end_date: Конечная дата

        Returns:
            Список метрик по временным интервалам
        """
        return await self.db.get_metrics_history(start_date, end_date)

    def get_status(self) -> Dict[str, Any]:
        """
        Получение статуса сервиса.

        Returns:
            Словарь со статусом
        """
        return {
            "running": self._running,
            "update_task_active": self._update_task is not None and not self._update_task.done(),
            "window_size": len(self.genuine_scores) + len(self.impostor_scores),
            "genuine_buffer_size": len(self.genuine_scores),
            "impostor_buffer_size": len(self.impostor_scores),
        }


# Global service instance (singleton pattern)
_metrics_service: Optional[MetricsService] = None


async def get_metrics_service() -> MetricsService:
    """
    Получение экземпляра MetricsService.

    Returns:
        Единственный экземпляр MetricsService
    """
    global _metrics_service

    if _metrics_service is None:
        _metrics_service = MetricsService()
        await _metrics_service.start()

    return _metrics_service