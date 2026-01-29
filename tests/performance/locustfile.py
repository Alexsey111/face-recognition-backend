# tests/performance/locustfile.py

"""
Locust configuration для различных сценариев нагрузочного тестирования
"""

import math

from locust import HttpUser, LoadTestShape, between, task

from tests.performance.load_test_verification import (
    StressTestUser,
    VerificationLoadTest,
)


class StepLoadShape(LoadTestShape):
    """
    Ступенчатое увеличение нагрузки

    Сценарий:
    - 0-60s:   10 пользователей
    - 60-120s: 50 пользователей
    - 120-180s: 100 пользователей
    - 180-240s: 200 пользователей
    - 240-300s: 100 пользователей (снижение)
    """

    step_time = 60
    step_load = 10
    spawn_rate = 10
    time_limit = 300

    def tick(self):
        run_time = self.get_run_time()

        if run_time > self.time_limit:
            return None

        current_step = math.floor(run_time / self.step_time)

        # Определяем количество пользователей для текущего шага
        if current_step == 0:
            user_count = 10
        elif current_step == 1:
            user_count = 50
        elif current_step == 2:
            user_count = 100
        elif current_step == 3:
            user_count = 200
        else:  # current_step >= 4
            user_count = 100

        return (user_count, self.spawn_rate)


class SpikeLoadShape(LoadTestShape):
    """
    Нагрузка с пиками (spike testing)

    Симулирует внезапные всплески трафика
    """

    time_limit = 300
    spawn_rate = 20

    def tick(self):
        run_time = self.get_run_time()

        if run_time > self.time_limit:
            return None

        # Базовая нагрузка: 20 пользователей
        base_load = 20

        # Пики каждые 60 секунд
        if 30 < (run_time % 60) < 35:
            # Пик: 200 пользователей в течение 5 секунд
            user_count = 200
        else:
            user_count = base_load

        return (user_count, self.spawn_rate)


class SoakLoadShape(LoadTestShape):
    """
    Длительное тестирование (soak test)

    Стабильная нагрузка в течение длительного времени
    для выявления memory leaks, degradation
    """

    time_limit = 3600  # 1 час
    spawn_rate = 10
    user_count = 50  # Постоянная нагрузка

    def tick(self):
        run_time = self.get_run_time()

        if run_time > self.time_limit:
            return None

        return (self.user_count, self.spawn_rate)
