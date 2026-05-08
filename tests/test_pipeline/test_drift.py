# tests/test_pipeline/test_drift.py

import pytest
import numpy as np
from sentinel.pipeline.drift import DriftMonitor, DriftSeverity, DriftResult


class TestDriftMonitor:

    def _make_monitor(self, finetune=0.1, retrain=0.3, window=10):
        return DriftMonitor(
            metric="test_metric",
            finetune_threshold=finetune,
            retrain_threshold=retrain,
            window_size=window,
        )

    def test_initial_check_returns_none_severity(self):
        monitor = self._make_monitor()
        result = monitor.check()
        assert result.severity == DriftSeverity.NONE
        assert result.mae == 0.0

    def test_record_single(self):
        monitor = self._make_monitor()
        monitor.record(predicted=1.0, actual=1.05)
        assert monitor.sample_count() == 1

    def test_no_drift_below_threshold(self):
        monitor = self._make_monitor()
        for _ in range(5):
            monitor.record(predicted=1.0, actual=1.01)  # MAE = 0.01
        result = monitor.check()
        assert result.severity == DriftSeverity.NONE

    def test_low_drift_between_thresholds(self):
        monitor = self._make_monitor(finetune=0.1, retrain=0.3)
        for _ in range(5):
            monitor.record(predicted=1.0, actual=1.2)  # MAE = 0.2
        result = monitor.check()
        assert result.severity == DriftSeverity.LOW

    def test_high_drift_above_retrain_threshold(self):
        monitor = self._make_monitor(finetune=0.1, retrain=0.3)
        for _ in range(5):
            monitor.record(predicted=1.0, actual=1.5)  # MAE = 0.5
        result = monitor.check()
        assert result.severity == DriftSeverity.HIGH

    def test_record_many(self):
        monitor = self._make_monitor()
        predicted = np.array([1.0, 2.0, 3.0])
        actual = np.array([1.1, 2.1, 3.1])
        monitor.record_many(predicted, actual)
        assert monitor.sample_count() == 3
        assert monitor.current_mae() == pytest.approx(0.1, abs=1e-6)

    def test_window_evicts_old_samples(self):
        monitor = self._make_monitor(window=3)
        for _ in range(3):
            monitor.record(predicted=1.0, actual=1.5)  # high residuals
        for _ in range(3):
            monitor.record(predicted=1.0, actual=1.01)  # low residuals
        # only last 3 should count
        assert monitor.current_mae() == pytest.approx(0.01, abs=1e-6)

    def test_reset_clears_residuals(self):
        monitor = self._make_monitor()
        for _ in range(5):
            monitor.record(predicted=1.0, actual=2.0)
        monitor.reset()
        assert monitor.sample_count() == 0
        assert monitor.current_mae() == 0.0
        result = monitor.check()
        assert result.severity == DriftSeverity.NONE

    def test_current_mae_empty(self):
        monitor = self._make_monitor()
        assert monitor.current_mae() == 0.0

    def test_drift_result_carries_thresholds(self):
        monitor = self._make_monitor(finetune=0.1, retrain=0.3)
        monitor.record(1.0, 1.2)
        result = monitor.check()
        assert result.threshold_finetune == 0.1
        assert result.threshold_retrain == 0.3

    def test_thread_safety(self):
        import threading
        monitor = self._make_monitor(window=100)
        errors = []

        def recorder():
            try:
                for _ in range(50):
                    monitor.record(1.0, 1.1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=recorder) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []