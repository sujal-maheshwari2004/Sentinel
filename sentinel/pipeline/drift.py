# sentinel/pipeline/drift.py

import threading
import numpy as np
from collections import deque
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)


class DriftSeverity:
    NONE = "none"
    LOW = "low"        # finetune
    HIGH = "high"      # full retrain


class DriftResult:
    def __init__(self, severity: str, mae: float, threshold_finetune: float, threshold_retrain: float):
        self.severity = severity
        self.mae = mae
        self.threshold_finetune = threshold_finetune
        self.threshold_retrain = threshold_retrain

    def __repr__(self):
        return (
            f"DriftResult(severity={self.severity}, mae={self.mae:.4f}, "
            f"finetune_threshold={self.threshold_finetune}, "
            f"retrain_threshold={self.threshold_retrain})"
        )


class DriftMonitor:
    """
    Rolling MAE drift monitor for a single metric.

    Maintains a sliding window of (predicted, actual) pairs.
    On each check, computes MAE over the window and classifies
    drift severity against the configured thresholds.

    Severity rules:
        mae < finetune_threshold  -> NONE
        mae >= finetune_threshold -> LOW  (trigger finetune)
        mae >= retrain_threshold  -> HIGH (trigger full retrain)
    """

    def __init__(
        self,
        metric: str,
        finetune_threshold: float,
        retrain_threshold: float,
        window_size: int = 60,
    ):
        """
        metric              : metric name for logging
        finetune_threshold  : MAE above this triggers finetune
        retrain_threshold   : MAE above this triggers full retrain
        window_size         : number of recent predictions to compute MAE over
        """
        self.metric = metric
        self.finetune_threshold = finetune_threshold
        self.retrain_threshold = retrain_threshold
        self.window_size = window_size

        self._residuals: deque[float] = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def record(self, predicted: float, actual: float) -> None:
        """
        Record a single predicted vs actual pair.
        Called by the emitter each time a new actual value arrives
        for a timestep that was previously predicted.
        """
        with self._lock:
            self._residuals.append(abs(predicted - actual))

    def record_many(self, predicted: np.ndarray, actual: np.ndarray) -> None:
        """
        Record a batch of predicted vs actual pairs.
        """
        with self._lock:
            for p, a in zip(predicted, actual):
                self._residuals.append(abs(float(p) - float(a)))

    def check(self) -> DriftResult:
        """
        Compute current MAE and classify drift severity.
        Returns DriftResult with severity, current MAE, and thresholds.
        """
        with self._lock:
            if not self._residuals:
                return DriftResult(
                    severity=DriftSeverity.NONE,
                    mae=0.0,
                    threshold_finetune=self.finetune_threshold,
                    threshold_retrain=self.retrain_threshold,
                )
            mae = float(np.mean(self._residuals))

        if mae >= self.retrain_threshold:
            severity = DriftSeverity.HIGH
        elif mae >= self.finetune_threshold:
            severity = DriftSeverity.LOW
        else:
            severity = DriftSeverity.NONE

        logger.debug(f"[{self.metric}] drift check: MAE={mae:.4f} severity={severity}")

        return DriftResult(
            severity=severity,
            mae=mae,
            threshold_finetune=self.finetune_threshold,
            threshold_retrain=self.retrain_threshold,
        )

    def current_mae(self) -> float:
        with self._lock:
            if not self._residuals:
                return 0.0
            return float(np.mean(self._residuals))

    def sample_count(self) -> int:
        with self._lock:
            return len(self._residuals)

    def reset(self) -> None:
        """
        Clear residual history. Called after a successful retrain
        so the new model starts with a clean drift slate.
        """
        with self._lock:
            self._residuals.clear()
        logger.debug(f"[{self.metric}] drift monitor reset")

    def __repr__(self) -> str:
        return (
            f"DriftMonitor(metric={self.metric}, "
            f"mae={self.current_mae():.4f}, "
            f"samples={self.sample_count()}/{self.window_size})"
        )