# sentinel/pipeline/scheduler.py

import threading
import time
from datetime import datetime, timezone
from typing import Callable
from sentinel.pipeline.drift import DriftMonitor, DriftSeverity
from sentinel.ingestor.buffer import MetricBuffer
from sentinel.config import WatchConfig
from sentinel.utils.cron import seconds_until_next_fire
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

# how often the scheduler wakes up to check cron and drift
_TICK_INTERVAL_SECONDS = 10


class MetricScheduler:
    """
    Manages the training schedule for a single metric.

    Two triggers:
        1. Cron — fires on the configured cron schedule
        2. Drift — fires when DriftMonitor reports LOW or HIGH severity

    On each trigger, calls the provided train_fn with the appropriate
    drift severity so Trainer knows which policy to apply.

    Runs in a dedicated daemon thread per metric.
    """

    def __init__(
        self,
        watch_config: WatchConfig,
        buffer: MetricBuffer,
        drift_monitor: DriftMonitor,
        train_fn: Callable[[str, float], None],
    ):
        """
        watch_config  : config for this metric
        buffer        : buffer to check readiness before training
        drift_monitor : drift monitor to poll for severity
        train_fn      : callable(drift_severity, drift_score) -> None
                        provided by core.py, calls Trainer.run()
        """
        self.watch_config = watch_config
        self.buffer = buffer
        self.drift_monitor = drift_monitor
        self.train_fn = train_fn

        self._stop_event = threading.Event()
        self._thread: threading.Thread = None
        self._last_cron_fire: datetime = None
        self._cold_start_done: bool = False

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop,
            name=f"scheduler-{self.watch_config.metric}",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"[{self._metric_key()}] scheduler started cron='{self.watch_config.cron}'")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info(f"[{self._metric_key()}] scheduler stopped")

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as e:
                logger.error(f"[{self._metric_key()}] scheduler tick error: {e}")
            time.sleep(_TICK_INTERVAL_SECONDS)

    def _tick(self) -> None:
        now = datetime.now(timezone.utc)

        # gate everything on buffer readiness
        if not self.buffer.is_ready():
            pct = self.buffer.fill_fraction() * 100
            logger.debug(f"[{self._metric_key()}] buffer {pct:.0f}% full, waiting for cold start window")
            return

        # cold start — first training run once buffer is full
        if not self._cold_start_done:
            logger.info(f"[{self._metric_key()}] buffer ready, triggering cold start training")
            self._fire(DriftSeverity.NONE, 0.0)
            self._cold_start_done = True
            self._last_cron_fire = now
            return

        # drift check — higher priority than cron
        drift_result = self.drift_monitor.check()
        if drift_result.severity != DriftSeverity.NONE:
            logger.info(
                f"[{self._metric_key()}] drift detected severity={drift_result.severity} "
                f"mae={drift_result.mae:.4f}, triggering retrain"
            )
            self._fire(drift_result.severity, drift_result.mae)
            self.drift_monitor.reset()
            self._last_cron_fire = now
            return

        # cron check
        if self._cron_due(now):
            logger.info(f"[{self._metric_key()}] cron fired, triggering scheduled retrain")
            self._fire(DriftSeverity.NONE, 0.0)
            self._last_cron_fire = now

    def _fire(self, severity: str, drift_score: float) -> None:
        """
        Dispatch training job in a separate thread so the scheduler
        loop is never blocked by a long training run.
        """
        t = threading.Thread(
            target=self.train_fn,
            args=(severity, drift_score),
            name=f"trainer-{self.watch_config.metric}",
            daemon=True,
        )
        t.start()

    def _cron_due(self, now: datetime) -> bool:
        """
        Returns True if the cron schedule has fired since the last
        recorded fire time.
        """
        if self._last_cron_fire is None:
            return False
        try:
            secs = seconds_until_next_fire(self.watch_config.cron, base=self._last_cron_fire)
            elapsed = (now - self._last_cron_fire).total_seconds()
            return elapsed >= secs
        except Exception as e:
            logger.error(f"[{self._metric_key()}] cron check failed: {e}")
            return False

    def _metric_key(self) -> str:
        cfg = self.watch_config
        if not cfg.labels:
            return cfg.metric
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(cfg.labels.items()))
        return f"{cfg.metric}{{{label_str}}}"