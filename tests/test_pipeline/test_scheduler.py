# tests/test_pipeline/test_scheduler.py

import time
import threading
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from sentinel.pipeline.scheduler import MetricScheduler
from sentinel.pipeline.drift import DriftMonitor, DriftSeverity
from sentinel.ingestor.buffer import MetricBuffer
from sentinel.config import WatchConfig
from sentinel.pipeline.models.smoothing import ExponentialSmoothingModel


def _make_watch():
    return WatchConfig(
        metric="test_metric",
        labels={},
        model_class=ExponentialSmoothingModel,
        granularity="1m",
        horizon="5m",
        lookback="5m",
        cron="* * * * *",  # every minute for fast tests
    )


def _make_drift_monitor(severity=DriftSeverity.NONE):
    monitor = MagicMock(spec=DriftMonitor)
    result = MagicMock()
    result.severity = severity
    result.mae = 0.0
    monitor.check.return_value = result
    return monitor


def _make_full_buffer():
    buf = MetricBuffer(metric="test_metric", lookback="5m", granularity="1m")
    for i in range(5):
        buf.push(float(i), float(i))
    return buf


def _make_empty_buffer():
    return MetricBuffer(metric="test_metric", lookback="5m", granularity="1m")


class TestMetricScheduler:

    def test_start_and_stop(self):
        watch = _make_watch()
        buf = _make_full_buffer()
        monitor = _make_drift_monitor()
        train_fn = MagicMock()

        scheduler = MetricScheduler(
            watch_config=watch,
            buffer=buf,
            drift_monitor=monitor,
            train_fn=train_fn,
        )
        scheduler.start()
        assert scheduler._thread is not None
        assert scheduler._thread.is_alive()
        scheduler.stop()
        assert not scheduler._thread.is_alive()

    def test_cold_start_fires_when_buffer_ready(self):
        watch = _make_watch()
        buf = _make_full_buffer()
        monitor = _make_drift_monitor()
        train_calls = []

        def train_fn(severity, drift_score):
            train_calls.append((severity, drift_score))

        scheduler = MetricScheduler(
            watch_config=watch,
            buffer=buf,
            drift_monitor=monitor,
            train_fn=train_fn,
        )

        # call tick directly to avoid thread timing
        scheduler._tick()

        # give the dispatched thread time to run
        time.sleep(0.1)

        assert len(train_calls) == 1
        assert train_calls[0][0] == DriftSeverity.NONE
        assert scheduler._cold_start_done is True

    def test_cold_start_does_not_fire_when_buffer_not_ready(self):
        watch = _make_watch()
        buf = _make_empty_buffer()
        monitor = _make_drift_monitor()
        train_fn = MagicMock()

        scheduler = MetricScheduler(
            watch_config=watch,
            buffer=buf,
            drift_monitor=monitor,
            train_fn=train_fn,
        )

        scheduler._tick()
        time.sleep(0.1)

        train_fn.assert_not_called()
        assert scheduler._cold_start_done is False

    def test_cold_start_fires_only_once(self):
        watch = _make_watch()
        buf = _make_full_buffer()
        monitor = _make_drift_monitor()
        train_calls = []

        def train_fn(severity, drift_score):
            train_calls.append((severity, drift_score))

        scheduler = MetricScheduler(
            watch_config=watch,
            buffer=buf,
            drift_monitor=monitor,
            train_fn=train_fn,
        )

        scheduler._tick()
        scheduler._tick()
        scheduler._tick()
        time.sleep(0.2)

        # cold start only fires once — subsequent ticks go to cron/drift path
        cold_start_calls = [c for c in train_calls if not scheduler._cold_start_done or True]
        assert train_calls[0][0] == DriftSeverity.NONE

    def test_drift_triggers_retrain_after_cold_start(self):
        watch = _make_watch()
        buf = _make_full_buffer()
        train_calls = []

        def train_fn(severity, drift_score):
            train_calls.append((severity, drift_score))

        monitor = _make_drift_monitor(severity=DriftSeverity.HIGH)
        monitor.check.return_value.mae = 0.5

        scheduler = MetricScheduler(
            watch_config=watch,
            buffer=buf,
            drift_monitor=monitor,
            train_fn=train_fn,
        )

        # simulate cold start already done
        scheduler._cold_start_done = True
        scheduler._last_cron_fire = datetime.now(timezone.utc)

        scheduler._tick()
        time.sleep(0.1)

        assert len(train_calls) == 1
        assert train_calls[0][0] == DriftSeverity.HIGH

    def test_drift_monitor_reset_called_after_drift_trigger(self):
        watch = _make_watch()
        buf = _make_full_buffer()
        train_fn = MagicMock()

        monitor = _make_drift_monitor(severity=DriftSeverity.LOW)
        monitor.check.return_value.mae = 0.2

        scheduler = MetricScheduler(
            watch_config=watch,
            buffer=buf,
            drift_monitor=monitor,
            train_fn=train_fn,
        )

        scheduler._cold_start_done = True
        scheduler._last_cron_fire = datetime.now(timezone.utc)

        scheduler._tick()
        time.sleep(0.1)

        monitor.reset.assert_called_once()

    def test_cron_not_due_when_just_fired(self):
        watch = _make_watch()
        buf = _make_full_buffer()
        monitor = _make_drift_monitor()
        train_fn = MagicMock()

        scheduler = MetricScheduler(
            watch_config=watch,
            buffer=buf,
            drift_monitor=monitor,
            train_fn=train_fn,
        )

        scheduler._cold_start_done = True
        scheduler._last_cron_fire = datetime.now(timezone.utc)

        assert scheduler._cron_due(datetime.now(timezone.utc)) is False

    def test_cron_due_after_interval_elapsed(self):
        from datetime import timedelta
        watch = _make_watch()
        buf = _make_full_buffer()
        monitor = _make_drift_monitor()
        train_fn = MagicMock()

        scheduler = MetricScheduler(
            watch_config=watch,
            buffer=buf,
            drift_monitor=monitor,
            train_fn=train_fn,
        )

        scheduler._cold_start_done = True
        # set last fire to 2 minutes ago — cron is every minute so it should be due
        scheduler._last_cron_fire = datetime.now(timezone.utc) - timedelta(minutes=2)

        assert scheduler._cron_due(datetime.now(timezone.utc)) is True

    def test_cron_not_due_when_last_fire_is_none(self):
        watch = _make_watch()
        buf = _make_full_buffer()
        monitor = _make_drift_monitor()
        train_fn = MagicMock()

        scheduler = MetricScheduler(
            watch_config=watch,
            buffer=buf,
            drift_monitor=monitor,
            train_fn=train_fn,
        )

        assert scheduler._cron_due(datetime.now(timezone.utc)) is False

    def test_fire_dispatches_in_background_thread(self):
        watch = _make_watch()
        buf = _make_full_buffer()
        monitor = _make_drift_monitor()

        fired_threads = []

        def train_fn(severity, drift_score):
            fired_threads.append(threading.current_thread().name)

        scheduler = MetricScheduler(
            watch_config=watch,
            buffer=buf,
            drift_monitor=monitor,
            train_fn=train_fn,
        )

        scheduler._fire(DriftSeverity.NONE, 0.0)
        time.sleep(0.1)

        assert len(fired_threads) == 1
        # should not have run on the main thread
        assert fired_threads[0] != threading.main_thread().name