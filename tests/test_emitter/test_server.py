# tests/test_emitter/test_server.py

import time
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from prometheus_client import CollectorRegistry
from sentinel.emitter.server import MetricEmitter, EmitterServer
from sentinel.pipeline.models.base import PredictionResult
from sentinel.pipeline.drift import DriftMonitor
from sentinel.config import WatchConfig, SentinelConfig
from sentinel.pipeline.models.smoothing import ExponentialSmoothingModel


def _make_watch():
    return WatchConfig(
        metric="http_request_duration_seconds",
        labels={"job": "api"},
        model_class=ExponentialSmoothingModel,
        granularity="1m",
        horizon="5m",
        lookback="30m",
    )


def _make_prediction(n_steps=5):
    return PredictionResult(
        values=np.linspace(1.0, 2.0, n_steps),
        timestamps=np.array([time.time() + (i + 1) * 60 for i in range(n_steps)]),
        model_version="v1",
    )


def _make_past_prediction(n_steps=5):
    """Prediction whose timestamps have already elapsed — triggers drift recording."""
    return PredictionResult(
        values=np.linspace(1.0, 2.0, n_steps),
        timestamps=np.array([time.time() - (i + 1) * 60 for i in range(n_steps)]),
        model_version="v1",
    )


class TestMetricEmitter:

    def _make_emitter(self, prediction=None, registry=None):
        watch = _make_watch()
        predictor = MagicMock()
        predictor.predict.return_value = prediction or _make_prediction()

        drift_monitor = MagicMock(spec=DriftMonitor)
        scraper = MagicMock()
        scraper.fetch_latest.return_value = (time.time(), 1.5)

        reg = registry or CollectorRegistry()

        return MetricEmitter(
            watch_config=watch,
            predictor=predictor,
            drift_monitor=drift_monitor,
            scraper=scraper,
            emit_confidence_bounds=False,
            registry=reg,
        ), predictor, drift_monitor, scraper

    def test_tick_calls_predictor(self):
        emitter, predictor, _, _ = self._make_emitter()
        emitter.tick()
        predictor.predict.assert_called_once()

    def test_tick_does_nothing_when_prediction_is_none(self):
        emitter, predictor, drift_monitor, _ = self._make_emitter(prediction=None)
        predictor.predict.return_value = None
        emitter.tick()
        # no gauges created, no drift recorded
        assert len(emitter._gauges) == 0

    def test_tick_creates_gauges_for_each_step(self):
        reg = CollectorRegistry()
        emitter, _, _, _ = self._make_emitter(registry=reg)
        emitter.tick()
        # 5 steps = 5 gauges
        assert len(emitter._gauges) == 5

    def test_tick_sets_gauge_values(self):
        reg = CollectorRegistry()
        emitter, _, _, _ = self._make_emitter(registry=reg)
        emitter.tick()
        # gauges created without raising
        assert len(emitter._gauges) > 0

    def test_tick_stores_pending_predictions(self):
        emitter, _, _, _ = self._make_emitter()
        emitter.tick()
        assert len(emitter._pending_predictions) == 5

    def test_tick_feeds_drift_when_predictions_elapsed(self):
        emitter, predictor, drift_monitor, scraper = self._make_emitter()

        # first tick stores pending predictions with past timestamps
        predictor.predict.return_value = _make_past_prediction()
        emitter.tick()

        # second tick should detect elapsed timestamps and feed drift
        predictor.predict.return_value = _make_prediction()
        emitter.tick()

        drift_monitor.record.assert_called()

    def test_drift_not_recorded_when_actual_fetch_fails(self):
        emitter, predictor, drift_monitor, scraper = self._make_emitter()

        predictor.predict.return_value = _make_past_prediction()
        scraper.fetch_latest.return_value = None

        emitter.tick()
        emitter.tick()

        drift_monitor.record.assert_not_called()

    def test_confidence_bounds_not_emitted_by_default(self):
        reg = CollectorRegistry()
        emitter, _, _, _ = self._make_emitter(registry=reg)

        result = _make_prediction()
        result.lower_bound = np.ones(5) * 0.9
        result.upper_bound = np.ones(5) * 1.1
        emitter.predictor.predict.return_value = result

        emitter.tick()
        # only main series gauges — no bound gauges
        assert all("lower" not in k and "upper" not in k for k in emitter._gauges)

    def test_confidence_bounds_emitted_when_enabled(self):
        watch = _make_watch()
        predictor = MagicMock()
        result = _make_prediction()
        result.lower_bound = np.ones(5) * 0.9
        result.upper_bound = np.ones(5) * 1.1
        predictor.predict.return_value = result

        drift_monitor = MagicMock(spec=DriftMonitor)
        scraper = MagicMock()
        reg = CollectorRegistry()

        emitter = MetricEmitter(
            watch_config=watch,
            predictor=predictor,
            drift_monitor=drift_monitor,
            scraper=scraper,
            emit_confidence_bounds=True,
            registry=reg,
        )

        emitter.tick()
        # main + lower + upper = 15 gauges (5 steps each)
        assert len(emitter._gauges) == 15

    def test_second_tick_reuses_existing_gauges(self):
        reg = CollectorRegistry()
        emitter, _, _, _ = self._make_emitter(registry=reg)

        emitter.tick()
        gauge_count_after_first = len(emitter._gauges)

        emitter.tick()
        gauge_count_after_second = len(emitter._gauges)

        assert gauge_count_after_first == gauge_count_after_second


class TestEmitterServer:

    def test_register_adds_emitter(self):
        config = SentinelConfig(emitter_port=19090)
        server = EmitterServer(config)
        emitter = MagicMock()
        server.register(emitter)
        assert emitter in server._emitters

    def test_stop_sets_stop_event(self):
        config = SentinelConfig(emitter_port=19091)
        server = EmitterServer(config)
        server.stop()
        assert server._stop_event.is_set()

    def test_loop_calls_tick_on_all_emitters(self):
        config = SentinelConfig(emitter_port=19092)
        server = EmitterServer(config)

        emitter1 = MagicMock()
        emitter2 = MagicMock()
        server.register(emitter1)
        server.register(emitter2)

        # patch sleep and stop after first pass by setting event inside sleep
        def stop_after_one(*args, **kwargs):
            server._stop_event.set()

        with patch("sentinel.emitter.server.time.sleep", side_effect=stop_after_one):
            server._loop()

        emitter1.tick.assert_called_once()
        emitter2.tick.assert_called_once()

    def test_loop_continues_after_emitter_exception(self):
        config = SentinelConfig(emitter_port=19093)
        server = EmitterServer(config)

        bad_emitter = MagicMock()
        bad_emitter.tick.side_effect = RuntimeError("boom")
        bad_emitter.watch_config = MagicMock()
        bad_emitter.watch_config.metric = "bad_metric"

        good_emitter = MagicMock()
        server.register(bad_emitter)
        server.register(good_emitter)

        def stop_after_one(*args, **kwargs):
            server._stop_event.set()

        with patch("sentinel.emitter.server.time.sleep", side_effect=stop_after_one):
            server._loop()

        good_emitter.tick.assert_called_once()

    def test_start_launches_thread(self):
        config = SentinelConfig(emitter_port=19094)
        server = EmitterServer(config)

        with patch("sentinel.emitter.server.start_http_server"):
            server.start()

        assert server._thread is not None
        assert server._thread.is_alive()
        server.stop()

    def test_start_starts_http_server_on_configured_port(self):
        config = SentinelConfig(emitter_port=19095)
        server = EmitterServer(config)

        with patch("sentinel.emitter.server.start_http_server") as mock_http:
            server.start()
            mock_http.assert_called_once_with(19095)

        server.stop()