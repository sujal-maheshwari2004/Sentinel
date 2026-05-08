# tests/test_pipeline/test_predictor.py

import pytest
import numpy as np
from sentinel.pipeline.versioning import VersionStore
from sentinel.pipeline.registry import ModelRegistry
from sentinel.pipeline.trainer import Trainer
from sentinel.pipeline.predictor import Predictor
from sentinel.ingestor.buffer import MetricBuffer
from sentinel.config import WatchConfig
from sentinel.pipeline.models.smoothing import ExponentialSmoothingModel
from sentinel.pipeline.drift import DriftSeverity


def _make_watch():
    return WatchConfig(
        metric="test_metric",
        labels={},
        model_class=ExponentialSmoothingModel,
        granularity="1m",
        horizon="5m",
        lookback="30m",
        cron="0 */6 * * *",
    )


def _fill_buffer(buf, n=35):
    for i in range(n):
        buf.push(float(1700000000 + i * 60), float(i) + np.random.normal(0, 0.1))


@pytest.fixture
def trained_setup(tmp_path):
    watch = _make_watch()
    buf = MetricBuffer(metric="test_metric", lookback="30m", granularity="1m")
    _fill_buffer(buf, n=35)

    store = VersionStore(
        metric_key="test_metric",
        artifact_store=str(tmp_path),
        max_versions=5,
    )
    registry = ModelRegistry(metric_key="test_metric", version_store=store)
    trainer = Trainer(watch_config=watch, buffer=buf, registry=registry)
    trainer.run(drift_severity=DriftSeverity.NONE)

    return watch, buf, registry


class TestPredictor:

    def test_returns_none_when_model_not_ready(self, tmp_path):
        watch = _make_watch()
        buf = MetricBuffer(metric="test_metric", lookback="30m", granularity="1m")
        _fill_buffer(buf, n=35)
        store = VersionStore(
            metric_key="test_metric",
            artifact_store=str(tmp_path),
            max_versions=5,
        )
        registry = ModelRegistry(metric_key="test_metric", version_store=store)
        predictor = Predictor(watch_config=watch, buffer=buf, registry=registry)
        result = predictor.predict()
        assert result is None

    def test_returns_prediction_result_after_training(self, trained_setup):
        watch, buf, registry = trained_setup
        predictor = Predictor(watch_config=watch, buffer=buf, registry=registry)
        result = predictor.predict()
        assert result is not None

    def test_prediction_values_length_equals_horizon_steps(self, trained_setup):
        watch, buf, registry = trained_setup
        predictor = Predictor(watch_config=watch, buffer=buf, registry=registry)
        result = predictor.predict()
        # horizon=5m, granularity=1m -> 5 steps
        assert len(result.values) == 5

    def test_prediction_timestamps_in_future(self, trained_setup):
        import time
        watch, buf, registry = trained_setup
        predictor = Predictor(watch_config=watch, buffer=buf, registry=registry)
        result = predictor.predict()
        now = time.time()
        assert all(ts > now for ts in result.timestamps)

    def test_prediction_has_model_version(self, trained_setup):
        watch, buf, registry = trained_setup
        predictor = Predictor(watch_config=watch, buffer=buf, registry=registry)
        result = predictor.predict()
        assert result.model_version is not None

    def test_returns_none_when_buffer_insufficient(self, tmp_path):
        watch = _make_watch()
        buf = MetricBuffer(metric="test_metric", lookback="30m", granularity="1m")
        buf.push(1.0, 1.0)  # only one sample

        store = VersionStore(
            metric_key="test_metric",
            artifact_store=str(tmp_path),
            max_versions=5,
        )
        registry = ModelRegistry(metric_key="test_metric", version_store=store)
        predictor = Predictor(watch_config=watch, buffer=buf, registry=registry)
        result = predictor.predict()
        assert result is None