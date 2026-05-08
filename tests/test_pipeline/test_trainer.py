# tests/test_pipeline/test_trainer.py

import pytest
import numpy as np
from sentinel.pipeline.drift import DriftSeverity
from sentinel.pipeline.versioning import VersionStore
from sentinel.pipeline.registry import ModelRegistry
from sentinel.pipeline.trainer import Trainer
from sentinel.ingestor.buffer import MetricBuffer
from sentinel.config import WatchConfig
from sentinel.pipeline.models.smoothing import ExponentialSmoothingModel
from sentinel.pipeline.models.linear import LinearTrendModel


def _make_watch_config(model_class=None):
    return WatchConfig(
        metric="test_metric",
        labels={},
        model_class=model_class or ExponentialSmoothingModel,
        granularity="1m",
        horizon="5m",
        lookback="10m",
        cron="0 */6 * * *",
    )


def _fill_buffer(buf, n=35):
    for i in range(n):
        buf.push(float(1700000000 + i * 60), float(i) + np.random.normal(0, 0.1))


@pytest.fixture
def tmp_registry(tmp_path):
    store = VersionStore(
        metric_key="test_metric",
        artifact_store=str(tmp_path),
        max_versions=5,
    )
    return ModelRegistry(metric_key="test_metric", version_store=store)


@pytest.fixture
def buffer():
    # buffer capacity must be > n_lags (10), so use 20m lookback
    buf = MetricBuffer(metric="test_metric", lookback="20m", granularity="1m")
    for i in range(20):
        buf.push(float(1700000000 + i * 60), float(i) + np.random.normal(0, 0.1))
    return buf




class TestTrainer:

    def test_run_cold_start_promotes_model(self, buffer, tmp_registry):
        watch = _make_watch_config()
        trainer = Trainer(watch_config=watch, buffer=buffer, registry=tmp_registry)
        result = trainer.run(drift_severity=DriftSeverity.NONE, drift_score=0.0)
        assert result is not None
        assert tmp_registry.is_ready()

    def test_run_returns_training_result(self, buffer, tmp_registry):
        watch = _make_watch_config()
        trainer = Trainer(watch_config=watch, buffer=buffer, registry=tmp_registry)
        result = trainer.run()
        assert result is not None
        assert result.mae >= 0
        assert result.mape >= 0
        assert result.n_samples > 0

    def test_full_retrain_policy_on_high_drift(self, buffer, tmp_registry):
        watch = _make_watch_config()
        trainer = Trainer(watch_config=watch, buffer=buffer, registry=tmp_registry)
        trainer.run(drift_severity=DriftSeverity.NONE)
        result = trainer.run(drift_severity=DriftSeverity.HIGH, drift_score=0.5)
        assert result is not None
        assert result.training_policy == "full_retrain"

    def test_finetune_policy_on_low_drift(self, buffer, tmp_registry):
        watch = _make_watch_config(model_class=LinearTrendModel)
        trainer = Trainer(watch_config=watch, buffer=buffer, registry=tmp_registry)
        trainer.run(drift_severity=DriftSeverity.NONE)
        result = trainer.run(drift_severity=DriftSeverity.LOW, drift_score=0.15)
        assert result is not None
        assert result.training_policy == "finetune"

    def test_run_with_empty_buffer_returns_none(self, tmp_registry):
        watch = _make_watch_config()
        buf = MetricBuffer(metric="test_metric", lookback="30m", granularity="1m")
        trainer = Trainer(watch_config=watch, buffer=buf, registry=tmp_registry)
        result = trainer.run()
        assert result is None

    def test_version_increments_on_each_run(self, buffer, tmp_registry):
        watch = _make_watch_config()
        trainer = Trainer(watch_config=watch, buffer=buffer, registry=tmp_registry)
        trainer.run()
        trainer.run()
        versions = tmp_registry.version_store.get_all()
        assert len(versions) == 2