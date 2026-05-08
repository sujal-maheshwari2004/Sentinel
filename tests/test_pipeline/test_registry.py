# tests/test_pipeline/test_registry.py

import os
import pytest
import tempfile
import numpy as np
from sentinel.pipeline.versioning import VersionStore, ModelVersion
from sentinel.pipeline.registry import ModelRegistry
from sentinel.pipeline.models.smoothing import ExponentialSmoothingModel


def _make_version(version_id="v1", mae=0.05):
    return ModelVersion(
        version_id=version_id,
        metric_key="test_metric",
        model_class="ExponentialSmoothingModel",
        trained_at="2024-01-01T00:00:00+00:00",
        training_policy="full_retrain",
        drift_score_at_trigger=0.0,
        mae=mae,
        mape=1.0,
        n_samples=100,
        artifact_path="",
        extra={"granularity": "1m", "horizon": "5m", "lookback": "30m"},
    )


def _make_model():
    return ExponentialSmoothingModel(
        granularity="1m",
        horizon="5m",
        lookback="30m",
    )


def _fit_model(model):
    y = np.linspace(1, 10, 50)
    X = np.zeros((50, 30))
    model.fit(X, y)
    return model


@pytest.fixture
def tmp_store(tmp_path):
    return VersionStore(
        metric_key="test_metric",
        artifact_store=str(tmp_path),
        max_versions=5,
    )


@pytest.fixture
def registry(tmp_store):
    return ModelRegistry(metric_key="test_metric", version_store=tmp_store)


class TestModelRegistry:

    def test_initial_not_ready(self, registry):
        assert registry.is_ready() is False

    def test_get_model_returns_none_initially(self, registry):
        assert registry.get_model() is None

    def test_promote_makes_registry_ready(self, registry):
        model = _fit_model(_make_model())
        version = _make_version()
        registry.promote(model, version)
        assert registry.is_ready() is True

    def test_promote_sets_active_version(self, registry):
        model = _fit_model(_make_model())
        version = _make_version(version_id="v1")
        registry.promote(model, version)
        assert registry.active_version().version_id == "v1"

    def test_promote_saves_artifact(self, registry, tmp_path):
        model = _fit_model(_make_model())
        version = _make_version(version_id="v1")
        registry.promote(model, version)
        assert os.path.exists(version.artifact_path)

    def test_get_model_returns_promoted_model(self, registry):
        model = _fit_model(_make_model())
        version = _make_version()
        registry.promote(model, version)
        assert registry.get_model() is model

    def test_rollback_with_no_previous_returns_false(self, registry):
        model = _fit_model(_make_model())
        version = _make_version(version_id="v1")
        registry.promote(model, version)
        result = registry.rollback()
        assert result is False

    def test_rollback_restores_previous(self, registry):
        model1 = _fit_model(_make_model())
        version1 = _make_version(version_id="v1")
        registry.promote(model1, version1)

        model2 = _fit_model(_make_model())
        version2 = _make_version(version_id="v2")
        registry.promote(model2, version2)

        assert registry.active_version().version_id == "v2"
        result = registry.rollback()
        assert result is True
        assert registry.active_version().version_id == "v1"

    def test_restore_from_disk_no_artifact(self, registry):
        result = registry.restore_from_disk(lambda: _make_model())
        assert result is False

    def test_restore_from_disk_after_promote(self, registry, tmp_store):
        model = _fit_model(_make_model())
        version = _make_version(version_id="v1")
        registry.promote(model, version)

        new_registry = ModelRegistry(
            metric_key="test_metric",
            version_store=tmp_store,
        )
        result = new_registry.restore_from_disk(lambda: _make_model())
        assert result is True
        assert new_registry.is_ready() is True