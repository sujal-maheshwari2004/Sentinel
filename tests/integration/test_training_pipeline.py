"""
Integration tests for the training pipeline.

These tests verify the wait threshold logic, artifact path resolution,
and hotswap behaviour without actually running subprocess scripts.
"""
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.config import InferenceConfig, ModelConfig, RetrainConfig, SentinelConfig, WaitConfig
from core.registry import ModelInstance, ModelRegistry, ModelState
from pipeline.hotswap.swapper import swap_artifact
from pipeline.training.trainer import _wait_threshold_met, _resolve_script


def make_model(strategy: str = "both", time_hours: int = 6, rows: int = 1000) -> ModelInstance:
    config = ModelConfig(
        name="test-model",
        path="",
        wait=WaitConfig(strategy=strategy, time_hours=time_hours, rows=rows),
        retrain=RetrainConfig(),
        inference=InferenceConfig(),
    )
    return ModelInstance(config=config)


# -----------------------------------------------------------------------------
# Wait threshold tests
# -----------------------------------------------------------------------------

def test_wait_strategy_rows_met():
    model = make_model(strategy="rows", rows=100)
    model.rows_collected = 100
    assert _wait_threshold_met(model) is True


def test_wait_strategy_rows_not_met():
    model = make_model(strategy="rows", rows=100)
    model.rows_collected = 50
    assert _wait_threshold_met(model) is False


def test_wait_strategy_time_met():
    model = make_model(strategy="time", time_hours=0)
    # waiting_since is set to now() on creation, 0 hours required → always met
    assert _wait_threshold_met(model) is True


def test_wait_strategy_time_not_met():
    model = make_model(strategy="time", time_hours=100)
    model.waiting_since = time.time()   # just started
    assert _wait_threshold_met(model) is False


def test_wait_strategy_both_requires_both():
    model = make_model(strategy="both", time_hours=100, rows=10)
    model.rows_collected = 100          # rows met
    model.waiting_since = time.time()   # time NOT met
    assert _wait_threshold_met(model) is False


def test_wait_strategy_both_met_when_both_satisfied():
    model = make_model(strategy="both", time_hours=0, rows=10)
    model.rows_collected = 100          # rows met
    # time_hours=0 → always met
    assert _wait_threshold_met(model) is True


# -----------------------------------------------------------------------------
# Script path resolution tests
# -----------------------------------------------------------------------------

def test_resolve_builtin_retrain_script():
    model = make_model()
    path = _resolve_script(model, "retrain.py")
    assert "models/builtin/test-model/retrain.py" in path


def test_resolve_custom_retrain_script():
    config = ModelConfig(
        name="custom",
        path="/opt/my-model",
        wait=WaitConfig(),
        retrain=RetrainConfig(),
        inference=InferenceConfig(),
    )
    model = ModelInstance(config=config)
    path = _resolve_script(model, "retrain.py")
    assert path == "/opt/my-model/retrain.py"


def test_resolve_infer_script():
    model = make_model()
    path = _resolve_script(model, "infer.py")
    assert "models/builtin/test-model/infer.py" in path


# -----------------------------------------------------------------------------
# Hotswap tests
# -----------------------------------------------------------------------------

def test_swap_artifact_first_install(tmp_path):
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"fake model")

    config = ModelConfig(name="test", path="", wait=WaitConfig(), retrain=RetrainConfig(), inference=InferenceConfig())
    model = ModelInstance(config=config)
    assert model.artifact_path is None

    swap_artifact(model, str(artifact))

    assert model.artifact_path == str(artifact)


def test_swap_artifact_replaces_existing(tmp_path):
    old_artifact = tmp_path / "old.pkl"
    old_artifact.write_bytes(b"old model")

    new_artifact = tmp_path / "new.pkl"
    new_artifact.write_bytes(b"new model")

    config = ModelConfig(name="test", path="", wait=WaitConfig(), retrain=RetrainConfig(), inference=InferenceConfig())
    model = ModelInstance(config=config, artifact_path=str(old_artifact))

    swap_artifact(model, str(new_artifact))

    # The old path now contains the new model content
    assert Path(model.artifact_path).read_bytes() == b"new model"


def test_swap_artifact_raises_if_new_artifact_missing(tmp_path):
    config = ModelConfig(name="test", path="", wait=WaitConfig(), retrain=RetrainConfig(), inference=InferenceConfig())
    model = ModelInstance(config=config)

    with pytest.raises(FileNotFoundError):
        swap_artifact(model, str(tmp_path / "nonexistent.pkl"))