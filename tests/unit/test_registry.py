import pytest

from core.config import InferenceConfig, ModelConfig, RetrainConfig, WaitConfig
from core.registry import ModelInstance, ModelRegistry, ModelState


def make_config(name: str = "test-model") -> ModelConfig:
    return ModelConfig(
        name=name,
        path="",
        wait=WaitConfig(strategy="both", time_hours=6, rows=1000),
        retrain=RetrainConfig(schedule="0 2 * * *", min_rows=1000),
        inference=InferenceConfig(interval_seconds=60, timeout_seconds=30, fallback="continue_old"),
    )


def test_register_model_starts_in_waiting():
    registry = ModelRegistry()
    instance = registry.register(make_config("my-model"))
    assert instance.state == ModelState.WAITING
    assert instance.config.name == "my-model"


def test_get_returns_registered_model():
    registry = ModelRegistry()
    registry.register(make_config("my-model"))
    instance = registry.get("my-model")
    assert instance is not None
    assert instance.config.name == "my-model"


def test_get_unknown_model_returns_none():
    registry = ModelRegistry()
    assert registry.get("nonexistent") is None


def test_get_all_returns_all_models():
    registry = ModelRegistry()
    registry.register(make_config("model-a"))
    registry.register(make_config("model-b"))
    all_models = registry.get_all()
    names = {m.config.name for m in all_models}
    assert names == {"model-a", "model-b"}


def test_get_by_state_filters_correctly():
    registry = ModelRegistry()
    m1 = registry.register(make_config("model-a"))
    m2 = registry.register(make_config("model-b"))
    m2.state = ModelState.INFERENCING

    waiting = registry.get_by_state(ModelState.WAITING)
    inferencing = registry.get_by_state(ModelState.INFERENCING)

    assert len(waiting) == 1
    assert waiting[0].config.name == "model-a"
    assert len(inferencing) == 1
    assert inferencing[0].config.name == "model-b"


def test_transition_changes_state():
    registry = ModelRegistry()
    model = registry.register(make_config())
    assert model.state == ModelState.WAITING

    registry.transition(model, ModelState.TRAINING)
    assert model.state == ModelState.TRAINING

    registry.transition(model, ModelState.INFERENCING)
    assert model.state == ModelState.INFERENCING


def test_register_multiple_models_independently():
    registry = ModelRegistry()
    m1 = registry.register(make_config("a"))
    m2 = registry.register(make_config("b"))

    registry.transition(m1, ModelState.ERROR)
    assert m2.state == ModelState.WAITING