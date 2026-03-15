import pytest

from core.config import ExpositionConfig, IdentityConfig, InferenceConfig, ModelConfig, RetrainConfig, WaitConfig
from core.registry import ModelInstance, ModelState
from exposition.metrics import MetricsStore
from models.base import Prediction


def make_identity() -> IdentityConfig:
    return IdentityConfig(
        service_name="api-gateway",
        namespace="prod",
        cluster="us-east-1",
        team="backend",
        extra_labels={},
    )


def make_exposition() -> ExpositionConfig:
    return ExpositionConfig(
        max_series_total=10000,
        cardinality_warning_threshold=5000,
        drop_high_cardinality_labels=[],
    )


def make_model_instance(name: str = "latency-spikes", state: ModelState = ModelState.INFERENCING) -> ModelInstance:
    config = ModelConfig(
        name=name,
        path="",
        wait=WaitConfig(),
        retrain=RetrainConfig(),
        inference=InferenceConfig(),
    )
    instance = ModelInstance(config=config, state=state)
    return instance


def test_render_empty_store_returns_valid_prometheus():
    store = MetricsStore(make_identity(), make_exposition())
    output = store.render()
    assert "# HELP" in output
    assert "# TYPE" in output


def test_update_and_render_predictions():
    store = MetricsStore(make_identity(), make_exposition())
    predictions = [
        Prediction(service="api-gateway", metric="latency_p99", score=0.83, horizon_seconds=600)
    ]
    store.update("latency-spikes", predictions)
    output = store.render()

    assert "predictivex_failure_probability" in output
    assert "0.8300" in output
    assert 'model="latency-spikes"' in output


def test_identity_labels_injected_into_all_series():
    store = MetricsStore(make_identity(), make_exposition())
    predictions = [Prediction(service="api-gateway", metric="latency_p99", score=0.5, horizon_seconds=600)]
    store.update("latency-spikes", predictions)
    output = store.render()

    assert 'service="api-gateway"' in output
    assert 'namespace="prod"' in output
    assert 'cluster="us-east-1"' in output


def test_clear_removes_predictions():
    store = MetricsStore(make_identity(), make_exposition())
    predictions = [Prediction(service="api-gateway", metric="latency_p99", score=0.9, horizon_seconds=600)]
    store.update("latency-spikes", predictions)
    store.clear("latency-spikes")

    output = store.render()
    assert "0.9000" not in output


def test_model_lifecycle_metrics_rendered():
    store = MetricsStore(make_identity(), make_exposition())
    model = make_model_instance("latency-spikes", ModelState.INFERENCING)
    output = store.render(models=[model])

    assert "predictivex_model_state" in output
    assert 'model="latency-spikes"' in output
    assert str(ModelState.INFERENCING.value) in output


def test_warmup_progress_only_rendered_when_waiting():
    store = MetricsStore(make_identity(), make_exposition())

    waiting_model = make_model_instance("latency-spikes", ModelState.WAITING)
    waiting_model.rows_collected = 500
    waiting_model.config.wait.rows = 1000

    output = store.render(models=[waiting_model])
    assert "predictivex_model_warmup_progress" in output
    assert "0.5000" in output


def test_warmup_progress_not_rendered_when_inferencing():
    store = MetricsStore(make_identity(), make_exposition())
    model = make_model_instance("latency-spikes", ModelState.INFERENCING)
    output = store.render(models=[model])

    # warmup progress only emitted for WAITING models
    assert "predictivex_model_warmup_progress" not in output


def test_cardinality_warning_not_triggered_below_threshold():
    store = MetricsStore(make_identity(), make_exposition())
    predictions = [Prediction(service="api-gateway", metric="latency_p99", score=0.1, horizon_seconds=600)]
    store.update("latency-spikes", predictions)
    output = store.render()

    assert "predictivex_cardinality_warning" in output
    assert "} 0" in output   # warning flag is 0


def test_multiple_models_render_independently():
    store = MetricsStore(make_identity(), make_exposition())
    store.update("latency-spikes", [
        Prediction(service="api-gateway", metric="latency_p99", score=0.8, horizon_seconds=600)
    ])
    store.update("cpu-exhaustion", [
        Prediction(service="api-gateway", metric="cpu_rate", score=0.3, horizon_seconds=600)
    ])
    output = store.render()

    assert 'model="latency-spikes"' in output
    assert 'model="cpu-exhaustion"' in output
    assert "0.8000" in output
    assert "0.3000" in output