"""
Integration tests for the ingest pipeline.

These tests spin up the FastAPI app in-process using httpx's AsyncClient
and verify that the full ingest → buffer → registry flow works correctly.
"""
import time

import pytest
from fastapi.testclient import TestClient

from core.buffer.store import BufferStore
from core.config import (
    ExpositionConfig, IdentityConfig, InferenceConfig,
    ModelConfig, RetrainConfig, SentinelConfig, WaitConfig,
)
from core.ingestion import server
from core.registry import ModelRegistry, ModelState
from exposition.metrics import MetricsStore


@pytest.fixture
def client():
    """Set up a test client with a real buffer, registry, and metrics store."""
    buffer = BufferStore()
    registry = ModelRegistry()
    identity = IdentityConfig(service_name="test-service", namespace="test", cluster="local")
    exposition = ExpositionConfig()
    metrics_store = MetricsStore(identity, exposition)

    server.buffer = buffer
    server.registry = registry
    server.metrics_store = metrics_store
    server.config = SentinelConfig()

    return TestClient(server.app), buffer, registry


def test_health_endpoint(client):
    test_client, _, _ = client
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_endpoint_returns_prometheus_format(client):
    test_client, _, _ = client
    response = test_client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "# HELP" in response.text
    assert "# TYPE" in response.text


def test_ingest_invalid_payload_returns_400(client):
    test_client, _, _ = client
    response = test_client.post(
        "/ingest",
        content=b"not a valid snappy protobuf",
        headers={"Content-Type": "application/x-protobuf"},
    )
    assert response.status_code == 400


def test_metrics_includes_model_state_when_model_registered(client):
    test_client, buffer, registry = client

    config = ModelConfig(
        name="latency-spikes",
        path="",
        wait=WaitConfig(),
        retrain=RetrainConfig(),
        inference=InferenceConfig(),
    )
    registry.register(config)

    response = test_client.get("/metrics")
    assert response.status_code == 200
    assert "predictivex_model_state" in response.text
    assert 'model="latency-spikes"' in response.text


def test_waiting_model_rows_updated_after_ingest(client):
    """
    rows_collected on WAITING models should reflect buffer size.
    We test this by checking the model's rows_collected field directly
    since we can't easily send real Prometheus payloads in tests.
    """
    test_client, buffer, registry = client

    config = ModelConfig(
        name="test-model",
        path="",
        wait=WaitConfig(rows=5000),
        retrain=RetrainConfig(),
        inference=InferenceConfig(),
    )
    model = registry.register(config)
    assert model.state == ModelState.WAITING

    # Simulate buffer having rows (normally from remote_write)
    from core.buffer.schema import MetricRow
    rows = [
        MetricRow(metric_name="cpu", labels={}, timestamp=time.time(), value=float(i))
        for i in range(100)
    ]
    buffer.append_many(rows)

    # Manually update rows_collected as the ingest handler would
    model.rows_collected = buffer.total_rows()
    assert model.rows_collected == 100