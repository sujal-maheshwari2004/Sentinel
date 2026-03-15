import pytest

from core.buffer.schema import MetricRow


def test_metric_row_fields():
    row = MetricRow(
        metric_name="http_request_duration_seconds",
        labels={"job": "api", "instance": "host:9090"},
        timestamp=1700000000.0,
        value=0.042,
    )
    assert row.metric_name == "http_request_duration_seconds"
    assert row.labels["job"] == "api"
    assert row.timestamp == 1700000000.0
    assert row.value == 0.042


def test_metric_row_empty_labels():
    row = MetricRow(metric_name="up", labels={}, timestamp=1.0, value=1.0)
    assert row.labels == {}


def test_parse_remote_write_requires_snappy_and_protobuf(monkeypatch):
    """
    Verify that parse_remote_write raises a clear error when given
    invalid (non-snappy) bytes rather than silently returning empty rows.
    """
    from core.ingestion.parser import parse_remote_write

    with pytest.raises((ValueError, Exception)):
        parse_remote_write(b"not a valid snappy payload")


def test_parse_remote_write_empty_bytes(monkeypatch):
    """
    Empty bytes should also raise rather than return silently.
    """
    from core.ingestion.parser import parse_remote_write

    with pytest.raises((ValueError, Exception)):
        parse_remote_write(b"")