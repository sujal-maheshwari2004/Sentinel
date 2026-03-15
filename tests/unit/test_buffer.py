import time

import pytest

from core.buffer.schema import MetricRow
from core.buffer.store import BufferStore


def make_row(metric_name: str, value: float, timestamp: float = None) -> MetricRow:
    return MetricRow(
        metric_name=metric_name,
        labels={"job": "test"},
        timestamp=timestamp or time.time(),
        value=value,
    )


def test_append_and_get():
    store = BufferStore()
    store.append(make_row("cpu", 0.5))
    rows = store.get("cpu")
    assert len(rows) == 1
    assert rows[0].value == 0.5


def test_append_many():
    store = BufferStore()
    rows = [make_row("cpu", float(i)) for i in range(5)]
    store.append_many(rows)
    assert store.total_rows() == 5


def test_get_returns_copy():
    store = BufferStore()
    store.append(make_row("cpu", 1.0))
    rows = store.get("cpu")
    rows.clear()
    assert store.total_rows() == 1


def test_get_all_returns_copy():
    store = BufferStore()
    store.append(make_row("cpu", 1.0))
    store.append(make_row("mem", 2.0))
    data = store.get_all()
    data.clear()
    assert store.total_rows() == 2


def test_get_missing_metric_returns_empty():
    store = BufferStore()
    assert store.get("nonexistent") == []


def test_metric_names():
    store = BufferStore()
    store.append(make_row("cpu", 1.0))
    store.append(make_row("mem", 2.0))
    names = store.metric_names()
    assert set(names) == {"cpu", "mem"}


def test_evict_old_samples():
    store = BufferStore(max_age_seconds=10)
    old_time = time.time() - 100
    recent_time = time.time()
    store.append(make_row("cpu", 1.0, timestamp=old_time))
    store.append(make_row("cpu", 2.0, timestamp=recent_time))
    removed = store.evict_old_samples()
    assert removed == 1
    assert store.total_rows() == 1
    assert store.get("cpu")[0].value == 2.0


def test_evict_keeps_all_recent():
    store = BufferStore(max_age_seconds=3600)
    store.append_many([make_row("cpu", float(i)) for i in range(10)])
    removed = store.evict_old_samples()
    assert removed == 0
    assert store.total_rows() == 10


def test_total_rows_across_metrics():
    store = BufferStore()
    store.append_many([make_row("cpu", 1.0)] * 3)
    store.append_many([make_row("mem", 2.0)] * 5)
    assert store.total_rows() == 8