# tests/test_ingestor/test_buffer.py

import time
import pytest
import numpy as np
from sentinel.ingestor.buffer import MetricBuffer, BufferRegistry


class TestMetricBuffer:

    def _make_buffer(self, lookback="5m", granularity="1m"):
        return MetricBuffer(metric="test_metric", lookback=lookback, granularity=granularity)

    def test_push_single(self):
        buf = self._make_buffer()
        buf.push(1000.0, 42.0)
        assert len(buf) == 1

    def test_push_many(self):
        buf = self._make_buffer()
        samples = [(float(i), float(i * 2)) for i in range(10)]
        buf.push_many(samples)
        assert len(buf) == 5  # maxlen is 5 (5m / 1m)

    def test_evicts_oldest_on_overflow(self):
        buf = self._make_buffer()
        for i in range(10):
            buf.push(float(i), float(i))
        values = buf.get_values()
        assert values[0] == 5.0  # first 5 evicted

    def test_get_values_returns_ndarray(self):
        buf = self._make_buffer()
        buf.push(1.0, 99.0)
        values = buf.get_values()
        assert isinstance(values, np.ndarray)
        assert values[0] == 99.0

    def test_get_timestamps_returns_ndarray(self):
        buf = self._make_buffer()
        buf.push(1234.0, 99.0)
        timestamps = buf.get_timestamps()
        assert isinstance(timestamps, np.ndarray)
        assert timestamps[0] == 1234.0

    def test_get_samples(self):
        buf = self._make_buffer()
        buf.push(1.0, 2.0)
        buf.push(3.0, 4.0)
        samples = buf.get_samples()
        assert samples == [(1.0, 2.0), (3.0, 4.0)]

    def test_get_recent(self):
        buf = self._make_buffer()
        for i in range(5):
            buf.push(float(i), float(i))
        recent = buf.get_recent(3)
        assert list(recent) == [2.0, 3.0, 4.0]

    def test_get_recent_fewer_than_n(self):
        buf = self._make_buffer()
        buf.push(1.0, 10.0)
        recent = buf.get_recent(10)
        assert len(recent) == 1

    def test_is_ready_false_when_not_full(self):
        buf = self._make_buffer()
        buf.push(1.0, 1.0)
        assert buf.is_ready() is False

    def test_is_ready_true_when_full(self):
        buf = self._make_buffer()
        for i in range(5):
            buf.push(float(i), float(i))
        assert buf.is_ready() is True

    def test_fill_fraction(self):
        buf = self._make_buffer()
        buf.push(1.0, 1.0)
        buf.push(2.0, 2.0)
        assert buf.fill_fraction() == pytest.approx(2 / 5)

    def test_fill_fraction_full(self):
        buf = self._make_buffer()
        for i in range(5):
            buf.push(float(i), float(i))
        assert buf.fill_fraction() == pytest.approx(1.0)

    def test_capacity(self):
        buf = self._make_buffer()
        assert buf.capacity() == 5

    def test_clear(self):
        buf = self._make_buffer()
        for i in range(5):
            buf.push(float(i), float(i))
        buf.clear()
        assert len(buf) == 0

    def test_thread_safety(self):
        import threading
        buf = self._make_buffer(lookback="10m", granularity="1m")
        errors = []

        def writer():
            try:
                for i in range(50):
                    buf.push(float(i), float(i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []


class TestBufferRegistry:

    def test_register_and_get(self):
        reg = BufferRegistry()
        buf = reg.register(key="m1", metric="m1", lookback="5m", granularity="1m")
        assert reg.get("m1") is buf

    def test_register_duplicate_returns_existing(self):
        reg = BufferRegistry()
        buf1 = reg.register(key="m1", metric="m1", lookback="5m", granularity="1m")
        buf2 = reg.register(key="m1", metric="m1", lookback="5m", granularity="1m")
        assert buf1 is buf2

    def test_get_missing_returns_none(self):
        reg = BufferRegistry()
        assert reg.get("nonexistent") is None

    def test_all_ready_false_when_empty_buffers(self):
        reg = BufferRegistry()
        reg.register(key="m1", metric="m1", lookback="5m", granularity="1m")
        assert reg.all_ready() is False

    def test_all_ready_true_when_all_full(self):
        reg = BufferRegistry()
        buf = reg.register(key="m1", metric="m1", lookback="5m", granularity="1m")
        for i in range(5):
            buf.push(float(i), float(i))
        assert reg.all_ready() is True

    def test_keys(self):
        reg = BufferRegistry()
        reg.register(key="m1", metric="m1", lookback="5m", granularity="1m")
        reg.register(key="m2", metric="m2", lookback="5m", granularity="1m")
        assert set(reg.keys()) == {"m1", "m2"}