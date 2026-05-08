# sentinel/ingestor/buffer.py

import threading
import numpy as np
from collections import deque
from datetime import datetime, timezone
from sentinel.utils.logging import get_logger
from sentinel.utils.time import parse_duration_to_seconds, parse_duration_to_steps

logger = get_logger(__name__)


class MetricBuffer:
    """
    Sliding window buffer for a single metric+labelset.
    Holds (timestamp, value) pairs up to the configured lookback window.
    Thread-safe — ingestor writes, trainer and drift monitor read concurrently.

    The buffer is the single source of truth for raw metric history inside Sentinel.
    Training, feature engineering, and drift detection all read from here.
    """

    def __init__(self, metric: str, lookback: str, granularity: str):
        """
        metric      : metric name, used for logging only
        lookback    : maximum duration to retain e.g. "30m"
        granularity : expected resolution of incoming data e.g. "1m"
        """
        self.metric = metric
        self.lookback = lookback
        self.granularity = granularity

        self._lookback_secs = parse_duration_to_seconds(lookback)
        self._max_steps = parse_duration_to_steps(lookback, granularity)

        # deque with maxlen automatically evicts oldest entries
        self._data: deque[tuple[float, float]] = deque(maxlen=self._max_steps)
        self._lock = threading.RLock()

    def push(self, timestamp: float, value: float) -> None:
        """
        Append a single (timestamp, value) pair.
        Oldest entries are evicted automatically once maxlen is reached.
        """
        with self._lock:
            self._data.append((timestamp, value))

    def push_many(self, samples: list[tuple[float, float]]) -> None:
        """
        Append a batch of (timestamp, value) pairs in order.
        Used during cold start when pulling historical range from Prometheus.
        """
        with self._lock:
            for ts, val in samples:
                self._data.append((ts, val))
        logger.debug(f"[{self.metric}] buffer loaded {len(samples)} samples")

    def get_values(self) -> np.ndarray:
        """
        Returns a numpy array of values in chronological order.
        """
        with self._lock:
            return np.array([v for _, v in self._data], dtype=float)

    def get_timestamps(self) -> np.ndarray:
        """
        Returns a numpy array of unix timestamps in chronological order.
        """
        with self._lock:
            return np.array([ts for ts, _ in self._data], dtype=float)

    def get_samples(self) -> list[tuple[float, float]]:
        """
        Returns all (timestamp, value) pairs as a list.
        """
        with self._lock:
            return list(self._data)

    def get_recent(self, n: int) -> np.ndarray:
        """
        Returns the n most recent values.
        If fewer than n values exist, returns all available.
        """
        with self._lock:
            data = list(self._data)
        recent = data[-n:] if len(data) >= n else data
        return np.array([v for _, v in recent], dtype=float)

    def is_ready(self) -> bool:
        """
        Returns True if the buffer has accumulated enough data
        to fill the full lookback window.
        Trainer checks this before triggering cold start training.
        """
        with self._lock:
            return len(self._data) >= self._max_steps

    def current_size(self) -> int:
        with self._lock:
            return len(self._data)

    def capacity(self) -> int:
        return self._max_steps

    def fill_fraction(self) -> float:
        """
        Returns how full the buffer is as a fraction between 0.0 and 1.0.
        Useful for logging cold start progress.
        """
        with self._lock:
            return len(self._data) / self._max_steps

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
        logger.debug(f"[{self.metric}] buffer cleared")

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __repr__(self) -> str:
        return (
            f"MetricBuffer(metric={self.metric}, "
            f"size={self.current_size()}/{self.capacity()}, "
            f"lookback={self.lookback}, "
            f"granularity={self.granularity})"
        )


class BufferRegistry:
    """
    Holds one MetricBuffer per watched metric.
    Core.py creates this once and passes it to both the ingestor and pipeline.
    """

    def __init__(self):
        self._buffers: dict[str, MetricBuffer] = {}
        self._lock = threading.Lock()

    def register(
        self,
        key: str,
        metric: str,
        lookback: str,
        granularity: str,
    ) -> MetricBuffer:
        """
        Register a new buffer for a metric.
        key is typically metric_name + label fingerprint.
        Returns the created buffer.
        """
        with self._lock:
            if key in self._buffers:
                logger.warning(f"Buffer already registered for key '{key}', returning existing.")
                return self._buffers[key]
            buf = MetricBuffer(metric=metric, lookback=lookback, granularity=granularity)
            self._buffers[key] = buf
            logger.debug(f"Registered buffer for key '{key}'")
            return buf

    def get(self, key: str) -> MetricBuffer | None:
        with self._lock:
            return self._buffers.get(key)

    def all_ready(self) -> bool:
        """
        Returns True only if every registered buffer has reached capacity.
        Used by scheduler to gate cold start training.
        """
        with self._lock:
            return all(buf.is_ready() for buf in self._buffers.values())

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._buffers.keys())

    def __repr__(self) -> str:
        with self._lock:
            summaries = ", ".join(
                f"{k}({buf.fill_fraction():.0%})"
                for k, buf in self._buffers.items()
            )
        return f"BufferRegistry([{summaries}])"