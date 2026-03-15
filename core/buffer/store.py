import threading
import time
from collections import defaultdict
from typing import Dict, List

from core.buffer.schema import MetricRow


class BufferStore:
    """
    Thread-safe in-memory rolling window of metric samples.

    Samples are stored per metric name. Anything older than
    max_age_seconds is dropped during eviction, which runs on a schedule
    via the scheduler. The buffer is the source of truth for both
    snapshot flushing and inference.
    """

    def __init__(self, max_age_seconds: int = 86400):
        self.max_age_seconds = max_age_seconds
        self._data: Dict[str, List[MetricRow]] = defaultdict(list)
        self._lock = threading.Lock()

    def append(self, row: MetricRow) -> None:
        """Add a single metric sample to the buffer."""
        with self._lock:
            self._data[row.metric_name].append(row)

    def append_many(self, rows: List[MetricRow]) -> None:
        """Add a batch of metric samples. More efficient than calling append in a loop."""
        with self._lock:
            for row in rows:
                self._data[row.metric_name].append(row)

    def get(self, metric_name: str) -> List[MetricRow]:
        """Return all buffered samples for a specific metric name."""
        with self._lock:
            return list(self._data.get(metric_name, []))

    def get_all(self) -> Dict[str, List[MetricRow]]:
        """Return a copy of all buffered data keyed by metric name."""
        with self._lock:
            return {metric: list(rows) for metric, rows in self._data.items()}

    def total_rows(self) -> int:
        """Return the total number of samples across all metrics."""
        with self._lock:
            return sum(len(rows) for rows in self._data.values())

    def metric_names(self) -> List[str]:
        """Return the list of metric names currently in the buffer."""
        with self._lock:
            return list(self._data.keys())

    def evict_old_samples(self) -> int:
        """
        Remove samples older than max_age_seconds.
        Returns the number of samples removed.
        Called on a schedule by the scheduler.
        """
        cutoff = time.time() - self.max_age_seconds
        removed = 0

        with self._lock:
            for metric_name in list(self._data.keys()):
                before = len(self._data[metric_name])
                self._data[metric_name] = [
                    row for row in self._data[metric_name]
                    if row.timestamp >= cutoff
                ]
                removed += before - len(self._data[metric_name])

        return removed