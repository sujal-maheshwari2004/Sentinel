from dataclasses import dataclass
from typing import Dict


@dataclass
class MetricRow:
    """
    A single metric sample received from Prometheus remote_write.

    metric_name : the metric name e.g. http_request_duration_seconds
    labels      : key/value pairs e.g. {"job": "api", "instance": "host:port"}
    timestamp   : unix epoch in seconds
    value       : the sample value
    """
    metric_name: str
    labels: Dict[str, str]
    timestamp: float
    value: float