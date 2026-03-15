import logging
from typing import List

from core.buffer.schema import MetricRow

log = logging.getLogger(__name__)


def parse_remote_write(body: bytes) -> List[MetricRow]:
    """
    Decode a Prometheus remote_write request body into a list of MetricRow objects.

    Prometheus snappy-compresses the protobuf payload before sending.
    We decompress first, then parse the WriteRequest protobuf message.
    The __name__ label carries the metric name and is removed from the label set.
    Timestamps arrive in milliseconds and are converted to seconds.
    """
    import snappy
    from prometheus_client.exposition import _build_metric_family  # noqa - triggers proto registration
    from sentinel_pb2 import WriteRequest

    try:
        decompressed = snappy.decompress(body)
    except Exception as exc:
        raise ValueError(f"Failed to snappy-decompress payload: {exc}") from exc

    write_request = WriteRequest()
    write_request.ParseFromString(decompressed)

    rows: List[MetricRow] = []

    for time_series in write_request.timeseries:
        labels = {label.name: label.value for label in time_series.labels}
        metric_name = labels.pop("__name__", "unknown")

        for sample in time_series.samples:
            rows.append(MetricRow(
                metric_name=metric_name,
                labels=labels,
                timestamp=sample.timestamp / 1000.0,   # ms -> seconds
                value=sample.value,
            ))

    log.debug("Parsed %d rows from remote_write payload", len(rows))
    return rows