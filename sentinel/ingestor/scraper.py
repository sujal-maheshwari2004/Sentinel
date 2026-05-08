# sentinel/ingestor/scraper.py

import time
import httpx
from datetime import datetime, timezone
from sentinel.utils.logging import get_logger
from sentinel.utils.time import (
    parse_duration_to_seconds,
    align_timestamp_to_granularity,
)

logger = get_logger(__name__)


class PrometheusScraper:
    """
    Pulls time series data from Prometheus HTTP API.
    One scraper instance is shared across all watched metrics.
    Uses httpx for sync HTTP — keeps it simple, no async complexity
    since scraping runs in its own background thread.
    """

    def __init__(self, prometheus_url: str, timeout: int = 10):
        self.prometheus_url = prometheus_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=self.timeout)

    def fetch_range(
        self,
        metric: str,
        labels: dict[str, str],
        start: datetime,
        end: datetime,
        granularity: str,
    ) -> list[tuple[float, float]]:
        """
        Fetch a range of samples for a metric+label set from Prometheus.

        Returns a list of (unix_timestamp, value) tuples sorted ascending.
        Empty list if no data found.

        metric      : metric name e.g. "http_request_duration_seconds"
        labels      : label filters e.g. {"job": "api", "status": "200"}
        start       : range start datetime (UTC)
        end         : range end datetime (UTC)
        granularity : step size e.g. "1m"
        """
        query = self._build_selector(metric, labels)
        step = granularity  # Prometheus accepts e.g. "1m", "30s" directly

        params = {
            "query": query,
            "start": start.timestamp(),
            "end": end.timestamp(),
            "step": step,
        }

        url = f"{self.prometheus_url}/api/v1/query_range"

        try:
            response = self._client.get(url, params=params)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Prometheus returned HTTP {e.response.status_code} "
                f"for query '{query}': {e.response.text}"
            )
            return []
        except httpx.RequestError as e:
            logger.error(f"Failed to reach Prometheus at {url}: {e}")
            return []

        data = response.json()

        if data.get("status") != "success":
            logger.warning(
                f"Prometheus query returned non-success status for '{query}': {data}"
            )
            return []

        results = data.get("data", {}).get("result", [])

        if not results:
            logger.debug(f"No data returned from Prometheus for query '{query}'")
            return []

        # take the first matching series
        # one model per metric+labelset so there should only be one
        values = results[0].get("values", [])

        return [(float(ts), float(val)) for ts, val in values]

    def fetch_latest(
        self,
        metric: str,
        labels: dict[str, str],
    ) -> tuple[float, float] | None:
        """
        Fetch the single most recent sample for a metric+label set.
        Used by the drift monitor and emitter to get current value.

        Returns (unix_timestamp, value) or None if no data.
        """
        query = self._build_selector(metric, labels)
        url = f"{self.prometheus_url}/api/v1/query"
        params = {
            "query": query,
            "time": datetime.now(timezone.utc).timestamp(),
        }

        try:
            response = self._client.get(url, params=params)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Prometheus returned HTTP {e.response.status_code} "
                f"for instant query '{query}': {e.response.text}"
            )
            return None
        except httpx.RequestError as e:
            logger.error(f"Failed to reach Prometheus at {url}: {e}")
            return None

        data = response.json()

        if data.get("status") != "success":
            return None

        results = data.get("data", {}).get("result", [])

        if not results:
            return None

        ts, val = results[0].get("value", [None, None])

        if ts is None or val is None:
            return None

        return (float(ts), float(val))

    def check_connectivity(self) -> bool:
        """
        Ping Prometheus /-/healthy endpoint.
        Used at startup to fail fast if Prometheus is unreachable.
        """
        url = f"{self.prometheus_url}/-/healthy"
        try:
            response = self._client.get(url)
            return response.status_code == 200
        except httpx.RequestError:
            return False

    def close(self) -> None:
        self._client.close()

    def _build_selector(self, metric: str, labels: dict[str, str]) -> str:
        """
        Build a Prometheus instant vector selector string.

        Examples:
            "http_requests_total"
            'http_requests_total{job="api",status="200"}'
        """
        if not labels:
            return metric
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{metric}{{{label_str}}}"