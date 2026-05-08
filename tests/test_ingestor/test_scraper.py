# tests/test_ingestor/test_scraper.py

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from sentinel.ingestor.scraper import PrometheusScraper


def _make_scraper():
    return PrometheusScraper(prometheus_url="http://localhost:9090", timeout=5)


def _mock_range_response(values: list):
    return {
        "status": "success",
        "data": {
            "result": [
                {"values": [[ts, str(val)] for ts, val in values]}
            ]
        }
    }


def _mock_instant_response(ts: float, val: float):
    return {
        "status": "success",
        "data": {
            "result": [
                {"value": [ts, str(val)]}
            ]
        }
    }


class TestBuildSelector:

    def test_no_labels(self):
        scraper = _make_scraper()
        assert scraper._build_selector("my_metric", {}) == "my_metric"

    def test_with_single_label(self):
        scraper = _make_scraper()
        result = scraper._build_selector("my_metric", {"job": "api"})
        assert result == 'my_metric{job="api"}'

    def test_with_multiple_labels_sorted(self):
        scraper = _make_scraper()
        result = scraper._build_selector("my_metric", {"job": "api", "env": "prod"})
        assert result == 'my_metric{env="prod",job="api"}'

    def test_labels_always_sorted(self):
        scraper = _make_scraper()
        r1 = scraper._build_selector("m", {"b": "2", "a": "1"})
        r2 = scraper._build_selector("m", {"a": "1", "b": "2"})
        assert r1 == r2


class TestFetchRange:

    def test_returns_samples_on_success(self):
        scraper = _make_scraper()
        mock_response = MagicMock()
        mock_response.json.return_value = _mock_range_response(
            [(1700000060.0, 1.5), (1700000120.0, 2.0)]
        )
        mock_response.raise_for_status = MagicMock()

        with patch.object(scraper._client, "get", return_value=mock_response):
            result = scraper.fetch_range(
                metric="http_requests_total",
                labels={"job": "api"},
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
                granularity="1m",
            )

        assert len(result) == 2
        assert result[0] == (1700000060.0, 1.5)
        assert result[1] == (1700000120.0, 2.0)

    def test_returns_empty_on_no_results(self):
        scraper = _make_scraper()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {"result": []}
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(scraper._client, "get", return_value=mock_response):
            result = scraper.fetch_range(
                metric="my_metric",
                labels={},
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
                granularity="1m",
            )

        assert result == []

    def test_returns_empty_on_non_success_status(self):
        scraper = _make_scraper()
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "error", "error": "bad query"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(scraper._client, "get", return_value=mock_response):
            result = scraper.fetch_range(
                metric="my_metric",
                labels={},
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
                granularity="1m",
            )

        assert result == []

    def test_returns_empty_on_http_error(self):
        import httpx
        scraper = _make_scraper()

        with patch.object(scraper._client, "get", side_effect=httpx.RequestError("timeout")):
            result = scraper.fetch_range(
                metric="my_metric",
                labels={},
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
                granularity="1m",
            )

        assert result == []

    def test_values_are_floats(self):
        scraper = _make_scraper()
        mock_response = MagicMock()
        mock_response.json.return_value = _mock_range_response(
            [(1700000060, 42)]
        )
        mock_response.raise_for_status = MagicMock()

        with patch.object(scraper._client, "get", return_value=mock_response):
            result = scraper.fetch_range(
                metric="m",
                labels={},
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
                granularity="1m",
            )

        ts, val = result[0]
        assert isinstance(ts, float)
        assert isinstance(val, float)

    def test_uses_first_series_only(self):
        scraper = _make_scraper()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "result": [
                    {"values": [[1.0, "10"]]},
                    {"values": [[2.0, "20"]]},  # second series, should be ignored
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(scraper._client, "get", return_value=mock_response):
            result = scraper.fetch_range(
                metric="m", labels={},
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
                granularity="1m",
            )

        assert len(result) == 1
        assert result[0] == (1.0, 10.0)


class TestFetchLatest:

    def test_returns_tuple_on_success(self):
        scraper = _make_scraper()
        mock_response = MagicMock()
        mock_response.json.return_value = _mock_instant_response(1700000060.0, 3.14)
        mock_response.raise_for_status = MagicMock()

        with patch.object(scraper._client, "get", return_value=mock_response):
            result = scraper.fetch_latest("my_metric", {})

        assert result == (1700000060.0, 3.14)

    def test_returns_none_on_empty_result(self):
        scraper = _make_scraper()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {"result": []}
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(scraper._client, "get", return_value=mock_response):
            result = scraper.fetch_latest("my_metric", {})

        assert result is None

    def test_returns_none_on_request_error(self):
        import httpx
        scraper = _make_scraper()

        with patch.object(scraper._client, "get", side_effect=httpx.RequestError("conn refused")):
            result = scraper.fetch_latest("my_metric", {})

        assert result is None

    def test_returns_none_on_non_success_status(self):
        scraper = _make_scraper()
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "error"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(scraper._client, "get", return_value=mock_response):
            result = scraper.fetch_latest("my_metric", {})

        assert result is None

    def test_values_are_floats(self):
        scraper = _make_scraper()
        mock_response = MagicMock()
        mock_response.json.return_value = _mock_instant_response(1700000000, 99)
        mock_response.raise_for_status = MagicMock()

        with patch.object(scraper._client, "get", return_value=mock_response):
            result = scraper.fetch_latest("my_metric", {})

        ts, val = result
        assert isinstance(ts, float)
        assert isinstance(val, float)


class TestCheckConnectivity:

    def test_returns_true_on_200(self):
        scraper = _make_scraper()
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(scraper._client, "get", return_value=mock_response):
            assert scraper.check_connectivity() is True

    def test_returns_false_on_non_200(self):
        scraper = _make_scraper()
        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch.object(scraper._client, "get", return_value=mock_response):
            assert scraper.check_connectivity() is False

    def test_returns_false_on_request_error(self):
        import httpx
        scraper = _make_scraper()

        with patch.object(scraper._client, "get", side_effect=httpx.RequestError("refused")):
            assert scraper.check_connectivity() is False