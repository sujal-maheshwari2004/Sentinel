# tests/test_emitter/test_formatter.py

import pytest
import numpy as np
from sentinel.emitter.formatter import (
    format_prediction,
    build_metric_key,
    FormattedPrediction,
)
from sentinel.pipeline.models.base import PredictionResult
from sentinel.config import WatchConfig
from sentinel.pipeline.models.smoothing import ExponentialSmoothingModel


def _make_watch(labels=None):
    return WatchConfig(
        metric="http_request_duration_seconds",
        labels=labels or {"job": "api"},
        model_class=ExponentialSmoothingModel,
        granularity="1m",
        horizon="5m",
        lookback="30m",
    )


def _make_result(n_steps=5, with_bounds=False):
    values = np.linspace(1.0, 2.0, n_steps)
    timestamps = np.array([1700000060.0 * (i + 1) for i in range(n_steps)])
    lower = values - 0.1 if with_bounds else None
    upper = values + 0.1 if with_bounds else None
    return PredictionResult(
        values=values,
        timestamps=timestamps,
        lower_bound=lower,
        upper_bound=upper,
        model_version="v1",
    )


class TestFormatPrediction:

    def test_returns_one_formatted_prediction_by_default(self):
        watch = _make_watch()
        result = _make_result()
        formatted = format_prediction(watch, result, emit_confidence_bounds=False)
        assert len(formatted) == 1

    def test_returns_three_with_confidence_bounds(self):
        watch = _make_watch()
        result = _make_result(with_bounds=True)
        formatted = format_prediction(watch, result, emit_confidence_bounds=True)
        assert len(formatted) == 3

    def test_metric_name_has_suffix(self):
        watch = _make_watch()
        result = _make_result()
        formatted = format_prediction(watch, result)
        assert formatted[0].metric_name == "http_request_duration_seconds_sentinel_predicted"

    def test_steps_count_matches_horizon(self):
        watch = _make_watch()
        result = _make_result(n_steps=5)
        formatted = format_prediction(watch, result)
        assert len(formatted[0].steps) == 5

    def test_steps_are_one_indexed(self):
        watch = _make_watch()
        result = _make_result(n_steps=3)
        formatted = format_prediction(watch, result)
        steps = [s["step"] for s in formatted[0].steps]
        assert steps == [1, 2, 3]

    def test_labels_carry_original_labels(self):
        watch = _make_watch(labels={"job": "api"})
        result = _make_result()
        formatted = format_prediction(watch, result)
        assert formatted[0].labels["job"] == "api"

    def test_labels_include_sentinel_metadata(self):
        watch = _make_watch()
        result = _make_result()
        formatted = format_prediction(watch, result)
        assert "sentinel_horizon" in formatted[0].labels
        assert "sentinel_version" in formatted[0].labels

    def test_sentinel_version_matches_result(self):
        watch = _make_watch()
        result = _make_result()
        result.model_version = "v3"
        formatted = format_prediction(watch, result)
        assert formatted[0].labels["sentinel_version"] == "v3"

    def test_no_bounds_emitted_when_result_has_no_bounds(self):
        watch = _make_watch()
        result = _make_result(with_bounds=False)
        formatted = format_prediction(watch, result, emit_confidence_bounds=True)
        assert len(formatted) == 1

    def test_unknown_version_when_model_version_is_none(self):
        watch = _make_watch()
        result = _make_result()
        result.model_version = None
        formatted = format_prediction(watch, result)
        assert formatted[0].labels["sentinel_version"] == "unknown"


class TestBuildMetricKey:

    def test_no_labels(self):
        assert build_metric_key("my_metric", {}) == "my_metric"

    def test_with_labels(self):
        key = build_metric_key("my_metric", {"job": "api", "env": "prod"})
        assert key == 'my_metric{env="prod",job="api"}'

    def test_labels_sorted(self):
        key1 = build_metric_key("m", {"b": "2", "a": "1"})
        key2 = build_metric_key("m", {"a": "1", "b": "2"})
        assert key1 == key2