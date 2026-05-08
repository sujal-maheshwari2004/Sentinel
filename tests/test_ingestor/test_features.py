# tests/test_ingestor/test_features.py

import pytest
import numpy as np
from sentinel.ingestor.features import (
    build_lag_features,
    build_feature_matrix,
    build_prediction_input,
)


class TestBuildLagFeatures:

    def test_basic_shape(self):
        values = np.arange(10, dtype=float)
        X, y = build_lag_features(values, n_lags=3)
        assert X.shape == (7, 3)
        assert y.shape == (7,)

    def test_first_row(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        X, y = build_lag_features(values, n_lags=3)
        assert list(X[0]) == [1.0, 2.0, 3.0]
        assert y[0] == 4.0

    def test_last_row(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        X, y = build_lag_features(values, n_lags=3)
        assert list(X[-1]) == [2.0, 3.0, 4.0]
        assert y[-1] == 5.0

    def test_not_enough_values_raises(self):
        values = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            build_lag_features(values, n_lags=3)

    def test_single_lag(self):
        values = np.array([1.0, 2.0, 3.0])
        X, y = build_lag_features(values, n_lags=1)
        assert X.shape == (2, 1)
        assert y.shape == (2,)


class TestBuildFeatureMatrix:

    def _make_values(self, n=50):
        return np.linspace(0, 10, n)

    def _make_timestamps(self, n=50):
        base = 1700000000.0
        return np.array([base + i * 60 for i in range(n)])

    def test_output_shapes_consistent(self):
        values = self._make_values()
        X, y = build_feature_matrix(
            values=values,
            lookback="5m",
            granularity="1m",
            include_rolling_mean=True,
            include_rolling_std=True,
            include_time_features=False,
        )
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] > 5  # lags + rolling mean + rolling std

    def test_with_time_features(self):
        values = self._make_values()
        timestamps = self._make_timestamps()
        X, y = build_feature_matrix(
            values=values,
            lookback="5m",
            granularity="1m",
            include_time_features=True,
            timestamps=timestamps,
        )
        assert X.shape[1] == 5 + 1 + 1 + 4  # lags + mean + std + 4 time features

    def test_without_optional_features(self):
        values = self._make_values()
        X, y = build_feature_matrix(
            values=values,
            lookback="5m",
            granularity="1m",
            include_rolling_mean=False,
            include_rolling_std=False,
            include_time_features=False,
        )
        assert X.shape[1] == 5  # only lag features

    def test_time_features_without_timestamps_skipped(self):
        values = self._make_values()
        X, y = build_feature_matrix(
            values=values,
            lookback="5m",
            granularity="1m",
            include_rolling_mean=False,
            include_rolling_std=False,
            include_time_features=True,
            timestamps=None,
        )
        assert X.shape[1] == 5  # time features skipped gracefully


class TestBuildPredictionInput:

    def _make_values(self, n=50):
        return np.linspace(0, 10, n)

    def _make_timestamps(self, n=50):
        base = 1700000000.0
        return np.array([base + i * 60 for i in range(n)])

    def test_output_shape(self):
        values = self._make_values()
        X = build_prediction_input(
            values=values,
            lookback="5m",
            granularity="1m",
            include_rolling_mean=False,
            include_rolling_std=False,
            include_time_features=False,
        )
        assert X.shape == (1, 5)

    def test_uses_last_n_values(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        X = build_prediction_input(
            values=values,
            lookback="5m",
            granularity="1m",
            include_rolling_mean=False,
            include_rolling_std=False,
            include_time_features=False,
        )
        assert list(X[0]) == [3.0, 4.0, 5.0, 6.0, 7.0]

    def test_not_enough_values_raises(self):
        values = np.array([1.0, 2.0])
        with pytest.raises(ValueError):
            build_prediction_input(
                values=values,
                lookback="5m",
                granularity="1m",
                include_rolling_mean=False,
                include_rolling_std=False,
                include_time_features=False,
            )

    def test_with_all_features(self):
        values = self._make_values()
        timestamps = self._make_timestamps()
        X = build_prediction_input(
            values=values,
            lookback="5m",
            granularity="1m",
            include_rolling_mean=True,
            include_rolling_std=True,
            include_time_features=True,
            timestamps=timestamps,
        )
        assert X.shape == (1, 5 + 1 + 1 + 4)