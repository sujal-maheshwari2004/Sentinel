# sentinel/ingestor/features.py

import numpy as np
from sentinel.utils.time import parse_duration_to_steps
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)


def build_lag_features(
    values: np.ndarray,
    n_lags: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a supervised learning dataset from a univariate time series
    using lag features.

    Given a series [v0, v1, v2, ..., vN], for n_lags=3 this produces:

        X (features):           y (targets):
        [v0, v1, v2]         ->  v3
        [v1, v2, v3]         ->  v4
        ...
        [vN-3, vN-2, vN-1]  ->  vN

    X shape: (n_samples, n_lags)
    y shape: (n_samples,)

    n_samples = len(values) - n_lags
    """
    if len(values) <= n_lags:
        raise ValueError(
            f"Not enough values to build lag features. "
            f"Need more than {n_lags} values, got {len(values)}."
        )

    X = np.array([values[i:i + n_lags] for i in range(len(values) - n_lags)])
    y = values[n_lags:]

    return X, y


def build_feature_matrix(
    values: np.ndarray,
    lookback: str,
    granularity: str,
    include_rolling_mean: bool = True,
    include_rolling_std: bool = True,
    include_time_features: bool = True,
    timestamps: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full feature engineering pipeline. Builds lag features and optionally
    appends rolling statistics and time-based features.

    values       : raw metric values in chronological order
    lookback     : lookback window e.g. "30m"
    granularity  : resolution e.g. "1m"
    timestamps   : unix timestamps aligned with values, required if
                   include_time_features=True

    Returns (X, y) where X has shape (n_samples, n_features).
    """
    n_lags = parse_duration_to_steps(lookback, granularity)
    X_lag, y = build_lag_features(values, n_lags)

    feature_blocks = [X_lag]

    if include_rolling_mean:
        rolling_mean = _rolling_mean(values, window=n_lags)
        # align to X rows — rolling mean is computed up to each sample's last lag
        rolling_mean_aligned = rolling_mean[n_lags:]
        feature_blocks.append(rolling_mean_aligned.reshape(-1, 1))

    if include_rolling_std:
        rolling_std = _rolling_std(values, window=n_lags)
        rolling_std_aligned = rolling_std[n_lags:]
        feature_blocks.append(rolling_std_aligned.reshape(-1, 1))

    if include_time_features and timestamps is not None:
        time_feats = _time_features(timestamps[n_lags:])
        feature_blocks.append(time_feats)
    elif include_time_features and timestamps is None:
        logger.warning(
            "include_time_features=True but no timestamps provided. "
            "Skipping time features."
        )

    X = np.hstack(feature_blocks)

    return X, y


def build_prediction_input(
    values: np.ndarray,
    lookback: str,
    granularity: str,
    include_rolling_mean: bool = True,
    include_rolling_std: bool = True,
    include_time_features: bool = True,
    timestamps: np.ndarray = None,
) -> np.ndarray:
    """
    Build the feature vector for the most recent observation.
    This is the input passed to model.predict() at inference time.

    Returns X with shape (1, n_features).
    """
    n_lags = parse_duration_to_steps(lookback, granularity)

    if len(values) < n_lags:
        raise ValueError(
            f"Need at least {n_lags} values to build prediction input, "
            f"got {len(values)}."
        )

    lag_window = values[-n_lags:].reshape(1, -1)
    feature_blocks = [lag_window]

    if include_rolling_mean:
        mean_val = np.mean(values[-n_lags:]).reshape(1, 1)
        feature_blocks.append(mean_val)

    if include_rolling_std:
        std_val = np.std(values[-n_lags:]).reshape(1, 1)
        feature_blocks.append(std_val)

    if include_time_features and timestamps is not None:
        latest_ts = timestamps[-1:]
        time_feats = _time_features(latest_ts)
        feature_blocks.append(time_feats)

    return np.hstack(feature_blocks)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling mean with the given window size.
    First (window-1) values are filled with the expanding mean.
    """
    result = np.empty(len(values))
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result[i] = np.mean(values[start:i + 1])
    return result


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling standard deviation with the given window size.
    First (window-1) values are filled with the expanding std.
    """
    result = np.empty(len(values))
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start:i + 1]
        result[i] = np.std(chunk) if len(chunk) > 1 else 0.0
    return result


def _time_features(timestamps: np.ndarray) -> np.ndarray:
    """
    Extract cyclical time features from unix timestamps.
    Encodes hour-of-day and day-of-week as sine/cosine pairs
    so the model sees time as continuous rather than categorical.

    Returns array of shape (n, 4):
        col 0: sin(hour_of_day)
        col 1: cos(hour_of_day)
        col 2: sin(day_of_week)
        col 3: cos(day_of_week)
    """
    hours = (timestamps % 86400) / 3600          # 0..23
    days = (timestamps // 86400) % 7             # 0..6

    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)
    day_sin = np.sin(2 * np.pi * days / 7)
    day_cos = np.cos(2 * np.pi * days / 7)

    return np.stack([hour_sin, hour_cos, day_sin, day_cos], axis=1)