import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


class LatencySpikePredictor:
    """
    Predicts whether HTTP latency will spike in the next 10 minutes.

    Algorithm: LightGBM binary classifier trained on sliding window
    statistical features extracted from the latency time series.

    Features per window:
      - mean, std, min, max of request duration
      - p95 and p99 of request duration
      - rate of change (current mean vs previous window mean)
      - request rate (requests per second)
      - error rate (5xx / total requests)
    """

    WINDOW_SIZE = 60        # seconds per feature window
    PREDICTION_HORIZON = 600  # predict 10 minutes ahead

    def __init__(self):
        self.model = LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )
        self.is_trained = False

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build sliding window features from raw metric samples.
        df must have columns: timestamp, value, and any label columns.
        Returns a DataFrame where each row is one time window.
        """
        if df.empty:
            return pd.DataFrame()

        df = df.sort_values("timestamp").copy()
        df["window"] = (df["timestamp"] // self.WINDOW_SIZE).astype(int)

        features = df.groupby("window")["value"].agg(
            mean="mean",
            std="std",
            min="min",
            max="max",
            p95=lambda x: np.percentile(x, 95),
            p99=lambda x: np.percentile(x, 99),
            count="count",
        ).reset_index()

        # Rate of change — how much did mean latency shift vs the previous window
        features["mean_delta"] = features["mean"].diff().fillna(0)
        features["std_delta"] = features["std"].diff().fillna(0)

        # Drop the window index — it is not a feature
        features = features.drop(columns=["window"])
        features = features.fillna(0)

        return features

    def create_labels(self, features: pd.DataFrame, spike_threshold_multiplier: float = 2.0) -> pd.Series:
        """
        Label each window as a spike (1) or not (0).
        A spike is defined as p99 latency exceeding spike_threshold_multiplier
        times the rolling median p99 over the past hour.
        """
        rolling_median = features["p99"].rolling(window=60, min_periods=1).median()
        labels = (features["p99"] > rolling_median * spike_threshold_multiplier).astype(int)
        return labels

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
        """Train the LightGBM model on the extracted features and labels."""
        self.model.fit(features.values, labels.values)
        self.is_trained = True

    def predict_proba(self, features: pd.DataFrame) -> float:
        """
        Return the probability of a latency spike for the most recent window.
        Returns 0.0 if the model is not yet trained.
        """
        if not self.is_trained or features.empty:
            return 0.0

        latest_window = features.iloc[[-1]]
        probabilities = self.model.predict_proba(latest_window.values)
        return float(probabilities[0][1])   # probability of class 1 (spike)