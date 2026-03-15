import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


class CpuExhaustionPredictor:
    """
    Predicts whether CPU usage will hit throttling levels in the next 10 minutes.

    Algorithm: LightGBM binary classifier trained on sliding window features
    computed from the CPU seconds counter. The counter is differentiated to
    get CPU rate (cores in use), then statistical features are extracted per window.

    Features per window:
      - mean, std, max CPU rate
      - rate of change between windows
      - sustained high usage flag (fraction of samples above 80% of observed max)
    """

    WINDOW_SIZE = 60            # seconds per feature window
    PREDICTION_HORIZON = 600    # predict 10 minutes ahead
    HIGH_CPU_FRACTION = 0.80    # threshold for "high CPU" flag

    def __init__(self):
        self.model = LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )
        self.observed_max_rate = None
        self.is_trained = False

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Differentiate the CPU counter to get rate, then compute
        sliding window features. df must have columns: timestamp, value.
        """
        if df.empty or len(df) < 2:
            return pd.DataFrame()

        df = df.sort_values("timestamp").copy()

        # CPU seconds is a counter — differentiate to get rate (cores per second)
        df["cpu_rate"] = df["value"].diff() / df["timestamp"].diff()
        df = df.dropna(subset=["cpu_rate"])
        df = df[df["cpu_rate"] >= 0]   # drop negative diffs from counter resets

        if df.empty:
            return pd.DataFrame()

        self.observed_max_rate = float(df["cpu_rate"].max()) if self.observed_max_rate is None else self.observed_max_rate
        high_threshold = self.observed_max_rate * self.HIGH_CPU_FRACTION

        df["window"] = (df["timestamp"] // self.WINDOW_SIZE).astype(int)

        features = df.groupby("window")["cpu_rate"].agg(
            mean="mean",
            std="std",
            max="max",
            high_cpu_fraction=lambda x: (x > high_threshold).mean(),
        ).reset_index()

        features["mean_delta"] = features["mean"].diff().fillna(0)
        features = features.drop(columns=["window"]).fillna(0)

        return features

    def create_labels(self, features: pd.DataFrame, exhaustion_threshold: float = 0.90) -> pd.Series:
        """
        Label each window as exhaustion risk (1) or not (0).
        A window is labelled 1 if max CPU rate exceeds exhaustion_threshold
        times the observed max in the training data.
        """
        threshold = features["max"].max() * exhaustion_threshold
        labels = (features["max"] > threshold).astype(int)
        return labels

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
        self.model.fit(features.values, labels.values)
        self.is_trained = True

    def predict_proba(self, features: pd.DataFrame) -> float:
        if not self.is_trained or features.empty:
            return 0.0
        latest_window = features.iloc[[-1]]
        probabilities = self.model.predict_proba(latest_window.values)
        return float(probabilities[0][1])