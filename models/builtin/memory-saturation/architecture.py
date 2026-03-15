import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class MemorySaturationPredictor:
    """
    Predicts whether memory usage will reach saturation in the next 15 minutes.

    Algorithm: Linear regression on recent memory usage samples to extrapolate
    the trend forward. If the projected value exceeds the saturation threshold
    (configurable as a fraction of observed max usage), it is flagged as high risk.

    This is intentionally simple and interpretable — memory leaks and steady
    growth are well captured by linear trends. Sudden spikes are caught by
    the anomaly score component.
    """

    PREDICTION_HORIZON = 900   # 15 minutes in seconds
    SATURATION_FRACTION = 0.90  # flag when projected usage exceeds 90% of observed max

    def __init__(self):
        self.model = LinearRegression()
        self.observed_max_bytes = None
        self.is_trained = False

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample memory usage to 1-minute buckets and compute trend features.
        df must have columns: timestamp, value.
        Returns a DataFrame with one row per minute bucket.
        """
        if df.empty:
            return pd.DataFrame()

        df = df.sort_values("timestamp").copy()
        df["minute"] = (df["timestamp"] // 60).astype(int)

        bucketed = df.groupby("minute")["value"].agg(
            mean="mean",
            max="max",
            min="min",
        ).reset_index()

        # Normalise time so regression coefficients are not astronomically large
        bucketed["time_offset"] = bucketed["minute"] - bucketed["minute"].min()

        return bucketed

    def fit(self, features: pd.DataFrame) -> None:
        """Fit the linear trend model on historical memory usage buckets."""
        if features.empty or len(features) < 5:
            return

        X = features[["time_offset"]].values
        y = features["mean"].values

        self.model.fit(X, y)
        self.observed_max_bytes = float(features["max"].max())
        self.is_trained = True

    def predict_proba(self, features: pd.DataFrame) -> float:
        """
        Project memory usage PREDICTION_HORIZON seconds ahead and return
        the probability of saturation as a value between 0.0 and 1.0.

        Returns 0.0 if the model is not yet trained or has no data.
        """
        if not self.is_trained or features.empty or self.observed_max_bytes is None:
            return 0.0

        latest_offset = float(features["time_offset"].max())
        future_offset = latest_offset + (self.PREDICTION_HORIZON / 60)

        projected_bytes = self.model.predict([[future_offset]])[0]
        saturation_threshold = self.observed_max_bytes * self.SATURATION_FRACTION

        if projected_bytes <= 0:
            return 0.0

        # Score is how close the projection is to the saturation threshold
        # Clipped to [0, 1]
        score = min(projected_bytes / saturation_threshold, 1.0)
        return float(score)