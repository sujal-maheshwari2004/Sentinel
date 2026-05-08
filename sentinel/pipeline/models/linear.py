# sentinel/pipeline/models/linear.py

import numpy as np
import joblib
from dataclasses import dataclass
from sentinel.pipeline.models.base import BaseModel, PredictionResult, TrainingResult
from sentinel.utils.time import parse_duration_to_steps


class LinearTrendModel(BaseModel):
    """
    Simple linear regression model using least squares.
    Best for metrics with a clear linear trend and no seasonality.
    Fastest model in the pack, good baseline.
    """

    def __init__(self, granularity: str, horizon: str, lookback: str):
        super().__init__(granularity, horizon, lookback)
        self._coefficients: np.ndarray = None
        self._intercept: float = 0.0
        self._horizon_steps = parse_duration_to_steps(horizon, granularity)

    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        coeffs, _, _, _ = np.linalg.lstsq(X_bias, y, rcond=None)
        self._intercept = coeffs[0]
        self._coefficients = coeffs[1:]
        self._is_fitted = True

        y_pred = X_bias @ coeffs
        mae = float(np.mean(np.abs(y - y_pred)))
        mape = float(np.mean(np.abs((y - y_pred) / (y + 1e-8)))) * 100

        return TrainingResult(
            mae=mae,
            mape=mape,
            n_samples=len(y),
            training_policy="full_retrain"
        )

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        if not self._is_fitted:
            return self.fit(X, y)

        # incremental least squares update using new data only
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        coeffs, _, _, _ = np.linalg.lstsq(X_bias, y, rcond=None)
        # blend old and new coefficients 70/30
        new_coeffs = coeffs[1:]
        self._coefficients = 0.7 * self._coefficients + 0.3 * new_coeffs
        self._intercept = 0.7 * self._intercept + 0.3 * coeffs[0]

        y_pred = X_bias @ np.hstack([self._intercept, self._coefficients])
        mae = float(np.mean(np.abs(y - y_pred)))
        mape = float(np.mean(np.abs((y - y_pred) / (y + 1e-8)))) * 100

        return TrainingResult(
            mae=mae,
            mape=mape,
            n_samples=len(y),
            training_policy="finetune"
        )

    def predict(self, X: np.ndarray) -> PredictionResult:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        values = []
        current = X.copy()

        for _ in range(self._horizon_steps):
            val = self._intercept + current @ self._coefficients
            values.append(float(val))
            # roll window forward
            current = np.roll(current, -1)
            current[0, -1] = val

        values = np.array(values)
        timestamps = np.arange(self._horizon_steps, dtype=float)

        return PredictionResult(values=values, timestamps=timestamps)

    def save(self, path: str) -> None:
        joblib.dump({
            "coefficients": self._coefficients,
            "intercept": self._intercept,
            "horizon_steps": self._horizon_steps,
        }, path)

    def load(self, path: str) -> None:
        state = joblib.load(path)
        self._coefficients = state["coefficients"]
        self._intercept = state["intercept"]
        self._horizon_steps = state["horizon_steps"]
        self._is_fitted = True