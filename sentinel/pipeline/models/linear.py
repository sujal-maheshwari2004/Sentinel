# sentinel/pipeline/models/linear.py

import numpy as np
import joblib
from sentinel.pipeline.models.base import BaseModel, PredictionResult, TrainingResult
from sentinel.utils.time import parse_duration_to_steps


class LinearTrendModel(BaseModel):

    def __init__(self, granularity: str, horizon: str, lookback: str):
        super().__init__(granularity, horizon, lookback)
        self._coefficients: np.ndarray = None
        self._intercept: float = 0.0
        self._horizon_steps = parse_duration_to_steps(horizon, granularity)

    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        coeffs, _, _, _ = np.linalg.lstsq(X_bias, y, rcond=None)
        self._intercept = float(coeffs[0])
        self._coefficients = coeffs[1:].ravel()  # ensure always 1-d
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

        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        coeffs, _, _, _ = np.linalg.lstsq(X_bias, y, rcond=None)
        new_coeffs = coeffs[1:].ravel()
        self._coefficients = 0.7 * self._coefficients + 0.3 * new_coeffs
        self._intercept = 0.7 * self._intercept + 0.3 * float(coeffs[0])

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
        # ensure current is always 2-d (1, n_features)
        current = np.atleast_2d(X[0])

        for _ in range(self._horizon_steps):
            # dot product of 1-d coefficient vector with 1-d feature vector
            val = float(self._intercept + np.dot(current.ravel(), self._coefficients))
            values.append(val)
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
        self._coefficients = np.asarray(state["coefficients"]).ravel()
        self._intercept = float(state["intercept"])
        self._horizon_steps = state["horizon_steps"]
        self._is_fitted = True