# sentinel/pipeline/models/smoothing.py

import numpy as np
import joblib
from sentinel.pipeline.models.base import BaseModel, PredictionResult, TrainingResult
from sentinel.utils.time import parse_duration_to_steps


class ExponentialSmoothingModel(BaseModel):
    """
    Double exponential smoothing (Holt's method).
    Handles level and trend. Good for metrics with a trend
    but no strong seasonality. Lightweight and interpretable.
    """

    def __init__(
        self,
        granularity: str,
        horizon: str,
        lookback: str,
        alpha: float = 0.3,
        beta: float = 0.1,
    ):
        super().__init__(granularity, horizon, lookback)
        self.alpha = alpha  # level smoothing factor
        self.beta = beta    # trend smoothing factor
        self._level: float = 0.0
        self._trend: float = 0.0
        self._horizon_steps = parse_duration_to_steps(horizon, granularity)

    def _holt_fit(self, y: np.ndarray):
        level = y[0]
        trend = y[1] - y[0]
        for val in y[1:]:
            prev_level = level
            level = self.alpha * val + (1 - self.alpha) * (level + trend)
            trend = self.beta * (level - prev_level) + (1 - self.beta) * trend
        self._level = level
        self._trend = trend

    def _compute_fitted_values(self, y: np.ndarray) -> np.ndarray:
        fitted = []
        level = y[0]
        trend = y[1] - y[0]
        for val in y[1:]:
            fitted.append(level + trend)
            prev_level = level
            level = self.alpha * val + (1 - self.alpha) * (level + trend)
            trend = self.beta * (level - prev_level) + (1 - self.beta) * trend
        return np.array(fitted)

    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        self._holt_fit(y)
        self._is_fitted = True

        fitted = self._compute_fitted_values(y)
        residuals = y[1:] - fitted
        mae = float(np.mean(np.abs(residuals)))
        mape = float(np.mean(np.abs(residuals / (y[1:] + 1e-8)))) * 100

        return TrainingResult(
            mae=mae,
            mape=mape,
            n_samples=len(y),
            training_policy="full_retrain",
            extra={"alpha": self.alpha, "beta": self.beta}
        )

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        if not self._is_fitted:
            return self.fit(X, y)

        # update level and trend on recent data only
        for val in y:
            prev_level = self._level
            self._level = self.alpha * val + (1 - self.alpha) * (self._level + self._trend)
            self._trend = self.beta * (self._level - prev_level) + (1 - self.beta) * self._trend

        fitted = self._compute_fitted_values(y)
        residuals = y[1:] - fitted if len(y) > 1 else np.array([0.0])
        mae = float(np.mean(np.abs(residuals)))
        mape = float(np.mean(np.abs(residuals / (y[1:] + 1e-8)))) * 100 if len(y) > 1 else 0.0

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
        for step in range(1, self._horizon_steps + 1):
            values.append(self._level + step * self._trend)

        values = np.array(values)
        timestamps = np.arange(self._horizon_steps, dtype=float)

        return PredictionResult(values=values, timestamps=timestamps)

    def save(self, path: str) -> None:
        joblib.dump({
            "level": self._level,
            "trend": self._trend,
            "alpha": self.alpha,
            "beta": self.beta,
            "horizon_steps": self._horizon_steps,
        }, path)

    def load(self, path: str) -> None:
        state = joblib.load(path)
        self._level = state["level"]
        self._trend = state["trend"]
        self.alpha = state["alpha"]
        self.beta = state["beta"]
        self._horizon_steps = state["horizon_steps"]
        self._is_fitted = True