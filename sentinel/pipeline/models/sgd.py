# sentinel/pipeline/models/sgd.py

import numpy as np
import joblib
from sentinel.pipeline.models.base import BaseModel, PredictionResult, TrainingResult
from sentinel.utils.time import parse_duration_to_steps


class SGDRegressorModel(BaseModel):
    """
    Online learning model using stochastic gradient descent.
    Best for metrics that evolve continuously and benefit from
    incremental updates. Most finetune-friendly model in the pack.
    Implements linear regression with SGD updates manually to
    avoid sklearn dependency at the core.
    """

    def __init__(
        self,
        granularity: str,
        horizon: str,
        lookback: str,
        learning_rate: float = 0.01,
        epochs: int = 10,
        l2: float = 1e-4,
    ):
        super().__init__(granularity, horizon, lookback)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self._weights: np.ndarray = None
        self._bias: float = 0.0
        self._horizon_steps = parse_duration_to_steps(horizon, granularity)

    def _init_weights(self, n_features: int):
        self._weights = np.zeros(n_features)
        self._bias = 0.0

    def _sgd_update(self, X: np.ndarray, y: np.ndarray):
        n = len(y)
        for _ in range(self.epochs):
            indices = np.random.permutation(n)
            for i in indices:
                xi = X[i]
                yi = y[i]
                pred = np.dot(self._weights, xi) + self._bias
                error = pred - yi
                self._weights -= self.learning_rate * (error * xi + self.l2 * self._weights)
                self._bias -= self.learning_rate * error

    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        self._init_weights(X.shape[1])
        self._sgd_update(X, y)
        self._is_fitted = True

        y_pred = X @ self._weights + self._bias
        mae = float(np.mean(np.abs(y - y_pred)))
        mape = float(np.mean(np.abs((y - y_pred) / (y + 1e-8)))) * 100

        return TrainingResult(
            mae=mae,
            mape=mape,
            n_samples=len(y),
            training_policy="full_retrain",
            extra={"learning_rate": self.learning_rate, "epochs": self.epochs}
        )

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        if not self._is_fitted:
            return self.fit(X, y)

        # continue SGD from current weights without reinitializing
        self._sgd_update(X, y)

        y_pred = X @ self._weights + self._bias
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
            val = float(np.dot(self._weights, current[0]) + self._bias)
            values.append(val)
            current = np.roll(current, -1)
            current[0, -1] = val

        values = np.array(values)
        timestamps = np.arange(self._horizon_steps, dtype=float)

        return PredictionResult(values=values, timestamps=timestamps)

    def save(self, path: str) -> None:
        joblib.dump({
            "weights": self._weights,
            "bias": self._bias,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "l2": self.l2,
            "horizon_steps": self._horizon_steps,
        }, path)

    def load(self, path: str) -> None:
        state = joblib.load(path)
        self._weights = state["weights"]
        self._bias = state["bias"]
        self.learning_rate = state["learning_rate"]
        self.epochs = state["epochs"]
        self.l2 = state["l2"]
        self._horizon_steps = state["horizon_steps"]
        self._is_fitted = True