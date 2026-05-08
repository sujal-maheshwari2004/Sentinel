# sentinel/pipeline/models/arima.py

import numpy as np
import joblib
from sentinel.pipeline.models.base import BaseModel, PredictionResult, TrainingResult
from sentinel.utils.time import parse_duration_to_steps


class ARIMAModel(BaseModel):
    """
    ARIMA(p, d, q) implemented without external dependencies.
    Uses simple AR(p) after differencing d times.
    MA(q) terms approximated via residual correction.
    Suitable for stationary or trend-stationary metrics.
    """

    def __init__(
        self,
        granularity: str,
        horizon: str,
        lookback: str,
        p: int = 3,
        d: int = 1,
        q: int = 1,
    ):
        super().__init__(granularity, horizon, lookback)
        self.p = p
        self.d = d
        self.q = q
        self._ar_coeffs: np.ndarray = None
        self._ma_coeffs: np.ndarray = None
        self._diff_init: list = []   # stores values needed to invert differencing
        self._residuals: np.ndarray = None
        self._horizon_steps = parse_duration_to_steps(horizon, granularity)

    def _difference(self, y: np.ndarray, d: int):
        diffs = [y.copy()]
        for _ in range(d):
            diffs.append(np.diff(diffs[-1]))
        return diffs

    def _invert_difference(self, forecast: np.ndarray, diffs: list) -> np.ndarray:
        result = forecast.copy()
        for orig in reversed(diffs[:-1]):
            result = np.cumsum(np.hstack([orig[-1], result]))
        return result

    def _fit_ar(self, y: np.ndarray) -> np.ndarray:
        n = len(y)
        if n <= self.p:
            return np.zeros(self.p)
        X = np.array([y[i:n - self.p + i] for i in range(self.p)]).T
        target = y[self.p:]
        coeffs, _, _, _ = np.linalg.lstsq(X, target, rcond=None)
        return coeffs

    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        diffs = self._difference(y, self.d)
        self._diff_init = diffs
        stationary = diffs[-1]

        self._ar_coeffs = self._fit_ar(stationary)

        # compute residuals for MA correction
        n = len(stationary)
        fitted = np.array([
            np.dot(self._ar_coeffs, stationary[i:i + self.p])
            for i in range(n - self.p)
        ])
        self._residuals = stationary[self.p:] - fitted

        # fit MA coefficients on residuals
        if self.q > 0 and len(self._residuals) > self.q:
            self._ma_coeffs = self._fit_ar(self._residuals)[:self.q]
        else:
            self._ma_coeffs = np.zeros(self.q)

        self._is_fitted = True

        mae = float(np.mean(np.abs(self._residuals)))
        mape = float(np.mean(np.abs(self._residuals / (stationary[self.p:] + 1e-8)))) * 100

        return TrainingResult(
            mae=mae,
            mape=mape,
            n_samples=len(y),
            training_policy="full_retrain",
            extra={"p": self.p, "d": self.d, "q": self.q}
        )

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        if not self._is_fitted:
            return self.fit(X, y)
        # for ARIMA finetune we do a full refit on the new window
        # keeping p, d, q fixed
        return self.fit(X, y)

    def predict(self, X: np.ndarray) -> PredictionResult:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        diffs = self._diff_init
        stationary = diffs[-1].tolist()
        residuals = self._residuals.tolist() if self._residuals is not None else []

        forecasts = []
        for step in range(self._horizon_steps):
            ar_input = stationary[-self.p:]
            ar_val = np.dot(self._ar_coeffs, ar_input)

            ma_val = 0.0
            if self.q > 0 and len(residuals) >= self.q:
                ma_val = np.dot(self._ma_coeffs, residuals[-self.q:])

            val = ar_val + ma_val
            forecasts.append(val)
            stationary.append(val)
            residuals.append(0.0)  # future residuals unknown, assume zero

        forecast_arr = np.array(forecasts)
        forecast_arr = self._invert_difference(forecast_arr, diffs)

        timestamps = np.arange(self._horizon_steps, dtype=float)

        return PredictionResult(values=forecast_arr, timestamps=timestamps)

    def save(self, path: str) -> None:
        joblib.dump({
            "ar_coeffs": self._ar_coeffs,
            "ma_coeffs": self._ma_coeffs,
            "diff_init": self._diff_init,
            "residuals": self._residuals,
            "p": self.p,
            "d": self.d,
            "q": self.q,
            "horizon_steps": self._horizon_steps,
        }, path)

    def load(self, path: str) -> None:
        state = joblib.load(path)
        self._ar_coeffs = state["ar_coeffs"]
        self._ma_coeffs = state["ma_coeffs"]
        self._diff_init = state["diff_init"]
        self._residuals = state["residuals"]
        self.p = state["p"]
        self.d = state["d"]
        self.q = state["q"]
        self._horizon_steps = state["horizon_steps"]
        self._is_fitted = True