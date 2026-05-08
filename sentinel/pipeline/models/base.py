# sentinel/pipeline/models/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class PredictionResult:
    values: np.ndarray
    timestamps: np.ndarray
    lower_bound: Optional[np.ndarray] = None
    upper_bound: Optional[np.ndarray] = None
    model_version: Optional[str] = None


@dataclass
class TrainingResult:
    mae: float
    mape: float
    n_samples: int
    training_policy: str  # "full_retrain" or "finetune"
    extra: dict = field(default_factory=dict)


class BaseModel(ABC):

    def __init__(self, granularity: str, horizon: str, lookback: str):
        self.granularity = granularity
        self.horizon = horizon
        self.lookback = lookback
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainingResult: ...

    @abstractmethod
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> TrainingResult: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> PredictionResult: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"granularity={self.granularity}, "
            f"horizon={self.horizon}, "
            f"lookback={self.lookback}, "
            f"fitted={self._is_fitted})"
        )