# sentinel/pipeline/trainer.py

import threading
from datetime import datetime, timezone
from typing import Optional
import numpy as np
from sentinel.pipeline.models.base import BaseModel, TrainingResult
from sentinel.pipeline.versioning import ModelVersion
from sentinel.pipeline.registry import ModelRegistry
from sentinel.pipeline.drift import DriftSeverity
from sentinel.ingestor.buffer import MetricBuffer
from sentinel.ingestor.features import build_feature_matrix
from sentinel.config import WatchConfig
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

_HOLDOUT_FRACTION = 0.1


class Trainer:

    def __init__(
        self,
        watch_config: WatchConfig,
        buffer: MetricBuffer,
        registry: ModelRegistry,
    ):
        self.watch_config = watch_config
        self.buffer = buffer
        self.registry = registry
        self._lock = threading.Lock()

    def run(
        self,
        drift_severity: str = DriftSeverity.NONE,
        drift_score: float = 0.0,
    ) -> Optional[TrainingResult]:
        with self._lock:
            return self._run_internal(drift_severity, drift_score)

    def _run_internal(
        self,
        drift_severity: str,
        drift_score: float,
    ) -> Optional[TrainingResult]:

        metric_key = self._metric_key()
        logger.info(
            f"[{metric_key}] training run started "
            f"severity={drift_severity} drift_score={drift_score:.4f}"
        )

        values = self.buffer.get_values()
        timestamps = self.buffer.get_timestamps()

        if len(values) < 2:
            logger.warning(f"[{metric_key}] not enough data to train, skipping")
            return None

        try:
            X, y = build_feature_matrix(
                values=values,
                lookback=self.watch_config.lookback,
                granularity=self.watch_config.granularity,
                timestamps=timestamps,
            )
        except ValueError as e:
            logger.error(f"[{metric_key}] feature build failed: {e}")
            return None

        if len(X) < 4:
            logger.warning(f"[{metric_key}] too few samples after feature build ({len(X)}), skipping")
            return None

        split = max(1, int(len(X) * (1 - _HOLDOUT_FRACTION)))
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

        current_model = self.registry.get_model()
        policy = self._determine_policy(drift_severity, current_model)
        model = self._get_or_create_model()

        try:
            if policy == "full_retrain" or not model.is_fitted:
                result = model.fit(X_train, y_train)
                result.training_policy = "full_retrain"
            else:
                result = model.partial_fit(X_train, y_train)
                result.training_policy = "finetune"
        except Exception as e:
            logger.error(f"[{metric_key}] model training failed: {e}")
            return None

        if len(X_val) > 0:
            val_result = self._validate(model, X_val, y_val)
            if val_result is None:
                logger.warning(f"[{metric_key}] validation failed, not promoting model")
                return None
            result.mae = val_result["mae"]
            result.mape = val_result["mape"]

        version_id = self.registry.version_store.next_version_id()
        version = ModelVersion(
            version_id=version_id,
            metric_key=metric_key,
            model_class=type(model).__name__,
            trained_at=datetime.now(timezone.utc).isoformat(),
            training_policy=result.training_policy,
            drift_score_at_trigger=drift_score,
            mae=result.mae,
            mape=result.mape,
            n_samples=result.n_samples,
            artifact_path="",
            extra={
                "granularity": self.watch_config.granularity,
                "horizon": self.watch_config.horizon,
                "lookback": self.watch_config.lookback,
            }
        )

        self.registry.promote(model, version)

        logger.info(
            f"[{metric_key}] training complete — version={version_id} "
            f"policy={result.training_policy} mae={result.mae:.4f} mape={result.mape:.2f}%"
        )

        return result

    def _validate(self, model: BaseModel, X_val: np.ndarray, y_val: np.ndarray) -> Optional[dict]:
        try:
            y_pred = []
            for i in range(len(X_val)):
                pred_result = model.predict(X_val[i:i + 1])
                # .flat[0] safely handles any array shape — 0-d, 1-d, multi-step
                val = float(np.asarray(pred_result.values).flat[0])
                y_pred.append(val)
            y_pred = np.array(y_pred)
            mae = float(np.mean(np.abs(y_val - y_pred)))
            mape = float(np.mean(np.abs((y_val - y_pred) / (y_val + 1e-8)))) * 100
            return {"mae": mae, "mape": mape}
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return None

    def _get_or_create_model(self) -> BaseModel:
        existing = self.registry.get_model()
        if existing is not None:
            return existing
        cls = self.watch_config.model_class
        return cls(
            granularity=self.watch_config.granularity,
            horizon=self.watch_config.horizon,
            lookback=self.watch_config.lookback,
        )

    def _determine_policy(self, drift_severity: str, current_model: Optional[BaseModel]) -> str:
        if current_model is None or not current_model.is_fitted:
            return "full_retrain"
        if drift_severity == DriftSeverity.HIGH:
            return "full_retrain"
        if drift_severity == DriftSeverity.LOW:
            return "finetune"
        return "full_retrain"

    def _metric_key(self) -> str:
        cfg = self.watch_config
        if not cfg.labels:
            return cfg.metric
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(cfg.labels.items()))
        return f"{cfg.metric}{{{label_str}}}"