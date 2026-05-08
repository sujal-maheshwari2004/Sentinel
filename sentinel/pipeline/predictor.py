# sentinel/pipeline/predictor.py

import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional
from sentinel.pipeline.registry import ModelRegistry
from sentinel.pipeline.models.base import PredictionResult
from sentinel.ingestor.buffer import MetricBuffer
from sentinel.ingestor.features import build_prediction_input
from sentinel.config import WatchConfig
from sentinel.utils.time import parse_duration_to_seconds
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)


class Predictor:
    """
    Runs inference for a single metric using the currently active model.

    Called on every emitter tick. Reads the latest window from the buffer,
    builds the prediction input, runs model.predict(), and returns a
    PredictionResult with absolute timestamps attached.

    Returns None if the model is not ready yet (cold start).
    """

    def __init__(
        self,
        watch_config: WatchConfig,
        buffer: MetricBuffer,
        registry: ModelRegistry,
    ):
        self.watch_config = watch_config
        self.buffer = buffer
        self.registry = registry

    def predict(self) -> Optional[PredictionResult]:
        """
        Run inference and return a PredictionResult with absolute timestamps.
        Returns None if the model is not ready or the buffer is insufficient.
        """
        if not self.registry.is_ready():
            logger.debug(f"[{self._metric_key()}] model not ready, skipping prediction")
            return None

        values = self.buffer.get_values()
        timestamps = self.buffer.get_timestamps()

        n_lags = self._n_lags()

        if len(values) < n_lags:
            logger.debug(
                f"[{self._metric_key()}] buffer has {len(values)} values, "
                f"need {n_lags} for prediction input"
            )
            return None

        try:
            X = build_prediction_input(
                values=values,
                lookback=self.watch_config.lookback,
                granularity=self.watch_config.granularity,
                timestamps=timestamps,
            )
        except ValueError as e:
            logger.error(f"[{self._metric_key()}] failed to build prediction input: {e}")
            return None

        model = self.registry.get_model()

        try:
            result = model.predict(X)
        except Exception as e:
            logger.error(f"[{self._metric_key()}] model.predict() failed: {e}")
            return None

        # attach absolute unix timestamps to each prediction step
        result.timestamps = self._build_timestamps(len(result.values))

        # attach active version id
        active = self.registry.active_version()
        if active:
            result.model_version = active.version_id

        return result

    def _build_timestamps(self, n_steps: int) -> np.ndarray:
        """
        Build absolute unix timestamps for each prediction step.
        Steps are offset from now by granularity increments.
        """
        now = datetime.now(timezone.utc).timestamp()
        granularity_secs = parse_duration_to_seconds(self.watch_config.granularity)
        return np.array([
            now + (i + 1) * granularity_secs
            for i in range(n_steps)
        ])

    def _n_lags(self) -> int:
        from sentinel.utils.time import parse_duration_to_steps
        return parse_duration_to_steps(
            self.watch_config.lookback,
            self.watch_config.granularity,
        )

    def _metric_key(self) -> str:
        cfg = self.watch_config
        if not cfg.labels:
            return cfg.metric
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(cfg.labels.items()))
        return f"{cfg.metric}{{{label_str}}}"