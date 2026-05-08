# sentinel/emitter/formatter.py

import time
import numpy as np
from datetime import datetime, timezone
from sentinel.pipeline.models.base import PredictionResult
from sentinel.config import WatchConfig
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

# sentinel appends this suffix to the original metric name
_PREDICTION_SUFFIX = "_sentinel_predicted"
_LOWER_BOUND_SUFFIX = "_sentinel_predicted_lower"
_UPPER_BOUND_SUFFIX = "_sentinel_predicted_upper"


class FormattedPrediction:
    """
    Holds a set of Prometheus-ready metric samples derived from
    a single PredictionResult. One FormattedPrediction per watched metric
    per emitter tick.
    """

    def __init__(
        self,
        metric_name: str,
        labels: dict[str, str],
        steps: list[dict],
    ):
        """
        metric_name : base predicted metric name e.g. "http_request_duration_seconds_sentinel_predicted"
        labels      : label set carried over from the original metric plus sentinel metadata labels
        steps       : list of {"timestamp": float, "value": float, "step": int}
        """
        self.metric_name = metric_name
        self.labels = labels
        self.steps = steps

    def __repr__(self) -> str:
        return (
            f"FormattedPrediction(metric={self.metric_name}, "
            f"labels={self.labels}, steps={len(self.steps)})"
        )


def format_prediction(
    watch_config: WatchConfig,
    result: PredictionResult,
    emit_confidence_bounds: bool = False,
) -> list[FormattedPrediction]:
    """
    Convert a PredictionResult into a list of FormattedPredictions
    ready to be registered as Prometheus gauges.

    One FormattedPrediction is always produced for the predicted values.
    Two more are produced if emit_confidence_bounds=True and bounds exist.

    Labels added by Sentinel on top of the original metric labels:
        sentinel_horizon : prediction horizon string e.g. "5m"
        sentinel_version : model version id e.g. "v3"
        sentinel_step    : step index within the horizon (1-indexed)
    """
    base_labels = dict(watch_config.labels)
    version = result.model_version or "unknown"
    output = []

    predicted_steps = []
    for i, (val, ts) in enumerate(zip(result.values, result.timestamps)):
        predicted_steps.append({
            "timestamp": float(ts),
            "value": float(val),
            "step": i + 1,
        })

    output.append(FormattedPrediction(
        metric_name=watch_config.metric + _PREDICTION_SUFFIX,
        labels={
            **base_labels,
            "sentinel_horizon": watch_config.horizon,
            "sentinel_version": version,
        },
        steps=predicted_steps,
    ))

    if emit_confidence_bounds and result.lower_bound is not None and result.upper_bound is not None:
        lower_steps = []
        upper_steps = []
        for i, (lo, hi, ts) in enumerate(zip(result.lower_bound, result.upper_bound, result.timestamps)):
            lower_steps.append({"timestamp": float(ts), "value": float(lo), "step": i + 1})
            upper_steps.append({"timestamp": float(ts), "value": float(hi), "step": i + 1})

        output.append(FormattedPrediction(
            metric_name=watch_config.metric + _LOWER_BOUND_SUFFIX,
            labels={**base_labels, "sentinel_horizon": watch_config.horizon, "sentinel_version": version},
            steps=lower_steps,
        ))

        output.append(FormattedPrediction(
            metric_name=watch_config.metric + _UPPER_BOUND_SUFFIX,
            labels={**base_labels, "sentinel_horizon": watch_config.horizon, "sentinel_version": version},
            steps=upper_steps,
        ))

    return output


def build_metric_key(metric: str, labels: dict[str, str]) -> str:
    """
    Build a unique string key for a metric+labelset combination.
    Used by the server to look up and update the correct Gauge.
    """
    if not labels:
        return metric
    label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return f"{metric}{{{label_str}}}"