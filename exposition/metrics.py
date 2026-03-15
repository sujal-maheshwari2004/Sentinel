import threading
from typing import Dict, List

from core.config import ExpositionConfig, IdentityConfig
from core.registry import ModelInstance, ModelState
from models.base import Prediction


class MetricsStore:
    """
    Holds the latest predictions from every active model and renders
    them as Prometheus text exposition format on demand.

    Thread-safe — the inference pipeline writes predictions from a
    background thread while FastAPI reads them on every GET /metrics.
    """

    def __init__(self, identity: IdentityConfig, exposition: ExpositionConfig):
        self.identity = identity
        self.exposition = exposition
        self._predictions: Dict[str, List[Prediction]] = {}
        self._lock = threading.Lock()

    def update(self, model_name: str, predictions: List[Prediction]) -> None:
        """Replace the full prediction set for a given model."""
        with self._lock:
            self._predictions[model_name] = predictions

    def clear(self, model_name: str) -> None:
        """Remove all predictions for a model e.g. when it enters ERROR state."""
        with self._lock:
            self._predictions.pop(model_name, None)

    def render(self, models: List[ModelInstance] = None) -> str:
        """
        Render all predictions and model lifecycle metrics as
        Prometheus text exposition format.
        Called on every GET /metrics request.
        """
        identity_labels = self._build_identity_label_str()

        with self._lock:
            predictions = dict(self._predictions)

        lines = []
        lines += _render_predictions(predictions, identity_labels)

        if models:
            lines += _render_model_lifecycle(models, identity_labels)
            lines += _render_cardinality(predictions, identity_labels, self.exposition)

        return "\n".join(lines) + "\n"

    def _build_identity_label_str(self) -> str:
        """Build the label string injected into every metric series."""
        labels = {
            "service":   self.identity.service_name,
            "namespace": self.identity.namespace,
            "cluster":   self.identity.cluster,
        }
        if self.identity.team:
            labels["team"] = self.identity.team

        labels.update(self.identity.extra_labels)

        # Drop any labels the user has configured to suppress
        return ",".join(f'{k}="{v}"' for k, v in labels.items())


# -----------------------------------------------------------------------------
# Rendering helpers — each returns a list of lines
# -----------------------------------------------------------------------------

def _render_predictions(
    predictions: Dict[str, List[Prediction]],
    identity_labels: str,
) -> List[str]:
    """Render failure probability scores for all active models."""
    lines = [
        "# HELP predictivex_failure_probability Predicted probability of service failure",
        "# TYPE predictivex_failure_probability gauge",
    ]
    for model_name, preds in predictions.items():
        for p in preds:
            extra = ""
            if p.metadata:
                extra = "," + ",".join(f'{k}="{v}"' for k, v in p.metadata.items())
            label_str = (
                f'{identity_labels},'
                f'model="{model_name}",'
                f'metric="{p.metric}",'
                f'horizon="{p.horizon_seconds}s"'
                f'{extra}'
            )
            lines.append(f"predictivex_failure_probability{{{label_str}}} {p.score:.4f}")
    return lines


def _render_model_lifecycle(
    models: List[ModelInstance],
    identity_labels: str,
) -> List[str]:
    """Render model state, warmup progress, and error counters."""
    lines = [
        "# HELP predictivex_model_state Lifecycle state (0=waiting 1=training 2=inferencing 3=retraining 4=error)",
        "# TYPE predictivex_model_state gauge",
    ]
    for model in models:
        label_str = f'{identity_labels},model="{model.config.name}"'
        lines.append(f"predictivex_model_state{{{label_str}}} {model.state.value}")

    lines += [
        "# HELP predictivex_model_warmup_progress Fraction of row threshold collected (0 to 1)",
        "# TYPE predictivex_model_warmup_progress gauge",
    ]
    for model in models:
        if model.state != ModelState.WAITING:
            continue
        if model.config.wait.rows == 0:
            continue
        progress = min(model.rows_collected / model.config.wait.rows, 1.0)
        label_str = f'{identity_labels},model="{model.config.name}"'
        lines.append(f"predictivex_model_warmup_progress{{{label_str}}} {progress:.4f}")

    lines += [
        "# HELP predictivex_model_consecutive_errors Number of consecutive inference errors",
        "# TYPE predictivex_model_consecutive_errors gauge",
    ]
    for model in models:
        label_str = f'{identity_labels},model="{model.config.name}"'
        lines.append(f"predictivex_model_consecutive_errors{{{label_str}}} {model.consecutive_errors}")

    return lines


def _render_cardinality(
    predictions: Dict[str, List[Prediction]],
    identity_labels: str,
    exposition: ExpositionConfig,
) -> List[str]:
    """Render cardinality gauges and warning flags per model."""
    lines = [
        "# HELP predictivex_cardinality_current Number of active prediction series for this model",
        "# TYPE predictivex_cardinality_current gauge",
    ]
    for model_name, preds in predictions.items():
        label_str = f'{identity_labels},model="{model_name}"'
        lines.append(f"predictivex_cardinality_current{{{label_str}}} {len(preds)}")

    lines += [
        "# HELP predictivex_cardinality_warning 1 if series count exceeds warning threshold",
        "# TYPE predictivex_cardinality_warning gauge",
    ]
    total_series = sum(len(p) for p in predictions.values())
    warning = 1 if total_series >= exposition.cardinality_warning_threshold else 0
    lines.append(f"predictivex_cardinality_warning{{{identity_labels}}} {warning}")

    return lines