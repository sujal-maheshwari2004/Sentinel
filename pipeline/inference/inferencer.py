import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import List, Optional

from core.config import SentinelConfig
from core.registry import ModelInstance, ModelState
from models.base import Prediction

log = logging.getLogger(__name__)


def run_inference(
    model: ModelInstance,
    config: SentinelConfig,
    metrics_store,          # exposition.metrics.MetricsStore — passed in to avoid circular import
) -> None:
    """
    Run the infer.py script for a model as an isolated subprocess.

    Flow:
      1. Check model is in an inferencing state
      2. Locate the artifact and latest snapshot
      3. Spawn infer.py as a subprocess with a timeout
      4. Parse the JSON predictions from stdout
      5. Push predictions to the MetricsStore so /metrics reflects them
      6. On timeout or error: apply the configured fallback behaviour
    """
    if model.state not in (ModelState.INFERENCING, ModelState.RETRAINING):
        return

    if not model.artifact_path or not Path(model.artifact_path).exists():
        log.warning("No artifact for %s — skipping inference", model.config.name)
        return

    snapshot_path = _get_latest_snapshot(config.snapshot.dir)
    if not snapshot_path:
        log.warning("No snapshot for %s — skipping inference", model.config.name)
        return

    infer_script = _resolve_script(model, "infer.py")

    env = {
        **os.environ,
        "SNAPSHOT_PATH":         snapshot_path,
        "MODEL_PATH":            model.artifact_path,
        "SENTINEL_SERVICE_NAME": config.identity.service_name,
        "SENTINEL_LOG_LEVEL":    config.log_level,
    }

    try:
        result = subprocess.run(
            ["python", infer_script],
            env=env,
            timeout=model.config.inference.timeout_seconds,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"infer.py exited with code {result.returncode}:\n{result.stderr}")

        predictions = _parse_predictions(result.stdout)

        metrics_store.update(model.config.name, predictions)
        model.last_inferred = time.time()
        model.consecutive_errors = 0

        log.debug("Inference complete for %s — %d predictions", model.config.name, len(predictions))

    except subprocess.TimeoutExpired:
        log.warning("Inference timed out for %s after %ds", model.config.name, model.config.inference.timeout_seconds)
        _apply_fallback(model, config, metrics_store)

    except Exception as exc:
        log.error("Inference error for %s: %s", model.config.name, exc)
        _apply_fallback(model, config, metrics_store)


# -----------------------------------------------------------------------------
# Private helpers
# -----------------------------------------------------------------------------

def _parse_predictions(stdout: str) -> List[Prediction]:
    """
    Parse the JSON array written to stdout by infer.py into Prediction objects.
    Raises ValueError if the output is not valid JSON or missing required fields.
    """
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"infer.py did not write valid JSON to stdout: {exc}") from exc

    return [
        Prediction(
            service=item["service"],
            metric=item["metric"],
            score=float(item["score"]),
            horizon_seconds=int(item["horizon_seconds"]),
            metadata=item.get("metadata", {}),
        )
        for item in data
    ]


def _apply_fallback(model: ModelInstance, config: SentinelConfig, metrics_store) -> None:
    """
    Apply the configured fallback behaviour after an inference failure.

    continue_old : keep the last predictions in the MetricsStore unchanged
    emit_zero    : overwrite with an empty prediction list
    drop         : remove this model's predictions from the MetricsStore entirely
    """
    model.consecutive_errors += 1

    if model.consecutive_errors >= model.config.inference.max_consecutive_errors:
        model.state = ModelState.ERROR
        metrics_store.clear(model.config.name)
        log.error(
            "Model %s moved to ERROR state after %d consecutive failures",
            model.config.name,
            model.consecutive_errors,
        )
        return

    fallback = model.config.inference.fallback

    if fallback == "emit_zero":
        metrics_store.update(model.config.name, [])
        log.info("Fallback emit_zero applied for %s", model.config.name)

    elif fallback == "drop":
        metrics_store.clear(model.config.name)
        log.info("Fallback drop applied for %s", model.config.name)

    # "continue_old" — do nothing, MetricsStore keeps the last predictions


def _get_latest_snapshot(snapshot_dir: str) -> Optional[str]:
    """Return the resolved path of the latest snapshot, or None if none exists."""
    latest = Path(snapshot_dir) / "latest"
    if latest.exists():
        return str(latest.resolve())
    return None


def _resolve_script(model: ModelInstance, script_name: str) -> str:
    """Return the path to infer.py for either a custom or builtin model."""
    if model.config.path:
        return str(Path(model.config.path) / script_name)
    return str(Path("models/builtin") / model.config.name / script_name)