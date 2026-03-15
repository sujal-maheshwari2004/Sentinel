import logging
import os
import subprocess
import time
import uuid
from pathlib import Path

from core.config import SentinelConfig
from core.registry import ModelInstance, ModelRegistry, ModelState
from pipeline.hotswap.swapper import swap_artifact
from versioning.experiment import ExperimentTracker

log = logging.getLogger(__name__)


def run_training(model: ModelInstance, config: SentinelConfig) -> None:
    """
    Run the retrain.py script for a model as an isolated subprocess.

    Flow:
      1. Check model is in a trainable state
      2. Locate the latest snapshot
      3. Set up paths and environment variables
      4. Spawn retrain.py as a subprocess
      5. On success: swap in the new artifact, log to MLflow/W&B
      6. On failure: log the error, move model to ERROR state
    """
    if model.state not in (ModelState.WAITING, ModelState.INFERENCING):
        log.info("Skipping training for %s — current state is %s", model.config.name, model.state.name)
        return

    snapshot_path = _get_latest_snapshot(config.snapshot.dir)
    if not snapshot_path:
        log.warning("No snapshot available for %s — skipping training", model.config.name)
        return

    run_id = str(uuid.uuid4())[:8]
    output_path = Path(config.artifacts.dir) / model.config.name / f"{run_id}.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    retrain_script = _resolve_script(model, "retrain.py")

    env = {
        **os.environ,
        "SNAPSHOT_PATH":         snapshot_path,
        "MODEL_OUTPUT_PATH":     str(output_path),
        "SENTINEL_RUN_ID":       run_id,
        "SENTINEL_SERVICE_NAME": config.identity.service_name,
        "SENTINEL_LOG_LEVEL":    config.log_level,
    }

    # Transition to TRAINING or RETRAINING so the scheduler and exposition
    # layer know what is happening. RETRAINING keeps the old artifact live.
    if model.state == ModelState.INFERENCING:
        model.state = ModelState.RETRAINING
    else:
        model.state = ModelState.TRAINING

    tracker = ExperimentTracker(
        run_name=f"{model.config.name}-{run_id}",
        tracking_uri=config.mlflow_tracking_uri,
    )
    tracker.log_params({
        "model_name":    model.config.name,
        "snapshot_path": snapshot_path,
        "run_id":        run_id,
    })

    try:
        result = subprocess.run(
            ["python", retrain_script],
            env=env,
            timeout=3600,           # 1 hour hard limit for any training run
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"retrain.py exited with code {result.returncode}:\n{result.stderr}")

        if not output_path.exists():
            raise FileNotFoundError(f"retrain.py did not write artifact to {output_path}")

        tracker.log_artifact(str(output_path))
        mlflow_run_id = tracker.finish()

        swap_artifact(model, str(output_path))

        model.last_trained = time.time()
        model.mlflow_run_id = mlflow_run_id
        model.consecutive_errors = 0
        model.state = ModelState.INFERENCING

        log.info("Training complete for %s (run %s)", model.config.name, run_id)

    except Exception as exc:
        tracker.finish()
        model.state = ModelState.ERROR
        log.error("Training failed for %s: %s", model.config.name, exc)


def check_and_train_waiting_models(registry: ModelRegistry, config: SentinelConfig) -> None:
    """
    Check every WAITING model and kick off training for any that have
    accumulated enough data to meet their configured threshold.
    Called on a schedule by the scheduler.
    """
    for model in registry.get_by_state(ModelState.WAITING):
        if _wait_threshold_met(model):
            log.info("Wait threshold met for %s — starting training", model.config.name)
            run_training(model, config)


# -----------------------------------------------------------------------------
# Private helpers
# -----------------------------------------------------------------------------

def _wait_threshold_met(model: ModelInstance) -> bool:
    """
    Return True if the model has collected enough data to begin training.
    Respects the wait strategy: time, rows, or both (both must be satisfied).
    """
    hours_waited = (time.time() - model.waiting_since) / 3600
    time_ok = hours_waited >= model.config.wait.time_hours
    rows_ok = model.rows_collected >= model.config.wait.rows

    if model.config.wait.strategy == "time":
        return time_ok
    if model.config.wait.strategy == "rows":
        return rows_ok
    return time_ok and rows_ok   # "both"


def _get_latest_snapshot(snapshot_dir: str) -> str:
    """Return the resolved path of the latest snapshot, or None if none exists."""
    latest = Path(snapshot_dir) / "latest"
    if latest.exists():
        return str(latest.resolve())
    return None


def _resolve_script(model: ModelInstance, script_name: str) -> str:
    """
    Return the path to a model script (retrain.py or infer.py).
    Custom models use their configured path.
    Builtin models live under models/builtin/<name>/.
    """
    if model.config.path:
        return str(Path(model.config.path) / script_name)
    return str(Path("models/builtin") / model.config.name / script_name)