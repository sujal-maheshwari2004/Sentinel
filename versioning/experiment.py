import logging
import os
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Unified logging interface over MLflow and Weights & Biases.

    MLflow is always active.
    W&B activates automatically when WANDB_API_KEY is set in the environment.
    If W&B fails to initialise for any reason it is silently skipped —
    MLflow always remains the source of truth.

    Usage in a retrain.py script:
        from versioning.experiment import ExperimentTracker
        tracker = ExperimentTracker(run_name="my-model-abc123", tracking_uri="http://mlflow:5000")
        tracker.log_params({"epochs": 50, "lr": 0.001})
        tracker.log_metrics({"val_loss": 0.032})
        tracker.log_artifact("./artifacts/my-model/abc123.pkl")
        run_id = tracker.finish()
    """

    def __init__(self, run_name: str, tracking_uri: str):
        self._run_name = run_name
        self._mlflow_run = self._start_mlflow(run_name, tracking_uri)
        self._wandb_run = self._start_wandb(run_name)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters and training configuration."""
        import mlflow
        mlflow.log_params(params)

        if self._wandb_run:
            self._wandb_run.config.update(params)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log training metrics such as loss, accuracy, or inference latency."""
        import mlflow
        mlflow.log_metrics(metrics)

        if self._wandb_run:
            self._wandb_run.log(metrics)

    def log_artifact(self, artifact_path: str) -> None:
        """Log a file as an artifact — typically the trained model file."""
        import mlflow
        mlflow.log_artifact(artifact_path)

        if self._wandb_run:
            self._wandb_run.save(artifact_path)

    def finish(self) -> str:
        """
        Close both tracking runs.
        Returns the MLflow run ID so it can be stored on the ModelInstance.
        """
        import mlflow
        run_id = self._mlflow_run.info.run_id
        mlflow.end_run()

        if self._wandb_run:
            self._wandb_run.finish()

        return run_id

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _start_mlflow(self, run_name: str, tracking_uri: str):
        """Start an MLflow run. Always called — MLflow is not optional."""
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        run = mlflow.start_run(run_name=run_name)
        log.info("MLflow run started: %s", run.info.run_id)
        return run

    def _start_wandb(self, run_name: str) -> Optional[object]:
        """
        Start a W&B run if WANDB_API_KEY is present.
        Returns None silently if the key is missing or W&B fails.
        """
        if not os.getenv("WANDB_API_KEY"):
            return None

        try:
            import wandb
            run = wandb.init(name=run_name, reinit=True)
            log.info("W&B run started: %s", run_name)
            return run
        except Exception as exc:
            log.warning("W&B init failed, skipping: %s", exc)
            return None