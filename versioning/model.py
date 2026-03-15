import logging

log = logging.getLogger(__name__)


def promote_to_production(model_name: str, run_id: str, tracking_uri: str) -> None:
    """
    Promote a specific MLflow run to the Production stage in the model registry.
    Any previously Production model is moved to Archived automatically.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Register the artifact from this run as a new model version
    model_uri = f"runs:/{run_id}/artifacts"
    version = mlflow.register_model(model_uri=model_uri, name=model_name)

    # Transition the new version to Production
    client.transition_model_version_stage(
        name=model_name,
        version=version.version,
        stage="Production",
        archive_existing_versions=True,   # moves old Production to Archived
    )
    log.info("Model %s version %s promoted to Production", model_name, version.version)


def rollback_to_run(model_name: str, run_id: str, tracking_uri: str, artifacts_dir: str) -> str:
    """
    Pull the artifact from a previous MLflow run back to disk so the
    hotswap can install it. Returns the local path of the restored artifact.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)

    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        dst_path=artifacts_dir,
    )
    log.info("Rolled back model %s to run %s -> %s", model_name, run_id, local_path)
    return local_path