# sentinel/pipeline/registry.py

import threading
from typing import Optional
from sentinel.pipeline.models.base import BaseModel
from sentinel.pipeline.versioning import VersionStore, ModelVersion
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """
    Holds the live model reference for a single metric key.
    Handles atomic model swaps on promotion and rollback.

    Works with VersionStore for persistence and metadata.
    The registry owns the in-memory model object.
    VersionStore owns the on-disk metadata and artifacts.
    """

    def __init__(self, metric_key: str, version_store: VersionStore):
        self.metric_key = metric_key
        self.version_store = version_store

        self._model: Optional[BaseModel] = None
        self._active_version: Optional[ModelVersion] = None
        self._lock = threading.RLock()

    def get_model(self) -> Optional[BaseModel]:
        """
        Returns the currently active model instance.
        Returns None if no model has been promoted yet (cold start).
        """
        with self._lock:
            return self._model

    def promote(self, model: BaseModel, version: ModelVersion) -> None:
        """
        Atomically swap in a newly trained model.
        Saves the artifact, updates version store, and replaces
        the in-memory model reference in a single lock acquisition.

        Called by trainer.py after successful training + validation.
        """
        artifact_path = self.version_store.artifact_path_for(version.version_id)
        model.save(artifact_path)
        version.artifact_path = artifact_path

        with self._lock:
            self.version_store.register(version)
            self.version_store.promote(version.version_id)
            self._model = model
            self._active_version = version

        logger.info(
            f"[{self.metric_key}] promoted model version {version.version_id} "
            f"mae={version.mae:.4f}"
        )

    def rollback(self) -> bool:
        """
        Roll back to the previous stable version.
        Loads the previous model artifact from disk and swaps it in atomically.
        Returns True if rollback succeeded, False if no previous version exists.
        """
        previous = self.version_store.rollback()

        if previous is None:
            logger.warning(f"[{self.metric_key}] rollback failed — no previous version available")
            return False

        # reconstruct model instance from artifact
        model = self._load_model_from_version(previous)
        if model is None:
            return False

        with self._lock:
            self._model = model
            self._active_version = previous

        logger.info(f"[{self.metric_key}] rollback complete — now serving {previous.version_id}")
        return True

    def restore_from_disk(self, model_factory) -> bool:
        """
        On Sentinel startup, check if an active version exists on disk
        and restore it into memory so predictions resume without retraining.

        model_factory: callable that returns a new unfitted BaseModel instance
                       e.g. lambda: ExponentialSmoothingModel("1m", "5m", "30m")
        """
        active = self.version_store.get_active()

        if active is None:
            logger.info(f"[{self.metric_key}] no persisted model found, cold start required")
            return False

        model = model_factory()
        try:
            model.load(active.artifact_path)
        except Exception as e:
            logger.error(
                f"[{self.metric_key}] failed to load model from {active.artifact_path}: {e}"
            )
            return False

        with self._lock:
            self._model = model
            self._active_version = active

        logger.info(
            f"[{self.metric_key}] restored model version {active.version_id} from disk"
        )
        return True

    def active_version(self) -> Optional[ModelVersion]:
        with self._lock:
            return self._active_version

    def is_ready(self) -> bool:
        """
        Returns True if a fitted model is available for inference.
        """
        with self._lock:
            return self._model is not None and self._model.is_fitted

    def _load_model_from_version(self, version: ModelVersion) -> Optional[BaseModel]:
        """
        Reconstruct a model instance from a version record.
        Uses the model_class name stored in the version metadata
        to import and instantiate the correct class.
        """
        import importlib
        import sentinel.pipeline.models as models_module

        model_class = getattr(models_module, version.model_class, None)
        if model_class is None:
            logger.error(
                f"[{self.metric_key}] unknown model class '{version.model_class}' "
                f"in version {version.version_id}"
            )
            return None

        extra = version.extra or {}
        granularity = extra.get("granularity", "1m")
        horizon = extra.get("horizon", "5m")
        lookback = extra.get("lookback", "30m")

        model = model_class(granularity=granularity, horizon=horizon, lookback=lookback)
        try:
            model.load(version.artifact_path)
        except Exception as e:
            logger.error(
                f"[{self.metric_key}] failed to load artifact "
                f"{version.artifact_path}: {e}"
            )
            return None

        return model

    def __repr__(self) -> str:
        v = self._active_version
        return (
            f"ModelRegistry(metric={self.metric_key}, "
            f"ready={self.is_ready()}, "
            f"version={v.version_id if v else None})"
        )