# sentinel/pipeline/versioning.py

import os
import json
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelVersion:
    """
    Metadata record for a single trained model version.
    Stored as JSON alongside the serialized model artifact.
    """
    version_id: str
    metric_key: str
    model_class: str
    trained_at: str                   # ISO8601 UTC
    training_policy: str              # "full_retrain" or "finetune"
    drift_score_at_trigger: float
    mae: float
    mape: float
    n_samples: int
    artifact_path: str
    status: str = "active"            # "active", "retired", "rolled_back"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelVersion":
        return cls(**d)


class VersionStore:
    """
    Manages model version metadata for a single metric key.
    Persists version records as JSON files in the artifact store.
    Handles pruning old versions beyond max_versions.

    Directory layout:
        artifact_store/
            {metric_key}/
                versions.json          <- version index
                v1/
                    model.joblib
                v2/
                    model.joblib
                ...
    """

    def __init__(self, metric_key: str, artifact_store: str, max_versions: int = 5):
        self.metric_key = metric_key
        self.artifact_store = artifact_store
        self.max_versions = max_versions

        self._lock = threading.Lock()
        self._base_dir = os.path.join(artifact_store, self._sanitize_key(metric_key))
        self._index_path = os.path.join(self._base_dir, "versions.json")
        self._versions: list[ModelVersion] = []

        os.makedirs(self._base_dir, exist_ok=True)
        self._load_index()

    def next_version_id(self) -> str:
        with self._lock:
            n = len(self._versions) + 1
            return f"v{n}"

    def artifact_path_for(self, version_id: str) -> str:
        """
        Returns the file path where a model artifact for this version should be saved.
        Creates the directory if it doesn't exist.
        """
        version_dir = os.path.join(self._base_dir, version_id)
        os.makedirs(version_dir, exist_ok=True)
        return os.path.join(version_dir, "model.joblib")

    def register(self, version: ModelVersion) -> None:
        """
        Add a new version to the index and persist.
        Prunes oldest retired versions if over max_versions.
        """
        with self._lock:
            self._versions.append(version)
            self._prune()
            self._save_index()
        logger.info(
            f"[{self.metric_key}] registered version {version.version_id} "
            f"policy={version.training_policy} mae={version.mae:.4f}"
        )

    def get_active(self) -> Optional[ModelVersion]:
        """
        Returns the currently active version, or None if no active version exists.
        """
        with self._lock:
            active = [v for v in self._versions if v.status == "active"]
            return active[-1] if active else None

    def get_all(self) -> list[ModelVersion]:
        with self._lock:
            return list(self._versions)

    def get_by_id(self, version_id: str) -> Optional[ModelVersion]:
        with self._lock:
            for v in self._versions:
                if v.version_id == version_id:
                    return v
            return None

    def promote(self, version_id: str) -> None:
        """
        Set a version as active, retire all others.
        Called after successful training + validation.
        """
        with self._lock:
            for v in self._versions:
                if v.version_id == version_id:
                    v.status = "active"
                elif v.status == "active":
                    v.status = "retired"
            self._save_index()
        logger.info(f"[{self.metric_key}] promoted version {version_id} to active")

    def rollback(self) -> Optional[ModelVersion]:
        """
        Retire the current active version and restore the most recent
        previously active (non-rolled-back) version.
        Returns the restored version, or None if no previous version exists.
        """
        with self._lock:
            active = [v for v in self._versions if v.status == "active"]
            retired = [v for v in self._versions if v.status == "retired"]

            if not active:
                logger.warning(f"[{self.metric_key}] rollback called but no active version")
                return None

            if not retired:
                logger.warning(f"[{self.metric_key}] rollback called but no retired version to restore")
                return None

            current = active[-1]
            current.status = "rolled_back"

            previous = retired[-1]
            previous.status = "active"

            self._save_index()

        logger.info(
            f"[{self.metric_key}] rolled back from {current.version_id} "
            f"to {previous.version_id}"
        )
        return previous

    def _prune(self) -> None:
        """
        Remove oldest retired versions beyond max_versions.
        Active and rolled_back versions are never pruned.
        """
        retired = [v for v in self._versions if v.status == "retired"]
        if len(self._versions) <= self.max_versions:
            return

        to_prune = retired[:len(self._versions) - self.max_versions]
        for v in to_prune:
            self._versions.remove(v)
            artifact = v.artifact_path
            if os.path.exists(artifact):
                try:
                    os.remove(artifact)
                    version_dir = os.path.dirname(artifact)
                    if not os.listdir(version_dir):
                        os.rmdir(version_dir)
                except OSError as e:
                    logger.warning(f"[{self.metric_key}] failed to delete artifact {artifact}: {e}")
            logger.debug(f"[{self.metric_key}] pruned version {v.version_id}")

    def _save_index(self) -> None:
        data = [v.to_dict() for v in self._versions]
        with open(self._index_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_index(self) -> None:
        if not os.path.exists(self._index_path):
            self._versions = []
            return
        with open(self._index_path, "r") as f:
            data = json.load(f)
        self._versions = [ModelVersion.from_dict(d) for d in data]
        logger.debug(f"[{self.metric_key}] loaded {len(self._versions)} versions from index")

    @staticmethod
    def _sanitize_key(key: str) -> str:
        return key.replace("{", "").replace("}", "").replace(",", "_").replace('"', "").replace("=", "-")