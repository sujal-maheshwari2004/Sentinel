import logging
import os
from pathlib import Path

from core.registry import ModelInstance

log = logging.getLogger(__name__)


def swap_artifact(model: ModelInstance, new_artifact_path: str) -> None:
    """
    Atomically replace the current model artifact with a newly trained one.

    Uses os.replace() which is atomic on both POSIX and Windows — the old
    model is serving predictions right up until the swap happens, with no
    window where neither artifact is available.

    On the very first training run there is no existing artifact, so we
    just set the path directly without needing to replace anything.
    """
    new_path = Path(new_artifact_path)

    if not new_path.exists():
        raise FileNotFoundError(f"New artifact not found at: {new_artifact_path}")

    if model.artifact_path and Path(model.artifact_path).exists():
        os.replace(new_artifact_path, model.artifact_path)
        log.info("Swapped artifact for model: %s", model.config.name)
    else:
        # First artifact for this model
        model.artifact_path = new_artifact_path
        log.info("First artifact installed for model: %s", model.config.name)