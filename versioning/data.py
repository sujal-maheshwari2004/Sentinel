import logging
import subprocess

log = logging.getLogger(__name__)


def track_snapshot(snapshot_path: str) -> None:
    """
    Add a new snapshot directory to DVC tracking.
    This records the snapshot in .dvc so it can be reproduced later with dvc pull.
    Logs a warning and continues if DVC is not available.
    """
    _run_dvc(["dvc", "add", snapshot_path])


def push_snapshot(snapshot_path: str) -> None:
    """
    Push a tracked snapshot to the configured DVC remote storage.
    Only needed if using a remote (S3, GCS, Azure). Local storage is immediate.
    """
    _run_dvc(["dvc", "push", snapshot_path + ".dvc"])


def pull_snapshot(snapshot_path: str) -> None:
    """
    Pull a specific snapshot from the DVC remote back to disk.
    Used when reproducing a training run from a past commit.
    """
    _run_dvc(["dvc", "pull", snapshot_path + ".dvc"])


def _run_dvc(command: list) -> None:
    """
    Run a DVC shell command. Logs a warning on failure but never raises —
    DVC tracking is best-effort and should not interrupt the main pipeline.
    """
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        log.debug("DVC: %s", result.stdout.strip())
    except FileNotFoundError:
        log.warning("DVC not found on PATH — snapshot tracking skipped")
    except subprocess.CalledProcessError as exc:
        log.warning("DVC command failed: %s\n%s", " ".join(command), exc.stderr)