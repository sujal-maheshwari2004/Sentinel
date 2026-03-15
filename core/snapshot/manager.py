import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from core.buffer.store import BufferStore
from versioning.data import track_snapshot

log = logging.getLogger(__name__)


def flush_buffer_to_snapshot(buffer: BufferStore, snapshot_dir: str) -> str:
    """
    Write all current buffer contents to a timestamped Parquet snapshot on disk.
    Each metric gets its own Parquet file inside the snapshot directory.
    Updates the 'latest' symlink to point at the new snapshot.
    Notifies DVC to track the new snapshot.
    Returns the path to the new snapshot directory.
    """
    all_data = buffer.get_all()

    if not all_data:
        log.info("Buffer is empty — skipping snapshot flush")
        return None

    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    snapshot_path = Path(snapshot_dir) / timestamp
    snapshot_path.mkdir(parents=True, exist_ok=True)

    for metric_name, rows in all_data.items():
        if not rows:
            continue

        df = pd.DataFrame([
            {"timestamp": r.timestamp, "value": r.value, **r.labels}
            for r in rows
        ])

        # Sanitise the metric name so it is safe as a filename
        safe_name = metric_name.replace("/", "_").replace(".", "_")
        df.to_parquet(snapshot_path / f"{safe_name}.parquet", index=False)

    _update_latest_symlink(snapshot_dir, snapshot_path)
    track_snapshot(str(snapshot_path))

    log.info("Snapshot written: %s (%d metrics)", snapshot_path, len(all_data))
    return str(snapshot_path)


def delete_old_snapshots(snapshot_dir: str, retention_days: int) -> None:
    """
    Remove snapshot directories that are older than retention_days.
    The 'latest' symlink is never deleted.
    Called on a daily schedule by the scheduler.
    """
    cutoff = datetime.utcnow() - timedelta(days=retention_days)
    snapshot_root = Path(snapshot_dir)

    if not snapshot_root.exists():
        return

    for entry in snapshot_root.iterdir():
        if entry.name == "latest" or not entry.is_dir():
            continue
        try:
            entry_time = datetime.strptime(entry.name, "%Y-%m-%dT%H-%M-%S")
            if entry_time < cutoff:
                shutil.rmtree(entry)
                log.info("Deleted old snapshot: %s", entry.name)
        except ValueError:
            # Skip directories whose names don't match the timestamp format
            pass


def _update_latest_symlink(snapshot_dir: str, snapshot_path: Path) -> None:
    """Point the 'latest' symlink at the newest snapshot directory."""
    latest_link = Path(snapshot_dir) / "latest"

    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()

    latest_link.symlink_to(snapshot_path.resolve())