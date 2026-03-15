"""
latency-spikes — training script

Sentinel injects these environment variables:
  SNAPSHOT_PATH       path to the latest snapshot directory (parquet files)
  MODEL_OUTPUT_PATH   where this script must write the trained model artifact
  SENTINEL_RUN_ID     unique ID for this training run
"""
import os
import sys
from pathlib import Path

import joblib
import pandas as pd

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.builtin.latency_spikes.architecture import LatencySpikePredictor
from versioning.experiment import ExperimentTracker

SNAPSHOT_PATH     = os.environ["SNAPSHOT_PATH"]
MODEL_OUTPUT_PATH = os.environ["MODEL_OUTPUT_PATH"]
SENTINEL_RUN_ID   = os.environ["SENTINEL_RUN_ID"]
TRACKING_URI      = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def load_latency_data(snapshot_path: str) -> pd.DataFrame:
    """Load http_request_duration_seconds samples from the snapshot."""
    parquet_file = Path(snapshot_path) / "http_request_duration_seconds.parquet"
    if not parquet_file.exists():
        raise FileNotFoundError(f"Required metric file not found: {parquet_file}")
    return pd.read_parquet(parquet_file)


def train():
    tracker = ExperimentTracker(
        run_name=f"latency-spikes-{SENTINEL_RUN_ID}",
        tracking_uri=TRACKING_URI,
    )

    df = load_latency_data(SNAPSHOT_PATH)
    print(f"Loaded {len(df)} rows from snapshot")

    predictor = LatencySpikePredictor()
    features = predictor.extract_features(df)

    if features.empty or len(features) < 10:
        print("Not enough data to train — need at least 10 feature windows")
        tracker.finish()
        sys.exit(1)

    labels = predictor.create_labels(features)

    spike_rate = labels.mean()
    print(f"Spike rate in training data: {spike_rate:.2%}")

    predictor.fit(features, labels)

    tracker.log_params({
        "model":            "latency-spikes",
        "n_estimators":     predictor.model.n_estimators,
        "max_depth":        predictor.model.max_depth,
        "training_rows":    len(df),
        "feature_windows":  len(features),
        "snapshot_path":    SNAPSHOT_PATH,
    })
    tracker.log_metrics({
        "spike_rate":       float(spike_rate),
        "feature_windows":  len(features),
    })

    Path(MODEL_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictor, MODEL_OUTPUT_PATH)
    tracker.log_artifact(MODEL_OUTPUT_PATH)

    print(f"Model saved to {MODEL_OUTPUT_PATH}")
    tracker.finish()


if __name__ == "__main__":
    train()