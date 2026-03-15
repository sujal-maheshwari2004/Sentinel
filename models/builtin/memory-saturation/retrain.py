"""
memory-saturation — training script

Sentinel injects these environment variables:
  SNAPSHOT_PATH       path to the latest snapshot directory
  MODEL_OUTPUT_PATH   where this script must write the trained model artifact
  SENTINEL_RUN_ID     unique ID for this training run
"""
import os
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.builtin.memory_saturation.architecture import MemorySaturationPredictor
from versioning.experiment import ExperimentTracker

SNAPSHOT_PATH     = os.environ["SNAPSHOT_PATH"]
MODEL_OUTPUT_PATH = os.environ["MODEL_OUTPUT_PATH"]
SENTINEL_RUN_ID   = os.environ["SENTINEL_RUN_ID"]
TRACKING_URI      = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def load_memory_data(snapshot_path: str) -> pd.DataFrame:
    parquet_file = Path(snapshot_path) / "container_memory_usage_bytes.parquet"
    if not parquet_file.exists():
        raise FileNotFoundError(f"Required metric file not found: {parquet_file}")
    return pd.read_parquet(parquet_file)


def train():
    tracker = ExperimentTracker(
        run_name=f"memory-saturation-{SENTINEL_RUN_ID}",
        tracking_uri=TRACKING_URI,
    )

    df = load_memory_data(SNAPSHOT_PATH)
    print(f"Loaded {len(df)} rows from snapshot")

    predictor = MemorySaturationPredictor()
    features = predictor.extract_features(df)

    if features.empty or len(features) < 5:
        print("Not enough data to train — need at least 5 minute buckets")
        tracker.finish()
        sys.exit(1)

    predictor.fit(features)

    tracker.log_params({
        "model":              "memory-saturation",
        "training_rows":      len(df),
        "feature_buckets":    len(features),
        "observed_max_bytes": predictor.observed_max_bytes,
        "saturation_fraction": predictor.SATURATION_FRACTION,
        "snapshot_path":      SNAPSHOT_PATH,
    })
    tracker.log_metrics({
        "feature_buckets":    len(features),
        "observed_max_gb":    round(predictor.observed_max_bytes / 1e9, 3),
    })

    Path(MODEL_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictor, MODEL_OUTPUT_PATH)
    tracker.log_artifact(MODEL_OUTPUT_PATH)

    print(f"Model saved to {MODEL_OUTPUT_PATH}")
    tracker.finish()


if __name__ == "__main__":
    train()