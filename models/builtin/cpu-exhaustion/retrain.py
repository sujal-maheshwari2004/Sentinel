"""
cpu-exhaustion — training script

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

from models.builtin.cpu_exhaustion.architecture import CpuExhaustionPredictor
from versioning.experiment import ExperimentTracker

SNAPSHOT_PATH     = os.environ["SNAPSHOT_PATH"]
MODEL_OUTPUT_PATH = os.environ["MODEL_OUTPUT_PATH"]
SENTINEL_RUN_ID   = os.environ["SENTINEL_RUN_ID"]
TRACKING_URI      = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def load_cpu_data(snapshot_path: str) -> pd.DataFrame:
    parquet_file = Path(snapshot_path) / "process_cpu_seconds_total.parquet"
    if not parquet_file.exists():
        raise FileNotFoundError(f"Required metric file not found: {parquet_file}")
    return pd.read_parquet(parquet_file)


def train():
    tracker = ExperimentTracker(
        run_name=f"cpu-exhaustion-{SENTINEL_RUN_ID}",
        tracking_uri=TRACKING_URI,
    )

    df = load_cpu_data(SNAPSHOT_PATH)
    print(f"Loaded {len(df)} rows from snapshot")

    predictor = CpuExhaustionPredictor()
    features = predictor.extract_features(df)

    if features.empty or len(features) < 10:
        print("Not enough data to train — need at least 10 feature windows")
        tracker.finish()
        sys.exit(1)

    labels = predictor.create_labels(features)
    exhaustion_rate = labels.mean()
    print(f"Exhaustion rate in training data: {exhaustion_rate:.2%}")

    predictor.fit(features, labels)

    tracker.log_params({
        "model":           "cpu-exhaustion",
        "n_estimators":    predictor.model.n_estimators,
        "max_depth":       predictor.model.max_depth,
        "training_rows":   len(df),
        "feature_windows": len(features),
        "snapshot_path":   SNAPSHOT_PATH,
    })
    tracker.log_metrics({
        "exhaustion_rate": float(exhaustion_rate),
        "feature_windows": len(features),
    })

    Path(MODEL_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictor, MODEL_OUTPUT_PATH)
    tracker.log_artifact(MODEL_OUTPUT_PATH)

    print(f"Model saved to {MODEL_OUTPUT_PATH}")
    tracker.finish()


if __name__ == "__main__":
    train()