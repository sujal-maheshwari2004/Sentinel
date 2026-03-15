"""
memory-saturation — inference script

Sentinel injects these environment variables:
  SNAPSHOT_PATH         path to the latest snapshot directory
  MODEL_PATH            path to the trained model artifact
  SENTINEL_SERVICE_NAME name of the service being monitored
"""
import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.builtin.memory_saturation.architecture import MemorySaturationPredictor

SNAPSHOT_PATH = os.environ["SNAPSHOT_PATH"]
MODEL_PATH    = os.environ["MODEL_PATH"]
SERVICE_NAME  = os.environ.get("SENTINEL_SERVICE_NAME", "unknown")


def load_recent_memory_data(snapshot_path: str) -> pd.DataFrame:
    parquet_file = Path(snapshot_path) / "container_memory_usage_bytes.parquet"
    if not parquet_file.exists():
        return pd.DataFrame()
    return pd.read_parquet(parquet_file)


def infer():
    predictor: MemorySaturationPredictor = joblib.load(MODEL_PATH)

    df = load_recent_memory_data(SNAPSHOT_PATH)
    if df.empty:
        print(json.dumps([]))
        return

    features = predictor.extract_features(df)
    if features.empty:
        print(json.dumps([]))
        return

    score = predictor.predict_proba(features)

    predictions = [
        {
            "service":         SERVICE_NAME,
            "metric":          "container_memory_usage_bytes",
            "score":           round(score, 4),
            "horizon_seconds": MemorySaturationPredictor.PREDICTION_HORIZON,
            "metadata":        {},
        }
    ]

    print(json.dumps(predictions))


if __name__ == "__main__":
    infer()