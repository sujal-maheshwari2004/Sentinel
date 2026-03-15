"""
latency-spikes — inference script

Sentinel injects these environment variables:
  SNAPSHOT_PATH         path to the latest snapshot directory
  MODEL_PATH            path to the trained model artifact
  SENTINEL_SERVICE_NAME name of the service being monitored

Writes a JSON array of predictions to stdout.
Exit 0 on success, non-zero on failure.
"""
import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.builtin.latency_spikes.architecture import LatencySpikePredictor

SNAPSHOT_PATH    = os.environ["SNAPSHOT_PATH"]
MODEL_PATH       = os.environ["MODEL_PATH"]
SERVICE_NAME     = os.environ.get("SENTINEL_SERVICE_NAME", "unknown")


def load_recent_latency_data(snapshot_path: str) -> pd.DataFrame:
    parquet_file = Path(snapshot_path) / "http_request_duration_seconds.parquet"
    if not parquet_file.exists():
        return pd.DataFrame()
    return pd.read_parquet(parquet_file)


def infer():
    predictor: LatencySpikePredictor = joblib.load(MODEL_PATH)

    df = load_recent_latency_data(SNAPSHOT_PATH)
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
            "service":          SERVICE_NAME,
            "metric":           "http_request_duration_seconds",
            "score":            round(score, 4),
            "horizon_seconds":  LatencySpikePredictor.PREDICTION_HORIZON,
            "metadata":         {},
        }
    ]

    print(json.dumps(predictions))


if __name__ == "__main__":
    infer()