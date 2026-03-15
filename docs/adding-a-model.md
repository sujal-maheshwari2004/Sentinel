# Adding a Custom Model

Sentinel's plugin interface lets you bring any model — any framework, any algorithm — as long as it satisfies a simple three-file contract.

---

## The contract

Your model lives in a directory with three files:

```
my-model/
  manifest.yaml    # metadata and configuration
  retrain.py       # training script
  infer.py         # inference script
```

---

## manifest.yaml

```yaml
name: my-latency-predictor
version: 1.0.0
description: "Predicts HTTP latency degradation 10 minutes ahead"

requires:
  metrics:
    - http_request_duration_seconds
    - http_requests_total
  min_history: 2h

wait:
  strategy: both       # time | rows | both
  time_hours: 6
  rows: 10000

retrain:
  schedule: "0 2 * * *"
  min_rows: 10000

inference:
  interval_seconds: 60
  timeout_seconds: 30
  fallback: continue_old    # continue_old | emit_zero | drop

outputs:
  - name: failure_probability
    type: probability
    horizon_seconds: 600
```

---

## retrain.py

Sentinel injects these environment variables before running your script:

| Variable | Description |
|---|---|
| `SNAPSHOT_PATH` | Path to the latest snapshot directory (contains parquet files) |
| `MODEL_OUTPUT_PATH` | Where you must write the trained artifact |
| `SENTINEL_RUN_ID` | Unique ID for this run |
| `SENTINEL_SERVICE_NAME` | Service name from `sentinel.yaml` identity block |
| `MLFLOW_TRACKING_URI` | MLflow server URL |

Your script must write an artifact to `MODEL_OUTPUT_PATH` and exit 0 on success.

```python
import os
import joblib
import pandas as pd
from pathlib import Path

# Optional — gives you MLflow + W&B logging for free
from versioning.experiment import ExperimentTracker

SNAPSHOT_PATH     = os.environ["SNAPSHOT_PATH"]
MODEL_OUTPUT_PATH = os.environ["MODEL_OUTPUT_PATH"]
SENTINEL_RUN_ID   = os.environ["SENTINEL_RUN_ID"]
TRACKING_URI      = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def train():
    tracker = ExperimentTracker(
        run_name=f"my-model-{SENTINEL_RUN_ID}",
        tracking_uri=TRACKING_URI,
    )

    df = pd.read_parquet(Path(SNAPSHOT_PATH) / "http_request_duration_seconds.parquet")

    # ... your training logic here ...
    model = MyModel()
    model.fit(df)

    tracker.log_params({"model": "my-model", "training_rows": len(df)})
    tracker.log_metrics({"val_loss": model.val_loss})

    Path(MODEL_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    tracker.log_artifact(MODEL_OUTPUT_PATH)
    tracker.finish()


if __name__ == "__main__":
    train()
```

---

## infer.py

Sentinel injects these environment variables:

| Variable | Description |
|---|---|
| `SNAPSHOT_PATH` | Path to the latest snapshot directory |
| `MODEL_PATH` | Path to the current trained artifact |
| `SENTINEL_SERVICE_NAME` | Service name from `sentinel.yaml` identity block |

Your script must write a JSON array to stdout and exit 0 on success.

```python
import json
import os
import joblib
import pandas as pd
from pathlib import Path

SNAPSHOT_PATH = os.environ["SNAPSHOT_PATH"]
MODEL_PATH    = os.environ["MODEL_PATH"]
SERVICE_NAME  = os.environ.get("SENTINEL_SERVICE_NAME", "unknown")


def infer():
    model = joblib.load(MODEL_PATH)
    df = pd.read_parquet(Path(SNAPSHOT_PATH) / "http_request_duration_seconds.parquet")

    score = model.predict(df)

    predictions = [
        {
            "service":         SERVICE_NAME,
            "metric":          "http_request_duration_seconds",
            "score":           round(float(score), 4),
            "horizon_seconds": 600,
            "metadata":        {},
        }
    ]

    print(json.dumps(predictions))


if __name__ == "__main__":
    infer()
```

---

## Register the model

```bash
sentinel add model ./my-model/
```

This adds the model to `sentinel.yaml`. Restart Sentinel and it will begin collecting data immediately.

---

## Notes

- Your scripts can use any language or framework — Sentinel only reads stdout from `infer.py` and checks the exit code of both scripts.
- The `ExperimentTracker` import is optional. If you don't use it, MLflow will not log anything for your model but the pipeline will still work.
- Snapshot parquet files are named after the metric, e.g. `http_request_duration_seconds.parquet`. Each file has `timestamp`, `value`, and label columns.
- If your model needs multiple metrics, read multiple parquet files from `SNAPSHOT_PATH`.