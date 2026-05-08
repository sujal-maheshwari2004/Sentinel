# Sentinel

An MLOps pipeline that sits transparently between Prometheus and Grafana. Sentinel scrapes your metrics, trains lightweight time series models, and emits predictions back as standard Prometheus metrics — so Grafana can display real and predicted data side by side with zero changes to either Prometheus or Grafana.

```
Prometheus → Sentinel → /metrics → Grafana
```

## How it works

Sentinel has three internal parts:

**Data Ingestor** pulls metrics from Prometheus on a configurable interval and maintains a sliding window buffer per metric. Once the buffer fills the configured lookback window, cold start training begins automatically.

**MLOps Pipeline** trains one model per watched metric. After cold start, it monitors prediction drift using a rolling MAE window. When drift crosses a threshold it triggers either a finetune (low severity) or full retrain (high severity). All training runs in a background thread so predictions keep serving during retraining. Every trained model is versioned, persisted to disk, and can be rolled back.

**Data Emitter** exposes predictions on a `/metrics` endpoint in standard Prometheus exposition format. Grafana scrapes it the same way it scrapes Prometheus. No plugins, no code changes on either side.

## Installation

```bash
pip install sentinel-mlops
```

## Quick start

```python
from sentinel import Sentinel, SentinelConfig, WatchConfig
from sentinel.pipeline.models import ExponentialSmoothingModel

config = SentinelConfig(
    prometheus_url="http://localhost:9090",
    emitter_port=8080,
    artifact_store="./sentinel_artifacts",
    watches=[
        WatchConfig(
            metric="http_request_duration_seconds",
            labels={"job": "api"},
            model_class=ExponentialSmoothingModel,
            granularity="1m",
            horizon="5m",
            lookback="30m",
            cron="0 */6 * * *",
        )
    ]
)

Sentinel(config).start()
```

Point Grafana at `http://localhost:8080` as a Prometheus datasource. Done.

## Grafana setup

Add a second Prometheus datasource in Grafana pointing to Sentinel's emitter port:

```
URL: http://localhost:8080
```

Then on any panel, add a second query using this datasource. Sentinel emits predictions with the original metric name suffixed with `_sentinel_predicted`:

```promql
# original metric
http_request_duration_seconds{job="api"}

# sentinel prediction
http_request_duration_seconds_sentinel_predicted{job="api", sentinel_horizon="5m"}
```

Both series appear on the same panel as overlapping lines.

## Configuration

### SentinelConfig

| Field | Description | Default |
|---|---|---|
| `prometheus_url` | URL of your Prometheus instance | `"http://localhost:9090"` |
| `emitter_port` | Port Sentinel serves `/metrics` on | `8080` |
| `artifact_store` | Directory for persisted model artifacts | `"./sentinel_artifacts"` |
| `max_versions_per_metric` | How many model versions to retain per metric | `5` |
| `scrape_timeout` | HTTP timeout for Prometheus requests in seconds | `10` |
| `emit_confidence_bounds` | Whether to emit lower/upper bound gauges | `False` |
| `watches` | List of `WatchConfig` instances | `[]` |

### WatchConfig

| Field | Description | Default |
|---|---|---|
| `metric` | Prometheus metric name | Required |
| `labels` | Label filters as a dict | `{}` |
| `model_class` | Model class to use | Required |
| `granularity` | Resolution of data and predictions e.g. `"1m"` | `"1m"` |
| `horizon` | How far ahead to predict e.g. `"5m"` | `"5m"` |
| `lookback` | Feature window size e.g. `"30m"` | `"30m"` |
| `cron` | Retraining schedule as a cron string | `"0 */6 * * *"` |
| `drift_finetune_threshold` | MAE above this triggers finetune | `0.1` |
| `drift_retrain_threshold` | MAE above this triggers full retrain | `0.3` |

## Available models

All models implement the same interface and are interchangeable.

### LinearTrendModel

Least squares linear regression. Fastest. Best for metrics with a clear linear trend and no seasonality. Good baseline.

```python
from sentinel.pipeline.models import LinearTrendModel
```

### ExponentialSmoothingModel

Holt's double exponential smoothing. Handles level and trend. Good for metrics with gradual trends but no strong seasonality. Lightweight and interpretable.

```python
from sentinel.pipeline.models import ExponentialSmoothingModel
```

### ARIMAModel

ARIMA(p, d, q) implemented without external dependencies. Suitable for stationary or trend-stationary metrics. Configurable order.

```python
from sentinel.pipeline.models import ARIMAModel

WatchConfig(
    model_class=ARIMAModel,
    ...
)
```

### SGDRegressorModel

Online learning via stochastic gradient descent. Most finetune-friendly — continues training from existing weights without reinitializing. Best for metrics that evolve continuously.

```python
from sentinel.pipeline.models import SGDRegressorModel
```

## Drift detection and retraining

Sentinel continuously compares its predictions against actual values fetched from Prometheus. It maintains a rolling MAE window per metric and classifies drift severity:

| Condition | Action |
|---|---|
| `MAE < drift_finetune_threshold` | No drift, keep serving current model |
| `drift_finetune_threshold <= MAE < drift_retrain_threshold` | Low drift, finetune on recent data |
| `MAE >= drift_retrain_threshold` | High drift, full retrain from scratch |

Retraining also fires on the cron schedule regardless of drift, so models stay fresh even on stable metrics.

## Model versioning and rollback

Every training run produces a versioned artifact stored in `artifact_store`. To roll back a metric to its previous model version:

```python
sentinel = Sentinel(config)
sentinel.rollback(metric="http_request_duration_seconds", labels={"job": "api"})
```

On restart, Sentinel automatically restores the last active model from disk so predictions resume without waiting for a cold start.

## Checking status

```python
status = sentinel.status()
# returns per-metric dict with:
# ready, active_version, active_version_mae, drift_mae, buffer_fill
```

## Emitted metrics

For each watched metric Sentinel emits:

| Metric | Description |
|---|---|
| `{metric}_sentinel_predicted` | Predicted values, one gauge per horizon step |
| `{metric}_sentinel_predicted_lower` | Lower confidence bound (if `emit_confidence_bounds=True`) |
| `{metric}_sentinel_predicted_upper` | Upper confidence bound (if `emit_confidence_bounds=True`) |

Each gauge carries these labels in addition to the original metric labels:

| Label | Description |
|---|---|
| `sentinel_horizon` | Prediction horizon e.g. `"5m"` |
| `sentinel_version` | Model version that produced the prediction e.g. `"v3"` |
| `sentinel_step` | Step index within the horizon, 1-indexed |

## Bringing your own model

Subclass `BaseModel` and implement five methods:

```python
from sentinel.pipeline.models import BaseModel, PredictionResult, TrainingResult
import numpy as np

class MyModel(BaseModel):

    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        ...

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        ...

    def predict(self, X: np.ndarray) -> PredictionResult:
        ...

    def save(self, path: str) -> None:
        ...

    def load(self, path: str) -> None:
        ...
```

Then pass it as `model_class` in `WatchConfig`.

`X` has shape `(n_samples, n_features)` where features are auto-generated lag values, rolling mean, rolling std, and cyclical time encodings. `y` has shape `(n_samples,)`. `predict()` receives `X` with shape `(1, n_features)` and must return a `PredictionResult` with `values` of length `horizon_steps`.

## Requirements

- Python >= 3.11
- numpy >= 1.26.0
- httpx >= 0.27.0
- joblib >= 1.3.0
- prometheus-client >= 0.20.0
- croniter >= 2.0.0

## License

MIT