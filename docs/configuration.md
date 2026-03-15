# Configuration Reference

All configuration lives in `sentinel.yaml` in the project root.

---

## sentinel

```yaml
sentinel:
  ingest_port: 9000      # port for POST /ingest (Prometheus remote_write target)
  metrics_port: 9001     # port for GET /metrics (Grafana scrape target)
  log_level: info        # debug | info | warning | error
```

---

## identity

Labels injected into every `predictivex_*` metric series emitted on `/metrics`. Used by Grafana to distinguish between services when multiple Prometheus instances each have their own Sentinel.

```yaml
identity:
  service_name: api-gateway    # required
  namespace: production
  cluster: us-east-1
  team: backend
  extra_labels:
    env: staging
```

---

## snapshot

Controls how often the in-memory buffer is flushed to disk and how long snapshots are retained.

```yaml
snapshot:
  dir: ./snapshots         # where snapshots are written
  retention_days: 30       # snapshots older than this are deleted
  interval_hours: 6        # how often to flush the buffer
```

---

## artifacts

Where trained model artifacts are stored.

```yaml
artifacts:
  dir: ./artifacts
```

---

## exposition

Safety guards for the `/metrics` endpoint to prevent series explosion downstream.

```yaml
exposition:
  max_series_total: 10000
  cardinality_warning_threshold: 5000
  drop_high_cardinality_labels:
    - instance
    - pod
```

---

## drift

Controls PSI-based data drift detection. When drift exceeds `retrain_threshold`, a retrain is triggered automatically.

```yaml
drift:
  enabled: true
  method: psi
  warning_threshold: 0.2    # emit predictivex_data_drift_psi warning
  retrain_threshold: 0.4    # auto-trigger retrain
```

---

## mlflow

```yaml
mlflow:
  tracking_uri: http://mlflow:5000
```

---

## dvc

```yaml
dvc:
  remote: local             # local | s3 | gcs | azure
  remote_path: ./dvc-storage
```

For S3, set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in `.env`.

---

## models

A list of models to run. Each model can be a builtin (name only) or a custom model (name + path).

```yaml
models:
  - name: latency-spikes          # builtin model
    wait:
      strategy: both              # time | rows | both
      time_hours: 6
      rows: 10000
    retrain:
      schedule: "0 2 * * *"      # cron expression
      min_rows: 10000
    inference:
      interval_seconds: 60
      timeout_seconds: 30
      fallback: continue_old      # continue_old | emit_zero | drop
      max_consecutive_errors: 5

  - name: my-custom-model         # custom model
    path: ./models/my-custom-model/
    wait:
      strategy: time
      time_hours: 24
    retrain:
      schedule: "0 3 * * *"
    inference:
      interval_seconds: 120
      timeout_seconds: 60
      fallback: emit_zero
```

### fallback options

| Value | Behaviour on inference failure |
|---|---|
| `continue_old` | Keep serving the last successful predictions |
| `emit_zero` | Replace predictions with zero scores |
| `drop` | Remove predictions from `/metrics` entirely |