# Architecture

## Overview

Sentinel is a single service with two network surfaces:

- `POST /ingest` — receives Prometheus `remote_write` payloads
- `GET /metrics` — serves predictions in Prometheus text exposition format

Everything else is internal.

```
[Prometheus]
     |
     | remote_write (snappy-compressed protobuf)
     v
[POST /ingest]
     |
     v
[BufferStore]                     in-memory rolling window
     |
     | on schedule (interval_hours)
     v
[SnapshotManager]                 flush to parquet on disk
     |                            DVC tracks each snapshot
     v
[ModelRegistry]                   owns all model state machines
     |
     +-- WAITING  --> TRAINING --> INFERENCING --> RETRAINING
                          |               |              |
                     retrain.py       infer.py      retrain.py
                     subprocess       subprocess    subprocess
                          |               |
                       MLflow          MetricsStore
                       W&B (opt)
                                         |
                                    [GET /metrics]
                                         |
                                    [Grafana]
```

## Components

**BufferStore** (`core/buffer/store.py`) — Thread-safe in-memory store keyed by metric name. Oldest samples are evicted hourly. This is the single source of truth for live data.

**SnapshotManager** (`core/snapshot/manager.py`) — Periodically flushes the buffer to timestamped Parquet directories on disk. Updates a `latest` symlink after each flush. Calls DVC to track each new snapshot.

**ModelRegistry** (`core/registry.py`) — Owns every `ModelInstance` and its state. All state transitions go through the registry. Nothing else mutates model state.

**Trainer** (`pipeline/training/trainer.py`) — Spawns `retrain.py` as a subprocess. Passes snapshot path, output path, and run ID as environment variables. On success, calls the hotswap to install the new artifact. Logs to MLflow and W&B.

**Inferencer** (`pipeline/inference/inferencer.py`) — Spawns `infer.py` as a subprocess on a fixed interval. Reads JSON predictions from stdout. Pushes to the MetricsStore. Handles timeouts and applies fallback behaviour.

**Hotswap** (`pipeline/hotswap/swapper.py`) — Atomically replaces the model artifact using `os.replace()`. The old model serves predictions until the exact moment of swap. No gap.

**MetricsStore** (`exposition/metrics.py`) — Holds the latest predictions from all active models. Renders to Prometheus text format on every `GET /metrics` request.

**Scheduler** (`core/scheduler/runner.py`) — APScheduler instance. Wires all recurring jobs: snapshot flush, snapshot cleanup, buffer eviction, wait threshold checks, per-model inference intervals, and per-model retrain cron schedules.

## Model lifecycle

```
REGISTERED
    |
    v
WAITING          data accumulating, no predictions
    |
    | wait threshold met (time, rows, or both)
    v
TRAINING         retrain.py subprocess running
    |
    v
INFERENCING      infer.py running on interval, predictions active
    |
    | retrain schedule or drift trigger
    v
RETRAINING       old model still inferencing during retrain
    |
    v
INFERENCING      new artifact installed via hotswap
```

If retrain or inference fails repeatedly, the model transitions to `ERROR`. It stays in `ERROR` until manually recovered via `sentinel retrain <model>` or `sentinel rollback <model>`.

## Deployment

One Sentinel per Prometheus instance. Service scoping is handled via labels in `sentinel.yaml` identity block — not by running multiple Sentinel instances.