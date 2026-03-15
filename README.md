# Sentinel

Predictive observability for Prometheus and Grafana.

Sentinel sits between Prometheus and Grafana. It receives your metrics via `remote_write`, runs ML models to predict service failures before they happen, and exposes those predictions back as standard Prometheus metrics for Grafana to scrape.

Zero intrusion into your existing stack. Two config lines in Prometheus, one in Grafana.

---

## How it works

```
Prometheus  →  remote_write  →  Sentinel  →  /metrics  →  Grafana
```

Sentinel receives all the metrics Prometheus already scrapes, trains failure prediction models on that data, and emits probability scores like:

```
predictivex_failure_probability{service="api",model="latency-spikes",horizon="10m"} 0.83
predictivex_model_state{service="api",model="latency-spikes"} 2
```

Grafana treats these exactly like any other metric — alert on them, build dashboards, route to PagerDuty.

---

## Quick start

**1. Add to your `prometheus.yml`:**
```yaml
remote_write:
  - url: http://sentinel:9000/ingest
```

**2. Add to your Grafana scrape config:**
```yaml
scrape_configs:
  - job_name: sentinel
    static_configs:
      - targets: ['sentinel:9001']
```

**3. Configure and start Sentinel:**
```bash
cp .env.example .env
# edit sentinel.yaml — set your service_name

docker compose up
```

**4. Add a model:**
```bash
sentinel add model latency-spikes
```

Sentinel will start collecting data and train automatically once enough data has accumulated.

---

## Built-in models

| Model | Predicts | Wait before training |
|---|---|---|
| `latency-spikes` | HTTP latency degradation | 6h or 10k rows |
| `memory-saturation` | OOM / memory exhaustion | 12h or 20k rows |
| `cpu-exhaustion` | CPU throttling onset | 6h or 10k rows |

All models train on your own metrics data — no pre-trained weights ship with Sentinel.

---

## CLI

```bash
sentinel init                              # set up sentinel.yaml and directories
sentinel add model latency-spikes          # add a builtin model
sentinel add model ./my-model/             # add a custom model
sentinel status                            # show all models and their state
sentinel retrain latency-spikes            # manually trigger retraining
sentinel rollback latency-spikes --run-id <id>   # restore a previous artifact
sentinel promote latency-spikes --run-id <id>    # promote a run to Production in MLflow
sentinel logs latency-spikes               # tail logs for a model
```

---

## Services

Once running via `docker compose up`:

| Service | URL |
|---|---|
| Sentinel ingest | `http://localhost:9000/ingest` |
| Sentinel metrics | `http://localhost:9001/metrics` |
| MLflow UI | `http://localhost:5000` |

---

## Bringing your own model

Drop in a directory with three files:

```
my-model/
  manifest.yaml    # declare required metrics, wait thresholds, inference config
  retrain.py       # reads snapshot parquet, trains, writes artifact to MODEL_OUTPUT_PATH
  infer.py         # loads artifact, reads snapshot, writes JSON predictions to stdout
```

Then:
```bash
sentinel add model ./my-model/
```

See [docs/adding-a-model.md](docs/adding-a-model.md) for the full guide.

---

## MLOps

Every training run is automatically tracked in MLflow. Snapshots are versioned with DVC. If `WANDB_API_KEY` is set in `.env`, W&B tracking activates automatically alongside MLflow.

---

## License

Apache 2.0