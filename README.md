# Sentinel

Predictive observability for Prometheus and Grafana.

Sentinel sits between Prometheus and Grafana. It receives your metrics via `remote_write`, runs ML models to predict service failures before they happen, and exposes those predictions back as standard Prometheus metrics for Grafana to scrape.

**Zero intrusion into your existing stack.** One config block in Prometheus, one scrape target in Grafana.

---

## How it works

```
Your Services  →  Prometheus  →  Sentinel  →  Grafana
                  (existing)    (new, no      (existing)
                                changes)
```

Sentinel receives all the metrics Prometheus already scrapes, trains failure prediction models on that data, and emits probability scores that Grafana treats like any other metric:

```
predictivex_failure_probability{service="api-gateway", model="latency-spikes", horizon="600s"} 0.83
predictivex_model_state{service="api-gateway", model="latency-spikes"} 2
```

Alert on them, build dashboards, route to PagerDuty — all through Grafana's existing tooling.

---

## Prerequisites

- Docker Desktop (or Docker + Docker Compose)
- Prometheus already running
- Grafana already connected to Prometheus

---

## Getting started

### 1. Clone Sentinel

```bash
git clone https://github.com/your-org/sentinel
cd sentinel
```

### 2. Configure

```bash
cp .env.example .env
```

Open `sentinel.yaml` and set your service name:

```yaml
identity:
  service_name: api-gateway    # change this to match your service
  namespace: production
  cluster: my-cluster
```

Add your first model:

```yaml
models:
  - name: latency-spikes
    wait:
      strategy: both
      time_hours: 6
      rows: 10000
    retrain:
      schedule: "0 2 * * *"
    inference:
      interval_seconds: 60
      timeout_seconds: 30
      fallback: continue_old
```

### 3. Start Sentinel

```bash
docker compose up -d
```

That's it. Sentinel, MLflow, and the metrics endpoint are now running.

### 4. Connect Prometheus

Add one block to your existing `prometheus.yml` and reload Prometheus:

```yaml
remote_write:
  - url: http://sentinel:9000/ingest
```

If Sentinel is running on a different host, replace `sentinel` with the hostname or IP.

### 5. Connect Grafana

Add one scrape target to your Grafana configuration:

```yaml
scrape_configs:
  - job_name: sentinel
    static_configs:
      - targets: ['sentinel:9001']
```

Or add it directly in the Grafana UI under **Configuration → Data Sources → Prometheus → Scrape configs**.

### 6. Import dashboards

In Grafana, go to **Dashboards → Import** and upload any of the pre-built dashboards from `docs/dashboards/`:

- `latency-spikes.json` — HTTP latency spike prediction
- `memory-saturation.json` — Memory exhaustion risk
- `cpu-exhaustion.json` — CPU throttling onset

---

## What happens next

Sentinel starts in `WAITING` state, quietly collecting your metrics data. Once enough data has accumulated (6 hours or 10,000 samples for the default models), training kicks off automatically overnight. After the first training run completes, predictions start appearing in Grafana.

Check the current state at any time:

```bash
# See all model states
curl http://localhost:9001/metrics | grep model_state

# Or use the CLI
sentinel status
```

Model states:
- `0` WAITING — collecting data, not yet trained
- `1` TRAINING — first training run in progress
- `2` INFERENCING — predictions active in Grafana
- `3` RETRAINING — updating model, old predictions still live
- `4` ERROR — needs attention, check logs

---

## Built-in models

| Model | Predicts | Minimum data needed |
|---|---|---|
| `latency-spikes` | HTTP latency degradation 10 min ahead | 6h or 10k samples |
| `memory-saturation` | OOM / memory exhaustion 15 min ahead | 12h or 20k samples |
| `cpu-exhaustion` | CPU throttling onset 10 min ahead | 6h or 10k samples |

All models train exclusively on **your own metrics data**. No pre-trained weights are included.

---

## CLI

Install the CLI by installing the package locally:

```bash
pip install -e .
```

Then:

```bash
sentinel init                                     # scaffold sentinel.yaml
sentinel add model latency-spikes                 # add a builtin model
sentinel add model ./my-model/                    # add a custom model
sentinel status                                   # show all models and their state
sentinel retrain latency-spikes                   # manually trigger retraining
sentinel rollback latency-spikes --run-id <id>    # restore a previous artifact
sentinel logs latency-spikes                      # tail logs for a model
```

---

## Bringing your own model

Drop a directory with three files into the repo and Sentinel picks it up:

```
my-model/
  manifest.yaml    # declare required metrics, wait thresholds, inference config
  retrain.py       # reads snapshot parquet files, trains, writes artifact
  infer.py         # loads artifact, writes JSON predictions to stdout
```

```bash
sentinel add model ./my-model/
```

See [docs/adding-a-model.md](docs/adding-a-model.md) for the full guide including the exact contract, environment variables, and examples.

---

## Services

| Service | URL | Purpose |
|---|---|---|
| Sentinel ingest | `http://localhost:9000/ingest` | Prometheus `remote_write` target |
| Sentinel metrics | `http://localhost:9001/metrics` | Grafana scrape target |
| MLflow UI | `http://localhost:5000` | Training run history and model registry |

---

## MLOps

Every training run is automatically tracked in MLflow — parameters, metrics, and the model artifact. Snapshots are versioned with DVC so any training run is reproducible. Set `WANDB_API_KEY` in `.env` to activate Weights & Biases tracking alongside MLflow.

---

## Troubleshooting

**No predictions in Grafana after several hours**
Check that Prometheus is sending data: `curl http://localhost:9001/metrics | grep rows_collected`. If it is zero, verify your `remote_write` config and that Prometheus can reach Sentinel on port 9000.

**Model stuck in WAITING**
The wait threshold hasn't been met yet. Check progress: `curl http://localhost:9001/metrics | grep warmup_progress`. A value of `0.43` means 43% of the row threshold has been collected.

**Model in ERROR state**
Check the logs: `docker compose logs sentinel | grep ERROR`. Then either fix the issue and retrain, or roll back to a previous working artifact: `sentinel rollback <model> --run-id <id>`.

---

## License

Apache 2.0