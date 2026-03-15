# Getting Started

## Prerequisites

- Docker and Docker Compose
- Prometheus already running in your stack
- Grafana already running in your stack

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-org/sentinel
cd sentinel
uv sync
cp .env.example .env
```

## Configuration

Run `sentinel init` to scaffold `sentinel.yaml` and the required directories:

```bash
sentinel init
```

Open `sentinel.yaml` and set your `service_name` under the `identity` block:

```yaml
identity:
  service_name: api-gateway    # change this
  namespace: production
  cluster: us-east-1
```

## Add a model

```bash
sentinel add model latency-spikes
```

This adds the `latency-spikes` builtin to `sentinel.yaml`. Sentinel will start collecting data as soon as Prometheus begins sending metrics, and will train automatically once the wait threshold is met.

## Connect Prometheus

Add one block to your `prometheus.yml`:

```yaml
remote_write:
  - url: http://sentinel:9000/ingest
```

Restart Prometheus.

## Connect Grafana

Add one scrape target to your Grafana configuration:

```yaml
scrape_configs:
  - job_name: sentinel
    static_configs:
      - targets: ['sentinel:9001']
```

## Start Sentinel

```bash
docker compose up
```

Sentinel, MLflow, and the metrics endpoint start together.

## Check status

```bash
sentinel status
```

This shows all configured models and their current lifecycle state. Models will be in `WAITING` state while collecting data, then move to `TRAINING`, then `INFERENCING` once the first model artifact is ready.

## View predictions in Grafana

Add a new panel in Grafana using the `sentinel` data source. Query:

```
predictivex_failure_probability{model="latency-spikes"}
```

You can also import the pre-built dashboard from `docs/dashboards/latency-spikes.json`.