from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="Initialise a new Sentinel project in the current directory.")
console = Console()

DEFAULT_CONFIG = """\
sentinel:
  ingest_port: 9000
  metrics_port: 9001
  log_level: info

identity:
  service_name: my-service    # change this to your service name
  namespace: default
  cluster: local
  team: ""
  extra_labels: {}

snapshot:
  dir: ./snapshots
  retention_days: 30
  interval_hours: 6

artifacts:
  dir: ./artifacts

exposition:
  max_series_total: 10000
  cardinality_warning_threshold: 5000
  drop_high_cardinality_labels:
    - instance
    - pod

drift:
  enabled: true
  method: psi
  warning_threshold: 0.2
  retrain_threshold: 0.4

mlflow:
  tracking_uri: http://mlflow:5000

dvc:
  remote: local
  remote_path: ./dvc-storage

models: []
"""


@app.command()
def init():
    """Scaffold sentinel.yaml and required runtime directories."""

    config_path = Path("sentinel.yaml")

    if config_path.exists():
        console.print("[yellow]sentinel.yaml already exists — skipping[/yellow]")
    else:
        config_path.write_text(DEFAULT_CONFIG)
        console.print("[green]Created sentinel.yaml[/green]")

    runtime_dirs = ["snapshots", "artifacts", "mlruns", "dvc-storage"]
    for directory in runtime_dirs:
        Path(directory).mkdir(exist_ok=True)
        console.print(f"[green]Created {directory}/[/green]")

    console.print("")
    console.print("[bold]Next steps:[/bold]")
    console.print("  1. Edit [cyan]sentinel.yaml[/cyan] and set your service_name")
    console.print("  2. Add a model:  [cyan]sentinel add model latency-spikes[/cyan]")
    console.print("  3. Start:        [cyan]docker compose up[/cyan]")