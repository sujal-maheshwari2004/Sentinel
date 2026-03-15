from pathlib import Path

import typer
import yaml
from rich.console import Console

app = typer.Typer(help="Manually trigger retraining for a model.")
console = Console()


@app.command()
def retrain(
    name: str = typer.Argument(..., help="Name of the model to retrain."),
):
    """Manually trigger a training run for a model outside of its schedule."""

    config_path = Path("sentinel.yaml")
    if not config_path.exists():
        console.print("[red]sentinel.yaml not found.[/red]")
        raise typer.Exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_names = [m.get("name") for m in config.get("models", [])]
    if name not in model_names:
        console.print(f"[red]Model '{name}' not found in sentinel.yaml[/red]")
        raise typer.Exit(1)

    # Signal Sentinel via its management endpoint
    try:
        import json
        import urllib.request

        url = f"http://localhost:9000/manage/retrain/{name}"
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read())
            console.print(f"[green]Retrain triggered for '{name}'[/green]")
            console.print(f"  Run ID: [cyan]{result.get('run_id', 'unknown')}[/cyan]")

    except Exception as exc:
        console.print(f"[red]Could not reach Sentinel at localhost:9000: {exc}[/red]")
        console.print("[dim]Is Sentinel running? Try: docker compose up[/dim]")
        raise typer.Exit(1)