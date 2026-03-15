from pathlib import Path

import typer
import yaml
from rich.console import Console

app = typer.Typer(help="Roll back a model to a previous artifact.")
console = Console()


@app.command()
def rollback(
    name: str = typer.Argument(..., help="Name of the model to roll back."),
    run_id: str = typer.Option(..., "--run-id", help="MLflow run ID to restore."),
):
    """
    Roll back a model to a previously trained artifact stored in MLflow.
    The restored artifact is immediately hot-swapped into the running model.
    """

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

    console.print(f"Rolling back [cyan]{name}[/cyan] to run [cyan]{run_id}[/cyan]...")

    try:
        import json
        import urllib.request

        url = f"http://localhost:9000/manage/rollback/{name}"
        payload = json.dumps({"run_id": run_id}).encode()
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")

        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            console.print(f"[green]Rollback complete for '{name}'[/green]")
            console.print(f"  Artifact: [dim]{result.get('artifact_path', 'unknown')}[/dim]")

    except Exception as exc:
        console.print(f"[red]Could not reach Sentinel at localhost:9000: {exc}[/red]")
        console.print("[dim]Is Sentinel running? Try: docker compose up[/dim]")
        raise typer.Exit(1)