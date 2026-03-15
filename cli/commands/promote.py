from pathlib import Path

import typer
import yaml
from rich.console import Console

app = typer.Typer(help="Promote a model run to Production in MLflow.")
console = Console()


@app.command()
def promote(
    name: str = typer.Argument(..., help="Name of the model to promote."),
    run_id: str = typer.Option(..., "--run-id", help="MLflow run ID to promote to Production."),
):
    """
    Promote a specific MLflow run to the Production stage in the model registry.
    Any previously Production version is automatically archived.
    """

    config_path = Path("sentinel.yaml")
    if not config_path.exists():
        console.print("[red]sentinel.yaml not found.[/red]")
        raise typer.Exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    tracking_uri = config.get("mlflow", {}).get("tracking_uri", "http://mlflow:5000")
    model_names = [m.get("name") for m in config.get("models", [])]

    if name not in model_names:
        console.print(f"[red]Model '{name}' not found in sentinel.yaml[/red]")
        raise typer.Exit(1)

    console.print(f"Promoting [cyan]{name}[/cyan] run [cyan]{run_id}[/cyan] to Production...")

    try:
        from versioning.model import promote_to_production
        promote_to_production(
            model_name=name,
            run_id=run_id,
            tracking_uri=tracking_uri,
        )
        console.print(f"[green]'{name}' run {run_id} is now Production in MLflow[/green]")
        console.print(f"  View at: [cyan]{tracking_uri}[/cyan]")

    except Exception as exc:
        console.print(f"[red]Promotion failed: {exc}[/red]")
        raise typer.Exit(1)