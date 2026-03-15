from pathlib import Path

import typer
import yaml
from rich.console import Console

app = typer.Typer(help="Remove a model from Sentinel.")
console = Console()


@app.command()
def model(
    name: str = typer.Argument(..., help="Name of the model to remove."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
):
    """Remove a model from sentinel.yaml."""

    config_path = Path("sentinel.yaml")
    if not config_path.exists():
        console.print("[red]sentinel.yaml not found.[/red]")
        raise typer.Exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    models = config.get("models", [])
    existing = [m for m in models if m.get("name") == name]

    if not existing:
        console.print(f"[yellow]Model '{name}' not found in sentinel.yaml[/yellow]")
        raise typer.Exit(1)

    if not yes:
        confirmed = typer.confirm(f"Remove model '{name}' from sentinel.yaml?")
        if not confirmed:
            console.print("Aborted.")
            raise typer.Exit(0)

    config["models"] = [m for m in models if m.get("name") != name]

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]Removed model '{name}' from sentinel.yaml[/green]")