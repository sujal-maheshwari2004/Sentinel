from pathlib import Path

import typer
import yaml
from rich.console import Console

app = typer.Typer(help="Add a model to Sentinel.")
console = Console()

BUILTIN_NAMES = ["latency-spikes", "memory-saturation", "cpu-exhaustion"]


@app.command()
def model(
    name_or_path: str = typer.Argument(
        ...,
        help="Builtin model name (e.g. latency-spikes) or path to a custom model directory.",
    )
):
    """Add a builtin or custom model to sentinel.yaml."""

    config_path = Path("sentinel.yaml")
    if not config_path.exists():
        console.print("[red]sentinel.yaml not found. Run 'sentinel init' first.[/red]")
        raise typer.Exit(1)

    is_builtin = name_or_path in BUILTIN_NAMES
    is_custom_path = Path(name_or_path).is_dir()

    if not is_builtin and not is_custom_path:
        console.print(f"[red]'{name_or_path}' is not a known builtin and is not a directory.[/red]")
        console.print(f"Available builtins: {', '.join(BUILTIN_NAMES)}")
        raise typer.Exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    config.setdefault("models", [])

    # Check for duplicate
    existing_names = [m.get("name") for m in config["models"]]
    model_name = name_or_path if is_builtin else Path(name_or_path).name

    if model_name in existing_names:
        console.print(f"[yellow]Model '{model_name}' is already in sentinel.yaml — skipping[/yellow]")
        raise typer.Exit(0)

    # Build the model entry
    model_entry = {"name": model_name}

    if is_custom_path:
        model_entry["path"] = str(Path(name_or_path).resolve())

    # Apply default wait/retrain/inference blocks
    model_entry["wait"] = {"strategy": "both", "time_hours": 6, "rows": 10000}
    model_entry["retrain"] = {"schedule": "0 2 * * *", "min_rows": 10000}
    model_entry["inference"] = {"interval_seconds": 60, "timeout_seconds": 30, "fallback": "continue_old"}

    config["models"].append(model_entry)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    label = "builtin" if is_builtin else "custom"
    console.print(f"[green]Added {label} model '{model_name}' to sentinel.yaml[/green]")
    console.print(f"  Edit [cyan]sentinel.yaml[/cyan] to adjust wait, retrain, and inference settings.")