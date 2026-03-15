import time
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Show the status of all models.")
console = Console()

STATE_LABELS = {
    0: ("[yellow]WAITING[/yellow]",     "Collecting data"),
    1: ("[blue]TRAINING[/blue]",        "Training in progress"),
    2: ("[green]INFERENCING[/green]",   "Predictions active"),
    3: ("[cyan]RETRAINING[/cyan]",      "Retraining, old model still live"),
    4: ("[red]ERROR[/red]",             "Needs attention"),
}


@app.command()
def status():
    """Show all configured models and their current lifecycle state."""

    config_path = Path("sentinel.yaml")
    if not config_path.exists():
        console.print("[red]sentinel.yaml not found. Run 'sentinel init' first.[/red]")
        raise typer.Exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    models = config.get("models", [])

    if not models:
        console.print("[yellow]No models configured. Run 'sentinel add model <name>' to add one.[/yellow]")
        raise typer.Exit(0)

    table = Table(title="Sentinel Model Status", show_lines=True)
    table.add_column("Model",           style="bold")
    table.add_column("Type",            style="dim")
    table.add_column("State")
    table.add_column("Wait Strategy",   style="dim")
    table.add_column("Retrain Schedule", style="dim")

    for m in models:
        name = m.get("name", "unknown")
        model_type = "custom" if m.get("path") else "builtin"
        wait = m.get("wait", {})
        retrain = m.get("retrain", {})

        wait_str = f"{wait.get('strategy', 'both')} / {wait.get('time_hours', 6)}h / {wait.get('rows', 10000)} rows"
        schedule = retrain.get("schedule", "0 2 * * *")

        # State is only available if Sentinel is running and has a state file.
        # For now show UNKNOWN if we cannot reach the runtime.
        state_label, state_desc = _get_runtime_state(name)

        table.add_row(name, model_type, f"{state_label}\n[dim]{state_desc}[/dim]", wait_str, schedule)

    console.print(table)


def _get_runtime_state(model_name: str):
    """
    Try to read the model state from the Sentinel metrics endpoint.
    Falls back to 'UNKNOWN' if Sentinel is not running.
    """
    try:
        import urllib.request
        url = "http://localhost:9001/metrics"
        with urllib.request.urlopen(url, timeout=2) as resp:
            content = resp.read().decode()

        # Find the state gauge for this model
        for line in content.splitlines():
            if "predictivex_model_state" in line and f'model="{model_name}"' in line:
                value = int(float(line.split()[-1]))
                label, desc = STATE_LABELS.get(value, ("[dim]UNKNOWN[/dim]", ""))
                return label, desc

        return "[dim]UNKNOWN[/dim]", "Not found in metrics"

    except Exception:
        return "[dim]OFFLINE[/dim]", "Sentinel not running"