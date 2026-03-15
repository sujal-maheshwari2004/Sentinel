from pathlib import Path

import typer
import yaml
from rich.console import Console

app = typer.Typer(help="Tail logs for a specific model.")
console = Console()


@app.command()
def logs(
    name: str = typer.Argument(..., help="Name of the model to tail logs for."),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of recent log lines to show."),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output."),
):
    """
    Tail the Sentinel log output filtered to a specific model.
    Reads from the Docker Compose log stream by default.
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

    console.print(f"Showing logs for model [cyan]{name}[/cyan] (last {lines} lines):")
    console.print("[dim]Tip: run with --follow / -f to stream live[/dim]\n")

    try:
        import subprocess

        cmd = ["docker", "compose", "logs", "sentinel", f"--tail={lines}"]
        if follow:
            cmd.append("--follow")

        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout + result.stderr

        # Filter lines relevant to this model
        filtered = [
            line for line in output.splitlines()
            if name in line or "sentinel" in line.lower()
        ]

        if not filtered:
            console.print(f"[yellow]No log lines found for model '{name}'[/yellow]")
        else:
            for line in filtered:
                console.print(line)

    except FileNotFoundError:
        console.print("[red]docker not found on PATH.[/red]")
        console.print("[dim]If running without Docker, check your process logs directly.[/dim]")
        raise typer.Exit(1)