import typer

from cli.commands import add, init, logs, promote, remove, retrain, rollback, status

app = typer.Typer(
    name="sentinel",
    help="Sentinel — predictive observability for Prometheus and Grafana.",
    no_args_is_help=True,
)

app.add_typer(init.app,     name="init")
app.add_typer(add.app,      name="add")
app.add_typer(remove.app,   name="remove")
app.add_typer(status.app,   name="status")
app.add_typer(retrain.app,  name="retrain")
app.add_typer(rollback.app, name="rollback")
app.add_typer(promote.app,  name="promote")
app.add_typer(logs.app,     name="logs")

if __name__ == "__main__":
    app()