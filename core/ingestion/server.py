import logging
import threading
import uuid

from fastapi import FastAPI, Request, Response, status
from pydantic import BaseModel

from core.buffer.store import BufferStore
from core.ingestion.parser import parse_remote_write
from core.registry import ModelRegistry, ModelState

log = logging.getLogger(__name__)

app = FastAPI(title="Sentinel", version="0.1.0")

# Injected by main.py on startup
buffer: BufferStore = None
registry: ModelRegistry = None
metrics_store = None    # exposition.metrics.MetricsStore
config = None           # core.config.SentinelConfig


class RollbackRequest(BaseModel):
    run_id: str


@app.post("/ingest", status_code=status.HTTP_204_NO_CONTENT)
async def ingest(request: Request):
    """
    Receive a Prometheus remote_write payload and write the samples to the buffer.
    Also updates the row count on all WAITING models so threshold checks stay current.
    """
    body = await request.body()

    try:
        rows = parse_remote_write(body)
    except Exception as exc:
        log.error("Failed to parse remote_write payload: %s", exc)
        return Response(status_code=status.HTTP_400_BAD_REQUEST)

    buffer.append_many(rows)

    # Keep waiting models informed of how much data has accumulated
    for model in registry.get_by_state(ModelState.WAITING):
        model.rows_collected = buffer.total_rows()

    log.debug("Ingested %d rows", len(rows))


@app.get("/metrics")
async def metrics():
    """
    Serve current predictions and model lifecycle metrics in
    Prometheus text exposition format. Grafana scrapes this endpoint
    the same way it scrapes any Prometheus exporter.
    """
    output = metrics_store.render(models=registry.get_all())
    return Response(content=output, media_type="text/plain; version=0.0.4")


@app.get("/health")
async def health():
    """Liveness check. Returns 200 when the service is running."""
    return {"status": "ok"}


# -----------------------------------------------------------------------------
# Management endpoints — used by the CLI
# -----------------------------------------------------------------------------

@app.post("/manage/retrain/{model_name}")
async def trigger_retrain(model_name: str):
    """Manually trigger a training run for a model outside of its schedule."""
    from pipeline.training.trainer import run_training

    model = registry.get(model_name)
    if not model:
        return Response(status_code=status.HTTP_404_NOT_FOUND)

    run_id = str(uuid.uuid4())[:8]

    # Run in background so the HTTP response returns immediately
    thread = threading.Thread(target=run_training, args=[model, config], daemon=True)
    thread.start()

    return {"status": "triggered", "model": model_name, "run_id": run_id}


@app.post("/manage/rollback/{model_name}")
async def trigger_rollback(model_name: str, body: RollbackRequest):
    """Roll back a model to a previously trained MLflow artifact."""
    from pipeline.hotswap.swapper import swap_artifact
    from versioning.model import rollback_to_run

    model = registry.get(model_name)
    if not model:
        return Response(status_code=status.HTTP_404_NOT_FOUND)

    try:
        artifact_path = rollback_to_run(
            model_name=model_name,
            run_id=body.run_id,
            tracking_uri=config.mlflow_tracking_uri,
            artifacts_dir=config.artifacts.dir,
        )
        swap_artifact(model, artifact_path)
        return {"status": "rolled_back", "model": model_name, "artifact_path": artifact_path}

    except Exception as exc:
        log.error("Rollback failed for %s: %s", model_name, exc)
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)