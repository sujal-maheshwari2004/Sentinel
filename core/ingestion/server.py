import logging

from fastapi import FastAPI, Request, Response, status

from core.buffer.store import BufferStore
from core.ingestion.parser import parse_remote_write
from core.registry import ModelRegistry, ModelState

log = logging.getLogger(__name__)

app = FastAPI(title="Sentinel", version="0.1.0")

# Injected by main.py on startup
buffer: BufferStore = None
registry: ModelRegistry = None
metrics_store = None   # exposition.metrics.MetricsStore - imported at runtime to avoid circular


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