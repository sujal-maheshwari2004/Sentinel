import logging
import signal
import sys
import threading

import uvicorn
from fastapi import FastAPI, Response

from core.buffer.store import BufferStore
from core.config import load_config
from core.ingestion import server as ingest_server
from core.registry import ModelRegistry
from core.scheduler.runner import start_scheduler
from exposition.metrics import MetricsStore


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_metrics_app(metrics_store: MetricsStore, registry: ModelRegistry) -> FastAPI:
    """
    Minimal FastAPI app that serves only the /metrics endpoint on port 9001.
    Grafana scrapes this. Kept separate from the ingest app so the two
    ports have clear, single responsibilities.
    """
    app = FastAPI(title="Sentinel Metrics")

    @app.get("/metrics")
    async def metrics():
        output = metrics_store.render(models=registry.get_all())
        return Response(content=output, media_type="text/plain; version=0.0.4")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def main() -> None:
    config = load_config("sentinel.yaml")
    setup_logging(config.log_level)

    log = logging.getLogger("sentinel")
    log.info("Starting Sentinel")

    # Build the shared components
    buffer = BufferStore()
    registry = ModelRegistry()
    metrics_store = MetricsStore(config.identity, config.exposition)

    # Register every model declared in sentinel.yaml
    for model_config in config.models:
        registry.register(model_config)
        log.info("Registered model: %s", model_config.name)

    # Inject shared state into the ingest FastAPI app
    ingest_server.buffer = buffer
    ingest_server.registry = registry
    ingest_server.metrics_store = metrics_store
    ingest_server.config = config

    # Build the separate metrics app for Grafana scraping
    metrics_app = build_metrics_app(metrics_store, registry)

    # Start the background scheduler
    scheduler = start_scheduler(config, registry, buffer, metrics_store)

    # Graceful shutdown
    def shutdown(sig, frame):
        log.info("Shutting down Sentinel")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    log.info("Ingest endpoint  : POST http://0.0.0.0:%d/ingest", config.ingest_port)
    log.info("Metrics endpoint : GET  http://0.0.0.0:%d/metrics", config.metrics_port)

    # Run the metrics server on port 9001 in a background thread
    metrics_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={
            "app": metrics_app,
            "host": "0.0.0.0",
            "port": config.metrics_port,
            "log_level": config.log_level.lower(),
        },
        daemon=True,
    )
    metrics_thread.start()

    # Run the ingest server on port 9000 in the main thread
    uvicorn.run(
        ingest_server.app,
        host="0.0.0.0",
        port=config.ingest_port,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    main()