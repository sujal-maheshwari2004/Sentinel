import logging
import signal
import sys

import uvicorn

from core.buffer.store import BufferStore
from core.config import load_config
from core.ingestion import server
from core.registry import ModelRegistry
from core.scheduler.runner import start_scheduler
from exposition.metrics import MetricsStore


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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

    # Inject shared state into the FastAPI app
    server.buffer = buffer
    server.registry = registry
    server.metrics_store = metrics_store

    # Start background scheduler (snapshot flush, inference, retrain)
    scheduler = start_scheduler(config, registry, buffer, metrics_store)

    # Graceful shutdown on SIGINT / SIGTERM
    def shutdown(sig, frame):
        log.info("Shutting down Sentinel")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    log.info("Ingest endpoint  : POST http://0.0.0.0:%d/ingest", config.ingest_port)
    log.info("Metrics endpoint : GET  http://0.0.0.0:%d/metrics", config.metrics_port)

    uvicorn.run(
        server.app,
        host="0.0.0.0",
        port=config.ingest_port,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    main()