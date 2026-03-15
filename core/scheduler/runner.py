import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from core.buffer.store import BufferStore
from core.config import SentinelConfig
from core.registry import ModelInstance, ModelRegistry
from core.snapshot.manager import delete_old_snapshots, flush_buffer_to_snapshot
from pipeline.inference.inferencer import run_inference
from pipeline.training.trainer import check_and_train_waiting_models, run_training

log = logging.getLogger(__name__)


def start_scheduler(
    config: SentinelConfig,
    registry: ModelRegistry,
    buffer: BufferStore,
    metrics_store,          # exposition.metrics.MetricsStore — passed in to avoid circular import
) -> BackgroundScheduler:
    """
    Register all recurring jobs and start the background scheduler.
    Returns the running scheduler so main.py can shut it down cleanly on exit.
    """
    scheduler = BackgroundScheduler()

    # Flush buffer to disk on a fixed interval
    scheduler.add_job(
        func=flush_buffer_to_snapshot,
        trigger=IntervalTrigger(hours=config.snapshot.interval_hours),
        args=[buffer, config.snapshot.dir],
        id="snapshot_flush",
        name="Flush buffer to snapshot",
    )

    # Delete snapshots older than the retention window, checked daily
    scheduler.add_job(
        func=delete_old_snapshots,
        trigger=IntervalTrigger(hours=24),
        args=[config.snapshot.dir, config.snapshot.retention_days],
        id="snapshot_cleanup",
        name="Delete old snapshots",
    )

    # Evict old samples from the in-memory buffer every hour
    scheduler.add_job(
        func=buffer.evict_old_samples,
        trigger=IntervalTrigger(hours=1),
        id="buffer_evict",
        name="Evict old buffer samples",
    )

    # Check waiting models every 10 minutes to see if threshold is met
    scheduler.add_job(
        func=check_and_train_waiting_models,
        trigger=IntervalTrigger(minutes=10),
        args=[registry, config],
        id="wait_threshold_check",
        name="Check wait thresholds",
    )

    # Register per-model inference and retrain jobs
    for model in registry.get_all():
        _register_model_jobs(scheduler, model, config, metrics_store)

    scheduler.start()
    log.info("Scheduler started with %d jobs", len(scheduler.get_jobs()))
    return scheduler


def _register_model_jobs(
    scheduler: BackgroundScheduler,
    model: ModelInstance,
    config: SentinelConfig,
    metrics_store,
) -> None:
    """
    Register the inference interval job and retrain cron job for a single model.
    Both are keyed by model name so they can be identified in the scheduler logs.
    """
    model_name = model.config.name

    # Inference runs on a fixed interval (e.g. every 60 seconds)
    scheduler.add_job(
        func=run_inference,
        trigger=IntervalTrigger(seconds=model.config.inference.interval_seconds),
        args=[model, config, metrics_store],
        id=f"infer_{model_name}",
        name=f"Inference: {model_name}",
    )

    # Retrain runs on a user-defined cron schedule (e.g. nightly at 2am)
    scheduler.add_job(
        func=run_training,
        trigger=CronTrigger.from_crontab(model.config.retrain.schedule),
        args=[model, config],
        id=f"retrain_{model_name}",
        name=f"Retrain: {model_name}",
    )

    log.info(
        "Registered jobs for model %s — inference every %ds, retrain on '%s'",
        model_name,
        model.config.inference.interval_seconds,
        model.config.retrain.schedule,
    )