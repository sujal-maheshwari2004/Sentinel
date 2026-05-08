# sentinel/core.py

import threading
import time
from datetime import datetime, timezone, timedelta
from sentinel.config import SentinelConfig, WatchConfig
from sentinel.ingestor.scraper import PrometheusScraper
from sentinel.ingestor.buffer import BufferRegistry
from sentinel.pipeline.drift import DriftMonitor
from sentinel.pipeline.versioning import VersionStore
from sentinel.pipeline.registry import ModelRegistry
from sentinel.pipeline.trainer import Trainer
from sentinel.pipeline.predictor import Predictor
from sentinel.pipeline.scheduler import MetricScheduler
from sentinel.emitter.server import EmitterServer, MetricEmitter
from sentinel.utils.validation import validate_sentinel_config
from sentinel.utils.logging import get_logger, configure_logging
from sentinel.utils.time import parse_duration_to_seconds

logger = get_logger(__name__)

# how often the ingestor pulls fresh data from Prometheus
_INGESTOR_TICK_SECONDS = 10


class Sentinel:
    """
    Top-level orchestrator. Wires together all three parts:

        Data Ingestor  — scrapes Prometheus, fills buffers
        MLOps Pipeline — trains, predicts, monitors drift, schedules retraining
        Data Emitter   — serves /metrics with predictions for Grafana

    Usage:

        from sentinel import Sentinel
        from sentinel.config import SentinelConfig, WatchConfig
        from sentinel.pipeline.models import ExponentialSmoothingModel

        config = SentinelConfig(
            prometheus_url="http://localhost:9090",
            emitter_port=8080,
            watches=[
                WatchConfig(
                    metric="http_request_duration_seconds",
                    labels={"job": "api"},
                    model_class=ExponentialSmoothingModel,
                    granularity="1m",
                    horizon="5m",
                    lookback="30m",
                    cron="0 */6 * * *",
                )
            ]
        )

        sentinel = Sentinel(config)
        sentinel.start()
    """

    def __init__(self, config: SentinelConfig, log_level: str = "INFO"):
        configure_logging(log_level)
        validate_sentinel_config(config)

        self.config = config
        self._stop_event = threading.Event()

        # shared scraper — one HTTP client for all metrics
        self._scraper = PrometheusScraper(
            prometheus_url=config.prometheus_url,
            timeout=config.scrape_timeout,
        )

        # shared buffer registry
        self._buffer_registry = BufferRegistry()

        # per-metric components
        self._drift_monitors: dict[str, DriftMonitor] = {}
        self._registries: dict[str, ModelRegistry] = {}
        self._trainers: dict[str, Trainer] = {}
        self._predictors: dict[str, Predictor] = {}
        self._schedulers: dict[str, MetricScheduler] = {}

        # emitter server
        self._emitter_server = EmitterServer(config)

        # ingestor background thread
        self._ingestor_thread: threading.Thread = None

        self._setup()

    def _setup(self) -> None:
        """
        Instantiate and wire all per-metric components.
        """
        for watch in self.config.watches:
            key = self._metric_key(watch)

            # buffer
            buffer = self._buffer_registry.register(
                key=key,
                metric=watch.metric,
                lookback=watch.lookback,
                granularity=watch.granularity,
            )

            # drift monitor
            drift_monitor = DriftMonitor(
                metric=key,
                finetune_threshold=watch.drift_finetune_threshold,
                retrain_threshold=watch.drift_retrain_threshold,
            )
            self._drift_monitors[key] = drift_monitor

            # version store + model registry
            version_store = VersionStore(
                metric_key=key,
                artifact_store=self.config.artifact_store,
                max_versions=self.config.max_versions_per_metric,
            )
            registry = ModelRegistry(metric_key=key, version_store=version_store)
            self._registries[key] = registry

            # attempt to restore model from disk so we don't cold start
            # on every restart if a trained model already exists
            def make_factory(w):
                def factory():
                    return w.model_class(
                        granularity=w.granularity,
                        horizon=w.horizon,
                        lookback=w.lookback,
                    )
                return factory

            registry.restore_from_disk(make_factory(watch))

            # trainer
            trainer = Trainer(
                watch_config=watch,
                buffer=buffer,
                registry=registry,
            )
            self._trainers[key] = trainer

            # predictor
            predictor = Predictor(
                watch_config=watch,
                buffer=buffer,
                registry=registry,
            )
            self._predictors[key] = predictor

            # scheduler
            def make_train_fn(t):
                def train_fn(severity, drift_score):
                    t.run(drift_severity=severity, drift_score=drift_score)
                return train_fn

            scheduler = MetricScheduler(
                watch_config=watch,
                buffer=buffer,
                drift_monitor=drift_monitor,
                train_fn=make_train_fn(trainer),
            )
            self._schedulers[key] = scheduler

            # emitter
            emitter = MetricEmitter(
                watch_config=watch,
                predictor=predictor,
                drift_monitor=drift_monitor,
                scraper=self._scraper,
                emit_confidence_bounds=self.config.emit_confidence_bounds,
            )
            self._emitter_server.register(emitter)

            logger.info(f"Sentinel wired up metric: {key}")

    def start(self) -> None:
        """
        Start all components. Blocks until stop() is called.
        """
        if not self._scraper.check_connectivity():
            raise RuntimeError(
                f"Cannot reach Prometheus at {self.config.prometheus_url}. "
                f"Check the URL and ensure Prometheus is running."
            )

        logger.info("Sentinel starting...")

        # start schedulers
        for scheduler in self._schedulers.values():
            scheduler.start()

        # start ingestor loop
        self._ingestor_thread = threading.Thread(
            target=self._ingestor_loop,
            name="ingestor",
            daemon=True,
        )
        self._ingestor_thread.start()

        # start emitter server
        self._emitter_server.start()

        logger.info(
            f"Sentinel running. "
            f"Watching {len(self.config.watches)} metric(s). "
            f"Emitting on port {self.config.emitter_port}."
        )

        # block main thread
        try:
            while not self._stop_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, shutting down...")
            self.stop()

    def stop(self) -> None:
        """
        Gracefully stop all components.
        """
        logger.info("Sentinel stopping...")
        self._stop_event.set()

        for scheduler in self._schedulers.values():
            scheduler.stop()

        self._emitter_server.stop()
        self._scraper.close()

        logger.info("Sentinel stopped.")

    def rollback(self, metric: str, labels: dict[str, str] = None) -> bool:
        """
        Manually roll back the model for a given metric to the previous version.
        Returns True if rollback succeeded.
        """
        key = self._metric_key_from_parts(metric, labels or {})
        registry = self._registries.get(key)
        if registry is None:
            logger.error(f"No registry found for metric key '{key}'")
            return False
        return registry.rollback()

    def status(self) -> dict:
        """
        Returns a snapshot of current state for all watched metrics.
        Useful for health checks and debugging.
        """
        result = {}
        for watch in self.config.watches:
            key = self._metric_key(watch)
            registry = self._registries.get(key)
            drift = self._drift_monitors.get(key)
            buffer = self._buffer_registry.get(key)
            active_version = registry.active_version() if registry else None

            result[key] = {
                "ready": registry.is_ready() if registry else False,
                "active_version": active_version.version_id if active_version else None,
                "active_version_mae": active_version.mae if active_version else None,
                "drift_mae": drift.current_mae() if drift else None,
                "buffer_fill": buffer.fill_fraction() if buffer else None,
            }
        return result

    def _ingestor_loop(self) -> None:
        """
        Continuously pulls fresh data from Prometheus for all watched metrics
        and pushes it into the corresponding buffers.
        """
        while not self._stop_event.is_set():
            for watch in self.config.watches:
                try:
                    self._ingestor_tick(watch)
                except Exception as e:
                    logger.error(f"Ingestor tick failed for {watch.metric}: {e}")
            time.sleep(_INGESTOR_TICK_SECONDS)

    def _ingestor_tick(self, watch: WatchConfig) -> None:
        key = self._metric_key(watch)
        buffer = self._buffer_registry.get(key)

        if buffer is None:
            return

        now = datetime.now(timezone.utc)
        granularity_secs = parse_duration_to_seconds(watch.granularity)

        # pull just the last two granularity steps to stay current
        start = now - timedelta(seconds=granularity_secs * 2)

        samples = self._scraper.fetch_range(
            metric=watch.metric,
            labels=watch.labels,
            start=start,
            end=now,
            granularity=watch.granularity,
        )

        if samples:
            buffer.push_many(samples)
            logger.debug(f"[{key}] ingestor pushed {len(samples)} new samples")

    def _initial_backfill(self, watch: WatchConfig) -> None:
        """
        On startup, backfill the buffer with historical data
        covering the full lookback window so cold start training
        can begin as soon as possible.
        """
        key = self._metric_key(watch)
        buffer = self._buffer_registry.get(key)

        if buffer is None or buffer.is_ready():
            return

        now = datetime.now(timezone.utc)
        lookback_secs = parse_duration_to_seconds(watch.lookback)
        start = now - timedelta(seconds=lookback_secs)

        logger.info(f"[{key}] backfilling {watch.lookback} of historical data...")

        samples = self._scraper.fetch_range(
            metric=watch.metric,
            labels=watch.labels,
            start=start,
            end=now,
            granularity=watch.granularity,
        )

        if samples:
            buffer.push_many(samples)
            logger.info(f"[{key}] backfill complete — {len(samples)} samples loaded")
        else:
            logger.warning(f"[{key}] backfill returned no data from Prometheus")

    @staticmethod
    def _metric_key(watch: WatchConfig) -> str:
        if not watch.labels:
            return watch.metric
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(watch.labels.items()))
        return f"{watch.metric}{{{label_str}}}"

    @staticmethod
    def _metric_key_from_parts(metric: str, labels: dict[str, str]) -> str:
        if not labels:
            return metric
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{metric}{{{label_str}}}"