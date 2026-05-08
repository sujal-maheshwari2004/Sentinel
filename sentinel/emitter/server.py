# sentinel/emitter/server.py

import threading
import time
from typing import Optional
from prometheus_client import Gauge, CollectorRegistry, start_http_server, REGISTRY
from sentinel.emitter.formatter import format_prediction, FormattedPrediction, build_metric_key
from sentinel.pipeline.predictor import Predictor
from sentinel.pipeline.drift import DriftMonitor
from sentinel.ingestor.scraper import PrometheusScraper
from sentinel.config import WatchConfig, SentinelConfig
from sentinel.utils.time import parse_duration_to_seconds
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

# how often the emitter tick runs in seconds
_EMITTER_TICK_SECONDS = 10


class MetricEmitter:
    """
    Manages the Prometheus Gauge objects for a single watched metric
    and updates them on each emitter tick.

    One Gauge per prediction step within the horizon. Each Gauge carries
    the step index as a label so Grafana can plot all horizon steps
    on the same panel or separately.

    Also handles feeding actual values back to the DriftMonitor
    so drift detection has ground truth to compare against.
    """

    def __init__(
        self,
        watch_config: WatchConfig,
        predictor: Predictor,
        drift_monitor: DriftMonitor,
        scraper: PrometheusScraper,
        emit_confidence_bounds: bool = False,
        registry: CollectorRegistry = REGISTRY,
    ):
        self.watch_config = watch_config
        self.predictor = predictor
        self.drift_monitor = drift_monitor
        self.scraper = scraper
        self.emit_confidence_bounds = emit_confidence_bounds
        self._registry = registry

        # gauge_key -> Gauge
        self._gauges: dict[str, Gauge] = {}
        self._gauge_lock = threading.Lock()

        # last set of predictions keyed by step for drift comparison
        # step -> (predicted_value, predicted_timestamp)
        self._pending_predictions: dict[int, tuple[float, float]] = {}
        self._pending_lock = threading.Lock()

    def tick(self) -> None:
        """
        Run one emitter cycle:
            1. Run inference via Predictor
            2. Format predictions
            3. Update Gauges
            4. Feed actual values to DriftMonitor for past predictions
        """
        self._feed_drift_actuals()

        result = self.predictor.predict()
        if result is None:
            logger.debug(f"[{self.watch_config.metric}] no prediction available this tick")
            return

        formatted_list = format_prediction(
            watch_config=self.watch_config,
            result=result,
            emit_confidence_bounds=self.emit_confidence_bounds,
        )

        for formatted in formatted_list:
            self._update_gauges(formatted)

        # store predictions for drift comparison on next ticks
        with self._pending_lock:
            for step_data in formatted_list[0].steps:  # use main prediction series
                step = step_data["step"]
                self._pending_predictions[step] = (
                    step_data["value"],
                    step_data["timestamp"],
                )

    def _update_gauges(self, formatted: FormattedPrediction) -> None:
        """
        For each step in the prediction, set the corresponding Gauge value.
        Creates the Gauge lazily on first encounter.
        """
        for step_data in formatted.steps:
            step = step_data["step"]
            value = step_data["value"]

            labels_with_step = {**formatted.labels, "sentinel_step": str(step)}
            gauge_key = build_metric_key(formatted.metric_name, labels_with_step)

            gauge = self._get_or_create_gauge(
                gauge_key=gauge_key,
                metric_name=formatted.metric_name,
                label_names=list(labels_with_step.keys()),
            )

            try:
                gauge.labels(**labels_with_step).set(value)
            except Exception as e:
                logger.error(f"Failed to set gauge '{gauge_key}': {e}")

    def _get_or_create_gauge(
        self,
        gauge_key: str,
        metric_name: str,
        label_names: list[str],
    ) -> Gauge:
        with self._gauge_lock:
            if gauge_key not in self._gauges:
                # sanitize metric name for Prometheus — only alphanumeric and underscores
                safe_name = metric_name.replace(".", "_").replace("-", "_")
                try:
                    g = Gauge(
                        name=safe_name,
                        documentation=f"Sentinel prediction for {self.watch_config.metric}",
                        labelnames=label_names,
                        registry=self._registry,
                    )
                    self._gauges[gauge_key] = g
                    logger.debug(f"Created Gauge '{safe_name}' with labels {label_names}")
                except ValueError:
                    # Gauge already registered (e.g. after hot reload), fetch existing
                    self._gauges[gauge_key] = self._registry._names_to_collectors.get(safe_name)
            return self._gauges[gauge_key]

    def _feed_drift_actuals(self) -> None:
        """
        For each pending prediction whose timestamp has now passed,
        fetch the actual value from Prometheus and record the residual
        in the DriftMonitor.
        """
        now = time.time()
        due_steps = []

        with self._pending_lock:
            for step, (predicted_val, predicted_ts) in list(self._pending_predictions.items()):
                if predicted_ts <= now:
                    due_steps.append((step, predicted_val, predicted_ts))

            for step, _, _ in due_steps:
                del self._pending_predictions[step]

        for step, predicted_val, predicted_ts in due_steps:
            actual = self.scraper.fetch_latest(
                metric=self.watch_config.metric,
                labels=self.watch_config.labels,
            )
            if actual is not None:
                _, actual_val = actual
                self.drift_monitor.record(predicted_val, actual_val)
                logger.debug(
                    f"[{self.watch_config.metric}] drift record step={step} "
                    f"predicted={predicted_val:.4f} actual={actual_val:.4f}"
                )


class EmitterServer:
    """
    Runs the /metrics HTTP server and orchestrates all MetricEmitters.
    One EmitterServer per Sentinel instance.

    On each tick it calls tick() on every MetricEmitter in a loop.
    The HTTP server is started once and serves all registered Gauges
    via prometheus_client's built-in exposition.
    """

    def __init__(self, config: SentinelConfig):
        self.config = config
        self._emitters: list[MetricEmitter] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def register(self, emitter: MetricEmitter) -> None:
        self._emitters.append(emitter)

    def start(self) -> None:
        """
        Start the /metrics HTTP server and the emitter loop thread.
        """
        start_http_server(self.config.emitter_port)
        logger.info(f"Sentinel /metrics server started on port {self.config.emitter_port}")

        self._thread = threading.Thread(
            target=self._loop,
            name="emitter-server",
            daemon=True,
        )
        self._thread.start()
        logger.info("Emitter loop started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Emitter server stopped")

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            for emitter in self._emitters:
                try:
                    emitter.tick()
                except Exception as e:
                    logger.error(
                        f"Emitter tick failed for {emitter.watch_config.metric}: {e}"
                    )
            time.sleep(_EMITTER_TICK_SECONDS)