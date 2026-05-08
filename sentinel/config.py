# sentinel/config.py

from dataclasses import dataclass, field
from typing import Optional, Type
from sentinel.pipeline.models.base import BaseModel


@dataclass
class WatchConfig:
    """
    Configuration for a single watched metric.
    One WatchConfig = one model = one prediction series emitted.
    """

    metric: str
    # e.g. {"job": "api", "instance": "localhost:8080"}
    labels: dict[str, str] = field(default_factory=dict)

    # model class to use, must be a subclass of BaseModel
    model_class: Type[BaseModel] = None

    # cron string defining cold start and retraining schedule
    # e.g. "0 */6 * * *" = every 6 hours
    cron: str = "0 */6 * * *"

    # resolution of data points to pull and train on
    # e.g. "1m", "5m", "1h"
    granularity: str = "1m"

    # how far ahead to predict
    # e.g. "5m", "15m", "1h"
    horizon: str = "5m"

    # how much historical data to use as features
    # e.g. "30m", "1h", "6h"
    lookback: str = "30m"

    # MAE threshold above which drift is considered low severity → finetune
    drift_finetune_threshold: float = 0.1

    # MAE threshold above which drift is considered high severity → full retrain
    drift_retrain_threshold: float = 0.3


@dataclass
class SentinelConfig:
    """
    Top level configuration for the Sentinel instance.
    """

    # Prometheus base URL to pull metrics from
    prometheus_url: str = "http://localhost:9090"

    # port Sentinel exposes its /metrics endpoint on
    emitter_port: int = 8080

    # where serialized model artifacts are stored
    # can be a local path or later extended to S3/GCS
    artifact_store: str = "./sentinel_artifacts"

    # how many versions to retain per metric before pruning old ones
    max_versions_per_metric: int = 5

    # list of metrics to watch
    watches: list[WatchConfig] = field(default_factory=list)

    # optional global scrape timeout in seconds
    scrape_timeout: int = 10

    # whether to emit confidence bounds alongside predictions
    emit_confidence_bounds: bool = False