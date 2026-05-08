# sentinel/config.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentinel.pipeline.models.base import BaseModel


@dataclass
class WatchConfig:
    metric: str
    labels: dict[str, str] = field(default_factory=dict)
    model_class: type[BaseModel] | None = None
    cron: str = "0 */6 * * *"
    granularity: str = "1m"
    horizon: str = "5m"
    lookback: str = "30m"
    drift_finetune_threshold: float = 0.1
    drift_retrain_threshold: float = 0.3


@dataclass
class SentinelConfig:
    prometheus_url: str = "http://localhost:9090"
    emitter_port: int = 8080
    artifact_store: str = "./sentinel_artifacts"
    max_versions_per_metric: int = 5
    watches: list[WatchConfig] = field(default_factory=list)
    scrape_timeout: int = 10
    emit_confidence_bounds: bool = False