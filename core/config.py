import yaml
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class WaitConfig:
    strategy: str = "both"   # time | rows | both
    time_hours: int = 6
    rows: int = 10000


@dataclass
class RetrainConfig:
    schedule: str = "0 2 * * *"
    min_rows: int = 10000


@dataclass
class InferenceConfig:
    interval_seconds: int = 60
    timeout_seconds: int = 30
    fallback: str = "continue_old"   # continue_old | emit_zero | drop
    max_consecutive_errors: int = 5


@dataclass
class ModelConfig:
    name: str = ""
    path: str = ""              # empty means it is a builtin model
    wait: WaitConfig = field(default_factory=WaitConfig)
    retrain: RetrainConfig = field(default_factory=RetrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


@dataclass
class IdentityConfig:
    service_name: str = "sentinel"
    namespace: str = "default"
    cluster: str = "local"
    team: str = ""
    extra_labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class SnapshotConfig:
    dir: str = "./snapshots"
    retention_days: int = 30
    interval_hours: int = 6


@dataclass
class ArtifactsConfig:
    dir: str = "./artifacts"


@dataclass
class ExpositionConfig:
    max_series_total: int = 10000
    cardinality_warning_threshold: int = 5000
    drop_high_cardinality_labels: List[str] = field(default_factory=list)


@dataclass
class DriftConfig:
    enabled: bool = True
    method: str = "psi"
    warning_threshold: float = 0.2
    retrain_threshold: float = 0.4


@dataclass
class SentinelConfig:
    ingest_port: int = 9000
    metrics_port: int = 9001
    log_level: str = "info"
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    snapshot: SnapshotConfig = field(default_factory=SnapshotConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)
    exposition: ExpositionConfig = field(default_factory=ExpositionConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    mlflow_tracking_uri: str = "http://mlflow:5000"
    models: List[ModelConfig] = field(default_factory=list)


def load_config(path: str = "sentinel.yaml") -> SentinelConfig:
    """
    Load sentinel.yaml and return a fully populated SentinelConfig.
    Missing keys fall back to the dataclass defaults so partial
    configs are always valid.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = SentinelConfig()

    sentinel_block = raw.get("sentinel", {})
    cfg.ingest_port = sentinel_block.get("ingest_port", cfg.ingest_port)
    cfg.metrics_port = sentinel_block.get("metrics_port", cfg.metrics_port)
    cfg.log_level = sentinel_block.get("log_level", cfg.log_level)

    identity_block = raw.get("identity", {})
    cfg.identity = IdentityConfig(
        service_name=identity_block.get("service_name", "sentinel"),
        namespace=identity_block.get("namespace", "default"),
        cluster=identity_block.get("cluster", "local"),
        team=identity_block.get("team", ""),
        extra_labels=identity_block.get("extra_labels", {}),
    )

    snapshot_block = raw.get("snapshot", {})
    cfg.snapshot = SnapshotConfig(
        dir=snapshot_block.get("dir", "./snapshots"),
        retention_days=snapshot_block.get("retention_days", 30),
        interval_hours=snapshot_block.get("interval_hours", 6),
    )

    artifacts_block = raw.get("artifacts", {})
    cfg.artifacts = ArtifactsConfig(
        dir=artifacts_block.get("dir", "./artifacts"),
    )

    exposition_block = raw.get("exposition", {})
    cfg.exposition = ExpositionConfig(
        max_series_total=exposition_block.get("max_series_total", 10000),
        cardinality_warning_threshold=exposition_block.get("cardinality_warning_threshold", 5000),
        drop_high_cardinality_labels=exposition_block.get("drop_high_cardinality_labels", []),
    )

    drift_block = raw.get("drift", {})
    cfg.drift = DriftConfig(
        enabled=drift_block.get("enabled", True),
        method=drift_block.get("method", "psi"),
        warning_threshold=drift_block.get("warning_threshold", 0.2),
        retrain_threshold=drift_block.get("retrain_threshold", 0.4),
    )

    cfg.mlflow_tracking_uri = raw.get("mlflow", {}).get("tracking_uri", cfg.mlflow_tracking_uri)

    for model_block in raw.get("models", []):
        wait_block = model_block.get("wait", {})
        retrain_block = model_block.get("retrain", {})
        inference_block = model_block.get("inference", {})

        cfg.models.append(ModelConfig(
            name=model_block.get("name", ""),
            path=model_block.get("path", ""),
            wait=WaitConfig(
                strategy=wait_block.get("strategy", "both"),
                time_hours=wait_block.get("time_hours", 6),
                rows=wait_block.get("rows", 10000),
            ),
            retrain=RetrainConfig(
                schedule=retrain_block.get("schedule", "0 2 * * *"),
                min_rows=retrain_block.get("min_rows", 10000),
            ),
            inference=InferenceConfig(
                interval_seconds=inference_block.get("interval_seconds", 60),
                timeout_seconds=inference_block.get("timeout_seconds", 30),
                fallback=inference_block.get("fallback", "continue_old"),
                max_consecutive_errors=inference_block.get("max_consecutive_errors", 5),
            ),
        ))

    return cfg