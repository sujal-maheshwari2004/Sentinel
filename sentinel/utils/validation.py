# sentinel/utils/validation.py

from sentinel.utils.time import parse_duration_to_seconds, parse_duration_to_steps
from sentinel.utils.cron import is_valid_cron


class ConfigValidationError(Exception):
    pass


def validate_duration(value: str, field_name: str) -> None:
    """
    Raises ConfigValidationError if value is not a valid duration string.
    """
    try:
        parse_duration_to_seconds(value)
    except ValueError as e:
        raise ConfigValidationError(f"Invalid duration for '{field_name}': {e}")


def validate_cron(value: str, field_name: str) -> None:
    """
    Raises ConfigValidationError if value is not a valid cron string.
    """
    if not is_valid_cron(value):
        raise ConfigValidationError(
            f"Invalid cron string for '{field_name}': '{value}'. "
            f"Expected standard 5-field cron e.g. '0 */6 * * *'."
        )


def validate_horizon_divisible_by_granularity(horizon: str, granularity: str) -> None:
    """
    Raises ConfigValidationError if horizon is not evenly divisible by granularity.
    """
    try:
        parse_duration_to_steps(horizon, granularity)
    except ValueError as e:
        raise ConfigValidationError(str(e))


def validate_lookback_greater_than_horizon(lookback: str, horizon: str) -> None:
    """
    Raises ConfigValidationError if lookback <= horizon.
    A lookback window must be larger than the prediction horizon.
    """
    lookback_secs = parse_duration_to_seconds(lookback)
    horizon_secs = parse_duration_to_seconds(horizon)
    if lookback_secs <= horizon_secs:
        raise ConfigValidationError(
            f"lookback '{lookback}' must be greater than horizon '{horizon}'."
        )


def validate_thresholds(finetune: float, retrain: float) -> None:
    """
    Raises ConfigValidationError if drift thresholds are misconfigured.
    finetune threshold must be less than retrain threshold.
    Both must be positive.
    """
    if finetune <= 0 or retrain <= 0:
        raise ConfigValidationError(
            "Drift thresholds must be positive floats."
        )
    if finetune >= retrain:
        raise ConfigValidationError(
            f"drift_finetune_threshold ({finetune}) must be less than "
            f"drift_retrain_threshold ({retrain})."
        )


def validate_watch_config(watch) -> None:
    """
    Full validation pass on a WatchConfig instance.
    Raises ConfigValidationError on first failure found.
    """
    if not watch.metric or not isinstance(watch.metric, str):
        raise ConfigValidationError("WatchConfig.metric must be a non-empty string.")

    if watch.model_class is None:
        raise ConfigValidationError(
            f"WatchConfig for '{watch.metric}' has no model_class set."
        )

    validate_cron(watch.cron, "cron")
    validate_duration(watch.granularity, "granularity")
    validate_duration(watch.horizon, "horizon")
    validate_duration(watch.lookback, "lookback")
    validate_horizon_divisible_by_granularity(watch.horizon, watch.granularity)
    validate_lookback_greater_than_horizon(watch.lookback, watch.horizon)
    validate_thresholds(watch.drift_finetune_threshold, watch.drift_retrain_threshold)


def validate_sentinel_config(config) -> None:
    """
    Full validation pass on a SentinelConfig instance.
    """
    if not config.prometheus_url:
        raise ConfigValidationError("SentinelConfig.prometheus_url must not be empty.")

    if not (1 <= config.emitter_port <= 65535):
        raise ConfigValidationError(
            f"emitter_port {config.emitter_port} is not a valid port number."
        )

    if config.max_versions_per_metric < 1:
        raise ConfigValidationError(
            "max_versions_per_metric must be at least 1."
        )

    if not config.watches:
        raise ConfigValidationError(
            "SentinelConfig.watches is empty — nothing to watch."
        )

    for watch in config.watches:
        validate_watch_config(watch)