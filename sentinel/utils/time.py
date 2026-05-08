# sentinel/utils/time.py

import re
from datetime import timedelta


# maps unit suffix to seconds
_UNIT_TO_SECONDS: dict[str, int] = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
    "w": 604800,
}


def parse_duration_to_seconds(duration: str) -> int:
    """
    Parse a human-readable duration string into total seconds.

    Examples:
        "30s" -> 30
        "5m"  -> 300
        "1h"  -> 3600
        "2d"  -> 172800
    """
    duration = duration.strip().lower()
    match = re.fullmatch(r"(\d+)([smhdw])", duration)
    if not match:
        raise ValueError(
            f"Invalid duration format: '{duration}'. "
            f"Expected a number followed by s, m, h, d, or w. e.g. '5m', '1h', '30s'."
        )
    value = int(match.group(1))
    unit = match.group(2)
    return value * _UNIT_TO_SECONDS[unit]


def parse_duration_to_steps(duration: str, granularity: str) -> int:
    """
    Compute how many granularity-sized steps fit in duration.

    Examples:
        parse_duration_to_steps("30m", "1m")  -> 30
        parse_duration_to_steps("1h",  "5m")  -> 12
        parse_duration_to_steps("1h",  "30s") -> 120

    Raises ValueError if duration is not evenly divisible by granularity.
    """
    duration_secs = parse_duration_to_seconds(duration)
    granularity_secs = parse_duration_to_seconds(granularity)

    if granularity_secs == 0:
        raise ValueError("Granularity must be greater than zero.")

    if duration_secs % granularity_secs != 0:
        raise ValueError(
            f"Duration '{duration}' ({duration_secs}s) is not evenly divisible "
            f"by granularity '{granularity}' ({granularity_secs}s)."
        )

    return duration_secs // granularity_secs


def steps_to_timedeltas(
    n_steps: int,
    granularity: str,
    start_offset_seconds: int = 0,
) -> list[timedelta]:
    """
    Generate a list of timedeltas representing future prediction timestamps.

    start_offset_seconds: offset from now to start at, default 0 (immediate next step).

    Example:
        steps_to_timedeltas(3, "5m") ->
            [timedelta(minutes=5), timedelta(minutes=10), timedelta(minutes=15)]
    """
    granularity_secs = parse_duration_to_seconds(granularity)
    return [
        timedelta(seconds=start_offset_seconds + (i + 1) * granularity_secs)
        for i in range(n_steps)
    ]


def align_timestamp_to_granularity(timestamp: float, granularity: str) -> float:
    """
    Floor a unix timestamp to the nearest granularity boundary.

    Example:
        align_timestamp_to_granularity(1700000123.0, "5m") -> 1700000100.0
    """
    granularity_secs = parse_duration_to_seconds(granularity)
    return float(int(timestamp) // granularity_secs * granularity_secs)