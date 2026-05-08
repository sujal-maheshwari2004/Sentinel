# sentinel/utils/cron.py

from croniter import croniter
from datetime import datetime, timezone


def is_valid_cron(cron_string: str) -> bool:
    """
    Returns True if the cron string is valid, False otherwise.
    """
    return croniter.is_valid(cron_string)


def get_next_fire_time(cron_string: str, base: datetime = None) -> datetime:
    """
    Get the next datetime this cron expression will fire.

    base: datetime to compute from, defaults to now UTC.
    """
    if not is_valid_cron(cron_string):
        raise ValueError(f"Invalid cron string: '{cron_string}'")

    base = base or datetime.now(timezone.utc)
    cron = croniter(cron_string, base)
    return cron.get_next(datetime)


def get_previous_fire_time(cron_string: str, base: datetime = None) -> datetime:
    """
    Get the most recent datetime this cron expression fired before base.

    Useful for computing the cold start window — how long ago did the
    last scheduled training run happen?
    """
    if not is_valid_cron(cron_string):
        raise ValueError(f"Invalid cron string: '{cron_string}'")

    base = base or datetime.now(timezone.utc)
    cron = croniter(cron_string, base)
    return cron.get_prev(datetime)


def seconds_until_next_fire(cron_string: str, base: datetime = None) -> float:
    """
    Returns how many seconds until the next cron fire time.
    """
    base = base or datetime.now(timezone.utc)
    next_fire = get_next_fire_time(cron_string, base)
    return (next_fire - base).total_seconds()


def seconds_since_last_fire(cron_string: str, base: datetime = None) -> float:
    """
    Returns how many seconds have elapsed since the last cron fire time.
    Used by the scheduler to determine if cold start window has elapsed.
    """
    base = base or datetime.now(timezone.utc)
    prev_fire = get_previous_fire_time(cron_string, base)
    return (base - prev_fire).total_seconds()


def get_fire_times_between(
    cron_string: str,
    start: datetime,
    end: datetime,
) -> list[datetime]:
    """
    Returns all fire times between start and end (inclusive of start).
    Useful for backfilling or auditing missed training runs.
    """
    if not is_valid_cron(cron_string):
        raise ValueError(f"Invalid cron string: '{cron_string}'")

    cron = croniter(cron_string, start)
    times = []
    while True:
        fire = cron.get_next(datetime)
        if fire > end:
            break
        times.append(fire)
    return times