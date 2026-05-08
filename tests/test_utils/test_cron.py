# tests/test_utils/test_cron.py

import pytest
from datetime import datetime, timezone, timedelta
from sentinel.utils.cron import (
    is_valid_cron,
    get_next_fire_time,
    get_previous_fire_time,
    seconds_until_next_fire,
    seconds_since_last_fire,
    get_fire_times_between,
)


class TestIsValidCron:

    def test_valid_every_hour(self):
        assert is_valid_cron("0 * * * *") is True

    def test_valid_every_six_hours(self):
        assert is_valid_cron("0 */6 * * *") is True

    def test_valid_every_minute(self):
        assert is_valid_cron("* * * * *") is True

    def test_invalid_too_few_fields(self):
        assert is_valid_cron("* * * *") is False

    def test_invalid_out_of_range(self):
        assert is_valid_cron("60 * * * *") is False

    def test_invalid_empty_string(self):
        assert is_valid_cron("") is False

    def test_invalid_random_string(self):
        assert is_valid_cron("every six hours") is False


class TestGetNextFireTime:

    def test_next_is_in_future(self):
        base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        next_fire = get_next_fire_time("0 * * * *", base=base)
        assert next_fire > base

    def test_every_hour_increments_by_one_hour(self):
        base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        next_fire = get_next_fire_time("0 * * * *", base=base)
        assert next_fire == datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

    def test_invalid_cron_raises(self):
        with pytest.raises(ValueError):
            get_next_fire_time("not a cron")


class TestGetPreviousFireTime:

    def test_prev_is_in_past(self):
        base = datetime(2024, 1, 1, 12, 30, 0, tzinfo=timezone.utc)
        prev_fire = get_previous_fire_time("0 * * * *", base=base)
        assert prev_fire < base

    def test_every_hour_returns_last_hour(self):
        base = datetime(2024, 1, 1, 12, 30, 0, tzinfo=timezone.utc)
        prev_fire = get_previous_fire_time("0 * * * *", base=base)
        assert prev_fire == datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_invalid_cron_raises(self):
        with pytest.raises(ValueError):
            get_previous_fire_time("not a cron")


class TestSecondsUntilNextFire:

    def test_returns_positive(self):
        base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        secs = seconds_until_next_fire("0 * * * *", base=base)
        assert secs > 0

    def test_roughly_one_hour(self):
        base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        secs = seconds_until_next_fire("0 * * * *", base=base)
        assert abs(secs - 3600) < 5


class TestSecondsSinceLastFire:

    def test_returns_positive(self):
        base = datetime(2024, 1, 1, 12, 30, 0, tzinfo=timezone.utc)
        secs = seconds_since_last_fire("0 * * * *", base=base)
        assert secs > 0

    def test_roughly_thirty_minutes(self):
        base = datetime(2024, 1, 1, 12, 30, 0, tzinfo=timezone.utc)
        secs = seconds_since_last_fire("0 * * * *", base=base)
        assert abs(secs - 1800) < 5


class TestGetFireTimesBetween:

    def test_correct_count(self):
        start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
        fires = get_fire_times_between("0 * * * *", start=start, end=end)
        assert len(fires) == 6

    def test_all_within_range(self):
        start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 3, 0, 0, tzinfo=timezone.utc)
        fires = get_fire_times_between("0 * * * *", start=start, end=end)
        for f in fires:
            assert start < f <= end

    def test_empty_when_no_fires_in_range(self):
        start = datetime(2024, 1, 1, 12, 0, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 12, 59, 0, tzinfo=timezone.utc)
        fires = get_fire_times_between("0 * * * *", start=start, end=end)
        assert fires == []

    def test_invalid_cron_raises(self):
        with pytest.raises(ValueError):
            get_fire_times_between(
                "bad cron",
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            )