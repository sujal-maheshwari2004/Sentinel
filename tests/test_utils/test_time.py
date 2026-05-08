# tests/test_utils/test_time.py

import pytest
from datetime import timedelta
from sentinel.utils.time import (
    parse_duration_to_seconds,
    parse_duration_to_steps,
    steps_to_timedeltas,
    align_timestamp_to_granularity,
)


class TestParseDurationToSeconds:

    def test_seconds(self):
        assert parse_duration_to_seconds("30s") == 30

    def test_minutes(self):
        assert parse_duration_to_seconds("5m") == 300

    def test_hours(self):
        assert parse_duration_to_seconds("1h") == 3600

    def test_days(self):
        assert parse_duration_to_seconds("2d") == 172800

    def test_weeks(self):
        assert parse_duration_to_seconds("1w") == 604800

    def test_strips_whitespace(self):
        assert parse_duration_to_seconds("  5m  ") == 300

    def test_case_insensitive(self):
        assert parse_duration_to_seconds("5M") == 300

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            parse_duration_to_seconds("5minutes")

    def test_missing_unit_raises(self):
        with pytest.raises(ValueError):
            parse_duration_to_seconds("300")

    def test_missing_value_raises(self):
        with pytest.raises(ValueError):
            parse_duration_to_seconds("m")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            parse_duration_to_seconds("")


class TestParseDurationToSteps:

    def test_basic(self):
        assert parse_duration_to_steps("30m", "1m") == 30

    def test_hour_to_five_minutes(self):
        assert parse_duration_to_steps("1h", "5m") == 12

    def test_hour_to_thirty_seconds(self):
        assert parse_duration_to_steps("1h", "30s") == 120

    def test_same_duration_and_granularity(self):
        assert parse_duration_to_steps("5m", "5m") == 1

    def test_not_divisible_raises(self):
        with pytest.raises(ValueError):
            parse_duration_to_steps("7m", "5m")

    def test_granularity_larger_than_duration_raises(self):
        with pytest.raises(ValueError):
            parse_duration_to_steps("1m", "5m")

    def test_day_to_hours(self):
        assert parse_duration_to_steps("1d", "1h") == 24


class TestStepsToTimedeltas:

    def test_basic(self):
        result = steps_to_timedeltas(3, "5m")
        assert result == [
            timedelta(minutes=5),
            timedelta(minutes=10),
            timedelta(minutes=15),
        ]

    def test_with_offset(self):
        result = steps_to_timedeltas(2, "1h", start_offset_seconds=3600)
        assert result == [
            timedelta(hours=2),
            timedelta(hours=3),
        ]

    def test_single_step(self):
        result = steps_to_timedeltas(1, "1m")
        assert result == [timedelta(minutes=1)]

    def test_seconds_granularity(self):
        result = steps_to_timedeltas(3, "30s")
        assert result == [
            timedelta(seconds=30),
            timedelta(seconds=60),
            timedelta(seconds=90),
        ]


class TestAlignTimestampToGranularity:

    def test_already_aligned(self):
        ts = 1700000100.0  # already on 1m boundary
        assert align_timestamp_to_granularity(ts, "1m") == 1700000100.0

    def test_floors_to_minute(self):
        ts = 1700000123.0
        assert align_timestamp_to_granularity(ts, "1m") == 1700000100.0

    def test_floors_to_five_minutes(self):
        ts = 1700000423.0  # 123s past a 5m boundary
        result = align_timestamp_to_granularity(ts, "5m")
        assert result % 300 == 0
        assert result <= ts

    def test_floors_to_hour(self):
        ts = 1700003723.0  # some seconds into an hour
        result = align_timestamp_to_granularity(ts, "1h")
        assert result % 3600 == 0
        assert result <= ts