"""Reminder time parsing (analytics.py) for the formats /help lists under /set_reminder."""
from datetime import datetime, timedelta

import pytest

from commands.analytics import MAX_INTERVAL, checked_interval, next_run, parse_interval, parse_schedule, parse_when
from global_config import TIMEZONE

NOW = datetime(2026, 10, 18, 4, 0, tzinfo=TIMEZONE)        # a Sunday, 4 AM Pacific


def at(day, hour, minute=0):
    return datetime(2026, 10, day, hour, minute, tzinfo=TIMEZONE)


WHEN_CASES = [
    ("13:00", at(18, 13)), ("1PM", at(18, 13)), ("1pm", at(18, 13)), (" 1 PM ", at(18, 13)),
    ("1:30PM", at(18, 13, 30)), ("1:30 pm", at(18, 13, 30)), ("12AM", at(18, 0)), ("12PM", at(18, 12)),
    ("3AM", at(18, 3)),                                      # passed today: set_reminder fires it right away
    ("today 5AM", at(18, 5)), ("tomorrow 9AM", at(19, 9)), ("tmrw 9am", at(19, 9)),
    ("sun 3AM", at(25, 3)),                                  # weekday: never in the past
    ("sun 5AM", at(18, 5)), ("Sunday 5AM", at(18, 5)), ("sun. 5am", at(18, 5)), ("sun, 5am", at(18, 5)),
    ("tues 9AM", at(20, 9)), ("sat 23:59", at(24, 23, 59)),
]


def test_when():
    for text, expected in WHEN_CASES:
        assert parse_when(text, NOW) == (expected, False)


def test_when_across_daylight_saving():
    sat = datetime(2026, 10, 31, 12, 0, tzinfo=TIMEZONE)
    when, _ = parse_when("sun 3AM", sat)
    assert when == datetime(2026, 11, 1, 3, tzinfo=TIMEZONE) and when.utcoffset() == timedelta(hours=-8)
    assert sat.utcoffset() == timedelta(hours=-7)


INTERVAL_CASES = [
    ("5s", 5), ("10m", 600), ("2h", 7200), ("1d", 86400), ("3w", 3 * 604800), ("1d12h", 129600),
    (" 1d 12h ", 129600), ("1H", 3600), ("0m", 0),
]


def test_interval():
    for text, seconds in INTERVAL_CASES:
        assert parse_interval(text) == timedelta(seconds=seconds)


def test_checked_interval():
    assert checked_interval("frequency", "  ") is None
    assert checked_interval("frequency", "1w") == timedelta(weeks=1)
    for raw, reason in (("0m", "longer than zero"), ("367d", "maximum"), ("1y", "invalid")):
        with pytest.raises(ValueError, match=reason):
            checked_interval("frequency", raw)
    assert checked_interval("offset", f"{MAX_INTERVAL.days}d") == MAX_INTERVAL
    with pytest.raises(ValueError, match="days away"):                 # a year typo
        parse_schedule("9999-12-31 11PM", "", "", NOW)


def test_next_run_skips_missed_runs():
    assert next_run(NOW - timedelta(hours=2, minutes=30), timedelta(hours=1), NOW) == NOW + timedelta(minutes=30)
    assert next_run(NOW - timedelta(hours=1), timedelta(hours=1), NOW) == NOW + timedelta(hours=1)
