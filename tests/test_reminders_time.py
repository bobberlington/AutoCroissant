"""Reminder time parsing (analytics.py) for the formats /help lists under /set_reminder."""
from datetime import datetime, timedelta

import pytest

from commands.analytics import (MAX_INTERVAL, checked_interval, format_interval, next_run, parse_interval,
                                parse_schedule, parse_when)
from global_config import TIMEZONE

NOW = datetime(2026, 10, 18, 4, 0, tzinfo=TIMEZONE)        # a Sunday, 4 AM Pacific


def at(day, hour, minute=0):
    return datetime(2026, 10, day, hour, minute, tzinfo=TIMEZONE)


@pytest.mark.parametrize("text,expected", [
    ("13:00", at(18, 13)), ("1PM", at(18, 13)), ("1pm", at(18, 13)), (" 1 PM ", at(18, 13)),
    ("1:30PM", at(18, 13, 30)), ("1:30 pm", at(18, 13, 30)), ("12AM", at(18, 0)), ("12PM", at(18, 12)),
    ("3AM", at(18, 3)),                                      # passed today: set_reminder fires it right away
    ("today 5AM", at(18, 5)), ("tomorrow 9AM", at(19, 9)), ("tmrw 9am", at(19, 9)),
    ("sun 3AM", at(25, 3)),                                  # weekday: never in the past
    ("sun 5AM", at(18, 5)), ("Sunday 5AM", at(18, 5)), ("sun. 5am", at(18, 5)), ("sun, 5am", at(18, 5)),
    ("tues 9AM", at(20, 9)), ("sat 23:59", at(24, 23, 59)),
])
def test_when(text, expected):
    assert parse_when(text, NOW) == (expected, False)


@pytest.mark.parametrize("text", ["2026-12-25 9AM", "2026-12-25, 9AM", "2026-12-25t9am"])
def test_when_date(text):
    assert parse_when(text, NOW) == (datetime(2026, 12, 25, 9, tzinfo=TIMEZONE), True)


@pytest.mark.parametrize("text", ["", "25:00", "blah 3AM", "su 3AM", "2026-02-30 9AM", "3", "13:00 tomorrow",
                                  "tomorrow", "1:30:00PM"])
def test_when_invalid(text):
    assert parse_when(text, NOW) == (None, False)


def test_when_across_daylight_saving():
    sat = datetime(2026, 10, 31, 12, 0, tzinfo=TIMEZONE)
    when, _ = parse_when("sun 3AM", sat)
    assert when == datetime(2026, 11, 1, 3, tzinfo=TIMEZONE) and when.utcoffset() == timedelta(hours=-8)
    assert sat.utcoffset() == timedelta(hours=-7)


@pytest.mark.parametrize("text,seconds", [
    ("5s", 5), ("10m", 600), ("2h", 7200), ("1d", 86400), ("3w", 3 * 604800), ("1d12h", 129600),
    (" 1d 12h ", 129600), ("1H", 3600), ("0m", 0),
])
def test_interval(text, seconds):
    assert parse_interval(text) == timedelta(seconds=seconds)


@pytest.mark.parametrize("text", ["", None, "1x", "1.5h", "h", "1d-2h", "every 1d"])
def test_interval_invalid(text):
    assert parse_interval(text) is None


def test_checked_interval():
    assert checked_interval("frequency", "  ") is None
    assert checked_interval("frequency", "1w") == timedelta(weeks=1)
    for raw, reason in (("0m", "longer than zero"), ("367d", "maximum"), ("1y", "invalid")):
        with pytest.raises(ValueError, match=reason):
            checked_interval("frequency", raw)
    assert checked_interval("offset", f"{MAX_INTERVAL.days}d") == MAX_INTERVAL
    with pytest.raises(ValueError, match="days away"):                 # a year typo
        parse_schedule("9999-12-31 11PM", "", "", NOW)


@pytest.mark.parametrize("text", ["1w", "1d12h", "30m", "1w2d3h4m5s"])
def test_format_interval_round_trip(text):
    assert format_interval(parse_interval(text)) == text


def test_next_run_skips_missed_runs():
    assert next_run(NOW - timedelta(hours=2, minutes=30), timedelta(hours=1), NOW) == NOW + timedelta(minutes=30)
    assert next_run(NOW - timedelta(hours=1), timedelta(hours=1), NOW) == NOW + timedelta(hours=1)
