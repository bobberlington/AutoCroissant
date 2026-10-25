"""Reminders (analytics.py): reminder.pkl -> reminder.json once, the on/off flag, the /reminders panel's navigation."""
import json
import pickle
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import commands.analytics as an
import commands.utils as utils
from global_config import TIMEZONE

GUILD, CHANNEL, ADMIN, OTHER = 11, 22, 1, 2


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    """Nothing loaded yet; ADMINS = [ADMIN]; the queues empty (cwd is the test's temp dir)."""
    monkeypatch.setattr(utils, "ADMINS", [ADMIN])
    for name, value in (("reminders", defaultdict(list)), ("_loaded", False), ("_load_error", None),
                        ("_backend", "json"), ("_conversion_error", None), ("_client", None)):
        monkeypatch.setattr(an, name, value)
    utils.dispatch_queue.clear()
    utils.command_queue.clear()
    yield
    utils.dispatch_queue.clear()
    utils.command_queue.clear()


def _reload(monkeypatch):
    monkeypatch.setattr(an, "_loaded", False)
    monkeypatch.setattr(an, "reminders", defaultdict(list))
    an.init_reminder()


def _user(user_id=ADMIN):
    return SimpleNamespace(guild_id=GUILD, channel_id=CHANNEL, user=SimpleNamespace(id=user_id, name="owner"))


def _jetson_pickle():
    """Shaped like the jetson's reminder.pkl: a legacy weekly message (no creator) and two daily commands, all v3."""
    def at(day, hour, minute=0):
        return datetime(2027, 1, day, hour, minute, tzinfo=TIMEZONE)
    data = defaultdict(list)
    data[GUILD] = [
        {"id": "134e3b24", "channel_id": CHANNEL, "msg": "Game night!", "when": at(10, 18), "frequency": timedelta(weeks=1),
         "command": None, "creator_id": None, "creator_name": "", "v": 3},
        {"id": "d2664752", "channel_id": CHANNEL, "msg": "", "when": at(10, 5), "frequency": timedelta(days=1),
         "command": "/update_stats", "creator_id": 123456789012345678, "creator_name": "owner", "v": 3},
        {"id": "054b5933", "channel_id": CHANNEL, "msg": "", "when": at(10, 5, 30), "frequency": timedelta(days=1),
         "command": "/push", "creator_id": 123456789012345678, "creator_name": "owner", "v": 3},
    ]
    Path(an.LEGACY_PATH).write_bytes(pickle.dumps(data))
    return data


def test_pkl_converts_once(monkeypatch):
    original = _jetson_pickle()
    an.init_reminder()
    assert not Path(an.LEGACY_PATH).exists() and Path(f"{an.LEGACY_PATH}.bak").exists() and an.status() is None
    doc = json.loads(Path(an.STORE_PATH).read_text())
    entries = doc["guilds"][str(GUILD)]
    assert [e["id"] for e in entries] == ["134e3b24", "d2664752", "054b5933"]
    assert entries[1] == {"id": "d2664752", "channel_id": str(CHANNEL), "msg": "", "when": "2027-01-10T05:00:00-08:00",
                          "frequency": 86400, "command": "/update_stats", "enabled": True,
                          "creator_id": "123456789012345678", "creator_name": "owner", "v": 3, "problem": None}
    assert entries[0]["creator_id"] is None and entries[0]["frequency"] == 604800

    loaded = an.snapshot(GUILD)
    for before, after in zip(original[GUILD], loaded):
        assert {k: after[k] for k in before} == before and after["when"].tzinfo is TIMEZONE
    assert loaded[1]["when"] + timedelta(days=180) == datetime(2027, 7, 9, 5, tzinfo=TIMEZONE)   # wall time across DST

    Path(an.LEGACY_PATH).write_bytes(pickle.dumps(defaultdict(list)))   # a stray pickle later is ignored
    _reload(monkeypatch)
    assert an.snapshot(GUILD) == loaded and an.ReminderStore(an.STORE_PATH).serialize(an.reminders) == \
        Path(an.STORE_PATH).read_text()                                 # a no-op save is a zero diff


def test_failed_conversion_keeps_pickle(monkeypatch):
    _jetson_pickle()
    before = Path(an.LEGACY_PATH).read_bytes()

    def broken(self):
        raise an.StoreError("boom")
    monkeypatch.setattr(an.ReminderStore, "load", broken)               # written, then can't be read back
    an.init_reminder()
    assert [r["id"] for r in an.snapshot(GUILD)] == ["134e3b24", "d2664752", "054b5933"]
    assert "conversion failed" in an.status() and not Path(an.STORE_PATH).exists()
    assert Path(an.LEGACY_PATH).read_bytes() == before and not Path(f"{an.LEGACY_PATH}.bak").exists()

    an.set_enabled(_user(), "134e3b24", False)                          # this session still saves to the pickle
    saved = pickle.loads(Path(an.LEGACY_PATH).read_bytes())
    assert saved[GUILD][0]["enabled"] is False and not Path(an.STORE_PATH).exists()


def test_disabled_skips_and_resumes():
    an.init_reminder()
    now = datetime.now(TIMEZONE)
    daily = {"id": "aaaa0001", "channel_id": CHANNEL, "msg": "daily", "when": now - timedelta(days=3, hours=1),
             "frequency": timedelta(days=1), "command": None, "enabled": False, "creator_id": OTHER,
             "creator_name": "other", "v": an.REMINDER_VERSION, "problem": None}
    once = dict(daily, id="aaaa0002", msg="once", frequency=None)
    due = daily["when"]
    an.reminders[GUILD] = [daily, once]
    an.check_reminder()
    assert not utils.dispatch_queue and [r["when"] for r in an.snapshot(GUILD)] == [due, due]

    with pytest.raises(an.ReminderError, match="admin or the reminder's creator"):
        an.set_enabled(_user(3), "aaaa0001", True)
    reminder, note = an.set_enabled(_user(OTHER), "aaaa0001", True)     # its creator
    assert reminder["enabled"] and reminder["when"] > now and reminder["when"] - due == timedelta(days=4)
    assert "missed runs skipped" in note
    with pytest.raises(an.ReminderError, match="Reschedule it first"):
        an.set_enabled(_user(), "aaaa0002", True)
    reminder, notes = an.reschedule(_user(), "aaaa0002", "30m", "")
    assert not reminder["enabled"] and reminder["when"] > now and notes == ["It is still off."]
    assert an.set_enabled(_user(), "aaaa0002", True)[0]["enabled"]
    an.check_reminder()
    assert not utils.dispatch_queue                                    # neither is due


def test_panel_navigation():
    assert [an.step(3, 2, 1), an.step(3, 0, -1), an.step(0, 0, 1)] == [0, 2, -1]
    ids = ["a", "b", "c"]
    assert an.locate(ids, "b", 0) == 1
    assert an.locate(["a", "c"], "b", 1) == 1 and an.locate(["a"], "b", 2) == 0 and an.locate([], "b", 0) == -1
    assert an.select_page(25, 24) == (0, 25, False, False)
    assert an.select_page(60, 0) == (0, 23, False, True)
    assert an.select_page(60, 30) == (23, 46, True, True)
    assert an.select_page(60, 59) == (46, 60, True, False)
    mine = {"creator_id": OTHER}
    assert an.can_edit(mine, OTHER, False) and not an.can_edit(mine, 3, False) and an.can_edit(mine, 3, True)
    assert not an.can_edit({"creator_id": None}, OTHER, False)
