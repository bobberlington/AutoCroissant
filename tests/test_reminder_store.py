"""Reminders (analytics.py): reminder.json read back as written, the on/off flag, the /reminders panel's navigation."""
import json
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
                        ("_client", None)):
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


def _jetson_store():
    """Shaped like the jetson's reminders: a legacy weekly message (no creator) and two daily commands, all v3."""
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
    an.ReminderStore(an.STORE_PATH).save(data)
    return data


def test_store_reads_back_as_written(monkeypatch):
    original = _jetson_store()
    an.init_reminder()
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

    _reload(monkeypatch)
    assert an.snapshot(GUILD) == loaded and an.ReminderStore(an.STORE_PATH).serialize(an.reminders) == \
        Path(an.STORE_PATH).read_text()                                 # a no-op save is a zero diff

    gap = datetime(2099, 3, 8, 2, 30, tzinfo=TIMEZONE)                  # spring-forward gap: kept as written
    an.reminders[GUILD].append(dict(loaded[1], id="aaaa0003", when=gap))
    an.save_reminders()
    _reload(monkeypatch)
    assert an._load_error is None and an.snapshot(GUILD)[3]["when"].utcoffset() == timedelta(hours=-8)

    doc = json.loads(Path(an.STORE_PATH).read_text())
    doc["guilds"][str(GUILD)][0]["frequency"] = 10 ** 15                # overflows timedelta
    Path(an.STORE_PATH).write_text(json.dumps(doc))
    before = Path(an.STORE_PATH).read_bytes()
    _reload(monkeypatch)
    assert "too large" in an._load_error and Path(an.STORE_PATH).read_bytes() == before   # off, file untouched


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




def test_backup_commits_once_after_saves(monkeypatch, tmp_path):
    """Saves in a row make one backup commit (only under systemd), pushed to origin; unchanged bytes commit nothing."""
    import subprocess
    import commands.backup as backup

    def git(*args, cwd=tmp_path):
        return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()

    git("init", "-q", "--bare", "-b", "main", "origin.git")
    git("clone", "-q", "origin.git", "live")
    live = tmp_path / "live"
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "start", cwd=live)
    git("push", "-q", "origin", "HEAD:main", cwd=live)
    monkeypatch.chdir(live)
    Path("reminder.json").write_text("{}")
    monkeypatch.setattr(backup, "BACKUP_CLONE", str(tmp_path / "backup-clone"))
    monkeypatch.setattr(backup, "BACKUP_DELAY", 0.2)
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    backup.changed("reminder.json")
    assert not backup._timers                       # not the production service: nothing scheduled
    monkeypatch.setenv("INVOCATION_ID", "x")
    for _ in range(3):
        backup.changed("reminder.json")
    backup._timers["reminder.json"].join(10)
    origin = tmp_path / "origin.git"
    assert git("log", "--format=%s", "main", cwd=origin).splitlines() == ["Backup: reminder.json", "start"]
    assert git("show", "main:backup/reminder.json", cwd=origin) == "{}"
    backup._commit("reminder.json")                 # main holds these bytes: no commit
    assert git("rev-list", "--count", "main", cwd=origin) == "2"
