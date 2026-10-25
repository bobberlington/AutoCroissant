"""Admin-only commands (2026-10-22): an admin passes, a non-admin is refused, a non-admin's reminder refuses at fire time."""
import asyncio
from types import SimpleNamespace

import pytest

import commands.analytics as an
import commands.card_sync as cs
import commands.help as help_module
import commands.utils as utils
import main

ADMIN, OTHER = 1, 2
DENIED = "You do not have permission."

# command -> (main.py worker name, callback args, reminder command text)
GATED = {
    "purge": ("purge", (None,), "/purge num:5"),
    "set_repo": ("set_repository", ("a/b",), "/set_repo new_repo:a/b"),
    "delete_song": ("delete_song", ("x",), "/delete_song song_name:x"),
    "delete_all_music": ("delete_all_music", (), "/delete_all_music"),
}
HELP_KEYS = {
    "purge": "/purge [user] [num] [bulk]:",
    "set_repo": "/set_repo [new_repo]:",
    "delete_song": "/delete_song <song_name>:",
    "delete_all_music": "/delete_all_music:",
}


# ========================
# Fixtures
# ========================
@pytest.fixture(autouse=True)
def setup(monkeypatch):
    """ADMINS = [ADMIN], every worker recorded, slash_registry filled as on_ready does, queues empty."""
    monkeypatch.setattr(utils, "ADMINS", [ADMIN])
    monkeypatch.setattr(main, "client", SimpleNamespace(user=SimpleNamespace(id=99)))
    monkeypatch.setattr(cs, "on_use", lambda command=None: False)   # no look for new card data
    ran = []
    for worker, _, _ in GATED.values():
        async def record(*args, _name=worker, **kwargs):
            ran.append(_name)
        monkeypatch.setattr(main, worker, record)
    for cmd in main.tree.get_commands():
        monkeypatch.setitem(utils.slash_registry, cmd.name, cmd.callback)
    queues = (utils.dispatch_queue, utils.edit_queue, utils.command_queue)
    for q in queues:
        q.clear()
    yield ran
    for q in queues:
        q.clear()


def _interaction(user_id):
    sent = []

    async def send_message(content=None, **kwargs):
        sent.append(content)

    async def defer(**kwargs):
        pass
    return SimpleNamespace(user=SimpleNamespace(id=user_id), sent=sent,
                           response=SimpleNamespace(send_message=send_message, defer=defer))


def _callback(name):
    return {cmd.name: cmd for cmd in main.tree.get_commands()}[name].callback


def _fire(command, creator_id):
    """Fire a reminder created by creator_id and run its queued command; returns the channel's messages."""
    reminder = {"id": "abcd1234", "channel_id": 5, "creator_id": creator_id, "creator_name": "someone",
                "command": command, "v": an.REMINDER_VERSION}
    assert an._fire(reminder, None)
    while utils.command_queue:
        (args, kwargs), func = utils.command_queue.popleft()
        asyncio.run(func(*args, **kwargs))
    return [kw.get("content") for _, kw in utils.dispatch_queue]


# ========================
# Tests
# ========================
@pytest.mark.parametrize("name", GATED)
def test_admin_passes(name, setup):
    worker, args, _ = GATED[name]
    inter = _interaction(ADMIN)
    asyncio.run(_callback(name)(inter, *args))
    assert setup == [worker] and DENIED not in inter.sent


@pytest.mark.parametrize("name", GATED)
def test_non_admin_refused(name, setup):
    _, args, _ = GATED[name]
    inter = _interaction(OTHER)
    asyncio.run(_callback(name)(inter, *args))
    assert setup == [] and inter.sent == [DENIED]


SCHEDULABLE = [name for name in GATED if name != "purge"]   # purge's user option has no default


@pytest.mark.parametrize("name", SCHEDULABLE)
def test_non_admin_reminder_refused_at_fire_time(name, setup):
    assert _fire(GATED[name][2], OTHER) == [DENIED] and setup == []


@pytest.mark.parametrize("name", SCHEDULABLE)
def test_admin_reminder_runs(name, setup):
    assert DENIED not in _fire(GATED[name][2], ADMIN) and setup == [GATED[name][0]]


@pytest.mark.parametrize("creator", (ADMIN, OTHER))
def test_purge_never_runs_from_a_reminder(creator, setup):
    # If purge gets a scheduled default, invert this (admin runs, non-admin refused); don't delete.
    sent = _fire(GATED["purge"][2], creator)
    assert setup == [] and len(sent) == 1 and "`user` is required" in sent[0]


@pytest.mark.parametrize("name", GATED)
def test_marked_admin_only_in_both_surfaces(name):
    cmd = {c.name: c for c in main.tree.get_commands()}[name]
    assert cmd.description.endswith("(admin only).") and len(cmd.description) <= 100
    entries = {**help_module.general_commands, **help_module.card_commands,
               **help_module.stats_commands, **help_module.music_commands}
    assert "(admin only)" in entries[HELP_KEYS[name]]
