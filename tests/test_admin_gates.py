"""Admin-only commands (2026-10-22): an admin passes, a non-admin is refused, a non-admin's reminder refuses at fire time."""
import asyncio
from types import SimpleNamespace

import pytest

import commands.analytics as an
import commands.card_sync as cs
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
def test_admin_passes(setup):
    """An admin runs each gated command; a non-admin is refused before its worker."""
    for name, (worker, args, _) in GATED.items():
        for user, ran in ((OTHER, []), (ADMIN, [worker])):
            setup.clear()
            inter = _interaction(user)
            asyncio.run(_callback(name)(inter, *args))
            assert setup == ran and (DENIED in inter.sent) == (user == OTHER), (name, user)


SCHEDULABLE = [name for name in GATED if name != "purge"]   # purge's user option has no default


@pytest.mark.parametrize("name", SCHEDULABLE)
def test_non_admin_reminder_refused_at_fire_time(name, setup):
    assert _fire(GATED[name][2], OTHER) == [DENIED] and setup == []


def test_purge_never_runs_from_a_reminder(setup):
    # If purge gets a scheduled default, invert this (admin runs, non-admin refused); don't delete.
    sent = _fire(GATED["purge"][2], ADMIN)
    assert setup == [] and len(sent) == 1 and "`user` is required" in sent[0]

