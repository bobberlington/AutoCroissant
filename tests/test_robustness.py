"""Robustness: queue loops survive bad messages (B5), init_vc (B4), card list kept on GitHub errors (B7),
temp files cleaned, git timeouts, help/describe text (C1-C3), unique handler names (C5), no /list_guild_members (C6)."""
import contextlib
import asyncio
import os
import re
import signal
import subprocess
import tempfile
from io import BytesIO
from types import SimpleNamespace

import pytest
from discord import File, Forbidden, HTTPException, Object
from requests.exceptions import ChunkedEncodingError, ConnectionError as RequestsConnectionError

import main
import commands.card_data as cd
import commands.github_api as ga
import commands.help as help_module
import commands.music_player as mp
import commands.psd_analyzer as pa
import commands.query_card as qc
import commands.type_emojis as te
import commands.update_bot as ub
import commands.utils as utils


# ========================
# Fakes
# ========================
def _http(status, cls=HTTPException):
    return cls(SimpleNamespace(status=status, reason="x"), "fail")


class Response:
    def __init__(self, inter):
        self.inter = inter

    def is_done(self):
        return self.inter.done

    async def send_message(self, **kw):
        self.inter.done = True
        self.inter.log.append(("response", kw))


class Followup:
    def __init__(self, inter):
        self.inter = inter

    async def send(self, **kw):
        for f in [kw.get("file")] + list(kw.get("files") or []):
            if f is not None:
                f.fp.read()                              # a failed upload still reads the file
        if self.inter.fail:
            raise self.inter.fail
        self.inter.log.append(("followup", kw))


class Inter:
    def __init__(self, channel_id=1, done=False, fail=None, edit_fail=None):
        self.channel_id, self.done, self.fail, self.edit_fail, self.log = channel_id, done, fail, edit_fail, []
        self.response, self.followup = Response(self), Followup(self)

    async def edit_original_response(self, **kw):
        if self.edit_fail:
            raise self.edit_fail
        self.log.append(("edit", kw))


class Channel:
    def __init__(self, fail=None):
        self.sent, self.fail = [], fail

    async def send(self, **kw):
        if self.fail:
            raise self.fail
        files = [kw.get("file")] + list(kw.get("files") or [])
        self.sent.append({**kw, "_bytes": [f.fp.read() for f in files if f is not None]})


@pytest.fixture
def channels(monkeypatch):
    found = {}
    monkeypatch.setattr(main.client, "get_channel", lambda cid: found.get(cid))
    for q in (utils.dispatch_queue, utils.edit_queue, utils.command_queue):
        q.clear()
    main._lanes.clear()
    main._lane_tasks.clear()
    yield found
    for q in (utils.dispatch_queue, utils.edit_queue, utils.command_queue):
        q.clear()


def _drain():
    """One tick of the dispatch loop, then its channel lanes until they are empty."""
    async def run():
        await main.process_dispatch_queue.coro()
        await asyncio.gather(*main._lane_tasks.values())
    asyncio.run(run())


# ========================
# B5: dispatch and edit loops
# ========================
def test_missing_channel_is_logged_and_the_loop_goes_on(channels, capsys):
    channels[2] = Channel()
    utils.queue_message(utils.make_simple_fake_interaction(1), "to a deleted channel")
    utils.queue_message(utils.make_simple_fake_interaction(2), "next")
    _drain()
    assert [m["content"] for m in channels[2].sent] == ["next"]
    assert "channel 1 not found" in capsys.readouterr().out


def test_each_channel_has_its_own_lane(channels):
    """A 40-card search in one channel doesn't hold another channel's reply; each channel keeps its order."""
    order = []

    class Slow(Channel):
        async def send(self, **kw):
            await asyncio.sleep(0)                      # a send takes a while: other lanes go meanwhile
            order.append(kw["content"])

    channels[1], channels[2] = Slow(), Slow()
    for i in range(40):
        utils.queue_message(utils.make_simple_fake_interaction(1), f"card {i}")
    utils.queue_message(utils.make_simple_fake_interaction(2), "other reply")
    _drain()
    assert [m for m in order if m.startswith("card")] == [f"card {i}" for i in range(40)]
    assert order.index("other reply") < 3


FAILING_FALLBACK_IS_LOGGED_AND_THE_LOOP_GOES_ON_CASES = [_http(403, Forbidden), _http(400), ValueError("bad file"), TypeError("bad kwarg")]


def test_failing_fallback_is_logged_and_the_loop_goes_on(channels, capsys):
    for error in FAILING_FALLBACK_IS_LOGGED_AND_THE_LOOP_GOES_ON_CASES:
        channels[1], channels[2] = Channel(fail=error), Channel()
        utils.queue_message(utils.make_simple_fake_interaction(1), "a reminder in a channel the bot can't post in")
        utils.queue_message(utils.make_simple_fake_interaction(2), "next")
        _drain()
        assert [m["content"] for m in channels[2].sent] == ["next"]
        assert f"Dispatch: dropped a message for channel 1: {type(error).__name__}" in capsys.readouterr().out


def test_long_content_is_split(channels):
    inter = Inter()
    lines = [f"song {i:04d} " + "x" * 40 for i in range(130)]           # like /queue of 130 songs
    text = "Queued songs:\n```" + "\n".join(lines) + "```"
    assert len(text) > 6000
    am = Object(1)                                                      # an allowed_mentions stand-in
    utils.queue_file(inter, File(BytesIO(b"abc"), filename="a.txt"), content=text, ephemeral=True, allowed_mentions=am)
    _drain()
    sent = [kw for _, kw in inter.log]
    assert all(kw["allowed_mentions"] is am for kw in sent)                # every chunk, not only the last
    assert [kind for kind, _ in inter.log] == ["response"] + ["followup"] * (len(sent) - 1)
    assert len(sent) >= 4 and all(len(kw["content"]) <= 2000 and kw["ephemeral"] for kw in sent)
    assert all("file" not in kw for kw in sent[:len(sent) - 1]) and sent[len(sent) - 1]["file"].filename == "a.txt"
    body = [l for kw in sent for l in kw["content"].split("\n") if l.strip("`")]
    assert [l.strip("`") for l in body] == ["Queued songs:"] + lines


def test_fallback_resends_whole_files(channels):
    channels[1] = Channel()
    inter = Inter(done=True, fail=_http(404))                            # expired interaction token
    utils.queue_file(inter, File(BytesIO(b"z" * 1000), filename="cards.csv"), content="export", ephemeral=True)
    _drain()
    (msg,) = channels[1].sent
    assert msg["_bytes"] == [b"z" * 1000] and "ephemeral" not in msg


def test_edit_loop_survives(channels, capsys):
    channels[2] = Channel()
    utils.queue_edit(Inter(channel_id=1, edit_fail=_http(404)), content="progress")
    good = Inter(channel_id=2)
    utils.queue_edit(good, content="x" * 4500)
    asyncio.run(main.process_edit_queue.coro())
    assert "channel 1 not found" in capsys.readouterr().out
    (kind, kw), = good.log
    assert kind == "edit" and len(kw["content"]) <= 2000
    rest = [kw["content"] for _, kw in utils.dispatch_queue]
    assert len(rest) == 2 and kw["content"] + "".join(rest) == "x" * 4500


class Sent:
    def __init__(self, ephemeral):
        self.flags = SimpleNamespace(ephemeral=ephemeral)


@pytest.mark.parametrize("ephemeral", [True])
def test_long_edit_keeps_ephemeral(channels, ephemeral):
    inter = Inter()
    async def edit(**kw):
        inter.log.append(("edit", kw))
        return Sent(ephemeral)
    inter.edit_original_response = edit
    am = Object(1)
    utils.queue_edit(inter, content="x" * 4500, allowed_mentions=am)
    asyncio.run(main.process_edit_queue.coro())
    assert [kw.get("ephemeral", False) for _, kw in utils.dispatch_queue] == [ephemeral, ephemeral]
    assert all(kw["allowed_mentions"] is am for _, kw in utils.dispatch_queue)


def test_command_queue_runs_and_survives(channels, capsys, monkeypatch):
    ran = []
    monkeypatch.setattr(main, "check_reminder", lambda: ran.append("tick"))
    async def tick():
        utils.command_queue.append("not a queued item")
        utils.queue_command(ran.append, "work")
        await main.process_reminders.coro()                 # the 1 s heartbeat queues check_reminder
        await main.process_command_queue.coro()
        async def done():
            while main._running:
                await asyncio.sleep(0.01)
        await asyncio.wait_for(done(), 5)                     # never hangs a worker
    asyncio.run(tick())
    assert sorted(ran) == ["tick", "work"] and "Command queue: dropped an item" in capsys.readouterr().out


# ========================
# B4: init_vc
# ========================
def test_init_vc_outside_voice_replies(channels, monkeypatch):
    inter = Inter()
    inter.user = SimpleNamespace(voice=None)
    assert asyncio.run(mp.init_vc(inter)) is False
    assert [kw["content"] for _, kw in utils.dispatch_queue] == ["You are not in a voice channel."]

    played = []
    async def defer(**kw):
        pass
    async def play(*a):
        played.append(a)
    inter.response.defer = defer
    monkeypatch.setattr(main, "play_music", play)
    monkeypatch.setattr(main, "play_all", play)
    asyncio.run(main.slash_play.callback(inter, "https://example.com/song"))
    asyncio.run(main.slash_play_all.callback(inter))
    assert played == []                                                  # nothing downloaded without a voice client


# ========================
# B7: card list and emoji listing
# ========================
class Reply:
    def __init__(self, status, body=None, etag=""):
        self.status_code, self._body, self.headers = status, body, {"ETag": etag} if etag else {}
        self.ok, self.content = status < 400, b""

    def json(self):
        if self._body is None:
            raise ValueError("Expecting value: line 1 column 1")       # an HTML error page
        return self._body

    def raise_for_status(self):
        if not self.ok:
            from requests.exceptions import HTTPError
            raise HTTPError(f"{self.status_code} Server Error")

    def close(self):
        pass


TREE = {"tree": [{"path": "Items/Red_Bloon.png", "sha": "r1"}, {"path": "Types/Fire.png", "sha": "f1"}]}


@pytest.fixture
def github(monkeypatch):
    replies, calls, waits = [], [], []
    def get(url, **kw):
        calls.append((url, kw))
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply
    monkeypatch.setattr(ga, "get", get)
    monkeypatch.setattr(ga, "sleep", waits.append)
    monkeypatch.setattr(qc.CardRepository, "_populate_site_ids", lambda self: None)
    return SimpleNamespace(replies=replies, calls=calls, waits=waits)


def test_get_with_retry(github):
    github.replies[:] = [RequestsConnectionError("reset"), Reply(502), Reply(200, {})]
    assert ga.get_with_retry("u").status_code == 200
    assert github.waits == [1, 2] and all(kw["timeout"] == 30 for _, kw in github.calls)
    github.replies[:] = [Reply(503)] * 3
    assert ga.get_with_retry("u").status_code == 503 and len(github.calls) == 6
    github.replies[:] = [Reply(401)]
    assert ga.get_with_retry("u").status_code == 401 and len(github.calls) == 7     # no retry on 4xx
    github.replies[:] = [RequestsConnectionError("down")] * 3
    with pytest.raises(RequestsConnectionError):
        ga.get_with_retry("u")


POPULATE_FILES_KEEPS_THE_LAST_GOOD_TREE_CASES = [[Reply(503)] * 3, [Reply(200)], [Reply(401, {"message": "Bad credentials"})],
                                     [RequestsConnectionError("down")] * 3, [Reply(200, ["not", "a", "tree"])]]


def test_populate_files_keeps_the_last_good_tree(github, capsys):
    for failure in POPULATE_FILES_KEEPS_THE_LAST_GOOD_TREE_CASES:
        repo = qc.CardRepository()
        github.replies[:] = [Reply(200, TREE, etag='"e1"')]
        assert repo.populate_files() == 200 and repo.git_files == {"red_bloon.png": "Items/Red_Bloon.png", "fire.png": "Types/Fire.png"}
        before = dict(repo.git_files)
        github.replies[:] = list(failure)
        assert repo.populate_files() != 200
        assert repo.git_files == before and repo._tree is TREE and repo._tree_etag == '"e1"'
        assert "keeping the last good list (2 entries)" in capsys.readouterr().out
        github.replies[:] = [Reply(304)]
        assert repo.populate_files() == 200 and repo.git_files == before


def test_set_repo_typo_keeps_the_old_repo(github, monkeypatch):
    repo = qc.CardRepository()
    github.replies[:] = [Reply(200, TREE)]
    repo.populate_files()
    monkeypatch.setattr(qc, "card_repo", repo)
    github.replies[:] = [Reply(404, {"message": "Not Found"})]
    sent = []
    async def defer(**kw):
        sent.append("defer")
    async def send(content):
        sent.append(content)
    inter = SimpleNamespace(response=SimpleNamespace(defer=defer), followup=SimpleNamespace(send=send))
    asyncio.run(qc.set_repository(inter, "MichaelJSr/TTSCardMakr"))
    assert repo.repository == qc.DEFAULT_REPOSITORY and len(repo.git_files) == 2
    assert sent == ["defer", f"Error 404 when connecting to MichaelJSr/TTSCardMakr; still using {qc.DEFAULT_REPOSITORY}."]


def test_set_repo_switches_only_after_the_new_tree_loads(github, monkeypatch):
    repo = qc.CardRepository()
    github.replies[:] = [Reply(200, TREE, etag='"e1"')]
    repo.populate_files()
    seen = []
    def get(url, **kw):
        seen.append((repo.repository, kw["headers"].get("If-None-Match")))
        return Reply(200, {"tree": [{"path": "Items/Other.png", "sha": "o1"}]}, etag='"e2"')
    monkeypatch.setattr(ga, "get", get)
    assert repo.populate_files("someone/Other") == 200
    assert seen == [(qc.DEFAULT_REPOSITORY, None)]                       # old repo while fetching, no stale ETag
    assert repo.repository == "someone/Other" and repo.git_files == {"other.png": "Items/Other.png"}


def test_emoji_listing_falls_back_to_the_card_tree(github, monkeypatch, capsys):
    repo = qc.CardRepository()
    monkeypatch.setattr(qc, "card_repo", repo)
    github.replies[:] = [Reply(503)] * 3
    with pytest.raises(Exception):
        te._list_type_icons()                                            # no tree yet: the sync reports it
    repo._tree = TREE
    github.replies[:] = [Reply(503)] * 3
    repository, icons = te._list_type_icons()
    assert set(icons) == {"fire", "fire_no"} and "using the last good card tree" in capsys.readouterr().out


def test_failed_emoji_listing_still_publishes_existing_emojis(github, monkeypatch, tmp_path, capsys):
    from helpers import Client
    monkeypatch.setattr(te, "EMOJI_JSON", str(tmp_path / "type_emojis.json"))
    monkeypatch.setattr(te, "_blobs", {})
    monkeypatch.setattr(te, "_blobs_loaded", True)
    monkeypatch.setattr(te, "type_emojis", {})
    monkeypatch.setattr(te, "_lock", asyncio.Lock())
    monkeypatch.setattr(qc, "card_repo", qc.CardRepository())
    github.replies[:] = [Reply(503)] * 3
    asyncio.run(te.init_type_emojis(Client(["fire", "fire_no"])))
    assert set(te.type_emojis) == {"fire", "fire_no"}                     # [fire] still renders after a startup 5xx
    assert "using the 2 existing emojis" in capsys.readouterr().out
    assert not (tmp_path / "type_emojis.json").exists()                  # a failed sync writes no blob map


# ========================
# Temp files
# ========================
def test_partial_download_is_removed(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    traverser = cd.CardBuild(pa.StatsDatabase())

    class Stream:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def iter_content(self, size):
            yield b"8BPS" * 1000
            raise ChunkedEncodingError("connection broken mid-stream")

    monkeypatch.setattr(traverser, "_blob", lambda sha, stream=False: Stream())
    assert traverser._download("Items/Card.psd", "s1") is None
    assert list(tmp_path.iterdir()) == [] and "download failed" in capsys.readouterr().out


def test_failed_json_saves_leave_no_tmp(monkeypatch, tmp_path):
    import commands.repo_history as rh
    def broken(*a, **kw):
        raise OSError("disk full")
    monkeypatch.setattr(te, "EMOJI_JSON", str(tmp_path / "type_emojis.json"))
    monkeypatch.setattr(te, "dump", broken)
    monkeypatch.setattr(rh, "HISTORY_JSON", str(tmp_path / "repo_history.json"))
    monkeypatch.setattr(rh, "dump", broken)
    for save in (te._save_blobs, rh._save):
        with pytest.raises(OSError):
            save()
    assert list(tmp_path.iterdir()) == []


# ========================
# Git timeouts
# ========================
def test_git_fetch_times_out(monkeypatch, tmp_path):
    for key, value in {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(tmp_path),
                       "XDG_CONFIG_HOME": str(tmp_path)}.items():   # no user config (git 2.25 ignores GIT_CONFIG_GLOBAL)
        monkeypatch.setenv(key, value)
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q")
    git("symbolic-ref", "HEAD", "refs/heads/main")                      # git 2.25 has no init -b
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "x")
    git("remote", "add", "origin", "ssh://git@example.invalid/repo.git")
    stall = tmp_path / "stall.sh"
    stall.write_text(f"#!/bin/sh\necho $$ >> {tmp_path}/stalls\nexec sleep 10\n")   # a stalled connection
    stall.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_SSH_COMMAND", str(stall))
    monkeypatch.setattr(ub, "NET_TIMEOUT", 1)
    sent = []
    async def send(content):
        sent.append(content)
    try:
        asyncio.run(ub.git_pull(SimpleNamespace(followup=SimpleNamespace(send=send))))
    finally:
        for pid in (tmp_path / "stalls").read_text().split() if (tmp_path / "stalls").exists() else ():
            with contextlib.suppress(OSError):                          # git's timeout leaves the stall orphaned
                os.kill(int(pid), signal.SIGKILL)
    assert sent[0] == "Fetching from remote..." and "did not complete in 1 secs" in sent[1]


# ========================
# on_ready again after a failed RESUME
# ========================
class FakeLoop:
    def __init__(self, name):
        self.coro = SimpleNamespace(__name__=name)
        self.starts, self.running, self.dead = 0, False, False

    def failed(self):
        return self.dead

    def start(self):
        if self.running and not self.dead:
            raise RuntimeError("Task is already launched and is not completed.")   # what tasks.Loop does
        self.starts, self.running, self.dead = self.starts + 1, True, False


def test_on_ready_twice_runs_the_inits_once(monkeypatch, capsys):
    loops = [FakeLoop(n) for n in ("reminders", "commands", "dispatch", "edits")]
    queued, synced = [], []
    guilds = [SimpleNamespace(id=1, name="a")]
    async def sync(guild, **kw):
        synced.append(guild.id)
    monkeypatch.setattr(main, "_loops", lambda: loops)
    monkeypatch.setattr(main, "queue_command", lambda func, *a: queued.append(func.__name__))
    monkeypatch.setattr(main, "client", SimpleNamespace(user="bot", guilds=guilds))
    monkeypatch.setattr(main.tree, "copy_global_to", lambda **kw: None)
    monkeypatch.setattr(main.tree, "sync", sync)
    monkeypatch.setattr(main, "_started", False)
    monkeypatch.setattr(main, "_synced_guilds", set())
    monkeypatch.setattr(main, "slash_registry", {})

    async def ready_twice():
        await main.on_ready()
        await asyncio.sleep(0)                                          # the guild sync tasks run
        loops[1].dead = True                                            # the command queue loop crashed
        guilds.append(SimpleNamespace(id=2, name="b"))                  # joined while disconnected
        await main.on_ready()                                           # the repeat: no RuntimeError
        await asyncio.sleep(0)
    asyncio.run(ready_twice())
    assert queued == ["init_reminder", "init_psd", "init_query", "refresh", "init_type_emojis"]
    assert [loop.starts for loop in loops] == [1, 2, 1, 1]
    assert synced == [1, 2] and main.slash_registry
    assert "Reconnected: inits skipped, revived commands" in capsys.readouterr().out


# ========================
# Commands and help (C1, C2, C3, C5, C6)
# ========================
def _registered() -> dict:
    return {cmd.name: cmd for cmd in main.tree.get_commands()}


def test_help_covers_every_command():
    documented = {re.match(r"/(\w+)", key).group(1) for d in (help_module.general_commands, help_module.text_commands,
                  help_module.card_commands, help_module.stats_commands, help_module.music_commands)
                  for key in d if key.startswith("/")}
    assert documented == set(_registered()) and len(documented) == 46
    assert "list_guild_members" not in documented and not hasattr(main, "list_guild_members")


def test_a_space_named_png_takes_the_underscore_lookups():
    """N.M.E/Black Hole.png (13 card files use spaces): Card Edits.json aliases and printed titles build underscore
    keys, which find it too."""
    repo = qc.CardRepository()
    repo._add_file_to_registry("N.M.E/Black Hole.png")
    repo._add_lookup("Black Hole", ["Void Thing"])
    assert repo.git_files["void_thing.png"] == repo.git_files["black hole.png"] == "N.M.E/Black Hole.png"
    assert repo._find_closest_match("Black Hole") == "black_hole.png"

