"""Robustness: queue loops survive bad messages (B5), init_vc (B4), card list kept on GitHub errors (B7),
temp files cleaned, git timeouts, help/describe text (C1-C3), unique handler names (C5), no /list_guild_members (C6)."""
import ast
import asyncio
import inspect
import os
import re
import subprocess
import tempfile
from io import BytesIO
from types import SimpleNamespace

import pytest
from discord import File, Forbidden, HTTPException, Object
from requests.exceptions import ChunkedEncodingError, ConnectionError as RequestsConnectionError

import main
import commands.github_api as ga
import commands.help as help_module
import commands.music_player as mp
import commands.psd_analyzer as pa
import commands.query_card as qc
import commands.type_emojis as te
import commands.update_bot as ub
import commands.utils as utils
from conftest import REPO


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
    yield found
    for q in (utils.dispatch_queue, utils.edit_queue, utils.command_queue):
        q.clear()


def _drain():
    asyncio.run(main.process_dispatch_queue.coro())


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


@pytest.mark.parametrize("error", [_http(403, Forbidden), _http(400), ValueError("bad file"), TypeError("bad kwarg")])
def test_failing_fallback_is_logged_and_the_loop_goes_on(channels, capsys, error):
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


@pytest.mark.parametrize("ephemeral", [True, False])
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


def test_failed_long_edit_keeps_its_order(channels, capsys):
    utils.queue_edit(Inter(channel_id=1, edit_fail=_http(404)), content="x" * 4500)
    asyncio.run(main.process_edit_queue.coro())
    assert not utils.dispatch_queue and "channel 1 not found" in capsys.readouterr().out    # no start, no rest
    channels[1] = Channel()
    utils.queue_edit(Inter(channel_id=1, edit_fail=_http(404)), content="x" * 4500)
    asyncio.run(main.process_edit_queue.coro())
    assert len(channels[1].sent) == 1 and len(utils.dispatch_queue) == 2                 # first chunk went first


def test_command_queue_runs_and_survives(channels, capsys, monkeypatch):
    ran = []
    monkeypatch.setattr(main, "check_reminder", lambda: ran.append("tick"))
    async def tick():
        utils.command_queue.append("not a queued item")
        utils.queue_command(ran.append, "work")
        await main.process_reminders.coro()                 # the 1 s heartbeat queues check_reminder
        await main.process_command_queue.coro()
        while main._running:
            await asyncio.sleep(0.01)
    asyncio.run(tick())
    assert sorted(ran) == ["tick", "work"] and "Command queue: dropped an item" in capsys.readouterr().out


def test_run_command_logs_failures(capsys):
    def broken_worker():
        raise RuntimeError("boom")
    asyncio.run(main.run_command(broken_worker, (), {}))
    out = capsys.readouterr().out
    assert "Queued command broken_worker failed" in out and "RuntimeError: boom" in out


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
        self.ok = status < 400

    def json(self):
        if self._body is None:
            raise ValueError("Expecting value: line 1 column 1")       # an HTML error page
        return self._body

    def raise_for_status(self):
        if not self.ok:
            from requests.exceptions import HTTPError
            raise HTTPError(f"{self.status_code} Server Error")


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


@pytest.mark.parametrize("failure", [[Reply(503)] * 3, [Reply(200)], [Reply(401, {"message": "Bad credentials"})],
                                     [RequestsConnectionError("down")] * 3, [Reply(200, ["not", "a", "tree"])]])
def test_populate_files_keeps_the_last_good_tree(github, capsys, failure):
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


def test_populate_files_retries_then_succeeds(github):
    repo = qc.CardRepository()
    github.replies[:] = [Reply(502), Reply(200, TREE)]
    assert repo.populate_files() == 200 and len(repo.git_files) == 2 and github.waits == [1]


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
    assert repo.populate_files(None, "someone/Other") == 200
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
    from test_type_emojis import Client
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
    traverser = pa.RepositoryTraverser(pa.StatsDatabase())

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
    stall.write_text("#!/bin/sh\nexec sleep 10\n")                      # a stalled connection
    stall.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_SSH_COMMAND", str(stall))
    monkeypatch.setattr(ub, "NET_TIMEOUT", 1)
    sent = []
    async def send(content):
        sent.append(content)
    asyncio.run(ub.git_pull(SimpleNamespace(followup=SimpleNamespace(send=send))))
    assert sent[0] == "Fetching from remote..." and "did not complete in 1 secs" in sent[1]


def test_push_reports_the_pull_error_not_the_abort(monkeypatch):
    class FakeGit:
        def __init__(self, *a):
            pass
        def add(self, *a):
            pass
        def commit(self, *a):
            raise ub.GitCommandError("commit", 1, b"nothing to commit")
        def pull(self, *a, **kw):
            raise ub.GitCommandError("pull", -9, b"did not complete in 120 secs")
        def rebase(self, *a):
            raise ub.GitCommandError("rebase", 128, b"No rebase in progress?")
    monkeypatch.setattr(ub, "Git", FakeGit)
    sent = []
    async def send(content):
        sent.append(content)
    with pytest.raises(ub.GitCommandError):
        asyncio.run(ub.git_push(SimpleNamespace(followup=SimpleNamespace(send=send))))
    assert "did not complete" in sent[0] and "rebase" not in sent[0]


# ========================
# Commands and help (C1, C2, C3, C5, C6)
# ========================
def _registered() -> dict:
    return {cmd.name: cmd for cmd in main.tree.get_commands()}


def test_handler_names_are_unique():
    tree = ast.parse((REPO / "main.py").read_text(encoding="utf-8"))
    names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert len(names) == len(set(names))
    cmds = _registered()
    assert cmds["set_ratio"].callback.__name__ == "slash_set_ratio"
    assert cmds["set_repo"].callback.__name__ == "slash_set_repo"
    assert "new_repo" in inspect.signature(cmds["set_repo"].callback).parameters
    assert cmds["delete_song"].callback.__name__ == "slash_delete_song"
    assert cmds["delete_all_music"].callback.__name__ == "slash_delete_all_music"


def test_help_covers_every_command():
    documented = {re.match(r"/(\w+)", key).group(1) for d in (help_module.general_commands, help_module.text_commands,
                  help_module.card_commands, help_module.stats_commands, help_module.music_commands)
                  for key in d if key.startswith("/")}
    assert documented == set(_registered()) and len(documented) == 52
    assert "list_guild_members" not in documented and not hasattr(main, "list_guild_members")


def test_help_and_describe_match_the_code():
    cmds = _registered()
    assert "hard reset" not in help_module.general_commands["/pull:"] and "hard reset" not in cmds["pull"].description
    assert "(admin only)" in cmds["pull"].description
    export = help_module.stats_commands["/export_cards [only_ability] [as_csv]:"]
    defaults = inspect.signature(cmds["export_cards"].callback).parameters
    assert defaults["only_ability"].default is False and "only_ability: Text export only" in export
    assert "(default: False)" in export.split("\n")[1] and "(default: True)" in export.split("\n")[2]
    params = {p.name: p.description for p in cmds["export_cards"].parameters}
    assert params["only_ability"].endswith("Default = False.") and params["as_csv"].endswith("Default = True.")
