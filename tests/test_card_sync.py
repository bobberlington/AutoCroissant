"""The bot after the cutover (global_config.CARD_SYNC): card_db.sync, card_sync's /update_stats (sync, report, Card sync
runs through ACTIONS_TOKEN), /fonts from the card data, and the switch: off is today's bot, on is the cutover's."""
import asyncio
import json
from collections import defaultdict
from types import SimpleNamespace

import pytest

import commands.card_db as cdb
import commands.card_models as cm
import commands.card_sync as cs
import commands.github_api as ga
import commands.utils as utils
import global_config
from commands.card_models import CardInfo, CardStats
from conftest import Reply

REPO = global_config.CARD_REPO
RAW = f"{global_config.RAW_BASE}{REPO}/"
TOKEN = "ghp_secret_never_shown"
NOW = "2026-10-22T18:04:11Z"


# ========================
# A fake GitHub: the data branch, raw files, Card sync's runs
# ========================
class FakeGitHub:
    def __init__(self):
        self.commits: dict = {}                 # data commit -> {db file name: bytes}
        self.tip = None
        self.calls: list = []
        self.runs: list = []                    # Card sync's runs, newest first
        self.states: dict = {}                  # run id -> the run dicts GET returns, one per look (the last repeats)
        self.posts: list = []
        self.post_replies: list = []            # Reply per POST, in order
        self.on_post = None                     # called after each POST (a run appears)
        self.compare = {"ahead_by": 0, "files": []}
        self.pushes: list = []                  # the card branch's activity, newest first
        self.due: list = []                     # card_bake.due's website edits
        self.runs_reply = None                  # a Reply for the run list instead of self.runs
        self.jobs = {"jobs": [{"steps": [{"name": "Gate", "status": "completed"}, {"name": "Build", "status": "in_progress"}]}]}

    def get(self, url, headers=None, timeout=None, stream=False, **kw):
        headers = headers or {}
        if url.endswith("/commits/data"):
            self.calls.append(("commit", headers.get("If-None-Match")))
            assert headers.get("Accept") == cdb.SHA_MEDIA
            if self.tip is None:
                return Reply(404)
            etag = f'"e-{self.tip}"'
            if headers.get("If-None-Match") == etag:
                return Reply(304)
            return Reply(200, self.tip.encode(), headers={"ETag": etag})
        if url.startswith(RAW):
            sha, _, name = url[len(RAW):].partition("/db/")
            self.calls.append(("raw", name))
            assert "Authorization" not in headers   # no token to raw.githubusercontent.com
            data = self.commits.get(sha, {}).get(name)
            return Reply(404) if data is None else Reply(200, data)
        if url.endswith("/runs?per_page=10"):
            self.calls.append(("runs", None))
            return self.runs_reply or Reply(200, body={"workflow_runs": self.runs})
        if url.endswith("/jobs"):
            return Reply(200, body=self.jobs)
        if "/actions/runs/" in url:
            run_id = int(url.rsplit("/", 1)[1])
            self.calls.append(("run", run_id))
            states = self.states[run_id]
            return Reply(200, body=states.pop(0) if len(states) > 1 else states[0])
        if "/activity?" in url:
            self.calls.append(("activity", None))
            return Reply(200, body=self.pushes)
        if "/compare/" in url:
            self.calls.append(("compare", url.rsplit("/compare/", 1)[1]))
            return Reply(200, body=self.compare)
        raise AssertionError(f"unexpected GET {url}")

    def post(self, url, json=None, headers=None, timeout=None):
        assert url.endswith("/actions/workflows/card-sync.yml/dispatches")
        assert headers["Authorization"] == f"Bearer {TOKEN}"
        self.posts.append(json)
        reply = self.post_replies.pop(0) if self.post_replies else Reply(204)
        if self.on_post:
            self.on_post(len(self.posts))
        if isinstance(reply, Exception):
            raise reply
        return reply

    def publish(self, stats: dict, old: dict, folder, details: str = '{"version": 1, "cards": {\n\n}}\n', **manifest) -> str:
        """A data commit holding these cards (card_db.write_db's bytes); the branch moves to it."""
        out = folder / f"pub{len(self.commits)}"
        written = cdb.write_db(str(out), stats, old, details,
                               {"built_at": NOW, "card_repo": REPO, "card_commit": "c0ffee0123456789", "run": "https://run/1",
                                **manifest})
        files = {name: (out / name).read_bytes() for name in (*cdb.DATA_FILES, cdb.MANIFEST_JSON)}
        sha = f"{len(self.commits) + 1:040x}"
        self.commits[sha] = files
        self.tip = sha
        return written

    def kinds(self, kind: str) -> list:
        return [what for k, what in self.calls if k == kind]


def _card(name, ability="Deal 2 damage.", path=None, **kw):
    card = CardInfo(name=name, card_type="creature", path=path or f"Creatures/{name.replace(' ', '_')}.psd",
                    timestamp=1_700_000_000.0, ability=ability, stars=3, types=["fire"],
                    stats=CardStats(5, 2, 3, 4), blob_sha=kw.pop("blob_sha", f"b-{name}"), parser_version=2)
    for key, value in kw.items():
        setattr(card, key, value)
    return card


def _run(run_id, status="completed", conclusion="success", event="workflow_dispatch", login="MichaelJSr", title=cs.RUN_NAME):
    return {"id": run_id, "status": status, "conclusion": conclusion if status == "completed" else None, "event": event,
            "display_title": title, "html_url": f"https://github.com/{REPO}/actions/runs/{run_id}",
            "actor": {"login": login}}


@pytest.fixture
def github(monkeypatch, tmp_path):
    """CARD_SYNC on, cwd a temp folder (card_data/ there), GitHub faked, stats_db and the queues emptied."""
    fake = FakeGitHub()
    monkeypatch.setattr(global_config, "CARD_SYNC", True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ga, "get", fake.get)
    monkeypatch.setattr(ga, "sleep", lambda s: None)
    monkeypatch.setattr(cs, "session", lambda: SimpleNamespace(post=fake.post))
    monkeypatch.setattr(cs, "sleep", lambda s: None)
    monkeypatch.setattr(cs, "ACTIONS_TOKEN", TOKEN)
    monkeypatch.setattr(cs, "_due_edits", lambda repository: fake.due)
    monkeypatch.setattr(cm.stats_db, "stats", {})
    monkeypatch.setattr(cm.stats_db, "old_stats", defaultdict(list))
    monkeypatch.setitem(cs._loaded, "commit", None)
    queues = (utils.dispatch_queue, utils.edit_queue, utils.command_queue)
    for q in queues:
        q.clear()
    yield fake
    for q in queues:
        q.clear()


def _said() -> list:
    return [kwargs.get("content") for _, kwargs in utils.dispatch_queue]


def _edits() -> list:
    return [kwargs.get("content") for _, kwargs in utils.edit_queue]


def _queued() -> list:
    return [getattr(func, "__name__", str(func)) for _, func in utils.command_queue]


ME = SimpleNamespace(user=SimpleNamespace(id=1))


# ========================
# card_db.sync: the bot's copy of db/
# ========================
def test_first_sync_writes_the_branch_files_byte_for_byte(github, tmp_path):
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    got = cdb.sync("card_data", REPO, "data")
    assert got["commit"] == github.tip and got["changed"] == sorted(cdb.DATA_FILES)
    for name, data in github.commits[github.tip].items():
        assert (tmp_path / "card_data" / name).read_bytes() == data
    assert cdb.check_db("card_data") == []
    assert json.loads((tmp_path / "card_data" / cdb.SYNC_JSON).read_text())["commit"] == github.tip
    assert github.calls[0] == ("commit", None)


def test_unchanged_branch_costs_one_conditional_request(github, tmp_path):
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    cdb.sync("card_data", REPO, "data")
    github.calls.clear()
    got = cdb.sync("card_data", REPO, "data")
    assert got["changed"] == [] and got["commit"] == github.tip
    assert github.calls == [("commit", f'"e-{github.tip}"')]          # a 304: no raw download


def test_a_new_commit_downloads_only_the_changed_files(github, tmp_path):
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    cdb.sync("card_data", REPO, "data")
    github.publish({"A": _card("A", ability="Heal 1.")}, defaultdict(list), tmp_path)
    github.calls.clear()
    got = cdb.sync("card_data", REPO, "data")
    assert got["changed"] == [cdb.STATS_JSON] and github.kinds("raw") == [cdb.MANIFEST_JSON, cdb.STATS_JSON]
    assert cdb.read_db("card_data")[0]["A"].ability == "Heal 1."


def test_a_bad_download_writes_nothing(github, tmp_path):
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    cdb.sync("card_data", REPO, "data")
    before = {p.name: p.read_bytes() for p in (tmp_path / "card_data").iterdir()}
    github.publish({"A": _card("A", ability="Heal 1.")}, defaultdict(list), tmp_path)
    github.commits[github.tip][cdb.STATS_JSON] += b" "                # not what the manifest hashed
    with pytest.raises(ValueError, match="sha256"):
        cdb.sync("card_data", REPO, "data")
    assert {p.name: p.read_bytes() for p in (tmp_path / "card_data").iterdir()} == before


def test_a_pickle_on_the_branch_is_never_loaded(github, tmp_path):
    import pickle
    from hashlib import sha256
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    files = github.commits[github.tip]
    files[cdb.STATS_JSON] = pickle.dumps({"A": _card("A")})            # hashed right, but not JSON
    manifest = json.loads(files[cdb.MANIFEST_JSON])
    manifest["files"][cdb.STATS_JSON] = sha256(files[cdb.STATS_JSON]).hexdigest()
    files[cdb.MANIFEST_JSON] = json.dumps(manifest).encode()
    with pytest.raises(ValueError):
        cdb.sync("card_data", REPO, "data")
    assert not (tmp_path / "card_data" / cdb.STATS_JSON).exists()


@pytest.mark.parametrize("name, cards", [(cdb.STATS_JSON, {"A": 5}), (cdb.STATS_JSON, {"A": {"stats": [1]}}),
                                         (cdb.OLD_STATS_JSON, {"A": 5}), (cdb.OLD_STATS_JSON, {"A": [5]})])
def test_a_bad_record_writes_nothing_and_is_never_loaded(github, tmp_path, monkeypatch, name, cards):
    """A file whose top level is right but a record isn't: the sync writes nothing; on disk, startup passes it by."""
    from hashlib import sha256
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    files = github.commits[github.tip]
    files[name] = json.dumps({"schema": cdb.SCHEMA, "cards": cards}).encode()
    manifest = json.loads(files[cdb.MANIFEST_JSON])
    manifest["files"][name] = sha256(files[name]).hexdigest()
    files[cdb.MANIFEST_JSON] = json.dumps(manifest).encode()
    with pytest.raises(ValueError, match="A: not a"):
        cdb.sync("card_data", REPO, "data")
    assert not (tmp_path / "card_data").exists()
    (tmp_path / "card_data").mkdir()
    for file, data in files.items():                                  # on disk anyway (an older bot, a hand copy)
        (tmp_path / "card_data" / file).write_bytes(data)
    assert cs.load_local("card_data") is False and cdb.check_db("card_data")
    loads = []
    monkeypatch.setattr(cm.stats_db, "load", lambda: loads.append(1))
    cs.startup()
    assert loads == [1]                                               # the pickles, as with no card data


def test_a_damaged_copy_is_fetched_again(github, tmp_path):
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    cdb.sync("card_data", REPO, "data")
    (tmp_path / "card_data" / cdb.OLD_STATS_JSON).write_text("{")    # a crash, or a hand edit
    github.calls.clear()
    got = cdb.sync("card_data", REPO, "data")
    assert github.calls[0] == ("commit", None) and got["changed"] == [cdb.OLD_STATS_JSON]
    assert cdb.check_db("card_data") == []


def test_no_data_branch(github):
    with pytest.raises(LookupError, match="no data branch"):
        cdb.sync("card_data", REPO, "data")


# ========================
# card_sync.sync: /update_stats without options
# ========================
def test_sync_reloads_and_reports_what_changed(github, tmp_path):
    github.publish({"A": _card("A"), "B": _card("B")}, defaultdict(list), tmp_path)
    cs.sync(ME)
    assert set(cm.stats_db.stats) == {"A", "B"} and cs._loaded["commit"] == github.tip
    assert _said()[0].startswith("Card data loaded: built ") and "TTSCardMaker@c0ffee01 (run: <https://run/1>)" in _said()[0]
    assert _queued() == ["refresh", "_refresh_cards"]
    utils.dispatch_queue.clear()
    github.publish({"A": _card("A", ability="Heal 1."), "C": _card("C")},
                   defaultdict(list, {"B": [_card("B", archived={"at": 1, "path": "x", "blob": None, "commit": None})]}),
                   tmp_path, problems=[{"card": "C", "path": "Creatures/C.psd", "notes": ["HP NOT FOUND"]}])
    problems = cs.sync(ME)
    assert ": 1 changed, 1 new, 1 archived; reloaded." in _said()[0]
    assert set(cm.stats_db.stats) == {"A", "C"} and cm.stats_db.archived().keys() == {"B"}
    assert problems == ["Creatures/C.psd\n```\nHP NOT FOUND\n```"] and "HP NOT FOUND" in _said()[1]
    utils.dispatch_queue.clear()
    cs.sync(ME)
    assert _said()[0].startswith("Card data is already current: built ") and len(_said()) == 1


def test_sync_notes(github, tmp_path, monkeypatch):
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path, card_commit="c0ffee0123456789",
                   pending={"stale": ["X.psd", "Y.psd"], "failed": [{"path": "Items/Bad_Card.psd", "blob": "b", "error": "e"}],
                            "full": None, "repin": []})
    github.compare = {"ahead_by": 2, "files": [{"filename": "site/index.html"}, {"filename": "Items/New.psd"}]}
    github.runs = [_run(9, "in_progress"), _run(8, conclusion="failure", event="schedule")]
    cs.sync(ME)
    text = _said()[0]
    assert "Re-parse in progress: 2 cards still to re-parse" in text
    assert "Could not read 1 PSD (Bad Card); its card keeps its last version" in text
    assert "the card sync running now picks them up (<https://github.com/MichaelJSr/TTSCardMaker/actions/runs/9>)" in text
    assert "The last card sync failed: <https://github.com/MichaelJSr/TTSCardMaker/actions/runs/8>" in text
    assert ("compare", "c0ffee0123456789...main") in github.calls
    utils.dispatch_queue.clear()
    github.runs = [_run(8, conclusion="success", event="schedule")]
    github.compare = {"ahead_by": 2, "files": [{"filename": "site/index.html"},
                                               {"filename": "MDW/Fonts/stoogedings-src/x.svg"}]}   # no card input (the gate's)
    cs.sync(ME)
    assert len(_said()[0].splitlines()) == 3                          # the two pending lines only


def test_a_reminder_sync_is_quiet(github, tmp_path, capsys):
    from commands.analytics import _ReminderUser
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    reminder = SimpleNamespace(user=_ReminderUser(1, "owner"))
    cs.update(reminder)
    assert _said() == [] and "Card data loaded" in capsys.readouterr().out and github.posts == []
    github.tip = None                                                 # a failure still speaks
    held = cs._loaded["commit"]
    cs.update(reminder)
    assert _said() and _said()[0].startswith("No card data yet") and cs._loaded["commit"] == held


@pytest.mark.parametrize("ahead, due, what", [
    (True, [], "the card changes not in the data yet"),
    (False, ["Mini Doomer"], "1 website edit waiting to bake (Mini Doomer)"),
])
def test_update_stats_starts_a_card_sync_after_the_grace(github, tmp_path, ahead, due, what):
    """/update_stats' own run: after 30 minutes without a push (Card sync's own pushes aside), one at a time."""
    from time import gmtime, strftime, time
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    github.compare = {"ahead_by": 1, "files": [{"filename": "Items/New.psd"}]} if ahead else github.compare
    github.due = due
    ago = lambda minutes: strftime("%Y-%m-%dT%H:%M:%SZ", gmtime(time() - minutes * 60))
    github.pushes = [{"timestamp": ago(1), "actor": {"login": cs.BOT_LOGIN}}, {"timestamp": ago(10), "actor": {"login": "artist"}}]
    cs.update(ME)
    assert github.posts == [] and f"A card sync starts for {what} once the card repo has had no push for 30 minutes" in _said()[0]
    utils.dispatch_queue.clear()
    github.pushes[1]["timestamp"] = ago(40)
    _dispatched_run(github, 42, _run(42, "queued"))
    cs.update(ME)
    assert github.posts == [{"ref": "main", "inputs": cs.AUTO_INPUTS, "return_run_details": True}]
    assert f"Card sync run started for {what}: <https://github.com/{REPO}/actions/runs/42>" in _said()[0]
    github.runs = [_run(42, "queued")]
    cs.update(ME)
    assert len(github.posts) == 1                                     # none while one is queued or running


def test_a_failed_sync_keeps_the_data(github, tmp_path, monkeypatch):
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    cs.sync(ME)
    held = cm.stats_db.stats
    github.publish({"A": _card("A", ability="x")}, defaultdict(list), tmp_path)
    github.commits[github.tip][cdb.STATS_JSON] = b"broken"
    utils.dispatch_queue.clear()
    cs.sync(ME)
    assert cm.stats_db.stats is held and "Card data sync failed (ValueError" in _said()[0]


def test_startup_loads_the_local_copy_then_syncs(github, tmp_path, monkeypatch):
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    cdb.sync("card_data", REPO, "data")
    github.tip = None                                                 # GitHub unreachable at startup: the copy serves
    loads = []
    monkeypatch.setattr(cm.stats_db, "load", lambda: loads.append(1))
    cs.startup()
    assert set(cm.stats_db.stats) == {"A"} and loads == [] and _queued() == []


def test_a_cut_off_sync_is_reloaded_whole_by_the_next(github, tmp_path, monkeypatch):
    """A sync cut off after stats.json: startup loads that copy with no commit, so the next sync reloads, even one at
    the commit the copy names (the branch back at it)."""
    github.publish({"A": _card("A", ability="Old.")}, defaultdict(list), tmp_path)
    first = github.tip
    cdb.sync("card_data", REPO, "data")
    github.publish({"A": _card("A", ability="New.")}, defaultdict(list), tmp_path)
    (tmp_path / "card_data" / cdb.STATS_JSON).write_bytes(github.commits[github.tip][cdb.STATS_JSON])
    github.tip = first
    assert cs.load_local("card_data") and cm.stats_db.stats["A"].ability == "New." and cs._loaded["commit"] is None
    cs.sync(ME)
    assert cm.stats_db.stats["A"].ability == "Old." and cs._loaded["commit"] == first and cdb.check_db("card_data") == []


def test_startup_without_data_falls_back_to_the_pickles(github, monkeypatch):
    loads = []
    monkeypatch.setattr(cm.stats_db, "load", lambda: loads.append(1))
    cs.startup()
    assert loads == [1]


# ========================
# Card sync runs: force_update and cards
# ========================
def _dispatched_run(github, run_id, *states):
    """The next POST answers with run_id (return_run_details); GET returns states."""
    github.post_replies.append(Reply(200, body={"workflow_run_id": run_id}))
    github.states[run_id] = list(states)


def test_cards_dispatch_follows_the_run_then_syncs(github, tmp_path):
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    _dispatched_run(github, 41, _run(41, "queued"), _run(41, "queued"), _run(41, "in_progress"), _run(41))
    cs.update(ME, cards="Shadow Duelist, 00 Table of Contents")
    assert github.posts == [{"ref": "main", "inputs": {"dry_run": "false", "cards": "Shadow Duelist, 00 Table of Contents"},
                             "return_run_details": True}]
    run_url = "https://github.com/MichaelJSr/TTSCardMaker/actions/runs/41"
    assert _edits() == [f"Card sync run <{run_url}>: waiting for its turn", f"Card sync run <{run_url}>: running: Build"]
    said = _said()
    assert said[0] == f"Card sync run started (re-parsing Shadow Duelist, 00 Table of Contents): <{run_url}>"
    assert said[1] == f"Card sync run finished: <{run_url}>" and said[2].startswith("Card data loaded")
    assert all(TOKEN not in s for s in said + _edits())


@pytest.mark.parametrize("full", [False, True])
def test_card_syncs_own_retry_is_followed(github, tmp_path, full):
    """Main moved during the run: it ends green with its data dropped and dispatches its retry, which is followed."""
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    _dispatched_run(github, 41, _run(41, "in_progress"), _run(41))
    retry = dict(login=cs.BOT_LOGIN, title=cs.RETRY_NAME)
    github.runs = [_run(42, "queued", **retry), _run(41)]
    github.states[42] = [_run(42, "in_progress", **retry), _run(42, **retry)]
    cs.update(ME, force_update=full, cards="" if full else "A")
    said, url = _said(), "https://github.com/MichaelJSr/TTSCardMaker/actions/runs/"
    assert said[1] == (f"The card repo changed during run <{url}41>, so Card sync dropped its data and started again: "
                       f"<{url}42>" + (" (without the full re-parse: ask again once it is done)." if full else "."))
    assert said[2] == f"Card sync run finished: <{url}42>" and len(github.posts) == 1


def test_a_card_guards_dispatch_is_not_a_retry(github, tmp_path):
    """card-guards' card-sync dispatch (GITHUB_TOKEN, named "Card sync") during the run: the run's data stands."""
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    _dispatched_run(github, 41, _run(41, "in_progress"), _run(41))
    github.runs = [_run(42, "queued", login=cs.BOT_LOGIN), _run(41)]
    cs.update(ME, force_update=True)
    assert _said()[1] == "Card sync run finished: <https://github.com/MichaelJSr/TTSCardMaker/actions/runs/41>"


def test_a_reminders_run_posts_no_progress_edits(github, tmp_path):
    """A reminder has no reply to edit (each edit would be a new channel message): only its messages."""
    from commands.analytics import _ReminderUser
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    _dispatched_run(github, 43, _run(43, "queued"), _run(43, "in_progress"), _run(43))
    cs.update(SimpleNamespace(user=_ReminderUser(1, "owner")), cards="A")
    assert _edits() == [] and "Card sync run finished: <https://github.com/MichaelJSr/TTSCardMaker/actions/runs/43>" in _said()


def test_full_dispatch_without_run_details_finds_its_run(github, tmp_path):
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    github.runs = [_run(7, event="schedule")]
    github.post_replies.append(Reply(422, body={"message": "Unexpected inputs provided: return_run_details"}))
    github.post_replies.append(Reply(204))
    github.on_post = lambda n: github.runs.insert(0, _run(50, "queued")) if n == 2 else None
    github.states[50] = [_run(50, conclusion="failure")]
    cs.update(ME, force_update=True)
    assert [p.get("return_run_details") for p in github.posts] == [True, None]
    assert github.posts[1]["inputs"] == {"dry_run": "false", "full": "true"}
    assert "Card sync run ended (failure): <https://github.com/MichaelJSr/TTSCardMaker/actions/runs/50>; the bot keeps" in _said()[1]


def test_a_run_replaced_by_a_scheduled_one_is_dispatched_again(github, tmp_path):
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    _dispatched_run(github, 60, _run(60, conclusion="cancelled"))
    _dispatched_run(github, 62, _run(62))
    github.runs = [_run(61, "pending", event="schedule"), _run(60, conclusion="cancelled")]
    cs.update(ME, cards="A")
    assert len(github.posts) == 2
    assert "replaced by a scheduled card sync before it started; starting it again." in " ".join(_said())
    assert "Card sync run finished: <https://github.com/MichaelJSr/TTSCardMaker/actions/runs/62>" in _said()


@pytest.mark.parametrize("newer, words", [
    ([], "was cancelled."),
    ([_run(71, "pending", login="someone")], "replaced by a newer request before it started"),
])
def test_a_cancelled_run_is_said_when_running_it_again_would_cancel_another(github, tmp_path, newer, words):
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    _dispatched_run(github, 70, _run(70, conclusion="cancelled"))
    github.runs = newer + [_run(70, conclusion="cancelled")]
    cs.update(ME, cards="A")
    assert len(github.posts) == 1 and any(words in s for s in _said())
    assert any(s.startswith("Card data") for s in _said())          # the sync still runs


def test_a_run_cancelled_by_hand_while_running_is_not_dispatched_again(github, tmp_path):
    """Seen running, then cancelled, with a newer scheduled run going: cancelled by hand, so said, not run again."""
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    _dispatched_run(github, 75, _run(75, "queued"), _run(75, "in_progress"), _run(75, conclusion="cancelled"))
    github.runs = [_run(76, "in_progress", event="schedule"), _run(75, conclusion="cancelled")]
    cs.update(ME, cards="A")
    assert len(github.posts) == 1 and "Card sync run <https://github.com/MichaelJSr/TTSCardMaker/actions/runs/75> " \
        "was cancelled." in _said()


def test_cancelled_twice_is_said(github, tmp_path):
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    _dispatched_run(github, 80, _run(80, conclusion="cancelled"))
    _dispatched_run(github, 82, _run(82, conclusion="cancelled"))
    github.runs = [_run(81, "pending", event="schedule")]
    cs.update(ME, cards="A")
    assert len(github.posts) == 2 and "was cancelled again" in " ".join(_said())


def test_without_a_token_the_link_and_the_gh_command(github, tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "ACTIONS_TOKEN", "")
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    cs.update(ME, force_update=True)
    assert github.posts == []
    assert ("`gh workflow run card-sync.yml -R MichaelJSr/TTSCardMaker -f full=true -f dry_run=false`" in _said()[0]
            and "<https://github.com/MichaelJSr/TTSCardMaker/actions/workflows/card-sync.yml>" in _said()[0])
    assert _said()[1].startswith("Card data loaded")


def test_a_refused_token_is_said_never_shown(github, tmp_path):
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    github.post_replies.append(Reply(403, body={"message": "Resource not accessible by personal access token"}))
    cs.update(ME, cards="A")
    assert "GitHub refused ACTIONS_TOKEN (HTTP 403)" in _said()[0] and all(TOKEN not in s for s in _said())


def test_a_dispatch_without_an_answer_may_have_started(github, tmp_path):
    from requests.exceptions import ReadTimeout
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    github.post_replies.append(ReadTimeout("read timed out"))
    cs.update(ME, cards="A")
    assert "the run may have started: look on <" in _said()[0] and "try again" not in _said()[0]


@pytest.mark.parametrize("reply", [Reply(403, body={"message": "You have exceeded a secondary rate limit."}),
                                   Reply(403, body={"message": "x"}, headers={"X-RateLimit-Remaining": "0"}),
                                   Reply(429, body={"message": "x"}, headers={"Retry-After": "60"})])
def test_a_rate_limit_is_not_a_refused_token(github, tmp_path, reply):
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    github.post_replies.append(reply)
    cs.update(ME, cards="A")
    assert "rate limit" in _said()[0] and "refused ACTIONS_TOKEN" not in _said()[0]


def test_an_expired_token_is_said_at_the_run_list(github, tmp_path):
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    github.runs_reply = Reply(401, body={"message": "Bad credentials"})
    cs.update(ME, cards="A")
    assert "GitHub refused ACTIONS_TOKEN (HTTP 401)" in _said()[0] and github.posts == []
    assert all(TOKEN not in s for s in _said())


def test_a_run_past_the_wait_is_left_going(github, tmp_path, monkeypatch):
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    _dispatched_run(github, 90, _run(90, "queued"), _run(90, "in_progress"))
    clock = iter([0, 0, 0, 900, 900, 2000])                           # started; then each look's deadline and edit checks
    monkeypatch.setattr(cs, "monotonic", lambda: next(clock))
    cs.update(ME, cards="A")
    assert "still going after 30 minutes" in " ".join(_said())
    assert _edits() == ["Card sync run <https://github.com/MichaelJSr/TTSCardMaker/actions/runs/90>: waiting for its turn"]


def test_one_followed_run_at_a_time(github, tmp_path):
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    assert cs._run_lock.acquire(blocking=False)
    try:
        cs.update(ME, cards="A")
    finally:
        cs._run_lock.release()
    assert github.posts == [] and "already being followed" in _said()[0]


# ========================
# /fonts from the card data
# ========================
def test_fonts_from_the_card_data(github, tmp_path, monkeypatch):
    import commands.font_check as fc
    details = {"cards": {
        "b-A": {"fonts": {"Arial": {"layers": ["Text/Name"], "supplied": True},
                          "Comic-Bold": {"layers": ["Text/Ability", "Text/Ability 2"], "supplied": False}}},
        "b-B": {"fonts": {"Arial": {"layers": ["Text/Name"], "supplied": None}}}}}
    cards = {"A": _card("A"), "B": _card("B"), "A2": _card("A2", path="Creatures/A.psd", blob_sha="b-A", duplicate_of="A"),
             "C": _card("C"), "M": _card("M", path="MDW/Template.psd")}
    github.publish(cards, defaultdict(list), tmp_path, details=json.dumps(details))
    cs.sync(ME)
    sent = []
    monkeypatch.setattr(utils, "queue_long_message", lambda interaction, text, **kw: sent.append(text))
    monkeypatch.setattr(fc, "repo_fonts", lambda *a, **k: pytest.fail("no PSD or font is downloaded after the cutover"))
    fc.check_repo_fonts("interaction")
    lines = sent[0].splitlines()
    assert lines[0].startswith("From the card data (built ") and lines[0].endswith("): 2 card PSDs.")
    assert lines[1] == "**Comic-Bold**: A (Ability, Ability 2)"
    assert "Creatures/B.psd: fonts not judged" in lines[2] and "Creatures/C.psd: no PSD details" in lines[2]


# ========================
# The switch
# ========================
def test_update_stats_follows_the_switch(monkeypatch):
    import commands.repo_traverser as rt
    calls = []
    monkeypatch.setattr(cs, "update", lambda *a, **k: calls.append(a) or ["p"])
    monkeypatch.setattr(rt, "_update_stats", lambda *a: ["legacy"])
    monkeypatch.setattr(global_config, "CARD_SYNC", False)
    assert rt.update_stats(None) == ["legacy"] and calls == []
    monkeypatch.setattr(global_config, "CARD_SYNC", True)
    assert rt.update_stats(None, True, False, True, "A") == ["p"] and calls == [(None, True, False, True, "A")]


def test_startup_follows_the_switch(monkeypatch):
    import commands.psd_analyzer as pa
    calls = []
    monkeypatch.setattr(cm.stats_db, "load", lambda: calls.append("pickles"))
    monkeypatch.setattr(cs, "startup", lambda: calls.append("card data"))
    monkeypatch.setattr(pa.card_repo, "prep_dataframes", lambda: None)
    monkeypatch.setattr(global_config, "CARD_SYNC", False)
    pa.init_psd()
    monkeypatch.setattr(global_config, "CARD_SYNC", True)
    pa.init_psd()
    assert calls == ["pickles", "card data"]


def _send_log():
    sent = []

    async def send(content):
        sent.append(content)
    return sent, SimpleNamespace(followup=SimpleNamespace(send=send))


def test_push_after_the_cutover_says_what_replaced_it(monkeypatch):
    import commands.update_bot as ub
    monkeypatch.setattr(ub, "_git", lambda: pytest.fail("no git after the cutover"))
    monkeypatch.setattr(global_config, "CARD_SYNC", True)
    sent, interaction = _send_log()
    asyncio.run(ub.git_push(interaction))
    assert sent == [ub.PUSH_RETIRED] and "Card sync Action" in sent[0] and "card-sync.yml" in sent[0]


@pytest.mark.parametrize("switched", [False, True])
def test_update_pushes_only_before_the_cutover(monkeypatch, switched):
    import commands.update_bot as ub
    done = []

    async def push(interaction):
        done.append("push")

    async def pull(interaction):
        done.append("pull")
    monkeypatch.setattr(ub, "git_push", push)
    monkeypatch.setattr(ub, "git_pull", pull)
    monkeypatch.setattr(ub, "restart_bot", lambda: done.append("restart"))
    monkeypatch.setattr(global_config, "CARD_SYNC", switched)
    sent, interaction = _send_log()
    asyncio.run(ub.update_bot(interaction))
    assert done == (["pull", "restart"] if switched else ["push", "pull", "restart"])
    assert any(s.startswith("Pushing ") for s in sent) is not switched


@pytest.mark.parametrize("switched", [False, True])
def test_pull_conflict_keeps_the_pickles_only_before_the_cutover(monkeypatch, switched):
    """A conflicted merge: before the cutover ours for the pickles and the rescue; after, theirs for every file."""
    import commands.update_bot as ub
    from git import GitCommandError
    log = []

    class FakeGit:
        def fetch(self, **kw):
            pass

        def pull(self, *a, **kw):
            if "--no-edit" in a:
                raise GitCommandError("pull", 1, b"CONFLICT (content)")
            return "Already up to date."

        def checkout(self, side, path):
            log.append((side, path))

        def add(self, path):
            pass

        def commit(self, *a):
            log.append(("commit", a[1]))

    class FakeRepo:
        head = SimpleNamespace(commit=SimpleNamespace(hexsha="local"))
        refs = {"origin/main": SimpleNamespace(commit=SimpleNamespace(hexsha="remote"))}
        index = SimpleNamespace(unmerged_blobs=lambda: {("stats.pkl", 0): None, ("main.py", 0): None})

        def merge_base(self, a, b):
            return [SimpleNamespace(hexsha="base")]

    async def rescue(interaction):
        log.append("rescue")
    monkeypatch.setattr(ub, "_git", FakeGit)
    monkeypatch.setattr(ub, "_repo", FakeRepo)
    monkeypatch.setattr(ub, "_rescue_after_merge", rescue)
    monkeypatch.setattr(global_config, "CARD_SYNC", switched)
    sent, interaction = _send_log()
    asyncio.run(ub.git_pull(interaction))
    if switched:
        assert log == [("--theirs", "stats.pkl"), ("--theirs", "main.py"),
                       ("commit", "AUTO-RESOLVE: Merge conflict resolved (accepted remote)")]
    else:
        assert log == [("--ours", "stats.pkl"), ("--theirs", "main.py"),
                       ("commit", "AUTO-RESOLVE: Merge conflict resolved (kept local pickles, accepted remote code)"), "rescue"]


def test_discord_edits_save_the_pickles_only_before_the_cutover(monkeypatch):
    import commands.discord_edits as de
    import commands.query_card as qc
    saves = []
    monkeypatch.setattr(cm.stats_db, "stats", {"A": _card("A")})
    monkeypatch.setattr(cm.stats_db, "save", lambda: saves.append(1))
    monkeypatch.setattr(qc.card_repo, "prep_dataframes", lambda: None)
    monkeypatch.setattr(qc.card_repo, "git_files", {})
    for key in ("_edits", "_edit_names", "_moved_hint"):              # refresh keeps the parsed file there
        monkeypatch.setattr(qc.card_repo, key, getattr(qc.card_repo, key))
    edits = json.dumps({"cards": {"A": {"note": "n1"}}})
    monkeypatch.setattr(global_config, "CARD_SYNC", False)
    assert de.refresh(["A"], edits) == ["A"] and saves == [1]
    monkeypatch.setattr(global_config, "CARD_SYNC", True)
    assert de.refresh(["A"], json.dumps({"cards": {"A": {"note": "n2"}}})) == ["A"] and saves == [1]
    assert cm.stats_db.stats["A"].note == "n2"


def test_help_follows_the_switch(monkeypatch):
    import commands.help as help_module
    sent = []
    monkeypatch.setattr(help_module, "queue_long_message", lambda interaction, text: sent.append(text))
    monkeypatch.setattr(global_config, "CARD_SYNC", False)
    help_module.print_help(None, "stats")
    today = "```Available commands:\n\n" + "".join(f"{k}\n{v}\n\n" for k, v in help_module.stats_commands.items()) + "```"
    assert sent[0] == today
    monkeypatch.setattr(global_config, "CARD_SYNC", True)
    for kind in ("general", "card", "stats"):
        help_module.print_help(None, kind)
    after = "".join(sent[1:])
    assert "Loads the card data the Card sync Action built" in after and "Retired:" in after
    assert "from the card data" in after and "keeping the local pickles" not in after
    assert "- force_update: Admin only." in after and "- cards: Admin only." in after


def test_slash_texts_in_both_states():
    """main.py's command and option descriptions with the switch either way: within Discord's 100 characters, and
    on today's texts while it is off (imported fresh in a subprocess)."""
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    script = (
        "import json, sys, types\n"
        "fake = types.ModuleType('config'); fake.TOKEN = fake.GIT_TOKEN = ''; fake.ADMINS = []; fake.browser_cookies = False\n"
        "sys.modules.setdefault('config', fake)\n"
        "import global_config; global_config.CARD_SYNC = sys.argv[1] == 'on'\n"
        "import main\n"
        "out = {c.name: [c.description] + [p.description for p in c.parameters] for c in main.tree.get_commands()}\n"
        "print(json.dumps(out))\n")
    texts = {}
    for state in ("off", "on"):
        done = subprocess.run([sys.executable, "-c", script, state], cwd=root, capture_output=True, text=True, check=True)
        texts[state] = json.loads(done.stdout.strip().splitlines()[-1])
    for state, commands in texts.items():
        for name, descriptions in commands.items():
            assert all(len(d) <= 100 for d in descriptions), (state, name)
    changed = sorted(n for n in texts["off"] if texts["off"][n] != texts["on"][n])
    assert changed == ["pull", "push", "update", "update_stats"]
    assert texts["off"]["update_stats"][0] == "Updates the card database from GitHub (re-parses changed PSDs)."
    assert texts["off"]["push"][0].startswith("Commits the pickles")


@pytest.mark.parametrize("switched, options, allowed", [
    (False, {"force_update": True}, True),                            # today: anyone
    (True, {}, True),
    (True, {"force_update": True}, False),
    (True, {"cards": "A"}, False),
])
def test_update_stats_options_are_admin_only_after_the_cutover(monkeypatch, switched, options, allowed):
    import main
    ran = []

    async def record(*a, **k):
        ran.append(a)
    monkeypatch.setattr(main, "update_stats", record)
    monkeypatch.setattr(utils, "ADMINS", [1])
    monkeypatch.setattr(global_config, "CARD_SYNC", switched)
    sent = []

    async def send_message(content=None, **kwargs):
        sent.append(content)

    async def defer(**kwargs):
        pass
    other = SimpleNamespace(user=SimpleNamespace(id=2), response=SimpleNamespace(send_message=send_message, defer=defer))
    callback = {c.name: c for c in main.tree.get_commands()}["update_stats"].callback
    asyncio.run(callback(other, **options))
    assert bool(ran) is allowed and (sent == [] if allowed else sent == ["You do not have permission."])
    if switched and not allowed:                                      # an admin may
        admin = SimpleNamespace(user=SimpleNamespace(id=1), response=SimpleNamespace(send_message=send_message, defer=defer))
        asyncio.run(callback(admin, **options))
        assert ran


# ========================
# Loads that read psd_details.json by its hash only; the website build after a /push
# ========================
def test_the_bots_loads_never_parse_the_details(github, tmp_path, monkeypatch):
    """load_local, a sync and card_db.sync's check read psd_details.json by its hash only (the bot keeps no details);
    a damaged details file still makes the copy a cut-off one, reloaded by the next sync."""
    details = '{"version": 1, "cards": {\n"b0": {"marker-details": 1}\n}}\n'
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path, details=details)
    cdb.sync("card_data", REPO, "data")
    seen, real = [], cdb.loads
    monkeypatch.setattr(cdb, "loads", lambda text, *a, **k: seen.append("marker-details" in text) or real(text, *a, **k))
    assert cs.load_local("card_data") and set(cm.stats_db.stats) == {"A"}
    cs.sync(ME)                                                       # unchanged: a 304
    github.publish({"A": _card("A"), "B": _card("B")}, defaultdict(list), tmp_path, details=details)
    cs.sync(ME)                                                       # moved: reloaded
    assert set(cm.stats_db.stats) == {"A", "B"} and seen and not any(seen)
    (tmp_path / "card_data" / cdb.DETAILS_JSON).write_text(details.replace("1}", "2}"), encoding="utf-8")
    assert cs.load_local("card_data") and cs._loaded["commit"] is None
    assert cdb.check_db("card_data", parse=False) == cdb.check_db("card_data") == [f"{cdb.DETAILS_JSON}: sha256 differs from the manifest"]
    assert cdb.read_db("card_data", details=False)[2] == {} and cdb.read_db("card_data")[2] == {"b0": {"marker-details": 2}}


def test_dispatch_site(monkeypatch):
    """The website build (site.yml) on the card repo's main with ACTIONS_TOKEN; a refusal or no answer is one line,
    never raised; no token, no request."""
    posts, replies = [], []

    def post(url, json=None, headers=None, timeout=None):
        posts.append((url, json, headers["Authorization"]))
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply
    monkeypatch.setattr(cs, "session", lambda: SimpleNamespace(post=post))
    monkeypatch.setattr(cs, "ACTIONS_TOKEN", TOKEN)
    replies[:] = [Reply(204), Reply(403), OSError("down")]
    assert cs.dispatch_site() == "Website build started."
    assert posts == [(f"https://api.github.com/repos/{REPO}/actions/workflows/site.yml/dispatches", {"ref": "main"}, f"Bearer {TOKEN}")]
    assert "HTTP 403" in cs.dispatch_site() and TOKEN not in cs.dispatch_site()
    monkeypatch.setattr(cs, "ACTIONS_TOKEN", "")
    posts.clear()
    assert "no ACTIONS_TOKEN" in cs.dispatch_site() and posts == []


class _PushGit:
    """update_bot's _git() for git_push: the pickles committed, the push's changed files."""
    def __init__(self, changed):
        self.changed, self.pushed = changed, False

    def add(self, *a): pass
    def commit(self, *a): pass
    def pull(self, *a, **k): pass
    def rev_list(self, *a): return "1"
    def diff(self, *a): return "\n".join(self.changed)
    def push(self, *a, **k): self.pushed = True


@pytest.mark.parametrize("changed, built", [(["stats.pkl", "psd_details.json"], True), (["commands/help.py"], False)])
def test_a_pickle_push_starts_the_website_build(monkeypatch, changed, built):
    """/push dispatches site.yml after pushing a file the website reads, after "Successfully pushed!"; a code-only push
    starts nothing."""
    import commands.update_bot as ub
    monkeypatch.setattr(global_config, "CARD_SYNC", False)
    git, sent, started = _PushGit(changed), [], []
    monkeypatch.setattr(ub, "_git", lambda: git)
    monkeypatch.setattr(cs, "dispatch_site", lambda: started.append(len(sent)) or "Website build started.")

    async def send(text, **kw):
        sent.append(text)
    asyncio.run(ub.git_push(SimpleNamespace(followup=SimpleNamespace(send=send))))
    assert git.pushed and sent == ["Successfully pushed!"] and started == ([1] if built else [])
