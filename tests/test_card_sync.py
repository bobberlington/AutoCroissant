"""The bot's card data: card_db.sync, card_sync's /update_stats (sync, report, Card sync runs through ACTIONS_TOKEN),
/fonts from the card data, and /update, /pull, /help and the slash texts without the pickles."""
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
        self.looks: defaultdict = defaultdict(int)  # (run or jobs, run id) -> looks so far (each kind walks the states)
        self.posts: list = []
        self.post_replies: list = []            # Reply per POST, in order
        self.on_post = None                     # called after each POST (a run appears)
        self.compare = {"ahead_by": 0, "files": []}
        self.runs_reply = None                  # a Reply for the run list instead of self.runs
        self.jobs: dict = {}                    # run id -> the jobs lists GET returns, one per look (the last repeats);
                                                # else a dispatched run's Card data job follows its states

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
            run_id = int(url.rsplit("/", 2)[1])
            self.calls.append(("jobs", run_id))
            if run_id in self.jobs:
                return Reply(200, body={"jobs": self.look("jobs", run_id, self.jobs[run_id])})
            if run_id in self.states:
                return Reply(200, body={"jobs": [_job(cs.DATA_JOB, self.look("jobs", run_id, self.states[run_id]))]})
            return Reply(200, body={"jobs": []})
        if "/actions/runs/" in url:
            run_id = int(url.rsplit("/", 1)[1])
            self.calls.append(("run", run_id))
            return Reply(200, body=self.look("run", run_id, self.states[run_id]))
        if "/compare/" in url:
            self.calls.append(("compare", url.rsplit("/compare/", 1)[1]))
            return Reply(200, body=self.compare)
        raise AssertionError(f"unexpected GET {url}")

    def look(self, kind: str, run_id: int, answers: list):
        """The next of answers for this kind of look at the run (the last repeats)."""
        n = self.looks[(kind, run_id)]
        self.looks[(kind, run_id)] += 1
        return answers[min(n, len(answers) - 1)]

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


def _job(name, state=None, status="completed", conclusion="success"):
    """A run's job; from a run state (its status and conclusion), steps as GitHub lists them (none before it starts)."""
    if state is not None:
        status, conclusion = state["status"], state["conclusion"]
    steps = [] if status in cs.WAITING or conclusion == "cancelled" else [
        {"name": "Gate", "status": "completed"}, {"name": "Build", "status": "in_progress" if status == "in_progress" else "completed"}]
    return {"name": name, "status": status, "conclusion": conclusion if status == "completed" else None, "steps": steps}


@pytest.fixture
def github(monkeypatch, tmp_path):
    """cwd a temp folder (card_data/ there), GitHub faked, stats_db and the queues emptied."""
    fake = FakeGitHub()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ga, "get", fake.get)
    monkeypatch.setattr(ga, "sleep", lambda s: None)
    monkeypatch.setattr(cs, "session", lambda: SimpleNamespace(post=fake.post))
    monkeypatch.setattr(cs, "sleep", lambda s: None)
    monkeypatch.setattr(cs, "ACTIONS_TOKEN", TOKEN)
    monkeypatch.setattr(cm.stats_db, "stats", {})
    monkeypatch.setattr(cm.stats_db, "old_stats", defaultdict(list))
    monkeypatch.setitem(cs._loaded, "commit", None)
    monkeypatch.setitem(cs._checked, "at", None)
    monkeypatch.setattr(cs, "_unreported", {})
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
    cs.startup()
    assert cm.stats_db.stats == {}                                    # no cards until a sync works


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
    github.runs = [_run(9, "in_progress", event="push"), _run(7, conclusion="success", event="schedule"),
                   _run(8, conclusion="failure", event="schedule")]
    github.jobs = {9: [[_job(cs.DATA_JOB), _job("Quiet window", status="waiting"), _job(cs.MAIN_JOB, status="pending")]],
                   7: [[_job(cs.DATA_JOB), _job(cs.MAIN_JOB, conclusion="skipped")]],     # the cron's data alone
                   8: [[_job(cs.DATA_JOB), _job(cs.MAIN_JOB, conclusion="failure")]]}
    cs.sync(ME)
    text = _said()[0]
    assert "Re-parse in progress: 2 cards still to re-parse" in text
    assert "Could not read 1 PSD (Bad Card); its card keeps its last version" in text
    assert "the card sync under way picks them up (<https://github.com/MichaelJSr/TTSCardMaker/actions/runs/9>)" in text
    assert "The last bake and save failed: <https://github.com/MichaelJSr/TTSCardMaker/actions/runs/8>" in text
    assert "card data build failed" not in text and github.kinds("jobs") == [9, 7, 8]
    utils.dispatch_queue.clear()
    github.jobs[9] = [[_job(cs.DATA_JOB, conclusion="failure"), _job(cs.MAIN_JOB, status="pending")]]
    cs.sync(ME)
    assert "The last card data build failed: <https://github.com/MichaelJSr/TTSCardMaker/actions/runs/9>" in _said()[0]
    assert ("compare", "c0ffee0123456789...main") in github.calls
    utils.dispatch_queue.clear()
    github.runs = [_run(8, conclusion="success", event="schedule")]
    github.jobs = {}                                                  # a run before the split: its conclusion
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


def test_update_stats_starts_no_card_sync(github, tmp_path):
    """Card sync starts itself (a card push's timer): /update_stats and its reminder only sync, even with changes waiting."""
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    github.compare = {"ahead_by": 1, "files": [{"filename": "Items/New.psd"}]}
    github.runs = [_run(8, event="push")]
    cs.update(ME)
    assert github.posts == [] and ("the next card sync picks them up within minutes of the push (website edits are baked "
                                   "into the PSDs 30 minutes after the first card push of a burst)" in _said()[0])


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
    cs.startup()
    assert set(cm.stats_db.stats) == {"A"} and _queued() == []


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


def test_startup_without_data_holds_no_cards(github):
    cs.startup()
    assert cm.stats_db.stats == {} and cs._loaded["commit"] is None


# ========================
# On use: card commands look for new data
# ========================
def test_on_use_looks_at_most_every_two_minutes(github, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(cs, "monotonic", lambda: clock[0])
    assert not cs.on_use("play") and _queued() == []                 # not a card command
    assert cs.on_use("query") and _queued() == ["check"]
    assert not cs.on_use() and not cs.on_use("query_ability")        # within the 2 minutes (autocomplete too)
    clock[0] += cs.REFRESH_SECONDS
    assert cs.on_use() and _queued() == ["check", "check"]


def test_a_sync_counts_as_the_look(github, tmp_path):
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    cs.sync(ME)                                                       # /update_stats, or startup
    utils.command_queue.clear()
    assert not cs.on_use("query") and _queued() == []


def test_check_reloads_quietly_then_refreshes(github, tmp_path, monkeypatch, capsys):
    import commands.query_card as qc
    looked = []
    monkeypatch.setattr(qc.card_repo, "_populate_site_ids", lambda: looked.append(1))
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    assert cs.check() and set(cm.stats_db.stats) == {"A"} and cs._loaded["commit"] == github.tip
    assert _said() == [] and "Card data loaded: built " in capsys.readouterr().out
    assert _queued() == ["refresh", "_refresh_cards"] and looked == []
    utils.command_queue.clear()
    calls = len(github.calls)
    assert not cs.check() and _queued() == [] and looked == [1]       # unchanged: one 304, the site's ids
    assert github.calls[calls:] == [("commit", f'"e-{github.tip}"')]
    github.publish({"A": _card("A", ability="Heal 1."), "B": _card("B")}, defaultdict(list), tmp_path)
    assert cs.check() and "1 changed, 1 new; reloaded." in capsys.readouterr().out and _said() == []


def test_a_failed_check_keeps_the_data_and_says_nothing(github, tmp_path, monkeypatch, capsys):
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    cs.check()
    held = cm.stats_db.stats
    utils.command_queue.clear()
    monkeypatch.setattr(cdb, "sync", lambda *a: (_ for _ in ()).throw(KeyError("boom")))
    assert not cs.check() and cm.stats_db.stats is held and _said() == [] and _queued() == []
    assert "Card data check failed (KeyError" in capsys.readouterr().out
    assert not cs._sync_lock.locked()


def test_check_skips_while_a_sync_or_a_followed_run_runs(github, tmp_path):
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    for lock in (cs._sync_lock, cs._run_lock):
        with lock:
            assert not cs.check()
    assert github.calls == [] and cm.stats_db.stats == {}


def test_the_next_sync_reports_what_checks_loaded(github, tmp_path):
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    cs.sync(ME)
    utils.dispatch_queue.clear()
    github.publish({"A": _card("A"), "C": _card("C")}, defaultdict(list), tmp_path,
                   problems=[{"card": "C", "path": "Creatures/C.psd", "notes": ["HP NOT FOUND"]}])
    assert cs.check() and _said() == []
    github.publish({"A": _card("A", ability="Heal 1."), "C": _card("C")}, defaultdict(list), tmp_path)
    assert cs.check()
    problems = cs.sync(ME)                                            # /update_stats, or the reminder's
    assert ": 1 changed, 1 new; reloaded." in _said()[0]
    assert len(problems) == 1 and "HP NOT FOUND" in problems[0] and "HP NOT FOUND" in _said()[1]
    utils.dispatch_queue.clear()
    assert cs.sync(ME) == [] and _said()[0].startswith("Card data is already current")


def test_the_start_counts_as_a_look(monkeypatch):
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("card_sync_fresh", cs.__file__)
    fresh = module_from_spec(spec)                                    # a new process's state, cs untouched
    spec.loader.exec_module(fresh)
    assert not fresh.on_use("query")                                  # no look before startup's sync


def test_card_commands_and_autocomplete_look(github, monkeypatch):
    import main
    from discord import InteractionType
    names = {cmd.name for cmd in main.tree.get_commands()}
    assert cs.READERS <= names                                        # no stale or misspelt name
    used = []
    monkeypatch.setattr(main, "card_data_used", lambda command=None: used.append(command))
    for kind, data in ((InteractionType.autocomplete, {"name": "query"}),
                       (InteractionType.application_command, {"name": "query", "options": []}),
                       (InteractionType.component, {})):
        asyncio.run(main.on_interaction(SimpleNamespace(type=kind, data=data, user=None, channel_id=1)))
    assert used == [None, "query"]


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
    assert said[1] == f"Card sync run's card data finished: <{run_url}>" and said[2].startswith("Card data loaded")
    assert all(TOKEN not in s for s in said + _edits())


def test_card_syncs_own_retry_is_not_followed(github, tmp_path):
    """Main moved during the run's bake: its retry redoes only the bake and the save; the run's data stands."""
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    _dispatched_run(github, 41, _run(41, "in_progress"), _run(41))
    retry = dict(login=cs.BOT_LOGIN, title=cs.RETRY_NAME)
    github.runs = [_run(42, "queued", **retry), _run(41)]
    cs.update(ME, cards="A")
    assert _said()[1] == "Card sync run's card data finished: <https://github.com/MichaelJSr/TTSCardMaker/actions/runs/41>"
    assert len(github.posts) == 1


def test_a_card_guards_dispatch_is_not_a_retry(github, tmp_path):
    """card-guards' card-sync dispatch (GITHUB_TOKEN, named "Card sync") during the run: the run's data stands."""
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    _dispatched_run(github, 41, _run(41, "in_progress"), _run(41))
    github.runs = [_run(42, "queued", login=cs.BOT_LOGIN), _run(41)]
    cs.update(ME, force_update=True)
    assert _said()[1] == "Card sync run's card data finished: <https://github.com/MichaelJSr/TTSCardMaker/actions/runs/41>"


def test_a_reminders_run_posts_no_progress_edits(github, tmp_path):
    """A reminder has no reply to edit (each edit would be a new channel message): only its messages."""
    from commands.analytics import _ReminderUser
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    _dispatched_run(github, 43, _run(43, "queued"), _run(43, "in_progress"), _run(43))
    cs.update(SimpleNamespace(user=_ReminderUser(1, "owner")), cards="A")
    assert _edits() == [] and "Card sync run's card data finished: <https://github.com/MichaelJSr/TTSCardMaker/actions/runs/43>" in _said()


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
    assert "Card sync run's card data ended (failure): <https://github.com/MichaelJSr/TTSCardMaker/actions/runs/50>; the bot keeps" in _said()[1]


def test_a_run_replaced_by_a_scheduled_one_is_dispatched_again(github, tmp_path):
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    _dispatched_run(github, 60, _run(60, conclusion="cancelled"))
    _dispatched_run(github, 62, _run(62))
    github.runs = [_run(61, "pending", event="schedule"), _run(60, conclusion="cancelled")]
    cs.update(ME, cards="A")
    assert len(github.posts) == 2
    assert "replaced by a newer card sync's before it started; starting it again." in " ".join(_said())
    assert "Card sync run's card data finished: <https://github.com/MichaelJSr/TTSCardMaker/actions/runs/62>" in _said()


def test_the_reply_comes_when_the_card_data_lands_not_the_bake(github, tmp_path):
    """The run's Card data job ended; its Bake and save still runs: the reply and the sync come now."""
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    _dispatched_run(github, 44, _run(44, "in_progress"))
    github.jobs[44] = [[_job(cs.DATA_JOB, status="in_progress"), _job(cs.MAIN_JOB, status="in_progress")],
                       [_job(cs.DATA_JOB), _job(cs.MAIN_JOB, status="in_progress")]]
    cs.update(ME, cards="A")
    assert _said()[1] == "Card sync run's card data finished: <https://github.com/MichaelJSr/TTSCardMaker/actions/runs/44>"
    assert _said()[2].startswith("Card data loaded") and github.kinds("run") == [44]       # the dispatch's look only


def test_a_data_job_replaced_while_the_bake_runs_is_dispatched_again(github, tmp_path):
    """The run is in progress (its Bake and save started at once), its Card data job was replaced by a push's: again."""
    github.publish({"A": _card("A")}, defaultdict(list), tmp_path)
    _dispatched_run(github, 64, _run(64, "in_progress"))
    github.jobs[64] = [[_job(cs.DATA_JOB, status="pending"), _job(cs.MAIN_JOB, status="in_progress")],
                       [_job(cs.DATA_JOB, conclusion="cancelled"), _job(cs.MAIN_JOB, status="in_progress")]]
    _dispatched_run(github, 66, _run(66))
    github.runs = [_run(65, "waiting", event="push"), _run(64, "in_progress")]
    cs.update(ME, cards="A")
    assert len(github.posts) == 2 and "Card sync run's card data finished: <https://github.com/MichaelJSr/TTSCardMaker/" \
        "actions/runs/66>" in _said()


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
    clock = iter([0, 0, 0, 900, 900, 2000, 2000])                     # started; each look's deadline and edit checks; the sync
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
def test_fonts_from_the_card_data(github, tmp_path):
    details = {"cards": {
        "b-A": {"fonts": {"Arial": {"layers": ["Text/Name"], "supplied": True},
                          "Comic-Bold": {"layers": ["Text/Ability", "Text/Ability 2"], "supplied": False}}},
        "b-B": {"fonts": {"Arial": {"layers": ["Text/Name"], "supplied": None}}}}}
    cards = {"A": _card("A"), "B": _card("B"), "A2": _card("A2", path="Creatures/A.psd", blob_sha="b-A", duplicate_of="A"),
             "C": _card("C"), "M": _card("M", path="MDW/Template.psd")}
    github.publish(cards, defaultdict(list), tmp_path, details=json.dumps(details))
    cs.sync(ME)
    lines = cs.fonts_text().splitlines()
    assert lines[0].startswith("From the card data (built ") and lines[0].endswith("): 2 card PSDs.")
    assert lines[1] == "**Comic-Bold**: A (Ability, Ability 2)"
    assert "Creatures/B.psd: fonts not judged" in lines[2] and "Creatures/C.psd: no PSD details" in lines[2]


# ========================
# /update, /pull, /help, the slash texts
# ========================
def _send_log():
    sent = []

    async def send(content):
        sent.append(content)
    return sent, SimpleNamespace(followup=SimpleNamespace(send=send))


def test_update_pulls_then_restarts(monkeypatch):
    import commands.update_bot as ub
    done = []

    async def pull(interaction):
        done.append("pull")
    monkeypatch.setattr(ub, "git_pull", pull)
    monkeypatch.setattr(ub, "restart_bot", lambda: done.append("restart"))
    sent, interaction = _send_log()
    asyncio.run(ub.update_bot(interaction))
    assert done == ["pull", "restart"] and not any("push" in s.lower() for s in sent)


def test_pull_conflict_takes_theirs(monkeypatch):
    """A conflicted merge: theirs for every file."""
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

    monkeypatch.setattr(ub, "_git", FakeGit)
    monkeypatch.setattr(ub, "_repo", FakeRepo)
    sent, interaction = _send_log()
    asyncio.run(ub.git_pull(interaction))
    assert log == [("--theirs", "stats.pkl"), ("--theirs", "main.py"),
                   ("commit", "AUTO-RESOLVE: Merge conflict resolved (accepted remote)")]


def test_help_says_the_card_sync(monkeypatch):
    import commands.help as help_module
    sent = []
    monkeypatch.setattr(help_module, "queue_long_message", lambda interaction, text: sent.append(text))
    for kind in ("general", "card", "stats"):
        help_module.print_help(None, kind)
    text = "".join(sent)
    assert "Loads the card data the Card sync Action built" in text and "/push" not in text and "pickle" not in text
    assert "from the card data" in text and "- force_update: Admin only." in text and "- cards: Admin only." in text


def test_slash_texts_fit():
    """main.py's command and option descriptions: within Discord's 100 characters; no /push."""
    import main
    commands = {c.name: [c.description] + [p.description for p in c.parameters] for c in main.tree.get_commands()}
    assert "push" not in commands
    for name, descriptions in commands.items():
        assert all(len(d) <= 100 for d in descriptions), name
    assert commands["update_stats"][0] == "Loads the card data the Card sync Action built."


@pytest.mark.parametrize("options, allowed", [({}, True), ({"force_update": True}, False), ({"cards": "A"}, False)])
def test_update_stats_options_are_admin_only(monkeypatch, options, allowed):
    import main
    ran = []

    async def record(*a, **k):
        ran.append(a)
    monkeypatch.setattr(main, "update_stats", record)
    monkeypatch.setattr(utils, "ADMINS", [1])
    sent = []

    async def send_message(content=None, **kwargs):
        sent.append(content)

    async def defer(**kwargs):
        pass
    other = SimpleNamespace(user=SimpleNamespace(id=2), response=SimpleNamespace(send_message=send_message, defer=defer))
    callback = {c.name: c for c in main.tree.get_commands()}["update_stats"].callback
    asyncio.run(callback(other, **options))
    assert bool(ran) is allowed and (sent == [] if allowed else sent == ["You do not have permission."])
    if not allowed:                                                   # an admin may
        admin = SimpleNamespace(user=SimpleNamespace(id=1), response=SimpleNamespace(send_message=send_message, defer=defer))
        asyncio.run(callback(admin, **options))
        assert ran


# ========================
# Loads that read psd_details.json by its hash only
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
