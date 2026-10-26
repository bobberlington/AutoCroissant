"""
card_bake: the edit.js editsFile port (today's file round-trips byte-identically; node cross-check), and the driver
against a fake GitHub (tree, blobs, refs, commits: the test_update_stats pattern) for the gates of EDIT_PIPELINE_DESIGN
sections 5 and 11: recent, duplicate, blob, classify, edit-changed, head-moved (409 and 422), one commit per run with a
moved card left out, the schedule gate (pending), the SSH fallback for a token that can't write, --dry-run, --card; the
verdicts on the data branch (CARD_SYNC_VERDICTS.md: the data_branch fixture merges each run's records as Card sync does). A
real bake (a stat, and an ability with an FX layer) needs the fork's renderer; those tests skip under stock psd-tools.
card-sync.yml's Pending edits line runs as written (card_bake.due). Fork as conftest.find_fork finds it; a Python without
requests skips this file:
    ~/.venvs/ac-py314-fork/bin/python -m pytest -q -n auto tests/test_psd_edit.py tests/test_card_bake.py
"""
import contextlib
import json
import re
import shutil
import textwrap
from datetime import datetime, timedelta, timezone
from io import StringIO
from os import environ
from os.path import expanduser
from pathlib import Path
from subprocess import run as run_process
from types import SimpleNamespace

import pytest

pytest.importorskip("requests", reason="card_bake imports requests (absent from this Python)")
import commands.card_bake as cb  # noqa: E402
import commands.git_commit as gc  # noqa: E402
import commands.psd_edit as pe  # noqa: E402
from conftest import (CLONE, Reply, blob_sha_bytes as _sha, fork_reason, story_sync as _story_sync,
                      text_engine as _text_engine)  # noqa: E402
from helpers import ABOUT, AT, DOOMER, EDITS, FREEZER, SNACK, _edits, files, github  # noqa: E402,F401 (fixtures)

GREEN_MAGE = "Creatures/Everhood/4 Stars/Green_Mage.psd"   # fixed 2026-10-23: its stat bars on the canvas
OFF_CANVAS_MAGE = "45cf8746^"   # Green Mage before its bars moved onto the card (they lay below the canvas)
PURE_VESSEL = "Creatures/Hollow Knight/5 Stars/Pure_Vessel.psd"     # its stat masks reach past the canvas
LIVE_EDITS = CLONE / "site" / "Card Edits.json"
EDIT_JS = CLONE / "site" / "assets" / "edit.js"
WORKFLOW = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE))) / ".github" / "workflows" / "card-sync.yml"
fork = pytest.mark.skipif(not _text_engine(), reason="a bake needs the fork's renderer: " + fork_reason())
fork_text = pytest.mark.skipif(not (_text_engine() and _story_sync()),
                               reason="a text bake needs the fork's Txt2 story sync: " + fork_reason())


# ========================
# Fixtures
# ========================
DB: list = []                                   # the test's data branch db/ (data_branch)


@pytest.fixture(autouse=True)
def data_branch(tmp_path, monkeypatch):
    """db/verdicts.json as Card sync keeps it: cb.run and cb.pending read it, and each run's records are merged after it
    (card_db.merge_verdicts: the refusals always, the landed part once its commit lands; never on a dry run)."""
    from commands.card_db import merge_verdicts
    db = tmp_path / "db"
    DB[:] = [db]
    real_run, real_pending = cb.run, cb.pending

    def run(**kwargs):
        text, commits = real_run(**{"verdicts": str(db / "verdicts.json"), **kwargs})
        out = tmp_path / cb.WORK_NAME / cb.VERDICTS_OUT
        if out.exists() and not kwargs.get("dry_run"):
            doc = json.loads(out.read_text(encoding="utf-8"))
            merge_verdicts(str(db), {**doc["cards"], **(doc["landed"] if commits else {})})
        return text, commits
    monkeypatch.setattr(cb, "run", run)
    monkeypatch.setattr(cb, "pending", lambda **kwargs: real_pending(**{"verdicts": str(db / "verdicts.json"), **kwargs}))


def _edits_at(github, sha=None) -> dict:
    return json.loads(github.files_at(sha or github.head)[cb.EDITS_PATH].decode("utf-8"))


def _verdict(name: str):
    """A card's verdict on the data branch, or None."""
    path = DB[0] / "verdicts.json"
    cards = json.loads(path.read_text(encoding="utf-8"))["cards"] if path.exists() else {}
    return cards.get(cb.CardAuthors.key(name))


def _records(tmp_path) -> tuple:
    """The last run's verdicts.json and bake-events.json in the work dir."""
    work = tmp_path / cb.WORK_NAME
    return tuple(json.loads((work / name).read_text(encoding="utf-8")) for name in (cb.VERDICTS_OUT, cb.EVENTS_OUT))


# ========================
# The editsFile port
# ========================
def test_live_edits_file_round_trips_byte_identical():
    """S4's gate: today's site/Card Edits.json through the port, unchanged (read-only)."""
    if not LIVE_EDITS.exists():
        pytest.skip(f"no {LIVE_EDITS}")
    text = LIVE_EDITS.read_text(encoding="utf-8")
    assert cb.dump_edits(json.loads(text)) == text


def test_dump_edits_layout():
    data = {"about": "x", "extra": [1, {"a": None}], "cards": {"b": {"hp": 1, "_by": "w"}, "A": {"note": "é \"q\"  "}}}
    assert cb.dump_edits(data) == ('{\n  "about": "x",\n  "extra": [1,{"a":null}],\n  "cards": {\n'
                                   '    "A": {"note":"é \\"q\\"  "},\n    "b": {"hp":1,"_by":"w"}\n  }\n}\n')
    assert cb.dump_edits({"about": "x", "cards": {}}) == '{\n  "about": "x",\n  "cards": {}\n}\n'


def _node(script_dir, text, ops, at="2026-10-20T00:00:00Z", by="tester") -> str:
    if not EDIT_JS.exists() or shutil.which("node") is None:
        pytest.skip("edit.js or node missing")
    script = script_dir / "edits_file.mjs"
    script.write_text('import { readFileSync } from "node:fs";\n'
                      f'const {{ editsFile }} = await import({json.dumps(EDIT_JS.as_uri())});\n'
                      'const [text, ops, at, by] = JSON.parse(readFileSync(0, "utf-8"));\n'
                      'process.stdout.write(editsFile(text, ops, at, by));\n', encoding="utf-8")
    done = run_process(["node", str(script)], input=json.dumps([text, ops, at, by]), capture_output=True, text=True,
                       encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_dump_edits_matches_edit_js(tmp_path):
    """The sort order and stringify against editsFile itself (node): mixed case, punctuation, digits, accents, unicode."""
    cards = {"Zed": {"hp": 1}, "éclair": {"note": "ñ \"q\" \\ \t   \U0001f600"}, "Bobbery": {"series": "Mario"},
             "Bob-ulk": {"aliases": ["x", "y"], "hidden": True, "creatures": [{"hp": 1, "def": None}]},
             "P.A.N.D.A. Mk A": {"stars": 3, "_psd": {"state": "refused", "for": {"stars": 3}, "psd": "abcdef01", "reason": "x: y"}},
             "Panda": {"dup_of": ""}, "a b": {"note": "n"}, "ab": {"note": "n"}, "PoD12BAD": {"name": "B.A.D."},
             "PoD11ZOMG": {"name": "Z.O.M.G."}, "Mom's Knife": {"spd": 0}, "Moms Knife": {"spd": 1}, "10 Ten": {}, "9 Nine": {"hp": 9}}
    text = json.dumps({"about": ABOUT, "cards": cards}, indent=3, ensure_ascii=True)
    assert cb.dump_edits(json.loads(text)) == _node(tmp_path, text, [])


def test_rewrite_entry_rules():
    verdict = {"state": "refused", "for": {"stars": 2}, "psd": "abcdef01", "reason": "unsupported-card: stars: not yet", "at": AT}
    entry = {"stars": 2, "note": "n", "_by": "w", "_at": AT}
    assert cb.rewrite_entry(entry, (), verdict) == {"stars": 2, "note": "n", "_psd": verdict, "_by": "w", "_at": AT}
    assert cb.rewrite_entry({"hp": 8, "_by": "w", "_at": AT, "_psd": verdict}, ["hp"]) is None       # the entry goes whole
    assert cb.rewrite_entry({"hp": 8, "_by": "w"}, ["hp"], verdict, keep=True) == {"_psd": verdict}   # a flagged bake's verdict stays
    assert cb.rewrite_entry({"name": "X", "hp": 8, "_by": "w"}, ["hp"], None) == {"name": "X", "_by": "w"}   # name stays
    assert cb.rewrite_entry({"_psd": verdict, "hp": 8}, (), None) == {"hp": 8}                       # None removes _psd
    text = cb.dump_edits({"about": ABOUT, "cards": {"Mini_Doomer": {"hp": 8, "note": "n"}, "Other": {"hp": 1}}})
    assert cb.rewrite_edits(text, "mini doomer", ["hp"]) == cb.dump_edits({"about": ABOUT, "cards": {"Mini_Doomer": {"note": "n"}, "Other": {"hp": 1}}})
    assert cb.rewrite_edits(text, "mini doomer", ["spd"]) is None and cb.rewrite_edits(text, "nobody", ["hp"]) is None
    assert cb.rewrite_edits(text, "other", ["hp"]) == cb.dump_edits({"about": ABOUT, "cards": {"Mini_Doomer": {"hp": 8, "note": "n"}}})
    assert cb.psd_fields({"hp": 8, "spd": None, "name": "X", "series": "m", "creatures": [{"hp": 1}]}) == {"hp": 8, "name": "X"}
    parts = {"1": {"field": "name"}, "2": {"field": "ability"}}
    assert cb.psd_fields({"name": "X", "hp": 1, "parts": parts}, ["name"]) == {"hp": 1, "parts": {"2": {"field": "ability"}}}   # _site
    assert cb.psd_fields({"name": "X", "parts": {"1": parts["1"]}}, ["name"]) == {}


# ========================
# The driver against the fake GitHub
# ========================
def test_run_drops_redundant_and_writes_verdicts(github, tmp_path):
    before = github.head
    text, commits = cb.run()
    assert len(commits) == 1 and github.head == commits[0]
    commit = github.commits[commits[0]]
    assert commit["message"] == "Card edits: 1 redundant" and commit["author"]["name"] == "AutoCroissant"
    assert [p["sha"] for p in commit["parents"]] == [before]                                     # one commit for the run
    data = _edits_at(github)
    assert data["cards"]["Mini Doomer"] == {"note": "checked", "_by": "web", "_at": AT}          # hp dropped, no verdict
    assert data["cards"]["Beldam"] == EDITS["Beldam"]
    assert not any("_psd" in entry for entry in data["cards"].values())                         # verdicts live on data
    freezer = _verdict("The Freezer")
    assert freezer["state"] == "refused" and freezer["for"] == {"stars": 2} and freezer["psd"] == _sha(github.files_at(before)[FREEZER])[:8]
    assert freezer["reason"] == ("unsupported-card: stars: Field/2 Stars/The_Freezer.png is already there: the card can't "
                                 "move into 'Field/2 Stars'")
    assert freezer["at"].endswith("Z")
    assert _verdict("Nobody")["reason"] == "unsupported-card: no PSD in the card repo" and _verdict("Nobody")["psd"] == ""
    assert _verdict("A Snack")["reason"] == ("duplicate: numbered PNGs share the PSD: A_Snack2.png (tag the "
                                             "variant layers: PARTS_EDITOR.md §9)")
    assert _verdict("A Snack2")["reason"] == "duplicate: a numbered PNG exported from `Items/Heal/2 Stars/A_Snack.psd`"
    verdicts, events = _records(tmp_path)
    assert sorted(verdicts["cards"]) == ["a snack", "a snack2", "nobody", "the freezer"] and verdicts["landed"] == {}
    assert sorted(e["card"] for e in events["events"]) == ["A Snack", "A Snack2", "Nobody", "The Freezer"]
    assert {e["kind"] for e in events["events"]} == {"bake-refused"} and events["landed"] == []      # nothing printed
    raw = github.files_at(github.head)[cb.EDITS_PATH].decode("utf-8")
    assert raw == cb.dump_edits(json.loads(raw))                                                 # canonical layout
    assert github.files_at(github.head)[DOOMER] == github.files_at(before)[DOOMER]              # no pixels touched
    assert "## Redundant (dropped from the file) (1)" in text and "## Refused (4)" in text
    assert text.count(f"committed `{commits[0][:8]}`") == 1                                    # the report line per committed card
    assert "**Nothing baked.**" in text and f"One commit: `{commits[0][:8]}` Card edits: 1 redundant." in text
    blobs = [route for method, route in github.calls if route.startswith("git/blobs/")]
    assert len(blobs) == 8      # the edits file + manifest, PSD + PNG of Mini Doomer and The Freezer, A Snack's PSD
    assert [route for _, route in github.calls].count("activity") == 1     # the recent gate reads the pushes once a run
    assert cb.pending() == []                                               # every PSD-kind edit has its final verdict

    # a rerun on the pipeline's own commit: the redundant field is gone, the verdicts are unchanged: nothing to commit
    text, commits = cb.run()
    assert commits == [] and "## Refused (4)" in text and "committed `" not in text and "One commit" not in text
    assert _records(tmp_path)[0]["cards"] == {} and _records(tmp_path)[1]["events"] == []      # told once

    # a human commit on top: the same
    github.commit(github.files_at(github.head), "web", "Website edit: something else")
    text, commits = cb.run()
    assert commits == [] and "## Refused (4)" in text and "committed `" not in text


def test_website_only_fields_never_bake(github, tmp_path):
    """`_site` (owner 10-25): its fields stay in the file, unbaked, with no verdict; the card's other fields bake as usual;
    unticked, the field is a normal PSD edit again."""
    files = github.files_at(github.head)
    entry = {"name": "Tiny Doomer", "hp": 9, "def": 2, "note": "checked", "_site": ["name", "hp"], "_by": "web", "_at": AT}
    files[cb.EDITS_PATH] = _edits({"Mini Doomer": entry})
    github.commit(files, "web", "Website edit: Mini Doomer")
    assert [c.fields for c in cb.select(cb.CardEdits.parse(files[cb.EDITS_PATH].decode()), cb.snapshot_at(cb.CARD_REPO, github.head))] == [{"def": 2}]
    text, commits = cb.run()
    assert len(commits) == 1 and "Card edits: 1 redundant" in github.commits[commits[0]]["message"]      # def 2 is printed
    assert _edits_at(github)["cards"]["Mini Doomer"] == {k: v for k, v in entry.items() if k != "def"}
    assert _verdict("Mini Doomer") is None and _records(tmp_path)[1]["landed"] == []
    assert cb.pending() == [] and cb.run()[1] == []                                     # nothing left to do
    files = github.files_at(github.head)
    files[cb.EDITS_PATH] = _edits({"Mini Doomer": {**entry, "_site": ["name"]}})        # hp unticked
    github.commit(files, "web", "Website edit: Mini Doomer")
    assert cb.pending() == ["Mini Doomer"] and [c.fields for c in cb._pending(cb.CARD_REPO, "", "")[1]] == [{"hp": 9, "def": 2}]


def test_dry_run_commits_nothing(github, tmp_path):
    head = github.head
    text, commits = cb.run(dry_run=True)
    assert commits == [] and github.head == head and "(dry run)" in text and "not committed (dry run)" in text
    assert not [m for m, _ in github.calls if m != "GET"]
    work = tmp_path / cb.WORK_NAME
    staged = json.loads((work / "Card Edits.json").read_text(encoding="utf-8"))      # the one file the commit would hold
    assert staged["cards"]["Mini Doomer"] == {"note": "checked", "_by": "web", "_at": AT}
    assert "_psd" not in staged["cards"]["The Freezer"] and _verdict("The Freezer") is None   # a dry run records nothing
    assert _records(tmp_path)[0]["cards"]["the freezer"]["state"] == "refused"               # but its artifact says it
    assert "Would commit: Card edits: 1 redundant." in text
    assert (work / "report.md").read_text(encoding="utf-8") == text


def test_recent_refuses_before_fetching(github):
    github.recent.add(DOOMER)
    text, commits = cb.run(card="Mini Doomer")
    assert commits == [] and "recent: PSD pushed in the last 30 min" in text
    assert not [route for _, route in github.calls if route.startswith("git/blobs/") and route.endswith(_sha(github.blobs[_sha(github.files_at(github.head)[DOOMER])]))]
    assert _verdict("Mini Doomer") is None                                         # transient: no verdict
    github.recent = {DOOMER[:len(DOOMER) - 4] + ".png"}                          # a teammate's PNG re-export
    text, commits = cb.run(card="Mini Doomer")
    assert commits == [] and "recent: PNG pushed in the last 30 min" in text
    assert _verdict("Mini Doomer") is None and "Mini Doomer" in cb.pending()
    github.recent = {f"Other/{i}.psd" for i in range(cb.COMPARE_FILES)}
    text, commits = cb.run(card="Mini Doomer")
    assert commits == [] and "recent: too many pushes in the last 30 min to tell" in text


def test_recent_paths_by_push_time(monkeypatch):
    """A commit dated hours ago but pushed just now is recent: the window is the pushes', the paths the compare's."""
    now = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)
    pushes = [{"before": "c2", "after": "c3", "timestamp": "2026-10-21T11:50:00Z"},
              {"before": "c1", "after": "c2", "timestamp": "2026-10-21T11:40:00Z"},
              {"before": "c0", "after": "c1", "timestamp": "2026-10-21T11:00:00Z"}]      # before the window
    asked = []

    def api(repository, method, path, **kwargs):
        asked.append(path)
        if path == "activity":
            return Reply(body=pushes)
        return Reply(body={"files": [{"filename": "a.psd"}, {"filename": "b.psd", "previous_filename": "old b.psd"}]})
    monkeypatch.setattr(cb, "api", api)
    assert cb.recent_paths(cb.CARD_REPO, "c3", now) == {"a.psd", "b.psd", "old b.psd"} and asked == ["activity", "compare/c1...c3"]
    assert cb.recent_paths(cb.CARD_REPO, "c3", now + timedelta(hours=1)) == set() and asked[2:] == ["activity"]
    pushes[1]["before"] = "0" * 40                                                      # a new branch: can't tell
    assert cb.recent_paths(cb.CARD_REPO, "c3", now) is None
    pushes[1]["before"] = "c1"
    monkeypatch.setattr(cb, "api", lambda repository, method, path, **kwargs: Reply(body=pushes) if path == "activity"
                        else Reply(status=422, body={"message": "Sorry, this diff is taking too long to generate."}))
    assert cb.recent_paths(cb.CARD_REPO, "c3", now) is None                              # a force push: no compare


def test_creature_one_bakes_and_website_only_fields(github):
    """The card's stats are creature 1's (owner 2026-10-26): its hp bakes and the entry's creatures list takes it for
    creature 1 (else the list's old value would show again), the others' stay; a stat past the bar's one row is website
    only: it stays in the entry, added to `_site`, no verdict."""
    files = github.files_at(github.head)
    files[cb.EDITS_PATH] = _edits({"Mini Doomer": {"hp": 6, "atk": 12, "creatures": [{"hp": 7, "def": 1, "atk": 1, "spd": 1},
                                                                                    {"hp": 2, "def": 2, "atk": 2, "spd": 2}],
                                                   "_by": "web", "_at": AT}})
    github.commit(files, "web", "Website edit: Mini Doomer")
    text, commits = cb.run(card="Mini Doomer")
    entry = _edits_at(github)["cards"]["Mini Doomer"]
    assert len(commits) == 1 and "hp" not in entry and entry["atk"] == 12 and entry["_site"] == ["atk"]
    assert [c["hp"] for c in entry["creatures"]] == [6, 2] and "website only: atk 12" in text


def test_blob_gate(github):
    psd = _sha(github.files_at(github.head)[DOOMER])
    github.broken[psd] = b"not the PSD"
    text, commits = cb.run(card="Mini Doomer")
    assert "blob: blob %s hashes to" % psd[:8] in text
    assert commits == [] and _verdict("Mini Doomer")["reason"].startswith("blob: ")


@pytest.mark.parametrize("status", [422])
def test_head_moved_after_three_conflicts(github, status):
    github.conflicts, github.conflict_status = 3, status
    text, commits = cb.run(card="Mini Doomer")
    assert commits == [] and "head-moved: main kept moving: not committed after 3 tries" in text
    assert _edits_at(github)["cards"]["Mini Doomer"]["hp"] == 5
    github.conflicts = 2
    text, commits = cb.run(card="Mini Doomer")
    assert len(commits) == 1 and _edits_at(github)["cards"]["Mini Doomer"] == {"note": "checked", "_by": "web", "_at": AT}


def test_refused_ref_update_is_an_error_not_a_race(github):
    """A 422 with the branch unmoved (a ruleset) is GitHub's refusal: one try, its message, no verdict."""
    github.stuck = True
    text, commits = cb.run(card="Mini Doomer")
    assert commits == [] and "error: GitHub 422: Repository rule violations found (on PATCH git/refs/heads/main)" in text
    assert [route for method, route in github.calls if method == "PATCH"] == ["git/refs/heads/main"]
    assert _edits_at(github)["cards"]["Mini Doomer"]["hp"] == 5


def test_edit_changed_meanwhile(github):
    snapshot = github.head
    files = github.files_at(snapshot)
    files[cb.EDITS_PATH] = _edits({**EDITS, "Mini Doomer": {"hp": 4, "note": "checked", "_by": "web", "_at": AT}})
    github.commit(files, "web", "Website edit: Mini Doomer")
    text, commits = cb.run(head=snapshot, card="Mini Doomer")
    assert commits == [] and "edit-changed: the edit changed on main meanwhile" in text
    assert _edits_at(github)["cards"]["Mini Doomer"]["hp"] == 4


def test_moved_psd_leaves_only_its_card_out(github):
    """A card whose PSD changed under the run stays out of the run's commit (a try-again refusal, no verdict); the rest lands."""
    snapshot = github.head
    files = github.files_at(snapshot)
    files[DOOMER] = files[DOOMER] + b"\0"
    github.commit(files, "artist", "Mini Doomer: new art")
    text, commits = cb.run(head=snapshot)
    assert commits == []                                                    # 4 refused: no main commit
    data = _edits_at(github)
    assert data["cards"]["Mini Doomer"] == {"hp": 5, "note": "checked", "_by": "web", "_at": AT}     # untouched, no verdict
    assert _verdict("Mini Doomer") is None
    assert _verdict("The Freezer")["state"] == "refused" and github.files_at(github.head)[DOOMER] == files[DOOMER]
    assert f"**Mini Doomer** `{DOOMER}`: hp 5: head-moved: {DOOMER} changed on main" in text
    assert cb.pending() == ["Mini Doomer"]                                  # the next scheduled run tries it again


def test_pending_is_the_schedule_gate(github):
    """pending: a PSD-kind edit without a refused verdict for exactly its values on the PSD blob the tree holds now."""
    assert cb.pending() == ["A Snack", "A Snack2", "Mini Doomer", "Nobody", "The Freezer"]     # nothing tried yet
    cb.run()
    assert cb.pending() == []
    files = github.files_at(github.head)
    data = json.loads(files[cb.EDITS_PATH].decode("utf-8"))
    data["cards"]["Beldam"]["series"] = "Paper Mario"                       # database-only: nothing to bake
    files[cb.EDITS_PATH] = cb.dump_edits(data).encode("utf-8")
    github.commit(files, "web", "Website edit: Beldam")
    assert cb.pending() == []
    data["cards"]["Nobody"]["hp"] = 4                                       # a refused value changed
    data["cards"]["Mini Doomer"]["atk"] = 2                                 # a new PSD field
    files[cb.EDITS_PATH] = cb.dump_edits(data).encode("utf-8")
    files[FREEZER] = files[FREEZER] + b"\0"                                 # a refused card's PSD changed
    github.commit(files, "web", "Website edit: Nobody, Mini Doomer; The Freezer's art")
    assert cb.pending() == ["Mini Doomer", "Nobody", "The Freezer"]
    from commands.card_db import merge_verdicts
    merge_verdicts(str(DB[0]), {"mini doomer": {"state": "baked", "for": {"atk": 2}, "psd": "01234567", "reason": "x", "at": AT}})
    assert "Mini Doomer" in cb.pending()                                    # only a refusal is final
    files[cb.EDITS_PATH] = b"not json"
    github.commit(files, "web", "Website edit: broken")
    with pytest.raises(ValueError):                                         # the gate step fails: Discord is told
        cb.pending()


def test_pending_run_skips_final_verdicts(github):
    """--pending (the schedule): a card whose refused verdict still holds is not looked at again; a dispatch re-checks it."""
    from commands.card_db import merge_verdicts
    cb.run()
    files = github.files_at(github.head)
    merge_verdicts(str(DB[0]), {"the freezer": {**_verdict("The Freezer"), "reason": "stars: changed by hand"}})   # a re-check would rewrite it
    github.calls.clear()
    text, commits = cb.run(pending_only=True)
    assert commits == [] and "4 with PSD fields, 4 refused before (not re-checked)" in text
    assert not [route for _, route in github.calls if route.startswith("git/blobs/") and route != "git/blobs/" + _sha(files[cb.EDITS_PATH])]
    text, commits = cb.run()
    assert commits == [] and "not re-checked" not in text                   # the dispatch rewrote The Freezer's verdict
    assert _verdict("The Freezer")["reason"] != "stars: changed by hand"


def test_a_printed_value_is_dropped_though_numbered_pngs_share_the_psd(github):
    """NME Stock: its name edit is what the PSD prints, but NME_Stock2.png shares the PSD. The redundancy check runs
    before the duplicate gate, so the value is dropped (no bake, no PNG needed) instead of refused for good; a refusal
    written before the check (no "printed") is looked at again."""
    files = github.files_at(github.head)
    files[cb.EDITS_PATH] = _edits({"A Snack": {"hp": 5, "note": "n", "_by": "web", "_at": AT}})
    github.commit(files, "web", "Website edit: A Snack")
    text, commits = cb.run()
    assert len(commits) == 1 and "already on the card: hp" in text
    assert _edits_at(github)["cards"]["A Snack"] == {"note": "n", "_by": "web", "_at": AT}
    data = _edits_at(github)
    old = {"state": "refused", "for": {"hp": 1}, "psd": _sha(files[SNACK])[:8], "reason": "duplicate: numbered PNGs "
           "share the PSD: A_Snack2.png", "at": AT}
    files = github.files_at(github.head)
    files[cb.EDITS_PATH] = _edits({**data["cards"], "A Snack": {"hp": 1, "_psd": old, "_by": "web", "_at": AT}})
    github.commit(files, "web", "Website edit: old verdict")                   # the migration's cycle: the entry's _psd read
    assert "A Snack" in cb.pending()                                            # no "printed": looked at again
    cb.run()
    verdict = _verdict("A Snack")
    assert verdict["printed"] is False and verdict["reason"].startswith("duplicate:") and "A Snack" not in cb.pending()


def test_commit_failure_is_reported_and_writes_nothing(github, tmp_path):
    """The run's commit is fenced: head-moved after three tries is a note and the redundant card a try-again refusal; the
    refusals' verdicts need no commit (the data branch takes them); the report survives."""
    github.conflicts = 3
    text, commits = cb.run()
    assert commits == [] and "**Nothing committed: head-moved: main kept moving: not committed after 3 tries**" in text
    data = _edits_at(github)
    assert data["cards"]["Mini Doomer"]["hp"] == 5 and _verdict("Mini Doomer") is None
    assert "head-moved: main kept moving" in text.split("## Refused", 1)[1] and "committed `" not in text
    assert _verdict("The Freezer")["state"] == "refused" and _records(tmp_path)[0]["landed"] == {}
    assert (tmp_path / cb.WORK_NAME / "report.md").read_text(encoding="utf-8") == text
    assert cb.pending() == ["Mini Doomer"]                                  # the redundant card again next run


def test_main_exits_1_when_the_commit_fails_or_a_card_errors(github, monkeypatch):
    """A failure needs a person: main exits 1 (card-sync.yml then tells Discord); refusals alone exit 0."""
    err = StringIO()
    monkeypatch.setattr(cb, "stderr", err)
    github.conflicts = 3
    assert cb.main([]) == 1 and "card_bake: Nothing committed: head-moved" in err.getvalue()
    github.conflicts = 0
    github.down.add(_sha(github.files_at(github.head)[DOOMER]))
    assert cb.main(["--card", "Mini Doomer"]) == 1 and err.getvalue().endswith("card_bake: 1 card with an error: Mini Doomer\n")
    github.down.clear()
    err.truncate(0)
    assert cb.main([]) == 0 and cb.main([]) == 0 and err.getvalue() == ""   # 1 redundant committed, 4 refused; then nothing to do


def test_cap_leaves_the_rest_for_the_next_run(github, monkeypatch):
    """At most MAX_BAKES cards baked and none started after BUDGET: the rest are `capped`, no verdict, pending next run."""
    monkeypatch.setattr(cb, "BUDGET", timedelta(0))
    text, commits = cb.run()
    assert commits == [] and "## Next run (capped) (5)" in text and "- Capped: 5 cards wait for the next run" in text
    assert "## Refused" not in text and len(cb.pending()) == 5
    monkeypatch.setattr(cb, "BUDGET", timedelta(minutes=40))

    files = github.files_at(github.head)
    data = json.loads(files[cb.EDITS_PATH].decode("utf-8"))
    data["cards"]["Mini Doomer"]["hp"] = 8
    files[cb.EDITS_PATH] = cb.dump_edits(data).encode("utf-8")
    github.commit(files, "web", "Website edit: Mini Doomer")

    def apply_jobs(old_psd, old_png, rel, jobs, parsed, parser, new_dir, *args, **kwargs):
        paths = {}
        for ext in ("psd", "png"):
            paths[ext] = f"{new_dir}/new.{ext}"
            shutil.copy(old_psd if ext == "psd" else old_png, paths[ext])
        with open(paths["psd"], "ab") as fh:
            fh.write(b"\0")
        return SimpleNamespace(psd_path=paths["psd"], png_path=paths["png"], edits=[SimpleNamespace(field="hp", old=5)],
                               region=None, reparse={"card": {}, "validator": []}, fx=[], parity={}, flag_bytes=0,
                               changed_pixels=0)
    monkeypatch.setattr(pe, "apply_jobs", apply_jobs)
    monkeypatch.setattr(pe, "artifacts", lambda *args: {})
    monkeypatch.setattr(cb, "MAX_BAKES", 1)
    text, commits = cb.run()                                                # A Snack, A Snack2, Mini Doomer, then the cap
    assert len(commits) == 1 and github.commits[commits[0]]["message"].startswith("Card edits: 1 baked\n")
    assert "## Next run (capped) (2)" in text and "**Nobody**" in text.split("## Next run", 1)[1]
    assert cb.pending() == ["Nobody", "The Freezer"]


def test_font_verdict_is_retried_when_the_fonts_change(github, monkeypatch):
    """A `font` refusal carries MDW/Fonts' key: final while the fonts hold, pending once a font file is added or changed."""
    files = github.files_at(github.head)
    files["MDW/Fonts/Some Face.ttf"] = b"face"
    files["MDW/Fonts/stoogedings-src/x.svg"] = b"not installed"
    files[cb.EDITS_PATH] = _edits({"Mini Doomer": {"hp": 8, "_by": "web", "_at": AT}})
    github.commit(files, "web", "Website edit: Mini Doomer")
    monkeypatch.setattr(pe, "apply_jobs", lambda *args, **kwargs: (_ for _ in ()).throw(pe.Refused("font", "no Some Face")))
    monkeypatch.setattr(pe, "text_engine", lambda: True)
    text, commits = cb.run()
    verdict = _verdict("Mini Doomer")
    key = cb.snapshot_at(cb.CARD_REPO, github.head).fonts_key()
    assert commits == [] and verdict["reason"] == "font: no Some Face" and verdict["fonts"] == key and len(key) == 8
    assert cb.pending() == []
    files = github.files_at(github.head)
    files["MDW/Fonts/stoogedings-src/x.svg"] = b"changed, still not installed"
    github.commit(files, "web", "Fonts: source only")
    assert cb.pending() == []
    files["MDW/Fonts/Some Face.ttf"] = b"a fixed face"
    github.commit(files, "web", "Fonts: Some Face")
    assert cb.pending() == ["Mini Doomer"]


def test_pixel_miss_bakes_with_a_note(github, monkeypatch):
    """A box past the parity bar, under the floor, bakes (owner, 2026-10-23): the commit note and the report say so."""
    files = github.files_at(github.head)
    files[cb.EDITS_PATH] = _edits({"Mini Doomer": {"hp": 8, "_by": "web", "_at": AT}})
    github.commit(files, "web", "Website edit: Mini Doomer")
    miss = {"bad_pct": 1.9725, "mean": 1.5955, "max": 217}

    def apply_jobs(old_psd, old_png, rel, jobs, parsed, parser, new_dir, *args, **kwargs):
        for ext, old in (("psd", old_psd), ("png", old_png)):
            shutil.copy(old, f"{new_dir}/new.{ext}")
        with open(f"{new_dir}/new.psd", "ab") as fh:
            fh.write(b"\0")
        return SimpleNamespace(psd_path=f"{new_dir}/new.psd", png_path=f"{new_dir}/new.png", edits=[SimpleNamespace(field="hp", old=5)],
                               region=None, reparse={"card": {}, "validator": []}, fx=[], parity={"hp": miss},
                               rerender={"hp": miss}, flag_bytes=2, changed_pixels=1)
    monkeypatch.setattr(pe, "apply_jobs", apply_jobs)
    monkeypatch.setattr(pe, "artifacts", lambda *args: {})
    text, commits = cb.run()
    note = "hp baked with a pixel difference: bad 1.97%, mean 1.60, max 217"
    assert len(commits) == 1 and github.commits[commits[0]]["message"] == f"Card edits: 1 baked\n\n- Mini Doomer: hp 5 -> 8; {note}\n\nEdit-by: web"
    assert f"parity hp 1.97% bad, mean 1.60; {note}; 2 flag bytes" in text.split("## Baked (1)", 1)[1]
    assert "Mini Doomer" not in _edits_at(github)["cards"]                  # baked: the entry goes whole


def test_fx_notes_reach_the_commit_and_the_verdict(github, clone, monkeypatch, tmp_path):
    """Review 2026-10-23: the FX layers an ability bake deleted, renamed or cut again are told in the commit line and in
    the website's verdict reason too, not only the run report; with no field left the verdict stays alone (review
    2026-10-24: the card page shows the flags)."""
    files = github.files_at(github.head)
    if not (clone / pe.STATUS_CHART).exists():
        pytest.skip("no status chart")
    files[pe.STATUS_CHART] = (clone / pe.STATUS_CHART).read_bytes()
    files[cb.EDITS_PATH] = _edits({"Mini Doomer": {"ability": "Die.", "_by": "web", "_at": AT}})
    github.commit(files, "web", "Website edit: Mini Doomer")
    ok = {"bad_pct": 0.0, "mean": 0.5, "max": 7}
    note = "Ability FX: (BOT) deleted: its words left the ability"

    def apply_jobs(old_psd, old_png, rel, jobs, parsed, parser, new_dir, *args, **kwargs):
        for ext, old in (("psd", old_psd), ("png", old_png)):
            shutil.copy(old, f"{new_dir}/new.{ext}")
        with open(f"{new_dir}/new.psd", "ab") as fh:
            fh.write(b"\0")
        return SimpleNamespace(psd_path=f"{new_dir}/new.psd", png_path=f"{new_dir}/new.png", edits=[SimpleNamespace(field="ability", old="a")],
                               region=None, reparse={"card": {}, "validator": []}, fx=[], fx_notes=[note], parity={"ability": ok},
                               rerender={"ability": ok}, flag_bytes=0, changed_pixels=1)
    monkeypatch.setattr(pe, "apply_jobs", apply_jobs)
    monkeypatch.setattr(pe, "artifacts", lambda *args: {})
    monkeypatch.setattr(pe, "text_engine", lambda: True)
    text, commits = cb.run()
    assert commits, text
    assert f"- Mini Doomer: ability; {note}\n" in github.commits[commits[0]]["message"] + "\n"
    assert "Mini Doomer" not in _edits_at(github)["cards"] and _verdict("Mini Doomer")["reason"] == f"ability; {note}"
    assert text.count(note) == 1
    (baked,) = _records(tmp_path)[1]["landed"]                              # the changelog's baked event
    assert baked == {"kind": "baked", "card": "Mini Doomer", "fields": ["ability"], "reason": f"ability; {note}",
                     "pngs": [DOOMER[:len(DOOMER) - 4] + ".png"]}


def test_transient_errors_write_no_verdict(github, monkeypatch):
    """A 5xx on a blob, a rate-limited POST: reported under Errors, retried next run, never a verdict for the editor."""
    github.down.add(_sha(github.files_at(github.head)[DOOMER]))
    text, commits = cb.run(card="Mini Doomer")
    assert commits == [] and "## Errors (1)" in text and "HTTPError: 503" in text
    assert _verdict("Mini Doomer") is None
    github.down.clear()
    github.rate_limited = True
    monkeypatch.setattr(gc, "git_commit", lambda *a, **k: pytest.fail("the SSH path is not for a rate limit"))
    text, commits = cb.run(card="Mini Doomer")
    assert commits == [] and "error: GitHub 403: API rate limit exceeded: rate limited (on POST git/blobs)" in text
    assert _verdict("Mini Doomer") is None and gc.cant_write is False
    github.rate_limited = False
    assert len(cb.run(card="Mini Doomer")[1]) == 1


def test_forbidden_token_falls_to_ssh(github, monkeypatch):
    """A token that reads but can't write (403, no rate-limit header): the commit lands over SSH; the API is not asked to write again."""
    github.forbidden = True
    pushed = []

    def ssh(repository, head, files, message):
        pushed.append(message)
        return github.commit({**github.files_at(head), **files}, "AutoCroissant", message)
    monkeypatch.setattr(gc, "git_commit", ssh)
    text, commits = cb.run(card="Mini Doomer")
    assert len(commits) == 1 and pushed == ["Card edits: 1 redundant"] and gc.cant_write is True
    assert [route for method, route in github.calls if method == "POST"] == ["git/blobs"]        # refused once, then SSH
    assert _edits_at(github)["cards"]["Mini Doomer"] == {"note": "checked", "_by": "web", "_at": AT} and f"committed `{commits[0][:8]}`" in text
    files = github.files_at(github.head)
    files[cb.EDITS_PATH] = _edits({"Mini Doomer": {"hp": 5, "_by": "web", "_at": AT}})            # printed already
    github.commit(files, "web", "Website edit: more")
    text, commits = cb.run(card="mini_doomer")
    assert len(commits) == 1 and len(pushed) == 2 and pushed[1] == "Card edits: 1 redundant"
    assert [route for method, route in github.calls if method == "POST"] == ["git/blobs"]        # no API write tried this run
    assert cb.run(card="the_freezer")[1] == [] and len(pushed) == 2                             # a refusal: no commit


def test_verdict_for_an_edit_changed_before_the_commit(github, monkeypatch):
    """A refused card whose edit changed meanwhile keeps the new edit; its verdict names the values it was tried on, so
    the new ones are pending; the others land."""
    real = cb.land

    def land(repository, shares, message, dry_run, out_path="", local=None):
        current = github.files_at(github.head)                  # before the commit: The Freezer's edit changes
        data = json.loads(current[cb.EDITS_PATH].decode("utf-8"))
        data["cards"]["The Freezer"]["stars"] = 3
        current[cb.EDITS_PATH] = cb.dump_edits(data).encode("utf-8")
        github.commit(current, "web", "Website edit: The Freezer again")
        return real(repository, shares, message, dry_run, out_path, local)
    monkeypatch.setattr(cb, "land", land)
    text, commits = cb.run()
    assert len(commits) == 1 and "**The Freezer**" in text
    assert github.commits[commits[0]]["message"] == "Card edits: 1 redundant"
    data = _edits_at(github)
    assert data["cards"]["The Freezer"]["stars"] == 3 and _verdict("The Freezer")["for"] == {"stars": 2}
    assert _verdict("Nobody")["state"] == "refused" and _verdict("A Snack")["state"] == "refused"
    assert cb.pending() == ["The Freezer"]


def test_missing_runner_dependency_is_transient(github, monkeypatch):
    """A runner without scipy or the text extra fails inside apply_jobs with an ImportError: an error, never a verdict."""
    files = github.files_at(github.head)
    files[cb.EDITS_PATH] = _edits({"Mini Doomer": {"hp": 8, "_by": "web", "_at": AT}})
    github.commit(files, "web", "Website edit: Mini Doomer")

    def no_scipy(*args, **kwargs):
        raise ImportError("Gradient fills require: scipy")
    monkeypatch.setattr(pe, "apply_jobs", no_scipy)
    text, commits = cb.run(card="Mini Doomer")
    assert commits == [] and "## Errors (1)" in text and "ImportError: Gradient fills require: scipy" in text
    entry = _edits_at(github)["cards"]["Mini Doomer"]
    assert _verdict("Mini Doomer") is None and entry["hp"] == 8


def test_select_and_snapshot(github):
    info = github.commits[github.head]
    snapshot = cb.Snapshot(cb.CARD_REPO, github.head, info, cb._tree(cb.CARD_REPO, info["tree"]["sha"]))
    assert "creature template" not in snapshot.psds and snapshot.psds["mini doomer"] == DOOMER
    assert snapshot.png_for(DOOMER) == (DOOMER[:len(DOOMER) - 4] + ".png", [])
    assert snapshot.png_for(SNACK) == (SNACK[:len(SNACK) - 4] + ".png", ["Items/Heal/2 Stars/A_Snack2.png"])
    assert snapshot.duplicate_of("a snack2") == SNACK and snapshot.duplicate_of("nobody") == ""
    edits = cb.CardEdits.parse(github.files_at(github.head)[cb.EDITS_PATH].decode("utf-8"))
    cards = cb.select(edits, snapshot)
    assert [c.name for c in cards] == ["A Snack", "A Snack2", "Mini Doomer", "Nobody", "The Freezer"]    # the file's order
    assert cards[2].fields == {"hp": 5} and cards[3].rel == "" and cards[4].psd_sha == snapshot.sha(FREEZER)
    assert [c.name for c in cb.select(edits, snapshot, "a_snack")] == ["A Snack"]


@fork
def test_bake_hp_lands_psd_png_and_file(github, tmp_path):
    """One stat bake end to end: PSD + PNG + the file in one commit, the verdict beside the fields that stay."""
    from psd_tools import PSDImage
    files = github.files_at(github.head)
    files[cb.EDITS_PATH] = _edits({"Mini Doomer": {"hp": 8, "note": "checked", "_by": "web", "_at": AT}})
    github.commit(files, "web", "Website edit: Mini Doomer")
    text, commits = cb.run()
    assert len(commits) == 1 and github.commits[commits[0]]["message"] == "Card edits: 1 baked\n\n- Mini Doomer: hp 5 -> 8\n\nEdit-by: web"
    after = github.files_at(github.head)
    changed = sorted(p for p in after if after[p] != files.get(p))
    assert changed == [DOOMER[:len(DOOMER) - 4] + ".png", DOOMER, cb.EDITS_PATH]
    data = json.loads(after[cb.EDITS_PATH].decode("utf-8"))
    verdict = _verdict("Mini Doomer")                       # the entry keeps a field: its verdict stays (on data)
    assert data["cards"]["Mini Doomer"] == {"note": "checked", "_by": "web", "_at": AT}
    assert verdict["state"] == "baked" and verdict["for"] == {"hp": 8} and verdict["psd"] == _sha(after[DOOMER])[:8]
    assert verdict["reason"] == "hp 5 -> 8"
    work = tmp_path / cb.WORK_NAME / "Mini Doomer"
    saved = PSDImage.open(str(work / "new" / "Mini_Doomer.psd"))
    assert pe.template_row(saved, "hp")[2] == [8] and (work / "new" / "Mini_Doomer.psd").read_bytes() == after[DOOMER]
    assert all((work / f"{name}.png").exists() for name in ("before", "after", "diff"))
    parse = json.loads((work / "parse.json").read_text(encoding="utf-8"))
    assert parse["old"]["hp"] == 5 and parse["new"]["hp"] == 8 and parse["validator"] == []
    assert "## Baked (1)" in text and "2 flag bytes" in text and f"committed `{commits[0][:8]}`" in text
    github.commit(github.files_at(github.head), "web", "Website edit: something else")
    text, commits = cb.run()                                 # the field is gone: nothing to look at
    assert commits == [] and "0 with PSD fields" in text


@fork_text
def test_a_bake_opens_each_psd_once(github, clone, monkeypatch):
    """One PSDImage.open per PSD file per bake (tests/test_open_once.py's count): the old card's is shared by the parse,
    classify and apply_jobs, the saved card's by gate_bytes, the re-parse and the FX gates; the status chart's once.
    apply_jobs' copies (edit_copy) open the old card's bytes again (K-3). A stat and an ability edit (its FX layer
    deleted, the parser's every gate run)."""
    from collections import Counter
    from hashlib import sha1
    from os import PathLike
    from psd_tools import PSDImage
    files = github.files_at(github.head)
    if not (clone / pe.STATUS_CHART).exists():
        pytest.skip("no status chart")
    files[pe.STATUS_CHART] = (clone / pe.STATUS_CHART).read_bytes()
    files[cb.EDITS_PATH] = _edits({"Mini Doomer": {"hp": 8, "ability": "Die.", "_by": "web", "_at": AT}})
    github.commit(files, "web", "Website edit: Mini Doomer")
    seen, copies, inner = Counter(), Counter(), PSDImage.open.__func__

    def counted(cls, fp, *args, **kwargs):
        if isinstance(fp, (str, bytes, PathLike)):
            with open(fp, "rb") as fh:
                seen[sha1(fh.read()).hexdigest()] += 1
        elif hasattr(fp, "getvalue"):
            copies[sha1(fp.getvalue()).hexdigest()] += 1
        else:
            seen[f"stream {id(fp)}"] += 1
        return inner(cls, fp, *args, **kwargs)
    monkeypatch.setattr(PSDImage, "open", classmethod(counted))
    text, commits = cb.run(card="Mini Doomer")
    assert len(commits) == 1, text
    assert github.commits[commits[0]]["message"].startswith("Card edits: 1 baked\n"), text
    after = github.files_at(github.head)
    assert after[DOOMER] != files[DOOMER], text            # the saved PSD's open is counted, not a refusal's
    assert seen == Counter({sha1(files[DOOMER]).hexdigest(): 1, sha1(after[DOOMER]).hexdigest(): 1,
                            sha1(files[pe.STATUS_CHART]).hexdigest(): 1}), text
    assert set(copies) == {sha1(files[DOOMER]).hexdigest()}, text


icons = pytest.mark.skipif(not pe.icon_engine(), reason="a stars or types bake needs the fork's layer copy API")


@icons
def test_a_card_refused_after_its_stars_names_both_refusals(github, monkeypatch):
    """Stars refused, then the card's last field (types, then hp) too: the card's verdict names both, the stars once."""
    def outside(before, psd, edit):
        raise pe.Refused("region", f"{edit.field}: the card changes 9 pixels outside the edited box", edit.field)
    monkeypatch.setattr(pe, "gate_outside", outside)
    monkeypatch.setattr(pe, "render_all", lambda psd: None)    # no render: the whole-card gates pass
    monkeypatch.setattr(pe, "gate_untouched", lambda before, psd, edits: None)
    monkeypatch.setattr(pe, "set_stat", lambda *a: (_ for _ in ()).throw(pe.Refused("unsupported-card", "hp: no stat row")))
    stars, types = (f"region: {f}: the card changes 9 pixels outside the edited box" for f in ("stars", "types"))
    for fields, last in (({"stars": 3, "types": ["wind", "light"]}, types), ({"stars": 3, "hp": 8}, "unsupported-card: hp: no stat row")):
        files = github.files_at(github.head)
        files[cb.EDITS_PATH] = _edits({"Mini Doomer": {**fields, "_by": "web", "_at": AT}})
        github.commit(files, "web", "Website edit: Mini Doomer")
        text, commits = cb.run()
        verdict = _verdict("Mini Doomer")
        assert (verdict["state"], verdict["for"], verdict["reason"]) == ("refused", fields, f"{last}; {stars}")
        assert "## Refused (1)" in text and text.count(stars) == 1


# ========================
# --local: the card sync's bake (one local commit B, nothing pushed)
# ========================
SPARSE = ("/site/Card Edits.json", "/" + cb.ICON_MANIFEST_PATH, "/TTS/")      # card-sync.yml's checkout


def _git(where, *args, env=None) -> str:
    done = run_process(["git", "-C", str(where), *args], capture_output=True, text=True, timeout=60,
                       env={**environ, "GIT_CONFIG_GLOBAL": "/dev/null", **(env or {})})
    assert done.returncode == 0, done.stderr
    return done.stdout


def _put(repo, files: dict) -> None:
    for path, data in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_bytes(data)


def _commit_in(repo, message, author="web") -> str:
    who = {"GIT_AUTHOR_NAME": author, "GIT_AUTHOR_EMAIL": "w@x", "GIT_COMMITTER_NAME": author, "GIT_COMMITTER_EMAIL": "w@x"}
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message, env=who)
    return _git(repo, "rev-parse", "HEAD").strip()


def _files_in(repo, sha) -> dict:
    """{path: blob sha} at a commit."""
    out = {}
    for line in _git(repo, "ls-tree", "-r", sha).splitlines():
        meta, path = line.split("\t", 1)
        out[path] = meta.split()[2]
    return out


@pytest.fixture
def local(github, files, tmp_path):
    """The Action's shape: origin holds the card repo; the clone is blobless and sparse (PSDs and PNGs by REST)."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "uploadpack.allowFilter", "true")
    _put(origin, files)
    _commit_in(origin, "Website edit: test")
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", "--filter=blob:none", "--sparse", f"file://{origin}", str(clone))
    _git(clone, "sparse-checkout", "set", "--no-cone", *SPARSE)
    return SimpleNamespace(origin=origin, clone=clone, head=_git(clone, "rev-parse", "HEAD").strip())


def test_local_makes_one_local_commit_and_pushes_nothing(github, local):
    """Pending edits -> one commit B on the clone's head, the checkout on it; origin and GitHub untouched; the small files
    from the clone, only the PSDs and PNGs by REST; the same files as the live run's commit."""
    lacks = run_process(["git", "-C", str(local.clone), "cat-file", "-e", _sha((local.origin / DOOMER).read_bytes())],
                        capture_output=True, env={**environ, "GIT_NO_LAZY_FETCH": "1"})
    assert lacks.returncode != 0                                                                # blobless: no PSD here
    text, commits = cb.run(local=str(local.clone))
    assert len(commits) == 1
    b = commits[0]
    assert _git(local.clone, "rev-parse", "HEAD").strip() == b and _git(local.clone, "rev-parse", f"{b}^").strip() == local.head
    assert _git(local.clone, "rev-list", "--count", "origin/main..HEAD").strip() == "1"
    assert _git(local.origin, "rev-parse", "main").strip() == local.head                       # nothing pushed
    assert _git(local.clone, "log", "-1", "--format=%an|%s", b).strip() == "AutoCroissant|Card edits: 1 redundant"
    assert not [m for m, _ in github.calls if m != "GET"] and "activity" in [r for _, r in github.calls]   # the recent gate as ever
    assert not [r for _, r in github.calls if r.startswith(("git/ref", "git/commits", "git/trees"))]       # the tree from the clone
    rest = sorted(r for _, r in github.calls if r.startswith("git/blobs/"))
    cards = {path: data for path, data in github.files_at(github.head).items()
             if path.startswith(("Auxiliary", "Field/1")) or path == SNACK}                     # A Snack's: the redundancy check
    assert rest == sorted(f"git/blobs/{_sha(data)}" for data in cards.values())                # only what the clone lacks
    assert f"One local commit: `{b}` Card edits: 1 redundant (not pushed)." in text and f"- Local: `{local.clone}`" in text
    assert "The checkout is on it." in text
    edits = (local.clone / cb.EDITS_PATH).read_bytes()
    assert _sha(edits) == _files_in(local.clone, b)[cb.EDITS_PATH]                         # the checkout is on B

    before = github.head                                                                        # the live run, for comparison
    cb.run()
    assert json.loads(edits) == _edits_at(github)
    live = sorted(p for p, data in github.files_at(github.head).items() if github.files_at(before).get(p) != data)
    ours = _git(local.clone, "diff-tree", "--no-commit-id", "-r", "--name-only", b).splitlines()
    assert ours == live == [cb.EDITS_PATH]


def test_local_dry_run_and_a_dirty_checkout(github, local, tmp_path):
    """--dry-run makes no B; a local change in B's way fails the run (exit 1) and leaves the checkout as it was."""
    text, commits = cb.run(dry_run=True, local=str(local.clone))
    assert commits == [] and "Would commit: Card edits: 1 redundant." in text
    assert _git(local.clone, "rev-parse", "HEAD").strip() == local.head and not _git(local.clone, "status", "--porcelain")
    path = local.clone / cb.EDITS_PATH
    path.write_bytes(path.read_bytes() + b" ")
    out = StringIO()
    with contextlib.redirect_stdout(out):
        assert cb.main(["--local", str(local.clone)]) == 1
    assert "Nothing committed: error: git merge in" in out.getvalue() and "left unreferenced)" in out.getvalue()
    assert _git(local.clone, "rev-parse", "HEAD").strip() == local.head and path.read_bytes().endswith(b"}\n ")


def test_local_head_the_clone_lacks_is_a_failed_run(github, local, monkeypatch):
    """--head main moved past the checkout: a report and exit 1, no B and no traceback (card-sync goes on from main)."""
    out, err = StringIO(), StringIO()
    monkeypatch.setattr(cb, "stderr", err)
    with contextlib.redirect_stdout(out):
        assert cb.main(["--local", str(local.clone), "--head", "f" * 40]) == 1
    out, err = out.getvalue(), err.getvalue()
    assert "**Nothing baked: git rev-parse in" in out and "card_bake: Nothing baked:" in err and "Traceback" not in out + err
    assert _git(local.clone, "rev-parse", "HEAD").strip() == local.head


def test_local_recent_gate_reads_the_clone(local, monkeypatch):
    """--local: a head GitHub lacks (a previous B) is diffed in the clone, renames as both paths; else the compare API."""
    _put(local.clone, {"TTS/a.txt": b"a"})
    base = _commit_in(local.clone, "add")
    _git(local.clone, "mv", "TTS/a.txt", "TTS/b.txt")
    head = _commit_in(local.clone, "rename")
    now = datetime.now(timezone.utc)
    pushes = [{"before": base, "after": "x" * 40, "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ")}]
    asked = []

    def api(repository, method, path, **kwargs):
        asked.append(path)
        return Reply(body=pushes) if path == "activity" else Reply(body={"files": [{"filename": "c.psd"}]})
    monkeypatch.setattr(cb, "api", api)
    assert cb.recent_paths(cb.CARD_REPO, head, now, str(local.clone)) == {"TTS/a.txt", "TTS/b.txt"} and asked == ["activity"]
    pushes[0]["before"] = "e" * 40                                                      # a base the clone lacks: the API
    assert cb.recent_paths(cb.CARD_REPO, head, now, str(local.clone)) == {"c.psd"} and asked[1:] == ["activity", f"compare/{'e' * 40}...{head}"]


@fork
def test_local_bake_lands_psd_png_and_file_in_b(github, local):
    """A real bake with --local: B holds the PSD, the PNG and the file; the verdict names B's PSD blob."""
    data = json.loads((local.clone / cb.EDITS_PATH).read_bytes())
    data["cards"] = {"Mini Doomer": {"hp": 8, "note": "checked", "_by": "web", "_at": AT}}
    _put(local.clone, {cb.EDITS_PATH: cb.dump_edits(data).encode("utf-8")})
    _commit_in(local.clone, "Website edit: Mini Doomer")
    text, (b,) = cb.run(local=str(local.clone))
    assert _git(local.clone, "log", "-1", "--format=%B", b).strip() == "Card edits: 1 baked\n\n- Mini Doomer: hp 5 -> 8\n\nEdit-by: web"
    changed = _git(local.clone, "diff-tree", "--no-commit-id", "-r", "--name-only", b).splitlines()
    assert changed == sorted([DOOMER[:len(DOOMER) - 4] + ".png", DOOMER, cb.EDITS_PATH])
    verdict = _verdict("Mini Doomer")
    assert verdict["state"] == "baked" and verdict["psd"] == _files_in(local.clone, b)[DOOMER][:8]
    assert not (local.clone / DOOMER).exists()                                                  # sparse: not checked out
    assert not [m for m, _ in github.calls if m != "GET"]


# ========================
# The Action's sparse copy
# ========================
def test_workflow_pending_edits_line_calls_due(github, capsys, monkeypatch):
    """card-sync.yml's Pending edits line, run as written: the due card names, one per line (recent ones wait); its
    Leftover edits script (the chain's counts): due, then held by a recent push."""
    if not WORKFLOW.exists():
        pytest.skip(f"no {WORKFLOW}")
    text = WORKFLOW.read_text(encoding="utf-8")
    code = re.search(r"Pending edits.*?python\" -c '([^']+)'", text, re.S).group(1)
    github.recent.add(DOOMER)
    monkeypatch.setattr("sys.argv", ["-c", cb.CARD_REPO, github.head, "", str(DB[0] / "verdicts.json")])
    exec(code, {})
    assert capsys.readouterr().out.splitlines() == ["A Snack", "A Snack2", "Nobody", "The Freezer"]
    left = re.search(r"name: Leftover edits.*?<<'PY'\n(.*?)\n *PY\n", text, re.S)
    if left:                                    # TTSCardMaker work/chain
        monkeypatch.setattr("sys.argv", ["-", cb.CARD_REPO, github.head, "", str(DB[0]), str(DB[0] / "none.json")])
        exec(textwrap.dedent(left.group(1)), {})
        assert capsys.readouterr().out.split() == ["4", "1"]



# ========================
# Parts: one box per layer (docs/Future/PARTS_EDITOR.md §4, §11 cases 4 and 5)
# ========================
MIRACLE = "Creatures/Kirby/3 Stars/Miracle_Matter.psd"            # five Ability layers, padded ('\r    \r    Earth: ...')


@fork_text
def test_parts_bake_deletes_skips_and_keeps_photoshop_work(github, clone):
    """Miracle Matter: 1368 edited (only its record, its indent kept), 417 deleted (its record and story gone, the later
    TextIndex values one lower), 1366 changed in Photoshop since (wrong sha: skipped, kept in _psd.skipped), 1367 done in
    Photoshop (redundant), a layer gone; the derived ability leaves with the last ability part."""
    from psd_tools import PSDImage
    if not (clone / pe.STATUS_CHART).exists():
        pytest.skip("no status chart")
    files = github.files_at(github.head)
    files.update({MIRACLE: (clone / MIRACLE).read_bytes(), MIRACLE[:-4] + ".png": (clone / (MIRACLE[:-4] + ".png")).read_bytes(),
                  pe.STATUS_CHART: (clone / pe.STATUS_CHART).read_bytes()})
    old = PSDImage.open(str(clone / MIRACLE))
    ids = {layer.layer_id: layer for layer in old.descendants()}
    psd8 = _sha(files[MIRACLE])[:8]

    def part(lyid, **kw):
        return {"field": "ability", **kw, "was": "x", "sha": pe.raw_sha(ids[lyid]) if lyid in ids else "0123abcd",
                "psd": psd8, "layer": "Text/Stat Txt/ability/Ability"}
    parts = {"1368": part(1368, text="Earth: 5/8/6/5 and heal 3 HP"), "417": part(417, delete=True),
             "1366": {**part(1366, text="Gains nothing."), "sha": "00000000"},
             "1367": {**part(1367, text="(1GCD) You may choose one of the following forms:"), "sha": "00000000"},
             "99999": part(99999, text="A layer remade in Photoshop.")}
    files[cb.EDITS_PATH] = _edits({"Miracle Matter": {"ability": "joined", "parts": parts, "_by": "web", "_at": AT}})
    github.commit(files, "web", "Website edit: Miracle Matter")
    text, commits = cb.run()
    assert len(commits) == 1, text
    message = github.commits[commits[0]]["message"]
    assert "- Miracle Matter: ability (3rd layer) 'heal 2 HP' -> 'heal 3 HP', 4th ability layer deleted; " in message, message
    assert "Miracle Matter" not in _edits_at(github)["cards"]  # the parts and the derived ability are out
    verdict = _verdict("Miracle Matter")                    # the flags stay (on data)
    assert verdict["parts"] == {"1366": "skipped", "1367": "redundant", "1368": "baked", "417": "deleted", "99999": "skipped"}
    assert verdict["skipped"]["1366"] == {"text": "Gains nothing.", "why": pe.CHANGED,
                                          "now": "This creature gains the type of whatever form it is in until the end of the turn."}
    assert verdict["skipped"]["99999"]["why"] == pe.GONE and "2 layers skipped" in verdict["reason"]
    from io import BytesIO
    new = PSDImage.open(BytesIO(github.files_at(github.head)[MIRACLE]))
    now = {layer.layer_id: layer for layer in new.descendants()}
    assert 417 not in now and now[1368].text == "\r    \r    Earth: 5/8/6/5 and heal 3 HP\r   "
    assert all(now[k].text == ids[k].text for k in (1366, 1367, 1369, 422))     # Photoshop's work stands
    assert len(pe.stories(new)) == len(pe.stories(old)) - 1 and pe.stale_stories(new) == []
    assert sorted(pe.story_index(layer) for layer in new.descendants() if layer.kind == "type") == list(range(9))
    report = "\n".join(line for line in text.splitlines() if "Miracle Matter" in line)
    assert "flag bytes" in report


@fork_text
def test_a_tagged_psd_bakes_each_picture_it_shows_in(github, clone, tmp_path):
    """§11 case 7 on A Snack tagged as §9 (the fixture script's tags, saved on a copy): the shared Ability writes all
    three PNGs; a 'Name #2' edit writes only A_Snack2.png, the others byte for byte; the pictures' entries lose their
    derived fields with the parts (but a website-only one), and are never baked on their own."""
    import sys
    from psd_tools import PSDImage
    sys.path.insert(0, str(Path(__file__).parent / "fixtures" / "parts"))
    from make_fixtures import CARDS
    rel, _, tags = CARDS["a_snack_tagged"]
    if not (clone / pe.STATUS_CHART).exists():
        pytest.skip("no status chart")
    psd = PSDImage.open(str(clone / rel))
    for layer in psd.descendants():
        if layer.layer_id in tags:
            layer.name = tags[layer.layer_id]
    psd.save(str(tmp_path / "A_Snack.psd"))
    pngs = [rel[:-4] + suffix + ".png" for suffix in ("", "2", "3")]
    files = github.files_at(github.head)
    files.update({rel: (tmp_path / "A_Snack.psd").read_bytes(), pe.STATUS_CHART: (clone / pe.STATUS_CHART).read_bytes(),
                  **{png: (clone / png).read_bytes() for png in pngs}})
    github.commit(files, "someone", "A Snack tagged")
    fixture = json.loads((Path(__file__).parent / "fixtures" / "parts" / "card_edits.parts.json").read_text("utf-8"))["cards"]
    ability = fixture["A Snack"]["parts"]["320"]
    for edit, changed in (({"A Snack": {"ability": ability["text"], "parts": {"320": ability}},
                            "A Snack2": {"ability": ability["text"]}, "A Snack3": {"ability": "Site", "_site": ["ability"]}}, pngs),
                          ({"A Snack": {"parts": {"1023": fixture["A Snack"]["parts"]["1023"]}},
                            "A Snack2": {"name": "Second Breakfast"}}, pngs[1:2])):
        before = github.files_at(github.head)
        before[cb.EDITS_PATH] = _edits({k: {**v, "_by": "web", "_at": AT} for k, v in edit.items()})
        github.commit(before, "web", "Website edit: A Snack")
        text, commits = cb.run()
        assert len(commits) == 1, text
        after = github.files_at(github.head)
        assert sorted(p for p in after if after[p] != before.get(p)) == sorted([rel, cb.EDITS_PATH, *changed]), text
        cards = _edits_at(github)["cards"]
        assert "A Snack2" not in cards and "A Snack" not in cards, cards
        assert cards.pop("A Snack3", {}).get("_site") == edit.get("A Snack3", {}).get("_site"), cards    # website only: kept
    saved = {layer.layer_id: layer for layer in PSDImage.open(str(tmp_path / "A_Snack.psd")).descendants()}
    from io import BytesIO
    new = {layer.layer_id: layer for layer in PSDImage.open(BytesIO(github.files_at(github.head)[rel])).descendants()}
    assert new[1023].text.startswith("Second\rBreakfast") and new[1023].name == "Name #2"
    assert {k: layer.visible for k, layer in new.items()} == {k: layer.visible for k, layer in saved.items()}


def test_a_stars_edit_moves_a_card_starred_by_its_folder(github):
    """§11 case 8: The Freezer's stars 1 -> 3 move its PSD and PNG into "Field/3 Stars" (created, told): one commit adds
    them there and deletes them here; its printed star (no Stars group) is not redrawn, told; the verdict says where."""
    files = github.files_at(github.head)
    moved = {path: path.replace("1 Stars", "3 Stars") for path in (FREEZER, FREEZER[:-4] + ".png")}
    files[cb.EDITS_PATH] = _edits({"The Freezer": {"stars": 3, "_by": "web", "_at": AT}})
    github.commit(files, "web", "Website edit: The Freezer")
    text, commits = cb.run()
    assert len(commits) == 1, text
    message = github.commits[commits[0]]["message"]
    assert message.startswith("Card edits: 1 baked\n\n- The Freezer: stars 1 -> 3 (moved to 3 Stars; created the folder "
                              "'Field/3 Stars'; the printed stars are not redrawn: "), message
    after = github.files_at(github.head)
    assert all(old not in after and after[new] == files[old] for old, new in moved.items())   # the same bytes, moved
    verdict = _verdict("The Freezer")
    assert verdict["moved"] == "Field/3 Stars" and verdict["psd"] == _sha(files[FREEZER])[:8]


@fork_text
def test_a_rename_bake_moves_the_files_and_everything_keyed_by_them(github, local):
    """The editor's "Also rename the card's files" (owner 10-25) on a scratch repo: Mini Doomer's name edit with
    `_rename` bakes the title and renames its PSD and PNG in B, with the files keyed by the card (its edits entry, the
    old name an alias, its authors line, creation date, ratings, the decks' rules); the Renamed trailers let the history
    follow the renamed and redrawn PSD as one card. The Freezer renamed to that name in the same run is refused."""
    from commands.card_history import GitIndex
    from conftest import blob_sha_bytes
    png, new = DOOMER[:len(DOOMER) - 4] + ".png", "Auxiliary/Minions/Mini_Destroyer"
    side = {cb.EDITS_PATH: _edits({"Mini Doomer": {"name": "Mini Destroyer", "_rename": True, "_by": "web", "_at": AT},
                                   "The Freezer": {"name": "Mini Destroyer", "_rename": True, "_by": "web", "_at": AT}}),
            cb.AUTHORS_PATH: b"[people]\nweb = Web\n\n[cards]\nMini Doomer = Chestnut\nThe Freezer = Web\n",
            cb.CREATED_PATH: _edits({"Mini Doomer": {"date": "2020-07-26", "from": "discord"}}),
            cb.RATINGS_PATH: _edits({"Mini Doomer": [{"date": "2023-04-28", "scale": "power", "value": 5, "by": "group",
                                                      "source": "TTS session"}]}),
            cb.RULES_PATH: json.dumps({"decks": {"d1": {"rule": {"include": [png]}, "card_list": [png]}}}, indent=2).encode()}
    _put(local.origin, side)
    _commit_in(local.origin, "Website edit: Mini Doomer")
    _git(local.clone, "pull", "-q", "--ff-only")
    for data in side.values():
        github.blobs[blob_sha_bytes(data)] = data                                # REST for what the sparse clone lacks
    text, (b,) = cb.run(local=str(local.clone))
    message = _git(local.clone, "log", "-1", "--format=%B", b)
    assert message.startswith("Card edits: 1 baked\n\n- Mini Doomer: name 'Mini Doomer' -> 'Mini Destroyer', renamed: Mini "
                              "Doomer -> Mini Destroyer"), message
    assert f"Renamed: {DOOMER} -> {new}.psd\nRenamed: {png} -> {new}.png" in message
    tree = _files_in(local.clone, b)
    assert DOOMER not in tree and png not in tree and {f"{new}.psd", f"{new}.png"} <= set(tree)
    show = lambda path: _git(local.clone, "show", f"{b}:{path}")
    entry = json.loads(show(cb.EDITS_PATH))["cards"]
    assert "another card of this run takes the name" in _verdict("The Freezer")["reason"]    # one name, one card
    assert entry.pop("The Freezer")["_rename"] and list(entry) == ["Mini Destroyer"] and entry["Mini Destroyer"]["aliases"] == ["Mini Doomer"]   # name baked, _rename gone
    assert entry["Mini Destroyer"]["_by"] == "AutoCroissant" and set(entry["Mini Destroyer"]) == {"aliases", "_by", "_at"}
    assert "Mini Destroyer = Chestnut\nThe Freezer = Web" in show(cb.AUTHORS_PATH)
    assert list(json.loads(show(cb.CREATED_PATH))["cards"]) == list(json.loads(show(cb.RATINGS_PATH))["cards"]) == ["Mini Destroyer"]
    assert json.loads(show(cb.RULES_PATH))["decks"]["d1"]["card_list"] == [f"{new}.png"]
    verdict = _verdict("Mini Destroyer")
    assert verdict["renamed"] == "Mini Destroyer" and verdict["psd"] == tree[f"{new}.psd"][:8] and _verdict("Mini Doomer") is None
    index = GitIndex.from_git(str(local.clone), b)
    assert index.renamed_from[f"{new}.psd"] == DOOMER and index.origin(f"{new}.png") == png
