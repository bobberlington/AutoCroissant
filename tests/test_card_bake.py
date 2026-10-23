"""
card_bake: the edit.js editsFile port (today's file round-trips byte-identically; node cross-check), and the driver
against a fake GitHub (tree, blobs, refs, commits: the test_update_stats pattern) for the gates of EDIT_PIPELINE_DESIGN
sections 5 and 11: recent, duplicate, blob, classify, edit-changed, head-moved (409 and 422), one commit per run with a
moved card left out, the schedule gate (pending), the SSH fallback for a token that can't write, --dry-run, --card. A
real bake (a stat, and an ability with an FX layer) needs the fork's renderer; those tests skip under stock psd-tools.
The Action-shaped import check copies exactly the files card-edits.yml checks out of this repo and runs --help there
without config.py, with an allowlist of third-party imports. Fork as conftest.find_fork finds it; a Python without
requests skips this file:
    AC_PSD_TOOLS_SRC=<fork>/src ~/Documents/Tools/psd-tools/.venv/bin/python -m pytest -q tests/test_psd_edit.py tests/test_card_bake.py
"""
import ast
import contextlib
import hashlib
import json
import re
import shutil
from datetime import datetime, timedelta, timezone
from io import StringIO
from os import environ
from subprocess import run as run_process
from sys import executable
from types import SimpleNamespace

import pytest

pytest.importorskip("requests", reason="card_bake imports requests (absent from this Python)")
import commands.card_bake as cb  # noqa: E402
import commands.git_commit as gc  # noqa: E402
import commands.github_api as ga  # noqa: E402
import commands.psd_edit as pe  # noqa: E402
from conftest import (CLONE, PLAIN_DOOMER, REPO, Reply, at_rev, blob_sha_bytes as _sha, fork_reason, story_sync as _story_sync,  # noqa: E402
                      text_engine as _text_engine)
from golden_cards import VOCAB  # noqa: E402

DOOMER = "Auxiliary/Minions/Mini_Doomer.psd"
FREEZER = "Field/1 Stars/The_Freezer.psd"
RED_KNIFE = "Auxiliary/Items/Red_Knife.psd"            # its title misses the parity bar (outer glow noise)
SNACK = "Items/Heal/2 Stars/A_Snack.psd"
LIVE_EDITS = CLONE / "site" / "Card Edits.json"
EDIT_JS = CLONE / "site" / "assets" / "edit.js"
WORKFLOW = CLONE / ".github" / "workflows" / "card-edits.yml"
AT = "2026-10-20T18:41:54Z"
ABOUT = "Card data fixes from the website's Edit button."
TOKEN = "test-token"


fork = pytest.mark.skipif(not _text_engine(), reason="a bake needs the fork's renderer: " + fork_reason())
fork_text = pytest.mark.skipif(not (_text_engine() and _story_sync()),
                               reason="a text bake needs the fork's Txt2 story sync: " + fork_reason())


# ========================
# Fake GitHub
# ========================
class FakeGitHub:
    """A repo with git objects: blobs, trees, commits and one branch. recent / conflicts / broken simulate the races."""

    def __init__(self, repo: str, files: dict, author="web", message="Website edit: test"):
        self.repo = repo
        self.blobs: dict = {}
        self.trees: dict = {}
        self.commits: dict = {}
        self.recent: set = set()            # paths pushed "in the last 30 minutes"
        self.conflicts = 0                  # PATCH ref answers conflict_status this many times, someone else's push landing first
        self.stuck = False                  # PATCH ref answers 422 with the branch unmoved (a ruleset)
        self.conflict_status = 422          # ... 409 or 422 (both mean "not a fast forward")
        self.conflict_on = ""               # ... only for commits whose message starts so ("" = any)
        self.broken: dict = {}              # blob sha -> the bytes served instead
        self.down: set = set()              # blob shas answering 503
        self.rate_limited = False           # POST git/blobs answers a rate-limit 403
        self.forbidden = False              # POST git/blobs answers 403 without rate-limit headers: the token can't write
        self.calls: list = []
        self.head = self.commit(files, author, message)

    def commit(self, files: dict, author: str, message: str, parents=None) -> str:
        """A commit of {path: bytes} on the branch (parents default to the head), returning its sha."""
        tree = {}
        for path, data in files.items():
            self.blobs[_sha(data)] = data
            tree[path] = ("100644", _sha(data))
        tree_sha = self._tree(tree)
        parents = [self.head] if parents is None and self.commits else list(parents or [])
        sha = hashlib.sha1(json.dumps([tree_sha, parents, message, author]).encode()).hexdigest()
        self.commits[sha] = {"sha": sha, "tree": {"sha": tree_sha}, "parents": [{"sha": p} for p in parents],
                             "message": message, "author": {"name": author, "email": f"{author}@x"}}
        self.head = sha
        return sha

    def files_at(self, sha: str) -> dict:
        """{path: bytes} at a commit."""
        return {path: self.blobs[blob] for path, (_, blob) in self.trees[self.commits[sha]["tree"]["sha"]].items()}

    def _tree(self, tree: dict) -> str:
        sha = hashlib.sha1(json.dumps(sorted(tree.items())).encode()).hexdigest()
        self.trees[sha] = dict(tree)
        return sha

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def request(self, method, url, headers=None, timeout=None, json=None, params=None, stream=False):
        route, _, query = url.split(f"/repos/{self.repo}/", 1)[1].partition("?")
        self.calls.append((method, route))
        if method == "GET" and route == "git/ref/heads/main":
            return Reply(body={"object": {"sha": self.head}})
        if method == "GET" and route.startswith("git/commits/"):
            sha = route.rsplit("/", 1)[1]
            return Reply(body=self.commits[sha]) if sha in self.commits else Reply(404, body={"message": "Not Found"})
        if method == "GET" and route.startswith("git/trees/"):
            sha = route.rsplit("/", 1)[1]
            if sha not in self.trees:
                return Reply(404, body={"message": "Not Found"})
            return Reply(body={"sha": sha, "truncated": False, "tree": [
                {"path": p, "mode": mode, "type": "blob", "sha": blob} for p, (mode, blob) in sorted(self.trees[sha].items())]})
        if method == "GET" and route.startswith("git/blobs/"):
            sha = route.rsplit("/", 1)[1]
            if sha not in self.blobs:
                return Reply(404, body={"message": "Not Found"})
            if sha in self.down:
                return Reply(503, body={"message": "Service Unavailable"})
            return Reply(data=self.broken.get(sha, self.blobs[sha]))
        if method == "GET" and route == "activity":
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            return Reply(body=[{"before": "b" * 40, "after": self.head, "timestamp": now}] if self.recent else [])
        if method == "GET" and route.startswith("compare/"):
            return Reply(body={"files": [{"filename": path} for path in sorted(self.recent)]})
        if method == "POST" and route == "git/blobs":
            from base64 import b64decode
            if self.rate_limited:
                return Reply(403, body={"message": "API rate limit exceeded"}, headers={"x-ratelimit-remaining": "0"})
            if self.forbidden:
                return Reply(403, body={"message": "Resource not accessible by integration"})
            data = b64decode(json["content"])
            self.blobs[_sha(data)] = data
            return Reply(201, body={"sha": _sha(data)})
        if method == "POST" and route == "git/trees":
            tree = dict(self.trees[json["base_tree"]])
            for entry in json["tree"]:
                tree[entry["path"]] = (entry["mode"], entry["sha"])
            return Reply(201, body={"sha": self._tree(tree)})
        if method == "POST" and route == "git/commits":
            sha = hashlib.sha1(json_dumps(json).encode()).hexdigest()
            self.commits[sha] = {"sha": sha, "tree": {"sha": json["tree"]}, "parents": [{"sha": p} for p in json["parents"]],
                                 "message": json["message"], "author": dict(json.get("author") or {"name": "token"})}
            return Reply(201, body={"sha": sha})
        if method == "PATCH" and route == "git/refs/heads/main":
            commit = self.commits.get(json["sha"])
            if self.stuck:
                return Reply(422, body={"message": "Repository rule violations found"})
            if self.conflicts and (commit or {}).get("message", "").startswith(self.conflict_on):
                self.conflicts -= 1
                self.commit({**self.files_at(self.head), "elsewhere.txt": b"%d" % self.conflicts}, "web", "Website edit: elsewhere")
                return Reply(self.conflict_status, body={"message": "Update is not a fast forward"})
            if not commit or [p["sha"] for p in commit["parents"]] != [self.head]:
                return Reply(422, body={"message": "Update is not a fast forward"})
            self.head = json["sha"]
            return Reply(body={"object": {"sha": self.head}})
        return Reply(404, body={"message": f"no route for {method} {route}"})


def json_dumps(value) -> str:
    return json.dumps(value, sort_keys=True)


# ========================
# Fixtures
# ========================
def _card_files(clone, rel: str) -> dict:
    png = rel[:len(rel) - 4] + ".png"
    if not (clone / rel).exists() or not (clone / png).exists():
        pytest.skip(f"{rel} not in the clone")
    return {rel: (clone / rel).read_bytes(), png: (clone / png).read_bytes()}


def _edits(cards: dict) -> bytes:
    return cb.dump_edits({"about": ABOUT, "cards": cards}).encode("utf-8")


EDITS = {
    "Beldam": {"series": "Mario", "_by": "web", "_at": AT},                 # no PSD field: not looked at
    "Mini Doomer": {"hp": 5, "note": "checked", "_by": "web", "_at": AT},   # hp 5 is on the card: redundant, dropped
    "Nobody": {"hp": 3, "_by": "web", "_at": AT},                           # no PSD
    "The Freezer": {"stars": 2, "_by": "web", "_at": AT},                   # stars: not yet
    "A Snack": {"hp": 1, "_by": "web", "_at": AT},                          # numbered PNGs share its PSD
    "A Snack2": {"ability": "Breakfast.", "_by": "web", "_at": AT},         # a numbered PNG itself
}


@pytest.fixture
def files(clone):
    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
    out = {f"Types/{t}.png": t.encode() for t in vocab["all_types"]}
    out.update({f"Types/Stars/{s}.png": s.encode() for s in vocab["stars"]})
    out[cb.ICON_MANIFEST_PATH] = json.dumps(vocab["manifest"]).encode()
    out.update(_card_files(clone, DOOMER))
    out.update(_card_files(clone, FREEZER))
    out[SNACK] = out[DOOMER]                    # read by the redundancy check before the duplicate gate (hp 5, not 1)
    out[SNACK[:len(SNACK) - 4] + ".png"] = b"not fetched"
    out["Items/Heal/2 Stars/A_Snack2.png"] = b"a duplicate"
    out["MDW/Templates/Creature_Template.psd"] = b"excluded folder"
    out[cb.EDITS_PATHS[0]] = _edits(EDITS)
    return out


@pytest.fixture
def github(monkeypatch, tmp_path, files, clone):
    fake = FakeGitHub(cb.CARD_REPO, files)
    monkeypatch.setattr(ga, "get", fake.get)
    monkeypatch.setattr(ga, "GIT_TOKEN", "")
    monkeypatch.setattr(ga, "HEADERS", {})
    monkeypatch.setattr(gc, "request", fake.request)
    monkeypatch.setattr(gc, "refresh", lambda: True)
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    monkeypatch.setenv(cb.WORK_ENV, str(tmp_path))
    monkeypatch.setenv(cb.FONT_DIRS_ENV, str(clone / pe.FONTS_DIR))
    monkeypatch.setattr(gc, "cant_write", False)
    return fake


def _edits_at(github, sha=None) -> dict:
    return json.loads(github.files_at(sha or github.head)[cb.EDITS_PATHS[0]].decode("utf-8"))


# ========================
# The editsFile port
# ========================
def test_live_edits_file_round_trips_byte_identical():
    """S4's gate: today's site/Card Edits.json through the port, unchanged (read-only)."""
    if not LIVE_EDITS.exists():
        pytest.skip(f"no {LIVE_EDITS}")
    text = LIVE_EDITS.read_text(encoding="utf-8")
    assert cb.dump_edits(json.loads(text)) == text


def test_collation_is_locale_compare_base():
    keys = ["Zed", "eclair2", "Éclair", "ab", "a b", "PoD12BAD", "PoD11ZOMG", "Panda", "P.A.N.D.A. Mk A", "Bobbery",
            "Bob-ulk", "Block Hopper", "The Evil, Fricked Up Cursed Idol", "The Evil Fricked", "Mom's Knife", "Moms Knife"]
    assert sorted(keys, key=cb._collate) == ["a b", "ab", "Block Hopper", "Bob-ulk", "Bobbery", "Éclair", "eclair2",
                                             "Mom's Knife", "Moms Knife", "P.A.N.D.A. Mk A", "Panda", "PoD11ZOMG", "PoD12BAD",
                                             "The Evil Fricked", "The Evil, Fricked Up Cursed Idol", "Zed"]   # node's localeCompare


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


def test_rewrite_entry_layout_matches_edit_js(tmp_path):
    """Dropping a field lays the entry out as editsFile does: ORDER, other keys, _by, _at (kept here, rewritten there)."""
    entry = {"_at": AT, "spd": 4, "unknown": 1, "hp": 8, "_by": "tester", "name": "X", "_psd": {"state": "baked", "for": {"hp": 8}, "psd": "01234567"}}
    text = json.dumps({"about": ABOUT, "cards": {"Card A": entry, "card b": {"hp": 1}}})
    ours = cb.rewrite_edits(text, "card_a", ["hp"], entry["_psd"])         # the verdict kept (None would remove it)
    theirs = _node(tmp_path, text, [{"key": "Card A", "set": {}, "unset": ["hp"]}], AT, "tester")
    assert ours == theirs
    assert json.loads(ours)["cards"]["Card A"] == {"name": "X", "spd": 4, "unknown": 1, "_psd": entry["_psd"], "_by": "tester", "_at": AT}
    assert json.loads(cb.rewrite_edits(text, "card_a", ["hp"]))["cards"]["Card A"] == {"name": "X", "spd": 4, "unknown": 1, "_by": "tester", "_at": AT}
    entry = {"hp": 8, "_psd": entry["_psd"], "unknown": 1, "other": [2], "name": "X", "_by": "tester", "_at": AT}   # _psd first among the other keys
    text = json.dumps({"about": ABOUT, "cards": {"Card A": entry}})
    ours = cb.rewrite_edits(text, "card_a", ["hp"], {**entry["_psd"], "at": AT})
    assert ours == _node(tmp_path, text, [{"key": "Card A", "set": {"_psd": {**entry["_psd"], "at": AT}}, "unset": ["hp"]}], AT, "tester")
    assert list(json.loads(ours)["cards"]["Card A"]) == ["name", "_psd", "unknown", "other", "_by", "_at"]


def test_rewrite_entry_rules():
    verdict = {"state": "refused", "for": {"stars": 2}, "psd": "abcdef01", "reason": "unsupported-card: stars: not yet", "at": AT}
    entry = {"stars": 2, "note": "n", "_by": "w", "_at": AT}
    assert cb.rewrite_entry(entry, (), verdict) == {"stars": 2, "note": "n", "_psd": verdict, "_by": "w", "_at": AT}
    assert cb.rewrite_entry({"hp": 8, "_by": "w", "_at": AT, "_psd": verdict}, ["hp"]) is None       # the entry goes whole
    assert cb.rewrite_entry({"name": "X", "hp": 8, "_by": "w"}, ["hp"], None) == {"name": "X", "_by": "w"}   # name stays
    assert cb.rewrite_entry({"_psd": verdict, "hp": 8}, (), None) == {"hp": 8}                       # None removes _psd
    text = cb.dump_edits({"about": ABOUT, "cards": {"Mini_Doomer": {"hp": 8, "note": "n"}, "Other": {"hp": 1}}})
    assert cb.rewrite_edits(text, "mini doomer", ["hp"]) == cb.dump_edits({"about": ABOUT, "cards": {"Mini_Doomer": {"note": "n"}, "Other": {"hp": 1}}})
    assert cb.rewrite_edits(text, "mini doomer", ["spd"]) is None and cb.rewrite_edits(text, "nobody", ["hp"]) is None
    assert cb.rewrite_edits(text, "other", ["hp"]) == cb.dump_edits({"about": ABOUT, "cards": {"Mini_Doomer": {"hp": 8, "note": "n"}}})
    assert cb.psd_fields({"hp": 8, "spd": None, "name": "X", "series": "m", "creatures": [{"hp": 1}]}) == {"hp": 8, "name": "X"}


# ========================
# The driver against the fake GitHub
# ========================
def test_run_drops_redundant_and_writes_verdicts(github):
    before = github.head
    text, commits = cb.run()
    assert len(commits) == 1 and github.head == commits[0]
    commit = github.commits[commits[0]]
    assert commit["message"] == "Card edits: 1 redundant, 4 refused" and commit["author"]["name"] == "AutoCroissant"
    assert [p["sha"] for p in commit["parents"]] == [before]                                     # one commit for the run
    data = _edits_at(github)
    assert data["cards"]["Mini Doomer"] == {"note": "checked", "_by": "web", "_at": AT}          # hp dropped, no verdict
    assert data["cards"]["Beldam"] == EDITS["Beldam"]
    freezer = data["cards"]["The Freezer"]["_psd"]
    assert freezer["state"] == "refused" and freezer["for"] == {"stars": 2} and freezer["psd"] == _sha(github.files_at(before)[FREEZER])[:8]
    assert freezer["reason"] == "unsupported-card: stars: not yet" and freezer["at"].endswith("Z")
    assert data["cards"]["Nobody"]["_psd"]["reason"] == "unsupported-card: no PSD in the card repo" and data["cards"]["Nobody"]["_psd"]["psd"] == ""
    assert data["cards"]["A Snack"]["_psd"]["reason"] == "duplicate: numbered PNGs share the PSD: A_Snack2.png"
    assert data["cards"]["A Snack2"]["_psd"]["reason"] == "duplicate: a numbered PNG exported from `Items/Heal/2 Stars/A_Snack.psd`"
    raw = github.files_at(github.head)[cb.EDITS_PATHS[0]].decode("utf-8")
    assert raw == cb.dump_edits(json.loads(raw))                                                 # canonical layout
    assert github.files_at(github.head)[DOOMER] == github.files_at(before)[DOOMER]              # no pixels touched
    assert "## Redundant (dropped from the file) (1)" in text and "## Refused (4)" in text
    assert text.count(f"committed `{commits[0][:8]}`") == 5                                    # card-edits.yml greps "committed `"
    assert "**Nothing baked.**" in text and f"One commit: `{commits[0][:8]}` Card edits: 1 redundant, 4 refused." in text
    blobs = [route for method, route in github.calls if route.startswith("git/blobs/")]
    assert len(blobs) == 8      # the edits file + manifest, PSD + PNG of Mini Doomer and The Freezer, A Snack's PSD
    assert [route for _, route in github.calls].count("activity") == 1     # the recent gate reads the pushes once a run
    assert cb.pending() == []                                               # every PSD-kind edit has its final verdict

    # a rerun on the pipeline's own commit: the redundant field is gone, the verdicts are unchanged: nothing to commit
    text, commits = cb.run()
    assert commits == [] and "## Refused (4)" in text and "committed `" not in text and "One commit" not in text

    # a human commit on top: the same
    github.commit(github.files_at(github.head), "web", "Website edit: something else")
    text, commits = cb.run()
    assert commits == [] and "## Refused (4)" in text and "committed `" not in text


def test_dry_run_commits_nothing(github, tmp_path):
    head = github.head
    text, commits = cb.run(dry_run=True)
    assert commits == [] and github.head == head and "(dry run)" in text and "not committed (dry run)" in text
    assert not [m for m, _ in github.calls if m != "GET"]
    work = tmp_path / cb.WORK_NAME
    staged = json.loads((work / "Card Edits.json").read_text(encoding="utf-8"))      # the one file the commit would hold
    assert staged["cards"]["Mini Doomer"] == {"note": "checked", "_by": "web", "_at": AT}
    assert staged["cards"]["The Freezer"]["_psd"]["state"] == "refused"
    assert "Would commit: Card edits: 1 redundant, 4 refused." in text
    assert (work / "report.md").read_text(encoding="utf-8") == text


def test_card_restricts_the_run(github):
    text, commits = cb.run(card="beldam")
    assert commits == [] and "No entry with a PSD field for 'beldam'" in text
    text, commits = cb.run(card="the_freezer")
    assert len(commits) == 1 and github.commits[commits[0]]["message"] == "Card edits: 1 refused"
    data = _edits_at(github)
    assert data["cards"]["Mini Doomer"]["hp"] == 5 and "_psd" not in data["cards"]["A Snack"]
    assert data["cards"]["The Freezer"]["_psd"]["for"] == {"stars": 2} and "1 with PSD fields (--card the_freezer)" in text


def test_no_card_to_bake_needs_no_fonts(github, monkeypatch):
    """The fonts dir (on the jetson: the fonts cache's fill) is found only when a card has PSD fields."""
    def boom(snapshot=None):
        raise cb.Transient("fonts cache: 1 font file failed to download")
    monkeypatch.setattr(cb, "_fonts_dir", boom)
    text, commits = cb.run(card="beldam")
    assert commits == [] and "No entry with a PSD field for 'beldam'" in text and "Runner:" not in text
    with pytest.raises(cb.Transient):
        cb.run(card="the_freezer")


def test_recent_refuses_before_fetching(github):
    github.recent.add(DOOMER)
    text, commits = cb.run(card="Mini Doomer")
    assert commits == [] and "recent: PSD pushed in the last 30 min" in text
    assert not [route for _, route in github.calls if route.startswith("git/blobs/") and route.endswith(_sha(github.blobs[_sha(github.files_at(github.head)[DOOMER])]))]
    assert "_psd" not in _edits_at(github)["cards"]["Mini Doomer"]                 # transient: no verdict
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


def test_stat_edit_with_creatures_is_refused(github):
    """CardEdits.apply sets the stats from creature 1: a baked hp would be undone, so the card is refused before any fetch."""
    files = github.files_at(github.head)
    files[cb.EDITS_PATHS[0]] = _edits({"Mini Doomer": {"hp": 6, "creatures": [{"hp": 7, "def": 1, "atk": 1, "spd": 1}], "_by": "web", "_at": AT}})
    github.commit(files, "web", "Website edit: Mini Doomer")
    text, commits = cb.run(card="Mini Doomer")
    entry = _edits_at(github)["cards"]["Mini Doomer"]
    assert len(commits) == 1 and entry["hp"] == 6 and entry["creatures"][0]["hp"] == 7
    assert entry["_psd"]["reason"] == "unsupported-card: hp: the entry's creatures list sets the stats"
    assert not [route for _, route in github.calls if route == f"git/blobs/{_sha(files[DOOMER])}"]


def test_blob_gate(github):
    psd = _sha(github.files_at(github.head)[DOOMER])
    github.broken[psd] = b"not the PSD"
    text, commits = cb.run(card="Mini Doomer")
    assert "blob: blob %s hashes to" % psd[:8] in text
    assert len(commits) == 1 and _edits_at(github)["cards"]["Mini Doomer"]["_psd"]["reason"].startswith("blob: ")


@pytest.mark.parametrize("status", [409, 422])
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
    files[cb.EDITS_PATHS[0]] = _edits({**EDITS, "Mini Doomer": {"hp": 4, "note": "checked", "_by": "web", "_at": AT}})
    github.commit(files, "web", "Website edit: Mini Doomer")
    text, commits = cb.run(head=snapshot, card="Mini Doomer")
    assert commits == [] and "edit-changed: the edit changed on main meanwhile" in text
    assert _edits_at(github)["cards"]["Mini Doomer"]["hp"] == 4


def test_psd_changed_meanwhile_is_head_moved(github):
    snapshot = github.head
    files = github.files_at(snapshot)
    files[DOOMER] = files[DOOMER] + b"\0"
    github.commit(files, "artist", "Mini Doomer: new art")
    text, commits = cb.run(head=snapshot, card="Mini Doomer")
    assert commits == [] and f"head-moved: {DOOMER} changed on main" in text


def test_moved_psd_leaves_only_its_card_out(github):
    """A card whose PSD changed under the run stays out of the run's commit (a try-again refusal, no verdict); the rest lands."""
    snapshot = github.head
    files = github.files_at(snapshot)
    files[DOOMER] = files[DOOMER] + b"\0"
    github.commit(files, "artist", "Mini Doomer: new art")
    text, commits = cb.run(head=snapshot)
    assert len(commits) == 1 and github.commits[commits[0]]["message"] == "Card edits: 4 refused"
    data = _edits_at(github)
    assert data["cards"]["Mini Doomer"] == {"hp": 5, "note": "checked", "_by": "web", "_at": AT}     # untouched, no verdict
    assert data["cards"]["The Freezer"]["_psd"]["state"] == "refused" and github.files_at(github.head)[DOOMER] == files[DOOMER]
    assert f"**Mini Doomer** `{DOOMER}`: hp 5: head-moved: {DOOMER} changed on main" in text
    assert cb.pending() == ["Mini Doomer"]                                  # the next scheduled run tries it again


def test_rerun_on_its_own_commit_retries_a_try_again_card(github):
    """No skip for the pipeline's own commit: a card refused `recent` there is retried by the next run, with nothing else new."""
    github.recent.add(DOOMER)
    text, commits = cb.run()
    assert len(commits) == 1 and "recent: PSD pushed in the last 30 min" in text and cb.pending() == ["Mini Doomer"]
    github.recent.clear()
    text, commits = cb.run()
    assert len(commits) == 1 and github.commits[commits[0]]["message"] == "Card edits: 1 redundant"
    assert cb.pending() == []


def test_pending_is_the_schedule_gate(github):
    """pending: a PSD-kind edit without a refused verdict for exactly its values on the PSD blob the tree holds now."""
    assert cb.pending() == ["A Snack", "A Snack2", "Mini Doomer", "Nobody", "The Freezer"]     # nothing tried yet
    cb.run()
    assert cb.pending() == []
    files = github.files_at(github.head)
    data = json.loads(files[cb.EDITS_PATHS[0]].decode("utf-8"))
    data["cards"]["Beldam"]["series"] = "Paper Mario"                       # database-only: nothing to bake
    files[cb.EDITS_PATHS[0]] = cb.dump_edits(data).encode("utf-8")
    github.commit(files, "web", "Website edit: Beldam")
    assert cb.pending() == []
    data["cards"]["Nobody"]["hp"] = 4                                       # a refused value changed
    data["cards"]["Mini Doomer"]["atk"] = 2                                 # a new PSD field
    files[cb.EDITS_PATHS[0]] = cb.dump_edits(data).encode("utf-8")
    files[FREEZER] = files[FREEZER] + b"\0"                                 # a refused card's PSD changed
    github.commit(files, "web", "Website edit: Nobody, Mini Doomer; The Freezer's art")
    assert cb.pending() == ["Mini Doomer", "Nobody", "The Freezer"]
    data["cards"]["Mini Doomer"]["_psd"] = {"state": "baked", "for": {"atk": 2}, "psd": "01234567", "reason": "x", "at": AT}
    files[cb.EDITS_PATHS[0]] = cb.dump_edits(data).encode("utf-8")
    github.commit(files, "web", "Website edit: a baked verdict beside a field")
    assert "Mini Doomer" in cb.pending()                                    # only a refusal is final
    files[cb.EDITS_PATHS[0]] = b"not json"
    github.commit(files, "web", "Website edit: broken")
    with pytest.raises(ValueError):                                         # the gate step fails: Discord is told
        cb.pending()


def test_pending_run_skips_final_verdicts(github):
    """--pending (the schedule): a card whose refused verdict still holds is not looked at again; a dispatch re-checks it."""
    cb.run()
    files = github.files_at(github.head)
    data = json.loads(files[cb.EDITS_PATHS[0]].decode("utf-8"))
    data["cards"]["The Freezer"]["_psd"]["reason"] = "stars: changed by hand"   # a re-check would rewrite it
    files[cb.EDITS_PATHS[0]] = cb.dump_edits(data).encode("utf-8")
    github.commit(files, "web", "Website edit: a verdict")
    github.calls.clear()
    text, commits = cb.run(pending_only=True)
    assert commits == [] and "4 with PSD fields, 4 refused before (not re-checked)" in text
    assert not [route for _, route in github.calls if route.startswith("git/blobs/") and route != "git/blobs/" + _sha(files[cb.EDITS_PATHS[0]])]
    text, commits = cb.run()
    assert len(commits) == 1 and "not re-checked" not in text                # the dispatch rewrote The Freezer's verdict
    assert _edits_at(github)["cards"]["The Freezer"]["_psd"]["reason"] != "stars: changed by hand"


def test_png_verdicts_follow_the_pngs(github):
    """A `duplicate` verdict rests on the PNGs by path: removing the numbered PNG reopens both cards, a re-export doesn't."""
    cb.run()
    data = _edits_at(github)
    assert data["cards"]["A Snack"]["_psd"]["pngs"] and data["cards"]["A Snack2"]["_psd"]["pngs"]
    assert "pngs" not in data["cards"]["Nobody"]["_psd"] and cb.pending() == []
    files = github.files_at(github.head)
    files["Items/Heal/2 Stars/A_Snack.png"] = b"re-exported"
    github.commit(files, "web", "A Snack: re-export")
    assert cb.pending() == []
    del files["Items/Heal/2 Stars/A_Snack2.png"]
    github.commit(files, "web", "A Snack2: removed")
    assert cb.pending() == ["A Snack", "A Snack2"]


def test_recent_png_is_refused(github):
    """A teammate's PNG re-export in the last 30 min is theirs too: refused `recent` before the PSD is fetched."""
    github.recent.add(DOOMER[:len(DOOMER) - 4] + ".png")
    text, commits = cb.run(card="Mini Doomer")
    assert commits == [] and "recent: PNG pushed in the last 30 min" in text
    assert "_psd" not in _edits_at(github)["cards"]["Mini Doomer"] and "Mini Doomer" in cb.pending()


def test_redundant_name_is_dropped(github):
    """A name the card prints is redundant like any field (the bot reads the title): dropped with its verdict, once."""
    baked = {"state": "baked", "for": {"name": "Mini Doomer"}, "psd": "01234567", "reason": "name 'x' -> 'Mini Doomer'", "at": AT}
    files = github.files_at(github.head)
    files[cb.EDITS_PATHS[0]] = _edits({"Mini Doomer": {"name": "Mini Doomer", "note": "n", "_psd": baked, "_by": "web", "_at": AT}})
    github.commit(files, "web", "Website edit: something else")
    text, commits = cb.run()
    assert len(commits) == 1 and github.commits[commits[0]]["message"] == "Card edits: 1 redundant"
    assert "already on the card: name" in text
    assert _edits_at(github)["cards"]["Mini Doomer"] == {"note": "n", "_by": "web", "_at": AT}
    github.commit(github.files_at(github.head), "web", "Website edit: more")
    assert cb.run()[1] == []
    # a name alone: the entry goes whole
    files = github.files_at(github.head)
    files[cb.EDITS_PATHS[0]] = _edits({"Mini Doomer": {"name": "Mini Doomer", "_psd": {**baked, "state": "refused"}, "_by": "web", "_at": AT}})
    github.commit(files, "web", "Website edit: again")
    text, commits = cb.run()
    assert len(commits) == 1 and "Mini Doomer" not in _edits_at(github)["cards"]


def test_a_printed_value_is_dropped_though_numbered_pngs_share_the_psd(github):
    """NME Stock: its name edit is what the PSD prints, but NME_Stock2.png shares the PSD. The redundancy check runs
    before the duplicate gate, so the value is dropped (no bake, no PNG needed) instead of refused for good; a refusal
    written before the check (no "printed") is looked at again."""
    files = github.files_at(github.head)
    files[cb.EDITS_PATHS[0]] = _edits({"A Snack": {"hp": 5, "note": "n", "_by": "web", "_at": AT}})
    github.commit(files, "web", "Website edit: A Snack")
    text, commits = cb.run()
    assert len(commits) == 1 and "already on the card: hp" in text
    assert _edits_at(github)["cards"]["A Snack"] == {"note": "n", "_by": "web", "_at": AT}
    data = _edits_at(github)
    old = {"state": "refused", "for": {"hp": 1}, "psd": _sha(files[SNACK])[:8], "reason": "duplicate: numbered PNGs "
           "share the PSD: A_Snack2.png", "at": AT}
    files = github.files_at(github.head)
    files[cb.EDITS_PATHS[0]] = _edits({**data["cards"], "A Snack": {"hp": 1, "_psd": old, "_by": "web", "_at": AT}})
    github.commit(files, "web", "Website edit: old verdict")
    assert "A Snack" in cb.pending()                                            # no "printed": looked at again
    cb.run()
    verdict = _edits_at(github)["cards"]["A Snack"]["_psd"]
    assert verdict["printed"] is False and verdict["reason"].startswith("duplicate:") and "A Snack" not in cb.pending()


def test_commit_failure_is_reported_and_writes_nothing(github, tmp_path):
    """The run's commit is fenced: head-moved after three tries is a note, no card gets a verdict, the report survives."""
    github.conflicts = 3
    text, commits = cb.run()
    assert commits == [] and "**Nothing committed: head-moved: main kept moving: not committed after 3 tries**" in text
    data = _edits_at(github)
    assert "_psd" not in data["cards"]["The Freezer"] and data["cards"]["Mini Doomer"]["hp"] == 5
    assert "head-moved: main kept moving" in text.split("## Refused", 1)[1] and "committed `" not in text
    assert text.count("verdict not written: nothing committed") == 4        # the refused cards say so too
    assert (tmp_path / cb.WORK_NAME / "report.md").read_text(encoding="utf-8") == text
    assert len(cb.pending()) == 5                                           # all of it again next run


def test_main_exits_1_when_the_commit_fails_or_a_card_errors(github, monkeypatch):
    """A failure needs a person: main exits 1 (card-edits.yml then tells Discord); refusals alone exit 0."""
    err = StringIO()
    monkeypatch.setattr(cb, "stderr", err)
    github.conflicts = 3
    assert cb.main([]) == 1 and "card_bake: Nothing committed: head-moved" in err.getvalue()
    github.conflicts = 0
    github.down.add(_sha(github.files_at(github.head)[DOOMER]))
    assert cb.main(["--card", "Mini Doomer"]) == 1 and err.getvalue().endswith("card_bake: 1 card with an error: Mini Doomer\n")
    github.down.clear()
    err.truncate(0)
    assert cb.main([]) == 0 and cb.main([]) == 0 and err.getvalue() == ""   # 1 redundant, 4 refused committed; then nothing to do


def test_cap_leaves_the_rest_for_the_next_run(github, monkeypatch):
    """At most MAX_BAKES cards baked and none started after BUDGET: the rest are `capped`, no verdict, pending next run."""
    monkeypatch.setattr(cb, "BUDGET", timedelta(0))
    text, commits = cb.run()
    assert commits == [] and "## Next run (capped) (5)" in text and "- Capped: 5 cards wait for the next run" in text
    assert "## Refused" not in text and len(cb.pending()) == 5
    monkeypatch.setattr(cb, "BUDGET", timedelta(minutes=40))

    files = github.files_at(github.head)
    data = json.loads(files[cb.EDITS_PATHS[0]].decode("utf-8"))
    data["cards"]["Mini Doomer"]["hp"] = 8
    files[cb.EDITS_PATHS[0]] = cb.dump_edits(data).encode("utf-8")
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
    assert len(commits) == 1 and github.commits[commits[0]]["message"].startswith("Card edits: 1 baked, 2 refused\n")
    assert "## Next run (capped) (2)" in text and "**Nobody**" in text.split("## Next run", 1)[1]
    assert cb.pending() == ["Nobody", "The Freezer"]


def test_font_verdict_is_retried_when_the_fonts_change(github, monkeypatch):
    """A `font` refusal carries MDW/Fonts' key: final while the fonts hold, pending once a font file is added or changed."""
    files = github.files_at(github.head)
    files["MDW/Fonts/Some Face.ttf"] = b"face"
    files["MDW/Fonts/stoogedings-src/x.svg"] = b"not installed"
    files[cb.EDITS_PATHS[0]] = _edits({"Mini Doomer": {"hp": 8, "_by": "web", "_at": AT}})
    github.commit(files, "web", "Website edit: Mini Doomer")
    monkeypatch.setattr(pe, "apply_jobs", lambda *args, **kwargs: (_ for _ in ()).throw(pe.Refused("font", "no Some Face")))
    monkeypatch.setattr(pe, "text_engine", lambda: True)
    text, commits = cb.run()
    verdict = _edits_at(github)["cards"]["Mini Doomer"]["_psd"]
    key = cb.snapshot_at(cb.CARD_REPO, github.head).fonts_key()
    assert len(commits) == 1 and verdict["reason"] == "font: no Some Face" and verdict["fonts"] == key and len(key) == 8
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
    files[cb.EDITS_PATHS[0]] = _edits({"Mini Doomer": {"hp": 8, "_by": "web", "_at": AT}})
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


def test_old_bar_pixel_verdicts_are_retried(github):
    """A `parity` or `idempotence` refusal under the floor (the old bar's, before 2026-10-23) is pending again: it bakes
    now; one past the floor, or a render that shows no change, stays final."""
    files = github.files_at(github.head)
    psd = _sha(files[DOOMER])[:8]

    def verdict(reason: str) -> list:
        entry = {"hp": 8, "_psd": {"state": "refused", "for": {"hp": 8}, "psd": psd, "reason": reason, "at": AT}, "_by": "web", "_at": AT}
        files[cb.EDITS_PATHS[0]] = _edits({"Mini Doomer": entry})
        github.commit(files, "web", "Website edit: a verdict")
        return cb.pending()
    assert verdict("parity: baseline bad 0.11% mean 3.01 max 136") == ["Mini Doomer"]
    assert verdict("idempotence: hp re-render bad 0.62% mean 0.80") == ["Mini Doomer"]
    assert verdict("parity: baseline bad 31.20% mean 40.00 max 255: past the 25% floor") == []
    assert verdict("idempotence: hp re-render bad 40.00% mean 50.00: past the 25% floor") == []
    assert verdict("parity: Hp Dark/5: effects unreadable: x") == []
    assert verdict("parity: ability: the render shows no change, baseline bad 10.70% mean 22.35 max 255") == []
    assert verdict("reparse: fields differ from the expected parse: hp") == []


@fork_text
def test_a_title_missing_the_bar_bakes_with_a_pixel_note(github, clone, tmp_path):
    """End to end: Red_Knife's title (its outer glow's noise, which no render matches) misses the parity bar and bakes;
    the commit note says so per edited box; the PSD, PNG and file land in one commit."""
    from psd_tools import PSDImage
    files = github.files_at(github.head)
    files.update(_card_files(clone, RED_KNIFE))
    (tmp_path / "knife.psd").write_bytes(files[RED_KNIFE])
    title = pe.read_name(PSDImage.open(str(tmp_path / "knife.psd")))
    name = title[:len(title) - 1] + ("e" if title[len(title) - 1] != "e" else "a")
    files[cb.EDITS_PATHS[0]] = _edits({"Red Knife": {"name": name, "_by": "web", "_at": AT}})
    github.commit(files, "web", "Website edit: Red Knife")
    text, commits = cb.run()
    message = github.commits[commits[0]]["message"]
    line = re.search(r"^- Red Knife: name .* -> .*; name baked with a pixel difference: bad ([\d.]+)%, mean ([\d.]+), max (\d+)$",
                     message, re.MULTILINE)
    assert len(commits) == 1 and message.startswith("Card edits: 1 baked\n") and line, message
    bad, mean = float(line.group(1)), float(line.group(2))
    assert (bad > pe.PARITY_BAD_MAX or mean > pe.PARITY_MEAN_MAX) and bad <= pe.PARITY_FLOOR
    assert "; name baked with a pixel difference: bad " in text.split("## Baked (1)", 1)[1]
    after = github.files_at(github.head)
    assert sorted(p for p in after if after[p] != files.get(p)) == sorted([RED_KNIFE[:len(RED_KNIFE) - 4] + ".png", RED_KNIFE, cb.EDITS_PATHS[0]])


def test_transient_errors_write_no_verdict(github, monkeypatch):
    """A 5xx on a blob, a rate-limited POST: reported under Errors, retried next run, never a verdict for the editor."""
    github.down.add(_sha(github.files_at(github.head)[DOOMER]))
    text, commits = cb.run(card="Mini Doomer")
    assert commits == [] and "## Errors (1)" in text and "HTTPError: 503" in text
    assert "_psd" not in _edits_at(github)["cards"]["Mini Doomer"]
    github.down.clear()
    github.rate_limited = True
    monkeypatch.setattr(gc, "git_commit", lambda *a, **k: pytest.fail("the SSH path is not for a rate limit"))
    text, commits = cb.run(card="Mini Doomer")
    assert commits == [] and "error: GitHub 403: API rate limit exceeded: rate limited (on POST git/blobs)" in text
    assert "_psd" not in _edits_at(github)["cards"]["Mini Doomer"] and gc.cant_write is False
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
    github.commit(github.files_at(github.head), "web", "Website edit: more")
    text, commits = cb.run(card="the_freezer")
    assert len(commits) == 1 and len(pushed) == 2 and pushed[1] == "Card edits: 1 refused"
    assert [route for method, route in github.calls if method == "POST"] == ["git/blobs"]        # no API write tried this run


def test_verdict_skipped_when_edit_changed_before_the_commit(github, monkeypatch):
    """land re-reads the file at H': a refused card whose edit changed meanwhile keeps the new edit and gets no verdict; the others land."""
    real = cb.land

    def land(repository, shares, message, dry_run, out_path="", local=None):
        current = github.files_at(github.head)                  # before the commit: The Freezer's edit changes
        data = json.loads(current[cb.EDITS_PATHS[0]].decode("utf-8"))
        data["cards"]["The Freezer"]["stars"] = 3
        current[cb.EDITS_PATHS[0]] = cb.dump_edits(data).encode("utf-8")
        github.commit(current, "web", "Website edit: The Freezer again")
        return real(repository, shares, message, dry_run, out_path, local)
    monkeypatch.setattr(cb, "land", land)
    text, commits = cb.run()
    assert len(commits) == 1 and "**The Freezer**" in text and "edit changed meanwhile: verdict not written" in text
    assert github.commits[commits[0]]["message"] == "Card edits: 1 redundant, 3 refused"
    data = _edits_at(github)
    assert data["cards"]["The Freezer"]["stars"] == 3 and "_psd" not in data["cards"]["The Freezer"]
    assert data["cards"]["Nobody"]["_psd"]["state"] == "refused" and data["cards"]["A Snack"]["_psd"]["state"] == "refused"
    assert cb.pending() == ["The Freezer"]


def test_missing_runner_dependency_is_transient(github, monkeypatch):
    """A runner without scipy or the text extra fails inside apply_jobs with an ImportError: an error, never a verdict."""
    files = github.files_at(github.head)
    files[cb.EDITS_PATHS[0]] = _edits({"Mini Doomer": {"hp": 8, "_by": "web", "_at": AT}})
    github.commit(files, "web", "Website edit: Mini Doomer")

    def no_scipy(*args, **kwargs):
        raise ImportError("Gradient fills require: scipy")
    monkeypatch.setattr(pe, "apply_jobs", no_scipy)
    text, commits = cb.run(card="Mini Doomer")
    assert commits == [] and "## Errors (1)" in text and "ImportError: Gradient fills require: scipy" in text
    entry = _edits_at(github)["cards"]["Mini Doomer"]
    assert "_psd" not in entry and entry["hp"] == 8


def test_select_and_snapshot(github):
    info = github.commits[github.head]
    snapshot = cb.Snapshot(cb.CARD_REPO, github.head, info, cb._tree(cb.CARD_REPO, info["tree"]["sha"]))
    assert "creature template" not in snapshot.psds and snapshot.psds["mini doomer"] == DOOMER
    assert snapshot.png_for(DOOMER) == (DOOMER[:len(DOOMER) - 4] + ".png", [])
    assert snapshot.png_for(SNACK) == (SNACK[:len(SNACK) - 4] + ".png", ["Items/Heal/2 Stars/A_Snack2.png"])
    assert snapshot.duplicate_of("a snack2") == SNACK and snapshot.duplicate_of("nobody") == ""
    edits = cb.CardEdits.parse(github.files_at(github.head)[cb.EDITS_PATHS[0]].decode("utf-8"))
    cards = cb.select(edits, snapshot)
    assert [c.name for c in cards] == ["A Snack", "A Snack2", "Mini Doomer", "Nobody", "The Freezer"]    # the file's order
    assert cards[2].fields == {"hp": 5} and cards[3].rel == "" and cards[4].psd_sha == snapshot.sha(FREEZER)
    assert [c.name for c in cb.select(edits, snapshot, "a_snack")] == ["A Snack"]


@fork
def test_bake_hp_lands_psd_png_and_file(github, tmp_path):
    """One stat bake end to end: PSD + PNG + the file in one commit, the verdict beside the fields that stay."""
    from psd_tools import PSDImage
    files = github.files_at(github.head)
    files[cb.EDITS_PATHS[0]] = _edits({"Mini Doomer": {"hp": 8, "note": "checked", "_by": "web", "_at": AT}})
    github.commit(files, "web", "Website edit: Mini Doomer")
    text, commits = cb.run()
    assert len(commits) == 1 and github.commits[commits[0]]["message"] == "Card edits: 1 baked\n\n- Mini Doomer: hp 5 -> 8\n\nEdit-by: web"
    after = github.files_at(github.head)
    changed = sorted(p for p in after if after[p] != files.get(p))
    assert changed == [DOOMER[:len(DOOMER) - 4] + ".png", DOOMER, cb.EDITS_PATHS[0]]
    data = json.loads(after[cb.EDITS_PATHS[0]].decode("utf-8"))
    verdict = data["cards"]["Mini Doomer"].pop("_psd")
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
def test_dry_run_ability_bake_regenerates_fx_layers(github, tmp_path, clone):
    """Idea 14 through the driver: an ability edit adding Blazing fetches the chart by sha, cuts "Ability FX: Blazing" above
    Text, passes every gate and names the layer in the report; --dry-run leaves the artifacts and commits nothing. Mini
    Doomer before its badge layer (PLAIN_DOOMER): with it the edit refuses (test_ability_bake_refuses_a_badge_card)."""
    from time import perf_counter
    from psd_tools import PSDImage
    chart = clone / pe.STATUS_CHART
    if not chart.exists():
        pytest.skip("no status chart")
    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
    parser = pe.offline_parser(vocab["all_types"], vocab["stars"], vocab["manifest"])
    png = DOOMER[:len(DOOMER) - 4] + ".png"
    plain = {DOOMER: at_rev(clone, PLAIN_DOOMER, DOOMER), png: at_rev(clone, PLAIN_DOOMER, png)}
    (tmp_path / "plain.psd").write_bytes(plain[DOOMER])
    old = pe.parse_card(parser, str(tmp_path / "plain.psd"), DOOMER)["card"]["ability"]
    files = github.files_at(github.head)
    files.update(plain)
    files[pe.STATUS_CHART] = chart.read_bytes()
    files[cb.EDITS_PATHS[0]] = _edits({"Mini Doomer": {"ability": old + " Inflict Blazing.", "note": "fx", "_by": "web", "_at": AT}})
    github.commit(files, "web", "Website edit: Mini Doomer ability")
    started = perf_counter()
    text, commits = cb.run(dry_run=True)
    seconds = perf_counter() - started
    assert commits == [] and "## Baked (1)" in text and "FX layers: Ability FX: Blazing" in text and "not committed (dry run)" in text
    assert github.files_at(github.head) == files                            # nothing landed
    work = tmp_path / cb.WORK_NAME / "Mini Doomer"
    saved = PSDImage.open(str(work / "new" / "Mini_Doomer.psd"))
    fx, = pe.fx_layers(saved, pe.ability_layer(saved))
    assert fx.name == "Ability FX: Blazing" and saved.index(fx) == saved.index(pe._text_group(saved)) + 1 and saved.has_preview()
    parse = json.loads((work / "parse.json").read_text(encoding="utf-8"))
    assert parse["fx"] == ["Ability FX: Blazing"] and parse["validator"] == [] and parse["new"]["ability"].endswith("Inflict Blazing.")
    rewritten = json.loads((work.parent / "Card Edits.json").read_text(encoding="utf-8"))
    entry = rewritten["cards"]["Mini Doomer"]                                # the note keeps the entry; the ability is baked out
    assert "ability" not in entry and entry["note"] == "fx" and entry["_psd"]["state"] == "baked" and entry["_psd"]["for"] == {"ability": old + " Inflict Blazing."}
    print(f"dry-run bake {seconds:.1f} s; PSD {len(files[DOOMER])} -> {(work / 'new' / 'Mini_Doomer.psd').stat().st_size} bytes")


@fork_text
def test_ability_bake_refuses_a_badge_card(github, clone):
    """Mini Doomer carries "Ability FX: (BOT)" (queue A's badges): an ability edit would drop it, so the card is refused
    with the reason in its verdict and nothing of it lands."""
    chart = clone / pe.STATUS_CHART
    if not chart.exists():
        pytest.skip("no status chart")
    files = github.files_at(github.head)
    files[pe.STATUS_CHART] = chart.read_bytes()
    files[cb.EDITS_PATHS[0]] = _edits({"Mini Doomer": {"ability": "Die.", "_by": "web", "_at": AT}})
    github.commit(files, "web", "Website edit: Mini Doomer ability")
    text, commits = cb.run(dry_run=True)
    assert commits == [] and "## Refused (1)" in text
    assert "unsupported-card: ability edit would drop the badge layer Ability FX: (BOT); edit in Photoshop" in text


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
    assert _git(local.clone, "log", "-1", "--format=%an|%s", b).strip() == "AutoCroissant|Card edits: 1 redundant, 4 refused"
    assert not [m for m, _ in github.calls if m != "GET"] and "activity" in [r for _, r in github.calls]   # the recent gate as ever
    assert not [r for _, r in github.calls if r.startswith(("git/ref", "git/commits", "git/trees"))]       # the tree from the clone
    rest = sorted(r for _, r in github.calls if r.startswith("git/blobs/"))
    cards = {path: data for path, data in github.files_at(github.head).items()
             if path.startswith(("Auxiliary", "Field")) or path == SNACK}                       # A Snack's: the redundancy check
    assert rest == sorted(f"git/blobs/{_sha(data)}" for data in cards.values())                # only what the clone lacks
    assert f"One local commit: `{b}` Card edits: 1 redundant, 4 refused (not pushed)." in text and f"- Local: `{local.clone}`" in text
    assert "The checkout is on it." in text
    edits = (local.clone / cb.EDITS_PATHS[0]).read_bytes()
    assert _sha(edits) == _files_in(local.clone, b)[cb.EDITS_PATHS[0]]                         # the checkout is on B

    before = github.head                                                                        # the live run, for comparison
    cb.run()
    strip = lambda data: {k: {f: ({**x, "at": ""} if f == "_psd" else x) for f, x in v.items()} for k, v in data["cards"].items()}
    assert strip(json.loads(edits)) == strip(_edits_at(github))
    live = sorted(p for p, data in github.files_at(github.head).items() if github.files_at(before).get(p) != data)
    ours = _git(local.clone, "diff-tree", "--no-commit-id", "-r", "--name-only", b).splitlines()
    assert ours == live == [cb.EDITS_PATHS[0]]


def test_local_with_nothing_pending_commits_nothing_and_keeps_verdicts(github, local):
    """A rerun on B: nothing pending, no commit, HEAD stays; a new pending edit gets B2, the refused verdicts kept as they were."""
    _, (b,) = cb.run(local=str(local.clone))
    verdicts = {k: v["_psd"] for k, v in json.loads((local.clone / cb.EDITS_PATHS[0]).read_bytes())["cards"].items() if "_psd" in v}
    assert sorted(verdicts) == ["A Snack", "A Snack2", "Nobody", "The Freezer"] and all(v["state"] == "refused" for v in verdicts.values())
    text, commits = cb.run(local=str(local.clone), pending_only=True)
    assert commits == [] and "4 refused before (not re-checked)" in text and _git(local.clone, "rev-parse", "HEAD").strip() == b
    text, commits = cb.run(local=str(local.clone))                                              # re-checked: the same verdicts
    assert commits == [] and _git(local.clone, "rev-parse", "HEAD").strip() == b

    data = json.loads((local.clone / cb.EDITS_PATHS[0]).read_bytes())                           # a website edit after B
    data["cards"]["Mini Doomer"]["hp"] = 5
    _put(local.clone, {cb.EDITS_PATHS[0]: cb.dump_edits(data).encode("utf-8")})
    c = _commit_in(local.clone, "Website edit: Mini Doomer")
    text, commits = cb.run(local=str(local.clone), pending_only=True)
    assert len(commits) == 1 and _git(local.clone, "rev-parse", f"{commits[0]}^").strip() == c
    after = json.loads((local.clone / cb.EDITS_PATHS[0]).read_bytes())["cards"]
    assert after["Mini Doomer"] == {"note": "checked", "_by": "web", "_at": AT}
    assert {k: v["_psd"] for k, v in after.items() if "_psd" in v} == verdicts                    # byte for byte, dates too
    assert _git(local.origin, "rev-parse", "main").strip() == local.head


def test_local_dry_run_and_a_dirty_checkout(github, local, tmp_path):
    """--dry-run makes no B; a local change in B's way fails the run (exit 1) and leaves the checkout as it was."""
    text, commits = cb.run(dry_run=True, local=str(local.clone))
    assert commits == [] and "Would commit: Card edits: 1 redundant, 4 refused." in text
    assert _git(local.clone, "rev-parse", "HEAD").strip() == local.head and not _git(local.clone, "status", "--porcelain")
    path = local.clone / cb.EDITS_PATHS[0]
    path.write_bytes(path.read_bytes() + b" ")
    out = StringIO()
    with contextlib.redirect_stdout(out):
        assert cb.main(["--local", str(local.clone)]) == 1
    assert "Nothing committed: error: git merge in" in out.getvalue() and "left unreferenced)" in out.getvalue()
    assert _git(local.clone, "rev-parse", "HEAD").strip() == local.head and path.read_bytes().endswith(b"}\n ")


def test_local_head_elsewhere_leaves_the_checkout(github, local):
    """--head another commit than the checkout's HEAD: B is made on it, the checkout doesn't move."""
    _put(local.clone, {"TTS/notes.txt": b"x"})
    newer = _commit_in(local.clone, "elsewhere")
    text, (b,) = cb.run(head=local.head, local=str(local.clone))
    assert _git(local.clone, "rev-parse", f"{b}^").strip() == local.head and _git(local.clone, "rev-parse", "HEAD").strip() == newer
    assert f"main `{local.head[:8]}`" in text
    assert f"**The checkout is not on it:** HEAD is `{newer[:8]}` (not --head), so check out `{b[:8]}`" in text


def test_local_bare_or_unborn_head_still_makes_b(github, local, tmp_path):
    """A bare clone, or one whose HEAD is unborn: B is made on --head, nothing to move, and the report says so."""
    bare = tmp_path / "bare.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(bare))                           # HEAD -> main, unborn
    _git(bare, "fetch", "-q", "--filter=blob:none", f"file://{local.origin}", "main:refs/remotes/origin/main")
    text, (b,) = cb.run(head=local.head, local=str(bare))
    assert _git(bare, "rev-parse", f"{b}^").strip() == local.head and "No checkout moved (a bare clone or an unborn HEAD)." in text
    assert gc.checkout_head(str(bare)) == ""


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
    data = json.loads((local.clone / cb.EDITS_PATHS[0]).read_bytes())
    data["cards"] = {"Mini Doomer": {"hp": 8, "note": "checked", "_by": "web", "_at": AT}}
    _put(local.clone, {cb.EDITS_PATHS[0]: cb.dump_edits(data).encode("utf-8")})
    _commit_in(local.clone, "Website edit: Mini Doomer")
    text, (b,) = cb.run(local=str(local.clone))
    assert _git(local.clone, "log", "-1", "--format=%B", b).strip() == "Card edits: 1 baked\n\n- Mini Doomer: hp 5 -> 8\n\nEdit-by: web"
    changed = _git(local.clone, "diff-tree", "--no-commit-id", "-r", "--name-only", b).splitlines()
    assert changed == sorted([DOOMER[:len(DOOMER) - 4] + ".png", DOOMER, cb.EDITS_PATHS[0]])
    verdict = json.loads((local.clone / cb.EDITS_PATHS[0]).read_bytes())["cards"]["Mini Doomer"]["_psd"]
    assert verdict["state"] == "baked" and verdict["psd"] == _files_in(local.clone, b)[DOOMER][:8]
    assert not (local.clone / DOOMER).exists()                                                  # sparse: not checked out
    assert not [m for m, _ in github.calls if m != "GET"]


# ========================
# The Action's sparse copy
# ========================
def _bot_files_in_workflow() -> list:
    """The AutoCroissant paths card-edits.yml checks out (its second checkout step)."""
    if not WORKFLOW.exists():
        pytest.skip(f"no {WORKFLOW}")
    text = WORKFLOW.read_text(encoding="utf-8")
    block = text.split("repository: bobberlington/AutoCroissant", 1)[1].split("sparse-checkout: |", 1)[1]
    paths = []
    for line in block.splitlines()[1:]:
        if not line.startswith(" " * 12) or not line.strip():
            break
        paths.append(line.strip().lstrip("/"))
    return paths


ALLOWED_IMPORTS = {"PIL", "attr", "attrs", "certifi", "charset_normalizer", "fontTools", "idna", "numpy", "psd_tools", "requests",
                   "urllib3", "scipy", "aggdraw"}      # what the Actions' pip lines install (psd-tools' own dependencies included)
SKIP_IMPORTS = {"commands", "global_config", "sitecustomize", "usercustomize", "cython_runtime"}


def third_party_imports(code: str, cwd) -> set:
    """The non-stdlib top-level modules an import leaves in sys.modules (a copy of the repo at cwd, no config.py)."""
    probe = (f"import sys; {code}; tops = {{m.split('.')[0] for m in sys.modules}}; "
             f"print(sorted(tops - set(sys.stdlib_module_names) - {SKIP_IMPORTS!r} - {{m for m in tops if m.startswith('_')}}))")
    done = run_process([executable, "-c", probe], cwd=cwd, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return set(ast.literal_eval(done.stdout.strip()))


def test_action_shaped_copy_imports_and_prints_help(tmp_path):
    """Exactly the yaml's files, no config.py: `python -m commands.card_bake --help` works (the import closure is complete)."""
    paths = _bot_files_in_workflow()
    assert "commands/card_bake.py" in paths and "commands/git_commit.py" in paths
    assert "commands/psd_analyzer.py" not in paths and "config.py" not in paths
    bot = tmp_path / "bot"
    for rel in paths:
        assert (REPO / rel).is_file(), f"{rel} is checked out by the Action but missing here"
        (bot / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, bot / rel)
    done = run_process([executable, "-m", "commands.card_bake", "--help"], cwd=bot, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("usage: python -m commands.card_bake") and "--dry-run" in done.stdout and "--card" in done.stdout
    assert set(re.findall(r"--[a-z-]+", done.stdout)) == {"--repo", "--head", "--dry-run", "--card", "--pending", "--local",
                                                          "--help"}   # no other flags
    code = "import sys, commands.card_bake; print(sorted(m for m in ('config', 'discord', 'commands.psd_analyzer') if m in sys.modules))"
    done = run_process([executable, "-c", code], cwd=bot, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and done.stdout.strip() == "[]", done.stderr      # the copy has no config.py, imports no discord
    extra = third_party_imports("import commands.card_bake", bot) - ALLOWED_IMPORTS
    assert not extra, f"card-edits.yml installs none of these: {sorted(extra)}"    # a stray pandas/discord fails here, not in the Action


def test_workflow_shape():
    """card-edits.yml: live every 30 minutes behind the pending gate, dispatch with dry_run on by default, never on the
    pipeline's own commits, one scheduled bake at a time in its own group (a running bake is never cancelled; sharing
    tts-save's group would let a tick replace a pending TTS save) and a group per dispatch (never replaced by a tick),
    the schedule baking only pending cards, step timeouts, the report line it greps before running site.yml and
    tts-save.yml, and one Discord message when a streak of failed scheduled runs starts (main exits 1)."""
    if not WORKFLOW.exists():
        pytest.skip(f"no {WORKFLOW}")
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "python -m commands.card_bake" in text and '--repo "$GITHUB_REPOSITORY" --head "$GITHUB_SHA"' in text
    assert "args+=(--dry-run)" in text and 'args+=(--card "$CARD")' in text and '[ "$EVENT" = schedule ] && args+=(--pending)' in text
    assert "if: github.actor != 'AutoCroissant'" in text and "actions/checkout@v7" in text
    grep = re.search(r"grep -q '([^']+)'", text).group(1)                          # the commit line the final step greps
    report = cb.Report("o/r", "h" * 40, False, "w")
    card = cb.Card("X", "x", {"hp": 1})
    card.redundant, card.commit = ["hp"], "abcdef01" * 5
    report.cards = [card]
    assert grep in cb.render(report) and grep not in cb.render(cb.Report("o/r", "h" * 40, True, "w"))
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(text)
    on = doc[True]                                                                  # PyYAML reads the `on` key as True
    assert on["schedule"] == [{"cron": "*/30 * * * *"}] and "push" not in on
    assert on["workflow_dispatch"]["inputs"]["dry_run"]["default"] is True
    assert doc["jobs"]["bake"]["if"] == "github.actor != 'AutoCroissant'"
    assert doc["concurrency"] == {"group": "${{ github.event_name == 'schedule' && 'card-edits' || format('card-edits-{0}', github.run_id) }}",
                                  "cancel-in-progress": False}
    assert doc["permissions"] == {"contents": "write", "actions": "write"}
    steps = doc["jobs"]["bake"]["steps"]
    gate = next(i for i, step in enumerate(steps) if step.get("id") == "gate")
    assert "from commands.card_bake import pending" in steps[gate]["run"] and '[ "$EVENT" != schedule ]' in steps[gate]["run"]
    alert = steps[len(steps) - 1]
    assert all("steps.gate.outputs.go == 'true'" in step.get("if", "") for step in steps[gate + 1:len(steps) - 1])   # nothing pending: done
    bake = next(step for step in steps if step.get("name") == "Bake")
    assert bake["env"]["DRY_RUN"] == "${{ github.event_name == 'workflow_dispatch' && inputs.dry_run }}"   # the schedule is live
    after = next(step for step in steps if "grep -q" in step.get("run", ""))
    assert after["if"].startswith("${{ !cancelled() &&")                     # a card's error exits 1: a commit still runs them
    assert alert["if"] == "(failure() || cancelled()) && github.event_name == 'schedule'"   # a job timeout is a cancel
    assert alert["env"]["WEBHOOK"] == "${{ secrets.DISCORD_CHANGELOG_WEBHOOK }}" and "$RUN" in alert["run"]
    assert "gh run list" in alert["run"] and '[ "$last" = failure ]' in alert["run"]       # once per streak
    assert steps[gate]["timeout-minutes"] == 10 and bake["timeout-minutes"] == 50        # a step timeout fails the run
    assert doc["jobs"]["bake"]["timeout-minutes"] == 60


def test_workflow_gate_line_calls_pending(github, capsys, monkeypatch):
    """The gate's python -c line, run as written: the pending card names, one per line."""
    if not WORKFLOW.exists():
        pytest.skip(f"no {WORKFLOW}")
    code = re.search(r"python -c '([^']+)'", WORKFLOW.read_text(encoding="utf-8")).group(1)
    monkeypatch.setattr("sys.argv", ["-c", cb.CARD_REPO, github.head])
    exec(code, {})
    assert capsys.readouterr().out.splitlines() == ["A Snack", "A Snack2", "Mini Doomer", "Nobody", "The Freezer"]
    cb.run()
    monkeypatch.setattr("sys.argv", ["-c", cb.CARD_REPO, github.head])
    exec(code, {})
    assert capsys.readouterr().out.strip() == ""
