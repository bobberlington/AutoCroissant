"""
Automatic flags (commands/card_checks.py): a bar read off the picture, the picture against the data, the file's life (a
fixed card clears, a person's flag goes once its PSD changed and the checks are clean), the soft notes told nowhere, and
card-checks.yml's copy of the bot (2026-11-07).
"""
import json
import shutil
import subprocess
from os import environ
from os.path import expanduser
from pathlib import Path
from sys import executable
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("requests", reason="card_bake imports requests (absent from this Python)")
import commands.card_checks as cc  # noqa: E402
from commands.card_edits import CardEdits  # noqa: E402
from commands.psd_edit import SOFT_NOTE, real_notes  # noqa: E402
from conftest import CLONE  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
LIT, EMPTY = (47, 191, 58), (16, 63, 19)


def bar(cells: int, bottom: int = None) -> np.ndarray:
    """An 813x84 template bar, the first cells lit; bottom: its bottom half's (past 10)."""
    row = np.zeros((84, 813, 3), np.uint8)
    edge = lambda n: cc.TEMPLATE_CELLS[n]
    row[:] = EMPTY
    row[:, :edge(cells)] = LIT
    if bottom is not None:
        row[:42] = LIT
        row[42:] = EMPTY
        row[42:, :edge(bottom)] = LIT
    return row


def test_read_row():
    """A step in both halves, a step in the bottom half alone, a flat row, a busy one."""
    assert cc.read_row(bar(5)) == {"best": 5, "could": [5]}
    assert cc.read_row(bar(10, 2))["best"] == 12
    soft = bar(10)
    soft[42:, cc.TEMPLATE_CELLS[2]:] -= 40                      # see-through: a soft step, the bottom half alone
    assert cc.read_row(soft)["best"] == 12
    assert cc.read_row(bar(10)).get("flat") and cc.read_row(bar(0)).get("flat")
    assert cc.read_row(np.random.default_rng(1).integers(0, 255, (84, 813, 3)).astype(np.uint8))["best"] is None


def test_flat_rows_by_the_others():
    """A flat row is full or empty by the card's stepped rows."""
    rows = [(row, cc.read_row(row)) for row in (bar(4), bar(10), bar(0))]
    cc._tell_flats(rows)
    assert [read["best"] for _, read in rows] == [4, 10, 0] and rows[1][1]["could"] == [10, 20]


def test_mismatches():
    """The note names the bar; a value the bar could show says nothing; a custom card speaks only when every bar reads;
    a website-only value differs on purpose; creatures by bar."""
    read = lambda best, could=None: {"best": best, "could": could if could is not None else [best]}
    record = {"style": "new", "reads": {"hp": [read(5)], "def": [read(3)], "atk": [read(None, None) | {"could": None}], "spd": [read(4, [4, 7])]}}
    data = {"stats": {"hp": 5, "defense": 1, "attack": 2, "speed": 7}}
    assert cc.mismatches(record, data) == ["The DEF bar shows 3 but the data says 1."]
    assert cc.mismatches(record, data, ["def"]) == [] and cc.mismatches(record, data, {"def": 1}) == []   # website only, pending
    assert cc.mismatches({**record, "style": "custom"}, data) == []
    many = {"style": "custom", "reads": {"hp": [read(5), read(6)]}}
    assert cc.mismatches(many, {"creatures": [{"hp": 5}, {"hp": 4}]}) == ["Creature 2's HP bar shows 6 but the data says 4."]
    assert cc.flags({"notes": ["N."], "bars": "B."}, None) == {"hand": "N.", "bars": "B."}


def test_refresh_clears():
    """A website-only fix clears without a push; a gone PSD's record goes; a person's flag goes once its PSD changed and
    the checks are clean, stays gone though they speak again, and a new flag starts over."""
    rel = "Creatures/A.psd"
    snapshot = SimpleNamespace(psds={"a": rel}, sha=lambda path: "abcd1234ffff")
    read = {"best": 2, "could": [2]}
    checks = {"cards": {"abcd1234ffff": {"style": "new", "reads": {"def": [read]}, "notes": [], "bars": None}, "gone": {}}, "cleared": {}}
    stats = {"A": {"blob_sha": "abcd1234ffff", "stats": {"defense": 8}}}
    edits = CardEdits.parse(json.dumps({"cards": {"A": {"flags": {"hand": "x", "psd": "00000000"}}}}))
    cc.refresh(checks, snapshot, stats, edits)
    assert list(checks["cards"]) == ["abcd1234ffff"] and checks["cards"]["abcd1234ffff"]["hand"].startswith("The DEF bar")
    assert checks["cleared"] == {}
    cc.refresh(checks, snapshot, {"A": {"blob_sha": "abcd1234ffff", "stats": {"defense": 2}}}, edits)
    assert checks["cards"]["abcd1234ffff"]["hand"] is None and checks["cleared"] == {rel: "00000000"}
    cc.refresh(checks, snapshot, stats, edits)
    assert checks["cleared"] == {rel: "00000000"}
    record = checks["cards"].pop("abcd1234ffff")
    cc.refresh(checks, snapshot, stats, edits)                  # a new PSD not read yet: still gone
    assert checks["cleared"] == {rel: "00000000"}
    checks["cards"]["abcd1234ffff"] = record
    edits = CardEdits.parse(json.dumps({"cards": {"A": {"flags": {"hand": "y", "psd": "abcd1234"}}}}))
    cc.refresh(checks, snapshot, stats, edits)
    assert checks["cleared"] == {}


def test_still_and_swapped(monkeypatch):
    """A bar a bake left as it was, though the others moved; a swap by the printed labels, equal values too."""
    psd = SimpleNamespace(width=100)
    boxes = {"hp": [(0, 0, 100, 10)], "def": [(0, 10, 100, 20)], "atk": [(0, 20, 100, 30)], "spd": [(0, 30, 300, 40)]}
    monkeypatch.setattr(cc, "bar_boxes", lambda psd: boxes)
    monkeypatch.setattr(cc, "by_label", lambda psd, found: {**found, "def": found["atk"], "atk": found["def"]})
    moved = np.zeros((40, 100), bool)
    assert cc._still(psd, moved, "custom") == ["hp", "def", "atk", "spd"]
    moved[:20] = True
    assert cc._still(psd, moved, "new") == ["def"]              # by its row; spd past the canvas has no say
    assert cc._still(psd, moved, "custom") == []                # its boxes need not be where the picture changes
    assert cc.swapped(psd) == ["def", "atk"]


def test_art_over_darks(monkeypatch):
    """Only Lines may lie over the new bars' Darks group: else a stat edit bakes for real (Sans, The Thin Man)."""
    layer = lambda name, shown=True: SimpleNamespace(name=name, is_visible=lambda: shown)
    darks, bars = layer("Darks"), layer("Bars")
    monkeypatch.setattr(cc, "comp_bars", lambda psd: {"hp": [(SimpleNamespace(parent=darks), 5, (0, 0, 1, 1))]})
    darks.parent = [bars, darks, layer("Lines")]
    assert not cc._art_over_darks(None)
    darks.parent = [darks, bars, layer("Lines")]
    assert cc._art_over_darks(None)
    darks.parent = [darks, layer("Bars", False)]
    assert not cc._art_over_darks(None)


def test_raises_stop_at_the_display(monkeypatch):
    """A stat raise stays within the card's own display (Computer Virus's rows: more is refused when typed, owner
    2026-11-08, never a note): a stat at its room steps down; creature 2's own row by its own room."""
    monkeypatch.setattr(cc.psd_edit, "stat_caps", lambda psd: {"hp": [5, 5], "def": [4, 4], "atk": [6, 6], "spd": [5, 5]})
    monkeypatch.setattr(cc.psd_edit, "creature_bar", lambda psd, stat, n: (stat, n))
    monkeypatch.setattr(cc, "stat_rows", lambda psd: {})
    baked = []
    monkeypatch.setattr(cc, "_real_bake", lambda ctx, rel, fields, png: (baked.append(fields), ("baked", "", None))[1])
    first, second = {"hp": 2, "def": 4, "atk": 3, "spd": 5}, {"hp": 3, "def": 2, "atk": 6, "spd": 5}
    assert cc._stat_notes(None, "C.psd", None, {"card": {**first, "creatures": [first, second]}}, "custom", None) == []
    up = baked[1]
    assert {k: up[k] for k in first} == {"hp": 3, "def": 3, "atk": 4, "spd": 4}
    assert up["creatures"] == [{"hp": 3, "def": 3, "atk": 4, "spd": 4}, {"hp": 4, "def": 3, "atk": 5, "spd": 4}]


def test_to_check_and_file():
    """No result, or a new PNG: checked; --all, a shard; the file reads back."""
    tree = {"A.psd": ("", "11111111aa"), "A.png": ("", "22222222bb"), "B.psd": ("", "33333333cc")}
    snapshot = SimpleNamespace(psds={"a": "A.psd", "b": "B.psd"}, sha=lambda path: tree[path][1],
                               png_for=lambda rel: ("A.png" if rel == "A.psd" else "", []), dup_pictures=dict)
    checks = {"version": 1, "cards": {"11111111aa": {"png": "22222222"}, "33333333cc": {"png": ""}}, "cleared": {}}
    assert cc.to_check(snapshot, checks) == []
    assert cc.to_check(snapshot, checks, True) == ["A.psd", "B.psd"]
    assert cc.to_check(snapshot, checks, only={"b"}) == ["B.psd"]
    assert cc.to_check(snapshot, {"cards": {"11111111aa": {"png": "old"}}}, shard=(1, 2)) == ["A.psd", "B.psd"]
    assert json.loads(cc.text_of(checks))["cards"] == checks["cards"]


def test_blobs_of_in_reading_order():
    """The chart, manifest and templates first, then each card's PSD, PNG and duplicates together."""
    paths = ["MDW/Templates/Creature_Template.psd", "A.psd", "A.png", "A2.png", "B.psd", "Types/Fire.png", "Types/Dark.png",
             cc.psd_edit.STATUS_CHART, cc.ICON_MANIFEST_PATH]
    tree = {path: ("", path) for path in paths}
    snapshot = SimpleNamespace(tree=tree, sha=lambda path: path, dup_pictures=lambda: {"A.psd": ["A2.png"]},
                               png_for=lambda rel: ("A.png" if rel == "A.psd" else "", []))
    assert cc.blobs_of(snapshot, ["A.psd", "B.psd"], {}) == [cc.psd_edit.STATUS_CHART, cc.ICON_MANIFEST_PATH,
                                                            "MDW/Templates/Creature_Template.psd", "A.psd", "A.png", "A2.png", "B.psd"]


def test_workflow_checks_cards_as_their_blobs_land(tmp_path):
    """card-checks.yml: Shared blobs fetches the chart early; Fetch them leaves it out and returns at once (the fetch in
    the background); one shard's Check starts with CARD_BLOBS_PENDING, `all` and long lists wait for the whole fetch;
    Fetched counts, and stops the fetches Check outlived."""
    yaml = pytest.importorskip("yaml")
    workflow = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE))) / ".github/workflows/card-checks.yml"
    if not workflow.is_file() or not shutil.which("timeout"):
        pytest.skip("no card-checks.yml or no timeout (coreutils)")
    steps = {s.get("name"): s["run"] for s in yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"]["check"]["steps"]
             if s.get("run")}
    head = steps["Check"].split("mapfile")[0] + 'echo "pending=${CARD_BLOBS_PENDING:-}"'    # before the bot runs
    git = lambda *args: subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout
    work, origin = tmp_path / "work", tmp_path / "origin.git"
    work.mkdir()
    git("-C", str(work), "init", "-q", "-b", "main")
    for name in ("a", "b", cc.psd_edit.STATUS_CHART):
        (work / name).parent.mkdir(parents=True, exist_ok=True)
        (work / name).write_bytes(name.encode() * 4096)
    git("-C", str(work), "add", "-A")
    git("-C", str(work), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x")
    git("clone", "-q", "--bare", str(work), str(origin))
    git("-C", str(origin), "config", "uploadpack.allowFilter", "true")
    git("-C", str(origin), "config", "uploadpack.allowAnySHA1InWant", "true")
    blobs = git("-C", str(work), "rev-parse", f"HEAD:{cc.psd_edit.STATUS_CHART}", "HEAD:a", "HEAD:b")

    def job(n, wait):
        repo, temp = tmp_path / f"clone{n}", tmp_path / f"temp{n}"
        temp.mkdir()
        git("clone", "-q", "--filter=blob:none", "--no-checkout", f"file://{origin}", str(repo))
        git("-C", str(repo), "config", "remote.origin.uploadpack", f"sleep {wait}; git-upload-pack")    # GitHub's time
        (temp / "blobs.txt").write_text(blobs)
        env = {"PATH": environ["PATH"], "HOME": environ.get("HOME", ""), "RUNNER_TEMP": str(temp), "GIT_NO_LAZY_FETCH": "1"}

        def step(script, **more):
            return subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", script], cwd=repo,
                                  env={**env, **more}, capture_output=True, text=True, timeout=30)
        assert step(steps["Shared blobs"]).stdout.split()[1] == "1"
        if wait < 10:                                               # landed before To check
            step('until [ -s "$RUNNER_TEMP/shared.txt.done" ]; do sleep 0.1; done')
        assert step(steps["Fetch them"]).returncode == 0 and not (temp / "blobs.txt.done").exists()
        return temp, step
    temp, step = job(0, 2)
    assert (temp / "blobs.txt.fetch").read_text().split() == blobs.split()[1:]
    assert step(head, ALL="false", N="3").stdout.split()[-1] == f"pending={temp / 'blobs.txt'}"
    assert not (temp / "blobs.txt.done").exists()
    for all_, n in (("false", "1001"), ("true", "3")):               # the whole fetch first, no list
        assert step(head, ALL=all_, N=n).stdout.split()[-1] == "pending=" and (temp / "blobs.txt.done").exists()
    done = step(steps["Fetched"])
    assert done.returncode == 0 and "3 of 3 blobs in the clone" in done.stdout, (done.stdout, done.stderr)
    temp, step = job(1, 20)
    done = step(steps["Fetched"])
    assert done.returncode == 0 and "the fetch stopped" in done.stdout, (done.stdout, done.stderr)


def test_workflow_site_build_on_shown_flags(tmp_path):
    """card-checks.yml's Commit: any change is pushed to data; site=true only when what build_site.py shows changed (hand
    and bars by card, cleared), not for new records without a flag, fields it never reads or a pushed card's same flags."""
    yaml = pytest.importorskip("yaml")
    workflow = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE))) / ".github/workflows/card-checks.yml"
    if not workflow.is_file() or not shutil.which("jq"):
        pytest.skip("no card-checks.yml or no jq")
    steps = yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"]["check"]["steps"]
    script = next(s for s in steps if s.get("id") == "commit")["run"]
    site = next(s for s in steps if s.get("name") == "Site build")
    assert site["if"] == "${{ !cancelled() && steps.commit.outputs.site == 'true' }}"
    env = {"PATH": environ["PATH"], "HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"}
    git = lambda *args: subprocess.run(["git", *args], capture_output=True, text=True, check=True, env=env).stdout.strip()
    data, origin = tmp_path / "carddata", tmp_path / "origin.git"
    git("init", "-q", "--bare", str(origin))
    git("init", "-q", "-b", "data", str(data))
    git("-C", str(data), "remote", "add", "origin", str(origin))
    db = data / "db/card_checks.json"
    db.parent.mkdir()
    checks = {"version": 1, "cards": {"a" * 40: {"card": "A", "hand": None, "bars": "Special bars.", "reads": {}, "notes": []}},
              "cleared": {}}
    db.write_text(json.dumps(checks), encoding="utf-8")
    git("-C", str(data), "add", "-A")
    git("-C", str(data), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x")
    git("-C", str(data), "push", "-q", "origin", "HEAD:refs/heads/data")

    def commit(change):
        change(checks)
        db.write_text(json.dumps(checks), encoding="utf-8")
        out = tmp_path / "out"
        out.write_text("")
        run = subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", script], cwd=data,
                             env={**env, "GITHUB_OUTPUT": str(out)}, capture_output=True, text=True, timeout=30)
        assert run.returncode == 0, (run.stdout, run.stderr)
        assert git("-C", str(origin), "rev-parse", "data") == git("-C", str(data), "rev-parse", "HEAD")
        return out.read_text().split()
    new = {"card": "B", "hand": None, "bars": None, "reads": {"hp": []}, "png": "1234abcd", "notes": [], "unredrawable": {}}
    assert commit(lambda c: c["cards"].update({"b" * 40: new})) == []                   # a new record, no flag
    assert commit(lambda c: c["cards"]["a" * 40].update(reads={"atk": []}, notes=["x"], style="new")) == []
    assert commit(lambda c: c["cards"]["b" * 40].update(hand="Bars read 5, data 6.")) == ["site=true"]
    assert commit(lambda c: c["cards"]["a" * 40].update(bars="Bars.")) == ["site=true"]
    assert commit(lambda c: c["cards"].update({"c" * 40: c["cards"].pop("a" * 40)})) == []         # A pushed, same flags
    assert commit(lambda c: c["cards"].update({"d" * 40: {**c["cards"].pop("c" * 40), "bars": None}})) == ["site=true"]
    assert commit(lambda c: c["cleared"].update({"B.psd": "1234abcd"})) == ["site=true"]
    head = git("-C", str(data), "rev-parse", "HEAD")
    assert commit(lambda c: None) == [] and git("-C", str(data), "rev-parse", "HEAD") == head    # nothing to commit


def test_blobs_of_templates():
    """A template only where first_type_spot can open one: a record with no types, its kind's; no record, every kind's."""
    shared = ["MDW/Croissant's Special Folder/Satus Effect Colors.psd", "MDW/Fonts/stoogedings-src/stoogedings.json"]
    stems = ("Creature", "2_Creature", "Item", "ActiveItem", "Status_Effect", "Raid_Boss")
    cards = ["C/A.psd", "C/B.psd", "Items/Active/D.psd", "C/E.psd"]
    tree = {path: ("", path) for path in shared + [f"MDW/Templates/{s}_Template.psd" for s in stems] + cards + ["Types/x.png", "Types/Fire.png"]}
    snapshot = SimpleNamespace(tree=tree, sha=lambda path: path, png_for=lambda rel: ("", []), dup_pictures=dict)
    stats = {"A": {"path": "C/A.psd", "card_type": "creature", "types": ["fire"]},
             "B": {"path": "C/B.psd", "card_type": "minion", "types": []},
             "D": {"path": "Items/Active/D.psd", "card_type": "item", "types": None}}
    templates = lambda rels: {p[14:-13] for p in cc.blobs_of(snapshot, rels, stats) if p.startswith("MDW/Templates/")}
    assert set(cc.blobs_of(snapshot, ["C/A.psd"], stats)) == {"C/A.psd", *shared}
    assert templates(["C/A.psd", "C/B.psd", "Items/Active/D.psd"]) == {"Creature", "ActiveItem"}
    assert templates(["C/E.psd"]) == {"Creature", "Item", "ActiveItem", "Status_Effect"}
    assert cc.blobs_of(snapshot, [], stats) == []


def test_soft_notes():
    """The old list's "bakes with flag" kinds are told nowhere; real problems and layouts stay."""
    soft = ["overflow: name reaches the stars or type icons", "overflow: name: moved +12 px to stay on the card",
            "types: the first type icon is placed at Creature_Template's spot: check it", "ability: also printed by Ability copy",
            "stars: Star (the card's own layout does not go on to 6)",
            "font: name: set in OMORI_GAME, the face its pixels show"]
    real = ["stars not baked: unsupported-card: stars: x", "font: not resolved exactly, drawn in a stand-in: A", "stars: rows of two"]
    assert real_notes(soft + real) == real and all(SOFT_NOTE.search(note) for note in soft)


def test_action_shaped_copy(tmp_path):
    """Exactly SPARSE_FILES, no config.py: --help works and the closure is that list; card-checks.yml checks it out."""
    bot = tmp_path / "bot"
    for rel in cc.SPARSE_FILES:
        (bot / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, bot / rel)
    done = subprocess.run([executable, "-m", "commands.card_checks", "--help"], cwd=bot, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    code = ("import sys, commands.card_checks, commands.txt2_repair, commands.ability_place; "
            "print(sorted(m for m in ('config', 'discord', 'pandas') if m in sys.modules)); "
            "print(sorted(m for m in sys.modules if m.startswith('commands.') or m == 'global_config'))")
    done = subprocess.run([executable, "-c", code], cwd=bot, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    bad, closure = done.stdout.splitlines()
    assert bad == "[]"
    assert {m.replace(".", "/") + ".py" for m in json.loads(closure.replace("'", '"'))} == {f for f in cc.SPARSE_FILES if f.endswith(".py")}
    workflow = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE))) / ".github/workflows/card-checks.yml"
    if workflow.is_file():
        yaml = pytest.importorskip("yaml")
        steps = yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"]["check"]["steps"]
        checkout = next(s for s in steps if s.get("with", {}).get("repository"))
        assert checkout["with"]["sparse-checkout"].split() == ["/" + f for f in cc.SPARSE_FILES]


def test_link_notes_word_the_links_against_the_tree():
    """db/links.json's hand notes (owner 11-07, plain words): a file the repo lacks (cleared once it is pushed), a
    look-alike of another linked file, a changed file Card sync could not draw again; none off the record's blob."""
    tree = {"MDW/Linked/a.png": "b1", "MDW/Linked/b.png": "b2", "Types/fire.png": "f2"}
    links = {"X/C.psd": {"blob": "c1", "links": [{"path": "MDW/Linked/gone.png", "name": "gone.png", "size": 3, "drawn": None},
                                                 {"path": "MDW/Linked/b.png", "name": "b.png", "size": 5, "drawn": None},
                                                 {"path": "Types/fire.png", "name": "fire.png", "size": 9, "drawn": "f1"}]},
             "X/D.psd": {"blob": "d1", "links": [{"path": "MDW/Linked/a.png", "name": "a.png", "size": 4, "drawn": None}]},
             "X/E.psd": {"blob": "e1", "links": [{"path": "MDW/Linked/a.png", "name": "a.png", "size": 4, "drawn": None}]},
             "files": {"Types/fire.png": {"f1": 9, "f2": 9}, "MDW/Linked/b.png": {"b2": 5}},
             "pixels": {"b1": "same", "b2": "same"}}

    def sha(path):
        return tree.get(path, "")
    alike = cc.look_alikes(links, sha, tree)
    assert alike == {"MDW/Linked/b.png": "MDW/Linked/a.png"}      # a.png has more users: b.png's get the note
    b = [{"path": "MDW/Linked/b.png", "name": "b.png", "size": 5, "drawn": None}]
    assert cc.look_alikes({**links, "X/D.psd": {"blob": "d1", "links": b}, "X/E.psd": {"blob": "e1", "links": b}},
                          sha, tree) == {"MDW/Linked/a.png": "MDW/Linked/b.png"}
    record = {"unredrawable": {"Types/fire.png": ["Fire", "a warp"]}}
    assert cc.link_notes("X/C.psd", "c1", links, sha, alike, record) == [
        "Links gone.png, which is not in the repo: push that file.",
        "Links b.png, which looks the same as a.png: link a.png instead.",
        "Fire links fire.png, changed since: Card sync could not draw it again (a warp); update it in Photoshop."]
    assert cc.link_notes("X/C.psd", "c0", links, sha, alike, record) == []
    tree["MDW/Linked/gone.png"] = "g1"
    assert len(cc.link_notes("X/C.psd", "c1", links, sha, alike, record)) == 2
