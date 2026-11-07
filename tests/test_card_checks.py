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


def test_to_check_and_file():
    """No result, or a new PNG: checked; --all, a shard; the file reads back."""
    tree = {"A.psd": ("", "11111111aa"), "A.png": ("", "22222222bb"), "B.psd": ("", "33333333cc")}
    snapshot = SimpleNamespace(psds={"a": "A.psd", "b": "B.psd"}, sha=lambda path: tree[path][1],
                               png_for=lambda rel: ("A.png" if rel == "A.psd" else "", []))
    checks = {"version": 1, "cards": {"11111111aa": {"png": "22222222"}, "33333333cc": {"png": ""}}, "cleared": {}}
    assert cc.to_check(snapshot, checks) == []
    assert cc.to_check(snapshot, checks, True) == ["A.psd", "B.psd"]
    assert cc.to_check(snapshot, checks, only={"b"}) == ["B.psd"]
    assert cc.to_check(snapshot, {"cards": {"11111111aa": {"png": "old"}}}, shard=(1, 2)) == ["A.psd", "B.psd"]
    assert json.loads(cc.text_of(checks))["cards"] == checks["cards"]


def test_soft_notes():
    """The old list's "bakes with flag" kinds are told nowhere; real problems and layouts stay."""
    soft = ["overflow: name reaches the stars or type icons", "overflow: name: moved +12 px to stay on the card",
            "types: the first type icon is placed at Creature_Template's spot: check it", "ability: also printed by Ability copy",
            "stars: Star (the card's own layout does not go on to 6)", "validator: SEVERAL CREATURES: check each one's stats on the website",
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
