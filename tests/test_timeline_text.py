"""scripts/timeline_text.py (by hand): the card timelines' title, ability, stars and types readers and their bars."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "timeline_text.py"


@pytest.fixture(scope="module")
def tt():
    spec = importlib.util.spec_from_file_location("timeline_text", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def line(text, x, y, w, h):
    return [text, 1.0, x, y, w, h, []]


def test_the_ability_under_the_bars_without_the_class_line(tt):
    """Lines under the Speed label make the ability (a first line right under the bars kept, the class line under
    CLASS_Y dropped); tokens and punctuation compare as words."""
    labels = {"hp": (0.02, 0.50, 0.2, 0.52), "def": (0.02, 0.54, 0.2, 0.56), "atk": (0.02, 0.58, 0.2, 0.60),
              "spd": (0.02, 0.62, 0.2, 0.64)}
    lines = [line("Health", 0.02, 0.50, 0.18, 0.02), line("Summon a Grodus X for every 3 HP", 0.0, 0.641, 0.9, 0.02),
             line("this creature loses.", 0.0, 0.67, 0.5, 0.02), line("Attack.", 0.0, 0.70, 0.2, 0.02),
             line("X Naut/Panda", 0.35, 0.975, 0.3, 0.02)]
    rows, cls = tt.ability_block(lines, "B", labels, 0.135, "card")
    assert tt.ability_text(rows) == "Summon a Grodus X for every 3 HP this creature loses. Attack."
    assert " ".join(l[0] for l in cls) == "X Naut/Panda"
    assert tt.ability_words("[fire]: +1 ATK.") == ["<fire>", "1", "atk"]
    assert tt.is_right("ability", "Deal 2 damage, then (OPT) heal", "deal 2 damage then (opt) heal.")


def test_a_title_takes_the_nearest_name_with_its_margin(tt, monkeypatch):
    """The tallest band lines (two-line titles joined) against every name: a misread takes its name, a near tie
    costs confidence, no name under NAME_MIN."""
    monkeypatch.setattr(tt, "_NAMES", [(n, tt.loose(n)) for n in ("Old Chica", "Old Freddy", "Toy Chica")])
    got = tt.read_title([line("OLD GHICA", 0.3, 0.02, 0.4, 0.05), line("★★", 0.8, 0.02, 0.1, 0.05)])
    assert got["name"] == "Old Chica" and got["s1"] > got["s2"] and got["conf"] <= got["s1"]
    assert tt.read_title([line("Zzyzx Qwv", 0.3, 0.02, 0.4, 0.05)])["name"] is None


def test_star_blobs_rows_and_clusters(tt):
    """Era A: like blobs in up to two rows count, an art blob of another size sharing the row makes it unsure;
    era B: an overlapped cluster's width over height."""
    sun = lambda x, y, a=100: [a, x, y, 0.03, 0.03]
    row = {"blobs": [sun(0.6, 0.02), sun(0.65, 0.02, 95), sun(0.7, 0.02, 105), sun(0.62, 0.06), sun(0.68, 0.06)]}
    assert tt.count_blobs(row, {"kind": "row", "tol": 1.43, "rows": 1.6}) == (5, 1.0)
    assert tt.count_blobs(row, {"kind": "row", "tol": 1.43, "rows": 0.6})[0] == 3
    art = {"blobs": [[300, 0.6, 0.02, 0.05, 0.03]] + row["blobs"][:3]}
    assert tt.count_blobs(art, {"kind": "row", "tol": 1.43, "rows": 0.6}) == (3, 0.3)
    side = 0.05
    cluster = {"blobs": [[int(0.55 * (2.11 * side * tt.WORK_W) * (side * tt.WORK_W)), 0.7, 0.0, 2.11 * side, side]]}
    assert tt.count_blobs(cluster, {"kind": "cluster", "step": 0.37, "base": 1.0})[0] == 4


def test_types_are_the_one_known_set_the_picture_shows(tt, monkeypatch):
    """A known set whose icons are all there while the others' are not; equip on a creature is the minion mark;
    a set known only from the version itself (exclude) is never offered."""
    monkeypatch.setattr(tt, "_SAME", {"undread": "undead"})
    rd = {"kind": "card", "types_known": {
        "sets": [[["fire"], ["psd:a"]], [["fire", "machine"], ["t:x"]], [["undread", "equip"], ["psd:b"]]],
        "presence": {"fire": [0.93, 0.8, 0, 0, 9], "machine": [0.41, 0.1, 0, 0, 9], "undead": [0.5, 0.2, 0, 0, 9]}}}
    assert tt.verified_types(rd, {"on": 0.8, "off": 0.7}) == (["fire"], 0.13)
    assert tt.verified_types(rd, {"on": 0.8, "off": 0.7}, exclude=["psd:a"]) == (None, 0.0)
    assert tt.is_right("types", ["undead"], ["undread", "equip"], card=True)


def test_the_bar_keeps_the_target_share_right(tt):
    """The lowest confidence whose reads at or above it are >= BAR_TARGET right; ties stay together; a wrong
    top read leaves no bar."""
    rows = [(1.0, True)] * 60 + [(0.9, True)] * 30 + [(0.9, False)] + [(0.5, False)] * 5
    bar, k, ok = tt.pick_bar(rows)
    assert (bar, k, ok) == (0.9, 91, 90)
    assert tt.pick_bar([(1.0, False), (0.9, True)]) == (None, 0, 0)


def test_template_eras_by_labels_and_size(tt):
    """Stat labels' right edge and the look give the era of a picture stage 1 didn't classify."""
    lab = lambda x1: {s: (0.02, y, x1, y + 0.02) for s, y in zip(("hp", "def", "atk", "spd"), (0.5, 0.55, 0.6, 0.65))}
    look = {"frame_white": 0.7, "low_black": 0.0, "low_white": 0.5}
    item = {"src": "discord", "file": "attachments/none.png"}
    tt._STAGE1 = {}
    assert tt.era_of(item, 390, 790, lab(0.19), look)[0] == "A"
    assert tt.era_of(item, 390, 790, lab(0.19), dict(look, frame_white=0.1))[0] == "C"
    assert tt.era_of(item, 1200, 2400, lab(0.28), look)[0] == "D2"
    assert tt.era_of({"src": "old"}, 1200, 1800, {}, look)[0] == "D-item"
    arr = np.zeros((10, 10, 3), np.uint8)
    assert tt.look_of(arr)["low_black"] == 1.0
