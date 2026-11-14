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
    rows, cls, odd = tt.ability_block(lines, "B", labels, 0.135, "card")
    assert tt.ability_text(rows) == "Summon a Grodus X for every 3 HP this creature loses. Attack."
    assert " ".join(l[0] for l in cls) == "X Naut/Panda" and odd == 0
    # a lone class line is the class (no ability); an ability's last line is never the class; a line in another
    # script (a lost word) counts as odd and halves the confidence
    assert tt.ability_block(lines[:1] + lines[-1:], "B", labels, 0.135, "card")[:2] == ([], [lines[-1]])
    for last in ("(OD) Revive as Morpho Knight,", "slot of your choice"):
        rows, cls, odd = tt.ability_block(lines[:2] + [line(last, 0.35, 0.975, 0.3, 0.02)], "B", labels, 0.135, "card")
        assert cls is None and tt.ability_text(rows).endswith(last)
    caps = line("ROYALTY/MAGE/HIGHER BEING,", 0.3, 0.975, 0.4, 0.02)
    assert tt.ability_block(lines[:2] + [caps], "B", labels, 0.135, "card")[1] == [caps]
    rows, cls, odd = tt.ability_block(lines[:4] + [line("\u0435\u043f\u0435\u0442\u0438.", 0.4, 0.73, 0.2, 0.02)],
                                      "B", labels, 0.135, "card")
    assert odd == 1 and tt.ability_conf("Attack.", True, 1.0, True, odd) == 0.5
    art = line("\u0441\u043e\u043b\u0442", 0.8, 0.96, 0.1, 0.02)          # art text far under the ability
    assert tt.ability_block(lines[:4] + [art], "B", labels, 0.135, "card")[2] == 0
    low = [line("Deal 2 damage to a", 0.0, 0.90, 0.9, 0.02), line("random creature.", 0.0, 0.925, 0.9, 0.02),
           line("\u0421\u0422\u0415\u0410\u0422\u041e\u0419", 0.4, 0.955, 0.2, 0.02)]   # a class line in capitals
    assert tt.ability_block(low, "B", labels, 0.135, "card")[2] == 0
    assert tt.ability_words("[fire]: +1 ATK.") == ["<fire>", "1", "atk"]
    # a letter one OCR pass gave in another script (Cyrillic) is a different word, not a dropped one
    assert not tt.passes_agree("(\u0410AT(OPT)This card", "(AT (OPT) This card")
    assert tt.passes_agree("(AAT)(OPT)This card", "(AAT) (OPT) This card.")
    assert tt.is_right("ability", "Deal 2 damage, then (OPT) heal", "deal 2 damage then (opt) heal.")
    tt._TAGS = {"OYT", "1CD"}
    assert tt.odd_marks("(QYT) Heal 2") == 1 and tt.odd_marks("(OYT) Heal 2 (1CD)") == 0


def test_a_title_takes_the_nearest_name_with_its_margin(tt, monkeypatch):
    """The tallest band lines joined against every name: a misread takes its name, a near tie costs confidence,
    no name under NAME_MIN; a two-line title keeps the longer name a line's name sits in; a line naming another
    card makes the read unsure."""
    names = ("Old Chica", "Old Freddy", "Toy Chica", "Tentacle", "Blooper Tentacle", "Metronome", "Golden Metronome",
             "Dark Mind", "Final Dark Mind Eye", "Pie")
    monkeypatch.setattr(tt, "_NAMES", [(n, tt.loose(n)) for n in names])
    got = tt.read_title([line("OLD GHICA", 0.3, 0.02, 0.4, 0.05), line("★★", 0.8, 0.02, 0.1, 0.05)])
    assert got["name"] == "Old Chica" and got["s1"] > got["s2"] and got["conf"] <= got["s1"]
    assert tt.read_title([line("Zzyzx Qwv", 0.3, 0.02, 0.4, 0.05)])["name"] is None
    two = lambda *ts: tt.read_title([line(t, 0.3, 0.02 + 0.06 * i, 0.4, 0.05) for i, t in enumerate(ts)])
    assert two("BLOOPER", "TENTACLE")["name"] == "Blooper Tentacle" and two("BLOOPER", "TENTACLE")["conf"] == 1.0
    assert two("GOLDEN", "METRONOME")["name"] == "Golden Metronome"
    assert two("Final", "Dark Mind", "Eve")["name"] == "Final Dark Mind Eye"
    clash = two("Pie", "OLD FREDDY")
    assert clash.get("clash") and clash["conf"] <= 0.5


def test_star_blobs_rows_and_clusters(tt):
    """Era A: like blobs in up to two rows count, an art blob of another size sharing the row makes it unsure;
    era B: an overlapped cluster's width over height."""
    sun = lambda x, y, a=60: [a, x, y, 0.03, 0.03]
    row = {"blobs": [sun(0.6, 0.02), sun(0.65, 0.02, 57), sun(0.7, 0.02, 63), sun(0.62, 0.06), sun(0.68, 0.06)]}
    assert tt.count_blobs(row, {"kind": "row", "tol": 1.43, "rows": 1.6}) == (5, 1.0)
    assert tt.count_blobs(row, {"kind": "row", "tol": 1.43, "rows": 0.6})[0] == 3
    art = {"blobs": [[180, 0.6, 0.02, 0.05, 0.03]] + row["blobs"][:3]}
    assert tt.count_blobs(art, {"kind": "row", "tol": 1.43, "rows": 0.6}) == (3, 0.3)
    side = 0.05
    cluster = {"blobs": [[int(0.55 * (2.11 * side * tt.WORK_W) * (side * tt.WORK_W)), 0.7, 0.0, 2.11 * side, side]]}
    assert tt.count_blobs(cluster, {"kind": "cluster", "step": 0.37, "base": 1.0})[0] == 4
    wall = {"blobs": [[5800, 0.04, 0.0, 0.29, 0.2]] + row["blobs"][:2]}           # a status card's yellow wall
    assert tt.count_blobs(wall, {"kind": "row", "tol": 1.7, "rows": 1.6}) == (None, 0.0)
    flat = {"blobs": [[110, 0.0, 0.08, 0.03, 0.03], [100, 0.1, 0.08, 0.03, 0.03]]}        # full boxes: no stars
    assert tt.count_blobs(flat, {"kind": "row", "tol": 1.7, "rows": 1.6}) == (None, 0.0)


def test_stars_never_count_a_full_copy_list(tt):
    """Every copy kept a star (the copy list full): no count; no non-star copy: the margin is taken from the floor;
    A/C plain counts are never sure, nor a count of 1 in B-item and F."""
    full = {"icon": "star", "bank": 0.9, "size": 0.05, "copies": [[1.0, 0.0, i, 0] for i in range(tt.MAX_COPIES)]}
    assert tt.count_stars(full, {"ncc": 0.85, "mad": 30}) == (None, 0.0)
    two = dict(full, copies=[[1.0, 0.0, 0, 0], [0.9, 2.0, 9, 0]])
    n, conf = tt.count_stars(two, {"ncc": 0.85, "mad": 30})
    assert n == 2 and conf == round(1.0 * min(1.0, 0.5 + 0.9 - 2 / 160 - tt.COPY_FLOOR), 4)
    rd = {"era": "C", "title": {"name": None}, "stars": two}
    assert tt.field_read(rd, "stars", {"rule": {"ncc": 0.85, "mad": 30}}) == (2, "icon", 0.0)
    one = dict(rd, stars=dict(two, copies=[[1.0, 0.0, 0, 0], [0.3, 60.0, 9, 0]]))
    assert tt.field_read(one, "stars", {"rule": {"ncc": 0.85, "mad": 30}})[2] == 0.0
    for era in ("B-item", "F"):                        # overlapped or unlike stars read 1: never sure, known or not
        assert tt.field_read(dict(one, era=era), "stars", {"rule": {"ncc": 0.85, "mad": 30}}) == (1, "icon", 0.0)


def test_star_counts_in_doubt_are_never_sure(tt):
    """A star cut by the picture's right edge is found (read_edge) and makes the count doubtful, as do shurikens and a
    copy on a title letter; two whole stars alone are no doubt."""
    import cv2
    side, y, x_from = 30, 20, 162
    star = np.array([[15, 0], [19, 11], [30, 11], [21, 18], [25, 30], [15, 23], [5, 30], [9, 18], [0, 11], [11, 11]])

    def card(xs):
        a = np.full((720, 360, 3), 30, np.uint8)
        for x in xs:
            cv2.fillPoly(a, [star + [x, y]], (230, 200, 40))
        return a
    st = {"icon": "star", "size": side / 360, "copies": [[1.0, 0.0, 250 - x_from, y], [0.95, 1.0, 286 - x_from, y],
                                                          [0.3, 90.0, 10, 100]]}
    rule = {"ncc": 0.7, "mad": 45}
    cut = tt.read_edge(card([250, 286, 340]), st, 0.28, x_from)
    assert cut and cut[0][0] > 0.9 and cut[0][3] in range(y - 2, y + 3)
    assert tt.star_doubt({"stars": st, "star_edge": {"x_from": x_from, "edge": cut}}, rule) == "a star cut by the edge"
    whole = tt.read_edge(card([250, 286]), st, 0.28, x_from)
    assert tt.star_doubt({"stars": st, "star_edge": {"x_from": x_from, "edge": whole}}, rule) is None
    assert tt.star_doubt({"stars": dict(st, icon="ninja star")}, rule) == "shuriken stars"
    word = [[240, 0, 330, 100]]                                     # a title word four times the copies' height
    small = dict(st, size=12 / 360)
    assert tt.star_doubt({"stars": small, "star_edge": {"x_from": x_from, "words": word}}, rule) == \
        "a title letter counted"
    assert tt.star_doubt({"stars": st, "star_edge": {"x_from": x_from, "words": [[240, 10, 330, 60]]}}, rule) is None


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
    # an icon no known set has, over its alarm level beside the set's icon: unsure; on the set's icon: no alarm
    rd["types_known"]["presence"]["skull"] = [[0.95, 0.9, 20, 0, 9], [0.5, 0.1, 40, 0, 9]]
    assert tt.verified_types(rd, {"on": 0.8, "off": 0.7}, alarms={"skull": 0.9}) == (["fire"], 0.0)
    rd["types_known"]["presence"]["skull"] = [[0.95, 0.9, 0, 0, 9]]
    assert tt.verified_types(rd, {"on": 0.8, "off": 0.7}, alarms={"skull": 0.9}) == (["fire"], 0.13)


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
    tt._BAR_ERAS = {"attachments/none.png": "C-item"}       # no rule fits: the bar readers' era
    assert tt.era_of(item, 500, 600, {}, dict(look, frame_white=0.1, low_white=0.0)) == ("C-item", "bars")
    assert tt.era_of(item, 390, 790, lab(0.19), look)[0] == "A"
    tt._BAR_ERAS = None
    arr = np.zeros((10, 10, 3), np.uint8)
    assert tt.look_of(arr)["low_black"] == 1.0


def test_field_cards_and_template_exports_have_their_own_text_eras(tt):
    """A field card (the bar readers' era F) reads every paragraph under the art; a 1200 px export whose labels sit
    off the template's (bars-other) reads as the alt template (D3) or the item template (D-item) by its size."""
    look = {"frame_white": 0.1, "low_black": 0.0, "low_white": 0.0}
    item = {"src": "discord", "file": "attachments/f.png"}
    tt._STAGE1, tt._BAR_ERAS = {}, {"attachments/f.png": "F"}
    assert tt.era_of(item, 880, 1130, {}, look) == ("F", "bars")
    tt._BAR_ERAS = {}
    lab = {s: (0.02, y, 0.4, y + 0.02) for s, y in zip(("hp", "def", "atk", "spd"), (0.5, 0.55, 0.6, 0.65))}
    assert tt.era_of(item, 1200, 2400, lab, look) == ("D3", "size")
    assert tt.era_of(item, 600, 950, lab, look)[0] == "bars-other"
    tt._BAR_ERAS = None
    lines = [line("Cloud Soiree", 0.2, 0.02, 0.6, 0.05), line("Quest: Have less than 10 stars,", 0.0, 0.5, 0.9, 0.03),
             line(": +1 ATK.", 0.1, 0.62, 0.3, 0.03), line("Nature: +2 SPD.", 0.0, 0.70, 0.4, 0.03),
             line("Industrial: -2 DEF.", 0.0, 0.92, 0.4, 0.03)]
    rows, cls, odd = tt.ability_block(lines, "F", {}, 0.2, "other")
    assert tt.ability_text(rows) == "Quest: Have less than 10 stars, : +1 ATK. Nature: +2 SPD. Industrial: -2 DEF."
