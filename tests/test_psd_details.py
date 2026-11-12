"""
commands/psd_details.py: the record on the goldens' PSDs, the file's text and prune, a card psd-tools can't walk.
"""
import json
from types import SimpleNamespace

import pytest

import commands.psd_details as pd
from commands.font_check import FontIndex, psd_fonts

DOOMER = "Auxiliary/Minions/Mini_Doomer.psd"                           # template rows
CANNON = "Creatures/FTL/5 Stars/Royal_Eradicator_Main_Cannon.psd"      # atk in two rows
LAZARUS = "Creatures/Isaac/3 Stars/Tainted_Lazarus.psd"                # drawn heart cells
VIRUS = "Creatures/Kirby/3 Stars/Computer_Virus.psd"                   # no digit rows, pixel blocks; root Name layer
LICH_KING = "Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd"     # stars set by its folder: planned as a move
FREEZER = "Field/1 Stars/The_Freezer.psd"                              # no Stars group; one type icon, a column of spots
KEYS = ("path", "name", "at", "parser", "version", "canvas", "fonts", "text", "effects", "smart_objects", "icons", "stats", "fx",
        "overlays", "template", "header", "tree", "parts")


# ========================
# Helpers
# ========================
def _name(rel: str) -> str:
    return rel.rsplit("/", 1)[1][:-4].replace("_", " ")


def _types(clone) -> list:
    """The clone's Types/ icon names (file stems, lowercased)."""
    return sorted(p.stem.lower() for p in (clone / "Types").iterdir() if not p.name.startswith(".") and p.suffix)


def _size(record: dict) -> int:
    return len(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


@pytest.fixture(scope="module")
def index(clone):
    return FontIndex(str(clone / "MDW/Fonts"))


@pytest.fixture(scope="module")
def records(clone, index):
    vocab = _types(clone)
    return {rel: pd.gather(str(clone / rel), rel, _name(rel), index, vocab) for rel in (DOOMER, CANNON, LAZARUS, VIRUS)}


# ========================
# The record
# ========================
def test_record_shape_and_size(records, clone):
    for rel, record in records.items():
        assert tuple(k for k in record if k not in ("creature_bars", "stat_caps")) == KEYS and record["path"] == rel \
            and record["name"] == _name(rel) and (record["parser"], record["version"]) == (pd.PARSER_VERSION, pd.VERSION)
        assert record.get("creature_bars") == ({"count": 2, "shared": []} if rel == LAZARUS else None)   # a row per creature
        assert not any(isinstance(v, str) and v.startswith("error") for v in record.values())
        assert record["canvas"] == {"w": 1200, "h": 2400, "mode": "RGB", "depth": 8, "icc": "Display"}
        assert record["fonts"] and all(e["supplied"] is True for e in record["fonts"].values())
        fonts = {font: [layer.rsplit("/", 1)[-1] for layer in e["layers"]] for font, e in record["fonts"].items()}
        assert fonts == psd_fonts(str(clone / rel))                                # the same rule as the font check
        assert all(set(t) - {"colors", "runs", "frame", "hx", "track", "lf", "badges", "just"} == {"layer", "id", "box", "text", "font", "size", "color", "visible", "px"}
                   for t in record["text"])
        assert all(t["color"] in t.get("colors", [t["color"]]) for t in record["text"] if t["color"])     # every run's colour
        assert all(len(t["text"]) <= pd.TEXT_CHARS and (t["color"] is None or len(t["color"]) == 7) for t in record["text"])
        assert all(isinstance(v, list) and v for v in record["effects"].values())
        assert record["icons"] == "mixed"                                           # image types, badge runs
        assert set(record["stats"]) == {"hp", "def", "atk", "spd"}
        assert all(len(e) == (4 if e[1] == "group" else 3) and isinstance(e[2], bool) for e in record["tree"])
    assert _size(records[DOOMER]) < 10_000 and _size(records[CANNON]) < 10_000     # parts: ~1.5 KB more
    assert _size(records[LAZARUS]) < 21_000                                         # 234 layers, 93 with effects; loose stars
    caps = {rel: record.get("stat_caps") for rel, record in records.items()}     # VERSION 4: own displays (old rows then)
    assert caps[LAZARUS] == dict.fromkeys(("hp", "def", "atk", "spd"), [10, 10]) and caps[VIRUS] is None
    assert caps[CANNON] == {"hp": [10], "def": [10], "atk": [20], "spd": [10]}                  # Top and Bottom rows: 20


def test_how_text_is_set(clone):
    """The mock-up's wrap (VERSION 5): Wendy's Ability is set 2.34 times wider than tall, Isma's tracked -20. VERSION 6:
    each badge's [em, track] (null: not in the icon font; Wendy's 1CD tracked 418), Chef Saltbaker's letters' font."""
    from psd_tools import PSDImage
    got = {}
    for rel, lid in (("Creatures/DST/3 Stars/Wendy.psd", 417), ("Creatures/Hollow Knight/4 Stars/Isma.psd", 417),
                     ("Creatures/Cuphead/5 Stars/Chef_Saltbaker.psd", 417)):
        layer = next(x for x in PSDImage.open(str(clone / rel)).descendants() if x.layer_id == lid)
        got[rel.rsplit("/", 1)[1]] = pd._set(layer, pd._style(layer)[0])
    assert got["Wendy.psd"] == {"frame": [2, 1704, 1202, 2314], "px": 88.04, "hx": 2.336, "badges": [[1.0, 418], None]}
    assert got["Isma.psd"] == {"frame": [-1, 1763, 1201, 2486], "px": 73.92, "track": -20.0,     # a 0.1 degree turn: still set
                               "badges": [[0.814, 0], [0.814, 345], None, [0.814, 0]]}
    assert got["Chef_Saltbaker.psd"]["lf"] == "CupheadMemphis-Medium"                # first run MyriadPro-Regular


def test_effects_switch_off_marks_every_effect_off(clone):
    """Springtrap's Ability has a Color Overlay under its Effects switch, off: the mock-up must not draw salt pills."""
    from psd_tools import PSDImage
    rel = "Auxiliary/Minions/Springtrap.psd"
    assert pd._effects(list(PSDImage.open(str(clone / rel)).descendants()))["Text/Ability"] == ["ColorOverlay (off)"]


def test_text_layers_and_template(records):
    doomer = records[DOOMER]
    ability = next(t for t in doomer["text"] if t["layer"] == "Text/Ability")
    assert ability["font"] == "FOTPopHappinessStrp-ExtraBold" and ability["size"] == 50.0 and ability["color"] == "#ffffff"
    assert ability["visible"] and ability["text"].startswith("Has stats equal to")
    assert doomer["template"] == {"name_layer": "Text/Name", "ability_layers": ["Text/Ability"], "deviations": ["icons as images"]}
    assert doomer["smart_objects"][:2] == ["images/Background (1)", "Stars/Warpstar"] and doomer["overlays"] == []
    badge = ["Ability FX: (BOT)"] if doomer["fx"] else []      # the badge's gradient; retired by the 2026-10-25 batch
    assert doomer["fx"] == badge and (["Ability FX: (BOT)", "pixel", True] in doomer["tree"]) == bool(badge)
    assert doomer["fonts"]["stoogedings-Regular"]["layers"] == ["Text/Ability"]                        # the (BOT) badge run
    assert doomer["effects"]["Creature Types"] == ["Stroke (off)"] and doomer["effects"]["Creature Types/Light"] == ["DropShadow", "OuterGlow"]
    cannon = records[CANNON]
    assert cannon["overlays"] == [] and cannon["fx"] == ["Ability FX: Blazing"] and cannon["template"]["deviations"] == ["icons as images"]
    assert cannon["effects"]["Ability FX: Blazing"] == ["OuterGlow"] and set(cannon["fonts"]["stoogedings-Regular"]["layers"]) == {"Text/Ability"}
    assert cannon["effects"]["Stars/stars"] == ["GradientOverlay", "BevelEmboss"]
    runs = next(t for t in cannon["text"] if t["layer"] == "Text/Ability")             # VERSION 3: where each colour is
    words = {runs["text"][a:b].strip(): runs["colors"][k] for a, b, k in runs["runs"]}
    assert words["Blazing"] == "#ff2a1a" and words["Afraid"] == "#0060ff" and runs["text"].endswith("in that slot.")
    assert all(a < b <= c for (a, b, _), (c, _, _) in zip(runs["runs"], runs["runs"][1:]))
    assert "runs" not in ability                                                       # one colour
    name = next(t for t in doomer["text"] if t["layer"] == doomer["template"]["name_layer"])
    assert ability["frame"] == [0, 1721, 1185, 2376] and ability["px"] == 150.0 and "frame" not in name   # 50 pt by 3; point text
    assert "just" not in name                                                          # centred
    virus = records[VIRUS]
    hidden = next(t for t in virus["text"] if t["layer"].startswith("Alternate/"))
    assert hidden["visible"] is False and "\n" in hidden["text"] and hidden["color"] == "#01c102"
    assert virus["template"]["name_layer"] == "Name" and "no title layer" not in virus["template"]["deviations"]
    assert next(t for t in virus["text"] if t["layer"] == "Name")["just"] == 0         # VERSION 6: left set
    assert virus["overlays"] == [] and "overlay word layers" not in virus["template"]["deviations"]   # L7 (739c7abb)
    assert virus["fx"] == ["Names/Ability FX: 3rd: Magician", "Names/Ability FX: 2nd: Puppet", "Names/Ability FX: 1st: Slime"]
    lazarus = records[LAZARUS]
    assert lazarus["template"]["ability_layers"] == ["Laz/Laz/Ability", "Dead Laz/Dead Laz/Ability"]
    assert lazarus["template"]["name_layer"] == "Laz/Laz/Name"
    assert lazarus["template"]["deviations"] == ["several visible Ability layers", "ability layer outside Text", "stat bars drawn",
                                                 "icons as images"]
    assert [e[:3] for e in lazarus["tree"]] == [["images", "group", True], ["Laz", "group", True], ["Dead Laz", "group", True]]


def test_header_boxes(records, clone):
    """The mock-up's header: Mini Doomer's two stars in reading order, rows of two past them, no count refused; its three
    type icons in slot order with spots carrying the row on (a second row for eight); a folder card's counts planned as
    its move (card_bake's), a card without a Stars group too (its loose stars: The Freezer's Patrick Star)."""
    stars, types = records[DOOMER]["header"]["stars"], records[DOOMER]["header"]["types"]
    assert [u["core"] for u in stars["units"]] == [[1049, 13, 1164, 115], [936, 15, 1051, 117]] and stars["icon"] == "warpstar"
    assert stars["at"]["2"] == [[1106, 64], [994, 66]] and stars["at"]["3"] == [[1106, 64], [994, 66], [1050, 167]]
    assert stars["at"]["6"][4:] == [[1106, 269], [994, 269]] and sorted(stars["at"], key=int) == [str(n) for n in range(1, 11)]
    assert stars["refused"] == {}
    assert [t["name"] for t in types["row"]] == ["light", "defensive_minion", "wind"] and types["tokens"] is False
    assert types["spots"]["4"] == [[414, 208, 556, 356]] and [len(types["row"]) + len(types["spots"][n]) for n in types["spots"]] == [4, 5, 6, 7, 8]
    assert types["spots"]["8"][0] == [81, 378, 241, 537] and types["fixed"] == [] and types["refused"] is None
    vocab = _types(clone)
    lich = pd.gather(str(clone / LICH_KING), LICH_KING, "The Lich King", None, vocab)["header"]
    assert lich["stars"]["refused"] == {} and sorted(lich["stars"]["at"], key=int) == [str(n) for n in range(1, 11)]
    assert lich["stars"]["at"]["6"][4:] == [[1130, 317], [998, 317]] and [t["name"] for t in lich["types"]["row"]] == ["undead", "ice"]
    freezer = pd.gather(str(clone / FREEZER), FREEZER, "The Freezer", None, vocab)["header"]
    stars = freezer["stars"]                            # no Stars group: its Patrick Star, laid out loose as the bake's
    assert [u["core"] for u in stars["units"]] == [[1061, 31, 1189, 198]] and stars["icon"] == "patrick star"
    assert stars["at"]["2"] == [[1125, 114], [1125, 282]] and stars["refused"] == {}     # past 5: stepped down (eef1b6b)
    assert stars["at"]["6"][:2] == [[1125, 114], [997, 182]] and len(stars["at"]["10"]) == 10
    assert [t["name"] for t in freezer["types"]["row"]] == ["field"]
    assert freezer["types"]["spots"]["2"] == [[7, 226, 136, 418]]                    # under a lone icon (Item_Template's column)


def test_gather_without_an_index_leaves_supplied_null(clone):
    record = pd.gather(str(clone / DOOMER), DOOMER, "Mini Doomer")
    assert {e["supplied"] for e in record["fonts"].values()} == {None}
    pd.mark_supplied(record, FontIndex())
    assert {e["supplied"] for e in record["fonts"].values()} == {False}


def test_runs_over_the_cleaned_text():
    """Runs move past a byte-order mark, a blank run joins the one before, one colour shows none."""
    red, white, blue = (1, 0, 0), (1, 1, 1), (0, 0, 1)
    text, at = pd._cleaned("\ufeffHot\rcold")
    assert text == "Hot\ncold" and at == [0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    colours = ["#ff0000", "#ffffff", "#0000ff"]
    assert pd._runs([(0, 4, red), (4, 5, white), (5, 9, blue)], text, at, colours) == [[0, 4, 0], [4, 8, 2]]
    assert pd._runs([(0, 4, red), (4, 5, white), (5, 9, red)], text, at, colours) == []           # a blank joins: one colour
    assert pd._runs([(0, 1, white), (1, 9, red)], " Hot cold", list(range(10)), colours) == [[0, 1, 1], [1, 9, 0]]


def test_gather_never_raises_for_a_broken_card():
    class Broken:
        width, height, depth = 1, 2, 8
        color_mode = SimpleNamespace(name="RGB")
        image_resources = SimpleNamespace(get_data=lambda key: None)
        def descendants(self):
            raise ValueError("bad layer records")
    record = pd.gather(Broken(), "Items/X.psd", "X")
    assert record["canvas"] == {"w": 1, "h": 2, "mode": "RGB", "depth": 8, "icc": None}
    assert record["fonts"] == record["tree"] == "error: ValueError: bad layer records"


class Group:
    """A group as _tree iterates it."""
    kind = "group"

    def __init__(self, name, children, visible=True):
        self.name, self.children, self.visible = name, children, visible

    def is_group(self):
        return True

    def __iter__(self):
        return iter(self.children)


def test_tree_is_cut_at_the_caps():
    leaves = [SimpleNamespace(name=f"L{i}", kind="pixel", visible=True, is_group=lambda: False) for i in range(pd.TREE_CHILDREN + 1)]
    tree = pd._tree(Group("root", [Group("G", leaves)]))
    assert tree[0][:3] == ["G", "group", True] and len(tree[0][3]) == pd.TREE_CHILDREN + 1 and tree[0][3][pd.TREE_CHILDREN] == pd.MORE
    assert tree[0][3][0] == ["L0", "pixel", True]
    deep = Group("D", [], visible=False)
    deep.children = [deep]
    nested, depth = pd._tree(Group("root", [deep])), 0
    while nested and nested[0][0] == "D":
        nested, depth = nested[0][3], depth + 1
    assert depth == pd.TREE_DEPTH and nested == [pd.MORE]


# ========================
# The file
# ========================
def test_text_prune_round_trip():
    cards = {"b" * 40: {"path": "Items/B.psd", "name": "B", "x": "é"}, "a" * 40: {"path": "Items/A.psd", "name": "A"}}
    text = pd.text_of(cards)
    lines = text.splitlines()
    assert lines[0] == '{"version": %d, "cards": {' % pd.VERSION and lines[1].startswith('"' + "a" * 40 + '": {"path":"Items/A.psd"') and lines[2].endswith("}")
    assert lines[3] == "}}" and "é" in text and json.loads(text)["cards"] == cards
    assert pd.prune(cards, {"a" * 40}) == 1 and set(cards) == {"a" * 40} and pd.prune(cards, {"a" * 40}) == 0
    assert json.loads(pd.text_of({})) == {"version": pd.VERSION, "cards": {}}
