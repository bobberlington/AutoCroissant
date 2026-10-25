"""
commands/psd_details.py: the record on the goldens' PSDs (skipped without the clone), the file's text, load and prune,
a card psd-tools can't walk.
"""
import json
from types import SimpleNamespace

import pytest

import commands.psd_details as pd
from commands.font_check import FontIndex, psd_fonts
from conftest import at_rev

DOOMER = "Auxiliary/Minions/Mini_Doomer.psd"                           # template rows
CANNON = "Creatures/FTL/5 Stars/Royal_Eradicator_Main_Cannon.psd"      # atk in two rows
LAZARUS = "Creatures/Isaac/3 Stars/Tainted_Lazarus.psd"                # drawn heart cells
VIRUS = "Creatures/Kirby/3 Stars/Computer_Virus.psd"                   # no digit rows, pixel blocks; root Name layer
OVERLAY_VIRUS = "739c7abb^"                                            # Computer Virus with overlay words and image icons only
BONE = "Creatures/Kirby/4 Stars/Wham_Bam_Bone.psd"                     # its title layer is named after the card
ABBI = "Creatures/Omori/5 Stars/ABBI.psd"                              # no title layer the parser reads
AFRAID = "Auxiliary/Status Effects/Debuffs/2 Stars/Afraid.psd"         # no Text group: a top-level title named after the card
PAPYRUS = "Creatures/Undertale/3 Stars/Papyrus.psd"                    # its title layer is named after its own text
GLOBAL_TURN = "Auxiliary/Rulebook/08_Global_Turn.psd"                  # a rule page: its heading too
LICH_KING = "Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd"     # stars set by its folder: every count refused
FREEZER = "Field/1 Stars/The_Freezer.psd"                              # no Stars group; one type icon, a column of spots
FROSTMOURNE = "Auxiliary/Items/Frostmourne.psd"                        # four stars in a shown "N stars" folder
JOY = "Auxiliary/Items/Joy.psd"                                        # "N stars" folders: the folder plan
BANK = "N.M.E/The_Bank.psd"                                            # the same, two stars
MAGNUM = "Auxiliary/Items/Magnum_Opus.psd"                             # its 'field' icon is a frame: fixed
KEYS = ("path", "name", "at", "parser", "canvas", "fonts", "text", "effects", "smart_objects", "icons", "stats", "fx",
        "overlays", "template", "header", "tree")


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
        assert tuple(record) == KEYS and record["path"] == rel and record["name"] == _name(rel) and record["parser"] == 2
        assert not any(isinstance(v, str) and v.startswith("error") for v in record.values())
        assert record["canvas"] == {"w": 1200, "h": 2400, "mode": "RGB", "depth": 8, "icc": "Display"}
        assert record["fonts"] and all(e["supplied"] is True for e in record["fonts"].values())
        fonts = {font: [layer.rsplit("/", 1)[-1] for layer in e["layers"]] for font, e in record["fonts"].items()}
        assert fonts == psd_fonts(str(clone / rel))                                # the same rule as the font check
        assert all(set(t) == {"layer", "text", "font", "size", "color", "visible"} for t in record["text"])
        assert all(len(t["text"]) <= pd.TEXT_CHARS and (t["color"] is None or len(t["color"]) == 7) for t in record["text"])
        assert all(isinstance(v, list) and v for v in record["effects"].values())
        assert record["icons"] == "mixed"                                           # image types, badge runs
        assert set(record["stats"]) == {"hp", "def", "atk", "spd"}
        assert all(len(e) == (4 if e[1] == "group" else 3) and isinstance(e[2], bool) for e in record["tree"])
    assert _size(records[DOOMER]) < 8_000 and _size(records[CANNON]) < 8_000
    assert _size(records[LAZARUS]) < 20_000                                         # 234 layers, 93 with effects


def test_effects_switch_off_marks_every_effect_off(clone):
    """Springtrap's Ability has a Color Overlay under its Effects switch, off: the mock-up must not draw salt pills."""
    from psd_tools import PSDImage
    rel = "Auxiliary/Minions/Springtrap.psd"
    if not (clone / rel).exists():
        pytest.skip(f"{rel} not in the clone")
    assert pd._effects(list(PSDImage.open(str(clone / rel)).descendants()))["Text/Ability"] == ["ColorOverlay (off)"]


def test_stats_encoding(records):
    assert records[DOOMER]["stats"] == {"hp": "template", "def": "template", "atk": "template", "spd": "template"}
    assert records[CANNON]["stats"] == {"hp": "template", "def": "template", "atk": "two-row", "spd": "template"}
    assert records[LAZARUS]["stats"] == {"hp": "pixel", "def": "pixel", "atk": "pixel", "spd": "pixel"}
    assert records[VIRUS]["stats"]["hp"] == "pixel"                                 # Stat Bars/Stats Bars Blocks/hp pixel cells


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
    virus = records[VIRUS]
    hidden = next(t for t in virus["text"] if t["layer"].startswith("Alternate/"))
    assert hidden["visible"] is False and "\n" in hidden["text"] and hidden["color"] == "#01c102"
    assert virus["template"]["name_layer"] == "Name" and "no title layer" not in virus["template"]["deviations"]
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
    type icons in slot order with spots carrying the row on (a second row for eight); the bake's refusals and a card
    without a Stars group."""
    stars, types = records[DOOMER]["header"]["stars"], records[DOOMER]["header"]["types"]
    assert [u["core"] for u in stars["units"]] == [[1049, 13, 1165, 115], [936, 15, 1052, 117]] and stars["icon"] == "warpstar"
    assert stars["at"]["2"] == [[1107, 64], [994, 66]] and stars["at"]["3"] == [[1107, 64], [994, 66], [1051, 167]]
    assert stars["at"]["6"][4:] == [[1107, 269], [994, 269]] and sorted(stars["at"], key=int) == [str(n) for n in range(1, 11)]
    assert stars["refused"] == {}
    assert [t["name"] for t in types["row"]] == ["light", "defensive_minion", "wind"] and types["tokens"] is False
    assert types["spots"]["4"] == [[414, 208, 556, 356]] and [len(types["row"]) + len(types["spots"][n]) for n in types["spots"]] == [4, 5, 6, 7, 8]
    assert types["spots"]["8"][0] == [81, 378, 241, 538] and types["fixed"] == [] and types["refused"] is None
    vocab = _types(clone)
    for rel in (LICH_KING, FREEZER):
        if not (clone / rel).exists():
            pytest.skip(f"{rel} not in the clone")
    lich = pd.gather(str(clone / LICH_KING), LICH_KING, "The Lich King", None, vocab)["header"]
    assert list(lich["stars"]["refused"]) == ["*"] and lich["stars"]["refused"]["*"].endswith("folder sets them")
    assert "6" not in lich["stars"]["at"] and [t["name"] for t in lich["types"]["row"]] == ["undead", "ice"]
    freezer = pd.gather(str(clone / FREEZER), FREEZER, "The Freezer", None, vocab)["header"]
    assert freezer["stars"] is None and [t["name"] for t in freezer["types"]["row"]] == ["field"]
    assert freezer["types"]["spots"]["2"] == [[7, 226, 136, 418]]                    # under a lone icon (Item_Template's column)


def test_header_star_folders_and_fixed_icons(records, clone):
    """"N stars" folders (the shown folder's stars, the folder plan): each count has that many centres; Magnum Opus's
    'field' frame is fixed, not offered to take off; no header part is an error."""
    for rel in (FROSTMOURNE, JOY, BANK, MAGNUM):
        if not (clone / rel).exists():
            pytest.skip(f"{rel} not in the clone")
    vocab = _types(clone)
    got = {rel: pd.gather(str(clone / rel), rel, _name(rel), None, vocab)["header"] for rel in (FROSTMOURNE, JOY, BANK, MAGNUM)}
    for rel, units in ((FROSTMOURNE, 4), (JOY, 3), (BANK, 2)):
        stars = got[rel]["stars"]
        assert len(stars["units"]) == units and stars["refused"] == {}, rel
        assert all(len(stars["at"][str(n)]) == n for n in range(1, 11)), rel
    assert got[MAGNUM]["types"]["fixed"] == [["field", "'field' is a frame round the card"]] and got[MAGNUM]["types"]["refused"] is None
    for header in [*got.values(), *(r["header"] for r in records.values())]:
        assert not any(isinstance(v, str) and v.startswith("error") for v in header.values())


def test_overlay_words_and_image_icons(clone, index):
    """Computer Virus before 739c7abb: its Names words were overlay text layers, its icons images only (no badge run)."""
    virus = pd.gather(at_rev(clone, OVERLAY_VIRUS, VIRUS), VIRUS, _name(VIRUS), index, _types(clone))
    assert virus["icons"] == "images" and virus["fx"] == []
    assert virus["overlays"] == ["Names/3rd: Magician", "Names/2nd: Puppet", "Names/1st: Slime"]
    assert "overlay word layers" in virus["template"]["deviations"]


def test_template_names_the_title_layer_the_parser_reads(clone):
    """A root Name (Computer Virus) or a title named after the card (Wham Bam Bone; Afraid's at the top level) is the
    developer's choice, no deviation (the bake takes it, ledger L6); only a card whose title the parser cannot read lists
    one (ABBI)."""
    for rel, layer in ((VIRUS, "Name"), (BONE, "Text/Wham Bam Bone"), (AFRAID, "Afraid"), (ABBI, None)):
        if not (clone / rel).exists():
            pytest.skip(f"{rel} not in the clone")
        template = pd.gather(str(clone / rel), rel, _name(rel))["template"]
        assert template["name_layer"] == layer and ("no title layer" in template["deviations"]) == (layer is None)


def test_template_takes_a_title_named_after_its_text_not_a_heading(clone):
    """Papyrus's 'Text/The Great Papyrus', named after its own text, is the title the parser reads (no deviation); a rule
    page's heading named so (The Global Turn's) is no title layer, as the parser has none there."""
    for rel, layer in ((PAPYRUS, "Text/The Great Papyrus"), (GLOBAL_TURN, None)):
        if not (clone / rel).exists():
            pytest.skip(f"{rel} not in the clone")
        template = pd.gather(str(clone / rel), rel, _name(rel))["template"]
        assert template["name_layer"] == layer and "no title layer" not in template["deviations"]


def test_gather_without_an_index_leaves_supplied_null(clone):
    record = pd.gather(str(clone / DOOMER), DOOMER, "Mini Doomer")
    assert {e["supplied"] for e in record["fonts"].values()} == {None}
    pd.mark_supplied(record, FontIndex())
    assert {e["supplied"] for e in record["fonts"].values()} == {False}


def test_gather_takes_a_path_object(clone):
    record = pd.gather(clone / DOOMER, DOOMER, "Mini Doomer")
    assert isinstance(record["fonts"], dict) and record["stats"]["hp"] == "template"


def test_text_section_keeps_a_layer_whose_text_cannot_be_read():
    class Layer:
        name, parent, kind = "Name", None, "type"
        def is_visible(self):
            return True
        @property
        def text(self):
            raise ValueError("bad text data")
    rows = pd._text([Layer()])
    assert rows == [{"layer": "Name", "text": None, "font": None, "size": None, "color": None, "visible": True}]


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
def test_text_load_prune_round_trip(tmp_path):
    path = tmp_path / "d.json"
    cards = {"b" * 40: {"path": "Items/B.psd", "name": "B", "x": "é"}, "a" * 40: {"path": "Items/A.psd", "name": "A"}}
    text = pd.text_of(cards)
    path.write_text(text, encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == '{"version": 1, "cards": {' and lines[1].startswith('"' + "a" * 40 + '": {"path":"Items/A.psd"') and lines[2].endswith("}")
    assert lines[3] == "}}" and "é" in text and json.loads(text)["cards"] == cards
    assert pd.load(str(path)) == cards
    assert pd.prune(cards, {"a" * 40}) == 1 and set(cards) == {"a" * 40} and pd.prune(cards, {"a" * 40}) == 0
    assert json.loads(pd.text_of({})) == {"version": 1, "cards": {}}


def test_load_tolerates_a_missing_or_odd_file(tmp_path, capsys):
    assert pd.load(str(tmp_path / "none.json")) == {}
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "list.json").write_text("[1, 2]", encoding="utf-8")
    assert pd.load(str(tmp_path / "bad.json")) == {} and pd.load(str(tmp_path / "list.json")) == {}
    out = capsys.readouterr().out
    assert "bad.json: unreadable" in out and "list.json: not a details file" in out
