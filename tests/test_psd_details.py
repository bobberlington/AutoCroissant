"""
commands/psd_details.py: the record on the goldens' PSDs (skipped without the clone), load/save/prune, the backfill CLI on
two cards, a card psd-tools can't walk, /push carrying the file.
"""
import asyncio
import json
import shutil
from types import SimpleNamespace

import pytest

import commands.psd_details as pd
import commands.update_bot as ub
from commands.font_check import FontIndex, psd_fonts
from conftest import blob_sha
from global_config import DETAILS_JSON

DOOMER = "Auxiliary/Minions/Mini_Doomer.psd"                           # template rows
CANNON = "Creatures/FTL/5 Stars/Royal_Eradicator_Main_Cannon.psd"      # atk in two rows
LAZARUS = "Creatures/Isaac/3 Stars/Tainted_Lazarus.psd"                # drawn heart cells
VIRUS = "Creatures/Kirby/3 Stars/Computer_Virus.psd"                   # no digit rows, pixel blocks; root Name layer
KEYS = ("path", "name", "at", "parser", "canvas", "fonts", "text", "effects", "smart_objects", "icons", "stats", "fx",
        "overlays", "template", "tree")


# ========================
# Helpers
# ========================
def _name(rel: str) -> str:
    return rel.rsplit("/", 1)[1][:-4].replace("_", " ")


def _size(record: dict) -> int:
    return len(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


@pytest.fixture(scope="module")
def index(clone):
    return FontIndex(str(clone / "MDW/Fonts"))


@pytest.fixture(scope="module")
def records(clone, index):
    vocab = pd.clone_types(str(clone))
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
        assert record["icons"] == ("mixed" if rel == CANNON else "images")             # Cannon: image types, badge runs
        assert set(record["stats"]) == {"hp", "def", "atk", "spd"}
        assert all(len(e) == (4 if e[1] == "group" else 3) and isinstance(e[2], bool) for e in record["tree"])
    assert _size(records[DOOMER]) < 8_000 and _size(records[CANNON]) < 8_000
    assert _size(records[LAZARUS]) < 20_000                                         # 234 layers, 93 with effects


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
    assert doomer["smart_objects"][:2] == ["images/Background (1)", "Stars/Warpstar"] and doomer["fx"] == [] and doomer["overlays"] == []
    assert doomer["effects"]["Creature Types"] == ["Stroke (off)"] and doomer["effects"]["Creature Types/Light"] == ["DropShadow", "OuterGlow"]
    cannon = records[CANNON]
    assert cannon["overlays"] == [] and cannon["fx"] == ["Ability FX: Blazing"] and cannon["template"]["deviations"] == ["icons as images"]
    assert cannon["effects"]["Ability FX: Blazing"] == ["OuterGlow"] and set(cannon["fonts"]["stoogedings-Regular"]["layers"]) == {"Text/Ability"}
    assert cannon["effects"]["Stars/stars"] == ["GradientOverlay", "BevelEmboss"]
    virus = records[VIRUS]
    hidden = next(t for t in virus["text"] if t["layer"].startswith("Alternate/"))
    assert hidden["visible"] is False and "\n" in hidden["text"] and hidden["color"] == "#01c102"
    assert virus["template"]["name_layer"] is None and "no Name layer" in virus["template"]["deviations"]
    assert virus["overlays"] == ["Names/3rd: Magician", "Names/2nd: Puppet", "Names/1st: Slime"]
    assert "overlay word layers" in virus["template"]["deviations"]
    lazarus = records[LAZARUS]
    assert lazarus["template"]["ability_layers"] == ["Laz/Laz/Ability", "Dead Laz/Dead Laz/Ability"]
    assert lazarus["template"]["deviations"] == ["no Name layer", "several visible Ability layers", "ability layer outside Text",
                                                 "stat bars drawn", "icons as images"]
    assert [e[:3] for e in lazarus["tree"]] == [["images", "group", True], ["Laz", "group", True], ["Dead Laz", "group", True]]


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
def test_save_load_prune_round_trip(tmp_path):
    path = str(tmp_path / "d.json")
    cards = {"b" * 40: {"path": "Items/B.psd", "name": "B", "x": "é"}, "a" * 40: {"path": "Items/A.psd", "name": "A"}}
    pd.save(cards, path)
    text = (tmp_path / "d.json").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == '{"version": 1, "cards": {' and lines[1].startswith('"' + "a" * 40 + '": {"path":"Items/A.psd"') and lines[2].endswith("}")
    assert lines[3] == "}}" and "é" in text and json.loads(text)["cards"] == cards
    assert pd.load(path) == cards and not (tmp_path / "d.json.tmp").exists()
    assert pd.prune(cards, {"a" * 40}) == 1 and set(cards) == {"a" * 40} and pd.prune(cards, {"a" * 40}) == 0
    pd.save({}, path)
    assert json.loads((tmp_path / "d.json").read_text(encoding="utf-8")) == {"version": 1, "cards": {}}


def test_load_tolerates_a_missing_or_odd_file(tmp_path, capsys):
    assert pd.load(str(tmp_path / "none.json")) == {}
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "list.json").write_text("[1, 2]", encoding="utf-8")
    assert pd.load(str(tmp_path / "bad.json")) == {} and pd.load(str(tmp_path / "list.json")) == {}
    out = capsys.readouterr().out
    assert "bad.json: unreadable" in out and "list.json: not a details file" in out


def test_save_failure_keeps_the_old_file(tmp_path, monkeypatch):
    path = str(tmp_path / "d.json")
    pd.save({"x" * 40: {"path": "p"}}, path)
    monkeypatch.setattr(pd, "fsync", lambda fd: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        pd.save({}, path)
    assert pd.load(path) == {"x" * 40: {"path": "p"}} and not (tmp_path / "d.json.tmp").exists()


# ========================
# The backfill CLI
# ========================
def test_cli_backfills_a_clone(clone, tmp_path, capsys):
    root = tmp_path / "clone"
    for rel in (DOOMER, CANNON):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(clone / rel, root / rel)
    for rel in ("MDW/Templates/Creature_Template.psd", "Auxiliary/Markers/M.psd", ".git/x.psd"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"")
    (root / "Types").mkdir()
    (root / "Types" / "Fire.png").write_bytes(b"")
    assert pd.card_psds(str(root)) == [(DOOMER, str(root / DOOMER)), (CANNON, str(root / CANNON))] and pd.clone_types(str(root)) == ["fire"]
    out = tmp_path / "out.json"
    assert pd.main(["--clone", str(root), "--out", str(out), "--fonts", str(clone / "MDW/Fonts")]) == 0
    text = capsys.readouterr().out
    assert text.startswith(f"2 cards into {out} in ") and "(0 errors); records " in text and "median" in text
    cards = pd.load(str(out))
    assert set(cards) == {blob_sha(clone / DOOMER), blob_sha(clone / CANNON)}
    doomer = cards[blob_sha(clone / DOOMER)]
    assert doomer["path"] == DOOMER and doomer["name"] == "Mini Doomer" and doomer["stats"]["hp"] == "template"
    assert {e["supplied"] for c in cards.values() for e in c["fonts"].values()} == {True}
    assert not (tmp_path / "out.json.tmp").exists()


def test_cli_reports_an_unreadable_psd(tmp_path, capsys):
    root = tmp_path / "clone"
    (root / "Items").mkdir(parents=True)
    (root / "Items" / "Bad.psd").write_bytes(b"not a psd")
    assert pd.main(["--clone", str(root), "--out", str(tmp_path / "o.json")]) == 0
    text = capsys.readouterr().out
    assert text.startswith("0 cards into") and "(1 errors); no records" in text and "  Items/Bad.psd: " in text
    assert pd.load(str(tmp_path / "o.json")) == {}


# ========================
# /push carries the file
# ========================
def test_push_commits_the_details_file_with_the_pickles(monkeypatch):
    from git import GitCommandError
    added = []

    class FakeGit:
        def add(self, path):
            added.append(path)
        def commit(self, *a):
            raise GitCommandError("commit", 1, b"nothing to commit")
        def pull(self, *a, **kw):
            pass
        def rev_list(self, *a):
            return "0\n"
    monkeypatch.setattr(ub, "_git", FakeGit)
    monkeypatch.setattr(ub, "isfile", lambda path: path == DETAILS_JSON)
    sent = []
    async def send(content):
        sent.append(content)
    asyncio.run(ub.git_push(SimpleNamespace(followup=SimpleNamespace(send=send))))
    assert added == ["aliases.pkl", "stats.pkl", "old_stats.pkl", DETAILS_JSON] and sent == ["Pickles are already up to date."]
    assert DETAILS_JSON == "psd_details.json"
    # a checkout without the file (before the backfill commit): the pickles still go, no "pathspec did not match"
    added.clear()
    monkeypatch.setattr(ub, "isfile", lambda path: False)
    asyncio.run(ub.git_push(SimpleNamespace(followup=SimpleNamespace(send=send))))
    assert added == ["aliases.pkl", "stats.pkl", "old_stats.pkl"]
