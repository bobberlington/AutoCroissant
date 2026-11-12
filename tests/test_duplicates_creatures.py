"""Duplicates (a dup_of PNG -> duplicate_of) and the multi-creature split, on the real cards."""

import commands.card_data as cd
import commands.psd_analyzer as pa
from commands.card_models import StatsDatabase
from golden_cards import parser


def _tree(paths: list) -> dict:
    """A tree of these paths, as traverse_remote reads it."""
    return {"tree": [{"path": p, "sha": "0" * 40} for p in paths]}


def _sync(tr: cd.CardBuild, paths: list) -> int:
    tr.tree = _tree(paths)
    return tr._sync_duplicates()


def _traverser(cards: dict) -> cd.CardBuild:
    db = StatsDatabase()
    db.stats = cards
    return cd.CardBuild(db)


# ========================
# Duplicates
# ========================
def test_duplicate_rules():
    cards = {n: pa.CardInfo(name=n, card_type="item", path=p) for n, p in (
        ("X", "Items/Heal/2 Stars/X.psd"), ("X2", "Items/Heal/2 Stars/X2.psd"), ("Z", "MDW/Z.psd"), ("Q", "Items/Q.psd"))}
    cards["Old2"] = pa.CardInfo(name="Old2", card_type="item", duplicate_of="Q")              # gone: the prune's
    tr = _traverser(cards)
    paths = ["Items/Heal/2 Stars/X.psd", "Items/Heal/2 Stars/X2.psd", "Items/Heal/2 Stars/X3.png",
             "Items/Heal/2 Stars/x2.png",                      # has its own PSD: not a duplicate
             "Items/Heal/Y3.png",                              # no Y.psd
             "MDW/Z.psd", "MDW/Z2.png", "Items/Q.psd"]
    assert _sync(tr, paths) == 0                  # numbered PNGs are variant pictures: no duplicates
    assert {n for n, c in tr.db.stats.items() if c.duplicate_of} == {"Old2"} and tr.db.seen_cards == []   # Old2: the prune's


def test_a_png_only_card_with_dup_of_is_a_duplicate():
    """Stubborn_Queen.png has no PSD; its Card Edits.json entry names Stubborn King: a copy of it under its own name
    (build_site's rule). Without the entry, or naming no card, it stays an image only."""
    from commands.card_edits import CardEdits
    king = pa.CardInfo(name="Stubborn King", card_type="creature", path="Creatures/Other/2 Stars/Stubborn_King.psd",
                       ability="Rule.")
    paths = ["Creatures/Other/2 Stars/Stubborn_King.psd", "Creatures/Other/2 Stars/Stubborn_King.png",
             "Creatures/Other/2 Stars/Stubborn_Queen.png", "Items/Support/2 Stars/Gust_of_Trees.png"]
    tr = _traverser({"Stubborn King": king})
    assert _sync(tr, paths) == 0
    tr.edits = CardEdits.parse('{"cards": {"Stubborn Queen": {"dup_of": "stubborn king", "aliases": ["queen"]}, '
                               '"Gust of Trees": {"dup_of": "Gust of Wind"}}}')
    assert _sync(tr, paths) == 1
    queen = tr.db.stats["Stubborn Queen"]
    assert (queen.duplicate_of, queen.ability, queen.path, queen.display_name) == ("Stubborn King", "Rule.", king.path, "Queen")
    assert "Gust of Trees" not in tr.db.stats and "Stubborn Queen" in tr.db.seen_cards
    tr.edits = CardEdits.parse('{"cards": {"Stubborn Queen": {"dup_of": "Stubborn King"}}}')
    _sync(tr, paths)
    assert tr.db.stats["Stubborn Queen"].display_name is None                  # its own name is its title


def test_unread_card_edits_keep_the_dup_of_cards(monkeypatch):
    """Card Edits.json in the tree but not read (a 5xx: lost): its dup_of entries can't be read, so
    Stubborn Queen (PNG and original still there) stays a card, never archived; gone PNG: archived as before."""
    from requests.exceptions import RequestException
    from commands.card_edits import CardEdits
    king = pa.CardInfo(name="Stubborn King", card_type="creature", path="Creatures/Other/2 Stars/Stubborn_King.psd",
                       ability="Rule.")
    paths = ["Creatures/Other/2 Stars/Stubborn_King.psd", "Creatures/Other/2 Stars/Stubborn_King.png",
             "Creatures/Other/2 Stars/Stubborn_Queen.png", "site/Card Edits.json"]
    tr = _traverser({"Stubborn King": king})
    tr.edits = CardEdits.parse('{"cards": {"Stubborn Queen": {"dup_of": "Stubborn King"}}}')
    tr.db.seen_cards.append("Stubborn King")
    _sync(tr, paths)
    assert "Stubborn Queen" in tr.db.stats

    def unreadable(path):
        raise RequestException("502")
    monkeypatch.setattr(tr, "_repo_text", unreadable)
    for tree, kept in ((paths, True), (paths[:2] + paths[3:], False)):
        tr.db.seen_cards.clear()
        tr.db.seen_cards.append("Stubborn King")
        tr.tree = _tree(tree)
        tr.edits = tr._edits_from_response()
        assert tr.edits is None and tr.edits_unread
        tr._sync_duplicates()
        gone = tr.db.prune_clean_cards(1.0)
        assert (gone == []) is kept and ("Stubborn Queen" in tr.db.stats) is kept


def test_variant_pictures_are_duplicates():
    """Every variant picture is a duplicate of its picture 1 (owner 10-25), or of the card its dup_of names (not a
    duplicate, another picture or itself), keeping its own data and version; Card Edits.json unread: one set stays.
    The bot's sync counts the marking as no card gone."""
    from copy import deepcopy
    from commands import card_sync
    from commands.card_edits import CardEdits

    def pic(name, n=None, of=None):
        card = pa.CardInfo(name=name, card_type="item", path="Items/Heal/2 Stars/Kit.psd", ability=f"{name}.", variant=n,
                           variant_of=of)
        card.version = {"from": float(n or 0)}
        return card
    cards = {"Kit": pic("Kit", 1), "Kit2": pic("Kit2", 2, "Kit"), "Kit3": pic("Kit3", 3, "Kit"), "Kit4": pic("Kit4", 4, "Kit"),
             "Other": pic("Other"), "Queen": pa.CardInfo(name="Queen", card_type="item", duplicate_of="Other")}
    before = deepcopy(cards)
    tr = _traverser(cards)
    tr.edits = CardEdits.parse('{"cards": {"Kit2": {"dup_of": "other"}, "Kit3": {"dup_of": "Queen"}, "Kit4": {"dup_of": "Kit3"}}}')
    assert tr._variant_duplicates() == 3 and tr._variant_duplicates() == 0
    tr.duplicate_versions()
    assert {n: c.duplicate_of for n, c in cards.items()} == {
        "Kit": None, "Kit2": "Other", "Kit3": "Kit", "Kit4": "Kit", "Other": None, "Queen": "Other"}
    assert [(cards[n].ability, cards[n].version) for n in ("Kit2", "Kit3")] == [("Kit2.", {"from": 2.0}), ("Kit3.", {"from": 3.0})]
    assert card_sync.changes(before, cards) == {"changed": 0, "new": 0, "archived": 0}
    tr.edits, tr.edits_unread = None, True
    assert tr._variant_duplicates() == 0 and cards["Kit2"].duplicate_of == "Other"     # unread: kept
    tr.edits_unread = False
    assert tr._variant_duplicates() == 1 and cards["Kit2"].duplicate_of == "Kit"       # no dup_of: picture 1


KIT_REV = "e9b00260fd030fb30e8e968605cf1d1a18ec0d87"     # the card repo before All_Purpose_Medical_Kit was renamed


def test_a_variant_picture_printing_its_files_name():
    """Picture 2 of All-Purpose_Medical_Kit.psd (renamed 11-11) prints the file's name: as the card
    All-Purpose Medical Kit2 that is its title; picture 1's stays None (the name)."""
    from io import BytesIO
    from psd_tools import PSDImage
    from conftest import CLONE, at_rev
    psd = PSDImage.open(BytesIO(at_rev(CLONE, KIT_REV, "Items/Heal/2 Stars/All_Purpose_Medical_Kit.psd")))
    rel, p = "Items/Heal/2 Stars/All-Purpose_Medical_Kit.psd", parser()
    assert [p.parse(psd, rel, variant=n).display_name for n in (1, 2)] == [None, "All-Purpose Medical Kit"]


def test_duplicate_name_order():
    """A duplicate's own Card Edits.json aliases name it (the longest); with none it is titled by its own name (None)."""
    assert cd.CardBuild._duplicate_name(["late_supper", "ls"]) == "Late Supper"
    assert cd.CardBuild._duplicate_name([]) is None


# ========================
# Multi-creature split
# ========================
def test_auxiliary_creature_without_bars(clone):
    """No stat bar read on an Auxiliary/Creatures card: one NO STATS FOUND note, worded for a creature."""
    from psd_tools import PSDImage
    import commands.psd_parser as pp
    rel = "Auxiliary/Creatures/Morpho_Knight.psd"
    psd = PSDImage.open(clone / rel)
    psd.remove(next(layer for layer in psd if layer.name == "Stats Bars"))
    card = parser().parse(psd, rel)
    note = f"{pp.NO_STATS}: no HP, DEF, ATK or SPD bars read on this creature"
    assert card.card_type == "creatures" and card.problems == [note] and pp.CardValidator.validate(card) == [note]


def _restacked(lower: tuple, upper: tuple, down: int = 0):
    """Tainted Lost's 11-11 layout: its old bars (a lit bar as wide as the value, 'Stats Bars'/'Bars'/'SPD'/'5') under a
    new 'Stats Bars' of whole bars on the same rows (down: px lower)."""
    from PIL import Image
    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer
    psd = PSDImage.new("RGB", (1200, 2400))
    for values, whole, shift in ((lower, False, 0), (upper, True, down)):
        bars = psd.create_group(name="Stats Bars").create_group(name="Bars")
        for row, (stat, value) in enumerate(zip(("HP", "DEF", "ATK", "SPD"), values)):
            width = 819 if whole else 81 * (value or 10)
            bar = PixelLayer.frompil(Image.new("RGB", (width, 72), (0, 160, 255)), psd, str(value), top=1207 + 126 * row + shift, left=363)
            bars.create_group([bar], name=stat)
    return psd


def test_bars_drawn_over_old_ones_hide_them():
    """Bars of another block drawn over a row hide it (Tainted Lost, Tainted Forgotten 11-11): one creature, the top bars'
    values; the same bars elsewhere are a second creature's (Tainted Lazarus). A drawn layer's box is inside its mask."""
    from types import SimpleNamespace
    from commands.psd_regions import drawn_box
    rel = "Creatures/Isaac/2 Stars/Tainted_Lost.psd"
    def read(psd):
        card = parser().parse(psd, rel)
        return tuple(card.stats.to_dict().values()), [tuple(c.to_dict().values()) for c in card.creatures or []]
    assert read(_restacked((1, 0, 5, 5), (1, 0, 5, 5))) == ((1, 0, 5, 5), [])
    assert read(_restacked((1, 0, 5, 5), (2, 3, 4, 6))) == ((2, 3, 4, 6), [])
    assert read(_restacked((1, 0, 5, 5), (2, 3, 4, 6), down=600))[1] == [(1, 0, 5, 5), (2, 3, 4, 6)]
    layer = lambda bbox, mask, outside=0: SimpleNamespace(bbox=bbox, has_mask=lambda: bool(mask), mask=SimpleNamespace(
        bbox=mask, disabled=False, background_color=outside))
    assert drawn_box(layer((363, 1458, 1170, 1530), (0, 1445, 766, 1542))) == (363, 1458, 766, 1530)    # Tainted Lost's ATK 5
    assert drawn_box(layer((0, 0, 0, 0), (0, 1445, 766, 1542))) == (0, 1445, 766, 1542)                 # an exposure
    assert drawn_box(layer((363, 1332, 1170, 1404), (0, 1320, 300, 1417))) is None                       # its DEF 0
    assert drawn_box(layer((363, 1332, 1170, 1404), (0, 1320, 300, 1417), 255)) == (363, 1332, 1170, 1404)   # shows outside
