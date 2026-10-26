"""Duplicates (a dup_of PNG -> duplicate_of) and the multi-creature split, on the real cards."""

import pytest

import commands.card_data as cd
import commands.psd_analyzer as pa
from golden_cards import parser


class Tree:
    """Stands in for the tree response: _sync_duplicates only calls .json()."""
    def __init__(self, paths):
        self.paths = list(paths)

    def json(self):
        return {"tree": [{"path": p, "sha": "0" * 40} for p in self.paths]}


def _traverser(cards: dict) -> cd.CardBuild:
    db = pa.StatsDatabase()
    db.stats = cards
    return cd.CardBuild(db)


# ========================
# Duplicates
# ========================
def test_duplicate_rules():
    cards = {n: pa.CardInfo(name=n, card_type="item", path=p) for n, p in (
        ("X", "Items/Heal/2 Stars/X.psd"), ("X2", "Items/Heal/2 Stars/X2.psd"), ("R", "Auxiliary/Rulebook/R.psd"),
        ("Z", "MDW/Z.psd"), ("Q", "Items/Q.psd"))}
    cards["Old2"] = pa.CardInfo(name="Old2", card_type="item", duplicate_of="Q")              # gone: the prune's
    tr = _traverser(cards)
    paths = ["Items/Heal/2 Stars/X.psd", "Items/Heal/2 Stars/X2.psd", "Items/Heal/2 Stars/X3.png",
             "Items/Heal/2 Stars/x2.png",                      # has its own PSD: not a duplicate
             "Items/Heal/Y3.png",                              # no Y.psd
             "Auxiliary/Rulebook/R.psd", "Auxiliary/Rulebook/R2.png", "MDW/Z.psd", "MDW/Z2.png", "Items/Q.psd"]
    assert tr._sync_duplicates(Tree(paths)) == 0                  # numbered PNGs are variant pictures: no duplicates
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
    assert tr._sync_duplicates(Tree(paths)) == 0
    tr.edits = CardEdits.parse('{"cards": {"Stubborn Queen": {"dup_of": "stubborn king", "aliases": ["queen"]}, '
                               '"Gust of Trees": {"dup_of": "Gust of Wind"}}}')
    assert tr._sync_duplicates(Tree(paths)) == 1
    queen = tr.db.stats["Stubborn Queen"]
    assert (queen.duplicate_of, queen.ability, queen.path, queen.display_name) == ("Stubborn King", "Rule.", king.path, "Queen")
    assert "Gust of Trees" not in tr.db.stats and "Stubborn Queen" in tr.db.seen_cards
    tr.edits = CardEdits.parse('{"cards": {"Stubborn Queen": {"dup_of": "Stubborn King"}}}')
    tr._sync_duplicates(Tree(paths))
    assert tr.db.stats["Stubborn Queen"].display_name is None                  # its own name is its title


@pytest.mark.parametrize("failure", ["5xx"])
def test_unread_card_edits_keep_the_dup_of_cards(monkeypatch, failure):
    """Card Edits.json in the tree but not read (a 5xx: lost; malformed JSON): its dup_of entries can't be read, so
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
    tr._sync_duplicates(Tree(paths))
    assert "Stubborn Queen" in tr.db.stats

    def unreadable(path):
        if failure == "5xx":
            raise RequestException("502")
        return "{not json"
    monkeypatch.setattr(tr, "_repo_text", unreadable)
    for tree, kept in ((paths, True), (paths[:2] + paths[3:], False)):
        tr.db.seen_cards.clear()
        tr.db.seen_cards.append("Stubborn King")
        tr.edits = tr._edits_from_response(Tree(tree), "o/r")
        assert tr.edits is None and tr.edits_unread
        tr._sync_duplicates(Tree(tree))
        gone, _ = tr.db.prune_clean_cards(1.0)
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


def test_duplicate_name_order():
    """A duplicate's own Card Edits.json aliases name it (the longest); with none it is titled by its own name (None)."""
    assert cd.CardBuild._duplicate_name(["late_supper", "ls"]) == "Late Supper"
    assert cd.CardBuild._duplicate_name([]) is None


# ========================
# Multi-creature split
# ========================
AUX_CREATURES = {"Azamoth_the_Possessed_Vessel": 5, "Igor_Roballtowski": 2, "Mettaton_EX": 4, "Mettaton_NEO": 4,
                 "Militron_(No_Armor)": 4, "Morpho_Knight": 5, "Shadow_Queen": 5}


def test_auxiliary_creatures_count_their_printed_stars(clone):
    """The 7 Auxiliary/Creatures cards print 2-5 stars; their folder says none, so the PSD's are counted."""
    import commands.psd_parser as pp
    assert pp.stars_from_psd("Auxiliary/Creatures/Shadow_Queen.psd") and not pp.stars_from_psd("Creatures/Other/3 Stars/X.psd")
    found = {}
    for stem, stars in AUX_CREATURES.items():
        rel = f"Auxiliary/Creatures/{stem}.psd"
        if not (clone / rel).exists():
            pytest.skip(f"{rel} not in the clone")
        found[stem] = parser().parse(str(clone / rel), rel).stars
    assert found == AUX_CREATURES
