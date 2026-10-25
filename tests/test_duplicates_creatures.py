"""Duplicates (<Name>N.png next to <Name>.psd -> duplicate_of) and the multi-creature split, on the real cards."""
import subprocess
from copy import deepcopy

import pytest

import commands.card_data as cd
import commands.psd_analyzer as pa
from conftest import blob_sha, no_lazy
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


@pytest.fixture(scope="module")
def repo_blobs(clone):
    """The clone's committed files: path -> blob sha."""
    out = subprocess.run(["git", "-C", str(clone), "ls-files", "-s", "-z"], capture_output=True, encoding="utf-8",
                         check=True)
    return {path: meta.split()[1] for meta, path in (e.split("\t", 1) for e in out.stdout.split("\0") if e)}


def _drift(stats: dict, blobs: dict) -> list:
    """PSDs the clone and the card data disagree on: added, removed or edited since the last card sync."""
    stored = {c.path: c.blob_sha for c in stats.values() if not c.duplicate_of}
    psds = {p: b for p, b in blobs.items()
            if p.lower().endswith(".psd") and not any(f in cd.EXCLUDE_FOLDERS for f in p.split("/"))}
    return sorted(p for p in stored.keys() | psds.keys() if stored.get(p) != psds.get(p))


# ========================
# Duplicates
# ========================
def _committed_edits(clone):
    """The clone's committed Card Edits.json (dup_of duplicates such as Stubborn_Queen.png need it), or None."""
    from commands.card_edits import CardEdits
    out = subprocess.run(["git", "-C", str(clone), "show", f"HEAD:{pa.EDITS_PATH}"], capture_output=True, encoding="utf-8",
                         env=no_lazy())
    return CardEdits.parse(out.stdout) if out.returncode == 0 else None


def test_real_duplicates(real_stats, repo_blobs, clone):
    drift = _drift(real_stats.stats, repo_blobs)
    if drift:
        pytest.skip(f"clone and card data out of step ({len(drift)} PSDs, e.g. {drift[0]})")
    repo_paths = list(repo_blobs)
    expected = {n: (c.duplicate_of, c.display_name) for n, c in real_stats.stats.items() if c.duplicate_of}
    assert len(expected) >= 5, "the card data should hold the A Snack / Watermelon / ... duplicates"
    tr = _traverser({n: deepcopy(c) for n, c in real_stats.stats.items() if not c.duplicate_of})
    tr.edits = _committed_edits(clone)
    assert tr._sync_duplicates(Tree(repo_paths)) == len(expected)
    got = {n: (c.duplicate_of, c.display_name) for n, c in tr.db.stats.items() if c.duplicate_of}
    assert got == expected
    for name in got:
        dup, src = tr.db.stats[name], tr.db.stats[got[name][0]]
        assert (dup.name, dup.path, dup.ability, dup.problems, dup.alt_names) == (name, src.path, src.ability, [], None)
        assert name in tr.db.seen_cards
    assert tr._sync_duplicates(Tree(repo_paths)) == 0                      # second pass: nothing new
    gone = [p for p in repo_paths if not p.endswith("A_Snack3.png")]
    tr.db.seen_cards.clear()
    if tr.edits is not None:                                    # A Snack2 keeps its own alias ("Breakfast")
        assert tr._sync_duplicates(Tree(gone)) == 0 and "A Snack3" not in tr.db.seen_cards     # the prune archives it
        assert tr.db.stats["A Snack2"].display_name == expected["A Snack2"][1]
    else:
        assert tr._sync_duplicates(Tree(gone)) == 1 and "A Snack3" not in tr.db.seen_cards
        assert tr.db.stats["A Snack2"].display_name == "A Snack"  # 2 hidden titles, 1 PNG: order unknown


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
    assert tr._sync_duplicates(Tree(paths)) == 1                  # X3 added; Old2 not seen: archived by the prune
    assert {n for n, c in tr.db.stats.items() if c.duplicate_of} == {"X3", "Old2"} and tr.db.seen_cards == ["X3"]
    assert tr.db.stats["X3"].duplicate_of == "X" and tr.db.stats["X3"].display_name == "X"


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


@pytest.mark.parametrize("failure", ["5xx", "bad json"])
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


def test_duplicate_name_order():
    """A duplicate's own Card Edits.json aliases name it (the longest), else its original's name: hidden title layers
    name nothing (owner 2026-10-25); a dup_of card with none is titled by its own name (None)."""
    src = pa.CardInfo(name="A Snack", card_type="item", alt_names=["Breakfast", "Supper"])     # older data
    assert cd.CardBuild._duplicate_name(src) == "A Snack"
    assert cd.CardBuild._duplicate_name(src, ["late_supper", "ls"]) == "Late Supper"
    assert cd.CardBuild._duplicate_name(src, [], own_name=True) is None


# ========================
# Multi-creature split
# ========================
def test_real_creature_split(real_stats, clone):
    multi = {n: c for n, c in real_stats.stats.items() if getattr(c, "creatures", None)}
    assert len(multi) >= 3, "Tainted Lazarus, Pix and Twin Emperors split into creatures"
    p, stale = parser(), []
    for name, stored in sorted(multi.items()):
        psd = clone / stored.path
        if not psd.exists() or not stored.blob_sha or blob_sha(psd) != stored.blob_sha:
            stale.append(name)                                    # PSD edited since the last card sync
            continue
        card = p.parse(str(psd), stored.path)
        from_psd = (getattr(stored, "parsed_values", None) or {}).get("creatures")   # the PSD's values under a website edit
        assert [c.to_dict() for c in card.creatures] == (from_psd or [c.to_dict() for c in stored.creatures]), name
        assert card.stats.to_dict() == card.creatures[0].to_dict()
        assert any(x.startswith("SEVERAL CREATURES") for x in card.problems)
        assert pa.CardValidator.validate(card) == card.problems   # creature 1 stays within 10: no STATS TOO HIGH
    if stale:
        pytest.skip(f"clone and card data out of step: {', '.join(stale)}")


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
