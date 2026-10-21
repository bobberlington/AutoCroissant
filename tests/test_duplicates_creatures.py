"""Duplicates (<Name>N.png next to <Name>.psd -> duplicate_of) and the multi-creature split, on the real cards."""
import subprocess
from copy import deepcopy

import pytest

import commands.psd_analyzer as pa
from conftest import blob_sha
from golden_cards import parser


class Tree:
    """Stands in for the tree response: _sync_duplicates only calls .json()."""
    def __init__(self, paths):
        self.paths = list(paths)

    def json(self):
        return {"tree": [{"path": p, "sha": "0" * 40} for p in self.paths]}


@pytest.fixture
def no_aliases(monkeypatch):
    monkeypatch.setattr(pa.card_repo, "git_file_alias", {})


def _traverser(cards: dict) -> pa.RepositoryTraverser:
    db = pa.StatsDatabase()
    db.stats = cards
    return pa.RepositoryTraverser(db)


@pytest.fixture(scope="module")
def repo_paths(clone):
    out = subprocess.run(["git", "-C", str(clone), "ls-files"], capture_output=True, encoding="utf-8", check=True)
    return out.stdout.splitlines()


# ========================
# Duplicates
# ========================
def test_real_duplicates(real_stats, repo_paths, no_aliases):
    expected = {n: (c.duplicate_of, c.display_name) for n, c in real_stats.stats.items() if c.duplicate_of}
    assert len(expected) >= 5, "the pickle should hold the A Snack / Watermelon / ... duplicates"
    tr = _traverser({n: deepcopy(c) for n, c in real_stats.stats.items() if not c.duplicate_of})
    assert tr._sync_duplicates(Tree(repo_paths)) == len(expected)
    got = {n: (c.duplicate_of, c.display_name) for n, c in tr.db.stats.items() if c.duplicate_of}
    assert got == expected
    for name in got:
        dup, src = tr.db.stats[name], tr.db.stats[got[name][0]]
        assert (dup.name, dup.path, dup.ability, dup.problems, dup.alt_names) == (name, src.path, src.ability, [], None)
        assert name in tr.db.seen_cards
    assert tr._sync_duplicates(Tree(repo_paths)) == 0                      # second pass: nothing new
    gone = [p for p in repo_paths if not p.endswith("A_Snack3.png")]
    assert tr._sync_duplicates(Tree(gone)) == 2 and "A Snack3" not in tr.db.stats
    assert tr.db.stats["A Snack2"].display_name == "A Snack"      # 2 hidden titles, 1 PNG: order unknown


def test_duplicate_rules(no_aliases):
    cards = {n: pa.CardInfo(name=n, card_type="item", path=p) for n, p in (
        ("X", "Items/Heal/2 Stars/X.psd"), ("X2", "Items/Heal/2 Stars/X2.psd"), ("R", "Auxiliary/Rulebook/R.psd"),
        ("Z", "MDW/Z.psd"), ("Q", "Items/Q.psd"))}
    cards["Old2"] = pa.CardInfo(name="Old2", card_type="item", duplicate_of="Q")              # stale: removed
    tr = _traverser(cards)
    paths = ["Items/Heal/2 Stars/X.psd", "Items/Heal/2 Stars/X2.psd", "Items/Heal/2 Stars/X3.png",
             "Items/Heal/2 Stars/x2.png",                      # has its own PSD: not a duplicate
             "Items/Heal/Y3.png",                              # no Y.psd
             "Auxiliary/Rulebook/R.psd", "Auxiliary/Rulebook/R2.png", "MDW/Z.psd", "MDW/Z2.png", "Items/Q.psd"]
    assert tr._sync_duplicates(Tree(paths)) == 2                  # X3 added, Old2 removed
    assert {n for n, c in tr.db.stats.items() if c.duplicate_of} == {"X3"}
    assert tr.db.stats["X3"].duplicate_of == "X" and tr.db.stats["X3"].display_name == "X"


def test_duplicate_name_order(monkeypatch):
    """Hidden titles name the duplicates in number order; an /alias to the PNG wins."""
    src = pa.CardInfo(name="A Snack", card_type="item", alt_names=["Breakfast", "Supper"])
    monkeypatch.setattr(pa.card_repo, "git_file_alias", {})
    assert pa.RepositoryTraverser._duplicate_name("a_snack3.png", src, 3, [2, 3]) == "Supper"
    assert pa.RepositoryTraverser._duplicate_name("a_snack3.png", src, 3, [2, 3, 4]) == "A Snack"   # count mismatch
    monkeypatch.setattr(pa.card_repo, "git_file_alias", {"late_supper": "a_snack3.png", "ls": "a_snack3.png"})
    assert pa.RepositoryTraverser._duplicate_name("a_snack3.png", src, 3, [2, 3]) == "Late Supper"


# ========================
# Multi-creature split
# ========================
def test_real_creature_split(real_stats, clone):
    multi = {n: c for n, c in real_stats.stats.items()
             if getattr(c, "creatures", None) and not getattr(c, "parsed_values", None)}
    assert len(multi) >= 3, "Tainted Lazarus, Pix and Twin Emperors split into creatures"
    p, checked = parser(), 0
    for name, stored in sorted(multi.items()):
        psd = clone / stored.path
        if not psd.exists() or (stored.blob_sha and blob_sha(psd) != stored.blob_sha):
            continue                                              # PSD edited since the last /update_stats
        card = p.parse(str(psd), stored.path)
        assert [c.to_dict() for c in card.creatures] == [c.to_dict() for c in stored.creatures], name
        assert card.stats.to_dict() == card.creatures[0].to_dict()
        assert any(x.startswith("SEVERAL CREATURES") for x in card.problems)
        assert pa.CardValidator.validate(card) == card.problems   # their high stats are excluded
        checked += 1
    assert checked >= 3
