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
def repo_blobs(clone):
    """The clone's committed files: path -> blob sha."""
    out = subprocess.run(["git", "-C", str(clone), "ls-files", "-s", "-z"], capture_output=True, encoding="utf-8",
                         check=True)
    return {path: meta.split()[1] for meta, path in (e.split("\t", 1) for e in out.stdout.split("\0") if e)}


def _drift(stats: dict, blobs: dict) -> list:
    """PSDs the clone and stats.pkl disagree on: added, removed or edited since the last /update_stats."""
    stored = {c.path: c.blob_sha for c in stats.values() if not c.duplicate_of}
    psds = {p: b for p, b in blobs.items()
            if p.lower().endswith(".psd") and not any(f in pa.EXCLUDE_FOLDERS for f in p.split("/"))}
    return sorted(p for p in stored.keys() | psds.keys() if stored.get(p) != psds.get(p))


# ========================
# Duplicates
# ========================
def test_real_duplicates(real_stats, repo_blobs, no_aliases):
    drift = _drift(real_stats.stats, repo_blobs)
    if drift:
        pytest.skip(f"clone and stats.pkl out of step ({len(drift)} PSDs, e.g. {drift[0]})")
    repo_paths = list(repo_blobs)
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
    """Hidden titles name the duplicates in number order; an /alias to the PNG wins (the build takes the aliases)."""
    src = pa.CardInfo(name="A Snack", card_type="item", alt_names=["Breakfast", "Supper"])
    assert pa.RepositoryTraverser._duplicate_name("a_snack3.png", src, 3, [2, 3]) == "Supper"
    assert pa.RepositoryTraverser._duplicate_name("a_snack3.png", src, 3, [2, 3, 4]) == "A Snack"   # count mismatch
    aliases = {"late_supper": "a_snack3.png", "ls": "a_snack3.png"}
    assert pa.RepositoryTraverser._duplicate_name("a_snack3.png", src, 3, [2, 3], aliases) == "Late Supper"
    monkeypatch.setattr(pa.card_repo, "git_file_alias", aliases)                 # /update_stats passes the bot's
    tr = _traverser({"A Snack": src})
    assert tr.aliases == {}
    import commands.repo_traverser as rt
    seen = []
    monkeypatch.setattr(rt, "update", lambda build, *a: seen.append(build.aliases) or [])
    monkeypatch.setattr(pa.stats_db, "load", lambda: None)
    monkeypatch.setattr(rt, "queue_command", lambda *a, **kw: None)
    pa.update_stats(None, verbose=False)
    assert seen == [aliases] and seen[0] is pa.card_repo.git_file_alias


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
            stale.append(name)                                    # PSD edited since the last /update_stats
            continue
        card = p.parse(str(psd), stored.path)
        from_psd = (getattr(stored, "parsed_values", None) or {}).get("creatures")   # the PSD's values under a website edit
        assert [c.to_dict() for c in card.creatures] == (from_psd or [c.to_dict() for c in stored.creatures]), name
        assert card.stats.to_dict() == card.creatures[0].to_dict()
        assert any(x.startswith("SEVERAL CREATURES") for x in card.problems)
        assert pa.CardValidator.validate(card) == card.problems   # creature 1 stays within 10: no STATS TOO HIGH
    if stale:
        pytest.skip(f"clone and stats.pkl out of step: {', '.join(stale)}")
