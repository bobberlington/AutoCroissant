"""Archived cards: a deleted PSD's record stays in old_stats.pkl, tagged CardInfo.archived (StatsDatabase.archived(),
restore(), prune_clean_cards); the end-to-end runs are in test_update_stats.py."""
import pickle
import shutil
from collections import defaultdict
from copy import deepcopy

import pytest

import commands.card_models as cm
from conftest import REPO

AT = 1_800_000_000.0


def _card(name, blob="a" * 40, ability="Does a thing.", **kw):
    return cm.CardInfo(name=name, card_type="item", path=f"Items/{name.replace(' ', '_')}.psd", ability=ability,
                       blob_sha=blob, timestamp=1_700_000_000.0, **kw)


def _db(stats=(), old=(), seen=None):
    db = cm.StatsDatabase()
    db.stats = {c.name: c for c in stats}
    for card in old:
        db.old_stats[card.name].append(card)
    db.seen_cards = list(db.stats) if seen is None else list(seen)
    return db


# ========================
# The field
# ========================
def test_card_pickled_before_the_field_loads_unarchived():
    card = _card("Old")
    del card.__dict__["archived"]                               # what an older pickle holds
    old = pickle.loads(pickle.dumps(card))
    assert old.archived is None and getattr(old, "archived", None) is None
    assert "archived" not in old.to_dict() and "archived" not in _card("New").to_dict()


def test_tag_is_plain_data():
    db = _db([_card("Gone"), _card("Kept")], seen=["Kept"])
    db.prune_clean_cards(AT)
    tag = db.old_stats["Gone"][0].archived
    assert tag == {"at": AT, "path": "Items/Gone.psd", "blob": "a" * 40, "commit": None}
    assert all(type(v) in (float, str, type(None)) for v in tag.values())
    assert "archived" not in db.old_stats["Gone"][0].to_dict()


# ========================
# prune_clean_cards, archived(), restore()
# ========================
def test_prune_archives_behind_the_history():
    v1 = _card("Gone", blob="1" * 40, ability="First.")
    db = _db([_card("Gone", blob="2" * 40), _card("Kept")], old=[v1], seen=["Kept"])
    assert db.prune_clean_cards(AT) == (["Gone"], [])
    versions = db.old_stats["Gone"]
    assert "Gone" not in db.stats and versions[0] is v1 and v1.archived is None
    assert versions[1].archived["blob"] == "2" * 40 and set(db.archived()) == {"Gone"}


def test_duplicates_are_not_archived():
    dup = _card("A Snack2", duplicate_of="A Snack")
    db = _db([_card("A Snack"), dup], seen=["A Snack"])
    assert db.prune_clean_cards(AT) == ([], []) and "A Snack2" in db.stats and db.archived() == {}


def test_card_without_a_blob_is_tagged_with_none():
    db = _db([_card("Legacy", blob=None)], seen=["Other"])
    db.prune_clean_cards(AT)
    assert db.old_stats["Legacy"][0].archived["blob"] is None


def test_restore_returns_the_record_and_keeps_the_history():
    v1 = _card("Gone", blob="1" * 40)
    db = _db([_card("Gone", blob="2" * 40)], old=[v1], seen=[])
    db.seen_cards = ["Other"]
    db.prune_clean_cards(AT)
    assert db.restore("Gone") and db.stats["Gone"].blob_sha == "2" * 40 and db.stats["Gone"].archived is None
    assert db.old_stats["Gone"] == [v1] and db.archived() == {}
    assert not db.restore("Gone")                               # listed again: nothing to restore


def test_restore_drops_an_emptied_history():
    db = _db([_card("Gone")], seen=["Other"])
    db.prune_clean_cards(AT)
    assert db.restore("Gone") and "Gone" not in db.old_stats


def test_untagged_history_is_never_restored():
    """A card reset by /update_metadata del_entry (its PSD still there) has untagged history: no restore."""
    db = _db([], old=[_card("Reset")])
    assert not db.restore("Reset") and "Reset" not in db.stats and len(db.old_stats["Reset"]) == 1
    assert not db.restore("Never there") and "Never there" not in db.old_stats


def test_archived_skips_listed_names_and_untagged_lasts():
    tagged = _card("Live", archived={"at": AT, "path": "Items/Live.psd", "blob": None, "commit": None})
    db = _db([_card("Live")], old=[tagged, _card("Untagged")])
    db.stats.pop("Untagged", None)
    assert db.archived() == {}                                  # Live is listed; Untagged's last has no tag


def test_backfill_tags_only_unlisted_unseen_untagged_names():
    done = _card("Done", archived={"at": 1.0, "path": "Items/Done.psd", "blob": None, "commit": None})
    db = _db([_card("Live")], old=[_card("Live", blob="0" * 40), _card("Gone"), done, _card("Dup", duplicate_of="X")])
    assert db.prune_clean_cards(AT) == ([], ["Gone"])
    assert db.old_stats["Gone"][0].archived == {"at": AT, "path": "Items/Gone.psd", "blob": "a" * 40, "commit": None,
                                                "backfill": True}
    assert db.old_stats["Live"][0].archived is None and db.old_stats["Done"][0].archived["at"] == 1.0
    assert db.old_stats["Dup"][0].archived is None
    assert db.prune_clean_cards(AT + 1) == ([], [])             # once


def test_a_copy_left_by_a_kill_is_replaced():
    card = _card("Gone")
    left = deepcopy(card)
    left.archived = {"at": 1.0, "path": card.path, "blob": card.blob_sha, "commit": None}
    db = _db([card], old=[left], seen=["Other"])
    db.prune_clean_cards(AT)
    assert len(db.old_stats["Gone"]) == 1 and db.old_stats["Gone"][0].archived["at"] == AT


def test_an_untagged_equal_version_stays_history():
    """Only a tagged copy is replaced: an old untagged version equal to the card (a 2025 balloon) stays."""
    card = _card("Gone")
    db = _db([card], old=[deepcopy(card)], seen=["Other"])
    db.prune_clean_cards(AT)
    assert [c.archived is None for c in db.old_stats["Gone"]] == [True, False]


def test_a_new_card_under_an_archived_name_keeps_the_archive():
    """A record made by /update_metadata for a name whose PSD is gone: archiving it again adds, never replaces."""
    real = _card("Gone", ability="The real card.")
    real.archived = {"at": 1.0, "path": real.path, "blob": real.blob_sha, "commit": None}
    stub = cm.CardInfo(name="Gone", card_type="unknown", path=real.path)
    db = _db([stub], old=[real], seen=["Other"])
    db.prune_clean_cards(AT)
    assert [c.ability for c in db.old_stats["Gone"]] == ["The real card.", None]


def test_save_writes_old_stats_first(monkeypatch):
    order = []
    monkeypatch.setattr(cm, "save_pickle", lambda obj, path, lock: order.append(path))
    cm.StatsDatabase().save()
    assert order == [cm.OLD_STATS_PKL, cm.STATS_PKL]


# ========================
# The live pickles (a copy)
# ========================
@pytest.fixture
def live_copy(tmp_path, monkeypatch):
    """A StatsDatabase over copies of the live stats.pkl and old_stats.pkl (saves land in tmp_path)."""
    for name in (cm.STATS_PKL, cm.OLD_STATS_PKL):
        if not (REPO / name).exists():
            pytest.skip(f"no {name}")
        shutil.copy(REPO / name, tmp_path / name)
    db = cm.StatsDatabase()
    db.load()
    return db


def test_backfill_on_the_live_pickles(live_copy):
    """Every name in old_stats.pkl but not stats.pkl (deleted before archiving) is tagged once; nothing listed moves,
    no version is added or lost, and the earlier versions are untouched."""
    db = live_copy
    db.seen_cards = list(db.stats)                              # as if every listed card is still in the tree
    before = {name: [repr(c) for c in versions] for name, versions in db.old_stats.items()}
    listed = dict(db.stats)
    expected = sorted(n for n, v in db.old_stats.items() if v and n not in db.stats
                      and not getattr(v[len(v) - 1], "archived", None) and not getattr(v[len(v) - 1], "duplicate_of", None))
    archived, backfilled = db.prune_clean_cards(AT)
    assert archived == [] and backfilled == expected and db.stats == listed
    assert set(db.archived()) == set(expected) | {n for n, v in db.old_stats.items()
                                                   if n not in db.stats and v and v[len(v) - 1].archived}
    for name, versions in db.old_stats.items():
        assert len(versions) == len(before[name])
        assert [repr(c) for c in versions[:len(versions) - 1]] == before[name][:len(versions) - 1]
    for name in backfilled:
        tag = db.old_stats[name][len(db.old_stats[name]) - 1].archived
        assert tag["backfill"] is True and tag["at"] == AT and tag["path"]
    db.save()
    again = cm.StatsDatabase()
    again.load()
    again.seen_cards = list(again.stats)
    assert set(again.archived()) == set(db.archived()) and again.prune_clean_cards(AT + 1) == ([], [])
    assert {n: v.archived for n, v in again.archived().items()} == {n: v.archived for n, v in db.archived().items()}
