"""Archived cards: a deleted PSD's record stays in old_stats, tagged CardInfo.archived (StatsDatabase.archived(),
restore(), prune_clean_cards); the end-to-end runs are in test_card_data.py."""
from copy import deepcopy

import commands.card_models as cm

AT = 1_800_000_000.0


def _card(name, blob="a" * 40, ability="Does a thing.", **kw):
    return cm.CardInfo(name=name, card_type="item", path=f"Items/{name.replace(' ', '_')}.psd", ability=ability,
                       blob_sha=blob, timestamp=1_700_000_000.0, **kw)


def _restore(db, name):
    """restore() with the PSD back at its archived path."""
    return db.restore(name, f"Items/{name}.psd", None)


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


def test_duplicates_are_archived_at_their_png():
    """An alt-art duplicate not seen (its PNG or original left) is a card under the archive rule: tagged at its PNG."""
    dup = _card("A Snack2", duplicate_of="A Snack")
    db = _db([_card("A Snack"), dup], seen=["A Snack"])
    assert db.prune_clean_cards(AT) == (["A Snack2"], []) and "A Snack2" not in db.stats
    record = db.old_stats["A Snack2"][0]
    assert record.duplicate_of == "A Snack" and record.archived == {"at": AT, "path": "Items/A_Snack2.png", "blob": None,
                                                                    "commit": None}
    assert set(db.archived()) == {"A Snack2"}


def test_restore_returns_the_record_and_keeps_the_history():
    v1 = _card("Gone", blob="1" * 40)
    db = _db([_card("Gone", blob="2" * 40)], old=[v1], seen=[])
    db.seen_cards = ["Other"]
    db.prune_clean_cards(AT)
    assert _restore(db, "Gone") and db.stats["Gone"].blob_sha == "2" * 40 and db.stats["Gone"].archived is None
    assert db.old_stats["Gone"][1].archived and db.archived() == {}   # the tagged record stays until drop_stale_archives
    assert db.stale_archives() == ["Gone"] and not _restore(db, "Gone")  # listed again: nothing to restore
    assert db.drop_stale_archives() and db.old_stats["Gone"] == [v1] and db.stale_archives() == []


def test_restore_needs_the_same_psd():
    """The archived path or blob: another PSD under the name is a new card (its own author and series)."""
    db = _db([_card("Gone")], seen=["Other"])
    db.prune_clean_cards(AT)
    assert not db.restore("Gone", "Items/New/Gone.psd", "b" * 40) and not db.restore("Gone", "Items/New/Gone.psd", None)
    assert "Gone" not in db.stats and db.restore("Gone", "Items/New/Gone.psd", "a" * 40)    # moved
    db.stats.clear()
    assert db.restore("Gone", "Items/Gone.psd", "b" * 40)                                  # edited in place


def test_restored_copy_matches_after_a_reparse():
    """A re-parse of the same blob (new parser, new timestamp) still drops the copy: same path and blob."""
    db = _db([_card("Gone")], seen=["Other"])
    db.prune_clean_cards(AT)
    _restore(db, "Gone")
    db.stats["Gone"] = _card("Gone", ability="Parsed again.")
    db.stats["Gone"].timestamp += 5
    assert db.drop_stale_archives() and "Gone" not in db.old_stats


def test_blobless_record_is_a_copy_only_when_equal():
    """No blob: a stub at the same path is not the archived card (never dropped for it); an equal restore is."""
    real = _card("Legacy", blob=None, ability="The real card.")
    db = _db([real], seen=["Other"])
    db.prune_clean_cards(AT)
    assert _restore(db, "Legacy") and db.drop_stale_archives() and "Legacy" not in db.old_stats
    db = _db([real], seen=["Other"])
    real.archived = None
    db.prune_clean_cards(AT)
    db.stats["Legacy"] = cm.CardInfo(name="Legacy", card_type="unknown", path=real.path)
    assert db.drop_stale_archives() and [c.ability for c in db.old_stats["Legacy"]] == ["The real card."]


def test_listed_name_with_another_tagged_version_untags_it():
    """A /pull mix (listed card, an older tagged version): another version stays as history, untagged; the listed
    card's data again (another blob) is no version, so it goes."""
    old, same = _card("Mixed", blob="0" * 40, ability="Did a thing."), _card("Same", blob="0" * 40)
    for card in (old, same):
        card.archived = {"at": AT, "path": card.path, "blob": card.blob_sha, "commit": None}
    db = _db([_card("Mixed"), _card("Same")], old=[old, same])
    assert db.stale_archives() == ["Mixed", "Same"] and db.drop_stale_archives()
    assert db.old_stats["Mixed"] == [old] and old.archived is None and "Same" not in db.old_stats


def test_untagged_history_is_never_restored():
    """A card reset by /update_metadata del_entry (its PSD still there) has untagged history: no restore."""
    db = _db([], old=[_card("Reset")])
    assert not _restore(db, "Reset") and "Reset" not in db.stats and len(db.old_stats["Reset"]) == 1
    assert not _restore(db, "Never there") and "Never there" not in db.old_stats


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


def test_a_copy_left_by_a_kill_is_dropped_when_the_psd_is_back():
    """Killed after old_stats.pkl while archiving, PSD back before the next run: the tagged copy is no 'old version'."""
    card = _card("Back")
    left = deepcopy(card)
    left.archived = {"at": 1.0, "path": card.path, "blob": card.blob_sha, "commit": None}
    db = _db([card], old=[_card("Back", blob="0" * 40), left])
    assert db.stale_archives() == ["Back"] and db.drop_stale_archives()
    assert [c.blob_sha for c in db.old_stats["Back"]] == ["0" * 40]
