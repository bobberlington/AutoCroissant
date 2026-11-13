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
    assert db.prune_clean_cards(AT) == ["Gone"]
    versions = db.old_stats["Gone"]
    assert "Gone" not in db.stats and versions[0] is v1 and v1.archived is None
    assert versions[1].archived["blob"] == "2" * 40 and set(db.archived()) == {"Gone"}


def test_duplicates_are_archived_at_their_png():
    """An alt-art duplicate not seen (its PNG or original left) is a card under the archive rule: tagged at its PNG."""
    dup = _card("A Snack2", duplicate_of="A Snack")
    dup.path = "Items/A_Snack.psd"                      # a picture-only copy: its original's path
    db = _db([_card("A Snack"), dup], seen=["A Snack"])
    assert db.prune_clean_cards(AT) == ["A Snack2"] and "A Snack2" not in db.stats
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
    assert not _restore(db, "Never there") and "Never there" not in db.old_stats


def test_restored_copy_matches_after_a_reparse():
    """A re-parse of the same blob (new parser, new timestamp) still drops the copy: same path and blob."""
    db = _db([_card("Gone")], seen=["Other"])
    db.prune_clean_cards(AT)
    _restore(db, "Gone")
    db.stats["Gone"] = _card("Gone", ability="Parsed again.")
    db.stats["Gone"].timestamp += 5
    assert db.drop_stale_archives() and "Gone" not in db.old_stats


def test_listed_name_with_another_tagged_version_untags_it():
    """A /pull mix (listed card, an older tagged version): another version stays as history, untagged; the listed
    card's data again (another blob) is no version, so it goes."""
    old, same = _card("Mixed", blob="0" * 40, ability="Did a thing."), _card("Same", blob="0" * 40)
    for card in (old, same):
        card.archived = {"at": AT, "path": card.path, "blob": card.blob_sha, "commit": None}
    db = _db([_card("Mixed"), _card("Same")], old=[old, same])
    assert db.stale_archives() == ["Mixed", "Same"] and db.drop_stale_archives()
    assert db.old_stats["Mixed"] == [old] and old.archived is None and "Same" not in db.old_stats


def test_an_untagged_equal_version_stays_history():
    """An old untagged version equal to the card (a 2025 balloon) stays: the archived card goes behind it."""
    card = _card("Gone")
    db = _db([card], old=[deepcopy(card)], seen=["Other"])
    db.prune_clean_cards(AT)
    assert [c.archived is None for c in db.old_stats["Gone"]] == [True, False]
