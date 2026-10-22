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
    assert _restore(db, "Gone") and db.stats["Gone"].blob_sha == "2" * 40 and db.stats["Gone"].archived is None
    assert db.old_stats["Gone"][1].archived and db.archived() == {}   # the tagged record stays until the save
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


def test_restore_drops_an_emptied_history():
    db = _db([_card("Gone")], seen=["Other"])
    db.prune_clean_cards(AT)
    assert _restore(db, "Gone") and db.drop_stale_archives() and "Gone" not in db.old_stats


def test_save_drops_the_restored_copy_after_stats(monkeypatch):
    """old_stats (record kept), stats (card in), old_stats again (record out): no kill leaves it in neither."""
    db = _db([_card("Gone")], seen=["Other"])
    db.prune_clean_cards(AT)
    _restore(db, "Gone")
    saved = []
    monkeypatch.setattr(cm, "save_pickle", lambda obj, path, lock: saved.append((path, "Gone" in obj)))
    db.save()
    assert saved == [(cm.OLD_STATS_PKL, True), (cm.STATS_PKL, True), (cm.OLD_STATS_PKL, False)]
    saved.clear()
    db.save()
    assert saved == [(cm.OLD_STATS_PKL, False), (cm.STATS_PKL, True)]          # nothing stale: two writes


def test_restored_copy_matches_after_a_reparse():
    """A re-parse of the same blob (new parser, new timestamp) still drops the copy: same path and blob."""
    db = _db([_card("Gone")], seen=["Other"])
    db.prune_clean_cards(AT)
    _restore(db, "Gone")
    db.stats["Gone"] = _card("Gone", ability="Parsed again.")
    db.stats["Gone"].timestamp += 5
    assert db.drop_stale_archives() and "Gone" not in db.old_stats


def test_history_keeps_the_archived_copy_until_the_save(monkeypatch):
    """add_history on a restored card: the tagged record stays (a kill leaves it archived), untagged by the save."""
    monkeypatch.setattr(cm, "save_pickle", lambda obj, path, lock: None)
    db = _db([_card("Gone")], seen=["Other"])
    db.prune_clean_cards(AT)
    _restore(db, "Gone")
    db.add_history("Gone", db.stats["Gone"])
    assert len(db.old_stats["Gone"]) == 1 and db.old_stats["Gone"][0].archived
    db.stats["Gone"] = _card("Gone", blob="b" * 40)                           # the re-parse
    db.save()
    assert len(db.old_stats["Gone"]) == 1 and db.old_stats["Gone"][0].archived is None and not db.kept_archives


def test_history_replaces_an_equal_last_even_moved():
    """The version a kill left (before or after its path followed the card) is replaced, not doubled."""
    card = _card("Kept")
    moved = deepcopy(card)
    moved.path = "Items/New/Kept.psd"
    db = _db([card], old=[moved])
    db.add_history("Kept", card)
    assert db.old_stats["Kept"] == [card]
    db.add_history("Kept", deepcopy(card))
    assert len(db.old_stats["Kept"]) == 1
    db.add_history("Kept", _card("Kept", blob="b" * 40))
    assert [c.blob_sha for c in db.old_stats["Kept"]] == ["a" * 40, "b" * 40]


def test_moved_restore_keeps_the_history():
    """Restored at its old path, parsed at a new one: the record the paths moved to is history, not a copy."""
    db = _db([_card("Gone")], seen=["Other"])
    db.prune_clean_cards(AT)
    _restore(db, "Gone")
    db.add_history("Gone", db.stats["Gone"])
    db.stats["Gone"] = _card("Gone")
    db.stats["Gone"].path = db.old_stats["Gone"][0].path = "Items/New/Gone.psd"
    assert db.drop_stale_archives() and len(db.old_stats["Gone"]) == 1 and db.old_stats["Gone"][0].archived is None


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
    """A /pull mix (listed card, an older tagged version): the version stays as history, untagged."""
    old = _card("Mixed", blob="0" * 40)
    old.archived = {"at": AT, "path": old.path, "blob": old.blob_sha, "commit": None}
    db = _db([_card("Mixed")], old=[old])
    assert db.stale_archives() == ["Mixed"] and db.drop_stale_archives()
    assert db.old_stats["Mixed"] == [old] and old.archived is None


# ========================
# rescue_dropped (/pull)
# ========================
def _tagged(card):
    card = deepcopy(card)
    card.archived = {"at": AT, "path": card.path, "blob": card.blob_sha, "commit": None}
    return card


def test_rescue_brings_back_an_archive_the_merge_dropped():
    """stats.pkl from the side that archived X, old_stats.pkl from a backfill-only side: X goes back, tagged."""
    x, v1 = _card("X"), _card("X", blob="1" * 40)
    archiver = ({"Y": _card("Y")}, {"X": [v1, _tagged(x)]})
    other = ({"X": x, "Y": _card("Y")}, {"X": [v1], "Z": [_tagged(_card("Z"))]})
    merged_old = deepcopy(other[1])
    assert cm.rescue_dropped(dict(archiver[0]), merged_old, [other, archiver]) == ["X"]
    assert [c.blob_sha for c in merged_old["X"]] == ["1" * 40, "a" * 40] and merged_old["X"][1].archived
    assert set(merged_old) == {"X", "Z"}


def test_rescue_takes_the_listed_card_when_no_side_archived():
    """Tagged, so the next /update_stats restores it (same PSD: the copy goes) or keeps it archived."""
    x = _card("X")
    merged_old = {}
    assert cm.rescue_dropped({}, merged_old, [({"X": x}, {}), ({}, {})]) == ["X"] and x.archived is None
    assert merged_old["X"][0].archived["path"] == x.path and cm._equal_but(merged_old["X"][0], x, "archived")
    db = _db(old=merged_old["X"], seen=["X"])
    assert set(db.archived()) == {"X"} and _restore(db, "X") and db.drop_stale_archives() and "X" not in db.old_stats


def test_rescue_leaves_listed_names_whole_histories_and_duplicates():
    x, dup = _card("X"), _card("X2", duplicate_of="X")
    sides = [({"X": x, "X2": dup}, {"X": [_card("X", blob="1" * 40)]}), ({"X": x}, {"X": [_tagged(x)]})]
    merged_old = {"X": [_tagged(x)]}
    assert cm.rescue_dropped({"X": x}, merged_old, sides) == [] and merged_old == {"X": [_tagged(x)]}
    gone = _tagged(_card("Gone"))
    merged_old = {"Gone": [gone]}
    assert cm.rescue_dropped({}, merged_old, [({"Gone": _card("Gone")}, {}), ({}, {"Gone": [gone]})]) == []


def test_untagged_history_is_never_restored():
    """A card reset by /update_metadata del_entry (its PSD still there) has untagged history: no restore."""
    db = _db([], old=[_card("Reset")])
    assert not _restore(db, "Reset") and "Reset" not in db.stats and len(db.old_stats["Reset"]) == 1
    assert not _restore(db, "Never there") and "Never there" not in db.old_stats


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


def test_a_copy_left_by_a_kill_is_dropped_when_the_psd_is_back():
    """Killed after old_stats.pkl while archiving, PSD back before the next run: the tagged copy is no 'old version'."""
    card = _card("Back")
    left = deepcopy(card)
    left.archived = {"at": 1.0, "path": card.path, "blob": card.blob_sha, "commit": None}
    db = _db([card], old=[_card("Back", blob="0" * 40), left])
    assert db.stale_archives() == ["Back"] and db.drop_stale_archives()
    assert [c.blob_sha for c in db.old_stats["Back"]] == ["0" * 40]


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


# ========================
# /update_metadata, /pull
# ========================
def test_update_metadata_never_stubs_an_archived_card(monkeypatch):
    """Its PNG may still be in the tree: show the archived record, refuse edits, make no stats entry."""
    import commands.psd_analyzer as pa
    import commands.query_card as qc
    from types import SimpleNamespace
    sent = []
    db = _db([_card("The Morgue")], seen=["Other"])
    db.prune_clean_cards(AT)
    monkeypatch.setattr(pa, "stats_db", db)
    monkeypatch.setattr(pa, "queue_message", lambda inter, text, **kw: sent.append(text))
    monkeypatch.setattr(db, "save", lambda: sent.append("saved"))
    monkeypatch.setattr(qc.card_repo, "git_files", {"the morgue.png": "MDW/The Morgue/The_Morgue.png"})
    monkeypatch.setattr(qc.card_repo, "git_file_alias", {})
    pa.manual_metadata_entry(SimpleNamespace(), "The Morgue")
    assert "is archived" in sent[0] and "Does a thing." in sent[0]
    pa.manual_metadata_entry(SimpleNamespace(), "The Morgue", hp=3)
    assert "can't be edited" in sent[1] and len(sent) == 2 and db.stats == {}
    assert db.old_stats["The Morgue"][0].archived and db.old_stats["The Morgue"][0].stats.hp != 3


def _git(*args):
    import subprocess
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True, capture_output=True)


def _commit(stats, old, message):
    for obj, name in ((stats, cm.STATS_PKL), (old, cm.OLD_STATS_PKL)):
        with open(name, "wb") as f:
            pickle.dump(obj, f)
    _git("add", cm.STATS_PKL, cm.OLD_STATS_PKL)
    _git("commit", "-q", "-m", message)


def test_pull_merge_keeps_an_archive(monkeypatch):
    """A /pull on the side that only changed old_stats.pkl, merging the side that archived X (the conflict keeps
    our old_stats.pkl): X goes back into old_stats.pkl, committed."""
    import commands.update_bot as ub
    x, y = _card("X"), _card("Y")
    _git("init", "-q", "-b", "main")
    _commit({"X": x, "Y": y}, defaultdict(list), "base")
    _git("checkout", "-q", "-b", "archiver")
    _commit({"Y": y}, defaultdict(list, {"X": [_tagged(x)]}), "archive X")
    _git("checkout", "-q", "main")
    _commit({"X": x, "Y": y}, defaultdict(list, {"Z": [_tagged(_card("Z"))]}), "backfill Z")
    with pytest.raises(Exception):
        _git("merge", "-q", "--no-edit", "archiver")
    _git("checkout", "--ours", cm.OLD_STATS_PKL)
    _git("add", cm.OLD_STATS_PKL)
    _git("commit", "-q", "--no-edit")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "t")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "t@t")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "t")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "t@t")
    assert ub._rescue_merged_cards() == ["X"]
    db = cm.StatsDatabase()
    db.load()
    assert set(db.archived()) == {"X", "Z"} and set(db.stats) == {"Y"}
    import subprocess
    log = subprocess.run(["git", "log", "-1", "--format=%s"], capture_output=True, text=True).stdout.strip()
    assert log == "PICKLE" and ub._rescue_merged_cards() == []              # the PICKLE commit is no merge
