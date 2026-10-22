"""B10: card pickle saves are atomic (tmp + fsync + replace) and serialized by one lock per pickle."""
import pickle
import threading
import time
from collections import defaultdict
from os import listdir

import pytest

import commands.card_models as cm
import commands.query_card as qc


# ========================
# Helpers
# ========================
def _db():
    db = cm.StatsDatabase()
    db.stats = {"A": cm.CardInfo(name="A", card_type="item", stats=cm.CardStats(hp=1)),
                "B": cm.CardInfo(name="B", card_type="creature", ability="Deal 2 damage.")}
    db.old_stats = defaultdict(list, {"A": [cm.CardInfo(name="A", card_type="item")]})
    return db


@pytest.fixture
def paths(tmp_path, monkeypatch):
    stats, old, alias = (str(tmp_path / n) for n in ("stats.pkl", "old_stats.pkl", "aliases.pkl"))
    monkeypatch.setattr(cm, "STATS_PKL", stats)
    monkeypatch.setattr(cm, "OLD_STATS_PKL", old)
    monkeypatch.setattr(qc, "ALIAS_PKL", alias)
    return tmp_path


def _partial_then_die(obj, f):
    f.write(b"\x80\x05partial")
    raise RuntimeError("killed mid-write")


# ========================
# StatsDatabase.save
# ========================
def test_saved_pickles_are_the_same_bytes_and_load_back_equal(paths):
    db = _db()
    db.save()
    assert (paths / "stats.pkl").read_bytes() == pickle.dumps(db.stats)          # what the plain dump wrote
    assert (paths / "old_stats.pkl").read_bytes() == pickle.dumps(db.old_stats)
    back = cm.StatsDatabase()
    back.load()
    assert back.stats == db.stats and back.old_stats == db.old_stats
    assert sorted(listdir(paths)) == ["old_stats.pkl", "stats.pkl"]                # no temp file left


def test_a_save_killed_mid_write_keeps_the_old_files(paths, monkeypatch):
    db = _db()
    db.save()
    before = {n: (paths / n).read_bytes() for n in ("stats.pkl", "old_stats.pkl")}
    db.stats["C"] = cm.CardInfo(name="C", card_type="field")
    monkeypatch.setattr(cm, "dump", _partial_then_die)
    with pytest.raises(RuntimeError):
        db.save()
    assert {n: (paths / n).read_bytes() for n in before} == before
    assert sorted(listdir(paths)) == ["old_stats.pkl", "stats.pkl"]
    back = cm.StatsDatabase()
    back.load()
    assert set(back.stats) == {"A", "B"}


def test_concurrent_saves_serialize_per_pickle(paths, monkeypatch):
    active, overlaps, real_dump = {}, [], pickle.dump

    def slow_dump(obj, f):
        active[f.name] = active.get(f.name, 0) + 1
        if active[f.name] > 1:
            overlaps.append(f.name)
        time.sleep(0.02)
        real_dump(obj, f)
        active[f.name] -= 1
    monkeypatch.setattr(cm, "dump", slow_dump)
    db = _db()
    threads = [threading.Thread(target=db.save) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert overlaps == []
    assert pickle.loads((paths / "stats.pkl").read_bytes()) == db.stats
    assert sorted(listdir(paths)) == ["old_stats.pkl", "stats.pkl"]


def test_save_holds_the_pickles_own_lock(paths, monkeypatch):
    held = []
    monkeypatch.setattr(cm, "dump", lambda obj, f: held.append((f.name, cm._stats_lock.locked(), cm._old_stats_lock.locked())))
    _db().save()
    assert held == [(f"{paths}/old_stats.pkl.tmp", False, True), (f"{paths}/stats.pkl.tmp", True, False)]   # old_stats first


# ========================
# save_aliases
# ========================
def test_aliases_save_is_atomic_and_locked(paths, monkeypatch):
    repo = qc.CardRepository()
    repo.git_file_alias = {"abyss_watchers": "the_abyss_watchers.png"}
    repo.save_aliases()
    assert (paths / "aliases.pkl").read_bytes() == pickle.dumps(repo.git_file_alias)
    back = qc.CardRepository()
    back.load_aliases()
    assert back.git_file_alias == repo.git_file_alias

    held = []
    monkeypatch.setattr(cm, "dump", lambda obj, f: held.append(qc._alias_lock.locked()))
    repo.save_aliases()
    assert held == [True]

    before = (paths / "aliases.pkl").read_bytes()
    monkeypatch.setattr(cm, "dump", _partial_then_die)
    with pytest.raises(RuntimeError):
        repo.save_aliases()
    assert (paths / "aliases.pkl").read_bytes() == before and sorted(listdir(paths)) == ["aliases.pkl"]
