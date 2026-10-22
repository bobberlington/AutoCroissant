"""card_history: the git index (newest, first author, blob intro, deletions, exact renames), the version rules, the seed
pass (dedupe that keeps archived tags, re-dating, legacy marking, archive dates) and a build's versions."""
import os
import pickle
import subprocess
from collections import defaultdict
from copy import deepcopy
from types import SimpleNamespace

import pytest

import commands.card_db as cdb
import commands.card_history as ch
import commands.card_models as cm


def git(repo, *args, env=None):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True,
                          env={**os.environ, **(env or {})}).stdout


def commit(repo, author, at, message="c"):
    stamp = f"@{at} +0000"
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message, env={"GIT_AUTHOR_NAME": author, "GIT_AUTHOR_EMAIL": f"{author}@x",
                                                  "GIT_COMMITTER_NAME": author, "GIT_COMMITTER_EMAIL": f"{author}@x",
                                                  "GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp})
    return git(repo, "rev-parse", "HEAD").strip()


def write(repo, rel, data: bytes):
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def blob(data: bytes) -> str:
    import hashlib
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


X, Y = b"psd one", b"psd two"


@pytest.fixture
def history(tmp_path):
    """c1 alice adds A (X) + PNG; c2 bob edits A (Y) + PNG; c3 carol moves A to Heal/; c4 dave adds B (X again);
    c5 erin deletes B and renames Heal/A to Heal/C (same blob)."""
    repo = tmp_path / "cards"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    write(repo, "Items/A.psd", X)
    write(repo, "Items/A.png", b"png 1")
    c1 = commit(repo, "alice", 1000)
    write(repo, "Items/A.psd", Y)
    write(repo, "Items/A.png", b"png 2")
    c2 = commit(repo, "bob", 2000)
    (repo / "Items/Heal").mkdir()
    git(repo, "mv", "Items/A.psd", "Items/Heal/A.psd")
    git(repo, "mv", "Items/A.png", "Items/Heal/A.png")
    c3 = commit(repo, "carol", 3000)
    write(repo, "Items/B.psd", X)
    c4 = commit(repo, "dave", 4000)
    git(repo, "rm", "-q", "Items/B.psd")
    git(repo, "mv", "Items/Heal/A.psd", "Items/Heal/C.psd")
    c5 = commit(repo, "erin", 5000)
    return SimpleNamespace(repo=repo, c=[None, c1, c2, c3, c4, c5])


# ========================
# The index
# ========================
def test_index_from_one_log(history):
    index = ch.GitIndex.from_git(str(history.repo))
    c = history.c
    assert [x.sha for x in index.commits] == [c[5], c[4], c[3], c[2], c[1]] and index.commits[0].at == 5000.0
    assert index.newest["Items/Heal/A.psd"].sha == c[5] and index.first["Items/A.psd"].author == "alice"
    assert index.intro[blob(X)][0].sha == c[1] and index.intro[blob(X)][1] == "Items/A.psd"      # B's same bytes: no new date
    assert index.blob_intro(blob(Y)).sha == c[2] and index.intro_at[(blob(Y), "Items/Heal/A.psd")].sha == c[3]
    assert index.deleted["Items/B.psd"].sha == c[5] and "Items/Heal/C.psd" not in index.deleted
    assert index.renames["Items/Heal/A.psd"] == ("Items/Heal/C.psd", index.by_sha[c[5]])
    assert index.renames["Items/A.psd"][0] == "Items/Heal/A.psd" and "Items/B.psd" not in index.renames
    assert [x.sha for x in index.pngs["Items/A.png"]] == [c[3], c[2], c[1]]
    assert index.png_commit("Items/A.psd", index.by_sha[c[3]]) == c[2]                    # older than the end commit
    assert index.png_commit("Items/A.psd", before_at=1500) == c[1] and index.png_commit("Items/A.psd") == c[3]


def test_index_reads_no_blob(history, tmp_path):
    """A blobless clone: the index is built with no blob fetched (GIT_NO_LAZY_FETCH would fail a fetch loudly)."""
    git(history.repo, "config", "uploadpack.allowFilter", "true")
    clone = tmp_path / "blobless.git"
    subprocess.run(["git", "clone", "-q", "--bare", "--filter=blob:none", f"file://{history.repo}", str(clone)], check=True)
    index = ch.GitIndex.from_git(str(clone))
    assert index.blob_intro(blob(Y)).sha == history.c[2]
    missing = subprocess.run(["git", "-C", str(clone), "cat-file", "-e", blob(Y)], env={**os.environ, "GIT_NO_LAZY_FETCH": "1"})
    assert missing.returncode != 0                                                         # the blob is still not there


def test_shallow_clone_is_refused(history, tmp_path):
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth", "1", f"file://{history.repo}", str(shallow)], check=True)
    with pytest.raises(ValueError, match="non-shallow"):
        ch.GitIndex.from_git(str(shallow))


def test_parse_handles_odd_paths_and_merges():
    text = ("C\taaa\t10\tann\0\n:000000 100644 0000000000000000000000000000000000000000 " + "1" * 40 + " A\0Items/We\tird \"x\".psd\0"
            "C\tbbb\t5\tbo b\0")                                                           # a commit with no raw lines (a merge)
    index = ch.GitIndex.parse(text)
    assert index.newest['Items/We\tird "x".psd'].author == "ann" and [c.author for c in index.commits] == ["ann", "bo b"]


# ========================
# Versions
# ========================
def _card(name="A", path="Items/A.psd", blob_sha=None, ability="Text.", timestamp=1.0, **kw):
    return cm.CardInfo(name=name, card_type="item", path=path, ability=ability, blob_sha=blob_sha, timestamp=timestamp, **kw)


def test_start_of_causes(history):
    index = ch.GitIndex.from_git(str(history.repo))
    new = _card(path="Items/Heal/A.psd", blob_sha=blob(Y))
    assert ch.start_of(new, index, None)["cause"] == "psd" and ch.start_of(new, index, None)["from"] == 2000.0
    same = _card(path="Items/Heal/A.psd", blob_sha=blob(Y))
    assert ch.start_of(new, index, same)["cause"] == "reparse"
    moved = _card(path="Items/A.psd", blob_sha=blob(Y))
    start = ch.start_of(new, index, moved)
    assert (start["cause"], start["from"], start["from_commit"]) == ("moved", 3000.0, history.c[3])
    legacy = ch.start_of(_card(timestamp=42.0), index, None)
    assert (legacy["cause"], legacy["from"], legacy["legacy"], legacy["from_commit"]) == ("legacy", 42.0, True, None)


def _index_with(intros: dict, deletions: dict = None, pngs: dict = None) -> ch.GitIndex:
    """An index by hand: blob -> (sha, at); path -> (sha, at) deleted; png path -> [(sha, at)] newest first."""
    index, order = ch.GitIndex(), 0
    for blob_sha, (sha, at) in intros.items():
        commit = index.by_sha.setdefault(sha, ch.Commit(order, sha, at, "someone"))
        index.intro[blob_sha] = (commit, "Items/A.psd")
        order += 1
    for path, (sha, at) in (deletions or {}).items():
        index.deleted[path] = index.by_sha.setdefault(sha, ch.Commit(order, sha, at, "remover"))
    for path, items in (pngs or {}).items():
        index.pngs[path] = [index.by_sha.setdefault(sha, ch.Commit(100 - at, sha, at, "x")) for sha, at in items]
    index.commits = sorted(index.by_sha.values(), key=lambda c: -c.at)
    for i, c in enumerate(index.commits):
        c.order = i
    return index


def test_refine_dedupes_dates_and_marks_legacy():
    """[v0 legacy D1, v1 legacy D1, v2 blob D2] + current D2 (another blob): v0 equals v1 and v2 equals the current card,
    so both go; their starts carry forward. The blob version is re-dated, the blob-less ones marked legacy."""
    index = _index_with({"b2": ("c2", 2000.0), "b3": ("c3", 3000.0)}, pngs={"Items/A.png": [("p3", 3000.0), ("p1", 500.0)]})
    v0, v1 = _card(timestamp=100.0), _card(timestamp=200.0)
    v2 = _card(ability="New.", blob_sha="b2", timestamp=2500.0)
    current = _card(ability="New.", blob_sha="b3", timestamp=3500.0)
    stats, old = {"A": current}, defaultdict(list, {"A": [v0, v1, v2]})
    report = ch.refine(stats, old, index)
    assert [n for n, _ in report.dropped] == ["A", "A"] and old["A"] == [v1]
    assert (report.legacy, report.legacy_kept, report.versions_with_blob, report.versions_dated) == (2, 1, 1, 1)
    assert v1.version["from"] == 100.0 and v1.version["legacy"] and v1.version["to"] == 2000.0     # v0's start, ends at v2's
    assert current.timestamp == 3000.0 and report.cards_dated == ["A"]
    assert (current.version["from"], current.version["from_commit"], current.version["cause"]) == (2000.0, "c2", "psd")
    assert v1.version["png_commit"] == "p1"                                                        # the PNG before v1 ended
    again = deepcopy((stats, old))
    assert not ch.refine(*again, index).changed() and cdb.record(again[0]["A"]) == cdb.record(stats["A"])
    assert [cdb.record(v) for v in again[1]["A"]] == [cdb.record(v) for v in old["A"]]          # idempotent


def test_refine_keeps_archived_tags_and_dates_them():
    index = _index_with({"b1": ("c1", 1000.0)}, deletions={"Items/Gone.psd": ("cd", 4000.0)})
    index.renames["Items/Gone.psd"] = ("Items/New.psd", index.by_sha["cd"])
    gone_old = _card("Gone", path="Items/Gone.psd", timestamp=10.0)
    gone = _card("Gone", path="Items/Gone.psd", blob_sha="b1", timestamp=900.0)                    # same data, last: kept
    tagged_twin = _card("Twin", path="Items/Twin.psd", timestamp=20.0, archived={"at": 1.0, "path": "Items/Twin.psd",
                                                                                  "blob": None, "commit": None})
    stats = {"New": _card("New", path="Items/New.psd", blob_sha="b1"), "Twin": _card("Twin", path="Items/Twin.psd", timestamp=30.0)}
    old = defaultdict(list, {"Gone": [gone_old, gone], "Twin": [tagged_twin]})
    report = ch.refine(stats, old, index)
    assert old["Gone"] == [gone] and report.dropped == [("Gone", 10.0)]                    # equal to the archived one after it
    tag = old["Gone"][0].archived
    assert tag == {"at": 4000.0, "path": "Items/Gone.psd", "blob": "b1", "commit": "cd", "backfill": True, "renamed_to": "New"}
    assert report.tagged == ["Gone"] and report.renamed == [("Gone", "New")]
    assert old["Twin"] == [tagged_twin]                                       # equal to the card, but tagged: never dropped
    assert old["Gone"][len(old["Gone"]) - 1].version["to"] == 4000.0       # an archived version ends at its deletion


def test_duplicates_follow_their_original_and_get_no_version():
    index = _index_with({"b1": ("c1", 1000.0)})
    stats = {"A": _card(blob_sha="b1", timestamp=5.0), "A2": _card("A2", blob_sha="b1", timestamp=5.0, duplicate_of="A")}
    report = ch.refine(stats, defaultdict(list), index)
    assert stats["A2"].timestamp == 1000.0 and report.duplicates_dated == 1 and stats["A2"].version is None


# ========================
# A build's versions
# ========================
def _db(stats, old=None):
    db = cm.StatsDatabase()
    db.stats, db.old_stats = stats, defaultdict(list, old or {})
    return db


def test_build_versions_only_on_data_changes():
    index = _index_with({"b1": ("c1", 1000.0), "b2": ("c2", 2000.0)})
    first = _card(blob_sha="b1", timestamp=1000.0)
    first.version = ch.start_of(first, index, None)
    art, moved = _card("Art", blob_sha="b1"), _card("Moved", blob_sha="b1")
    for c in (art, moved):
        c.version = ch.start_of(c, index, None)
    before = {"A": first, "Art": art, "Moved": moved}
    after = {"A": _card(ability="Changed.", blob_sha="b2", timestamp=2000.0),
             "Art": _card("Art", blob_sha="b2"), "Moved": _card("Moved", path="Items/Heal/Moved.psd", blob_sha="b1")}
    db = _db(after)
    made = ch.record_versions(deepcopy(before), {}, db, {"A": 1, "Art": 1, "Moved": 1}, [], index)
    assert made == [("A", "psd")] and list(db.old_stats) == ["A"]
    version = db.old_stats["A"][0]
    assert version.path == "Items/A.psd" and version.ability == "Text." and version.version["to"] == 2000.0
    assert db.stats["A"].version["from_commit"] == "c2"
    assert db.stats["Art"].version["from"] == 1000.0 and db.stats["Moved"].version["to"] is None   # their versions go on


def test_build_versions_from_edits_and_authors():
    index = _index_with({"b1": ("c1", 1000.0)})
    edits, authors = ch.Commit(0, "e1", 7000.0, "web"), ch.Commit(0, "a1", 8000.0, "web")
    before = {"E": _card("E", blob_sha="b1", note=None), "W": _card("W", blob_sha="b1", author="Old")}
    after = {"E": _card("E", blob_sha="b1", note="checked"), "W": _card("W", blob_sha="b1", author="New")}
    db = _db(after)
    assert sorted(ch.record_versions(before, {}, db, {}, [], index, edits, authors)) == [("E", "edit"), ("W", "authors")]
    assert db.stats["E"].version["from_commit"] == "e1" and db.stats["W"].version["from"] == 8000.0
    assert db.old_stats["E"][0].note is None and db.old_stats["E"][0].version["to_commit"] == "e1"


def test_build_versions_of_a_restored_card():
    """Restored with other data: the archived record (still last) is closed where the card starts again, not doubled."""
    index = _index_with({"b1": ("c1", 1000.0), "b2": ("c2", 2000.0)})
    record = _card(blob_sha="b1", archived={"at": 1500.0, "path": "Items/A.psd", "blob": "b1", "commit": None})
    record.version = ch.start_of(record, index, None)
    db = _db({"A": _card(ability="Back.", blob_sha="b2")}, {"A": [record]})
    ch.record_versions({}, {"A": record}, db, {"A": 1}, ["A"], index)
    assert len(db.old_stats["A"]) == 1 and record.version["to"] == 2000.0 and db.stats["A"].version["cause"] == "psd"


# ========================
# CLI
# ========================
def test_cli_refine_on_copies(history, tmp_path, capsys):
    folder, out = tmp_path / "pickles", tmp_path / "out"
    folder.mkdir()
    stats = {"C": _card("C", path="Items/Heal/C.psd", blob_sha=blob(Y), timestamp=9.0)}
    old = defaultdict(list, {"C": [_card("C", path="Items/Heal/C.psd", ability="Old.")], "B": [_card("B", path="Items/B.psd")]})
    (folder / "stats.pkl").write_bytes(pickle.dumps(stats))
    (folder / "old_stats.pkl").write_bytes(pickle.dumps(old))
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    assert ch.main(["refine", "--git", str(history.repo), "--pickles", str(folder), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "Cards re-dated: 1 of 1" in text and "Archive tags added: 1 (B)" in text and "Archive tags dated from git: 1 (B)" in text
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before                  # the input is never written
    refined = pickle.loads((out / "stats.pkl").read_bytes())
    assert refined["C"].timestamp == 2000.0 and refined["C"].version["cause"] == "psd"
    assert pickle.loads((out / "old_stats.pkl").read_bytes())["B"][0].archived["commit"] == history.c[5]
    assert ch.main(["refine", "--git", str(history.repo), "--pickles", str(folder), "--out", str(folder)]) == 1
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before
