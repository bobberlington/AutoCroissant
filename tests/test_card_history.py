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
    assert index.origin("Items/Heal/C.psd") == "Items/A.psd" and index.origin("Items/Heal/C.psd", once=True) == \
        "Items/Heal/A.psd" and index.first[index.origin("Items/Heal/C.psd")].author == "alice"   # not the renamers
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


def test_a_revert_dates_from_the_revert(tmp_path):
    """King Dedede (2025-04-04): bytes X, then Y, then X again (a revert): the card and a version read after the revert
    start at the revert, one read before it at X's intro. Holo Defense API: renamed after its revert (same bytes), it
    still starts at the revert. The same bytes at a path that held no others keep their intro, and so does an exact
    rename onto a path that held others. A start dates itself."""
    repo = tmp_path / "cards"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    write(repo, "Items/A.psd", X)
    c1 = commit(repo, "alice", 1000)
    write(repo, "Items/A.psd", Y)
    c2 = commit(repo, "bob", 2000)
    write(repo, "Items/A.psd", X)
    c3 = commit(repo, "carol", 3000)
    index = ch.GitIndex.from_git(str(repo))
    since = lambda path, at=None: index.blob_since(blob(X), path, at).sha
    assert (since("Items/A.psd"), since("Items/A.psd", 3500.0), since("Items/A.psd", 3000.0)) == (c3, c3, c3)
    assert since("Items/A.psd", 1500.0) == since("Items/A.psd", 1000.0) == index.blob_intro(blob(X)).sha == c1
    stats = {"A": _card(blob_sha=blob(X), timestamp=3000.0)}
    old = defaultdict(list, {"A": [_card(ability="One.", blob_sha=blob(X), timestamp=1000.0),
                                   _card(ability="Two.", blob_sha=blob(Y), timestamp=2000.0)]})
    ch.refine(stats, old, index)
    assert [(v.version["from"], v.version["to"]) for v in old["A"]] == [(1000.0, 2000.0), (2000.0, 3000.0)]
    assert (stats["A"].timestamp, stats["A"].version["from"], stats["A"].version["from_commit"]) == (3000.0, 3000.0, c3)
    git(repo, "mv", "Items/A.psd", "Items/B.psd")
    commit(repo, "dave", 4000)
    write(repo, "Items/C.psd", X)
    commit(repo, "erin", 5000)
    write(repo, "Items/D.psd", b"other")
    commit(repo, "fay", 6000)
    git(repo, "rm", "-q", "Items/D.psd")
    commit(repo, "fay", 6500)
    git(repo, "mv", "Items/C.psd", "Items/D.psd")
    commit(repo, "gus", 7000)
    index = ch.GitIndex.from_git(str(repo))
    assert since("Items/B.psd") == since("Items/B.psd", 4000.0) == since("Items/B.psd", 3000.0) == c3
    assert since("Items/D.psd") == c1             # a copy (no other bytes at its path), renamed onto a used path


# ========================
# Versions
# ========================
def _card(name="A", path="Items/A.psd", blob_sha=None, ability="Text.", timestamp=1.0, card_type="item", **kw):
    return cm.CardInfo(name=name, card_type=card_type, path=path, ability=ability, blob_sha=blob_sha, timestamp=timestamp,
                       **kw)


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


def test_the_seed_dates_legacy_versions_from_their_commit(history):
    """A blob-less version whose timestamp is its path's commit time gets that commit's blob, so git dates it (no
    longer approximate); a carried start gets its commit; a PNG named in another case still pins."""
    index = ch.GitIndex.from_git(str(history.repo))
    c = history.c
    early = _card(ability="Old.", timestamp=1000.0)                                 # Items/A.psd at c1, no blob kept
    current = _card(blob_sha=blob(Y), timestamp=2000.0)
    stats, old = {"A": current}, defaultdict(list, {"A": [early]})
    report = ch.refine(stats, old, index)
    assert early.blob_sha == blob(X) and (early.version["cause"], early.version["legacy"]) == ("psd", False)
    assert (early.version["from"], early.version["from_commit"], early.version["to"]) == (1000.0, c[1], 2000.0)
    assert (report.blobs_found, report.commits_found) == (1, 0)
    carried = _card(blob_sha=blob(Y), timestamp=2000.0)
    carried.version = ch.meta(None, 1000.0, "legacy")                               # a start /update_stats carried
    stats, old = {"A": carried}, defaultdict(list)
    report = ch.refine(stats, old, index)
    assert (carried.version["from"], carried.version["from_commit"], report.commits_found) == (1000.0, c[1], 1)
    assert not ch.refine(stats, old, index).changed()
    assert index.png_commit("Items/a.psd") == index.png_commit("Items/A.psd") is not None
    assert index.blob_at("Items/A.psd", 1500.0) is None


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
    """[v0 legacy D1, v1 legacy D1, v2 blob D2] + current D2 (another blob): v1 repeats v0 and v2 equals the current card,
    so both go (the first of a run stays, a run equal to the card goes into it, its start carried). The blob version is
    re-dated, the blob-less ones marked legacy."""
    index = _index_with({"b2": ("c2", 2000.0), "b3": ("c3", 3000.0)}, pngs={"Items/A.png": [("p3", 3000.0), ("p1", 500.0)]})
    v0, v1 = _card(timestamp=100.0), _card(timestamp=200.0)
    v2 = _card(ability="New.", blob_sha="b2", timestamp=2500.0)
    current = _card(ability="New.", blob_sha="b3", timestamp=3500.0)
    stats, old = {"A": current}, defaultdict(list, {"A": [v0, v1, v2]})
    report = ch.refine(stats, old, index)
    assert report.dropped == [("A", 200.0), ("A", 2000.0)] and old["A"] == [v0]
    assert (report.legacy, report.legacy_kept, report.versions_with_blob, report.versions_dated) == (2, 1, 1, 1)
    assert v0.version["from"] == 100.0 and v0.version["legacy"] and v0.version["to"] == 2000.0     # its start, ends at v2's
    assert current.timestamp == 3000.0 and report.cards_dated == ["A"]
    assert (current.version["from"], current.version["from_commit"], current.version["cause"]) == (2000.0, "c2", "psd")
    assert v0.version["png_commit"] == "p1"                                                        # the PNG before v0 ended
    again = deepcopy((stats, old))
    assert not ch.refine(*again, index).changed() and cdb.record(again[0]["A"]) == cdb.record(stats["A"])
    assert [cdb.record(v) for v in again[1]["A"]] == [cdb.record(v) for v in old["A"]]          # idempotent


def test_dropped_legacy_versions_leave_the_current_card_its_cause():
    """Every old version equal to the current card (The Lich King's case): the current card takes the earliest date,
    but keeps its own cause and is no legacy version (as_of would call it approximate)."""
    index = _index_with({"b3": ("c3", 3000.0)})
    current = _card(blob_sha="b3", timestamp=3500.0)
    stats, old = {"A": current}, defaultdict(list, {"A": [_card(timestamp=100.0), _card(timestamp=200.0)]})
    ch.refine(stats, old, index)
    assert "A" not in old
    assert (current.version["from"], current.version["from_commit"], current.version["cause"], current.version["legacy"]) == \
        (100.0, None, "psd", False)


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


def test_a_stale_index_never_dates_a_tag_back_to_an_older_deletion():
    """Items/A.psd deleted at 150, added back at 200 and deleted again at 300 (tag from _archive_sources): a clone not
    fetched since the second deletion knows only the first, which dates nothing, with or without the tag's commit."""
    index = _index_with({}, {"Items/A.psd": ("old", 150.0)})
    for commit in ("new", None):
        record = _card(timestamp=200.0, archived={"at": 300.0, "path": "Items/A.psd", "blob": None, "commit": commit})
        report = ch.Refined()
        ch.date_archive({}, defaultdict(list, {"A": [record]}), index, report, backfill=False)
        assert (record.archived["at"], record.archived["commit"], report.tags_dated) == (300.0, commit, [])


def test_a_stale_index_never_dates_a_restored_cards_tag():
    """Read at 100, deleted at 150, restored with the same bytes (its timestamp stays 100) and deleted at 300: a clone
    not fetched since knows only the 150 deletion (and a rename then), newer than the reading. The tag's commit is not
    in it, so neither dates the tag nor names a rename."""
    index = _index_with({}, {"Items/A.psd": ("old", 150.0)})
    index.renames["Items/A.psd"] = ("Items/B.psd", index.by_sha["old"])
    record = _card(timestamp=100.0, blob_sha="b1", archived={"at": 300.0, "path": "Items/A.psd", "blob": "b1",
                                                              "commit": "new"})
    report = ch.Refined()
    ch.date_archive({"B": _card("B", path="Items/B.psd")}, defaultdict(list, {"A": [record]}), index, report,
                    backfill=False)
    assert record.archived == {"at": 300.0, "path": "Items/A.psd", "blob": "b1", "commit": "new"}
    assert not report.changed()


def test_a_psd_moved_into_mdw_is_retired_not_deleted():
    """White Palace: its PSD moved (same bytes) into MDW/The Morgue, a folder of no cards. Its tag takes the commit's
    date (not the run's clock) and says where it went; a second pass changes nothing."""
    path = "Field/4 Stars/White_Palace.psd"
    index = _index_with({}, deletions={path: ("b633", 4000.0)})
    index.renames[path] = ("MDW/The Morgue/White_Palace.psd", index.by_sha["b633"])
    palace = _card("White Palace", path=path, blob_sha="a66e",
                   archived={"at": 9000.0, "path": path, "blob": "a66e", "commit": "b633"})
    old = defaultdict(list, {"White Palace": [palace]})
    report = ch.Refined()
    ch.date_archive({}, old, index, report)
    assert palace.archived == {"at": 4000.0, "path": path, "blob": "a66e", "commit": "b633",
                               "moved_to": "MDW/The Morgue/White_Palace.psd"}
    assert (report.tags_dated, report.retired, report.renamed) == (["White Palace"], ["White Palace"], [])
    again = ch.Refined()
    ch.date_archive({}, old, index, again)
    assert not again.changed()


def test_a_rename_that_edited_the_psd_follows_its_art():
    """Chestnut -> The Forbidden One: the PSD was renamed and edited (no exact PSD rename), its PNG renamed unchanged."""
    path = "Creatures/Other/4 Stars/Chestnut.psd"
    index = _index_with({}, deletions={path: ("6b23", 4000.0)})
    index.renames["Creatures/Other/4 Stars/Chestnut.png"] = ("Creatures/Other/4 Stars/The_Forbidden_One.png",
                                                              index.by_sha["6b23"])
    chestnut = _card("Chestnut", path=path, archived={"at": None, "path": path, "blob": None, "commit": None})
    report = ch.Refined()
    ch.date_archive({"The Forbidden One": _card("The Forbidden One")}, defaultdict(list, {"Chestnut": [chestnut]}), index,
                    report)
    assert chestnut.archived["renamed_to"] == "The Forbidden One" and report.renamed == [("Chestnut", "The Forbidden One")]


def test_collapse_keeps_the_first_of_a_run_and_folds_the_last_into_the_card():
    """/update_stats' cleanup (no records): [A 100, A 200 (another blob), B 300, B 400] + current B 500: the first A stays
    with its date, both B versions go into the current card, which starts at 300 (a legacy record: no git here)."""
    a1, a2 = _card(timestamp=100.0, blob_sha="b1", **V2), _card(timestamp=200.0, blob_sha="b2", **V2)
    b1, b2 = _card(ability="B.", timestamp=300.0), _card(ability="B.", timestamp=400.0, parsed_values={"name": "a"})
    current = _card(ability="B.", timestamp=500.0, blob_sha="b5")
    kept, dropped = ch.collapse([a1, a2, b1, b2], current)
    assert kept == [a1] and dropped == [a2, b1, b2] and a1.version is None
    assert current.version == {"from": 300.0, "from_commit": None, "to": None, "to_commit": None, "cause": "legacy",
                               "png_commit": None, "legacy": False}
    assert ch.collapse(kept, current) == (kept, [])                              # idempotent
    assert ch.start(current) == (300.0, None) and ch.start(a1) == (100.0, None)


def test_collapse_never_drops_an_archived_record():
    """An archived card's run goes into its tagged record; a tagged record equal to the listed card stays (a stale one)."""
    early, tagged = _card(timestamp=100.0), _card(timestamp=200.0, archived={"at": 300.0, "path": "Items/A.psd"})
    kept, dropped = ch.collapse([early, tagged], None)
    assert (kept, dropped) == ([tagged], [early]) and ch.start(tagged) == (100.0, None)
    stale = _card(timestamp=100.0, archived={"at": 300.0, "path": "Items/A.psd"})
    assert ch.collapse([stale], _card(timestamp=400.0)) == ([stale], [])


def test_a_tagged_record_ends_its_run():
    """[A1, A2 archived, A3] + another current card: the versions after the tag are the card's next life, so A3 stays
    (no tag left inside a list that had none after it); A1 goes into the tagged record."""
    a1, a3 = _card(timestamp=100.0), _card(timestamp=300.0)
    a2 = _card(timestamp=200.0, archived={"at": 250.0, "path": "Items/A.psd"})
    kept, dropped = ch.collapse([a1, a2, a3], _card(ability="B.", timestamp=400.0))
    assert (kept, dropped) == ([a2, a3], [a1]) and ch.start(a2) == (100.0, None) and a3.version is None
    current = _card(timestamp=400.0)                                  # the same data again: A3 goes into the card
    kept, dropped = ch.collapse([a1, a2, a3], current)
    assert (kept, dropped) == ([a2], [a1, a3]) and ch.start(current) == (300.0, None)


def test_dedupe_cleans_every_name_once():
    stats = {"A": _card(timestamp=300.0), "B": _card("B", ability="B2.", timestamp=300.0)}
    old = defaultdict(list, {"A": [_card(timestamp=100.0), _card(timestamp=200.0)],
                             "B": [_card("B", ability="B1.", timestamp=100.0), _card("B", ability="B1.", timestamp=200.0)],
                             "Gone": [_card("Gone", timestamp=50.0)]})
    assert ch.dedupe(stats, old) == [("A", 100.0), ("A", 200.0), ("B", 200.0)]
    assert "A" not in old and [v.timestamp for v in old["B"]] == [100.0] and len(old["Gone"]) == 1
    assert ch.start(stats["A"]) == (100.0, None) and stats["B"].version is None
    assert ch.dedupe(stats, old) == []


def test_dedupe_keeps_a_name_record_cant_read(capsys):
    """A value that isn't JSON data (record() raises) leaves its name's versions alone; the others are still cleaned."""
    stats = {"A": _card(timestamp=300.0), "B": _card("B", timestamp=300.0)}
    odd = _card("B", timestamp=200.0, parsed_values={"series": {"x"}})            # a set: not JSON data
    old = defaultdict(list, {"A": [_card(timestamp=100.0)], "B": [_card("B", timestamp=100.0), odd]})
    assert ch.dedupe(stats, old) == [("A", 100.0)]
    assert old["B"][1] is odd and len(old["B"]) == 2 and stats["B"].version is None
    assert "B compared as changed" in capsys.readouterr().out


def test_the_seed_after_the_cleanup_is_the_seed_alone():
    """The same history either way: the seed pass on pickles with duplicates, or /update_stats' cleanup first and then the
    seed (its legacy record dated from git, the earlier start kept; only that start's commit is unknown)."""
    index = _index_with({"b1": ("c1", 1000.0), "b2": ("c2", 2000.0), "b3": ("c3", 3000.0)})
    def pickles():
        old = defaultdict(list, {"A": [_card(timestamp=500.0), _card(timestamp=600.0),
                                       _card(ability="New.", blob_sha="b2", timestamp=2000.0)]})
        return {"A": _card(ability="New.", blob_sha="b3", timestamp=3000.0)}, old
    alone, cleaned = pickles(), pickles()
    ch.refine(*alone, index)
    assert len(ch.dedupe(*cleaned)) == 2
    assert cleaned[0]["A"].version["cause"] == "legacy" and cleaned[0]["A"].version["from"] == 2000.0
    ch.refine(*cleaned, index)
    for stats, old in (alone, cleaned):
        assert [(ch.data_of(v), v.version["from"], v.version["to"]) for v in old["A"]] == [(ch.data_of(_card()), 500.0, 2000.0)]
        assert (stats["A"].version["from"], stats["A"].version["cause"]) == (2000.0, "psd")
    assert (alone[0]["A"].version["from_commit"], cleaned[0]["A"].version["from_commit"]) == ("c2", None)
    cleaned[0]["A"].version["from_commit"] = "c2"
    assert cdb.record(alone[0]["A"]) == cdb.record(cleaned[0]["A"])


def test_duplicates_follow_their_original_and_get_no_version():
    index = _index_with({"b1": ("c1", 1000.0)})
    stats = {"A": _card(blob_sha="b1", timestamp=5.0), "A2": _card("A2", blob_sha="b1", timestamp=5.0, duplicate_of="A")}
    report = ch.refine(stats, defaultdict(list), index)
    assert stats["A2"].timestamp == 1000.0 and report.duplicates_dated == 1 and stats["A2"].version is None


# ========================
# The cleanup's rules (2026-10-23: the dupes left after the first collapse)
# ========================
V2 = dict(parser_version=2)


def test_a_title_before_parser_2_is_unknown():
    """Mol D Truffy: a reading before parser 2 read no title (None); the card read later with its printed title is no
    new version. Between two readings that knew it, the title is still data."""
    early = _card(timestamp=100.0)
    current = _card(timestamp=200.0, display_name="Mol D. Truffy", blob_sha="b2", **V2)
    assert ch.same_data(early, current)
    assert not ch.same_data(_card(display_name="X", **V2), _card(display_name="Y", **V2))
    assert ch.collapse([early], current) == ([], [early]) and ch.start(current) == (100.0, None)


def test_a_run_never_joins_two_known_titles():
    """A title-less early reading matches either title, but X and Y never share a run: the step X -> Y stays."""
    early = _card(timestamp=100.0)
    x = _card(timestamp=200.0, display_name="X", blob_sha="b2", **V2)
    y = _card(timestamp=300.0, display_name="Y", blob_sha="b3", **V2)
    current = _card(ability="New.", timestamp=400.0, display_name="Y", blob_sha="b4", **V2)
    kept, dropped = ch.collapse([early, x, y], current)
    assert (kept, dropped) == ([x, y], [early]) and ch.start(x) == (100.0, None) and ch.start(y) == (300.0, None)


def test_a_near_equal_run_keeps_the_reading_that_knew_most():
    """Wokey D pronouns: [no title, titled] then a real change: the titled reading stays, with the earliest start."""
    early = _card(timestamp=100.0)
    titled = _card(timestamp=200.0, display_name="Wokey D. Pronouns", blob_sha="b2", **V2)
    current = _card(ability="New.", timestamp=300.0, display_name="Wokey D. Pronouns", blob_sha="b3", **V2)
    assert ch.collapse([early, titled], current) == ([titled], [early]) and ch.start(titled) == (100.0, None)
    assert ch.collapse([titled], current) == ([titled], [])                                  # idempotent


def test_a_run_holds_readings_that_each_knew_less_than_one_of_them():
    """A title-less early reading, the full one, then a reading without its stars: the two that knew less differ in
    what they missed (neither stands for the other), but the full reading knew all of both, so all three are one run
    and it alone stays, from the earliest start. The card after it (other stars) agrees with the starless reading
    too, but that reading belongs to the run: the card does not start at it."""
    early = _card(timestamp=100.0, stars=1)
    full = _card(timestamp=200.0, stars=1, display_name="T", blob_sha="b2", **V2)
    starless = _card(timestamp=300.0, stars=0, display_name="T", blob_sha="b3", problems=["NO STARS FOUND"], **V2)
    current = _card(timestamp=400.0, stars=2, display_name="T", blob_sha="b4", **V2)
    assert not ch.same_data(early, starless) and ch.knows(full, early) and ch.knows(full, starless)
    assert ch.knows(current, starless) and not ch.knows(current, full)
    assert ch.collapse([early, full, starless], current) == ([full], [early, starless])
    assert ch.start(full) == (100.0, None) and ch.start(current) == (400.0, None) and current.version is None


def test_an_earlier_reading_of_the_same_commit_goes():
    """Tainted Bethany, omori, How To Jump: one commit read before parser 2 (no blob) and again by parser 2 (its blob),
    both kept at the same second. The earlier reading goes (nothing to carry), so the kept one has the blob and no
    version is zero-length. Two blobs at one second are two versions."""
    v0 = _card(timestamp=50.0)
    plain = _card(ability="Same.", timestamp=100.0)
    read = _card(ability="Same.", timestamp=100.0, blob_sha="b1", display_name="Shown", **V2)
    current = _card(ability="New.", timestamp=300.0, blob_sha="b3", **V2)
    assert ch.collapse([v0, plain, read], current) == ([v0, read], [plain]) and read.version is None
    before = _card(ability="Old.", timestamp=100.0)
    holder = _card(ability="Edited.", timestamp=100.0, blob_sha="b1", display_name="How to jump", **V2)
    assert ch.collapse([before], holder) == ([], [before]) and holder.version is None
    a, b = _card(timestamp=100.0, blob_sha="b1", **V2), _card(ability="B.", timestamp=100.0, blob_sha="b2", **V2)
    assert ch.collapse([a, b], current) == ([a, b], [])


def test_an_edit_older_than_the_card_it_changes_stays_a_version():
    """Card Edits.json's newest commit older than the card's start (an edits commit merged after a PSD push): the
    edit's version starts at the card's start (never before it), and the cleanup keeps the card before it (no re-read:
    its cause is edit), from record_edits (legacy) and a build's record_versions (authors too)."""
    old = _card(timestamp=1000.0, blob_sha="b1", **V2)
    db = _db({"A": _card(ability="Edited.", timestamp=1000.0, blob_sha="b1", **V2)})
    assert ch.record_edits(db, {"A": deepcopy(old)}, 900.0, "e1") == ["A"]
    assert ch.start(db.stats["A"])[0] == 1000.0 and ch.dedupe(db.stats, db.old_stats) == []
    assert [v.ability for v in db.old_stats["A"]] == ["Text."] and db.stats["A"].ability == "Edited."
    index = _index_with({"b1": ("c1", 1000.0)})
    old.version = ch.start_of(old, index, None)
    for after, cause in ((_card(ability="Edited.", blob_sha="b1", **V2), "edit"),
                         (_card(blob_sha="b1", author="New", **V2), "authors")):
        db = _db({"A": after})
        assert ch.record_versions({"A": deepcopy(old)}, {}, db, {}, [], index, ch.Commit(0, "e1", 900.0, "web"),
                                  ch.Commit(0, "a1", 900.0, "web")) == [("A", cause)]
        assert ch.start(after)[0] == 1000.0 and ch.dedupe(db.stats, db.old_stats) == [] and len(db.old_stats["A"]) == 1


def test_a_field_its_parse_failed_on_is_unknown():
    """Bubble and God Tamer (NO STARS FOUND), 20 Creature Types (MISSPELT TYPE): the later reading that found them is
    the same card, which keeps the earlier start."""
    bubble = _card(timestamp=100.0, stars=0, blob_sha="b1", problems=['NO STARS FOUND (put the star layers in a group '
                                                                       'named "Stars")'], **V2)
    current = _card(timestamp=300.0, stars=1, blob_sha="b3", **V2)
    assert ch.collapse([bubble], current) == ([], [bubble]) and ch.start(current) == (100.0, None)
    misspelt = _card(types=["ice"], problems=["MISSPELT TYPE: tornado"], **V2)
    assert ch.same_data(misspelt, _card(types=["ice", "wind"], **V2))
    assert not ch.same_data(_card(types=["ice"], **V2), _card(types=["ice", "wind"], **V2))


def test_one_form_for_the_same_data():
    """An empty ability is None or '' (Anubisath Guardian); a title equal to the name is no title (a Card Edits name
    fix equal to the file name); a retired type icon is its name now (offense.png became Negation.png)."""
    assert ch.same_data(_card(ability=None), _card(ability=""))
    named = _card(display_name="A", **V2)
    assert ch.same_data(named, _card(**V2)) and ch.data_of(named)["display_name"] is None
    assert ch.same_data(_card(types=["offense"], **V2), _card(types=["negation"], **V2))


NO_LAYER = ["NO ABILITY LAYER"]


def test_a_reading_that_knew_less_never_stands_for_one_that_knew_more():
    """A commit with a broken ability layer (U) knew less than X: U never absorbs X (as the card after it, a run's
    member or an archived record), while the reading that knew more absorbs U. Two readings each blind to a field the
    other knew never join."""
    x = _card(ability="Text X.", timestamp=100.0, blob_sha="b1", **V2)
    u = _card(ability=None, timestamp=200.0, blob_sha="b2", problems=NO_LAYER, **V2)
    assert ch.knows(x, u) and not ch.knows(u, x) and ch.same_data(x, u)
    assert ch.collapse([x], u) == ([x], []) and ch.start(u) == (200.0, None)                  # U holds nothing of X
    y = _card(ability="Text Y.", timestamp=300.0, blob_sha="b3", **V2)
    assert ch.collapse([x, u], y) == ([x], [u]) and ch.start(x) == (100.0, None)              # U goes into X
    assert ch.collapse([deepcopy(u)], y)[0] == [] and ch.start(y) == (200.0, None)            # into the card after it
    gone = _card(ability=None, timestamp=200.0, blob_sha="b2", problems=NO_LAYER,
                 archived={"at": 250.0, "path": "Items/A.psd", "blob": "b2", "commit": None}, **V2)
    x2 = _card(ability="Text X.", timestamp=100.0, blob_sha="b1", **V2)
    assert ch.collapse([x2, gone], None) == ([x2, gone], [])
    stars = _card(stars=0, timestamp=100.0, blob_sha="b1", problems=["NO STARS FOUND"], **V2)
    ability = _card(ability=None, stars=3, timestamp=200.0, blob_sha="b2", problems=NO_LAYER, **V2)
    assert not ch.same_data(stars, ability) and ch.collapse([stars, ability], y) == ([stars, ability], [])
    early, read = _card(timestamp=100.0), _card(timestamp=200.0, blob_sha="b2", **V2)       # equal, early read no title
    assert ch.collapse([early, read], y) == ([read], [early]) and ch.start(read) == (100.0, None)


def test_legacy_history_keeps_the_card_a_broken_layer_replaced():
    """Legacy /update_stats: X, then a commit without its ability layer (U), then Y: X stays a version, U goes into Y
    (the reading that knew more), and the next cleanup changes nothing. History [X], card Y, then U: Y is kept."""
    from commands.card_data import CardBuild
    x = _card(ability="Old text X.", timestamp=1000.0, blob_sha="a" * 40, **V2)
    u = _card(ability=None, timestamp=2000.0, blob_sha="b" * 40, problems=NO_LAYER, **V2)
    y = _card(ability="New text Y.", timestamp=3000.0, blob_sha="c" * 40, **V2)
    build = CardBuild.__new__(CardBuild)
    build.db, build.legacy = _db({}), True
    for old, new in ((x, u), (u, y)):
        build.db.stats["A"], build.replaced = deepcopy(new), {"A": deepcopy(old)}
        build._record_history()
    assert [v.ability for v in build.db.old_stats["A"]] == ["Old text X."]
    assert build.db.stats["A"].ability == "New text Y." and ch.start(build.db.stats["A"]) == (2000.0, None)
    assert ch.dedupe(build.db.stats, build.db.old_stats) == []
    build.db = _db({"A": deepcopy(u)}, {"A": [deepcopy(x)]})
    build.replaced = {"A": deepcopy(y)}
    build._record_history()
    assert [v.ability for v in build.db.old_stats["A"]] == ["Old text X.", "New text Y."]
    assert ch.dedupe(build.db.stats, build.db.old_stats) == [] and ch.start(build.db.stats["A"]) == (2000.0, None)


def test_build_and_untag_keep_the_card_a_reading_that_knew_less_replaced():
    """The data build: a new blob, or the same blob read again, without its ability layer keeps the card before as a
    version. A stale archived record the listed card knew less than stays history (untagged)."""
    index = _index_with({"b1": ("c1", 1000.0), "b2": ("c2", 2000.0)})
    for blob_sha in ("b2", "b1"):
        old = _card(blob_sha="b1", timestamp=1000.0, **V2)
        old.version = ch.start_of(old, index, None)
        db = _db({"A": _card(ability=None, blob_sha=blob_sha, timestamp=2000.0, problems=NO_LAYER, **V2)})
        assert ch.record_versions({"A": old}, {}, db, {"A": 1}, [], index) == [("A", "psd" if blob_sha == "b2" else "reparse")]
        assert [v.ability for v in db.old_stats["A"]] == ["Text."]
        assert ch.dedupe(db.stats, db.old_stats) == []
    record = _card(blob_sha="b1", timestamp=100.0, archived={"at": 150.0, "path": "Items/A.psd", "blob": "b1"}, **V2)
    db = _db({"A": _card(ability=None, path="Items/Heal/A.psd", blob_sha="b2", timestamp=500.0, problems=NO_LAYER,
                         **V2)}, {"A": [record]})
    assert db.drop_stale_archives() and db.old_stats["A"] == [record] and record.archived is None


def _creature(name, series, subseries=None, ability="Old.", timestamp=100.0):
    return cm.CardInfo(name=name, card_type="creature", path=f"Creatures/Mario/2 Stars/{name}.psd", ability=ability,
                       series=series, subseries=subseries, timestamp=timestamp)


def test_normalize_series_and_type_names():
    """Spicy Spray: an item's series typed in after its old version had none is that version's too, so it folds into
    the card. Beldam: the 2026-10-20 split moved its series under 'Mario', no change. A creature's series is its
    folder's, never filled. The retired 'offense' icon is 'negation'. Idempotent."""
    spray, metronome = _card("Spicy Spray", timestamp=100.0), _card("Metronome", types=["offense"], timestamp=100.0)
    beldam, goomba = _creature("Beldam", "Paper Mario: The Thousand-Year Door"), _creature("Goomba", None)
    stats = {"Spicy Spray": _card("Spicy Spray", timestamp=200.0, series="Pikmin 2", blob_sha="b2"),
             "Metronome": _card("Metronome", types=["negation"], timestamp=300.0),
             "Beldam": _creature("Beldam", "Mario", "Paper Mario: The Thousand-Year Door", "New.", 300.0),
             "Goomba": _creature("Goomba", "Mario", None, "New.", 300.0)}
    old = defaultdict(list, {"Spicy Spray": [spray], "Metronome": [metronome], "Beldam": [beldam], "Goomba": [goomba]})
    assert sorted(n for n, _ in ch.normalize(stats, old)) == ["Beldam", "Metronome", "Spicy Spray"]
    assert spray.series == "Pikmin 2" and metronome.types == ["negation"] and goomba.series is None
    assert (beldam.series, beldam.subseries) == ("Mario", "Paper Mario: The Thousand-Year Door")
    assert ch.normalize(stats, old) == []
    assert sorted(ch.dedupe(stats, old)) == [("Metronome", 100.0), ("Spicy Spray", 100.0)]
    assert list(old) == ["Beldam", "Goomba"] and ch.start(stats["Spicy Spray"]) == (100.0, None)


def test_versions_keep_no_creation_date():
    """The card's creation date stays on the card (and an archived record): normalize drops a version's copy, and
    add_history and a build's history copy none."""
    version = _card(timestamp=100.0)
    gone = _card("Gone", timestamp=100.0, archived={"at": 150.0, "path": "Items/Gone.psd"})
    version.created = gone.created = {"date": "2026-01-01", "from": "file"}
    old = defaultdict(list, {"A": [version], "Gone": [gone]})
    assert ch.normalize({"A": _card(ability="New.", timestamp=200.0)}, old) == [("A", 100.0)]
    assert "created" not in vars(version) and gone.created
    db = cm.StatsDatabase()
    card = _card()
    card.created = {"date": "2026-01-01", "from": "file"}
    db.add_history("A", card)
    assert "created" not in vars(db.old_stats["A"][0])
    index = _index_with({"b1": ("c1", 1000.0), "b2": ("c2", 2000.0)})
    before = _card(blob_sha="b1")
    before.created = {"date": "2026-01-01", "from": "file"}
    db = _db({"A": _card(ability="Changed.", blob_sha="b2")})
    ch.record_versions({"A": before}, {}, db, {"A": 1}, [], index)
    assert "created" not in vars(db.old_stats["A"][0])


def test_the_cleanup_on_a_post_deploy_chain():
    """Bob-ulk's chain as the jetson holds it: v2 read before parser 2 and v3 read by parser 2 at one commit, the series
    split, the title: the cleanup leaves [v0, v1] and the card, which starts at that commit; a second pass changes
    nothing."""
    path = "Creatures/Mario/2 Stars/Bob-ulk.psd"
    ttyd = "Paper Mario: The Thousand-Year Door"

    def bob(ability, at, **kw):
        return cm.CardInfo(name="Bob-ulk", card_type="creature", path=path, ability=ability, timestamp=at, **kw)
    v0, v1 = bob("Cannot attack.", 100.0, series=ttyd), bob("Cannot attack, defend.", 200.0, series=ttyd)
    v2 = bob("Has no innate actions.", 300.0, series=ttyd)
    v3 = bob("Has no innate actions.", 300.0, series="Mario", subseries=ttyd, display_name="Bob-Ulk", blob_sha="b3",
             **V2)
    current = bob("Has no innate actions.", 400.0, series="Mario", subseries=ttyd, display_name="Bob-Ulk",
                  blob_sha="b4", **V2)
    stats, old = {"Bob-ulk": current}, defaultdict(list, {"Bob-ulk": [v0, v1, v2, v3]})
    ch.normalize(stats, old)
    assert ch.dedupe(stats, old) == [("Bob-ulk", 300.0), ("Bob-ulk", 300.0)]
    assert old["Bob-ulk"] == [v0, v1] and ch.start(current) == (300.0, None)
    assert (v0.series, v0.subseries) == ("Mario", ttyd)
    assert ch.normalize(stats, old) == [] and ch.dedupe(stats, old) == []


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



def test_build_versions_never_end_before_they_start():
    """A PSD change read after a later edit (held by the budget, or a re-read): the new version starts no earlier
    than the one it ends."""
    index = _index_with({"b1": ("c1", 1000.0), "b2": ("c2", 2000.0)})
    old = _card(blob_sha="b1", note="checked")
    old.version = ch.meta(ch.Commit(0, "e1", 3000.0, "web"), 3000.0, "edit")
    db = _db({"A": _card(ability="Changed.", blob_sha="b2", note="checked")})
    ch.record_versions({"A": old}, {}, db, {"A": 1}, [], index)
    previous, current = db.old_stats["A"][0].version, db.stats["A"].version
    assert previous["from"] <= previous["to"] == current["from"] == 3000.0 and current["cause"] == "psd"


def test_build_reread_of_the_same_blob_is_no_version():
    """A parser change read on the same PSD (same blob and path): the reading is replaced, its start kept, no
    zero-length 'reparse' version (the changelog still lists the re-read)."""
    index = _index_with({"b1": ("c1", 1000.0)})
    old = _card(blob_sha="b1", timestamp=1000.0)
    old.version = ch.start_of(old, index, None)
    old.version["from"] = 500.0                                                     # a carried start
    db = _db({"A": _card(ability="Read better.", blob_sha="b1", timestamp=1000.0, **V2)})
    assert ch.record_versions({"A": old}, {}, db, {"A": 1}, [], index) == [("A", "reparse")]
    assert "A" not in db.old_stats and (db.stats["A"].version["from"], db.stats["A"].version["to"]) == (500.0, None)


def test_restored_with_the_same_data_at_another_path_is_no_version():
    """An archived card back at another path with the same data: its archived record goes, its version goes on."""
    index = _index_with({"b1": ("c1", 1000.0)})
    record = _card(blob_sha="b1", archived={"at": 1500.0, "path": "Items/A.psd", "blob": "b1", "commit": None})
    record.version = ch.start_of(record, index, None)
    ch.close(record, index, 1500.0)
    db = _db({"A": _card(path="Items/Heal/A.psd", blob_sha="b1")}, {"A": [record]})
    assert ch.record_versions({}, {"A": record}, db, {"A": 1}, ["A"], index) == []
    db.drop_stale_archives()
    assert "A" not in db.old_stats and (db.stats["A"].version["from"], db.stats["A"].version["to"]) == (1000.0, None)


def test_a_stale_archive_with_the_same_data_gives_the_card_its_start():
    """restore() declined (back at another path with another blob) but the data is the same: the tagged record goes,
    its earlier start to the listed card (as_of before the re-add still finds it)."""
    record = _card(blob_sha="b1", timestamp=100.0, archived={"at": 150.0, "path": "Items/A.psd", "blob": "b1"})
    listed = _card(path="Items/Heal/A.psd", blob_sha="b2", timestamp=500.0)
    db = _db({"A": listed}, {"A": [record]})
    assert db.drop_stale_archives() and "A" not in db.old_stats
    assert ch.start(listed) == (100.0, None) and listed.version["cause"] == "legacy" and not listed.version["legacy"]



def _stale_pair():
    record = _card(blob_sha="b1", timestamp=100.0, archived={"at": 150.0, "path": "Items/A.psd", "blob": "b1"})
    return _db({"A": _card(path="Items/Heal/A.psd", blob_sha="b2", timestamp=500.0)}, {"A": [record]})


def _reloaded():
    db = cm.StatsDatabase()
    db.load()
    return db


def test_a_carried_start_reaches_stats_pkl():
    """/update_stats saves (drop after stats.pkl): the carried start is on disk, the record gone, nothing stale."""
    _stale_pair().save()
    db = _reloaded()
    assert ch.start(db.stats["A"]) == (100.0, None) and "A" not in db.old_stats and db.stale_archives() == []


def test_a_kill_after_the_carried_start_keeps_the_record(monkeypatch):
    """Killed before old_stats.pkl's second write: the record is still there, and the next save drops it again."""
    real, calls = cm.save_pickle, []

    def kill_fourth(obj, path, lock):
        calls.append(path)
        if len(calls) == 4:
            raise KeyboardInterrupt
        real(obj, path, lock)
    monkeypatch.setattr(cm, "save_pickle", kill_fourth)
    with pytest.raises(KeyboardInterrupt):
        _stale_pair().save()
    monkeypatch.setattr(cm, "save_pickle", real)
    db = _reloaded()
    assert ch.start(db.stats["A"]) == (100.0, None) and db.stale_archives() == ["A"]
    db.save()
    db = _reloaded()
    assert ch.start(db.stats["A"]) == (100.0, None) and "A" not in db.old_stats

# ========================
# Recovery from the bot's PICKLE commits, misreads, impossible values
# ========================
A1, A2, A3, G1, O1 = b"a one", b"a two", b"a three", b"gone one", b"old one"


@pytest.fixture
def lost(tmp_path):
    """A card repo: A edited at 1000, 2000, 3000; Gone and Old added at 4000; at 5000 Gone deleted and Old renamed to
    New (same bytes). The bot's repo: PICKLEs at 1500 (an April-2025 dict keyed by path, an mtime), 2500 and 4500
    (CardInfo), a bad one at 4600 that 4700 reverts. The database lost every version: A is listed with its 3000
    reading (dated 3000, no record), New with Old's blob."""
    cards = tmp_path / "cards"
    cards.mkdir()
    git(cards, "init", "-q", "-b", "main")
    c = [None]
    for at, data in ((1000, A1), (2000, A2), (3000, A3)):
        write(cards, "Items/A.psd", data)
        write(cards, "Items/A.png", b"png %d" % at)
        c.append(commit(cards, "alice", at))
    write(cards, "Items/Gone.psd", G1)
    write(cards, "Items/Old.psd", O1)
    c.append(commit(cards, "bob", 4000))
    git(cards, "rm", "-q", "Items/Gone.psd")
    git(cards, "mv", "Items/Old.psd", "Items/New.psd")
    c.append(commit(cards, "carol", 5000))
    bot = tmp_path / "bot"
    bot.mkdir()
    git(bot, "init", "-q", "-b", "main")

    def pickles(at, stats, message="PICKLE"):
        cdb.dump_pickle(stats, str(bot / "stats.pkl"))
        cdb.dump_pickle(defaultdict(list), str(bot / "old_stats.pkl"))
        return commit(bot, "bot", at, message)
    pickles(1500, {"Items/A.psd": {"type": "item", "ability": "Old words.", "types": [], "name": "A", "timestamp": 1400.25}})
    pickles(2500, {"A": _card(ability="New words.", timestamp=2000.0)})
    held = {"A": _card(ability="New  words.", timestamp=3000.0),
            "Gone": _card("Gone", path="Items/Gone.psd", ability="Gone words.", timestamp=4000.0),
            "Old": _card("Old", path="Items/Old.psd", ability="Old card.", timestamp=4000.0, author="maker")}
    pickles(4500, held)
    bad = pickles(4600, {**held, "A": _card(ability="Bad.", timestamp=3000.0)})
    pickles(4700, held, f'Revert "PICKLE"\n\nThis reverts commit {bad}.')
    stats = {"A": _card(ability="New words.", blob_sha=blob(A3), timestamp=3000.0),
             "New": _card("New", path="Items/New.psd", ability="Old card.", blob_sha=blob(O1), timestamp=5000.0)}
    return SimpleNamespace(cards=cards, bot=bot, c=c, stats=stats, old=defaultdict(list))


def test_bot_snapshots_skip_reverted_and_non_clones(lost, tmp_path):
    snaps = ch.bot_snapshots(str(lost.bot))
    assert [s.at for s in snaps] == [1500.0, 2500.0, 4500.0, 4700.0]                    # 4600 reverted by 4700
    assert "Items/A.psd" in snaps[0].stats and snaps[2].stats["Gone"].ability == "Gone words."
    assert ch.bot_snapshots(str(tmp_path / "nothing")) == [] and ch.bot_snapshots("") == []
    subprocess.run(["git", "clone", "-q", "--depth", "1", f"file://{lost.bot}", str(tmp_path / "shallow")], check=True)
    assert ch.bot_snapshots(str(tmp_path / "shallow")) == []                              # a shallow clone: no history


def test_recover_restores_lost_versions_starts_and_gone_cards(lost):
    """A's April reading (an mtime: dated by its blob's commit) becomes a legacy version; its 2000 reading prints what
    the card prints (spaces aside): the card's start moves there; Gone is archived at its deletion, Old too, renamed to
    New (date_archive), whose author stays maker's. Nothing changes the second time."""
    index = ch.GitIndex.from_git(str(lost.cards))
    snaps = ch.bot_snapshots(str(lost.bot))
    pickles = ch.bot_pickles(str(lost.bot))
    assert ch.bot_pickles(str(lost.bot)) is pickles and set(pickles.at_of) == {s.sha for s in snaps}   # read once
    stats, old = lost.stats, lost.old
    report = ch.recover(stats, old, index, pickles)
    first = old["A"][0]
    assert [v.ability for v in old["A"]] == ["Old words."] and first.blob_sha == blob(A1)
    assert (first.timestamp, first.version["from"], first.version["from_commit"]) == (1000.0, 1000.0, lost.c[1])
    assert first.version["restored_from"] == snaps[0].sha and first.version["cause"] == "legacy"
    assert (stats["A"].version["from"], stats["A"].version["from_commit"]) == (2000.0, lost.c[2])
    gone = old["Gone"][0]
    assert gone.archived == {"at": 5000.0, "path": "Items/Gone.psd", "blob": blob(G1), "commit": lost.c[5], "backfill": True}
    assert sorted(report.archived) == ["Gone", "Old"] and report.starts == [("A", 2000.0)]
    assert len(report.added) == 3 and not any(v.ability == "Bad." for vs in old.values() for v in vs)
    found = ch.Refined()
    ch.date_archive(stats, old, index, found, backfill=False)
    assert old["Old"][0].archived["renamed_to"] == "New" and old["Old"][0].author == "maker"
    again = ch.recover(stats, old, index, ch.Pickles.of(snaps))
    assert not again.changed()
    report = ch.refine(stats, old, index, pickles)                # the seed pass after it: nothing more to recover
    assert report.recovered is not None and not report.recovered.changed()
    assert old["A"][0].version["restored_from"] == snaps[0].sha   # the seed keeps where it came from


def test_seed_pass_recovers_from_the_bot_clone(lost, tmp_path, capsys):
    folder, out = tmp_path / "pickles", tmp_path / "out"
    folder.mkdir()
    cdb.dump_pickle(lost.stats, str(folder / "stats.pkl"))
    cdb.dump_pickle(lost.old, str(folder / "old_stats.pkl"))
    assert ch.main(["refine", "--git", str(lost.cards), "--pickles", str(folder), "--out", str(out),
                    "--bot-git", str(lost.bot)]) == 0
    text = capsys.readouterr().out
    assert "Versions recovered from the bot's PICKLE commits: 3 on 3 cards" in text
    assert "Cards gone before archiving, archived: 2 (Gone, Old)" in text or "archived: 2 (Old, Gone)" in text
    assert pickle.loads((out / "old_stats.pkl").read_bytes())["A"][0].ability == "Old words."


def _misread_repo(tmp_path, png_changes: bool):
    repo = tmp_path / "misread"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    write(repo, "Items/B.psd", b"b one")
    write(repo, "Items/B.png", b"b png")
    c1 = commit(repo, "a", 100000)
    write(repo, "Items/B.psd", b"b two")
    if png_changes:
        write(repo, "Items/B.png", b"b png two")
    c2 = commit(repo, "a", 200000)
    return ch.GitIndex.from_git(str(repo)), c1, c2


@pytest.mark.parametrize("png_changes", [False, True])
def test_a_types_only_step_with_the_same_png_is_a_misread(tmp_path, png_changes):
    """Wooden Spikes: the PSD changed, the PNG did not, only the type list differs: one version read twice, merged into
    the newer with the earlier start. A new PNG makes it a real change."""
    index, c1, c2 = _misread_repo(tmp_path, png_changes)
    v0 = _card("B", path="Items/B.psd", blob_sha=blob(b"b one"), timestamp=100000.0)
    card = _card("B", path="Items/B.psd", blob_sha=blob(b"b two"), timestamp=200000.0, types=["counter"])
    stats, old = {"B": card}, defaultdict(list, {"B": [v0]})
    merged = ch.misreads(stats, old, index)
    if png_changes:
        assert merged == [] and old["B"] == [v0]
        return
    assert merged == [("B", 100000.0, [], ["counter"])] and "B" not in old and ch.start(card)[0] == 100000.0
    assert ch.misreads(stats, old, index) == []


def test_a_misread_is_judged_at_the_newer_versions_start(tmp_path):
    """The newer reading folded into the card (its start carried, its own timestamp a later re-export's): the PNG is
    compared where the newer data began, so the cleanups' order does not change the result."""
    _misread_repo(tmp_path, False)
    repo = tmp_path / "misread"
    write(repo, "Items/B.png", b"b png re-exported")
    commit(repo, "a", 800000)
    index = ch.GitIndex.from_git(str(repo))
    v0 = _card("B", path="Items/B.psd", blob_sha=blob(b"b one"), timestamp=100000.0)
    card = _card("B", path="Items/B.psd", blob_sha="9" * 40, timestamp=900000.0, types=["counter"])
    card.version = ch.meta(None, 200000.0, "legacy")
    stats, old = {"B": card}, defaultdict(list, {"B": [v0]})
    assert [n for n, *_ in ch.misreads(stats, old, index)] == ["B"] and ch.start(card)[0] == 100000.0


def test_a_misread_after_a_rename_reads_the_png_at_its_own_path(tmp_path):
    """You are going DOWN you Sputnik!: the older version's date is its bytes' first commit under the old name (the
    seed dates so); the PNG beside it is its own path's from when the rename brought the bytes there."""
    repo = tmp_path / "renamed"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    write(repo, "Items/Old.psd", b"b one")
    write(repo, "Items/Old.png", b"b png")
    commit(repo, "a", 100000)
    git(repo, "mv", "Items/Old.psd", "Items/B.psd")
    git(repo, "mv", "Items/Old.png", "Items/B.png")
    commit(repo, "a", 150000)
    write(repo, "Items/B.psd", b"b two")
    commit(repo, "a", 300000)
    index = ch.GitIndex.from_git(str(repo))
    v0 = _card("B", path="Items/B.psd", blob_sha=blob(b"b one"), timestamp=100000.0)
    card = _card("B", path="Items/B.psd", blob_sha=blob(b"b two"), timestamp=300000.0, types=["counter"])
    assert [n for n, *_ in ch.misreads({"B": card}, defaultdict(list, {"B": [v0]}), index)] == ["B"]


def test_impossible_values_become_unknown_once():
    lazarus = _card("L", stats=cm.CardStats(110, 110, 110, 110), stars=3, timestamp=1.0)
    spawn = _card("S", stars=10, timestamp=2.0)
    cannon = _card("C", stats=cm.CardStats(9, 8, 12, 3), timestamp=3.0)
    stats = {"C": _card("C", stats=cm.CardStats(9, 8, 12, 3), allow_high_stats=True)}
    old = defaultdict(list, {"L": [lazarus], "S": [spawn], "C": [cannon]})
    changed = ch.unknown_values(stats, old)
    assert [(n, f) for n, _, f in changed] == [("L", ["stats"]), ("S", ["stars"])]
    assert lazarus.stats == cm.CardStats() and spawn.stars is None and cannon.stats.attack == 12   # its card allows it
    assert lazarus.problems == ["OUT OF RANGE: stats 110/110/110/110 (unknown)"]
    assert ch.unknown_values(stats, old) == []
    assert "Impossible values set to unknown: 0" in ch.Refined().lines()[10]


def test_a_rename_chain_reaches_the_listed_card():
    """Azamoth the Possessed King: renamed twice; the tag names the card listed now."""
    index = _index_with({}, deletions={"Items/K.psd": ("d1", 4000.0)})
    index.renames["Items/K.psd"] = ("Items/K2.psd", index.by_sha["d1"])
    index.renames["Items/K2.psd"] = ("Items/V.psd", index.by_sha["d1"])
    king = _card("K", path="Items/K.psd", archived={"at": None, "path": "Items/K.psd", "blob": None, "commit": None})
    report = ch.Refined()
    ch.date_archive({"V": _card("V")}, defaultdict(list, {"K": [king]}), index, report)
    assert king.archived["renamed_to"] == "V" and report.renamed == [("K", "V")]


def _bot_repo(tmp_path, snapshots: list):
    """The bot's repo: one PICKLE commit per (at, stats) of snapshots."""
    bot = tmp_path / "bot"
    bot.mkdir()
    git(bot, "init", "-q", "-b", "main")
    shas = []
    for at, stats in snapshots:
        cdb.dump_pickle(stats, str(bot / "stats.pkl"))
        cdb.dump_pickle(defaultdict(list), str(bot / "old_stats.pkl"))
        shas.append(commit(bot, "bot", at, "PICKLE"))
    return bot, shas


def _cards_repo(tmp_path, steps: list):
    """A card repo: one commit per (at, {path: bytes, or None to delete, or ("mv", new path)})."""
    repo = tmp_path / "cards"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    shas = []
    for at, files in steps:
        for path, data in files.items():
            if data is None:
                git(repo, "rm", "-q", path)
            elif isinstance(data, tuple):
                (repo / data[1]).parent.mkdir(parents=True, exist_ok=True)
                git(repo, "mv", path, data[1])
            else:
                write(repo, path, data)
        shas.append(commit(repo, "a", at))
    return ch.GitIndex.from_git(str(repo)), shas


def test_a_name_at_two_paths_in_one_pickle_keeps_its_own(tmp_path):
    """Auto Croissant: a 2025 PICKLE keyed by path held the minion Auxiliary/Minions/Auto.psd and the creature
    Creatures/Other/2 Stars/Auto.psd, both named Auto; the creature's history takes only its own path's readings."""
    minion, creature = "Auxiliary/Minions/Auto.psd", "Creatures/Other/2 Stars/Auto.psd"
    index, c = _cards_repo(tmp_path, [(1000, {minion: b"m one", creature: b"c one"}), (2000, {creature: b"c two"})])
    bot, _ = _bot_repo(tmp_path, [(1500, {minion: {"type": "minion", "ability": "Minion words.", "timestamp": 1400.0},
                                          creature: {"type": "creature", "ability": "Creature words.", "timestamp": 1400.0}})])
    card = cm.CardInfo(name="Auto", card_type="creature", path=creature, ability="New words.", blob_sha=blob(b"c two"),
                       timestamp=2000.0)
    stats, old = {"Auto": card}, defaultdict(list)
    report = ch.recover(stats, old, index, ch.Pickles.of(ch.bot_snapshots(str(bot))))
    assert [(v.path, v.ability) for v in old["Auto"]] == [(creature, "Creature words.")] and report.clashed == 1
    assert not ch.recover(stats, old, index, ch.Pickles.of(ch.bot_snapshots(str(bot)))).changed()


def test_readings_of_one_blob_in_two_folders_are_two_versions_dated_at_their_own_path(tmp_path):
    """Air Totem: the same bytes as a minion (Auxiliary/Minions) and, moved, an item (Auxiliary/Items): two versions,
    each its own card type, stars and path, the item from the move. Stone Spikes: a 2025 dict reading (an mtime) of
    bytes first committed in another folder starts when they came to its own path (a reading never stands for a time
    its file sat elsewhere)."""
    minion, item, offense, counter = ("Auxiliary/Minions/T.psd", "Auxiliary/Items/T.psd", "Items/Offense/1 Stars/S.psd",
                                      "Items/Counter/1 Stars/S.psd")
    index, c = _cards_repo(tmp_path, [(1000, {minion: b"t one", offense: b"s one"}), (2000, {minion: ("mv", item)}),
                                      (2100, {offense: ("mv", counter)}), (3000, {item: b"t two", counter: b"s two"})])
    bot, _ = _bot_repo(tmp_path, [
        (1500, {"T": _card("T", path=minion, ability="Words.", timestamp=1000.0, card_type="minion", stars=2)}),
        (2500, {"T": _card("T", path=item, ability="Words.", timestamp=2000.0, card_type="aux item"),
                counter: {"type": "item", "ability": "Spikes.", "timestamp": 2200.0}})])
    stats = {"T": _card("T", path=item, ability="New.", blob_sha=blob(b"t two"), timestamp=3000.0, card_type="aux item"),
             "S": _card("S", path=counter, ability="New spikes.", blob_sha=blob(b"s two"), timestamp=3000.0)}
    old = defaultdict(list)
    ch.recover(stats, old, index, ch.Pickles.of(ch.bot_snapshots(str(bot))))
    assert [(v.card_type, v.path, v.stars, ch.start(v)) for v in old["T"]] == [
        ("minion", minion, 2, (1000.0, c[0])), ("aux item", item, None, (2000.0, c[1]))]
    assert [(v.path, ch.start(v), v.blob_sha) for v in old["S"]] == [(counter, (2100.0, c[2]), blob(b"s one"))]


def test_a_restored_version_pairs_its_start_with_that_commits_blob(tmp_path):
    """Two readings printing the same (spacing aside) from two commits are one version: the newer parse's data, the
    older start and that start's own blob (a pair git had)."""
    index, c = _cards_repo(tmp_path, [(1000, {"Items/A.psd": A1}), (2000, {"Items/A.psd": A2}), (3000, {"Items/A.psd": A3})])
    bot, shas = _bot_repo(tmp_path, [(1500, {"A": _card(ability="Old words.", timestamp=1000.0)}),
                                     (2500, {"A": _card(ability="Old  words.", timestamp=2000.0, parser_version=2)})])
    stats, old = {"A": _card(ability="New.", blob_sha=blob(A3), timestamp=3000.0)}, defaultdict(list)
    ch.recover(stats, old, index, ch.Pickles.of(ch.bot_snapshots(str(bot))))
    [version] = old["A"]
    assert (version.ability, version.blob_sha, ch.start(version), version.version["restored_from"]) == (
        "Old  words.", blob(A1), (1000.0, c[0]), shas[1])


def test_a_renamed_card_starts_where_its_old_names_archive_ends(tmp_path):
    """Blashemous Blade, renamed to Blasphemous Blade with the same bytes: the old name's archive holds the time
    before, so the seed (and a build) date the new name from the rename, not its bytes' first commit."""
    index, c = _cards_repo(tmp_path, [(1000, {"Items/Old.psd": O1}), (2000, {"Items/Old.psd": ("mv", "Items/New.psd")})])
    gone = _card("Old", path="Items/Old.psd", ability="Same.", blob_sha=blob(O1), timestamp=1000.0,
                 archived={"at": 2000.0, "path": "Items/Old.psd", "blob": blob(O1), "commit": c[1], "renamed_to": "New"})
    stats = {"New": _card("New", path="Items/New.psd", ability="Same.", blob_sha=blob(O1), timestamp=1000.0)}
    old = defaultdict(list, {"Old": [gone]})
    ch.refine(stats, old, index)
    assert (stats["New"].timestamp, ch.start(stats["New"])) == (2000.0, (2000.0, c[1]))
    assert ch.start_of(stats["New"], index, None)["from"] == 1000.0           # no other name: the bytes' first commit
    assert index.intro_of(blob(O1), "New", {"Old", "New"}).sha == c[1] and index.intro_of(blob(O1), "Old", {"New"}).sha == c[0]


def test_the_seed_after_the_bots_cleanup_reaches_the_same_starts(tmp_path):
    """You are going DOWN you Sputnik!: a blob-less version the bot's cleanup folds into the card (its stored time, the
    rename's commit) and the seed of the raw pickles (it dates that version by its bytes' first commit, under the old
    name no card holds) give the card one start."""
    from commands import card_data as cd
    index, c = _cards_repo(tmp_path, [(1000, {"Items/Old.psd": b"b one"}), (2000, {"Items/Old.psd": ("mv", "Items/B.psd")}),
                                      (3000, {"Items/B.psd": b"b two"})])

    def raw():
        v0 = _card("B", path="Items/B.psd", ability="Same.", timestamp=2000.0)
        return {"B": _card("B", path="Items/B.psd", ability="Same.", blob_sha=blob(b"b two"), timestamp=3000.0)}, \
            defaultdict(list, {"B": [v0]})
    stats, old = raw()
    ch.refine(stats, old, index)
    db = cm.StatsDatabase()
    db.stats, db.old_stats = raw()
    cleaned = cd.clean_history(db, index)
    assert ch.start(db.stats["B"]) == (2000.0, c[1]) and cleaned[6] == ["B"]     # carried, given its commit
    ch.refine(db.stats, db.old_stats, index)
    assert ch.start(db.stats["B"]) == ch.start(stats["B"]) == (1000.0, c[0])


def test_carried_starts_get_their_commit_in_the_bot(tmp_path):
    index, c = _cards_repo(tmp_path, [(1000, {"Items/A.psd": A1}), (2000, {"Items/A.psd": A2})])
    card = _card(blob_sha=blob(A2), timestamp=2000.0)
    card.version = ch.meta(None, 1000.0, "legacy")
    elsewhere = _card(blob_sha=blob(A2), timestamp=2000.0)
    elsewhere.version = ch.meta(None, 1500.0, "legacy")                         # no commit at that time: left alone
    assert ch.commits_of({"A": card, "E": elsewhere}, defaultdict(list), index) == ["A"]
    assert card.version["from_commit"] == c[0] and elsewhere.version["from_commit"] is None
    assert ch.commits_of({"A": card}, defaultdict(list), index) == []


def test_bot_pickles_reads_only_a_new_pickle_commit(lost, monkeypatch):
    """Each snapshot is unpickled, folded in and dropped before the next; a later PICKLE commit (a /push) is the only
    one read then, and the readings are those of reading all of them at once."""
    ch._PICKLES.clear()
    read = []
    real = ch._snapshots
    monkeypatch.setattr(ch, "_snapshots", lambda clone, commits: (read.append(len(commits)), real(clone, commits))[1])
    first = ch.bot_pickles(str(lost.bot))
    cdb.dump_pickle({"A": _card(ability="Newest.", timestamp=3000.0)}, str(lost.bot / "stats.pkl"))
    commit(lost.bot, "bot", 4800, "PICKLE")
    second = ch.bot_pickles(str(lost.bot))
    assert read == [4, 1] and second is first and len(second.shas) == 5
    whole = ch.Pickles.of(ch.bot_snapshots(str(lost.bot)))
    assert {n: sorted((r.card.path, r.card.timestamp, r.card.ability) for r in rs) for n, rs in second.readings.items()} == \
        {n: sorted((r.card.path, r.card.timestamp, r.card.ability) for r in rs) for n, rs in whole.readings.items()}
    ch._PICKLES.clear()


def test_high_history_stats_need_an_allowance_and_are_listed():
    """INT-4: a stat above 10 is unknown without an allowance (the card's allow_high_stats, the retired exclusion
    list, or no current card to tell); with one it is kept up to two digits and listed for the owner."""
    sonic = _card("Sonic", stats=cm.CardStats(7, 4, 4, 23), timestamp=1.0)
    twins = _card("Twin Emperors", stats=cm.CardStats(3, 7, 12, 13), timestamp=2.0)
    bill = _card("Bill", stats=cm.CardStats(16, 6, 6, 10), timestamp=3.0)
    tamer = _card("Tamer", stats=cm.CardStats(7, 9, 10, 15), timestamp=4.0)
    lazarus = _card("Tainted Lazarus", stats=cm.CardStats(110, 110, 110, 110), timestamp=5.0)
    stats = {"Sonic": _card("Sonic", allow_high_stats=True), "Twin Emperors": _card("Twin Emperors"),
             "Tamer": _card("Tamer"), "Tainted Lazarus": _card("Tainted Lazarus")}
    old = defaultdict(list, {"Sonic": [sonic], "Twin Emperors": [twins], "Bill": [bill], "Tamer": [tamer],
                             "Tainted Lazarus": [lazarus]})
    assert sorted(n for n, _, _ in ch.unknown_values(stats, old)) == ["Tainted Lazarus", "Tamer"]
    assert sonic.stats.speed == 23 and twins.stats.speed == 13 and bill.stats.hp == 16 and tamer.stats == cm.CardStats()
    assert [(n, v) for n, _, v in ch.high_values(stats, old)] == [("Sonic", "7/4/4/23"), ("Twin Emperors", "3/7/12/13"),
                                                                 ("Bill", "16/6/6/10")]


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

