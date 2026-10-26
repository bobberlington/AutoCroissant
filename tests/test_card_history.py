"""card_history: the git index (newest, first author, blob intro, deletions, exact renames), the version rules, the
archive tags' dates and a build's versions."""
import os
import subprocess
from collections import defaultdict
from copy import deepcopy
from types import SimpleNamespace

import pytest

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


@pytest.fixture
def mailmapped(tmp_path):
    """One email under two names (A by Michael, B by MichaelJSr), a .mailmap naming it MichaelJSr, C by someone else."""
    repo = tmp_path / "mailmapped"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    who = lambda name, email="Me@X": {"GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": email, "GIT_COMMITTER_NAME": name,
                                      "GIT_COMMITTER_EMAIL": email}
    shas = []
    for rel, name, text in (("Items/A.psd", "Michael", b"a"), ("Items/B.psd", "MichaelJSr", b"b"),
                            (".mailmap", "Michael", b"# one person\nMichaelJSr <me@x>\n")):
        write(repo, rel, text)
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", rel, env=who(name))
        shas.append(git(repo, "rev-parse", "HEAD").strip())
    write(repo, "Items/C.psd", b"c")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "c", env=who("Someone Else", "else@x"))
    return SimpleNamespace(repo=repo, shas=shas)


def test_index_names_authors_through_the_mailmap(mailmapped, tmp_path):
    """%aN with head's .mailmap: in the clone, a bare blobless one and a sparse blobless checkout without the file (its
    one blob fetched); at a head before the .mailmap, the git names as they are."""
    git(mailmapped.repo, "config", "uploadpack.allowFilter", "true")
    bare, sparse = tmp_path / "bare.git", tmp_path / "sparse"
    url = f"file://{mailmapped.repo}"
    subprocess.run(["git", "clone", "-q", "--bare", "--filter=blob:none", url, str(bare)], check=True)
    subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--sparse", "--no-checkout", url, str(sparse)], check=True)
    git(sparse, "sparse-checkout", "set", "--no-cone", "/Items/A.psd")
    git(sparse, "checkout", "-q", "main")
    no_lazy = {**os.environ, "GIT_NO_LAZY_FETCH": "1"}
    assert not (sparse / ".mailmap").exists()
    assert subprocess.run(["git", "-C", str(sparse), "cat-file", "-e", "HEAD:.mailmap"], env=no_lazy).returncode != 0
    for repo in (mailmapped.repo, bare, sparse):
        index = ch.GitIndex.from_git(str(repo))
        assert [c.author for c in index.commits] == ["Someone Else", "MichaelJSr", "MichaelJSr"], repo
        assert index.first["Items/A.psd"].author == "MichaelJSr"
    before = mailmapped.shas[1]                       # (a bare clone would still read HEAD's: git's default there)
    assert ch.mailmap_args(str(sparse), before) == []
    assert [c.author for c in ch.GitIndex.from_git(str(sparse), before).commits] == ["MichaelJSr", "Michael"]


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
    commit(repo, "bob", 2000)
    write(repo, "Items/A.psd", X)
    c3 = commit(repo, "carol", 3000)
    index = ch.GitIndex.from_git(str(repo))
    since = lambda path, at=None: index.blob_since(blob(X), path, at).sha
    assert (since("Items/A.psd"), since("Items/A.psd", 3500.0), since("Items/A.psd", 3000.0)) == (c3, c3, c3)
    assert since("Items/A.psd", 1500.0) == since("Items/A.psd", 1000.0) == index.blob_intro(blob(X)).sha == c1
    assert ch.start_of(_card(blob_sha=blob(X), timestamp=3000.0), index, None)["from_commit"] == c3
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


def test_a_stale_index_never_dates_a_tag_back_to_an_older_deletion():
    """Items/A.psd deleted at 150, added back at 200 and deleted again at 300 (tag from _archive_sources): a clone not
    fetched since the second deletion knows only the first, which dates nothing, with or without the tag's commit."""
    index = _index_with({}, {"Items/A.psd": ("old", 150.0)})
    for commit in ("new", None):
        record = _card(timestamp=200.0, archived={"at": 300.0, "path": "Items/A.psd", "blob": None, "commit": commit})
        report = ch.Refined()
        ch.date_archive({}, defaultdict(list, {"A": [record]}), index, report, backfill=False)
        assert (record.archived["at"], record.archived["commit"], report.tags_dated) == (300.0, commit, [])


V2 = dict(parser_version=2)


def test_one_form_for_the_same_data():
    """An empty ability is None or '' (Anubisath Guardian); a title equal to the name is no title (a Card Edits name
    fix equal to the file name)."""
    assert ch.knows(_card(ability=None), _card(ability=""))
    named = _card(display_name="A", **V2)
    assert ch.knows(named, _card(**V2)) and ch.data_of(named)["display_name"] is None


NO_LAYER = ["NO ABILITY LAYER"]


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
    record = _card(blob_sha="b1", timestamp=100.0, archived={"at": 150.0, "path": "Items/A.psd", "blob": "b1"}, **V2)
    db = _db({"A": _card(ability=None, path="Items/Heal/A.psd", blob_sha="b2", timestamp=500.0, problems=NO_LAYER,
                         **V2)}, {"A": [record]})
    assert db.drop_stale_archives() and db.old_stats["A"] == [record] and record.archived is None


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


def test_a_stale_archive_with_the_same_data_gives_the_card_its_start():
    """restore() declined (back at another path with another blob) but the data is the same: the tagged record goes,
    its earlier start to the listed card (as_of before the re-add still finds it)."""
    record = _card(blob_sha="b1", timestamp=100.0, archived={"at": 150.0, "path": "Items/A.psd", "blob": "b1"})
    listed = _card(path="Items/Heal/A.psd", blob_sha="b2", timestamp=500.0)
    db = _db({"A": listed}, {"A": [record]})
    assert db.drop_stale_archives() and "A" not in db.old_stats
    assert ch.start(listed) == (100.0, None) and listed.version["cause"] == "legacy" and not listed.version["legacy"]


# ========================
# Impossible values, renames
# ========================
G1, O1 = b"gone one", b"old one"


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
    printed = _card("P", stats=cm.CardStats(12, 3, 3, 3), stars=3, timestamp=4.0, parser_version=2)
    old["P"], stats["P"] = [printed], _card("P")      # read by parser 2: its problem was shown; kept for an allowance
    assert ch.unknown_values(stats, old) == [] and printed.stats.hp == 12 and ch.kept_values(stats, old) == ["P"]
    stats["P"] = _card("P", allow_high_stats=True)
    assert ch.kept_values(stats, old) == []


def test_a_rename_chain_reaches_the_listed_card(tmp_path):
    """Azamoth the Possessed King: moved within its name (2000), then renamed to Vessel (3000); Pandiety: renamed to
    Pandeity (2000, a name no record holds), then moved into MDW (3000). Each tag names where the chain ends, dated by
    the step that reached it (the old name's last day), where the new name starts; a second pass changes nothing."""
    index, c = _cards_repo(tmp_path, [(1000, {"Items/K.psd": O1, "Items/P.psd": G1}),
                                      (2000, {"Items/K.psd": ("mv", "Aux/K.psd"), "Items/P.psd": ("mv", "Items/P2.psd")}),
                                      (3000, {"Aux/K.psd": ("mv", "Aux/V.psd"), "Items/P2.psd": ("mv", "MDW/P2.psd")})])
    tags = {n: {"at": 2000.0, "path": f"Items/{n}.psd", "blob": b, "commit": c[1], "backfill": True}
            for n, b in (("K", blob(O1)), ("P", blob(G1)))}
    old = defaultdict(list, {n: [_card(n, path=f"Items/{n}.psd", blob_sha=tag["blob"], timestamp=1000.0, archived=tag)]
                             for n, tag in tags.items()})
    stats = {"V": _card("V", path="Aux/V.psd", blob_sha=blob(O1), timestamp=1000.0)}
    report = ch.Refined()
    ch.date_archive(stats, old, index, report)
    assert [(t.get("renamed_to"), t.get("moved_to"), t["at"], t["commit"]) for t in tags.values()] == [
        ("V", None, 3000.0, c[2]), (None, "MDW/P2.psd", 3000.0, c[2])]
    assert sorted(report.tags_dated) == ["K", "P"] and report.renamed == [("K", "V")] and report.retired == ["P"]
    again = ch.Refined()
    ch.date_archive(stats, old, index, again)
    assert again == ch.Refined()


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

