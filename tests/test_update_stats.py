"""update_stats / RepositoryTraverser against a fake GitHub (tree + blobs through a patched requests.get)."""
import hashlib
import json
import os
import pickle
from datetime import datetime, timezone
from threading import Thread
from types import SimpleNamespace

import pytest
from requests import HTTPError

import commands.card_runner as cr
import commands.github_api as ga
import commands.psd_analyzer as pa
import commands.repo_traverser as rt
import commands.utils as utils
from golden_cards import VOCAB, golden_file

DOOMER = "Auxiliary/Minions/Mini_Doomer.psd"
FREEZER = "Field/1 Stars/The_Freezer.psd"
OTHER = "Auxiliary/Minions/Anubisath_Guardian.psd"     # its bytes stand in for an edited Mini Doomer
TIME = 1_700_000_000.0


def _sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class Reply:
    def __init__(self, status=200, data=b"", tree=None, etag=""):
        self.status_code, self.content, self._tree = status, data, tree
        self.headers = {"ETag": etag} if etag else {}
        self.encoding = None

    @property
    def text(self):
        return self.content.decode(self.encoding or "utf-8")

    def json(self):
        return self._tree

    def raise_for_status(self):
        if self.status_code >= 400:
            raise HTTPError(f"{self.status_code} Server Error")

    def iter_content(self, size):
        for i in range(0, len(self.content), size):
            yield self.content[i:i + size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        pass


class FakeGitHub:
    """files: repo path -> bytes. tree_status / broken blobs simulate outages. calls: what was requested."""
    def __init__(self, files: dict):
        self.files = dict(files)
        self.tree_status = 200
        self.truncated = False
        self.broken: set = set()
        self.flaky: dict = {}                   # blob sha -> how many more 503s before it answers
        self.calls: list = []
        self.waits: list = []                   # get_with_retry's sleeps

    def tree(self) -> dict:
        return {"tree": [{"path": p, "sha": _sha(d), "type": "blob"} for p, d in sorted(self.files.items())],
                "truncated": self.truncated}

    def get(self, url, headers=None, timeout=None, stream=False):
        headers = headers or {}
        if "/git/trees/" in url:
            self.calls.append(("tree", headers.get("If-None-Match")))
            if self.tree_status != 200:
                return Reply(self.tree_status)
            tree = self.tree()
            etag = '"%s"' % hashlib.sha1(json.dumps(tree).encode()).hexdigest()
            return Reply(304) if headers.get("If-None-Match") == etag else Reply(200, tree=tree, etag=etag)
        sha = url.rsplit("/", 1)[1]
        path = next((p for p, d in self.files.items() if _sha(d) == sha), None)
        self.calls.append(("blob", path))
        if path is None or sha in self.broken:
            return Reply(503)
        if self.flaky.get(sha):
            self.flaky[sha] -= 1
            return Reply(503)
        return Reply(200, self.files[path])

    def blobs(self) -> list:
        return [p for kind, p in self.calls if kind == "blob"]


@pytest.fixture
def psds(clone):
    out = {}
    for rel in (DOOMER, FREEZER, OTHER):
        data = (clone / rel).read_bytes()
        if _sha(data) != json.loads(golden_file(rel).read_text(encoding="utf-8"))["blob"]:
            pytest.skip(f"{rel} changed upstream since the goldens were made")
        out[rel] = data
    return out


@pytest.fixture
def github(monkeypatch, psds):
    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
    files = {f"Types/{t}.png": t.encode() for t in vocab["all_types"]}
    files.update({f"Types/Stars/{s}.png": s.encode() for s in vocab["stars"]})
    files[pa.ICON_MANIFEST_PATH] = json.dumps(vocab["manifest"]).encode()
    files[pa.AUTHORS_PATH] = b"[people]\nbobgit = Bob\n[cards]\nmini_doomer = bobgit\n"
    files[pa.EDITS_PATH] = json.dumps({"cards": {"The Freezer": {"note": "checked", "_by": "web"}}}).encode()
    files[DOOMER], files[FREEZER] = psds[DOOMER], psds[FREEZER]
    fake = FakeGitHub(files)
    monkeypatch.setattr(ga, "get", fake.get)
    monkeypatch.setattr(ga, "sleep", fake.waits.append)
    monkeypatch.setattr(rt, "_tree_etags", {})
    monkeypatch.setattr(rt, "_file_cache", {})
    monkeypatch.setattr(pa.RepositoryTraverser, "_repo", lambda self: None)
    monkeypatch.setattr(pa.RepositoryTraverser, "_get_remote_timestamp", lambda self, repo, path, name: (TIME, "Tester"))
    for q in (utils.dispatch_queue, utils.edit_queue, utils.command_queue):
        q.clear()
    yield fake
    for q in (utils.dispatch_queue, utils.edit_queue, utils.command_queue):
        q.clear()


def _run(**kw):
    utils.command_queue.clear()
    out = pa.update_stats(None, **kw)
    return out, [func.__name__ for _, func in utils.command_queue]


def _saved():
    with open("stats.pkl", "rb") as f:
        stats = pickle.load(f)
    with open("old_stats.pkl", "rb") as f:
        return stats, pickle.load(f)


def _golden(rel):
    return json.loads(golden_file(rel).read_text(encoding="utf-8"))["card"]


def _file_sha(github, path):
    return _sha(github.files[path])


def _strip(d):
    return {k: v for k, v in d.items() if k not in ("timestamp", "author", "note")}


# ========================
# Tests
# ========================
def test_first_run_then_304(github, capsys):
    problems, queued = _run()
    assert problems == []
    assert sorted(github.blobs()) == sorted([DOOMER, FREEZER, pa.ICON_MANIFEST_PATH, pa.AUTHORS_PATH, pa.EDITS_PATH])
    assert queued == ["sync_changed_emojis", "refresh", "populate_files", "prep_dataframes"]   # no TTS save or bake run
    stats, old = _saved()
    assert set(stats) == {"Mini Doomer", "The Freezer"} and not old
    doomer, freezer = stats["Mini Doomer"], stats["The Freezer"]
    assert _strip(doomer.to_dict()) == _strip(_golden(DOOMER))
    assert _strip(freezer.to_dict()) == _strip(_golden(FREEZER))
    assert (doomer.author, freezer.author) == ("Bob", "Tester")      # authors file wins over the first commit
    assert freezer.note == "checked" and freezer.parsed_values == {"note": None, "problems": []}
    assert doomer.timestamp == TIME and doomer.blob_sha == _file_sha(github, DOOMER)
    assert "2 cards changed or new, 0 unchanged, 0 moved." in capsys.readouterr().out

    mtime = os.stat("stats.pkl").st_mtime_ns
    github.calls.clear()
    problems, queued = _run()
    assert problems == [] and github.calls[0][0] == "tree" and github.calls[0][1]   # sent If-None-Match
    assert github.blobs() == [] and os.stat("stats.pkl").st_mtime_ns == mtime     # 304: nothing parsed or saved
    assert queued == ["refresh", "populate_files", "prep_dataframes"]                 # the links still refresh; no bake
    assert "Nothing changed in the card repo" in capsys.readouterr().out


def test_changed_blob(github, psds, capsys):
    _run()
    github.files[DOOMER] = psds[OTHER]
    github.calls.clear()
    problems, _ = _run()
    assert github.blobs() == [DOOMER]                                 # small files cached by blob sha
    stats, old = _saved()
    assert stats["Mini Doomer"].ability == _golden(OTHER)["ability"]
    assert stats["Mini Doomer"].blob_sha == _sha(psds[OTHER]) and stats["Mini Doomer"].author == "Bob"
    assert [c.blob_sha for c in old["Mini Doomer"]] == [_sha(psds[DOOMER])]
    assert "The Freezer" not in old and stats["The Freezer"].note == "checked"
    assert problems == [f"{DOOMER}\n```\nABILITY TEXT NOT FOUND\n```"]
    assert "1 cards changed or new, 1 unchanged, 0 moved." in capsys.readouterr().out


def test_same_data_from_a_new_blob_is_no_version(github, psds, monkeypatch):
    """A re-saved PSD (new blob, newer commit) that parses the same: no history version; the card takes the blob and
    the date, and keeps its start for as_of. Then a real change archives the card before, its start kept too."""
    _run()
    later = TIME + 86400
    monkeypatch.setattr(pa.RepositoryTraverser, "_get_remote_timestamp", lambda self, repo, path, name: (later, None))
    github.files[DOOMER] = psds[DOOMER] + b"\0"                       # another blob, the same layers
    _run()
    stats, old = _saved()
    doomer = stats["Mini Doomer"]
    assert "Mini Doomer" not in old and doomer.blob_sha == _file_sha(github, DOOMER) and doomer.timestamp == later
    assert (doomer.version["from"], doomer.version["cause"], doomer.version["legacy"]) == (TIME, "legacy", False)
    github.files[DOOMER] = psds[OTHER]
    _run()
    stats, old = _saved()
    assert [c.blob_sha for c in old["Mini Doomer"]] == [_sha(psds[DOOMER] + b"\0")]
    assert old["Mini Doomer"][0].version["from"] == TIME and stats["Mini Doomer"].version is None


def test_duplicate_versions_are_cleaned_once(github, capsys):
    """The duplicates archived before 2026-10-22 go at the next /update_stats, even on a 304: the first of each run stays,
    a run equal to the card goes into it (its start kept); one line says so, and the run after says nothing."""
    _run()
    stats, _ = _saved()
    freezer = stats["The Freezer"]
    older = [pickle.loads(pickle.dumps(freezer)) for _ in range(3)]
    for card, when, blob in zip(older, (TIME - 300, TIME - 200, TIME - 100), ("1" * 40, "2" * 40, "3" * 40)):
        card.timestamp, card.blob_sha = when, blob
    older[0].ability = older[1].ability = "Before."
    with open("old_stats.pkl", "wb") as f:
        pickle.dump({"The Freezer": older}, f)
    capsys.readouterr()
    _run()
    stats, old = _saved()
    assert [c.blob_sha for c in old["The Freezer"]] == ["1" * 40]
    assert stats["The Freezer"].version["from"] == TIME - 100
    out = capsys.readouterr().out
    assert "History: 2 duplicate versions removed (1 card)." in out and "Nothing changed in the card repo" in out
    _run()
    assert "History:" not in capsys.readouterr().out


def test_moved_card(github, capsys):
    _run()
    github.files["Field/2 Stars/The_Freezer.psd"] = github.files.pop(FREEZER)
    github.calls.clear()
    _run()
    stats, old = _saved()
    assert stats["The Freezer"].path == "Field/2 Stars/The_Freezer.psd" and stats["The Freezer"].stars == 2
    assert [c.path for c in old["The Freezer"]] == [FREEZER]           # each version keeps its own path (git dates it)
    assert [c.stars for c in old["The Freezer"]] == [1]
    out = capsys.readouterr().out
    assert "0 cards changed or new, 1 unchanged, 1 moved." in out and "Archived" not in out   # a move archives nothing
    assert not any(c.archived for versions in old.values() for c in versions) and pa.stats_db.archived() == {}


def test_two_psds_with_one_name_never_flip_the_card(github, psds, capsys):
    """A copy of The Freezer's PSD under another folder (another blob): the stored card keeps its path, the copy is
    ignored with one line, and no run moves the card or adds a version."""
    _run()
    github.files["Field/2 Stars/The_Freezer.psd"] = psds[OTHER]
    capsys.readouterr()
    for _ in range(2):
        _run()
        stats, old = _saved()
        assert stats["The Freezer"].path == FREEZER and "The Freezer" not in old
    assert ("Two PSDs share the card name The Freezer: Field/1 Stars/The_Freezer.psd, Field/2 Stars/The_Freezer.psd; "
            "Field/2 Stars/The_Freezer.psd ignored until one is renamed.") in capsys.readouterr().out


def test_deleted_card_archived_with_a_change(github, psds):
    _run()
    del github.files[FREEZER]
    github.files[DOOMER] = psds[OTHER]
    _run()
    stats, old = _saved()
    assert set(stats) == {"Mini Doomer"} and len(old["The Freezer"]) == 1 and old["The Freezer"][0].archived
    assert old["Mini Doomer"][0].archived is None                     # an edit's history is no archive


def test_force_update_reparses_without_history(github):
    _run()
    github.calls.clear()
    _run(force_update=True)
    assert github.calls[0] == ("tree", None) and sorted(github.blobs()) == sorted([DOOMER, FREEZER])
    assert _saved()[1] == {}                                          # same blobs: nothing archived


def test_same_blob_reparse_keeps_the_start(github, capsys):
    """A carried start (the duplicate cleanup) survives cards: and force_update: the same PSD read again is no new
    version, so as_of and the website keep the earlier date."""
    _run()
    stats, _ = _saved()
    older = pickle.loads(pickle.dumps(stats["The Freezer"]))
    older.timestamp, older.blob_sha = TIME - 100, "1" * 40
    with open("old_stats.pkl", "wb") as f:
        pickle.dump({"The Freezer": [older]}, f)
    _run()
    assert _saved()[0]["The Freezer"].version["from"] == TIME - 100
    _run(cards="the_freezer")
    assert _saved()[0]["The Freezer"].version["from"] == TIME - 100
    _run(force_update=True)
    stats, old = _saved()
    assert stats["The Freezer"].version["from"] == TIME - 100 and old == {}


def test_older_parser_version_reparsed_once(github):
    """Every parse is stamped PARSER_VERSION; a card stamped by an older parser (or none: old pickles) is read again
    once, with the full tree, and without history (same blob)."""
    _run()
    stats, _ = _saved()
    assert {c.parser_version for c in stats.values()} == {pa.PARSER_VERSION}
    del stats["Mini Doomer"].__dict__["parser_version"]
    with open("stats.pkl", "wb") as f:
        pickle.dump(stats, f)
    github.calls.clear()
    _run()
    assert github.calls[0] == ("tree", None) and github.blobs() == [DOOMER]
    stats, old = _saved()
    assert stats["Mini Doomer"].parser_version == pa.PARSER_VERSION and old == {}
    github.calls.clear()
    _run()
    assert github.blobs() == [] and github.calls[0][1]                       # 304 again


def test_only_cards_reparsed(github):
    _run()
    github.calls.clear()
    _run(cards="mini_doomer")
    assert github.calls[0] == ("tree", None) and github.blobs() == [DOOMER]


def test_tree_5xx_retried_then_raises_and_saves_nothing(github):
    """B9: the tree is tried 3 times (1 s, then 2 s apart); a lasting 5xx escapes update_stats and saves nothing."""
    github.tree_status = 502
    with pytest.raises(HTTPError):
        _run()
    assert [kind for kind, _ in github.calls] == ["tree"] * 3 and github.waits == [1, 2]
    assert not os.path.exists("stats.pkl") and rt._tree_etags == {}


def test_tree_5xx_then_200_updates(github, monkeypatch):
    github.tree_status = 503
    real = github.get

    def recover(url, **kw):
        reply = real(url, **kw)
        github.tree_status = 200
        return reply
    monkeypatch.setattr(ga, "get", recover)
    _run()
    assert set(_saved()[0]) == {"Mini Doomer", "The Freezer"} and github.waits == [1]


def test_blob_5xx_skips_the_card_and_keeps_no_etag(github, psds, capsys):
    """B9: a lasting blob 5xx (3 tries) leaves the card at its old parse, and the tree ETag is not recorded, so the next
    run fetches the tree again and re-downloads the card instead of getting a 304."""
    _run()
    github.files[DOOMER] = psds[OTHER]
    github.broken.add(_sha(psds[OTHER]))
    github.calls.clear()
    _run()
    out = capsys.readouterr().out
    assert "download failed" in out and "trying once more" not in out           # a 5xx is not tried a 4th time
    assert "Could not download 1 file from the card repo; the next /update_stats tries again." in out
    assert github.blobs() == [DOOMER] * 3 and github.waits == [1, 2]
    stats, old = _saved()
    assert stats["Mini Doomer"].blob_sha == _sha(psds[DOOMER]) and old == {}
    assert rt._tree_etags == {}
    github.broken.clear()
    github.calls.clear()
    _run()
    assert github.calls[0] == ("tree", None) and github.blobs() == [DOOMER]       # no 304: the card is fetched again
    stats, old = _saved()
    assert stats["Mini Doomer"].blob_sha == _sha(psds[OTHER]) and [c.blob_sha for c in old["Mini Doomer"]] == [_sha(psds[DOOMER])]
    assert rt._tree_etags
    github.calls.clear()
    _run()
    assert github.blobs() == [] and github.calls[0][1]                          # 304 again once nothing failed


def test_failed_card_keeps_the_rest_of_the_run(github, capsys):
    """B9: the other cards of a run with a failed download are saved; only the failed one is fetched next time."""
    github.broken.add(_file_sha(github, DOOMER))
    _run()
    assert set(_saved()[0]) == {"The Freezer"} and rt._tree_etags == {}
    github.broken.clear()
    github.calls.clear()
    _run()
    assert github.calls[0] == ("tree", None) and github.blobs() == [DOOMER]
    assert set(_saved()[0]) == {"Mini Doomer", "The Freezer"} and rt._tree_etags


# ========================
# A PSD that raises: its card keeps its data, the run goes on
# ========================
BROKEN = b"8BPS" + b"\0" * 30                                         # a PSD header psd-tools refuses


def test_unreadable_psd_among_good_ones(github, psds, capsys):
    """An unreadable new PSD beside two good ones: 2 parsed, 1 failed, nothing archived, no ETag kept; fixed, it is read."""
    _seed_old_stats(_gone_card())                                     # a leftover the backfill would tag
    github.files["Items/Broken_Card.psd"] = BROKEN
    _run()
    stats, old = _saved()
    assert set(stats) == {"Mini Doomer", "The Freezer"} and old["Gone Card"][0].archived is None
    out = capsys.readouterr().out
    assert "Items/Broken_Card.psd: not read (" in out and "Archived" not in out
    assert ("Could not read 1 PSD from the card repo (Items/Broken_Card.psd); its card keeps its last version and the next "
            "/update_stats tries again.") in out and "Could not download" not in out
    assert "2 cards changed or new, 0 unchanged, 0 moved." in out and rt._tree_etags == {}
    github.files["Items/Broken_Card.psd"] = psds[OTHER]
    github.calls.clear()
    _run()
    assert github.calls[0] == ("tree", None) and github.blobs() == ["Items/Broken_Card.psd"]
    assert "Broken Card" in _saved()[0] and rt._tree_etags


def test_unreadable_change_keeps_the_card_and_its_details(github, psds, capsys):
    """A listed card's new blob raises: its last version stays (no history), its details too; a deletion beside it is
    still archived (a listed name is no rename)."""
    _run()
    before, details = _saved()[0]["Mini Doomer"], _details()
    github.files[DOOMER] = BROKEN
    del github.files[FREEZER]
    capsys.readouterr()
    _run()
    stats, old = _saved()
    assert set(stats) == {"Mini Doomer"} and stats["Mini Doomer"] == before and "Mini Doomer" not in old
    assert old["The Freezer"][0].archived and _details() == details
    assert f"Could not read 1 PSD from the card repo ({DOOMER})" in capsys.readouterr().out
    github.files[DOOMER] = psds[OTHER]
    _run()
    stats, old = _saved()
    assert stats["Mini Doomer"].blob_sha == _sha(psds[OTHER]) and [c.blob_sha for c in old["Mini Doomer"]] == [before.blob_sha]


def test_failed_commit_lookup_fails_only_its_card(github, monkeypatch, capsys):
    """A commit lookup that raises for one PSD fails that card alone; the others are saved."""
    def lookup(self, repo, path, name):
        if path == DOOMER:
            raise HTTPError("502 Server Error")
        return (TIME, "Tester")
    monkeypatch.setattr(pa.RepositoryTraverser, "_get_remote_timestamp", lookup)
    _run()
    assert set(_saved()[0]) == {"The Freezer"} and rt._tree_etags == {}
    out = capsys.readouterr().out
    assert f"{DOOMER}: commits not looked up (502 Server Error)" in out and "Could not read" not in out
    assert f"Could not look up the commits of 1 PSD ({DOOMER}); its card keeps its last version" in out


def test_blob_5xx_then_200_parses(github, psds):
    _run()
    github.files[DOOMER] = psds[OTHER]
    github.flaky[_sha(psds[OTHER])] = 2
    github.calls.clear()
    _run()
    assert github.blobs() == [DOOMER] * 3 and github.waits == [1, 2] and rt._tree_etags
    assert _saved()[0]["Mini Doomer"].blob_sha == _sha(psds[OTHER])


def test_failed_small_file_keeps_no_etag(github, capsys):
    """An authors or edits file that could not be read is fetched again next run (it applies to every card)."""
    github.broken.add(_file_sha(github, pa.EDITS_PATH))
    _run()
    assert "edits stay as they are" in capsys.readouterr().out and rt._tree_etags == {}
    assert _saved()[0]["The Freezer"].note != "checked"
    github.broken.clear()
    _run()
    assert _saved()[0]["The Freezer"].note == "checked" and rt._tree_etags


def test_broken_download_is_tried_once_more(github, psds, capsys):
    """A stream that breaks mid-file (IncompleteRead) is fetched again once; the card parses on the second try."""
    from requests.exceptions import ChunkedEncodingError

    class Broken:
        def __init__(self, reply):
            self.reply = reply
        def __getattr__(self, name):
            return getattr(self.reply, name)
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def iter_content(self, size):
            yield self.reply.content[:size]
            raise ChunkedEncodingError("Connection broken: IncompleteRead")
    _run()
    github.files[DOOMER] = psds[OTHER]
    real, once = github.get, [_sha(psds[OTHER])]

    def flaky(url, **kw):
        reply = real(url, **kw)
        if once and url.endswith(once[0]):
            once.pop()
            return Broken(reply)
        return reply
    github.calls.clear()
    import commands.github_api as ga
    patched = ga.get                                                   # the fixture's fake.get
    ga.get = flaky
    try:
        _run()
    finally:
        ga.get = patched
    out = capsys.readouterr().out
    assert "download failed" in out and "trying once more" in out
    stats, _ = _saved()
    assert stats["Mini Doomer"].blob_sha == _sha(psds[OTHER])          # the second try landed
    assert github.blobs().count(DOOMER) == 2


def test_deleted_card_alone_is_archived(github, capsys):
    """B8: a deletion with no other change moves the card to old_stats, tagged archived (its last path and blob)."""
    _run()
    freezer = _saved()[0]["The Freezer"]
    del github.files[FREEZER]
    capsys.readouterr()
    _run()
    stats, old = _saved()
    assert set(stats) == {"Mini Doomer"} and [c.blob_sha for c in old["The Freezer"]] == [freezer.blob_sha]
    tag = old["The Freezer"][0].archived
    assert tag == {"at": tag["at"], "path": FREEZER, "blob": freezer.blob_sha, "commit": None} and tag["at"] > TIME
    out = capsys.readouterr().out
    assert "Archived (PSD deleted from the card repo): The Freezer" in out
    assert "0 cards changed or new, 1 unchanged, 0 moved." in out
    assert set(pa.stats_db.archived()) == {"The Freezer"}
    from copy import copy
    archived = copy(old["The Freezer"][0])
    archived.archived = None
    freezer.__dict__.setdefault("archived", None)
    assert archived == freezer                                        # the record itself is unchanged
    _run()                                                            # 304: nothing archived or saved again
    assert _saved()[1]["The Freezer"] == old["The Freezer"] and len(_saved()[1]["The Freezer"]) == 1


def test_rename_with_failed_download_keeps_the_old_card(github, capsys):
    """A renamed PSD whose new blob fails: nothing is pruned, the old name stays listed until the retry lands."""
    _run()
    data = github.files.pop(DOOMER)
    github.files["Auxiliary/Minions/Mega_Doomer.psd"] = data
    github.broken.add(_sha(data))
    capsys.readouterr()
    _run()
    stats, old = _saved()
    assert set(stats) == {"Mini Doomer", "The Freezer"} and old == {}
    assert "Removed (PSD deleted" not in capsys.readouterr().out
    github.broken.clear()
    _run()
    stats, old = _saved()
    assert set(stats) == {"Mega Doomer", "The Freezer"} and len(old["Mini Doomer"]) == 1
    assert old["Mini Doomer"][0].archived["renamed_to"] == "Mega Doomer" and stats["Mega Doomer"].archived is None
    assert "Archived (PSD deleted from the card repo): Mini Doomer (now Mega Doomer)" in capsys.readouterr().out


def test_failed_listed_card_still_prunes(github, psds):
    """A failed download of a card already listed is no rename: a deleted card is still removed."""
    _run()
    github.files[DOOMER] = psds[OTHER]
    github.broken.add(_sha(psds[OTHER]))
    del github.files[FREEZER]
    _run()
    stats, old = _saved()
    assert set(stats) == {"Mini Doomer"} and stats["Mini Doomer"].blob_sha == _sha(psds[DOOMER])
    assert list(old) == ["The Freezer"] and old["The Freezer"][0].archived and rt._tree_etags == {}


def test_truncated_tree_prunes_nothing(github, capsys):
    _run()
    details = _details()
    del github.files[FREEZER]
    github.truncated = True
    capsys.readouterr()
    _run()
    stats, old = _saved()
    assert set(stats) == {"Mini Doomer", "The Freezer"} and old == {} and _details() == details
    assert "Card repo tree truncated or without cards; no card archived." in capsys.readouterr().out


def test_tree_without_cards_prunes_nothing(github, capsys):
    """A tree with no PSDs (say a /set_repo to another repo) moves no card to old_stats."""
    _run()
    del github.files[FREEZER], github.files[DOOMER]
    capsys.readouterr()
    _run()
    stats, old = _saved()
    assert set(stats) == {"Mini Doomer", "The Freezer"} and old == {}
    assert "no card archived" in capsys.readouterr().out


# ========================
# Archived cards (a deleted PSD's record stays, restorable)
# ========================
def _gone_card(name="Gone Card", blob="0" * 40):
    return pa.CardInfo(name=name, card_type="item", path=f"Items/{name.replace(' ', '_')}.psd", ability="Was a card.",
                       blob_sha=blob, timestamp=TIME - 1)


def _seed_old_stats(*cards):
    from collections import defaultdict
    old = defaultdict(list)
    for card in cards:
        old[card.name].append(card)
    with open("old_stats.pkl", "wb") as f:
        pickle.dump(old, f)


def test_archived_card_keeps_its_history(github, psds):
    """The earlier versions stay in front of the archived record, untouched."""
    _run()
    github.files[DOOMER] = psds[OTHER]
    _run()
    first = _saved()[1]["Mini Doomer"]
    del github.files[DOOMER]
    _run()
    stats, old = _saved()
    versions = old["Mini Doomer"]
    assert "Mini Doomer" not in stats and [c.blob_sha for c in versions] == [_sha(psds[DOOMER]), _sha(psds[OTHER])]
    assert versions[:1] == first and versions[0].archived is None
    assert versions[1].archived["blob"] == _sha(psds[OTHER]) and versions[1].archived["path"] == DOOMER


def test_readded_card_is_restored_without_a_download(github, psds, capsys):
    """The same PSD back: the archived record returns to stats as it was, no download, no history."""
    _run()
    before = _saved()[0]["The Freezer"]
    del github.files[FREEZER]
    _run()
    github.files[FREEZER] = psds[FREEZER]
    github.calls.clear()
    capsys.readouterr()
    _run()
    stats, old = _saved()
    assert github.blobs() == [] and "The Freezer" not in old and pa.stats_db.archived() == {}
    before.__dict__.setdefault("archived", None)
    assert stats["The Freezer"] == before and stats["The Freezer"].archived is None
    out = capsys.readouterr().out
    assert "Restored (PSD back in the card repo): The Freezer" in out and "0 moved." in out
    github.calls.clear()
    _run()
    assert github.blobs() == [] and github.calls[0][1]                        # 304 next time


def test_readded_card_with_a_new_blob_is_restored_then_reparsed(github, psds):
    """Another PSD under the name: restored, then re-parsed like an edit; the restored record becomes history."""
    _run()
    del github.files[DOOMER]
    _run()
    github.files[DOOMER] = psds[OTHER]
    github.calls.clear()
    _run()
    stats, old = _saved()
    assert github.blobs() == [DOOMER] and stats["Mini Doomer"].blob_sha == _sha(psds[OTHER])
    assert [c.blob_sha for c in old["Mini Doomer"]] == [_sha(psds[DOOMER])] and old["Mini Doomer"][0].archived is None
    assert stats["Mini Doomer"].author == "Bob"


def test_readded_card_in_another_folder_is_a_move(github, psds, capsys):
    _run()
    del github.files[FREEZER]
    _run()
    github.files["Field/2 Stars/The_Freezer.psd"] = psds[FREEZER]
    capsys.readouterr()
    _run()
    stats, old = _saved()
    assert stats["The Freezer"].stars == 2 and [c.stars for c in old["The Freezer"]] == [1]
    assert not any(c.archived for c in old["The Freezer"])
    out = capsys.readouterr().out
    assert "Restored (PSD back in the card repo): The Freezer" in out and "1 moved." in out


def test_deleted_card_without_a_listed_entry_is_not_restored(github, psds):
    """An untagged history (a card /update_metadata del_entry reset) is never restored: the PSD parses afresh."""
    _run()
    stale = _gone_card("The Freezer", blob=_file_sha(github, FREEZER))
    stats, old = _saved()
    del stats["The Freezer"]
    old["The Freezer"].append(stale)
    with open("stats.pkl", "wb") as f:
        pickle.dump(stats, f)
    with open("old_stats.pkl", "wb") as f:
        pickle.dump(old, f)
    github.calls.clear()
    _run(force_update=True)
    stats, old = _saved()
    assert stats["The Freezer"].ability != stale.ability
    assert [c.ability for c in old["The Freezer"]] == [stale.ability] and old["The Freezer"][0].archived is None


def test_backfill_tags_earlier_deleted_cards(github, capsys):
    """Names in old_stats only, from before archiving: the first full run tags their last version (backfill)."""
    gone = _gone_card()
    _seed_old_stats(gone)
    _run()
    stats, old = _saved()
    tag = old["Gone Card"][0].archived
    assert tag == {"at": tag["at"], "path": "Items/Gone_Card.psd", "blob": gone.blob_sha, "commit": None, "backfill": True}
    assert "Gone Card" not in stats and set(pa.stats_db.archived()) == {"Gone Card"}
    assert "Archived 1 card deleted before this update: Gone Card" in capsys.readouterr().out
    github.files["notes.txt"] = b"a change"
    _run()
    assert _saved()[1]["Gone Card"][0].archived == tag                       # tagged once
    assert "deleted before this update" not in capsys.readouterr().out


@pytest.mark.parametrize("guard", ["failed new download", "truncated tree", "no cards"])
def test_guarded_runs_archive_and_backfill_nothing(github, psds, guard, capsys):
    """A run that may not have seen the whole repo moves no card and tags no leftover."""
    _seed_old_stats(_gone_card())
    github.truncated = True
    _run()                                                                  # stats filled, nothing tagged
    github.truncated = False
    if guard == "failed new download":
        github.files["Items/New_Card.psd"] = psds[OTHER]
        github.broken.add(_sha(psds[OTHER]))
        del github.files[FREEZER]
    elif guard == "truncated tree":
        github.truncated = True
        del github.files[FREEZER]
    else:
        del github.files[FREEZER], github.files[DOOMER]
    capsys.readouterr()
    _run()
    stats, old = _saved()
    assert set(stats) == {"Mini Doomer", "The Freezer"} and list(old) == ["Gone Card"]
    assert old["Gone Card"][0].archived is None and pa.stats_db.archived() == {}
    assert "Archived" not in capsys.readouterr().out


def test_kill_between_the_saves_leaves_one_record(github, monkeypatch):
    """old_stats.pkl is saved first: a kill before stats.pkl leaves the card in both; the next run replaces the
    tagged copy instead of adding a second one."""
    import commands.card_models as cm
    _run()
    del github.files[FREEZER]
    real = cm.save_pickle

    def killed(obj, path, lock):
        if path == cm.STATS_PKL:
            raise KeyboardInterrupt("killed")
        real(obj, path, lock)
    monkeypatch.setattr(cm, "save_pickle", killed)
    with pytest.raises(KeyboardInterrupt):
        _run()
    stats, old = _saved()
    assert "The Freezer" in stats and len(old["The Freezer"]) == 1 and old["The Freezer"][0].archived
    monkeypatch.setattr(cm, "save_pickle", real)
    monkeypatch.setattr(rt, "_tree_etags", {})                            # the restart after the kill
    _run()
    stats, old = _saved()
    assert "The Freezer" not in stats and len(old["The Freezer"]) == 1 and old["The Freezer"][0].archived


def _kill_on(monkeypatch, target):
    """save_pickle raises on target's first write (a kill there); returns the real one."""
    import commands.card_models as cm
    real = cm.save_pickle

    def killed(obj, path, lock):
        if path == target:
            raise KeyboardInterrupt("killed")
        real(obj, path, lock)
    monkeypatch.setattr(cm, "save_pickle", killed)
    return real


def _restart(monkeypatch, real):
    import commands.card_models as cm
    monkeypatch.setattr(cm, "save_pickle", real)
    monkeypatch.setattr(rt, "_tree_etags", {})


def test_kill_between_the_saves_after_a_restore_keeps_the_record(github, psds, monkeypatch):
    """A same-blob restore keeps the archived record in old_stats.pkl until stats.pkl holds the card."""
    import commands.card_models as cm
    _run()
    del github.files[FREEZER]
    _run()
    archived = _saved()[1]["The Freezer"]
    github.files[FREEZER] = psds[FREEZER]
    real = _kill_on(monkeypatch, cm.STATS_PKL)
    with pytest.raises(KeyboardInterrupt):
        _run()
    stats, old = _saved()
    assert "The Freezer" not in stats and old["The Freezer"] == archived           # still archived, not lost
    _restart(monkeypatch, real)
    github.calls.clear()
    _run()
    stats, old = _saved()
    assert "The Freezer" in stats and "The Freezer" not in old and github.blobs() == []


def test_kill_after_a_restore_before_the_drop_is_cleaned_next_run(github, psds, monkeypatch):
    """Killed after stats.pkl but before old_stats.pkl drops the copy: the next run drops it."""
    import commands.card_models as cm
    _run()
    del github.files[FREEZER]
    _run()
    github.files[FREEZER] = psds[FREEZER]
    real = cm.save_pickle
    writes = []

    def killed(obj, path, lock):
        writes.append(path)
        if len(writes) == 3:
            raise KeyboardInterrupt("killed")
        real(obj, path, lock)
    monkeypatch.setattr(cm, "save_pickle", killed)
    with pytest.raises(KeyboardInterrupt):
        _run()
    stats, old = _saved()
    assert "The Freezer" in stats and old["The Freezer"][0].archived                # in both
    _restart(monkeypatch, real)
    _run()
    stats, old = _saved()
    assert "The Freezer" in stats and "The Freezer" not in old


def test_change_kill_between_the_saves_keeps_one_history(github, psds, monkeypatch):
    """An edit killed before stats.pkl: the next run replaces the equal history entry instead of doubling it."""
    import commands.card_models as cm
    _run()
    github.files[DOOMER] = psds[OTHER]
    real = _kill_on(monkeypatch, cm.STATS_PKL)
    with pytest.raises(KeyboardInterrupt):
        _run()
    assert len(_saved()[1]["Mini Doomer"]) == 1
    _restart(monkeypatch, real)
    _run()
    stats, old = _saved()
    assert stats["Mini Doomer"].blob_sha == _sha(psds[OTHER])
    assert [c.blob_sha for c in old["Mini Doomer"]] == [_sha(psds[DOOMER])]


def test_move_kill_between_the_saves_keeps_one_history(github, monkeypatch):
    """A same-blob move killed before stats.pkl: its history entry (at its own path) is already saved; the next run
    replaces it instead of doubling it."""
    import commands.card_models as cm
    _run()
    moved = "Field/2 Stars/The_Freezer.psd"
    github.files[moved] = github.files.pop(FREEZER)
    real = _kill_on(monkeypatch, cm.STATS_PKL)
    with pytest.raises(KeyboardInterrupt):
        _run()
    assert [c.path for c in _saved()[1]["The Freezer"]] == [FREEZER]
    _restart(monkeypatch, real)
    _run()
    stats, old = _saved()
    assert stats["The Freezer"].path == moved and [(c.path, c.stars) for c in old["The Freezer"]] == [(FREEZER, 1)]


def test_restore_reparse_kill_between_the_saves_stays_archived(github, psds, monkeypatch):
    """Restored then re-parsed (a new blob), killed before stats.pkl: old_stats.pkl still holds the tagged record,
    so the card is archived, not in neither; the next run restores and re-parses it again."""
    import commands.card_models as cm
    _run()
    del github.files[DOOMER]
    _run()
    github.files[DOOMER] = psds[OTHER]
    real = _kill_on(monkeypatch, cm.STATS_PKL)
    with pytest.raises(KeyboardInterrupt):
        _run()
    stats, old = _saved()
    assert "Mini Doomer" not in stats and len(old["Mini Doomer"]) == 1 and old["Mini Doomer"][0].archived
    db = cm.StatsDatabase()
    db.load()
    assert set(db.archived()) == {"Mini Doomer"}
    _restart(monkeypatch, real)
    _run()
    stats, old = _saved()
    assert stats["Mini Doomer"].blob_sha == _sha(psds[OTHER]) and pa.stats_db.archived() == {}
    assert [(c.blob_sha, c.archived) for c in old["Mini Doomer"]] == [(_sha(psds[DOOMER]), None)]


def test_kill_while_archiving_then_psd_back_leaves_no_old_version(github, psds, monkeypatch):
    """The tagged copy a kill left is dropped once the PSD is back, not shown as an old version."""
    import commands.card_models as cm
    _run()
    del github.files[FREEZER]
    real = _kill_on(monkeypatch, cm.STATS_PKL)
    with pytest.raises(KeyboardInterrupt):
        _run()
    assert _saved()[1]["The Freezer"][0].archived
    github.files[FREEZER] = psds[FREEZER]
    _restart(monkeypatch, real)
    _run()
    stats, old = _saved()
    assert "The Freezer" in stats and "The Freezer" not in old


def test_restored_card_with_a_failed_download_holds_the_prune(github, psds):
    """A new blob under an archived name that fails to download counts as a failed new name: nothing archived."""
    _run()
    del github.files[DOOMER]
    _run()
    github.files[DOOMER] = psds[OTHER]
    github.broken.add(_sha(psds[OTHER]))
    del github.files[FREEZER]
    _run()
    stats, old = _saved()
    assert set(stats) == {"Mini Doomer", "The Freezer"} and "The Freezer" not in old
    assert stats["Mini Doomer"].blob_sha == _sha(psds[DOOMER]) and "Mini Doomer" not in old   # restored as it was


def test_another_psd_under_an_archived_name_is_a_new_card(github, psds, monkeypatch, capsys):
    """Another path and blob: not restored, so the author is its own; the archived record stays as history."""
    _run()
    gone = _saved()[0]["The Freezer"]
    del github.files[FREEZER]
    _run()
    new = "Field/2 Stars/The_Freezer.psd"
    github.files[new] = psds[OTHER]
    monkeypatch.setattr(pa.RepositoryTraverser, "_get_remote_timestamp",
                        lambda self, repo, path, name: (TIME, "Newcomer" if path == new else "Tester"))
    capsys.readouterr()
    _run()
    stats, old = _saved()
    assert stats["The Freezer"].author == "Newcomer" and stats["The Freezer"].blob_sha == _sha(psds[OTHER])
    assert [(c.blob_sha, c.archived) for c in old["The Freezer"]] == [(gone.blob_sha, None)]
    assert pa.stats_db.archived() == {}
    out = capsys.readouterr().out
    assert "Restored" not in out and "Back under a new PSD (the archived record stays as history): The Freezer" in out


def test_rename_onto_a_listed_name_records_renamed_to(github, psds):
    """Freezer deleted, its PSD renamed over Mini Doomer's: the archive names Mini Doomer (same blob)."""
    _run()
    github.files[DOOMER] = github.files.pop(FREEZER)
    _run()
    old = _saved()[1]
    assert old["The Freezer"][0].archived["renamed_to"] == "Mini Doomer"


class _File:
    def __init__(self, filename, status="removed", previous_filename=None):
        self.filename, self.status, self.previous_filename = filename, status, previous_filename


class _Commit:
    def __init__(self, sha, files=(), at=TIME - 500):
        self.sha, self._files = sha, list(files)
        self.commit = SimpleNamespace(committer=SimpleNamespace(date=datetime.fromtimestamp(at, timezone.utc)))

    @property
    def files(self):
        return self._files


def test_archive_records_the_deleting_commit(github, psds, monkeypatch, capsys):
    """The path's newest commit is the one that deleted it; a rename with an edit is found in its files."""
    pages = {}
    monkeypatch.setattr(pa.RepositoryTraverser, "_commit_page", lambda self, path: pages.get(path))
    _run()
    pages[FREEZER] = [_Commit("f" * 40, [_File(FREEZER)])]
    pages[DOOMER] = [_Commit("d" * 40, [_File("Auxiliary/Minions/Mega_Doomer.psd", "renamed", DOOMER)])]
    del github.files[FREEZER]
    github.files["Auxiliary/Minions/Mega_Doomer.psd"] = psds[OTHER]       # renamed and edited: another blob
    del github.files[DOOMER]
    capsys.readouterr()
    _run()
    old = _saved()[1]
    assert old["The Freezer"][0].archived["commit"] == "f" * 40 and "renamed_to" not in old["The Freezer"][0].archived
    assert old["The Freezer"][0].archived["at"] == TIME - 500           # the deleting commit's date, not the run's
    assert old["Mini Doomer"][0].archived["commit"] == "d" * 40
    assert old["Mini Doomer"][0].archived["renamed_to"] == "Mega Doomer"
    assert "Mini Doomer (now Mega Doomer), The Freezer" in capsys.readouterr().out


def test_cards_deleted_together_read_the_commit_once(github, psds, monkeypatch):
    reads = []

    class Counted(_Commit):
        @property
        def files(self):
            reads.append(self.sha)
            return self._files
    files = [_File(FREEZER), _File("Auxiliary/Minions/Mega_Doomer.psd", "renamed", DOOMER)]
    monkeypatch.setattr(pa.RepositoryTraverser, "_commit_page", lambda self, path: [Counted("c" * 40, files)])
    _run()
    del github.files[FREEZER]
    github.files["Auxiliary/Minions/Mega_Doomer.psd"] = psds[OTHER]
    del github.files[DOOMER]
    _run()
    old = _saved()[1]
    assert reads == ["c" * 40] and old["Mini Doomer"][0].archived["renamed_to"] == "Mega Doomer"
    assert old["The Freezer"][0].archived["commit"] == "c" * 40 and "renamed_to" not in old["The Freezer"][0].archived


def test_plain_deletion_reads_no_commit_files(github, monkeypatch):
    """No card added this run: no rename to find, so the commit's files (another request) are not read."""
    class Unread(_Commit):
        @property
        def files(self):
            raise AssertionError("files read")
    monkeypatch.setattr(pa.RepositoryTraverser, "_commit_page", lambda self, path: [Unread("f" * 40)])
    _run()
    del github.files[FREEZER]
    _run()
    assert _saved()[1]["The Freezer"][0].archived["commit"] == "f" * 40


def test_failed_commit_lookup_still_archives(github, monkeypatch, capsys):
    def broken(self, path):
        raise HTTPError("502 Server Error")
    _run()
    monkeypatch.setattr(pa.RepositoryTraverser, "_commit_page", broken)
    del github.files[FREEZER]
    capsys.readouterr()
    _run()
    old = _saved()[1]
    assert old["The Freezer"][0].archived["commit"] is None
    assert f"{FREEZER}: deleting commit not found (502 Server Error)" in capsys.readouterr().out


def _git(repo, *args, at=None):
    import subprocess
    stamp = {} if at is None else {"GIT_AUTHOR_DATE": f"@{at} +0000", "GIT_COMMITTER_DATE": f"@{at} +0000"}
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True,
                          env={**os.environ, "GIT_AUTHOR_NAME": "Zomoroko", "GIT_AUTHOR_EMAIL": "z@x",
                               "GIT_COMMITTER_NAME": "Zomoroko", "GIT_COMMITTER_EMAIL": "z@x", **stamp}).stdout


def test_history_clone_dates_archive_tags(github, psds, tmp_path, monkeypatch, capsys):
    """ZOEY!!!!!, archived by the run's clock (2026-10-22) though git renamed it to Z!!!!! (same bytes) months before:
    with the bot's history clone the cleanup gives its tag the deleting commit's date and the new name, once."""
    import commands.card_data as cd
    zoey, z = "Creatures/Other/5 Stars/ZOEY!!!!!.psd", "Creatures/Other/5 Stars/Z!!!!!.psd"
    repo = tmp_path / "cards"
    (repo / "Creatures/Other/5 Stars").mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    (repo / zoey).write_bytes(b"zoey")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "ZOEY", at=1000)
    _git(repo, "mv", zoey, z)
    _git(repo, "commit", "-q", "-m", "Z!!!!!!", at=2000)
    renaming = _git(repo, "rev-parse", "HEAD").strip()
    _git(tmp_path, "clone", "-q", "--bare", str(repo), str(tmp_path / "history.git"))
    monkeypatch.setattr(cd, "HISTORY_CLONE", str(tmp_path / "history.git"))
    github.files[z] = psds[OTHER]
    _run()
    gone = pa.CardInfo(name="ZOEY!!!!!", card_type="creature", path=zoey, ability="Was a card.", timestamp=500.0,
                       archived={"at": TIME, "path": zoey, "blob": None, "commit": None, "backfill": True})
    _seed_old_stats(gone)
    capsys.readouterr()
    _run()
    tag = _saved()[1]["ZOEY!!!!!"][0].archived
    assert (tag["at"], tag["commit"], tag["renamed_to"]) == (2000.0, renaming, "Z!!!!!")
    line = "History: 0 duplicate versions removed (0 cards); archive tag from git: ZOEY!!!!!."
    assert line in capsys.readouterr().out
    _run()
    assert "History:" not in capsys.readouterr().out and _saved()[1]["ZOEY!!!!!"][0].archived == tag
    old = _saved()[1]                                           # a lost stats.pkl: the cleanup tags nothing (prune's job)
    old["Lost Card"] = [_gone_card("Lost Card")]
    db = pa.StatsDatabase()
    db.stats, db.old_stats = {}, old
    assert cd.clean_history(db, cd.history_index())[2] == [] and db.old_stats["Lost Card"][0].archived is None


def test_archived_cards_reach_no_reader(github, monkeypatch, capsys):
    """Every stats.pkl reader skips an archived card: the query dataframes and DSL, /query's card lookup, titles,
    /list_orphans, /mass_replace, the exports and the problem reports. aliases.pkl keeps its aliases."""
    import commands.query_card as qc
    with open("aliases.pkl", "wb") as f:
        pickle.dump({"freezy": "the_freezer.png"}, f)
    aliases = open("aliases.pkl", "rb").read()
    _run()
    freezer = _saved()[0]["The Freezer"]
    del github.files[FREEZER]
    problems, _ = _run(cards="the freezer, mini doomer")                    # a name lookup asked for too
    assert all("Freezer" not in p for p in problems) and set(pa.stats_db.archived()) == {"The Freezer"}
    assert open("aliases.pkl", "rb").read() == aliases

    assert "The Freezer" not in pa.get_stats_as_dict()
    repo = qc.CardRepository()
    repo.prep_dataframes()
    assert "The Freezer" not in repo.cards_dff.index and "The Freezer" not in repo.rulebook_dff.index
    assert all("Freezer" not in str(i) for i in repo.ability_search_engine("ice").index)
    assert all("Freezer" not in str(i) for i in repo.ability_search_engine("type==field").index)

    repo.git_files = {"the_freezer.png": "Field/1 Stars/The_Freezer.png", "freezy.png": "Field/1 Stars/The_Freezer.png"}
    monkeypatch.setattr(qc, "card_repo", repo)
    assert qc._stats_card("the_freezer.png") is None and qc._stats_card("freezy.png") is None
    pa.stats_db.old_stats["The Freezer"][0].display_name = "Freezer Title"
    repo._populate_titles()
    assert "freezer_title.png" not in repo.git_files

    who = utils.make_simple_fake_interaction(1)
    utils.dispatch_queue.clear()
    pa.list_orphans(who)
    import commands.discord_edits as de
    monkeypatch.setattr(de, "commit", lambda subject, by, body, edits=None, authors=None: ("abc1234", {"authors": authors(None)}))
    de.mass_replace_field(who, "author", freezer.author, "Somebody")
    assert pa.stats_db.old_stats["The Freezer"][0].author == freezer.author
    for csv in (True, False):
        pa.export_stats_to_file(who, as_csv=csv)
    assert "Freezer" not in open("stats.csv").read() and "Freezer" not in open("stats.txt").read()
    sent = " ".join(str(kw.get("content")) for _, kw in utils.dispatch_queue)
    assert "Freezer" not in sent
    utils.dispatch_queue.clear()


def test_set_repo_update_queues_no_emoji_sync(github, monkeypatch):
    """A /set_repo'd update queues no emoji sync: the type emojis follow the card repo only; its fonts judge the details."""
    import commands.psd_details as pd
    asked = []
    monkeypatch.setattr(pd, "font_index", lambda repository, tree=None: asked.append(repository))
    monkeypatch.setattr(pa.card_repo, "repository", "someone/else")
    _, queued = _run()
    assert "sync_changed_emojis" not in queued and asked == ["someone/else"]


def test_bake_lock_serialises_update_stats(github, capsys):
    """update_stats waits for a running bake (card_runner.bake_lock, shared with /bake_cards) and says so."""
    cr.bake_lock.acquire()
    try:
        worker = Thread(target=_run)
        worker.start()
        worker.join(0.5)
        assert worker.is_alive() and not os.path.exists("stats.pkl")
    finally:
        cr.bake_lock.release()
    worker.join(120)
    assert not worker.is_alive() and os.path.exists("stats.pkl")
    assert "Waiting for the running card update, bake or font check to finish" in capsys.readouterr().out


def _details():
    import commands.psd_details as pd
    return pd.load("psd_details.json")                                 # cwd-relative, like the pickles


def test_details_follow_the_parsed_cards(github, psds, clone, monkeypatch):
    """Every parsed card's PSD facts land in psd_details.json by blob sha, saved with the pickles; a blob that left
    the tree takes its record with it unless an archived card holds it, and a 304 writes nothing."""
    import commands.font_check as fc
    import commands.psd_details as pd
    fonts = clone / fc.FONTS_DIR
    monkeypatch.setenv(fc.FONT_DIRS_ENV, str(fonts))                            # the clone's fonts, whatever the shell says
    _run()
    details = _details()
    assert set(details) == {_file_sha(github, DOOMER), _file_sha(github, FREEZER)}
    doomer = details[_file_sha(github, DOOMER)]
    assert (doomer["path"], doomer["name"], doomer["parser"]) == (DOOMER, "Mini Doomer", pa.PARSER_VERSION)
    assert doomer["stats"]["hp"] == "template" and doomer["icons"] == "mixed" and doomer["template"]["name_layer"] == "Text/Name"
    assert {e["supplied"] for e in doomer["fonts"].values()} == ({True} if fonts.is_dir() else {None})
    mtime = os.stat("psd_details.json").st_mtime_ns
    _run()                                                                      # 304
    assert os.stat("psd_details.json").st_mtime_ns == mtime
    github.files[DOOMER] = psds[OTHER]
    freezer = _file_sha(github, FREEZER)
    del github.files[FREEZER]
    _run()
    details = _details()
    assert set(details) == {_sha(psds[OTHER]), freezer} and details[_sha(psds[OTHER])]["path"] == DOOMER   # archived: kept
    assert not os.path.exists("psd_details.json.tmp")
    assert pd.load("psd_details.json") == details


def test_details_flags_are_refilled_on_a_changed_tree(github, monkeypatch):
    """Records gathered without fonts (a failed download) get their flags on the next update whose tree changed,
    re-parsed or not."""
    import commands.psd_details as pd

    class Everything:
        def resolve(self, name):
            return "exact"
    monkeypatch.setattr(pd, "font_index", lambda repository, tree=None: None)
    _run()
    doomer = _file_sha(github, DOOMER)
    assert {e["supplied"] for e in _details()[doomer]["fonts"].values()} == {None}
    monkeypatch.setattr(pd, "font_index", lambda repository, tree=None: Everything())
    github.files["notes.txt"] = b"not a card"
    _run()
    assert {e["supplied"] for e in _details()[doomer]["fonts"].values()} == {True}


def test_truncated_tree_saves_the_details_prunes_none(github, psds, monkeypatch):
    """A truncated tree: the records gathered and refilled are saved; the ones whose blob it lacks stay."""
    import commands.psd_details as pd

    class Everything:
        def resolve(self, name):
            return "exact"
    monkeypatch.setattr(pd, "font_index", lambda repository, tree=None: None)
    _run()
    doomer, freezer = _file_sha(github, DOOMER), _file_sha(github, FREEZER)
    monkeypatch.setattr(pd, "font_index", lambda repository, tree=None: Everything())
    github.files[DOOMER] = psds[OTHER]
    del github.files[FREEZER]
    github.truncated = True
    _run()
    details = _details()
    assert set(details) == {doomer, freezer, _sha(psds[OTHER])}
    assert {e["supplied"] for e in details[doomer]["fonts"].values()} == {True}


def test_no_fonts_keeps_no_etag(github, monkeypatch):
    """A tree with MDW/Fonts whose fonts could not be had: no 304 next run, so the supplied flags are tried again."""
    import commands.psd_details as pd
    monkeypatch.setattr(pd, "font_index", lambda repository, tree=None: None)
    _run()
    assert rt._tree_etags                                                       # no MDW/Fonts in the tree
    github.files["MDW/Fonts/A.ttf"] = b"font"
    _run()
    assert rt._tree_etags == {}
    github.calls.clear()
    _run()
    assert github.calls[0] == ("tree", None)


def test_details_gather_error_never_fails_the_update(github, monkeypatch, capsys):
    import commands.psd_details as pd
    monkeypatch.setattr(pd, "gather", lambda *a, **kw: (_ for _ in ()).throw(ValueError("odd effects block")))
    problems, _ = _run()
    assert problems == [] and set(_saved()[0]) == {"Mini Doomer", "The Freezer"}
    out = capsys.readouterr().out
    assert f"{DOOMER}: details not gathered (odd effects block)" in out and not os.path.exists("psd_details.json")


def test_website_files_live_in_site():
    """The authors and edits files are read from the card repo's site/ only (moved from MDW/ on 2026-10-20)."""
    assert (pa.AUTHORS_PATHS, pa.EDITS_PATHS) == (("site/Card Authors.txt",), ("site/Card Edits.json",))
    assert (pa.AUTHORS_PATH, pa.EDITS_PATH) == ("site/Card Authors.txt", "site/Card Edits.json")


def test_alt_art_duplicates_are_archived_never_dropped(github, psds, capsys):
    """A duplicate whose PNG leaves is archived like any card (its tag at its own PNG), listed, and back when the PNG
    is; one whose original's PSD leaves goes with it, dated like it."""
    dup = "Auxiliary/Minions/Mini_Doomer2.png"
    github.files[dup] = b"alt art"
    _run()
    assert _saved()[0]["Mini Doomer2"].duplicate_of == "Mini Doomer"
    del github.files[dup]
    capsys.readouterr()
    _run()
    stats, old = _saved()
    record = old["Mini Doomer2"][0]
    assert "Mini Doomer2" not in stats and (record.duplicate_of, record.archived["path"]) == ("Mini Doomer", dup)
    assert "Archived alt-art cards (their PNG or original left the card repo): Mini Doomer2 (same card as Mini Doomer)" \
        in capsys.readouterr().out
    github.files[dup] = b"alt art"
    _run()
    stats, old = _saved()
    assert stats["Mini Doomer2"].archived is None and "Mini Doomer2" not in old             # back: the record goes
    del github.files[DOOMER]
    _run()
    stats, old = _saved()
    assert set(stats) == {"The Freezer"} and old["Mini Doomer2"][0].archived["at"] == old["Mini Doomer"][0].archived["at"]


def test_a_rename_keeps_the_cards_author_and_series(github, psds, capsys):
    """An exact rename (same bytes, new name): the new card is parsed as the old one, so the author and series curated
    on it stay (never the renamer's); its name-keyed side-file entries are reported under the old name."""
    _run()
    stats, _ = _saved()
    assert stats["Mini Doomer"].author == "Bob"
    stats["Mini Doomer"].series = "Doom"
    with open("stats.pkl", "wb") as f:
        pickle.dump(stats, f)
    github.files["Auxiliary/Minions/Mega_Doomer.psd"] = github.files.pop(DOOMER)
    capsys.readouterr()
    _run()
    stats, old = _saved()
    assert (stats["Mega Doomer"].author, stats["Mega Doomer"].series) == ("Bob", "Doom")
    assert old["Mini Doomer"][0].archived["renamed_to"] == "Mega Doomer"
    out = capsys.readouterr().out
    assert "Name-keyed entries left under archived names" in out and "Mini Doomer (now Mega Doomer): Card Authors.txt" in out
