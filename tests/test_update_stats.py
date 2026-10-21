"""update_stats / RepositoryTraverser against a fake GitHub (tree + blobs through a patched requests.get)."""
import hashlib
import json
import os
import pickle

import pytest
from requests import HTTPError

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


class FakeGitHub:
    """files: repo path -> bytes. tree_status / broken blobs simulate outages. calls: what was requested."""
    def __init__(self, files: dict):
        self.files = dict(files)
        self.tree_status = 200
        self.broken: set = set()
        self.calls: list = []

    def tree(self) -> dict:
        return {"tree": [{"path": p, "sha": _sha(d), "type": "blob"} for p, d in sorted(self.files.items())]}

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
    assert queued == ["_refresh_links", "sync_changed_emojis", "populate_files", "prep_dataframes"]   # no TTS save run
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
    assert queued == ["populate_files", "prep_dataframes"]
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


def test_moved_card(github, capsys):
    _run()
    github.files["Field/2 Stars/The_Freezer.psd"] = github.files.pop(FREEZER)
    github.calls.clear()
    _run()
    stats, old = _saved()
    assert stats["The Freezer"].path == "Field/2 Stars/The_Freezer.psd" and stats["The Freezer"].stars == 2
    assert [c.path for c in old["The Freezer"]] == ["Field/2 Stars/The_Freezer.psd"]   # history follows the move
    assert [c.stars for c in old["The Freezer"]] == [1]
    assert "0 cards changed or new, 1 unchanged, 1 moved." in capsys.readouterr().out


def test_deleted_card_pruned_with_a_change(github, psds):
    _run()
    del github.files[FREEZER]
    github.files[DOOMER] = psds[OTHER]
    _run()
    stats, old = _saved()
    assert set(stats) == {"Mini Doomer"} and len(old["The Freezer"]) == 1


def test_force_update_reparses_without_history(github):
    _run()
    github.calls.clear()
    _run(force_update=True)
    assert github.calls[0] == ("tree", None) and sorted(github.blobs()) == sorted([DOOMER, FREEZER])
    assert _saved()[1] == {}                                          # same blobs: nothing archived


def test_only_cards_reparsed(github):
    _run()
    github.calls.clear()
    _run(cards="mini_doomer")
    assert github.calls[0] == ("tree", None) and github.blobs() == [DOOMER]


def test_tree_5xx_raises_and_saves_nothing(github):
    """Current behaviour (BUGS B9): the error escapes update_stats."""
    github.tree_status = 502
    with pytest.raises(HTTPError):
        _run()
    assert not os.path.exists("stats.pkl") and rt._tree_etags == {}


def test_blob_5xx_skips_the_card(github, psds, capsys):
    """Current behaviour (BUGS B9): the card keeps its old parse and the tree ETag is still recorded."""
    _run()
    github.files[DOOMER] = psds[OTHER]
    github.broken.add(_sha(psds[OTHER]))
    _run()
    assert "download failed" in capsys.readouterr().out
    stats, _ = _saved()
    assert stats["Mini Doomer"].blob_sha == _sha(psds[DOOMER])
    github.calls.clear()
    _run()
    assert github.blobs() == []                                       # 304: stays stale until the repo changes


def test_deleted_card_alone_is_kept(github):
    """Current behaviour (BUGS B8): pruning needs a changed card, so a deletion alone leaves the card listed."""
    _run()
    del github.files[FREEZER]
    _run()
    stats, old = _saved()
    assert set(stats) == {"Mini Doomer", "The Freezer"} and not old


def test_old_mdw_paths_still_read(github):
    """Until the card repo moves them, the authors and edits files are read from MDW/ (the second of the two paths)."""
    old_authors, old_edits = pa.AUTHORS_PATHS[1], pa.EDITS_PATHS[1]
    assert (old_authors, old_edits) == ("MDW/Card Authors.txt", "MDW/Card Edits.json")
    github.files[old_authors] = github.files.pop(pa.AUTHORS_PATH)
    github.files[old_edits] = github.files.pop(pa.EDITS_PATH)
    assert _run()[0] == []
    stats, _ = _saved()
    assert (stats["Mini Doomer"].author, stats["The Freezer"].note) == ("Bob", "checked")
    assert {old_authors, old_edits} <= set(github.blobs()) and not {pa.AUTHORS_PATH, pa.EDITS_PATH} & set(github.blobs())


def test_site_paths_win_over_mdw(github):
    """Both layouts present (mid-move): site/ is read, MDW/ is not even downloaded."""
    github.files[pa.AUTHORS_PATHS[1]] = b"[people]\nbobgit = Old Bob\n[cards]\nmini_doomer = bobgit\n"
    github.files[pa.EDITS_PATHS[1]] = json.dumps({"cards": {"The Freezer": {"note": "old"}}}).encode()
    assert _run()[0] == []
    stats, _ = _saved()
    assert (stats["Mini Doomer"].author, stats["The Freezer"].note) == ("Bob", "checked")
    assert not {pa.AUTHORS_PATHS[1], pa.EDITS_PATHS[1]} & set(github.blobs())
