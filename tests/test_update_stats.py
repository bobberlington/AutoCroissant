"""update_stats / RepositoryTraverser against a fake GitHub (tree + blobs through a patched requests.get)."""
import hashlib
import json
import os
import pickle
from threading import Thread

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


def test_deleted_card_alone_is_pruned(github, capsys):
    """B8: a deletion with no other change prunes the card into old_stats, like the prune after a change."""
    _run()
    freezer = _saved()[0]["The Freezer"]
    del github.files[FREEZER]
    capsys.readouterr()
    _run()
    stats, old = _saved()
    assert set(stats) == {"Mini Doomer"} and [c.blob_sha for c in old["The Freezer"]] == [freezer.blob_sha]
    out = capsys.readouterr().out
    assert "Removed (PSD deleted from the card repo): The Freezer" in out
    assert "0 cards changed or new, 1 unchanged, 0 moved." in out
    _run()                                                            # 304: nothing pruned or saved again
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
    assert "Removed (PSD deleted from the card repo): Mini Doomer" in capsys.readouterr().out


def test_truncated_tree_prunes_nothing(github, capsys):
    _run()
    details = _details()
    del github.files[FREEZER]
    github.truncated = True
    capsys.readouterr()
    _run()
    stats, old = _saved()
    assert set(stats) == {"Mini Doomer", "The Freezer"} and old == {} and _details() == details
    assert "Card repo tree truncated or without cards; no card removed." in capsys.readouterr().out


def test_tree_without_cards_prunes_nothing(github, capsys):
    """A tree with no PSDs (say a /set_repo to another repo) moves no card to old_stats."""
    _run()
    del github.files[FREEZER], github.files[DOOMER]
    capsys.readouterr()
    _run()
    stats, old = _saved()
    assert set(stats) == {"Mini Doomer", "The Freezer"} and old == {}
    assert "no card removed" in capsys.readouterr().out


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
    the tree takes its record with it, and a 304 writes nothing."""
    import commands.font_check as fc
    import commands.psd_details as pd
    fonts = clone / fc.FONTS_DIR
    monkeypatch.setenv(fc.FONT_DIRS_ENV, str(fonts))                            # the clone's fonts, whatever the shell says
    _run()
    details = _details()
    assert set(details) == {_file_sha(github, DOOMER), _file_sha(github, FREEZER)}
    doomer = details[_file_sha(github, DOOMER)]
    assert (doomer["path"], doomer["name"], doomer["parser"]) == (DOOMER, "Mini Doomer", pa.PARSER_VERSION)
    assert doomer["stats"]["hp"] == "template" and doomer["icons"] == "images" and doomer["template"]["name_layer"] == "Text/Name"
    assert {e["supplied"] for e in doomer["fonts"].values()} == ({True} if fonts.is_dir() else {None})
    mtime = os.stat("psd_details.json").st_mtime_ns
    _run()                                                                      # 304
    assert os.stat("psd_details.json").st_mtime_ns == mtime
    github.files[DOOMER] = psds[OTHER]
    del github.files[FREEZER]
    _run()
    details = _details()
    assert set(details) == {_sha(psds[OTHER])} and details[_sha(psds[OTHER])]["path"] == DOOMER
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


def test_truncated_tree_refills_nothing(github, monkeypatch):
    """A truncated tree leaves psd_details.json alone: no prune, no refill."""
    import commands.psd_details as pd

    class Everything:
        def resolve(self, name):
            return "exact"
    monkeypatch.setattr(pd, "font_index", lambda repository, tree=None: None)
    _run()
    mtime = os.stat("psd_details.json").st_mtime_ns
    monkeypatch.setattr(pd, "font_index", lambda repository, tree=None: Everything())
    github.files["notes.txt"] = b"not a card"
    github.truncated = True
    _run()
    assert os.stat("psd_details.json").st_mtime_ns == mtime
    assert {e["supplied"] for e in _details()[_file_sha(github, DOOMER)]["fonts"].values()} == {None}


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
