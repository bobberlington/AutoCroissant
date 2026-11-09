"""Range reads: a card PSD opened by HTTP range requests (psd-tools' lazy open) parses as the whole file does; a broken
range read marks the PSD failed, so the caller reads it whole. card_check reads ranges at the push, the build at its head
(or a bake commit's parent). Also the parse from an open PSD and the download workers' parse (AHEAD)."""
import hashlib
import json
import os
import threading
import time
from types import SimpleNamespace
from urllib.parse import unquote

import pytest
from requests import HTTPError

import commands.card_check as cc
import commands.card_data as cd
import commands.github_api as ga
from global_config import RAW_BASE
from golden_cards import golden_file
import helpers
from helpers import DOOMER, FREEZER, OTHER, cards, cards_template  # noqa: F401 (fixtures)
COMMIT = "c" * 40
needs_ranges = pytest.mark.skipif(not ga.range_reads(), reason="psd-tools without lazy reads (the fork's feat/lazy-records)")


def _sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _golden(rel):
    return json.loads(golden_file(rel).read_text(encoding="utf-8"))["card"]


def _strip(d):
    return {k: v for k, v in d.items() if k not in ("timestamp", "author", "note")}


class Reply:
    def __init__(self, status=200, data=b""):
        self.status_code, self.content, self.headers, self.encoding = status, data, {}, None

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
    """GitHub's blobs (by sha) and raw.githubusercontent's range reads at COMMIT, for two PSDs. calls: (kind, path);
    whole: paths raw answers 200 (whole)."""
    fake = SimpleNamespace(files={DOOMER: psds[DOOMER], FREEZER: psds[FREEZER]}, calls=[], whole=set())

    def get(url, headers=None, timeout=None, stream=False):
        headers = headers or {}
        if url.startswith(RAW_BASE):
            _, _, ref, quoted = url[len(RAW_BASE):].split("/", 3)
            path = unquote(quoted)
            fake.calls.append(("range", path))
            data = fake.files.get(path)
            if ref != COMMIT or data is None:
                return Reply(404)
            if path in fake.whole:
                return Reply(200, data)
            start, end = (int(x) for x in headers["Range"].removeprefix("bytes=").split("-"))
            end = min(end, len(data) - 1)
            reply = Reply(206, data[start:end + 1])
            reply.headers["Content-Range"] = f"bytes {start}-{end}/{len(data)}"
            return reply
        sha = url.rsplit("/", 1)[1]
        path = next((p for p, d in fake.files.items() if _sha(d) == sha), None)
        fake.calls.append(("blob", path))
        return Reply(200, fake.files[path]) if path is not None else Reply(503)

    monkeypatch.setattr(ga, "get", get)
    monkeypatch.setattr(ga, "sleep", lambda seconds: None)
    return fake


def _blobs(github) -> list:
    return [p for kind, p in github.calls if kind == "blob"]


def _ranges(github) -> set:
    return {call[1] for call in github.calls if call[0] == "range"}


# ========================
# card_check --repo
# ========================
@needs_ranges
def test_card_check_reads_ranges_at_the_push(github):
    pushed = {DOOMER: (_sha(github.files[DOOMER]), "bob"), FREEZER: (_sha(github.files[FREEZER]), "ann")}
    parser = cc.Vocabulary().parser()
    whole = cc.check_blobs("o/r", pushed, parser, None)
    assert {DOOMER, FREEZER} <= set(_blobs(github)) and not _ranges(github)
    github.calls.clear()
    ranges = cc.check_blobs("o/r", pushed, parser, None, ref=COMMIT)
    assert ranges.as_dict() == whole.as_dict() and ranges.checked == 2
    assert _ranges(github) == {DOOMER, FREEZER} and not _blobs(github)
    github.calls.clear()
    github.whole = {FREEZER}
    assert cc.check_blobs("o/r", pushed, parser, None, ref=COMMIT).as_dict() == whole.as_dict()
    assert _blobs(github) == [FREEZER]


def test_a_clone_is_read_first_and_kept_filled(github, tmp_path):
    """--clone (the guards' check job): whole blobs, no ranges; each fetched once into the clone, then read from it."""
    import subprocess
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    pushed = {DOOMER: (_sha(github.files[DOOMER]), "bob")}
    parser = cc.Vocabulary().parser()
    first = cc.check_blobs("o/r", pushed, parser, None, ref=COMMIT, clone=str(tmp_path))
    assert _blobs(github) == [DOOMER] and not _ranges(github)
    github.calls.clear()
    assert cc.check_blobs("o/r", pushed, parser, None, ref=COMMIT, clone=str(tmp_path)).as_dict() == first.as_dict()
    assert ga.fetch_verified("o/r", pushed[DOOMER][0], clone=str(tmp_path)) == github.files[DOOMER] and not github.calls


# ========================
# RemotePSD
# ========================
def _short(data: bytes, start: int, end: int):
    """A 206 for bytes start-end with the last 10 bytes cut from the Content-Range and the body."""
    reply = Reply(206, data[start:end - 9])
    reply.headers["Content-Range"] = f"bytes {start}-{end - 10}/{len(data)}"
    return reply


def _cut(data: bytes, start: int, end: int):
    """A 206 whose Content-Range is the one asked for and whose body lacks its last 10 bytes."""
    reply = Reply(206, data[start:end - 9])
    reply.headers["Content-Range"] = f"bytes {start}-{end}/{len(data)}"
    return reply


def _grown(data: bytes, start: int, end: int):
    """A 206 for the range asked, from a file that grew by a byte."""
    reply = Reply(206, data[start:end + 1])
    reply.headers["Content-Range"] = f"bytes {start}-{end}/{len(data) + 1}"
    return reply


@needs_ranges
@pytest.mark.parametrize("answer", [_short, _cut, _grown])
def test_a_wrong_range_after_the_open_marks_the_psd_failed(psds, monkeypatch, answer):
    """psd_details catches what a section raises: failed is what makes the caller read the PSD whole."""
    data, opened = psds[DOOMER], []

    def get(url, headers=None, timeout=None, stream=False):
        start, end = (int(x) for x in headers["Range"].removeprefix("bytes=").split("-"))
        end = min(end, len(data) - 1)
        if opened:
            return answer(data, start, end)
        reply = Reply(206, data[start:end + 1])
        reply.headers["Content-Range"] = f"bytes {start}-{end}/{len(data)}"
        return reply

    monkeypatch.setattr(ga, "get", get)
    remote = ga.RemotePSD("o/r", COMMIT, DOOMER, len(data))
    opened.append(True)
    assert not remote.failed
    remote.reader.seek(len(data) - 100)                            # pixels: not fetched by the open
    with pytest.raises(OSError):
        remote.reader.read(100)
    assert remote.failed
    remote.close()


# ========================
# Without the fork
# ========================
def test_psds_are_read_in_the_workers_and_kept_at_most_AHEAD_ahead(cards, monkeypatch):
    monkeypatch.setattr(cd, "AHEAD", 2)
    events, waited, freezer = [], [], threading.Event()
    read = cd.CardBuild._read

    def reading(self, source, path, name, sha):
        events.append(("read", path))
        if path == DOOMER:
            waited.append(freezer.wait(10))                        # FREEZER is read meanwhile, by another worker
        found = read(self, source, path, name, sha)
        if path == FREEZER:
            freezer.set()
        events.append(("done", path))
        return found

    monkeypatch.setattr(cd.CardBuild, "_read", reading)
    third = "Items/Anubisath_Guardian.psd"
    helpers.put(cards.repo, third, cards.psds[OTHER])
    helpers.commit(cards.repo, "alice", 1100)
    assert helpers._build(cards, "--empty")[0] == cd.EXIT_OK
    assert waited == [True]
    assert events.index(("read", third)) > events.index(("done", DOOMER))
    stats = helpers._read(cards)[0]
    assert _strip(stats["Mini Doomer"].to_dict()) == _strip(_golden(DOOMER))
    assert _strip(stats["The Freezer"].to_dict()) == _strip(_golden(FREEZER))


def test_forked_workers_read_as_threads_do(cards, monkeypatch, capsys):
    """A runner's forked workers: each PSD's Read comes back (pickled) as a thread's; one that hangs past FORK_WAIT is
    killed and its PSD read here."""
    reads, fetch, keep, parent = {}, cd.CardBuild._fetch_read, cd.CardBuild._keep, os.getpid()
    hang = []

    def fetched(self, path, name, sha):
        if path in hang and os.getpid() != parent:
            time.sleep(60)
        read = fetch(self, path, name, sha)
        read.pid = os.getpid()
        return read

    def kept(self, read, path, name, sha):
        pid = vars(read).pop("pid")
        reads[path] = (pid != os.getpid(), read.card, read.variants, {**read.details, "at": ""}, read.links, read.error)
        return keep(self, read, path, name, sha)

    monkeypatch.setattr(cd.CardBuild, "_fetch_read", fetched)
    monkeypatch.setattr(cd.CardBuild, "_keep", kept)
    monkeypatch.setenv("no_proxy", "*")                           # macOS: no proxy lookup in a fork
    monkeypatch.setattr(cd, "FORK_WAIT", 2)
    found = []
    for forked, hung in ((False, []), (True, []), (True, [FREEZER])):
        hang[:] = hung
        monkeypatch.setattr(cd, "_forks", lambda: forked)
        cards.db = cards.tmp / f"data{forked}{len(hung)}" / "db"
        cards.db.mkdir(parents=True)
        reads.clear()
        assert helpers._build(cards, "--empty")[0] == cd.EXIT_OK
        found.append(dict(reads))
    assert {p: r[0] for p, r in found[1].items()} == {DOOMER: True, FREEZER: True} and not found[2][FREEZER][0]
    assert {p: r[1:] for p, r in found[0].items()} == {p: r[1:] for p, r in found[1].items()} == {p: r[1:] for p, r in found[2].items()}
    assert capsys.readouterr().out.count("Forked reads stopped (TimeoutError()); the rest in threads") == 1


# ========================
# The build (GitBuild on a blobless clone)
# ========================


def _raw_from_git(repo, served: list):
    """raw.githubusercontent served from the repo's git (any other request fails the test)."""
    def get(url, headers=None, timeout=None, stream=False):
        assert url.startswith(RAW_BASE), f"not a raw request: {url}"
        _, _, ref, quoted = url[len(RAW_BASE):].split("/", 3)
        path = unquote(quoted)
        served.append((ref, path))
        data = helpers.git_bytes(repo, "show", f"{ref}:{path}")
        start, end = (int(x) for x in headers["Range"].removeprefix("bytes=").split("-"))
        end = min(end, len(data) - 1)
        reply = Reply(206, data[start:end + 1])
        reply.headers["Content-Range"] = f"bytes {start}-{end}/{len(data)}"
        return reply
    return get


def _hide(monkeypatch, shas: set) -> None:
    real = cd.GitBuild._store
    monkeypatch.setattr(cd.GitBuild, "_store", lambda self, sha: "" if sha in shas else real(self, sha))


@needs_ranges
def test_the_build_reads_blobs_its_clone_lacks_by_ranges_at_head(cards, monkeypatch):
    _hide(monkeypatch, {_sha(data) for data in cards.psds.values()})
    served: list = []
    monkeypatch.setattr(ga, "get", _raw_from_git(cards.repo, served))
    monkeypatch.setattr(cd.CardBuild, "_download", lambda self, path, sha: pytest.fail(f"{path} downloaded whole"))
    assert helpers._build(cards, "--empty")[0] == cd.EXIT_OK
    stats = helpers._read(cards)[0]
    assert _strip(stats["Mini Doomer"].to_dict()) == _strip(_golden(DOOMER))
    assert _strip(stats["The Freezer"].to_dict()) == _strip(_golden(FREEZER))
    assert {path for _, path in served} == {DOOMER, FREEZER} and {ref for ref, _ in served} == {cards.first}


@needs_ranges
def test_a_baked_build_reads_ranges_at_the_bake_commits_parent(cards, monkeypatch):
    helpers._build(cards, "--empty")
    helpers.put(cards.repo, DOOMER, cards.psds[OTHER])
    baked = helpers.commit(cards.repo, "AutoCroissant", 2000)          # B: local, not on GitHub
    _hide(monkeypatch, {_sha(cards.psds[FREEZER])})
    served: list = []
    monkeypatch.setattr(ga, "get", _raw_from_git(cards.repo, served))
    assert helpers._build(cards, "--head", baked, "--baked", "--full")[0] == cd.EXIT_OK
    assert served and set(served) == {(cards.first, FREEZER)}      # B's own PSD came from the clone
    assert _strip(helpers._read(cards)[0]["The Freezer"].to_dict()) == _strip(_golden(FREEZER))


@needs_ranges
def test_other_psds_links_read_in_the_workers_as_one_at_a_time(cards, clone, monkeypatch):
    """gather_links reads MDW PSDs and .psb files by ranges in the download workers: links.json as one worker gives."""
    rel = "Auxiliary/Creatures/Militron_(No_Armor).psd"
    if not (clone / rel).exists():
        pytest.skip("no Militron (No Armor)")
    linked = (clone / rel).read_bytes()
    others = {"MDW/Assets/A.psd": linked, "MDW/Linked/B - c.psb": cards.psds[DOOMER],
              "MDW/Templates/C.psd": cards.psds[FREEZER], "MDW/Unfinished/D.psd": linked}
    for path, data in others.items():
        helpers.put(cards.repo, path, data)
    helpers.put(cards.repo, "Types/unknown.png", b"unknown")
    helpers.commit(cards.repo, "bob", 2000)
    _hide(monkeypatch, {_sha(data) for data in others.values()})
    monkeypatch.setattr(ga, "get", _raw_from_git(cards.repo, []))
    monkeypatch.setattr(cd.CardBuild, "_download", lambda self, path, sha: pytest.fail(f"{path} downloaded whole"))
    linked_of, waited, second = cd.CardBuild._linked, [], threading.Event()

    def reading(self, psd, path, sha):
        if cd.DOWNLOAD_WORKERS > 1 and path == "MDW/Assets/A.psd":
            waited.append(second.wait(10))                       # B is read meanwhile, by another worker
        if cd.DOWNLOAD_WORKERS > 1 and path == "MDW/Linked/B - c.psb":
            second.set()
        return linked_of(self, psd, path, sha)

    monkeypatch.setattr(cd.CardBuild, "_linked", reading)
    found = []
    for workers in (1, 4):
        monkeypatch.setattr(cd, "DOWNLOAD_WORKERS", workers)
        cards.db = cards.tmp / f"data{workers}" / "db"
        cards.db.mkdir(parents=True)
        assert helpers._build(cards, "--empty")[0] == cd.EXIT_OK
        found.append(cd.read_links(str(cards.db)))
    assert waited == [True] and found[0] == found[1]
    assert {path: len(found[1][path].get("links", [])) for path in others} == {
        "MDW/Assets/A.psd": 1, "MDW/Linked/B - c.psb": 0, "MDW/Templates/C.psd": 0, "MDW/Unfinished/D.psd": 1}
