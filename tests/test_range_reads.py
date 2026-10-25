"""Range reads: a card PSD opened by HTTP range requests (psd-tools' lazy open) parses as the whole file does; a broken
range read marks the PSD failed, so the caller reads it whole. card_check reads ranges at the push, the build at its head
(or a bake commit's parent). Also the parse from an open PSD and the bound on fetches ahead of the parse."""
import hashlib
import json
import os
import tempfile
import time
from types import SimpleNamespace
from urllib.parse import unquote

import pytest
from psd_tools import PSDImage
from requests import HTTPError

import commands.card_check as cc
import commands.card_data as cd
import commands.github_api as ga
from global_config import RAW_BASE
from golden_cards import golden_file
import test_card_data as tcd

cards = tcd.cards

DOOMER = "Auxiliary/Minions/Mini_Doomer.psd"
FREEZER = "Field/1 Stars/The_Freezer.psd"
OTHER = "Auxiliary/Minions/Anubisath_Guardian.psd"     # its bytes stand in for an edited Mini Doomer
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
def test_the_parser_takes_an_open_psd(psds, tmp_path):
    path = tmp_path / "card.psd"
    path.write_bytes(psds[DOOMER])
    parser = cc.Vocabulary().parser()
    assert parser.parse(PSDImage.open(path), DOOMER).to_dict() == parser.parse(str(path), DOOMER).to_dict()


def test_fetches_run_at_most_AHEAD_before_the_parse(cards, monkeypatch):
    monkeypatch.setattr(cd, "AHEAD", 1)
    events = []
    fetch, parse = cd.GitBuild._fetch_psd, cd.CardBuild._parse

    def fetching(self, path, sha):
        events.append(("fetch", path))
        return fetch(self, path, sha)

    def parsing(self, source, path, name, sha):
        events.append(("parse", path))
        if path == DOOMER:
            time.sleep(0.5)                                        # the first card's parse is slow
        card = parse(self, source, path, name, sha)
        events.append(("parsed", path))
        return card

    monkeypatch.setattr(cd.GitBuild, "_fetch_psd", fetching)
    monkeypatch.setattr(cd.CardBuild, "_parse", parsing)
    third = "Items/Anubisath_Guardian.psd"
    tcd.put(cards.repo, third, cards.psds[OTHER])
    tcd.commit(cards.repo, "alice", 1100)
    assert tcd._build(cards, "--empty")[0] == cd.EXIT_OK
    assert [p for kind, p in events if kind == "fetch"] == [DOOMER, FREEZER, third]
    assert events.index(("fetch", third)) > events.index(("parsed", DOOMER))


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
        data = tcd.git_bytes(repo, "show", f"{ref}:{path}")
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
    assert tcd._build(cards, "--empty")[0] == cd.EXIT_OK
    stats = tcd._read(cards)[0]
    assert _strip(stats["Mini Doomer"].to_dict()) == _strip(_golden(DOOMER))
    assert _strip(stats["The Freezer"].to_dict()) == _strip(_golden(FREEZER))
    assert {path for _, path in served} == {DOOMER, FREEZER} and {ref for ref, _ in served} == {cards.first}


@needs_ranges
def test_a_baked_build_reads_a_blob_new_at_its_path_whole(cards, monkeypatch):
    """B puts an existing blob at another path (git mv -f): the parent has other bytes there, so no range read."""
    tcd._build(cards, "--empty")
    tcd.put(cards.repo, FREEZER, cards.psds[DOOMER])
    baked = tcd.commit(cards.repo, "AutoCroissant", 2000)
    _hide(monkeypatch, {_sha(cards.psds[DOOMER])})
    served, whole = [], []
    monkeypatch.setattr(ga, "get", _raw_from_git(cards.repo, served))

    def download(self, path, sha):
        whole.append(path)
        fd, name = tempfile.mkstemp(suffix=".psd")
        with os.fdopen(fd, "wb") as fh:
            fh.write(tcd.git_bytes(cards.repo, "cat-file", "blob", sha))
        return name

    monkeypatch.setattr(cd.CardBuild, "_download", download)
    assert tcd._build(cards, "--head", baked, "--baked", "--full")[0] == cd.EXIT_OK
    assert set(served) == {(cards.first, DOOMER)} and whole == [FREEZER]
    assert tcd._read(cards)[0]["The Freezer"].ability == _golden(DOOMER)["ability"]


@needs_ranges
def test_a_baked_build_reads_ranges_at_the_bake_commits_parent(cards, monkeypatch):
    tcd._build(cards, "--empty")
    tcd.put(cards.repo, DOOMER, cards.psds[OTHER])
    baked = tcd.commit(cards.repo, "AutoCroissant", 2000)          # B: local, not on GitHub
    _hide(monkeypatch, {_sha(cards.psds[FREEZER])})
    served: list = []
    monkeypatch.setattr(ga, "get", _raw_from_git(cards.repo, served))
    assert tcd._build(cards, "--head", baked, "--baked", "--full")[0] == cd.EXIT_OK
    assert served and set(served) == {(cards.first, FREEZER)}      # B's own PSD came from the clone
    assert _strip(tcd._read(cards)[0]["The Freezer"].to_dict()) == _strip(_golden(FREEZER))
