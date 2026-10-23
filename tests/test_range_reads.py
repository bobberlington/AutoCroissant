"""Range reads: a card PSD opened by HTTP range requests at the tree's commit (psd-tools' lazy open) parses as the whole
file does; a refused or broken range read, or a branch that moved, falls back to the whole blob. Also the parse from an
open PSD and the bound on fetches ahead of the parse (both without the fork)."""
import os
import tempfile
import threading
import time
from types import SimpleNamespace
from urllib.parse import unquote

import pytest
from psd_tools import PSDImage
from requests.exceptions import ConnectionError as NoConnection

import commands.card_check as cc
import commands.card_data as cd
import commands.github_api as ga
from global_config import CARD_BRANCH, RAW_BASE
from test_update_stats import (DOOMER, FREEZER, OTHER, Reply, _golden, _run, _saved, _sha, _strip,  # noqa: F401
                               github, psds)

COMMIT, TREE = "c" * 40, "t" * 40
needs_ranges = pytest.mark.skipif(not ga.range_reads(), reason="psd-tools without lazy reads (the fork's feat/lazy-records)")


@pytest.fixture
def ranged(github, monkeypatch):
    """The fake GitHub with its tree's sha, the branch head and raw.githubusercontent answering Range requests at
    COMMIT. state.whole: paths answered 200 (whole); state.refused: path -> the 4xx its range requests get;
    state.broken: paths whose range reads from the main thread (the parse, after the open in a download worker) answer
    503 (state.parse_reads lists those reads), state.short: a 206 10 bytes short; state.head_tree: the branch head's
    tree; state.down: every range request answers this status, or raises ConnectionError when it is 0."""
    state = SimpleNamespace(whole=set(), refused={}, broken=set(), short=set(), head_tree=TREE, parse_reads=[], down=None)
    tree = github.tree
    monkeypatch.setattr(github, "tree", lambda: {**tree(), "sha": TREE})
    inner = github.get

    def get(url, headers=None, timeout=None, stream=False):
        headers = headers or {}
        if url.endswith(f"/branches/{CARD_BRANCH}"):
            github.calls.append(("branch", None))
            return Reply(200, tree={"commit": {"sha": COMMIT, "commit": {"tree": {"sha": state.head_tree}}}})
        if not url.startswith(RAW_BASE):
            return inner(url, headers=headers, timeout=timeout, stream=stream)
        _, _, ref, quoted = url[len(RAW_BASE):].split("/", 3)
        path = unquote(quoted)
        main = threading.current_thread() is threading.main_thread()
        github.calls.append(("range", path))
        if main:
            state.parse_reads.append(path)
        if state.down is not None:
            if not state.down:
                raise NoConnection(f"{RAW_BASE} unreachable")
            return Reply(state.down)
        data = github.files.get(path)
        if ref != COMMIT or data is None:
            return Reply(404)
        if path in state.whole:
            return Reply(200, data)
        if path in state.refused:
            return Reply(state.refused[path])
        if path in state.broken and main:
            return Reply(503)
        start, end = (int(x) for x in headers["Range"].removeprefix("bytes=").split("-"))
        end = min(end, len(data) - 1) - (10 if path in state.short and main else 0)
        reply = Reply(206, data[start:end + 1])
        reply.headers["Content-Range"] = f"bytes {start}-{end}/{len(data)}"
        return reply

    monkeypatch.setattr(ga, "get", get)
    return state


def _ranges(github) -> set:
    return {call[1] for call in github.calls if call[0] == "range"}


def _goldens_hold() -> None:
    stats, _ = _saved()
    assert _strip(stats["Mini Doomer"].to_dict()) == _strip(_golden(DOOMER))
    assert _strip(stats["The Freezer"].to_dict()) == _strip(_golden(FREEZER))


# ========================
# /update_stats (the legacy build)
# ========================
@needs_ranges
def test_psds_are_read_by_ranges_at_the_trees_commit(ranged, github, capsys):
    problems, _ = _run()
    assert problems == [] and _ranges(github) == {DOOMER, FREEZER}
    assert DOOMER not in github.blobs() and FREEZER not in github.blobs()     # no whole download
    _goldens_hold()
    assert f"2 PSDs read by range requests at {COMMIT[:8]}" in capsys.readouterr().out


@needs_ranges
def test_a_server_answering_whole_files_means_a_blob_download(ranged, github):
    ranged.whole = {DOOMER}
    _run()
    assert DOOMER in github.blobs() and FREEZER not in github.blobs()
    _goldens_hold()


@needs_ranges
def test_a_range_read_broken_during_the_parse_means_a_blob_download(ranged, github):
    ranged.broken = {DOOMER}
    _run()
    assert DOOMER in ranged.parse_reads                           # the parse read pixels (stars) after the open
    assert DOOMER in github.blobs() and FREEZER not in github.blobs()
    _goldens_hold()


@needs_ranges
def test_a_short_range_a_details_section_caught_means_a_blob_download(ranged, github, monkeypatch):
    """A 206 shorter than asked fails in RangeReader, after the fetch; psd_details' sections catch everything, so
    RemotePSD.failed is what reads the PSD whole (no error section saved)."""
    import commands.psd_details as pd
    real = pd.gather

    def gather(psd, *args):
        record = real(psd, *args)
        record["pixels"] = pd._guard(lambda: len(psd._record.image_data.data))     # a section that reads pixels
        return record

    monkeypatch.setattr(pd, "gather", gather)
    ranged.short = {FREEZER}
    _run()
    assert FREEZER in ranged.parse_reads and FREEZER in github.blobs() and DOOMER not in github.blobs()
    record = pd.load("psd_details.json")[_sha(github.files[FREEZER])]
    assert isinstance(record["pixels"], int) and not [v for v in record.values() if str(v).startswith("error:")]
    _goldens_hold()


@needs_ranges
def test_a_branch_that_moved_means_whole_downloads(ranged, github):
    ranged.head_tree = "x" * 40                                    # a push after the tree was read
    _run()
    assert not _ranges(github) and {DOOMER, FREEZER} <= set(github.blobs())
    _goldens_hold()


@needs_ranges
def test_one_refused_psd_leaves_range_reads_on(ranged, github, monkeypatch):
    monkeypatch.setattr(cd, "DOWNLOAD_WORKERS", 1)                 # fetched in path order: Mini Doomer first
    ranged.refused = {DOOMER: 416}                                 # e.g. an empty file
    _run()
    assert DOOMER in github.blobs() and FREEZER not in github.blobs() and FREEZER in _ranges(github)
    _goldens_hold()


@needs_ranges
def test_RANGE_REFUSALS_refused_psds_turn_range_reads_off(ranged, github, psds, monkeypatch):
    github.files[OTHER] = psds[OTHER]
    monkeypatch.setattr(cd, "DOWNLOAD_WORKERS", 1)
    monkeypatch.setattr(cd, "RANGE_REFUSALS", 2)
    ranged.refused = {OTHER: 404, DOOMER: 404}                     # e.g. a commit not on GitHub
    _run()
    assert _ranges(github) == {OTHER, DOOMER} and {OTHER, DOOMER, FREEZER} <= set(github.blobs())
    _goldens_hold()


@needs_ranges
@pytest.mark.parametrize("down", [503, 0])
def test_raw_down_turns_range_reads_off_after_RANGE_REFUSALS(ranged, github, psds, monkeypatch, down):
    """A 5xx or no connection on raw (the API still up) counts as a refusal: each PSD costs get_with_retry's 3 tries
    only until RANGE_REFUSALS of them, then the run downloads whole."""
    github.files[OTHER] = psds[OTHER]
    monkeypatch.setattr(cd, "DOWNLOAD_WORKERS", 1)
    monkeypatch.setattr(cd, "RANGE_REFUSALS", 2)
    ranged.down = down
    _run()
    assert _ranges(github) == {OTHER, DOOMER} and {OTHER, DOOMER, FREEZER} <= set(github.blobs())
    assert [c for c in github.calls if c == ("range", OTHER)] == [("range", OTHER)] * 3     # get_with_retry's tries
    _goldens_hold()


def test_the_off_switch_covers_the_build_and_card_check(ranged, github, monkeypatch):
    monkeypatch.setattr(ga, "RANGE_READS", False)
    _run()
    assert not _ranges(github) and ("branch", None) not in github.calls
    pushed = {DOOMER: (_sha(github.files[DOOMER]), "bob"), FREEZER: (_sha(github.files[FREEZER]), "ann")}
    github.calls.clear()
    cc.check_blobs("o/r", pushed, cc.Vocabulary().parser(), None, ref=COMMIT)
    assert not _ranges(github) and {DOOMER, FREEZER} <= set(github.blobs())


def test_without_lazy_reads_the_psds_are_downloaded_whole(ranged, github, monkeypatch):
    monkeypatch.setattr(cd, "range_reads", lambda: False)
    _run()
    assert not _ranges(github) and ("branch", None) not in github.calls
    assert {DOOMER, FREEZER} <= set(github.blobs())
    _goldens_hold()


# ========================
# card_check --repo
# ========================
@needs_ranges
def test_card_check_reads_ranges_at_the_push(ranged, github):
    pushed = {DOOMER: (_sha(github.files[DOOMER]), "bob"), FREEZER: (_sha(github.files[FREEZER]), "ann")}
    parser = cc.Vocabulary().parser()
    whole = cc.check_blobs("o/r", pushed, parser, None)
    assert {DOOMER, FREEZER} <= set(github.blobs()) and not _ranges(github)
    github.calls.clear()
    ranges = cc.check_blobs("o/r", pushed, parser, None, ref=COMMIT)
    assert ranges.as_dict() == whole.as_dict() and ranges.checked == 2
    assert _ranges(github) == {DOOMER, FREEZER} and not github.blobs()
    github.calls.clear()
    ranged.whole = {FREEZER}
    assert cc.check_blobs("o/r", pushed, parser, None, ref=COMMIT).as_dict() == whole.as_dict()
    assert github.blobs() == [FREEZER]


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


def test_fetches_run_at_most_AHEAD_before_the_parse(github, psds, monkeypatch):
    github.files[OTHER] = psds[OTHER]
    monkeypatch.setattr(cd, "AHEAD", 1)
    events = []
    fetch, parse = cd.CardBuild._fetch_psd, cd.CardBuild._parse

    def fetching(self, path, sha):
        events.append(("fetch", path))
        return fetch(self, path, sha)

    def parsing(self, source, path, name, sha):
        events.append(("parse", path))
        if path == OTHER:
            time.sleep(0.5)                                        # the first card's parse is slow
        card = parse(self, source, path, name, sha)
        events.append(("parsed", path))
        return card

    monkeypatch.setattr(cd.CardBuild, "_fetch_psd", fetching)
    monkeypatch.setattr(cd.CardBuild, "_parse", parsing)
    _run()
    assert [p for kind, p in events if kind == "fetch"] == [OTHER, DOOMER, FREEZER]
    assert events.index(("fetch", FREEZER)) > events.index(("parsed", OTHER))


# ========================
# The build (GitBuild on a blobless clone)
# ========================
import test_card_data as tcd  # noqa: E402

cards = tcd.cards


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
    assert tcd._build(cards, "--seed", str(cards.seed))[0] == cd.EXIT_OK
    stats = tcd._read(cards)[0]
    assert _strip(stats["Mini Doomer"].to_dict()) == _strip(_golden(DOOMER))
    assert _strip(stats["The Freezer"].to_dict()) == _strip(_golden(FREEZER))
    assert {path for _, path in served} == {DOOMER, FREEZER} and {ref for ref, _ in served} == {cards.first}


@needs_ranges
def test_a_baked_build_reads_a_blob_new_at_its_path_whole(cards, monkeypatch):
    """B puts an existing blob at another path (git mv -f): the parent has other bytes there, so no range read."""
    tcd._build(cards, "--seed", str(cards.seed))
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
    tcd._build(cards, "--seed", str(cards.seed))
    tcd.put(cards.repo, DOOMER, cards.psds[OTHER])
    baked = tcd.commit(cards.repo, "AutoCroissant", 2000)          # B: local, not on GitHub
    _hide(monkeypatch, {_sha(cards.psds[FREEZER])})
    served: list = []
    monkeypatch.setattr(ga, "get", _raw_from_git(cards.repo, served))
    assert tcd._build(cards, "--head", baked, "--baked", "--full")[0] == cd.EXIT_OK
    assert served and set(served) == {(cards.first, FREEZER)}      # B's own PSD came from the clone
    assert _strip(tcd._read(cards)[0]["The Freezer"].to_dict()) == _strip(_golden(FREEZER))
