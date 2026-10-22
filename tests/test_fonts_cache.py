"""
The fonts cache (font_check.refresh_cache / find_fonts) offline: a fake GitHub serving the MDW/Fonts listing with an
ETag and the font blobs; card_bake's fonts dir, psd_details' supplied flags and their --refill CLI, /fonts on the cache.
"""
import json
import os
from pathlib import Path

import pytest

import commands.card_bake as cb
import commands.font_check as fc
import commands.github_api as ga
import commands.psd_details as pd
import global_config
from conftest import REPO, Reply, blob_sha_bytes as _sha
from test_font_check import build_font


# ========================
# Fixtures
# ========================
@pytest.fixture(scope="module")
def font_bytes(tmp_path_factory):
    """Three small real fonts: name -> bytes."""
    d = tmp_path_factory.mktemp("cache-fonts")
    out = {}
    for ps in ("CacheFixture-A", "CacheFixture-B", "CacheFixture-C"):
        out[ps] = Path(build_font(str(d / f"{ps}.ttf"), ps, "Cache Fixture", ps.rsplit("-", 1)[1])).read_bytes()
    return out


class FakeFonts:
    """The subtree listing git/trees/<branch>:MDW/Fonts (ETag: the body's hash) and git/blobs/<sha>."""
    def __init__(self, files: dict):
        self.files = dict(files)                 # name -> bytes
        self.extra: list = []                    # non-font entries of the listing
        self.listing_status = 200
        self.missing: set = set()                # blob shas answered 404
        self.calls: list = []

    def body(self) -> dict:
        entries = [{"path": n, "type": "blob", "sha": _sha(d)} for n, d in sorted(self.files.items())] + self.extra
        return {"sha": _sha(json.dumps(entries).encode()), "tree": entries, "truncated": False}

    def get(self, url, headers=None, timeout=None, stream=False):
        headers = headers or {}
        if f"/git/trees/{global_config.CARD_BRANCH}:{fc.FONTS_DIR}" in url:
            self.calls.append(("list", headers.get("If-None-Match")))
            if self.listing_status != 200:
                return Reply(self.listing_status)
            body = self.body()
            etag = 'W/"%s"' % body["sha"]
            return Reply(304) if headers.get("If-None-Match") == etag else Reply(body=body, headers={"ETag": etag})
        sha = url.rsplit("/", 1)[1]
        self.calls.append(("blob", sha))
        data = next((d for d in self.files.values() if _sha(d) == sha), None)
        return Reply(404) if data is None or sha in self.missing else Reply(data=data)

    def blobs(self) -> int:
        return sum(kind == "blob" for kind, _ in self.calls)

    def lists(self) -> list:
        return [etag for kind, etag in self.calls if kind == "list"]


@pytest.fixture
def no_checkout(monkeypatch, tmp_path):
    """No PSD_TOOLS_FONT_DIRS and no card repo checkout: the jetson's case."""
    monkeypatch.delenv(fc.FONT_DIRS_ENV, raising=False)
    monkeypatch.setattr(fc, "LOCAL_DIR_LOC", str(tmp_path / "no-checkout"))
    monkeypatch.setattr(cb, "LOCAL_DIR_LOC", str(tmp_path / "no-checkout"))


@pytest.fixture
def fake(monkeypatch, font_bytes):
    fake = FakeFonts({"A.ttf": font_bytes["CacheFixture-A"], "B.ttf": font_bytes["CacheFixture-B"]})
    fake.extra = [{"path": "readme.txt", "type": "blob", "sha": _sha(b"readme")},
                  {"path": "stoogedings-src", "type": "tree", "sha": _sha(b"src")}]
    monkeypatch.setattr(ga, "get", fake.get)
    return fake


def _manifest(directory) -> dict:
    return json.loads((Path(directory) / fc.MANIFEST).read_text(encoding="utf-8"))


# ========================
# The refresh
# ========================
def test_tree_fonts_takes_the_top_level_font_files():
    blobs = {"MDW/Fonts/A.ttf": "a", "MDW/Fonts/B.OTF": "b", "MDW/Fonts/readme.txt": "r", "MDW/Fonts/stoogedings-src/x.ttf": "x",
             "MDW/Other/C.ttf": "c", "Creatures/A.psd": "p"}
    assert fc.tree_fonts(blobs) == {"A.ttf": "a", "B.OTF": "b"}


def test_first_fill_then_a_304(fake, font_bytes, tmp_path):
    first = fc.refresh_cache("o/r", directory="cache")
    assert (first.downloaded, first.copied, first.removed, first.requests, first.failed, first.error) == (2, 0, 0, 3, {}, "")
    assert first.directory == str(tmp_path / "cache") and first.tree == fake.body()["sha"]
    assert (tmp_path / "cache" / "A.ttf").read_bytes() == font_bytes["CacheFixture-A"]
    assert sorted(p.name for p in (tmp_path / "cache").iterdir()) == [fc.MANIFEST, "A.ttf", "B.ttf"]   # no temp files left
    saved = _manifest("cache")
    assert saved["tree"] == first.tree and saved["etag"] and set(saved["files"]) == {"A.ttf", "B.ttf"}
    second = fc.refresh_cache("o/r", directory="cache")
    assert fake.lists() == [None, saved["etag"]] and fake.blobs() == 2            # conditional: a 304, no download
    assert (second.requests, second.downloaded, second.files, second.tree) == (1, 0, first.files, first.tree)
    index = fc.FontIndex(second.directory)
    assert (index.files, index.resolve("CacheFixture-B")) == (2, "exact")


def test_a_changed_tree_downloads_only_new_blobs(fake, font_bytes, tmp_path):
    fc.refresh_cache("o/r", directory="cache")
    fake.files = {"Renamed.ttf": font_bytes["CacheFixture-A"],                   # A's blob under a new name: copied
                  "C.ttf": font_bytes["CacheFixture-C"]}                         # new: downloaded; B gone: deleted
    (tmp_path / "cache" / "Stray.otf").write_bytes(b"x")                         # not the cache's: deleted
    result = fc.refresh_cache("o/r", directory="cache")
    assert (result.downloaded, result.copied, result.removed, result.failed) == (1, 1, 3, {})
    assert fake.blobs() == 3 and result.requests == 2
    assert sorted(p.name for p in (tmp_path / "cache").iterdir()) == [fc.MANIFEST, "C.ttf", "Renamed.ttf"]
    assert _manifest("cache")["files"] == {"C.ttf": _sha(font_bytes["CacheFixture-C"]), "Renamed.ttf": _sha(font_bytes["CacheFixture-A"])}


def test_swapped_names_are_checked_not_trusted(fake, font_bytes, tmp_path):
    fc.refresh_cache("o/r", directory="cache")
    fake.files["A.ttf"], fake.files["B.ttf"] = fake.files["B.ttf"], fake.files["A.ttf"]
    result = fc.refresh_cache("o/r", directory="cache")
    assert (result.copied, result.downloaded) == (1, 1)                         # B's old bytes were already rewritten
    assert (tmp_path / "cache" / "A.ttf").read_bytes() == font_bytes["CacheFixture-B"]
    assert (tmp_path / "cache" / "B.ttf").read_bytes() == font_bytes["CacheFixture-A"]


def test_a_failed_download_keeps_the_old_file_and_retries(fake, font_bytes, tmp_path):
    fc.refresh_cache("o/r", directory="cache")
    fake.files["A.ttf"] = font_bytes["CacheFixture-C"]
    fake.missing.add(_sha(font_bytes["CacheFixture-C"]))
    result = fc.refresh_cache("o/r", directory="cache")
    assert result.failed == {"A.ttf": "404 Error"} and result.files["A.ttf"] == _sha(font_bytes["CacheFixture-A"])
    assert (tmp_path / "cache" / "A.ttf").read_bytes() == font_bytes["CacheFixture-A"]
    assert _manifest("cache")["etag"] == _manifest("cache")["tree"] == ""         # no 304 shortcut next time
    fake.missing.clear()
    again = fc.refresh_cache("o/r", directory="cache")
    assert fake.lists()[2] is None and (again.downloaded, again.failed) == (1, {})
    assert (tmp_path / "cache" / "A.ttf").read_bytes() == font_bytes["CacheFixture-C"]


def test_a_bad_blob_is_a_failure(fake, font_bytes, monkeypatch):
    real = fake.get
    monkeypatch.setattr(ga, "get", lambda url, **kw: Reply(data=b"garbage") if "/git/blobs/" in url else real(url, **kw))
    result = fc.refresh_cache("o/r", directory="cache")
    assert set(result.failed) == {"A.ttf", "B.ttf"} and "hashes to" in result.failed["A.ttf"] and result.files == {}


def test_a_failed_listing_leaves_the_cache(fake, tmp_path):
    fc.refresh_cache("o/r", directory="cache")
    fake.listing_status = 404
    result = fc.refresh_cache("o/r", directory="cache")
    assert result.error.startswith("MDW/Fonts listing: 404") and set(result.files) == {"A.ttf", "B.ttf"}
    assert (tmp_path / "cache" / "A.ttf").exists() and fake.blobs() == 2


def test_an_unfinished_refresh_is_not_complete_on_a_failed_listing(fake, no_checkout, font_bytes, tmp_path, capsys):
    """B failed on the last refresh, then the listing fails: the cache is incomplete, so no index (never a wrong False)."""
    fake.missing.add(_sha(font_bytes["CacheFixture-B"]))
    first = fc.refresh_cache("o/r", directory="cache")
    assert set(first.failed) == {"B.ttf"} and not first.complete and _manifest("cache")["complete"] is False
    fake.listing_status = 503
    second = fc.refresh_cache("o/r", directory="cache")
    assert second.error and second.failed == {} and not second.complete and set(second.files) == {"A.ttf"}
    assert pd.font_index("o/r", "cache") is None and "incomplete" in capsys.readouterr().out
    fake.listing_status = 200
    fake.missing.clear()
    third = fc.refresh_cache("o/r", directory="cache")
    assert third.complete and third.downloaded == 1 and _manifest("cache")["complete"] is True
    fake.listing_status = 503
    assert fc.refresh_cache("o/r", directory="cache").complete and pd.font_index("o/r", "cache").files == 2


def test_a_lost_file_is_not_complete_on_a_failed_listing(fake, tmp_path):
    fc.refresh_cache("o/r", directory="cache")
    (tmp_path / "cache" / "B.ttf").unlink()
    fake.listing_status = 503
    assert not fc.refresh_cache("o/r", directory="cache").complete


def test_a_case_only_rename_keeps_the_file(fake, font_bytes, tmp_path):
    fc.refresh_cache("o/r", directory="cache")
    fake.files["a.ttf"] = fake.files.pop("A.ttf")
    result = fc.refresh_cache("o/r", directory="cache")
    assert (result.failed, result.complete, result.removed) == ({}, True, 0 if (tmp_path / "CACHE").exists() else 1)
    assert sorted(p.name for p in (tmp_path / "cache").iterdir()) == [fc.MANIFEST, "B.ttf", "a.ttf"]
    assert fc.FontIndex(result.directory).resolve("CacheFixture-A") == "exact"
    assert (tmp_path / "cache" / "a.ttf").read_bytes() == font_bytes["CacheFixture-A"]


def test_a_lost_file_is_fetched_again(fake, tmp_path):
    fc.refresh_cache("o/r", directory="cache")
    (tmp_path / "cache" / "B.ttf").unlink()
    result = fc.refresh_cache("o/r", directory="cache")
    assert fake.lists()[1] is None and result.downloaded == 1 and (tmp_path / "cache" / "B.ttf").exists()
    (tmp_path / "cache" / fc.MANIFEST).write_text("{not json")
    assert fc.refresh_cache("o/r", directory="cache").downloaded == 2             # a corrupt manifest: all again


def test_the_callers_tree_needs_no_listing(fake, font_bytes):
    files = {"A.ttf": _sha(font_bytes["CacheFixture-A"])}
    result = fc.refresh_cache("o/r", files, "cache")
    assert fake.lists() == [] and (result.requests, result.downloaded, result.files) == (1, 1, files)
    assert fc.refresh_cache("o/r", files, "cache").requests == 0
    empty = fc.refresh_cache("o/r", {}, "cache")                                 # a tree without fonts never empties it
    assert empty.error and empty.files == files


def test_find_fonts_prefers_the_override_then_the_checkout(fake, monkeypatch, tmp_path, no_checkout):
    monkeypatch.setenv(fc.FONT_DIRS_ENV, os.pathsep.join(["/first", "/second"]))
    assert (fc.find_fonts("o/r").directory, fc.find_fonts("o/r").source) == ("/first", "env")
    monkeypatch.delenv(fc.FONT_DIRS_ENV)
    checkout = tmp_path / "TTSCardMaker"
    (checkout / fc.FONTS_DIR).mkdir(parents=True)
    monkeypatch.setattr(fc, "LOCAL_DIR_LOC", str(checkout))
    assert (fc.find_fonts("o/r").directory, fc.find_fonts("o/r").source) == (str(checkout / fc.FONTS_DIR), "checkout")
    assert fake.calls == []
    monkeypatch.setattr(fc, "LOCAL_DIR_LOC", str(tmp_path / "gone"))
    cache = fc.find_fonts("o/r", directory="cache")
    assert (cache.source, cache.directory, cache.downloaded) == ("cache", str(tmp_path / "cache"), 2)


# ========================
# The consumers
# ========================
def test_details_index_from_the_cache(fake, no_checkout, capsys):
    index = pd.font_index("o/r", "cache")
    assert index is not None and index.files == 2 and index.resolve("CacheFixture-A") == "exact"
    fake.listing_status = 404
    stale = pd.font_index("o/r", "cache")                                       # the listing failed: the cache stands
    assert stale is not None and stale.files == 2 and "MDW/Fonts listing" in capsys.readouterr().out


def test_details_index_is_none_without_fonts(fake, no_checkout, font_bytes, capsys):
    fake.listing_status = 404
    assert pd.font_index("o/r", "cache") is None                                # nothing cached yet
    fake.listing_status = 200
    fake.missing.add(_sha(font_bytes["CacheFixture-B"]))
    assert pd.font_index("o/r", "cache") is None                                # an incomplete set: null, never a wrong False
    assert "1 MDW/Fonts files failed" in capsys.readouterr().out


def test_gather_fills_supplied_from_the_cache(fake, no_checkout, clone):
    """The /update_stats gather on a host without a checkout: the flags come from the cache, no /fonts run needed."""
    index = pd.font_index("o/r", "cache")
    record = pd.gather(str(clone / "Creatures/Other/5 Stars/Baba.psd"), "x.psd", "Baba", index)
    assert record["fonts"] and {e["supplied"] for e in record["fonts"].values()} == {False}   # BabaUI is not a fixture font


def _records(font_bytes):
    return {"s1": {"path": "A.psd", "fonts": {"CacheFixture-A": {"layers": ["Text/Name"], "supplied": None},
                                              "Missing-Font": {"layers": ["Text/Ability"], "supplied": None}}},
            "s2": {"path": "B.psd", "fonts": {"CacheFixture-B": {"layers": ["Text/Name"], "supplied": True}}},
            "s3": {"path": "C.psd", "fonts": "error: unreadable"}}


def test_refill_sets_the_flags(fake, no_checkout, font_bytes):
    cards = _records(font_bytes)
    index = pd.font_index("o/r", "cache")
    assert pd.refill(cards, index) == 1
    assert cards["s1"]["fonts"]["CacheFixture-A"]["supplied"] is True and cards["s1"]["fonts"]["Missing-Font"]["supplied"] is False
    assert pd.refill(cards, index) == 0 and cards["s3"]["fonts"] == "error: unreadable"


def test_refill_cli(fake, no_checkout, font_bytes, tmp_path, capsys):
    """python -m commands.psd_details --refill: the fonts cache in the cwd (the bot's dir), the file written only on a change."""
    out = tmp_path / "details.json"
    pd.save(_records(font_bytes), str(out))
    assert pd.main(["--refill", "--out", str(out)]) == 0
    assert "1 of 3 records' supplied flags changed" in capsys.readouterr().out
    assert pd.load(str(out))["s1"]["fonts"]["Missing-Font"]["supplied"] is False
    assert (tmp_path / global_config.FONTS_CACHE / "A.ttf").exists()
    mtime = out.stat().st_mtime_ns
    assert pd.main(["--refill", "--out", str(out)]) == 0 and out.stat().st_mtime_ns == mtime   # nothing changed: not written
    assert pd.main(["--refill", "--out", str(out), "--fonts", str(tmp_path / "empty")]) == 1
    assert "left as it is" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        pd.main(["--refill", "--clone", str(tmp_path)])
    with pytest.raises(SystemExit):
        pd.main([])


def test_bake_fonts_dir(fake, no_checkout, font_bytes, monkeypatch, tmp_path):
    monkeypatch.setenv(cb.FONT_DIRS_ENV, "/runner/fonts")
    assert cb._fonts_dir() == "/runner/fonts"
    monkeypatch.delenv(cb.FONT_DIRS_ENV)
    assert cb._fonts_dir() == str(tmp_path / "no-checkout" / cb.psd_edit.FONTS_DIR)   # no snapshot: the old fallback
    tree = {f"{fc.FONTS_DIR}/{n}": ("100644", _sha(d)) for n, d in fake.files.items()}
    tree["Creatures/X.psd"] = ("100644", _sha(b"psd"))
    snapshot = cb.Snapshot("o/r", "head", {}, tree)
    assert cb._fonts_dir(snapshot) == str(tmp_path / global_config.FONTS_CACHE)
    assert fake.lists() == [] and fake.blobs() == 2                             # the snapshot's tree: no listing
    assert cb._fonts_dir(snapshot) == str(tmp_path / global_config.FONTS_CACHE) and fake.blobs() == 2
    tree[f"{fc.FONTS_DIR}/New.ttf"] = ("100644", _sha(b"gone"))
    with pytest.raises(cb.Transient, match="1 font file failed to download"):
        cb._fonts_dir(cb.Snapshot("o/r", "head", {}, tree))
    checkout = tmp_path / "TTSCardMaker"
    (checkout / cb.psd_edit.FONTS_DIR).mkdir(parents=True)
    monkeypatch.setattr(cb, "LOCAL_DIR_LOC", str(checkout))
    assert cb._fonts_dir(snapshot) == str(checkout / cb.psd_edit.FONTS_DIR)


def test_fonts_command_reuses_the_cached_files(fake, no_checkout, font_bytes, monkeypatch, tmp_path):
    """/fonts keeps the font files: a lost font_names.json costs no font download."""
    fonts = {n: _sha(d) for n, d in fake.files.items()}
    tree = {"tree": [{"path": f"{fc.FONTS_DIR}/{n}", "type": "blob", "sha": s} for n, s in fonts.items()]}
    real = fake.get
    monkeypatch.setattr(ga, "get", lambda url, **kw: Reply(body=tree) if "/git/trees/main?" in url else real(url, **kw))
    first = fc.repo_fonts("o/r", "font_names.json")
    assert (first.fonts_files, first.fonts_faces, first.errors) == (2, 2, []) and fake.blobs() == 2
    assert (tmp_path / global_config.FONTS_CACHE / "A.ttf").exists()
    (tmp_path / "font_names.json").unlink()
    second = fc.repo_fonts("o/r", "font_names.json")
    assert (second.fonts_files, second.fonts_faces) == (2, 2) and fake.blobs() == 2


def _fonts_tree(fake, monkeypatch):
    tree = {"tree": [{"path": f"{fc.FONTS_DIR}/{n}", "type": "blob", "sha": _sha(d)} for n, d in fake.files.items()]}
    real = fake.get
    monkeypatch.setattr(ga, "get", lambda url, **kw: Reply(body=tree) if "/git/trees/main?" in url else real(url, **kw))
    return tree


def test_fonts_command_with_a_warm_cache_requests_no_font(fake, no_checkout, monkeypatch, tmp_path):
    _fonts_tree(fake, monkeypatch)
    fc.repo_fonts("o/r", "font_names.json")
    import shutil
    shutil.rmtree(tmp_path / global_config.FONTS_CACHE)                         # font_names.json knows every blob
    again = fc.repo_fonts("o/r", "font_names.json")
    assert (again.fonts_files, again.fonts_faces, again.errors) == (2, 2, []) and fake.blobs() == 2
    assert not (tmp_path / global_config.FONTS_CACHE).exists()


def test_fonts_command_reads_a_matching_checkout(fake, no_checkout, font_bytes, monkeypatch, tmp_path):
    """A local copy is used only where its sha is the branch's: the rest comes through the cache."""
    _fonts_tree(fake, monkeypatch)
    checkout = tmp_path / "TTSCardMaker"
    (checkout / fc.FONTS_DIR).mkdir(parents=True)
    (checkout / fc.FONTS_DIR / "A.ttf").write_bytes(font_bytes["CacheFixture-A"])
    (checkout / fc.FONTS_DIR / "B.ttf").write_bytes(font_bytes["CacheFixture-C"])   # out of date
    monkeypatch.setattr(fc, "LOCAL_DIR_LOC", str(checkout))
    report = fc.repo_fonts("o/r", "font_names.json")
    assert (report.fonts_files, report.errors) == (2, []) and fake.blobs() == 2  # the cache's refresh fills both
    assert json.loads(Path("font_names.json").read_text(encoding="utf-8"))["fonts"].keys() == {_sha(d) for d in fake.files.values()}
    Path("font_names.json").unlink()
    (checkout / fc.FONTS_DIR / "B.ttf").write_bytes(font_bytes["CacheFixture-B"])
    import shutil
    shutil.rmtree(tmp_path / global_config.FONTS_CACHE)
    assert fc.repo_fonts("o/r", "font_names.json").fonts_files == 2 and fake.blobs() == 2   # all from the checkout
    assert not (tmp_path / global_config.FONTS_CACHE).exists()


def test_fonts_command_never_refreshes_under_a_bake(fake, no_checkout, monkeypatch, tmp_path):
    from threading import Lock
    _fonts_tree(fake, monkeypatch)
    lock = Lock()
    lock.acquire()
    report = fc.repo_fonts("o/r", "font_names.json", fonts_lock=lock)
    assert report.fonts_unread == 2 and fake.blobs() == 0 and "holds the fonts cache" in report.errors[0]
    lock.release()
    report = fc.repo_fonts("o/r", "font_names.json", fonts_lock=lock)
    assert (report.fonts_unread, report.fonts_files) == (0, 2) and not lock.locked()


def test_fonts_command_saves_on_a_cache_failure(fake, no_checkout, monkeypatch, tmp_path):
    _fonts_tree(fake, monkeypatch)

    def full(*args, **kw):
        raise OSError("No space left on device")
    monkeypatch.setattr(fc, "refresh_cache", full)
    details = tmp_path / "details.json"
    pd.save({_sha(b"psd"): {"path": "Creatures/X.psd", "fonts": {"F": {"layers": ["Text/Name"], "supplied": None}}}}, str(details))
    with pytest.raises(OSError):
        fc.repo_fonts("o/r", "font_names.json", details_path=str(details))
    assert json.loads(Path("font_names.json").read_text(encoding="utf-8"))["psds"] == {_sha(b"psd"): {"F": ["Name"]}}


def test_cache_is_gitignored_and_named():
    assert global_config.FONTS_CACHE == "fonts_cache"
    assert "fonts_cache/" in (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
