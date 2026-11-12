"""
The website build (TTSCardMaker site/build/build_site.py) on a small card repo made here, offline, from the card sync's
db/ (CARD_DATA_DIR, load_card_data; no CARD_DATA_DIR, or no readable stats.json, builds nothing): the current text dated
from its start (since), a duplicate's history, dates by committer time, an archived card's moved_to, tag,
author and created, image revisions (the person pages), Pacific contribution months, the person stats, index.html's
pre-written first screen (checked against grid.js in node), the Font filter's data (cardfonts.json), a profile's
subseries, PSD sizes, the Changes tabs' own files. Loads
build_site.py from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without it.
"""
import copy
import importlib.util
import io
import json
import re
import subprocess
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from html import unescape as html_unescape
from os import environ
from pathlib import Path

import pytest

import commands.card_db as cdb
import commands.card_models as cm
from conftest import SITE_SRC

BUILD_SITE = SITE_SRC / "site" / "build" / "build_site.py"
EDIT_JS = BUILD_SITE.parents[1] / "assets" / "edit.js"
DAY = 86400
REBASED = 20 * DAY                            # committer time after author time (a rebase: Patch 1.9's commits)
ALPHA = "Creatures/Other/2 Stars/Alpha"
MELON = "Items/Heal/2 Stars/Watermelon"
PALACE = "Field/4 Stars/White_Palace.psd"


@pytest.fixture(scope="module")
def bs(tmp_path_factory):
    if not BUILD_SITE.is_file():
        pytest.skip(f"no card repo build_site at {BUILD_SITE}")
    spec = importlib.util.spec_from_file_location("site_build_site", BUILD_SITE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.covers = tmp_path_factory.mktemp("covers")      # the cover's AVIF copies, made once per worker
    return module


def _png(path: Path, shade: int) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (60, 84), (shade, 40, 90)).save(path)


def _commit(repo: Path, message: str, at: int) -> tuple:
    """(sha, committer time): authored at `at`, committed REBASED later."""
    env = {**environ, "GIT_AUTHOR_DATE": f"@{at} +0000", "GIT_COMMITTER_DATE": f"@{at + REBASED} +0000",
           "GIT_AUTHOR_NAME": "Maker", "GIT_AUTHOR_EMAIL": "maker@example.com",
           "GIT_COMMITTER_NAME": "Maker", "GIT_COMMITTER_EMAIL": "maker@example.com"}
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], env=env, capture_output=True, text=True, check=True).stdout
    git("add", "-A")
    git("commit", "-q", "-m", message)
    return git("rev-parse", "HEAD").strip(), at + REBASED


def _card(name, path, ability, timestamp, **kw):
    return cm.CardInfo(name=name, card_type=kw.pop("card_type", "creature"), path=path, ability=ability, timestamp=float(timestamp),
                       stars=2, author="Maker", stats=cm.CardStats(hp=3, defense=2, attack=2, speed=1), **kw)


@pytest.fixture
def repo(tmp_path):
    """A card repo of three commits and the bot's data for it:
    - Alpha: "Old text." (c1), "New text." (c2), the same data re-saved with new art (c3): the bot's card is dated c3,
      its start carried to c2 (version.from, no from_commit);
    - Watermelon with its duplicate Watermelon2 (Inconspicuous Watermelon, its own PNG), one earlier text;
    - White Palace, archived: its PSD moved into MDW/The Morgue in c3."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    start = 1_780_000_000
    (root / f"{ALPHA}.psd").parent.mkdir(parents=True)
    (root / f"{ALPHA}.psd").write_text("psd 1")
    _png(root / f"{ALPHA}.png", 10)
    (root / f"{MELON}.psd").parent.mkdir(parents=True)
    (root / f"{MELON}.psd").write_text("melon 1")
    _png(root / f"{MELON}.png", 50)
    _png(root / f"{MELON}2.png", 60)
    (root / PALACE).parent.mkdir(parents=True)
    (root / PALACE).write_text("palace")
    c1, t1 = _commit(root, "Cards", start)
    (root / f"{ALPHA}.psd").write_text("psd 2")
    _png(root / f"{ALPHA}.png", 20)
    (root / f"{MELON}.psd").write_text("melon 2")
    c2, t2 = _commit(root, "New text", start + DAY)
    (root / f"{ALPHA}.psd").write_text("psd 3")
    _png(root / f"{ALPHA}.png", 30)
    (root / "MDW/The Morgue").mkdir(parents=True)
    (root / PALACE).rename(root / "MDW/The Morgue/White_Palace.psd")
    c3, t3 = _commit(root, "Standardize", start + 2 * DAY)
    (root / "site").mkdir()                       # ratings: a commit of no card file (no image version, no edit batch)
    (root / "site" / "Card Ratings.json").write_text(json.dumps({"cards": {
        "alpha": [{"date": "2023-04-28", "scale": "power", "value": 6, "by": "group", "source": "TTS session"}],
        "White Palace": [{"date": "2025-04-28", "scale": "favorability", "value": 7, "by": "group", "source": "TTS session"}]}}))
    _commit(root, "Ratings", start + 3 * DAY)

    stats = {"Alpha": _card("Alpha", f"{ALPHA}.psd", "New text.", t3, blob_sha="b3", series="Kirby"),
             "Watermelon": _card("Watermelon", f"{MELON}.psd", "Heal 2.", t2, blob_sha="m2", card_type="item"),
             "Watermelon2": _card("Watermelon2", f"{MELON}.psd", "Heal 2.", t2, blob_sha="m2", card_type="item",
                                  duplicate_of="Watermelon", display_name="Inconspicuous Watermelon")}
    stats["Alpha"].version = {"from": float(t2), "to": None, "cause": "psd"}     # a carried start (no from_commit)
    palace = _card("White Palace", PALACE, "Field.", t1, card_type="field", series="kirby")     # an old spelling
    palace.archived = {"at": float(t3), "path": PALACE, "commit": c3, "moved_to": "MDW/The Morgue/White_Palace.psd"}
    old = defaultdict(list, {"Alpha": [_card("Alpha", f"{ALPHA}.psd", "Old text.", t1)],
                             "Watermelon": [_card("Watermelon", f"{MELON}.psd", "Heal 1.", t1, card_type="item")],
                             "White Palace": [palace]})
    for card in list(stats.values()) + [v for vs in old.values() for v in vs]:
        card.parser_version = 2
    details = json.dumps({"version": 1, "cards": {}})
    db = tmp_path / "db"
    cdb.write_db(str(db), stats, old, details, {"card_commit": c3})
    (db / "changelog").mkdir()
    event = {"seq": 1, "at": time.time(), "kind": "archived", "card": "White Palace", "title": "White Palace", "path": PALACE,
             "moved_to": "MDW/The Morgue/White_Palace.psd", "card_commit": c3}
    (db / "changelog" / "2026-10.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    return {"root": root, "db": db, "t": (t1, t2, t3), "commits": (c1, c2, c3)}


def _build(bs, monkeypatch, repo, out, data_dir=None):
    """build_site.main() offline into out/ from repo's db/ (or data_dir); {name: data/<name> as JSON}."""
    monkeypatch.setattr(bs, "REPO", str(repo["root"]))
    monkeypatch.setattr(bs, "HISTORY_REPO", str(repo["root"]))
    for name, sub in (("DIST", "dist"), ("CACHE", ".cache"), ("OG_CACHE", ".og-cache")):
        monkeypatch.setattr(bs, name, str(out / sub))
    (out / ".cache").mkdir(parents=True, exist_ok=True)
    if not (out / ".cache" / "cover").exists():
        (out / ".cache" / "cover").symlink_to(bs.covers, target_is_directory=True)
    monkeypatch.setattr(bs, "CARD_DATA_DIR", str(repo["db"]) if data_dir is None else data_dir)
    monkeypatch.setattr(bs, "MIN_CARDS", 1)

    def fetch(url, tries=3, accept=None):
        raise OSError(f"offline: {url}")
    monkeypatch.setattr(bs, "fetch", fetch)
    monkeypatch.setattr(sys, "argv", ["build_site.py"])
    bs.main()
    data = out / "dist" / "data"
    return {p.name: json.loads(p.read_text(encoding="utf-8")) for p in data.glob("*.json")}


def _by_name(cards: list) -> dict:
    return {c["name"]: c for c in cards}


def test_the_current_text_is_dated_from_its_start(bs, repo, tmp_path, monkeypatch):
    """Alpha's text started at c2 (the bot's carried start), though its PSD was last committed at c3: cards.json keeps
    ts (the last commit, "updated") and adds since; history.json holds the older text at c1."""
    t1, t2, t3 = repo["t"]
    out = _build(bs, monkeypatch, repo, tmp_path / "a")
    alpha = _by_name(out["cards.json"]["cards"])["Alpha"]
    assert (alpha["ts"], alpha["since"]) == (t3, t2)
    assert out["history.json"][alpha["id"]] == [{"ts": t1, "ability": "Old text."}]
    assert "since" not in _by_name(out["cards.json"]["cards"])["Watermelon"]      # its text started at its last commit


def test_minor_picture_changes_fold_and_updated_follows_real_changes(bs, repo, tmp_path, monkeypatch):
    """Owner 11-09. Alpha's c3 picture changed under the share that shows (db/png_changes.json) with no data change: its
    row folds (versions.json minor), and its "updated" goes back to its text change at c2 (since goes with it). An older
    text that differs only in formatting is one text with the one before it (history.json); the data keeps both."""
    if not hasattr(bs, "load_png_changes"):
        pytest.skip("the site has no picture changes yet")
    (c1, c2, c3), (t1, t2, t3) = repo["commits"], repo["t"]
    blob = lambda c: subprocess.run(["git", "-C", str(repo["root"]), "rev-parse", f"{c}:{ALPHA}.png"], capture_output=True,
                                    text=True, check=True).stdout.strip()
    stats, old, details, _ = cdb.read_db(str(repo["db"]))
    old["Alpha"].append(_card("Alpha", f"{ALPHA}.psd", "old  text", t1 + DAY // 2))       # spacing, case, a full stop
    old["Alpha"][1].parser_version = 2
    cdb.write_db(str(repo["db"]), stats, old, json.dumps({"version": 1, "cards": details}), {"card_commit": c3})
    (repo["db"] / "png_changes.json").write_text(json.dumps({"rule": bs.PNG_RULE, "pairs": {
        f"{blob(c2)}..{blob(c3)}": [bs.PNG_VISIBLE / 4, bs.PNG_RULE]}}), encoding="utf-8")
    out = _build(bs, monkeypatch, repo, tmp_path / "a")
    alpha = _by_name(out["cards.json"]["cards"])["Alpha"]
    assert out["versions.json"]["minor"] == {alpha["id"]: [0]} and (alpha["iv"], alpha["ivm"]) == (3, 1)
    assert alpha["ts"] == t2 and "since" not in alpha
    assert out["history.json"][alpha["id"]] == [{"ts": t1, "ability": "old  text"}]


def test_psd_sizes_come_by_blob_and_a_duplicate_has_its_psds(bs, repo, tmp_path, monkeypatch):
    """PSD_TREE (site.yml's trees API reply) sizes each card by its PSD's blob at HEAD: Watermelon2 (a numbered PNG of
    Watermelon's PSD) gets Watermelon's; a blob the reply doesn't list gets none."""
    if not hasattr(bs, "psd_sizes"):
        pytest.skip("the site has no PSD sizes yet")
    blob = lambda path: subprocess.run(["git", "-C", str(repo["root"]), "rev-parse", f"HEAD:{path}"], capture_output=True,
                                       text=True, check=True).stdout.strip()
    tree = tmp_path / "tree.json"
    tree.write_text(json.dumps({"truncated": False, "tree": [
        {"path": f"{MELON}.psd", "type": "blob", "sha": blob(f"{MELON}.psd"), "size": 34_712_345},
        {"path": f"{ALPHA}.psd", "type": "blob", "sha": "0" * 40, "size": 9}]}))      # another blob than HEAD's
    monkeypatch.setattr(bs, "PSD_TREE", str(tree))
    cards = _by_name(_build(bs, monkeypatch, repo, tmp_path / "a")["cards.json"]["cards"])
    assert cards["Watermelon"]["psdSize"] == cards["Inconspicuous Watermelon"]["psdSize"] == 34_712_345
    assert "psdSize" not in cards["Alpha"]


def test_a_picture_only_duplicate_is_its_originals_card(bs, repo, tmp_path, monkeypatch):
    """Gamma.png has no PSD and its Card Edits.json entry names Alpha in dup_of; the card data copies Alpha as Gamma
    (duplicate_of): the site lists it parsed, with Alpha's data, its own name and picture (owner 10-26)."""
    root = repo["root"]
    _png(root / "Creatures/Other/2 Stars/Gamma.png", 70)
    stats, old, details, _ = cdb.read_db(str(repo["db"]))
    hp = stats["Alpha"].stats.hp + 1                         # a PSD edit on Alpha: pending on Alpha, never on Gamma
    (root / "site" / "Card Edits.json").write_text(json.dumps({"cards": {"Gamma": {"dup_of": "Alpha"}, "Alpha": {"hp": hp}}}), encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=M", "-c", "user.email=m@x", "commit", "-q", "-m", "Gamma"], check=True)
    gamma = copy.deepcopy(stats["Alpha"])
    gamma.name, gamma.duplicate_of = "Gamma", "Alpha"
    stats["Gamma"] = gamma
    cdb.write_db(str(repo["db"]), stats, old, json.dumps({"version": 1, "cards": details}), {"card_commit": repo["commits"][2]})
    cards = _by_name(_build(bs, monkeypatch, repo, tmp_path / "a")["cards.json"]["cards"])
    alpha, g = cards["Alpha"], cards["Gamma"]
    assert "parsed" not in g and g["dup"] == alpha["id"] and alpha["dups"] == [g["id"]]
    assert g["png"] == "Creatures/Other/2 Stars/Gamma.png" and g["img"] != alpha["img"]
    assert all(g.get(k) == alpha.get(k) for k in ("type", "series", "stars", "types", "hp", "def", "atk", "spd", "ability", "author"))
    assert alpha["pending"] == ["hp"] and "pending" not in g          # the bake never redraws Gamma.png
    game = (tmp_path / "a" / "dist" / "game.txt").read_text(encoding="utf-8")
    assert "duplicate of Alpha" in next(line for line in game.splitlines() if line.startswith("Gamma |"))


def test_each_version_links_the_commit_that_made_its_psd(bs, repo, tmp_path, monkeypatch):
    """A text's or change's commit is the one that gave its PSD the bytes it was read from (blob_sha): history.json
    texts carry c (an index into _shas), the newest older one n (the current text's), analytics.json change rows c (into
    commits). An edit version gets none (no card commit made it)."""
    (c1, c2, c3), (t1, t2, t3) = repo["commits"], repo["t"]
    blob = lambda c, p: subprocess.run(["git", "-C", str(repo["root"]), "rev-parse", f"{c}:{p}"], capture_output=True, text=True, check=True).stdout.strip()
    stats, old, details, _ = cdb.read_db(str(repo["db"]))
    old["Alpha"][0].blob_sha, old["Alpha"][0].version = blob(c1, f"{ALPHA}.psd"), {"from": float(t1), "cause": "legacy"}
    stats["Alpha"].blob_sha, stats["Alpha"].version = blob(c3, f"{ALPHA}.psd"), {"from": float(t3), "from_commit": c3, "cause": "psd"}
    old["Watermelon"][0].blob_sha = blob(c1, f"{MELON}.psd")
    stats["Watermelon"].blob_sha, stats["Watermelon"].version = blob(c2, f"{MELON}.psd"), {"from": float(t3), "from_commit": c3, "cause": "edit"}
    cdb.write_db(str(repo["db"]), stats, old, json.dumps({"version": 1, "cards": details}), {"card_commit": c3})
    out = _build(bs, monkeypatch, repo, tmp_path / "a")
    cards, h, an = _by_name(out["cards.json"]["cards"]), out["history.json"], out["analytics.json"]
    sha = lambda i: h["_shas"][i]
    alpha, melon = cards["Alpha"]["id"], cards["Watermelon"]["id"]
    assert h[alpha] == [{"ts": t1, "ability": "Old text.", "c": 0, "n": 1}] and h["_shas"][:2] == [c1[:12], c3[:12]]
    assert [sha(v["c"]) for v in h[melon]] == [c1[:12]] and "n" not in h[melon][0]     # an edit: no link
    assert [an["commits"][r["c"]] for r in an["cards"][alpha]["changes"]] == [c3[:12]]


def test_a_commit_of_many_card_psds_is_heavy(bs, repo, tmp_path, monkeypatch):
    """cards.json heavy: the 12-char shas of commits changing HEAVY_PSDS card PSDs or more, whose GitHub page fails
    (text.js commitLink links the card's history instead). c1 added 3; c2 and c3 changed 2 (the Morgue's is no card)."""
    monkeypatch.setattr(bs, "HEAVY_PSDS", 3)
    assert _build(bs, monkeypatch, repo, tmp_path / "a")["cards.json"]["heavy"] == [repo["commits"][0][:12]]


def test_editor_fields_load_with_the_editor(bs, repo, tmp_path, monkeypatch):
    """cards.json leaves the fields only edit mode reads (EDITOR_FIELDS: blob, ink, ratings...) to editing.json by id."""
    if not hasattr(bs, "EDITOR_FIELDS"):
        pytest.skip("before TC's editing.json")
    out = _build(bs, monkeypatch, repo, tmp_path / "a")
    assert not set(bs.EDITOR_FIELDS) & {k for c in out["cards.json"]["cards"] for k in c}
    alpha = out["editing.json"][_by_name(out["cards.json"]["cards"])["Alpha"]["id"]]
    assert len(alpha["blob"]) == 8 and [r["value"] for r in alpha["ratings"]] == [6]


def test_a_version_commit_follows_renames_and_refuses_doubt(bs):
    """version_commit on parse_history rows: the commit that made the bytes the PSD held at the version's start, pure
    renames followed back (also a later one: a record's path is its newest); none for an edit, the previous version's
    bytes (a move only for a folder change), bytes from after a blobless previous version's start, or another path's."""
    a = bs.analytics
    c = lambda n: n * 40
    commits = [(c("6"), 60, "", "", "", [("M", "y", "z", "New/C.psd")], []),            # newer bytes: a later reading
               (c("5"), 50, "", "", "", [("D", "y", "0", "Old/C.psd"), ("A", "0", "y", "New/C.psd")], []),   # a move
               (c("4"), 40, "", "", "", [("M", "x", "y", "Old/C.psd")], []),
               (c("3"), 30, "", "", "", [("D", "x", "0", "Typo/C.psd"), ("A", "0", "x", "Old/C.psd")], []),  # a rename
               (c("1"), 10, "", "", "", [("A", "0", "x", "Typo/C.psd"), ("A", "0", "q", "Other.psd")], [])]
    lin = a.psd_lineage(commits)
    v = lambda ts, blob, cause="psd", path="New/C.psd", **info: bs.stand_in(bs.CardInfo, {
        "timestamp": float(ts), "blob_sha": blob, "path": path, "version": {"from": float(ts), "cause": cause, **info}})
    first, second = v(30, "x", "legacy"), v(40, "z")
    assert a.version_commit(first, None, lin) == c("1")[:12]                 # the rename at 30 made no bytes
    assert a.version_commit(second, first, lin) == c("4")[:12]               # read later (z), started at 4's bytes
    assert a.version_commit(v(50, "z"), second, lin) is None                 # the move: same bytes as before
    assert a.version_commit(v(50, "z", "moved", from_commit=c("5")), second, lin, moved=True) == c("5")[:12]
    assert a.version_commit(v(40, "z", "edit"), first, lin) is None
    assert a.version_commit(second, v(45, None), lin) is None                # 4's bytes may be the blobless one's
    assert a.version_commit(v(40, "q"), None, lin) is None                   # Other.psd's bytes, not this path's
    assert [a.version_commit(x, None, lin) for x in (v(40, None), v(5, "x"), None)] == [None] * 3


def test_balance_follows_renames_and_folds_a_rename_alone(bs):
    """analytics.renames_followed: an old name's versions go on under the name it became; a rename alone (the same PSD
    bytes) starts no version (the new reading takes the old one's start, date and path), a redraw does."""
    a = bs.analytics
    if not hasattr(a, "renames_followed"):
        pytest.skip("the site's Balance follows no renames yet")
    v = lambda ts, blob, path, **kw: bs.stand_in(bs.CardInfo, {"timestamp": float(ts), "blob_sha": blob, "path": path,
                                                               "version": {"from": float(ts)}, **kw})
    first, last, current = v(10, "x", "Old.psd"), v(20, "y", "Old.psd", archived={"renamed_to": "New"}), v(30, "y", "New.psd")
    old_stats = {"Old": [first, last], "Other": [v(5, "q", "Other.psd")]}
    old, stats = a.renames_followed(old_stats, {"New": current}, {"New": ["Old"]})
    assert set(old) == {"New", "Other"} and old["New"] == [first] and current.timestamp == 30.0
    assert (stats["New"].timestamp, stats["New"].path, stats["New"].version) == (20.0, "Old.psd", {"from": 20.0})
    assert a.chain_of("New", stats, old) == [first, stats["New"]]
    redrawn = v(30, "z", "New.psd")
    old, stats = a.renames_followed(old_stats, {"New": redrawn}, {"New": ["Old"]})
    assert old["New"] == [first, last] and stats["New"] is redrawn


def test_types_before_parser_2_are_no_change(bs):
    """Parser 1 missed and mistook type icons (Dumplings' tomato read as panda): a type swap from such a reading is no
    change row (analytics.change) and no real change (real_change); the shown card (no parser_version) is parsed."""
    a = bs.analytics
    v = lambda types, pv: bs.stand_in(bs.CardInfo, {"types": types, "parser_version": pv, "ability": "Heal 2.", "timestamp": 1.0})
    legacy, parsed = v(["panda"], None), v(["food"], 7)
    assert a.change(legacy, parsed, []) == {} and not bs.real_change(legacy, parsed)
    assert bs.real_change(parsed, v(["fire"], 7))
    shown = a.shown({"types": ["fire"], "ability": "Heal 2."}, parsed)
    assert a.change(parsed, shown, []) == {"types": {"add": ["fire"], "drop": ["food"]}}


def test_values_a_reading_never_recorded_are_no_change(bs):
    """Older readings recorded no item type, stars or series (aux items before parser 7, a first series fill), and a
    reading whose stats failed (NO STATS FOUND, or from parser 2 a stat over 20 not allowed) read none: such a side is
    unknown, no change row; two values still are."""
    a = bs.analytics
    v = lambda **f: bs.stand_in(bs.CardInfo, {"ability": "Heal 2.", "timestamp": 1.0, "parser_version": 7, **f})
    assert a.change(v(subtype=None, stars=None, series=None), v(subtype="active", stars=5, series="X"), []) == {}
    assert a.change(v(series="Mother 2"), v(series="MOTHER"), []) == {"series": ["Mother 2", "MOTHER"]}
    line = lambda *s: bs.stand_in(bs.CardStats, dict(zip(("hp", "defense", "attack", "speed"), s)))
    read, failed = v(stats=line(5, 5, 6, 7)), v(stats=line(-1, -1, -1, -1), problems=["NO STATS FOUND: no HP, DEF, ATK or SPD bars"])
    assert a.change(read, failed, []) == {} == a.change(failed, read, [])
    assert a.change(read, a.shown({"ability": "Heal 2."}, failed), []) == {}         # the current reading failed
    edited = a.shown({"hp": 5, "def": 5, "atk": 6, "spd": 8, "ability": "Heal 2."}, failed)      # Card Edits' stats
    assert a.change(read, edited, [])["stats"] == [[5, 5, 6, 7], [5, 5, 6, 8]]
    tyler = v(stats=line(23, -1, -1, -1), stars=5)               # four bars named Health: 7+6+4+6, no problem stored
    assert a.change(read, tyler, []) == {} == a.change(tyler, read, [])
    assert a.change(read, v(stats=line(23, 5, 6, 7), allow_high_stats=True), [])["stats"] == [[5, 5, 6, 7], [23, 5, 6, 7]]


def test_a_renamed_card_goes_on_and_its_old_page_redirects(bs, repo, tmp_path, monkeypatch):
    """The bake renames and redraws Alpha as New Alpha in one commit (its Renamed trailers): the card's image versions,
    text history and Balance changes go on as one card, and c/alpha/ sends its links on to c/new-alpha/; a profile's
    pick, a published hand and the app's old id (cards.json "moved") follow it."""
    root, t3 = repo["root"], repo["t"][2]
    new = "Creatures/Other/2 Stars/New_Alpha"
    (root / f"{ALPHA}.psd").rename(root / f"{new}.psd")
    (root / f"{ALPHA}.png").unlink()
    _png(root / f"{new}.png", 40)                 # redrawn: no exact rename
    for path, doc in (("site/profiles/Someone.json", {"login": "Someone", "favourite": "alpha"}),
                      ("site/decks/Someone/hand.json", {"v": 1, "owner": "Someone", "name": "Hand", "cards": ["alpha"]})):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(json.dumps(doc), encoding="utf-8")
    git = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True, check=True,
                                    env={**environ, "GIT_AUTHOR_NAME": "AutoCroissant", "GIT_AUTHOR_EMAIL": "a@x",
                                         "GIT_COMMITTER_NAME": "AutoCroissant", "GIT_COMMITTER_EMAIL": "a@x"}).stdout
    git("add", "-A")
    git("commit", "-q", "-m", f"Card edits: 1 baked\n\n- Alpha: renamed: Alpha -> New Alpha\n\nRenamed: {ALPHA}.psd -> {new}.psd\n"
                              f"Renamed: {ALPHA}.png -> {new}.png")
    stats, old, details, _ = cdb.read_db(str(repo["db"]))
    card = stats.pop("Alpha")
    gone = copy.deepcopy(card)
    gone.archived = {"at": float(t3 + DAY), "path": f"{ALPHA}.psd", "renamed_to": "New Alpha"}
    old["Alpha"].append(gone)
    card.name, card.path = "New Alpha", f"{new}.psd"
    card.version = {"from": float(t3 + DAY), "to": None, "cause": "psd"}
    stats["New Alpha"] = card
    cdb.write_db(str(repo["db"]), stats, old, json.dumps({"version": 1, "cards": details}),
                 {"card_commit": git("rev-parse", "HEAD").strip()})
    out = _build(bs, monkeypatch, repo, tmp_path / "a")
    page = (tmp_path / "a" / "dist" / "c" / "alpha" / "index.html").read_text(encoding="utf-8")
    assert 'http-equiv="refresh" content="0; url=../new-alpha/"' in page and (tmp_path / "a" / "dist" / "c" / "new-alpha" / "index.html").exists()
    alpha = _by_name(out["cards.json"]["cards"])["New Alpha"]
    assert alpha["was"] == ["Alpha"]                                              # its old name finds it
    assert out["history.json"][alpha["id"]] == [{"ts": repo["t"][0], "ability": "Old text."}]
    assert len(out["versions.json"]["cards"][alpha["id"]]) == 4                      # c1, c2, c3 and the rename
    if hasattr(bs.analytics, "renames_followed"):                                 # TC 11-11
        (row,) = out["analytics.json"]["cards"][alpha["id"]]["changes"]               # Old text. -> New text.
        assert row["ts"] == repo["t"][1] and "text" in row
    assert out["cards.json"]["moved"] == {"alpha": alpha["id"]} == {"alpha": "new-alpha"}
    assert out["users.json"]["users"][0]["favourite"] == out["decks.json"]["decks"][0]["cards"][0] == "new-alpha"


def test_the_morgue_is_its_own_list(bs, repo, tmp_path, monkeypatch):
    """Owner 11-08: MDW/The Morgue's cards (a PSD with its PNG; the card back is none) are cards.json morgue, picture
    only, kind and stars from the archive row that moved them there, with a share page; never a live card, the game
    text or the TTS details."""
    if not hasattr(bs, "morgue_cards"):
        pytest.skip("the site has no Morgue yet")
    root = repo["root"]
    _png(root / "MDW/The Morgue/White_Palace.png", 90)
    _png(root / "MDW/The Morgue/cardback.png", 95)
    _commit(root, "Morgue pictures", 1_780_000_000 + 4 * DAY)
    out = _build(bs, monkeypatch, repo, tmp_path / "a")
    data, dist = out["cards.json"], tmp_path / "a" / "dist"
    (palace,) = data["morgue"]
    assert {k: palace[k] for k in ("id", "name", "type", "stars", "png", "morgue")} == {
        "id": "white-palace", "name": "White Palace", "type": "field", "stars": 2, "png": "MDW/The Morgue/White_Palace.png",
        "morgue": True} and palace["img"] and palace["retired"] == repo["t"][2]
    assert "White Palace" not in _by_name(data["cards"]) and (dist / "c" / "white-palace" / "index.html").exists()
    assert "White Palace" not in (dist / "game.txt").read_text(encoding="utf-8") + json.dumps(out["tts.json"])


def test_archived_author_and_created_precedence(bs):
    """author: Card Authors.txt's credit > the tag's (backfill) > the last version's, shown through [people]; created:
    the last version's created date, else the Pacific day of its oldest version (its version record's start)."""

    def v(ts, **kw):
        card = bs.CardInfo()
        card.__dict__.update(timestamp=float(ts), card_type="creature", stars=2, author="Last", **kw)
        return card
    start = 1_700_000_000
    old = {"Gone": [v(start + 9 * DAY, version={"from": float(start)}), v(start + 20 * DAY, archived={"at": start + 30 * DAY, "author": "Tagged"})]}
    rows = lambda **kw: bs.archived_cards(old, lambda name: None, (), **kw)[0]
    day = datetime.fromtimestamp(start, bs.analytics.ZONE).strftime("%Y-%m-%d")
    assert (rows()["author"], rows()["created"], rows()["tag"]) == ("Tagged", day, "deleted")
    assert rows(credited={"gone": "Listed"}, people={"listed": "Shown"})["author"] == "Shown"
    del old["Gone"][1].archived["author"]
    old["Gone"][1].created = {"date": "2019-05-06", "from": "discord"}
    assert (rows()["author"], rows()["created"]) == ("Last", "2019-05-06")


def test_a_profile_subseries_is_checked_like_its_series(bs):
    """A profile's subseries must be one cards use (any series, or the profile's series when it has one); a bad one is
    dropped with a warning, like the series."""
    if "subseries" not in getattr(bs, "PROFILE_FIELDS", ()):
        pytest.skip(f"{BUILD_SITE} has no profile subseries")
    cards = {"grapes": {"id": "grapes", "type": "item", "series": "Don't Starve", "subseries": "Gourmet Expansion"},
             "kirby": {"id": "kirby", "type": "creature", "series": "Kirby", "subseries": "Kirby Star Allies"}}
    clean = lambda **kw: bs.clean_profile({"login": "Me", **kw}, "Me", cards, [])
    assert clean(series="Don't Starve", subseries="Gourmet Expansion") == (
        {"login": "Me", "series": "Don't Starve", "subseries": "Gourmet Expansion"}, [])
    assert clean(subseries="Gourmet Expansion") == ({"login": "Me", "subseries": "Gourmet Expansion"}, [])
    p, notes = clean(series="Kirby", subseries="Gourmet Expansion")
    assert p == {"login": "Me", "series": "Kirby"} and notes == ["subseries 'Gourmet Expansion' has no cards in series 'Kirby', dropped"]
    p, notes = clean(series="Don't Starve Gourmet Expansion", subseries="Gourmet")
    assert p == {"login": "Me"} and notes == ["series \"Don't Starve Gourmet Expansion\" isn't a known series, dropped",
                                              "subseries 'Gourmet' isn't a known subseries, dropped"]


def test_contributions_json_has_person_stats(bs, repo, tmp_path, monkeypatch):
    """Maker's row: Alpha's PNG changed in c2 and c3 (self 2), White Palace archived (Retired, without path or commit),
    Alpha's strength."""
    out = _build(bs, monkeypatch, repo, tmp_path / "a")
    maker = {r["key"]: r for r in out["contributions.json"]["people"]}["Maker"]
    assert (maker["revisions"]["self"], maker["revisions"]["others"]) == (2, 0)
    assert [(r["card"], r["tag"]) for r in maker["archived"]] == [("White Palace", "retired")]
    assert not {"path", "commit", "ratings"} & set(maker["archived"][0])
    assert maker["power"]["n"] >= 1 and "z" in out["contributions.json"]["site"]


def test_each_changes_tab_has_its_own_file(bs, repo, tmp_path, monkeypatch):
    """11-11: changelog.json has every event (no archived rows or batches), archive.json the archived rows, edits.json
    the edited and authors events and the batches."""
    if not hasattr(bs, "changelog_files"):
        pytest.skip("the site has one changelog file")
    log = repo["db"] / "changelog" / "2026-10.jsonl"
    edit = {"seq": 2, "at": time.time(), "kind": "edited", "card": "Alpha", "title": "Alpha"}
    log.write_text(log.read_text(encoding="utf-8") + json.dumps(edit) + "\n", encoding="utf-8")
    out = _build(bs, monkeypatch, repo, tmp_path / "a")
    kinds = lambda name: [e["kind"] for e in out[name]["events"]]
    assert kinds("changelog.json") == ["archived", "edited"] and kinds("edits.json") == ["edited"]
    assert not {"archived", "edits"} & set(out["changelog.json"]) and "batches" in out["edits.json"]["edits"]
    assert [r["card"] for r in out["archive.json"]["archived"]] == ["White Palace"]

def test_index_carries_the_first_screen(bs, repo, tmp_path, monkeypatch):
    """index.html's pre-written grid: the cards in Name order as links to their pages, checked against grid.js in node;
    a difference fails the build; its grid has grid.js's M size. A card's page carries its dialog's picture, the one it
    preloads, openCard's close button (card_frame) and style.css inline, as index.html; other share pages link it."""
    out = _build(bs, monkeypatch, repo, tmp_path / "a")
    page = (tmp_path / "a" / "dist" / "index.html").read_text(encoding="utf-8")
    ids = [c["id"] for c in sorted(out["cards.json"]["cards"], key=lambda c: c["name"].lower())]
    assert re.findall(r'<a class="tile" href="c/([^/"]+)/"', page) == ids
    assert page.count('fetchpriority="high"') == min(len(ids), bs.FIRST_EAGER)
    m = re.search(r'TILE_SIZES = \{[^}]*\bm: \["([^"]+)"', (BUILD_SITE.parents[1] / "assets" / "grid.js").read_text(encoding="utf-8"))[1]
    assert f'<div class="grid" style="--tile:{m};' in page
    card = (tmp_path / "a" / "dist" / "c" / "alpha" / "index.html").read_text(encoding="utf-8")
    assert re.search(r'<img id="detail-img" src="([^"]+)"', card)[1] == re.search(r'<link rel="preload" as="image" href="([^"]+)"', card)[1]
    app = (BUILD_SITE.parents[1] / "assets" / "app.js").read_text(encoding="utf-8")
    close = re.search(r'aria-label="\$\{esc\(c\.name\)\}">\s*(<button [^>]*>✕</button>)\s*<div class="detail">', app)[1]
    assert f'{close}<div class="detail">' in card                # a phone's close row: the picture stays put
    assert (f'href="{bs.BADGE_FONT}"' in card) == bs.has_badges(next(c for c in out["cards.json"]["cards"] if c["id"] == "alpha").get("ability") or "")
    css = lambda p: re.findall(r'<style>.*?</style>|<link rel="stylesheet"[^>]*>', p, re.S)
    assert css(card) == css(page) and css(page)[0].startswith("<style>")
    assert css((tmp_path / "a" / "dist" / "d" / "index.html").read_text(encoding="utf-8"))[0].startswith("<link")
    monkeypatch.setattr(bs, "tile_html", lambda c, eager, icons: f'<a class="tile" href="c/{c["id"]}/">')
    with pytest.raises(SystemExit, match="differs from grid.js"):
        _build(bs, monkeypatch, repo, tmp_path / "b")


def test_editor_stats_and_people_load_on_first_use(bs, repo, tmp_path, monkeypatch):
    """The editor (editing.js), Stats (charts.js), People (profile.js), the font box (tokens.js) and the rulebook (rulebook.js, its
    editor rulebook-edit.js) are imported by app.js with import(): without
    esbuild as stamped modules no page preloads; with it as chunks of their own, each with its stamped source map, none of
    their code in what a first visit loads (app.js and the chunks it imports), and the meta chunks list naming
    each one's files (app.js early()). Share pages preload the chunks app.js imports, index.html doesn't; its head script
    names the rulebook's for a Rulebook link."""
    exe = bs.esbuild()
    monkeypatch.setattr(bs, "esbuild", lambda: None)
    _build(bs, monkeypatch, repo, tmp_path / "a")
    assets, page = tmp_path / "a" / "dist" / "assets", (tmp_path / "a" / "dist" / "index.html").read_text(encoding="utf-8")
    lazy = ("editing.js", "charts.js", "profile.js", "tokens.js", "rulebook.js", "rulebook-edit.js")
    app = (assets / "app.js").read_text(encoding="utf-8")
    assert all(re.search(rf'import\("\./{re.escape(m)}\?v=\w+"\)', app) for m in lazy)
    assert not re.search(r'modulepreload" href="assets/(edit|stats|profile|preview|mockup|balance)\.js', page)
    if not exe:
        pytest.skip("no esbuild: the local build only")
    monkeypatch.setattr(bs, "esbuild", lambda: exe)
    _build(bs, monkeypatch, repo, tmp_path / "b")
    assets, page = tmp_path / "b" / "dist" / "assets", (tmp_path / "b" / "dist" / "index.html").read_text(encoding="utf-8")
    first, todo = [], ["app.js"]                  # what a first visit loads: app.js and the chunks it imports
    while todo:
        f = todo.pop()
        if f not in first:
            first.append(f)
            todo += re.findall(r'from\s*"\./([\w-]+\.js)"', (assets / f).read_text(encoding="utf-8"))
    chunks = {k: [p.removeprefix("assets/") for p in v] for k, v in json.loads(html_unescape(re.search(r'<meta name="chunks" content="([^"]*)">', page)[1])).items()}
    assert sorted(chunks) == sorted(lazy) and chunks["editing.js"][0].startswith("editing-")
    assert f'["modulepreload", "assets/{chunks["rulebook.js"][0]}"]' in page
    hinted = lambda p: re.findall(r'<link rel="modulepreload" href="assets/(?!app\.js)([^"]+)">', p)
    card = (tmp_path / "b" / "dist" / "c" / "alpha" / "index.html").read_text(encoding="utf-8")
    assert sorted(hinted(card)) == sorted(first[1:]) and first[1:] and not hinted(page)
    sources = {}
    for f in {*first, *(n for v in chunks.values() for n in v)}:
        code = (assets / f).read_text(encoding="utf-8")
        assert re.search(rf"//# sourceMappingURL={re.escape(f)}\.map\?v=\w+\n$", code)
        sources[f] = {s.removeprefix("src/") for s in json.loads((assets / f"{f}.map").read_text(encoding="utf-8"))["sources"]}
    loaded = set().union(*(sources[f] for f in first))
    assert {"app.js", "grid.js", "text.js"} <= loaded
    assert not loaded & {"edit.js", "inline.js", "preview.js", "mockup.js", "tokens.js", "stats.js", "balance.js", "insights.js", "profile.js",
                         "rulebook.js", "rulebook-edit.js"}
    assert {"edit.js", "mockup.js"} <= set().union(*(sources[f] for f in chunks["editing.js"]))
    assert {"stats.js", "balance.js"} <= set().union(*(sources[f] for f in chunks["charts.js"]))


def test_inlined_css_urls(bs):
    """index.html and card pages inline style.css: its relative url()s, relative to assets/, now to the page (../rulebook/... left the site)."""
    page = bs.inline_css('<link rel="stylesheet" href="assets/style.css?v=1">',
                         'a{src:url("../rulebook/x.woff2")}b{background:url(data:image/png;base64,AA)}c{background:url(i.png)}')
    assert page == '<style>a{src:url("rulebook/x.woff2")}b{background:url(data:image/png;base64,AA)}c{background:url(assets/i.png)}</style>'


def test_website_only_fields(bs, tmp_path, monkeypatch):
    """`_site` (owner 10-25): applied, never "PSD pending", the card's siteOnly (its "Website only" tag); edit.js stores it
    from an op's ticks as parse_edits and CardEdits.site_only clean it (a field without a value goes); unticked: pending."""
    from commands.card_edits import CardEdits
    if not EDIT_JS.is_file():
        pytest.skip("edit.js missing")
    script = tmp_path / "site.mjs"
    script.write_text(f'const {{ editsFile }} = await import({json.dumps(EDIT_JS.as_uri())});\n'
                      'const text = JSON.stringify({about: "x", cards: {"Mini Doomer": {"name": "Tiny Doomer", "hp": 9, "_by": "w"}}});\n'
                      'const ops = [{key: "Mini Doomer", set: {spd: 7}, unset: [], site: {name: true, hp: true, atk: true}}];\n'
                      'process.stdout.write(editsFile(text, ops, "T", "web"));\n', encoding="utf-8")
    done = subprocess.run(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    entry = json.loads(done.stdout)["cards"]["Mini Doomer"]
    assert entry == {"name": "Tiny Doomer", "hp": 9, "spd": 7, "_site": ["name", "hp"], "_by": "web", "_at": "T"}   # atk: no value
    assert CardEdits.parse(done.stdout).site("mini doomer") == ["name", "hp"]
    (tmp_path / "site").mkdir()
    (tmp_path / "site" / "Card Edits.json").write_text(done.stdout, encoding="utf-8")
    monkeypatch.setattr(bs, "REPO", str(tmp_path))
    edits = bs.parse_edits()
    card = {"key": "Mini_Doomer", "name": "Mini Doomer", "hp": 5, "spd": 5, "problems": []}
    bs.apply_edits(card, edits, shown="Mini Doomer")
    assert card["name"] == "Tiny Doomer" and card["hp"] == 9 and card["siteOnly"] == ["name", "hp"] and card["pending"] == ["spd"]
    fields, meta = edits["mini doomer"]
    card = {"key": "Mini_Doomer", "name": "Mini Doomer", "hp": 5, "spd": 5, "problems": []}
    bs.apply_edits(card, {"mini doomer": (fields, {**meta, "_site": ["name"]})}, shown="Mini Doomer")   # hp unticked
    assert card["siteOnly"] == ["name"] and card["pending"] == ["hp", "spd"]
    refused = {"state": "refused", "for": {"name": "Tiny Doomer"}, "psd": "abcd1234", "reason": "r"}    # then ticked
    card = {"key": "Mini_Doomer", "name": "Mini Doomer", "hp": 5, "spd": 5, "problems": []}
    bs.apply_edits(card, {"mini doomer": (fields, {**meta, "_psd": refused})}, psd_sha="abcd1234ff", shown="Mini Doomer")
    assert "refused" not in card and card["siteOnly"] == ["name", "hp"]


def test_parser_notes_answered_as_the_bot_does(bs):
    """build_site's FIXED_BY and FIXED_BY_ALL are card_edits': NO STATS FOUND stays until all four stats or creatures are
    edited, so a card with a stat still unread keeps its note (B3, 2026-10-26)."""
    from commands.card_edits import CardEdits
    from commands.psd_parser import NO_STATS
    assert (bs.FIXED_BY, bs.FIXED_BY_ALL) == (CardEdits.FIXED_BY, CardEdits.FIXED_BY_ALL)
    note = f"{NO_STATS}: no HP, DEF, ATK or SPD bars read on this creature"
    four = {"hp": 6, "def": 2, "atk": 3, "spd": 4}
    for fields, left in (({"note": "n"}, [note]), ({"hp": 6}, [note]), ({**four, "spd": None}, [note]), (four, []),
                         ({"creatures": [four]}, [])):
        assert bs.answered(fields) == CardEdits.answered(fields), fields
        card = {"key": "Computer_Virus", "name": "Computer Virus", "type": "creature", "problems": [note]}
        bs.apply_edits(card, {"computer virus": (fields, {})}, shown="Computer Virus")
        assert card["problems"] == left, fields


def test_card_inks_line_the_psds_colours_up(bs):
    """cards.json `ink` (owner 11-07/08): the title's and Ability's accents on the card's text as the parser made it (line
    breaks, a type token injected); each layer's main ink and colours near it left out, so a one-colour title or black
    text gains none; a word edited away leaves no stray letters; badges and icons unpainted."""
    if not hasattr(bs, "card_inks"):
        pytest.skip("before TC's card colours land")
    layer = lambda i, text, colors, runs=None: {"layer": "Text/L", "id": i, "text": text, "color": colors[0], "visible": True,
                                                "colors": colors, **({"runs": runs} if runs else {})}
    rec = lambda path, title, ability: {"path": path, "at": "1", "text": [title, ability], "parts": {"title": [{"id": 1}], "ability": [{"id": 2}]}}
    pain = rec("A.psd", layer(1, "Double\nPain", ["#ffffff", "#cf0000"], [[0, 7, 0], [7, 11, 1]]),
               layer(2, "(1CD) Inflict 2T Burning.\nGain [fire] +1 ATK.", ["#ffffff", "#ff7a1a", "#eeeeee", "#00ff00"],
                     [[0, 17, 0], [17, 24, 1], [24, 26, 0], [26, 31, 2], [31, 37, 0], [37, 45, 3]]))
    shown = "(1CD) Inflict 2T Burning. Gain [fire] +1 ATK. [undead]"
    assert bs.card_inks({"name": "Double Pain", "ability": shown}, pain) == {"name": [[7, 11, "#cf0000"]], "ability": [[17, 24, "#ff7a1a"], [38, 45, "#00ff00"]]}
    assert bs.card_inks({"name": "Double Pain", "ability": shown.replace("Burning", "Freezing")}, pain)["ability"] == [[39, 46, "#00ff00"]]
    cards = [{"id": "a", "key": "A", "name": "Double Pain", "ability": shown, "psd": "A.psd"},
             {"id": "s", "key": "S", "name": "Springtrap", "ability": "Draw a card.", "psd": "S.psd"}]
    plain = rec("S.psd", layer(1, "Springtrap", ["#d4a416"]), layer(2, "Draw a card.", ["#000000"]))
    bs.join_details(cards, {"b1": pain, "b2": plain}, {"A.psd": "b1", "S.psd": "b2"})
    assert cards[0]["ink"]["ability"][0] == [17, 24, "#ff7a1a"] and "ink" not in cards[1]


def test_card_inks_on_a_real_card(bs, clone):
    """Tainted Lazarus: its two Ability layers' runs on the parsed ability (joined; no [undead] since 11-09, its creature 2
    header icon): the status words in their colours, the white title none."""
    if not hasattr(bs, "card_inks"):
        pytest.skip("before TC's card colours land")
    import commands.psd_details as pd
    import commands.psd_edit as pe
    from psd_tools import PSDImage
    vocab = json.loads((Path(__file__).resolve().parent / "goldens" / "_vocab.json").read_text(encoding="utf-8"))
    parser = pe.offline_parser(vocab["all_types"], vocab["stars"], vocab["manifest"])
    rel = "Creatures/Isaac/3 Stars/Tainted_Lazarus.psd"
    psd = PSDImage.open(str(clone / rel))
    ability = parser.parse(psd, rel).ability
    ink = bs.card_inks({"name": "Tainted Lazarus", "ability": ability}, pd.gather(psd, rel, "Tainted Lazarus", None, vocab["all_types"], parser))
    words = {ability[a:b]: c for a, b, c in ink["ability"]}
    assert list(ink) == ["ability"] and words["Blindness"] == "#f9f2c2" and words["Afraid"] == "#0060ff"
    assert ability.endswith("1T Afraid.") and max(b for _, b, _ in ink["ability"]) <= len(ability)


def test_font_filter_data_is_the_details_fonts(bs):
    """data/cardfonts.json (owner 10-25, the grid's Font filter): a card's fonts are its Details record's, a variant picture
    its PSD's, a PNG-only duplicate (no record) and an unreadable fonts section none; names by use, then name."""
    rec = lambda path, fonts: {"path": path, "at": "2026-10-25", "fonts": fonts}
    records = {"b1": rec("A.psd", {"Perpetua": {"layers": ["Text/Name"], "supplied": True}, "stoogedings-Regular": {"layers": ["Text/Ability"], "supplied": True}}),
               "b2": rec("B.psd", {"stoogedings-Regular": {"layers": ["Text/Ability"], "supplied": True}}),
               "b3": rec("C.psd", "unreadable")}
    cards = [{"id": "a", "key": "A", "name": "A", "psd": "A.psd"}, {"id": "a2", "key": "A2", "name": "A2", "psd": "A.psd", "dup": "a", "variantOf": "a"},
             {"id": "b", "key": "B", "name": "B", "psd": "B.psd"}, {"id": "b2", "key": "B2", "name": "B2", "dup": "b"},
             {"id": "c", "key": "C", "name": "C", "psd": "C.psd"}]
    files = bs.join_details(cards, records, {"A.psd": "b1", "B.psd": "b2", "C.psd": "b3"})
    got = bs.card_fonts(cards, files)
    assert got == {"fonts": ["stoogedings-Regular", "Perpetua"], "cards": {"a": [1, 0], "a2": [1, 0], "b": [0]}}
    for cid, ix in got["cards"].items():                  # the Details panel's list, in its order
        assert [got["fonts"][i] for i in ix] == list(files[cid]["fonts"])


def test_rulebook_picture_outline(bs):
    """A transparent rulebook picture's outline (owner 10-30: text wraps to a ball's edges): its opaque extent per band
    as polygon points in %, right side down, left side up; none for an opaque picture."""
    from PIL import Image, ImageDraw
    ball = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    ImageDraw.Draw(ball).ellipse((20, 0, 80, 100), fill=(255, 0, 0, 255))
    points = [tuple(float(v.rstrip("%")) for v in q.split()) for q in bs.picture_outline(ball).split(", ")]
    xs = [x for x, _ in points]
    assert min(xs) >= 19 and max(xs) <= 81 and points[0][1] == 0 and max(y for _, y in points) == 100
    assert min(xs) <= 21 and max(xs) >= 79 and points[0][0] < 70      # widest at the middle, narrow at the top
    assert bs.picture_outline(Image.new("RGB", (40, 40))) == ""
    assert bs.picture_outline(Image.new("RGBA", (40, 40), (1, 2, 3, 255))) == ""


def test_overhang_card_body(bs, tmp_path):
    """An OVERHANG card's own box (owner 10-31, Sumo Panda): the solid card inside see-through edges, art spilling out
    the left; none for an opaque picture."""
    from PIL import Image, ImageDraw
    pic = Image.new("RGBA", (120, 240), (0, 0, 0, 0))
    ImageDraw.Draw(pic).rectangle((18, 0, 116, 239), fill=(200, 180, 150, 255))
    ImageDraw.Draw(pic).ellipse((0, 60, 60, 100), fill=(255, 255, 255, 255))
    pic.save(tmp_path / "hang.png")
    Image.new("RGBA", (120, 240), (1, 2, 3, 255)).save(tmp_path / "solid.png")
    assert bs.card_body(tmp_path / "hang.png") == [0.15, 0.0, 0.975, 1.0]
    assert bs.card_body(tmp_path / "solid.png") is None


def test_early_images_are_the_builds(bs, repo, tmp_path, monkeypatch, capsys):
    """An early build's Images start (build_site.py --images <commit> <stop>): the card PNGs' images at that commit with
    their meta, not MDW's or Types'; none once stop exists. The build then makes no new file."""
    root, out, stop = repo["root"], tmp_path / "out", tmp_path / "stop"
    for other in ("MDW/Assets/back.png", "Types/fire.png"):
        _png(root / other, 99)
    at, _ = _commit(root, "Assets", 1_790_000_000)
    monkeypatch.setattr(bs, "REPO", str(root))
    monkeypatch.setattr(bs, "CACHE", str(out / ".cache"))
    stop.touch()
    bs.early_images(at, str(stop))
    assert not list((out / ".cache/img").iterdir())
    stop.unlink()
    bs.early_images(at, str(stop))
    pngs = {p: s for p, s in bs.repo_tree(at).items() if p.endswith(".png")}
    cards = {s for p, s in pngs.items() if p.startswith(("Creatures/", "Items/"))}
    assert len(pngs) == 5 and len(cards) == 3 and set(bs.load_image_meta()) == cards
    assert {n for s in cards for n in bs.image_names(s).values()} == {p.name for p in (out / ".cache/img").iterdir()}
    capsys.readouterr()
    _build(bs, monkeypatch, repo, out)
    assert "images: 3 cards, 0 new files" in capsys.readouterr().out


def test_small_image_from_the_720(bs, tmp_path, monkeypatch):
    """The 96 px image (SMALL_SIZES: table rows, 64 px lists) of a cached blob is made from its 720 px AVIF, no PNG
    download; its name follows the 720's settings too."""
    from PIL import Image
    monkeypatch.setattr(bs, "CACHE", str(tmp_path))
    (tmp_path / "img").mkdir()
    sha, known = "ab" * 20, [720, 1008, "#c8285a"]
    names = bs.image_names(sha)
    Image.new("RGB", (240, 336), (200, 40, 90)).save(tmp_path / "img" / names[720], "AVIF")
    for w in (240, 360):
        (tmp_path / "img" / names[w]).write_bytes(b"x")
    monkeypatch.setattr(bs, "image_bytes", lambda *a: pytest.fail("PNG downloaded"))
    assert bs.make_images("Creatures/A.png", sha, known) == (sha[:16], *known)
    with Image.open(tmp_path / "img" / names[96]) as im:
        assert im.size == (96, 134)
    tag = bs.thumb_tag(96)
    monkeypatch.setitem(bs.IMAGE_SIZES, 720, 49)
    assert bs.thumb_tag(96) != tag


def test_card_page_preview_is_the_pinned_png(bs, repo, tmp_path, monkeypatch):
    """A card page's og:image: its PNG on raw GitHub at the commit that last changed it (no -og.jpg made); without
    history, at the build's commit (card_raw)."""
    monkeypatch.setattr(bs, "SITE_URL", "https://site")
    og = lambda d, i: re.search(r'og:image" content="([^"]+)"', (d / "dist" / "c" / i / "index.html").read_text(encoding="utf-8"))[1]
    _build(bs, monkeypatch, repo, tmp_path / "a")
    c1, _, c3 = repo["commits"]
    raw = f"https://raw.githubusercontent.com/{bs.CARD_REPO}/"
    assert og(tmp_path / "a", "alpha") == f"{raw}{c3}/{ALPHA.replace(' ', '%20')}.png"
    assert og(tmp_path / "a", "watermelon") == f"{raw}{c1}/{MELON.replace(' ', '%20')}.png"
    assert not list((tmp_path / "a" / "dist" / "thumbs").glob("*.jpg"))
    monkeypatch.setattr(bs, "git_history", lambda *a: None)
    monkeypatch.setattr(bs, "_card_raw", f"{raw}{'f' * 40}/")
    _build(bs, monkeypatch, repo, tmp_path / "b")
    assert og(tmp_path / "b", "alpha") == f"{raw}{'f' * 40}/{ALPHA.replace(' ', '%20')}.png"


def test_guards_name_rule_matches_the_site(bs):
    """AutoCroissant's R14 (card_standards.miscased_names) and build_site.name_links share text.js's rule of which words
    name a card: on the same text, R14's names are the links typed in another case."""
    import commands.card_standards as cs
    cards = [{"name": "Silithid", "key": "Silithid", "type": "minion"}, {"name": "Stay", "key": "Stay", "type": "creature"},
             {"name": "Polevaulter Zombie", "key": "Polevaulter Zombie", "type": "minion"}, {"name": "Burning", "key": "Burning", "type": "debuff"},
             {"name": "Cat", "key": "Cat", "type": "minion"}, {"name": "Flip", "key": "Flip", "type": "item"}]
    text = ("(2GCD)(OYT) summon a silithid. Stay here, stay calm. Summon a polevaulter zombie and a Cat. Cat: +2 SPD.\n"
            "Inflict 2T1 burning. Flip to the other side. (BOGT) Flip. Revive the cat.")
    links = bs.name_links(cards, {}, {})
    typed = {c["name"] for c in links(text)}
    miscased = cs.miscased_names(text, tuple((c["name"], c["key"], c["type"]) for c in cards))
    assert {right for _, right in miscased} == {"Silithid", "Polevaulter Zombie", "Burning", "Cat"}
    assert {right for _, right in miscased} <= typed


def test_a_card_naming_itself_mentions_itself(tmp_path):
    """Owner 2026-10-30: a card whose text names it is in its own Mentions and Mentioned by (text.js mentions)."""
    text_js = BUILD_SITE.parents[1] / "assets" / "text.js"
    if not text_js.is_file():
        pytest.skip("no text.js")
    probe = tmp_path / "probe.mjs"
    probe.write_text(f"const t = await import({json.dumps(text_js.as_uri())});\n"
                     "const r = t.makeRenderer({ cards: [{ id: 'mk', name: 'Mirror Knight', key: 'Mirror Knight', type: 'creature' }], icons: {} });\n"
                     "process.stdout.write(JSON.stringify([...r.mentions('(OD) Summon another Mirror Knight.')]));\n", encoding="utf-8")
    done = subprocess.run(["node", str(probe)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == ["mk"]


def test_a_web_chapter_is_not_printed(bs, tmp_path):
    """Owner 2026-11-07: a chapter titled {.web} is on the website only (rulebook.js printed, build_site printed_chapters)."""
    book_js = BUILD_SITE.parents[1] / "assets" / "rulebook.js"
    if not hasattr(bs, "printed_chapters"):
        pytest.skip("a site before {.web}")
    book = [{"id": "stats", "source": "# Stats\n\n## HEALTH {.hp}\nWords.\n"}, {"id": "font", "source": "# Font {.web}\n\nWords.\n"}]
    assert [c["id"] for c in bs.printed_chapters(book)] == ["stats"]
    probe = tmp_path / "probe.mjs"
    probe.write_text(f"const rb = await import({json.dumps(book_js.as_uri())});\n"
                     f"process.stdout.write(JSON.stringify(rb.printed({json.dumps(book)}).map(c => c.id)));\n", encoding="utf-8")
    done = subprocess.run(["node", str(probe)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == ["stats"]


def test_rulebook_pictures_sized_before_they_load(tmp_path):
    """rulebook.js: a mascot (rules.json's size) and a card picture (tw/th) carry width/height, so nothing above a contents
    link's chapter grows while the jump runs; a picture of unknown size none."""
    book_js = BUILD_SITE.parents[1] / "assets" / "rulebook.js"
    if "rb-mascot" not in book_js.read_text(encoding="utf-8"):
        pytest.skip("a rulebook without mascots")
    probe = tmp_path / "probe.mjs"
    probe.write_text(f"const rb = await import({json.dumps(book_js.as_uri())});\n"
                     "const ctx = { size: f => (f === 'kirby' ? [120, 96] : null), card: n => (n === 'Alpha' ? { id: 'a1', name: 'Alpha', tw: 360, th: 504 } : null),\n"
                     "  cardImg: () => 't.avif', imgUrl: f => f + '.webp' };\n"
                     "const html = rb.chapterHtml({ id: 'x', source: '# X\\n\\n> [!kirby] Hi.\\n\\n![[Alpha]]\\n\\n> [!sonic] Who?\\n' }, ctx);\n"
                     "process.stdout.write(JSON.stringify([...html.matchAll(/<img [^>]*>/g)].map(m => [m[0].match(/width=\"(\\d+)\" height=\"(\\d+)\"/)?.slice(1)])));\n",
                     encoding="utf-8")
    done = subprocess.run(["node", str(probe)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [[["120", "96"]], [["360", "504"]], [None]]


def test_a_heavy_commit_links_the_cards_history(tmp_path):
    """text.js commitLink: a heavy commit (cards.json heavy) links GitHub's history of the card's PNG from it, its path
    encoded, and nothing without a path; any other commit its commit page."""
    text_js = BUILD_SITE.parents[1] / "assets" / "text.js"
    if not text_js.is_file():
        pytest.skip("no text.js")
    probe = tmp_path / "probe.mjs"
    probe.write_text(f"const t = await import({json.dumps(text_js.as_uri())});\n"
                     "t.setHeavy(['aaaaaaaaaaaa']); const a = 'a'.repeat(40), b = 'b'.repeat(40), p = \"Items/2 Stars/Ahn'Qiraj #1.png\";\n"
                     "process.stdout.write(JSON.stringify([t.commitLink('o/r', b, 'x', p), t.commitLink('o/r', a, 'x'), t.commitLink('o/r', a, 'x', p)]));\n",
                     encoding="utf-8")
    done = subprocess.run(["node", str(probe)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    hrefs = [html_unescape(m[1]) if m else None for m in (re.search(r'href="([^"]*)"', h) for h in json.loads(done.stdout))]
    assert hrefs == [f"https://github.com/o/r/commit/{'b' * 40}", None,
                     f"https://github.com/o/r/commits/{'a' * 40}/Items/2%20Stars/Ahn'Qiraj%20%231.png"]


def test_tts_details_for_the_save(bs):
    """data/tts.json (owner 10-30, the TTS hover strip and details): types' meanings from the chapters (an icon heading's
    section to the page break, a "[token]:" line, a line ending in tokens; secret text, "(See: ...)" and non-icons left
    out), tags as text.js reads them, the statuses an ability names, cards by PNG path, hidden cards left out, only the stats
    a card has (an active item: HP and DEF); a [field] card's Field Cards first in r; aliases not part of the name (al); the
    ability's words in the site's palette (ac, owner 2026-11-11: never the card's own colours or Card Edits colors)."""
    chapters = [{"id": "creature-types", "num": "20", "title": "Creature Types", "source":
                 "# Creature Types\n## FIRE {icon=fire}\nImmune to [[Burning]] (See: [“X”](#/rules/x))\n## WIND {icon=wind}\nSwap.\n"},
                {"id": "item-types", "num": "14", "title": "Item Types", "source":
                 "## [support] SUPPORT\nGeneral use. [joke]{.secret}\n=== page ===\nNot this.\n"},
                {"id": "minions", "num": "15", "title": "Minions", "source": "[offensive_minion]:Same field as the [fire] summoner.\n"},
                {"id": "status-effects", "num": "19", "title": "Status Effects", "source": "One instance only. [debuff] [nothere]\n"},
                {"id": "timing-tags", "num": "23", "title": "Timing Tags", "source": "| Tag | Meaning |\n|---|---|\n| (XCD) | Wait X turn(s) |\n"},
                {"id": "damage-types", "num": "21", "title": "Damage Types", "source": "## PIERCING\nIgnores DEF.\n"},
                {"id": "negation", "num": "22", "title": "Negation", "source": "## NEGATION\nTaken out of play.\n"},
                {"id": "field-cards", "num": "18", "title": "Field Cards", "source": "## PLAYING\nOne at a time.\n"}]
    icons = {k: {} for k in ("fire", "wind", "support", "offensiveminion", "debuff")}
    cards = [{"name": "Hot Guy", "type": "creature", "types": ["fire", "offensive_minion"], "png": "Creatures/Hot_Guy.png", "stars": 3,
              "hp": 5, "def": 1, "atk": 2, "spd": 3, "ability": "(1cd) Inflict 2T burning. (3CD) [fire] again. (OD) Burning! Starts with 2 Dark Tentacles and a Secret. Deals 2 CDMG; negates.", "author": "Bob"},
             {"name": "Burning", "type": "debuff", "types": ["debuff"], "png": "Aux/Burning.png", "ability": "3 DMG, can't affect [fire]"},
             {"name": "Secret", "type": "item", "png": "Items/Secret.png", "hidden": True, "ability": "Burning"},
             {"name": "Dark Tentacle", "type": "minion", "png": "Aux/Dark_Tentacle.png", "ability": ""},
             {"name": "Gate", "type": "field", "types": ["field"], "png": "Field/Gate.png", "ability": "(3CD) summon a dark tentacle, then stay. Can't be negated."},
             {"name": "Stay", "type": "creature", "png": "Creatures/Stay.png", "ability": ""},
             {"name": "Mirror Knight", "type": "creature", "png": "Creatures/Mirror_Knight.png", "ability": "(OD) Summon another Mirror Knight.",
              "aliases": ["MK", "Knight"], "webAliases": ["mk"]},
             {"name": "Totem", "type": "item", "subtype": "active", "png": "Items/Totem.png", "hp": 2, "def": 0, "ability": ""}]
    got = bs.tts_details(cards, chapters, icons, bs.tag_meanings(chapters))
    assert got["types"] == {"fire": {"n": "Fire", "x": "Immune to Burning", "c": "creature-types"},
                            "wind": {"n": "Wind", "x": "Swap.", "c": "creature-types"},
                            "support": {"n": "Support", "x": "General use.", "c": "item-types"},
                            "offensiveminion": {"n": "Offensive Minion", "x": "Same field as the fire summoner.", "c": "minions"},
                            "debuff": {"n": "Debuff", "x": "One instance only.", "c": "status-effects"}}
    assert got["tags"] == {"1CD": {"g": "turn", "m": "Wait 1 turn"}, "3CD": {"g": "turn", "m": "Wait 3 turns"},
                           "OD": {"g": "trigger", "m": "On death"}}
    assert got["statuses"] == {"Burning": {"x": "3 DMG, can't affect fire", "k": "debuff", "p": "Aux/Burning.png"}}
    assert set(got["cards"]) == {"Creatures/Hot_Guy.png", "Aux/Burning.png", "Aux/Dark_Tentacle.png", "Field/Gate.png", "Creatures/Stay.png",
                                 "Creatures/Mirror_Knight.png", "Items/Totem.png"}
    assert got["cards"]["Items/Totem.png"]["s"] == "HP 2 DEF 0"
    assert got["cards"]["Creatures/Mirror_Knight.png"]["m"] == ["Creatures/Mirror_Knight.png"]   # a card naming itself spawns itself (owner 10-30)
    assert got["cards"]["Creatures/Mirror_Knight.png"]["al"] == ["mk"]   # "Knight" is in the name
    assert got["cards"]["Field/Gate.png"]["m"] == ["Aux/Dark_Tentacle.png"]   # as text.js links: a lowercase minion after its verb
    assert got["cards"]["Field/Gate.png"]["r"] == ["field-cards", "negation"]   # [field] has no meaning: its chapter by hand
    assert got["cards"]["Creatures/Hot_Guy.png"] == {
        "n": "Hot Guy", "k": "Creature", "st": 3, "s": "HP 5 DEF 1 ATK 2 SPD 3",
        "a": "(1cd) Inflict 2T burning. (3CD) fire again. (OD) Burning! Starts with 2 Dark Tentacles and a Secret. Deals 2 CDMG; negates.",
        "t": ["fire", "offensiveminion"], "g": ["1CD", "3CD", "OD"], "x": ["Burning"], "m": ["Aux/Dark_Tentacle.png"], "r": ["damage-types", "negation"], "by": "Bob"}
    assert "x" not in got["cards"]["Aux/Burning.png"] and got["chapters"]["minions"] == {"num": "15", "title": "Minions"}
    if not hasattr(bs, "tts_words"):                    # before TC's site palette lands
        return
    hot = cards[0]                                     # its own colours and an edit's: the card image's, never drawn here
    hot["ink"] = {"name": [[4, 7, "#ffd21f"]], "ability": [[0, 5, "#00ff00"]]}
    hot["colors"] = [{"field": "ability", "phrase": "again", "color": "#ff0000"}]
    palette = bs.text_palette(cards, {"Burning": "#db923d"})
    got = bs.tts_details(cards, chapters, icons, bs.tag_meanings(chapters), palette)
    assert got["cards"]["Creatures/Hot_Guy.png"]["ac"] == [["Burning", "#db923d"]] and "nc" not in got["cards"]["Creatures/Hot_Guy.png"]
    assert got["statuses"]["Burning"]["c"] == "#db923d"                       # its STATUS EFFECTS line's colour
    words = bs.tts_words("Pay (2 points): +2 ATK, 3 SP, [fire] HPs HP (1CD) Burning, Dark; HP and HP.", {**palette, "statuses": {"Burning": "#db923d", "Dark": "#101010"}}, icons)
    dark = [["Dark", "#646464"]] if hasattr(bs, "tts_seen") else []   # too dark: a lighter shade (TC 11-11), plain before
    assert words == [["2 points", "#00ffae"], ["+2 ATK", "#ff3b3b"], ["3 SP", "#ffff00"], ["HP", "#2fbf3a", 2], ["Burning", "#db923d"],
                     *dark, ["HP", "#2fbf3a"], ["HP", "#2fbf3a"]]   # a cost's words too (no badge in TTS), not a tag; HPs first


def test_badge_font_preloaded_where_badges_show(bs):
    """A card page preloads the badge font when its text draws a badge (text.js's tags and costs, any case), not otherwise."""
    assert all(map(bs.has_badges, ["(1cd) Hit.", "(-1/1GCD) Wait.", "Pay (2 points).", "(Action) Swap.", "(0.5 SP)"]))
    assert not any(map(bs.has_badges, ["Deal 2 (x) damage.", "(Actions) twice", "x(2 points)", "Plain text."]))


def test_site_workflow_sets_up_in_the_background(tmp_path):
    """site.yml's Setup start returns at once, uv, npm, the PSD sizes, the data clone and the deploy's versions (the version
    job's artifact, not waited for; none when that job did not run in this attempt) running on (their output not held);
    Setup wait prints each one's log and fails on uv or npm, only warns on the sizes (3 tries); a failed data clone: the
    data checked out between it and the build. uv is fetched there
    (no setup-uv), else from the GitHub release, else pip installs. An early build's Images start runs once uv is in;
    Setup wait stops it, prints its log, only warns on a failure (none started: no wait)."""
    yaml = pytest.importorskip("yaml")
    path = SITE_SRC / ".github" / "workflows" / "site.yml"
    if not path.is_file():
        pytest.skip(f"no {path}")
    steps = yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]["build"]["steps"]
    names = [s.get("name") or s.get("id") or s.get("uses") for s in steps]
    assert names.index("py") < names.index("python") < names.index("Setup start") < names.index("Setup wait") == names.index("build") - 2
    fallback = steps[names.index("build") - 1]
    assert fallback["if"] == "steps.setup.outputs.data != 'true'"
    assert fallback["with"] == {"ref": "data", "path": "carddata", "fetch-depth": 1}
    assert steps[names.index("python")]["if"] == "steps.py.outputs.python == ''"
    gone = ("actions/configure-pages", "astral-sh/setup-uv", "actions/upload-pages-artifact")    # never downloaded
    assert not [s for s in steps if s.get("uses", "").split("@")[0] in gone]
    assert names.index("cache-before") < names.index("Images start") < names.index("Data wait")
    assert steps[names.index("Images start")]["if"] == "inputs.after"
    start, images, wait = (steps[names.index(n)]["run"] for n in ("Setup start", "Images start", "Setup wait"))
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    call = ('echo "$0 $*" >> "$RUNNER_TEMP/calls"; [ -z "$HOLD" ] '      # held till go (60 s at most)
            '|| until [ -e "$RUNNER_TEMP/go" ] || [ $(( held += 1 )) -gt 6000 ]; do /bin/sleep 0.01; done; ')
    curl = ('echo "$0 $*" >> "$RUNNER_TEMP/calls"\ncase "$*" in *releases.astral.sh*) [ -z "$ASTRAL_CODE" ] || exit "$ASTRAL_CODE";; esac\n'
            'while [ $# -gt 1 ]; do [ "$1" != -o ] || out=$2; shift; done\ncp "$TARBALL" "$out"')
    for name, body in {"npm": call + 'echo "npm said"; exit "${NPM_CODE:-0}"', "curl": curl,
                       "gh": call + 'case "$*" in *runs/5/artifacts\\?name=pages-versions-3*) echo 77;; *artifacts/77/zip) cat "$ZIP";; '
                              '*runs/5/jobs\\?filter=latest*) echo "${VERSION_AT:-3}";; *) echo \'{"tree": []}\';; esac; '
                              'exit "${GH_CODE:-0}"', "timeout": 'shift; exec "$@"',
                       "git": 'case "$1" in clone) ' + call + '[ "$GIT_CONFIG_KEY_0" = http.extraheader ] || exit 9; '
                              'exit "${CLONE_CODE:-0}";; esac\necho 0123abcd',
                       "python": call + 'until [ -e "$4" ]; do /bin/sleep 0.01; done; echo "python said"',     # till stopped
                       "sha256sum": 'cat > /dev/null; exit "${SHA_CODE:-0}"', "pip-python": call + 'echo "pip said"',
                       "sleep": "exec /bin/sleep 0.01", "seq": 'exec /usr/bin/seq "$(( $1 < 3 ? $1 : 3 ))"'}.items():   # 3 tries
        (bin_ / name).write_text(f"#!/bin/sh\n{body}\n")
        (bin_ / name).chmod(0o755)
    uv = f"#!/bin/sh\n{call}echo \"uv said\"; exit \"${{UV_CODE:-0}}\"\n".encode()
    with tarfile.open(tmp_path / "uv.tar.gz", "w:gz") as tar:      # the release's layout
        info = tarfile.TarInfo("uv-x86_64-unknown-linux-gnu/uv")
        info.size, info.mode = len(uv), 0o755
        tar.addfile(info, io.BytesIO(uv))
    versions = f"{'c1' * 20}\n{'c2' * 20}\n"
    with zipfile.ZipFile(tmp_path / "versions.zip", "w") as zf:        # the version job's artifact
        zf.writestr("versions.txt", versions)
    shell = lambda run, env, limit=30: subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", run], env=env,
                                                      capture_output=True, text=True, timeout=limit)
    # 1: a wrong checksum from both: pip. 2: the mirror down: the GitHub release
    for n, (codes, failed, warns) in enumerate((({"HOLD": "1"}, "", False), ({"GH_CODE": "1", "SHA_CODE": "1"}, "", True),
                                                ({"UV_CODE": "2", "NPM_CODE": "3", "ASTRAL_CODE": "22", "CLONE_CODE": "128", "VERSION_AT": "2"},
                                                 " uv npm", False))):            # 2: the version job of attempt 2 (a re-run)
        temp = tmp_path / f"t{n}"
        temp.mkdir()
        (temp / "out").write_text("")
        env = {"PATH": f"{bin_}:/usr/bin:/bin", "HOME": str(tmp_path), "RUNNER_TEMP": str(temp), "GITHUB_REPOSITORY": "o/r",
               "GITHUB_SERVER_URL": "https://x", "GITHUB_OUTPUT": str(temp / "out"), "TARBALL": str(tmp_path / "uv.tar.gz"),
               "UV_PYTHON": str(bin_ / "pip-python"), "GITHUB_RUN_ID": "5", "GITHUB_RUN_ATTEMPT": "3",
               "ZIP": str(tmp_path / "versions.zip"), **codes}
        try:
            done = shell(start, env)
            assert done.returncode == 0 and not (n == 0 and list((temp / "setup").glob("*.code"))), done.stderr   # not waited for
        finally:
            (temp / "go").touch()                                     # 0: the held jobs go on (never left spinning)
        assert n == 1 or shell(images, env).returncode == 0         # 1: not an early build
        end = time.monotonic() + 120                                 # a loaded machine: slow, never stuck
        while n == 0 and time.monotonic() < end:                     # 0: Setup wait once it runs (else never started)
            if (temp / "calls").exists() and "--images" in (temp / "calls").read_text():
                break
            time.sleep(0.01)
        done = shell(wait, env, 120)
        out = done.stdout
        assert (done.returncode != 0) == bool(failed) and ("::error::" in out) == bool(failed), (out, done.stderr)
        assert not failed or f"::error::failed:{failed}" in out
        assert ("::warning::PSD sizes failed" in out) == warns and ("pip said" if n == 1 else "uv said") in out and "npm said" in out
        assert ("::warning::uv not fetched" in out) == (n == 1)
        assert out.index(f"::group::uv (exit {codes.get('UV_CODE', 0)})") < out.index(f"::group::npm (exit {codes.get('NPM_CODE', 0)})")
        calls = (temp / "calls").read_text().splitlines()
        assert sum("/gh api repos/o/r/git/trees/0123abcd?recursive=1" in c for c in calls) == (3 if warns else 1)
        assert any(("/pip-python -m pip" if n == 1 else "/uv/uv pip") + " install -q -r site/build/requirements.txt" in c for c in calls)
        hosts = [c.split()[-1].split("/")[2] for c in calls if c.split()[0].endswith("/curl")]
        assert hosts == ["releases.astral.sh"] + ["github.com"] * (n > 0)
        assert (temp / "tree.json").read_text().strip() == '{"tree": []}'
        early = {0: "0", 2: "uv"}.get(n)
        assert ("::group::images" in out) == bool(early) and ("::warning::early images failed" in out) == (n == 2)
        assert not early or f"::group::images (exit {early})" in out and ("python said" in out) == (n == 0)
        assert sum(f"build_site.py --images 0123abcd {temp}/setup/images.stop" in c for c in calls) == (n == 0)
        assert sum("/git clone -q --depth 1 --no-tags --single-branch --branch data https://x/o/r carddata" in c for c in calls) == 1
        assert ((temp / "out").read_text() == "data=true\n") == (n != 2) and ("::warning::the data clone failed" in out) == (n == 2)
        code, end = temp / "setup" / "versions.code", time.monotonic() + 120
        while time.monotonic() < end:                                # 1: every call fails, all tries
            if code.exists() and code.read_text().strip():
                break
            time.sleep(0.01)
        assert code.read_text().strip() == ("0" if n == 0 else "1")
        assert ((temp / "versions.txt").read_text() if n == 0 else versions) == versions
        log = (temp / "setup" / "versions.log").read_text()
        artifacts = [c for c in (temp / "calls").read_text().splitlines() if "/artifacts" in c]
        assert ("did not run in attempt 3" in log) == (n == 2) and bool(artifacts) == (n != 2)


def test_site_workflow_python_from_the_tool_cache(tmp_path):
    """site.yml's py step puts the image's newest 3.14.x on PATH with its lib, as setup-python does; none: nothing (setup-python)."""
    yaml = pytest.importorskip("yaml")
    path = SITE_SRC / ".github" / "workflows" / "site.yml"
    if not path.is_file():
        pytest.skip(f"no {path}")
    run = next(s for s in yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]["build"]["steps"] if s.get("id") == "py")["run"]
    for n, have in enumerate((["3.13.9/x64", "3.14.2/x64", "3.14.10/x64", "3.14.11/x64-freethreaded", "3.15.0/x64"], ["3.13.9/x64"])):
        cache, out = tmp_path / f"c{n}", tmp_path / f"o{n}"
        for v in have:
            (cache / "Python" / v / "bin").mkdir(parents=True)
            (cache / "Python" / v / "bin" / "python").touch(mode=0o755)
        out.mkdir()
        files = {k: out / k for k in ("GITHUB_PATH", "GITHUB_ENV", "GITHUB_OUTPUT")}
        env = {"PATH": "/usr/bin:/bin", "RUNNER_TOOL_CACHE": str(cache), "LD_LIBRARY_PATH": "/old", **{k: str(f) for k, f in files.items()}}
        done = subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", run], env=env, capture_output=True, text=True)
        got = {k: f.read_text() if f.exists() else "" for k, f in files.items()}
        py = cache / "Python" / "3.14.10" / "x64"
        assert done.returncode == 0 and got == ({"GITHUB_PATH": f"{py}/bin\n", "GITHUB_ENV": f"LD_LIBRARY_PATH={py}/lib:/old\n",
                                                 "GITHUB_OUTPUT": f"python={py}/bin/python\n"} if n == 0 else
                                                dict.fromkeys(files, "")), (done.stderr, got)


def test_site_image_cache_saved_whenever_the_build_changed_it(tmp_path):
    """site.yml's card images: restored only; saved last (a passed build too) when their names or JSON indexes changed,
    under the run id (the newest entry, restore-keys' pick); the job's saved (caches.yml prunes) only after a save."""
    yaml = pytest.importorskip("yaml")
    path = SITE_SRC / ".github" / "workflows" / "site.yml"
    if not path.is_file():
        pytest.skip(f"no {path}")
    job = yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]["build"]
    step = {s["id"]: s for s in job["steps"] if "id" in s}
    restore, before, after, save = (step[n] for n in ("cache", "cache-before", "cache-after", "cache-save"))
    assert restore["uses"].startswith("actions/cache/restore@") and restore["with"]["restore-keys"] == "card-images-v2-"
    assert job["steps"][-2:] == [after, save] and before["run"] == after["run"] and after["if"].startswith("always() && ")
    assert save["if"] == "always() && steps.cache-after.outputs.names != '' && steps.cache-after.outputs.names != steps.cache-before.outputs.names"
    assert save["with"] == {"path": "site/.cache", "key": restore["with"]["key"] + "-${{ github.run_id }}-${{ github.run_attempt }}"}
    assert job["outputs"]["saved"] == "${{ steps.og-save.outcome == 'success' || steps.cache-save.outcome == 'success' }}"
    cache = tmp_path / "site" / ".cache"
    (cache / "img").mkdir(parents=True)
    out = tmp_path / "out"
    def names(*change):
        for f, text in change:
            (cache / f).write_text(text)
        out.unlink(missing_ok=True)
        subprocess.run(["bash", "-e", "-c", after["run"]], cwd=tmp_path, env={**environ, "GITHUB_OUTPUT": str(out)}, check=True)
        return out.read_text()
    seen = [names(("images.json", "{}"))]
    assert names() == seen[0]
    for change in (("img/a.avif", "x"), ("fonts.json", '{"f": 1}'), ("fonts.json", '{"f": 2}')):    # a file, an index
        seen.append(names(change))
    assert len(set(seen)) == 4


def test_site_deploy_under_a_version_of_its_own(tmp_path):
    """site.yml's deployment: the version job (the one that writes) makes two commits, the run's commit their parent and
    its tree theirs (3 tries a call); each try deploys under its own (TRY; none handed over: GITHUB_SHA) with the OIDC
    token (masked), polling its status. 2 min syncing files while a newer main build waits (never an early build):
    cancelled and handed on (green: no retry); a failed cancel polls on. A failed deploy: red, no cancel; 6 minutes or 15
    unread statuses: cancelled. The retry, on the same artifact: the first commit's deploy still syncing files at
    the end (stuck), the same script under the second; else, or that failing too, deploy-pages under GITHUB_SHA (a
    version never fails a deploy)."""
    yaml = pytest.importorskip("yaml")
    path = SITE_SRC / ".github" / "workflows" / "site.yml"
    if not path.is_file():
        pytest.skip(f"no {path}")
    jobs = yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]
    steps = {s.get("id"): s for s in jobs["build"]["steps"]}
    assert jobs["build"]["permissions"]["contents"] == "read" and jobs["version"]["permissions"] == {"contents": "write"}
    assert jobs["version"]["continue-on-error"] is True
    made, upload = jobs["version"]["steps"]
    assert upload["if"] == "always()"
    assert upload["uses"].startswith("actions/upload-artifact@") and upload["with"]["name"] == "pages-versions-${{ github.run_attempt }}"
    run, retry = steps["deployment"]["run"], steps["retry"]
    assert "-ge 120 ]" in run and "-lt 360 ]" in run and steps["deployment"]["continue-on-error"] is True
    assert steps["deployment"]["env"] == {"GH_TOKEN": "${{ github.token }}", "ARTIFACT": "${{ steps.artifact.outputs.artifact-id }}",
                                          "TRY": "1"}
    assert retry["run"] == 'exec bash "$RUNNER_TEMP/deploy.sh"' and retry["env"]["TRY"] == "2"
    assert retry["env"]["ARTIFACT"] == steps["deployment"]["env"]["ARTIFACT"]
    pages, order = steps["pages"], [s.get("id") or s.get("if") for s in jobs["build"]["steps"]]
    assert retry["if"] == "steps.deployment.outcome == 'failure' && steps.deployment.outputs.stuck == 'true'"
    assert retry["continue-on-error"] is True and order.index("retry") + 2 == order.index("pages")
    assert order[order.index("retry") + 1] == "steps.retry.outcome == 'failure'"           # the cancel settles
    assert pages["if"] == ("steps.deployment.outcome == 'failure' && (steps.deployment.outputs.stuck != 'true' "
                           "|| steps.retry.outcome == 'failure')")
    assert pages["uses"].startswith("actions/deploy-pages@") and "artifact_name" not in pages["with"]      # github-pages
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    gh = r"""echo "$*" >> "$RUNNER_TEMP/calls"
case "$*" in
  *git/commits/*) [ ! -e "$RUNNER_TEMP/fail" ] || { rm "$RUNNER_TEMP/fail"; echo '{"message":"Server Error"}'; exit 1; }; echo tree1;;
  *"POST repos/o/r/git/commits "*) [ ! -e "$RUNNER_TEMP/failpost" ] || { rm "$RUNNER_TEMP/failpost"; echo '{"message":"Bad"}'; exit 1; }
    n=$(( $(cat "$RUNNER_TEMP/made" 2> /dev/null || echo 0) + 1 )); echo "$n" > "$RUNNER_TEMP/made"; printf "c$n%.0s" $(seq 20); echo;;
  *"pages/deployments --input -") cat > "$RUNNER_TEMP/body"; v=$(jq -r .pages_build_version "$RUNNER_TEMP/body")
    echo "{\"id\": \"$v\", \"status_url\": \"https://x/$v\", \"page_url\": \"https://o.github.io/r/\"}";;
  *cancel*) [ "$CANCEL" = 0 ] || echo '{"message":"Conflict"}'; exit "$CANCEL";;
  *runs*) while [ $# -gt 1 ]; do [ "$1" != --jq ] || f=$2; shift; done; jq -r "$f" "$RUNS";;
  *pages/deployments/*) s=$(head -1 "$STATUSES"); [ "$(wc -l < "$STATUSES")" -le 1 ] || sed -i.bak 1d "$STATUSES"
    [ "$s" != ERR ] || { echo '{"message":"Bad Gateway"}'; exit 1; }; echo "$s";;
esac"""
    for name, body in {"gh": gh, "curl": """echo '{"value": "tok3n"}'""", "sleep": "exec /bin/sleep 0.01",
                       "timeout": 'shift; exec "$@"'}.items():
        (bin_ / name).write_text(f"#!/bin/sh\n{body}\n")
        (bin_ / name).chmod(0o755)
    v1, v2, sha = "c1" * 20, "c2" * 20, "ab" * 20
    temp = tmp_path / "version"
    temp.mkdir()
    (temp / "fail").touch()                                          # the tree's first read fails, and the first commit
    (temp / "failpost").touch()
    env = {"PATH": f"{bin_}:/usr/bin:/bin", "RUNNER_TEMP": str(temp), "GITHUB_REPOSITORY": "o/r", "GITHUB_RUN_ID": "5",
           "GITHUB_SHA": sha}
    done = subprocess.run(["bash", "-e", "-c", made["run"]], env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and (temp / "versions.txt").read_text() == f"{v1}\n{v2}\n", (done.stdout, done.stderr)
    assert (temp / "calls").read_text().splitlines() == [f"api repos/o/r/git/commits/{sha} --jq .tree.sha"] * 2 + [
        f"api -X POST repos/o/r/git/commits -f message=Pages deploy: run 5, try {n} -f tree=tree1 -f parents[]={sha} --jq .sha"
        for n in (1, 1, 2)]
    runs = tmp_path / "runs.json"            # 7: the newest waiting main build (4 older, 8 done, 9 an early build)
    runs.write_text(json.dumps({"workflow_runs": [{"id": 4, "status": "queued", "display_title": "Card browser"},
                                                  {"id": 9, "status": "queued", "display_title": "Card browser (after 3 data)"},
                                                  {"id": 7, "status": "queued", "display_title": "Card browser"},
                                                  {"id": 8, "status": "completed", "display_title": "Card browser"}]}))
    fast = {"-ge 120 ]": "-ge 0 ]"}
    stuck = {"-lt 360 ]": '-lt 360 ] && [ "$last" != syncing_files ]'}     # times out after a status read, not at a clock tick
    for n, (statuses, cancel, swap, code, said, cancels) in enumerate((
            (["deployment_queued", "syncing_files", "succeed"], "0", {}, 0, "Deployed in", 0),
            (["syncing_files"], "0", fast, 0, "Run 7 waits: deploy handed to it", 1),
            (["syncing_files"] * 3 + ["succeed"], "22", fast, 0, "Run 7 waits: cancel failed", 1),
            (["deployment_queued", "deployment_failed"], "0", {}, 1, "::error::deployment_failed", 0),
            (["deployment_queued"], "0", {"-lt 360 ]": "-lt 0 ]"}, 1, "::error::not deployed in 6 minutes", 1),
            (["ERR"], "0", {"-lt 15 ]": "-lt 2 ]"}, 1, "::error::15 status reads failed", 1),   # the 2nd failed read, as the 15th
            (["syncing_files"], "0", stuck, 1, "::error::not deployed in 6 minutes", 1))):   # stuck
        temp = tmp_path / f"t{n}"
        (temp / "setup").mkdir(parents=True)
        (temp / "setup" / "versions.code").write_text("0\n")          # the hand-over
        (temp / "versions.txt").write_text(f"{v1}\n{v2}\n")
        (temp / "statuses").write_text("\n".join(statuses) + "\n")
        env = {"PATH": f"{bin_}:/usr/bin:/bin", "RUNNER_TEMP": str(temp), "GITHUB_REPOSITORY": "o/r", "GITHUB_RUN_ID": "5",
               "GITHUB_SHA": sha, "GITHUB_OUTPUT": str(temp / "out"), "GITHUB_STEP_SUMMARY": str(temp / "summary"),
               "ARTIFACT": "42", "TRY": "1", "ACTIONS_ID_TOKEN_REQUEST_URL": "https://t/?x=1",
               "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "rt", "CANCEL": cancel, "STATUSES": str(temp / "statuses"), "RUNS": str(runs)}
        step = run
        for a, b in swap.items():
            assert a in step, a
            step = step.replace(a, b)
        done = subprocess.run(["bash", "-e", "-c", step], env=env, capture_output=True, text=True, timeout=120)
        calls = (temp / "calls").read_text().splitlines()
        out = done.stdout
        assert done.returncode == code and said in out and "::warning::" not in out, (n, out, done.stderr)
        assert not [c for c in calls if "git/commits" in c]
        assert json.loads((temp / "body").read_text()) == {"artifact_id": 42, "pages_build_version": v1, "oidc_token": "tok3n"}
        assert [line for line in out.splitlines() if "tok3n" in line] == ["::add-mask::tok3n"]
        assert (temp / "out").read_text() == "page_url=https://o.github.io/r/\n" + "stuck=true\n" * (n == 6)
        assert sum(c == f"api -X POST repos/o/r/pages/deployments/{v1}/cancel" for c in calls) == cancels
        assert ((temp / "summary").exists() and "Run 7 waits" in (temp / "summary").read_text()) == (n == 1)
        assert f"Deployment {v1}: version {v1}" in out and "  ".join(calls).count(f"pages/deployments/{v1} --jq") >= len(statuses) - 1
    (temp / "statuses").write_text("succeed\n")                      # the retry, after the stuck try
    done = subprocess.run(["bash", "-e", "-c", retry["run"]], env={**env, "ARTIFACT": "43", "TRY": "2"}, capture_output=True,
                          text=True, timeout=120)
    assert done.returncode == 0 and f"Deployment {v2}: version {v2}" in done.stdout, done.stdout
    assert json.loads((temp / "body").read_text()) == {"artifact_id": 43, "pages_build_version": v2, "oidc_token": "tok3n"}
    (temp / "setup" / "versions.code").write_text("1\n")              # none handed over: the run's commit, and why
    (temp / "setup" / "versions.log").write_text("the version job did not run in attempt 2\n")
    (temp / "versions.txt").unlink()
    (temp / "statuses").write_text("succeed\n")
    done = subprocess.run(["bash", "-e", "-c", retry["run"]], env={**env, "ARTIFACT": "43"}, capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0 and f"::warning::no version commit handed over: {sha}, as deploy-pages\nthe version job" in done.stdout
    assert json.loads((temp / "body").read_text())["pages_build_version"] == sha
    (temp / "out").write_text("")                                     # under GITHUB_SHA: never stuck, never to a run at it
    (temp / "statuses").write_text("syncing_files\n")
    runs.write_text(json.dumps({"workflow_runs": [{"id": 7, "status": "queued", "display_title": "Card browser", "head_sha": sha}]}))
    done = subprocess.run(["bash", "-e", "-c", run.replace("-ge 120 ]", "-ge 0 ]").replace("-lt 360 ]", stuck["-lt 360 ]"])],
                          env={**env, "ARTIFACT": "43"}, capture_output=True, text=True, timeout=120)
    assert done.returncode == 1 and "stuck" not in (temp / "out").read_text() and "Run 7" not in done.stdout, done.stdout
    assert sum("actions/workflows/site.yml/runs" in c for c in (temp / "calls").read_text().splitlines()) == 1
