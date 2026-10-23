"""
The website build (TTSCardMaker site/build/build_site.py) on a small card repo made here, offline: the same site from the
bot's pickles and from the card sync's db/ (CARD_DATA_DIR, load_card_data: the shadow's comparison in small), the current
text dated from its start (since), a duplicate's history, dates by committer time, an archived card's moved_to, Pacific
contribution months. Loads
build_site.py from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without it or before load_card_data.
"""
import importlib.util
import json
import subprocess
import sys
import time
from collections import defaultdict
from os import environ
from os.path import expanduser
from pathlib import Path

import pytest

import commands.card_db as cdb
import commands.card_models as cm
from conftest import CLONE

BUILD_SITE = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE))) / "site" / "build" / "build_site.py"
DAY = 86400
REBASED = 20 * DAY                            # committer time after author time (a rebase: Patch 1.9's commits)
ALPHA = "Creatures/Other/2 Stars/Alpha"
MELON = "Items/Heal/2 Stars/Watermelon"
PALACE = "Field/4 Stars/White_Palace.psd"
BOT_DATA = ("stats.pkl", "old_stats.pkl", "psd_details.json")    # what db/ replaces


@pytest.fixture(scope="module")
def bs():
    if not BUILD_SITE.is_file():
        pytest.skip(f"no card repo build_site at {BUILD_SITE}")
    spec = importlib.util.spec_from_file_location("site_build_site", BUILD_SITE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "load_card_data"):
        pytest.skip(f"{BUILD_SITE} reads only the pickles: land TTSCardMaker work/site-history")
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

    stats = {"Alpha": _card("Alpha", f"{ALPHA}.psd", "New text.", t3, blob_sha="b3"),
             "Watermelon": _card("Watermelon", f"{MELON}.psd", "Heal 2.", t2, blob_sha="m2", card_type="item"),
             "Watermelon2": _card("Watermelon2", f"{MELON}.psd", "Heal 2.", t2, blob_sha="m2", card_type="item",
                                  duplicate_of="Watermelon", display_name="Inconspicuous Watermelon")}
    stats["Alpha"].version = {"from": float(t2), "to": None, "cause": "psd"}     # a carried start (no from_commit)
    palace = _card("White Palace", PALACE, "Field.", t1, card_type="field")
    palace.archived = {"at": float(t3), "path": PALACE, "commit": c3, "moved_to": "MDW/The Morgue/White_Palace.psd"}
    old = defaultdict(list, {"Alpha": [_card("Alpha", f"{ALPHA}.psd", "Old text.", t1)],
                             "Watermelon": [_card("Watermelon", f"{MELON}.psd", "Heal 1.", t1, card_type="item")],
                             "White Palace": [palace]})
    for card in list(stats.values()) + [v for vs in old.values() for v in vs]:
        card.parser_version = 2
    details = json.dumps({"version": 1, "cards": {}})
    pickles = tmp_path / "pickles"
    pickles.mkdir()
    cdb.dump_pickle(stats, str(pickles / "stats.pkl"))
    cdb.dump_pickle(old, str(pickles / "old_stats.pkl"))
    cdb.dump_pickle({}, str(pickles / "aliases.pkl"))
    (pickles / "psd_details.json").write_text(details)
    db = tmp_path / "db"
    cdb.write_db(str(db), stats, old, details, {"card_commit": c3})
    (db / "changelog").mkdir()
    event = {"seq": 1, "at": time.time(), "kind": "archived", "card": "White Palace", "title": "White Palace", "path": PALACE,
             "moved_to": "MDW/The Morgue/White_Palace.psd", "card_commit": c3}
    (db / "changelog" / "2026-10.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    return {"root": root, "pickles": pickles, "db": db, "t": (t1, t2, t3), "commits": (c1, c2, c3)}


def _build(bs, monkeypatch, repo, out, data_dir=""):
    """build_site.main() offline into out/; {name: data/<name> as JSON}."""
    monkeypatch.setattr(bs, "REPO", str(repo["root"]))
    monkeypatch.setattr(bs, "HISTORY_REPO", str(repo["root"]))
    for name, sub in (("DIST", "dist"), ("CACHE", ".cache"), ("OG_CACHE", ".og-cache")):
        monkeypatch.setattr(bs, name, str(out / sub))
    monkeypatch.setattr(bs, "CARD_DATA_DIR", data_dir)
    monkeypatch.setattr(bs, "MIN_CARDS", 1)
    monkeypatch.setattr(bs, "_bot_raw", "local://")

    def fetch(url, tries=3, accept=None):
        name = url[len("local://"):]
        if data_dir and name in BOT_DATA:               # with CARD_DATA_DIR only db/ can give them
            raise OSError(f"{name} fetched with CARD_DATA_DIR set")
        return (repo["pickles"] / name).read_bytes()
    monkeypatch.setattr(bs, "fetch", fetch)
    monkeypatch.setattr(sys, "argv", ["build_site.py"])
    bs.main()
    data = out / "dist" / "data"
    return {p.name: json.loads(p.read_text(encoding="utf-8")) for p in data.glob("*.json")}


def _by_name(cards: list) -> dict:
    return {c["name"]: c for c in cards}


def test_pickles_and_db_build_the_same_site(bs, repo, tmp_path, monkeypatch):
    """The shadow's check in small: every data file equal but the build time and the changelog's events (only db/
    has them)."""
    a = _build(bs, monkeypatch, repo, tmp_path / "a")
    b = _build(bs, monkeypatch, repo, tmp_path / "b", str(repo["db"]))
    assert set(a) == set(b)
    for name in a:
        if name == "cards.json":
            a[name].pop("generated"), b[name].pop("generated")
        if name == "changelog.json":
            assert b[name]["archived"] == a[name]["archived"]
            continue
        if name == "decks.json":
            a[name].pop("generated"), b[name].pop("generated")
        assert a[name] == b[name], name


def test_the_current_text_is_dated_from_its_start(bs, repo, tmp_path, monkeypatch):
    """Alpha's text started at c2 (the bot's carried start), though its PSD was last committed at c3: cards.json keeps
    ts (the last commit, "updated") and adds since; history.json holds the older text at c1."""
    t1, t2, t3 = repo["t"]
    out = _build(bs, monkeypatch, repo, tmp_path / "a")
    alpha = _by_name(out["cards.json"]["cards"])["Alpha"]
    assert (alpha["ts"], alpha["since"]) == (t3, t2)
    assert out["history.json"][alpha["id"]] == [{"ts": t1, "ability": "Old text."}]
    assert "since" not in _by_name(out["cards.json"]["cards"])["Watermelon"]      # its text started at its last commit


def test_dates_are_committer_times(bs, repo, tmp_path, monkeypatch):
    """Image versions are dated by committer time, as the bot dates versions, so the carried start (c2's committer
    time) is a known commit and Alpha's change is exact, not approximate."""
    out = _build(bs, monkeypatch, repo, tmp_path / "a")
    alpha = _by_name(out["cards.json"]["cards"])["Alpha"]
    assert sorted(ts for _, ts, _, _ in out["versions.json"]["commits"]) == list(repo["t"])
    rows = out["analytics.json"]["cards"][alpha["id"]]["changes"]
    assert [r["ts"] for r in rows] == [repo["t"][1]] and not any(r.get("approx") for r in rows)


def test_a_duplicate_shows_its_originals_history(bs, repo, tmp_path, monkeypatch):
    """Watermelon2 is Watermelon's PSD with other art: its page gets the original's History block (hv, history.json)."""
    out = _build(bs, monkeypatch, repo, tmp_path / "a")
    cards = _by_name(out["cards.json"]["cards"])
    melon, dup = cards["Watermelon"], cards["Inconspicuous Watermelon"]
    assert dup["dup"] == melon["id"] and dup["hv"] == melon["hv"] == 1
    assert out["history.json"][dup["id"]] == out["history.json"][melon["id"]] == [{"ts": repo["t"][0], "ability": "Heal 1."}]


def test_an_archived_card_keeps_where_its_psd_went(bs, repo, tmp_path, monkeypatch):
    out = _build(bs, monkeypatch, repo, tmp_path / "a")
    assert [(r["card"], r.get("moved_to")) for r in out["changelog.json"]["archived"]] == \
        [("White Palace", "MDW/The Morgue/White_Palace.psd")]


def test_an_archived_event_keeps_moved_to(bs, repo, tmp_path, monkeypatch):
    """The card sync's archived event reaches the Changelog tab with moved_to (changes.js: "moved to MDW/The Morgue")."""
    out = _build(bs, monkeypatch, repo, tmp_path / "b", str(repo["db"]))
    assert [(e["kind"], e["card"], e.get("moved_to")) for e in out["changelog.json"]["events"]] == \
        [("archived", "White Palace", "MDW/The Morgue/White_Palace.psd")]


def test_a_text_starts_with_the_versions_that_already_held_it(bs):
    """since is the text's start, not the data's: a stats change or a later edit (version.from after the PSD commit)
    keeps the earlier start of the versions just before with the same text; an older stretch of it does not count."""
    def v(ts, text, start=None):
        card = bs.CardInfo()
        card.__dict__.update(timestamp=float(ts), ability=text, version={"from": float(start)} if start else None)
        return card
    assert bs.text_since(v(300, "A."), [v(100, "A."), v(200, "A.")], "A.") == 100        # stats changes
    assert bs.text_since(v(400, "A."), [v(100, "A."), v(200, "B."), v(300, "A.")], "A.") == 300
    assert bs.text_since(v(300, "A.", start=200), [v(100, "B.")], "A.") == 200         # the bot's carried start
    assert bs.text_since(v(300, "A.", start=500), [v(100, "B."), v(250, "A.")], "A.") == 250   # an edit after the PSD
    assert bs.text_since(v(300, "A."), None, "A.") == 300 and bs.text_since(bs.CardInfo(), [], "A.") is None


def test_no_readable_stats_json_stops_the_build(bs, repo, tmp_path, monkeypatch):
    """CARD_DATA_DIR without a readable stats.json: no site, never one from the pickles instead."""
    stats = repo["db"] / "stats.json"
    for text in (None, "not json", '{"schema": 1, "cards": []}', '{"schema": 2, "cards": {}}'):
        stats.unlink(missing_ok=True)
        if text is not None:
            stats.write_text(text, encoding="utf-8")
        with pytest.raises(SystemExit, match=r"can't load the card data \("):
            _build(bs, monkeypatch, repo, tmp_path / "b", str(repo["db"]))


def test_contribution_months_are_pacific(bs):
    """A commit on the evening of Oct 31 in Pacific time (Nov 1 in UTC) counts in October, as the site dates it."""
    if bs.analytics.ZONE_MISSING:
        pytest.skip("no tz database")
    ts = 1793502000                                # 2026-11-01 03:00 UTC, 2026-10-31 20:00 PDT
    commits = [("a" * 40, ts, "Maker", "maker@example.com", "Alpha", [("M", "1" * 40, "2" * 40, f"{ALPHA}.psd")], [])]
    people = bs.contributions([], {}, [], {}, commits)["people"]
    assert [(p["key"], p["months"]) for p in people] == [("Maker", [["2026-10", 0, 0, 1]])]


def test_a_duplicate_with_its_own_text_has_its_own_history(bs, repo, tmp_path, monkeypatch):
    """A duplicate whose Card Edits.json entry sets its ability gets the PSD's history against that text: its own
    text is never listed as an older one, and the original keeps its History."""
    edits = repo["root"] / "site" / "Card Edits.json"
    edits.parent.mkdir()
    edits.write_text(json.dumps({"cards": {"Watermelon2": {"ability": "Heal 1."}}}), encoding="utf-8")
    out = _build(bs, monkeypatch, repo, tmp_path / "a")
    cards = _by_name(out["cards.json"]["cards"])
    melon, dup = cards["Watermelon"], cards["Inconspicuous Watermelon"]
    assert dup["ability"] == "Heal 1." and "hv" not in dup and dup["id"] not in out["history.json"]
    assert out["history.json"][melon["id"]] == [{"ts": repo["t"][0], "ability": "Heal 1."}]
