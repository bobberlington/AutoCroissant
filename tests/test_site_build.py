"""
The website build (TTSCardMaker site/build/build_site.py) on a small card repo made here, offline, from the card sync's
db/ (CARD_DATA_DIR, load_card_data; no CARD_DATA_DIR, or no readable stats.json, builds nothing): the current text dated
from its start (since), a duplicate's history, dates by committer time, an archived card's moved_to, tag,
author and created, image revisions (the person pages), Pacific contribution months, the person stats. Loads
build_site.py from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without it or before load_card_data.
"""
import importlib.util
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from os import environ
from os.path import expanduser
from pathlib import Path

import pytest

import commands.card_db as cdb
import commands.card_models as cm
from conftest import CLONE

BUILD_SITE = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE))) / "site" / "build" / "build_site.py"
EDIT_JS = BUILD_SITE.parents[1] / "assets" / "edit.js"
DAY = 86400
REBASED = 20 * DAY                            # committer time after author time (a rebase: Patch 1.9's commits)
ALPHA = "Creatures/Other/2 Stars/Alpha"
MELON = "Items/Heal/2 Stars/Watermelon"
PALACE = "Field/4 Stars/White_Palace.psd"


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
    (root / "site").mkdir()                       # ratings: a commit of no card file (no image version, no edit batch)
    (root / "site" / "Card Ratings.json").write_text(json.dumps({"cards": {
        "alpha": [{"date": "2023-04-28", "scale": "power", "value": 5.5, "by": "group", "source": "TTS session"}],
        "White Palace": [{"date": "2025-04-28", "scale": "favorability", "value": 7.1, "by": "group", "source": "TTS session"}]}}))
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


def test_archived_author_and_created_precedence(bs):
    """author: Card Authors.txt's credit > the tag's (backfill) > the last version's, shown through [people]; created:
    the last version's created date, else the Pacific day of its oldest version (its version record's start)."""
    if not hasattr(bs, "archive_tag"):
        pytest.skip(f"{BUILD_SITE} has no archive tags")

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


def test_contributions_json_has_person_stats(bs, repo, tmp_path, monkeypatch):
    """Maker's row: Alpha's PNG changed in c2 and c3 (self 2), White Palace archived (Retired, without path or commit),
    Alpha's strength."""
    if not hasattr(bs, "person_extras"):
        pytest.skip(f"{BUILD_SITE} has no person stats")
    out = _build(bs, monkeypatch, repo, tmp_path / "a")
    maker = {r["key"]: r for r in out["contributions.json"]["people"]}["Maker"]
    assert (maker["revisions"]["self"], maker["revisions"]["others"]) == (2, 0)
    assert [(r["card"], r["tag"]) for r in maker["archived"]] == [("White Palace", "retired")]
    assert not {"path", "commit", "ratings"} & set(maker["archived"][0])
    assert maker["power"]["n"] >= 1 and "z" in out["contributions.json"]["site"]

