"""
The website build (TTSCardMaker site/build/build_site.py) on a small card repo made here, offline, from the card sync's
db/ (CARD_DATA_DIR, load_card_data; no CARD_DATA_DIR, or no readable stats.json, builds nothing): the current text dated
from its start (since), a duplicate's history, dates by committer time, an archived card's moved_to, tag,
author and created, image revisions (the person pages), Pacific contribution months, the person stats, index.html's
pre-written first screen (checked against grid.js in node), the Font filter's data (cardfonts.json), a profile's
subseries. Loads
build_site.py from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without it or before load_card_data.
"""
import copy
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from html import unescape as html_unescape
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


def test_a_renamed_card_goes_on_and_its_old_page_redirects(bs, repo, tmp_path, monkeypatch):
    """The bake renames and redraws Alpha as New Alpha in one commit (its Renamed trailers): the card's image versions
    and text history go on as one card, and c/alpha/ sends its links on to c/new-alpha/; a profile's pick, a published
    hand and the app's old id (cards.json "moved") follow it."""
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
    assert out["history.json"][alpha["id"]] == [{"ts": repo["t"][0], "ability": "Old text."}]
    assert len(out["versions.json"]["cards"][alpha["id"]]) == 4                      # c1, c2, c3 and the rename
    assert out["cards.json"]["moved"] == {"alpha": alpha["id"]} == {"alpha": "new-alpha"}
    assert out["users.json"]["users"][0]["favourite"] == out["decks.json"]["decks"][0]["cards"][0] == "new-alpha"


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


def test_index_carries_the_first_screen(bs, repo, tmp_path, monkeypatch):
    """index.html's pre-written grid: the cards in Name order as links to their pages, checked against grid.js in node;
    a difference fails the build."""
    if not shutil.which("node"):
        pytest.skip("no node")
    out = _build(bs, monkeypatch, repo, tmp_path / "a")
    page = (tmp_path / "a" / "dist" / "index.html").read_text(encoding="utf-8")
    ids = [c["id"] for c in sorted(out["cards.json"]["cards"], key=lambda c: c["name"].lower())]
    assert re.findall(r'<a class="tile" href="c/([^/"]+)/"', page) == ids
    assert page.count('fetchpriority="high"') == min(len(ids), bs.FIRST_EAGER)
    monkeypatch.setattr(bs, "tile_html", lambda c, eager, icons: f'<a class="tile" href="c/{c["id"]}/">')
    with pytest.raises(SystemExit, match="differs from grid.js"):
        _build(bs, monkeypatch, repo, tmp_path / "b")


def test_editor_stats_and_people_load_on_first_use(bs, repo, tmp_path, monkeypatch):
    """The editor (editing.js), Stats (charts.js), People (profile.js), the font box (tokens.js) and the rulebook (rulebook.js, its
    editor rulebook-edit.js) are imported by app.js with import(): without
    esbuild as stamped modules no page preloads; with it as chunks of their own, each with its stamped source map, none of
    their code in what a first visit loads (app.js and the chunks it imports), and the meta chunks list naming
    each one's files (app.js early())."""
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


def test_website_only_fields(bs, tmp_path, monkeypatch):
    """`_site` (owner 10-25): applied, never "PSD pending", the card's siteOnly (its "Website only" tag); edit.js stores it
    from an op's ticks as parse_edits and CardEdits.site_only clean it (a field without a value goes); unticked: pending."""
    from commands.card_edits import CardEdits
    if shutil.which("node") is None or not EDIT_JS.is_file():
        pytest.skip("node or edit.js missing")
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
    edited, so the Parse problems filter keeps a card with a stat still unread (B3, 2026-10-26)."""
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
    if shutil.which("node") is None or not text_js.is_file():
        pytest.skip("no node or text.js")
    probe = tmp_path / "probe.mjs"
    probe.write_text(f"const t = await import({json.dumps(text_js.as_uri())});\n"
                     "const r = t.makeRenderer({ cards: [{ id: 'mk', name: 'Mirror Knight', key: 'Mirror Knight', type: 'creature' }], icons: {} });\n"
                     "process.stdout.write(JSON.stringify([...r.mentions('(OD) Summon another Mirror Knight.')]));\n", encoding="utf-8")
    done = subprocess.run(["node", str(probe)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == ["mk"]


def test_tts_details_for_the_save(bs):
    """data/tts.json (owner 10-30, the TTS hover strip and details): types' meanings from the chapters (an icon heading's
    section to the page break, a "[token]:" line, a line ending in tokens; secret text, "(See: ...)" and non-icons left
    out), tags as text.js reads them, the statuses an ability names, cards by PNG path, hidden cards left out."""
    chapters = [{"id": "creature-types", "num": "20", "title": "Creature Types", "source":
                 "# Creature Types\n## FIRE {icon=fire}\nImmune to [[Burning]] (See: [“X”](#/rules/x))\n## WIND {icon=wind}\nSwap.\n"},
                {"id": "item-types", "num": "14", "title": "Item Types", "source":
                 "## [support] SUPPORT\nGeneral use. [joke]{.secret}\n=== page ===\nNot this.\n"},
                {"id": "minions", "num": "15", "title": "Minions", "source": "[offensive_minion]:Same field as the [fire] summoner.\n"},
                {"id": "status-effects", "num": "19", "title": "Status Effects", "source": "One instance only. [debuff] [nothere]\n"},
                {"id": "timing-tags", "num": "23", "title": "Timing Tags", "source": "| Tag | Meaning |\n|---|---|\n| (XCD) | Wait X turn(s) |\n"},
                {"id": "damage-types", "num": "21", "title": "Damage Types", "source": "## PIERCING\nIgnores DEF.\n"},
                {"id": "negation", "num": "22", "title": "Negation", "source": "## NEGATION\nTaken out of play.\n"}]
    icons = {k: {} for k in ("fire", "wind", "support", "offensiveminion", "debuff")}
    cards = [{"name": "Hot Guy", "type": "creature", "types": ["fire", "offensive_minion"], "png": "Creatures/Hot_Guy.png", "stars": 3,
              "hp": 5, "def": 1, "atk": 2, "spd": 3, "ability": "(1cd) Inflict 2T burning. (3CD) [fire] again. (OD) Burning! Starts with 2 Dark Tentacles and a Secret. Deals piercing damage; negates.", "author": "Bob"},
             {"name": "Burning", "type": "debuff", "types": ["debuff"], "png": "Aux/Burning.png", "ability": "3 DMG, can't affect [fire]"},
             {"name": "Secret", "type": "item", "png": "Items/Secret.png", "hidden": True, "ability": "Burning"},
             {"name": "Dark Tentacle", "type": "minion", "png": "Aux/Dark_Tentacle.png", "ability": ""},
             {"name": "Gate", "type": "field", "png": "Field/Gate.png", "ability": "(3CD) summon a dark tentacle, then stay."},
             {"name": "Stay", "type": "creature", "png": "Creatures/Stay.png", "ability": ""},
             {"name": "Mirror Knight", "type": "creature", "png": "Creatures/Mirror_Knight.png", "ability": "(OD) Summon another Mirror Knight."}]
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
                                 "Creatures/Mirror_Knight.png"}
    assert got["cards"]["Creatures/Mirror_Knight.png"]["m"] == ["Creatures/Mirror_Knight.png"]   # a card naming itself spawns itself (owner 10-30)
    assert got["cards"]["Field/Gate.png"]["m"] == ["Aux/Dark_Tentacle.png"]   # as text.js links: a lowercase minion after its verb
    assert got["cards"]["Creatures/Hot_Guy.png"] == {
        "n": "Hot Guy", "k": "Creature", "st": 3, "s": "HP 5 DEF 1 ATK 2 SPD 3",
        "a": "(1cd) Inflict 2T burning. (3CD) fire again. (OD) Burning! Starts with 2 Dark Tentacles and a Secret. Deals piercing damage; negates.",
        "t": ["fire", "offensiveminion"], "g": ["1CD", "3CD", "OD"], "x": ["Burning"], "m": ["Aux/Dark_Tentacle.png"], "r": ["damage-types", "negation"], "by": "Bob"}
    assert "x" not in got["cards"]["Aux/Burning.png"] and got["chapters"]["minions"] == {"num": "15", "title": "Minions"}
