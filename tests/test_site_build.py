"""
The website build (TTSCardMaker site/build/build_site.py) on a small card repo made here, offline: the same site from the
bot's pickles and from the card sync's db/ (CARD_DATA_DIR, load_card_data: the shadow's comparison in small), the current
text dated from its start (since), a duplicate's history, dates by committer time, an archived card's moved_to, tag,
author and created, image revisions (Stats), Pacific contribution months, the person stats. Loads
build_site.py from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without it or before load_card_data.
"""
import importlib.util
import inspect
import json
import shutil
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
APP_JS = BUILD_SITE.parents[1] / "assets" / "app.js"
BALANCE_JS = BUILD_SITE.parents[1] / "assets" / "balance.js"
ROUND3 = (BUILD_SITE.parents[1] / "assets" / "insights.js").is_file()    # round 3's site (2026-10-24)
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
    edits.parent.mkdir(exist_ok=True)
    edits.write_text(json.dumps({"cards": {"Watermelon2": {"ability": "Heal 1."}}}), encoding="utf-8")
    out = _build(bs, monkeypatch, repo, tmp_path / "a")
    cards = _by_name(out["cards.json"]["cards"])
    melon, dup = cards["Watermelon"], cards["Inconspicuous Watermelon"]
    assert dup["ability"] == "Heal 1." and "hv" not in dup and dup["id"] not in out["history.json"]
    assert out["history.json"][melon["id"]] == [{"ts": repo["t"][0], "ability": "Heal 1."}]


def test_a_recompression_is_no_new_image_version(bs, tmp_path, monkeypatch):
    """The push guards' commit and the batch name the PNGs they only recompressed in Recompressed trailers (the same
    image): no image version for those; the guards' re-export in the same commit is one, and a person's commit is
    always one, trailer or not."""
    import commands.card_guards as cg
    import commands.png_recompress as batch
    from PIL import Image
    root = tmp_path / "cards"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    snack, bread = f"{MELON}.png", f"{ALPHA}.png"

    def save(path, shade, level=6):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (60, 84), (shade, 40, 90)).save(root / path, compress_level=level)

    def commit(message, author="Maker"):
        email = "autocroissant@gmail.com" if author == "AutoCroissant" else "maker@example.com"
        env = {**environ, "GIT_AUTHOR_NAME": author, "GIT_AUTHOR_EMAIL": email, "GIT_COMMITTER_NAME": author,
               "GIT_COMMITTER_EMAIL": email}
        subprocess.run(["git", "-C", str(root), "add", "-A"], env=env, check=True)
        subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", message], env=env, check=True)
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()

    save(snack, 10), save(bread, 10)
    c1 = commit("Cards")
    save(snack, 20)
    c2 = commit("Snack art")
    save(snack, 20, 0), save(bread, 30)                     # the guards: Snack.png recompressed, Bread.png re-exported
    c3 = commit(cg.message([{"login": "Maker", "commit": c2, "writes": [
        {"path": snack, "kind": "recompress", "line": "recompressed losslessly (9.0 MB -> 2.0 MB, the same pixels)"},
        {"path": bread, "kind": "png", "line": "re-exported by the bot's render of the PSD (12 px changed)"}]}], c2),
        "AutoCroissant")
    save(snack, 20, 9), save(bread, 30, 0)                  # the batch
    c4 = commit(batch.message([{"path": p, "outcome": "recompressed", "before": 9, "after": 2} for p in (snack, bread)]),
                "AutoCroissant")
    save(snack, 40)
    c5 = commit(f"Snack again\n\n{cg.RECOMPRESSED}: {snack}")
    monkeypatch.setattr(bs, "HISTORY_REPO", str(root))
    history = bs.git_history()
    found = bs.image_versions([snack, bread], history)
    assert {path: [commit[0] for commit, _ in rows] for path, rows in found.items()} == {snack: [c5, c2, c1], bread: [c3, c1]}
    assert {sha: entries for sha, _, _, _, _, entries, _ in history}[c4] == []     # the batch changed no image


def test_a_types_edit_that_only_reorders_waits_for_no_bake(bs):
    """'PSD pending' on types as the bake's psd_edit.same_types sees it: any order and spelling (TTSCardMaker
    work/types-any-order; the website's toggles list them sorted)."""
    def pending(types):
        entry = {"key": "Mini_Doomer", "types": ["wind", "defensive_minion", "light"]}
        bs.apply_edits(entry, {bs.card_key("Mini Doomer"): ({"types": types}, {})})
        return entry.get("pending")
    assert pending(["defensive_minion", "light", "wind"]) is None
    assert pending(["Light", "Defensive Minion", "wind"]) is None
    assert pending(["light", "wind"]) == pending(["light", "wind", "wind"]) == ["types"]


def test_the_editor_writes_no_types_edit_that_only_reorders(tmp_path):
    """edit.js mergeField (readForm the same): its toggles list the types sorted, so a card's own types in another
    order are no edit, and back to the PSD's set unsets an edit."""
    if not EDIT_JS.is_file() or shutil.which("node") is None:
        pytest.skip("edit.js or node missing")
    script = tmp_path / "merge.mjs"
    script.write_text(f"const {{ mergeField }} = await import({json.dumps(EDIT_JS.as_uri())});\n"
                      "const psd = ['wind', 'defensive_minion', 'light'], sorted = [...psd].sort(), op = { set: {}, unset: [] };\n"
                      "const cases = [[{ pub: psd, psd, edited: false }, sorted], [{ pub: psd, psd, edited: false }, ['light', 'wind']],\n"
                      "  [{ pub: ['fire'], psd, edited: true }, sorted]];\n"
                      "process.stdout.write(JSON.stringify(cases.map(([base, v]) => mergeField(op, base, 'types', v))));\n",
                      encoding="utf-8")
    done = subprocess.run(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    assert [(o["set"], o["unset"]) for o in json.loads(done.stdout)] == [({}, []), ({"types": ["light", "wind"]}, []), ({}, ["types"])]


def test_the_dist_pager_keeps_focus_at_its_ends(tmp_path):
    """app.js distPage: › to the last page disables the focused button; focus moves to ‹ (a browser would drop it to
    <body>), and with neither left, to the panel."""
    if not APP_JS.is_file() or shutil.which("node") is None:
        pytest.skip(f"no app.js at {APP_JS}, or no node")
    text = APP_JS.read_text(encoding="utf-8")
    if "function distPage(" not in text:
        pytest.skip("no paged distributions")
    body = text[text.index("function distPage("):]
    body = body[:body.index("\n}\n") + 3]
    script = tmp_path / "pager.mjs"
    script.write_text("""
const document = { body: { tag: "body" } };
document.activeElement = document.body;
const el = (o = {}) => Object.defineProperties({ dataset: {}, focus() { document.activeElement = this; } }, Object.getOwnPropertyDescriptors(o));
const button = step => { let off = false; return el({ dataset: { step: String(step) }, label: step > 0 ? "next" : "prev",
  get disabled() { return off; }, set disabled(v) { off = v; if (v && document.activeElement === this) document.activeElement = document.body; } }); };
const btns = [button(-1), button(1)], panel = el({ label: "panel", dataset: { pages: "0,1", cur: "0", stat: "atk" } });
btns[0].disabled = true;
const $ = () => ({}), $$ = sel => sel === ".dist-pager button" ? btns : [], S = { distPage: {} };
%s
const seen = [];
btns[1].focus(); distPage(panel, 1); seen.push([String(panel.dataset.cur), document.activeElement.label]);
distPage(panel, -1); seen.push([String(panel.dataset.cur), document.activeElement.label]);
panel.dataset.pages = "0"; panel.dataset.cur = "0"; btns[1].disabled = false; btns[1].focus();
distPage(panel, 0); seen.push([String(panel.dataset.cur), document.activeElement.label]);
console.log(JSON.stringify(seen));
""" % body, encoding="utf-8")
    done = subprocess.run(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [["1", "prev"], ["0", "next"], ["0", "panel"]]


def test_the_card_page_has_no_share_full_image_or_psd_button():
    """Gone at the owner's ask (2026-10-23): the card page's Share (and its s key), Full image and PSD on GitHub."""
    if not APP_JS.is_file():
        pytest.skip(f"no app.js at {APP_JS}")
    text = APP_JS.read_text(encoding="utf-8")
    assert [s for s in ('data-action="share"', ">Full image<", ">PSD on GitHub<", 'e.key === "s"') if s in text] == []


def test_ratings_reach_the_cards_and_the_archived_rows(bs, repo, tmp_path, monkeypatch):
    """site/Card Ratings.json: Alpha's entries on its card (matched in any case), White Palace's on its archived row."""
    if not hasattr(bs, "parse_ratings"):
        pytest.skip(f"{BUILD_SITE} has no ratings")
    out = _build(bs, monkeypatch, repo, tmp_path / "a")
    cards = _by_name(out["cards.json"]["cards"])
    assert cards["Alpha"]["ratings"] == [{"date": "2023-04-28", "scale": "power", "value": 5.5, "by": "group", "source": "TTS session"}]
    assert "ratings" not in cards["Watermelon"] and out["cards.json"]["files"]["ratings"] == "site/Card Ratings.json"
    assert [(r["card"], [e["value"] for e in r.get("ratings", [])]) for r in out["changelog.json"]["archived"]] == [("White Palace", [7.1])]


def test_archived_rows_carry_tag_author_created(bs, repo, tmp_path, monkeypatch):
    """White Palace's changelog row: tag retired, author Maker, created the Pacific day of its oldest version; its
    analytics.json copy (Stats) has only the slim fields, its series spelled as the live cards' ('kirby' -> 'Kirby')."""
    if not hasattr(bs, "ARCHIVED_SLIM"):
        pytest.skip(f"{BUILD_SITE} has no archive tags")
    out = _build(bs, monkeypatch, repo, tmp_path / "a")
    t1 = repo["t"][0]
    day = datetime.fromtimestamp(t1, bs.analytics.ZONE).strftime("%Y-%m-%d")
    row = out["changelog.json"]["archived"][0]
    assert (row["tag"], row["author"], row["created"]) == ("retired", "Maker", day)
    assert out["analytics.json"]["archived"] == [{"card": "White Palace", "title": "White Palace", "tag": "retired", "card_type": "field",
                                                   "stars": 2, "series": "Kirby", "at": repo["t"][2], "created": day, "author": "Maker"}]


def test_archived_author_and_created_precedence(bs):
    """author: Card Authors.txt's credit > the tag's (backfill) > the last version's, shown through [people]; created:
    the last version's created date, else the Pacific day of its oldest version (its version record's start)."""
    if not hasattr(bs, "ARCHIVED_SLIM"):
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


def test_archived_series_take_the_live_spelling(bs):
    """series: the renamed card's, else the old one in a live series' case or through SERIES_ALIASES, else as it was."""
    if not hasattr(bs, "SERIES_ALIASES"):
        pytest.skip(f"{BUILD_SITE} keeps the old series")

    def old(series, **tag):
        card = bs.CardInfo()
        card.__dict__.update(timestamp=1.7e9, card_type="creature", stars=2, series=series, archived={"at": 1_700_000_100, **tag})
        return [card]
    olds = {"A": old("kirby"), "B": old("rotmg"), "C": old("pandas"), "D": old("isaac", renamed_to="New D"), "E": old(None)}
    live = {"Kirby", "Realm of the Mad God", "Mario"}
    rows = bs.archived_cards(olds, lambda name: "new-d" if name == "New D" else None, (), series_of=lambda name: "Mario" if name == "New D" else None, live_series=live)
    assert {r["card"]: r.get("series") for r in rows} == {"A": "Kirby", "B": "Realm of the Mad God", "C": "pandas", "D": "Mario", "E": None}


def test_an_archived_card_titled_like_a_live_one_keeps_its_author(bs):
    """'Pizza (item)' (title Pizza) isn't credited with the live creature Pizza's Card Authors.txt line; a title no live
    card has still finds its line."""
    if not hasattr(bs, "ARCHIVED_SLIM"):
        pytest.skip(f"{BUILD_SITE} has no archive tags")
    card = bs.CardInfo()
    card.__dict__.update(timestamp=1.7e9, card_type="item", stars=2, author="Last", display_name="Pizza",
                         archived={"at": 1_700_000_100, "author": "Chestnut", "renamed_to": "Pepperoni Pizza"})
    old, credited = {"Pizza (item)": [card]}, {"pizza": "Croissant", "pepperoni pizza": "Chestnut"}
    live = lambda name: {"Pizza": "pizza", "Pepperoni Pizza": "pepperoni-pizza"}.get(name)
    assert bs.archived_cards(old, live, (), credited=credited)[0]["author"] == "Chestnut"
    assert bs.archived_cards(old, lambda name: None, (), credited=credited)[0]["author"] == "Croissant"


def test_a_renamed_card_keeps_the_credit_of_the_card_it_became(bs):
    """Meta Mato -> Maxim Tomato: the new name's Card Authors.txt line, else the old name's, else the live card's
    author, before the tag's and the last version's."""
    if "author_of" not in inspect.signature(bs.archived_cards).parameters:
        pytest.skip(f"{BUILD_SITE} credits renamed cards by their old name")
    card = bs.CardInfo()
    card.__dict__.update(timestamp=1.7e9, card_type="item", stars=2, author="MichaelJSr",
                         archived={"at": 1_700_000_100, "renamed_to": "Maxim Tomato", "backfill": True})
    old = {"Meta Mato": [card]}
    live = lambda name: "maxim-tomato" if name == "Maxim Tomato" else None
    rows = lambda credited, *of: bs.archived_cards(old, live, (), credited=credited, **({"author_of": of[0]} if of else {}))[0]["author"]
    assert rows({"maxim tomato": "Chestnut"}) == "Chestnut"
    assert rows({"maxim tomato": "Chestnut", "meta mato": "Croissant"}) == "Chestnut"
    assert rows({}, lambda name: "Croissant") == "Croissant"
    assert rows({"meta mato": "Listed"}, lambda name: "Croissant") == "Listed"
    assert rows({}) == "MichaelJSr"
    assert bs.archived_cards(old, lambda name: None, (), credited={"maxim tomato": "Chestnut"})[0]["author"] == "MichaelJSr"


def test_revisions_leave_out_uploads_and_mass_passes(bs, repo, tmp_path, monkeypatch):
    """analytics.json revisions: Alpha's two image changes after its upload by Maker, nothing for Watermelon. The helper:
    a commit in more than MASS_PASS cards' versions is left out; a bot commit credits its Edit-by logins (not a discord:
    name), one without trailers nobody; every id of the PNG but skip's gets the counts; no history: None."""
    if not hasattr(bs, "card_revisions"):
        pytest.skip(f"{BUILD_SITE} has no revisions")
    out = _build(bs, monkeypatch, repo, tmp_path / "a")
    cards = _by_name(out["cards.json"]["cards"])
    assert out["analytics.json"]["revisions"] == {"available": True, "mass": bs.MASS_PASS, "cards": {cards["Alpha"]["id"]: {"Maker": 2}}}

    row = lambda sha, name, editors=(): (sha, 0, name, f"{name.lower()}@example.com", "s", [], list(editors))
    commits = [row("mass", "Maker"), row("bot", bs.BOT_AUTHOR, ["alice", "discord:carl"]), row("bare", bs.BOT_AUTHOR), row("bob", "Bob"),
               row("up", "Maker")]
    v = lambda sha: ((sha, 0, "x", "s"), "p")
    found = {f"m{i}.png": [v("mass"), v("up")] for i in range(bs.MASS_PASS + 1)}
    found["a.png"] = [v("bot"), v("bare"), v("bob"), v("mass"), v("up")]
    by_png = {"a.png": [{"id": "a"}, {"id": "a2"}, {"id": "page"}], **{p: [{"id": p}] for p in found if p != "a.png"}}
    got = bs.card_revisions(found, by_png, commits, {"alice": "Alice"}, {}, skip={"page"})
    assert got == {"a": {"Alice": 1, "Bob": 1}, "a2": {"Alice": 1, "Bob": 1}}
    assert bs.card_revisions(None, by_png, None, {}, {}) is None


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


def test_contributions_get_edits_by_field_and_ratings(bs, repo, tmp_path, monkeypatch):
    """main() feeds contributions() the ratings and every edit batch of all time: Maker's row counts an old Card
    Edits.json note, one Card Authors.txt line (not the lines of the commit that made the file) and a website rating,
    and has others' ratings of Alpha (P4)."""
    if not hasattr(bs, "person_extras"):
        pytest.skip(f"{BUILD_SITE} has no person stats")
    root, start = repo["root"], repo["t"][0] - REBASED
    (root / "site" / "Card Edits.json").write_text(json.dumps({"cards": {"Alpha": {"note": "Checked.", "_by": "Maker",
                                                                                     "_at": "2025-01-01T00:00:00Z"}}}))
    _commit(root, "Website edit: Alpha", start - 400 * DAY)                    # older than CHANGELOG_DAYS
    authors = root / "site" / "Card Authors.txt"
    authors.write_text("[cards]\nAlpha = Maker\n")
    _commit(root, "Card authors file", start + 4 * DAY)
    authors.write_text("[cards]\nAlpha = Maker\nWatermelon = Maker\n")
    _commit(root, "Credit Watermelon", start + 5 * DAY)
    ratings = root / "site" / "Card Ratings.json"
    data = json.loads(ratings.read_text())
    data["cards"]["Watermelon"] = [{"date": "2026-10-20", "scale": "favorability", "value": 8, "by": "Maker", "source": "website"}]
    ratings.write_text(json.dumps(data))
    _commit(root, "Website rating", start + 6 * DAY)
    out = _build(bs, monkeypatch, repo, tmp_path / "a")
    maker = {r["key"]: r for r in out["contributions.json"]["people"]}["Maker"]
    assert dict(maker["editFields"]) == {"note": 1, "author": 1, "rating": 1}
    assert maker["ratings"]["powerN"] == 1 and maker["ratings"]["power"] == 5.5


def test_the_cards_page_has_a_banned_flag():
    """The person page's banned link filters the Cards tab with flags=banned."""
    if not APP_JS.is_file():
        pytest.skip(f"no app.js at {APP_JS}")
    if "styleOf" not in (APP_JS.parent / "profile.js").read_text(encoding="utf-8"):
        pytest.skip("no round 3 person pages")
    text = APP_JS.read_text(encoding="utf-8")
    assert '["banned", "Banned (constructed)"]' in text and "banned: c => !!c._banned" in text


def test_long_history_notes_are_gone():
    """Dropped at the owner's ask (2026-10-24): the notes under Image versions and Recorded changes."""
    if not APP_JS.is_file() or not BALANCE_JS.is_file():
        pytest.skip(f"no site assets beside {APP_JS}")
    if not ROUND3:
        pytest.skip("no round 3 site")
    assert "Each commit that changed the image" not in APP_JS.read_text(encoding="utf-8")
    assert "From the bot's card history" not in BALANCE_JS.read_text(encoding="utf-8")


def test_no_psd_details_flag():
    """The Cards tab's "PSD details" chip and the search term has:details are gone (owner, 2026-10-24)."""
    if not APP_JS.is_file():
        pytest.skip(f"no app.js at {APP_JS}")
    if not ROUND3:
        pytest.skip("no round 3 site")
    text = APP_JS.read_text(encoding="utf-8")
    assert '"PSD details"]' not in text and "details: c => !!c.details" not in text
    assert "has:details" not in (APP_JS.parent / "search.js").read_text(encoding="utf-8")


def test_card_buttons_sit_under_the_image():
    """Add to deck, Compare and Edit under the card image (owner, 2026-10-24); the info column stays .detail's second child."""
    if not APP_JS.is_file():
        pytest.skip(f"no app.js at {APP_JS}")
    if not ROUND3:
        pytest.skip("no round 3 site")
    text = APP_JS.read_text(encoding="utf-8")
    body = text[text.index("function openCard("):]
    body = body[:body.index("\n}\n")]
    at = [body.index(s) for s in ('class="image-wrap"', 'class="actions card-actions"', 'data-action="deck-toggle"',
                                  'data-action="compare-add"', 'data-action="edit-card"', 'data-nav="prev"', "<h2>${esc(c.name)}</h2>")]
    assert at == sorted(at)
    assert body.count('data-action="deck-toggle"') == 1
    wrap = body[body.index('<div class="detail">'):body.index("<h2>")]
    assert wrap.rstrip().endswith("</div></div>\n      <div>")      # image column closes, then the info column


def _stamps(dist: Path) -> dict:
    """{asset or data URL in index.html: its ?v= stamp} and the build meta"""
    import re
    page = (dist / "index.html").read_text(encoding="utf-8")
    out = dict(re.findall(r'((?:assets/[\w-]+\.(?:js|css))|data/cards\.json)\?v=([0-9a-f]+)"', page))
    out["build"] = re.search(r'<meta name="build" content="([0-9a-f]+)">', page).group(1)
    return out


def test_stamps_follow_the_content_not_the_clock(bs, repo, tmp_path, monkeypatch):
    """Two builds of the same inputs give the same ?v= stamps, served assets and build.txt (only cards.json's and
    decks.json's generated and changelog.json's since differ); new data changes the data's stamp and build.txt, never an
    asset's."""
    if not hasattr(bs, "asset_stamps"):
        pytest.skip(f"{BUILD_SITE} stamps every URL with one build version")
    _build(bs, monkeypatch, repo, tmp_path / "a")
    _build(bs, monkeypatch, repo, tmp_path / "b")
    a, b = tmp_path / "a" / "dist", tmp_path / "b" / "dist"
    assert _stamps(a) == _stamps(b)
    digest = hasattr(bs, "BUILD_TXT")             # build.txt, the site's digest for site.yml
    assert not digest or (a / "build.txt").read_text() == (b / "build.txt").read_text()
    for path in sorted((a / "assets").iterdir()):
        assert path.read_bytes() == (b / "assets" / path.name).read_bytes(), path.name
    assert _stamps(a)["data/cards.json"] == _stamps(a)["build"]

    stats = cdb.load_pickle(str(repo["pickles"] / "stats.pkl"))
    stats["Alpha"].ability = "Newer text."
    cdb.dump_pickle(stats, str(repo["pickles"] / "stats.pkl"))
    _build(bs, monkeypatch, repo, tmp_path / "c")
    c = tmp_path / "c" / "dist"
    before, after = _stamps(a), _stamps(c)
    assert after["build"] != before["build"] and after["data/cards.json"] == after["build"]
    assert {k: v for k, v in after.items() if k.startswith("assets/")} == {k: v for k, v in before.items() if k.startswith("assets/")}
    assert not digest or (c / "build.txt").read_text() != (a / "build.txt").read_text()


def test_an_asset_stamp_covers_its_imports(bs):
    """A module's stamp changes with its own text and its imports' (their stamps are in it), not with a module it
    doesn't import; the served text names each import's stamp; an import cycle gives every file one stamp."""
    if not hasattr(bs, "asset_stamps"):
        pytest.skip(f"{BUILD_SITE} stamps every URL with one build version")
    texts = {"app.js": 'import { a } from "./a.js";\nimport { b } from "./b.js";\n', "a.js": 'import { t } from "./t.js";\nexport const a = 1;\n',
             "b.js": "export const b = 2;\n", "t.js": "export const t = 3;\n", "style.css": "body { color: red; }\n"}
    one = bs.asset_stamps(texts)
    assert one["app.js"][1] == (f'import {{ a }} from "./a.js?v={one["a.js"][0]}";\nimport {{ b }} from "./b.js?v={one["b.js"][0]}";\n')
    assert one["a.js"][1] == f'import {{ t }} from "./t.js?v={one["t.js"][0]}";\nexport const a = 1;\n'
    two = bs.asset_stamps({**texts, "t.js": "export const t = 4;\n"})
    assert {n for n in texts if one[n][0] != two[n][0]} == {"t.js", "a.js", "app.js"}
    three = bs.asset_stamps({**texts, "b.js": "export const b = 5;\n"})
    assert {n for n in texts if one[n][0] != three[n][0]} == {"b.js", "app.js"}
    assert bs.asset_stamps(texts) == one
    cycle = bs.asset_stamps({**texts, "t.js": 'import { a } from "./a.js";\nexport const t = 3;\n'})
    assert len({stamp for stamp, _ in cycle.values()}) == 1
    assert 'from "./a.js?v=' + cycle["a.js"][0] + '"' in cycle["t.js"][1]
