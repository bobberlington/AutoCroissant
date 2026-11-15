"""
The website build (TTSCardMaker site/build/build_site.py): one smoke test per piece (owner 2026-11-12) and the checks of
the bot against the site. The build on a small card repo made here, offline, from the card sync's db/ (CARD_DATA_DIR,
load_card_data): the current text dated from its start; the bundle's lazy chunks; website-only fields (edit.js and
CardEdits); the parser notes an edit answers; a real card's inks (psd_details); R14's names (card_standards); web-only
rulebook chapters (rulebook.js). Loads build_site.py from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone.
"""
import importlib.util
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
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
    - Watermelon with its picture 2 Watermelon2 (its own PNG, no history of its own), one earlier text;
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

    stats = {"Alpha": _card("Alpha", f"{ALPHA}.psd", "New text.", t3, blob_sha="b3", series="Kirby"),
             "Watermelon": _card("Watermelon", f"{MELON}.psd", "Heal 2.", t2, blob_sha="m2", card_type="item", variant=1),
             "Watermelon2": _card("Watermelon2", f"{MELON}.psd", "Heal 2.", t2, blob_sha="m2", card_type="item",
                                  duplicate_of="Watermelon", variant=2, variant_of="Watermelon")}
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
    for name, sub in (("DIST", "dist"), ("CACHE", ".cache"), ("OG_CACHE", ".og-cache")):
        monkeypatch.setattr(bs, name, str(out / sub))
    monkeypatch.setattr(bs, "VCACHE", str(out / ".vcache"), raising=False)   # TTSCardMaker since 11-14
    (out / ".cache").mkdir(parents=True, exist_ok=True)
    if not (out / ".cache" / "cover").exists():
        (out / ".cache" / "cover").symlink_to(bs.covers, target_is_directory=True)
    monkeypatch.setattr(bs, "CARD_DATA_DIR", str(repo["db"]) if data_dir is None else data_dir)
    monkeypatch.setattr(bs, "MIN_CARDS", 1)

    def fetch(url, tries=3):
        raise OSError(f"offline: {url}")
    monkeypatch.setattr(bs, "fetch", fetch)
    monkeypatch.setattr(sys, "argv", ["build_site.py"])
    bs.main()
    data = out / "dist" / "data"
    return {p.name: json.loads(p.read_text(encoding="utf-8")) for p in data.glob("*.json")}


def _by_name(data: dict) -> dict:
    """cards.json's cards by name (rows or columns)."""
    return {c["name"]: c for c in cm.site_cards(data)}


def test_cards_json_columns(bs):
    """cards.json's cards as columns read back as the same rows, keys in order (app.js rowsOf, site_cards); rows when no
    field order fits every card or a card holds a None."""
    rows = [{"id": "a", "name": "A", "hp": 1}, {"id": "b", "dup": "a", "name": "B"}, {"id": "c", "name": "C", "hp": 0}]
    got = cm.site_cards({"cards": bs.columns(rows)})
    assert got == rows and [list(c) for c in got] == [list(c) for c in rows]
    for odd in ([{"a": 1, "b": 2}, {"b": 1, "a": 2}], [{"a": None}]):
        assert bs.columns(odd) is odd


def test_the_current_text_is_dated_from_its_start(bs, repo, tmp_path, monkeypatch):
    """Alpha's text started at c2 (the bot's carried start), though its PSD was last committed at c3: cards.json keeps
    ts (the last commit, "updated") and adds since; history.json holds the older text at c1. A variant picture's goes
    back through its picture 1's."""
    t1, t2, t3 = repo["t"]
    out = _build(bs, monkeypatch, repo, tmp_path / "a")
    alpha = _by_name(out["cards.json"])["Alpha"]
    assert (alpha["ts"], alpha["since"]) == (t3, t2)
    assert out["history.json"][alpha["id"]] == [{"ts": t1, "ability": "Old text."}]
    assert "since" not in _by_name(out["cards.json"])["Watermelon"]      # its text started at its last commit
    melon2 = _by_name(out["cards.json"])["Watermelon2"]
    assert out["history.json"][melon2["id"]] == [{"ts": t1, "ability": "Heal 1."}]


def test_older_versions_copied_by_the_versions_job(bs, repo, tmp_path, monkeypatch):
    """site.yml's build makes no older-version copies (VERSION_BUDGET 0): it lists every older version's PNG blob, newest
    first (VERSION_LIST), for its versions job (--versions); the next build links that job's copies (app.js versionCopy)."""
    listed, (c1, c2, _) = tmp_path / "copies.txt", repo["commits"]
    monkeypatch.setattr(bs, "VERSION_BUDGET", 0)
    monkeypatch.setattr(bs, "VERSION_LIST", str(listed))
    out = _build(bs, monkeypatch, repo, tmp_path / "a")
    copies = tmp_path / "a" / "dist" / "thumbs" / "v"
    blob = lambda c: subprocess.run(["git", "-C", str(repo["root"]), "rev-parse", f"{c}:{ALPHA}.png"], capture_output=True,
                                    text=True, check=True).stdout.strip()
    assert listed.read_text().split() == [blob(c2), blob(c1)] and not any(copies.iterdir())
    bs.version_copies(listed.read_text().split(), 60)                 # the versions job
    listed.unlink()
    _build(bs, monkeypatch, repo, tmp_path / "a")
    alpha, tag = _by_name(out["cards.json"])["Alpha"]["id"], out["cards.json"]["thumbs"]["v"]
    assert sorted(p.name for p in copies.iterdir()) == sorted(f"{alpha}-{c[:10]}-{tag}.avif" for c in (c1, c2))
    assert not listed.exists()


def test_a_renamed_row_shows_its_old_name_and_when_the_card_was_made(bs):
    """Archive rows: a renamed one has no title (its old name shows, as Discord's "renamed from") and the created date of
    the card it became, its renamed_to chain followed; a deleted one keeps its title and its own date."""
    from types import SimpleNamespace
    old = lambda title, day, **tag: [SimpleNamespace(display_name=title, created={"date": day, "from": "git"}, archived={"at": 1.0, **tag})]
    old_stats = {"Papyrus": old("The Great Papyrus", "2025-03-22", renamed_to="Papyrus 2"),
                 "Papyrus 2": old("Papyrus 2", "2025-04-01", renamed_to="The Great Papyrus"), "Gone": old("Gone Card", "2024-01-01")}
    rows = bs.archived_cards(old_stats, {"The Great Papyrus": "papyrus"}.get, created_of={"The Great Papyrus": "2020-07-22"}.get)
    shown = {r["card"]: (r.get("title"), r["created"]) for r in rows}
    assert shown == {"Papyrus": (None, "2020-07-22"), "Papyrus 2": (None, "2020-07-22"), "Gone": ("Gone Card", "2024-01-01")}


def test_editor_stats_and_people_load_on_first_use(bs, repo, tmp_path, monkeypatch):
    """The editor (editing.js, with the sign-in: signin.js), Stats (charts.js), People (profile.js), the font box (tokens.js), the
    rulebook (rulebook.js, its editor rulebook-edit.js), the Changes tabs (changelog.js), the published decks (decks.js) and the
    deck builder's picker (pick.js) are imported by app.js with import(): without
    esbuild as stamped modules no page preloads; with it as chunks of their own, each with its stamped source map, none of
    their code in what a first visit loads (app.js and the chunks it imports), and the meta chunks list naming
    each one's files (app.js early()). Share pages preload the chunks app.js imports, index.html doesn't; its head script
    names the rulebook's for a Rulebook link. A card page preloads the picture size its own picture picks (one download)."""
    exe = bs.esbuild()
    monkeypatch.setattr(bs, "esbuild", lambda: None)
    _build(bs, monkeypatch, repo, tmp_path / "a")
    assets, page = tmp_path / "a" / "dist" / "assets", (tmp_path / "a" / "dist" / "index.html").read_text(encoding="utf-8")
    lazy = ("editing.js", "charts.js", "profile.js", "tokens.js", "rulebook.js", "rulebook-edit.js", "changelog.js", "decks.js", "pick.js")
    app = (assets / "app.js").read_text(encoding="utf-8")
    assert all(re.search(rf'import\("\./{re.escape(m)}\?v=\w+"\)', app) for m in lazy)
    assert not re.search(r'modulepreload" href="assets/(edit|stats|profile|preview|mockup|balance|decks|pick|signin|changelog)\.js', page)
    alpha = (tmp_path / "a" / "dist" / "c" / "alpha" / "index.html").read_text(encoding="utf-8")
    pre = re.search(r'<link rel="preload" as="image" imagesrcset="([^"]+)" imagesizes="([^"]+)"', alpha)
    assert pre and pre.groups() == re.search(r'<img id="detail-img" [^>]*srcset="([^"]+)" sizes="([^"]+)"', alpha).groups() and " 480w, " in pre[1]
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
                         "rulebook.js", "rulebook-edit.js", "changelog.js", "decks.js", "pick.js", "signin.js"}
    assert {"edit.js", "mockup.js", "signin.js"} <= set().union(*(sources[f] for f in chunks["editing.js"]))
    assert {"stats.js", "balance.js"} <= set().union(*(sources[f] for f in chunks["charts.js"]))


def test_website_only_fields(bs, tmp_path, monkeypatch):
    """`_site` (owner 10-25): applied, never "PSD pending", the card's siteOnly (its "Website only" tag); edit.js stores it
    from an op's ticks as parse_edits and CardEdits.site_only clean it (a field without a value goes); unticked: pending."""
    from commands.card_edits import CardEdits
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


def test_card_inks_on_a_real_card(bs, clone):
    """Tainted Lazarus: its two Ability layers' runs on the parsed ability (joined; no [undead] since 11-09, its creature 2
    header icon): the status words in their colours, the white title none."""
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


def test_a_web_chapter_is_not_printed(bs, tmp_path):
    """Owner 2026-11-07: a chapter titled {.web} is on the website only (rulebook.js printed, build_site printed_chapters)."""
    book_js = BUILD_SITE.parents[1] / "assets" / "rulebook.js"
    book = [{"id": "stats", "source": "# Stats\n\n## HEALTH {.hp}\nWords.\n"}, {"id": "font", "source": "# Font {.web}\n\nWords.\n"}]
    assert [c["id"] for c in bs.printed_chapters(book)] == ["stats"]
    probe = tmp_path / "probe.mjs"
    probe.write_text(f"const rb = await import({json.dumps(book_js.as_uri())});\n"
                     f"process.stdout.write(JSON.stringify(rb.printed({json.dumps(book)}).map(c => c.id)));\n", encoding="utf-8")
    done = subprocess.run(["node", str(probe)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == ["stats"]


def test_psd_sizes_read_only_new_folders(bs, tmp_path, monkeypatch):
    """site.yml's PSD sizes (psd_sizes.py): kept by blob; the first run reads the whole tree, a later one only a new
    PSD's folder; build_site.psd_sizes reads its file as the trees API's reply."""
    spec = importlib.util.spec_from_file_location("site_psd_sizes", BUILD_SITE.parent / "psd_sizes.py")
    ps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ps)
    repo, out = tmp_path / "repo", str(tmp_path / "tree.json")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for path, data in (("A/x.psd", b"aaa"), ("B/y.PSD", b"bbbbb"), ("B/y.png", b"p")):
        (repo / path).parent.mkdir(exist_ok=True)
        (repo / path).write_bytes(data)
    first, _ = _commit(repo, "one", 1_700_000_000)
    (repo / "A/x.psd").write_bytes(b"aaaa")
    second, _ = _commit(repo, "two", 1_700_000_100)
    git = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True).stdout
    asked = []

    def trees(path):        # the trees API, from git
        asked.append(path)
        rows = git("ls-tree", "-l", *(["-r", path.split("?")[0]] if "?" in path else [path])).splitlines()
        return [{"path": p, "type": m.split()[1], "sha": m.split()[2], "size": int(m.split()[3])}
                for m, p in (r.split("\t") for r in rows)], False

    monkeypatch.chdir(repo)
    monkeypatch.setattr(ps, "CACHE", str(tmp_path / "cache" / "psd" / "sizes.json"))
    monkeypatch.setattr(ps, "MANY", 1)
    monkeypatch.setattr(ps, "trees", trees)
    assert ps.main(first, out) == 0 and asked == [f"{first}?recursive=1"]
    assert sorted(bs.psd_sizes(out).values()) == [3, 5]
    asked.clear()
    assert ps.main(second, out) == 0 and asked == [git("rev-parse", f"{second}:A").strip()]
    assert sorted(bs.psd_sizes(out).values()) == [4, 5]
    assert sorted(json.loads((tmp_path / "cache" / "psd" / "sizes.json").read_text()).values()) == [4, 5]
