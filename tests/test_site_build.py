"""
The website build (TTSCardMaker site/build/build_site.py): one smoke test per piece (owner 2026-11-12) and the checks of
the bot against the site. The build on a small card repo made here, offline, from the card sync's db/ (CARD_DATA_DIR,
load_card_data): the current text dated from its start; the bundle's lazy chunks; website-only fields (edit.js and
CardEdits); the parser notes an edit answers; a real card's inks (psd_details); R14's names (card_standards); web-only
rulebook chapters (rulebook.js); the card timelines before the repo (HISTORY_DIR, discord_preview). Loads build_site.py
from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone.
"""
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from html import unescape as html_unescape
from io import BytesIO
from os import environ
from pathlib import Path

import pytest

import commands.card_db as cdb
import commands.card_models as cm
from conftest import SITE_SRC

BUILD_SITE = SITE_SRC / "site" / "build" / "build_site.py"
EDIT_JS = BUILD_SITE.parents[1] / "assets" / "edit.js"
STYLE_CSS = BUILD_SITE.parents[1] / "assets" / "style.css"
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
    got = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in data.glob("*.json")}
    got["versions/"] = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (data / "versions").glob("*.json")}
    return got


def _versions(out, cid: str) -> dict:
    """A card's data/versions/<id>.json (app.js loadVersions): {repo, visible, hb?, commits, cards, minor?, early?}."""
    return out["versions/"][cid]


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


def test_mentions_built_as_the_site_finds_them(bs, monkeypatch):
    """cards.json mentions (mention_lists: text.js in node): per card the indexes of the cards its ability names, 0 for
    none, hidden cards neither naming nor named (app.js derive() at boot); no node, none (the boot parses)."""
    if not hasattr(bs, "mention_lists"):
        pytest.skip("a site before built mentions")
    cards = [{"id": "alpha", "name": "Alpha", "type": "creature", "ability": "Summon a Beta. Alpha heals."},
             {"id": "beta", "name": "Beta", "type": "minion", "ability": "Hits Gamma."},
             {"id": "gamma", "name": "Gamma", "type": "creature", "ability": "Alpha.", "hidden": True},
             {"id": "delta", "name": "Delta", "type": "item", "ability": ""}]
    assert bs.mention_lists(cards, {}, {}) == [[1, 0], 0, 0, 0]
    monkeypatch.delenv("NODE", raising=False)
    monkeypatch.setattr(bs.shutil, "which", lambda name: None)
    assert bs.mention_lists(cards, {}, {}) is None


def test_the_current_text_is_dated_from_its_start(bs, repo, tmp_path, monkeypatch):
    """Alpha's text started at c2 (the bot's carried start), though its PSD was last committed at c3: cards.json keeps
    ts (the last commit, "updated") and adds since; history.json holds the older text at c1. A variant picture's goes
    back through its picture 1's. cards.json's recent holds the earlier texts of the cards
    whose update changed the text (not Alpha's): Recently updated draws at once."""
    t1, t2, t3 = repo["t"]
    out = _build(bs, monkeypatch, repo, tmp_path / "a")
    alpha = _by_name(out["cards.json"])["Alpha"]
    assert (alpha["ts"], alpha["since"]) == (t3, t2)
    assert out["history.json"][alpha["id"]] == [{"ts": t1, "ability": "Old text."}]
    assert "since" not in _by_name(out["cards.json"])["Watermelon"]      # its text started at its last commit
    melon2 = _by_name(out["cards.json"])["Watermelon2"]
    assert out["history.json"][melon2["id"]] == [{"ts": t1, "ability": "Heal 1."}]
    melon = _by_name(out["cards.json"])["Watermelon"]                   # Recently updated: only the texts that changed
    assert out["cards.json"]["recent"] == {melon["id"]: "Heal 1.", melon2["id"]: "Heal 1."}


def test_a_merge_keeping_the_older_picture_stays_pinned(bs, tmp_path, monkeypatch):
    """A merge that kept the other side's older PNG (TC a7b787f0, Chris & Jordi): the pin (og:image, deck.js) is the
    commit that made that picture, not none (raw at each build's HEAD: every build redeployed)."""
    root, png, start = tmp_path / "merge", tmp_path / "merge" / f"{ALPHA}.png", 1_780_000_000
    for key in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{key}_NAME", "t")
        monkeypatch.setenv(f"GIT_{key}_EMAIL", "t@t")
    git = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    _png(png, 10)
    _commit(root, "Card", start)
    git("checkout", "-q", "-b", "guards")
    _png(png, 20)
    kept, _ = _commit(root, "Guards: re-export", start + DAY)
    git("checkout", "-q", "main")
    _png(png, 30)
    (root / f"{ALPHA}.psd").write_text("psd")                    # both sides differ: the walk takes both
    _commit(root, "New art", start + 2 * DAY)
    git("merge", "-q", "-X", "theirs", "guards", "-m", "Merge")
    last = {}
    monkeypatch.setattr(bs, "REPO", str(root))
    commits = bs.git_history(last)
    newest = next(c[0] for c in commits if any(e[3] == f"{ALPHA}.png" for e in c[5]))
    assert newest != kept and last[(f"{ALPHA}.png", git("rev-parse", f"HEAD:{ALPHA}.png"))] == kept


def test_older_versions_copied_onto_the_plates_branch(bs, repo, tmp_path, monkeypatch):
    """site.yml's build lists every older version's PNG blob and plates file, newest first (VERSION_LIST), while the plates
    branch's v/ (VERSION_PLATES) lacks one, and puts none in the site; its versions job (--versions, a blobless depth-1
    checkout) commits the copies onto the branch beside the plates (app.js versionCopy). All there: no commit, no list."""
    listed, have, (c1, c2, _) = tmp_path / "copies.txt", tmp_path / "plates-v.txt", repo["commits"]
    origin, job = tmp_path / "origin.git", tmp_path / "job"
    for key in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{key}_NAME", "t")
        monkeypatch.setenv(f"GIT_{key}_EMAIL", "t@t")
    git = lambda where, *a, data=None: subprocess.run(["git", "-C", str(where), *a], input=data, capture_output=True,
                                                      check=True).stdout.decode().strip()
    git(tmp_path, "init", "-q", "--bare", str(origin))
    for key in ("allowFilter", "allowAnySHA1InWant"):
        git(origin, "config", f"uploadpack.{key}", "true")
    git(repo["root"], "push", "-q", str(origin), "HEAD:refs/heads/main")
    plate = f"{'a' * 40}.json"                    # card_plates' file, kept
    tree = git(origin, "mktree", data=f"100644 blob {git(origin, 'hash-object', '-w', '--stdin', data=b'{}')}\t{plate}\n".encode())
    git(origin, "update-ref", "refs/heads/plates", git(origin, "commit-tree", tree, "-m", "plates"))
    monkeypatch.setattr(bs, "VERSION_LIST", str(listed))
    monkeypatch.setattr(bs, "VERSION_PLATES", str(have))
    out = _build(bs, monkeypatch, repo, tmp_path / "a")
    alpha, tag = _by_name(out["cards.json"])["Alpha"]["id"], out["cards.json"]["thumbs"]["v"]
    files = [f"v/{alpha}/{c[:10]}-{tag}.avif" for c in (c2, c1)]
    blob = lambda c: git(repo["root"], "rev-parse", f"{c}:{ALPHA}.png")
    assert listed.read_text().split() == [blob(c2), files[0], blob(c1), files[1]]
    assert not (tmp_path / "a" / "dist" / "thumbs" / "v").exists()
    git(tmp_path, "clone", "-q", "--filter=blob:none", "--depth", "1", f"file://{origin}", str(job))
    monkeypatch.setattr(bs, "REPO", str(job))
    assert bs.publish_versions(str(listed))
    assert git(origin, "ls-tree", "-r", "--name-only", "plates").split() == sorted([plate, *files])
    from PIL import Image
    assert Image.open(BytesIO(subprocess.run(["git", "-C", str(origin), "show", f"plates:{files[1]}"], capture_output=True,
                                             check=True).stdout)).format == "AVIF"
    assert bs.publish_versions(str(listed)) == ""                       # nothing missing
    have.write_text("\n".join(files))
    listed.unlink()
    _build(bs, monkeypatch, repo, tmp_path / "a")
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


def test_a_rename_redrawn_under_another_file_name_keeps_its_versions(bs, tmp_path, monkeypatch):
    """A person's rename that also redrew the picture under another file name (TC c05e5d4a, Man_Serpent_Shield.png ->
    Man-Serpent's_Shield.png: no Renamed trailer, no same blob or file name): Image versions go on under the old name
    through the card data's archive tag."""
    root, folder = tmp_path / "repo", "Items/Heal/2 Stars"
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / folder).mkdir(parents=True)
    (root / f"{folder}/Pizza.psd").write_text("pizza")
    _png(root / f"{folder}/Pizza.png", 10)
    c1, t1 = _commit(root, "Pizza", 1_780_000_000)
    for ext in ("psd", "png"):
        (root / f"{folder}/Pizza.{ext}").unlink()
    (root / f"{folder}/Pepperoni_Pizza.psd").write_text("pepperoni")
    _png(root / f"{folder}/Pepperoni_Pizza.png", 20)
    c2, t2 = _commit(root, "Pepperoni", 1_780_000_000 + DAY)
    new = _card("Pepperoni Pizza", f"{folder}/Pepperoni_Pizza.psd", "Heal 3.", t2, card_type="item")
    old = _card("Pizza", f"{folder}/Pizza.psd", "Heal 2.", t1, card_type="item")
    old.archived = {"at": float(t2), "path": f"{folder}/Pizza.psd", "commit": c2, "renamed_to": "Pepperoni Pizza"}
    new.parser_version = old.parser_version = 2
    db = tmp_path / "db"
    cdb.write_db(str(db), {"Pepperoni Pizza": new}, defaultdict(list, {"Pizza": [old]}), json.dumps({"version": 1, "cards": {}}),
                 {"card_commit": c2})
    out = _build(bs, monkeypatch, {"root": root, "db": db}, tmp_path / "a")
    cid = _by_name(out["cards.json"])["Pepperoni Pizza"]["id"]
    one = _versions(out, cid)
    assert [(one["commits"][r[0]][0], r[1:]) for r in one["cards"][cid]] == [(c2, []), (c1, [f"{folder}/Pizza.png"])]


def test_editor_stats_and_people_load_on_first_use(bs, repo, tmp_path, monkeypatch):
    """The editor (editing.js), the sign-in form shown while it loads (signin.js, also in the editor's), Stats (charts.js), a card's Balance block (balance.js, no stats.js:
    the chart primitives are chartkit.js), People (profile.js), the font box (tokens.js), the
    rulebook (rulebook.js, its editor rulebook-edit.js), the Changes tabs (changelog.js), the published decks (decks.js) and the
    deck builder's picker (pick.js) are imported by app.js with import(): without
    esbuild as stamped modules no page preloads; with it as chunks of their own, each with its stamped source map, none of
    their code in what a first visit loads (app.js and the chunks it imports), and the meta chunks list naming
    each one's files (app.js early()). Share pages preload the chunks app.js imports, index.html doesn't; its head script
    names the rulebook's for a Rulebook link. A card page preloads the picture size its own picture picks (one download); a
    person page People's files and chunks, after style.css (its first paint first)."""
    exe = bs.esbuild()
    monkeypatch.setattr(bs, "esbuild", lambda: None)
    _build(bs, monkeypatch, repo, tmp_path / "a")
    assets, page = tmp_path / "a" / "dist" / "assets", (tmp_path / "a" / "dist" / "index.html").read_text(encoding="utf-8")
    lazy = ("editing.js", "signin.js", "charts.js", "balance.js", "profile.js", "tokens.js", "rulebook.js", "rulebook-edit.js", "changelog.js", "decks.js", "pick.js")
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
    person = next((tmp_path / "b" / "dist" / "u").glob("*/index.html")).read_text(encoding="utf-8")
    head, css, after = person.partition('<link rel="stylesheet" href="assets/style.css')
    assert css and sorted(hinted(head)) == sorted(first[1:]) and "data/users.json" not in head
    assert sorted(hinted(after)) == sorted({*chunks["profile.js"], *chunks["decks.js"]}) and 'href="data/contributions.json?v=' in after
    sources = {}
    for f in {*first, *(n for v in chunks.values() for n in v)}:
        code = (assets / f).read_text(encoding="utf-8")
        assert re.search(rf"//# sourceMappingURL={re.escape(f)}\.map\?v=\w+\n$", code)
        sources[f] = {s.removeprefix("src/") for s in json.loads((assets / f"{f}.map").read_text(encoding="utf-8"))["sources"]}
    loaded = set().union(*(sources[f] for f in first))
    assert {"app.js", "grid.js", "text.js"} <= loaded
    assert not loaded & {"edit.js", "inline.js", "preview.js", "mockup.js", "tokens.js", "stats.js", "balance.js", "insights.js", "chartkit.js",
                         "profile.js", "rulebook.js", "rulebook-edit.js", "changelog.js", "decks.js", "pick.js", "signin.js"}
    src = lambda k: set().union(*(sources[f] for f in chunks[k]))
    assert {"edit.js", "mockup.js", "signin.js"} <= src("editing.js")
    assert "signin.js" in src("signin.js") and not src("signin.js") & {"edit.js", "mockup.js", "preview.js", "inline.js", "tokens.js"}
    assert {"stats.js", "balance.js"} <= src("charts.js") and {"balance.js", "chartkit.js"} <= src("balance.js")
    assert "stats.js" not in src("balance.js") | src("profile.js") | src("editing.js")


def test_head_script_starts_the_pages_data(bs, tmp_path):
    """index.html's head script, in node: a link to the Rulebook, a Changes tab, Stats or People starts its files (app.js
    dataUrl's URLs) and chunks, the Changelog's and Database edits' first screen's file instead of the whole unless a ?query
    or a reload draws them whole; the plain grid its first pictures; any other link (the profile editor too) nothing."""
    lazy = {"rulebook.js": ["rulebook-A.js"], "changelog.js": ["changelog-E.js"], "charts.js": ["charts-B.js", "chunk-C.js"],
            "balance.js": ["balance-F.js", "chunk-C.js"], "profile.js": ["profile-D.js", "chunk-C.js"], "decks.js": ["decks-G.js"]}
    tag = bs.grid_preload([{"id": "a", "name": "A", "img": "i"}], version="ab12", lazy=lazy, own={"users.json": "cd34", "edits.json": "x"})
    script = tmp_path / "head.mjs"
    script.write_text(f"""const out = {{}}, set = (k, value) => Object.defineProperty(globalThis, k, {{ value, configurable: true, writable: true }});
for (const [hash, type] of [["#/rules/3"], ["#/changes/changelog"], ["#changes/changelog?kind=x"], ["#/changes/changelog", "reload"], ["#/changes/edits/"],
    ["#/changes/archive"], ["#/stats"], ["#/users"], ["#/u/Me"], ["#/u/Me/edit"], ["#/changes"], ["#/changes/parser"], ["#/decks"], [""]]) {{
  const links = out[hash + (type ? " " + type : "")] = [];
  set("location", {{ hash }}); set("localStorage", {{ getItem: () => null }}); set("performance", {{ getEntriesByType: () => [{{ type: type || "navigate" }}] }});
  set("document", {{ createElement: () => ({{}}), head: {{ append: l => links.push(l.as === "image" ? "image" : l.href) }} }});
  {re.fullmatch(r"<script>(.*)</script>", tag, re.S)[1]}
}}
process.stdout.write(JSON.stringify(out));""", encoding="utf-8")
    done = subprocess.run(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    data = lambda *names: [f"data/{n}?v={'cd34' if n == 'users.json' else 'ab12'}" for n in names]
    log, people = data("changelog.json") + ["assets/changelog-E.js"], data("users.json", "contributions.json", "decks.json") + ["assets/profile-D.js", "assets/chunk-C.js", "assets/decks-G.js"]
    assert json.loads(done.stdout) == {
        "#/rules/3": data("rules.json") + ["assets/rulebook-A.js"], "#/changes/changelog": data("changelog-head.json") + log[1:],
        "#changes/changelog?kind=x": log, "#/changes/changelog reload": log, "#/changes/edits/": data("edits-head.json") + log[1:],
        "#/changes/archive": data("archive.json") + ["assets/changelog-E.js"],
        "#/stats": data("analytics.json") + ["assets/charts-B.js", "assets/chunk-C.js", "assets/balance-F.js"],
        "#/users": people, "#/u/Me": people, "#/u/Me/edit": [], "#/changes": [], "#/changes/parser": [], "#/decks": [], "": ["image"]}


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


def test_a_shrunk_ability_reaches_the_mock_up(bs):
    """r16: a bake's ability_fit (card_bake.FIT) on the tree's PSD is the card's abilityFit (the mock-up grows it back);
    another PSD's, or one out of range, nothing."""
    import commands.card_bake as cb
    verdict = {"state": "baked", "psd": "abcd1234", "reason": "ability", cb.FIT: 0.8836}
    for psd, fit, want in (("abcd1234ff", 0.8836, 0.8836), ("99999999aa", 0.8836, None), ("abcd1234ff", 0.1, None)):
        card = {"key": "Mini_Doomer", "name": "Mini Doomer", "problems": []}
        bs.apply_edits(card, {"mini doomer": ({}, {"_psd": {**verdict, cb.FIT: fit, cb.TITLE_FIT: fit,
                                                            cb.TITLE_WAS: [[236, 34, 964, 120]]}})},
                       psd_sha=psd, shown="Mini Doomer")
        assert card.get("abilityFit") == card.get("titleFit") == want     # a title set smaller (r52) likewise
        assert card.get("titleWas") == ([[236, 34, 964, 120]] if want else None)    # with the maker's boxes
    card = {"key": "Mini_Doomer", "name": "Mini Doomer", "problems": []}
    bs.apply_edits(card, {"mini doomer": ({}, {"_psd": {**verdict, cb.TITLE_FIT: 0.8836}})}, psd_sha="abcd1234ff",
                   shown="Mini Doomer")
    assert "titleFit" not in card and "titleWas" not in card     # without them: as the bake, no title fit
    parts = {"422": [0.78, [[259, 26, 935, 107]]], "423": [0.5, [[1, 2, 3, 4]]], "424": [0.9, []]}
    card = {"key": "Mini_Doomer", "name": "Mini Doomer", "problems": []}     # a split title's parts (r4-2)
    bs.apply_edits(card, {"mini doomer": ({}, {"_psd": {**verdict, cb.TITLE_FITS: parts}})}, psd_sha="abcd1234ff",
                   shown="Mini Doomer")
    assert card["titleFits"] == {"422": [0.78, [[259, 26, 935, 107]]]}


def test_web_fonts_say_which_typed_characters_they_lack(bs, tmp_path, monkeypatch):
    """data/fonts.json: a used font's missing typed characters and its .notdef (mockup.js draws them as the bake does),
    read once per blob and only for fonts the cards use."""
    from helpers import build_font
    raw, reads = Path(build_font(str(tmp_path / "a.ttf"), "Fix-Regular", "Fix", "Regular")).read_bytes(), []
    monkeypatch.setattr(bs, "CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(bs, "font_bytes", lambda p: reads.append(p) or raw)
    tree = {"MDW/Fonts/a.ttf": "1" * 40, "MDW/Fonts/b.ttf": "2" * 40}   # b: the same face, unused (a wins by path)
    first, again = bs.web_fonts(tree, {"Fix-Regular"}), bs.web_fonts(tree, {"Fix-Regular"})
    fix = first["Fix-Regular"]
    assert first == again and sorted(reads) == ["MDW/Fonts/a.ttf", "MDW/Fonts/a.ttf", "MDW/Fonts/b.ttf"]
    assert fix["file"] == "MDW/Fonts/a.ttf" and fix["notdef"] == [500]   # an empty .notdef: the bake leaves a gap
    assert "A" not in fix["lacks"] and {"+", ".", "\u2014", "\u2026"} <= set(fix["lacks"]) and "empty" not in fix
    if getattr(bs, "GAPS_VERSION", 1) >= 2:     # drawn empty (Belisa plumilla manual's): the bake sets another font
        blank = Path(build_font(str(tmp_path / "c.ttf"), "Gap-Regular", "Gap", "Regular", blank=":%")).read_bytes()
        assert bs.font_gaps(blank)[0][2] == "%:"


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


def test_game_txt_rulebook_says_what_the_bot_does(bs):
    """game.txt's chapters (build_site.rulebook_text) read as the bot's (rulebook.plain_text): a panel's "Title" kept, a
    row's widths dropped."""
    from commands import rulebook
    source = '::: row 3 1\n::: panel "🕐FORCED" {.light}\n(BOT) = Beginning of turn.\n:::\n:::\n'
    assert bs.rulebook_text(source) == rulebook.plain_text(source) == "🕐FORCED\n(BOT) = Beginning of turn."


def test_tts_and_game_txt_kinds_name_each_item_type(bs):
    """tts.json's k (TTS Card details, Find a card) and game.txt's kind name each item type an item prints, as the card
    page (grid.js itemTypes; one map, card_models'); a field card's own icon none."""
    cards = [dict(name="Life Shroom", type="item", subtype="heal", types=["food", "fragile equip"], stars=3, png="a.png"),
             dict(name="Electric Dart", type="nme", subtype="attack", types=["nme", "attack", "food"], stars=2, png="b.png"),
             dict(name="Nexus", type="field", types=["field"], stars=4, png="c.png")]
    kinds = ["Item (heal, equip)", "N.M.E. item (attack, heal)", "Field card"]
    assert [c["k"] for c in bs.tts_details(cards, [], {}, {})["cards"].values()] == kinds
    game = bs.game_text(cards, [], {"notes": [], "groups": []}, {})
    assert all(f"{c['name']} | {k} | {c['stars']}★ |" in game for c, k in zip(cards, kinds))
    assert bs.ICON_ITEM_TYPES == cm.AUX_ITEM_TYPES


def test_tts_rules_buttons_follow_minions_death_attacks_and_the_field(bs):
    """tts.json r (Card details' Rules buttons): a minion card named or 'minions' -> Minions ('On first summon' not, nor
    a minion card: its type's), revive/graveyard -> Death, redirect -> Attacking, PF as printed/primary field -> The Field."""
    chapters = [{"id": i, "num": n, "title": i, "source": s} for i, n, s in (
        ("attacking", "10", "# Attacking\n"), ("the-field", "16", "# The Field\n"), ("death", "17", "# Death\n"),
        ("minions", "15", "# Minions\n[offensive_minion]:Offensive minions occupy the same field.\n"))]
    cards = [dict(name="Mini Bee", type="minion", types=["offensive_minion"], ability="Attacks the minion's PF target.", png="m.png"),
             dict(name="Hive Knight", type="creature", ability="(OYT) Summon a Mini Bee.", png="a.png"),
             dict(name="Grimora", type="creature", ability="Revive a creature from the graveyard to the primary field.", png="b.png"),
             dict(name="Portal", type="item", ability="Redirect an attack. On first summon, pf and sf.", png="c.png")]
    r = {c["n"]: c.get("r") for c in bs.tts_details(cards, chapters, {"offensiveminion": {}}, {})["cards"].values()}
    assert r == {"Mini Bee": ["the-field"], "Hive Knight": ["minions"], "Grimora": ["death", "the-field"], "Portal": ["attacking"]}


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


def test_every_dialog_has_a_name():
    """Screen readers say a dialog's name, not just 'dialog' (2026-11-17): every role="dialog" carries aria-label or
    aria-labelledby; showDialog's points at the heading it shows."""
    for js in sorted(STYLE_CSS.parent.glob("*.js")):
        for tag in re.findall(r'<div [^>]*role="dialog"[^>]*>', js.read_text(encoding="utf-8")):
            assert re.search(r"aria-label(ledby)?=", tag), (js.name, tag)
    app = (STYLE_CSS.parent / "app.js").read_text(encoding="utf-8")
    show = re.search(r"^function showDialog\(html\) \{.*?modal\.hidden = false; \}", app, re.S | re.M)[0]
    assert show.count('"dialog-title"') == 2                            # the tag's aria-labelledby and the heading's id


def test_the_page_under_a_dialog_goes_inert_after_its_paint():
    """An open dialog hides the page under it from screen readers at once (aria-hidden) and makes it inert just after
    its paint, or at once at a key, a tap or a focus under it (2026-11-20: the whole page's restyle slowed each open
    and close; Cmd+A and find must still skip it); a close clears inert before focus goes back, and a view drawn
    before the seal clears it first (11-21: the profile editor opened from a dialog must get its focus), and so does a
    mouse move after a close (a press on an inert view goes through it: the click was lost)."""
    app = (STYLE_CSS.parent / "app.js").read_text(encoding="utf-8")
    seal = app[app.index("function seal() {"):app.index("function shut() {")]
    shut = app[app.index("function shut() {"):app.index("\nshut();")]
    assert "view]" not in shut and 'view.ariaHidden = open ? "true" : null' in shut
    assert "requestAnimationFrame(() => setTimeout(() => sealing && seal()))" in shut
    assert "view.inert = open" in app[app.index("function inertNow() {"):app.index("function seal() {")]
    assert seal.index("inertNow()") < seal.index(".focus(")
    assert re.search(r"function renderView\(r, hash\) \{\n  if \(sealing\) inertNow\(\);", app)
    assert 'for (const t of ["keydown", "pointerdown"]) document.addEventListener(t, () => sealing && seal(), true)' in app
    assert 'document.addEventListener("pointermove", () => { if (sealing && modal.hidden) seal(); }, true)' in app
    assert re.search(r'view\.addEventListener\("focusin", e => \{ if \(!modal\.hidden\) \{ seal\(\);', app)


def _contrast(a: str, b: str) -> float:
    """WCAG contrast ratio of two #rgb / #rrggbb colours."""
    def lum(h):
        h = h[1:] if len(h) == 7 else "".join(c * 2 for c in h[1:])
        c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        c = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    lo, hi = sorted((lum(a), lum(b)))
    return (hi + 0.05) / (lo + 0.05)


def test_theme_colours_pass_the_contrast_test():
    """Owner D1/D2 2026-11-15: grey text, links and red 4.5:1 on each theme's page and surfaces, white 4.5:1 on the blue
    fill (and the light red); light stars keep the yellow with a thin dark edge, 3:1 on every light surface (non-text),
    on all four sides, the star picker's lit stars too."""
    css = STYLE_CSS.read_text(encoding="utf-8")
    for theme in ("dark", "light"):
        block = re.search(r':root\[data-theme="%s"\] \{(.*?)\n\}' % theme, css, re.S)[1]
        props = dict(re.findall(r"(--[\w-]+):\s*([^;]+);", block))
        col = lambda k, p=props: col(p[k][4:-1]) if p[k].startswith("var(") else p[k]
        for fg in ("--muted", "--accent", "--danger"):
            for bg in ("--page", "--surface", "--surface-2"):
                assert _contrast(col(fg), col(bg)) >= 4.5, (theme, fg, bg)
        assert _contrast(col("--accent-ink"), col("--accent-fill")) >= 4.5, theme
    assert _contrast("#fff", col("--danger")) >= 4.5                    # light, from here on
    for bg in ("--page", "--surface", "--surface-2", "--surface-3"):
        assert _contrast(col("--star-edge"), col(bg)) >= 3, bg
    for side in ("var(--star-o) 0", "var(--star-n) 0", "0 var(--star-o)", "0 var(--star-n)"):
        assert f"{side} var(--star-edge)" in props["--star-shadow"]
    assert ".stars { color: #f5c542;" in css and ':root[data-theme="light"] .stars { text-shadow: var(--star-shadow); }' in css
    assert re.search(r':root\[data-theme="light"\] :is\(\.star-pick i\.on,[^{]*\{ text-shadow: var\(--star-shadow\); \}', css)


def _timelines(root):
    """A HISTORY_DIR: Alpha's stale paper row (none since owner Q6, 11-14), Discord look, changelog facts (one its
    old-repo version confirms), old-repo version, git_first and a Discord post after its first version here; one Morgue
    joke card and M'uru ('Before 2020'); White Palace's (MDW) Discord pictures, one after its PNG's first record;
    Alpha's Created proposed as a physical card photo's. The joke card's list copy; M'uru's names another picture."""
    from PIL import Image

    def pic(folder, name):
        (root / folder).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 80), (90, 40, 20)).save(root / folder / name, "WEBP")
        return name
    stats = lambda hp, atk: {"value": {"hp": hp, "def": 2, "atk": atk, "spd": 1}, "method": "picture", "confidence": "sure"}
    alpha = {"card": "alpha", "key": "Alpha", "name": "Alpha", "kind": "creature", "entries": [
        {"id": "D-P-1", "kind": "paper", "date": None, "day": None, "label": "Before 2020", "image": pic("alpha", "P-1.webp"),
         "w": 40, "h": 80, "fields": {"name": {"value": "Alpha", "method": "eye", "confidence": "sure"}}, "changes": {}},
        {"id": "D-P-2", "kind": "discord", "date": "2021-03-04T05:06:07+00:00", "channel": "new-cards", "image": pic("alpha", "P-2.webp"),
         "w": 40, "h": 80, "fields": {"stats": stats(3, 5), "ability": {"method": "ocr", "confidence": "review"}}, "changes": {}},
        {"id": "F-1", "kind": "text", "date": "2022-10-01T00:00:00+00:00", "channel": "update-06", "change": {"field": "atk", "from": 5, "to": 4}},
        {"id": "F-2", "kind": "text", "date": "2023-04-20T00:00:00+00:00", "channel": "update-06", "change": {"field": "atk", "from": 5, "to": 4},
         "confirmed": True, "confirmed_by": "O-1"},
        {"id": "F-3", "kind": "text", "date": "2022-11-01T00:00:00+00:00", "channel": "update-06", "change": {"field": "type", "to": "dark", "op": "remove"}},
        {"id": "O-1", "kind": "old repo", "date": "2023-05-01T00:00:00+00:00", "author": "Maker", "subject": "Alpha art",
         "file": "alpha draft", "image": pic("alpha", "O-1.webp"), "w": 40, "h": 80, "fields": {"stats": stats(3, 4)},
         "changes": {"stats": [{"hp": 3, "def": 2, "atk": 5, "spd": 1}, {"hp": 3, "def": 2, "atk": 4, "spd": 1}], "bal": "nerf"}},
        {"id": "D-P-3", "kind": "discord", "date": "2027-01-01T00:00:00+00:00", "channel": "new-cards", "image": pic("alpha", "P-3.webp"),
         "w": 40, "h": 80, "fields": {}, "changes": {}}],
        "git_first": {"date": "2023-06-01T00:00:00+00:00", "approx": True,
                      "changes": {"stats": [{"hp": 3, "def": 2, "atk": 4, "spd": 1}, {"hp": 3, "def": 2, "atk": 3, "spd": 1}], "bal": "nerf"}}}
    joke = {"card": "joke-card", "key": "Joke Card", "name": "Joke Card", "kind_of": "joke", "live": "alpha",
            "first": {"date": "2020-07-14T03:00:00+00:00", "day": "2020-07-14", "channel": "meme-cards"},
            "entries": [{"id": "P-9", "kind": "discord", "date": "2020-07-14T00:00:00+00:00", "channel": "meme-cards",
                         "image": pic("_morgue/joke-card", "P-9.webp"), "w": 40, "h": 80, "fields": {}, "changes": {}}]}
    Image.new("RGB", (20, 40), (90, 40, 20)).save(root / "_morgue/joke-card/P-9-list.avif", "AVIF")
    joke["list"] = "P-9-list.avif"                                         # discord_preview list_copies
    muru = {"card": "m-uru", "key": "M'uru", "name": "M'uru", "kind_of": "played", "git_file": "first repo", "list": "P-1-list.avif",
            "first": {"date": None, "day": "Before 2020", "channel": None, "source": "physical card photo"},
            "entries": [{"id": "P-10", "kind": "discord", "date": "2020-07-18T00:00:00+00:00", "channel": "port",
                         "image": pic("_morgue/m-uru", "P-10.webp"), "w": 40, "h": 80, "fields": {}, "changes": {}}]}
    palace = {"card": "white-palace", "key": "White Palace", "name": "White Palace", "morgue": "mdw", "entries": [
        {"id": "O-p1", "kind": "old repo", "repo": "first", "date": "2023-03-01T00:00:00+00:00", "author": "Croissant",
         "subject": "Palace art", "image": pic("_morgue/white-palace", "O-p1.webp"), "w": 40, "h": 80,
         "fields": {"name": {"value": "White Palace", "method": "psd", "confidence": "sure"}}, "changes": {}}] + [
        {"id": f"P-{n}", "kind": "discord", "date": when, "channel": "new-cards", "image": pic("_morgue/white-palace", f"P-{n}.webp"),
         "w": 40, "h": 80, "fields": {}, "changes": {}} for n, when in ((7, "2026-01-01T00:00:00+00:00"), (8, "2027-06-01T00:00:00+00:00"))]}
    for folder, t in (("alpha", alpha), ("_morgue/joke-card", joke), ("_morgue/m-uru", muru),
                      ("_morgue/white-palace", palace)):
        (root / folder / "timeline.json").write_text(json.dumps(t), encoding="utf-8")
    index = {"version": 1, "cards": {"alpha": {"key": "Alpha", "dir": "alpha"}},
             "morgue": [{"dir": "_morgue/joke-card", "name": "Joke Card"}, {"dir": "_morgue/m-uru", "name": "M'uru"}],
             "mdw": [{"dir": "_morgue/white-palace", "name": "White Palace", "card": "White Palace"}],
             "created": {"Alpha": {"date": "Before 2020", "source": "physical card photo"}}}
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return root


def test_timelines_before_the_repo(bs, repo, tmp_path, monkeypatch):
    """The card timelines before the card repo (HISTORY_DIR, AutoCroissant discord_preview's layout): Image versions rows
    after the repo's (data/versions/<id>.json early) with their webps, none at or after the card's first version here, the Morgue's
    entries from Discord (never cards[], never on a renamed card's old page), an MDW Morgue card's pictures after its
    PNG's current version (none after its first record here), a physical card photo's Created ("Before 2020", owner Q6:
    a live card's and M'uru's; no paper rows) and their changes under Recorded changes (a type fact's "remove" as a
    removal), a confirmed fact once; without HISTORY_DIR, none of it."""
    if not hasattr(bs, "HISTORY_DIR"):
        pytest.skip("a site before HISTORY_DIR")
    monkeypatch.setattr(bs, "HISTORY_DIR", str(_timelines(tmp_path / "history")))
    _png(repo["root"] / "MDW/The Morgue/White_Palace.png", 70)
    palace_commit, _ = _commit(repo["root"], "Palace picture", 1_780_000_000 + 3 * DAY)
    out = _build(bs, monkeypatch, repo, tmp_path / "a")
    alpha = _by_name(out["cards.json"])["Alpha"]
    rows = _versions(out, alpha["id"])["early"][alpha["id"]]
    # after the repo's rows, newest first; the stale paper row left out; no review field, no link
    assert [r["k"] for r in rows] == ["old repo", "discord"] and not any("label" in r or "cov" in r for r in rows)
    assert rows[0]["a"] == "Maker" and rows[0]["s"] == "Alpha art" and rows[1]["ch"] == "new-cards"
    assert rows[0]["fl"] == "alpha draft" and "fl" not in rows[1]                  # a renamed or MDW file's name
    assert rows[1]["f"] == {"stats": {"hp": 3, "def": 2, "atk": 5, "spd": 1}}
    assert all((tmp_path / "a" / "dist" / r["img"]).is_file() for r in rows)
    assert alpha["ive"] == 2 and alpha["iv"] == len(_versions(out, alpha["id"])["cards"][alpha["id"]])
    # Created: a physical card photo's "Before 2020" sorts first; the file's date is kept as createdWas
    assert (alpha["created"], alpha["createdFrom"], alpha["createdBefore"]) == ("2019-01-01", "photo", 2020)
    assert "paper card" not in bs.HISTORY_FROM and "paper" not in bs.HISTORY_ROWS
    # the Morgue's entry from Discord: picture only, never a live card
    joke = [c for c in out["cards.json"]["morgue"] if c["name"] == "Joke Card"]
    assert joke and joke[0]["lost"] == "joke" and joke[0]["pic"] == "history/_morgue/joke-card/P-9.webp"
    if hasattr(bs, "HISTORY_LIST"):                                            # the Morgue list's copy of its picture
        assert joke[0]["lpic"] == "history/_morgue/joke-card/P-9-list.avif" and (tmp_path / "a/dist" / joke[0]["lpic"]).is_file()
        assert "lpic" not in next(c for c in out["cards.json"]["morgue"] if c["name"] == "M'uru")   # not its picture's
    assert joke[0]["first"] == "2020-07-13"                                         # the Pacific day, as its rows
    muru = next(c for c in out["cards.json"]["morgue"] if c["name"] == "M'uru")     # owner Q6: Created, no first day
    assert (muru["created"], muru["createdFrom"], muru["createdBefore"], muru["lost"]) == ("2019-01-01", "photo", 2020,
                                                                                          "played") and "first" not in muru
    assert muru["inGit"] and "inGit" not in joke[0]                                  # its first repo file: lost before this repo
    assert joke[0]["live"] == alpha["id"] and "live" not in muru and not any("_live" in c for c in out["cards.json"]["morgue"])   # an in-joke's live card
    assert "M'uru" not in _by_name(out["cards.json"])
    taken = bs.history_morgue({"morgue": [("_morgue/joke-card", json.loads((tmp_path / "history/_morgue/joke-card/timeline.json").read_text()))]},
                              bs.slugify, {"joke-card"})[0]
    assert taken[0]["id"] == "joke-card-morgue"                                      # a renamed card's old page stays its
    eld = {**json.loads((tmp_path / "history/_morgue/joke-card/timeline.json").read_text()),
           "first": {"date": "2023-10-29T17:04:44+00:00", "day": "2023-10-29", "channel": None, "source": "old repo"}}
    eld = bs.history_morgue({"morgue": [("_morgue/joke-card", eld)]}, bs.slugify)[0][0]
    assert eld["firstOld"] and eld["inGit"] and "firstCh" not in eld and "firstOld" not in joke[0]   # first seen in the old repo
    idea = {**json.loads((tmp_path / "history/_morgue/joke-card/timeline.json").read_text()), "kind_of": "idea"}
    assert bs.history_morgue({"morgue": [("_morgue/joke-card", idea)]}, bs.slugify)[0][0]["lost"] == "idea"   # owner 11-21
    if '"also"' in open(bs.__file__, encoding="utf-8").read():                   # one picture posted again (11-21)
        twice = json.loads((tmp_path / "history/_morgue/joke-card/timeline.json").read_text())
        twice["first"]["date"] = twice["entries"][0]["date"]                          # its first post
        twice["entries"][0]["also"] = [{"date": "2020-07-14T06:00:00+00:00", "day": "2020-07-14", "channel": "meme-cards"}]
        assert bs.history_morgue({"morgue": [("_morgue/joke-card", twice)]}, bs.slugify)[0][0]["also"] == [
            ["2020-07-13", "meme-cards"]] and "also" not in joke[0]                # its Pacific day
    assert "Joke Card" not in _by_name(out["cards.json"]) and _versions(out, joke[0]["id"])["early"][joke[0]["id"]]
    # an MDW Morgue card: its PNG's newest version is the current one; its Discord picture after its first record cut
    palace = next(c for c in out["cards.json"]["morgue"] if c["name"] == "White Palace")
    mine = _versions(out, palace["id"])
    assert [mine["commits"][r[0]][0] for r in mine["cards"][palace["id"]]] == [palace_commit]
    early = mine["early"][palace["id"]]                                             # its first repo row stays
    assert [r["img"] for r in early] == ["history/_morgue/white-palace/P-7.webp", "history/_morgue/white-palace/O-p1.webp"]
    assert (early[1]["r"], early[1]["a"], early[1]["f"]) == ("first", "Croissant", {"name": "White Palace"})
    assert (palace["iv"], palace["ive"]) == (1, 2)
    # Recorded changes: the old repo's nerf and the changelog's fact (data, never its text), each saying where from
    changes = out["analytics.json"]["cards"][alpha["id"]]["changes"]
    assert {"ts": 1682899200, "stats": [[3, 2, 5, 1], [3, 2, 4, 1]], "bal": "nerf", "early": "old repo"} in changes
    assert {"ts": 1664582400, "fact": {"field": "atk", "from": 5, "to": 4}, "early": "text"} in changes
    assert {"ts": 1667260800, "types": {"add": [], "drop": ["dark"]}, "early": "text"} in changes   # a type removed
    assert bs.history_shown({"field": "type", "to": "dark", "op": "remove"}, {"types": {"add": [], "drop": ["dark"]}})
    assert not [r for r in changes if r["ts"] == 1681948800]                         # confirmed: the old repo's nerf says it
    assert {"ts": 1685577600, "stats": [[3, 2, 4, 1], [3, 2, 3, 1]], "bal": "nerf", "early": "git", "approx": True} in changes
    # the first repo's versions: old repo rows marked r 'first', their changes from 'first repo'
    first = {"id": "O-0", "kind": "old repo", "repo": "first", "date": "2022-08-24T00:00:00+00:00", "author": "Maker",
             "subject": "Initial commit", "fields": {}, "changes": {"stars": [2, 3]}}
    assert bs.history_rows("alpha", {"entries": [first]})[0]["r"] == "first"
    assert bs.history_stats({"entries": [first]}, "creature")["rows"][0]["early"] == "first repo"
    assert bs.history_change({"ability_change": "removed"}) == {"words": True, "gone": True}   # a blank box (11-17)
    meme = {**first, "id": "O-1", "changes": {}, "meme_copy": True,
            "fields": {"stars": {"value": 3, "method": "psd", "confidence": "sure"}}}
    assert len(bs.history_stats({"entries": [first, meme]}, "creature")["points"]) == 1   # a Meme copy: no point
    assert bs.HISTORY_FROM["first repo"] == "first repo" and bs.HISTORY_FROM["changelog"] == "changelog"
    # owner 11-17 (3): a history branch checkout's pictures load from raw.githubusercontent.com at its commit (hb), none
    # in dist; one its tree lacks loses its img (the page says the picture is missing); the Morgue's pic that URL too
    if hasattr(bs, "history_host"):
        hist = tmp_path / "history"
        (hist / "alpha" / "O-1.webp").rename(tmp_path / "O-1.webp")          # committed without it
        git = lambda *a: subprocess.run(["git", "-C", str(hist), *a], capture_output=True, text=True, check=True).stdout
        git("init", "-q")
        _commit(hist, "History", 1_790_000_000)
        sha = git("rev-parse", "HEAD").strip()
        (tmp_path / "O-1.webp").rename(hist / "alpha" / "O-1.webp")          # on disk: the tree decides, not the folder
        branch = _build(bs, monkeypatch, repo, tmp_path / "h")
        hb = f"https://raw.githubusercontent.com/{bs.CARD_REPO}/{sha}/"
        rows = _versions(branch, alpha["id"])["early"][alpha["id"]]
        assert _versions(branch, alpha["id"])["hb"] == hb and not (tmp_path / "h" / "dist" / "history").exists()
        assert [r.get("img") for r in rows] == [None, "alpha/P-2.webp"]
        joke = next(c for c in branch["cards.json"]["morgue"] if c["name"] == "Joke Card")
        assert joke["pic"] == hb + "_morgue/joke-card/P-9.webp"
        assert not hasattr(bs, "HISTORY_LIST") or joke["lpic"] == hb + "_morgue/joke-card/P-9-list.avif"
        page = (tmp_path / "h/dist/c" / joke["id"] / "index.html").read_text(encoding="utf-8")   # its link preview and preload
        assert f'<meta property="og:image" content="{joke["pic"]}">' in page and 'og:image:height" content="80">' in page
        assert f'<link rel="preload" as="image" href="{joke["pic"]}" fetchpriority="high">' in page and "summary_large_image" in page
        monkeypatch.setattr(bs, "HISTORY_URL", "http://127.0.0.1:8000/h")    # a local preview's server
        assert bs.history_host(str(hist)) == ("http://127.0.0.1:8000/h/", None)
        monkeypatch.setattr(bs, "HISTORY_URL", "")
        shutil.rmtree(hist / ".git")
        # Created (owner 11-17 ~18:10): Pacific days; the timelines' label on the file's own day; made before a year
        evening = 1556862973                                                  # 2019-05-03 05:56 UTC, the 2nd in Pacific
        made = [{"key": "Up", "fs": evening}, {"key": "Pizza", "fs": 1700000000}, {"key": "Pep", "fs": 1600000000},
                {"key": "Hand", "fs": evening}]
        bs.apply_created(made, {"pizza": {"date": "2023-09-25", "from": "git"}, "hand": {"date": "2019-05-03", "from": "hand"},
                                "pep": {"date": "2020-07-21", "from": "discord", "before": 2020}},
                         {"pizza": {"date": "2023-09-25", "source": "old repo"}, "hand": {"date": "2019-05-02", "source": "discord"}})
        assert [(e["created"], e["createdFrom"]) for e in made] == [("2019-05-02", "git"), ("2023-09-25", "old repo"),
                                                                     ("2019-01-01", "photo"), ("2019-05-03", "hand")]
        assert made[2]["createdBefore"] == 2020 and "createdWas" not in made[1]
    # without the timelines: none of it
    monkeypatch.setattr(bs, "HISTORY_DIR", "")
    plain = _build(bs, monkeypatch, repo, tmp_path / "b")
    assert not any("early" in v for v in plain["versions/"].values()) and "ive" not in _by_name(plain["cards.json"])["Alpha"]
    assert all(not c.get("lost") and "iv" not in c for c in plain["cards.json"]["morgue"])
    assert not set(plain["versions/"]) & {c["id"] for c in plain["cards.json"]["morgue"]}


def test_versions_one_file_per_card(bs, repo, tmp_path, monkeypatch):
    """Image versions read one small file per card (data/versions/<id>.json, app.js loadVersions when the card opens): every
    card and Morgue entry with rows has one, holding its rows alone (its commits, minor, early) and the shared repo,
    visible and hb; one stamp for the folder (index.html's meta data, app.js dataUrl). Each also holds the card's own
    history.json texts and analytics.json Balance row, their commit indexes into its own lists (its dialog needs neither
    all-cards file)."""
    if not hasattr(bs, "version_files"):
        pytest.skip("a site before data/versions/")
    monkeypatch.setattr(bs, "HISTORY_DIR", str(_timelines(tmp_path / "history")))
    out = _build(bs, monkeypatch, repo, tmp_path / "a")
    per = out["versions/"]
    cards = cm.site_cards(out["cards.json"]) + out["cards.json"]["morgue"]
    assert set(per) == {c["id"] for c in cards if c.get("iv") or c.get("ive")}
    for cid, one in per.items():
        assert len(one["commits"]) == len({r[0] for r in one["cards"].get(cid, [])}) and set(one["cards"]) <= {cid}
        assert all(set(one.get(k, {})) <= {cid} for k in ("minor", "early")) and (cid in one["cards"] or cid in one["early"])
        assert all(one.get(k) == next(iter(per.values())).get(k) for k in ("repo", "visible", "hb"))
    hist, bal, texts = out["history.json"], out["analytics.json"], 0     # a card's own texts and Balance row (card_sections)
    for cid, one in per.items():
        if "texts" in one:
            texts += 1
            again = lambda v, shas: {**v, **{k: shas[v[k]] for k in ("c", "n") if k in v}}
            assert [again(v, one["shas"]) for v in one["texts"]] == [again(v, hist["_shas"]) for v in hist[cid]]
        else:
            assert cid not in hist
        if "bal" in one:
            undo = lambda a, commits: [{**r, **({"c": commits[r["c"]]} if "c" in r else {})} for r in a.get("changes", [])]
            mine, row = one["bal"]["card"], bal["cards"][cid]
            assert undo(mine, one["bal"]["commits"]) == undo(row, bal["commits"]) and {**mine, "changes": 0} == {**row, "changes": 0}
            co = row.get("cohort")
            assert one["bal"]["cohorts"] == ({co: bal["cohorts"][co]} if co in bal["cohorts"] else {}) and one["bal"]["minCohort"] == bal.get("minCohort")
        else:
            assert cid not in bal["cards"]
    assert texts and any("bal" in one for one in per.values())         # Alpha's older text, its Balance row
    meta = json.loads(html_unescape(re.search(r'<meta name="data" content="([^"]*)">', (tmp_path / "a" / "dist" / "index.html").read_text(encoding="utf-8"))[1]))
    assert re.fullmatch(r"[0-9a-f]{10}", meta["versions/"])
    app = (SITE_SRC / "site" / "assets" / "app.js").read_text(encoding="utf-8")
    assert 'getJson(`versions/${encodeURIComponent(c.id)}.json`)' in app and "(details|versions)\\/" in app and 'getJson("versions.json")' not in app


def _changes(seed_events: int, edit_events: int, batches: int) -> dict:
    """A card sync changelog: pushes over 40 days (minor art, a file rename, re-reads), one push older than the next
    build's (its seq after, its time before), the edit events and batches."""
    t0, events, seq = 1_790_000_000, [], 0
    def push(at, kinds, build, by="Maker"):
        nonlocal seq
        after = f"{seq + 1:040x}"
        for kind in kinds:
            seq += 1
            e = {"seq": seq, "at": at, "build": build, "kind": kind, "card": f"Card {seq}", "title": f"Card {seq}",
                 "path": f"Creatures/Other/2 Stars/Card_{seq}.psd", "push": {"after": after, "before": "0" * 40, "by": by, "at": at}}
            if kind in ("changed", "edited"):
                e["fields"] = {"hp": [1, 2], "ability": ["Hit the foe.", "Hit the foe twice."]}
            if kind == "art" and seq % 3:
                e["minor"] = True
            if kind == "moved":
                e["from_path"] = f"Creatures/Other/2 Stars/Old_{seq}.psd" if seq % 2 else f"Items/Heal/2 Stars/Card_{seq}.psd"
            if kind == "edited":
                e["edit_by"] = "Ann"
            events.append(e)
    n = 0
    while n < seed_events:
        at = t0 + n * 9000
        kinds = ["changed", "art", "art", "art", "new", "moved", "reparse"][n % 5:n % 5 + 1 + n % 4]
        push(at, kinds, f"b{n // 7}")
        n += len(kinds)
    push(t0 + n * 9000 - 30 * 86400, ["changed", "art"], "late")      # an older push carried in by a later build
    for i in range(edit_events):
        push(t0 + (n + i) * 9000, ["edited", "authors"][i % 2:i % 2 + 1 + i % 3], f"e{i // 5}", by="Bob")
    rows = [{"by": ["Ann"], "at": t0 + i * 5000, "via": ["website", "git"][i % 2], "sha": f"{i:040x}", "subject": f"Website edit {i}",
             "cards": [{"id": f"card-{i}", "name": f"Card {i}", "fields": {"hp": [1, i]}}]} for i in range(batches, 0, -1)]
    return {"days": 180, "repo": "o/r", "source": {"card_commit": "c" * 40}, "events": events, "archived": [], "edits": {"batches": rows}}


def test_changes_tabs_first_screens(bs, tmp_path):
    """The Changelog's and Database edits' first screens come from small head files (build_site changelog_heads): the
    newest pushes (at least HEAD_ROWS events, none older than a push left out, also when a push's seq and time
    disagree) with the whole file's heading counts, and the newest batches when every edit event fits. Drawn from the
    head, then the whole file's other parts appended (changelog.js headDrawn, changelogParts, editsParts), a tab is the
    same HTML as drawn from the whole file at once, also with no events (one heading); a head with no blocks while
    the whole file has some is drawn again whole."""
    cases = []
    for name, (seed, edits, batches) in {"many": (400, 300, 12), "few edits": (600, 30, 12), "no edits": (60, 0, 0), "empty": (0, 0, 0)}.items():
        data = _changes(seed, edits, batches)
        if name == "empty":                           # 180 quiet days, or card data without a changelog
            data["events"] = []
        files = bs.changelog_files(data)
        heads = bs.changelog_heads(files)
        head = heads["changelog-head.json"]["events"]
        if name == "few edits":                       # the late push: every push after it in time comes along
            late = next(e["at"] for e in files["changelog.json"]["events"] if e.get("build") == "late")
            assert len(head) < len(files["changelog.json"]["events"]) and min(e["at"] for e in head) == late
        for f in ("changelog.json", "edits.json", "changelog-head.json", "edits-head.json"):
            (tmp_path / f"{len(cases)}-{f}").write_text(json.dumps({**files, **heads}[f]), encoding="utf-8")
        cases.append(name)
    probe = tmp_path / "heads.mjs"
    probe.write_text(f"const cl = await import({json.dumps((BUILD_SITE.parents[1] / 'assets' / 'changelog.js').as_uri())});\n"
                     "const fs = await import('fs'), p = new URLSearchParams(), read = f => JSON.parse(fs.readFileSync(f));\n"
                     f"const dir = {json.dumps(str(tmp_path))}, out = [];\n"
                     f"for (let i = 0; i < {len(cases)}; i++) {{ const r = {{}};\n"
                     "  for (const [f, h, edits] of [['changelog.json', 'changelog-head.json', false], ['edits.json', 'edits-head.json', true]]) {\n"
                     "    const C = read(`${dir}/${i}-${f}`), H = read(`${dir}/${i}-${h}`), from = cl.headDrawn(H, C, edits);\n"
                     "    const whole = edits ? cl.editsHtml(C, p) : cl.changelogHtml(C, p), first = edits ? cl.editsHtml(H, p) : cl.changelogHtml(H, p);\n"
                     "    const rest = from === null ? null : [...(edits ? cl.editsParts(C, p, from) : cl.changelogParts(C, p, from))].join('');\n"
                     "    const stale = cl.headDrawn({ ...H, events: [] }, C, edits);\n"
                     "    r[f] = { same: first + rest === whole, from, part: first.length / whole.length, stale }; }\n"
                     "  out.push(r); }\n"
                     "process.stdout.write(JSON.stringify(out));\n", encoding="utf-8")
    done = subprocess.run(["node", str(probe)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    got = dict(zip(cases, json.loads(done.stdout)))
    assert all(r["same"] for c in got.values() for r in c.values()), got
    assert got["many"]["changelog.json"]["part"] < 0.5 and got["many"]["edits.json"]["from"]["header"] is False
    assert got["few edits"]["edits.json"]["from"] == {"groups": got["few edits"]["edits.json"]["from"]["groups"], "header": True, "batches": bs.HEAD_BATCHES}
    assert got["no edits"]["edits.json"]["from"]["batches"] == 0
    assert got["empty"]["changelog.json"]["from"] == 0 and got["many"]["changelog.json"]["stale"] is None
    assert got["many"]["edits.json"]["stale"] is None
    app =(BUILD_SITE.parents[1] / "assets" / "app.js").read_text(encoding="utf-8")   # a filter or a place: whole, as before
    assert "!S.restoring && !S.placing && ![...r.params.keys()].length ? HEADS[file] : null" in app
    assert 'if (S.restoring || S.placing || [...r.params.keys()].length) { page([...parts].join("")); return; }' in app
    assert "window.requestIdleCallback || (f => setTimeout(f, 0))" in app[app.index("function appendParts("):]   # Safari: no 50 ms gaps


def test_rename_keeping_the_shown_name_left_off(bs, tmp_path):
    """11-11's file names: a rename from the name the card shows today (P.A.N.D.A. Mk A to PANDA_Mk_A.psd) is off the
    Changelog (and its head counts), the Archive tab and person pages; a name change stays, a letter's case too (D11),
    and 'renamed to' shows the card's name today, not its file's."""
    renames = [("P.A.N.D.A. Mk A", "PANDA Mk A", "panda-mk-a"), ("omori", "Omori", "omori"), ("Avatar Of Wrath", "Avatar of Wrath", "avatar-of-wrath")]
    names = {"panda-mk-a": "P.A.N.D.A. Mk A", "omori": "OMORI", "avatar-of-wrath": "Avatar of Wrath"}
    push = {"after": "a" * 40, "before": "0" * 40, "by": "MichaelJSr", "at": 1_794_392_205}
    events = [{"seq": i + 1, "at": 1_794_392_205, "build": "b", "kind": "renamed", "card": old, "title": names[i_d], "renamed_to": new,
               "renamedId": i_d, "push": push} for i, (old, new, i_d) in enumerate(renames)]
    rows = [{"card": old, "renamed_to": new, "renamedId": i_d, "author": "MichaelJSr", "at": 1_794_392_205} for old, new, i_d in renames]
    assert bs.changelog_counts(events, names) == (2, {"renamed": 2}) and bs.changelog_counts(events) == (3, {"renamed": 3})
    assets = BUILD_SITE.parents[1] / "assets"
    probe = tmp_path / "renames.mjs"
    probe.write_text(f"const cl = await import({json.dumps((assets / 'changelog.js').as_uri())}), pf = await import({json.dumps((assets / 'profile.js').as_uri())});\n"
                     f"const names = new Map(Object.entries({json.dumps(names)}).map(([id, name]) => [id, {{ name }}])), rows = {json.dumps(rows)};\n"
                     "cl.useNames(id => names.get(id)?.name);\n"
                     f"const C = {{ days: 180, repo: 'o/r', events: {json.dumps(events)} }}, p = new URLSearchParams();\n"
                     "process.stdout.write(JSON.stringify([cl.changelogHtml(C, p), cl.archiveHtml(rows, 'o/r', p), pf.archivedPanelHtml(rows, names)]));\n",
                     encoding="utf-8")
    done = subprocess.run(["node", str(probe)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    for html, heading in zip(json.loads(done.stdout), ("Changelog", "Archive", "Archived cards")):
        assert re.findall(r'<a href="#/card/[^"]+">([^<]*)</a>', html) == ["OMORI", "Avatar of Wrath"], html
        assert "omori" in html and "Avatar Of Wrath" in html and "PANDA" not in html and "P.A.N.D.A." not in html
        assert f'{heading} <span class="muted">2</span>' in html


def test_morgue_tiles_show_the_list_copy(tmp_path):
    """A Morgue entry from Discord (cards.json morgue pic, lpic): its tiles, table rows and tips (grid.js thumb at 360 px
    or less, imgAttrs) show the 360 px list copy; the card view's sizes (480, 720: the dialog, the compare view) the
    webp; without a copy, the webp everywhere (grid.js in node)."""
    grid = SITE_SRC / "site" / "assets" / "grid.js"
    if "lpic" not in grid.read_text(encoding="utf-8"):
        pytest.skip("a site before the Morgue list copies")
    probe = tmp_path / "probe.mjs"
    probe.write_text(f"""const G = await import({json.dumps(grid.as_uri())});
const c = {{ pic: "h/P-1.webp", lpic: "h/P-1-list.avif", tw: 40, th: 80 }}, old = {{ pic: "h/P-1.webp" }};
process.stdout.write(JSON.stringify([G.imgAttrs({{}}, c, 168), ...[30, 360, 480, 720].map(w => G.thumb({{}}, c, w)),
  G.imgAttrs({{}}, old, 168), G.thumb({{}}, old, 360)]));
""", encoding="utf-8")
    done = subprocess.run(["node", str(probe)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == ['src="h/P-1-list.avif" width="40" height="80"', "h/P-1-list.avif", "h/P-1-list.avif",
                                       "h/P-1.webp", "h/P-1.webp", 'src="h/P-1.webp"', "h/P-1.webp"]


def test_compare_pictures_sized_to_their_column(tmp_path):
    """Compare's pictures (grid.js compareAttrs) offer the card view's 480 and 720 px copies, and their sizes is the
    column's width (style.css .compare: 32 px of padding, 14 px gaps, columns min(220px, half) at least, the panel's
    30 px), at every screen width and card count."""
    grid = SITE_SRC / "site" / "assets" / "grid.js"
    if "compareAttrs" not in grid.read_text(encoding="utf-8"):
        pytest.skip("a site before Compare's sized pictures")
    probe = tmp_path / "probe.mjs"
    probe.write_text(f"""const G = await import({json.dumps(grid.as_uri())});
process.stdout.write(JSON.stringify([1, 2, 3, 4, 5].map(n => G.compareAttrs({{ 360: "a", 480: "b", 720: "c" }}, {{ img: "x" }}, n))));
""", encoding="utf-8")
    done = subprocess.run(["node", str(probe)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    for n, attrs in enumerate(json.loads(done.stdout), 1):
        assert 'srcset="thumbs/x-360-a.avif 360w, thumbs/x-480-b.avif 480w, thumbs/x-720-c.avif 720w"' in attrs
        sizes = re.search(r'sizes="([^"]+)"', attrs)[1]
        for vw in (320, 390, 485, 486, 600, 820, 954, 1188, 1280, 1440, 1920, 2560):
            length = next(s.split(") ", 1)[1] if s.startswith("(") else s for s in sizes.split(", ")
                          if not s.startswith("(") or vw <= int(re.match(r"\(max-width: (\d+)px\)", s)[1]))
            got = eval(re.sub(r"(\d+)vw", lambda m: f"{vw * int(m[1]) / 100}", length).replace("calc", "").replace("px", ""))
            inner = vw - 32
            assert got == pytest.approx(max(min(220, (inner - 14) / 2), (inner - 14 * (n - 1)) / n) - 30), (n, vw, sizes)


def test_icons_34_68_96(bs, tmp_path, monkeypatch):
    """build_icons writes each icon at 96, 68 and 34 px; grid.js iconSet lists them. Picked as browsers do (first sizes
    entry whose condition holds, then the first width covering the screen): up to 2.4x an icon under 26 px takes the 26 px
    toggles' file at any width (1x, 1.25x: 34; 2x phones: 68), so a page fetches one file; above, as before."""
    from PIL import Image
    root = tmp_path / "repo"
    (root / "Types").mkdir(parents=True)
    Image.new("RGBA", (200, 160), (200, 40, 40, 255)).save(root / "Types" / "fire.png")
    for name, value in (("REPO", root), ("DIST", tmp_path / "dist"), ("CACHE", tmp_path / ".cache")):
        monkeypatch.setattr(bs, name, str(value))
    icons, _ = bs.build_icons({"Types/fire.png": "ab" * 20})
    file = tmp_path / "dist" / icons["fire"]["file"]
    assert [max(Image.open(str(file).replace(".webp", s)).size) for s in (".webp", ".68.webp", ".34.webp")] == [96, 68, 34]
    probe = tmp_path / "probe.mjs"
    probe.write_text(f"const G = await import({json.dumps((SITE_SRC / 'site/assets/grid.js').as_uri())});\n"
                     "process.stdout.write(JSON.stringify([16, 18, 22, 26, 34].map(px => G.iconSet({ file: 'i/f.webp' }, px))));\n",
                     encoding="utf-8")
    done = subprocess.run(["node", str(probe)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    sets = dict(zip((16, 18, 22, 26, 34), json.loads(done.stdout)))

    def pick(px, dpr):
        srcset, sizes = re.fullmatch(r' srcset="([^"]*)" sizes="([^"]*)"', sets[px]).groups()
        for entry in sizes.split(", "):
            cond, _, size = entry.rpartition(" ")
            limit = re.fullmatch(r"\((?:max-resolution: ([\d.]+)dppx|-webkit-max-device-pixel-ratio: ([\d.]+))\)", cond)
            if not cond or limit and dpr <= float(limit[1] or limit[2]):     # any other condition: false, as on a phone
                break
        files = [(int(w[:-1]) / float(size[:-2]), f) for f, w in (c.split(" ") for c in srcset.split(", "))]
        return next((f for d, f in files if d >= dpr), files[-1][1])

    for dpr in (1, 1.25, 1.5, 1.75, 2, 2.25, 2.4):
        assert {pick(px, dpr) for px in (16, 18, 22, 26)} == {pick(26, dpr)}, dpr
    assert [pick(26, d) for d in (1, 1.25, 1.5, 2, 2.625)] == ["i/f.34.webp"] * 2 + ["i/f.68.webp"] * 2 + ["i/f.webp"]
    assert [pick(px, 3) for px in (16, 18, 22, 26, 34)] == ["i/f.68.webp"] * 3 + ["i/f.webp"] * 2


def test_table_rows_take_the_small_copies_unless_drawn_wider(tmp_path):
    """Table rows (30x42, object-fit: cover; app.js imgAttrs(c, 30, 42)) list the 48 and 64 px copies for a card drawn 30
    px wide; a square or wide one is drawn wider, so it keeps 96 first, as before (grid.js in node)."""
    grid = SITE_SRC / "site" / "assets" / "grid.js"
    if "coverH" not in grid.read_text(encoding="utf-8"):
        pytest.skip("a site before the 48 and 64 px copies")
    probe = tmp_path / "probe.mjs"
    probe.write_text(f"""const G = await import({json.dumps(grid.as_uri())});
const t = {{ 48: "a", 64: "b", 96: "c", 192: "d", 240: "e", 360: "f" }}, w = s => /srcset="([^"]*)"/.exec(s)[1].match(/\\d+w/g).join(" ");
process.stdout.write(JSON.stringify([[360, 540], [360, 720], [360, 360], [360, 250], [0, 0]].map(([tw, th]) => w(G.imgAttrs(t, {{ img: "x", tw, th }}, 30, 42)))
  .concat([w(G.imgAttrs(t, {{ img: "x", tw: 360, th: 360 }}, 64))])));
""", encoding="utf-8")
    done = subprocess.run(["node", str(probe)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    small, old = "48w 64w 96w 192w 240w 360w", "96w 192w 240w 360w"
    assert json.loads(done.stdout) == [small, small, old, old, old, "64w 96w 192w 240w 360w"]
