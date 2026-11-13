"""
The website build (TTSCardMaker site/build/build_site.py): one smoke test per piece (owner 2026-11-12) and the checks of
the bot against the site. The build on a small card repo made here, offline, from the card sync's db/ (CARD_DATA_DIR,
load_card_data): the current text dated from its start; the bundle's lazy chunks; website-only fields (edit.js and
CardEdits); the parser notes an edit answers; a real card's inks (psd_details); R14's names (card_standards); web-only
rulebook chapters (rulebook.js). Loads build_site.py from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips
without it.
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
    - Watermelon with its duplicate Watermelon2 (its own PNG), one earlier text;
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
             "Watermelon": _card("Watermelon", f"{MELON}.psd", "Heal 2.", t2, blob_sha="m2", card_type="item"),
             "Watermelon2": _card("Watermelon2", f"{MELON}.psd", "Heal 2.", t2, blob_sha="m2", card_type="item",
                                  duplicate_of="Watermelon")}
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


def test_site_push_with_card_data_left_to_the_early_build(tmp_path):
    """site.yml's Early build: a push that also changed a Card sync data input (card-sync.yml's push paths but
    TTS/Global.lua, Global.xml and the save; the payload's lists, else the compare API) stops once Card sync's unfinished
    early build for it is found (every later step skipped, the summary names it); site code only: no call, it builds;
    no such build in 30 s: it builds."""
    yaml = pytest.importorskip("yaml")
    workflows = SITE_SRC / ".github" / "workflows"
    if not (workflows / "site.yml").is_file():
        pytest.skip(f"no {workflows / 'site.yml'}")
    steps = yaml.safe_load((workflows / "site.yml").read_text(encoding="utf-8"))["jobs"]["build"]["steps"]
    names = [s.get("name") or s.get("id") or s.get("uses") for s in steps]
    early, guard = steps[0], "steps.early.outputs.run == ''"
    assert early["id"] == "early" and early["if"] == "github.event_name == 'push' && github.run_attempt == '1'" and early["continue-on-error"] is True
    assert not [n for n, s in zip(names[1:], steps[1:]) if "if" not in s]           # none runs unasked
    assert all(steps[names.index(n) + d]["if"].startswith(guard + " && ") for n, d in (("python", 0), ("build", -1)))
    paths = yaml.safe_load((workflows / "card-sync.yml").read_text(encoding="utf-8"))[True]["push"]["paths"]
    no_data = {"TTS/Global.lua", "TTS/Global.xml", "TTS/Impossibility Simulator.json"}
    assert no_data <= set(paths)
    glob = lambda p: re.compile("".join(".*" if t == "**" else "[^/]*" if t == "*" else t if t[0] == "[" else re.escape(t)  # noqa: E731
                                        for t in re.findall(r"\*\*|\*|\[[^]]*\]|.", p)) + r"\Z")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    gh = ('echo "$*" >> "$RUNNER_TEMP/calls"\ncase "$*" in *compare/b4...ab*) j=$COMPARE;; *card-sync.yml/runs?event=push\\&head_sha=ab*) j=$SYNC;;'
          ' *site.yml/runs?event=workflow_dispatch*) j=$RUNS;; *) exit 1;; esac\nwhile [ $# -gt 1 ]; do [ "$1" != --jq ] || f=$2; shift; done\n'
          'jq -r "$f" "$j"')
    for name, body in {"gh": gh, "timeout": 'shift; exec "$@"', "sleep": "exec /bin/sleep 0.2"}.items():
        (bin_ / name).write_text(f"#!/bin/sh\n{body}\n")
        (bin_ / name).chmod(0o755)
    def data(name, obj):
        (tmp_path / name).write_text(json.dumps(obj))
        return str(tmp_path / name)
    runs = [{"id": 87, "display_title": "Card browser (after 77 after)", "status": "pending"},
            {"id": 88, "display_title": "Card browser (after 77 data)", "status": "pending"}]
    env = {"PATH": f"{bin_}:/usr/bin:/bin", "GITHUB_REPOSITORY": "o/r", "GITHUB_SHA": "ab" * 20, "GITHUB_SERVER_URL": "https://x",
           "SYNC": data("sync.json", {"workflow_runs": [{"id": 77}]}), "RUNS": data("runs.json", {"workflow_runs": runs}),
           "COMPARE": data("compare.json", {"files": [{"filename": "site/app.js"}, {"filename": "Types/A.png", "previous_filename": "Types/B.png"}]})}

    def step(n, event, swap=("", "")):
        temp = tmp_path / f"t{n}"
        temp.mkdir()
        done = subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", early["run"].replace(*swap)], capture_output=True,
                              text=True, timeout=60, env={**env, "RUNNER_TEMP": str(temp), "GITHUB_EVENT_PATH": data(f"e{n}.json", event),
                                                          "GITHUB_OUTPUT": str(temp / "out"), "GITHUB_STEP_SUMMARY": str(temp / "summary")})
        assert done.returncode == 0, done.stderr
        read = lambda f: (temp / f).read_text() if (temp / f).exists() else ""  # noqa: E731
        return read("out"), read("calls").count("api repos/"), read("summary"), done.stdout
    files = ["Creatures/A/1 Stars/X.psd", "Items/Y.PNG", "Types/Fire.webp", "MDW/Linked/a/b.psb", "MDW/Fonts/Futura.ttf",
             "MDW/Fonts/stoogedings-src/stoogedings.json", "MDW/Fonts/stoogedings-src/build_stoogedings.py", "site/Card Edits.json",
             "site/Card Authors.txt", "site/Card Creation Dates.json", "site/Card Ratings.json", "site/rulebook/1 Basics.md",
             "site/assets/app.js", "TTS/decks.json", ".mailmap", *sorted(no_data)]
    for n, f in enumerate(files):                                     # card-sync.yml's paths but the three
        out, calls, summary, _ = step(n, {"before": "b4", "commits": [{"added": [], "modified": [f], "removed": []}]})
        want = f not in no_data and any(glob(p).match(f) for p in paths)
        assert (out, calls, "actions/runs/88" in summary) == (("run=88\n", 2, True) if want else ("", 0, False)), f
    two = [{"added": ["site/x.js"], "modified": [], "removed": []}, {"added": [], "modified": ["TTS/Global.lua"], "removed": ["Items/Z.psd"]}]
    assert step("both", {"before": "b4", "commits": two})[:2] == ("run=88\n", 2)
    assert step("lists", {"before": "b4", "commits": []})[:2] == ("run=88\n", 3)  # the compare API's
    runs[1]["status"] = "completed"                                   # ended (or not dispatched): this run builds
    data("runs.json", {"workflow_runs": runs})
    out, calls, summary, said = step("none", {"before": "b4", "commits": two}, ('"$SECONDS" -lt 28 ]', "$(( looks += 1 )) -lt 3 ]"))
    assert (out, summary, calls) == ("", "", 4) and "No early build of Card sync's run 77 in 30 s" in said   # 3 looks, run 77 read once
