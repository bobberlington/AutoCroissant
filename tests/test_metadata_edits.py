"""
Discord's card edits as card-repo commits (commands/discord_edits.py; docs/Campaigns/DATA_BUILD_ACTION.md section 11 and
section 14 step 5): the editsFile and authorsFile ports against edit.js itself (node), then /update_metadata,
/mass_replace and /alias on the test_card_bake fake GitHub: each one commit with the "Discord:" subject, an Edit-by
trailer and _by/_at, again from the new head when main moved, applied in memory at once, never a pickle. Then the alias
migration's mapping, aliases.pkl ignored once it ran (the bot's lookups, the duplicates' names), and on the live
aliases.pkl every alias finding the same card image as before.
"""
import json
import pickle
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest

pytest.importorskip("requests", reason="card_bake imports requests (absent from this Python)")
import commands.card_bake as cb  # noqa: E402
import commands.card_data as cd  # noqa: E402
import commands.card_models as cm  # noqa: E402
import commands.discord_edits as de  # noqa: E402
import commands.git_commit as gc  # noqa: E402
import commands.github_api as ga  # noqa: E402
import commands.query_card as qc  # noqa: E402
import commands.utils as utils  # noqa: E402
from commands.card_edits import ALIASES_MOVED, CardAuthors, CardEdits, dump_edits  # noqa: E402
from conftest import CLONE, REPO  # noqa: E402
from test_card_bake import ABOUT, EDIT_JS, FakeGitHub, _node  # noqa: E402

EDITS_PATH, AUTHORS_PATH = cb.EDITS_PATHS[0], cb.AUTHORS_PATHS[0]
AT = "2026-10-20T18:41:54Z"
BY = "discord:someone"
SNACK, SNACK2 = "Items/Heal/2 Stars/A_Snack.png", "Items/Heal/2 Stars/A_Snack2.png"
DOOMER, FREEZER = "Auxiliary/Minions/Mini_Doomer.png", "Field/1 Stars/The_Freezer.png"
GOOMBELLA, FIX = "Creatures/Mario/2 Stars/Goombella.png", "Items/Tools/Mr._Fix.png"
AUTHORS = """# Who made each card.
[people]
Michael = MichaelJSr
MichaelJSr = MichaelJSr
Zomoroko = Zomoroko

[cards]

# --- credited to MichaelJSr (2) ---
A Snack = MichaelJSr
Mini Doomer = MichaelJSr

# --- credited to Zomoroko (1) ---
The Freezer = Zomoroko

# --- rulebook pages ---
Rule One = MichaelJSr
"""
EDITS = {"Goombella": {"series": "Mario", "_by": "web", "_at": AT},
         "The Freezer": {"stars": 2, "_psd": {"state": "refused", "for": {"stars": 2}, "psd": "abcdef01", "reason": "x: y"},
                         "_by": "web", "_at": AT}}


# ========================
# Fixtures
# ========================
def _card(name, path, series=None, author=None, **kw):
    return cm.CardInfo(name=name, card_type=kw.pop("card_type", "creature"), path=path, ability=kw.pop("ability", "Does a thing."),
                       series=series, author=author, stats=cm.CardStats(5, 2, 5, 5), **kw)


@pytest.fixture
def repo(monkeypatch, tmp_path):
    """The fake card repo (Card Edits.json, Card Authors.txt, the PNGs) and the bot's view of it: stats, the registry."""
    files = {EDITS_PATH: dump_edits({"about": ABOUT, "cards": EDITS}).encode(), AUTHORS_PATH: AUTHORS.encode(),
             **{p: p.encode() for p in (SNACK, SNACK2, DOOMER, FREEZER, GOOMBELLA, FIX)}}
    fake = FakeGitHub(cb.CARD_REPO, files)
    monkeypatch.setattr(ga, "get", fake.get)
    monkeypatch.setattr(ga, "GIT_TOKEN", "")
    monkeypatch.setattr(ga, "HEADERS", {})
    monkeypatch.setattr(gc, "request", fake.request)
    monkeypatch.setattr(gc, "refresh", lambda: True)
    monkeypatch.setattr(gc, "cant_write", False)
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    stats = {"A Snack": _card("A Snack", SNACK[:-4] + ".psd", author="MichaelJSr", card_type="item"),
             "Mini Doomer": _card("Mini Doomer", DOOMER[:-4] + ".psd", author="MichaelJSr", card_type="minion"),
             "The Freezer": _card("The Freezer", FREEZER[:-4] + ".psd", "mario", author="Zomoroko", card_type="field"),
             "Goombella": _card("Goombella", GOOMBELLA[:-4] + ".psd", "mario", author="Michael")}
    stats["A Snack2"] = _card("A Snack2", SNACK[:-4] + ".psd", author="MichaelJSr", card_type="item", duplicate_of="A Snack",
                              display_name="Breakfast")
    db = cm.StatsDatabase()
    db.stats = stats
    monkeypatch.setattr(cm.stats_db, "stats", stats)
    monkeypatch.setattr(cm.stats_db, "old_stats", {})
    saves = []
    monkeypatch.setattr(cm.stats_db, "save", lambda *a: saves.append(1))
    registry = qc.CardRepository()
    registry.git_file_alias = {"breakfast": "a_snack2.png"}                      # aliases.pkl's: A Snack2's name
    registry.git_files = {p.rsplit("/", 1)[1].lower(): p for p in (SNACK, SNACK2, DOOMER, FREEZER, GOOMBELLA, FIX)}
    registry._edits = CardEdits.parse(files[EDITS_PATH].decode())
    monkeypatch.setattr(qc, "card_repo", registry)
    prepped = []
    monkeypatch.setattr(registry, "prep_dataframes", lambda: prepped.append(1))
    monkeypatch.setattr(registry, "save_aliases", lambda: pytest.fail("a Discord edit saved aliases.pkl"))
    sent = []
    monkeypatch.setattr(utils, "say", lambda inter, text, **kw: sent.append(text))
    return SimpleNamespace(github=fake, stats=stats, registry=registry, sent=sent, prepped=prepped, saves=saves,
                           who=SimpleNamespace(user=SimpleNamespace(name="someone", id=7)))


def _at_head(github, path):
    return github.files_at(github.head)[path].decode("utf-8")


def _commits(github, since):
    """Commit messages from head back to since (exclusive), newest first."""
    out, sha = [], github.head
    while sha != since:
        out.append(github.commits[sha]["message"])
        sha = github.commits[sha]["parents"][0]["sha"]
    return out


# ========================
# The edit.js ports
# ========================
def test_edits_file_matches_edit_js(tmp_path):
    """editsFile itself (node): set, unset, a new entry, an entry left empty, _psd kept, _by/_at replaced."""
    text = dump_edits({"about": ABOUT, "cards": {**EDITS, "Gone": {"note": "n", "_by": "web", "_at": AT}}})
    ops = [{"key": "goombella", "set": {"subseries": "Paper Mario", "hp": 6}, "unset": ["series"]},
           {"key": "The_Freezer", "set": {"ability": "Cold.", "types": ["ice"]}, "unset": []},
           {"key": "New Card", "set": {"hidden": True}, "unset": []},
           {"key": "Gone", "set": {}, "unset": ["note"]},
           {"key": "Mini Doomer", "set": {}, "unset": []}]
    ours = de.edits_file(text, ops, AT, BY)
    assert ours == _node(tmp_path, text, ops, AT, BY)
    cards = json.loads(ours)["cards"]
    assert cards["Goombella"] == {"subseries": "Paper Mario", "hp": 6, "_by": BY, "_at": AT} and "Gone" not in cards
    assert list(cards["The Freezer"]) == ["stars", "types", "ability", "_psd", "_by", "_at"]
    assert de.edits_file("", [{"key": "X", "set": {"hp": 1}, "unset": []}], AT, BY) == _node(tmp_path, "", [{"key": "X", "set": {"hp": 1}, "unset": []}], AT, BY)


def _authors_js(tmp_path, text, ops, people):
    """authorsFile from edit.js (it is not exported: its source, from section() to banFile, with cardKey)."""
    if not EDIT_JS.exists() or shutil.which("node") is None:
        pytest.skip("edit.js or node missing")
    src = EDIT_JS.read_text(encoding="utf-8")
    key = re.search(r"^const cardKey = .*;$", src, re.M)
    if not key or "  function section(lines, name) {" not in src or "  function banFile(" not in src:
        pytest.skip("edit.js's authorsFile moved")
    code = src[src.index("  function section(lines, name) {"):src.index("  function banFile(")]
    script = tmp_path / "authors_file.mjs"
    script.write_text(key.group(0) + "\n" + code + '\nimport { readFileSync } from "node:fs";\n'
                      'const [text, ops, people] = JSON.parse(readFileSync(0, "utf-8"));\n'
                      'process.stdout.write(authorsFile(text, ops, people));\n', encoding="utf-8")
    done = subprocess.run(["node", str(script)], input=json.dumps([text, ops, people]), capture_output=True, text=True,
                          encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_authors_file_matches_edit_js(tmp_path):
    """authorsFile itself (node): a card moved between groups, a new group before the rulebook pages, a line removed,
    a [people] line set and one added, a missing file."""
    cases = [
        ([{"key": "Mini_Doomer", "author": "Zomoroko"}, {"key": "Goombella", "author": "Chestnut"},
          {"key": "the freezer", "author": ""}, {"key": "A Snack", "author": "MichaelJSr"}], {"Zomoroko": "Zomo", "Chestnut": "Chestnut"}),
        ([{"key": "Abc", "author": "MichaelJSr"}, {"key": "Zed", "author": "MichaelJSr"}], {}),
    ]
    for ops, people in cases:
        assert de.authors_file(AUTHORS, ops, people) == _authors_js(tmp_path, AUTHORS, ops, people)
    assert de.authors_file(None, cases[0][0], {}) == _authors_js(tmp_path, None, cases[0][0], {})
    assert de.authors_file("[cards]\nX = A\n", [], {"A": "Bee"}) == _authors_js(tmp_path, "[cards]\nX = A\n", [], {"A": "Bee"})
    out = de.authors_file(AUTHORS, cases[0][0], cases[0][1])
    authors = CardAuthors.parse(out)
    assert authors.resolve("Mini Doomer", None) == "Zomo" and authors.resolve("Goombella", None) == "Chestnut"
    assert "The Freezer" not in out and "# --- credited to Chestnut (1) ---\nGoombella = Chestnut\n\n# --- rulebook pages ---" in out


def test_replace_author_rewrites_right_hand_sides_in_place(clone_authors):
    """Design section 11: the right-hand sides change where they stand; the group is renamed, a rulebook page stays a
    rulebook page; into new's own group when new has one; a person nobody lists: unchanged unless the bot knows them."""
    text, moved, people = de.replace_author(AUTHORS, "michaeljsr", "Michael S")
    assert moved == ["A Snack", "Mini Doomer", "Rule One"]
    assert people == {"Michael": "Michael S", "MichaelJSr": "Michael S"}            # old's own line rewritten: none added
    assert text == AUTHORS.replace("MichaelJSr\nZomoroko = Zomoroko", "Michael S\nZomoroko = Zomoroko").replace(
        "Michael = MichaelJSr", "Michael = Michael S").replace("credited to MichaelJSr (2)", "credited to Michael S (2)").replace(
        "A Snack = MichaelJSr", "A Snack = Michael S").replace("Mini Doomer = MichaelJSr", "Mini Doomer = Michael S").replace(
        "Rule One = MichaelJSr", "Rule One = Michael S")
    authors = CardAuthors.parse(text)
    assert {authors.resolve(n, "x") for n in ("A Snack", "Mini Doomer", "Rule One")} == {"Michael S"}
    assert authors.resolve("Unlisted", "MichaelJSr") == "Michael S" and authors.resolve("Unlisted", "Michael") == "Michael S"
    assert authors.resolve("The Freezer", None) == "Zomoroko"
    text, moved, people = de.replace_author(AUTHORS, "Zomoroko", "MichaelJSr")    # into MichaelJSr's group, in order
    assert moved == ["The Freezer"] and "credited to Zomoroko" not in text
    assert "# --- credited to MichaelJSr (3) ---\nA Snack = MichaelJSr\nMini Doomer = MichaelJSr\nThe Freezer = MichaelJSr\n\n# --- rulebook" in text
    text, moved, people = de.replace_author(AUTHORS, "Bob", "Robert")              # nobody listed: the git name maps
    assert moved == [] and people == {"Bob": "Robert"} and CardAuthors.parse(text).resolve("X", "Bob") == "Robert"
    assert de.replace_author(AUTHORS, "Bob", "Robert", known=False) == (AUTHORS, [], {})
    real = clone_authors()                                                       # the card repo's file: groups and counts kept
    groups = lambda t: re.findall(r"# --- (?:credited to (.+) \((\d+)\)|rulebook pages) ---", t)
    rules = lambda t: t[t.index("# --- rulebook pages ---"):].count(" = ")
    who = max(set(re.findall(r"credited to (.+) \(", real)) & {r.split(" = ", 1)[1] for r in real[real.index("# --- rulebook"):].splitlines() if " = " in r})
    text, moved, _ = de.replace_author(real, who, who + " Prime")
    cards = real[real.index("\n[cards]\n"):]
    assert rules(text) == rules(real) and len(moved) == cards.count(f" = {who}\n") + cards.endswith(f" = {who}")
    assert groups(text) == [(g + " Prime" if g == who else g, n) for g, n in groups(real)]


@pytest.fixture
def clone_authors(clone):
    def read():
        done = subprocess.run(["git", "-C", str(clone), "show", f"HEAD:{AUTHORS_PATH}"], capture_output=True)
        if done.returncode != 0:
            pytest.skip(f"no {AUTHORS_PATH} at the clone's HEAD")
        return done.stdout.decode("utf-8")
    return read


# ========================
# /update_metadata
# ========================
def test_update_metadata_makes_one_commit(repo):
    start = repo.github.head
    de.manual_metadata_entry(repo.who, "Mini Doomer", del_entry=True, author="Zomoroko", ability="Explodes.  [fire] then.",
                             subtype="Bomb", series="Doom", subseries="Mini", hp=6, defense=0, attack=7, speed=1,
                             card_type="item")
    (message,) = _commits(repo.github, start)
    subject, _, rest = message.partition("\n\n")
    assert subject == "Discord: /update_metadata Mini Doomer (ability, subtype, series, subseries, hp, def, atk, spd, type, hidden, author)"
    assert rest.endswith(f"\n\nEdit-by: {BY}") and "- hp: 6" in rest
    entry = json.loads(_at_head(repo.github, EDITS_PATH))["cards"]["Mini Doomer"]
    at = entry["_at"]
    assert entry == {"type": "item", "subtype": "Bomb", "series": "Doom", "subseries": "Mini",
                     "hp": 6, "def": 0, "atk": 7, "spd": 1, "ability": "Explodes.  [fire] then.", "hidden": True, "_by": BY, "_at": at}
    given = {k: v for k, v in entry.items() if not k.startswith("_")}
    before = dump_edits({"about": ABOUT, "cards": EDITS})
    assert _at_head(repo.github, EDITS_PATH) == de.edits_file(before, [{"key": "Mini Doomer", "set": given, "unset": []}], at, BY)
    assert _at_head(repo.github, AUTHORS_PATH) == de.authors_file(AUTHORS, [{"key": "Mini Doomer", "author": "Zomoroko"}], {})
    card = repo.stats["Mini Doomer"]                                             # applied in memory at once
    assert (card.stats.hp, card.stats.defense, card.ability, card.hidden, card.subseries, card.author) == (6, 0, "Explodes.  [fire] then.", True, "Mini", "Zomoroko")
    assert card.parsed_values["hp"] == 5 and repo.prepped and repo.saves == [1]         # stats.pkl kept for a restart
    assert "Updated **Mini Doomer**" in repo.sent[len(repo.sent) - 1] and "pending on the card image" in repo.sent[len(repo.sent) - 1]
    assert cb.select(CardEdits.parse(_at_head(repo.github, EDITS_PATH)), cb.Snapshot(cb.CARD_REPO, "h", {}, {}), "Mini Doomer")[0].fields["hp"] == 6


def test_update_metadata_retries_when_main_moved(repo):
    repo.github.conflicts = 1                                                    # someone pushes first
    start = repo.github.head
    de.manual_metadata_entry(repo.who, "goombella", subseries="Paper Mario")
    messages = _commits(repo.github, start)
    assert len(messages) == 2 and messages[1] == "Website edit: elsewhere" and messages[0].startswith("Discord: /update_metadata Goombella (subseries)")
    assert "elsewhere.txt" in repo.github.files_at(repo.github.head)
    assert json.loads(_at_head(repo.github, EDITS_PATH))["cards"]["Goombella"]["series"] == "Mario"   # the web edit kept
    assert repo.stats["Goombella"].subseries == "Paper Mario"


def test_update_metadata_png_only_card_and_bad_values(repo):
    start = repo.github.head
    de.manual_metadata_entry(repo.who, "Mr. Fix", hp=-1)
    assert repo.github.head == start and "must be a whole number" in repo.sent[0]
    de.manual_metadata_entry(repo.who, "Mr. Fix")
    assert "no data yet" in repo.sent[1]
    de.manual_metadata_entry(repo.who, "Mr. Fix", series="Tools")                # the PNG's name, dots and all
    assert json.loads(_at_head(repo.github, EDITS_PATH))["cards"]["Mr. Fix"]["series"] == "Tools"
    assert "Mr. Fix" not in repo.stats and "No data for this card yet" in repo.sent[2]
    de.manual_metadata_entry(repo.who, "Mr. Fix", series="Tools")                # the same again: nothing to commit
    assert len(_commits(repo.github, start)) == 1 and "nothing committed" in repo.sent[3]
    for bad in ({"stars": 3}, {"types": "fire"}, {"types": " , ", "hp": 2}):      # the bake can't print them: refused
        de.manual_metadata_entry(repo.who, "Mini Doomer", **bad)
        assert "can't be set from Discord yet" in repo.sent[len(repo.sent) - 1]
    assert len(_commits(repo.github, start)) == 1


def test_update_metadata_skips_values_the_card_repo_has(repo, monkeypatch):
    """A value the head's entry or [cards] line already has is no op: no restamp, no commit, the web editor's _by kept."""
    stamps = iter(f"2026-10-22T10:00:{s:02d}Z" for s in range(60))
    monkeypatch.setattr(de, "now", lambda: next(stamps))                        # every call a second later
    start = repo.github.head
    de.manual_metadata_entry(repo.who, "Mr. Fix", series="Tools")
    de.manual_metadata_entry(repo.who, "Mr. Fix", series="Tools")
    assert len(_commits(repo.github, start)) == 1 and "nothing committed" in repo.sent[1]
    de.manual_metadata_entry(repo.who, "Goombella", series="Mario")              # set by "web" already
    de.manual_metadata_entry(repo.who, "Mini Doomer", author="MichaelJSr")       # its [cards] line already
    assert len(_commits(repo.github, start)) == 1 and all("nothing committed" in t for t in repo.sent[2:4])
    assert json.loads(_at_head(repo.github, EDITS_PATH))["cards"]["Goombella"] == EDITS["Goombella"]
    de.manual_metadata_entry(repo.who, "Goombella", series="Mario", hp=6, author=" Zomo \n roko ")   # only hp and author
    subject, _, rest = _commits(repo.github, start)[0].partition("\n\n")
    assert subject == "Discord: /update_metadata Goombella (hp, author)" and "- series" not in rest
    assert "- author: Zomo roko" in rest and "Goombella = Zomo roko" in _at_head(repo.github, AUTHORS_PATH)
    assert json.loads(_at_head(repo.github, EDITS_PATH))["cards"]["Goombella"] == {"series": "Mario", "hp": 6, "_by": BY,
                                                                                  "_at": "2026-10-22T10:00:04Z"}
    assert "Already so: series." in repo.sent[4]


def test_update_metadata_skips_values_the_card_has(repo):
    """No entry and the card already so (edit.js same(v, pub)): no override, no [cards] line."""
    start = repo.github.head
    de.manual_metadata_entry(repo.who, "Mini Doomer", hp=5)                      # the PSD prints 5
    de.manual_metadata_entry(repo.who, "The Freezer", series="mario", card_type="field")
    de.manual_metadata_entry(repo.who, "Goombella", author="Michael")            # from git history, no [cards] line
    assert repo.github.head == start and all("nothing committed" in t for t in repo.sent)
    de.manual_metadata_entry(repo.who, "Mini Doomer", hp=5, attack=6)
    entry = json.loads(_at_head(repo.github, EDITS_PATH))["cards"]["Mini Doomer"]
    assert {k: v for k, v in entry.items() if not k.startswith("_")} == {"atk": 6} and "Already so: hp." in repo.sent[3]


def test_update_metadata_on_a_duplicate_and_failure(repo, monkeypatch):
    de.manual_metadata_entry(repo.who, "A Snack", attack=9)                      # the original: its duplicate copied again
    assert repo.stats["A Snack2"].stats.attack == 9 and repo.stats["A Snack2"].display_name == "Breakfast"
    assert repo.stats["A Snack2"].duplicate_of == "A Snack"
    assert "the card bake doesn't print it (numbered PNGs share" in repo.sent[0] and "next card bake" not in repo.sent[0]
    de.manual_metadata_entry(repo.who, "A Snack2", ability="Breakfast.")         # the duplicate's own entry
    assert repo.stats["A Snack2"].ability == "Breakfast." and repo.stats["A Snack"].ability == "Does a thing."
    assert "the card bake doesn't print it" in repo.sent[1]
    head = repo.github.head
    de.manual_metadata_entry(repo.who, "A Snack2", author="Zomoroko")            # the original's author, always
    assert repo.github.head == head and "Set the author on A Snack" in repo.sent[2]
    de.manual_metadata_entry(repo.who, "Goombella", ability=" \r\n  ")            # blank once trimmed: refused
    assert repo.github.head == head and "must be" in repo.sent[3]
    de.manual_metadata_entry(repo.who, "Goombella", ability="\r\n Two  spaces. \n")   # ends trimmed, the inside kept
    assert json.loads(_at_head(repo.github, EDITS_PATH))["cards"]["Goombella"]["ability"] == "Two  spaces."
    assert "pending on the card image until the next card bake" in repo.sent[4]
    repo.github.stuck = True
    head = repo.github.head
    de.manual_metadata_entry(repo.who, "Goombella", hp=1)
    assert repo.github.head == head and "not committed" in repo.sent[len(repo.sent) - 1] and repo.stats["Goombella"].stats.hp == 5


# ========================
# /mass_replace
# ========================
def test_mass_replace_series_one_commit(repo):
    start = repo.github.head
    repo.stats["A Snack2"].series = "Mario"                                      # a duplicate: its original's copy
    de.mass_replace_field(repo.who, "series", "MARIO", "Mario Bros")
    (message,) = _commits(repo.github, start)
    assert message.startswith("Discord: /mass_replace series 'MARIO' -> 'Mario Bros' (2 cards)\n\n- Goombella\n- The Freezer")
    cards = json.loads(_at_head(repo.github, EDITS_PATH))["cards"]
    assert cards["Goombella"]["series"] == cards["The Freezer"]["series"] == "Mario Bros" and "A Snack2" not in cards
    assert cards["The Freezer"]["_psd"] == EDITS["The Freezer"]["_psd"] and cards["Goombella"]["_by"] == BY
    assert (cards["The Freezer"]["_by"], cards["The Freezer"]["_at"]) == ("web", AT)   # its PSD field's editor kept
    assert repo.stats["Goombella"].series == "Mario Bros" and "Replaced series='MARIO'" in repo.sent[0]
    de.mass_replace_field(repo.who, "subtype", "nothing", "x")
    assert "No card has subtype" in repo.sent[1] and len(_commits(repo.github, start)) == 1


def test_mass_replace_counts_and_no_ops(repo):
    """The subject and the reply count the entries changed; old == new and a hand-formatted file make no commit."""
    start = repo.github.head
    repo.stats["Goombella"].series = "MARIO"                                     # its entry already says "Mario"
    repo.stats["Mini Doomer"].series = "Mario"                                   # already new (its folder's): no override
    de.mass_replace_field(repo.who, "series", "mario", "Mario")
    (message,) = _commits(repo.github, start)
    assert message.startswith("Discord: /mass_replace series 'mario' -> 'Mario' (1 card)\n\n- The Freezer\n\nEdit-by")
    assert "Mini Doomer" not in json.loads(_at_head(repo.github, EDITS_PATH))["cards"]
    assert "1 card set in Card Edits.json; " in repo.sent[0]
    repo.stats["Goombella"].series = "Mario"
    de.mass_replace_field(repo.who, "series", "MARIO", "Mario")                  # every one already so: no commit
    assert len(_commits(repo.github, start)) == 1 and "already has 'Mario'" in repo.sent[1]
    repo.stats["Goombella"].series = "MARIO"
    for field, value in (("series", "Mario"), ("author", "Zomoroko")):
        de.mass_replace_field(repo.who, field, value, f" {value} ")
        assert "are both" in repo.sent[len(repo.sent) - 1]
    files = repo.github.files_at(repo.github.head)                               # a hand-formatted Card Edits.json
    loose = json.dumps(json.loads(files[EDITS_PATH]), indent=4).encode()
    head = repo.github.commit({**files, EDITS_PATH: loose}, "web", "hand edit")
    de.mass_replace_field(repo.who, "series", "mario", "Mario")
    assert repo.github.head == head and "already has 'Mario'" in repo.sent[len(repo.sent) - 1]
    de.manage_alias(repo.who, "nothing", None)
    assert repo.github.head == head and "No alias exists" in repo.sent[len(repo.sent) - 1]


def test_card_type_is_a_known_kind(repo):
    """card_type from Discord: a CardType value in its own spelling, else refused with no commit."""
    start = repo.github.head
    de.mass_replace_field(repo.who, "card_type", "minion", "minnion")
    de.manual_metadata_entry(repo.who, "Goombella", card_type="beast")
    assert repo.github.head == start and all("card_type must be one of" in s for s in repo.sent)
    de.manual_metadata_entry(repo.who, "Goombella", card_type="Unknown")         # edit.js offers no "unknown"
    de.mass_replace_field(repo.who, "card_type", "creature", "unknown")
    assert repo.github.head == start and all("card_type must be one of" in s and "unknown" not in s for s in repo.sent)
    de.mass_replace_field(repo.who, "card_type", "Creature", "creature")         # old canonicalized too: the same kind
    assert repo.github.head == start and "are both 'creature'" in repo.sent[len(repo.sent) - 1]
    de.mass_replace_field(repo.who, "card_type", "MINION", " Creature ")
    assert json.loads(_at_head(repo.github, EDITS_PATH))["cards"]["Mini Doomer"]["type"] == "creature"
    assert "Goombella" not in json.loads(_at_head(repo.github, EDITS_PATH))["cards"] or \
        "type" not in json.loads(_at_head(repo.github, EDITS_PATH))["cards"]["Goombella"]
    assert _commits(repo.github, start)[0].startswith("Discord: /mass_replace card_type 'minion' -> 'creature' (1 card)")
    de.manual_metadata_entry(repo.who, "Goombella", card_type="mdw")
    assert json.loads(_at_head(repo.github, EDITS_PATH))["cards"]["Goombella"]["type"] == "MDW"
    assert len(_commits(repo.github, start)) == 2


def test_mass_replace_author_spellings(repo):
    """new in another case than a person the file names: that person's spelling, one group; a case respelling of old;
    no authors file and an unknown author: no file."""
    start = repo.github.head
    repo.stats["The Freezer"].author = "Zomoroko"
    de.mass_replace_field(repo.who, "author", "Zomoroko", "michaeljsr")
    text = _at_head(repo.github, AUTHORS_PATH)
    assert "michaeljsr" not in text and text.count("credited to") == 1
    assert "# --- credited to MichaelJSr (3) ---\nA Snack = MichaelJSr\nMini Doomer = MichaelJSr\nThe Freezer = MichaelJSr\n" in text
    assert _commits(repo.github, start)[0].startswith("Discord: /mass_replace author 'Zomoroko' -> 'MichaelJSr'")
    assert "now credit MichaelJSr" in repo.sent[0]
    de.mass_replace_field(repo.who, "author", "MichaelJSr", "Michaeljsr")        # old's own case changed: a rename
    assert "# --- credited to Michaeljsr (3) ---" in _at_head(repo.github, AUTHORS_PATH)
    de.mass_replace_field(repo.who, "author", "michaelJSR", "Michaeljsr")        # already so everywhere
    assert len(_commits(repo.github, start)) == 2 and "already credits 'Michaeljsr'" in repo.sent[2]
    assert de.replace_author(None, "Bob", "Robert", known=False) == (None, [], {})
    text, moved, people = de.replace_author(AUTHORS, "Bob", "zomoroko")          # new's spelling from [people]/[cards]
    assert people == {"Bob": "Zomoroko"}


def test_mass_replace_author_rewrites_the_authors_file(repo):
    start = repo.github.head
    de.mass_replace_field(repo.who, "author", "MichaelJSr", "Michael S")
    (message,) = _commits(repo.github, start)
    assert message.split("\n\n")[0] == "Discord: /mass_replace author 'MichaelJSr' -> 'Michael S'"
    assert "- A Snack\n- Mini Doomer\n- Rule One\n- [people] Michael = Michael S" in message
    assert _at_head(repo.github, AUTHORS_PATH) == de.replace_author(AUTHORS, "MichaelJSr", "Michael S")[0]
    assert _at_head(repo.github, EDITS_PATH) == dump_edits({"about": ABOUT, "cards": EDITS})   # untouched
    assert {repo.stats[n].author for n in ("A Snack", "Mini Doomer", "Goombella")} == {"Michael S"}   # Goombella: git "Michael"
    assert repo.stats["The Freezer"].author == "Zomoroko"
    de.mass_replace_field(repo.who, "author", "Nobody", "X")
    assert "No card has author 'Nobody'" in repo.sent[1] and len(_commits(repo.github, start)) == 1
    repo.stats["The Freezer"].author = "Somebody"                                # credited by the file alone
    de.mass_replace_field(repo.who, "author", "zomoroko", " Zomo \n Prime ")    # one line, whitespace collapsed
    assert len(_commits(repo.github, start)) == 2 and "Replaced author='zomoroko' with 'Zomo Prime'" in repo.sent[2]
    assert _commits(repo.github, start)[0].startswith("Discord: /mass_replace author 'zomoroko' -> 'Zomo Prime'\n\n- The Freezer\n")
    assert "# --- credited to Zomo Prime (1) ---\nThe Freezer = Zomo Prime\n" in _at_head(repo.github, AUTHORS_PATH)


# ========================
# /alias
# ========================
def test_alias_create_move_delete(repo):
    start = repo.github.head
    de.manage_alias(repo.who, "Late Breakfast", "a_snack2")
    cards = json.loads(_at_head(repo.github, EDITS_PATH))["cards"]
    assert cards["A Snack2"] == {"aliases": ["Late Breakfast"], "_by": BY, "_at": cards["A Snack2"]["_at"]}
    assert repo.registry.git_files["late_breakfast.png"] == SNACK2 and "Created alias: Late Breakfast -> A Snack2" in repo.sent[0]
    assert repo.stats["A Snack2"].display_name == "Late Breakfast"               # the duplicate's longest alias names it
    assert _commits(repo.github, start)[0] == f"Discord: /alias Late Breakfast -> A Snack2\n\nEdit-by: {BY}"
    de.manage_alias(repo.who, "late breakfast", "Goombella.png")                # moved: off A Snack2, onto Goombella, one commit
    cards = json.loads(_at_head(repo.github, EDITS_PATH))["cards"]
    assert "A Snack2" not in cards and cards["Goombella"]["aliases"] == ["late breakfast"] and cards["Goombella"]["series"] == "Mario"
    assert repo.registry.git_files["late_breakfast.png"] == GOOMBELLA
    assert repo.stats["A Snack2"].display_name == "Breakfast" and repo.stats["Goombella"].display_name is None
    de.manage_alias(repo.who, " late  breakfast", "goombella")                  # the same spelling: nothing to commit
    assert "already finds Goombella" in repo.sent[2] and len(_commits(repo.github, start)) == 2
    de.manage_alias(repo.who, "", None)
    assert "late breakfast       -> Goombella" in repo.sent[3]
    de.manage_alias(repo.who, "LATE BREAKFAST", None)
    assert "aliases" not in json.loads(_at_head(repo.github, EDITS_PATH))["cards"]["Goombella"]
    assert "late_breakfast.png" not in repo.registry.git_files and "Deleted alias: LATE BREAKFAST" in repo.sent[4]
    assert _commits(repo.github, start)[0].startswith("Discord: /alias LATE BREAKFAST removed (Goombella)")
    de.manage_alias(repo.who, "nothing", None)
    de.manage_alias(repo.who, "x", "no such card")
    de.manage_alias(repo.who, "mini_doomer", "goombella")                        # a card image's own name
    assert "No alias exists" in repo.sent[5] and "not created" in repo.sent[6] and "already the name" in repo.sent[7]
    assert len(_commits(repo.github, start)) == 3


def _reloaded(repo):
    """The registry a card-list reload builds from the committed Card Edits.json (populate_files' order)."""
    fresh = qc.CardRepository()
    fresh.git_files = {p.rsplit("/", 1)[1].lower(): p for p in (SNACK, SNACK2, DOOMER, FREEZER, GOOMBELLA, FIX)}
    edits = CardEdits.parse(_at_head(repo.github, EDITS_PATH))
    fresh._edits, fresh._edit_names = edits, ("x", edits.names())
    fresh.relink()
    return {k: v for k, v in fresh.git_files.items() if k != "breakfast.png"}   # A Snack2's title (fixture: no reload ran)


def test_alias_never_loses_its_key_on_a_reload(repo):
    """An alias whose key is another card's Card Edits.json name: refused (a reload gives the key to the card sorting
    first). The card's own name, or a printed title (added after Card Edits.json's): created, and a reload agrees."""
    files = repo.github.files_at(repo.github.head)
    data = json.loads(files[EDITS_PATH])
    data["cards"]["Goombella"] = {**data["cards"]["Goombella"], "name": "Bella"}
    head = repo.github.commit({**files, EDITS_PATH: dump_edits(data).encode()}, "web", "Website edit: Goombella")
    repo.registry._edits = CardEdits.parse(dump_edits(data))
    repo.registry._edit_names = ("x", repo.registry._edits.names())
    repo.stats["The Freezer"].display_name = "Cold Room"
    repo.registry.relink()
    de.manage_alias(repo.who, " BELLA ", "mini doomer")
    assert repo.github.head == head and "Card Edits.json name of another card (Goombella)" in repo.sent[0]
    assert repo.registry.git_files["bella.png"] == GOOMBELLA
    de.manage_alias(repo.who, "Bella", "goombella")                              # its own name: no other card wins
    de.manage_alias(repo.who, "Cold Room", "mini doomer")
    assert "Created alias: Cold Room -> Mini Doomer" in repo.sent[2] and len(_commits(repo.github, head)) == 2
    assert repo.registry.git_files["cold_room.png"] == DOOMER
    reloaded = _reloaded(repo)
    assert {k: v for k, v in repo.registry.git_files.items() if k != "breakfast.png"} == reloaded


def test_alias_delete_keeps_name_and_title_lookups(repo):
    """Deleting an alias whose key is also a Card Edits.json name or a printed title: that lookup finds its card again."""
    files = repo.github.files_at(repo.github.head)
    data = json.loads(files[EDITS_PATH])
    data["cards"]["Goombella"] = {**data["cards"]["Goombella"], "name": "Bella"}
    data["cards"]["Mini Doomer"] = {"aliases": ["Bella"], "_by": "web", "_at": AT}   # a website alias on Goombella's name
    repo.github.commit({**files, EDITS_PATH: dump_edits(data).encode()}, "web", "Website edit: Goombella")
    repo.registry._edits = CardEdits.parse(dump_edits(data))
    repo.registry._edit_names = ("x", repo.registry._edits.names())
    repo.stats["The Freezer"].display_name = "Cold Room"
    repo.registry.git_files["bella.png"] = DOOMER
    repo.registry._populate_titles()
    assert repo.registry.git_files["cold_room.png"] == FREEZER
    de.manage_alias(repo.who, "Cold Room", "mini doomer")
    assert repo.registry.git_files["cold_room.png"] == DOOMER
    for alias in ("Bella", "Cold Room"):
        de.manage_alias(repo.who, alias, None)
    assert "Deleted alias: Bella" in repo.sent[1] and "Deleted alias: Cold Room" in repo.sent[2]
    assert repo.registry.git_files["bella.png"] == GOOMBELLA and repo.registry.git_files["cold_room.png"] == FREEZER
    assert {k: v for k, v in repo.registry.git_files.items() if k != "breakfast.png"} == _reloaded(repo)


def test_alias_on_a_card_with_two_aliases(repo, monkeypatch):
    """An alias the card already has (not its last, any spelling) is no change: no reorder, no restamp, no commit. An entry
    holding a website PSD edit keeps its editor's stamps (the bake's Edit-by credit)."""
    text = '{"about":"x","cards":{"Foo":{"aliases":["a","b"],"_by":"web:x","_at":"2026"}}}'
    assert de._alias_ops(text, "a", "Foo") == [] and de._alias_ops(text, "b", "Foo") == []
    assert de._alias_ops(text, "A", "Foo") == [] and de._alias_ops(text, "a", "Bar")[0]["set"] == {"aliases": ["b"]}
    assert de._alias_ops(text, "c", "Foo") == [{"key": "Foo", "set": {"aliases": ["a", "b", "c"]}, "unset": []}]
    assert de._stamped(text, [], AT, BY) == text                                 # never reformatted
    files = repo.github.files_at(repo.github.head)
    data = json.loads(files[EDITS_PATH])
    data["cards"]["Goombella"] = {"hp": 6, "aliases": ["Goomba", "Bella"], "_by": "webber", "_at": AT}
    head = repo.github.commit({**files, EDITS_PATH: dump_edits(data).encode()}, "webber", "Website edit: Goombella")
    repo.registry._edits = CardEdits.parse(dump_edits(data))
    de.manage_alias(repo.who, "goomba", "goombella")
    assert repo.github.head == head and "'goomba' already finds Goombella" in repo.sent[0]
    de.manage_alias(repo.who, "Paper Goomba", "goombella")
    entry = json.loads(_at_head(repo.github, EDITS_PATH))["cards"]["Goombella"]
    credit = entry.pop("_credit")                                               # the aliases are the Discord user's
    assert entry == {"hp": 6, "aliases": ["Goomba", "Bella", "Paper Goomba"], "_bot": ["aliases"], "_by": "webber", "_at": AT}
    assert list(credit) == ["aliases"] and credit["aliases"]["_by"] == BY
    assert _commits(repo.github, head)[0].endswith(f"Edit-by: {BY}")            # the commit is the Discord user's
    card = cb.select(CardEdits.parse(_at_head(repo.github, EDITS_PATH)), cb.Snapshot(cb.CARD_REPO, "h", {}, {}), "Goombella")[0]
    assert card.by == "webber"
    de.manage_alias(repo.who, "Late", "mini doomer")                             # no PSD field: the Discord user's stamps
    assert json.loads(_at_head(repo.github, EDITS_PATH))["cards"]["Mini Doomer"]["_by"] == BY


def test_discord_fields_in_an_editors_entry_are_credited_apart():
    """A Discord edit to an entry holding another editor's PSD fields keeps their stamps (the bake's credit), lists its
    fields in _bot (not that editor's) and credits them in _credit; the changelog credits them to the Discord user."""
    import commands.changelog as cl
    later, last = "2026-10-23T00:00:00Z", "2026-10-24T00:00:00Z"
    text = dump_edits({"about": ABOUT, "cards": {"Goombella": {"ability": "Hit.", "_by": "web", "_at": AT},
                                                 "Mine": {"hp": 3, "_by": BY, "_at": AT}}})
    series = lambda key, value: [{"key": key, "set": {"series": value}, "unset": []}]
    cards = json.loads(de._stamped(text, series("goombella", "Kirby"), later, BY))["cards"]
    assert cards["Goombella"] == {"series": "Kirby", "ability": "Hit.", "_bot": ["series"],
                                  "_credit": {"series": {"_by": BY, "_at": later}}, "_by": "web", "_at": AT}
    assert cl.edit_credit(cards["Goombella"], {"series": [None, "Kirby"]}) == BY
    assert cl.edit_credit(cards["Goombella"], {"ability": ["a", "b"], "series": [None, "Kirby"]}) == "web"
    assert cards["Mine"] == {"hp": 3, "_by": BY, "_at": AT}                     # the stamp's own editor: nothing apart
    assert json.loads(de._stamped(text, series("Mine", "X"), later, BY))["cards"]["Mine"] == \
        {"series": "X", "hp": 3, "_by": BY, "_at": AT}
    text = dump_edits({"about": ABOUT, "cards": cards})
    other = json.loads(de._stamped(text, [{"key": "Goombella", "set": {"aliases": ["G"]}, "unset": []}], last, "discord:b"))
    assert other["cards"]["Goombella"]["_credit"] == {"series": {"_by": BY, "_at": later}, "aliases": {"_by": "discord:b", "_at": last}}
    psd = json.loads(de._stamped(text, [{"key": "Goombella", "set": {"hp": 4}, "unset": []}], last, "discord:b"))
    assert psd["cards"]["Goombella"] == {"series": "Kirby", "hp": 4, "ability": "Hit.", "_bot": ["series"],
                                         "_credit": {"series": {"_by": BY, "_at": later}}, "_by": "discord:b", "_at": last}
    web = de.edits_file(text, series("Goombella", "Mario"), last, "w2")       # edit.js: the field is w2's, its credit goes
    assert "_bot" not in json.loads(web)["cards"]["Goombella"] and "_credit" not in json.loads(web)["cards"]["Goombella"]
    tidy = json.loads(de._stamped(web, [{"key": "Goombella", "set": {"aliases": ["G"]}, "unset": []}], last, BY))
    assert tidy["cards"]["Goombella"] == {"series": "Mario", "ability": "Hit.", "aliases": ["G"], "_bot": ["aliases"],
                                          "_credit": {"aliases": {"_by": BY, "_at": last}}, "_by": "w2", "_at": last}
    gone = json.loads(de._stamped(text, [{"key": "Goombella", "set": {}, "unset": ["series"]}], last, BY))
    assert gone["cards"]["Goombella"] == {"ability": "Hit.", "_by": "web", "_at": AT}


def test_an_entry_left_with_only_its_credit_goes():
    """An entry whose last field goes leaves no stamps behind, its _credit included (edits_file, as edit.js, keeps a
    credited _bot field's entry)."""
    held = {"hp": 6, "aliases": ["X"], "_bot": ["aliases"], "_credit": {"aliases": {"_by": "discord:a", "_at": AT}},
            "_by": "web", "_at": AT}
    text = dump_edits({"about": ABOUT, "cards": {"G": held, "H": {"hp": 1, "_by": "web", "_at": AT}}})
    unset = lambda t, *names: json.loads(de._stamped(t, [{"key": "G", "set": {}, "unset": list(names)}], "T2", "discord:b"))
    assert "G" not in unset(text, "hp", "aliases")["cards"]
    web = de.edits_file(text, [{"key": "G", "set": {}, "unset": ["hp"]}], "T1", "web")     # the website drops the PSD field
    assert json.loads(web)["cards"]["G"] == {"aliases": ["X"], "_bot": ["aliases"], "_credit": held["_credit"], "_by": "web",
                                             "_at": "T1"}
    after = unset(web, "aliases")                                                # then /alias drops the last alias
    assert "G" not in after["cards"] and after["cards"]["H"] == {"hp": 1, "_by": "web", "_at": AT}
    psd = dump_edits({"about": ABOUT, "cards": {"G": {"hp": 6, "_psd": {"refused": "x"}, "_by": "web", "_at": AT}}})
    assert unset(psd, "hp")["cards"]["G"] == {"_psd": {"refused": "x"}, "_by": "discord:b", "_at": "T2"}    # as edit.js


def test_edits_file_prunes_credit_as_edit_js():
    """edits_file keeps a _credit field only while it is in _bot; an empty or malformed _credit goes (edit.js editsFile)."""
    cred = {"aliases": {"_by": "discord:a", "_at": AT}, "series": {"_by": "discord:b", "_at": AT}}
    held = {"hp": 6, "series": "S", "aliases": ["X"], "_bot": ["series", "aliases"], "_credit": cred, "_by": "web", "_at": AT}
    text = dump_edits({"about": ABOUT, "cards": {"G": held}})
    one = json.loads(de.edits_file(text, [{"key": "G", "set": {"series": "T"}, "unset": []}], "T1", "web"))["cards"]["G"]
    assert one["_bot"] == ["aliases"] and one["_credit"] == {"aliases": cred["aliases"]}
    both = json.loads(de.edits_file(text, [{"key": "G", "set": {"series": "T", "aliases": ["Y"]}, "unset": []}], "T1", "web"))
    assert "_bot" not in both["cards"]["G"] and "_credit" not in both["cards"]["G"]
    bad = dump_edits({"about": ABOUT, "cards": {"G": {**held, "_credit": ["x"]}}})
    assert "_credit" not in json.loads(de.edits_file(bad, [{"key": "G", "set": {"hp": 7}, "unset": []}], "T1", "web"))["cards"]["G"]


def test_alias_refuses_aliases_pkl_until_the_migration(repo, monkeypatch):
    monkeypatch.setattr(repo.registry, "git_file_alias", {"breakfast": "a_snack2.png"})
    head = repo.github.head
    de.manage_alias(repo.who, "Breakfast", "goombella")
    de.manage_alias(repo.who, "breakfast", None)
    assert repo.github.head == head and all("alias migration" in s for s in repo.sent)
    de.manage_alias(repo.who, None, None)
    assert "aliases.pkl (until the alias migration):\nbreakfast" in repo.sent[2]
    repo.registry._edits.aliases_moved = True                                    # migrated: the pickle no longer counts
    de.manage_alias(repo.who, "breakfast", "goombella")
    assert json.loads(_at_head(repo.github, EDITS_PATH))["cards"]["Goombella"]["aliases"] == ["breakfast"]


# ========================
# The alias migration
# ========================
LEGACY = {"breakfast": "a_snack2.png", "grape juice": "a_snack.png", "mini_doomer": "mini_doomer.png", "zoey": "z!!!!!.png",
          "doomy": "doomer.png", "freezy": "the_freezer.png"}
TREE = [SNACK, SNACK2, DOOMER, FREEZER, GOOMBELLA]


def test_migrate_aliases_mapping():
    text = dump_edits({"about": ABOUT, "cards": {**EDITS, "A Snack": {"aliases": ["Snack"], "_by": "web", "_at": AT}}})
    new, report = de.migrate_aliases(LEGACY, text, TREE, AT)
    data = json.loads(new)
    assert list(data) == ["about", ALIASES_MOVED, "cards"] and CardEdits.parse(new).aliases_moved
    cards = data["cards"]
    assert cards["A Snack2"] == {"aliases": ["breakfast"], "_by": "AutoCroissant", "_at": AT}
    assert cards["A Snack"] == {"aliases": ["Snack", "grape juice"], "_by": "web", "_at": AT}   # merged; its stamps kept
    assert cards["Mini Doomer"]["aliases"] == ["doomy", "mini doomer"]          # the end of a file name, as the bot finds it
    assert cards["The Freezer"]["aliases"] == ["freezy"] and cards["Goombella"] == EDITS["Goombella"]
    assert report[0] == "6 aliases: 5 moved onto 4 cards, 0 already on their card (the same alias, any spelling), 1 left out"
    assert "- zoey -> z!!!!!.png: no such PNG, left out" in report and f"- doomy -> Mini Doomer ({DOOMER})" in report
    assert de.migrate_aliases(LEGACY, new, TREE, AT)[0] == new                    # idempotent
    again = de.migrate_aliases({**LEGACY, "MINI DOOMER": "mini_doomer.png"}, text, TREE, AT)
    assert again[1][0].startswith("7 aliases: 5 moved onto 4 cards, 1 already on their card")   # a same-card spelling


def test_aliases_moved_kept_when_card_edits_fails_to_parse(monkeypatch):
    """A Card Edits.json that names aliases_moved but doesn't parse still keeps aliases.pkl off."""
    for text, moved in (('{"aliases_moved": "x", "cards": {,}}', True), ('{"cards": {,}}', False)):
        monkeypatch.setattr(qc, "blob_text", lambda repo, sha, **kw: text)
        registry = qc.CardRepository()
        registry.git_file_alias = {"stale": "a_snack.png"}
        registry._tree = {"tree": [{"path": EDITS_PATH, "sha": "bad"}]}
        registry.git_files = {p.rsplit("/", 1)[1].lower(): p for p in TREE}
        registry._populate_edit_names()
        registry._populate_aliases()
        assert registry.aliases_moved() is moved and ("stale.png" in registry.git_files) is not moved


def test_failed_card_edits_read_keeps_the_last_names(monkeypatch):
    """A newer Card Edits.json that can't be downloaded: the last good aliases still find their cards."""
    good = dump_edits({"about": ABOUT, ALIASES_MOVED: "x", "cards": {"A Snack": {"aliases": ["Snack"], "_by": "web", "_at": AT}}})

    def fetch(repo, sha, **kw):
        if sha == "bad":
            raise OSError("502")
        return good
    monkeypatch.setattr(qc, "blob_text", fetch)
    registry = qc.CardRepository()
    for sha in ("good", "bad"):
        registry.git_files = {p.rsplit("/", 1)[1].lower(): p for p in TREE}
        registry._tree = {"tree": [{"path": EDITS_PATH, "sha": sha}]}
        registry._populate_edit_names()
        assert registry.git_files.get("snack.png") == SNACK and registry.aliases_moved()


def test_aliases_moved_named_by_a_newer_unparsable_file(monkeypatch):
    """An older parsed file without aliases_moved in memory, then a newer one naming it that doesn't parse: aliases.pkl off;
    the next file that parses decides again."""
    texts = {"old": dump_edits({"about": ABOUT, "cards": {}}), "bad": '{"aliases_moved": "x", "cards": {,}}',
             "new": dump_edits({"about": ABOUT, "cards": {}})}
    monkeypatch.setattr(qc, "blob_text", lambda repo, sha, **kw: texts[sha])
    registry = qc.CardRepository()
    for sha, moved in (("old", False), ("bad", True), ("new", False)):
        registry._tree = {"tree": [{"path": EDITS_PATH, "sha": sha}]}
        registry._populate_edit_names()
        assert registry.aliases_moved() is moved


def test_refresh_uses_aliases_pkl_when_card_edits_was_never_read(repo):
    """No Card Edits.json parsed: a duplicate's name comes from aliases.pkl, as the build names it; the hint turns it off."""
    repo.registry._edits = None
    repo.stats["A Snack2"].display_name = None
    de.refresh(None, None, AUTHORS.replace("A Snack = MichaelJSr", "A Snack = Zomoroko"))     # A Snack changes: copied again
    assert (repo.stats["A Snack2"].author, repo.stats["A Snack2"].display_name) == ("Zomoroko", "Breakfast")
    repo.registry._moved_hint = True                                             # an unparsed file that names aliases_moved
    de.refresh(None, None, AUTHORS)
    assert (repo.stats["A Snack2"].author, repo.stats["A Snack2"].display_name) == ("MichaelJSr", "A Snack")


def test_refresh_waits_for_a_running_update(repo):
    """A Discord edit's refresh never saves stats.pkl inside /update_stats or a bake: it waits for bake_lock."""
    from threading import Thread
    from commands.card_runner import bake_lock
    done = []
    with bake_lock:
        worker = Thread(target=lambda: done.append(de.refresh(None, None, AUTHORS.replace("A Snack = MichaelJSr", "A Snack = Zomoroko"))))
        worker.start()
        worker.join(0.3)
        assert worker.is_alive() and not done and repo.stats["A Snack"].author == "MichaelJSr"
    worker.join(10)
    assert not worker.is_alive() and "A Snack" in done[0] and repo.stats["A Snack"].author == "Zomoroko"


def test_aliases_pkl_ignored_once_moved(monkeypatch):
    """The bot's registry and the duplicates' names read Card Edits.json alone once aliases_moved is set."""
    monkeypatch.setattr(qc, "blob_text", lambda repo, sha, **kw: TEXTS[sha])
    moved, _ = de.migrate_aliases({"breakfast": "a_snack2.png"}, dump_edits({"about": ABOUT, "cards": {}}), TREE, AT)
    TEXTS = {"before": dump_edits({"about": ABOUT, "cards": {}}), "after": moved}
    for sha, found in (("before", SNACK2), ("after", SNACK2)):
        registry = qc.CardRepository()
        registry.git_file_alias = {"breakfast": "a_snack2.png", "stale": "a_snack.png"}
        registry._tree = {"tree": [{"path": EDITS_PATH, "sha": sha}] + [{"path": p, "sha": p} for p in TREE]}
        registry.git_files = {p.rsplit("/", 1)[1].lower(): p for p in TREE}
        registry._populate_edit_names()
        registry._populate_aliases()
        assert registry.git_files["breakfast.png"] == found
        assert ("stale.png" in registry.git_files) == (sha == "before")         # aliases.pkl only before the migration
    tree = SimpleNamespace(json=lambda: {"tree": [{"path": SNACK[:-4] + ".psd"}, {"path": SNACK2}]})
    for edits, name in ((None, "Late Supper"), (dump_edits({"about": ABOUT, "cards": {}}), "Late Supper"), (moved, "Breakfast")):
        db = cm.StatsDatabase()
        db.stats = {"A Snack": _card("A Snack", SNACK[:-4] + ".psd")}
        build = cd.CardBuild(db, aliases={"late supper": "a_snack2.png"})      # aliases.pkl's
        build.edits = CardEdits.parse(edits) if edits else None
        assert build._sync_duplicates(tree) == 1 and db.stats["A Snack2"].display_name == name


OMORI = "Creatures/Omori/2 Stars/Omori_Kid.psd"


def _carry_stats():
    stats = {"Omori Kid": _card("Omori Kid", OMORI, "OMORI", author="MichaelJSr"),               # the folder: "omori"
             "Goombella": _card("Goombella", GOOMBELLA[:-4] + ".psd", "Paper Mario", author="Michael"),   # Card Edits wins
             "Plain": _card("Plain", "Creatures/Kirby/1 Stars/Plain.psd", "kirby", author="Zomoroko"),
             "A Snack": _card("A Snack", SNACK[:-4] + ".psd", "The Binding of Isaac", author="MichaelJSr", card_type="item"),
             "Lonely": _card("Lonely", "Items/Tools/1 Stars/Lonely.psd", None, author="Chestnut", subseries="Odd", card_type="item"),
             "A Snack2": _card("A Snack2", SNACK[:-4] + ".psd", "Copy", duplicate_of="A Snack", card_type="item")}
    stats["Plain"].series, stats["Plain"].parsed_values = "Edited", {"series": "kirby"}    # an edit on top: the PSD's value counts
    return stats


def test_carried_values_are_the_discord_only_ones():
    """What a re-parse keeps from the database (series, subseries, author) and no card-repo input gives."""
    first = {OMORI: "MichaelJSr", "Creatures/Kirby/1 Stars/Plain.psd": "Zomoroko", "Items/Tools/1 Stars/Lonely.psd": "Tyler",
             GOOMBELLA[:-4] + ".psd": "MichaelJSr"}                              # "Michael" shows as MichaelJSr: the same
    text = dump_edits({"about": ABOUT, "cards": EDITS})
    ops, credits, report = de.carried(_carry_stats(), text, AUTHORS, first)
    assert ops == [{"key": "A Snack", "set": {"series": "The Binding of Isaac"}, "unset": []},
                   {"key": "Lonely", "set": {"subseries": "Odd"}, "unset": []},
                   {"key": "Omori Kid", "set": {"series": "OMORI"}, "unset": []}]
    assert credits == [{"key": "Lonely", "author": "Chestnut"}]                 # A Snack's [cards] line already says so
    assert report[0] == "3 Discord-only values on 3 cards into Card Edits.json, 1 authors into Card Authors.txt"
    edits, authors, lines = de.migrate({"breakfast": "a_snack2.png"}, _carry_stats(), text, AUTHORS, TREE, first, AT)
    cards = json.loads(edits)["cards"]                                          # one commit's files: aliases and values
    assert CardEdits.parse(edits).aliases_moved and cards["A Snack2"]["aliases"] == ["breakfast"]
    assert cards["Omori Kid"] == {"series": "OMORI", "_by": "AutoCroissant", "_at": AT} and cards["Goombella"] == EDITS["Goombella"]
    assert cards["The Freezer"] == EDITS["The Freezer"] and CardAuthors.parse(authors).resolve("Lonely", "Tyler") == "Chestnut"
    assert lines[0].startswith("1 aliases: 1 moved") and lines[2].startswith("3 Discord-only values")


def test_migration_lists_the_fields_it_adds_to_an_editors_entry(tmp_path):
    """An entry another editor stamped keeps its stamps, and the fields the migration adds to it go in _bot (credited to
    nobody); the bot's own entries and fields merged into the editor's own get none; a later edit takes its field out."""
    psd = EDITS["The Freezer"]["_psd"]
    text = dump_edits({"about": ABOUT, "cards": {
        "Dark matter": {"name": "Dark Matter", "_by": "MichaelJSr", "_at": "2026-10-22T14:35:30Z"},
        "Sonic": {"allow_high_stats": True, "_by": "AutoCroissant", "_at": "2026-10-21T00:00:00Z"},
        "A Snack": {"aliases": ["Snack"], "_by": "web", "_at": AT}, "The Freezer": EDITS["The Freezer"]}})
    ops = [{"key": "Dark matter", "set": {"series": "Kirby"}, "unset": []},
           {"key": "Sonic", "set": {"series": "Sonic the Hedgehog"}, "unset": []},
           {"key": "A Snack", "set": {"aliases": ["Snack", "grape juice"]}, "unset": []},
           {"key": "The Freezer", "set": {"series": "Ice", "subseries": "Cold"}, "unset": []},
           {"key": "New", "set": {"series": "X"}, "unset": []}]
    later = "2026-10-23T00:00:00Z"
    cards = de._bot_edits(text, ops, later)["cards"]
    assert cards["Dark matter"] == {"name": "Dark Matter", "series": "Kirby", "_bot": ["series"], "_by": "MichaelJSr",
                                    "_at": "2026-10-22T14:35:30Z"}
    assert cards["Sonic"] == {"series": "Sonic the Hedgehog", "allow_high_stats": True, "_by": "AutoCroissant",
                              "_at": "2026-10-21T00:00:00Z"}
    assert cards["A Snack"] == {"aliases": ["Snack", "grape juice"], "_by": "web", "_at": AT}
    assert list(cards["The Freezer"].items()) == [("series", "Ice"), ("subseries", "Cold"), ("stars", 2), ("_psd", psd),
                                                  ("_bot", ["series", "subseries"]), ("_by", "web"), ("_at", AT)]
    assert cards["New"] == {"series": "X", "_by": "AutoCroissant", "_at": later}
    moved = dump_edits({"about": ABOUT, "cards": cards})
    assert dump_edits(de._bot_edits(moved, ops, later)) == moved                    # again: the same file
    edit = lambda ops: json.loads(de.edits_file(moved, ops, later, BY))["cards"]
    assert edit([{"key": "The Freezer", "set": {"series": "Snow"}, "unset": []}])["The Freezer"]["_bot"] == ["subseries"]
    assert "_bot" not in edit([{"key": "Dark matter", "set": {}, "unset": ["series"]}])["Dark matter"]
    assert edit([{"key": "Dark matter", "set": {"name": "Dark Matter Blade"}, "unset": []}])["Dark matter"]["_bot"] == ["series"]
    if EDIT_JS.exists() and "_bot" in EDIT_JS.read_text(encoding="utf-8"):         # edit.js drops them the same way
        for op in ({"key": "The Freezer", "set": {"series": "Snow"}, "unset": []}, {"key": "Dark matter", "set": {}, "unset": ["series"]}):
            assert de.edits_file(moved, [op], later, BY) == _node(tmp_path, moved, [op], later, BY)


def test_an_edit_to_a_bot_entry_lists_its_fields_in_bot(tmp_path):
    """An editor's edit to a _by AutoCroissant entry lists the bot's untouched fields in _bot: the migrated series and
    aliases stay nobody's edit (the editor gets only what they change); the bot's own edit adds none."""
    text = dump_edits({"about": ABOUT, "cards": {
        "Chef Saltbaker": {"series": "Cuphead", "aliases": ["saltbaker"], "_by": "AutoCroissant", "_at": AT}}})
    edit = lambda op, by=BY: json.loads(de.edits_file(text, [op], "2026-10-23T00:00:00Z", by))["cards"]["Chef Saltbaker"]
    hp = {"key": "chef saltbaker", "set": {"hp": 6}, "unset": []}
    assert edit(hp) == {"series": "Cuphead", "hp": 6, "aliases": ["saltbaker"], "_bot": ["series", "aliases"], "_by": BY,
                        "_at": "2026-10-23T00:00:00Z"}
    assert "_bot" not in edit({"key": "Chef Saltbaker", "set": {"series": "Cuphead DLC"}, "unset": ["aliases"]})
    assert edit({"key": "Chef Saltbaker", "set": {}, "unset": ["aliases"]})["_bot"] == ["series"]
    assert "_bot" not in edit(hp, "AutoCroissant")
    if EDIT_JS.exists() and "isBot" in EDIT_JS.read_text(encoding="utf-8"):        # the website's editsFile the same
        for op in (hp, {"key": "Chef Saltbaker", "set": {"aliases": ["salt"]}, "unset": []}):
            assert de.edits_file(text, [op], AT, "MichaelJSr") == _node(tmp_path, text, [op], AT, "MichaelJSr")


def test_migration_cli(tmp_path, capsys):
    clone = tmp_path / "cards"
    files = {EDITS_PATH: dump_edits({"about": ABOUT, "cards": EDITS}).encode(), AUTHORS_PATH: AUTHORS.encode(),
             OMORI: b"psd", **{p: b"png" for p in TREE}}
    for path, data in files.items():
        (clone / path).parent.mkdir(parents=True, exist_ok=True)
        (clone / path).write_bytes(data)
    git = ["git", "-C", str(clone), "-c", "user.name=MichaelJSr", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(clone)], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-qm", "cards"], check=True)
    (tmp_path / "aliases.pkl").write_bytes(pickle.dumps(LEGACY))
    (tmp_path / "stats.pkl").write_bytes(pickle.dumps({"Omori Kid": _card("Omori Kid", OMORI, "OMORI", author="Someone")}))
    out = tmp_path / "out"
    assert de.main(["migrate", "--clone", str(clone), "--aliases", str(tmp_path / "aliases.pkl"),
                    "--stats", str(tmp_path / "stats.pkl"), "--out-dir", str(out)]) == 0
    cards = json.loads((out / "Card Edits.json").read_text(encoding="utf-8"))["cards"]
    assert cards["A Snack2"]["aliases"] == ["breakfast"] and cards["Omori Kid"]["series"] == "OMORI"
    assert "Omori Kid = Someone" in (out / "Card Authors.txt").read_text(encoding="utf-8")   # not the first commit's author
    assert capsys.readouterr().out.rstrip().endswith("git commit --author 'AutoCroissant <autocroissant@gmail.com>'")
    with pytest.raises(SystemExit):                                             # no alias-only migration
        de.main(["migrate-aliases", "--clone", str(clone), "--out", str(out / "x.json")])


def test_live_aliases_find_the_same_card_image_after_the_migration(clone):
    """Every alias of the live aliases.pkl finds the same PNG through Card Edits.json as through the pickle, and the
    duplicates keep their names (read-only: the clone's HEAD tree, its Card Edits.json)."""
    path = REPO / "aliases.pkl"
    if not path.exists():
        pytest.skip("no aliases.pkl")
    with open(path, "rb") as fh:
        legacy = pickle.load(fh)
    listed = subprocess.run(["git", "-C", str(clone), "ls-tree", "-r", "-z", "--name-only", "HEAD"], capture_output=True, check=True)
    pngs = [p for p in listed.stdout.decode("utf-8").split("\0") if p.endswith(".png")]
    text = (clone / EDITS_PATH).read_text(encoding="utf-8")
    if json.loads(text).get("aliases_moved"):
        pytest.skip("the clone's Card Edits.json is already migrated")
    moved, report = de.migrate_aliases(legacy, text, pngs, AT)

    def registry(aliases, edits_text):
        repo = qc.CardRepository()
        for p in pngs:
            repo._add_file_to_registry(p)
        repo.git_file_alias = {k.lower(): v.lower() for k, v in aliases.items()}
        repo._edits = CardEdits.parse(edits_text)
        repo._edit_names = ("x", repo._edits.names())
        repo._tree = {"tree": [{"path": EDITS_PATH, "sha": "x"}]}
        repo._populate_edit_names()
        repo._populate_aliases()
        return repo.git_files
    before, after = registry(legacy, text), registry(legacy, moved)
    lines = {a: next(line for line in report if line.startswith(f"- {a} -> ")) for a in legacy}
    kept = [a for a in legacy if "left out" not in lines[a]]
    assert len(kept) >= len(legacy) - 3, report                                 # a PNG gone since (ZOEY!!!!!'s) at most
    assert {a: after.get(f"{de.alias_key(a.replace('_', ' '))}.png") for a in kept} == {a: before[f"{a.lower()}.png"] for a in kept}
    edits = CardEdits.parse(moved)
    src = cm.CardInfo(name="x", card_type="item")
    for target in {str(t).lower() for a, t in legacy.items() if a in kept}:     # the duplicates' names
        own = edits.cards.get(CardAuthors.key(cm.card_name(target)), {}).get("aliases") or []
        assert cd.CardBuild._duplicate_name(target, src, 2, [2], {}, own) == cd.CardBuild._duplicate_name(target, src, 2, [2], legacy)
