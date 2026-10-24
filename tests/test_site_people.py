"""
The website's person pages (TTSCardMaker site/build/build_site.py person_extras, site/assets/profile.js in node): image
revisions after the first upload, a bake as its editors' revision, untouched and most revised cards, the strength profile,
buffs and nerfs and who made them, ratings of their cards and as a rater (hidden without per-person ratings), edits by
field, archived cards; the style, rank, career, series and most-referenced helpers and the panels. Loads the site from
$AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without person_extras / styleOf, or node.
"""
import importlib.util
import json
import shutil
import statistics
from os import environ
from os.path import expanduser
from pathlib import Path
from subprocess import run as run_process

import pytest

from conftest import CLONE

SITE = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE))) / "site"
BUILD_SITE = SITE / "build" / "build_site.py"
PROFILE_JS = SITE / "assets" / "profile.js"
T = 1_780_000_000


@pytest.fixture(scope="module")
def bs():
    if not BUILD_SITE.is_file():
        pytest.skip(f"no card repo build_site at {BUILD_SITE}")
    spec = importlib.util.spec_from_file_location("site_build_site_people", BUILD_SITE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "person_extras"):
        pytest.skip(f"{BUILD_SITE} has no person stats")
    return module


def node(tmp_path, body: str):
    """Run body after importing profile.js as pf; its stdout as JSON."""
    if not (PROFILE_JS.is_file() and shutil.which("node") and "styleOf" in PROFILE_JS.read_text(encoding="utf-8")):
        pytest.skip(f"no profile.js with styleOf at {PROFILE_JS}, or no node")
    script = tmp_path / "probe.mjs"
    script.write_text(f"const pf = await import({json.dumps(PROFILE_JS.as_uri())});\n"
                      "const out = x => process.stdout.write(JSON.stringify(x));\n"
                      "const text = html => html.replace(/<[^>]+>/g, ' ').replace(/\\s+/g, ' ').trim();\n"
                      f"{body}\n", encoding="utf-8")
    return json.loads(run_process(["node", str(script)], capture_output=True, text=True, check=True).stdout)


def card(cid, author, **kw):
    return {"id": cid, "key": cid, "name": cid.title(), "author": author, **kw}


def commit(sha, ts, name, paths=(), editors=()):
    """A parse_history row: (sha, committer time, author, email, subject, entries, Edit-by values)."""
    return (sha, ts, name, f"{name.lower()}@example.com", "s", [("M", "1" * 40, "2" * 40, p) for p in paths], list(editors))


def rows_of(bs, cards, commits=(), people=None, **kw):
    out = bs.contributions(cards, people or {}, [], {}, list(commits), **kw)
    return {r["key"]: r for r in out["people"]}, out


def test_revisions_leave_out_the_first_upload_and_mass_passes(bs):
    """A's card: uploaded by A, changed by B, then by A; a commit in more than MASS_PASS cards' versions is no revision.
    A: self 1, others 1, top [[B, 1]]; B revised A's card once."""
    v = lambda sha: ((sha, 0, "x", "s"), "p")
    found = {"a.png": [v("mass"), v("byA"), v("byB"), v("up")], **{f"m{i}.png": [v("mass"), v("up2")] for i in range(bs.MASS_PASS + 1)}}
    commits = [commit("mass", 4, "A"), commit("byA", 3, "A"), commit("byB", 2, "B"), commit("up", 1, "A"), commit("up2", 1, "A")]
    cards = [card("alpha", "A", png="a.png"), card("beta", "B")]
    revs = bs.card_revisions(found, {"a.png": [cards[0]]}, commits, {}, {})
    rows, _ = rows_of(bs, cards, revisions=revs)
    a, b = rows["A"]["revisions"], rows["B"]["revisions"]
    assert (a["self"], a["others"], a["top"], a["untouched"]) == (1, 1, [["B", 1]], 0)
    assert b["theirs"] == [["A", 1]] and b["untouched"] == 1


def test_a_bake_is_its_editors_revision(bs):
    """A BOT_AUTHOR commit with Edit-by editor1 counts for editor1's row (through [people]); one without trailers, nobody's."""
    v = lambda sha: ((sha, 0, "x", "s"), "p")
    found = {"a.png": [v("bake"), v("bare"), v("up")]}
    commits = [commit("bake", 3, bs.BOT_AUTHOR, editors=["editor1"]), commit("bare", 2, bs.BOT_AUTHOR), commit("up", 1, "A")]
    cards = [card("alpha", "A", png="a.png"), card("ed", "Ed")]
    revs = bs.card_revisions(found, {"a.png": [cards[0]]}, commits, {"editor1": "Ed"}, {})
    rows, _ = rows_of(bs, cards, people={"editor1": "Ed"}, revisions=revs)
    assert rows["A"]["revisions"]["top"] == [["Ed", 1]] and rows["A"]["revisions"]["others"] == 1
    assert rows["Ed"]["revisions"]["theirs"] == [["A", 1]]


def test_untouched_and_most_revised(bs):
    """untouched: own cards with no revision, oldest: the earliest created of them; most needs n >= 2; no history: no key."""
    cards = [card("a1", "A"), card("a2", "A", created="2020-01-01"), card("a3", "A", created="2019-05-05"), card("a4", "A"),
             card("dup", "A", dup=True)]
    rows, _ = rows_of(bs, cards, revisions={"a1": {"A": 2, "B": 1}, "a4": {"B": 1}, "dup": {"B": 9}})
    r = rows["A"]["revisions"]
    assert (r["untouched"], r["oldest"], r["most"], r["others"]) == (2, "a3", ["a1", 3], 2)
    rows, _ = rows_of(bs, cards, revisions={"a1": {"A": 1}})
    assert "most" not in rows["A"]["revisions"]
    rows, _ = rows_of(bs, cards, revisions=None)
    assert "revisions" not in rows["A"]


def test_power_profile(bs):
    """power: own cards with z_total; z = the mean (2 dp), top/bottom = flag counts, dots by z with their flag; site.z: all."""
    balance = {"cards": {"a1": {"z_total": 1.5, "flags": ["top"]}, "a2": {"z_total": -0.25, "flags": []},
                         "a3": {"z_total": -2.0, "flags": ["bottom"]}, "b1": {"z_total": 0.5, "flags": []}}}
    rows, out = rows_of(bs, [card("a1", "A"), card("a2", "A"), card("a3", "A"), card("b1", "B")], balance=balance)
    assert rows["A"]["power"] == {"n": 3, "z": -0.25, "top": 1, "bottom": 1, "dots": [["a3", -2.0, "bottom"], ["a2", -0.25], ["a1", 1.5, "top"]]}
    assert out["site"] == {"z": -0.06}


def test_buffs_and_nerfs_and_who_made_them(bs):
    """Change rows with bal on A's card: a nerf at B's PSD commit (B made it), a buff at a second two commits share (the
    one that touched Alpha.psd: B's), a buff at a second with no commit (counted, nobody's). swing = the largest, recent
    newest first."""
    psd = "Creatures/Other/2 Stars/Alpha.psd"
    changes = [{"ts": 300, "stats": [[3, 2, 2, 1], [4, 2, 2, 1]], "bal": "buff"},
               {"ts": 200, "stats": [[3, 2, 2, 1], [3, 3, 3, 1]], "bal": "buff"},
               {"ts": 100, "stats": [[5, 3, 2, 1], [3, 2, 2, 1]], "bal": "nerf"},
               {"ts": 50, "stats": [[3, 2, 2, 1], [3, 2, 2, 1]], "words": True}]
    commits = [commit("c3", 200, "B", [psd]), commit("c2", 200, "C", ["Creatures/Other/2 Stars/Beta.psd"]), commit("c1", 100, "B", [psd])]
    cards = [card("alpha", "A", psd=psd), card("beta", "C", psd="Creatures/Other/2 Stars/Beta.psd")]
    rows, _ = rows_of(bs, cards, commits, balance={"cards": {"alpha": {"changes": changes}}})
    a = rows["A"]["bal"]
    assert (a["buff"], a["nerf"], a["madeBuff"], a["madeNerf"], a["swing"]) == (2, 1, 0, 0, ["alpha", -3, 100])
    assert a["recent"] == [["alpha", "buff", 1, 300], ["alpha", "buff", 2, 200], ["alpha", "nerf", -3, 100]]
    assert (rows["B"]["bal"]["madeBuff"], rows["B"]["bal"]["madeNerf"]) == (1, 1) and "bal" not in rows["C"]
    assert rows["A"]["rework"] == {"n": 1, "of": 1, "most": ["alpha", 4]}


def test_no_stat_history_is_an_empty_bal(bs):
    """Rows without bal: bal all zeros, kept because power is, so the page shows its empty state."""
    balance = {"cards": {"a1": {"z_total": 0.5, "flags": [], "changes": [{"ts": 1, "words": True}]}}}
    rows, _ = rows_of(bs, [card("a1", "A")], balance=balance)
    assert rows["A"]["bal"] == {"buff": 0, "nerf": 0, "shift": 0, "madeBuff": 0, "madeNerf": 0, "recent": []}


def rate(value, by="group", day="2025-04-28", scale="favorability", source="TTS session"):
    return {"date": day, "scale": scale, "value": value, "by": by, "source": source}


def test_ratings_use_each_raters_latest_and_leave_out_their_own(bs):
    """Group 2023 = 4, group 2025 = 8 and the author's own 10: P4 fav 8 for the author. best/worst need two rated cards
    and never share one."""
    cards = [card("a1", "A", ratings=[rate(4, day="2023-04-28"), rate(8), rate(10, by="A", day="2026-10-24", source="website")])]
    rows, _ = rows_of(bs, cards)
    assert rows["A"]["ratings"] == {"favN": 1, "powerN": 0, "fav": 8}
    cards += [card("a2", "A", ratings=[rate(3)]), card("a3", "A", ratings=[rate(6), rate(2, scale="power")])]
    rows, _ = rows_of(bs, cards)
    r = rows["A"]["ratings"]
    assert (r["best"], r["worst"], r["power"], r["powerN"]) == ([["a1", 8], ["a3", 6]], [["a2", 3]], 2, 1)
    assert rows["A"]["editFields"] == [["rating", 1]]


def test_least_liked_is_below_every_most_liked(bs):
    """A card rated as a most liked one isn't least liked (a tie split by id): 7.1, 7.1, 7.1, 6 -> worst 6 only."""
    cards = [card(i, "A", ratings=[rate(v)]) for i, v in (("a1", 7.1), ("a2", 7.1), ("a3", 7.1), ("a4", 6))]
    r = rows_of(bs, cards)[0]["A"]["ratings"]
    assert (r["best"], r["worst"]) == ([["a1", 7.1], ["a2", 7.1]], [["a4", 6]])


def test_rater_stats_hide_without_per_person_ratings(bs):
    cards = [card(f"a{i}", "A", ratings=[rate(i + 1), rate(i + 2, scale="power")]) for i in range(6)]
    rows, _ = rows_of(bs, cards, balance={"cards": {f"a{i}": {"z_total": i / 10, "flags": []} for i in range(6)}})
    assert not {"selfBias", "rater", "powerVsStats"} & set(rows["A"]) and rows["A"]["ratings"]["favN"] == 6


def test_rater_stats_with_per_person_ratings(bs):
    """X's six cards rated by X and Y (favorability), X's power vs the cards' z_total: harshness, agreement, taste twin,
    self-rating bias and power vs stats, by hand."""
    xs, ys, pw, zs = [2, 4, 6, 8, 10, 6], [3, 5, 7, 9, 9, 5], [1, 2, 3, 4, 5, 6], [0.1, 0.25, 0.3, 0.45, 0.5, 0.7]
    cards = [card(f"c{i}", "X", ratings=[rate(xs[i], "X"), rate(ys[i], "Y"), rate(pw[i], "X", scale="power")]) for i in range(6)]
    cards.append(card("y1", "Y"))
    rows, _ = rows_of(bs, cards, balance={"cards": {f"c{i}": {"z_total": zs[i], "flags": []} for i in range(6)}})
    r = round(statistics.correlation(xs, ys), 2)
    assert rows["X"]["rater"] == {"n": 6, "harsh": -0.3, "agree": r, "twin": ["Y", r]}
    assert rows["Y"]["rater"] == {"n": 6, "harsh": 0.3, "agree": r, "twin": ["X", r]}
    assert rows["X"]["selfBias"] == {"fav": -0.3, "n": 6}
    assert rows["X"]["powerVsStats"] == {"r": round(statistics.correlation(pw, zs), 2), "n": 6}
    assert "selfBias" not in rows["Y"] and "powerVsStats" not in rows["Y"]


def test_power_only_raters_get_power_bias_and_power_vs_stats(bs):
    """X power-rates their six cards and no favorability: selfBias power and powerVsStats, no rater (P7: favorability)."""
    pw, zs = [1, 2, 3, 4, 5, 6], [0.1, 0.25, 0.3, 0.45, 0.5, 0.7]
    cards = [card(f"c{i}", "X", ratings=[rate(5, scale="power"), rate(pw[i], "X", scale="power", source="website")]) for i in range(6)]
    rows, _ = rows_of(bs, cards, balance={"cards": {f"c{i}": {"z_total": zs[i], "flags": []} for i in range(6)}})
    assert rows["X"]["selfBias"] == {"power": -1.5, "powerN": 6} and "rater" not in rows["X"]
    assert rows["X"]["powerVsStats"] == {"r": round(statistics.correlation(pw, zs), 2), "n": 6}


def test_a_noreply_login_rates_and_edits_as_its_row(bs):
    """A login known only through Chestnut's noreply commits: its ratings and edit fields land on Chestnut's row."""
    commits = [("a" * 40, T, "Chestnut", "123+JasonBechdolt@users.noreply.github.com", "s", [], [])]
    row_of = bs.login_rows(commits, {}, {})
    assert row_of("jasonbechdolt") == "Chestnut" and row_of("nobody") is None
    cards = [card(f"c{i}", "Chestnut", ratings=[rate(5)]) for i in range(3)]
    ratings = {bs.card_key(c["key"]): (c["key"], [rate(8, "JasonBechdolt", "2026-10-24", source="website")]) for c in cards}
    for c in cards:
        c["ratings"] = []
    bs.apply_ratings(cards, ratings, [], {}, {}, row_of)
    assert cards[0]["ratings"][0]["by"] == "Chestnut" and cards[0]["ratings"][0]["login"] == "JasonBechdolt"
    fields_by = {}
    edits = {"c0": ({"series": "Mario"}, {"_by": "JasonBechdolt", "_at": "2026-10-20T00:00:00Z"})}
    bs.edit_batches(None, {}, {}, edits, lambda n: None, lambda k: k, 0, fields_by=fields_by, row_of=row_of)
    assert fields_by == {"Chestnut": {"series": 1}}
    for c in cards:
        c["ratings"].insert(0, rate(5))
    rows, _ = rows_of(bs, cards, commits, fields_by=fields_by)
    assert rows["Chestnut"]["selfBias"] == {"fav": 3, "n": 3} and rows["Chestnut"]["editFields"] == [["rating", 3], ["series", 1]]


def test_mass_passes_count_card_images_only(bs):
    """A commit over 21 rulebook pages and one card is no mass pass: the card's revision counts."""
    v = lambda sha: ((sha, 0, "x", "s"), "p")
    found = {"a.png": [v("big"), v("up")], **{f"r{i}.png": [v("big"), v(f"up{i}")] for i in range(bs.MASS_PASS + 1)}}
    by_png = {"a.png": [card("alpha", "A")], **{f"r{i}.png": [card(f"rule{i}", "")] for i in range(bs.MASS_PASS + 1)}}
    rules = {f"rule{i}" for i in range(bs.MASS_PASS + 1)}
    assert bs.mass_passes(found) == {"big"} and bs.mass_passes(found, by_png, rules) == set()
    revs = bs.card_revisions(found, by_png, [commit("big", 2, "B"), commit("up", 1, "A")], {}, {}, skip=rules)
    assert revs == {"alpha": {"B": 1}}


def test_edit_fields_count_every_line(bs):
    """edit_batches without history: a stamp group of 70 cards counts 70 series edits, past the BATCH_LINES cut."""
    edits = {f"card{i}": ({"series": "Mario"}, {"_by": "editor", "_at": "2026-10-20T00:00:00Z"}) for i in range(70)}
    fields_by = {}
    bs.edit_batches(None, {"editor": "Ed"}, {}, edits, lambda n: None, lambda k: k, 0, fields_by=fields_by)
    assert fields_by == {"Ed": {"series": 70}} and 70 > bs.BATCH_LINES
    rows, _ = rows_of(bs, [card("e1", "Ed")], fields_by=fields_by)
    assert rows["Ed"]["editFields"] == [["series", 70]]


def test_archived_cards_reach_their_authors_row(bs):
    """archived_cards rows with author A reach A's row, newest first, without ratings, commit or path."""
    archived = [{"card": "New", "title": "New", "tag": "renamed", "renamed_to": "Newer", "renamedId": "newer", "at": 300, "created": "2024-01-02",
                 "author": "A", "path": "Creatures/New.psd", "commit": "c" * 40, "ratings": [rate(5)], "card_type": "creature"},
                {"card": "Old", "title": "Old", "tag": "retired", "moved_to": "MDW/Old.psd", "at": 100, "author": "A"},
                {"card": "Other", "title": "Other", "tag": "deleted", "at": 200, "author": "B"}]
    rows, _ = rows_of(bs, [card("a1", "A")], archived=archived)
    assert rows["A"]["archived"] == [{"card": "New", "title": "New", "tag": "renamed", "renamed_to": "Newer", "renamedId": "newer", "at": 300,
                                      "created": "2024-01-02"},
                                     {"card": "Old", "title": "Old", "tag": "retired", "moved_to": "MDW/Old.psd", "at": 100}]


def test_old_positional_call_still_works(bs):
    """contributions(cards, people, profiles, edits, commits) gives the same rows as before: no new keys, no site."""
    out = bs.contributions([], {}, [], {}, [commit("a" * 40, T, "Maker", ["Creatures/Other/2 Stars/Alpha.psd"])])
    assert set(out) == {"available", "since", "people"}
    assert set(out["people"][0]) == {"key", "cards", "edits", "commits", "first", "last", "recent", "recentEdits", "months", "logins"}


# ---- profile.js ---------------------------------------------------------------------------------------------------
def test_style_of_keeps_over_used_features(tmp_path):
    """Types, status effects and timing tags at n >= STYLE_MIN and ratio >= STYLE_RATIO (their share / all cards'),
    neutral tags (Action) left out, duplicates and hidden cards not in the site's counts; words per ability."""
    out = node(tmp_path, """
const C = (id, author, types = [], ability = "Hit.", extra = {}) => ({ id, name: id, author, types, ability, ...extra });
const all = [C("o1", "A", ["panda"], "(OS) (Action) Hit the foe."), C("o2", "A", ["panda"], "(OS) Hit."), C("o3", "A", ["panda", "fire"], "(OS) Hit."),
  C("o4", "A", ["fire"], "Burn it. Frost."), C("o5", "A", [], "Frost bite."), C("o6", "A", [], "Frost.")];
for (let i = 0; i < 14; i++) all.push(C(`s${i}`, "B", i < 2 ? ["panda"] : [], i < 1 ? "(Action) Run." : "Run."));
all.push(C("d1", "B", ["panda"], "(OS) x", { dup: true }), C("frost", "B", [], "Frozen.", { type: "debuff" }));
const own = pf.cardsOf("A", all);
const mentions = new Map(all.map(c => [c.id, new Set(/Frost/.test(c.ability) ? ["frost"] : [])]));
const s = pf.styleOf(own, all, { mentions, statusIds: new Set(["frost"]) });
out({ s });
""")
    s = out["s"]
    assert [r[:3] for r in s["types"]] == [["panda", 3, 2.1]] and s["types"][0][3] == pytest.approx(5 / 21)
    assert [r[:3] for r in s["status"]] == [["frost", 3, 3.5]]
    assert [r[:3] for r in s["tags"]] == [["OS", 3, 3.5]]
    assert (s["words"], s["siteWords"]) == (2.5, 1.5)


def test_ranks_share_ties(tmp_path):
    """Competition ranking among the rows with a value, two or more (B alone has edits); strength needs power.n >= 5,
    words per ability 5 abilities."""
    out = node(tmp_path, """
const people = [{ key: "A", cards: 5, commits: 2, edits: 0, power: { n: 5, z: 0.2 } }, { key: "B", cards: 5, commits: 3, edits: 1, power: { n: 4, z: 1 } },
  { key: "C", cards: 3, commits: 3, edits: 0, power: { n: 6, z: 0.5 } }];
const words = new Map([["A", { words: 20, n: 5 }], ["B", { words: 30, n: 4 }], ["C", { words: 25, n: 9 }]]);
out(people.map(p => pf.ranksOf(p, people, words)));
""")
    assert out == [[["cards", 1, 3], ["card commits", 3, 3], ["strength", 2, 2], ["words per ability", 2, 2]],
                   [["cards", 1, 3], ["card commits", 1, 3]],
                   [["cards", 3, 3], ["card commits", 1, 3], ["strength", 1, 2], ["words per ability", 1, 2]]]


def test_career_series_mix_and_most_referenced(tmp_path):
    out = node(tmp_path, """
const C = (id, author, series, created, stars, extra = {}) => ({ id, name: id, author, series, created, stars, type: "creature", ...extra });
const all = [C("a1", "A", "Mario", "2019-03-01", 2), C("a2", "A", "Mario", "2020-03-01", 2), C("a3", "A", "Kirby", "2020-05-01", 3),
  C("a4", "A", "Zelda", null, null, { type: "item" }), C("b1", "B", "Kirby", "2021-01-01", 3), C("b2", "B", "Zelda", "2021-01-01", 3), C("b3", "B", "Zelda", "2021-02-01", 3)];
const own = pf.cardsOf("A", all), c = pf.careerOf(own);
const mentionedBy = new Map([["a1", new Set(["x", "y"])], ["a3", new Set(["x", "y"])], ["a2", new Set(["z"])]]);
out({ career: [c.first.id, c.last.id, c.years, c.best], series: pf.seriesOf("A", own, all), none: pf.careerOf([C("u", "A")]),
  ref: (([card, n]) => [card.id, n])(pf.mostReferenced(own, mentionedBy)), noRef: pf.mostReferenced(own, new Map([["a2", new Set(["z"])]])),
  mix: pf.mixOf(own, all) });
""")
    assert out["career"] == ["a1", "a3", [[2019, 1], [2020, 2]], [2020, 2]] and out["none"] is None
    assert out["series"] == {"n": 3, "leads": ["Kirby", "Mario"]}         # Kirby 1-1: ties share; Zelda is B's
    assert out["ref"] == ["a1", 2] and out["noRef"] is None
    assert out["mix"]["stars"] == [[2, 2 / 3, 2 / 6], [3, 1 / 3, 4 / 6]]
    assert out["mix"]["kinds"] == [["creature", 0.75, 6 / 7], ["item", 0.25, 1 / 7]]


PAGE = """
const person = %s;
const html = pf.personHtml({ login: person.key, profile: null, person, state: "", own: false, canEdit: false, byId: new Map(), cards: [], tile: () => "",
  available: false, titles: new Map(), date: String, month: String, kinds: [], kindOf: c => c.type, decks: {}, people: [person], site: { z: 0 } });
out(text(html));
"""


def test_panels_hide_and_empty_states(tmp_path):
    """An old-shape person has none of the new panels; power with an all-zero bal shows the empty line; no rater, no
    taste twin."""
    old = node(tmp_path, PAGE % '{ key: "A", logins: ["A"], cards: 0, edits: 1, commits: 0, recent: [], months: [] }')
    assert not [h for h in ("Balance", "Ratings", "Revisions", "Archived cards", "Style") if h in old]
    new = node(tmp_path, PAGE % '{ key: "A", logins: ["A"], cards: 0, edits: 1, commits: 0, recent: [], months: [], '
                                'power: { n: 1, z: 0.5, top: 0, bottom: 0, dots: [["x", 0.5]] }, '
                                'bal: { buff: 0, nerf: 0, shift: 0, madeBuff: 0, madeNerf: 2, recent: [] }, ratings: { favN: 1, powerN: 0, fav: 7 }, '
                                'editFields: [["author", 3], ["problems_ok", 1]] }')
    assert "Balance" in new and "No buffs or nerfs recorded yet." in new and "Made: 2 nerfs to others' cards" in new
    assert "Favorability 7 · 1 card" in new and "taste twin" not in new and "Own cards" not in new
    assert "Edited: author 3 · parser notes checked 1" in new


def test_own_cards_line_shows_power_bias(tmp_path):
    """P5: a power-only rater's bias has its own line; with both scales, both parts."""
    out = node(tmp_path, """
const html = sb => text(pf.ratingsPanelHtml({ ratings: { powerN: 6, power: 5.5, favN: 0 }, selfBias: sb }, { byId: new Map(), tile: () => "" }));
out([html({ power: -1.5, powerN: 6 }), html({ fav: -0.1, n: 8, power: 2.1, powerN: 8 })]);
""")
    assert "Own cards vs others: power −1.5 (6)" in out[0] and "favorability" not in out[0]
    assert "Own cards vs others: favorability −0.1 (8) · power +2.1 (8)" in out[1]


def test_activity_labels_clear_the_top_tick(tmp_path):
    """Each activity row's label sits right above its columns: the highest y-axis number (11 px text) starts inside
    the chart, not over the label."""
    out = node(tmp_path, """
const html = pf.activityHtml({ months: [["2024-01", 3, 1, 2], ["2024-03", 9, 0, 4]] });
out(html.match(/<svg[\\s\\S]*?<\\/svg>/g).map(svg => Math.min(...[...svg.matchAll(/<text x="[^"]+" y="([-\\d.]+)" text-anchor="end">/g)].map(m => +m[1]))));
""")
    assert len(out) == 3 and all(y >= 11 for y in out), out


def test_balance_shows_the_biggest_swing(tmp_path):
    """P1's biggest swing: its card, the signed change and month, beside the newest rows."""
    out = node(tmp_path, """
const x = { byId: new Map([["big", { id: "big", name: "Big" }]]), month: ts => `M${ts}` };
const rows = ["a", "b", "c", "d", "e"].map((id, i) => [id, "buff", 1, 200 + i]);
const html = pf.balancePanelHtml({ bal: { buff: 3, nerf: 3, shift: 0, madeBuff: 0, madeNerf: 0, swing: ["big", -9, 100], recent: rows } }, x);
out({ text: text(html), html });
""")
    assert "Biggest swing: Big −9 · M100" in out["text"] and 'data-card="big"' in out["html"]


def test_archived_tags(tmp_path):
    """renamed_to / moved_to / neither: Renamed with "renamed to" its card, Retired, Deleted; the years it lived."""
    out = node(tmp_path, """
const rows = [{ card: "Zoey", title: "Zoey", renamed_to: "Z!!!!!", renamedId: "z", at: Date.parse("2025-10-09T12:00:00Z") / 1000, created: "2020-01-01" },
  { card: "Ech", title: "Ech", moved_to: "MDW/The Morgue/Ech.psd", at: Date.parse("2025-03-22T12:00:00Z") / 1000, created: "2025-01-01" },
  { card: "Gone", title: "Gone", tag: "deleted" }];
const html = pf.archivedPanelHtml(rows);
out({ html, text: text(html), none: pf.archivedPanelHtml([]) });
""")
    assert "Archived cards 3" in out["text"] and out["none"] == ""
    assert "Zoey Renamed renamed to Z!!!!! · 2020–2025" in out["text"] and '<a href="#/card/z">Z!!!!!</a>' in out["html"]
    assert "Ech Retired · 2025" in out["text"] and "Gone Deleted" in out["text"]
