"""
The website's Stats page (TTSCardMaker site/assets/stats.js, balance.js, insights.js, in node; build/analytics.py): stat
distributions paged past 10, the cooldown update gone, short chart texts, which Ratings and History charts show on
sparse and backfilled data, Too Stronk then nerfed, power creep and series strength. Loads the site from $AC_SITE_SRC
(a TTSCardMaker checkout) or the clone; skips without node or a stats.js with distPages.
"""
import importlib.util
import json
import shutil
from os import environ
from os.path import expanduser
from pathlib import Path
from subprocess import run as run_process
from types import SimpleNamespace

import pytest

from conftest import CLONE

SITE = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE))) / "site"
ASSETS = SITE / "assets"
STATS_JS = ASSETS / "stats.js"
pytestmark = pytest.mark.skipif(not (STATS_JS.is_file() and shutil.which("node") and (ASSETS / "insights.js").is_file()
                                     and "distPages" in STATS_JS.read_text(encoding="utf-8")),
                                reason=f"no stats.js with distPages at {STATS_JS}, or no node")

# twelve creatures in two series and two years; 2★ median 10, 3★ median 14 (strength = total - median):
# Mario +2 -1 +2 -1 0 +1 (mean 0.5), Kirby -2 -2 +1 +3 -4 0 (mean -0.7); 2020: 0.0, 2021: -0.2
PRELUDE = """
globalThis.getComputedStyle = () => ({ getPropertyValue: () => "#336699" });
globalThis.document = { documentElement: {} };
const st = await import(%(stats)s), bal = await import(%(balance)s), ins = await import(%(insights)s);
const text = html => html.replace(/<[^>]+>/g, " ").replace(/\\s+/g, " ").trim();
const out = x => process.stdout.write(JSON.stringify(x));
const chart = (html, id) => (html.match(new RegExp(`<section class="chart[^"]*" id="chart-${id}">[\\\\s\\\\S]*?</section>`)) || [""])[0];
const rows = html => [...html.matchAll(/<tr>([\\s\\S]*?)<\\/tr>/g)].map(m => [...m[1].matchAll(/<td[^>]*>([\\s\\S]*?)<\\/td>/g)].map(x => text(x[1]))).filter(r => r.length);
const ts = s => Date.parse(s) / 1000;
const C = (name, series, stars, line, created, author, extra = {}) => ({ id: name.toLowerCase(), name, key: name, type: "creature", series, stars,
  hp: line[0], def: line[1], atk: line[2], spd: line[3], created, author, ability: "(1CD) Hit the foe.", ts: ts(created), ...extra });
function fixture() {
  return [C("M1", "Mario", 2, [3, 3, 3, 3], "2020-05-01", "Ann"), C("M2", "Mario", 2, [3, 2, 2, 2], "2020-05-01", "Ann"),
    C("M3", "Mario", 3, [4, 4, 4, 4], "2020-05-01", "Ann"), C("M4", "Mario", 3, [4, 3, 3, 3], "2021-06-01", "Ann"),
    C("M5", "Mario", 2, [3, 3, 2, 2], "2021-06-01", "Ann"), C("M6", "Mario", 3, [4, 4, 4, 3], "2021-06-01", "Ann"),
    C("K1", "Kirby", 2, [2, 2, 2, 2], "2020-05-01", "Bob"), C("K2", "Kirby", 3, [3, 3, 3, 3], "2020-05-01", "Bob"),
    C("K3", "Kirby", 2, [3, 3, 3, 2], "2020-05-01", "Bob"), C("K4", "Kirby", 3, [5, 4, 4, 4], "2021-06-01", "Bob"),
    C("K5", "Kirby", 2, [2, 2, 1, 1], "2021-06-01", "Bob"), C("K6", "Kirby", 3, [4, 4, 3, 3], "2021-06-01", "Ann"),
    { id: "potion", name: "Potion", key: "Potion", type: "item", subtype: "heal", series: "Mario", stars: 1, created: "2022-01-01", author: "Ann",
      ability: "(2GCD) Heal 2. (1/3CD) Heal 1.", ts: ts("2022-01-01") }];
}
function analytics(cards, extra = {}) {
  const co = (n, p50) => ({ n, kind: "creature", total: { p10: p50 - 2, p50, p90: p50 + 2, min: p50 - 4, max: p50 + 3, mean: p50 } });
  const A = { cohorts: { "creature|2": co(6, 10), "creature|3": co(6, 14) }, cards: {}, counts: { creature: 12 }, versions: 0, since: null,
    time: { months: [], series: {} }, minCohort: 5, offScale: 40, ...extra };
  for (const c of cards) if (c.type === "creature") A.cards[c.id] = { cohort: `creature|${c.stars}`, stats: [c.hp, c.def, c.atk, c.spd], flags: [], below: 0, above: 0 };
  return A;
}
function render(cards, A, q = "", { banned = new Map(), distPage = {} } = {}) {
  const byId = new Map(cards.map(c => [c.id, c])), ctx = { cards, byId, banned, icons: {}, mentions: new Map(), banlist: { groups: [] }, distPage };
  const root = { innerHTML: "" }, p = new URLSearchParams(q);
  st.renderStats(root, ctx, p);
  return { top: root.innerHTML, balance: A ? bal.statsHtml(A, { cards, byId }, p) : "", ratings: ins.ratingsHtml(A, ctx, p), history: ins.historyHtml(A, ctx, p) };
}
"""


def node(tmp_path, body: str):
    """Run body after PRELUDE (st, bal, ins; fixture(), analytics(), render()); its stdout as JSON."""
    script = tmp_path / "probe.mjs"
    uri = lambda name: json.dumps((ASSETS / name).as_uri())
    script.write_text(PRELUDE % {"stats": uri("stats.js"), "balance": uri("balance.js"), "insights": uri("insights.js")} + body,
                      encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@pytest.fixture(scope="module")
def an():
    spec = importlib.util.spec_from_file_location("site_analytics_stats", SITE / "build" / "analytics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_distributions_page_only_past_ten(tmp_path):
    """0-10 always; a pager only on a stat with values past 10 (ATK 12: 11-20; SPD 20 and 35: 11-20, 31-40); one y
    scale per stat, a non-zero column at least 3 px and counts over pages above 0; the table lists 0-10 and each higher value, no "21+"; ctx.distPage opens a page."""
    out = node(tmp_path, """
const cards = fixture().filter(c => c.type === "creature");
cards[0].atk = 12; cards[1].spd = 20; cards[2].spd = 35;
const pages = k => st.distPages(st.countValues(cards, k));
const html = chart(render(cards, null).top, "dist"), opened = chart(render(cards, null, "", { distPage: { atk: 1 } }).top, "dist");
const panel = (h, k) => h.match(new RegExp(`<div class="dist-panel" data-stat="${k}"[\\\\s\\\\S]*?(?=<div class="dist-panel"|</div></div><div class="view-table")`))[0];
const pageOf = (h, k, p) => panel(h, k).match(new RegExp(`<div class="dist-page" data-page="${p}"[^>]*>([\\\\s\\\\S]*?)</svg>`));
const labels = svg => [...svg.matchAll(/<text x="[^"]+" y="(?:164|162)" text-anchor="middle">([^<]*)</g)].map(m => m[1]);
const topTick = svg => [...svg.matchAll(/text-anchor="end">([^<]*)</g)].map(m => m[1]).pop();
out({ pages: ["hp", "atk", "spd"].map(pages), pager: ["hp", "def", "atk", "spd"].map(k => panel(html, k).includes("dist-pager")),
  p0: labels(pageOf(html, "spd", 0)[1]), atk1: labels(pageOf(html, "atk", 1)[1]),
  ticks: [0, 1, 3].map(p => topTick(pageOf(html, "spd", p)[1])), hidden: pageOf(html, "atk", 1)[0].includes("hidden"),
  values: rows(html.split('class="view-table"')[1]).map(r => r[0]),
  opened: [panel(opened, "atk").match(/dist-range">([^<]*)/)[1], pageOf(opened, "atk", 1)[0].includes("hidden"), pageOf(opened, "atk", 0)[0].includes("hidden")],
  lone: [...pageOf(html, "atk", 1)[1].matchAll(/<text class="val"[^>]*>([^<]*)</g)].map(m => m[1]), p0vals: pageOf(html, "spd", 0)[1].includes('class="val"'),
  tops: [...pageOf(html, "atk", 1)[1].matchAll(/class="mark" d="M[^Q]*Q[\\d.]+,([\\d.]+)/g)].map(m => +m[1]) });
""")
    assert out["pages"] == [[0], [0, 1], [0, 1, 3]]
    assert out["pager"] == [False, False, True, True]
    assert out["p0"] == [str(v) for v in range(11)] and out["atk1"] == [str(v) for v in range(11, 21)]
    assert len(set(out["ticks"])) == 1 and out["hidden"]
    assert out["values"] == [str(v) for v in range(11)] + ["12", "20", "35"]
    assert out["opened"] == ["11–20", False, True]
    assert out["lone"] == ["1"] and not out["p0vals"] and len(out["tops"]) == 1 and out["tops"][0] <= 144 - 3   # a lone card shows


def test_item_types_read_as_in_the_rulebook(tmp_path):
    """S19 rows: "Heal", not the subtype key "heal" (chart and table)."""
    out = node(tmp_path, """
const html = chart(render(fixture(), null).top, "itemtypes");
out({ cells: [...html.matchAll(/<t[hd][^>]*>(\\w+)<\\/t[hd]>/g)].map(m => m[1]).filter(s => /^heal$/i.test(s)), table: rows(html.split('class="view-table"')[1] || "").map(r => r[0]) });
""")
    assert out["table"] == ["Heal"] and out["cells"] and set(out["cells"]) == {"Heal"}


def test_cooldown_update_is_gone(tmp_path, an):
    """No "The cooldown update" chart (balance.js), no `old` flag on a change row and no "old tags" in the build log."""
    out = node(tmp_path, """
const cards = fixture(), A = analytics(cards);
A.cards.m1.changes = [{ ts: ts("2026-10-01"), text: [0, 1], tags: [["(2PT)", "(1CD)"]] }];
for (const c of cards.slice(1, 7)) A.cards[c.id].changes = [{ ts: ts("2026-10-01"), text: [0, 1], tags: [["(2PT)", "(1CD)"]] }];
const html = bal.statsHtml(A, { cards, byId: new Map(cards.map(c => [c.id, c])) }, new URLSearchParams());
out([html.includes("chart-bal-cooldown"), html.includes("The cooldown update"), html.includes("chart-bal-months")]);
""")
    assert out == [False, False, True]
    card = lambda text: SimpleNamespace(ability=text, stats=None, stars=2, card_type="creature", subtype=None, series=None, types=[], creatures=None)
    row = an.change(card("(2PT) Hit."), card("(1CD) Hit."), ["(2PT) Hit.", "(1CD) Hit."])
    assert row["tags"] == [["(2PT)", "(1CD)"]] and "old" not in row and "words" not in row
    lines = an.summary({"cards": {"a": {"changes": [row]}}, "cohorts": {}, "counts": {}, "versions": 1})
    assert "timing tags 1," in lines[1] and "old tags" not in lines[1]


def test_cooldowns_count_uses_too(tmp_path):
    """S15: cooldowns (CD 1/2/3+, GCD 1/2+) and uses (U 1/2/3+) per star level; average turns of the cooldowns only."""
    out = node(tmp_path, """
const cards = fixture();
cards[0].ability = "(2U) Hit. (4U) Hit."; cards[1].ability = "(3CD) Hit.";
out(rows(chart(render(cards, null).top, "cooldowns").split('class="view-table"')[1]));
""")
    assert out[0] == ["1★", "0", "0", "1", "0", "1", "0", "0", "0", "2.5"]                  # Potion: (2GCD), (1/3CD)
    assert out[1] == ["2★", "4", "0", "1", "0", "0", "0", "1", "1", "1.4"]                  # M1's uses count, not in the turns


def test_descriptions_are_short(tmp_path):
    """Every chart's sub line is 70 characters or fewer, on the top part, Balance, Ratings and History; the long notes
    are gone (the Balance intro's, the card page's recorded-changes and image-versions notes)."""
    out = node(tmp_path, """
const cards = fixture(), A = analytics(cards, { revisions: { available: true, mass: 20, cards: { m1: { Bob: 2 } } },
  archived: [{ card: "Old", title: "Old", tag: "retired", card_type: "creature", at: ts("2025-01-01"), created: "2023-01-01" }] });
cards.forEach((c, i) => (c.ratings = [{ date: "2023-04-28", scale: "power", value: 2 + (i % 8), by: "group", source: "TTS session" },
  { date: "2023-04-28", scale: "favorability", value: 3 + (i % 6), by: "group", source: "TTS session" },
  { date: "2025-04-28", scale: "favorability", value: 8 - (i % 6), by: "group", source: "TTS session" }]));
A.cards.m1.changes = [{ ts: ts("2025-08-01"), stats: [[2, 3, 3, 3], [3, 3, 3, 3]], bal: "buff" }];
const r = render(cards, A), all = r.top + r.balance + r.ratings + r.history;
const card = bal.changesBlock(cards[0], A, { byId: new Map(cards.map(c => [c.id, c])) });
out({ subs: [...all.matchAll(/<p class="sub">([^<]*)<\\/p>/g)].map(m => text(m[1]).replace(/&#39;/g, "'")), text: text(all + card),
  ids: [...all.matchAll(/id="chart-([^"]+)"/g)].map(m => m[1]) });
""")
    assert len(out["ids"]) > 25
    assert [s for s in out["subs"] if len(s) > 70] == []
    app = (ASSETS / "app.js").read_text(encoding="utf-8")
    for gone in ("Dates are Pacific time, the bot's", "From the bot's card history", "Each commit that changed the image"):
        assert gone not in out["text"] and gone not in app, gone


def _rated(body):
    """fixture() with group ratings: power on every creature (2023-04-28), favorability twice (2023, 2025)."""
    return """
const cards = fixture(), A = analytics(cards);
cards.forEach((c, i) => (c.ratings = [{ date: "2023-04-28", scale: "power", value: 1.5 + (i % 9), by: "group", source: "TTS session" },
  { date: "2023-04-28", scale: "favorability", value: 3 + (i % 6), by: "group", source: "TTS session" },
  { date: "2025-04-28", scale: "favorability", value: 8 - (i % 6), by: "group", source: "TTS session" }]));
""" + body


def test_ratings_section_visibility(tmp_path):
    """No ratings: no section. Group ratings: rated power vs stats, favourites, risers and fallers, power tier vs ban
    list; no most divisive until a card has 3 people's ratings. Never NaN, undefined or Infinity."""
    out = node(tmp_path, _rated("""
const ids = h => [...h.matchAll(/id="chart-([^"]+)"/g)].map(m => m[1]);
const none = fixture(), group = render(cards, A).ratings;
const people = cards.map(c => ({ ...c, ratings: [...c.ratings] }));
for (const who of ["ann", "bob", "cy"]) people[0].ratings.push({ date: "2026-10-24", scale: "favorability", value: who === "bob" ? 2 : 9, by: who, source: "website" });
const personal = render(people, A).ratings;
const bans = chart(render(cards, A, "", { banned: new Map([["m1", {}], ["k5", {}]]) }).ratings, "rat-bans");
out({ none: render(none, analytics(none)).ratings, jump: [render(none, null).top.includes('data-scroll="stats-ratings"'), render(cards, A).top.includes('data-scroll="stats-ratings"')],
  group: ids(group), personal: ids(personal), noA: ids(render(cards, null).ratings),
  bad: ["NaN", "undefined", "Infinity"].filter(w => (group + personal).includes(w)), bans: rows(bans) });
"""))
    assert out["none"] == "" and out["jump"] == [False, True]
    assert {"rat-power", "rat-fav", "rat-movers", "rat-bans"} <= set(out["group"]) and "rat-divisive" not in out["group"]
    assert "rat-divisive" in out["personal"]
    assert "rat-power" not in out["noA"] and "rat-nerfed" not in out["noA"] and "rat-fav" in out["noA"]
    assert out["bad"] == []
    assert sum(int(r[1]) for r in out["bans"]) == 13 and sum(int(r[2]) for r in out["bans"]) == 2


def test_too_stronk_nerfed_after_rating(tmp_path):
    """Rated Too Stronk (9.4) on 2025-03-02 and nerfed 2025-08-27: counted; a nerf on 2025-03-01 (Pacific) is not; a
    Plz buff card buffed later counts under Buffed later. No buff or nerf in scope: the shared empty line."""
    out = node(tmp_path, """
const cards = fixture(), A = analytics(cards), r = (date, value) => ({ date, scale: "power", value, by: "group", source: "TTS session" });
const by = Object.fromEntries(cards.map(c => [c.id, c]));
by.m1.ratings = [r("2025-03-02", 9.4)]; by.m2.ratings = [r("2025-03-02", 9.0)]; by.k1.ratings = [r("2023-04-28", 1.5)]; by.k2.ratings = [r("2023-04-28", 5)];
const before = analytics(cards);
A.cards.m1.changes = [{ ts: ts("2025-08-27T20:00:00Z"), stats: [[3, 3, 3, 3], [3, 3, 3, 2]], bal: "nerf" }];
A.cards.m2.changes = [{ ts: ts("2025-03-01T20:00:00Z"), stats: [[3, 3, 2, 2], [3, 2, 2, 2]], bal: "nerf" }];
A.cards.k1.changes = [{ ts: ts("2024-01-01T20:00:00Z"), stats: [[2, 2, 2, 1], [2, 2, 2, 2]], bal: "buff" }];
const html = chart(render(cards, A).ratings, "rat-nerfed");
out({ facts: [...html.matchAll(/<span class="fact">([\\s\\S]*?)<\\/span>/g)].map(m => text(m[1])), table: rows(html).map(r => r.slice(0, 2)),
  empty: text(chart(render(cards, before).ratings, "rat-nerfed")) });
""")
    assert out["facts"] == ["Too Stronk 2", "Nerfed later 1", "Plz buff 1", "Buffed later 1"]
    assert out["table"] == [["M1", "Too Stronk"], ["K1", "Plz buff"]]
    assert "No buffs or nerfs recorded yet." in out["empty"]


def test_history_charts_sparse_and_backfilled(tmp_path):
    """Without analytics.json: cards created per year and text freshness only. No full history: the shared line. After
    the backfill (90 buff/nerf rows over 36 months, revisions, 66 archived): who changes whose cards, most revised,
    retired and deleted (renames left out); Balance's by-stat sums are the rows'."""
    out = node(tmp_path, """
const cards = fixture(), ids = h => [...h.matchAll(/id="chart-([^"]+)"/g)].map(m => m[1]);
const sparse = render(cards, null).history, partial = render(cards, analytics(cards, { revisions: { available: false, mass: 20, cards: {} } })).history;
const A = analytics(cards, { revisions: { available: true, mass: 20, cards: { m1: { Bob: 3, Ann: 1 }, k1: { Ann: 2 } } }, archived: [] });
let up = [0, 0, 0, 0], down = [0, 0, 0, 0], seed = 7;
const rnd = n => (seed = (seed * 48271) % 2147483647) % n;
const creatures = cards.filter(c => c.type === "creature");
for (let i = 0; i < 90; i++) {
  const c = creatures[i % 12], a = [c.hp, c.def, c.atk, c.spd], b = [...a], k = rnd(4), d = rnd(2) ? 1 : -2;
  b[k] = Math.max(0, b[k] + d);
  if (b[k] === a[k]) b[k] += 1;
  b.forEach((v, j) => { up[j] += Math.max(0, v - a[j]); down[j] += Math.max(0, a[j] - v); });
  (A.cards[c.id].changes ||= []).push({ ts: ts("2023-11-01") + i * 12 * 86400, stats: [a, b], bal: b[k] > a[k] ? "buff" : "nerf" });
}
const tags = [...Array(26).fill("retired"), ...Array(5).fill("deleted"), ...Array(35).fill("renamed")];
A.archived = tags.map((tag, i) => ({ card: `Old ${i}`, title: `Old ${i}`, tag, card_type: "creature", series: "Mario", author: "Ann",
  at: ts("2024-02-01") + i * 9 * 86400, created: "2022-03-01" }));
const r = render(cards, A), statsRows = rows(chart(r.balance, "bal-stats")), archive = chart(r.history, "hist-archive");
out({ sparse: ids(sparse), partial: [ids(partial), text(partial)], full: ids(r.history), archived: rows(archive).length, life: text(archive).includes("Median life"),
  sums: statsRows.map(x => [+x[1], +x[2]]), up, down, who: rows(chart(r.history, "hist-who")).map(x => x.slice(0, 3)),
  revised: rows(chart(r.history, "hist-revised")).map(x => x.slice(0, 3)) });
""")
    assert out["sparse"] == ["hist-created", "hist-fresh"]
    assert out["partial"][0] == ["hist-who", "hist-revised", "hist-created", "hist-fresh"]
    assert out["partial"][1].count("Needs a full-history build.") == 2
    assert {"hist-who", "hist-revised", "hist-archive"} <= set(out["full"])
    assert out["archived"] == 31 and out["life"]
    assert out["sums"] == [list(p) for p in zip(out["up"], out["down"])]
    assert out["revised"] == [["M1", "4", "Bob"], ["K1", "2", "Ann"]]


def test_power_creep_and_series_strength(tmp_path):
    """Hand-computed: 2020 average vs median 0, 2021 -0.2 (12 creatures); Mario +0.5 and Kirby -0.7; the series chart
    opens on its table and is left out under a series filter."""
    out = node(tmp_path, """
const cards = fixture(), A = analytics(cards), r = render(cards, A);
const series = chart(r.balance, "bal-series");
out({ creep: rows(chart(r.balance, "bal-creep")), series: rows(series).map(x => x.slice(0, 5)),
  startsOnTable: series.includes('aria-pressed="true">Chart') && /<div class="view-chart" hidden>/.test(series),
  filtered: render(cards, A, "series=Mario").balance.includes("chart-bal-series"), item: render(cards, A, "kind=item").balance.includes("chart-bal-creep") });
""")
    minus = lambda s: s.replace("−", "-")
    assert [[minus(c) for c in row] for row in out["creep"]] == [["2020", "6", "0", "11.3"], ["2021", "6", "-0.2", "12.5"]]
    assert [[minus(c) for c in row] for row in out["series"]] == [["Mario", "7", "6", "2.5", "0.5"], ["Kirby", "6", "6", "2.5", "-0.7"]]
    assert out["startsOnTable"] and not out["filtered"] and not out["item"]
