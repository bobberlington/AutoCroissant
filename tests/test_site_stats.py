"""
The website's Stats page (TTSCardMaker site/assets/stats.js, balance.js, insights.js, in node; build/analytics.py): stat
distributions paged past 10, the cooldown update gone, timing tags only left out of the changes by month, short chart
texts, which Ratings and History charts show, power creep, subseries under their series, and the charts' links to the
Cards grid. Loads the site from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without node or a stats.js
with cardsHref (2026-10-25).
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
                                     and "cardsHref" in STATS_JS.read_text(encoding="utf-8")),
                                reason=f"no stats.js with cardsHref at {STATS_JS}, or no node")

# twelve creatures in two series and two years; 2★ median 10, 3★ median 14 (strength = total - median):
# Mario +2 -1 +2 -1 0 +1 (mean 0.5), Kirby -2 -2 +1 +3 -4 0 (mean -0.7); 2020: 0.0, 2021: -0.2
PRELUDE = """
globalThis.getComputedStyle = () => ({ getPropertyValue: () => "#336699" });
globalThis.document = { documentElement: {} };
const st = await import(%(stats)s), bal = await import(%(balance)s), ins = await import(%(insights)s);
const text = html => html.replace(/<[^>]+>/g, " ").replace(/\\s+/g, " ").trim();
const out = x => process.stdout.write(JSON.stringify(x));
const chart = (html, id) => (html.match(new RegExp(`<section class="chart[^"]*" id="chart-${id}">[\\\\s\\\\S]*?</section>`)) || [""])[0];
const rows = html => [...html.matchAll(/<tr[^>]*>([\\s\\S]*?)<\\/tr>/g)].map(m => [...m[1].matchAll(/<td[^>]*>([\\s\\S]*?)<\\/td>/g)].map(x => text(x[1]))).filter(r => r.length);
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
  const byId = new Map(cards.map(c => [c.id, c])), ctx = { cards, byId, banned, icons: {}, mentions: new Map(), distPage };
  const root = { innerHTML: "" }, p = new URLSearchParams(q);
  st.renderStats(root, ctx, p);
  return { top: root.innerHTML, balance: A ? bal.statsHtml(A, { cards, byId }, p) : "", ratings: ins.ratingsHtml(ctx, p), history: ins.historyHtml(ctx, p) };
}
const hrefs = html => [...html.matchAll(/<a class="go" href="([^"]+)"/g)].map(m => m[1].replace(/&amp;/g, "&"));
const params = href => { const p = Object.fromEntries(new URLSearchParams(href.split("?")[1])); if (p.nodup !== "1") throw new Error(`no nodup: ${href}`); delete p.nodup; return p; };
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
const labels = svg => [...svg.matchAll(/<text x="[^"]+" y="176" text-anchor="middle">([^<]*)</g)].map(m => m[1]);
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
    assert out["lone"] == ["1"] and not out["p0vals"] and len(out["tops"]) == 1 and out["tops"][0] <= 158 - 3   # a lone card shows


def test_stat_labels_clear_the_top_tick(tmp_path):
    """The stat's label (HP, ATK...) sits right above its chart: the highest y-axis number (11 px text) starts inside
    the chart, not over the label."""
    out = node(tmp_path, """
const cards = fixture().filter(c => c.type === "creature");
cards[0].atk = 12;
const svgs = chart(render(cards, null).top, "dist").match(/<svg[\\s\\S]*?<\\/svg>/g);
out(svgs.map(svg => Math.min(...[...svg.matchAll(/<text x="[^"]+" y="([-\\d.]+)" text-anchor="end">/g)].map(m => +m[1]))));
""")
    assert len(out) == 5 and all(y >= 11 for y in out), out


def test_item_types_read_as_in_the_rulebook(tmp_path):
    """S19 rows: "Heal", not the subtype key "heal" (chart and table)."""
    out = node(tmp_path, """
const html = chart(render(fixture(), null).top, "itemtypes");
out({ cells: [...html.matchAll(/<t[hd][^>]*>(\\w+)<\\/t[hd]>/g)].map(m => m[1]).filter(s => /^heal$/i.test(s)), table: rows(html.split('class="view-table"')[1] || "").map(r => r[0]) });
""")
    assert out["table"] == ["Heal"] and out["cells"] and set(out["cells"]) == {"Heal"}


def test_cooldown_update_and_tags_only_are_gone(tmp_path, an):
    """No "The cooldown update" chart (balance.js), no `old` flag on a change row and no "old tags" in the build log.
    Recorded changes by month: no "Timing tags only"; a tags-only change isn't counted, one with a series change is Other."""
    out = node(tmp_path, """
const cards = fixture(), A = analytics(cards), byId = new Map(cards.map(c => [c.id, c]));
for (const c of cards.slice(0, 7)) A.cards[c.id].changes = [{ ts: ts("2026-10-01"), text: [0, 1], tags: [["(2PT)", "(1CD)"]] }];
const only = bal.statsHtml(A, { cards, byId }, new URLSearchParams());
A.cards.m1.changes.push({ ts: ts("2026-09-10T20:00:00Z"), text: [0, 1], words: true }, { ts: ts("2026-09-12T20:00:00Z"), tags: [["(2PT)", "(1CD)"]], series: ["Mario", "Kirby"] });
const html = chart(bal.statsHtml(A, { cards, byId }, new URLSearchParams()), "bal-months");
out({ only: [only.includes("chart-bal-cooldown"), only.includes("The cooldown update"), only.includes("chart-bal-months")],
  tags: html.includes("Timing tags"), head: [...html.matchAll(/<th[^>]*>([^<]*)<\\/th>/g)].map(m => m[1]), rows: rows(html) });
""")
    assert out["only"] == [False, False, False] and not out["tags"]
    assert out["head"] == ["Month", "Ability reworded", "Stats", "Other (stars, kind, types, series)", "All"]
    assert out["rows"] == [["Sep 26", "1", "0", "1", "2"]]
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
const cards = fixture(), A = analytics(cards);
cards.forEach((c, i) => (c.ratings = [{ date: "2023-04-28", scale: "power", value: 2 + (i % 8), by: "group", source: "TTS session" },
  { date: "2023-04-28", scale: "favorability", value: 3 + (i % 6), by: "group", source: "TTS session" },
  { date: "2025-04-28", scale: "favorability", value: 8 - (i % 6), by: "group", source: "TTS session" }]));
A.cards.m1.changes = [{ ts: ts("2025-08-01"), stats: [[2, 3, 3, 3], [3, 3, 3, 3]], bal: "buff" }];
const r = render(cards, A), all = r.top + r.balance + r.ratings + r.history;
const card = bal.changesBlock(cards[0], A, { byId: new Map(cards.map(c => [c.id, c])) });
out({ subs: [...all.matchAll(/<p class="sub">([^<]*)<\\/p>/g)].map(m => text(m[1]).replace(/&#39;/g, "'")), text: text(all + card),
  ids: [...all.matchAll(/id="chart-([^"]+)"/g)].map(m => m[1]) });
""")
    assert len(out["ids"]) >= 20
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
    """No ratings: no section. Group ratings: favourites, risers and fallers; no most divisive until a card has 3
    people's ratings. Never NaN, undefined or Infinity."""
    out = node(tmp_path, _rated("""
const ids = h => [...h.matchAll(/id="chart-([^"]+)"/g)].map(m => m[1]);
const none = fixture(), group = render(cards, A).ratings;
const people = cards.map(c => ({ ...c, ratings: [...c.ratings] }));
for (const who of ["ann", "bob", "cy"]) people[0].ratings.push({ date: "2026-10-24", scale: "favorability", value: who === "bob" ? 2 : 9, by: who, source: "website" });
const personal = render(people, A).ratings;
out({ none: render(none, analytics(none)).ratings, jump: [render(none, null).top.includes('data-scroll="stats-ratings"'), render(cards, A).top.includes('data-scroll="stats-ratings"')],
  group: ids(group), personal: ids(personal), bad: ["NaN", "undefined", "Infinity"].filter(w => (group + personal).includes(w)) });
"""))
    assert out["none"] == "" and out["jump"] == [False, True]
    assert out["group"] == ["rat-fav", "rat-movers"] and out["personal"] == ["rat-fav", "rat-divisive", "rat-movers"]
    assert out["bad"] == []


def test_history_charts_and_stat_sums(tmp_path):
    """History: cards created per year and text freshness, with or without analytics.json. After 90 buff/nerf rows
    over 36 months, Balance's by-stat sums are the rows'."""
    out = node(tmp_path, """
const cards = fixture(), ids = h => [...h.matchAll(/id="chart-([^"]+)"/g)].map(m => m[1]);
const sparse = render(cards, null).history, A = analytics(cards);
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
const r = render(cards, A), statsRows = rows(chart(r.balance, "bal-stats"));
out({ sparse: ids(sparse), full: ids(r.history), sums: statsRows.map(x => [+x[1], +x[2]]), up, down });
""")
    assert out["sparse"] == out["full"] == ["hist-created", "hist-fresh"]
    assert out["sums"] == [list(p) for p in zip(out["up"], out["down"])]


def test_power_creep(tmp_path):
    """Hand-computed: 2020 average vs median 0, 2021 -0.2 (12 creatures); left out for items. The removed charts stay
    gone: series strength, ban reasons, last change by month, top subseries."""
    out = node(tmp_path, """
const cards = fixture(), A = analytics(cards), r = render(cards, A), all = r.top + r.balance + r.ratings + r.history;
out({ creep: rows(chart(r.balance, "bal-creep")), item: render(cards, A, "kind=item").balance.includes("chart-bal-creep"),
  gone: ["bal-series", "bans", "timeline", "subseries"].filter(id => all.includes(`id="chart-${id}"`)) });
""")
    minus = lambda s: s.replace("−", "-")
    assert [[minus(c) for c in row] for row in out["creep"]] == [["2020", "6", "0", "11.3"], ["2021", "6", "-0.2", "12.5"]]
    assert not out["item"] and out["gone"] == []


def test_series_rows_list_their_subseries(tmp_path):
    """Creatures by series: Mario, then its subseries (creatures) and No subseries, each a sub row in the chart and the
    table; a series without subseries has none. The Subseries tile counts the chart's; kind=item: Items by series."""
    out = node(tmp_path, """
const cards = fixture();
cards[0].subseries = cards[1].subseries = "Paper Mario"; cards[2].subseries = "Super Mario 64"; cards[12].subseries = "Paper Mario";
const html = chart(render(cards, null).top, "series");
const tile = h => (h.match(/subseries<\\/div><div class="value">(\\d+)/i) || [])[1];
const items = render(cards, null, "kind=item").top;
out({ table: rows(html), subs: [...html.matchAll(/<tr class="sub">/g)].length, bars: [...html.matchAll(/class="lbl( sub)?"[^>]*>([^<]*)</g)].map(m => [m[2], !!m[1]]),
  tiles: [tile(render(cards, null).top), tile(items)], items: rows(chart(items, "series")), title: chart(items, "series").match(/<h3>([^<]*)/)[1] });
""")
    assert out["table"] == [["Mario", "6"], ["↳ Paper Mario", "2"], ["↳ Super Mario 64", "1"], ["↳ No subseries", "3"], ["Kirby", "6"]]
    assert out["subs"] == 3 and out["tiles"] == ["2", "1"]
    assert out["title"] == "Items by series" and out["items"] == [["Mario", "1"], ["↳ Paper Mario", "1"]]
    assert out["bars"] == [["Mario", False], ["Paper Mario", True], ["Super Mario 64", True], ["No subseries", True], ["Kirby", False]]


def test_chart_links_open_the_filtered_grid(tmp_path):
    """Bars, columns, cells and legend entries that stand for cards link to #/cards with the grid's filters, within the
    Stats page's series; cards keep opening the card; Others and No subseries link nowhere."""
    out = node(tmp_path, """
const cards = fixture(), A = analytics(cards);
cards[0].subseries = "Paper Mario";
A.cards.m1.changes = [{ ts: ts("2025-03-10T20:00:00Z"), stats: [[3, 3, 3, 2], [3, 3, 3, 3]], bal: "buff" }];
A.cards.k1.changes = [{ ts: ts("2025-03-12T20:00:00Z"), text: [0, 1], words: true }];
const r = render(cards, A), m = render(cards, A, "series=Mario"), first = (html, id) => hrefs(chart(html, id)).map(params);
out({ kinds: first(r.top, "kinds"), mario: first(m.top, "kinds"), series: first(r.top, "series").slice(0, 3),
  shapes: first(r.top, "shapes")[0], dist: first(r.top, "dist")[0], items: first(r.top, "itemtypes"), tiers: first(r.top, "tiers")[0],
  cd: first(r.top, "cooldowns")[0], tags: first(r.top, "tags"), authors: first(r.top, "authors"),
  cohorts: first(m.balance, "bal-cohorts").slice(0, 4), creep: first(r.balance, "bal-creep")[0], buffs: first(r.balance, "bal-buffs"), months: first(r.balance, "bal-months"),
  created: first(r.history, "hist-created").filter(p => p.q).slice(0, 3), legend: first(r.history, "hist-created").filter(p => !p.q), fresh: first(r.history, "hist-fresh")[0],
  outliers: [chart(r.top, "outliers").includes('data-card="m1"') || chart(r.top, "outliers").includes("data-card="), hrefs(chart(r.top, "outliers")).length],
  noSub: chart(r.top, "series").includes("No subseries") });
""")
    assert out["kinds"] == [{"kind": "creature"}, {"kind": "item"}]
    assert out["mario"] == [{"kind": "creature", "series": "Mario"}, {"kind": "item", "series": "Mario"}]
    assert out["series"] == [{"kind": "creature", "series": "Mario"}, {"kind": "creature", "series": "Mario", "subseries": "Paper Mario"},
                             {"kind": "creature", "series": "Kirby"}]
    minions = "creature,creatures,minion"
    assert out["shapes"] == {"kind": minions, "shape": "hp"} and out["dist"] == {"kind": minions, "hp": "2-2"}      # HP 0 and 1: no cards, no link
    assert out["items"] == [{"kind": "item", "sub": "heal"}, {"kind": "item", "sub": "heal", "stars": "1"}, {"kind": "item", "sub": "heal"}]
    assert out["tiers"] == {"kind": minions, "stars": "2", "hp": "0-99"} and out["cd"] == {"stars": "1", "cd": "3CD+"}       # Potion's (1/3CD)
    assert out["tags"] == [{"tag": "1CD"}, {"tag": "2GCD"}, {"tag": "1/3CD"}] and out["authors"] == [{"author": "Ann"}, {"author": "Bob"}]
    assert out["cohorts"] == [{"bal": "top", "series": "Mario"}, {"bal": "bottom", "series": "Mario"},      # the legend: in scope
                              {"cohort": "creature|2"}, {"cohort": "creature|3"}]                            # a row: the whole cohort
    assert out["creep"] == {"cohort": "any", "q": "created==2020"}
    assert out["buffs"] == [{"change": "buff"}, {"change": "nerf"}, {"month": "2025-03", "change": "buff"}]      # no nerf that month: no link
    assert out["months"] == [{"change": "text"}, {"change": "stats"}, {"change": "other"}, {"month": "2025-03"},
                             {"month": "2025-03", "change": "text"}, {"month": "2025-03", "change": "stats"}]
    assert out["created"] == [{"q": "created==2020"}, {"q": "created==2020", "author": "Ann"}, {"q": "created==2020", "author": "Bob"}]
    assert out["legend"] == [{"author": "Ann"}, {"author": "Bob"}] and out["fresh"] == {"fresh": "2020"}
    assert out["outliers"] == [True, 0] and out["noSub"]



def test_linked_cards_are_the_counted_ones(tmp_path):
    """The predicates the grid filters with (shapeOf, cooldownCols, tagsOf, freshYear, changedAs, flagged) find as many
    cards as the bars count cards: stat shapes, a buffs column, a recorded-changes segment (Pacific months), a cohort's top 10%."""
    out = node(tmp_path, """
const cards = fixture(), A = analytics(cards), byId = new Map(cards.map(c => [c.id, c]));
A.cards.m1.changes = [{ ts: ts("2025-04-01T03:00:00Z"), stats: [[3, 3, 3, 2], [3, 3, 3, 3]], bal: "buff" }];   // Mar 31 in Pacific time
A.cards.k1.changes = [{ ts: ts("2025-03-12T20:00:00Z"), text: [0, 1], words: true }, { ts: ts("2025-03-13T20:00:00Z"), text: [0, 1], tags: [["(2PT)", "(1CD)"]] }];
A.cards.k2.flags = ["top"];
const r = render(cards, A), creatures = cards.filter(c => c.type === "creature");
const shapes = rows(chart(r.top, "shapes")).map(([l, n]) => [+n, creatures.filter(c => st.shapeOf(c) === (l === "Tied" ? "tied" : l.toLowerCase())).length]);
out({ shapes, buff: cards.filter(c => bal.changedAs(A, c, "2025-03", "buff")).map(c => c.id), april: cards.filter(c => bal.changedAs(A, c, "2025-04", "")).length,
  text: cards.filter(c => bal.changedAs(A, c, "2025-03", "text")).map(c => c.id), any: cards.filter(c => bal.changedAs(A, c, "2025-03", "")).map(c => c.id),
  raised: cards.filter(c => bal.changedAs(A, c, "", "spd+")).map(c => c.id), top: cards.filter(c => bal.flagged(A, c, "top")).map(c => c.id),
  cd: cards.filter(c => c.stars === 1 && st.cooldownCols(c).some(x => x.key === "3CD+")).map(c => c.id), tag: cards.filter(c => st.tagsOf(c).includes("2GCD")).map(c => c.id),
  fresh: cards.filter(c => ins.freshYear(c) === "2021").length, freshBar: rows(chart(r.history, "hist-fresh")).find(x => x[0] === "2021")[1] });
""")
    assert out["shapes"] and all(n == m for n, m in out["shapes"])
    assert out["buff"] == ["m1"] and out["april"] == 0 and out["raised"] == ["m1"]
    assert out["text"] == ["k1"] and out["any"] == ["m1", "k1"] and out["top"] == ["k2"]          # k1's tags-only change: not counted
    assert out["cd"] == ["potion"] and out["tag"] == ["potion"] and out["fresh"] == int(out["freshBar"]) == 6


def test_links_and_counts_agree(tmp_path):
    """inCohort finds a cohort's n (|all: every card of the kind; any: every member), not a creature without stats;
    a type listed twice on a card counts once; a segment under 1 unit tall isn't a link (the column is)."""
    out = node(tmp_path, """
const cards = fixture(), A = analytics(cards);
cards.push({ ...cards[0], id: "virus", name: "Virus", hp: null, def: null, atk: null, spd: null });
cards[0].types = ["panda", "panda"]; cards[1].types = ["panda"];
const types = rows(chart(render(cards, A).top, "types"));
const svg = st.stacked(["a", "b"], [{ key: "x", label: "X", color: "red" }, { key: "y", label: "Y", color: "blue" }], (c, k) => (k === "x" ? 1000 : c === "a" ? 1 : 0),
  { href: (c, k) => `#/cards?c=${c}${k ? `&k=${k}` : ""}` });
out({ n: ["creature|2", "creature|all", "any"].map(k => cards.filter(c => bal.inCohort(A, c, k)).length), types,
  links: [...svg.matchAll(/href="([^"]+)"/g)].map(m => m[1].replace(/&amp;/g, "&")) });
""")
    assert out["n"] == [6, 12, 12] and out["types"] == [["panda", "2"]]
    assert out["links"] == ["#/cards?c=a", "#/cards?c=a&k=x", "#/cards?c=b", "#/cards?c=b&k=x"]
