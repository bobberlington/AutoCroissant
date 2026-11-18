"""
The website's Stats page (TTSCardMaker site/assets/stats.js, balance.js, insights.js, in node), one smoke test (owner
2026-11-12): power creep; and a card's Balance block for an auxiliary creature (analytics.py, owner D21) and for a stat
mistake put back the same day. Loads the site from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone.
"""
import importlib.util
import json
from subprocess import run as run_process
from types import SimpleNamespace

from conftest import SITE_SRC

SITE = SITE_SRC / "site"
ASSETS = SITE / "assets"
STATS_JS = ASSETS / "stats.js"

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
    time: { months: [], series: {} }, minCohort: 5, ...extra };
  for (const c of cards) if (c.type === "creature") A.cards[c.id] = { cohort: `creature|${c.stars}`, stats: [c.hp, c.def, c.atk, c.spd], flags: [], below: 0, above: 0 };
  return A;
}
function render(cards, A, q = "", { banned = new Map(), distPage = {}, narrow = false } = {}) {
  const byId = new Map(cards.map(c => [c.id, c])), ctx = { cards, byId, banned, icons: {}, mentions: new Map(), distPage, narrow };
  const root = { innerHTML: "" }, p = new URLSearchParams(q);
  st.renderStats(root, ctx, p);
  return { top: root.innerHTML, balance: A ? bal.statsHtml(A, ctx, p) : "", history: ins.historyHtml(ctx, p) };
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


def test_power_creep(tmp_path):
    """Hand-computed: 2020 average vs median 0, 2021 -0.2 (12 creatures); left out for items."""
    out = node(tmp_path, """
const cards = fixture(), A = analytics(cards), r = render(cards, A);
out({ creep: rows(chart(r.balance, "bal-creep")), item: render(cards, A, "kind=item").balance.includes("chart-bal-creep") });
""")
    minus = lambda s: s.replace("−", "-")
    assert [[minus(c) for c in row] for row in out["creep"]] == [["2020", "6", "0", "11.3"], ["2021", "6", "-0.2", "12.5"]]
    assert not out["item"]


def test_narrow_charts(tmp_path):
    """A phone's panel (narrow): every chart drawn at most 360 wide, so its labels read; desktop as before."""
    out = node(tmp_path, """
const cards = fixture(), A = analytics(cards), widths = r => [...new Set(Object.values(r).join("").match(/(?<=<svg viewBox="0 0 )\\d+/g).map(Number).filter(w => w > 100))].sort((a, b) => a - b);   // not the 22-wide tag rings
out({ wide: widths(render(cards, A)), narrow: widths(render(cards, A, "", { narrow: true })) });
""")
    assert out == {"wide": [320, 560, 1000], "narrow": [320, 340, 360]}


def test_before_2020_column_opens_its_cards(tmp_path):
    """Cards created per year: the 'Before 2020' column (a physical card photo's Created) opens only its cards, not
    2019's (search.js reads its link)."""
    out = node(tmp_path, f"""const {{ compile }} = await import({json.dumps((ASSETS / "search.js").as_uri())});
const cards = [...fixture(), C("Pep", "Mario", 1, [1, 1, 1, 1], "2019-01-01", "Ann", {{ createdBefore: 2020 }}),
  C("Old", "Mario", 1, [1, 1, 1, 1], "2019-05-02", "Ann")];
const html = chart(render(cards, null).history, "hist-created");
const q = {{}}; for (const m of html.matchAll(/href="([^"]+)" aria-label="([^"]+): show these cards"/g)) q[m[2]] = new URLSearchParams(m[1].replace(/&amp;/g, "&").split("?")[1]).get("q");
const names = k => cards.filter(compile(q[k])).map(c => c.name);
out({{ before: names("Before 2020"), y2019: names("2019") }});""")
    assert out == {"before": ["Pep"], "y2019": ["Old"]}


def site_analytics():
    spec = importlib.util.spec_from_file_location("site_analytics", SITE / "build" / "analytics.py")
    an = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(an)
    return an


def test_auxiliary_creature_by_printed_stars(tmp_path):
    """Owner D21: an auxiliary creature is compared with the creatures of its printed stars, counted only in all
    creatures (nobody else's numbers move, nor the 4★ list's cards); its Balance line has no 'no star level' sentence."""
    an = site_analytics()
    card = lambda id, kind, total: {"id": id, "key": id, "type": kind, "stars": 4, "hp": total - 3, "def": 1, "atk": 1, "spd": 1}
    plain, aux = [card(f"c{t}", "creature", t) for t in (10, 12, 14, 16, 18)], card("mett", "creatures", 20)
    A, without = an.build(plain + [aux], {}, {}, now=1e9), an.build(plain, {}, {}, now=1e9)
    mine = A["cards"].pop("mett")
    assert (mine["cohort"], mine["below"], mine["above"], mine["flags"]) == ("creature|4", 5, 0, ["top"])
    assert A["cards"] == without["cards"] and A["cohorts"]["creature|4"] == without["cohorts"]["creature|4"]
    assert A["cohorts"]["creature|all"]["n"] == 6
    A["cards"]["mett"] = mine
    out = node(tmp_path, f"""const A = {json.dumps(A)}, aux = {json.dumps(aux)}, plain = {json.dumps(plain[0])};
out({{ line: text(bal.balanceBlock(aux, A)), aux: ["creature|4", "creature|all", "any"].map(k => bal.inCohort(A, aux, k)),
  plain: bal.inCohort(A, plain, "creature|4") }});""")
    assert "in the top 10% of 4★ creatures: 5 of the 5 have a lower total" in out["line"] and "star level" not in out["line"]
    assert (out["aux"], out["plain"]) == ([False, True, True], True)   # the cohort= list holds the n it counts


def test_stat_mistake_put_back_within_a_day():
    """A stat line the next version puts back within a day (Tainted Cain's bars, 16 min) is no nerf and buff, nor a dip
    in the trend; put back two days later, it stays."""
    an = site_analytics()
    day, line = 86400, lambda *x: SimpleNamespace(**dict(zip(("hp", "defense", "attack", "speed"), x)))
    v = lambda t, *x: SimpleNamespace(timestamp=t, stats=line(*x), stars=4, card_type="creature")
    cards = [{"id": k, "key": k, "type": "creature", "stars": 4, "hp": 6, "def": 4, "atk": 3, "spd": 8} for k in ("quick", "slow")]
    old = {"quick": [v(0, 6, 4, 3, 7), v(day, 6, 4, 3, 5), v(day + 960, 6, 4, 3, 7)],
           "slow": [v(0, 6, 4, 3, 7), v(day, 6, 4, 3, 5), v(3 * day, 6, 4, 3, 7)]}
    stats = {k: v(30 * day, 6, 4, 3, 8) for k in old}
    A = an.build(cards, stats, old, now=40 * day)["cards"]
    bal = lambda k: [(r["ts"], r["bal"]) for r in A[k]["changes"]]
    totals = lambda k: [p["total"] for p in A[k]["trend"]]
    assert (bal("quick"), totals("quick")) == ([(30 * day, "buff")], [20, 21])
    assert (bal("slow"), totals("slow")) == ([(30 * day, "buff"), (3 * day, "buff"), (day, "nerf")], [20, 18, 20, 21])


def test_stats_open_starts_its_data_and_draws_charts_near():
    """Stats' first open starts analytics.json beside the charts chunks (as a link to it does); its charts below the first
    screen are drawn once near (content-visibility) or two at a time while the page is idle, and every chart at once
    when Back/Forward or a reload restores a place (charts-all), so the place lands where it was."""
    app, css = (ASSETS / "app.js").read_text(encoding="utf-8"), (ASSETS / "style.css").read_text(encoding="utf-8")
    first = app[app.index("function renderStatsView(r) {"):app.index("whenLoaded(loadCharts()")]
    assert "if (!S.analytics) loadAnalytics().catch(() => {});" in first
    assert "#view:not(.charts-all) .charts > section.chart:not(.drawn) { content-visibility: auto; contain-intrinsic-size: auto 420px; }" in css
    assert 'view.classList.toggle("charts-all", S.restoring || !!S.placing);' in app and "S.placing = !!at && !at.m;" in app
    assert "done = true; S.placing = false;" in app and 's.classList.add("drawn")' in app and "setTimeout(drawRest, 300)" in app
