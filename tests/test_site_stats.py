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


def test_distribution_counts_each_creature(tmp_path):
    """A card with several creatures counts once at each of their values, as its def=10-10 link meets any creature
    (Twin Emperors: DEF 4 and 10; HP 7 once)."""
    out = node(tmp_path, """
const twin = { ...C("Twin", "Mario", 3, [7, 4, 5, 6], "2020-05-01", "Ann"), creatures: [{ hp: 7, def: 4, atk: 5, spd: 6 }, { hp: 7, def: 10, atk: 4, spd: 4 }] };
const dist = cards => rows(chart(render(cards, null).top, "dist")).map(r => r.map(Number));
const a = dist(fixture()), b = dist([...fixture(), twin]);
out(b.map((r, i) => [r[0], ...r.slice(1).map((v, j) => v - (a[i]?.[j + 1] || 0))]).filter(r => r.slice(1).some(Boolean)));
""")
    assert out == [[4, 0, 1, 1, 1], [5, 0, 0, 1, 0], [6, 0, 0, 0, 1], [7, 1, 0, 0, 0], [10, 0, 1, 0, 0]]


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
    when Back/Forward or a reload restores a place, a redraw starts mid-page or a Balance/History link jumps (charts-all),
    so the place lands where it was. Balance and History below the first screen fill after it has painted, also when
    analytics.json lands first."""
    app, css = (ASSETS / "app.js").read_text(encoding="utf-8"), (ASSETS / "style.css").read_text(encoding="utf-8")
    first = app[app.index("function renderStatsView(r) {"):app.index("whenLoaded(loadCharts()")]
    assert "if (!S.analytics) loadAnalytics().catch(() => {});" in first
    assert "#view:not(.charts-all) .charts > section.chart:not(.drawn) { content-visibility: auto; contain-intrinsic-size: auto 420px; }" in css
    assert 'view.classList.toggle("charts-all", !top || !!S.placing);' in app and "S.placing = !!at && !at.m;" in app
    assert "const top = scrollY === 0 && !S.restoring," in app                     # !top: Back/Forward too
    jump = app[app.index('const scroll = e.target.closest("[data-scroll]");'):]
    assert jump.index('if (scroll.closest(".stats-jump")) view.classList.add("charts-all");') < jump.index("el?.scrollIntoView(")
    assert "done = true; S.placing = false;" in app and 's.classList.add("drawn")' in app and "setTimeout(drawRest, 300)" in app
    view = app[app.index("function renderStatsView(r) {"):app.index("// a turned phone (across 600 px)")]
    assert "if (S.analytics) fillSoon();" in view and "loadAnalytics().then(() => fillSoon()," in view   # landed first: after the paint too


def test_stats_after_a_card_has_its_data_ahead():
    """A card dialog reads its Balance from its own file, so Stats opened next would load analytics.json itself: once the
    card's Balance comes near, analytics.json is fetched at low priority after the dialog's on-screen pictures are in and
    the page is idle (never with Save-Data). Another card or page opening first cancels it; Stats (any real use) takes
    it over instead. A version picked or pinned in the dialog while it is in flight cancels it too, and it starts again
    once the compare's pictures are in (they come from another host: priority alone can't keep them first)."""
    app = (ASSETS / "app.js").read_text(encoding="utf-8")
    load = app[app.index("const loadAnalytics = (low = false) => {"):app.index("function analyticsSoon(id) {")]
    assert "if (!low) ahead = null;" in load and 'low ? { priority: "low", signal: ahead.signal } : undefined' in load
    assert "if (analyticsP === p) analyticsP = null;" in load                        # a cancelled one never clears a newer
    soon = app[app.index("function analyticsSoon(id) {"):app.index("const aheadDrop = () =>")]
    assert "navigator.connection?.saveData" in soon and "{ root: modal }" in soon and "!e.target.complete" in soon
    assert 'e.target.closest("#ver-compare")' in soon                              # a compare's pictures, on screen or not
    assert soon.index("if (wait.length)") < soon.index("requestIdleCallback") < soon.index("loadAnalytics(true)")
    assert "() => tick(true)" in soon and soon.index("if (!idle)") < soon.index("loadAnalytics(true)")   # looked again once idle
    assert 'if (e?.name !== "AbortError") soon = false;' in soon                     # a cancel's re-arm stays armed
    hold = app[app.index("const aheadHold = id =>"):]
    assert hold.index("ahead && soonFor === id") < hold.index("aheadDrop(); analyticsSoon(id);")
    assert "function showVersion(c, i) { drawVersion(c, i); aheadHold(c.id); }" in app
    assert 'nearby.on(["#card-balance", "#card-changes"], () => analyticsSoon(c.id));' in app
    route = app[app.index("function route(e) {"):app.index("function renderView(r, hash) {")]
    assert route.index("if (r.arg !== soonFor) aheadDrop();") < route.index("openCard(r.arg")
    assert route.index('if (r.name !== "stats") aheadDrop();') < route.index("renderView(r, hash);")


def test_rulebook_chapters_drawn_near():
    """The Rulebook's chapters below the first screen are drawn once near (content-visibility, scoped to #view: the
    rulebook editor's preview is untouched) or two at a time while the page is idle; every chapter at once before a
    contents jump, a same-book scroll, Back/Forward, a reload's place or a redraw mid-book (rb-all), and a deep link
    draws every chapter down to its target before it settles there, so nothing above the reader changes size later.
    Without scroll anchoring (WebKit), a jump past undrawn chapters draws them and keeps the place by hand."""
    app, css = (ASSETS / "app.js").read_text(encoding="utf-8"), (ASSETS / "style.css").read_text(encoding="utf-8")
    assert "#view:not(.rb-all) .rb-book > .rb-chapter:not(.drawn) { content-visibility: auto; contain-intrinsic-size: auto 3000px; }" in css
    rules = app[app.index("function renderRules(r) {"):app.index("// a deep link's place, again once")]
    assert "top = scrollY === 0 && !S.restoring" in rules and 'view.classList.toggle("rb-all", !top || !!S.placing);' in rules
    assert rules.index('view.classList.toggle("rb-all"') < rules.index("watchToc();")
    assert 'c.classList.add("drawn"); if (c.contains(to)) break;' in rules
    assert rules.index('c.classList.add("drawn"); if (c.contains(to)) break;') < rules.index("if (to) settleAt(to);")
    keep = rules[rules.index("const keep = () =>"):rules.index('addEventListener("scroll", keep, { passive: true });')]
    assert 'getComputedStyle(document.documentElement).overflowAnchor === "auto"' in keep
    assert keep.index('for (const c of above) c.classList.add("drawn");') < keep.index("scrollBy(0, at.getBoundingClientRect().top - y);")
    jump = app[app.index('const scroll = e.target.closest("[data-scroll]");'):]
    assert jump.index('if (scroll.closest(".rules")) view.classList.add("rb-all");') < jump.index("el?.scrollIntoView(")
    same = app[app.index("// another chapter of the book drawn: scroll"):]
    assert same.index('view.classList.add("rb-all");') < same.index("?.scrollIntoView();")
