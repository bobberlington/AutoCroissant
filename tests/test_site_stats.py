"""
The website's Stats page (TTSCardMaker site/assets/stats.js, balance.js, insights.js, in node), one smoke test (owner
2026-11-12): power creep. Loads the site from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone.
"""
import json
from subprocess import run as run_process

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
function render(cards, A, q = "", { banned = new Map(), distPage = {} } = {}) {
  const byId = new Map(cards.map(c => [c.id, c])), ctx = { cards, byId, banned, icons: {}, mentions: new Map(), distPage };
  const root = { innerHTML: "" }, p = new URLSearchParams(q);
  st.renderStats(root, ctx, p);
  return { top: root.innerHTML, balance: A ? bal.statsHtml(A, { cards, byId }, p) : "", history: ins.historyHtml(ctx, p) };
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
