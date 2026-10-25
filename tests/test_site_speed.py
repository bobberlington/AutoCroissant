"""
The website's speed pass (TTSCardMaker site/assets and build/build_site.py, 2026-10-24): the cached date formatters print
what toLocaleDateString printed, every module is preloaded, the faster mention scan agrees with render(), and the build's
image preload hints match the pictures app.js asks for (the grid's first ones on index.html, the 720 px one on a card's
share page). Loads the site from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without node or before the
pass (text.js dateFmt).
"""
import json
import re
import shutil
from os import environ
from os.path import expanduser
from pathlib import Path
from subprocess import run as run_process

import pytest

from conftest import CLONE
from test_site_build import _build, bs, repo  # noqa: F401  (fixtures)

SITE = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE))) / "site"
ASSETS = SITE / "assets"
TEXT_JS = ASSETS / "text.js"
pytestmark = pytest.mark.skipif(not (shutil.which("node") and TEXT_JS.is_file() and "dateFmt" in TEXT_JS.read_text(encoding="utf-8")),
                                reason=f"no node, or no text.js with dateFmt at {TEXT_JS}")


def node(tmp_path, body: str, tz: str = "UTC"):
    """Run body with text.js (T) and history.js (H) imported; its stdout as JSON."""
    script = tmp_path / "probe.mjs"
    uri = lambda name: json.dumps((ASSETS / name).as_uri())
    script.write_text(f"const T = await import({uri('text.js')}), H = await import({uri('history.js')});\n{body}\n", encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60, env={**environ, "TZ": tz})
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


# every options object the site formats dates with (app.js day, history.js, balance.js, changes.js, stats.js, profile.js)
OPTIONS = [{"year": "numeric", "month": "short", "day": "numeric", "timeZone": "UTC"},
           {"year": "numeric", "month": "short", "day": "numeric", "timeZone": "America/Los_Angeles"},
           {"year": "numeric", "month": "short", "timeZone": "America/Los_Angeles"},
           {"month": "short", "year": "2-digit", "timeZone": "UTC"},
           {"weekday": "long", "year": "numeric", "month": "short", "day": "numeric", "timeZone": "America/Los_Angeles"}]


@pytest.mark.parametrize("tz", ["UTC", "America/Los_Angeles", "Asia/Kolkata"])
def test_cached_formatters_print_as_tolocaledatestring(tmp_path, tz):
    """500 times from 2015 to 2031 (month ends, Pacific midnights and DST switches among them), in three local zones."""
    out = node(tmp_path, f"""
const opts = {json.dumps(OPTIONS)}, bad = [];
let x = 12345;
const ts = Array.from({{ length: 500 }}, () => (x = (x * 1103515245 + 12345) % 2147483648, 1420070400 + x % 504921600));
ts.push(1710061200, 1710064800, 1730624400, 1730628000, 1704095999, 1704096000);   // DST and New Year in Pacific time
for (const o of opts) {{ const f = T.dateFmt(o); for (const t of ts) {{ const d = new Date(t * 1000); if (f(d) !== d.toLocaleDateString(undefined, o)) bad.push([o, t]); }} }}
for (const t of ts) {{
  const d = new Date(t * 1000);
  if (H.date(t) !== d.toLocaleDateString(undefined, {{ year: "numeric", month: "short", day: "numeric", timeZone: H.TZ }})) bad.push(["date", t]);
  if (H.month(t) !== d.toLocaleDateString(undefined, {{ year: "numeric", month: "short", timeZone: H.TZ }})) bad.push(["month", t]);
}}
process.stdout.write(JSON.stringify({{ n: ts.length, bad: bad.slice(0, 5), sample: [H.date(ts[0]), H.date(0), H.month(ts[1])] }}));
""", tz)
    assert out["bad"] == [], out
    assert out["sample"][1] == ""                 # no time: no date, as before


def test_every_module_is_preloaded():
    """Each module app.js loads (static imports, any depth) has its modulepreload line: changes.js came late (F1)."""
    index = (SITE / "index.html").read_text(encoding="utf-8")
    preloaded = set(re.findall(r'<link rel="modulepreload" href="assets/([\w-]+\.js)">', index))
    need, todo = set(), ["app.js"]
    while todo:
        name = todo.pop()
        if name in need:
            continue
        need.add(name)
        todo += re.findall(r'^import [^;]*?from "\./([\w-]+\.js)";', (ASSETS / name).read_text(encoding="utf-8"), re.M)
    assert need - preloaded == set()
    assert "changes.js" in need


def test_mentions_agree_with_render(tmp_path):
    """mentions() skips the icons' HTML and tries a word's plural stems once each: the same cards as render()'s links."""
    out = node(tmp_path, r"""
const C = (id, name, type = "creature", key = name) => ({ id, name, key, type });
const cards = [C("drone", "Drone"), C("cabbage", "Cabbage", "item"), C("party", "Party!", "item"), C("wall", "The Wall By Ech", "creature", "Blockade"),
  C("imp", "Fire Imp"), C("fist", "Copper Fists", "item"), C("burn", "Burn", "debuff"), C("es", "Es"), C("bus", "Bus")];
const R = T.makeRenderer({ cards, icons: { fire: { file: "icons/fire.webp" }, drone: { file: "icons/drone.webp" } }, variants: {} });
const texts = ["Summon two Drones, then eat the Cabbages.", "(1CD) Throw a Party at [fire] Fire Imps and The Wall By Ech's friends.",
  "[drone] is an icon, not a card; Drone is.", "Gain a Copper Fist. Inflict Burn. Es and Buses.", "[fire]Fire Imp[fire] (2U) [nothing]Cabbage",
  "PARTY: the Blockade holds.", "Cabbagees and Droness"];
const rows = texts.map(t => { const links = [...R.render(t).matchAll(/data-card="([^"]+)"/g)].map(m => m[1]); return [[...new Set(links)].sort(), [...R.mentions(t)].sort()]; });
process.stdout.write(JSON.stringify(rows));
""")
    for links, mentions in out:
        assert links == mentions
    assert out[0][1] == ["cabbage", "drone"]
    assert "imp" in out[1][1] and "party" in out[1][1]


def test_grid_preload_picks_the_first_tiles(bs):  # noqa: F811
    """index.html's hints: the first four visible cards with a picture in Name order (the grid's default), as a script
    that adds them only for the plain grid, with app.js's srcset and the saved tile size."""
    cards = [{"id": i, "name": n, "img": img, **extra} for i, n, img, extra in [
        ("z", "zeta", "z1", {}), ("a", "Alpha", "a1", {}), ("h", "Aardvark", "h1", {"hidden": True}), ("n", "Beta", None, {}),
        ("c", "charlie", "c1", {}), ("d", "Delta", "d1", {}), ("e", "echo", "e1", {})]]
    html = bs.grid_preload(cards)
    assert json.dumps(["a1", "c1", "d1", "e1"]) in html
    assert html.startswith("<script>") and "localStorage" in html and "imageSrcset" in html
    assert bs.grid_preload([{"id": "x", "name": "X"}]) == ""
    app = (ASSETS / "app.js").read_text(encoding="utf-8")
    # the hints and the tiles must ask for the same files, or the browser downloads both
    assert 'const img = (c, w) => (c.img ? `thumbs/${c.img}-${w}.webp` : "");' in app
    assert 'srcset="${img(c, 240)} 240w, ${img(c, 360)} 360w${big ?' in app and 'sizes="${css}px"' in app
    assert "css: { s: 160, m: 200, l: 270 }[size] || 200" in app
    assert "`thumbs/${id}-240.webp 240w, thumbs/${id}-360.webp 360w`" in html and "{ s: 160, l: 270 }" in html


def test_the_build_writes_the_preload_hints(bs, repo, tmp_path, monkeypatch):  # noqa: F811
    """The site's front page preloads the grid's first pictures; a card's share page its 720 px one (app.js placeholderOf
    shows it at once); no other page gets the grid's."""
    data = _build(bs, monkeypatch, repo, tmp_path / "s")
    dist = tmp_path / "s" / "dist"
    cards = sorted((c for c in data["cards.json"]["cards"] if c.get("img") and not c.get("hidden")), key=lambda c: c["name"].lower())
    index = (dist / "index.html").read_text(encoding="utf-8")
    assert json.dumps([c["img"] for c in cards][:4]) in index
    for page in ("404.html", "d/index.html", f"c/{cards[0]['id']}/index.html"):
        assert "imageSrcset" not in (dist / page).read_text(encoding="utf-8"), page
    for c in cards:
        page = (dist / "c" / c["id"] / "index.html").read_text(encoding="utf-8")
        assert f'<link rel="preload" as="image" href="thumbs/{c["img"]}-720.webp" fetchpriority="high">' in page
        assert (dist / "thumbs" / f"{c['img']}-720.webp").is_file()


def test_stats_numbers_and_pacific_days_are_unchanged(tmp_path):
    """stats.js fmt prints whole numbers under 1000 with String() only where the formatter would print the same; the
    Pacific day keys are kept per UTC hour (Pacific days start on the hour), DST switches included."""
    if "PLAIN" not in (ASSETS / "stats.js").read_text(encoding="utf-8"):
        pytest.skip("stats.js before the fmt fast path")
    out = node(tmp_path, f"""
globalThis.document = {{ documentElement: {{}} }};
const st = await import({json.dumps((ASSETS / "stats.js").as_uri())});
const NF = new Intl.NumberFormat(undefined, {{ maximumFractionDigits: 1 }}), bad = [];
const xs = [-0, 0, -0.04, 0.04, 999.94, 999.96, -999.96, 1000, -1000, 1e6, NaN, Infinity, 0.5, -0.5, 12.25, 7, -7];
let x = 7; for (let i = 0; i < 20000; i++) xs.push(((x = (x * 1103515245 + 12345) % 2147483648) / 2147483648 - 0.5) * 3000);
for (const v of xs) if (st.fmt(v) !== NF.format(Math.round(v * 10) / 10)) bad.push(v);
const ymd = new Intl.DateTimeFormat("en-US", {{ year: "numeric", month: "2-digit", day: "2-digit", timeZone: H.TZ }});
const ref = ts => {{ const p = Object.fromEntries(ymd.formatToParts(new Date(ts * 1000)).map(x => [x.type, x.value])); return `${{p.year}}-${{p.month}}-${{p.day}}`; }};
for (let ts = 1704000000; ts < 1735700000; ts += 1777) if (H.dayOf(ts) !== ref(ts) || H.dayOf(ts + 0.5) !== ref(ts + 0.5)) bad.push(ts);
process.stdout.write(JSON.stringify({{ bad: bad.slice(0, 5), n: bad.length }}));
""", "Europe/Berlin")
    assert out["n"] == 0, out


def test_a_dialog_closes_onto_a_drawn_view():
    """route() closes a card without a redraw only while the view under it is current: nothing still loading (data that
    came while the dialog was open), the theme unchanged; Stats fills its sections under a dialog (S.renderedHash)."""
    app = (ASSETS / "app.js").read_text(encoding="utf-8")
    close = re.search(r"^  if \(hash === S\.renderedHash && !modal\.hidden(.*)\{ closeModal\(\);", app, re.M)
    assert close and "viewSig() === S.viewSig" in close[1] and '!$(".loading", view)' in close[1]
    sig = re.search(r"^function viewSig\(\) \{(.*)\}$", app, re.M)
    assert sig and "dataset.theme" in sig[1]
    stats = app[app.index("function renderStatsView"):app.index("function distPage")]
    assert "parseHash(S.renderedHash)" in stats and 'parseHash().name === "stats"' not in stats
