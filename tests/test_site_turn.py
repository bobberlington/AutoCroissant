"""
A turned phone keeps the reader's place (TTSCardMaker site/assets/app.js turnPlace, in node, on a modelled cards grid):
the tile just under the bar, noted as the page scrolls, goes back to its spot when the width changes, though the
browser's own turn went to the top (short windows unpin the bar: its turn anchor); turning back lands on the same tile,
a reader's scroll notes another, a card dialog closed after a turn keeps it (not Back's older scroll), and a tile not
drawn yet (content-visibility) is never noted (2026-11-22).
"""
import json
from subprocess import run as run_process

from conftest import SITE_SRC

APP_JS = SITE_SRC / "site" / "assets" / "app.js"

GRID = r"""
let innerWidth = 390, innerHeight = 844, scrollY = 30000, now = 0, drawn = true, lag = 0;   // lag: undrawn tiles' guess
const on = {}, frames = [], window = { visualViewport: { scale: 1 } }, performance = { now: () => now };
const modal = { hidden: true };
const addEventListener = (k, f) => (on[k] ||= []).push(f), fire = k => (on[k] || []).forEach(f => f({}));
const requestAnimationFrame = f => frames.push(f);
const scrollBy = (x, d) => { scrollY = Math.max(0, scrollY + d); fire("scroll"); };
const run = () => { for (let i = 0; i < 120 && frames.length; i++) { now += 16; frames.splice(0).forEach(f => f()); } };
// the grid: 16 px sides, 14 px gaps, 5:8 pictures over a 60 px name; the bar 116 px, pinned unless the window is short
const cols = () => Math.max(2, Math.floor((innerWidth - 18) / 182)), tw = () => (innerWidth - 18) / cols() - 14;
const th = () => tw() * 1.6 + 60, row = () => th() + 14, TOP = 200;
const view = { contains: e => !!e && e !== document }, grid = { parentElement: view, isConnected: true,
  closest: () => null, style: { display: "grid" }, getBoundingClientRect: () => ({ top: TOP - scrollY, height: 1e5 }) };
const tiles = Array.from({ length: 600 }, (_, i) => ({ i, parentElement: grid, isConnected: true,
  style: { contentVisibility: "auto" }, closest: s => (s.includes(".tile") ? tiles[i] : null),
  firstElementChild: { checkVisibility: () => drawn },
  getBoundingClientRect: () => ({ top: TOP + lag + Math.floor(i / cols()) * row() - scrollY, height: th() }) }));
const bar = { getBoundingClientRect: () => ({ bottom: innerHeight > 450 ? 116 : 116 - scrollY }) };
const $ = s => (s === "header.top" ? bar : null);
const getComputedStyle = e => ({ position: "static", display: "block", contentVisibility: "visible", ...e.style });
const document = { elementFromPoint(x, y) {
  const d = y + scrollY - TOP - lag, r = Math.floor(d / row()), c = Math.floor((x - 16) / (tw() + 14));
  return d - r * row() > th() || x - 16 - c * (tw() + 14) > tw() ? grid : tiles[r * cols() + c];   // a gap: the grid
} };
const under = () => { const t = document.elementFromPoint(16 + tw() / 2, 124);   // the left column's tile under the bar
  return [t.i, Math.round(t.getBoundingClientRect().top)]; };
const turn = (w, h, y) => {                              // y: where the browser's own turn went
  innerWidth = w; innerHeight = h; scrollY = y; fire("scroll"); fire("resize"); run(); };
"""


def test_a_turned_phone_keeps_the_tile(tmp_path):
    app = APP_JS.read_text(encoding="utf-8")
    start = app.index("const turnPlace = (() => {")
    block = app[start:app.index("\n})();\n", start) + 6]
    # after the grid's --tile-h reset, before the Stats charts' redraw (it keeps the place put back)
    stats = app.index("if (!St || innerWidth === statsW) return;")
    assert app.index('view.style.removeProperty("--tile-h")') < start < stats
    script = tmp_path / "probe.mjs"
    script.write_text(GRID + block + r"""
const r = {};
fire("scroll"); run(); r.start = under();
turn(844, 390, 33); r.turned = [under(), scrollY > 1000];
turn(390, 844, 116); r.back = under();
turn(844, 390, 33); turn(390, 844, 116); r.again = under();
fire("touchstart"); scrollBy(0, 6000); run(); r.moved = under();      // the reader's: another tile
modal.hidden = false; turn(844, 390, 33); turn(390, 844, 116); r.dialog = under();   // turned under a card dialog
modal.hidden = true; turnPlace.closed(); scrollY += 1500; fire("scroll"); run(); r.closed = under();   // Back's scroll
drawn = false; lag = 3000; fire("touchstart"); scrollBy(0, 3000); frames.splice(0).forEach(f => f());   // placeholders
drawn = true; lag = 0; run(); r.guess = under();                         // drawn, no scroll event (no scroll anchoring)
turn(844, 390, 33); r.guessTurn = under();
process.stdout.write(JSON.stringify(r));
""", encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    r = json.loads(done.stdout)
    tile, top = r["start"]
    assert tile > 100 and r["turned"][1]                       # not the browser's top
    assert r["turned"][0][0] // 4 == tile // 4                 # landscape: the same row
    assert r["back"] == r["again"] == [tile, top]              # turned back: the same tile at the same spot
    assert r["moved"][0] > tile + 20 and r["dialog"] == r["closed"] == r["moved"]   # closed: not Back's older scroll
    assert r["guessTurn"][0] != r["moved"][0] and r["guessTurn"][0] // 4 == r["guess"][0] // 4   # noted once drawn
