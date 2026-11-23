"""
A card dialog's close (X, Esc) in a browser without the Navigation API (Safari before 26.2, Firefox before 147): it goes
as many entries Back as the card entries opened over the view (arrows too), so the history doesn't grow; a card from a
link, or a chain cut from the history (50 entries), closes as before. One history call per card step (Safari allows 100
in 10 s). app.js syncUrl(), route() and actions.close in node, with a fake history. Reads $AC_SITE_SRC.
"""
import json
from subprocess import run as run_process

from conftest import SITE_SRC

APP = SITE_SRC / "site" / "assets" / "app.js"

PRELUDE = """const window = {}, tasks = [], modal = { hidden: true }, view = { replaceChildren() {} }, pageH1 = null;
const setTimeout = (f, ms) => { tasks.push({ f, ms }); tasks.sort((a, b) => a.ms - b.ms); };
const flush = () => { while (tasks.length) tasks.shift().f(); };
const H = { e: [{ url: "#/cards", state: null }], i: 0, calls: 0 };
const history = {
  get state() { return H.e[H.i].state; }, get length() { return H.e.length; },
  replaceState(s, _, url) { H.calls++; H.e[H.i] = { url: url ?? H.e[H.i].url, state: s && JSON.parse(JSON.stringify(s)) }; },
  go(n) { const j = H.i + n; if (j >= 0 && j < H.e.length) setTimeout(() => { H.i = j; route({ type: "hashchange", newURL: H.e[j].url }); }, 0); },
};
const location = { get hash() { return H.e[H.i].url; }, get href() { return H.e[H.i].url; },
  set hash(h) { H.e.splice(H.i + 1); H.e.push({ url: h, state: null }); if (H.e.length > 50) H.e.shift(); H.i = H.e.length - 1; route({ type: "hashchange", newURL: h }); } };
function parseHash(h = location.hash) { const [n, a = ""] = h.slice(2).split("/"); return { name: n, arg: a, params: new URLSearchParams() }; }
const hashOf = r => `#/${r.name}`, ROOT = "", shareUrl = id => `#/card/${id}`, staticDeck = () => null;
const S = { data: {}, byId: { has: () => true }, renderedHash: null, listHash: "#/cards" }, $ = () => null, viewSig = () => 1;
const hideTip = () => {}, aheadDrop = () => {}, cardOf = () => true, listOf = () => [], markTab = () => {}, setTitle = () => {}, settle = () => {};
const openCard = () => { modal.hidden = false; }, closeModal = () => { modal.hidden = true; }, turnPlace = { closed() {} };
let soonFor = null, listAt = null, rulesAt = "";
function renderView(r, hash) { S.renderedHash = S.listHash = hash; S.viewSig = viewSig(); }
"""

SCRIPT = """
const out = {}, at = () => ({ i: H.i, n: H.e.length, hash: location.hash, open: !modal.hidden });
const close = () => { actions.close(); flush(); };
route();                                                 // the grid
const calls = H.calls;
for (let k = 0; k < 5; k++) { location.hash = `#/card/c${k}`; location.hash = `#/card/c${k}x`; location.hash = `#/card/c${k}y`; close(); }
out.arrows = at(); out.calls = H.calls - calls;           // 5 cards, 2 arrow steps each
location.hash = "#/card/a"; location.hash = "#/card/b"; history.go(-1); flush(); location.hash = "#/card/c"; close();
out.branch = at();                                       // Back, then another card
for (let k = 0; k < 56; k++) location.hash = `#/card/p${k}`;
actions.close(); out.long = at(); flush();               // more than the history keeps: as before, at once
for (let k = 0; k < 55; k++) location.hash = `#/card/q${k}`;
history.go(-10); flush(); actions.close(); out.cutPending = at(); flush(); out.cut = at();   // the view cut from the history
H.e = [{ url: "#/card/l", state: null }]; H.i = 0; S.renderedHash = null; route();
location.hash = "#/card/m"; close(); out.link = at();    // from a link: as before
process.stdout.write(JSON.stringify(out));
"""


def test_close_goes_back_without_the_navigation_api(tmp_path):
    app = APP.read_text(encoding="utf-8")
    at = app.index("function syncUrl(r")
    decl = app[at:app.index("\n}\n", at) + 3] + app[app.index("let popped = null;"):app.index("function route(e) {")]
    at = app.index("function route(e) {")
    route = app[at:app.index("\n}\n", at) + 3]
    at = app.index('  "close": () => {')
    close = app[at:app.index("\n  },\n", at) + 5]
    script = tmp_path / "close.mjs"
    script.write_text(PRELUDE + decl + route + "const actions = {\n" + close + "};\n" + SCRIPT, encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    got = json.loads(done.stdout)
    assert got["arrows"] == {"i": 0, "n": 4, "hash": "#/cards", "open": False}         # was 21 entries, 20 Back
    assert got["calls"] == 15                                                            # one replaceState per card step
    assert got["branch"] == {"i": 0, "n": 3, "hash": "#/cards", "open": False}
    assert got["long"] == {"i": 49, "n": 50, "hash": "#/cards", "open": False}         # a page added
    assert got["cutPending"]["open"] and got["cut"] == {"i": 40, "n": 41, "hash": "#/cards", "open": False}
    assert got["link"] == {"i": 2, "n": 3, "hash": "#/cards", "open": False}
