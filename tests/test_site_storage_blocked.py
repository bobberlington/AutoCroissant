"""
Site storage blocked (TTSCardMaker site/assets, in node): reading localStorage throws a SecurityError, as in a browser
that blocks site data. deck.js: the hands still load, saving never throws, and the picked hand is kept for the page.
app.js: the single-key shortcuts' setting (WCAG 2.1.4) is on by default and, turned off, kept for the page.
"""
import json
import re
from subprocess import run as run_process

from conftest import SITE_SRC

DECK_JS = SITE_SRC / "site" / "assets" / "deck.js"
APP_JS = SITE_SRC / "site" / "assets" / "app.js"
BLOCKED = 'Object.defineProperty(globalThis, "localStorage", { get() { throw new DOMException("insecure", "SecurityError"); } });\n'


def _node(tmp_path, code):
    script = tmp_path / "probe.mjs"
    script.write_text(BLOCKED + code, encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_hands_work_with_storage_blocked(tmp_path):
    assert _node(tmp_path, f"""const D = await import({json.dumps(DECK_JS.as_uri())});
const decks = D.loadDecks(), first = D.activeDeck(decks).id;
const two = {{ id: "h2", name: "Hand 2", cards: [], field: null }};
decks.push(two); D.saveDecks(decks); D.setActive(two);
process.stdout.write(JSON.stringify({{ n: decks.length, first: first === decks[0].id, picked: D.activeDeck(decks).id }}));
""") == {"n": 2, "first": True, "picked": "h2"}


def test_single_key_shortcuts_turn_off(tmp_path):
    """In a card dialog 'r' and 'd' fire; turned off in the help dialog they do nothing, while Esc still closes it and
    the arrows still change cards; the buttons' key hints go (no-keys), and the header's."""
    app = APP_JS.read_text(encoding="utf-8")
    page = (SITE_SRC / "site" / "index.html").read_text(encoding="utf-8")
    html = {f"#{i}": re.search(rf'id="{i}"[^>]*?{a}="([^"]*)"', page)[1] for i, a in (("q", "placeholder"), ("random", "title"), ("keys", "title"))}
    at = app.index("let charKeys")
    keys = app[at:app.index('$("#account")', at)]
    start = app.index('document.addEventListener("keydown", e => {\n  const item')
    handler = app[start:app.index("\n});\n", start) + 5]
    out = _node(tmp_path, r"""
const log = [], el = { addEventListener() {}, closest: () => null, dataset: {}, tagName: "DIV" }, note = { hidden: true };
let keydown, dialog = "", hints = false;
const document = { activeElement: el, documentElement: { classList: { toggle: (c, on) => { hints = c === "no-keys" && on; } } },
  addEventListener: (t, h) => { keydown = h; } };
const modal = { hidden: false, contains: () => true }, location = { hash: "#/card/the-lich-king" };
const HTML = """ + json.dumps(html) + r""", tips = {}, hinted = sel => ({ ...el, getAttribute: () => HTML[sel], setAttribute: (a, v) => { tips[sel] = v; } });
const $ = (sel) => (sel === ".keys-off" ? note : sel.startsWith("[data-nav") ? { getAttribute: () => "#/card/next" } : sel in HTML ? hinted(sel) : sel === "header.top" ? el : null);
const S = { ready: true, selected: new Set(), byId: new Map([["the-lich-king", {}]]) };
const actions = { close: () => { log.push("close"); modal.hidden = true; }, "random-from-list": () => log.push("random") };
const parseHash = () => ({ name: "card", arg: "the-lich-king" }), editing = () => false, escField = () => false;
const toggleDeck = () => log.push("deck"), showDialog = h => { dialog = h; }, qInput = { focus: () => log.push("search"), select() {} };
""" + re.search(r"^const pref = .*$", app, re.M)[0] + "\n" + keys + handler + r"""
const press = key => keydown({ key, metaKey: false, ctrlKey: false, altKey: false, preventDefault() {} });
const r = { on: (press("r"), press("d"), log.splice(0)), tipsOn: { ...tips } };
showKeys(); r.box = /data-action="char-keys" checked/.test(dialog);
actions["char-keys"]({ checked: false }); showKeys();
r.unchecked = !/char-keys" checked/.test(dialog); r.note = !note.hidden; r.hints = hints; r.tipsOff = { ...tips };
press("r"); press("d"); press("/"); r.off = log.splice(0);
press("ArrowRight"); r.arrow = location.hash; press("Escape"); r.esc = log.splice(0);
process.stdout.write(JSON.stringify(r));
""")
    assert out["on"] == ["random", "deck"] and out["box"]                       # on by default (storage blocked: no value)
    assert out["unchecked"] and out["note"] and out["hints"] and out["off"] == []
    assert out["arrow"] == "#/card/next" and out["esc"] == ["close"]
    assert out["tipsOn"] == html and out["tipsOff"] == {"#q": "Search cards…", "#random": "Random card", "#keys": "Keyboard shortcuts"}
