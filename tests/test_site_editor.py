"""
The website editor's helpers (TTSCardMaker site/assets, in node): the tooltip's hide rules (tip.js: the stuck "Click to
edit" of 2026-10-23), drafts of unsaved edits (drafts.js), the Edit dialog's live preview (live.js), the stars slider
(stars.js), the type palette (palette.js) and the dialogs' focus (app.js). Loads them from $AC_SITE_SRC (a
TTSCardMaker checkout) or the clone; skips without node or a site before them (TTSCardMaker work/site2-editor).
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

ASSETS = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE))) / "site" / "assets"
MODULES = ("tip", "drafts", "live", "stars", "palette")
pytestmark = pytest.mark.skipif(not (shutil.which("node") and all((ASSETS / f"{m}.js").is_file() for m in MODULES)),
                                reason=f"no node, or no {', '.join(m + '.js' for m in MODULES)} in {ASSETS}")

# Stand-in elements: enough of the DOM for tip.js (closest, matches, contains, isConnected, dataset) and stars.js
# (children, classList, setAttribute). Selectors: comma lists of tag.class[attr] parts, with one descendant space.
FAKE_DOM = r"""
class El {
  constructor(tag, attrs = {}, cls = []) { this.tag = tag; this.attrs = { ...attrs }; this.cls = new Set(cls); this.parent = null; this.children = [];
    const self = this; this.classList = { toggle(c, on) { if (on ?? !self.cls.has(c)) self.cls.add(c); else self.cls.delete(c); }, contains: c => self.cls.has(c) }; }
  add(...kids) { for (const k of kids) { k.parent = this; this.children.push(k); } return this; }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter(x => x !== this); this.parent = null; }
  get isConnected() { let e = this; while (e.parent) e = e.parent; return e === ROOT; }
  get dataset() { return Object.fromEntries(Object.entries(this.attrs).filter(([k]) => k.startsWith("data-")).map(([k, v]) => [k.slice(5).replace(/-(\w)/g, (_, c) => c.toUpperCase()), v])); }
  setAttribute(k, v) { this.attrs[k] = String(v); } getAttribute(k) { return this.attrs[k] ?? null; } removeAttribute(k) { delete this.attrs[k]; }
  one(part) { const m = /^([a-z]*)((?:\.[\w-]+)*)((?:\[[\w-]+\])*)$/.exec(part); if (!m) throw new Error(`selector ${part}`);
    return (!m[1] || m[1] === this.tag) && [...m[2].matchAll(/\.([\w-]+)/g)].every(x => this.cls.has(x[1])) && [...m[3].matchAll(/\[([\w-]+)\]/g)].every(x => x[1] in this.attrs); }
  matches(sel) { return sel.split(",").map(s => s.trim()).some(s => { const [a, b] = s.split(/\s+/); if (!b) return this.one(a); if (!this.one(b)) return false;
    for (let e = this.parent; e; e = e.parent) if (e.one(a)) return true; return false; }); }
  closest(sel) { for (let e = this; e; e = e.parent) if (e.matches(sel)) return e; return null; }
  contains(o) { for (let e = o; e; e = e.parent) if (e === this) return true; return false; }
}
const ROOT = new El("html");
"""


def node(tmp_path, body: str, fake_dom: bool = False):
    """Run body with the modules imported (tip, drafts, live, stars, palette); its stdout as JSON."""
    imports = "\n".join(f"const {m} = await import({json.dumps((ASSETS / f'{m}.js').as_uri())});" for m in MODULES)
    script = tmp_path / "probe.mjs"
    script.write_text(f"{imports}\n{FAKE_DOM if fake_dom else ''}\n{body}\n", encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


# ---- drafts.js
STORE = """
const mem = new Map(), storage = { getItem: k => mem.get(k) ?? null, setItem: (k, v) => mem.set(k, String(v)), removeItem: k => mem.delete(k) };
let now = 1000; const store = drafts.draftStore(storage, drafts.DRAFT_KEY, () => now);
"""


def test_drafts_per_card_and_field_in_local_storage(tmp_path):
    """One entry per card id (the stage's key) with each field's typed text; an inline field joins the card's draft,
    the Edit dialog's changed inputs replace it; dropping a field, a card, several; a broken or odd entry reads as none."""
    out = node(tmp_path, STORE + """
store.put("goombella", { ability: "Burns." }); now = 2000; store.put("goombella", { hp: "7" });
const joined = store.get("goombella");
store.put("goombella", { name: "G" }, true); const whole = store.get("goombella").fields;
store.put("mini-doomer", { note: "x" }); store.drop("goombella", ["name"]); const afterDrop = store.ids();
store.put("a", { name: "A" }); store.put("b", { name: "B" }); store.dropAll(["a", "mini-doomer"]); const left = store.ids();
store.put("b", {}, true); const emptied = store.ids();
mem.set(drafts.DRAFT_KEY, "{not json"); const broken = store.ids();
mem.set(drafts.DRAFT_KEY, JSON.stringify({ c: { at: 5, fields: { name: "ok", hp: 7, x: { y: 1 }, types: ["fire"] } }, d: { fields: [] }, e: 3 }));
const odd = store.get("c"), oddIds = store.ids();
process.stdout.write(JSON.stringify({ joined, whole, afterDrop, left, emptied, broken, odd, oddIds, key: drafts.DRAFT_KEY, stored: [...mem.keys()] }));
""")
    assert out["key"] == "edit-drafts"
    assert out["joined"] == {"at": 2000, "fields": {"ability": "Burns.", "hp": "7"}, "base": {}}
    assert out["whole"] == {"name": "G"}
    assert out["afterDrop"] == ["mini-doomer"] and out["left"] == ["b"] and out["emptied"] == []
    assert out["broken"] == []
    assert out["odd"] == {"at": 5, "fields": {"name": "ok", "types": ["fire"]}, "base": {}} and out["oddIds"] == ["c"]


# ---- palette.js
ICONS = """
const icons = { fire: { file: "icons/fire.webp", name: "fire", kind: "type" }, flame: { file: "icons/fire.webp", name: "fire", kind: "type" },
  field: { file: "icons/field.webp", name: "field", kind: "type" }, defensiveminion: { file: "icons/dm.webp", name: "defensive_minion", kind: "type" },
  water: { file: "icons/water.webp", name: "water", kind: "type" }, wetness: { file: "icons/water.webp", name: "water", kind: "type" },
  wet: { file: "icons/water.webp", name: "water", kind: "type" }, wetland: { file: "icons/wetland.webp", name: "wetland", kind: "type" },
  star3: { file: "icons/star3.webp", name: "star3", kind: "star" }, ice: { file: "icons/ice.webp", name: "ice", kind: "type" } };
const vocab = palette.typeVocab(icons), names = list => list.map(e => e.name);
"""


@pytest.mark.skipif("readToggles" not in (ASSETS / "palette.js").read_text(encoding="utf-8"), reason="a site before picking a type twice")
def test_a_type_picked_twice(tmp_path):
    """A press goes once, twice, off; the All fields toggles read a card's duplicate back, so it is no edit."""
    out = node(tmp_path, ICONS + """
const t = palette.toggleType, one = t([], "fire"), two = t(one, "fire");
const toggle = (type, on, count) => ({ dataset: count ? { type, count } : { type }, classList: { contains: () => on } });
const box = { querySelectorAll: () => [toggle("water", false), toggle("panda", true, "2"), toggle("fire", true)] };
process.stdout.write(JSON.stringify({ one, two, off: t(two, "fire"), read: palette.readToggles(box), words: palette.typesWords(["fire", "panda", "panda"]),
  chip: palette.chipsHtml(vocab.slice(2, 3), ["fire", "fire"]) }));
""")
    assert out["one"] == ["fire"] and out["two"] == ["fire", "fire"] and out["off"] == []
    assert out["read"] == ["fire", "panda", "panda"] and out["words"] == "3 types: fire, panda ×2"
    assert 'data-count="2"' in out["chip"]


def test_the_type_palette_vocabulary_and_search(tmp_path):
    """The types only (no star icons), one entry per icon with its aliases; the token picker's ranking: an exact name
    or alias, a name prefix, an alias prefix, then (2+ letters) inside a name or alias; one letter matches prefixes only."""
    out = node(tmp_path, ICONS + """
const q = s => names(palette.findTypes(vocab, s));
process.stdout.write(JSON.stringify({ vocab, all: q(""), fi: q("fi"), flame: q("flame"), f: q("f"), e: q("e"), et: q("et"), min: q("Defensive Min"), none: q("zz"),
  wet: q("wet"), wetn: q("wetn") }));
""")
    assert out["vocab"] == [{"name": "defensive_minion", "file": "icons/dm.webp", "keys": []}, {"name": "field", "file": "icons/field.webp", "keys": []},
                            {"name": "fire", "file": "icons/fire.webp", "keys": ["flame"]}, {"name": "ice", "file": "icons/ice.webp", "keys": []},
                            {"name": "water", "file": "icons/water.webp", "keys": ["wetness", "wet"]},
                            {"name": "wetland", "file": "icons/wetland.webp", "keys": []}]
    assert out["all"] == ["defensive_minion", "field", "fire", "ice", "water", "wetland"]
    assert out["wet"] == ["water", "wetland"] and out["wetn"] == ["water"]   # an exact alias before a name prefix
    assert out["fi"] == ["field", "fire"] and out["flame"] == ["fire"] and out["f"] == ["field", "fire"]
    assert out["e"] == [] and out["et"] == ["wetland", "water"] and out["min"] == ["defensive_minion"] and out["none"] == []



# ---- app.js dialogs
def test_the_page_under_a_dialog_takes_no_focus(tmp_path):
    """app.js: under an open dialog the skip link, header and #view are inert (a card page's from the start), set once
    focus is in the dialog (the opener known) and lifted in closeModal itself, so focus goes back to the opener."""
    app = (ASSETS / "app.js").read_text(encoding="utf-8")
    close = re.search(r"^function closeModal\(\) \{\n.*?^\}\n", app, re.S | re.M)[0]
    code = close + app[app.index("let opener = null;"):app.index("function showDialog(")]
    out = node(tmp_path, r"""
let cb, queued = false, hid = false, dialog;
const notify = () => { if (!queued) { queued = true; queueMicrotask(() => { queued = false; cb(); }); } }, tick = () => new Promise(r => setTimeout(r));
class N { constructor(name, parent) { this.name = name; this.parent = parent; this.on = false; this.dataset = {}; this.isConnected = true; }
  get inert() { return this.on; } set inert(v) { this.on = v; if (v && this.contains(document.activeElement)) document.activeElement = body; }   // focus inside drops
  contains(e) { for (; e; e = e.parent) if (e === this) return true; return false; }
  focus() { for (let e = this; e; e = e.parent) if (e.on) return; const was = document.activeElement; document.activeElement = this;
    if (modal.contains(this) && !modal.contains(was)) modal.focusin({ relatedTarget: was === body ? null : was }); }
  hasAttribute() { return false; } matches() { return false; } addEventListener(t, h) { this[t] = h; } }
const body = new N("body"), skip = new N("skip", body), header = new N("header", body), view = new N("view", body), tile = new N("tile", view);
const modal = new N("modal", body), toast = new N("toast", body), document = { activeElement: body, title: "", body: Object.assign(body, { style: {} }) };
Object.defineProperty(modal, "hidden", { get: () => hid, set: v => { hid = v; notify(); } });
Object.defineProperty(modal, "innerHTML", { set: v => { if (modal.contains(document.activeElement)) document.activeElement = body; dialog = v ? new N("dialog", modal) : null; notify(); } });
dialog = new N("dialog", modal);                                      // a card page: its dialog already up
const $ = sel => ({ ".skip": skip, "header.top": header, ".dialog": dialog })[sel] ?? null, nearby = { disconnect() {} };
globalThis.MutationObserver = class { constructor(f) { cb = f; } observe() {} };
const inert = () => [skip, header, view, toast].map(e => e.inert), r = {};
""" + code + r"""
r.page = inert(); closeModal(); r.closed = inert(); await tick();
tile.focus(); modal.innerHTML = "x"; modal.hidden = false; await tick();
r.open = inert(); r.at = document.activeElement.name; tile.focus(); r.tab = document.activeElement.name;
closeModal(); r.shut = inert(); await tick(); r.back = document.activeElement.name;
process.stdout.write(JSON.stringify(r));
""")
    assert out["page"] == out["open"] == [True, True, True, False]          # #toast (its Undo) stays reachable
    assert out["closed"] == out["shut"] == [False] * 4                      # at once: a focus() right after a close works
    assert (out["at"], out["tab"], out["back"]) == ("dialog", "dialog", "tile")
