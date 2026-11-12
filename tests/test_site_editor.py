"""
The website editor's helpers (TTSCardMaker site/assets, in node): drafts of unsaved edits (drafts.js), the type palette
(palette.js) and the dialogs' focus (app.js). Loads them from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone;
skips with a site before them.
"""
import json
import re
from subprocess import run as run_process

import pytest

from conftest import SITE_SRC

ASSETS = SITE_SRC / "site" / "assets"
MODULES = ("drafts", "palette")
pytestmark = pytest.mark.skipif(not all((ASSETS / f"{m}.js").is_file() for m in MODULES),
                                reason=f"no {', '.join(m + '.js' for m in MODULES)} in {ASSETS}")


def node(tmp_path, body: str):
    """Run body with the modules imported (drafts, palette); its stdout as JSON."""
    imports = "\n".join(f"const {m} = await import({json.dumps((ASSETS / f'{m}.js').as_uri())});" for m in MODULES)
    script = tmp_path / "probe.mjs"
    script.write_text(f"{imports}\n{body}\n", encoding="utf-8")
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
    the Edit dialog's changed inputs replace it; dropping a field, a card; a broken or odd entry reads as none."""
    out = node(tmp_path, STORE + """
const ids = () => ["goombella", "mini-doomer", "a", "b", "c", "d", "e"].filter(id => store.get(id));
store.put("goombella", { ability: "Burns." }); now = 2000; store.put("goombella", { hp: "7" });
const joined = store.get("goombella");
store.put("goombella", { name: "G" }, true); const whole = store.get("goombella").fields;
store.put("mini-doomer", { note: "x" }); store.drop("goombella", ["name"]); const afterDrop = ids();
store.put("a", { name: "A" }); store.put("b", { name: "B" }); store.drop("a"); store.drop("mini-doomer"); const left = ids();
store.put("b", {}, true); const emptied = ids();
mem.set(drafts.DRAFT_KEY, "{not json"); const broken = ids();
mem.set(drafts.DRAFT_KEY, JSON.stringify({ c: { at: 5, fields: { name: "ok", hp: 7, x: { y: 1 }, types: ["fire"] } }, d: { fields: [] }, e: 3 }));
const odd = store.get("c"), oddIds = ids();
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


def test_the_type_palette_vocabulary(tmp_path):
    """The types only (no star icons), one entry per icon, by name."""
    out = node(tmp_path, ICONS + "process.stdout.write(JSON.stringify(vocab));")
    assert out == [{"name": "defensive_minion", "file": "icons/dm.webp"}, {"name": "field", "file": "icons/field.webp"},
                   {"name": "fire", "file": "icons/fire.webp"}, {"name": "ice", "file": "icons/ice.webp"},
                   {"name": "water", "file": "icons/water.webp"}, {"name": "wetland", "file": "icons/wetland.webp"}]



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


def test_card_dialog_loads_wait_for_their_slots(tmp_path):
    """app.js nearby: each load runs once, when the first of its slots nears; two loads on one slot (history.json and
    analytics.json on #card-changes) both run; disconnect drops the pending ones."""
    app = (ASSETS / "app.js").read_text(encoding="utf-8")
    start = app.index("const nearby = (() => {")
    code = app[start:app.index("})();", start) + 5]
    out = node(tmp_path, r"""
let io; const modal = {}, $ = sel => sel, ran = [];
globalThis.IntersectionObserver = class { constructor(f) { io = this; this.f = f; this.seen = new Set(); }
  observe(el) { this.seen.add(el); } unobserve(el) { this.seen.delete(el); } disconnect() { this.seen.clear(); } };
const near = (...els) => io.f(els.filter(el => io.seen.has(el)).map(target => ({ target, isIntersecting: true })));
""" + code + r"""
const r = {};
nearby.on(["changes", "history"], () => ran.push("history"));
nearby.on(["balance", "changes"], () => ran.push("analytics"));
near("changes"); r.shared = [...ran]; near("balance", "history", "changes"); r.once = [...ran]; r.left = [...io.seen];
nearby.on(["balance", "changes"], () => ran.push("late")); near("balance", "changes"); r.both = [...ran];
nearby.on(["balance"], () => ran.push("dropped")); nearby.disconnect(); r.end = [...ran];
process.stdout.write(JSON.stringify(r));
""")
    assert out["shared"] == out["once"] == ["history", "analytics"] and out["left"] == []
    assert out["both"] == out["end"] == ["history", "analytics", "late"]
