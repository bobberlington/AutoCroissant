"""
The website editor's helpers (TTSCardMaker site/assets, in node): the tooltip's hide rules (tip.js: the stuck "Click to
edit" of 2026-10-23), drafts of unsaved edits (drafts.js), the Edit dialog's live preview (live.js), the stars slider
(stars.js) and the type palette (palette.js). Loads them from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone;
skips without node or a site before them (TTSCardMaker work/site2-editor).
"""
import json
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


# ---- tip.js
TIP_SETUP = """
const log = [];
const mk = () => tip.tipController({ fill: s => (s.attrs["data-card"] === "gone" ? false : log.push(`fill ${s.attrs.id}`)), show: () => log.push("show"), hide: () => log.push("hide") });
const shown = c => !!c.source;
const box = new El("div", { id: "box", "data-tip": "Click to edit" }, ["ability"]), span = new El("span"), outside = new El("p", { id: "out" });
ROOT.add(box.add(span), outside);
"""


def test_the_tooltip_goes_when_an_editor_takes_its_data_tip(tmp_path):
    """The bug: click the ability box, its editor opens (inline.js removes data-tip), the pointer leaves through the
    textarea, a box with no data-tip any more; the old rule (hide on leaving an element that has a tooltip) kept it."""
    out = node(tmp_path, TIP_SETUP + """
const c = mk(); c.over(span);
const before = shown(c), src = c.source?.attrs.id;
box.removeAttribute("data-tip"); span.remove(); const ta = new El("textarea"); box.add(ta);   // the editor opens
c.over(ta);                                        // moving inside it: nothing new to show
const inside = shown(c);
c.out(outside);                                    // leaving through the textarea
const left = shown(c);
const c2 = mk(); ta.remove(); box.add(span); box.setAttribute("data-tip", "Click to edit"); c2.over(span);
box.removeAttribute("data-tip"); const moved = c2.move();        // the pointer still over it: the next move hides it
process.stdout.write(JSON.stringify({ before, src, inside, left, moved, after: shown(c2), log }));
""", fake_dom=True)
    assert out["before"] and out["src"] == "box"
    assert out["inside"] is True and out["left"] is False
    assert out["moved"] is False and out["after"] is False
    assert out["log"].count("hide") == 2


def test_the_tooltip_goes_with_a_re_rendered_element_and_on_request(tmp_path):
    out = node(tmp_path, TIP_SETUP + """
const c = mk(); c.over(span); box.remove();       // openCard replaced the dialog
const gone = c.move();
const c2 = mk(); ROOT.add(box); c2.over(span); c2.hide();       // inline.js: an editor opened, or a commit re-renders
process.stdout.write(JSON.stringify({ gone, a: shown(c), b: shown(c2) }));
""", fake_dom=True)
    assert out == {"gone": False, "a": False, "b": False}


def test_the_tooltip_stays_while_the_pointer_moves_inside_and_follows_nested_ones(tmp_path):
    out = node(tmp_path, TIP_SETUP + """
const c = mk(), other = new El("b"), img = new El("img", { id: "img", "data-tip": "fire" }, ["tok"]);
box.add(other, img);
c.over(span); c.out(other); const stays = c.source?.attrs.id;   // child to child: the same tooltip
c.out(img); c.over(img); const nested = c.source?.attrs.id;      // into an icon with its own
c.out(box); const leftIcon = shown(c); c.over(box); const back = c.source?.attrs.id;
process.stdout.write(JSON.stringify({ stays, nested, leftIcon, back, moves: c.move() }));
""", fake_dom=True)
    assert out == {"stays": "box", "nested": "img", "leftIcon": False, "back": "box", "moves": True}


def test_tip_source_order_and_an_unknown_card(tmp_path):
    """A chart value (data-v) is its own source; a card link inside a data-tip box is the link (app.js's order); a card
    the data doesn't know (fill false) shows nothing and leaves the current tooltip alone."""
    out = node(tmp_path, TIP_SETUP + """
const chart = new El("div", {}, ["chart"]), g = new El("g", { id: "g", "data-v": "5", "data-card": "x" }), ref = new El("a", { id: "ref", "data-card": "gone" }, ["card-ref"]);
ROOT.add(chart.add(g)); box.add(ref);
const c = mk(); c.over(span); c.over(ref);
process.stdout.write(JSON.stringify({ g: tip.tipSource(g).attrs.id, ref: tip.tipSource(ref).attrs.id, none: tip.tipSource(outside), still: c.source?.attrs.id }));
""", fake_dom=True)
    assert out == {"g": "g", "ref": "ref", "none": None, "still": "box"}


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
    assert out["joined"] == {"at": 2000, "fields": {"ability": "Burns.", "hp": "7"}}
    assert out["whole"] == {"name": "G"}
    assert out["afterDrop"] == ["mini-doomer"] and out["left"] == ["b"] and out["emptied"] == []
    assert out["broken"] == []
    assert out["odd"] == {"at": 5, "fields": {"name": "ok", "types": ["fire"]}} and out["oddIds"] == ["c"]


def test_a_draft_the_card_already_shows_is_not_offered(tmp_path):
    """Never resurrect what the batch has: restorable() keeps only the fields the card shows differently now (text
    compared trimmed, without \\r; types as sets, sorted by rawOf); a field the card page can't tell (the ban row) stays."""
    out = node(tmp_path, """
const card = { name: "Goombella", hp: 7, types: ["wind", "fire"], ability: "Burns.\\r\\nTwice.", series: "Mario", webAliases: ["goom", "g"], hidden: true, creatures: null };
const now = Object.fromEntries(["name", "hp", "types", "ability", "series", "aliases", "hidden", "note"].map(f => [f, drafts.rawOf(card, f)]));
const draft = { name: "Goombella ", hp: "7", types: ["fire", "wind"], ability: "Burns.\\nTwice.", series: "Luigi", aliases: "goom, g", hidden: true, note: "", banned: true };
process.stdout.write(JSON.stringify({ now, restore: drafts.restorable(draft, now), typed: drafts.changed({ name: "A", hp: "7", types: ["fire"] }, { name: "B", hp: "7 ", types: ["fire"] }),
  names: drafts.fieldNames({ ability: 1, hp: 1, banned: 1, "ban-reason": 1 }), creatures: drafts.rawOf({ creatures: [{ hp: 3, def: 1, atk: 2, spd: null }, { hp: 4 }] }, "creatures") }));
""")
    assert out["now"] == {"name": "Goombella", "hp": "7", "types": ["fire", "wind"], "ability": "Burns.\nTwice.", "series": "Mario",
                          "aliases": "goom, g", "hidden": True, "note": ""}
    assert out["restore"] == {"series": "Luigi", "banned": True}
    assert out["typed"] == {"name": "A"}
    assert out["names"] == "Ability, HP, Ban"
    assert out["creatures"] == [["3", "1", "2", ""], ["4", "", "", ""]]


def test_the_offer_and_the_autosave_timer(tmp_path):
    out = node(tmp_path, """
const t0 = Date.UTC(2026, 9, 23, 18, 0);
const agos = [0, 20e3, 5 * 60e3, 3 * 3600e3, 3 * 86400e3].map(d => drafts.ago(t0 - d, t0));
const html = drafts.offerHtml({ name: "<b>" }, t0 - 120e3, t0);
let runs = 0; const l = drafts.later(() => runs++, 30);
l.run(); l.run(); await new Promise(r => setTimeout(r, 80)); const debounced = runs;
l.run(); l.flush(); const flushed = runs; l.flush(); const again = runs;
l.run(); l.cancel(); await new Promise(r => setTimeout(r, 60));
process.stdout.write(JSON.stringify({ agos, html, debounced, flushed, again, cancelled: runs, delay: drafts.DELAY }));
""")
    assert out["agos"] == ["just now", "just now", "5 min ago", "3 h ago", "3 days ago"]
    assert "Restore your unsaved edit?" in out["html"] and "Name · 2 min ago" in out["html"]
    assert 'data-draft="restore"' in out["html"] and 'data-draft="discard"' in out["html"] and "<b>" not in out["html"]
    assert (out["debounced"], out["flushed"], out["again"], out["cancelled"]) == (1, 2, 2, 2)
    assert 0 < out["delay"] <= 1000


def test_the_form_s_inputs_read_and_filled(tmp_path):
    """readValues: #ef-<field> inputs (text, checkbox), the type toggles sorted, creature rows or null; fillValues puts
    them back (types as toggles, creatures through edit.js's hook)."""
    out = node(tmp_path, """
const input = (id, value, type = "text") => ({ id, value, type, checked: value === true });
const els = [input("ef-name", "Goombella\\r"), input("ef-hp", "7"), input("ef-hidden", true, "checkbox"), input("ef-ability", "a\\r\\nb")];
const toggles = ["wind", "fire", "ice"].map((t, i) => ({ dataset: { type: t }, on: i < 2, attrs: {}, classList: { contains() { return this.o.on; }, toggle(c, v) { this.o.on = v; } }, setAttribute(k, v) { this.attrs[k] = v; } }));
for (const b of toggles) b.classList.o = b;
const box = { hidden: true };
const form = { querySelectorAll: s => (s.startsWith("input") ? els : s === "#ef-types [data-type]" ? toggles : s === "[data-type]" ? toggles : []),
  querySelector: s => (s === "#ef-types" ? { querySelectorAll: () => toggles } : s === "#ef-creatures-box" ? box : els.find(e => `#${e.id}` === s) || null) };
const read = drafts.readValues(form);
let rows = "none";
drafts.fillValues(form, { name: "G2", hidden: false, types: ["ice"], creatures: [["1", "2", "3", "4"]], gone: "x" }, { creatures: r => { rows = r; } });
process.stdout.write(JSON.stringify({ read, name: els[0].value, hidden: els[2].checked, on: toggles.map(b => b.on), pressed: toggles.map(b => b.attrs["aria-pressed"]), rows }));
""")
    assert out["read"] == {"name": "Goombella", "hp": "7", "hidden": True, "ability": "a\nb", "types": ["fire", "wind"], "creatures": None}
    assert out["name"] == "G2" and out["hidden"] is False
    assert out["on"] == [False, False, True] and out["pressed"] == ["false", "false", "true"]
    assert out["rows"] == [["1", "2", "3", "4"]]


# ---- live.js
def test_the_live_preview_card(tmp_path):
    """The published card with the form's values (unset fields as their PSD values), author null = from git; PSD
    pending lists the printed fields that differ from the PSD (types in any order and case are no change, back to the
    PSD clears the card's own); the published card is left as it was."""
    out = node(tmp_path, """
const b = { id: "g", name: "Goombella", stars: 2, types: ["wind", "fire"], hp: 4, def: 2, atk: 4, spd: 5, ability: "Old.", author: "Chestnut", pending: ["atk"], aliases: ["bot"], webAliases: [] };
const psd = f => ({ name: "Goombella", stars: 2, types: ["wind", "fire"], hp: 4, def: 2, atk: 4, spd: 5, ability: "Old." })[f];
const values = { name: "Goombella the Great", hp: 7, types: ["Fire", "wind"], atk: 4, note: "A ruling", aliases: ["goom"], hidden: true };
const p = live.previewCard(b, values, null, psd);
const q = live.previewCard(b, { creatures: [{ hp: 1, def: 2, atk: 3, spd: 4 }, { hp: 5, def: 6, atk: 7, spd: 8 }] }, "Bob", psd);
process.stdout.write(JSON.stringify({ p, q: { hp: q.hp, spd: q.spd, creatures: q.creatures.length, author: q.author, pending: q.pending }, b,
  changed: [...live.changedFields(b, values, undefined)].sort(), changedQ: [...live.changedFields(b, { creatures: [] }, "Bob")].sort() }));
""")
    p, b = out["p"], out["b"]
    assert p["name"] == "Goombella the Great" and p["hp"] == 7 and p["note"] == "A ruling" and p["author"] == "Chestnut"
    assert p["pending"] == ["name", "hp"]                     # atk back to the PSD's 4: no longer pending; types: same set
    assert p["aliases"] == ["bot", "goom"] and p["webAliases"] == ["goom"] and p["hidden"] is True
    assert b["name"] == "Goombella" and b["pending"] == ["atk"] and "note" not in b
    assert out["q"] == {"hp": 1, "spd": 4, "creatures": 2, "author": "Bob", "pending": ["atk"]}   # creatures: no PSD field (build_site.apply_edits)
    assert out["changed"] == ["aliases", "hidden", "name", "note", "stats"]   # types: the same set in another order and case
    assert out["changedQ"] == ["author", "stats"]


def test_the_live_preview_markup(tmp_path):
    out = node(tmp_path, """
const ctx = { render: (t, n) => `R(${t}|${n})`, statBars: c => `<div class="statbars">${c.hp}</div>`, iconFor: n => (n === "fire" ? { file: "icons/fire.webp" } : null),
  kindLabel: { creature: "Creature" }, image: c => `thumbs/${c.id}-360.webp` };
const p = { id: "g", name: "<Goom>", type: "creature", stars: 3, types: ["fire", "odd_one"], hp: 7, ability: "Burns.", note: "", pending: ["name", "hp"], author: "Chestnut" };
const html = live.previewHtml(p, new Set(["name", "stats"]), ctx);
const bare = live.previewHtml({ id: "x", name: "X", type: "creature" }, new Set(["ability"]), { ...ctx, image: () => "" });
process.stdout.write(JSON.stringify({ html, bare, pane: live.paneHtml(false), open: live.paneHtml(true), blocks: ["name", "hp", "creatures", "series", "zzz"].map(live.blockOf) }));
""")
    html = out["html"]
    assert '<h2 class="lp-name lp-changed">&lt;Goom&gt;</h2>' in html
    assert 'class="lp-name lp-changed"' in html and 'class="lp-stats lp-changed"' in html
    assert "R(Burns.|<Goom>)" in html                       # the renderer (text.js, escaping) gets the raw text and name
    assert "PSD pending: Name, HP" in html and "3-star" in html and "by Chestnut" in html
    assert 'src="icons/fire.webp"' in html and ">odd_one<" in html and 'src="thumbs/g-360.webp"' in html
    assert "No ability text." in out["bare"] and "lp-changed" in out["bare"] and "lp-img" not in out["bare"]
    assert 'aria-expanded="false"' in out["pane"] and 'aria-controls="lp-body"' in out["pane"] and 'class="lp-pane open"' in out["open"]
    assert out["blocks"] == ["h2", ".lp-stats", ".lp-stats", ".facts", None]


# ---- stars.js
def test_the_stars_slider(tmp_path):
    out = node(tmp_path, """
const keys = ["ArrowLeft", "ArrowDown", "ArrowRight", "ArrowUp", "Home", "End", "a", "Enter"];
const at3 = keys.map(k => stars.stepValue(3, k)), edges = [stars.stepValue(0, "ArrowLeft"), stars.stepValue(10, "ArrowRight"), stars.stepValue(4, "End", 5)];
const html = stars.starsHtml(3);
const el = new El("span"); for (let i = 0; i < 10; i++) el.add(new El("i"));
stars.paintStars(el, 1);
process.stdout.write(JSON.stringify({ at3, edges, html, lit: el.children.filter(i => i.classList.contains("on")).length, now: el.getAttribute("aria-valuenow"),
  text: el.getAttribute("aria-valuetext"), words: [0, 1, 4].map(stars.starWords), max: stars.MAX }));
""", fake_dom=True)
    assert out["at3"] == [2, 2, 4, 4, 0, 10, None, None]
    assert out["edges"] == [0, 10, 5]
    html = out["html"]
    assert 'role="slider"' in html and 'aria-valuenow="3"' in html and 'aria-valuemax="10"' in html and 'aria-valuetext="3 stars"' in html
    assert html.count("<i ") == 10 and html.count('class="on"') == 3 and 'data-inline="stars"' in html and 'tabindex="0"' in html
    assert (out["lit"], out["now"], out["text"]) == (1, "1", "1 star")
    assert out["words"] == ["No stars", "1 star", "4 stars"] and out["max"] == 10


# ---- palette.js
ICONS = """
const icons = { fire: { file: "icons/fire.webp", name: "fire", kind: "type" }, flame: { file: "icons/fire.webp", name: "fire", kind: "type" },
  field: { file: "icons/field.webp", name: "field", kind: "type" }, defensiveminion: { file: "icons/dm.webp", name: "defensive_minion", kind: "type" },
  water: { file: "icons/water.webp", name: "water", kind: "type" }, wetness: { file: "icons/water.webp", name: "water", kind: "type" },
  wet: { file: "icons/water.webp", name: "water", kind: "type" }, wetland: { file: "icons/wetland.webp", name: "wetland", kind: "type" },
  star3: { file: "icons/star3.webp", name: "star3", kind: "star" }, ice: { file: "icons/ice.webp", name: "ice", kind: "type" } };
const vocab = palette.typeVocab(icons), names = list => list.map(e => e.name);
"""


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


def test_the_type_palette_order_toggles_and_search_switch(tmp_path):
    """No query: the card's types first; a toggle keeps the list sorted (as All fields lists them); useSearch() swaps
    in another search (the token picker's, once on main) and back."""
    out = node(tmp_path, ICONS + """
const first = names(palette.paletteOrder(vocab, ["water", "fire"], ""));
const t = palette.toggleType(palette.toggleType(["wind", "fire"], "ice"), "fire");
palette.useSearch((v, s) => v.filter(e => e.name === "water" && s === "zz"));
const swapped = names(palette.paletteOrder(vocab, [], "zz"));
palette.useSearch(null);
const back = names(palette.paletteOrder(vocab, [], "zz"));
const chips = palette.chipsHtml([{ name: "a<b", file: "x.webp" }], ["a<b"]);
const html = palette.paletteHtml(vocab, ["fire"], "zz");
process.stdout.write(JSON.stringify({ first, t, swapped, back, chips, html, words: [palette.typesWords([]), palette.typesWords(["defensive_minion", "fire"])],
  row: palette.rowHtml(["fire", "nope"], vocab) }));
""")
    assert out["first"] == ["fire", "water", "defensive_minion", "field", "ice", "wetland"]
    assert out["t"] == ["ice", "wind"]
    assert out["swapped"] == ["water"] and out["back"] == []
    assert 'aria-pressed="true"' in out["chips"] and "a&lt;b" in out["chips"] and "a<b" not in out["chips"]
    assert "No type matches “zz”" in out["html"] and 'type="search"' in out["html"] and "1 type: fire" in out["html"]
    assert out["words"] == ["No types", "2 types: defensive minion, fire"]
    assert 'alt="fire"' in out["row"] and ">nope<" in out["row"]


def test_the_palette_takes_the_token_picker_s_search(tmp_path):
    """pickerSearch: the token picker's suggest() as the palette's search (the switch once tokens.js is on main): asked
    for a plain icon of the typed text in the picker's whole vocabulary (read at each search), its plain type rows kept
    (no stars, no variants, none the palette lacks), in its order."""
    out = node(tmp_path, ICONS + """
const asked = [], all = [{ name: "fire" }, { name: "star3" }];
let reads = 0;
const suggest = (open, v) => { asked.push([open, v.length]); return [
  { token: "[water]", kind: "type", variant: "" }, { token: "[star3]", kind: "star", variant: "" }, { token: "[fire:glow]", kind: "type", variant: "glow" },
  { token: "[fire]", kind: "type", variant: "" }, { token: "[gone]", kind: "type", variant: "" }]; };
const search = palette.pickerSearch(suggest, () => { reads++; return all; });
const got = names(search(vocab, "wa"));
palette.useSearch(search);
const ordered = names(palette.paletteOrder(vocab, [], "wa")), none = names(palette.paletteOrder(vocab, ["ice"], ""));
palette.useSearch(null);
process.stdout.write(JSON.stringify({ got, asked, reads, ordered, none }));
""")
    assert out["got"] == ["water", "fire"] and out["ordered"] == ["water", "fire"]
    assert out["asked"][0] == [{"type": "icon", "base": "wa", "variant": None}, 2] and out["reads"] == 2
    assert out["none"][0] == "ice" and len(out["none"]) == 6           # no query: the palette's own order, not the search


@pytest.mark.skipif(not (ASSETS / "tokens.js").is_file(), reason=f"no tokens.js (the token picker) in {ASSETS}")
def test_the_palette_with_the_real_token_picker(tmp_path):
    """With tokens.js on the site: its ranking through pickerSearch finds types by prefix and with a slip."""
    out = node(tmp_path, ICONS + f"""
const tokens = await import({json.dumps((ASSETS / "tokens.js").as_uri())});
const search = palette.pickerSearch(tokens.suggest, () => tokens.vocabulary(icons, ["[fire] [fire] [field]"]));
process.stdout.write(JSON.stringify({{ fi: names(search(vocab, "fi")), frie: names(search(vocab, "frie")), wet: names(search(vocab, "wet")) }}));
""")
    assert out["fi"] == ["fire", "field"]                               # the most used first
    assert out["frie"][0] == "fire" and out["wet"][0] == "water"
