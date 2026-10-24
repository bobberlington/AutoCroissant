"""
The website's editing helpers of round 3 (TTSCardMaker site/assets, in node and by text): the item type list (edit.js
ITEM_TYPES = the Items/ folders), drops on a filter-bar value or a pinned box (inline.js targetOf/dropOn, dragenter),
the click-away rule for the "+" editors, the name autofill (tokens.js suggestNames) and the pinned tray (tray.js).
Loads the site from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without node or a site before them.
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

SITE = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE))) / "site"
ASSETS = SITE / "assets"
EDIT_JS = ASSETS / "edit.js"
INLINE_JS = ASSETS / "inline.js"
pytestmark = pytest.mark.skipif(not EDIT_JS.is_file() or "ITEM_TYPES" not in EDIT_JS.read_text(encoding="utf-8"),
                                reason=f"no ITEM_TYPES in {EDIT_JS} (land TTSCardMaker work/r3-editing)")
needs_node = pytest.mark.skipif(not shutil.which("node"), reason="needs node")


def node(tmp_path: Path, body: str, **modules: Path):
    """Run body with each module imported under its name; its stdout as JSON."""
    script = tmp_path / "probe.mjs"
    imports = "".join(f"const {k} = await import({json.dumps(v.as_uri())});\n" for k, v in modules.items())
    script.write_text(f"{imports}{body}\n", encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


# ---- item 15: item type
@needs_node
def test_item_types_are_the_items_folders(tmp_path):
    items = SITE.parent / "Items"
    if not items.is_dir():
        pytest.skip(f"no {items}")
    got = node(tmp_path, "process.stdout.write(JSON.stringify(E.ITEM_TYPES));", E=EDIT_JS)
    assert got == sorted(p.name.lower() for p in items.iterdir() if p.is_dir())


def test_item_type_is_a_select():
    text = EDIT_JS.read_text(encoding="utf-8")
    assert '<select id="ef-subtype">' in text
    assert "dl-sub\"" not in text and 'id="ef-subtype" list=' not in text


# ---- item 18: drops
def test_drop_targets_take_dragenter_too():
    """Safari and Firefox drop only on an element that cancelled dragenter (Chrome is lenient); the tile drags, not its
    picture; the dragged ids are kept in memory, not only in dataTransfer."""
    text = INLINE_JS.read_text(encoding="utf-8")
    assert re.search(r'addEventListener\("dragenter", onDragOver\)', text)
    assert re.search(r'querySelector\("img"\)\?\.setAttribute\("draggable", "false"\)', text)
    assert "dragged = dragIds(" in text


@needs_node
def test_target_of_a_chip_and_a_box(tmp_path):
    out = node(tmp_path, """
const list = (drop, series) => ({ dataset: series == null ? { drop } : { drop, series } });
const chip = (value, l) => ({ dataset: { value }, closest: () => l });
const box = d => ({ dataset: { pin: "0", ...Object.fromEntries(Object.entries(d).map(([k, v]) => [`pin${k.toUpperCase()}`, v])) } });   // data-pin-k
process.stdout.write(JSON.stringify([I.targetOf(chip("Mario", list("series"))), I.targetOf(chip("Paper Mario", list("subseries", "Mario"))),
  I.targetOf(box({ k: "author", v: "bobberlington" })), I.targetOf(box({ k: "subseries", v: "Paper Mario", s: "Mario" }))]));""", I=INLINE_JS)
    assert out == [{"key": "series", "value": "Mario", "series": ""}, {"key": "subseries", "value": "Paper Mario", "series": "Mario"},
                   {"key": "author", "value": "bobberlington", "series": ""}, {"key": "subseries", "value": "Paper Mario", "series": "Mario"}]


# ---- item 3: the stuck "+" box
def test_inline_click_away_rules():
    """A click or tap away closes an inline editor even when no focus moves (a capture pointerdown); "+ stars" opens a
    fresh slider that turns back into "+ stars" when left at none (focus out, a click away, Esc)."""
    text = INLINE_JS.read_text(encoding="utf-8")
    assert re.search(r'document\.addEventListener\("pointerdown", away, true\)', text)
    assert 'sp.dataset.fresh = ""' in text
    assert re.search(r'\.star-pick\[data-fresh\]"\) && !e\.target\.contains\(e\.relatedTarget\)\) unfresh\(', text)
    assert 'row.outerHTML = ghost("stars")' in text
    assert re.search(r'Escape" && row\.hasAttribute\("data-fresh"\)', text)


# ---- item 17: the pinned tray
TRAY_JS = ASSETS / "tray.js"
needs_tray = pytest.mark.skipif(not TRAY_JS.is_file(), reason=f"no {TRAY_JS}")


@needs_node
@needs_tray
def test_read_pins_keeps_only_good_ones(tmp_path):
    long = "x" * 61
    pins = ([{"k": "series", "v": " Mario "}, {"k": "kind", "v": "item"}, {"k": "author", "v": ""}, {"k": "author", "v": long},
             {"k": "subseries", "v": "Paper Mario"}, {"k": "series", "v": "mario"}, {"k": "series", "v": "Ma rio"}, None, "x",
             {"k": "subseries", "v": "Paper Mario", "s": "Mario"}, {"k": "subseries", "v": "paper mario", "s": "Kirby"},
             {"k": "author", "v": 3}, {"k": "toString", "v": "x"}]
            + [{"k": "author", "v": f"A{i}"} for i in range(12)])
    out = node(tmp_path, f"""
process.stdout.write(JSON.stringify([T.readPins({json.dumps(json.dumps(pins))}), T.readPins("{{junk"), T.readPins(null), T.readPins('{{"k":"series"}}')]));""",
               T=TRAY_JS)
    assert out[0] == [{"k": "series", "v": "Mario"}, {"k": "subseries", "v": "Paper Mario", "s": "Mario"},
                      {"k": "subseries", "v": "paper mario", "s": "Kirby"}] + [{"k": "author", "v": f"A{i}"} for i in range(6)]
    assert out[1:] == [[], [], []]


@needs_node
@needs_tray
def test_add_and_remove_pins(tmp_path):
    out = node(tmp_path, """
const one = [{ k: "series", v: "Mario" }], nine = Array.from({ length: 9 }, (_, i) => ({ k: "author", v: `A${i}` }));
process.stdout.write(JSON.stringify([T.addPin(one, { k: "series", v: "mario " }), T.addPin(one, { k: "author", v: "Mario" }),
  T.addPin(nine, { k: "series", v: "Kirby" }), T.removePin(nine, 0).length, T.removePin(one, 5)]));""", T=TRAY_JS)
    assert out == [{"error": "Already pinned"}, {"pins": [{"k": "series", "v": "Mario"}, {"k": "author", "v": "Mario"}]},
                   {"error": "At most 9 pins"}, 8, [{"k": "series", "v": "Mario"}]]


@needs_node
@needs_tray
def test_number_keys_pick_a_pin(tmp_path):
    out = node(tmp_path, """
const two = [{ k: "series", v: "A" }, { k: "series", v: "B" }];
process.stdout.write(JSON.stringify(["1", "2", "3", "0", "a", "12"].map(k => T.pinOfKey(k, two))));""", T=TRAY_JS)
    assert out == [0, 1, None, None, None, None]


@needs_node
@needs_tray
def test_count_of_a_pin(tmp_path):
    """A subseries counts only in its series; case and spaces don't count."""
    cards = [{"series": "Mario", "subseries": "Paper Mario", "author": "Bob"}, {"series": "mario", "subseries": "paper  mario"},
             {"series": "Kirby", "subseries": "Paper Mario", "author": "bob"}, {"series": "Kirby"}]
    out = node(tmp_path, f"""
process.stdout.write(JSON.stringify([T.countPins([{{ k: "subseries", v: "Paper Mario", s: "Mario" }}, {{ k: "series", v: "Kirby" }},
  {{ k: "author", v: "BOB" }}, {{ k: "series", v: "Zelda" }}], {json.dumps(cards)}), T.countPins([], {json.dumps(cards)})]));""", T=TRAY_JS)
    assert out == [[2, 2, 2, 0], []]


@needs_node
@needs_tray
def test_tray_html(tmp_path):
    out = node(tmp_path, """
const pins = [{ k: "series", v: "Mario" }, { k: "subseries", v: "Paper <Mario>", s: "Mario" }, { k: "author", v: "Bob" }];
process.stdout.write(JSON.stringify([T.trayHtml([], []), T.trayHtml(pins, [42, 1, 0]), T.trayHtml(pins, [1, 1, 1], { form: { k: "subseries", v: "pa" }, series: "Mario", selecting: true }),
  T.trayHtml([], [], { form: { k: "series", v: "" } })]));""", T=TRAY_JS)
    empty, html, with_form, form_only = out
    assert empty == ""
    assert re.findall(r'data-pin="(\d)" data-pin-k="(\w+)" data-pin-v="([^"]*)"(?: data-pin-s="([^"]*)")?', html) == [
        ("0", "series", "Mario", ""), ("1", "subseries", "Paper &lt;Mario&gt;", "Mario"), ("2", "author", "Bob", "")]
    assert re.findall(r'<span class="kbd" aria-hidden="true">(\d)</span>', html) == ["1", "2", "3"]
    assert [m for m in re.findall(r'class="b-meta">([^<]*)<', html)] == ["42 cards", "1 card", "no cards yet"]
    assert 'aria-label="Series Mario, 42 cards. Drop cards here, or select cards and press 1"' in html
    assert ">Subseries · Mario<" in html and "<Mario>" not in html
    assert 'data-action="tray-add">+ Pin<' in html and "pin-form" not in html
    assert 'class="tray selecting"' in with_form and "+ Pin" not in with_form
    assert re.search(r'<input class="pin-name" data-suggest="subseries" data-series="Mario"[^>]*value="pa">', with_form)
    assert "<option value=\"subseries\" selected>" in with_form
    assert "pin-form" in form_only and "data-pin" not in form_only
    assert not re.search(r'\sdata-(v|tip)=', html)    # tip.js reads data-v as a chart value: no tooltip over the boxes


@needs_node
def test_a_subseries_drop_refuses_other_series(tmp_path):
    cards = [{"id": "a", "series": "mario"}, {"id": "b", "series": "Mario "}, {"id": "c", "series": "Kirby"}, {"id": "d"}]
    out = node(tmp_path, f"""
const r = E.splitBySeries({json.dumps(cards)}, "Mario");
process.stdout.write(JSON.stringify([r.ok.map(c => c.id), r.refused.map(c => c.id)]));""", E=EDIT_JS)
    assert out == [["a", "b"], ["c", "d"]]


@needs_tray
def test_the_tray_is_wired():
    """app.js imports tray.js, index.html preloads it, the keydown routes 1-9 through tray.dropKey before the tab
    shortcuts, and every stage goes through inline.dropOn."""
    app = (ASSETS / "app.js").read_text(encoding="utf-8")
    assert 'import * as tray from "./tray.js";' in app
    assert '<link rel="modulepreload" href="assets/tray.js">' in (SITE / "index.html").read_text(encoding="utf-8")
    key = app.index("tray.dropKey(e.key)")
    assert app.index('if ($("#edit-form")) return;') < key < app.index('/^[1-6]$/.test(e.key)')
    assert "drop: (ids, t) => inline.dropOn(ids, t)" in app
    assert "A.editor.stageMany(" not in TRAY_JS.read_text(encoding="utf-8")
