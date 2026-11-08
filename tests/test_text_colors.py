"""
Text colours (Card Edits.json `colors`, owner 2026-11-07): the field's rules and file handling, the website's menu logic
(TTSCardMaker site/assets/textcolor.js, from $AC_SITE_SRC or the clone), and a bake on a real card copy: the runs and the
stored raster recoloured in place, a phrase not there refused alone, the note's never baked.
"""
import importlib.util
import json
import shutil
from os import environ
from os.path import expanduser
from pathlib import Path
from subprocess import run as run_process

import numpy as np
import pytest
from psd_tools import PSDImage

import commands.psd_edit as pe
from commands.card_bake import psd_fields, rewrite_entry
from commands.card_edits import ORDER, CardEdits, color_text, dump_edits
from conftest import CLONE
from helpers import _card

SITE = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE))) / "site"
ASSETS = SITE / "assets"
site = pytest.mark.skipif(not ((ASSETS / "textcolor.js").is_file() and shutil.which("node")), reason=f"no textcolor.js in {ASSETS}, or no node")
FREEZER = "Field/1 Stars/The_Freezer.psd"
BURNING = {"field": "ability", "phrase": "Burning", "color": "#ff7a1a"}


def test_field_rules_and_file():
    """Valid lists only; the note's colours are never baked; a bake takes out what it printed and keeps the rest."""
    good = [BURNING, {"field": "name", "color": "#ffffff"}, {"field": "note", "phrase": "ice#2", "color": "#00bfff"}]
    assert CardEdits.valid("colors", good) and ORDER.index("colors") == ORDER.index("parts") + 1
    for bad in ([], [{**BURNING, "color": "#FF7A1A"}], [{**BURNING, "phrase": " Burning"}], [BURNING, dict(BURNING)],
                [{"field": "series", "color": "#ffffff"}], [{**BURNING, "x": 1}]):
        assert not CardEdits.valid("colors", bad), bad
    edits = CardEdits.parse(json.dumps({"cards": {"A": {"colors": good, "_site": ["colors"]}, "B": {"colors": "red"}}}))
    assert edits.cards["a"]["colors"] == good and edits.site("a") == ["colors"] and "colors" not in edits.cards["b"]
    assert psd_fields({"colors": good}) == {"colors": good[:2]} and psd_fields({"colors": good[2:]}) == {}
    assert psd_fields({"colors": good}, ["colors"]) == {}
    entry = rewrite_entry({"colors": good, "hp": 3, "_by": "web"}, ["hp"], colors_done=good[:2])
    assert entry == {"colors": good[2:], "_by": "web"} and rewrite_entry({"colors": good[:1]}, colors_done=good[:1]) is None
    assert color_text(BURNING) == "Ability: 'Burning' coloured #ff7a1a" and color_text(good[1]) == "Name coloured #ffffff"
    spec = importlib.util.spec_from_file_location("site_build_colors", SITE / "build" / "build_site.py")
    if spec and Path(spec.origin).is_file() and "def valid_colors" in Path(spec.origin).read_text(encoding="utf-8"):   # TC landed
        bs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bs)
        assert all(bs.valid_colors(v) == CardEdits.valid("colors", v) for v in (good, [], [dict(BURNING), dict(BURNING)]))
        assert "colors" in bs.PSD_FIELDS and "colors" in bs.SITE_FIELDS
        aco = CLONE / bs.SWATCHES[0]
        if aco.is_file():
            names = dict(bs.parse_aco(aco.read_bytes()))
            assert names.get("HP") == "#00ff00" and len(names) > 50


@site
def test_menu_logic(tmp_path):
    """The phrase a selection names (whole words, "#n" when it repeats), the list a pick leaves, the runs drawn (over the
    card's own colours), the plain words, and the file edit.js writes."""
    script = tmp_path / "probe.mjs"
    script.write_text(f"""const T = await import({json.dumps((ASSETS / 'textcolor.js').as_uri())});
const E = await import({json.dumps((ASSETS / 'edit.js').as_uri())});
const t = "Inflict Burning. Burning foes take +1 DEF.";
const at = w => [t.indexOf(w), t.indexOf(w) + w.length];
const second = t.indexOf("Burning", 10);
const list = T.withColor(null, "ability", "Burning#2", "#ff7a1a", t);
const out = {{
  phrases: [T.phraseAt(t, ...at("urni")), T.phraseAt(t, second, second + 7), T.phraseAt(t, ...at("+1 DEF")), T.phraseAt(t, 3, 3), T.phraseAt(t, 0, t.length), T.phraseAt(t, ...at(". "))],
  list, ranges: T.colorRanges(t, [...list, {{ field: "ability", color: "#ffffff" }}], "ability"),
  cleared: T.withColor([...list, {{ field: "ability", phrase: "foes", color: "#000000" }}], "ability", "Burning#2", null, t),
  whole: T.withColor(list, "ability", undefined, null, t), valid: [T.validColors(list), T.validColors([{{ field: "x", color: "#000000" }}])],
  html: T.paintHtml("<b>Fire</b>", [{{ start: 3, end: 7, color: "#ff0000" }}, {{ start: 0, end: 2, color: "red;x" }}]),
  words: T.colorsChange([{{ field: "name", color: "#ffffff" }}], list),
  file: E.editsFile(JSON.stringify({{ cards: {{}} }}), [{{ key: "X", set: {{ note: "n", colors: list }}, unset: [] }}], "T", "web"),
  lines: E.describeItems({{ set: {{ colors: list }}, unset: [] }}, {{}}).map(x => [x.text, x.psd]),
  note: E.describeItems({{ set: {{ colors: [{{ field: "note", color: "#000000" }}] }}, unset: [] }}, {{}}).map(x => x.psd),
  status: T.statusSpans("Inflict 2T Burning. burning prose.", ["Burning"]),
  kept: T.colorRanges("Inflict 2T Burning. burning prose.", [{{ field: "ability", color: "#ffd21f" }}], "ability", ["Burning"]),
  ink: T.inkOf ? (() => {{                         // the card's own colours under the edits; an edit keeps the unchanged ends'
    const c = {{ name: "Ant", ability: "Inflict Burning now.", ink: {{ ability: [[8, 15, "#ff7a1a"], [16, 19, "#0060ff"]] }} }};
    T.anchorInks([c]); c.ability = "Inflict Burning right now.";
    const base = T.inkOf(c, "ability");
    return {{ base, name: T.inkOf(c, "name"), drawn: T.colorRanges(c.ability, [{{ field: "ability", phrase: "now", color: "#ffffff" }}], "ability", null, base),
      at: T.colorAt(null, "ability", c.ability, 8, 15, null, base), cut: T.inkOf(c, "ability", "Inflict Freezing right now."),
      low: T.paintHtml("ab", [{{ start: 0, end: 1, color: "#ffff00" }}, {{ start: 1, end: 2, color: "#202020" }}]) }};
  }})() : null,
}};
process.stdout.write(JSON.stringify(out));
""", encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout)
    t = "Inflict Burning. Burning foes take +1 DEF."
    assert [p and p.get("phrase") for p in out["phrases"]] == ["Burning#1", "Burning#2", "+1 DEF", None, None, None]
    assert out["phrases"][3] == {"start": 0, "end": len(t)} and out["phrases"][5] is None
    assert out["list"] == [{"field": "ability", "phrase": "Burning#2", "color": "#ff7a1a"}] and out["valid"] == [True, False]
    second = t.index("Burning", 10)
    assert {"start": second, "end": second + 7, "color": "#ff7a1a"} in out["ranges"]
    assert out["ranges"][0] == {"start": 0, "end": second - 1, "color": "#ffffff"}      # the field first, the phrase over it
    assert out["cleared"] == [{"field": "ability", "phrase": "foes", "color": "#000000"}] and out["whole"] is None
    assert out["html"] == '&lt;b&gt;<span class="tc" style="color:#ff0000">Fire</span>&lt;/b&gt;'
    assert out["words"] == ["Ability: 'Burning' (2nd) coloured #ff7a1a", "Name back to its own colour"]
    entry = json.loads(out["file"])["cards"]["X"]
    assert list(entry) == ["colors", "note", "_by", "_at"] and out["file"] == dump_edits(json.loads(out["file"]))
    assert out["lines"] == [["Ability: 'Burning' (2nd) coloured #ff7a1a", True]] and out["note"] == [False]
    assert out["status"] == [[11, 18]]                  # psd_edit.status_letters' (test_status_words_keep_the_chart)
    assert not any(r["start"] < 18 and 11 < r["end"] for r in out["kept"]) and any(r["start"] <= 20 < r["end"] for r in out["kept"])
    ink = out["ink"]
    if ink is None:                                     # before TC's card colours land
        return
    assert ink["cut"] == [{"start": 23, "end": 26, "color": "#0060ff"}]                 # no stray letters of an edited word
    assert ink["base"] == [{"start": 8, "end": 15, "color": "#ff7a1a"}, {"start": 22, "end": 25, "color": "#0060ff"}] and ink["name"] is None
    assert ink["drawn"] == [{"start": 8, "end": 15, "color": "#ff7a1a"}, {"start": 22, "end": 25, "color": "#ffffff"}] and ink["at"] == "#ff7a1a"
    assert ink["low"] == '<span class="tc lo-l" style="color:#ffff00">a</span><span class="tc lo-d" style="color:#202020">b</span>'


@pytest.mark.skipif(not pe.text_engine(), reason="no psd-tools text engine (the fork)")
def test_colour_bake_on_a_real_card(clone, tmp_path):
    """The Freezer: two words recoloured through every gate, in the runs and the raster, the letters where Photoshop drew
    them; nothing left to bake after; a phrase not on the card refused alone; the note's colour no job."""
    from commands.psd_edit import offline_parser
    vocab = json.loads((Path(__file__).resolve().parent / "goldens" / "_vocab.json").read_text(encoding="utf-8"))
    parser = offline_parser(vocab["all_types"], vocab["stars"], vocab["manifest"])
    psd_path, png = _card(clone, FREEZER)
    psd = PSDImage.open(psd_path)
    parsed = pe.parse_card(parser, psd_path, FREEZER, psd)
    edits = {"colors": [{"field": "ability", "phrase": "Turns into", "color": "#00bfff"}, BURNING,
                        {"field": "note", "color": "#000000"}]}
    got = pe.classify(parsed, edits, psd, parser)
    assert got.jobs == {"colors": edits["colors"][:1]} and [r.field for r in got.refusals] == ["colors"]
    assert "'Burning' is not in the ability" in got.refusals[0].why
    old = pe.ability_layer(psd)
    ink = (old.numpy("shape")[..., 0] > 0.5, tuple(old.bbox))
    baked = pe.apply_jobs(psd_path, png, FREEZER, got.jobs, parsed, parser, str(tmp_path), str(clone / pe.FONTS_DIR), psd=psd)
    assert [(name, fields) for _, name, fields in baked.records] == [("Ability", ["blocks", "channels"])]
    saved = PSDImage.open(baked.psd_path)
    layer = pe.ability_layer(saved)
    start = layer.text.index("Turns into")
    assert [pe._hex(f) for s, e, f in pe.run_fills(layer) if s <= start < e] == ["#00bfff"]
    assert tuple(layer.bbox) == ink[1] and np.array_equal(layer.numpy("shape")[..., 0] > 0.5, ink[0])   # not moved
    rgb = layer.numpy("color")[layer.numpy("shape")[..., 0] > 0.9]
    assert (np.abs(rgb - np.array([0, 191, 255]) / 255).max(axis=1) < 0.01).any()
    assert pe.stale_stories(saved) == [] and baked.reparse["card"] == parsed["card"]
    again = pe.classify(baked.reparse, {"colors": got.jobs["colors"]}, saved, parser)
    assert again.jobs == {} and again.redundant == ["colors"]


@pytest.mark.skipif(not pe.text_engine(), reason="no psd-tools text engine (the fork)")
def test_status_words_keep_the_chart(clone, tmp_path):
    """Capitalised chart status words keep the chart's colour (the guards' rule, owner 2026-11-07): a whole-field colour
    leaves them, a phrase of them alone is refused; prose ("burning") is no status word."""
    from commands.psd_edit import offline_parser
    chart = pe.status_fills(PSDImage.open(str(clone / pe.STATUS_CHART)))
    text = "Inflict 2T Burning. burning prose."
    assert pe.status_letters(text, chart) == set(range(11, 18))
    vocab = json.loads((Path(__file__).resolve().parent / "goldens" / "_vocab.json").read_text(encoding="utf-8"))
    parser = offline_parser(vocab["all_types"], vocab["stars"], vocab["manifest"])
    rel = "Auxiliary/Minions/Sigma.psd"
    psd_path, png = _card(clone, rel)
    psd = PSDImage.open(psd_path)
    parsed = pe.parse_card(parser, psd_path, rel, psd)
    edits = {"colors": [{"field": "ability", "color": "#ffd21f"}, {"field": "ability", "phrase": "Hardened", "color": "#ff0000"}]}
    got = pe.classify(parsed, edits, psd, parser, chart=chart)
    assert got.jobs == {"colors": edits["colors"][:1]} and "status word" in got.refusals[0].why
    before = {pe._hex(f) for s, e, f in pe.run_fills(pe.ability_layer(psd)) if "Hardened" in pe.ability_layer(psd).text[s:e]}
    baked = pe.apply_jobs(psd_path, png, rel, got.jobs, parsed, parser, str(tmp_path), str(clone / pe.FONTS_DIR), chart, psd=psd)
    layer = pe.ability_layer(PSDImage.open(baked.psd_path))
    fills = {layer.text[s:e].strip(): pe._hex(f) for s, e, f in pe.run_fills(layer) if layer.text[s:e].strip()}
    assert fills.get("Hardened") in before and "#ffd21f" in fills.values() and "status words" in " ".join(baked.fx_notes)
