"""
Text colours (Card Edits.json `colors`, owner 2026-11-07; the card image's alone since 2026-11-11): the field's rules and
file handling, the website's menu logic (TTSCardMaker site/assets/textcolor.js, from $AC_SITE_SRC or the clone), and a
bake on a real card copy: the runs and the stored raster recoloured in place, a phrase not there refused alone. The
site's own word colours (owner 2026-11-11): textcolor.js wordColors and build_site word_colors (TTS's) agree.
"""
import importlib.util
import json
import shutil
from os import environ
from os.path import expanduser
from pathlib import Path
from subprocess import run as run_process
from types import SimpleNamespace

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
    """Valid lists only, on the name and the ability (the note's option went, owner 2026-11-11); a bake takes out what
    it printed and keeps the rest."""
    good = [BURNING, {"field": "name", "color": "#ffffff"}, {"field": "ability", "phrase": "ice#2", "color": "#00bfff"}]
    note = [{"field": "note", "phrase": "ice#2", "color": "#00bfff"}]
    assert CardEdits.valid("colors", good) and ORDER.index("colors") == ORDER.index("parts") + 1
    for bad in ([], [{**BURNING, "color": "#FF7A1A"}], [{**BURNING, "phrase": " Burning"}], [BURNING, dict(BURNING)],
                [{"field": "series", "color": "#ffffff"}], [{**BURNING, "x": 1}], note):
        assert not CardEdits.valid("colors", bad), bad
    edits = CardEdits.parse(json.dumps({"cards": {"A": {"colors": good, "_site": ["colors"]}, "B": {"colors": "red"}}}))
    assert edits.cards["a"]["colors"] == good and edits.site("a") == ["colors"] and "colors" not in edits.cards["b"]
    assert psd_fields({"colors": good}) == {"colors": good} and psd_fields({"colors": good}, ["colors"]) == {}
    entry = rewrite_entry({"colors": good, "hp": 3, "_by": "web"}, ["hp"], colors_done=good[:2])
    assert entry == {"colors": good[2:], "_by": "web"} and rewrite_entry({"colors": good[:1]}, colors_done=good[:1]) is None
    assert color_text(BURNING) == "Ability: 'Burning' coloured #ff7a1a" and color_text(good[1]) == "Name coloured #ffffff"
    spec = importlib.util.spec_from_file_location("site_build_colors", SITE / "build" / "build_site.py")
    if spec and Path(spec.origin).is_file() and "def valid_colors" in Path(spec.origin).read_text(encoding="utf-8"):   # TC landed
        bs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bs)
        lists = (good, [], [dict(BURNING), dict(BURNING)], *([note] if "note" not in bs.COLOR_FIELDS else []))   # the note's: TC since 11-11
        assert all(bs.valid_colors(v) == CardEdits.valid("colors", v) for v in lists)
        assert "colors" in bs.PSD_FIELDS and "colors" in bs.SITE_FIELDS
        aco = CLONE / bs.SWATCHES[0]
        if aco.is_file():
            names = dict(bs.parse_aco(aco.read_bytes()))
            assert names.get("HP") == "#00ff00" and len(names) > 50


@site
def test_menu_logic(tmp_path):
    """The phrase a selection names (whole words, "#n" when it repeats), the list a pick leaves, the runs drawn (over the
    card's own colours, on the card page and the mock-up), the plain words, and the file edit.js writes."""
    script = tmp_path / "probe.mjs"
    script.write_text(f"""const T = await import({json.dumps((ASSETS / 'textcolor.js').as_uri())});
const E = await import({json.dumps((ASSETS / 'edit.js').as_uri())});
const M = await import({json.dumps((ASSETS / 'mockup.js').as_uri())});
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
  status: T.statusSpans("Inflict 2T Burning. burning prose.", ["Burning"]),
  kept: T.colorRanges("Inflict 2T Burning. burning prose.", [{{ field: "ability", color: "#ffd21f" }}], "ability", ["Burning"]),
  parts: T.layerColors ? (() => {{                 // a field over several layers: "word#n" counted across them
    const texts = ["New fire.", "Bonfire.", "Fire foes."], ones = [{{ field: "ability", phrase: "fire#2", color: "#ff0000" }}, {{ field: "ability", phrase: "fire", color: "#ff0000" }}];
    const rec = {{ parts: {{ title: [{{ id: 1, text: "Ant" }}], ability: [{{ id: 5, text: "fire", variants: [2] }}, {{ id: 2, text: "Old." }}, {{ id: 3, text: texts[1] }}, {{ id: 4, text: texts[2] }}] }} }};
    return {{ spans: ones.map(one => texts.map((t, i) => (T.layerColors(texts, i, [one], "ability") || []).flatMap(c => T.spansOf(t, c) || []))),
      jobs: M.partJobs(rec, {{ 2: {{ text: texts[0] }} }}, {{ colors: [ones[0], {{ field: "name", color: "#00ff00" }}] }}).map(j => [j.id, j.text === j.old, j.colors]) }};
  }})() : null,
  ink: T.inkOf ? (() => {{                         // the card's own colours under the edits; an edit keeps the unchanged ends'
    const c = {{ name: "Ant", ability: "Inflict Burning now.", ink: {{ ability: [[8, 15, "#ff7a1a"], [16, 19, "#0060ff"]] }} }};
    T.anchorInks([c]); c.ability = "Inflict Burning right now.";
    const base = T.inkOf(c, "ability");
    return {{ base, name: T.inkOf(c, "name"), drawn: T.colorRanges(c.ability, [{{ field: "ability", phrase: "now", color: "#ffffff" }}], "ability", null, base),
      at: T.colorAt(null, "ability", c.ability, 8, 15, null, base), cut: T.inkOf(c, "ability", "Inflict Freezing right now."),
      low: T.paintHtml("ab", [{{ start: 0, end: 1, color: "#ffff00" }}, {{ start: 1, end: 2, color: "#202020" }}]) }};
  }})() : null,
  mock: M.baseInk ? (() => {{                     // the mock-up keeps them as the card page does: the field's, a part's share
    const c = {{ ability: "Inflict Burning now.\\nGain 2 HP.", ink: {{ ability: [[8, 15, "#ff7a1a"], [26, 30, "#00ff00"]] }} }}, now = "Inflict Burning now.\\nGain 3 HP.";
    T.anchorInks([c]);
    return {{ drawn: M.colourize(M.tokenize(now, {{}}), now, null, "ability", null, M.baseInk(c, "ability", now)).flat(2).filter(p => p.color).map(p => [p.text, p.color]),
      part: M.baseInk(c, "ability", "Gain 3 HP.", "Gain 2 HP."), bake: !!M.bakeInk }};
  }})() : null,
  chart: M.colourize.toString().includes("fills") ? (() => {{   // capitalised status words in the chart's, over all
    const t = "Inflict Burning. burning foes.";
    return M.colourize(M.tokenize(t, {{}}), t, [{{ field: "ability", color: "#ffffff" }}], "ability", ["Burning"], [{{ start: 8, end: 15, color: "#123456" }}], null, {{ burning: "#ff7a1a" }})
      .flat(2).filter(p => p.color).map(p => [p.text, p.color]);
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
    assert out["lines"] == [["Ability: 'Burning' (2nd) coloured #ff7a1a", True]]
    assert out["status"] == [[11, 18]]                  # psd_edit.status_letters' (test_status_words_keep_the_chart)
    assert not any(r["start"] < 18 and 11 < r["end"] for r in out["kept"]) and any(r["start"] <= 20 < r["end"] for r in out["kept"])
    if out["parts"] is not None:                       # as color_spans counts; unchanged layers the colours reach redrawn
        layers = [SimpleNamespace(text=t) for t in ("New fire.", "Bonfire.", "Fire foes.")]
        assert out["parts"]["spans"] == [[[[a, b] for x, a, b in pe.color_spans(layers, {"field": "ability", "phrase": phrase})[0] if x is layer]
                                          for layer in layers] for phrase in ("fire#2", "fire")]
        assert out["parts"]["jobs"] == [["1", True, [{"field": "name", "color": "#00ff00"}]], ["2", False, None],
                                        ["4", True, [{"field": "ability", "phrase": "fire#1", "color": "#ff0000"}]]]
    ink = out["ink"]
    if ink is None:                                     # before TC's card colours land
        return
    assert ink["cut"] == [{"start": 23, "end": 26, "color": "#0060ff"}]                 # no stray letters of an edited word
    assert ink["base"] == [{"start": 8, "end": 15, "color": "#ff7a1a"}, {"start": 22, "end": 25, "color": "#0060ff"}] and ink["name"] is None
    assert ink["drawn"] == [{"start": 8, "end": 15, "color": "#ff7a1a"}, {"start": 22, "end": 25, "color": "#ffffff"}] and ink["at"] == "#ff7a1a"
    assert ink["low"] == '<span class="tc lo-l" style="color:#ffff00">a</span><span class="tc lo-d" style="color:#202020">b</span>'
    if out["mock"] is not None:                         # as the bake: "3" takes the "2"'s colour (bakeInk); before it none
        three = out["mock"].pop("bake")
        assert out["mock"] == {"drawn": [["Burning", "#ff7a1a"], *[[w, "#00ff00"] for w in (["3", "HP"] if three else ["HP"])]],
                               "part": [{"start": 5 if three else 6, "end": 9, "color": "#00ff00"}]}
    if out["chart"] is not None:                        # as psd_edit's status words; lowercase stays the field's
        assert out["chart"] == [["Inflict", "#ffffff"], ["Burning", "#ff7a1a"], [".", "#ffffff"], ["burning", "#ffffff"], ["foes.", "#ffffff"]]


PALETTE = {"stats": {"hp": "#2fbf3a", "def": "#3d63ff", "atk": "#ff3b3b", "spd": "#ffd21f"},
           "counters": {"points": "#00ffae", "bricks": "#a0522d", "sp": "#ffff00"},
           "statuses": {"Burning": "#db923d", "Frozen In Time": "#07cccd", "Hardened": "#8a8a8a"}}
WORDS = ["Inflict 2T Burning. burning foes take +1 DEF, -1 SPD, \u22121 HP and 0.5 ATK; x2 ATK, 50% SPD, max HP.",
         "Gain 2 points (2 points), 3 SP, 1 star point, 2 star points, BRICKS, 1 brick. +2 Attack; attack twice. At any point. 1 point.",
         "[fire] HP [Burning] (1CD) Frozen  In Time 3T2 Hardened. 3.5 HP, 1.5SPD, X HP, x HP, 2HP, ATK/DEF, +1/+1 ATK.",
         "Speed 3, 3 speed, 4 Health, 2 Defense, 1 Defence, (0.5 SP) (3 souls) Deal 2(x counters) DMG. HPs ATKS 3.HP a2 HP",
         "Negate 1 attack, 3 Attack, -1 attack, (2 x DEF + 4), (3 x Health), \U0001f550 3 SP"]
LINKED = "Deploy a Burning Ring. Inflict 2T Burning."   # a status word in another card's name stays plain
LINK_CARDS = [{"id": "burning", "name": "Burning", "type": "debuff"}, {"id": "burning-ring", "name": "Burning Ring", "type": "item"}]
REAL = json.loads((Path(__file__).resolve().parent / "fixtures" / "palette_texts.json").read_text(encoding="utf-8"))


@site
def test_site_words_agree_with_tts(tmp_path):
    """The site's word colours (owner 2026-11-11): textcolor.js wordColors (the website's card text) and build_site
    word_colors (tts.json's) paint the same letters in the same colours from one palette, on these and 50 real
    abilities: status words, stats with their numbers (Health & co. only after one, Attack after a signed one; a lone
    "x" a times sign), counters; icons and badges never; a word in another card's name plain on both (text.js render,
    build_site unlinked); build_site's palette is the one cards.json and tts.json carry."""
    spec = importlib.util.spec_from_file_location("site_build_words", SITE / "build" / "build_site.py")
    if not (spec and Path(spec.origin).is_file() and "def unlinked" in Path(spec.origin).read_text(encoding="utf-8")):
        pytest.skip("a site before its 11-11 palette rules")
    bs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bs)
    script = tmp_path / "words.mjs"
    cases = [[t, PALETTE] for t in WORDS] + [[t, REAL["palette"]] for t in REAL["texts"]]
    script.write_text(f"""const T = await import({json.dumps((ASSETS / 'textcolor.js').as_uri())});
const X = await import({json.dumps((ASSETS / 'text.js').as_uri())});
const cp = (t, i) => [...t.slice(0, i)].length;   // code points, as Python counts
const runs = {json.dumps(cases)}.map(([t, p]) => T.wordColors(t, p).map(r => [cp(t, r.start), cp(t, r.end), r.color]));
const linked = X.makeRenderer({{ cards: {json.dumps(LINK_CARDS)}, icons: {{}} }}).render({json.dumps(LINKED)}, {{ colors: T.wordColors({json.dumps(LINKED)}, {json.dumps(PALETTE)}) }});
process.stdout.write(JSON.stringify({{ runs, linked }}));
""", encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout)
    site = out["runs"]
    assert site == [[list(r) for r in bs.word_colors(t, p)] for t, p in cases] and len(site) == len(WORDS) + 50
    words = [[t[a:b] for a, b, _ in runs] for t, runs in zip(WORDS, site)]
    assert words[0] == ["Burning", "+1 DEF", "-1 SPD", "\u22121 HP", "0.5 ATK", "x2 ATK", "50% SPD", "HP"]
    assert words[1] == ["2 points", "3 SP", "1 star point", "2 star points", "BRICKS", "1 brick", "+2 Attack", "1 point"]
    assert words[2] == ["HP", "Frozen  In Time", "Hardened", "3.5 HP", "1.5SPD", "X HP", "HP", "2HP", "ATK", "DEF", "+1 ATK"]
    assert words[3] == ["3 speed", "4 Health", "2 Defense", "1 Defence", "HP"] and site[0][0][2] == "#db923d"
    assert words[4] == ["-1 attack", "DEF", "3 SP"]
    assert out["linked"] == ('Deploy a <a class="card-ref" href="c/burning-ring/" data-card="burning-ring">Burning Ring</a>. Inflict 2T '
                             '<a class="card-ref" href="c/burning/" data-card="burning"><span class="tc lo-l" style="color:#db923d">Burning</span></a>.')
    assert bs.tts_words(LINKED, PALETTE, {}, None, bs.name_links(LINK_CARDS, {}, {})) == [["Burning", "#db923d", 2]]
    cards = [{"name": "Burning", "type": "debuff"}, {"name": "Despair", "type": "debuff"}, {"name": "Hardened", "type": "buff"}, {"name": "Ant", "type": "item"}]
    palette = bs.text_palette(cards, {"burning": "#db923d", "Despair": "#000000", "Ant": "#123456"}, [{"name": "Hardened", "color": "#8a8a8a"}])
    assert palette["statuses"] == {"Burning": "#db923d", "Despair": "#000000", "Hardened": "#8a8a8a"}   # the chart's, else the swatch
    assert {k: palette[k] for k in ("stats", "counters")} == {k: PALETTE[k] for k in ("stats", "counters")}


def test_colour_bake_on_a_real_card(clone, tmp_path):
    """The Freezer: two words recoloured through every gate, in the runs and the raster, the letters where Photoshop drew
    them; nothing left to bake after; a phrase not on the card refused alone."""
    from commands.psd_edit import offline_parser
    vocab = json.loads((Path(__file__).resolve().parent / "goldens" / "_vocab.json").read_text(encoding="utf-8"))
    parser = offline_parser(vocab["all_types"], vocab["stars"], vocab["manifest"])
    psd_path, png = _card(clone, FREEZER)
    psd = PSDImage.open(psd_path)
    parsed = pe.parse_card(parser, psd_path, FREEZER, psd)
    edits = {"colors": [{"field": "ability", "phrase": "Turns into", "color": "#00bfff"}, BURNING]}
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


def test_colour_on_a_raster_off_the_card(clone, tmp_path):
    """PoD05 Lead Bloon's Ability raster starts left of the card: a word's colour bakes, the letters unmoved."""
    from commands.psd_edit import offline_parser
    from psd_tools.composite.text.verify import stored_raster
    vocab = json.loads((Path(__file__).resolve().parent / "goldens" / "_vocab.json").read_text(encoding="utf-8"))
    parser = offline_parser(vocab["all_types"], vocab["stars"], vocab["manifest"])
    rel = "Auxiliary/Minions/PoD05Lead_Bloon.psd"
    psd_path, png = _card(clone, rel)
    psd = PSDImage.open(psd_path)
    rgba, bbox = stored_raster(pe.ability_layer(psd))
    assert bbox[0] < 0
    parsed = pe.parse_card(parser, psd_path, rel, psd)
    edit = {"field": "ability", "phrase": "intercept", "color": "#ff2200"}
    got = pe.classify(parsed, {"colors": [edit]}, psd, parser)
    assert got.jobs == {"colors": [edit]} and not got.refusals
    baked = pe.apply_jobs(psd_path, png, rel, got.jobs, parsed, parser, str(tmp_path), str(clone / pe.FONTS_DIR), psd=psd)
    out, box = stored_raster(pe.ability_layer(PSDImage.open(baked.psd_path)))
    assert box == bbox and np.array_equal(out[..., 3], rgba[..., 3])               # not moved
    red = (out[..., 3] > 0.9) & (np.abs(out[..., :3] - np.array([1, 0x22 / 255, 0])).max(axis=2) < 0.01)
    assert red.sum() > 100


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


def test_status_colours_record(clone, tmp_path):
    """Card sync's status colours (db/status_colors.json, the mock-up's): each chart status as it prints, a gradient
    one in its mean colour, read once per chart blob; the site ships them (build_site.status_colors)."""
    from commands import card_data
    path = clone / pe.STATUS_CHART
    reads = []
    run_build = SimpleNamespace(shas={pe.STATUS_CHART: "c" * 40}, _file_bytes=lambda p: reads.append(p) or path.read_bytes())
    record = card_data.status_colors(run_build, {})
    chart, colors = pe.status_fills(PSDImage.open(str(path))), record["colors"]
    assert record["blob"] == "c" * 40 and {name.lower() for name in colors} == set(chart)
    assert pe._hex(chart["become back money"].fill) == "#ffffff" and colors["Become Back Money"] == "#d8982f"   # its gradient
    assert colors["Thickened"] == pe._hex(chart["thickened"].fill) and colors["Burning"] == pe._hex(chart["burning"].fill)
    assert card_data.status_colors(run_build, record) is record and len(reads) == 1
    spec = importlib.util.spec_from_file_location("site_build_status", SITE / "build" / "build_site.py")
    if spec and Path(spec.origin).is_file() and "def status_colors" in Path(spec.origin).read_text(encoding="utf-8"):   # TC landed
        bs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bs)
        (tmp_path / card_data.STATUS_COLORS).write_text(card_data.records_text({**record, "colors": {**colors, "X": "red"}}))
        assert bs.status_colors(str(tmp_path)) == colors and bs.status_colors(str(tmp_path / "none")) == {}


def test_title_overlay_goes_off(clone, tmp_path):
    """A whole title's colour turns the title's own Gradient Overlay off and bakes (white too, its fill's); its phrase
    after it bakes; a phrase alone stays refused."""
    from commands.psd_edit import offline_parser
    vocab = json.loads((Path(__file__).resolve().parent / "goldens" / "_vocab.json").read_text(encoding="utf-8"))
    parser = offline_parser(vocab["all_types"], vocab["stars"], vocab["manifest"])
    rel = "Creatures/Kirby/4 Stars/Wham_Bam_Bone.psd"
    psd_path, png = _card(clone, rel)
    psd = PSDImage.open(psd_path)
    parsed = pe.parse_card(parser, psd_path, rel, psd)
    bam = {"field": "name", "phrase": "Bam", "color": "#00bfff"}
    alone = pe.classify(parsed, {"colors": [bam]}, psd, parser)
    assert not alone.jobs and "Gradient Overlay paints over" in alone.refusals[0].why
    typed = {"field": "name", "phrase": "Boom", "color": "#00bfff"}     # a word the entry types: refused before the bake
    assert "Gradient Overlay paints over" in pe.plan_colors(psd, [typed], "Wham Bam Bone", {"name": "Wham Boom Bone"})[0][1]
    assert pe.plan_colors(psd, [{"field": "name", "color": "#ff2200"}, typed], "Wham Bam Bone",
                          {"name": "Wham Boom Bone"})[1][1] == "baked"
    assert pe.plan_colors(psd, [{"field": "name", "color": "#ffffff"}], "Wham Bam Bone")[0][1] == "baked"
    edits = {"colors": [{"field": "name", "color": "#ff2200"}, bam]}
    got = pe.classify(parsed, edits, psd, parser)
    assert got.jobs == edits and not got.refusals
    baked = pe.apply_jobs(psd_path, png, rel, got.jobs, parsed, parser, str(tmp_path), str(clone / pe.FONTS_DIR), psd=psd)
    assert "colors: title: its Gradient Overlay is off for the colour" in baked.fx_notes
    saved = PSDImage.open(baked.psd_path)
    layer = pe.color_layers(saved, "name", "Wham Bam Bone")[0]
    assert pe._painted_over(layer) is None and {pe._hex(f) for *_, f in pe.run_fills(layer)} >= {"#ff2200", "#00bfff"}


def test_split_title_part_overlay_goes_off(clone, tmp_path):
    """Crowned Magolor's title on two layers, each with its Gradient Overlay: a colour on all of one turns only that one's
    off and bakes; the other keeps its gradient; part of a word stays refused."""
    from commands.psd_edit import offline_parser
    vocab = json.loads((Path(__file__).resolve().parent / "goldens" / "_vocab.json").read_text(encoding="utf-8"))
    parser = offline_parser(vocab["all_types"], vocab["stars"], vocab["manifest"])
    rel = "Creatures/Kirby/5 Stars/Crowned_Magolor.psd"
    psd_path, png = _card(clone, rel)
    psd = PSDImage.open(psd_path)
    parsed = pe.parse_card(parser, psd_path, rel, psd)
    part = {"field": "name", "phrase": "Magolor", "color": "#ff2200"}
    assert "Gradient Overlay paints over" in pe.plan_colors(psd, [{**part, "phrase": "Mag"}], "Crowned Magolor")[0][1]
    got = pe.classify(parsed, {"colors": [part]}, psd, parser)
    assert got.jobs == {"colors": [part]} and not got.refusals
    baked = pe.apply_jobs(psd_path, png, rel, got.jobs, parsed, parser, str(tmp_path), str(clone / pe.FONTS_DIR), psd=psd)
    assert "colors: title: its Gradient Overlay is off for the colour" in baked.fx_notes
    crowned, magolor = pe.color_layers(PSDImage.open(baked.psd_path), "name", "Crowned Magolor")
    assert pe._painted_over(magolor) is None and "#ff2200" in {pe._hex(f) for *_, f in pe.run_fills(magolor)}
    assert pe._painted_over(crowned) == "its Gradient Overlay"


def test_clipped_title_layer_hidden(clone, tmp_path):
    """Cake Mix's cake clipped onto its title: a phrase alone refused with the reason; a whole title's colour hides the
    cake and bakes, its phrase after it too."""
    from commands.psd_edit import offline_parser
    vocab = json.loads((Path(__file__).resolve().parent / "goldens" / "_vocab.json").read_text(encoding="utf-8"))
    parser = offline_parser(vocab["all_types"], vocab["stars"], vocab["manifest"])
    rel = "Items/Heal/2 Stars/Cake_Mix.psd"
    psd_path, png = _card(clone, rel)
    psd = PSDImage.open(psd_path)
    parsed = pe.parse_card(parser, psd_path, rel, psd)
    mix = {"field": "name", "phrase": "Mix", "color": "#00bfff"}
    alone = pe.classify(parsed, {"colors": [mix]}, psd, parser)
    assert not alone.jobs and "its clipped layer 'HD-wallpaper-cake-colores-food-wall' paints over" in alone.refusals[0].why
    edits = {"colors": [{"field": "name", "color": "#ff2200"}, mix]}
    got = pe.classify(parsed, edits, psd, parser)
    assert got.jobs == edits and not got.refusals
    baked = pe.apply_jobs(psd_path, png, rel, got.jobs, parsed, parser, str(tmp_path), str(clone / pe.FONTS_DIR), psd=psd)
    assert any("clipped layer" in note for note in baked.fx_notes)
    layer = pe.color_layers(PSDImage.open(baked.psd_path), "name", "Cake Mix")[0]
    assert not any(one.visible for one in pe._clip_chain(layer)) and pe._painted_over(layer) is None
    assert {pe._hex(f) for *_, f in pe.run_fills(layer)} >= {"#ff2200", "#00bfff"}


def test_ability_overlay_goes_off(clone, tmp_path):
    """Cat's phrase under its lone Color Overlay: the overlay off, the rest in its colour; Mini Doomer's whole Ability
    turns its Gradient off, its one-colour badges coloured too; a phrase under that Gradient stays refused."""
    from commands.psd_edit import offline_parser
    vocab = json.loads((Path(__file__).resolve().parent / "goldens" / "_vocab.json").read_text(encoding="utf-8"))
    parser = offline_parser(vocab["all_types"], vocab["stars"], vocab["manifest"])
    for rel, edit, kind in (("Auxiliary/Minions/Cat.psd", {"field": "ability", "phrase": "attack", "color": "#ff2200"}, "Color"),
                            ("Auxiliary/Minions/Mini_Doomer.psd", {"field": "ability", "color": "#ff2200"}, "Gradient")):
        psd_path, png = _card(clone, rel)
        psd = PSDImage.open(psd_path)
        parsed = pe.parse_card(parser, psd_path, rel, psd)
        rest = pe._hex(pe._solid_overlay(pe.ability_layer(psd)) or (1, 0x22 / 255, 0))
        got = pe.classify(parsed, {"colors": [edit]}, psd, parser)
        assert got.jobs == {"colors": [edit]} and not got.refusals
        (tmp_path / kind).mkdir()
        baked = pe.apply_jobs(psd_path, png, rel, got.jobs, parsed, parser, str(tmp_path / kind), str(clone / pe.FONTS_DIR), psd=psd)
        assert f"colors: ability: its {kind} Overlay is off for the colour" in baked.fx_notes
        layer = pe.ability_layer(PSDImage.open(baked.psd_path))
        fills = {layer.text[s:e]: pe._hex(f) for s, e, f in pe.run_fills(layer)}
        assert pe._painted_over(layer) is None and set(fills.values()) == {"#ff2200", rest}
        assert all(fills[t] == "#ff2200" for t in fills if "(" in t), fills       # Mini Doomer's (BOT)
    word = {"field": "ability", "phrase": "stats", "color": "#00bfff"}
    assert "Gradient Overlay paints over" in pe.classify(parsed, {"colors": [word]}, psd, parser).refusals[0].why


@site
def test_mockup_refuses_as_the_bake(clone, tmp_path):
    """mockup.js paintedOver, on the cards' details records, names what plan_colors refuses as painted over, in its words:
    a phrase under the title's or Ability's Gradient; not a whole field's, a phrase after one, one under a lone Color
    Overlay, or one on all of a title part (and a phrase after it). A retyped title or Ability box: the same, in the new text."""
    if "export function paintedOver(rec, colors, typed = [], edit" not in (ASSETS / "mockup.js").read_text(encoding="utf-8"):
        pytest.skip("mockup.js has no retyped paintedOver yet")
    from commands import psd_details as pd
    from commands.psd_edit import offline_parser
    vocab = json.loads((Path(__file__).resolve().parent / "goldens" / "_vocab.json").read_text(encoding="utf-8"))
    parser = offline_parser(vocab["all_types"], vocab["stars"], vocab["manifest"])
    name, ability = {"field": "name", "color": "#ff2200"}, {"field": "ability", "color": "#ff2200"}
    bam, stats = {**name, "phrase": "Bam"}, {**ability, "phrase": "stats"}
    now = "Has stats equal to the base stats -2 of the creature that summoned it now.\n(BOT) Destroy this creature."
    cases = {"Creatures/Kirby/4 Stars/Wham_Bam_Bone.psd": [[bam], [name], [name, bam], ([bam], {"name": "Wham Bam Bone II"})],
             "Auxiliary/Minions/Mini_Doomer.psd": [[stats], [ability, stats], ([stats], {"ability": now, "parts": {"2068": {"text": now}}})],
             "Auxiliary/Minions/Cat.psd": [[{**ability, "phrase": "attack"}]],
             "Creatures/Kirby/5 Stars/Crowned_Magolor.psd": [[{**name, "phrase": "Magolor"}], [{**name, "phrase": "Crown"}],
                                                             [{**name, "phrase": "Magolor"}, {**name, "phrase": "Mag"}]]}
    jobs, want = [], []
    for rel, sets in cases.items():
        psd, card = PSDImage.open(_card(clone, rel)[0]), Path(rel).stem.replace("_", " ")
        record = pd.gather(psd, rel, card, None, vocab["all_types"], parser)
        for colors, edit in (s if isinstance(s, tuple) else (s, {}) for s in sets):
            typed, boxes = {f: edit[f] for f in ("name", "ability") if f in edit}, {k: v["text"] for k, v in edit.get("parts", {}).items()}
            jobs.append({"rec": record, "colors": colors, "typed": list(typed), "edit": edit})
            want.append([[one, how] for one, how in pe.plan_colors(psd, colors, card, typed, None, boxes) if "paints over" in how])
    (tmp_path / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
    (tmp_path / "over.mjs").write_text(f"""const M = await import({json.dumps((ASSETS / 'mockup.js').as_uri())});
const jobs = JSON.parse((await import("fs")).readFileSync({json.dumps(str(tmp_path / 'jobs.json'))}, "utf8"));
process.stdout.write(JSON.stringify(jobs.map(j => M.paintedOver(j.rec, j.colors, j.typed, j.edit).map(n => [n.one, n.why]))));""", encoding="utf-8")
    got = json.loads(run_process(["node", str(tmp_path / "over.mjs")], capture_output=True, text=True, check=True).stdout)
    assert got == want and [len(w) for w in want] == [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0], (got, want)
