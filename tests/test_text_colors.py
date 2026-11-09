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
  note: E.describeItems({{ set: {{ colors: [{{ field: "note", color: "#000000" }}] }}, unset: [] }}, {{}}).map(x => x.psd),
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
      part: M.baseInk(c, "ability", "Gain 3 HP.", "Gain 2 HP.") }};
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
    if out["mock"] is not None:                         # an edit cuts into "2": "HP" keeps its colour, "3" none
        assert out["mock"] == {"drawn": [["Burning", "#ff7a1a"], ["HP", "#00ff00"]], "part": [{"start": 6, "end": 9, "color": "#00ff00"}]}


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


@pytest.mark.skipif(not pe.text_engine(), reason="no psd-tools text engine (the fork)")
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


@pytest.mark.skipif(not pe.text_engine(), reason="no psd-tools text engine (the fork)")
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
@pytest.mark.skipif(not pe.text_engine(), reason="no psd-tools text engine (the fork)")
def test_mockup_refuses_as_the_bake(clone, tmp_path):
    """mockup.js paintedOver, on the cards' details records, names what plan_colors refuses as painted over, in its words:
    a phrase under the title's or Ability's Gradient; not a whole field's, a phrase after one, or one under a lone Color Overlay."""
    if "export function paintedOver" not in (ASSETS / "mockup.js").read_text(encoding="utf-8"):
        pytest.skip("mockup.js has no paintedOver yet")
    from commands import psd_details as pd
    from commands.psd_edit import offline_parser
    vocab = json.loads((Path(__file__).resolve().parent / "goldens" / "_vocab.json").read_text(encoding="utf-8"))
    parser = offline_parser(vocab["all_types"], vocab["stars"], vocab["manifest"])
    name, ability = {"field": "name", "color": "#ff2200"}, {"field": "ability", "color": "#ff2200"}
    cases = {"Creatures/Kirby/4 Stars/Wham_Bam_Bone.psd": [[{**name, "phrase": "Bam"}], [name], [name, {**name, "phrase": "Bam"}]],
             "Auxiliary/Minions/Mini_Doomer.psd": [[{**ability, "phrase": "stats"}], [ability, {**ability, "phrase": "stats"}]],
             "Auxiliary/Minions/Cat.psd": [[{**ability, "phrase": "attack"}]]}
    jobs, want = [], []
    for rel, sets in cases.items():
        psd, card = PSDImage.open(_card(clone, rel)[0]), Path(rel).stem.replace("_", " ")
        record = pd.gather(psd, rel, card, None, vocab["all_types"], parser)
        jobs += [{"rec": record, "colors": colors} for colors in sets]
        want += [[[one, how] for one, how in pe.plan_colors(psd, colors, card) if "paints over" in how] for colors in sets]
    (tmp_path / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
    (tmp_path / "over.mjs").write_text(f"""const M = await import({json.dumps((ASSETS / 'mockup.js').as_uri())});
const jobs = JSON.parse((await import("fs")).readFileSync({json.dumps(str(tmp_path / 'jobs.json'))}, "utf8"));
process.stdout.write(JSON.stringify(jobs.map(j => M.paintedOver(j.rec, j.colors).map(n => [n.one, n.why]))));""", encoding="utf-8")
    got = json.loads(run_process(["node", str(tmp_path / "over.mjs")], capture_output=True, text=True, check=True).stdout)
    assert got == want and [len(w) for w in want] == [1, 0, 0, 1, 0, 0], (got, want)
