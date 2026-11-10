"""
The website's card image previews in edit mode (TTSCardMaker site/assets/preview.js and mockup.js, in node; the build's
data/fonts.json from site/build/build_site.py): the rendered preview's request, store, polling states, staleness and
messages against card-preview.yml's contract (commands/card_preview.py's result.json), and the quick mock-up's layout,
erase, wrapping and stat cells on drawn pixel arrays; the build reads font names as font_check does (fontTools) and maps
the PSD fonts to MDW/Fonts files, with WOFF2 copies (web_woff2). Loads the site from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without
it, node, or a site before preview.js.
"""
import importlib.util
import json
import shutil
from os import environ
from os.path import expanduser
from pathlib import Path
from subprocess import run as run_process

import pytest

from conftest import CLONE

SITE = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE)))
ASSETS = SITE / "site" / "assets"
PREVIEW_JS, MOCKUP_JS = ASSETS / "preview.js", ASSETS / "mockup.js"
BUILD_SITE = SITE / "site" / "build" / "build_site.py"
FONTS = SITE / "MDW" / "Fonts"
needs_js = pytest.mark.skipif(not (PREVIEW_JS.is_file() and shutil.which("node")), reason=f"no preview.js at {ASSETS}, or no node")


def has(path: Path, word: str) -> bool:
    return path.is_file() and word in path.read_text(encoding="utf-8")


viewer = pytest.mark.skipif(not has(PREVIEW_JS, "tabsHtml"), reason="no image viewer in preview.js")


def node(tmp_path, body: str):
    """Run body with preview.js as p and mockup.js as m; its stdout as JSON."""
    script = tmp_path / "probe.mjs"
    script.write_text(f"const p = await import({json.dumps(PREVIEW_JS.as_uri())});\nconst m = await import({json.dumps(MOCKUP_JS.as_uri())});\n"
                      f"const out = x => process.stdout.write(JSON.stringify(x));\n{body}\n", encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@needs_js
def test_the_panel_for_each_state(tmp_path):
    """Baked: current and preview side by side with Diff and Full size, each field's outcome, notes escaped; refused: the
    reason (with the bake's render when it made one); an input error is the website's bug; a transient one says try
    again; running: how long; no edit: the button disabled. Baked with no image file: said so, not an error; an error
    for an edit since changed: out of date, Preview again instead of Try again."""
    out = node(tmp_path, """
const v = { card: { name: "Mini <Doomer>", ts: 1, img: "ab" }, n: 1, hash: "h", now: 3 * 60000, scopes: "repo", images: ["https://r/p.webp", "https://r/p.png"], current: "thumbs/ab-720.avif", diff: false };
const e = state => ({ request: "req-00001", at: 0, hash: "h", base: { ts: 1, img: "ab" }, edit: { hp: 8 }, state });
const res = (state, extra = {}) => ({ ...e(state), result: { fields: { hp: state }, notes: ["<script>x</script>"], base: {}, files: {}, ...extra } });
const html = (x, w) => { const r = p.renderedParts(x, w); return r.stage + r.caption; };
out([html(res("baked"), v), html(res("refused", { reason: "stat>10: hp 25", image: "refused" }), v),
  html(res("refused", { reason: "overflow" }), { ...v, images: [] }), html(res("error", { reason: "input: bad edit" }), v),
  html(res("error", { reason: "error: timeout", transient: true }), v), html(e("running"), v), html(null, { ...v, n: 0 }),
  html(res("baked"), { ...v, hash: "other" }), html(res("baked"), { ...v, images: [] }),
  html(res("error", { reason: "error: timeout", transient: true }), { ...v, hash: "other" })]);""")
    baked, refused_img, refused, bug, transient, running, none, stale, no_file, err_stale = out
    assert 'data-pv-img="current"' in baked and 'src="https://r/p.webp"' in baked and 'href="https://r/p.png"' in baked and 'data-pv="diff"' in baked
    assert "HP baked" in baked and "&lt;script&gt;" in baked and "<script>" not in baked and "Mini &lt;Doomer&gt;" in baked and "3 min ago" in baked
    assert "stat&gt;10: hp 25" in refused_img and "before refusing" in refused_img and "data-pv-img" not in refused
    assert "website bug" in bug and "try again" in transient.lower()
    assert "pv-spin" in running and "about a minute" in running and "16 min" in running and "disabled" in none
    assert "Out of date: the edit changed since" in stale
    assert "no image came with the result" in no_file and 'class="err"' not in no_file and "HP baked" in no_file   # not an error
    assert "Out of date" in err_stale and "Try again" not in err_stale and "Preview again" in err_stale   # Try again would send the newer edit


discards = pytest.mark.skipif(not has(PREVIEW_JS, "export function discard"), reason="no Discard in preview.js")


@needs_js
@discards
def test_a_preview_can_be_discarded(tmp_path):
    """Discard (owner 2026-10-30, a trash icon): the preview shown goes from this browser's list, a card with none left goes
    from the store; the trash shows on a finished, expired or stale preview, never while one is drawn."""
    out = node(tmp_path, """
const e = (request, state, at) => ({ request, at, hash: "h", card: "c1", edit: { hp: 8 }, state, result: { fields: {}, base: {}, files: {} } });
const store = { c1: [e("req-00002", "baked", 2), e("req-00001", "refused", 1)], c2: [{ ...e("req-00003", "baked", 3), card: "c2" }] };
const v = { card: { name: "X", ts: 1, img: "ab" }, n: 1, hash: "h", now: 5, scopes: "repo", images: [], current: null, diff: false };
const html = (x, w) => { const r = p.renderedParts(x, w); return r.stage + r.caption; };
out([p.discard(store, "req-00002"), p.discard(store, "req-00003"), html(e("req-1", "baked", 2), v), html(e("req-1", "running", 2), v),
  html(e("req-1", "expired", 2), v), html(e("req-1", "baked", 2), { ...v, hash: "other" })]);""")
    first, last, baked, running, expired, stale = out
    assert [x["request"] for x in first["c1"]] == ["req-00001"] and "c2" in first
    assert "c2" not in last and len(last["c1"]) == 2
    assert 'data-pv="discard"' in baked and 'data-pv="discard"' in expired and 'data-pv="discard"' in stale
    assert 'data-pv="discard"' not in running and "Discard this preview" in baked


brisk = pytest.mark.skipif(not has(PREVIEW_JS, "POLL_BRISK"), reason="no 1 s checks in preview.js")


@needs_js
@brisk
def test_dispatch_first_and_brisk_checks(tmp_path):
    """Start sends the dispatch at once; only its 404 asks for the workflow, to word it. Checks: none before 17 s, every
    second until a minute while at most 3 wait, else every 3 s, then 10 s past 2 minutes."""
    out = node(tmp_path, """
const mem = new Map();
Object.assign(globalThis, { localStorage: { getItem: k => mem.get(k) ?? null, setItem: (k, v) => mem.set(k, v), removeItem: k => mem.delete(k) },
  window: { addEventListener() {} }, setTimeout: () => 0 });
const run = async replies => {
  const calls = [];
  const github = async (path, o = {}) => { calls.push(`${o.method || "GET"} ${path.split("/").pop()}`); const s = replies.shift(); if (s) throw Object.assign(new Error(`GitHub ${s}: Not Found`), { status: s }); return null; };
  mem.clear();
  const pv = p.createPreviews({ S: { data: { repo: "o/r" }, byId: new Map() }, editor: { me: { login: "ed" }, github, scopes: "repo", psdOf: () => ({}) }, toast() {}, badges() {} });
  await pv.start({ id: "c", key: "c", name: "C" }, { hp: 8 });
  const e = pv.entries("c")[0];
  return [calls, e.state, e.error || ""];
};
const at = ages => p.pollDelay({ c: ages.map((a, i) => ({ request: `r${i}`, at: 1e6 - a, state: "running" })) }, 1e6);
out([await run([0]), await run([404, 404]), await run([404, 0]), await run([401]),
  [at([5000]), at([17000]), at([59000]), at([60000]), at([1000, 2000, 30000]), at([1000, 2000, 3000, 30000]), at([130000])]]);""")
    ok, missing, scope, expired, delays = out
    assert ok == [["POST dispatches"], "running", ""]
    assert missing[0] == ["POST dispatches", "GET card-preview.yml"] and "isn't set up" in missing[2]
    assert scope[0] == ["POST dispatches", "GET card-preview.yml"] and "can't start Actions" in scope[2]
    assert expired[0] == ["POST dispatches"] and "sign in again" in expired[2]
    assert delays == [12000, 1000, 1000, 3000, 1000, 3000, 10000]


@needs_js
def test_tokens_wrap_and_fit(tmp_path):
    """[fire] is an icon (":no" crossed out), an unknown bracket stays text, no break inside "[ice].", a blank line is an
    empty line; a long text shrinks until it fits, a name keeps its line count."""
    out = node(tmp_path, """
const icons = { fire: { file: "icons/fire.webp" }, ice: { file: "icons/ice.webp" } };
const paras = m.tokenize("Inflict [fire] and [nope].\\n\\nTurns into [ice]. [fire:no]x", icons);
const measure = (s, z) => s.length * z * 0.5;
const lines = m.layout(paras, 60, 10, measure).map(l => l.pieces.map(q => q.icon ? `<${q.name}${q.no ? ":no" : ""}>` : q.text).join(""));
const fitted = m.fit(m.tokenize("a ".repeat(20), icons), 100, 40, 20, measure), name = m.fit(m.tokenize("A very long card name", icons), 100, 100, 20, measure, { lines: 1 });
out([lines, fitted.lines.length * fitted.size * m.LINE <= 40.5, fitted.size < 20, name.lines.length]);""")
    assert out[0] == ["Inflict <fire>", "and [nope].", "", "Turns into", "<ice>. <fire:no>x"]   # 5 px a character, icons 11
    assert out[1] is True and out[2] is True and out[3] == 1


@needs_js
@pytest.mark.skipif(not has(MOCKUP_JS, "pointLayout"), reason="no pointLayout in mockup.js")
def test_point_text_keeps_its_lines(tmp_path):
    """Point text (the status effects' Ability: px, no frame) never wraps: a paragraph a line at its size; smaller by
    0.94 only while a line runs off the card's sides, down to 40%."""
    out = node(tmp_path, """
const measure = (s, z) => s.length * z * 0.5, paras = m.tokenize("-2 ATK\\nCan't affect anyone. Lasts 2 turns.");
const r = reach => { const x = m.pointLayout(paras, 20, reach, measure); return [+x.size.toFixed(2), x.lines.map(l => Math.round(l.w))]; };
out([r(1000), r(300), r(10)]);""")
    assert out == [[20, [60, 350]], [16.61, [50, 291]], [7.91, [24, 138]]]


@needs_js
@pytest.mark.skipif(not has(MOCKUP_JS, "steadyPitch"), reason="no steadyPitch in mockup.js")
def test_old_line_pitch(tmp_path):
    """The old lines' pitch: the mean of their gaps near the median, not the least (a capital moves a line's middle); a
    paragraph's gap left out; one line: none."""
    out = node(tmp_path, """
const L = mids => mids.map(c => [0, c - 8, 100, c + 8]);
out([m.steadyPitch(L([10, 41, 71, 105])), m.steadyPitch(L([10, 40, 70, 130, 160])), m.steadyPitch(L([10]))]);""")
    assert out[0] == pytest.approx(95 / 3) and out[1:] == [30, 0]


@needs_js
@pytest.mark.skipif(not has(MOCKUP_JS, "mainInk"), reason="no mainInk in mockup.js")
def test_the_records_fill(tmp_path):
    """The record's main ink fills when the text shows it (white letters with a black outline on parchment stay white;
    one just above a mid background takes the light outline; orange below it keeps its black one); else the pixels'
    guess. Main ink: the most used colour, however few, none under an overlay."""
    out = node(tmp_path, """
const px = (c, n) => Array(n).fill([(c[0] * 77 + c[1] * 150 + c[2] * 29) >> 8, ...c]), bg = [230, 210, 170], g = [100, 100, 100];
const text = (fill, n, b = bg, o = [0, 0, 0]) => [...px(fill, n), ...px(o, 30), ...px(b, 20)], tan = [190, 160, 121], or = [211, 124, 1];
const t = (colors, runs) => ({ layer: "Text/Ability", text: "Deal 3 Burn.", colors, runs });
const two = t(["#ffffff", "#ff0000"], [[0, 7, 0], [7, 12, 1]]), W = "#ffffff", R = "#ff0000", B = "#0000ff";
out([m.textColours(text([255, 255, 255], 50), bg), m.textColours(text([255, 255, 255], 50), bg, [255, 255, 255]),
  m.textColours(text([255, 255, 255], 50), bg, [0, 0, 255]), m.textColours(text([120, 120, 120], 50, g, [250, 250, 250]), g, [120, 120, 120]),
  m.textColours([...text(or, 50, tan), ...px([240, 230, 210], 15)], tan, or),
  m.mainInk({}, two), m.mainInk({ effects: { Text: ["ColorOverlay"] } }, two), m.mainInk({}, { layer: "Name", text: "X", color: "#102030" }),
  m.mainInk({}, t([R, W, B], [[0, 3, 0], [3, 8, 1], [8, 12, 2]])), m.mainInk({}, t([W, "#f0f0f0", R], [[0, 4, 0], [4, 8, 1], [8, 12, 2]]))]);""")
    guess, white, unseen, light, orange, ink, over, one, split, near = out
    assert guess == {"fill": [0, 0, 0], "stroke": [255, 255, 255]} == unseen   # today's guess: black letters
    assert white == {"fill": [255, 255, 255], "stroke": [0, 0, 0]} and light == {"fill": [120, 120, 120], "stroke": [250, 250, 250]}
    assert orange == {"fill": [211, 124, 1], "stroke": [0, 0, 0]}
    assert ink == near == [255, 255, 255] and over is None and one == [16, 32, 48] and split == [0, 0, 255]


@needs_js
@pytest.mark.skipif(not has(MOCKUP_JS, "gradientOn"), reason="no gradient overlay shades in mockup.js")
def test_gradient_overlay_shades(tmp_path):
    """Under an enabled Gradient Overlay (the layer's or a group's) erase reads the old letters' colours top to bottom,
    past the background and the black outline; without one: no shades, the same colours."""
    out = node(tmp_path, """
const W = 200, H = 100, data = new Uint8ClampedArray(W * H * 4);
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
  const t = (y - 30) / 40, k = x % 12, ink = x >= 40 && x < 160 && y >= 30 && y < 70 && k < 5;
  data.set([...(!ink ? [128, 128, 128] : k === 0 || k === 4 ? [0, 0, 0] : [255 * (1 - t), 0, 255 * t]), 255], (y * W + x) * 4);
}
const run = profile => m.erase(Uint8ClampedArray.from(data), W, H, [20, 20, 180, 80], { lines: [[40, 30, 160, 70]], reach: 2, cell: 8, edge: 25, profile });
const fx = { effects: { Text: ["GradientOverlay"], Off: ["GradientOverlay (off)"], Name: ["ColorOverlay"] } };
out([run(true), run(false), ["Text/Ability", "Off/Ability", "Name", ""].map(p => m.gradientOn(fx, p)), m.oneColour(fx, "Name")]);""")
    shaded, flat, on, one = out
    s = shaded.pop("shades")
    assert s["top"] == 30 and s["bottom"] == 70 and len(s["stops"]) == 8
    (r0, _, b0), (r1, _, b1) = s["stops"][0][1], s["stops"][-1][1]
    assert r0 > 200 > 50 > b0 and b1 > 200 > 50 > r1   # red at the top, blue at the bottom: not the grey, not the outline
    assert flat.pop("shades") is None and flat == shaded and shaded["stroke"] == [0, 0, 0]
    assert on == [True, False, False, False] and one is True


kept = pytest.mark.skipif(not has(MOCKUP_JS, "statSite"), reason="no statSite in mockup.js")


@needs_js
@kept
def test_what_the_bake_wont_print(tmp_path):
    """The mock-up leaves what the bake won't print and says so: stars the bake keeps (no Stars group: the card only moves
    folder; an older record's "folder sets them" is still redrawn), a title in the picture, a stat past an old bar's 10
    (a layer-comp bar bakes it)."""
    out = node(tmp_path, f"""
const P = await import({json.dumps((ASSETS / "parts.js").as_uri())});
const none = {{ units: [], at: {{}}, refused: {{ "*": 'no group named "Stars" draws them' }} }};
const rec = (title, so, stats) => ({{ canvas: {{ h: 1000 }}, parts: {{ title, ability: [{{ id: 1, text: "x" }}] }}, smart_objects: so, stats,
  text: [{{ layer: "Text/Ability", text: "x", visible: true, box: [0, 600, 10, 700] }}] }});
const digits = rec([], [], {{ hp: "template" }}), why = m.statSite(digits, "hp", 12);
out([P.starsKept(none, 3), P.starsKept({{ units: [{{}}], refused: {{ "*": "the '2 Stars' folder sets them" }} }}, 3),
  P.titleInPicture(rec([], [])), P.titleInPicture(rec([{{ id: 2, text: "Name" }}], [])),
  why, m.statSite(digits, "hp", 9), m.statSite(rec([], ["Stats Bars/Darks/Health"], {{ hp: "template" }}), "hp", 15),
  m.statSite(rec([], [], {{ hp: "two-row" }}), "hp", 15),
  p.mockNote({{ starsKept: {{ why: none.refused["*"], to: "Creatures/Mario/4 Stars/" }}, site: [["name", "title"], ["HP 12", why]] }})]);""")
    stars, folder, pictured, titled, old, low, comp, rows, note = out
    assert stars == 'no group named "Stars" draws them' and folder is None
    assert pictured is True and titled is False
    assert old == "the card's old-style bars print up to 10" and low is comp is rows is None   # several rows: can't tell
    assert "The printed stars stay as they are" in note and "only moves to the 4 Stars folder" in note
    assert "website only until it's drawn in Photoshop" in note and "HP 12 shows on the website only: the card's old-style bars" in note


@needs_js
@pytest.mark.skipif(not has(MOCKUP_JS, "paintComp"), reason="no paintComp in mockup.js")
def test_comp_bar_past_10(tmp_path):
    """A layer-comp bar's 4 -> 13 as its comps draw it: the top half all lit, the bottom half lit to 3, a soft line in the
    borders' colour between the halves lit in both; 10 -> 20 lit everywhere with the line across."""
    out = node(tmp_path, """
const W = 120, H = 20, row = { box: [10, 2, 110, 18], cells: 10 };
const bar = from => { const d = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const c = Math.floor((x - 10) / 10), edge = x % 10 === 0 || x % 10 === 9 || y === 2 || y === 17;
    const v = x < 10 || x >= 110 || y < 2 || y >= 18 ? [0, 0, 255] : edge ? [0, 0, 0] : c < from ? [200, 200, 0] : [40, 40, 0];
    d.set([...v, 255], (y * W + x) * 4); }
  return d; };
const at = (d, y) => Array.from({ length: 10 }, (_, c) => d[(y * W + 15 + 10 * c) * 4]);
const a = bar(4), b = bar(10); m.paintComp(a, W, H, row, 4, 13); m.paintComp(b, W, H, row, 10, 20);
out([at(a, 5), at(a, 14), at(a, 10), at(b, 14), at(b, 10)]);""")
    top, bottom, line, full, across = out
    assert top == [200] * 10 and bottom == [200] * 3 + [40] * 7
    assert line == [60] * 3 + [40] * 7 and full == [200] * 10 and across == [60] * 10


@needs_js
@pytest.mark.skipif(not has(MOCKUP_JS, "paintLit"), reason="no paintLit in mockup.js")
def test_stat_from_0_lit_from_other_rows(tmp_path):
    """DEF 0 -> 5 between HP 1 and ATK 4: each cell copied from the nearest row lit there (HP's rounded end into cell 0),
    else HP's rounded end's flat right half; borders and the cells past 5 untouched."""
    out = node(tmp_path, """
const W = 120, H = 60, rows = [0, 1, 2].map(i => ({ box: [10, 2 + 20 * i, 110, 18 + 20 * i], cells: 10 })), d = new Uint8ClampedArray(W * H * 4);
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const r = Math.floor(y / 20), c = Math.floor((x - 10) / 10);
  const edge = x < 10 || x >= 110 || y % 20 < 3 || y % 20 >= 17 || x % 10 === 0 || x % 10 === 9;
  const v = edge ? [0, 0, 0] : r === 0 && c === 0 ? [x % 10 < 5 ? 150 : 200, 0, 0] : r === 2 && c < 4 ? [0, 60, 0] : [40, 40, 40];   // HP's rounded end, ATK lit to 4
  d.set([...v, 255], (y * W + x) * 4); }
m.paintLit(d, W, H, rows[1], [{ row: rows[0], v: 1 }, { row: rows[2], v: 4 }], 5);
const at = x => [d[(30 * W + x) * 4], d[(30 * W + x) * 4 + 1]];
out([Array.from({ length: 10 }, (_, c) => [...at(12 + 10 * c), at(17 + 10 * c)[0]]), at(20)]);""")
    cells, border = out
    assert cells == [[150, 0, 200]] + [[0, 60, 0]] * 3 + [[200, 0, 200]] + [[40, 40, 40]] * 5 and border == [0, 0]


@needs_js
@pytest.mark.skipif(not has(MOCKUP_JS, "wide: true"), reason="no wide-font titles in mockup.js")
def test_wide_font_title(tmp_path):
    """A one-line title 2.3 times as wide as its height says (a wide font) is found on the second pass, not boxed; at 3.3
    it stays boxed."""
    out = node(tmp_path, """
const W = 400, H = 60, paras = m.tokenize("Rapier"), measure = (s, z) => s.length * z * 0.5, area = { box: [0, 0, W, H], seed: null };
const title = ink => { const L = new Uint8Array(W * H);
  for (let y = 20; y < 40; y++) for (let x = 100; x < 100 + ink; x++) L[y * W + x] = x % 4 < 2 ? 255 : 0;   // letters 20 px high
  const hit = m.bestBlock(L, W, H, area, paras, measure, { name: true }), b = m.textBlocks(L, W, H, area.box, { join: 1.2, span: 2 })[0];
  return [m.calibrate(b, paras, measure, { exact: true }).ok, hit && +hit.cal.k.toFixed(2), hit && hit.b.box]; };
out([title(146), title(208)]);""")
    (first, k, box), (_, wider, _) = out
    assert first is False and 2.2 < k < 2.4 and box == [99, 19, 246, 41]
    assert wider is None


@needs_js
@pytest.mark.skipif(not has(MOCKUP_JS, "joins = ["), reason="no title set-size pass in mockup.js")
def test_title_at_set_size(tmp_path):
    """Letters a quarter em high and a wide word space: by its ink one word fits; at the record's px (40), words joined
    up to 4 line heights, the whole title; not joined, or a px it does not fill (20): none."""
    out = node(tmp_path, """
const W = 400, H = 60, paras = m.tokenize("Gold Clot"), measure = (s, z) => s.length * z * 0.5, area = { box: [0, 0, W, H], seed: null };
const L = new Uint8Array(W * H);
for (let y = 20; y < 30; y++) for (const [a, b] of [[100, 175], [205, 280]]) for (let x = a; x < b; x++) L[y * W + x] = x % 4 < 2 ? 255 : 0;
const hit = o => { const h = m.bestBlock(L, W, H, area, paras, measure, { name: true, ...o }); return h && [h.b.box, Math.round(h.cal.size)]; };
out([hit({}), hit({ wide: true, px: 40, joins: [1.2, 4] }), hit({ wide: true, px: 40 }), hit({ wide: true, px: 20, joins: [1.2, 4] })]);""")
    assert out == [[[99, 19, 174, 31], 17], [[99, 19, 278, 31], 40], None, None]


@needs_js
@pytest.mark.skipif(not has(MOCKUP_JS, "px = 0, wrap = 0"), reason="no set-size pass in mockup.js")
def test_tight_leading_at_set_size(tmp_path):
    """Three lines set tighter than their size says: not found by their pitch; at the record's px (20) and frame width
    (100) found at that size; a frame where they wrap to two lines, or a px they do not fill, stays boxed."""
    out = node(tmp_path, """
const W = 400, H = 120, paras = m.tokenize("abcd efgh ijkl mnop qrst uvwx"), measure = (s, z) => s.length * z * 0.5, area = { box: [0, 0, W, H], seed: null };
const L = new Uint8Array(W * H);
for (let i = 0; i < 3; i++) for (let y = 40 + 16 * i; y < 47 + 16 * i; y++) for (let x = 155; x < 245; x++) L[y * W + x] = x % 4 < 2 ? 255 : 0;
const hit = o => { const h = m.bestBlock(L, W, H, area, paras, measure, o); return h && [h.cal.size, +h.cal.k.toFixed(2), h.cal.nPred]; };
out([hit({}), hit({ px: 20, wrap: 100 }), hit({ px: 20, wrap: 150 }), hit({ px: 30, wrap: 100 })]);""")
    assert out == [None, [20, 0.93, 3], None, None]


@needs_js
@pytest.mark.skipif(not has(MOCKUP_JS, "titleLayout"), reason="no titleLayout in mockup.js")
def test_title_keeps_its_size(tmp_path):
    """A longer title stays at its size (psd_edit never shrinks one): point text on its lines, a box's wrapped at its
    width and told past its line count; one off the card's side shifts back on (back_on), not one wider than the card."""
    out = node(tmp_path, """
const measure = (s, z) => s.length * z * 0.5, at = (x0, x1, n = 1) => ({ lines: Array.from({ length: n }, (_, i) => [x0, 20 + 50 * i, x1, 60 + 50 * i]), box: [x0, 20, x1, 10 + 50 * n] });
const t = (text, n0, found) => { const r = m.titleLayout(m.tokenize(text), n0, found, 200, 20, measure, 1000); return [r.size, r.lines.length, r.over, Math.round(r.dx)]; };
const long = "A much longer card name";
out([t(long, 1, at(400, 600)), t(long, 1, at(420, 580, 2)), t(long + " that runs on and on and on", 1, at(420, 580, 2)),
  t("Royal Eradicator", 2, at(420, 580, 2)), t("Royal\\nEradicator II", 1, at(400, 600)), t("A long name", 1, at(940, 990)), t("x".repeat(120), 1, at(400, 600))]);""")
    assert out == [[20, 1, False, 0], [20, 2, False, 0], [20, 3, True, 0], [20, 1, False, 0], [20, 2, False, 0], [20, 1, False, -20], [20, 1, False, 0]]


@needs_js
@pytest.mark.skipif(not has(ASSETS / "edit.js", "capsError"), reason="no stat caps in edit.js")
def test_stat_caps_refused_as_typed(tmp_path):
    """A card's own display (owner 2026-11-08): a value past it is told as typed (edit.js capsError, cards.json statCaps),
    one as published passes, the template's bars have none; the bake's refusal reads plainly in the preview."""
    out = node(tmp_path, f"""
const E = await import({json.dumps((ASSETS / "edit.js").as_uri())});
const virus = {{ name: "Computer Virus", hp: 2, spd: 5, creatures: [{{ hp: 2, spd: 5 }}, {{ hp: 3, spd: 5 }}, {{ hp: 3, spd: 5 }}],
  statCaps: {{ hp: [5, 5, 5], spd: [5, 5, 5] }} }}, letion = {{ name: "Monkey D Letion", hp: 8, statCaps: {{ hp: [10] }} }};
out([E.capsError(virus, virus, "spd", 6), E.capsError(virus, virus, "spd", 5), E.capsError(virus, virus, "creatures", [{{ hp: 2 }}, {{ hp: 6 }}, {{ hp: 3 }}]),
  E.capsError(letion, letion, "hp", 11), E.capsError({{ name: "Sonic", hp: 7 }}, {{ hp: 7 }}, "hp", 15), E.statCap(virus, "hp", 4),
  p.plainReason("unsupported-card: creatures: def 5: creature 2's DEF row has room for 4 pictures")]);""")
    assert out == ["Computer Virus's SPD row (creature 1) holds 5 pictures: 6 won't fit", "",
                   "Computer Virus's HP row (creature 2) holds 5 pictures: 6 won't fit",
                   "Monkey D Letion's stat bars show up to 10: HP 11 won't fit", "", 10,
                   "The bake would refuse DEF 5: creature 2's DEF row has room for 4 pictures."]


@needs_js
@pytest.mark.skipif(not has(MOCKUP_JS, "first = null"), reason="no template type spot in mockup.js")
def test_first_type_at_the_template(tmp_path):
    """A card with no type icon takes one new type at Creature_Template's spot, as plan_types does; two at once, a
    token row or no spot: boxed."""
    out = node(tmp_path, """
const spot = m.TEMPLATES.tall.first.map((v, i) => Math.round(v * (i % 2 ? 2400 : 1200)));
out([spot, m.typePlan(null, [], ["fire"], spot), m.typePlan(null, [], ["fire", "water"], spot).boxed,
  m.typePlan({ tokens: true, refused: "tokens" }, [], ["fire"], spot).boxed, m.typePlan(null, [], ["fire"]).boxed]);""")
    spot, one, *boxed = out
    assert spot == [62, 27, 175, 183]   # template_icon on main's Creature_Template (11-08)
    assert one == {"erase": [], "draw": [{"name": "fire", "core": spot}], "refused": None, "boxed": False}
    assert boxed == [True, True, True]


@needs_js
@pytest.mark.skipif(not has(MOCKUP_JS, "fitLook"), reason="no icon looks in mockup.js")
def test_type_icons_in_the_cards_colours(tmp_path):
    """A shown type row recoloured by a Gradient Map or Curves (the look the bake copies onto a new icon) gives the map
    that reads it; stock colours (or near them), the icon's own art or too few pixels: none (the stock icon is drawn)."""
    out = node(tmp_path, """
const W = 24, icon = f => { const d = new Uint8ClampedArray(W * W * 4); for (let i = 0; i < W * W; i++) { const x = i % W, y = (i / W) | 0; d.set([...f(x, y), x < 3 || y < 3 ? 0 : 255], i * 4); } return d; };
const st = icon((x, y) => [x * 11, y * 11, (x + y) * 5]), row = [st, icon((x, y) => [255 - y * 9, x * 10, 128])];   // two icons
let s = 7; const art = () => [0, 0, 0].map(() => (s = (s * 16807) % 2147483647) % 256);
const as = (d, fn) => { const th = d.slice(); for (let i = 0; i < d.length; i += 4) th.set(d[i + 3] ? fn(d[i], d[i + 1], d[i + 2]) : art(), i); return th; };   // card art around the icon
const gmap = (r, g, b) => { const t = (0.299 * r + 0.587 * g + 0.114 * b) / 255; return [255 * t, 30 + 200 * t, 20]; };
const curves = (r, g, b) => [255 - r, g / 2, b + 80];
const fit = fn => m.fitLook(row.map(d => [d, as(d, fn)]));
const err = (look, fn) => { let e = 0, n = 0; for (let i = 0; i < st.length; i += 4) if (st[i + 3]) { const a = m.lookMap(look, st[i], st[i + 1], st[i + 2]), b = fn(st[i], st[i + 1], st[i + 2]).map(v => Math.min(255, v)); e += a.reduce((t, v, k) => t + Math.abs(v - b[k]), 0) / 3; n++; } return e / n; };
const g = fit(gmap), c = fit(curves);
out([Object.keys(g), err(g, gmap), Object.keys(c), err(c, curves), fit((r, g, b) => [r, g, b]), fit((r, g, b) => [r + 20, g + 20, b + 20]), fit(art), m.fitLook([[st.slice(3 * W * 4, 5 * W * 4), as(st.slice(3 * W * 4, 5 * W * 4), gmap)]])]);   // 42 opaque pixels""")
    luma, luma_err, chan, chan_err, *none = out
    assert luma == ["luma"] and luma_err < 3 and chan == ["chan"] and chan_err < 3
    assert none == [None, None, None, None]


@needs_js
@pytest.mark.skipif(not has(MOCKUP_JS, "labelRow"), reason="no labelRow in mockup.js")
def test_stat_row_at_its_label(tmp_path):
    """A template row with no bar moves to its stat's visible label (x span and height kept) when the card's bar is
    there; a hidden label, none, or cells of another size (a bar a quarter cell aside too): no row."""
    out = node(tmp_path, """
const W = 600, H = 1200, row = m.barRows("tall", W, H, { hp: 3 }, null)[0], [x0, , x1] = row.box;
const rec = text => ({ canvas: { w: 1200, h: 2400 }, text }), label = { text: "Health ", visible: true, box: [50, 1660, 320, 1730] };
const bar = cw => { const L = new Uint8Array(W * H);
  for (let y = 822; y < 870; y++) for (let x = 0; x < W; x++) L[y * W + x] = x < x0 || x >= x0 + 10 * cw ? 255 : (x - x0) % cw < 2 ? 20 : 200;
  return L; }, L = bar((x1 - x0) / 10);   // 10 cells at y 822-870
const moved = m.labelRow(L, W, H, rec([{ text: "Speed", box: [50, 300, 300, 360] }, { ...label, text: "Health", visible: false }, label]), row);
const small = bar((x1 - x0) / 12.5);   // cells 0.8 as wide: a bar to isBar there, and a quarter cell aside
out([row.box, moved && moved.box, m.isBar(L, W, H, row.box), m.isBar(small, W, H, moved.box), m.labelRow(small, W, H, rec([label]), row),
  m.labelRow(L, W, H, rec([{ ...label, visible: false }]), row), m.labelRow(L, W, H, { text: [] }, row)]);""")
    tpl, moved, at_tpl, small_bar, *none = out
    assert moved == [tpl[0], 826.5, tpl[2], 868.5] and tpl[3] - tpl[1] == 42 and at_tpl is False and small_bar is True
    assert none == [None, None, None]


@needs_js
@pytest.mark.skipif(not has(MOCKUP_JS, "layerBlock"), reason="no layerBlock in mockup.js")
def test_layer_block_takes_its_own_lines(tmp_path):
    """One layer's Ability: a bigger header and a short last line in its own box join the found block (erased, the new
    text from the header down); the same header above the box stays."""
    out = node(tmp_path, """
const W = 400, H = 200, L = new Uint8Array(W * H), lines = [[100, 80, 300, 94], [100, 100, 300, 114], [100, 120, 300, 134]];
const ink = (x0, y0, x1, y1) => { for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) L[y * W + x] = x % 4 < 2 ? 255 : 0; };
[[100, 52, 220, 70], [100, 140, 160, 154], ...lines].forEach(l => ink(...l));
const b = { lines, box: [100, 80, 300, 134], align: "left", pitch: 20 };
const r = own => { const { found, strays } = m.layerBlock(L, W, H, b, own, 14); return [strays.length, found.box, found.lines.length, found === b]; };
out([r([90, 50, 310, 160]), r([90, 76, 310, 160]), r([90, 76, 310, 136])]);""")
    both, below, none = out
    assert both == [2, [92, 51, 300, 155], 5, False] and below[0] == 1 and below[1][1] == 80
    assert none == [0, [100, 80, 300, 134], 3, True]


@needs_js
@pytest.mark.skipif(not has(MOCKUP_JS, "recordBlock"), reason="no recordBlock in mockup.js")
def test_record_block_when_not_found(tmp_path):
    """Text not found: the record's box split into one band per old line at its size (wrapped at the frame, blank
    lines not counted); left when the box sits at the frame's left, else centred."""
    out = node(tmp_path, """
const measure = (s, z) => s.length * z * 0.5, own = [100, 200, 300, 260];
const r = (t, wrap, frame) => { const f = m.recordBlock(own, m.tokenize(t), measure, 20, wrap, frame); return [f.lines.map(l => [l[1], l[3]]), f.align, f.size]; };
out([r("abcd efgh ijkl", 0, null), r("abcd efgh ijkl", 100, [95, 0, 400, 0]), r("abcd\\n\\nefgh", 100, [0, 0, 400, 0])]);""")
    assert out == [[[[200, 260]], "center", 20], [[[200, 230], [230, 260]], "left", 20], [[[200, 230], [230, 260]], "center", 20]]


# ---- the build's web fonts (build_site.py)
@pytest.fixture(scope="module")
def bs():
    if not BUILD_SITE.is_file():
        pytest.skip(f"no card repo build_site at {BUILD_SITE}")
    spec = importlib.util.spec_from_file_location("site_build_site_previews", BUILD_SITE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def test_woff2_copies(bs, tmp_path, monkeypatch):
    """web_woff2: one copy per used font, named by blob and encoder, cached across builds (unused ones pruned), none for the
    icon font or a font that fails; a real encode keeps every glyph (with brotli); mockup.js loads it, raw as the fallback."""
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.ttLib import TTFont
    pen = TTGlyphPen(None)
    pen.moveTo((0, 0)); pen.lineTo((0, 500)); pen.lineTo((400, 0)); pen.closePath()
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder([".notdef", "A"]); fb.setupCharacterMap({65: "A"})
    fb.setupGlyf({".notdef": TTGlyphPen(None).glyph(), "A": pen.glyph()}); fb.setupHorizontalMetrics({".notdef": (500, 0), "A": (450, 0)})
    fb.setupHorizontalHeader(ascent=800, descent=-200); fb.setupNameTable({"familyName": "Tiny", "styleName": "Regular"})
    fb.setupOS2(); fb.setupPost()
    (tmp_path / "MDW" / "Fonts").mkdir(parents=True)
    fb.save(str(tmp_path / "MDW" / "Fonts" / "Tiny.ttf"))
    for name in ("Bad.ttf", "stoogedings.ttf"):
        (tmp_path / "MDW" / "Fonts" / name).write_bytes(b"x")
    tree = {f"MDW/Fonts/{n}": bs.blob_sha((tmp_path / "MDW" / "Fonts" / n).read_bytes()) for n in ("Tiny.ttf", "Bad.ttf", "stoogedings.ttf")}
    for k, v in {"REPO": tmp_path, "CACHE": tmp_path / "cache", "DIST": tmp_path / "dist"}.items():
        monkeypatch.setattr(bs, k, str(v))
    fonts = lambda: {"Tiny-Regular": {"file": "MDW/Fonts/Tiny.ttf", "bytes": 1}, "Bad": {"file": "MDW/Fonts/Bad.ttf", "bytes": 1},
                     "stoogedings-Regular": {"file": "MDW/Fonts/stoogedings.ttf", "bytes": 1}}
    real = importlib.util.find_spec("brotli") is not None
    if not real:                                   # no brotli here: the build's logic with a stand-in encoder
        monkeypatch.setattr(bs, "woff2_tag", lambda: "abc123")
        monkeypatch.setattr(bs, "woff2_bytes", lambda data: b"wOF2" + data if data != b"x" else 1 / 0)
    (tmp_path / "cache" / "woff2").mkdir(parents=True)
    (tmp_path / "cache" / "woff2" / "old.woff2").write_bytes(b"")
    first = fonts()
    assert bs.web_woff2(first, tree) == 1 and bs.web_woff2(fonts(), tree) == 0      # the second build reuses the copy
    name = f"{tree['MDW/Fonts/Tiny.ttf'][:16]}-{bs.woff2_tag()}.woff2"
    assert first["Tiny-Regular"]["woff2"] == f"fonts/{name}" and "woff2" not in first["Bad"] and "woff2" not in first["stoogedings-Regular"]
    assert sorted(p.name for p in (tmp_path / "cache" / "woff2").iterdir()) == [name] and (tmp_path / "dist" / "fonts" / name).is_file()
    if real:
        back = TTFont(str(tmp_path / "dist" / "fonts" / name))
        assert back.flavor == "woff2" and back.getGlyphOrder() == [".notdef", "A"] and back.getBestCmap() == {65: "A"}
        assert back["glyf"]["A"].coordinates == TTFont(str(tmp_path / "MDW" / "Fonts" / "Tiny.ttf"))["glyf"]["A"].coordinates
    if shutil.which("node") and has(MOCKUP_JS, "fontSrc"):
        out = node(tmp_path, 'out([m.fontSrc({file: "MDW/Fonts/A B.ttf", woff2: "fonts/x.woff2"}, "https://raw/", "https://site/c/"), m.fontSrc({file: "MDW/Fonts/A.ttf"}, "https://raw/", "https://site/")]);')
        assert out == ['url("https://site/c/fonts/x.woff2") format("woff2"), url("https://raw/MDW/Fonts/A%20B.ttf")', 'url("https://raw/MDW/Fonts/A.ttf")']
