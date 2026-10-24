"""
The website's card image previews in edit mode (TTSCardMaker site/assets/preview.js and mockup.js, in node; the build's
data/fonts.json from site/build/build_site.py): the rendered preview's request, store, polling states, staleness and
messages against card-preview.yml's contract (commands/card_preview.py's result.json), and the quick mock-up's layout,
erase, wrapping and stat cells on drawn pixel arrays; the build reads font names as font_check does (fontTools) and maps
the PSD fonts to MDW/Fonts files. Loads the site from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without
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

from commands.font_check import face_names
from conftest import CLONE

SITE = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE)))
ASSETS = SITE / "site" / "assets"
PREVIEW_JS, MOCKUP_JS = ASSETS / "preview.js", ASSETS / "mockup.js"
BUILD_SITE = SITE / "site" / "build" / "build_site.py"
FONTS = SITE / "MDW" / "Fonts"
needs_js = pytest.mark.skipif(not (PREVIEW_JS.is_file() and shutil.which("node")), reason=f"no preview.js at {ASSETS}, or no node")
HOUR = 3600_000


def node(tmp_path, body: str):
    """Run body with preview.js as p and mockup.js as m; its stdout as JSON."""
    script = tmp_path / "probe.mjs"
    script.write_text(f"const p = await import({json.dumps(PREVIEW_JS.as_uri())});\nconst m = await import({json.dumps(MOCKUP_JS.as_uri())});\n"
                      f"const out = x => process.stdout.write(JSON.stringify(x));\n{body}\n", encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


# ---- the rendered preview (preview.js)
@needs_js
def test_the_edit_is_the_entry_as_published(tmp_path):
    """The edit sent is every PSD-kind field of the card's Card Edits entry after the batch (equal to the PSD or not), in
    the edits file's shapes; database fields stay out; creatures only as a list of several. What the image changes by
    compares types in any order and text trimmed, and never counts a stat cleared to nothing."""
    out = node(tmp_path, """
const card = { name: " Mini Doomer EX ", hp: 8, def: 2, types: ["light", "wind"], ability: "Text\\r\\n", note: "x", series: "Kirby", creatures: null };
out([p.entryEdit(["name", "hp", "def", "types", "ability", "note", "series", "creatures"], f => card[f]),
  p.entryEdit(["creatures"], () => [{ hp: 1 }, { hp: 2 }]), p.entryEdit(["creatures"], () => [{ hp: 1 }]),
  p.changedOf({ name: "Mini Doomer", types: ["wind", "light"], hp: null, ability: " A \\n", stars: 2 }, { name: "Mini Doomer", types: ["light", "wind"], hp: 5, ability: "A", stars: 3 })]);""")
    assert out[0] == {"name": "Mini Doomer EX", "types": ["light", "wind"], "hp": 8, "def": 2, "ability": "Text"}
    assert out[1] == {"creatures": [{"hp": 1}, {"hp": 2}]} and out[2] == {}
    assert out[3] == {"stars": 2}


@needs_js
def test_the_same_edit_hashes_the_same(tmp_path):
    """Keys in any order and types in any order are the same edit (the contract compares sorted JSON); a value is not."""
    out = node(tmp_path, """
const a = { hp: 8, types: ["wind", "light"], name: "X" }, b = { name: "X", types: ["light", "wind"], hp: 8 };
out([p.canonical(a), p.hashOf(a), p.hashOf(b), p.hashOf({ ...a, hp: 9 })]);""")
    assert out[0] == '{"hp":8,"name":"X","types":["light","wind"]}'
    assert out[1] == out[2] != out[3] and len(out[1]) == 14


@needs_js
def test_request_ids_fit_the_action(tmp_path):
    """card-preview.yml takes 8-64 of A-Z a-z 0-9 -: any login makes one; the store keeps nothing else."""
    out = node(tmp_path, """
out([["MichaelJSr", "a_b.c", "", "x".repeat(80)].map(l => p.newRequest(l, 1792700000000, 0.5)),
  ["short", "a/b-cdefghij", "x".repeat(65), "ok-request-1"].map(p.validRequest)]);""")
    for r in out[0]:
        assert 8 <= len(r) <= 64 and all(ch.isalnum() or ch == "-" for ch in r)
    assert out[0][0].startswith("michaeljsr-") and out[0][1].startswith("a-b-c-") and out[0][2].startswith("editor-")
    assert out[1] == [False, False, False, True]


@needs_js
def test_result_json_is_read_defensively(tmp_path):
    """card_preview's result.json: states, fields, files and the run link kept; anything unreadable is a transient error;
    a file name with a path, a run link off GitHub and over-long notes are dropped or cut."""
    result = {"v": 1, "state": "baked", "transient": False, "reason": "hp 5 -> 8", "fields": {"hp": "baked", "stars": "refused", "evil": "x"},
              "changes": [{"field": "hp", "old": 5, "new": 8}], "notes": ["n"] * 30 + ["x" * 900], "base": {"psd_blob": "a" * 40, "png_blob": "b" * 40, "n": 3},
              "image": "baked", "files": {"webp": "preview.webp", "png": "../x.png"}, "run": "https://evil.example/run"}
    out = node(tmp_path, f"""out([p.readResult({json.dumps(json.dumps(result))}), p.readResult("not json"), p.readResult({{ state: "done" }}),
  p.readResult({{ state: "error", transient: true, reason: "error: timeout", run: "https://github.com/o/r/actions/runs/1" }})]);""")
    ok = out[0]
    assert ok["state"] == "baked" and ok["fields"] == {"hp": "baked", "stars": "refused"} and ok["changes"] == [{"field": "hp", "old": 5, "new": 8}]
    assert ok["files"] == {"webp": "preview.webp"} and ok["run"] == "" and ok["image"] == "baked"
    assert ok["base"] == {"psd_blob": "a" * 40, "png_blob": "b" * 40} and len(ok["notes"]) == 20
    assert out[1]["state"] == out[2]["state"] == "error" and out[1]["transient"] and "JSON" in out[1]["reason"]
    assert out[3]["transient"] and out[3]["run"] == "https://github.com/o/r/actions/runs/1"


@needs_js
def test_the_store_keeps_the_last_three_per_card(tmp_path):
    """Newest first, KEEP (3) per card, a request replaced in place; a malformed stored entry is dropped; a running one
    times out after MAX_WAIT_MS (20 minutes) or its own wait; one still starting a minute on (the page closed) failed."""
    out = node(tmp_path, """
let s = {};
for (let i = 0; i < 5; i++) s = p.remember(s, { card: "a", request: `req-0000${i}`, at: i, hash: "h", state: "running" });
s = p.remember(s, { card: "b", request: "req-b-0001", at: 1, hash: "h", state: "running", wait: 1000 });
s = p.remember(s, { card: "d", request: "req-d-0001", at: 1, hash: "h", state: "starting" });
s = p.remember(s, { card: "a", request: "req-00004", at: 4, hash: "h", state: "baked" });
const read = p.readStore(JSON.stringify({ ...s, c: [{ request: "bad/one", at: 1, hash: "h" }, { request: "req-c-0001", at: "x", hash: "h" }] }));
const t = p.expire(s, 20 * 60000 + 10);
out([s.a.map(e => e.request), s.a[0].state, Object.keys(read).sort(), p.waiting(t).map(e => e.request), t.b[0].state, p.patch(s, "req-00003", { seen: true }).a[1].seen, p.KEEP, p.MAX_WAIT_MS,
  [t.d[0].state, t.d[0].seen, p.expire(s, 30000).d[0].state]]);""")
    assert out[0] == ["req-00004", "req-00003", "req-00002"] and out[1] == "baked"
    assert out[2] == ["a", "b", "d"] and out[3] == [] and out[4] == "timeout" and out[5] is True
    assert out[6:8] == [3, 20 * 60000]
    assert out[8] == ["error", True, "starting"]       # a page closed mid-dispatch: not a spinner for ever, after a minute


@needs_js
def test_a_preview_goes_out_of_date(tmp_path):
    """Out of date when the fields the site would send changed, or the card's image moved: the result's PNG blob against
    cards.json img (its first 16 hex digits); before a result, the card's ts and img at the dispatch."""
    out = node(tmp_path, """
const card = { ts: 10, img: "abcdef0123456789" };
const e = { hash: "h1", base: { ts: 10, img: "abcdef0123456789" } };
out([p.staleWhy(e, card, "h1"), p.staleWhy(e, card, "h2"), p.staleWhy(e, { ...card, ts: 11 }, "h1"),
  p.staleWhy({ ...e, result: { base: { png_blob: "abcdef0123456789" + "f".repeat(24) } } }, { ...card, ts: 11 }, "h1"),
  p.staleWhy({ ...e, result: { base: { png_blob: "0".repeat(40) } } }, card, "h2")]);""")
    assert out == [[], ["edit"], ["psd"], [], ["edit", "psd"]]


@needs_js
def test_the_badge_is_for_an_unseen_result(tmp_path):
    """A tile is badged while its card's newest preview has a result nobody has seen; not while it runs, not once seen."""
    out = node(tmp_path, """
const s = { a: [{ state: "baked" }], b: [{ state: "running" }, { state: "baked" }], c: [{ state: "refused", seen: true }], d: [{ state: "error" }] };
out([["a", "b", "c", "d", "e"].map(id => p.hasNews(s, id)), p.badgeTip("baked"), p.badgeTip("error")]);""")
    assert out == [[True, False, False, True, False], "Its preview image is ready: open the card", "Its preview failed: open the card"]


@needs_js
def test_a_run_that_cant_start_says_why(tmp_path):
    """The Action missing on main; a public_repo token (repo scope needed, or Actions: Read and write); an expired token;
    GitHub's refusal; no network."""
    out = node(tmp_path, """
out([p.startProblem(404, "", { installed: false }), p.startProblem(403, "", { scopes: "public_repo, gist" }), p.startProblem(404, "", { scopes: "repo, workflow" }),
  p.startProblem(403, "", { scopes: null }), p.startProblem(401), p.startProblem(422, "Unexpected inputs provided"), p.startProblem(0, "Failed to fetch"),
  [p.scopeOk("public_repo"), p.scopeOk("repo"), p.scopeOk(null)]]);""")
    assert "isn't set up" in out[0] and "card-preview.yml" in out[0]
    assert "only the public_repo scope" in out[1] and "repo scope" in out[1] and "Actions: Read and write" in out[1]
    assert "can't start Actions" in out[2] and "can't start Actions" in out[3]
    assert "sign in again" in out[4] and "Unexpected inputs provided" in out[5] and "Couldn't reach GitHub" in out[6]
    assert out[7] == [False, True, None]


@needs_js
def test_the_panel_for_each_state(tmp_path):
    """Baked: current and preview side by side with Diff and Full size, each field's outcome, notes escaped; refused: the
    reason (with the bake's render when it made one); an input error is the website's bug; a transient one says try
    again; running: the check interval; no edit: the button disabled."""
    out = node(tmp_path, """
const v = { card: { name: "Mini <Doomer>", ts: 1, img: "ab" }, n: 1, hash: "h", now: 3 * 60000, scopes: "repo", images: ["https://r/p.webp", "https://r/p.png"], current: "thumbs/ab-720.webp", diff: false };
const e = state => ({ request: "req-00001", at: 0, hash: "h", base: { ts: 1, img: "ab" }, edit: { hp: 8 }, state });
const res = (state, extra = {}) => ({ ...e(state), result: { fields: { hp: state }, notes: ["<script>x</script>"], base: {}, files: {}, ...extra } });
out([p.renderedHtml(res("baked"), v), p.renderedHtml(res("refused", { reason: "stat>10: hp 25", image: "refused" }), v),
  p.renderedHtml(res("refused", { reason: "overflow" }), { ...v, images: [] }), p.renderedHtml(res("error", { reason: "input: bad edit" }), v),
  p.renderedHtml(res("error", { reason: "error: timeout", transient: true }), v), p.renderedHtml(e("running"), v), p.renderedHtml(null, { ...v, n: 0 }),
  p.renderedHtml(res("baked"), { ...v, hash: "other" })]);""")
    baked, refused_img, refused, bug, transient, running, none, stale = out
    assert 'data-pv-img="current"' in baked and 'src="https://r/p.webp"' in baked and 'href="https://r/p.png"' in baked and 'data-pv="diff"' in baked
    assert "HP baked" in baked and "&lt;script&gt;" in baked and "<script>" not in baked and "Mini &lt;Doomer&gt;" in baked and "3 min ago" in baked
    assert "stat&gt;10: hp 25" in refused_img and "before refusing" in refused_img and "data-pv-img" not in refused
    assert "website bug" in bug and "try again" in transient.lower()
    assert "pv-spin" in running and "every 15 s" in running and "disabled" in none
    assert "Out of date: the edit changed since" in stale


@needs_js
def test_the_diff_marks_changed_pixels(tmp_path):
    """Changed pixels red, the rest the current image dimmed; the share of changed pixels."""
    out = node(tmp_path, """
const a = new Uint8ClampedArray([100, 100, 100, 255, 10, 10, 10, 255]), b = new Uint8ClampedArray([100, 104, 100, 255, 200, 10, 10, 255]);
const d = p.diffPixels(a, b); out([[...d.data], d.changed]);""")
    assert out == [[30, 30, 30, 255, 255, 40, 40, 255], 0.5]


# ---- the quick mock-up (mockup.js)
PAINT = """
// a W x H grey image; text(lines): glyph-like stripes (light fill, dark outline) on rows [y0, y1) from x0 to x1
const W = 240, H = 240, img = new Uint8ClampedArray(W * H * 4);
const put = (x, y, v) => { const i = (y * W + x) * 4; img[i] = img[i + 1] = img[i + 2] = v; img[i + 3] = 255; };
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) put(x, y, 60 + Math.round(y / 8));
const text = lines => { for (const [x0, x1, y0, y1] of lines) for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { const k = (x - x0) % 6; put(x, y, k < 2 ? 240 : k < 3 ? 0 : 60 + Math.round(y / 8)); } };
"""


@needs_js
def test_the_text_on_the_image_is_found(tmp_path):
    """Three centred lines: one paragraph of three lines, its box, a size from their pitch, centred; left-aligned lines
    read as left; a short weak line between two strong ones stays in the paragraph; an empty region has none."""
    out = node(tmp_path, PAINT + """
const clean0 = new Uint8ClampedArray(img);
text([[40, 200, 150, 162], [52, 188, 170, 182], [46, 194, 190, 202]]);
const L = m.luma(img), a = m.findText(L, W, H, [0, 140, W, H]);
text([[10, 150, 20, 32], [10, 120, 40, 52], [10, 170, 60, 72]]);
const b = m.findText(m.luma(img), W, H, [0, 10, W, 90]);
const c = m.findText(L, W, H, [0, 100, W, 140]);
img.set(clean0);
text([[40, 200, 150, 162], [112, 124, 170, 182], [46, 194, 190, 202]]);       // "it." between two long lines
const d = m.findText(m.luma(img), W, H, [0, 140, W, H]);
out([a.lines.length, a.box, Math.round(a.size), a.align, b.align, c, d.box[1], d.box[3]]);""")
    assert out[0] == 3 and out[3] == "center" and out[4] == "left" and out[5] is None
    x0, y0, x1, y1 = out[1]
    assert 38 <= x0 <= 42 and 148 <= y0 <= 151 and 196 <= x1 <= 202 and 200 <= y1 <= 204
    assert 14 <= out[2] <= 18                      # pitch 20 px / 1.22
    assert 148 <= out[6] <= 151 and 200 <= out[7] <= 204    # the weak line kept the last one in the paragraph


@needs_js
def test_erase_puts_the_background_back(tmp_path):
    """Inside the box the text's pixels take the background around them (a gradient here), outside nothing changes; the
    colours read on the way: a light fill with a dark outline on a dark background."""
    out = node(tmp_path, PAINT + """
const clean = new Uint8ClampedArray(img);
text([[40, 200, 150, 162]]);
const before = new Uint8ClampedArray(img);
const col = m.erase(img, W, H, [30, 140, 210, 172], { reach: 2, cell: 10 });
let worst = 0, outside = 0;
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const i = (y * W + x) * 4, inBox = x >= 30 && x < 210 && y >= 140 && y < 172;
  if (inBox) worst = Math.max(worst, Math.abs(img[i] - clean[i])); else outside = Math.max(outside, Math.abs(img[i] - before[i])); }
out([worst, outside, col.fill, col.stroke]);""")
    worst, outside, fill, stroke = out
    assert worst <= 4 and outside == 0
    assert min(fill) >= 200 and max(stroke) <= 20


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
def test_what_the_mock_up_redraws(tmp_path):
    """Only fields unlike the PSD; stats above 10, stars and types are named, not drawn. A stat cell lights as the last lit
    one (the bar's rounded first cell: its colour) and darkens as the first dark one (none or the rounded last: a shade).
    Bars only where the template has them and the details call the stats "template"."""
    out = node(tmp_path, """
const psd = { name: "A", ability: "Text", hp: 5, def: 2, atk: 3, spd: 9, stars: 2, types: ["wind", "light"] };
const c = m.changes({ name: " A ", ability: "New", hp: 8, def: 12, atk: 3, spd: 10, stars: 3, types: ["light", "wind"] }, psd);
const card = { hp: 5, def: 2, atk: 3, spd: 9 };
out([c, m.cellPlan(5, 8), m.cellPlan(1, 3), m.cellPlan(10, 8), m.cellPlan(9, 8), m.cellPlan(5, 3),
  m.barRows("tall", 720, 1440, card, { stats: { hp: "template", def: "template", atk: "template", spd: "template" } }).map(r => [r.k, r.box.map(Math.round)]),
  m.barRows("tall", 720, 1440, card, { stats: { hp: "pixel" } }).length, m.barRows("square", 720, 720, card, null).length,
  [m.shapeOf(1200, 2400), m.shapeOf(1200, 1800), m.shapeOf(1200, 1200), m.shapeOf(1080, 1920)]]);""")
    assert out[0] == {"name": None, "ability": "New", "stats": {"hp": [5, 8], "spd": [9, 10]}, "skipped": ["DEF", "stars"]}
    assert out[1] == [{"cell": 5, "as": 4}, {"cell": 6, "as": 4}, {"cell": 7, "as": 4}]
    assert out[2] == [{"cell": 1, "as": "lit"}, {"cell": 2, "as": "lit"}]
    assert out[3] == [{"cell": 8, "as": "dark"}, {"cell": 9, "as": "dark"}] and out[4] == [{"cell": 8, "as": "dark"}]
    assert out[5] == [{"cell": 3, "as": 5}, {"cell": 4, "as": 5}]
    assert out[6] == [["hp", [216, 720, 704, 770]], ["def", [216, 796, 704, 846]], ["atk", [216, 871, 704, 922]], ["spd", [216, 947, 704, 997]]]
    assert out[7:] == [0, 0, ["tall", "item", "square", ""]]


@needs_js
def test_a_bar_is_told_from_art(tmp_path):
    """Ten cells with sharp borders read as a bar; a smooth gradient doesn't."""
    out = node(tmp_path, PAINT + """
for (let y = 100; y < 110; y++) for (let x = 20; x < 220; x++) put(x, y, (x - 20) % 20 < 2 ? 10 : 200);
const L = m.luma(img); out([m.isBar(L, W, H, [20, 100, 220, 110]), m.isBar(L, W, H, [20, 30, 220, 40])]);""")
    assert out == [True, False]


@needs_js
def test_layout_from_the_details_record(tmp_path):
    """The record's own text boxes (text[].box, canvas px) scaled to the image when the bot has them; the name and text
    layers from template.name_layer / ability_layers, else by name."""
    out = node(tmp_path, """
const rec = { canvas: { w: 1200, h: 2400 }, template: { name_layer: "Text/Title", ability_layers: ["Text/Ability"] },
  text: [{ layer: "Text/Title", font: "KirbyClassic", box: [240, 20, 960, 140] }, { layer: "Text/Ability", font: "Calibri" }, { layer: "Name", font: "X" }] };
out([m.recordBox(rec, "Text/Title", 720), m.recordBox(rec, "Text/Ability", 720), m.layerOf(rec, "name").font, m.layerOf(rec, "ability").font,
  m.layerOf({ text: [{ layer: "Name", font: "Y" }] }, "name").font, m.layerOf(null, "ability")]);""")
    assert out == [[144, 12, 576, 84], None, "KirbyClassic", "Calibri", "Y", None]


# ---- the build's web fonts (build_site.py)
@pytest.fixture(scope="module")
def bs():
    if not BUILD_SITE.is_file():
        pytest.skip(f"no card repo build_site at {BUILD_SITE}")
    spec = importlib.util.spec_from_file_location("site_build_site_previews", BUILD_SITE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "web_fonts"):
        pytest.skip(f"{BUILD_SITE} makes no data/fonts.json: land TTSCardMaker work/site2-previews")
    return module


def _fonts():
    files = sorted(p for p in FONTS.glob("*") if p.suffix.lower() in (".ttf", ".otf", ".ttc", ".otc")) if FONTS.is_dir() else []
    if not files:
        pytest.skip(f"no MDW/Fonts at {FONTS}")
    return files


def test_the_build_reads_font_names_as_font_check(bs):
    """Every font file in MDW/Fonts: the build's reader (no fontTools) gives font_check.face_names' PostScript and loose
    names, collections face by face."""
    differ = []
    for path in _fonts():
        data = path.read_bytes()
        try:
            want = face_names(data)
        except Exception:                          # noqa: BLE001 - fontTools can't read it: the build must refuse it too
            with pytest.raises(ValueError):
                bs.font_faces(data)
            continue
        if bs.font_faces(data) != want:
            differ.append(path.name)
    assert differ == []


def test_web_fonts_map_the_psd_fonts(bs, tmp_path, monkeypatch):
    """data/fonts.json: a PSD font name to its MDW/Fonts file as font_check resolves it (exact PostScript name, an alias,
    a loose match without -Regular); never a collection or a file over FONT_WEB_MAX; names read once per blob (cached)."""
    files = {p.name: p for p in _fonts()}
    pick = [n for n in ("consolab.ttf", "Arial.ttf", "BookAntiqua.ttc") if n in files]
    if len(pick) < 3:
        pytest.skip("MDW/Fonts lacks consolab.ttf, Arial.ttf or BookAntiqua.ttc")
    repo = tmp_path / "repo"
    (repo / "MDW" / "Fonts").mkdir(parents=True)
    tree = {}
    for i, name in enumerate(pick):
        shutil.copy(files[name], repo / "MDW" / "Fonts" / name)
        tree[f"MDW/Fonts/{name}"] = f"{i:040x}"
    monkeypatch.setattr(bs, "REPO", str(repo))
    monkeypatch.setattr(bs, "CACHE", str(tmp_path / "cache"))
    faces = {name: bs.font_faces((repo / "MDW" / "Fonts" / name).read_bytes()) for name in pick}
    consolas, arial, antiqua = faces["consolab.ttf"][0][0], faces["Arial.ttf"][0][0], faces["BookAntiqua.ttc"][0][0]
    names = {consolas, f"{arial}-Regular", antiqua, "NoSuchFont"}
    first = bs.web_fonts(tree, names)
    assert first == {consolas: {"file": "MDW/Fonts/consolab.ttf", "bytes": files["consolab.ttf"].stat().st_size},
                     f"{arial}-Regular": {"file": "MDW/Fonts/Arial.ttf", "bytes": files["Arial.ttf"].stat().st_size}}
    monkeypatch.setattr(bs, "font_bytes", lambda path: (_ for _ in ()).throw(AssertionError(f"read {path} again")))
    assert bs.web_fonts(tree, names) == first      # from site/.cache/fonts.json
    monkeypatch.setattr(bs, "FONT_WEB_MAX", 10)
    assert bs.web_fonts(tree, names) == {}
    monkeypatch.setattr(bs, "FONT_WEB_MAX", 8 * 1024 * 1024)
    assert set(bs.web_fonts(tree, {"Papyrus-Regular"})) == set()          # an alias to a face not in this folder
    assert bs.FONT_ALIASES == {"Papyrus-Regular": "Papyrus", "Kung-Fu-Panda": "KungFuPanda"}


def test_font_aliases_are_the_bots(bs):
    """The build resolves names with the bot's FONT_ALIASES (a copy: change both)."""
    from global_config import FONT_ALIASES
    assert bs.FONT_ALIASES == FONT_ALIASES
