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
    assert "pv-spin" in running and "1 to 3 minutes" in running and "16 min" in running and "disabled" in none
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
