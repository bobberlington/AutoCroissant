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
out([p.renderedHtml(res("baked"), v), p.renderedHtml(res("refused", { reason: "stat>10: hp 25", image: "refused" }), v),
  p.renderedHtml(res("refused", { reason: "overflow" }), { ...v, images: [] }), p.renderedHtml(res("error", { reason: "input: bad edit" }), v),
  p.renderedHtml(res("error", { reason: "error: timeout", transient: true }), v), p.renderedHtml(e("running"), v), p.renderedHtml(null, { ...v, n: 0 }),
  p.renderedHtml(res("baked"), { ...v, hash: "other" }), p.renderedHtml(res("baked"), { ...v, images: [] }),
  p.renderedHtml(res("error", { reason: "error: timeout", transient: true }), { ...v, hash: "other" })]);""")
    baked, refused_img, refused, bug, transient, running, none, stale, no_file, err_stale = out
    assert 'data-pv-img="current"' in baked and 'src="https://r/p.webp"' in baked and 'href="https://r/p.png"' in baked and 'data-pv="diff"' in baked
    assert "HP baked" in baked and "&lt;script&gt;" in baked and "<script>" not in baked and "Mini &lt;Doomer&gt;" in baked and "3 min ago" in baked
    assert "stat&gt;10: hp 25" in refused_img and "before refusing" in refused_img and "data-pv-img" not in refused
    assert "website bug" in bug and "try again" in transient.lower()
    assert "pv-spin" in running and "1 to 3 minutes" in running and "16 min" in running and "disabled" in none
    assert "Out of date: the edit changed since" in stale
    assert "no image came with the result" in no_file and 'class="err"' not in no_file and "HP baked" in no_file   # not an error
    assert "Out of date" in err_stale and "Try again" not in err_stale and "Preview again" in err_stale   # Try again would send the newer edit


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
    if not hasattr(module, "web_fonts"):
        pytest.skip(f"{BUILD_SITE} makes no data/fonts.json: land TTSCardMaker work/site2-previews")
    return module

