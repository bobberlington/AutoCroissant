"""
Token variants (2026-10-23): the variants the stoogedings font draws ([fire:glow], [fire:darkglow]...) are the parser's
ICON_VARIANTS, and the parser reads all but :no as the plain icon; every name and alias with the longest variant is one
token to the parser, the bake, the emojis and the site. Reads the card repo clone; skips without it.
"""
import json
import re
import shutil
from subprocess import run as run_process

import pytest

from conftest import CLONE

BUILD = CLONE / "MDW" / "Fonts" / "stoogedings-src" / "build_stoogedings.py"
FONT = CLONE / "MDW" / "Fonts" / "stoogedings.ttf"
TEXT_JS = CLONE / "site" / "assets" / "text.js"

pytestmark = pytest.mark.skipif(not (BUILD.is_file() and TEXT_JS.is_file()), reason=f"no card repo clone at {CLONE}")


def shaper(order: list):
    """texts -> {text: [glyph name]}: uharfbuzz, else the hb-shape tool, else skip."""
    try:
        import uharfbuzz as hb
    except ImportError:
        hb = None
    if hb:
        font = hb.Font(hb.Face(hb.Blob.from_file_path(str(FONT))))

        def one(text: str) -> list:
            buf = hb.Buffer()
            buf.add_str(text)
            buf.guess_segment_properties()
            hb.shape(font, buf, {})
            return [order[i.codepoint] for i in buf.glyph_infos]
        return lambda texts: {t: one(t) for t in texts}
    if not shutil.which("hb-shape"):
        pytest.skip("needs uharfbuzz or hb-shape")

    def tool(texts: list) -> dict:
        lines = run_process(["hb-shape", "--no-glyph-names", "--no-positions", "--no-clusters", "--text-file=/dev/stdin",
                             str(FONT)], input="\n".join(texts) + "\n", capture_output=True, text=True,
                            check=True).stdout.splitlines()
        assert len(lines) == len(texts)
        return {t: [order[int(g)] for g in ln.strip("[]").split("|")] for t, ln in zip(texts, lines)}
    return tool


# ========================
# Token variants
# ========================
MANIFEST = CLONE / "MDW" / "Fonts" / "stoogedings-src" / "stoogedings.json"
VARIANT_PROBES = ["tint", "sticker", "no", "glow", "darkglow", "Glow", "DARKGLOW", "glo", "glows", "dark glow",
                  "dark_glow", "darkglo", "lightglow", "shadow", "outline", "bogus"]


def font_order():
    ttlib = pytest.importorskip("fontTools.ttLib")
    return ttlib.TTFont(FONT, lazy=True).getGlyphOrder()


@pytest.fixture(scope="module")
def font_variants() -> set:
    """The :variant suffixes the clone's font draws as an icon on [fire:...], lowercased."""
    order = font_order()
    got = shaper(order)([f"[fire:{v}]" for v in VARIANT_PROBES])
    return {v.lower() for v in VARIANT_PROBES if all(g.startswith("icon_") or g.endswith("_space") for g in got[f"[fire:{v}]"])}


def test_parser_variants_match_font(font_variants):
    from commands.psd_parser import ICON_VARIANTS
    assert {v.lower() for v in VARIANT_PROBES if v.lower() in ICON_VARIANTS} == font_variants


def test_parser_reads_variants_as_the_plain_icon():
    from types import SimpleNamespace
    from commands.psd_parser import IconVocabulary, PSDParser
    vocab = IconVocabulary.build(["fire", "darkmatter"], [], {"icons": {"darkmatter": {"aliases": ["dark matter"]}}})
    parser = SimpleNamespace(icons=vocab, _covered=PSDParser._covered, _add_problem=PSDParser._add_problem)
    card = SimpleNamespace(problems=[])
    text = "[fire:glow] [Fire:DarkGlow] [fire:sticker] [fire:tint] [fire:no] [dark matter:glow] [fire:glows] [fire:dark glow]"
    out, icons = PSDParser._read_icon_tokens(parser, text, [(0, len(text))], card)
    assert out == "[fire] [fire] [fire] [fire] [fire:no] [darkmatter] [fire:glows] [fire:dark glow]"
    assert [name for name, _ in icons] == ["fire"] * 5 + ["darkmatter"]
    assert card.problems == ["UNKNOWN ICON TOKEN: [fire:glows]", "UNKNOWN ICON TOKEN: [fire:dark glow]"]


def test_token_limit_fits_every_icon():
    """Every name and alias with the longest variant is one token to the parser, the bake, the emojis and the site
    ([thunderfury blessed blade of the starseeker:darkglow]: 52 characters)."""
    from commands.psd_edit import TOKEN
    from commands.psd_parser import ICON_SCAN_PATTERN, ICON_VARIANTS
    from commands.type_emojis import TOKEN_PATTERN
    if not MANIFEST.is_file():
        pytest.skip(f"no manifest in {CLONE}")
    icons = json.loads(MANIFEST.read_text())["icons"]
    longest = max((n for key, icon in icons.items() for n in [key] + icon.get("aliases", [])), key=len)
    token = f"[{longest}:{max(ICON_VARIANTS, key=len)}]"
    assert len(token) - 2 >= 52
    for pattern in (ICON_SCAN_PATTERN, TOKEN, TOKEN_PATTERN):
        assert pattern.fullmatch(token), pattern.pattern
    site = int(re.search(r"const PIECE = .*?\{1,(\d+)\}", TEXT_JS.read_text()).group(1))
    assert site >= len(token) - 2



# ========================
# Badges drawn four ways
# ========================
# (review 2026-11-04) the font (build_stoogedings), the website (text.js, style.css) and the mock-up (mockup.js) each
# draw a badge: their colours, rings and Roboto Black metrics agree.
SITE = CLONE / "site" / "assets"
LABEL_FONT = CLONE / "MDW" / "Fonts" / "Roboto Black.ttf"
RING_SAMPLE = ["1CD", "2CD", "3GCD", "8CD", "-1/1CD", "-1/2GCD", "-5/2CD", "1/2CD", "2/2CD", "12CD", "4/12CD", "-3/12GCD",
               "-9/12CD", "XCD", "XGCD", "X/XCD", "X/2CD", "1/XCD", "-1/XCD", "-3/XGCD", "20GCD", "11CD", "1/12CD"]


@pytest.fixture(scope="module")
def build():
    """The clone's build_stoogedings, cv2 stubbed (only its icon tracing needs it)."""
    import importlib.util
    import sys
    from types import ModuleType
    stub = "cv2" not in sys.modules
    if stub:
        sys.modules["cv2"] = ModuleType("cv2")
    try:
        spec = importlib.util.spec_from_file_location("build_stoogedings_clone", BUILD)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if stub:
            del sys.modules["cv2"]
    return mod


def site_node(tmp_path, body: str) -> object:
    """body's stdout as JSON, text.js imported as t."""
    if not shutil.which("node"):
        pytest.skip("needs node")
    script = tmp_path / "probe.mjs"
    script.write_text(f"const t = await import({json.dumps(TEXT_JS.as_uri())});\n{body}\n")
    return json.loads(run_process(["node", str(script)], capture_output=True, text=True, check=True, timeout=60).stdout)


def test_badge_colours_agree(build):
    """Group and cost pills, cost words and the ring's paints: build_stoogedings, style.css, mockup.js and text.js."""
    css, mock, text = (SITE / "style.css").read_text(), (SITE / "mockup.js").read_text(), TEXT_JS.read_text()
    trio = lambda sel: tuple(re.search(re.escape(sel) + r" \{ --bt: (#\w+); --bb: (#\w+); --be: (#\w+);", css).groups())
    pill = lambda key: tuple(re.findall(r"#\w+", re.search(rf"\b{key}: (\[[^\]]+\])", mock).group(1)))
    for group, colours in build.BADGE_GROUPS.items():
        assert trio(f".badge.{group}") == colours == pill(group), group
    for kind, (_, _, colours, word) in build.COSTS.items():
        own = kind in ("souls", "counters")
        assert trio(".badge.cost.own" if own else f".badge.cost.{kind}") == colours == pill("own" if own else kind), kind
        if not own:
            assert re.search(rf"\.badge\.cost\.{kind} \{{[^}}]*; color: (#\w+)", css).group(1) == word
            assert re.search(rf"\b{kind}: \"(#\w+)\"", mock.split("const COST_WORD")[1]).group(1) == word
    js = lambda name: re.search(rf"\b{name} = (\"#\w+\"|\[[^\]]+\])", text).group(1)
    for name in ("RING_GREEN", "RING_RED", "RING_BODY", "RING_HOLLOW", "RING_FAINT", "RING_NUMBER"):
        assert tuple(re.findall(r"#\w+", js(name))) == tuple(re.findall(r"#\w+", str(getattr(build, name)))), name
    rim = re.search(r"const RING_RIM = \{ turn: (\[[^\]]+\]), any: (\[[^\]]+\]) \}", text).groups()
    for group, stops in zip(("turn", "any"), rim):
        assert tuple(re.findall(r"#\w+", stops)) == build.RING_RIM[group], group
        assert re.search(rf"\.badge\.{group} \.ring \{{ background: linear-gradient\(180deg, (#\w+), (#\w+)\)", css).groups() == build.RING_RIM[group]


def test_rings_agree(build, tmp_path):
    """text.js ringParts against build_stoogedings ring_parts for a sample of cooldowns: the paints in order and each
    one's outlines (colour and salt), the number's characters (origin and scale, box units) and its colour."""
    from fontTools.ttLib import TTFont
    label = TTFont(LABEL_FONT)
    glyphs, cmap = label.getGlyphSet(), label.getBestCmap()
    caps = build.cap_height(glyphs, cmap)
    js = site_node(tmp_path, f"""const runs = ps => ps.reduce((o, [p, d]) => {{ const n = (d.match(/M/g) || []).length;
  if (o.length && o.at(-1)[0] === p) o.at(-1)[1] += n; else o.push([p, n]); return o; }}, []);
const out = {{}};
for (const tag of {json.dumps(RING_SAMPLE)}) {{ const c = t.cooldownOf(tag), r = t.ringParts(c), s = t.ringParts(c, true);
  out[tag] = {{ paints: runs(r.parts), salt: runs(s.parts), number: r.number.map(([, x, y, k]) => [x, y, k]), colour: r.colour }}; }}
console.log(JSON.stringify(out));""")
    for tag in RING_SAMPLE:
        cooldown = build.cooldown_of(tag)
        parts, (number, colour, _) = build.ring_parts(*cooldown)
        salt, _ = build.ring_parts(*cooldown, one=True)
        got = js[tag]
        assert [[p, len(cs)] for p, cs in parts if p not in ("body", "rim")] == got["paints"], tag
        assert [[p, len(cs)] for p, cs in salt] == got["salt"], tag
        assert colour == got["colour"], tag
        placed = build.ring_number(glyphs, cmap, caps, number)
        assert len(placed) == len(got["number"]), tag
        assert all(abs(a - b) < 1e-3 for (_, *p), q in zip(placed, got["number"]) for a, b in zip(p, q)), tag


def test_web_metrics_match_the_label_font(build):
    """text.js's Roboto Black metrics (RING_DIGITS, inkShift's LSB and RSB, costHtml's POINT) are the label font's as
    badges set it (build_stoogedings _set_ink: LABEL_METRICS), as badge_web_font.py writes them into the woff2."""
    from fontTools.ttLib import TTFont
    label = TTFont(LABEL_FONT)
    glyphs, cmap = label.getGlyphSet(), label.getBestCmap()
    text = TEXT_JS.read_text()
    digits = json.loads(re.search(r"const RING_DIGITS = (\{.*?\});\n", text).group(1))
    for ch, (advance, x0, y0, x1, y1, _) in digits.items():
        ink = build._set_ink(glyphs, cmap, ch)
        assert [advance, x0, y0, x1, y1] == pytest.approx([ink[4], ink[0], -ink[3], ink[2], -ink[1]], abs=0.06), ch
    lsb = json.loads(re.search(r"const LSB = (\{.*?\});\n", text).group(1))
    rsb = json.loads(re.search(r"const RSB = (\{.*?\});\n", text).group(1))
    assert {ch: pytest.approx(build._set_ink(glyphs, cmap, ch)[0], abs=0.06) for ch in lsb} == lsb
    assert {ch: pytest.approx((lambda d: d[4] - d[2])(build._set_ink(glyphs, cmap, ch)), abs=0.06) for ch in rsb} == rsb
    tags = [t for t, _ in build.badge_tags()] + [f"{n} {w}" for n in ("1", "X", "12") for w in ("Soul", "Souls", "Counter", "Counters")]
    assert {build.BADGE_LABEL.get(t[0], t[0]) for t in tags} <= set(lsb) and {t[-1] for t in tags} <= set(rsb)
    point = json.loads(re.search(r"POINT = (\[[^\]]+\])", text).group(1))
    dot = build._set_ink(glyphs, cmap, ".")
    assert [point[0], point[1], point[3]] == [round(dot[4]), round(dot[0]), round(dot[2])]


def test_label_places_agree(build):
    """A label's place (review 2026-11-04): after a ring by RING_ENDS (style.css .badge.ringed's padding, the mock-up's
    shift), a plain one by BADGE_PAD (.badge's padding; both by their ink: --ink), a 1 nudged optically (RING_ONE: text.js
    ONE, the mock-up's coin)."""
    css, mock, text = (SITE / "style.css").read_text(), (SITE / "mockup.js").read_text(), TEXT_JS.read_text()
    em = 1.45 / build.RING_D
    after = (build.RING_GAP + build.BADGE_PAD - build.RING_ENDS) / 2
    left, right = map(float, re.search(r"\.badge\.ringed \{ padding-left: calc\(([\d.]+)em \+ var\(--ink, 0em\)\); "
                                       r"padding-right: calc\(([\d.]+)em - var\(--ink, 0em\)\); \}", css).groups())
    assert left == pytest.approx((build.RING_D + after) * em, abs=1e-4)
    assert right == pytest.approx((build.RING_GAP + build.BADGE_PAD - after - build.BADGE_TRACKING) * em, abs=1e-4)
    right, left = map(float, re.search(r"padding: 0 calc\(([\d.]+)em - var\(--ink, 0em\)\) 0 calc\(([\d.]+)em \+ "
                                       r"var\(--ink, 0em\)\);", css).groups())
    assert left == pytest.approx(build.BADGE_PAD * em, abs=1e-4)
    assert right == pytest.approx((build.BADGE_PAD - build.BADGE_TRACKING) * em, abs=1e-4)
    assert float(re.search(r"\(x0 \+ h \+ x1\) / 2 - ([\d.]+) \* em", mock).group(1)) == build.RING_ENDS / 2 / 1000
    one = re.search(r"\bONE = \[(-?[\d.]+), (-?[\d.]+)\]", text).groups()
    assert tuple(map(float, one)) == build.RING_ONE
    assert re.search(r"\(c\.n\.length > 1 \? (-?[\d.]+) / ([\d.]+) : (-?[\d.]+) / ([\d.]+)\)", mock).groups() == tuple(
        map(str, (build.RING_ONE[1], build.RING_CAPS[1], build.RING_ONE[0], build.RING_CAPS[0])))
