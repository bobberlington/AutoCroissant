"""
Token variants (2026-10-23): the variants the stoogedings font draws ([fire:glow], [fire:darkglow]...) are the parser's
ICON_VARIANTS, and the parser reads all but :no as the plain icon; every name and alias with the longest variant is one
token to the parser, the bake, the emojis and the site. Reads the clone's fonts and the site from $AC_SITE_SRC or the
clone; skips without them.
"""
import json
import re
from subprocess import run as run_process

import pytest

from conftest import CLONE, SITE_SRC

BUILD = CLONE / "MDW" / "Fonts" / "stoogedings-src" / "build_stoogedings.py"
FONT = CLONE / "MDW" / "Fonts" / "stoogedings.ttf"
TEXT_JS = SITE_SRC / "site" / "assets" / "text.js"

pytestmark = pytest.mark.skipif(not (BUILD.is_file() and TEXT_JS.is_file()), reason=f"no {BUILD} or {TEXT_JS}")


def shaper(order: list):
    """texts -> {text: [glyph name]} (uharfbuzz)."""
    import uharfbuzz as hb
    font = hb.Font(hb.Face(hb.Blob.from_file_path(str(FONT))))

    def one(text: str) -> list:
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(font, buf, {})
        return [order[i.codepoint] for i in buf.glyph_infos]
    return lambda texts: {t: one(t) for t in texts}


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
SITE = SITE_SRC / "site" / "assets"
LABEL_FONT = CLONE / "MDW" / "Fonts" / "Roboto Black.ttf"
RING_SAMPLE = ["1CD", "2CD", "3GCD", "8CD", "-1/1CD", "-1/2GCD", "-5/2CD", "1/2CD", "2/2CD", "12CD", "4/12CD", "-3/12GCD",
               "-9/12CD", "XCD", "XGCD", "X/XCD", "X/2CD", "1/XCD", "-1/XCD", "-3/XGCD", "20GCD", "11CD", "1/12CD",
               "-2/2CD", "-2/1CD", "-3/1GCD", "-6/1CD", "-9/1CD", "-4/3CD", "-9/8GCD"]
DEEP_BY_SHARE = [(12, -13), (12, -25), (12, -30), (9, -10)]   # past 8 segments (website only): by share


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


def _py_commands(build, contours):
    """build_stoogedings contours as text.js path commands: (M|L, x, y), (A, r, r, 0, large, sweep, x, y), (Z,); a whole
    circle as two halves (SVG can't draw one arc of 360 degrees)."""
    out = []
    for c in contours:
        for op in c:
            if op[0] != "A":
                out.append((op[0], op[1], op[2]))
                continue
            _, r, a0, a1 = op
            for f, t in ([(a0, (a0 + a1) / 2), ((a0 + a1) / 2, a1)] if abs(a1 - a0) >= 360 - 1e-9 else [(a0, a1)]):
                out.append(("A", r, r, 0, int(abs(t - f) > 180), int(t > f), *build._rpt(r, t)))
        out.append(("Z",))
    return out


def _js_commands(d: str):
    toks, out, i = re.findall(r"[MLAZ]|-?\d*\.?\d+(?:e-?\d+)?", d), [], 0
    while i < len(toks):
        n = {"M": 2, "L": 2, "A": 7, "Z": 0}[toks[i]]
        out.append((toks[i], *map(float, toks[i + 1:i + 1 + n])))
        i += 1 + n
    return out


def test_ring_shapes_agree(build, tmp_path):
    """(owner 2026-11-04, deep starts) the same shapes point by point, colour and salt: text.js ringParts' path data
    against build_stoogedings ring_parts' contours (arc radii and flags, points within toFixed(2)), every sample without
    an X (an X ring's dashes come in another order) and deep starts by share past 8 segments."""
    keys = [t for t in RING_SAMPLE if "X" not in t] + [f"{s}/{c}" for c, s in DEEP_BY_SHARE]
    js = site_node(tmp_path, f"""const out = {{}}, runs = ps => ps.reduce((o, [p, d]) => {{
  if (o.length && o.at(-1)[0] === p) o.at(-1)[1] += d; else o.push([p, d]); return o; }}, []);
for (const key of {json.dumps(keys)}) {{ const m = /^(-?\\d+)\\/(\\d+)$/.exec(key), c = m ? {{ cap: +m[2], start: +m[1], y: false }} : t.cooldownOf(key);
  out[key] = [runs(t.ringParts(c).parts), runs(t.ringParts(c, true).parts)]; }}
console.log(JSON.stringify(out));""")
    for key in keys:
        cap, start = build.cooldown_of(key) if key in RING_SAMPLE else map(int, reversed(key.split("/")))
        for one, got in zip((False, True), js[key]):
            parts = [(p, cs) for p, cs in build.ring_parts(cap, start, one=one)[0] if p not in ("body", "rim")]
            assert [p for p, _ in parts] == [p for p, _ in got], (key, one)
            for (_, cs), (_, d) in zip(parts, got):
                a, b = _py_commands(build, cs), _js_commands(d)
                assert len(a) == len(b) and all(x[0] == y[0] and len(x) == len(y) for x, y in zip(a, b)), (key, one)
                assert all(x[4:6] == y[4:6] for x, y in zip(a, b) if x[0] == "A"), (key, one)
                assert max((abs(u - v) for x, y in zip(a, b) for u, v in zip(x[1:], y[1:])), default=0) < 0.006, (key, one)


def test_deep_starts_draw_lap_rings(build):
    """A start more than a whole ring below zero (owner 2026-11-04, "C2"): its red band thinner (RING_DEEP_IN) and a thin
    red lap ring per further ring, up to 2 (RING_LAPS), on the segments that deep; a start up to a whole ring below
    zero keeps the band from RING_IN, no lap ring; salt: the band from RING_DEEP_IN, the lap rings in ink."""
    radii = lambda cs: sorted({op[1] for c in cs for op in c if op[0] == "A"})
    for tag, (bands, laps) in {"-1/1CD": (1, 0), "-2/2CD": (2, 0), "-2/1CD": (1, 1), "-3/1GCD": (1, 2), "-9/1CD": (1, 2),
                               "-3/2CD": (2, 1), "-5/2CD": (2, 3), "-6/2CD": (2, 4), "-4/3CD": (3, 1), "-9/8GCD": (8, 1)}.items():
        cap, start = build.cooldown_of(tag)
        red = [cs for p, cs in build.ring_parts(cap, start)[0] if p == "red"]
        assert len(red) == 1 and len(red[0]) == bands + laps, tag
        inner = build.RING_DEEP_IN if laps else build.RING_IN
        rings = build.RING_LAPS[:-(-(-start - cap) // cap)] if laps else []   # as deep as its deepest segment
        assert radii(red[0]) == sorted({inner, build.RING_OUT, *(r for lap in rings for r in lap)}), tag
        ink = [c for p, cs in build.ring_parts(cap, start, one=True)[0] if p == "ink" for c in cs]
        assert len(ink) == bands + laps + 1, tag   # and the divider
        assert {inner, inner + build.SALT_BAND} <= set(radii(ink)), tag
    assert build.RING_LAPS[1][0] < build.RING_LAPS[1][1] < build.RING_LAPS[0][0] < build.RING_LAPS[0][1] < build.RING_DEEP_IN


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
