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

