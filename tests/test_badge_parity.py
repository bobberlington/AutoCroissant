"""
Timing-tag badges: the stoogedings font's badge_tags(), the website's TAG/tagInfo and the migration walk's TAG must
hold the same tags (companion skill §1.2, "three places"); std batch copies may only lag by BATCH_LAG. Any letter case
is the same tag since the font's CASE lookup (2026-10-22): "(action)" draws and shows the (Action) badge. Reads the card
repo clone; skips without it or node, when the clone is behind the walk, or while the walk is still exact-case.
Token variants too (2026-10-23): the variants the font draws ([fire:glow], [fire:darkglow]...) are the parser's
ICON_VARIANTS, and the parser reads all but :no as the plain icon; so does the website without variant images (an older
build; with them it draws each variant, tests/test_token_picker.py).
"""
import ast
import json
import re
import shutil
from os.path import expanduser
from pathlib import Path
from subprocess import run as run_process

import pytest

from conftest import CLONE

BUILD = CLONE / "MDW" / "Fonts" / "stoogedings-src" / "build_stoogedings.py"
FONT = CLONE / "MDW" / "Fonts" / "stoogedings.ttf"
TEXT_JS = CLONE / "site" / "assets" / "text.js"
WALK = Path(expanduser("~/Documents/Tools/psd-research/std"))   # out of repo; this Mac only
WALK_PATCH = WALK / "case_insensitive_tag.patch"                 # the walk's any-case TAG, not applied yet
BATCH_LAG = {"std-batch-2026-10-22": {"Action"}}                 # started before the tag

pytestmark = pytest.mark.skipif(not (BUILD.is_file() and TEXT_JS.is_file()), reason=f"no card repo clone at {CLONE}")


def build_fns() -> dict:
    """badge_tags(), badge_glyph() and spellings() run on their own (the build script imports cv2 etc.)."""
    tree = ast.parse(BUILD.read_text())
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("badge_tags", "badge_glyph", "spellings")]
    ns = {"re": re}
    exec(compile(ast.Module(fns, []), str(BUILD), "exec"), ns)
    return ns


def spellings(tag: str) -> list:
    """The tag, then every other upper/lower combination of its letters (2**letters in all)."""
    letters = [i for i, c in enumerate(tag) if c.isalpha()]
    out = [tag]
    for mask in range(2 ** len(letters)):
        s = list(tag)
        for k, i in enumerate(letters):
            s[i] = s[i].upper() if mask >> k & 1 else s[i].lower()
        out.append("".join(s))
    return list(dict.fromkeys(out))


def node(tmp_path: Path, src: str) -> str:
    """Run an ES module in node and return its stdout."""
    script = tmp_path / "probe.mjs"
    script.write_text(src)
    return run_process(["node", str(script)], capture_output=True, text=True, check=True).stdout


@pytest.fixture(scope="module")
def font() -> dict:
    """{tag: group}"""
    tags = dict(build_fns()["badge_tags"]())
    assert len(tags) > 300   # 341 on 2026-10-22
    return tags


@pytest.fixture(scope="module")
def cased(font) -> dict:
    """{spelling: tag} for every tag in every case (2530 for 341 tags)"""
    out = {s: t for t in font for s in spellings(t)}
    assert len(out) > 4 * len(font)
    return out


@pytest.fixture(scope="module")
def any_case() -> bool:
    """The clone's font has the CASE lookup (TTSCardMaker work/badges-any-case); an older one is exact-case."""
    return "lookup CASE" in BUILD.read_text()


def expected(cased: dict, any_case: bool) -> dict:
    """{spelling: tag} a consumer must draw as a badge: every case, or each tag's own case before CASE."""
    return {s: t for s, t in cased.items() if any_case or s == t}


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


def check(found: set, font: dict, cased: dict, any_case: bool, where: str, lag: set = frozenset(), exact_ok=False):
    """found, read in each tag's own case, must equal the font's tags less lag (skip if the clone's font is the one
    behind); then found holds every case of those tags, or only their own (exact_ok, or skip until the patch)."""
    own = {cased.get(t, t) for t in found}
    want = set(font) - lag
    if own > want and not (own - want) & lag:
        pytest.skip(f"clone behind {where}: {CLONE}'s font lacks {sorted(own - want)}; pull TTSCardMaker")
    assert own == want, where
    every = {s for s, t in cased.items() if t in own}
    assert found in (own, every), f"{where} matches some other-case spellings only"
    if found == every and not any_case:
        pytest.skip(f"clone behind {where}: {CLONE}'s font predates any-case badges; pull TTSCardMaker")
    if found == own and any_case and not exact_ok:
        pytest.skip(f"{where} still exact-case: apply {WALK_PATCH}")


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
def glow_build() -> bool:
    """The clone's build script makes :glow and :darkglow (TTSCardMaker work/glow-tokens); an older one doesn't."""
    return "GLOWS" in BUILD.read_text()


def behind(glow_build: bool) -> None:
    if not glow_build:
        pytest.skip(f"clone behind: {CLONE}'s font predates :glow; pull TTSCardMaker")


@pytest.fixture(scope="module")
def font_variants() -> set:
    """The :variant suffixes the clone's font draws as an icon on [fire:...], lowercased."""
    order = font_order()
    got = shaper(order)([f"[fire:{v}]" for v in VARIANT_PROBES])
    return {v.lower() for v in VARIANT_PROBES if all(g.startswith("icon_") or g.endswith("_space") for g in got[f"[fire:{v}]"])}


def test_parser_variants_match_font(font_variants, glow_build):
    from commands.psd_parser import ICON_VARIANTS
    behind(glow_build)
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

