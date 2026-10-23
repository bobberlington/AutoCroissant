"""
Timing-tag badges: the stoogedings font's badge_tags(), the website's TAG/tagInfo and the migration walk's TAG must
hold the same tags (companion skill §1.2, "three places"); std batch copies may only lag by BATCH_LAG. Any letter case
is the same tag since the font's CASE lookup (2026-10-22): "(action)" draws and shows the (Action) badge. Reads the card
repo clone; skips without it or node, when the clone is behind the walk, or while the walk is still exact-case.
Token variants too (2026-10-23): the variants the font draws ([fire:glow], [fire:darkglow]...) are the parser's
ICON_VARIANTS, and the parser and the website read all but :no as the plain icon.
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
BATCHES = Path(expanduser("~/Documents/Tools/psd-research"))     # std-batch-*/ copies of the walk's TAG
BATCH_LAG = {"std-batch-2026-10-22": {"Action"}}                 # started before the tag
# near misses in any case, and tags one place may add first
PROBES = ["Actions", "actions", "ACTIONS", "Actio", "OPT", "opt", "1 point", "0CD", "0cd", "10CD", "-1CD", "-2/1GCD",
          "-2/1gcd", "1/0CD", "XX", "xx", "0U", "1PGX", "1pgx", "cd", "x cd", "1 cd",
          "oſ", "Actıon", "Actİon"]   # Unicode folds these to tags; ASCII case only (the font, JS /i, (?ai:))

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


def test_font_draws_any_case(cased, any_case):
    ttlib = pytest.importorskip("fontTools.ttLib")
    glyph = build_fns()["badge_glyph"]
    tt = ttlib.TTFont(FONT)
    order, cmap = tt.getGlyphOrder(), tt.getBestCmap()
    shape = shaper(order)
    plain = "(actions) (Action (cd) action (x cd) (ACTIONS)"
    cands = sorted(set(cased) | set(PROBES))
    got = shape([f"({s})" for s in cands] + [plain])
    badges = {s: got[f"({s})"][0] for s in cands if any(x.startswith("badge_") for x in got[f"({s})"])}
    assert badges == {s: glyph(t) for s, t in expected(cased, any_case).items()}
    assert all(len(got[f"({s})"]) == 1 for s in badges)                 # one glyph for the whole "(tag)"
    assert got[plain] == [cmap[ord(c)] for c in plain]                  # not a tag: its letters, as typed


@pytest.mark.skipif(not shutil.which("node"), reason="needs node")
def test_site_matches_font(font, cased, any_case, tmp_path):
    shutil.copy(TEXT_JS, tmp_path / "text.mjs")
    cands = sorted(set(cased) | set(PROBES))
    out = node(tmp_path, f"""import {{ TAG, tagInfo, badgeHtml, GROUP_NAMES }} from "./text.mjs";
const res = {{}};
for (const t of {json.dumps(cands)}) {{
  const hit = [...`x (${{t}}) y`.matchAll(new RegExp(TAG.source, "g"))].map(m => m[1]);
  const info = hit.length === 1 && hit[0] === t ? tagInfo(t) : null;
  const tag = info && (info.tag ?? t);   // a site from before any case has no info.tag
  if (info) res[t] = [tag, info.group, info.group in GROUP_NAMES, badgeHtml(t) === badgeHtml(tag)];
}}
console.log(JSON.stringify(res));""")
    site = json.loads(out)
    want = {s: (t, font[t]) for s, t in expected(cased, any_case).items()}
    assert {s: (t, g) for s, (t, g, _, _) in site.items()} == want
    assert all(named for _, _, named, _ in site.values())   # every group has a glossary heading
    assert all(same for _, _, _, same in site.values())     # every spelling shows its tag's own badge


def py_tags(src: str, start: str, cands: list) -> set:
    """Tags a Python raw-string TAG regex (the line after start) matches whole."""
    py = re.compile(src.split(start, 1)[1].split('"', 1)[0])
    return {t for t in cands if py.fullmatch(f"({t})")}


def jsx_tags(tmp_path: Path, src: str, var: str, cands: list) -> set:
    """Tags a JSX regex variable matches at the start of "(tag) rest"."""
    line = next(ln for ln in src.splitlines() if ln.startswith(f"var {var} = "))
    out = node(tmp_path, line.split(", STD_", 1)[0].rstrip(";") + f""";
console.log(JSON.stringify({json.dumps(cands)}.filter(t => {{ const m = ("(" + t + ") rest").substr(0, 12).match({var});
  return m && m[0] === "(" + t + ")"; }})));""")
    return set(json.loads(out))


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


@pytest.mark.skipif(not WALK.is_dir(), reason=f"no migration walk at {WALK}")
def test_walk_matches_font(font, cased, any_case, tmp_path):
    cands = sorted(set(cased) | set(PROBES))
    found = {"l1_check.py": py_tags((WALK / "l1_check.py").read_text(), 'TAG = re.compile(r"', cands)}
    if shutil.which("node"):
        for name in ("std.jsx", "std_l1.jsx"):
            found[name] = jsx_tags(tmp_path, (WALK / name).read_text(), "TAG", cands)
    for name, tags in found.items():
        assert tags == found["l1_check.py"], name   # the walk agrees with itself
    check(found["l1_check.py"], font, cased, any_case, "the walk")


@pytest.mark.skipif(not shutil.which("node"), reason="needs node")
def test_batches_match_font(font, cased, any_case, tmp_path):
    cands = sorted(set(cased) | set(PROBES))
    batches = sorted(d for d in BATCHES.glob("std-batch-*") if (d / "tools" / "std_driver.py").is_file())
    if not batches:
        pytest.skip(f"no std batch under {BATCHES}")
    for d in batches:
        py = py_tags((d / "tools" / "std_driver.py").read_text(), '\nTAG = r"', cands)
        jsx = jsx_tags(tmp_path, (d / "scripts" / "std_card.jsx").read_text(), "STD_TAG", cands)
        assert py == jsx, d.name
        check(py, font, cased, any_case, d.name, BATCH_LAG.get(d.name, set()), exact_ok=True)   # a copy as it ran


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


def test_font_draws_every_glow(glow_build):
    behind(glow_build)
    icons = json.loads(MANIFEST.read_text())["icons"]
    spell = build_fns()["spellings"]
    order = font_order()
    bases = {key: "icon_" + re.sub(r"[^a-z0-9]", "_", key) for key in icons}
    bases = {key: base for key, base in bases.items() if base in order}   # a PNG gone keeps its codepoint, no glyph
    want = {}
    for key, base in bases.items():
        for name in [key] + icons[key].get("aliases", []):
            for sp in spell(name):
                for v in ("glow", "darkglow"):
                    for text in (f"[{sp}:{v}]", f"[{sp.upper()}:{v.upper()}]"):
                        want.setdefault(text, [f"{base}.{v}_bg", base, "glow_space"])
    got = shaper(order)(list(want))
    assert {t: got[t] for t in want} == want
    assert len({tuple(w) for w in want.values()}) == 2 * len(bases) > 100   # 182 on 2026-10-23


def test_glow_halo_reach(glow_build):
    """Halos reach 0.3 em past each icon (owner 2026-10-23) and fill the token's advance."""
    behind(glow_build)
    ttlib = pytest.importorskip("fontTools.ttLib")
    font = ttlib.TTFont(FONT)
    order, em, hmtx = font.getGlyphOrder(), font["head"].unitsPerEm, font["hmtx"]
    docs = {order[g]: doc for doc, g, _ in font["SVG "].docList}
    box = lambda name: [float(v) for v in re.search(r'<image x="([-\d.]+)" y="([-\d.]+)" width="([-\d.]+)" '
                                                    r'height="([-\d.]+)"', docs[name]).groups()]
    pad = hmtx["glow_space"][0]
    halos = [n for n in docs if n.endswith("glow_bg")]
    assert len(halos) > 100
    for name in halos:
        icon = name.rsplit(".", 1)[0]
        (hx, hy, hw, hh), (ix, iy, iw, ih) = box(name), box(icon)
        reach = (pad + ix - hx, iy - hy, hx + hw - (pad + ix + iw), hy + hh - (iy + ih))
        assert all(abs(r - 0.3 * em) <= 2 for r in reach), (name, reach)
        assert hmtx[name][0] == pad and abs(hx) <= 2 and abs(hx + hw - (2 * pad + hmtx[icon][0])) <= 2, name


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


@pytest.mark.skipif(not shutil.which("node"), reason="needs node")
def test_site_reads_variants_as_the_plain_icon(tmp_path):
    from commands.psd_parser import ICON_VARIANTS
    shutil.copy(TEXT_JS, tmp_path / "text.mjs")
    out = node(tmp_path, f"""import {{ makeRenderer, plain }} from "./text.mjs";
const r = makeRenderer({{ cards: [], icons: {{ fire: {{ file: "fire.webp" }} }} }});
const res = {{}};
for (const v of {json.dumps([v for v in ICON_VARIANTS if v])}) res[v] = [r.render(`a [fire:${{v}}] b`) === r.render("a [fire] b"), plain(`[fire:${{v}}]`)];
console.log(JSON.stringify(res));""")
    site = json.loads(out)
    assert {v: same for v, (same, _) in site.items()} == {v: v != "no" for v in ICON_VARIANTS if v}
    assert {v: flat for v, (_, flat) in site.items()} == {v: "non-fire" if v == "no" else "fire" for v in ICON_VARIANTS if v}


def test_site_previews_flatten_variants(glow_build):
    from importlib.util import module_from_spec, spec_from_file_location
    import sys
    from commands.psd_parser import ICON_VARIANTS
    behind(glow_build)
    build = CLONE / "site" / "build"
    sys.path.insert(0, str(build))
    try:
        spec = spec_from_file_location("site_build_parity", build / "build_site.py")
        site = module_from_spec(spec)
        spec.loader.exec_module(site)
    except ImportError as e:
        pytest.skip(f"build_site.py needs {e.name}")
    finally:
        sys.path.remove(str(build))
    for v in ICON_VARIANTS:
        text = site.share_text({"type": "item", "ability": f"Burn [fire{':' + v if v else ''}]"})
        assert text.endswith("Burn non-Fire" if v == "no" else "Burn Fire"), v
