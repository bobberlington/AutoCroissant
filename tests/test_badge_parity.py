"""
Timing-tag badges: the stoogedings font's badge_tags(), the website's TAG/tagInfo and the migration walk's TAG must
hold the same tags (companion skill §1.2, "three places"); std batch copies may only lag by BATCH_LAG. Any letter case
is the same tag since the font's CASE lookup (2026-10-22): "(action)" draws and shows the (Action) badge. Reads the card
repo clone; skips without it or node, when the clone is behind the walk, or while the walk is still exact-case.
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
          "-2/1gcd", "1/0CD", "XX", "xx", "0U", "1PGX", "1pgx", "cd", "x cd", "1 cd"]

pytestmark = pytest.mark.skipif(not (BUILD.is_file() and TEXT_JS.is_file()), reason=f"no card repo clone at {CLONE}")


def build_fns() -> dict:
    """badge_tags() and badge_glyph() run on their own (the build script imports cv2 etc.)."""
    tree = ast.parse(BUILD.read_text())
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("badge_tags", "badge_glyph")]
    ns = {}
    exec(compile(ast.Module(fns, []), str(BUILD), "exec"), ns)
    return ns


def spellings(tag: str) -> list:
    """The tag, then lower, upper and title case and two alternating mixed cases of its letters, without repeats."""
    letters = [i for i, c in enumerate(tag) if c.isalpha()]

    def alternate(first_upper: bool) -> str:
        out = list(tag.lower())
        for k, i in enumerate(letters):
            if (k % 2 == 0) == first_upper:
                out[i] = out[i].upper()
        return "".join(out)
    return list(dict.fromkeys([tag, tag.lower(), tag.upper(), tag.title(), alternate(True), alternate(False)]))


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
    """{spelling: tag} for every tag in every case (1540 for 341 tags)"""
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


def test_font_draws_any_case(cased, any_case):
    hb = pytest.importorskip("uharfbuzz")
    ttlib = pytest.importorskip("fontTools.ttLib")
    glyph = build_fns()["badge_glyph"]
    tt = ttlib.TTFont(FONT)
    order, cmap = tt.getGlyphOrder(), tt.getBestCmap()
    shaper = hb.Font(hb.Face(hb.Blob.from_file_path(str(FONT))))

    def shape(text: str) -> list:
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(shaper, buf, {})
        return [order[i.codepoint] for i in buf.glyph_infos]
    got = {s: shape(f"({s})") for s in sorted(set(cased) | set(PROBES))}
    badges = {s: g[0] for s, g in got.items() if any(x.startswith("badge_") for x in g)}
    assert badges == {s: glyph(t) for s, t in expected(cased, any_case).items()}
    assert all(len(got[s]) == 1 for s in badges)                      # one glyph for the whole "(tag)"
    plain = "(actions) (Action (cd) action (x cd) (ACTIONS)"
    assert shape(plain) == [cmap[ord(c)] for c in plain]              # not a tag: its letters, as typed


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
