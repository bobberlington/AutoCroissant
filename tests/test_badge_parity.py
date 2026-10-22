"""
Timing-tag badges: the stoogedings font's badge_tags(), the website's TAG/tagInfo and the migration walk's TAG must
hold the same tags (companion skill §1.2, "three places"); std batch copies may only lag by BATCH_LAG. Reads the card
repo clone; skips without it or node, or when the clone is behind the walk.
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
TEXT_JS = CLONE / "site" / "assets" / "text.js"
WALK = Path(expanduser("~/Documents/Tools/psd-research/std"))   # out of repo; this Mac only
BATCHES = Path(expanduser("~/Documents/Tools/psd-research"))     # std-batch-*/ copies of the walk's TAG
BATCH_LAG = {"std-batch-2026-10-22": {"Action"}}                 # started before the tag
PROBES = ["Action", "action", "ACTION", "Actions", "Actio", "OPT", "1 point", "0CD", "10CD", "-1CD", "-2/1GCD",
          "1/0CD", "1/2cd", "XX", "0U", "1PGX"]               # near misses, and tags one place may add first

pytestmark = pytest.mark.skipif(not (BUILD.is_file() and TEXT_JS.is_file()), reason=f"no card repo clone at {CLONE}")


def font_tags() -> dict:
    """badge_tags() run on its own (the build script imports cv2 etc.): {tag: group}."""
    tree = ast.parse(BUILD.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "badge_tags")
    ns = {}
    exec(compile(ast.Module([fn], []), str(BUILD), "exec"), ns)
    return dict(ns["badge_tags"]())


def node(tmp_path: Path, src: str) -> str:
    """Run an ES module in node and return its stdout."""
    script = tmp_path / "probe.mjs"
    script.write_text(src)
    return run_process(["node", str(script)], capture_output=True, text=True, check=True).stdout


@pytest.fixture(scope="module")
def font() -> dict:
    tags = font_tags()
    assert len(tags) > 300   # 341 on 2026-10-22
    return tags


@pytest.mark.skipif(not shutil.which("node"), reason="needs node")
def test_site_matches_font(font, tmp_path):
    shutil.copy(TEXT_JS, tmp_path / "text.mjs")
    cands = sorted(set(font) | set(PROBES))
    out = node(tmp_path, f"""import {{ TAG, tagInfo, GROUP_NAMES }} from "./text.mjs";
const res = {{}};
for (const t of {json.dumps(cands)}) {{
  const hit = [...`x (${{t}}) y`.matchAll(new RegExp(TAG.source, "g"))].map(m => m[1]);
  const info = hit.length === 1 && hit[0] === t ? tagInfo(t) : null;
  if (info) res[t] = [info.group, info.group in GROUP_NAMES];
}}
console.log(JSON.stringify(res));""")
    site = json.loads(out)
    assert {t: g for t, (g, _) in site.items()} == font
    assert all(named for _, named in site.values())   # every group has a glossary heading


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


def check(found: set, font: dict, where: str, lag: set = frozenset()):
    """found must equal the font's tags less lag; skip if the clone's font is the one behind."""
    want = set(font) - lag
    if found > want and not (found - want) & lag:
        pytest.skip(f"clone behind {where}: {CLONE}'s font lacks {sorted(found - want)}; pull TTSCardMaker")
    assert found == want, where


@pytest.mark.skipif(not WALK.is_dir(), reason=f"no migration walk at {WALK}")
def test_walk_matches_font(font, tmp_path):
    cands = sorted(set(font) | set(PROBES))
    found = {"l1_check.py": py_tags((WALK / "l1_check.py").read_text(), 'TAG = re.compile(r"', cands)}
    if shutil.which("node"):
        for name in ("std.jsx", "std_l1.jsx"):
            found[name] = jsx_tags(tmp_path, (WALK / name).read_text(), "TAG", cands)
    for name, tags in found.items():
        assert tags == found["l1_check.py"], name   # the walk agrees with itself
    check(found["l1_check.py"], font, "the walk")


@pytest.mark.skipif(not shutil.which("node"), reason="needs node")
def test_batches_match_font(font, tmp_path):
    cands = sorted(set(font) | set(PROBES))
    batches = sorted(d for d in BATCHES.glob("std-batch-*") if (d / "tools" / "std_driver.py").is_file())
    if not batches:
        pytest.skip(f"no std batch under {BATCHES}")
    for d in batches:
        py = py_tags((d / "tools" / "std_driver.py").read_text(), '\nTAG = r"', cands)
        jsx = jsx_tags(tmp_path, (d / "scripts" / "std_card.jsx").read_text(), "STD_TAG", cands)
        assert py == jsx, d.name
        check(py, font, d.name, BATCH_LAG.get(d.name, set()))
