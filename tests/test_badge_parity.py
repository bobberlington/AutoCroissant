"""
Timing-tag badges: the stoogedings font's badge_tags(), the website's TAG/tagInfo and the migration walk's TAG must
hold the same tags (companion skill §1.2, "three places"). Reads the card repo clone; skips without it or node.
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


@pytest.mark.skipif(not WALK.is_dir(), reason=f"no migration walk at {WALK}")
def test_walk_matches_font(font, tmp_path):
    cands = sorted(set(font) | set(PROBES))
    src = (WALK / "l1_check.py").read_text()
    py = re.compile(src.split('TAG = re.compile(r"', 1)[1].split('")\n', 1)[0])
    assert {t for t in cands if py.fullmatch(f"({t})")} == set(font)
    if not shutil.which("node"):
        return
    for name in ("std.jsx", "std_l1.jsx"):
        line = next(ln for ln in (WALK / name).read_text().splitlines() if ln.startswith("var TAG = "))
        out = node(tmp_path, line + f"""
console.log(JSON.stringify({json.dumps(cands)}.filter(t => {{ const m = ("(" + t + ") rest").substr(0, 12).match(TAG);
  return m && m[0] === "(" + t + ")"; }})));""")
        assert set(json.loads(out)) == set(font), name
