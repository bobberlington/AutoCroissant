"""
The cards grid's exclusions (TTSCardMaker site/assets/grid.js dropExcluded, flip, in node): x<key> params drop the cards
matching any of their values, xmissing keeps the cards that have it, and a click / right-click cycle the include and
exclude lists. Loads the site from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without node or a grid.js
with dropExcluded (2026-10-25).
"""
import json
import shutil
from os import environ
from os.path import expanduser
from pathlib import Path
from subprocess import run as run_process

import pytest

from conftest import CLONE

GRID_JS = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE))) / "site" / "assets" / "grid.js"
pytestmark = pytest.mark.skipif(not (GRID_JS.is_file() and shutil.which("node") and "dropExcluded" in GRID_JS.read_text(encoding="utf-8")),
                                reason=f"no grid.js with dropExcluded at {GRID_JS}, or no node")


def test_exclusions_and_clicks(tmp_path):
    script = tmp_path / "probe.mjs"
    script.write_text(f"""const G = await import({json.dumps(GRID_JS.as_uri())});
const cards = [
  {{ name: "Bob", type: "creature", series: "Kirby", subseries: "Dream", types: ["fire"], author: "Ann", stars: 3 }},
  {{ name: "Al", type: "creature", series: "", types: [], author: "", stars: 5, dup: "x" }},
  {{ name: "Cy", type: "minion", series: null, types: null, stars: 0 }},
  {{ name: "Di", type: "debuff", series: "Kirby", subseries: "Air", types: ["fire", "water"], author: "Ann" }},
  {{ name: "Ed", type: "item", series: "Mario", subtype: "heal", types: ["water"], author: "Bo", stars: 1 }},
];
const flags = {{ dup: c => !!c.dup }};
const names = q => G.dropExcluded(cards, new URLSearchParams(q), flags).map(c => c.name).join(",");
const flips = [[[], [], false], [["a"], [], false], [["a"], [], true], [[], ["a"], false], [[], ["a"], true], [[], [], true]]
  .map(([i, x, ex]) => G.flip(i, x, "a", ex));
process.stdout.write(JSON.stringify({{ out: Object.fromEntries(["", "xkind=creature", "xkind=status", "xstars=5,3", "xsub=heal", "xtypes=fire",
  "xtypes=fire,water", "xflags=dup", "xflags=nope", "xmissing=series", "xseries=Kirby", "xsubseries=Air", "xauthor=Ann",
  "xkind=creature&xtypes=water"].map(q => [q, names(q)])), flips, keys: G.X_KEYS }}));
""", encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout)
    assert out["out"] == {"": "Bob,Al,Cy,Di,Ed", "xkind=creature": "Cy,Di,Ed", "xkind=status": "Bob,Al,Cy,Ed",
                          "xstars=5,3": "Cy,Di,Ed", "xsub=heal": "Bob,Al,Cy,Di", "xtypes=fire": "Al,Cy,Ed",
                          "xtypes=fire,water": "Al,Cy", "xflags=dup": "Bob,Cy,Di,Ed", "xflags=nope": "Bob,Al,Cy,Di,Ed",
                          "xmissing=series": "Bob,Di,Ed", "xseries=Kirby": "Al,Cy,Ed", "xsubseries=Air": "Bob,Al,Cy,Ed",
                          "xauthor=Ann": "Al,Cy,Ed", "xkind=creature&xtypes=water": "Cy"}
    # click: neutral -> in -> neutral; right-click: in -> out; click or right-click on an excluded one: neutral
    assert out["flips"] == [[["a"], []], [[], []], [[], ["a"]], [[], []], [[], []], [[], ["a"]]]
    assert sorted(out["keys"]) == ["author", "flags", "kind", "missing", "series", "stars", "sub", "subseries", "types"]
