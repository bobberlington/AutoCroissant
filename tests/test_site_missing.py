"""
The cards grid's Missing filter and sorts (TTSCardMaker site/assets/grid.js MISSING, byKey, in node): a value absent, null,
"" or [] is missing; a sort puts the cards without its value last either way, in name order. Loads the site from
$AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without node or a grid.js with MISSING (2026-10-25).
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
pytestmark = pytest.mark.skipif(not (GRID_JS.is_file() and shutil.which("node")), reason=f"no {GRID_JS}, or no node")


def test_missing_options_and_sorts_last(tmp_path):
    script = tmp_path / "probe.mjs"
    script.write_text(f"""const G = await import({json.dumps(GRID_JS.as_uri())});
const cards = [
  {{ name: "Bob", type: "creature", series: "Kirby", subseries: "Dream", types: ["fire"], ability: "Hit.", author: "Ann", stars: 3, hp: 5, def: 1, atk: 2, spd: 3, ratings: [{{ value: 5 }}], art: "https://x.com/1", ability_art: "Ann" }},
  {{ name: "Al", type: "creature", series: "", types: [], ability: " ", author: "", hp: 4, def: 1, atk: 2, art: " ", ability_art: null }},
  {{ name: "Cy", type: "minion", series: null, subseries: null, types: null, ability: null, stars: 0, hp: 9, def: 1, atk: 2, spd: 3, ratings: [] }},
  {{ name: "Di", type: "item", series: "Kirby", types: ["support"], ability: "Heal.", author: "Ann", stars: 2, art: "Kirby Super Star" }},
  {{ name: "Ed", type: "item", series: "Kirby", subseries: "Air", subtype: "heal", ability: "Heal.", author: "Ann", stars: 1, hp: 3 }},
];
const missing = Object.fromEntries(G.MISSING.map(([k, , , test]) => [k, cards.filter(test).map(c => c.name)]));
const order = (key, desc) => cards.slice().sort(G.byKey(key, desc)).map(c => c.name);
const low = v => v?.toLowerCase() || null, series = [c => low(c.series), c => low(c.subseries)];
const x = G.dropExcluded(cards, new URLSearchParams("xmissing=art")).map(c => c.name);
process.stdout.write(JSON.stringify({{ missing, x, hp: [order(c => c.hp ?? null, false), order(c => c.hp ?? null, true)],
  series: [order(series, false), order(series, true)] }}));
""", encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout)
    assert out["missing"] == {"series": ["Al", "Cy"], "subseries": ["Al", "Cy", "Di"], "types": ["Al", "Cy", "Ed"],
                              "ability": ["Al", "Cy"], "author": ["Al", "Cy"], "stars": ["Al", "Cy"], "stats": ["Al"],
                              "itemtype": ["Di"], "ratings": ["Al", "Cy", "Di", "Ed"], "art": ["Al", "Cy", "Ed"]}
    assert out["x"] == ["Bob", "Di"]                                      # the anti-filter: those with a source
    assert out["hp"] == [["Ed", "Al", "Bob", "Cy", "Di"], ["Cy", "Bob", "Al", "Ed", "Di"]]
    # by series, then subseries: no series last, and in a series no subseries last, either way
    assert out["series"] == [["Ed", "Bob", "Di", "Al", "Cy"], ["Bob", "Ed", "Di", "Al", "Cy"]]
