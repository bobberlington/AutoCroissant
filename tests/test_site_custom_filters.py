"""
My filters (TTSCardMaker site/assets/myfilters.js, in node): saved searches in localStorage (junk dropped, one search saved
once in any case, Undo puts it back where it was); cf / xcf params, one per search, so any search text round-trips the
address; every cf keeps its matches and every xcf drops them, in the search box's syntax. Loads the site from
$AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without node or myfilters.js (2026-10-25).
"""
import json
import shutil
from os import environ
from os.path import expanduser
from pathlib import Path
from subprocess import run as run_process

import pytest

from conftest import CLONE

CF_JS = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE))) / "site" / "assets" / "myfilters.js"
pytestmark = pytest.mark.skipif(not (CF_JS.is_file() and shutil.which("node")), reason=f"no {CF_JS}, or no node")


def test_saved_searches_and_params(tmp_path):
    script = tmp_path / "probe.mjs"
    script.write_text(f"""const store = new Map();
Object.defineProperty(globalThis, "localStorage", {{ configurable: true, value: {{ getItem: k => store.get(k) ?? null, setItem: (k, v) => store.set(k, String(v)) }} }});
const CF = await import({json.dumps(CF_JS.as_uri())});
const cards = [
  {{ name: "Bob", ability: "Heal 2. (1GCD)", type: "creature", hp: 5 }},
  {{ name: "Al", ability: "Deal 3, then draw", type: "item" }},
  {{ name: "Cy", ability: "(1gcd) Strike", type: "creature", hp: 9 }},
  {{ name: "Di", ability: "heal & mend", type: "minion", hp: 2 }},
];
const names = pairs => CF.apply(cards, new URLSearchParams(pairs)).map(c => c.name).join(",");
const cases = [[], [["cf", "(1GCD)"]], [["cf", "(1GCD)"], ["cf", "heal"]], [["xcf", "heal"]], [["cf", "(1GCD)"], ["xcf", "heal"]],
  [["cf", "type==creature & hp>4"]], [["xcf", "deal 3, then"]], [["cf", "strike | mend"]], [["cf", " "]]];
// as app.js setParams writes them, as parseHash reads them back
const tricky = ["(1GCD)", "deal 3, then", "heal & mend", "a=b?c#d", "100% +1", '"draw 2"', "type==creature & hp>4 | heal", "ü ★"];
const p = new URLSearchParams("kind=creature");
for (const t of tricky) p.append("cf", t);
p.append("xcf", "x,y");
const hash = `#/cards?${{p}}`, back = new URLSearchParams(hash.replace(/^#\\/?/, "").split("?")[1]);
const saves = [CF.add(" (1GCD) "), CF.add("heal"), CF.add("(1gcd)"), CF.add(""), CF.saved()];
const at = CF.remove("(1GCD)"), gone = CF.saved();
CF.add("(1GCD)", at);
process.stdout.write(JSON.stringify({{ apply: cases.map(names), back: CF.vals(back, "cf"), xcf: CF.vals(back, "xcf"), split: hash.split("?").length,
  kind: back.get("kind"), saves, at, gone, restored: CF.saved(), stored: store.get(CF.KEY),
  junk: [CF.readSaved("nope"), CF.readSaved('{{"a": 1}}'), CF.readSaved('[" a ", "a", 3, "", null, "b"]')],
  vals: CF.vals(new URLSearchParams("cf=a&cf=%20a%20&cf=&cf=b"), "cf") }}));
""", encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout)
    assert out["apply"] == ["Bob,Al,Cy,Di", "Bob,Cy", "Bob", "Al,Cy", "Cy", "Bob,Cy", "Bob,Cy,Di", "Cy,Di", "Bob,Al,Cy,Di"]
    # any text survives the address: one param each, no raw "?" for parseHash to split on, commas kept
    assert out["back"] == ["(1GCD)", "deal 3, then", "heal & mend", "a=b?c#d", "100% +1", '"draw 2"',
                           "type==creature & hp>4 | heal", "ü ★"]
    assert out["xcf"] == ["x,y"] and out["split"] == 2 and out["kind"] == "creature"
    # saved trimmed; the same search in another case is not saved twice (its saved spelling comes back); Undo restores the place
    assert out["saves"] == ["", "", "(1GCD)", "", ["(1GCD)", "heal"]]
    assert out["at"] == 0 and out["gone"] == ["heal"] and out["restored"] == ["(1GCD)", "heal"]
    assert json.loads(out["stored"]) == ["(1GCD)", "heal"]
    assert out["junk"] == [[], [], ["a", "b"]]
    assert out["vals"] == ["a", "b"]
