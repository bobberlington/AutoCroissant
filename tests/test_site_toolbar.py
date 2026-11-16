"""
The Cards toolbar's Grid/Table and S/M/L toggles (TTSCardMaker site/assets/grid.js toolbarHtml, in node): each button
says whether it is on (aria-pressed, the same state as its .on colour) inside a named group. Loads the site from
$AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without a grid.js with toolbarHtml (2026-11-16).
"""
import json
import re
from subprocess import run as run_process

import pytest

from conftest import SITE_SRC

GRID_JS = SITE_SRC / "site" / "assets" / "grid.js"
pytestmark = pytest.mark.skipif(not GRID_JS.is_file() or "toolbarHtml" not in GRID_JS.read_text(encoding="utf-8"),
                                reason=f"no toolbarHtml in {GRID_JS}")


def test_toggles_say_which_is_on(tmp_path):
    script = tmp_path / "probe.mjs"
    script.write_text(f"""const G = await import({json.dumps(GRID_JS.as_uri())});
process.stdout.write(JSON.stringify([["grid", "s"], ["grid", "l"], ["table", "m"]].map(([view, size]) =>
  G.toolbarHtml({{ count: 3, sort: "name", view, size }}))));""", encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    states = []
    for html in json.loads(done.stdout):
        buttons = re.findall(r'<button type="button" class="(on|)" data-action="set-pref"[^>]*aria-pressed="(\w+)">(\w+)<', html)
        assert all((cls == "on") == (pressed == "true") for cls, pressed, _ in buttons), buttons
        groups = html.count('role="group"')
        states.append("".join(label for cls, _, label in buttons if cls) + f" {groups}")
    assert states == ["GridS 2", "GridL 2", "Table 1"]          # the table has no tile sizes
