"""
The website's pinned tray (TTSCardMaker site/assets/tray.js, in node), one smoke test (owner 2026-11-12): adding and
removing pins. Loads the site from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone.
"""
import json
from pathlib import Path
from subprocess import run as run_process

from conftest import SITE_SRC

SITE = SITE_SRC / "site"
ASSETS = SITE / "assets"


def node(tmp_path: Path, body: str, **modules: Path):
    """Run body with each module imported under its name; its stdout as JSON."""
    script = tmp_path / "probe.mjs"
    imports = "".join(f"const {k} = await import({json.dumps(v.as_uri())});\n" for k, v in modules.items())
    script.write_text(f"{imports}{body}\n", encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


# ---- item 17: the pinned tray
TRAY_JS = ASSETS / "tray.js"


def test_add_and_remove_pins(tmp_path):
    out = node(tmp_path, """
const one = [{ k: "series", v: "Mario" }], full = Array.from({ length: 100 }, (_, i) => ({ k: "author", v: `A${i}` }));
process.stdout.write(JSON.stringify([T.addPin(one, { k: "series", v: "mario " }), T.addPin(one, { k: "author", v: "Mario" }),
  T.addPin(full, { k: "series", v: "Kirby" }), T.addPin(full.slice(0, 99), { k: "series", v: "Kirby" }).pins.length,
  T.removePin(full, 0).length, T.removePin(one, 5)]));""", T=TRAY_JS)
    assert out == [{"error": "Already pinned"}, {"pins": [{"k": "series", "v": "Mario"}, {"k": "author", "v": "Mario"}]},
                   {"error": "At most 100 pins"}, 100, 99, [{"k": "series", "v": "Mario"}]]
