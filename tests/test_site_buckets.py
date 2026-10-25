"""
The website's editing helpers of round 3 (TTSCardMaker site/assets, in node and by text): the item type list (edit.js
ITEM_TYPES = the Items/ folders), drops on a filter-bar value or a pinned box (inline.js targetOf/dropOn, dragenter),
the click-away rule for the "+" editors, the name autofill (tokens.js suggestNames), the pinned tray (tray.js) and
Rename everywhere (edit.js renameAll: the ops it queues).
Loads the site from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without node or a site before them.
"""
import json
import shutil
from os import environ
from os.path import expanduser
from pathlib import Path
from subprocess import run as run_process

import pytest

from conftest import CLONE

SITE = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE))) / "site"
ASSETS = SITE / "assets"
EDIT_JS = ASSETS / "edit.js"
INLINE_JS = ASSETS / "inline.js"
pytestmark = pytest.mark.skipif(not EDIT_JS.is_file() or "ITEM_TYPES" not in EDIT_JS.read_text(encoding="utf-8"),
                                reason=f"no ITEM_TYPES in {EDIT_JS} (land TTSCardMaker work/r3-editing)")
needs_node = pytest.mark.skipif(not shutil.which("node"), reason="needs node")


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
needs_tray = pytest.mark.skipif(not TRAY_JS.is_file(), reason=f"no {TRAY_JS}")


@needs_node
@needs_tray
def test_add_and_remove_pins(tmp_path):
    out = node(tmp_path, """
const one = [{ k: "series", v: "Mario" }], nine = Array.from({ length: 9 }, (_, i) => ({ k: "author", v: `A${i}` }));
process.stdout.write(JSON.stringify([T.addPin(one, { k: "series", v: "mario " }), T.addPin(one, { k: "author", v: "Mario" }),
  T.addPin(nine, { k: "series", v: "Kirby" }), T.removePin(nine, 0).length, T.removePin(one, 5)]));""", T=TRAY_JS)
    assert out == [{"error": "Already pinned"}, {"pins": [{"k": "series", "v": "Mario"}, {"k": "author", "v": "Mario"}]},
                   {"error": "At most 9 pins"}, 8, [{"k": "series", "v": "Mario"}]]


