"""
The website's editing helpers of round 3 (TTSCardMaker site/assets, in node and by text): the item type list (edit.js
ITEM_TYPES = the Items/ folders), drops on a filter-bar value or a pinned box (inline.js targetOf/dropOn, dragenter),
the click-away rule for the "+" editors, the name autofill (tokens.js suggestNames) and the pinned tray (tray.js).
Loads the site from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without node or a site before them.
"""
import json
import re
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


# ---- item 15: item type
@needs_node
def test_item_types_are_the_items_folders(tmp_path):
    items = SITE.parent / "Items"
    if not items.is_dir():
        pytest.skip(f"no {items}")
    got = node(tmp_path, "process.stdout.write(JSON.stringify(E.ITEM_TYPES));", E=EDIT_JS)
    assert got == sorted(p.name.lower() for p in items.iterdir() if p.is_dir())


def test_item_type_is_a_select():
    text = EDIT_JS.read_text(encoding="utf-8")
    assert '<select id="ef-subtype">' in text
    assert "dl-sub\"" not in text and 'id="ef-subtype" list=' not in text
