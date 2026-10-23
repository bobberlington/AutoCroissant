"""
The website's PSD-details dropdown (TTSCardMaker site/assets/details.js, in node): its Parser section names an editor's
fields as edited, never the bot's (its own entries, the migration's _bot fields), like the Edited badge. Loads details.js
from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without it, node, or a details.js before editorFields.
"""
import json
import shutil
from os import environ
from os.path import expanduser
from pathlib import Path
from subprocess import run as run_process

import pytest

from conftest import CLONE

DETAILS_JS = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE))) / "site" / "assets" / "details.js"
pytestmark = pytest.mark.skipif(not (DETAILS_JS.is_file() and shutil.which("node")
                                     and "editorFields" in DETAILS_JS.read_text(encoding="utf-8")),
                                reason=f"no details.js with editorFields at {DETAILS_JS}, or no node")
LABELS = {"name": "Name", "hp": "HP", "ability": "Ability"}


def parser_section(tmp_path, cards: list) -> list:
    """Each card's Parser section (bodyHtml's), as text."""
    script = tmp_path / "probe.mjs"
    script.write_text(f"""const d = await import({json.dumps(DETAILS_JS.as_uri())});
const cards = {json.dumps(cards)};
process.stdout.write(JSON.stringify(cards.map(c => {{
  const html = d.bodyHtml({{}}, c, {{ labels: {json.dumps(LABELS)} }});
  return html.slice(html.indexOf("Parser")).replace(/<[^>]+>/g, " ").replace(/\\s+/g, " ").trim();
}})));
""", encoding="utf-8")
    return json.loads(run_process(["node", str(script)], capture_output=True, text=True, check=True).stdout)


def test_parser_section_names_editors_fields_only(tmp_path):
    bot = {"edited": ["aliases", "series"], "orig": {"series": "omori"}, "series": "OMORI", "editBy": "AutoCroissant"}
    moved = {"edited": ["hp", "series"], "botEdited": ["series"], "orig": {"hp": 2, "series": "x"}, "hp": 3,
             "editBy": "MichaelJSr"}
    editor = {"edited": ["note"], "orig": {}, "editBy": "MichaelJSr"}
    out = parser_section(tmp_path, [bot, moved, editor, {}])
    assert "Also edited" not in out[0] and "Set by AutoCroissant: aliases, series" in out[0]
    assert "HP 2 3" in out[1] and "Also edited" not in out[1] and "Set by AutoCroissant: series" in out[1]
    assert "Also edited: note" in out[2] and "Set by AutoCroissant" not in out[2]
    assert "nothing edited" in out[3]
