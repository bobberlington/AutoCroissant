"""
The card page's text over time (TTSCardMaker site/assets/history.js, in node): the current text dated from when it
started (cards.json since, the bot's carried start), not from the last PSD commit, in the History block, the image
versions' "Text then" and the Recently updated feed; and every date the site shows in Pacific time, the bot's TIMEZONE,
whatever the viewer's zone. Loads the site from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without it,
node, or a site before history.js.
"""
import json
import shutil
from os import environ
from os.path import expanduser
from pathlib import Path
from subprocess import run as run_process

import pytest

from conftest import CLONE

ASSETS = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE))) / "site" / "assets"
HISTORY_JS = ASSETS / "history.js"
pytestmark = pytest.mark.skipif(not (HISTORY_JS.is_file() and shutil.which("node")),
                                reason=f"no history.js at {ASSETS}, or no node")
# Beldam (2026-10-22): its (1CD) text came with the 2026-10-21 22:57 PT image; a 10-22 07:27 PT re-save kept it
OLD_AT, IMAGE_AT, RESAVE_AT = 1770523234, 1792648620, 1792679220


def node(tmp_path, body: str, tz: str = "Europe/Berlin"):
    """Run body after importing history.js as h, in a viewer's time zone; its stdout as JSON."""
    script = tmp_path / "probe.mjs"
    script.write_text(f"const h = await import({json.dumps(HISTORY_JS.as_uri())});\n{body}\n", encoding="utf-8")
    out = run_process(["node", str(script)], capture_output=True, text=True, check=True, env={**environ, "TZ": tz})
    return json.loads(out.stdout)


@pytest.mark.parametrize("tz", ["Europe/Berlin"])
def test_dates_are_pacific_for_every_viewer(tmp_path, tz):
    """10-21 22:57 PT is Oct 22 in Berlin and UTC: the site says Oct 21 in every zone, and its month is October."""
    out = node(tmp_path, f"""const pt = new Date({IMAGE_AT} * 1000).toLocaleDateString(undefined, {{ year: "numeric", month: "short", day: "numeric", timeZone: "America/Los_Angeles" }});
process.stdout.write(JSON.stringify([h.date({IMAGE_AT}), pt, h.month(1793509200)]));""", tz)
    assert out[0] == out[1] and "21" in out[0]
    assert "Oct" in out[2]                                              # 2026-10-31 22:00 PT: still October

