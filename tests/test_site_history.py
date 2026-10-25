"""
The card page's text over time (TTSCardMaker site/assets/history.js, in node): the current text dated from when it
started (cards.json since, the bot's carried start), not from the last PSD commit, in the History block, the image
versions' "Text then" and the Recently updated feed; and every date the site shows in Pacific time, the bot's TIMEZONE,
whatever the viewer's zone. Loads the site from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without it,
node, or a site before history.js.
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

ASSETS = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE))) / "site" / "assets"
HISTORY_JS = ASSETS / "history.js"
pytestmark = pytest.mark.skipif(not (HISTORY_JS.is_file() and shutil.which("node")),
                                reason=f"no history.js at {ASSETS}, or no node")
# Beldam (2026-10-22): its (1CD) text came with the 2026-10-21 22:57 PT image; a 10-22 07:27 PT re-save kept it
OLD_AT, IMAGE_AT, RESAVE_AT = 1770523234, 1792648620, 1792679220
BELDAM = {"id": "beldam", "hv": 1, "history": [{"ts": OLD_AT, "ability": "(1PT) Inflict any creature."}],
          "ability": "(1CD) Inflict any creature.", "ts": RESAVE_AT, "since": IMAGE_AT}


def node(tmp_path, body: str, tz: str = "Europe/Berlin"):
    """Run body after importing history.js as h, in a viewer's time zone; its stdout as JSON."""
    script = tmp_path / "probe.mjs"
    script.write_text(f"const h = await import({json.dumps(HISTORY_JS.as_uri())});\n{body}\n", encoding="utf-8")
    out = run_process(["node", str(script)], capture_output=True, text=True, check=True, env={**environ, "TZ": tz})
    return json.loads(out.stdout)


def test_the_image_that_brought_the_text_shows_no_older_text(tmp_path):
    """Image versions: the text at the 10-21 22:57 image is the current one (no "Text then"); before it, the older one."""
    out = node(tmp_path, f"""const c = {json.dumps(BELDAM)};
process.stdout.write(JSON.stringify([h.textAt(c, {IMAGE_AT}), h.textAt(c, {IMAGE_AT} - 86400), h.textAt(c, {OLD_AT} - 86400),
  h.textAt({{ ...c, since: undefined }}, {IMAGE_AT})]));""")
    assert out == [BELDAM["ability"], BELDAM["history"][0]["ability"], None, BELDAM["history"][0]["ability"]]


def test_history_dates_the_current_text_from_its_start(tmp_path):
    """The History block's Current is the day the text started (Oct 21, Pacific), not the re-save's (Oct 22)."""
    out = node(tmp_path, f"""const c = {json.dumps(BELDAM)};
process.stdout.write(JSON.stringify([h.historyHtml(c), h.date({IMAGE_AT}), h.date({RESAVE_AT})]));""")
    html, started, resaved = out
    assert f"Current · {started}" in html and resaved not in html and started != resaved


def test_the_feed_highlights_only_a_text_its_update_brought(tmp_path):
    """Recently updated: a re-save that kept the text (since before ts) shows no diff; a text change (no since) does."""
    out = node(tmp_path, f"""const c = {json.dumps(BELDAM)};
process.stdout.write(JSON.stringify([h.textChanged(c), h.textChanged({{ ...c, since: undefined }}), h.textChanged({{ ...c, since: c.ts }})]));""")
    assert out == [False, True, True]


@pytest.mark.parametrize("tz", ["Europe/Berlin", "UTC", "America/New_York", "Asia/Tokyo"])
def test_dates_are_pacific_for_every_viewer(tmp_path, tz):
    """10-21 22:57 PT is Oct 22 in Berlin and UTC: the site says Oct 21 in every zone, and its month is October."""
    out = node(tmp_path, f"""const pt = new Date({IMAGE_AT} * 1000).toLocaleDateString(undefined, {{ year: "numeric", month: "short", day: "numeric", timeZone: "America/Los_Angeles" }});
process.stdout.write(JSON.stringify([h.date({IMAGE_AT}), pt, h.month(1793509200)]));""", tz)
    assert out[0] == out[1] and "21" in out[0]
    assert "Oct" in out[2]                                              # 2026-10-31 22:00 PT: still October


def test_no_date_in_the_viewers_zone():
    """Every date the site formats names its zone (Pacific, or UTC for a plain YYYY-MM-DD or a month key); no local-time
    getters. deck.js's TTS save stamp (new Date() with no time) is the saver's own clock and stays local."""
    found = []
    for js in sorted(ASSETS.glob("*.js")):
        text = js.read_text(encoding="utf-8")
        for m in re.finditer(r"new Date\(([^()]|\([^()]*\))+\)\.toLocale(?:Date|Time)?String\(([^)]*)\)", text):
            if "timeZone" not in m.group(2):
                found.append(f"{js.name}: {m.group(0)[:80]}")
        found += [f"{js.name}: {m.group(0)}" for m in re.finditer(r"\.get(?:FullYear|Month|Date|Day|Hours|Minutes)\(\)", text)]
    assert found == []
