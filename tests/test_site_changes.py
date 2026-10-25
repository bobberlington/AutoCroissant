"""
The website's Changes page helpers (TTSCardMaker site/assets/changes.js, in node): the Website edits tab's cap, the
tab row's focus rule, and the Archive tab's tags. Loads changes.js from $AC_SITE_SRC (a TTSCardMaker
checkout) or the clone; skips without it, node, or a changes.js that has them.
"""
import json
import shutil
from os import environ
from os.path import expanduser
from pathlib import Path
from subprocess import run as run_process

import pytest

from conftest import CLONE

CHANGES_JS = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE))) / "site" / "assets" / "changes.js"
pytestmark = pytest.mark.skipif(not (CHANGES_JS.is_file() and shutil.which("node")
                                     and "keepsTabFocus" in CHANGES_JS.read_text(encoding="utf-8")),
                                reason=f"no changes.js with webEdits/keepsTabFocus at {CHANGES_JS}, or no node")
archive_tags = pytest.mark.skipif(not (CHANGES_JS.is_file() and "archiveTagHtml" in CHANGES_JS.read_text(encoding="utf-8")),
                                  reason="no archive tags in changes.js")


def node(tmp_path, body: str):
    """Run body after importing changes.js as ch; its stdout as JSON."""
    script = tmp_path / "probe.mjs"
    script.write_text(f"const ch = await import({json.dumps(CHANGES_JS.as_uri())});\n{body}\n", encoding="utf-8")
    return json.loads(run_process(["node", str(script)], capture_output=True, text=True, check=True).stdout)


def test_web_edits_newest_first_capped(tmp_path):
    """Editors' cards only (not the bot's, not a _bot field alone), newest first, at most WEB_EDITS_SHOWN; total: all.
    Rows are {c, by, at, fields} (fieldEditors)."""
    out = node(tmp_path, """
const cards = Array.from({ length: ch.WEB_EDITS_SHOWN + 5 }, (_, i) =>
  ({ id: `c${i}`, edited: ["hp"], editBy: "MichaelJSr", editAt: `2026-10-${String(1 + (i % 28)).padStart(2, "0")}T00:00:${String(i % 60).padStart(2, "0")}Z` }));
cards.push({ id: "bot", edited: ["series"], editBy: "AutoCroissant", editAt: "2030-01-01" },
           { id: "moved", edited: ["series"], botEdited: ["series"], editBy: "MichaelJSr", editAt: "2030-01-01" });
const { shown, total } = ch.webEdits(cards);
process.stdout.write(JSON.stringify({ n: ch.WEB_EDITS_SHOWN, total, shown: shown.map(r => r.at), ids: shown.map(r => r.c.id) }));
""")
    assert out["total"] == out["n"] + 5 and len(out["shown"]) == out["n"]
    assert out["shown"] == sorted(out["shown"], reverse=True) and not {"bot", "moved"} & set(out["ids"])


def test_tab_keeps_focus_from_the_keyboard_only(tmp_path):
    """A tab focused from the keyboard (:focus-visible) keeps it; a mouse click's focus or an element outside the row
    doesn't; a browser without :focus-visible keeps it as before."""
    out = node(tmp_path, """
const el = (inRow, visible) => ({ closest: s => (inRow && s === ".change-tabs" ? {} : null),
  matches: s => { if (visible === null) throw new SyntaxError(s); return s === ":focus-visible" && visible; } });
process.stdout.write(JSON.stringify([ch.keepsTabFocus(el(true, true)), ch.keepsTabFocus(el(true, false)),
  ch.keepsTabFocus(el(false, true)), ch.keepsTabFocus(null), ch.keepsTabFocus(el(true, null))]));
""")
    assert out == [True, False, False, False, True]


@archive_tags
def test_an_archived_card_has_its_tag(tmp_path):
    """The Archive tab and the archived event show the tag: moved_to (White Palace into MDW/The Morgue) Retired, its
    paths in the tip; renamed_to Renamed and "renamed to"; neither Deleted; the build's tag wins over the fields."""
    rows = [{"card": "White Palace", "title": "White Palace", "card_type": "field", "path": "Field/4 Stars/White_Palace.psd",
             "moved_to": "MDW/The Morgue/White_Palace.psd"},
            {"card": "ZOEY!!!!!", "renamed_to": "Z!!!!!", "renamedId": "z"},
            {"card": "Gone", "path": "Items/Gone.psd"},
            {"card": "Tagged", "tag": "deleted", "moved_to": "MDW/x.psd"}]
    out = node(tmp_path, f"""const rows = {json.dumps(rows)};
const text = html => html.replace(/<[^>]+>/g, "").replace(/\\s+/g, " ").trim();
process.stdout.write(JSON.stringify([...rows.map(r => text(ch.archiveHtml([r], "o/r"))),
  ...rows.map(r => text(ch.eventHtml({{ kind: "archived", ...r }}))), ch.archiveTagHtml(rows[0])]));""")
    palace, zoey, gone, tagged, palace_event, _, gone_event, tagged_event, pill = out
    assert "White Palace Retired" in palace and "PSD deleted" not in palace
    assert "ZOEY!!!!! Renamed renamed to Z!!!!!" in zoey and "Gone Deleted" in gone and "Tagged Deleted" in tagged
    assert (palace_event, gone_event, tagged_event) == ("White Palace Retired", "Gone Deleted", "Tagged Deleted")
    assert 'class="fact arch retired"' in pill and 'data-tip="Field/4 Stars/White_Palace.psd → MDW/The Morgue"' in pill


@archive_tags
def test_archive_tag_chips_filter(tmp_path):
    """?tag=retired lists only the retired cards; the chips count every tag; the Changelog tab has no archive panel."""
    rows = [{"card": "A", "moved_to": "MDW/A.psd"}, {"card": "B", "renamed_to": "C"}, {"card": "D"}, {"card": "E", "moved_to": "MDW/E.psd"},
            {"card": "F", "moved_to": "MDW/Unfinished/F.psd"}]
    out = node(tmp_path, f"""const rows = {json.dumps(rows)};
const html = ch.archiveHtml(rows, "o/r", new URLSearchParams("tag=retired"));
const chips = [...html.matchAll(/data-value="([^"]+)">([^<]+)<span class="n">(\\d+)/g)].map(m => [m[1], m[2], +m[3]]);
const names = [...html.matchAll(/<li><b>([^<]+)<\\/b>/g)].map(m => m[1]);
const log = ch.changelogHtml({{ available: true, days: 180, events: [], archived: rows, repo: "o/r" }}, new URLSearchParams());
process.stdout.write(JSON.stringify({{ chips, names, on: /class="chip on"[^>]*data-value="retired"/.test(html), log: log.includes("<b>A</b>") }}));""")
    assert out["chips"] == [["renamed", "Renamed", 1], ["retired", "Retired", 2], ["unfinished", "Unfinished", 1], ["deleted", "Deleted", 1]]
    assert out["names"] == ["A", "E"] and out["on"] and not out["log"]
