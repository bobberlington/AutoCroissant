"""
The website's Changes page helpers (TTSCardMaker site/assets/changes.js, in node): the Website edits tab's cap, the
tab row's focus rule, and where an archived card's PSD went. Loads changes.js from $AC_SITE_SRC (a TTSCardMaker
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


def test_an_archived_card_says_where_its_psd_went(tmp_path):
    """moved_to (White Palace into MDW/The Morgue): "moved to MDW/The Morgue", not "PSD deleted", in the Archived cards
    list and the archived event; renamed_to: "renamed to"; neither: "PSD deleted" (AutoCroissant changelog._line)."""
    rows = [{"card": "White Palace", "title": "White Palace", "card_type": "field", "path": "Field/4 Stars/White_Palace.psd",
             "moved_to": "MDW/The Morgue/White_Palace.psd"},
            {"card": "ZOEY!!!!!", "renamed_to": "Z!!!!!", "renamedId": "z"},
            {"card": "Gone", "path": "Items/Gone.psd"}]
    out = node(tmp_path, f"""const rows = {json.dumps(rows)};
const text = html => html.replace(/<[^>]+>/g, "").replace(/\\s+/g, " ");
process.stdout.write(JSON.stringify([...rows.map(r => text(ch.archivedHtml([r], "o/r"))),
  ...rows.map(r => text(ch.eventHtml({{ kind: "archived", ...r }})))]));""")
    palace, zoey, gone, palace_event, _, gone_event = out
    assert "moved to MDW/The Morgue" in palace and "PSD deleted" not in palace
    assert "renamed to Z!!!!!" in zoey and "PSD deleted" not in zoey
    assert "PSD deleted" in gone
    assert palace_event == "White Palace (moved to MDW/The Morgue)" and gone_event == "Gone (PSD deleted)"
