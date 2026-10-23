"""
The website's Changes page helpers (TTSCardMaker site/assets/changes.js, in node): the Website edits tab's cap and the
tab row's focus rule. Skips without the card repo clone, node, or a changes.js that has them.
"""
import json
import shutil
from subprocess import run as run_process

import pytest

from conftest import CLONE

CHANGES_JS = CLONE / "site" / "assets" / "changes.js"
pytestmark = pytest.mark.skipif(not (CHANGES_JS.is_file() and shutil.which("node")
                                     and "keepsTabFocus" in CHANGES_JS.read_text(encoding="utf-8")),
                                reason=f"no changes.js with webEdits/keepsTabFocus at {CLONE}, or no node")


def node(tmp_path, body: str):
    """Run body after importing changes.js as ch; its stdout as JSON."""
    script = tmp_path / "probe.mjs"
    script.write_text(f"const ch = await import({json.dumps(CHANGES_JS.as_uri())});\n{body}\n", encoding="utf-8")
    return json.loads(run_process(["node", str(script)], capture_output=True, text=True, check=True).stdout)


def test_web_edits_newest_first_capped(tmp_path):
    """Editors' cards only (not the bot's, not a _bot field alone), newest first, at most WEB_EDITS_SHOWN; total: all."""
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
