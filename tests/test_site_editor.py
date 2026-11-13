"""
The website editor's helpers (TTSCardMaker site/assets, in node), one smoke test (owner 2026-11-12): drafts of unsaved
edits (drafts.js), the type palette (palette.js) loaded beside it. Loads them from $AC_SITE_SRC (a TTSCardMaker checkout)
or the clone; skips without them.
"""
import json
from subprocess import run as run_process

import pytest

from conftest import SITE_SRC

ASSETS = SITE_SRC / "site" / "assets"
MODULES = ("drafts", "palette")
pytestmark = pytest.mark.skipif(not all((ASSETS / f"{m}.js").is_file() for m in MODULES),
                                reason=f"no {', '.join(m + '.js' for m in MODULES)} in {ASSETS}")


def node(tmp_path, body: str):
    """Run body with the modules imported (drafts, palette); its stdout as JSON."""
    imports = "\n".join(f"const {m} = await import({json.dumps((ASSETS / f'{m}.js').as_uri())});" for m in MODULES)
    script = tmp_path / "probe.mjs"
    script.write_text(f"{imports}\n{body}\n", encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


# ---- drafts.js
STORE = """
const mem = new Map(), storage = { getItem: k => mem.get(k) ?? null, setItem: (k, v) => mem.set(k, String(v)), removeItem: k => mem.delete(k) };
let now = 1000; const store = drafts.draftStore(storage, drafts.DRAFT_KEY, () => now);
"""


def test_drafts_per_card_and_field_in_local_storage(tmp_path):
    """One entry per card id (the stage's key) with each field's typed text; an inline field joins the card's draft,
    the Edit dialog's changed inputs replace it; dropping a field, a card; a broken or odd entry reads as none."""
    out = node(tmp_path, STORE + """
const ids = () => ["goombella", "mini-doomer", "a", "b", "c", "d", "e"].filter(id => store.get(id));
store.put("goombella", { ability: "Burns." }); now = 2000; store.put("goombella", { hp: "7" });
const joined = store.get("goombella");
store.put("goombella", { name: "G" }, true); const whole = store.get("goombella").fields;
store.put("mini-doomer", { note: "x" }); store.drop("goombella", ["name"]); const afterDrop = ids();
store.put("a", { name: "A" }); store.put("b", { name: "B" }); store.drop("a"); store.drop("mini-doomer"); const left = ids();
store.put("b", {}, true); const emptied = ids();
mem.set(drafts.DRAFT_KEY, "{not json"); const broken = ids();
mem.set(drafts.DRAFT_KEY, JSON.stringify({ c: { at: 5, fields: { name: "ok", hp: 7, x: { y: 1 }, types: ["fire"] } }, d: { fields: [] }, e: 3 }));
const odd = store.get("c"), oddIds = ids();
process.stdout.write(JSON.stringify({ joined, whole, afterDrop, left, emptied, broken, odd, oddIds, key: drafts.DRAFT_KEY, stored: [...mem.keys()] }));
""")
    assert out["key"] == "edit-drafts"
    assert out["joined"] == {"at": 2000, "fields": {"ability": "Burns.", "hp": "7"}, "base": {}}
    assert out["whole"] == {"name": "G"}
    assert out["afterDrop"] == ["mini-doomer"] and out["left"] == ["b"] and out["emptied"] == []
    assert out["broken"] == []
    assert out["odd"] == {"at": 5, "fields": {"name": "ok", "types": ["fire"]}, "base": {}} and out["oddIds"] == ["c"]
