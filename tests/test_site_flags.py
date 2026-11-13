"""
Card flags (Card Edits.json flags, TTSCardMaker site/assets/flags.js): the bot's field lists are the build's and
flags.js's, and the build carries flags, never counting them as pending. Loads the site from $AC_SITE_SRC (a TTSCardMaker
checkout) or the clone; skips without a flags.js (2026-11-07).
"""
import importlib.util
import json
from subprocess import run as run_process

import pytest

from commands.card_edits import FLAG_MAX, FLAGS, ORDER, CardEdits
from conftest import SITE_SRC

SITE = SITE_SRC / "site"
ASSETS = SITE / "assets"
pytestmark = pytest.mark.skipif(not (ASSETS / "flags.js").is_file(), reason=f"no flags.js in {ASSETS}")


def node(tmp_path, body: str):
    """Run body with flags.js, search.js, edit.js and drafts.js imported as F, S, E and D; its stdout as JSON."""
    imports = "\n".join(f"const {n} = await import({json.dumps((ASSETS / f'{m}.js').as_uri())});"
                        for n, m in (("F", "flags"), ("S", "search"), ("E", "edit"), ("D", "drafts")))
    script = tmp_path / "probe.mjs"
    script.write_text(f"{imports}\n{body}\n", encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_build_carries_flags_never_pending(tmp_path, monkeypatch):
    """The build's lists are the bot's and flags.js's; parse_edits keeps a good entry and skips a bad one; a flagged card
    has its flags in cards.json, as edited, and nothing pending."""
    spec = importlib.util.spec_from_file_location("site_build_flags", SITE / "build" / "build_site.py")
    bs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bs)
    out = node(tmp_path, "process.stdout.write(JSON.stringify({ keys: F.CARD_FLAGS.map(([k]) => k), max: F.FLAG_MAX }));")
    assert tuple(out["keys"]) == bs.FLAGS == FLAGS and out["max"] == bs.FLAG_MAX == FLAG_MAX
    assert set(bs.EDIT_FIELDS) | {"parts"} == set(CardEdits.TYPES) == set(ORDER) == set(CardEdits.KIND)
    (tmp_path / "site").mkdir()
    (tmp_path / "site" / "Card Edits.json").write_text(json.dumps({"cards": {"Mini Doomer": {"flags": {"hand": "why", "bars": ""}},
                                                                             "Bad": {"flags": {"other": ""}}}}), encoding="utf-8")
    monkeypatch.setattr(bs, "REPO", str(tmp_path))
    edits = bs.parse_edits()
    card = {"key": "Mini_Doomer", "name": "Mini Doomer", "hp": 5, "problems": []}
    bs.apply_edits(card, edits, shown="Mini Doomer")
    assert card["flags"] == {"hand": "why", "bars": ""} and card["edited"] == ["flags"] and "pending" not in card
    assert edits["bad"][0] == {}
    assert bs.valid_edit("flags", {"hand": "", "psd": "1a2b3c4d"}) and CardEdits.valid("flags", {"hand": "", "psd": "1a2b3c4d"})
    assert not bs.valid_edit("flags", {"psd": "1a2b3c4d"}) and not bs.valid_edit("flags", {"hand": "", "psd": "x"})
