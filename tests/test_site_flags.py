"""
Card flags (Card Edits.json flags, TTSCardMaker site/assets/flags.js): the three field lists agree, the build carries them
and never counts them as pending, and the website's search, batch lines and file writing read them. Loads the site from
$AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without node or a flags.js (2026-11-07).
"""
import importlib.util
import json
import shutil
from os import environ
from os.path import expanduser
from pathlib import Path
from subprocess import run as run_process

import pytest

from commands.card_edits import FLAG_MAX, FLAGS, ORDER, CardEdits, dump_edits
from conftest import CLONE

SITE = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE))) / "site"
ASSETS = SITE / "assets"
pytestmark = pytest.mark.skipif(not ((ASSETS / "flags.js").is_file() and shutil.which("node")), reason=f"no flags.js in {ASSETS}, or no node")


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


def test_build_merges_automatic_flags(tmp_path):
    """The checks' notes beside a person's flag; a person's flag that went is left out, a newer one stays."""
    spec = importlib.util.spec_from_file_location("site_build_flags", SITE / "build" / "build_site.py")
    bs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bs)
    (tmp_path / "card_checks.json").write_text(json.dumps({"cards": {"sha": {"hand": "note", "bars": None}}, "cleared": {"A.psd": "00000000"}}))
    records, cleared = bs.load_checks(str(tmp_path))
    card = {"psd": "A.psd", "flags": {"bars": "", "psd": "00000000"}}
    bs.merge_flags(card, records["sha"], cleared)
    assert card == {"psd": "A.psd", "checks": {"hand": "note"}}
    card = {"psd": "A.psd", "flags": {"bars": "", "psd": "11111111"}}
    bs.merge_flags(card, None, cleared)
    assert card == {"psd": "A.psd", "flags": {"bars": "", "psd": "11111111"}}
    assert bs.load_checks(str(tmp_path / "none")) == ({}, {})


def test_search_lines_and_file(tmp_path):
    """flag: searches; the reason's rules; the batch's lines; a set flag sits in ORDER, the last one cleared unsets it; a
    form draft's tick and reason read back from the card."""
    out = node(tmp_path, r"""
const cards = [{ name: "A", flags: { hand: "why" } }, { name: "B", flags: { bars: "" } }, { name: "C", flags: { hand: "", bars: "x" } }, { name: "D" }, { name: "E", checks: { hand: "note" } }];
const hits = q => cards.filter(S.compile(q)).map(c => c.name).join("");
const set = { flags: F.withFlag({ hand: "why" }, "bars", "  two   words ") };
const file = E.editsFile(JSON.stringify({ cards: { X: { creatures: null, note: "n", _by: "a", _at: "b" } } }), [{ key: "X", set, unset: [] }], "T", "web");
const lines = op => E.describeItems(op, { flags: { hand: "why", bars: "" }, edited: ["flags"] }).map(x => x.text);
process.stdout.write(JSON.stringify({ hits: ["flag:hand", "flag:bars", "flag:any", "!flag:any", "FLAG:HAND & flag:bars"].map(hits),
  clean: [F.cleanFlags({ bars: "", hand: 1, other: "x" }), F.cleanFlags({}), F.withFlag({ hand: "" }, "hand", null), F.withFlag({ hand: "", psd: "0" }, "bars", "", "1a2b3c4d"), F.withFlag({ hand: "", psd: "0" }, "hand", null, "1a2b3c4d")],
  note: F.flagsHtml(cards[4]), own: [F.hasFlag(cards[4], "hand"), F.ownFlag(cards[4], "hand")], error: [F.flagError({ hand: "x".repeat(300) }), F.flagError({ bars: "x".repeat(301) }), F.flagError({ hand: "" }), F.flagError({ bars: "" })],
  file, unset: E.editsFile(file, [{ key: "X", set: {}, unset: ["flags"] }], "T", "web"), nulled: E.editsFile(file, [{ key: "X", set: { flags: null }, unset: [] }], "T", "web"),
  lines: [lines({ set: { flags: { hand: "new" } }, unset: [] }), lines({ set: {}, unset: ["flags"] })],
  raw: [D.rawOf(cards[1], "flag-bars"), D.rawOf(cards[0], "flag:hand"), D.rawOf(cards[3], "flag-hand"), D.rawOf(cards[3], "flag:bars")], names: D.fieldNames({ "flag:bars": "x", "flag-bars": true }) }));
""")
    assert out["hits"] == ["ACE", "BC", "ABCE", "D", "C"]
    assert out["clean"] == [{"bars": ""}, None, None, {"hand": "", "bars": "", "psd": "1a2b3c4d"}, None]
    assert '<span class="flag-note">note</span>' in out["note"] and "data-flag" not in out["note"] and out["own"] == [True, False]
    assert out["error"] == ["", "Special stat bars: at most 300 characters", "Needs doing by hand: say what the issue is", ""]
    assert out["file"] == dump_edits(json.loads(out["file"]))
    entry = json.loads(out["file"])["cards"]["X"]
    assert list(entry) == ["note", "creatures", "flags", "_by", "_at"] and entry["flags"] == {"hand": "why", "bars": "two words"}
    assert "flags" not in json.loads(out["unset"])["cards"]["X"] and "flags" not in json.loads(out["nulled"])["cards"]["X"]
    assert out["lines"] == [["Flags: Needs doing by hand: new; Special stat bars cleared"], ["Flags: Needs doing by hand cleared; Special stat bars cleared"]]
    assert out["raw"] == [True, "why", False, ""] and out["names"] == "Special stat bars"
