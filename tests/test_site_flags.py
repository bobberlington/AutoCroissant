"""
Card flags (Card Edits.json flags, TTSCardMaker site/assets/flags.js): the three field lists agree, the build carries them
and never counts them as pending, and the website's search, batch lines and file writing read them. Loads the site from
$AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without node or a flags.js (2026-11-07).
"""
import importlib.util
import json
import shutil
from subprocess import run as run_process

import pytest

from commands.card_edits import FLAG_MAX, FLAGS, ORDER, CardEdits, dump_edits
from conftest import SITE_SRC

SITE = SITE_SRC / "site"
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
    """The checks' notes beside a person's flag; a person's flag that went is left out, a newer one stays; a PSD not checked
    yet takes its card's last record."""
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
    card = {"psd": "B.psd"}
    bs.merge_flags(card, {"hand": None, "bars": "B.", "name": "N."}, cleared, True)       # a second picture: no name flag
    assert card == {"psd": "B.psd", "checks": {"bars": "B."}}
    assert bs.load_checks(str(tmp_path / "none")) == ({}, {})
    checks = {"old": {"card": "A B", "hand": "note"}, "c1": {"card": "C"}, "c2": {"card": "C"}}
    at = (("new", "X/A_B.psd"), ("new", "X/A_C.psd"), ("new", "X/C.psd"), ("c1", "X/C.psd"))  # pushed, renamed, a shared name, its own
    assert [bs.check_of(checks, *a) for a in at] == [checks["old"], None, None, checks["c1"]]


def test_search_lines_and_file(tmp_path):
    """flag: searches; the reason's rules; the batch's lines; a set flag sits in ORDER, the last one cleared unsets it; a
    form draft's tick and reason read back from the card; flags as labels, each flag's issue behind its own info mark,
    its note and a person's reason (owner 11-11 evening); names_ok holds while the names are as marked."""
    out = node(tmp_path, r"""
const cards = [{ name: "A", flags: { hand: "why" } }, { name: "B", flags: { bars: "" } }, { name: "C", flags: { hand: "", bars: "x" } }, { name: "D" }, { name: "E", checks: { hand: "note" } }, { name: "F", checks: { name: "note" } }];
const hits = q => cards.filter(S.compile(q)).map(c => c.name).join("");
const set = { flags: F.withFlag({ hand: "why" }, "bars", "  two   words "), names_ok: ["Wall", "The Wall By Ech", "Blockade"] };
const wall = { name: "Wall", edited: ["name"], orig: { name: "The Wall By Ech" }, psd: "X/Blockade.psd", names_ok: set.names_ok };
const file = E.editsFile(JSON.stringify({ cards: { X: { creatures: null, note: "n", _by: "a", _at: "b" } } }), [{ key: "X", set, unset: [] }], "T", "web");
const lines = op => E.describeItems(op, { flags: { hand: "why", bars: "" }, edited: ["flags"] }).map(x => x.text);
process.stdout.write(JSON.stringify({ hits: ["flag:hand", "flag:bars", "flag:any", "!flag:any", "FLAG:HAND & flag:bars", "flag:name"].map(hits),
  clean: [F.cleanFlags({ bars: "", hand: 1, other: "x" }), F.cleanFlags({}), F.withFlag({ hand: "" }, "hand", null), F.withFlag({ hand: "", psd: "0" }, "bars", "", "1a2b3c4d"), F.withFlag({ hand: "", psd: "0" }, "hand", null, "1a2b3c4d")],
  note: [F.flagsHtml({ ...cards[4], id: "e" }), F.flagsHtml({ id: "g", checks: { hand: "Bars <1>" }, flags: { hand: `Hive's "bars"` } }), F.flagsHtml(cards[1]), F.flagsHtml({ ...cards[2], id: "c" }), F.flagsHtml({ id: "h", checks: { hand: "H", bars: "B" } })], own: [F.hasFlag(cards[4], "hand"), F.ownFlag(cards[4], "hand")], error: [F.flagError({ hand: "x".repeat(300) }), F.flagError({ bars: "x".repeat(301) }), F.flagError({ hand: "" }), F.flagError({ bars: "" })],
  file, unset: E.editsFile(file, [{ key: "X", set: {}, unset: ["flags"] }], "T", "web"), nulled: E.editsFile(file, [{ key: "X", set: { flags: null }, unset: [] }], "T", "web"),
  lines: [lines({ set: { flags: { hand: "new" } }, unset: [] }), lines({ set: {}, unset: ["flags"] })],
  raw: [D.rawOf(cards[1], "flag-bars"), D.rawOf(cards[0], "flag:hand"), D.rawOf(cards[3], "flag-hand"), D.rawOf(cards[3], "flag:bars")], names: D.fieldNames({ "flag:bars": "x", "flag-bars": true }),
  ok: [F.namesOk(wall), F.namesOk({ ...wall, name: "Walls" }), F.namesOk({ ...wall, psd: "X/Wall.psd" }), D.rawOf(wall, "names_ok"), F.namesNow(wall, wall, "W")] }));
""")
    assert out["hits"] == ["ACE", "BC", "ABCEF", "D", "C", "F"]
    assert out["clean"] == [{"bars": ""}, None, None, {"hand": "", "bars": "", "psd": "1a2b3c4d"}, None]
    assert out["note"][0] == ('<div class="card-flags"><div class="card-flag"><b>Needs doing by hand</b><button type="button" class="flag-i" '
                              'data-action="flag-issue" data-tip="Automatic check: note" aria-label="Needs doing by hand: what the issue is" '
                              'aria-expanded="false" aria-controls="flag-issue-e-hand" aria-describedby="flag-issue-e-hand">i</button></div></div>'
                              '<div class="flag-issue" id="flag-issue-e-hand" hidden><p>Automatic check: note</p></div>') and out["own"] == [True, False]
    both = ["Automatic check: Bars &lt;1&gt;", "Reason given: Hive&#39;s &quot;bars&quot;"]           # both, the note first
    assert f'data-tip="{" · ".join(both)}"' in out["note"][1] and "".join(f"<p>{t}</p>" for t in both) in out["note"][1]
    assert out["note"][2] == '<div class="card-flags"><div class="card-flag"><b>Special stat bars</b></div></div>'
    c = out["note"][3]                                                   # hand with no reason: no mark; bars' reason: one
    assert c.count('class="flag-i"') == 1 and 'aria-controls="flag-issue-c-bars"' in c and "Needs doing by hand</b></div>" in c
    assert c.endswith('<div class="flag-issue" id="flag-issue-c-bars" hidden><p>Reason given: x</p></div>')
    h = out["note"][4]                                                   # both noted: two marks, then two blocks, in order
    at = [h.index(x) for x in ('data-tip="Automatic check: H"', 'data-tip="Automatic check: B"', 'id="flag-issue-h-hand"', 'id="flag-issue-h-bars"')]
    assert at == sorted(at) and h.count('class="flag-i"') == 2 and h.count('class="flag-issue"') == 2
    assert out["error"] == ["", "Special stat bars: at most 300 characters", "Needs doing by hand: say what the issue is", ""]
    assert out["file"] == dump_edits(json.loads(out["file"]))
    entry = json.loads(out["file"])["cards"]["X"]
    assert list(entry) == ["note", "creatures", "flags", "names_ok", "_by", "_at"] and entry["flags"] == {"hand": "why", "bars": "two words"}
    assert "flags" not in json.loads(out["unset"])["cards"]["X"] and "flags" not in json.loads(out["nulled"])["cards"]["X"]
    assert out["lines"] == [["Flags: Needs doing by hand: new; Special stat bars cleared"], ["Flags: Needs doing by hand cleared; Special stat bars cleared"]]
    assert out["raw"] == [True, "why", False, ""] and out["names"] == "Special stat bars"
    assert out["ok"] == [True, False, False, True, ["W", "The Wall By Ech", "Blockade"]]
