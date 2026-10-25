"""
The parts editor's website side (TTSCardMaker site/, docs/Future/PARTS_EDITOR.md §2.5, §3, §5), against the frozen
contract fixtures in tests/fixtures/parts/: build_site's details pass-through (partState, skipped, variants), the
editor's per-layer reading (parts.js), Card Edits.json's per-layer merge (edit.js editsFile) and the mock-up's part jobs
(mockup.js). Loads the site from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips one without parts.js.
"""
import importlib.util
import json
import shutil
from os import environ
from os.path import expanduser
from pathlib import Path
from subprocess import run as run_process

import pytest

from conftest import CLONE

SITE = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE)))
ASSETS = SITE / "site" / "assets"
FIXTURES = Path(__file__).parent / "fixtures" / "parts"
pytestmark = pytest.mark.skipif(not (ASSETS / "parts.js").is_file(), reason=f"no parts.js at {ASSETS}")


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bs():
    spec = importlib.util.spec_from_file_location("site_build_site_parts", SITE / "site" / "build" / "build_site.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def node(tmp_path, body: str):
    """Run body with parts.js as P, edit.js as E and mockup.js as M; its stdout as JSON."""
    if not shutil.which("node"):
        pytest.skip("no node")
    script = tmp_path / "probe.mjs"
    script.write_text("".join(f"const {k} = await import({json.dumps((ASSETS / f).as_uri())});\n" for k, f in
                              (("P", "parts.js"), ("E", "edit.js"), ("M", "mockup.js")))
                      + f"const out = x => process.stdout.write(JSON.stringify(x));\n{body}\n", encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, encoding="utf-8")
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_build_passes_the_parts_on(bs):
    """Card Edits.json's parts are read per layer (a bad one dropped), name/ability wait on them, and each picture's
    details get its parts (shared, follow), partState (pending, stale, gone), the skipped edits while that PSD is current."""
    edits = fixture("card_edits.parts")["cards"]
    assert bs.valid_parts(edits["Miracle Matter"]["parts"]) and bs.valid_parts(edits["Panda"]["parts"])
    assert not bs.valid_parts({"1368": {"field": "ability", "text": "x", "delete": True, "sha": "bdc91d46", "psd": "a46deb90"}})
    assert not bs.valid_parts({"x1": {"field": "ability", "text": "x", "sha": "bdc91d46", "psd": "a46deb90"}})
    mm = dict(edits["Miracle Matter"])
    mm["parts"] = {**mm["parts"], "1369": {**mm["parts"]["1368"], "sha": "deadbeef"}, "77": mm["parts"]["1368"]}
    meta = {"parts": mm["parts"]}
    rec = {"parts": fixture("miracle_matter")["parts"]}
    card = {"id": "miracle-matter", "key": "Miracle Matter", "name": "Miracle Matter"}
    d = bs.details_parts(rec, card, {"miracle-matter": card}, meta, fixture("miracle_matter")["blob"], fixture("miracle_matter")["blob"])
    assert d["partState"] == {"1368": "pending", "417": "pending", "1369": "stale", "77": "gone"}
    assert [p["id"] for p in d["parts"]["ability"]] == [1367, 1369, 1368, 417, 1366] and d["picture"] is None and not d["stale"]

    entry = {"key": "Miracle Matter", "ability": "old", "stars": 3}
    found = {bs.card_key("Miracle Matter"): ({"ability": edits["Miracle Matter"]["ability"]}, {"parts": edits["Miracle Matter"]["parts"]})}
    bs.apply_edits(entry, found)
    assert entry["pending"] == ["ability"]

    snack = fixture("a_snack_tagged")
    cards = [{"id": "a-snack", "key": "A Snack", "name": "A snack", "variant": 1, "_variantOf": "A Snack"},
             {"id": "a-snack2", "key": "A Snack2", "name": "Breakfast", "variant": 2, "_variantOf": "A Snack"},
             {"id": "a-snack3", "key": "A Snack3", "name": "Supper", "variant": 3, "_variantOf": "A Snack"}]
    bs.link_variants(cards)
    assert cards[0]["variants"] == ["a-snack2", "a-snack3"] and cards[2]["variantOf"] == "a-snack"
    by_id = {c["id"]: c for c in cards}
    doomer = fixture("mini_doomer")
    v2 = bs.details_parts({"parts": snack["parts"]}, cards[1], by_id, {"parts": edits["A Snack"]["parts"]}, snack["blob"], snack["blob"])
    assert [p["id"] for p in v2["parts"]["title"]] == [1023] and v2["parts"]["ability"][0]["shared"] == ["A snack", "Supper"]
    assert v2["partState"] == {"320": "pending", "1023": "pending"} and v2["picture"]["n"] == 2
    skipped = bs.details_parts({"parts": doomer["parts"]}, {"id": "mini-doomer", "key": "Mini Doomer", "name": "Mini Doomer"}, {},
                               {"_psd": edits["Mini Doomer"]["_psd"]}, doomer["blob"], doomer["blob"])
    assert skipped["skipped"]["2068"]["text"].startswith("Has stats equal to the base stats -1")
    gone = bs.details_parts({"parts": doomer["parts"]}, {"id": "x", "key": "x", "name": "x"}, {}, {"_psd": edits["Mini Doomer"]["_psd"]},
                            "0" * 40, doomer["blob"])
    assert gone["skipped"] == {} and gone["stale"]


def test_the_boxes_read_per_layer(tmp_path):
    """A changed box is a part edit with the data's was/sha/psd; a box back at the PSD's text drops its edit; an emptied
    ability box deletes its layer; a title box can't be emptied; Enter in a title is a line break, the name one line; a
    stale edit stays until dropped; an untouched box with curly apostrophes is no edit; the derived join."""
    recs = {k: fixture(k)["parts"] for k in ("miracle_matter", "warcraftian_druid", "getting_started")}
    out = node(tmp_path, f"""
const F = {json.dumps(recs)}, rec = (k, extra = {{}}) => ({{ parts: F[k], psdBlob: "233e29c4aaaa", ...extra }});
const mm = rec("miracle_matter", {{ partEdits: {{ "1368": {{ field: "ability", text: "Earth: 5/8/6/5 and heal 3 HP", was: "x", sha: "bdc91d46", psd: "233e29c4" }},
  "1369": {{ field: "ability", text: "Fire: 9", was: "x", sha: "deadbeef", psd: "233e29c4" }} }}, partState: {{ "1368": "pending", "1369": "stale" }} }});
const L = P.boxes(mm), r1 = P.readParts(mm, L, {{ values: {{ "1366": "Gains the form's type.", "1368": "Earth: 5/8/6/5 and heal 2 HP", "417": "" }} }});
const r2 = P.readParts(mm, L, {{ drops: new Set(["1369"]) }});
const dr = rec("warcraftian_druid"), DL = P.boxes(dr), r3 = P.readParts(dr, DL, {{ values: {{ "1034": "Druid of\\nthe Claw" }} }});
const r4 = P.readParts(dr, DL, {{ values: {{ "422": " " }} }});
const gs = rec("getting_started"), GL = P.boxes(gs, true), r5 = P.readParts(gs, GL, {{ values: {{ "15": "HOW TO WIN:" }} }});
out({{ start: L.find(b => b.key === "1368").start, labels: L.map(b => b.label), r1, r2: r2.unset, r3, d3: P.derived(DL, r3.texts),
  r4: r4.error, r5: Object.keys(r5.set), blocks: GL.map(b => b.label).slice(0, 3) }});
""")
    assert out["start"] == "Earth: 5/8/6/5 and heal 3 HP" and out["labels"][:2] == ["Title", "Ability 1/5"]
    r1 = out["r1"]
    assert r1["set"]["1366"] == {"field": "ability", "text": "Gains the form's type.", "was": recs["miracle_matter"]["ability"][4]["text"],
                                 "sha": "80ae66fc", "psd": "233e29c4", "layer": "Text/Stat Txt/ability/Ability"}
    assert r1["set"]["417"]["delete"] and "1368" in r1["unset"] and "1369" not in r1["unset"]
    assert out["r2"] == ["1369"]
    assert out["r3"]["set"]["1034"]["field"] == "name" and out["d3"]["name"] == "Warcraftian Druid of the Claw"
    assert "title box can't be empty" in out["r4"]
    assert out["r5"] == ["15"] and out["blocks"] == ["Header", "Welcome", "TO WIN:"]


def test_card_edits_merge_per_layer_and_say_who_edited_a_box_meanwhile(tmp_path):
    """editsFile: parts merged per layer id (set and part:<id> unset), right after ability; a box someone else changed
    since it was staged calls onConflict."""
    out = node(tmp_path, """
const file = JSON.stringify({ cards: { "Miracle Matter": { stars: 4, ability: "a", parts: {
  "1368": { field: "ability", text: "theirs", was: "w", sha: "bdc91d46", psd: "233e29c4" },
  "417": { field: "ability", delete: true, was: "w", sha: "e8e12acf", psd: "233e29c4" } }, _by: "x", _at: "t" } } });
const op = { key: "Miracle Matter", name: "Miracle Matter", set: { ability: "b", parts: { "1366": { field: "ability", text: "new", was: "w", sha: "80ae66fc", psd: "233e29c4" },
  "1368": { field: "ability", text: "mine", was: "w", sha: "bdc91d46", psd: "233e29c4" } } }, unset: ["part:417"],
  partsBase: { "1368": { field: "ability", text: "before", was: "w", sha: "bdc91d46", psd: "233e29c4" }, "1366": null } };
const clashes = [];
const e = JSON.parse(E.editsFile(file, [op], "now", "me", "site/Card Edits.json", (o, keys) => clashes.push(...keys))).cards["Miracle Matter"];
out({ keys: Object.keys(e), parts: Object.keys(e.parts).sort(), mine: e.parts["1368"].text, clashes });
""")
    assert out["keys"][:4] == ["stars", "ability", "parts", "_by"] and out["parts"] == ["1366", "1368"]
    assert out["mine"] == "mine" and out["clashes"] == ["1368"]


def test_the_mock_up_redraws_changed_layers_only(tmp_path):
    """partJobs: a changed part in its own box, a deleted one erased (text null), an unchanged one left; an added ability
    in the template's box."""
    rec = {"parts": fixture("miracle_matter")["parts"]}
    druid = {"parts": {**fixture("warcraftian_druid")["parts"], "ability": [], "template": {"ability": {"box": [1, 2, 3, 4]}}},
             "template": {"ability": {"box": [1, 2, 3, 4]}}}
    out = node(tmp_path, f"""
const jobs = M.partJobs({json.dumps(rec)}, {{ "1368": {{ text: "Earth: 9" }}, "417": {{ delete: true }}, "1369": {{ text: "Fire: 5/5/8/7" }} }});
const add = M.partJobs({json.dumps(druid)}, {{ "+ability": {{ text: "Heal 1 HP." }} }});
out({{ jobs: jobs.map(j => [j.id, j.role, j.text, j.box]), add: add.map(j => [j.id, j.added, j.box]) }});
""")
    assert out["jobs"] == [["1368", "ability", "Earth: 9", [124, 1969, 1035, 2024]], ["417", "ability", None, [125, 2031, 958, 2086]]]
    assert out["add"] == [["+ability", True, [1, 2, 3, 4]]]
