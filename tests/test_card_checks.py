"""
Automatic flags (commands/card_checks.py): a bar read off the picture, the picture against the data, the file's life (a
fixed card clears, a person's flag goes once its PSD changed and the checks are clean), the soft notes told nowhere, and
card-checks.yml's copy of the bot (2026-11-07).
"""
import json
import shutil
import subprocess
from pathlib import Path
from sys import executable
from types import SimpleNamespace

import numpy as np
import pytest

import commands.card_checks as cc
from commands.card_edits import CardEdits
from commands.psd_edit import SOFT_NOTE, real_notes
from conftest import SITE_SRC

REPO = Path(__file__).resolve().parent.parent
LIT, EMPTY = (47, 191, 58), (16, 63, 19)


def bar(cells: int, bottom: int = None) -> np.ndarray:
    """An 813x84 template bar, the first cells lit; bottom: its bottom half's (past 10)."""
    row = np.zeros((84, 813, 3), np.uint8)
    edge = lambda n: cc.TEMPLATE_CELLS[n]
    row[:] = EMPTY
    row[:, :edge(cells)] = LIT
    if bottom is not None:
        row[:42] = LIT
        row[42:] = EMPTY
        row[42:, :edge(bottom)] = LIT
    return row


def test_read_row():
    """A step in both halves, a step in the bottom half alone, a flat row, a busy one."""
    assert cc.read_row(bar(5)) == {"best": 5, "could": [5]}
    assert cc.read_row(bar(10, 2))["best"] == 12
    soft = bar(10)
    soft[42:, cc.TEMPLATE_CELLS[2]:] -= 40                      # see-through: a soft step, the bottom half alone
    assert cc.read_row(soft)["best"] == 12
    assert cc.read_row(bar(10)).get("flat") and cc.read_row(bar(0)).get("flat")
    assert cc.read_row(np.random.default_rng(1).integers(0, 255, (84, 813, 3)).astype(np.uint8))["best"] is None


def test_flat_rows_by_the_others():
    """A flat row is full or empty by the card's stepped rows."""
    rows = [(row, cc.read_row(row)) for row in (bar(4), bar(10), bar(0))]
    cc._tell_flats(rows)
    assert [read["best"] for _, read in rows] == [4, 10, 0] and rows[1][1]["could"] == [10, 20]


def test_split_row():
    """A busy row of textured cells (Kyle's, Michael's) reads by its cells' lightness; noise and fading paint (Paint
    Warrior's 5 that splits best at 2) read nothing."""
    rng = np.random.default_rng(2)
    row, fade = np.zeros((84, 813, 3), np.uint8), np.zeros((84, 813, 3), np.uint8)
    for n, grey in enumerate((135, 110, 52, 50, 78, 9, 14, 7, 5, 30)):
        row[:, cc.TEMPLATE_CELLS[n]:cc.TEMPLATE_CELLS[n + 1]] = rng.integers(140, 256, 3)
        fade[:, cc.TEMPLATE_CELLS[n]:cc.TEMPLATE_CELLS[n + 1]] = grey
    row[:, cc.TEMPLATE_CELLS[7]:] //= 3                         # the darks over cells 8-10
    assert cc.read_row(row)["could"] is None and cc.split_row(row) == {"best": 7, "could": [7, 17]}
    assert cc.split_row(np.random.default_rng(1).integers(0, 255, (84, 813, 3)).astype(np.uint8)) is None
    assert cc.read_row(fade)["could"] is None and cc.split_row(fade) is None


def cell_row(*colours) -> np.ndarray:
    """An 813x84 template bar, a colour a cell."""
    row = np.zeros((84, 813, 3), np.uint8)
    for n, colour in enumerate(colours):
        row[:, cc.TEMPLATE_CELLS[n]:cc.TEMPLATE_CELLS[n + 1]] = colour
    return row


def test_busy_rows():
    """Cells are read on their middles: a light gap between cells 5 and 6 (Nailsage Sly's) stays out. A busy row left
    unsplit reads full or empty by the card's stepped rows (Claw's ATK); a mixed one stays open."""
    gap = cell_row(110, 75, 110, 75, 40, 30, 40, 30, 40, 30)
    gap[:, cc.TEMPLATE_CELLS[5] - 14:cc.TEMPLATE_CELLS[5] + 14] = 230
    assert cc.read_row(gap)["could"] is None and cc.split_row(gap) == {"best": 4, "could": [4, 14]}
    rows = [(row, cc.read_row(row)) for row in (bar(4), cell_row(*[LIT, (90, 230, 100)] * 5),
                                                cell_row(*[EMPTY, (40, 100, 45)] * 5), cell_row(*[LIT, EMPTY] * 5))]
    assert [read["could"] for _, read in rows[1:]] == [None] * 3
    cc._tell_busy(rows)
    assert [read["could"] for _, read in rows] == [[4], [10, 20], [0], None]


def test_colour_row():
    """Lit and empty cells of one lightness (zebra art on green: PoD04Zebra Bloon) read by colour; noise: nothing."""
    row = cell_row(*[(60, 150, 60)] * 10)
    stripes = (np.arange(cc.TEMPLATE_CELLS[3]) // 24) % 2 == 0
    row[:, :cc.TEMPLATE_CELLS[3]] = np.where(stripes[:, None], 255, 0)
    assert cc.read_row(row)["could"] is None and cc.split_row(row) is None
    assert cc.colour_row(row) == {"best": 3, "could": [3, 13]}
    assert cc.colour_row(np.random.default_rng(1).integers(0, 255, (84, 813, 3)).astype(np.uint8)) is None


def test_mismatches():
    """The note names the bar; a value the bar could show says nothing; a custom card speaks only when every bar reads
    by a step, or bar by bar read by the bake's drawing (one with no say silences none); a website-only value differs on
    purpose; creatures by bar."""
    read = lambda best, could=None: {"best": best, "could": could if could is not None else [best]}
    record = {"style": "new", "reads": {"hp": [read(5)], "def": [read(3)], "atk": [read(None, None) | {"could": None}], "spd": [read(4, [4, 7])]}}
    data = {"stats": {"hp": 5, "defense": 1, "attack": 2, "speed": 7}}
    assert cc.mismatches(record, data) == ["The DEF bar shows 3 but the data says 1."]
    assert cc.mismatches(record, data, ["def"]) == [] and cc.mismatches(record, data, {"def": 1}) == []   # website only, pending
    assert cc.mismatches({**record, "style": "custom"}, data) == []
    many = {"style": "custom", "reads": {"hp": [read(5), read(6)]}}
    assert cc.mismatches(many, {"creatures": [{"hp": 5}, {"hp": 4}]}) == ["Creature 2's HP bar shows 6 but the data says 4."]
    flat = {"style": "custom", "reads": {"hp": [{"best": 0, "could": [0], "flat": True}], "atk": [read(5)]}}
    assert cc.mismatches(flat, {"stats": {"hp": 1, "attack": 5}}) == []          # a flat custom row: no step read
    drawn = {"style": "custom", "reads": {"hp": [cc.drawn_read({4: (-90, 100), 6: (-95, 100)}, 5)],
                                          "def": [cc.drawn_read({2: (80, 100), 4: (-100, 100)}, 3)],
                                          "atk": [cc.drawn_read({1: (0, 100)}, 2)]}}
    assert [rows[0]["best"] for rows in drawn["reads"].values()] == [5, 2, None]
    assert cc.mismatches(drawn, {"stats": {"hp": 5, "defense": 3, "attack": 9}}) == ["The DEF bar shows 2 but the data says 3."]
    assert cc.flags({"notes": ["N."], "bars": "B."}, None) == {"hand": "N.", "bars": "B.", "name": None}


def test_custom_bars_read_by_the_bakes_drawing(clone):
    """Puzzled Cat's PNG shows her DEF 4 against the bake's picture of 9 (card_plates.looks); a picture of 9 shows 9,
    flagged against the data's 4, though the bake makes no picture off one that far from the card's own."""
    from io import BytesIO
    from PIL import Image
    from psd_tools import PSDImage
    import commands.card_plates as cp
    import commands.psd_edit as psd_edit
    rel = "Creatures/Other/3 Stars/Puzzled_Cat.psd"
    raw = (clone / rel).read_bytes()
    psd, png = PSDImage.open(BytesIO(raw)), Image.open(clone / rel.replace(".psd", ".png"))
    png.load()
    parser = psd_edit._bare_parser()
    parsed = psd_edit.parse_card(parser, str(clone / rel), rel, psd)
    psd_edit.use_record(psd, rel)
    icc = psd_edit.export_icc(psd, png)

    def read(picture):
        seen = []
        cp.bar_value(psd, raw, picture, parsed, parser, rel, "def", 9, icc, seen=seen)
        return cc.drawn_read({9: seen[0]}, 4)
    wrong = cp.bar_value(psd, raw, png, parsed, parser, rel, "def", 9, icc)
    assert read(png)["best"] == 4 and cp.bar_value(psd, raw, wrong, parsed, parser, rel, "def", 9, icc) is None
    record = {"style": "custom", "reads": {"def": [read(wrong)]}}
    assert cc.mismatches(record, {"stats": {"defense": 4}}) == ["The DEF bar shows 9 but the data says 4."]


def test_printed_bars_read_by_the_bakes_drawing(clone):
    """Baba's word bars print their value ('IS 6'), which the cells misread (its own PNG after a DEF 11 bake read 0: a
    false hand flag): read_picture gives them no say; the bake's drawing reads that PNG as 11, the old one as 6."""
    from io import BytesIO
    from PIL import Image
    from psd_tools import PSDImage
    import commands.card_plates as cp
    import commands.psd_edit as psd_edit
    from conftest import card_blob
    rel, parser = "Creatures/Other/5 Stars/Baba.psd", psd_edit._bare_parser()
    old = Image.open(BytesIO(card_blob(clone, "31b457d7a5bbb076e1c10a4eebe1999ab46ef4c8")))
    old.load()

    def card(raw):
        psd = psd_edit.use_record(PSDImage.open(BytesIO(raw)), rel)
        return psd, raw, psd_edit.parse_card(parser, rel, rel, psd)
    psd, raw, parsed = card(card_blob(clone, "2746abaa00679647b385787c4bb4e384b135ad87"))
    icc = psd_edit.export_icc(psd, old)
    png = cp.bar_value(psd, raw, old, parsed, parser, rel, "def", 11, icc)          # the bake's PNG
    new = psd_edit.edit_copy(psd, raw)
    psd_edit.set_stat(new, parsed, "def", 11)
    new.save(saved := BytesIO())
    psd, raw, parsed = card(saved.getvalue())
    assert cc.read_picture(psd, png, cells=True)["def"] == [{"best": None, "could": None, "printed": True}]

    def read(picture):
        seen = []
        cp.bar_value(psd, raw, picture, parsed, parser, rel, "def", 6, icc, seen=seen)
        return {"style": "new", "reads": {"def": [cc.drawn_read({6: seen[0]}, parsed["card"]["def"])]}}
    data = {"stats": {"defense": 11}}
    assert cc.mismatches(read(png), data) == []
    assert cc.mismatches(read(old), data) == ["The DEF bar shows 6 but the data says 11."]


def test_name_note():
    """Owner 11-10/11-11: the website's name against the printed title (exact), the title's file form against the file
    name (no . : [ ] ? * " < > | / \\, spaces as underscores, small words' case and TM (R) (C) aside); a second picture
    never; names a person marked intended (names_ok) stay clear until one changes; refresh words it from the data, no
    PSD read."""
    from commands.card_models import file_stem
    assert [file_stem(t) for t in ("Cave of The Past", "P.A.N.D.A. Mk A", "[Thunderfury: Blessed Blade of the Windseeker]",
                                   "Dr. Robotnik\u2019s Roboticizer", "Foo.Bar", "...", "con")] == \
        ["Cave_of_the_Past", "PANDA_Mk_A", "Thunderfury_Blessed_Blade_of_the_Windseeker", "Dr_Robotnik's_Roboticizer", "FooBar", "", ""]
    rel = "Creatures/X/Avatar_Of_Wrath.psd"
    card = {"path": rel, "display_name": "Avatar of Wrath"}
    assert cc.name_note(card, {}, rel) is None                                          # small words' case alone
    assert cc.name_note({**card, "display_name": "Dr. Avatar: of Wrath"}, {}, "C/Dr_Avatar_of_Wrath.psd") is None
    assert cc.name_note({**card, "display_name": "Avatar of Wrath"}, {}, "C/Avatar_of_Wrath.psd") is None
    assert cc.name_note({"path": "C/E._Gadd.psd"}, {}, "C/E._Gadd.psd") == 'The card prints "E. Gadd" but its file is E._Gadd.psd.'
    assert cc.name_note({"path": "C/E_Gadd.psd"}, {}, "C/E_Gadd.psd") is None
    assert cc.name_note({"path": "C/Marked For Death.psd"}, {}, "C/Marked For Death.psd") is None          # spaces as underscores
    stock = {"path": "N/NME_Stock.psd", "display_name": "NME Stock\u2122"}                   # TM (R) (C) never count
    assert cc.name_note(stock, {}, "N/NME_Stock.psd") is None and cc.name_note(stock, {}, "N/NME_Stock\u00ae.psd") is None
    assert cc.name_note({**stock, "display_name": "\u00a9 NME Stock"}, {}, "N/NME_Stock.psd") is None
    assert cc.name_note(stock, {}, "N/NME_Stocks.psd") == 'The card prints "NME Stock\u2122" but its file is NME_Stocks.psd.'
    blockade = {"path": "C/Blockade.psd", "display_name": "The Wall By Ech"}
    assert cc.name_note(blockade, {}, "C/Blockade.psd") == 'The card prints "The Wall By Ech" but its file is Blockade.psd.'
    assert cc.name_note({**blockade, "variant_of": "A"}, {}, rel) is None
    assert cc.name_note({**blockade, "name": "Wall2", "duplicate_of": "A"}, {}, rel) is None            # picture only
    assert cc.name_note({**blockade, "name": "Blockade", "duplicate_of": "A"}, {}, "C/Blockade.psd") == \
        'The card prints "The Wall By Ech" but its file is Blockade.psd.'                            # marked, own PSD
    shield = "Items/Man_Serpent_Shield.psd"
    edited = {"path": shield, "display_name": "Man-Serpent's Shield", "parsed_values": {"name": None}}
    assert cc.name_note(edited, {"name": "Man-Serpent's Shield"}, shield) == \
        'The website shows "Man-Serpent\'s Shield" but the card prints "Man Serpent Shield".'
    caps = {"path": "C/Delta.psd", "display_name": "Delta", "parsed_values": {"name": "DELTA"}}
    assert cc.name_note(caps, {"name": "Delta"}, "C/Delta.psd") == \
        'The card prints "DELTA" but the website shows "Delta" and its file is Delta.psd.'
    assert cc.name_note(caps, {"name": "Delta", "names_ok": ["Delta", "DELTA", "Delta"]}, "C/Delta.psd") is None
    assert cc.name_note(caps, {"name": "Delta", "names_ok": ["Delta", "DELTA", "Delta"]}, "C/Delta2.psd") is not None   # a name changed
    snapshot = SimpleNamespace(psds={"blockade": "C/Blockade.psd"}, sha=lambda path: "a" * 40)
    checks = {"cards": {"a" * 40: {"notes": [], "bars": None}}, "cleared": {}}
    stats = {"Blockade": {**blockade, "blob_sha": "a" * 40},
             "Wall2": {"path": "C/Blockade.psd", "display_name": "Wall", "duplicate_of": "Blockade", "variant_of": "Blockade"}}
    cc.refresh(checks, snapshot, stats, CardEdits.parse("{}"))
    assert checks["cards"]["a" * 40]["name"] == cc.NAME_FILE.format(printed="The Wall By Ech", file="Blockade.psd")
    assert checks["cards"]["a" * 40]["card"] == "Blockade"                            # its file's, after a rename too
    cc.refresh(checks, snapshot, {**stats, "Blockade": {**stats["Blockade"], "name": "Blockade", "duplicate_of": "A"}},
               CardEdits.parse("{}"))
    assert checks["cards"]["a" * 40]["name"] == cc.NAME_FILE.format(printed="The Wall By Ech", file="Blockade.psd")
    cc.refresh(checks, snapshot, stats, CardEdits.parse(json.dumps({"cards": {"Blockade": {"name": "Wall", "_site": ["name"]}}})))
    assert checks["cards"]["a" * 40]["name"].endswith('but the website shows "Wall" and its file is Blockade.psd.')
    marked = {"cards": {"Blockade": {"names_ok": ["The Wall By Ech", "The Wall By Ech", "Blockade"]}}}
    cc.refresh(checks, snapshot, stats, CardEdits.parse(json.dumps(marked)))
    assert checks["cards"]["a" * 40]["name"] is None
    assert not CardEdits.valid("names_ok", ["a", "b"]) and not CardEdits.valid("names_ok", ["a", "", "c"])


def test_refresh_clears():
    """A website-only fix clears without a push; a gone PSD's record goes; a person's flag goes once its PSD changed and
    the checks are clean, stays gone though they speak again, and a new flag starts over."""
    rel = "Creatures/A.psd"
    snapshot = SimpleNamespace(psds={"a": rel}, sha=lambda path: "abcd1234ffff")
    read = {"best": 2, "could": [2]}
    checks = {"cards": {"abcd1234ffff": {"style": "new", "reads": {"def": [read]}, "notes": [], "bars": None}, "gone": {}}, "cleared": {}}
    stats = {"A": {"blob_sha": "abcd1234ffff", "stats": {"defense": 8}}}
    edits = CardEdits.parse(json.dumps({"cards": {"A": {"flags": {"hand": "x", "psd": "00000000"}}}}))
    steps = {rel: [("00000000ffff", "abcd1234ffff", False)]}
    cc.refresh(checks, snapshot, stats, edits, steps=steps)
    assert list(checks["cards"]) == ["abcd1234ffff"] and checks["cards"]["abcd1234ffff"]["hand"].startswith("The DEF bar")
    assert checks["cleared"] == {}
    cc.refresh(checks, snapshot, {"A": {"blob_sha": "abcd1234ffff", "stats": {"defense": 2}}}, edits)
    assert checks["cleared"] == {}                              # no changelog: unknown never clears
    cc.refresh(checks, snapshot, {"A": {"blob_sha": "abcd1234ffff", "stats": {"defense": 2}}}, edits, steps=steps)
    assert checks["cards"]["abcd1234ffff"]["hand"] is None and checks["cleared"] == {rel: "00000000"}
    cc.refresh(checks, snapshot, stats, edits)
    assert checks["cleared"] == {rel: "00000000"}
    record = checks["cards"].pop("abcd1234ffff")
    cc.refresh(checks, snapshot, stats, edits)                  # a new PSD not read yet: still gone
    assert checks["cleared"] == {rel: "00000000"}
    checks["cards"]["abcd1234ffff"] = record
    edits = CardEdits.parse(json.dumps({"cards": {"A": {"flags": {"hand": "y", "psd": "abcd1234"}}}}))
    cc.refresh(checks, snapshot, stats, edits)
    assert checks["cleared"] == {}


def test_a_drawn_duplicate_is_no_flag(clone, monkeypatch):
    """Owner 11-09: a picture-only duplicate (dup_of) its PSD's variant tags draw (picture 2 titled as it: Gust of Trees)
    is no "drawn apart" flag, never measured (Regeneration Potion's clipped adjustment misreads); untagged, it is."""
    import sys
    from hashlib import sha1
    from io import BytesIO
    from psd_tools import PSDImage
    from commands.card_bake import Snapshot
    from commands.psd_edit import _bare_parser
    sys.path.insert(0, str(Path(__file__).parent / "fixtures" / "parts"))
    from make_fixtures import CARDS
    rel, _, tags = CARDS["a_snack_tagged"]
    stem = rel[:-4]
    psd = PSDImage.open(str(clone / rel))
    for layer in psd.descendants():
        if layer.layer_id in tags:
            layer.name = tags[layer.layer_id]
    tagged = BytesIO()
    psd.save(tagged)
    files = {stem + ".png": (clone / (stem + ".png")).read_bytes(), stem + "3.png": (clone / (stem + "3.png")).read_bytes(),
             rel.rsplit("/", 1)[0] + "/Breakfast.png": (clone / (stem + "2.png")).read_bytes(),
             cc.EDITS_PATH: json.dumps({"cards": {"Breakfast": {"dup_of": "A Snack"}}}).encode()}
    monkeypatch.setattr(cc, "_stat_notes", lambda *args: [])    # the stat bakes: test_card_bake's
    monkeypatch.setattr(cc.png_export, "drawn_apart", lambda *args: True)  # a drawn picture is never measured
    sha = lambda data: sha1(b"blob %d\0" % len(data) + data).hexdigest()
    for data, flagged in (((clone / rel).read_bytes(), True), (tagged.getvalue(), False)):
        found = {rel: data, **files}
        snapshot = Snapshot("", "", {}, {path: ("100644", sha(blob)) for path, blob in found.items()})
        snapshot.fetch = {sha(blob): blob for blob in found.values()}.get
        record = cc.check_card(cc.Context(snapshot, _bare_parser(), ""), rel)
        assert (cc.DUP_NOTE.format(name="Breakfast.png") in record["notes"]) is flagged, record["notes"]


def test_sweeps_keep_flags(tmp_path):
    """A person's flag stays through sweeps (db/changelog's "sweep") and an unknown chain; a person's push clears it."""
    rel, mark = "Creatures/A.psd", "11111111"
    log = tmp_path / "changelog"
    log.mkdir()
    event = lambda seq, old, new, sweep=None: json.dumps({"seq": seq, "kind": "art", "path": rel, "blob": [old, new],
                                                           **({"sweep": sweep} if sweep else {})})
    (log / "2026-11.jsonl").write_text("\n".join([event(1, "0" * 40, mark + "f" * 32), event(2, mark + "f" * 32, "2" * 40, "x"),
                                                  event(3, "2" * 40, "3" * 40, "y")]) + "\n", encoding="utf-8")
    steps = cc.psd_steps(str(tmp_path))
    assert not cc.by_person(steps[rel], mark) and not cc.by_person([], mark) and not cc.by_person(steps[rel][2:], mark)
    assert cc.by_person(steps[rel] + [("3" * 40, "4" * 40, False)], mark)
    snapshot = SimpleNamespace(psds={"a": rel}, sha=lambda path: "4" * 40)
    checks = {"cards": {"4" * 40: {"style": "new", "reads": {}, "notes": [], "bars": None}}, "cleared": {}}
    edits = CardEdits.parse(json.dumps({"cards": {"A": {"flags": {"hand": "x", "psd": mark}}}}))
    cc.refresh(checks, snapshot, {}, edits, steps={rel: steps[rel] + [("3" * 40, "4" * 40, True)]})
    assert checks["cleared"] == {}
    cc.refresh(checks, snapshot, {}, edits, steps={rel: steps[rel] + [("3" * 40, "4" * 40, False)]})
    assert checks["cleared"] == {rel: mark}


def test_still_and_swapped(monkeypatch):
    """A bar a bake left as it was, though the others moved; a swap by the printed labels, equal values too."""
    psd = SimpleNamespace(width=100)
    boxes = {"hp": [(0, 0, 100, 10)], "def": [(0, 10, 100, 20)], "atk": [(0, 20, 100, 30)], "spd": [(0, 30, 300, 40)]}
    monkeypatch.setattr(cc, "bar_boxes", lambda psd: boxes)
    monkeypatch.setattr(cc, "by_label", lambda psd, found: {**found, "def": found["atk"], "atk": found["def"]})
    moved = np.zeros((40, 100), bool)
    assert cc._still(psd, moved, "custom") == ["hp", "def", "atk", "spd"]
    moved[:20] = True
    assert cc._still(psd, moved, "new") == ["def"]              # by its row; spd past the canvas has no say
    assert cc._still(psd, moved, "custom") == []                # its boxes need not be where the picture changes
    assert cc.swapped(psd) == ["def", "atk"]


def test_bar_style_by_art_fill(monkeypatch):
    """Croissant's continuous bars (0.86 of their box: Zigg, Uumuu) are the template's; Monkey D Letion's icons (0.68) not."""
    monkeypatch.setattr(cc, "comp_bars", lambda psd: {"hp": [(None, 5, (0, 0, 1, 1))]})
    for fill, style in ((0.86, ("new", "")), (0.68, ("custom", "icons"))):
        monkeypatch.setattr(cc.psd_edit, "art_fill", lambda psd, box: fill)
        assert cc.bar_style(None, None) == style


def test_art_over_darks(monkeypatch):
    """Only Lines may lie over the new bars' Darks group: else a stat edit bakes for real (Sans, The Thin Man)."""
    layer = lambda name, shown=True: SimpleNamespace(name=name, is_visible=lambda: shown)
    darks, bars = layer("Darks"), layer("Bars")
    monkeypatch.setattr(cc, "comp_bars", lambda psd: {"hp": [(SimpleNamespace(parent=darks), 5, (0, 0, 1, 1))]})
    darks.parent = [bars, darks, layer("Lines")]
    assert not cc._art_over_darks(None)
    darks.parent = [darks, bars, layer("Lines")]
    assert cc._art_over_darks(None)
    darks.parent = [darks, layer("Bars", False)]
    assert not cc._art_over_darks(None)


def test_raises_stop_at_the_display(monkeypatch):
    """A stat raise stays within the card's own display (Computer Virus's rows: more is refused when typed, owner
    2026-11-08, never a note): a stat at its room steps down; creature 2's own row by its own room. Both bakes share
    one render cache."""
    monkeypatch.setattr(cc.psd_edit, "stat_caps", lambda psd: {"hp": [5, 5], "def": [4, 4], "atk": [6, 6], "spd": [5, 5]})
    monkeypatch.setattr(cc.psd_edit, "creature_bar", lambda psd, stat, n: (stat, n))
    monkeypatch.setattr(cc, "stat_rows", lambda psd: {})
    baked, caches = [], []
    monkeypatch.setattr(cc, "_real_bake", lambda ctx, rel, fields, png: (
        baked.append(fields), caches.append(cc.psd_edit.render_cache(shared=True)), ("baked", "", None))[2])
    first, second = {"hp": 2, "def": 4, "atk": 3, "spd": 5}, {"hp": 3, "def": 2, "atk": 6, "spd": 5}
    assert cc._stat_notes(None, "C.psd", None, {"card": {**first, "creatures": [first, second]}}, "custom", None) == []
    assert caches[0] is caches[1] and cc.psd_edit.render_cache(shared=True) is not caches[0]
    up = baked[1]
    assert {k: up[k] for k in first} == {"hp": 3, "def": 3, "atk": 4, "spd": 4}
    assert up["creatures"] == [{"hp": 3, "def": 3, "atk": 4, "spd": 4}, {"hp": 4, "def": 3, "atk": 5, "spd": 4}]


def test_items_rows_bake_too(monkeypatch):
    """An item's HP and DEF rows are bake-tested as well: a refused one is a note (Dirty Bed's DEF, 2026-11-11)."""
    monkeypatch.setattr(cc.psd_edit, "stat_caps", lambda psd: None)
    row = [([], [], None)]                      # (group, shown digits, box)
    monkeypatch.setattr(cc, "stat_rows", lambda psd: {"hp": row, "def": row, "atk": [], "spd": []})
    baked = []
    monkeypatch.setattr(cc, "_real_bake", lambda ctx, rel, fields, png: (
        baked.append(fields), ("refused", "unsupported-card: def 1: the parser would read 2 off 'def'", None))[1])
    card = {"card": {"hp": 3, "def": 0, "atk": None, "spd": None}}
    assert cc._stat_notes(None, "I.psd", None, card, "few", None) == \
        ["Stat edits need Photoshop: def 1: the parser would read 2 off 'def'."]
    assert baked == [{"hp": 2, "def": 1}]


def test_to_check_and_file():
    """No result, or a new PNG: checked; --all, a shard; the file reads back."""
    tree = {"A.psd": ("", "11111111aa"), "A.png": ("", "22222222bb"), "B.psd": ("", "33333333cc")}
    snapshot = SimpleNamespace(psds={"a": "A.psd", "b": "B.psd"}, sha=lambda path: tree[path][1],
                               png_for=lambda rel: ("A.png" if rel == "A.psd" else "", []), dup_pictures=dict)
    checks = {"version": 1, "cards": {"11111111aa": {"png": "22222222"}, "33333333cc": {"png": ""}}, "cleared": {}}
    assert cc.to_check(snapshot, checks) == []
    assert cc.to_check(snapshot, checks, True) == ["A.psd", "B.psd"]
    assert cc.to_check(snapshot, checks, only={"b"}) == ["B.psd"]
    assert cc.to_check(snapshot, {"cards": {"11111111aa": {"png": "old"}}}, shard=(1, 2)) == ["A.psd", "B.psd"]
    assert json.loads(cc.text_of(checks))["cards"] == checks["cards"]


def test_blobs_of_in_reading_order():
    """The chart and manifest first, then each card's PSD, PNG and duplicates together."""
    paths = ["MDW/Templates/Creature_Template.psd", "A.psd", "A.png", "A2.png", "B.psd", "Types/Fire.png", "Types/Dark.png",
             cc.psd_edit.STATUS_CHART, cc.ICON_MANIFEST_PATH]
    tree = {path: ("", path) for path in paths}
    snapshot = SimpleNamespace(tree=tree, sha=lambda path: path, dup_pictures=lambda: {"A.psd": ["A2.png"]},
                               png_for=lambda rel: ("A.png" if rel == "A.psd" else "", []))
    assert cc.blobs_of(snapshot, ["A.psd", "B.psd"]) == [cc.psd_edit.STATUS_CHART, cc.ICON_MANIFEST_PATH,
                                                        "A.psd", "A.png", "A2.png", "B.psd"]
    assert cc.blobs_of(snapshot, []) == []


def test_soft_notes():
    """The old list's "bakes with flag" kinds are told nowhere; real problems and layouts stay."""
    soft = ["overflow: name reaches the stars or type icons", "overflow: name: moved +12 px to stay on the card",
            "types: 1 made from Types/, the first at Creature_Template's spot: check it", "ability: also printed by Ability copy"]
    real = ["stars not baked: unsupported-card: stars: x", "font: not resolved exactly, drawn in a stand-in: A", "stars: rows of two"]
    assert real_notes(soft + real) == real and all(SOFT_NOTE.search(note) for note in soft)


def test_action_shaped_copy(tmp_path):
    """Exactly SPARSE_FILES, no config.py: --help works and the closure is that list; card-checks.yml checks it out."""
    bot = tmp_path / "bot"
    for rel in cc.SPARSE_FILES:
        (bot / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, bot / rel)
    done = subprocess.run([executable, "-m", "commands.card_checks", "--help"], cwd=bot, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    code = ("import sys, commands.card_checks, commands.txt2_repair, commands.ability_place; "
            "print(sorted(m for m in ('config', 'discord', 'pandas') if m in sys.modules)); "
            "print(sorted(m for m in sys.modules if m.startswith('commands.') or m == 'global_config'))")
    done = subprocess.run([executable, "-c", code], cwd=bot, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    bad, closure = done.stdout.splitlines()
    assert bad == "[]"
    assert {m.replace(".", "/") + ".py" for m in json.loads(closure.replace("'", '"'))} == {f for f in cc.SPARSE_FILES if f.endswith(".py")}
    workflow = SITE_SRC / ".github/workflows/card-checks.yml"
    if workflow.is_file():
        yaml = pytest.importorskip("yaml")
        doc = yaml.safe_load(workflow.read_text(encoding="utf-8"))
        checkout = next(s for s in doc["jobs"]["check"]["steps"] if s.get("with", {}).get("repository"))
        assert checkout["with"]["sparse-checkout"] == "${{ env.BOT_FILES }}"     # and Clones start's
        assert doc["env"]["BOT_FILES"].split() == ["/" + f for f in cc.SPARSE_FILES]


def test_link_notes_word_the_links_against_the_tree():
    """db/links.json's hand notes (owner 11-07, plain words): a file the repo lacks (cleared once it is pushed), a
    look-alike of another linked file, a changed file Card sync could not draw again; none off the record's blob."""
    tree = {"MDW/Linked/a.png": "b1", "MDW/Linked/b.png": "b2", "Types/fire.png": "f2"}
    links = {"X/C.psd": {"blob": "c1", "links": [{"path": "MDW/Linked/gone.png", "name": "gone.png", "size": 3, "drawn": None},
                                                 {"path": "MDW/Linked/b.png", "name": "b.png", "size": 5, "drawn": None},
                                                 {"path": "Types/fire.png", "name": "fire.png", "size": 9, "drawn": "f1"}]},
             "X/D.psd": {"blob": "d1", "links": [{"path": "MDW/Linked/a.png", "name": "a.png", "size": 4, "drawn": None}]},
             "X/E.psd": {"blob": "e1", "links": [{"path": "MDW/Linked/a.png", "name": "a.png", "size": 4, "drawn": None}]},
             "files": {"Types/fire.png": {"f1": 9, "f2": 9}, "MDW/Linked/b.png": {"b2": 5}},
             "pixels": {"b1": "same", "b2": "same"}}

    def sha(path):
        return tree.get(path, "")
    alike = cc.look_alikes(links, sha, tree)
    assert alike == {"MDW/Linked/b.png": "MDW/Linked/a.png"}      # a.png has more users: b.png's get the note
    b = [{"path": "MDW/Linked/b.png", "name": "b.png", "size": 5, "drawn": None}]
    assert cc.look_alikes({**links, "X/D.psd": {"blob": "d1", "links": b}, "X/E.psd": {"blob": "e1", "links": b}},
                          sha, tree) == {"MDW/Linked/a.png": "MDW/Linked/b.png"}
    record = {"unredrawable": {"Types/fire.png": ["Fire", "a warp"]}}
    assert cc.link_notes("X/C.psd", "c1", links, sha, alike, record) == [
        "Links gone.png, which is not in the repo: push that file.",
        "Links b.png, which looks the same as a.png: link a.png instead.",
        "Fire links fire.png, changed since: Card sync could not draw it again (a warp); update it in Photoshop."]
    assert cc.link_notes("X/C.psd", "c0", links, sha, alike, record) == []
    tree["MDW/Linked/gone.png"] = "g1"
    assert len(cc.link_notes("X/C.psd", "c1", links, sha, alike, record)) == 2


def test_the_mock_up_reads_the_stat_notes(tmp_path, monkeypatch):
    """The stat notes as TTSCardMaker's mockup.js handStats reads them (those stats aren't drawn); old bars name none."""
    mockup = SITE_SRC / "site" / "assets" / "mockup.js"
    if not mockup.is_file():
        pytest.skip(f"no {mockup}")
    monkeypatch.setattr(cc.psd_edit, "stat_caps", lambda psd: None)
    monkeypatch.setattr(cc, "stat_rows", lambda psd: dict.fromkeys(cc.STAT_ORDER, [([], [], None)]))
    monkeypatch.setattr(cc, "_real_bake", lambda ctx, rel, fields, png: ("refused", "unsupported-card: 'Health': x", None))
    notes = [*cc._stat_notes(None, "C.psd", None, {"card": dict.fromkeys(cc.STAT_ORDER, 3)}, "few", None),
             cc.NO_CHANGE_NOTE.format("Stat", "raises"), cc.NO_CHANGE_NOTE.format("HP and DEF", "edits"),
             cc.OLD_NOTE + " " + cc.SWAP_NOTE.format("ATK and SPD"), cc.BRIGHT_NOTE, cc.OLD_NOTE]
    script = tmp_path / "probe.mjs"
    script.write_text(f"const m = await import({json.dumps(mockup.as_uri())});\nconsole.log(JSON.stringify("
                      f"{json.dumps(notes)}.map(hand => m.handStats({{ checks: {{ hand }} }}))));\n", encoding="utf-8")
    done = subprocess.run(["node", str(script)], capture_output=True, text=True)
    every = ["hp", "def", "atk", "spd"]
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [every, every, ["hp", "def"], ["atk", "spd"], every, []]


ABIGAIL = ("69876f29527feae72106a91cf0aba6e59197851a", "0e0cf53cfd524eee909c05e1d729097d4bde8212")   # before / after 11-13
HALBERD = "ab6eb4738ce468e4bb1ba0700703899415c0e1cc"     # Halberd Reactor: DEF at 10 over a plain dark
OWN_LOOKS = ("193ca223f27b71fe410e702c0ef9c1023fa1eb2f",     # King Bob-omb: a silver ATK row among orange
             "e8655fcc6e586b11a6dffce5e58e8b2499260d07",     # Massive Moss Charger: a split ATK row
             "a61ae7a8575d4548cd9adb5a8ab9e4c417041ac4",     # Pizza: a colour per row, the empties shared
             "93c385b61bfd868aabfde9ae5903aadcc5802e07",     # Frog: a texture per row
             "2746abaa00679647b385787c4bb4e384b135ad87")     # Baba: word bars printing 'IS 6'


def test_hidden_bar_looks_are_flagged():
    """A bar's hidden cells a stat edit bakes unlike the card's look (cardissues 11-13): Abigail's DEF before its repaint
    (plain white under the darks), not after; Halberd Reactor's DEF at 10 (a plain dark). A card maker's own looks
    flag at no value of any bar (cardfix2's sweep). A row of its own colour, a texture across every bar and one stopping
    at the value (PoD05 Lead Bloon) read as they are."""
    from io import BytesIO
    from psd_tools import PSDImage
    from conftest import CLONE, card_blob
    psd = lambda sha: (PSDImage.open(BytesIO(raw := card_blob(CLONE, sha))), raw)
    assert cc.look_notes(*psd(ABIGAIL[0])) == [cc.LIT_LOOK_NOTE.format("DEF bar's")]
    assert cc.look_notes(*psd(ABIGAIL[1])) == []
    assert cc.look_notes(*psd(HALBERD)) == [cc.DARK_LOOK_NOTE.format("DEF bar's")]
    for sha in OWN_LOOKS:
        rows = cc.bar_looks(*psd(sha))
        assert [(n, v) for n, (label, _, lit, dark) in enumerate(rows) for v in range(11)
                if cc.unlike_looks([*rows[:n], (label, v, lit, dark), *rows[n + 1:]])] == [], sha
    flat = lambda rgb: np.tile(np.asarray(rgb, float), (10, 3, 1))
    pink, red, grey = flat((250, 210, 215)), flat((95, 30, 45)), flat((80, 70, 70))
    row = lambda label, value, lit, dark=grey: (label, value, lit, dark)
    assert cc.unlike_looks([row("HP", 5, pink), row("DEF", 3, pink), row("ATK", 7, flat((130, 150, 175))),
                            row("SPD", 4, pink)]) == []                                   # ATK: its own colour
    stops = np.concatenate([pink[:4], red[4:]])                                            # a look stopping at 4
    assert cc.unlike_looks([row("HP", 4, stops), row("DEF", 4, stops), row("ATK", 4, stops), row("SPD", 4, stops)]) == \
        [(k, "lit") for k in ("HP", "DEF", "ATK", "SPD")]
    texture = np.random.default_rng(3).uniform(40, 220, (10, 3, 3))
    assert cc.unlike_looks([row(k, v, texture) for k, v in (("HP", 2), ("DEF", 6), ("ATK", 4), ("SPD", 9))]) == []
