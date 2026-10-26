"""
psd_edit's stars and types bakes (owner, 2026-10-23: teach the bake instead of refusing the card): the layouts the
planners know (a count's own folder, hidden stars, rows of two, a line; a type row that closes up, icons from the card's
own hidden layers or Types/), their per-field refusals, and the gates (records, read-back, nothing changed outside).
Fixture cards are built here with psd-tools (no clone needed); the real-card tests copy cards from the clone and skip
without it. Bakes need the fork's layer copy API (Layer.duplicate, PixelLayer.fromarray):
    ~/.venvs/ac-py314-fork/bin/python -m pytest -q tests/test_stars_types.py
"""
import json
from math import cos, pi, sin
from pathlib import Path
from shutil import copy

import numpy as np
import pytest
from PIL import Image, ImageDraw
from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer

import commands.psd_edit as pe

VOCAB = Path(__file__).resolve().parent / "goldens" / "_vocab.json"
REL = "Auxiliary/Items/Fixture.psd"                     # an aux item: the parser counts its stars from the PSD
KRABBY = "Auxiliary/Items/Krabby_Patty.psd"             # a column of two, a third hidden under them
LOVE = "Auxiliary/Items/Love.psd"                       # rows of two, the third centred: 4 moves it aside
METTATON = "Auxiliary/Creatures/Mettaton_EX.psd"        # four stars in a zigzag: a fifth carries it on
SPRINGTRAP = "Auxiliary/Minions/Springtrap.psd"         # two stars over three: a sixth lays them out again as rows of two
CLAW = "N.M.E/The_Claw.psd"                             # one star over a pair: a fourth makes two rows of two
FREEZER = "Field/1 Stars/The_Freezer.psd"               # a field card: its "1 Stars" folder sets the stars
MORPHO = "Auxiliary/Creatures/Morpho_Knight.psd"        # Fire over Undead: dropping Fire moves Undead up
MAGNUM = "Auxiliary/Items/Magnum_Opus.psd"              # its field type is a frame round the card
ABIGAIL = "Auxiliary/Minions/Abigail.psd"               # a row of two icons running into the title: a third goes under

icons = pytest.mark.skipif(not pe.icon_engine(), reason="the fork's layer copy API (Layer.duplicate, PixelLayer.fromarray)")


# ========================
# Fixture cards
# ========================
@pytest.fixture(scope="module")
def parser():
    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
    return pe.offline_parser(vocab["all_types"], vocab["stars"], vocab["manifest"])


def _star(size: int = 100) -> Image.Image:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    points = [(size / 2 + size / 2 * (1 if k % 2 == 0 else 0.45) * cos(-pi / 2 + k * pi / 5),
               size / 2 + size / 2 * (1 if k % 2 == 0 else 0.45) * sin(-pi / 2 + k * pi / 5)) for k in range(10)]
    ImageDraw.Draw(image).polygon(points, fill=(250, 200, 30, 255), outline=(0, 0, 0, 255))
    return image


def _icon(colour: tuple, size: tuple = (120, 140)) -> Image.Image:
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(image).ellipse((4, 4, size[0] - 4, size[1] - 4), fill=colour + (255,))
    return image


def _types_dir(tmp_path) -> str:
    """A Types/ folder of plain icons (what the card repo's are, as far as the bake cares)."""
    folder = tmp_path / "Types"
    folder.mkdir(exist_ok=True)
    for name, colour in (("fire", (230, 80, 20)), ("water", (30, 90, 230)), ("ice", (150, 220, 250)),
                         ("Fragile Equip", (120, 120, 120))):
        _icon(colour, (200, 160)).save(folder / f"{name}.png")
    return str(folder)


def _fixture(tmp_path, stars=((1080, 20), (970, 20)), hidden=(), folders=None, types=(("Fire", (30, 30)),),
             library=(), name="Fixture") -> tuple:
    """
    A 1200x1800 card: a gradient art layer, a Stars group (stars at the given top-lefts, hidden ones, or folders
    {"N stars": [top-lefts]} with the first shown), a Types group of icons ((name, top-left)) and hidden library icons.

    Returns:
        (PSD path, PNG path): the PNG is the PSD's own render (the official export)
    """
    psd = PSDImage.new("RGB", (1200, 1800), color=(120, 90, 60))
    art = np.zeros((1800, 1200, 3), np.uint8)
    art[..., 0] = np.linspace(60, 200, 1200)[None, :].astype(np.uint8)
    art[..., 1] = np.linspace(40, 160, 1800)[:, None].astype(np.uint8)
    art[..., 2] = 90
    psd.append(PixelLayer.fromarray(art, psd, name="Art"))
    group = psd.create_group(name="Stars")
    if folders:
        for k, (folder, spots) in enumerate(folders.items()):
            inner = group.create_group(name=folder)
            for x, y in spots:
                inner.append(PixelLayer.fromarray(np.asarray(_star()), psd, name="Star", top=y, left=x))
            inner.visible = k == 0
    for x, y in stars:
        group.append(PixelLayer.fromarray(np.asarray(_star()), psd, name="Star", top=y, left=x))
    for x, y in hidden:
        layer = PixelLayer.fromarray(np.asarray(_star()), psd, name="Star", top=y, left=x)
        group.append(layer)
        layer.visible = False
    row = psd.create_group(name="Types")
    for icon, (x, y) in types:
        row.append(PixelLayer.fromarray(np.asarray(_icon((200, 40 + 30 * len(icon), 60))), psd, name=icon, top=y, left=x))
    for icon, (x, y) in library:
        layer = PixelLayer.fromarray(np.asarray(_icon((40, 200, 200))), psd, name=icon, top=y, left=x)
        row.append(layer)
        layer.visible = False
    path = tmp_path / "old" / f"{name}.psd"
    path.parent.mkdir(exist_ok=True)
    psd.save(str(path))
    PSDImage.open(str(path)).composite(ignore_preview=True).save(str(path)[:-4] + ".png")
    return str(path), str(path)[:-4] + ".png"


def _bake(parser, paths, jobs, tmp_path, rel=REL, types_dir=None):
    psd, png = paths
    parsed = pe.parse_card(parser, psd, rel)
    classified = pe.classify(parsed, jobs, PSDImage.open(psd), parser)
    assert classified.jobs == jobs, classified.refusals
    out = tmp_path / "new"
    out.mkdir(exist_ok=True)
    return parsed, pe.apply_jobs(psd, png, rel, jobs, parsed, parser, str(out), str(tmp_path / "no-fonts"),
                                 update_preview=False, types_dir=types_dir)


def _plan(parser, path, rel, field, value):
    parsed = pe.parse_card(parser, path, rel)
    psd = PSDImage.open(path)
    return pe.plan_stars(psd, parsed, value, parser) if field == "stars" else pe.plan_types(psd, parsed, value, parser)


def _clone_card(clone, rel, tmp_path) -> tuple:
    source, png = clone / rel, clone / (rel[:-4] + ".png")
    if not source.exists() or not png.exists():
        pytest.skip(f"{rel} not in the clone")
    (tmp_path / "old").mkdir(exist_ok=True)
    copy(source, tmp_path / "old" / source.name)
    copy(png, tmp_path / "old" / png.name)
    return str(tmp_path / "old" / source.name), str(tmp_path / "old" / png.name)


# ========================
# Pure functions
# ========================
def _unit(x, y, size=100):
    return pe._Unit([], (x, y, x + size, y + size))


def test_layouts_rows_of_two_and_lines():
    """Rows of two (an odd last one centred), a line, neither; a lone star is a pair's right one."""
    size = (100, 100)
    pair = [_unit(1080, 20), _unit(970, 20)]
    assert pe._pyramid(pair, size) == (1130, 1020, 70, 100, False)
    assert pe._pyramid_spots(pe._pyramid(pair, size), 3) == [(1130, 70), (1020, 70), (1075, 170)]
    assert pe._pyramid_spots(pe._pyramid(pair, size), 1) == [(1130, 70)]         # the right one stays
    triangle = pair + [_unit(1025, 125)]
    assert pe._pyramid(triangle, size) == (1130, 1020, 70, 105, False)
    claw = [_unit(1025, 20), _unit(1080, 125), _unit(970, 125)]                       # the odd one on top (The Claw)
    assert pe._pyramid(claw, size) == (1130, 1020, 70, 105, True)
    assert pe._pyramid_spots(pe._pyramid(claw, size), 4) == [(1130, 70), (1020, 70), (1130, 175), (1020, 175)]
    assert pe._pyramid_spots(pe._pyramid(claw, size), 5) == [(1075, 70), (1130, 175), (1020, 175), (1130, 280), (1020, 280)]
    zigzag2 = [_unit(1080, 20), _unit(970, 130), _unit(1080, 240)]                    # the bloons: right, left, right
    assert pe._zigzag(zigzag2, size) == ((1130, 70), (1020, 180), (0, 220))
    assert pe._pyramid([_unit(1080, 20), _unit(970, 60)], size) is None               # a pair askew: no row
    column = [_unit(1080, 20), _unit(1080, 160)]
    assert pe._pyramid(column, size) is None and pe._line(column, size, True) == ((1130, 70), (0, 140))
    ell = [_unit(1080, 20), _unit(970, 20), _unit(1080, 300)]
    assert pe._pyramid(ell, size) is None and pe._line(ell, size, True) is None and pe._zigzag(ell, size) is None
    lone = pe._Unit([SimpleLayer("Star", (1080, 20, 1190, 130))], (1085, 25, 1185, 125))
    assert pe._pyramid([lone], size) == (1135, 1025, 75, 110, False)
    assert [how for how, _ in pe._star_layouts([lone], size, 2)] == ["rows of two", "a column"]
    moves, free = pe._place(triangle, pe._pyramid_spots(pe._pyramid(triangle, size), 4), size)
    assert [(unit.core, dx, dy) for unit, dx, dy in moves] == [((1025, 125, 1125, 225), 55, 0)] and free == [(1020, 175)]


class SimpleLayer:
    """A layer's name and bbox, all the layout code reads."""
    def __init__(self, name, bbox):
        self.name, self.bbox = name, bbox


def test_record_moved_checks_rect_mask_and_blocks():
    a = {"name": "Star", "flags": b"\x08", "rect": (10, 20, 110, 120), "channels": ((-1, 9), (0, 9)), "mask": b"",
         "blend": (0, 255, 0), "blocks": ((b"lyid", b"1"), (b"SoLd", b"old"), (b"lfx2", b"glow"))}
    b = dict(a, rect=(60, 20, 160, 120), blocks=((b"lyid", b"1"), (b"SoLd", b"new"), (b"lfx2", b"glow")))
    assert pe._record_moved(a, b, 0, 50, pe.MOVE_BLOCKS) is None
    assert "rect" in pe._record_moved(a, b, 0, 40, pe.MOVE_BLOCKS)
    assert "lfx2" in pe._record_moved(a, dict(b, blocks=((b"lyid", b"1"), (b"SoLd", b"new"), (b"lfx2", b"x"))), 0, 50, pe.MOVE_BLOCKS)
    assert "lyid" in pe._record_moved(a, dict(b, blocks=((b"lyid", b"2"), (b"SoLd", b"new"), (b"lfx2", b"glow"))), 0, 50, pe.MOVE_BLOCKS)
    copied = dict(b, name="Star copy", blocks=((b"lyid", b"2"), (b"SoLd", b"new"), (b"lfx2", b"glow")))
    assert pe._record_moved(a, copied, 0, 50, pe.COPY_BLOCKS, renamed=True) is None
    assert pe._record_moved(a, copied, 0, 50, pe.COPY_BLOCKS) == "name"
    assert "flag" in pe._record_moved(a, dict(b, flags=b"\x0a"), 0, 50, pe.MOVE_BLOCKS)
    assert pe._record_moved(a, dict(b, flags=b"\x0a"), 0, 50, pe.MOVE_BLOCKS, flags=True) is None
    group = dict(a, rect=(0, 0, 0, 0))                                          # an empty rect stays where it is
    assert pe._record_moved(group, group, 5, 5, pe.MOVE_BLOCKS) is None
    masked = dict(a, mask_box=(10, 20, 110, 120, None, None, None, None), mask_norm=b"m")
    moved = dict(masked, rect=(60, 20, 160, 120), blocks=b["blocks"])
    assert pe._record_moved(masked, dict(moved, mask_box=(60, 20, 160, 120, None, None, None, None)), 0, 50, pe.MOVE_BLOCKS) is None
    assert pe._record_moved(masked, moved, 0, 50, pe.MOVE_BLOCKS) == "mask"


def test_gate_bytes_takes_declared_copies_moves_toggles_and_icons(monkeypatch):
    """A star copied (its record a moved copy of the source's), one moved, one hidden, a type icon made from Types/;
    anything undeclared refuses, with the field it belongs to."""
    def record(name, rect=(0, 0, 10, 10), flags=b"\x08", blocks=((b"lyid", b"1"),), channels=((-1, 9), (0, 9), (1, 9), (2, 9))):
        return {"name": name, "flags": flags, "rect": rect, "channels": channels, "mask": b"", "blend": (0, 255, 0),
                "blocks": blocks}
    old = [record("Art"), record("Star", (10, 900, 110, 1000)), record("Star", (10, 1000, 110, 1100)), record("Fire")]
    new = [record("Art"), record("Star", (10, 900, 110, 1000), flags=b"\x0a"),
           record("Star", (10, 1000, 110, 1100)), record("Star copy", (120, 950, 220, 1050), blocks=((b"lyid", b"9"),)),
           record("Fire", (40, 0, 50, 10)), record("water", blocks=((b"lfx2", b"glow"),))]
    files = {"old": old, "new": new}
    monkeypatch.setattr(pe, "layer_records", lambda path: [dict(r) for r in files[path]])
    stars = pe.Edit("stars", 2, 2, [], [], (0, 0, 1, 1), toggled=[1], copied=[(3, 2, -50, 110)])
    types = pe.Edit("types", ["fire"], ["fire", "water"], [], [], (0, 0, 1, 1), shifted=[(4, 3, 0, 40)],
                    made=[(5, {b"lfx2": b"glow"})])
    records, flag_bytes, block_bytes = pe.gate_bytes("old", "new", [stars, types])
    assert records == [(1, "Star", ["flags"]), (3, "Star copy", ["added"]), (4, "Fire", ["rect"]), (5, "water", ["added"])]
    assert (flag_bytes, block_bytes) == (1, 4)
    with pytest.raises(pe.Refused, match="no copy of 'Star' moved: rect") as refused:
        pe.gate_bytes("old", "new", [pe.Edit("stars", 2, 3, [], [], None, toggled=[1], copied=[(3, 2, -50, 100)]), types])
    assert refused.value.field == "stars"
    with pytest.raises(pe.Refused, match="moved layer 4 'Fire'") as refused:
        pe.gate_bytes("old", "new", [stars, pe.Edit("types", [], [], [], [], None, shifted=[(4, 3, 0, 30)],
                                                    made=[(5, {b"lfx2": b"glow"})])])
    assert refused.value.field == "types"
    with pytest.raises(pe.Refused, match="not the declared pixel layer"):
        pe.gate_bytes("old", "new", [stars, pe.Edit("types", [], [], [], [], None, shifted=[(4, 3, 0, 40)],
                                                    made=[(5, {b"lfx2": b"other"})])])
    with pytest.raises(pe.Refused, match="layer 1 'Star' changed flags"):          # a toggle nobody declared
        pe.gate_bytes("old", "new", [pe.Edit("stars", 2, 3, [], [], None, copied=[(3, 2, -50, 110)]), types])


def test_classify_icon_fields_per_field():
    """Equal stars and types (any order or spelling) are redundant; without a PSD a refusal names the field."""
    parsed = {"card": {"type": "minion", "stars": 2, "types": ["fire", "fragile equip"], "hp": 5}}
    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
    parser = pe.offline_parser(vocab["all_types"], vocab["stars"], vocab["manifest"])
    out = pe.classify(parsed, {"stars": 2, "types": ["Fragile_Equip", "Fire"], "hp": 6}, None, parser)
    assert out.redundant == ["stars", "types"] and out.jobs == {"hp": 6}
    out = pe.classify(parsed, {"stars": 3, "types": ["water"]}, None, parser)
    assert [(r.code, r.field) for r in out.refusals] == [("unsupported-card", "stars"), ("unsupported-card", "types")]
    assert out.refusals[0].why == "stars: no PSD to place them on"
    assert pe.same_types(parser, ["fire", "water"], ["Water", "FIRE"]) and not pe.same_types(parser, ["fire"], ["fire", "fire"])


# ========================
# Fixture cards (the fork)
# ========================
@icons
def test_stars_pair_to_three_copies_centred_below(parser, tmp_path):
    """Two stars in a row: the third is the last one copied, centred under the pair (the templates' '3 star')."""
    paths = _fixture(tmp_path)
    parsed, baked = _bake(parser, paths, {"stars": 3}, tmp_path)
    assert baked.reparse["card"]["stars"] == 3 and baked.icons == {"stars": "rows of two"}
    assert [fields for _, _, fields in baked.records] == [["added"]]
    saved = PSDImage.open(baked.psd_path)
    stars = [layer for layer in saved.descendants() if layer.name.startswith("Star") and not layer.is_group()]
    assert [(layer.name, tuple(layer.bbox)) for layer in stars] == [   # a row below: the stars' core height (91) down
        ("Star", (1080, 20, 1180, 120)), ("Star", (970, 20, 1070, 120)), ("Star copy", (1025, 111, 1125, 211))]
    png = np.asarray(Image.open(baked.png_path).convert("RGB"))
    assert tuple(png[161, 1075]) == (250, 200, 30)                     # the new star's middle, in the merged PNG
    old = np.asarray(Image.open(paths[1]).convert("RGB"))
    outside = np.ones(old.shape[:2], bool)
    x0, y0, x1, y1 = baked.region
    outside[y0:y1, x0:x1] = False
    assert (old[outside] == png[outside]).all()


@icons
def test_stars_fewer_hide_the_last_and_more_show_hidden_first(parser, tmp_path):
    paths = _fixture(tmp_path, stars=((1080, 20), (970, 20), (1025, 125)), hidden=((1080, 230),))
    plan = _plan(parser, paths[0], REL, "stars", 2)
    assert [tuple(layer.bbox) for layer in plan.hide] == [(1025, 125, 1125, 225)] and not plan.moves
    plan = _plan(parser, paths[0], REL, "stars", 4)                    # the hidden star first, then the layout
    assert [tuple(layer.bbox) for layer in plan.show] == [(1080, 230, 1180, 330)] and plan.copies == []
    on = _fixture(tmp_path, stars=((1080, 20), (970, 20)), hidden=((1075, 25),), name="Spare")
    plan = _plan(parser, on[0], REL, "stars", 3)                       # a spare on a shown star stays hidden (R5)
    assert not plan.show and len(plan.copies) == 1
    _, baked = _bake(parser, paths, {"stars": 1}, tmp_path)
    assert baked.reparse["card"]["stars"] == 1 and baked.flag_bytes == 2      # two stars hidden: one flag byte each
    hidden = [tuple(layer.bbox) for layer in PSDImage.open(baked.psd_path).descendants()
              if layer.name == "Star" and layer.is_visible()]
    assert hidden == [(1080, 20, 1180, 120)]                           # the pair's right one stays


@icons
def test_stars_refuse_per_field(parser, tmp_path):
    paths = _fixture(tmp_path, stars=((1080, 20), (970, 20), (1080, 300)))      # an L: its short row grows (layout support)
    plan = _plan(parser, paths[0], REL, "stars", 4)
    assert plan.how == "its rows carried on" and [(dx, dy) for _, dx, dy, _ in plan.copies] == [(-110, 0)]
    assert _plan(parser, paths[0], REL, "stars", 2).hide                # fewer still bakes: hide the last
    with pytest.raises(pe.Refused, match="at least one"):
        _plan(parser, paths[0], REL, "stars", 0)
    with pytest.raises(pe.Refused, match=r"the parser counts 5, the Stars group draws 3$"):    # no header text blamed
        pe.plan_stars(PSDImage.open(paths[0]), {"card": {"stars": 5, "path": REL}, "info": None}, 4, parser)
    with pytest.raises(pe.Refused, match="the '1 Stars' folder sets them"):
        _plan(parser, paths[0], "Items/Equip/1 Stars/Fixture.psd", "stars", 2)
    assert pe.star_target("Auxiliary/Status Effects/Debuffs/2 Stars/Afraid.psd", 3) == "Auxiliary/Status Effects/Debuffs/3 Stars"
    assert pe.star_folder(REL) is None and pe.star_folder("Auxiliary/Status Effects/Buffs/Fixture.psd") is None
    with pytest.raises(pe.Refused, match="the parser reads no stars on a buff card"):
        _plan(parser, paths[0], "Auxiliary/Status Effects/Buffs/Fixture.psd", "stars", 2)
    low = _fixture(tmp_path, stars=((1095, 20), (1095, 1650)), name="Low")      # a star below the middle: no count reads
    with pytest.raises(pe.Refused, match="draws 1 more below its middle"):      # it, so no layout redraws it (refused)
        _plan(parser, low[0], REL, "stars", 2)


@icons
def test_gate_outside_refuses_a_change_beyond_the_region(tmp_path):
    paths = _fixture(tmp_path)
    old, new = PSDImage.open(paths[0]), PSDImage.open(paths[0])
    before = pe.render_all(old)
    new.append(PixelLayer.fromarray(np.asarray(_star()), new, name="Stray", top=900, left=500))
    with pytest.raises(pe.Refused, match="stars: the card changes") as refused:
        pe.gate_outside(before, new, pe.Edit("stars", 2, 3, [], [], (950, 0, 1200, 300)))
    assert refused.value.code == "region" and refused.value.field == "stars"
    after = pe.gate_outside(before, new, pe.Edit("types", [], [], [], [], (480, 880, 620, 1020)))
    assert (after != before).any() and (after == pe.render_all(new)).all()


@icons
def test_a_new_icon_goes_above_the_clip_chain_and_takes_it(parser, clone, tmp_path):
    """Mini Doomer's Wind has a Curves clipped to it: a Fire beside it goes above that Curves (else Wind would lose it
    to Fire, outside Fire's box: gate_outside's catch) and gets a copy clipped to itself."""
    rel = "Auxiliary/Minions/Mini_Doomer.psd"
    paths = _clone_card(clone, rel, tmp_path)
    _, baked = _bake(parser, paths, {"types": ["wind", "light", "defensive_minion", "fire"]}, tmp_path, rel,
                     str(clone / "Types"))
    row = [(layer.name, layer.clipping) for layer in next(layer for layer in PSDImage.open(baked.psd_path).descendants()
                                                         if layer.name == "fire").parent]
    assert row[:4] == [("wind", False), ("Curves 1 copy", True), ("fire", False), ("Curves 1 copy", True)]
    assert sorted(fields for _, _, fields in baked.records) == [["added"], ["added"]]


@icons
def test_types_add_from_types_png_and_close_up(parser, tmp_path):
    """Add Water under a lone Fire (made from Types/, fitted to Fire's box); drop the first of two and the row closes up."""
    types_dir = _types_dir(tmp_path)
    paths = _fixture(tmp_path, types=(("Fire", (30, 30)),))
    _, baked = _bake(parser, paths, {"types": ["fire", "water"]}, tmp_path, types_dir=types_dir)
    assert sorted(baked.reparse["card"]["types"]) == ["fire", "water"] and baked.icons == {"types": "1 made from Types/"}
    saved = PSDImage.open(baked.psd_path)
    water = next(layer for layer in saved.descendants() if layer.name == "water")
    fire = pe._core(next(layer for layer in saved.descendants() if layer.name == "Fire"))
    assert water.kind == "pixel" and water.parent.name == "Types"
    x0, y0, x1, y1 = water.bbox                                        # fitted into Fire's core, centred under it
    assert x1 - x0 == fire[2] - fire[0] and y1 - y0 < fire[3] - fire[1]
    assert abs((x0 + x1) - (fire[0] + fire[2])) <= 1 and abs((y0 + y1) / 2 - (fire[1] + fire[3]) / 2 - (fire[3] - fire[1] + 22)) <= 1
    two = _fixture(tmp_path, types=(("Fire", (30, 30)), ("Water", (30, 200))), name="Two")
    plan = _plan(parser, two[0], REL, "types", ["water"])
    assert [layer.name for layer in plan.hide] == ["Fire"] and [(layer.name, dx, dy) for layer, dx, dy in plan.moves] == [
        ("Water", 0, -170)]


@icons
def test_types_refuse_per_field(parser, tmp_path, monkeypatch):
    paths = _fixture(tmp_path)
    with pytest.raises(pe.Refused, match="'lava' is no type icon") as refused:
        _plan(parser, paths[0], REL, "types", ["lava"])
    assert refused.value.field == "types"
    bare = _fixture(tmp_path, types=(), name="Bare")
    with pytest.raises(pe.Refused, match="no type icon on the card to line a new one up with"):
        _plan(parser, bare[0], REL, "types", ["fire"])
    monkeypatch.setattr(pe, "type_row", lambda psd, parser, *_: ([], [SimpleLayer("Types", (0, 0, 1, 1))]))
    with pytest.raises(pe.Refused, match="printed as icon tokens in 'Types'"):
        _plan(parser, paths[0], REL, "types", ["fire"])


# ========================
# Real cards (copies from the clone)
# ========================
@icons
def test_real_star_layouts(clone, parser, tmp_path):
    """Krabby Patty shows its own hidden third star; Love's centred third moves aside for a fourth, The Claw's top one
    too; Mettaton's zigzag takes a fifth; Springtrap's two over three hides one and takes a sixth as rows of two; a field
    card's folder sets its stars."""
    krabby = _plan(parser, _clone_card(clone, KRABBY, tmp_path)[0], KRABBY, "stars", 3)
    assert [layer.name for layer in krabby.show] == ["Patrick Star copy 2"] and not krabby.copies
    love = _plan(parser, _clone_card(clone, LOVE, tmp_path)[0], LOVE, "stars", 4)
    assert [(layer.name, dx, dy) for layer, dx, dy in love.moves] == [("Starstorm copy 2", 43, 0)]
    assert [(unit.layers[0].name, dx, dy, name) for unit, dx, dy, name in love.copies] == [
        ("Starstorm copy 2", -46, 0, "Starstorm copy 3")]
    claw = _plan(parser, _clone_card(clone, CLAW, tmp_path)[0], CLAW, "stars", 4)
    assert claw.how == "rows of two" and [(layer.name, dx, dy) for layer, dx, dy in claw.moves] == [("Top", 54, 0)]
    mettaton = _plan(parser, _clone_card(clone, METTATON, tmp_path)[0], METTATON, "stars", 5)
    assert mettaton.how == "a zigzag" and [(dx, dy) for _, dx, dy, _ in mettaton.copies] == [(77, 90)]
    springtrap = _clone_card(clone, SPRINGTRAP, tmp_path)[0]
    assert len(_plan(parser, springtrap, SPRINGTRAP, "stars", 4).hide) == 1
    six = _plan(parser, springtrap, SPRINGTRAP, "stars", 6)        # no room to grow its top row: laid out again, told
    assert six.how == "rows of two, laid out again" and len(six.copies) == 1 and "does not go on to 6" in six.notes[0]
    with pytest.raises(pe.Refused, match="the '1 Stars' folder sets them"):
        _plan(parser, _clone_card(clone, FREEZER, tmp_path)[0], FREEZER, "stars", 2)


LAYOUT_STARS = [   # LAYOUT_SUPPORT.md's star rules, one card each: (card, stars, how, notes)
    ("Auxiliary/Minions/Lorian.psd", 6, "its rows carried on", 0),             # 2 over 3: the top row grows
    ("Auxiliary/Minions/God_Tamer.psd", 6, "rows of two, laid out again", 1),  # a dice five: its six, told
    ("Auxiliary/Minions/Cat.psd", 2, "1 hidden", 0),                           # a group drawing two beside a group of one
    ("Auxiliary/Items/Nasty_Patty.psd", 4, "rows of two", 0),                  # no Stars group: Patrick Stars in a group
    ("Auxiliary/Minions/Paint_Warrior.psd", 3, "rows of two", 0),              # two Stars groups, one star each
]


@icons
@pytest.mark.parametrize("rel, value, how, notes", LAYOUT_STARS)
def test_layout_star_rules(clone, parser, tmp_path, rel, value, how, notes):
    plan = _plan(parser, _clone_card(clone, rel, tmp_path)[0], rel, "stars", value)
    assert plan.how == how and len(plan.notes) == notes


STAR_COUNTS = [   # the star counter (psd_parser.star_sets) on real cards: (card, stars, N-1 and N+1 plan or refusal)
    ("Creatures/FTL/2 Stars/Zoltan.psd", 2, None),                         # two stars overlapping by half
    ("Creatures/Kirby/5 Stars/Nightmare.psd", 5, None),                    # a dice face: the centre's core on a corner's
    ("Creatures/Other/5 Stars/Kyle.psd", 5, None),                         # a canvas-wide Color Fill clipped to each star
    ("Creatures/Fnaf/3 Stars/Toy_Chica.psd", 3, None),                     # a LINE below the middle in its stars group
    ("Creatures/Isaac/4 Stars/Tainted_Bethany.psd", 4, None),              # glows spanning two rows join their stars
    ("Creatures/Other/2 Stars/Auto_Croissant.psd", 2, None),               # the cat's art in the Stars group
    ("Creatures/Other/3 Stars/Grimora.psd", 3, None),                      # one layer, three stars: split (R4)
    ("Auxiliary/Minions/Living_Bread.psd", 2, None),                       # each star drawn twice (R3)
    ("Creatures/Hollow Knight/4 Stars/Nosk.psd", 4, None),                 # outside the group (R2; MMC's 5th: refused)
    ("Creatures/Isaac/3 Stars/Tainted_Lazarus.psd", 3, None),              # Dead Laz upside down: both sets
]


@icons
def test_star_counter_on_real_cards(clone, parser, tmp_path):
    """The parser and the bake count one way: N on each card; plan_stars at N-1 and N+1 plans (one layer drawing
    several split, stacked copies one star, stars outside the group laid out with it, a second creature's turned)."""
    for rel, stars, refused in STAR_COUNTS:
        path = _clone_card(clone, rel, tmp_path)[0]
        psd = PSDImage.open(path)
        parsed = pe.parse_card(parser, path, rel, psd)
        assert parser._count_stars(psd) == stars == pe._stars_total(psd, types=parser.all_types, stars=parser.star_names()), rel
        for value in (stars - 1, stars + 1):
            if refused is None:
                assert pe.plan_stars(psd, parsed, value, parser, moving=True).how, (rel, value)
            else:
                with pytest.raises(pe.Refused, match=refused):
                    pe.plan_stars(psd, parsed, value, parser, moving=True)
    staff = "N.M.E/Staff_of_Purity.psd"          # a column of 5 would cross the middle, where no count reads a star
    assert _plan(parser, _clone_card(clone, staff, tmp_path)[0], staff, "stars", 5).how == "its rows carried on"


@icons
def test_star_layout_fixes_on_real_cards(clone, parser, tmp_path):
    """A dice 5 at 4 loses its middle (Nightmare, Kyle's 2 over 3); a moved star under a shadow is copied on top
    instead (Flower of Decay); a new star off the banner the stars sit on refuses (Remote Detonator); the website's
    header lays out the stars outside the group as the bake does (Massive Moss Charger), whose fifth would sit under the
    moss (refused)."""
    import commands.psd_details as pd
    for rel, gone in (("Creatures/Kirby/5 Stars/Nightmare.psd", "Warpstar copy 4"), ("Creatures/Other/5 Stars/Kyle.psd", "middle")):
        plan = pe.plan_stars(*_open(parser, clone, rel, tmp_path), 4, parser, moving=True)
        assert [layer.name for layer in plan.hide] == [gone] and not plan.moves, rel
    rel = "Items/Infliction/3 Stars/Flower_Of_Decay.psd"
    plan = pe.plan_stars(*_open(parser, clone, rel, tmp_path), 4, parser, moving=True)
    assert plan.how == "rows of two" and [layer.name for layer in plan.hide] == ["star"] and not plan.moves
    assert len(plan.copies) == 2
    rel = "N.M.E/Remote_Detonator.psd"
    with pytest.raises(pe.Refused, match="would leave 'Layer 13' under the stars"):
        pe.plan_stars(*_open(parser, clone, rel, tmp_path), 3, parser)
    rel = "Auxiliary/Minions/Massive_Moss_Charger.psd"
    psd = PSDImage.open(_clone_card(clone, rel, tmp_path)[0])
    header = pd._star_header(psd, rel, tuple(parser.all_types), parser)
    assert parser._count_stars(psd) == 4 and len(header["units"]) == 4 and len(header["at"]["5"]) == 5   # the two
    #   outside are stars of the column
    assert "would sit under 'Massive_Moss_Charger'" in header["refused"]["5"]


def _open(parser, clone, rel, tmp_path) -> tuple:
    path = _clone_card(clone, rel, tmp_path)[0]
    psd = PSDImage.open(path)
    return psd, pe.parse_card(parser, path, rel, psd)


@icons
def test_star_pieces_split_and_read_masks():
    """One layer drawing two stars is two pieces; a mask hiding one leaves one, and the core box is the shown star's."""
    psd = PSDImage.new("RGB", (1200, 1800))
    two = Image.new("RGBA", (230, 100), (0, 0, 0, 0))
    two.paste(_star(), (0, 0)), two.paste(_star(), (130, 0))
    layer = PixelLayer.fromarray(np.asarray(two), psd, name="Stars art", top=20, left=900)
    psd.append(layer)
    pieces = sorted(pe.star_pieces(layer))
    assert len(pieces) == 2 and pieces[0][2] < pieces[1][0]
    mask = Image.new("L", (230, 100), 0)
    mask.paste(255, (0, 0, 115, 100))
    layer.create_mask(mask)
    assert len(pe.star_pieces(layer)) == 1 and pe.PSDParser._core_box(layer)[2] <= 1015


@icons
def test_type_icons_in_stars_are_no_stars(clone, parser, tmp_path):
    """Toy Freddy's stars group holds its Machine icon beside 3 stars: a move redraws them (a fourth copied); Pike Beam's
    Offensive_Minion icon in Stars reads 2 stars, not 3, and taking it off moves Machine into its spot (it blocked
    that as a star), the stars unchanged; Dark Matter's two icons in Stars likewise."""
    freddy = "Creatures/Fnaf/3 Stars/Toy_Freddy.psd"
    path = _clone_card(clone, freddy, tmp_path)[0]
    plan = pe.plan_stars(PSDImage.open(path), pe.parse_card(parser, path, freddy), 4, parser, moving=True)
    assert len(plan.copies) == 1 and not plan.hide
    pike = "Auxiliary/Minions/Pike_Beam.psd"
    paths = _clone_card(clone, pike, tmp_path)
    assert pe.parse_card(parser, paths[0], pike)["card"]["stars"] == 2
    _, baked = _bake(parser, paths, {"types": ["machine"]}, tmp_path, pike)
    assert baked.reparse["card"]["types"] == ["machine"] and baked.reparse["card"]["stars"] == 2
    dark = "Creatures/Kirby/3 Stars/Dark_matter.psd"           # its PNG is Dark_Matter.png: the PSD alone
    plan = _plan(parser, str(clone / dark), dark, "types", ["darkmatter"])
    assert [layer.name for layer, _, _ in plan.moves] == ["darkmatter"] and [layer.name for layer in plan.hide] == ["electric"]


@icons
def test_copied_smart_stars_bake_alike(clone, parser, tmp_path):
    """Mini Doomer's Warpstars are smart objects: two bakes of a third copy one with the same placed ID (bake_pictures
    compares each picture's layers), one the card doesn't hold yet."""
    rel = "Auxiliary/Minions/Mini_Doomer.psd"
    path = _clone_card(clone, rel, tmp_path)[0]
    records = []
    for k in range(2):
        psd = PSDImage.open(path)
        plan = pe.plan_stars(psd, pe.parse_card(parser, path, rel), 3, parser)
        pe.set_icons(psd, plan)
        psd.save(str(tmp_path / f"{k}.psd"))
        records.append(pe.layer_records(str(tmp_path / f"{k}.psd")))
        ids = [str(data[b"placed"].value) for layer in psd.descendants() for data in pe._placed(layer)[:1]]
        assert len(plan.copies) == 1 and len(ids) == len(set(ids))
    assert records[0] == records[1]


@icons
def test_a_first_type_icon_on_the_title_bakes_flagged(clone, parser, tmp_path):
    """John Ward has no type icon; Creature_Template's spot touches its Name: the icon is made there, told."""
    rel = "Creatures/Other/3 Stars/John_Ward.psd"
    path = _clone_card(clone, rel, tmp_path)[0]
    template = clone / "MDW/Templates/Creature_Template.psd"
    psd = PSDImage.open(path)
    spot = pe.template_icon(PSDImage.open(str(template)), psd, parser, "Creature_Template")
    plan = pe.plan_types(psd, pe.parse_card(parser, path, rel), ["fire"], parser, spot)
    edit = pe.set_icons(psd, plan, _types_dir(tmp_path))
    assert any("at Creature_Template's spot touches 'Name'" in note for note in edit.notes), edit.notes


@icons
def test_real_type_rows(clone, parser, tmp_path):
    morpho = _plan(parser, _clone_card(clone, MORPHO, tmp_path)[0], MORPHO, "types", ["undead"])
    assert [layer.name for layer in morpho.hide] == ["Fire"] and [layer.name for layer, _, _ in morpho.moves] == ["Undead", "Gradient Map 2"]
    with pytest.raises(pe.Refused, match="'field' is a frame round the card, not an icon to take off"):
        _plan(parser, _clone_card(clone, MAGNUM, tmp_path)[0], MAGNUM, "types", ["active"])
    freezer = _clone_card(clone, FREEZER, tmp_path)
    _, baked = _bake(parser, freezer, {"types": ["field", "ice"]}, tmp_path, FREEZER, str(clone / "Types"))
    assert sorted(baked.reparse["card"]["types"]) == ["field", "ice"]
    abigail = _plan(parser, _clone_card(clone, ABIGAIL, tmp_path)[0], ABIGAIL, "types", ["undead", "defensive_minion", "fire"])
    (stem, box, reference), = abigail.icons
    shield = next(layer for layer in PSDImage.open(_clone_card(clone, ABIGAIL, tmp_path)[0]).descendants()
                  if layer.name == "Defensive_Minion")
    core = pe._core(shield)
    assert stem == "fire" and box[0] == core[0] and box[1] > core[3]       # under the row's first icon, not by the title



@icons
def test_lane3_star_layouts(clone, parser, tmp_path):
    """Grimora's one layer drawing three is split into masked copies and its centred row stays centred (a bake: the
    split's records pass gate_bytes); Tainted Lazarus's dead form takes the new count too, turned half round;
    Polevaulter Zombie's skewed 2 over 1 goes to rows of two again (owner 2026-10-26)."""
    rel = "Creatures/Other/3 Stars/Grimora.psd"
    paths = _clone_card(clone, rel, tmp_path)
    parsed = pe.parse_card(parser, paths[0], rel)
    out = tmp_path / "new"
    out.mkdir(exist_ok=True)
    baked = pe.apply_jobs(*paths, "Creatures/Other/2 Stars/Grimora.psd", {"stars": 2}, parsed, parser, str(out),
                          str(tmp_path / "no-fonts"), update_preview=False)
    assert baked.reparse["card"]["stars"] == 2 and baked.icons["stars"] == "a centred row"
    saved = PSDImage.open(baked.psd_path)
    shown = sorted(pe.PSDParser._core_box(layer) for layer in saved.descendants()
                   if layer.name.startswith("Layer 4") and layer.is_visible())
    assert len(shown) == 2 and all(layer.has_mask() for layer in saved.descendants() if layer.name.startswith("Layer 4"))
    assert abs((shown[0][0] + shown[1][2]) / 2 - saved.width / 2) <= 2
    rel = "Creatures/Isaac/3 Stars/Tainted_Lazarus.psd"
    psd, parsed = _open(parser, clone, rel, tmp_path)
    plan = pe.plan_stars(psd, parsed, 4, parser, moving=True)
    assert plan.how == "rows of two; below, turned: rows of two" and len(plan.copies) == 2
    rel = "Auxiliary/Minions/Polevaulter_Zombie.psd"
    assert pe._skewed_pairs([(980, 68), (1094, 57), (1051, 151), (937, 151)], 99, 13)
    plan = pe.plan_stars(*_open(parser, clone, rel, tmp_path), 4, parser)    # its stars carry smart filters: S1
    assert plan.how == "rows of two, laid out again"


@icons
def test_smart_filtered_stars_move_as_photoshop_does(clone, parser, tmp_path):
    """Magnet-shroom's star carries smart filters: its copy gets its own FEid item (the source's planes, its rectangle
    moved with the copy, named after the copy's placed ID, in file order) and is marked not valid at its new place, as
    Photoshop writes one; gate_filters refuses an FEid change no edit declares."""
    assert pe.filter_items(), "the fork lacks SmartObject.filter_item (cards-20261026)"
    rel = "N.M.E/Magnet-shroom.psd"
    paths = _clone_card(clone, rel, tmp_path)
    parsed, baked = _bake(parser, paths, {"stars": 2}, tmp_path, rel)
    assert baked.reparse["card"]["stars"] == 2
    old, new = PSDImage.open(paths[0]), PSDImage.open(baked.psd_path)
    filtered = [layer for layer in new.descendants() if layer.kind == "smartobject" and layer.smart_object.filters]
    assert [str(item.uuid) for item in pe._filter_block(new)] == [layer.smart_object.placed_id for layer in filtered]
    copy = next(layer for layer in filtered if layer.name == "Layer 1 copy")
    source = next(layer for layer in old.descendants() if layer.name == "Layer 1 copy 3")
    item, was = copy.smart_object.filter_item, source.smart_object.filter_item
    assert item.rectangle == pe._moved_feid(was.rectangle, 0, copy.top - source.top) and item.channels == was.channels
    assert not copy.smart_object.filters[b"validAtPosition"].value
    with pytest.raises(pe.Refused, match="no smart-filtered layer moved or copied"):
        pe.gate_filters(old, new, [pe.Edit("stars", 1, 2, [], [], None)])


@icons
def test_type_rows_carried_on_and_pulled_back(clone, parser, tmp_path):
    """Masher's 2 over 1 takes a fourth icon beside its third, at the top row's pitch (S2); removing Sword's first icon
    pulls the next back onto the card (S3), but not over another icon, kept or made (Nasus: refused)."""
    masher = "Auxiliary/Minions/Masher.psd"
    plan = _plan(parser, _clone_card(clone, masher, tmp_path)[0], masher, "types",
                 ["machine", "dark", "defensive_minion", "fire"])
    (stem, box, _), = plan.icons
    assert stem == "fire" and box[0] > 160 and box[1] > 160                  # beside 'dark', under the top row
    sword = "Auxiliary/Minions/Sword.psd"
    plan = _plan(parser, _clone_card(clone, sword, tmp_path)[0], sword, "types", ["undead"])
    (layer, dx, dy), = plan.moves
    assert layer.name == "Undead" and layer.left + dx == 0 and plan.hide
    nasus = "Creatures/Other/4 Stars/Nasus.psd"
    with pytest.raises(pe.Refused, match="pulled back onto it, would cover 'Light'"):
        _plan(parser, _clone_card(clone, nasus, tmp_path)[0], nasus, "types", ["undead", "light"])
    psd, parsed = _open(parser, clone, nasus, tmp_path)
    plan = pe.plan_types(psd, parsed, ["undead", "water"], parser)
    with pytest.raises(pe.Refused, match="would cover the 'water' icon"):
        pe.set_icons(psd, plan, str(clone / "Types"))
