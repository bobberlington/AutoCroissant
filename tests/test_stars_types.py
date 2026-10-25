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
    source = clone / rel
    if not source.exists() or not Path(str(source)[:-4] + ".png").exists():
        pytest.skip(f"{rel} not in the clone")
    (tmp_path / "old").mkdir(exist_ok=True)
    psd, png = tmp_path / "old" / source.name, tmp_path / "old" / (source.stem + ".png")
    copy(source, psd)
    copy(str(source)[:-4] + ".png", png)
    return str(psd), str(png)


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
    with pytest.raises(pe.Refused, match="the '1 Stars' folder sets them"):
        _plan(parser, paths[0], "Items/Equip/1 Stars/Fixture.psd", "stars", 2)
    with pytest.raises(pe.Refused, match="the parser reads no stars on a buff card"):
        _plan(parser, paths[0], "Auxiliary/Status Effects/Buffs/Fixture.psd", "stars", 2)
    edge = _fixture(tmp_path, stars=((1095, 20), (1095, 1650)), name="Edge")     # a column whose next star is off the card:
    assert _plan(parser, edge[0], REL, "stars", 3).how == "its rows carried on"     # its last star takes one beside it


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
def test_a_new_icon_goes_above_the_clip_chain_and_takes_it(parser, tmp_path):
    """Mini Doomer's Wind has a Curves clipped to it: a Fire beside it goes above that Curves (else Wind would lose it
    to Fire, outside Fire's box: gate_outside's catch) and gets a copy clipped to itself."""
    from conftest import CLONE
    rel = "Auxiliary/Minions/Mini_Doomer.psd"
    paths = _clone_card(CLONE, rel, tmp_path) if (CLONE / rel).exists() else pytest.skip("no clone")
    _, baked = _bake(parser, paths, {"types": ["wind", "light", "defensive_minion", "fire"]}, tmp_path, rel,
                     str(CLONE / "Types"))
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


@icons
def test_stacked_stars_refuse(clone, parser, tmp_path):
    """Living Bread draws its two stars twice, stacked: the parser counts four, so no edit can show what it reads."""
    rel = "Auxiliary/Minions/Living_Bread.psd"
    with pytest.raises(pe.Refused, match="sits on another star"):
        _plan(parser, _clone_card(clone, rel, tmp_path)[0], rel, "stars", 5)


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

