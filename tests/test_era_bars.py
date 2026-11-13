"""era_bars on drawn cards: template, uniform, two-creature, item and raid rows, the template fallback, eras, the bar."""
import json

from PIL import Image, ImageDraw

import commands.era_bars as eb
from commands.card_checks import TEMPLATE_CELLS

LIT, EMPTY = (230, 150, 40), (40, 30, 60)
STATS = {"hp": 7, "def": 3, "atk": 10, "spd": 0}


def bars(draw, box, values, cells, rows, height, second=None):
    """Bar rows: lit cells from the left, black cell lines; second: the bottom halves' values (two creatures)."""
    x0, x1 = box
    edges = [x0 + (x1 - x0) * e / cells[-1] for e in cells]
    for (stat, y), v2 in zip(rows, [None] * 4 if second is None else [second[s] for s in eb.ORDER]):
        for k in range(len(cells) - 1):
            if v2 is None:
                draw.rectangle((edges[k], y, edges[k + 1], y + height), fill=LIT if k < values[stat] else EMPTY)
            else:
                mid = y + height // 2
                draw.rectangle((edges[k], y, edges[k + 1], mid), fill=LIT if k < values[stat] else EMPTY)
                draw.rectangle((edges[k], mid, edges[k + 1], y + height), fill=(60, 200, 230) if k < v2 else EMPTY)
            draw.line((edges[k], y, edges[k], y + height), fill=(0, 0, 0), width=3)


def label_lines(rows, height, size, x1=0.25):
    """OCR lines of the stat labels beside the rows (fractions)."""
    w, h = size
    names = {"hp": "Health", "def": "Defense", "atk": "Attack", "spd": "Speed"}
    return [[names[s], 1.0, 0.02, y / h, x1 - 0.02, height / h] for s, y in rows]


def template_card(values, second=None):
    img = Image.new("RGB", (1200, 2400), (90, 120, 160))
    rows = list(zip(eb.ORDER, eb.D_ROWS))
    bars(ImageDraw.Draw(img), (360, 1173), values, TEMPLATE_CELLS, rows, eb.D_H, second)
    return img, label_lines(rows, eb.D_H, img.size, 0.28)


def test_template_cells_follow_card_checks():
    assert eb.TEMPLATE_CELLS == TEMPLATE_CELLS


def test_template_card_with_and_without_labels():
    img, lines = template_card(STATS)
    got = eb.read_card(img, lines, era="D2")
    assert got["reader"] == "best" and {s: v["v"] for s, v in got["stats"].items()} == STATS
    assert all(v["conf"] > 0.5 for v in got["stats"].values())
    small = img.resize((400, 800), Image.LANCZOS)
    fallback = eb.read_card(small, [], era="D1")             # no labels read: the template's own rows
    assert fallback["reader"] == "template" and {s: v["v"] for s, v in fallback["stats"].items()} == STATS


def test_uniform_grid_and_two_creatures():
    img = Image.new("RGB", (400, 800), (250, 250, 250))
    rows = [(s, 400 + 45 * n) for n, s in enumerate(eb.ORDER)]
    values = {"hp": 6, "def": 4, "atk": 7, "spd": 5}
    bars(ImageDraw.Draw(img), (90, 390), values, eb.uniform(10), rows, 30)
    got = eb.read_card(img, label_lines(rows, 30, img.size, 0.21), era="A")
    assert got["reader"] == "uniform" and {s: v["v"] for s, v in got["stats"].items()} == values
    twin, lines = template_card(STATS, second={"hp": 2, "def": 4, "atk": 4, "spd": 6})
    got = eb.read_card(twin, lines, era="D2", title="TWIN PRINCES")
    assert got["two"] and {s: v["v"] for s, v in got["stats"].items()} == STATS
    assert {s: v["v"] for s, v in got["stats2"].items()} == {"hp": 2, "def": 4, "atk": 4, "spd": 6}
    low = [[t, c, x, y + 0.012, w, h] for t, c, x, y, w, h in lines]       # labels sit lower than the bars
    got = eb.read_card(twin, low, era="D2", title="TWIM EMPERORS")            # OCR slips
    assert got["two"] and {s: v["v"] for s, v in got["stats"].items()} == STATS
    assert {s: v["v"] for s, v in got["stats2"].items()} == {"hp": 2, "def": 4, "atk": 4, "spd": 6}
    assert eb.read_card(twin, lines, era="D2", title="", card="god-tamer").get("two")   # the matcher's card
    assert eb.group_of(eb.read_card(twin, lines, era="D2", title="PiX")) == "multi"     # three forms: never sure
    assert eb.flags("OMORI").get("busy") and eb.flags("Frog", "D2").get("busy")         # drawn / textured bars
    assert not eb.flags("Frog", "C") and not eb.flags("Shattered Omori")


def test_rows_ocr_missed_and_items():
    img, lines = template_card(STATS)
    got = eb.read_card(img, lines[:3], era="D2")             # SPEED not read: its row placed by the others
    assert got["stats"]["spd"]["row"] == "placed" and "hold" not in got["stats"]["spd"]     # the template agrees
    card = Image.new("RGB", (400, 800), (250, 250, 250))
    rows = [(s, 400 + 45 * n) for n, s in enumerate(eb.ORDER)]
    bars(ImageDraw.Draw(card), (90, 390), STATS, eb.uniform(10), rows, 30)
    got = eb.read_card(card, label_lines(rows, 30, card.size, 0.21)[:3], era="A")
    assert got["stats"]["spd"]["hold"] and "hold" not in got["stats"]["hp"]
    item = Image.new("RGB", (1200, 1800), (90, 120, 160))
    rows = [("hp", 1100), ("def", 1210)]
    bars(ImageDraw.Draw(item), (360, 1150), {"hp": 6, "def": 2}, eb.uniform(10), rows, 80)
    lines = label_lines(rows, 80, item.size, 0.28) + [["+2 HP", 1.0, 0.4, 0.8, 0.2, 0.03]]
    look = {"w": 1200, "h": 1800, "ts": "2023-06-01", "frame_white": 0.0, "low_white": 0.0, "low_black": 0.1}
    got = eb.read_card(item, lines, look)
    assert got["era"] == "D-item" and got["labels"] == 2
    assert {s: v["v"] for s, v in got["stats"].items()} == {"hp": 6, "def": 2, "atk": None, "spd": None}
    assert got["stats"]["atk"]["row"] == "none"
    one = {"hp": {"v": 1, "margin": 14.0}, "def": {"v": 0, "flat": True}, "atk": {"v": 10, "flat": True}}
    eb.flat_holds(one)                                       # one weak split row: one-colour rows can't be told
    assert one["def"]["hold"] and one["atk"]["hold"] and "hold" not in one["hp"]


def test_raid_rows_of_twenty():
    img = Image.new("RGB", (2400, 2400), (60, 20, 30))
    rows = list(zip(eb.ORDER, eb.RAID_ROWS))
    values = {"hp": 16, "def": 6, "atk": 10, "spd": 1}
    bars(ImageDraw.Draw(img), (451, 2317), values, eb.uniform(20), rows, eb.RAID_H)
    got = eb.read_card(img, label_lines(rows, eb.RAID_H, img.size, 0.17), era="R")
    assert got["cells"] == 20 and {s: v["v"] for s, v in got["stats"].items()} == values


def test_eras_and_the_bar():
    white = Image.new("RGB", (400, 610), (255, 255, 255))
    assert eb.era_of(eb.look(white, "2020-07-20"), None)[0] == "A-item"
    look = {"w": 1200, "h": 2400, "ts": "2023-01-01", "frame_white": 0.0, "low_white": 0.0, "low_black": 0.1}
    assert eb.era_of(look, (0.30, 0.975))[0] == "D2"
    assert eb.era_of({**look, "w": 400, "h": 800, "frame_white": 0.6}, (0.2, 0.97))[0] == "A"
    assert eb.era_of({**look, "w": 240, "h": 342, "low_black": 0.5}, None)[0] == "B-item"
    assert eb.era_of({**look, "w": 880, "h": 1130}, None, [["Quest: kill 1 creature", 1, 0, 0.6, 0.5, 0.02]])[0] == "F"
    points = [(0.9, True)] * 60 + [(0.5, True)] * 30 + [(0.5, False)] * 3 + [(0.2, False)] * 10
    assert eb.bar_for(points) == 0.9                          # 90/93 right from 0.5: under 98%
    assert eb.bar_for(points + [(0.5, True)] * 100) == 0.5
    assert eb.bar_for([(0.9, True)] * 5) is None              # too few to set a bar


def test_truth_eye_none_drops_the_psd(tmp_path):
    base = tmp_path.joinpath(*eb.OUT, "truth")
    base.mkdir(parents=True)
    base.joinpath("psd.json").write_text(json.dumps({"versions": {"b1": {"stats": {"hp": 3, "def": 2, "atk": 5, "spd": 5}},
                                                                     "b2": {"stats": {"hp": 1, "def": 1, "atk": 1, "spd": 1}}}}))
    base.joinpath("eye_x.json").write_text(json.dumps({"items": {"b1": {"stats": None}, "f1": {
        "stats": {"hp": 6, "def": 1, "atk": "none", "spd": "none"}}}}))
    truth = eb.load_truth(str(tmp_path))
    assert "b1" not in truth and truth["b2"]["from"] == "psd"  # three forms / upside down: no truth at all
    row = {"file": "f1", "era": "D-item", "stats": {"hp": {"v": 6, "conf": 1.0}, "def": {"v": 1, "conf": 1.0},
                                                    "atk": {"v": None}, "spd": {"v": 4, "conf": 0.9}}}
    assert [(p[2], p[4]) for p in eb.scored([row], truth)] == [("hp", True), ("def", True), ("spd", False)]
