"""
The mock-up's plates (commands/card_plates.py): a real card drawn without its text layers, each crop the PNG's backdrop
off the layer's ink (an FX layer's word in a second crop) with the layer's own lines and letters, and the plates branch:
added, kept, dropped with the PSDs.
"""
import base64
import dataclasses
import io
import json
import subprocess

import numpy as np
from PIL import Image, ImageFilter
from psd_tools import PSDImage

import commands.card_plates as cp
import commands.psd_edit as psd_edit
from commands.psd_edit import card_parts, _bare_parser
from helpers import parser  # noqa: F401 (fixture)

DEDEDE = "Creatures/Kirby/5 Stars/Masked_Dedede.psd"   # Name, Ability and "Ability FX: Blazing"
PUZZLED = "Creatures/Other/3 Stars/Puzzled_Cat.psd"    # her own pictures under the old template's digit rows
PIX = "Creatures/Kirby/3 Stars/Pix.psd"                 # three creatures: their own DEF rows, one HP bar for all
BOLTS = "Items/Heal/1 Stars/Bolts_and_Screws.psd"      # two type icons over its art (Mr Krabs)
EARTH = ("dd70ddfd9e430de6b51b301f822b52294382b57b",    # Earth_Totem.psd (TC 5743332b): its title under clipped stone
         "d7bf86ab789310481b48a3db0c8b9b45db837b98")    # its PNG
CLOT = "3f7040af67a5f605e0220b79051da3d0f9b43e57"       # Clot_Gold.psd (TC 96c3608e): its title's group under a Gradient Map
AUTOPSY = "Auxiliary/Items/Autopsy_Report.psd"        # its title printed again in its Evidence box (parts' follow)
EDEN = "Creatures/Isaac/2 Stars/Tainted_Eden.psd"     # two copies on its title, a few px aside: its red and cyan
SNACK = ("7fb550763c7c910e221aba2a95c98bc826b73a7e",     # A_Snack.psd tagged: Name #1, #2, #3 (TC be0eaeeb)
         "1b9d58befdada2b3ac2d7b7f20d4a361396be8e3", "d25bf9243f525e042e16218ff5479995415eb8bc",   # its PNG, A_Snack2.png
         "b342ed97bdea33d1f205de1c0eb86a1e856c5bb0")     # A_Snack3.png


def test_plates_of_a_card(clone):
    psd, png = PSDImage.open(clone / DEDEDE), Image.open(clone / DEDEDE.replace(".psd", ".png"))
    parts = card_parts(psd, DEDEDE, "Masked Dedede", _bare_parser())
    found = cp.plates(psd, png, parts, str(clone / "MDW/Fonts"))
    ids = [p["id"] for k in ("title", "ability") for p in parts[k]]
    assert [c["id"] for c in found["crops"] if "fx" not in c] == ids
    assert [c["fx"] for c in found["crops"] if "fx" in c] == [["Blazing"]]
    sheet = np.asarray(Image.open(io.BytesIO(base64.b64decode(found["img"].split(",", 1)[1]))).convert("RGB"), np.int16)
    small = np.asarray(png.convert("RGB").resize((cp.VIEW_W, round(png.height * cp.VIEW_W / png.width)), Image.LANCZOS), np.int16)
    for c in found["crops"]:
        x0, y0, x1, y1 = (round(v * cp.VIEW_W / psd.width) for v in c["box"])
        assert c["at"][2:] == [x1 - x0, y1 - y0] and c["box"][0] <= c["ink"][0] < c["ink"][2] <= c["box"][2]
        crop = sheet[c["at"][1]:c["at"][1] + c["at"][3], :c["at"][2]]
        off = np.ones(crop.shape[:2], bool)
        ix0, iy0, ix1, iy1 = (round(v * cp.VIEW_W / psd.width) for v in c["ink"])
        off[max(0, iy0 - y0 - 1):iy1 - y0 + 1, max(0, ix0 - x0 - 1):ix1 - x0 + 1] = False
        assert np.abs(crop - small[y0:y1, x0:x1]).max(-1)[off].mean() < 8   # the card's own backdrop
        rim = np.zeros(off.shape, bool)
        rim[[0, -1], :] = rim[:, [0, -1]] = True
        assert np.abs(crop - small[y0:y1, x0:x1]).max(-1)[rim & off].mean() < 4   # no seam from the sheet's gaps
        assert np.abs(crop - small[y0:y1, x0:x1]).max(-1)[~off].mean() > 20  # no letters
    for c in (c for c in found["crops"] if "fx" not in c):     # the layer's layout: its own text, line by line
        layer = next(x for x in psd.descendants() if x.layer_id == c["id"])
        assert c["text"].replace("\x03", "\r").strip() == layer.text.replace("\x03", "\r").strip()
        for start, end, base, left, right, top, bottom, _ in c["lines"]:
            xs = [x for i, x, *_ in c["glyphs"] if start <= i < end]
            assert c["ink"][1] - 2 <= top < base <= bottom + 2 <= c["ink"][3] + 4 and xs == sorted(xs) and left - 1 <= xs[0] < right
    assert all("lines" in c for c in found["crops"] if "fx" not in c) and not any("lines" in c for c in found["crops"] if "fx" in c)


def test_copies_of_a_title(clone):
    """A title's copy elsewhere has a crop of its own (Autopsy Report's Evidence title: its layout, its orange); copies
    on the title are told with its crop: Tainted Eden's two, each a few px aside over it, painting R or G and B."""
    psd, png = PSDImage.open(clone / AUTOPSY), Image.open(clone / AUTOPSY.replace(".psd", ".png"))
    parts = card_parts(psd, AUTOPSY, "Autopsy Report", _bare_parser())
    (copy,) = parts["follow"]
    crops = cp.plates(psd, png, parts, str(clone / "MDW/Fonts"))["crops"]
    own, title = (next(c for c in crops if c["id"] == i and "fx" not in c) for i in (copy["id"], copy["of"][0]))
    assert own["text"].strip() == "Autopsy Report" and own["lines"] and "over" not in own and "stacked" not in title
    assert own["box"][1] > title["box"][3] and max(abs(a - b) for a, b in zip(own["fill"], (0xe7, 0xa3, 0x1b))) < 16
    psd, png = PSDImage.open(clone / EDEN), Image.open(clone / EDEN.replace(".psd", ".png"))
    by = {x.layer_id: x for x in psd.descendants()}
    name = next(x for x in by.values() if x.kind == "type" and x.name == "Name")
    copies = [x for x in by.values() if x.kind == "type" and x.name.startswith("Name copy")]
    got = cp._stacked(psd, name, copies, cp._rgb(png), psd_edit.export_icc(psd, png), cp._step(psd))
    assert sorted((s["at"], s["ch"]) for s in got) == [([-5.0, 0.0], [1, 2]), ([4.0, 0.0], [0])]
    assert all(s["above"] and max(s["fill"][c] for c in s["ch"]) > 200 for s in got)


def test_the_card_under_its_type_icons(clone, parser):
    """with_under: Bolts and Screws' art where its two type icons were, the card's own pixels round them, a mask of
    each icon's ink."""
    psd, png = PSDImage.open(clone / BOLTS), Image.open(clone / BOLTS.replace(".psd", ".png"))
    u = cp.with_under(None, psd, png, parser, "Bolts and Screws")["under"]
    sheet = np.asarray(Image.open(io.BytesIO(base64.b64decode(u["img"].split(",", 1)[1]))).convert("RGB"), np.int16)
    s = cp.VIEW_W / psd.width
    x0, y0, x1, y1 = (round(v * s) for v in u["box"])
    bare, masks = sheet[:y1 - y0], [sheet[n * u["pitch"]:][:y1 - y0, :, 0] > 127 for n in (1, 2)]
    off = np.abs(bare - np.asarray(cp.view(png).convert("RGB"), np.int16)[y0:y1, x0:x1]).max(-1)
    inked = np.logical_or.reduce(masks)
    assert bare.shape[:2] == inked.shape == (y1 - y0, x1 - x0) and off[~inked].mean() < 8 and off[inked].mean() > 40
    icons = {tuple(layer.bbox): layer for _, layer in psd_edit.type_row(psd, parser, "Bolts and Screws")[0]}
    assert len(u["icons"]) == len(icons) == len(u["patch"]) == 2
    for at, mask in zip(u["icons"], masks):
        a, b, c, d = (round(v * s) for v in psd_edit._core(icons[tuple(at)]))
        assert mask[b - y0:d - y0, a - x0:c - x0].mean() > 0.5


def test_an_icons_drawing_keeps_the_layout(clone, monkeypatch):
    """The layout gate leaves icon glyphs' ink boxes out (card_standards._ink_off): Masked Dedede's two icons drawn
    otherwise (here: not at all) keep its Ability's layout, though counting them is past LAYOUT_OFF."""
    import psd_tools.composite.text as text
    from psd_tools.composite.text.geometry import text_geometry
    import commands.card_standards as cs
    layer = next(x for x in PSDImage.open(clone / DEDEDE).descendants() if x.kind == "type" and x.name.strip() == "Ability")
    fonts = str(clone / "MDW/Fonts")
    faces, kept, real = psd_edit.font_map(fonts, layer), cp.layout(layer, fonts), text.render
    icons = [g["ink"] for g in text_geometry(layer, faces)["glyphs"] if g["font"].startswith("stoogedings") and g["ink"]]

    def undrawn(layer, **kw):
        got = real(layer, **kw)
        rgba = got.rgba.copy()
        for x0, y0, x1, y1 in icons:
            rgba[max(round(y0) - got.bbox[1], 0):max(round(y1) - got.bbox[1], 0),
                 max(round(x0) - got.bbox[0], 0):max(round(x1) - got.bbox[0], 0)] = 0
        return dataclasses.replace(got, rgba=rgba)
    monkeypatch.setattr(text, "render", undrawn)
    assert len(icons) == 2 and kept and cs._ink_off(layer, faces, {"glyphs": []}) > cp.LAYOUT_OFF
    assert cp.layout(layer, fonts) == kept


def test_a_tagged_cards_other_pictures_titles(clone):
    """A Snack (Name #1/#2/#3): picture 1's crops as without pictures; Name #2 and #3 cut against A_Snack2/3.png."""
    from conftest import card_blob
    psd, *pngs = (io.BytesIO(card_blob(clone, sha)) for sha in SNACK)
    psd, pngs = PSDImage.open(psd), dict(zip((1, 2, 3), map(Image.open, pngs)))
    parts = card_parts(psd, "Items/Heal/2 Stars/A_Snack.psd", "A Snack", _bare_parser())
    own = [p["id"] for n in (2, 3) for p in parts["title"] if p["variants"] == [n]]
    alone, found = cp.plates(psd, pngs[1], parts), cp.plates(psd, pngs[1], parts, pictures={2: pngs[2], 3: pngs[3]})
    assert len(own) == 2 and [c["id"] for c in found["crops"]] == [c["id"] for c in alone["crops"]] + own
    sheet = np.asarray(Image.open(io.BytesIO(base64.b64decode(found["img"].split(",", 1)[1]))).convert("RGB"), np.int16)
    for c, n in zip(found["crops"][-2:], (2, 3)):
        x0, y0, x1, y1 = (round(v * cp.VIEW_W / psd.width) for v in c["box"])
        small = np.asarray(cp.view(pngs[n]).convert("RGB"), np.int16)[y0:y1, x0:x1]
        off = np.ones(small.shape[:2], bool)
        ix0, iy0, ix1, iy1 = (round(v * cp.VIEW_W / psd.width) for v in c["ink"])
        off[max(0, iy0 - y0 - 1):iy1 - y0 + 1, max(0, ix0 - x0 - 1):ix1 - x0 + 1] = False
        crop = sheet[c["at"][1]:c["at"][1] + c["at"][3], :c["at"][2]]
        assert np.abs(crop - small).max(-1)[off].mean() < 8 and np.abs(crop - small).max(-1)[~off].mean() > 20


def test_a_textured_titles_letter_look(clone):
    """Earth Totem's white title under its clipped stone (earthfx): a band across the card whose look inside the card's
    own letters is the PNG's, where flat white is not; Clot Gold's title under a Gradient Map alone gets none."""
    from conftest import card_blob
    raw = card_blob(clone, EARTH[0])
    psd, png = PSDImage.open(io.BytesIO(raw)), Image.open(io.BytesIO(card_blob(clone, EARTH[1])))
    parts = card_parts(psd, "Auxiliary/Items/Earth_Totem.psd", "Earth Totem", _bare_parser())
    title = next(x for x in psd.descendants() if x.layer_id == parts["title"][0]["id"])
    found = cp.plates(psd, png, parts, raw=raw)
    (look,) = found["letters"]
    x0, y0, x1, y1 = look["box"]
    assert look["id"] == title.layer_id and look["fill"] == "#ffffff" and (x0, x1) == (0, psd.width)
    assert y0 < title.bbox[1] < title.bbox[3] < y1
    sheet = np.asarray(Image.open(io.BytesIO(base64.b64decode(found["img"].split(",", 1)[1]))).convert("RGB"), np.int16)
    x, y, w, h = look["at"]
    inside = np.zeros((psd.height, psd.width), np.uint8)       # the letters' own pixels, 2 px in
    inside[title.top:title.bottom, title.left:title.right] = (title.numpy("shape")[..., 0] > 0.5) * 255
    inside = np.asarray(Image.fromarray(inside[y0:y1]).filter(ImageFilter.MinFilter(5)).resize((w, h), Image.BOX)) == 255
    small = np.asarray(png.convert("RGB").crop(look["box"]).resize((w, h), Image.LANCZOS), np.int16)
    assert inside.sum() > 1000 and np.abs(sheet[y:y + h, x:x + w] - small).max(-1)[inside].mean() < 25   # AVIF noise: 11
    assert np.abs(255 - small).max(-1)[inside].mean() > 100
    clot = PSDImage.open(io.BytesIO(card_blob(clone, CLOT)))
    name = next(x for x in clot.descendants() if x.kind == "type" and x.name == "Name")
    assert name.parent.clip_layers and not cp.textured(clot, name)


def test_bar_pictures_are_the_bakes(clone, tmp_path, monkeypatch):
    """A stat's picture is the bake's PNG of that edit (apply_jobs); the sheet keeps per value the view pixels moving past
    BAR_MOVED: pasted on the card's view, the bake's view there and within BAR_MOVED of it elsewhere; none for 11. A lossy
    sheet has the same cuts and alpha. A stat past 10 is pictured (the cells draw 0-10)."""
    raw = (clone / PUZZLED).read_bytes()
    psd, png = PSDImage.open(io.BytesIO(raw)), Image.open(clone / PUZZLED.replace(".psd", ".png"))
    png.load()
    parser = _bare_parser()
    parsed = psd_edit.parse_card(parser, str(clone / PUZZLED), PUZZLED, psd)
    psd_edit.use_record(psd, PUZZLED)
    icc = psd_edit.export_icc(psd, png)
    ours = cp.bar_value(psd, raw, png, parsed, parser, PUZZLED, "def", 9, icc)
    baked = psd_edit.apply_jobs(str(clone / PUZZLED), str(clone / PUZZLED.replace(".psd", ".png")), PUZZLED, {"def": 9},
                                parsed, parser, str(tmp_path), str(clone / "MDW/Fonts"), psd=psd)
    assert np.array_equal(np.asarray(ours), np.asarray(Image.open(baked.png_path)))
    assert cp.bar_value(psd, raw, png, parsed, parser, PUZZLED, "def", 11, icc) is None
    monkeypatch.setattr(cp, "bar_value", lambda *args: ours if args[6:8] == ("def", 9) else None)   # one value baked
    changes = cp.bar_changes(psd, raw, png, parsed, parser, PUZZLED)
    entry = _moved(png, ours)
    assert list(changes) == ["def"] and list(changes["def"]) == [9] and changes["def"][9][:2] == entry[:2]
    assert all(np.array_equal(a, b) for a, b in zip(changes["def"][9][2:], entry[2:]))
    sheet = cp.bar_sheet(changes, parsed["card"])
    assert cp.with_bars(None, psd, raw, png, parsed, parser, PUZZLED, "custom")["bars"] == sheet
    x, y, w, h, sx, sy = sheet["cuts"][sheet["bars"]["def"]["to"]["9"]]
    cut = np.asarray(Image.open(io.BytesIO(base64.b64decode(sheet["img"].split(",", 1)[1]))).convert("RGBA"))[sy:sy + h, sx:sx + w]
    shown, want = np.asarray(cp.view(png).convert("RGBA")).copy(), np.asarray(cp.view(ours).convert("RGBA"))
    held = cut[..., 3] >= 128
    shown[y:y + h, x:x + w][held] = cut[held]
    assert sheet["bars"]["def"]["at"] == 4 and held.sum() > 1000
    assert np.array_equal(shown[y:y + h, x:x + w][held], want[y:y + h, x:x + w][held])
    assert np.abs(shown.astype(int) - want).max() <= cp.BAR_MOVED
    assert cp.pictured(psd, png, {"atk": 12}, "new") and not cp.pictured(psd, png, {}, "custom")
    tall = np.random.default_rng(1).integers(3, 256, (2, 300, 40, 4), np.uint8)      # past `side`: two columns
    two = cp.bar_sheet({"def": {v: (0, 0, tall[v], np.zeros_like(tall[v])) for v in (0, 1)}}, {"def": 4}, side=400)
    img = np.asarray(Image.open(io.BytesIO(base64.b64decode(two["img"].split(",", 1)[1]))).convert("RGBA"))
    assert img.shape == (300, 80, 4) and [c[4:] for c in two["cuts"]] == [[0, 0], [40, 0]]
    assert all(np.array_equal(img[:, 40 * v:40 * v + 40, :3], tall[v][..., :3]) for v in (0, 1))
    monkeypatch.setattr(cp, "BAR_BYTES", 0)
    lossy = cp.bar_sheet(changes, parsed["card"])
    alpha = np.asarray(Image.open(io.BytesIO(base64.b64decode(lossy["img"].split(",", 1)[1]))).convert("RGBA"))[sy:sy + h, sx:sx + w, 3]
    assert lossy["img"].startswith("data:image/avif") and lossy["cuts"] == sheet["cuts"] and np.array_equal(alpha >= 128, held)


def test_star_pictures_are_the_bakes(clone, tmp_path):
    """A stars count's picture is the bake's PNG of that edit (apply_jobs into its "N Stars" folder, as card_bake's
    plan_move); with_stars pictures every count to the top but the card's, at the card's."""
    raw = (clone / PUZZLED).read_bytes()
    psd, png = PSDImage.open(io.BytesIO(raw)), Image.open(clone / PUZZLED.replace(".psd", ".png"))
    png.load()
    parser = _bare_parser()
    parsed = psd_edit.parse_card(parser, str(clone / PUZZLED), PUZZLED, psd)
    psd_edit.use_record(psd, PUZZLED)
    ours = cp.star_value(psd, raw, png, parsed, parser, PUZZLED, 4, psd_edit.export_icc(psd, png))
    baked = psd_edit.apply_jobs(str(clone / PUZZLED), str(clone / PUZZLED.replace(".psd", ".png")),
                                "Creatures/Other/4 Stars/Puzzled_Cat.psd", {"stars": 4}, parsed, parser, str(tmp_path),
                                str(clone / "MDW/Fonts"), psd=psd)
    assert np.array_equal(np.asarray(ours), np.asarray(Image.open(baked.png_path)))
    stars = cp.with_stars(None, psd, raw, png, parsed, parser, PUZZLED, 5)["icons"]["bars"]["stars"]
    assert stars["at"] == 3 and sorted(stars["to"]) == ["1", "2", "4", "5"]


def test_a_published_sheet_with_the_same_pixels_is_kept(monkeypatch):
    """bar_sheet keeps old's WebP when its pixels are the new sheet's (colours under alpha 0 aside), without an encode;
    another pixel, an AVIF or an unreadable one is encoded again."""
    now = np.random.default_rng(2).integers(3, 256, (2, 30, 20, 4), np.uint8)
    was = now.copy()
    was[:, ::2, ::3] = 0                        # moved there only: alpha 0 between
    changes = {"def": {v: (0, 0, now[v], was[v]) for v in (0, 1)}}
    fresh = cp.bar_sheet(changes, {"def": 4})
    decoded = lambda sheet: np.array(Image.open(io.BytesIO(base64.b64decode(sheet["img"].split(",", 1)[1]))))
    def url(arr, kind="WEBP"):
        buf = io.BytesIO()
        Image.fromarray(arr, "RGBA").save(buf, kind, lossless=True, exact=True)
        return {"img": f"data:image/{kind.lower()};base64," + base64.b64encode(buf.getvalue()).decode()}
    pixels = decoded(fresh)
    hidden, moved, shown = pixels.copy(), pixels.copy(), pixels[..., 3] > 0
    hidden[~shown, :3] = 7
    moved[tuple(np.argwhere(shown)[0])][0] ^= 1
    same, other = [fresh, url(hidden)], [url(moved), url(pixels, "AVIF"), {"img": "data:image/webp;base64,AAAA"}, None]
    assert (~shown).sum() > 100 and (decoded(same[1])[~shown][:, :3] == 7).all()
    saves, save = [], Image.Image.save
    monkeypatch.setattr(Image.Image, "save", lambda self, *a, **k: saves.append(a[1]) or save(self, *a, **k))
    for old in same:
        assert cp.bar_sheet(changes, {"def": 4}, old=old) == {**fresh, "img": old["img"]} and not saves
    for old in other:
        assert cp.bar_sheet(changes, {"def": 4}, old=old) == fresh and saves == ["WEBP"]
        saves.clear()


def test_creature_pictures_are_the_bakes(clone, tmp_path):
    """Creature 2's own stat picture is the bake's PNG of that `creatures` edit, keyed "<stat>@2" at its value; none
    past what its row prints, nor on a bar creature 1 shares (creature 1's picture)."""
    raw = (clone / PIX).read_bytes()
    psd, png = PSDImage.open(io.BytesIO(raw)), Image.open(clone / PIX.replace(".psd", ".png"))
    png.load()
    parser = _bare_parser()
    parsed = psd_edit.parse_card(parser, str(clone / PIX), PIX, psd)
    psd_edit.use_record(psd, PIX)
    icc = psd_edit.export_icc(psd, png)
    wanted = [dict(one) for one in parsed["card"]["creatures"]]
    wanted[1]["def"] = 9
    job = psd_edit.classify(parsed, {"creatures": wanted}, psd, parser).jobs["creatures"]
    ours = cp.bar_value(psd, raw, png, parsed, parser, PIX, "def", 9, icc, creature=2)
    baked = psd_edit.apply_jobs(str(clone / PIX), str(clone / PIX.replace(".psd", ".png")), PIX, {"creatures": job},
                                parsed, parser, str(tmp_path), str(clone / "MDW/Fonts"), psd=psd)
    assert np.array_equal(np.asarray(ours), np.asarray(Image.open(baked.png_path)))
    assert cp.bar_value(psd, raw, png, parsed, parser, PIX, "def", 11, icc, creature=2) is None
    assert cp.bar_value(psd, raw, png, parsed, parser, PIX, "hp", 5, icc, creature=2) is None
    at = {**parsed["card"], **cp._creature_stats(parsed["card"])}
    assert cp.bar_sheet({"def@2": {9: _moved(png, ours)}}, at)["bars"]["def@2"]["at"] == 8 and at["def@3"] == 2


def test_a_comp_bar_at_zero_is_pictured(monkeypatch):
    """A layer-comp stat at 0 is pictured: its lit look is under the darks (the 4 own-look rows, 11-12), the cells could
    only borrow another row's."""
    psd = type("P", (), {"size": (10, 20), "descendants": lambda self: []})()
    monkeypatch.setattr(cp, "variants_of", lambda psd: [None])
    monkeypatch.setattr(cp, "comp_bars", lambda psd: {"hp": [], "def": [("bar", 0, (0, 0, 1, 1))], "atk": [], "spd": []})
    monkeypatch.setattr(cp, "_own", lambda layer: False)
    monkeypatch.setattr(cp, "mock_misses", lambda *args: [])
    monkeypatch.setattr(cp, "view", lambda png: png)
    monkeypatch.setattr(cp, "_stats", lambda psd: {})
    png = type("I", (), {"size": (10, 20)})()
    assert cp.pictured(psd, png, {"hp": 0, "def": 0}, "new") and not cp.pictured(psd, png, {"hp": 0, "def": 2}, "new")


def test_bar_changes_read_caps_once(clone, monkeypatch):
    """bar_changes reads stat_caps once for all its values; a value's picture is bar_value's own (classify reading them).
    Its frozen card and gate_dirty's are only read (rcache re-keys them at exit)."""
    monkeypatch.setenv("PSD_RCACHE_VERIFY", "1")
    raw = (clone / PUZZLED).read_bytes()
    psd, png = PSDImage.open(io.BytesIO(raw)), Image.open(clone / PUZZLED.replace(".psd", ".png"))
    png.load()
    parser = _bare_parser()
    parsed = psd_edit.parse_card(parser, str(clone / PUZZLED), PUZZLED, psd)
    calls, read = [], psd_edit.stat_caps
    monkeypatch.setattr(psd_edit, "stat_caps", lambda p: calls.append(p) or read(p))
    monkeypatch.setattr(cp, "BAR_TOP", 1)
    changes = cp.bar_changes(psd, raw, png, parsed, parser, PUZZLED)
    assert len(calls) == 1 and calls[0] is psd and sorted(changes) == sorted(psd_edit.STAT_FIELDS)
    alone = _moved(png, cp.bar_value(psd, raw, png, parsed, parser, PUZZLED, "def", 0, psd_edit.export_icc(psd, png)))
    assert len(calls) == 2 and all(np.array_equal(a, b) for a, b in zip(changes["def"][0], alone))


def test_moved_views_only_the_moved_rows():
    """_moved with the PNG's array (_band: a crop of the moved rows) equals view() of the whole card, the card's edges too;
    a scale view() can't match that way takes the whole view."""
    rng = np.random.default_rng(5)
    for channels, h in ((3, 2400), (4, 1800)):
        png = Image.fromarray(rng.integers(0, 256, (h, 1200, channels), np.uint8))
        was, base = np.asarray(cp.view(png).convert("RGBA")), cp._base(png)
        assert cp._moved(was, png, base) is None
        for y0, y1 in ((0, 3), (h - 2, h), (700, 701), (1000, 1290)):
            arr = base.copy()
            arr[y0:y1, 100:900] = rng.integers(0, 256, (y1 - y0, 800, channels), np.uint8)
            merged = Image.fromarray(arr)
            got, want = cp._moved(was, merged, base), cp._moved(was, merged)
            assert got[:2] == want[:2] and all(np.array_equal(a, b) for a, b in zip(got[2:], want[2:]))
    assert cp._band(was[:, :719], merged, base) is None


def _moved(png, merged):
    """bar_changes' entry of one value."""
    was, now = (np.asarray(cp.view(image).convert("RGBA")) for image in (png, merged))
    ys, xs = np.nonzero((now != was).any(-1))
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    return int(x0), int(y0), now[y0:y1, x0:x1], was[y0:y1, x0:x1]


def test_publish_adds_keeps_and_drops(tmp_path, monkeypatch):
    env = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "HOME": str(tmp_path)}
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    git = lambda *args: subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    origin, work = tmp_path / "origin.git", tmp_path / "work"
    git("init", "-q", "--bare", "-b", "main", str(origin))
    git("-C", str(origin), "config", "uploadpack.allowFilter", "true")
    seed = tmp_path / "seed"
    git("init", "-q", "-b", "main", str(seed))
    for name in ("A.psd", "B.psd", "C.png"):
        (seed / name).write_bytes(name.encode())
    git("-C", str(seed), "add", "-A")
    git("-C", str(seed), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x")
    git("-C", str(seed), "push", "-q", str(origin), "main")
    git("clone", "-q", "--filter=blob:none", f"file://{origin}", str(work))     # blobless, as the job's checkout
    monkeypatch.setenv("GIT_NO_LAZY_FETCH", "1")
    a, b, c = (git("-C", str(work), "rev-parse", f"HEAD:{n}") for n in ("A.psd", "B.psd", "C.png"))
    files = lambda: sorted(git("-C", str(origin), "ls-tree", "--name-only", cp.BRANCH).split())
    plate = {"v": 1, "crops": []}
    assert cp.publish({}, str(work))["commit"] == "" and not git("-C", str(origin), "branch", "--list", cp.BRANCH)
    done = cp.publish({a: plate, b: plate, c: plate, "f" * 40: plate}, str(work))
    assert (done["added"], done["dropped"]) == (2, 0) and files() == sorted([f"{a}.json", f"{b}.json"])
    git("-C", str(seed), "rm", "-q", "B.psd")
    git("-C", str(seed), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "y")
    git("-C", str(seed), "push", "-q", str(origin), "main")
    o = str(origin)                                                             # site.yml's versions job adds v/
    put = lambda *args, data="": subprocess.run(["git", "-C", o, "-c", "user.name=t", "-c", "user.email=t@t", *args], input=data,
                                                capture_output=True, text=True, check=True).stdout.strip()
    copy = put("hash-object", "-w", "--stdin", data="x")
    v = put("mktree", data=f"040000 tree {put('mktree', data=f'100644 blob {copy}\tc.avif')}\tid\n")
    git("-C", o, "update-ref", f"refs/heads/{cp.BRANCH}", put("commit-tree", put("mktree", data=git("-C", o, "ls-tree", cp.BRANCH)
                                                                                  + f"\n040000 tree {v}\tv\n"), "-p", cp.BRANCH, "-m", "v"))
    later = tmp_path / "later"                                                  # the next run: the branch's blobs not here
    monkeypatch.delenv("GIT_NO_LAZY_FETCH")
    git("clone", "-q", "--filter=blob:none", f"file://{origin}", str(later))
    monkeypatch.setenv("GIT_NO_LAZY_FETCH", "1")
    done = cp.publish({}, str(later))
    assert (done["added"], done["dropped"], done["kept"]) == (0, 1, 1) and files() == [f"{a}.json", "v"]
    assert git("-C", o, "rev-parse", f"{cp.BRANCH}:v") == v                     # kept as it was
    assert json.loads(git("-C", str(origin), "show", f"{cp.BRANCH}:{a}.json")) == plate
    assert cp.publish({a: plate}, str(later))["commit"] == ""                                   # nothing changed
    assert cp.on_branch(str(later)) == {a: git("-C", str(origin), "rev-parse", f"{cp.BRANCH}:{a}.json")}
    fresh = tmp_path / "fresh"                                                  # a plate replaced, its old blob not here
    monkeypatch.delenv("GIT_NO_LAZY_FETCH")
    git("clone", "-q", "--filter=blob:none", f"file://{origin}", str(fresh))
    monkeypatch.setenv("GIT_NO_LAZY_FETCH", "1")
    assert cp.on_branch(str(fresh)) == {}                                       # not fetched yet
    run = cp.run_process                                                        # HTTPS pushes thin whatever --no-thin says
    monkeypatch.setattr(cp, "run_process", lambda cmd, **kw: run(["--thin" if a == "--no-thin" else a for a in cmd], **kw))
    newer = {"v": 1, "crops": [[n, n, 9, 9] for n in range(99)]}
    assert cp.publish({a: newer}, str(fresh))["added"] == 1
    assert json.loads(git("-C", str(origin), "show", f"{cp.BRANCH}:{a}.json")) == newer
