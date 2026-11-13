"""
psd_links (the big pass, PLAN §3 A4): new files' names and the catalogue's classes, the trim map on Star of beth, links
placed from any depth and from absolute URLs (with the guards' housekeeping), and Card sync's redraw of one MDW/Linked
file with two users.
    ~/.venvs/ac-py314-fork/bin/python -m pytest -q tests/test_psd_links.py
"""
import json
import subprocess
from base64 import b64decode
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer
from psd_tools.api.links import link_content_id

import commands.png_export as png_export
import commands.psd_edit as pe
import commands.psd_links as pl

DELTA = "Auxiliary/Minions/Delta.psd"            # three Star of beth items (Croissant trimmed the file, 11-07)
MINI_DOOMER = "Auxiliary/Minions/Mini_Doomer.psd"   # a 'Display' profile card
FREEZER = "Field/1 Stars/The_Freezer.psd"        # an embedded PSB holding a gradient fill (before the big pass)


def _png(image: Image.Image, level: int = 6) -> bytes:
    out = BytesIO()
    image.save(out, "PNG", compress_level=level)
    return out.getvalue()


def _art() -> Image.Image:
    art = Image.new("RGBA", (100, 80), (0, 0, 0, 0))
    draw = ImageDraw.Draw(art)
    draw.ellipse((5, 5, 95, 75), fill=(200, 60, 40, 255))
    draw.rectangle((30, 20, 70, 60), fill=(20, 90, 200, 255))
    return art


def _strip(psd: PSDImage) -> PSDImage:
    """No merged image, as every card after the pass (Photoshop's planes with Maximize Compatibility off)."""
    from psd_tools.constants import Compression, Resource
    header, data = psd._record.header, psd._record.image_data
    data.compression = Compression.RLE
    data.set_data([b"\xff" * (header.width * header.height)] * 3, header)
    psd._record.image_resources.get_data(Resource.VERSION_INFO).has_composite = False
    return psd


def test_names_and_classes():
    """New files: "<PSD stem> - <layer>.<ext>", safe, unique case-folded, under 120 characters. The classes in order:
    Types bytes, ink, an older version, a name in its row or a name that looks close anywhere, the icon at another size
    under another name (else a new file: a big picture merely named "fire"), an existing file's bytes; first user names."""
    taken = {"delta - bg.png"}
    assert pl.file_name("Delta", "bg", "png", taken) == "Delta - bg (2).png"
    assert pl.file_name("Delta", 'a/b:c*?"<>|\x01. ', "jpg", taken) == "Delta - a_b_c_______.jpg"
    long = pl.file_name("Card", "x" * 300, "png", taken)
    assert len(long) < 120 and long.endswith(".png") and pl.file_name("Card", "x" * 300, "png", taken) != long

    fire, star = Image.new("RGBA", (40, 40), (0, 0, 0, 0)), Image.new("RGBA", (30, 30), (250, 200, 30, 255))
    ImageDraw.Draw(fire).ellipse((8, 8, 32, 32), fill=(230, 80, 20, 255))
    old_fire = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    ImageDraw.Draw(old_fire).rectangle((8, 8, 32, 32), fill=(230, 80, 20, 255))
    trimmed = fire.crop(fire.getbbox())
    files = {"Types/fire.png": _png(trimmed), "Types/Stars/star.png": _png(star)}
    types = pl.TypesIndex(files, {pl.git_blob(_png(old_fire)): "Types/fire.png"})
    art, scene = _png(_art()), _png(_art().transpose(Image.Transpose.ROTATE_90))
    big_fire = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    ImageDraw.Draw(big_fire).ellipse((20, 20, 180, 180), fill=(230, 80, 20, 255))

    def entry(data, name, row=False):
        rgba = np.asarray(Image.open(BytesIO(data)).convert("RGBA"))
        return {"uuid": name + str(row), "kind": "data", "blob": pl.git_blob(data), "ext": "png", "file": "",
                "key": pl.ink_key(rgba), "look": pl.look_of(rgba), "layers": [{"name": name, "row": row}]}
    psds = [("Items/A.psd", [entry(files["Types/Stars/star.png"], "Star"), entry(_png(fire), "fire"),
                             entry(_png(old_fire), "Fire"), entry(_png(star.resize((20, 20))), "star", True),
                             entry(_png(star.resize((20, 20))), "star"), entry(art, "Art"), entry(scene, "fire"),
                             entry(_png(big_fire), "blaze")]),
            ("Items/B.psd", [entry(art, "Other"), entry(_png(Image.new("RGBA", (5, 5), (1, 2, 3, 255))), "cain")])]
    linked = {"MDW/Linked/cain.jpeg": pl.git_blob(_png(Image.new("RGBA", (5, 5), (1, 2, 3, 255))))}
    found = pl.assign(psds, types, linked)
    pl.fallback_names(found, linked)
    rows = found["psds"]
    assert [(r["class"], r["target"]) for r in rows["Items/A.psd"]] == [
        (pl.BYTES, "Types/Stars/star.png"), (pl.INK, "Types/fire.png"), (pl.REDRAW, "Types/fire.png"),
        (pl.REDRAW, "Types/Stars/star.png"), (pl.REDRAW, "Types/Stars/star.png"), (pl.NEW, "MDW/Linked/A - Art.png"),
        (pl.NEW, "MDW/Linked/A - fire.png"), (pl.REDRAW, "Types/fire.png")]
    assert [(r["class"], r["target"]) for r in rows["Items/B.psd"]] == [(pl.LINKED, "MDW/Linked/A - Art.png"),
                                                                       (pl.LINKED, "MDW/Linked/cain.jpeg")]
    assert rows["Items/A.psd"][3]["fallback"] == rows["Items/A.psd"][4]["fallback"] == "MDW/Linked/A - star.png"
    assert found["files"]["MDW/Linked/A - Art.png"]["first"] == "Items/A.psd"
    lookalike = np.asarray(big_fire.resize((60, 60)).convert("RGBA")).copy()
    lookalike[..., :3] = (200, 110, 40)                     # the shape, other colours: resembles only, no name
    assert types.match("x", None, ["ember"], False, pl.look_of(lookalike)) is None


def test_trim_map_on_star_of_beth(clone):
    """Croissant's trim (the same ink, the margins cut): each of Delta's Star of beth items links the trimmed file, its
    raster kept, its quad moved so the file's ink draws where the old one did; the rest by bytes or as new files."""
    rel = DELTA
    with Image.open(clone / "Types/Stars/Star of beth.png") as image:
        full = image.convert("RGBA")
    trimmed = _png(full.crop(full.getbbox()))
    files = {"Types/Stars/Star of beth.png": trimmed, "Types/Offensive_Minion.png": (clone / "Types/Offensive_Minion.png").read_bytes()}
    types = pl.TypesIndex(files)
    psd = PSDImage.open(clone / rel)
    before = {layer.layer_id: (pl.corners_of(layer), [c.data for c in layer._channels]) for layer in psd.descendants()
              if layer.name == "Isaac"}
    assert pl.link_issues(psd, rel, files) == ["embedded: The_Queen_bg", "embedded: Delta_sprite", "embedded: Isaac",
                                               "embedded: Offensive_Minion"]
    catalogue = pl.assign([(rel, pl.items(psd, types.vocab()))], types, {})
    pl.fallback_names(catalogue, {})
    rows = catalogue["psds"][rel]
    assert sorted(r["class"] for r in rows) == [pl.BYTES, pl.INK, pl.INK, pl.INK, pl.LINKED, pl.NEW, pl.NEW]
    done = pl.link_card(psd, rel, rows, files.get, set(files), pl.ppis_of(files.get))
    assert sorted(done.files) == ["MDW/Linked/Delta - Delta_sprite.psb", "MDW/Linked/Delta - The_Queen_bg.png"]
    assert not done.left and not done.redrawn and not done.notes
    saved = BytesIO()
    psd.save(saved, update_preview=False)
    psd = PSDImage.open(BytesIO(saved.getvalue()))
    new = np.asarray(Image.open(BytesIO(trimmed)).convert("RGBA"))
    old = np.asarray(full)
    for layer in (layer for layer in psd.descendants() if layer.name == "Isaac"):
        corners, channels = before[layer.layer_id]
        assert [c.data for c in layer._channels] == channels                          # the raster kept
        a, box_a = pl.draw(old, corners, 0)
        b, box_b = pl.draw(new, pl.corners_of(layer), 0)
        assert box_a == box_b and np.abs(a - b).max() <= 1 / 255                       # the same ink, the same place
    tree = set(files) | set(done.files) | {"MDW/Linked/x"}
    links = pl.links_of(psd, rel, tree)
    assert len(links) == 7 and all(link.canonical and link.path for link in links) and not pl.link_issues(psd, rel, tree)
    assert not set(pl._items_of(psd)) - {pl._uid(layer) for layer in psd.descendants() if layer.kind == "smartobject"}


def test_update_snaps_like_photoshop():
    """S0 (11-07): Photoshop's Update of Delta's background resized to 922x518 placed it at 153,37-2977,1623.
    S2 (11-08): Dragon Mount's Warpstar at 1.2x: corner and size rounded apart (round the centre gave top 101)."""
    assert pl.snapped([153.188, 36.5, 2976.812, 36.5, 2976.812, 1622.875, 153.188, 1622.875]) == \
        [153, 37, 2977, 37, 2977, 1623, 153, 1623]
    assert pl.snapped([60.473, 1738.007, 1208.498, 1738.007, 1208.498, 2382.993, 60.473, 2382.993]) == \
        [60, 1738, 1208, 1738, 1208, 2383, 60, 2383]
    assert pl.snapped([999.955, 100.459, 1150.533, 100.459, 1150.533, 234.775, 999.955, 234.775]) == \
        [1000, 100, 1151, 100, 1151, 234, 1000, 234]


def test_links_placed_from_any_depth_and_url():
    """links_of places a link from its PSD's folder 1, 2 and 3 deep, and from absolute forms cut at /TTSCardMaker/
    (Militron's Windows file:// URL, Tomorrow I'll's Mac one, a Windows path), case-blind; repair() makes each canonical
    and leaves its size; a link no rule places is a hand note. The guards' housekeeping: unused files, duplicates, a
    restore."""
    tree = ["Types/Fire.png", "Types/Stars/star.png", "MDW/Linked/cain.jpeg"]
    fire = _png(Image.new("RGBA", (20, 20), (230, 80, 20, 255)))
    forms = {"relPath": None, "fullPath": "file:///C//Users/Zoey/Documents/GitHub/TTSCardMaker/Types/fire.png",
             "originalPath": r"A:\Chris\Documents\GitHub\TTSCardMaker\Types\Fire.png"}
    for rel in ("Field/A.psd", "Items/Heal/B.psd", "Creatures/Kirby/3 Stars/C.psd"):
        psd = PSDImage.new("RGB", (40, 40))
        layer = PixelLayer.fromarray(np.full((20, 20, 4), (230, 80, 20, 255), np.uint8), psd, name="Fire")
        made = pl.link_pixels(layer, rel, "Types/Fire.png", fire, [0, 0, 20, 0, 20, 20, 0, 20])
        strings = made.smart_object._data.linked_file
        assert pl.links_of(psd, rel, tree)[0].canonical and pl._text(strings[b"relPath"]) == "../" * rel.count("/") + "Types/Fire.png"
        for key, value in forms.items():                    # one string left at a time, the others blank
            strings = made.smart_object._data.linked_file
            for other in forms:
                strings[other.encode()] = type(strings[b"relPath"])((value if other == key else "") + "\x00") if value \
                    else type(strings[b"relPath"])(("../" * rel.count("/") + "types/FIRE.png" if other == key else "") + "\x00")
            made.smart_object._data.filesize = 7
            assert [(link.path, link.canonical) for link in pl.links_of(psd, rel, tree)] == [("Types/Fire.png", False)]
            assert pl.repair(psd, rel, tree) == (["Fire.png"], [])
            link = pl.links_of(psd, rel, tree)[0]
            assert link.canonical and link.size == 7
        strings = made.smart_object._data.linked_file
        strings[b"relPath"] = type(strings[b"relPath"])("../" * (rel.count("/") + 1) + "Types/Fire.png\x00")
        strings[b"fullPath"] = strings[b"originalPath"] = type(strings[b"relPath"])("C:/Users/x/Fire.png\x00")
        assert pl.links_of(psd, rel, tree)[0].path is None                   # a climb above the root places nowhere
        assert pl.repair(psd, rel, tree) == ([], ["Links Fire.png, which is not in the repo: push that file."])

    links = {"Field/A.psd": {"blob": "a", "links": [{"path": "MDW/Linked/x.png"}, {"path": "MDW/Linked/inner.psb"}]},
             "MDW/Linked/inner.psb": {"blob": "p", "links": [{"path": None, "name": "Kept.PNG"}]},
             "MDW/Templates/T.psd": {"blob": "t", "links": [{"path": "MDW/Linked/cain.jpeg"}]},
             "MDW/Linked/Layer 1.psb": {"blob": "4", "links": []}}
    blobs = {"Field/A.psd": "a", "MDW/Linked/inner.psb": "p", "MDW/Templates/T.psd": "t", "MDW/Linked/x.png": "1",
             "MDW/Linked/cain.jpeg": "2", "MDW/Linked/kept.png": "3", "MDW/Linked/Layer 1.psb": "4", "MDW/Linked/new.png": "5",
             "Types/Fire.png": "6"}
    assert pl.unused(links, blobs, pushed=["MDW/Linked/new.png"]) == ["MDW/Linked/Layer 1.psb"]
    assert pl.unused(links, {**blobs, "Items/B.psd": "b"}) == []                  # a PSD without its record blocks it
    assert pl.duplicates({**blobs, "MDW/Linked/dup.png": "6", "MDW/Linked/x2.png": "1"},
                         pushed=["MDW/Linked/dup.png", "MDW/Linked/x2.png"]) == {"MDW/Linked/dup.png": "Types/Fire.png",
                                                                                 "MDW/Linked/x2.png": "MDW/Linked/x.png"}
    assert pl.restore(10, [("old", 4), ("last", 10)]) == "last" and pl.restore(10, [("last", 9)]) is None


def test_rslt_is_the_files_ppi():
    """A link stores its file's ppi as Photoshop reads it (r9: a 72 made Update shrink Wind to 24%): pHYs rounded, a
    JPEG's Photoshop ResolutionInfo over JFIF (Fawful: 2.54, JFIF 3), 72 with none; link_pixels writes it, repair()
    mends another, link_issues and needs() see it only with ppis."""
    icon = Image.new("RGBA", (20, 20), (90, 200, 250, 255))
    saved = {}
    for dpi in (300, 96, None):
        out = BytesIO()
        icon.save(out, "PNG", **({"dpi": (dpi, dpi)} if dpi else {}))
        saved[dpi] = out.getvalue()
    assert [pl.ppi(saved[d], "x.png") for d in (300, 96, None)] == [300.0, 96.0, 72.0]
    out = BytesIO()
    icon.convert("RGB").save(out, "JPEG", dpi=(3, 3))
    res = b"8BIM\x03\xed\x00\x00\x00\x00\x00\x10" + b"\x00\x02\x8a\x3d\x00\x01\x00\x02" * 2
    irb = b"Photoshop 3.0\x00" + b"8BIM\x04\x04\x00\x00\x00\x00\x00\x03abc\x00" + res
    jpeg = out.getvalue()[:2] + b"\xff\xed" + (len(irb) + 2).to_bytes(2, "big") + irb + out.getvalue()[2:]
    assert pl.ppi(jpeg, "x.jpg") == 0x28a3d / 65536 and pl.ppi(out.getvalue(), "x.jpg") == 3.0

    rel, tree = "Items/A.psd", ["Types/Wind.png"]
    psd = PSDImage.new("RGB", (40, 40))
    layer = PixelLayer.fromarray(np.full((20, 20, 4), (90, 200, 250, 255), np.uint8), psd, name="Wind")
    made = pl.link_pixels(layer, rel, "Types/Wind.png", saved[300], [0, 0, 20, 0, 20, 20, 0, 20])
    assert pl.placed_ppi(made) == 300.0
    pl.set_ppi([made], 72.0)
    ppis = pl.ppis_of({"Types/Wind.png": saved[300]}.get)
    assert pl.link_issues(psd, rel, tree) == [] and pl.link_issues(psd, rel, tree, ppis) == ["repair: Wind.png"]
    psd.repo_path, psd.link_ppi = rel, ppis
    assert pl.needs(psd) == ["repair: Wind.png"]
    assert pl.repair(psd, rel, tree, ppis) == (["Wind.png"], []) and pl.placed_ppi(made) == 300.0
    assert pl.needs(psd) == [] and pl.repair(psd, rel, tree, ppis) == ([], [])


def test_redraw_linked_one_file_two_users(tmp_path, capsys):
    """One MDW/Linked file, two users at two depths and scales (one PNG named in another case). A lossless re-save: their
    sizes and contentIDs only, no raster, no PNG. A pixel edit: both drawn again as their stored pixels show (nearest;
    Bicubic Automatic) on their own quads (same size: no snap), their PNGs re-exported to the new drawing; a relink takes
    nearest alone from them; card-sync.yml's due and redraw command lines see both."""
    repo = tmp_path / "cards"
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True).stdout
    repo.mkdir()
    git("init", "-q")
    art, users = _art(), {"Creatures/X/3 Stars/A.psd": (40.4, 60.3, 2.0, pl.PS_NEAREST), "Items/B.psd": (300, 200, 1.5, pl.PS_AUTO)}
    pngs = {"Creatures/X/3 Stars/A.psd": "Creatures/X/3 Stars/A.png", "Items/B.psd": "Items/b.png"}
    data = _png(art, 9)
    (repo / "MDW/Linked").mkdir(parents=True)
    (repo / "MDW/Linked/art.png").write_bytes(data)
    for rel, (x, y, s, mode) in users.items():
        psd = PSDImage.new("RGB", (600, 500))
        PixelLayer.fromarray(np.full((500, 600, 4), (240, 240, 230, 255), np.uint8), psd, name="bg")
        corners = [x, y, x + 100 * s, y, x + 100 * s, y + 80 * s, x, y + 80 * s]
        out, box = pl.draw(np.asarray(art), corners, mode)
        layer = PixelLayer.fromarray((out * 255 + 0.5).astype(np.uint8), psd, name="art", top=box[1], left=box[0])
        pl.link_pixels(layer, rel, "MDW/Linked/art.png", data, corners)
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        _strip(psd).save(repo / rel, update_preview=False)
        (repo / pngs[rel]).write_bytes(png_export.encode(PSDImage.open(repo / rel).composite(ignore_preview=True)
                                                                   .convert("RGB"), None))
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")

    def links_json(seen: dict) -> dict:
        tree = pl._tree_at(str(repo), "HEAD")
        out = {"files": {"MDW/Linked/art.png": {**seen, tree["MDW/Linked/art.png"]: (repo / "MDW/Linked/art.png").stat().st_size}}}
        for rel in users:
            found = pl.links_of(PSDImage.open(repo / rel), rel, tree)
            out[rel] = {"blob": tree[rel], "links": [{"path": f.path, "size": f.size, "drawn": tree[f.path]} for f in found]}
        return out
    known = links_json({})

    def commit(picture: Image.Image, level: int) -> bytes:
        (repo / "MDW/Linked/art.png").write_bytes(_png(picture, level))
        git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "art")
        return (repo / "MDW/Linked/art.png").read_bytes()
    lossless = commit(art, 1)
    done = pl.redraw_linked(str(repo), known)
    assert sorted(done.files) == sorted(users) and not done.notes and not done.due
    for rel in users:
        old, new = PSDImage.open(repo / rel), PSDImage.open(BytesIO(done.files[rel]))
        assert [c.data for c in old[1]._channels] == [c.data for c in new[1]._channels]
        assert new[1].smart_object.filesize == len(lossless)
        assert pl.links_of(new, rel, ["MDW/Linked/art.png"])[0].content == link_content_id(pl.git_blob(lossless), rel,
                                                                                           new[1].smart_object.unique_id)
        (repo / rel).write_bytes(done.files[rel])
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "R")
    known = links_json(known["files"]["MDW/Linked/art.png"])                    # the data job after R
    edit = art.copy()
    ImageDraw.Draw(edit).rectangle((10, 30, 40, 50), fill=(250, 250, 0, 255))
    pixels = commit(edit, 6)
    db, report = tmp_path / "db", tmp_path / "linked"
    db.mkdir()
    (db / "links.json").write_text(json.dumps(known))
    capsys.readouterr()
    assert pl.main(["due", "--head", "HEAD", "--clone", str(repo), "--db", str(db)]) == 0
    assert capsys.readouterr().out == "2\n"
    assert pl.main(["redraw", "--head", "HEAD", "--local", str(repo), "--db", str(db), "--out", str(report),
                    "--deadline", "0", "--bytes", "1000000000", "--dry-run"]) == 0
    assert json.loads((report / "redraw.json").read_text())["cards"] == sorted(users)
    done = pl.redraw_linked(str(repo), known)
    assert sorted(done.files) == sorted([*users, *pngs.values()]) and not done.notes
    for rel, (x, y, s, mode) in users.items():
        assert pe.image_mode(PSDImage.open(repo / rel)[1], np.asarray(art), relink=True) == mode
        new = PSDImage.open(BytesIO(done.files[rel]))
        want, box = pl.draw(np.asarray(edit), [x, y, x + 100 * s, y, x + 100 * s, y + 80 * s, x, y + 80 * s], mode)
        assert tuple(new[1].bbox) == box and np.abs(new[1].numpy() - want).max() <= 1 / 255 and new[1].smart_object.filesize == len(pixels)
        render = np.asarray(new.composite(ignore_preview=True).convert("RGB"), np.int16)
        png = np.asarray(Image.open(BytesIO(done.files[pngs[rel]])).convert("RGB"), np.int16)
        assert np.abs(png - render).max() <= png_export.TOL


def test_photoshop_resampling():
    """draw's PS_ modes against Photoshop 2023's own pixels (fixtures/ps_resample.json: placed noise at its Image
    Interpolation preference: Bicubic Automatic enlarging and reducing, straight alpha, nearest): within 6 levels,
    nearest exact; PIL's bicubic is not."""
    cases = json.loads((Path(__file__).parent / "fixtures/ps_resample.json").read_text())["cases"]
    for case in cases:
        source = np.asarray(Image.open(BytesIO(b64decode(case["source"]))).convert("RGBA"))
        want = np.asarray(Image.open(BytesIO(b64decode(case["photoshop"]))).convert("RGBA"), np.int16)
        x0, y0, x1, y1 = case["box"]

        def off(mode) -> int:
            rgba, box = pl.draw(source, case["corners"], mode)
            return np.abs(np.round(rgba * 255).astype(np.int16)[y0 - box[1]:y1 - box[1], x0 - box[0]:x1 - box[0]] - want).max()
        mode = getattr(pl, case["mode"])
        assert off(mode) <= (0 if mode == pl.PS_NEAREST else 6) < off(Image.BICUBIC), case["what"]


def test_a_display_card_keeps_its_nearest_picture(clone):
    """Card sync's same-size redraw of an untagged file on a card in a 'Display' profile (Mini Doomer's) whose stored
    pixels are nearest: image_mode weighs the old file in the card's colours, so it stays nearest (in the file's own,
    no mode was clear and it went smooth, effectsgaps 11-13)."""
    from psd_tools.constants import Resource
    from psd_tools.psd.image_resources import ImageResource
    rel, path = "Creatures/X/3 Stars/A.psd", "MDW/Linked/art.png"
    psd = PSDImage.new("RGB", (600, 500))
    icc = PSDImage.open(clone / MINI_DOOMER).image_resources.get_data(Resource.ICC_PROFILE)
    psd.image_resources[Resource.ICC_PROFILE] = ImageResource(key=Resource.ICC_PROFILE, data=icc)
    PixelLayer.fromarray(np.full((500, 600, 4), (240, 240, 230, 255), np.uint8), psd, name="bg")
    art, corners = Image.new("RGBA", (100, 80)), [40.4, 60.3, 240.4, 60.3, 240.4, 220.3, 40.4, 220.3]
    ImageDraw.Draw(art).ellipse((5, 5, 95, 75), fill=(0, 255, 0, 255))     # 41 levels off in the card's colours
    ImageDraw.Draw(art).rectangle((30, 20, 70, 60), fill=(255, 0, 40, 255))
    out, box = pl.draw(pl.in_document(psd, np.asarray(art), None), corners, pl.PS_NEAREST)
    layer = PixelLayer.fromarray((out * 255 + 0.5).astype(np.uint8), psd, name="art", top=box[1], left=box[0])
    pl.link_pixels(layer, rel, path, (old := _png(art)), corners)
    edit = art.copy()
    ImageDraw.Draw(edit).rectangle((10, 30, 40, 50), fill=(250, 250, 0, 255))
    blobs = {pl.git_blob(data): data for data in (old, _png(edit))}
    changed, pixels, notes = pl.redraw_psd(psd, rel, {path: pl.git_blob(_png(edit))}, blobs.get,
                                           {path: {pl.git_blob(old): len(old)}})
    want, box = pl.draw(pl.in_document(psd, np.asarray(edit), None), corners, pl.PS_NEAREST)
    assert changed and pixels and not notes
    assert tuple(psd[1].bbox) == box and np.abs(psd[1].numpy() - want).max() <= 1 / 255


def test_a_linked_document_keeps_its_colours(clone):
    """A document in the card's own profile is placed as it is (through sRGB, 46 linked ones came back up to 56 levels
    off, effectsgaps 11-13); an untagged one is converted once, sRGB into the card's."""
    from psd_tools.constants import Resource
    from psd_tools.psd.image_resources import ImageResource
    card = PSDImage.open(clone / MINI_DOOMER)
    icc = card.image_resources.get_data(Resource.ICC_PROFILE)
    pixels = np.array([[[0, 255, 0, 255], [255, 0, 40, 255], [10, 40, 250, 255], [250, 240, 10, 255]]], np.uint8).repeat(3, 0)

    def document(profile) -> bytes:
        doc = PSDImage.new("RGB", (4, 3))
        PixelLayer.frompil(Image.fromarray(pixels, "RGBA"), doc, "art")
        if profile:
            doc.image_resources[Resource.ICC_PROFILE] = ImageResource(key=Resource.ICC_PROFILE, data=profile)
        out = BytesIO()
        doc.save(out)
        return out.getvalue()
    assert pe.foreign_profile(card) == icc and np.array_equal(pl.picture(card, document(icc), "psb"), pixels)
    untagged = pl.picture(card, document(None), "psb")
    assert np.array_equal(untagged, pl.in_document(card, pixels, None)) and not np.array_equal(untagged, pixels)


COPPER = "Items/Equip/1 Stars/Copper_Sword.psd"     # its art under one Gaussian Blur smart filter (6.6 px)


def test_gaussian_blur_redrawn(clone):
    """A lone Gaussian Blur is drawn again: Copper Sword's art from its own picture lands near Photoshop's pixels (not
    without the blur), its FEid source is the new picture unblurred; a painted filter mask refuses. blurred(): no halo
    on a side at the canvas's edge."""
    from psd_tools.compression import compress, decompress
    psd = PSDImage.open(clone / COPPER)
    layer = next(layer for layer in psd.descendants() if layer.kind == "smartobject" and pl._filtered(layer))
    assert pl.blur_radius(layer) == 6.6 and pl.why_not_redrawn(layer) is None
    mask, data = layer.smart_object.filter_item.extra, layer.smart_object.filter_item.extra.data
    top, left, bottom, right = mask.rectangle
    painted = np.full((bottom - top, right - left), 255, np.uint8)
    painted[:50, :50] = 0
    mask.data = compress(painted.tobytes(), mask.compression, right - left, bottom - top, 8, 2)
    assert pl.why_not_redrawn(layer) == "smart filters"
    mask.data = data

    def canvas(rgba, box):
        out = np.zeros((psd.height, psd.width, 4), np.float32)
        x0, y0, x1, y1 = max(box[0], 0), max(box[1], 0), min(box[2], psd.width), min(box[3], psd.height)
        part = rgba[y0 - box[1]:y1 - box[1], x0 - box[0]:x1 - box[0]]
        out[y0:y1, x0:x1] = np.dstack([part[..., :3] * part[..., 3:], part[..., 3]])
        return out
    stored = canvas(layer.numpy(), layer.bbox)
    picture = pl.picture(psd, layer.smart_object.data, "jpg")
    plain, box = pl.draw(picture, pl.corners_of(layer), Image.BICUBIC)
    pl.redraw(layer, picture, pl.corners_of(layer), Image.BICUBIC)
    error = lambda rgba, where: np.abs(canvas(rgba, where) - stored).mean() * 255
    assert error(layer.numpy(), layer.bbox) < 1 and error(plain, box) > 3          # 0.65 and 3.95 levels
    saved = BytesIO()
    psd.save(saved)
    again = PSDImage.open(BytesIO(saved.getvalue()))
    item = next(one for one in again.descendants() if one.name == layer.name).smart_object.filter_item
    top, left, bottom, right = item.rectangle
    source = np.dstack([np.frombuffer(decompress(item.channels[k].data, item.channels[k].compression, right - left,
                                                 bottom - top, 8, 2), np.uint8).reshape(bottom - top, right - left)
                        for k in (0, 1, 2, -1)])
    assert (left, top, right, bottom) == box and np.abs(source / 255 - plain).max() <= 1 / 255

    out, where = pl.blurred(np.ones((20, 20, 4), np.float32), (0, 10, 20, 30), 2.0, (100, 100))
    assert where == (0, 3, 27, 37) and out[17, 0, 3] == 1 and out[0, 0, 3] < 0.01


GUST = "Items/Support/2 Stars/Gust_of_Wind.psd"     # Croissant's names, stars and type icon under Motion Blurs (11-11)
GUST_REV = "aa8a491f5b1fdb78249f71c4cc95f7f7609425b9"


def test_motion_blur_redrawn(clone):
    """A lone Motion Blur at angle 0 of a proven distance (6 px) is drawn as Photoshop 2023 stores it: Gust of Wind's names
    from their FEid sources land on Photoshop's pixels exactly; a 17 px one (its stars) is not proven: not redrawn."""
    from psd_tools.compression import decompress
    from conftest import at_rev
    psd = PSDImage.open(BytesIO(at_rev(clone, GUST_REV, GUST)))
    layers = {layer.name: layer for layer in psd.descendants() if layer.kind == "smartobject"}
    assert pl.motion_distance(layers["star"]) is None and pl.why_not_redrawn(layers["star"]) == "smart filters"
    for name in ("Name #1", "Name #2"):
        layer = layers[name]
        assert pl.motion_distance(layer) == 6 and pl.blur_radius(layer) is None and pl.why_not_redrawn(layer) is None
        item = layer.smart_object.filter_item
        top, left, bottom, right = item.rectangle
        source = np.dstack([np.frombuffer(decompress(item.channels[k].data, item.channels[k].compression, right - left,
                                                     bottom - top, 8, 2), np.uint8).reshape(bottom - top, right - left)
                            for k in (0, 1, 2, -1)])
        ys, xs = np.nonzero(source[..., 3])
        box = (left + int(xs.min()), top + int(ys.min()), left + int(xs.max()) + 1, top + int(ys.max()) + 1)
        part = source[box[1] - top:box[3] - top, box[0] - left:box[2] - left].astype(np.float32) / 255
        out, where = pl.motion_blurred(part, box, 6)
        assert where == tuple(layer.bbox)
        assert np.array_equal((out * 255 + 0.5).astype(np.uint8), np.round(layer.numpy() * 255).astype(np.uint8))


SALTBAKER = "Creatures/Cuphead/5 Stars/Chef_Saltbaker.psd"   # Croissant's bars: knockout groups in smart objects (11-11)
SALTBAKER_REV = "b3931ee781490fcffcb41cb45462398c96416a59"


def test_a_smart_objects_knockout_is_drawn_from_its_pixels(clone, monkeypatch):
    """kofix 11-11: Chef Saltbaker's bars are knockout groups (Transparency Shapes Layer off) inside smart objects; the
    card draws their stored pixels (the guards' PNG of cf0b0bd0 is Photoshop 2023's within 3 levels), so its own tree
    has nothing undrawable, while each bar's document has until the fork draws knockout (decode: None, not drawn)."""
    from conftest import at_rev
    monkeypatch.setattr(pl, "fork_draws_knockout", lambda: False)
    psd = PSDImage.open(BytesIO(at_rev(clone, SALTBAKER_REV, SALTBAKER)))
    assert pl.undrawable(psd) == []
    bars = [layer for layer in psd.descendants() if layer.kind == "smartobject" and layer.smart_object.kind == "data"]
    assert sorted(layer.name for layer in bars) == ["Attack", "Defense", "Health", "Speed"]
    for layer in bars:
        assert pl.undrawable(PSDImage.open(BytesIO(layer.smart_object.data))) == ["Group 1"]
        assert pl.decode(layer.smart_object.data, "psb") is None


def test_normalise_links_through_its_link_step(clone, tmp_path):
    """card_standards.normalise's link step (needs, link) on Delta through a LinkContext, as the guards and the sweep pass
    it: text normal, pictures linked, new files beside it, no merged image; a second normalise writes nothing."""
    import commands.card_standards as cs
    from types import SimpleNamespace
    from commands.card_guards import LinkContext
    rel = DELTA
    files = {str(p.relative_to(clone)): p.read_bytes() for p in (clone / "Types").rglob("*.png")}
    tree = {path: pl.git_blob(data) for path, data in files.items()}
    by_blob = {pl.git_blob(data): data for data in files.values()}
    psd = PSDImage.open(clone / rel)
    psd.repo_path = rel
    assert "embedded: Isaac" in pl.needs(psd) and "embedded: Isaac" in cs.needs_normalise(psd)
    types = pl.TypesIndex(files)
    catalogue = pl.assign([(rel, pl.items(psd, types.vocab()))], types, {})
    pl.fallback_names(catalogue, {})
    out = tmp_path / "out.psd"
    done = cs.normalise(str(clone / rel), str(out), SimpleNamespace(links=LinkContext(tree, catalogue, by_blob.get)), rel)
    assert "Isaac" in done and done.files and all(path.startswith("MDW/Linked/Delta - ") for path in done.files)
    assert not done.left and not done.notes
    new = PSDImage.open(out)
    new.repo_path = rel
    assert not new.has_preview() and pl.needs(new) == []
    tree.update({path: pl.git_blob(data) for path, data in done.files.items()})
    by_blob.update({pl.git_blob(data): data for data in done.files.values()})
    again = cs.normalise(str(out), str(tmp_path / "again.psd"), SimpleNamespace(links=LinkContext(tree, catalogue, by_blob.get)), rel)
    assert list(again) == [] and not (tmp_path / "again.psd").exists()


def test_fills_stay_embedded(clone):
    """A smart object holding a fill, an adjustment or a gradient effect stays embedded (owner 11-08): no item, no need;
    the card's other pictures still link."""
    psd = PSDImage.open(clone / FREEZER)
    psd.repo_path = FREEZER
    found = pl.needs(psd)
    assert "embedded: Gradient Fill 1" not in found and "embedded: Patrick Star" in found
    assert "Gradient Fill 1" not in {layer["name"] for item in pl.items(psd) for layer in item.get("layers", [])}


def test_tree_at_non_ascii_paths(tmp_path):
    """_tree_at reads paths git would quote (Cyrillic, an em dash) as they are."""
    repo = tmp_path / "cards"
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True).stdout
    (repo / "MDW/Linked").mkdir(parents=True)
    git("init", "-q")
    names = ["MDW/Linked/A - Игры.jpg", "MDW/Linked/B - —Pngtree—.png", "MDW/Linked/plain.png"]
    for name in names:
        (repo / name).write_bytes(name.encode())
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    assert sorted(pl._tree_at(str(repo), "HEAD")) == sorted(names)


BLAZING = "Auxiliary/Status Effects/Debuffs/2 Stars/Blazing"     # [fire] in its Ability, a Stroke on it


def test_icon_glyphs_redrawn_after_a_font_commit(clone, tmp_path, monkeypatch):
    """The stoogedings workflow's font commit (icon_fire's picture recoloured): Card sync's redraw finds the card set in
    it, a bot rewrite since aside, splices that glyph alone and re-exports its PNG (Card-sweep: glyphs); its own commit
    ends it. A refusal is a hand flag at the PSD's blob that keeps it from being due; a person's save ends it too."""
    import base64
    import os
    import re
    from fontTools.ttLib import TTFont
    from psd_tools.composite.text import capture
    from psd_tools.composite.text.splice import _glyph_box
    import commands.card_standards as cs
    from commands import psd_glyphs
    from commands.font_check import layer_fonts
    from commands.psd_details import layer_path
    from commands.psd_edit import Refused, save_psd
    from conftest import CARDS_REV, at_rev
    rel, png, font = BLAZING + ".psd", BLAZING + ".png", psd_glyphs.FONT_FILE
    repo, fonts = tmp_path / "cards", tmp_path / "fonts"
    git = lambda *a, who="t": subprocess.run(["git", "-C", str(repo), "-c", f"user.name={who}", "-c", "user.email=t@t", *a],
                                             check=True, capture_output=True, text=True).stdout
    for path in (rel, png, font):
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_bytes(at_rev(clone, CARDS_REV, path))
    (repo / "site").mkdir()
    (repo / "site/Card Edits.json").write_text('{\n  "cards": {}\n}\n')
    git("init", "-q")
    git("add", "-A")
    git("commit", "-qm", "base")
    psd = PSDImage.open(repo / rel)
    texts = [layer for layer in psd.descendants() if layer.kind == "type"]
    details = {"cards": {pl._tree_at(str(repo), "HEAD")[rel]: {"path": rel, "text": [
        {"layer": layer_path(t), "visible": t.is_visible(), "text": t.text} for t in texts],
        "fonts": {f: {"layers": [layer_path(t) for t in texts if f in layer_fonts(t)]} for t in texts for f in layer_fonts(t)}}}}
    old = TTFont(repo / font)                               # icon_fire's picture, colours inverted
    gid = old.getGlyphOrder().index("icon_fire")
    doc = next(d for d in old["SVG "].docList if d.startGlyphID <= gid <= d.endGlyphID)
    head, data = re.search(r'(href="data:image/png;base64,)([^"]+)', doc.data).groups()
    picture = Image.open(BytesIO(base64.b64decode(data))).convert("RGBA")
    pixels = np.asarray(picture).copy()
    pixels[..., :3] = 255 - pixels[..., :3]
    doc.data = doc.data.replace(head + data, head + base64.b64encode(_png(Image.fromarray(pixels))).decode())
    old.save(repo / font)
    git("commit", "-qam", "Font: stoogedings rebuilt for fire.png", who=psd_glyphs.BOT)
    psd[0].name += " x"                                     # a bot rewrite since: its text pixels kept
    save_psd(psd, str(repo / rel))
    git("commit", "-qam", "Card sync: 1 card baked", who=psd_glyphs.BOT)
    fonts.mkdir()
    for name in os.listdir(clone / "MDW" / "Fonts"):
        if name.lower().endswith((".ttf", ".otf", ".ttc")) and name != "stoogedings.ttf":
            (fonts / name).symlink_to(clone / "MDW" / "Fonts" / name)
    (fonts / "stoogedings.ttf").write_bytes((repo / font).read_bytes())
    tree = pl._tree_at(str(repo), "HEAD")
    assert psd_glyphs.font_commits(str(repo)) == [git("rev-parse", "HEAD~1").strip()]
    assert list(psd_glyphs.due(str(repo), "HEAD", tree, details)) == [rel]

    found = pl.redraw_linked(str(repo), {"files": {}}, details=details, fonts_dir=str(fonts))
    assert found.glyphs == [rel] and sorted(found.files) == [png, rel] and not found.flagged and not found.notes
    assert pl.redraw_message(found).startswith("Card sync: icon glyphs redrawn\n") and pl.redraw_message(found).endswith("\nCard-sweep: glyphs")
    new = PSDImage.open(BytesIO(found.files[rel]))
    layer = cs.icon_layers(new)[0]
    cap = capture(layer, cs._fonts_of(layer, lambda names: str(fonts)))
    x0, y0, x1, y1 = next(_glyph_box(g, cap.model.transform, 8) for g in cap.layout.glyphs        # 8: its Stroke
                          if cs._icon_face(g.face) and g.face.hb.get_glyph_name(g.gid) == "icon_fire")
    before = np.asarray(Image.open(repo / png).convert("RGB"), np.int16)
    after = Image.open(BytesIO(found.files[png]))
    moved = np.abs(np.asarray(after.convert("RGB"), np.int16) - before).max(axis=2) > png_export.TOL
    inside = np.zeros(moved.shape, bool)
    inside[y0:y1, x0:x1] = True
    assert moved.sum() > 1000 and not (moved & ~inside).any()
    assert png_export.png_check(new, after)["ok"] is True

    flagged = pl.redraw_linked(str(repo), {"files": {}}, details=details, fonts_dir=str(fonts),
                               cards=0)             # capped: still due, nothing written
    assert flagged.due == [rel] and not flagged.files
    monkeypatch.setattr(psd_glyphs, "redraw", lambda *a: (_ for _ in ()).throw(Refused("unsupported-card", "Ability lays out otherwise")))
    refused = pl.redraw_linked(str(repo), {"files": {}}, details=details, fonts_dir=str(fonts))
    edits = refused.files.pop("site/Card Edits.json")
    assert not refused.files and refused.flagged == {rel: psd_glyphs.NOTE.format(why="Ability lays out otherwise")}
    assert json.loads(edits)["cards"]["Blazing"]["flags"] == {"hand": refused.flagged[rel], "psd": tree[rel][:8]}
    assert psd_glyphs.due(str(repo), "HEAD", tree, details, edits) == {}
    theirs = json.dumps({"cards": {"Blazing": {"flags": {"hand": "x. " * 99, "psd": tree[rel][:8]}}}})
    hand = json.loads(psd_glyphs.flag(theirs, rel, tree[rel], refused.flagged[rel]))["cards"]["Blazing"]["flags"]["hand"]
    assert hand.startswith(refused.flagged[rel]) and len(hand) <= 300                # a person's long reason: ours kept

    for path, data in found.files.items():                  # its own commit, then a person's save: not due
        (repo / path).write_bytes(data)
    git("commit", "-qam", pl.redraw_message(found), who=psd_glyphs.BOT)
    assert psd_glyphs.due(str(repo), "HEAD", pl._tree_at(str(repo), "HEAD"), details) == {}
    git("reset", "-q", "--hard", "HEAD~1")
    psd[0].name += "y"
    save_psd(psd, str(repo / rel))
    git("commit", "-qam", "Blazing: saved")
    assert psd_glyphs.due(str(repo), "HEAD", pl._tree_at(str(repo), "HEAD"), details) == {}
    git("reset", "-q", "--hard", "HEAD~1")                  # a person's move alone: still due, at its new path
    git("mv", str(repo / rel), str(repo / "Blazing.psd"))
    git("commit", "-qm", "Blazing: moved")
    assert list(psd_glyphs.due(str(repo), "HEAD", pl._tree_at(str(repo), "HEAD"), details)) == ["Blazing.psd"]
