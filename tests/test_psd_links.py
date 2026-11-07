"""
psd_links (the big pass, PLAN §3 A4): new files' names and the catalogue's classes, the trim map on Star of beth, links
placed from any depth and from absolute URLs (with the guards' housekeeping), and Card sync's redraw of one MDW/Linked
file with two users.
    ~/.venvs/ac-py314-fork/bin/python -m pytest -q tests/test_psd_links.py
"""
import subprocess
from io import BytesIO

import numpy as np
import pytest
from PIL import Image, ImageDraw
from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer
from psd_tools.api.links import link_content_id

import commands.png_export as png_export
import commands.psd_links as pl

DELTA = "Auxiliary/Minions/Delta.psd"            # three Star of beth items (Croissant trimmed the file, 11-07)


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
    Types bytes, ink, an older version, a name in its row (else a new file), an existing file's bytes; first user names."""
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
    art = _png(_art())

    def entry(data, name, row=False):
        rgba = np.asarray(Image.open(BytesIO(data)).convert("RGBA"))
        return {"uuid": name + str(row), "kind": "data", "blob": pl.git_blob(data), "ext": "png", "file": "", "size": len(data),
                "key": pl.ink_key(rgba), "layers": [{"name": name, "row": row}]}
    psds = [("Items/A.psd", [entry(files["Types/Stars/star.png"], "Star"), entry(_png(fire), "fire"),
                             entry(_png(old_fire), "Fire"), entry(_png(star.resize((20, 20))), "star", True),
                             entry(_png(star.resize((20, 20))), "star"), entry(art, "Art")]),
            ("Items/B.psd", [entry(art, "Other"), entry(_png(Image.new("RGBA", (5, 5), (1, 2, 3, 255))), "cain")])]
    linked = {"MDW/Linked/cain.jpeg": pl.git_blob(_png(Image.new("RGBA", (5, 5), (1, 2, 3, 255))))}
    found = pl.assign(psds, types, linked)
    pl.fallback_names(found, linked)
    rows = found["psds"]
    assert [(r["class"], r["target"]) for r in rows["Items/A.psd"]] == [
        (pl.BYTES, "Types/Stars/star.png"), (pl.INK, "Types/fire.png"), (pl.REDRAW, "Types/fire.png"),
        (pl.REDRAW, "Types/Stars/star.png"), (pl.NEW, "MDW/Linked/A - star.png"), (pl.NEW, "MDW/Linked/A - Art.png")]
    assert [(r["class"], r["target"]) for r in rows["Items/B.psd"]] == [(pl.LINKED, "MDW/Linked/A - Art.png"),
                                                                       (pl.LINKED, "MDW/Linked/cain.jpeg")]
    assert rows["Items/A.psd"][3]["fallback"] == "MDW/Linked/A - star.png"     # its exact copy dedupes onto row 5's file
    assert found["files"]["MDW/Linked/A - Art.png"]["first"] == "Items/A.psd"


def test_trim_map_on_star_of_beth(clone):
    """Croissant's trim (the same ink, the margins cut): each of Delta's Star of beth items links the trimmed file, its
    raster kept, its quad moved so the file's ink draws where the old one did; the rest by bytes or as new files."""
    rel = DELTA
    if not (clone / rel).exists():
        pytest.skip(f"{rel} not in the clone")
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
    assert sorted(pl.CLASSES[r["class"]] for r in rows) == ["bytes", "ink", "ink", "ink", "linked", "new", "new"]
    done = pl.link_card(psd, rel, rows, lambda path: files.get(path))
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


def test_redraw_linked_one_file_two_users(tmp_path):
    """One MDW/Linked file, two users at two depths and scales (one PNG named in another case). A lossless re-save: their
    sizes and contentIDs only, no raster, no PNG. A pixel edit: both drawn again (the stored mode, bicubic), their PNGs
    re-exported to the new drawing."""
    repo = tmp_path / "cards"
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True).stdout
    repo.mkdir()
    git("init", "-q")
    art, users = _art(), {"Creatures/X/3 Stars/A.psd": (40, 60, 2.0), "Items/B.psd": (300, 200, 1.5)}
    pngs = {"Creatures/X/3 Stars/A.psd": "Creatures/X/3 Stars/A.png", "Items/B.psd": "Items/b.png"}
    data = _png(art, 9)
    (repo / "MDW/Linked").mkdir(parents=True)
    (repo / "MDW/Linked/art.png").write_bytes(data)
    for rel, (x, y, s) in users.items():
        psd = PSDImage.new("RGB", (600, 500))
        PixelLayer.fromarray(np.full((500, 600, 4), (240, 240, 230, 255), np.uint8), psd, name="bg")
        corners = [x, y, x + 100 * s, y, x + 100 * s, y + 80 * s, x, y + 80 * s]
        out, box = pl.draw(np.asarray(art), corners, Image.BICUBIC)
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
    done = pl.redraw_linked(str(repo), known, fetch=False)
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
    done = pl.redraw_linked(str(repo), known, fetch=False)
    assert sorted(done.files) == sorted([*users, *pngs.values()]) and not done.notes
    for rel, (x, y, s) in users.items():
        new = PSDImage.open(BytesIO(done.files[rel]))
        want, box = pl.draw(np.asarray(edit), [x, y, x + 100 * s, y, x + 100 * s, y + 80 * s, x, y + 80 * s], Image.BICUBIC)
        assert tuple(new[1].bbox) == box and np.abs(new[1].numpy() - want).max() <= 1 / 255 and new[1].smart_object.filesize == len(pixels)
        render = np.asarray(new.composite(ignore_preview=True).convert("RGB"), np.int16)
        png = np.asarray(Image.open(BytesIO(done.files[pngs[rel]])).convert("RGB"), np.int16)
        assert np.abs(png - render).max() <= png_export.TOL
