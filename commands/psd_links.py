"""
Linked pictures (the big pass, owner 2026-11-07): every picture smart object that is no stat bar links a file in the
card repo, stars and type icons Types/ (Types/Stars/), every other picture MDW/Linked/ ("<Card> - <layer>.<ext>", flat; a
file several PSDs share is named after the first). Each link carries the file's size and one fixed date (the fork's
LINK_DATE), as Photoshop 2023 writes it (lnkE v8: the fork's SmartObject.link), so Photoshop finds every link up to date
and keeps the stored pixels; stat bars stay embedded. A link is read as repo-path strings only (the fork's
SmartObject.link_target), never from the filesystem.

    TypesIndex(files, history)                       Types/ files by git blob, ink pixels and stem; older blobs
    items(psd, vocab) -> [dict]                      a PSD's picture items (the catalogue's worker)
    assign(psds, types, linked) -> dict              every item's class and target (PLAN 5.1), new files named
    link_card(psd, rel, entries, read) -> Linked     one PSD's pictures linked in place (5.2 step 4)
    link_pixels(layer, rel, path, data, corners)     a pixel layer made a smart object linking a Types/ file (Q4)
    place_trim(corners, old, new, old_box, new_box)  a file's quad when its ink moved (Croissant's trims, name matches)
    draw(rgba, corners, mode) -> (rgba, box)         a picture drawn on its quad as Photoshop places it
    redraw(layer, rgba, corners, mode)               a placed layer's pixels drawn again from a file
    links_of(psd, rel, tree) -> [Link]               a PSD's links as repo paths, repair() making them canonical
    link_issues(psd, rel, tree) -> [str]             what its links need (needs_normalise's cheap test)
    needs(psd) / link(psd, rel, fonts) -> Step       card_standards.normalise's link step (fonts.links: the rows, files)
    check(psd, rel, files, tree, read) -> [str]      K4 on one rewritten PSD (card_sweep)
    catalogue(source, tree, jobs, path) -> dict      card_sweep's catalogue file, checked against its jobs' blobs
    unused / duplicates / restore / repair           the guards' housekeeping (G2-G4)
    redraw_linked(...) -> Redrawn                    Card sync: a changed linked file drawn again in its users (C1-C6)

    python -m commands.psd_links catalogue --clone DIR --ref SHA --out FILE [--held LIST --held-ref REF --held-base SHA]
                                           [--recheck catalogue.json] [--history CLONE]
    python -m commands.psd_links redraw --checkout DIR --links FILE [--head SHA] [--cards N] [--bytes N] [--until SECONDS]
                                        [--out DIR]   (card-sync.yml: --local, --db, --deadline)
    python -m commands.psd_links due --clone DIR --db DIR [--head SHA]       prints how many PSDs are due

Classes (assign): 1 the bytes of a Types file; 2 its ink pixels (raster kept, placed by place_trim); 3 that file drawn
again (an older version of it, or a star or type icon named like it in its row); 4 the bytes of an MDW/Linked file; 5 a
new MDW/Linked file. Light imports: card-guards.yml and card-sync.yml check this file out with the card modules.
No Discord, network or config.
"""
from argparse import ArgumentParser
from dataclasses import dataclass, field
from hashlib import sha1
from io import BytesIO
from json import dump, dumps, load
from math import floor
from os import environ, makedirs
from os.path import expanduser, isdir
from posixpath import basename, dirname, join as posixpath_join, splitext
from re import IGNORECASE, compile as re_compile, sub
from subprocess import PIPE, Popen, run as run_process
from threading import Thread
from time import time
from typing import Callable, Optional
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.links import FILETYPES, LINK_DATE, comp_info, content_descriptor, link_content_id, linked_file, place
from psd_tools.constants import LinkedLayerType, Tag
from psd_tools.psd.descriptor import Descriptor, DescriptorBlock, Double, Enumerated, Integer, List, String, UnitFloat
from psd_tools.psd.linked_layer import LinkedLayer, LinkedLayers
from psd_tools.psd.tagged_blocks import SmartObjectLayerData, TaggedBlock
from psd_tools.terminology import Unit

from commands.psd_regions import _ancestors, badge_layers, comp_names, placed_data
from global_config import HISTORY_CLONE, STARS_DIR, TYPES_DIR

# ========================
# Configuration
# ========================
LINKED_DIR = "MDW/Linked"
INK_ALPHA = 32                  # place_trim: ink boxes over this alpha (P2: faint alpha moved Equip.png by 20 px)
NAME_MAX = 119                  # a new file's name, characters (under 120)
UNSAFE = re_compile(r'[\\/:*?"<>|\x00-\x1f\x7f]')
MAGIC = ((b"\x89PNG", "png"), (b"\xff\xd8\xff", "jpg"), (b"GIF8", "gif"), (b"8BPS\x00\x01", "psd"),
         (b"8BPS\x00\x02", "psb"), (b"II*\x00", "tif"), (b"MM\x00*", "tif"), (b"%PDF", "pdf"))
RASTER = ("png", "jpg", "gif", "webp")      # decoded by PIL; psd and psb by the fork's composite
BYTES, INK, REDRAW, LINKED, NEW = 1, 2, 3, 4, 5
CLASSES = {BYTES: "bytes", INK: "ink", REDRAW: "redraw", LINKED: "linked", NEW: "new"}
MESH_TOL = 0.01                 # px: a warpCustom mesh this close to its quad's grid is no warp
QUAD_TOL = 0.01                 # px: corners this close make an upright rectangle
NOT_IN_REPO = "Links {name}, which is not in the repo: push that file."
NOT_REDRAWN = "{layer} links {name}, changed since: Card sync could not draw it again ({why}); update it in Photoshop."
REDRAW_SUBJECT = "Card sync: linked pictures redrawn"
SWEEP_TRAILER = "Card-sweep: linked"
CARDS_CAP = 60                  # Card sync redraws at most this many PSDs a run (C6; M7 sets it)


# ========================
# Helper Functions
# ========================
def git_blob(data: bytes) -> str:
    """The git blob sha of bytes."""
    return sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def file_type(data: bytes, filename: str = "") -> str:
    """A file's type by its magic bytes, else its name's extension ("" when neither tells)."""
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    for magic, kind in MAGIC:
        if data.startswith(magic):
            return kind
    ext = splitext(filename.strip("\x00"))[1].lower().lstrip(".")
    return {"jpeg": "jpg", "tiff": "tif"}.get(ext, ext)


def _text(value) -> str:
    value = getattr(value, "value", value) or ""
    return (value.decode("latin-1") if isinstance(value, bytes) else str(value)).replace("\x00", "")


def decode(data: bytes, kind: str) -> Optional[np.ndarray]:
    """A file's pixels as straight RGBA uint8 (psd and psb through the fork's composite), None when unreadable."""
    try:
        if kind in ("psd", "psb"):
            return np.asarray(PSDImage.open(BytesIO(data)).composite(ignore_preview=True).convert("RGBA"))
        with Image.open(BytesIO(data)) as image:
            return np.asarray(image.convert("RGBA"))
    except Exception:
        return None


def ink_box(rgba: np.ndarray, alpha: int = 1) -> Optional[tuple]:
    """The box of the pixels at alpha or more (x0, y0, x1, y1), None when there are none."""
    ys, xs = np.nonzero(rgba[..., 3] >= alpha)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def ink_key(rgba: Optional[np.ndarray]) -> Optional[tuple]:
    """Equal for two pictures with the same ink: RGBA over the alpha > 0 box, colour under alpha 0 cleared."""
    box = ink_box(rgba) if rgba is not None else None
    if box is None:
        return None
    crop = rgba[box[1]:box[3], box[0]:box[2]].copy()
    crop[crop[..., 3] == 0] = 0
    return crop.shape[1], crop.shape[0], sha1(crop.tobytes()).hexdigest()


def stem_key(name: str) -> str:
    """A layer or file name as names match: no extension or ' copy N', letters and digits only."""
    name = sub(r"\.(png|jpe?g|gif|webp|psb|psd)$", "", name.strip(), flags=IGNORECASE)
    return sub(r"[^a-z0-9]+", "", sub(r"\s+copy(\s+\d+)?$", "", name, flags=IGNORECASE).lower())


def file_name(stem: str, layer: str, ext: str, taken: set) -> str:
    """
    A new MDW/Linked file's name, "<PSD stem> - <layer>.<ext>": no `/ \\ : * ? " < > |` or control characters, no
    trailing dot or space, under 120 characters, unique case-folded against taken (" (2)" and on added); taken gets it.
    """
    def clean(text: str) -> str:
        return sub(r"\s+", " ", UNSAFE.sub("_", text)).strip().rstrip(". ") or "layer"
    base = f"{clean(stem)} - {clean(layer)}"
    n = 1
    while True:
        tail = f" ({n})" if n > 1 else ""
        room = NAME_MAX - len(tail) - len(ext) - 1
        name = f"{base[:room].rstrip('. ')}{tail}.{ext}"
        if name.casefold() not in taken:
            taken.add(name.casefold())
            return name
        n += 1


def _bar(placed) -> bool:
    """A stat bar: a smart object whose layer comps are all numbers (psd_regions.comp_bars' test). It stays embedded."""
    names = comp_names(placed) if placed is not None else {}
    return bool(names) and all(name.isdigit() for name in names.values())


def _path(layer) -> str:
    """The groups a layer sits in, outermost first ("Stars/5 stars")."""
    return "/".join(group.name for group in reversed(list(_ancestors(layer))))


def _items_of(psd: PSDImage) -> dict:
    """{uuid: (tag, LinkedLayer)} of the document's lnk2/lnk3/lnkE items."""
    out, blocks = {}, psd._record.layer_and_mask_information.tagged_blocks
    for tag in (Tag.LINKED_LAYER1, Tag.LINKED_LAYER2, Tag.LINKED_LAYER3, Tag.LINKED_LAYER_EXTERNAL):
        for item in (blocks.get_data(tag) or []) if blocks is not None and tag in blocks else []:
            out[_text(item.uuid)] = (tag, item)
    return out


def _uid(layer) -> Optional[str]:
    placed = placed_data(layer)
    return _text(placed.get(b"Idnt")) if placed is not None and b"Idnt" in placed else None


def corners_of(layer) -> list:
    """A placed layer's quad: its Trnf corners TL TR BR BL (8 floats)."""
    return [float(v) for v in placed_data(layer).get(b"Trnf") or []]


def _size_of(layer) -> tuple:
    size = placed_data(layer).get(b"Sz  ") or {}
    return float(size.get(b"Wdth", 0)), float(size.get(b"Hght", 0))


def upright(corners: list) -> bool:
    """The quad is an upright rectangle (no turn, skew or perspective)."""
    x0, y0, x1, y1, x2, y2, x3, y3 = corners
    return (abs(y0 - y1) < QUAD_TOL and abs(y2 - y3) < QUAD_TOL and abs(x0 - x3) < QUAD_TOL and abs(x1 - x2) < QUAD_TOL
            and x1 > x0 and y3 > y0)


def snapped(corners: list) -> list:
    """An upright quad as Photoshop's Update places a file of a new pixel size (S0, 11-07): its size rounded to whole
    pixels, then its corner round the same centre."""
    x0, y0, x1, _, _, y1, _, _ = corners
    w, h = floor(x1 - x0 + 0.5), floor(y1 - y0 + 0.5)
    left, top = floor((x0 + x1 - w) / 2 + 0.5), floor((y0 + y1 - h) / 2 + 0.5)
    return [float(v) for v in (left, top, left + w, top, left + w, top + h, left, top + h)]


def warped(layer) -> bool:
    """A real warp (PLAN M4): a preset warp with a bend, or a warpCustom mesh off its quad's own grid. Photoshop stores
    warpCustom on plain placements too."""
    warp = (placed_data(layer) or {}).get(b"warp") or {}
    style = getattr(warp.get(b"warpStyle"), "enum", b"warpNone")
    if style == b"warpNone":
        return False
    if style != b"warpCustom":
        return any(abs(float(warp.get(key, 0.0))) > 1e-9 for key in (b"warpValue", b"warpPerspective", b"warpPerspectiveOther"))
    mesh = ((warp.get(b"customEnvelopeWarp") or {}).get(b"meshPoints")) or {}
    bounds = warp.get(b"bounds") or {}
    xs, ys = mesh.get(b"Hrzn"), mesh.get(b"Vrtc")
    if xs is None or ys is None or not bounds:
        return True
    left, top = float(bounds[b"Left"]), float(bounds[b"Top "])
    width, height = float(bounds[b"Rght"]) - left, float(bounds[b"Btom"]) - top
    u, v = int(warp.get(b"uOrder", 4)), int(warp.get(b"vOrder", 4))
    xs, ys = [float(x) for x in xs.values], [float(y) for y in ys.values]
    if len(xs) != u * v or len(ys) != u * v:
        return True
    return any(abs(xs[r * u + c] - (left + c * width / (u - 1))) > MESH_TOL or
               abs(ys[r * u + c] - (top + r * height / (v - 1))) > MESH_TOL for r in range(v) for c in range(u))


def _filtered(layer) -> bool:
    """Smart filters: their FEid source would show the old file after a redraw (PLAN R7)."""
    blocks = layer._record.tagged_blocks
    return any(key in blocks for key in (Tag.FILTER_EFFECTS1, Tag.FILTER_EFFECTS2, Tag.FILTER_EFFECTS3)) or \
        bool((placed_data(layer) or {}).get(b"filterFX"))


def why_not_redrawn(layer) -> Optional[str]:
    """Why a placed layer's pixels cannot be drawn again from its file, else None."""
    if warped(layer):
        return "a warp"
    if _filtered(layer):
        return "smart filters"
    corners = corners_of(layer)
    if len(corners) != 8 or abs(_area(corners)) < 1.0:
        return "a placement the redraw cannot reproduce"
    return None


def _area(corners: list) -> float:
    xs, ys = corners[0::2], corners[1::2]
    return 0.5 * sum(xs[i] * ys[(i + 1) % 4] - xs[(i + 1) % 4] * ys[i] for i in range(4))


# ========================
# Placement
# ========================
def homography(width: float, height: float, corners: list) -> np.ndarray:
    """3x3 H taking content (0,0) (w,0) (w,h) (0,h) to the quad's corners TL TR BR BL."""
    src = ((0.0, 0.0), (width, 0.0), (width, height), (0.0, height))
    rows, rhs = [], []
    for (u, v), x, y in zip(src, corners[0::2], corners[1::2]):
        rows += [[u, v, 1, 0, 0, 0, -u * x, -v * x], [0, 0, 0, u, v, 1, -u * y, -v * y]]
        rhs += [x, y]
    h = np.linalg.solve(np.array(rows, float), np.array(rhs, float))
    return np.append(h, 1.0).reshape(3, 3)


def _apply(h: np.ndarray, u: float, v: float) -> tuple:
    x, y, w = h @ np.array([u, v, 1.0])
    return x / w, y / w


def place_trim(corners: list, old_size: tuple, new_size: tuple, old_box: tuple, new_box: tuple) -> list:
    """
    The quad a new file takes so its ink lands where the old content's did: new pixel (u, v) is old content
    (ox0 + (u - nx0) * sx, oy0 + (v - ny0) * sy), sx and sy the ink boxes' ratio (1 for a trim), mapped through the
    homography of the old content onto its quad (exact on parallelograms and on perspective quads).

    Args:
        corners: The layer's Trnf (TL TR BR BL)
        old_size, new_size: (width, height) of the old content (Sz) and of the new file
        old_box, new_box: Their ink boxes (x0, y0, x1, y1) in their own pixels

    Returns:
        The new corners (8 floats)
    """
    h = homography(old_size[0], old_size[1], corners)
    sx = (old_box[2] - old_box[0]) / (new_box[2] - new_box[0])
    sy = (old_box[3] - old_box[1]) / (new_box[3] - new_box[1])
    w, ht = new_size
    out = []
    for u, v in ((0.0, 0.0), (w, 0.0), (w, ht), (0.0, ht)):
        out += _apply(h, old_box[0] + (u - new_box[0]) * sx, old_box[1] + (v - new_box[1]) * sy)
    return [float(c) for c in out]


def draw(rgba: np.ndarray, corners: list, mode: int) -> tuple:
    """
    A picture drawn on its quad as a placed layer's pixels: premultiplied, resampled by mode (Image.NEAREST, BILINEAR,
    BICUBIC; an upright rectangle by resize over its exact box, any other quad by the inverse homography).

    Returns:
        (straight RGBA float32 over the drawn pixels, their canvas box), or (None, (0, 0, 0, 0)) when nothing shows
    """
    src = rgba.astype(np.float32) / 255.0
    pre = np.dstack([src[..., :3] * src[..., 3:], src[..., 3]])
    height, width = pre.shape[:2]
    xs, ys = corners[0::2], corners[1::2]
    x0, y0, x1, y1 = int(np.floor(min(xs))), int(np.floor(min(ys))), int(np.ceil(max(xs))), int(np.ceil(max(ys)))
    if x1 <= x0 or y1 <= y0:
        return None, (0, 0, 0, 0)
    size = (x1 - x0, y1 - y0)
    sx, sy = (corners[2] - corners[0]) / width, (corners[5] - corners[1]) / height
    margin = 8 + (int(np.ceil(max(1 / abs(sx), 1 / abs(sy)))) if upright(corners) else 0)
    padded = np.zeros((height + 2 * margin, width + 2 * margin, 4), np.float32)
    padded[margin:margin + height, margin:margin + width] = pre
    if upright(corners):
        box = ((x0 - corners[0]) / sx + margin, (y0 - corners[1]) / sy + margin,
               (x1 - corners[0]) / sx + margin, (y1 - corners[1]) / sy + margin)
        out = np.dstack([np.asarray(Image.fromarray(np.ascontiguousarray(padded[..., k]), "F").resize(size, mode, box=box))
                         for k in range(4)])
    else:
        inverse = np.linalg.inv(homography(width, height, corners))
        shift = np.array([[1, 0, margin], [0, 1, margin], [0, 0, 1]], float) @ inverse @ \
            np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], float)
        shift = shift / shift[2, 2]
        coeffs = tuple(shift.flatten()[:8])
        out = np.dstack([np.asarray(Image.fromarray(np.ascontiguousarray(padded[..., k]), "F").transform(
            size, Image.Transform.PERSPECTIVE, coeffs, mode)) for k in range(4)])
    alpha = np.clip(out[..., 3], 0.0, 1.0)
    rgb = np.where(alpha[..., None] > 0, np.clip(out[..., :3], 0.0, None) / np.maximum(alpha, 1e-6)[..., None], 0.0)
    ys_, xs_ = np.nonzero(np.round(alpha * 255) > 0)
    if not len(xs_):
        return None, (0, 0, 0, 0)
    c0, c1, r0, r1 = int(xs_.min()), int(xs_.max()) + 1, int(ys_.min()), int(ys_.max()) + 1
    straight = np.dstack([np.clip(rgb, 0.0, 1.0), alpha])[r0:r1, c0:c1].astype(np.float32)
    return straight, (x0 + c0, y0 + r0, x0 + c1, y0 + r1)


def in_document(psd: PSDImage, rgba: np.ndarray, profile: Optional[bytes]) -> np.ndarray:
    """A file's pixels in the card's colours, as Photoshop places them: its own profile (else sRGB) into the card's
    when either is not sRGB (psd_edit._make_icon's rule)."""
    from commands.psd_edit import foreign_profile
    target = foreign_profile(psd)
    if target is None and not profile:
        return rgba
    from PIL import ImageCms
    def cms(data): return ImageCms.ImageCmsProfile(BytesIO(data)) if data else ImageCms.createProfile("sRGB")
    image = Image.fromarray(rgba, "RGBA")
    out = ImageCms.profileToProfile(image.convert("RGB"), cms(profile), cms(target), outputMode="RGB")
    out.putalpha(image.getchannel("A"))
    return np.asarray(out)


def picture(psd: PSDImage, data: bytes, kind: str) -> Optional[np.ndarray]:
    """A linked file's pixels in the card's colours (decode, then in_document), None when unreadable."""
    rgba = decode(data, kind)
    if rgba is None:
        return None
    profile = None
    if kind in RASTER:
        try:
            with Image.open(BytesIO(data)) as image:
                profile = image.info.get("icc_profile")
        except Exception:
            profile = None
    return in_document(psd, rgba, profile)


def redraw(layer, rgba: np.ndarray, corners: list, mode: int) -> tuple:
    """
    A placed layer's pixels drawn again from a file (rgba: its pixels in the card's colours) on corners at mode, its
    mask and effects kept (the link's Trnf and Sz are written by SmartObject.link or relink). Returns the canvas box
    drawn (old and new pixels' union).
    """
    from psd_tools.api.layers import TypeLayer      # the fork's raster writer: channels and rect, the record kept
    before = tuple(layer.bbox)
    out, box = draw(rgba, corners, mode)
    TypeLayer._store_raster(layer, out if out is not None else np.zeros((0, 0, 4), np.float32), box)
    layer._invalidate_bbox()
    boxes = [b for b in (before, box) if b[2] > b[0] and b[3] > b[1]]
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)) \
        if boxes else (0, 0, 0, 0)


# ========================
# Link records (the fork's SmartObject.link, relink and psd_tools.api.links)
# ========================
def canonical(rel: str, path: str) -> dict:
    """A link's path strings as the fork writes them: relPath from the PSD's folder, fullPath = originalPath =
    LINK_ROOT + the repo path."""
    made = linked_file(path, rel, basename(path))
    return {key: _text(made[key.encode()]) for key in ("relPath", "fullPath", "originalPath")}


def _layer_of(psd: PSDImage, uuid: str):
    return next(layer for layer in psd.descendants() if layer.kind == "smartobject" and _uid(layer) == uuid)


def link_item(psd: PSDImage, uuid: str, rel: str, path: str, data: bytes, quads: Optional[dict] = None,
              pixels: Optional[tuple] = None) -> None:
    """An embedded item made a link to path (a repo file of these bytes) by the fork (SmartObject.link): every layer
    sharing it follows; quads: {old corners: new corners} per layer, pixels the file's (width, height) when not the
    embedded file's."""
    corners = (lambda old: quads[tuple(old)]) if quads else None
    _layer_of(psd, uuid).smart_object.link(path, len(data), blob=git_blob(data), psd_path=rel, pixels=pixels,
                                           corners=corners)


def relink(psd: PSDImage, uuid: str, rel: str, path: str, size: int, blob: str, quads: Optional[dict] = None,
           pixels: Optional[tuple] = None) -> None:
    """A link pointed at a file (G3's duplicate merge) or given its new size (Card sync): strings, size, contentID,
    and per layer its new corners (quads) and Sz (pixels)."""
    corners = (lambda old: quads[tuple(old)]) if quads else None
    _layer_of(psd, uuid).smart_object.relink(path, size, blob=blob, psd_path=rel, pixels=pixels, corners=corners)


def _new_item(uuid: str, rel: str, path: str, data: bytes) -> LinkedLayer:
    """A new lnkE item linking path, as the fork's SmartObject.link writes one (link_pixels: no embedded item to turn)."""
    return LinkedLayer(kind=LinkedLayerType.EXTERNAL, version=8, uuid=uuid, filename=basename(path) + "\x00",
                       filetype=FILETYPES.get(file_type(data, path), b"\x00\x00\x00\x00"), creator=b"\x00\x00\x00\x00",
                       filesize=len(data), open_file=comp_info(), linked_file=linked_file(path, rel, basename(path)),
                       timestamp=LINK_DATE, data=b"", child_id="\x00", mod_time=0.0, lock_state=0,
                       content_descriptor=content_descriptor(link_content_id(git_blob(data), rel, uuid)))


def _add_item(psd: PSDImage, item: LinkedLayer) -> None:
    """item appended to the document's lnkE (a new lnkE after the last lnk2/lnk3 block, where Photoshop writes it)."""
    blocks = psd._record.layer_and_mask_information.tagged_blocks
    if Tag.LINKED_LAYER_EXTERNAL in blocks:
        blocks.get_data(Tag.LINKED_LAYER_EXTERNAL).append(item)
        return
    lists = (Tag.LINKED_LAYER1, Tag.LINKED_LAYER2, Tag.LINKED_LAYER3)
    items = list(blocks.items())
    at = max((i + 1 for i, (key, _) in enumerate(items) if key in lists), default=len(items))
    items.insert(at, (Tag.LINKED_LAYER_EXTERNAL, TaggedBlock(signature=b"8BIM", key=Tag.LINKED_LAYER_EXTERNAL,
                                                            data=LinkedLayers([item]))))
    blocks.clear()
    for key, value in items:
        blocks[key] = value


def _block(items: dict, name: str = "\x00", cls: bytes = b"null", block: bool = True):
    out = (DescriptorBlock if block else Descriptor)(name=name, classID=cls)
    for key, value in items.items():
        out[key] = value
    return out


def _placed_block(uuid: str, placed: str, corners: list, size: tuple) -> DescriptorBlock:
    """A new linked layer's SoLE descriptor (Photoshop 2023's for a placed PNG, no warp)."""
    w, h = (float(v) for v in size)
    frames = lambda: _block({b"numerator": Integer(0), b"denominator": Integer(600)}, block=False)
    bounds = _block({b"Top ": Double(0.0), b"Left": Double(0.0), b"Btom": Double(h), b"Rght": Double(w)},
                    cls=b"classFloatRect", block=False)
    warp = _block({b"warpStyle": Enumerated(typeID=b"warpStyle", enum=b"warpNone"), b"warpValue": Double(0.0),
                   b"warpPerspective": Double(0.0), b"warpPerspectiveOther": Double(0.0),
                   b"warpRotate": Enumerated(typeID=b"Ornt", enum=b"Hrzn"), b"bounds": bounds,
                   b"uOrder": Integer(4), b"vOrder": Integer(4)}, cls=b"warp", block=False)
    return _block({b"Idnt": String(uuid + "\x00"), b"placed": String(placed + "\x00"), b"PgNm": Integer(1),
                   b"totalPages": Integer(1), b"Crop": Integer(1), b"frameStep": frames(), b"duration": frames(),
                   b"frameCount": Integer(1), b"Annt": Integer(16), b"Type": Integer(2),
                   b"Trnf": List([Double(float(c)) for c in corners]),
                   b"nonAffineTransform": List([Double(float(c)) for c in corners]), b"warp": warp,
                   b"Sz  ": _block({b"Wdth": Double(w), b"Hght": Double(h)}, cls=b"Pnt ", block=False),
                   b"Rslt": UnitFloat(unit=Unit.Density, value=72.0), b"comp": Integer(-1),
                   b"compInfo": _block({b"compID": Integer(-1), b"originalCompID": Integer(-1)}, block=False),
                   b"ClMg": _block({b"placedLayerOCIOConversion": Enumerated(typeID=b"placedLayerOCIOConversion",
                                                                             enum=b"placedLayerOCIOConvertEmbedded")},
                                   cls=b"ClMg", block=False)})


def link_pixels(layer, rel: str, path: str, data: bytes, corners: list, salt: str = ""):
    """
    A pixel layer made a smart object linking path (a Types/ file of these bytes) placed on corners, its pixels, mask,
    effects and blending kept (the bake's type icons, Q4; a header star or icon the same ink as a Types file). The uuid
    and placed ID come from rel, path, the layer id and salt, so the same edit makes the same bytes.

    Returns:
        The SmartObjectLayer that took the layer's place in its group
    """
    from psd_tools.api.layers import SmartObjectLayer
    psd, parent = layer._psd, layer.parent
    rgba = decode(data, file_type(data, path))
    if rgba is None:
        raise ValueError(f"{path}: not a picture")
    seed = f"{rel}|{path}|{layer.layer_id}|{salt}"
    uuid, placed = str(uuid5(NAMESPACE_URL, "item|" + seed)), str(uuid5(NAMESPACE_URL, "placed|" + seed))
    record = layer._record
    record.tagged_blocks[Tag.SMART_OBJECT_LAYER_DATA2] = TaggedBlock(
        signature=b"8BIM", key=Tag.SMART_OBJECT_LAYER_DATA2,
        data=SmartObjectLayerData(kind=b"soLD", version=4, data=_placed_block(uuid, placed, corners,
                                                                              (rgba.shape[1], rgba.shape[0]))))
    _add_item(psd, _new_item(uuid, rel, path, data))
    made = SmartObjectLayer(parent, record, layer._channels)
    parent._layers[parent._layers.index(layer)] = made
    psd.mark_updated()
    return made


def pixel_corners(layer, rgba: np.ndarray) -> Optional[list]:
    """The quad on which a file (rgba) draws a pixel layer's own ink pixel for pixel, when their ink is the same, else
    None."""
    own = layer.numpy()
    if own is None or own.shape[2] < 4:
        return None
    own = np.clip(np.round(own * 255), 0, 255).astype(np.uint8)
    if ink_key(own) != ink_key(rgba):
        return None
    mine, theirs = ink_box(own), ink_box(rgba)
    left, top = layer.bbox[0] + mine[0] - theirs[0], layer.bbox[1] + mine[1] - theirs[1]
    w, h = rgba.shape[1], rgba.shape[0]
    return [float(c) for c in (left, top, left + w, top, left + w, top + h, left, top + h)]


# ========================
# Types index and the catalogue (PLAN 5.1)
# ========================
class TypesIndex:
    """
    Types/ and Types/Stars/ files: by git blob, by ink pixels (ink_key), by name (stem_key, png first); older blobs of
    a path that still exists (history: {blob: path}, from `git log --raw -- Types/` or db/links.json's `files`).
    """
    def __init__(self, files: dict, history: Optional[dict] = None):
        self.data, self.blob, self.pixels, self.inks = {}, {}, {}, {}
        self.by_blob, self.by_ink, self.by_name = {}, {}, {}
        for path in sorted(files, key=lambda p: (not p.lower().endswith(".png"), p)):
            if dirname(path) not in (TYPES_DIR, STARS_DIR) or basename(path).startswith("."):
                continue
            data = files[path]
            rgba = decode(data, file_type(data, path))
            if rgba is None:
                continue
            self.data[path], self.blob[path] = data, git_blob(data)
            self.pixels[path] = (rgba.shape[1], rgba.shape[0])
            self.inks[path] = (ink_box(rgba), ink_box(rgba, INK_ALPHA))
            self.by_blob.setdefault(self.blob[path], path)
            key = ink_key(rgba)
            if key is not None:
                self.by_ink.setdefault(key, path)
            self.by_name.setdefault(stem_key(basename(path)), path)
        lower = {p.lower(): p for p in self.data}
        self.older = {blob: lower[p.lower()] for blob, p in (history or {}).items()
                      if p.lower() in lower and blob != self.blob[lower[p.lower()]]}

    def vocab(self) -> list:
        """Type icon names (lowercase Types/ stems), psd_regions.badge_layers' vocab."""
        return sorted({splitext(basename(p))[0].lower() for p in self.data if dirname(p) == TYPES_DIR})

    def match(self, blob: str, key: Optional[tuple], names: list, rows: bool) -> Optional[tuple]:
        """(class, Types path) for a picture by its git blob, ink key and names (rows: every layer showing it sits in a
        star or type row), else None."""
        if blob in self.by_blob:
            return BYTES, self.by_blob[blob]
        if key is not None and key in self.by_ink:
            return INK, self.by_ink[key]
        if blob in self.older:
            return REDRAW, self.older[blob]
        if rows:
            for name in names:
                path = self.by_name.get(stem_key(name))
                if path is not None and self.inks[path][1] is not None:
                    return REDRAW, path
        return None


def types_history(clone: str) -> dict:
    """{blob: path} of every blob Types/ ever held in a clone (trees only: nothing fetched)."""
    done = run_process(["git", "-C", clone, "log", "--all", "--raw", "--no-abbrev", "--no-renames", "--format=", "--",
                        TYPES_DIR + "/"], capture_output=True, text=True, env={**environ, "GIT_NO_LAZY_FETCH": "1"})
    out = {}
    for line in done.stdout.splitlines():
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) >= 4 and set(parts[3]) != {"0"}:
            out.setdefault(parts[3], path)
    return out


def items(psd: PSDImage, vocab=()) -> list:
    """
    A PSD's picture items, the catalogue's worker: each smart-object item no stat bar uses, in the file order of its
    first layer, and each plain pixel layer in a star or type row (psd_regions.badge_layers: the header's).

    Returns:
        [{"uuid", "kind" (data / external), "blob", "ext", "size", "file", "key" (ink_key or None), "links" (an
        external item's path strings), "layers": [{"name", "id", "path", "visible", "row"}]}] then
        [{"pixel": layer id, "name", "path", "key"}]
    """
    stars, icons, _ = badge_layers(psd, vocab)
    rows = {id(layer) for layer in stars + icons}
    known, out, bars = _items_of(psd), {}, set()
    for layer in psd.descendants():
        if layer.kind != "smartobject":
            continue
        uid = _uid(layer)
        if uid is None or uid not in known:
            continue
        if _bar(placed_data(layer)):
            bars.add(uid)
            continue
        entry = out.setdefault(uid, {"uuid": uid, "layers": []})
        entry["layers"].append({"name": layer.name, "id": layer.layer_id, "path": _path(layer),
                                "visible": layer.is_visible(), "row": id(layer) in rows})
    for uid in list(out):
        if uid in bars:                         # one item, a bar and a picture: it stays embedded with the bar
            del out[uid]
            continue
        tag, item = known[uid]
        entry = out[uid]
        entry["file"] = _text(item.filename)
        if item.kind == LinkedLayerType.EXTERNAL:
            entry.update(kind="external", size=item.filesize, blob=None, key=None, ext=file_type(b"", entry["file"]),
                         links={_text(k): _text(v) for k, v in item.linked_file.items() if _text(k).endswith("Path")})
            continue
        if item.kind != LinkedLayerType.DATA:
            del out[uid]
            continue
        data = item.data or b""
        ext = file_type(data, entry["file"])
        entry.update(kind="data", size=len(data), blob=git_blob(data), ext=ext,
                     key=ink_key(decode(data, ext)) if ext in RASTER else None)
    pixels = []
    for layer in stars + icons:
        if layer.kind != "pixel":
            continue
        own = layer.numpy()
        key = ink_key(np.clip(np.round(own * 255), 0, 255).astype(np.uint8)) if own is not None and own.shape[2] >= 4 else None
        if key is not None:
            pixels.append({"pixel": layer.layer_id, "name": layer.name, "path": _path(layer), "key": key})
    return list(out.values()) + pixels


def assign(psds: list, types: TypesIndex, linked: dict, kept: Optional[dict] = None) -> dict:
    """
    Every item's class and target in catalogue order (PLAN 5.1: cards by repo path, then MDW/Unfinished, then
    MDW/Templates; "first user" is first in it). An item takes the most cautious class its layers allow: a name-only
    match needs every layer in a star or type row. New MDW/Linked files are named after their first user (file_name);
    fallback_names then gives a class 2 or 3 item the name its exact copy would take (5.2 step 9).

    Args:
        psds: [(repo path, items())] in catalogue order
        types: TypesIndex
        linked: {MDW/Linked path: blob} at the base (Croissant's names kept, Q1)
        kept: An earlier catalogue's new files (the per-batch re-check): their names stay, unless main now holds the
            same bytes under another name (that file is used and the unlanded name dropped)

    Returns:
        {"psds": {rel: [entry + {"class", "target"}]}, "files": {new path: {"blob", "first", "ext"}}}
    """
    by_blob = {}
    for path, blob in sorted(linked.items()):
        by_blob.setdefault(blob, path)
    taken = {basename(path).casefold() for path in linked}
    files, out, made = {}, {}, set()
    for path, info in (kept or {}).items():
        if info["blob"] not in by_blob and basename(path).casefold() not in taken and not info.get("fallback"):
            by_blob[info["blob"]] = path
            taken.add(basename(path).casefold())
            files[path] = {key: info[key] for key in ("blob", "first", "ext")}

    def new_file(entry: dict, rel: str) -> str:
        path = by_blob.get(entry["blob"])
        if path is None:
            path = f"{LINKED_DIR}/{file_name(splitext(basename(rel))[0], entry['layers'][0]['name'], entry['ext'], taken)}"
            by_blob[entry["blob"]] = path
            files[path] = {"blob": entry["blob"], "first": rel, "ext": entry["ext"]}
        return path
    for rel, found in psds:
        rows = []
        for entry in found:
            entry = dict(entry)
            if "pixel" in entry:
                hit = types.by_ink.get(tuple(entry["key"])) if entry.get("key") else None
                if hit is not None:
                    rows.append({**entry, "class": INK, "target": hit})
                continue
            if entry["kind"] == "external":
                rows.append({**entry, "class": None, "target": None})
                continue
            if not entry.get("ext") or entry["ext"] not in FILETYPES:
                rows.append({**entry, "class": None, "target": None, "why": f"no picture type ({entry.get('ext')!r})"})
                continue
            key = tuple(entry["key"]) if entry.get("key") else None
            names = [layer["name"] for layer in entry["layers"]] + [entry["file"]]
            hit = types.match(entry["blob"], key, names, all(layer["row"] for layer in entry["layers"]))
            if hit is not None:
                entry.update({"class": hit[0], "target": hit[1]})
            else:
                path = new_file(entry, rel)
                first = (files.get(path) or {}).get("first") == rel and path not in made
                made.add(path)
                entry.update({"class": NEW if first else LINKED, "target": path})
            rows.append(entry)
        out[rel] = rows
    return {"psds": out, "files": files}


def fallback_names(catalogue: dict, linked: dict, kept: Optional[dict] = None) -> None:
    """A fallback file name for every class 2 or 3 item still without one: the name its exact copy takes (dedupe by blob
    first; kept: an earlier catalogue's fallback names stay), registered in the catalogue's files like a new one."""
    by_blob = {info["blob"]: path for path, info in catalogue["files"].items()}
    for path, blob in linked.items():
        by_blob.setdefault(blob, path)
    taken = {basename(p).casefold() for p in list(linked) + list(catalogue["files"])}
    for path, info in (kept or {}).items():
        if info.get("fallback") and info["blob"] not in by_blob and basename(path).casefold() not in taken:
            by_blob[info["blob"]] = path
            catalogue["files"][path] = dict(info)
            taken.add(basename(path).casefold())
    for rel, rows in catalogue["psds"].items():
        for entry in rows:
            if entry.get("class") in (INK, REDRAW) and not entry.get("fallback") and "pixel" not in entry:
                path = by_blob.get(entry["blob"])
                if path is None:
                    name = file_name(splitext(basename(rel))[0], entry["layers"][0]["name"], entry["ext"], taken)
                    path = by_blob[entry["blob"]] = f"{LINKED_DIR}/{name}"
                    catalogue["files"][path] = {"blob": entry["blob"], "first": rel, "ext": entry["ext"], "fallback": True}
                entry["fallback"] = path


# ========================
# One PSD (PLAN 5.2 step 4)
# ========================
@dataclass
class Linked:
    """link_card's result for one PSD."""
    linked: list = field(default_factory=list)      # (uuid or pixel layer id, class, repo path)
    redrawn: list = field(default_factory=list)     # layer names whose pixels were drawn again (class 3): PNG follows
    files: dict = field(default_factory=dict)       # {MDW/Linked path: bytes} this PSD is the first user of
    copies: list = field(default_factory=list)      # uuids taken as an exact copy instead (class 5 fallback)
    repaired: list = field(default_factory=list)    # existing links rewritten canonical
    notes: list = field(default_factory=list)       # hand notes, plain words
    left: list = field(default_factory=list)        # (uuid, why) not linked


def _layers_of(psd: PSDImage, uuid: str) -> list:
    return [layer for layer in psd.descendants() if layer.kind == "smartobject" and _uid(layer) == uuid]


def _trim_corners(layer, old: Optional[np.ndarray], new: Optional[np.ndarray], alpha: int) -> Optional[list]:
    """place_trim's quad for one layer moving from its embedded picture (old) to the target file (new), ink boxes at
    alpha."""
    if old is None or new is None:
        return None
    ob, nb = ink_box(old, alpha), ink_box(new, alpha)
    if ob is None or nb is None:
        return None
    corners = corners_of(layer)
    sw, sh = _size_of(layer)
    if (sw, sh) != (float(old.shape[1]), float(old.shape[0])):   # the content's own pixels are what Sz names
        sx, sy = sw / old.shape[1], sh / old.shape[0]
        ob = (ob[0] * sx, ob[1] * sy, ob[2] * sx, ob[3] * sy)
        sizes = (sw, sh)
    else:
        sizes = (float(old.shape[1]), float(old.shape[0]))
    return place_trim(corners, sizes, (float(new.shape[1]), float(new.shape[0])), ob, nb)


def _fit_corners(layer, old: Optional[np.ndarray], rgba: Optional[np.ndarray]) -> Optional[list]:
    """
    Class 3's quad: the target's ink box onto the old picture's, at alpha > 0 (an older version of the same art: P2's
    Infliction) or at INK_ALPHA (other art: faint alpha moved P2's Equip.png by 20 px), whichever draws nearer the
    layer's stored alpha (bicubic).
    """
    stored = layer.numpy()
    quads = [q for q in (_trim_corners(layer, old, rgba, a) for a in (1, INK_ALPHA)) if q is not None]
    if len(quads) < 2 or stored is None or stored.shape[2] < 4 or rgba is None:
        return quads[-1] if quads else None
    box = tuple(layer.bbox)

    def miss(quad: list) -> float:
        out, where = draw(rgba, quad, Image.BICUBIC)
        if out is None:
            return float("inf")
        area = (min(box[0], where[0]), min(box[1], where[1]), max(box[2], where[2]), max(box[3], where[3]))
        plane = np.zeros((area[3] - area[1], area[2] - area[0]), np.float32)
        plane[where[1] - area[1]:where[3] - area[1], where[0] - area[0]:where[2] - area[0]] = out[..., 3]
        plane[box[1] - area[1]:box[3] - area[1], box[0] - area[0]:box[2] - area[0]] -= stored[..., 3]
        return float(np.abs(plane).sum())
    return min(quads, key=miss)


def link_card(psd: PSDImage, rel: str, entries: list, read: Callable[[str], Optional[bytes]],
              first: Optional[set] = None, copies: frozenset = frozenset(), tree: Optional[set] = None) -> Linked:
    """
    One PSD's pictures linked in place by its catalogue entries (assign), its stat bars left embedded:
    classes 1, 4 and 5 keep the raster; 2 takes place_trim's quad and keeps it; 3 is drawn again from the Types file
    at psd_edit.image_mode (refused: an exact copy, class 5, under its fallback name); a header pixel star or icon with
    a Types file's ink becomes a smart object linking it (link_pixels). Existing links are rewritten canonical (repair;
    one to a file the repo lacks stays, with a hand note). No lnk item is left without a layer.

    Args:
        psd: The open PSD (edited in place)
        rel: Its repo path
        entries: Its catalogue rows
        read: repo path -> the file's bytes (Types/, MDW/Linked/, the batch's new files), None when missing
        first: The new files this PSD is the first user of (it writes them); None: every new file it uses
        copies: uuids to take as exact copies (5.2 step 9's redo)
        tree: Repo paths at the base (links_of, for existing links); None: placed by read()

    Returns:
        Linked
    """
    from commands.psd_edit import image_mode
    out, known = Linked(), _items_of(psd)
    for entry in entries:
        if "pixel" in entry:
            layer = next((one for one in psd.descendants() if one.kind == "pixel" and one.layer_id == entry["pixel"]), None)
            data = read(entry["target"]) if layer is not None else None
            rgba = decode(data, file_type(data, entry["target"])) if data is not None else None
            corners = pixel_corners(layer, rgba) if rgba is not None else None
            if corners is None:
                out.left.append((entry["pixel"], "its pixels are no longer the file's"))
                continue
            link_pixels(layer, rel, entry["target"], data, corners)
            out.linked.append((entry["pixel"], INK, entry["target"]))
            continue
        uid, cls = entry["uuid"], entry.get("class")
        if uid not in known:
            out.left.append((uid, "no such item"))
            continue
        if entry.get("kind") == "external":
            continue                            # repaired below
        if cls is None:
            out.left.append((uid, entry.get("why", "no target")))
            continue
        item = known[uid][1]
        data, kind, layers = item.data or b"", entry["ext"], _layers_of(psd, uid)
        if git_blob(data) != entry["blob"]:     # a stale catalogue: classes are of other bytes
            out.left.append((uid, "its picture changed since the catalogue"))
            continue
        path = entry["target"]
        if uid in copies and cls in (INK, REDRAW):
            cls, path = NEW, entry.get("fallback")
        target = read(path) if path else None
        if cls in (LINKED, NEW) and target is None and path and git_blob(data) == entry["blob"]:
            target = data                       # a new file: the embedded bytes (written below)
        if target is None:
            out.left.append((uid, f"{path} is missing"))
            continue
        if cls in (BYTES, LINKED, NEW) and git_blob(target) != entry["blob"]:
            out.left.append((uid, f"{path} changed since the catalogue"))   # no new size over an old raster (5.1)
            continue
        if cls in (INK, REDRAW):
            old, new = decode(data, kind), decode(target, file_type(target, path))
            if cls == INK and old is not None and new is not None and ink_key(old) != ink_key(new):
                out.left.append((uid, f"{path} changed since the catalogue"))
                continue
            quads = [_trim_corners(layer, old, new, 1) if cls == INK else _fit_corners(layer, old, new) for layer in layers]
            why = next((why for layer in layers for why in [why_not_redrawn(layer) if cls == REDRAW else
                                                            "a warp" if warped(layer) else None] if why), None)
            if None in quads or why:
                path = entry.get("fallback")
                if not path:
                    out.left.append((uid, why or "its ink box"))
                    continue
                cls, target, quads = NEW, (read(path) or data), None
                out.copies.append(uid)
            pixels = None
            if quads is not None:
                pixels = (new.shape[1], new.shape[0])
                drawn = picture(psd, target, file_type(target, path)) if cls == REDRAW else None
                for layer, corners in zip(layers, quads):
                    if cls == REDRAW:
                        redraw(layer, drawn, corners, image_mode(layer, drawn, old))
                        out.redrawn.append(layer.name)
                quads = {tuple(corners_of(layer)): corners for layer, corners in zip(layers, quads)}
        else:
            quads = pixels = None
        if cls in (LINKED, NEW) and path.startswith(LINKED_DIR + "/") and read(path) is None:
            if first is None or path in first or uid in out.copies:
                out.files[path] = target
        link_item(psd, uid, rel, path, target, quads, pixels)
        out.linked.append((uid, cls, path))
    known = set(tree) if tree is not None else _tree_of(read, psd, rel)
    out.repaired, notes = repair(psd, rel, known | {path for _, _, path in out.linked})
    out.notes += notes
    return out


def _tree_of(read: Callable[[str], Optional[bytes]], psd: PSDImage, rel: str) -> set:
    """The repo paths a PSD's existing links name (placed with no tree: the first repo path each names), those read()
    has."""
    out = set()
    for _, item in _items_of(psd).values():
        if item.kind == LinkedLayerType.EXTERNAL:
            path = place(item.linked_file, rel)
            if path is not None and read(path) is not None:
                out.add(path)
    return out


@dataclass
class Link:
    """One linked item of a PSD (links_of)."""
    uuid: str
    path: Optional[str]                 # the repo path, None when it cannot be placed
    name: str                           # the file name it records
    size: int                           # the size it records
    content: str                        # its contentID ("" when Photoshop made it)
    canonical: bool                     # LINK_DATE, its file's name and the fork's strings (canonical())
    layers: list = field(default_factory=list)


def links_of(psd: PSDImage, rel: str, tree) -> list:
    """A PSD's links as repo paths (the fork's links.place: relPath from rel's folder, then the absolute forms cut at
    /TTSCardMaker/, case-blind against tree), every lnkE item a layer uses."""
    used = {}
    for layer in psd.descendants():
        if layer.kind == "smartobject":
            used.setdefault(_uid(layer), []).append(layer.name)
    out = []
    for uid, (_, item) in _items_of(psd).items():
        if item.kind != LinkedLayerType.EXTERNAL or uid not in used:
            continue
        path = place(item.linked_file, rel, tree)
        strings = {_text(k): _text(v) for k, v in (item.linked_file or {}).items()}
        good = path is not None and tuple(item.timestamp or ()) == LINK_DATE and _text(item.filename) == basename(path) \
            and all(strings.get(k) == v for k, v in canonical(rel, path).items())
        content = _text((item.content_descriptor or {}).get(b"contentID", ""))
        out.append(Link(uid, path, _text(item.filename), int(item.filesize), content, good, used[uid]))
    return out


def repair(psd: PSDImage, rel: str, tree) -> tuple:
    """
    Every link's strings, name and date made canonical (the fork's SmartObject.relink with no size: a size off the
    file's means a redraw is due, Card sync's C1).

    Returns:
        ([repaired file names], [hand notes for links the repo lacks])
    """
    done, notes = [], []
    for link in links_of(psd, rel, tree):
        if link.path is None:
            notes.append(NOT_IN_REPO.format(name=link.name))
        elif not link.canonical:
            _layer_of(psd, link.uuid).smart_object.relink(link.path, psd_path=rel)
            done.append(link.name)
    return done, notes


def link_issues(psd: PSDImage, rel: str, tree) -> list:
    """
    What a PSD's links need (the cheap test of A5's needs_normalise and the guards' plan step; nothing decoded): an
    embedded picture that is no stat bar, a linked stat bar (it goes back to embedded), a link to re-date or rewrite
    (repair), a link the repo lacks (a hand note), an lnk item no layer uses.

    Returns:
        ["embedded: <layer>", "linked bar: <layer>", "repair: <file>", "missing: <file>", "orphan: <uuid>"]
    """
    known, used, out = _items_of(psd), set(), []
    placed = [layer for layer in psd.descendants() if layer.kind == "smartobject"]
    bars = {_uid(layer) for layer in placed if _bar(placed_data(layer))}     # an item a bar shares stays embedded (items)
    for layer in placed:
        uid = _uid(layer)
        used.add(uid)
        item = known.get(uid, (None, None))[1]
        if item is None:
            continue
        bar = _bar(placed_data(layer))
        if item.kind == LinkedLayerType.DATA and uid not in bars and file_type(item.data[:16] if item.data else b"",
                                                                               _text(item.filename)) in FILETYPES:
            out.append(f"embedded: {layer.name}")
        elif item.kind == LinkedLayerType.EXTERNAL and bar:
            out.append(f"linked bar: {layer.name}")
    for link in links_of(psd, rel, tree):
        if link.path is None:
            out.append(f"missing: {link.name}")
        elif not link.canonical:
            out.append(f"repair: {link.name}")
    out += [f"orphan: {uid}" for uid in sorted(set(known) - used)]
    return list(dict.fromkeys(out))


def check(psd: PSDImage, rel: str, files: dict, tree, read: Optional[Callable[[str], Optional[bytes]]] = None,
          old: Optional[PSDImage] = None) -> list:
    """
    K4 on one rewritten PSD (card_sweep): every link placed in tree plus files (the PSD's new files) and canonical, its
    size the file's (read: repo path -> bytes; a link old already had at that size is left due, 5.2 step 4), contentIDs
    unique, no item without a layer, no linked bar, and every new MDW/Linked file in files linked by it.

    Returns:
        The problems, one line each ([] passes)
    """
    paths = set(tree) | {path for path, data in files.items() if data is not None}
    out = [found for found in link_issues(psd, rel, paths) if not found.startswith("embedded: ")]
    links = links_of(psd, rel, paths)
    seen: dict = {}
    due = {(link.path, link.size) for link in links_of(old, rel, paths)} if old is not None else set()
    for link in links:
        if link.content:
            seen.setdefault(link.content, []).append(link.name)
        data = files.get(link.path) if link.path in files else read(link.path) if read and link.path else None
        if data is not None and len(data) != link.size and (link.path, link.size) not in due:
            out.append(f"size: {link.name} {link.size}, file {len(data)}")
    out += [f"contentID shared: {', '.join(names)}" for names in seen.values() if len(names) > 1]
    used = {link.path for link in links}
    out += [f"unlinked file: {path}" for path, data in files.items()
            if data is not None and path.startswith(LINKED_DIR + "/") and path not in used]
    return out


# ========================
# The normalise fix's link step (card_standards.needs_normalise, normalise)
# ========================
NEEDS = ("embedded: ", "repair: ")      # the link_issues link() acts on (a linked bar, a missing file, an orphan: notes)


class Step(list):
    """link()'s result as card_standards.normalise reads it: the layers linked or repaired, then files, redraws, left
    ("layer: why") and hand notes."""

    def __init__(self, *args):
        super().__init__(*args)
        self.files, self.redraws, self.left, self.notes = {}, [], [], []


def needs(psd: PSDImage) -> list:
    """What link() would do: an embedded picture, a link to make canonical (judged from psd.repo_path, which the
    readers set; without it, embedded pictures alone)."""
    rel = getattr(psd, "repo_path", "")
    return [found for found in link_issues(psd, rel, None) if found.startswith(NEEDS if rel else NEEDS[:1])]


def catalogue(source, tree: dict, jobs: list, path: str) -> Optional[dict]:
    """card_sweep's catalogue: the file the catalogue command wrote (none: nothing linked). Every job's PSD it lists
    must be at the job's blob (the per-batch re-check: --recheck), else ValueError."""
    if not path:
        return None
    with open(path, encoding="utf-8") as fh:
        found = load(fh)
    blobs = found.get("blobs") or {}
    stale = [job.psd for job in jobs if job.psd in found["psds"] and blobs.get(job.psd) != job.psd_new]
    if stale:
        raise ValueError(f"{path}: {len(stale)} PSDs moved since it was made ({stale[0]} first): run catalogue --recheck")
    return found


def link(psd: PSDImage, rel: str, fonts) -> Step:
    """
    normalise's link step: link_card over the PSD's catalogue rows from fonts.links (card_guards.LinkContext: entries,
    read, tree, redraw; the guards' and the sweep's), class 3 as exact copies when it says redraw is off. Without
    fonts.links, nothing linked.
    """
    ctx, out = getattr(fonts, "links", None), Step()
    if ctx is None:
        return out
    entries = ctx.entries(rel)
    copies = frozenset() if ctx.redraw else frozenset(e["uuid"] for e in entries if e.get("class") == REDRAW)
    done = link_card(psd, rel, entries, ctx.read, ctx.first(rel), copies, set(ctx.tree))
    named = {}
    for layer in psd.descendants():
        named.setdefault(_uid(layer) if layer.kind == "smartobject" else None, []).append(layer.name)
        named.setdefault(layer.layer_id, []).append(layer.name)
    out += [name for key, _, _ in done.linked for name in named.get(key, [])] + done.repaired
    out.files, out.redraws, out.notes = dict(done.files), list(done.redrawn), list(done.notes)
    out.left = [f"{(named.get(key) or [key])[0]}: {why}" for key, why in done.left]
    return out


# ========================
# Housekeeping (PLAN G2-G4)
# ========================
def records_of(links: dict) -> tuple:
    """db/links.json as ({PSD path: record}, files): {"psds": ..., "files": ...}, or PSD paths beside "files"."""
    files = links.get("files") or {}
    if isinstance(links.get("psds"), dict):
        return links["psds"], files
    return {rel: record for rel, record in links.items() if rel != "files"}, files


def unused(links: dict, tree: dict, pushed=(), read=()) -> list:
    """
    MDW/Linked files no PSD links (G2), to delete silently. Used: every link in db/links.json at `after`, every link of
    the PSDs read directly (read: [(rel, [Link])]: the pushed ones), and a file named like a link no rule places
    (case-blind). Any PSD or PSB in the tree without a record at its blob, or one the fork could not read, blocks every
    deletion. A file added or changed in this push is never deleted on it; Types/ never.

    Args:
        links: db/links.json's {path: {"blob", "links": [{"path", "name"}]} or {"blob", "error"}}
        tree: {repo path: blob} at after
        pushed: Paths added or changed in this push

    Returns:
        The paths to delete, sorted ([] when anything blocks)
    """
    direct, (links, _) = {rel: found for rel, found in read}, records_of(links)
    used, names = set(), set()
    for rel, blob in tree.items():
        if not rel.lower().endswith((".psd", ".psb")):
            continue
        if rel in direct:
            found = [{"path": link.path, "name": link.name} for link in direct[rel]]
        else:
            record = links.get(rel)
            if record is None or record.get("blob") != blob or record.get("error"):
                return []
            found = record.get("links") or []
        for link in found:
            if link.get("path"):
                used.add(link["path"].lower())
            else:
                names.add(basename(link.get("name") or "").lower())
    pushed = set(pushed)
    return sorted(path for path in tree if path.startswith(LINKED_DIR + "/") and path not in pushed
                  and path.lower() not in used and basename(path).lower() not in names)


def duplicates(tree: dict, pushed=()) -> dict:
    """
    {duplicate: kept} for each pushed MDW/Linked file with the same git blob as another MDW/Linked or Types/ file (G3):
    kept is the Types file, else one not pushed now (the older), else the first by path. Look-alikes are not here.
    """
    groups = {}
    for path, blob in tree.items():
        if path.startswith((LINKED_DIR + "/", TYPES_DIR + "/")):
            groups.setdefault(blob, []).append(path)
    pushed, out = set(pushed), {}
    for paths in groups.values():
        if len(paths) < 2 or not any(p in pushed and p.startswith(LINKED_DIR + "/") for p in paths):
            continue
        kept = sorted(paths, key=lambda p: (not p.startswith(TYPES_DIR + "/"), p in pushed, p))[0]
        out.update({p: kept for p in paths if p != kept and p.startswith(LINKED_DIR + "/")})
    return out


def restore(size: int, history: list) -> Optional[str]:
    """The blob to put back for a link to an MDW/Linked path missing at after (G4): the path's last blob when its size is
    the link's (history: [(blob, size)] oldest first), else None (a hand note)."""
    return history[-1][0] if history and history[-1][1] == size else None


# ========================
# Card sync: a changed linked file drawn again in its users (PLAN section 8)
# ========================
@dataclass
class Redrawn:
    """redraw_linked's result."""
    files: dict = field(default_factory=dict)       # {repo path: bytes} PSDs and PNGs to commit
    cards: list = field(default_factory=list)       # PSD paths written
    due: list = field(default_factory=list)         # PSD paths still due (capped or skipped): Leftover work chains
    notes: dict = field(default_factory=dict)       # {PSD path: [hand note]} layers left as they are
    lines: list = field(default_factory=list)       # the report


def due_links(links: dict, tree: dict, sizes: dict) -> dict:
    """
    C1: {PSD path: [(link record, current blob)]} whose file changed, read off main's tree (no diff range: a capped run
    loses nothing): a size off the file's, or a drawn-from blob older than the file's of the same size. A link to a
    missing file is G4's; a link Photoshop made (no `drawn`) of the same size is never due (R21). A PSD whose record is
    not at its blob is skipped (the caller reads a pushed one directly).
    """
    out = {}
    for rel, record in sorted(links.items()):
        if record.get("error") or tree.get(rel) != record.get("blob"):
            continue
        for link in record.get("links") or []:
            path = link.get("path")
            if not path or path not in tree:
                continue
            blob = tree[path]
            size = sizes.get(path)
            if size is None:
                continue
            drawn = link.get("drawn")
            if link.get("size") != size or (drawn and drawn != blob):
                out.setdefault(rel, []).append((link, blob))
    return out


def redraw_psd(psd: PSDImage, rel: str, tree: dict, read_blob: Callable[[str], Optional[bytes]], seen: dict) -> tuple:
    """
    C4 on one PSD, per link whose file changed: the size it records is not the file's, or its contentID says it was
    drawn from an older blob of the path (a same-size edit; a link Photoshop made of the file's size is never due, R21).
    Per layer: (i) the drawn-from file's pixels equal the new file's: size, Sz and contentID only; (ii) the same ink at a
    new size (a trim): place_trim's quad, raster kept; (iii) else drawn again at psd_edit.image_mode, the new file at the
    old one's scale round the same centre, whole pixels when upright (snapped), as Photoshop updates a link. Refused (a
    warp, smart filters, a quad the redraw cannot reproduce): the link stays as it is, with a hand note.

    Args:
        tree: {repo path: blob} at the head
        read_blob: blob sha -> bytes (None when the clone lacks it)
        seen: db/links.json's files, {repo path: {blob: size}}: the path's blobs so far

    Returns:
        (changed: bool, pixels: bool (a raster was drawn again: the PNG may move), notes)
    """
    from commands.psd_edit import image_mode
    changed = pixels = False
    notes = []
    for link in links_of(psd, rel, tree):
        if link.path is None or link.path not in tree:
            continue
        blob, known = tree[link.path], seen.get(link.path) or {}
        ours = next((b for b in [blob, *known] if link_content_id(b, rel, link.uuid) == link.content), None)
        size = known.get(blob)                  # known sizes first: no file is read for a link that is not due
        if size is None:
            new = read_blob(blob)
            if new is None:
                continue
            size = len(new)
        if link.size == size and (ours is None or ours == blob):
            continue
        new = read_blob(blob)
        if new is None:
            continue
        old_blob = ours or next((b for b, size in known.items() if size == link.size and b != blob), None)
        old = read_blob(old_blob) if old_blob else None
        kind = file_type(new, link.path)
        new_rgba = decode(new, kind)
        layers = _layers_of(psd, link.uuid)
        if new_rgba is None:
            notes.append(NOT_REDRAWN.format(layer=layers[0].name, name=basename(link.path), why="it is no picture"))
            continue
        old_rgba = decode(old, file_type(old, link.path)) if old is not None else None
        size = (new_rgba.shape[1], new_rgba.shape[0])
        same = old_rgba is not None and old_rgba.shape == new_rgba.shape and np.array_equal(old_rgba, new_rgba)
        trim = not same and old_rgba is not None and ink_key(old_rgba) == ink_key(new_rgba)
        why = next((w for layer in layers for w in [why_not_redrawn(layer) if not same else None] if w), None)
        if why:
            notes.append(NOT_REDRAWN.format(layer=layers[0].name, name=basename(link.path), why=why))
            continue
        drawn = picture(psd, new, kind) if not same and not trim else None
        quads = {}
        for layer in layers:
            corners, (w0, h0) = corners_of(layer), _size_of(layer)
            if same:
                quads[tuple(corners)] = corners
            elif trim:
                quads[tuple(corners)] = place_trim(corners, (w0, h0), size, ink_box(old_rgba), ink_box(new_rgba))
            else:
                dx, dy = (size[0] - w0) / 2, (size[1] - h0) / 2
                quad = place_trim(corners, (w0, h0), size, (0, 0, w0, h0), (dx, dy, dx + w0, dy + h0))
                quads[tuple(corners)] = quad = snapped(quad) if upright(quad) else quad
                redraw(layer, drawn, quad, image_mode(layer, drawn, old_rgba))
                pixels = True
        relink(psd, link.uuid, rel, link.path, len(new), blob, quads, size)
        changed = True
    return changed, pixels, notes


def _git_bytes(clone: str, blobs: list) -> dict:
    """{blob: bytes} read from a clone in one cat-file --batch (never a lazy fetch); a missing one is left out."""
    if not blobs:
        return {}
    proc = Popen(["git", "-C", clone, "cat-file", "--batch"], stdin=PIPE, stdout=PIPE,
                 env={**environ, "GIT_NO_LAZY_FETCH": "1"})
    out = {}

    def feed():                                 # beside the reads: a long list would fill both pipes
        proc.stdin.write("".join(f"{b}\n" for b in blobs).encode())
        proc.stdin.close()
    writer = Thread(target=feed, daemon=True)
    writer.start()
    for blob in blobs:
        head = proc.stdout.readline().split()
        if len(head) < 3 or head[1] != b"blob":
            continue
        data = proc.stdout.read(int(head[2]))
        proc.stdout.read(1)
        out[blob] = data
    writer.join()
    proc.wait()
    return out


def fetch_blobs(clone: str, blobs: list) -> None:
    """The blobs a blobless clone lacks, fetched in batches (8 a fetch, 4 at a time: the guards' Pushed blobs step)."""
    wanted = [b for b in dict.fromkeys(blobs) if b]
    if not wanted:
        return
    env = {**environ, "GIT_NO_LAZY_FETCH": "1"}
    check = run_process(["git", "-C", clone, "cat-file", "--batch-check"], input="".join(f"{b}\n" for b in wanted),
                        capture_output=True, text=True, env=env)
    missing = [line.split()[0] for line in check.stdout.splitlines() if line.endswith(" missing")]
    if missing:
        run_process(["xargs", "-P", "4", "-n", "8", "git", "-C", clone, "-c", "fetch.negotiationAlgorithm=noop", "fetch",
                     "-q", "--no-tags", "--no-write-fetch-head", "--recurse-submodules=no", "--filter=blob:none",
                     "origin"], input="\n".join(missing) + "\n", text=True, env=env)


def _tree_at(clone: str, head: str) -> dict:
    """{repo path: blob} at head (trees only: nothing fetched)."""
    listing = run_process(["git", "-C", clone, "ls-tree", "-r", head], capture_output=True, text=True, check=True,
                          env={**environ, "GIT_NO_LAZY_FETCH": "1"})
    tree = {}
    for line in listing.stdout.splitlines():
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) == 3 and parts[1] == "blob":
            tree[path] = parts[2]
    return tree


def _pictures(tree: dict, rel: str) -> tuple:
    """A PSD's PNGs as card_bake's png_for finds them: (Name.png in the tree's case or "", the numbered NameN.png
    beside it that are no other PSD's)."""
    folder, name = dirname(rel), basename(rel)[:len(basename(rel)) - 4]
    lower = {p.lower() for p in tree}
    main, numbered = "", []
    for path in tree:
        if dirname(path) != folder or not path.lower().endswith(".png"):
            continue
        base = basename(path)[:len(basename(path)) - 4]
        if base.lower() == name.lower():
            main = path
        elif base[:len(name)].lower() == name.lower() and base[len(name):].isdigit() and \
                posixpath_join(folder, base + ".psd").lower() not in lower:
            numbered.append(path)
    return main, sorted(numbered)


def due_count(clone: str, links: dict, head: str = "HEAD") -> int:
    """Card sync's Linked due: how many PSDs redraw_linked would draw again (C1), from the clone's trees and links.json
    alone (nothing fetched: a file whose size neither knows counts its users as due)."""
    tree = _tree_at(clone, head)
    records, seen = records_of(links)
    linked = {link["path"] for record in records.values() for link in record.get("links") or [] if link.get("path")}
    sizes = {path: (seen.get(path) or {}).get(tree[path]) for path in linked if path in tree}
    unknown = [path for path, size in sizes.items() if size is None]
    if unknown:
        check = run_process(["git", "-C", clone, "cat-file", "--batch-check"], input="".join(f"{tree[p]}\n" for p in unknown),
                            capture_output=True, text=True, env={**environ, "GIT_NO_LAZY_FETCH": "1"})
        known = {parts[0]: int(parts[2]) for parts in map(str.split, check.stdout.splitlines()) if len(parts) == 3}
        sizes.update({path: known[tree[path]] for path in unknown if tree[path] in known})
    due = set(due_links(records, tree, sizes))
    due |= {rel for rel, record in records.items() if not record.get("error") and tree.get(rel) == record.get("blob")
            and any(link.get("path") in sizes and sizes[link["path"]] is None for link in record.get("links") or [])}
    return len(due)


def redraw_linked(clone: str, links: dict, head: str = "HEAD", cards: int = CARDS_CAP, budget: int = 1 << 30,
                  until: float = 0.0, pushed=(), fetch: bool = True) -> Redrawn:
    """
    Card sync's Linked redraw (C1-C6): the PSDs whose linked files changed, drawn again at head (C4: redraw_psd), their
    PNGs re-exported only where the drawing moved past png_export.TOL (C5: png_export.export, main and numbered
    pictures; a refusal lands neither). Users in repo path order (C2), blobs fetched in batches (C3). At most `cards`
    PSDs, none started past `until` (epoch seconds) or once `budget` bytes are written; the rest stay due.

    Args:
        clone: The card repo's checkout (blobless is enough)
        links: db/links.json ({PSD path: record}, "files": {repo path: {blob: size}})
        pushed: PSD paths pushed in this range: read directly when their record is not at their blob

    Returns:
        Redrawn (the files of commit R on head; nothing is committed here)
    """
    import commands.png_export as png_export
    from commands.psd_edit import save_psd
    tree = _tree_at(clone, head)
    records, seen = records_of(links)
    linked = sorted({link["path"] for record in records.values() for link in record.get("links") or [] if link.get("path")})
    sizes = {path: (seen.get(path) or {}).get(tree[path]) for path in linked if path in tree}
    unknown = [tree[p] for p, size in sizes.items() if size is None]
    if fetch:
        fetch_blobs(clone, unknown)
    got = _git_bytes(clone, unknown)
    sizes.update({p: len(got[tree[p]]) for p, size in sizes.items() if size is None and tree[p] in got})
    seen = {**seen, **{p: {**(seen.get(p) or {}), tree[p]: size} for p, size in sizes.items() if size is not None}}
    due = due_links(records, tree, sizes)
    for rel in pushed:
        if rel in tree and (records.get(rel) or {}).get("blob") != tree[rel]:
            due.setdefault(rel, [])
    out, written = Redrawn(), 0
    order = sorted(due)
    for n, rel in enumerate(order):
        if len(out.cards) >= cards or (until and time() > until) or written >= budget:
            out.due += order[n:]
            break
        main, numbered_pngs = _pictures(tree, rel)
        pngs = ([main] if main else []) + numbered_pngs
        wanted = [tree[rel]] + [tree[p] for p in pngs] + [blob for _, blob in due[rel]]
        for link, blob in due[rel]:             # C3: the blob each was drawn from, else the file's blobs of its size
            drawn = link.get("drawn")
            wanted += [drawn] if drawn and drawn != blob else \
                [b for b, size in (seen.get(link["path"]) or {}).items() if size == link.get("size") and b != blob]
        if fetch:
            fetch_blobs(clone, wanted)
        blobs = _git_bytes(clone, list(dict.fromkeys(wanted)))
        if tree[rel] not in blobs:
            out.due.append(rel)
            continue

        def read_blob(blob: str) -> Optional[bytes]:
            if blob not in blobs:                   # a pushed PSD's own links (its record is stale)
                if fetch:
                    fetch_blobs(clone, [blob])
                blobs.update(_git_bytes(clone, [blob]))
            return blobs.get(blob)
        raw = blobs[tree[rel]]
        old, new = PSDImage.open(BytesIO(raw)), PSDImage.open(BytesIO(raw))
        changed, moved, notes = redraw_psd(new, rel, tree, read_blob, seen)
        if notes:
            out.notes[rel] = notes
        if not changed:
            continue
        buf = BytesIO()
        save_psd(new, buf)                      # stripped (A1): the export below reads the saved PSD
        files = {rel: buf.getvalue()}
        if moved and main:
            if any(tree[p] not in blobs for p in pngs):     # C5: the PSD lands only with its PNGs
                out.due.append(rel)
                continue
            numbered = {p: png_export.open_png(blobs[tree[p]]) for p in numbered_pngs}
            found = png_export.export(old, PSDImage.open(BytesIO(files[rel])), png_export.open_png(blobs[tree[main]]),
                                      numbered, main)
            refused = [why for data, why in (found.numbered or {}).values() if data == b""]
            if found.outcome not in ("re-exported", "current") or refused:
                out.notes.setdefault(rel, []).append(
                    f"Its pictures could not be drawn again ({found.why or (refused or [found.outcome])[0]}): export them in Photoshop.")
                continue
            if found.outcome == "re-exported":
                files[main] = found.data
            files.update({path: data for path, (data, _) in (found.numbered or {}).items() if data})
        out.files.update(files)
        out.cards.append(rel)
        written += sum(len(data) for data in files.values())
        out.lines.append(f"- {rel}" + (f" (+ {', '.join(sorted(basename(p) for p in files if p != rel))})"
                                       if len(files) > 1 else ""))
    return out


# ========================
# The command lines
# ========================
def _catalogue_job(args: tuple) -> tuple:
    clone, rel, blob, vocab = args
    data = _git_bytes(clone, [blob]).get(blob)
    if data is None:
        return rel, None, "blob missing"
    try:
        return rel, items(PSDImage.open(BytesIO(data)), vocab), ""
    except Exception as error:
        return rel, None, f"{type(error).__name__}: {error}"


def catalogue_order(paths: list) -> list:
    """Catalogue order (PLAN 5.1): cards by repo path, then MDW/Unfinished, then MDW/Templates."""
    rank = lambda p: 2 if p.startswith("MDW/Templates/") else 1 if p.startswith("MDW/Unfinished/") else 0
    return sorted(paths, key=lambda p: (rank(p), p))


def run_catalogue(clone: str, ref: str, held: dict, workers: int = 4, prev: Optional[dict] = None,
                  history=()) -> dict:
    """
    The whole scope's catalogue (PLAN 5.1) read from a clone's objects at ref (held: {rel: blob} read instead). With
    prev (the per-batch re-check): only PSDs whose blob moved are read again, the rest keep their items; prev's names
    stay (assign's kept) and every Types target is classed again against ref's Types. Older Types blobs: the clone's
    history and each of history's clones (a shallow clone misses some).
    """
    from concurrent.futures import ProcessPoolExecutor
    from commands.font_check import is_card_psd
    ref = run_process(["git", "-C", clone, "rev-parse", "--verify", ref + "^{commit}"], capture_output=True, text=True,
                      check=True, env={**environ, "GIT_NO_LAZY_FETCH": "1"}).stdout.strip()     # base: a commit, not a name
    tree = _tree_at(clone, ref)
    types_files = [p for p in tree if dirname(p) in (TYPES_DIR, STARS_DIR)]
    data = _git_bytes(clone, [tree[p] for p in types_files])
    older = {}
    for source in (clone, *history):
        for blob, path in types_history(source).items():
            older.setdefault(blob, path)
    types = TypesIndex({p: data[tree[p]] for p in types_files if tree[p] in data}, older)
    scope = [p for p in tree if p.lower().endswith(".psd") and (is_card_psd(p) or p.startswith(("MDW/Templates/",
                                                                                               "MDW/Unfinished/")))]
    blobs = {rel: held.get(rel, tree[rel]) for rel in scope}
    found, errors = {}, {}
    if prev:
        for rel in scope:
            if (prev.get("blobs") or {}).get(rel) == blobs[rel] and rel in prev["psds"]:
                found[rel] = [{key: value for key, value in entry.items() if key not in ("class", "target", "fallback")}
                              for entry in prev["psds"][rel]]
    jobs = [(clone, rel, blobs[rel], types.vocab()) for rel in catalogue_order(scope) if rel not in found]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for rel, rows, error in pool.map(_catalogue_job, jobs, chunksize=2):
            if rows is None:
                errors[rel] = error
            else:
                found[rel] = rows
    linked = {p: b for p, b in tree.items() if p.startswith(LINKED_DIR + "/")}
    kept = (prev or {}).get("files")
    out = assign([(rel, found[rel]) for rel in catalogue_order(found)], types, linked, kept)
    fallback_names(out, linked, kept)
    out.update(base=ref, blobs={rel: blobs[rel] for rel in found}, errors=errors, held=sorted(held), linked=linked,
               read=[rel for _, rel, _, _ in jobs])
    return out


def _counts(catalogue: dict) -> dict:
    counts = {}
    for rows in catalogue["psds"].values():
        for entry in rows:
            key = "pixel" if "pixel" in entry else CLASSES.get(entry.get("class"), entry.get("kind") if
                                                                entry.get("kind") == "external" else "left")
            counts[key] = counts.get(key, 0) + 1
    return counts


def main(args: Optional[list] = None) -> int:
    parser = ArgumentParser(prog="python -m commands.psd_links", description="Linked pictures: the catalogue and Card sync's redraw")
    sub_ = parser.add_subparsers(dest="command", required=True)
    cat = sub_.add_parser("catalogue", help="every PSD's picture items, classes and targets (PLAN 5.1)")
    cat.add_argument("--clone", required=True, help="the card repo (its objects; never a lazy fetch)")
    cat.add_argument("--ref", default="HEAD", help="the commit to read")
    cat.add_argument("--held", default="", help="a list of PSD paths read at --held-ref instead")
    cat.add_argument("--held-ref", default="", help="where the held PSDs are read")
    cat.add_argument("--held-base", default="", help="a held PSD that moved on --ref since this commit is read at --ref")
    cat.add_argument("--recheck", default="", help="an earlier catalogue.json: its names kept, moved PSDs read again")
    cat.add_argument("--history", action="append", help="a clone whose Types/ history counts too (default HISTORY_CLONE)")
    cat.add_argument("--out", required=True, help="catalogue.json")
    cat.add_argument("--workers", type=int, default=4)
    red = sub_.add_parser("redraw", help="Card sync: PSDs whose linked files changed drawn again (C1-C6)")
    red.add_argument("--checkout", "--local", required=True, help="the card repo's checkout (blobless is enough)")
    red.add_argument("--links", default="", help="db/links.json")
    red.add_argument("--db", default="", help="the data's db folder (its links.json), for --links")
    red.add_argument("--repo", default="", help="the card repo (card-sync.yml's; blobs come from the checkout's origin)")
    red.add_argument("--head", default="HEAD", help="the commit to read the PSDs at and commit R on")
    red.add_argument("--cards", type=int, default=CARDS_CAP, help="PSDs at most")
    red.add_argument("--bytes", type=int, default=1 << 30, help="bytes written at most")
    red.add_argument("--until", "--deadline", type=float, default=0.0, help="epoch seconds: no PSD started past it")
    red.add_argument("--out", default="", help="a folder for redraw.json (due: the PSDs still due)")
    red.add_argument("--pushed", default="", help="a file of PSD paths pushed in this range")
    red.add_argument("--dry-run", action="store_true", help="commit nothing")
    red.add_argument("--output", default="", help="append redraw_due= and redrawn= here (GITHUB_OUTPUT)")
    due = sub_.add_parser("due", help="Card sync's Linked due: prints how many PSDs a redraw would draw again")
    due.add_argument("--clone", required=True, help="the card repo's checkout (its trees; nothing fetched)")
    due.add_argument("--db", required=True, help="the data's db folder (its links.json)")
    due.add_argument("--head", default="HEAD", help="the commit to read")
    due.add_argument("--repo", default="", help="card-sync.yml's (unused)")
    opts = parser.parse_args(args)
    if opts.command == "catalogue" and opts.held and not (opts.held_ref and opts.held_base):
        parser.error("--held needs --held-ref and --held-base (PLAN 5.3: a held card pushed meanwhile is read at --ref)")
    if opts.command == "catalogue":
        held = {}
        if opts.held:
            env = {**environ, "GIT_NO_LAZY_FETCH": "1"}
            for rel in open(opts.held, encoding="utf-8").read().splitlines():
                if rel.strip():
                    found = run_process(["git", "-C", opts.clone, "rev-parse", f"{opts.held_ref}:{rel.strip()}"],
                                        capture_output=True, text=True, env=env)
                    if found.returncode == 0:
                        held[rel.strip()] = found.stdout.strip()
            if opts.held_base:                  # PLAN 5.3: a held card someone pushed meanwhile is rebuilt from theirs
                base, now = _tree_at(opts.clone, opts.held_base), _tree_at(opts.clone, opts.ref)
                held = {rel: blob for rel, blob in held.items() if base.get(rel) == now.get(rel)}
        prev = None
        if opts.recheck:
            with open(opts.recheck, encoding="utf-8") as fh:
                prev = load(fh)
        history = opts.history if opts.history is not None else [HISTORY_CLONE] if isdir(expanduser(HISTORY_CLONE)) else []
        history = [expanduser(path) for path in history]
        if not all(isdir(path) for path in history):
            parser.error(f"--history: no such clone in {history}")
        out = run_catalogue(opts.clone, opts.ref, held, opts.workers, prev, history)
        makedirs(dirname(opts.out) or ".", exist_ok=True)
        with open(opts.out, "w", encoding="utf-8") as fh:
            dump(out, fh, indent=0, sort_keys=True)
        print(dumps({"psds": len(out["psds"]), "files": len(out["files"]), "errors": len(out["errors"]), **_counts(out)}))
        return 1 if out["errors"] else 0
    if opts.command == "redraw" and not (opts.links or opts.db):
        parser.error("redraw needs --links or --db")
    with open(getattr(opts, "links", "") or posixpath_join(opts.db, "links.json"), encoding="utf-8") as fh:
        links = load(fh)
    if opts.command == "due":
        print(due_count(opts.clone, links, opts.head))
        return 0
    pushed = ()
    if opts.pushed:
        with open(opts.pushed, encoding="utf-8") as fh:
            pushed = fh.read().splitlines()
    found = redraw_linked(opts.checkout, links, opts.head, opts.cards, opts.bytes, opts.until,
                          [p for p in pushed if p.strip()])
    commit = ""
    if found.files and not opts.dry_run:
        from commands.git_commit import local_commit
        env = {**environ, "GIT_NO_LAZY_FETCH": "1"}
        head = run_process(["git", "-C", opts.checkout, "rev-parse", opts.head], capture_output=True, text=True,
                           env=env, check=True).stdout.strip()
        message = f"{REDRAW_SUBJECT}\n\n" + "\n".join(found.lines) + f"\n\n{SWEEP_TRAILER}"
        commit = local_commit(opts.checkout, head, found.files, message)
    print(f"## Linked redraw\n\n{len(found.cards)} PSDs drawn again" + (f", commit {commit[:8]}" if commit else "")
          + (f", {len(found.due)} still due" if found.due else "") + "\n")
    print("\n".join(found.lines))
    for rel, notes in sorted(found.notes.items()):
        for note in notes:
            print(f"- {rel}: {note}")
    if opts.output:
        with open(opts.output, "a", encoding="utf-8") as fh:
            fh.write(f"redraw_due={'true' if found.due else 'false'}\nredrawn={dumps(found.cards)}\n")
    if opts.out:
        makedirs(opts.out, exist_ok=True)
        with open(posixpath_join(opts.out, "redraw.json"), "w", encoding="utf-8") as fh:
            dump({"due": len(found.due), "cards": found.cards, "left": found.due, "commit": commit, "notes": found.notes},
                 fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
