"""
Guard 1 of the push guards (docs/Campaigns/PUSH_GUARDS.md section 5): a card PSD pushed without its PNG gets the PNG the
bake would make (owner, 2026-10-22). The fork renders the changed layers' box of the old and the new PSD
(psd_edit.render_region, the card's ICC export mode by icc_vote, or with no stored image by the nearer render) and the
old PNG takes the new render's pixels where it differs from the old one (psd_edit.merge_png), gated as the bake is (the
baseline's parity, shape, region). Photoshop's stored composite is never written: it is only a cross-check that can
refuse (the old PNG outside the change, the drawn change against Photoshop's own image). A PSD saved without one (no
Maximize Compatibility, G2) gets its PNG anyway (owner, 2026-10-24) under the bake's own gates alone: the baseline,
every text and stat row the parser reads differently drawn with a change, the merge's shape and region; the Export is
`unchecked` and the guards tell "no stored image to cross-check". Beside its PNG with at most KEPT_PX px past
BESIDE_CUT it keeps the PNG, told (KEPT; default 2026-10-24, owner to confirm). A refusal is data (an Export with
outcome "refused" and the line card makers read).
As the bake (owner, 2026-10-23), the bot's drawing refuses only past psd_edit.PARITY_FLOOR; a miss of the bar under it is
written and told (Export.misses). A card drawn whole (new, or beside its PNG) is held to the floor per text and stat
box too, and a text box drawn blank refuses (the bake's blank render). A stale old PNG
(Photoshop's own image against it, G5) and a change the bot cannot see (G6_UNSEEN) still refuse: no psd-tools fix can
draw those.

    export(old_psd, new_psd, old_png, numbered) -> Export   a PSD pushed without its PNG (gates G2-G8); old None: a new
                                                             PSD beside an existing PNG
    reread(old_psd, new_psd) -> [(what, box)]                the texts and stat rows the parser reads differently
    export_new(new_psd) -> Export                            a new PSD with no PNG (5.5)
    png_check(psd, png) -> dict                              R10: a PNG against its PSD's composite
    changed_layers(old, new) -> (box, n)                     the layer records that really changed
    photoshop_saved(old, new) -> bool                        a new xmpMM:InstanceID and a later xmp:MetadataDate
    recompress(data, minimum, floor) -> bytes or None        the PNG losslessly smaller (oxipng, checked), or None
    same_png(a, b) -> bool                                   the same image: header, pixels and every other chunk

Every PNG the guards write is recompressed losslessly (owner, 2026-10-23; Photoshop's exports measured 4.8x too big).

No Discord, network or config.
"""
import hashlib
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from re import search
from struct import unpack
from time import process_time
from typing import Optional

import numpy as np
from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.pil_io import post_process
from psd_tools.constants import ColorMode, Resource, Tag

import commands.psd_edit as psd_edit
from commands.psd_edit import Refused
from commands.psd_regions import fx_pad, stat_rows

# ========================
# Configuration
# ========================
TOL = 2                                         # |diff| a pixel may show and still be unchanged (ICC rounding)
BESIDE_CUT = 8                                  # new PSD beside its PNG: within this, the PNG is current (owner)
KEPT_PX = 300                                   # G2 beside: px past BESIDE_CUT up to which the PNG is kept, told (fork noise)
VOLATILE = (b"shmd", b"SoLd", b"PlLd", b"TySh", b"lnsr")   # blocks Photoshop rewrites on every save (2.2)
DILATE = 8                                      # px around a numbered PNG's toggled area (5.4)
FX_EXTRA = 2                                    # px beyond fx_pad around a changed layer
OUTCOMES = ("re-exported", "current", "refused", "error")
EXPORT = "export it in Photoshop"               # every refusal ends so (owner, 2026-10-22)
UNCHECKED = "no stored image to cross-check the bot's PNG against: save the PSD with Maximize Compatibility on"   # told
KEPT = ("PNG kept as it is: the PSD has no stored image, and the bot's drawing differs from the PNG in only {px:,} px, "
        "which may be its own drawing noise; if the PNG is out of date, " + EXPORT)   # told
G2_BLANK = "PNG not made: the bot draws nothing for {what} (and the PSD has no stored image to check it against); " + EXPORT
G2_MODE = "PNG not made: the PSD is not 8-bit RGB, so the bot cannot check its drawing; " + EXPORT
G5 = "PNG not made: the PSD and its old PNG already disagreed outside your change ({bad:.1f}% of pixels); " + EXPORT
G5_IN = "PNG not made: the PSD and its old PNG already disagreed where you changed it ({bad:.1f}% of pixels); " + EXPORT
G6_BASE = ("PNG not made: the bot cannot draw this part of the card as its old PNG shows it ({bad:.1f}% of pixels "
           "differ); " + EXPORT)
G6 = "PNG not made: the bot's drawing of your change does not match the PSD's own image ({bad:.1f}% of pixels); " + EXPORT
G6_UNSEEN = "PNG not made: the PSD's own image changed where the bot sees no layer change; " + EXPORT
G6_STALE = ("PNG not made: the PSD's stored image no longer shows its layers (it was not saved by Photoshop); save it in "
            "Photoshop with Maximize Compatibility, or " + EXPORT)
G6_NEW = "PNG not made: the bot's drawing of the new PSD does not match its own image ({bad:.1f}% of pixels); " + EXPORT
G6_BOX = "PNG not made: the bot's drawing of {what} does not match the PSD's own image ({bad:.1f}% of pixels there); " + EXPORT
G6_BLANK = "PNG not made: the bot draws nothing for {what}, which the PSD's own image shows; " + EXPORT
G7 = "PNG not made: the card changed size; " + EXPORT
G8 = "{name} not updated: re-export it in Photoshop"
MISSED = {"baseline": "the old PSD drawn against the old PNG", "cross": "against the PSD's own image"}   # miss_notes' boxes
RECOMPRESS_MIN = 0.10                           # a pushed PNG is recompressed when this much smaller (ours: the owner set none)
RECOMPRESS_BYTES = 65536                        # and at least this many bytes smaller: a tiny saving is not worth a commit
OXIPNG_LEVEL = 2                                # oxipng's default preset (4.8x on Photoshop's exports; level 4 adds 1%)
OXIPNG_TIMEOUT = 120                            # seconds per PNG: oxipng keeps its best so far
SIGNATURE = b"\x89PNG\r\n\x1a\n"


# ========================
# Data
# ========================
@dataclass
class Export:
    """One card's guard-1 verdict."""
    outcome: str                                # one of OUTCOMES
    why: str = ""                               # the refusal line (card makers read it), or the error
    gate: str = ""                              # G2..G8 for a refusal
    image: Optional[Image.Image] = None         # the new PNG's pixels (re-exported)
    data: bytes = b""                           # the new PNG file
    changed_px: int = 0                         # |F|: pixels the fork's render moved past TOL
    box: Optional[tuple] = None                 # F's bbox (x0, y0, x1, y1)
    icc: str = ""                               # "converted", "raw" or "as stored"
    vote: Optional[tuple] = None                # psd_edit.icc_vote over the old pair
    writer: str = ""                            # "photoshop", "tool" (no new XMP stamp) or "new" (no old PSD)
    numbers: dict = field(default_factory=dict) # the gates' measures
    misses: dict = field(default_factory=dict)  # "baseline" / "cross" / a box: a measure past the bar, under the floor (told)
    numbered: dict = field(default_factory=dict)   # numbered PNG path -> (bytes, "") or (b"", refusal line)
    unchecked: bool = False                     # no stored composite (G2): the bake's own gates alone, nothing cross-checked


# ========================
# Pixels
# ========================
def _flat(image: Image.Image) -> np.ndarray:
    """int16 (H, W, 4) over white (psd_edit._flat)."""
    return psd_edit._flat(image).astype(np.int16)


def _diff(a: Image.Image, b: Image.Image) -> np.ndarray:
    """Per-pixel max |a - b| over the channels, both over white."""
    return np.abs(_flat(a) - _flat(b)).max(axis=2)


def _stats(diff: np.ndarray) -> dict:
    """bad_pct, mean, max over a 1-D or 2-D diff; {"px": 0} when empty."""
    if diff.size == 0:
        return {"px": 0, "bad_pct": 0.0, "mean": 0.0, "max": 0}
    return {"px": int(diff.size), "bad_pct": round(float((diff > psd_edit.PARITY_THRESH).mean() * 100), 4),
            "mean": round(float(diff.mean()), 4), "max": int(diff.max())}


def _within(measured: dict) -> bool:
    return psd_edit._within_bar(measured)


def _drawn(result: "Export", key: str) -> bool:
    """The bot's drawing over numbers[key]: False past the floor (refuse); a miss of the bar under it is kept in misses."""
    measured = result.numbers[key]
    if psd_edit._past_floor(measured):
        return False
    if measured.get("px", 1) and not _within(measured):
        result.misses[key] = measured
    return True


def _boxes(psd: PSDImage) -> list:
    """The text and stat boxes of a card drawn whole (the bake's edit boxes): (what, layer, box) per visible, non-empty
    type layer drawn at some opacity, and per stat row (psd_regions.stat_rows), padded as changed_layers pads."""
    found = [(f"the {layer.name.strip() or 'untitled'} text", layer, layer.bbox) for layer in psd.descendants()
             if layer.kind == "type" and layer.is_visible() and layer.opacity]
    for stat, rows in stat_rows(psd).items():
        found += [(f"the {stat} row" + (f" {n + 1}" if len(rows) > 1 else ""), group, bbox)
                  for n, (group, _, bbox) in enumerate(rows) if bbox and group.is_visible()]
    out, seen = [], {}
    for what, layer, (x0, y0, x1, y1) in found:
        if x1 <= x0 or y1 <= y0:                # an empty text layer draws nothing in Photoshop either
            continue
        pad = fx_pad(layer) + FX_EXTRA
        box = (max(0, x0 - pad), max(0, y0 - pad), min(psd.width, x1 + pad), min(psd.height, y1 + pad))
        if box[2] > box[0] and box[3] > box[1]:
            seen[what] = seen.get(what, 0) + 1
            out.append((what if seen[what] == 1 else f"{what} ({seen[what]})", layer, box))
    return out


def _ink(layer, box: tuple) -> Optional[np.ndarray]:
    """A text layer's letters as Photoshop rasterized them (alpha past psd_edit.INK_ALPHA), a box-sized mask; None
    without stored pixels."""
    alpha = psd_edit._alpha(layer)
    if alpha is None:
        return None
    lx0, ly0 = layer.bbox[:2]
    x0, y0, x1, y1 = max(box[0], lx0), max(box[1], ly0), min(box[2], lx0 + alpha.shape[1]), min(box[3], ly0 + alpha.shape[0])
    mask = np.zeros((box[3] - box[1], box[2] - box[0]), bool)
    if x1 > x0 and y1 > y0:
        mask[y0 - box[1]:y1 - box[1], x0 - box[0]:x1 - box[0]] = alpha[y0 - ly0:y1 - ly0, x0 - lx0:x1 - lx0] > psd_edit.INK_ALPHA
    return mask


def _boxes_drawn(result: "Export", psd: PSDImage, image: Image.Image, render: Image.Image, truth: Image.Image,
                 icc: bool) -> bool:
    """
    A card drawn whole, per text and stat box against Photoshop's image (review 2026-10-23: over the whole card a blank
    text box stays under the floor). False past the floor, or for a missed text box drawn blank: more than the floor of
    its letters' own pixels (_ink) off, and hiding the layer moves no pixel of the render (the bake's blank render; a
    layer covered by a copy moves none either, but its letters show). A miss under it is kept in misses, the whole
    card's then dropped.
    """
    diff, boxes, missed = _diff(image, truth), result.numbers.setdefault("boxes", {}), {}
    for what, layer, box in _boxes(psd):
        x0, y0, x1, y1 = box
        measured = boxes[what] = _stats(diff[y0:y1, x0:x1])
        if _within(measured):
            continue
        if psd_edit._past_floor(measured):
            _refuse(result, G6_BOX.format(what=what, bad=measured["bad_pct"]), "G6")
            return False
        ink = _ink(layer, box) if layer.kind == "type" else None
        if ink is not None and ink.any():
            measured["ink"] = _stats(diff[y0:y1, x0:x1][ink])
        if "ink" in measured and psd_edit._past_floor(measured["ink"]):
            was = layer.visible
            layer.visible = False
            try:
                hidden = _render(psd, box, icc, result.numbers)
            finally:
                layer.visible = was
            if not (_diff(hidden, render.crop(box)) > TOL).any():
                measured["blank"] = True
                _refuse(result, G6_BLANK.format(what=what), "G6")
                return False
        missed[what] = measured
    if missed:
        result.misses = missed
    return True


def composite(psd: PSDImage) -> Optional[Image.Image]:
    """
    Photoshop's stored composite in the PSD's own values.

    Returns:
        RGB or RGBA of the canvas size; None without one, past 8-bit RGB, or another size
    """
    if psd.color_mode != ColorMode.RGB or psd.depth != 8 or not psd.has_preview():
        return None
    image = psd.topil(apply_icc=False)
    if image is None or image.mode not in ("RGB", "RGBA") or image.size != (psd.width, psd.height):
        return None
    return image


def convert(psd: PSDImage, image: Image.Image, icc: bool) -> Image.Image:
    """The composite as exported: converted to sRGB by a foreign embedded profile when icc, else as stored."""
    profile = psd_edit.foreign_profile(psd)
    return post_process(image, None, profile) if icc and profile is not None else image


def icc_mode(psd: PSDImage, icc: bool) -> str:
    return "as stored" if psd_edit.foreign_profile(psd) is None else ("converted" if icc else "raw")


def like_mode(image: Image.Image, mode: str) -> Image.Image:
    """image in mode: RGBA -> RGB composites over white (what a flattened export shows), RGB -> RGBA is opaque."""
    if image.mode == mode:
        return image
    if image.mode == "RGBA" and mode == "RGB":
        return Image.fromarray(psd_edit._flat(image)[..., :3])
    return image.convert(mode)


def dilate(mask: np.ndarray, reach: int) -> np.ndarray:
    """A square dilation by reach px (separable shifts, no scipy)."""
    out = mask.copy()
    for axis in (0, 1):
        grown = out.copy()
        for step in range(1, reach + 1):
            for shift in (step, -step):
                moved = np.roll(out, shift, axis=axis)
                if shift > 0:
                    (moved[:shift] if axis == 0 else moved[:, :shift])[...] = False
                else:
                    (moved[shift:] if axis == 0 else moved[:, shift:])[...] = False
                grown |= moved
        out = grown
    return out


def _bbox(mask: np.ndarray) -> Optional[tuple]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def encode(image: Image.Image, like: Optional[Image.Image]) -> bytes:
    """The PNG file: like's colour chunks kept (psd_edit.save_png), recompressed losslessly when that is smaller."""
    buf = BytesIO()
    if like is not None:
        psd_edit.save_png(image, buf, like)
    else:
        image.save(buf, "PNG")
    return recompress(buf.getvalue()) or buf.getvalue()


# ========================
# Lossless recompression (owner, 2026-10-23)
# ========================
def _oxipng():
    """pyoxipng, or None (imported here: the bot and the bake never need it)."""
    try:
        import oxipng
    except ImportError:
        return None
    return oxipng


def _chunks(data: bytes) -> list:
    """[(type, payload)] of a PNG file but its image data (IDAT); raises ValueError when it is not one."""
    if not data.startswith(SIGNATURE):
        raise ValueError("not a PNG")
    out, i = [], len(SIGNATURE)
    while i + 8 <= len(data):
        size, kind = unpack(">I4s", data[i:i + 8])
        if i + 12 + size > len(data):
            raise ValueError("truncated chunk")
        if kind != b"IDAT":
            out.append((kind, data[i + 8:i + 8 + size]))
        i += 12 + size
        if kind == b"IEND":
            break
    return out


def _meaning(kind: bytes, payload: bytes):
    """A chunk as read (oxipng re-deflates iCCP, zTXt and compressed iTXt): their decompressed content, else the bytes.
    An ICC profile is its data: oxipng names every one "icc" (the name is a label no reader uses; PIL drops it)."""
    if kind == b"iCCP":
        _, _, rest = payload.partition(b"\0")
        return rest[:1], zlib.decompress(rest[1:])
    if kind == b"zTXt":
        name, _, rest = payload.partition(b"\0")
        return name, rest[:1], zlib.decompress(rest[1:])
    if kind == b"iTXt":
        name, _, rest = payload.partition(b"\0")
        flag, method, rest = rest[:1], rest[1:2], rest[2:]
        language, _, rest = rest.partition(b"\0")
        translated, _, text = rest.partition(b"\0")
        return name, flag, method, language, translated, zlib.decompress(text) if flag == b"\x01" else text
    return payload


def _pixels(data: bytes) -> tuple:
    image = Image.open(BytesIO(data))
    image.load()
    return image.mode, image.size, np.asarray(image)


def same_png(a: bytes, b: bytes) -> bool:
    """
    Whether two PNG files are the same image: the same header (size, bit depth, colour type, interlace), the same
    pixels decoded in the file's own mode, and every other chunk (colour, text, Exif...) the same, in the same order
    (iCCP, zTXt and compressed iTXt by their decompressed content, _meaning); only the image data's deflate may differ.
    """
    try:
        if [(k, _meaning(k, p)) for k, p in _chunks(a)] != [(k, _meaning(k, p)) for k, p in _chunks(b)]:
            return False
        (mode_a, size_a, pixels_a), (mode_b, size_b, pixels_b) = _pixels(a), _pixels(b)
    except (ValueError, OSError, SyntaxError, zlib.error):     # not a PNG PIL or zlib can read
        return False
    return mode_a == mode_b and size_a == size_b and np.array_equal(pixels_a, pixels_b)


def optimize(data: bytes) -> Optional[bytes]:
    """
    oxipng's lossless recompression of an 8-bit PNG: the colour type and bit depth kept (the bake and guard 1 read RGB
    and RGBA only), every chunk kept. Unchecked: recompress checks it.

    Returns:
        The new file (maybe not smaller); None without pyoxipng, for a PNG not 8-bit, or when oxipng fails in any way
    """
    oxipng = _oxipng()
    if oxipng is None or not data.startswith(SIGNATURE) or data[12:16] != b"IHDR" or len(data) < 33 or data[24] != 8:
        return None                             # 16-bit: PIL reads 8 bits of it, so same_png could not tell (cards: 8)
    try:
        return oxipng.optimize_from_memory(data, level=OXIPNG_LEVEL, bit_depth_reduction=False, color_type_reduction=False,
                                           palette_reduction=False, grayscale_reduction=False,
                                           strip=oxipng.StripChunks.none(), timeout=OXIPNG_TIMEOUT)
    except Exception:                           # PngError, or anything else oxipng raises: no recompression
        return None
    except BaseException as error:              # a Rust panic: pyo3's PanicException is no Exception
        if type(error).__name__ == "PanicException":
            return None
        raise


def worth(before: int, after: int, minimum: float = RECOMPRESS_MIN, floor: int = RECOMPRESS_BYTES) -> bool:
    """Whether a file of before bytes recompressed to after is worth writing: more than minimum of it saved, and floor."""
    saved = before - after
    return saved > 0 and saved > minimum * before and saved >= floor


def recompress(data: bytes, minimum: float = 0.0, floor: int = 0) -> Optional[bytes]:
    """
    The PNG losslessly smaller (optimize), the result checked by same_png.

    Args:
        data: A PNG file
        minimum: The fraction of its size it must shrink by, more than (RECOMPRESS_MIN for a pushed PNG)
        floor: The bytes it must shrink by, at least (RECOMPRESS_BYTES)

    Returns:
        The smaller file; None when optimize gives none, it saves too little, or it is not the same image
    """
    out = optimize(data)
    if out is None or not worth(len(data), len(out), minimum, floor) or not same_png(data, out):
        return None
    return out


# ========================
# What changed between two saves
# ========================
def _layer_id(record) -> Optional[int]:
    blocks = record.tagged_blocks
    if blocks is None or Tag.LAYER_ID not in blocks:
        return None
    value = blocks.get_data(Tag.LAYER_ID)
    return int(getattr(value, "value", value))


def fingerprints(psd: PSDImage) -> dict:
    """
    {layer id (or "#index"): (sha1 of the record and its channel data, record rect)}, Photoshop's VOLATILE blocks left out:
    two saves of an unchanged layer fingerprint the same.
    """
    info = psd._record.layer_and_mask_information.layer_info
    if info is None:
        return {}
    out = {}
    for i, (record, channels) in enumerate(zip(info.layer_records, info.channel_image_data)):
        digest = hashlib.sha1(repr((record.left, record.top, record.right, record.bottom, record.blend_mode,
                                    record.opacity, record.clipping, record.flags.tobytes())).encode())
        digest.update(record.mask_data.tobytes() if record.mask_data is not None else b"")
        digest.update(record.blending_ranges.tobytes() if record.blending_ranges is not None else b"")
        for key, block in (record.tagged_blocks or {}).items():
            if bytes(key) not in VOLATILE:
                digest.update(bytes(key) + block.tobytes())
        for channel in channels:
            digest.update(channel.tobytes())
        lid = _layer_id(record)
        out[lid if lid is not None else f"#{i}"] = (digest.hexdigest(), (record.left, record.top, record.right, record.bottom))
    return out


def changed_layers(old: PSDImage, new: PSDImage) -> tuple:
    """
    The union box of the layer records that differ (each padded by its effects' reach, in both PSDs), and how many.

    Returns:
        (box or None when no record changed, number of changed records)
    """
    before, after = fingerprints(old), fingerprints(new)
    ids = {key for key in set(before) | set(after) if before.get(key, (None,))[0] != after.get(key, (None,))[0]}
    if not ids:
        return None, 0
    by_id: dict = {}
    for psd in (old, new):
        for layer in psd.descendants():
            by_id.setdefault(layer.layer_id, []).append(layer)
    boxes = []
    for key in ids:
        for layer in by_id.get(key, []):
            x0, y0, x1, y1 = layer.bbox
            if x1 > x0 and y1 > y0:
                pad = fx_pad(layer) + FX_EXTRA
                boxes.append((x0 - pad, y0 - pad, x1 + pad, y1 + pad))
        for found in (before.get(key), after.get(key)):     # dividers and empty layers: the record rect
            if found and found[1][2] > found[1][0] and found[1][3] > found[1][1]:
                boxes.append(found[1])
    if not boxes:
        return (0, 0, 0, 0), len(ids)
    return (max(0, min(b[0] for b in boxes)), max(0, min(b[1] for b in boxes)), min(new.width, max(b[2] for b in boxes)),
            min(new.height, max(b[3] for b in boxes))), len(ids)


def xmp(psd: PSDImage) -> dict:
    """xmpMM:InstanceID and xmp:MetadataDate from the XMP resource ({} without one)."""
    if Resource.XMP_METADATA not in psd.image_resources:
        return {}
    data = psd.image_resources.get_data(Resource.XMP_METADATA)
    text = data.decode("utf-8", "replace") if isinstance(data, bytes) else str(data)
    out = {}
    for key in ("xmpMM:InstanceID", "xmp:MetadataDate"):
        found = search(key + r'="([^"]*)"', text) or search("<" + key + r">([^<]*)<", text)
        if found:
            out[key] = found.group(1)
    return out


def _when(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def photoshop_saved(old: Optional[PSDImage], new: PSDImage) -> bool:
    """
    Whether Photoshop saved new after old: a new xmpMM:InstanceID and a later xmp:MetadataDate (every Photoshop save
    stamps both; the fork's saves keep them). With no old PSD, unknown: False.
    """
    if old is None:
        return False
    a, b = xmp(old), xmp(new)
    if not b.get("xmpMM:InstanceID") or b.get("xmpMM:InstanceID") == a.get("xmpMM:InstanceID"):
        return False
    when_a, when_b = _when(a.get("xmp:MetadataDate", "")), _when(b.get("xmp:MetadataDate", ""))
    if when_b is None:
        return False
    if when_a is None:
        return True
    try:
        return when_b > when_a
    except TypeError:                           # one has a zone, the other not
        return when_b.replace(tzinfo=None) > when_a.replace(tzinfo=None)


# ========================
# The render (the bake's: psd_edit.render_region, merge_png)
# ========================
def _box_mask(shape: tuple, box: Optional[tuple]) -> np.ndarray:
    mask = np.zeros(shape, bool)
    if box:
        mask[box[1]:box[3], box[0]:box[2]] = True
    return mask


def _render(psd: PSDImage, box: tuple, icc: bool, numbers: dict) -> Image.Image:
    """psd_edit.render_region, its CPU seconds added to numbers["render_s"]."""
    start = process_time()
    image = psd_edit.render_region(psd, box, icc)
    numbers["render_s"] = round(numbers.get("render_s", 0.0) + process_time() - start, 2)
    return image


def _render_voted(psd: PSDImage, box: tuple, icc: bool, vote: Optional[tuple], png: Image.Image,
                  result: "Export") -> tuple:
    """
    (icc, render) of box: with no vote (no stored image) on a card with a foreign profile, drawn both ways and the mode
    nearer png (mean) kept, ties converted (raw exports: Block_Hopper); else icc's alone.

    Args:
        png: The PNG over box (the old PNG, or its crop)
    """
    render = _render(psd, box, icc, result.numbers)
    if vote is not None or psd_edit.foreign_profile(psd) is None:
        return icc, render
    other = _render(psd, box, not icc, result.numbers)          # no vote: icc is True (export_icc)
    means = {icc: float(_diff(render, png).mean()), not icc: float(_diff(other, png).mean())}
    result.numbers["icc_means"] = {"converted": round(means[True], 3), "raw": round(means[False], 3)}
    return (icc, render) if means[icc] <= means[not icc] else (not icc, other)


def _composite_moved(old: PSDImage, new: PSDImage, before: Optional[Image.Image], composed: Image.Image, icc: bool,
                     old_png: Image.Image, truth: Image.Image) -> np.ndarray:
    """D: where Photoshop's image moved past TOL (the composites in their own values when the profile stayed, else as
    exported; the old PNG stands in for an old PSD without one)."""
    if before is not None and before.size == composed.size and psd_edit.foreign_profile(old) == psd_edit.foreign_profile(new):
        return _diff(composed, before) > TOL
    if before is not None and before.size == composed.size:
        return _diff(truth, like_mode(convert(old, before, icc), old_png.mode)) > TOL
    return _diff(truth, old_png) > TOL


# ========================
# Public API Functions
# ========================
def export(old: Optional[PSDImage], new: PSDImage, old_png: Image.Image, numbered: Optional[dict] = None) -> Export:
    """
    Guard 1 on a PSD pushed without its PNG: the bake's render of the changed layers' box (R), merged into the old PNG.

    Args:
        old: The PSD before the push (None: a new PSD beside an existing PNG)
        new: The pushed PSD
        old_png: The PNG in the tree (as opened)
        numbered: {path: opened PNG} of the numbered duplicates (A_Snack2.png)

    Returns:
        Export: re-exported (image, data), current (numbers["png_ahead"]: the PNG already shows the new PSD; beside, no
        pixel of it past the parity threshold), or refused
        (why, gate). numbers: records, region, render_s (CPU seconds), composite_px (D), where (D outside R), png_vs_psd,
        outside (G5), baseline and png_vs_old_psd (G5/G6), cross (G6), png_ahead; with no stored image reread, and
        beside png_vs_render (the new PSD drawn against its PNG: where the PNG takes the render, never a refusal) and
        kept (at most KEPT_PX: the PNG kept, its px past BESIDE_CUT)
    """
    result = Export("current")
    if old_png.size != (new.width, new.height) or old_png.mode not in ("RGB", "RGBA"):
        return Export("refused", G7, "G7")
    composed = composite(new)
    if composed is None:                        # nothing to cross-check the render against: the bake's gates alone
        return _no_composite(new) or _unchecked(old, new, old_png, numbered)
    if old is None:
        return _beside(new, old_png, composed, result, numbered)
    before = composite(old)
    # G4: how the old PNG was exported (raw or converted), from the old pair
    result.vote = psd_edit.icc_vote(old, old_png)
    icc = psd_edit.export_icc(old, old_png)
    result.icc = icc_mode(new, icc)
    result.writer = "photoshop" if photoshop_saved(old, new) else "tool"
    truth = like_mode(convert(new, composed, icc), old_png.mode)     # Photoshop's image, as exported: cross-checks only
    shape = (new.height, new.width)
    composite_moved = _composite_moved(old, new, before, composed, icc, old_png, truth)
    # G3: the changed layers' box, rendered for both PSDs; F is where the render moved
    region, records = changed_layers(old, new)
    region = region if region and region[2] > region[0] and region[3] > region[1] else None
    inside, moved = _box_mask(shape, region), np.zeros(shape, bool)
    if region:
        base, render = _render(old, region, icc, result.numbers), _render(new, region, icc, result.numbers)
        moved[region[1]:region[3], region[0]:region[2]] = _diff(render, base) > TOL
    result.changed_px, result.box = int(moved.sum()), _bbox(moved)
    result.numbers.update(records=records, region=list(region) if region else None, composite_px=int(composite_moved.sum()),
                          where=int((composite_moved & ~inside).sum()), png_vs_psd=_stats(_diff(truth, old_png)))
    if not moved.any() and not composite_moved.any():
        return result
    # the PNG already shows the pushed PSD (pushed before it): over F nearer the new render than the old, within the bar
    # of the new render or of Photoshop's image there (a change the fork draws otherwise), and of Photoshop's image over
    # the card. Nothing to make, nothing to tell
    if moved.any():
        crop = old_png.crop(region)
        changed = moved[region[1]:region[3], region[0]:region[2]]
        to_new, to_old = _stats(_diff(render, crop)[changed]), _stats(_diff(base, crop)[changed])
        shown = _within(to_new) or _within(_stats(_diff(truth, old_png)[moved | composite_moved]))
        if shown and to_new["mean"] < to_old["mean"] and _within(result.numbers["png_vs_psd"]):
            result.numbers["png_ahead"] = {"new": to_new, "old": to_old}
            return result
    # G5: outside the change, the old PNG shows the pushed PSD (Photoshop's image: the merge keeps the old PNG there)
    outside = _stats(_diff(truth, old_png)[~(inside | composite_moved)])
    result.numbers["outside"] = outside
    if outside["px"] and not _within(outside):
        return _refuse(result, G5.format(bad=outside["bad_pct"]), "G5")
    if not moved.any():                         # Photoshop's image moved where no layer the fork draws did
        return _refuse(result, G6_UNSEEN, "G6")
    # G6: the bake's parity gate (the fork draws the old PSD over R as the old PNG shows it), then its merge
    result.numbers["baseline"] = _stats(_diff(base, crop))
    if not _within(result.numbers["baseline"]):
        if before is not None and before.size == composed.size:     # Photoshop's old image tells a stale PNG from the fork
            was = _stats(_diff(like_mode(convert(old, before, icc), old_png.mode).crop(region), crop))
            result.numbers["png_vs_old_psd"] = was
            if not _within(was):
                return _refuse(result, G5_IN.format(bad=was["bad_pct"]), "G5")
        if not _drawn(result, "baseline"):
            return _refuse(result, G6_BASE.format(bad=result.numbers["baseline"]["bad_pct"]), "G6")
    merged, _ = psd_edit.merge_png(old_png, base, render, region)
    try:                                        # G7: the bake's shape and region gates
        psd_edit.gate_shape(old_png, merged, new)
        psd_edit.gate_region(old_png, merged, region)
    except Refused:
        return _refuse(result, G7, "G7")
    # G6: where the render or Photoshop's image moved, the new PNG shows Photoshop's image
    result.numbers["cross"] = _stats(_diff(merged, truth)[moved | composite_moved])
    if not _within(result.numbers["cross"]):
        unseen = composite_moved & ~inside      # Photoshop's image moved outside every changed layer
        if unseen.any() and not _within(_stats(_diff(merged, truth)[unseen])):
            return _refuse(result, G6_UNSEEN, "G6")
        if not _drawn(result, "cross"):
            why = G6 if result.writer == "photoshop" else G6_STALE
            return _refuse(result, why.format(bad=result.numbers["cross"]["bad_pct"]), "G6")
    result.outcome, result.image, result.data = "re-exported", merged, encode(merged, old_png)
    # G8: numbered duplicates take F's pixels when F stays clear of their toggled titles
    for path, image in sorted((numbered or {}).items()):
        result.numbered[path] = delta_transfer(old_png, merged, image, moved, path)
    return result


def _beside(new: PSDImage, old_png: Image.Image, composed: Image.Image, result: Export, numbered: Optional[dict]) -> Export:
    """
    A new PSD beside an existing PNG (no old PSD, so no baseline). D is where Photoshop's image differs from the PNG past
    TOL. No pixel past BESIDE_CUT: the PNG already shows the PSD (pushed first), current, nothing rendered.
    Else the whole card is rendered and the PNG takes it over D only (elsewhere it already shows the PSD), then G6 (b).
    """
    result.vote = psd_edit.icc_vote(new, old_png)               # that PNG's own export
    icc = psd_edit.export_icc(new, old_png)
    result.icc, result.writer = icc_mode(new, icc), "new"
    truth = like_mode(convert(new, composed, icc), old_png.mode)
    diff = _diff(truth, old_png)
    moved = diff > TOL
    result.numbers.update(composite_px=int(moved.sum()), png_vs_psd=_stats(diff))
    if int(diff.max()) <= BESIDE_CUT:                           # export noise at most: never redrawn by the fork
        result.numbers["png_ahead"] = {"card": result.numbers["png_vs_psd"]}
        return result
    box = (0, 0, new.width, new.height)
    render = like_mode(_render(new, box, icc, result.numbers), old_png.mode)
    result.changed_px, result.box = int(moved.sum()), _bbox(moved)
    result.numbers["region"] = list(box)
    out = np.array(old_png)
    out[moved] = np.asarray(render)[moved]
    merged = Image.fromarray(out)
    result.numbers["cross"] = _stats(_diff(merged, truth)[moved])
    if not _drawn(result, "cross"):
        return _refuse(result, G6.format(bad=result.numbers["cross"]["bad_pct"]), "G6")
    if not _boxes_drawn(result, new, merged, render, truth, icc):
        return result
    result.outcome, result.image, result.data = "re-exported", merged, encode(merged, old_png)
    for path, image in sorted((numbered or {}).items()):
        result.numbered[path] = delta_transfer(old_png, merged, image, moved, path)
    return result


def _no_composite(psd: PSDImage) -> Optional[Export]:
    """G2 for a PSD with no stored composite: refused when it is not 8-bit RGB (the fork's render is no export of it); None
    for one saved without Maximize Compatibility (rendered anyway, _unchecked)."""
    if psd.color_mode != ColorMode.RGB or psd.depth != 8:
        return Export("refused", G2_MODE, "G2")
    return None


def _type_layers(psd: PSDImage) -> dict:
    """{layer id: (name, text stripped, drawn, bbox)} of every type layer (drawn: visible with some opacity)."""
    out = {}
    for layer in psd.descendants():
        if layer.kind == "type":
            try:
                text = str(layer.text or "").strip()
            except Exception:                   # an engine the reader cannot read: no text to compare
                text = ""
            out[layer.layer_id] = (layer.name.strip() or "untitled", text, layer.is_visible() and bool(layer.opacity),
                                   tuple(layer.bbox))
    return out


def reread(old: PSDImage, new: PSDImage) -> list:
    """
    What the parser reads differently in the pushed PSD (the bake's read-back, here for a push): each type layer whose
    drawn text changed (new, edited, shown or hidden) and each stat row whose visible digits changed.

    Returns:
        [(what, box)] with the box padded as changed_layers pads, clipped to the canvas
    """
    out = []
    before, after = _type_layers(old), _type_layers(new)
    for key in sorted(set(before) | set(after)):
        was, now = before.get(key), after.get(key)
        seen = [(t[1] if t and t[2] else "") for t in (was, now)]
        if seen[0] == seen[1]:
            continue
        name, _, _, bbox = now if now and now[2] else was
        out.append((f"the {name} text", bbox))
    rows_old, rows_new = stat_rows(old), stat_rows(new)
    for stat in rows_new:
        for n, (_, digits, bbox) in enumerate(rows_new[stat]):
            then = rows_old.get(stat, [])
            if bbox and (n >= len(then) or then[n][1] != digits):
                out.append((f"the {stat} row" + (f" {n + 1}" if len(rows_new[stat]) > 1 else ""), bbox))
    boxes = []
    for what, (x0, y0, x1, y1) in out:
        box = (max(0, x0 - FX_EXTRA), max(0, y0 - FX_EXTRA), min(new.width, x1 + FX_EXTRA), min(new.height, y1 + FX_EXTRA))
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append((what, box))
    return boxes


def _texts_drawn(result: Export, psd: PSDImage, render: Image.Image, icc: bool) -> bool:
    """A card drawn whole with no composite: every drawn text layer's box changes when the layer is hidden (the bake's
    blank render); False (refused, G2_BLANK) for one the render does not show."""
    for what, layer, box in _boxes(psd):
        if layer.kind != "type" or not str(layer.text or "").strip():
            continue
        was = layer.visible
        layer.visible = False
        try:
            hidden = _render(psd, box, icc, result.numbers)
        finally:
            layer.visible = was
        if not (_diff(hidden, render.crop(box)) > TOL).any():
            _refuse(result, G2_BLANK.format(what=what), "G2")
            return False
    return True


def _unchecked(old: Optional[PSDImage], new: PSDImage, old_png: Image.Image, numbered: Optional[dict]) -> Export:
    """
    G2 rendered anyway (owner, 2026-10-24): a PSD saved without Maximize Compatibility has no stored image, so nothing is
    compared with Photoshop's own; the bake's gates stand. With an old PSD: the changed layers' box rendered for both,
    each text and stat row reread() finds drawn with a change (else G2_BLANK), the baseline (the old PSD drawn against the
    old PNG: past the floor G6, a miss told), the delta merge, the shape and region gates (G7). A new PSD beside its PNG:
    the whole card rendered; the PNG is current with no pixel past BESIDE_CUT, kept (told) up to KEPT_PX past it, else it
    takes the render past the cut, every text drawn (_texts_drawn; no baseline, so no pixel refusal). A PNG pushed before
    its PSD is current (png_ahead): over F nearer the new render than the old, within the bar of the new render or with
    its bad pixels under PARITY_BAD_MAX and the old render past the bar (a change the fork draws a few levels off:
    Pandamonium's gradient). ICC mode: the old pair's vote; with no stored image to vote, a foreign profile is drawn
    both ways and the mode nearer the old PNG kept (_render_voted: the 21 raw exports).
    """
    result = Export("current", unchecked=True)
    if old is None:
        return _beside_unchecked(new, old_png, result, numbered)
    result.vote = psd_edit.icc_vote(old, old_png)
    icc = psd_edit.export_icc(old, old_png)
    result.writer = "photoshop" if photoshop_saved(old, new) else "tool"
    region, records = changed_layers(old, new)
    region = region if region and region[2] > region[0] and region[3] > region[1] else None
    changed = reread(old, new)
    result.numbers.update(records=records, region=list(region) if region else None, reread=[what for what, _ in changed])
    moved = np.zeros((new.height, new.width), bool)
    if region:                                  # no vote: the old PSD drawn both ways against the old PNG
        icc, base = _render_voted(old, region, icc, result.vote, old_png.crop(region), result)
        render = _render(new, region, icc, result.numbers)
        moved[region[1]:region[3], region[0]:region[2]] = _diff(render, base) > TOL
    result.icc = icc_mode(new, icc)
    result.changed_px, result.box = int(moved.sum()), _bbox(moved)
    for what, (x0, y0, x1, y1) in changed:
        if not moved[y0:y1, x0:x1].any():
            return _refuse(result, G2_BLANK.format(what=what), "G2")
    if not moved.any():
        return result
    crop = old_png.crop(region)
    # the PNG already shows the pushed PSD (pushed before it): over F nearer the new render than the old, within the bar
    # of it, or its bad pixels under the bar with the old render past it (a stale PNG is within the bar of the old)
    changed_box = moved[region[1]:region[3], region[0]:region[2]]
    to_new, to_old = _stats(_diff(render, crop)[changed_box]), _stats(_diff(base, crop)[changed_box])
    shown = _within(to_new) or (to_new["bad_pct"] <= psd_edit.PARITY_BAD_MAX and not _within(to_old))
    if shown and to_new["mean"] < to_old["mean"]:
        result.numbers["png_ahead"] = {"new": to_new, "old": to_old}
        return result
    result.numbers["baseline"] = _stats(_diff(base, crop))
    if not _drawn(result, "baseline"):
        return _refuse(result, G6_BASE.format(bad=result.numbers["baseline"]["bad_pct"]), "G6")
    merged, _ = psd_edit.merge_png(old_png, base, render, region)
    try:
        psd_edit.gate_shape(old_png, merged, new)
        psd_edit.gate_region(old_png, merged, region)
    except Refused:
        return _refuse(result, G7, "G7")
    result.outcome, result.image, result.data = "re-exported", merged, encode(merged, old_png)
    for path, image in sorted((numbered or {}).items()):
        result.numbered[path] = delta_transfer(old_png, merged, image, moved, path)
    return result


def _beside_unchecked(new: PSDImage, old_png: Image.Image, result: Export, numbered: Optional[dict]) -> Export:
    """_unchecked for a new PSD beside an existing PNG. No old PSD, so no baseline: the render of the whole card against
    the PNG (png_vs_render) holds the author's change too, so it only finds where the PNG takes the render; it is never a
    drawing miss and never refuses. Up to KEPT_PX px past BESIDE_CUT the PNG is kept (numbers["kept"], told). The gate is
    every text drawn (_texts_drawn). ICC mode by the nearer render (_render_voted)."""
    result.vote = psd_edit.icc_vote(new, old_png)
    box = (0, 0, new.width, new.height)
    icc, render = _render_voted(new, box, psd_edit.export_icc(new, old_png), result.vote, old_png, result)
    render = like_mode(render, old_png.mode)
    result.icc, result.writer = icc_mode(new, icc), "new"
    diff = _diff(render, old_png)
    result.numbers.update(region=list(box), png_vs_render=_stats(diff))
    if int(diff.max()) <= BESIDE_CUT:
        result.numbers["png_ahead"] = {"card": result.numbers["png_vs_render"]}
        return result
    moved = diff > BESIDE_CUT
    # default 2026-10-24, owner to confirm: a few hundred px, the fork's noise, kept and told (KEPT); past it redrawn
    # (the replayed stale edits, 1,684 px and up, land within the bar of the author's export)
    if int(moved.sum()) <= KEPT_PX:
        result.numbers["kept"] = int(moved.sum())
        return result
    out = np.array(old_png)
    out[moved] = np.asarray(render)[moved]
    merged = Image.fromarray(out)
    result.changed_px, result.box = int(moved.sum()), _bbox(moved)
    if not _texts_drawn(result, new, render, icc):
        return result
    result.outcome, result.image, result.data = "re-exported", merged, encode(merged, old_png)
    for path, image in sorted((numbered or {}).items()):
        result.numbered[path] = delta_transfer(old_png, merged, image, moved, path)
    return result


def _refuse(result: Export, why: str, gate: str) -> Export:
    result.outcome, result.why, result.gate, result.image, result.data = "refused", why, gate, None, b""
    result.misses = {}
    return result


def miss_notes(found: Export) -> list:
    """The note per measure that missed the bar under the floor (as the bake's psd_edit.miss_notes): "made with a pixel
    difference (the old PSD drawn against the old PNG): bad 1.35%, mean 1.00, max 79" (Defcon Panda)."""
    return ["made with a pixel difference (%s): bad %.2f%%, mean %.2f, max %d"
            % (MISSED.get(key, f"{key} against the PSD's own image"), m["bad_pct"], m["mean"], m["max"])
            for key, m in found.misses.items()]


def delta_transfer(old_base: Image.Image, new_base: Image.Image, numbered: Image.Image, moved: np.ndarray,
                   path: str) -> tuple:
    """
    A numbered PNG (the base export with other title layers shown) updated with D's pixels from the new base PNG.

    Returns:
        (PNG bytes, "") or (b"", refusal line) when D touches its toggled area (or it is another size)
    """
    name = path.rsplit("/", 1)[-1]
    if numbered.size != old_base.size:
        return b"", G8.format(name=name)
    toggled = dilate(_diff(numbered, old_base) > TOL, DILATE)
    if (toggled & moved).any():
        return b"", G8.format(name=name)
    out = np.array(like_mode(numbered, "RGBA"))
    new = np.asarray(like_mode(new_base, "RGBA"))
    out[moved] = new[moved]
    image = like_mode(Image.fromarray(out), numbered.mode)
    return encode(image, numbered), ""


def export_new(new: PSDImage) -> Export:
    """
    Guard 1 on a new PSD with no PNG (5.5): the fork renders the whole card, converted (Export As' default), RGBA when it
    has transparency; written unless it is past the floor against the PSD's own image, over the card or a text or stat
    box, or draws a text box blank (_boxes_drawn); a miss of the bar is told. With no stored image (G2, rendered anyway):
    written unless a text is drawn blank (_texts_drawn), unchecked.

    Returns:
        Export
    """
    composed = composite(new)
    refused = _no_composite(new) if composed is None else None
    if refused:
        return refused
    result = Export("current", icc=icc_mode(new, True), writer="new", unchecked=composed is None)
    box = (0, 0, new.width, new.height)
    out = _render(new, box, True, result.numbers)
    if out.mode == "RGBA" and int(np.asarray(out)[..., 3].min()) == 255:
        out = out.convert("RGB")
    result.changed_px, result.box = out.width * out.height, box
    if composed is None:                        # G2 rendered anyway: every text drawn, nothing cross-checked
        if not _texts_drawn(result, new, out, True):
            return result
        result.outcome, result.image, result.data = "re-exported", out, encode(out, None)
        return result
    truth = convert(new, composed, True)
    result.numbers["cross"] = _stats(_diff(out, truth))
    if not _drawn(result, "cross"):
        return _refuse(result, G6_NEW.format(bad=result.numbers["cross"]["bad_pct"]), "G6")
    if not _boxes_drawn(result, new, out, out, truth, True):
        return result
    result.outcome, result.image, result.data = "re-exported", out, encode(out, None)
    return result


def png_check(psd: PSDImage, png: Image.Image) -> dict:
    """
    R10: the PNG against its PSD's composite, converted as voted (icc_vote over this pair).

    Returns:
        {"ok": bool or None (no composite to compare), "icc", "bad_pct", "mean", "max", "px"}
    """
    composed = composite(psd)
    if composed is None or composed.size != png.size:
        return {"ok": None, "why": "no stored composite" if composed is None else "another size"}
    icc = psd_edit.export_icc(psd, png)
    measured = _stats(_diff(like_mode(convert(psd, composed, icc), png.mode), png))
    return {"ok": _within(measured), "icc": icc_mode(psd, icc), **measured}


def open_png(data: bytes) -> Image.Image:
    image = Image.open(BytesIO(data))
    image.load()
    return image


def open_psd(data: bytes) -> PSDImage:
    return PSDImage.open(BytesIO(data))


# ========================
# Module Exports
# ========================
__all__ = [
    'DILATE',
    'Export',
    'KEPT',
    'KEPT_PX',
    'OUTCOMES',
    'TOL',
    'UNCHECKED',
    'VOLATILE',
    'changed_layers',
    'composite',
    'convert',
    'delta_transfer',
    'dilate',
    'encode',
    'EXPORT',
    'G2_BLANK',
    'G2_MODE',
    'G5',
    'G5_IN',
    'G6',
    'G6_BASE',
    'G6_BLANK',
    'G6_BOX',
    'G6_NEW',
    'G6_STALE',
    'G6_UNSEEN',
    'G7',
    'G8',
    'MISSED',
    'RECOMPRESS_BYTES',
    'RECOMPRESS_MIN',
    'export',
    'export_new',
    'miss_notes',
    'fingerprints',
    'optimize',
    'like_mode',
    'open_png',
    'open_psd',
    'photoshop_saved',
    'png_check',
    'recompress',
    'reread',
    'same_png',
    'worth',
    'xmp',
]
