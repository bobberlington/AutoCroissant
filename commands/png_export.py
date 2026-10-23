"""
Guard 1 of the push guards (docs/Campaigns/PUSH_GUARDS.md section 5): a card PSD pushed without its PNG gets the PNG
from Photoshop's own stored composite ("Maximize Compatibility"), converted the way the card was exported (the ICC
vote), gated. The fork renders nothing that is written: it only referees a PSD Photoshop did not save. A refusal is
data (an Export with outcome "refused" and the line card makers read).

    export(old_psd, new_psd, old_png, numbered, referee) -> Export   a PSD pushed without its PNG (gates G2-G8)
    export_new(new_psd) -> Export                                     a new PSD with no PNG (5.5)
    png_check(psd, png) -> dict                                       R10: a PNG against its PSD's composite
    changed_layers(old, new) -> (box, n)                              the layer records that really changed
    photoshop_saved(old, new) -> bool                                 a new xmpMM:InstanceID and a later xmp:MetadataDate

No Discord, network or config.
"""
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from re import search
from typing import Optional

import numpy as np
from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.pil_io import post_process
from psd_tools.constants import ColorMode, Resource, Tag

import commands.psd_edit as psd_edit
from commands.psd_edit import Refused
from commands.psd_regions import fx_pad

# ========================
# Configuration
# ========================
TOL = 2                                         # |diff| a pixel may show and still be unchanged (ICC rounding)
VOLATILE = (b"shmd", b"SoLd", b"PlLd", b"TySh", b"lnsr")   # blocks Photoshop rewrites on every save (2.2)
DILATE = 8                                      # px around a numbered PNG's toggled area (5.4)
REFEREE_PAD = 4                                 # px around D's box for the referee's render
FX_EXTRA = 2                                    # px beyond fx_pad around a changed layer
OUTCOMES = ("re-exported", "current", "refused", "error")
G2 = "PNG not made: the PSD was saved without Maximize Compatibility; save it with it on, or export the PNG"
G5 = ("PNG not made: the PSD and its old PNG already disagreed outside your change ({bad:.1f}% of pixels); "
      "export it in Photoshop")
G6 = ("PNG not made: the PSD was not saved by Photoshop and the bot could not check its stored image; "
      "export it in Photoshop")
G6_NEW = "PNG not made: the bot could not check the new PSD's stored image against its layers; export it in Photoshop"
G6_STALE = ("PNG not made: the PSD's stored image no longer shows its layers (it was not saved by Photoshop); open and save it "
            "in Photoshop with Maximize Compatibility, or export the PNG")
G7 = "PNG not made: the card changed size"
G8 = "{name} not updated: re-export it in Photoshop"


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
    changed_px: int = 0                         # |D|: pixels the composite moved past TOL
    box: Optional[tuple] = None                 # D's bbox (x0, y0, x1, y1)
    icc: str = ""                               # "converted", "raw" or "as stored"
    vote: Optional[tuple] = None                # psd_edit.icc_vote over the old pair
    writer: str = ""                            # "photoshop" or "tool"
    numbers: dict = field(default_factory=dict) # the gates' measures
    numbered: dict = field(default_factory=dict)   # numbered PNG path -> (bytes, "") or (b"", refusal line)


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
    """The PNG file: like's colour chunks kept (psd_edit.save_png); PIL's default deflate."""
    buf = BytesIO()
    if like is not None:
        psd_edit.save_png(image, buf, like)
    else:
        image.save(buf, "PNG")
    return buf.getvalue()


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
# The referee
# ========================
def referee(psd: PSDImage, image: Image.Image, box: tuple, icc: bool) -> dict:
    """The fork's render of psd over box against image's crop there (the parity measure)."""
    render = psd_edit.render_region(psd, box, icc)
    return _stats(_diff(render, image.crop(box)))


# ========================
# Public API Functions
# ========================
def export(old: Optional[PSDImage], new: PSDImage, old_png: Image.Image, numbered: Optional[dict] = None,
           run_referee: bool = True) -> Export:
    """
    Guard 1 on a PSD pushed without its PNG.

    Args:
        old: The PSD before the push (None: a new PSD beside an existing PNG)
        new: The pushed PSD
        old_png: The PNG in the tree (as opened)
        numbered: {path: opened PNG} of the numbered duplicates (A_Snack2.png)
        run_referee: Run the fork as referee when Photoshop did not save the PSD (False refuses instead)

    Returns:
        Export: re-exported (image, data), current (numbers["png_ahead"]: the PNG already shows the new PSD), or refused
        (why, gate)
    """
    result = Export("current")
    composed = composite(new)
    if composed is None:
        return Export("refused", G2, "G2")
    if composed.size != old_png.size:
        return Export("refused", G7, "G7")
    before = composite(old) if old is not None else None
    # G4: how the old PNG was exported (raw or converted), from the old pair
    voter = old if old is not None and before is not None else new     # a new PSD beside a PNG: that PNG's own export
    result.vote = psd_edit.icc_vote(voter, old_png)
    icc = psd_edit.export_icc(voter, old_png)
    result.icc = icc_mode(new, icc)
    out = like_mode(convert(new, composed, icc), old_png.mode)
    # G3: did any pixel move? Composites in their own values when the profile stayed, else as exported
    same_profile = old is not None and psd_edit.foreign_profile(old) == psd_edit.foreign_profile(new)
    if before is not None and before.size == composed.size and same_profile:
        moved = _diff(composed, before) > TOL
    elif before is not None and before.size == composed.size:
        moved = _diff(out, like_mode(convert(old, before, icc), old_png.mode)) > TOL
    else:
        moved = _diff(out, old_png) > TOL
    result.changed_px = int(moved.sum())
    result.box = _bbox(moved)
    result.numbers["png_vs_psd"] = _stats(_diff(out, old_png))
    if not result.changed_px:
        return _stale_composite(result, old, new, old_png, out, icc) if run_referee else result
    # the PNG already shows the pushed PSD (pushed before it): over D nearer the new composite than the old, and within
    # the bar there and over the card. Nothing to make (no churn over the author's export)
    if before is not None and before.size == composed.size:
        was = like_mode(convert(old, before, icc), old_png.mode)
        to_new, to_old = _stats(_diff(out, old_png)[moved]), _stats(_diff(was, old_png)[moved])
        if _within(to_new) and to_new["mean"] < to_old["mean"] and _within(result.numbers["png_vs_psd"]):
            result.numbers["png_ahead"] = {"new": to_new, "old": to_old}
            return result
    # G5: outside the change, the new PNG is the old one (else the old PNG was stale already)
    inside = np.zeros(moved.shape, bool)
    region, records = changed_layers(old, new) if old is not None else (None, 0)
    if region:
        inside[region[1]:region[3], region[0]:region[2]] = True
    result.numbers.update(records=records, region=list(region) if region else None,
                          where=int((moved & ~inside).sum()))
    outside = _stats(_diff(out, old_png)[~(inside | moved)])
    result.numbers["outside"] = outside
    if outside["px"] and not _within(outside):
        return _refuse(result, G5.format(bad=outside["bad_pct"]), "G5")
    # G6: Photoshop's composite, or the fork as referee over D's box (with its own baseline on the old pair)
    if photoshop_saved(old, new):
        result.writer = "photoshop"
    else:
        result.writer = "tool"
        if not run_referee:
            return _refuse(result, G6, "G6")
        x0, y0, x1, y1 = result.box
        box = (max(0, x0 - REFEREE_PAD), max(0, y0 - REFEREE_PAD), min(new.width, x1 + REFEREE_PAD),
               min(new.height, y1 + REFEREE_PAD))
        result.numbers["referee"] = referee(new, out, box, icc)
        if old is not None:
            result.numbers["baseline"] = referee(old, old_png, box, icc)
        if not _within(result.numbers["referee"]) or not _within(result.numbers.get("baseline", result.numbers["referee"])):
            return _refuse(result, G6 if old is not None else G6_NEW, "G6")
    # G7: the PNG's shape
    try:
        psd_edit.gate_shape(old_png, out, new)
    except Refused:
        return _refuse(result, G7, "G7")
    result.outcome, result.image, result.data = "re-exported", out, encode(out, old_png)
    # G8: numbered duplicates take D's pixels when D stays clear of their toggled titles
    for path, image in sorted((numbered or {}).items()):
        result.numbered[path] = delta_transfer(old_png, out, image, moved, path)
    return result


def _stale_composite(result: Export, old: Optional[PSDImage], new: PSDImage, old_png: Image.Image, out: Image.Image,
                     icc: bool) -> Export:
    """
    No pixel of the composite moved. A PSD Photoshop did not save may still have changed layers under a stale composite
    (a script's save): the fork referees the changed layers' box, the new render against the composite, when its
    baseline (the old PSD against the old PNG there) holds. Stale: refused; else current.
    """
    if old is None or photoshop_saved(old, new):
        return result
    region, records = changed_layers(old, new)
    if not region or region[2] <= region[0] or region[3] <= region[1]:
        return result
    result.writer = "tool"
    result.numbers.update(records=records, region=list(region), baseline=referee(old, old_png, region, icc))
    if not _within(result.numbers["baseline"]):
        return result                           # the fork can't judge this box: the composite stands
    result.numbers["referee"] = referee(new, out, region, icc)
    if not _within(result.numbers["referee"]):
        return _refuse(result, G6_STALE, "G6")
    return result


def _refuse(result: Export, why: str, gate: str) -> Export:
    result.outcome, result.why, result.gate, result.image, result.data = "refused", why, gate, None, b""
    return result


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


def export_new(new: PSDImage, run_referee: bool = True) -> Export:
    """
    Guard 1 on a new PSD with no PNG (5.5): its composite converted (Export As' default), RGBA when it has
    transparency, refereed over the whole card.

    Returns:
        Export
    """
    composed = composite(new)
    if composed is None:
        return Export("refused", G2, "G2")
    result = Export("current", icc=icc_mode(new, True), writer="new")
    out = convert(new, composed, True)
    if out.mode == "RGBA" and int(np.asarray(out)[..., 3].min()) == 255:
        out = out.convert("RGB")
    result.changed_px = out.width * out.height
    result.box = (0, 0, out.width, out.height)
    if not run_referee:
        return _refuse(result, G6_NEW, "G6")
    result.numbers["referee"] = referee(new, out, result.box, True)
    if not _within(result.numbers["referee"]):
        return _refuse(result, G6_NEW, "G6")
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
    'OUTCOMES',
    'TOL',
    'VOLATILE',
    'changed_layers',
    'composite',
    'convert',
    'delta_transfer',
    'dilate',
    'encode',
    'G6_NEW',
    'G6_STALE',
    'export',
    'export_new',
    'fingerprints',
    'like_mode',
    'open_png',
    'open_psd',
    'photoshop_saved',
    'png_check',
    'referee',
    'xmp',
]
