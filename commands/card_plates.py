"""
Plates for the website's mock-up: the card drawn without one text layer, cut to that layer's box (its effects' reach and
PAD past it), per title and ability layer of its parts (psd_parts; a tagged PSD's layers shown in one other picture
alone, Name #2, against that picture's PNG). The mock-up pastes a plate over the old text instead of guessing the
backdrop from the PNG. One file per PSD blob, crops at the site's card view scale (VIEW_W wide) in one AVIF; a crop
whose backdrop differs from the PNG off the layer's ink is left out (the mock-up's old way). Card checks makes them
(check_card's "plate") and publishes them on the `plates` branch as <PSD blob>.json, those of PSDs gone from main
dropped; the website fetches the open card's in edit mode. With the fonts, a layer's crop also carries the fork's layout
of it (its lines and letters, layout()) when that is the stored pixels'. A card whose stat rows the mock-up's cells
cannot redraw also gets "bars" (with_bars): the bake's own picture of each stat at every value it takes, at that view,
which the mock-up pastes instead; every starred card gets "icons" (with_stars): the bake's own stars at every count,
pasted in place of the mock-up's erased and redrawn ones; a card with header type icons gets "under" (with_under): the
card without them, shown where a types edit takes one away, and "typeProbe" (with_probe): the bake's own drawing of
type icons of two hues it makes there, whose look the mock-up's new icons take. A title under a clipped texture or an
adjustment layer (an Ability under an adjustment) gets "tints": the bake's own pictures of it in 9 colours, mixed for a
whole-field colour.

    python -m commands.card_plates publish --git <checkout> [--head <sha>] <run results.json> ...
"""
from argparse import ArgumentParser
from base64 import b64decode, b64encode
from concurrent.futures import Future, ThreadPoolExecutor
from fractions import Fraction
from functools import cache
from io import BytesIO
from json import dumps, loads
from math import ceil, floor, gcd
from random import uniform
from re import IGNORECASE, fullmatch
from subprocess import CalledProcessError, TimeoutExpired, run as run_process
from sys import argv, stderr
from time import sleep
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageFilter
from psd_tools import PSDImage
from psd_tools.composite import rcache

import commands.psd_edit as psd_edit
from commands.png_export import FX_EXTRA, like_mode
from commands.psd_details import _stats      # at the top: card-checks.yml's sparse list must hold it
from commands.psd_parser import EFFECT_LAYER_TAG, variant_view, variants_of
from commands.psd_regions import comp_bars, fx_pad
from commands.repo_history import batch_env

# ========================
# Configuration
# ========================
VERSION = 1
VIEW_W = 720                                # the site's card view width (the mock-up draws on it)
PAD = 16                                    # px past a layer's box and effects
INK = 2                                     # |full - plate| past this: the layer's own pixels
SEAM = 24                                   # |plate - PNG| past this off the ink: a seam
SEAM_SHARE = 0.002                          # of a crop's off-ink pixels: more, and the crop is left out
GAP = 8                                     # px between crops in the sheet (AVIF blocks stay apart)
QUALITY = 60
KINDS = ("title", "ability")
OVER_SHARE = 0.01                           # of a copy's shown px another layer may cover (Wickerbottom's crest)
LAYOUT_OFF = 0.05                           # card_standards._ink_off past this: kept only when _lines_match
LINE_SHIFT = 6                              # px a line's stored ink may sit off the drawn (live 5 at most)
LINE_SEARCH = 12                            # px searched for that shift
LINE_GROW = 2                               # px the glyphs' ink boxes grow for LINE_COVER
LINE_COVER = 0.99                           # of the stored ink inside them, at least (live 0.997 and up)
TILT = 0.002                                # a layer turned past this (xy, yx over xx): no layout
TEXTURES = ("pixel", "smartobject", "shape", "solidcolorfill", "gradientfill", "patternfill")   # adjustments alone: none
COVER_MIN = 0.5                             # a picture over a text layer covers it where its alpha passes this
COVER_RIM = 2                               # px of fainter alpha kept round that (its edges); a faint overlay: none
BAND = 4                                    # a letter look's band: the layer's box and 1/BAND of its height past it
TINT_RIM = 8                                # px past where a layer's tints differ (the view's LANCZOS reach)
# tints' colours (TTSCardMaker mockup.js PROBES): the RGB cube's corners, then grey
PROBES = ("#000000", "#ff0000", "#00ff00", "#0000ff", "#00ffff", "#ff00ff", "#ffff00", "#ffffff", "#808080")
BRANCH = "plates"
LOCAL_REF = "refs/card-plates/head"         # where publish fetches the branch head (no remote-tracking ref moved)
TRIES = 5
GIT_TIMEOUT = 300
IDENTITY = {"GIT_AUTHOR_NAME": "AutoCroissant", "GIT_AUTHOR_EMAIL": "autocroissant@gmail.com",
            "GIT_COMMITTER_NAME": "AutoCroissant", "GIT_COMMITTER_EMAIL": "autocroissant@gmail.com"}  # git_commit's


# ========================
# Helper Functions
# ========================
def _snap(box: tuple, step: int, size: tuple) -> tuple:
    """box grown to multiples of step (its scaled edges whole pixels), inside size."""
    x0, y0, x1, y1 = box
    return (max(0, x0 // step * step), max(0, y0 // step * step),
            min(size[0], -(-x1 // step) * step), min(size[1], -(-y1 // step) * step))


def _step(psd: PSDImage) -> int:
    """Canvas px whose multiples are whole px at the view's scale (1 past 20)."""
    step = psd.width // gcd(psd.width, VIEW_W)
    return step if step <= 20 else 1


def _rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(like_mode(image, "RGB"), np.int16)


def layer_ids(parts: dict) -> list:
    """The title and ability layers of a details record's parts, in order."""
    return [p["id"] for k in KINDS for p in (parts or {}).get(k) or [] if isinstance(p, dict) and p.get("id") is not None]


def _padded(layer) -> Optional[tuple]:
    pad = max(PAD, fx_pad(layer) + FX_EXTRA)
    x0, y0, x1, y1 = layer.bbox
    return (x0 - pad, y0 - pad, x1 + pad, y1 + pad) if x1 > x0 and y1 > y0 else None


def _meets(layer, other) -> bool:
    """other drawn where layer's padded box is."""
    a, b = _padded(layer), _padded(other) if other.is_visible() else None
    return bool(a and b and b[0] < a[2] and a[0] < b[2] and b[1] < a[3] and a[1] < b[3])


def crop(psd: PSDImage, layer, png: np.ndarray, icc: bool, step: int, extra=()) -> Optional[tuple]:
    """(box, plate image, seam px, ink box) of one layer (canvas px), or None (not drawn, or its backdrop not the PNG's).
    extra: layers the bake redraws with it (its FX layers, copies), hidden too where they meet its box."""
    if layer.kind != "type" or not layer.is_visible() or not layer.opacity:
        return None
    if _padded(layer) is None:
        return None
    got = _without(psd, [[layer, *(other for other in extra if _meets(layer, other))]], png, icc, step)
    if got is None:
        return None
    box, (plate,), seam, (ink,) = got
    ys, xs = np.nonzero(ink)
    return box, plate, seam, (box[0] + int(xs.min()), box[1] + int(ys.min()), box[0] + int(xs.max()) + 1, box[1] + int(ys.max()) + 1)


def covers(psd: PSDImage, layer) -> list:
    """The pictures drawn over layer that meet its padded box: shown pixel, smart-object, shape or fill layers above it in
    a normal blend (their groups' too), neither clipped nor FX layers (the bake moves those with their words)."""
    from psd_tools.constants import BlendMode
    box, above, out = _padded(layer), False, []
    if box is None:
        return out
    for other in psd.descendants():
        if other is layer:
            above = True
            continue
        if not above or other.kind not in TEXTURES or other.clipping or EFFECT_LAYER_TAG in other.name or not other.is_visible():
            continue
        node, plain = other, True
        while node is not None and node is not psd:
            plain = plain and node.blend_mode in (BlendMode.NORMAL, BlendMode.PASS_THROUGH, BlendMode.DISSOLVE) and \
                (node is other or not node.clipping)
            node = node.parent
        b = other.bbox
        if plain and b[0] < box[2] and box[0] < b[2] and b[1] < box[3] and box[1] < b[3]:
            out.append(other)
    return out


def cover_alpha(psd: PSDImage, layer, box: tuple) -> Optional[Image.Image]:
    """covers()' alpha over box (canvas px) as a grey image, drawn alone as the card stacks them, where it passes
    COVER_MIN and COVER_RIM round it; None when it never does (the bake leaves them over the layer's new letters: the
    mock-up puts them back; fainter, they tint the letters, whose colours the mock-up reads off the card)."""
    from psd_tools.composite import composite
    ids = {x.layer_id for x in covers(psd, layer)}
    if not ids:
        return None
    _, _, alpha = composite(psd, viewport=box, layer_filter=lambda x: (x.is_group() and x.is_visible()) or x.layer_id in ids)
    alpha = alpha[..., 0] if alpha.ndim == 3 else alpha
    solid = alpha > COVER_MIN
    if not solid.any():
        return None
    near = np.asarray(Image.fromarray(solid.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * COVER_RIM + 1))) > 0
    return Image.fromarray(np.round(np.clip(np.where(near, alpha, 0), 0, 1) * 255).astype(np.uint8))   # a tint's letters have it


def textured(psd: PSDImage, layer) -> bool:
    """A shown pixel, smart-object or fill layer (TEXTURES) clipped onto the layer or onto a group above it."""
    while layer is not None and layer is not psd:
        if any(clip.visible and clip.kind in TEXTURES for clip in layer.clip_layers):
            return True
        layer = layer.parent
    return False


def letter_look(psd: PSDImage, raw: bytes, layer, icc: bool, step: int) -> Optional[tuple]:
    """
    (box, image, colour) of the card drawn with the layer's letters an opaque band of their own colour ("#rrggbb", its
    most used), across the card and its box with 1/BAND of its height past it both ways (canvas px): a typed letter's
    look under the layer's clipped textures (Earth Totem's stone) and effects, as the bake draws it. None when not
    textured().

    Args:
        raw: The bytes psd was opened from (the band goes on a copy)
    """
    if layer.kind != "type" or not layer.is_visible() or not textured(psd, layer):
        return None
    colour, shape = layer.numpy("color"), layer.numpy("shape")
    if colour is None or shape is None or not (shape[..., 0] > 0.5).any():
        return None
    inked = np.round(colour[shape[..., 0] > 0.5] * 255).astype(np.uint8)
    values, counts = np.unique(inked, axis=0, return_counts=True)
    x0, y0, x1, y1 = layer.bbox
    reach = (y1 - y0) // BAND
    box = _snap((0, y0 - reach, psd.width, y1 + reach), step, psd.size)
    from psd_tools.api.layers import TypeLayer  # the bake's raster writer (psd_edit): its effects and clips kept
    copy = psd_edit.edit_copy(psd, raw)
    band = next(other for other in copy.descendants() if other.layer_id == layer.layer_id)
    fill = values[counts.argmax()]
    pixels = np.ones((box[3] - box[1], box[2] - box[0], 4), np.float32)
    pixels[..., :3] = fill / 255
    TypeLayer._store_raster(band, pixels, box)
    band._invalidate_bbox()
    image = like_mode(psd_edit.render_region(copy, box, icc, edited=(layer.layer_id,)), "RGB")
    return box, image, "#%02x%02x%02x" % tuple(fill)


def adjusted(psd: PSDImage, layer) -> list:
    """The shown adjustment layers over the layer: clipped onto it or a group above it, or above it in its groups."""
    from psd_tools.api.layers import AdjustmentLayer
    out, node = [], layer
    while node is not None and node is not psd:
        kids = list(node.parent)
        out += [x for x in node.clip_layers if x.visible and isinstance(x, AdjustmentLayer)]
        out += [x for x in kids[kids.index(node) + 1:] if x.visible and not x.clipping and isinstance(x, AdjustmentLayer)]
        node = node.parent
    return out


def tint_looks(psd: PSDImage, raw: bytes, layer, field: str, card: str, fonts_dir: str, chart, icc: bool, step: int,
               view: Optional[int] = None) -> Optional[tuple]:
    """
    (box, [image per PROBES], luma) of a textured() or adjusted() title, one under a group's overlay a whole colour turns
    off (psd_edit._owned_group: The Big Yellow Cat's), or an adjusted() Ability (a textured one alone
    tints fine: Pizza's; its box is large): its padded box (canvas px) drawn after the bake's whole-field colour job in
    each probe colour (plan_colors, set_colors on an edit copy; one it calls redundant: the card itself), luma when a
    Gradient Map is over it; the box cut to where the looks differ and TINT_RIM past it. None for other layers, looks
    all alike, or a colour the bake refuses.

    Args:
        field: "name" or "ability"
        card: The card's name (color_layers')
        chart: Returns status_fills() (an Ability's status words keep its colours)
    """
    from psd_tools.api.adjustments import GradientMap
    over = adjusted(psd, layer)
    if layer.kind != "type" or not layer.is_visible() or _padded(layer) is None or not (over or field == "name" and (
            textured(psd, layer) or psd_edit._owned_group(layer, psd_edit.color_layers(psd, field, card, view)))):
        return None
    box, images = _snap(_padded(layer), step, psd.size), []
    colours = chart() if chart and field == "ability" else None
    for colour in PROBES:
        one = {"field": field, "color": colour}
        how = psd_edit.plan_colors(psd, [one], card, chart=colours, view=view)[0][1]
        if how == "redundant":
            images.append(like_mode(psd_edit.render_region(psd, box, icc), "RGB"))
            continue
        if how != "baked":
            return None
        copy = psd_edit.edit_copy(psd, raw)
        edit = psd_edit.set_colors(copy, [one], fonts_dir, card, colours, view)
        images.append(like_mode(psd_edit.render_region(copy, box, icc, edited=psd_edit.edited_ids([edit])), "RGB"))
    arrays = [_rgb(image) for image in images]
    ys, xs = np.nonzero(np.logical_or.reduce([np.abs(a - arrays[0]).max(-1) > INK for a in arrays[1:]]))
    if not len(ys):
        return None
    cut = _snap((box[0] + int(xs.min()) - TINT_RIM, box[1] + int(ys.min()) - TINT_RIM, box[0] + int(xs.max()) + 1 + TINT_RIM,
                 box[1] + int(ys.max()) + 1 + TINT_RIM), step, psd.size)
    cut = (max(cut[0], box[0]), max(cut[1], box[1]), min(cut[2], box[2]), min(cut[3], box[3]))
    images = [image.crop((cut[0] - box[0], cut[1] - box[1], cut[2] - box[0], cut[3] - box[1])) for image in images]
    return cut, images, any(isinstance(x, GradientMap) for x in over)


def _without(psd: PSDImage, groups: list, png: np.ndarray, icc: bool, step: int, tight: int = 0) -> Optional[tuple]:
    """
    (box, [image], seam px, [ink mask]) of the card drawn without each group of layers in turn, over their padded boxes
    (canvas px), or None: none draws there, or a drawing's backdrop (off every group's ink) is not the PNG's. tight: the
    box cut to their ink and this many px past it.
    """
    boxes = [b for hide in groups for b in map(_padded, hide) if b]
    if not boxes:
        return None
    box = _snap((min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)), step, psd.size)
    full = _rgb(psd_edit.render_region(psd, box, icc))
    images, arrays, inks = [], [], []
    for hide in groups:
        eyes = [(x, x.visible) for x in hide]
        for x in hide:
            x.visible = False
        try:
            plate = psd_edit.render_region(psd, box, icc)
        finally:
            for x, eye in eyes:
                x.visible = eye
        images.append(like_mode(plate, "RGB"))
        arrays.append(_rgb(plate))
        inks.append(np.abs(full - arrays[-1]).max(-1) > INK)
    every = np.logical_or.reduce(inks)
    if not every.any():
        return None
    if tight:                                   # a layer's box far past its pixels (a card-sized smart object)
        ys, xs = np.nonzero(every)
        cut = _snap((box[0] + int(xs.min()) - tight, box[1] + int(ys.min()) - tight, box[0] + int(xs.max()) + 1 + tight,
                     box[1] + int(ys.max()) + 1 + tight), step, psd.size)
        cut = (max(cut[0], box[0]), max(cut[1], box[1]), min(cut[2], box[2]), min(cut[3], box[3]))
        x0, y0, x1, y1 = (cut[0] - box[0], cut[1] - box[1], cut[2] - box[0], cut[3] - box[1])
        box, images = cut, [image.crop((x0, y0, x1, y1)) for image in images]
        arrays, inks = [a[y0:y1, x0:x1] for a in arrays], [ink[y0:y1, x0:x1] for ink in inks]
    seam, off = 0, ~np.logical_or.reduce(inks)  # off every group's ink: the backdrop
    for b in arrays:
        n = int(((np.abs(b - png[box[1]:box[3], box[0]:box[2]]).max(-1) > SEAM) & off).sum())
        if n > SEAM_SHARE * off.sum():
            return None
        seam += n
    return box, images, seam, inks


def layout(layer, fonts_dir: str, stored: bool = True) -> Optional[dict]:
    """
    The fork's layout of a type layer in canvas px, when it draws the stored pixels (card_standards._ink_off within
    LAYOUT_OFF, or past it each line still theirs: _lines_match) and is not turned (TILT): {"text": the engine's text
    (\\r paragraphs, \\x03 breaks), "lines": [[start, end, baseline, left, right, top, bottom, justification]],
    "glyphs": [[char index, pen x, ink left, ink right]], "frame": [left, right] or None (point text)}, else None.
    start/end index text (end: the break excluded); baseline: its letters' (BaselineShift in); left/right: the pen's
    ends; top/bottom: the ink's, underlines too; a blank glyph has no ink ends. stored False: also when it doesn't draw
    them (a warped layer: the bake re-lays it unwarped).
    """
    from statistics import median
    from psd_tools.composite.text.geometry import text_geometry
    from psd_tools.composite.text.layout import frame, layout as lay_out
    from psd_tools.composite.text.model import TextModel
    from psd_tools.composite.text import capture, render
    from commands.card_standards import _ink_off
    faces = psd_edit.font_map(fonts_dir, layer)
    model = TextModel.from_layer(layer)
    xx, xy, yx, yy, tx, ty = model.transform
    if xx <= 0 or yy <= 0 or max(abs(xy), abs(yx)) > TILT * xx:
        return None
    geo = text_geometry(layer, faces)
    pair = (capture(layer, faces), render(layer, fonts=faces)) if stored else None
    off = stored and _ink_off(layer, faces, geo, pair) > LAYOUT_OFF
    lay = lay_out(model, faces)
    if (geo["missing_fonts"] or not geo["lines"] or not all(line["drawn"] for line in geo["lines"])
            or [g["i"] for g in geo["glyphs"]] != [g.cluster for g in lay.glyphs]):
        return None
    r = lambda v: round(v, 1)
    ys, ink = {}, {}
    for g, at in zip(geo["glyphs"], lay.glyphs):    # text_geometry's line baseline leaves out BaselineShift
        ys.setdefault(g["line"], []).append(xy * at.x + yy * at.y + ty)
        if g["ink"]:
            top, bottom = ink.get(g["line"], (g["ink"][1], g["ink"][3]))
            ink[g["line"]] = (min(top, g["ink"][1]), max(bottom, g["ink"][3]))
    pen = [(line["baseline"][1] + line["baseline"][3]) / 2 for line in geo["lines"]]
    shift = median([median(ys[n]) - y for n, y in enumerate(pen) if n in ys] or [0])
    base = [median(ys[n]) if n in ys else y + shift for n, y in enumerate(pen)]     # a blank line: as the others
    for rule in lay.rules:                          # underlines and strikes
        y0, y1 = (yy * (rule.y + d) + ty for d in (-rule.thickness, rule.thickness))
        n = min(range(len(base)), key=lambda k: abs(base[k] - (y0 + y1) / 2))
        top, bottom = ink.get(n, (y0, y1))
        ink[n] = (min(top, y0), max(bottom, y1))
    lines = []
    for n, line in enumerate(geo["lines"]):
        just = next((p.justification for p in model.paragraphs if p.start <= line["start"] < max(p.end, p.start + 1)),
                    0)
        top, bottom = ink.get(n, (base[n], base[n]))
        lines.append([line["start"], line["end"], r(base[n]), r(line["baseline"][0]), r(line["baseline"][2]), r(top),
                      r(bottom), just])
    glyphs = [[g["i"], r(g["cell"][0]), *(r(g["ink"][k]) for k in (0, 2) if g["ink"])] for g in geo["glyphs"]]
    if off and not _lines_match(*pair, lines, glyphs):
        return None
    box = frame(model)
    return {"text": geo["text"], "lines": lines, "glyphs": glyphs,
            "frame": [r(xx * box[0] + tx), r(xx * (box[0] + box[2]) + tx)] if box else None}


def _alpha(found, window: tuple) -> np.ndarray:
    """A capture's or render's alpha on window (canvas px), 0 off it."""
    x0, y0, x1, y1 = window
    out = np.zeros((y1 - y0, x1 - x0), np.float32)
    if found.rgba is not None and found.bbox is not None:
        bx0, by0, bx1, by1 = (int(v) for v in found.bbox)
        cx0, cy0, cx1, cy1 = max(bx0, x0), max(by0, y0), min(bx1, x1), min(by1, y1)
        if cx1 > cx0 and cy1 > cy0:
            a = np.asarray(found.rgba)[cy0 - by0:cy1 - by0, cx0 - bx0:cx1 - bx0, 3].astype(np.float32)
            out[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = a / 255 if found.rgba.dtype == np.uint8 else a
    return out


def _lines_match(stored, drawn, lines: list, glyphs: list) -> bool:
    """
    Whether a layout's lines are the stored pixels' though the fork draws them off (layout() past LAYOUT_OFF): on each
    line's ink band the stored and drawn ink within LINE_SHIFT px across, every inked glyph's box with stored ink, and
    LINE_COVER of the stored ink in the glyphs' boxes (LINE_GROW px out). A word on another line (Monkey D Letion's
    Ability) fails.
    """
    if stored.rgba is None or stored.bbox is None:
        return False
    m, xs = LINE_SEARCH + LINE_GROW + 2, [x for n in lines for x in n[3:5]] + [x for one in glyphs for x in one[2:]]
    tops, bottoms = floor(min(n[5] for n in lines)), ceil(max(n[6] for n in lines))
    window = (min(int(stored.bbox[0]), floor(min(xs))) - m, min(int(stored.bbox[1]), tops) - m,
              max(int(stored.bbox[2]), ceil(max(xs))) + m, max(int(stored.bbox[3]), bottoms) + m)
    a, d = _alpha(stored, window), _alpha(drawn, window)
    ink, boxes = a > 0.5, np.zeros(a.shape, bool)
    x0, y0, g = window[0], window[1], LINE_GROW
    for start, end, _, _, _, top, bottom, _ in lines:
        inked = [one for one in glyphs if start <= one[0] < end and len(one) > 3]
        if not inked:
            continue
        t, b = floor(top) - y0, ceil(bottom) - y0
        band = a[t - 2:b + 2].sum(0), d[t - 2:b + 2].sum(0)
        dx = max(range(-LINE_SEARCH, LINE_SEARCH + 1),
                 key=lambda k: (float((band[0] * np.roll(band[1], k)).sum()), -abs(k)))
        if abs(dx) > LINE_SHIFT:
            return False
        for _, _, left, right in inked:
            boxes[t - g:b + g, floor(left) - x0 - g:ceil(right) - x0 + g] = True
            if right - left >= 2 and not ink[t - 2:b + 2, floor(left) - x0 - 1:ceil(right) - x0 + 1].any():
                return False
    return bool(ink.any()) and float((ink & boxes).sum()) >= LINE_COVER * int(ink.sum())


def _layout(layer, fonts_dir: Optional[str], relaid: bool = False) -> dict:
    """layout()'s keys, or {}; relaid: else {"relaid": [[left, top, right, bottom]]}, its lines as the bake sets them."""
    try:
        got = layout(layer, fonts_dir) if fonts_dir else None
        if got or not relaid or not fonts_dir:
            return got or {}
        got = layout(layer, fonts_dir, stored=False)
        return {"relaid": [[left, top, right, bottom] for _, _, _, left, right, top, bottom, _ in got["lines"]]} if got else {}
    except Exception as error:                  # noqa: BLE001 - no layout: the mock-up finds the lines
        print(f"{layer.name}: no layout: {type(error).__name__}: {error}", file=stderr)
        return {}


def _over(psd: PSDImage, layer, box: tuple, shown: np.ndarray, cropped: set) -> Optional[list]:
    """The ids of the cropped layers drawn over layer where it shows (shown: a mask of box, canvas px), or None:
    another layer covers more than OVER_SHARE of it."""
    above, out = False, []
    for other in psd.descendants():
        if other is layer:
            above = True
            continue
        b = other.bbox if above and other.kind != "group" and other.is_visible() else (0, 0, 0, 0)
        x0, y0, x1, y1 = max(b[0], box[0]), max(b[1], box[1]), min(b[2], box[2]), min(b[3], box[3])
        if x1 <= x0 or y1 <= y0 or not shown[y0 - box[1]:y1 - box[1], x0 - box[0]:x1 - box[0]].any():
            continue
        if other.layer_id in cropped:
            out.append(other.layer_id)
            continue
        try:
            alpha = np.asarray(other.topil().convert("RGBA"))[y0 - b[1]:y1 - b[1], x0 - b[0]:x1 - b[0], 3] > 0
        except Exception:                       # noqa: BLE001 - no pixels to tell: it covers it
            return None
        if (alpha & shown[y0 - box[1]:y1 - box[1], x0 - box[0]:x1 - box[0]]).sum() > OVER_SHARE * shown.sum():
            return None
    return out


def _fill(layer, px: np.ndarray) -> Optional[list]:
    """The median colour of a layer's shown pixels (px: [n, 3]) off its Stroke's (psd_details'), or None (too few)."""
    from commands.psd_details import _stroke
    stroke = _stroke(layer)
    if stroke:
        px = px[np.abs(px - np.array(stroke[:3])).max(-1) > 60]
    return [int(v) for v in np.median(px, 0)] if len(px) >= 20 else None


def _stacked(psd: PSDImage, layer, copies: list, png: np.ndarray, icc: bool, step: int) -> list:
    """
    Copies of layer stacked on it (its words moved a little: Tainted Eden's red and cyan titles), hidden in its crop:
    [{"id", "at": [dx, dy] canvas px from it, "above": drawn over it, "ch": the RGB channels it paints (its blending
    restrictions; all: none), "fill": its colour where it shows (its own overlay's; none: the layer's)}]. A copy of
    other words or size, or an overlay whose colour can't be read: none.
    """
    if not copies:
        return []
    from psd_tools.api.effects import ColorOverlay, GradientOverlay, PatternOverlay
    from psd_tools.constants import Tag
    order = {x.layer_id: n for n, x in enumerate(psd.descendants())}
    out = []
    for copy in copies:
        try:
            a, b = layer.transform, copy.transform
            same = copy.text.strip() == layer.text.strip() and tuple(a[:4]) == tuple(b[:4])
        except Exception:                       # noqa: BLE001 - text psd-tools can't read
            same = False
        if not same:
            continue
        one = {"id": copy.layer_id, "at": [round(b[4] - a[4], 1), round(b[5] - a[5], 1)],
               "above": order[copy.layer_id] > order[layer.layer_id]}
        blocks = copy._record.tagged_blocks
        shut = (blocks.get_data(Tag.CHANNEL_BLENDING_RESTRICTIONS_SETTING) if blocks else None) or []
        if any(c in shut for c in range(3)):
            one["ch"] = [c for c in range(3) if c not in shut]
        if copy.effects.enabled and any(isinstance(e, (ColorOverlay, GradientOverlay, PatternOverlay)) and e.enabled
                                        for e in copy.effects):
            got = _without(psd, [[copy]], png, icc, step)
            if got is None:
                continue
            box, (plate,), _, _ = got
            shown = png[box[1]:box[3], box[0]:box[2]]
            one["fill"] = _fill(copy, shown[np.abs(shown - _rgb(plate)).max(-1) > SEAM])
            if not one["fill"]:
                continue
        out.append(one)
    return out


def plates(psd: PSDImage, png: Image.Image, parts: dict, fonts_dir: Optional[str] = None,
           pictures: Optional[dict] = None, raw: Optional[bytes] = None, card: str = "",
           chart: Optional[Callable] = None) -> Optional[dict]:
    """
    The plates file of a card: {"v", "w", "crops": [{"id", "box", "ink", "at"}], "img"} (+ "letters", "tints", "covers"),
    or None when no layer gave one.
    box: the crop, ink: where the layer drew in it (canvas px); at: [x, y, w, h] of the crop in img (a data URL). A
    layer with FX layers on it has a second crop without them too, its "fx" their status words (the mock-up's when
    one leaves the text: the bake moves FX layers with their words). With fonts_dir, a layer's first crop also has
    its layout() keys ("text", "lines", "glyphs", "frame"). Its copies stacked on it (parts' follow) are hidden in its
    crops, the first one's "stacked" (_stacked) telling how the bake draws them; a copy or watermark elsewhere has a crop
    of its own ("fill": its colour where it shows, "over": the cropped layers drawn over it, "relaid": _layout's), one
    that never shows none. With raw, a textured layer's letter_look() is in img too: "letters": [{"id", "fill", "box",
    "at"}] (the mock-up fills typed letters of that colour with it); with raw and fonts_dir, a textured or adjusted
    layer's tint_looks() too: "tints": [{"id", "box", "at": [a cut per PROBES], "luma"}] (the mock-up's whole-field
    colour, mixed from them). A cropped layer under pictures (cover_alpha: Papyrus's spaghetti) has their alpha over its
    first crop's box, at the view's scale, a PNG of its own: "covers": [{"id", "box", "img"}] (the mock-up puts them
    back over its new letters).

    Args:
        psd: The card
        png: Its PNG (picture 1)
        parts: Its details record's parts (psd_parts.card_parts)
        fonts_dir: MDW/Fonts (the layouts'); None: none
        pictures: A tagged PSD's {n: picture n's PNG}: the layers shown in picture n alone (Name #2) cropped against it
        raw: The bytes psd was opened from (letter looks, tints); None: none
        card: The card's name (tints')
        chart: Returns status_fills() (tints')
    """
    if psd.width <= 0 or png.size != psd.size:
        return None
    scale, step = VIEW_W / psd.width, _step(psd)
    icc = psd_edit.export_icc(psd, png)
    by_id = {layer.layer_id: layer for layer in psd.descendants()}
    follow = [f for f in (parts or {}).get("follow") or [] if isinstance(f, dict)]
    found, looks, tints, covered, cropped = [], [], [], [], set()
    fields = {p["id"]: "name" if k == "title" else "ability" for k in KINDS for p in (parts or {}).get(k) or []
              if isinstance(p, dict) and p.get("id") is not None}
    tagged = len(variants_of(psd)) > 1
    views = [(1 if tagged else None, png, dict.fromkeys(layer_ids(parts)))]
    for n, picture in sorted((pictures or {}).items()) if tagged else ():
        ids = [p["id"] for k in KINDS for p in (parts or {}).get(k) or [] if isinstance(p, dict) and p.get("variants") == [n]]
        if n > 1 and ids and picture.size == psd.size:
            views.append((n, picture, ids))
    for n, picture, ids in views:
        truth = _rgb(picture)
        with variant_view(psd, n):
            for lid in ids:
                layer = by_id.get(lid)
                if layer is None:
                    continue
                copies = [by_id[f["id"]] for f in follow if lid in (f.get("of") or []) and f.get("id") in by_id]
                got = crop(psd, layer, truth, icc, step, copies)
                if got:
                    stacked = _stacked(psd, layer, [c for c in copies if _meets(layer, c)], truth, icc, step)
                    found.append(({"id": lid, **_layout(layer, fonts_dir), **({"stacked": stacked} if stacked else {})},
                                  *got))
                    cropped.add(lid)
                    try:
                        cover = cover_alpha(psd, layer, got[0])
                    except Exception as error:      # noqa: BLE001 - none: the mock-up's letters stay on top
                        print(f"{layer.name}: no cover: {type(error).__name__}: {error}", file=stderr)
                        cover = None
                    if cover is not None:
                        covered.append(({"id": lid}, got[0], cover))
                fx = [x for x in psd_edit.fx_layers(psd, layer) if _meets(layer, x)]
                got = crop(psd, layer, truth, icc, step, [*copies, *fx]) if fx else None
                if got:
                    words = [w.strip() for x in fx for w in x.name.split(EFFECT_LAYER_TAG, 1)[1].split("#")[0].split("|") if w.strip()]
                    found.append(({"id": lid, "fx": list(dict.fromkeys(words))}, *got))
                try:
                    look = letter_look(psd, raw, layer, icc, step) if raw is not None else None
                except Exception as error:          # noqa: BLE001 - none: flat letters
                    print(f"{layer.name}: no letter look: {type(error).__name__}: {error}", file=stderr)
                    look = None
                if look:
                    looks.append(({"id": lid, "fill": look[2]}, look[0], look[1], None, None))
                try:
                    tint = tint_looks(psd, raw, layer, fields[lid], card, fonts_dir, chart, icc, step, n) \
                        if raw is not None and fonts_dir else None
                except Exception as error:          # noqa: BLE001 - none: the mock-up tints the letters
                    print(f"{layer.name}: no tints: {type(error).__name__}: {error}", file=stderr)
                    tint = None
                if tint:
                    tints.append(({"id": lid, **({"luma": True} if tint[2] else {})}, tint[0], tint[1]))
            for f in follow:                    # a copy or watermark elsewhere: its own crop
                one, of = by_id.get(f.get("id")), [by_id[i] for i in f.get("of") or [] if i in by_id and i in ids]
                if one is None or not of or one.layer_id in cropped or any(_meets(x, one) for x in of):
                    continue
                got = crop(psd, one, truth, icc, step)
                if got:
                    box, plate = got[0], got[1]
                    shown = np.abs(truth[box[1]:box[3], box[0]:box[2]] - _rgb(plate)).max(-1) > SEAM
                over = _over(psd, one, box, shown, cropped) if got else None
                fill = _fill(one, truth[box[1]:box[3], box[0]:box[2]][shown]) if over is not None else None
                if fill:
                    found.append(({"id": one.layer_id, **_layout(one, fonts_dir, True), "fill": fill, **({"over": over} if over else {})},
                                  *got))
                    cropped.add(one.layer_id)
    if not found and not looks and not tints:
        return None
    entries = [(key, box, image, ink) for key, box, image, _, ink in found + looks]
    entries += [(key, box, image, "tint") for key, box, images in tints for image in images]
    small = [(key, box, ink, image.resize((max(1, round((box[2] - box[0]) * scale)), max(1, round((box[3] - box[1]) * scale))),
                                          Image.LANCZOS)) for key, box, image, ink in entries]
    width, last, crops, letters, tinted, parts, y = max(im.width for *_, im in small), len(small) - 1, [], [], {}, [], 0
    for n, (key, box, ink, im) in enumerate(small):     # gaps and the right side: edge pixels, no seam bleeds in
        top, bottom = (GAP // 2 if n else 0), (GAP - GAP // 2 if n < last else 0)
        parts.append(np.pad(np.asarray(im), ((top, bottom), (0, width - im.width), (0, 0)), mode="edge"))
        at = [0, y + top, im.width, im.height]
        if ink == "tint":                       # a look per probe, in PROBES' order
            tinted.setdefault(id(key), {**key, "box": list(box), "at": []})["at"].append(at)
        elif ink is None:
            letters.append({**key, "box": list(box), "at": at})
        else:
            crops.append({**key, "box": list(box), "ink": list(ink), "at": at})
        y += top + im.height + bottom
    buf = BytesIO()
    Image.fromarray(np.concatenate(parts)).save(buf, "AVIF", quality=QUALITY)
    shut = []                                   # apart, lossless: the sheet's pixels stay as they were
    for key, box, image in covered:
        one = BytesIO()
        image.resize((max(1, round((box[2] - box[0]) * scale)), max(1, round((box[3] - box[1]) * scale))),
                     Image.LANCZOS).save(one, "PNG", optimize=True)
        shut.append({**key, "box": list(box), "img": "data:image/png;base64," + b64encode(one.getvalue()).decode()})
    return {"v": VERSION, "w": psd.width, "crops": crops, **({"letters": letters} if letters else {}),
            **({"tints": list(tinted.values())} if tinted else {}), **({"covers": shut} if shut else {}),
            "img": "data:image/avif;base64," + b64encode(buf.getvalue()).decode()}


# ========================
# Stat bars: the bake's picture of every value
# ========================
BAR_TOP = 20                                # values tried, 0 to this; the bake refuses what its bars cannot show
BAR_MOVED = 2                               # view levels: a bar pixel moved less shows the site's own image
BAR_BYTES = 192_000                         # a lossless bar sheet past this goes lossy (BAR_AVIF)
BAR_AVIF = 55                               # AVIF quality (screen tuning): under the site's 720 px image's noise (q50)
BAR_SIDE = 16383                            # WebP's largest side: the sheet's cuts stack in columns of this height
BAR_THREADS = 2                             # bar_sheet(later=True)'s encoders (Pillow's WebP lets the GIL go)
WEBP_URL = "data:image/webp;base64,"
REACH = 3                                   # LANCZOS's reach in view rows
EDGE = REACH + 1                            # view rows at a crop's cut whose clipped reach isn't view()'s
UNDER_GROW = 2                              # view px the under mask reaches past the icons' ink (LANCZOS's rim)
TYPE_PROBE = (("fire", "electric"), ("water", "ice"))   # type_probe's warm and cool type: a pair's second on a card
PROBE_TYPES = tuple(name for pair in TYPE_PROBE for name in pair)   # printing its first


def _view_mode(image: Image.Image) -> str:
    return "RGBA" if image.mode in ("RGBA", "LA", "P") else "RGB"


def view(image: Image.Image) -> Image.Image:
    """image as the site's VIEW_W card image before its encode (build_site.make_images: its mode, thumbnail, LANCZOS)."""
    image = image.convert(_view_mode(image))
    image.thumbnail((VIEW_W, VIEW_W * 3), Image.LANCZOS)
    return image


def _base(png: Image.Image) -> np.ndarray:
    """png's array in view()'s mode (_band's)."""
    return np.asarray(png.convert(_view_mode(png)))


# The mock-up's own stat rows (TTSCardMaker site/assets/mockup.js TEMPLATES bars, barRows, isBar, labelRow, rowShows),
# ported with a margin: a row it may miss gets the bake's pictures
ROWS = {"tall": ((0.3, 0.5, 0.978, 0.535), 0.0525), "item": ((0.308, 0.624, 0.986, 0.671), 0.053)}
STAT_NAME = {"hp": r"(hp|health)", "def": r"(def|defen[cs]e|defemse)", "atk": r"(atk|attack)", "spd": r"(spd|speed)"}
SURE = 1.2                                  # the mock-up's thresholds this much past theirs, both ways (live thumbs: 1.15)


def _round(v: float) -> int:
    return floor(v + 0.5)                   # JavaScript's Math.round


def _shape(w: int, h: int) -> str:
    r = h / w
    return "tall" if 1.9 < r < 2.1 else "item" if 1.4 < r < 1.6 else "square" if 0.9 < r < 1.1 else ""


def _px(box, w: int, h: int) -> tuple:
    return tuple(min(max(_round(v), 0), top) for v, top in zip(box, (w, h, w, h)))


def _is_bar(L: np.ndarray, box, k: float = 1.0) -> bool:
    """mockup.js isBar, its thresholds times k (k > 1: surely a bar; k < 1: maybe one)."""
    h, w = L.shape
    x0, y0, x1, y1 = _px(box, w, h)
    cw, y = (x1 - x0) / 10, _round((y0 + y1) / 2)
    if cw < 3 or y <= 0 or y >= h:
        return False
    at = lambda x: sum(int(L[min(max(y + d, 0), h - 1), min(max(_round(x), 0), w - 1)]) for d in (-1, 0, 1)) / 3
    jump = lambda x: max(abs(at(x + 1) - at(x - 1)), abs(at(x + 2) - at(x - 2)))
    border = sum(max(jump(x0 + j * cw - 1), jump(x0 + j * cw), jump(x0 + j * cw + 1)) for j in range(1, 10))
    inside = sum(jump(x0 + (j - 0.5) * cw) for j in range(1, 10))
    return border / 9 > 12 * k and border > 2 * k * inside


def _shows(L: np.ndarray, box, v: int, k: float = 1.0) -> bool:
    """mockup.js rowShows, its 8 levels times k."""
    if not 0 < v < 10:
        return True
    h, w = L.shape
    x0, y0, x1, y1 = _px(box, w, h)
    cw, m = (x1 - x0) / 10, []
    for c in range(10):
        ys = range(_round(y0 + (y1 - y0) * 0.3), ceil(y0 + (y1 - y0) * 0.7))
        xs = range(_round(x0 + (c + 0.25) * cw), ceil(x0 + (c + 0.75) * cw))
        cell = L[ys.start:ys.stop, xs.start:xs.stop]
        m.append(float(cell.mean()) if cell.size else 0.0)
    return min(m[:v]) > max(m[v:]) + 8 * k or max(m[:v]) + 8 * k < min(m[v:])


def mock_misses(image: Image.Image, psd: PSDImage, card: dict, kinds: dict, sure: float = SURE) -> list:
    """The stats of card the mock-up may not find a row for on image (the card's PNG at the site's view, view()), or
    whose drawn row may not show the card's value: its finder (ROWS) with SURE's margin."""
    rgb = np.asarray(image.convert("RGB"), np.int32)
    L = ((rgb[..., 0] * 77 + rgb[..., 1] * 150 + rgb[..., 2] * 29) >> 8).astype(np.uint8)
    h, w = L.shape
    stats = [stat for stat in psd_edit.STAT_FIELDS if card.get(stat) is not None]
    rows = ROWS.get(_shape(w, h))
    keys = [stat for stat in psd_edit.STAT_FIELDS if card.get(stat) is not None or kinds.get(stat) == "template"]
    if rows is None or any(kinds.get(stat) not in ("template", "pixel") for stat in keys):
        return stats
    (b0, b1, b2, b3), step = rows
    k = w / psd.width
    labels = [layer for layer in psd.descendants() if layer.kind == "type" and layer.is_visible()]
    out = []
    for n, stat in enumerate(keys):
        box = (b0 * w, (b1 + n * step) * h, b2 * w, (b3 + n * step) * h)
        if not _is_bar(L, box, sure):
            label = None
            for layer in labels if not _is_bar(L, box, 1 / sure) else ():     # maybe the template's row: a miss
                try:
                    text = (layer.text or "").replace("\ufeff", "").strip()
                except Exception:               # text psd-tools can't read: not a label
                    continue
                if fullmatch(STAT_NAME[stat], text, IGNORECASE):
                    label = layer
                    break
            if label is None:
                out.append(stat)
                continue
            mid, half, q = (label.bbox[1] + label.bbox[3]) / 2 * k, (box[3] - box[1]) / 2, (box[2] - box[0]) / 40
            at = lambda d: (box[0] + d, mid - half, box[2] + d, mid + half)
            if not _is_bar(L, at(0), sure) or _is_bar(L, at(-q), 1 / sure) or _is_bar(L, at(q), 1 / sure):
                out.append(stat)
                continue
            box = at(0)
        if kinds.get(stat) == "pixel" and not _shows(L, box, card.get(stat) or 0, sure):
            out.append(stat)
    return [stat for stat in out if stat in stats]


def bar_value(psd: PSDImage, raw: bytes, png: Image.Image, parsed: dict, parser, rel: str, stat: str, value: int,
              icc: bool, creature: int = 1, caps=..., seen: Optional[list] = None) -> Optional[Image.Image]:
    """
    The card's PNG after a website edit of one stat, as the bake makes it (psd_edit._apply_jobs' steps for a stat job:
    classify, set_stat on a copy, gate_dirty, baseline, the merge), or None when the bake refuses it or the re-parse
    would not read the value. The saved file's gates (gate_bytes, stories, filters, fx, idempotence) are not run:
    saving every value is too slow here, and no stat value only they refuse is known (Chef, Computer Virus HP 0-20).
    creature: past 1, that creature's stat on a card of several (a `creatures` job, set_creatures); None when the job
    moves another creature too (a shared bar: creature 1's picture).
    caps: stat_caps(psd) when read already (bar_changes), else classify reads it.
    seen: gets looks() of the PNG against this value's picture, summed over the job's edits, when the bake makes it or
    refuses only for the PNG's sake (baseline's parity floor: the PNG far off the card's own picture there; no picture).
    """
    if creature == 1:
        job = psd_edit.classify(parsed, {stat: value}, psd, parser, caps=caps).jobs.get(stat)
    else:
        wanted = [dict(one) for one in parsed["card"]["creatures"]]
        wanted[creature - 1][stat] = value
        job = psd_edit.classify(parsed, {"creatures": wanted}, psd, parser, caps=caps).jobs.get("creatures")
        if job is not None and any(one for n, one in enumerate(job) if n != creature - 1):
            return None
    if job is None:
        return None
    try:
        new = psd_edit.edit_copy(psd, raw)
        made = psd_edit.set_creatures(new, parsed, job) if creature > 1 \
            else [psd_edit.set_stat(new, parsed, stat, value)]
        edits = [edit for edit in made if edit.layers or edit.removed]
        if not edits:
            return None
        kept, merged, edited, votes, far = {}, png, psd_edit.edited_ids(edits), np.zeros(2, np.int64), False
        first = psd_edit.gate_dirty(psd, new, edits, kept)
        for edit in edits:
            try:
                base, measured = psd_edit.baseline(psd, png, edit.region, icc, first)
            except psd_edit.Refused as refused:
                if seen is None or refused.code != "parity":
                    raise
                cut, far = psd_edit.crop_pair(psd, first, edit.region, icc), True
                base, measured = cut[0] if cut else psd_edit.render_region(psd, edit.region, icc), {"text": "preview"}
            cut = psd_edit.crop_pair(new, kept.get("after"), edit.region, icc, kept.get("box")) \
                if measured["text"] == "preview" and kept.get("edited") == edited else None
            render = cut[0] if cut else psd_edit.render_region(new, edit.region, icc, measured["text"], edited)
            merged, replaced = psd_edit.merge_png(merged, base, render, edit.region)
            if not replaced.any() and (far or not psd_edit._within_bar(measured)):
                return None
            if seen is not None:
                votes += looks(png.crop(edit.region), base, render, replaced)
    except psd_edit.Refused:
        return None
    if parser is not None:                      # gate_reparse's
        got = psd_edit.parse_card(parser, rel, rel, new)["card"]
        if creature > 1:
            got = dict(enumerate(got.get("creatures") or [], 1)).get(creature, {})
        if got.get(stat) != value:
            return None
    if seen is not None:
        seen.append(tuple(int(v) for v in votes))
    return None if far else merged


def looks(png: Image.Image, base: Image.Image, render: Image.Image, changed: np.ndarray) -> np.ndarray:
    """
    Whether a PNG shows a value's picture or the card's own over the pixels the value changes (all region-sized):
    [sum of how much nearer the PNG is to render than to base, sum of how far apart the two are]; their ratio runs
    from -1 (the card's own) to 1 (the value's).
    """
    ys, xs = np.nonzero(changed)
    if not len(xs):
        return np.zeros(2, np.int64)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    png, base, render = (image.crop(box) for image in (png, base, render))
    changed = changed[box[1]:box[3], box[0]:box[2]]
    near = psd_edit._px_diff(png, base).astype(np.int64) - psd_edit._px_diff(png, render)
    return np.array([near[changed].sum(), psd_edit._px_diff(base, render)[changed].sum(dtype=np.int64)])


def star_value(psd: PSDImage, raw: bytes, png: Image.Image, parsed: dict, parser, rel: str, value: int,
               icc: bool) -> Optional[Image.Image]:
    """
    The card's PNG after a website edit of its stars, as the bake makes it (card_bake.plan_move: a folder card moves into
    its "N Stars" folder; psd_edit._apply_jobs' steps for a lone stars job: plan_stars, set_icons, gate_dirty, baseline,
    the merge, gate_reparse), or None when the bake refuses it or keeps the printed stars. bar_value's: no save gates.
    """
    if "stars" not in psd_edit.classify(parsed, {"stars": value}, psd, parser).jobs:
        return None
    target = psd_edit.star_target(rel, value)
    target = f"{target}/{rel.rsplit('/', 1)[-1]}" if target else rel
    try:
        if target != rel:                       # plan_move's: refused, the printed stars stay
            psd_edit.plan_stars(psd, parsed, value, parser)
        new = psd_edit.edit_copy(psd, raw)
        edit = psd_edit.set_icons(new, psd_edit.plan_stars(new, parsed, value, parser, split=True), None,
                                  target)
        first = psd_edit.gate_dirty(psd, new, [edit], icon=True)
        if not (edit.layers or edit.removed or edit.icons.removed):   # an undo may only delete (D14)
            return None
        base, measured = psd_edit.baseline(psd, png, edit.region, icc, first)
        render = psd_edit.render_region(new, edit.region, icc, measured["text"], psd_edit.edited_ids([edit]))
        merged, replaced = psd_edit.merge_png(png, base, render, edit.region)
        if not replaced.any() and not psd_edit._within_bar(measured):
            return None
        psd_edit.gate_reparse(parser, parsed, target, target, {"stars": value}, new, [edit], [])
    except Exception:                           # noqa: BLE001 - refused or the bake's error: no picture of that count
        return None
    return merged


def _band(was: np.ndarray, merged: Image.Image, base: np.ndarray) -> Optional[tuple]:
    """(first row, rows) of view(merged) where merged differs from base (_base of the PNG): a full-width crop at whole
    periods of view()'s scale, resized alike, EDGE rows dropped at each cut; the same pixels as view(). None where that
    can't hold (another scale, thumbnail's reduce step, another size or mode)."""
    (w, h), (vh, vw) = merged.size, was.shape[:2]
    scale = Fraction(h, vh)
    p, q = scale.numerator, scale.denominator
    if Fraction(w, vw) != scale or not 1 < scale < 4 or p > 40:
        return None
    mode = _view_mode(merged)
    now = np.asarray(merged if merged.mode == mode else merged.convert(mode))
    if now.shape != base.shape:
        return None
    rows = np.flatnonzero((now != base).reshape(h, -1).any(1))
    if not len(rows):
        return 0, was[:0]
    pad = -(-(EDGE + REACH + 1) // q) * p       # input rows, whole periods
    y0, y1 = max(0, (int(rows[0]) - pad) // p * p), min(h, (int(rows[-1]) + pad) // p * p + p)
    band = np.asarray(Image.fromarray(now[y0:y1]).resize((vw, (y1 - y0) * q // p), Image.LANCZOS).convert("RGBA"))
    cut0, cut1 = EDGE if y0 else 0, EDGE if y1 < h else 0
    return y0 * q // p + cut0, band[cut0:len(band) - cut1]


def _moved(was: np.ndarray, merged: Image.Image, base: Optional[np.ndarray] = None) -> Optional[tuple]:
    """(x, y, the view's pixels there after the edit, before it) over the view pixels merged moves, None for none.
    base: _base of the PNG, to view only the rows merged moves (_band)."""
    got = _band(was, merged, base) if base is not None else None
    top, now = got or (0, np.asarray(view(merged).convert("RGBA")))
    ys, xs = np.nonzero((now != was[top:top + len(now)]).any(-1))
    if not len(xs):
        return None
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    return x0, top + y0, now[y0:y1, x0:x1].copy(), was[top + y0:top + y1, x0:x1].copy()


def _creature_stats(card: dict) -> dict:
    """{"<stat>@<n>": value} of creatures 2..N on a card of several (bar_changes' keys past creature 1's)."""
    return {f"{stat}@{n}": one[stat] for n, one in enumerate(card.get("creatures") or [], 1) if n > 1
            for stat in psd_edit.STAT_FIELDS if isinstance(one.get(stat), int) and one[stat] >= 0}


def bar_changes(psd: PSDImage, raw: bytes, png: Image.Image, parsed: dict, parser, rel: str,
                votes: Optional[dict] = None) -> dict:
    """{stat: {value: _moved()}} over the view pixels a stat edit moves (bar_value at the site's card view, view());
    every stat the card has, every value the bake takes; creatures 2..N's own as "<stat>@<n>" (_creature_stats).
    votes: gets {key: {value: bar_value's seen}} too."""
    card = parsed["card"]
    stats = [stat for stat in psd_edit.STAT_FIELDS if isinstance(card.get(stat), int) and card[stat] >= 0]
    if not stats or png.size != psd.size or psd_edit.story_style_drift(psd, kerning=False):
        return {}                               # the bake refuses every edit of a card whose stories drift
    psd_edit.use_record(psd, rel)
    icc = psd_edit.export_icc(psd, png)
    was, base = np.asarray(view(png).convert("RGBA")), _base(png)
    out, at = {}, {**{stat: card[stat] for stat in stats}, **_creature_stats(card)}
    with psd_edit.render_cache(shared=True), rcache.frozen(psd):     # psd only read: its keys once
        caps = psd_edit.stat_caps(psd)          # once: each value edits its own copy
        for key, now_at in at.items():
            stat, _, n = key.partition("@")
            for value in range(BAR_TOP + 1):
                seen = [] if votes is not None else None
                merged = None if value == now_at else \
                    bar_value(psd, raw, png, parsed, parser, rel, stat, value, icc, int(n or 1), caps, seen)
                if seen:
                    votes.setdefault(key, {})[value] = seen[0]
                moved = _moved(was, merged, base) if merged is not None else None
                if moved:
                    out.setdefault(key, {})[value] = moved
    return out


def star_changes(psd: PSDImage, raw: bytes, png: Image.Image, parsed: dict, parser, rel: str, top: int) -> dict:
    """{count: _moved()} of star_value at every count 1 to top but the card's; {} for a card whose stars no edit
    moves (bar_changes' cards, a tagged PSD: one picture only)."""
    count = parsed["card"].get("stars")
    if not isinstance(count, int) or png.size != psd.size or len(variants_of(psd)) > 1 \
            or psd_edit.story_style_drift(psd, kerning=False):
        return {}
    psd_edit.use_record(psd, rel)
    icc = psd_edit.export_icc(psd, png)
    was, base = np.asarray(view(png).convert("RGBA")), _base(png)
    out = {}
    with psd_edit.render_cache(shared=True), rcache.frozen(psd):     # psd only read: its keys once
        for value in range(1, top + 1):
            merged = None if value == count else star_value(psd, raw, png, parsed, parser, rel, value, icc)
            moved = _moved(was, merged, base) if merged is not None else None
            if moved:
                out[value] = moved
    return out


def _same(old: Optional[dict], sheet: np.ndarray) -> bool:
    """old (a published sheet) a lossless WebP of sheet: its alpha, and its colours where alpha > 0 (libwebp may change
    them under alpha 0)."""
    url = (old or {}).get("img") or ""
    if not url.startswith(WEBP_URL):
        return False
    try:
        data = b64decode(url[len(WEBP_URL):])
        image = Image.open(BytesIO(data))
        if len(data) > BAR_BYTES or image.format != "WEBP" or image.size != (sheet.shape[1], sheet.shape[0]) \
                or image.mode not in ("RGB", "RGBA"):
            return False
        pixels = np.array(image.convert("RGBA"))
    except Exception:                           # noqa: BLE001 - unreadable: encoded again
        return False
    pixels[pixels[..., 3] == 0] = 0
    return np.array_equal(pixels, sheet)


def bar_sheet(changes: dict, at: dict, moved: int = BAR_MOVED, side: int = BAR_SIDE, vw: int = VIEW_W,
              old: Optional[dict] = None, later: bool = False) -> Optional[dict]:
    """
    bar_changes() as the plates file's "bars": {"vw": the view's width, "bars": {stat: {"at": the PSD's value, "to": {value:
    cut index}}}, "cuts": [[x, y, w, h, sheet x, sheet y]] (view px), "img": a data URL}: per value the pixels moved past
    `moved` levels (alpha 255, the rest 0), identical cuts once, stacked in columns at most `side` tall; lossless WebP,
    or past BAR_BYTES AVIF 4:4:4 at BAR_AVIF, tuned for screen content (aom's palette and block copy: the values' repeats);
    None for none. old: the published plate's same sheet, its WebP kept when its pixels are these (no encode). later:
    "img" is a Future of it, encoded in a thread (settled() waits).
    """
    bars, cuts, pieces, seen = {}, [], [], {}
    for stat, values in changes.items():
        for value, (x, y, now, was) in sorted(values.items()):
            mask = np.abs(now.astype(np.int16) - was.astype(np.int16)).max(-1) > moved
            if not mask.any():
                continue
            ys, xs = np.nonzero(mask)
            a0, b0, a1, b1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            piece = now[b0:b1, a0:a1].copy()
            piece[..., 3] = np.where(mask[b0:b1, a0:a1], 255, 0)
            piece[piece[..., 3] == 0] = 0
            key = (x + a0, y + b0, piece.shape[1], piece.shape[0], piece.tobytes())
            if key not in seen:
                seen[key] = len(cuts)
                cuts.append([x + a0, y + b0, piece.shape[1], piece.shape[0]])
                pieces.append((piece, was[b0:b1, a0:a1]))
            bars.setdefault(stat, {"at": at[stat], "to": {}})["to"][str(value)] = seen[key]
    if not bars:
        return None
    spots, x, y, wide, high = [], 0, 0, 0, 0
    for piece, _ in pieces:
        if y and y + piece.shape[0] > side:     # the next column
            x, y, wide = x + wide, 0, 0
        spots.append((x, y))
        wide, y = max(wide, piece.shape[1]), y + piece.shape[0]
        high = max(high, y)
    sheet, under = (np.zeros((high, x + wide, 4), np.uint8) for _ in range(2))
    for cut, (piece, was), (sx, sy) in zip(cuts, pieces, spots):
        sheet[sy:sy + piece.shape[0], sx:sx + piece.shape[1]] = piece
        under[sy:sy + piece.shape[0], sx:sx + piece.shape[1], :3] = was[..., :3]
        cut += [sx, sy]
    if _same(old, sheet):
        return {"vw": vw, "bars": bars, "cuts": cuts, "img": old["img"]}
    return {"vw": vw, "bars": bars, "cuts": cuts,
            "img": _encoders().submit(_encoded, sheet, under) if later else _encoded(sheet, under)}


def _encoded(sheet: np.ndarray, under: np.ndarray) -> str:
    """bar_sheet's data URL: lossless WebP, or past BAR_BYTES AVIF with `under` (the old pixels) off the cuts."""
    buf = BytesIO()
    Image.fromarray(sheet, "RGBA").save(buf, "WEBP", lossless=True, quality=100, method=6)
    kind = "webp"
    if buf.tell() > BAR_BYTES:                  # a big one: lossy, quieter than the site's own card image
        lossy, opts = Image.fromarray(np.where(sheet[..., 3:] > 0, sheet, under), "RGBA"), {"subsampling": "4:4:4", "speed": 6}
        buf = BytesIO()
        try:                                    # aom's screen tools: the values' repeats cost little
            lossy.save(buf, "AVIF", quality=BAR_AVIF, advanced={"tune-content": "screen"}, **opts)
        except ValueError:                      # an encoder without that option: the plain one
            buf = BytesIO()
            lossy.save(buf, "AVIF", quality=BAR_AVIF, **opts)
        kind = "avif"
    return f"data:image/{kind};base64," + b64encode(buf.getvalue()).decode()


@cache
def _encoders() -> ThreadPoolExecutor:
    """bar_sheet(later=True)'s threads, made on first use (in the process that encodes: Card checks' workers)."""
    return ThreadPoolExecutor(BAR_THREADS, thread_name_prefix="bar-sheet")


def settled(plate: Optional[dict], rel: str = "") -> Optional[dict]:
    """plate with its sheets' later images (bar_sheet(later=True)) waited for. A failed encode: check_card's "no plate"
    line and plate as its try then left it (the sheets from that one on dropped; None if no part but that was left)."""
    if not plate:
        return plate
    out = {}
    for key, part in plate.items():
        if isinstance(part, dict) and isinstance(part.get("img"), Future):
            try:
                part = {**part, "img": part["img"].result()}
            except Exception as error:          # noqa: BLE001 - as check_card's try
                print(f"{rel}: no plate: {type(error).__name__}: {error}", flush=True)
                return out if set(out) - {"v", "w", "crops"} else None
        out[key] = part
    return out


def _own(layer) -> bool:
    """A layer-comp bar of the card's own (psd_edit.comp_custom); False when its document can't be read."""
    try:
        return psd_edit.comp_custom(layer)
    except Exception:                           # noqa: BLE001 - not embedded or unreadable: the template's way
        return False


def pictured(psd: PSDImage, png: Image.Image, card: dict, style: str) -> bool:
    """
    Whether with_bars pictures the card: custom or item bars (card_checks.bar_style "custom", "few"), layer-comp bars of
    its own (psd_edit.comp_custom: the Isaac bars), a stat past 10 (the cells draw 0-10), a layer-comp bar at 0 (its lit
    look is hidden under the darks: the cells can only borrow another row's) or printing its value (Baba's digit: the
    cells draw none), else a stat mock_misses finds; never a tagged PSD's (one picture only: the cells') or a card
    without stats.
    """
    if not any(isinstance(card.get(stat), int) for stat in psd_edit.STAT_FIELDS) or png.size != psd.size \
            or len(variants_of(psd)) > 1:
        return False
    over = any(isinstance(card.get(stat), int) and card[stat] > 10 for stat in psd_edit.STAT_FIELDS)   # the cells: 0-10
    if style in ("custom", "few") or over:
        return True
    comps = comp_bars(psd)
    if any(_own(layer) for bars in comps.values() for layer, _, _ in bars):
        return True
    if any(card.get(stat) == 0 and comps.get(stat) for stat in psd_edit.STAT_FIELDS):
        return True
    if any(psd_edit.bar_digits(psd, box, value) for bars in comps.values() for _, value, box in bars if value is not None):
        return True
    return bool(mock_misses(view(png), psd, card, _stats(psd)))


def with_bars(plate: Optional[dict], psd: PSDImage, raw: bytes, png: Image.Image, parsed: dict, parser, rel: str,
              style: str, votes: Optional[dict] = None, old: Optional[dict] = None, later: bool = False) -> Optional[dict]:
    """The plates file with its "bars" (bar_sheet) on a card pictured() picks; plate as it was otherwise. votes:
    bar_changes'. old: the card's published plates file. later: bar_sheet's."""
    card = parsed["card"]
    if not pictured(psd, png, card, style):
        return plate
    sheet = bar_sheet(bar_changes(psd, raw, png, parsed, parser, rel, votes), {**card, **_creature_stats(card)},
                      vw=view(png).width, old=(old or {}).get("bars"), later=later)
    if sheet is None:
        return plate
    return {**(plate or {"v": VERSION, "w": psd.width, "crops": []}), "bars": sheet}


def with_stars(plate: Optional[dict], psd: PSDImage, raw: bytes, png: Image.Image, parsed: dict, parser, rel: str,
               top: int, old: Optional[dict] = None, later: bool = False) -> Optional[dict]:
    """The plates file with its "icons": bar_sheet of star_changes under "stars"; plate as it was without any. old: the
    card's published plates file. later: bar_sheet's."""
    sheet = bar_sheet({"stars": star_changes(psd, raw, png, parsed, parser, rel, top)}, parsed["card"], vw=view(png).width,
                      old=(old or {}).get("icons"), later=later)
    if sheet is None:
        return plate
    return {**(plate or {"v": VERSION, "w": psd.width, "crops": []}), "icons": sheet}


def with_under(plate: Optional[dict], psd: PSDImage, png: Image.Image, parser, name: str) -> Optional[dict]:
    """
    The plates file with its "under": the card drawn without its header type icons (psd_edit.type_row's, those the bake
    hides or moves; their clipped layers and effects go with them) over their ink and a margin, at the view's scale,
    which the mock-up shows where a types edit takes an icon away. {"box": canvas px, "icons": each icon's layer box
    (the details record's header row "box"), "reach": its padded box (its effects'), "pitch": px from one band of "img"
    (AVIF) to the next, "patch": per icon, its band where its ink is in another's reach, or 0}; band 0: the card
    without them, then a mask of each icon's ink (white, UNDER_GROW view px wider), then the patches (the card without
    that icon alone). plate as it was without icons, or when a backdrop off their ink is not the PNG's (crop()'s seam
    rule).

    Args:
        name: The card's name (type_row's title rule)
    """
    if psd.width <= 0 or png.size != psd.size:
        return plate
    scale, step = VIEW_W / psd.width, _step(psd)
    with variant_view(psd, 1 if len(variants_of(psd)) > 1 else None):     # a tagged PSD: picture 1's
        images, _ = psd_edit.type_row(psd, parser, name)
        icons = [layer for _, layer in images if not psd_edit._icon_reason(layer)]
        alone = [[layer] for layer in icons] if len(icons) > 1 else []     # each on its own: where inks meet
        got = _without(psd, [icons, *alone], _rgb(png), psd_edit.export_icc(psd, png), step,
                       tight=PAD + ceil((EDGE + UNDER_GROW) / scale)) if icons else None
    if got is None:
        return plate
    box, drawn, _, inks = got
    size = (max(1, round((box[2] - box[0]) * scale)), max(1, round((box[3] - box[1]) * scale)))
    small = [np.asarray(image.resize(size, Image.LANCZOS)) for image in drawn]
    grow = ImageFilter.MaxFilter(2 * UNDER_GROW + 1)
    masks = [np.asarray(Image.fromarray(ink.astype(np.uint8) * 255).resize(size, Image.BOX)
                        .point(lambda v: 255 if v else 0).filter(grow)) > 127 for ink in (inks[1:] or inks)]
    reach = [_padded(layer) for layer in icons]
    spans = []                                  # each icon's reach in the bands
    for r in reach:
        span = np.zeros(size[::-1], bool)
        x0, y0, x1, y1 = (min(max(round((v - o) * scale), 0), top) for v, o, top in zip(r, box[:2] * 2, size * 2))
        span[y0:y1, x0:x1] = True
        spans.append(span)
    bands, patch, soft = [small[0], *(np.repeat(m[..., None] * np.uint8(255), 3, -1) for m in masks)], [], None
    for n, mask in enumerate(masks):
        meets = mask & np.logical_or.reduce([m for k, m in enumerate(spans) if k != n] or [np.zeros_like(mask)])
        if not meets.any():
            patch.append(0)
            continue
        if soft is None:                        # off the patch: smooth, so it costs little
            soft = np.asarray(Image.fromarray(small[0]).filter(ImageFilter.GaussianBlur(8)))
        near = np.asarray(Image.fromarray(meets.astype(np.uint8) * 255).filter(grow)) > 127
        patch.append(len(bands))
        bands.append(np.where(near[..., None], small[n + 1], soft))
    pitch = -(-(size[1] + GAP) // 8) * 8        # each band on an 8 px block
    sheet = np.concatenate([np.pad(band, ((0, pitch - size[1] if n < len(bands) - 1 else 0), (0, 0), (0, 0)),
                                   mode="edge") for n, band in enumerate(bands)])
    buf = BytesIO()
    Image.fromarray(sheet).save(buf, "AVIF", quality=QUALITY)
    return {**(plate or {"v": VERSION, "w": psd.width, "crops": []}),
            "under": {"box": list(box), "icons": [list(layer.bbox) for layer in icons],
                      "reach": [list(r) for r in reach], "pitch": pitch, "patch": patch,
                      "img": "data:image/avif;base64," + b64encode(buf.getvalue()).decode()}}


def _made(psd: PSDImage, raw: bytes, png: Image.Image, parsed: dict, parser, rel: str, types_dir: str, names: list,
          icc: bool) -> list:
    """[(type, the made icon's ink box, the PNG after)] of a lone types edit adding names, as the bake makes it
    (star_value's steps: classify, plan_types, set_icons, gate_dirty, baseline, the merge); [] when it refuses, steps an
    icon down or shows a hidden icon of the card's for one."""
    value = list(parsed["card"].get("types") or []) + names
    try:
        if "types" not in psd_edit.classify(parsed, {"types": value}, psd, parser).jobs:
            return []
        new = psd_edit.edit_copy(psd, raw)
        plan = psd_edit.plan_types(new, parsed, value, parser, None, types_dir)
        edit = psd_edit.set_icons(new, plan, types_dir, rel)
        made = [done[1] for done in plan.done if done[0] == "made"]
        if len(made) != len(names) or edit.notes:
            return []
        with psd_edit.render_cache(shared=True):
            first = psd_edit.gate_dirty(psd, new, [edit], icon=True)
            base, measured = psd_edit.baseline(psd, png, edit.region, icc, first)
            render = psd_edit.render_region(new, edit.region, icc, measured["text"], psd_edit.edited_ids([edit]))
        merged, replaced = psd_edit.merge_png(png, base, render, edit.region)
    except Exception:                           # noqa: BLE001 - refused or the bake's error: none
        return []
    if not replaced.any():
        return []
    return [(psd_edit.canonical_type(parser, layer.name), psd_edit._core(layer) or tuple(layer.bbox), merged)
            for layer in made]


def type_probe(psd: PSDImage, raw: bytes, png: Image.Image, parsed: dict, parser, rel: str, types_dir: str,
               old: Optional[dict] = None) -> Optional[dict]:
    """
    The looks the bake gives type icons it makes on this card, for the mock-up's new icons: {"vw": the view's width,
    "probes": per type of TYPE_PROBE (a warm and a cool hue: a look the mock-up fits must draw both, never one's alone)
    {"name", "core": the made icon's ink box (PSD px), "at": [x, y, w, h] (view px), "img": a data URL of the view
    there}}, from one types edit adding both, else one edit each (_made); a type the bake refuses, steps down or shows
    a hidden icon of the card's for is left out; None for none, a card without types or a tagged PSD. old: the
    published typeProbe, an img kept where its pixels are these (no encode).
    """
    have = list(parsed["card"].get("types") or [])
    if not have or png.size != psd.size or len(variants_of(psd)) > 1 or psd_edit.story_style_drift(psd, kerning=False):
        return None
    printed = {psd_edit.canonical_type(parser, one) for one in have}
    names = [name for name in (next((one for one in pair if one not in printed), "") for pair in TYPE_PROBE) if name]
    psd_edit.use_record(psd, rel)
    icc = psd_edit.export_icc(psd, png)
    made = _made(psd, raw, png, parsed, parser, rel, types_dir, names, icc) if len(names) > 1 else []
    if not made:                                # the pair refused, moved or stepped down: one edit each
        made = [one for name in names for one in _made(psd, raw, png, parsed, parser, rel, types_dir, [name], icc)]
    out, views, kept = [], {}, (old or {}).get("probes") or []
    for name, core, merged in made:
        if id(merged) not in views:
            views[id(merged)] = view(merged)
        shown = views[id(merged)]
        k = shown.width / psd.width
        x0, y0 = max(0, floor(core[0] * k) - 1), max(0, floor(core[1] * k) - 1)
        x1, y1 = min(shown.width, ceil(core[2] * k) + 1), min(shown.height, ceil(core[3] * k) + 1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        probe = {"name": name, "core": [int(v) for v in core], "at": [x0, y0, x1 - x0, y1 - y0]}
        cut = np.asarray(shown.crop((x0, y0, x1, y1)).convert("RGB").convert("RGBA"))
        same = next((one for one in kept if {**one, "img": ""} == {**probe, "img": ""} and _same(one, cut)), None)
        if same is None:
            buf = BytesIO()
            Image.fromarray(cut[..., :3]).save(buf, "WEBP", lossless=True, quality=100, method=6)
        out.append({**probe, "img": same["img"] if same else WEBP_URL + b64encode(buf.getvalue()).decode()})
    return {"vw": shown.width, "probes": out} if out else None


def with_probe(plate: Optional[dict], psd: PSDImage, raw: bytes, png: Image.Image, parsed: dict, parser, rel: str,
               types_dir: str, old: Optional[dict] = None) -> Optional[dict]:
    """The plates file with its "typeProbe" (type_probe); plate as it was without one. old: the card's published
    plates file."""
    probe = type_probe(psd, raw, png, parsed, parser, rel, types_dir, (old or {}).get("typeProbe"))
    if probe is None:
        return plate
    return {**(plate or {"v": VERSION, "w": psd.width, "crops": []}), "typeProbe": probe}


# ========================
# The plates branch
# ========================
class Unpublished(RuntimeError):
    """A git failure: publish tries again from the branch head, TRIES times."""


def _git(where: str, *args: str, stdin: Optional[bytes] = None) -> str:
    try:
        return run_process(["git", "-C", where, *args], input=stdin, capture_output=True, check=True, timeout=GIT_TIMEOUT,
                           env={**batch_env(where), **IDENTITY}).stdout.decode("utf-8")
    except CalledProcessError as error:
        raise Unpublished(f"git {_verb(args)} failed: {(error.stderr or b'').decode('utf-8', 'replace').strip()}") from error
    except (TimeoutExpired, OSError) as error:
        raise Unpublished(f"git {_verb(args)} failed: {error}") from error


def _verb(args: tuple) -> str:
    """The git command past any -c option."""
    return args[2] if args[0] == "-c" else args[0]


def _files(where: str, head: str) -> dict:
    """{name: its ls-tree entry, "<mode> <type> <sha>"} at the branch head ({} with none)."""
    if not head:
        return {}
    return dict(line.split("\t", 1)[::-1] for line in _git(where, "ls-tree", "-z", head).split("\0") if "\t" in line)


def _plate(name: str) -> bool:
    """A plates file's name (<PSD blob>.json); anything else on the branch (site.yml's v/ copies) is kept as it is."""
    return fullmatch(r"[0-9a-f]{40}\.json", name) is not None


def psd_blobs(where: str, head: str) -> set:
    """The PSD blobs of head's tree."""
    return {line.split()[2] for line in _git(where, "ls-tree", "-r", "-z", head).split("\0")
            if "\t" in line and line.split("\t", 1)[1].lower().endswith(".psd")}


def on_branch(where: str) -> dict:
    """{PSD blob: its plates file's blob} on the branch as last fetched to LOCAL_REF (card-checks.yml's To check); {}
    without it."""
    try:
        return {name[:-5]: entry.split()[2] for name, entry in _files(where, LOCAL_REF).items() if _plate(name)}
    except Unpublished:
        return {}


def collected(paths: list) -> dict:
    """{PSD blob: plates file} of card_checks run results."""
    out = {}
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            out.update({sha: record["plate"] for sha, record in loads(fh.read()).items()
                        if isinstance(record, dict) and record.get("plate")})
    return out


def publish(found: dict, where: str, head: str = "HEAD") -> dict:
    """
    One commit on the plates branch: found ({PSD blob: plates file}) added, those of PSDs not in head's tree dropped,
    other entries (site.yml's v/) kept; none when nothing changed. Built on the branch head, TRIES times from a moved one.

    Args:
        where: A checkout of the card repo (its git and credentials; blobless is enough: pushes without delta search)
        head: The commit whose PSDs keep their plates

    Returns:
        {"commit" ("" for none), "added", "dropped", "kept"}; raises Unpublished when no push landed
    """
    keep = psd_blobs(where, head)
    new = {f"{sha}.json": "100644 blob " + _git(where, "hash-object", "-w", "--stdin", stdin=dumps(plate).encode()).strip()
           for sha, plate in found.items() if sha in keep}
    why = ""
    for attempt in range(TRIES):
        if attempt:
            sleep(uniform(1, 2) * attempt)
        try:
            parent = ""
            if _git(where, "ls-remote", "--heads", "origin", f"refs/heads/{BRANCH}").split():
                _git(where, "fetch", "-q", "--no-tags", "--no-write-fetch-head", "--depth", "1", "--filter=blob:none", "origin",
                     f"+refs/heads/{BRANCH}:{LOCAL_REF}")
                parent = _git(where, "rev-parse", "--verify", f"{LOCAL_REF}^{{commit}}").strip()
            old = _files(where, parent)
            files = {**{n: e for n, e in old.items() if not _plate(n) or n[:-5] in keep}, **new}
            added, dropped = sum(1 for n, e in new.items() if old.get(n) != e), len(set(old) - set(files))
            kept = sum(map(_plate, files))
            if files == old:                    # no branch yet and no plates: none made
                return {"commit": "", "added": 0, "dropped": 0, "kept": kept}
            tree = _git(where, "mktree", "--missing", stdin="".join(f"{e}\t{n}\n" for n, e in sorted(files.items())).encode()).strip()
            commit = _git(where, "commit-tree", tree, *(["-p", parent] if parent else []), "-m",
                          f"Card plates: {added} added, {dropped} dropped").strip()
            _git(where, "-c", "pack.window=0", "push", "-q", "--no-thin", "origin",   # no delta search: it would want
                 f"{commit}:refs/heads/{BRANCH}")                                # replaced plates' blobs (HTTPS is thin)
        except Unpublished as error:            # most often a push refused: the branch moved since the fetch
            why = str(error)
            print(f"card_plates: attempt {attempt + 1} of {TRIES}: {why}", file=stderr)
            continue
        return {"commit": commit, "added": added, "dropped": dropped, "kept": kept}
    raise Unpublished(f"not published after {TRIES} tries: {why}")


def main(args: Optional[list] = None) -> int:
    """The command line (module docstring)."""
    parser = ArgumentParser(prog="python -m commands.card_plates", description="The mock-up's plates")
    parser.add_argument("action", choices=("publish",))
    parser.add_argument("results", nargs="*", help="card_checks run's files")
    parser.add_argument("--git", required=True, help="a checkout of the card repo (blobless is enough)")
    parser.add_argument("--head", default="HEAD", help="the commit whose PSDs keep their plates")
    options = parser.parse_args(list(argv[1:] if args is None else args))
    done = publish(collected(options.results), options.git, options.head)
    print(f"Plates: {done['added']} added, {done['dropped']} dropped, {done['kept']} on the branch"
          + (f" ({done['commit'][:8]})." if done["commit"] else ": no change."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'BAR_MOVED',
    'BAR_TOP',
    'BRANCH',
    'PROBES',
    'PROBE_TYPES',
    'TYPE_PROBE',
    'Unpublished',
    'VERSION',
    'VIEW_W',
    'adjusted',
    'bar_changes',
    'bar_sheet',
    'bar_value',
    'collected',
    'crop',
    'layer_ids',
    'layout',
    'letter_look',
    'looks',
    'main',
    'mock_misses',
    'on_branch',
    'pictured',
    'plates',
    'psd_blobs',
    'publish',
    'settled',
    'star_changes',
    'star_value',
    'textured',
    'tint_looks',
    'type_probe',
    'view',
    'with_bars',
    'with_probe',
    'with_stars',
    'with_under',
]
