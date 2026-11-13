"""
Plates for the website's mock-up: the card drawn without one text layer, cut to that layer's box (its effects' reach
and PAD past it), per title and ability layer of its parts (psd_parts). The mock-up pastes a plate over the old text
instead of guessing the backdrop from the PNG. One file per PSD blob, crops at the site's card view scale (VIEW_W
wide) in one AVIF; a crop whose backdrop differs from the PNG off the layer's ink is left out (the mock-up's old way).
Card checks makes them (check_card's "plate") and publishes them on the `plates` branch as <PSD blob>.json, those of
PSDs gone from main dropped; the website fetches the open card's in edit mode. With the fonts, a layer's crop also
carries the fork's layout of it (its lines and letters, layout()) when that is the stored pixels'. A card whose stat rows
the mock-up's cells cannot redraw also gets "bars" (with_bars): the bake's own picture of each stat at every value it
takes, at that view, which the mock-up pastes instead; every starred card gets "icons" (with_stars): the bake's own
stars at every count, pasted in place of the mock-up's erased and redrawn ones.

    python -m commands.card_plates publish --git <checkout> [--head <sha>] <run results.json> ...
"""
from argparse import ArgumentParser
from base64 import b64encode
from io import BytesIO
from json import dumps, loads
from math import ceil, floor, gcd
from random import uniform
from re import IGNORECASE, fullmatch
from subprocess import CalledProcessError, TimeoutExpired, run as run_process
from sys import argv, stderr
from time import sleep
from typing import Optional

import numpy as np
from PIL import Image
from psd_tools import PSDImage

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
LAYOUT_OFF = 0.05                           # card_standards._ink_off past this: the layout isn't the card's, none kept
TILT = 0.002                                # a layer turned past this (xy, yx over xx): no layout
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
    hide = [layer, *(other for other in extra if _meets(layer, other))]
    boxes = [_padded(x) for x in hide]
    box = _snap((min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)), step, psd.size)
    full = psd_edit.render_region(psd, box, icc)
    eyes = [(x, x.visible) for x in hide]
    for x in hide:
        x.visible = False
    try:
        plate = psd_edit.render_region(psd, box, icc)
    finally:
        for x, eye in eyes:
            x.visible = eye
    a, b = _rgb(full), _rgb(plate)
    ink = np.abs(a - b).max(-1) > INK
    if not ink.any():
        return None
    off = ~ink
    seam = int(((np.abs(b - png[box[1]:box[3], box[0]:box[2]]).max(-1) > SEAM) & off).sum())
    if seam > SEAM_SHARE * off.sum():
        return None
    ys, xs = np.nonzero(ink)
    return box, like_mode(plate, "RGB"), seam, (box[0] + int(xs.min()), box[1] + int(ys.min()), box[0] + int(xs.max()) + 1, box[1] + int(ys.max()) + 1)


def layout(layer, fonts_dir: str) -> Optional[dict]:
    """
    The fork's layout of a type layer in canvas px, when it draws the stored pixels (card_standards._ink_off within
    LAYOUT_OFF) and is not turned (TILT): {"text": the engine's text (\\r paragraphs, \\x03 breaks), "lines": [[start,
    end, baseline, left, right, top, bottom, justification]], "glyphs": [[char index, pen x, ink left, ink right]],
    "frame": [left, right] or None (point text)}, else None. start/end index text (end: the break excluded); baseline:
    its letters' (BaselineShift in); left/right: the pen's ends; top/bottom: the ink's, underlines too; a blank glyph
    has no ink ends.
    """
    from statistics import median
    from psd_tools.composite.text.geometry import text_geometry
    from psd_tools.composite.text.layout import frame, layout as lay_out
    from psd_tools.composite.text.model import TextModel
    from commands.card_standards import _ink_off
    faces = psd_edit.font_map(fonts_dir, layer)
    model = TextModel.from_layer(layer)
    xx, xy, yx, yy, tx, ty = model.transform
    if xx <= 0 or yy <= 0 or max(abs(xy), abs(yx)) > TILT * xx or _ink_off(layer, faces) > LAYOUT_OFF:
        return None
    geo, lay = text_geometry(layer, faces), lay_out(model, faces)
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
    box = frame(model)
    return {"text": geo["text"], "lines": lines,
            "glyphs": [[g["i"], r(g["cell"][0]), *(r(g["ink"][k]) for k in (0, 2) if g["ink"])] for g in geo["glyphs"]],
            "frame": [r(xx * box[0] + tx), r(xx * (box[0] + box[2]) + tx)] if box else None}


def plates(psd: PSDImage, png: Image.Image, parts: dict, fonts_dir: Optional[str] = None) -> Optional[dict]:
    """
    The plates file of a card: {"v", "w", "crops": [{"id", "box", "ink", "at"}], "img"}, or None when no layer gave one.
    box: the crop, ink: where the layer drew in it (canvas px); at: [x, y, w, h] of the crop in img (a data URL). A
    layer with FX layers on it has a second crop without them too, its "fx" their status words (the mock-up's when
    one leaves the text: the bake moves FX layers with their words). With fonts_dir, a layer's first crop also has
    its layout() keys ("text", "lines", "glyphs", "frame").

    Args:
        psd: The card
        png: Its PNG (picture 1)
        parts: Its details record's parts (psd_parts.card_parts)
        fonts_dir: MDW/Fonts (the layouts'); None: none
    """
    if psd.width <= 0 or png.size != psd.size:
        return None
    scale = VIEW_W / psd.width
    step = psd.width // gcd(psd.width, VIEW_W)
    step = step if step <= 20 else 1
    icc = psd_edit.export_icc(psd, png)
    by_id = {layer.layer_id: layer for layer in psd.descendants()}
    truth = _rgb(png)
    follow = [f for f in (parts or {}).get("follow") or [] if isinstance(f, dict)]
    found = []
    with variant_view(psd, 1 if len(variants_of(psd)) > 1 else None):
        for lid in dict.fromkeys(layer_ids(parts)):
            layer = by_id.get(lid)
            if layer is None:
                continue
            copies = [by_id[f["id"]] for f in follow if lid in (f.get("of") or []) and f.get("id") in by_id]
            got = crop(psd, layer, truth, icc, step, copies)
            if got:
                try:
                    lay = layout(layer, fonts_dir) if fonts_dir else None
                except Exception as error:      # noqa: BLE001 - no layout: the mock-up finds the lines
                    print(f"{layer.name}: no layout: {type(error).__name__}: {error}", file=stderr)
                    lay = None
                found.append(({"id": lid, **(lay or {})}, *got))
            fx = [x for x in psd_edit.fx_layers(psd, layer) if _meets(layer, x)]
            got = crop(psd, layer, truth, icc, step, [*copies, *fx]) if fx else None
            if got:
                words = [w.strip() for x in fx for w in x.name.split(EFFECT_LAYER_TAG, 1)[1].split("#")[0].split("|") if w.strip()]
                found.append(({"id": lid, "fx": list(dict.fromkeys(words))}, *got))
    if not found:
        return None
    small = [(key, box, ink, image.resize((max(1, round((box[2] - box[0]) * scale)), max(1, round((box[3] - box[1]) * scale))),
                                          Image.LANCZOS)) for key, box, image, _, ink in found]
    width, last, crops, parts, y = max(im.width for *_, im in small), len(small) - 1, [], [], 0
    for n, (key, box, ink, im) in enumerate(small):     # gaps and the right side: edge pixels, no seam bleeds in
        top, bottom = (GAP // 2 if n else 0), (GAP - GAP // 2 if n < last else 0)
        parts.append(np.pad(np.asarray(im), ((top, bottom), (0, width - im.width), (0, 0)), mode="edge"))
        crops.append({**key, "box": list(box), "ink": list(ink), "at": [0, y + top, im.width, im.height]})
        y += top + im.height + bottom
    buf = BytesIO()
    Image.fromarray(np.concatenate(parts)).save(buf, "AVIF", quality=QUALITY)
    return {"v": VERSION, "w": psd.width, "crops": crops, "img": "data:image/avif;base64," + b64encode(buf.getvalue()).decode()}


# ========================
# Stat bars: the bake's picture of every value
# ========================
BAR_TOP = 20                                # values tried, 0 to this; the bake refuses what its bars cannot show
BAR_MOVED = 2                               # view levels: a bar pixel moved less shows the site's own image
BAR_BYTES = 192_000                         # a lossless bar sheet past this goes lossy (BAR_AVIF)
BAR_AVIF = 55                               # AVIF quality (screen tuning): under the site's 720 px image's noise (q50)
BAR_SIDE = 16383                            # WebP's largest side: the sheet's cuts stack in columns of this height


def view(image: Image.Image) -> Image.Image:
    """image as the site's VIEW_W card image before its encode (build_site.make_images: its mode, thumbnail, LANCZOS)."""
    image = image.convert("RGBA" if image.mode in ("RGBA", "LA", "P") else "RGB")
    image.thumbnail((VIEW_W, VIEW_W * 3), Image.LANCZOS)
    return image


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
        if not (edit.layers or edit.removed):
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


def _moved(was: np.ndarray, merged: Image.Image) -> Optional[tuple]:
    """(x, y, the view's pixels there after the edit, before it) over the view pixels merged moves, None for none."""
    now = np.asarray(view(merged).convert("RGBA"))
    ys, xs = np.nonzero((now != was).any(-1))
    if not len(xs):
        return None
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    return x0, y0, now[y0:y1, x0:x1].copy(), was[y0:y1, x0:x1].copy()


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
    was = np.asarray(view(png).convert("RGBA"))
    out, at = {}, {**{stat: card[stat] for stat in stats}, **_creature_stats(card)}
    with psd_edit.render_cache(shared=True):
        caps = psd_edit.stat_caps(psd)          # once: each value edits its own copy
        for key, now_at in at.items():
            stat, _, n = key.partition("@")
            for value in range(BAR_TOP + 1):
                seen = [] if votes is not None else None
                merged = None if value == now_at else \
                    bar_value(psd, raw, png, parsed, parser, rel, stat, value, icc, int(n or 1), caps, seen)
                if seen:
                    votes.setdefault(key, {})[value] = seen[0]
                moved = _moved(was, merged) if merged is not None else None
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
    was = np.asarray(view(png).convert("RGBA"))
    out = {}
    with psd_edit.render_cache(shared=True):
        for value in range(1, top + 1):
            merged = None if value == count else star_value(psd, raw, png, parsed, parser, rel, value, icc)
            moved = _moved(was, merged) if merged is not None else None
            if moved:
                out[value] = moved
    return out


def bar_sheet(changes: dict, at: dict, moved: int = BAR_MOVED, side: int = BAR_SIDE, vw: int = VIEW_W) -> Optional[dict]:
    """
    bar_changes() as the plates file's "bars": {"vw": the view's width, "bars": {stat: {"at": the PSD's value, "to": {value:
    cut index}}}, "cuts": [[x, y, w, h, sheet x, sheet y]] (view px), "img": a data URL}: per value the pixels moved past
    `moved` levels (alpha 255, the rest 0), identical cuts once, stacked in columns at most `side` tall; lossless WebP,
    or past BAR_BYTES AVIF 4:4:4 at BAR_AVIF, tuned for screen content (aom's palette and block copy: the values' repeats);
    None for none.
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
    return {"vw": vw, "bars": bars, "cuts": cuts, "img": f"data:image/{kind};base64," + b64encode(buf.getvalue()).decode()}


def _own(layer) -> bool:
    """A layer-comp bar of the card's own (psd_edit.comp_custom); False when its document can't be read."""
    try:
        return psd_edit.comp_custom(layer)
    except Exception:                           # noqa: BLE001 - not embedded or unreadable: the template's way
        return False


def pictured(psd: PSDImage, png: Image.Image, card: dict, style: str) -> bool:
    """
    Whether with_bars pictures the card: custom or item bars (card_checks.bar_style "custom", "few"), layer-comp bars of
    its own (psd_edit.comp_custom: the Isaac bars), a stat past 10 (the cells draw 0-10), else a stat mock_misses finds;
    never a tagged PSD's (one picture only: the cells') or a card without stats.
    """
    if not any(isinstance(card.get(stat), int) for stat in psd_edit.STAT_FIELDS) or png.size != psd.size \
            or len(variants_of(psd)) > 1:
        return False
    over = any(isinstance(card.get(stat), int) and card[stat] > 10 for stat in psd_edit.STAT_FIELDS)   # the cells: 0-10
    if style in ("custom", "few") or over:
        return True
    if any(_own(layer) for bars in comp_bars(psd).values() for layer, _, _ in bars):
        return True
    return bool(mock_misses(view(png), psd, card, _stats(psd)))


def with_bars(plate: Optional[dict], psd: PSDImage, raw: bytes, png: Image.Image, parsed: dict, parser, rel: str,
              style: str, votes: Optional[dict] = None) -> Optional[dict]:
    """The plates file with its "bars" (bar_sheet) on a card pictured() picks; plate as it was otherwise. votes:
    bar_changes'."""
    card = parsed["card"]
    if not pictured(psd, png, card, style):
        return plate
    sheet = bar_sheet(bar_changes(psd, raw, png, parsed, parser, rel, votes), {**card, **_creature_stats(card)},
                      vw=view(png).width)
    if sheet is None:
        return plate
    return {**(plate or {"v": VERSION, "w": psd.width, "crops": []}), "bars": sheet}


def with_stars(plate: Optional[dict], psd: PSDImage, raw: bytes, png: Image.Image, parsed: dict, parser, rel: str,
               top: int) -> Optional[dict]:
    """The plates file with its "icons": bar_sheet of star_changes under "stars"; plate as it was without any."""
    sheet = bar_sheet({"stars": star_changes(psd, raw, png, parsed, parser, rel, top)}, parsed["card"], vw=view(png).width)
    if sheet is None:
        return plate
    return {**(plate or {"v": VERSION, "w": psd.width, "crops": []}), "icons": sheet}


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
    """{name: blob sha} at the branch head ({} with none)."""
    if not head:
        return {}
    return {line.split("\t", 1)[1]: line.split()[2] for line in _git(where, "ls-tree", "-z", head).split("\0") if "\t" in line}


def psd_blobs(where: str, head: str) -> set:
    """The PSD blobs of head's tree."""
    return {line.split()[2] for line in _git(where, "ls-tree", "-r", "-z", head).split("\0")
            if "\t" in line and line.split("\t", 1)[1].lower().endswith(".psd")}


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
    One commit on the plates branch: found ({PSD blob: plates file}) added, those of PSDs not in head's tree dropped;
    none when nothing changed. Built on the branch head, TRIES times from a moved one.

    Args:
        where: A checkout of the card repo (its git and credentials; blobless is enough: pushes without delta search)
        head: The commit whose PSDs keep their plates

    Returns:
        {"commit" ("" for none), "added", "dropped", "kept"}; raises Unpublished when no push landed
    """
    keep = psd_blobs(where, head)
    new = {f"{sha}.json": _git(where, "hash-object", "-w", "--stdin", stdin=dumps(plate).encode()).strip()
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
            files = {**{name: sha for name, sha in old.items() if name[:-5] in keep}, **new}
            added, dropped = sum(1 for n, s in new.items() if old.get(n) != s), len(set(old) - set(files))
            if files == old:                    # no branch yet and no plates: none made
                return {"commit": "", "added": 0, "dropped": 0, "kept": len(files)}
            tree = _git(where, "mktree", "--missing", stdin="".join(f"100644 blob {s}\t{n}\n" for n, s in sorted(files.items())).encode()).strip()
            commit = _git(where, "commit-tree", tree, *(["-p", parent] if parent else []), "-m",
                          f"Card plates: {added} added, {dropped} dropped").strip()
            _git(where, "-c", "pack.window=0", "push", "-q", "--no-thin", "origin",   # no delta search: it would want
                 f"{commit}:refs/heads/{BRANCH}")                                # replaced plates' blobs (HTTPS is thin)
        except Unpublished as error:            # most often a push refused: the branch moved since the fetch
            why = str(error)
            print(f"card_plates: attempt {attempt + 1} of {TRIES}: {why}", file=stderr)
            continue
        return {"commit": commit, "added": added, "dropped": dropped, "kept": len(files)}
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
    'Unpublished',
    'VERSION',
    'VIEW_W',
    'bar_changes',
    'bar_sheet',
    'bar_value',
    'collected',
    'crop',
    'layer_ids',
    'layout',
    'looks',
    'main',
    'mock_misses',
    'pictured',
    'plates',
    'psd_blobs',
    'publish',
    'star_changes',
    'star_value',
    'view',
    'with_bars',
    'with_stars',
]
