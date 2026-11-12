"""
Status cards in Krabby Patty (owner 2026-11-09): every visible text layer's runs set in Krabby Patty but the icon
font's; the title in its chart status's colour and, for a styled status, the chart's effects (a type the chart has
replaces the title's own; the title's Stroke kept, and a glow or shadow cast from the glyphs grown by it so it shows
past it as on the chart); badges and icons fitted to the new text (psd_edit.badge_fits); a layer that no longer fits
its box, the card (EDGE) or, a title, the header icons set smaller (one scale for the layer, as little as needed); the
rasters redrawn by the fork; FX layers on the words cut again. The template takes the font alone (no status).

    restyle(psd, rel, chart, fonts_dir) -> [what it did]      the sweep: edits the open PSD; save with psd_edit.save_psd
    plan(psd, rel, chart) -> [layer]                           what restyle would change (no drawing)
    fix(path, out_path, fonts_dir, chart, rel) -> [layer name] the guards' kind (card_standards "status font"), gated
"""
import re
from copy import deepcopy
from math import floor
from os.path import basename
from typing import Optional

import numpy as np
from psd_tools import PSDImage
from psd_tools.constants import Tag

import commands.psd_edit as psd_edit
from commands.psd_edit import Refused
from commands.psd_parser import EFFECT_LAYER_TAG, ICON_FONT_PREFIX, PSDParser

KP = "KrabbyPatty"                  # MDW/Fonts/Krabby Patty.ttf (PostScript name)
TEMPLATE = "MDW/Templates/Status_Effect_Template.psd"
SCOPE = re.compile(r"Auxiliary/Status Effects/(Buffs|Debuffs/\d+ Stars)/[^/]+\.psd")
SKIP_FONTS = ("adobeinvisfont",)    # draws nothing
FIT_STEP = 0.005                    # a shrink's grain (share of the size)
FIT_MIN = 0.6                       # no smaller than this: refused
EDGE = 5                            # px ink and stroke keep from the card's edge (its old margin when less): the least
                                    # any status card's title kept (Imprisoned, Sickness)
HEADER = 0.25                       # share of the card's height: icons above it are the header's
LFX2, LRFX = Tag.OBJECT_BASED_EFFECTS_LAYER_INFO, Tag.EFFECTS_LAYER
FX_TYPE = {b"dropShadowMulti": b"DrSh", b"innerShadowMulti": b"IrSh", b"solidFillMulti": b"SoFi",
           b"gradientFillMulti": b"GrFl", b"frameFXMulti": b"FrFX"}
CAST = {b"OrGl": b"oglw", b"DrSh": b"dsdw"}    # effects cast from the glyphs' edge (lfx2 type: lrFX key)
STROKE_REACH = {b"OutF": 1.0, b"CtrF": 0.5, b"InsF": 0.0}


def in_scope(rel: str) -> bool:
    return rel == TEMPLATE or bool(SCOPE.fullmatch(rel))


# ---------- effects ----------
def _items(value) -> list:
    if value is None:
        return []
    return [i for i in (value if type(value).__name__ == "ListElement" else [value]) if hasattr(i, "items")]


def _on(item) -> bool:
    return bool(item.get(b"enab")) and bool(item.get(b"present", True))


def _types(data) -> dict:
    """{effect type: [keys]} of a lfx2 descriptor's effect entries (single and Multi keys)."""
    out = {}
    for key, value in data.items():
        if hasattr(value, "items") or type(value).__name__ == "ListElement":
            out.setdefault(FX_TYPE.get(key, key), []).append(key)
    return out


def _num(value) -> float:
    return float(getattr(value, "value", value))


def _reach(data) -> float:
    """Px the enabled strokes of a lfx2 descriptor draw past the glyphs (the widest)."""
    out = 0.0
    for key in (b"FrFX", b"frameFXMulti"):
        for item in _items(data.get(key)) if key in data else []:
            if _on(item):
                style = getattr(item.get(b"Styl"), "enum", b"OutF")
                out = max(out, _num(item.get(b"Sz  ", 0.0)) * STROKE_REACH.get(style, 1.0))
    return out


def grown(size: float, spread: float, by: int) -> tuple:
    """(size, spread %) of a glow or shadow cast from a silhouette grown by `by` px: the spread's whole pixels (Photoshop
    dilates by floor(size * spread / 100 + 0.5)) grow by it, the blur (the rest of the size) stays."""
    pixels = floor(size * spread / 100 + 0.5)
    new, want = size + by, pixels + by
    for p in sorted(range(101), key=lambda p: abs(p - 100 * want / new)):
        if floor(new * p / 100 + 0.5) == want:
            return new, float(p)
    return new, 100.0 * want / new


def _grow(item, by: int) -> None:
    size, spread = _num(item[b"blur"]), _num(item.get(b"Ckmt", 0.0))
    new, p = grown(size, spread, by)
    item[b"blur"].value = new
    if b"Ckmt" in item:
        item[b"Ckmt"].value = p


def _strokes(data) -> bool:
    """An enabled Stroke in a lfx2 descriptor."""
    return any(_on(i) for key in (b"FrFX", b"frameFXMulti") if key in data for i in _items(data[key]))


def _effects(layer, status):
    """The title's effect blocks as the chart makes them: (lfx2, lrFX or None, [types taken], whole), or None when the
    chart has no lfx2 or enables nothing to take. A title with no style drawn (no lfx2, its master switch off) takes the
    chart's blocks whole, as its status words do (whole). Else each type the chart's layer enables replaces the title's
    own of that type (in the title's key order; one it lacks added before numModifyingFX); the title's own Stroke
    stays; a glow or shadow cast from the glyphs grows by the title's stroke past the chart's (grown), so it shows past
    it as on the chart. lrFX: the chart's enabled entries (sized as lfx2's)."""
    mine, theirs = layer.tagged_blocks.get(LFX2), status.blocks.get(LFX2)
    if theirs is None:
        return None
    b = theirs.data
    take = {kind: keys for kind, keys in _types(b).items() if any(_on(i) for k in keys for i in _items(b[k]))}
    if not take:
        return None
    if mine is None or not mine.data.get(b"masterFXSwitch", True):
        legacy = status.blocks.get(LRFX)
        legacy = deepcopy(legacy) if legacy is not None else None
        return deepcopy(theirs), legacy, sorted(k.decode() for k in take), True
    if _strokes(mine.data):
        take.pop(b"FrFX", None)
    if not take:
        return None
    block = deepcopy(mine)
    a = block.data
    items, done = [], set()
    for key, value in a.items():
        kind = FX_TYPE.get(key, key)
        if kind in take:
            if kind not in done:
                items += [(k, deepcopy(b[k])) for k in take[kind]]
                done.add(kind)
            continue
        items.append((key, value))
    tail = [(k, deepcopy(b[k])) for kind, keys in take.items() if kind not in done for k in keys]
    at = next((i for i, (k, _) in enumerate(items) if k == b"numModifyingFX"), len(items))
    items[at:at] = tail
    a.clear()
    for key, value in items:
        a[key] = value
    by = round(_reach(a) - _reach(b))
    sizes = {}
    if by > 0:
        for kind in CAST:
            for key in take.get(kind, ()):
                for item in _items(a[key]):
                    if _on(item) and b"blur" in item:
                        _grow(item, by)
                        sizes[kind] = _num(item[b"blur"])
    if b"numModifyingFX" in a:              # Photoshop's count of enabled effects
        n = sum(1 for keys in _types(a).values() for k in keys for i in _items(a[k]) if _on(i))
        a[b"numModifyingFX"] = type(a[b"numModifyingFX"])(n)
    legacy = layer.tagged_blocks.get(LRFX)
    new = status.blocks.get(LRFX)
    if legacy is not None and new is not None:
        legacy = deepcopy(legacy)
        for key, item in new.data.items():
            if getattr(item, "enabled", 0) and key in legacy.data:
                item = deepcopy(item)
                for kind, code in CAST.items():
                    if kind in sizes and bytes(getattr(key, "value", key)) == code and hasattr(item, "blur"):
                        item.blur = int(round(sizes[kind] * 65536))
                legacy.data[key] = item
    else:
        legacy = None
    return block, legacy, sorted(k.decode() for k in take), False


def _effects_off(layer, status):
    """_effects when the layer's enabled lfx2 effects draw otherwise (psd_edit._style_key), else None. lrFX is not
    compared: Photoshop writes it again from lfx2 on every save (Hexed's glow colour 2 -> 0 of 65535)."""
    made = _effects(layer, status)
    if made is None:
        return None
    now = psd_edit._style_key({LFX2: layer.tagged_blocks.get(LFX2)})
    return None if now == psd_edit._style_key({LFX2: made[0]}) else made


def _set_effects(layer, made) -> None:
    block, legacy, _, whole = made
    if whole:                               # an existing block replaced where it stands, a new one added
        layer.tagged_blocks[LFX2] = block
        if legacy is not None:
            layer.tagged_blocks[LRFX] = legacy
        layer._psd.mark_updated()
        return
    mine = layer.tagged_blocks.get(LFX2).data
    mine.clear()
    for key, value in block.data.items():
        mine[key] = value
    if legacy is not None:
        old = layer.tagged_blocks.get(LRFX).data
        for key, value in legacy.data.items():
            old[key] = value
    layer._psd.mark_updated()


# ---------- fit ----------
def _ink_box(layer):
    alpha = psd_edit._alpha(layer)
    if alpha is None or not alpha.size:
        return None
    ys, xs = np.nonzero(alpha > psd_edit.INK_ALPHA)
    if not len(xs):
        return None
    x0, y0 = layer.bbox[:2]
    return int(xs.min()) + x0, int(ys.min()) + y0, int(xs.max()) + 1 + x0, int(ys.max()) + 1 + y0


def margins(box, w: int, h: int, reach: int) -> Optional[tuple]:
    """Px the box (grown by reach) keeps from each canvas side: (left, top, right, bottom); negative past it."""
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return x0 - reach, y0 - reach, w - x1 - reach, h - y1 - reach


def _canvas_ink(layer, shape, reach: int) -> np.ndarray:
    """The layer's ink (alpha past INK_ALPHA) on the canvas, grown by reach px (its outside stroke)."""
    out = np.zeros(shape, bool)
    alpha = psd_edit._alpha(layer)
    if alpha is None:
        return out
    x0, y0 = layer.bbox[:2]
    ink = alpha > psd_edit.INK_ALPHA
    h, w = shape
    if not ink.any():
        return out
    if reach:
        from scipy.ndimage import binary_dilation
        yy, xx = np.mgrid[-reach:reach + 1, -reach:reach + 1]
        ink = binary_dilation(np.pad(ink, reach), structure=xx * xx + yy * yy <= reach * reach)
        x0, y0 = x0 - reach, y0 - reach
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(w, x0 + ink.shape[1]), min(h, y0 + ink.shape[0])
    if cx1 > cx0 and cy1 > cy0:
        out[cy0:cy1, cx0:cx1] = ink[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    return out


def header_ink(psd, title_layers) -> np.ndarray:
    """The header icons' ink (status symbol, stars: visible pixel and smart object layers in the top HEADER of the card)."""
    shape = (psd.height, psd.width)
    out = np.zeros(shape, bool)
    for layer in psd.descendants():
        if layer.is_group() or layer.kind == "type" or not layer.is_visible() or any(layer is t for t in title_layers):
            continue
        x0, y0, x1, y1 = layer.bbox
        if y1 <= y0 or x1 <= x0 or y1 > psd.height * HEADER or (x1 - x0) * (y1 - y0) > 0.1 * psd.width * psd.height:
            continue
        out |= _canvas_ink(layer, shape, 0)
    return out


def header_distance(header: np.ndarray) -> Optional[np.ndarray]:
    """Each canvas pixel's distance to the nearest header icon ink pixel (None: no header icons)."""
    if not header.any():
        return None
    from scipy.ndimage import distance_transform_edt
    return distance_transform_edt(~header)


class _Fit:
    """What the layer must keep after the font change: its box (every line drawn, none below it), the card (ink and
    outside stroke at least min(its old margin, EDGE) from each side) and, a title, the header icons (no more of their
    ink under it, and at least min(its old gap, EDGE) px from them)."""

    def __init__(self, psd, layer, keep, header, distance=None):
        self.psd, self.layer, self.keep, self.header, self.distance = psd, layer, keep, header, distance
        self.reach = psd_edit._stroke_reach(layer)
        box = _ink_box(layer)
        old = margins(box, psd.width, psd.height, self.reach)
        self.need = None if old is None else tuple(min(m, EDGE) for m in old)
        self.centre = (box[0] + box[2]) / 2 if box is not None and _lopsided(layer) else None
        self.hit = self._hit() if header is not None else 0
        self.gap = min(self._gap(), EDGE) if header is not None else None

    def _hit(self) -> int:
        return int((_canvas_ink(self.layer, (self.psd.height, self.psd.width), self.reach) & self.header).sum())

    def _gap(self) -> float:
        """Px between the layer's ink with its outside stroke and the header icons' ink (0 or less: touching)."""
        if self.distance is None:
            return float("inf")
        ink = _canvas_ink(self.layer, (self.psd.height, self.psd.width), 0)
        return float(self.distance[ink].min()) - 1 - self.reach if ink.any() else float("inf")

    def why(self, report) -> Optional[str]:
        lines = [li for li in report.lines if li.text.strip()]
        if any(not li.drawn for li in lines) or psd_edit.below_box(self.keep, report.lines):
            return "a line falls below its box"
        now = margins(_ink_box(self.layer), self.psd.width, self.psd.height, self.reach)
        if self.need is not None and now is not None and any(n < m for n, m in zip(now, self.need)):
            return f"too near the card's edge {now}"
        if self.header is not None and (self._hit() > self.hit or self._gap() < self.gap):
            return "too near the header icons"
        return None


def _size_runs(layer) -> list:
    """[(start, end, {key: value})] per style run, what a smaller size scales: FontSize, Leading when fixed (the Normal
    sheet's where a run has none) and an icon run's BaselineShift (a text run's stays: the 11-07 subscript conversion
    placed lines with it)."""
    from psd_tools.api import text_edit
    text = text_edit._editor_text(layer)
    units = text_edit.utf16_prefix(text)
    resources = layer.resource_dict
    try:
        normal = resources["StyleSheetSet"][int(_num(resources["TheNormalStyleSheet"]))]["StyleSheetData"]
    except (KeyError, IndexError, TypeError, ValueError):
        normal = {}
    names = [str(getattr(font["Name"], "value", font["Name"])) for font in resources["FontSet"]]
    out = []
    for u0, u1, data in text_edit._spans(text_edit._section(layer, "StyleRun"), units[-1]):
        sheet = data["StyleSheet"]["StyleSheetData"]
        font = sheet.get("Font", normal.get("Font"))
        font = names[int(_num(font))] if font is not None and 0 <= int(_num(font)) < len(names) else ""
        icon = font.lower().startswith(ICON_FONT_PREFIX)
        keys = {}
        for key in ("FontSize", "Leading") + (("BaselineShift",) if icon else ()):
            value = sheet.get(key, normal.get(key))
            if value is not None:
                keys[key] = _num(value)
        auto = sheet.get("AutoLeading", normal.get("AutoLeading"))
        if auto is None or bool(_num(auto)):
            keys.pop("Leading", None)
        out.append((text_edit._str_index(units, u0), text_edit._str_index(units, u1), keys))
    return out


def _lopsided(layer) -> bool:
    """Point text not centred (left or right justified): it grows from one side, the line off its place."""
    if not psd_edit.point_text(layer):
        return False
    engine, resources = layer.engine_dict, layer.resource_dict
    try:
        normal = resources["ParagraphSheetSet"][int(_num(resources["TheNormalParagraphSheet"]))]["Properties"]
    except (KeyError, IndexError, TypeError, ValueError):
        normal = {}
    return any(int(_num(item["ParagraphSheet"]["Properties"].get("Justification", normal.get("Justification", 0)))) != 2
               for item in engine["ParagraphRun"]["RunArray"])


def _lay(layer, fonts, centre):
    """The layer rasterized; point text not centred moved across (TySh and raster, whole px) so its ink keeps its old
    centre, then rasterized again there."""
    report = layer.rasterize(fonts=fonts)
    box = _ink_box(layer) if centre is not None else None
    dx = round(centre - (box[0] + box[2]) / 2) if box is not None else 0
    if dx:
        psd_edit.nudge(layer, dx, 0)
        report = layer.rasterize(fonts=fonts)
    return report


def _badges(layer, fonts) -> int:
    """The layer's badges and icons fitted to its text as the guards fit them (badge_fits, badge_wants)."""
    wants = [(fit, keys) for fit in psd_edit.badge_fits(layer, fonts) for keys in [psd_edit.badge_wants(fit)] if keys]
    for fit, keys in wants:
        layer.set_span_style(fit["start"], fit["end"], **keys)
    return len(wants)


def fit(psd, layer, fonts, keep, header, distance=None) -> tuple:
    """The layer's badges fitted and it rasterized as it stands, or at the largest scale (FIT_STEP grain, down to
    FIT_MIN) that fits (_Fit), its badges fitted again at that size; point text not centred kept on its old centre
    (_lay). Returns (scale, badges fitted, report); raises Refused."""
    check = _Fit(psd, layer, keep, header, distance)
    fitted = _badges(layer, fonts)
    runs = _size_runs(layer)
    report = _lay(layer, fonts, check.centre)
    first = check.why(report)
    if first is None:
        return 1.0, fitted, report
    k = 1
    while True:
        scale = round(1 - FIT_STEP * k, 4)
        if scale < FIT_MIN:
            raise Refused("overflow", f"{layer.name.strip()}: {first}, even at {FIT_MIN:.0%} of its size")
        for a, b, keys in runs:
            if b > a and keys:
                layer.set_span_style(a, b, **{key: round(value * scale, 2) for key, value in keys.items()})
        fitted = _badges(layer, fonts)
        report = _lay(layer, fonts, check.centre)
        if check.why(report) is None:
            return scale, fitted, report
        k += 1


# ---------- the recipe ----------
def status_of(psd, rel: str, chart: dict) -> tuple:
    """(title layers, StatusColor or None): the parser's title, its words' chart status (the template: none)."""
    texts = [layer for layer in psd.descendants() if layer.kind == "type"]
    name = basename(rel)[:-4].replace("_", " ")
    titles = PSDParser.title_layers(psd, texts, name)
    if rel == TEMPLATE:
        return titles, None
    words = psd_edit.collapse(" ".join(layer.text or "" for layer in titles)).lower()
    return titles, chart.get(words) or chart.get(name.lower())


def _rgb(fill) -> tuple:
    return tuple(round(float(v) * 255) for v in tuple(fill)[-3:])


def _wants(layer, title: bool, status) -> tuple:
    """(body runs, font off, fill off, effects off) of a visible text layer: what restyle changes on it."""
    from commands.card_standards import run_fonts
    _, runs = run_fonts(layer)
    body = [(a, b) for a, b, font in runs
            if b > a and not font.lower().startswith(ICON_FONT_PREFIX) and font.lower() not in SKIP_FONTS]
    if not body:
        return body, False, False, None
    font = any(f != KP for a, b, f in runs if (a, b) in body)
    titled = title and status is not None
    fill = titled and any(c is not None and _rgb(c) != _rgb(status.fill)
                          for a, b, c in psd_edit.run_fills(layer) if any(a < y and x < b for x, y in body))
    effects = _effects_off(layer, status) if titled and status.styled else None
    return body, font, fill, effects


def _start(psd, rel: str, chart: dict) -> tuple:
    if not in_scope(rel):
        raise Refused("unsupported-card", f"{rel}: not a status effect card")
    titles, status = status_of(psd, rel, chart)
    if not titles:
        raise Refused("unsupported-card", f"{rel}: no title")
    if status is None and rel != TEMPLATE:
        raise Refused("unsupported-card", f"{rel}: its title is no chart status")
    return titles, status


def plan(psd, rel: str, chart: Optional[dict]) -> list:
    """The visible text layers restyle would change (nothing drawn, nothing edited): [] out of scope or unreadable."""
    if not chart or not rel or not in_scope(rel):
        return []
    try:
        titles, status = _start(psd, rel, chart)
    except Refused:
        return []
    out = []
    for layer in psd.descendants():
        if layer.kind == "type" and layer.is_visible():
            body, font, fill, effects = _wants(layer, any(layer is t for t in titles), status)
            if font or fill or effects:
                out.append(layer)
    return out


def restyle(psd, rel: str, chart: dict, fonts_dir: str) -> list:
    """The card's visible text in Krabby Patty, its title in its status's look (module doc); in place. Returns what it
    did, one line per layer changed; raises Refused."""
    from psd_tools.composite.text import capture
    from commands.card_standards import _recut_fx
    titles, status = _start(psd, rel, chart)
    fonts = psd_edit.font_map(fonts_dir)
    header = header_ink(psd, titles)
    distance = header_distance(header)
    done = []
    for layer in [layer for layer in psd.descendants() if layer.kind == "type" and layer.is_visible()]:
        title = any(layer is t for t in titles)
        body, font, fill, effects = _wants(layer, title, status)
        if not (font or fill or effects):
            continue
        what = []
        if font or fill:
            psd_edit.follow_kerning(psd, layer)
            keep = capture(layer, fonts)
            if font:
                layer.add_font(KP)
                for a, b in body:
                    layer.set_span_style(a, b, Font=KP)
                what.append("Krabby Patty")
            if fill:
                for a, b in body:
                    layer.set_span_style(a, b, fill=tuple(status.fill))
                what.append("chart colour #%02x%02x%02x" % _rgb(status.fill))
        if effects:
            _set_effects(layer, effects)
            what.append("chart effects " + "+".join(effects[2]))
        if font or fill:
            scale, badges, report = fit(psd, layer, fonts, keep, header if title else None, distance)
            bad = [name for name, (how, _) in report.fonts.items() if how != "exact"]
            if bad:
                raise Refused("unsupported-card", f"{layer.name.strip()}: fonts not exact: {bad}")
            if badges:
                what.append(f"{badges} badges and icons fitted")
            if scale < 1:
                what.append(f"set at {scale:.1%} of its size to fit")
            engine = psd_edit._fx_engine()
            cut = _recut_fx(psd, layer, set(range(len(engine[0].from_layer(layer).text))), fonts)
            what += [f"{fx.name.strip()}: cut again" for fx, _ in cut]
        done.append(f"{layer.name.strip()}: " + ", ".join(what))
    return done


def gate(old_path: str, new_path: str, rel: str, chart: dict) -> None:
    """restyle's gate: the same records; a visible text record changed in its rect, channels, TySh and fxrp alone (moved:
    _lay; the title also its lfx2 and lrFX), an FX layer in its rect and channels; the document as it was but the edited layers' Txt2
    stories (gate_stories, Krabby Patty added). Raises Refused("bytes")."""
    a, b = psd_edit.layer_records(old_path), psd_edit.layer_records(new_path)
    if len(a) != len(b):
        raise Refused("bytes", "layer count changed")
    before, after = PSDImage.open(old_path), PSDImage.open(new_path)
    texts = [layer for layer in before.descendants() if layer.kind == "type" and layer.is_visible()]
    titles = {psd_edit._record_index(layer) for layer in status_of(before, rel, chart)[0]}
    edited = {psd_edit._record_index(layer) for layer in texts}
    fxs = {psd_edit._record_index(layer) for layer in before.descendants() if EFFECT_LAYER_TAG in layer.name}
    text_blocks = {bytes(Tag.TYPE_TOOL_OBJECT_SETTING), bytes(Tag.REFERENCE_POINT)}
    fx_blocks = {bytes(LFX2), bytes(LRFX)}
    for i, (x, y) in enumerate(zip(a, b)):
        if x == y:
            continue
        keys = {k for k in x if x[k] != y[k]}
        allowed = text_blocks | (fx_blocks if i in titles else set()) if i in edited else set()
        had = {k for k, _ in x["blocks"]}
        kept, added = [entry for entry in y["blocks"] if entry[0] in had], [k for k, _ in y["blocks"] if k not in had]
        if [k for k, _ in kept] != [k for k, _ in x["blocks"]] or set(added) - (fx_blocks & allowed) \
                or len(added) != len(set(added)):         # a title may gain lfx2 and lrFX (a title with no style)
            raise Refused("bytes", f"{x['name']}: its blocks changed")
        blocks = {k for (k, v), (_, w) in zip(x["blocks"], kept) if v != w} | set(added)
        if i in edited:
            if keys - {"rect", "channels", "blocks"} or blocks - allowed:
                raise Refused("bytes", f"{x['name']}: changed beyond its text and effects")
        elif i in fxs:
            if keys - {"rect", "channels"}:
                raise Refused("bytes", f"{x['name']}: changed beyond its pixels")
        else:
            raise Refused("bytes", f"{x['name']}: changed")
    part = psd_edit.changed_part(before, after, ("header", "color_mode_data", "image_resources"))
    if part:
        raise Refused("bytes", f"{part} changed")
    psd_edit.gate_stories(before, after, [psd_edit.story_index(layer) for layer in texts
                                          if psd_edit._record_index(layer) in edited], fonts=[KP])


def fix(path: str, out_path: str, fonts_dir: str, chart: Optional[dict], rel: str) -> list:
    """The guards' "status font" kind (owner 2026-11-09: the guards keep status cards this way, silently): restyle,
    saved, gated, then checked: nothing left to restyle and no badge left to fit. Returns the layer names changed."""
    from commands.card_standards import badge_plan
    psd = PSDImage.open(path)
    if not plan(psd, rel, chart):
        return []
    done = restyle(psd, rel, chart, fonts_dir)
    if not done:
        return []
    psd_edit.save_psd(psd, out_path)
    gate(path, out_path, rel, chart)
    after = PSDImage.open(out_path)
    if plan(after, rel, chart):
        raise Refused("bytes", "a status card's text is left off its look")
    if badge_plan(after, lambda names: fonts_dir):
        raise Refused("bytes", "a badge is left off its text")
    return [line.partition(":")[0] for line in done]
