"""
Where an added ability ("+ability", PARTS_EDITOR.md §4.7) goes and in what font: plan() for psd_edit.add_ability.

Place (owner 2026-10-25: below the stats, not on them; at the top of an item's panel, not its bottom): the card's free
text area under the lowest element above it, the text flowing down from a top-aligned box. The offsets come from the
card's siblings: other cards with an ability, the same template and canvas, and the same layout (stats within
STATS_TOL px, else the text panel within PANEL_TOL px, else a shared distinctive layer name), the nearest folder first.
No sibling: just below the stats, else the template's medians. Never over the stats, title, stars or type icons, never
off the canvas: shrink-to-fit (FIT_STEP down to FIT_MIN), then raised towards what is above.

Font (owner 2026-10-25 "we should use a card's own font when it exists"): the commonest real ability font of the
cards sharing this card's title font in its folder, else its series, else anywhere, else of its series, else its own
title font, else the template's (FONT_STEPS; a tie falls through). One not in MDW/Fonts: the template's, told. Size in
rendered px (FontSize x the layer's scale x the fork's super/subscript factor): the siblings' in that font, else the
font's median, else the template's by cap height. Colours, effects, alignment and leading ratio stay the template's.

The other cards come from ability_index.json, made by hand from the card repo's working tree:
    python -m commands.ability_place ~/Desktop/TTSCardMaker
"""
import json
import statistics as st
from collections import Counter
from dataclasses import dataclass, field, replace
from functools import lru_cache
from math import hypot
from os.path import dirname, join
from pathlib import Path
from re import IGNORECASE, compile as re_compile
from typing import Optional

from psd_tools import PSDImage

from commands.psd_parser import ICON_FONT_PREFIX, base_name, variants_of
from commands.psd_regions import regions

# ========================
# Configuration
# ========================
INDEX = Path(__file__).with_name("ability_index.json")
STATS_TOL, PANEL_TOL = 40, 30       # px: two layouts match
GAP = 15                            # px: least room between the text and an element
EDGE = 12                           # px: least room to the canvas or panel bottom
SIDE = 30                           # px: side margin with no sibling (and inside a panel)
BELOW_STATS = 30                    # px: no sibling: the first line this far under the stats
FIT_STEP, FIT_MIN = 0.94, 0.4       # shrink-to-fit, as psd_edit.fit_below
ROW_MAX, ROW_HALF, FRAME = 130, 42, 0.1     # a taller stat row is a digit mask (its label's band); a bigger icon a frame
LABELS = ("health", "defense", "attack", "speed")
STYLE_COUNT = (2, 60)               # a distinctive layer name is on this many cards
JUNK_FONTS = {"AdobeInvisFont", "MyriadPro-Regular", "TimesNewRomanPSMT"}
SKIP_STYLE = re_compile(r"star|^(def|hp|atk|spd|dark|header|page number|offensive_minion|debuff|buff|debuffstack|buffstack)$"
                        r"|^clip |fx:|^(gradient map|selective color|hue/saturation|curves|levels|brightness|color balance|"
                        r"vibrance|exposure|color fill|solid color)|^\d+$", IGNORECASE)
DEFAULT_NAME = re_compile(r"^(layer|rectangle|ellipse|group|shape|polygon|path)\s*\d+(\s+copy(\s+\d+)?)?$", IGNORECASE)
FONT_STEPS = ("folder+title font", "series+title font", "title font", "series", "own title font", "template")
TALL = 100000.0                     # pt: a box no text reaches the bottom of (measuring)


@dataclass
class Plan:
    """plan()'s answer: the paragraph box and the ink expected in it (canvas px), the font, size and leading (rendered
    px), and the notes."""
    box: tuple
    ink: tuple
    font: str
    size: float
    leading: float
    notes: list = field(default_factory=list)


# ========================
# The card's layout
# ========================
def elements(psd: PSDImage, vocab=()) -> dict:
    """Stats rows, title, stars and type icons (their own boxes, no effects; a digit mask far past its bar is its
    label's band, a star or icon box over FRAME of the canvas a frame) and the text panel."""
    found = regions(psd, vocab, False, pad=False)
    labels = {layer.name.strip().lower(): layer.bbox for layer in psd.descendants()
              if layer.kind == "type" and layer.is_visible() and layer.name.strip().lower() in LABELS}
    stats = [tuple(b) for b in found.get("stat_labels", [])]
    for key, label in zip(("stat_hp", "stat_def", "stat_atk", "stat_spd"), LABELS):
        for b in found.get(key, []):
            if b[3] - b[1] > ROW_MAX * psd.width / 1200 and label in labels:
                mid = (labels[label][1] + labels[label][3]) // 2
                b = (b[0], mid - ROW_HALF, b[2], mid + ROW_HALF)
            stats.append(tuple(b))
    small = lambda boxes: [tuple(b) for b in boxes if (b[2] - b[0]) * (b[3] - b[1]) <= FRAME * psd.width * psd.height]
    return {"stats": stats, "title": [tuple(b) for b in found.get("name", [])], "stars": small(found.get("stars", [])),
            "types": small(found.get("types", [])), "panel": panel(psd)}


def panel(psd: PSDImage) -> Optional[tuple]:
    """The text panel: the biggest visible drawn layer starting in the lower part, spanning most of the width (Isaac's
    paper)."""
    W, H = psd.width, psd.height
    best = None
    for layer in psd.descendants():
        if layer.kind in ("group", "type", "solidcolorfill", "gradientfill", "patternfill") or not layer.is_visible():
            continue
        x0, y0, x1, y1 = layer.bbox
        w, h = min(x1, W) - max(x0, 0), y1 - y0
        if w >= 0.75 * W and 0.1 * H <= h <= 0.6 * H and y0 >= 0.45 * H and y1 <= H + 5 and x0 >= -20 and x1 <= W + 20:
            if best is None or w * h > (best[2] - best[0]) * (best[3] - best[1]):
                best = tuple(layer.bbox)
    return best


def layer_font(layer) -> tuple:
    """(font, rendered px) with the most visible characters, the icon font and Photoshop's defaults aside; rendered:
    FontSize x the layer's y scale x the super/subscript factor."""
    from psd_tools.composite.text.model import TextModel
    model = TextModel.from_layer(layer)
    scale = hypot(*model.transform[2:4])
    count = Counter()
    for run in model.runs:
        font = run.style.font
        if font and font not in JUNK_FONTS and not font.lower().startswith(ICON_FONT_PREFIX):
            size = run.style.size * scale * (model.sub_size if run.style.baseline in (1, 2) else 1)
            count[(font, round(size, 1))] += sum(1 for c in model.text[run.start:run.end] if not c.isspace() and c != "﻿")
    return count.most_common(1)[0][0] if count and count.most_common(1)[0][1] else (None, None)


def title_font(psd: PSDImage, rel: str, parser) -> tuple:
    """layer_font() of the card's title (its title parts, else a visible Name layer)."""
    from commands.psd_edit import card_parts, lyid_index
    try:
        ids = [part["id"] for part in card_parts(psd, rel, Path(rel).stem.replace("_", " "), parser).get("title") or []]
    except Exception:                           # a card the parts record refuses: its Name layer
        ids = []
    index = lyid_index(psd)
    layers = [index[i] for i in ids if i in index and index[i].is_visible()]
    layers += [layer for layer in psd.descendants() if layer.kind == "type" and layer.is_visible()
               and layer.name.strip().lower() == "name"]
    for layer in layers:
        found = layer_font(layer)
        if found[0]:
            return found
    return None, None


def own_names(psd: PSDImage, template: PSDImage, vocab=()) -> set:
    """The card's layer names its template lacks (its style), Photoshop's default names and type icons aside."""
    theirs = {layer.name.strip() for layer in template.descendants()}
    icons = {name.lower() for name in vocab}
    return {name for name in (layer.name.strip() for layer in psd.descendants())
            if name not in theirs and name.lower() not in icons and not DEFAULT_NAME.match(name)}


def para_box(layer) -> Optional[tuple]:
    """A paragraph text layer's box in canvas px (BoxBounds through its transform); None for point text."""
    try:
        child = layer.engine_dict["Rendered"]["Shapes"]["Children"][0]
        if int(child.get("ShapeType", 0)) != 1:
            return None
        b = [float(v) for v in child["Cookie"]["Photoshop"]["BoxBounds"]]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    xx, xy, yx, yy, tx, ty = (float(v) for v in layer.transform)
    pts = [(xx * u + yx * v + tx, xy * u + yy * v + ty) for u in (b[0], b[2]) for v in (b[1], b[3])]
    return (round(min(p[0] for p in pts)), round(min(p[1] for p in pts)),
            round(max(p[0] for p in pts)), round(max(p[1] for p in pts)))


def ink_rows(layer) -> tuple:
    """(ink box, top of its first ink row run) from the layer's saved pixels, canvas px; (bbox, None) when blank."""
    import numpy as np
    x0, y0 = layer.bbox[:2]
    try:
        ink = layer.numpy("shape")[..., 0] > 0.5
    except Exception:
        return tuple(layer.bbox), None
    rows, cols = np.flatnonzero(ink.any(1)), np.flatnonzero(ink.any(0))
    if not len(rows):
        return tuple(layer.bbox), None
    return (int(x0 + cols[0]), int(y0 + rows[0]), int(x0 + cols[-1] + 1), int(y0 + rows[-1] + 1)), int(y0 + rows[0])


# ========================
# The other cards (ability_index.json)
# ========================
@lru_cache(None)
def _index(path: str) -> dict:
    try:
        cards = json.loads(Path(path).read_text(encoding="utf-8"))["cards"]
    except (OSError, ValueError, KeyError):
        return {}
    for rel, card in cards.items():
        card.update(rel=rel, folder=dirname(rel), top=rel.split("/")[0], style=set(card.get("style") or ()),
                    canvas=tuple(card["canvas"]), stats=tuple(card["stats"]) if card.get("stats") else None,
                    panel=tuple(card["panel"]) if card.get("panel") else None,
                    title=tuple(card["title"]) if card.get("title") else (None, None))
    return cards


def index(path: Optional[str] = None) -> dict:
    """rel -> the card's facts (build_card's, plus rel, folder, top)."""
    return _index(str(path or INDEX))


@lru_cache(None)
def _styles(path: str) -> frozenset:
    return frozenset(name for card in _index(path).values() for name in card["style"])


def _mode(c: dict) -> str:
    return "stats" if c["stats"] else "panel" if c["panel"] else "plain"


def _anchor(c: dict) -> float:
    return c["stats"][1] if c["stats"] else c["panel"][1] if c["panel"] else 0


def _near(a, b, tol: int) -> bool:
    return bool(a and b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def _sibling(c: dict, o: dict) -> bool:
    if o["rel"] == c["rel"] or "first" not in o or o["tmpl"] != c["tmpl"] or o["canvas"] != c["canvas"]:
        return False
    if c["stats"] or o["stats"]:
        return _near(c["stats"], o["stats"], STATS_TOL)
    if c["panel"] or o["panel"]:
        return _near(c["panel"], o["panel"], PANEL_TOL)
    return bool(c["style"] & o["style"])


def siblings(c: dict, cards: dict) -> tuple:
    """(the cards laid out like c, the scope: folder, top or anywhere), nearest scope first; c itself never."""
    pool = [o for o in cards.values() if _sibling(c, o)]
    for scope in ("folder", "top", None):
        found = [o for o in pool if scope is None or o[scope] == c[scope]]
        if found:
            return found, scope or "anywhere"
    return [], None


def _fallback(cards: dict, tmpl: str, mode: str, rel: str) -> Optional[dict]:
    """The template's medians in a mode: the first line's offset from the anchor, the ink's gap to the bottom, the
    median first line."""
    group = [o for o in cards.values() if "first" in o and o["tmpl"] == tmpl and _mode(o) == mode and o["rel"] != rel]
    if not group:
        return None
    return {"offset": st.median(o["first"] - _anchor(o) for o in group),
            "bottom_gap": st.median(o["canvas"][1] - o["ink"][3] for o in group),
            "high": st.median(o["first"] for o in group)}


# ========================
# Font and size
# ========================
def _most(fonts) -> Optional[str]:
    """The commonest font; None for none or a tie."""
    top = Counter(f for f in fonts if f).most_common(2)
    return top[0][0] if top and (len(top) < 2 or top[0][1] > top[1][1]) else None


def font_steps(c: dict, cards: dict) -> list:
    """[(step, its font or None)] in FONT_STEPS order, the template's left out; c's own ability never counts."""
    tf = c["title"][0]
    others = [o for o in cards.values() if o["rel"] != c["rel"] and o.get("font")]
    titled = [o for o in others if tf and o["title"][0] == tf]
    series = c.get("series")
    return [("folder+title font", _most(o["font"] for o in titled if o["folder"] == c["folder"])),
            ("series+title font", _most(o["font"] for o in titled if series and o.get("series") == series)),
            ("title font", _most(o["font"] for o in titled)),
            ("series", _most(o["font"] for o in others if series and o.get("series") == series
                             and o["canvas"] == c["canvas"])),
            ("own title font", tf)]


def size_for(c: dict, font: str, cards: dict) -> tuple:
    """(rendered px, source) of the font on c: the median over c's folder or style cards and its series or folder
    cards with its title font that use it (by canvas width), else the font's median, else None."""
    w = c["canvas"][0]
    tf = c["title"][0]
    near = [o for o in cards.values() if o["rel"] != c["rel"] and o.get("font") == font and o.get("size")
            and ((o["canvas"] == c["canvas"] and (o["folder"] == c["folder"] or o["style"] & c["style"]))
                 or (tf and o["title"][0] == tf and ((c.get("series") and o.get("series") == c["series"])
                                                    or o["folder"] == c["folder"])))]
    if near:
        return st.median(o["size"] * w / o["canvas"][0] for o in near), "siblings"
    every = [o["size"] / o["canvas"][0] for o in cards.values() if o["rel"] != c["rel"] and o.get("font") == font
             and o.get("size")]
    return (st.median(every) * w, "the font's median") if every else (None, None)


def _cap(fonts, name: str) -> Optional[float]:
    """Cap height / em of a font (the engine's ascent glyph, when it is H)."""
    try:
        face = fonts.resolve(name).face
        metrics = face.metrics
        return metrics.ascent_units / metrics.upem if metrics.ascent_glyph == "H" and metrics.upem else None
    except Exception:
        return None


def in_fonts_dir(fonts, name: str, fonts_dir: str) -> bool:
    """A file in fonts_dir has this exact PostScript name."""
    found = fonts.find(name)
    return bool(found) and found[2] == "exact" and Path(found[0]).resolve().is_relative_to(Path(fonts_dir).resolve())


def pick_font(c: dict, cards: dict, tmpl_font: str, tmpl_size: float, fonts, fonts_dir: str) -> tuple:
    """(font, rendered px, notes): the card's own font (FONT_STEPS) and its size, else the template's."""
    step, font = next(((s, f) for s, f in font_steps(c, cards) if f), ("template", tmpl_font))
    notes = []
    if font != tmpl_font and not in_fonts_dir(fonts, font, fonts_dir):
        notes.append(f"ability: the card's font {font} is not in MDW/Fonts: the template's {tmpl_font} used")
        step, font = "template", tmpl_font
    if font == tmpl_font:
        return font, tmpl_size, notes + [f"ability: font {font} ({step}), {tmpl_size:.0f} px (the template's)"]
    size, how = size_for(c, font, cards)
    if size is None:
        a, b = _cap(fonts, tmpl_font), _cap(fonts, font)
        size, how = (tmpl_size * a / b, "the template's by cap height") if a and b else (tmpl_size, "the template's")
    return font, size, notes + [f"ability: font {font} ({step}), {size:.0f} px ({how})"]


# ========================
# Measuring (the fork's layout, nothing drawn)
# ========================
def measure(model, text: str, font: str, size_pt: float, lead_pt: float, width_pt: float, fonts) -> tuple:
    """
    The text laid out in the template's style with the font and size, in a box width_pt wide.

    Args:
        model: TextModel of the template's Ability layer
        text: The text, paragraphs split by \\r

    Returns:
        (lines, ink x0, ink x1 from the box's left, the first line's ink top below the box's top, ink height,
        the face drawn) in canvas px; zeros and None when nothing inks
    """
    from psd_tools.composite.text import layout as engine
    from psd_tools.composite.text.geometry import _ink, _map_box
    from psd_tools.composite.text.model import Paragraph, Run
    style = replace(model.runs[0].style, font=font, size=size_pt, leading=lead_pt, baseline=0)
    paragraphs, start = [], 0
    for piece in text.split("\r"):
        end = min(start + len(piece) + 1, len(text))
        paragraphs.append(Paragraph(start, end, model.paragraphs[0].justification))
        start = end
    xx, xy, yx, yy = model.transform[:4]
    one = replace(model, text=text, runs=(Run(0, len(text), style),), paragraphs=tuple(paragraphs),
                  box=(0.0, 0.0, width_pt, TALL), transform=(xx, xy, yx, yy, 0.0, 0.0))
    laid = engine.layout(one, fonts)
    inks = []
    for glyph in laid.glyphs:
        ink = _ink(glyph, 0.0)
        if ink is not None:
            inks.append((glyph, _map_box(one.transform, ink)))
    if not inks:
        return 0, 0, 0, 0, 0, None
    starts = [line.start for line in laid.lines]
    line_of = lambda g: max(i for i, s in enumerate(starts) if s <= g.cluster) if starts else 0
    lines = {line_of(g) for g, _ in inks}
    first = [box for g, box in inks if line_of(g) == min(lines)]
    drawn = Counter((g.face.ft.postscript_name or b"").decode() for g, _ in inks).most_common(1)[0][0]
    top = min(box[1] for box in first)
    return (len(lines), min(b[0] for _, b in inks), max(b[2] for _, b in inks), top,
            max(b[3] for _, b in inks) - top, drawn)


# ========================
# The plan
# ========================
def _clear(top: float, bottom: float, x0: float, x1: float, boxes: list) -> tuple:
    """The free band [top, bottom] in x0..x1: its top pushed under any element it starts on, its bottom above the
    first element under it."""
    for _ in range(4):
        hit = [b for b in boxes if b[0] < x1 and b[2] > x0 and b[1] - 4 < top < b[3] + GAP]
        if not hit:
            break
        top = max(b[3] for b in hit) + GAP
    below = [b[1] - GAP for b in boxes if b[0] < x1 and b[2] > x0 and b[1] >= top]
    return top, min([bottom] + below)


def _bands(H: int, x0: float, x1: float, boxes: list, pan) -> list:
    """Free vertical bands in x0..x1 between the elements, in the lower half (the art is above)."""
    ys = sorted((b[1], b[3]) for b in boxes if b[0] < x1 and b[2] > x0)
    out, y = [], H // 2
    for a, b in ys:
        if a - GAP > y + GAP:
            out.append((y + GAP, a - GAP))
        y = max(y, b)
    end = (pan[3] if pan else H) - EDGE
    if end > y + GAP:
        out.append((y + GAP, end))
    return out


def card_facts(psd: PSDImage, rel: str, template_rel: str, template: PSDImage, parser) -> dict:
    """What plan() and the index know of a card, live from its PSD: template, canvas, folder, stats span, text panel,
    distinctive names, title font; series from the index."""
    found = elements(psd, getattr(parser, "all_types", ()))
    stats = found["stats"]
    return {"rel": rel, "tmpl": template_rel, "canvas": (psd.width, psd.height), "folder": dirname(rel),
            "top": rel.split("/")[0], "stats": (min(b[1] for b in stats), max(b[3] for b in stats)) if stats else None,
            "panel": found["panel"], "title": title_font(psd, rel, parser),
            "names": own_names(psd, template, getattr(parser, "all_types", ())),
            "elements": found}


def plan(psd: PSDImage, rel: str, template: PSDImage, template_rel: str, text: str, fonts_dir: str, parser=None,
         index_path: Optional[str] = None) -> Plan:
    """
    Where the added ability goes on the card (as it is, without one) and in what font and size.

    Args:
        psd: The card
        rel: Its repo path
        template: Its kind's template (template_for's), read only
        template_rel: That template's repo path
        text: The ability, paragraphs split by \\r
        fonts_dir: MDW/Fonts
        parser: offline_parser() (its icon names); None: none

    Returns:
        Plan; raises psd_edit.Refused when no free area is left or the font can't be laid out
    """
    from commands.psd_edit import Refused, _is_text_layer, font_map
    from psd_tools.composite.text.model import TextModel
    source = next((layer for layer in template.descendants() if _is_text_layer(layer) and layer.is_visible()
                   and base_name(layer).strip().lower() == "ability"), None)
    if source is None:
        raise Refused("unsupported-card", "ability: the template has no visible Ability layer")
    cards = index(index_path)
    W, H = psd.width, psd.height
    me = card_facts(psd, rel, template_rel, template, parser)
    el = me.pop("elements")
    me["style"] = me.pop("names") & _styles(str(index_path or INDEX))
    me["series"] = (cards.get(rel) or {}).get("series") or (rel.split("/")[1].lower() if rel.startswith("Creatures/")
                                                           and rel.count("/") > 2 else None)
    fonts = font_map(fonts_dir)
    model = TextModel.from_layer(source)
    scale = hypot(*model.transform[2:4])
    base = model.runs[0].style
    tmpl_font, tmpl_size = layer_font(source)
    tmpl_font, tmpl_size = tmpl_font or base.font, tmpl_size or base.size * scale
    font, size, notes = pick_font(me, cards, tmpl_font, tmpl_size, fonts, fonts_dir)
    ratio = base.leading / base.size if base.size else 1.0

    sib, scope = siblings(me, cards)
    mode, pan = _mode(me), me["panel"]
    if sib:
        notes.append(f"ability: placed as {len(sib)} sibling{'s' if len(sib) > 1 else ''} ({scope}, {mode})")
        boxes = [o.get("box") or o["ink"] for o in sib]
        x0, x1 = st.median(b[0] for b in boxes), st.median(b[2] for b in boxes)
        first = _anchor(me) + st.median(o["first"] - _anchor(o) for o in sib)
        ink_bottom = st.median(o["ink"][3] for o in sib)
        highest = max(H * 0.5, st.median(o["first"] for o in sib))
    else:                                       # just below the stats; else the template's medians
        fb = _fallback(cards, template_rel, mode, rel) or {"offset": 0, "bottom_gap": EDGE, "high": H * 0.5}
        notes.append(f"ability: no sibling: geometric ({mode})")
        x0, x1 = SIDE, W - SIDE
        first = _anchor(me) + (BELOW_STATS if mode == "stats" else fb["offset"])
        ink_bottom = H - fb["bottom_gap"]
        highest = max(H * 0.5, fb["high"])
    if pan:                                     # inside the panel
        x0, x1 = max(x0, pan[0] + SIDE), min(x1, pan[2] - SIDE)
    x0, x1 = max(x0, SIDE // 2), min(x1, W - SIDE // 2)
    floor = (pan[3] if pan else H) - EDGE
    obstacles = el["stats"] + el["title"] + el["stars"] + el["types"]
    width = x1 - x0

    def laid(s: float) -> tuple:
        try:
            return measure(model, text, font, s / scale, s * ratio / scale, width / model.transform[0], fonts)
        except Exception as error:              # FreeType's raster overflow on Holy Blade's ChiQuel
            raise Refused("font", f"ability: {font} could not be laid out ({error})")

    def fit(top_ink: float, bottom: float) -> tuple:
        """The largest size down to FIT_MIN whose block fits between top_ink and bottom."""
        s = size
        while True:
            m = laid(s)
            if top_ink + m[4] <= bottom or s * FIT_STEP < size * FIT_MIN:
                return s, m
            s *= FIT_STEP

    if mode == "plain":                         # the block ends where the siblings' blocks end, no higher than theirs
        end = min(ink_bottom, floor)
        first = max(highest, end - fit(highest, end)[1][4])
    top, bottom = _clear(first, floor, x0, x1, obstacles)
    if bottom - top < 0.3 * size:               # no room under what it landed on: the biggest free band
        bands = _bands(H, x0, x1, obstacles, pan)
        if not bands:
            raise Refused("unsupported-card", "ability: no free area for it on the card")
        top, bottom = max(bands, key=lambda b: b[1] - b[0])
        notes.append(f"ability: moved to the free band {top:.0f}-{bottom:.0f}")
    s, m = fit(top, bottom)
    _, ix0, ix1, off, h, drawn = m
    if top + h > bottom:                        # past the bottom at the least size: towards what is above
        band_top = _clear(_anchor(me) if mode == "stats" else (pan[1] if pan else H * 0.5), floor, x0, x1, obstacles)[0]
        risen = max(min(top, band_top), bottom - h)
        if risen < top:
            notes.append(f"ability: raised {top - risen:.0f} px to fit")
            top = risen
    if s < size:
        notes.append(f"ability: shrunk to {s / size:.0%}")
    if mode == "plain" and top + h < bottom:    # on the bottom edge after a push
        top = max(top, min(ink_bottom, bottom) - h)
    if drawn and drawn != font:
        notes.append(f"ability: drawn in {drawn}, not {font}")
    box = (round(x0), round(top - off), round(x1), round(bottom))
    return Plan(box, (round(x0 + ix0), round(top), round(x0 + ix1), round(top + h)), font, s, s * ratio, notes)


# ========================
# The index (by hand: python -m commands.ability_place <card repo>)
# ========================
def build_card(clone: str, rel: str, parser) -> Optional[dict]:
    """One card's index entry: card_facts' (no series) plus its real ability's font, rendered size, paragraph box,
    ink box and first ink row (one visible Ability, no variants); None for a card with no template."""
    from commands.psd_edit import Refused, _is_text_layer, ability_layer, template_for
    from commands.psd_parser import CardClassifier
    template_rel = template_for(CardClassifier.classify(rel).get("type"), rel)
    if template_rel is None:
        return None
    psd = PSDImage.open(join(clone, rel))
    facts = card_facts(psd, rel, template_rel, _template(clone, template_rel), parser)
    out = {"tmpl": template_rel, "canvas": list(facts["canvas"]), "stats": facts["stats"], "panel": facts["panel"],
           "title": list(facts["title"]) if facts["title"][0] else None, "names": sorted(facts["names"])}
    shown = [layer for layer in psd.descendants() if _is_text_layer(layer) and layer.is_visible()
             and layer.name.strip().lower() == "ability" and (layer.text or "").strip()]
    if shown:
        out["font"], out["size"] = layer_font(max(shown, key=lambda layer: len(layer.text)))
    try:
        layer = ability_layer(psd) if variants_of(psd) == [1] else None
    except Refused:
        layer = None
    if layer is not None and (layer.text or "").strip():
        ink, first = ink_rows(layer)
        if first is not None:
            out.update(box=para_box(layer), ink=list(ink), first=first)
    return out


@lru_cache(None)
def _template(clone: str, template_rel: str) -> PSDImage:
    return PSDImage.open(join(clone, template_rel))


_WORKER = {}


def _work_init(clone: str) -> None:
    _WORKER.update(clone=clone, parser=_offline(clone))


def _offline(clone: str):
    """offline_parser() over the clone's Types/ names and icon manifest."""
    from commands.psd_edit import offline_parser
    from global_config import ICON_MANIFEST_PATH
    types = Path(clone) / "Types"
    names = sorted(p.stem.lower() for p in types.glob("*.png") if not p.name.startswith("."))
    stars = sorted(p.stem.lower().replace(" ", "_") for p in (types / "Stars").glob("*.png") if not p.name.startswith("."))
    try:
        manifest = json.loads((Path(clone) / ICON_MANIFEST_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = None
    return offline_parser(names, stars, manifest)


def _work(rel: str) -> tuple:
    try:
        return rel, build_card(_WORKER["clone"], rel, _WORKER["parser"]), None
    except Exception as error:
        return rel, None, repr(error)


def build(clone: str, out: Path = INDEX, jobs: int = 2) -> dict:
    """Every card PSD in the clone's working tree (no rule pages, templates or markers) into the index; series from the
    data branch's stats.json when the clone has origin/data. Distinctive names: on STYLE_COUNT cards, SKIP_STYLE aside."""
    import multiprocessing
    import subprocess
    files = subprocess.run(["git", "-C", clone, "ls-files", "*.psd"], capture_output=True, text=True, check=True).stdout
    rels = [r for r in files.splitlines() if not r.startswith(("MDW/", "Types/")) and "/Rulebook/" not in r
            and "/Markers/" not in r]
    shown = subprocess.run(["git", "-C", clone, "show", "origin/data:db/stats.json"], capture_output=True, text=True)
    series = {card.get("path"): card.get("series") for card in json.loads(shown.stdout)["cards"].values()} \
        if shown.returncode == 0 else {}
    cards, errors = {}, {}
    with multiprocessing.get_context("spawn").Pool(max(1, min(jobs, 2)), _work_init, (clone,), maxtasksperchild=25) as pool:
        for rel, card, error in pool.imap_unordered(_work, rels, chunksize=2):
            if error:
                errors[rel] = error
            elif card is not None:
                card["series"] = series.get(rel)
                cards[rel] = card
    count = Counter(name for card in cards.values() for name in card["names"])
    for card in cards.values():
        card["style"] = sorted(n for n in card.pop("names") if STYLE_COUNT[0] <= count[n] <= STYLE_COUNT[1]
                               and not SKIP_STYLE.search(n))
    lines = ",\n".join(f"{json.dumps(rel)}: {json.dumps(cards[rel], sort_keys=True)}" for rel in sorted(cards))
    out.write_text('{"version": 1, "cards": {\n' + lines + "\n}}\n", encoding="utf-8")
    return {"cards": len(cards), "errors": errors}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Make ability_index.json from the card repo's working tree.")
    ap.add_argument("clone", help="the card repo checkout (~/Desktop/TTSCardMaker)")
    ap.add_argument("--out", default=str(INDEX))
    ap.add_argument("--jobs", type=int, default=2, help="worker processes (at most 2)")
    a = ap.parse_args()
    got = build(a.clone, Path(a.out), a.jobs)
    print(f"{got['cards']} cards -> {a.out}")
    for rel, error in sorted(got["errors"].items()):
        print(f"  error {rel}: {error}")
