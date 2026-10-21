"""
Edits to a card PSD for the edit pipeline (docs/EDIT_PIPELINE_DESIGN.md sections 3, 5, 6 and 10): classify a card's
website edits against a fresh offline parse, toggle a stat digit, set the name or ability text through the fork's
text engine with status words coloured from the chart, render only the changed region, delta-merge it onto the
official PNG and gate the result. Pure functions over an open PSDImage; a refusal is a Refused exception carrying
one of CODES, which the per-card driver turns into data. No Discord, network or config; the text extra
(psd_tools.composite.text) is imported only where text is rendered, so the module loads under stock psd-tools.
"""
from dataclasses import dataclass, field
from inspect import signature
from os.path import basename, getsize, join
from re import IGNORECASE, compile as re_compile, escape, fullmatch, sub
from struct import pack
from time import perf_counter
from typing import Optional

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from psd_tools import PSDImage

from commands.card_edits import CardEdits
from commands.card_models import StatsDatabase
from commands.psd_parser import DRAWN_LAYER_KINDS, CardValidator, IconVocabulary, PSDParser
from commands.psd_regions import _fx_pad, badge_layers, clip, overlay_words, stat_rows

# ========================
# Configuration
# ========================
STATUS_CHART = "MDW/Croissant's Special Folder/Satus Effect Colors.psd"   # one text layer per status; its fill is the colour
FONTS_DIR = "MDW/Fonts"
CODES = ("duplicate", "unsupported-card", "stat>10", "recent", "blob", "parity", "styled-word", "font", "overflow",
         "bytes", "size", "region", "shape", "reparse", "idempotence", "edit-changed", "head-moved", "error")
STAT_FIELDS = ("hp", "def", "atk", "spd")
TEXT_FIELDS = ("name", "ability")
NOT_YET = ("stars", "types")                    # PSD-kind, refused in round 1 (section 3)
KIND = getattr(CardEdits, "KIND", {**{f: "psd" for f in STAT_FIELDS + TEXT_FIELDS + NOT_YET},
                                   "type": "folder", "subtype": "folder", "series": "folder"})
STAT_MAX = 10                                   # above it there is no canonical encoding (idea 13)
DIGITS = tuple(str(n) for n in range(10))
PARITY_THRESH = 24          # a pixel is bad past this |diff| (psd-research score_cards.py THRESH)
PARITY_BAD_MAX = 0.5        # % of bad pixels a region may show against the official PNG
PARITY_MEAN_MAX = 1.0       # mean |diff| over the region (Mini_Doomer hp row: 0.83, max 2)
SIZE_TOLERANCE = 0.2        # a saved PSD stays within 20% of the old blob
FILL_TOLERANCE = 2 / 255    # a status word's fill is the swatch within this per channel
OK_FONTS = ("exact", "system")                  # how a font may resolve; loose or substituted faces refuse
FONT_ALIASES = {"Papyrus-Regular": "Papyrus", "Kung-Fu-Panda": "KungFuPanda"}   # Windows names of two faces in MDW/Fonts
TOKEN = re_compile(r"\[[^\[\]\r\n\x03]{1,48}\]")   # an icon token in ability text ([fire]); psd_parser.ICON_SCAN_PATTERN
LETTER = re_compile(r"[A-Za-z]")                   # status_spans' word boundary
INK_ALPHA = 0.5             # a pixel is ink past this alpha (a title grazing a badge's anti-aliased edge is not an overlap)
ALLOWED_RECORD_FIELDS = {"stat": {"flags"}, "text": {"rect", "channels", "blocks"}}   # what an edit may change per record


# ========================
# Data
# ========================
@dataclass
class Refusal:
    """One refusal: a code from CODES and why."""
    code: str
    why: str


class Refused(Exception):
    """A refusal raised inside a card's steps; the driver turns it into a Refusal (one bad card never blocks another)."""
    def __init__(self, code: str, why: str):
        super().__init__(f"{code}: {why}")
        self.code = code
        self.why = why

    @property
    def refusal(self) -> Refusal:
        return Refusal(self.code, self.why)


@dataclass
class Classification:
    """A card's edits sorted by what the pipeline does with them."""
    jobs: dict = field(default_factory=dict)        # field -> value to bake into the PSD
    redundant: list = field(default_factory=list)   # PSD-kind fields equal to the PSD: dropped, no pixels (name stays)
    refusals: list = field(default_factory=list)    # Refusal per field the pipeline won't bake
    other: list = field(default_factory=list)       # folder and database-only fields: never touched


@dataclass
class StatusColour:
    """A status effect's colour from the chart."""
    name: str
    fill: tuple             # RGB, 0..1
    styled: bool            # an enabled effect on the chart layer (glow, gradient, shadow): refused in round 1


@dataclass
class Edit:
    """One applied field."""
    field: str
    old: object
    new: object
    layers: list            # layers whose records change: the toggled digits, or the text layer
    records: list           # their layer record indices (gate bytes)
    region: Optional[tuple]  # padded canvas bbox the edit can reach
    report: object = None   # the text engine's RenderReport (text edits)


@dataclass
class Baked:
    """A card's edits applied, rendered, merged and gated; the files are in the work dir."""
    jobs: dict
    edits: list
    region: tuple           # union of the edits' regions (the artifacts' crop)
    parity: dict            # field -> baseline measurement of its region: bad_pct, mean, max
    changed_pixels: int     # pixels the delta merge replaced
    records: list           # (record index, layer name, changed fields) between the old and new PSD
    flag_bytes: int         # differing flag bytes over those records (a toggle: 2)
    psd_path: str
    png_path: str
    reparse: dict           # parse of the new PSD (parse_card)
    timings: dict           # seconds: open, baseline, render, save, reparse, idempotence


# ========================
# Helper Functions
# ========================
def _union(boxes):
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def _pad(box, reach: int, psd: PSDImage) -> tuple:
    return clip((box[0] - reach, box[1] - reach, box[2] + reach, box[3] + reach), psd.width, psd.height)


def _alpha(layer) -> Optional[np.ndarray]:
    """The layer's alpha (H, W) in 0..1 over its bbox; None when the layer has no pixels to read (vectors)."""
    try:
        shape = layer.numpy("shape")
    except Exception:
        return None
    return None if shape is None or shape.ndim != 3 else shape[..., 0]


def _touches(a, b) -> bool:
    """Whether two layers' ink overlaps: solid alpha on both over the same canvas pixel (bboxes alone are loose)."""
    ax0, ay0, ax1, ay1 = a.bbox
    bx0, by0, bx1, by1 = b.bbox
    x0, y0, x1, y1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return False
    ink_a, ink_b = _alpha(a), _alpha(b)
    solid_a = True if ink_a is None else ink_a[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0] > INK_ALPHA
    solid_b = True if ink_b is None else ink_b[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0] > INK_ALPHA
    return bool(np.any(solid_a & solid_b))


def _flat(image: Image.Image) -> np.ndarray:
    """uint8 (H, W, 4): RGB composited over white, alpha kept (RGB under alpha 0 is arbitrary)."""
    a = np.asarray(image.convert("RGBA"), np.float32)
    k = a[..., 3:] / 255
    a[..., :3] = a[..., :3] * k + 255 * (1 - k)
    return np.rint(a).astype(np.uint8)


def _measure(a: Image.Image, b: Image.Image) -> dict:
    """bad_pct, mean and max of |a - b| (over white, alpha included)."""
    diff = np.abs(_flat(a).astype(np.int16) - _flat(b).astype(np.int16)).max(axis=2)
    return {"bad_pct": float((diff > PARITY_THRESH).mean() * 100), "mean": float(diff.mean()), "max": int(diff.max())}


def _within_bar(measured: dict) -> bool:
    return measured["bad_pct"] <= PARITY_BAD_MAX and measured["mean"] <= PARITY_MEAN_MAX


def _record_index(layer) -> int:
    records = layer._psd._record.layer_and_mask_information.layer_info.layer_records
    return next(i for i, record in enumerate(records) if record is layer._record)


def _enabled_effects(layer) -> list:
    """Names of the layer's effects whose own enabled flag is set (the block flag alone is not enough, section 5)."""
    try:
        return [type(e).__name__ for e in layer.effects if e.enabled]
    except Exception:
        return []


def _text_group(psd: PSDImage):
    return next((layer for layer in psd if layer.is_group() and layer.name.strip() == "Text"), None)


def _is_text_layer(layer) -> bool:
    return layer.kind == "type"


def collapse(text: Optional[str]) -> str:
    """Whitespace collapsed to single spaces (how the parser reads titles)."""
    return " ".join((text or "").split())


def normalise_ability(text: Optional[str]) -> str:
    """
    The parser's own whitespace rule, so an edit and a re-parse compare alike.

    Args:
        text: Ability text from the website (\\n) or a layer (\\r, \\x03)

    Returns:
        What the parser returns for a layer holding the text (psd_parser._extract_text_layer, _process_abilities):
        breaks as \\n, tabs as spaces, no space before punctuation, quotes and blank ends stripped
    """
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ").replace("\x03", "\n")
    text = text.replace("﻿", "").rstrip()
    text = text.strip("'\"\n").rstrip(" '\"")
    if text:
        text = sub(r"\s+([:;,\.\?!])", r"\1", text)
    return text.strip(" '\"\n")


# ========================
# Parsing (the goldens' route, without psd_analyzer)
# ========================
def offline_parser(all_types: list, stars: list, manifest: Optional[dict]) -> PSDParser:
    """
    A PSDParser on an empty database: a fresh parse with no author or series preservation.

    Args:
        all_types: Types/ icon names (lowercase stems)
        stars: Types/Stars/ icon names
        manifest: The stoogedings manifest, or None

    Returns:
        PSDParser
    """
    db = StatsDatabase()
    db.all_types = list(all_types)
    return PSDParser(db.all_types, db, IconVocabulary.build(db.all_types, stars, manifest))


def parse_card(parser: PSDParser, path: str, rel: str) -> dict:
    """
    Everything a gate compares: to_dict(), the hidden titles, the validator's problems and the CardInfo itself.

    Args:
        parser: offline_parser()
        path: The PSD on disk
        rel: Its repo path (the classification comes from the folders)

    Returns:
        {"card", "alt_names", "validator", "info"}
    """
    card = parser.parse(path, rel)
    return {"card": card.to_dict(), "alt_names": card.alt_names, "validator": CardValidator.validate(card), "info": card}


# ========================
# Layers
# ========================
def name_layer(psd: PSDImage):
    """
    The card's title layer: Text/Name, a visible text layer.

    Returns:
        The layer; raises Refused("unsupported-card") for the card-named, hidden and rasterized titles
    """
    group = _text_group(psd)
    if group is None:
        raise Refused("unsupported-card", "no Text group")
    layer = next((child for child in group if child.name.strip() == "Name"), None)
    if layer is None:
        titles = [child.name for child in group if _is_text_layer(child) and child.name.strip().lower() != "ability"
                  and child.bbox[1] < psd.height * 0.15]
        raise Refused("unsupported-card", f"title layer named {titles[0]!r}, not Name" if titles else "no Text/Name layer")
    if not _is_text_layer(layer):
        raise Refused("unsupported-card", "rasterized name layer")
    if not layer.is_visible():
        raise Refused("unsupported-card", "hidden name layer")
    return layer


def read_name(psd: PSDImage) -> Optional[str]:
    """The printed title, whitespace collapsed; None when the card has no readable Text/Name."""
    try:
        return collapse(name_layer(psd).text)
    except Refused:
        return None


def ability_layer(psd: PSDImage):
    """
    The single visible Ability text layer directly under Text.

    Returns:
        The layer; raises Refused("unsupported-card") for hidden, several or stray Ability layers
    """
    visible = [layer for layer in psd.descendants()
               if _is_text_layer(layer) and layer.is_visible() and layer.name.strip().lower() == "ability"]
    if len(visible) != 1:
        raise Refused("unsupported-card", f"{len(visible)} visible Ability layers")
    layer = visible[0]
    if layer.parent is not _text_group(psd):
        raise Refused("unsupported-card", "Ability layer outside the Text group")
    return layer


def uses_icon_font(layer) -> bool:
    """Whether any run of the text layer is set in the stoogedings icon font."""
    try:
        fonts = layer.resource_dict["FontSet"]
        for run in layer.engine_dict["StyleRun"]["RunArray"]:
            name = str(fonts[int(run["StyleSheet"]["StyleSheetData"]["Font"])]["Name"])
            if name.lower().startswith("stoogedings"):
                return True
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    return False


def run_fills(layer) -> list:
    """[(start, end, RGB 0..1 or None)] over the layer's text, str indices."""
    out = []
    for paragraph in layer.typesetting:
        for run in paragraph.runs:
            fill = run.style.fill_color
            out.append((run.start, run.end, tuple(float(v) for v in fill)[-3:] if fill else None))
    return out


# ========================
# Status chart
# ========================
def status_fills(chart: PSDImage) -> dict:
    """
    The status colours: one text layer per status in the chart's Buffs and Debuffs groups.

    Args:
        chart: The open STATUS_CHART

    Returns:
        {status name lowercased: StatusColour}; fills are the engine's 0..1 values
    """
    out = {}
    for group in chart:
        if not group.is_group() or group.name.strip() not in ("Buffs", "Debuffs"):
            continue
        for layer in group:
            if not _is_text_layer(layer):
                continue
            name = collapse(layer.text)
            fills = [fill for _, _, fill in run_fills(layer) if fill]
            if name and fills:
                out[name.lower()] = StatusColour(name, fills[0], bool(_enabled_effects(layer)))
    return out


def status_spans(text: str, chart: dict) -> list:
    """
    Every status name in a text, longest names first so "Frozen In Time" is not read as "Frozen".

    Args:
        text: Ability text
        chart: status_fills()

    Returns:
        [(start, end, StatusColour)] sorted by start, non-overlapping
    """
    taken = []
    for key in sorted(chart, key=len, reverse=True):
        words = r"\s+".join(escape(word) for word in chart[key].name.split())
        pattern = re_compile(r"(?<![A-Za-z])" + words + r"(?![A-Za-z])", IGNORECASE)
        for match in pattern.finditer(text):
            if not any(match.start() < end and start < match.end() for start, end, _ in taken):
                taken.append((match.start(), match.end(), chart[key]))
    return sorted(taken, key=lambda span: span[0])


def body_fill(layer, chart: dict) -> Optional[tuple]:
    """
    The text's own colour: the fill covering most characters outside the status words.

    Args:
        layer: A text layer
        chart: status_fills()

    Returns:
        RGB 0..1, or None when the layer has no filled run
    """
    spans = status_spans(layer.text, chart)
    weight = {}
    for start, end, fill in run_fills(layer):
        if fill is not None:
            inside = sum(max(0, min(end, e) - max(start, s)) for s, e, _ in spans)
            weight[fill] = weight.get(fill, 0) + (end - start) - inside
    return max(weight, key=weight.get) if weight else None


def changed_range(old: str, new: str) -> tuple:
    """
    The str indices [start, end) of the new text that set_text(keep="runs") restyles: everything outside the
    common prefix and suffix (the engine's own comparison, over the stored text with its final \\r).

    Args:
        old: The layer's text
        new: The text to set

    Returns:
        (start, end), start == end when text was only removed
    """
    old, new = old + "\r", new + "\r"
    prefix = 0
    while prefix < min(len(old), len(new)) and old[prefix] == new[prefix]:
        prefix += 1
    suffix = 0
    while suffix < min(len(old), len(new)) - prefix and old[len(old) - 1 - suffix] == new[len(new) - 1 - suffix]:
        suffix += 1
    return prefix, len(new) - suffix


def _word_bounds(text: str, start: int, end: int) -> tuple:
    """[start, end) widened to whole words, so "Poison" -> "Poisoned" restyles the whole word."""
    while start > 0 and LETTER.match(text[start - 1]):
        start -= 1
    while end < len(text) and LETTER.match(text[end]):
        end += 1
    return start, end


def _off_swatch(layer, chart: dict) -> Optional[str]:
    """A status word in the layer's text whose fill is not its swatch (a deliberate colour), else None."""
    fills = run_fills(layer)
    for start, end, colour in status_spans(layer.text, chart):
        for run_start, run_end, fill in fills:
            if run_start <= start < run_end and fill is not None:
                if any(abs(a - b) > FILL_TOLERANCE for a, b in zip(fill, colour.fill)):
                    return colour.name
    return None


# ========================
# Classify (section 3)
# ========================
def classify(parsed: dict, edits: dict, psd: Optional[PSDImage] = None, excluded: bool = False) -> Classification:
    """
    Sort a card's edits: PSD-kind fields whose value differs from the fresh parse are jobs, equal ones redundant.

    Args:
        parsed: parse_card() of the card's PSD
        edits: The card's fields from CardEdits.parse (no _keys)
        psd: The open PSD, needed to read Text/Name for a name edit
        excluded: The card is on CardValidator.EXCESSIVE_STAT_EXCLUSIONS (stats refuse as stat>10)

    Returns:
        Classification
    """
    card = parsed["card"]
    out = Classification()
    for name, value in edits.items():
        if KIND.get(name, "database") != "psd" or value is None:
            out.other.append(name)
        elif name in NOT_YET:
            out.refusals.append(Refusal("unsupported-card", f"{name}: not yet"))
        elif name in STAT_FIELDS:
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} {value!r}: not a stat")     # CardEdits.valid never lets one through
            if value > STAT_MAX:
                out.refusals.append(Refusal("stat>10", f"{name} {value}: not yet"))
            elif excluded:
                out.refusals.append(Refusal("stat>10", f"{name}: the card is on EXCESSIVE_STAT_EXCLUSIONS"))
            elif card.get(name) == value:
                out.redundant.append(name)
            else:
                out.jobs[name] = value
        elif name == "ability":
            if normalise_ability(card.get("ability")) == normalise_ability(value):
                out.redundant.append(name)
            else:
                out.jobs[name] = value
        elif name == "name":
            title = read_name(psd) if psd is not None else None
            if title is None:
                out.refusals.append(Refusal("unsupported-card", "no readable Text/Name layer"))
            elif title == collapse(value):
                out.redundant.append(name)
            else:
                out.jobs[name] = value
    return out


# ========================
# Apply (section 5 step 4)
# ========================
def template_row(psd: PSDImage, stat: str):
    """
    The template bar row of a stat: one "<Stat> Dark" group holding the ten masked exposure digits 0-9.

    Returns:
        (group, {digit: layer}, visible digits); raises Refused("unsupported-card") for any other row shape
    """
    rows = stat_rows(psd)[stat]
    if len(rows) != 1:
        raise Refused("unsupported-card", f"{stat}: {len(rows)} bar rows")
    group, visible, _ = rows[0]
    digits = {child.name.strip(): child for child in group if not child.is_group()}
    template = (set(digits) == set(DIGITS) and len(group) == len(DIGITS)
                and fullmatch(rf"\s*{stat}\s*dark\s*", group.name, IGNORECASE) is not None
                and all(layer.kind not in DRAWN_LAYER_KINDS and layer.has_mask() for layer in digits.values()))
    if not template:
        raise Refused("unsupported-card", f"{stat}: non-template row {group.name!r}")
    return group, digits, visible


def set_stat(psd: PSDImage, parsed: dict, stat: str, value: int) -> Edit:
    """
    Toggle a stat: the digit str(value) visible in "<Stat> Dark", the other nine hidden, 10 = none.

    Args:
        psd: The open card (edited in place)
        parsed: parse_card() of the card before the edit
        stat: hp, def, atk or spd
        value: 0-10

    Returns:
        Edit over the toggled digits; raises Refused for non-template rows (unsupported-card), values above
        10 (stat>10) and a digit that carries an enabled effect (parity)
    """
    group, digits, visible = template_row(psd, stat)
    old = min(visible) if visible else STAT_MAX
    if parsed["card"].get(stat) != old:
        raise Refused("unsupported-card", f"{stat}: the parser reads {parsed['card'].get(stat)}, the row shows {old}")
    if value < 0:
        raise ValueError(f"{stat} {value}: not a stat")
    if value > STAT_MAX:
        raise Refused("stat>10", f"{stat} {value}: not yet")
    show = digits[str(value)] if value < STAT_MAX else None
    hide = [digits[d] for d in DIGITS if digits[d].visible and digits[d] is not show]
    for layer in ([show] if show is not None else []) + hide:
        effects = _enabled_effects(layer)
        if effects:
            raise Refused("parity", f"{stat}: digit {layer.name} carries an enabled {', '.join(effects)}")
    changed = []
    if show is not None and not show.visible:
        show.visible = True
        changed.append(show)
    for layer in hide:
        layer.visible = False
        changed.append(layer)
    region = _union([tuple(layer.mask.bbox) for layer in changed])
    return Edit(stat, old, value, changed, [_record_index(layer) for layer in changed],
                _pad(region, _fx_pad(group), psd) if region else None)


def set_text_layer(psd: PSDImage, which: str, text: str, fonts_dir: str, chart: Optional[dict] = None) -> Edit:
    """
    Replace the name or the ability text, keeping the runs, and rasterize it with the text engine.

    Args:
        psd: The open card (edited in place)
        which: "name" (Text/Name) or "ability" (the single visible Text/Ability)
        text: The new text; newlines become paragraph breaks
        fonts_dir: Directory of the card fonts (MDW/Fonts)
        chart: status_fills(); ability status words take their swatch fill

    Returns:
        Edit with the engine's report; raises Refused: unsupported-card, styled-word, font, overflow

    The engine gives replaced or inserted text the style of the run before it, so an edit after a coloured status
    word would inherit its colour: the edited words are reset to the body fill, then the status words among them
    take their swatch. Status words the edit leaves alone keep their runs.
    """
    from psd_tools.composite.text import FontMap, MissingFontError
    layer = name_layer(psd) if which == "name" else ability_layer(psd)
    text = text.replace("\r\n", "\n").rstrip("\n").replace("\n", "\r")     # a final \n would add an empty paragraph
    old_text, old_bbox = layer.text, tuple(layer.bbox)
    start, end = changed_range(old_text, text)
    end = min(end, len(text))
    spans, body, badges = [], None, []
    if which == "name":     # point text has no box: the title must stay on the canvas and off the badges it clears today
        stars, icons, _ = badge_layers(psd)
        badges = [badge for badge in stars + icons if not _touches(layer, badge)]
    if which == "ability":
        if TOKEN.search(text) and not uses_icon_font(layer):
            raise Refused("unsupported-card", "[type] tokens on a card with image icons")
        if overlay_words(psd) or any(" FX: " in layer.name for layer in psd.descendants()):
            raise Refused("unsupported-card", "status word layers over the ability")
        if chart:
            word = _off_swatch(layer, chart)
            if word:
                raise Refused("styled-word", f"{word}: off-swatch colour on the card")
            body = body_fill(layer, chart)
            start, end = _word_bounds(text, start, end)
            spans = [span for span in status_spans(text, chart) if span[0] < end and start < span[1]]
            styled = [colour.name for _, _, colour in spans if colour.styled]
            if styled:
                raise Refused("styled-word", f"{', '.join(sorted(set(styled)))}: styled on the chart")
    layer.set_text(text, keep="runs")
    if body is not None and start < end:
        layer.set_span_style(start, end, fill=body)
    for span_start, span_end, colour in spans:
        layer.set_span_style(span_start, span_end, fill=colour.fill)
    try:
        report = layer.rasterize(fonts=FontMap(dirs=[fonts_dir], aliases=FONT_ALIASES))
    except MissingFontError as error:
        raise Refused("font", str(error))
    loose = sorted(name for name, (how, _) in report.fonts.items() if how not in OK_FONTS)
    if loose:
        raise Refused("font", f"not resolved exactly: {', '.join(loose)}")
    if "overflow" in report.warnings:
        raise Refused("overflow", f"{which} does not fit")
    bbox = tuple(layer.bbox)
    if clip(bbox, psd.width, psd.height) != bbox:
        raise Refused("overflow", f"{which} runs off the canvas")
    if any(_touches(layer, badge) for badge in badges):
        raise Refused("overflow", "name reaches the stars or type icons")
    region = _pad(_union([old_bbox, bbox]), _fx_pad(layer), psd)
    return Edit(which, old_text, text, [layer], [_record_index(layer)], region, report)


# ========================
# Render and merge (section 6)
# ========================
def changed_region(edits: list, psd: PSDImage) -> tuple:
    """The union of the edits' padded regions, clipped to the canvas (the artifacts' crop; gates run per edit)."""
    region = _union([edit.region for edit in edits])
    if region is None:
        raise Refused("idempotence", "no layer changed")
    return clip(region, psd.width, psd.height)


def render_region(psd: PSDImage, region: tuple) -> Image.Image:
    """psd.composite over a viewport, never from the stored preview and never forced (section 15)."""
    return psd.composite(viewport=region, ignore_preview=True)


def baseline(psd: PSDImage, png: Image.Image, region: tuple) -> tuple:
    """
    Render the unedited region and measure it against the official PNG (gate parity).

    Returns:
        (render, measurement); raises Refused("parity") when the fork already renders the region wrong
    """
    render = render_region(psd, region)
    measured = _measure(render, png.crop(region))
    if not _within_bar(measured):
        raise Refused("parity", "baseline bad %.2f%% mean %.2f max %d" % (measured["bad_pct"], measured["mean"], measured["max"]))
    return render, measured


def merge_png(png: Image.Image, base: Image.Image, render: Image.Image, region: tuple) -> tuple:
    """
    Delta-merge: inside the region, the official PNG takes the new render's pixels where it differs from the
    baseline render; every other pixel is left as Photoshop exported it.

    Returns:
        (merged image in the PNG's mode, pixels replaced)
    """
    if png.mode not in ("RGB", "RGBA"):
        raise Refused("shape", f"PNG mode {png.mode}")
    x0, y0, x1, y1 = region
    out = np.array(png.convert("RGBA"))
    before = np.asarray(base.convert("RGBA"))
    after = np.asarray(render.convert("RGBA"))
    changed = (before != after).any(axis=2)
    patch = out[y0:y1, x0:x1]
    patch[changed] = after[changed]
    return Image.fromarray(out).convert(png.mode), int(changed.sum())


def artifacts(png: Image.Image, merged: Image.Image, region: tuple) -> dict:
    """Before, after and an amplified diff crop of the region (the --dry-run artifacts)."""
    before, after = png.crop(region), merged.crop(region)
    diff = np.abs(_flat(before).astype(np.int16) - _flat(after).astype(np.int16))[..., :3].max(axis=2)
    return {"before": before, "after": after, "diff": Image.fromarray(np.clip(diff * 8, 0, 255).astype(np.uint8))}


# ========================
# Gates (sections 5 and 10)
# ========================
def gate_shape(png: Image.Image, merged: Image.Image, psd: PSDImage) -> None:
    if png.size != (psd.width, psd.height):
        raise Refused("shape", f"PNG {png.size} is not the PSD canvas {(psd.width, psd.height)}")
    if merged.size != png.size or merged.mode != png.mode:
        raise Refused("shape", "PNG size or mode changed")


def gate_region(png: Image.Image, merged: Image.Image, region: tuple) -> int:
    """Pixels differing between the old and new PNG lie inside the region. Returns how many differ."""
    diff = (np.asarray(png.convert("RGBA")) != np.asarray(merged.convert("RGBA"))).any(axis=2)
    inside = np.zeros_like(diff)
    inside[region[1]:region[3], region[0]:region[2]] = True
    outside = int((diff & ~inside).sum())
    if outside:
        raise Refused("region", f"{outside} pixels differ outside the region")
    return int(diff.sum())


def gate_size(old_path: str, new_path: str) -> None:
    old, new = getsize(old_path), getsize(new_path)
    if abs(new - old) > SIZE_TOLERANCE * old:
        raise Refused("size", f"PSD {old} -> {new} bytes")


def layer_records(path: str) -> list:
    """Each layer record's identity-free fingerprint: name, flags, rect, channels, mask, blend and tagged blocks."""
    psd = PSDImage.open(path)
    out = []
    for record in psd._record.layer_and_mask_information.layer_info.layer_records:
        out.append({"name": record.name, "flags": record.flags.tobytes(),
                    "rect": (record.top, record.left, record.bottom, record.right),
                    "channels": tuple((int(info.id), info.length) for info in record.channel_info),
                    "mask": record.mask_data.tobytes() if record.mask_data is not None else b"",
                    "blend": (record.blend_mode, record.opacity, record.clipping),
                    "blocks": record.tagged_blocks.tobytes()})
    return out


def gate_bytes(old_path: str, new_path: str, edits: list) -> tuple:
    """
    Compare the layer records of the old and new PSD: only the edited layers change, and only in the fields
    their kind of edit touches (a toggle: exactly one flag byte per toggled digit; a text edit: rect, channels,
    EngineData).

    Returns:
        ([(index, name, changed fields)], differing flag bytes)
    """
    old, new = layer_records(old_path), layer_records(new_path)
    if len(old) != len(new):
        raise Refused("bytes", f"layer count {len(old)} -> {len(new)}")
    allowed = {}
    for edit in edits:
        for index in edit.records:
            allowed[index] = ALLOWED_RECORD_FIELDS["stat" if edit.field in STAT_FIELDS else "text"]
    changed, flag_bytes = [], 0
    for index, (a, b) in enumerate(zip(old, new)):
        fields = sorted(key for key in a if a[key] != b[key])
        if not fields:
            continue
        if not set(fields) <= allowed.get(index, set()):
            raise Refused("bytes", f"layer {index} {a['name']!r} changed {', '.join(fields)}")
        differing = sum(1 for x, y in zip(a["flags"], b["flags"]) if x != y)
        if allowed[index] == ALLOWED_RECORD_FIELDS["stat"] and differing != 1:
            raise Refused("bytes", f"digit {a['name']!r} changed {differing} flag bytes")
        flag_bytes += differing
        changed.append((index, a["name"], fields))
    toggled = {index for edit in edits if edit.field in STAT_FIELDS for index in edit.records}
    if toggled - {index for index, _, _ in changed}:
        raise Refused("bytes", "a toggled digit did not change")
    return changed, flag_bytes


def gate_reparse(parser: PSDParser, parsed: dict, new_path: str, rel: str, jobs: dict, psd: PSDImage) -> dict:
    """
    Re-parse the new PSD: the old parse with exactly the edited fields replaced, no new problems, one digit per
    edited row and the title read back for a name edit.

    Returns:
        parse_card() of the new PSD
    """
    for stat in jobs:
        if stat in STAT_FIELDS:
            visible = template_row(psd, stat)[2]
            if visible != ([jobs[stat]] if jobs[stat] < STAT_MAX else []):
                raise Refused("reparse", f"{stat}: the row shows {visible}")
    new = parse_card(parser, new_path, rel)
    expected, got = dict(parsed["card"]), dict(new["card"])
    for name, value in jobs.items():
        if name in STAT_FIELDS:
            expected[name] = value
        elif name == "ability":
            expected["ability"], got["ability"] = normalise_ability(value), normalise_ability(got.get("ability"))
    expected.pop("problem", None)
    got.pop("problem", None)
    different = sorted(key for key in set(expected) | set(got) if expected.get(key) != got.get(key))
    if different:
        raise Refused("reparse", "fields differ from the expected parse: " + ", ".join(different))
    if new["alt_names"] != parsed["alt_names"]:
        raise Refused("reparse", "hidden titles changed")
    answered = tuple(p for name in jobs for p in CardEdits.FIXED_BY.get(name, ()))
    old_problems = {p for p in parsed["validator"] if not (answered and p.startswith(answered))}
    added = sorted(set(new["validator"]) - old_problems)
    if added:
        raise Refused("reparse", "new problems: " + "; ".join(added))
    if any(p.startswith("EFFECT LAYER") or p.startswith("STATS TOO HIGH") for p in new["validator"]):
        raise Refused("reparse", "effect layer or stats problems remain")
    if "name" in jobs and read_name(psd) != collapse(jobs["name"]):
        raise Refused("reparse", f"Text/Name reads {read_name(psd)!r}")
    return new


def gate_idempotence(new_parsed: dict, jobs: dict, psd: PSDImage, merged: Image.Image, edits: list) -> dict:
    """
    Nothing left to bake on the new PSD, and each edit's region re-rendered from it matches the merged PNG within
    the parity bar (not byte-equal: the delta merge keeps Photoshop's pixels where the render matched the baseline).

    Returns:
        {field: region measurement}
    """
    again = classify(new_parsed, jobs, psd)
    if again.jobs:
        raise Refused("idempotence", "still to bake: " + ", ".join(sorted(again.jobs)))
    out = {}
    for edit in edits:
        measured = _measure(render_region(psd, edit.region), merged.crop(edit.region))
        if not _within_bar(measured):
            raise Refused("idempotence", "%s re-render bad %.2f%% mean %.2f" % (edit.field, measured["bad_pct"], measured["mean"]))
        out[edit.field] = measured
    return out


# ========================
# Save
# ========================
def save_psd(psd: PSDImage, path: str, update_preview: bool = True) -> None:
    """
    Write the PSD. With update_preview the fork regenerates the flattened preview of an edited document (20-45 s);
    without it the stale preview is kept (0.2 s; the jetson later, question 2).
    """
    if update_preview:
        psd.save(path)
    elif "update_preview" in signature(psd.save).parameters:
        psd.save(path, update_preview=False)
    else:                                       # older fork: the private flag is what save() checks
        updated = psd._updated
        psd._updated = False
        try:
            psd.save(path)
        finally:
            psd._updated = updated


def save_png(image: Image.Image, path: str, like: Image.Image) -> None:
    """
    Write the merged PNG with the official PNG's colour chunks (iCCP, gAMA, cHRM, pHYs): only pixels change.

    Args:
        image: The merged image
        path: Where it goes
        like: The official PNG as opened (its info holds the chunks Photoshop wrote)
    """
    chunks = PngInfo()
    if like.info.get("gamma") is not None:
        chunks.add(b"gAMA", pack(">I", int(round(like.info["gamma"] * 100000))))
    if like.info.get("chromaticity"):
        chunks.add(b"cHRM", b"".join(pack(">I", int(round(v * 100000))) for v in like.info["chromaticity"]))
    keep = {key: like.info[key] for key in ("icc_profile", "dpi") if like.info.get(key)}
    image.save(path, "PNG", pnginfo=chunks, **keep)


# ========================
# Per-card driver (section 5 steps 3-9)
# ========================
def apply_jobs(psd_path: str, png_path: str, rel: str, jobs: dict, parsed: dict, parser: PSDParser, out_dir: str,
               fonts_dir: str, chart: Optional[dict] = None, update_preview: bool = True) -> Baked:
    """
    Apply a card's jobs, render the changed region, delta-merge the PNG, save and run every gate.

    Args:
        psd_path: The card's PSD (never written)
        png_path: Its official PNG (never written)
        rel: The card's repo path
        jobs: Classification.jobs
        parsed: parse_card() of psd_path
        parser: offline_parser()
        out_dir: Where the new PSD and PNG go (same file names)
        fonts_dir: MDW/Fonts
        chart: status_fills() for ability edits
        update_preview: save_psd()

    Returns:
        Baked; raises Refused (any other exception is the driver's "error")
    """
    timings = {}
    start = perf_counter()
    old, new = PSDImage.open(psd_path), PSDImage.open(psd_path)
    png = Image.open(png_path)
    png.load()
    timings["open"] = perf_counter() - start
    if png.size != (new.width, new.height) or png.mode not in ("RGB", "RGBA"):
        raise Refused("shape", f"PNG {png.mode} {png.size} against a {new.width}x{new.height} canvas")

    edits = []
    for name, value in jobs.items():
        if name in STAT_FIELDS:
            edits.append(set_stat(new, parsed, name, value))
        elif name in TEXT_FIELDS:
            edits.append(set_text_layer(new, name, value, fonts_dir, chart))
        else:
            raise Refused("unsupported-card", f"{name}: not yet")
    edits = [edit for edit in edits if edit.layers]
    region = changed_region(edits, new)

    # region per edited field (section 5 step 3): the parity bar is measured over the pixels that edit can reach
    merged, parity, changed_pixels = png, {}, 0
    timings["baseline"] = timings["render"] = 0.0
    for edit in edits:
        start = perf_counter()
        base, parity[edit.field] = baseline(old, png, edit.region)
        timings["baseline"] += perf_counter() - start
        start = perf_counter()
        render = render_region(new, edit.region)
        timings["render"] += perf_counter() - start
        merged, replaced = merge_png(merged, base, render, edit.region)
        changed_pixels += replaced
    gate_shape(png, merged, new)
    gate_region(png, merged, region)     # merge_png writes only inside the regions: the gate restates its contract

    new_psd, new_png = join(out_dir, basename(psd_path)), join(out_dir, basename(png_path))
    start = perf_counter()
    save_psd(new, new_psd, update_preview)
    timings["save"] = perf_counter() - start
    save_png(merged, new_png, png)
    gate_size(psd_path, new_psd)
    records, flag_bytes = gate_bytes(psd_path, new_psd, edits)

    start = perf_counter()
    reopened = PSDImage.open(new_psd)
    reparse = gate_reparse(parser, parsed, new_psd, rel, jobs, reopened)
    timings["reparse"] = perf_counter() - start
    start = perf_counter()
    gate_idempotence(reparse, jobs, reopened, merged, edits)
    timings["idempotence"] = perf_counter() - start
    return Baked(jobs, edits, region, parity, changed_pixels, records, flag_bytes, new_psd, new_png, reparse, timings)


# ========================
# Module Exports
# ========================
__all__ = [
    'Baked',
    'CODES',
    'Classification',
    'Edit',
    'FONTS_DIR',
    'Refusal',
    'Refused',
    'STATUS_CHART',
    'StatusColour',
    'ability_layer',
    'apply_jobs',
    'artifacts',
    'baseline',
    'body_fill',
    'changed_range',
    'changed_region',
    'classify',
    'collapse',
    'gate_bytes',
    'gate_idempotence',
    'gate_region',
    'gate_reparse',
    'gate_shape',
    'gate_size',
    'layer_records',
    'merge_png',
    'name_layer',
    'normalise_ability',
    'offline_parser',
    'parse_card',
    'read_name',
    'render_region',
    'save_png',
    'save_psd',
    'set_stat',
    'set_text_layer',
    'status_fills',
    'status_spans',
    'template_row',
]
