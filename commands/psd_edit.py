"""
Edits to a card PSD for the edit pipeline (docs/EDIT_PIPELINE_DESIGN.md sections 3, 5, 6, 7 and 10): classify a card's
website edits against a fresh offline parse, toggle a stat digit, set the name or ability text through the fork's
text engine with status words colored from the chart and the styled ones' FX layers cut from the new raster, render
only the changed region, delta-merge it onto the official PNG and gate the result. Pure functions over an open
PSDImage; a refusal is a Refused exception carrying one of CODES, which the per-card driver turns into data. No
Discord, network or config; the text extra (psd_tools.composite.text) is imported only where text is rendered, so the
module loads under stock psd-tools.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from inspect import signature
from io import BytesIO
from os.path import basename, getsize, join
from re import IGNORECASE, compile as re_compile, escape, fullmatch, sub
from struct import pack
from time import perf_counter
from typing import Optional

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from psd_tools import PSDImage
from psd_tools.api.pil_io import post_process
from psd_tools.constants import BlendMode, ColorMode, Resource, Tag

from commands.card_edits import CardEdits
from commands.card_models import StatsDatabase
from commands.psd_parser import (DRAWN_LAYER_KINDS, EFFECT_LAYER_TAG, ICON_FONT_PREFIX, CardValidator, IconVocabulary,
                                 PSDParser, title_text)
from commands.psd_regions import badge_layers, clip, fx_pad, overlay_words, stat_rows, union_box
from global_config import FONT_ALIASES

# ========================
# Configuration
# ========================
STATUS_CHART = "MDW/Croissant's Special Folder/Satus Effect Colors.psd"   # one text layer per status; its fill is the color
FONTS_DIR = "MDW/Fonts"
CODES = ("duplicate", "unsupported-card", "stat>10", "recent", "blob", "parity", "styled-word", "font", "overflow",
         "bytes", "size", "region", "shape", "reparse", "idempotence", "edit-changed", "head-moved", "capped", "error")
STAT_FIELDS = ("hp", "def", "atk", "spd")
TEXT_FIELDS = ("name", "ability")
NOT_YET = ("stars", "types")                    # PSD-kind, refused in round 1 (section 3)
KIND = CardEdits.KIND
STAT_MAX = 10                                   # one bar row shows 0-10; Top + Bottom rows show 11-20 (idea 13)
DIGITS = tuple(str(n) for n in range(10))
MARKER = "10"                                   # a two-row card's parser marker: "10" visible at opacity 0, never toggled
EFFECT_BLOCKS = (Tag.OBJECT_BASED_EFFECTS_LAYER_INFO, Tag.OBJECT_BASED_EFFECTS_LAYER_INFO_V0,
                 Tag.OBJECT_BASED_EFFECTS_LAYER_INFO_V1, Tag.EFFECTS_LAYER)   # a digit's effect blocks: lfx2 and its kin
PARITY_THRESH = 24          # a pixel is bad past this |diff| (psd-research score_cards.py THRESH)
PARITY_BAD_MAX = 0.5        # % of bad pixels a region may show against the official PNG
PARITY_MEAN_MAX = 2.0       # mean |diff| over the region (owner, 2026-10-22; Mini_Doomer hp row: 0.83, max 2)
ICC_MOVED = 4               # the ICC vote (score_cards.py): pixels the profile's conversion moves this far in a channel
ICC_RAW_VOTE = 0.65         # share of them nearer the unconverted values: the PNG was exported raw
ICC_MIN_PX = 1000           # fewer moved pixels: no evidence, converted
SIZE_TOLERANCE = 0.2        # a saved PSD stays within 20% of the old blob
OK_FONTS = ("exact", "system")                  # how a font may resolve; loose or substituted faces refuse
TOKEN = re_compile(r"\[[^\[\]\r\n\x03]{1,48}\]")   # an icon token in ability text ([fire]); psd_parser.ICON_SCAN_PATTERN
LETTER = re_compile(r"[A-Za-z]")                   # status_spans' word boundary
INK_ALPHA = 0.5             # a pixel is ink past this alpha (a title grazing a badge's anti-aliased edge is not an overlap)
FX_TAG = EFFECT_LAYER_TAG   # "Ability FX: Blazing": a status word's glyph pixels under the chart's effect (Effect Layers.jsx)
BADGE_FX = re_compile(r"[(\[][^()\[\]]+[)\]](?: \| [(\[][^()\[\]]+[)\]])*")   # "(1CD) | (3U)", "[fire]": badge and token runs
KEY_INK = 0.5 / 255         # the JSX's key copy, the word white over black, selects every pixel its red channel reaches (level 1)
ALLOWED_RECORD_FIELDS = {"stat": {"flags", "blocks"}, "text": {"rect", "channels", "blocks"},
                         "overlay": {"flags"}}  # what an edit may change per record (overlay: a hidden hand-made word layer)
STORY_KEYS = (("1", "1", "0", "0"), ("DocumentObjects", "TextObjects", "Model", "Text"))   # Txt2's stories and their
#   text: Photoshop CC numbers the keys, older versions name them
STORY_STYLE_KEYS = (     # per STORY_KEYS spelling: a story's style runs, a run's features, the FontSet, a paint's values
    {"runs": ("6", "0"), "data": "0", "length": "1", "features": ("0", "6"), "fonts": ("0", "1", "0"),
     "font": ("0", "0", "0"), "paint": ("0", "1")},
    {"runs": ("StyleRun", "RunArray"), "data": "RunData", "length": "Length", "features": ("StyleSheet", "Features"),
     "fonts": ("DocumentResources", "FontSet", "Resources"), "font": ("Resource", "Identifier", "Name"),
     "paint": ("Color", "Values")})
STORY_MIRROR = {"Font": 0, "FontSize": 1, "FauxBold": 2, "FauxItalic": 3, "AutoLeading": 4, "Leading": 5,
                "HorizontalScale": 6, "VerticalScale": 7, "Tracking": 8, "BaselineShift": 9, "FillColor": 53,
                "StrokeColor": 54}   # TySh StyleSheetData keys a story run holds too, with Photoshop CC's feature number


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
class StatusColor:
    """A status effect's color from the chart."""
    name: str
    fill: tuple             # RGB, 0..1
    styled: bool            # an enabled effect on the chart layer (glow, gradient, shadow): the word gets an FX layer
    blocks: dict = field(default_factory=dict)  # the chart layer's effect blocks by Tag (lfx2, lrFX): copied onto the FX layer


@dataclass
class TemplateRow:
    """A template bar row: the ten masked exposure digits of "<Stat> Dark", or of Top/Bottom under it (idea 13)."""
    group: object
    digits: dict            # "0".."9" -> layer
    marker: object = None   # a two-row card's "10", visible at opacity 0 so the parser reads a full row; never toggled

    @property
    def visible(self) -> list:
        """The visible digits, sorted (the marker is not one)."""
        return sorted(int(name) for name, layer in self.digits.items() if layer.is_visible())

    def reads(self) -> Optional[int]:
        """What the parser reads for the row: the lowest visible digit, the marker counting 10; None when none shows."""
        items = self.visible + ([STAT_MAX] if self.marker is not None and self.marker.is_visible() else [])
        return min(items) if items else None


@dataclass
class Edit:
    """One applied field."""
    field: str
    old: object
    new: object
    layers: list            # layers whose records change: the toggled digits, or the text layer
    records: list           # their layer record indices (gate bytes)
    region: Optional[tuple]  # padded canvas bbox the edit can reach
    moves: list = field(default_factory=list)   # (record from, record to or None, block key): effect blocks moved or cleared
    hidden: list = field(default_factory=list)  # record indices of the overlay word layers an ability edit hid
    fx: list = field(default_factory=list)      # (name, {block key: bytes}) per FX layer cut above Text, bottom to top
    dropped: list = field(default_factory=list)  # names of the FX layers deleted first (derived state: all of the text layer's)
    story: Optional[int] = None  # the text layer's TextIndex before the edit: the Txt2 story it may change


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
    block_bytes: int = 0    # effect block bytes that moved with a toggle (idea 12) or landed on an FX layer (idea 14)
    fx: list = field(default_factory=list)      # names of the FX layers regenerated
    icc: bool = True        # export_icc(): rendered converted by the embedded profile (False: the PNG was exported raw)


# ========================
# Helper Functions
# ========================
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


def text_engine() -> bool:
    """Is the fork's text engine (psd-tools[text]) importable: the renders and text edits need it."""
    try:
        import psd_tools.composite.text  # noqa: F401
    except ImportError:
        return False
    return True


def _record_index(layer) -> int:
    records = layer._psd._record.layer_and_mask_information.layer_info.layer_records
    return next(i for i, record in enumerate(records) if record is layer._record)


def _enabled_effects(layer, strict: bool = False) -> list:
    """
    Names of the layer's effects whose own enabled flag is set (the block flag alone is not enough, section 5).

    Args:
        layer: Any layer
        strict: An unreadable effect stack refuses (parity) instead of counting as none
    """
    try:
        return [type(e).__name__ for e in layer.effects if e.enabled]
    except Exception as error:                  # no effects block, or one psd-tools can't read
        if strict:
            raise Refused("parity", f"{layer.name}: effects unreadable: {error}") from error
        return []


def _text_group(psd: PSDImage):
    return next((layer for layer in psd if layer.is_group() and layer.name.strip() == "Text"), None)


def _is_text_layer(layer) -> bool:
    return layer.kind == "type"


def collapse(text: Optional[str]) -> str:
    """Whitespace collapsed to single spaces (how the parser reads titles: psd_parser.title_text)."""
    return title_text(text)


def normalize_ability(text: Optional[str]) -> str:
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


def story_sync() -> bool:
    """Does the fork's set_text also write the layer's Txt2 story (TypeLayer.sync_story): the text Photoshop reads."""
    from psd_tools.api.layers import TypeLayer
    return hasattr(TypeLayer, "sync_story")


def _txt2(psd: PSDImage):
    blocks = psd._record.layer_and_mask_information.tagged_blocks
    return blocks.get_data(Tag.TEXT_ENGINE_DATA) if blocks is not None and Tag.TEXT_ENGINE_DATA in blocks else None


def _story_list(root) -> tuple:
    """(Txt2 DocumentObjects, its TextObjects key, the stories, the Model and Text keys, STORY_KEYS index)."""
    for spelling, (objects_key, items_key, model, text) in enumerate(STORY_KEYS):
        objects = root.get(objects_key)
        items = objects.get(items_key) if hasattr(objects, "get") else None
        if items is not None:
            return objects, items_key, items, model, text, spelling
    raise Refused("unsupported-card", "a Txt2 block without text objects")


def stories(psd: PSDImage) -> Optional[list]:
    """
    The texts of the document's Txt2 stories by TextIndex, final \\r included; None without a Txt2 block.

    Photoshop reads a type layer's text from its story, not from the layer's own TySh, and rebuilds TySh from it on
    save. Read here, not through the fork, so the gates check the fork.
    """
    root = _txt2(psd)
    if root is None:
        return None
    _, _, items, model, text, _ = _story_list(root)
    return [str(story[model][text].value) for story in items]


def story_index(layer) -> Optional[int]:
    """The type layer's TextIndex: its story's index in Txt2."""
    index = layer._data.text_data.get(b"TextIndex")
    return None if index is None else int(getattr(index, "value", index))


def stale_stories(psd: PSDImage) -> list:
    """(layer name, story text, layer text) per type layer whose Txt2 story lacks its text: Photoshop shows the story's."""
    texts = stories(psd)
    if texts is None:
        return []
    out = []
    for layer in psd.descendants():
        index = story_index(layer) if _is_text_layer(layer) else None
        if index is None:
            continue
        own = str(layer.engine_dict["Editor"]["Text"].value)
        story = texts[index] if 0 <= index < len(texts) else None
        if story != own:
            out.append((layer.name, story, own))
    return out


def _dig(node, path):
    for key in path:
        node = node[key]
    return node


def _plain(value, fonts: list):
    """A style value comparable across TySh and Txt2: a font's name, a paint's ARGB, a rounded number."""
    if fonts is not None:
        return fonts[int(value)]
    value = getattr(value, "value", value)
    return round(float(value), 4) if isinstance(value, (int, float)) and not isinstance(value, bool) else value


def _style_spans(runs: list) -> list:
    """[(unit start, unit end, features)] from (length, features) runs."""
    out, unit = [], 0
    for length, features in runs:
        out.append((unit, unit + length, features))
        unit += length
    return out


def story_style_drift(psd: PSDImage) -> list:
    """
    (layer name, why) per type layer whose Txt2 story holds its text but other styles: per character, each
    STORY_MIRROR key both hold must agree (a font by name, a color by its values). Photoshop would show the story's
    styles, so a fill written to TySh alone would silently go back. Keys a story run leaves out (inherited) are skipped.
    """
    root = _txt2(psd)
    if root is None:
        return []
    _, _, items, model_key, text_key, spelling = _story_list(root)
    keys = STORY_STYLE_KEYS[spelling]
    story_fonts = [str(_dig(font, keys["font"]).value) for font in _dig(root, keys["fonts"])]
    out = []
    for layer in psd.descendants():
        index = story_index(layer) if _is_text_layer(layer) else None
        if index is None or not 0 <= index < len(items):
            continue
        model = items[index][model_key]
        if str(model[text_key].value) != str(layer.engine_dict["Editor"]["Text"].value):
            continue                            # stale_stories' to report
        style = layer.engine_dict["StyleRun"]
        tysh = _style_spans([(int(n), run["StyleSheet"]["StyleSheetData"])
                             for n, run in zip(style["RunLengthArray"], style["RunArray"])])
        story = _style_spans([(int(run[keys["length"]].value), _dig(run[keys["data"]], keys["features"]))
                              for run in _dig(model, keys["runs"])])
        tysh_fonts = [str(font["Name"].value) for font in layer.resource_dict["FontSet"]]
        cuts = sorted({u for a, b, _ in tysh + story for u in (a, b)})
        why = None
        for a, b in zip(cuts, cuts[1:]):
            mine = next((f for u0, u1, f in tysh if u0 <= a < u1), None)
            theirs = next((f for u0, u1, f in story if u0 <= a < u1), None)
            if mine is None or theirs is None:
                why = f"style runs end at {tysh[-1][1]} and {story[-1][1]}"
                break
            for name, number in STORY_MIRROR.items():
                feature = name if spelling else str(number)
                if name not in mine or feature not in theirs:
                    continue
                left, right = mine[name], theirs[feature]
                if name.endswith("Color"):
                    left, right = ([_plain(v, None) for v in _dig(x, path)] for x, path in
                                   ((left, ("Values",)), (right, keys["paint"])))
                else:
                    left = _plain(left, tysh_fonts if name == "Font" else None)
                    right = _plain(right, story_fonts if name == "Font" else None)
                if left != right:
                    why = f"{name} at {a}-{b}: layer {left!r}, story {right!r}"
                    break
            if why:
                break
        if why:
            out.append((layer.name, why))
    return out


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


def icon_font(layer) -> Optional[str]:
    """The PostScript name of the stoogedings icon font a run of the text layer is set in, or None."""
    try:
        fonts = layer.resource_dict["FontSet"]
        for run in layer.engine_dict["StyleRun"]["RunArray"]:
            name = str(fonts[int(run["StyleSheet"]["StyleSheetData"]["Font"])]["Name"].value)   # str() of the String: quoted
            if name.lower().startswith(ICON_FONT_PREFIX):
                return name
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        pass
    return None


def _overlaid(layer) -> bool:
    """
    Whether an opaque Normal overlay effect paints over the layer's own colours (Mini Doomer's gradient, its badges
    on FX layers above): then the engine's one-colour icon glyphs look like Photoshop's colour ones under it.
    """
    from psd_tools.api.effects import ColorOverlay, GradientOverlay, PatternOverlay
    effects = layer.effects
    return bool(effects.enabled) and any(
        isinstance(effect, (ColorOverlay, GradientOverlay, PatternOverlay)) and effect.enabled
        and float(effect.opacity) >= 100 and effect.blend_mode == BlendMode.NORMAL for effect in effects)


def uses_icon_font(layer) -> bool:
    """Whether any run of the text layer is set in the stoogedings icon font."""
    return icon_font(layer) is not None


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
    The status colors: one text layer per status in the chart's Buffs and Debuffs groups.

    Args:
        chart: The open STATUS_CHART

    Returns:
        {status name lowercased: StatusColor}; fills are the engine's 0..1 values
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
                blocks = {key: layer.tagged_blocks[key] for key in EFFECT_BLOCKS if key in layer.tagged_blocks}
                out[name.lower()] = StatusColor(name, fills[0], bool(_enabled_effects(layer)), blocks)
    return out


def status_spans(text: str, chart: dict) -> list:
    """
    Every status name in a text, longest names first so "Frozen In Time" is not read as "Frozen".

    Args:
        text: Ability text
        chart: status_fills()

    Returns:
        [(start, end, StatusColor)] sorted by start, non-overlapping
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
    The text's own color: the fill covering most characters outside the status words.

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


def keep_breaks(old: str, new: str, which: str) -> str:
    """
    The new text with the layer's own breaks back where the edit kept them: the website shows an ability's forced line
    break (\\x03) as a newline and a name's breaks as spaces, so setting its text as given would turn a soft break
    into a paragraph (Mini Doomer's "it. \\x03(BOT) Die.") or a two-line title into one line.

    Args:
        old: The layer's text
        new: The text to set, ability breaks as \\r
        which: "name" or "ability"

    Returns:
        new, each \\r (ability) or space (name) the diff pairs with an old break of those kinds made that break
    """
    kinds, shown = ("\r\x03", " ") if which == "name" else ("\x03", "\r")
    if not any(c in old for c in kinds):
        return new
    seen = "".join(shown if c in kinds else c for c in old)
    out = list(new)
    for a, b, size in SequenceMatcher(None, seen, new, autojunk=False).get_matching_blocks():
        for i in range(size):
            if old[a + i] in kinds:
                out[b + i] = old[a + i]
    return "".join(out)


def _word_bounds(text: str, start: int, end: int) -> tuple:
    """[start, end) widened to whole words, so "Poison" -> "Poisoned" restyles the whole word."""
    while start > 0 and LETTER.match(text[start - 1]):
        start -= 1
    while end < len(text) and LETTER.match(text[end]):
        end += 1
    return start, end


# ========================
# Classify (section 3)
# ========================
def shown_name(parsed: dict) -> str:
    """The name the bot shows for a parse: the printed title (display_name), else the card's name (file stem)."""
    info = parsed.get("info")
    return parsed["card"].get("display_name") or (info.name if info is not None else "")


def _two_rows(card: dict, psd: Optional[PSDImage], stat: str) -> bool:
    """Whether the stat has Top and Bottom rows (idea 13): read off the PSD, else inferred from a parse above 10."""
    if psd is None:
        return (card.get(stat) or 0) > STAT_MAX
    try:
        return len(template_rows(psd, stat)) == 2
    except Refused:
        return False


def classify(parsed: dict, edits: dict, psd: Optional[PSDImage] = None) -> Classification:
    """
    Sort a card's edits: PSD-kind fields whose value differs from the fresh parse are jobs, equal ones redundant.
    A stat above 10 is a job only on a two-row card (idea 13), above 20 never.

    Args:
        parsed: parse_card() of the card's PSD
        edits: The card's fields from CardEdits.parse (no _keys)
        psd: The open PSD, needed to check Text/Name is editable for a name edit and the rows for a stat above 10

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
            if value > 2 * STAT_MAX:
                out.refusals.append(Refusal("stat>10", f"{name} {value}: above 20"))
            elif value > STAT_MAX and not _two_rows(card, psd, name):
                out.refusals.append(Refusal("stat>10", f"{name} {value}: one bar row; a second row is standardization work "
                                            "(ledger L3)"))
            elif card.get(name) == value:
                out.redundant.append(name)
            else:
                out.jobs[name] = value
        elif name == "ability":
            if normalize_ability(card.get("ability")) == normalize_ability(value):
                out.redundant.append(name)
            else:
                out.jobs[name] = value
        elif name == "name":                    # redundant when the bot already shows it (the parser reads the title)
            if shown_name(parsed) == collapse(value):
                out.redundant.append(name)
            elif psd is None or read_name(psd) is None:
                out.refusals.append(Refusal("unsupported-card", "no readable Text/Name layer"))
            else:
                out.jobs[name] = value
    return out


# ========================
# FX layers (section 7 idea 14)
# ========================
def fx_name(base: str, status: str, nth: int = 0) -> str:
    """
    The name Effect Layers.jsx gives a status word's FX layer.

    Args:
        base: The text layer's name (Ability)
        status: The chart's status name
        nth: Which occurrence when the word repeats (1-based); 0 for a word that occurs once

    Returns:
        "Ability FX: Blazing", "Ability FX: Frozen#2"
    """
    return f"{base}{FX_TAG}{status}" + (f"#{nth}" if nth else "")


def fx_layers(psd: PSDImage, base) -> list:
    """The FX layers cut from a text layer ("<its name> FX: ..."), bottom to top, wherever they sit."""
    prefix = base.name + FX_TAG
    return [layer for layer in psd.descendants() if not layer.is_group() and layer.name.startswith(prefix)]


def fx_label(base, fx) -> str:
    """What an FX layer of base holds, without a repeat's number: "Ability FX: Frozen#2" -> "Frozen"."""
    return sub(r"#\d+$", "", fx.name[len(base.name + FX_TAG):])


def kept_fx(old_fx: list, base, chart: Optional[dict]) -> tuple:
    """
    The FX layers an ability edit cannot make again (it rebuilds only the chart's styled status words).

    Returns:
        (badge layers: badge and token runs, "Ability FX: (1CD) | (2U)" on queue A's gradient cards; other layers: words
        without a chart style), names
    """
    badges = [fx.name for fx in old_fx if BADGE_FX.fullmatch(fx_label(base, fx))]
    words = [fx.name for fx in old_fx if fx.name not in badges
             and not getattr((chart or {}).get(collapse(fx_label(base, fx)).lower()), "styled", False)]
    return (badges, words)


def _fx_engine() -> tuple:
    """The fork's text engine pieces that cut glyph pixels and make pixel layers; Refused("styled-word") without them."""
    try:
        from psd_tools.api.layers import PixelLayer
        from psd_tools.composite.text.layout import layout
        from psd_tools.composite.text.model import TextModel
        from psd_tools.composite.text.raster import GlyphCache, Options, rasterize
        PixelLayer.fromarray
    except (ImportError, AttributeError) as error:
        raise Refused("styled-word", f"this psd-tools cannot cut glyph pixels into an FX layer: {error}") from error
    return TextModel, layout, rasterize, Options, GlyphCache, PixelLayer


def _word_mask(layer, start: int, end: int, fonts, engine: tuple) -> np.ndarray:
    """
    Where the layer's text[start:end] inks, over the layer's bbox: Effect Layers.jsx's key copy (the word white, every
    other run black) laid out and rasterized again; a fill changes nothing in the layout, so the key's alpha is the
    raster's own.

    Args:
        layer: The rasterized text layer
        start: The word's start, a str index into the engine's text (TextModel.text)
        end: Its end
        fonts: The FontMap the layer was rasterized with
        engine: _fx_engine()

    Returns:
        bool (H, W); raises Refused("styled-word") when the key does not line up with the layer's pixels
    """
    TextModel, layout, rasterize, Options, GlyphCache = engine[:5]
    model = TextModel.from_layer(layer)
    runs = []
    for run in model.runs:
        pieces = ((run.start, min(run.end, start)), (max(run.start, start), min(run.end, end)), (max(run.start, end), run.end))
        for a, b in pieces:
            if a < b:
                fill = (1.0, 1.0, 1.0) if start <= a and b <= end else (0.0, 0.0, 0.0)
                runs.append(replace(run, start=a, end=b, style=replace(run.style, fill=fill)))
    key = replace(model, runs=tuple(runs))
    rgba, bbox = rasterize(layout(key, fonts), key.transform, key.antialias, Options(), GlyphCache(),
                           max_alloc_bytes=layer._psd._max_alloc_bytes)
    if tuple(bbox) != tuple(layer.bbox):
        raise Refused("styled-word", f"the key render {tuple(bbox)} does not line up with the text layer {tuple(layer.bbox)}")
    return rgba[..., 0] * rgba[..., 3] >= KEY_INK      # the key over black: its red channel at Photoshop's threshold level 1


def _issue_id(psd: PSDImage, layer) -> None:
    """A layer id Photoshop has not issued (past the document's seed and every id in the tree); the seed moves past it."""
    resource = psd.image_resources.get(Resource.IDS_SEED_NUMBER)
    seed = int(resource.data.value) if resource is not None else 0
    used = max((other.layer_id for other in psd.descendants()), default=0)
    new = max(seed, used) + 1
    layer.tagged_blocks.set_data(Tag.LAYER_ID, new)
    if resource is not None:
        resource.data.value = new + 1


def _cut_fx(psd: PSDImage, layer, name: str, mask: np.ndarray, color: np.ndarray, shape: np.ndarray, blocks: dict,
            engine: tuple):
    """
    A pixel layer holding the text layer's own pixels under the mask, carrying a copy of the chart layer's effect blocks.

    Args:
        psd: The card
        layer: The rasterized text layer
        name: fx_name()
        mask: _word_mask()
        color: layer.numpy("color"), the stored raster
        shape: layer.numpy("shape")
        blocks: StatusColor.blocks
        engine: _fx_engine()

    Returns:
        The layer, appended at the top of the document (the caller places it); raises Refused("styled-word") for a word
        with no ink
    """
    ys, xs = mask.nonzero()
    if not len(xs):
        raise Refused("styled-word", f"{name}: the word has no pixels")
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    rgba = np.concatenate([color[y0:y1, x0:x1], shape[y0:y1, x0:x1]], axis=2).astype(np.float32)
    rgba[~mask[y0:y1, x0:x1]] = 0.0
    fx = engine[5].fromarray(rgba, psd, name=name, top=layer.bbox[1] + y0, left=layer.bbox[0] + x0)
    fx.name = name                              # the unicode name too (fromarray writes the Pascal one)
    for key, block in blocks.items():
        fx.tagged_blocks[key] = deepcopy(block)
    _issue_id(psd, fx)
    return fx


def _regenerate_fx(psd: PSDImage, layer, text: str, styled: list, fonts, engine: tuple) -> list:
    """
    One FX layer per styled status word of the rasterized layer, directly above the top-level Text group in text order.

    Args:
        psd: The card
        layer: The ability layer, text set and rasterized
        text: The text set (the engine's text adds the final \\r and may keep a byte-order mark)
        styled: status_spans() over `text`, the styled ones
        fonts: The FontMap used
        engine: _fx_engine()

    Returns:
        The new layers, bottom to top
    """
    group = _text_group(psd)
    offset = engine[0].from_layer(layer).text.find(text)
    if offset < 0:
        raise Refused("styled-word", "the engine's text differs from the layer's")
    color, shape = layer.numpy("color"), layer.numpy("shape")
    if color is None or shape is None:
        raise Refused("styled-word", "the text layer has no pixels to cut")
    counts, seen, made = Counter(c.name for _, _, c in styled), Counter(), []
    for start, end, status in styled:
        seen[status.name] += 1
        name = fx_name(layer.name, status.name, seen[status.name] if counts[status.name] > 1 else 0)
        mask = _word_mask(layer, start + offset, end + offset, fonts, engine)
        fx = _cut_fx(psd, layer, name, mask, color, shape, status.blocks, engine)
        fx.move_to(psd, index=psd.index(group) + 1 + len(made))
        made.append(fx)
    return made


# ========================
# Apply (section 5 step 4)
# ========================
def _row(group, stat: str, two: bool) -> TemplateRow:
    """One row group as a TemplateRow (the "10" marker split off on a two-row card); else Refused("unsupported-card")."""
    layers = {child.name.strip(): child for child in group if not child.is_group()}
    marker = layers.pop(MARKER, None) if two else None
    shaped = (set(layers) == set(DIGITS) and len(group) == len(layers) + (marker is not None)
              and all(layer.kind not in DRAWN_LAYER_KINDS and layer.has_mask() for layer in layers.values()))
    if not shaped:
        raise Refused("unsupported-card", f"{stat}: non-template row {group.name!r}")
    if marker is not None and not (marker.is_visible() and marker.opacity == 0 and marker.kind not in DRAWN_LAYER_KINDS):
        raise Refused("unsupported-card", f"{stat}: the 10 marker of {group.name!r} must be visible at opacity 0")
    return TemplateRow(group, layers, marker)


def template_rows(psd: PSDImage, stat: str) -> list:
    """
    The template bar rows of a stat: one "<Stat> Dark" group holding the ten masked exposure digits 0-9, or (idea 13,
    Royal Eradicator Main Cannon's layout) a "<Stat> Dark" group holding two such rows named Top and Bottom, Top also
    holding a "10" marker: an eleventh digit visible at opacity 0, which the parser adds up as a full top row.

    Args:
        psd: An open card
        stat: hp, def, atk or spd

    Returns:
        [row] or [Top, Bottom]; raises Refused("unsupported-card") for any other row shape
    """
    rows = stat_rows(psd)[stat]
    named = lambda layer: fullmatch(rf"\s*{stat}\s*dark\s*", layer.name, IGNORECASE) is not None
    if len(rows) == 1:
        group = rows[0][0]
        if not named(group):
            raise Refused("unsupported-card", f"{stat}: non-template row {group.name!r}")
        return [_row(group, stat, False)]
    if len(rows) == 2:
        halves = {}
        for group, _, _ in rows:
            match = fullmatch(r"\s*(top|bottom)\s*", group.name, IGNORECASE)
            if match:
                halves[match.group(1).lower()] = group
        parent = getattr(halves.get("top"), "parent", None)
        if (set(halves) == {"top", "bottom"} and halves["bottom"].parent is parent
                and not isinstance(parent, PSDImage) and named(parent)):
            top, bottom = _row(halves["top"], stat, True), _row(halves["bottom"], stat, True)
            if top.marker is None:
                raise Refused("unsupported-card", f"{stat}: Top row without its 10 marker (the parser would read Bottom alone)")
            return [top, bottom]
    raise Refused("unsupported-card", f"{stat}: {len(rows)} bar rows")


def template_row(psd: PSDImage, stat: str):
    """
    The single template bar row of a stat (template_rows() for the two-row layout).

    Args:
        psd: An open card
        stat: hp, def, atk or spd

    Returns:
        (group, {digit: layer}, visible digits); raises Refused("unsupported-card") for any other row shape, two rows included
    """
    rows = template_rows(psd, stat)
    if len(rows) != 1:
        raise Refused("unsupported-card", f"{stat}: two bar rows (Top and Bottom)")
    return rows[0].group, rows[0].digits, rows[0].visible


def stat_value(rows: list) -> int:
    """What the parser reads for the rows (psd_parser.StatTracker.total): rows showing a digit add up; none shown = 10."""
    shown = [value for value in (row.reads() for row in rows) if value is not None]
    return sum(shown) if shown else STAT_MAX


def _targets(rows: list, stat: str, value: int) -> list:
    """The digit each row shows for a value (None: no digit): one row 0-10; Top the value and Bottom 0 up to 10, then
    Top none and Bottom value - 10 (20: none, when Bottom has its own 10 marker for the parser)."""
    if len(rows) == 1:
        return [value if value < STAT_MAX else None]
    if value <= STAT_MAX:
        return [value if value < STAT_MAX else None, 0]
    if value == 2 * STAT_MAX and rows[1].marker is None:
        raise Refused("stat>10", f"{stat} {value}: the Bottom row has no 10 marker for the parser to read 20")
    return [None, value - STAT_MAX if value < 2 * STAT_MAX else None]


def _tiles(layer) -> bool:
    """Whether the layer's style tiles a pattern from its fxrp: a pattern overlay, a pattern stroke or a bevel texture."""
    for effect in layer.effects:
        kind = type(effect).__name__
        fill = getattr(effect, "fill_type", None)
        if (kind == "PatternOverlay" or (kind == "BevelEmboss" and getattr(effect, "use_texture", True))
                or (kind == "Stroke" and getattr(fill, "value", fill) == b"Ptrn")):
            return True
    return False


def _trade_anchor(source, target) -> list:
    """
    The moves trading two digits' reference points (fxrp): a style's pattern and bevel texture tile from its layer's
    fxrp, so the anchor travels with the effect (the Cannon's digits sit 138-160 px apart, off the 64 px texture).

    Returns:
        [(from layer, to layer, b"fxrp")]: none when the source's style tiles nothing (_tiles) or the two hold the same bytes
    """
    key = Tag.REFERENCE_POINT
    held = {layer: layer.tagged_blocks[key].tobytes() if key in layer.tagged_blocks else None for layer in (source, target)}
    if held[source] == held[target] or not _tiles(source):
        return []
    return [(a, b, key) for a, b in ((source, target), (target, source)) if held[a] is not None]


def _move_blocks(moves: list) -> None:
    """
    Apply [(from layer, to layer or None, block key)] moves, every record keeping its block order: a block replaces the
    target's own in place, or lands after the block it followed on its source (a round trip gives back the same bytes).
    """
    before = {}
    for source, target, _ in moves:
        for layer in (source, target):
            if layer is not None and layer not in before:
                before[layer] = list(layer.tagged_blocks.items())
    gone = {(source, key) for source, _, key in moves}
    landing = {}
    for source, target, key in moves:
        if target is not None:
            landing.setdefault(target, {})[key] = (dict(before[source])[key], source)
    for layer, items in before.items():
        lands = landing.get(layer, {})
        kept = [(key, lands.pop(key)[0] if key in lands else block) for key, block in items
                if (layer, key) not in gone or key in lands]
        for key, (block, source) in lands.items():      # new to the record: after its source's nearest earlier block
            order = [k for k, _ in before[source]]
            earlier = order[:order.index(key)]
            at = max((i + 1 for i, (k, _) in enumerate(kept) if k in earlier), default=0)
            kept.insert(at, (key, block))
        layer.tagged_blocks.clear()
        for key, block in kept:
            layer.tagged_blocks[key] = block


def _toggle_row(row: TemplateRow, target: Optional[int], stat: str) -> tuple:
    """
    Show one digit of a row (None: none), hide the others, the row's effect following the visible digit (idea 12).

    Args:
        row: The row
        target: The digit to show
        stat: For the refusal text

    Returns:
        (layers whose record changes, [(from layer, to layer or None, block key)] effect blocks moved or cleared,
        pixels the effects of the layers involved can reach beyond their masks)
    """
    show = row.digits[str(target)] if target is not None else None
    hide = [layer for layer in row.digits.values() if layer.visible and layer is not show]
    source = hide if row.visible else [layer for layer in row.digits.values() if not layer.visible]
    styled = [layer for layer in source if _enabled_effects(layer, strict=True)]
    reach = max(fx_pad(layer) for layer in (row.group, *styled, *hide, *([show] if show is not None else [])))
    moves = []
    if show is not None and styled:
        if _enabled_effects(show, strict=True):         # the new digit keeps its own; the hidden one's goes
            moves += [(layer, None, key) for layer in styled if layer in hide
                      for key in EFFECT_BLOCKS if key in layer.tagged_blocks]
        elif len(styled) > 1:
            names = ", ".join(sorted(layer.name for layer in styled))
            raise Refused("unsupported-card",
                          f"{stat}: digits {names} each carry an enabled effect: which one the row shows is ambiguous")
        else:                                           # replace, never merge: a disabled block on the new digit goes first
            moves += [(show, None, key) for key in EFFECT_BLOCKS if key in show.tagged_blocks]
            moves += [(styled[0], show, key) for key in EFFECT_BLOCKS if key in styled[0].tagged_blocks]
            moves += _trade_anchor(styled[0], show)
    _move_blocks(moves)
    changed = []
    if show is not None and not show.visible:
        show.visible = True
        changed.append(show)
    for layer in hide:
        layer.visible = False
        changed.append(layer)
    for source_layer, target_layer, _ in moves:
        for layer in (source_layer, target_layer):
            if layer is not None and layer not in changed:
                changed.append(layer)
    return changed, moves, reach


def set_stat(psd: PSDImage, parsed: dict, stat: str, value: int) -> Edit:
    """
    Toggle a stat: the digit str(value) visible in "<Stat> Dark", the other nine hidden, 10 = none. On a two-row card
    (idea 13) Top shows the value and Bottom 0 up to 10, then Top none and Bottom value - 10 (20: none in both rows,
    only when Bottom carries its own 10 marker).

    The row's effect follows the visible digit (idea 12): when digit O goes hidden and N shows, O's effect blocks
    (lfx2 and its lmfx/lfxs/lrFX kin) move onto N's record, replacing any block of N's with nothing enabled, and O
    and N trade reference points when the effect tiles a pattern or texture (fxrp, where it tiles from; a round trip
    gives both records their blocks back, in their order); when N carries an enabled effect of its own it keeps it
    and O's is cleared. Nothing moves to a 10 (no digit shows: the effect stays on the hidden O), and a toggle from
    10 takes the effect of the one hidden digit carrying one; several such digits refuse (unsupported-card). A
    hidden styled digit beside a visible unstyled one (Mettaton_EX's Atk 6) is left alone: showing it later shows it
    styled. So the path matters: atk 7 -> 4 moves nothing, 7 -> 10 -> 4 gives 4 the hidden 6's effect (from 10 the
    row cannot tell a stale style from its own).

    Args:
        psd: The open card (edited in place)
        parsed: parse_card() of the card before the edit
        stat: hp, def, atk or spd
        value: 0-10 (one row) or 0-20 (two rows)

    Returns:
        Edit over the toggled digits; raises Refused for other row shapes and ambiguous effects (unsupported-card),
        values the rows cannot show (stat>10) and unreadable effect stacks (parity)
    """
    rows = template_rows(psd, stat)
    old = stat_value(rows)
    if parsed["card"].get(stat) != old:
        raise Refused("unsupported-card", f"{stat}: the parser reads {parsed['card'].get(stat)}, the row shows {old}")
    if value < 0:
        raise ValueError(f"{stat} {value}: not a stat")
    if value > STAT_MAX * len(rows):
        why = "above 20" if len(rows) > 1 else "one bar row; a second row is standardization work (ledger L3)"
        raise Refused("stat>10", f"{stat} {value}: {why}")
    changed, moves, reach = [], [], 0
    for row, target in zip(rows, _targets(rows, stat, value)):
        row_changed, row_moves, row_reach = _toggle_row(row, target, stat)
        changed += row_changed
        moves += row_moves
        reach = max(reach, row_reach)
    region = union_box([tuple(layer.mask.bbox) for layer in changed])
    return Edit(stat, old, value, changed, [_record_index(layer) for layer in changed],
                _pad(region, reach, psd) if region else None,
                [(_record_index(a), None if b is None else _record_index(b), bytes(key)) for a, b, key in moves])


def set_text_layer(psd: PSDImage, which: str, text: str, fonts_dir: str, chart: Optional[dict] = None) -> Edit:
    """
    Replace the name or the ability text, keeping the runs, and rasterize it with the text engine.

    Args:
        psd: The open card (edited in place)
        which: "name" (Text/Name) or "ability" (the single visible Text/Ability)
        text: The new text; newlines become paragraph breaks but where the layer had a forced line break (keep_breaks)
        fonts_dir: Directory of the card fonts (MDW/Fonts)
        chart: status_fills(); ability status words take their swatch fill and the styled ones their FX layer

    Returns:
        Edit with the engine's report; raises Refused: unsupported-card, styled-word, font (also colour glyphs),
        overflow; ImportError when the fork edits TySh alone (no story_sync: Photoshop would show the story's old text)

    The engine gives replaced or inserted text the style of the run before it, so an edit after a colored status
    word would inherit its color: the edited words are reset to the body fill, then every status word in the text
    takes its swatch (the chart is the authority, an off-swatch word is normalized). A chart status's FX layers are
    derived state (idea 14): every "<Ability> FX: ..." layer goes and one comes back per styled status word in the new text
    ("Blazing", "Frozen#2" for a repeat), the word's glyph pixels cut from the new raster under a copy of the chart
    layer's effect blocks, directly above the top-level Text group as Effect Layers.jsx places them. A hand-made
    overlay word layer over the ability (Gasoline's "Blazing") is hidden when it names a styled chart status, the FX
    layer standing in for it; one naming anything else refuses (unsupported-card). Any other FX layer refuses too, before
    anything changes (unsupported-card): a badge or token run's (queue A's gradient cards) or a word's the chart has no
    style for; the edit would drop it and none of them can be made again exactly. styled-word now means only that
    this psd-tools cannot cut glyph pixels or make a pixel layer.
    """
    from psd_tools.composite.text import FontMap, MissingFontError
    if not story_sync():
        raise ImportError("this psd-tools fork's set_text leaves the Txt2 story stale (no TypeLayer.sync_story)")
    layer = name_layer(psd) if which == "name" else ability_layer(psd)
    old_text, old_bbox, story = layer.text, tuple(layer.bbox), story_index(layer)
    text = text.replace("\r\n", "\n").rstrip("\n").replace("\n", "\r")     # a final \n would add an empty paragraph
    text = keep_breaks(old_text, text, which)
    start, end = changed_range(old_text, text)
    end = min(end, len(text))
    start = min(start, end)                     # identical text: the engine's range sits past the end
    spans, body, badges, styled, old_fx, hide, engine = [], None, [], [], [], [], None
    if which == "name":     # point text has no box: the title must stay on the canvas and off the badges it clears today
        stars, icons, _ = badge_layers(psd)
        badges = [badge for badge in stars + icons if not _touches(layer, badge)]
    tokens = []
    if which == "ability":
        icons = icon_font(layer)
        if TOKEN.search(text) and icons is None:
            raise Refused("unsupported-card", "[type] tokens on a card with image icons")
        # a token the edit typed takes the run before it: set it in the icon font, as the old ones are
        tokens = [match.span() for match in TOKEN.finditer(text) if match.start() < end and match.end() > start]
        old_fx, overlays = fx_layers(psd, layer), overlay_words(psd)
        badges, words = kept_fx(old_fx, layer, chart)
        if badges:
            raise Refused("unsupported-card", f"ability edit would drop the badge layer{'s' if len(badges) > 1 else ''} "
                                              f"{', '.join(badges)}; edit in Photoshop")
        if not chart:
            if old_fx or overlays:
                raise Refused("unsupported-card", "status word layers over the ability")
        else:
            if words:
                raise Refused("unsupported-card", f"ability edit would drop FX layers the chart has no style for: "
                                                  f"{', '.join(words)}; edit in Photoshop")
            hide = [overlay for overlay in overlays if getattr(chart.get(collapse(overlay.text).lower()), "styled", False)]
            stray = sorted({collapse(overlay.text) for overlay in overlays if overlay not in hide})
            if stray:
                raise Refused("unsupported-card", f"overlay word layers the chart has no style for: {', '.join(stray)}")
            body = body_fill(layer, chart)
            start, end = _word_bounds(text, start, end)
            spans = status_spans(text, chart)
            styled = [span for span in spans if span[2].styled]
            if styled or old_fx or hide:
                engine = _fx_engine()
    layer.set_text(text, keep="runs")
    if body is not None and start < end:
        layer.set_span_style(start, end, fill=body)
    for span_start, span_end, color in spans:
        layer.set_span_style(span_start, span_end, fill=color.fill)
    for token_start, token_end in tokens:
        layer.set_span_style(token_start, token_end, Font=icons)
    fonts = FontMap(dirs=[fonts_dir], aliases=FONT_ALIASES)
    try:
        report = layer.rasterize(fonts=fonts)
    except MissingFontError as error:
        raise Refused("font", str(error))
    loose = sorted(name for name, (how, _) in report.fonts.items() if how not in OK_FONTS)
    if loose:
        raise Refused("font", f"not resolved exactly: {', '.join(loose)}")
    if "overflow" in report.warnings:
        raise Refused("overflow", f"{which} does not fit")
    if "monochrome_emoji" in report.warnings and not _overlaid(layer):    # stoogedings' icons are SVG colour glyphs
        raise Refused("font", f"{which}: colour glyphs (the icon font) would be drawn in one colour")
    bbox = tuple(layer.bbox)
    if clip(bbox, psd.width, psd.height) != bbox:
        raise Refused("overflow", f"{which} runs off the canvas")
    if any(_touches(layer, badge) for badge in badges):
        raise Refused("overflow", "name reaches the stars or type icons")
    reach = [(union_box([old_bbox, bbox]), fx_pad(layer))]
    reach += [(tuple(fx.bbox), fx_pad(fx)) for fx in old_fx] + [(tuple(overlay.bbox), fx_pad(overlay)) for overlay in hide]
    for fx in old_fx:
        fx.parent.remove(fx)
    for overlay in hide:
        overlay.visible = False
    made = _regenerate_fx(psd, layer, text, styled, fonts, engine) if styled else []
    reach += [(tuple(fx.bbox), fx_pad(fx)) for fx in made]
    region = union_box([_pad(box, pad, psd) for box, pad in reach])
    return Edit(which, old_text, text, [layer], [_record_index(layer)], region, story=story,
                hidden=[_record_index(overlay) for overlay in hide],
                fx=[(fx.name, {bytes(key): fx.tagged_blocks[key].tobytes() for key in EFFECT_BLOCKS if key in fx.tagged_blocks})
                    for fx in made],
                dropped=[fx.name for fx in old_fx])


# ========================
# Render and merge (section 6)
# ========================
def changed_region(edits: list, psd: PSDImage) -> tuple:
    """
    The union of the edits' padded regions, clipped to the canvas (the artifacts' crop; gates run per edit).

    Args:
        edits: The applied Edits
        psd: The card (its canvas size)

    Returns:
        The bbox; raises Refused("idempotence") when no edit changed a layer
    """
    region = union_box([edit.region for edit in edits])
    if region is None:
        raise Refused("idempotence", "no layer changed")
    return clip(region, psd.width, psd.height)


def foreign_profile(psd: PSDImage) -> Optional[bytes]:
    """The embedded ICC profile of an RGB card when it is not sRGB (score_cards.py _icc_profile), else None."""
    from PIL import ImageCms
    if psd.color_mode != ColorMode.RGB or Resource.ICC_PROFILE not in psd.image_resources:
        return None
    data = psd.image_resources.get_data(Resource.ICC_PROFILE)
    try:
        description = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(BytesIO(data)))
    except Exception:
        return data
    return None if "srgb" in description.lower() else data


def icc_vote(psd: PSDImage, png: Image.Image) -> Optional[tuple]:
    """
    The harness's ICC vote (score_cards.py render) over the stored preview: of the pixels the profile's conversion
    moves by ICC_MOVED or more, the share where the PNG is nearer the unconverted values.

    Args:
        psd: The card
        png: Its official PNG

    Returns:
        (vote, moved pixels); None without a foreign profile or a stored preview, or when the preview is not RGB(A) of the
        PNG's size
    """
    profile = foreign_profile(psd)
    if profile is None or not psd.has_preview():
        return None
    preview = psd.topil(apply_icc=False)
    if preview is None or preview.mode not in ("RGB", "RGBA") or preview.size != png.size:
        return None
    raw, conv = _flat(preview)[..., :3].astype(np.int16), _flat(post_process(preview, None, profile))[..., :3].astype(np.int16)
    official = _flat(png)[..., :3].astype(np.int16)
    moved = np.abs(raw - conv).max(axis=2) >= ICC_MOVED
    count = int(moved.sum())
    if not count:
        return 0.0, 0
    nearer = np.abs(raw - official).sum(axis=2) < np.abs(conv - official).sum(axis=2)
    return float(nearer[moved].mean()), count


def export_icc(psd: PSDImage, png: Image.Image) -> bool:
    """
    Whether the card's PNG was exported converted to sRGB (render with apply_icc) or raw, unconverted (21 cards on
    2026-10-22: Block_Hopper, PoD01Red_Bloon...): raw when icc_vote() moved ICC_MIN_PX pixels and voted ICC_RAW_VOTE.

    Args:
        psd: The card before the edit
        png: Its official PNG

    Returns:
        False for a raw export; True otherwise (no foreign profile, no preview, no evidence)
    """
    vote = icc_vote(psd, png)
    return vote is None or vote[1] < ICC_MIN_PX or vote[0] < ICC_RAW_VOTE


def render_region(psd: PSDImage, region: tuple, icc: bool = True) -> Image.Image:
    """
    psd.composite over a viewport, never from the stored preview and never forced (section 15).

    Args:
        psd: The card
        region: Canvas bbox to render
        icc: Convert by the embedded profile (export_icc(): how the card's PNG was exported)

    Returns:
        The render, region-sized
    """
    return psd.composite(viewport=region, ignore_preview=True, apply_icc=icc)


def render_pair(psd: PSDImage, region: tuple, icc: bool = True) -> tuple:
    """
    One render of the region, as render_region() returns it and in the PSD's own values (what its preview stores).

    Args:
        psd: The card
        region: Canvas bbox to render
        icc: render_region()'s

    Returns:
        (render, raw): raw before the embedded profile's sRGB conversion; the same image without a profile or icc
    """
    raw = psd.composite(viewport=region, ignore_preview=True, apply_icc=False)
    if not icc or Resource.ICC_PROFILE not in psd.image_resources:
        return raw, raw
    return post_process(raw, None, psd.image_resources.get_data(Resource.ICC_PROFILE)), raw


def baseline(psd: PSDImage, png: Image.Image, region: tuple, icc: bool = True) -> tuple:
    """
    Render the unedited region and measure it against the official PNG (gate parity).

    Args:
        psd: The card before the edit
        png: Its official PNG
        region: The edit's padded region
        icc: render_region()'s

    Returns:
        (render, measurement); raises Refused("parity") when the fork already renders the region wrong
    """
    render = render_region(psd, region, icc)
    measured = _measure(render, png.crop(region))
    if not _within_bar(measured):
        raise Refused("parity", "baseline bad %.2f%% mean %.2f max %d" % (measured["bad_pct"], measured["mean"], measured["max"]))
    return render, measured


def merge_png(png: Image.Image, base: Image.Image, render: Image.Image, region: tuple) -> tuple:
    """
    Delta-merge: inside the region, the official PNG takes the new render's pixels where it differs from the
    baseline render; every other pixel is left as Photoshop exported it.

    Args:
        png: The official PNG (or the merge so far)
        base: baseline()'s render of the region
        render: The region rendered from the edited PSD
        region: The edit's padded region

    Returns:
        (merged image in the PNG's mode, region-sized mask of the replaced pixels)
    """
    if png.mode not in ("RGB", "RGBA"):
        raise Refused("shape", f"PNG mode {png.mode}")
    x0, y0, x1, y1 = region
    out = np.array(png.convert("RGBA"))
    after = np.asarray(render.convert("RGBA"))
    changed = (np.asarray(base.convert("RGBA")) != after).any(axis=2)
    patch = out[y0:y1, x0:x1]
    patch[changed] = after[changed]
    return Image.fromarray(out).convert(png.mode), changed


def stored_preview(psd: PSDImage) -> Optional[Image.Image]:
    """
    The PSD's flattened preview in its own values, for merge_preview() and save_psd().

    Args:
        psd: The card before the edit

    Returns:
        RGB or RGBA; None (save() regenerates it) without a preview, past 8-bit RGB, or under stock psd-tools (no
        save(preview=))
    """
    if "preview" not in signature(psd.save).parameters or psd.color_mode != ColorMode.RGB or psd.depth != 8:
        return None
    if not psd.has_preview():
        return None
    image = psd.topil(apply_icc=False)
    return image if image is not None and image.mode in ("RGB", "RGBA") else None


def merge_preview(preview: Image.Image, patches: list) -> Image.Image:
    """
    merge_png() for the PSD's own preview: the pixels it replaced take the raw render (render_pair()).

    Args:
        preview: stored_preview()
        patches: (merge_png()'s mask, raw render, region) per edit

    Returns:
        The preview, its mode kept (an RGB one takes the raw render over white)
    """
    out = np.array(preview.convert("RGBA"))
    for changed, raw, (x0, y0, x1, y1) in patches:
        after = _flat(raw) if preview.mode == "RGB" else np.asarray(raw.convert("RGBA"))
        out[y0:y1, x0:x1][changed] = after[changed]
    return Image.fromarray(out).convert(preview.mode)


def artifacts(png: Image.Image, merged: Image.Image, region: tuple) -> dict:
    """
    Before, after and an amplified diff crop of the region (the --dry-run artifacts).

    Args:
        png: The official PNG
        merged: The merged PNG
        region: The crop

    Returns:
        {"before", "after", "diff": image}
    """
    before, after = png.crop(region), merged.crop(region)
    diff = np.abs(_flat(before).astype(np.int16) - _flat(after).astype(np.int16))[..., :3].max(axis=2)
    return {"before": before, "after": after, "diff": Image.fromarray(np.clip(diff * 8, 0, 255).astype(np.uint8))}


# ========================
# Gates (sections 5 and 10)
# ========================
def gate_shape(png: Image.Image, merged: Image.Image, psd: PSDImage) -> None:
    """The PNG is the PSD's canvas and the merge kept its size and mode; else Refused("shape")."""
    if png.size != (psd.width, psd.height):
        raise Refused("shape", f"PNG {png.size} is not the PSD canvas {(psd.width, psd.height)}")
    if merged.size != png.size or merged.mode != png.mode:
        raise Refused("shape", "PNG size or mode changed")


def gate_region(png: Image.Image, merged: Image.Image, region: tuple) -> int:
    """
    Every pixel differing between the old and new PNG lies inside the region.

    Args:
        png: The official PNG
        merged: The merged PNG
        region: The union of the edits' regions

    Returns:
        How many pixels differ; raises Refused("region") for any outside
    """
    diff = (np.asarray(png.convert("RGBA")) != np.asarray(merged.convert("RGBA"))).any(axis=2)
    inside = np.zeros_like(diff)
    inside[region[1]:region[3], region[0]:region[2]] = True
    outside = int((diff & ~inside).sum())
    if outside:
        raise Refused("region", f"{outside} pixels differ outside the region")
    return int(diff.sum())


def gate_size(old_path: str, new_path: str) -> None:
    """The saved PSD stays within SIZE_TOLERANCE of the old one; else Refused("size")."""
    old, new = getsize(old_path), getsize(new_path)
    if abs(new - old) > SIZE_TOLERANCE * old:
        raise Refused("size", f"PSD {old} -> {new} bytes")


def layer_records(path: str) -> list:
    """Each layer record's identity-free fingerprint: name, flags, rect, channels, mask, blend and tagged blocks by key."""
    psd = PSDImage.open(path)
    out = []
    for record in psd._record.layer_and_mask_information.layer_info.layer_records:
        blocks = record.tagged_blocks or {}
        out.append({"name": blocks.get_data(Tag.UNICODE_LAYER_NAME, record.name) if blocks else record.name,
                    "flags": record.flags.tobytes(),
                    "rect": (record.top, record.left, record.bottom, record.right),
                    "channels": tuple((int(info.id), info.length) for info in record.channel_info),
                    "mask": record.mask_data.tobytes() if record.mask_data is not None else b"",
                    "blend": (record.blend_mode, record.opacity, record.clipping),
                    "blocks": tuple((bytes(key), block.tobytes()) for key, block in (record.tagged_blocks or {}).items())})
    return out


def _pixel_record(record: dict) -> bool:
    """A plain pixel record: a transparency channel then the color channels in order."""
    ids = [channel for channel, _ in record["channels"]]
    return bool(ids) and ids[0] == -1 and ids[1:] == list(range(len(ids) - 1))


def gate_bytes(old_path: str, new_path: str, edits: list) -> tuple:
    """
    Compare the layer records of the old and new PSD: only the edited layers change, and only in the fields their
    kind of edit touches (a toggle: exactly one flag byte per toggled digit, plus the effect blocks its Edit declares
    moved, the same bytes leaving one digit's record and landing on another's, or leaving only when cleared; a text
    edit: rect, channels, EngineData; its FX layers: exactly the declared records gone and come, the new ones pixel
    records carrying the chart's effect blocks byte for byte; a hidden overlay word layer: one flag byte). Records
    are paired in file order once the FX records declared removed and added are taken out.

    Returns:
        ([(index, name, changed fields)] in file order ("removed" at the old index, "added" at the new one), differing
        flag bytes, effect block bytes that landed on a digit or an FX layer)
    """
    old, new = layer_records(old_path), layer_records(new_path)
    dropped = Counter(name for edit in edits for name in edit.dropped)
    added = {name: blocks for edit in edits for name, blocks in edit.fx}
    allowed = {}
    for edit in edits:
        kind = "stat" if edit.field in STAT_FIELDS else "text"
        for index in edit.records:
            allowed[index] = (kind, ALLOWED_RECORD_FIELDS[kind])
        for index in edit.hidden:
            allowed[index] = ("overlay", ALLOWED_RECORD_FIELDS["overlay"])
    changed, flag_bytes, block_bytes, old_of = [], 0, 0, {}
    i = j = 0
    while i < len(old) or j < len(new):
        a = old[i] if i < len(old) else None
        b = new[j] if j < len(new) else None
        if a is not None and dropped.get(a["name"]):
            dropped[a["name"]] -= 1
            changed.append((i, a["name"], ["removed"]))
            i += 1
            continue
        if b is not None and b["name"] in added:
            blocks = dict(b["blocks"])
            if not _pixel_record(b) or any(blocks.get(key) != data for key, data in added[b["name"]].items()):
                raise Refused("bytes", f"FX layer {b['name']!r} is not the declared pixel layer")
            block_bytes += sum(len(data) for data in added.pop(b["name"]).values())
            changed.append((j, b["name"], ["added"]))
            j += 1
            continue
        if a is None or b is None:
            raise Refused("bytes", f"layer count {len(old)} -> {len(new)}")
        old_of[j] = i
        fields = sorted(key for key in a if a[key] != b[key])
        if fields:
            kind, ok = allowed.get(j, ("", set()))
            if not set(fields) <= ok:
                raise Refused("bytes", f"layer {j} {a['name']!r} changed {', '.join(fields)}")
            if "flags" in fields:
                differing = sum(1 for x, y in zip(a["flags"], b["flags"]) if x != y)
                if kind in ("stat", "overlay") and differing != 1:
                    raise Refused("bytes", f"{kind} {a['name']!r} changed {differing} flag bytes")
                flag_bytes += differing
            changed.append((j, a["name"], fields))
        i += 1
        j += 1
    if any(dropped.values()) or added:
        raise Refused("bytes", "an FX layer declared removed or added is not")
    hidden = {index for edit in edits for index in edit.hidden}
    toggled = {index for edit in edits if edit.field in STAT_FIELDS for index in edit.records}
    if (toggled | hidden) - {index for index, _, fields in changed if fields != ["removed"]}:
        raise Refused("bytes", "a toggled digit or hidden overlay did not change")
    # effect blocks on the toggled digits: exactly the declared moves, byte for byte
    moves = [move for edit in edits if edit.field in STAT_FIELDS for move in edit.moves]
    blocks_of = lambda records, index: dict(records[index]["blocks"])
    left = {(src, key): blocks_of(old, old_of[src]).get(key) for src, _, key in moves}
    landed = {(dst, key): blocks_of(old, old_of[src]).get(key) for src, dst, key in moves if dst is not None}
    if None in left.values():
        raise Refused("bytes", "a moved effect block was not on its digit")
    went, came = {}, {}
    for index in toggled:
        before, after = blocks_of(old, old_of[index]), blocks_of(new, index)
        went.update({(index, key): data for key, data in before.items() if after.get(key) != data})
        came.update({(index, key): data for key, data in after.items() if before.get(key) != data})
    if went != left or came != landed:
        raise Refused("bytes", "effect blocks changed beyond the declared moves")
    return changed, flag_bytes, block_bytes + sum(len(data) for data in landed.values())


def _raw(element) -> bytes:
    buf = BytesIO()
    element.write(buf)
    return buf.getvalue()


def _document_blocks(psd: PSDImage) -> dict:
    return {bytes(key): block.tobytes() for key, block in (psd._record.layer_and_mask_information.tagged_blocks or {}).items()}


def gate_stories(old: PSDImage, new: PSDImage, edited) -> list:
    """
    Photoshop reads a type layer's text from its Txt2 story and rebuilds the layer's TySh from it on save, so an edit
    to TySh alone is undone the first time the card is opened (2026-10-22: B.F.B. went back to B.F.B). Every type
    layer keeps its TextIndex and its story holds its layer's text and styles (story_style_drift), only the edited
    layers' stories changed, and the rest of Txt2 and every other document-wide block are the old bytes.

    Args:
        old: The card as opened
        new: The saved card, reopened
        edited: TextIndex of each layer whose story may change (taken before the edit)

    Returns:
        The changed stories' indices, sorted; raises Refused("bytes")
    """
    before, after = _document_blocks(old), _document_blocks(new)
    txt2 = bytes(Tag.TEXT_ENGINE_DATA)
    moved = sorted(key.decode("latin-1") for key in set(before) | set(after)
                   if (key != txt2 or (key in before) != (key in after)) and before.get(key) != after.get(key))
    if moved:
        raise Refused("bytes", f"document blocks changed: {', '.join(moved)}")
    stale = stale_stories(new)
    if stale:
        name, story, own = stale[0]
        raise Refused("bytes", f"{name}'s Txt2 story holds {collapse(story)!r}, the layer {collapse(own)!r}")
    if txt2 not in before:
        return []
    indices = [sorted(story_index(layer) for layer in psd.descendants() if _is_text_layer(layer)
                      and story_index(layer) is not None) for psd in (old, new)]
    if indices[0] != indices[1]:                # a layer without one is orphaned from the story Photoshop would read
        raise Refused("bytes", f"type layers' TextIndex {indices[0]} -> {indices[1]}")
    (objects_a, _, items_a, _, _, _), (objects_b, _, items_b, _, _, _) = _story_list(_txt2(old)), _story_list(_txt2(new))
    if len(items_a) != len(items_b):
        raise Refused("bytes", f"Txt2 stories {len(items_a)} -> {len(items_b)}")
    changed = [i for i, (a, b) in enumerate(zip(items_a, items_b)) if _raw(a) != _raw(b)]
    beyond = sorted(set(changed) - set(edited))
    if beyond:
        raise Refused("bytes", f"Txt2 stories {beyond} changed: no edit there")
    rest = [[(str(key), _raw(value)) for key, value in root.items() if value is not objects]
            + [(str(key), _raw(value)) for key, value in objects.items() if value is not items]
            for root, objects, items in ((_txt2(old), objects_a, items_a), (_txt2(new), objects_b, items_b))]
    if rest[0] != rest[1]:
        raise Refused("bytes", "Txt2 changed outside its stories")
    drift = story_style_drift(new)
    if drift:
        raise Refused("bytes", f"{drift[0][0]}'s Txt2 story has other styles: {drift[0][1]}")
    return changed


def gate_reparse(parser: PSDParser, parsed: dict, new_path: str, rel: str, jobs: dict, psd: PSDImage, edits=()) -> dict:
    """
    Re-parse the new PSD: the old parse with exactly the edited fields replaced, no new problems (a stat above 10
    reads STATS TOO HIGH by design when it sits on Top and Bottom rows; no EFFECT LAYER problem: the validator's
    coverage check is the FX layers' gate), one digit per edited row, the title read back for a name edit, and an
    ability edit's FX layers back as declared: visible, styled, directly above Text, no overlay word layer showing.

    Args:
        parser: offline_parser()
        parsed: parse_card() of the old PSD
        new_path: The saved PSD
        rel: The card's repo path
        jobs: The applied jobs
        psd: The saved PSD, reopened
        edits: The applied Edits (their FX layers and hidden overlays are checked)

    Returns:
        parse_card() of the new PSD
    """
    for stat in jobs:
        if stat in STAT_FIELDS:
            rows = template_rows(psd, stat)
            for row, target in zip(rows, _targets(rows, stat, jobs[stat])):
                if row.visible != ([target] if target is not None else []):
                    raise Refused("reparse", f"{stat}: row {row.group.name!r} shows {row.visible}")
    for edit in edits:
        if edit.field == "ability" and (edit.fx or edit.dropped or edit.hidden):
            layer, group = ability_layer(psd), _text_group(psd)
            found = fx_layers(psd, layer)
            if [fx.name for fx in found] != [name for name, _ in edit.fx]:
                raise Refused("reparse", f"FX layers after reopen: {[fx.name for fx in found]}")
            for k, fx in enumerate(found):
                if (fx.parent is not psd or psd.index(fx) != psd.index(group) + 1 + k or not fx.is_visible()
                        or not _enabled_effects(fx, strict=True)):
                    raise Refused("reparse", f"{fx.name!r} is not a visible styled layer directly above Text")
            if overlay_words(psd):
                raise Refused("reparse", "overlay word layers still show over the ability")
    new = parse_card(parser, new_path, rel)
    expected, got = dict(parsed["card"]), dict(new["card"])
    for name, value in jobs.items():
        if name in STAT_FIELDS:
            expected[name] = value
        elif name == "ability":
            expected["ability"], got["ability"] = normalize_ability(value), normalize_ability(got.get("ability"))
        elif name == "name":                    # the new title is read back below
            expected.pop("display_name", None)
            got.pop("display_name", None)
    expected.pop("problem", None)
    got.pop("problem", None)
    different = sorted(key for key in set(expected) | set(got) if expected.get(key) != got.get(key))
    if different:
        raise Refused("reparse", "fields differ from the expected parse: " + ", ".join(different))
    if new["alt_names"] != parsed["alt_names"]:
        raise Refused("reparse", "hidden titles changed")
    # STATS TOO HIGH is by design on two-row cards (a raise from <= 10 adds it), a problem anywhere else
    two_rows = all(_two_rows(got, psd, stat) for stat in STAT_FIELDS if (got.get(stat) or 0) > STAT_MAX)
    if any(p.startswith("STATS TOO HIGH") for p in new["validator"]) and not two_rows:
        raise Refused("reparse", "a stat above 10 without Top and Bottom rows")
    answered = tuple(p for name in jobs for p in CardEdits.FIXED_BY.get(name, ()))
    old_problems = {p for p in parsed["validator"] if not (answered and p.startswith(answered))}
    added = sorted(p for p in set(new["validator"]) - old_problems if not (two_rows and p.startswith("STATS TOO HIGH")))
    if added:
        raise Refused("reparse", "new problems: " + "; ".join(added))
    if any(p.startswith("EFFECT LAYER") for p in new["validator"]):
        raise Refused("reparse", "effect layer problems remain")
    if "name" in jobs and read_name(psd) != collapse(jobs["name"]):
        raise Refused("reparse", f"Text/Name reads {read_name(psd)!r}")
    return new


def gate_idempotence(new_parsed: dict, jobs: dict, psd: PSDImage, merged: Image.Image, edits: list,
                     icc: bool = True) -> dict:
    """
    Nothing left to bake on the new PSD, and each edit's region re-rendered from it matches the merged PNG within
    the parity bar (not byte-equal: the delta merge keeps Photoshop's pixels where the render matched the baseline).

    Args:
        new_parsed: gate_reparse()'s parse of the saved PSD
        jobs: The applied jobs
        psd: The saved PSD, reopened
        merged: The merged PNG
        edits: The applied Edits
        icc: render_region()'s

    Returns:
        {field: region measurement}
    """
    again = classify(new_parsed, jobs, psd)
    if again.jobs:
        raise Refused("idempotence", "still to bake: " + ", ".join(sorted(again.jobs)))
    out = {}
    for edit in edits:
        measured = _measure(render_region(psd, edit.region, icc), merged.crop(edit.region))
        if not _within_bar(measured):
            raise Refused("idempotence", "%s re-render bad %.2f%% mean %.2f" % (edit.field, measured["bad_pct"], measured["mean"]))
        out[edit.field] = measured
    return out


# ========================
# Save
# ========================
def save_psd(psd: PSDImage, path: str, update_preview: bool = True, preview: Optional[Image.Image] = None) -> None:
    """
    Write the PSD.

    Args:
        psd: The edited card
        path: Where it goes
        update_preview: Store a current flattened preview (decision 2: always, from the driver); False keeps the
            stale one (the fork-only tests)
        preview: merge_preview()'s image, stored as it is; None regenerates it from the layers (a full composite)
    """
    if update_preview and preview is not None:
        psd.save(path, preview=preview)
    elif update_preview:
        psd.save(path)
    elif "update_preview" in signature(psd.save).parameters:
        psd.save(path, update_preview=False)
    else:                                       # the fork's save(): its private flag is what triggers the preview
        updated = psd._updated
        psd._updated = False
        try:
            psd.save(path)
        finally:
            psd._updated = updated


def save_png(image: Image.Image, path: str, like: Image.Image) -> None:
    """
    Write the merged PNG with the official PNG's color chunks (iCCP, gAMA, cHRM, pHYs): only pixels change.

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
        update_preview: save_psd(); the preview is the old one with the merge's pixels when the fork can store it

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
    stale = stale_stories(new)
    if stale:                                   # a pre-fix bake's PSD: Photoshop already shows the old text
        raise Refused("unsupported-card", f"{stale[0][0]}'s Txt2 story is stale: run commands.txt2_repair on it")
    drift = story_style_drift(new)
    if drift:                                   # Photoshop shows the story's styles, not the layer's
        raise Refused("unsupported-card", f"{drift[0][0]}'s Txt2 story has other styles: {drift[0][1]}")

    edits = []                                  # text first: an ability's FX records shift the digits' record indices
    for name, value in sorted(jobs.items(), key=lambda job: job[0] in STAT_FIELDS):
        if name in STAT_FIELDS:
            edits.append(set_stat(new, parsed, name, value))
        elif name in TEXT_FIELDS:
            edits.append(set_text_layer(new, name, value, fonts_dir, chart))
        else:
            raise Refused("unsupported-card", f"{name}: not yet")
    edits = [edit for edit in edits if edit.layers]
    region = changed_region(edits, new)

    # region per edited field (section 5 step 3): the parity bar is measured over the pixels that edit can reach
    merged, parity, changed_pixels, patches = png, {}, 0, []
    preview = stored_preview(old) if update_preview else None
    icc = export_icc(old, png)
    timings["baseline"] = timings["render"] = 0.0
    for edit in edits:
        start = perf_counter()
        base, parity[edit.field] = baseline(old, png, edit.region, icc)
        timings["baseline"] += perf_counter() - start
        start = perf_counter()
        if preview is None:
            render = render_region(new, edit.region, icc)
        else:
            render, raw = render_pair(new, edit.region, icc)
        timings["render"] += perf_counter() - start
        merged, replaced = merge_png(merged, base, render, edit.region)
        if preview is not None:
            patches.append((replaced, raw, edit.region))
        changed_pixels += int(replaced.sum())
    if preview is not None:
        preview = merge_preview(preview, patches)
    gate_shape(png, merged, new)
    gate_region(png, merged, region)     # merge_png writes only inside the regions: the gate restates its contract

    new_psd, new_png = join(out_dir, basename(psd_path)), join(out_dir, basename(png_path))
    start = perf_counter()
    save_psd(new, new_psd, update_preview, preview)
    timings["save"] = perf_counter() - start
    save_png(merged, new_png, png)
    gate_size(psd_path, new_psd)
    records, flag_bytes, block_bytes = gate_bytes(psd_path, new_psd, edits)

    start = perf_counter()
    reopened = PSDImage.open(new_psd)
    gate_stories(old, reopened, {edit.story for edit in edits if edit.field in TEXT_FIELDS})
    reparse = gate_reparse(parser, parsed, new_psd, rel, jobs, reopened, edits)
    timings["reparse"] = perf_counter() - start
    start = perf_counter()
    gate_idempotence(reparse, jobs, reopened, merged, edits, icc)
    timings["idempotence"] = perf_counter() - start
    return Baked(jobs, edits, region, parity, changed_pixels, records, flag_bytes, new_psd, new_png, reparse, timings,
                 block_bytes, [name for edit in edits for name, _ in edit.fx], icc)


# ========================
# Module Exports
# ========================
__all__ = [
    'Baked',
    'CODES',
    'Classification',
    'Edit',
    'FONTS_DIR',
    'KIND',
    'NOT_YET',
    'OK_FONTS',
    'Refusal',
    'Refused',
    'STATUS_CHART',
    'STORY_KEYS',
    'STORY_MIRROR',
    'STORY_STYLE_KEYS',
    'STAT_FIELDS',
    'STAT_MAX',
    'StatusColor',
    'TEXT_FIELDS',
    'TemplateRow',
    'ability_layer',
    'apply_jobs',
    'artifacts',
    'baseline',
    'body_fill',
    'changed_range',
    'changed_region',
    'classify',
    'collapse',
    'export_icc',
    'foreign_profile',
    'fx_label',
    'fx_layers',
    'fx_name',
    'gate_bytes',
    'gate_idempotence',
    'gate_region',
    'gate_reparse',
    'gate_shape',
    'gate_size',
    'gate_stories',
    'icc_vote',
    'icon_font',
    'keep_breaks',
    'kept_fx',
    'layer_records',
    'merge_png',
    'merge_preview',
    'name_layer',
    'normalize_ability',
    'offline_parser',
    'parse_card',
    'read_name',
    'shown_name',
    'render_pair',
    'render_region',
    'run_fills',
    'save_png',
    'save_psd',
    'set_stat',
    'stale_stories',
    'stat_value',
    'text_engine',
    'set_text_layer',
    'status_fills',
    'status_spans',
    'stored_preview',
    'stories',
    'story_index',
    'story_style_drift',
    'story_sync',
    'template_row',
    'template_rows',
    'uses_icon_font',
]
