"""
Edits to a card PSD for the edit pipeline (docs/EDIT_PIPELINE_DESIGN.md sections 3, 5, 6, 7 and 10): classify a card's
website edits against a fresh offline parse, toggle a stat digit, set the name or ability text through the fork's
text engine with status words colored from the chart and the styled ones' FX layers cut from the new raster, show,
hide, move or copy star layers and type icons for a stars or types edit (a refusal of those is per field, never the
card's), render only the changed region, delta-merge it onto the official PNG and gate the result. Pure functions
over an open PSDImage; a refusal is a Refused exception carrying one of CODES, which the per-card driver turns into
data. No Discord, network or config; the text extra (psd_tools.composite.text) is imported only where text is
rendered, so the module loads under stock psd-tools.
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
from commands.psd_parser import (DRAWN_LAYER_KINDS, EFFECT_LAYER_TAG, ICON_FONT_PREFIX, TITLE_REGION_RATIO,
                                 TYPE_REGION_RATIO, CardValidator, IconVocabulary, PSDParser, name_key, named_after_text,
                                 stars_from_psd, title_text)
from commands.psd_regions import badge_layers, clip, fx_pad, overlay_words, stat_rows, union_box
from global_config import FONT_ALIASES, RULEBOOK_DIR

# ========================
# Configuration
# ========================
STATUS_CHART = "MDW/Croissant's Special Folder/Satus Effect Colors.psd"   # one text layer per status; its fill is the color
FONTS_DIR = "MDW/Fonts"
CODES = ("duplicate", "unsupported-card", "stat>10", "recent", "blob", "parity", "styled-word", "font", "overflow",
         "bytes", "size", "region", "shape", "reparse", "idempotence", "edit-changed", "head-moved", "capped", "error")
STAT_FIELDS = ("hp", "def", "atk", "spd")
TEXT_FIELDS = ("name", "ability")
ICON_FIELDS = ("stars", "types")                # printed as icons; refused per field, never for the card (owner, 2026-10-23)
KIND = CardEdits.KIND
STAT_MAX = 10                                   # one bar row shows 0-10; Top + Bottom rows show 11-20 (idea 13)
DIGITS = tuple(str(n) for n in range(10))
MARKER = "10"                                   # a two-row card's parser marker: "10" visible at opacity 0, never toggled
EFFECT_BLOCKS = (Tag.OBJECT_BASED_EFFECTS_LAYER_INFO, Tag.OBJECT_BASED_EFFECTS_LAYER_INFO_V0,
                 Tag.OBJECT_BASED_EFFECTS_LAYER_INFO_V1, Tag.EFFECTS_LAYER)   # a digit's effect blocks: lfx2 and its kin
PARITY_THRESH = 24          # a pixel is bad past this |diff| (psd-research score_cards.py THRESH)
PARITY_BAD_MAX = 0.5        # % of bad pixels a region may show against the official PNG (past it: a miss, told)
PARITY_FLOOR = 25.0         # % of bad pixels past which a region's render is broken: refused; a miss under it is told (owner, 2026-10-23)
PARITY_MEAN_MAX = 2.0       # mean |diff| over the region (owner, 2026-10-22; Mini_Doomer hp row: 0.83, max 2)
COPY_SEEN = 1               # a copy of the title or ability shows when leaving it out moves a pixel past rounding
ICC_MOVED = 4               # the ICC vote (score_cards.py): pixels the profile's conversion moves this far in a channel
ICC_RAW_VOTE = 0.65         # share of them nearer the unconverted values: the PNG was exported raw
ICC_MIN_PX = 1000           # fewer moved pixels: no evidence, converted
# How type layers' edges blend, tried in order by baseline(): the fork reads each layer's off the PSD's preview; a
# PNG exported in another Photoshop session than the preview's may show them all in the text gamma, or all straight.
# Under "preview" the edited layers are pinned to the gamma (text_kwargs()): the old preview does not show them
TEXT_BLENDS = {"preview": {}, "gamma": {"text_preview": False}, "straight": {"text_gamma": None}}
SIZE_TOLERANCE = 0.2        # a saved PSD stays within 20% of the old blob
OK_FONTS = ("exact", "system")                  # how a font may resolve; loose or substituted faces refuse
TOKEN = re_compile(r"\[[^\[\]\r\n\x03]{1,52}\]")   # an icon token in ability text ([fire]); psd_parser.ICON_SCAN_PATTERN
GLOWS = ("glow", "darkglow")  # token variants whose halo the fork draws as its glyf outline: a solid block, not a soft glow
LETTER = re_compile(r"[A-Za-z]")                   # status_spans' word boundary
SAME_RUN = 3                # edit_parts: characters an edit keeps this many in a row are not restyled
STYLE_KEYS = ("Font", "FontSize", "FauxBold", "FauxItalic", "AutoLeading", "Leading", "HorizontalScale",
              "VerticalScale", "Tracking", "BaselineShift", "FontCaps", "FontBaseline", "Ligatures",
              "BaselineDirection", "Tsume", "StyleRunAlignment", "Language", "NoBreak")   # edit_styles copies these
TYPED = re_compile(r"\([^()\r\x03]*\)|\[[^\[\]\r\x03]*\]")   # a badge or token as typed: "(1CD)", "[fire]"
INK_ALPHA = 0.5             # a pixel is ink past this alpha (a title grazing a badge's anti-aliased edge is not an overlap)
FX_TAG = EFFECT_LAYER_TAG   # "Ability FX: Blazing": a status word's glyph pixels under the chart's effect (Effect Layers.jsx)
BADGE_FX = re_compile(r"[(\[][^()\[\]]+[)\]](?: \| [(\[][^()\[\]]+[)\]])*")   # "(1CD) | (3U)", "[fire]": badge and token runs
KEY_INK = 0.5 / 255         # the JSX's key copy, the word white over black, selects every pixel its red channel reaches (level 1)
SPLICE_UNIT = "glyph"       # a text edit redraws its changed glyphs ("line": their lines); Photoshop's raster stays elsewhere
KEY_SLACK = 2               # px a key render may pass a spliced layer's bbox (Photoshop's ink beside the engine's)
ALLOWED_RECORD_FIELDS = {"stat": {"flags", "blocks"}, "text": {"rect", "channels", "blocks"},
                         "overlay": {"flags"}, "toggle": {"flags"}}  # what an edit may change per record (overlay: a hidden
#   hand-made word layer; toggle: a star or type icon shown or hidden)
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
STAR_FOLDER = r"\s*(\d+)\s*stars?\s*"           # a count's own folder in a Stars group: "5 stars", "1 star" (the templates)
LAYOUT_TOL = 0.12           # a star or icon sits on a layout's spot within this share of its size
LAYOUT_TOL_MIN = 6          # px, at least
ICON_FRAME = 0.1            # a type layer over more of the canvas is a frame (Magnum Opus's field), not an icon
ICON_GAP = 22               # px between an icon and the one under it on a 1200-wide card (Item_Template: Attack, NME)
CARD_WIDTH = 1200
PLACED = (Tag.SMART_OBJECT_LAYER_DATA1, Tag.SMART_OBJECT_LAYER_DATA2, Tag.PLACED_LAYER1, Tag.PLACED_LAYER2)
MOVE_BLOCKS = {bytes(key) for key in (*PLACED, Tag.REFERENCE_POINT)}    # a moved layer's record: its transform, anchor
COPY_BLOCKS = MOVE_BLOCKS | {bytes(Tag.LAYER_ID), bytes(Tag.UNICODE_LAYER_NAME)}   # a copy's: those, its id and name


# ========================
# Data
# ========================
@dataclass
class Refusal:
    """One refusal: a code from CODES and why; field: the stars or types edit it is about (the card's others still bake)."""
    code: str
    why: str
    field: Optional[str] = None


class Refused(Exception):
    """
    A refusal raised inside a card's steps; the driver turns it into a Refusal (one bad card never blocks another).
    field: the stars or types edit it is about; the driver bakes the card's other fields without it.
    """
    def __init__(self, code: str, why: str, field: Optional[str] = None):
        super().__init__(f"{code}: {why}")
        self.code = code
        self.why = why
        self.field = field

    @property
    def refusal(self) -> Refusal:
        return Refusal(self.code, self.why, self.field)


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
class IconPlan:
    """A stars or types edit worked out on an open card (plan_stars, plan_types): what shows, hides, moves and comes."""
    field: str
    old: object
    new: object
    how: str = ""                                # the layout followed: "folder '3 stars'", "rows of two", "a line"...
    show: list = field(default_factory=list)     # layers shown
    hide: list = field(default_factory=list)     # layers hidden
    moves: list = field(default_factory=list)    # (layer, dx, dy): moved (a group with everything in it)
    copies: list = field(default_factory=list)   # (source, dx, dy, name): a star copied above itself, then moved
    icons: list = field(default_factory=list)    # (Types/ stem, canvas box, reference layer): a type icon made in the box
    done: list = field(default_factory=list)     # set_icons(): ("copy", source, copy, dx, dy) and ("made", layer, blocks)


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
    fx: list = field(default_factory=list)      # (name, {block key: bytes}) per FX layer cut above its group, bottom to top
    dropped: list = field(default_factory=list)  # names of the FX layers deleted first (derived state: all of the text layer's)
    story: Optional[int] = None  # the text layer's TextIndex before the edit: the Txt2 story it may change
    band: Optional[tuple] = None  # canvas bbox of the text pixels redrawn (a splice: its changed glyphs); None: none
    renamed: Optional[str] = None  # a title layer named after its text: its new name, the new title (as Photoshop does)
    icons: Optional["IconPlan"] = None  # a stars or types edit's plan (index_icons() fills the four lists below)
    toggled: list = field(default_factory=list)  # record indices of the layers it showed or hid: one flag byte each
    shifted: list = field(default_factory=list)  # (record index, old record index, dx, dy): a layer it moved
    copied: list = field(default_factory=list)   # (record index, old record index, dx, dy): a record of a star it copied
    made: list = field(default_factory=list)     # (record index, {block key: bytes}): a type icon it made from Types/


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
    rerender: dict = field(default_factory=dict)    # field -> gate_idempotence()'s measurement of its region
    icons: dict = field(default_factory=dict)       # stars / types -> the layout their edit followed (IconPlan.how)


# ========================
# Helper Functions
# ========================
def _pad(box, reach: int, psd: PSDImage) -> tuple:
    return clip((box[0] - reach, box[1] - reach, box[2] + reach, box[3] + reach), psd.width, psd.height)


def _off_canvas(box, psd: PSDImage) -> bool:
    """Whether a bbox has nothing on the canvas (wholly outside it, or empty): clipped, it is empty."""
    x0, y0, x1, y1 = clip(box, psd.width, psd.height)
    return x1 <= x0 or y1 <= y0


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


def _past_floor(measured: dict) -> bool:
    """A broken render: more than PARITY_FLOOR % of the region's pixels past PARITY_THRESH (the only pixel refusal)."""
    return measured["bad_pct"] > PARITY_FLOOR


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


def edited_ids(edits: list) -> tuple:
    """Layer IDs of the edits' layers (text_kwargs()): the same in the edited PSD and the saved one."""
    return tuple(layer.layer_id for edit in edits for layer in edit.layers)


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
def _top(layer):
    """The layer's top-level group, or the layer at the root: Effect Layers.jsx puts an FX layer directly above it."""
    while layer.parent is not None and layer.parent.parent is not None:
        layer = layer.parent
    return layer


def _shows(psd: PSDImage, layer) -> bool:
    """Does the card's render change without the layer (not when the art covers it: Doctor Zomboss's 'images/Name')."""
    box = clip(layer.bbox, psd.width, psd.height)
    if box[2] <= box[0] or box[3] <= box[1]:
        return False
    shown = psd.composite(viewport=box, ignore_preview=True, apply_icc=False)
    left_out = psd.composite(viewport=box, ignore_preview=True, apply_icc=False,
                             layer_filter=lambda other: other.is_visible() and other is not layer)
    return bool((np.abs(_flat(shown).astype(np.int16) - _flat(left_out).astype(np.int16)) > COPY_SEEN).any())


def _copy(psd: PSDImage, layer) -> Optional[str]:
    """
    The first other visible text layer printing the layer's words where they show (a copy for an effect, a watermark):
    an edit would change one and leave the rest. None when there is none.
    """
    def words(text):
        return " ".join((text or "").replace("\x03", " ").split()).casefold()
    own = words(layer.text)
    if not own:
        return None
    return next((other.name for other in psd.descendants() if _is_text_layer(other) and other is not layer
                 and other.is_visible() and words(other.text) == own and _shows(psd, other)), None)


def _no_title(psd: PSDImage, own_text: bool = True) -> str:
    """Why the parser reads no title layer (name_layer's refusal); own_text: title_layers' last rule applies."""
    group = _text_group(psd)
    if group is None:
        return "no Text group"
    named = [child for child in group if child.name.strip() == "Name"]
    if named:
        if not any(_is_text_layer(child) for child in named):
            return "rasterized name layer"
        if not any(_is_text_layer(child) and child.is_visible() for child in named):
            return "hidden name layer"
        return f"{len(named)} Name layers, none in the title region"
    own = [layer for layer in psd.descendants() if _is_text_layer(layer) and layer.is_visible()
           and layer.bbox[1] < psd.height * TITLE_REGION_RATIO and named_after_text(layer)]
    if own_text and len(own) > 1:
        return f"{len(own)} layers in the title region named after their text"
    titles = [child.name for child in group if _is_text_layer(child) and child.name.strip().lower() != "ability"
              and child.bbox[1] < psd.height * 0.15]
    names = "Name, the card's name or its text" if own_text else "Name or the card's name"
    return f"title layer named {titles[0]!r}, not {names}" if titles else "no Text/Name layer"


def name_layer(psd: PSDImage, card: str = "", own_text: bool = True):
    """
    The card's title layer: the one visible text layer the parser reads the printed title from (PSDParser.title_layers:
    Text/Name, else a Name layer in the title region, else a layer under Text named after the card, else one elsewhere
    in the title region: a status effect's top-level title, else the one there named after its text: Sans's 'snas.').

    Args:
        psd: The open card
        card: The card's name (file stem), which a title layer may be named after
        own_text: Whether the last rule applies (False on a rule page, as the parser: _own_text)

    Returns:
        The layer; raises Refused("unsupported-card") when no layer or several print the title (an epithet layer, a
        title split over layers, a copy for an effect that shows)
    """
    found = PSDParser.title_layers(psd, [layer for layer in psd.descendants() if _is_text_layer(layer)], card, own_text)
    if not found:
        raise Refused("unsupported-card", _no_title(psd, own_text))
    if len(found) > 1:
        raise Refused("unsupported-card", f"{len(found)} title layers")
    if PSDParser.split_title(psd, found, title_text(found[0].text), card):
        raise Refused("unsupported-card", "the title is split over several layers")
    copy = _copy(psd, found[0])
    if copy is not None:
        raise Refused("unsupported-card", f"the title is printed again by {copy!r}")
    return found[0]


def read_name(psd: PSDImage, card: str = "", own_text: bool = True) -> Optional[str]:
    """The printed title, whitespace collapsed; None when name_layer refuses (own_text: as name_layer's)."""
    try:
        return collapse(name_layer(psd, card, own_text).text)
    except Refused:
        return None


def own_text_title(layer, card: str = "") -> bool:
    """Whether name_layer found the layer by its own text alone (title_layers' last rule): a bake renames it."""
    return layer.name.strip() != "Name" and name_key(layer.name) != name_key(card) and named_after_text(layer)


def _card_name(parsed: dict) -> str:
    """The parse's card name (file stem)."""
    info = parsed.get("info")
    return info.name if info is not None else parsed["card"].get("name") or ""


def _own_text(parsed: dict) -> bool:
    """Whether a title named after its own text counts for the parse's card: not on a rule page (its headings)."""
    info = parsed.get("info")
    return RULEBOOK_DIR not in ((info.path if info is not None else parsed["card"].get("path")) or "")


def story_sync() -> bool:
    """Does the fork's set_text also write the layer's Txt2 story (TypeLayer.sync_story): the text Photoshop reads."""
    from psd_tools.api.layers import TypeLayer
    return hasattr(TypeLayer, "sync_story")


def splices() -> bool:
    """Does the fork's rasterize keep Photoshop's raster of the unchanged lines (rasterize(keep=))."""
    from psd_tools.api.layers import TypeLayer
    return "keep" in signature(TypeLayer.rasterize).parameters


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
    The single visible Ability text layer, in whatever group the developer put it (the parser reads it wherever it is).

    Returns:
        The layer; raises Refused("unsupported-card") for none or several, or another layer printing its words where
        they show
    """
    visible = [layer for layer in psd.descendants()
               if _is_text_layer(layer) and layer.is_visible() and layer.name.strip().lower() == "ability"]
    if len(visible) != 1:
        raise Refused("unsupported-card", f"{len(visible)} visible Ability layers")
    copy = _copy(psd, visible[0])
    if copy is not None:
        raise Refused("unsupported-card", f"the ability is printed again by {copy!r}")
    return visible[0]


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


def _char_styles(layer) -> tuple:
    """
    Per str index of the layer's text: its fill (RGB 0..1 or None), whether it is set in the icon font, its run's
    STYLE_KEYS as set_span_style takes them (Font by PostScript name; {} when unreadable) and its run's number.
    """
    text = layer.text
    fills, icons, styles, runs = [None] * len(text), [False] * len(text), [{}] * len(text), [-1] * len(text)
    for paragraph in layer.typesetting:
        for run in paragraph.runs:
            fill = run.style.fill_color
            fill = tuple(float(v) for v in fill)[-3:] if fill else None
            icon = run.style.font_name.lower().startswith(ICON_FONT_PREFIX)
            for i in range(max(run.start, 0), min(run.end, len(text))):
                fills[i], icons[i], runs[i] = fill, icon, (paragraph.start, run.start)
    try:
        fonts = [str(font["Name"].value) for font in layer.resource_dict["FontSet"]]
        style = layer.engine_dict["StyleRun"]
        i = unit = 0
        for length, run in zip(style["RunLengthArray"], style["RunArray"]):
            data = run["StyleSheet"]["StyleSheetData"]
            keys = {key: deepcopy(data[key]) for key in STYLE_KEYS if key in data}
            if "Font" in keys:
                keys["Font"] = fonts[int(keys["Font"])]
            unit += int(length)
            while i < len(text) and unit > 0:
                styles[i] = keys
                unit -= 2 if ord(text[i]) > 0xFFFF else 1
                i += 1
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        styles = [{}] * len(text)
    return fills, icons, styles, runs


def _stretches(indices: list) -> list:
    """[(start, end)] of the runs of consecutive ints in a sorted list."""
    out = []
    for i in indices:
        if out and out[len(out) - 1][1] == i:
            out[len(out) - 1] = (out[len(out) - 1][0], i + 1)
        else:
            out.append((i, i + 1))
    return out


def edit_parts(old: str, new: str, start: int, end: int) -> list:
    """
    The edited span new[start:end] against the old text it replaces, in order: stretches of SAME_RUN or more characters
    the edit keeps, and the changed parts between them (one side may be empty: an insertion or a deletion).

    Args:
        old: The layer's text
        new: The text to set
        start, end: The edited span of new (changed_range, maybe widened to whole words)

    Returns:
        [(old start, old end, new start, new end, kept)]
    """
    lo = min(start, len(old))                   # the span starts in the common prefix: the same index in old
    hi = min(max(end + len(old) - len(new), lo), len(old))
    out, a0, b0 = [], 0, 0
    for a, b, size in SequenceMatcher(None, old[lo:hi], new[start:end], autojunk=False).get_matching_blocks():
        if 0 < size < SAME_RUN:
            continue
        if a > a0 or b > b0:
            out.append((lo + a0, lo + a, start + b0, start + b, False))
        if size:
            out.append((lo + a, lo + a + size, start + b, start + b + size, True))
        a0, b0 = a + size, b + size
    return out


def splice_text(layer, text: str, parts: list) -> None:
    """
    set_text(keep="runs") one changed part at a time (edit_parts), last first: the engine restyles only what changed, so
    an edit at both ends of an ability leaves the badges and colours between them alone (in one call, everything from
    the first change to the last would take the first changed character's style).
    """
    current = layer.text
    for a0, a1, b0, b1, kept in reversed(parts):
        if not kept:
            current = current[:a0] + text[b0:b1] + current[a1:]
            layer.set_text(current, keep="runs")
    layer.set_text(text, keep="runs")           # nothing left but a stale story to bring along


def edit_styles(layer, parts: list, chart: Optional[dict]) -> list:
    """
    The style of each changed part (edit_parts) of an ability edit, from the old text: a part replacing text takes the
    commonest fill of what it replaces outside status words and blanks (set in the icon font only if all of it is: a
    badge edit); an insertion, or a part replacing only those, the nearest character before it in its line (\\r or
    \\x03), else after, that is no status word, blank or badge and shares no run with a status word (Astr's "3" of its
    red "1T3 Empowered"), else its line's commonest fill, its paragraph's, the text's. A kept status word's characters
    take that too: the edit resets status words, set_text_layer giving the new text's their swatch after. Before
    set_text.

    Args:
        layer: The text layer
        parts: edit_parts()
        chart: status_fills(), or None

    Returns:
        [(new start, new end, RGB 0..1 or None, the source's STYLE_KEYS or None when it is a badge character)]
    """
    old = layer.text
    fills, icons, styles, runs = _char_styles(layer)
    status = {i for s, e, _ in (status_spans(old, chart) if chart else []) for i in range(s, e)}
    tied = {runs[i] for i in status}

    def source(i: int, icon: bool = False) -> bool:
        return fills[i] is not None and i not in status and old[i] not in " \t\r\x03" and (icon or not icons[i])

    def commonest(found: list, at: int) -> int:
        fill = Counter(fills[i] for i in found).most_common(1)[0][0]
        same = [i for i in found if fills[i] == fill]
        return next((i for i in reversed(same) if i < at), same[0])

    def near(lo: int, hi: int) -> Optional[int]:
        for breaks in ("\r\x03", "\r", ""):
            top = max([old.rfind(c, 0, lo) for c in breaks] or [-1]) + 1
            bottom = min([i for i in (old.find(c, hi) for c in breaks) if i >= 0] or [len(old)])
            clean = [i for i in [*range(lo - 1, top - 1, -1), *range(hi, bottom)] if source(i) and runs[i] not in tied]
            if clean and breaks == "\r\x03":
                return clean[0]
            found = [i for i in range(top, bottom) if source(i)]
            if found:
                return commonest(found, lo)
        return None

    out = []
    for a0, a1, b0, b1, kept in parts:
        if kept:
            spans = [(b0 + k, b0 + k + 1, near(a0 + k, a0 + k + 1)) for k in range(a1 - a0) if a0 + k in status]
        elif b1 > b0:
            i = next((commonest(found, a0) for icon in (False, True)
                      for found in [[i for i in range(a0, a1) if source(i, icon)]] if found), None)
            spans = [(b0, b1, near(a0, a1) if i is None else i)]
        else:
            spans = []
        for start, end, i in spans:
            out.append((start, end, body_fill(layer, chart or {}) if i is None else fills[i],
                        None if i is None or icons[i] else styles[i]))
    return out


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


def keep_ends(old: str, new: str) -> str:
    """
    The new text with the layer's blank first and last paragraphs back: the website and Discord trim an ability's ends,
    so setting it as given would drop PoD10BFB's leading empty paragraph (a spacing line) and move the text up.

    Args:
        old: The layer's text
        new: The text to set

    Returns:
        new, with old's leading breaks (through its last break before any text) and trailing ones (from its first break
        after the text) when new has none there
    """
    blank = " \t\r\x03"
    if not old.strip(blank) or not new.strip(blank):
        return new
    head = old[:len(old) - len(old.lstrip(blank))]
    tail = old[len(old.rstrip(blank)):]
    head = head[:max(head.rfind("\r"), head.rfind("\x03")) + 1]
    cut = [i for i in (tail.find("\r"), tail.find("\x03")) if i >= 0]
    tail = tail[min(cut):] if cut else ""
    if head and new[0] not in blank:
        new = head + new
    if tail and new[len(new) - 1] not in blank:
        new = new + tail
    return new


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


def classify(parsed: dict, edits: dict, psd: Optional[PSDImage] = None, parser: Optional[PSDParser] = None) -> Classification:
    """
    Sort a card's edits: PSD-kind fields whose value differs from the fresh parse are jobs, equal ones redundant.
    A stat above 10 is a job only on a two-row card (idea 13), above 20 never. Stars and types are jobs when the card's
    layout is one plan_stars / plan_types knows; their refusals carry the field (the card's other fields still bake).

    Args:
        parsed: parse_card() of the card's PSD
        edits: The card's fields from CardEdits.parse (no _keys)
        psd: The open PSD, needed to find the title layer for a name edit (name_layer), the rows for a stat above 10
            and the stars and type icons (not changed)
        parser: offline_parser(): its icon vocabulary reads a types edit (types compare in any order and spelling)

    Returns:
        Classification
    """
    card = parsed["card"]
    out = Classification()
    for name, value in edits.items():
        if KIND.get(name, "database") != "psd" or value is None:
            out.other.append(name)
        elif name in ICON_FIELDS:
            if card.get(name) == value or (name == "types" and same_types(parser, card.get("types"), value)):
                out.redundant.append(name)
            elif psd is None:
                out.refusals.append(Refusal("unsupported-card", f"{name}: no PSD to place them on", name))
            else:
                try:
                    if name == "stars":
                        plan_stars(psd, parsed, value, parser)
                    else:
                        plan_types(psd, parsed, value, parser)
                    out.jobs[name] = value
                except Refused as refused:
                    out.refusals.append(refused.refusal)
                except Exception as error:      # noqa: BLE001 - a layout the planner trips on is that field's alone
                    out.refusals.append(Refusal("error", f"{name}: {type(error).__name__}: {error}", name))
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
            elif psd is None:
                out.refusals.append(Refusal("unsupported-card", "no PSD to find the title layer in"))
            else:
                try:
                    name_layer(psd, _card_name(parsed), _own_text(parsed))
                    out.jobs[name] = value
                except Refused as refused:
                    out.refusals.append(refused.refusal)
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


def _key_ink(model, fills, layer, fonts, engine: tuple) -> np.ndarray:
    """A key copy of the layer's text, runs refilled by fills(run) -> [(start, end, fill)]: where its red channel reaches."""
    _, layout, rasterize, Options, GlyphCache = engine[:5]
    runs = [replace(run, start=a, end=b, style=replace(run.style, fill=fill)) for run in model.runs for a, b, fill in fills(run)]
    key = replace(model, runs=tuple(runs))
    rgba, bbox = rasterize(layout(key, fonts), key.transform, key.antialias, Options(), GlyphCache(),
                           max_alloc_bytes=layer._psd._max_alloc_bytes)
    x0, y0, x1, y1 = (int(v) for v in bbox)
    lx0, ly0, lx1, ly1 = layer.bbox
    if x0 < lx0 - KEY_SLACK or y0 < ly0 - KEY_SLACK or x1 > lx1 + KEY_SLACK or y1 > ly1 + KEY_SLACK:
        raise Refused("styled-word", f"the key render {tuple(bbox)} does not line up with the text layer {tuple(layer.bbox)}")
    ink = rgba[..., 0] * rgba[..., 3] >= KEY_INK      # the key over black: its red channel at Photoshop's threshold level 1
    out = np.zeros((ly1 - ly0, lx1 - lx0), bool)     # on the layer's bbox (a splice keeps Photoshop's, not the key's)
    cx0, cy0, cx1, cy1 = max(x0, lx0), max(y0, ly0), min(x1, lx1), min(y1, ly1)
    if cx1 > cx0 and cy1 > cy0:
        out[cy0 - ly0:cy1 - ly0, cx0 - lx0:cx1 - lx0] = ink[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    return out


def _word_mask(layer, start: int, end: int, fonts, engine: tuple) -> np.ndarray:
    """
    Where the layer's text[start:end] inks, over the layer's bbox: Effect Layers.jsx's key copy (the word white, every
    other run black) laid out and rasterized again; a fill changes nothing in the layout, so the key's alpha is the
    raster's own. The icon font's colour glyphs keep their SVG colours in any key, so an all-black key finds them and
    they stay out of the mask, as in the JSX's fix for badge pixels in FX layers (std-batch2 FX.md).

    Args:
        layer: The rasterized text layer
        start: The word's start, a str index into the engine's text (TextModel.text)
        end: Its end
        fonts: The FontMap the layer was rasterized with
        engine: _fx_engine()

    Returns:
        bool (H, W); raises Refused("styled-word") when the key does not line up with the layer's pixels
    """
    model = engine[0].from_layer(layer)
    black, white = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)

    def word(run):
        pieces = ((run.start, min(run.end, start)), (max(run.start, start), min(run.end, end)), (max(run.start, end), run.end))
        return [(a, b, white if start <= a and b <= end else black) for a, b in pieces if a < b]

    mask = _key_ink(model, word, layer, fonts, engine)
    if uses_icon_font(layer):
        mask &= ~_key_ink(model, lambda run: [(run.start, run.end, black)], layer, fonts, engine)
    return mask


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
    One FX layer per styled status word of the rasterized layer, directly above its top-level group (the Text group;
    the layer itself at the root) in text order, where Effect Layers.jsx puts it.

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
    group = _top(layer)
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
        Edit over the toggled digits; raises Refused for other row shapes, a digit mask (or the whole bar) off the
        canvas and ambiguous effects (unsupported-card), values the rows cannot show (stat>10) and unreadable effect
        stacks (parity)
    """
    rows = template_rows(psd, stat)
    for row in rows:                            # Green_Mage: masks below the canvas, the cells drawn as pixels
        boxes = {name: tuple(layer.mask.bbox) for name, layer in row.digits.items()}
        off = sorted(name for name, box in boxes.items() if _off_canvas(box, psd))
        if off:                                 # a toggle there changes nothing (and its region would be empty)
            what = (f"the bar {row.group.name!r} lies" if len(off) == len(boxes) else
                    f"digit {off[0]} of the bar {row.group.name!r} lies" if len(off) == 1 else
                    f"digits {', '.join(off)} of the bar {row.group.name!r} lie")
            raise Refused("unsupported-card", f"{stat}: {what} outside the card (digit masks "
                                              f"{union_box([boxes[name] for name in off])} on the {psd.width}x{psd.height} "
                                              "canvas): a toggle would change nothing on the card")
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


def set_text_layer(psd: PSDImage, which: str, text: str, fonts_dir: str, chart: Optional[dict] = None,
                   card: str = "", own_text: bool = True) -> Edit:
    """
    Replace the name or the ability text, keeping the runs, and rasterize it with the text engine: only the changed
    glyphs when the fork splices (splices(), SPLICE_UNIT), Photoshop's raster staying on the rest. A title layer found
    by its own text alone (own_text_title: Sans's 'snas.') takes the new title as its name, as Photoshop renames a layer
    it named, so the parser still reads it (gate_bytes allows that name).

    Args:
        psd: The open card (edited in place)
        which: "name" (name_layer) or "ability" (ability_layer)
        text: The new text; newlines become paragraph breaks but where the layer had a forced line break (keep_breaks);
            the layer's blank first and last paragraphs stay (keep_ends)
        fonts_dir: Directory of the card fonts (MDW/Fonts)
        chart: status_fills(); ability status words take their swatch fill and the styled ones their FX layer
        card: The card's name (file stem), for name_layer
        own_text: name_layer's (False on a rule page: _own_text)

    Returns:
        Edit with the engine's report; raises Refused: unsupported-card, styled-word, font (also one-colour glyphs and glow tokens),
        overflow; ImportError when the fork edits TySh alone (no story_sync: Photoshop would show the story's old text)

    The text is set one changed part at a time (splice_text), so what the edit keeps keeps its runs. The engine gives
    inserted text the style of the run before it, so an edit after a colored status word or a badge would take its
    color or icon font: an ability's changed parts take the fill of the text they replace (edit_styles; status words
    give none), words set in the icon font outside a typed badge or token that text's whole style, then every status
    word in the text takes its swatch (the chart is the authority, an off-swatch word is normalized). A chart status's
    FX layers are derived state (idea 14): every "<Ability> FX: ..." layer goes and one comes back per styled status
    word in the new text ("Blazing", "Frozen#2" for a repeat), the word's glyph pixels cut from the new raster under a
    copy of the chart layer's effect blocks, directly above the layer's top-level group as Effect Layers.jsx places
    them. A hand-made overlay word layer over the ability (Gasoline's "Blazing") is hidden when it names a styled chart
    status, the FX layer standing in for it; one naming anything else refuses (unsupported-card). Any other FX layer
    refuses too, before anything changes (unsupported-card): a badge or token run's (queue A's gradient cards) or a
    word's the chart has no style for; the edit would drop it and none of them can be made again exactly. styled-word
    now means only that this psd-tools cannot cut glyph pixels or make a pixel layer.
    """
    from psd_tools.composite.text import FontMap, MissingFontError
    if not story_sync():
        raise ImportError("this psd-tools fork's set_text leaves the Txt2 story stale (no TypeLayer.sync_story)")
    layer = name_layer(psd, card, own_text) if which == "name" else ability_layer(psd)
    old_text, old_bbox, story = layer.text, tuple(layer.bbox), story_index(layer)
    own = which == "name" and own_text_title(layer, card)
    text = text.replace("\r\n", "\n").rstrip("\n").replace("\n", "\r")     # a final \n would add an empty paragraph
    text = keep_ends(old_text, keep_breaks(old_text, text, which))
    start, end = changed_range(old_text, text)
    end = min(end, len(text))
    start = min(start, end)                     # identical text: the engine's range sits past the end
    spans, badges, styled, old_fx, hide, engine = [], [], [], [], [], None
    if which == "name":     # point text has no box: the title must stay on the canvas and off the badges it clears today
        stars, icons, _ = badge_layers(psd)
        badges = [badge for badge in stars + icons if not _touches(layer, badge)]
    tokens = []
    if which == "ability":
        icons = icon_font(layer)
        if TOKEN.search(text) and icons is None:
            raise Refused("unsupported-card", "[type] tokens on a card with image icons")
        glows = [m.group(0) for m in TOKEN.finditer(text) if m.group(0)[1:-1].partition(":")[2].strip().lower() in GLOWS]
        if glows:
            raise Refused("font", f"{', '.join(glows)}: this psd-tools draws a glow halo as a solid block; edit in Photoshop")
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
            start, end = _word_bounds(text, start, end)
            spans = status_spans(text, chart)
            styled = [span for span in spans if span[2].styled]
            if styled or old_fx or hide:
                engine = _fx_engine()
    parts = edit_parts(old_text, text, start, end)
    styles = edit_styles(layer, parts, chart) if which == "ability" else []
    fonts = FontMap(dirs=[fonts_dir], aliases=FONT_ALIASES)
    before = None
    if splices():                               # Photoshop's raster stays on the lines the edit leaves alone
        from psd_tools.composite.text import capture
        try:
            before = capture(layer, fonts)
        except MissingFontError as error:
            raise Refused("font", str(error))
    splice_text(layer, text, parts)
    if styles:          # words the engine set in a badge's icon font (typed after one) take their source's style
        now, typed = _char_styles(layer)[1], {i for m in TYPED.finditer(text) for i in range(*m.span())}
        for part_start, part_end, fill, like in styles:
            for a, b in _stretches([i for i in range(part_start, part_end) if like and now[i] and i not in typed]):
                layer.set_span_style(a, b, **like)
            layer.set_span_style(part_start, part_end, fill=fill)
    for span_start, span_end, color in spans:
        layer.set_span_style(span_start, span_end, fill=color.fill)
    for token_start, token_end in tokens:
        layer.set_span_style(token_start, token_end, Font=icons)
    try:
        report = layer.rasterize(fonts=fonts, keep=before, unit=SPLICE_UNIT) if before else layer.rasterize(fonts=fonts)
    except MissingFontError as error:
        raise Refused("font", str(error))
    loose = sorted(name for name, (how, _) in report.fonts.items() if how not in OK_FONTS)
    if loose:
        raise Refused("font", f"not resolved exactly: {', '.join(loose)}")
    if "overflow" in report.warnings:
        raise Refused("overflow", f"{which} does not fit")
    if "monochrome_emoji" in report.warnings and not _overlaid(layer):    # an SVG colour glyph the fork draws in one colour
        raise Refused("font", f"{which}: colour glyphs (the icon font) would be drawn in one colour")
    bbox = tuple(layer.bbox)
    spliced = getattr(report, "splice", None)
    band = bbox if spliced is None or spliced.fallback else spliced.zone
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
    if own:                                     # else the parser would no longer find the title
        layer.name = collapse(text)
    return Edit(which, old_text, text, [layer], [_record_index(layer)], region, story=story, band=band,
                hidden=[_record_index(overlay) for overlay in hide],
                fx=[(fx.name, {bytes(key): fx.tagged_blocks[key].tobytes() for key in EFFECT_BLOCKS if key in fx.tagged_blocks})
                    for fx in made],
                dropped=[fx.name for fx in old_fx], renamed=layer.name if own else None)


# ========================
# Stars and types (the icon fields, owner 2026-10-23)
# ========================
def _tol(size: float) -> float:
    return max(LAYOUT_TOL_MIN, LAYOUT_TOL * size)


def _centre(box) -> tuple:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _moved_box(box, dx: int, dy: int) -> tuple:
    return box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy


def _clusters(boxes: list) -> list:
    """Index lists of the stars among core boxes, joined as PSDParser._count_clusters joins them."""
    parent = list(range(len(boxes)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if PSDParser._center_inside(boxes[i], boxes[j]) or PSDParser._center_inside(boxes[j], boxes[i]):
                parent[root(i)] = root(j)
    out: dict = {}
    for i in range(len(boxes)):
        out.setdefault(root(i), []).append(i)
    return list(out.values())


def _drawn(layer) -> list:
    """The layer, or a group's leaves with pixels that show when the group does (their own flags and their groups')."""
    if not layer.is_group():
        return [layer] if layer.has_pixels() else []
    return [leaf for child in layer if child.visible for leaf in _drawn(child)]


def _core(layer) -> Optional[tuple]:
    """The union of the core boxes the parser clusters stars by (PSDParser._core_box), None when nothing draws."""
    return union_box([PSDParser._core_box(leaf) for leaf in _drawn(layer)])


def _star_count(group, show: dict, boxes: dict, counts: dict) -> int:
    """
    What PSDParser._stars_in counts in a group once layers are shown or hidden and copies added.

    Args:
        group: A group in a Stars group
        show: id(layer) -> visible, overriding its flag
        boxes: id(group) -> core boxes counted among its loose layers (a loose star's copy)
        counts: id(group) -> stars added there (a star group's copy)
    """
    count, loose = counts.get(id(group), 0), list(boxes.get(id(group), ()))
    for layer in group:
        if not show.get(id(layer), layer.visible):
            continue
        if layer.is_group():
            count += _star_count(layer, show, boxes, counts)
        elif layer.has_pixels():
            loose.append(PSDParser._core_box(layer))
    return count + len(_clusters(loose))


def _stars_total(psd: PSDImage, show=None, boxes=None, counts=None) -> int:
    """The parser's count over the card's Stars groups (PSDParser._count_stars), with _star_count's changes."""
    return sum(_star_count(group, show or {}, boxes or {}, counts or {}) for group in PSDParser._star_groups(psd))


@dataclass
class _Unit:
    """One star of a layout: a group counting one, or the loose layers drawing one; or an icon of the type row."""
    layers: list            # panel order, bottom to top
    core: tuple             # canvas box of its core
    count: int = 1

    @property
    def box(self) -> tuple:
        """Its layers' bboxes' union (a star's glow and frame included)."""
        return union_box([tuple(layer.bbox) for layer in self.layers])

    @property
    def centre(self) -> tuple:
        return _centre(self.core)

    @property
    def size(self) -> tuple:
        return self.core[2] - self.core[0], self.core[3] - self.core[1]


def _star_units(container, types=()) -> tuple:
    """
    (shown, hidden) stars directly in a group: its child groups that count stars, and its loose layers' clusters; a
    layer named after a type icon (types: Types/ names) is left out (the parser reads it as a type too: Pike Beam).
    """
    shown, hidden, loose = [], [], {True: [], False: []}
    for layer in container:
        if layer.name.strip().lower() in types:
            continue
        if layer.is_group():
            count = _star_count(layer, {}, {}, {})
            core = _core(layer)
            if count and core:
                (shown if layer.visible else hidden).append(_Unit([layer], core, count))
        elif layer.has_pixels() and layer.bbox[2] > layer.bbox[0]:
            loose[layer.visible].append(layer)
    for on, layers in loose.items():
        boxes = [PSDParser._core_box(layer) for layer in layers]
        for members in _clusters(boxes):
            (shown if on else hidden).append(_Unit([layers[i] for i in members], union_box([boxes[i] for i in members])))
    return shown, hidden


def _rows(units: list, size: float, right_first: bool) -> list:
    """Units in rows by centre height (within half a unit), top to bottom; each row right to left or left to right."""
    rows: list = []
    for unit in sorted(units, key=lambda u: u.centre[1]):
        if rows and abs(unit.centre[1] - rows[len(rows) - 1][0].centre[1]) <= size / 2:
            rows[len(rows) - 1].append(unit)
        else:
            rows.append([unit])
    return [sorted(row, key=lambda u: -u.centre[0] if right_first else u.centre[0]) for row in rows]


def _reading(units: list, size: float, right_first: bool) -> list:
    return [unit for row in _rows(units, size, right_first) for unit in row]


def _median_size(units: list) -> tuple:
    widths, heights = sorted(u.size[0] for u in units), sorted(u.size[1] for u in units)
    return widths[len(widths) // 2], heights[len(heights) // 2]


def _pyramid(units: list, size: tuple) -> Optional[tuple]:
    """
    (right x, left x, top y, row step, odd row on top) when the stars are rows of two, an odd one centred under them
    (the templates' 1-5 star layouts: Status_Effect_Template's '3 star', Dull Knife, Felhunter, Love) or over them (The
    Claw); None otherwise. A lone star is the right of a pair: the second goes to its left, a star's width over, the
    next row a star's height down (the templates' '2 star'; the cards' single stars sit where their pairs' right ones do).
    """
    if len(units) == 1:
        (x, y), box = units[0].centre, units[0].box
        return x, x - (box[2] - box[0]), y, box[3] - box[1], False
    rows = _rows(units, size[1], True)
    last = len(rows) - 1
    top_odd = last > 0 and len(rows[0]) == 1 and len(rows[last]) == 2
    full = rows[1:] if top_odd else rows[:last] if len(rows[last]) == 1 else rows
    if not full or any(len(row) != 2 for row in full):
        return None
    tol = _tol(max(size))
    if any(abs(row[0].centre[1] - row[1].centre[1]) > tol for row in full):
        return None
    right = sum(row[0].centre[0] for row in full) / len(full)
    left = sum(row[1].centre[0] for row in full) / len(full)
    if any(abs(row[0].centre[0] - right) > tol or abs(row[1].centre[0] - left) > tol for row in full):
        return None
    ys = [sum(u.centre[1] for u in row) / len(row) for row in rows]
    step = (ys[last] - ys[0]) / last if last else size[1]
    if any(abs(y - (ys[0] + i * step)) > tol for i, y in enumerate(ys)) or step < size[1] / 2:
        return None
    odd = rows[0] if top_odd else rows[last] if len(full) < len(rows) else None
    if odd and abs(odd[0].centre[0] - (right + left) / 2) > tol:
        return None
    return right, left, ys[0], step, top_odd


def _pyramid_spots(model: tuple, count: int) -> list:
    """The centres of `count` stars in rows of two (_pyramid), reading order; an odd one centred where the model has it."""
    right, left, top, step, top_odd = model
    middle = (right + left) / 2
    if count == 1:                              # a lone star stays where a pair's right one is (or the top one)
        return [(middle, top)] if top_odd else [(right, top)]
    rows = [2] * (count // 2)
    if count % 2:
        rows = [1] + rows if top_odd else rows + [1]
    out = []
    for row, stars in enumerate(rows):
        y = top + row * step
        out += [(right, y), (left, y)] if stars == 2 else [(middle, y)]
    return out


def _line(units: list, size: tuple, right_first: bool) -> Optional[tuple]:
    """(first centre, step) when the units in reading order are evenly spaced in a row or a column; else None."""
    order = _reading(units, size[1], right_first)
    if len(order) < 2:
        return None
    first, last = order[0].centre, order[len(order) - 1].centre
    step = ((last[0] - first[0]) / (len(order) - 1), (last[1] - first[1]) / (len(order) - 1))
    tol = _tol(max(size))
    if max(abs(step[0]), abs(step[1])) < max(size) / 2 or min(abs(step[0]), abs(step[1])) > tol:
        return None
    for i, unit in enumerate(order):
        if abs(unit.centre[0] - first[0] - i * step[0]) > tol or abs(unit.centre[1] - first[1] - i * step[1]) > tol:
            return None
    return first, step


def _place(units: list, spots: list, size: tuple) -> tuple:
    """
    Units onto spots: one already on a spot (within the tolerance) stays, the others move to the free spots in order.

    Returns:
        ([(unit, dx, dy)] moves, free spots left over)
    """
    tol, free, moves, taken = _tol(max(size)), list(spots), [], []
    for unit in units:
        spot = next((s for s in free if abs(s[0] - unit.centre[0]) <= tol and abs(s[1] - unit.centre[1]) <= tol), None)
        if spot is None:
            taken.append(unit)
        else:
            free.remove(spot)
    for unit in taken:
        if not free:
            break
        spot = free.pop(0)
        moves.append((unit, round(spot[0] - unit.centre[0]), round(spot[1] - unit.centre[1])))
    return moves, free


def _ink(layer) -> Optional[tuple]:
    """(bbox, solid alpha) of a drawn layer; None when it has no pixels to read."""
    alpha = _alpha(layer)
    if alpha is None or not alpha.size:
        return None
    return tuple(layer.bbox), alpha > INK_ALPHA


def _inks_touch(a: tuple, b: tuple) -> bool:
    (ax0, ay0, _, _), ink_a = a
    (bx0, by0, _, _), ink_b = b
    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax0 + ink_a.shape[1], bx0 + ink_b.shape[1]), min(ay0 + ink_a.shape[0], by0 + ink_b.shape[0])
    if x1 <= x0 or y1 <= y0:
        return False
    return bool(np.any(ink_a[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0] & ink_b[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]))


def _blocked(psd: PSDImage, inks: list, core: tuple, others: list) -> Optional[str]:
    """
    Why a star or icon cannot go where its inks are ((bbox, alpha) on the canvas): its core off the canvas, or its ink on
    the ink of another layer that shows (others: the visible text layers, and the stars or icons it must clear); None.
    """
    slack = _tol(max(core[2] - core[0], core[3] - core[1]))     # the cards' own icons pass the edge a little
    if core[0] < -slack or core[1] < -slack or core[2] > psd.width + slack or core[3] > psd.height + slack:
        return "runs off the canvas"
    for other in others:
        theirs = _ink(other)
        if theirs is not None and any(_inks_touch(mine, theirs) for mine in inks):
            return f"would touch {other.name!r}"
    return None


def _moved_inks(unit: _Unit, dx: int, dy: int) -> list:
    out = []
    for layer in unit.layers:
        for leaf in _drawn(layer):
            ink = _ink(leaf)
            if ink is not None:
                out.append((_moved_box(ink[0], dx, dy), ink[1]))
    return out


def _texts(psd: PSDImage) -> list:
    return [layer for layer in psd.descendants() if _is_text_layer(layer) and layer.is_visible()]


def _copy_name(unit: _Unit, container, taken: set) -> str:
    """Photoshop's name for a copy of the unit's top layer: "<name> copy", "<name> copy 2"... (taken: names given already)."""
    base = sub(r"\s+copy(\s+\d+)?$", "", unit.layers[len(unit.layers) - 1].name)
    names = {layer.name for layer in container.descendants()} | taken
    name, n = f"{base} copy", 2
    while name in names:
        name, n = f"{base} copy {n}", n + 1
    taken.add(name)
    return name


def _shiftable(layer) -> Optional[str]:
    """Why a layer (a group: everything in it) cannot be moved; None when it can."""
    for leaf in [layer, *(layer.descendants() if layer.is_group() else ())]:
        if leaf.is_group():
            continue
        if leaf.kind in ("type", "shape") or leaf.has_vector_mask() or Tag.VECTOR_ORIGINATION_DATA in leaf.tagged_blocks:
            return f"{leaf.name!r} is {'text' if leaf.kind == 'type' else 'a vector shape'}"
        placed = next((leaf.tagged_blocks.get_data(key) for key in PLACED[:2] if key in leaf.tagged_blocks), None)
        if placed is not None and b"filterFX" in placed.data:
            return f"{leaf.name!r} has smart filters"
    return None


def _shift(layer, dx: int, dy: int) -> None:
    """Move a layer (a group: every layer in it) on the canvas: pixels, masks, a smart object's transform, fxrp."""
    if not (dx or dy):
        return
    for leaf in [layer, *(layer.descendants() if layer.is_group() else ())]:
        if not leaf.is_group() and leaf.width and leaf.height:
            leaf.left, leaf.top = leaf.left + dx, leaf.top + dy      # the fork carries fxrp along
        if leaf.has_mask() and leaf.mask.width and leaf.mask.height:
            leaf.mask.offset = (leaf.mask.left + dx, leaf.mask.top + dy)
        for key in PLACED[:2]:
            data = leaf.tagged_blocks.get_data(key) if key in leaf.tagged_blocks else None
            for name in (b"Trnf", b"nonAffineTransform"):
                for i, value in enumerate(data.data.get(name, ()) if data is not None else ()):
                    value.value = float(value.value) + (dx if i % 2 == 0 else dy)
        for key in PLACED[2:]:
            data = leaf.tagged_blocks.get_data(key) if key in leaf.tagged_blocks else None
            if data is not None:
                data.transform = tuple(float(v) + (dx if i % 2 == 0 else dy) for i, v in enumerate(data.transform))
        leaf._psd.mark_updated()


def _clip_chain(layer) -> list:
    """The layers clipped to this one (directly above it, clipping set): its Curves or Levels, which go where it goes."""
    parent, out = layer.parent, []
    index = parent.index(layer) + 1
    while index < len(parent) and parent[index].clipping:
        out.append(parent[index])
        index += 1
    return out


def _with_chains(layers: list) -> list:
    """The layers, each followed by its clip chain (panel order)."""
    return [one for layer in layers for one in [layer, *_clip_chain(layer)]]


def _copy_index(unit: _Unit, container) -> int:
    """Where a copy of the unit goes: above its top layer and the layers clipped to it (none move to the copy)."""
    top = max(unit.layers, key=container.index)
    return container.index(top) + 1 + len(_clip_chain(top))


def _zigzag(units: list, size: tuple) -> Optional[tuple]:
    """
    (first centre, second centre, step) when the stars, top to bottom, alternate between two evenly spaced columns
    (the bloons' and Dark Esau's: right, left, right...); None otherwise.
    """
    order = sorted(units, key=lambda unit: (unit.centre[1], -unit.centre[0]))
    if len(order) < 3:
        return None
    c = [unit.centre for unit in order]
    step, tol = (c[2][0] - c[0][0], c[2][1] - c[0][1]), _tol(max(size))
    along = 1 if abs(step[1]) >= abs(step[0]) else 0      # the columns run down (or the rows across)
    across = abs(c[1][1 - along] - c[0][1 - along])
    if abs(step[1 - along]) > tol or abs(step[along]) < max(size) / 2 or across < max(size) / 2:
        return None
    if not 0.25 <= (c[1][along] - c[0][along]) / step[along] <= 0.75:    # the second sits between the first and third
        return None
    for i, centre in enumerate(c):
        base = c[i % 2]
        if abs(centre[0] - base[0] - i // 2 * step[0]) > tol or abs(centre[1] - base[1] - i // 2 * step[1]) > tol:
            return None
    return c[0], c[1], step


def _known_layout(units: list, size: tuple) -> bool:
    """Whether stars sit in a layout _star_layouts knows (one star, rows of two, a row or column, a zigzag)."""
    return len(units) == 1 or any(found(units, size) is not None for found in (_pyramid, _zigzag)) \
        or _line(units, size, True) is not None


def _star_layouts(order: list, size: tuple, count: int) -> list:
    """
    The layouts the stars could grow to `count` in, as (name, spots in reading order), tried in turn: rows of two (the
    templates'), a row or column carried on, a zigzag carried on; a lone star also as a column (its pair would run into
    the title).
    """
    out = []
    model = _pyramid(order, size)
    if model:
        out.append(("rows of two", _pyramid_spots(model, count)))
    line = _line(order, size, True)
    if line:
        (x, y), (dx, dy) = line
        out.append(("a line", [(x + i * dx, y + i * dy) for i in range(count)]))
    zigzag = _zigzag(order, size)
    if zigzag:
        first, second, (dx, dy) = zigzag
        out.append(("a zigzag", [((first, second)[i % 2][0] + i // 2 * dx, (first, second)[i % 2][1] + i // 2 * dy)
                                 for i in range(count)]))
    if len(order) == 1:
        (x, y), box = order[0].centre, order[0].box
        out.append(("a column", [(x, y + i * (box[3] - box[1])) for i in range(count)]))
    return out


def _stars_blocked(psd: PSDImage, moves: list, clear: list) -> Optional[str]:
    """Why stars cannot move or be copied by [(unit, dx, dy)]: a layer that cannot move or is clipped, or _blocked."""
    for unit, dx, dy in moves:
        why = next((w for w in (_shiftable(layer) for layer in unit.layers) if w), None)
        if why:
            return f"{why}: it cannot move"
        if any(layer.clipping for layer in unit.layers):
            return f"{unit.layers[0].name!r} is clipped to the layer under it"
        blocked = _blocked(psd, _moved_inks(unit, dx, dy), _moved_box(unit.core, dx, dy), clear)
        if blocked:
            return f"a star at {tuple(int(v) for v in _moved_box(unit.core, dx, dy))} {blocked}"
    return None


def plan_stars(psd: PSDImage, parsed: dict, value: int, parser: Optional[PSDParser] = None) -> IconPlan:
    """
    How to show `value` stars on a card whose stars the parser counts from its PSD (minions, aux items and creatures,
    N.M.E.; elsewhere the "N Stars" folder sets them). In the one Stars group that draws them:

    - a count's own folder ("3 stars" beside "5 stars", the templates' 1-5 layouts): it shows, the others hide;
    - else, in the group holding the stars one by one: fewer hide the last in reading order (rows top to bottom, each
      right to left; the first when only that leaves a layout); more show the card's own hidden stars first, then copy
      the last star layer at the layout's spacing (_star_layouts, the first that stays clear): rows of two with an odd
      one centred under or over them (the templates; the odd one moves aside when a pair fills its row, and back to
      the middle when one goes), a row or column carried on, a zigzag carried on, a lone star's column. A new or moved
      star must stay on the canvas, off the text and type icons, and add exactly one to what the parser counts.

    Args:
        psd: The open card (not changed)
        parsed: parse_card() of it
        value: The stars to show, 1 or more
        parser: offline_parser(): a layer in the Stars group named after a type icon is no star to hide or copy

    Returns:
        IconPlan; raises Refused("unsupported-card", field="stars") when the layout is not one of those
    """
    def refuse(why: str):
        raise Refused("unsupported-card", f"stars: {why}", "stars")
    info, old = parsed.get("info"), parsed["card"].get("stars")
    rel = info.path if info is not None else parsed["card"].get("path", "")
    if not stars_from_psd(rel):
        folder = rel.split("/")[len(rel.split("/")) - 2] if "/" in rel else ""
        if fullmatch(STAR_FOLDER, folder, IGNORECASE):
            refuse(f"the {folder!r} folder sets them (moving the card is ledger S9)")
        refuse(f"the parser reads no stars on a {parsed['card'].get('type') or 'card'} card")
    if value < 1:
        refuse("a card shows at least one")
    groups = [group for group in PSDParser._star_groups(psd) if _star_count(group, {}, {}, {})]
    if len(groups) != 1:
        refuse(f"{len(groups)} Stars groups draw them" if groups else 'no group named "Stars" draws them')
    total = _stars_total(psd)
    if total != old:
        refuse(f"the parser counts {old}, the Stars group draws {total} (stars in the header text)")
    plan, container = IconPlan("stars", old, value), groups[0]
    while True:
        folders: dict = {}
        for child in container:
            found = fullmatch(STAR_FOLDER, child.name, IGNORECASE) if child.is_group() else None
            if found:
                folders.setdefault(int(found.group(1)), []).append(child)
        if folders:
            shown = [f for group in folders.values() for f in group if f.visible]
            target = folders.get(value - total + sum(_star_count(f, {}, {}, {}) for f in shown))
            if target:
                show = {id(f): f is target[0] for group in folders.values() for f in group}
                drawn = _stars_total(psd, show)
                if drawn == value:
                    plan.show = [] if target[0].visible else [target[0]]
                    plan.hide = [f for f in shown if f is not target[0]]
                    plan.how = f"folder {target[0].name!r}"
                    return plan
                if len(shown) != 1:
                    refuse(f"the folder {target[0].name!r} would make {drawn}")
            elif len(shown) != 1:
                refuse(f"no folder for {value} in {container.name!r}")
            container = shown[0]                # the shown folder's stars, one by one (a '3 stars' drawing 4)
            continue
        shown, hidden = _star_units(container, parser.all_types if parser is not None else ())
        if len(shown) == 1 and shown[0].count > 1 and not hidden:
            container = shown[0].layers[0]
            continue
        break
    if not shown:
        refuse(f"no star shows in {container.name!r}")
    many = next((unit for unit in shown if unit.count != 1), None)
    if many is not None:
        refuse(f"{many.layers[0].name!r} draws {many.count} in one group")
    count = len(shown) + value - total
    if count < 1:
        refuse(f"{container.name!r} draws {len(shown)}: the other {total - len(shown)} are elsewhere")
    size = _median_size(shown)
    clear = _texts(psd) + badge_layers(psd)[1]          # a star stays off the text and the type icons
    show: dict = {}
    if count < len(shown):                  # fewer: hide the last in reading order (or the first, when that leaves a
        order = _reading(shown, size[1], True)      # layout and the last does not); rows of two centre an odd one
        drop = len(shown) - count
        hide = next((gone for kept, gone in ((order[:count], order[count:]), (order[drop:], order[:drop]))
                     if _known_layout(kept, size)), order[count:])
        plan.hide = [layer for unit in hide for layer in unit.layers]
        plan.how = f"{len(hide)} hidden"
        show.update({id(layer): False for layer in plan.hide})
        model = _pyramid(order, size)
        kept = [unit for unit in order if unit not in hide]
        moves = _place(kept, _pyramid_spots(model, count), size)[0] if model and not _known_layout(kept, size) else []
        if moves and _stars_blocked(psd, moves, clear) is None:
            plan.moves, plan.how = moves, "rows of two"
    else:                                   # more: the card's own hidden stars, then the layout's spacing
        for unit in _reading(hidden, size[1], True):
            if len(shown) == count:
                break
            trial = {**show, **{id(layer): True for layer in unit.layers}}
            if _stars_total(psd, trial) == _stars_total(psd, show) + 1 and _stars_blocked(psd, [(unit, 0, 0)], clear) is None:
                show, shown = trial, shown + [unit]
                plan.show += unit.layers
        plan.how = "hidden stars shown" if plan.show else ""
        if len(shown) < count:
            order = _reading(shown, size[1], True)
            source = max(order, key=lambda unit: max(container.index(layer) for layer in unit.layers))   # the last star layer
            first = None
            for how, spots in _star_layouts(order, size, count):
                moves, free = _place(order, spots, size)
                copies = [(source, round(spot[0] - source.centre[0]), round(spot[1] - source.centre[1]), "") for spot in free]
                why = _stars_blocked(psd, moves + [(unit, dx, dy) for unit, dx, dy, _ in copies], clear)
                if why is None:
                    plan.moves, plan.copies, plan.how = moves, copies, how
                    break
                first = first or why
            else:
                refuse(first or f"{len(order)} stars laid out as no row, column or rows of two")
    boxes: dict = {}
    counts: dict = {}
    for unit, dx, dy in plan.moves:
        if unit.layers[0].is_group():
            continue
        show.update({id(layer): False for layer in unit.layers})
        boxes.setdefault(id(container), []).extend(_moved_box(PSDParser._core_box(layer), dx, dy) for layer in unit.layers)
    for unit, dx, dy, _ in plan.copies:
        if unit.layers[0].is_group():
            counts[id(container)] = counts.get(id(container), 0) + 1
        else:
            boxes.setdefault(id(container), []).extend(_moved_box(PSDParser._core_box(layer), dx, dy) for layer in unit.layers)
    drawn = _stars_total(psd, show, boxes, counts)
    if drawn != value:
        refuse(f"the new layout would count {drawn}")
    plan.moves = [(layer, dx, dy) for unit, dx, dy in plan.moves for layer in _with_chains(unit.layers)]
    taken: set = set()
    plan.copies = [(unit, dx, dy, _copy_name(unit, container, taken)) for unit, dx, dy, _ in plan.copies]
    return plan


def canonical_type(parser: Optional[PSDParser], name: str) -> str:
    """A type's vocabulary name ("Fragile Equip", "fragile_equip" -> one spelling); lowercase without a vocabulary."""
    hit = parser.icons.lookup(str(name)) if parser is not None else None
    return hit[0] if hit else str(name).lower()


def same_types(parser: Optional[PSDParser], a, b) -> bool:
    """Whether two type lists name the same icons, in any order."""
    return Counter(canonical_type(parser, t) for t in a or []) == Counter(canonical_type(parser, t) for t in b or [])


def type_row(psd: PSDImage, parser: PSDParser) -> tuple:
    """
    The header type icons the parser reads: (image layers [(canonical name, layer)], icon-font text layers).

    An image layer counts as PSDParser._process_layer counts it: named after a Types/ icon, visible, with pixels, its
    top above the midline, and not under a stat row's group.
    """
    mid = int(psd.height * TYPE_REGION_RATIO)
    images, tokens = [], []
    for layer in psd.descendants():
        if EFFECT_LAYER_TAG in layer.name and not layer.is_group():
            continue
        if _is_text_layer(layer):
            if (layer.name.strip().lower() != "ability" and layer.is_visible() and layer.bbox[1] < mid
                    and PSDParser._icon_font_spans(layer, layer.engine_dict["Editor"]["Text"].value)):
                tokens.append(layer)
        elif (not parser._is_stat_layer(layer) and layer.name.lower() in parser.all_types and layer.is_visible()
              and layer.has_pixels() and layer.bbox[1] < mid):
            images.append((canonical_type(parser, layer.name), layer))
    return images, tokens


def _icon_reason(psd: PSDImage, layer) -> Optional[str]:
    """Why a header type layer is no icon the bake moves or hides (a group, a frame round the card, text); None."""
    if layer.is_group():
        return "a group"
    if layer.kind not in ("pixel", "smartobject"):
        return f"a {layer.kind} layer"
    x0, y0, x1, y1 = layer.bbox
    if (x1 - x0) * (y1 - y0) > ICON_FRAME * psd.width * psd.height:
        return "a frame round the card"
    return None


def _icon_spots(psd: PSDImage, slots: list, extra: int, clear: list) -> list:
    """
    Boxes for `extra` icons past the row's last: the row carried on at its spacing (under a lone icon: Item_Template's
    column), else, when those boxes would touch the text or stars, a second row under the first icon (a row running
    into the title) or a lone icon's right; the first whose boxes stay clear, else the row carried on.
    """
    first, last = slots[0].core, slots[len(slots) - 1].core
    gap = round(ICON_GAP * psd.width / CARD_WIDTH)
    if len(slots) > 1:
        (x0, y0), (x1, y1) = slots[len(slots) - 2].centre, slots[len(slots) - 1].centre
        step = (round(x1 - x0), round(y1 - y0))
    else:
        step = (0, last[3] - last[1] + gap)
    options = [[_moved_box(last, step[0] * k, step[1] * k) for k in range(1, extra + 1)]]
    if len(slots) > 1 and abs(step[0]) > abs(step[1]):
        down = max(unit.core[3] for unit in slots) - first[1] + gap
        options.append([_moved_box(first, step[0] * k, down) for k in range(extra)])
    elif len(slots) == 1:
        options.append([_moved_box(last, (last[2] - last[0] + gap) * k, 0) for k in range(1, extra + 1)])
    for boxes in options:
        if all(_blocked(psd, [((box[0], box[1], box[2], box[3]), np.ones((box[3] - box[1], box[2] - box[0]), bool))],
                        box, clear) is None for box in boxes):
            return boxes
    return options[0]


def _fits(core: tuple, spot: tuple) -> bool:
    """Whether a hidden icon's core is the spot's size (its longer side within LAYOUT_TOL twice over)."""
    mine, theirs = max(core[2] - core[0], core[3] - core[1]), max(spot[2] - spot[0], spot[3] - spot[1])
    return abs(mine - theirs) <= 2 * LAYOUT_TOL * theirs


def _in_stars(layer) -> bool:
    return any("stars" in owner.name.lower() for owner in [layer, *layer_parents(layer)])


def layer_parents(layer) -> list:
    """The layer's groups, innermost first."""
    out, parent = [], layer.parent
    while parent is not None and not isinstance(parent, PSDImage):
        out.append(parent)
        parent = parent.parent
    return out


def plan_types(psd: PSDImage, parsed: dict, value: list, parser: Optional[PSDParser]) -> IconPlan:
    """
    How to show the type icons `value` in the card's header row (Types/ icons as image layers, the parser's rule):
    the icons the edit keeps stay where they are, a removed one hides, and the row closes up: a kept icon past the new
    end moves into the first free spot, centre on centre; a new icon takes the next free spot, from the card's own
    hidden layer of that type (the templates keep one per type) or made from its Types/ PNG, fitted to the spot and
    styled like the icon it replaces (else the row's last). Spots past the last icon carry the row on at its spacing,
    or go under a lone icon (Item_Template's column), or wrap clear of the title (_icon_spots). Refused per field: a
    token row ([fire] in header text), a removed type drawn as a frame or group, no icon to line a new one up with, a
    type with no Types/ icon, a layer that cannot move, or an icon that would run off the canvas or onto the text or
    stars.

    Args:
        psd: The open card (not changed)
        parsed: parse_card() of it
        value: The type names to show (any spelling the icon vocabulary knows)
        parser: offline_parser() (its Types/ names and vocabulary)

    Returns:
        IconPlan (icons: Types/ stems to make at a box, styled after a reference layer)
    """
    def refuse(why: str):
        raise Refused("unsupported-card", f"types: {why}", "types")
    if parser is None:
        refuse("no icon vocabulary")
    want = []
    for name in value:
        hit = parser.icons.lookup(str(name))
        if hit is None or hit[1] != "type":
            refuse(f"{name!r} is no type icon")
        want.append(hit[0])
    images, tokens = type_row(psd, parser)
    if tokens:
        refuse(f"printed as icon tokens in {tokens[0].name!r}: edit them in Photoshop")
    plan = IconPlan("types", [name for name, _ in images], value)
    reasons = [_icon_reason(psd, layer) for _, layer in images]
    left = Counter(want)
    for (name, layer), reason in zip(images, reasons):
        if reason and not left[name]:
            refuse(f"{layer.name!r} is {reason}, not an icon to take off")
        left[name] -= 1 if reason else 0
    row = [(name, layer) for (name, layer), reason in zip(images, reasons) if not reason]
    icons = [_Unit([layer], _core(layer) or tuple(layer.bbox)) for _, layer in row]
    names = {id(unit): name for unit, (name, _) in zip(icons, row)}
    size = _median_size(icons) if icons else (0, 0)
    slots = _reading(icons, size[1], False)
    kept, removed = [], []
    for unit in slots:
        if left[names[id(unit)]]:
            left[names[id(unit)]] -= 1
            kept.append(unit)
        else:
            removed.append(unit)
    added = []
    for name in want:
        if left[name] > 0:
            added.append(name)
            left[name] -= 1
    count = len(kept) + len(added)
    spots = [unit.core for unit in slots[:count]]
    if count > len(slots):
        if not slots:
            refuse("no type icon on the card to line a new one up with")
        spots += _icon_spots(psd, slots, count - len(slots), _texts(psd) + badge_layers(psd)[0])
    holder = {i: unit for i, unit in enumerate(slots[:count]) if unit in kept}
    free = [i for i in range(count) if i not in holder]
    for unit in [unit for unit in kept if slots.index(unit) >= count]:
        i = free.pop(0)
        (cx, cy), (sx, sy) = unit.centre, _centre(spots[i])
        plan.moves.append((unit.layers[0], round(sx - cx), round(sy - cy)))
        holder[i] = unit
    plan.hide = [layer for unit in removed for layer in unit.layers]
    stars = badge_layers(psd)[0]
    texts = _texts(psd)
    library = [layer for layer in psd.descendants() if not layer.is_group() and not layer.visible
               and layer.kind in ("pixel", "smartobject") and layer.name.lower() in parser.all_types and layer.has_pixels()
               and all(parent.visible for parent in layer_parents(layer)) and not _in_stars(layer)]
    for name, i in zip(added, free):
        spot = spots[i]
        replaced = slots[i] if i < len(slots) and slots[i] in removed else None
        reference = (replaced or (kept[len(kept) - 1] if kept else slots[len(slots) - 1])).layers[0]
        own = next((layer for layer in library if canonical_type(parser, layer.name) == name and not _shiftable(layer)
                    and _fits(_core(layer) or tuple(layer.bbox), spot)), None)
        if own is not None:
            library.remove(own)
            core = _core(own) or tuple(own.bbox)
            (cx, cy), (sx, sy) = _centre(core), _centre(spot)
            plan.show.append(own)
            plan.moves.append((own, round(sx - cx), round(sy - cy)))
            continue
        stem = next((t for t in parser.all_types if canonical_type(parser, t) == name), None)
        if stem is None:
            refuse(f"no Types/ icon for {name!r}")
        plan.icons.append((stem, spot, reference))
    for layer, dx, dy in plan.moves:
        why = _shiftable(layer)
        if why:
            refuse(f"{why}: it cannot move")
        unit = _Unit([layer], _core(layer) or tuple(layer.bbox))
        others = [other for other in texts + stars if other is not layer]
        blocked = _blocked(psd, _moved_inks(unit, dx, dy), _moved_box(unit.core, dx, dy), others)
        if blocked:
            refuse(f"{layer.name!r} at {tuple(int(v) for v in _moved_box(unit.core, dx, dy))} {blocked}")
    for _, spot, _ in plan.icons:
        slack = _tol(max(spot[2] - spot[0], spot[3] - spot[1]))
        if spot[0] < -slack or spot[1] < -slack or spot[2] > psd.width + slack or spot[3] > psd.height + slack:
            refuse(f"a new icon at {tuple(int(v) for v in spot)} runs off the canvas")
    moved = len(plan.moves)
    plan.moves = [(one, dx, dy) for layer, dx, dy in plan.moves for one in _with_chains([layer])]
    plan.how = ", ".join(part for part in (f"{len(plan.hide)} hidden" if plan.hide else "",
                                           f"{moved} moved" if moved else "",
                                           f"{len(plan.icons)} made from Types/" if plan.icons else "") if part)
    return plan


def icon_engine() -> bool:
    """Does this psd-tools copy, move and make layers (the fork's duplicate, copy_effects_from, fromarray)."""
    from psd_tools.api.layers import Layer, PixelLayer
    return all(hasattr(Layer, name) for name in ("duplicate", "copy_effects_from")) and hasattr(PixelLayer, "fromarray")


def _icon_png(types_dir: str, stem: str) -> str:
    """The Types/ PNG of a stem (any case)."""
    from os import listdir
    for name in listdir(types_dir):
        if name.lower() == f"{stem}.png".lower():
            return join(types_dir, name)
    raise Refused("unsupported-card", f"types: no {stem}.png in Types/", "types")


def _make_icon(psd: PSDImage, types_dir: str, stem: str, box: tuple, reference):
    """
    A pixel layer of the Types/ icon fitted into the box (its opaque part, aspect kept, centred), the reference's
    effects, blending and opacity copied, directly above the reference.
    """
    from psd_tools.api.layers import PixelLayer
    path = _icon_png(types_dir, stem)
    image = Image.open(path).convert("RGBA")
    solid = image.getchannel("A").point(lambda a: 255 if a >= 128 else 0).getbbox()
    if solid is None:
        raise Refused("unsupported-card", f"types: {basename(path)} draws nothing", "types")
    image = image.crop(solid)
    scale = min((box[2] - box[0]) / image.width, (box[3] - box[1]) / image.height)
    width, height = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    image = image.convert("RGBa").resize((width, height), Image.LANCZOS).convert("RGBA")
    cx, cy = _centre(box)
    name = basename(path)[:len(basename(path)) - 4]
    layer = PixelLayer.fromarray(np.asarray(image), psd, name=name, top=round(cy - height / 2), left=round(cx - width / 2))
    layer.name = name
    layer.copy_effects_from(reference)
    layer.blend_mode, layer.opacity = reference.blend_mode, reference.opacity
    _issue_id(psd, layer)
    chain = _clip_chain(reference)
    parent = reference.parent
    layer.move_to(parent, index=parent.index(reference) + 1 + len(chain))
    return layer


def _copy_chain(reference, layer, done: list) -> list:
    """The reference icon's clip chain (its Curves...) copied onto a new icon, moved as far as the icon is from it."""
    (rx, ry), (lx, ly) = _centre(_core(reference) or tuple(reference.bbox)), _centre(_core(layer) or tuple(layer.bbox))
    dx, dy = round(lx - rx), round(ly - ry)
    parent, out = layer.parent, []
    for k, clipped in enumerate(_clip_chain(reference)):
        copy = clipped.duplicate(parent, index=parent.index(layer) + 1 + k)
        _shift(copy, dx, dy)
        done.append(("copy", clipped, copy, dx, dy))
        out.append(copy)
    return out


def _extent(layer) -> tuple:
    """The layer's bbox; a group's, its drawn leaves' (psd-tools gives a hidden group none)."""
    if not layer.is_group():
        return tuple(layer.bbox)
    return union_box([tuple(leaf.bbox) for leaf in _drawn(layer)]) or (0, 0, 0, 0)


def _reach(layer) -> int:
    """fx_pad of the layer, or of the furthest-reaching layer in a group (a star's glow inside its group)."""
    return max(fx_pad(leaf) for leaf in [layer, *(layer.descendants() if layer.is_group() else ())])


def _subtree_records(layer) -> list:
    """The layer's records in file order: a group's divider, its layers', its own."""
    if not layer.is_group():
        return [layer._record]
    out = [layer._bounding_record] if layer._bounding_record is not None else []
    for child in layer:
        out += _subtree_records(child)
    return out + [layer._record]


def record_map(psd: PSDImage) -> dict:
    """id(layer record) -> its index in the file (what layer_records() reads back)."""
    records = psd._record.layer_and_mask_information.layer_info.layer_records
    return {id(record): i for i, record in enumerate(records)}


def set_icons(psd: PSDImage, plan: IconPlan, types_dir: Optional[str] = None) -> Edit:
    """
    Apply a plan_stars() or plan_types() plan to the open card (the same file it was planned on): hide, show, move,
    copy and make layers. Record indices come later (index_icons), once every edit is in.

    Returns:
        Edit over the layers it touched, its region their old and new boxes padded by their effects; raises Refused
        (field set) for a type icon that lands on the text or stars
    """
    boxes = [(_extent(layer), layer) for layer in plan.hide + plan.show]
    boxes += [(_extent(layer), layer) for layer, _, _ in plan.moves]
    for unit, dx, dy, name in plan.copies:     # copies first: from where the star is now, with its clipped layers
        parent = unit.layers[0].parent
        index = _copy_index(unit, parent)
        top = max(unit.layers, key=parent.index)
        for k, layer in enumerate(_with_chains(sorted(unit.layers, key=parent.index))):
            copy = layer.duplicate(parent, index=index + k, name=name if layer is top else None)
            _shift(copy, dx, dy)
            plan.done.append(("copy", layer, copy, dx, dy))
            boxes.append((_extent(copy), copy))
    for layer in plan.hide:
        layer.visible = False
    for layer in plan.show:
        layer.visible = True
    for layer, dx, dy in plan.moves:
        _shift(layer, dx, dy)
        boxes.append((_extent(layer), layer))
    stars, texts = badge_layers(psd)[0], _texts(psd)
    for stem, box, reference in plan.icons:
        if types_dir is None:
            raise Refused("unsupported-card", "types: no Types/ icons to make one from", "types")
        layer = _make_icon(psd, types_dir, stem, box, reference)
        others = [other for other in texts + stars if other is not layer]
        blocked = _blocked(psd, [_ink(layer)], _core(layer) or tuple(layer.bbox), others)
        if blocked:
            raise Refused("unsupported-card", f"types: the {layer.name!r} icon {blocked}", "types")
        blocks = {bytes(key): layer.tagged_blocks[key].tobytes() for key in EFFECT_BLOCKS if key in layer.tagged_blocks}
        plan.done.append(("made", layer, blocks))
        boxes.append((tuple(layer.bbox), layer))
        _copy_chain(reference, layer, plan.done)
    region = union_box([_pad(box, _reach(layer), psd) for box, layer in boxes if box[2] > box[0] and box[3] > box[1]])
    layers = plan.hide + plan.show + [layer for layer, _, _ in plan.moves] + [done[2] if done[0] == "copy" else done[1]
                                                                             for done in plan.done]
    return Edit(plan.field, plan.old, plan.new, layers, [], region, icons=plan)


def index_icons(edit: Edit, before: dict, after: dict) -> None:
    """
    Fill an icon edit's record indices once every edit is applied: before = record_map() of the card as opened (the
    records keep their identity through the edits), after = record_map() now (what the saved file holds).
    """
    plan = edit.icons
    edit.toggled = sorted(after[id(layer._record)] for layer in plan.hide + plan.show)
    edit.shifted = [(after[id(record)], before[id(record)], dx, dy) for layer, dx, dy in plan.moves
                    for record in _subtree_records(layer)]
    edit.copied = [(after[id(new)], before[id(old)], dx, dy) for done in plan.done if done[0] == "copy"
                   for old, new, dx, dy in [(o, n, done[3], done[4]) for o, n in zip(_subtree_records(done[1]),
                                                                                     _subtree_records(done[2]))]]
    edit.made = [(after[id(done[1]._record)], done[2]) for done in plan.done if done[0] == "made"]


def render_all(psd: PSDImage) -> np.ndarray:
    """The whole card rendered from its layers (no ICC, no preview), uint8 RGBA: gate_outside's before and after."""
    return np.asarray(psd.composite(viewport=(0, 0, psd.width, psd.height), ignore_preview=True,
                                    apply_icc=False).convert("RGBA"))


def gate_outside(before: np.ndarray, psd: PSDImage, edit: Edit) -> np.ndarray:
    """
    A stars or types edit changes the card's render only inside its region: the whole card rendered after it differs
    from the render before it nowhere else (a star's clipped Curves left on another layer, an adjustment in a copied
    pass-through group would). Run after each such edit, before the text and stat edits.

    Args:
        before: render_all() of the card before the edit
        psd: The card with the edit applied
        edit: The edit

    Returns:
        render_all() now (the next edit's before); raises Refused("region", field: the edit's) for any pixel outside
    """
    after = render_all(psd)
    diff = (before != after).any(axis=2)
    if edit.region:
        x0, y0, x1, y1 = edit.region
        diff[y0:y1, x0:x1] = False
    outside = int(diff.sum())
    if outside:
        raise Refused("region", f"{edit.field}: the card changes {outside} pixels outside the edited box", edit.field)
    return after


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


def text_blends(psd: PSDImage) -> list:
    """The TEXT_BLENDS keys the fork's psd.composite() takes, in order ("preview", its default, always)."""
    parameters = signature(psd.composite).parameters
    return [text for text, kwargs in TEXT_BLENDS.items() if all(name in parameters for name in kwargs)]


def text_kwargs(psd: PSDImage, text: str, edited: tuple = ()) -> dict:
    """
    psd.composite()'s keywords for a TEXT_BLENDS key. Under "preview" the edited type layers are pinned to the text
    gamma: the preview still shows their old text, and Photoshop blends a layer it redraws in the gamma.

    Args:
        psd: The card
        text: A TEXT_BLENDS key
        edited: Layer IDs of the edited layers

    Returns:
        The keywords
    """
    kwargs = dict(TEXT_BLENDS[text])
    parameters = signature(psd.composite).parameters
    if text == "preview" and edited and "layer_text_gamma" in parameters:
        kwargs["layer_text_gamma"] = dict.fromkeys(edited, parameters["text_gamma"].default)
    return kwargs


def render_region(psd: PSDImage, region: tuple, icc: bool = True, text: str = "preview",
                  edited: tuple = ()) -> Image.Image:
    """
    psd.composite over a viewport, never from the stored preview and never forced (section 15).

    Args:
        psd: The card
        region: Canvas bbox to render
        icc: Convert by the embedded profile (export_icc(): how the card's PNG was exported)
        text: A TEXT_BLENDS key (baseline()'s choice)
        edited: text_kwargs()'s

    Returns:
        The render, region-sized
    """
    return psd.composite(viewport=region, ignore_preview=True, apply_icc=icc, **text_kwargs(psd, text, edited))


def render_pair(psd: PSDImage, region: tuple, icc: bool = True, text: str = "preview", edited: tuple = ()) -> tuple:
    """
    One render of the region, as render_region() returns it and in the PSD's own values (what its preview stores).

    Args:
        psd: The card
        region: Canvas bbox to render
        icc: render_region()'s
        text: render_region()'s
        edited: render_region()'s

    Returns:
        (render, raw): raw before the embedded profile's sRGB conversion; the same image without a profile or icc
    """
    raw = psd.composite(viewport=region, ignore_preview=True, apply_icc=False, **text_kwargs(psd, text, edited))
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
        (render, measurement); the measurement's "text" is the first text_blends() key whose render is within the
        bar, else the preview blend's (a miss of the bar bakes, told: miss_notes; apply_jobs refuses a miss whose edit
        renders no change). Raises Refused("parity") past PARITY_FLOOR only.
    """
    first = None
    for text in text_blends(psd):
        render = render_region(psd, region, icc, text)
        measured = _measure(render, png.crop(region))
        measured["text"] = text
        if _within_bar(measured):
            return render, measured
        first = first or (render, measured)
    render, measured = first
    if _past_floor(measured):
        raise Refused("parity", "baseline bad %.2f%% mean %.2f max %d: past the %g%% floor"
                      % (measured["bad_pct"], measured["mean"], measured["max"], PARITY_FLOOR))
    return render, measured


def miss_notes(baked) -> list:
    """
    The note per edited box that missed the parity bar under the floor (owner, 2026-10-23): "name baked with a pixel
    difference: bad 1.97%, mean 1.60, max 217"; the re-render's numbers when only it missed.
    """
    out = []
    for name, measured in baked.parity.items():
        again = getattr(baked, "rerender", {}).get(name)
        miss = measured if not _within_bar(measured) else again if again and not _within_bar(again) else None
        if miss:
            out.append("%s baked with a pixel difference%s: bad %.2f%%, mean %.2f, max %d" % (
                name, "" if miss is measured else " (re-render)", miss["bad_pct"], miss["mean"], miss["max"]))
    return out


def own_pixels(replaced: np.ndarray, region: tuple, others: list) -> np.ndarray:
    """
    The merge's replaced pixels (region-sized) that no other edit's region reaches: what this edit drew (another edit's
    change inside an overlapping region is not this one's); all of them when the others cover the whole region.
    """
    x0, y0, x1, y1 = region
    free = np.ones(replaced.shape, bool)
    for ox0, oy0, ox1, oy1 in others:
        free[max(oy0 - y0, 0):max(oy1 - y0, 0), max(ox0 - x0, 0):max(ox1 - x0, 0)] = False
    return replaced & free if free.any() else replaced


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
        mask = record.mask_data
        out.append({"name": blocks.get_data(Tag.UNICODE_LAYER_NAME, record.name) if blocks else record.name,
                    "flags": record.flags.tobytes(),
                    "rect": (record.top, record.left, record.bottom, record.right),
                    "channels": tuple((int(info.id), info.length) for info in record.channel_info),
                    "mask": mask.tobytes() if mask is not None else b"",
                    "blend": (record.blend_mode, record.opacity, record.clipping),
                    "blocks": tuple((bytes(key), block.tobytes()) for key, block in (record.tagged_blocks or {}).items()),
                    **(_mask_parts(mask) if mask is not None else {})})
    return out


MASK_RECT = ("top", "left", "bottom", "right", "real_top", "real_left", "real_bottom", "real_right")


def _mask_parts(mask) -> dict:
    """A mask's rectangles ("mask_box": MASK_RECT) and its bytes with them zeroed ("mask_norm", what a move keeps)."""
    box = tuple(getattr(mask, name, None) for name in MASK_RECT)
    norm = deepcopy(mask)
    for name, value in zip(MASK_RECT, box):
        if value is not None:
            setattr(norm, name, 0)
    return {"mask_box": box, "mask_norm": norm.tobytes()}


def _moved_rect(rect: tuple, dx: int, dy: int) -> tuple:
    """A record's (top, left, bottom, right) moved; an empty one stays (a group's, an adjustment's)."""
    top, left, bottom, right = rect
    return rect if bottom <= top or right <= left else (top + dy, left + dx, bottom + dy, right + dx)


def _moved_mask(box, dx: int, dy: int):
    """A mask_box (MASK_RECT) moved: each of its two rectangles that is there and not empty."""
    if box is None:
        return None
    out: list = []
    for i in (0, 4):
        part = box[i:i + 4]
        out += list(part) if part[0] is None else list(_moved_rect(part, dx, dy))
    return tuple(out)


def _record_moved(a: dict, b: dict, dx: int, dy: int, keys: set, flags: bool = False, renamed: bool = False) -> Optional[str]:
    """
    Why record b is not record a moved by (dx, dy) (a layer _shift moved, or a copy of it); None when it is: the rect
    and mask moved, the channels, blending and other blocks the same, only `keys` blocks changed; flags: one flag
    byte may differ (the layer was shown or hidden too); renamed: the name may differ (a copy's).
    """
    if b["rect"] != _moved_rect(a["rect"], dx, dy):
        return f"rect {a['rect']} -> {b['rect']}"
    if a["channels"] != b["channels"] or a["blend"] != b["blend"]:
        return "channels or blending"
    if not renamed and a["name"] != b["name"]:
        return "name"
    if a.get("mask_norm") != b.get("mask_norm") or _moved_mask(a.get("mask_box"), dx, dy) != b.get("mask_box"):
        return "mask"
    differing = sum(1 for x, y in zip(a["flags"], b["flags"]) if x != y)
    if differing > (1 if flags else 0):
        return f"{differing} flag bytes"
    before, after = dict(a["blocks"]), dict(b["blocks"])
    changed = sorted(key.decode("latin-1") for key in set(before) | set(after)
                     if before.get(key) != after.get(key) and key not in keys)
    return f"blocks {', '.join(changed)}" if changed else None


def _pixel_record(record: dict) -> bool:
    """A plain pixel record: a transparency channel then the color channels in order."""
    ids = [channel for channel, _ in record["channels"]]
    return bool(ids) and ids[0] == -1 and ids[1:] == list(range(len(ids) - 1))


def gate_bytes(old_path: str, new_path: str, edits: list) -> tuple:
    """
    Compare the layer records of the old and new PSD: only the edited layers change, and only in the fields their
    kind of edit touches (a toggle: exactly one flag byte per toggled digit, plus the effect blocks its Edit declares
    moved, the same bytes leaving one digit's record and landing on another's, or leaving only when cleared; a text
    edit: rect, channels, EngineData, and the declared name of a title named after its text (Edit.renamed); its FX
    layers: exactly the declared records gone and come, the new ones pixel records carrying the chart's effect blocks
    byte for byte; a hidden overlay word layer: one flag byte; a star or type icon shown or hidden: one flag byte; one
    moved: its rect and mask moved by the declared offset, only its transform and anchor blocks changed (_record_moved);
    a star's copy: the source's record moved, a new id and name; a type icon from Types/: a pixel record carrying the
    reference icon's effect blocks). Records are paired in file order once the FX records declared removed and added
    and the copies and icons declared added are taken out. A refusal over a stars or types edit's record carries its
    field.

    Returns:
        ([(index, name, changed fields)] in file order ("removed" at the old index, "added" at the new one), differing
        flag bytes, effect block bytes that landed on a digit, an FX layer or a type icon)
    """
    old, new = layer_records(old_path), layer_records(new_path)
    dropped = Counter(name for edit in edits for name in edit.dropped)
    added = {name: blocks for edit in edits for name, blocks in edit.fx}
    allowed, names = {}, {}
    for edit in edits:
        kind = "stat" if edit.field in STAT_FIELDS else "text"
        for index in edit.records:
            allowed[index] = (kind, ALLOWED_RECORD_FIELDS[kind] | ({"name"} if edit.renamed else set()))
            names[index] = edit.renamed
        for index in edit.hidden:
            allowed[index] = ("overlay", ALLOWED_RECORD_FIELDS["overlay"])
    toggles = {index for edit in edits for index in edit.toggled}
    allowed.update({index: ("toggle", ALLOWED_RECORD_FIELDS["toggle"]) for index in toggles})
    moved = {index: (source, dx, dy) for edit in edits for index, source, dx, dy in edit.shifted}
    inserted = {index: ("copy", source, dx, dy) for edit in edits for index, source, dx, dy in edit.copied}
    inserted.update({index: ("made", blocks) for edit in edits for index, blocks in edit.made})
    owner = {index: edit.field for edit in edits
             for index in [*edit.toggled, *(record[0] for record in edit.shifted + edit.copied + edit.made)]}
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
        if b is not None and j in inserted:     # a star's copy, or a type icon made from Types/
            spec = inserted.pop(j)
            if spec[0] == "made":
                blocks = dict(b["blocks"])
                if not _pixel_record(b) or any(blocks.get(key) != data for key, data in spec[1].items()):
                    raise Refused("bytes", f"type icon {b['name']!r} is not the declared pixel layer", owner.get(j))
                block_bytes += sum(len(data) for data in spec[1].values())
            else:
                why = _record_moved(old[spec[1]], b, spec[2], spec[3], COPY_BLOCKS, renamed=True)
                if why:
                    raise Refused("bytes", f"layer {j} {b['name']!r} is no copy of {old[spec[1]]['name']!r} moved: {why}",
                                  owner.get(j))
            changed.append((j, b["name"], ["added"]))
            j += 1
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
        fields = sorted(key for key in a if a.get(key) != b.get(key))
        if j in moved:                          # a star or type icon moved: its rect, mask, transform and anchor
            source, dx, dy = moved[j]
            why = f"paired with old layer {i}" if source != i else _record_moved(a, b, dx, dy, MOVE_BLOCKS, j in toggles)
            if why:
                raise Refused("bytes", f"moved layer {j} {a['name']!r}: {why}", owner.get(j))
            flag_bytes += sum(1 for x, y in zip(a["flags"], b["flags"]) if x != y)
            if fields:
                changed.append((j, a["name"], fields))
        elif fields:
            kind, ok = allowed.get(j, ("", set()))
            if not set(fields) <= ok:
                raise Refused("bytes", f"layer {j} {a['name']!r} changed {', '.join(fields)}")
            if "name" in fields and b["name"] != names.get(j):
                raise Refused("bytes", f"layer {j} {a['name']!r} renamed {b['name']!r}, not {names.get(j)!r}")
            if "flags" in fields:
                differing = sum(1 for x, y in zip(a["flags"], b["flags"]) if x != y)
                if kind in ("stat", "overlay", "toggle") and differing != 1:
                    raise Refused("bytes", f"{kind} {a['name']!r} changed {differing} flag bytes", owner.get(j))
                flag_bytes += differing
            changed.append((j, a["name"], fields))
        i += 1
        j += 1
    if any(dropped.values()) or added or inserted:
        raise Refused("bytes", "an FX layer, star copy or type icon declared removed or added is not")
    hidden = {index for edit in edits for index in edit.hidden}
    toggled = {index for edit in edits if edit.field in STAT_FIELDS for index in edit.records}
    if (toggled | hidden | toggles) - {index for index, _, fields in changed if fields != ["removed"]}:
        raise Refused("bytes", "a toggled digit, hidden overlay or shown or hidden star or icon did not change")
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
    Re-parse the new PSD: the old parse with exactly the edited fields replaced (the stars read back as the edit, the
    types as its icons in any order: a miss names the field), no new problems (a stat above 10
    reads STATS TOO HIGH by design when it sits on Top and Bottom rows; no EFFECT LAYER problem: the validator's
    coverage check is the FX layers' gate), one digit per edited row, the title read back for a name edit, and an
    ability edit's FX layers back as declared: visible, styled, directly above the ability's top-level group (_top), no
    overlay word layer showing.

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
            layer = ability_layer(psd)
            group = _top(layer)
            found = fx_layers(psd, layer)
            if [fx.name for fx in found] != [name for name, _ in edit.fx]:
                raise Refused("reparse", f"FX layers after reopen: {[fx.name for fx in found]}")
            for k, fx in enumerate(found):
                if (fx.parent is not psd or psd.index(fx) != psd.index(group) + 1 + k or not fx.is_visible()
                        or not _enabled_effects(fx, strict=True)):
                    raise Refused("reparse", f"{fx.name!r} is not a visible styled layer directly above {group.name!r}")
            if overlay_words(psd):
                raise Refused("reparse", "overlay word layers still show over the ability")
    new = parse_card(parser, new_path, rel)
    expected, got = dict(parsed["card"]), dict(new["card"])
    for name, value in jobs.items():
        if name in STAT_FIELDS or name == "stars":
            expected[name] = value
        elif name == "types":                   # in any order: the parser lists them in layer order
            expected[name] = got.get(name) if same_types(parser, got.get(name), value) else value
        elif name == "ability":
            expected["ability"], got["ability"] = normalize_ability(value), normalize_ability(got.get("ability"))
        elif name == "name":                    # the new title is read back below
            expected.pop("display_name", None)
            got.pop("display_name", None)
    expected.pop("problem", None)
    got.pop("problem", None)
    different = sorted(key for key in set(expected) | set(got) if expected.get(key) != got.get(key))
    if different:
        raise Refused("reparse", "fields differ from the expected parse: " + ", ".join(
            f"{key} {got.get(key)!r} (not {expected.get(key)!r})" if key in ICON_FIELDS else key for key in different),
            different[0] if set(different) <= set(ICON_FIELDS) else None)
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
    title = read_name(psd, _card_name(parsed), _own_text(parsed)) if "name" in jobs else None
    if "name" in jobs and title != collapse(jobs["name"]):
        raise Refused("reparse", f"the title reads {title!r}")
    return new


def gate_idempotence(new_parsed: dict, jobs: dict, psd: PSDImage, merged: Image.Image, edits: list,
                     icc: bool = True, texts: Optional[dict] = None, parser: Optional[PSDParser] = None) -> dict:
    """
    Nothing left to bake on the new PSD, and each edit's region re-rendered from it is not broken against the merged PNG
    (PARITY_FLOOR; a miss of the bar is told: the delta merge keeps Photoshop's pixels where the render matched the baseline).

    Args:
        new_parsed: gate_reparse()'s parse of the saved PSD
        jobs: The applied jobs
        psd: The saved PSD, reopened
        merged: The merged PNG
        edits: The applied Edits
        icc: render_region()'s
        texts: {field: render_region() text}; "preview" for a field not in it
        parser: offline_parser(), to read a types edit back (classify)

    Returns:
        {field: region measurement}
    """
    again = classify(new_parsed, jobs, psd, parser)
    left = sorted(set(again.jobs) | {r.field for r in again.refusals if r.field})
    if left:
        raise Refused("idempotence", "still to bake: " + ", ".join(left), left[0] if set(left) <= set(ICON_FIELDS) else None)
    out = {}
    edited = edited_ids(edits)
    for edit in edits:
        text = (texts or {}).get(edit.field, "preview")
        measured = _measure(render_region(psd, edit.region, icc, text, edited), merged.crop(edit.region))
        if _past_floor(measured):
            raise Refused("idempotence", "%s re-render bad %.2f%% mean %.2f: past the %g%% floor"
                          % (edit.field, measured["bad_pct"], measured["mean"], PARITY_FLOOR),
                          edit.field if edit.field in ICON_FIELDS else None)
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
               fonts_dir: str, chart: Optional[dict] = None, update_preview: bool = True,
               types_dir: Optional[str] = None) -> Baked:
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
        types_dir: The card repo's Types/ folder (its PNGs), for a types edit that makes an icon

    Returns:
        Baked; raises Refused (any other exception is the driver's "error"); a refusal of a stars or types edit's own
        gates carries its field, so the driver can bake the other fields without it
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

    # stars and types first (a title edit then clears the new stars), text next: an ability's FX records shift the digits'
    # record indices; the icon edits' indices are read once all are in (index_icons)
    before, edits, shown = record_map(new), [], None
    for name, value in sorted(jobs.items(), key=lambda job: (job[0] not in ICON_FIELDS, job[0] in STAT_FIELDS)):
        if name in ICON_FIELDS:
            try:
                plan = plan_stars(new, parsed, value, parser) if name == "stars" else plan_types(new, parsed, value, parser)
                edit = set_icons(new, plan, types_dir)
                start = perf_counter()
                shown = gate_outside(render_all(old) if shown is None else shown, new, edit)   # nothing else changes
                timings["outside"] = timings.get("outside", 0.0) + perf_counter() - start
            except Refused:
                raise
            except Exception as error:          # a layout the code trips on: that field's alone
                raise Refused("error", f"{name}: {type(error).__name__}: {error}", name) from error
            edits.append(edit)
        elif name in STAT_FIELDS:
            edits.append(set_stat(new, parsed, name, value))
        elif name in TEXT_FIELDS:
            edits.append(set_text_layer(new, name, value, fonts_dir, chart, _card_name(parsed), _own_text(parsed)))
        else:
            raise Refused("unsupported-card", f"{name}: not yet")
    after = record_map(new)
    for edit in edits:
        if edit.icons is not None:
            index_icons(edit, before, after)
    edits = [edit for edit in edits if edit.layers]
    region = changed_region(edits, new)

    # region per edited field (section 5 step 3): the parity bar is measured over the pixels that edit can reach
    merged, parity, changed_pixels, patches = png, {}, 0, []
    preview = stored_preview(old) if update_preview else None
    icc = export_icc(old, png)
    edited = edited_ids(edits)
    timings["baseline"] = timings["render"] = 0.0
    for edit in edits:
        start = perf_counter()
        try:
            base, parity[edit.field] = baseline(old, png, edit.region, icc)
        except Refused as refused:
            refused.field = edit.field if edit.field in ICON_FIELDS else None
            raise
        text = parity[edit.field]["text"]
        timings["baseline"] += perf_counter() - start
        start = perf_counter()
        if preview is None:
            render = render_region(new, edit.region, icc, text, edited)
        else:
            render, raw = render_pair(new, edit.region, icc, text, edited)
        timings["render"] += perf_counter() - start
        merged, replaced = merge_png(merged, base, render, edit.region)
        own = own_pixels(replaced, edit.region, [other.region for other in edits if other is not edit and other.region])
        if not own.any() and not _within_bar(parity[edit.field]):     # a blank render: the PNG would keep the old one
            raise Refused("parity", "%s: the render shows no change, baseline bad %.2f%% mean %.2f max %d"
                          % (edit.field, parity[edit.field]["bad_pct"], parity[edit.field]["mean"], parity[edit.field]["max"]),
                          edit.field if edit.field in ICON_FIELDS else None)
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
    rerender = gate_idempotence(reparse, jobs, reopened, merged, edits, icc, {field: m["text"] for field, m in parity.items()},
                                parser)
    timings["idempotence"] = perf_counter() - start
    return Baked(jobs, edits, region, parity, changed_pixels, records, flag_bytes, new_psd, new_png, reparse, timings,
                 block_bytes, [name for edit in edits for name, _ in edit.fx], icc, rerender,
                 {edit.field: edit.icons.how for edit in edits if edit.icons is not None})


# ========================
# Module Exports
# ========================
__all__ = [
    'Baked',
    'CODES',
    'Classification',
    'Edit',
    'FONTS_DIR',
    'ICON_FIELDS',
    'IconPlan',
    'KIND',
    'OK_FONTS',
    'Refusal',
    'Refused',
    'STATUS_CHART',
    'STORY_KEYS',
    'STORY_MIRROR',
    'STORY_STYLE_KEYS',
    'STAT_FIELDS',
    'SAME_RUN',
    'STAT_MAX',
    'STYLE_KEYS',
    'StatusColor',
    'TEXT_BLENDS',
    'TEXT_FIELDS',
    'TemplateRow',
    'ability_layer',
    'apply_jobs',
    'artifacts',
    'baseline',
    'body_fill',
    'canonical_type',
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
    'gate_outside',
    'gate_region',
    'gate_reparse',
    'gate_shape',
    'gate_size',
    'gate_stories',
    'edit_parts',
    'edit_styles',
    'edited_ids',
    'icc_vote',
    'icon_engine',
    'icon_font',
    'index_icons',
    'keep_breaks',
    'keep_ends',
    'kept_fx',
    'layer_records',
    'merge_png',
    'merge_preview',
    'miss_notes',
    'name_layer',
    'normalize_ability',
    'offline_parser',
    'own_pixels',
    'parse_card',
    'plan_stars',
    'plan_types',
    'read_name',
    'record_map',
    'render_all',
    'shown_name',
    'splice_text',
    'splices',
    'render_pair',
    'render_region',
    'run_fills',
    'save_png',
    'same_types',
    'save_psd',
    'set_icons',
    'set_stat',
    'stale_stories',
    'stat_value',
    'text_blends',
    'text_engine',
    'text_kwargs',
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
    'type_row',
    'uses_icon_font',
]
