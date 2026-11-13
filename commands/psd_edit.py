"""
Edits to a card PSD for the edit pipeline (docs/EDIT_PIPELINE_DESIGN.md sections 3, 5, 6, 7 and 10): classify a card's
website edits against a fresh offline parse, toggle a stat digit, set the name or ability text through the fork's
text engine with status words colored from the chart and the ability's FX layers cut again from the new raster (Effect
Layers.jsx's Update all), show,
hide, move or copy star layers and type icons for a stars or types edit (a refusal of those is per field, never the
card's), render only the changed region, delta-merge it onto the official PNG and gate the result. Pure functions
over an open PSDImage; a refusal is a Refused exception carrying a refusal code, which the per-card driver turns into
data. A content problem never refuses (owner 2026-10-24): it bakes best-effort, told in the notes as "<code>: <why>". No Discord, network or config; the text extra (psd_tools.composite.text) is imported only where text is
rendered.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import ExitStack, contextmanager
from contextvars import Context
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import lru_cache
from json import load
from difflib import SequenceMatcher
from hashlib import blake2b, sha1
from inspect import signature
from itertools import combinations, product, zip_longest
from io import BytesIO
from math import comb, hypot
from os import environ, pathsep, scandir
from os.path import basename, dirname, getsize, join
from re import IGNORECASE, compile as re_compile, escape, fullmatch, search, sub
from struct import pack, unpack
from threading import Lock
from time import perf_counter
from typing import Optional
from uuid import NAMESPACE_URL, uuid5
from weakref import WeakKeyDictionary

import freetype  # noqa: F401  before uharfbuzz: as font_check
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from psd_tools import PSDImage
from psd_tools.api.pil_io import post_process
from psd_tools.api.text_edit import stylistic_alternates
from psd_tools.constants import BlendMode, ColorMode, Resource, Tag
from psd_tools.psd.base import ListElement
from psd_tools.psd.descriptor import Bool, Integer, String

from commands.card_edits import ADD_ABILITY, COLOR_FIELDS, CardEdits, color_text
from commands.card_models import CardInfo, StatsDatabase, straight
from commands.psd_parser import (DRAWN_LAYER_KINDS, EFFECT_LAYER_TAG, ICON_FONT_PREFIX, ICON_SCAN_PATTERN,
                                 TITLE_REGION_RATIO, TYPE_REGION_RATIO, BoundingBox, CardValidator, IconVocabulary,
                                 MutableValue, PSDParser,
                                 StatTracker, base_name, layer_tag, name_key, named_after_text, stars_from_psd,
                                 shown_in, title_text, token_variant, variant_view, variants_of)
from commands.psd_parser import AUX_ITEM_TYPES, FRAME_SHARE, in_header, star_leaf, star_pieces, star_sets
from commands import psd_parts
from commands import psd_links
from commands.psd_links import corners_of, decode, draw, file_type, link_pixels, undrawable, warped
from commands.psd_parts import ACTIVE_TEMPLATE, TEMPLATE_DIR, TEMPLATES, lyid_index, raw_sha
from commands.text_align import (BREAKS as TEXT_BREAKS, blocks as text_blocks, moves as text_moves,
                                 opcodes as text_opcodes, segments as text_segments)
from commands.psd_regions import (STATS as STAT_PATTERNS, _ancestors, badge_layers, clip, comp_bars, comp_creature,
                                  comp_names, comp_row, comp_value, digit_span, fx_pad, overlay_words, placed_data,
                                  reaches_over, stat_rows, union_box)
from global_config import FONT_DIRS_ENV, TYPES_DIR

# ========================
# Configuration
# ========================
STATUS_CHART = "MDW/Croissant's Special Folder/Satus Effect Colors.psd"   # one text layer per status; its fill is the color
FONTS_DIR = "MDW/Fonts"
STAT_FIELDS = ("hp", "def", "atk", "spd")
TEXT_FIELDS = ("name", "ability")
ICON_FIELDS = ("stars", "types")                # printed as icons; refused per field, never for the card (owner, 2026-10-23)
PARTIAL_FIELDS = ICON_FIELDS + ("colors",)      # refused alone while the card's other fields bake (a colour: each its own)
STATUS_KEPT = "{!r} is a status word, which keeps the chart's colour"   # plan_colors (owner 2026-11-07)
KIND = CardEdits.KIND
STAT_MAX = 10                                   # one bar row shows 0-10
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
# How type layers' edges blend, tried in order by baseline(): the fork reads each layer's off the PSD's preview (with
# none, the render record's: look_kwargs()); a PNG exported in another Photoshop session than the preview's may show
# them all in the text gamma, or all straight. Under "preview" and "straight" the edited layers are pinned to the gamma
# (text_kwargs()): Photoshop redraws them in it
TEXT_BLENDS = {"preview": {}, "gamma": {"text_preview": False}, "straight": {"text_gamma": None}}
PNG_LEVEL = 6               # save_png's zlib level (PIL's default; a preview's PNG, never published: 1)
ON_MERGED = None            # a preview's: (merged, path, like) -> a Future of save_png, made beside the gates
OLD_BESIDE = False          # a preview's: gate_dirty draws old's box in a thread beside new's (Card sync's pool: serial)
FONTS_WAIT = None           # a preview's: returns once MDW/Fonts is in (it restores beside the bake); called before any FontMap
SIZE_TOLERANCE = 0.2        # a saved PSD stays within 20% of the old blob
SIZE_FLOOR = 262144         # or this many bytes: one icon on a small PSD (Poorly Drawn Cabbage)
OK_FONTS = ("exact", "system")                  # how a font may resolve; loose or substituted faces refuse
TOKEN = re_compile(r"\[[^\[\]\r\n\x03]{1,52}\]")   # an icon token in ability text ([fire]); psd_parser.ICON_SCAN_PATTERN
WEB_NORM = re_compile(r"[\s_\-.'\u2019]+")          # what the website's norm() drops from a token's base (site/assets/text.js)
TEXT_COLOURED = ("sticker", "tint")    # token variants drawn in the text's colour: the :sticker outline, the :tint silhouette
GLOWS = ("glow", "darkglow")  # token variants drawn with an image halo
LAYERED = ("sticker", "no") + GLOWS    # variants drawn as overlapping glyphs in one cluster: tracking splits them
LETTER = re_compile(r"[A-Za-z]")                   # status_spans' word boundary
SAME_RUN = 3                # edit_parts: characters an edit keeps this many in a row are not restyled
STYLE_KEYS = ("Font", "FontSize", "FauxBold", "FauxItalic", "AutoLeading", "Leading", "HorizontalScale",
              "VerticalScale", "Tracking", "BaselineShift", "FontCaps", "FontBaseline", "Ligatures",
              "BaselineDirection", "Tsume", "StyleRunAlignment", "Language", "NoBreak")   # edit_styles copies these
PAREN = re_compile(r"\(([^()\r\n\x03]{1,20})\)")    # a timing tag or cost candidate: "(1CD)", "(2 points)"
GLUED = re_compile(r"[\w.]")                          # a cost right after one is a multiplier: "2(x counters)" (the font's GLUED)
ICON_FONT = "stoogedings-Regular"                   # the icon font's PostScript name (every card's FontSet entry)
INK_ALPHA = 0.5             # a pixel is ink past this alpha (a title grazing a badge's anti-aliased edge is not an overlap)
ICON_HEIGHT = (0.6, 2.2)    # a typed icon at the text's own size: its height over its line's letters' (Holy Blade 0.2)
BADGE_RATIO = 1.35          # a badge's pill over the text's capital height (the cards' median, 1.357 over 731 badges)
BADGE_PILL = (-130, 820)    # the pill's bottom and top, icon font units of 1000 (build_stoogedings BADGE_BOTTOM, BADGE_TOP)
BADGE_RESIZE = 0.10         # the guards resize a badge more than this off its computed size
HARD_SOFT = 0.005           # hard_edged: the share of a picture's pixels with in-between alpha
HARD_COLOURS = 256          # its colours at most
HARD_SMALL = 128            # px: its pixels show at this size; a larger one must be blocky
HARD_BLOCKY = 0.5           # the share of its neighbouring pixels that match
HARD_CLEAR = 4              # stored_mode, image_mode: times closer one mode must redraw the stored pixels
GLYPH_TOL = 0.0001          # a restyled text layer's glyphs within this of where they were draw nothing new (canvas px)
FIT_PX, FIT_PCT = 2.0, 3.0  # and shift a badge or icon whose middle is off by more than both: px, % of the text's em
FIT_SKIP = ("glow", "no_over", "space")     # icon-font glyphs that are no icon's body (halos, the red sign, spacers)
ASPECT_TOL = 0.02           # unstretch: a typed icon's x over y scale may be this far off 1 (designers': 678 of 682 at 1.0)
FONT_IOU_MIN = 0.3          # a text layer redrawn whole against its saved pixels: their ink's IoU (Omori's swapped fonts
                            # 0.11-0.14; of 1,453 titles and Abilities the next is 0.40, Wendy's few px shift)
FIT_STEP, FIT_MIN = 0.94, 0.4  # fit_below: smaller by this, down to this share of its size (the website's preview)
BOX_DESCENT = 0.35          # hold_lines: em of the layer's largest size a grown box keeps under its last baseline
BEYOND = (6, 60, 20, 100)   # ink_beyond: px past the old ink's box; luminance levels off the backdrop's median under the
                            # old text (at least; and past its 95th percentile by); px of new ink on such detail to tell
FX_TAG = EFFECT_LAYER_TAG   # "Ability FX: Blazing": a status word's glyph pixels under the chart's effect (Effect Layers.jsx)
KEY_INK = 0.5 / 255         # the JSX's key copy, the word white over black, selects every pixel its red channel reaches (level 1)
KEY_WHITE, KEY_BLACK = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
FX_SEP = " | "              # Effect Layers.jsx's SEP between an FX layer's phrases
ANCHORED = ("styled", "own")   # FX layers whose style is their own words': they stay on the characters they covered
ALNUM = re_compile(r"[A-Za-z0-9]")                  # phrase_ranges' whole-word boundary (the JSX's)
FX_BOX_PAD = 3              # px gate_fx lets an FX layer ink past its words' glyph boxes (edges, a badge's pill)
FX_COVER = 0.99             # share of its words' own ink (no other glyph's box) an FX layer must cover (gate_fx)
FAINT_INK = 0.05            # a colour glyph's pixel past this alpha is its ink (bare_icons; [support]'s glow)
COLOUR_TOL = 2 / 255        # two fills' renders agree within this on a colour glyph's pixel (_colour_ink)
EFFECT_MULTI = {b"dropShadowMulti": b"DrSh", b"innerShadowMulti": b"IrSh", b"solidFillMulti": b"SoFi",
                b"gradientFillMulti": b"GrFl", b"frameFXMulti": b"FrFX"}    # lfx2's lists of an effect: _style_key
SPLICE_UNIT = "glyph"       # a text edit redraws its changed glyphs ("line": their lines); Photoshop's raster stays elsewhere
KEY_SLACK = 2               # px a key render may pass a spliced layer's bbox (Photoshop's ink beside the engine's)
ALLOWED_RECORD_FIELDS = {"stat": {"flags", "blocks"}, "text": {"rect", "channels", "blocks"},
                         "overlay": {"flags"}, "toggle": {"flags"}, "fx": {"rect", "channels", "name", "blocks"},
                         "comp": {"rect", "channels", "blocks"}}
#   what an edit may change per record (overlay: a clipped layer hidden, Edit.hidden; toggle: a star or type icon shown or
#   hidden; fx: an FX layer cut again in place, its name and blocks only as _fx_rebuilt allows; comp: a layer-comp stat
#   bar set, its SoLd block alone)
STORY_KEYS = ("1", "1", "0", "0")      # Txt2's stories and their text (Photoshop CC's numbered keys)
STORY_STYLE_KEYS = {    # a story's style runs, a run's features, the FontSet, a paint's values
    "runs": ("6", "0"), "data": "0", "length": "1", "features": ("0", "6"), "fonts": ("0", "1", "0"),
    "font": ("0", "0", "0"), "paint": ("0", "1")}
STORY_MIRROR = {"Font": 0, "FontSize": 1, "FauxBold": 2, "FauxItalic": 3, "AutoLeading": 4, "Leading": 5,
                "HorizontalScale": 6, "VerticalScale": 7, "Tracking": 8, "BaselineShift": 9, "FillColor": 53,
                "StrokeColor": 54}   # TySh StyleSheetData keys a story run holds too, with Photoshop CC's feature number
KERN_FEATURE = "11"      # a numbered story run's kerning: 0 off, else on (TySh's AutoKerning; B12)
COMPOSER_KEY = "29"      # a numbered story paragraph's EveryLineComposer: Photoshop 2023 sets box text without it Every-line
STAR_FOLDER = r"\s*(\d+)\s*stars?\s*"           # a count's own folder in a Stars group: "5 stars", "1 star" (the templates)
LAYOUT_TOL = 0.12           # a star or icon sits on a layout's spot within this share of its size
LAYOUT_TOL_MIN = 6          # px, at least
ICON_GAP = 22               # px between an icon and the one under it on a 1200-wide card (Item_Template: Attack, NME)
LOWER_STEP = 0.1            # a blocked added star or icon steps down this share of its height at a time (_drops)
LOWER_MIN = 4               # px, at least
ICON_TOUCH = 0.05           # a moved, shown or made type icon may ink over this share of another's (the cards'
#   own rows: 158 of 161 rows of two or more touch nowhere, Pandora's pandas 5.5%; 2026-10-26)
CARD_WIDTH = 1200
PLACED = (Tag.SMART_OBJECT_LAYER_DATA1, Tag.SMART_OBJECT_LAYER_DATA2, Tag.PLACED_LAYER1, Tag.PLACED_LAYER2)
MOVE_BLOCKS = {bytes(key) for key in (*PLACED, Tag.REFERENCE_POINT)}    # a moved layer's record: its transform, anchor
COPY_BLOCKS = MOVE_BLOCKS | {bytes(Tag.LAYER_ID), bytes(Tag.UNICODE_LAYER_NAME)}   # a copy's: those, its id and name


def badge_tags() -> list:
    """Every timing tag the stoogedings font draws as a badge: TTSCardMaker build_stoogedings.badge_tags(), mirrored: the
    341 of 2026-10-22, then the wider range (owner 2026-11-03: n 1-20 or X, a/b with a from -9 up to b)."""
    nums, firsts = "123456789X", ["-1", *"0123456789", "X"]
    tags = ["OYT", "AAT", "BOT", "ENDT", "BOGT", "ENDGT", "OFS", "OS", "OD", "WOF", "Action"]
    for n in nums:                              # the old system (rule pages)
        tags += [f"{n}PT", f"E{n}T", f"FTE{n}T", f"{n}PGT", f"E{n}GT", f"{n}PG"]
    for n in nums:                              # cooldowns and uses (2026-10-21)
        tags += [f"{n}CD", f"{n}GCD", f"{n}U"]
    for cd in ("CD", "GCD"):
        tags += [f"{a}/{b}{cd}" for a in firsts for b in nums]
    counts, wide = [*map(str, range(1, 21)), "X"], [*map(str, range(-9, 21)), "X"]
    more = [f"{n}{kind}" for n in counts for kind in ("CD", "GCD", "U", "PG")]
    for cd in ("CD", "GCD"):
        more += [f"{a}/{b}{cd}" for b in counts for a in wide if "X" in (a, b) or int(a) <= int(b)]
    seen = set(tags)
    return tags + [t for t in more if t not in seen and not seen.add(t)]


def cost_tags() -> list:
    """Every cost the font draws as a badge (build_stoogedings.cost_badges' spellings, owner 2026-11-03): "2 points",
    "1 point", "1.5 points", "X bricks", "2 SP", "7 star points", "4 souls", "3 counters"; 0-50, halves to 9.5 and X."""
    numbers = [*map(str, range(51)), *(f"{n}.5" for n in range(10)), "X"]
    words = ("point", "points", "brick", "bricks", "SP", "star point", "star points", "soul", "souls", "counter", "counters")
    return [f"{n} {w}" for w in words for n in numbers]


BADGE_TAGS = tuple(badge_tags())
COST_TAGS = tuple(cost_tags())
BADGE_KEYS = {tag.lower(): tag for tag in BADGE_TAGS + COST_TAGS}   # any letter case is the same (TTSCardMaker 6e50c3ee)


def badge_spans(text: str) -> list:
    """[(start, end)] of each timing tag and cost in text the font draws as a badge, any letter case; a cost glued to a
    word or number before it ("2(x counters)") is a multiplier the font leaves as letters."""
    return [m.span() for m in PAREN.finditer(text) if m.group(1).lower() in BADGE_KEYS
            and not (" " in m.group(1) and m.start() and GLUED.match(text[m.start() - 1]))]


# ========================
# Data
# ========================
@dataclass
class Refusal:
    """One refusal: a refusal code and why; field: the stars or types edit it is about (the card's others still bake)."""
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
    site: dict = field(default_factory=dict)        # field -> why: website only, as the card image can't print it (a
    #   title that is part of the picture, a stat past what its bar rows print; owner 2026-10-26): the bake adds it to _site
    parts: Optional["PartsPlan"] = None             # the entry's `parts` sorted (plan_parts), when it has them
    colors: list = field(default_factory=list)      # plan_colors(): [(colour, outcome)] of the entry's `colors`


@dataclass
class StatusColor:
    """A status effect's color from the chart."""
    name: str
    fill: tuple             # RGB, 0..1
    styled: bool            # an enabled effect on the chart layer (glow, gradient, shadow): the word gets an FX layer
    blocks: dict = field(default_factory=dict)  # the chart layer's effect blocks by Tag (lfx2, lrFX): copied onto the FX layer


@dataclass
class TemplateRow:
    """A template bar row: the ten masked exposure digits of "<Stat> Dark"."""
    group: object
    digits: dict            # "0".."9" -> layer

    @property
    def visible(self) -> list:
        """The visible digits, sorted."""
        return sorted(int(name) for name, layer in self.digits.items() if layer.is_visible())

    def reads(self) -> Optional[int]:
        """What the parser reads for the row: the lowest visible digit; None when none shows."""
        return min(self.visible) if self.visible else None


@dataclass
class IconPlan:
    """A stars or types edit worked out on an open card (plan_stars, plan_types): what shows, hides, moves and comes."""
    field: str
    old: object
    new: object
    how: str = ""                                # the layout followed: "rows of two", "a line"...
    show: list = field(default_factory=list)     # layers shown
    hide: list = field(default_factory=list)     # layers hidden
    moves: list = field(default_factory=list)    # (layer, dx, dy): moved (a group with everything in it)
    copies: list = field(default_factory=list)   # (source, dx, dy, name): a star copied above itself, then moved
    icons: list = field(default_factory=list)    # (Types/ stem, canvas box, reference layer): a type icon made in the box
    done: list = field(default_factory=list)     # set_icons(): ("copy", source, copy, dx, dy), ("made", layer,
    #                                              blocks), ("masked copy", texture, copy, dx, dy): _texture_copies
    first: str = ""                              # plan_types(): the template whose spot a card's first type icon takes
    centres: list = field(default_factory=list)  # plan_stars(): the header stars' centres once applied, reading order
    notes: list = field(default_factory=list)    # told with the bake: stars or icons stepped down, shrunk, off banner
    pulled: list = field(default_factory=list)   # plan_types(): icons pulled back onto the card (_close_row, S3)
    row: list = field(default_factory=list)      # plan_types(): the row's icons that show, kept or shown (clear of made ones)
    textures: list = field(default_factory=list) # plan_stars(): textures clipped onto the stars' group that follow them
    scaled: list = field(default_factory=list)   # plan_stars(): (layer, s, centre, to): a star shrunk to fit its banner
    #                                              (option B, _scale), before the copies and moves


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
    hidden: list = field(default_factory=list)  # record indices of the clipped layers hidden (set_colors)
    fx: list = field(default_factory=list)      # (name, {block key: bytes}) per FX layer made above its group, bottom to top
    dropped: list = field(default_factory=list)  # names of the FX layers deleted (their words left the text)
    rebuilt: list = field(default_factory=list)  # (record index, old name, name): an FX layer cut again in place, its own
    #   style kept (Effect Layers.jsx's Update all; owner q6) but on a chart status word (restyled)
    restyled: list = field(default_factory=list)  # (record index, {block key: bytes}): an FX layer on a chart status word
    #   given the chart's style (owner 2026-11-07)
    fx_names: list = field(default_factory=list)  # the text layer's FX layers after the edit, bottom to top
    notes: list = field(default_factory=list)   # for the report: FX layers deleted, renamed or cut again; ink_beyond's
    held: list = field(default_factory=list)    # (FX layer name, [(start, end)]): a "styled" or "own" layer's kept words,
    #   which it still covers (gate_fx)
    story: Optional[int] = None  # the text layer's TextIndex before the edit: the Txt2 story it may change
    fonts: list = field(default_factory=list)   # faces it added to the document's FontSets (ICON_FONT on a plain card)
    whole: bool = False     # the engine redrew every glyph (no splice, or it fell back): the lines it left laid out again
    lines: Optional[tuple] = None  # a plain layer spliced in place: the splice zone, the only pixels it changes
    lifted: int = 0         # px an Ability edit moved its layer up (lift)
    boxed: bool = False     # its box grown to hold its lines (hold_lines): its Txt2 frames change (gate_stories)
    icons: Optional["IconPlan"] = None  # a stars or types edit's plan (index_icons() fills the four lists below)
    toggled: list = field(default_factory=list)  # record indices of the layers it showed or hid: one flag byte each
    shifted: list = field(default_factory=list)  # (record index, old record index, dx, dy): a layer it moved
    copied: list = field(default_factory=list)   # (record index, old record index, dx, dy): a record of a star it copied
    made: list = field(default_factory=list)     # (record index, {block key: bytes}): a type icon it made from Types/
    masked: list = field(default_factory=list)   # record indices given a pixel mask: a star layer split (split_stars)
    filters: list = field(default_factory=list)  # (placed ID, its source's, dx, dy): a smart-filtered layer moved or
    #   copied, its FEid item with it (S1; gate_filters)
    stories: list = field(default_factory=list)  # a part bake: every TextIndex it may change
    parts: list = field(default_factory=list)    # and (layer id, that layer's Edit, a copy?) per layer it set
    several: bool = False   # a part bake's Edit (set_parts): its layers are in parts
    removed: list = field(default_factory=list)  # old record indices of the layers a part edit deleted (gate_bytes)
    removed_stories: list = field(default_factory=list)  # their Txt2 stories' old TextIndex (gate_stories)
    renumbered: list = field(default_factory=list)  # record indices of type layers whose TextIndex moved down (gate_bytes)
    added: list = field(default_factory=list)   # record index of a layer an added ability made (gate_bytes)
    added_stories: list = field(default_factory=list)  # its Txt2 story's TextIndex in the new card (gate_stories)
    bar: list = field(default_factory=list)     # a stat on the card maker's own bar: (record index, _bar_spec()) per layer
    comps: list = field(default_factory=list)   # record indices of the layer-comp bars it set (set_comp)
    scaled: dict = field(default_factory=dict)  # old record index -> new rect (top, left, bottom, right): a shrunk star's
    #   layer (IconPlan.scaled), with its copies and moves in shifted and copied
    files: dict = field(default_factory=dict)   # repo path -> bytes: a placed title's file set again (set_placed_title)
    sources: list = field(default_factory=list)  # placed IDs whose FEid source it drew again (gate_filters)
    relinked: list = field(default_factory=list)  # uuids of the links it rewrote, size and contentID (gate_stories)


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
    fx: list = field(default_factory=list)      # names of the FX layers cut again or made
    fx_notes: list = field(default_factory=list)  # Edit.notes: FX layers deleted, renamed or cut again; ink_beyond's
    icc: bool = True        # export_icc(): rendered converted by the embedded profile (False: the PNG was exported raw)
    rerender: dict = field(default_factory=dict)    # field -> gate_idempotence()'s measurement of its region
    icons: dict = field(default_factory=dict)       # stars / types -> the layout their edit followed (IconPlan.how)
    whole: dict = field(default_factory=dict)       # field -> whole_redraw()'s measurement (Edit.whole)
    files: dict = field(default_factory=dict)       # repo path -> bytes: other files the edits set (Edit.files)


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


def _ink_off_canvas(layer, width: int, height: int) -> bool:
    """Whether the layer inks (past INK_ALPHA) off the canvas; a faint fringe Photoshop clips does not count (a :glow
    halo's tail past a title's top: Salt's 0.16)."""
    bbox = tuple(layer.bbox)
    if clip(bbox, width, height) == bbox:
        return False
    alpha = _alpha(layer)
    if alpha is None:
        return True
    x0, y0 = bbox[0], bbox[1]
    inside = alpha[max(0, -y0):max(0, height - y0), max(0, -x0):max(0, width - x0)]
    return int((alpha > INK_ALPHA).sum()) != int((inside > INK_ALPHA).sum())


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


def _clip_base(layer):
    """The layer a clipped one clips to (the nearest one below it in its group that is not clipped), else None."""
    if not layer.clipping or layer.parent is None:
        return None
    base = None
    for other in layer.parent:
        if other._record is layer._record:
            return base
        if not other.clipping:
            base = other
    return None


def _flat(image: Image.Image) -> np.ndarray:
    """uint8 (H, W, 4): RGB composited over white, alpha kept (RGB under alpha 0 is arbitrary)."""
    a = np.array(image.convert("RGBA"))
    part = a[..., 3] != 255
    n = int(np.count_nonzero(part))
    if n * 5 > part.size:  # mostly non-opaque: whole image is faster
        a = a.astype(np.float32)
        k = a[..., 3:] / 255
        a[..., :3] = a[..., :3] * k + 255 * (1 - k)
        return np.rint(a).astype(np.uint8)
    if n:  # alpha 255 is exact as is; same float32 maths on the rest
        sub = a[part].astype(np.float32)
        k = sub[:, 3:] / 255
        sub[:, :3] = sub[:, :3] * k + 255 * (1 - k)
        a[part] = np.rint(sub).astype(np.uint8)
    return a


def _px_differ(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a != b).any(axis=2) for (H, W, 4) uint8 canvases: one 32-bit compare per pixel."""
    if a.dtype == b.dtype == np.uint8 and a.shape == b.shape and a.shape[-1:] == (4,) \
            and a.strides[-1] == b.strides[-1] == 1:
        return a.view(np.uint32)[..., 0] != b.view(np.uint32)[..., 0]
    return (a != b).any(axis=2)


def _px_max(d: np.ndarray) -> np.ndarray:
    """d.max(axis=2) as channel maxima (numpy's last-axis reduce is slow)."""
    out = d[..., 0].copy()
    for c in range(1, d.shape[2]):
        np.maximum(out, d[..., c], out=out)
    return out


def _absdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a - b| of two uint8 arrays, in uint8."""
    return np.maximum(a, b) - np.minimum(a, b)


def _px_diff(a: Image.Image, b: Image.Image) -> np.ndarray:
    """Per-pixel max |a - b| over the channels, both over white (uint8); two RGB images as they are."""
    if a.mode == b.mode == "RGB":
        return _px_max(_absdiff(np.asarray(a), np.asarray(b)))
    return _px_max(_absdiff(_flat(a), _flat(b)))


def _measure(a: Image.Image, b: Image.Image) -> dict:
    """bad_pct, mean and max of |a - b| (over white, alpha included)."""
    diff = _px_diff(a, b)
    return {"bad_pct": float((diff > PARITY_THRESH).mean() * 100), "mean": float(diff.mean()), "max": int(diff.max())}


def _within_bar(measured: dict) -> bool:
    return measured["bad_pct"] <= PARITY_BAD_MAX and measured["mean"] <= PARITY_MEAN_MAX


def _past_floor(measured: dict) -> bool:
    """A broken render: more than PARITY_FLOOR % of the region's pixels past PARITY_THRESH (the only pixel refusal)."""
    return measured["bad_pct"] > PARITY_FLOOR


def _record_index(layer) -> int:
    records = layer._psd._record.layer_and_mask_information.layer_info.layer_records
    return next(i for i, record in enumerate(records) if record is layer._record)


def edited_ids(edits: list) -> tuple:
    """Layer IDs of the edits' layers (text_kwargs()): the same in the edited PSD and the saved one."""
    return tuple(layer.layer_id for edit in edits for layer in edit.layers)


def _enabled_effects(layer) -> list:
    """Names of the layer's effects whose own enabled flag is set (the block flag alone is not enough, section 5)."""
    return [type(e).__name__ for e in layer.effects if e.enabled]


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
        breaks as \\n, tabs as spaces, no lines of blanks, no space before punctuation, quotes and blank ends stripped
    """
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ").replace("\x03", "\n")
    text = straight(text).replace("﻿", "").rstrip()    # apostrophes compare straight (owner, 2026-10-25)
    text = text.strip("'\"\n").rstrip(" '\"")
    if text:
        text = sub(r"(?m)^[^\S\n]+\n", "", text)
        text = sub(r"\s+([:;,\.\?!])", r"\1", text)
    return text.strip(" '\"\n")


# ========================
# Parsing (the goldens' route, without psd_analyzer)
# ========================
def offline_parser(all_types: list, stars: list, manifest: Optional[dict]) -> PSDParser:
    """
    A PSDParser on an empty database: a fresh parse with no author or series preservation. With a manifest, its icon
    vocabulary knows only the icons the font was built with: a Types/ icon added before the font is rebuilt would be
    set in the icon font as its letters and read back as its token (review 2026-10-23); it stays a plain [token].

    Args:
        all_types: Types/ icon names (lowercase stems)
        stars: Types/Stars/ icon names
        manifest: The stoogedings manifest, or None

    Returns:
        PSDParser
    """
    db = StatsDatabase()
    db.all_types = list(all_types)
    icons = IconVocabulary.build(db.all_types, stars, manifest)
    fonted = set((manifest or {}).get("icons", {}))
    for spelling in [spelling for spelling, (name, _) in icons.tokens.items() if fonted and name not in fonted]:
        del icons.tokens[spelling]
    return PSDParser(db.all_types, db, icons)


def parse_card(parser: PSDParser, path: str, rel: str, psd: Optional[PSDImage] = None) -> dict:
    """
    Everything a gate compares: to_dict(), the validator's problems and the CardInfo itself.

    Args:
        parser: offline_parser()
        path: The PSD on disk
        rel: Its repo path (the classification comes from the folders)
        psd: path already open (one open per PSD per bake); None: path is opened

    Returns:
        {"card", "validator", "info"}
    """
    card = parser.parse(psd if psd is not None else path, rel)
    return {"card": card.to_dict(), "validator": CardValidator.validate(card), "info": card}


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


def _copies(psd: PSDImage, layer) -> list:
    """The other visible text layers printing the layer's words where they show (a copy for an effect, a watermark)."""
    def words(text):
        return " ".join((text or "").replace("\x03", " ").split()).casefold()
    own = words(layer.text)
    if not own:
        return []
    return [other for other in psd.descendants() if _is_text_layer(other) and other is not layer
            and other.is_visible() and words(other.text) == own and _shows(psd, other)]


def _copy(psd: PSDImage, layer) -> Optional[str]:
    """The first of _copies' names (an edit would change one and leave the rest); None when there is none."""
    found = _copies(psd, layer)
    return found[0].name if found else None


def _no_title(psd: PSDImage) -> str:
    """Why the parser reads no title layer (name_layer's refusal)."""
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
    if len(own) > 1:
        return f"{len(own)} layers in the title region named after their text"
    titles = [child.name for child in group if _is_text_layer(child) and child.name.strip().lower() != "ability"
              and child.bbox[1] < psd.height * 0.15]
    return f"title layer named {titles[0]!r}, not Name, the card's name or its text" if titles else "no Text/Name layer"


def name_layer(psd: PSDImage, card: str = ""):
    """
    The card's title layer: the one visible text layer the parser reads the printed title from (PSDParser.title_layers:
    Text/Name, else a Name layer in the title region, else a layer under Text named after the card, else one elsewhere
    in the title region: a status effect's top-level title, else the one there named after its text: Sans's 'snas.').

    Args:
        psd: The open card
        card: The card's name (file stem), which a title layer may be named after

    Returns:
        The layer; raises Refused("unsupported-card") when no layer or several print the title (an epithet layer, a
        title split over layers, a copy for an effect that shows)
    """
    found = PSDParser.title_layers(psd, [layer for layer in psd.descendants() if _is_text_layer(layer)], card)
    if not found:
        raise Refused("unsupported-card", _no_title(psd))
    if len(found) > 1:
        raise Refused("unsupported-card", f"{SPLIT_TITLE} over {len(found)} layers: edit each box on the website")
    joined, whole = "".join(title_text(found[0].text).lower().split()), "".join(card.lower().split())
    if PSDParser.split_title(psd, found, title_text(found[0].text), card) or PSDParser._named_pieces(psd, found, joined, whole, card):
        raise Refused("unsupported-card", f"{SPLIT_TITLE} over several layers: edit each box on the website")
    copy = _copy(psd, found[0])
    if copy is not None:
        raise Refused("unsupported-card", f"the title is printed again by {copy!r}")
    return found[0]


PICTURE_TITLE = "the title is part of the picture"     # classify(): a name edit there is website only (owner 2026-10-26)
SPLIT_TITLE = "the title is split"      # name_layer(): a whole-name edit on one; the website edits each part's box


def title_in_picture(psd: PSDImage, card: str = "") -> bool:
    """
    Whether the printed title is part of the picture: no layer the parser reads it from and no visible text in the title
    region of the Text group (of the card, without one): ABBI's key pictures, Baba's sprite, E. Gadd's Garage over a
    hidden Name layer. A title layer with another name (Alex YIIK's) or split over layers (Knight Lost A Lot) is text.
    """
    texts = [layer for layer in psd.descendants() if _is_text_layer(layer)]
    if PSDParser.title_layers(psd, texts, card):
        return False
    group = _text_group(psd)
    scope = [layer for layer in group.descendants() if _is_text_layer(layer)] if group is not None else texts
    top = psd.height * TITLE_REGION_RATIO
    return not any(layer.is_visible() and (layer.text or "").strip() and layer.bbox[1] < top and FX_TAG not in layer.name
                   and base_name(layer).strip().lower() != "ability" for layer in scope)


def read_name(psd: PSDImage, card: str = "", read=None, rel: str = "") -> Optional[str]:
    """The printed title, whitespace collapsed; a placed title's (placed_title; read and rel its); None when name_layer
    refuses."""
    try:
        return collapse(name_layer(psd, card).text)
    except Refused:
        found = placed_title(psd, card, read, rel)
        return collapse(found[4].text) if found else None


def placed_title(psd: PSDImage, card: str = "", read=None, rel: str = "") -> Optional[tuple]:
    """
    The card's title as a smart object the bake can set (PSDParser.placed_titles, read: a repo path -> its bytes): the
    one there, linked, upright, its smart filters none or one psd_links.redraw draws (a lone Gaussian or Motion Blur:
    Gust of Wind's names, owner 2026-11-11). Its tuple, or None.
    """
    if PSDParser.title_layers(psd, [layer for layer in psd.descendants() if _is_text_layer(layer)], card):
        return None
    found = PSDParser.placed_titles(psd, rel, read)
    if len(found) != 1 or not found[0][1]:
        return None
    layer = found[0][0]
    return found[0] if psd_links.why_not_redrawn(layer) is None and psd_links.upright(corners_of(layer)) else None


def set_placed_title(psd: PSDImage, found: tuple, text: str, fonts_dir: str, rel: str, parser=None) -> Edit:
    """
    A name edit on a placed title (placed_title): its file's type layer set to the text keeping its runs, Photoshop's
    raster kept on the glyphs the edit leaves (capture), the file's canvas cropped to the new raster as Photoshop's
    Convert to Smart Object makes it, so the text keeps its place on the card; the placed layer drawn again from the file
    (psd_links.redraw: its Motion Blur and FEid source too) and its link rewritten (size, contentID, pixels, corners).
    Off the canvas or onto the stars or type icons is told (overflow), never refused.

    Returns:
        Edit, the file's new bytes in Edit.files
    """
    from psd_tools.composite.text import MissingFontError, capture
    layer, path, _, inner, title = found
    old_text, old_box = title.text, tuple(layer.bbox)
    text = keep_ends(old_text, straight(text).replace("\r\n", "\n").rstrip("\n").replace("\n", "\r"))
    fonts = font_map(fonts_dir)
    try:
        before = capture(title, fonts)
        title.set_text(text, keep="runs")
        title.rasterize(fonts=fonts, keep=before, unit=SPLICE_UNIT)
    except MissingFontError as error:
        raise Refused("font", str(error))
    box = tuple(title.bbox)
    if box[2] <= box[0] or box[3] <= box[1]:
        raise Refused("unsupported-card", "name: the new title draws nothing")
    h = psd_links.homography(inner.width, inner.height, corners_of(layer))
    corners = [float(c) for u, v in ((box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3]))
               for c in psd_links._apply(h, u, v)]
    data = _cropped(inner, title, box)
    region = psd_links.redraw(layer, psd_links.picture(psd, data, psd_links.file_type(data, path)), corners, Image.BICUBIC)
    so = layer.smart_object
    so.relink(size=len(data), blob=psd_links.git_blob(data), psd_path=rel, pixels=(box[2] - box[0], box[3] - box[1]),
              corners=corners)
    _warp_bounds(layer, corners)
    notes = []
    if clip(tuple(layer.bbox), psd.width, psd.height) != tuple(layer.bbox):
        notes.append("overflow: name runs off the canvas")
    stars, icons, _ = badge_layers(psd, getattr(parser, "all_types", ()))
    if any(_touches(layer, badge) for badge in stars + icons):
        notes.append("overflow: name reaches the stars or type icons")
    return Edit("name", old_text, text, [layer], [_record_index(layer)],
                union_box([_pad(one, fx_pad(layer), psd) for one in (old_box, region)]), notes=notes,
                files={path: data}, sources=[so.placed_id] if so.filter_item is not None else [],
                relinked=[str(so.unique_id).strip("\x00")])


def _cropped(inner: PSDImage, title, box: tuple) -> bytes:
    """The placed title's file cropped to box (its type layer's raster, file px) and saved: the layer moved by box's
    corner (nudge), the canvas and its full-canvas slices box's size, the merged image white again (strip_merged)."""
    width, height = inner.width, inner.height
    nudge(title, -box[0], -box[1])
    header = inner._record.header
    header.width, header.height = box[2] - box[0], box[3] - box[1]
    slices = inner.image_resources.get_data(Resource.SLICES)
    found = getattr(slices, "data", None)
    if found is not None:
        _slice_bounds(found, width, height, header.width, header.height)
    inner.strip_merged()
    out = BytesIO()
    save_psd(inner, out)
    return out.getvalue()


def _slice_bounds(node, width: int, height: int, new_w: int, new_h: int) -> None:
    """Every whole-canvas bounds in a Slices resource (its bbox, its slices', their descriptors') made the new size."""
    if isinstance(getattr(node, "bbox", None), list) and len(node.bbox) == 4:
        if list(node.bbox) == [0, 0, height, width]:
            node.bbox[2], node.bbox[3] = new_h, new_w
        elif list(node.bbox) == [0, 0, width, height]:
            node.bbox[2], node.bbox[3] = new_w, new_h
    for item in getattr(node, "items", None) or []:
        _slice_bounds(item, width, height, new_w, new_h)
    data = getattr(node, "data", None)
    stack = [data] if hasattr(data, "items") else []
    while stack:
        desc = stack.pop()
        for key, value in desc.items():
            if key == b"bounds" and hasattr(value, "get") and [_number(value.get(k)) for k in
                                                               (b"Top ", b"Left", b"Btom", b"Rght")] == [0, 0, height, width]:
                _set_number(value, b"Btom", new_h)
                _set_number(value, b"Rght", new_w)
            elif hasattr(value, "items"):
                stack.append(value)
            elif isinstance(value, (list, ListElement)):
                stack.extend(one for one in value if hasattr(one, "items"))


def _number(value) -> Optional[float]:
    value = getattr(value, "value", value)
    return float(value) if isinstance(value, (int, float)) else None


def _set_number(desc, key: bytes, value: float) -> None:
    old = desc.get(key)
    if hasattr(old, "value"):
        old.value = type(old.value)(value)
    else:
        desc[key] = type(old)(value) if old is not None else value


def _warp_bounds(layer, corners: list) -> None:
    """The placed layer's warp bounds (an unwarped one's: its quad's box), as Photoshop writes them, set to corners'."""
    bounds = ((placed_data(layer) or {}).get(b"warp") or {}).get(b"bounds")
    if hasattr(bounds, "get"):
        xs, ys = corners[0::2], corners[1::2]
        for key, value in ((b"Top ", min(ys)), (b"Left", min(xs)), (b"Btom", max(ys)), (b"Rght", max(xs))):
            _set_number(bounds, key, value)


def _card_name(parsed: dict) -> str:
    """The parse's card name (file stem)."""
    info = parsed.get("info")
    return info.name if info is not None else parsed["card"].get("name") or ""


def _txt2(psd: PSDImage):
    blocks = psd._record.layer_and_mask_information.tagged_blocks
    return blocks.get_data(Tag.TEXT_ENGINE_DATA) if blocks is not None and Tag.TEXT_ENGINE_DATA in blocks else None


def _story_list(root) -> tuple:
    """(Txt2 DocumentObjects, its TextObjects key, the stories, the Model and Text keys)."""
    objects_key, items_key, model, text = STORY_KEYS
    objects = root.get(objects_key)
    items = objects.get(items_key) if hasattr(objects, "get") else None
    if items is None:
        raise Refused("unsupported-card", "a Txt2 block without text objects")
    return objects, items_key, items, model, text


def stories(psd: PSDImage) -> Optional[list]:
    """
    The texts of the document's Txt2 stories by TextIndex, final \\r included; None without a Txt2 block.

    Photoshop reads a type layer's text from its story, not from the layer's own TySh, and rebuilds TySh from it on
    save. Read here, not through the fork, so the gates check the fork.
    """
    root = _txt2(psd)
    if root is None:
        return None
    _, _, items, model, text = _story_list(root)
    return [str(story[model][text].value) for story in items]


def story_index(layer) -> Optional[int]:
    """The type layer's TextIndex: its story's index in Txt2."""
    index = layer._data.text_data.get(b"TextIndex")
    return None if index is None else int(getattr(index, "value", index))


def story_paragraphs(psd: PSDImage, index: Optional[int]) -> list:
    """The paragraph properties of Txt2 story index (Photoshop CC's numbered keys); [] without it."""
    root = _txt2(psd)
    if root is None or index is None:
        return []
    _, _, items, model, _ = _story_list(root)
    if not 0 <= index < len(items):
        return []
    return [run["0"]["0"]["5"] for run in items[index][model]["5"]["0"]]


def point_text(layer) -> bool:
    """No box: a line ends only at a return, whatever the composer."""
    shapes = layer._engine_data["EngineDict"]["Rendered"]["Shapes"]["Children"]
    return all(int(getattr(shape["Cookie"]["Photoshop"]["ShapeType"], "value",
                           shape["Cookie"]["Photoshop"]["ShapeType"])) == 0 for shape in shapes)


def unset_composer(psd: PSDImage, layer) -> list:
    """(run, properties) of box text's story paragraphs without COMPOSER_KEY (Photoshop lays them out Every-line)."""
    try:
        paragraphs = [] if point_text(layer) else story_paragraphs(psd, story_index(layer))
    except Refused:                             # no text objects: none known
        return []
    return [(run, props) for run, props in enumerate(paragraphs) if COMPOSER_KEY not in props]


def set_single_line(psd: PSDImage, layer) -> list:
    """COMPOSER_KEY False where unset_composer finds none, as the fork lays it out; the runs set."""
    from psd_tools.psd import engine_data
    missing = unset_composer(psd, layer)
    for _, props in missing:
        if COMPOSER_KEY not in props:           # paragraphs may share one
            props[engine_data.Property(COMPOSER_KEY)] = engine_data.Bool(False)
    return [run for run, _ in missing]


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


def story_style_drift(psd: PSDImage, kerning: bool = True) -> list:
    """
    (layer name, why) per type layer whose Txt2 story holds its text but other styles: per character, each
    STORY_MIRROR key both hold must agree (a font by name, a color by its values). Photoshop would show the story's
    styles, so a fill written to TySh alone would silently go back. Keys a story run leaves out (inherited) are skipped.
    With kerning, also the kerning of each character that ends a pair (_kerning; B12: Pandamonium's Health).
    """
    root = _txt2(psd)
    if root is None:
        return []
    _, _, items, model_key, text_key = _story_list(root)
    keys = STORY_STYLE_KEYS
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
                feature = str(number)
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
        kerned = _kerning(layer, model, keys, root) if kerning and not why else None
        drift = [(u, t, k) for u, (t, k, pair) in enumerate(kerned or []) if pair and k is not None and t != k]
        if drift:
            u, t, k = drift[0]
            why = f"AutoKerning at {u}: layer {t!r}, story {k!r}"
        if why:
            out.append((layer.name, why))
    return out


def _sheet_kerning(root, sheet) -> Optional[bool]:
    """A numbered Txt2 style sheet's kerning, else its parents' (Normal last); None when none sets it."""
    try:
        sheets = root["0"]["5"]["0"]
    except (KeyError, TypeError):
        return None
    seen = set()
    while sheet is not None and 0 <= int(sheet) < len(sheets) and int(sheet) not in seen:
        seen.add(int(sheet))
        data = sheets[int(sheet)]["0"]
        features = data.get("6")
        value = features.get(KERN_FEATURE) if hasattr(features, "get") else None
        if value is not None:
            return bool(getattr(value, "value", value))
        sheet = getattr(data.get("5"), "value", data.get("5"))
    return None


def _kerning(layer, model, keys: dict, root) -> list:
    """
    Per UTF-16 unit of a type layer's text: (TySh's AutoKerning, its Txt2 story's kerning or None when neither the
    story run nor its style sheets set it, whether the unit ends a pair: no paragraph's first). A run leaving it out
    takes its sheet's (_sheet_kerning; Normal kerns): a paragraph's first letter, often unkerned in TySh, passes that
    to text typed before it, which Photoshop kerns (Gulka re-wrapped). Photoshop kerns as the story says, the engine
    as TySh does.
    """
    text = str(layer.engine_dict["Editor"]["Text"].value)
    chars, second = [], []                      # per unit: its character, whether it is an astral one's second half
    for c in text:
        chars += [c] * (2 if ord(c) > 0xFFFF else 1)
        second += [False, True][:2 if ord(c) > 0xFFFF else 1]
    resources = layer.resource_dict
    try:
        normal = resources["StyleSheetSet"][int(getattr(resources["TheNormalStyleSheet"], "value",
                                                        resources["TheNormalStyleSheet"]))]["StyleSheetData"]
        default = bool(getattr(normal.get("AutoKerning", True), "value", normal.get("AutoKerning", True)))
    except (KeyError, IndexError, TypeError, ValueError):
        default = True
    style = layer.engine_dict["StyleRun"]
    tysh = [default] * len(chars)
    for a, b, data in _style_spans([(int(n), run["StyleSheet"]["StyleSheetData"])
                                    for n, run in zip(style["RunLengthArray"], style["RunArray"])]):
        value = data.get("AutoKerning")
        for u in range(a, min(b, len(chars))):
            tysh[u] = default if value is None else bool(getattr(value, "value", value))
    story = [None] * len(chars)
    runs = list(_dig(model, keys["runs"]))
    spans = _style_spans([(int(run[keys["length"]].value), _dig(run[keys["data"]], keys["features"])) for run in runs])
    for (a, b, data), run in zip(spans, runs):
        value = data.get(KERN_FEATURE)
        if value is None:                       # left to its style sheet
            sheet = _dig(run[keys["data"]], keys["features"][:1]).get("5")
            kern = _sheet_kerning(root, getattr(sheet, "value", sheet))
        else:
            kern = bool(getattr(value, "value", value))
        for u in range(a, min(b, len(chars))):
            story[u] = kern
    return [(tysh[u], story[u], u > 0 and chars[u - 1] not in "\r\x03" and not second[u])
            for u in range(len(chars))]


def follow_kerning(psd: PSDImage, layer) -> bool:
    """
    The layer's TySh AutoKerning made its Txt2 story's wherever the story sets it, before the engine draws it: Photoshop
    kerns as the story says (B12: a Kung-Fu-Panda layer kerned off in its story, on in TySh, drew 0.58 IoU against its
    pixels, 0.97 followed). TySh only (the story already says it). True when it changed.
    """
    root = _txt2(psd)
    index = story_index(layer)
    if root is None or index is None:
        return False
    try:
        _, _, items, model_key, text_key = _story_list(root)
    except Refused:                             # no text objects: no story to follow
        return False
    if not 0 <= index < len(items) or str(items[index][model_key][text_key].value) != \
            str(layer.engine_dict["Editor"]["Text"].value):
        return False
    kerned = _kerning(layer, items[index][model_key], STORY_STYLE_KEYS, root)
    wrong = [(u, k) for u, (t, k, _) in enumerate(kerned) if k is not None and t != k]
    if not wrong:
        return False
    from psd_tools.api import text_edit
    from psd_tools.psd import engine_data
    section = text_edit._section(layer, "StyleRun")
    spans = text_edit._spans(section, len(kerned))
    for u, k in wrong:
        spans = text_edit._restyled(spans, u, u + 1, {"AutoKerning": engine_data.Bool(k)}, text_edit._sheet)
    text_edit._write_spans(section, spans, merge=True)
    text_edit._touch(layer)
    return True


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


# ========================
# Putting a box's edit back over its layer (PARTS_EDITOR.md §4.4)
# ========================
PAD = " \t\r\n\x03\ufeff"   # a layer's blank ends: its place on the card (Miracle Matter's '\r    Fire: 5/5/8/7')



def _align(old: str, raw: str) -> dict:
    """old index -> raw index of each character of old that raw holds (\\r and \\x03 read as \\n, tabs as spaces)."""
    seen = raw.replace("\r", "\n").replace("\x03", "\n").replace("\t", " ")
    return {a + k: b + k for a, b, size in SequenceMatcher(None, old, seen, autojunk=False).get_matching_blocks()
            for k in range(size)}


def _opcodes(old: str, new: str) -> list:
    """
    The edit's opcodes (text_align: paragraphs, then words), an insertion between kept text slid back to start with
    the nearest line break it can ("use.\\nD" + "iscard after use.\\nD" + "eal": the added line "\\nDiscard after
    use." after "use.", in its layer).
    """
    ops = [list(op) for op in text_opcodes(old, new)]
    for k in range(1, len(ops) - 1):
        (_, _, _, pj1, _), (tag, i1, _, j1, j2), nxt = ops[k - 1], ops[k], ops[k + 1]
        if tag != "insert" or ops[k - 1][0] != "equal" or nxt[0] != "equal":
            continue
        d = next((d for d in range(1, j1 - pj1 + 1) if new[j1 - d] != new[j2 - d] or new[j1 - d] == "\n"), 0)
        if not d or new[j1 - d] != "\n" or new[j1 - d] != new[j2 - d]:
            continue
        ops[k - 1][2] -= d
        ops[k - 1][4] -= d
        ops[k] = [tag, i1 - d, i1 - d, j1 - d, j2 - d]
        nxt[1] -= d
        nxt[3] -= d
    return [tuple(op) for op in ops if op[2] > op[1] or op[4] > op[3]]


def project(old: str, new: str, raw: str, brk: str = "\r") -> str:
    """
    A layer's text with the edit old -> new of what the parser read from it applied where it prints: each changed part
    spliced over the characters of raw it replaces, an insertion after the character before it; what the edit keeps,
    and the layer's own blank ends, indents and breaks between, stay as they are.

    Args:
        old: The text the parser read from raw (or a part of it)
        new: That text edited (\\n breaks)
        raw: The layer's text
        brk: What a new line becomes (\\r; a part's own \\x03: _part_raw)

    Returns:
        The layer's new text (\\r breaks)
    """
    at, end = _align(old, raw), len(raw.rstrip(PAD))
    for tag, i1, i2, j1, j2 in reversed(_opcodes(old, new)):
        if tag == "equal":
            continue
        mine = [at[i] for i in range(i1, i2) if i in at]
        if mine:
            lo, hi = mine[0], mine[len(mine) - 1] + 1
        else:
            lo = next((at[i] + 1 for i in range(i1 - 1, -1, -1) if i in at), None)
            if lo is None:
                lo = next((at[i] for i in range(i2, len(old)) if i in at), end)
            hi = lo
        raw = raw[:lo] + new[j1:j2].replace("\n", brk) + raw[hi:]
    return raw


def _layer_reading(psd: PSDImage, layer, parser: Optional[PSDParser]) -> str:
    """The ability the parser reads from one layer (_process_text_layer, _process_abilities), image icons' tokens in."""
    if parser is None:
        return normalize_ability(layer.text)
    mid, read = int(psd.height * TYPE_REGION_RATIO), []
    parser._process_text_layer(layer, CardInfo(name="", card_type=""), MutableValue(0), read, False, mid)
    icons = [(icon.name.lower(), BoundingBox.from_tuple(icon.bbox[:2])) for icon in image_icons(psd, parser)]
    return parser._process_abilities(read, icons, mid)


def from_reading(psd: PSDImage, layer, text: str, parser: Optional[PSDParser]) -> Optional[str]:
    """
    A whole-ability edit (the website's text: the parser's reading, edited) put back over the one Ability layer's own
    text (project): what the reading leaves out stays where the edit keeps it (Deepnest's "reroll :3" kept its space;
    a space before punctuation and end quotes were dropped), its breaks, tabs and blank ends too. None when the parser
    reads nothing there.
    """
    old = _layer_reading(psd, layer, parser)
    new = straight(text).replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    return project(old, new, straight(layer.text)) if old.strip() else None


def _record_ids(psd: PSDImage) -> list:
    """id() of each layer record, by index (they keep their identity through the edits: record_map)."""
    return [id(record) for record in psd._record.layer_and_mask_information.layer_info.layer_records]


def _renumber(edit: Edit, ids: list, now: dict) -> None:
    """An Edit's record indices, taken when its layer was set (ids: _record_ids then), as they are now (record_map)."""
    edit.records = [now[ids[i]] for i in edit.records]
    edit.hidden = [now[ids[i]] for i in edit.hidden]
    edit.rebuilt = [(now[ids[i]], was, name) for i, was, name in edit.rebuilt]
    edit.restyled = [(now[ids[i]], blocks) for i, blocks in edit.restyled]


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
    Whether an opaque Normal overlay effect paints over the layer's own colours (Mini Doomer's gradient): then the
    engine's one-colour icon glyphs look like Photoshop's colour ones under it.
    """
    from psd_tools.api.effects import ColorOverlay, GradientOverlay, PatternOverlay
    effects = layer.effects
    return bool(effects.enabled) and any(
        isinstance(effect, (ColorOverlay, GradientOverlay, PatternOverlay)) and effect.enabled
        and float(effect.opacity) >= 100 and effect.blend_mode == BlendMode.NORMAL for effect in effects)


def one_colour(layer) -> bool:
    """
    Whether an enabled Color, Gradient or Pattern Overlay, the layer's own or a group's above it, paints over its text
    at any opacity: then its badges and stickers take the icon font's Stylistic Alternates, one-colour shapes (an
    outlined pill, an outlined icon) the overlay colours like the letters; a colour badge would be a blob (owner
    2026-10-25: the timing colours are lost there).
    """
    from psd_tools.api.effects import ColorOverlay, GradientOverlay, PatternOverlay
    node = layer
    while node is not None and node.kind != "psdimage":
        effects = node.effects
        if effects.enabled and any(isinstance(effect, (ColorOverlay, GradientOverlay, PatternOverlay)) and effect.enabled
                                   for effect in effects):
            return True
        node = node.parent
    return False


def alternate_spans(text: str, drawn: list, icons: Optional[list] = None) -> list:
    """
    The badges and :stickers among icon_text spans: what the icon font's Stylistic Alternates draw in one colour. icons
    (_char_styles' [1]): only those set in the icon font (a badge in its body font stays as it is; review).
    """
    return [(a, b) for a, b in drawn if (text[a] == "(" or token_variant(text[a:b]) == "sticker")
            and (icons is None or all(icons[i] for i in range(a, b)))]


def set_alternates(layer, spans: list, changed: list) -> None:
    """
    Stylistic Alternates on the spans (alternate_spans, typed or kept), written to the Txt2 story alone as Photoshop
    writes them, and off on any other character the edit typed (it takes them from the badge before it).
    """
    if not spans and not changed:
        return
    on = stylistic_alternates(layer)
    if on is None:
        if spans:
            raise Refused("unsupported-card", "no Txt2 story to hold the one-colour badges; edit in Photoshop")
        return
    flagged = {i for a, b in on for i in range(a, b)}
    for a, b in spans:
        if not all(i in flagged for i in range(a, b)):
            layer.set_span_style(a, b, StylisticAlternates=True)
    want = {i for a, b in spans for i in range(a, b)}
    stray = sorted(i for i in flagged if i not in want and any(c <= i < d for c, d in changed))
    for a, b in _stretches(stray):
        layer.set_span_style(a, b, StylisticAlternates=False)


def _tinted(layer) -> Optional[str]:
    """
    What in the Ability's look changes its badges' and icons' colours: any enabled Color, Gradient or Pattern Overlay at
    any opacity (Ice Staff's two at 69%), a blend mode other than Normal or an opacity or fill below full, its own or a
    group's above it (the Isaac items' Pin Light: Soul of Eden's white-on-cyan 1CD; The Patients' 78%: a dimmed pill).
    Also a visible layer clipped to it or to such a group: an adjustment or fill (Claw's and Miasma's Exposure 1 on Text:
    a pale pill), a drawn layer over the Ability's box (Pizza's Multiply 'Layer 12'; review 2026-10-24).
    A Satin, Inner Glow or Inner Shadow paints over them too, and a group's effects as its own (review 2026-10-24: The
    Radiance's title in 'Group 1' with a Gradient Overlay printed a typed [fire] as a cream silhouette).
    Under an overlay the badges and stickers take one-colour shapes (one_colour).
    None when nothing does; else the first cause, for a refusal ("its Inner Glow", "the Gradient Overlay of its group
    'Group 1'"; review 2026-10-24: Butterfly's title refused naming an overlay it lacks).
    """
    from psd_tools.api.effects import ColorOverlay, GradientOverlay, InnerGlow, InnerShadow, PatternOverlay, Satin
    kinds = {ColorOverlay: "Color Overlay", GradientOverlay: "Gradient Overlay", PatternOverlay: "Pattern Overlay",
             Satin: "Satin", InnerGlow: "Inner Glow", InnerShadow: "Inner Shadow"}
    x0, y0, x1, y1 = layer.bbox
    node = layer
    while node is not None and node.kind != "psdimage":
        whose = (lambda what: f"its {what}") if node is layer else (lambda what: f"the {what} of its group {node.name!r}")
        effects = node.effects
        found = [name for effect in effects if effect.enabled for kind, name in kinds.items()
                 if isinstance(effect, kind)] if effects.enabled else []
        if found:
            return whose(found[0])
        if node.blend_mode not in (BlendMode.NORMAL, BlendMode.PASS_THROUGH):
            return whose(f"{node.blend_mode.name.replace('_', ' ').title()} blending")
        if node.opacity < 255 or node.fill_opacity < 255:
            return whose("opacity or fill below full")
        for clipped in _clip_chain(node) if node.parent is not None else ():
            if not clipped.is_visible():
                continue
            a0, b0, a1, b1 = clipped.bbox
            if clipped.kind not in DRAWN_LAYER_KINDS or (a0 < x1 and x0 < a1 and b0 < y1 and y0 < b1):
                return f"{clipped.name!r}, clipped to {'it' if node is layer else f'its group {node.name!r}'}"
        node = node.parent
    return None


def uses_icon_font(layer) -> bool:
    """Whether any run of the text layer is set in the stoogedings icon font."""
    return icon_font(layer) is not None


def icon_text(text: str, vocab: Optional[IconVocabulary]) -> list:
    """
    [(start, end)] of the text the stoogedings font draws as a badge or an icon, as its GSUB ligates it: a timing tag
    in parentheses (badge_tags(), any letter case: "(1cd)" is (1CD)) and a [token] the vocabulary knows with a variant
    the font has (token_variant: "[Fire:Sticker]", "[warp star]", "[fire:no]", not "[fire:]") and an icon character it
    maps ("★", the gold star; the website writes "5★"); every [token] and icon character when vocab is None.
    """
    out = badge_spans(text)
    for match in ICON_SCAN_PATTERN.finditer(text):
        piece = match.group(0)
        if not piece.startswith("["):
            name = vocab.codepoints.get(piece) if vocab is not None else piece
            if name and (vocab is None or vocab.lookup(name) is not None):
                out.append(match.span())
            continue
        if vocab is None or (vocab.lookup(piece[1:-1].partition(":")[0]) is not None and token_variant(piece) is not None):
            out.append(match.span())
    return sorted(out)


def canonical_tokens(text: Optional[str], vocab: Optional[IconVocabulary], chars: bool = True) -> str:
    """
    The text with each [token] the vocabulary knows as the parser reads it in the icon font (_read_icon_tokens):
    "[Fire:Sticker]" -> "[fire]", "[warp star]" -> "[warpstar]", "[fire:NO]" -> "[fire:no]", and with chars each icon
    character too ("★" -> "[star]"); unknown ones, and spellings the font draws as letters ("[fire:]"), as typed.
    """
    def one(match) -> str:
        piece = match.group(0)
        if not piece.startswith("["):
            name = vocab.codepoints.get(piece) if vocab is not None and chars else None
            return f"[{name}]" if name and vocab.lookup(name) is not None else piece
        hit = vocab.lookup(piece[1:-1].partition(":")[0]) if vocab is not None else None
        variant = token_variant(piece)
        if hit is None or variant is None:
            return piece
        return f"[{hit[0]}:no]" if variant == "no" else f"[{hit[0]}]"
    return ICON_SCAN_PATTERN.sub(one, text or "")


def unlettered_icons(text: str, parser: Optional[PSDParser]) -> list:
    """
    The [tokens] of text the website draws as an icon that the icon font prints as letters: text.js's icon() takes the
    base before the first colon, norm()ed (WEB_NORM), as any icon's name or alias whatever the variant, where the font
    ligates only exact spellings (icon_text): "[fire:]", "[fire: no]", "[fire.]", "[warp  star]", a Types/ icon the
    font lacks. Review 2026-10-24: "[fire:]" baked as the letters while the editor showed the fire icon.
    """
    vocab = getattr(parser, "icons", None)
    if vocab is None:
        return []
    known = {WEB_NORM.sub("", name.lower()) for name in (*vocab.tokens, *parser.all_types)}
    drawn = set(icon_text(text, vocab))
    return [match.group(0) for match in TOKEN.finditer(text) if match.span() not in drawn
            and WEB_NORM.sub("", match.group(0)[1:-1].split(":")[0].lower()) in known]


def keep_tokens(layer, new: str, vocab: Optional[IconVocabulary]) -> str:
    """
    The new text with the layer's own spelling of each icon it prints where the edit kept it. The parser reads an icon
    set in the icon font as its canonical token ("[dark:sticker]", "[Fire]", "★" -> "[dark]", "[fire]", "[star]";
    psd_parser._read_icon_tokens) and the website shows and sends that, so setting the website's text as given would
    redraw Palace of Shadow's stickers as plain icons (review 2026-10-23). A token the edit typed stays as typed.

    Args:
        layer: The text layer, before the edit
        new: The text to set (breaks already the layer's: keep_breaks, keep_ends)
        vocab: The parser's IconVocabulary; None: new as given

    Returns:
        new, each old icon whose canonical token the diff pairs with one in new spelt as the layer spells it
    """
    old = layer.text
    if vocab is None:
        return new
    icons = _char_styles(layer)[1]
    out, spots, last, size = [], [], 0, 0
    for match in ICON_SCAN_PATTERN.finditer(old):
        a, b = match.span()
        piece = match.group(0)
        if not all(icons[i] for i in range(a, b)):
            continue
        if piece.startswith("["):
            token = canonical_tokens(piece, vocab)
        else:
            name = vocab.codepoints.get(piece)
            token = f"[{name}]" if name and vocab.lookup(name) else piece
        if token == piece:
            continue
        out.append(old[last:a])
        size += a - last
        spots.append((size, size + len(token), piece))
        out.append(token)
        size += len(token)
        last = b
    if not spots:
        return new
    seen = "".join(out) + old[last:]

    def units(text):
        """Each [token] or icon character one unit, every other character its own: a diff of characters paired a
        typed token's ']' with the kept one's (Error Room's "[unknown] [fire]:" lost its sticker; review 2026-10-24)."""
        cut, at = [], 0
        for match in ICON_SCAN_PATTERN.finditer(text):
            cut += [(k, k + 1) for k in range(at, match.start())] + [match.span()]
            at = match.end()
        return cut + [(k, k + 1) for k in range(at, len(text))]
    a_cut, b_cut = units(seen), units(new)
    back = []
    for i, j, n in SequenceMatcher(None, [seen[x:y] for x, y in a_cut], [new[x:y] for x, y in b_cut],
                                   autojunk=False).get_matching_blocks():
        for start, end, piece in spots:
            for k in range(i, i + n):
                if a_cut[k] == (start, end):
                    back.append((*b_cut[j + k - i], piece))
    for start, end, piece in sorted(back, reverse=True):
        new = new[:start] + piece + new[end:]
    return new


def icon_style(layer, badge: bool, vocab: Optional[IconVocabulary]) -> Optional[dict]:
    """
    The style a typed badge or token takes (owner 2026-10-23): the layer's own icon-font badges' or tokens' STYLE_KEYS
    (size, baseline shift, tracking, ...; the other kind's when it has none of this kind) at their commonest size (owner
    2026-10-24: Pizza's 32 pt, not its odd 18.66 pt first badge), ties going to the body text's commonest size, then to
    the first; of those, the commonest style, then the first. None when no badge or token of the layer is set in the
    icon font: the caller sets it in ICON_FONT at the text's own style, the most common card style (measured on main
    2026-10-23: 358 of 659 icon-font runs share every key with the text beside them, 442 its size; the templates have
    none). Before the text changes.
    """
    text = layer.text
    _, icons, styles, _ = _char_styles(layer)
    found = [(text[a] == "(", styles[a]) for a, b in icon_text(text, vocab)
             if styles[a] and all(icons[i] for i in range(a, b))]
    pool = [style for kind, style in found if kind == badge] or [style for _, style in found]
    if not pool:
        return None
    body = Counter(_canon(styles[i].get("FontSize")) for i in range(len(text))
                   if not icons[i] and styles[i] and text[i] not in " \t\r\x03").most_common(1)
    sizes = Counter(_canon(style.get("FontSize")) for style in pool)
    size = max(sizes, key=lambda key: (sizes[key], bool(body) and key == body[0][0]))
    pool = [style for style in pool if _canon(style.get("FontSize")) == size]
    count = Counter(_canon(style) for style in pool)
    return deepcopy(max(pool, key=lambda style: count[_canon(style)]))


@lru_cache(maxsize=None)
def cap_of(path: str, index: int = 0) -> tuple:
    """(units per em, capital height, capital middle) of a face: its drawn H's ink (OS/2 sCapHeight lies on 7 fonts),
    else sCapHeight, else 0.7 em."""
    from fontTools.pens.boundsPen import BoundsPen
    from fontTools.ttLib import TTFont
    font = TTFont(path, fontNumber=index, lazy=True)
    upem = font["head"].unitsPerEm
    try:
        cmap, glyphs = font.getBestCmap(), font.getGlyphSet()
        pen = BoundsPen(glyphs)
        glyphs[cmap[ord("H")]].draw(pen)
        if pen.bounds and pen.bounds[3] > pen.bounds[1]:
            low, high = pen.bounds[1], pen.bounds[3]
            return upem, high - low, (high + low) / 2
    except (KeyError, TypeError, AttributeError):
        pass
    os2 = font["OS/2"] if "OS/2" in font else None
    cap = getattr(os2, "sCapHeight", 0) if os2 is not None and os2.version >= 2 else 0
    cap = cap if cap > 0 else 0.7 * upem
    return upem, cap, cap / 2


class _GlyphBoxes:
    """gid -> (glyph name, (xmin, ymin, xmax, ymax) or None) of the icon font, each drawn when first asked."""

    def __init__(self, path: str):
        from fontTools.pens.boundsPen import BoundsPen
        from fontTools.ttLib import TTFont
        font = TTFont(path, lazy=True)
        self.glyphs, self.order = font.getGlyphSet(), font.getGlyphOrder()
        self.pen, self.lock, self.boxes = BoundsPen, Lock(), {}
        font.close()                            # the outlines are read: no file held open

    def get(self, gid: int, default=None):
        with self.lock:
            if gid not in self.boxes:
                if not 0 <= gid < len(self.order):
                    return default
                pen = self.pen(self.glyphs)
                self.glyphs[self.order[gid]].draw(pen)
                self.boxes[gid] = (self.order[gid], pen.bounds)
            return self.boxes[gid]


@lru_cache(maxsize=None)
def _glyph_boxes(path: str) -> _GlyphBoxes:
    """The icon font's _GlyphBoxes, one per process."""
    return _GlyphBoxes(path)


def _drop(style, model) -> float:
    """pt a sub- (down) or superscript (up) run's baseline sits below the line's: Photoshop's 33.3% of the run size."""
    return {1: -1.0, 2: 1.0}.get(style.baseline, 0.0) * model.sub_position * style.size * style.vertical_scale


def _shrink(style, model) -> float:
    """What sub- and superscript draw a run at: Photoshop's 58.3%."""
    return model.sub_size if style.baseline in (1, 2) else 1.0


def badge_fit(model, badge, text_style, cap: tuple, box: tuple = BADGE_PILL, resize: bool = True) -> tuple:
    """
    (FontSize, BaselineShift) that set a badge's pill (box: ymin, ymax in 1000 units) at BADGE_RATIO times the text's
    capital height (resize) and its middle at the text's capital middle, both drawn as Photoshop draws them (sub- and
    superscript included). cap: cap_of() of the text's face.
    """
    upem, height, middle = cap
    em = text_style.size * _shrink(text_style, model) * text_style.vertical_scale
    target = -_drop(text_style, model) + text_style.baseline_shift + middle / upem * em
    size = badge.size
    if resize:
        size = BADGE_RATIO * height / upem * em / ((box[1] - box[0]) / 1000 * _shrink(badge, model) * badge.vertical_scale)
    sized = replace(badge, size=size)
    shift = target + _drop(sized, model) - (box[0] + box[1]) / 2000 * size * _shrink(badge, model) * badge.vertical_scale
    return round(size, 2), round(shift, 2)


def _neighbour(text: str, styles: list, icons: list, a: int, b: int, lines) -> Optional[int]:
    """The nearest letter or digit of the text beside a badge: in its line, left then right, then its paragraph, then
    the layer."""
    lo, hi = 0, len(text)
    for line in lines:
        if line.start <= a < max(line.end, line.start + 1):
            lo, hi = line.start, line.end
            break

    def ok(k):
        return not icons[k] and styles[k] is not None and text[k].isalnum()

    for k in [*range(a - 1, lo - 1, -1), *range(b, hi)]:
        if ok(k):
            return k
    start, end = text.rfind("\r", 0, a) + 1, text.find("\r", b)
    end = len(text) if end < 0 else end
    for k in [*sorted(range(start, end), key=lambda k: abs(k - a)), *sorted(range(len(text)), key=lambda k: abs(k - a))]:
        if ok(k):
            return k
    return None


def badge_fits(layer, fonts, spans: Optional[list] = None) -> list:
    """
    Each badge and icon the icon font draws in a text layer, measured (owner 2026-10-25): its run's FontSize and
    BaselineShift, the badge_fit values (icons and stars keep their size: only shifted), how far its middle sits from
    the text's capital middle (px, % of the text's em) and its size from the computed one. spans: only those (start,
    end). [] when the layer has none; raises MissingFontError.

    Returns:
        [{"start", "end", "token", "kind" (badge, icon), "size", "shift", "fit_size", "fit_shift", "off_px", "off_pct",
          "size_off"}]
    """
    from psd_tools.composite.text.layout import baseline_shift, layout
    from psd_tools.composite.text.model import TextModel
    model = TextModel.from_layer(layer)
    text = model.text
    style_at, icons = [None] * len(text), [False] * len(text)
    for run in model.runs:
        for k in range(max(run.start, 0), min(run.end, len(text))):
            style_at[k], icons[k] = run.style, run.style.font.lower().startswith(ICON_FONT_PREFIX)
    found = [(a, b) for a, b in icon_text(text, None) if all(icons[k] for k in range(a, b))
             and (spans is None or (a, b) in spans)]
    if not found:
        return []
    placed = layout(model, fonts)
    scale = hypot(model.transform[2], model.transform[3])
    out = []
    for a, b in found:
        bottom = top = None
        for glyph in placed.glyphs:
            if not a <= glyph.cluster < b or not glyph.face.key[0].lower().endswith(".ttf"):
                continue
            name, bounds = _glyph_boxes(glyph.face.key[0]).get(glyph.gid, (None, None))
            if not name or not bounds or not name.startswith(("badge_", "icon_")) or any(s in name for s in FIT_SKIP):
                continue
            bottom = bounds[1] if bottom is None else min(bottom, bounds[1])
            top = bounds[3] if top is None else max(top, bounds[3])
        if bottom is None:                      # drawn as letters: no badge or icon to fit
            continue
        badge = text[a] == "("
        k = _neighbour(text, style_at, icons, a, b, placed.lines)
        if k is None:
            continue
        here, beside = style_at[a], style_at[k]
        hit = fonts.find(beside.font)
        face = (hit[0], hit[1]) if hit else (fonts.resolve(beside.font).face.key if fonts.resolve(beside.font).face else None)
        if face is None:
            continue
        cap = cap_of(*face)
        box = BADGE_PILL if badge else (bottom, top)
        size, shift = badge_fit(model, here, beside, cap, box, resize=badge)
        em = beside.size * _shrink(beside, model) * beside.vertical_scale
        now = -baseline_shift(here, model) + (box[0] + box[1]) / 2000 * here.size * _shrink(here, model) * here.vertical_scale
        off = now - (-baseline_shift(beside, model) + cap[2] / cap[0] * em)
        out.append({"start": a, "end": b, "token": text[a:b], "kind": "badge" if badge else "icon",
                    "size": round(here.size, 3), "shift": round(here.baseline_shift, 3), "fit_size": size,
                    "fit_shift": shift, "own_shift": badge_fit(model, here, beside, cap, box, resize=False)[1],
                    "off_px": round(off * scale, 2), "off_pct": round(100 * off / em, 2) if em else 0.0,
                    "size_off": round(here.size / size - 1, 4) if size else 0.0})
    return out


def badge_wants(fit: dict) -> dict:
    """The style keys a measured badge or icon (badge_fits) needs: FontSize and BaselineShift for a badge past
    BADGE_RESIZE, else BaselineShift at its own size past FIT_PX and FIT_PCT; {} when it fits."""
    if fit["kind"] == "badge" and abs(fit["size_off"]) > BADGE_RESIZE:
        return {"FontSize": fit["fit_size"], "BaselineShift": fit["fit_shift"]}
    if abs(fit["off_px"]) > FIT_PX and abs(fit["off_pct"]) > FIT_PCT:
        return {"BaselineShift": fit["own_shift"]}
    return {}


def variant_fills(layer, vocab: Optional[IconVocabulary]) -> list:
    """[(icon name, variant, fill)] of the layer's :sticker and :tint tokens set in the icon font, in text order."""
    if layer is None or vocab is None:
        return []
    text = layer.text
    fills, icons = _char_styles(layer)[:2]
    out = []
    for a, b in icon_text(text, vocab):
        piece = text[a:b]
        variant = token_variant(piece) if piece.startswith("[") else None
        if variant in TEXT_COLOURED and fills[a] is not None and all(icons[i] for i in range(a, b)):
            out.append((vocab.lookup(piece[1:-1].partition(":")[0])[0], variant, fills[a]))
    return out


def variant_fill(piece: str, vocab: Optional[IconVocabulary], found: list, flag) -> Optional[tuple]:
    """
    The fill a typed :sticker or :tint token takes (owner 2026-10-24): the colour of its kind already on the card, the
    same icon's first (Palace of Shadow's [dark:sticker] white, its [light:sticker] the body's grey), the edited text's
    before the other's. None: the surrounding text's (edit_styles), as on a card with none.

    Args:
        piece: The typed token ("[dark:sticker]")
        vocab: The parser's IconVocabulary
        found: [variant_fills() of the edited layer, of the card's other text layer]
        flag: set_text_layer's flag(code, why): colours of its kind that tie are told, the first taken

    Returns:
        RGB 0..1 or None
    """
    variant = token_variant(piece) if piece.startswith("[") and vocab is not None else None
    if variant not in TEXT_COLOURED:
        return None
    name = vocab.lookup(piece[1:-1].partition(":")[0])[0]
    for fills in found:
        for same in ([fill for n, v, fill in fills if (n, v) == (name, variant)], [fill for _, v, fill in fills if v == variant]):
            top = Counter(same).most_common(2)
            if len(top) > 1 and top[0][1] == top[1][1]:
                flag("unsupported-card", f"{piece}: the card's :{variant} tokens are in as many of one colour as of "
                                         "another; the typed one takes the first")
            if top:
                return top[0][0]
    return None


def _other_text(psd: PSDImage, which: str, card: str):
    """The card's other text layer: the Ability for a title edit, the title for an ability edit; None when none reads."""
    try:
        return ability_layer(psd) if which == "name" else name_layer(psd, card)
    except Refused:
        return None


def image_icons(psd: PSDImage, parser: PSDParser) -> list:
    """
    The type icon image layers the parser reads into the ability's gaps (PSDParser._process_layer: named after a Types/
    icon, visible, with pixels, its top at or below the midline, under no stat row): on such a card the website's
    ability holds their [type] tokens, which a text edit would type a second time.
    """
    mid = int(psd.height * TYPE_REGION_RATIO)
    return [layer for layer in psd.descendants()
            if not (EFFECT_LAYER_TAG in layer.name and not layer.is_group()) and not _is_text_layer(layer)
            and not parser._is_stat_layer(layer) and layer.name.lower() in parser.all_types and layer.is_visible()
            and layer.has_pixels() and layer.bbox[1] >= mid]


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
    for name, layer, fill in _status_layers(chart):
        blocks = {key: layer.tagged_blocks[key] for key in EFFECT_BLOCKS if key in layer.tagged_blocks}
        out[name.lower()] = StatusColor(name, fill, bool(_enabled_effects(layer)), blocks)
    return out


def _status_layers(chart: PSDImage):
    """(name, text layer, first fill) of each status in the chart's Buffs and Debuffs groups."""
    for group in chart:
        if not group.is_group() or group.name.strip() not in ("Buffs", "Debuffs"):
            continue
        for layer in group:
            if not _is_text_layer(layer):
                continue
            name = collapse(layer.text)
            fills = [fill for _, _, fill in run_fills(layer) if fill]
            if name and fills:
                yield name, layer, fills[0]


def status_inks(chart: PSDImage) -> dict:
    """
    {status name: "#rrggbb"}: the colour each chart status prints in (the site's mock-up; Card sync's record). Its fill,
    under its enabled Normal Gradient then Color Overlays at their opacity, a gradient as its mean colour.
    """
    from psd_tools.api.effects import ColorOverlay, GradientOverlay
    from psd_tools.composite import paint
    out = {}
    for name, layer, fill in _status_layers(chart):
        ink = np.array(fill[-3:], np.float64)
        effects = list(layer.effects) if layer.effects.enabled else []
        for kind in (GradientOverlay, ColorOverlay):            # Photoshop paints a Color Overlay over a Gradient one
            for effect in effects:
                if not (isinstance(effect, kind) and effect.enabled and effect.blend_mode == BlendMode.NORMAL):
                    continue
                alpha = float(effect.opacity) / 100
                if kind is ColorOverlay:
                    rgb = [effect.color.get(key) for key in (b"Rd  ", b"Grn ", b"Bl  ")]
                    if None in rgb:                             # not an RGB colour
                        continue
                    color = np.array([float(value) / 255 for value in rgb])
                else:
                    ramp, ramp_alpha = paint._make_gradient_color(ColorMode.RGB, effect.gradient,
                                                                  paint._gradient_method(effect.descriptor))
                    if ramp is None:
                        continue
                    along = np.linspace(0.0, 1.0, 256)
                    color = np.asarray(ramp(along), np.float64).reshape(-1, 3).mean(0)
                    alpha *= float(np.mean(ramp_alpha(along))) if ramp_alpha is not None else 1.0
                ink = ink * (1 - alpha) + color * alpha
        out[name] = _hex(ink)
    return out


def status_spans(text: str, chart: dict, icons: list) -> list:
    """
    Every status name in a text, longest names first so "Frozen In Time" is not read as "Frozen"; blanks or forced
    breaks between its words, as Effect Layers.jsx's phraseRanges; none inside what the icon font draws ("[stuffed
    star]" is an icon, not the Stuffed status: its swatch showed on a :tint or :sticker one; review 2026-10-24), but
    one in brackets the font draws as letters is ("[Blazing]": Effect Layers.jsx styles it; it baked unstyled).

    Args:
        text: Ability text
        chart: status_fills()
        icons: [(start, end)] of the text the icon font draws: icon_text() of a text to set, icon_spans() of a layer's

    Returns:
        [(start, end, StatusColor)] sorted by start, non-overlapping
    """
    taken = [(start, end, None) for start, end in icons]
    for key in sorted(chart, key=len, reverse=True):
        words = r"[\s\x03]+".join(escape(word) for word in chart[key].name.split())
        pattern = re_compile(r"(?<![A-Za-z])" + words + r"(?![A-Za-z])", IGNORECASE)
        for match in pattern.finditer(text):
            if not any(match.start() < end and start < match.end() for start, end, _ in taken):
                taken.append((match.start(), match.end(), chart[key]))
    return sorted((span for span in taken if span[2] is not None), key=lambda span: span[0])


def status_word_spans(text: str, chart: dict, icons: list) -> list:
    """status_spans() whose first letter is a capital: the words that take the status look, a lowercase use ("an empty
    slot") being prose (owner 2026-11-07)."""
    return [span for span in status_spans(text, chart, icons) if text[span[0]].isupper()]


def icon_spans(layer) -> list:
    """[(start, end)] of the text layer's characters set in the icon font, in a row (_char_styles)."""
    return _stretches([i for i, icon in enumerate(_char_styles(layer)[1]) if icon])


def body_fill(layer, chart: dict) -> Optional[tuple]:
    """
    The text's own color: the fill covering most characters outside the status words.

    Args:
        layer: A text layer
        chart: status_fills()

    Returns:
        RGB 0..1, or None when the layer has no filled run
    """
    spans = status_spans(layer.text, chart, icon_spans(layer))
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
    the edit keeps (a kept break whatever its length), and the changed parts between them (one side may be empty: an
    insertion or a deletion). The alignment is text_align's over the span's whole paragraphs: a changed part never
    spans two paragraphs, and a deleted or added paragraph is a part of its own (Pot Full of Spiders!, 2026-11-03).

    Args:
        old: The layer's text
        new: The text to set
        start, end: The edited span of new (changed_range, maybe widened to whole words)

    Returns:
        [(old start, old end, new start, new end, kept)]
    """
    # aligned over the whole text (text_align: a new line never takes the line above's leftover letters, Pot Full of
    # Spiders! 2026-11-03; a common "." at the end is no sign two paragraphs are one, Azamoth's swapped lines); kept
    # text outside the span left out
    out = []
    for tag, a0, a1, b0, b1, group in text_segments(old, new):
        if tag == "equal" and (a1 - a0 >= SAME_RUN or any(c in TEXT_BREAKS for c in old[a0:a1])):
            out.append([a0, a1, b0, b1, True, group])
        elif out and not out[-1][4] and out[-1][5] == group:     # a short kept stretch joins its paragraph's change
            out[-1][1], out[-1][3] = a1, b1
        else:
            out.append([a0, a1, b0, b1, False, group])
    out = [_inside(tuple(part[:5]), start, end) for part in out]
    out = [part for part in out if part is not None]
    for k, (a0, a1, b0, b1, kept) in enumerate(out):     # a deletion slid to whole lines or words (deletion_start)
        if kept or b1 > b0 or a1 <= a0:
            continue
        floor = max([part[1] for part in out[:k] if not part[4]] or [0])
        shift = a0 - _slide(old, a0, a1, floor)
        if not shift or k and not out[k - 1][4] or k + 1 < len(out) and not out[k + 1][4]:
            continue                            # slides only between kept text
        out[k] = (a0 - shift, a1 - shift, b0 - shift, b0 - shift, False)
        if k + 1 < len(out):
            na0, na1, nb0, nb1, _ = out[k + 1]
            out[k + 1] = (na0 - shift, na1, nb0 - shift, nb1, True)
        else:
            out.append((a1 - shift, a1, b0 - shift, b0, True))
        if k:
            pa0, pa1, pb0, pb1, _ = out[k - 1]
            out[k - 1] = (pa0, pa1 - shift, pb0, pb1 - shift, True)
    return [part for part in out if part[1] > part[0] or part[3] > part[2]]


def _inside(part: tuple, start: int, end: int) -> Optional[tuple]:
    """A kept part trimmed to the new span [start, end) (None when outside it); a changed part as it is."""
    a0, a1, b0, b1, kept = part
    if not kept:
        return part
    lo, hi = max(b0, start), min(b1, end)
    return (a0 + lo - b0, a1 - (b1 - hi), lo, hi, True) if hi > lo else None


def splice_text(layer, text: str, parts: list) -> None:
    """
    set_text(keep="runs") one changed part at a time (edit_parts), last first: the engine restyles only what changed, so
    an edit at both ends of an ability leaves the badges and colours between them alone (in one call, everything from
    the first change to the last would take the first changed character's style). A part the engine's greedy prefix
    would run past (deleting "+2 ATK\r" before "+25% SPD" kept "+2" in the deleted line's red: review 2026-10-24) first
    has its first character made one no text holds, so the engine's prefix stops where the part starts.
    """
    current = layer.text
    for a0, a1, b0, b1, kept in reversed(parts):
        if not kept:
            nxt = current[:a0] + text[b0:b1] + current[a1:]
            if a1 - a0 != b1 - b0 and changed_range(current, nxt)[0] > a0 + b1 - b0:
                mark = next(c for c in "\ufffc\ufffd\u2060\u2061" if c not in current + text)
                layer.set_text(current[:a0] + mark + current[a0 + (a1 > a0):], keep="runs")
            layer.set_text(nxt, keep="runs")
            current = nxt
    layer.set_text(text, keep="runs")           # nothing left but a stale story to bring along


def edit_styles(layer, parts: list, chart: Optional[dict], moves: tuple = ()) -> list:
    """
    The style of each changed part (edit_parts) of an ability edit, from the old text: a part replacing text takes the
    commonest fill of what it replaces outside status words and blanks (set in the icon font only if all of it is: a
    badge edit); an insertion, or a part replacing only those, the nearest character before it in its line (\\r or
    \\x03), else after, that is no status word, blank or badge and shares no run with a status word (Astr's "3" of its
    red "1T3 Empowered"), else its line's commonest fill, its paragraph's, the text's; its fill the body's (body_fill)
    when typed between two runs or at a line's end in a line of several colours (Binge Eater's ", 1T" after "SPD" took
    the olive). A kept status word's characters take that too: the edit resets status words, set_text_layer giving the
    new text's their swatch after. A paragraph the edit moved (moves: text_align.moves) takes its own old colours,
    letter for letter where it kept them, else its commonest (a line moved up no longer takes the colour of the line
    it lands under). Before set_text.

    Args:
        layer: The text layer
        parts: edit_parts()
        chart: status_fills(), or None
        moves: text_align.moves() of the old and new text

    Returns:
        [(new start, new end, RGB 0..1 or None, the source's STYLE_KEYS or None when it is a badge character)]
    """
    old = layer.text
    fills, icons, styles, runs = _char_styles(layer)
    status = {i for s, e, _ in (status_spans(old, chart, icon_spans(layer)) if chart else []) for i in range(s, e)}
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

    body = body_fill(layer, chart or {})

    def typed(lo: int, hi: int) -> Optional[tuple]:
        """Text typed between two runs, or at a line's end, of a line in several colours: the body fill, not the
        coloured word beside it (Binge Eater's ", 1T" after its olive "SPD"); typed inside a run: that run's."""
        inside = 0 < lo and hi < len(old) and old[lo - 1] not in "\r\x03" and old[hi] not in "\r\x03" \
            and runs[lo - 1] == runs[hi]
        top = max(old.rfind("\r", 0, lo), old.rfind("\x03", 0, lo)) + 1
        bottom = min([j for j in (old.find("\r", hi), old.find("\x03", hi)) if j >= 0] or [len(old)])
        line = {fills[j] for j in range(top, bottom) if source(j)}
        return body if not inside and len(line) > 1 else None

    out = []
    for a0, a1, b0, b1, kept in parts:
        if kept:
            spans = [(b0 + k, b0 + k + 1, near(a0 + k, a0 + k + 1), None) for k in range(a1 - a0) if a0 + k in status]
        elif b1 > b0:
            i = next((commonest(found, a0) for icon in (False, True)
                      for found in [[i for i in range(a0, a1) if source(i, icon)]] if found), None)
            spans = [(b0, b1, i, None) if i is not None else (b0, b1, near(a0, a1), typed(a0, a1))]
        else:
            spans = []
        for start, end, i, fill in spans:
            out.append((start, end, fill or (body if i is None else fills[i]), None if i is None or icons[i] else styles[i]))
    changed = {i for _, _, b0, b1, kept in parts if not kept for i in range(b0, b1)}
    for (s0, e0), (t0, t1), kept_letters in moves:
        found = [i for i in range(s0, e0) if source(i)]
        if not found:
            continue
        common = commonest(found, s0)
        mapped = {t0 + q + k: s0 + p + k for p, q, n in kept_letters for k in range(n)}
        for k in range(t0, t1):
            if k in changed:
                i = mapped.get(k, common)
                out.append((k, k + 1, fills[i], None if icons[i] else styles[i]))
    return out


def title_styles(layer, parts: list) -> list:
    """
    A title edit's counterpart of edit_styles: each changed part (edit_parts) takes the fill and STYLE_KEYS of the
    nearest old character before it that is no blank and no icon-font one, else after it. Only what the engine sets in
    the icon font outside a badge or token takes it (set_text_layer): words typed after a title's token took its font
    and refused (review 2026-10-24: "Bench [fire] II"). Before set_text.

    Returns:
        [(new start, new end, RGB 0..1 or None, STYLE_KEYS)]
    """
    old = layer.text
    fills, icons, styles, _ = _char_styles(layer)
    clean = [i for i in range(len(old)) if not icons[i] and styles[i] and old[i] not in " \t\r\x03"]
    out = []
    for a0, a1, b0, b1, kept in parts:
        if kept or b1 <= b0:
            continue
        i = next((i for i in reversed(clean) if i < a0), next((i for i in clean if i >= a1), None))
        if i is not None:
            out.append((b0, b1, fills[i], styles[i]))
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


def deletion_start(old: str, new: str, at: int) -> int:
    """
    Where a pure deletion starts among the equal windows it could be (changed_range takes the rightmost): _slide's.
    The kept characters then keep their own runs (review 2026-10-24: deleting Dragon Mount's "+2 ATK" line printed
    "+2" of "+25% SPD" in its red; a "(2CD) ..." line before a "(1CD) ..." one split the kept badge across two sizes).

    Args:
        old: The layer's text
        new: old with one span deleted
        at: changed_range's start

    Returns:
        The start in old (and new) of the deleted span
    """
    d = len(old) - len(new)
    if d <= 0 or old[:at] + old[at + d:] != new:
        return at
    return _slide(old, at, at + d, 0)


def _slide(old: str, a0: int, a1: int, floor: int) -> int:
    """
    The start of the deleted old[a0:a1] among the equal windows down to floor: the nearest to a0 that cuts whole
    paragraphs ("D\r", or "\rD" for the last), else whole words or badges ("(1CD) "), else a0.
    """
    d, starts = a1 - a0, [a0]
    while starts[-1] > floor and old[starts[-1] - 1] == old[starts[-1] - 1 + d]:
        starts.append(starts[-1] - 1)
    for cuts in ("\r\x03", " \t\r\x03"):
        def head(k):
            return k == 0 or old[k - 1] in cuts

        def tail(k):
            return k == len(old) or old[k] in cuts
        for i in starts:
            if head(i) and head(i + d) or tail(i) and tail(i + d):
                return i
    return a0


def keep_breaks(old: str, new: str, which: str) -> str:
    """
    The new text with the layer's own breaks back where the edit kept them: the website shows an ability's forced line
    break (\\x03) as a newline and its tabs as spaces, so setting its text as given would turn a soft break into a
    paragraph (Mini Doomer's "it. \\x03(BOT) Die.") or Michael's Blessing's tab columns into spaces. A name is set as
    given (its box shows its breaks).

    Args:
        old: The layer's text
        new: The text to set, ability breaks as \\r
        which: "name" or "ability"

    Returns:
        new, each \\r (ability) the diff pairs with an old \\x03 made that break and each space with an old tab that
        tab; only between the old text's characters: its blank first and last paragraphs are keep_ends' (review 2026-10-24: "Old Foxy\\r" set a typed " [fire]" on a new
        line, its space paired with the trailing \\r)
    """
    if which == "name" or ("\x03" not in old and "\t" not in old):     # a title box shows its breaks
        return new
    blank = " \t\r\x03"
    lo, hi = len(old) - len(old.lstrip(blank)), len(old.rstrip(blank))
    seen = old.replace("\x03", "\r").replace("\t", " ")
    out = list(new)
    for a, b, size in text_blocks(seen, new):
        for i in range(size):
            if old[a + i] in "\x03\t" and lo <= a + i < hi:
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


def comp_top(psd: Optional[PSDImage], stat: str) -> Optional[int]:
    """The highest layer comp of the stat's layer-comp bars; None without a PSD or such a bar."""
    names = [name for layer, _, _ in (comp_bars(psd)[stat] if psd is not None else [])
             for name in comp_names(placed_data(layer)).values()]
    return max(map(int, names)) if names else None


def classify(parsed: dict, edits: dict, psd: Optional[PSDImage] = None, parser: Optional[PSDParser] = None,
             template: Optional[tuple] = None, chart: Optional[dict] = None, caps=...) -> Classification:
    """
    Sort a card's edits: PSD-kind fields whose value differs from the fresh parse are jobs, equal ones redundant.
    A stat above 10 is website only; on a layer-comp bar, up to its top comp (comp_top). `creatures` bakes on a bar per creature (_classify_creatures). Stars and types are jobs when the card's
    layout is one plan_stars / plan_types knows; their refusals carry the field (the card's other fields still bake).

    Args:
        parsed: parse_card() of the card's PSD
        edits: The card's fields from CardEdits.parse (no _keys)
        psd: The open PSD, needed to find the title layer for a name edit (name_layer), a stat's bars and the stars
            and type icons (not changed)
        parser: offline_parser(): its icon vocabulary reads a types edit (types compare in any order and spelling)
        template: template_icon() of the card's template: where a card with no type icon takes its first one
        caps: stat_caps(psd) when the caller has read it (bar_changes: once for all its values); ... reads it here

    Returns:
        Classification
    """
    card = parsed["card"]
    out = Classification()
    derived = _classify_parts(out, parsed, edits, psd, parser)
    if caps is ...:
        asked = psd is not None and any(edits.get(k) is not None for k in (*STAT_FIELDS, "creatures"))
        caps = stat_caps(psd) if asked else None
    for name, value in edits.items():
        if name in ("parts", "colors") or name in derived:
            continue
        if KIND.get(name, "database") != "psd" or value is None or (name == "creatures" and psd is None):
            continue                            # folder/database-only fields: never touched
        elif name == "creatures":
            _classify_creatures(out, card, value, {k: edits[k] for k in STAT_FIELDS if edits.get(k) is not None}, psd, caps)
        elif name in ICON_FIELDS:
            if card.get(name) == value or (name == "types" and same_types(parser, card.get("types"), value)):
                out.redundant.append(name)
            elif name == "stars" and star_folder(_card_path(parsed)):    # a move into its "N Stars" folder (owner 10-25)
                if value < 1:
                    out.refusals.append(Refusal("unsupported-card", "stars: a card shows at least one", name))
                else:
                    out.jobs[name] = value
            elif psd is None:
                out.refusals.append(Refusal("unsupported-card", f"{name}: no PSD to place them on", name))
            else:
                try:
                    if name == "stars":
                        plan_stars(psd, parsed, value, parser)
                    else:
                        plan_types(psd, parsed, value, parser, template)
                    out.jobs[name] = value
                except Refused as refused:
                    out.refusals.append(refused.refusal)
                except Exception as error:      # noqa: BLE001 - a layout the planner trips on is that field's alone
                    out.refusals.append(Refusal("error", f"{name}: {type(error).__name__}: {error}", name))
        elif name in STAT_FIELDS:
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} {value!r}: not a stat")     # CardEdits.valid never lets one through
            top = comp_top(psd, name)
            if card.get(name) == value:
                out.redundant.append(name)
            elif caps and value > caps[name][0]:    # its own display: refused, never website only (owner 2026-11-08)
                out.refusals.append(Refusal("unsupported-card", cap_why(name, value, caps[name][0]), name))
            elif top is not None:               # what its layer comps show (0-20, raid boss 0-40)
                if value > top:
                    out.site[name] = f"{name} {value}: the bar's layer comps show up to {top}"
                else:
                    out.jobs[name] = value
            elif value > STAT_MAX:              # past what the row prints: website only (owner 2026-10-26)
                out.site[name] = f"{name} {value}: the bar's one row prints up to 10"
            else:
                out.jobs[name] = value
        elif name == "ability":                 # [Fire] for [fire] prints the same icon: redundant
            vocab = getattr(parser, "icons", None)
            was, now = (normalize_ability(canonical_tokens(text, vocab)) for text in (card.get("ability"), value))
            if was == now:
                out.redundant.append(name)
            else:
                out.jobs[name] = value
        elif name == "name":                    # redundant when the bot already shows it (the parser reads the title)
            if straight(shown_name(parsed)) == straight(collapse(value)):
                out.redundant.append(name)
            elif psd is None:
                out.refusals.append(Refusal("unsupported-card", "no PSD to find the title layer in"))
            elif placed_title(psd, _card_name(parsed), getattr(parser, "linked", None), _card_path(parsed)):
                out.jobs[name] = value          # a smart object's title: its file's text set (set_placed_title)
            elif title_in_picture(psd, _card_name(parsed)):
                out.site[name] = PICTURE_TITLE
            else:
                try:
                    name_layer(psd, _card_name(parsed))
                    out.jobs[name] = value
                except Refused as refused:
                    out.refusals.append(refused.refusal)
    if edits.get("colors"):
        _classify_colors(out, parsed, edits, psd, chart)
    return out


def _classify_colors(out: Classification, parsed: dict, edits: dict, psd: Optional[PSDImage], chart=None) -> None:
    """
    classify()'s `colors` (plan_colors): the colours to bake are the job, each refused one a refusal of the field (its
    reason; the card's other fields and colours still bake); with none to bake, the field is redundant when one was.
    A colour on a field a text edit of the entry sets is found in its new text.
    """
    if psd is None:
        out.refusals.append(Refusal("unsupported-card", "colors: no PSD to find the text layers in", "colors"))
        return
    parted = {job.get("field") for job in (out.jobs.get("parts") or {}).values() if isinstance(job, dict)}
    retyped = {f: edits.get(f) or "" for f in COLOR_FIELDS if f in out.jobs or f in parted}
    boxes = {key: job.get("text") or "" for key, job in (out.jobs.get("parts") or {}).items() if isinstance(job, dict)}
    out.colors = plan_colors(psd, edits["colors"], _card_name(parsed), retyped, chart, boxes)
    if any(how == "baked" for _, how in out.colors):    # the redundant ones too: they paint in the colours' order
        out.jobs["colors"] = [one for one, how in out.colors if how in ("baked", "redundant")]
    elif any(how == "redundant" for _, how in out.colors):
        out.redundant.append("colors")
    out.refusals += [Refusal("unsupported-card", f"colors: {color_text(one)}: {how}", "colors")
                     for one, how in out.colors if how not in ("baked", "redundant")]


def _classify_creatures(out: Classification, card: dict, wanted: list, stats: dict, psd: PSDImage,
                        caps: Optional[dict] = None) -> None:
    """
    classify()'s `creatures`: what the list changes on the card's bars, as the job [{stat: value} per creature]: a
    layer-comp bar per creature, or digit rows per creature (several_rows: each creature's own, creature_bar). A shared
    bar has one value, and every creature on it takes the one asked. What the bars can't print is website only, but
    past a card's own display (caps: stat_caps) refused.

    Args:
        card: The parse's card
        wanted: The entry's list
        stats: The entry's stat fields: creature 1's, over the list (CardEdits.apply), their own jobs
        psd: The open card
        caps: stat_caps(psd)
    """
    printed = card.get("creatures") or []
    comps = any(comp_creature(bar[0]) for bars in comp_bars(psd).values() for bar in bars)
    if len(wanted) != len(printed) or not (comps or several_rows(psd)):
        out.site["creatures"] = "creatures: each creature's stats print only on a bar per creature"
        return
    for n, one in enumerate(wanted):
        for stat in STAT_FIELDS if caps else ():
            value, cap = one.get(stat), caps[stat][n] if n < len(caps[stat]) else STAT_MAX
            if not (n == 0 and stat in stats) and isinstance(value, int) and value != printed[n].get(stat) and value > cap:
                out.refusals.append(Refusal("unsupported-card", f"creatures: {cap_why(stat, value, cap, n + 1)}", "creatures"))
                return
    job = [{} for _ in printed]
    for stat in STAT_FIELDS:
        bars = [creature_bar(psd, stat, n + 1) for n in range(len(printed))]
        bars = [-n - 1 if bar is None else bar for n, bar in enumerate(bars)]   # no one bar: set_stat says why
        asked = {bars[0]: {stats[stat]}} if stat in stats else {}
        for n, one in enumerate(wanted):
            if stat not in one or (n == 0 and stat in stats):
                continue
            value = one[stat]
            if value is None and printed[n].get(stat) is not None:
                out.site["creatures"] = f"creatures: {stat}: a bar can't show no value"
                return
            if value is not None and value != printed[n].get(stat):
                asked.setdefault(bars[n], set()).add(value)
        top = comp_top(psd, stat) if comp_bars(psd)[stat] else STAT_MAX
        for bar, values in asked.items():
            if len(values) > 1:
                out.site["creatures"] = f"creatures: {stat}: a shared bar shows one value, not {sorted(values)}"
                return
            if top is None or max(values) > top:
                out.site["creatures"] = (f"creatures: {stat} {max(values)}: the bar's layer comps show up to {top}"
                                         if comp_bars(psd)[stat] else f"creatures: {stat} {max(values)}: a creature's "
                                         f"row prints up to {STAT_MAX}")
                return
            if stat not in stats or bar != bars[0]:
                job = [{**one, stat: min(values)} if bars[n] == bar else one for n, one in enumerate(job)]
    if any(job):
        out.jobs["creatures"] = job
    else:
        out.redundant.append("creatures")


def _classify_parts(out: Classification, parsed: dict, edits: dict, psd: Optional[PSDImage],
                    parser: Optional[PSDParser]) -> set:
    """
    classify()'s `parts` (§4.3): plan_parts' jobs as the "parts" job, its impossible cases as refusals. The plain
    name or ability of a field the parts hold is derived (shown until the bake), never baked on its own.

    Returns:
        The derived fields
    """
    parts = edits.get("parts")
    if not parts:
        return set()
    derived = {part["field"] for part in parts.values()}
    if psd is None:
        out.refusals.append(Refusal("unsupported-card", "parts: no PSD to find the layers in"))
        return derived
    plan = plan_parts(psd, parsed, parts, parser)
    out.parts = plan
    out.refusals += plan.refusals
    if plan.jobs:
        out.jobs["parts"] = plan.jobs
    elif not plan.refusals and any(how == "redundant" for how in plan.outcome.values()):
        out.redundant.append("parts")
    return derived


# ========================
# FX layers (section 7 idea 14; BUG 2: rebuilt as Effect Layers.jsx's Update all)
# ========================
@dataclass
class FxPlan:
    """An FX layer of the edited ability, read before the text changes (plan_fx)."""
    layer: object
    phrases: list           # its name's phrases ("Frozen#2" keeps its number)
    kind: str               # fx_kind(): "status", "styled" or "own"
    fill: Optional[tuple] = None    # "styled", "own": its words' colour, the first drawn run (Effect Layers.jsx 8308103f)
    box: tuple = ()         # its bbox before the edit
    ranges: list = field(default_factory=list)  # phrase_ranges() of its phrases in the old text
    found: list = field(default_factory=list)   # [(phrase, its phrase_ranges() in the old text)]
    status: str = ""        # "status": its chart status, lowercased (status_fills' key)
    own_style: bool = False  # its style is no chart status's for its words ("own", "styled"; a "status" layer's of L7)
    edges: dict = field(default_factory=dict)   # "styled", "own": per old range, whether a word ends at its start, end
    mixed: bool = False     # "styled", "own": its letters (status words apart) in more than one colour


def fx_name(base: str, status: str) -> str:
    """
    The name Effect Layers.jsx gives a status word's FX layer.

    Args:
        base: The text layer's name (Ability)
        status: The chart's status name

    Returns:
        "Ability FX: Blazing", "Ability FX: Frozen"
    """
    return f"{base}{FX_TAG}{status}"


def fx_layers(psd: PSDImage, base) -> list:
    """
    The FX layers cut from a text layer ("<its name> FX: ..."), bottom to top, wherever they sit; of several text
    layers of its name (Miracle Matter's Abilities) only those on it, as the parser pairs them: the first visible one an
    FX layer overlaps, else a hidden one's, else the first visible.
    """
    prefix = base.name + FX_TAG
    found = [layer for layer in psd.descendants() if not layer.is_group() and layer.name.startswith(prefix)]
    same = [layer for layer in psd.descendants() if _is_text_layer(layer) and layer.name == base.name
            and EFFECT_LAYER_TAG not in layer.name] if found else []
    twins = [layer for layer in same if layer.is_visible()]
    if len(same) > 1 and base in twins:
        def over(fx, t):
            x0, y0, x1, y1 = fx.bbox
            return t.bbox[0] < x1 and x0 < t.bbox[2] and t.bbox[1] < y1 and y0 < t.bbox[3]

        def on(fx):
            return next((t for t in twins if over(fx, t)), None) or next((t for t in same if over(fx, t)), twins[0])
        found = [fx for fx in found if on(fx) is base]
    return found


def fx_phrases(base, fx) -> list:
    """An FX layer's phrases as Effect Layers.jsx's parseFxName reads them: "Ability FX: (1CD) | (2U)" -> ["(1CD)", "(2U)"]."""
    return [phrase.strip() for phrase in fx.name[len(base.name + FX_TAG):].split("|") if phrase.strip()]


def _numbered(phrase: str) -> tuple:
    """A phrase's words and its "#n" (0: none): "Frozen#2" -> ("Frozen", 2)."""
    numbered = fullmatch(r"(.*\S)#(\d+)", phrase)
    return (numbered.group(1), int(numbered.group(2))) if numbered else (phrase, 0)


def _edges(text: str, a: int, b: int) -> tuple:
    """Whether a word starts at a and ends at b in the text (no letter or digit beside: the JSX's whole-word rule)."""
    return not ALNUM.match(text[a - 1] if a else " "), not ALNUM.match(text[b] if b < len(text) else " ")


def _matches(text: str, words: str) -> tuple:
    """A phrase's words' matches in the text (any case, any blanks or forced breaks between them), and the whole-word ones."""
    pattern = r"[\s\x03]+".join(escape(word) for word in words.replace("\x03", " ").split())
    found, whole = [], []
    for match in re_compile(pattern, IGNORECASE).finditer(text) if pattern else ():
        a, b = match.span()
        if b > a:
            found.append((a, b))
            if all(_edges(text, a, b)):
                whole.append((a, b))
    return found, whole


def phrase_ranges(text: str, phrases: list) -> list:
    """
    Effect Layers.jsx's phraseRanges: [(start, end)] of each phrase's matches in the text, any case, any blanks or forced
    breaks between its words; whole words when the phrase occurs as one, else every match (the "1" of "3T1"); "phrase#2"
    only the 2nd of those.
    """
    out = []
    for phrase in phrases:
        words, nth = _numbered(phrase)
        found, whole = _matches(text, words)
        found = whole or found
        out += found[nth - 1:nth] if nth else found
    return out


def _canon(value):
    """A descriptor value as plain, comparable data (floats to 0.01; showInDialog left out: it draws nothing)."""
    if hasattr(value, "items"):
        return tuple(sorted((key, _canon(item)) for key, item in value.items() if key != b"showInDialog"))
    if isinstance(value, (list, ListElement)):
        return tuple(_canon(item) for item in value)
    if hasattr(value, "enum"):
        return value.typeID, value.enum
    plain = getattr(value, "value", value)
    return round(plain, 2) if isinstance(plain, float) else plain


def _style_key(blocks) -> tuple:
    """
    The enabled effects of a layer's effect blocks (lfx2 and its lmfx kin) as comparable data: equal keys draw the same
    style (Monkey D Goofy's "1#1" carries the chart's Become Back Money gradient, a disabled pattern fill apart); () for none.
    """
    out = set()
    for tag in EFFECT_BLOCKS[:3]:
        block = blocks.get(tag) if blocks else None
        data = getattr(block, "data", None)
        if data is None or not data.get(b"masterFXSwitch", True):
            continue
        for key, value in data.items():
            for item in value if isinstance(value, ListElement) else [value]:
                if hasattr(item, "items") and item.get(b"enab") and item.get(b"present", True):
                    out.add((EFFECT_MULTI.get(key, key), _canon(item)))
        if out:
            out.add((b"Scl ", _canon(data.get(b"Scl "))))
    return tuple(sorted(out))


def fx_kind(fx, base, chart: Optional[dict]) -> str:
    """
    How an ability edit makes an FX layer again (each keeps its own style, owner q6): "status" (a styled chart status), "styled" (another phrase under a styled chart
    status's style: Monkey D Goofy's "1#1"), "own" (a style no chart status has, or none: Smoldering's stroke, Mr
    Krabs' "2T"; rebuilt in it since 2026-10-23, the owner's ruling); "" when it is no pixel layer (Photoshop draws a
    type layer from its text): refused.
    """
    if fx.kind != "pixel":
        return ""
    styled = [status for status in (chart or {}).values() if status.styled]
    names = {collapse(_numbered(phrase)[0]).lower() for phrase in fx_phrases(base, fx)}   # "Frozen #2": no status (the JSX)
    if len(names) == 1 and getattr((chart or {}).get(next(iter(names))), "styled", False):
        return "status"
    key = _style_key(fx.tagged_blocks)
    return "styled" if key and any(key == _style_key(status.blocks) for status in styled) else "own"


def chart_style(base, fx, text: str, words: list):
    """The styled chart status whose style an FX layer takes (owner 2026-11-07, over q6 for chart status words): the one
    status of the status words (status_word_spans of text) its phrases hold whole; None for another phrase (its own
    style) or words of several statuses."""
    ranges = phrase_ranges(text, fx_phrases(base, fx))
    held = {status.name: status for a, b, status in words if status.styled and any(c <= a and b <= d for c, d in ranges)}
    return next(iter(held.values())) if len(held) == 1 else None


def restyle_fx(fx, status) -> Optional[dict]:
    """The chart's effect blocks on an FX layer whose style draws otherwise (_style_key): {block key: bytes} as set (the
    Edit's restyled), else None."""
    if _style_key(fx.tagged_blocks) == _style_key(status.blocks):
        return None
    for key in EFFECT_BLOCKS:
        if key in fx.tagged_blocks and key not in status.blocks:
            del fx.tagged_blocks[key]
    for key, block in status.blocks.items():
        fx.tagged_blocks[key] = deepcopy(block)
    fx._psd.mark_updated()
    return {bytes(key): block.tobytes() for key, block in status.blocks.items()}


def plan_fx(base, old_fx: list, chart: dict) -> list:
    """The FxPlans of a text layer's FX layers, before its text changes (fx_kind() of each is not "")."""
    fills, text = _char_styles(base)[0], base.text
    words = {i for a, b, _ in status_spans(text, chart, icon_spans(base)) for i in range(a, b)}
    out = []
    for fx in old_fx:
        kind, phrases = fx_kind(fx, base, chart), fx_phrases(base, fx)
        fill, edges, mixed = None, {}, False
        if kind in ANCHORED:
            drawn = [i for a, b in phrase_ranges(text, phrases) for i in range(a, b) if fills[i] is not None]
            seen = [i for i in drawn if text[i] not in " \t\r\x03"] or drawn
            fill = fills[seen[0]] if seen else None
            edges = {(a, b): _edges(text, a, b) for a, b in phrase_ranges(text, phrases)}
            mixed = len({fills[i] for i in seen if i not in words and text[i] not in " \t\r\x03"}) > 1
        found = [(phrase, phrase_ranges(text, [phrase])) for phrase in phrases]
        status = collapse(_numbered(phrases[0])[0]).lower() if kind == "status" else ""
        own = kind in ANCHORED or (kind == "status" and _style_key(fx.tagged_blocks) != _style_key(chart[status].blocks))
        out.append(FxPlan(fx, phrases, kind, fill, tuple(fx.bbox), phrase_ranges(text, phrases), found, status, own,
                          edges, mixed))
    return out


def _images(spans: list, kept: list, matches: list) -> list:
    """The matches (ranges of the new text) holding a character the edit kept (kept_chars) from one of the old spans."""
    moved = {kept[i] for a, b in spans for i in range(a, b) if kept[i] >= 0}
    return [match for match in matches if any(match[0] <= i < match[1] for i in moved)]


def _word_at(text: str, span: tuple, spans: list = ()) -> str:
    """The word a match sits in ("3T1" for the "1" of "3T1"), or the status word holding it ("Frozen In Time")."""
    a, b = span
    held = next(((s, e) for s, e, _ in spans if s <= a and b <= e and (s, e) != (a, b)), None)
    if held:
        return text[held[0]:held[1]]
    while a > 0 and ALNUM.match(text[a - 1]):
        a -= 1
    while b < len(text) and ALNUM.match(text[b]):
        b += 1
    return text[a:b]


def follow_fx(plan: FxPlan, text: str, kept: list, spans: list = ()) -> list:
    """
    Where an FX layer's phrases land in the new text, before it is set: phrase_ranges() over it, as Effect Layers.jsx's
    Update all, but a layer whose style is its own words' stays on them. A "styled" or "own" layer (its own words
    under its own colour: Goofy's "1#1"; Fire Staff's stroked Smoldering) stays on the characters it covered: a
    numbered phrase follows its occurrence, renumbered
    ("1#1" -> "1#2" when a "1" is typed before it), and leaves with it; a phrase whose kept occurrence its name no
    longer reaches (a whole-word "1" typed: the JSX's "1#1" leaves the "1" of "3T1" for it), that would spread to part
    of another word, or whose occurrence the edit changed rather than removed (letters kept, the phrase gone: Flameblade's
    "Attacks inflict 2T" made "3T"; ruling B, review 2026-10-23), that the edit makes part of a longer word (letters
    glued on where a word of it ended: "Despair" made "Despairing", Goofy's "3T1" made "3T10": half a word in its
    style) or whose every occurrence the edit deleted while typing it
    again (a moved "+2 DEF": the layer would leave its characters for typed ones) raises unsupported-card
    (set_text_layer deletes the layer, told: its words print plain). A typed
    whole-word copy beside a kept one joins it. A "status" layer (its own style, owner q6) lands only on its status's own
    words (spans): never the "Frozen" of "Frozen In Time" or of "Unfrozen", which new_fx gives a layer of the chart's
    style or none; a numbered phrase follows its occurrence as a "styled" one does ("Frozen In Time#2" -> "#3" when one
    is typed before it), an unnumbered one takes every word of its status, named by number when the JSX's phrase would
    reach other words too ("Frozen" -> "Frozen#2" in "Frozen In Time or Frozen").

    Args:
        plan: plan_fx()
        text: The text to set
        kept: kept_chars() of the old text and text
        spans: status_word_spans() over text (for a "status" layer)

    Returns:
        [(phrase, its name now or None when it left, its ranges in text, those holding characters kept from its old
        ones, why it left or was renumbered or "")]
    """
    def moving(word: str, span: tuple) -> Refused:
        return Refused("unsupported-card", f"a whole-word {word!r} in the ability would move the layer {plan.layer.name} "
                                           f"off the {word!r} of {_word_at(text, span)!r}; edit in Photoshop")

    out = []
    mine = [(a, b) for a, b, status in spans if status.name.lower() == plan.status]
    for phrase, was in plan.found:
        words, nth = _numbered(phrase)
        found, whole = _matches(text, words)
        every = whole or found
        now = every[nth - 1:nth] if nth else every
        images = _images(was, kept, found)
        why = "" if now else f"the ability has {len(every)} {words} now" if nth and every else f"{phrase} left the ability"
        name = (f"{words}#{every.index(now[0]) + 1}" if nth else phrase) if now else None
        if plan.kind in ANCHORED:
            for a, b in was:                    # "2T" made "3T": its words edited, not gone (ruling B: refuse)
                held = [kept[i] for i in range(a, b) if kept[i] >= 0 and text[kept[i]] not in " \t\r\x03"]
                if held and not _images([(a, b)], kept, found):
                    raise Refused("unsupported-card", f"{plan.layer.name}: the edit changes its {words!r} (now "
                                                      f"{_word_at(text, (min(held), max(held) + 1))!r}), which would lose "
                                                      f"the layer's style; edit in Photoshop")
            for span in images:
                if span not in every:
                    raise moving(words, span)
            for a, b in was:                    # "Despair" made "Despairing": letters glued on where a word ended
                for c, d in _images([(a, b)], kept, found):
                    if any(was_edge and not edge for was_edge, edge in zip(plan.edges.get((a, b), ()), _edges(text, c, d))):
                        raise Refused("unsupported-card", f"{plan.layer.name}: the edit makes its {words!r} part of "
                                                          f"{_word_at(text, (c, d))!r}, half a word in its style; edit "
                                                          f"in Photoshop")
            if not nth and now and not images:  # deleted and typed elsewhere: it would leave its characters
                raise Refused("unsupported-card", f"{plan.layer.name}: the edit deletes its {words!r} and types it again, "
                                                  f"which would move the layer; edit in Photoshop")
            if nth:
                now = images[:1]
                k = every.index(now[0]) + 1 if now else 0
                why = f"its {words} left the ability" if not now else f"its {words} is #{k} now" if k != nth else ""
                name = f"{words}#{k}" if now else None
            elif not whole and any(span not in images for span in now):
                new = next(span for span in now if span not in images)
                raise Refused("unsupported-card", f"the {words!r} of {_word_at(text, new)!r} would join the layer "
                                                  f"{plan.layer.name}; edit in Photoshop")
        elif plan.kind == "status":
            other = [repr(_word_at(text, span, spans)) for span in (images if nth else every) if span not in mine]
            if nth:
                now = [span for span in images if span in mine][:1]
                if now and now[0] not in every:
                    raise moving(words, now[0])
                k = every.index(now[0]) + 1 if now else 0
                why = (f"its {words} is #{k} now" if k != nth else "") if now else \
                    f"{', '.join(other)} is no {words}" if other else f"its {words} left the ability"
                name = f"{words}#{k}" if now else None
            elif other:
                now = [span for span in every if span in mine]
                why = f"{', '.join(dict.fromkeys(other))} {'is' if len(other) == 1 else 'are'} no {words}"
                name = FX_SEP.join(f"{words}#{every.index(span) + 1}" for span in now) if now else None
        out.append((phrase, name, now, [span for span in now if span in images], why))
    return out


def _fx_engine() -> tuple:
    """The fork's text engine pieces that cut glyph pixels and make or refill pixel layers."""
    from psd_tools.api.layers import PixelLayer, TypeLayer
    from psd_tools.composite.text.layout import layout
    from psd_tools.composite.text.model import TextModel
    from psd_tools.composite.text.raster import GlyphCache, Options, rasterize
    store = TypeLayer._store_raster         # a type layer's raster writer: channels and rect, the record kept
    return TextModel, layout, rasterize, Options, GlyphCache, PixelLayer, store


def _key_raster(lay, model, layer, engine: tuple) -> np.ndarray:
    """A layout rasterized by the engine, on the layer's bbox: (H, W, 4) straight RGBA."""
    rasterize, Options, GlyphCache = engine[2:5]
    lx0, ly0, lx1, ly1 = layer.bbox
    out = np.zeros((ly1 - ly0, lx1 - lx0, 4), np.float32)     # on the layer's bbox (a splice keeps Photoshop's)
    if not lay.glyphs and not lay.rules:
        return out
    rgba, bbox = rasterize(lay, model.transform, model.antialias, Options(color_glyphs=True), GlyphCache(),
                           max_alloc_bytes=layer._psd._max_alloc_bytes)
    x0, y0, x1, y1 = (int(v) for v in bbox)
    if x0 < lx0 - KEY_SLACK or y0 < ly0 - KEY_SLACK or x1 > lx1 + KEY_SLACK or y1 > ly1 + KEY_SLACK:
        raise Refused("styled-word", f"the key render {tuple(bbox)} does not line up with the text layer {tuple(layer.bbox)}")
    cx0, cy0, cx1, cy1 = max(x0, lx0), max(y0, ly0), min(x1, lx1), min(y1, ly1)
    if cx1 > cx0 and cy1 > cy0:
        out[cy0 - ly0:cy1 - ly0, cx0 - lx0:cx1 - lx0] = rgba[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    return out


def _key_ink(model, fills, layer, fonts, engine: tuple) -> np.ndarray:
    """
    A key copy of the layer's text, runs refilled by fills(run) -> [(start, end, fill)]: where its red channel reaches,
    on the layer's bbox (_key_raster).
    """
    runs = [replace(run, start=a, end=b, style=replace(run.style, fill=fill)) for run in model.runs for a, b, fill in fills(run)]
    key = replace(model, runs=tuple(runs))
    rgba = _key_raster(engine[1](key, fonts), key, layer, engine)
    return rgba[..., 0] * rgba[..., 3] >= KEY_INK      # the key over black: its red channel at Photoshop's threshold level 1


def _dilate(mask: np.ndarray) -> np.ndarray:
    """The mask grown by one pixel, the 8 neighbours."""
    out = mask.copy()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    return out


def _phrase_mask(layer, model, ranges: list, glyphs: Optional[np.ndarray], fonts, engine: tuple) -> np.ndarray:
    """
    Where the layer's text inks under ranges (indices into the engine's text, TextModel.text), over the layer's bbox:
    Effect Layers.jsx's key copy (those runs white, every other black) laid out and rasterized again; a fill changes
    nothing in the layout. The icon font's colour glyphs keep their SVG colours in any key, so glyphs, an all-black key's
    ink, takes them out (the JSX's fix for badge pixels in FX layers, 11faa634). The raster may be Photoshop's (a splice keeps it),
    whose anti-aliasing reaches a pixel past the engine's on some faces (Perpetua): the mask takes the raster's ink one
    pixel around the key's that is no other glyph's (the complement key's, grown a pixel), as the JSX's key, drawn by
    Photoshop, does.
    """
    def fills(white, black):
        def split(run):
            cuts = sorted({run.start, run.end, *(min(max(x, run.start), run.end) for span in ranges for x in span)})
            return [(a, b, white if any(s <= a and b <= e for s, e in ranges) else black)
                    for a, b in zip(cuts, cuts[1:]) if a < b]
        return split

    mask = _key_ink(model, fills(KEY_WHITE, KEY_BLACK), layer, fonts, engine)
    rest = _key_ink(model, fills(KEY_BLACK, KEY_WHITE), layer, fonts, engine)
    if glyphs is not None:
        mask &= ~glyphs
        rest |= glyphs
    rest &= ~mask
    ink = layer.numpy("shape")[..., 0] > 0
    return mask | (_dilate(mask) & ink & ~_dilate(rest))


def _cut(layer, mask: np.ndarray, color: np.ndarray, shape: np.ndarray) -> Optional[tuple]:
    """
    The layer's own pixels under the mask, as Layer via Copy makes them: (straight RGBA float32, canvas bbox) trimmed to
    the copied alpha; None when nothing inks.
    """
    rgba = np.concatenate([color, shape], axis=2).astype(np.float32)
    rgba[~mask] = 0.0
    ys, xs = (rgba[..., 3] > 0).nonzero()
    if not len(xs):
        return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    left, top = layer.bbox[:2]
    return rgba[y0:y1, x0:x1], (left + x0, top + y0, left + x1, top + y1)


def _issue_id(psd: PSDImage, layer) -> None:
    """A layer id Photoshop has not issued (past the document's seed and every id in the tree); the seed moves past it."""
    resource = psd.image_resources.get(Resource.IDS_SEED_NUMBER)
    seed = int(resource.data.value) if resource is not None else 0
    used = max((other.layer_id for other in psd.descendants()), default=0)
    new = max(seed, used) + 1
    layer.tagged_blocks.set_data(Tag.LAYER_ID, new)
    if resource is not None:
        resource.data.value = new + 1


def _cut_fx(psd: PSDImage, name: str, cut: tuple, blocks: dict, engine: tuple):
    """
    A new pixel layer holding a cut (_cut), carrying a copy of the chart layer's effect blocks.

    Returns:
        The layer, appended at the top of the document (the caller places it)
    """
    rgba, (x0, y0, _, _) = cut
    fx = engine[5].fromarray(rgba, psd, name=name, top=y0, left=x0)
    fx.name = name                              # the unicode name too (fromarray writes the Pascal one)
    for key, block in blocks.items():
        fx.tagged_blocks[key] = deepcopy(block)
    _issue_id(psd, fx)
    return fx


def kept_chars(old: str, new: str) -> list:
    """Per index of old: where the edit keeps that character in new (text_align's matching blocks), else -1."""
    out = [-1] * len(old)
    for a, b, size in text_blocks(old, new):
        out[a:a + size] = range(b, b + size)
    return out


def kept_spans(ranges: list, kept: list) -> list:
    """The ranges of the old text the edit kept whole (kept_chars: every character, in a row), as ranges of the new."""
    return [(kept[a], kept[a] + b - a) for a, b in ranges
            if kept[a] >= 0 and all(kept[i] == kept[a] + i - a for i in range(a, b))]


def _kept(plan: FxPlan, ranges: list, band: Optional[tuple], kept: list) -> bool:
    """
    Whether an FX layer stays as it is: none of its glyphs was redrawn (the splice's band misses its box) and its phrases
    cover the same characters, kept by the edit (kept_chars), as before. Update all would cut Photoshop's same pixels again.
    """
    x0, y0, x1, y1 = plan.box
    if band is not None and band[0] <= x1 and x0 <= band[2] and band[1] <= y1 and y0 <= band[3]:
        return False
    moved = kept_spans(plan.ranges, kept)
    return len(moved) == len(plan.ranges) and sorted(moved) == sorted(ranges)


def held_words(plans: list, kept: list, alive: list) -> list:
    """
    Edit.held: (its name now, kept_spans() of its old ranges) per "styled" or "own" FX layer the rebuild left in the
    card (alive: the FX layers after it). By layer, not name: a renamed layer may take a deleted one's name ("2T#2"
    -> "2T#1" when the first "2T" left).
    """
    ids = {id(fx) for fx in alive}
    return [(plan.layer.name, kept_spans(plan.ranges, kept)) for plan in plans if plan.kind in ANCHORED and id(plan.layer) in ids]


def _holds(fx, mask: np.ndarray, layer) -> bool:
    """Whether every pixel of an FX layer lies on its words' mask grown a pixel (no leaked badge, the B1 batch's L7b)."""
    alpha = _alpha(fx)
    if alpha is None:
        return False
    x0, y0, x1, y1 = fx.bbox
    lx0, ly0, lx1, ly1 = layer.bbox
    if x0 < lx0 or y0 < ly0 or x1 > lx1 or y1 > ly1:
        return False
    return not (alpha > 0)[~_dilate(mask)[y0 - ly0:y1 - ly0, x0 - lx0:x1 - lx0]].any()


def _left(left: list, every: bool = False) -> list:
    """
    Why phrases left an FX layer's name, for the report, from [(phrase, follow_fx()'s why)]: "(1GCD) left the ability",
    "the ability has 1 Frozen In Time now"; "its words left the ability" for every phrase gone that way when every.
    """
    plain = [phrase for phrase, why in left if why == f"{phrase} left the ability"]
    other = list(dict.fromkeys(why for phrase, why in left if phrase not in plain))
    if every and not other:
        return ["its words left the ability"]
    return ([f"{', '.join(plain)} left the ability"] if plain else []) + other


def _rebuild_fx(psd: PSDImage, layer, text: str, plans: list, band: Optional[tuple], kept: list, fonts,
                engine: tuple, targets: list, words: list = ()) -> tuple:
    """
    Effect Layers.jsx's Update all after an ability edit: each FX layer cut again in place from the new raster under its
    phrases' key (follow_fx: phrase_ranges over the new text, so a phrase that occurs more often is cut on every
    occurrence; a "styled" or "own" layer's stays on its words, a "status" layer's on its status's own words), its record kept
    (name, flags, opacity, fill, blending, clipping, id, place, style): only its pixels change. A layer whose words were not redrawn keeps its record whole
    (_kept) unless it holds pixels its words' cut does not (_holds: a badge an older JSX leaked into it, cut again
    clean) or covers less than FX_COVER of their ink (_fx_cover: its words re-laid in Photoshop since it was cut,
    Marx's Glued; cut again). A phrase that left the text leaves the name (the parser flags a name the text lacks; a
    renumbered one is renamed); a layer with none left is deleted. Every layer keeps its own style, the chart's
    or not (owner q6, unattended call 2026-10-23), but one on a chart status word (chart_style of words, owner
    2026-11-07) takes the chart's (restyle_fx), as a new one does (_regenerate_fx); a typed word that
    joins a layer whose style is no chart status's for it (FxPlan.own_style) is told ("moved to" it when none of the
    layer's old words stayed: Doctor Zomboss's Blazing swapped with Frostbite).

    Args:
        psd: The card
        layer: The ability layer, text set and rasterized
        text: The text set
        plans: plan_fx() of its FX layers
        band: The canvas box the rasterize redrew; None: nothing
        kept: kept_chars() of the old text and text
        fonts: The FontMap used
        engine: _fx_engine()
        targets: follow_fx() per plan
        words: status_word_spans() over text

    Returns:
        (rebuilt [(layer, old name)], deleted names, the str ranges of text the layers cover, report notes, restyled
        [(layer, its chart blocks)]); raises Refused("styled-word") for words with no pixels
    """
    from psd_tools.composite.text.geometry import text_geometry
    model = engine[0].from_layer(layer)
    offset = model.text.find(text)
    if offset < 0:
        raise Refused("styled-word", "the engine's text differs from the layer's")
    color, shape = layer.numpy("color"), layer.numpy("shape")      # None for a blank text: every layer deleted below
    icons, glyphs, geometry, ink = _char_styles(layer)[1], None, None, None
    rebuilt, deleted, covered, notes = [], [], [], []
    for plan, target in zip(plans, targets):
        fx, old = plan.layer, plan.layer.name
        ranges = [span for _, name, now, _, _ in target if name for span in now]
        left = [(phrase, why) for phrase, name, _, _, why in target if name is None]
        names = [name for _, name, _, _, _ in target if name]
        if not ranges:
            fx.parent.remove(fx)
            deleted.append(old)
            notes.append(f"{old} deleted: {'; '.join(_left(left, True))}")
            continue
        covered += ranges
        if color is None or shape is None:
            raise Refused("styled-word", f"{old}: the text layer has no pixels to cut")
        if any(icons[i] for a, b in ranges for i in range(a, b)):     # "1#1" once its "1" left: the JSX would cut
            fx.parent.remove(fx)                # every badge; it goes
            deleted.append(old)
            covered = covered[:len(covered) - len(ranges)]
            notes.append(f"{old} deleted: its words now fall in a badge or token")
            continue
        if any(icons) and glyphs is None:
            glyphs = _key_ink(model, lambda run: [(run.start, run.end, KEY_BLACK)], layer, fonts, engine)
        mask = _phrase_mask(layer, model, [(a + offset, b + offset) for a, b in ranges], glyphs, fonts, engine)
        cut = _cut(layer, mask, color, shape)
        if cut is None:
            raise Refused("styled-word", f"{old}: its words have no pixels")
        name = old if names == plan.phrases else layer.name + FX_TAG + FX_SEP.join(names)
        if name == old and _kept(plan, ranges, band, kept):
            if not _holds(fx, mask, layer):
                notes.append(f"{old} cut again: it held pixels outside its words")
            else:
                if geometry is None:
                    geometry = text_geometry(layer, fonts)
                    ink = _on_canvas(layer, (psd.height, psd.width))[..., 3] > INK_ALPHA * 255
                _, got, want = _fx_cover(layer, fx, geometry, ink)
                if got >= FX_COVER * want:
                    continue                    # else on its words' old pixels (Marx's Glued): cut again, silently
        engine[6](fx, cut[0], cut[1])
        fx._invalidate_bbox()
        if name != old:
            fx.name = name
            whys = _left(left) + [why for _, still, _, _, why in target if still and why]
            notes.append(f"{old} renamed {name!r}: {'; '.join(whys)}")
        joined = list(dict.fromkeys(text[a:b] for _, still, now, anchored, _ in target if still and plan.own_style
                                    for a, b in now if (a, b) not in anchored))
        if joined:                              # the typed word takes this layer's style, not the chart's (q6)
            stays = any(anchored for _, still, _, anchored, _ in target if still)     # else its old words left
            notes.append(f"{name} {'also covers' if stays else 'moved to'} the typed {', '.join(map(repr, joined))}, "
                         f"in its own style")
        rebuilt.append((fx, old))
    alive, restyled = {id(fx) for fx in fx_layers(psd, layer)}, []
    for fx in [plan.layer for plan in plans if id(plan.layer) in alive]:
        status = chart_style(layer, fx, text, words)
        blocks = restyle_fx(fx, status) if status else None
        if blocks:
            restyled.append((fx, blocks))
    return rebuilt, deleted, covered, notes, restyled


def new_fx(text: str, styled: list, covered: list, flag) -> list:
    """
    The FX layers an ability edit adds (idea 14): one per styled status word no FX layer covers (a range of a rebuilt
    layer holds the whole word: its status's layer on it, or a "styled" phrase around it, Clot Gold's "1 Become Back
    Money"), named as Effect Layers.jsx's Add names it: the status alone when that covers just those words, else one
    "status#n" per word (the nth of its phrase_ranges). A word a rebuilt layer covers only in part, or no such name
    reaches (a digit glued to it, "2Blazing", beside a whole-word one), gets none (told): it prints in the chart's
    fill alone.

    Args:
        text: The text set
        styled: status_spans() over it, the styled ones
        covered: The str ranges the rebuilt FX layers cover (_rebuild_fx)
        flag: set_text_layer's flag(code, why)

    Returns:
        [(phrase, [(start, end)], StatusColor)] in text order
    """
    left = {}
    for start, end, status in styled:
        if any(a <= start and end <= b for a, b in covered):
            continue
        part = next(((a, b) for a, b in covered if a < end and start < b), None)
        if part:
            flag("styled-word", f"{text[start:end]!r} is partly under the FX layer of {text[part[0]:part[1]]!r}: no FX "
                                "layer of its own")
            continue
        left.setdefault(status.name, (status, []))[1].append((start, end))
    out = []
    for name, (status, spans) in left.items():
        every = phrase_ranges(text, [name])
        lost = [text[max(a - 1, 0):b + 1].strip() for a, b in spans if (a, b) not in every]
        if lost:
            flag("styled-word", f"{', '.join(map(repr, lost))}: Effect Layers.jsx cannot name an FX layer for a {name} a "
                                "digit touches beside a whole-word one: no FX layer of its own")
            spans = [span for span in spans if span in every]
            if not spans:
                continue
        if spans == every:
            out.append((name, spans, status))
        else:
            out += [(f"{name}#{every.index(span) + 1}", [span], status) for span in spans]
    return out


def _regenerate_fx(psd: PSDImage, layer, text: str, fresh: list, fonts, engine: tuple) -> list:
    """
    The FX layers new_fx() names, cut from the rasterized layer, directly above its top-level group (the Text group; the
    layer itself at the root) and the layers clipped to it, in text order, where Effect Layers.jsx's Add puts them (the
    group's clip chain kept on the group: Claw's MagentaSmorg and Exposure 1; review 2026-10-24), under the chart's style.

    Args:
        psd: The card
        layer: The ability layer, text set and rasterized
        text: The text set (the engine's text adds the final \\r and may keep a byte-order mark)
        fresh: new_fx()
        fonts: The FontMap used
        engine: _fx_engine()

    Returns:
        The new layers, bottom to top
    """
    group = _top(layer)
    above = len(_clip_chain(group))
    model = engine[0].from_layer(layer)
    offset = model.text.find(text)
    if offset < 0:
        raise Refused("styled-word", "the engine's text differs from the layer's")
    color, shape = layer.numpy("color"), layer.numpy("shape")
    if color is None or shape is None:
        raise Refused("styled-word", "the text layer has no pixels to cut")
    icons = _char_styles(layer)[1]
    glyphs = _key_ink(model, lambda run: [(run.start, run.end, KEY_BLACK)], layer, fonts, engine) if any(icons) else None
    made = []
    for phrase, spans, status in fresh:
        name = fx_name(layer.name, phrase)
        cut = _cut(layer, _phrase_mask(layer, model, [(a + offset, b + offset) for a, b in spans], glyphs, fonts, engine),
                   color, shape)
        if cut is None:
            raise Refused("styled-word", f"{name}: the word has no pixels")
        fx = _cut_fx(psd, name, cut, status.blocks, engine)
        fx.move_to(psd, index=psd.index(group) + 1 + above + len(made))
        made.append(fx)
    return made


# ========================
# Apply (section 5 step 4)
# ========================
def _digit(layer) -> Optional[str]:
    """A bar layer's digit, "0" to "10" (its name without a variant tag), else None."""
    name = base_name(layer).strip()
    return name if name.isdigit() and int(name) <= STAT_MAX and str(int(name)) == name else None


def _row(group, stat: str) -> TemplateRow:
    """
    One row group as a TemplateRow; else Refused("unsupported-card").
    The card maker's own rows count too (layout support): other layers beside the digits (left alone: Anubisath
    Guardian's empty text, Dark Mind's "5 copy", Bulborb Larva's gauge), digits 1-10 or a missing one (set_stat refuses
    showing it), or each digit a picture of the whole row (Bulborb Larva, Empress Bulblax).
    """
    layers, extras = {}, []
    for child in group:
        digit = None if child.is_group() else _digit(child)
        if digit is None or digit in layers:
            if child.is_group() or digit is not None:
                raise Refused("unsupported-card", f"{stat}: non-template row {group.name!r}")
            extras.append(child)
        else:
            layers[digit] = child
    masked = all(layer.kind not in DRAWN_LAYER_KINDS and layer.has_mask() for layer in layers.values())
    boxes = [tuple(layer.bbox) for layer in layers.values()]
    whole = union_box(boxes)
    pictures = (whole is not None and all(layer.kind in DRAWN_LAYER_KINDS and not layer.is_group() for layer in layers.values())
                and all(box[2] - box[0] >= (whole[2] - whole[0]) / 2 for box in boxes))
    if len(layers) < 5 or not (masked or pictures):
        raise Refused("unsupported-card", f"{stat}: non-template row {group.name!r}")
    return TemplateRow(group, layers)


def creature_of(rows: list) -> dict:
    """
    id(row group) -> its creature as the parser counts them (StatTracker.by_creature): a row named "1", "2"... (Pix),
    else the n-th of a stat block repeated in other groups (Tainted Lazarus's Laz and Dead Laz), else 0 (shared).

    Args:
        rows: stat_rows()'s rows of one stat
    """
    outers: dict = {}
    for group, shown, _ in rows:
        block = group.parent
        outer, key = getattr(block, "parent", None), (getattr(block, "name", "") or "").lower()
        if shown and not any(outer is other for other in outers.setdefault(key, [])):
            outers[key].append(outer)
    repeated = any(len(found) > 1 for found in outers.values())
    out = {}
    for group, _, _ in rows:
        block = group.parent
        outer, key = getattr(block, "parent", None), (getattr(block, "name", "") or "").lower()
        index = next((k for k, other in enumerate(outers.get(key, []), 1) if other is outer), 0)
        out[id(group)] = int(base_name(group)) if base_name(group).strip().isdigit() else index if repeated else 0
    return out


def bar_rows(psd: PSDImage, stat: str, creature: int = 1) -> tuple:
    """
    The rows a stat edit changes: stat_rows(), but on a stat drawn by several rows the card's own (R1, R3; owner
    2026-10-26): the creature's rows on a card of several creatures (_creature_rows; the card's stats are creature 1's),
    else the shared ones; then a row showing no digit left out when another shows one (the parser skips it: Ze'mer's
    hidden template row under its flowers, Pix's empty HP rows under the one shared mask).

    Returns:
        (rows, picked): picked when the stat has several rows and these were chosen from them
    """
    rows = stat_rows(psd)[stat]
    if len(rows) < 2:
        return rows, False
    index = creature_of(rows)
    if len({index[id(group)] for group, _, _ in rows} - {0}) > 1:      # several creatures: the creature's, else shared
        rows = _creature_rows(rows, index, creature)
    live = [row for row in rows if row[1]]
    return (live if live else rows), True


def _creature_rows(rows: list, index: dict, creature: int) -> list:
    """
    The rows of one stat a creature's value is read off, as the parser reads them (StatTrackers.creatures): its own when
    one shows a digit, else the shared ones; another creature's empty rows read creature 1's when only those show and
    they reach over its rows (Pix's tall HP mask), else its own (none shown: a full bar). [] for none.
    """
    of = lambda n: [row for row in rows if index[id(row[0])] == n]
    shows = lambda found: any(shown for _, shown, _ in found)
    mine, shared = of(creature), of(0)
    if shows(mine):
        return mine
    if creature == 1:
        return shared or mine
    if shows(shared):
        return shared
    ones = of(1)
    spans = lambda group, shown: [span for layer in group if not layer.is_group() and base_name(layer).strip().isdigit()
                                  and (layer.is_visible() or not shown) for span in [digit_span(layer)] if span]
    band = lambda found: (min(a for a, _ in found), max(b for _, b in found)) if found else None
    if shows(ones) and not any(shown for group, shown, _ in rows if index[id(group)] > 1) and reaches_over(
            [span for group, _, _ in ones for span in spans(group, True)], [band(spans(group, False)) for group, _, _ in mine]):
        return ones
    return mine


def creature_stats(parsed: dict, creature: int = 1) -> dict:
    """The stats a creature's edit starts from: the card's (creature 1's), else its entry in the parse's creatures."""
    return parsed["card"] if creature == 1 else parsed["card"]["creatures"][creature - 1]


def creature_bar(psd: PSDImage, stat: str, creature: int) -> Optional[tuple]:
    """What a creature's stat edit sets, alike for the creatures sharing it: its layer-comp bar (comp_row), else its rows
    (bar_rows); None for no one bar."""
    if comp_bars(psd)[stat]:
        bars = comp_row(psd, stat, creature)
        return ("comp", bars[0][0].layer_id) if len(bars) == 1 else None
    return tuple(sorted(id(group) for group, _, _ in bar_rows(psd, stat, creature)[0])) or None


def several_rows(psd: PSDImage) -> bool:
    """A card the parser reads as several creatures off its digit rows (_still_creatures): Pix's rows "1" to "3", Tainted
    Lazarus's blocks; not Sonic's two SPD rows."""
    return _still_creatures(psd, None)


ART_FILL = 0.8              # bar art filling less of its box is separate icons (template 0.977, Croissant's 0.86)


def art_fill(psd: PSDImage, box: tuple) -> Optional[float]:
    """The share of a bar's box the visible art under its Bars group fills; None without such art."""
    groups = [g for g in psd.descendants() if g.is_group() and g.name.strip() == "Bars"] \
        or [g for g in psd.descendants() if g.is_group() and "bar" in g.name.lower()]
    x0, y0, x1, y1 = box
    mask = np.zeros((max(1, y1 - y0), max(1, x1 - x0)), bool)
    for layer in groups[0].descendants() if groups else ():
        if layer.is_group() or not layer.is_visible() or layer.bbox[3] <= y0 or layer.bbox[1] >= y1 \
                or layer.bbox[2] <= x0 or layer.bbox[0] >= x1:
            continue
        image = layer.topil() if layer.kind in ("pixel", "smartobject") else None
        if image is None:
            return 1.0                          # a shape or a fill: whole
        alpha = np.asarray(image.convert("RGBA"))[..., 3] > 128
        lx, ly = layer.bbox[0], layer.bbox[1]
        ax0, ay0, ax1, ay1 = max(x0, lx), max(y0, ly), min(x1, lx + alpha.shape[1]), min(y1, ly + alpha.shape[0])
        if ax1 > ax0 and ay1 > ay0:
            mask[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] |= alpha[ay0 - ly:ay1 - ly, ax0 - lx:ax1 - lx]
    return float(mask.mean()) if mask.any() else None


def stat_caps(psd: PSDImage) -> Optional[dict]:
    """
    {stat: [the most each creature's display shows]} on a card whose stats are not on the template's layer-comp bars
    (separate icons, pictures, own shapes, digit rows, comp bars of their own such as the Isaac bars: comp_custom): 10,
    less where a counted row has room for fewer (owner 2026-11-08: more is refused when typed); None on the template's
    bars, an item's rows or no bars.
    """
    comps = comp_bars(psd)
    if any(comps.values()):
        fills = (art_fill(psd, tuple(box)) for bars in comps.values() for _, _, box in bars)
        if not any(comp_custom(layer) for bars in comps.values() for layer, _, _ in bars) \
                and not any(fill is not None and fill < ART_FILL for fill in fills):
            return None
        count = max((comp_creature(bar[0]) for bars in comps.values() for bar in bars), default=0) or 1
        return {stat: [STAT_MAX] * count for stat in STAT_FIELDS}
    rows = stat_rows(psd)
    if not all(rows.values()):
        return None
    count = max((n for found in rows.values() for n in creature_of(found).values()), default=1) if several_rows(psd) else 1
    count = max(count, 1)
    out = {}
    for stat in STAT_FIELDS:
        out[stat] = []
        for n in range(1, count + 1):
            found = bar_rows(psd, stat, n)[0]
            room = count_room(psd, found[0][0]) if len(found) == 1 else None
            out[stat].append(STAT_MAX if room is None else min(STAT_MAX, room))
    return out


def cap_why(stat: str, value: int, cap: int, creature: int = 0) -> str:
    """A stat past stat_caps' cap, in words (the bake's refusal; the website says the same)."""
    whose = f"creature {creature}'s" if creature else "its"
    if cap >= STAT_MAX:
        return f"{stat} {value}: {whose} bars show up to {cap}"
    return f"{stat} {value}: {whose} {stat.upper()} row has room for {cap} pictures"


def template_rows(psd: PSDImage, stat: str, creature: int = 1) -> list:
    """
    The template bar row of a stat: one "<Stat> Dark" group holding the ten masked exposure digits 0-9. A stat of
    several rows takes bar_rows()' one row when it is a template row (Pix's DEF "1").

    Args:
        psd: An open card
        stat: hp, def, atk or spd
        creature: Whose rows on a card of several creatures (bar_rows)

    Returns:
        [row]; raises Refused("unsupported-card") for any other row shape
    """
    rows = stat_rows(psd)[stat]
    named = lambda layer: fullmatch(rf"\s*{stat}\s*dark\s*", layer.name, IGNORECASE) is not None
    if len(rows) == 1:
        group = rows[0][0]
        if not named(group):
            raise Refused("unsupported-card", f"{stat}: non-template row {group.name!r}")
        return [_row(group, stat)]
    picked, _ = bar_rows(psd, stat, creature)
    if len(picked) == 1:
        return [_row(picked[0][0], stat)]
    raise Refused("unsupported-card", f"{stat}: {len(rows)} bar rows")


def _still_creatures(psd: PSDImage, emptied) -> bool:
    """Whether the parser still reads several creatures once the row `emptied` shows no digit (StatTracker.creatures: at
    least two stats with two creatures' rows showing one; Twin Emperors' ATK at 10 would leave one)."""
    many, found = 0, stat_rows(psd)
    for key in STAT_FIELDS:
        rows = found[key]
        index = creature_of(rows)
        filled = {index[id(group)] for group, shown, _ in rows if shown and group is not emptied} - {0}
        many += len(filled) >= 2
    return many >= 2


def stat_value(rows: list) -> int:
    """What the parser reads for the rows (psd_parser.StatTracker.total): rows showing a digit add up; none shown = 10."""
    shown = [value for value in (row.reads() for row in rows) if value is not None]
    return sum(shown) if shown else STAT_MAX


def _targets(rows: list, value: int) -> list:
    """The digit the row shows for a value 0-10 (None: no digit; a row with its own "10" digit shows it: Bulborb
    Larva)."""
    return [value if value < STAT_MAX or str(STAT_MAX) in rows[0].digits else None]


def _marker(row: TemplateRow) -> Optional[object]:
    """The row's visible "10" that draws nothing (Twin Emperors' opacity-0 exposures), kept on by every toggle."""
    layer = row.digits.get(str(STAT_MAX))
    return layer if layer is not None and layer.visible and layer.opacity == 0 else None


def _row_shows(row: TemplateRow, target: Optional[int]) -> list:
    """The digits a row shows once toggled to target (_toggle_row): target and a kept marker."""
    return sorted({*([target] if target is not None else []), *([STAT_MAX] if _marker(row) else [])})


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


def _toggle_row(row: TemplateRow, target: Optional[int]) -> tuple:
    """
    Show one digit of a row (None: none), hide the others but a marker (_marker), the row's effect following the
    visible digit (idea 12).

    Args:
        row: The row
        target: The digit to show

    Returns:
        (layers whose record changes, [(from layer, to layer or None, block key)] effect blocks moved or cleared,
        pixels the effects of the layers involved can reach beyond their masks)
    """
    show = row.digits[str(target)] if target is not None else None
    hide = [layer for layer in row.digits.values() if layer.visible and layer is not show and layer is not _marker(row)]
    source = hide if row.visible else [layer for layer in row.digits.values() if not layer.visible]
    styled = [layer for layer in source if _enabled_effects(layer)]
    reach = max(fx_pad(layer) for layer in (row.group, *styled, *hide, *([show] if show is not None else [])))
    moves = []
    if show is not None and styled:
        if _enabled_effects(show):              # the new digit keeps its own; the hidden one's goes
            moves += [(layer, None, key) for layer in styled if layer in hide
                      for key in EFFECT_BLOCKS if key in layer.tagged_blocks]
        else:                                           # replace, never merge: a disabled block on the new digit goes first
            moves += [(show, None, key) for key in EFFECT_BLOCKS if key in show.tagged_blocks]
            moves += [(styled[0], show, key) for key in EFFECT_BLOCKS if key in styled[0].tagged_blocks]
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


def _comp_states(layer, order: list) -> dict:
    """
    {comp ID: {setting key: value}} of one embedded layer: its layer comp settings (cmls) list each comp in order with
    only what changed since the comp before (a comp's missing keys carry over, blendOptions' too), "enab" its eye,
    "Ofst" its place, imageMask/vectorMask their offsets; an ID no entry lists (lastAppliedComp unset: -1) gets the
    trailing entry's, the document's own.
    """
    settings = next((item.data for item in layer.tagged_blocks.get_data(Tag.METADATA_SETTING) or []
                     if item.key == b"cmls"), None)
    state, out = {}, {}
    for setting in (settings or {}).get(b"layerSettings") or []:
        change = {key: value for key, value in setting.items() if key != b"compList"}
        if b"blendOptions" in change:           # lists only the options changed (Puzzled Cat's Speed)
            change[b"blendOptions"] = {**(state.get(b"blendOptions") or {}), **change[b"blendOptions"]}
        state = {**state, **change}
        for cid in setting.get(b"compList") or []:
            out[int(cid)] = state
    return {cid: out.get(cid, state) for cid in order}


def _offset(state: dict, key: Optional[bytes] = None) -> tuple:
    """A comp state's offset: the layer's own place (key None), else its imageMask's or vectorMask's."""
    off = (state if key is None else state.get(key) or {}).get(b"Ofst") or {}
    return int(off.get(b"Hrzn", 0)), int(off.get(b"Vrtc", 0))


def _percent(value) -> float:
    """A comp state's opacity (blendOptions Opct); unlisted is 100: cmls list only what is off the defaults."""
    return 100.0 if value is None else float(getattr(value, "value", value))


def hard_edged(rgba: np.ndarray) -> bool:
    """Pixel art or a flat shape (uint8 RGBA): no in-between alpha, few colours, blocky at its size. It resizes nearest."""
    alpha = rgba[..., 3]
    solid = alpha >= 128
    if not solid.any() or ((alpha > 8) & (alpha < 247)).sum() > HARD_SOFT * (alpha > 0).sum():
        return False
    if len(np.unique(rgba[solid][:, :3].astype(np.uint32) @ np.array([65536, 256, 1], np.uint32))) > HARD_COLOURS:
        return False
    pairs = solid[:, 1:] & solid[:, :-1]
    same = (rgba[:, 1:, :3] == rgba[:, :-1, :3]).all(axis=2) & pairs
    return max(alpha.shape) <= HARD_SMALL or same.sum() >= HARD_BLOCKY * pairs.sum()


def stored_mode(layer) -> Optional[int]:
    """The resize mode a bar's stored pixels show, or None when they tell neither (moved since, or drawn the same)."""
    value, stored = comp_value(layer), layer.numpy()
    if value is None or stored is None:
        return None
    drawn = {mode: comp_render(layer, value, mode, False) for mode in (Image.NEAREST, Image.BICUBIC)}
    if any(rgba is None for rgba, _ in drawn.values()):
        return None
    box = tuple(layer.bbox)
    area = union_box([box, *(where for _, where in drawn.values())])

    def plane(alpha, where):
        out = np.zeros((area[3] - area[1], area[2] - area[0]), np.float32)
        out[where[1] - area[1]:where[3] - area[1], where[0] - area[0]:where[2] - area[0]] = alpha
        return out
    own = plane(stored[..., 3], box)
    near, cubic = (float(np.abs(plane(rgba[..., 3], where) - own).sum()) for rgba, where in drawn.values())
    return Image.NEAREST if near * HARD_CLEAR < cubic else Image.BICUBIC if cubic * HARD_CLEAR < near else None


def image_mode(layer, image: np.ndarray, drawn: Optional[np.ndarray] = None) -> int:
    """
    The resampling a placed picture (any image smart object; a bar's is stored_mode) is drawn again with, per object
    as comp_render's: the mode its stored pixels show, when one mode redraws `drawn` (the picture they were drawn from:
    by default the embedded one) on the layer's quad HARD_CLEAR times closer in alpha than every other; else nearest
    for pixel art or a flat shape (hard_edged(image)); else bicubic (Photoshop's default; 227 of 740 pictures show no
    mode, ../resize/evid.json).
    """
    stored = layer.numpy()
    if drawn is None and layer.smart_object.kind == "data":
        data = layer.smart_object.data
        drawn = decode(data, file_type(data, layer.smart_object.filename))
    corners = corners_of(layer)
    errors = {}
    if drawn is not None and stored is not None and stored.shape[2] >= 4 and len(corners) == 8:
        box = tuple(layer.bbox)
        for mode in (Image.NEAREST, Image.BILINEAR, Image.BICUBIC):
            rgba, where = draw(drawn, corners, mode)
            if rgba is None:
                errors = {}
                break
            area = union_box([box, where])
            plane = np.zeros((area[3] - area[1], area[2] - area[0]), np.float32)
            plane[where[1] - area[1]:where[3] - area[1], where[0] - area[0]:where[2] - area[0]] = rgba[..., 3]
            plane[box[1] - area[1]:box[3] - area[1], box[0] - area[0]:box[2] - area[0]] -= stored[..., 3]
            errors[mode] = float(np.abs(plane).sum())
    if errors:
        best = min(errors, key=errors.get)
        if all(errors[best] * HARD_CLEAR < error for mode, error in errors.items() if mode != best):
            return best
    return Image.NEAREST if hard_edged(image) else Image.BICUBIC


def comp_render(layer, value: int, mode: Optional[int] = None, check: bool = True) -> tuple:
    """
    A layer-comp stat bar drawn at comp `value` as Photoshop draws it (equal to its stored renders at
    every comp, 0-20, 2026-11-06): each embedded layer that comp shows, bottom to top in normal blending, a shape as its
    vector mask in its fill colour, a pixel layer as its pixels, each through its pixel mask moved to that comp's offset;
    placed on its transform's rectangle. Resized: by mode, else as its stored pixels show (stored_mode), else nearest if
    hard_edged, else bicubic. The card stores no mode; Photoshop's redraw on a copy equals nearest (2026-11-07).
    Any other embedded document (groups, effects, knockout: the Isaac bars) is drawn by comp_document; it, and a bar
    with no comp applied last (Pandanoko), is refused unless it draws the comp the bar shows as Photoshop stored it
    (check; COMP_SEEN).

    Returns:
        (straight RGBA float32, canvas bbox), or (None, (0, 0, 0, 0)) when nothing shows; Refused("unsupported-card")
        for any other embedded document or placement
    """
    from psd_tools.composite.vector import draw_vector_mask
    def refuse(why: str):
        raise Refused("unsupported-card", f"{layer.name!r}: {why}: set it in Photoshop")
    placed = placed_data(layer)
    warp = (placed.get(b"warp") or {}).get(b"warpStyle")
    if warp is not None and getattr(warp, "enum", b"warpNone") != b"warpNone":
        refuse("a warped layer-comp bar")
    if layer.smart_object.has_enabled_filters:
        refuse("a smart-filtered layer-comp bar")
    doc = PSDImage.open(BytesIO(layer.smart_object.data))
    comps = doc.image_resources.get_data(Resource.LAYER_COMPS) or {}
    names = {int(c[b"compID"]): str(getattr(c[b"Nm  "], "value", c[b"Nm  "])).rstrip("\x00").strip()
             for c in comps.get(b"list") or []}
    want = next((cid for cid, name in names.items() if name == str(value)), None)
    last = int(comps.get(b"lastAppliedComp", -1))
    if want is None:
        refuse(f"no comp {value} to draw")
    corners = [float(v) for v in placed.get(b"Trnf") or []]
    if len(corners) != 8 or not (corners[0] == corners[6] and corners[2] == corners[4] and corners[1] == corners[3]
                                 and corners[5] == corners[7] and corners[2] > corners[0] and corners[5] > corners[1]):
        refuse("a turned or skewed layer-comp bar")
    x0, y0, x1, y1 = corners[0], corners[1], corners[2], corners[5]
    width, height = doc.width, doc.height
    rgb, alpha = np.zeros((height, width, 3), np.float32), np.zeros((height, width), np.float32)
    flat = all(_flat_layer(inner) for inner in doc if _comp_states(inner, [want])[want].get(b"enab", True))
    if check and not (flat and last in names) and _comp_off(layer, mode) > COMP_SEEN:
        refuse("its bar draws otherwise than Photoshop stored it")
    if not flat:
        rgb, alpha = comp_document(layer, doc, comps, want, last)
    for inner in doc if flat else ():
        states = _comp_states(inner, [want, last])
        now, then = states[want], states[last]
        if not now.get(b"enab", True):
            continue
        knockout = inner.tagged_blocks is not None and inner.tagged_blocks.get_data(Tag.KNOCKOUT_SETTING, 0)
        if inner.is_group() or inner.kind not in ("shape", "pixel") or inner.blend_mode != BlendMode.NORMAL or inner.has_effects() \
                or knockout:
            refuse(f"its bar holds a {inner.kind} layer {inner.name!r} the bake cannot draw")
        if _offset(now, b"vectorMask") != _offset(then, b"vectorMask"):
            refuse("a comp moves its vector mask")
        if _offset(now) != _offset(then):       # comp_document draws these (no flat bar has one: allbars census)
            refuse(f"a comp moves {inner.name!r}")
        if any(now.get(key) != then.get(key) for key in COMP_LOOKS):
            refuse(f"a layer comp changes the look of {inner.name!r}")
        if inner.kind == "shape":
            cover = np.asarray(draw_vector_mask(inner, (0, 0, width, height)), np.float32).reshape(height, width)
            cover = cover / 255.0 if cover.max() > 1.0 else cover
            fill = (inner.tagged_blocks.get_data(Tag.SOLID_COLOR_SHEET_SETTING) or {}).get(b"Clr ") or {}
            color = np.array([float(fill.get(key, 0.0)) / 255.0 for key in (b"Rd  ", b"Grn ", b"Bl  ")], np.float32)
            paint = np.broadcast_to(color, (height, width, 3))
        else:
            cover, paint = np.zeros((height, width), np.float32), np.zeros((height, width, 3), np.float32)
            image = inner.topil()
            if image is not None:
                px = np.asarray(image.convert("RGBA"), np.float32) / 255.0
                l, t = inner.bbox[0], inner.bbox[1]
                a0, b0, a1, b1 = max(0, l), max(0, t), min(width, l + px.shape[1]), min(height, t + px.shape[0])
                if a1 > a0 and b1 > b0:
                    cover[b0:b1, a0:a1] = px[b0 - t:b1 - t, a0 - l:a1 - l, 3]
                    paint[b0:b1, a0:a1] = px[b0 - t:b1 - t, a0 - l:a1 - l, :3]
        if inner.has_mask():
            mask = inner.mask
            raster = np.asarray(mask.topil().convert("L"), np.float32) / 255.0
            dx, dy = (a - b for a, b in zip(_offset(now, b"imageMask"), _offset(then, b"imageMask")))
            full = np.full((height, width), mask.background_color / 255.0, np.float32)
            left, top = mask.bbox[0] + dx, mask.bbox[1] + dy
            a0, b0 = max(0, left), max(0, top)
            a1, b1 = min(width, left + raster.shape[1]), min(height, top + raster.shape[0])
            if a1 > a0 and b1 > b0:
                full[b0:b1, a0:a1] = raster[b0 - top:b1 - top, a0 - left:a1 - left]
            cover = cover * full
        a = cover * inner.opacity / 255.0 * inner.fill_opacity / 255.0
        out = a + alpha * (1 - a)
        rgb = np.where(out[..., None] > 0, (paint * a[..., None] + rgb * (alpha * (1 - a))[..., None])
                       / np.maximum(out, 1e-12)[..., None], rgb)
        alpha = out
    left, top = round(x0), round(y0)
    size = (round(x1) - left, round(y1) - top)
    if size != (width, height):                 # a resized bar: premultiplied
        pre = np.concatenate([rgb * alpha[..., None], alpha[..., None]], axis=2)
        if mode is None:
            mode = stored_mode(layer)
        if mode is None:
            mode = Image.NEAREST if hard_edged(np.round(np.dstack([rgb, alpha]) * 255).astype(np.uint8)) else Image.BICUBIC
        pre = np.stack([np.asarray(Image.fromarray(pre[..., k], "F").resize(size, mode)) for k in range(4)], axis=2)
        alpha = np.clip(pre[..., 3], 0.0, 1.0)
        rgb = np.where(alpha[..., None] > 0, np.clip(pre[..., :3], 0.0, None) / np.maximum(alpha, 1e-6)[..., None], 0.0)
        rgb = np.clip(rgb, 0.0, 1.0)
    ys, xs = (np.round(alpha * 255) > 0).nonzero()
    if not len(xs):
        return None, (0, 0, 0, 0)
    c0, c1, r0, r1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    rgba = np.concatenate([rgb[r0:r1, c0:c1], alpha[r0:r1, c0:c1, None]], axis=2).astype(np.float32)
    return rgba, (left + c0, top + r0, left + c1, top + r1)


def _flat_layer(inner) -> bool:
    """A bar document's layer comp_render's own path draws (the template's): a shape or pixel layer, Normal, no
    effects, no knockout."""
    return not inner.is_group() and inner.kind in ("shape", "pixel") and not inner.has_effects() \
        and inner.blend_mode == BlendMode.NORMAL \
        and not (inner.tagged_blocks is not None and inner.tagged_blocks.get_data(Tag.KNOCKOUT_SETTING, 0))


def comp_custom(layer) -> bool:
    """A layer-comp bar of a card's own (groups, effects, knockout: the Isaac bars, drawn by comp_document), not the
    template's."""
    return not all(_flat_layer(inner) for inner in PSDImage.open(BytesIO(layer.smart_object.data)))


COMP_SHOWN, COMP_PLACED, COMP_LOOK = 1, 2, 4      # a layer comp's capturedInfo: visibility, position, appearance
COMP_LOOKS = (b"Lefx", b"blendOptions", b"FXRefPoint", b"layerSpecific")   # cmls keys COMP_LOOK applies
COMP_SEEN = 8               # levels a checked bar at its shown comp may be off the stored pixels (Tainted Lost 4)


def _comp_off(layer, mode: Optional[int]) -> float:
    """The most a bar drawn at the comp it shows is off Photoshop's stored pixels, premultiplied 0-255."""
    value, stored = comp_value(layer), layer.numpy()
    if value is None:
        return float("inf")
    rgba, where = comp_render(layer, value, mode, False)
    box = tuple(layer.bbox)
    area = union_box([box if stored is not None else None, where if rgba is not None else None])
    if area is None:
        return 0.0

    def plane(pixels, at):
        out = np.zeros((area[3] - area[1], area[2] - area[0], 4), np.float32)
        if pixels is not None:
            pixels = np.asarray(pixels, np.float32)
            alpha = pixels[..., 3:4] if pixels.shape[2] > 3 else np.ones_like(pixels[..., :1])
            out[at[1] - area[1]:at[3] - area[1], at[0] - area[0]:at[2] - area[0]] = np.concatenate(
                [pixels[..., :3] * alpha, alpha], axis=2)
        return out
    return float(np.abs(plane(rgba, where) - plane(stored, box)).max()) * 255


def comp_document(layer, doc: PSDImage, comps: dict, want: int, last: int) -> tuple:
    """
    comp_render's second path: the comp applied to every layer of the bar's embedded document at any depth, as
    Photoshop applies it (only what its capturedInfo holds), and the document composited by psd-tools: its pixel and
    smart object layers and pixel masks moved by the comp's offset less the saved state's (the last applied comp's,
    else the cmls' trailing entry's; a group's own offset is left to its layers, each lists it too), its opacities set,
    its eyes set when it captures visibility. Equal to Photoshop's renders at every comp of the Isaac bars (knockout
    groups, Transparency Shapes Layer off, nested smart objects; custombars 2026-11-11), Ze'mer's moving flowers and
    Puzzled Cat's Speed opacity (allbars 2026-11-12).

    Returns:
        (straight RGB, alpha) float32 at the document's size; Refused for another colour profile or mode, a layer
        the installed fork cannot draw (undrawable: knockout before cards-20261112), a comp that moves a vector mask,
        a masked layer or another kind of layer, changes another look, or captures what is not read
    """
    def refuse(why: str):
        raise Refused("unsupported-card", f"{layer.name!r}: {why}: set it in Photoshop")
    if doc.color_mode != ColorMode.RGB or doc.image_resources.get_data(Resource.ICC_PROFILE) != \
            layer._psd.image_resources.get_data(Resource.ICC_PROFILE):
        refuse("its bar is in another colour profile")
    undrawn = undrawable(doc)
    if undrawn:
        refuse(f"the bot cannot draw {undrawn[0]!r}'s knockout or Transparency Shapes Layer as Photoshop does")
    captured = next(int(c.get(b"capturedInfo", 0)) for c in comps.get(b"list") or [] if int(c[b"compID"]) == want)
    if captured & ~(COMP_SHOWN | COMP_PLACED | COMP_LOOK):
        refuse("a layer comp holding more than eyes, places and looks")
    for inner in doc.descendants():
        states = _comp_states(inner, [want, last])
        now, then = states[want], states[last]
        if captured & COMP_LOOK:
            blend, was = (dict(state.get(b"blendOptions") or {}) for state in (now, then))
            opacity, before = (_percent(options.pop(b"Opct", None)) for options in (blend, was))
            if blend != was or any(now.get(key) != then.get(key) for key in COMP_LOOKS if key != b"blendOptions"):
                refuse(f"a layer comp changes the look of {inner.name!r}")
            if opacity != before:
                inner.opacity = round(opacity * 2.55)
        if captured & COMP_PLACED:
            if _offset(now, b"vectorMask") != _offset(then, b"vectorMask"):
                refuse("a comp moves its vector mask")
            dx, dy = (a - b for a, b in zip(_offset(now), _offset(then)))
            if (dx or dy) and not inner.is_group():         # a group's own Ofst: each of its layers lists it too
                if inner.kind not in ("pixel", "smartobject") or inner.has_mask():
                    refuse(f"a comp moves {inner.name!r}")
                inner.offset = (inner.left + dx, inner.top + dy)
            dx, dy = (a - b for a, b in zip(_offset(now, b"imageMask"), _offset(then, b"imageMask")))
            if (dx or dy) and inner.has_mask():
                try:
                    inner.mask.offset = (inner.mask.left + dx, inner.mask.top + dy)
                except ValueError:
                    refuse(f"a comp moves the mask of {inner.name!r}, which has a vector mask")
        if captured & COMP_SHOWN and b"enab" in now:
            inner.visible = bool(now[b"enab"])
    image = doc.composite(ignore_preview=True, apply_icc=False, force=True)
    if image is None:
        return np.zeros((doc.height, doc.width, 3), np.float32), np.zeros((doc.height, doc.width), np.float32)
    rgba = np.asarray(image.convert("RGBA"), np.float32) / 255.0
    return np.ascontiguousarray(rgba[..., :3]), np.ascontiguousarray(rgba[..., 3])


def set_comp(psd: PSDImage, parsed: dict, stat: str, value: int, creature: int = 1) -> Edit:
    """
    A stat on a creature's layer-comp bar (comp_row): its chosen comp (SoLd "comp") set to the comp named str(value) and
    its pixels drawn again (comp_render), as Photoshop's Properties panel does (it also gives the instance a new "placed"
    ID); nothing else changes (gate_bytes, _comp_record).

    Returns:
        Edit over the bar (Edit.comps); raises Refused for no one bar, a hidden or comp-less one, a value it has no
        comp for (stat>10) and a bar comp_render cannot draw
    """
    bars = comp_row(psd, stat, creature)
    if len(bars) != 1:
        raise Refused("unsupported-card", f"{stat}: {len(bars)} layer-comp bars")
    layer, old, box = bars[0]
    if old is None or not layer.is_visible():
        raise Refused("unsupported-card", f"{stat}: {layer.name!r} shows no layer comp: set it in Photoshop")
    read = creature_stats(parsed, creature)
    if read.get(stat) != old:
        raise Refused("unsupported-card", f"{stat}: the parser reads {read.get(stat)}, the bar shows {old}")
    if value < 0:
        raise ValueError(f"{stat} {value}: not a stat")
    placed = placed_data(layer)
    ids = {int(name): cid for cid, name in comp_names(placed).items()}
    if value not in ids:
        raise Refused("stat>10", f"{stat} {value}: {layer.name!r} has comps {min(ids)}-{max(ids)}")
    rgba, where = comp_render(layer, value)
    reach = fx_pad(layer)
    region = _pad(union_box([box, tuple(layer.bbox) if layer.bbox[2] > layer.bbox[0] else None]), reach, psd)
    placed[b"comp"] = Integer(ids[value])
    placed[b"placed"] = String(f"{_comp_placed(layer, placed, stat, value)}\x00")
    from psd_tools.api.layers import TypeLayer     # the fork's raster writer: channels and rect, the record kept
    TypeLayer._store_raster(layer, rgba if rgba is not None else np.zeros((0, 0, 4), np.float32), where)
    layer._invalidate_bbox()
    return Edit(stat, old, value, [layer], [_record_index(layer)], region, comps=[_record_index(layer)])


def _comp_placed(layer, placed, stat: str, value: int) -> str:
    """set_comp's new placed ID: from the old one, the stat and value, not at random, so every picture's bake of a
    tagged PSD makes the same layers (bake_pictures' check; Stubborn King); one the card holds is skipped."""
    old = str(placed[b"placed"].value).rstrip("\x00") if b"placed" in placed else ""
    taken = {str(data[b"placed"].value).rstrip("\x00") for other in layer._psd.descendants() for data in _placed(other)}
    n = 0
    while (made := str(uuid5(NAMESPACE_URL, f"{old}|{stat}|{value}|{n}"))) in taken:
        n += 1
    return made


def set_creatures(psd: PSDImage, parsed: dict, changes: list) -> list:
    """
    A `creatures` job (_classify_creatures): each creature's changed stats on its bar (creature_bar, found before any
    is set), a shared bar set once: layer-comp bars by set_comp, digit rows by set_stat on the creature's rows.

    Args:
        psd: The open card (edited in place)
        parsed: parse_card() of the card before the edit
        changes: [{stat: value} per creature]

    Returns:
        [Edit]: one over the layer-comp bars, one per digit-row bar; each old as changes with the printed values, field
        "creatures"; raises what set_comp and set_stat do (another creature's naming it)
    """
    keys = {(n, stat): creature_bar(psd, stat, n) for n, stats in enumerate(changes, 1) for stat in stats}
    done, comps, rows = set(), [], []
    for n, stats in enumerate(changes, 1):
        for stat, value in stats.items():
            key = keys[(n, stat)]
            if key is not None and key in done:
                continue
            done.add(key)
            if comp_bars(psd)[stat]:
                comps.append(set_comp(psd, parsed, stat, value, n))
                continue
            try:
                rows.append(set_stat(psd, parsed, stat, value, n))
            except Refused as refused:
                if n == 1:
                    raise
                raise Refused(refused.code, f"creature {n}'s {refused.why}", refused.field) from refused
    old = [{stat: one.get(stat) for stat in stats} for one, stats in zip(parsed["card"]["creatures"], changes)]
    records = [index for edit in comps for index in edit.comps]
    out = [Edit("creatures", old, changes, [layer for edit in comps for layer in edit.layers], records,
                union_box([edit.region for edit in comps]), comps=records)] if comps else []
    for edit in rows:
        edit.field, edit.old, edit.new = "creatures", old, changes
    return out + rows


def set_stat(psd: PSDImage, parsed: dict, stat: str, value: int, creature: int = 1) -> Edit:
    """
    Toggle a stat: the digit str(value) visible in "<Stat> Dark", the other nine hidden, 10 = none.

    The row's effect follows the visible digit (idea 12): when digit O goes hidden and N shows, O's effect blocks
    (lfx2 and its lmfx/lfxs/lrFX kin) move onto N's record, replacing any block of N's with nothing enabled; when N
    carries an enabled effect of its own it keeps it and O's is cleared. Nothing moves to a 10 (no digit shows: the
    effect stays on the hidden O), and a toggle from 10 takes the effect of the one hidden digit carrying one. A
    hidden styled digit beside a visible unstyled one (Mettaton_EX's Atk 6) is left alone: showing it later shows it
    styled. So the path matters: atk 7 -> 4 moves nothing, 7 -> 10 -> 4 gives 4 the hidden 6's effect (from 10 the
    row cannot tell a stale style from its own).

    Args:
        psd: The open card (edited in place)
        parsed: parse_card() of the card before the edit
        stat: hp, def, atk or spd
        value: 0-10
        creature: Whose bar on a card of several creatures (set_creatures): its own rows, or the ones it shares

    Returns:
        Edit over the toggled digits; raises Refused for other row shapes and a missing digit (unsupported-card) and
        values the row cannot show (stat>10)
    """
    if comp_bars(psd)[stat]:                    # one smart object, its layer comp the value
        return set_comp(psd, parsed, stat, value, creature)
    try:
        rows = template_rows(psd, stat, creature)
    except Refused as refused:                  # the card maker's own bar (layout support)
        return set_bar(psd, parsed, stat, value, refused, creature)
    old, read = stat_value(rows), creature_stats(parsed, creature)
    if read.get(stat) != old:
        raise Refused("unsupported-card", f"{stat}: the parser reads {read.get(stat)}, the row shows {old}")
    if value == STAT_MAX and len(stat_rows(psd)[stat]) > 1 and str(STAT_MAX) not in rows[0].digits \
            and len(parsed["card"].get("creatures") or []) > 1 and not _still_creatures(psd, rows[0].group):
        raise Refused("unsupported-card", f"{stat} 10: {rows[0].group.name!r} would show no digit, and the parser would "
                                          "then read the card as one creature, another row's value: edit it in Photoshop")
    if value < 0:
        raise ValueError(f"{stat} {value}: not a stat")
    if value > STAT_MAX:
        raise Refused("stat>10", f"{stat} {value}: the bar's one row prints up to {STAT_MAX}")
    targets = _targets(rows, value)
    for row, target in zip(rows, targets):
        if target is not None and str(target) not in row.digits:
            raise Refused("unsupported-card", f"{stat} {target}: {row.group.name!r} has no digit {target}")
    changed, moves, reach = [], [], 0
    for row, target in zip(rows, targets):
        row_changed, row_moves, row_reach = _toggle_row(row, target)
        changed += row_changed
        moves += row_moves
        reach = max(reach, row_reach)
    region = union_box([_where(layer) for layer in changed])
    return Edit(stat, old, value, changed, [_record_index(layer) for layer in changed],
                _pad(region, reach, psd) if region else None,
                [(_record_index(a), None if b is None else _record_index(b), bytes(key)) for a, b, key in moves])


# ========================
# Layout support: the card makers' own stat bars (docs/Finished/session-2026-10-25/LAYOUT_SUPPORT.md)
# ========================
BAR_EDGES = (360, 435, 518, 605, 686, 764, 847, 928, 1009, 1086, 1173)   # the old template's digit masks: where cell
#   k starts; its bar's end
EDGE_TOL = 0.4              # a layer's edge sits on the value's cell edge within this share of a cell
CELL_SHARE = 0.25           # a bar layer narrower than this share of its bar draws one cell
COVER_SHARE = 0.8           # a layer under the cells covers a cell across this share of its width


def _where(layer) -> tuple:
    """Where a bar layer acts: an adjustment's mask, else its pixels."""
    if layer.kind not in DRAWN_LAYER_KINDS and layer.has_mask():
        return tuple(layer.mask.bbox)
    return tuple(layer.bbox)


def _snap(layer) -> dict:
    """What a stat edit may change on a layer: name, eye, place, mask place, effect blocks."""
    return {"name": layer.name, "visible": layer.visible, "pos": (layer.left, layer.top),
            "mask": (layer.mask.left, layer.mask.top) if layer.has_mask() else None,
            "blocks": {bytes(key): layer.tagged_blocks[key].tobytes() for key in EFFECT_BLOCKS if key in layer.tagged_blocks}}


def _bar_spec(layer, snap: dict, remask: bool = False) -> Optional[dict]:
    """The change a stat edit made to a layer since _snap() (gate_bytes checks it record by record); None: none.
    remask: its mask was redrawn (_remask)."""
    now = _snap(layer)
    move = ((now["pos"][0] - snap["pos"][0], now["pos"][1] - snap["pos"][1]) if layer.width and layer.height else (0, 0))
    mask = ((now["mask"][0] - snap["mask"][0], now["mask"][1] - snap["mask"][1]) if now["mask"] and snap["mask"]
            and layer.mask.width and layer.mask.height else (0, 0))
    blocks = {key: now["blocks"].get(key) for key in set(snap["blocks"]) | set(now["blocks"])
              if snap["blocks"].get(key) != now["blocks"].get(key)}
    spec = {"flags": now["visible"] != snap["visible"], "name": now["name"] if now["name"] != snap["name"] else None,
            "move": move, "mask": mask, "blocks": blocks, "remask": remask}
    return spec if spec["flags"] or spec["name"] is not None or any(move) or any(mask) or blocks or remask else None


def set_bar(psd: PSDImage, parsed: dict, stat: str, value: int, why: Refused, creature: int = 1) -> Edit:
    """
    A stat on a bar the card maker drew their own way (plan_bar), the edit following their layout.

    Args:
        psd: The open card (edited in place)
        parsed: parse_card() of the card before the edit
        stat: hp, def, atk or spd
        value: 0-10
        why: template_rows()'s refusal, raised again when the bar is none plan_bar knows
        creature: Whose rows (bar_rows)

    Returns:
        Edit, its changes declared per layer (Edit.bar); raises Refused
    """
    rows, picked = bar_rows(psd, stat, creature)
    old = creature_stats(parsed, creature).get(stat)
    if not rows or not isinstance(old, int):
        raise why
    if value < 0:
        raise ValueError(f"{stat} {value}: not a stat")
    if value > STAT_MAX:
        raise Refused("stat>10", f"{stat} {value}: the bar's one row prints up to {STAT_MAX}")
    pair = _cell_groups(psd, stat, rows) if picked or len(rows) == 1 else None
    if len(rows) != 1:                          # a stat's cells split over a lit and a dark row (R2)
        if pair is None or pair[0] is pair[1]:
            raise why
        plan = plan_cells(psd, stat, pair, old, value, parsed["card"])
    else:
        try:
            plan = plan_bar(psd, rows[0][0], stat, old, value, why, values=parsed["card"])
        except Refused as strict:               # only where the strict plan refuses, so a bar that baked before bakes
            try:                                # the same bytes: edges anywhere (R4), then the cells by name or look
                plan = plan_bar(psd, rows[0][0], stat, old, value, why, loose=True, tall=picked, values=parsed["card"])
            except Refused:
                try:
                    if pair is None:
                        raise strict
                    plan = plan_cells(psd, stat, pair, old, value, parsed["card"])
                except Refused as cells:
                    if pair is not None and cells.why == none_shows(stat, value, pair[0]):   # the clearer reason
                        raise cells
                    plan = plan_count(psd, rows[0][0], stat, old, value)
                    if plan is None:
                        raise strict
    layers = list(dict.fromkeys([*plan.show, *plan.hide, *(layer for layer, _ in plan.names),
                                 *(layer for layer, _ in plan.moves),
                                 *(layer for layer, _ in plan.styles), *(layer for layer, *_ in plan.remasks)]))
    snaps = {id(layer): _snap(layer) for layer in layers}
    before = [_where(layer) for layer in layers]
    for layer in plan.show:
        layer.visible = True
    for layer in plan.hide:
        layer.visible = False
    for layer, name in plan.names:
        layer.name = name
    for layer, dx in plan.moves:
        _shift(layer, dx, 0)
    for layer, source in plan.styles:
        _copy_blocks(source, layer)
    for layer, dx, edge, tol, loose in plan.remasks:
        _remask(layer, dx, edge, tol, loose)
    icons = None
    if plan.copies:                             # a cell's lit picture copied from where the card draws one (R6)
        icons = IconPlan(stat, old, value, how=plan.how)
        for source, group, index, dx, dy, name in plan.copies:
            copy = source.duplicate(group, index=index, name=name)
            same_placed(source, copy, f"{stat}|{name}|{dx},{dy}")
            _shift(copy, dx, dy)
            icons.done.append(("copy", source, copy, dx, dy))
            before.append(_where(copy))
    bases = [base for layer in layers for base in [_clip_base(layer)] if base is not None]
    copies = [done[2] for done in icons.done] if icons is not None else []
    reach = max([fx_pad(layer) for layer in layers + bases + copies] + [0])     # a copy's effects too
    # where a layer acts: its own box, inside what it clips to (an exposure clipped to a bar) and the bar's rows
    within = lambda layer: union_box([plan.box] + [tuple(base.bbox) for base in [_clip_base(layer)] if base is not None]
                                     + ([_where(layer)] if picked else []))     # a mask over several rows (Pix's HP)
    boxes = [clip_to(box, within(layer)) for layer, box in zip(layers + layers, before + [_where(layer) for layer in layers])]
    region = union_box([box for box in boxes if box])
    if icons is not None:
        region = union_box([region] + [_where(done[2]) for done in icons.done] if region else [_where(done[2]) for done in icons.done])
    if region and _off_canvas(region, psd):     # Computer Virus: a bar beside the canvas (its region would be empty)
        raise Refused("unsupported-card", f"{stat}: the bar lies outside the card ({region} on the {psd.width}x"
                                          f"{psd.height} canvas): a change would show nothing on the card")
    remasked = {id(layer) for layer, *_ in plan.remasks}
    specs = [(layer, _bar_spec(layer, snaps[id(layer)], id(layer) in remasked)) for layer in layers]
    touched = [(layer, spec) for layer, spec in specs if spec is not None]
    return Edit(stat, old, value, [layer for layer, _ in touched] + copies, [], _pad(region, reach, psd) if region else None,
                notes=plan.notes, bar=[(_record_index(layer), spec) for layer, spec in touched], icons=icons)


def clip_to(box: tuple, bounds: tuple) -> Optional[tuple]:
    """box inside bounds (x0, y0, x1, y1); None when they do not meet."""
    out = (max(box[0], bounds[0]), max(box[1], bounds[1]), min(box[2], bounds[2]), min(box[3], bounds[3]))
    return out if out[2] > out[0] and out[3] > out[1] else None


def _copy_blocks(source, target) -> None:
    """The source's effect blocks onto the target (replaced in place, added after the target's last, removed when the
    source has none): a cell taking its lit or dark neighbour's style."""
    have = {bytes(key) for key in source.tagged_blocks if key in EFFECT_BLOCKS}
    items = list(target.tagged_blocks.items())
    target.tagged_blocks.clear()
    for key, block in items:
        if key not in EFFECT_BLOCKS:
            target.tagged_blocks[key] = block
        elif bytes(key) in have:
            target.tagged_blocks[key] = deepcopy(source.tagged_blocks[key])
    for key in EFFECT_BLOCKS:
        if key in source.tagged_blocks and key not in target.tagged_blocks:
            target.tagged_blocks[key] = deepcopy(source.tagged_blocks[key])
    target._psd.mark_updated()


@dataclass
class BarPlan:
    """plan_bar(): what a stat edit changes on the card maker's own bar."""
    how: str                                     # "cells shown and hidden", "cell styles", "the value's edge moved"...
    box: tuple                                   # the bar (x0, y0, x1, y1): the edit's region stays in it
    show: list = field(default_factory=list)
    hide: list = field(default_factory=list)
    names: list = field(default_factory=list)    # (layer, new name): the layer named after the value
    moves: list = field(default_factory=list)    # (layer, dx): moved along the bar with its pixels
    styles: list = field(default_factory=list)   # (cell, source cell): the source's effect blocks copied onto the cell
    copies: list = field(default_factory=list)   # (source, group, index, dx, dy, name): a lit picture copied into a cell
    remasks: list = field(default_factory=list)  # (layer, dx, old edge, tol, loose): its mask redrawn, the value edge dx
    #   along (_remask)
    notes: list = field(default_factory=list)


def _bar_layer(psd: PSDImage, group, stat: str):
    """
    The layer drawing the stat's bar: a wide drawn layer in the row (Isaac's hearts), else one named after the stat or
    in a group so named, inside a bar or stat group (Uumuu's "Bars" > "hp" > "hp", Chef Saltbaker's "Bars" > "ATK");
    None when there is none.
    """
    pattern = STAT_PATTERNS[stat]
    wide = lambda layer: (not layer.is_group() and layer.kind in DRAWN_LAYER_KINDS and layer.kind != "type"
                          and layer.is_visible() and layer.bbox[2] - layer.bbox[0] >= psd.width / 3 and layer.bbox[3] - layer.bbox[1] < psd.height / 8)
    own = [layer for layer in group if wide(layer)]
    if own:
        return max(own, key=lambda layer: layer.bbox[2] - layer.bbox[0])
    named = lambda layer: layer is not None and not isinstance(layer, PSDImage) \
        and fullmatch(rf"\s*({pattern})\s*(bars?)?\s*", layer.name, IGNORECASE) is not None
    found = []
    for layer in psd.descendants():
        if wide(layer) and any(search(r"bar|stat", owner.name, IGNORECASE) for owner in _ancestors(layer)):
            if named(layer.parent):
                found.append((0, layer))
            elif named(layer):
                found.append((1, layer))
    return min(found, key=lambda item: item[0])[1] if found else None


def _bar_box(psd: PSDImage, group, stat: str, layer=None) -> Optional[tuple]:
    """The bar a row draws (x0, y0, x1, y1): _bar_layer's with the row's cells mostly past it (Clot Black's dark strip),
    else its cells' (Clots), else where psd_regions finds it (the template's row by its label)."""
    if layer is not None:
        x0, x1 = layer.bbox[0], layer.bbox[2]
        past = [] if layer.parent is not group else [
            tuple(leaf.bbox) for leaf in group if not leaf.is_group() and leaf.kind in DRAWN_LAYER_KINDS
            and leaf.kind != "type" and leaf.bbox[2] > leaf.bbox[0]
            and max(0, min(leaf.bbox[2], x1) - max(leaf.bbox[0], x0)) < (leaf.bbox[2] - leaf.bbox[0]) / 2]
        return clip(union_box([tuple(layer.bbox)] + past), psd.width, psd.height)
    cells = union_box([tuple(leaf.bbox) for leaf in group if not leaf.is_group() and leaf.kind in DRAWN_LAYER_KINDS
                       and leaf.kind != "type" and leaf.bbox[2] > leaf.bbox[0]])
    if cells and cells[2] - cells[0] >= psd.width / 3:
        return cells
    from commands.psd_regions import regions
    found = regions(psd).get("stat_" + stat) or []
    return found[0] if found else None


def _islands(layer) -> Optional[list]:
    """The (start, end) x of the ten shapes a layer draws along its middle row (Isaac's hearts), else None."""
    alpha = _alpha(layer)
    if alpha is None or not alpha.size:
        return None
    band = alpha[alpha.shape[0] // 4:3 * alpha.shape[0] // 4].max(axis=0) > INK_ALPHA
    runs, start = [], None
    for x, on in enumerate(list(band) + [False]):
        if on and start is None:
            start = x
        elif not on and start is not None:
            runs.append((layer.bbox[0] + start, layer.bbox[0] + x))
            start = None
    return runs if len(runs) == 10 else None


def _cell_edges(box: tuple, cells: Optional[list] = None, islands: Optional[list] = None) -> list:
    """Where the bar's cells start, 0-10 (10: the bar's end): from the card's own cells or the ten shapes its bar draws,
    else the template's spaced over the bar."""
    if islands:
        return [islands[0][0]] + [(islands[k - 1][1] + islands[k][0]) / 2 for k in range(1, 10)] + [islands[9][1]]
    if cells:
        starts = [min(layer.bbox[0] for layer in cell) for cell in cells]
        ends = [max(layer.bbox[2] for layer in cell) for cell in cells]
        return [starts[0]] + [(ends[k - 1] + starts[k]) / 2 for k in range(1, 10)] + [ends[9]]
    x0, x1 = box[0], box[2]
    scale = (x1 - x0) / (BAR_EDGES[10] - BAR_EDGES[0])
    return [x0 + (edge - BAR_EDGES[0]) * scale for edge in BAR_EDGES]


def _cells(group, box: tuple) -> Optional[list]:
    """The row's one-cell layers (drawn; not clipped onto a cell: Chris & Jordi's art), ten cells by position:
    [[layer, ...] per cell], else None."""
    width = box[2] - box[0]
    narrow = lambda layer: 0 < layer.bbox[2] - layer.bbox[0] < CELL_SHARE * width
    small = [layer for layer in group if not layer.is_group() and layer.kind in DRAWN_LAYER_KINDS and layer.kind != "type"
             and narrow(layer) and not (layer.clipping and _clip_base(layer) is not None and narrow(_clip_base(layer)))]
    if len(small) < 10:
        return None
    widths = sorted(layer.bbox[2] - layer.bbox[0] for layer in small)
    median = widths[len(widths) // 2]
    small = [layer for layer in small if abs(layer.bbox[2] - layer.bbox[0] - median) <= median / 4]  # art over a cell
    if len(small) < 10:
        return None
    small.sort(key=lambda layer: (layer.bbox[0] + layer.bbox[2]) / 2)
    pitch = width / 10
    cells: list = [[small[0]]]
    for layer in small[1:]:
        centre = (layer.bbox[0] + layer.bbox[2]) / 2
        last = cells[len(cells) - 1]
        if centre - (last[0].bbox[0] + last[0].bbox[2]) / 2 < pitch / 2:
            last.append(layer)
        else:
            cells.append([layer])
    return cells if len(cells) == 10 else None


def _row_reads(group, names: Optional[dict] = None, shown: Optional[dict] = None) -> Optional[int]:
    """What the parser reads for a row (StatTracker._row_value) once layers are renamed or shown or hidden."""
    names, shown = names or {}, shown or {}
    items = []
    for layer in group:
        name = names.get(id(layer), base_name(layer)).strip()
        if name.isdigit() and shown.get(id(layer), layer.visible) and not layer.is_group():
            items.append((int(name), layer.kind))
    return StatTracker._row_value(items)[0] if items else None


def _profile(layer, use_mask: bool, y: int, x0: int, x1: int, dx: int = 0) -> Optional[np.ndarray]:
    """
    A layer along the bar's middle row over x0..x1, as on/off: its mask (outside it the mask's background) moved by dx,
    or its pixels' alpha moved by dx (nothing outside them); None when it has none to read.
    """
    xs = np.arange(x0, x1) - dx
    if use_mask:
        mask = layer.mask
        if mask.disabled:
            return None
        try:
            image = mask.topil()
        except Exception:
            image = None
        out = np.full(len(xs), mask.background_color >= 128)
        if image is not None and mask.width and mask.height and mask.top <= y < mask.bottom:
            row = np.asarray(image)[y - mask.top] >= 128
            inside = (xs >= mask.left) & (xs < mask.left + len(row))
            out[inside] = row[xs[inside] - mask.left]
        return out
    alpha = _alpha(layer)
    if alpha is None or not alpha.size:
        return None
    left, top = layer.bbox[0], layer.bbox[1]
    out = np.zeros(len(xs), dtype=bool)
    if top <= y < top + alpha.shape[0]:
        row = alpha[y - top] > INK_ALPHA
        inside = (xs >= left) & (xs < left + len(row))
        out[inside] = row[xs[inside] - left]
    return out


def _step(profile: np.ndarray, x0: int, at: float, tol: float, bar: tuple, loose: bool = False) -> Optional[tuple]:
    """
    (left side, right side) of an on/off profile from x0 that is constant on each side of `at` (outside +-tol) and
    changes across it; None otherwise. The sides are read over the bar (x0, x1); a side the bar does not have (`at` on
    its start or end) just past it. loose (R4, edges anywhere): the sides are read half a tolerance inside the bar's
    ends (Twin Emperors' masks stop 3 px short of it), and a side the bar does not have is unread (None): past the bar
    an exposure clipped to it darkens nothing (Slugcat's 0 and 10).
    """
    if loose:
        margin = int(tol / 2)
        sides = []
        for lo, hi, there in ((bar[0] + margin, int(at - tol), at - tol > bar[0]),
                              (int(at + tol), bar[1] - margin, at + tol < bar[1])):
            part = profile[max(lo - x0, 0):max(hi - x0, 0)] if there else profile[:0]
            if part.size and not (part.all() or not part.any()):
                return None
            sides.append(bool(part[0]) if part.size else None)
        if sides == [None, None] or (None not in sides and sides[0] == sides[1]):
            return None
        return sides[0], sides[1]
    lo, hi = (bar[0], int(at - tol)) if at - tol > bar[0] else (int(at - 2 * tol), int(at - tol))
    lo2, hi2 = (int(at + tol), bar[1]) if at + tol < bar[1] else (int(at + tol), int(at + 2 * tol))
    left = profile[max(lo - x0, 0):max(hi - x0, 0)]
    right = profile[max(lo2 - x0, 0):max(hi2 - x0, 0)]
    sides = []
    for part in (left, right):
        if part.size and not (part.all() or not part.any()):
            return None
        sides.append(bool(part[0]) if part.size else None)
    if sides[0] is not None and sides[1] is not None and sides[0] == sides[1]:
        return None
    return sides[0], sides[1]


def _edge_elements(psd: PSDImage, stat: str, box: tuple, edges: list, old: int, loose: bool = False,
                   tall: bool = False, bar_layer=None) -> list:
    """
    The layers drawing the bar's value edge: a mask or pixels that switch on or off at the old value's cell edge and
    stay so across the rest of the bar (Isaac's lit hearts' mask, Chef Saltbaker's dark exposures, Pizza's dark
    overlays), each inside the stat's own row. loose: _step's (R4); tall: one picked from several rows of the stat
    (bar_rows) may span them all (Pix's shared HP mask over its three creatures' rows). The bar layer's own pixels, and
    any over the whole bar, are the bar, never its edge (at 0 or 10 they switch at its end: Uumuu's "hp").

    Returns:
        [(layer, use_mask)]
    """
    y0, y1 = box[1], box[3]
    y = (y0 + y1) // 2
    tol = EDGE_TOL * (edges[10] - edges[0]) / 10
    lo, hi = int(edges[0] - 2 * tol), int(edges[10] + 2 * tol)
    others = [found for key in STAT_FIELDS if key != stat for found in stat_rows(psd)[key]]
    other_bands = [b for _, _, b in others if b]
    out = []
    for layer in psd.descendants():
        if layer.is_group() or layer.kind == "type" or not layer.is_visible():
            continue
        for use_mask in (True, False):
            if use_mask and not (layer.has_mask() and layer.mask.width and layer.mask.height):
                continue
            if not use_mask and (layer.kind not in DRAWN_LAYER_KINDS or layer.bbox[2] <= layer.bbox[0] or layer is bar_layer):
                continue
            extent = tuple(layer.mask.bbox) if use_mask else tuple(layer.bbox)
            if layer.kind in DRAWN_LAYER_KINDS and layer.bbox[2] > layer.bbox[0]:
                extent = (extent[0], max(extent[1], layer.bbox[1]), extent[2], min(extent[3], layer.bbox[3]))
            if not (extent[1] < y < extent[3]) or extent[3] - extent[1] > (4 if tall else 2.5) * (y1 - y0):
                continue
            if any(min(extent[3], b[3]) - max(extent[1], b[1]) > (b[3] - b[1]) / 4 for b in other_bands):
                continue
            profile = _profile(layer, use_mask, y, lo, hi)
            if profile is not None and not use_mask and old in (0, STAT_MAX) \
                    and profile[int(edges[0] + tol) - lo:int(edges[10] - tol) - lo].all():
                continue                        # pixels over the whole bar at 0 or 10: its art, not its edge (Pix's
                #   "White Lines copy", Twin Emperors' "Shared Hp": moved, they left the bar)
            if profile is not None and _step(profile, lo, edges[old], tol, (int(edges[0]), int(edges[10])), loose):
                out.append((layer, use_mask))
                break
    return out


def _roles(cells: list, old: int, elsewhere=None) -> dict:
    """
    id(layer) -> "lit", "dark", "base" or "spare" for a row of cells shown and hidden: per layer series (the n-th layer
    of each cell, bottom to top) when every cell has as many: shown up to the value (lit), past it (dark) or always
    (base: the Clots' dark hearts under the lit ones); else per layer, by whether it shows on its side of the value.

    A series hidden in every cell is spare art, never shown (Clot Gold's HP; E2), save at 0 (10) with no other series
    lit (dark): then it is that side, the ones whose picture another bar row shows (elsewhere(layer)), else the topmost.
    """
    out = {}
    sizes = {len(cell) for cell in cells}
    if len(sizes) == 1 and sizes.pop() > 1:
        ordered = [sorted(cell, key=lambda layer: layer.parent.index(layer)) for cell in cells]
        hidden = [n for n in range(len(ordered[0])) if not any(cell[n].visible for cell in ordered)]
        for n in range(len(ordered[0])):         # a cell or two off the pattern (Clot White's dark 7 under its lit 7)
            shown = [cell[n].visible for cell in ordered]
            lit = sum(on == (k <= old) for k, on in enumerate(shown, 1))
            dark = sum(on == (k > old) for k, on in enumerate(shown, 1))
            role = "lit" if lit >= 8 and lit > dark else "dark" if dark >= 8 and dark > lit else "base" if all(shown) else None
            if role is None and n not in hidden:
                break
            out.update({id(cell[n]): role for cell in ordered})
        else:
            side = "lit" if old == 0 else "dark" if old == STAT_MAX else None
            keep, first = [], ordered[0]
            if side and hidden and not any(out[id(first[n])] == side for n in range(len(first)) if n not in hidden):
                keep = [n for n in hidden if elsewhere is not None and elsewhere(first[n])] or [hidden[len(hidden) - 1]]
            for n in hidden:
                out.update({id(cell[n]): side if n in keep else "spare" for cell in ordered})
            return out
        out = {}
    for k, cell in enumerate(cells, 1):
        for layer in cell:
            out[id(layer)] = "lit" if layer.visible == (k <= old) else "dark"
    return out


def _elsewhere(psd: PSDImage, cells: list):
    """Whether a layer's picture shows on another bar row of the card: a visible layer of the same smart object source,
    or of the same pixels (_roles)."""
    mine = {id(layer) for cell in cells for layer in cell}
    shown: list = []

    def same(a, b) -> bool:
        if a.kind != b.kind:
            return False
        if a.kind == "smartobject":
            return a.smart_object.unique_id == b.smart_object.unique_id
        if a.kind != "pixel" or (a.width, a.height) != (b.width, b.height):
            return False
        x, y = a.numpy(), b.numpy()
        return x is not None and y is not None and np.array_equal(x, y)

    def shows(layer) -> bool:
        if not shown:                           # read once, when asked
            shown.append([one for stat in STAT_FIELDS for group, _, _ in stat_rows(psd)[stat] if group.is_group()
                          for one in group.descendants()
                          if id(one) not in mine and not one.is_group() and one.is_visible()])
        return any(same(layer, other) for other in shown[0])
    return shows


def _mask_start(layer) -> int:
    """Where a mask starts darkening along its middle row: its first column on (a redrawn mask keeps blank columns
    before it: _held), the rows' most common start when the middle row's is off theirs, else its left."""
    mask = layer.mask
    try:
        raster = np.asarray(mask.topil().convert("L"))
    except Exception:
        return mask.left
    if mask.background_color >= 128 or not raster.size:
        return mask.left
    on = np.flatnonzero(raster[raster.shape[0] // 2] >= 128)
    if not on.size:
        return mask.left
    starts = [int(row[0]) for row in (np.flatnonzero(line >= 128) for line in raster) if row.size]
    if abs(int(on[0]) - np.median(starts)) > 2:     # a gap in the middle row (Twin Emperors' "1"): the most common start
        return mask.left + Counter(starts).most_common(1)[0][0]
    return mask.left + int(on[0])


def _digit_edges(group) -> dict:
    """{k: x} where the row's masked digit k (1-9, adjustment layers) starts darkening: the template's cell edges as
    the card maker drew them (of two k, the row's own: the shorter mask; Pix's tall shared one is the value's edge)."""
    found: dict = {}
    for layer in group:
        digit = None if layer.is_group() else _digit(layer)
        if digit is not None and 0 < int(digit) < STAT_MAX and layer.kind not in DRAWN_LAYER_KINDS and layer.has_mask() \
                and layer.mask.width:
            found.setdefault(int(digit), []).append((layer.mask.height, _mask_start(layer)))
    return {k: min(found[k])[1] for k in found}


def _row_edges(psd: PSDImage, stat: str, group, own: dict) -> dict:
    """The row's _digit_edges (own), the stat's other rows giving those it lacks (Twin Emperors' "7" renamed by a bake)."""
    out: dict = {}
    for other, _, _ in stat_rows(psd)[stat]:
        if other is not group:
            for k, x in _digit_edges(other).items():
                out.setdefault(k, x)
    return {**out, **own}


def _held(layer, dx: int, edge: float, tol: float, loose: bool = False) -> tuple:
    """
    A mask's raster with its value edge moved dx and nothing else: between the old and new edge (tol past each) every
    column takes the one dx before it, the rest of the mask stays as drawn (its far side, a rounded bar end: a moved
    mask would stop short of the bar or darken the art past it: Dark Esau's hearts).

    loose (R4, a move of any length): only the edge's own columns (within tol) move dx; the rest of the stretch takes
    the column beside the old edge on the side that grows over it (a long move copied the mask's outside: Slugcat's).

    Returns:
        (raster over the mask and that stretch, its left)
    """
    mask = layer.mask
    raster = np.asarray(mask.topil().convert("L"))
    height, width = raster.shape
    start, end = int(min(edge, edge + dx) - tol), int(max(edge, edge + dx) + tol) + 1
    left = min(mask.left, start)
    xs = np.arange(left, max(mask.left + width, end))
    source = np.where((xs >= start) & (xs < end), xs - dx, xs) - mask.left
    if loose:
        grows = int(edge - tol - 1) if dx > 0 else int(edge + tol + 1)      # the side that grows over the stretch;
        last = mask.left + width - 1                                        # past the mask's end, its last column there
        if dx > 0 and grows < mask.left < edge - tol / 4:                   # (an edge moved to 0 or 10 before)
            grows = mask.left
        elif dx < 0 and grows > last > edge + tol / 4:
            grows = last
        inside = (xs >= start) & (xs < end)
        far = inside & (np.abs(xs - dx - edge) > tol) & ((xs < edge + dx - tol) if dx > 0 else (xs > edge + dx + tol))
        source = np.where(far, grows, source) - np.where(far, mask.left, 0)
    out = np.full((height, len(xs)), mask.background_color, dtype=np.uint8)
    there = (source >= 0) & (source < width)
    out[:, there] = raster[:, source[there]]
    return out, left


def _remask(layer, dx: int, edge: float, tol: float, loose: bool = False) -> None:
    """Redraw the layer's mask as _held() makes it."""
    raster, left = _held(layer, dx, edge, tol, loose)
    layer.update_mask(Image.fromarray(np.ascontiguousarray(raster)), top=layer.mask.top, left=left)


def _held_profile(layer, dx: int, edge: float, tol: float, y: int, x0: int, x1: int, loose: bool = False) -> np.ndarray:
    """_profile() of the mask _held() would make (the layer is not changed)."""
    mask = layer.mask
    raster, left = _held(layer, dx, edge, tol, loose)
    xs = np.arange(x0, x1)
    out = np.full(len(xs), mask.background_color >= 128)
    if mask.top <= y < mask.top + raster.shape[0]:
        row = raster[y - mask.top] >= 128
        inside = (xs >= left) & (xs < left + len(row))
        out[inside] = row[xs[inside] - left]
    return out


def none_shows(stat: str, value: int, group) -> str:
    """The refusal of a row that would show no cell (a 0 with no dark picture): what the card maker can add."""
    return (f"{stat} {value}: {group.name!r} would show no cell, and a row showing none has no value to read: give its "
            "cells dark pictures named 1-10 under the lit ones (as Grimora's), or a dark strip named 0 in the row (as "
            "the Clots')")


def plan_bar(psd: PSDImage, group, stat: str, old: int, value: int, why: Refused, loose: bool = False,
             tall: bool = False, values: Optional[dict] = None) -> BarPlan:
    """
    How a stat changes on a bar the card maker drew their own way, following their layout (owner 10-24: support, not
    refusals; 10-25: no art swap). The row (the stat's one digit row, stat_rows) may draw its value with:

    - cells: one layer per cell, lit ones and dark ones shown or hidden (Blank Card's pairs, the Clots' and Minecraft
      Panda's lit cells over dark ones or a dark strip): each cell shows its lit or dark layer (_roles); a cell with
      neither refuses;
    - cell styles: every cell shown, lit and dark by their effects (Zigg, Chris & Jordi, Uumuu): a cell crossing over
      takes the effect blocks of the last lit or first dark cell;
    - an edge: a mask or a layer switching at the value's cell edge (Isaac's hearts, Chef Saltbaker's and Slugcat's
      exposures, Pizza's dark overlays, Uumuu's): moved by the cells between the two values; a mask that would stop
      short of the bar is redrawn with its far side held (_remask); a layer, refused;

    and the parser reads it from the layers' names: a layer named after the value is renamed (cells shown and hidden: the
    lit cell at the new value takes the name, the old one "0"); the result must read `value` (StatTracker's rules).

    Returns:
        BarPlan; raises Refused("unsupported-card") naming what the bar lacks, or `why` when nothing draws the value
    """
    bar_layer = _bar_layer(psd, group, stat)
    box = _bar_box(psd, group, stat, bar_layer)
    if box is None:
        raise why
    if _row_reads(group) != old:
        raise Refused("unsupported-card", f"{stat}: the parser reads {old}, the row {group.name!r} reads {_row_reads(group)}")
    cells = _cells(group, box)
    if cells is None and bar_layer is not None and bar_layer.parent is not group and not isinstance(bar_layer.parent, PSDImage):
        cells = _cells(bar_layer.parent, box)   # Uumuu: the digit in "Darks", the cells beside the bar
    drawn = [layer for layer in group if not layer.is_group() and layer.kind in DRAWN_LAYER_KINDS and layer.kind != "type"
             and layer.bbox[2] - layer.bbox[0] >= psd.width / 3] + ([bar_layer] if bar_layer is not None else [])
    shapes = None if cells else next((found for layer in sorted(drawn, key=lambda layer: layer.bbox[2] - layer.bbox[0])
                                      for found in [_islands(layer)] if found), None)
    edges = _cell_edges(box, cells, shapes)
    own = _digit_edges(group)
    if len(own) >= 5:                           # a digit row: its own edges, strict too (a renamed digit: other rows')
        own = _row_edges(psd, stat, group, own)
    if loose or len(own) >= 5:                  # the row's own digit masks say where its cells start (Pix's)
        for k, x in own.items():
            edges[k] = x
    plan = BarPlan("", box)
    names, shown, hows = {}, {}, []
    strip = cells is not None and old == STAT_MAX and any(    # at 10 every lit cell shows over a "0" strip (Minecraft Panda)
        not layer.is_group() and layer.visible and layer.kind in DRAWN_LAYER_KINDS and _digit(layer) == "0"
        and all(layer is not one for cell in cells for one in cell) for layer in group)
    toggled = cells is not None and (strip or any(not layer.visible for cell in cells for layer in cell))
    if toggled:
        hows.append("cells shown and hidden")
        roles = _roles(cells, old, _elsewhere(psd, cells))
        under = [layer for container in (group, group.parent) if container is not None and not isinstance(container, PSDImage)
                 for layer in container if not layer.is_group() and layer.is_visible() and layer.kind in DRAWN_LAYER_KINDS
                 and all(layer is not one for cell in cells for one in cell)
                 and (container is group or container.index(layer) < container.index(group))]
        for k, cell in enumerate(cells, 1):
            if not min(old, value) < k <= max(old, value):
                continue                        # cells that keep their side keep their layers as they are
            lit = [layer for layer in cell if roles[id(layer)] == "lit"]
            dark = [layer for layer in cell if roles[id(layer)] == "dark"]
            base = [layer for layer in cell if roles[id(layer)] == "base"]
            if k <= value:
                if not lit:
                    raise Refused("unsupported-card", f"{stat} {value}: cell {k} of {group.name!r} has no lit picture")
                plan.show += [layer for layer in lit if not layer.visible]
                plan.hide += [layer for layer in dark if layer.visible]
            else:
                x0, x1 = min(l.bbox[0] for l in cell), max(l.bbox[2] for l in cell)
                covered = base or any(min(x1, u.bbox[2]) - max(x0, u.bbox[0]) >= COVER_SHARE * (x1 - x0)
                                      and u.bbox[1] < box[3] and u.bbox[3] > box[1] for u in under)
                if not dark and not covered:
                    raise Refused("unsupported-card", f"{stat} {value}: cell {k} of {group.name!r} has no dark picture")
                if dark and not any(layer.visible for layer in dark):   # one dark picture, not a stack of them
                    plan.show.append(next((layer for layer in dark if base_name(layer).strip() == str(k)), dark[0]))
                plan.hide += [layer for layer in lit if layer.visible]
        shown = {id(layer): True for layer in plan.show} | {id(layer): False for layer in plan.hide}
    elif cells:
        bases = [[layer for layer in cell if layer.visible] for cell in cells]
        sig = lambda layer: tuple(sorted((bytes(key), layer.tagged_blocks[key].tobytes()) for key in EFFECT_BLOCKS
                                         if key in layer.tagged_blocks))
        lit_ex = bases[old - 1][0] if old >= 1 and bases[old - 1] else None
        dark_ex = bases[old][0] if old < STAT_MAX and bases[old] else None
        if (lit_ex is None) != (dark_ex is None):       # 0 or 10: the other side's style is on another row of the card
            ex = lit_ex or dark_ex
            other = _side_example(psd, stat, values, lit_ex is None, ex,
                                  (min(l.bbox[0] for l in cells[0]) + max(l.bbox[2] for l in cells[0])) / 2 if lit_ex is None
                                  else (min(l.bbox[0] for l in cells[9]) + max(l.bbox[2] for l in cells[9])) / 2)
            lit_ex, dark_ex = (other, dark_ex) if lit_ex is None else (lit_ex, other)
        if lit_ex is not None and dark_ex is not None and sig(lit_ex) != sig(dark_ex):
            hows.append("cell styles")
        if hows:
            for k in range(min(old, value) + 1, max(old, value) + 1):
                source = lit_ex if k <= value else dark_ex
                plan.styles += [(layer, source) for layer in bases[k - 1] if sig(layer) != sig(source)]
    elements = _edge_elements(psd, stat, box, edges, old, loose, tall, bar_layer)
    if elements:
        hows.append("the value's edge moved")
        dx = round(edges[value] - edges[old])
        tol = EDGE_TOL * (edges[10] - edges[0]) / 10
        y = (box[1] + box[3]) // 2
        lo, hi, bar = int(edges[0] - 2 * tol), int(edges[10] + 2 * tol), (int(edges[0]), int(edges[10]))
        for layer, use_mask in elements:     # a mask is redrawn between the two edges, the rest kept (_held)
            before = _step(_profile(layer, use_mask, y, lo, hi), lo, edges[old], tol, bar, loose)
            same = lambda moved: moved is not None and all(m is None or b is None or m == b for m, b in zip(moved, before))
            if not use_mask and same(_step(_profile(layer, use_mask, y, lo, hi, dx), lo, edges[value], tol, bar, loose)):
                plan.moves.append((layer, dx))
            elif use_mask and same(_step(_held_profile(layer, dx, edges[old], tol, y, lo, hi, loose), lo, edges[value], tol,
                                         bar, loose)):
                plan.remasks.append((layer, dx, edges[old], tol, loose))
            else:
                raise Refused("unsupported-card", f"{stat} {value}: {layer.name!r} moved {dx:+d} px would no longer cover "
                                                  f"the bar on both sides of the value")
    if not hows:
        if all(not layer.is_group() and _digit(layer) is not None and layer.bbox[2] <= layer.bbox[0] for layer in group):
            raise Refused("unsupported-card", f"{stat}: {group.name!r} only names the value; no layer draws it (the bar's "
                                              "light and dark cells are painted): edit it in Photoshop")
        if _row_reads(group) is not None:
            raise Refused("unsupported-card", f"{stat}: no layer of the bar shows or hides a cell, or switches at {old}, "
                                              f"the value {group.name!r} names: edit it in Photoshop")
        raise why
    if _row_reads(group, names, shown) != value:    # the parser's reading: the layer named after the value takes the new one
        named = [layer for layer in group if not layer.is_group() and _digit(layer) == str(old) and layer.visible]
        # a lit cell named, or at 0 a "0" strip (Minecraft Panda's): the lit cell at the value takes the name
        if toggled and named and (not old or any(layer is one for cell in cells for one in cell for layer in named)):
            for layer in named:
                names[id(layer)] = "0"
            for cell in cells:                  # lit cells shown again drop an old value's name (1 -> 0 -> 2)
                for layer in cell:
                    if shown.get(id(layer), layer.visible) and _digit(layer) not in (None, "0"):
                        names[id(layer)] = "0"
            if value:
                lit = [layer for layer in cells[value - 1] if shown.get(id(layer), layer.visible)]
                if lit:
                    names[id(max(lit, key=lambda layer: layer.parent.index(layer)))] = str(value)
        elif len(named) == 1:
            names[id(named[0])] = str(value)
        elif len(named) > 1:                    # Keeper's "0" dark and "0" lit: the one with the edge
            edged = [layer for layer in named if any(layer is element for element, _ in elements)]
            if len(edged) == 1:
                names[id(edged[0])] = str(value)
    every = [layer for layer in group] + [layer for cell in cells or [] for layer in cell]
    plan.names = list({id(layer): (layer, names[id(layer)]) for layer in every
                       if id(layer) in names and layer.name.strip() != names[id(layer)]}.values())
    if _row_reads(group, names, shown) is None:
        raise Refused("unsupported-card", none_shows(stat, value, group))
    if _row_reads(group, names, shown) != value:
        raise Refused("unsupported-card", f"{stat} {value}: the parser would read {_row_reads(group, names, shown)} "
                                          f"off {group.name!r}")
    plan.how = ", ".join(hows)
    return plan


COUNT_TOL = 0.15            # a counted row's pictures: sizes and steps within this share of each other


def counted_row(group) -> Optional[dict]:
    """{k: layer} of a row of pictures counted: pixel layers alone, named 1..n by place, one size, on one line at one
    pitch; None else."""
    layers = list(group)
    if not layers or any(layer.is_group() or layer.kind != "pixel" or not base_name(layer).strip().isdigit()
                         for layer in layers):
        return None
    by = {int(base_name(layer)): layer for layer in layers}
    if sorted(by) != list(range(1, len(layers) + 1)):
        return None
    w, h = by[1].bbox[2] - by[1].bbox[0], by[1].bbox[3] - by[1].bbox[1]
    mid = lambda layer: (layer.bbox[1] + layer.bbox[3]) / 2
    if w <= 0 or any(abs(layer.bbox[2] - layer.bbox[0] - w) > COUNT_TOL * w or abs(layer.bbox[3] - layer.bbox[1] - h)
                     > COUNT_TOL * h or abs(mid(layer) - mid(by[1])) > COUNT_TOL * h for layer in layers):
        return None
    steps = [by[k + 1].bbox[0] - by[k].bbox[0] for k in range(1, len(by))]
    if steps and (min(steps) <= w or max(steps) - min(steps) > COUNT_TOL * min(steps)):
        return None
    return by


def _count_room(group, by: dict) -> tuple:
    """(pitch, right edge) of a counted row: its own pitch, else a sibling counted row's of its size; the edge its
    stat's counted rows of that size reach, shown or hidden (a copy goes no further)."""
    rows = [by] + [found for g in group.parent if g is not group and g.is_group() for found in [counted_row(g)] if found]
    width = by[1].bbox[2] - by[1].bbox[0]
    alike = [found for found in rows if abs(found[1].bbox[2] - found[1].bbox[0] - width) <= COUNT_TOL * width]
    pitch = next(((found[len(found)].bbox[0] - found[1].bbox[0]) / (len(found) - 1) for found in alike if len(found) > 1),
                 None)
    return pitch, max(layer.bbox[2] for found in alike for layer in found.values())


def _count_box(by: dict, k: int, pitch: float) -> tuple:
    """(dx, box) of picture k copied from a counted row's last one along its pitch."""
    last = by[max(by)]
    dx = round(by[1].bbox[0] + (k - 1) * pitch - last.bbox[0])
    return dx, (last.bbox[0] + dx, last.bbox[1], last.bbox[2] + dx, last.bbox[3])


def count_room(psd: PSDImage, group) -> Optional[int]:
    """The most pictures a counted row shows (plan_count's reach, up to 20); None for no counted row."""
    by = counted_row(group)
    if by is None:
        return None
    room = max(by)
    pitch, edge = _count_room(group, by)
    while pitch is not None and room < 2 * STAT_MAX:
        _, box = _count_box(by, room + 1, pitch)
        if box[2] > edge + 1 or clip(box, psd.width, psd.height) != box:
            break
        room += 1
    return room


def plan_count(psd: PSDImage, group, stat: str, old: int, value: int) -> Optional[BarPlan]:
    """
    A stat drawn as pictures counted, no empty cells (Computer Virus's squares): a counted_row whose pictures 1..old
    show and none past them, as the parser reads it. The edit hides those past the value, shows hidden ones up to it and
    copies the last one along the row's pitch where it has none, as far as the stat's rows reach (hidden pictures mark
    the room). None when the row is no such row.
    """
    by = counted_row(group)
    if by is None or sorted(k for k, layer in by.items() if layer.visible) != list(range(1, old + 1)):
        return None
    if value < 1:
        raise Refused("unsupported-card", f"{stat} {value}: {group.name!r} would show no picture, and a row showing none "
                                          "has no value to read")
    plan = BarPlan("pictures counted", union_box([tuple(layer.bbox) for layer in by.values()]))
    plan.hide = [by[k] for k in range(value + 1, old + 1)]
    plan.show = [by[k] for k in range(old + 1, value + 1) if k in by]
    last = max(by)
    pitch = _count_room(group, by)[0] if value > last else None
    if value > last and pitch is None:
        raise Refused("unsupported-card", f"{stat} {value}: {group.name!r} has one picture, no step to place more by")
    room = count_room(psd, group)
    if value > room:
        raise Refused("unsupported-card", f"{stat} {value}: {group.name!r} has room for {room} pictures (add hidden "
                                          "ones where more go)")
    for n, k in enumerate(range(last + 1, value + 1)):
        dx, box = _count_box(by, k, pitch)
        plan.copies.append((by[last], group, len(by) + n, dx, 0, str(k)))
        plan.box = union_box([plan.box, box])
    if plan.copies:
        plan.notes.append(f"{stat}: {group.name!r} had {last} pictures: {len(plan.copies)} copied along the row")
    return plan


def _side_example(psd: PSDImage, stat: str, values: Optional[dict], lit: bool, like, centre: float):
    """
    The cell another stat's row shows on the side this row has none of (at 0 no lit cell, at 10 no dark one): the
    topmost visible layer of like's kind and size at x `centre` (this row's first or last cell) in the nearest row whose
    value has a cell on that side (False Knight's DEF at 10 takes ATK's dark look; Uumuu's HP at 0 a lit one). None
    when no row has one.
    """
    if not values:
        return None
    width, height = like.bbox[2] - like.bbox[0], like.bbox[3] - like.bbox[1]
    mine = (like.bbox[1] + like.bbox[3]) / 2
    order = {id(layer): k for k, layer in enumerate(psd.descendants())}
    best = None
    for key in STAT_FIELDS:
        value = values.get(key)
        if key == stat or not isinstance(value, int) or not (value >= 1 if lit else value < STAT_MAX):
            continue
        bands = [band for group, _, row in bar_rows(psd, key)[0]
                 for band in [row or _bar_box(psd, group, key, _bar_layer(psd, group, key))] if band]
        found = [layer for layer in psd.descendants() if not layer.is_group() and layer.kind == like.kind
                 and layer.is_visible() and abs(layer.bbox[2] - layer.bbox[0] - width) <= 0.1 * width
                 and abs(layer.bbox[3] - layer.bbox[1] - height) <= 0.25 * height
                 and abs((layer.bbox[0] + layer.bbox[2]) / 2 - centre) < width / 2
                 and any(band[1] <= (layer.bbox[1] + layer.bbox[3]) / 2 <= band[3] for band in bands)]
        if found:
            top = max(found, key=lambda layer: order[id(layer)])
            far = abs((top.bbox[1] + top.bbox[3]) / 2 - mine)
            if best is None or far < best[0]:
                best = (far, top)
    return best[1] if best else None


LIT_ROW = r"fill|light|lit\b|full"           # a row of lit cells (False Knight's "Filled", Tainted Lazarus's "Light")
DARK_ROW = r"empty|dark"                     # its dark row ("Empty", "Hp Dark")


def _cell_groups(psd: PSDImage, stat: str, rows: list) -> Optional[tuple]:
    """
    (lit row, dark row) of a stat whose cells are separate layers: two sibling rows named lit and dark (R2: False
    Knight's "Filled" over "Empty", even hidden: its DEF; Tainted Lazarus's "Light" over "Hp Dark"), or one row holding
    both, told apart by look (R6: pixel layers lit, smart objects dark, as Dirty Bed draws them): (row, row). None else.
    """
    groups = [group for group, _, _ in rows]
    if len(groups) == 1:
        groups += [group for group, _, _ in stat_rows(psd)[stat] if group is not groups[0] and group.parent is groups[0].parent]
    if len(groups) == 1:
        return groups[0], groups[0]
    if len(groups) != 2 or groups[0].parent is not groups[1].parent:
        return None
    lit = [group for group in groups if search(LIT_ROW, group.name, IGNORECASE)]
    dark = [group for group in groups if search(DARK_ROW, group.name, IGNORECASE) and group not in lit]
    return (lit[0], dark[0]) if len(lit) == 1 and len(dark) == 1 else None


def _shown(layer, shown: dict) -> bool:
    """layer.is_visible() once the layers in shown ({id: eye}) are shown or hidden."""
    while layer is not None and not isinstance(layer, PSDImage):
        if not shown.get(id(layer), layer.visible):
            return False
        layer = layer.parent
    return True


def _cells_read(groups: list, names: dict, shown: dict) -> Optional[int]:
    """What the parser reads off rows in one stat block (StatTracker.total): rows drawn per cell count once, others add."""
    found = []
    for group in dict.fromkeys(groups):
        items = [(int(names.get(id(layer), base_name(layer)).strip()), layer.kind) for layer in group if not layer.is_group()
                 and names.get(id(layer), base_name(layer)).strip().isdigit() and _shown(layer, shown)]
        if items:
            found.append(StatTracker._row_value(items))
    if not found:
        return None
    if len(found) > 1 and all(per_cell for _, per_cell in found):
        return max(value for value, _ in found)
    return sum(value for value, _ in found)


def _plus(reads: Optional[int], copies: int) -> Optional[int]:
    """A row's reading with its copied lit cells; None when it shows no digit."""
    return None if reads is None else reads + copies


def plan_cells(psd: PSDImage, stat: str, pair: tuple, old: int, value: int, values: Optional[dict] = None) -> BarPlan:
    """
    A stat whose cells are separate layers, edited by showing cells 1..value lit and the rest dark (the numbered layers
    rule and R2, owner 2026-10-26): the lit and dark layers of each cell, by their names 1-10 when the lit ones carry
    them (Tainted Lazarus), else by place along the bar (False Knight's "0"s). A dark layer under a lit one follows the
    card (hidden where the lit cells hide theirs, as Lazarus, else left as it is); a hidden dark row is shown (False
    Knight's DEF at 10). A cell with no lit picture takes a copy of the lit one the card draws at that cell on another
    stat's row, moved onto it (R6: an item's row drawn without one of its lit cells). Numbered layers 1-10 with no dark
    picture: those past the value are hidden, nothing shown. A crossing cell takes its new side's look, from another row
    when this one shows none (False Knight's DEF at 10: _side_example); a dark cell left under a lit one takes the
    card's look there, and a full row hides the dark row again. The layer named after the value is renamed when the
    parser reads the names (False Knight). Only where plan_bar refuses (set_bar).

    Returns:
        BarPlan; raises Refused("unsupported-card")
    """
    lit_group, dark_group = pair
    leaves = lambda group, kinds: [layer for layer in group if not layer.is_group() and layer.kind in kinds
                                   and base_name(layer).strip().isdigit() and layer.bbox[2] > layer.bbox[0]]
    drawn = tuple(kind for kind in DRAWN_LAYER_KINDS if kind != "type")
    if lit_group is dark_group:                 # numbered layers 1-10 alone (no dark picture: hidden past the value),
        lits, darks = leaves(lit_group, drawn), []      # else by look
        if sorted(int(base_name(layer)) for layer in lits) != list(range(1, STAT_MAX + 1)):
            lits, darks = leaves(lit_group, ("pixel",)), leaves(lit_group, ("smartobject",))
    else:
        lits, darks = leaves(lit_group, drawn), leaves(dark_group, drawn)
    if not lits or not (darks or lit_group is dark_group):
        raise Refused("unsupported-card", f"{stat}: no lit and dark cell layers in {lit_group.name!r}")
    every = lits + darks
    x0, x1 = min(layer.bbox[0] for layer in every), max(layer.bbox[2] for layer in every)
    pitch = (x1 - x0) / STAT_MAX
    names = [int(base_name(layer)) for layer in lits]
    by_name = sorted(names) == list(range(1, STAT_MAX + 1))
    mirrored = by_name and (min(lits, key=lambda layer: layer.bbox[0]) is next(l for l in lits if base_name(l).strip() == "10"))
    cell = lambda layer: min(STAT_MAX, max(1, int(((layer.bbox[0] + layer.bbox[2]) / 2 - x0) // pitch) + 1))
    place = (lambda layer: STAT_MAX + 1 - cell(layer)) if mirrored else cell
    cells = {k: ([], []) for k in range(1, STAT_MAX + 1)}
    for layer in lits:
        cells[int(base_name(layer)) if by_name else place(layer)][0].append(layer)
    for layer in darks:
        cells[place(layer)][1].append(layer)
    groups = [lit_group, dark_group]
    if _cells_read(groups, {}, {}) != old:
        raise Refused("unsupported-card", f"{stat}: the parser reads {old}, the cells read {_cells_read(groups, {}, {})}")
    lit_now = [k for k, (lit, _) in cells.items() if any(layer.is_visible() for layer in lit)]
    if lit_now != list(range(1, old + 1)):
        raise Refused("unsupported-card", f"{stat}: the lit cells {lit_now} are not cells 1 to {old}")
    under = [any(layer.visible for layer in cells[k][1]) for k in lit_now if cells[k][1]]
    hide_under = not under or not any(under)    # the card hides the dark layers under its lit cells
    plan, shown = BarPlan("cells by name" if by_name else "cells by place", (x0, min(l.bbox[1] for l in every), x1,
                                                                             max(l.bbox[3] for l in every))), {}
    if dark_group is not lit_group and not dark_group.visible and value < STAT_MAX:
        plan.show.append(dark_group)
        shown[id(dark_group)] = True
        hide_under = True
    for k in range(1, STAT_MAX + 1):
        lit, dark = cells[k]
        if k <= value:
            if not any(layer.visible for layer in lit):
                if lit:
                    pick = next((layer for layer in lit if base_name(layer).strip() == str(k)), lit[0])
                    plan.show.append(pick)
                    shown[id(pick)] = True
                else:
                    plan.copies.append(_lit_copy(psd, stat, lit_group, cells, k, pitch, x0, mirrored))
            for layer in dark:
                if layer.visible and (hide_under or k > old):
                    plan.hide.append(layer)
                    shown[id(layer)] = False
        else:
            for layer in lit:
                if layer.visible:
                    plan.hide.append(layer)
                    shown[id(layer)] = False
            if dark and not any(shown.get(id(layer), layer.visible) for layer in dark):
                pick = next((layer for layer in dark if base_name(layer).strip() == str(k)), dark[0])
                plan.show.append(pick)
                shown[id(pick)] = True
            elif not dark and not by_name:      # numbered layers: the lit one hidden is all
                raise Refused("unsupported-card", f"{stat} {value}: cell {k} of {lit_group.name!r} has no dark picture")
    sig = lambda layer: tuple(sorted((bytes(key), layer.tagged_blocks[key].tobytes()) for key in EFFECT_BLOCKS
                                     if key in layer.tagged_blocks))
    examples = [next((layer for k in range(1, old + 1) for layer in cells[k][0] if layer.visible), None),
                next((layer for k in range(old + 1, STAT_MAX + 1) for layer in cells[k][1] if layer.visible), None)]
    centre = lambda k: x0 + ((STAT_MAX - k) if mirrored else (k - 1)) * pitch + pitch / 2
    if examples[1] is None and value < old and darks:       # no dark cell shows (10): another row's (False Knight's DEF)
        examples[1] = _side_example(psd, stat, values, False, darks[0], centre(STAT_MAX))
    if examples[0] is None and value > old and lits:        # none lit (0)
        examples[0] = _side_example(psd, stat, values, True, lits[0], centre(1))
    covered = next((layer for k in range(1, old + 1) for layer in cells[k][1]   # a dark cell under a lit one
                    if layer.visible and any(one.visible for one in cells[k][0])), None)
    for k in range(min(old, value) + 1, max(old, value) + 1):   # a cell crossing over takes its new side's style (FK's
        lit, dark = cells[k]                                    # dark cells past 6 carry an effect the hidden ones lack)
        example = examples[0] if k <= value else examples[1]
        for layer in (lit if k <= value else dark):
            if example is not None and shown.get(id(layer), layer.visible) and sig(layer) != sig(example):
                plan.styles.append((layer, example))
        for layer in (dark if k <= value and covered is not None else ()):  # a dark one left under it: the card's look
            if shown.get(id(layer), layer.visible) and sig(layer) != sig(covered):     # there (a round trip's own)
                plan.styles.append((layer, covered))
    def place_names() -> dict:                  # the parser adds up the names: the value's lit cell takes it (FK)
        out = {}
        for k, (lit, _) in cells.items():
            for layer in lit:
                want = str(value) if k == value and shown.get(id(layer), layer.visible) else "0"
                if base_name(layer).strip() != want and base_name(layer).strip().isdigit():
                    out[id(layer)] = want
        return out
    names = {}
    if not by_name and dark_group is not lit_group and _plus(_cells_read(groups, {}, shown), len(plan.copies)) != value:
        names = place_names()                   # first, so the dark cells stay as the card shows them
    reads = _plus(_cells_read(groups, names, shown), len(plan.copies))
    for layer in reversed(list(plan.hide)):     # one dark cell left reads as its name (StatTracker._row_value): the
        if reads == value or layer not in darks:    # dark under the value's lit cell stays, covered (Lazarus's SPD 9)
            continue
        plan.hide.remove(layer)
        shown.pop(id(layer))
        reads = _plus(_cells_read(groups, names, shown), len(plan.copies))
    for k in range(value, 0, -1):               # one lit cell left reads as its name too: a dark under it shows, covered
        for layer in cells[k][1]:               # (Lazarus at 1)
            if reads != value and not shown.get(id(layer), layer.visible) and dark_group is not lit_group:
                plan.show.append(layer)
                shown[id(layer)] = True
                reads = _plus(_cells_read(groups, names, shown), len(plan.copies))
    if value == STAT_MAX and dark_group is not lit_group and _shown(dark_group, shown) \
            and _plus(_cells_read(groups, names, {**shown, id(dark_group): False}), len(plan.copies)) == reads:
        plan.hide.append(dark_group)            # a full row shows no dark row (False Knight's DEF at 10)
        shown[id(dark_group)] = False
    if reads != value and not by_name and not names:
        names = place_names()
        reads = _cells_read(groups, names, shown)
    if reads is None:                           # numbered layers at 0: none shows
        raise Refused("unsupported-card", none_shows(stat, value, lit_group))
    if reads != value:
        raise Refused("unsupported-card", f"{stat} {value}: the parser would read {reads} off {lit_group.name!r}")
    plan.names = [(layer, names[id(layer)]) for layer in lits if id(layer) in names and layer.name.strip() != names[id(layer)]]
    plan.how = plan.how + (", a lit cell copied" if plan.copies else "")
    return plan


def _lit_copy(psd: PSDImage, stat: str, group, cells: dict, k: int, pitch: float, x0: float, mirrored: bool) -> tuple:
    """
    A lit picture for cell k of a row that has none (R6: an item's row drawn without one of its lit cells): the visible
    lit layer of the same kind the card draws at that place on another stat's row, moved onto the row; (source, group,
    index, dx, dy, name). Refused when the card draws none there.
    """
    mine = [layer for lit, _ in cells.values() for layer in lit]
    width = sorted(layer.bbox[2] - layer.bbox[0] for layer in mine)[len(mine) // 2]
    top = sorted(layer.bbox[1] for layer in mine)[len(mine) // 2]
    centre = x0 + ((STAT_MAX - k) if mirrored else (k - 1)) * pitch + pitch / 2
    found = []
    for key in STAT_FIELDS:
        for other, _, _ in stat_rows(psd)[key] if key != stat else ():
            for layer in other:
                if (not layer.is_group() and layer.kind == mine[0].kind and layer.is_visible()
                        and abs((layer.bbox[0] + layer.bbox[2]) / 2 - centre) < pitch / 4
                        and abs(layer.bbox[2] - layer.bbox[0] - width) <= 0.1 * width):
                    found.append(layer)
    if not found:
        raise Refused("unsupported-card", f"{stat}: cell {k} of {group.name!r} has no lit picture, nor does the card draw one "
                                          "there on another row")
    source = min(found, key=lambda layer: (base_name(layer).strip() != str(k), abs(layer.bbox[1] - top)))
    anchor = min(mine, key=lambda layer: abs((layer.bbox[0] + layer.bbox[2]) / 2 - centre))
    return source, group, group.index(anchor), 0, top - source.bbox[1], str(k)


def no_glyph(layer, text: str, changed: list, fonts) -> list:
    """
    The characters of the changed ranges of text (set on the layer) whose font has no glyph for them (.notdef, gid 0):
    the engine and Photoshop draw a box ("5★" in the body font: review 2026-10-23), and a re-parse reads them as typed.
    """
    from psd_tools.composite.text.layout import layout
    from psd_tools.composite.text.model import TextModel
    model = TextModel.from_layer(layer)
    offset = max(model.text.find(text), 0)
    return [text[glyph.cluster - offset] for glyph in layout(model, fonts).glyphs
            if glyph.gid == 0 and any(a <= glyph.cluster - offset < b for a, b in changed)
            and text[glyph.cluster - offset] not in " \t\r\x03"]


def _ink_box(psd: PSDImage, layer, before: np.ndarray, pad: int = 0) -> Optional[tuple]:
    """
    The canvas box (x0, y0, x1, y1) holding all of the layer's pixels and before's ink past INK_ALPHA, padded by pad
    and clipped to the canvas; None when both are blank. The ink checks' masks are 0 past it: they look only there.
    """
    boxes = [clip(tuple(layer.bbox), psd.width, psd.height)]
    alpha = before[..., 3]
    rows = np.flatnonzero(alpha.max(axis=1) > INK_ALPHA * 255)
    if len(rows):
        y0, y1 = int(rows[0]), int(rows[len(rows) - 1]) + 1
        cols = np.flatnonzero(alpha[y0:y1].max(axis=0) > INK_ALPHA * 255)
        boxes.append((int(cols[0]), y0, int(cols[len(cols) - 1]) + 1, y1))
    box = union_box([b for b in boxes if b[2] > b[0] and b[3] > b[1]])
    return None if box is None else _pad(box, pad, psd)


def inks_over(psd: PSDImage, layer, before: np.ndarray) -> Optional[str]:
    """
    What an Ability's new ink would print over (ink_beyond's fallback): a visible layer drawn below it (earlier in paint
    order; no group, no clipped one) that, alone, inks past INK_ALPHA under the text's changed ink past INK_ALPHA, the
    old text's ink touching none of it. By pixels, not names. The panel and background the old text lies on are the
    design.

    Args:
        psd: The card, the text set and rasterized
        layer: The Ability layer
        before: _on_canvas() of it before the edit

    Returns:
        Why, or None
    """
    win = _ink_box(psd, layer, before)
    if win is None:
        return None
    ox, oy, wx, wy = win
    after, near = _on_canvas(layer, (psd.height, psd.width), win), before[oy:wy, ox:wx]
    ink = (after[..., 3] > INK_ALPHA * 255) & _px_differ(after, near)
    was = near[..., 3] > INK_ALPHA * 255
    if not ink.any():
        return None
    ys, xs = (ink | was).nonzero()
    box = (int(xs.min()) + ox, int(ys.min()) + oy, int(xs.max()) + 1 + ox, int(ys.max()) + 1 + oy)
    order = list(psd.descendants())
    below = order[:next(i for i, other in enumerate(order) if other._record is layer._record)]
    for other in below:
        if other.is_group() or other.clipping or other.kind not in DRAWN_LAYER_KINDS or not other.is_visible():
            continue
        reach = _pad(tuple(other.bbox), fx_pad(other), psd)
        x0, y0, x1, y1 = max(reach[0], box[0]), max(reach[1], box[1]), min(reach[2], box[2]), min(reach[3], box[3])
        if x1 <= x0 or y1 <= y0 or not ink[y0 - oy:y1 - oy, x0 - ox:x1 - ox].any():
            continue
        alone = other.composite(viewport=(x0, y0, x1, y1), apply_icc=False)
        if alone is None:
            continue
        drawn = np.asarray(alone.convert("RGBA"))[..., 3] > INK_ALPHA * 255
        over = int((ink[y0 - oy:y1 - oy, x0 - ox:x1 - ox] & drawn).sum())
        if over and not (was[y0 - oy:y1 - oy, x0 - ox:x1 - ox] & drawn).any():
            return f"it would print over {other.name!r} below it ({over} px)"
    return None


def ink_beyond(psd: PSDImage, layer, before: np.ndarray) -> Optional[str]:
    """
    What an ability's new ink past the old text prints on, for a note (the owner bakes it; never silently): its pixels
    past INK_ALPHA more than BEYOND[0] px outside the old ink's box over a backdrop (the layers drawn below it) whose
    luminance lies off the backdrop's median under that box by BEYOND[1] levels and by BEYOND[2] more than 95% of the
    box does (text on busy art: Kingchin's added line on its art told nothing), when BEYOND[3] px or more (review
    2026-10-24: Deep Focus' added "Die." on its flourish, Holy Card's last line on its panel's torn border, Autopsy
    Report's off its box; the old ink touches those layers elsewhere, so inks_over passed them; 18 of 503 cards given a
    line), else inks_over's finding (a layer below the old ink never touched).

    Args:
        psd: The card, the text set and rasterized
        layer: The Ability layer
        before: _on_canvas() of it before the edit

    Returns:
        Why, or None
    """
    pad, levels, margin, least = BEYOND
    win = _ink_box(psd, layer, before, pad)
    if win is None:
        return None
    ox, oy, wx, wy = win
    after, near = _on_canvas(layer, (psd.height, psd.width), win), before[oy:wy, ox:wx]
    was = near[..., 3] > INK_ALPHA * 255
    ink = (after[..., 3] > INK_ALPHA * 255) & _px_differ(after, near)
    if not was.any() or not ink.any():
        return None
    ys, xs = was.nonzero()
    old = (max(int(xs.min()) + ox - pad, 0), max(int(ys.min()) + oy - pad, 0), int(xs.max()) + ox + 1 + pad,
           int(ys.max()) + oy + 1 + pad)
    fresh = ink.copy()
    fresh[old[1] - oy:old[3] - oy, old[0] - ox:old[2] - ox] = False
    if not fresh.any():
        return inks_over(psd, layer, before)
    ys, xs = fresh.nonzero()
    x0, y0 = min(old[0], int(xs.min()) + ox), min(old[1], int(ys.min()) + oy)
    x1, y1 = min(max(old[2], int(xs.max()) + 1 + ox), psd.width), min(max(old[3], int(ys.max()) + 1 + oy), psd.height)
    order = list(psd.descendants())
    below = {id(other._record) for other in order[:next(i for i, other in enumerate(order) if other._record is layer._record)]}
    backdrop = _flat(psd.composite(viewport=(x0, y0, x1, y1), ignore_preview=True, apply_icc=False,
                                   layer_filter=lambda other: other.is_visible() and (other.is_group()
                                                                                      or id(other._record) in below)))
    luma = backdrop[..., :3].astype(np.float32).mean(axis=2)
    under = luma[old[1] - y0:old[3] - y0, old[0] - x0:old[2] - x0]
    median = float(np.median(under))
    bar = max(levels, float(np.percentile(np.abs(under - median), 95)) + margin)
    detail = fresh[y0 - oy:y1 - oy, x0 - ox:x1 - ox] & (np.abs(luma - median) > bar)
    count = int(detail.sum())
    if count < least:
        return inks_over(psd, layer, before)
    best, name = 0, None
    for other in order:
        if id(other._record) not in below or other.is_group() or other.kind not in DRAWN_LAYER_KINDS \
                or not other.is_visible():
            continue
        alone = other.composite(viewport=(x0, y0, x1, y1), apply_icc=False)
        hits = 0 if alone is None else int((detail & (np.asarray(alone.convert("RGBA"))[..., 3] > INK_ALPHA * 255)).sum())
        if hits >= best and hits:               # the topmost of the most
            best, name = hits, other.name
    return f"the new text prints past the old text onto the design below it ({count} px" + \
        (f", {name!r})" if name else ")")


def on_panel_design(psd: PSDImage, layer, box: tuple) -> Optional[str]:
    """
    An added Ability's ink on its text panel's own design (the pixel layer whose box is box: Deep Focus' flourishes;
    review 2026-10-25, the box being the text's own, ink_beyond looks only past it): the panel's pixels whose luminance
    lies off its median by BEYOND[1] levels and by BEYOND[2] more than 95% of it does, under BEYOND[3] px or more of ink.
    """
    panel = next((other for other in psd.descendants() if other.kind == "pixel" and other.is_visible()
                  and tuple(other.bbox) == tuple(box)), None)
    x0, y0, x1, y1 = clip(tuple(box), psd.width, psd.height)
    if panel is None or x1 <= x0 or y1 <= y0:
        return None
    try:                                        # its own pixels (a clipping base's art is not its design)
        pixels = panel.numpy()[y0 - panel.bbox[1]:y1 - panel.bbox[1], x0 - panel.bbox[0]:x1 - panel.bbox[0]] * 255
    except Exception:
        return None
    if pixels.ndim != 3 or pixels.shape[2] < 4:
        return None
    solid = pixels[..., 3] > INK_ALPHA * 255
    if not solid.any():
        return None
    luma = pixels[..., :3].mean(axis=2)
    off = np.abs(luma - float(np.median(luma[solid])))
    bar = max(BEYOND[1], float(np.percentile(off[solid], 95)) + BEYOND[2])
    ink = _on_canvas(layer, (psd.height, psd.width))[y0:y1, x0:x1, 3] > INK_ALPHA * 255
    count = int((ink & solid & (off > bar)).sum())
    return f"the new text prints on its panel's design ({count} px, {panel.name!r})" if count >= BEYOND[3] else None


def below_box(keep, lines) -> bool:
    """
    Whether lines (a new layout's) draw more baselines with letters below the text box's bottom than keep's layout did:
    Photoshop hides such a line (overset: Croissant, 2026-11-04), though the fork draws it. Point text has no box.

    Args:
        keep: The layer's capture before the change (its model's box, its layout's lines)
        lines: The new layout's lines (the engine's report)

    Returns:
        Whether a line more falls below the box
    """
    from psd_tools.composite.text.layout import frame
    box = frame(keep.model)
    if box is None:
        return False
    below = lambda ls: sum(1 for line in ls if line.drawn and line.text.strip() and line.y > box[1] + box[3] + 1e-6)
    return below(lines) > below(keep.layout.lines)


def hold_lines(layer, fonts) -> int:
    """
    A paragraph box with drawn lines below its bottom grown down to hold them (set_text_box, its pixels kept): Photoshop
    hides such a line when it lays the text out again (below_box). Its top, sides and transform stay; the bottom goes
    BOX_DESCENT em of its largest size under the last baseline. Not point text, a rotated or offset box, nor one with a
    line the engine left undrawn.

    Returns:
        px the box grew (0: not)
    """
    from psd_tools.composite.text.layout import frame, layout
    from psd_tools.composite.text.model import TextModel
    model = TextModel.from_layer(layer)
    box = frame(model)
    xx, xy, yx, yy, tx, ty = model.transform
    if box is None or abs(xy) > 1e-6 or abs(yx) > 1e-6 or any(abs(v) > 1e-6 for v in model.box[:2]):
        return 0
    lines = [line for line in layout(model, fonts).lines if line.text.strip()]
    past = [line for line in lines if line.y > box[1] + box[3] + 1e-6]
    if not past or not all(line.drawn for line in lines):
        return 0
    bottom = past[-1].y + BOX_DESCENT * max(run.style.size for run in model.runs if run.end > run.start)
    set_text_box(layer, (tx, ty, tx + xx * box[2], ty + yy * bottom))     # at its origin: nothing moves
    return max(1, round(yy * (bottom - box[3])))


def fit_below(psd: PSDImage, layer, before: np.ndarray, fonts, draw, off_before: bool, check) -> tuple:
    """
    An Ability whose new text prints past its old text onto the design below it (ink_beyond: Autopsy Report's third
    line under its panel) or off the canvas (Soul Sanctum's wrapped line) set smaller until it doesn't, no lower than
    the old text, as the website's preview draws it (mockup.js drawText): by FIT_STEP, down to FIT_MIN of its size,
    every run's FontSize, Leading and BaselineShift scaled alike (badges keep their size and middle against the
    letters). No size fits, or a layer with tabs (its stops don't shrink: Michael's Blessing's arrow column would
    break): left as it was, ink_beyond tells it.

    Args:
        psd: The card, the text set and rasterized
        layer: The Ability layer
        before: _on_canvas() of it before the edit
        fonts: font_map()
        draw: Rasterizes the layer as set_text_layer did (its splice); returns the engine's report
        off_before: It inked off the canvas before the edit (then that is no reason)
        check: ink_beyond(psd, layer, before) (set_text_layer's, which keeps the last verdict)

    Returns:
        (scale, report): (1.0, None) when it was left as it was
    """
    off = not off_before and _ink_off_canvas(layer, psd.width, psd.height)
    if "\t" in layer.text or not off and check() is None:
        return 1.0, None
    from psd_tools.api import text_edit
    text = layer.text
    units = text_edit.utf16_prefix(text)
    resources = layer.resource_dict
    try:
        normal = resources["StyleSheetSet"][int(getattr(resources["TheNormalStyleSheet"], "value",
                                                        resources["TheNormalStyleSheet"]))]["StyleSheetData"]
    except (KeyError, IndexError, TypeError, ValueError):
        normal = {}
    runs = []
    for u0, u1, data in text_edit._spans(text_edit._section(layer, "StyleRun"), units[-1]):
        sheet = data["StyleSheet"]["StyleSheetData"]
        keys = {key: float(getattr(sheet.get(key, normal.get(key)), "value", sheet.get(key, normal.get(key))))
                for key in ("FontSize", "Leading", "BaselineShift") if sheet.get(key, normal.get(key)) is not None}
        runs.append((text_edit._str_index(units, u0), text_edit._str_index(units, u1), keys))

    def scaled(scale: float) -> None:
        for a, b, keys in runs:
            if b > a and keys:
                layer.set_span_style(a, b, **{key: value if scale == 1 else round(value * scale, 2)
                                              for key, value in keys.items()})
    rows = (before[..., 3] > INK_ALPHA * 255).any(axis=1).nonzero()[0]
    floor = int(rows.max()) + 1 + BEYOND[0] if len(rows) else psd.height

    def fits() -> bool:                         # no lower than the old text (ink_beyond lets a few px onto the design)
        ink = (_on_canvas(layer, (psd.height, psd.width))[..., 3] > INK_ALPHA * 255).any(axis=1).nonzero()[0]
        return (not len(ink) or int(ink.max()) + 1 <= floor) and check() is None \
            and (off_before or not _ink_off_canvas(layer, psd.width, psd.height))
    for scale in (FIT_STEP ** k for k in range(1, 99)):    # the largest that fits: a box's first line rises as it
        if scale < FIT_MIN:                                 # shrinks, so a smaller size can print past the top
            break
        scaled(scale)
        report = layer.rasterize(fonts=fonts)
        if fits():
            return scale, report
    scaled(1.0)
    return 1.0, draw()


def _title_lines(geometry: dict) -> list:
    """A title's drawn lines with letters (text_geometry): their baselines' start y."""
    return [line["baseline"][1] for line in geometry["lines"] if line["drawn"] and line["text"].strip()]


def typed_icon_fit(layer, typed: list, sized: set, fonts) -> Optional[tuple]:
    """
    Whether each typed badge or token (typed: its text ranges) prints as an icon of the text: one in sized (set at the
    text's own style: no badge or token on the card, or a title) within ICON_HEIGHT of the tallest letter's ink on its
    line (on the layer, when its line has none), by its smallest glyph (a layered token's icon). Fonts whose letters
    are far from their em: Holy Blade's ChiQuel drew a [fire] as a speck (review 2026-10-24).

    Returns:
        (code, why) naming the first that does not, or None
    """
    from psd_tools.composite.text.geometry import text_geometry
    TextModel, layout = _fx_engine()[:2]
    lay = layout(TextModel.from_layer(layer), fonts)
    geometry = text_geometry(layer, fonts)
    text = geometry["text"]
    letters = [g for g in geometry["glyphs"] if g["ink"] and g["ch"].isalpha()
               and not g["font"].lower().startswith(ICON_FONT_PREFIX)]
    for a, b in typed:
        piece = text[a:b]
        if not any(a <= g.cluster < b for g in lay.glyphs):
            continue
        boxes = [g["ink"] for g in geometry["glyphs"] if g["ink"] and a <= g["i"] < b]
        if (a, b) not in sized or not boxes or not letters:
            continue
        line = next(g["line"] for g in geometry["glyphs"] if a <= g["i"] < b)
        near = [g for g in letters if g["line"] == line] or letters
        tall = max(g["ink"][3] - g["ink"][1] for g in near)
        ratio = min(box[3] - box[1] for box in boxes) / tall if tall > 0 else 1.0    # a :sticker's icon, not its rim
        if not ICON_HEIGHT[0] <= ratio <= ICON_HEIGHT[1]:
            return "unsupported-card", (f"{piece} at the text's font size would be {ratio:.2f}x the height of its "
                                        f"letters (their font draws them far from its size)")
    return None


def unstretch(layer, spans: list) -> None:
    """
    Typed badges and tokens drawn as the font draws them: HorizontalScale set so that x scale x HorizontalScale over
    y scale x VerticalScale is 1 under the layer's transform, as designers set theirs (review 2026-10-24: 678 of 682
    icons at 1.0; a [fire] typed at Monster Meat's body style, 2.64, printed stretched).
    """
    if not spans:
        return
    from psd_tools.composite.text.model import TextModel
    xx, xy, yx, yy = TextModel.from_layer(layer).transform[:4]
    cx, cy = hypot(xx, xy), hypot(yx, yy)
    if not cx or not cy:
        return
    styles = _char_styles(layer)[2]
    for a, b in spans:
        style = styles[a] or {}
        hs, vs = float(style.get("HorizontalScale", 1.0)), float(style.get("VerticalScale", 1.0))
        if vs and abs(cx * hs / (cy * vs) - 1) > ASPECT_TOL:
            layer.set_span_style(a, b, HorizontalScale=vs * cy / cx)


def font_map(fonts_dir: str, layer=None):
    """
    The engine's FontMap over MDW/Fonts; a face no file holds is drawn in the fork's stand-in (told: OK_FONTS). With a
    layer, its named faces are drawn in the faces its pixels show (shown_faces: Dull Knife's title names OMORI_GAME2,
    Photoshop drew OMORI_GAME), for redrawing that layer alone (R3, owner 2026-10-26).
    """
    aliases = shown_faces(layer, fonts_dir) if layer is not None else {}
    return _fonts(fonts_dir, aliases)


def fonts_in() -> None:
    """Returns once the fonts are in (FONTS_WAIT's wait; none: at once)."""
    if FONTS_WAIT is not None:
        FONTS_WAIT()


def _own_dirs(fonts_dir: str) -> list:
    """[fonts_dir], or [] when FONT_DIRS_ENV starts with it: the FontMap reads that too (one walk, the same index)."""
    return [] if environ.get(FONT_DIRS_ENV, "").split(pathsep)[0] == fonts_dir else [fonts_dir]


def _fonts(fonts_dir: str, aliases: Optional[dict] = None):
    """Every FontMap over fonts_dir, made once the fonts are in (fonts_in). Without aliases: one per process while the
    fonts dirs' files stand (name, size, mtime)."""
    from psd_tools.composite.text import FontMap
    fonts_in()
    if aliases:
        return FontMap(dirs=_own_dirs(fonts_dir), policy="substitute", aliases=aliases)
    env = environ.get(FONT_DIRS_ENV, "")
    return _shared_fonts(fonts_dir, env, _files_state(fonts_dir, *env.split(pathsep)))


def _files_state(*dirs: str) -> tuple:
    out = []
    for folder in dict.fromkeys(filter(None, dirs)):
        try:
            with scandir(folder) as found:
                out.append(tuple(sorted((e.name, s.st_size, s.st_mtime_ns) for e in found for s in [e.stat()])))
        except OSError:
            out.append(None)
    return tuple(out)


@lru_cache(maxsize=4)
def _shared_fonts(fonts_dir: str, env: str, state: tuple):
    from psd_tools.composite.text import FontMap
    return FontMap(dirs=_own_dirs(fonts_dir), policy="substitute")


FACE_IOU_MIN = 0.8          # a sibling face redraws a layer's saved pixels this alike to be the face they show (the four
#   Omori titles: 0.98-1.0 in the other face, 0.12-0.15 in the named one)
_SHOWN: dict = {}           # shown_faces' answers: (layer id, text, bbox, faces, fonts dir, pixels) -> {named: shown}


@lru_cache(maxsize=4)
def _font_faces(fonts_dir: str) -> tuple:
    """The PostScript names MDW/Fonts holds (the FontMap's exact index)."""
    return tuple(sorted(_fonts(fonts_dir)._indexes()[0].exact))


def _face_stem(name: str) -> str:
    """A face's name without separators, case or a trailing number: OMORI_GAME2 and OMORI_GAME are one stem."""
    return sub(r"\d+$", "", sub(r"[^a-z0-9]", "", name.lower()))


def _face_iou(layer, fonts, shape: tuple) -> float:
    """The IoU of the layer's whole redraw in fonts and its saved pixels, ink past INK_ALPHA."""
    from psd_tools.composite.text import render
    result = render(layer, fonts=fonts)
    redraw = np.zeros(shape, bool)
    x0, y0, x1, y1 = clip(result.bbox, shape[1], shape[0])
    if x1 > x0 and y1 > y0:
        bx, by = result.bbox[:2]
        redraw[y0:y1, x0:x1] = result.rgba[y0 - by:y1 - by, x0 - bx:x1 - bx, 3] > INK_ALPHA
    saved = _on_canvas(layer, shape)[..., 3] > INK_ALPHA * 255
    union = int((redraw | saved).sum())
    return int((redraw & saved).sum()) / union if union else 1.0


def _layer_faces(layer) -> list:
    from psd_tools.composite.text.model import TextModel
    return sorted({run.style.font for run in TextModel.from_layer(layer).runs if run.style.font})


def shown_faces(layer, fonts_dir: str, shape: Optional[tuple] = None) -> dict:
    """
    {named face: face its saved pixels show}: when the layer's redraw in the faces it names is unlike its pixels
    (drawn_in_font) and a face of the same stem in MDW/Fonts draws them (FACE_IOU_MIN). Photoshop drew the four Omori
    titles in OMORI_GAME's sibling. {} otherwise; only faces with a sibling are tried, so most layers cost nothing.
    """
    from psd_tools.composite.text import MissingFontError
    if not _is_text_layer(layer) or not fonts_dir:
        return {}
    try:
        named = _layer_faces(layer)
    except Exception:                           # noqa: BLE001 - an unreadable layer is drawn as it names
        return {}
    faces = _font_faces(fonts_dir)
    siblings = {name: [face for face in faces if face != name and _face_stem(face) == _face_stem(name)] for name in named}
    siblings = {name: found for name, found in siblings.items() if found}
    if not siblings:
        return {}
    shape = shape or (layer._psd.height, layer._psd.width)
    alpha = _alpha(layer)
    key = (layer.layer_id, layer.text, tuple(layer.bbox), tuple(named), fonts_dir,
           blake2b(alpha.tobytes() if alpha is not None else b"", digest_size=8).hexdigest())
    if key not in _SHOWN:
        out = {}
        try:
            if _face_iou(layer, _fonts(fonts_dir), shape) < FONT_IOU_MIN:
                tried = [(_face_iou(layer, _fonts(fonts_dir, {name: face}), shape),
                          name, face) for name, found in siblings.items() for face in found]
                best = max(tried)
                out = {best[1]: best[2]} if best[0] >= FACE_IOU_MIN else {}
        except MissingFontError:
            out = {}
        _SHOWN[key] = out
    return dict(_SHOWN[key])


def set_faces(layer, shown: dict, fonts_dir: str) -> list:
    """
    The layer's runs in a named face set in the face its pixels show (shown_faces), the face added to its FontSets (the
    layer's and Txt2's), so the PSD names what it shows (owner 2026-10-26). Returns the faces added.
    """
    index = _fonts(fonts_dir)._indexes()[0].exact
    added = []
    names = lambda: [str(font["Name"].value) for font in layer.resource_dict["FontSet"]]
    run_fonts = lambda: [names()[int(run["StyleSheet"]["StyleSheetData"]["Font"])]
                         if "Font" in run["StyleSheet"]["StyleSheetData"] else None
                         for run in layer.engine_dict["StyleRun"]["RunArray"]]
    for old, new in shown.items():
        if old not in run_fonts():
            continue
        with open(index[new][0], "rb") as fh:
            outlines = 0 if fh.read(4) == b"OTTO" else 1      # CFF or TrueType (add_font's font_type)
        if new not in names():
            added.append(new)
        layer.add_font(new, font_type=outlines)
        for _ in range(len(run_fonts())):       # every run (the last one, past the text, too); runs merge as they go
            if old not in run_fonts():
                break
            layer.set_run_style(run_fonts().index(old), Font=new)
    return added


def drawn_in_font(layer, fonts, shape: tuple) -> Optional[str]:
    """
    Whether the layer's saved pixels are drawn in the fonts it names: its whole redraw by the engine against them, ink
    past INK_ALPHA, IoU at least FONT_IOU_MIN. Dull Knife's title names OMORI_GAME2 but Photoshop drew it in OMORI_GAME:
    an edit reprinted every letter in the other typeface, every gate passing (review 2026-10-24). Raises
    MissingFontError.

    Returns:
        Why not, or None
    """
    iou = _face_iou(layer, fonts, shape)
    if iou >= FONT_IOU_MIN:
        return None
    return f"its saved pixels are not drawn in the font it names ({', '.join(_layer_faces(layer))}; {iou:.0%} alike)"


def _stroke_reach(layer) -> int:
    """Pixels the outside strokes of the layer and its groups draw past its ink (fx_pad's strokes, no shadow)."""
    total = 0
    for owner in (layer, *_ancestors(layer)):
        for effect in owner.effects:
            if effect.enabled and type(effect).__name__ == "Stroke":
                position = str(getattr(effect, "position", ""))
                if "INSIDE" not in position:
                    total += int(effect.size if "OUTSIDE" in position else effect.size / 2 + 0.5)
    return total


def back_on(psd: PSDImage, layer) -> int:
    """
    A title layer whose new ink runs off the card's side, moved across until it is on (R4): its TySh position and its
    raster together (Photoshop keeps a type layer's place in TySh alone: a point text's Txt2 frame holds none). Not when
    its ink is wider than the card or it runs off the top or bottom, nor a layer something is clipped to or that clips.

    Returns:
        The px moved (0: not moved)
    """
    alpha = _alpha(layer)
    if alpha is None or not alpha.size or layer.clipping or _clip_chain(layer):
        return 0
    ink = np.nonzero((alpha > INK_ALPHA).any(axis=0))[0]
    rows = np.nonzero((alpha > INK_ALPHA).any(axis=1))[0]
    if not ink.size:
        return 0
    x0, x1 = layer.bbox[0] + int(ink[0]), layer.bbox[0] + int(ink[len(ink) - 1]) + 1
    y0, y1 = layer.bbox[1] + int(rows[0]), layer.bbox[1] + int(rows[len(rows) - 1]) + 1
    if x1 - x0 > psd.width or y0 < 0 or y1 > psd.height or (x0 >= 0 and x1 <= psd.width):
        return 0
    edge = _stroke_reach(layer)                 # its outline on the card too, when it fits
    edge = edge if x1 - x0 + 2 * edge <= psd.width else 0
    dx = edge - x0 if x0 < edge else psd.width - edge - x1
    nudge(layer, dx, 0)
    return dx


def lift(psd: PSDImage, layer, before: np.ndarray) -> int:
    """
    An Ability whose new ink runs off the card's bottom (no size fits it: fit_below) moved up until its last line ends
    where the old text's did: back_on's rule for the bottom (a box's Txt2 frame is relative to the transform). Not past
    the top, nor a layer that clips, is clipped to, or has a mask.

    Returns:
        The px moved (0: not moved)
    """
    alpha = _alpha(layer)
    if alpha is None or not alpha.size or layer.clipping or _clip_chain(layer) or layer.has_mask() \
            or layer.has_vector_mask():
        return 0
    rows = np.nonzero((alpha > INK_ALPHA).any(axis=1))[0]
    old = np.nonzero((before[..., 3] > INK_ALPHA * 255).any(axis=1))[0]
    if not rows.size or not old.size:
        return 0
    y0, y1 = layer.bbox[1] + int(rows[0]), layer.bbox[1] + int(rows[len(rows) - 1]) + 1
    dy = y1 - int(old[len(old) - 1]) - 1
    if y1 <= psd.height or dy <= 0 or y0 < dy:
        return 0
    nudge(layer, 0, -dy)
    return dy


def nudge(layer, dx: int, dy: int) -> None:
    """A type layer moved: its TySh position and its raster together (Photoshop keeps its place in TySh alone)."""
    xx, xy, yx, yy, tx, ty = layer._data.transform
    layer._data.transform = (xx, xy, yx, yy, tx + dx, ty + dy)
    if dx:
        layer.left = layer.left + dx
    if dy:
        layer.top = layer.top + dy


def set_text_layer(psd: PSDImage, which: str, text: str, fonts_dir: str, chart: Optional[dict] = None,
                   card: str = "", parser: Optional[PSDParser] = None,
                   original: Optional[PSDImage] = None, layer=None, copy: bool = False,
                   area: Optional[tuple] = None, anchor: int = 0, up: int = 0) -> Edit:
    """
    Replace the name or the ability text, keeping the runs, and rasterize it with the text engine: only the changed
    glyphs (SPLICE_UNIT), Photoshop's raster staying on the rest.

    Args:
        psd: The open card (edited in place)
        which: "name" (name_layer) or "ability" (ability_layer)
        text: The new text; newlines become paragraph breaks but where the layer had a forced line break (keep_breaks);
            the layer's blank first and last paragraphs stay (keep_ends)
        fonts_dir: Directory of the card fonts (MDW/Fonts)
        chart: status_fills(); ability status words take their swatch fill and the styled ones their FX layer
        card: The card's name (file stem), for name_layer
        parser: offline_parser(): its icon vocabulary says which typed [tokens] the icon font draws (icon_text) and its
            types which image icons the parser reads into the ability (image_icons); None: every [token], and a layer
            without the icon font counts as one with image icons
        original: The card before any of the bake's edits (apply_jobs' old), where a typed :sticker or :tint reads the
            other text layer's colours (variant_fill): the Ability's edit, made first, would give a title's typed
            [dark:sticker] the colour of the one it typed (review 2026-10-24: Salt's purple on its white title); None: psd
        layer: The text layer to set (set_parts' part); None: name_layer's or ability_layer's
        copy: The layer is a copy of a part (set_parts): no FX layers or overlay words of its own, its
            ink past the old text not told (the Ability's is)
        area: An added Ability's box (add_ability, canvas px): the text fits it (fit_below), its ink past it is told;
            nothing was printed before (its placeholder is no old text)
        anchor: px an earlier bake moved this title to keep it on the card (R4, its verdict): moved back first, so a
            shorter title sits where the card maker put it and a move is always counted from there
        up: A copy's: px its Ability moved up (lift), which it follows

    Returns:
        Edit with the engine's report. Content problems bake best-effort and are told in Edit.notes as "<code>: <field>:
        <why>" (owner 2026-10-24: the editor previewed it): text that won't fit (overflow: off the canvas, a title on
        more lines or lower), typed badges and tokens (a tint, letters, a typed icon
        far from its letters' height: typed_icon_fit), font stand-ins, boxes and saved pixels in another font (drawn_in_font); an FX layer whose style
        can't follow its words is deleted, its words printing plain (styled words given the chart's layer by new_fx).
        Raises Refused for what it cannot edit: unsupported-card (hand-made status word layers over the ability, FX
        layers that are no pixel layers, two FX layers one name, FX layers without a chart), styled-word (FX words with
        no pixels to cut)

    The text is set one changed part at a time (splice_text), a deletion cutting whole lines or words (deletion_start),
    so what the edit keeps keeps its runs; an Ability's new ink past its old text on the design below it is told in
    the notes (ink_beyond). The engine gives
    inserted text the style of the run before it, so an edit after a colored status word or a badge would take its
    color or icon font: an ability's changed parts take the fill of the text they replace (edit_styles; status words
    give none), words set in the icon font outside what the font draws (icon_text) that text's whole style; every badge
    or token the edit types is set in the icon font at the card's own badge or token style (icon_style; owner
    2026-10-23), on a card with none at the text's own style (ICON_FONT added, Edit.fonts), with ligatures on and a
    LAYERED one (:sticker, :no) at tracking 0 (tracking split its glyphs apart), the layer's own spellings
    of the icons it kept staying (keep_tokens: "[dark:sticker]", where the website sends "[dark]"); then every
    capitalised status word in the text takes its swatch (status_word_spans: a lowercase one is prose, owner 2026-11-07;
    the chart is the authority, an off-swatch word is normalized). The "<Ability> FX:
    ..." layers are then made again as Effect Layers.jsx's Update all makes them (BUG 2, _rebuild_fx): each cut in place
    from the new raster over its phrases' new places, its record, place and own style kept (owner q6), but one on a
    chart status word takes the chart's style (owner 2026-11-07: restyle_fx); a phrase gone from the text leaves the
    name, a layer with none left goes; a "styled" or "own" layer
    stays on its own words (follow_fx: Goofy's "1#1" renumbered, or deleted, told, when its name would reach another
    "1" or when the edit changes its words, "+2 DEF" made "+3 DEF"), a
    "status" layer on its status's own words (a numbered one on its own word, renumbered: Standstill's "Frozen In
    Time#2"; never the "Frozen" of "Frozen In Time"). A styled status word no FX layer holds whole gets one (idea 14, new_fx: "Blazing", "Frozen#2"
    when only that one; one a layer covers in part none, told), its glyph pixels cut under a
    copy of the chart layer's effect blocks, directly above the layer's top-level group as Effect Layers.jsx's Add
    places it; text the edit typed into a "styled" or "own" layer's own words, or a typed copy joining it ("Allies get
    +2 DEF." on Palace of Shadow), takes that phrase's colour (the first drawn run, as the JSX's overlay conversion; a
    stroke-only layer shows it: the copy printed in the body's grey, review 2026-10-23), a status word among them keeping
    the chart's (a layer whose words are in several colours: the first, told). An FX layer whose phrase the ability
    lacks is deleted. Under an overlay on the Ability or its group (one_colour) every badge and :sticker,
    typed or kept, takes the icon font's Stylistic Alternates (set_alternates; owner 2026-10-25). Told too: another typed
    badge or token under the Ability's overlay (any opacity) or blending (Pin Light) (_tinted), a [type] token
    on a card with image icons in its gaps (image_icons), letters the icon font draws plainly (typed, or left by
    deleting: "(GCD)"), a typed character its font has no glyph for (no_glyph: a box), a timing badge typed in a title
    (plain words, as the website shows it). Typed badges and tokens are unstretched (unstretch). An icon character
    typed ("★") is set in the icon font as a token. An FX layer in no chart status's style (Fire Staff's Smoldering
    stroke) is rebuilt in it since 2026-10-23 (owner). styled-word now means only that an FX layer's words have no pixels.
    """
    text, projected = straight(text), None      # the bake writes ' (owner, 2026-10-25): the layer's kept ones too
    if layer is None and which == "ability":
        layer = ability_layer(psd)
        projected = from_reading(psd, layer, text, parser)
    elif layer is None:
        layer = name_layer(psd, card)
    follow_kerning(psd, layer)                  # drawn kerned as Photoshop shows it (B12)
    old_text, old_bbox, story = layer.text, tuple(layer.bbox), story_index(layer)
    gamma = _drew_gamma(psd, layer)             # as the edited layer is drawn (text_kwargs): its kept glyphs stay
    flags = []

    def flag(code: str, why: str) -> None:      # a content problem: baked best-effort and told (owner 2026-10-24)
        why = why.removesuffix("; edit in Photoshop")
        flags.append(f"{code}: {why if why.startswith(which) else f'{which}: {why}'}")
    text = text.replace("\r\n", "\n").rstrip("\n").replace("\n", "\r")     # a final \n would add an empty paragraph
    text = projected if projected is not None else keep_ends(old_text, keep_breaks(old_text, text, which))
    if which == "ability":                      # "[dark]" on the website, "[dark:sticker]" on the card
        text = keep_tokens(layer, text, getattr(parser, "icons", None))
    start, end = changed_range(old_text, text)
    end = min(end, len(text))
    start = min(start, end)                     # identical text: the engine's range sits past the end
    if start == end:                            # a deletion: the window that keeps the kept characters' runs
        start = end = deletion_start(old_text, text, start)
    words, styled, old_fx, plans, targets, kept, engine = [], [], [], [], [], [], None
    vocab, faces = getattr(parser, "icons", None), []
    icons_too = which == "ability" or vocab is not None     # a title's tokens only with the vocabulary (owner 2026-10-24)
    if which == "ability":
        images = icon_font(layer) is None if parser is None else bool(image_icons(psd, parser))
        if TOKEN.search(text) and images:
            flag("unsupported-card", "[type] tokens on a card with image icons, set in the icon font")
    if icons_too:
        odd = Counter(unlettered_icons(text, parser)) - Counter(unlettered_icons(old_text, parser)) \
            if which == "ability" else None     # the website shows a title's "[fire:]" as its letters, as the card does
        if odd:                                 # typed, or "[fire:no]" made "[fire:]" by deleting letters
            flag("font", f"{', '.join(odd)}: the website shows an icon, but the icon font has no such spelling "
                         f"and prints its letters")
    drawn = icon_text(text, vocab) if icons_too else []    # what the icon font draws (owner 2026-10-23: a typed one
    looks = {kind: icon_style(layer, kind, vocab) for kind in (True, False)}   # is set in it); badge, token
    tints = [variant_fills(each, vocab) for each in
             (layer, _other_text(psd if original is None else original, which, card))] \
        if any(text[a] == "[" and token_variant(text[a:b]) in TEXT_COLOURED for a, b in drawn) else []
    if which == "ability":
        old_fx = fx_layers(psd, layer)
        if not copy and overlay_words(psd):    # hand-made word layers over it (R5 tells them)
            raise Refused("unsupported-card", "status word layers over the ability; edit in Photoshop")
        if not chart:
            if old_fx:
                raise Refused("unsupported-card", "status word layers over the ability")
        else:
            odd = [fx.name for fx in old_fx if not fx_kind(fx, layer, chart)]
            if odd:
                raise Refused("unsupported-card", f"FX layers that are no pixel layers: {', '.join(odd)}; edit in Photoshop")
            cut = old_text[start:start + len(old_text) - len(text)] if start == end else ""
            lo, hi = _word_bounds(text, start, end)     # a deletion widens only into a word it cut (Gasoline's kept "If"
            start = start if cut and not LETTER.match(cut[0]) else lo           # after a deleted line took its colour)
            end = end if cut and not LETTER.match(cut[-1]) else hi
            words = status_word_spans(text, chart, drawn)
            styled = [span for span in words if span[2].styled]
            plans = plan_fx(layer, old_fx, chart)
            kept = kept_chars(old_text, text)
            for plan in plans:                  # a layer whose style can't follow goes: its words print without it
                lost = [fx_phrase for fx_phrase, was in plan.found if not was]
                try:
                    if lost:                    # what they cover can't be told ("Frozen #2": no JSX phrase)
                        raise Refused("styled-word", f"its phrases are not in the ability: {', '.join(lost)}")
                    targets.append(follow_fx(plan, text, kept, words))
                except Refused as refused:
                    why = refused.why.removeprefix(f"{plan.layer.name}: ").removesuffix("; edit in Photoshop")
                    targets.append([(phrase, None, [], [], why) for phrase, _ in plan.found])
            for plan, target in zip(plans, targets):
                joined = [text[a:b] for _, _, now, anchored, _ in target for a, b in now if (a, b) not in anchored]
                if joined and plan.mixed:       # the typed copy takes the first drawn run's colour (plan.fill)
                    flag("styled-word", f"{plan.layer.name}: its words are in more than one colour; the typed "
                                        f"{joined[0]!r} takes the first")
                inside = [text[a:b] for _, _, now, _, _ in target for a, b in now if a < end and start < b]
                if inside and plan.mixed:       # the fill below paints every letter of it one colour
                    flag("styled-word", f"{plan.layer.name}: its words are in more than one colour, which the edit in "
                                        f"{inside[0]!r} makes one")
            if styled or old_fx:
                engine = _fx_engine()
    parts = edit_parts(old_text, text, start, end)
    styles = edit_styles(layer, parts, chart, text_moves(old_text, text)) if which == "ability" else []
    restyle = title_styles(layer, parts) if which == "name" else []
    changed = [(b0, b1) for _, _, b0, b1, same in parts if not same and b1 > b0]
    typed = [(a, b) for a, b in drawn if any(a < b1 and b0 < b for b0, b1 in changed)]     # badges and tokens it types
    badged = [(a, b) for a, b in typed if which == "name" and text[a] == "("]
    if badged:                                  # the website shows plain words there: so they print
        flag("unsupported-card", f"{', '.join(text[a:b] for a, b in badged)}: a timing badge typed in the title, "
                                 f"printed as plain words (as the website shows it)")
        drawn = [span for span in drawn if span not in badged]
        typed = [span for span in typed if span not in badged]
    shaped = alternate_spans(text, typed) if which == "ability" and one_colour(layer) else []
    recoloured = [span for span in typed if span not in shaped]     # a one-colour badge or sticker is meant so
    tint = _tinted(layer) if recoloured else None
    if tint:
        thing = "badge" if all(text[a] == "(" for a, b in recoloured) else "icon"
        flag("unsupported-card", f"{', '.join(text[a:b] for a, b in recoloured)}: the "
                                 + (f"Ability's look ({tint}) changes the typed {thing}'s colours" if which == "ability"
                                    else f"title's look ({tint}) recolours the typed {thing}"))
    fonts = font_map(fonts_dir)
    alien = drawn_in_font(layer, fonts, (psd.height, psd.width))     # the edit reprints its letters in the named font
    shown = shown_faces(layer, fonts_dir, (psd.height, psd.width)) if alien else {}
    if shown:                                   # in the face its pixels show, which the PSD names from now on (R3)
        faces = list(dict.fromkeys(faces + set_faces(layer, shown, fonts_dir)))
    elif alien:
        flag("font", f"{which}: {alien}")
    from psd_tools.composite.text import capture  # Photoshop's raster stays on the lines the edit leaves alone
    before = capture(layer, fonts)
    if which == "name":                         # the title stays on its lines (review 2026-10-24: Basic Shield's box
        from psd_tools.composite.text.geometry import text_geometry     # wrapped a typed [fire] onto the art)
        lines_before = _title_lines(text_geometry(layer, fonts))
    icons_before, blank = _char_styles(layer)[1], " \t\r\x03"
    spans_before = icon_text(old_text, vocab) if icons_too else []
    drawn_before = {i for a, b in spans_before for i in range(a, b)}
    moved = kept_chars(old_text, text)          # letters the icon font drew plainly before the edit may stay
    plain = {moved[i] for i, c in enumerate(old_text) if icons_before[i] and i not in drawn_before and c not in blank
             and moved[i] >= 0}
    room = _on_canvas(layer, (psd.height, psd.width))   # what an ability's new text may fill (fit_below, ink_beyond)
    if area is not None:                        # an added Ability: its planned box; nothing printed before
        x0, y0, x1, y1 = clip(tuple(int(round(v)) for v in area), psd.width, psd.height)
        room = np.zeros_like(room)
        room[y0:y1, x0:x1, 3] = 255
    off_before = _ink_off_canvas(layer, psd.width, psd.height)
    splice_text(layer, text, parts)
    glyphs = {i for a, b in drawn for i in range(a, b)}
    if styles:          # words the engine set in a badge's icon font (typed after one) take their source's style
        now = _char_styles(layer)[1]
        for part_start, part_end, fill, like in styles:
            for a, b in _stretches([i for i in range(part_start, part_end) if like and now[i] and i not in glyphs]):
                layer.set_span_style(a, b, **like)
            layer.set_span_style(part_start, part_end, fill=fill)
    if restyle:         # likewise in a title: the title's own style
        now = _char_styles(layer)[1]
        for part_start, part_end, fill, like in restyle:
            for a, b in _stretches([i for i in range(part_start, part_end) if now[i] and i not in glyphs]):
                layer.set_span_style(a, b, **like, **({} if fill is None else {"fill": fill}))
    now = _char_styles(layer)[1]
    edited, sized = set(), set()    # an old badge or token edited in place: the new one holding most of its kept chars
    for a0, b0 in spans_before:     # (difflib may pair the ")" of "(1GCD)" with that of a "(2U)" typed after it)
        homes = Counter(k for k, (a, b) in enumerate(drawn) for i in range(a0, b0) if a <= moved[i] < b)
        if homes:
            edited.add(drawn[homes.most_common(1)[0][0]])
    stay = {span for span in typed if span in edited and all(now[i] for i in range(*span))}     # its size stays
    for a, b in typed:  # in the icon font at the card's badge or token style (the commonest size: owner 2026-10-24, not
        whole = all(now[i] for i in range(a, b))    # the style of the badge beside it the engine gives it)
        look = looks[text[a] == "("]
        if look is None:                        # sized as the text: typed_icon_fit checks its height
            sized.add((a, b))
        if not (whole and (look is None or (a, b) in edited)):     # one edited in place stays
            if look is None:                    # no badge or token on the card: at the text's own style
                layer.add_font(ICON_FONT)
                look, faces = {"Font": ICON_FONT}, list(dict.fromkeys(faces + [ICON_FONT]))
            layer.set_span_style(a, b, **look)
        piece = text[a:b]       # ligatures on (a title's off: Gust_of_Wind); a layered token untracked (review 2026-10-24)
        layered = piece[0] == "[" and token_variant(piece) in LAYERED
        layer.set_span_style(a, b, Ligatures=True, **({"Tracking": 0} if layered else {}))
        fill = variant_fill(text[a:b], vocab, tints, flag)
        if fill is not None:                    # a :sticker or :tint the colour of its kind on the card (owner 2026-10-24)
            layer.set_span_style(a, b, fill=fill)
    unstretch(layer, typed)
    # typed badges at the computed size and middle, typed icons shifted when off (owner 2026-10-25);
    for fit in badge_fits(layer, fonts, typed) if typed else ():    # a kept one as the guards keep it
        keys = ({"FontSize": fit["fit_size"], "BaselineShift": fit["fit_shift"]}
                if fit["kind"] == "badge" and (fit["start"], fit["end"]) not in stay else badge_wants(fit))
        if keys:
            layer.set_span_style(fit["start"], fit["end"], **keys)
    now = _char_styles(layer)[1]
    stray = [c for i, c in enumerate(text) if now[i] and i not in glyphs and c not in blank and i not in plain]
    if stray:           # letters the icon font draws plainly: typed in place of a badge, or "(1GCD)" made "(GCD)"
        flag("unsupported-card", f"{''.join(stray)!r} set in the icon font, which draws no badge or icon of it")
    for span_start, span_end, color in words:
        layer.set_span_style(span_start, span_end, fill=color.fill)
    status = {i for span_start, span_end, _ in words for i in range(span_start, span_end)}
    for plan, target in zip(plans, targets):    # typed into a styled phrase's words, or a copy joining it: its colour
        for a, b, joined in [(*span, span not in anchored) for _, _, now, anchored, _ in target for span in now]:
            if plan.fill is not None and (joined or a < end and start < b):    # (8308103f); a status word keeps the
                for c, d in _stretches([i for i in range(a, b) if i not in status]):   # chart's swatch (owner 2026-10-23)
                    layer.set_span_style(c, d, fill=plan.fill)
    if which == "ability":                      # one-colour badges and stickers under an overlay (owner 2026-10-25)
        set_alternates(layer, alternate_spans(text, drawn, _char_styles(layer)[1]) if one_colour(layer) else [], changed)

    def draw():
        return layer.rasterize(fonts=fonts, keep=before, unit=SPLICE_UNIT)
    seen = {}

    def check() -> Optional[str]:               # kept until the layer's pixels or a layer's box or visibility change
        key = (raster_key(layer), [(id(other._record), other.visible, tuple(other.bbox)) for other in psd.descendants()])
        if not seen or seen["key"] != key:
            seen.update(key=key, verdict=ink_beyond(psd, layer, room))
        return seen["verdict"]
    report = draw()
    if which == "ability" and not copy:         # onto the design below or off the canvas: smaller (the preview's fit)
        scale, again = fit_below(psd, layer, room, fonts, draw, off_before, check)
        report = again or report
        if scale >= 1 and not off_before:       # no size fits: up, to stay on the card
            up = lift(psd, layer, room)
    elif up:
        nudge(layer, 0, -up)
    loose = sorted(name for name, (how, _) in report.fonts.items() if how not in OK_FONTS)
    if loose:
        flag("font", f"not resolved exactly, drawn in a stand-in: {', '.join(loose)}")
    boxes = no_glyph(layer, text, changed, fonts)
    if boxes:
        flag("font", f"{which}: {''.join(dict.fromkeys(boxes))!r} has no glyph in its font (drawn as a box)")
    if "monochrome_emoji" in report.warnings and not _overlaid(layer):    # an SVG colour glyph the fork draws in one colour
        flag("font", f"{which}: colour glyphs (the icon font) drawn in one colour")
    anchor = anchor if which == "name" else 0
    if anchor:                                  # where the card maker put it, before an earlier bake's move
        nudge(layer, -anchor, 0)
    moved = back_on(psd, layer) if which == "name" and not off_before else 0
    if moved:                                   # a title running off the card shifts back on (R4, owner 2026-10-26)
        flags.append(f"overflow: name: moved {moved:+d} px to stay on the card")
    elif anchor:
        flags.append(f"name: back where the card maker put it ({-anchor:+d} px)")
    boxed = hold_lines(layer, fonts)
    if boxed:                                   # else Photoshop hides its last line when it lays it out again (soft)
        flags.append(f"overflow: {which}: its box {boxed} px taller to hold its last line")
    bbox = tuple(layer.bbox)
    spliced = report.splice
    band = bbox if spliced is None or spliced.fallback or moved or anchor or up else spliced.zone
    if not off_before and _ink_off_canvas(layer, psd.width, psd.height):
        flag("overflow", f"{which} runs off the canvas")
    if which == "name":
        lines = _title_lines(text_geometry(layer, fonts))
        if len(lines) > len(lines_before):
            flag("overflow", f"name: it runs onto {len(lines)} lines (it has {len(lines_before)})")
        if lines and max(lines) > max(lines_before + [old_bbox[3]]):
            flag("overflow", "name: a line of it moves below the title")
    misfit = typed_icon_fit(layer, typed, sized, fonts) if typed else None
    if misfit:
        flag(misfit[0], f"{which}: {misfit[1]}")
    reach = [(union_box([old_bbox, bbox]), fx_pad(layer))]
    reach += [(tuple(fx.bbox), fx_pad(fx)) for fx in old_fx]
    beyond = check() if which == "ability" and not copy else None
    rebuilt, dropped, covered, notes, restyled = (_rebuild_fx(psd, layer, text, plans, band, kept, fonts, engine, targets,
                                                          words) if plans else ([], [], [], [], []))
    if beyond:                                  # told, not refused (owner 2026-10-23: refuse only gross breakage)
        notes = [*notes, f"ability: {beyond}; check the card"]
    held = held_words(plans, kept, fx_layers(psd, layer) if plans else [])
    fresh = new_fx(text, styled, covered, flag)
    made = _regenerate_fx(psd, layer, text, fresh, fonts, engine) if fresh else []
    fx_names = [fx.name for fx in fx_layers(psd, layer)] if which == "ability" else []
    twice = sorted({name for name in fx_names if fx_names.count(name) > 1})
    if twice:                                   # two layers' words renumbered onto one name
        raise Refused("unsupported-card", f"two FX layers would be named {', '.join(twice)}; edit in Photoshop")
    reach += [(tuple(fx.bbox), fx_pad(fx)) for fx in made + [fx for fx, _ in rebuilt]]
    plain = not (old_fx or plans or made or rebuilt or dropped or restyled or _clip_chain(layer)
                 or (_effect_margin(layer)[1] and bbox != old_bbox)
                 or any(_effect_margin(owner)[1] for owner in _ancestors(layer)))
    lines = band if spliced is not None and not spliced.fallback and band and not (moved or anchor or up) \
        and area is None and gamma and plain else None
    if lines:                                   # its other glyphs keep their pixels
        reach[0] = (lines, fx_pad(layer))
    region = union_box([_pad(box, pad, psd) for box, pad in reach])
    set_single_line(psd, layer)                 # Photoshop lays it out as drawn
    return Edit(which, old_text, text, [layer], [_record_index(layer)], region, story=story,
                fx=[(fx.name, {bytes(key): fx.tagged_blocks[key].tobytes() for key in EFFECT_BLOCKS if key in fx.tagged_blocks})
                    for fx in made],
                dropped=dropped, rebuilt=[(_record_index(fx), old, fx.name) for fx, old in rebuilt],
                restyled=[(_record_index(fx), blocks) for fx, blocks in restyled],
                fx_names=fx_names, notes=notes + flags, held=held, fonts=faces,
                whole=spliced is None or bool(spliced.fallback), lifted=up,
                lines=lines, boxed=bool(boxed))


# ========================
# Text colours (Card Edits.json `colors`, owner 2026-11-07)
# ========================
def _hex(fill) -> Optional[str]:
    """"#rrggbb" of an RGB 0..1 fill (the website's), None for none."""
    return None if fill is None else "#%02x%02x%02x" % tuple(max(0, min(255, int(round(float(v) * 255)))) for v in fill[-3:])


def _fill(color: str) -> tuple:
    """An "#rrggbb" colour as the engine's RGB 0..1."""
    return tuple(int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))


def color_order(value: list) -> list:
    """The colours in the order they paint: whole fields first, then phrases as listed (a phrase wins over its field)."""
    return sorted(value, key=lambda one: bool(one.get("phrase")))


def color_layers(psd: PSDImage, field: str, card: str) -> list:
    """The text layers printing a field, in reading order: the title's parts (one split over layers too), the Ability's;
    as picture 1 shows them (the entry's picture: every picture's bake recolours the same layers)."""
    texts = [layer for layer in psd.descendants() if _is_text_layer(layer) and EFFECT_LAYER_TAG not in layer.name]
    with variant_view(psd, 1 if len(variants_of(psd)) > 1 else None):
        if field == "name":
            return PSDParser.title_parts(psd, texts, card.replace("_", " "))[0]
        return PSDParser.ability_parts(texts)


def _hiding(node) -> list:
    """[(effect, kind)]: the node's enabled opaque Normal Color, Gradient and Pattern Overlays."""
    from psd_tools.api.effects import ColorOverlay, GradientOverlay, PatternOverlay
    kinds = {ColorOverlay: "Color Overlay", GradientOverlay: "Gradient Overlay", PatternOverlay: "Pattern Overlay"}
    effects = node.effects
    return [(effect, kind) for effect in (effects if effects.enabled else ()) for cls, kind in kinds.items()
            if isinstance(effect, cls) and effect.enabled and float(effect.opacity) >= 100 and effect.blend_mode == BlendMode.NORMAL]


def _opaque(one) -> bool:
    """A layer that hides what it is clipped onto: Normal, opaque (Paint Bomb's at 73% does not)."""
    return one.blend_mode == BlendMode.NORMAL and one.opacity >= 250 and one.fill_opacity >= 250


def _clipped_over(layer) -> list:
    """The shown layers clipped onto a text layer when one of them hides its letters (_opaque: the Totems' fire,
    Paintbrush's rainbow), else []."""
    chain = [one for one in _clip_chain(layer) if one.visible]
    return chain if any(map(_opaque, chain)) else []


def _clip_text(chain: list) -> str:
    return f"clipped layer {next(one for one in chain if _opaque(one)).name!r}"


def _painted_over(layer, own: bool = True) -> Optional[str]:
    """What hides a text layer's own colours, its own or a group's: an opaque Normal Color, Gradient or Pattern Overlay
    ("its Gradient Overlay"), a layer clipped onto it, or a fill at 0; None when they show. own False: not its own
    overlays or clipped layers (_overlay_off's)."""
    node = layer
    while node is not None and node.kind != "psdimage":
        whose = "its" if node is layer else f"its group {node.name!r}'s"
        if node is layer and node.fill_opacity == 0:
            return f"{whose} fill at 0%"
        clipped = _clipped_over(layer) if node is layer and own else []
        if clipped:
            return f"its {_clip_text(clipped)}"
        hiding = _hiding(node) if own or node is not layer else []
        if hiding:
            return f"{whose} {hiding[0][1]}"
        node = node.parent
    return None


def _solid_overlay(layer) -> Optional[tuple]:
    """The RGB 0..1 of the layer's lone opaque Normal Color Overlay (its only enabled overlay, nothing above hides it,
    full fill, every icon-font glyph a one-colour badge): a phrase's colour turns it off and gives its colour to every
    other letter and badge (the same look)."""
    from psd_tools.api.effects import ColorOverlay, GradientOverlay, PatternOverlay
    effects = layer.effects
    overlays = [e for e in (effects if effects.enabled else ()) if e.enabled and isinstance(e, (ColorOverlay, GradientOverlay, PatternOverlay))]
    hiding = _hiding(layer)
    if len(overlays) != 1 or len(hiding) != 1 or hiding[0][1] != "Color Overlay" or _painted_over(layer, own=False) \
            or layer.fill_opacity < 255 or _clipped_over(layer):
        return None
    text, salted = layer.text, _salted(layer)
    if any(icon and i not in salted and not text[i].isspace() for i, icon in enumerate(_char_styles(layer)[1])):
        return None                             # a colour icon would show its colours
    color = hiding[0][0].descriptor.get(b"Clr ")
    keys = (b"Rd  ", b"Grn ", b"Bl  ")
    if color is None or any(k not in color for k in keys):
        return None
    return tuple(max(0.0, min(1.0, float(getattr(color[k], "value", color[k])) / 255)) for k in keys)


def _salted(layer) -> set:
    """The str indices in the icon font's Stylistic Alternates: one-colour badges, coloured as letters."""
    icons = _char_styles(layer)[1]
    return {i for a, b in (stylistic_alternates(layer) or ()) for i in range(a, b) if i < len(icons) and icons[i]}


def _unveils(layer) -> bool:
    """Whether turning the layer's own overlays off and hiding its clipped layers shows its colours (nothing else hides
    them)."""
    return bool(_hiding(layer) or _clipped_over(layer)) and _painted_over(layer, own=False) is None


def _overlay_off(layer) -> str:
    """A whole field's colour (owner 2026-10-24: no refusing), or a phrase's on all of the layer's text (_whole_span) or
    under _solid_overlay: its own overlays off when that shows it, the layers clipped onto it hidden (Edit.hidden); the
    first's kind, or ""."""
    if not _unveils(layer):
        return ""
    hiding, clipped = _hiding(layer), _clipped_over(layer)
    for effect, _ in hiding:
        effect.descriptor[b"enab"] = Bool(False)
    for one in clipped:
        one.visible = False
    layer._psd.mark_updated()
    return hiding[0][1] if hiding else _clip_text(clipped)


def color_spans(layers: list, one: dict) -> tuple:
    """
    Where a colour paints: ([(layer, start, end)], why not). A phrase matches as the effect layers' (phrase_ranges: any
    case, any blanks between its words, whole words when it occurs as one, "word#n" the nth), counted over the layers in
    reading order; no phrase: every layer whole. why: "" when found.
    """
    what = "title" if one["field"] == "name" else "ability"
    if not layers:
        return [], f"no {what} text layer"
    phrase = one.get("phrase")
    if not phrase:
        return [(layer, 0, len(layer.text)) for layer in layers], ""
    words, nth = _numbered(phrase)
    found, whole = [], []
    for layer in layers:
        some, own = _matches(layer.text, words)
        found += [(layer, a, b) for a, b in some]
        whole += [(layer, a, b) for a, b in own]
    hits = whole or found
    hits = hits[nth - 1:nth] if nth else hits
    if hits:
        return hits, ""
    return [], (f"there is no {_nth(nth)} {words!r} in the {what}" if nth and found else f"{words!r} is not in the {what}")


def _nth(k: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(k, f"{k}th")


def _fx_words(psd: PSDImage, layer) -> list:
    """[(start, end, FX layer name)]: the layer's words an Ability FX layer cuts (its own pixels, drawn over them)."""
    if EFFECT_LAYER_TAG in layer.name:
        return []
    return [(a, b, fx.name) for fx in fx_layers(psd, layer) for a, b in phrase_ranges(layer.text, fx_phrases(layer, fx))]


def _painted(layer, a: int, b: int, fx: list, kept=frozenset()) -> list:
    """The str indices a colour paints in layer.text[a:b]: not the icon font's (badges and icons keep their colours), not
    an FX layer's words (its pixels show), not kept's (status_letters)."""
    icons = _char_styles(layer)[1]
    return [i for i in range(a, b) if not icons[i] and i not in kept and not any(s <= i < e for s, e, _ in fx)]


def _whole_span(text: str, a: int, b: int) -> bool:
    """Whether [a, b) holds all of a layer's text (blanks and breaks around it): a phrase colouring the whole layer, as
    a whole field's colour does there (its own overlays go off)."""
    return not text[:a].replace("\x03", " ").strip() and not text[b:].replace("\x03", " ").strip()


def _whole_words(text: str, words: str) -> bool:
    """Whether the phrase's words are all of the text (_whole_span of a match)."""
    return any(_whole_span(text, a, b) for a, b in _matches(text, words)[0])


def status_letters(text: str, chart: Optional[dict], icons=()) -> set:
    """The str indices of the text's capitalised chart status words (status_word_spans): they keep the chart's colour,
    as the guards set them (owner 2026-11-07)."""
    return {i for a, b, _ in status_word_spans(text, chart or {}, list(icons)) for i in range(a, b)}


def _status_kept(layer, field: str, chart: Optional[dict]) -> set:
    """status_letters of an Ability layer (titles have none)."""
    return status_letters(layer.text, chart, icon_spans(layer)) if field == "ability" and chart else set()


def plan_colors(psd: PSDImage, value: list, card: str, retyped: Optional[dict] = None, chart: Optional[dict] = None,
                boxes: Optional[dict] = None) -> list:
    """
    classify()'s `colors`: each colour's outcome, [(colour, "baked" | "redundant" | why not)] in the entry's order.
    Found on the card's layers (color_spans): redundant when every letter it is the last to paint (color_order) has
    that colour already, or none is left to it; a phrase an effect layer cuts (its own pixels show) is refused, as one
    of badges and icons alone. retyped: the fields a text edit of the entry sets
    ({field: its new text}): found in that text, baked (the bake finds it again once the text is set) unless an overlay
    paints over the layers it lands in (boxes: the parts job's {str(layer id): new text}). chart: status_fills(); the
    Ability's capitalised status words keep its colours (a phrase of them alone refused). A whole field's colour turns
    its layers' own overlays off and hides the layers clipped onto them (set_colors), as a phrase's does on a layer it
    is all of (_whole_span) or under a lone Color Overlay (_solid_overlay): not refused for them, nor its phrases after
    it, and baked even in the colour it has.
    """
    out, final, painted, lit, unveil = {}, {}, {}, set(), set()
    ordered = color_order(value)
    for k, one in enumerate(ordered):
        field, phrase = one["field"], one.get("phrase")
        what = "title" if field == "name" else "ability"
        if retyped and field in retyped:      # found in the new text (the bake looks again once it is set)
            words, nth = _numbered(phrase) if phrase else ("", 0)
            some, own = _matches(retyped[field] or "", words) if phrase else ([], [])
            hits = (own or some)[nth - 1:nth] if nth else own or some
            kept = status_letters(retyped[field] or "", chart) if field == "ability" else set()
            alone = hits and all(i in kept or retyped[field][i].isspace() for a, b in hits for i in range(a, b))
            out[k] = (STATUS_KEPT.format(phrase) if alone else "baked") if hits or not phrase \
                else f"{words!r} is not in the new {what}"
            into = []
            if out[k] == "baked":               # the rule below, on the layers it lands in
                layers = color_layers(psd, field, card)
                new = {id(layer): (boxes or {}).get(str(layer.layer_id), retyped[field] if len(layers) == 1 else layer.text)
                       for layer in layers}
                into = [layer for layer in layers if not phrase or _matches(new[id(layer)] or "", words)[0]]
            full = {id(layer) for layer in into if phrase and _whole_words(new[id(layer)] or "", words)}
            over = {_painted_over(layer, bool(phrase) and id(layer) not in lit | full and not _solid_overlay(layer)) for layer in into}
            if into and None not in over:
                out[k] = f"{sorted(over)[0]} paints over the {what}'s colours: recolour it in Photoshop"
            elif into and not phrase:           # its own overlays go off (set_colors)
                lit |= {id(layer) for layer in into}
            elif into:
                lit |= full
            continue
        spans, why = color_spans(color_layers(psd, field, card), one)
        if field == "name" and (why or not spans) and title_in_picture(psd, card):
            why = PICTURE_TITLE                  # the website's alone, as a name edit there (owner 2026-10-26)
        cut = [name for layer, a, b in spans for s, e, name in _fx_words(psd, layer) if s < b and a < e] \
            if phrase and not why else []
        why = why or (f"{phrase!r} is drawn by its effect layer {cut[0]!r}, which keeps its own pixels" if cut else "")
        mine = [(layer, i) for layer, a, b in spans for i in _painted(layer, a, b, _fx_words(psd, layer), _status_kept(layer, field, chart))] \
            if not why else []
        if not why and not mine and phrase and [(layer, i) for layer, a, b in spans for i in _painted(layer, a, b, [])]:
            why = STATUS_KEPT.format(phrase)
        if not why and not mine:
            why = (f"{phrase!r} is only badges or icons, which keep their own colours" if phrase
                   else "only badges, icons, effect layers' or status words, which keep their own colours")
        whole = not phrase                          # its own overlays go off (set_colors)
        solid = {id(layer) for layer, _, _ in spans if _solid_overlay(layer)} if phrase and not why else set()
        full = {id(layer) for layer, a, b in spans if _whole_span(layer.text, a, b)} if phrase and not why else set()
        over = {_painted_over(layer, not whole and id(layer) not in lit | solid | full) for layer, _ in mine}
        if not why and None not in over:        # nothing would show on the card image
            why = f"{sorted(over)[0]} paints over the {what}'s colours: recolour it in Photoshop"
        if why:
            out[k] = why
            continue
        unveil |= {k} if (whole or solid) and any(_unveils(layer) for layer, _, _ in spans) \
            or any(_unveils(layer) for layer, _, _ in spans if id(layer) in full) else set()
        lit |= {id(layer) for layer, _, _ in spans if whole} | solid | full
        for layer, i in mine:
            painted[id(layer)] = layer
            final[(id(layer), i)] = k
    fills = {key: _char_styles(layer)[0] for key, layer in painted.items()}
    for k, one in enumerate(ordered):
        if k not in out:
            last = [(key, i) for (key, i), by in final.items() if by == k]
            out[k] = "baked" if k in unveil or any(_hex(fills[key][i]) != one["color"] for key, i in last) else "redundant"
    found = {id(one): how for k, one in enumerate(ordered) for how in [out[k]]}
    return [(one, found[id(one)]) for one in value]


def recolour(layer, paint: dict, fonts) -> None:
    """
    The layer's stored raster recoloured where its glyphs of the str indices in paint are ({index: RGB 0..1}): each
    inked pixel takes the colour of the glyph nearest it (text_geometry's ink boxes, _nearest), its alpha kept, so the
    letters stay where Photoshop drew them (a redraw would lay them out again: Leeroy Jenkins' JENKINS 2 px wider). The
    story's strikes go as a redraw's do (Photoshop composes the story at open).
    """
    from psd_tools.api import text_edit
    from psd_tools.composite.text.geometry import text_geometry
    from psd_tools.composite.text.verify import stored_raster
    rgba, bbox = stored_raster(layer)
    if rgba is None:
        return
    glyphs = [glyph for glyph in text_geometry(layer, fonts)["glyphs"] if glyph["ink"]]
    x0, y0 = bbox[0], bbox[1]
    ys, xs, nearest = _nearest(glyphs, rgba[..., 3] > 0, (x0, y0))     # the raster may start off the canvas
    out, changed = rgba.copy(), False
    for k, glyph in enumerate(glyphs):
        fill = paint.get(glyph["i"])
        if fill is None:
            continue
        hit = nearest == k
        if hit.any():
            out[ys[hit] - y0, xs[hit] - x0, :3] = np.asarray(fill, np.float32)
            changed = True
    if not changed or np.array_equal(out, rgba):
        return
    layer._store_raster(out, bbox)
    text_edit.drop_strikes(layer, [str(font["Name"].value) for font in layer.resource_dict["FontSet"]])
    layer._invalidate_bbox()
    layer._psd.mark_updated()


def color_views(psd: PSDImage, value: list, card: str, numbers: list) -> set:
    """The pictures a `colors` job shows in: those showing a layer it recolours."""
    layers = [layer for one in value for layer, _, _ in color_spans(color_layers(psd, one["field"], card), one)[0]]
    return {n for n in numbers if any(shown_in(layer, n) for layer in layers)}


def set_colors(psd: PSDImage, value: list, fonts_dir: str, card: str = "", chart: Optional[dict] = None) -> Edit:
    """
    Recolour what a `colors` job names, found again on the card as its earlier edits left it (color_spans): each colour's
    letters take it in the runs (set_span_style splits them, the Txt2 story with them; whole fields first, then
    phrases), the icon font's badges and icons and an Ability FX layer's words keeping theirs, and in the stored raster
    (recolour: the letters stay where Photoshop drew them). Fonts, sizes, effects and the composer are untouched.

    Returns:
        One Edit ("colors", over every layer it set: gate_bytes reads it as a text edit, gate_stories lets their stories
        change). Told: an overlay on a layer that paints over its colours (one_colour), the layer's own turned off for a
        whole field's colour or a whole layer's (_whole_span), or a phrase's under a lone Color Overlay (_overlay_off;
        that one's colour then goes to the rest, its one-colour badges too), words an effect layer keeps. Raises Refused(field "colors") for a colour no longer found (the other fields still bake)
    """
    jobs, notes, hidden = {}, [], []
    for one in color_order(value):
        spans, why = color_spans(color_layers(psd, one["field"], card), one)
        if why:
            raise Refused("unsupported-card", f"colors: {color_text(one)}: {why}", "colors")
        for layer, a, b in spans:
            jobs.setdefault(layer.layer_id, (layer, one["field"], []))[2].append((a, b, one))
    fonts = font_map(fonts_dir)
    layers, records, stories, boxes = [], [], [], []
    for layer, which, spans in jobs.values():
        what = "title" if which == "name" else "ability"
        fx, paint, status = _fx_words(psd, layer), {}, _status_kept(layer, which, chart)
        whole = any(not one.get("phrase") or _whole_span(layer.text, a, b) for a, b, one in spans)
        base = None if whole else _solid_overlay(layer)
        clipped = _clipped_over(layer)
        off = _overlay_off(layer) if whole or base else ""
        hidden += [_record_index(one) for one in clipped if not one.visible]       # _overlay_off hid them
        salted = _salted(layer) if off else set()       # its badges, coloured as the overlay did
        todo = ([(0, len(layer.text), None)] if base and off else []) + spans
        kept = sorted({name for a, b, _ in spans for s, e, name in fx if s < b and a < e})
        if kept:
            notes.append(f"colors: {what}: the words of {', '.join(kept)} keep their look (the effect layer's own pixels)")
        if any(a <= i < b for a, b, _ in todo for i in status):
            notes.append(f"colors: {what}: status words keep the chart's colour")
        if off:
            notes.append(f"colors: {what}: its {off} is off for the colour")
        for a, b, one in todo:
            fill, phrase = (base, None) if one is None else (_fill(one["color"]), one.get("phrase"))
            got = _painted(layer, a, b, fx, status) + ([] if phrase and not _whole_span(layer.text, a, b) else [i for i in range(a, b) if i in salted])
            for c, d in _stretches(sorted(got)):
                layer.set_span_style(c, d, fill=fill)
                paint.update(dict.fromkeys(range(c, d), fill))
        recolour(layer, paint, fonts)
        if one_colour(layer):
            notes.append(f"colors: {what}: {_tinted(layer) or 'an overlay'} paints over its colours")
        layers.append(layer)
        records.append(_record_index(layer))
        stories.append(story_index(layer))
        boxes.append(_pad(tuple(layer.bbox), fx_pad(layer), psd))
    if not layers:
        return Edit("colors", value, value, [], [], None)
    return Edit("colors", value, value, layers, records, union_box(boxes), story=stories[0], stories=stories[1:],
                notes=notes, hidden=hidden)


# ========================
# Stars and types (the icon fields, owner 2026-10-23)
# ========================
def _tol(size: float) -> float:
    return max(LAYOUT_TOL_MIN, LAYOUT_TOL * size)


def _centre(box) -> tuple:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _moved_box(box, dx: int, dy: int) -> tuple:
    return box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy


def _drawn(layer) -> list:
    """The layer, or a group's leaves with pixels that show when the group does (their own flags and their groups')."""
    if not layer.is_group():
        return [layer] if layer.has_pixels() else []
    return [leaf for child in layer if child.visible for leaf in _drawn(child)]


def _core(layer) -> Optional[tuple]:
    """The union of the core boxes the parser clusters stars by (PSDParser._core_box), None when nothing draws."""
    return union_box([PSDParser._core_box(leaf) for leaf in _drawn(layer)])


# The records (owner 11-07): kept on the card data's branch (db/, Card sync's data commit)
RECORDS_DIR = "CARD_RECORDS_DIR"        # env: the data's db/ folder; unset, or a file it lacks: {}
SPOT_RECORD = "spot_record.json"        # {PSD path: {layer id: entry}}: icons a redraw moved, their earlier ink cores
RENDER_RECORD = "render_record.json"    # {PSD path: facts}: each card's look read off its stored image before the strip
_RECORDS: dict = {}                     # file name -> its record, read once


def _record(name: str) -> dict:
    """A record from the data's db/ (RECORDS_DIR)."""
    if name not in _RECORDS:
        folder = environ.get(RECORDS_DIR, "")
        try:
            with open(join(folder, name) if folder else "", encoding="utf-8") as fh:     # "": FileNotFoundError
                _RECORDS[name] = load(fh)
        except FileNotFoundError:
            _RECORDS[name] = {}
    return _RECORDS[name]


def set_records(render: Optional[dict] = None, spot: Optional[dict] = None) -> None:
    """Use these records from now on (the data build's merged ones); None keeps that one."""
    for name, found in ((RENDER_RECORD, render), (SPOT_RECORD, spot)):
        if found is not None:
            _RECORDS[name] = found
    _record_names.cache_clear()


def spot_record() -> dict:
    """{PSD path: {layer id: {"link", "trnf", "content", "core"}}}; {} without the file."""
    return _record(SPOT_RECORD)


def spot_key(layer, rel: str) -> Optional[dict]:
    """What a record entry holds while the layer is as the pass left it: its link (repo path), transform and content
    ID; None for a layer that links no file."""
    try:
        linked = layer.kind == "smartobject" and layer.smart_object.kind == "external"
        if not linked:
            return None
        return {"link": layer.smart_object.link_target(rel or None), "trnf": [round(v, 3) for v in corners_of(layer)],
                "content": layer.smart_object.content_id or ""}
    except (ValueError, KeyError, AttributeError):
        return None


def _icon_core(layer, rel: str) -> tuple:
    """A header type icon's core for its spots: the spot record's while its key holds, else its ink core."""
    record = spot_record()
    entry = (record.get(rel) or record.get(_record_names(SPOT_RECORD).get(basename(rel or ""), "")) or {}).get(
        str(layer.layer_id))
    if entry and spot_key(layer, rel) == {key: entry[key] for key in ("link", "trnf", "content")}:
        return tuple(entry["core"])
    return _core(layer) or tuple(layer.bbox)


def spot_entries(before: PSDImage, after: PSDImage, parser: PSDParser, rel: str, name: str = "") -> dict:
    """One PSD's spot record entries: each header type icon left linked whose ink core moved, its core from before (the
    PSD before a pass, a conversion or a link redraw; a core the record already holds carries on); {layer id: entry}."""
    was = {layer.layer_id: _icon_core(layer, rel) for _, layer in type_row(before, parser, name)[0]}
    out = {}
    for _, layer in type_row(after, parser, name)[0]:
        key, core = spot_key(layer, rel), was.get(layer.layer_id)
        if key and core and tuple(core) != (_core(layer) or tuple(layer.bbox)):
            out[str(layer.layer_id)] = {**key, "core": [int(v) for v in core]}
    return out


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


def _star_units(container, types=(), below: bool = False) -> tuple:
    """
    (shown, hidden) stars directly in a group above the midline (below: under it, a second creature's): its child
    groups that count stars, and its loose layers' stars as the parser judges their pieces (star_leaf, star_pieces,
    star_sets: a glow joins its star, art is none); one layer drawing several stars is one unit counting them (Grimora's
    row). A layer named after a type icon (types: Types/ names) is left out (the parser reads it as a type too: Pike Beam).
    """
    height = (container if isinstance(container, PSDImage) else container._psd).height
    side = (lambda box: not in_header(box, height)) if below else (lambda box: in_header(box, height))
    shown, hidden, loose = [], [], {True: [], False: []}
    for layer in container:
        if layer.name.strip().lower() in types:
            continue
        if layer.is_group():
            count = PSDParser.stars_in(layer, types)
            core = _core(layer)
            if count and core and side(core):
                (shown if layer.visible else hidden).append(_Unit([layer], core, count))
        elif star_leaf(layer, types):
            loose[layer.visible].append(layer)
    for on, layers in loose.items():
        pieces = {id(layer): star_pieces(layer) for layer in layers}
        items = [(layer, box) for layer in layers for box in pieces[id(layer)] if side(box)]
        several: set = set()
        for members in star_sets([box for _, box in items]):
            owners = []
            for i in members:
                if items[i][0] not in owners:
                    owners.append(items[i][0])
            if len(owners) == 1 and len(pieces[id(owners[0])]) > 1:     # one layer drawing several: one unit
                if id(owners[0]) not in several:
                    several.add(id(owners[0]))
                    (shown if on else hidden).append(_Unit(owners, PSDParser._core_box(owners[0]), len(pieces[id(owners[0])])))
                continue
            (shown if on else hidden).append(_Unit(owners, union_box([items[i][1] for i in members])))
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


def _line(units: list, size: tuple) -> Optional[tuple]:
    """(first centre, step) when the units in reading order are evenly spaced in a row or a column; else None."""
    order = _reading(units, size[1], True)
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


def _place(units: list, spots: list, size: tuple, snap: bool = False) -> tuple:
    """
    Units onto spots: one already on a spot (within the tolerance) stays (snap: moves onto it, a shrunk row kept even),
    the others move to the free spots in order.

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
            if snap and (round(spot[0] - unit.centre[0]) or round(spot[1] - unit.centre[1])):
                moves.append((unit, round(spot[0] - unit.centre[0]), round(spot[1] - unit.centre[1])))
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


def _ink_overlap(a: tuple, b: tuple) -> int:
    """Pixels where two inks ((bbox, solid alpha), _ink) overlap."""
    (ax0, ay0, _, _), ink_a = a
    (bx0, by0, _, _), ink_b = b
    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax0 + ink_a.shape[1], bx0 + ink_b.shape[1]), min(ay0 + ink_a.shape[0], by0 + ink_b.shape[0])
    if x1 <= x0 or y1 <= y0:
        return 0
    return int((ink_a[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0] & ink_b[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]).sum())


def _grown(ink: tuple, px: int) -> tuple:
    """An ink ((bbox, solid alpha), _ink) grown by px on every side: a square dilation."""
    (x0, y0, x1, y1), out = ink
    out = np.pad(out, px)
    for axis in (0, 1):                         # a sliding window's sum along each axis
        sums = np.cumsum(np.pad(out, [(px + 1, px) if k == axis else (0, 0) for k in (0, 1)]), axis=axis, dtype=np.int32)
        n = sums.shape[axis] - 2 * px - 1
        out = (np.take(sums, range(2 * px + 1, n + 2 * px + 1), axis=axis) - np.take(sums, range(n), axis=axis)) > 0
    return (x0 - px, y0 - px, x1 + px, y1 + px), out


def _ink_off(ink: tuple, inks: list) -> float:
    """The share of inks' pixels ((bbox, solid alpha) each, _ink) off another ink."""
    total = sum(int(mine[1].sum()) for mine in inks)
    return 1 - sum(_ink_overlap(mine, ink) for mine in inks) / total if total else 0.0


def _blocked(psd: PSDImage, inks: list, core: tuple, others: list) -> Optional[str]:
    """
    Why a star or icon cannot go where its inks are ((bbox, alpha) on the canvas): its core off the canvas, or its ink on
    the ink of another layer that shows (others: the visible text layers, and the stars or icons it must clear); None.
    """
    slack = _tol(max(core[2] - core[0], core[3] - core[1]))     # the cards' own icons pass the edge a little
    if core[0] < -slack or core[1] < -slack or core[2] > psd.width + slack or core[3] > psd.height + slack:
        return "runs off the canvas"
    reach = union_box([(b[0], b[1], b[0] + solid.shape[1], b[1] + solid.shape[0]) for b, solid in filter(None, inks)])
    for other in others:
        b = tuple(other.bbox)
        if reach is None or b[2] <= reach[0] or b[0] >= reach[2] or b[3] <= reach[1] or b[1] >= reach[3]:
            continue                            # no ink of its can meet mine: not read
        theirs = _ink(other)
        if theirs is None or (other.kind != "type" and theirs[1].sum() > FRAME_SHARE * psd.width * psd.height):
            continue                            # a fill or art filed with the stars (Kyle's) is neither
        if any(_inks_touch(mine, theirs) for mine in inks):
            return f"would touch {other.name!r}"
    return None


def _drops(psd: PSDImage, core: tuple) -> range:
    """
    The px an added star or type icon whose spot is blocked steps down, in turn (owner 11-08: down until it fits):
    LOWER_STEP of its height at a time, its centre kept above the card's middle (no star or type is read below it) and
    its core above the Ability text under it.
    """
    step = max(LOWER_MIN, round(LOWER_STEP * (core[3] - core[1])))
    last = min([TYPE_REGION_RATIO * psd.height - (core[1] + core[3]) / 2] +
               [layer.bbox[1] - core[3] for layer in psd.descendants() if _is_text_layer(layer) and layer.is_visible()
                and base_name(layer).strip().lower() == "ability" and layer.bbox[1] >= core[3]])
    return range(step, int(last) + 1, step)


def _lowered(psd: PSDImage, inks: list, core: tuple, others: list, icons: list) -> Optional[int]:
    """The px an added type icon (inks, core: at its spot) steps down to fit (_drops): clear of others (_blocked) and
    of the row's icons (inks) past ICON_TOUCH; None when it fits nowhere."""
    for d in _drops(psd, core):
        mine = [(_moved_box(box, 0, d), solid) for box, solid in inks]
        if _blocked(psd, mine, _moved_box(core, 0, d), others) is None and not any(
                _ink_overlap(a, b) > ICON_TOUCH * min(a[1].sum(), b[1].sum()) for a in mine for b in icons):
            return d
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
    """The text stars and icons keep clear of: on any of the card's pictures, none in the images group (picture)."""
    pictures, out = variants_of(psd), {}
    for n in [None, *pictures] if len(pictures) > 1 else [None]:
        with variant_view(psd, n):
            out.update({id(layer): layer for layer in psd.descendants() if _is_text_layer(layer) and layer.is_visible()
                        and not any(owner.name.strip().lower() == "images" for owner in layer_parents(layer))})
    return list(out.values())


def _copy_name(unit: _Unit, container, taken: set) -> str:
    """Photoshop's name for a copy of the unit's top layer: "<name> copy", "<name> copy 2"... (taken: names given already)."""
    base = sub(r"\s+copy(\s+\d+)?$", "", unit.layers[len(unit.layers) - 1].name)
    names = {layer.name for layer in container.descendants()} | taken
    name, n = f"{base} copy", 2
    while name in names:
        name, n = f"{base} copy {n}", n + 1
    taken.add(name)
    return name


def _filtered(leaf) -> bool:
    """A smart object carrying smart filters (filterFX)."""
    placed = next((leaf.tagged_blocks.get_data(key) for key in PLACED[:2] if key in leaf.tagged_blocks), None)
    return placed is not None and b"filterFX" in placed.data


def _shiftable(layer) -> Optional[str]:
    """Why a layer (a group: everything in it) cannot be moved; None when it can. A smart-filtered layer moves when its
    FEid item is found by its placed ID (the fork's filter_item; S1), as Photoshop moves one (_shift_filters)."""
    for leaf in [layer, *(layer.descendants() if layer.is_group() else ())]:
        if leaf.is_group():
            continue
        if leaf.kind in ("type", "shape") or leaf.has_vector_mask() or Tag.VECTOR_ORIGINATION_DATA in leaf.tagged_blocks:
            return f"{leaf.name!r} is {'text' if leaf.kind == 'type' else 'a vector shape'}"
        if _filtered(leaf) and not (leaf.kind == "smartobject" and leaf.smart_object.filter_item is not None):
            return f"{leaf.name!r} has smart filters"
    return None


def _shift_filters(leaf, dx: int, dy: int) -> None:
    """
    A smart-filtered layer moved as Photoshop moves one (The Flash's 'pngwing.com copy 3'): its FEid item's rectangle
    moves (its planes stay), the filter mask's too when linked to the layer, and its filters are marked as not valid
    at the new place (validAtPosition false). Nothing for another layer.
    """
    if leaf.is_group() or leaf.kind != "smartobject" or not _filtered(leaf):
        return
    so = leaf.smart_object
    item = so.filter_item
    if item is None:                            # _shiftable refuses such a layer
        return
    top, left, bottom, right = item.rectangle
    item.rectangle = (top + dy, left + dx, bottom + dy, right + dx)
    extra = item.extra
    if getattr(extra, "rectangle", None) and bool(getattr(so.filters.get(b"filterMaskLinked"), "value", False)):
        top, left, bottom, right = extra.rectangle
        extra.rectangle = [top + dy, left + dx, bottom + dy, right + dx]
    so.filters[b"validAtPosition"] = Bool(False)


def _filter_block(psd: PSDImage):
    """The document's FEid items (a ListElement), None without them."""
    for key in (Tag.FILTER_EFFECTS1, Tag.FILTER_EFFECTS2, Tag.FILTER_EFFECTS3):
        items = psd.tagged_blocks.get_data(key) if psd.tagged_blocks is not None else None
        if items is not None:
            return items
    return None


def _copy_filters(source, copy) -> None:
    """
    A copied smart-filtered layer's own FEid item, as Photoshop gives a copy one: its source's item (planes, mask) named
    after the copy's placed ID, right after the source's (copy and source are next to each other in file order); a
    later _shift moves it with the copy. Call after same_placed().
    """
    for old, new in zip([source, *(source.descendants() if source.is_group() else ())],
                        [copy, *(copy.descendants() if copy.is_group() else ())]):
        if old.is_group() or old.kind != "smartobject" or not _filtered(old):
            continue
        item, items = old.smart_object.filter_item, _filter_block(copy._psd)
        if item is None or items is None:
            continue
        made = deepcopy(item)
        made.uuid = new.smart_object.placed_id
        at = next(i for i, one in enumerate(items) if one is item)
        items.insert(at + 1, made)
        copy._psd.mark_updated()


def _order_filters(psd: PSDImage) -> None:
    """The FEid items in file order of their layers (Photoshop's order), when every item names one filtered layer."""
    items = _filter_block(psd)
    if not items:
        return
    at = {}
    for i, layer in enumerate(psd.descendants()):
        if not layer.is_group() and layer.kind == "smartobject" and _filtered(layer):
            at.setdefault(layer.smart_object.placed_id, i)
    keys = [at.get((item.uuid or "").strip("\x00")) for item in items]
    if None in keys or len(set(keys)) != len(keys) or keys == sorted(keys):
        return
    ordered = [item for _, item in sorted(zip(keys, list(items)), key=lambda pair: pair[0])]
    for _ in range(len(items)):
        items.pop()
    for item in ordered:
        items.append(item)
    psd.mark_updated()


def _shift(layer, dx: int, dy: int, filters: bool = False) -> None:
    """Move a layer (a group: every layer in it) on the canvas: pixels, masks, a smart object's transform, fxrp, and
    with filters a smart-filtered one's FEid item (_shift_filters: the stars and type icons, S1)."""
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
        if filters:
            _shift_filters(leaf, dx, dy)
        leaf._psd.mark_updated()


def _scalable(layer) -> Optional[str]:
    """Why a star's layer (a group: every layer in it) cannot shrink (_scale); None when it can: its drawn layers pixel
    layers or unwarped smart objects with no mask, effects or smart filters (an adjustment clipped to it has none), its
    groups no mask or effects (_scale leaves them as they were)."""
    for leaf in [layer, *(layer.descendants() if layer.is_group() else ())]:
        if leaf.is_group():
            if leaf.has_mask() or leaf.has_vector_mask() or _enabled_effects(leaf):
                return f"{leaf.name!r} has a mask or effects"
            continue
        if leaf.kind in ("type", "shape") or leaf.has_pixels() and leaf.kind not in ("pixel", "smartobject"):
            return f"{leaf.name!r} is a {leaf.kind} layer"
        if leaf.has_mask() or leaf.has_vector_mask() or _enabled_effects(leaf) or _filtered(leaf) \
                or leaf.kind == "smartobject" and warped(leaf):
            return f"{leaf.name!r} has a mask, effects, smart filters or a warp"
    return None


def _scale(layer, s: float, centre: tuple, to: tuple) -> list:
    """
    Shrink a star's layer (a group: every layer in it) by s about centre, which lands on `to` (option B): its pixels
    drawn again on their box so shrunk (psd_links.draw: nearest for pixel art, else bicubic, Photoshop's), a smart
    object's transform and every fxrp point mapped the same way. Returns [(leaf, its bbox before, after)].
    """
    from psd_tools.api.layers import TypeLayer      # the fork's raster writer: channels and rect, the record kept
    (cx, cy), (ox, oy) = centre, to
    at = lambda x, y: (ox + s * (float(x) - cx), oy + s * (float(y) - cy))
    out = []
    for leaf in [layer, *(layer.descendants() if layer.is_group() else ())]:
        if leaf.is_group() or not leaf.has_pixels():
            continue
        before = tuple(leaf.bbox)
        rgba = leaf.numpy()
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
            continue
        u8 = np.round(np.clip(rgba[..., :4], 0.0, 1.0) * 255).astype(np.uint8)
        x0, y0, x1, y1 = before
        corners = [v for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)) for v in at(x, y)]
        pixels, box = draw(u8, corners, Image.NEAREST if hard_edged(u8) else Image.BICUBIC)
        TypeLayer._store_raster(leaf, pixels if pixels is not None else np.zeros((0, 0, 4), np.float32), box)
        leaf._invalidate_bbox()
        for key in PLACED[:2]:
            data = leaf.tagged_blocks.get_data(key) if key in leaf.tagged_blocks else None
            for name in (b"Trnf", b"nonAffineTransform"):
                values = data.data.get(name, ()) if data is not None else ()
                for i in range(0, len(values) - 1, 2):
                    values[i].value, values[i + 1].value = at(values[i].value, values[i + 1].value)
        for key in PLACED[2:]:
            data = leaf.tagged_blocks.get_data(key) if key in leaf.tagged_blocks else None
            if data is not None:
                data.transform = tuple(v for i in range(0, len(data.transform) - 1, 2)
                                       for v in at(data.transform[i], data.transform[i + 1]))
        point = leaf.tagged_blocks.get_data(Tag.REFERENCE_POINT) if Tag.REFERENCE_POINT in leaf.tagged_blocks else None
        if point is not None:
            leaf.tagged_blocks.set_data(Tag.REFERENCE_POINT, list(at(point[0], point[1])))
        leaf._psd.mark_updated()
        out.append((leaf, before, tuple(leaf.bbox)))
    return out


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
        or _line(units, size) is not None


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
    line = _line(order, size)
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


def _stars_blocked(psd: PSDImage, moves: list, clear: list, below: bool = False, textures: bool = True,
                   follow: tuple = (), gap: int = 0, grown: Optional[dict] = None) -> Optional[str]:
    """Why stars cannot move or be copied by [(unit, dx, dy)]: a layer that cannot move or is clipped, _blocked (gap: a
    moved star's ink this many px clear of clear's too; grown: its inks so grown, kept per star), or a spot below the
    midline (no count reads it; below: a second creature's star over it); textures: off a texture clipped onto its
    group too, but one in follow (_followed: it is copied over them)."""
    for unit, dx, dy in moves:
        why = next((w for w in (_shiftable(layer) for layer in unit.layers) if w), None)
        if why:
            return f"{why}: it cannot move"
        if any(layer.clipping for layer in unit.layers):
            return f"{unit.layers[0].name!r} is clipped to the layer under it"
        if gap and (dx or dy):
            grown = {} if grown is None else grown
            if grown.get(id(unit), (None,))[0] is not unit:
                grown[id(unit)] = unit, [_grown(ink, gap) for ink in _moved_inks(unit, 0, 0)]
            inks = [(_moved_box(box, dx, dy), solid) for box, solid in grown[id(unit)][1]]
        else:
            inks = _moved_inks(unit, dx, dy)
        blocked = _blocked(psd, inks, _moved_box(unit.core, dx, dy), clear)
        if blocked:
            return f"a star at {tuple(int(v) for v in _moved_box(unit.core, dx, dy))} {blocked}" + \
                (f" (or comes within {gap} px)" if gap and (dx or dy) else "")
        if in_header(_moved_box(unit.core, dx, dy), psd.height) == below:     # the count reads no star there
            return f"a star at {tuple(int(v) for v in _moved_box(unit.core, dx, dy))} would sit " + \
                ("above" if below else "below") + " the card's middle"
    for unit, dx, dy in moves if textures else ():     # off a texture clipped onto its group, a star prints bare
        box = _moved_box(unit.core, dx, dy)
        for texture in _textures(unit):
            if any(texture is one for one in follow):
                continue
            ink = _ink(texture)
            if ink is not None and _ink_share(ink, box) < TEXTURE_KEEP:
                return f"a star at {tuple(int(v) for v in box)} would leave {texture.name!r}, clipped onto its group"
    return None


def _textures(unit: _Unit) -> list:
    """The drawn layers clipped onto a group holding the star (Yin Yarn's yarn 'Layer 6' on its Stars group)."""
    found: dict = {}
    for layer in unit.layers:
        for group in _ancestors(layer):
            for above in _clip_chain(group):
                b = tuple(above.bbox)
                if above.visible and above.kind in TEXTURE_KINDS and b[2] > b[0] and b[3] > b[1]:
                    found[id(above)] = above
    return list(found.values())


def _followed(shown: list) -> list:
    """The textures clipped onto the stars' group that follow them (owner 11-09: Yin Yarn's yarn, Paint Bomb's paint,
    Paint Warrior's Multiply paint): a plain pixel layer or smart object (no effects, filters or mask, full fill)
    inking TEXTURE_KEEP of every shown star's core in the group it clips onto."""
    out: list = []
    for unit in shown:
        for texture in _textures(_drawn_as(unit)):
            if any(texture is one for one in out) or texture.kind not in ("pixel", "smartobject") \
                    or texture.kind == "smartobject" and _filtered(texture) \
                    or texture.fill_opacity != 255 \
                    or texture.has_mask() or texture.has_vector_mask() or _enabled_effects(texture):
                continue
            base, ink = _clip_base(texture), _ink(texture)
            mine = [one for one in map(_drawn_as, shown)
                    if any(base is up for layer in one.layers for up in _ancestors(layer))]
            if ink is not None and mine and all(_ink_share(ink, one.core) >= TEXTURE_KEEP for one in mine):
                out.append(texture)
    return out


def _texture_copies(psd: PSDImage, plan: IconPlan) -> list:
    """
    Each texture plan_stars found following the stars (plan.textures) copied over the stars set_icons copied, moved or
    showed that it no longer covers (TEXTURE_KEEP of their ink): a pixel layer clipped above it (adjustments over it
    stay over), its opacity and blending, the texture as it lies on the nearest star it covers (one of that size
    first), the same place relative to the star; never where the texture or an earlier copy draws. A smart object's
    paint varies across it (Paint Bomb's): each of those stars instead gets the group as drawn under it, opaque, and a
    copy of the smart object moved as from the nearest star it covers to it, masked to the star (_paint_cover), so
    Photoshop draws its paint as on that star. Returns [(set_icons' done entry, its box)].
    """
    moved = [done[2] for done in plan.done if done[0] == "copy"] + [layer for layer, _, _ in plan.moves] + plan.show
    leaves = lambda layers: list({id(leaf): leaf for layer in layers for leaf in [layer, *(layer.descendants()
                                  if layer.is_group() else ())] if not leaf.is_group() and not leaf.clipping
                                  and leaf.is_visible() and leaf.has_pixels()}.values())
    made: list = []
    for texture in plan.textures:
        base, ink = _clip_base(texture), _ink(texture)
        if base is None or ink is None:
            continue
        under = lambda leaf: (lambda mine: mine is not None and mine[1].any()
                              and _ink_overlap(mine, ink) >= TEXTURE_KEEP * mine[1].sum())(_ink(leaf))
        mine = [leaf for leaf in leaves(moved) if any(base is up for up in _ancestors(leaf))]
        if texture.kind == "smartobject":       # the paint as a star that stays shows it, over each one
            stay = [leaf for leaf in leaves([base]) if under(leaf) and not any(leaf is one for one in mine)]
            for leaf in mine:
                cover = _paint_cover(texture, base, leaf, stay, psd)
                if cover is None:
                    continue
                (dx, dy), box, shows, drawn = cover
                pixels = np.zeros((*shows.shape, 4), np.uint8)
                pixels[shows, :3], pixels[shows, 3] = drawn[shows, :3], 255
                layer = _clipped_copy(psd, texture, pixels, box[0], box[1], BlendMode.NORMAL, 255, made,
                                      f"{texture.name} base")
                made.append((texture, layer, ("made", layer, {}), box))
                index = texture.parent.index(layer) + 1
                copy = texture.duplicate(texture.parent, index=index, name=_copy_name(_Unit([texture], box),
                                                                                     texture.parent, set()))
                same_placed(texture, copy, f"paint {leaf.name}|{dx},{dy}")
                _shift(copy, dx, dy, True)
                copy.create_mask(Image.fromarray(np.where(shows, 255, 0).astype(np.uint8), "L"), top=box[1],
                                 left=box[0])
                made.append((texture, copy, ("masked copy", texture, copy, dx, dy), box))
            continue
        bare = [leaf for leaf in mine if not under(leaf)]
        refs = [leaf for leaf in leaves([base]) if under(leaf)]
        if not bare or not refs:
            continue
        pixels = np.asarray(texture.topil().convert("RGBA"))
        taken = np.zeros((psd.height, psd.width), bool)
        tx0, ty0 = texture.left, texture.top
        h, w = pixels.shape[:2]
        cx0, cy0, cx1, cy1 = max(tx0, 0), max(ty0, 0), min(tx0 + w, psd.width), min(ty0 + h, psd.height)
        if cx1 > cx0 and cy1 > cy0:
            taken[cy0:cy1, cx0:cx1] = pixels[cy0 - ty0:cy1 - ty0, cx0 - tx0:cx1 - tx0, 3] > 0
        for leaf in bare:
            same = [ref for ref in refs if (ref.width, ref.height) == (leaf.width, leaf.height)] or refs
            ref = min(same, key=lambda one: hypot(one.left + one.width / 2 - leaf.left - leaf.width / 2,
                                                  one.top + one.height / 2 - leaf.top - leaf.height / 2))
            dx, dy = leaf.left - ref.left, leaf.top - ref.top
            x0, y0, x1, y1 = _pad(tuple(leaf.bbox), _reach(leaf), psd)
            if x1 <= x0 or y1 <= y0:
                continue
            ys, xs = np.mgrid[y0:y1, x0:x1]
            u, v = xs - dx - tx0, ys - dy - ty0     # the texture's pixel over the same spot of the reference
            inside = (u >= 0) & (v >= 0) & (u < w) & (v < h)
            patch = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
            patch[inside] = pixels[v[inside], u[inside]]
            patch[taken[y0:y1, x0:x1]] = 0
            if not patch[..., 3].any():
                continue
            taken[y0:y1, x0:x1] |= patch[..., 3] > 0
            layer = _clipped_copy(psd, texture, patch, x0, y0, texture.blend_mode, texture.opacity, made)
            made.append((texture, layer, ("made", layer, {}), tuple(layer.bbox)))
    return [(done, box) for _, _, done, box in made]


def _clipped_copy(psd: PSDImage, texture, pixels: np.ndarray, left: int, top: int, blend, opacity: int, made: list,
                  name: str = ""):
    """A pixel layer of pixels named as texture's copy, clipped right above it and the copies of it made so far."""
    from psd_tools.api.layers import PixelLayer
    name = name or f"{texture.name} copy" + (f" {len(made) + 1}" if made else "")
    layer = PixelLayer.fromarray(pixels, psd, name=name, top=top, left=left)
    layer.name = name
    layer.blend_mode, layer.opacity = blend, opacity
    _issue_id(psd, layer)
    parent = texture.parent
    layer.move_to(parent, index=parent.index(texture) + 1 + sum(1 for one in made if one[0] is texture))
    layer.clipping = True
    return layer


def _paint_cover(texture, base, leaf, refs: list, psd: PSDImage) -> Optional[tuple]:
    """
    ((dx, dy) from the nearest star of refs (one of its size first) to leaf, leaf's box, where leaf shows in it (no
    star over it), the group drawn there under texture (RGBA)); None without a star in refs or a shown pixel.
    """
    same = [ref for ref in refs if (ref.width, ref.height) == (leaf.width, leaf.height)] or refs
    if not same:
        return None
    ref = min(same, key=lambda one: hypot(one.left + one.width / 2 - leaf.left - leaf.width / 2,
                                          one.top + one.height / 2 - leaf.top - leaf.height / 2))
    x0, y0, x1, y1 = _pad(tuple(leaf.bbox), _reach(leaf), psd)
    if x1 <= x0 or y1 <= y0:
        return None
    stars = [one for one in base.descendants() if not one.is_group() and not one.clipping and one.is_visible()]
    shows = np.zeros((y1 - y0, x1 - x0), bool)
    for one in stars[next(i for i, one in enumerate(stars) if one is leaf):]:
        a = _alpha(one)
        ax0, ay0 = max(one.left, x0), max(one.top, y0)
        if a is None or min(one.left + a.shape[1], x1) <= ax0 or min(one.top + a.shape[0], y1) <= ay0:
            continue
        ax1, ay1 = min(one.left + a.shape[1], x1), min(one.top + a.shape[0], y1)
        part = a[ay0 - one.top:ay1 - one.top, ax0 - one.left:ax1 - one.left] > 0
        if one is leaf:
            shows[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] |= part
        else:                                   # a star over it keeps its own look
            shows[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] &= ~part
    if not shows.any():
        return None
    chain = _clip_chain(base)
    later = chain[chain.index(texture):]
    drawn = base.composite(viewport=(x0, y0, x1, y1), apply_icc=False,
                           layer_filter=lambda layer: not any(layer is one for one in later))
    drawn = np.asarray(drawn.convert("RGBA"))
    return (leaf.left - ref.left, leaf.top - ref.top), (x0, y0, x1, y1), shows, drawn


RELAID = ("rows of two, laid out again", "a V's point as a pair", "a row on its banner")  # _more_layouts'
OVERLAP = 0.15              # a new star's core may cover this share of another's (the cards' own shapes overlap a little)
SPARE_OVERLAP = 0.1         # a hidden spare star covering this share of a shown one (or covered) is not shown (R5)
BACKDROP_SHARE = 0.9        # a layer behind the stars inking this share of each one's core is what they sit on
BACKDROP_KEEP = 0.75        # a moved or new star keeps this share of its core on that layer
SHADE_ALPHA = (0.05, INK_ALPHA)   # see-through ink: a shadow strip over another star greys it
SHADE_RIM = 2               # px: an anti-aliased rim is no strip
SHADE_PX = 100              # px of a star under a shadow strip that _shaded allows (none on the cards' own)
COVER_MORE = 0.05           # a moved or new star may have this share more of its ink under art drawn over it than now
KEPT_TRIES = 252            # _kept_shape's most sets of stars to hide tried (10 choose 5)
REVEAL_REACH = 0.15         # a kept star's ink this share of its core's size off its core: its art (Leshy's shadow)
TEXTURE_KEEP = 0.9          # a moved or new star keeps this share of its core under a texture clipped onto its group
TEXTURE_KINDS = ("pixel", "smartobject", "shape")
PLAIN_LEVELS = 16           # a backdrop's pixels within this of their median colour (each channel, 0-255)...
PLAIN_SHARE = 0.45          # ...this share of them round the stars: a banner, not art (Bolts and Screws' gear)
BANNER_EDGES = 2            # a backdrop whose ink reaches this many canvas edges is the card's background, no banner
BANNER_GAP = 6              # px a row on its banner keeps from the text's ink and the banner's end
BANNER_FULL = 0.9           # a banner's column counts where it inks this share of the stars' middle half
BANNER_TEXT_GAP = 8         # px at least between the text and the row (else the stars' own gap)
BANNER_PITCH_MIN = 1 - OVERLAP   # star widths: a row on its banner tightens down to this pitch
FUNNEL_DROP = 0.1           # star heights a V's point drops as it becomes a pair
SHRINK_STEP = 0.05          # stars with no room for the added ones on their banner all shrink this much at a time
SHRINK_MIN = 0.5            # (owner 11-10, option B) down to this size; else off the banner as before
SHRUNK_INK_OFF = 0.005      # a shrunk layout's moved or new star: at most this share of its ink off the banner (the
#                             cards' own: none)
DICE_FACES = {1: ("middle",), 2: ("top right", "bottom left"), 3: ("top right", "middle", "bottom left")}  # kept


def _header_star_tokens(psd: PSDImage, parser: PSDParser, name: str = "") -> int:
    """Stars the parser adds from icon-font tokens in the header text (PSDParser._process_text_layer's)."""
    mid = int(psd.height * TYPE_REGION_RATIO)
    texts = [layer for layer in psd.descendants() if layer.kind == "type" and EFFECT_LAYER_TAG not in layer.name]
    titles = {id(layer) for layer in PSDParser.title_layers(psd, texts, name)}
    count = 0
    for layer in texts:
        if base_name(layer).lower() == "ability" or id(layer) in titles or not layer.is_visible() or layer.bbox[1] >= mid:
            continue
        text = layer.engine_dict["Editor"]["Text"].value
        spans = PSDParser._icon_font_spans(layer, text)
        if spans:
            icons = parser._read_icon_tokens(text, spans, CardInfo(name="", card_type="unknown"))[1]
            count += sum(1 for _, kind in icons if kind == "star")
    return count


def _loose_units(psd: PSDImage, groups: list, named: bool, types, stars=()) -> tuple:
    """(shown, hidden) stars of a card whose stars are no one group's: several Stars groups (Paint Warrior), or layers
    named star or after a star icon (stars: star_names()) in any group (Nasty Patty, Shell Shield, Warpstars)."""
    if not named:
        shown, hidden = [], []
        for group in groups:
            one, other = _star_units(group, types)
            shown, hidden = shown + one, hidden + other
        return shown, hidden
    out: dict = {True: [], False: []}
    by_parent: dict = {}
    icons = {id(layer) for layer in PSDParser.outside_stars(psd, (), stars)}
    for layer in psd.descendants():
        if (PSDParser.named_star(layer) or id(layer) in icons) and all(p.visible for p in _ancestors(layer)) \
                and star_leaf(layer):
            by_parent.setdefault(id(layer.parent), []).append(layer)
    for layers in by_parent.values():
        for on in (True, False):
            some = [layer for layer in layers if layer.visible == on]
            boxes = [PSDParser._core_box(layer) for layer in some]
            out[on] += [_Unit([some[i] for i in members], union_box([boxes[i] for i in members]))
                        for members in star_sets(boxes)]
    return out[True], out[False]


def _flat_units(shown: list, hidden: list, types, below: bool = False) -> tuple:
    """Stars inside a shown group drawing several (Cat's "Group 1" of two) as their own units, its hidden ones too."""
    while any(unit.count > 1 and unit.layers[0].is_group() for unit in shown):
        unit = next(unit for unit in shown if unit.count > 1 and unit.layers[0].is_group())
        inner, spare = _star_units(unit.layers[0], types, below)
        shown = [one for one in shown if one is not unit] + inner
        hidden = hidden + spare
    return shown, hidden


def _same_spot(a: _Unit, b: _Unit) -> bool:
    """Two stars drawn on one spot (a stacked copy): their cores nearly the same box."""
    tol = 0.15 * max(a.size + b.size)
    return all(abs(x - y) <= tol for x, y in zip(a.core, b.core))


def _merge_twins(units: list) -> list:
    """Stars drawn twice on one spot (a stacked copy: Tainted Eden's 'Stars copy', Living Bread's second 'stars') as one
    star whose layers hide, move and are copied together (R3); the count reads them as one too (star_sets)."""
    out: list = []
    for unit in units:
        twin = next((one for one in out if one.count == 1 and unit.count == 1 and _same_spot(one, unit)), None)
        if twin is None:
            out.append(_Unit(list(unit.layers), unit.core, unit.count))
        else:
            twin.layers = twin.layers + [layer for layer in unit.layers if layer not in twin.layers]
    return out


def _outside_units(layers: list) -> list:
    """The star-icon layers outside the Stars groups (PSDParser.outside_stars: Nosk's and Massive Moss Charger's 'Hk
    star') as stars, clustered per parent as the count judges them: laid out with the group's (R2)."""
    by_parent: dict = {}
    for layer in layers:
        by_parent.setdefault(id(layer.parent), []).append(layer)
    out = []
    for some in by_parent.values():
        boxes = [PSDParser._core_box(layer) for layer in some]
        out += [_Unit([some[i] for i in members], union_box([boxes[i] for i in members]))
                for members in star_sets(boxes)]
    return out


@dataclass
class _Turned(_Unit):
    """A star of a set drawn upside down below the midline (a second creature's: Tainted Lazarus's dead form) turned half
    round about the canvas's centre, where the layouts read it as a header star; real: the star as drawn."""
    real: Optional[_Unit] = None
    turned: tuple = ()

    @property
    def box(self) -> tuple:
        return self.turned


def _turn_box(box, psd: PSDImage) -> tuple:
    """A canvas box turned half round about the canvas's centre."""
    return psd.width - box[2], psd.height - box[3], psd.width - box[0], psd.height - box[1]


def _turned(units: list, psd: PSDImage) -> list:
    return [_Turned(unit.layers, _turn_box(unit.core, psd), unit.count, unit, _turn_box(unit.box, psd)) for unit in units]


def _drawn_as(unit: _Unit) -> _Unit:
    """The star as drawn (a turned one's real)."""
    return unit.real if isinstance(unit, _Turned) else unit


def _as_drawn(moves: list) -> list:
    """[(unit, dx, dy, ...)] with turned units as drawn: their offsets turned back."""
    return [(unit.real, -dx, -dy, *rest) if isinstance(unit, _Turned) else (unit, dx, dy, *rest)
            for unit, dx, dy, *rest in moves]


def _second_set(groups: list, types) -> Optional[tuple]:
    """(group, shown, hidden) of a second creature's stars drawn upside down below the midline (Tainted Lazarus's dead
    form): the one Stars group drawing stars there and none above it; None otherwise."""
    found = [group for group in groups if _star_units(group, types, True)[0] and not _star_units(group, types)[0]]
    if len(found) != 1:
        return None
    shown, hidden = _flat_units(*_star_units(found[0], types, True), types, True)
    return found[0], shown, hidden


def _centred_row(units: list, size: tuple, width: int) -> Optional[tuple]:
    """(y, pitch) when the stars are a row of two or more centred on the card (Grimora's three); None otherwise."""
    line = _line(units, size)
    if line is None:
        return None
    (x, _), (dx, dy) = line
    tol = _tol(max(size))
    if abs(dy) > tol or dx >= 0 or abs(x + (len(units) - 1) * dx / 2 - width / 2) > tol:
        return None
    return sum(unit.centre[1] for unit in units) / len(units), -dx


def _centred_spots(model: tuple, count: int, width: int) -> list:
    """The centres of `count` stars in a row centred on the card at the model's pitch (_centred_row), right to left."""
    y, pitch = model
    return [(width / 2 + ((count - 1) / 2 - i) * pitch, y) for i in range(count)]


def _skewed_pairs(spots: list, height: float, tol: float) -> bool:
    """Rows of two whose columns do not line up (a 2 over 1 carried on: Polevaulter Zombie's skewed 2 over 2)."""
    rows = _rows([_Unit([], (x, y, x, y)) for x, y in spots], height, True)
    return len(rows) > 1 and all(len(row) == 2 for row in rows) and any(
        abs(row[k].centre[0] - rows[0][k].centre[0]) > tol for row in rows for k in (0, 1))


def _splittable(layer) -> Optional[str]:
    """Why one layer drawing several stars cannot be split into masked copies (split_stars); None when it can: a pixel
    layer or smart object with no mask, no layers clipped to it, no effects and no smart filters."""
    if layer.is_group() or layer.kind not in ("pixel", "smartobject"):
        return f"it is a {'group' if layer.is_group() else layer.kind + ' layer'}"
    if layer.has_mask() or layer.has_vector_mask():
        return "it has a mask"
    if _clip_chain(layer) or _enabled_effects(layer) or _filtered(layer):
        return "it has clipped layers, effects or smart filters"
    return None


def _split_masks(layer, pieces: list) -> list:
    """One mask per star over the layer's bbox (255 shown): each pixel goes to the star whose box is nearest (its glow
    and shadow with it), so the masks together show the whole layer and the split alone changes no pixel."""
    x0, y0, x1, y1 = layer.bbox
    ys, xs = np.ogrid[y0:y1, x0:x1]
    far = [np.maximum(np.maximum(bx0 - xs, xs - (bx1 - 1)), 0) ** 2 + np.maximum(np.maximum(by0 - ys, ys - (by1 - 1)), 0) ** 2
           for bx0, by0, bx1, by1 in pieces]
    owner = np.argmin(np.stack(np.broadcast_arrays(*far)), axis=0)
    return [np.where(owner == k, 255, 0).astype(np.uint8) for k in range(len(pieces))]


def split_stars(layer) -> list:
    """
    One layer drawing several stars split into masked copies, one per star (R4: Grimora's row, Dimension D's hearts):
    it keeps the first star in reading order and a copy above it ("<name> copy", Photoshop's names) takes each other,
    each masked to its star's share of the layer (_split_masks).

    Returns:
        set_icons()' done entries: ("split", layer, copy) per copy, then ("masked", layer)
    """
    units = [_Unit([layer], box) for box in star_pieces(layer)]
    pieces = [unit.core for unit in _reading(units, _median_size(units)[1], True)]
    masks = _split_masks(layer, pieces)
    x0, y0 = layer.bbox[:2]
    parent, taken, done = layer.parent, set(), []
    for k in range(1, len(pieces)):
        copy = layer.duplicate(parent, index=parent.index(layer) + k, name=_copy_name(units[0], parent, taken))
        same_placed(layer, copy, f"split {layer.name}|{k}")
        copy.create_mask(Image.fromarray(masks[k], "L"), top=y0, left=x0)
        done.append(("split", layer, copy))
    layer.create_mask(Image.fromarray(masks[0], "L"), top=y0, left=x0)
    done.append(("masked", layer))
    return done


def _reopened(psd: PSDImage) -> PSDImage:
    """The card as it stands, saved and opened again: plan_stars splits a layer there when it may not change psd."""
    buf = BytesIO()
    psd.save(buf, update_preview=False)
    return PSDImage.open(BytesIO(buf.getvalue()))


def _covers(a: tuple, b: tuple) -> float:
    """The share of box b that box a covers."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h / max(1, (b[2] - b[0]) * (b[3] - b[1])) if w > 0 and h > 0 else 0.0


def _kept_shape(order: list, size: tuple, count: int) -> Optional[list]:
    """
    The stars to hide, 3 or more staying, when neither end of the reading order leaves a layout: the latest ones whose
    rest is a known layout, else whose rest is mirrored about the shape's middle (a dice 5 loses its middle: Nightmare,
    Kyle's 2 over 3, The Master's), none that bares a kept star's shadow (_bares: Leshy's); None.
    """
    drop = len(order) - count
    if comb(len(order), drop) > KEPT_TRIES:
        return None
    tol = 2 * _tol(max(size))                   # hand-placed: Zero Override's bottom row is 14 px off level
    middle = (min(u.core[0] for u in order) + max(u.core[2] for u in order)) / 2
    mirrored = lambda kept: all(any(abs(2 * middle - a.centre[0] - b.centre[0]) <= tol and abs(a.centre[1] - b.centre[1])
                                    <= tol for b in kept) for a in kept)
    tries = [set(gone) for gone in combinations(range(len(order) - 1, -1, -1), drop)]
    alphas: dict = {}
    for good in (lambda kept: _known_layout(kept, size), mirrored):
        for gone in tries:
            kept = [unit for i, unit in enumerate(order) if i not in gone]
            hidden = [unit for i, unit in enumerate(order) if i in gone]
            if good(kept) and not _bares(hidden, kept, alphas):
                return hidden
    return None


def _dice_five(units: list, size: tuple) -> Optional[dict]:
    """
    The stars by spot ("top right", "middle", "bottom left"...) when they are a die's five: four corners in two level
    rows whose columns line up, the fifth on their middle (Nightmare, The Lich King; hand-placed: twice _tol); None.
    """
    if len(units) != 5:
        return None
    tol = 2 * _tol(max(size))
    mx, my = sum(unit.centre[0] for unit in units) / 5, sum(unit.centre[1] for unit in units) / 5
    middle = min(units, key=lambda unit: (unit.centre[0] - mx) ** 2 + (unit.centre[1] - my) ** 2)
    rest = sorted((unit for unit in units if unit is not middle), key=lambda unit: unit.centre[1])
    top, bottom = (sorted(row, key=lambda unit: -unit.centre[0]) for row in (rest[:2], rest[2:]))
    if any(abs(row[0].centre[1] - row[1].centre[1]) > tol for row in (top, bottom)) \
            or any(abs(top[k].centre[0] - bottom[k].centre[0]) > tol for k in (0, 1)) \
            or bottom[0].centre[1] - top[0].centre[1] < size[1] / 2 \
            or top[0].centre[0] - top[1].centre[0] < size[0] / 2:
        return None
    if abs(middle.centre[0] - sum(unit.centre[0] for unit in rest) / 4) > tol \
            or abs(middle.centre[1] - sum(unit.centre[1] for unit in rest) / 4) > tol:
        return None
    return {"top right": top[0], "top left": top[1], "middle": middle, "bottom right": bottom[0],
            "bottom left": bottom[1]}


def _bares(gone: list, kept: list, alphas: dict) -> bool:
    """
    Whether hiding the gone stars bares more than SHADE_PX of a kept star's ink (any alpha) under their solid ink and
    away from its own core (grown by REVEAL_REACH: no glow round it): Leshy's middle drop over its neighbour's shadow
    strip. alphas: id(layer) -> _alpha.
    """
    def inks(unit):
        out = []
        for leaf in [leaf for layer in unit.layers for leaf in _drawn(layer)]:
            if id(leaf) not in alphas:
                alphas[id(leaf)] = _alpha(leaf)
            if alphas[id(leaf)] is not None and alphas[id(leaf)].size:
                out.append((tuple(leaf.bbox), alphas[id(leaf)]))
        return out
    over = [one for unit in gone for one in inks(unit)]
    for unit in kept:
        x0, y0, x1, y1 = unit.core
        gx, gy = REVEAL_REACH * (x1 - x0), REVEAL_REACH * (y1 - y0)
        for (bx, by, _, _), alpha in inks(unit):
            ys, xs = np.ogrid[by:by + alpha.shape[0], bx:bx + alpha.shape[1]]
            away = (alpha > SHADE_ALPHA[0]) & ~((xs >= x0 - gx) & (xs < x1 + gx) & (ys >= y0 - gy) & (ys < y1 + gy))
            for (hx, hy, _, _), theirs in over:
                cx0, cy0 = max(bx, hx), max(by, hy)
                cx1, cy1 = min(bx + alpha.shape[1], hx + theirs.shape[1]), min(by + alpha.shape[0], hy + theirs.shape[0])
                if cx1 <= cx0 or cy1 <= cy0:
                    continue
                solid = theirs[cy0 - hy:cy1 - hy, cx0 - hx:cx1 - hx] > 1 - SHADE_ALPHA[0]
                if int((away[cy0 - by:cy1 - by, cx0 - bx:cx1 - bx] & solid).sum()) > SHADE_PX:
                    return True
    return False


def _shade(lower: list, upper: list) -> int:
    """
    Pixels where an upper layer's see-through ink (a shadow strip, SHADE_ALPHA, not its anti-aliased rim: SHADE_RIM px
    in) lies on a lower layer's solid ink; lower and upper are [(bbox, alpha)].
    """
    total = 0
    for (lx0, ly0, _, _), la in lower:
        for (ux0, uy0, _, _), ua in upper:
            x0, y0 = max(lx0, ux0), max(ly0, uy0)
            x1, y1 = min(lx0 + la.shape[1], ux0 + ua.shape[1]), min(ly0 + la.shape[0], uy0 + ua.shape[0])
            if x1 <= x0 or y1 <= y0:
                continue
            thin = (ua > SHADE_ALPHA[0]) & (ua < SHADE_ALPHA[1])
            core = thin.copy()
            for k in range(1, SHADE_RIM + 1):
                core[k:] &= thin[:-k]
                core[:-k] &= thin[k:]
                core[:, k:] &= thin[:, :-k]
                core[:, :-k] &= thin[:, k:]
            solid = la[y0 - ly0:y1 - ly0, x0 - lx0:x1 - lx0] > 1 - SHADE_ALPHA[0]
            total += int((solid & core[y0 - uy0:y1 - uy0, x0 - ux0:x1 - ux0]).sum())
    return total


def _shaded(psd: PSDImage, kept: list, moves: list) -> list:
    """
    The moves (unit, dx, dy) that put a star under another star's shadow: more than SHADE_PX of its solid ink under the
    see-through ink of a star drawn above it, past what lies there now (Flower of Decay's rows of two: the moved star
    under its neighbour's shadow strip, a grey band).
    """
    at = {id(layer): i for i, layer in enumerate(psd.descendants())}
    moved = {id(unit): (dx, dy) for unit, dx, dy in moves}
    def inks(unit, shift):
        out = []
        for leaf in [leaf for layer in unit.layers for leaf in _drawn(layer)]:
            alpha = _alpha(leaf)
            if alpha is not None and alpha.size:
                out.append((_moved_box(tuple(leaf.bbox), *shift), alpha))
        return out
    out = []
    for unit, dx, dy in moves:
        top = max(at[id(layer)] for layer in unit.layers)
        for other in kept:
            if other is unit or min(at[id(layer)] for layer in other.layers) < top:
                continue
            shift = moved.get(id(other), (0, 0))
            now = _shade(inks(unit, (0, 0)), inks(other, (0, 0)))
            if _shade(inks(unit, (dx, dy)), inks(other, shift)) > now + SHADE_PX:
                out.append((unit, dx, dy))
                break
    return out


def _covered(psd: PSDImage, moves: list, skip: set, cache: dict) -> Optional[str]:
    """
    Why a moved or copied star [(unit, dx, dy)] would sit under art drawn over it: more than COVER_MORE more of its solid
    ink under the solid ink of a visible normal layer above it (skip: ids of the stars' own layers) than where it is now
    (Massive Moss Charger's 'Hk star' copy under the moss bush); None. cache: id(layer) -> _ink, per plan.
    """
    layers = list(psd.descendants())
    at = {id(layer): i for i, layer in enumerate(layers)}

    def ink(layer):
        if id(layer) not in cache:
            found = _ink(layer)
            if found is not None and layer.opacity < 255:
                found = (found[0], _alpha(layer) * layer.opacity / 255 > INK_ALPHA)
            cache[id(layer)] = found
        return cache[id(layer)]

    def under(stars: list, over: list) -> int:
        hidden = 0
        for box, solid in stars:
            mask = np.zeros_like(solid)
            for (x0, y0, _, _), theirs in over:
                cx0, cy0 = max(box[0], x0), max(box[1], y0)
                cx1, cy1 = min(box[0] + solid.shape[1], x0 + theirs.shape[1]), min(box[1] + solid.shape[0], y0 + theirs.shape[0])
                if cx1 > cx0 and cy1 > cy0:
                    mask[cy0 - box[1]:cy1 - box[1], cx0 - box[0]:cx1 - box[0]] |= theirs[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
            hidden += int((mask & solid).sum())
        return hidden
    for unit, dx, dy in moves:
        if not (dx or dy):
            continue
        own = [one for layer in unit.layers for one in [layer, *(layer.descendants() if layer.is_group() else ())]]
        top = max(at[id(one)] for one in own)
        now, new = _moved_inks(unit, 0, 0), _moved_inks(unit, dx, dy)
        total = sum(int(solid.sum()) for _, solid in now)
        if not total:
            continue
        area = union_box([box for box, _ in now + new])
        over = []
        for layer in layers[top + 1:]:
            b = tuple(layer.bbox)
            if (id(layer) in skip or layer.is_group() or layer.clipping or _is_text_layer(layer) or not layer.is_visible()
                    or layer.blend_mode not in (BlendMode.NORMAL, BlendMode.DISSOLVE) or b[2] <= area[0]
                    or b[0] >= area[2] or b[3] <= area[1] or b[1] >= area[3] or not layer.has_pixels()):
                continue
            found = ink(layer)
            if found is not None:
                over.append((layer, found))
        if not over:
            continue
        was, will = under(now, [one for _, one in over]), under(new, [one for _, one in over])
        if will - was > COVER_MORE * total:
            worst = max(over, key=lambda one: under(new, [one[1]]))[0]
            return (f"a star at {tuple(int(v) for v in _moved_box(unit.core, dx, dy))} would sit under {worst.name!r} "
                    f"({round(100 * will / total)}% of it hidden, {round(100 * was / total)}% now)")
    return None


def _ink_share(ink: tuple, box: tuple) -> float:
    """The share of box that a layer's ink ((bbox, solid alpha), _ink) covers."""
    (x0, y0, _, _), solid = ink
    bx0, by0, bx1, by1 = (int(round(v)) for v in box)
    area = max(1, (bx1 - bx0) * (by1 - by0))
    part = solid[max(0, by0 - y0):max(0, by1 - y0), max(0, bx0 - x0):max(0, bx1 - x0)]
    return float(part.sum()) / area


def _backdrop(psd: PSDImage, shown: list) -> Optional[tuple]:
    """
    (layer, ink) of what the stars sit on: the smallest drawn layer behind them, under half the canvas, whose ink holds
    BACKDROP_SHARE of every shown star's core (Remote Detonator's paper banner), if plain there (_plain_banner) and
    drawn on the card (its ink reaching under BANNER_EDGES of the canvas's edges: Spiky Ball Trap's brick wall and
    Forgotten Crossroads' gradient are background); None.
    """
    layers = list(psd.descendants())
    at = {id(layer): i for i, layer in enumerate(layers)}
    first = min(at[id(layer)] for unit in shown for layer in unit.layers)
    box = union_box([unit.core for unit in shown])
    found = []
    for layer in layers[:first]:
        b = tuple(layer.bbox)
        if (layer.is_group() or layer.kind == "type" or not layer.is_visible() or b[0] > box[0] or b[1] > box[1]
                or b[2] < box[2] or b[3] < box[3] or (b[2] - b[0]) * (b[3] - b[1]) > psd.width * psd.height / 2):
            continue
        found.append(((b[2] - b[0]) * (b[3] - b[1]), layer))
    for _, layer in sorted(found, key=lambda one: one[0]):
        ink = _ink(layer)
        if ink is not None and all(_ink_share(ink, unit.core) >= BACKDROP_SHARE for unit in shown):
            banner = _plain_banner(layer, shown) and _canvas_edges(psd, ink) < BANNER_EDGES
            return (layer, ink) if banner else None     # art (a gear) or the background is no banner
    return None


def _canvas_edges(psd: PSDImage, ink: tuple) -> int:
    """How many of the canvas's edges an ink ((bbox, solid), _ink) reaches or runs past."""
    (x, y, _, _), solid = ink
    rows, cols = np.flatnonzero(solid.any(axis=1)), np.flatnonzero(solid.any(axis=0))
    if not rows.size:
        return 0
    return int(x + cols[0] <= 0) + int(y + rows[0] <= 0) + int(x + cols[len(cols) - 1] + 1 >= psd.width) + \
        int(y + rows[len(rows) - 1] + 1 >= psd.height)


def _plain_banner(layer, shown: list) -> bool:
    """Whether a backdrop is plain round the stars (their box grown by half a star): PLAIN_SHARE of its solid pixels
    within PLAIN_LEVELS of their median colour (a banner, not Bolts and Screws' gear); True when unread."""
    box = union_box([unit.core for unit in shown])
    w = max(unit.core[2] - unit.core[0] for unit in shown) // 2
    h = max(unit.core[3] - unit.core[1] for unit in shown) // 2
    b = tuple(layer.bbox)
    alpha = _alpha(layer)
    try:
        pixels = layer.numpy()
    except Exception:
        return True
    if alpha is None or pixels is None or pixels.ndim != 3 or pixels.shape[2] < 3:
        return True
    x0, y0 = int(max(0, box[0] - w - b[0])), int(max(0, box[1] - h - b[1]))
    x1, y1 = int(max(0, box[2] + w - b[0])), int(max(0, box[3] + h - b[1]))
    rgb = pixels[y0:y1, x0:x1, :3][alpha[y0:y1, x0:x1] > INK_ALPHA] * 255
    if not len(rgb):
        return True
    median = np.median(rgb, axis=0)
    return float((np.abs(rgb - median).max(axis=1) <= PLAIN_LEVELS).mean()) >= PLAIN_SHARE


def _off_backdrop(back: Optional[tuple], moves: list, copies: list, shrunk: bool = False) -> Optional[str]:
    """Why a moved or copied star leaves the stars' backdrop (_backdrop): under BACKDROP_KEEP of its core on it (shrunk:
    under BACKDROP_SHARE, as the card's own, or over SHRUNK_INK_OFF of its ink off it); None."""
    if back is None:
        return None
    for unit, dx, dy, *_ in moves + copies:
        box = _moved_box(unit.core, dx, dy)
        if (dx or dy) and (_ink_share(back[1], box) < (BACKDROP_SHARE if shrunk else BACKDROP_KEEP)
                           or shrunk and _ink_off(back[1], _moved_inks(unit, dx, dy)) > SHRUNK_INK_OFF):
            return f"a star at {tuple(int(v) for v in box)} would leave {back[0].name!r} under the stars"
    return None


def _more_layouts(order: list, size: tuple, count: int, psd: PSDImage, back: Optional[tuple] = None,
                  clear: list = ()) -> list:
    """
    More layouts for stars in a shape _star_layouts does not carry on (LAYOUT_SUPPORT.md): a V's point becomes a pair
    (_funnel: Crowned Magolor's); its shortest row grows at the end that keeps the shape centred (Lorian's 2 over 3,
    Springtrap's), each new star clear of the others and on the card; else, or when the stars overlap (a dice face),
    every star laid out again as rows of two from the shape's top-right star (God Tamer's and Michael's Blessing's dice:
    their six; Morpho Knight's 2-1-2). Rows of two whose columns do not line up once carried on (the zombies' 2
    over 1) try the rows of two again first. Last, a row on its backdrop (back, _backdrop; clear: the text and icons
    it keeps from) centred and tightened there (_banner_row: Remote Detonator's).
    """
    out = []
    if count <= len(order):
        return out
    funnel, banner = _funnel(order, size, count), _banner_row(order, size, count, psd, back, clear)
    w, h = size
    rows = _rows(order, h, False)
    pitches = sorted(row[i + 1].centre[0] - row[i].centre[0] for row in rows for i in range(len(row) - 1))
    pitch = max(pitches[len(pitches) // 2] if pitches else 0, w * 1.05)
    state = [[unit.centre for unit in row] for row in rows]
    cores = [unit.core for unit in order]
    middle = sum(unit.centre[0] for unit in order) / len(order)
    slack = _tol(max(size))
    on_card = lambda box: box[0] >= -slack and box[1] >= -slack and box[2] <= psd.width + slack and box[3] <= psd.height + slack
    packed = any(sum(_covers(a.core, b.core) > OVERLAP / 3 for b in order if b is not a) >= 3 for a in order)  # a dice's middle
    new = [] if not packed else None
    for _ in range(count - len(order) if not packed else 0):
        target = min(range(len(state)), key=lambda i: (len(state[i]), -i))
        row = state[target]
        y = sum(c[1] for c in row) / len(row)
        xs = [c[0] for c in row]
        choices = sorted([min(xs) - pitch, max(xs) + pitch], key=lambda x: abs((sum(xs) + x) / (len(xs) + 1) - middle))
        for x in choices:
            core = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
            if on_card(core) and all(_covers(core, other) <= OVERLAP and _covers(other, core) <= OVERLAP for other in cores):
                row.append((x, y))
                cores.append(core)
                new.append((x, y))
                break
        else:
            new = None
            break
    if new:
        out.append(("its rows carried on", [unit.centre for unit in order] + new))
    tops = sorted(order, key=lambda unit: (unit.centre[1], -unit.centre[0]))
    right, top = max(unit.centre[0] for unit in order), tops[0].centre[1]
    ys = sorted({round(sum(u.centre[1] for u in row) / len(row)) for row in rows})
    steps = [b - a for a, b in zip(ys, ys[1:]) if b - a >= h * 0.9]
    relaid = (RELAID[0], _pyramid_spots((right, right - pitch, top, min(steps) if steps else h * 1.05, False), count))
    if new and _skewed_pairs(out[0][1], h, slack):     # skewed rows of two: the templates' rows of two first (owner
        return funnel + [relaid] + out + banner     # 2026-10-26, Polevaulter Zombie), then carried on
    return funnel + out + [relaid] + banner


def _funnel(order: list, size: tuple, count: int) -> list:
    """
    One star more on a V (rows of two narrowing down to one centred star: Crowned Magolor's round its light icon): its
    point becomes a touching pair centred where it was, FUNNEL_DROP of a star lower, so the V still narrows; [] for
    another shape or count, or when the pair would cover more than OVERLAP of a star.
    """
    w, h = size
    rows = _rows(order, h, True)
    last = len(rows) - 1
    if count != len(order) + 1 or last < 2 or len(rows[last]) != 1 or any(len(row) != 2 for row in rows[:last]):
        return []
    tol = _tol(max(size))
    (x, y), widths = rows[last][0].centre, [row[0].centre[0] - row[1].centre[0] for row in rows[:last]]
    if any(abs(row[0].centre[1] - row[1].centre[1]) > tol or abs((row[0].centre[0] + row[1].centre[0]) / 2 - x) > tol
           for row in rows[:last]) or any(lower > upper - tol for upper, lower in zip(widths, widths[1:])) \
            or widths[last - 1] - tol <= w:
        return []
    pair = [(x + w / 2, y + FUNNEL_DROP * h), (x - w / 2, y + FUNNEL_DROP * h)]
    cores = [(px - w / 2, py - h / 2, px + w / 2, py + h / 2) for px, py in pair]
    if any(_covers(core, unit.core) > OVERLAP or _covers(unit.core, core) > OVERLAP for core in cores for unit in order
           if unit is not rows[last][0]):
        return []
    return [(RELAID[1], [unit.centre for row in rows[:last] for unit in row] + pair)]


def _banner_row(order: list, size: tuple, count: int, psd: PSDImage, back: Optional[tuple], clear: list) -> list:
    """
    One row of stars on its backdrop (back: _backdrop's (layer, ink)) at `count`: the free run along the backdrop's
    middle half at the stars' height (its columns BANNER_FULL inked, BANNER_GAP in from its end and from the text's ink
    left of the stars, then the stars' own gap, BANNER_TEXT_GAP at least); centred there at the card's pitch, else
    tightened to fill it down to BANNER_PITCH_MIN; [] for no backdrop, more than one row, or no room.
    """
    if back is None or count <= len(order) or len(_rows(order, size[1], True)) != 1:
        return []
    w, h = size
    (bx, by, _, _), solid = back[1]
    y = sum(unit.centre[1] for unit in order) / len(order)
    band = solid[max(0, int(y - h / 4) - by):max(0, int(y + h / 4) - by)]
    full = np.nonzero(band.mean(axis=0) >= BANNER_FULL)[0] if band.size else []
    if not len(full):
        return []
    left, right = bx + int(full.min()) + BANNER_GAP, min(bx + int(full.max()) + 1, psd.width) - BANNER_GAP
    y0, y1 = min(unit.core[1] for unit in order), max(unit.core[3] for unit in order)
    first = min(unit.core[0] for unit in order)
    for layer in clear:
        ink = _ink(layer)
        if ink is None:
            continue
        (tx, ty, _, _), theirs = ink
        inked = np.nonzero(theirs[max(0, int(y0) - ty):max(0, int(y1) - ty)].any(axis=0))[0]
        if len(inked) and tx + int(inked.max()) < first + w:          # text left of the stars, in their rows
            left = max(left, tx + int(inked.max()) + 1 + BANNER_GAP)
    pitches = sorted(order[i].centre[0] - order[i + 1].centre[0] for i in range(len(order) - 1))
    pitch = pitches[len(pitches) // 2] if pitches else w * 1.05
    left += max(BANNER_TEXT_GAP, pitch - w)
    span = right - left
    if (count - 1) * pitch + w <= span:
        x = right - (span - (count - 1) * pitch - w) / 2 - w / 2
    else:
        pitch = (span - w) / (count - 1)
        if pitch < BANNER_PITCH_MIN * w:
            return []
        x = right - w / 2
    return [(RELAID[2], [(x - i * pitch, y) for i in range(count)])]


class _Shrink(Exception):
    """A stars plan's added stars fit nowhere on their banner, full size: _shrunk tries them all smaller (option B)."""
    def __init__(self, shown: list, hidden: list, banner: tuple):
        super().__init__("stars: shrink to fit the banner")
        self.shown, self.hidden, self.banner = shown, hidden, banner


def plan_stars(psd: PSDImage, parsed: dict, value: int, parser: Optional[PSDParser] = None,
               split: bool = False) -> IconPlan:
    """
    How to show `value` stars on a card whose stars the parser counts from its PSD (minions, aux items and creatures,
    N.M.E.; elsewhere the "N Stars" folder sets them). In the one Stars group that draws them:

    - in the group holding the stars one by one (in its one shown count folder, "3 stars", when it has those): fewer
      hide the last in reading order (rows top to bottom, each right to left; the first when only that leaves a
      layout; none that bares a kept star's shadow: _bares), a die's five keeps the die's faces (DICE_FACES); more show
      the card's own hidden stars first, then copy
      the last star layer at the layout's spacing (_star_layouts, the first that stays clear): rows of two with an odd
      one centred under or over them (the templates; the odd one moves aside when a pair fills its row, and back to
      the middle when one goes), a row or column carried on, a zigzag carried on, a lone star's column, then
      _more_layouts'. A row centred on the card stays centred at its pitch (Grimora's). A new or moved star must stay on
      the canvas, off the text and type icons, and add exactly one to what the parser counts; it should stay under a
      texture clipped onto its group (but one that follows them: _followed, copied by set_icons), on its backdrop and
      out from under art. When no layout fits, the first that fits stepped down (_drops: its new stars, else all it
      moves and adds); then every star shrunk evenly on the banner they sit on (_backdrop) until one fits there at its
      spots, as fully on it as the card's own and a banner row's gap clear of the text (owner 11-10, option B:
      _shrunk); then both off the banner; else the first that fits on the canvas off the text and
      icons alone, at its spot or stepped down (owner 11-08: a developer fixes its look; all told); a refusal gives the
      reason of the layout that came closest.
    - star-icon layers outside the group (Nosk's 'Hk star') are stars of the layout (R2); a star drawn twice on one
      spot is one star (R3: Tainted Eden, Living Bread); one layer drawing several is split into masked copies first
      (R4: split_stars; Grimora, Dimension D); a second creature's stars upside down below the midline (Tainted
      Lazarus's dead form) take the same count, laid out turned half round, or neither set is redrawn.

    Args:
        psd: The open card (not changed, but split: a layer drawing several stars is split in it)
        parsed: parse_card() of it
        value: The stars to show, 1 or more
        parser: offline_parser(): a layer in the Stars group named after a type icon is no star to hide or copy
        split: May split a layer in psd (apply_jobs, on its working copy); else the plan is made on a copy of the card

    Returns:
        IconPlan; raises Refused("unsupported-card", field="stars") when the layout is not one of those
    """
    try:
        return _plan_stars(psd, parsed, value, parser, split)
    except _Shrink as wanted:                   # option B, else off the banner as before (told)
        return _shrunk(psd, parsed, value, parser, wanted) or _plan_stars(psd, parsed, value, parser, split, 0.0)


def _plan_stars(psd: PSDImage, parsed: dict, value: int, parser: Optional[PSDParser] = None, split: bool = False,
                shrink: Optional[float] = None) -> IconPlan:
    """
    plan_stars' work. shrink: None, a layout that fits nowhere on the stars' banner raises _Shrink before going off it;
    a factor, the stars were shrunk by it (_shrunk's copy): at a layout's spots on their banner, as fully as the
    card's own and a banner row's gap clear of the text, or refused; 0, off it as before.
    """
    def refuse(why: str):
        raise Refused("unsupported-card", f"stars: {why}", "stars")
    info, old = parsed.get("info"), parsed["card"].get("stars")
    rel = info.path if info is not None else parsed["card"].get("path", "")
    if not stars_from_psd(rel):
        folder = rel.split("/")[len(rel.split("/")) - 2] if "/" in rel else ""
        if not fullmatch(STAR_FOLDER, folder, IGNORECASE):
            refuse(f"the parser reads no stars on a {parsed['card'].get('type') or 'card'} card")
    if value < 1:
        refuse("a card shows at least one")
    types = parser.all_types if parser is not None else ()
    stars = parser.star_names() if parser is not None else ()
    groups = [group for group in PSDParser._star_groups(psd) if PSDParser.stars_in(group, types)]
    # the parser's fallback (layers named "... Star ..." or after a star icon); a folder card's, or one beside an empty
    # Stars group (Double Pain's), when they add up
    named = not groups and ((stars_from_psd(rel) and not PSDParser._star_groups(psd))
                            or bool(old) and PSDParser.stars_drawn(psd, stars=stars) == old)
    if not groups and not named:
        refuse('no group named "Stars" draws them')
    total = PSDParser.stars_drawn(psd, () if named else types, stars)
    if total != old:
        tokens = _header_star_tokens(psd, parser, info.name if info is not None else "") if parser is not None else 0
        refuse(f"the parser counts {old}, the Stars group draws {total}" + (f" and the header text {tokens}" if tokens else ""))
    below = PSDParser.stars_below(psd, types, stars)
    second = _second_set(groups, types) if below else None
    if below:                                   # Tainted Lazarus's dead form: the same count, turned half round
        if second is None or below != old or len(second[1]) != below or any(unit.count != 1 for unit in second[1]):
            refuse(f"the card draws {below} more below its middle, not one second creature's {old}")
        groups = [group for group in groups if group is not second[0]]
    outside = [] if named else PSDParser.outside_stars(psd, PSDParser._star_groups(psd), stars)
    plan = IconPlan("stars", old, value)
    if named or len(groups) > 1:                # Nasty Patty's Patrick Stars, Paint Warrior's two Stars groups
        shown, hidden = _loose_units(psd, groups, named, types, stars)
        if not shown:
            refuse('no group named "Stars" draws them')
        container = shown[len(shown) - 1].layers[0].parent
    else:
        container = groups[0] if groups else None
    while container is not None and not named and len(groups) == 1:
        folders: dict = {}
        for child in container:
            found = fullmatch(STAR_FOLDER, child.name, IGNORECASE) if child.is_group() else None
            if found:
                folders.setdefault(int(found.group(1)), []).append(child)
        if folders and not outside and second is None:
            shown = [f for group in folders.values() for f in group if f.visible]
            if len(shown) != 1:
                refuse(f"{len(shown)} shown star folders in {container.name!r}")
            container = shown[0]                # the shown folder's stars, one by one (a '3 stars' drawing 4)
            continue
        shown, hidden = _star_units(container, types)
        if len(shown) == 1 and shown[0].count > 1 and not hidden and shown[0].layers[0].is_group():
            container = shown[0].layers[0]
            continue
        break
    if container is None:
        refuse('no group named "Stars" draws them')
    shown = shown + _outside_units(outside)     # Nosk's 'Hk star': laid out with the group's (R2)
    if not shown:
        refuse(f"no star shows in {container.name!r}")
    shown, hidden = _flat_units(shown, hidden, types)       # Cat's group of two beside its group of one
    many = next((unit for unit in shown if unit.count != 1), None)
    if many is not None:                        # R4: one layer drawing several, split into masked copies first
        why = "it is a group" if many.layers[0].is_group() or len(many.layers) != 1 else _splittable(many.layers[0])
        if why:
            refuse(f"{many.layers[0].name!r} draws {many.count} in one {'group' if many.layers[0].is_group() else 'layer'}"
                   f" ({why})")
        if shrink:
            refuse(f"{many.layers[0].name!r} draws {many.count} in one layer")
        if not split:                           # psd stays as it is: the plan is made on a copy
            return plan_stars(_reopened(psd), parsed, value, parser, True)
        done = split_stars(many.layers[0])
        plan = plan_stars(psd, parsed, value, parser, True)
        plan.done = done + plan.done
        return plan
    several = len({id(unit.layers[0].parent) for unit in shown + hidden}) > 1
    shown, hidden = _merge_twins(shown), _merge_twins(hidden)       # R3: a star drawn twice on one spot is one
    follow = _followed(shown)                   # a texture on their group that set_icons copies over them
    count = len(shown) + value - total
    if count < 1:
        refuse(f"{container.name!r} draws {len(shown)}: the other {total - len(shown)} are elsewhere")
    clear = _texts(psd) + badge_layers(psd, parser.all_types if parser is not None else ())[1]   # off the text and type icons
    mine = {id(one) for unit in shown + hidden + (second[1] + second[2] if second is not None else [])
            for layer in _with_chains(_drawn_as(unit).layers)
            for one in [layer, *(layer.descendants() if layer.is_group() else ())]}     # the stars' own layers
    inks: dict = {}
    show: dict = {}
    total_of = lambda show_: PSDParser.stars_drawn(psd, () if named else types, stars, show_)

    def lay_out(shown: list, hidden: list, container, count: int, below: bool = False) -> tuple:
        """
        One set's stars hidden, shown, moved and copied into plan to show `count` (below: a second creature's, turned
        half round: the layouts read it as a header). Returns (how, the set's shown stars, its copies: (unit, dx, dy, ""),
        its stars' median size).
        """
        size = _median_size(shown)
        given = (list(shown), list(hidden))     # what _Shrink shrinks
        counted = (lambda show_: PSDParser.stars_below(psd, types, stars, show_)) if below else total_of

        def failed(moves: list, soft: bool = True) -> tuple:
            """(why, how far its checks got: 0 not placed, 1 under art), else (None, 2): the deciding reason's rank.
            soft False: where it fits alone (on the canvas, off the text and icons), art and textures aside."""
            why = _stars_blocked(psd, _as_drawn(moves), clear, below, soft, follow,
                                 BANNER_GAP + BANNER_TEXT_GAP if shrink else 0, grown)  # shrunk: a banner row's gap
            if why:
                return why, 0
            why = _covered(psd, _as_drawn(moves), mine, inks) if soft else None     # not under the art (MMC's moss)
            return (why, 1) if why else (None, 2)
        blocked = lambda moves: failed(moves)[0]
        grown: dict = {}                        # failed's grown inks per star (shrunk)
        backdrop = lambda units: _backdrop(psd, [_drawn_as(unit) for unit in units])
        off_back = lambda back, moves, copies: _off_backdrop(back, _as_drawn(moves), _as_drawn(copies), bool(shrink))

        def shaded(kept: list, moves: list) -> list:
            flagged = {id(unit) for unit, _, _ in _shaded(psd, [_drawn_as(unit) for unit in kept], _as_drawn(moves))}
            return [move for move in moves if id(_drawn_as(move[0])) in flagged]

        def lowered(order: list, moves: list, copies: list, under: list, back, soft: bool = True) -> Optional[tuple]:
            """
            A layout that does not fit, stepped down (_drops): its new stars one by one, else every star it moves or
            adds together, until failed and off_back pass (soft False: failed's hard checks alone) and no star sits on
            another. Returns (moves, copies, px, what was lowered) or None.
            """
            moved = {id(unit) for unit, _, _ in moves + under}
            stay = [unit.core for unit in order if id(unit) not in moved]
            apart = lambda box, boxes: all(_covers(box, b) <= OVERLAP and _covers(b, box) <= OVERLAP for b in boxes)
            fits = lambda moves_, copies_: failed(moves_ + [copy[:3] for copy in copies_], soft)[0] is None \
                and (not soft or off_back(back, moves_, copies_) is None)
            if copies and fits(moves, []):
                taken, out = stay + [_moved_box(unit.core, dx, dy) for unit, dx, dy in moves], []
                for unit, dx, dy, name in copies:
                    box = _moved_box(unit.core, dx, dy)
                    d = next((d for d in (0, *_drops(psd, box)) if apart(_moved_box(box, 0, d), taken)
                              and fits([], [(unit, dx, dy + d, name)])), None)
                    if d is None:
                        break
                    taken.append(_moved_box(box, 0, d))
                    out.append((unit, dx, dy + d, name))
                else:
                    drop = max(dy - old[2] for (_, _, dy, _), old in zip(out, copies))
                    if drop:
                        return moves, out, drop, "the new star" + ("s" if len(out) > 1 else "")
            boxes = [_moved_box(unit.core, dx, dy) for unit, dx, dy, *_ in moves + copies]
            for d in _drops(psd, max(boxes, key=lambda box: box[3])) if boxes else ():
                down, new = [(u, dx, dy + d) for u, dx, dy in moves], [(u, dx, dy + d, n) for u, dx, dy, n in copies]
                if all(apart(_moved_box(box, 0, d), stay) for box in boxes) and fits(down, new) \
                        and not (soft and shaded(order, down)):
                    return down, new, d, "the stars" if moves else "the new stars"
            return None
        centred = _centred_row(shown, size, psd.width)
        how, copies = "", []
        if count < len(shown):              # fewer: hide the last in reading order (or the first, when that leaves a
            order = _reading(shown, size[1], True)      # layout and the last does not); rows of two centre an odd one
            drop, hide, moves, alphas = len(shown) - count, None, [], {}
            if centred:                     # a centred row stays centred (Grimora's)
                moves = _place(order[:count], _centred_spots(centred, count, psd.width), size)[0]
                if not (blocked(moves) or shaded(order[:count], moves) or off_back(backdrop(shown), moves, [])):
                    hide = order[count:]
                else:
                    moves = []
            die = _dice_five(order, size) if hide is None and count in DICE_FACES else None
            if die:                         # a die's five: the die's own face (1 the middle, 2 and 3 a diagonal)
                keep = [die[spot] for spot in DICE_FACES[count]]
                hide = [unit for unit in order if not any(unit is one for one in keep)]
            if hide is None:                # none that bares a kept star's shadow (Leshy's 1: its right drop's)
                hide = next((gone for kept, gone in ((order[:count], order[count:]), (order[drop:], order[:drop]))
                             if _known_layout(kept, size) and not _bares(gone, kept, alphas)), None)
            if hide is None:
                model = _pyramid(order, size)
                moves = _place(order[:count], _pyramid_spots(model, count), size)[0] if model else []
                if moves and (blocked(moves) or shaded(order[:count], moves)
                              or off_back(backdrop(shown), moves, [])):        # under a shadow: no moves
                    moves = []
                shape = None if moves or count < 3 or _known_layout(order, size) else _kept_shape(order, size, count)
                hide = shape or order[count:]           # a zigzag or a diagonal keeps its first stars (Dark Esau 2)
            plan.hide += [layer for unit in hide for layer in unit.layers]
            how = f"{len(hide)} hidden"
            show.update({id(layer): False for unit in hide for layer in unit.layers})
            if moves:
                plan.moves += _as_drawn(moves)
                how = "a centred row" if centred else "rows of two"
            return how, [unit for unit in shown if not any(unit is one for one in hide)], copies, size
        layouts = _star_layouts(_reading(shown, size[1], True), size, count) + \
            _more_layouts(_reading(shown, size[1], True), size, count, psd)
        if centred:                         # more: a centred row stays centred, else the card's own hidden stars,
            layouts = [("a centred row", _centred_spots(centred, count, psd.width))] + layouts   # then the layout's
        spots = [spot for _, layout in layouts for spot in layout]
        for unit in _reading(hidden, size[1], True):
            if len(shown) == count:
                break
            if several and not any(abs(unit.centre[0] - x) <= _tol(max(size)) and abs(unit.centre[1] - y) <= _tol(max(size))
                                   for x, y in spots):
                continue                        # Shadow Queen's spare art: shown only where the layout puts a star
            if any(_covers(unit.core, other.core) > SPARE_OVERLAP or _covers(other.core, unit.core) > SPARE_OVERLAP
                   for other in shown):
                continue                        # on a shown star (Hornet's spare flower): a star is copied instead
            if any(not size[k] / 2 <= unit.size[k] <= size[k] * 2 for k in (0, 1)):
                continue                        # another size: no star of the row (Toy Freddy's hidden 'star test')
            trial = {**show, **{id(layer): True for layer in unit.layers}}
            if counted(trial) == counted(show) + 1 and blocked([(unit, 0, 0)]) is None:
                show.update(trial)
                shown = shown + [unit]
                plan.show += unit.layers
                how = "hidden stars shown"
        if len(shown) < count:
            order = _reading(shown, size[1], True)
            now = [unit for unit in order if all(layer.visible for layer in unit.layers)] or order  # a copy keeps its flag
            source = max((unit for unit in now if unit.layers[0].parent is container), default=now[len(now) - 1],
                         key=lambda unit: max(container.index(layer) for layer in unit.layers
                                              if layer.parent is container))   # the last star layer shown now
            first, back = None, backdrop(order)
            more = _more_layouts(order, size, count, psd, None if below else back, clear)   # a row on its banner last
            if centred:                     # the centred row once the hidden stars shown are in it
                layouts = [("a centred row", _centred_spots(centred, count, psd.width))] + \
                    _star_layouts(order, size, count) + more
            else:
                layouts = _star_layouts(order, size, count) + more
            banner = back
            if shrink:                                  # shrunk (_shrunk): on the banner or refused
                stages = [("spot", back)] if back is not None else []
            elif shrink is None and back is not None and not below and second is None and not named and not outside:
                # owner 11-10: all shrink until they fit (option B), else off it (shrink 0, told)
                stages = [("spot", back), ("lower", back), ("shrink", back)]
            else:
                backs = (back, None) if back is not None and len(order) == 1 else (back,)   # a lone star's, while a layout keeps it
                stages = list(product(("spot", "lower"), backs))
                if back is not None and len(order) > 1:     # owner 11-08: else off its banner, told
                    stages += [("spot", None), ("lower", None)]
                stages.append(("fit", banner))              # else where they fit alone, art aside (owner 11-08), told
            places: dict = {}
            every = lambda moves, copies: moves + [(unit, dx, dy) for unit, dx, dy, _ in copies]
            for (mode, back), j in product(stages, range(len(layouts))):
                if mode == "shrink":
                    raise _Shrink(*given, back)
                name, spots = layouts[j]
                if j not in places:
                    moves, free = _place(order, spots, size, bool(shrink))
                    under = [move for move in shaded(order, moves) if move[0] is not source]
                    moves = [move for move in moves if move not in under]   # under a shadow: hidden, a copy (on top)
                    free += [(unit.centre[0] + dx, unit.centre[1] + dy) for unit, dx, dy in under]
                    places[j] = moves, under, [(source, round(spot[0] - source.centre[0]),
                                                round(spot[1] - source.centre[1]), "") for spot in free]
                moves, under, copies = places[j]
                soft, drop = mode != "fit", 0   # fit: none clear of the art, the first that fits alone (owner 11-08)
                if mode == "lower" or (mode == "fit" and failed(every(moves, copies), False)[0]):
                    found = None if below else lowered(order, moves, copies, under, back, soft)   # owner 11-08: down
                    if found is None:                                                               # until it fits
                        continue
                    moves, copies, drop, what = found
                why, far = failed(every(moves, copies), soft)
                why = why or (off_back(back, moves, copies) if soft else None)
                if why is None:
                    art = None if soft else failed(every(moves, copies))[0] or off_back(back, moves, copies)
                    plan.moves += _as_drawn(moves)
                    copies = _as_drawn(copies)
                    how = name
                    gone = [layer for unit, _, _ in under for layer in unit.layers]
                    plan.hide += gone
                    show.update({id(layer): False for layer in gone})
                    if drop:
                        plan.notes.append(f"stars: {what} placed {drop} px lower to fit: check it")
                    if art:
                        plan.notes.append(f"stars: placed where they fit, though {art}: check it")
                    if back is None and banner is not None and (len(order) > 1 or shrink == 0):   # a lone star's too
                        plan.notes.append(f"stars: placed off {banner[0].name!r} under them: check it")    # once B failed
                    shown = [unit for unit in shown if not any(unit is one for one, _, _ in under)]
                    break
                if mode == "spot" and (first is None or far > first[0]):   # the deciding reason: the layout whose
                    first = (far, why)                                      # checks got furthest
            else:
                refuse(first[1] if first else f"{len(order)} stars laid out as no row, column or rows of two")
        return how, shown, copies, size

    how, kept, copies, size = lay_out(shown, hidden, container, count)
    plan.how = how
    if second is not None:                      # the second creature's: the same count, turned half round
        dead_how, _, dead_copies, _ = lay_out(_turned(_merge_twins(second[1]), psd), _turned(_merge_twins(second[2]), psd),
                                           second[0], value, True)
        plan.how = f"{how}; below, turned: {dead_how}"
        copies = copies + dead_copies
    plan.copies = copies
    boxes: dict = {}
    counts: dict = {}
    for unit, dx, dy in plan.moves:
        if unit.layers[0].is_group():
            continue
        show.update({id(layer): False for layer in unit.layers})
        boxes.setdefault(id(unit.layers[0].parent), []).extend(_moved_box(PSDParser._core_box(layer), dx, dy)
                                                              for layer in unit.layers)
    for unit, dx, dy, _ in plan.copies:
        if unit.layers[0].is_group():
            counts[id(unit.layers[0].parent)] = counts.get(id(unit.layers[0].parent), 0) + 1
        else:
            boxes.setdefault(id(unit.layers[0].parent), []).extend(_moved_box(PSDParser._core_box(layer), dx, dy)
                                                                  for layer in unit.layers)
    drawn = PSDParser.stars_drawn(psd, stars=stars, show=show, boxes=boxes) if named \
        else PSDParser.stars_drawn(psd, types, stars, show, boxes, counts)
    if drawn != value:
        refuse(f"the new layout would count {drawn}")
    if second is not None and PSDParser.stars_below(psd, types, stars, show, boxes) != value:
        refuse(f"the second creature's new layout below would count {PSDParser.stars_below(psd, types, stars, show, boxes)}")
    moved = {id(unit): (dx, dy) for unit, dx, dy in plan.moves}
    gone = {id(layer) for layer in plan.hide}
    plan.centres = [unit.centre for unit in _reading(
        [_Unit(unit.layers, _moved_box(unit.core, *moved.get(id(unit), (0, 0)))) for unit in kept
         if not any(id(layer) in gone for layer in unit.layers)]
        + [_Unit(unit.layers, _moved_box(unit.core, dx, dy)) for unit, dx, dy, _ in copies if not isinstance(unit, _Turned)
           and in_header(_moved_box(unit.core, dx, dy), psd.height)], size[1], True)]
    plan.moves = [(layer, dx, dy) for unit, dx, dy in plan.moves for layer in _with_chains(unit.layers)]
    taken: set = set()
    out = []
    for unit, dx, dy, _ in plan.copies:         # a star drawn twice: a copy in each group it is drawn in
        parents = []
        for layer in unit.layers:
            if layer.parent not in parents:
                parents.append(layer.parent)
        for parent in parents:
            one = unit if len(parents) == 1 else _Unit([layer for layer in unit.layers if layer.parent is parent], unit.core)
            out.append((one, dx, dy, _copy_name(one, parent, taken)))
    plan.copies = out
    plan.textures = follow
    return plan


def _shrunk(psd: PSDImage, parsed: dict, value: int, parser: Optional[PSDParser], wanted: _Shrink) -> Optional[IconPlan]:
    """
    Option B (owner 11-10): stars whose added ones fit nowhere on their banner all shrink evenly, SHRINK_STEP at a time
    down to SHRINK_MIN, until a layout fits there at its spots (none stepped down: no staircases): each star (shown or
    spare, with the layers clipped to it) by s about its core's centre, the centres drawn s times closer to the shown
    stars' top-right corner (their right edge, the top row's middle height). Each try is planned on a copy of the card
    shrunk so (_scale), then told onto psd's layers: plan.scaled shrinks them the same way first (set_icons). None when
    no size fits, or a star cannot shrink (_scalable: a split star's mask too; an 8-bit RGB card only), or a texture
    is clipped onto the stars' group (_textures: Yin Yarn's; none needs B today).
    """
    units = wanted.shown + wanted.hidden
    if psd.color_mode != ColorMode.RGB or psd.depth != 8 \
            or any(_scalable(layer) for unit in units for layer in _with_chains(unit.layers)) \
            or any(_textures(_drawn_as(unit)) for unit in units):
        return None
    mine = list(psd.descendants())
    at = {id(layer): i for i, layer in enumerate(mine)}
    buf = BytesIO()
    psd.save(buf, update_preview=False)
    raw = buf.getvalue()
    top = _rows(wanted.shown, _median_size(wanted.shown)[1], True)[0]
    ax, ay = max(unit.core[2] for unit in wanted.shown), sum(unit.centre[1] for unit in top) / len(top)
    sizes = [round(1 - k * SHRINK_STEP, 6) for k in range(1, round((1 - SHRINK_MIN) / SHRINK_STEP) + 1)]
    for s in sizes:
        work = PSDImage.open(BytesIO(raw))
        theirs = list(work.descendants())
        scaled = []
        for unit in units:
            (cx, cy) = unit.centre
            to = (ax + s * (cx - ax), ay + s * (cy - ay))
            for layer in _with_chains(unit.layers):
                _scale(theirs[at[id(layer)]], s, (cx, cy), to)
                scaled.append((mine[at[id(layer)]], s, (cx, cy), to))
        try:
            plan = _plan_stars(work, parsed, value, parser, True, s)
        except Refused:
            continue
        back = {id(layer): one for layer, one in zip(theirs, mine)}
        plan.show, plan.hide = [back[id(layer)] for layer in plan.show], [back[id(layer)] for layer in plan.hide]
        plan.moves = [(back[id(layer)], dx, dy) for layer, dx, dy in plan.moves]
        plan.copies = [(replace(unit, layers=[back[id(layer)] for layer in unit.layers]), dx, dy, name)
                       for unit, dx, dy, name in plan.copies]
        plan.textures = [back[id(layer)] for layer in plan.textures]
        plan.scaled = scaled
        plan.how = f"{plan.how}, every star at {round(100 * s)}%"
        plan.notes.append(f"stars: every star shrunk to {round(100 * s)}% to fit {wanted.banner[0].name!r}")   # told
        return plan
    return None


def canonical_type(parser: Optional[PSDParser], name: str) -> str:
    """A type's vocabulary name ("Fragile Equip", "fragile_equip" -> one spelling); lowercase without a vocabulary."""
    hit = parser.icons.lookup(str(name)) if parser is not None else None
    return hit[0] if hit else str(name).lower()


def same_types(parser: Optional[PSDParser], a, b) -> bool:
    """Whether two type lists name the same icons, in any order."""
    return Counter(canonical_type(parser, t) for t in a or []) == Counter(canonical_type(parser, t) for t in b or [])


def type_row(psd: PSDImage, parser: PSDParser, card: str = "") -> tuple:
    """
    The header type icons the parser reads: (image layers [(canonical name, layer)], icon-font text layers).

    An image layer counts as PSDParser._process_layer counts it: named after a Types/ icon, visible, with pixels, its
    top above the midline, not under a stat row's group, and no frame (PSDParser.is_frame). A text layer too, but the
    title's (title_layers: the parser reads its icons as the name; review 2026-10-24: a baked "Po [fire]" refused every
    types edit after it).

    Args:
        psd: The open card
        parser: offline_parser()
        card: The card's name (file stem), for title_layers
    """
    mid = int(psd.height * TYPE_REGION_RATIO)
    texts = [layer for layer in psd.descendants() if layer.kind == "type" and EFFECT_LAYER_TAG not in layer.name]
    titles = {id(layer) for layer in PSDParser.title_layers(psd, texts, card)}
    images, tokens = [], []
    for layer in psd.descendants():
        if EFFECT_LAYER_TAG in layer.name and not layer.is_group():
            continue
        if _is_text_layer(layer):
            if (layer.name.strip().lower() != "ability" and id(layer) not in titles and layer.is_visible()
                    and layer.bbox[1] < mid
                    and PSDParser._icon_font_spans(layer, layer.engine_dict["Editor"]["Text"].value)):
                tokens.append(layer)
        elif (not parser._is_stat_layer(layer) and layer.name.lower() in parser.all_types and layer.is_visible()
              and layer.has_pixels() and layer.bbox[1] < mid and not PSDParser.is_frame(layer, psd)):
            images.append((canonical_type(parser, layer.name), layer))
    return images, tokens


def _icon_reason(layer) -> Optional[str]:
    """Why a header type layer is no icon the bake moves or hides (a group, text); None. A frame is in no row
    (type_row)."""
    if layer.is_group():
        return "a group"
    if layer.kind not in ("pixel", "smartobject"):
        return f"a {layer.kind} layer"
    return None


def _icon_spots(psd: PSDImage, slots: list, extra: int, clear: list) -> list:
    """
    Boxes for `extra` icons past the row's last: the row carried on at its spacing (under a lone icon: Item_Template's
    column), else, when those boxes would touch the text or stars, a second row under the first icon (a row running
    into the title) or a second column at a lone icon's or a column's right; the first whose boxes stay clear, else the
    row carried on; a row of overlapping icons carries on an icon's width apart (Kyle's). When the last two
    icons sit in two rows (Masher's 2 over 1), the last row first carries on at the pitch of the row above it, toward
    the card's middle, else away (S2: the new icon beside the last, not down the diagonal).
    """
    first, last = slots[0].core, slots[len(slots) - 1].core
    gap = round(ICON_GAP * psd.width / CARD_WIDTH)
    if len(slots) > 1:
        (x0, y0), (x1, y1) = slots[len(slots) - 2].centre, slots[len(slots) - 1].centre
        step = (round(x1 - x0), round(y1 - y0))
        if abs(step[1]) < abs(step[0]) < last[2] - last[0]:
            step = (last[2] - last[0] if step[0] > 0 else last[0] - last[2], step[1])
    else:
        step = (0, last[3] - last[1] + gap)
    options = [[_moved_box(last, step[0] * k, step[1] * k) for k in range(1, extra + 1)]]
    if len(slots) > 1 and abs(step[0]) > abs(step[1]):
        down = max(unit.core[3] for unit in slots) - first[1] + gap
        options.append([_moved_box(first, step[0] * k, down) for k in range(extra)])
    else:
        options.append([_moved_box(last, (last[2] - last[0] + gap) * k, 0) for k in range(1, extra + 1)])
    clear_of = lambda box: _blocked(psd, [((box[0], box[1], box[2], box[3]),
                                           np.ones((box[3] - box[1], box[2] - box[0]), bool))], box, clear) is None
    rows = [row for row in _rows(slots, _median_size(slots)[1], False) if len(row) > 1]
    if len(slots) > 1 and abs(step[1]) > (last[3] - last[1]) / 2 and rows:     # the last two in two rows
        row = rows[len(rows) - 1]
        pitch = abs(round(row[len(row) - 1].centre[0] - row[len(row) - 2].centre[0]))
        for dx in sorted((pitch, -pitch), key=lambda dx: abs(_centre(last)[0] + dx - psd.width / 2)) if pitch else ():
            boxes = [_moved_box(last, dx * k, 0) for k in range(1, extra + 1)]
            inside = lambda point, box: box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]
            if all(clear_of(box) and not any(inside(unit.centre, box) or inside(_centre(box), unit.core) for unit in slots)
                   for box in boxes):          # on no icon of another row
                return boxes
    for boxes in options:
        if all(clear_of(box) for box in boxes):
            return boxes
    return options[0]


def _fits(core: tuple, spot: tuple) -> bool:
    """Whether a hidden icon's core is the spot's size (its longer side within LAYOUT_TOL twice over)."""
    mine, theirs = max(core[2] - core[0], core[3] - core[1]), max(spot[2] - spot[0], spot[3] - spot[1])
    return abs(mine - theirs) <= 2 * LAYOUT_TOL * theirs


def _close_row(psd: PSDImage, slots: list, spots: list, holders: list, height: float, others: list,
               made: Optional[dict] = None) -> tuple:
    """
    The row once it closes (owner 10-26): holders[i] (a kept icon's unit, or an added one's: its own hidden layer, or
    one made from Types/, its ink at its spot in made) takes spots[i] (the slots' cores, then the spots past them),
    centred on it as before (pulled back onto the card at the row's inset), else after the icon before it in its row
    at the card's gap there, else just clear of it: the first clear of the text, the stars and the icons placed. An
    icon in its own spot stays when nothing moved before it in its row or no icon placed touches it there.

    Returns:
        ((dx, dy) per holder, the indices of those pulled back onto the card)
    """
    made = made or {}
    row_of = {id(unit): r for r, row in enumerate(_rows(slots, height, False)) for unit in row}
    rows = [row_of[id(slots[j])] if j < len(slots) else None for j in range(len(spots))]
    for j in range(len(slots), len(spots)):     # past the row: in its row while level with the spot before
        level = j > 0 and abs(_centre(spots[j])[1] - _centre(spots[j - 1])[1]) <= height / 2
        rows[j] = rows[j - 1] if level else -j
    inset = max(0, min(unit.core[0] for unit in slots))
    inks_of = lambda unit, dx, dy: [(_moved_box(box, dx, dy), solid) for box, solid in made[id(unit)]] \
        if id(unit) in made else _moved_inks(unit, dx, dy)
    shifts, pulled, placed, inks = [], [], [], []
    for j, unit in enumerate(holders):
        spot, slot = tuple(spots[j]), slots[j] if j < len(slots) else None
        same = j > 0 and rows[j] == rows[j - 1]
        own = inks_of(unit, 0, 0) if unit is slot else []
        if unit is None or (unit is slot and (not same or tuple(placed[j - 1][::2]) == tuple(spots[j - 1][::2])
                                              or not any(_inks_touch(a, b) for a in own for b in inks))):
            placed.append(spot)
            inks += own
            shifts.append((0, 0))
            continue
        w, h = unit.size
        sx, sy = _centre(spot)
        xs = [sx - w / 2]
        if same:
            right = placed[j - 1][2]
            xs += [right + spot[0] - spots[j - 1][2], right]
        pull = lambda x: inset if x < -_tol(max(w, h)) else x
        for x in xs:
            dx, dy = round(pull(x) - unit.core[0]), round(sy - h / 2 - unit.core[1])
            mine = inks_of(unit, dx, dy)
            if _blocked(psd, mine, _moved_box(unit.core, dx, dy), others) is None and \
                    not any(_inks_touch(a, b) for a in mine for b in inks):
                break
        else:                                   # none clear: centred (refused, or an added one stepped down, later)
            x = xs[0]
            dx, dy = round(pull(x) - unit.core[0]), round(sy - h / 2 - unit.core[1])
            mine = inks_of(unit, dx, dy)
        placed.append(_moved_box(unit.core, dx, dy))
        inks += mine
        shifts.append((dx, dy))
        if (dx or dy) and pull(x) != x:
            pulled.append(j)
    return shifts, pulled


def _grid(slots: list, height: float, extra: int) -> Optional[tuple]:
    """
    A header of two rows as a grid (owner 10-26, Masher's mock-up): the short last row lines up under the top row's
    columns, from the row's start and on the card, the new icons under the next columns at their icons' size.

    Returns:
        ([(unit, dx)], new spots), or None
    """
    rows = _rows(slots, height, False)
    if len(rows) < 2 or len(rows[len(rows) - 1]) + extra > len(rows[0]):
        return None
    top, last = rows[0], rows[len(rows) - 1]
    inset = max(0, min(unit.core[0] for unit in slots))
    moves, right = [], inset
    for k, unit in enumerate(last):
        x0 = max(inset, right, round(top[k].centre[0] - unit.size[0] / 2))
        if x0 != unit.core[0]:
            moves.append((unit, x0 - unit.core[0]))
        right = x0 + unit.size[0]
    y = sum(unit.centre[1] for unit in last) / len(last)
    spots = [(round(cx - w / 2), round(y - h / 2), round(cx + w / 2), round(y + h / 2))
             for (w, h), cx in [(unit.size, unit.centre[0]) for unit in top[len(last):len(last) + extra]]]
    return moves, spots


def _grid_clear(psd: PSDImage, plan: IconPlan, kept: list, others: list, types_dir: Optional[str]) -> bool:
    """Whether the grid's moved and new icons can move, sit on the card, and touch no text, star or other icon (a made
    one's ink from its Types/ PNG: not without types_dir)."""
    if any(_shiftable(layer) for layer, _, _ in plan.moves):
        return False
    shifts = {id(layer): (dx, dy) for layer, dx, dy in plan.moves}
    units = kept + [_Unit([layer], _core(layer) or tuple(layer.bbox)) for layer in plan.show]
    icons = [(id(unit.layers[0]) in shifts, _moved_inks(unit, *shifts.get(id(unit.layers[0]), (0, 0))),
              _moved_box(unit.core, *shifts.get(id(unit.layers[0]), (0, 0)))) for unit in units]
    for stem, box, _ in plan.icons:
        if types_dir is None:
            return False
        try:
            image, left, top, _ = _icon_image(_icon_png(types_dir, stem), box)
        except Refused:
            return False
        ink = ((left, top, left + image.width, top + image.height), np.asarray(image)[..., 3] / 255 > INK_ALPHA)
        icons.append((True, [ink], ink[0]))
    for k, (new, inks, core) in enumerate(icons):
        if new and _blocked(psd, inks, core, others):
            return False
        if any((new or theirs_new) and _inks_touch(a, b) for theirs_new, theirs, _ in icons[k + 1:]
               for a in inks for b in theirs):
            return False
    return True


def _in_stars(layer) -> bool:
    return any("stars" in owner.name.lower() for owner in [layer, *layer_parents(layer)])


def layer_parents(layer) -> list:
    """The layer's groups, innermost first."""
    out, parent = [], layer.parent
    while parent is not None and not isinstance(parent, PSDImage):
        out.append(parent)
        parent = parent.parent
    return out


def plan_types(psd: PSDImage, parsed: dict, value: list, parser: Optional[PSDParser],
               template: Optional[tuple] = None, types_dir: Optional[str] = None) -> IconPlan:
    """
    How to show the type icons `value` in the card's header row (Types/ icons as image layers, the parser's rule):
    the icons the edit keeps stay where they are, a removed one hides, and the row closes up: a kept icon past the new
    end moves into the first free spot, centre on centre, and the icons after it slide just clear of it (_close_row);
    a new icon takes the next free spot, from the card's own hidden layer of that type (the templates keep one per
    type) or made from its Types/ PNG, fitted to the spot and styled like the icon it replaces (else the row's last);
    with icons removed too it goes through the same closing (_close_row; a made one's ink needs types_dir).
    Spots past the last icon carry the row on at its spacing, or go under a lone icon (Item_Template's column), or wrap
    clear of the title (_icon_spots); a header of two rows takes them as a grid when nothing touches (_grid). Refused per
    field: a token row ([fire] in header text), a removed type drawn as a frame or group, no icon to line a new one up
    with, a type with no Types/ icon, a layer that cannot move, or an icon that would run off the canvas, onto the text
    or stars, or over another icon, where an added one stepped down fits nowhere (_lowered; owner 11-08, told).

    Args:
        psd: The open card (not changed)
        parsed: parse_card() of it
        value: The type names to show (any spelling the icon vocabulary knows)
        parser: offline_parser() (its Types/ names and vocabulary)
        template: template_icon(): a card with no type icon takes its first at the template's spot, styled as the
            template's icon (owner 10-25; flagged), the rest carried on from it
        types_dir: The Types/ PNGs, for _grid's check of a made icon's ink (none: no grid with a made icon)

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
    images, tokens = type_row(psd, parser, _card_name(parsed))
    if tokens:
        refuse(f"printed as icon tokens in {tokens[0].name!r}: edit them in Photoshop")
    plan = IconPlan("types", [name for name, _ in images], value)
    reasons = [_icon_reason(layer) for _, layer in images]
    left = Counter(want)
    for (name, layer), reason in zip(images, reasons):
        if reason and not left[name]:
            refuse(f"{layer.name!r} is {reason}, not an icon to take off")
        left[name] -= 1 if reason else 0
    row = [(name, layer) for (name, layer), reason in zip(images, reasons) if not reason]
    rel = _card_path(parsed)
    icons = [_Unit([layer], _icon_core(layer, rel)) for _, layer in row]
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
    first = None
    if count > len(slots) and not slots and template is not None and count:   # the first at the template's spot
        spot, reference, first = template
        slots = [_Unit([reference], spot)]
        spots = [spot]
    if count > len(slots):
        if not slots:
            refuse("no type icon on the card to line a new one up with")
        spots += _icon_spots(psd, slots, count - len(slots), _texts(psd) + badge_layers(psd, parser.all_types)[0])
    holder = {i: unit for i, unit in enumerate(slots[:count]) if unit in kept}
    free = [i for i in range(count) if i not in holder]
    for unit in [unit for unit in kept if slots.index(unit) >= count]:
        i = free.pop(0)
        (cx, cy), (sx, sy) = unit.centre, _centre(spots[i])
        plan.moves.append((unit.layers[0], round(sx - cx), round(sy - cy)))
        holder[i] = unit
    plan.hide = [layer for unit in removed for layer in unit.layers]
    row_ids = {id(layer) for _, layer in images}       # the row's own icons are no stars (Pike Beam)
    stars = [layer for layer in badge_layers(psd, parser.all_types)[0] if id(layer) not in row_ids]
    texts = _texts(psd)
    library = [layer for layer in psd.descendants() if not layer.is_group() and not layer.visible
               and layer.kind in ("pixel", "smartobject") and layer.name.lower() in parser.all_types and layer.has_pixels()
               and all(parent.visible for parent in layer_parents(layer)) and not _in_stars(layer)]

    def picks(spots: list) -> list:             # the new icons: (spot index, own hidden layer or None, stem, reference)
        shelf, out = list(library), []
        for name, i in zip(added, free):
            spot = spots[i]
            replaced = slots[i] if i < len(slots) and slots[i] in removed else None
            reference = (replaced or (kept[len(kept) - 1] if kept else slots[len(slots) - 1])).layers[0]
            own = next((layer for layer in shelf if canonical_type(parser, layer.name) == name and not _shiftable(layer)
                        and _fits(_core(layer) or tuple(layer.bbox), spot)), None)
            if own is not None:
                shelf.remove(own)
                out.append((i, own, None, reference))
                continue
            stem = next((t for t in parser.all_types if canonical_type(parser, t) == name), None)
            if stem is None:
                refuse(f"no Types/ icon for {name!r}")
            out.append((i, None, stem, reference))
        return out

    def add(spots: list, closed: Optional[dict] = None) -> None:     # the new icons into the free spots (closed: their
        for i, own, stem, reference in picks(spots):                 # moves, or a made one's shift, as the row closed)
            if own is not None:
                (cx, cy), (sx, sy) = _centre(_core(own) or tuple(own.bbox)), _centre(spots[i])
                plan.show.append(own)
                plan.moves.append((own, *(closed or {}).get(i, (round(sx - cx), round(sy - cy)))))
            else:
                plan.icons.append((stem, _moved_box(spots[i], *(closed or {}).get(i, (0, 0))), reference))
    closing = None
    if plan.moves or (removed and added):       # the icons after a moved, removed or added one make room (owner 10-26),
        new, made = {}, {}                      # the added ones through the same rule
        for i, own, stem, _ in picks(spots):
            if own is not None:
                new[i] = _Unit([own], _core(own) or tuple(own.bbox))
                continue
            box, inks = tuple(spots[i]), []
            if types_dir is not None:           # its ink as made (none to read without Types/: it keeps its spot)
                image, left, top, _ = _icon_image(_icon_png(types_dir, stem), spots[i])
                box = (left, top, left + image.width, top + image.height)
                inks = [(box, np.asarray(image)[..., 3] / 255 > INK_ALPHA)]
            new[i] = _Unit([], box)
            made[id(new[i])] = inks
        holders = [holder.get(i) or new.get(i) for i in range(count)]
        shifts, pulled = _close_row(psd, slots, spots, holders, size[1], texts + stars, made)
        plan.moves = [(unit.layers[0], dx, dy) for unit, (dx, dy) in zip(holders, shifts)
                      if unit in kept and (dx or dy)]
        plan.pulled = [holders[j].layers[0] for j in pulled if holders[j].layers]
        closing = {i: shifts[i] for i in new}
    grid = _grid(slots, size[1], len(added)) if added and not removed and not first else None
    if grid:                                    # else (or touching) today's spots
        plan.moves = [(unit.layers[0], dx, 0) for unit, dx in grid[0]]
        add(spots[:len(slots)] + grid[1])
        if not _grid_clear(psd, plan, kept, texts + stars, types_dir):
            plan.moves, plan.show, plan.icons = [], [], []
            grid = None
    if not grid:
        add(spots, closing)
    inset = (max(0, min(u.core[0] for u in slots)), max(0, min(u.core[1] for u in slots)),
             max(0, min(psd.width - u.core[2] for u in slots))) if slots else (0, 0, 0)
    for k, (layer, dx, dy) in enumerate(list(plan.moves)):
        why = _shiftable(layer)
        if why:
            refuse(f"{why}: it cannot move")
        unit = _Unit([layer], _icon_core(layer, rel))
        others = [other for other in texts + stars if other is not layer]
        blocked = _blocked(psd, _moved_inks(unit, dx, dy), _moved_box(unit.core, dx, dy), others)
        box = _moved_box(unit.core, dx, dy)
        if blocked == "runs off the canvas":    # S3: pulled back onto the card, at the row's own inset
            dx += max(0, inset[0] - box[0]) - max(0, box[2] - (psd.width - inset[2]))
            dy += max(0, inset[1] - box[1])
            plan.moves[k] = (layer, dx, dy)
            blocked = _blocked(psd, _moved_inks(unit, dx, dy), _moved_box(unit.core, dx, dy), others)
            plan.pulled.append(layer)
        if blocked and not (first and blocked.startswith("would touch")):    # the template's spot bakes (rulings)
            drop = None
            if any(layer is one for one in plan.show):      # an added icon steps down until it fits (owner 11-08)
                shifts = {id(one): (mx, my) for one, mx, my in plan.moves}
                icons = [(_moved_box(ink[0], *shifts.get(id(one), (0, 0))), ink[1])
                         for one in [unit.layers[0] for unit in kept] + plan.show if one is not layer
                         for ink in [_ink(one)] if ink is not None]
                drop = _lowered(psd, _moved_inks(unit, dx, dy), _moved_box(unit.core, dx, dy), others, icons)
            if drop is None:
                refuse(f"{layer.name!r} at {tuple(int(v) for v in _moved_box(unit.core, dx, dy))} {blocked}")
            plan.moves[k] = (layer, dx, dy + drop)
            plan.notes.append(f"types: the {layer.name!r} icon placed {drop} px lower to fit: check it")
    plan.row = [unit.layers[0] for unit in kept] + list(plan.show)
    shifts = {id(layer): (dx, dy) for layer, dx, dy in plan.moves}
    finals = [(layer, _ink(layer)) for layer in plan.row] if shifts else []
    finals = [(layer, (_moved_box(ink[0], *shifts.get(id(layer), (0, 0))), ink[1])) for layer, ink in finals
              if ink is not None and ink[1].any()]
    for k, (layer, mine) in enumerate(finals):  # every moved or shown icon clear of the others
        for other, theirs in finals[k + 1:]:
            if (id(layer) in shifts or id(other) in shifts) and \
                    _ink_overlap(mine, theirs) > ICON_TOUCH * min(mine[1].sum(), theirs[1].sum()):
                one, two = (layer, other) if id(layer) in shifts else (other, layer)
                box = tuple(int(v) for v in _moved_box(_icon_core(one, rel), *shifts[id(one)]))
                pulled = "runs off the canvas; pulled back onto it, " if any(one is x for x in plan.pulled) else ""
                refuse(f"{one.name!r} at {box} {pulled}would cover {two.name!r}")
    for _, spot, _ in plan.icons:
        slack = _tol(max(spot[2] - spot[0], spot[3] - spot[1]))
        if spot[0] < -slack or spot[1] < -slack or spot[2] > psd.width + slack or spot[3] > psd.height + slack:
            refuse(f"a new icon at {tuple(int(v) for v in spot)} runs off the canvas")
    moved = len(plan.moves)
    plan.moves = [(one, dx, dy) for layer, dx, dy in plan.moves for one in _with_chains([layer])]
    plan.how = ", ".join(part for part in (f"{len(plan.hide)} hidden" if plan.hide else "",
                                           f"{moved} moved" if moved else "",
                                           f"{len(plan.icons)} made from Types/" if plan.icons else "",
                                           f"the first at {first}'s spot: check it" if first else "") if part)
    plan.first = first or ""
    return plan


def _icon_png(types_dir: str, stem: str) -> str:
    """The Types/ PNG of a stem (any case)."""
    from os import listdir
    for name in listdir(types_dir):
        if name.lower() == f"{stem}.png".lower():
            return join(types_dir, name)
    raise Refused("unsupported-card", f"types: no {stem}.png in Types/", "types")


def _icon_image(path: str, box: tuple) -> tuple:
    """
    A Types/ PNG fitted into the box (its opaque part, aspect kept, centred; hard_edged: nearest): (RGBA image, left,
    top, the whole file's quad: where a smart object linking it draws those pixels).
    """
    image = Image.open(path).convert("RGBA")
    solid = image.getchannel("A").point(lambda a: 255 if a >= 128 else 0).getbbox()
    if solid is None:
        raise Refused("unsupported-card", f"types: {basename(path)} draws nothing", "types")
    full = image.size
    image = image.crop(solid)
    scale = min((box[2] - box[0]) / image.width, (box[3] - box[1]) / image.height)
    width, height = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    mode = Image.NEAREST if hard_edged(np.asarray(image)) else Image.LANCZOS
    sx, sy = width / image.width, height / image.height
    image = image.convert("RGBa").resize((width, height), mode).convert("RGBA")
    cx, cy = _centre(box)
    left, top = round(cx - width / 2), round(cy - height / 2)
    x0, y0 = left - solid[0] * sx, top - solid[1] * sy
    x1, y1 = x0 + full[0] * sx, y0 + full[1] * sy
    return image, left, top, [x0, y0, x1, y0, x1, y1, x0, y1]


def _make_icon(psd: PSDImage, types_dir: str, stem: str, box: tuple, reference, rel: str = ""):
    """
    A smart object linking the Types/ icon (psd_links.link_pixels; owner Q4, 2026-11-07), its pixels fitted into the
    box (_icon_image), the reference's effects, blending and opacity copied, directly above the reference. rel: the
    card's repo path (the link's relPath); "" makes a pixel layer.
    """
    from psd_tools.api.layers import PixelLayer
    path = _icon_png(types_dir, stem)
    image, left, top, quad = _icon_image(path, box)
    profile, source = foreign_profile(psd), image.info.get("icc_profile")      # field.png: Adobe RGB
    if profile is not None or source:           # the PNG's profile (else sRGB) into the card's, as Photoshop places it
        from PIL import ImageCms
        def cms(data): return ImageCms.ImageCmsProfile(BytesIO(data)) if data else ImageCms.createProfile("sRGB")
        alpha = image.getchannel("A")
        image = ImageCms.profileToProfile(image.convert("RGB"), cms(source), cms(profile), outputMode="RGB")
        image.putalpha(alpha)
    name = basename(path)[:len(basename(path)) - 4]
    layer = PixelLayer.fromarray(np.asarray(image), psd, name=name, top=top, left=left)
    layer.name = name
    layer.copy_effects_from(reference)
    layer.blend_mode, layer.opacity = reference.blend_mode, reference.opacity
    _issue_id(psd, layer)
    if rel:
        with open(path, "rb") as fh:
            layer = link_pixels(layer, rel, f"{TYPES_DIR}/{basename(path)}", fh.read(), quad)
    if reference._psd is not psd:  # a template's icon: the card's types group, else the top
        parent = _types_group(psd)
        layer.move_to(parent, index=len(parent) - (1 if layer.parent is parent else 0))   # the top of it
        return layer
    chain = _clip_chain(reference)
    parent = reference.parent
    layer.move_to(parent, index=parent.index(reference) + 1 + len(chain))
    return layer


def _types_group(psd: PSDImage):
    """Where a card's first type icon goes: its visible top-level group named for types, else the root."""
    return next((layer for layer in psd if layer.is_group() and layer.visible and "type" in layer.name.lower()), psd)


SMART_BLOCKS = (Tag.SMART_OBJECT_LAYER_DATA1, Tag.SMART_OBJECT_LAYER_DATA2)


def _placed(layer) -> list:
    """A smart object's placed-instance descriptors (its SoLd/SoLE data), [] for another layer."""
    blocks = layer._record.tagged_blocks
    found = [blocks.get_data(key) for key in SMART_BLOCKS] if blocks else []
    return [data.data for data in found if data is not None and b"placed" in getattr(data, "data", {})]


def same_placed(source, copy, salt: str) -> None:
    """
    A copied smart object's placed ID made from its source's and salt, not at random (the fork's duplicate gives a
    uuid4): every picture's bake of a tagged PSD makes the same layers (bake_pictures' check). An ID the card already
    holds is skipped (a star copied again where a hidden copy sits).
    """
    pairs = zip([source, *(source.descendants() if source.is_group() else ())],
                [copy, *(copy.descendants() if copy.is_group() else ())])
    taken = {str(data[b"placed"].value) for layer in copy._psd.descendants() for data in _placed(layer)}
    for old, new in pairs:
        theirs, mine = _placed(old), _placed(new)
        if not theirs or not mine:
            continue
        n = 0
        while (made := str(uuid5(NAMESPACE_URL, f"{theirs[0][b'placed'].value}|{salt}|{n}"))) in taken:
            n += 1
        taken.add(made)
        for data in mine:                       # both blocks name one instance
            data[b"placed"] = String(made)


def _copy_chain(reference, layer, done: list) -> None:
    """The reference icon's clip chain (its Curves...) copied onto a new icon, moved as far as the icon is from it."""
    (rx, ry), (lx, ly) = _centre(_core(reference) or tuple(reference.bbox)), _centre(_core(layer) or tuple(layer.bbox))
    dx, dy = round(lx - rx), round(ly - ry)
    parent = layer.parent
    for k, clipped in enumerate(_clip_chain(reference)):
        copy = clipped.duplicate(parent, index=parent.index(layer) + 1 + k)
        same_placed(clipped, copy, f"chain {layer.name}|{dx},{dy}|{k}")
        _copy_filters(clipped, copy)
        _shift(copy, dx, dy, True)
        done.append(("copy", clipped, copy, dx, dy))


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


def set_icons(psd: PSDImage, plan: IconPlan, types_dir: Optional[str] = None, rel: str = "") -> Edit:
    """
    Apply a plan_stars() or plan_types() plan to the open card (the same file it was planned on): shrink (option B),
    hide, show, move, copy and make layers (a made type icon links its Types/ file from rel, the card's repo path:
    _make_icon; a texture that follows the stars copied over those it left: _texture_copies). Record indices come
    later (index_icons), once every edit is in.

    Returns:
        Edit over the layers it touched, its region their old and new boxes padded by their effects; raises Refused
        (field set) for a type icon that lands on the text or stars (at the template's spot: baked), a made one first
        stepped down until it fits (_lowered, told)
    """
    boxes = [(_extent(layer), layer) for layer in plan.hide + plan.show]
    boxes += [(_extent(layer), layer) for layer, _, _ in plan.moves]
    for layer, s, centre, to in plan.scaled:   # shrunk first (option B): copied and moved from there
        for leaf, was, now in _scale(layer, s, centre, to):
            plan.done.append(("scaled", leaf, now))
            boxes += [(was, leaf), (now, leaf)]
    for unit, dx, dy, name in plan.copies:     # copies first: from where the star is now, with its clipped layers
        parent = unit.layers[0].parent
        index = _copy_index(unit, parent)
        top = max(unit.layers, key=parent.index)
        for k, layer in enumerate(_with_chains(sorted(unit.layers, key=parent.index))):
            copy = layer.duplicate(parent, index=index + k, name=name if layer is top else None)
            same_placed(layer, copy, f"{name}|{dx},{dy}|{k}")
            _copy_filters(layer, copy)          # a smart-filtered star's own FEid item (S1)
            _shift(copy, dx, dy, True)
            plan.done.append(("copy", layer, copy, dx, dy))
            boxes.append((_extent(copy), copy))
    if plan.copies:
        _order_filters(psd)
    for layer in plan.hide:
        layer.visible = False
    for layer in plan.show:
        layer.visible = True
    for layer, dx, dy in plan.moves:
        _shift(layer, dx, dy, True)
        boxes.append((_extent(layer), layer))
    for done, box in _texture_copies(psd, plan):    # a texture on the stars' group follows them (owner 11-09)
        plan.done.append(done)
        boxes.append((box, done[2] if done[0] == "masked copy" else done[1]))
    stars, texts = badge_layers(psd)[0], _texts(psd)
    touching, made = [], []
    for stem, box, reference in plan.icons:
        if types_dir is None:
            raise Refused("unsupported-card", "types: no Types/ icons to make one from", "types")
        layer = _make_icon(psd, types_dir, stem, box, reference, rel)
        others = [other for other in texts + stars if other is not layer]
        mine = _ink(layer)
        blocked = _blocked(psd, [mine], _core(layer) or tuple(layer.bbox), others)
        icons = [ink for ink in (_ink(other) for other in plan.row + made) if ink is not None]
        spot = plan.first and blocked and blocked.startswith("would touch")    # the template's spot bakes (rulings)
        if not spot and (blocked or mine is not None and any(
                _ink_overlap(mine, ink) > ICON_TOUCH * min(mine[1].sum(), ink[1].sum()) for ink in icons)):
            drop = _lowered(psd, [mine], _core(layer) or tuple(layer.bbox), others, icons)   # owner 11-08: down to fit
            if drop:
                _shift(layer, 0, drop, True)
                touching.append(f"types: the {layer.name!r} icon placed {drop} px lower to fit: check it")
            elif blocked:
                raise Refused("unsupported-card", f"types: the {layer.name!r} icon {blocked}", "types")
        mine = _ink(layer)
        for other in plan.row + made:           # clear of the row's other icons
            theirs = _ink(other)
            if mine is not None and theirs is not None and \
                    _ink_overlap(mine, theirs) > ICON_TOUCH * min(mine[1].sum(), theirs[1].sum()):
                if any(other is one for one in plan.pulled):
                    raise Refused("unsupported-card", f"types: {other.name!r} runs off the canvas; pulled back onto it, "
                                  f"would cover the {layer.name!r} icon", "types")
                raise Refused("unsupported-card", f"types: the {layer.name!r} icon would cover {other.name!r}", "types")
        made.append(layer)
        blocks = {bytes(key): layer.tagged_blocks[key].tobytes() for key in EFFECT_BLOCKS if key in layer.tagged_blocks}
        plan.done.append(("made", layer, blocks))
        boxes.append((tuple(layer.bbox), layer))
        if reference._psd is psd:      # a template's chain stays in the template
            _copy_chain(reference, layer, plan.done)
    region = union_box([_pad(box, _reach(layer), psd) for box, layer in boxes if box[2] > box[0] and box[3] > box[1]])
    layers = plan.hide + plan.show + [layer for layer, _, _ in plan.moves] + [
        done[2] if done[0] in ("copy", "masked copy") else done[1] for done in plan.done]
    notes = list(plan.notes) + touching
    return Edit(plan.field, plan.old, plan.new, layers, [], region, icons=plan, notes=notes)


def index_icons(edit: Edit, before: dict, after: dict) -> None:
    """
    Fill an icon edit's record indices once every edit is applied: before = record_map() of the card as opened (the
    records keep their identity through the edits), after = record_map() now (what the saved file holds).
    """
    plan = edit.icons
    split = {id(done[2]): done[1] for done in plan.done if done[0] == "split"}      # a split's copy: its layer's record
    moved = {id(layer): (dx, dy) for layer, dx, dy in plan.moves}
    edit.toggled = sorted(after[id(layer._record)] for layer in plan.hide + plan.show)
    edit.shifted = [(after[id(record)], before[id(record)], dx, dy) for layer, dx, dy in plan.moves
                    if id(layer) not in split for record in _subtree_records(layer)]
    edit.shifted += [(after[id(done[1]._record)], before[id(done[1]._record)], 0, 0) for done in plan.done
                     if done[0] == "masked" and id(done[1]) not in moved]   # a split layer staying where it is
    origin = {id(done[2]._record): done[1]._record for done in plan.done if done[0] == "split"}   # a split's copy: the
    edit.copied = [(after[id(new)], before[id(origin.get(id(old), old))], dx, dy) for done in plan.done   # layer it was
                   if done[0] == "copy" for old, new, dx, dy in [(o, n, done[3], done[4]) for o, n in
                                                                 zip(_subtree_records(done[1]), _subtree_records(done[2]))]]
    edit.copied += [(after[id(done[2]._record)], before[id(done[1]._record)], *moved.get(id(done[2]), (0, 0)))
                    for done in plan.done if done[0] == "split"]
    edit.copied += [(after[id(done[2]._record)], before[id(done[1]._record)], done[3], done[4]) for done in plan.done
                    if done[0] == "masked copy"]          # a texture's copy masked to a star (_texture_copies)
    edit.made = [(after[id(done[1]._record)], done[2]) for done in plan.done if done[0] == "made"]
    edit.scaled = {before[id(done[1]._record)]: (done[2][1], done[2][0], done[2][3], done[2][2])
                   for done in plan.done if done[0] == "scaled"}     # a shrunk star's rect: its moves and copies from it
    shifted = {old for _, old, _, _ in edit.shifted}
    edit.shifted += [(after[id(done[1]._record)], before[id(done[1]._record)], 0, 0) for done in plan.done
                     if done[0] == "scaled" and before[id(done[1]._record)] not in shifted]
    edit.masked = sorted({after[id(done[2 if done[0] == "split" else 1]._record)] for done in plan.done
                          if done[0] in ("split", "masked")}
                         | {after[id(done[2]._record)] for done in plan.done if done[0] == "copy"
                            and (id(done[1]._record) in origin or any(one[0] == "masked" and one[1] is done[1]
                                                                     for one in plan.done))}
                         | {after[id(done[2]._record)] for done in plan.done if done[0] == "masked copy"})
    leaves = lambda layer: [one for one in [layer, *(layer.descendants() if layer.is_group() else ())]
                            if not one.is_group() and one.kind == "smartobject" and _filtered(one)]
    edit.filters = [(leaf.smart_object.placed_id, leaf.smart_object.placed_id, dx, dy) for layer, dx, dy in plan.moves
                    for leaf in leaves(layer)]
    edit.filters += [(copy.smart_object.placed_id, source.smart_object.placed_id, done[3], done[4]) for done in plan.done
                     if done[0] == "copy" for source, copy in zip(leaves(done[1]), leaves(done[2]))]


def drawable(psd: PSDImage) -> None:
    """Refused("unsupported-card") for a PSD with a layer the fork cannot draw as Photoshop does (undrawable)."""
    names = undrawable(psd)
    if names:
        raise Refused("unsupported-card", f"the bot cannot draw {names[0]!r}'s knockout or Transparency Shapes Layer as "
                                          f"Photoshop does: set it in Photoshop")


def render_all(psd: PSDImage) -> np.ndarray:
    """The whole card rendered from its layers (no ICC, no preview), uint8 RGBA: gate_outside's before and after;
    refused when the fork cannot draw it (drawable)."""
    drawable(psd)
    return np.asarray(psd.composite(viewport=(0, 0, psd.width, psd.height), ignore_preview=True,
                                    apply_icc=False, **look_kwargs(psd)).convert("RGBA"))


def edit_copy(psd: PSDImage, raw: bytes) -> PSDImage:
    """
    The card opened again from raw, the bytes psd was opened from, with psd's eye flags (variant_view()): a fresh open
    pairs its smart filters itself (a deepcopy kept the source's id() keys and lost them: review 2026-10-24, Lorian's
    art) and takes a third of a deepcopy's time. psd's stored preview, once read, is shared (_share_preview). Raises
    Refused("error") when the layers are not psd's.
    """
    new = PSDImage.open(BytesIO(raw))
    mine, theirs = list(new.descendants()), list(psd.descendants())
    if [layer.layer_id for layer in mine] != [layer.layer_id for layer in theirs]:
        raise Refused("error", "the card's copy has other layers than the card")
    for layer, like in zip(mine, theirs):
        if layer.visible != like.visible:
            layer.visible = like.visible
    _share_preview(psd, new)
    return new


@lru_cache(maxsize=4)
def layout_fonts(fonts_dir: str = ""):
    """The fork's FontMap for glyph_places: MDW/Fonts (a face no file holds drawn in the stand-in), else the
    environment's fonts."""
    from psd_tools.composite.text import default_font_map
    if fonts_dir:
        return _fonts(fonts_dir)
    fonts_in()                                  # the environment's: PSD_TOOLS_FONT_DIRS
    return default_font_map("substitute")


def glyph_places(layer, fonts) -> list:
    """The fork's layout of a type layer: (glyph id, x, y, em width, em height) per glyph, origins in canvas px."""
    from psd_tools.composite.text import layout
    from psd_tools.composite.text.model import TextModel
    model = TextModel.from_layer(layer)
    xx, xy, yx, yy, tx, ty = model.transform
    return [(g.gid, tx + xx * g.x + yx * g.y, ty + xy * g.x + yy * g.y, g.em_w, g.em_h)
            for g in layout.layout(model, fonts).glyphs]


def same_glyphs(a, b, fonts=None) -> bool:
    """Two versions of a type layer laid out alike by the fork (GLYPH_TOL) in the faces fonts gives (names -> MDW/Fonts:
    card_guards.Source.fonts_for), else the environment's; False when it cannot lay them out."""
    try:
        from psd_tools.composite.text import font_names
        faces = layout_fonts((fonts(font_names(b)) if fonts else "") or "")
        return same_places(glyph_places(a, faces), glyph_places(b, faces), GLYPH_TOL)
    except Exception:                           # no layout: not alike
        return False


def same_places(a: list, b: list, tol: float) -> bool:
    """Two glyph_places() the same glyphs, each within tol (px and em)."""
    return len(a) == len(b) and all(p[0] == q[0] and max(abs(x - y) for x, y in zip(p[1:], q[1:])) <= tol
                                    for p, q in zip(a, b))


def render_cache(shared: bool = False):
    """
    One bake's psd-tools RenderCache (the fork's psd_tools.composite.rcache: the card's renders before and after its
    edits share every layer the edits leave alone; keys hash what a layer is, never id()).
    shared: this thread's open one when there is one (card_checks' two stat bakes of a card).
    """
    from psd_tools.composite import rcache
    return (shared and rcache.active()) or rcache.RenderCache()


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
    diff = _px_differ(before, after)
    if edit.region:
        x0, y0, x1, y1 = edit.region
        diff[y0:y1, x0:x1] = False
    outside = int(diff.sum())
    if outside:
        raise Refused("region", f"{edit.field}: the card changes {outside} pixels outside the edited box", edit.field)
    return after


def gate_untouched(before: np.ndarray, psd: PSDImage, edits: list) -> None:
    """
    The text and stat edits change the card's render only inside their regions, where the PNG takes the new pixels: a
    layer put above the Text group, its clip chain's base, or an adjustment left on another layer would change the PSD
    and not the PNG (review 2026-10-24: Claw's new FX layer took MagentaSmorg and Exposure 1 off its title). The stat
    edits' digits stay in their rows: every edit's region is left out (review 2026-10-24: a stat's with a text edit's).

    Args:
        before: render_all() of the card before them (after its stars and types edits)
        psd: The card with them applied
        edits: Them, the stat edits too

    Returns:
        None; raises Refused("region") for any pixel outside every region
    """
    after = render_all(psd)
    diff = _px_differ(before, after)
    for edit in edits:
        if edit.region:
            x0, y0, x1, y1 = edit.region
            diff[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = False
    outside = int(diff.sum())
    if outside:
        raise Refused("region", "%s: the card changes %d pixels outside the edited boxes"
                      % (", ".join(edit.field for edit in edits), outside))


# ========================
# Dirty box: the untouched gate renders only where old and new can differ (K-1, owner 2026-10-25)
# ========================
DIRTY_VERIFY = "PSD_DIRTY_VERIFY"   # env: "1" renders whole too and raises on any difference (on in tests/conftest.py)
DIRTY_SLACK = 4                     # px past a group effect's reach
FULL = "full"                       # a change no box bounds: the whole canvas


def _record_bytes(layer) -> bytes:
    out = BytesIO()
    layer._record.write(out, version=layer._psd.version)
    return out.getvalue()


def _join(box, other):
    """Union of two boxes, None or an empty box adding nothing (the fork's union_bbox)."""
    empty = lambda b: b is None or b[0] >= b[2] or b[1] >= b[3]
    return other if empty(box) else box if empty(other) else union_box([box, other])


def _effect_margin(layer) -> tuple:
    """(px a group's effects reach past what they draw on, whether one reads the group's bbox)."""
    effects = layer.effects
    if not effects or not effects.enabled:
        return 0, False
    from psd_tools.api.effects import BevelEmboss, GradientOverlay, PatternOverlay, Stroke
    from psd_tools.constants import StrokeFillType
    margin, reads = 0, False
    for effect in effects:
        if not effect.enabled:
            continue
        reach = float(getattr(effect, "size", 0) or 0) + float(getattr(effect, "distance", 0) or 0)
        margin = max(margin, int(2 * reach) + DIRTY_SLACK)
        reads = reads or isinstance(effect, (GradientOverlay, PatternOverlay, BevelEmboss)) \
            or (isinstance(effect, Stroke) and effect.fill_type != StrokeFillType.SOLID_COLOR)
    return margin, reads


class _DirtyDoc:
    """One side's fork pass state: layer keys (rcache) and change boxes (composite._PassCache); new's: the lines."""
    def __init__(self, lines: Optional[dict] = None):
        from importlib import import_module
        from psd_tools.composite import rcache
        self.composite, self.rcache = import_module("psd_tools.composite.composite"), rcache   # the module, not the function
        self.cache, self.keys, self.lines = self.composite._PassCache(), {}, lines or {}

    def key(self, layer) -> bytes:
        return self.rcache.layer_key(layer, self.keys)

    def change(self, layer) -> tuple:
        return self.composite._change_bbox(layer, self.cache)


def _same_look(old, new) -> bool:
    """Type layers alike but their pixels: opacity, blend, clipping, eye, fill, mask, own effects; none of those
    reading the bbox when it moved."""
    if old.kind != "type" or new.kind != "type":
        return False
    blocks = lambda l: [l.tagged_blocks[k].tobytes() if l.tagged_blocks is not None and k in l.tagged_blocks else None
                        for k in EFFECT_BLOCKS]
    mask = lambda l: None if l._record.mask_data is None else tuple(
        getattr(l._record.mask_data, k) for k in ("left", "top", "right", "bottom", "flags", "background_color"))
    look = lambda l: (l.opacity, l.blend_mode, l.clipping, l.visible, l.fill_opacity, mask(l), blocks(l))
    return look(old) == look(new) and not (_effect_margin(new)[1] and tuple(old.bbox) != tuple(new.bbox))


def _children_box(do: _DirtyDoc, dn: _DirtyDoc, old, new):
    """Box old's and new's children (one group, or the documents) differ in; None if nowhere, FULL if unbounded."""
    oc, nc = list(old), list(new)
    if [do.key(layer) for layer in oc] == [dn.key(layer) for layer in nc]:
        return None
    oi, ni = {l.layer_id: i for i, l in enumerate(oc)}, {l.layer_id: i for i, l in enumerate(nc)}
    if len(oi) != len(oc) or len(ni) != len(nc):
        return FULL                     # repeated ids: no matching
    if [i for i in oi if i in ni] != [i for i in ni if i in oi]:
        return FULL                     # reordered
    box, touched = None, ([], [])
    for lid in oi.keys() | ni.keys():
        o = oc[oi[lid]] if lid in oi else None
        n = nc[ni[lid]] if lid in ni else None
        if o is not None and n is not None and do.key(o) == dn.key(n):
            continue
        if o is not None:
            touched[0].append(oi[lid])
        if n is not None:
            touched[1].append(ni[lid])
        if o is not None and n is not None and o.is_group() and n.is_group() and _record_bytes(o) == _record_bytes(n):
            inner = _group_box(do, dn, o, n)
            if inner == FULL:
                return FULL
            box = _join(box, inner)
        elif lid in dn.lines and o is not None and n is not None and _same_look(o, n):     # its lines alone
            margin, (x0, y0, x1, y1) = _effect_margin(n)[0], dn.lines[lid]
            box = _join(box, (max(x0 - margin, 0), max(y0 - margin, 0), min(x1 + margin, n._psd.width),
                              min(y1 + margin, n._psd.height)))
        else:
            box = _join(_join(box, o is not None and _change_reach(do, o, oc, oi[lid]) or None),
                        n is not None and _change_reach(dn, n, nc, ni[lid]) or None)
    for seq, indices, doc, other, at, odoc in ((oc, touched[0], do, nc, ni, dn), (nc, touched[1], dn, oc, oi, do)):
        for i in indices:               # the clip layers above a changed one, both sides' (their base may differ)
            for k in range(i + 1, len(seq)):
                if not seq[k].clipping:
                    break
                j = at.get(seq[k].layer_id)
                box = _join(_join(box, _change_reach(doc, seq[k], seq, k)),
                            None if j is None else _change_reach(odoc, other[j], other, j))
    return box


def _change_reach(doc: _DirtyDoc, layer, seq: list, i: int):
    """doc.change(layer), but a clipped layer's met with its clip base's (it draws only inside it) and a pass-through
    group's from its children so: the fork counts any adjustment, clipped too, as the whole canvas."""
    if layer.clipping:
        j = next((j for j in range(i - 1, -1, -1) if not seq[j].clipping), None)
        if j is not None:
            base = _change_reach(doc, seq[j], seq, j)
            return base and clip_to(doc.change(layer), base)
    if not (layer.is_group() and layer.blend_mode == BlendMode.PASS_THROUGH):
        return doc.change(layer)
    from psd_tools.api.layers import AdjustmentLayer
    C, memo = doc.composite, doc.cache.content_bboxes
    box, kids = _join(C._effect_reach(layer, memo), C._content_bbox(layer, memo)), list(layer)
    for k, child in enumerate(kids):
        if isinstance(child, AdjustmentLayer) or child.is_group():
            box = _join(box, _change_reach(doc, child, kids, k))
    return box


def _group_box(do: _DirtyDoc, dn: _DirtyDoc, old, new):
    """Box group old / new (same own record) differ in as their parent sees them."""
    C = do.composite
    if not C._box_safe(old, do.cache) or not C._box_safe(new, dn.cache):
        return _join(do.change(old), dn.change(new))       # a knockout reads its backdrop
    inner = _children_box(do, dn, old, new)
    if inner in (None, FULL):
        return inner
    margin, reads = _effect_margin(new)
    if reads and (old.bbox != new.bbox or C._content_bbox(old, do.cache.content_bboxes)
                  != C._content_bbox(new, dn.cache.content_bboxes)):
        return _join(do.change(old), dn.change(new))       # its effect follows the moved bbox
    if margin and inner[2] > inner[0] and inner[3] > inner[1]:
        inner = (max(inner[0] - margin, 0), max(inner[1] - margin, 0),
                 min(inner[2] + margin, new._psd.width), min(inner[3] + margin, new._psd.height))
    return inner


_LINK_TAGS = ("LINKED_LAYER1", "LINKED_LAYER2", "LINKED_LAYER3", "LINKED_LAYER_EXTERNAL")


def _links_grown(old: PSDImage, new: PSDImage) -> bool:
    """rcache.doc_key's records alike but the linked files, every old link item still in new byte for byte: a link no
    old layer names changes no old layer's render (a types edit's Types/ link)."""
    from collections import Counter
    from psd_tools.composite import rcache
    from psd_tools.constants import Tag
    links = {getattr(Tag, name) for name in _LINK_TAGS}

    def parts(psd):
        version, blocks = psd._record.header.version, psd.tagged_blocks or {}
        raw = lambda record: rcache._record_bytes(record, version)
        same = [raw(psd._record.header), raw(psd._record.color_mode_data), repr(psd._max_alloc_bytes)]
        same += [raw(psd.image_resources[r]) if r in psd.image_resources else None for r in rcache._DOCUMENT_RESOURCES]
        same += [raw(blocks[t]) if t in blocks else None for t in rcache._DOCUMENT_BLOCKS if t not in links]
        return same, Counter((t, raw(item)) for t in rcache._DOCUMENT_BLOCKS if t in links and t in blocks
                             for item in blocks[t].data)
    (a, was), (b, now) = parts(old), parts(new)
    return a == b and not was - now


def dirty_box(old: PSDImage, new: PSDImage, lines: Optional[dict] = None) -> Optional[tuple]:
    """
    Every pixel where new (an edited copy of old) can render unlike old, from the layer records: layers matched by id
    per group, an unchanged subtree skipped, a changed one's fork change box (_change_bbox) plus its clip layers', a
    clipped layer's met with its clip base's (_change_reach: a clipped adjustment on a star or type icon stays in its
    icon's box), a group's effect reach. The whole canvas on a reorder, a document-level change (but links added:
    _links_grown), an external link, or when the walk fails (raised under PSD_DIRTY_VERIFY). lines: {layer id:
    Edit.lines} (_lines): such a type layer, its look the same (_same_look), counts as its lines and its effects' reach.

    Returns:
        The bbox on the canvas, None when nothing differs
    """
    w, h = new.width, new.height
    try:
        from psd_tools.composite import rcache
        key = rcache.doc_key(old)
        if key is None or (old.width, old.height) != (w, h) or (key != rcache.doc_key(new) and not _links_grown(old, new)):
            return (0, 0, w, h)
        box = _children_box(_DirtyDoc(), _DirtyDoc(lines), old, new)
    except Exception:
        if environ.get(DIRTY_VERIFY) == "1":
            raise
        return (0, 0, w, h)             # a layout the walk trips on: the gate as before
    if box == FULL:
        return (0, 0, w, h)
    return box and clip_to(box, (0, 0, w, h))


def gate_dirty(old: PSDImage, new: PSDImage, edits: list, keep: Optional[dict] = None,
               icon: bool = False) -> np.ndarray:
    """
    gate_untouched over the edits' regions and dirty_box(old, new) only: the card rendered there before and after
    (in the bake's RenderCache; under OLD_BESIDE before in a thread of its own, uncached), not whole twice. With
    PSD_DIRTY_VERIFY=1 both are rendered whole too and any difference (a pixel changed outside the box, a box render
    unlike the whole one's) raises RuntimeError.

    Args:
        keep: apply_jobs' dict: new is drawn as the merge draws it (text_kwargs "preview", the edits' layers pinned)
            and its box, that render and the ids go in it (the merge and gate_idempotence cut their regions from them)
        icon: edits is a card's lone stars or types edit: refused as gate_outside refuses it (its field)

    Returns:
        A canvas-sized render_all(old) valid inside the box only (baseline's crop_pair reads the regions, inside it);
        raises Refused("region") as gate_untouched does
    """
    w, h = new.width, new.height
    box = union_box([clip_to(edit.region, (0, 0, w, h)) for edit in edits if edit.region]
                    + [dirty_box(old, new, _lines(edits))])
    shown = np.zeros((h, w, 4), np.uint8)
    diff = after = None
    pins = look_kwargs(new) if keep is None else text_kwargs(new, "preview", edited_ids(edits))
    if box is not None:
        x0, y0, x1, y1 = box
        flags = dict(viewport=box, ignore_preview=True, apply_icc=False)
        drawn = lambda psd, kw: np.asarray(psd.composite(**flags, **kw).convert("RGBA"))
        if OLD_BESIDE:                          # a RenderCache is one thread's: old's draws with none: one pass
            pool = ThreadPoolExecutor(1, thread_name_prefix="old")
            ahead = pool.submit(_uncached, drawn, old, look_kwargs(old))
            pool.shutdown(wait=False)
            try:
                after = drawn(new, pins)
            finally:
                wait([ahead])
            before = ahead.result()
        else:
            before, after = drawn(old, look_kwargs(old)), drawn(new, pins)
        shown[y0:y1, x0:x1] = before
        if keep is not None:
            keep.update(box=box, after=after, edited=edited_ids(edits))
        diff = np.zeros((h, w), bool)
        diff[y0:y1, x0:x1] = _px_differ(before, after)
    if environ.get(DIRTY_VERIFY) == "1":
        _verify_dirty(old, new, box, shown, after, pins)
    if diff is not None:
        for edit in edits:
            if edit.region:
                x0, y0, x1, y1 = edit.region
                diff[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = False
        outside = int(diff.sum())
        if outside and icon:
            raise Refused("region", f"{edits[0].field}: the card changes {outside} pixels outside the edited box",
                          edits[0].field)
        if outside:
            raise Refused("region", "%s: the card changes %d pixels outside the edited boxes"
                          % (", ".join(edit.field for edit in edits), outside))
    return shown


def _lines(edits: list) -> dict:
    """{layer id: Edit.lines} of the text layers one edit alone set in place (a colours edit after it draws more)."""
    seen = Counter(layer.layer_id for edit in edits for layer in edit.layers)
    out = {}
    for edit in edits:
        for lid, one, _ in edit.parts if edit.several else [(layer.layer_id, edit, False) for layer in edit.layers[:1]]:
            if one.lines and seen[lid] == 1:
                out[lid] = one.lines
    return out


def _uncached(call, *args):
    """call(*args) with no RenderCache (another thread's lone pass reuses nothing), in an empty context: a thread that
    inherits its starter's (free-threaded builds) must not share the bake's cache."""
    return Context().run(call, *args)


def _verify_dirty(old: PSDImage, new: PSDImage, box: Optional[tuple], shown: np.ndarray,
                  after: Optional[np.ndarray], pins: dict) -> None:
    """PSD_DIRTY_VERIFY: the whole renders (no cache; new with gate_dirty's pins) equal gate_dirty's inside the box and
    each other outside it."""
    from psd_tools.composite.rcache import RenderCache
    with RenderCache(max_bytes=0):              # stores nothing: every pass rendered again
        before = render_all(old)
        whole = np.asarray(new.composite(viewport=(0, 0, new.width, new.height), ignore_preview=True, apply_icc=False,
                                         **pins).convert("RGBA"))
    x0, y0, x1, y1 = box or (0, 0, 0, 0)
    outside = _px_differ(before, whole)
    outside[y0:y1, x0:x1] = False
    if outside.any():
        raise RuntimeError(f"dirty box {box}: {int(outside.sum())} pixels changed outside it")
    if box and not (np.array_equal(before[y0:y1, x0:x1], shown[y0:y1, x0:x1])
                    and np.array_equal(whole[y0:y1, x0:x1], after)):
        raise RuntimeError(f"dirty box {box}: its render differs from the whole card's")


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


# The render record (A16): each card's look read off its stored image while it had one, for its renders after the strip
_LOOKS: "WeakKeyDictionary" = WeakKeyDictionary()   # an open PSD -> its facts (use_record)


def render_record() -> dict:
    """{PSD path: {"icc": icc_vote(), "text": {layer id: [raster_key(), "straight" or "gamma"]}}}; {} without the file."""
    return _record(RENDER_RECORD)


@lru_cache(maxsize=2)
def _record_names(name: str = RENDER_RECORD) -> dict:
    """{file name: PSD path} of the record's paths whose file name no other has (a card moved by a stars edit)."""
    seen = Counter(basename(path) for path in _record(name))
    return {basename(path): path for path in _record(name) if seen[basename(path)] == 1}


def use_record(psd: PSDImage, rel: str, facts: Optional[dict] = None) -> PSDImage:
    """psd renders with rel's facts from the render record (its file name's when rel is not in it), or the facts given
    (look_facts' before the strip), while it has no stored image; psd."""
    record = render_record()
    facts = facts or record.get(rel) or record.get(_record_names().get(basename(rel or ""), ""))
    if facts:
        _LOOKS[psd] = facts
    return psd


def raster_key(layer) -> str:
    """A short digest of a layer's stored pixels: a text fact holds while Photoshop's raster does."""
    digest = blake2b(digest_size=6)
    for channel in layer._channels:
        digest.update(channel.data)
    return digest.hexdigest()


def look_kwargs(psd: PSDImage) -> dict:
    """psd.composite()'s layer_text_gamma from the render record for a PSD with no stored image: each type layer whose
    raster is the recorded one straight or in the gamma as Photoshop drew it; {} with a stored image or no facts."""
    facts = _LOOKS.get(psd, {}).get("text")
    if not facts or psd.has_preview():
        return {}
    gamma = signature(psd.composite).parameters["text_gamma"].default
    pins = {}
    for layer in psd.descendants():
        fact = facts.get(str(layer.layer_id)) if layer.kind == "type" else None
        if fact and fact[0] == raster_key(layer):
            pins[layer.layer_id] = None if fact[1] == "straight" else gamma
    return {"layer_text_gamma": pins} if pins else {}


def _drew_gamma(psd: PSDImage, layer) -> bool:
    """look_kwargs draws the layer in the text gamma, as text_kwargs pins an edited one (no stored image, no straight
    fact on its raster)."""
    if psd.has_preview():
        return False
    fact = (_LOOKS.get(psd, {}).get("text") or {}).get(str(layer.layer_id))
    return not (fact and fact[0] == raster_key(layer) and fact[1] == "straight")


def text_choices(psd: PSDImage) -> dict:
    """{layer id: "straight" or "gamma"} as the fork's render of the whole card blends each drawn type layer off the
    stored preview (its _TextBlend decisions: the first pass's, with the layers its last pass settled straight); {}
    without a stored preview."""
    if _text_reference(psd) is None:
        return {}
    from importlib import import_module
    fork = import_module("psd_tools.composite.composite")
    made, real = [], fork._PassCache

    def capture(*args, **kwargs):
        made.append(real(*args, **kwargs))
        return made[-1]
    fork._PassCache = capture
    try:
        psd.composite(ignore_preview=True, apply_icc=False)
    finally:
        fork._PassCache = real
    passes = [cache.text for cache in made if cache.text.reference is not None]
    if not passes:
        return {}
    chosen = (passes[2] if len(passes) > 2 else passes[0]).chosen
    return {layer.layer_id: "straight" if chosen[id(layer)] else "gamma" for layer in psd.descendants()
            if layer.kind == "type" and id(layer) in chosen}


def look_facts(psd: PSDImage, png: Optional[Image.Image] = None) -> dict:
    """The render record's facts read off psd's stored image (before its strip): {"icc": icc_vote against png, "text":
    {layer id: [raster_key(), text_choices()]}}, each only when found; {} without a stored image."""
    if not psd.has_preview():
        return {}
    facts: dict = {}
    vote = icc_vote(psd, png) if png is not None else None
    if vote is not None:
        facts["icc"] = [round(vote[0], 4), vote[1]]
    layers = {layer.layer_id: layer for layer in psd.descendants() if layer.kind == "type"}
    text = {str(lid): [raster_key(layers[lid]), how] for lid, how in sorted(text_choices(psd).items())}
    if text:
        facts["text"] = text
    return facts


def records_of(before: PSDImage, after: Optional[PSDImage], rel: str, parser: Optional[PSDParser] = None,
               png: Optional[Image.Image] = None, name: str = "") -> dict:
    """What a rewrite of before (a conversion, a bake, a link redraw) hands the records: its look facts while it still
    has a stored image (look_facts), and its icons' earlier cores when after moved them (spot_entries; needs a parser);
    {"render": {rel: facts}, "spot": {rel: entries}}, each only when found."""
    out: dict = {}
    facts = look_facts(before, png) if before.has_preview() else {}
    if facts:
        out["render"] = {rel: facts}
    entries = spot_entries(before, after, parser, rel, name) if after is not None and parser is not None else {}
    if entries:
        out["spot"] = {rel: entries}
    return out


def _alike(path: str) -> tuple:
    """(folder, file name casefolded, letters and digits only): Bob-ulk.psd and Bob-Ulk.psd alike."""
    return dirname(path), "".join(c for c in basename(path).casefold() if c.isalnum())


def merge_records(render: dict, spot: dict, docs: list, paths=(), moved=None) -> tuple:
    """(render, spot) with each doc's records on top, in order (records_of's form): a later card's facts and entries win
    key by key, a text fact or an entry no later doc names is kept. Then an entry follows moved ({old path: new path},
    git's renames) onto a path with none; a path still gone from paths (the tree's PSDs) moves to the one PSD there
    with its file name (a stars edit's move), else to the one in its folder named alike (_alike), when that has none;
    none: kept as it is."""
    render, spot = {k: dict(v) for k, v in render.items()}, {k: dict(v) for k, v in spot.items()}
    for doc in docs:
        for rel, facts in (doc.get("render") or {}).items():
            old = render.get(rel) or {}
            render[rel] = {**old, **facts, **({"text": {**old.get("text", {}), **facts["text"]}} if "text" in facts else {})}
        for rel, entries in (doc.get("spot") or {}).items():
            spot[rel] = {**spot.get(rel, {}), **entries}
    for record in (render, spot):
        for old, new in (moved or {}).items():
            if old in record and new not in record:
                record[new] = record.pop(old)
    if paths:
        names, alike = Counter(map(basename, paths)), Counter(map(_alike, paths))
        by_name, by_alike = {basename(p): p for p in paths}, {_alike(p): p for p in paths}
        for record in (render, spot):
            for rel in [rel for rel in record if rel not in paths]:
                to = by_name[basename(rel)] if names[basename(rel)] == 1 else \
                    by_alike[_alike(rel)] if alike[_alike(rel)] == 1 else ""
                if to and to not in record:
                    record[to] = record.pop(rel)
    return render, spot


def _text_reference(psd: PSDImage) -> Optional[np.ndarray]:
    """The fork's read of psd's stored preview for text blends (kept with psd until the preview changes); None without
    a stored 8-bit RGB preview or a read."""
    if psd.color_mode != ColorMode.RGB or psd.depth != 8 or not psd.has_preview():
        return None
    from psd_tools.composite.composite import _text_reference as read
    return read(psd)


def _share_preview(psd: PSDImage, copy: PSDImage) -> None:
    """copy, opened from psd's bytes (edit_copy), takes psd's stored preview as the fork read it (one decode a card)
    and its render record facts (use_record)."""
    if psd in _LOOKS:
        _LOOKS[copy] = _LOOKS[psd]
    reference = _text_reference(psd)
    if reference is None:
        return
    from psd_tools.composite.composite import _TEXT_REFERENCES
    stored = copy._record.image_data
    if stored.compression == psd._record.image_data.compression and stored.data == psd._record.image_data.data:
        _TEXT_REFERENCES[copy] = (stored.data, reference)


def preview_image(psd: PSDImage) -> Optional[Image.Image]:
    """
    psd.topil(apply_icc=False) from the fork's kept read (_text_reference: the same values on every 8-bit RGB card,
    swept 2026-10-25), so icc_vote and the renders decode the stored preview once; a copy of it.
    """
    reference = _text_reference(psd)
    if reference is not None:
        return Image.fromarray(reference.copy())
    return psd.topil(apply_icc=False) if psd.has_preview() else None


def icc_vote(psd: PSDImage, png: Image.Image) -> Optional[tuple]:
    """
    The harness's ICC vote (score_cards.py render) over the stored preview: of the pixels the profile's conversion
    moves by ICC_MOVED or more, the share where the PNG is nearer the unconverted values.

    Args:
        psd: The card
        png: Its official PNG

    Returns:
        (vote, moved pixels); without a stored preview the render record's (use_record), else None; None without a
        foreign profile, or when the preview is not RGB(A) of the PNG's size
    """
    profile = foreign_profile(psd)
    if profile is None:
        return None
    if not psd.has_preview():
        vote = _LOOKS.get(psd, {}).get("icc")
        return tuple(vote) if vote else None
    return _vote(preview_image(psd), png, profile)


def _vote(preview: Optional[Image.Image], png: Image.Image, profile: bytes) -> Optional[tuple]:
    """icc_vote's count over an image in the PSD's own values (its stored preview, or a render of it)."""
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
        False for a raw export; True otherwise (no foreign profile, no evidence). With neither a stored preview nor a
        render record vote, the vote is over the fork's render of the card in the preview's place
    """
    vote = icc_vote(psd, png)
    profile = foreign_profile(psd)
    if vote is None and profile is not None and not psd.has_preview():
        drawn = psd.composite(ignore_preview=True, apply_icc=False, **look_kwargs(psd))
        vote = _vote(drawn, png, profile)
    return vote is None or vote[1] < ICC_MIN_PX or vote[0] < ICC_RAW_VOTE


def text_kwargs(psd: PSDImage, text: str, edited: tuple = ()) -> dict:
    """
    psd.composite()'s keywords for a TEXT_BLENDS key. "preview" blends as the stored preview shows, or with none as the
    render record says (look_kwargs); under every key but "gamma" (all in it) the edited type layers are pinned to the
    text gamma: Photoshop blends a layer it redraws in the gamma, on a straight card too (Wooden_Spikes).

    Args:
        psd: The card
        text: A TEXT_BLENDS key
        edited: Layer IDs of the edited layers

    Returns:
        The keywords
    """
    kwargs = dict(TEXT_BLENDS[text], **(look_kwargs(psd) if text == "preview" else {}))
    if text != "gamma" and edited:
        kwargs["layer_text_gamma"] = {**kwargs.get("layer_text_gamma", {}),
                                      **dict.fromkeys(edited, signature(psd.composite).parameters["text_gamma"].default)}
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
        The render, region-sized; refused when the fork cannot draw the card (drawable)
    """
    drawable(psd)
    return psd.composite(viewport=region, ignore_preview=True, apply_icc=icc, **text_kwargs(psd, text, edited))


def crop_pair(psd: PSDImage, full: Optional[np.ndarray], region: tuple, icc: bool = True,
              at: Optional[tuple] = None) -> Optional[tuple]:
    """
    (render, raw) of the region under "preview", nothing edited, cut from render_all() of the card (the same pixels:
    review 2026-10-24), or from a render over box `at` (gate_dirty's keep); None without it or when the region leaves it.
    """
    x0, y0, x1, y1 = region
    ax0, ay0, ax1, ay1 = at or (0, 0, psd.width, psd.height)
    if full is None or x0 < ax0 or y0 < ay0 or x1 > ax1 or y1 > ay1 or x1 <= x0 or y1 <= y0:
        return None
    raw = Image.fromarray(np.ascontiguousarray(full[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0]), "RGBA")
    if not icc or Resource.ICC_PROFILE not in psd.image_resources:
        return raw, raw
    return post_process(raw, None, psd.image_resources.get_data(Resource.ICC_PROFILE)), raw


def baseline(psd: PSDImage, png: Image.Image, region: tuple, icc: bool = True,
             full: Optional[np.ndarray] = None) -> tuple:
    """
    Render the unedited region and measure it against the official PNG (gate parity).

    Args:
        psd: The card before the edit
        png: Its official PNG
        region: The edit's padded region
        icc: render_region()'s
        full: render_all() of psd: the "preview" render is cut from it (crop_pair); None renders the region

    Returns:
        (render, measurement); the measurement's "text" is the first TEXT_BLENDS key whose render is within the
        bar, else the preview blend's (a miss of the bar bakes; apply_jobs refuses a miss whose edit
        renders no change). Raises Refused("parity") past PARITY_FLOOR only.
    """
    first = None
    for text in TEXT_BLENDS:
        cut = crop_pair(psd, full, region, icc) if text == "preview" else None
        render = cut[0] if cut else render_region(psd, region, icc, text)
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


def whole_redraw(old: PSDImage, raw: bytes, edit: Edit, png: Image.Image, fonts_dir: str, icc: bool = True,
                 text: str = "preview") -> dict:
    """
    The edited layer's old text redrawn whole by the engine on a copy of the card, against the official PNG over the
    edit's region: what an edit the engine redrew whole (Edit.whole) prints on the lines it left. The parity bar sees
    only the stored raster and gate_idempotence the engine's own render (review 2026-10-24: Wendy's 2CD badge moved
    6 px, '(3 points)' 8 px, the box 15.7% bad, no note).
    """
    copy = edit_copy(old, raw)
    ids = [layer.layer_id for layer in edit.layers]
    for layer in copy.descendants():
        if layer.layer_id in ids:
            follow_kerning(copy, layer)         # as the edit drew it (B12)
            layer.rasterize(fonts=font_map(fonts_dir, layer))     # in the face its pixels show (R3)
    return _measure(render_region(copy, edit.region, icc, text, tuple(ids)), png.crop(edit.region))


# The soft notes (owner 2026-11-07: non-issues): a bake's own remarks on a card that baked, told nowhere
SOFT_NOTE = re_compile(r"^overflow: |^types: .*spot.*: check it$|^[a-z]+: cells by place|deleted: the edit changes"
                       r"|: also printed by ")


def real_notes(notes: list) -> list:
    """A bake's notes without the soft ones."""
    return [note for note in notes if not SOFT_NOTE.search(str(note))]


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
    changed = _px_differ(np.asarray(base.convert("RGBA")), after)
    patch = out[y0:y1, x0:x1]
    patch[changed] = after[changed]
    return Image.fromarray(out).convert(png.mode), changed


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
def body_sizes(old_path: str, new_path: str) -> tuple:
    """(old, new) file sizes, both without their image data section when new stripped old's merged image (strip())."""
    old, new = getsize(old_path), getsize(new_path)
    (a, b, merged), (c, d, kept) = _image_section(old_path), _image_section(new_path)
    return (old - (b - a), new - (d - c)) if merged and not kept else (old, new)


def gate_size(old_path: str, new_path: str) -> None:
    """The saved PSD stays within SIZE_TOLERANCE of the old one or SIZE_FLOOR bytes, the old merged image left out of
    both when the save stripped it (a card's first strip: body_sizes); else Refused("size")."""
    old, new = body_sizes(old_path, new_path)
    if abs(new - old) > max(SIZE_TOLERANCE * old, SIZE_FLOOR):
        raise Refused("size", f"PSD {old} -> {new} bytes")


def _fx_bytes(key, block) -> bytes:
    """A block's bytes, an effects block's at version 0: save_psd's repair is no change to the gates."""
    raw = block.tobytes()
    return raw[:12] + bytes(4) + raw[16:] if key in EFFECT_BLOCKS[:3] and len(raw) >= 16 else raw


def layer_records(path) -> list:
    """
    Each layer record's identity-free fingerprint: name, flags, rect, channels (id, length, pixel digest), mask, blend
    and tagged blocks by key;
    path a file, an open PSD (read as opened: no edit on it) or its layer_records() already.
    """
    if isinstance(path, list):
        return path
    psd = path if isinstance(path, PSDImage) else PSDImage.open(path)
    out = []
    info = psd._record.layer_and_mask_information.layer_info
    records = info.layer_records or []
    for record, data in zip(records, [*(info.channel_image_data or []), *[[]] * len(records)]):
        blocks = record.tagged_blocks or {}
        mask = record.mask_data
        out.append({"name": blocks.get_data(Tag.UNICODE_LAYER_NAME, record.name) if blocks else record.name,
                    "flags": record.flags.tobytes(),
                    "rect": (record.top, record.left, record.bottom, record.right),
                    "channels": tuple((int(channel.id), channel.length, _digest(part))
                                      for channel, part in zip_longest(record.channel_info, data)),
                    "mask": mask.tobytes() if mask is not None else b"",
                    "blend": (record.blend_mode, record.opacity, record.clipping),
                    "blocks": tuple((bytes(key), _fx_bytes(key, block))
                                    for key, block in (record.tagged_blocks or {}).items()),
                    **(_mask_parts(mask) if mask is not None else {})})
    return out


def _comp_record(a: dict, b: dict) -> Optional[str]:
    """What a layer-comp bar's record changed besides its rect, channels and SoLd block (set_comp), else None."""
    if [key for key, _ in a["blocks"]] != [key for key, _ in b["blocks"]]:
        return "changed its blocks' list"
    placed = (Tag.SMART_OBJECT_LAYER_DATA1.value, Tag.SMART_OBJECT_LAYER_DATA2.value)
    other = [key for (key, x), (_, y) in zip(a["blocks"], b["blocks"]) if x != y and key not in placed]
    return f"changed {', '.join(map(repr, other))}" if other else None


def _digest(channel) -> bytes:
    """A channel's compression and data, hashed (sha1: hardware on arm64 and x86): a same-length change to a layer's
    pixels shows. Compared within a run only."""
    if channel is None:
        return b""
    digest = sha1(pack(">H", channel.compression.value))
    digest.update(channel.data)
    return digest.digest()


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


def _record_moved(a: dict, b: dict, dx: int, dy: int, keys: set, flags: bool = False, renamed: bool = False,
                  masked: bool = False, scaled: Optional[tuple] = None) -> Optional[str]:
    """
    Why record b is not record a moved by (dx, dy) (a layer _shift moved, or a copy of it); None when it is: the rect
    and mask moved, the channels, blending and other blocks the same, only `keys` blocks changed; flags: one flag
    byte may differ (the layer was shown or hidden too); renamed: the name may differ (a copy's); masked: b gained a
    pixel mask a had none of (split_stars), its one more channel the mask's; scaled: a shrunk star's rect (option B),
    moved from there, its channels drawn again (the same ones) and no mask.
    """
    if b["rect"] != _moved_rect(a["rect"] if scaled is None else scaled, dx, dy):
        return f"rect {a['rect']} -> {b['rect']}"
    if scaled is not None:
        if a["mask"] or b["mask"] or sorted(c[0] for c in a["channels"]) != sorted(c[0] for c in b["channels"]):
            return "channels or mask"
        a = {**a, "channels": b["channels"]}
    channels = b["channels"]
    if masked:
        if a["mask"] or not b["mask"] or [c[0] for c in channels].count(-2) != 1:
            return "mask (no new pixel mask)"
        channels = tuple(c for c in channels if c[0] != -2)
    if a["channels"] != channels or a["blend"] != b["blend"]:
        return "channels or blending"
    if not renamed and a["name"] != b["name"]:
        return "name"
    if not masked and (a.get("mask_norm") != b.get("mask_norm")
                       or _moved_mask(a.get("mask_box"), dx, dy) != b.get("mask_box")):
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
    ids = [channel[0] for channel in record["channels"]]
    return bool(ids) and ids[0] == -1 and ids[1:] == list(range(len(ids) - 1))


def _fx_rebuilt(a: dict, b: dict, old: str, name: str, styled: Optional[dict] = None) -> Optional[str]:
    """
    Why record b is not record a's FX layer cut again in place (Edit.rebuilt), or None: a pixel record, its name the
    declared one, its blocks a's byte for byte but its unicode name when renamed: its own style, no effect block changed
    (owner q6; rect and channels may differ, nothing else: gate_bytes); styled (Edit.restyled: a chart status word's),
    its effect blocks exactly those.
    """
    if a["name"] != old or b["name"] != name:
        return f"name {a['name']!r} -> {b['name']!r}"
    if not _pixel_record(b):
        return "not a pixel record"
    before, after = dict(a["blocks"]), dict(b["blocks"])
    may = {bytes(Tag.UNICODE_LAYER_NAME)} if old != name else set()
    if styled is not None:
        effects = {bytes(key) for key in EFFECT_BLOCKS}
        if {key: data for key, data in after.items() if key in effects} != styled:
            return "effect blocks not the chart's"
        may |= effects
    changed = sorted(key.decode("latin-1") for key in set(before) | set(after) if before.get(key) != after.get(key)
                     and key not in may)
    return f"blocks {', '.join(changed)}" if changed else None


def gate_bytes(old_path, new_path, edits: list) -> tuple:
    """
    Compare the layer records of the old and new PSD: only the edited layers change, and only in the fields their
    kind of edit touches (a toggle: exactly one flag byte per toggled digit, plus the effect blocks its Edit declares
    moved, the same bytes leaving one digit's record and landing on another's, or leaving only when cleared; a text
    edit: rect, channels, EngineData; its FX
    layers: exactly the declared records gone and come, the new ones pixel records carrying the chart's effect blocks
    byte for byte, the rebuilt ones their rect and channels (and the declared name; never an effect block,
    _fx_rebuilt); a clipped layer hidden (Edit.hidden): one flag byte; a star or type icon shown or hidden: one flag byte; one
    moved: its rect and mask moved by the declared offset, only its transform and anchor blocks changed (_record_moved);
    a star's copy: the source's record moved, a new id and name; a type icon from Types/: a pixel record carrying the
    reference icon's effect blocks). Records are paired in file order once the FX records declared removed and added
    and the copies and icons declared added are taken out. A refusal over a stars or types edit's record carries its
    field. old_path and new_path: what layer_records() takes (a file, an open PSD, its records).

    Returns:
        ([(index, name, changed fields)] in file order ("removed" at the old index, "added" at the new one), differing
        flag bytes, effect block bytes that landed on a digit, an FX layer or a type icon)
    """
    old, new = layer_records(old_path), layer_records(new_path)
    dropped = Counter(name for edit in edits for name in edit.dropped)
    added = {name: blocks for edit in edits for name, blocks in edit.fx}
    allowed, names = {}, {}
    for edit in edits:
        kind = "stat" if edit.field in (*STAT_FIELDS, "creatures") else "text"
        for index in edit.records:
            allowed[index] = (kind, ALLOWED_RECORD_FIELDS[kind])
        for index in edit.hidden:
            allowed[index] = ("overlay", ALLOWED_RECORD_FIELDS["overlay"])
        for index, _, name in edit.rebuilt:
            allowed[index] = ("fx", ALLOWED_RECORD_FIELDS["fx"])
            names[index] = name                 # a rebuilt FX layer's declared name (_fx_rebuilt checks it too)
        for index, _ in edit.restyled:          # given the chart's style (cut again or not)
            allowed[index] = ("fx", ALLOWED_RECORD_FIELDS["fx"])
    for edit in edits:
        for index in edit.comps:                # a layer-comp bar: its rect, pixels and SoLd block (_comp_record)
            allowed[index] = ("comp", ALLOWED_RECORD_FIELDS["comp"])
    rebuilt = {index: spec for edit in edits for index, *spec in edit.rebuilt}
    restyled = {index: blocks for edit in edits for index, blocks in edit.restyled}
    toggles = {index for edit in edits for index in edit.toggled}
    allowed.update({index: ("toggle", ALLOWED_RECORD_FIELDS["toggle"]) for index in toggles})
    moved = {index: (source, dx, dy) for edit in edits for index, source, dx, dy in edit.shifted}
    scaled = {index: rect for edit in edits for index, rect in edit.scaled.items()}     # old index -> its shrunk rect
    masked = {index for edit in edits for index in edit.masked}
    inserted = {index: ("copy", source, dx, dy) for edit in edits for index, source, dx, dy in edit.copied}
    inserted.update({index: ("made", blocks) for edit in edits for index, blocks in edit.made})
    inserted.update({index: ("ability",) for edit in edits for index in edit.added})
    owner = {index: edit.field for edit in edits
             for index in [*edit.toggled, *(record[0] for record in edit.shifted + edit.copied + edit.made)]}
    bar = {index: (edit.field, spec) for edit in edits for index, spec in edit.bar}  # a stat on a card maker's own bar
    removed = {index for edit in edits for index in edit.removed}         # a part edit's deleted layers
    renumbered = {index for edit in edits for index in edit.renumbered}   # their stories' later layers
    changed, flag_bytes, block_bytes, old_of = [], 0, 0, {}
    i = j = 0
    while i < len(old) or j < len(new):
        a = old[i] if i < len(old) else None
        b = new[j] if j < len(new) else None
        if a is not None and i in removed:
            removed.discard(i)
            changed.append((i, a["name"], ["removed"]))
            i += 1
            continue
        if a is not None and dropped.get(a["name"]):
            dropped[a["name"]] -= 1
            changed.append((i, a["name"], ["removed"]))
            i += 1
            continue
        if b is not None and j in inserted:     # a star's copy, a type icon made from Types/, an added ability
            spec = inserted.pop(j)
            if spec[0] == "ability":            # an added ability: a type layer named Ability (§4.7)
                if b["name"] != "Ability" or bytes(Tag.TYPE_TOOL_OBJECT_SETTING) not in dict(b["blocks"]):
                    raise Refused("bytes", f"layer {j} {b['name']!r} is not the declared Ability type layer")
            elif spec[0] == "made":
                blocks = dict(b["blocks"])
                if not _pixel_record(b) or any(blocks.get(key) != data for key, data in spec[1].items()):
                    raise Refused("bytes", f"type icon {b['name']!r} is not the declared pixel layer", owner.get(j))
                block_bytes += sum(len(data) for data in spec[1].values())
            else:
                why = _record_moved(old[spec[1]], b, spec[2], spec[3], COPY_BLOCKS, j in toggles, True, j in masked,
                                    scaled.get(spec[1]))
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
        if j in bar:                            # a bar layer: exactly its declared eye, name, place, mask and style
            stat, spec = bar.pop(j)
            why = _bar_record(a, b, spec) or ("" if fields else "the edit changed it, but the saved file did not")
            if why:                             # card_checks' note keeps the words before ';'
                kind = "smart object" if {bytes(key) for key in PLACED} & {key for key, _ in a["blocks"]} else "layer"
                raise Refused("bytes", f"the {stat} bar's {kind} {a['name']!r}: {why}; the bake saves nothing: set "
                                       "this stat in Photoshop")
            flag_bytes += sum(1 for x, y in zip(a["flags"], b["flags"]) if x != y)
            changed.append((j, a["name"], fields))
        elif j in moved:                        # a star or type icon moved: its rect, mask, transform and anchor
            source, dx, dy = moved[j]
            why = f"paired with old layer {i}" if source != i else _record_moved(a, b, dx, dy, MOVE_BLOCKS, j in toggles,
                                                                                    masked=j in masked, scaled=scaled.get(i))
            if why:
                raise Refused("bytes", f"moved layer {j} {a['name']!r}: {why}", owner.get(j))
            flag_bytes += sum(1 for x, y in zip(a["flags"], b["flags"]) if x != y)
            if fields:
                changed.append((j, a["name"], fields))
        elif fields:
            kind, ok = allowed.get(j, ("", set()))
            if j in renumbered and j not in allowed:     # its TextIndex moved down: that alone
                why = _text_index_only(a, b) if fields == ["blocks"] else f"changed {', '.join(fields)}"
                if why:
                    raise Refused("bytes", f"layer {j} {a['name']!r} (its TextIndex renumbered) {why}")
                kind, ok = "story", {"blocks"}
            if not set(fields) <= ok:
                raise Refused("bytes", f"layer {j} {a['name']!r} changed {', '.join(fields)}")
            if "name" in fields and b["name"] != names.get(j):
                raise Refused("bytes", f"layer {j} {a['name']!r} renamed {b['name']!r}, not {names.get(j)!r}")
            if "flags" in fields:
                differing = sum(1 for x, y in zip(a["flags"], b["flags"]) if x != y)
                if kind in ("stat", "overlay", "toggle") and differing != 1:
                    raise Refused("bytes", f"{kind} {a['name']!r} changed {differing} flag bytes", owner.get(j))
                flag_bytes += differing
            if kind == "fx":
                why = _fx_rebuilt(a, b, *rebuilt.get(j, (a["name"], a["name"])), restyled.get(j))
                if why:
                    raise Refused("bytes", f"FX layer {j} {a['name']!r} is not the declared rebuild: {why}")
            if kind == "comp":
                why = _comp_record(a, b)
                if why:
                    raise Refused("bytes", f"layer-comp bar {j} {a['name']!r} {why}")
            changed.append((j, a["name"], fields))
        i += 1
        j += 1
    if any(dropped.values()) or added or inserted or removed or bar:
        raise Refused("bytes", "an FX layer, star copy, type icon, ability or deleted layer declared removed or added "
                               "is not")
    hidden = {index for edit in edits for index in edit.hidden}
    toggled = {index for edit in edits if edit.field in (*STAT_FIELDS, "creatures") for index in edit.records}
    if (toggled | hidden | toggles) - {index for index, _, fields in changed if fields != ["removed"]}:
        raise Refused("bytes", "a toggled digit, hidden overlay or shown or hidden star or icon did not change")
    # effect blocks on the toggled digits: exactly the declared moves, byte for byte
    moves = [move for edit in edits if edit.field in (*STAT_FIELDS, "creatures") for move in edit.moves]
    blocks_of = lambda records, index: dict(records[index]["blocks"])
    left = {(src, key): blocks_of(old, old_of[src]).get(key) for src, _, key in moves}
    landed = {(dst, key): blocks_of(old, old_of[src]).get(key) for src, dst, key in moves if dst is not None}
    if None in left.values():
        raise Refused("bytes", "a moved effect block was not on its digit")
    comped = {index for edit in edits for index in edit.comps}      # their SoLd: _comp_record above
    for index in toggled - comped:              # each digit's blocks as the moves leave them (a block may land equal to
        before, after = blocks_of(old, old_of[index]), blocks_of(new, index)    # the one it replaced: Dark Mind's lrFX)
        expected = {key: data for key, data in before.items() if (index, key) not in left}
        expected.update({key: data for (dst, key), data in landed.items() if dst == index})
        if after != expected:
            raise Refused("bytes", "effect blocks changed beyond the declared moves")
    return changed, flag_bytes, block_bytes + sum(len(data) for data in landed.values())


def _bar_record(a: dict, b: dict, spec: dict) -> Optional[str]:
    """Why record b is not record a changed as a stat edit declared (_bar_spec): eye, name, place, mask place and
    effect blocks, nothing else; None when it is."""
    dx, dy = spec["move"]
    if b["rect"] != _moved_rect(a["rect"], dx, dy):
        return f"it landed at {b['rect']}, not where the edit moved it"
    user_mask = -2                              # ChannelID.USER_LAYER_MASK: a redrawn mask's raster
    channels = (lambda record: tuple(c for c in record["channels"] if c[0] != user_mask)) if spec.get("remask") \
        else (lambda record: record["channels"])
    if channels(a) != channels(b) or a["blend"] != b["blend"]:
        return "its pixels or blend mode changed"
    if b["name"] != (a["name"] if spec["name"] is None else spec["name"]):
        return f"the saved file names it {b['name']!r}"
    if spec.get("remask"):                      # its raster and width new, its rows and settings the same
        box_a, box_b = a.get("mask_box") or (), b.get("mask_box") or ()
        if a.get("mask_norm") != b.get("mask_norm") or box_a[0:1] + box_a[2:3] != box_b[0:1] + box_b[2:3]:
            return "its mask is not the one the edit drew"
    elif a.get("mask_norm") != b.get("mask_norm") or _moved_mask(a.get("mask_box"), *spec["mask"]) != b.get("mask_box"):
        return "its mask is not where the edit put it"
    if sum(1 for x, y in zip(a["flags"], b["flags"]) if x != y) != (1 if spec["flags"] else 0):
        return ("the edit showed or hid it, but the saved file did not" if spec["flags"] else
                "the saved file shows or hides it (or changed another layer flag), though the edit left it alone")
    before, after = dict(a["blocks"]), dict(b["blocks"])
    if any(after.get(key) != data for key, data in spec["blocks"].items()):
        return "its layer style is not the one the edit set"
    free = set(spec["blocks"]) | ({bytes(Tag.UNICODE_LAYER_NAME)} if spec["name"] is not None else set()) \
        | (MOVE_BLOCKS if dx or dy else set())
    changed = sorted(key.decode("latin-1") for key in set(before) | set(after)
                     if before.get(key) != after.get(key) and key not in free)
    return f"its {', '.join(changed)} data changed too" if changed else None


def _text_index_only(a: dict, b: dict) -> Optional[str]:
    """None when two layer records differ only in their TySh block, in place, within one TextIndex's 4 bytes."""
    key = bytes(Tag.TYPE_TOOL_OBJECT_SETTING)
    x, y = dict(a["blocks"]), dict(b["blocks"])
    if set(x) != set(y) or any(x[k] != y[k] for k in x if k != key):
        return "changed blocks other than its TySh"
    if key not in x or len(x[key]) != len(y[key]):
        return "changed its TySh block's size"
    diff = [k for k, (p, q) in enumerate(zip(x[key], y[key])) if p != q]
    return "changed its TySh block beyond a TextIndex" if diff and diff[-1] - diff[0] >= 4 else None


def _raw(element) -> bytes:
    buf = BytesIO()
    element.write(buf)
    return buf.getvalue()


FILTER_KEYS = {bytes(Tag.FILTER_EFFECTS1), bytes(Tag.FILTER_EFFECTS2), bytes(Tag.FILTER_EFFECTS3)}


def _item_bytes(item, rect: Optional[tuple] = None, uuid: Optional[str] = None, mask: Optional[list] = None) -> bytes:
    """An FEid item's bytes, with its rectangle, name or mask rectangle replaced."""
    one = deepcopy(item)
    one.rectangle = one.rectangle if rect is None else rect
    one.uuid = one.uuid if uuid is None else uuid
    if mask is not None:
        one.extra.rectangle = mask
    buf = BytesIO()
    one.write(buf)
    return buf.getvalue()


def _moved_feid(rect, dx: int, dy: int):
    """An FEid (top, left, bottom, right) moved, as the type it was."""
    top, left, bottom, right = rect
    return type(rect)((top + dy, left + dx, bottom + dy, right + dx))


def gate_filters(old: PSDImage, new: PSDImage, edits: list) -> None:
    """
    The document's FEid block (the smart filters' sources and masks) holds the old bytes but what the edits declare
    (Edit.filters, S1), as Photoshop writes them (The Flash's 'pngwing.com copy 3'): a moved smart-filtered layer's item
    moved by its offset, a copy's its source's moved by the copy's and named after it (the planes the same bytes, the
    mask's rectangle moved only when linked to the layer), the items in file order of their layers; raises
    Refused("bytes").
    """
    declared = {uid: (src, dx, dy) for edit in edits for uid, src, dx, dy in edit.filters}
    a, b = _filter_block(old), _filter_block(new)
    if not declared:
        before, after = _document_blocks(old), _document_blocks(new)
        if any(before.get(key) != after.get(key) for key in FILTER_KEYS) and \
                not _sources_only(a, b, {uid for edit in edits for uid in edit.sources}):
            raise Refused("bytes", "the FEid block changed: no smart-filtered layer moved or copied")
        return
    if a is None or b is None or a.version != b.version:
        raise Refused("bytes", "the FEid block came, went or changed its version")
    olds = {(item.uuid or "").strip("\x00"): item for item in a}
    order = [layer for layer in new.descendants() if not layer.is_group() and layer.kind == "smartobject" and _filtered(layer)]
    linked = {layer.smart_object.placed_id: bool(getattr(layer.smart_object.filters.get(b"filterMaskLinked"), "value", False))
              for layer in order}
    seen = set()
    for item in b:
        uid = (item.uuid or "").strip("\x00")
        src, dx, dy = declared.get(uid, (uid, 0, 0))
        base = olds.get(src)
        if base is None or uid in seen:
            raise Refused("bytes", f"FEid item {uid}: no item's, or twice")
        mask = getattr(base.extra, "rectangle", None)
        mask = _moved_feid(mask, dx, dy) if mask and linked.get(uid) else None
        if _item_bytes(item) != _item_bytes(base, _moved_feid(base.rectangle, dx, dy), item.uuid, mask):
            raise Refused("bytes", f"FEid item {uid}: not {src}'s moved by ({dx}, {dy})")
        seen.add(uid)
    if set(olds) - seen:
        raise Refused("bytes", f"FEid items gone: {sorted(set(olds) - seen)}")
    if [(item.uuid or "").strip("\x00") for item in b] != [layer.smart_object.placed_id for layer in order] and \
            [(item.uuid or "").strip("\x00") for item in a] == [layer.smart_object.placed_id for layer in old.descendants()
                                                              if not layer.is_group() and layer.kind == "smartobject"
                                                              and _filtered(layer)]:
        raise Refused("bytes", "FEid items out of their layers' file order")


def _sources_only(a, b, uids: set) -> bool:
    """FEid blocks a and b alike but the planes of the items uids (a placed layer drawn again: set_placed_title's)."""
    import attr
    if not uids or a is None or b is None or a.version != b.version or len(a) != len(b):
        return False
    for x, y in zip(a, b):
        uid = (x.uuid or "").strip("\x00")
        if uid != (y.uuid or "").strip("\x00"):
            return False
        if uid not in uids:
            if _item_bytes(x) != _item_bytes(y):
                return False
        elif any(getattr(x, one.name) != getattr(y, one.name) for one in attr.fields(type(x)) if one.name != "channels"):
            return False
    return True


def _links_added(old: PSDImage, new: PSDImage, uuids) -> bool:
    """new's lnkE items are old's, byte for byte, then exactly the items uuids."""
    def items(psd):
        blocks = psd._record.layer_and_mask_information.tagged_blocks
        return list(blocks.get_data(Tag.LINKED_LAYER_EXTERNAL) or []) if Tag.LINKED_LAYER_EXTERNAL in blocks else []
    was, now = items(old), items(new)
    return bool(uuids) and [item.tobytes() for item in was] == [item.tobytes() for item in now[:len(was)]] and \
        [str(item.uuid).strip("\x00") for item in now[len(was):]] == list(uuids)


def _links_relinked(old: PSDImage, new: PSDImage, uuids) -> bool:
    """new's lnkE items are old's, byte for byte, but the items uuids, which differ only in size and contentID."""
    import attr
    def items(psd):
        blocks = psd._record.layer_and_mask_information.tagged_blocks
        return list(blocks.get_data(Tag.LINKED_LAYER_EXTERNAL) or []) if Tag.LINKED_LAYER_EXTERNAL in blocks else []
    was, now, uuids = items(old), items(new), set(uuids)
    if not uuids or len(was) != len(now):
        return False
    for a, b in zip(was, now):
        uuid = str(a.uuid).strip("\x00")
        if uuid != str(b.uuid).strip("\x00"):
            return False
        if uuid not in uuids:
            if a.tobytes() != b.tobytes():
                return False
        elif any(getattr(a, one.name) != getattr(b, one.name) for one in attr.fields(type(a))
                 if one.name not in ("filesize", "content_descriptor")):
            return False
    return True


def _document_blocks(psd: PSDImage) -> dict:
    return {bytes(key): block.tobytes() for key, block in (psd._record.layer_and_mask_information.tagged_blocks or {}).items()}


def _story_fonts(root) -> Optional[list]:
    """The document's Txt2 FontSet (its Resources list), or None."""
    node = root
    for key in STORY_STYLE_KEYS["fonts"]:
        node = node.get(key) if hasattr(node, "get") else None
    return node if isinstance(node, (list, ListElement)) else None


def _story_keys(root) -> tuple:
    """(the fork's numbered keys, the stories list) of a Txt2 root."""
    from psd_tools.api import text_edit
    keys = text_edit._NUMBERED
    objects = root.get(keys.objects) if hasattr(root, "get") else None
    items = objects.get(keys.stories) if hasattr(objects, "get") else None
    if items is None:
        raise Refused("bytes", "a Txt2 block without text objects")
    return keys, items


def _story_font_name(entry) -> str:
    for key in STORY_STYLE_KEYS["font"]:
        entry = entry[key]
    return str(getattr(entry, "value", entry))


def gate_stories(old: PSDImage, new: PSDImage, edited, fonts=(), removed=(), added=(), filters: bool = False,
                 links=(), also: frozenset = frozenset(), frames=(), relinked=()) -> list:
    """
    Photoshop reads a type layer's text from its Txt2 story and rebuilds the layer's TySh from it on save, so an edit
    to TySh alone is undone the first time the card is opened (2026-10-22: B.F.B. went back to B.F.B). Every type
    layer keeps its TextIndex and its story holds its layer's text and styles (story_style_drift), only the edited
    layers' stories changed, and the rest of Txt2 and every other document-wide block are the old bytes, but the
    faces the edits added at the end of Txt2's FontSet (Edit.fonts: a badge typed on a card without the icon font).

    Args:
        old: The card as opened
        new: The saved card, reopened
        edited: TextIndex of each layer whose story may change (taken before the edit)
        fonts: The faces the edits added (Edit.fonts)
        removed: TextIndex of each deleted layer (Edit.removed_stories): the old card is compared with those stories
            and their own frames taken out (_drop_story_in), the later indices one lower each
        added: TextIndex of each added layer in the new card (Edit.added_stories): the new card is compared with those
            stories and their own frames taken out
        filters: the FEid block is gate_filters' (a bake's: smart-filtered stars moved or copied, S1)
        links: uuids of the links the edits added (made type icons): lnkE is the old items and those, appended
        also: Other document blocks (keys) that may change: the link step's (card_standards.normalise)
        frames: TextIndex of each edited layer whose box was resized (set_text_box): its own Txt2 frames may change
        relinked: uuids of the links the edits rewrote (Edit.relinked): lnkE the old items, those changed only in
            their file's size and contentID

    Returns:
        The changed stories' indices (the new card's), sorted; raises Refused("bytes")
    """
    before, after = _document_blocks(old), _document_blocks(new)
    txt2, lnke = bytes(Tag.TEXT_ENGINE_DATA), bytes(Tag.LINKED_LAYER_EXTERNAL)
    moved = sorted(key.decode("latin-1") for key in set(before) | set(after)
                   if (key != txt2 or (key in before) != (key in after)) and before.get(key) != after.get(key)
                   and not (filters and key in FILTER_KEYS) and key not in also
                   and not (key == lnke and (_links_added(old, new, links) or _links_relinked(old, new, relinked))))
    if moved:
        raise Refused("bytes", f"document blocks changed: {', '.join(moved)}")
    stale = stale_stories(new)
    if stale:
        name, story, own = stale[0]
        raise Refused("bytes", f"{name}'s Txt2 story holds {collapse(story)!r}, the layer {collapse(own)!r}")
    if txt2 not in before:
        return []
    gone, came = sorted(set(removed)), sorted(set(added))

    def renumber(i: int) -> int:
        return i - sum(1 for r in gone if r < i)
    indices = [sorted(story_index(layer) for layer in psd.descendants() if _is_text_layer(layer)
                      and story_index(layer) is not None) for psd in (old, new)]
    if came and not set(came) <= set(indices[1]):
        raise Refused("bytes", f"an added layer's story {came} is no type layer's")
    indices[1] = [i - sum(1 for c in came if c < i) for i in indices[1] if i not in came]
    if [renumber(i) for i in indices[0] if i not in gone] != indices[1]:   # a layer without one is orphaned from the
        raise Refused("bytes", f"type layers' TextIndex {indices[0]} -> {indices[1]}")   # story Photoshop would read
    root_a = _txt2(old)
    if gone:                                    # the old card with the deleted layers' stories taken out
        root_a = deepcopy(root_a)
        keys, items = _story_keys(root_a)
        for index in reversed(gone):
            _drop_story_in(root_a, keys, items, index)
    root_b = _txt2(new)
    if came:                                    # the new card with the added layers' stories taken out
        root_b = deepcopy(root_b)
        keys, items = _story_keys(root_b)
        for index in reversed(came):
            _drop_story_in(root_b, keys, items, index)
    edited = {renumber(i) for i in edited if i is not None and i not in gone}
    (objects_a, _, items_a, _, _), (objects_b, _, items_b, _, _) = _story_list(root_a), _story_list(root_b)
    if len(items_a) != len(items_b):
        raise Refused("bytes", f"Txt2 stories {len(items_a)} -> {len(items_b)}")
    changed = [i for i, (a, b) in enumerate(zip(items_a, items_b)) if _raw(a) != _raw(b)]
    beyond = sorted(set(changed) - set(edited))
    if beyond:
        raise Refused("bytes", f"Txt2 stories {beyond} changed: no edit there")
    faces_a, faces_b = _story_fonts(root_a), _story_fonts(root_b)
    if faces_a is not None and faces_b is not None and len(faces_b) > len(faces_a):
        added = [_story_font_name(entry) for entry in list(faces_b)[len(faces_a):]]
        if sorted(added) != sorted(set(added)) or not set(added) <= set(fonts):
            raise Refused("bytes", f"Txt2 FontSet gained {', '.join(added)}: no edit added them")
        root_b = deepcopy(root_b)               # the rest compared without them
        objects_b, _, items_b, _, _ = _story_list(root_b)
        trimmed = _story_fonts(root_b)
        while len(trimmed) > len(faces_a):
            trimmed.pop()
    frames = [renumber(i) for i in frames if i not in gone]
    if frames:                                  # the resized boxes' own frames left out of both
        root_a, root_b = _frames_out(root_a, frames), _frames_out(root_b, frames)
        objects_a, objects_b = _story_list(root_a)[0], _story_list(root_b)[0]
        items_a, items_b = _story_list(root_a)[2], _story_list(root_b)[2]
    rest = [[(str(key), _raw(value)) for key, value in root.items() if value is not objects]
            + [(str(key), _raw(value)) for key, value in objects.items() if value is not items]
            for root, objects, items in ((root_a, objects_a, items_a), (root_b, objects_b, items_b))]
    if rest[0] != rest[1]:
        raise Refused("bytes", "Txt2 changed outside its stories")
    drift = [d for d in story_style_drift(new) if d not in story_style_drift(old)]    # a layer no edit set keeps its own
    if drift:
        raise Refused("bytes", f"{drift[0][0]}'s Txt2 story has other styles: {drift[0][1]}")
    return changed


def gate_reparse(parser: PSDParser, parsed: dict, new_path: str, rel: str, jobs: dict, psd: PSDImage, edits: list,
                 notes: list) -> dict:
    """
    Re-parse the new PSD: the old parse with exactly the edited fields replaced (the stars read back as the edit, a
    folder card's as drawn too; the types as its icons in any order: a miss names the field), no new problems (no
    EFFECT LAYER problem: the validator's coverage check is the FX layers' gate), one digit per edited row, the title
    read back for a name edit, and an ability edit's FX layers back as declared: its FX layers' names in order, the ones
    it made visible, styled, directly above the ability's top-level group (_top) (gate_fx measures what they cover).

    Args:
        parser: offline_parser()
        parsed: parse_card() of the old PSD
        new_path: The saved PSD
        rel: The card's repo path
        jobs: The applied jobs
        psd: The saved PSD, reopened
        edits: The applied Edits (their FX layers are checked)
        notes: Where new validator problems go (the edit's content: baked, told; owner 2026-10-24)

    Returns:
        parse_card() of the new PSD
    """
    for stat in jobs:
        if stat in STAT_FIELDS and not any(edit.field == stat and edit.bar for edit in edits):     # set_bar: its
            try:                                # layers' changes were declared (gate_bytes); read back below
                rows = template_rows(psd, stat)
            except Refused:                     # the card maker's own bar: the parse reads it back below
                continue
            for row, target in zip(rows, _targets(rows, jobs[stat])):
                if row.visible != _row_shows(row, target):
                    raise Refused("reparse", f"{stat}: row {row.group.name!r} shows {row.visible}")
    for whole in edits:
        for edit, layer, _ in _set_layers(psd, whole) if whole.field == "ability" else []:
            if not (edit.fx or edit.dropped or edit.fx_names):
                continue
            group = _top(layer)
            found = fx_layers(psd, layer)
            if [fx.name for fx in found] != edit.fx_names:
                raise Refused("reparse", f"FX layers after reopen: {[fx.name for fx in found]}")
            made, above = {name for name, _ in edit.fx}, psd.index(group) + 1 + len(_clip_chain(group))
            for k, fx in enumerate(fx for fx in found if fx.name in made):
                if (fx.parent is not psd or psd.index(fx) != above + k or not fx.is_visible()
                        or not _enabled_effects(fx)):
                    raise Refused("reparse", f"{fx.name!r} is not a visible styled layer directly above {group.name!r} "
                                             "and its clip chain")
    new = parse_card(parser, new_path, rel, psd)     # the reopened PSD: no second open
    expected, got = dict(parsed["card"]), dict(new["card"])
    for name, value in jobs.items():
        if name in STAT_FIELDS or name == "stars":
            expected[name] = value
        elif name == "types":                   # in any order: the parser lists them in layer order
            expected[name] = got.get(name) if same_types(parser, got.get(name), value) else value
            if expected.get("type") == "aux item":     # its icon is its item type (72b4876): the icons left's
                kinds = [AUX_ITEM_TYPES[t] for t in (canonical_type(parser, t) for t in value) if t in AUX_ITEM_TYPES]
                expected["subtype"] = got.get("subtype") if got.get("subtype") in kinds else (kinds or [None])[0]
        elif name == "ability":                 # a typed token as the parser reads it in the icon font ([Fire] -> [fire]);
            expected["ability"], got["ability"] = (normalize_ability(canonical_tokens(text, parser.icons, chars))
                                                   for text, chars in ((value, True), (got.get("ability"), False)))
            #   a "★" the parser reads as typed is not in the icon font: no [star]
        elif name == "name":                    # the new title is read back below
            expected.pop("display_name", None)
            got.pop("display_name", None)
        elif name == "parts":                   # read back per part (gate_parts), not as the joined fields
            for key in {"display_name" if part["field"] == "name" else "ability" for part in value.values()}:
                expected.pop(key, None)
                got.pop(key, None)
    expected.pop("problem", None)
    got.pop("problem", None)
    stats = {name: value for name, value in jobs.items() if name in STAT_FIELDS}
    listed = jobs.get("creatures") or []
    if (stats or listed) and expected.get("creatures") and got.get("creatures") \
            and len(got["creatures"]) == len(expected["creatures"]):
        expected["creatures"] = _creatures_after(expected["creatures"], got["creatures"], stats, listed)
        expected.update(listed[0] if listed else {})
    if "stars" in jobs and star_folder(expected.get("path") or "") and got.get("path") == rel:
        expected["path"] = rel                  # moved into its new "N Stars" folder
    if "stars" in jobs and star_folder(rel):    # the folder reads back as the edit: count the redrawn stars
        types, stars = parser.all_types, parser.star_names()
        counted = any(PSDParser.stars_in(group, types) for group in PSDParser._star_groups(psd))
        drawn = PSDParser.stars_drawn(psd, types if counted else (), stars)
        if drawn != jobs["stars"]:
            raise Refused("reparse", f"stars: the card draws {drawn}, not {jobs['stars']}", "stars")
    different = sorted(key for key in set(expected) | set(got) if expected.get(key) != got.get(key))
    if different:
        raise Refused("reparse", "fields differ from the expected parse: " + ", ".join(
            f"{key} {got.get(key)!r} (not {expected.get(key)!r})" if key in ICON_FIELDS else key for key in different),
            different[0] if set(different) <= set(ICON_FIELDS) else None)
    answered = CardEdits.answered(jobs)
    old_problems = {p for p in parsed["validator"] if not (answered and p.startswith(answered))}
    added = sorted(set(new["validator"]) - old_problems)
    if added:
        notes.append("validator: new problems: " + "; ".join(added))
    if any(p.startswith("EFFECT LAYER") for p in new["validator"]):
        raise Refused("reparse", "effect layer problems remain")
    title = read_name(psd, _card_name(parsed), getattr(parser, "linked", None), rel) if "name" in jobs else None
    if "name" in jobs and straight(title) != straight(collapse(jobs["name"])):
        raise Refused("reparse", f"the title reads {title!r}")
    return new


def _creatures_after(before: list, after: list, stats: dict, listed=()) -> list:
    """
    The creatures a stat edit should leave (gate_reparse): creature 1 takes the edited values (its rows: bar_rows); another
    creature keeps its stats, but an edited one it shared with creature 1 (Pix's one HP mask, Twin Emperors' shared
    health) may follow it; listed: a `creatures` job's values. Returns before with those changes, so any other difference
    still fails the read-back.
    """
    out = [dict(creature) for creature in before]
    for stat, value in stats.items():
        out[0][stat] = value
        for k in range(1, len(out)):
            if before[k].get(stat) == before[0].get(stat) and after[k].get(stat) == value:
                out[k][stat] = value
    for creature, changes in zip(out, listed):
        creature.update(changes)
    return out


def gate_parts(parser: PSDParser, parsed: dict, rel: str, jobs: dict, old: PSDImage, new: PSDImage,
               fonts_dir: str) -> list:
    """
    §4.10's read-back per part, on the saved card: each job's layer reads its box's text, a deleted one is gone, every
    other part's layer reads as before (the joined name and ability are not compared: gate_reparse); a title part the
    edit set takes the title's icon checks (_gate_icons).

    Returns:
        Notes: a layer the parser no longer reads as a part, or a new one (the card's title or ability changed shape);
        raises Refused("reparse") for a layer that doesn't read as declared
    """
    name, vocab = _card_name(parsed), getattr(parser, "icons", None)
    before = record_parts(card_parts(old, rel, name, parser))
    after = record_parts(card_parts(new, rel, name, parser))
    index, was, notes = lyid_index(new), lyid_index(old), []
    for key, (field_name, _, part) in before.items():
        layer, job = index.get(int(key)), jobs.get(key)
        if job and job.get("delete"):
            if layer is not None:
                raise Refused("reparse", f"{part['layer']!r} was to be deleted and is still in the card")
            continue
        if layer is None:
            raise Refused("reparse", f"{part['layer']!r} is gone: no edit deleted it")
        reads, want = part_text(parser, layer), job["text"] if job else part["text"]
        if not _part_same(reads, want, vocab):
            raise Refused("reparse", f"{part['layer']!r} reads {reads[:60]!r}, not {want[:60]!r}")
        if job and field_name == "name":
            notes += _gate_icons(new, layer, old, was.get(layer.layer_id), font_map(fonts_dir), vocab, "title")
    deleted = {key for key, job in jobs.items() if job.get("delete")}
    lost = [part["layer"] for key, (_, _, part) in before.items() if key not in after and key not in deleted]
    came = [(key, part) for key, (_, _, part) in after.items() if key not in before]
    if ADD_ABILITY in jobs:                     # the added Ability reads its box's text (§4.10)
        new = [(key, part) for key, part in came if after[key][0] == "ability" and base_name(index[int(key)]) == "Ability"]
        if len(new) != 1:
            raise Refused("reparse", f"{len(new)} added Ability layers read as the ability")
        reads, want = part_text(parser, index[int(new[0][0])]), jobs[ADD_ABILITY]["text"]
        if not _part_same(reads, want, vocab):
            raise Refused("reparse", f"the added Ability reads {reads[:60]!r}, not {want[:60]!r}")
        came.remove(new[0])
    came = [part["layer"] for _, part in came]
    if lost:
        notes.append(f"parts: {', '.join(map(repr, lost))} no longer read as a title or ability part: check the card")
    if came:
        notes.append(f"parts: {', '.join(map(repr, came))} now read as a title or ability part: check the card")
    return notes


def _set_layers(psd: PSDImage, edit: Edit) -> list:
    """(Edit, its layer in psd, a copy?) per text layer an ability edit set: the Ability, or each of several (parts)."""
    if not edit.several:
        return [(edit, ability_layer(psd), False)]
    ids = {layer.layer_id: layer for layer in psd.descendants()}
    return [(part, ids[layer_id], copy) for layer_id, part, copy in edit.parts]


def _on_canvas(layer, shape: tuple, box: Optional[tuple] = None) -> np.ndarray:
    """The layer's stored RGBA (8-bit values) on a canvas of shape (H, W), or on its box (x0, y0, x1, y1) alone."""
    ox, oy, wx, wy = box or (0, 0, shape[1], shape[0])
    out = np.zeros((wy - oy, wx - ox, 4), np.uint8)
    x0, y0, x1, y1 = clip_to(tuple(layer.bbox), (ox, oy, wx, wy)) or (0, 0, 0, 0)
    if x1 > x0 and y1 > y0:
        lx, ly = layer.bbox[:2]
        rgba = np.concatenate([layer.numpy("color"), layer.numpy("shape")], axis=2)[y0 - ly:y1 - ly, x0 - lx:x1 - lx]
        out[y0 - oy:y1 - oy, x0 - ox:x1 - ox] = np.rint(rgba * 255)
    return out


def _boxes(glyphs: list, pad: int, shape: tuple) -> np.ndarray:
    """The union of the glyphs' ink boxes (text_geometry, canvas px) padded by pad, as a bool (H, W) canvas."""
    out = np.zeros(shape, bool)
    for glyph in glyphs:
        x0, y0, x1, y1 = glyph["ink"]
        out[max(int(y0) - pad, 0):max(int(-(-y1 // 1)) + pad, 0), max(int(x0) - pad, 0):max(int(-(-x1 // 1)) + pad, 0)] = True
    return out


def _colour_ink(layer, fonts, shape: tuple) -> np.ndarray:
    """
    Canvas alpha (0..1) of the pixels the text layer draws in its icon font's own colours, 0 elsewhere: where the
    engine's renders of its text with every run filled red and filled blue agree (COLOUR_TOL), at any alpha; a pixel a
    letter reaches changes. Independent of the badge cut's keys and glyph render (_phrase_mask).
    """
    TextModel, layout, rasterize, Options, GlyphCache = _fx_engine()[:5]
    model = TextModel.from_layer(layer)
    renders = []
    for fill in ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)):
        key = replace(model, runs=tuple(replace(run, style=replace(run.style, fill=fill)) for run in model.runs))
        rgba, bbox = rasterize(layout(key, fonts), key.transform, key.antialias, Options(color_glyphs=True), GlyphCache(),
                               max_alloc_bytes=layer._psd._max_alloc_bytes)
        out = np.zeros((*shape, 4), np.float32)
        x0, y0, x1, y1 = (int(v) for v in bbox)
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, shape[1]), min(y1, shape[0])
        if cx1 > cx0 and cy1 > cy0:
            out[cy0:cy1, cx0:cx1] = rgba[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
        renders.append(out)
    red, blue = renders
    same = np.abs(red - blue).max(axis=2) <= COLOUR_TOL
    return np.where(same, red[..., 3], 0.0)


def _nearest(glyphs: list, mask: np.ndarray, origin: tuple = (0, 0)) -> tuple:
    """(ys, xs, k): the mask's pixels on the canvas (its top left at origin) and, per pixel, the index into glyphs of the
    one whose ink box is nearest."""
    ys, xs = mask.nonzero()
    ys, xs = ys + origin[1], xs + origin[0]
    if not glyphs or not len(xs):
        return ys, xs, np.zeros(len(xs), int)
    box = np.array([glyph["ink"] for glyph in glyphs], float)
    cx, cy = xs[:, None] + 0.5, ys[:, None] + 0.5
    dx = np.maximum(np.maximum(box[None, :, 0] - cx, cx - box[None, :, 2]), 0)
    dy = np.maximum(np.maximum(box[None, :, 1] - cy, cy - box[None, :, 3]), 0)
    return ys, xs, np.maximum(dx, dy).argmin(axis=1)


def _fx_status(base, fx, chart: dict) -> str:
    """The styled chart status an FX layer's phrases all name (status_fills' key: "frozen" for "Frozen#2"), else ""."""
    names = {collapse(_numbered(phrase)[0]).lower() for phrase in fx_phrases(base, fx)}
    name = next(iter(names)) if len(names) == 1 else ""
    return name if getattr(chart.get(name), "styled", False) else ""


def gate_status_words(psd: PSDImage, chart: dict, layer=None) -> None:
    """
    Each styled status word of the Ability (status_spans) is held whole by one of its FX layers (phrase_ranges of its
    name): its status's layer on exactly that word, or another phrase around it (Clot Gold's "1 Become Back Money"); and
    a status's layer lies on its status's own words only, not the "Frozen" of "Frozen In Time" or of "Unfrozen".
    Independent of follow_fx and new_fx. Returns a note per capitalised styled status word no FX layer holds (it prints
    in the chart's fill alone: new_fx told why; a lowercase one is prose); raises Refused("reparse") naming the layer.
    layer: one of several (None: the Ability).
    """
    layer = ability_layer(psd) if layer is None else layer
    text, found = layer.text, fx_layers(psd, layer)
    spans = status_spans(text, chart, icon_spans(layer))
    words = {(a, b, status.name.lower()) for a, b, status in spans if status.styled}
    held = [(_fx_status(layer, fx, chart), phrase_ranges(text, fx_phrases(layer, fx)), fx) for fx in found]
    for status, ranges, fx in held:
        for a, b in ranges if status else []:
            if (a, b, status) not in words:
                raise Refused("reparse", f"{fx.name!r} covers {_word_at(text, (a, b), spans)!r}, no "
                                         f"{chart[status].name} of its own")
    return [f"styled-word: ability: {text[a:b]!r} has no FX layer of its own" for a, b, status in sorted(words)
            if text[a].isupper() and not any((a, b) in ranges if mine == status else
                                             not mine and any(c <= a and b <= d for c, d in ranges)
                                             for mine, ranges, _ in held)]


def gate_fx_colours(psd: PSDImage, old: PSDImage, chart: dict, layer=None, was=None) -> None:
    """
    Each "styled" or "own" FX layer of the old Ability (fx_kind) still in the card covers letters of only the colours
    its words had before the edit, status words apart (the chart's swatch): its style draws over its words' own fill (a
    stroke alone shows it). Matched by layer id. Independent of set_text_layer's fills (review 2026-10-23: a "+2 DEF"
    typed on Palace of Shadow joined its blue "+2 DEF" layer in the body's grey, every other gate passing). Raises
    Refused("reparse") naming the layer and the word. layer, was: one of several, in psd and old (None: the Ability).
    """
    def letters(base):          # an FX layer's letters under base's text: [(its phrase's range, the letter's fill)]
        text, fills = base.text, _char_styles(base)[0]
        words = {i for a, b, _ in status_spans(text, chart, icon_spans(base)) for i in range(a, b)}
        return lambda fx: [((a, b), tuple(round(v, 3) for v in fills[i]))
                           for a, b in phrase_ranges(text, fx_phrases(base, fx)) for i in range(a, b)
                           if fills[i] is not None and i not in words and text[i] not in " \t\r\x03"]

    def key(fx):
        return fx.layer_id if fx.layer_id != -1 else fx.name

    was, layer = ability_layer(old) if was is None else was, ability_layer(psd) if layer is None else layer
    before, after = letters(was), letters(layer)
    colours = {key(fx): {fill for _, fill in before(fx)} for fx in fx_layers(old, was) if fx_kind(fx, was, chart) in ANCHORED}
    for fx in fx_layers(psd, layer):
        odd = [span for span, fill in after(fx) if fill not in colours.get(key(fx), {fill})]
        if odd:
            raise Refused("reparse", f"{fx.name!r} covers {layer.text[odd[0][0]:odd[0][1]]!r} in a colour its words did "
                                     f"not have")


def bare_icons(psd: PSDImage, fonts, layer=None) -> Counter:
    """
    The icon-font glyphs of a tinted Ability (_tinted: an overlay at any opacity, a blend other than Normal, opacity
    below full) whose colours its look changes, by what they draw, lowercased ("[fire]"): each with ink of its own (its
    text_geometry box, other glyphs' boxes left out: an [undead] skull's glowing eyes) or the nearest icon glyph to a
    pixel past FAINT_INK it draws in its own colours (_colour_ink: [support]'s see-through glow, past its box; reviews
    2026-10-24). A badge or :sticker in Stylistic Alternates under an overlay (one_colour) is no such glyph: drawn in
    one colour on purpose (owner 2026-10-25). Empty for an Ability that is not tinted.

    Args:
        psd: The card
        fonts: A FontMap of MDW/Fonts
        layer: Its text layer (None: the Ability; a title edit's title, owner 2026-10-24)
    """
    from psd_tools.composite.text.geometry import text_geometry
    layer = ability_layer(psd) if layer is None else layer
    if not _tinted(layer):
        return Counter()
    geometry = text_geometry(layer, fonts)
    glyphs = [glyph for glyph in geometry["glyphs"] if glyph["ink"]]
    icon = [glyph for glyph in glyphs if glyph["font"].lower().startswith(ICON_FONT_PREFIX)]
    if not icon:
        return Counter()
    text = geometry["text"]
    salted = (stylistic_alternates(layer) or ()) if one_colour(layer) else ()
    shape = (psd.height, psd.width)
    alpha = _on_canvas(layer, shape)[..., 3]
    ink = alpha > INK_ALPHA * 255
    others = ~_boxes([glyph for glyph in glyphs if not glyph["font"].lower().startswith(ICON_FONT_PREFIX)], 1, shape)
    faint = (_colour_ink(layer, fonts, shape) > 0) & (alpha > FAINT_INK * 255)
    bare = set(_nearest(icon, faint)[2].tolist())
    out = Counter()
    for k, glyph in enumerate(icon):
        a, b = glyph["i"], glyph["end"]
        if alternate_spans(text, [(a, b)]) and any(c <= a and b <= d for c, d in salted):
            continue
        if k in bare or (ink & _boxes([glyph], 0, shape) & others).any():
            out[text[a:b].strip().lower() or glyph["ch"]] += 1
    return out


def lettered_icons(psd: PSDImage, fonts, layer=None) -> Counter:
    """
    What the Ability's icon font draws as letters, by character: an icon-font glyph the shaper made of one printable
    ASCII character, no ligature (the font's GSUB makes each badge and token it knows one cluster, "★" is no ASCII;
    "[fire:]" stays seven letters). Independent of icon_text, whose rule let "[fire:]" in while the re-parse read it
    back as [fire] (review 2026-10-24). layer: the title's for a title edit (None: the Ability).
    """
    from psd_tools.composite.text.geometry import text_geometry
    layer = ability_layer(psd) if layer is None else layer
    if not uses_icon_font(layer):
        return Counter()
    return Counter(glyph["ch"] for glyph in text_geometry(layer, fonts)["glyphs"]
                   if glyph["font"].lower().startswith(ICON_FONT_PREFIX) and glyph["end"] - glyph["i"] == 1
                   and "!" <= glyph["ch"] <= "~")


def unshaped_icons(psd: PSDImage, fonts, vocab: Optional[IconVocabulary], layer=None) -> Counter:
    """
    The Ability's icon_text spans (what the icon font should draw as one badge or icon) the layout does not shape as
    one icon-font cluster spanning all of it, by text: "[fire:]" if icon_text took it (seven letters), a token in the
    body font. The layout's own check of icon_text's rule (review 2026-10-24: gate_reparse shared it with the parser).
    A layer without the icon font counts every span: the plain cards' tags printed as body text (Papyrus' (1CD)) cancel
    out against the old card's (review 2026-10-24: the new side alone counted them once a typed badge added the font).
    layer: the title's for a title edit (None: the Ability).
    """
    from psd_tools.composite.text.geometry import text_geometry
    layer = ability_layer(psd) if layer is None else layer
    if vocab is None or not icon_text(layer.text, vocab):
        return Counter()
    geometry = text_geometry(layer, fonts)
    out = Counter()
    for a, b in icon_text(geometry["text"], vocab):
        mine = [glyph for glyph in geometry["glyphs"] if glyph["i"] < b and a < glyph["end"]]
        if not mine or any((glyph["i"], glyph["end"]) != (a, b) or not glyph["font"].lower().startswith(ICON_FONT_PREFIX)
                           for glyph in mine):
            out[geometry["text"][a:b]] += 1
    return out


def gate_fx(psd: PSDImage, edits: list, fonts_dir: str, chart: Optional[dict] = None,
            old: Optional[PSDImage] = None, vocab: Optional[IconVocabulary] = None, titles: Optional[tuple] = None) -> None:
    """
    An ability edit's FX layers cover exactly their words in the new text (BUG 2), measured on the saved card with the
    fork's text geometry (text_geometry: each glyph's ink box in the new layout): each FX layer of the Ability inks only
    (any alpha) inside the boxes of its phrases' glyphs (phrase_ranges over the new text) padded by FX_BOX_PAD; it
    covers FX_COVER of the Ability's ink in its words' boxes (the icon font's glyphs left out) that no other glyph's
    box reaches; one the edit cut (rebuilt
    or made) holds the Ability's own pixels byte for byte; and a "styled" layer's name still reaches the words of it the
    edit kept (Edit.held: Goofy's "1#1" on the "1" of "3T1"); with the chart, every styled status word has its FX layer
    (gate_status_words) and, given the old card, a "styled" or "own" layer covers no letter in a colour its words lacked
    (gate_fx_colours); under a tinted Ability no more icon-font glyphs in its look's colours than before
    (bare_icons: Blood Bomb's [machine] was so already; review 2026-10-23); the icon font draws no more letters
    than before (lettered_icons: each typed badge or token one ligature); and, given the vocabulary, each badge or
    token icon_text reads is one icon-font cluster that was not one before (unshaped_icons).

    Args:
        psd: The saved PSD, reopened
        edits: The applied Edits
        fonts_dir: MDW/Fonts (the layout's fonts)
        chart: status_fills() (None: the status words are not checked)
        old: The card before the edit (its bare icons may stay bare); None: none may be bare
        vocab: The parser's IconVocabulary (None: unshaped_icons is not checked)
        titles: (new title layer, old title layer or None) of a title edit: its icons take the same icon checks (owner
            2026-10-24: a typed token is set in the icon font there too)

    Returns the notes of what the edit's content leaves as it is (owner 2026-10-24: baked, told): the icon checks, a
    status word without an FX layer; raises Refused("reparse") naming the layer for the rest.
    """
    fields = {e.field for e in edits}
    if "ability" not in fields and not ("name" in fields and titles):
        return []
    fonts, notes = font_map(fonts_dir), []
    if "name" in fields and titles:
        notes += _gate_icons(psd, titles[0], old, titles[1], fonts, vocab, "title")
    if "ability" not in fields:
        return notes
    several = next((e for e in edits if e.field == "ability" and e.several), None)
    if several is not None:                     # each layer it set, as the Ability's (a copy: its icons only)
        ids = {layer.layer_id: layer for layer in old.descendants()} if old is not None else {}
        for edit, layer, copy in _set_layers(psd, several):
            was = ids.get(layer.layer_id)
            if chart and not copy:
                notes += gate_status_words(psd, chart, layer)
                if was is not None:
                    gate_fx_colours(psd, old, chart, layer, was)
            if edit.fx_names:
                _gate_fx_layers(psd, edit, fonts, layer)
            notes += _gate_icons(psd, layer, old, was, fonts, vocab, "Ability")
        return notes
    if chart:
        notes += gate_status_words(psd, chart)
        if old is not None:
            gate_fx_colours(psd, old, chart)
    edit = next((e for e in edits if e.field == "ability" and e.fx_names), None)
    if edit is not None:
        _gate_fx_layers(psd, edit, fonts)
    return notes + _gate_icons(psd, None, old, None, fonts, vocab, "Ability")


def _gate_icons(psd: PSDImage, layer, old: Optional[PSDImage], old_layer, fonts, vocab: Optional[IconVocabulary],
                what: str) -> list:
    """gate_fx's icon checks of one text layer (None: the Ability) against the old card's (None: none may fail), as
    notes ("unsupported-card: ability: [fire]: not drawn as one badge or icon")."""
    def was(check, *args) -> Counter:
        if old is None or (layer is not None and old_layer is None):
            return Counter()
        return check(old, fonts, *args, layer=old_layer)
    out, field = [], "name" if what == "title" else "ability"
    bare = bare_icons(psd, fonts, layer=layer) - was(bare_icons) if what == "title" else None
    if bare:
        out.append(f"{', '.join(sorted(bare))}: under the {what}'s look "
                   f"({_tinted(ability_layer(psd) if layer is None else layer)}), which changes their colours")
    lettered = lettered_icons(psd, fonts, layer=layer) - was(lettered_icons)
    if lettered:
        out.append(f"{''.join(sorted(lettered.elements()))!r}: the icon font draws them as letters, no badge or icon")
    unshaped = unshaped_icons(psd, fonts, vocab, layer=layer) - was(unshaped_icons, vocab)
    if unshaped:
        out.append(f"{', '.join(sorted(unshaped))}: not drawn as one badge or icon")
    return [f"unsupported-card: {field}: {why}" for why in out]


def _gate_fx_layers(psd: PSDImage, edit: Edit, fonts, layer=None) -> None:
    """gate_fx's checks of each FX layer of the edited Ability (its docstring; layer: one of several)."""
    from psd_tools.composite.text.geometry import text_geometry
    layer = ability_layer(psd) if layer is None else layer
    geometry = text_geometry(layer, fonts)
    shape = (psd.height, psd.width)
    base = _on_canvas(layer, shape)
    ink = base[..., 3] > INK_ALPHA * 255
    cut = {name for name, _ in edit.fx} | {name for _, _, name in edit.rebuilt}
    for name, spans in edit.held:
        fx = next((fx for fx in fx_layers(psd, layer) if fx.name == name), None)
        ranges = phrase_ranges(layer.text, fx_phrases(layer, fx)) if fx else []
        lost = [layer.text[a:b] for a, b in spans if not any(c <= a and b <= d for c, d in ranges)]
        if lost:
            raise Refused("reparse", f"{name!r} left its words {', '.join(map(repr, lost))}")
    for fx in fx_layers(psd, layer):
        mine, got, want = _fx_cover(layer, fx, geometry, ink)
        ours = _on_canvas(fx, shape)
        stray = int(((ours[..., 3] > 0) & ~_boxes(mine, FX_BOX_PAD, shape)).sum())   # faint ones too
        if stray:
            raise Refused("reparse", f"{fx.name!r} inks {stray} px outside its words' glyph boxes")
        if got < FX_COVER * want:
            raise Refused("reparse", f"{fx.name!r} covers {got} of its words' {want} px")
        if fx.name in cut:
            held = ours[..., 3] > 0
            if not np.array_equal(ours[held], base[held]):
                raise Refused("reparse", f"{fx.name!r} is not the Ability's own pixels")


def _fx_cover(layer, fx, geometry: dict, ink: np.ndarray) -> tuple:
    """
    (its phrases' glyphs, px of their own ink an FX layer covers past INK_ALPHA, those px): gate_fx's FX_COVER measure.
    Their own ink: ink (the text layer's past INK_ALPHA, on the canvas) in the ink boxes of its phrases' glyphs
    (text_geometry; the icon font's left out) that no other glyph's box, grown a pixel, reaches.
    """
    glyphs = [glyph for glyph in geometry["glyphs"] if glyph["ink"]]
    ranges = phrase_ranges(geometry["text"], fx_phrases(layer, fx))
    mine = [k for k, glyph in enumerate(glyphs) if any(a <= glyph["i"] < b for a, b in ranges)]
    need = [glyphs[k] for k in mine if not glyphs[k]["font"].lower().startswith(ICON_FONT_PREFIX)]
    others = [glyph for k, glyph in enumerate(glyphs) if k not in mine]
    want = ink & _boxes(need, 0, ink.shape) & ~_boxes(others, 1, ink.shape)
    drawn = _on_canvas(fx, ink.shape)[..., 3] > INK_ALPHA * 255
    return [glyphs[k] for k in mine], int((drawn & want).sum()), int(want.sum())


def gate_idempotence(new_parsed: dict, jobs: dict, psd: PSDImage, merged: Image.Image, edits: list,
                     icc: bool = True, texts: Optional[dict] = None, parser: Optional[PSDParser] = None,
                     chart: Optional[dict] = None, kept: Optional[dict] = None) -> dict:
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
        kept: gate_dirty's keep: psd drawn once over its box, as it was (its pass resumed), each "preview" region cut

    Returns:
        {field: region measurement}
    """
    again = classify(new_parsed, jobs, psd, parser, chart=chart)
    left = sorted(set(again.jobs) | {r.field for r in again.refusals if r.field})
    if left:
        raise Refused("idempotence", "still to bake: " + ", ".join(left), left[0] if set(left) <= set(PARTIAL_FIELDS) else None)
    out, drawn = {}, None
    edited = edited_ids(edits)
    box = kept.get("box") if kept and kept.get("edited") == edited else None
    for edit in edits:
        text = (texts or {}).get(edit.field, "preview")
        if box and text == "preview" and drawn is None:
            drawn = np.asarray(psd.composite(viewport=box, ignore_preview=True, apply_icc=False,
                                             **text_kwargs(psd, text, edited)).convert("RGBA"))
        cut = crop_pair(psd, drawn, edit.region, icc, box) if text == "preview" else None
        measured = _measure(cut[0] if cut else render_region(psd, edit.region, icc, text, edited), merged.crop(edit.region))
        if _past_floor(measured):
            raise Refused("idempotence", "%s re-render bad %.2f%% mean %.2f: past the %g%% floor"
                          % (edit.field, measured["bad_pct"], measured["mean"], PARITY_FLOOR),
                          edit.field if edit.field in PARTIAL_FIELDS else None)
        out[edit.field] = measured
    return out


# ========================
# Save
# ========================
def strip(psd: PSDImage) -> None:
    """The merged image dropped in place as Photoshop 2023 writes it with Maximize Compatibility off (the fork's
    strip_merged, proofs P3)."""
    if psd.has_preview():
        psd.strip_merged()


def _image_section(path: str) -> tuple:
    """(start, end, merged) of a PSD file's image data section, read off the sections' lengths and VersionInfo (nothing
    decoded); merged: it holds a real merged image."""
    with open(path, "rb") as fh:
        head = fh.read(26)
        if len(head) < 26 or head[:4] != b"8BPS":
            return 0, 0, False
        psb = unpack(">H", head[4:6])[0] == 2
        fh.seek(unpack(">I", fh.read(4))[0], 1)             # colour mode data
        resources = fh.read(unpack(">I", fh.read(4))[0])
        size = unpack(">Q" if psb else ">I", fh.read(8 if psb else 4))[0]
        start = fh.seek(size, 1)
        end = fh.seek(0, 2)
    i, merged = 0, True
    while i + 12 <= len(resources):
        rid = unpack(">H", resources[i + 4:i + 6])[0]
        name = resources[i + 6]
        i += 6 + name + 1 + ((name + 1) & 1)
        length = unpack(">I", resources[i:i + 4])[0]
        if rid == Resource.VERSION_INFO.value and length >= 5:
            merged = bool(resources[i + 8])
        i += 4 + length + (length & 1)
    return start, max(end, start), merged


def changed_part(old: PSDImage, new: PSDImage, parts: tuple = ("header", "color_mode_data", "image_resources",
                                                                "image_data")) -> Optional[str]:
    """
    The first of the document's parts outside its layers that new changed, or None. A stripped merged image (strip()) is
    no change: new's image data is not compared, and its resources are compared with old's hasRealMergedData off.
    """
    stripped = old.has_preview() and not new.has_preview()
    for part in parts:
        if part == "image_data" and not new.has_preview():
            continue
        was = getattr(old._record, part)
        if part == "image_resources" and stripped and was.get_data(Resource.VERSION_INFO) is not None:
            was = deepcopy(was)
            was.get_data(Resource.VERSION_INFO).has_composite = False
        if _raw(was) != _raw(getattr(new._record, part)):
            return part
    return None


def _fx_version(data) -> int:
    """An effects block's version, parsed or kept as raw bytes."""
    if isinstance(data, (bytes, bytearray)):
        return unpack(">I", data[:4])[0] if len(data) >= 4 else 0
    return getattr(data, "version", 0)


def unreadable_effects(psd: PSDImage) -> list:
    """Layers whose effects block (lfx2, lmfx, lfxs) is not at Photoshop's version 0: Photoshop 2023 cannot read their
    effects ("Problems were encountered reading layers"; psd-tools' set_data wrote 1, 2026-11-08)."""
    return [layer for layer in psd.descendants()
            if any(_fx_version(layer.tagged_blocks[key].data) for key in EFFECT_BLOCKS[:3]
                   if key in layer.tagged_blocks)]


def save_psd(psd: PSDImage, path) -> None:
    """Write the PSD with no merged image (Maximize Compatibility off, owner 2026-11-07): strip(), then saved without a
    regenerated preview. Effects blocks are written at Photoshop's version (unreadable_effects; layer_records ignores
    the repair)."""
    for layer in unreadable_effects(psd):
        for key in EFFECT_BLOCKS[:3]:
            block = layer.tagged_blocks[key] if key in layer.tagged_blocks else None
            if block is None or not _fx_version(block.data):
                continue
            if isinstance(block.data, (bytes, bytearray)):
                block.data = bytes(4) + bytes(block.data[4:])
            else:
                block.data.version = 0
    strip(psd)
    psd.save(path, update_preview=False)


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
    image.save(path, "PNG", pnginfo=chunks, compress_level=PNG_LEVEL, **keep)


# ========================
# Per-card driver (section 5 steps 3-9)
# ========================
def apply_jobs(psd_path: str, png_path: str, rel: str, jobs: dict, parsed: dict, parser: PSDParser, out_dir: str,
               fonts_dir: str, chart: Optional[dict] = None, types_dir: Optional[str] = None,
               psd: Optional[PSDImage] = None, view: Optional[int] = None, template: Optional[tuple] = None,
               ability_from: Optional[tuple] = None, title_anchor: int = 0) -> Baked:
    """
    _apply_jobs, as picture `view` of a card with variant tags shows it (variant_view(): its edits, renders, merge
    onto that picture's PNG png_path and its content gates; parsed is that picture's parse), the saved PSD keeping its
    own eye flags. view None: the card as saved. template: plan_types'. rel: where the card will be (a stars edit moves
    a card into its "N Stars" folder: its re-parse reads the stars there). ability_from: set_parts' (a "+ability" job).
    title_anchor: set_text_layer's anchor for a name job or the one title box of a parts job.
    """
    hold = ExitStack()
    try:
        with _files_seen(parser) as files:
            return _apply_jobs(psd_path, png_path, rel, jobs, parsed, parser, out_dir, fonts_dir, chart, types_dir, psd,
                               view, hold, template, ability_from, title_anchor, files)
    finally:
        hold.close()


@contextmanager
def _files_seen(parser):
    """The parser reads the files the edits set (Edit.files, filled in) while the bake gates them; its own reader after."""
    was, shown = getattr(parser, "linked", None), {}
    if parser is not None:
        parser.linked = lambda path: shown[path] if path in shown else (was(path) if was is not None else None)
    try:
        yield shown
    finally:
        if parser is not None:
            parser.linked = was


def _apply_jobs(psd_path: str, png_path: str, rel: str, jobs: dict, parsed: dict, parser: PSDParser, out_dir: str,
                fonts_dir: str, chart: Optional[dict] = None, types_dir: Optional[str] = None,
                psd: Optional[PSDImage] = None, view: Optional[int] = None, hold: ExitStack = None,
                template: Optional[tuple] = None,
                ability_from: Optional[tuple] = None, title_anchor: int = 0, files: Optional[dict] = None) -> Baked:
    """
    Apply a card's jobs, render the changed region, delta-merge the PNG, save (stripped: save_psd) and run every gate.

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
        types_dir: The card repo's Types/ folder (its PNGs), for a types edit that makes an icon
        psd: psd_path already open (card_bake's parse and classify share it), never changed: the edit goes on a copy
            (edit_copy: opened again from psd_path's bytes); None: opened here. Each PSD file is opened once per bake,
            the saved one too (tests/test_card_bake.py)
        files: Filled with the files the edits set (Edit.files), which the parser reads meanwhile (_files_seen)

    Returns:
        Baked; raises Refused (any other exception is the driver's "error"); a refusal of a stars or types edit's own
        gates carries its field, so the driver can bake the other fields without it
    """
    timings = {}
    start = perf_counter()
    with open(psd_path, "rb") as fh:
        raw = fh.read()
    old = use_record(psd if psd is not None else PSDImage.open(BytesIO(raw)), rel)
    new, old_records = edit_copy(old, raw), layer_records(old)     # the copy: a second open of the same bytes
    hold.enter_context(variant_view(old, view))     # picture `view`'s eye flags until the save
    hold.enter_context(variant_view(new, view))
    png = Image.open(png_path)
    png.load()
    timings["open"] = perf_counter() - start
    if png.size != (new.width, new.height) or png.mode not in ("RGB", "RGBA"):
        raise Refused("shape", f"PNG {png.mode} {png.size} against a {new.width}x{new.height} canvas")
    stale, flags = stale_stories(new), []
    repaired = set()
    if stale:                                   # a pre-fix bake's PSD: Photoshop already shows the old text
        from commands.txt2_repair import sync_stories
        repaired = sync_stories(new)
        if stale_stories(new):
            raise Refused("unsupported-card", f"{stale[0][0]}'s Txt2 story is stale and txt2_repair could not sync it")
        flags.append(f"Txt2 story repaired (commands.txt2_repair): {', '.join(name for name, _, _ in stale)}")
    drift = story_style_drift(new, kerning=False)   # kerning: the edited layers follow the story (follow_kerning)
    if drift:                                   # Photoshop shows the story's styles, not the layer's
        raise Refused("unsupported-card", f"{drift[0][0]}'s Txt2 story has other styles: {drift[0][1]}")

    # stars and types first (a title edit then clears the new stars), text next, the ability first: its FX records shift
    # the record indices of the digits and of a Name above them (Fire Staff's); the icon edits' indices are read once
    # all are in (index_icons). The card is rendered whole only once an edit is made (a refused one costs none), in one
    # RenderCache (the caller's when open), and without a stars or types edit or with a lone one only inside the dirty
    # box (gate_dirty)
    cache, first = render_cache(shared=True), None
    lone = len(jobs) == 1 and next(iter(jobs)) in ICON_FIELDS
    before, edits = record_map(new), []
    placed = placed_title(new, _card_name(parsed), getattr(parser, "linked", None), _card_path(parsed) or rel) \
        if "name" in jobs else None
    for name, value in sorted(jobs.items(), key=lambda job: (job[0] == "colors", job[0] not in ICON_FIELDS,
                                                             job[0] in (*STAT_FIELDS, "creatures"), job[0] == "name")):
        if name in ICON_FIELDS:
            try:
                plan = plan_stars(new, parsed, value, parser, split=True) if name == "stars" \
                    else plan_types(new, parsed, value, parser, template, types_dir)
                start = perf_counter()
                if first is None and not lone:
                    with cache:
                        first = shown = render_all(old)
                timings["outside"] = timings.get("outside", 0.0) + perf_counter() - start
                edit = set_icons(new, plan, types_dir, rel)
                start = perf_counter()
                with cache:
                    if lone:                            # old and new inside its dirty box only
                        first = gate_dirty(old, new, [edit], icon=True)
                    else:
                        shown = gate_outside(shown, new, edit)      # nothing else changes
                timings["outside"] = timings.get("outside", 0.0) + perf_counter() - start
            except Refused:
                raise
            except Exception as error:          # a layout the code trips on: that field's alone
                raise Refused("error", f"{name}: {type(error).__name__}: {error}", name) from error
            edits.append(edit)
        elif name in STAT_FIELDS:
            edits.append(set_stat(new, parsed, name, value))
        elif name == "creatures":
            edits += set_creatures(new, parsed, value)
        elif name == "ability" and _lone_part(new) is None:     # copies: as its one box (§3)
            record = card_parts(old, rel, _card_name(parsed), parser)
            found = record.get("ability") or []
            if len(found) != 1:
                raise Refused("unsupported-card", f"{len(found)} ability layers: edit each box on the website")
            edits += set_parts(new, {str(found[0]["id"]): {"text": value}}, record, fonts_dir, chart, _card_name(parsed),
                               parser, old)
        elif name == "name" and placed:         # a smart object's title (Gust of Wind's, its Motion Blur kept)
            edits.append(set_placed_title(new, placed, value, fonts_dir, rel, parser))
        elif name in TEXT_FIELDS:
            edits.append(set_text_layer(new, name, value, fonts_dir, chart, _card_name(parsed), parser,
                                        old, anchor=title_anchor if name == "name" else 0))
        elif name == "parts":                   # one box per layer (PARTS_EDITOR.md §4.4-4.6)
            record = card_parts(old, rel, _card_name(parsed), parser)
            edits += set_parts(new, value, record, fonts_dir, chart, _card_name(parsed), parser, old,
                               ability_from, rel, title_anchor)
        elif name == "colors":                  # last: on the text the edits above set
            edits.append(set_colors(new, value, fonts_dir, _card_name(parsed), chart))
        else:
            raise Refused("unsupported-card", f"{name}: not yet")
    for edit in edits:
        (files if files is not None else {}).update(edit.files)
    others = [edit for edit in edits if edit.field not in ICON_FIELDS and (edit.layers or edit.removed)]
    start = perf_counter()
    dirty = bool(others) and not any(edit.field in ICON_FIELDS for edit in edits)   # renders the dirty box only
    kept: dict = {}
    if dirty:
        with cache:
            first = gate_dirty(old, new, others, kept)     # valid inside the regions only (baseline's crop_pair)
    elif others:
        with cache:
            gate_untouched(shown, new, others)
    timings["outside"] = timings.get("outside", 0.0) + perf_counter() - start
    after = record_map(new)
    for edit in edits:
        if edit.icons is not None:
            index_icons(edit, before, after)
    edits = [edit for edit in edits if edit.layers or edit.removed]
    region = changed_region(edits, new)

    # region per edited field (section 5 step 3): the parity bar is measured over the pixels that edit can reach
    merged, parity, changed_pixels, whole = png, {}, 0, {}
    icc = export_icc(old, png)
    edited = edited_ids(edits)
    timings["baseline"] = timings["render"] = 0.0
    for edit in edits:
        start = perf_counter()
        try:
            base, parity[edit.field] = baseline(old, png, edit.region, icc, first)
        except Refused as refused:
            refused.field = edit.field if edit.field in PARTIAL_FIELDS else None
            raise
        text = parity[edit.field]["text"]
        if edit.whole:                          # every line laid out again: told, refused past the floor
            whole[edit.field] = measured = whole_redraw(old, raw, edit, png, fonts_dir, icc, text)
            if _past_floor(measured):
                raise Refused("parity", "%s redrawn whole bad %.2f%% mean %.2f max %d: past the %g%% floor"
                              % (edit.field, measured["bad_pct"], measured["mean"], measured["max"], PARITY_FLOOR))
        timings["baseline"] += perf_counter() - start
        start = perf_counter()
        cut = crop_pair(new, kept.get("after"), edit.region, icc, kept.get("box")) \
            if text == "preview" and kept.get("edited") == edited else None    # drawn by gate_dirty already
        if cut:
            render = cut[0]
        else:
            with cache:     # a region's first pinned pass (its key: the region and the pins): gate_idempotence's resumes it
                render = render_region(new, edit.region, icc, text, edited)
        timings["render"] += perf_counter() - start
        merged, replaced = merge_png(merged, base, render, edit.region)
        own = own_pixels(replaced, edit.region, [other.region for other in edits if other is not edit and other.region])
        if not own.any() and not _within_bar(parity[edit.field]):     # a blank render: the PNG would keep the old one
            raise Refused("parity", "%s: the render shows no change, baseline bad %.2f%% mean %.2f max %d"
                          % (edit.field, parity[edit.field]["bad_pct"], parity[edit.field]["mean"], parity[edit.field]["max"]),
                          edit.field if edit.field in PARTIAL_FIELDS else None)
        changed_pixels += int(replaced.sum())

    new_psd, new_png = join(out_dir, basename(psd_path)), join(out_dir, basename(png_path))
    saving = ON_MERGED(merged, new_png, png) if ON_MERGED is not None else None
    start = perf_counter()
    hold.close()                                # the saved card keeps its own eye flags
    save_psd(new, new_psd)
    for path, data in (files or {}).items():    # a placed title's file beside them (set_placed_title)
        with open(join(out_dir, basename(path)), "wb") as fh:
            fh.write(data)
    timings["save"] = perf_counter() - start
    if saving is None:
        save_png(merged, new_png, png)
    gate_size(psd_path, new_psd)
    reopened = frozen(use_record(PSDImage.open(new_psd), rel, _LOOKS.get(new)))   # the saved PSD's one open: every gate
    #                                           after reads it, drawn with the facts the merge was (it has no stored image)
    records, flag_bytes, block_bytes = gate_bytes(old_records, reopened, edits)
    hold.enter_context(variant_view(reopened, view))    # the read-back and re-render as picture `view`

    start = perf_counter()
    stories = {story for edit in edits if edit.field in (*TEXT_FIELDS, "colors") for story in [edit.story, *edit.stories]}
    gate_stories(old, reopened, stories | repaired,
                 [face for edit in edits for face in edit.fonts],
                 [index for edit in edits for index in edit.removed_stories],
                 [index for edit in edits for index in edit.added_stories], filters=True,
                 frames=[one.story for edit in edits for one in ([p for _, p, _ in edit.parts] or [edit])
                         if one.boxed and one.story is not None],
                 links=[done[1].smart_object.unique_id for edit in edits if edit.icons is not None
                        for done in edit.icons.done if done[0] == "made" and done[1].kind == "smartobject"],
                 relinked=[uuid for edit in edits for uuid in edit.relinked])
    gate_filters(old, reopened, edits)
    reparse = gate_reparse(parser, parsed, new_psd, rel, jobs, reopened, edits, flags)
    if "parts" in jobs:
        flags += gate_parts(parser, parsed, rel, jobs["parts"], old, reopened, fonts_dir)
    titles = (name_layer(reopened, _card_name(parsed)),
              name_layer(old, _card_name(parsed))) if "name" in jobs and not placed else None
    flags += gate_fx(reopened, edits, fonts_dir, chart, old, parser.icons, titles)
    timings["reparse"] = perf_counter() - start
    start = perf_counter()
    with cache:     # its keys hash the saved records: a layer the save changed is rendered again
        rerender = gate_idempotence(reparse, jobs, reopened, merged, edits, icc,
                                    {field: m["text"] for field, m in parity.items()}, parser, chart, kept)
    timings["idempotence"] = perf_counter() - start
    if saving is not None:
        saving.result()
    return Baked(jobs, edits, region, parity, changed_pixels, records, flag_bytes, new_psd, new_png, reparse, timings,
                 block_bytes, [name for edit in edits for name in edit.fx_names
                               if name in {n for n, _ in edit.fx} | {n for _, _, n in edit.rebuilt}],
                 list(dict.fromkeys([note for edit in edits for note in edit.notes] + flags)), icc, rerender,
                 {edit.field: edit.icons.how for edit in edits if edit.icons is not None}, whole,
                 {path: data for edit in edits for path, data in edit.files.items()})


# ========================
# Parts: one text box per PSD layer (docs/Finished/session-2026-10-26/PARTS_EDITOR.md §2-4)
# ========================
GONE = "the layer is gone from the PSD (deleted or remade in Photoshop since this edit): redo it on the website if still wanted"
CHANGED = "changed in Photoshop since this edit: redo it on the website if still wanted"
HAS_ABILITY = "the card has an Ability layer now (added in Photoshop): redo it on the website"
PART_KINDS = (("ability", "ability"), ("title", "name"))   # record key -> field, in bake order


@dataclass
class PartsPlan:
    """A card's `parts` edits sorted against its PSD (plan_parts, §4.3)."""
    jobs: dict = field(default_factory=dict)        # key -> the part edit to bake (text or delete)
    outcome: dict = field(default_factory=dict)     # key -> "redundant" | "skipped"
    skipped: dict = field(default_factory=dict)     # key -> {"text", "now", "why"} (_psd.skipped)
    refusals: list = field(default_factory=list)    # Refusal: impossible cases
    notes: list = field(default_factory=list)
    where: dict = field(default_factory=dict)       # key -> (field, kind, 1-based place among its kind's parts, count)


def _bare_parser() -> PSDParser:
    """A parser without the icon vocabulary, for callers that have none (tests, one-off checks)."""
    return PSDParser([], StatsDatabase(), IconVocabulary.build([], [], None))


def saved_view(psd: PSDImage) -> Optional[int]:
    """The picture the saved eye flags show (each tagged layer visible exactly in it), or None for none."""
    tagged = [(layer.visible, tags) for layer in psd.descendants() for tags in [layer_tag(layer.name)[1]] if tags]
    return next((n for n in variants_of(psd) if all(shown == (n in tags) for shown, tags in tagged)), None)


def part_text(parser: Optional[PSDParser], layer) -> str:
    """One layer's reading as its box shows it (PSDParser.layer_text)."""
    return (parser or _bare_parser()).layer_text(layer)


def _lone_part(psd: PSDImage):
    """ability_layer(), or None when it refuses (a plain ability edit then goes through set_parts)."""
    try:
        return ability_layer(psd)
    except Refused:
        return None


_PARTS: "WeakKeyDictionary" = WeakKeyDictionary()   # a frozen PSD -> {(rel, name, id(parser)): (parser, card_parts)}


def frozen(psd: PSDImage) -> PSDImage:
    """psd, which the bake only reads (the card as opened, the saved one read back): its card_parts kept; psd."""
    _PARTS.setdefault(psd, {})
    return psd


def card_parts(psd: PSDImage, rel: str, name: str, parser: Optional[PSDParser]) -> dict:
    """The details record's `parts` (§2.3) over every variant: psd_parts.card_parts, a bare parser when none; once
    per frozen PSD (a copy each call)."""
    kept, key = _PARTS.get(psd), (rel, name, id(parser))
    if kept is not None and key in kept:
        return deepcopy(kept[key][1])
    out = psd_parts.card_parts(psd, rel, name, parser or _bare_parser())
    if kept is not None:
        kept[key] = (parser, deepcopy(out))     # the parser held: its id stays its own
    return out


def record_parts(record: dict) -> dict:
    """str(layer id) -> (field, record key, part) over a parts record."""
    return {str(part["id"]): (field_name, kind, part) for kind, field_name in PART_KINDS for part in record.get(kind) or []}


def _part_same(a: Optional[str], b: Optional[str], vocab) -> bool:
    """Two part texts read alike: tokens canonical ([Fire] is [fire]), layer_text's whitespace rule (a part keeps its
    quotes and a space before punctuation: normalize_ability's stripping made "Lifesteal ." -> "Lifesteal." redundant)."""
    def key(text: Optional[str]) -> str:
        text = canonical_tokens(text, vocab) or ""
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x03", "\n").replace("\t", " ")
        return straight(text).replace("\ufeff", "").strip(" \n")
    return key(a) == key(b)


def plan_parts(psd: PSDImage, parsed: dict, parts: dict, parser: Optional[PSDParser], record: Optional[dict] = None) -> PartsPlan:
    """
    Sort a card's `parts` edits against the PSD (§4.3): a gone layer, one no longer a part of its field, or one whose raw
    text changed since the edit (sha) is skipped and flagged ("redo it on the website"), unless it already reads the
    edit (redundant); one reading the edit is redundant; the rest are jobs. Never overwrites Photoshop work: a sha
    mismatch only ever skips. Impossible: emptying the title.

    Args:
        psd: The open card
        parsed: parse_card() of it
        parts: The entry's `parts`
        parser: offline_parser()
        record: card_parts() of psd (computed when None)

    Returns:
        PartsPlan
    """
    info = parsed.get("info")
    rel = (info.path if info is not None else parsed["card"].get("path")) or ""
    record = card_parts(psd, rel, _card_name(parsed), parser) if record is None else record
    found, index, vocab, plan = record_parts(record), lyid_index(psd), getattr(parser, "icons", None), PartsPlan()
    for kind, field_name in PART_KINDS:
        for k, part in enumerate(record.get(kind) or []):
            plan.where[str(part["id"])] = (field_name, kind, k + 1, len(record[kind]))

    def skip(key, edit, why, now=None):
        plan.outcome[key] = "skipped"
        plan.skipped[key] = {"text": edit.get("text"), "now": now, "why": why}
    for key, edit in parts.items():
        if key == ADD_ABILITY:
            if record.get("ability"):
                skip(key, edit, HAS_ABILITY)
            elif template_for(parsed["card"].get("type"), rel) is None:
                plan.refusals.append(Refusal("unsupported-card", "ability: no template for this kind of card"))
            else:
                plan.jobs[key] = edit
            continue
        layer = index.get(int(key))
        if layer is None:
            skip(key, edit, GONE)
            continue
        hit = found.get(key)
        if not _is_text_layer(layer) or hit is None or hit[0] != edit["field"]:
            skip(key, edit, CHANGED, part_text(parser, layer) if _is_text_layer(layer) else None)
            continue
        now = hit[2]["text"]
        if raw_sha(layer) != edit["sha"]:       # Photoshop's work stands
            if not edit.get("delete") and _part_same(now, edit["text"], vocab):
                plan.outcome[key] = "redundant"
            else:
                skip(key, edit, CHANGED, now)
        elif not edit.get("delete") and _part_same(now, edit["text"], vocab):
            plan.outcome[key] = "redundant"
        else:
            plan.jobs[key] = edit
    titles = [str(p["id"]) for p in record.get("title") or []]
    if titles and all(key in plan.jobs and plan.jobs[key].get("delete") for key in titles):
        plan.refusals.append(Refusal("unsupported-card", "the title can't be emptied: keep one title box"))
    abilities = [str(p["id"]) for p in record.get("ability") or []]
    if abilities and all(key in plan.jobs and plan.jobs[key].get("delete") for key in abilities):
        plan.notes.append("ability: every Ability layer is deleted: the card has no ability now")
    return plan


def _drop_story(psd: PSDImage, index: int) -> list:
    """
    Remove the Txt2 story at index and the text frames only it names (the frames after them renumbered in every story),
    and move each later type layer's TextIndex down by one: Photoshop reads a layer's text from its story.

    Returns:
        The type layers whose TextIndex moved
    """
    from psd_tools.api import text_edit
    found = text_edit._txt2(psd)
    if found is None:
        return []
    root, keys, items = found
    _drop_story_in(root, keys, items, index)
    moved = []
    for layer in psd.descendants():
        at = layer._data.text_data.get(b"TextIndex") if _is_text_layer(layer) else None
        if at is not None and int(getattr(at, "value", at)) > index:
            at.value = int(at.value) - 1
            moved.append(layer)
    return moved


def _frames_out(root, indices):
    """A copy of a Txt2 root with the frames only these stories use blanked (gate_stories' frames)."""
    from psd_tools.api import text_edit
    from psd_tools.psd import engine_data
    root = deepcopy(root)
    keys, items = _story_keys(root)
    own = {i for index in indices for i in text_edit._frame_refs(items[index], keys)}
    shared = {i for k, story in enumerate(items) if k not in set(indices) for i in text_edit._frame_refs(story, keys)}
    frames = text_edit._resource_items(root, keys, keys.frame_set)
    for i in own - shared:
        frames[i] = engine_data.Dict()
    return root


def _drop_story_in(root, keys, items, index: int) -> None:
    """_drop_story's Txt2 part, on any root (gate_stories replays it on the old card's)."""
    from psd_tools.api import text_edit
    refs = text_edit._frame_refs(items[index], keys)
    shared = {i for k, story in enumerate(items) if k != index for i in text_edit._frame_refs(story, keys)}
    del items[index]
    gone = sorted(set(refs) - shared)
    if gone:
        frames = text_edit._resource_items(root, keys, keys.frame_set)
        kept = [i for i in range(len(frames)) if i not in gone]
        for i in reversed(gone):
            del frames[i]
        moved = {old: new for new, old in enumerate(kept) if old != new}
        for story in items:
            text_edit._set_frame_refs(story, keys, moved)


def remove_layer(psd: PSDImage, layer) -> list:
    """
    Delete a layer (its record and channels) and, for a type layer, its Txt2 story, renumbering the later TextIndex
    values (§4.6).

    Returns:
        The type layers whose TextIndex moved
    """
    index = story_index(layer) if _is_text_layer(layer) else None
    layer.parent.remove(layer)
    return _drop_story(psd, index) if index is not None else []


def _part_raw(raw: str, was: str, text: str) -> str:
    """A part's new raw text: the box's edit was -> text put back over the layer's own text (project: its blank ends,
    indents and breaks between stay); a new line takes the layer's own break (\\x03 when it has one, else \\r)."""
    brk = "\x03" if "\x03" in raw else "\r"
    return project(was, text.replace("\r\n", "\n").replace("\r", "\n"), raw, brk)


def _combine(which: str, edits: list, subs: list, psd: PSDImage, olds: dict, news: dict, removed=(),
             removed_stories=(), renumbered=(), boxes=(), notes=()) -> Edit:
    """One Edit for a field's part jobs (Edit.parts: each layer's own), the deletions in it; old and new:
    {layer id: its part's text} (new None: deleted)."""
    now = record_map(psd)
    for (_, edit, _), ids in zip(edits, subs):
        _renumber(edit, ids, now)
    kept = set(_record_ids(psd))
    region = union_box([edit.region for _, edit, _ in edits] + list(boxes))
    first = edits[0][1] if edits else None
    return Edit(which, olds, news, [layer for layer, _, _ in edits],
                [i for _, edit, _ in edits for i in edit.records],
                region, hidden=[i for _, edit, _ in edits for i in edit.hidden],
                fx=[fx for _, edit, _ in edits for fx in edit.fx],
                dropped=[name for _, edit, _ in edits for name in edit.dropped],
                rebuilt=[spec for _, edit, _ in edits for spec in edit.rebuilt],
                restyled=[spec for _, edit, _ in edits for spec in edit.restyled],
                fx_names=[name for _, edit, _ in edits for name in edit.fx_names],
                notes=[note for _, edit, _ in edits for note in edit.notes] + list(notes),
                held=[held for _, edit, _ in edits for held in edit.held],
                story=first.story if first else None,
                stories=[edit.story for _, edit, _ in edits],
                fonts=list(dict.fromkeys(face for _, edit, _ in edits for face in edit.fonts)),
                whole=any(edit.whole for _, edit, _ in edits),
                parts=[(layer.layer_id, edit, copy) for layer, edit, copy in edits], several=True,
                removed=list(removed), removed_stories=list(removed_stories),
                renumbered=sorted({_record_index(layer) for layer in renumbered if id(layer._record) in kept}))


def set_parts(psd: PSDImage, jobs: dict, record: dict, fonts_dir: str, chart: Optional[dict] = None, card: str = "",
              parser: Optional[PSDParser] = None, original: Optional[PSDImage] = None,
              ability_from: Optional[tuple] = None, rel: str = "", anchor: int = 0) -> list:
    """
    Bake a card's part jobs (plan_parts) on the open card (§4.4-4.6): the deletions first (remove_layer: the layer, its
    visible FX layers and copies, its story), then each ability part and title part set as one text
    layer (set_text_layer, layer=) with its box's edit put back over the layer's raw text (_part_raw), its copies the
    same, a watermark of the title the new title when it printed the old one (else told).

    Args:
        psd: The open card (edited in place)
        jobs: plan_parts().jobs
        record: card_parts() of the card as opened
        original: The card before the bake's edits (set_text_layer's; record ids and story indices of what is deleted)
        ability_from: (the card kind's template, its name): template_for's, for a "+ability" job (added last)
        rel: The card's repo path (where a "+ability" goes and its font: ability_place)
        anchor: set_text_layer's, for the card's one title box (R4: an earlier bake's move, undone first)

    Returns:
        One Edit per field ("ability" then "name"), then the added ability's; raises Refused as set_text_layer does
    """
    found, index = record_parts(record), lyid_index(psd)
    was_index = lyid_index(original) if original is not None else index
    follow = record.get("follow") or []

    def followers(lyid: int) -> list:       # its copies
        return [index[f["id"]] for f in follow if f["how"] == "copy" and lyid in f["of"] and f["id"] in index]

    def first_story(layer) -> Optional[int]:    # gate_stories reads TextIndex as the card was opened
        before = was_index.get(layer.layer_id)
        return story_index(before) if before is not None else story_index(layer)
    out, gone, deleted = [], set(), {}
    for key, job in jobs.items():               # deletions first: an edited part's ink is told against what stays
        if not job.get("delete"):
            continue
        field_name, _, part = found[key]
        layer = index[int(key)]
        doomed = [layer, *(fx for fx in fx_layers(psd, layer) if fx.is_visible()), *followers(layer.layer_id)]
        info = deleted.setdefault(field_name, {"removed": [], "stories": [], "boxes": [], "notes": [], "moved": []})
        for one in doomed:
            if one.layer_id in gone:
                continue
            gone.add(one.layer_id)
            before = was_index.get(one.layer_id)
            info["removed"].append(_record_index(before) if before is not None else None)
            if _is_text_layer(one) and before is not None and story_index(before) is not None:
                info["stories"].append(story_index(before))
            info["boxes"].append(_pad(tuple(one.bbox), fx_pad(one), psd))
            info["moved"] += remove_layer(psd, one)
        info["notes"].append(f"{field_name}: {part['text'][:40]!r} deleted"
                             + (" with its FX and copy layers" if len(doomed) > 1 else "")
                             + ": the text below it doesn't move up")
        if any(r is None for r in info["removed"]):
            raise Refused("bytes", f"{part['layer']!r}: a deleted layer is not in the card as opened")
    joined = {kind: [p for p in record.get(kind) or []] for kind, _ in PART_KINDS}
    for kind, which in PART_KINDS:
        todo = [(str(p["id"]), p) for p in joined[kind] if str(p["id"]) in jobs and not jobs[str(p["id"])].get("delete")]
        info = deleted.pop(which, None) or {"removed": [], "stories": [], "boxes": [], "notes": [], "moved": []}
        if not todo and not info["removed"]:
            continue
        edits, subs, notes = [], [], list(info["notes"])
        for key, part in todo:
            layer = index[int(key)]
            text = jobs[key]["text"]
            layer_was = layer.text
            raw = _part_raw(layer_was, part["text"], text)
            copies = followers(layer.layer_id)
            edit = set_text_layer(psd, which, raw, fonts_dir, chart if kind == "ability" else None, card, parser,
                                  original, layer=layer,
                                  anchor=anchor if kind == "title" and len(joined[kind]) == 1 else 0)
            edit.story = first_story(layer)
            typed = sum(raw.strip(PAD).count(c) for c in "\r\x03") - sum(layer_was.strip(PAD).count(c) for c in "\r\x03")
            edit.notes = [note for note in edit.notes if not _typed_lines(note, typed)]    # an Enter in the box is no overflow
            many = len(joined[kind]) > 1
            if many:
                label = "ability" if which == "ability" else "title"
                where = f"{label} {joined[kind].index(part) + 1}/{len(joined[kind])}"
                edit.notes = [sub(rf"^((?:[\w-]+: )?){which}\b", rf"\1{which} ({where})", note, count=1)
                              for note in edit.notes]
            edits.append((layer, edit, False))
            subs.append(_record_ids(psd))
            for copy in copies:
                again = _part_raw(copy.text, part["text"], text)
                if again != copy.text or edit.lifted:
                    done = set_text_layer(psd, which, again, fonts_dir, None, card, parser, original,
                                          layer=copy, copy=True, up=edit.lifted)
                    done.story = first_story(copy)
                    done.notes = [f"{note} (its copy {copy.name[:24]!r})" for note in done.notes if note not in edit.notes]
                    edits.append((copy, done, True))
                    subs.append(_record_ids(psd))
            if copies:
                notes.append(f"{which}: also printed by {', '.join(c.name for c in copies)}: "
                             f"{'they follow' if len(copies) > 1 else 'it follows'}")
        if which == "name" and todo and len(joined["title"]) > 1:
            split_title_check(psd, joined["title"], jobs, index, was_index,
                              {layer.layer_id: edit.notes for layer, edit, copy in edits if not copy})
        if which == "name" and todo:            # a watermark of the joined title follows it
            old_title = " ".join(title_text(p["text"]) for p in joined["title"])
            new_title = " ".join(title_text(jobs[str(p["id"])]["text"]) if str(p["id"]) in jobs else title_text(p["text"])
                                 for p in joined["title"] if not jobs.get(str(p["id"]), {}).get("delete"))
            for f in follow:
                mark = index.get(f["id"])
                if f["how"] != "watermark" or mark is None:
                    continue
                if name_key(mark.text) != name_key(old_title):
                    notes.append(f"name: the watermark {f['layer']!r} doesn't match the title: not changed")
                    continue
                done = set_text_layer(psd, "name", project(title_text(mark.text), new_title, mark.text), fonts_dir,
                                      None, card, parser, original, layer=mark, copy=True)
                done.story = first_story(mark)
                edits.append((mark, done, True))
                subs.append(_record_ids(psd))
                notes.append(f"name: the watermark {f['layer']!r} follows the title")
        mine = [str(p["id"]) for p in joined[kind] if str(p["id"]) in jobs]
        out.append(_combine(which, edits, subs, psd, {key: found[key][2]["text"] for key in mine},
                            {key: jobs[key].get("text") for key in mine}, info["removed"],
                            info["stories"], info["moved"], info["boxes"], notes))
    if ADD_ABILITY in jobs:                     # scheduled last (§4.7)
        if ability_from is None:
            raise Refused("unsupported-card", "ability: no template to add the Ability layer from")
        out.append(add_ability(psd, jobs[ADD_ABILITY]["text"], *ability_from, fonts_dir, chart, card, parser,
                               original, rel))
    return out


def insert_ability(psd: PSDImage, template: PSDImage):
    """
    The template's visible Ability type layer copied into the card (§4.7, the fork's copy_to: its box, font, size and
    Txt2 story, its fonts merged): at the top of the card's Text group (the root when there is none), named Ability,
    a fresh layer id. A card whose canvas is not its template's is impossible (none, 2026-10-25).

    Returns:
        The new layer
    """
    if (psd.width, psd.height) != (template.width, template.height):
        raise Refused("unsupported-card", f"ability: the card's canvas {psd.width}x{psd.height} is not its template's "
                                          f"{template.width}x{template.height}")
    source = next((layer for layer in template.descendants() if _is_text_layer(layer) and layer.is_visible()
                   and base_name(layer).strip().lower() == "ability"), None)
    if source is None:
        raise Refused("unsupported-card", "ability: the template has no visible Ability layer")
    group = _text_group(psd)
    layer = source.copy_to(group if group is not None else psd)
    layer.name = "Ability"
    return layer


def set_text_font(layer, font: str, size: float, leading: float) -> None:
    """Every run of a type layer in font at size and leading (rendered px: over the layer's y scale), as the bake sets a
    typed icon's font (add_font, set_span_style); an auto leading stays auto."""
    from psd_tools.composite.text.model import TextModel
    model = TextModel.from_layer(layer)
    scale = hypot(*model.transform[2:4]) or 1.0
    keys = {"Font": font, "FontSize": round(size / scale, 4)}
    if not model.runs[0].style.auto_leading:
        keys["Leading"] = round(leading / scale, 4)
    layer.add_font(font)
    layer.set_span_style(0, len(layer.text), **keys)


def set_text_box(layer, box: tuple) -> None:
    """
    A paragraph type layer's box moved and sized to box (x0, y0, x1, y1, canvas px; the transform's scale kept): the
    TySh transform and the record (its pixels and fxrp follow), BoxBounds, the
    TySh bounds, and the Txt2 frame's Bezier (its points scaled, Photoshop's order kept) and its frame and region
    strikes, which must agree (+ability, §4.7). Point text raises Refused.
    """
    from psd_tools.api import text_edit
    from psd_tools.psd import engine_data
    from psd_tools.psd.descriptor import UnitFloat
    child = layer.engine_dict["Rendered"]["Shapes"]["Children"][0]
    if int(getattr(child.get("ShapeType"), "value", child.get("ShapeType", 0))) != 1:
        raise Refused("unsupported-card", "ability: the template's Ability is point text")
    xx, xy, yx, yy, tx, ty = layer._data.transform
    dx, dy = round(box[0] - tx), round(box[1] - ty)
    w, h = (box[2] - box[0]) / xx, (box[3] - box[1]) / yy           # text-space pt
    f = lambda v: engine_data.Float(float(v))
    was = [float(v) for v in child["Cookie"]["Photoshop"]["BoxBounds"]]
    sx, sy = w / ((was[2] - was[0]) or 1.0), h / ((was[3] - was[1]) or 1.0)
    layer._data.transform = (xx, xy, yx, yy, tx + dx, ty + dy)
    layer.left, layer.top = layer.left + dx, layer.top + dy           # the fork carries fxrp along
    child["Cookie"]["Photoshop"]["BoxBounds"] = engine_data.List([f(0), f(0), f(w), f(h)])
    bounds = layer._data.text_data[b"bounds"]
    for key, value in ((b"Rght", w), (b"Btom", h)):
        old = bounds[key]
        bounds[key] = UnitFloat(float(value), old.unit) if isinstance(old, UnitFloat) else type(old)(float(value))
    found = text_edit._txt2(layer._psd)
    if found is not None:
        root, keys, stories = found
        story = text_edit._nth_story(stories, int(layer._data.text_data[b"TextIndex"].value), keys)
        frames = text_edit._resource_items(root, keys, keys.frame_set)
        for i in text_edit._frame_refs(story, keys):   # the Bezier scaled: Photoshop's point order kept
            points = [float(v) for v in frames[i]["0"]["1"]["0"]]
            frames[i]["0"]["1"]["0"] = engine_data.List([f((v - was[k % 2]) * (sx, sy)[k % 2])
                                                         for k, v in enumerate(points)])
        tag_key, tag, _ = keys.frame_strike

        def same(rect) -> bool:                 # the old box's rect
            try:
                return len(rect) == 4 and all(abs(float(a) - b) < 0.01 for a, b in zip(rect, was))
            except (TypeError, ValueError):
                return False
        pending = [story[keys.view].get(keys.strikes)]
        while pending:                          # the frame strike's rect and its region strikes' (the old box)
            node = pending.pop()
            if isinstance(node, engine_data.List):
                pending.extend(node)
            elif isinstance(node, engine_data.Dict):
                rect = node.get("1")
                if rect is not None and (str(getattr(node.get(tag_key), "value", node.get(tag_key))) == tag
                                         or isinstance(rect, engine_data.List) and same(rect)):
                    node["1"] = engine_data.List([f(0), f(0), f(w), f(h)])
                pending.extend(node.values())
    layer.__dict__.pop("_typesetting", None)
    layer._psd.mark_updated()


def add_ability(psd: PSDImage, text: str, template: PSDImage, name: str, fonts_dir: str, chart: Optional[dict] = None,
                card: str = "", parser: Optional[PSDParser] = None,
                original: Optional[PSDImage] = None, rel: str = "") -> Edit:
    """A "+ability" job: insert_ability, in the card's own font and size and in its free text area (ability_place.plan:
    owner 2026-10-25), the placeholder drawn there, then the text set as any ability's (set_text_layer) and fitted to
    that area, flagged. Its record and story are declared added (gate_bytes, gate_stories), the faces the Txt2 FontSet
    gained in Edit.fonts."""
    from commands import ability_place
    faces = lambda: [_story_font_name(entry) for entry in _story_fonts(_txt2(psd)) or []] if _txt2(psd) is not None else []
    before = faces()
    raw = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n").replace("\n", "\r")
    planned = ability_place.plan(psd, rel or f"{card}.psd", template, f"{TEMPLATE_DIR}/{name}.psd", raw, fonts_dir, parser)
    layer = insert_ability(psd, template)
    story = story_index(layer)
    set_text_font(layer, planned.font, planned.size, planned.leading)
    set_text_box(layer, planned.box)
    layer.rasterize(fonts=font_map(fonts_dir))     # the placeholder in its new font and place (drawn_in_font)
    edit = set_text_layer(psd, "ability", raw, fonts_dir, chart, card, parser, original, layer=layer,
                          area=planned.box)
    edit.story = None                           # a new story: gate_stories takes it out (added_stories)
    x0, y0, x1, y1 = planned.box
    design = on_panel_design(psd, layer, planned.panel) if planned.panel else None
    out = _combine("ability", [(layer, edit, False)], [_record_ids(psd)], psd, {}, {ADD_ABILITY: text},
                   notes=[f"ability: ability layer created from {name}: check its place",
                          f"ability: box {x0},{y0} to {x1},{y1}", *planned.notes]
                   + ([f"ability: {design}; check the card"] if design else []))
    out.added, out.added_stories = [_record_index(layer)], [story]
    out.fonts = list(dict.fromkeys(faces()[len(before):] + out.fonts))
    return out


SPLIT_SHAPE = re_compile(r"runs onto \d+ lines|runs off the canvas|a line of it moves below the title")


def split_title_check(psd: PSDImage, parts: list, jobs: dict, index: dict, was_index: dict, notes: dict) -> None:
    """
    A split title's box edits bake right or refuse plainly (owner 2026-11-08): among the parts one picture shows
    together, an edited part may not run into another (more ink shared than before), onto more lines than typed, off the
    card, nor change their reading order.

    Args:
        psd: The card, its title parts set
        parts: card_parts()' title parts
        jobs: plan_parts().jobs
        index, was_index: lyid_index() of psd and of the card as opened
        notes: {layer id: its Edit's notes}

    Raises:
        Refused("unsupported-card", field "parts"), naming the part
    """
    shape = (psd.height, psd.width)
    kept = [p for p in parts if not jobs.get(str(p["id"]), {}).get("delete") and p["id"] in index]
    pictures = sorted({n for p in kept for n in p.get("variants") or ()}) or [None]
    ink = lambda layer: _on_canvas(layer, shape)[..., 3] > INK_ALPHA * 255
    now, was = {}, {}

    def refuse(p, group: list, why: str) -> None:
        text = title_text(index[p["id"]].text)
        raise Refused("unsupported-card", f"name: title part {group.index(p) + 1}/{len(group)} ({text[:30]!r}) {why}: "
                                          "shorten it, or change the title in Photoshop", "parts")
    for n in pictures:
        group = [p for p in kept if p.get("variants") is None or n in p["variants"]]
        edited = [p for p in group if str(p["id"]) in jobs]
        if len(group) < 2 or not edited:
            continue
        for p in edited:
            for note in notes.get(p["id"], []):
                if SPLIT_SHAPE.search(note):
                    refuse(p, group, "would run off the card" if "canvas" in note else "would break onto another line")
        for p in group:
            if p["id"] not in now:
                now[p["id"]] = ink(index[p["id"]])
                was[p["id"]] = ink(was_index[p["id"]]) if p["id"] in was_index else now[p["id"]]
        for p in edited:
            for q in group:
                if q is not p and int((now[p["id"]] & now[q["id"]]).sum()) > int((was[p["id"]] & was[q["id"]]).sum()):
                    refuse(p, group, f"would run into {title_text(index[q['id']].text)[:30]!r}, another part of the title")
        order = lambda found: [layer.layer_id for layer in PSDParser.reading([found[p["id"]] for p in group])]
        if all(p["id"] in was_index for p in group) and order(index) != order(was_index):
            refuse(edited[0], group, "would put the title's parts in another order")


def _typed_lines(note: str, typed: int) -> bool:
    """A title's "runs onto N lines (it has M)" or "a line of it moves below" note the box's own line breaks account for."""
    found = search(r"runs onto (\d+) lines \(it has (\d+)\)", note)
    if typed > 0 and note.endswith("a line of it moves below the title"):
        return True
    return bool(found) and typed > 0 and int(found.group(1)) <= int(found.group(2)) + typed


# ========================
# Stars by folder and first type icons (PARTS_EDITOR.md §4.8)
# ========================
STAR_PATH = re_compile(r"^((?:[^/]+/)*)(\d+) Stars/[^/]+$", IGNORECASE)   # a card in an "N Stars" folder (debuffs too)


def _card_path(parsed: dict) -> str:
    info = parsed.get("info")
    return (info.path if info is not None else parsed["card"].get("path")) or ""


def star_folder(rel: str) -> Optional[tuple]:
    """(the path up to the "N Stars" folder, N) of a card whose stars come from its folder, else None."""
    found = None if stars_from_psd(rel or "") else STAR_PATH.match(rel or "")
    return (found.group(1), int(found.group(2))) if found else None


def star_target(rel: str, stars: int) -> Optional[str]:
    """The "N Stars" folder a stars edit moves a folder-starred card into ("Items/Heal/3 Stars"), or None."""
    found = star_folder(rel)
    return f"{found[0]}{stars} Stars" if found and found[1] != stars else None


def template_for(card_type: str, rel: str = "") -> Optional[str]:
    """The template file a card's first ability or type icon comes from (Items/Active: ActiveItem_Template), or None."""
    if card_type == "item" and rel.split("/")[1:2] == ["Active"]:
        return f"{TEMPLATE_DIR}/{ACTIVE_TEMPLATE}.psd"
    stem = TEMPLATES.get(card_type or "")
    return f"{TEMPLATE_DIR}/{stem}.psd" if stem else None


def template_icon(template: PSDImage, psd: PSDImage, parser: PSDParser, name: str = "") -> Optional[tuple]:
    """(spot on the card, the template's first type icon, the template's name): the template's spot scaled to the
    card's canvas, or None when it has no type icon."""
    images, _ = type_row(template, parser, name)
    if not images:
        return None
    rel = f"{TEMPLATE_DIR}/{name}.psd" if name else ""
    units = [_Unit([layer], _icon_core(layer, rel)) for _, layer in images]
    first = _reading(units, _median_size(units)[1], False)[0]
    sx, sy = psd.width / template.width, psd.height / template.height
    x0, y0, x1, y1 = first.core
    return (round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy)), first.layers[0], name


def parts_jobs(jobs: dict) -> dict:
    """The part edits among a Classification's jobs (its "parts" job), else {}."""
    return jobs.get("parts") or {}


# ========================
# Module Exports
# ========================
__all__ = [
    'BADGE_PILL',
    'BADGE_RATIO',
    'BADGE_RESIZE',
    'Baked',
    'Classification',
    'Edit',
    'FONTS_DIR',
    'ICON_FIELDS',
    'IconPlan',
    'KIND',
    'OK_FONTS',
    'Refusal',
    'Refused',
    'PARTIAL_FIELDS',
    'STATUS_CHART',
    'STORY_KEYS',
    'STORY_MIRROR',
    'STORY_STYLE_KEYS',
    'STAT_FIELDS',
    'SAME_RUN',
    'STAT_MAX',
    'ART_FILL',
    'STYLE_KEYS',
    'StatusColor',
    'TEXT_BLENDS',
    'TEXT_FIELDS',
    'TemplateRow',
    'ANCHORED',
    'BADGE_KEYS',
    'BADGE_TAGS',
    'COST_TAGS',
    'ICON_FONT',
    'ability_layer',
    'project',
    'badge_fit',
    'badge_fits',
    'badge_spans',
    'below_box',
    'badge_tags',
    'cost_tags',
    'badge_wants',
    'cap_of',
    'bare_icons',
    'apply_jobs',
    'artifacts',
    'baseline',
    'body_fill',
    'canonical_tokens',
    'canonical_type',
    'changed_range',
    'deletion_start',
    'changed_region',
    'classify',
    'comp_top',
    'comp_custom',
    'count_room',
    'art_fill',
    'cap_why',
    'crop_pair',
    'ink_beyond',
    'inks_over',
    'typed_icon_fit',
    'collapse',
    'export_icc',
    'foreign_profile',
    'frozen',
    'fx_layers',
    'FxPlan',
    'fx_kind',
    'fx_name',
    'fx_phrases',
    'gate_bytes',
    'gate_dirty',
    'gate_fx',
    'gate_fx_colours',
    'gate_idempotence',
    'gate_outside',
    'gate_untouched',
    'dirty_box',
    'drawable',
    'gate_reparse',
    'body_sizes',
    'gate_size',
    'glyph_places',
    'gate_status_words',
    'gate_stories',
    'edit_parts',
    'edit_styles',
    'edit_copy',
    'edited_ids',
    'icc_vote',
    'preview_image',
    'icon_font',
    'icon_style',
    'variant_fill',
    'variant_fills',
    'icon_text',
    'keep_tokens',
    'image_icons',
    'index_icons',
    'keep_breaks',
    'keep_ends',
    'title_styles',
    'kept_chars',
    'held_words',
    'kept_spans',
    'new_fx',
    'no_glyph',
    'layer_records',
    'layout_fonts',
    'fonts_in',
    'look_facts',
    'look_kwargs',
    'changed_part',
    'lettered_icons',
    'merge_png',
    'whole_redraw',
    'name_layer',
    'normalize_ability',
    'offline_parser',
    'own_pixels',
    'parse_card',
    'phrase_ranges',
    'plan_fx',
    'plan_stars',
    'plan_types',
    'read_name',
    'real_notes',
    'record_map',
    'render_all',
    'render_cache',
    'shown_name',
    'splice_text',
    'spot_entries',
    'spot_key',
    'spot_record',
    'render_record',
    'records_of',
    'merge_records',
    'set_records',
    'render_region',
    'raster_key',
    'run_fills',
    'save_png',
    'same_types',
    'save_psd',
    'same_glyphs',
    'same_places',
    'strip',
    'text_choices',
    'use_record',
    'set_icons',
    'set_creatures',
    'creature_bar',
    'creature_stats',
    'several_rows',
    'stat_caps',
    'set_stat',
    'stale_stories',
    'stat_value',
    'text_kwargs',
    'set_text_layer',
    'one_colour',
    'alternate_spans',
    'set_alternates',
    'stylistic_alternates',
    'status_fills',
    'status_spans',
    'status_word_spans',
    'stories',
    'story_index',
    'story_style_drift',
    'template_rows',
    'type_row',
    'unlettered_icons',
    'unshaped_icons',
    'uses_icon_font',
]
