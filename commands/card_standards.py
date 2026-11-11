"""
Guard 2 of the push guards (docs/Campaigns/PUSH_GUARDS.md section 6): a pushed card PSD against the card standards of
docs/Campaigns/CARD_STANDARDIZATION_CAMPAIGN.md. Rules R1-R6, R9 and R11 run over an open PSD (R10, the PNG against
the PSD, is png_export.png_check; R7, the fonts, is the font check's alone since 2026-10-24, owner). R8, the Single-line
Composer (L10), is always set and never told (owner, 2026-10-25): fix() sets it with the other fixes; box text that
rewraps is redrawn (the guards re-export its PNG). R3 and R4 keep only
the real issues (several visible Ability layers, or a hidden one with words the parser reads nowhere; a hidden or
rasterized Name layer the title can't be read from): the template layout they also checked was dropped by the owner on
2026-10-23.
Since the parts editor (PARTS_EDITOR.md §7) R3's several Ability layers are a note ("not standardized yet"); a title
split over several layers is standard (owner 2026-11-08: every part named Name, read in reading order). R11 also
suggests names for the parts not named Name or Ability, R12 checks variant tags and R13
duplicate layer ids. R14 (owner 2026-10-30) reports card names typed in another letter case (report only); R15 the
ability's or title's text run off the card's edge (report only; existing cards waived as R4).
R9 (Txt2 stories) is fixed by the fork with byte gates, the rest is reported. R11's rename (exact layer names) is built
and gated but reported until the owner says yes (AUTO_FIX). The pixel fixes (PIXEL_FIXES,
silent, owner 2026-10-25) run on every card, in this order: badges laid out at older widths laid out again
(relay_badges), straight apostrophes, icon font glyphs Photoshop drew with an older font drawn again (never one whose
advance moved: relay_badges'), badges and icons fitted to their text (fit_badges: a badge's pill 1.35 times
the text's capital height, its middle on the capitals' middle, even when that rewraps the text; an FX layer on words
that moved is cut again), badges under an overlay set one-colour (one_colour_badges), timing tags set as badges and
capitalised status words in the chart's colours with their FX layers (status_words, owner 2026-11-07, the bake's own
steps); each redraws only the changed glyphs and the guards re-export the PNG. Every save strips the merged image
(Maximize Compatibility off, owner 2026-11-07: psd_edit.save_psd). The big pass's normalise (card_sweep alone until the
guards land its files, G1): sub and superscript text set normal at the same size and place (to_normal), the pictures
linked (the link step, psd_links), the merged image stripped, in one gated save. Clean (owner 2026-11-09, cards
only, silent): hidden star and type icon layers, hidden pictures that are an exact copy of a visible layer, and data
that draws nothing deleted, when every picture and the parse stay the same; with the guards' Hidden the hidden
leftovers too (_leftovers: adjustments and fills, 'Line if needed' and 'LINE' shapes, title copies and old words, empty
layers and groups, pictures shown on the card or linked files another PSD shows; never Ability text, stat digits, a
card with picture-only duplicates, nor a layer clipped onto a pass-through group: _isolating). After clean,
before the composer, the folders as the templates' (standard_folders, owner 2026-11-09, silent): names, colours,
order (an order alone too), loose layers and extra folders in their folder (Overlay Text and Overlay where Text, Stat
Text and Name or images would change their look), hidden Curves and nested star and type folders gone, at the first
FOLDER_LEVELS level whose card renders and reads as before (folders_gate), then linked icons in the middle of images
into their folder (mid_icons, gated alone; star icons that are art stay, untold: art_icons); no PNG follows. The sweep also isolates folders with adjustments clipped to them (isolate; the
guards do not).

    check(psd, rel, context, old=None, added=False) -> [Issue]
    fix(data, rel, issues, work, parser=None, fonts=None, only=None) -> Fixed   the fixes, gated (one PSD's new bytes)
    normalise(path, out_path, fonts, rel) -> Normalised           the big pass's rewrite (fix(only="normalise"))
    composer_layers(psd) -> [layer]                               the card text layers set Every-line (R8)
    redraw(path, out_path, names, fonts) -> [name]                the rewrapped layers drawn again (R8)
    standard_folders(path, out_path, fonts, rel, parser) -> Foldered  the templates' folders, gated
    folder_issues(psd, rel, parser) -> [str]                     what keeps the folders off the templates'
    BADGE_TAGS, WAIVED                                            the badge vocabulary and the owner's class waivers

No Discord, network or config.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import sha1
from io import BytesIO
from os import makedirs, remove, replace as move_file
from os.path import basename, getsize, join, splitext
from re import DOTALL, IGNORECASE, compile as re_compile, escape, fullmatch, search, sub
from shutil import rmtree
from tempfile import mkdtemp
from typing import Callable, Optional

import numpy as np
from psd_tools import PSDImage
from psd_tools.constants import BlendMode, Resource, SheetColorType, Tag

import commands.psd_edit as psd_edit
from commands.card_edits import CardAuthors
from commands.card_models import APOSTROPHE, straight
from commands.font_check import _get, _value, layer_fonts
from commands.psd_edit import BADGE_KEYS, BADGE_TAGS, ICON_FONT, Refused, badge_spans, badge_tags
from commands.psd_parser import (EFFECT_LAYER_TAG, ICON_FONT_PREFIX, TAG, CardValidator, PSDParser, base_name, layer_tag,
                                 picture_numbers, shown_in, star_icon_name, variant_view, variants_of)
from commands.psd_regions import (STATS as STAT_PATTERNS, bar_box, clip, comp_bars, comp_stat, comp_value, overlay_words,
                                  stat_rows, union_box)

# ========================
# Configuration
# ========================
SCOPE = ("creature", "aux creature", "minion", "item", "aux item", "nme")   # the ledger's 651
STARS_KINDS = ("minion", "aux item", "nme")     # stars drawn from a Stars group (L14)
EXACT = {"ability": "Ability", "name": "Name"}  # R11: Text/Ability, Text/Name
FIXABLE = ("R9", "R11")                         # 0-pixel by construction, gated: what fix() can do
AUTO_FIX = ("R9",)                              # what the bot fixes: the owner's "easy fixes only (today only R9 Txt2
                                                # sync)", 2026-10-22; R11 is told until the owner adds it
SIZE_MAX = 0.01                                 # a fixed PSD stays within 1% of the pushed one
COMPOSER_LAYERS = ("ability", "name", "health", "attack", "defense", "speed")   # R8: visible ones set Single-line
TXT2_COMPOSER = psd_edit.COMPOSER_KEY           # Txt2's EveryLineComposer (Photoshop CC's numbered keys)
LINE_IOU = 0.9                                  # R8 box text: a line rewraps under it (each line's ink, the fork's
                                                # Single-line layout vs the stored; matches 0.94-1.0, a moved word 0.04-0.34)
LINE_PAD = 2                                    # px around a line's box
NAME_BLOCK = bytes(Tag.UNICODE_LAYER_NAME)      # luni
WAIVED = {                                      # the owner's class waivers (section 6.2): rule -> what and why
    "R2": {"cards": {"tainted lazarus"}, "why": "its header icon (2026-10-22)"},
    "R3": {"cards": {"lit bob-omb", "tainted lazarus", "miracle matter", "twin emperors", "warcraftian druid", "pix",
                     "bulborb larva", "butterfly", "empress bulblax", "apex predator"},     # relaid in pieces (11-09)
           "why": "several-Ability and multi-creature cards stay deviations (question 2, 2026-10-21)"},
    "R4": {"why": "existing Name layers are left as they are (owner, 2026-10-22); new cards and regressions are reported"},
    "R5": {"why": "overlay words with no chart style stay (decision 2026-10-21, L7)"},
    "R15": {"why": "text already off the card's edge is left as it is (as R4); new cards and regressions are reported"},
}


# ========================
# Data
# ========================
@dataclass
class Issue:
    """One standards finding on one PSD."""
    code: str                                   # R1..R15
    text: str                                   # for card makers: "the Name layer is hidden"
    items: tuple = ()                           # what it is about (tags, layer names): "(new)" compares these
    fix: bool = False                           # AUTO_FIX and fixable here
    waived: str = ""                            # the waiver's reason
    new: bool = False                           # absent from the old blob
    discord: bool = True                        # told in the standards post; False: comment only

    def line(self) -> str:
        return self.text + (" (new)" if self.new else "")

    def as_dict(self) -> dict:
        return {"code": self.code, "text": self.text, "items": list(self.items), "fix": self.fix, "waived": self.waived,
                "new": self.new}


@dataclass
class Context:
    """What the rules need beside the PSD."""
    types: frozenset = frozenset()              # Types/ stems, lowercased (card_check.Vocabulary.all_types)
    chart: Optional[dict] = None                # psd_edit.status_fills of the status chart (R5 reports every overlay without)
    story_sync: bool = False                    # the fork can sync Txt2 stories (R9 is a fix, else a report)
    pngs: Optional[frozenset] = None            # the tree's PNG paths, lowercased (R12's pictures); None: not checked
    cards: Optional[tuple] = None               # ((title, key, kind), ...) of the shown cards (R14); None: not checked
    art: Optional[dict] = None                  # {linked file, lowercased: ((PSD path, blob), ...)} of the PSDs linking it
                                                # (db/links.json): where clean's hidden copies may show; None: unknown


@dataclass
class Fixed:
    """fix()'s result for one PSD."""
    data: bytes = b""                           # the fixed PSD ("" when nothing was fixed)
    applied: list = field(default_factory=list) # Issue lines fixed
    refused: list = field(default_factory=list) # (code, why) a gate refused
    composer: list = field(default_factory=list)   # R8: layers set Single-line (never told)
    composer_left: list = field(default_factory=list)   # R8: (layer, why) not redrawn or not set (plan.json only)
    rewrapped: list = field(default_factory=list)   # R8: box text layers redrawn in data (their lines moved)
    plain: bytes = b""                          # data without the redraw (when the PNG can't follow it)
    redrawn: dict = field(default_factory=dict) # PIXEL_FIXES kind -> layers redrawn in data (silent; the PNG follows)
    pixel_left: list = field(default_factory=list)  # (kind, why) a pixel fix not made (plan.json only)
    notes: list = field(default_factory=list)   # what a pixel fix did that a reader should look at (plan.json only)
    stale_glyphs: list = field(default_factory=list)    # layers old glyphs found in another font build (told)
    normalised: list = field(default_factory=list)  # normalise: what it did (silent; its redrawn layers in redrawn)
    files: dict = field(default_factory=dict)   # normalise: other files it writes (repo path -> bytes; None deletes)
    left: list = field(default_factory=list)    # normalise: what it left as it was, and why (plan.json only)
    cleaned: list = field(default_factory=list) # clean: what it deleted (silent; every picture the same: no PNG)
    folders: list = field(default_factory=list) # standard_folders: what it did (silent, no pixel changes)


# ========================
# Helper Functions
# ========================
def kind_of(rel: str) -> str:
    """The card's class from its repo path."""
    for prefix, kind in (("Auxiliary/Status Effects/", "status"), ("Auxiliary/Minions/", "minion"),
                         ("Auxiliary/Items/", "aux item"), ("Auxiliary/Creatures/", "aux creature"), ("N.M.E/", "nme"),
                         ("Field/", "field"), ("Items/", "item"), ("Creatures/", "creature")):
        if rel.startswith(prefix):
            return kind
    return "other"


def card_key(rel: str) -> str:
    """The card's key as Card Authors.txt and Card Edits.json match it."""
    stem = basename(rel)
    return CardAuthors.key(stem[:len(stem) - 4])


def _visible_texts(psd: PSDImage) -> list:
    return [layer for layer in psd.descendants() if layer.kind == "type" and layer.is_visible()]


def run_fonts(layer) -> tuple:
    """(text, [(start, end, PostScript font)]) in str indices over the layer's engine text."""
    resources, engine = layer.resource_dict, layer.engine_dict
    font_set = [str(_value(_get(font, "Name", ""))) for font in _get(resources, "FontSet", ())]
    text = str(_value(engine["Editor"]["Text"]))
    encoded = text.encode("utf-16-le")
    out, unit, pos = [], 0, 0
    runs = _get(engine, "StyleRun")
    for data, length in zip(_get(runs, "RunArray", ()), _get(runs, "RunLengthArray", ())):
        n = int(_value(length))
        piece = encoded[2 * unit:2 * (unit + n)].decode("utf-16-le", "ignore")
        sheet = _get(_get(data, "StyleSheet"), "StyleSheetData")
        index = int(_value(sheet.get("Font"))) if sheet is not None and sheet.get("Font") is not None else 0
        out.append((pos, pos + len(piece), font_set[index] if 0 <= index < len(font_set) else "?"))
        pos += len(piece)
        unit += n
    return text, out


def every_line(layer) -> bool:
    """Whether any paragraph of the layer uses the Every-line Composer."""
    try:
        for run in layer.engine_dict["ParagraphRun"]["RunArray"]:
            value = run["ParagraphSheet"]["Properties"].get("EveryLineComposer")
            if value is not None and bool(getattr(value, "value", value)):
                return True
    except (KeyError, TypeError, AttributeError):
        pass
    return False


def _on(value) -> bool:
    return bool(getattr(value, "value", value))


_story_props = psd_edit.story_paragraphs
_point_text = psd_edit.point_text


def _composer_values(psd: PSDImage, records: set) -> dict:
    """{(block, index, run): (properties, key)} of the type layers' composer values at these record indices: TySh by
    record, Txt2 by story."""
    out = {}
    for layer in psd.descendants():
        record = psd_edit._record_index(layer) if layer.kind == "type" else None
        if record not in records:
            continue
        for run, data in enumerate(layer.engine_dict["ParagraphRun"]["RunArray"]):
            props = data["ParagraphSheet"]["Properties"]
            if "EveryLineComposer" in props:
                out[("TySh", record, run)] = (props, "EveryLineComposer")
        index = psd_edit.story_index(layer)
        for run, props in enumerate(_story_props(psd, index)):
            if TXT2_COMPOSER in props:
                out[("Txt2", index, run)] = (props, TXT2_COMPOSER)
    return out


def composer_layers(psd: PSDImage) -> list:
    """R8: the visible card text layers (COMPOSER_LAYERS) set with the Every-line Composer, in TySh or their Txt2 story
    (box text without the story's key too: psd_edit.unset_composer)."""
    out = []
    for layer in _visible_texts(psd):
        if layer.name.strip().lower() not in COMPOSER_LAYERS:
            continue
        try:
            story = _story_props(psd, psd_edit.story_index(layer))
            unset = bool(psd_edit.unset_composer(psd, layer))
        except (Refused, KeyError, TypeError, AttributeError):
            story, unset = [], False
        if every_line(layer) or unset or any(_on(props.get(TXT2_COMPOSER, False)) for props in story):
            out.append(layer)
    return out


def _ink(rgba, bbox, shape: tuple):
    import numpy as np
    out = np.zeros(shape, bool)
    if rgba is None or bbox is None:
        return out
    x0, y0, x1, y1 = clip(bbox, shape[1], shape[0])
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = rgba[y0 - bbox[1]:y1 - bbox[1], x0 - bbox[0]:x1 - bbox[0], 3] > psd_edit.INK_ALPHA
    return out


def reflow(layer, shape: tuple, fonts_dir: Optional[str]) -> Optional[str]:
    """
    R8: which line moves when the layer is set Single-line, or None. Point text cannot; box text is laid out by the fork
    (its composer is Single-line's, greedy) and each line's ink is held to the stored drawing's (LINE_IOU). Raises
    ValueError when it can't be told (no fonts).
    """
    if _point_text(layer):
        return None
    if not fonts_dir:
        raise ValueError("box text, no fonts to lay it out")
    from psd_tools.composite.text import render, text_geometry
    from psd_tools.composite.text.verify import stored_raster
    fonts = psd_edit.font_map(fonts_dir, layer)         # in the faces its pixels show (R3)
    drawn, lines = render(layer, fonts=fonts), text_geometry(layer, fonts)["lines"]
    mine, saved = _ink(drawn.rgba, drawn.bbox, shape), _ink(*stored_raster(layer), shape)
    for line in lines:
        if not line["box"] or not line["text"].strip():
            continue
        x0, y0, x1, y1 = (int(round(v)) for v in line["box"])
        x0, y0, x1, y1 = clip((x0 - LINE_PAD, y0 - LINE_PAD, x1 + LINE_PAD, y1 + LINE_PAD), shape[1], shape[0])
        a, b = mine[y0:y1, x0:x1], saved[y0:y1, x0:x1]
        union = int((a | b).sum())
        alike = int((a & b).sum()) / union if union else 1.0
        if alike < LINE_IOU:
            return f"its line {line['text'].strip()[:30]!r} would move ({alike:.0%} alike)"
    return None


def single_line(path: str, out_path: str, fonts: Optional[Callable[[list], Optional[str]]] = None) -> tuple:
    """
    R8: set every composer_layers Single-line (TySh and the Txt2 story, the story's key written where box text has
    none) and save without a new preview; gated: set back and the written keys taken out, every layer record and
    document block is the old bytes. Each box text layer's lines are measured (reflow) for redraw().

    Args:
        fonts: Gives MDW/Fonts for a box text layer's faces (PostScript names); "" when they can't be had

    Returns:
        ([layer names set], [(layer name, the line that moves)], [(layer name, why not measured)]); raises
        Refused("bytes")
    """
    psd = PSDImage.open(path)
    chosen = composer_layers(psd)
    if not chosen:
        return [], [], []
    shape, moves, left = (psd.height, psd.width), [], []
    for layer in chosen:
        try:
            fonts_dir = None if _point_text(layer) else ((fonts(layer_fonts(layer)) if fonts else "") or "")
            why = reflow(layer, shape, fonts_dir)
        except Exception as error:              # a render or font failure: set, not redrawn, never told
            left.append((layer.name.strip(), f"{type(error).__name__}: {error}"))
            continue
        if why:
            moves.append((layer.name.strip(), why))
    records = {psd_edit._record_index(layer) for layer in chosen}
    flipped = [key for key, (props, name) in _composer_values(psd, records).items() if _on(props[name])]
    for key, (props, name) in _composer_values(psd, records).items():
        if key in flipped:
            props[name].value = False
    written = [("Txt2", psd_edit.story_index(layer), run) for layer in chosen for run in psd_edit.set_single_line(psd, layer)]
    psd_edit.save_psd(psd, out_path)
    new = PSDImage.open(out_path)
    if any(psd_edit._record_index(layer) in records for layer in composer_layers(new)):
        raise Refused("bytes", "a layer is still Every-line")
    values = _composer_values(new, records)
    if set(flipped + written) - set(values):
        raise Refused("bytes", "a composer value went missing")
    for key in flipped:
        props, name = values[key]
        props[name].value = True
    for key in written:
        props, name = values[key]
        del props[name]
    if psd_edit.layer_records(path) != psd_edit.layer_records(new):
        raise Refused("bytes", "a layer record changed beyond its composer")
    part = _same_document(PSDImage.open(path), new)
    if part:
        raise Refused("bytes", f"{part} changed beyond the composer")
    return [layer.name.strip() for layer in chosen], moves, left


def redraw(path: str, out_path: str, names: list, fonts: Callable[[list], Optional[str]]) -> list:
    """
    R8: the named text layers (lines that rewrap Single-line) drawn again whole by the fork and saved stripped
    (psd_edit.save_psd), so the guards' export can follow (png_export.export); gated: only those records' pixels,
    rectangle, TySh and stories change.

    Args:
        fonts: single_line()'s

    Returns:
        The layer names redrawn; raises Refused("bytes")
    """
    psd = PSDImage.open(path)
    layers = [layer for layer in _visible_texts(psd) if layer.name.strip() in names]
    if not layers:
        return []
    for layer in layers:
        layer.rasterize(fonts=psd_edit.font_map(fonts(layer_fonts(layer)), layer))
    psd_edit.save_psd(psd, out_path)
    _only_pixels(path, out_path, {psd_edit._record_index(layer) for layer in layers},
                 [psd_edit.story_index(layer) for layer in layers])
    return [layer.name.strip() for layer in layers]


def _unique(items) -> tuple:
    return tuple(dict.fromkeys(items))


def _list(items, quote: bool = False) -> str:
    items = [f'"{item}"' if quote else item for item in items]
    return items[0] if len(items) == 1 else ", ".join(items[:len(items) - 1]) + " and " + items[len(items) - 1]


# ========================
# The rules
# ========================
def ability_layers(psd: PSDImage) -> list:
    """The layers the parser reads as the card's ability, in every picture: psd_parts' selection (ability_parts: the
    visible Ability layers, else the pieces laying a hidden one out again, Empress Bulblax's "text 4")."""
    texts = [layer for layer in psd.descendants() if psd_edit._is_text_layer(layer)]
    pictures = variants_of(psd)
    out: dict = {}
    for n in (pictures if len(pictures) > 1 else [None]):
        with variant_view(psd, n):
            for layer in PSDParser.ability_parts(texts)[0]:
                out.setdefault(id(layer), layer)
    return list(out.values())


def plain_tags(layer) -> list:
    """The layer's timing tags and costs (psd_edit.badge_spans: "(1CD)", "(2 points)") with a letter outside the icon
    font: [(start, end)] into its text."""
    try:
        text, runs = run_fonts(layer)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return []
    out = []
    for a, b in badge_spans(text):
        fonts = {font for start, end, font in runs if start < b and end > a}
        if not fonts or any(not font.lower().startswith(ICON_FONT_PREFIX) for font in fonts):
            out.append((a, b))
    return out


def tag_layers(psd: PSDImage) -> list:
    """The layers R1 reads (ability_layers; a visible "Ability " or " Ability" layer too) holding plain_tags."""
    layers = {id(layer): layer for layer in ability_layers(psd)}
    for layer in _visible_texts(psd):
        if base_name(layer).strip().lower() == "ability":
            layers.setdefault(id(layer), layer)
    return [layer for layer in layers.values() if plain_tags(layer)]


def rule_tags(psd: PSDImage) -> Optional[Issue]:
    """R1: timing tags in the ability's layers (ability_layers) set in the stoogedings badge font (L1); a visible
    "Ability " or " Ability" layer too (the parser reads only "Ability")."""
    bad = [layer.text[a:b] for layer in tag_layers(psd) for a, b in plain_tags(layer)]
    if not bad:
        return None
    tags = _unique(bad)
    what = "cost" if all(" " in t for t in tags) else "timing tag" if not any(" " in t for t in tags) else "badge"
    return Issue("R1", f"{what}{'s' if len(tags) > 1 else ''} {_list(tags)} not set in the stoogedings badge font", tags)


def rule_icons(psd: PSDImage, types) -> Optional[Issue]:
    """R2: no Types/ icon image layers below the midline (the parser's inline-icon rule; stat bars skipped)."""
    mid = int(psd.height * 0.5)
    found = []
    for layer in psd.descendants():
        if layer.kind in ("type", "group") or EFFECT_LAYER_TAG in layer.name:
            continue
        if (layer.name.lower() in types and layer.is_visible() and layer.has_pixels()
                and not PSDParser._is_stat_layer(None, layer) and layer.bbox[1] >= mid):
            found.append(layer.name.strip().lower())
    if not found:
        return None
    names = _unique(found)
    return Issue("R2", f"icon image layer{'s' if len(names) > 1 else ''} in the ability ({_list(names)}): use "
                       f"{_list(['[' + n + ']' for n in names])} in the stoogedings font", names)


def _words(layer) -> list:
    """A text layer's words, split as PSDParser._relaid splits them."""
    return (layer.text or "").replace("\x03", " ").replace("\ufeff", "").split()


def rule_ability(psd: PSDImage) -> Optional[Issue]:
    """
    R3: one visible Ability text layer, in any group (the standard; a bake takes several too). Several are told; none only
    when a hidden Ability layer holds words the parser reads nowhere. An empty one (a card with no ability), one laid
    out again in visible pieces (PSDParser._relaid) or none at all (the ability named otherwise) is the developer's.
    """
    named = [layer for layer in psd.descendants()
             if psd_edit._is_text_layer(layer) and base_name(layer).strip().lower() == "ability"]
    count = sum(1 for layer in named if layer.is_visible())
    if count > 1:
        return Issue("R3", f"not standardized yet: {count} ability layers (a box each on the website)",
                     (f"{count} visible Ability layers",))
    if count or not any(_words(layer) for layer in named):
        return None
    texts = [layer for layer in psd.descendants() if psd_edit._is_text_layer(layer) and EFFECT_LAYER_TAG not in layer.name]
    if any(PSDParser._relaid(layer, texts) for layer in named):
        return None
    return Issue("R3", "the Ability layer is hidden (the bot reads no ability)", ("hidden ability layer",))


def title_problem(psd: PSDImage, card: str) -> Optional[str]:
    """
    Why no title can be read, when a Name layer is the cause: "hidden name layer" or "rasterized name layer". None when
    the parser reads a title (PSDParser.title_layers), or when no layer is named Name (the title's layout is the
    developer's, owner 2026-10-23).
    """
    if PSDParser.title_layers(psd, [layer for layer in psd.descendants() if psd_edit._is_text_layer(layer)], card):
        return None
    named = [layer for layer in psd.descendants() if not layer.is_group() and base_name(layer).strip().lower() == "name"]
    if not named or any(psd_edit._is_text_layer(layer) and layer.is_visible() for layer in named):
        return None
    if any(not psd_edit._is_text_layer(layer) for layer in named):
        return "rasterized name layer"
    return "hidden name layer"


def rule_name(psd: PSDImage, card: str) -> Optional[Issue]:
    """R4: the title is not left in a hidden or rasterized Name layer (no title read). A title split over several
    layers is the standard's too (owner 2026-11-08: every part named Name; R11 names the others)."""
    why = title_problem(psd, card)
    if why is None:
        return None
    return Issue("R4", {"hidden name layer": "the Name layer is hidden",
                        "rasterized name layer": "the Name layer is rasterized"}[why], (why,))


def rule_overlays(psd: PSDImage, chart: Optional[dict]) -> list:
    """R5: status words styled by Ability FX: layers, not overlay text layers (L7); chart-less ones are waived."""
    report, waived = [], []
    for layer in overlay_words(psd):
        word = psd_edit.collapse(layer.text)
        styled = chart is None or getattr(chart.get(word.lower()), "styled", False)
        (report if styled else waived).append(word)
    out = []
    if report:
        words = _unique(report)
        out.append(Issue("R5", f"{_list(words, quote=True)} {'is an overlay text layer' if len(words) == 1 else 'are overlay text layers'}, "
                               f"not an Ability FX: layer", words))
    if waived:
        words = _unique(waived)
        out.append(Issue("R5", f"overlay text {_list(words, quote=True)} (no chart style)", words, waived=WAIVED["R5"]["why"]))
    return out


def rule_stars(psd: PSDImage) -> Optional[Issue]:
    """R6: minions, auxiliary items and N.M.E. items draw their stars from a visible group named ... stars (L14)."""
    if PSDParser._star_groups(psd):
        return None
    return Issue("R6", "the stars are not in a group named Stars", ("no stars group",))


def rule_stories(psd: PSDImage, story_sync: bool) -> Optional[Issue]:
    """R9: every type layer's Txt2 story holds its text (Photoshop shows the story's)."""
    stale = psd_edit.stale_stories(psd)
    if not stale:
        return None
    names = _unique(name for name, _, _ in stale)
    return Issue("R9", f"Photoshop would show old text in {_list(names)} (Txt2 story out of date)", names,
                 fix=story_sync and "R9" in AUTO_FIX)


def inexact_names(psd: PSDImage) -> list:
    """R11's renames: [(layer, exact name)] for Text/Ability, Text/Name and the FX layers keyed on an inexact base."""
    group = psd_edit._text_group(psd)
    out, bases = [], {}
    for layer in (group or []):
        base, tags = layer_tag(layer.name)
        target = EXACT.get(base.strip().lower())
        suffix = layer.name[len(base):] if tags is not None else ""      # a variant tag stays: "Name #2"
        if layer.kind == "type" and target and base != target:
            out.append((layer, target + suffix))
            bases[layer.name] = target + suffix
    for layer in psd.descendants():
        base, tag, rest = layer.name.partition(EFFECT_LAYER_TAG)
        if tag and base in bases:
            out.append((layer, bases[base] + tag + rest))
    return out


def rule_part_names(psd: PSDImage, card: str) -> Optional[Issue]:
    """R11's suggestion: a title or ability part (a box on the website) whose layer is not named Name or Ability."""
    texts = [layer for layer in psd.descendants() if psd_edit._is_text_layer(layer)]
    titles = PSDParser.title_parts(psd, texts, card.replace("_", " "))[0]
    abilities = PSDParser.ability_parts(texts)[0]
    tagged = lambda layer, target: target + (layer.name[len(base_name(layer)):] if layer_tag(layer.name)[1] else "")
    pairs = tuple(f"{layer.name!r} -> {tagged(layer, target)!r}" for layers, target in ((titles, "Name"), (abilities, "Ability"))
                  for layer in layers if base_name(layer).strip().lower() != target.lower())   # case: rule_names'
    if not pairs:
        return None
    return Issue("R11", f"{'a part is' if len(pairs) == 1 else 'parts are'} not named so (the website labels "
                        f"{'it' if len(pairs) == 1 else 'them'}): rename {', '.join(pairs)}", pairs, discord=False)


def title_renames(psd: PSDImage, card: str) -> list:
    """R11's title renames (owner 2026-11-08: a split title's parts all named Name): [(layer, name)] for each title
    part (PSDParser.title_parts, in every picture) not named Name, its variant tag kept, and the FX layers on it."""
    texts = [layer for layer in psd.descendants() if psd_edit._is_text_layer(layer) and EFFECT_LAYER_TAG not in layer.name]
    pictures = variants_of(psd)
    found: dict = {}
    for n in (pictures if len(pictures) > 1 else [None]):
        with variant_view(psd, n):
            for layer in PSDParser.title_parts(psd, texts, card.replace("_", " "))[0]:
                base, tags = layer_tag(layer.name)
                if base.strip() != "Name" and id(layer) not in found:
                    found[id(layer)] = (layer, "Name" + (layer.name[len(base):] if tags is not None else ""))
    out = list(found.values())
    for layer, target in list(out):             # its FX layers, as the parser pairs them
        for fx in psd_edit.fx_layers(psd, layer):
            out.append((fx, target + EFFECT_LAYER_TAG + fx.name.partition(EFFECT_LAYER_TAG)[2]))
    return out


def rename_titles(path: str, out_path: str, card: str) -> list:
    """title_renames saved without a new preview; gated as rename(): only the renamed records' names change."""
    psd = PSDImage.open(path)
    return _renamed(psd, path, out_path, title_renames(psd, card))


def rule_variants(psd: PSDImage, rel: str, pngs: Optional[frozenset]) -> list:
    """
    R12, variant tags (' #2', ' #2,3': PARTS_EDITOR.md §9): a #0, an FX layer's (ignored), a tag past the pictures, a
    picture without its PNG, a saved eye that is not variant 1's. pngs None: the PNG checks are skipped.
    """
    out = []
    fx = _unique(layer.name for layer in psd.descendants() if EFFECT_LAYER_TAG in layer.name and TAG.match(layer.name))
    if fx:
        out.append(Issue("R12", f"{_list(fx, quote=True)} {'looks' if len(fx) == 1 else 'look'} like a variant tag; FX layers "
                                f"are never tagged (ignored)", fx, discord=False))
    tagged = [(layer, layer_tag(layer.name)[1]) for layer in psd.descendants() if layer_tag(layer.name)[1] is not None]
    if not tagged:
        return out
    zero = _unique(layer.name for layer, tags in tagged if 0 in tags)
    if zero:
        out.append(Issue("R12", f"{_list(zero, quote=True)}: #0 is no picture (variant 1 is Name.png)", zero))
    eyes = _unique(layer.name for layer, tags in tagged if bool(layer.visible) != (1 in tags))
    if eyes:
        out.append(Issue("R12", f"saved showing another picture than variant 1: {_list(eyes, quote=True)}", eyes))
    if pngs is not None:
        stem = rel[:len(rel) - 4]
        have = [n for n in variants_of(psd) if f"{stem}{'' if n == 1 else n}.png".lower() in pngs]
        folder = stem[:stem.rfind("/") + 1].lower()
        others = [p for p in pngs if p.startswith(folder) and "/" not in p[len(folder):] and not p[len(p) - 5].isdigit()]
        have += sorted(set(picture_numbers(psd, others).values()) - set(have))   # a drawn duplicate's (owner 11-09)
        missing = tuple(f"{basename(stem)}{'' if n == 1 else n}.png" for n in variants_of(psd) if n and n not in have)
        if missing:
            out.append(Issue("R12", f"tagged pictures without their PNG: {_list(missing)}", missing))
        past = _unique(layer.name for layer, tags in tagged if any(n > len(have) + 1 for n in tags))
        if past:
            out.append(Issue("R12", f"{_list(past, quote=True)}: a tag past the card's pictures ({len(have)} PNGs)", past))
    return out


def rule_layer_ids(psd: PSDImage) -> Optional[Issue]:
    """R13: every layer has its own id (the website's boxes are keyed by it)."""
    seen, twice = set(), []
    for layer in psd.descendants():
        if layer.layer_id < 0:
            continue                            # no id stored (a PSD made by psd-tools): nothing to key
        if layer.layer_id in seen:
            twice.append(layer.layer_id)
        seen.add(layer.layer_id)
    if not twice:
        return None
    ids = _unique(str(i) for i in twice)
    return Issue("R13", f"layers share an id ({', '.join(ids)}): Photoshop's Duplicate Layer gives a new one", ids)


def rule_names(psd: PSDImage) -> Optional[Issue]:
    """R11: Text/Ability, Text/Name and their FX layers named exactly so (case, outer spaces)."""
    renames = inexact_names(psd)
    if not renames:
        return None
    pairs = tuple(f"{layer.name!r} -> {target!r}" for layer, target in renames)
    return Issue("R11", f"layer names to correct: {', '.join(pairs)}", pairs, fix="R11" in AUTO_FIX)


# ========================
# Public API Functions
# ========================
# R14 (owner 2026-10-30, report only): a card named in the ability text in another letter case (while TTSCardMaker
# text.js EXACT_CASE is on, the website doesn't link it); which words name a card is text.js makeRenderer's rule
# (build_site.name_links ports it too): any name of several words, a capitalised one whatever comes before it, a
# lowercase one after its verb
_STATUS_VERB = re_compile(r"\b(?:inflict|cleanse|stacks? of)(?:[^\S\n]+[\dX]+T[\dX]+(?:\([^)\n]*\))?)?[^\S\n]+$", IGNORECASE)
NAME_VERB = {"buff": _STATUS_VERB, "debuff": _STATUS_VERB,
             "minion": re_compile(r"\b(?:summon|resummon|revive)(?:[^\S\n]+(?:a|an|the|\d+))?[^\S\n]+$", IGNORECASE)}


def _fold(s) -> str:
    return " ".join(str(s or "").replace("\u2018", "'").replace("\u2019", "'").split()).lower()


def _same(s) -> str:
    return " ".join(str(s or "").replace("\u2018", "'").replace("\u2019", "'").split())


@lru_cache(maxsize=2)
def _name_index(cards: tuple) -> tuple:
    """(names regex, {fold: (title, key, kind)} first come, {fold: the spellings in their own case}) of cards, as text.js
    adds them: titles, keys, titles without a trailing !?., several-word titles' singulars."""
    by_name, spelt = {}, {}
    one_word = lambda s: len(str(s or "").split()) <= 1

    def add(n, card):
        k = _fold(n)
        if len(k) >= 3:
            by_name.setdefault(k, card)
            spelt.setdefault(k, set()).add(_same(n))
    for card in cards:
        add(card[0], card)
    for card in cards:
        if not one_word(card[1]) or one_word(card[0]):
            add(card[1], card)
    for card in cards:
        add(card[0].strip().rstrip("!?."), card)
    for card in cards:
        title = card[0].strip()
        if not one_word(title) and title[-1:].lower() == "s" and title[-2:-1].lower() not in ("s", "i", "u"):
            add(title[:-1], card)
    if not by_name:
        return None, by_name, spelt
    pat = lambda k: escape(k).replace("'", "['\u2018\u2019]").replace(r"\ ", "[ \t]+")
    names = re_compile(r"(?<![\w'\u2018\u2019])(" + "|".join(pat(k) for k in sorted(by_name, key=len, reverse=True)) + r")"
                       r"((?:['\u2018\u2019]s|(?<=s)['\u2018\u2019]|e?s)(?![\w'\u2018\u2019]))?(?![\w'\u2018\u2019])", IGNORECASE)
    return names, by_name, spelt


def _is_ref(found, text: str, card: tuple) -> bool:
    """text.js isRef: whether the matched words name the card (not a heading, label, verb or plain word)."""
    typed, spelt = found[1], [card[0].strip(), str(card[1] or "").strip()]
    if typed == typed.upper() and typed != typed.lower() and typed not in spelt:
        return False
    if len(typed.split()) > 1:
        return True
    after, before = text[found.end():], text[:found.start()]
    if after[:1] == ":" or re_compile(r"[^\S\n]+(?:\d|X\b)").match(after):
        return False
    if not (typed in spelt and not found[2]) and not typed[0].isupper():
        verb = NAME_VERB.get(card[2])
        return bool(verb and verb.search(before))
    return not (re_compile(r"(?:^|[\n.!?:)])[^\S\n]*$").search(before)
                and re_compile(r"[^\S\n]+(?:to|the|a|an|your|its|all|any|each|it|this|that)\b").match(after))


def miscased_names(text: str, cards: tuple) -> list:
    """[(typed, the card's spelling)]: the cards the text names in another letter case ("silithid" for Silithid), its own
    name too (owner 10-30: a card naming itself mentions itself); badges and [tokens] read as blanks."""
    names, by_name, spelt = _name_index(cards)
    if names is None:
        return []
    flat = spaced = text.replace("\r", "\n").replace("\x03", "\n")
    for a, b in psd_edit.icon_text(flat, None):
        flat, spaced = flat[:a] + "\0" * (b - a) + flat[b:], spaced[:a] + " " * (b - a) + spaced[b:]
    out = []
    for found in names.finditer(flat):
        card = by_name[_fold(found[1])]
        if _same(found[1]) in spelt[_fold(found[1])]:
            continue
        if _is_ref(found, spaced, card):
            out.append((found[1], card[0].strip()))
    return out


def _lowercase_card(texts: list, own: str) -> bool:
    """Written in lowercase on purpose (snas.): its own name and every sentence of its ability text start lowercase,
    badges and [tokens] read as blanks (owner 2026-11-05; a lowercase name alone, as hdckighfkvhvgmk's, is not)."""
    if own != own.lower() or not any(ch.isalpha() for ch in own):
        return False
    starts = []
    for text in texts:
        for a, b in reversed(psd_edit.icon_text(text, None)):
            text = text[:a] + " " * (b - a) + text[b:]
        starts += [m[1] for m in re_compile(r"(?:^|[\n\r.!?])[^A-Za-z\n\r.!?]*([A-Za-z])").finditer(text)]
    return bool(starts) and not any(ch.isupper() for ch in starts)


def rule_name_case(psd: PSDImage, cards: Optional[tuple], key: str = "") -> Optional[Issue]:
    """R14: a card named in the ability text in another letter case (report only); a card written in lowercase on
    purpose (_lowercase_card) has its lowercase mentions pass."""
    if not cards:
        return None
    texts = [layer.text or "" for layer in ability_layers(psd)]
    lower = _lowercase_card(texts, next((title for title, k, _ in cards if CardAuthors.key(k) == key), ""))
    found = list(dict.fromkeys(pair for text in texts for pair in miscased_names(text, cards)
                               if not (lower and pair[0] == pair[0].lower())))
    if not found:
        return None
    return Issue("R14", "card names typed in another letter case: "
                 + ", ".join(f"'{typed}' should be '{right}'" for typed, right in found), tuple(typed for typed, _ in found))


def _off_card(layer, width: int, height: int) -> bool:
    bbox = tuple(layer.bbox)
    if clip(bbox, width, height) == bbox or psd_edit._alpha(layer) is None:
        return False                            # inside, or no stored pixels to read
    return psd_edit._ink_off_canvas(layer, width, height)


def rule_off_card(psd: PSDImage, card: str) -> list:
    """R15: the ability's layers (ability_layers) and the title's (PSDParser.title_parts) ink inside the card in
    Photoshop's stored pixels (psd_edit._ink_off_canvas): text off its edge is cut in the PNG and TTS (report only)."""
    texts = [layer for layer in psd.descendants() if psd_edit._is_text_layer(layer)]
    layers = {id(layer): layer for layer in ability_layers(psd)}
    for layer in PSDParser.title_parts(psd, texts, card.replace("_", " "))[0]:
        layers.setdefault(id(layer), layer)
    return [Issue("R15", f"{layer.name.strip()}'s text runs off the card's edge (cut in the PNG and TTS)",
                  (layer.name.strip(),)) for layer in layers.values() if _off_card(layer, psd.width, psd.height)]


def check(psd: PSDImage, rel: str, context: Context, old: Optional[PSDImage] = None, added: bool = False) -> list:
    """
    The standards issues of one card PSD.

    Args:
        psd: The pushed PSD
        rel: Its repo path (the class: scope, waivers)
        context: Types/, the chart, the story sync
        old: The PSD before the push (R4 and R15 regressions; "(new)" marks)
        added: A new card (R4 and R15 apply in full)

    Returns:
        [Issue] in rule order, waived ones included (waived set)
    """
    kind = kind_of(rel)
    key = card_key(rel)
    stem = basename(rel)
    card = stem[:len(stem) - 4]
    out: list = []

    def run(code: str, rule, *args) -> None:
        try:
            found = rule(*args)
        except Exception as error:              # one unreadable part never stops the other rules
            found = Issue("error", f"{code} could not be checked ({type(error).__name__}: {error})", (code,), discord=False)
        out.extend(found if isinstance(found, list) else [found])

    run("R1", rule_tags, psd)
    run("R2", rule_icons, psd, context.types)
    if kind in SCOPE:
        run("R3", rule_ability, psd)
        run("R4", rule_name, psd, card)
        name = out[len(out) - 1]
        if name is not None and name.code == "R4" and not added:
            try:
                regression = old is not None and rule_name(old, card) is None
            except Exception:                   # an unreadable old blob: the existing card's waiver
                regression = False
            if not regression:
                name.waived = WAIVED["R4"]["why"]
    run("R5", rule_overlays, psd, context.chart)
    if kind in STARS_KINDS:
        run("R6", rule_stars, psd)
    run("R9", rule_stories, psd, context.story_sync)
    if kind in SCOPE:
        run("R11", rule_names, psd)
        run("R11", rule_part_names, psd, card)
    run("R12", rule_variants, psd, rel, context.pngs)
    run("R13", rule_layer_ids, psd)
    run("R14", rule_name_case, psd, context.cards, key)
    start = len(out)
    run("R15", rule_off_card, psd, card)
    off = [issue for issue in out[start:] if issue is not None and issue.code == "R15"]
    if off and not added:                       # as R4: waived unless the old blob had the layer inside the card
        try:
            had = None if old is None else {item for issue in rule_off_card(old, card) for item in issue.items}
        except Exception:                       # an unreadable old blob: the existing card's waiver
            had = None
        for issue in off:
            if had is None or set(issue.items) <= had:
                issue.waived = WAIVED["R15"]["why"]
    issues = [issue for issue in out if issue is not None]
    for issue in issues:
        waiver = WAIVED.get(issue.code, {})
        if not issue.waived and key in waiver.get("cards", ()):
            issue.waived = waiver["why"]
    if old is not None:
        mark_new(issues, check(old, rel, context))
    return issues


def mark_new(issues: list, before: list) -> None:
    """Set new on the issues with an item the old blob's issues of the same rule lack."""
    seen: dict = {}
    for issue in before:
        seen.setdefault(issue.code, set()).update(issue.items)
    for issue in issues:
        issue.new = bool(set(issue.items) - seen.get(issue.code, set()))


def reported(issues: list) -> list:
    """The issues told to the author: not waived, not fixed by the bot."""
    return [issue for issue in issues if not issue.waived and not issue.fix]


def _records_except_names(path: str) -> list:
    """layer_records with the name and luni left out (R11's byte proof)."""
    out = []
    for record in psd_edit.layer_records(path):
        out.append({**{k: v for k, v in record.items() if k != "name"},
                    "blocks": tuple((key, data) for key, data in record["blocks"] if key != NAME_BLOCK)})
    return out


def _same_document(old: PSDImage, new: PSDImage) -> Optional[str]:
    """The part of the document other than the layer records that changed (the merged image's strip is none), or None."""
    part = psd_edit.changed_part(old, new)
    if part:
        return part
    if psd_edit._document_blocks(old) != psd_edit._document_blocks(new):
        return "document blocks"
    return None


def rename(path: str, out_path: str) -> list:
    """
    R11: rename the inexact layers and save without a new preview; gated: only the renamed records' names change.

    Returns:
        The renames ("old -> new"); raises Refused("bytes")
    """
    psd = PSDImage.open(path)
    return _renamed(psd, path, out_path, inexact_names(psd))


def _renamed(psd: PSDImage, path: str, out_path: str, renames: list) -> list:
    """Rename [(layer, name)] and save without a new preview; gated: only the renamed records' names change."""
    if not renames:
        return []
    indices = set()
    records = psd._record.layer_and_mask_information.layer_info.layer_records
    for layer, target in renames:
        indices.add(next(i for i, record in enumerate(records) if record is layer._record))
        layer.name = target
    psd_edit.save_psd(psd, out_path)
    old_records, new_records = psd_edit.layer_records(path), psd_edit.layer_records(out_path)
    if len(old_records) != len(new_records):
        raise Refused("bytes", "layer count changed")
    names = [i for i, (a, b) in enumerate(zip(old_records, new_records)) if a["name"] != b["name"]]
    if set(names) != indices:
        raise Refused("bytes", f"renamed records {names}, expected {sorted(indices)}")
    if _records_except_names(path) != _records_except_names(out_path):
        raise Refused("bytes", "a layer record changed beyond its name")
    part = _same_document(PSDImage.open(path), PSDImage.open(out_path))
    if part:
        raise Refused("bytes", f"{part} changed")
    return [f"{a['name']!r} -> {b['name']!r}" for i, (a, b) in enumerate(zip(old_records, new_records)) if i in indices]


def _folders_in(data: bytes, rel: str, parser: Optional[PSDParser] = None) -> bool:
    try:
        return bool(folder_issues(PSDImage.open(BytesIO(data)), rel, parser))
    except Exception:                           # unreadable: nothing to arrange
        return False


def _every_line_in(data: bytes) -> bool:
    try:
        return bool(composer_layers(PSDImage.open(BytesIO(data))))
    except Exception:                           # unreadable: nothing to set
        return False


def _fix_in(folder: str, data: bytes, rel: str, wanted: list, parser: Optional[PSDParser], result: Fixed,
            fonts: Optional[Callable[[list], Optional[str]]] = None, composer: bool = True, pixels: tuple = (),
            whole: bool = False, folders: bool = False) -> None:
    """fix()'s work in its own temporary folder: the fixable issues (gated together), clean, then the folders (each
    gated on its own, silent), then the composer and the pixel fixes over them."""
    from commands.txt2_repair import repair
    pushed = join(folder, "pushed.psd")
    with open(pushed, "wb") as fh:
        fh.write(data)
    current, codes = pushed, []
    for issue in sorted(wanted, key=lambda i: FIXABLE.index(i.code)):
        out = join(folder, f"{issue.code}.psd")
        try:
            if issue.code == "R11":
                if not rename(current, out):
                    continue
            else:
                done = repair(current, out)
                if not done.out:
                    continue
        except (Refused, ImportError, ValueError) as error:
            result.refused.append((issue.code, str(error)))
            continue
        current = out
        codes.append(issue.code)
        result.applied.append(issue.line())
    if current != pushed:
        try:
            result.data = _gated(pushed, current, rel, parser)
        except Refused as refused:
            result.refused += [(code, str(refused)) for code in codes]
            result.applied, current = [], pushed
    base = current                              # what the composer and the pixel fixes start from
    if "clean" in pixels:                       # first of the silent ones: what follows works on what stays (11-09)
        out = join(folder, "clean.psd")
        try:
            names = clean(base, out, fonts, rel)
            if names:
                result.data = _gated(base, out, rel, parser, None, sized=False)
                base, result.cleaned = out, list(names)
        except Exception as error:              # best effort, never told: plan.json says why
            result.pixel_left.append(("clean", f"{type(error).__name__}: {error}"))
        pixels = tuple(kind for kind in pixels if kind != "clean")
    if folders:                                 # the templates' folders (owner 11-09): silent, no PNG
        out = join(folder, "folders.psd")
        try:
            done = standard_folders(base, out, fonts, rel, parser, isolated=False)
        except Exception as error:              # best effort, never told
            result.pixel_left.append(("folders", f"{type(error).__name__}: {error}"))
        else:
            try:                                # and the size gate: a folder adds a few records
                if done:
                    result.data = _gated(base, out, rel, parser)
                    base, result.folders = out, list(done)
            except Refused as refused:
                result.pixel_left.append(("folders", str(refused)))
    current = base
    out, moves = join(folder, "R8.psd"), []
    if composer:
        try:
            result.composer, moves, result.composer_left = single_line(current, out, fonts)
        except Exception as error:              # best effort: never told, the other fixes stand
            result.composer_left = [("", f"{type(error).__name__}: {error}")]
    if result.composer:
        current = out
        try:
            result.data = _gated(base, current, rel, parser)
        except Refused as refused:
            result.composer_left += [(name, str(refused)) for name in result.composer]
            result.composer, current, moves = [], base, []      # the pixel fixes go on what stood
    if moves:
        drawn = join(folder, "R8-redrawn.psd")
        try:                                    # the lines that rewrap drawn again: the guards re-export the PNG
            names = redraw(current, drawn, [name for name, _ in moves], fonts)
            data = _gated(base, drawn, rel, parser)
        except Exception as error:              # best effort: set, not redrawn
            result.composer_left += [(name, f"rewraps ({why}), not redrawn: {error}") for name, why in moves]
        else:
            result.plain, result.data, result.rewrapped = result.data, data, names
            current = drawn
    if pixels:
        _pixel_fixes(folder, base, current, rel, parser, result, fonts, pixels, whole)


def _gated(base: str, path: str, rel: str, parser: Optional[PSDParser], reads: Optional[Callable] = None,
           sized: bool = True) -> bytes:
    """A fixed PSD's bytes through the size and the reparse gates (reads: what the fix makes of the old parse; sized
    False: a pixel fix, its bytes proven by _only_pixels; the merged image's strip left out of the size: body_sizes);
    raises Refused."""
    was, now = psd_edit.body_sizes(base, path)
    if sized and abs(now - was) > SIZE_MAX * was:
        raise Refused("size", f"PSD {getsize(base)} -> {getsize(path)} bytes")
    if parser is not None:
        before, after = psd_edit.parse_card(parser, base, rel), psd_edit.parse_card(parser, path, rel)
        if reads is not None:
            before = reads(before)
        if (before["card"], before["validator"]) != (after["card"], after["validator"]):
            raise Refused("reparse", "the bot reads the fixed PSD differently")
    with open(path, "rb") as fh:
        return fh.read()


def fix(data: bytes, rel: str, issues: list, work: str, parser: Optional[PSDParser] = None,
        fonts: Optional[Callable[[list], Optional[str]]] = None, only: Optional[str] = None, whole: bool = False,
        skip: tuple = ()) -> Fixed:
    """
    The fixable issues of one PSD (R11 renames, then R9 story syncs), each gated, the folders (standard_folders,
    with a parser; skip "folders" to leave them), then R8's Single-line Composer
    (single_line: never told, a refusal left in composer_left); then the size and the reparse gates over the result. A
    refusal leaves that fix out (and reports it). Box text that rewraps is
    redrawn (redraw, gated again: data, the result without it in plain), then the pixel fixes (PIXEL_FIXES, each gated;
    redrawn, a refusal in pixel_left; plain without them).

    Args:
        data: The pushed PSD
        rel: Its repo path
        issues: check()'s issues
        work: A folder for the temporary files
        parser: The offline parser for gate_reparse (skipped without)
        fonts: Gives MDW/Fonts for R8's box text (single_line) and the pixel fixes; fonts.chart: the status chart's
            fills (status words)
        only: One PIXEL_FIXES kind alone (a sweep: card_sweep), no other fix
        whole: fit_badges may redraw a layer whole (a sweep's --whole)
        skip: PIXEL_FIXES kinds left out (the guards convert in a step of their own)

    Returns:
        Fixed
    """
    result = Fixed()
    wanted = [] if only else [issue for issue in issues if issue.fix and not issue.waived]
    composer = not only and _every_line_in(data)
    pixels = tuple(kind for kind in pixel_kinds(data, (only,) if only else (), rel, getattr(fonts, "chart", None),
                                                 getattr(fonts, "links", None), getattr(fonts, "hidden", None))
                   if kind not in skip)
    folders = not only and parser is not None and "folders" not in skip and _folders_in(data, rel, parser)
    if not (wanted or composer or pixels or folders):
        return result
    makedirs(work, exist_ok=True)
    folder = mkdtemp(prefix="fix-", dir=work)   # one per call: cards of one key in two folders never share a file
    try:
        _fix_in(folder, data, rel, wanted, parser, result, fonts, composer, pixels, whole, folders)
    finally:
        rmtree(folder, ignore_errors=True)
    return result


# ========================
# Pixel fixes (silent: the commit message alone says them, owner 2026-10-25)
# ========================
OLD_GLYPH_BAD = 0.40        # an icon font glyph is old past this share of its ink off the current font's render, shift-
                            # tolerant (census 2026-10-25: Photoshop's current glyphs 0.26 at most, older fonts' 0.43 and up)
OLD_GLYPH_LEVEL = 24 / 255  # a pixel is off past this in a premultiplied channel
OLD_GLYPH_SHIFT = 2         # px the render may sit off Photoshop's glyph
OLD_GLYPH_INK = 0.05        # alpha that counts as ink
OLD_GLYPH_MIN_INK = 20      # px of ink: a smaller glyph is not judged
STALE_FILL = (-1.0, -1.0, -1.0)   # marks a glyph changed for the splice (it redraws that glyph alone)
FIT_CHECK = 1.0             # px a fitted badge or icon may sit off its place after fit_badges (the check)
WHOLE_OFF = 0.01            # fit_badges(whole): the share of a layer's ink the fork's render may have off its stored pixels
LAID_TOL = 0.01             # pt: an icon glyph Photoshop stepped this far off the current font was laid out at another advance
EARLIER_FIT = OLD_GLYPH_BAD # badge widths: an icon glyph is drawn in an earlier icon font when that font's own render of the layer
EARLIER_GAIN = 0.03         # fits it this well and better by this than the current font's (owner 2026-11-04: Roboto labels
                            # move a badge's letters less than OLD_GLYPH_BAD; census of 376 badge cards: short labels such
                            # as Halberd Reactor's (BOT) gain 0.046, a glyph the fonts draw alike 0; review 2026-11-04: not
                            # 0.05, Soul Master's (OD) fits main's font 0.136 against the current 0.186)
RAZOR = 0.001               # badge widths: a re-laid layer whose breaks change in a box this share narrower or wider fits
                            # by a hair (Photoshop measures its lines a little otherwise: Croissant 0.015% wrapped)
DRAWN_ODD = 0.30            # badge widths: a moved icon glyph this far off the current font's render that no earlier font
                            # draws (past Photoshop's own current glyphs, 0.26 at most: OLD_GLYPH_BAD's census) is refused
LAID_OFF = 0.15             # badge widths: the share of a layer's ink (its re-laid glyphs aside) its render at the advances
                            # it was laid out with may have off its stored pixels (census 2026-11-04, 219 cooldown cards:
                            # 0.13 at most at their old widths, 0.19 and up at others)


def _straightened(value):
    """value's strings with straight apostrophes."""
    if isinstance(value, dict):
        return {key: _straightened(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_straightened(item) for item in value)
    return straight(value)


def straight_parse(parsed: dict) -> dict:
    """What straighten() makes the parser read: the parse straightened, a title now the card's name no display_name."""
    out = {key: _straightened(parsed[key]) for key in ("card", "validator")}
    title, name = out["card"].get("display_name"), getattr(parsed.get("info"), "name", None) or ""
    if title is not None and (title == name or sorted(title.lower().split()) == sorted(name.lower().split())):
        out["card"].pop("display_name")         # _printed_title's rules (the second for a title of several layers)
    return out


def curly_layers(psd: PSDImage) -> list:
    """The type layers whose text holds a curly apostrophe (hidden ones too)."""
    return [layer for layer in psd.descendants() if layer.kind == "type" and APOSTROPHE in (layer.text or "")]


def _icon_runs(layer) -> bool:
    """Whether a run of the layer is set in the icon font."""
    try:
        _, runs = run_fonts(layer)
    except (KeyError, TypeError, AttributeError, IndexError):
        return False
    return any(end > start and font.lower().startswith(ICON_FONT_PREFIX) for start, end, font in runs)


def icon_layers(psd: PSDImage) -> list:
    """The visible type layers with icon font runs."""
    return [layer for layer in _visible_texts(psd) if _icon_runs(layer)]


def _fonts_of(layer, fonts: Optional[Callable[[list], Optional[str]]]):
    folder = fonts(layer_fonts(layer)) if fonts else ""
    if not folder:
        raise ValueError(f"{layer.name.strip()}: no fonts to draw it")
    return psd_edit.font_map(folder, layer)


def _premultiplied(rgba, bbox, window: tuple):
    """Straight RGBA at bbox, premultiplied on the window (transparent elsewhere)."""
    import numpy as np
    x0, y0, x1, y1 = window
    out = np.zeros((y1 - y0, x1 - x0, 4), np.float32)
    if rgba is not None and bbox is not None and rgba.size:
        bx0, by0, bx1, by1 = (int(v) for v in bbox)
        cx0, cy0, cx1, cy1 = max(bx0, x0), max(by0, y0), min(bx1, x1), min(by1, y1)
        if cx1 > cx0 and cy1 > cy0:
            out[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = rgba[cy0 - by0:cy1 - by0, cx0 - bx0:cx1 - bx0]
    out[..., :3] *= out[..., 3:4]
    return out


def _off_share(stored, drawn, box: tuple) -> float:
    """
    The share of a glyph's ink (stored or drawn) off the render, the render shifted up to OLD_GLYPH_SHIFT px and each
    pixel matched to the nearest of its 3x3 (the oldfont census' score); 0.0 for too little ink. box: in window pixels,
    at least OLD_GLYPH_SHIFT + 1 inside it.
    """
    import numpy as np
    x0, y0, x1, y1 = box
    a = stored[y0:y1, x0:x1]
    m = OLD_GLYPH_SHIFT + 1
    big = drawn[y0 - m:y1 + m, x0 - m:x1 + m]
    h, w = a.shape[:2]
    best = None
    for dy in range(-OLD_GLYPH_SHIFT, OLD_GLYPH_SHIFT + 1):
        for dx in range(-OLD_GLYPH_SHIFT, OLD_GLYPH_SHIFT + 1):
            near = None
            for ny in (-1, 0, 1):
                for nx in (-1, 0, 1):
                    b = big[m + dy + ny:m + dy + ny + h, m + dx + nx:m + dx + nx + w]
                    off = np.abs(a - b).max(axis=2)
                    near = off if near is None else np.minimum(near, off)
            bad = int((near > OLD_GLYPH_LEVEL).sum())
            if best is None or bad < best[0]:
                best = (bad, dy, dx)
    _, dy, dx = best
    b = big[m + dy:m + dy + h, m + dx:m + dx + w]
    ink = int(((a[..., 3] > OLD_GLYPH_INK) | (b[..., 3] > OLD_GLYPH_INK)).sum())
    return best[0] / ink if ink >= OLD_GLYPH_MIN_INK else 0.0


def _icon_face(face) -> bool:
    name = face.ft.postscript_name
    name = name.decode("latin-1") if isinstance(name, bytes) else str(name or "")
    return name.lower().startswith(ICON_FONT_PREFIX)


def _drawn_old(layer, fonts) -> Optional[tuple]:
    """old_glyphs with what it compared: (capture, window, stored, drawn: premultiplied on the window, the old glyphs'
    indices); None without stored pixels."""
    found = _glyph_offs(layer, fonts)
    return None if found is None else (*found[:4], [i for i, share in found[4].items() if share > OLD_GLYPH_BAD])


def _glyph_offs(layer, fonts) -> Optional[tuple]:
    """(capture, window, stored, drawn, {icon glyph index: _off_share of its box}): the layer's stored pixels against the
    render in fonts, at the layout fonts give; None without stored pixels."""
    from psd_tools.composite.text import capture, render
    from psd_tools.composite.text.splice import _glyph_box
    before = capture(layer, fonts)
    if before.rgba is None or before.bbox is None:
        return None
    drawn = render(layer, fonts=fonts)
    pad = OLD_GLYPH_SHIFT + 2
    boxes = [b for b in (before.bbox, drawn.bbox) if b is not None]
    window = (min(b[0] for b in boxes) - pad, min(b[1] for b in boxes) - pad,
              max(b[2] for b in boxes) + pad, max(b[3] for b in boxes) + pad)
    stored, mine = _premultiplied(before.rgba, before.bbox, window), _premultiplied(drawn.rgba, drawn.bbox, window)
    out = {}
    for index, glyph in enumerate(before.layout.glyphs):
        if not _icon_face(glyph.face):
            continue
        box = _glyph_box(glyph, before.model.transform, 1)
        if box is None:
            continue
        x0, y0, x1, y1 = (box[0] - window[0], box[1] - window[1], box[2] - window[0], box[3] - window[1])
        x0, y0 = max(x0, pad), max(y0, pad)
        x1, y1 = min(x1, window[2] - window[0] - pad), min(y1, window[3] - window[1] - pad)
        if x1 > x0 and y1 > y0:
            out[index] = _off_share(stored, mine, (x0, y0, x1, y1))
    return before, window, stored, mine, out


def old_glyphs(layer, fonts) -> list:
    """
    The icon font glyphs Photoshop drew with an older version of the font: their stored pixels against the current
    font's render (_off_share past OLD_GLYPH_BAD).

    Args:
        fonts: A FontMap (psd_edit.font_map)

    Returns:
        Indices into the layout's glyphs (capture(layer, fonts).layout)
    """
    drawn = _drawn_old(layer, fonts)
    return drawn[4] if drawn else []


def _only_pixels(path: str, out_path: str, records: set, stories: Optional[list] = None, fonts: tuple = ()) -> None:
    """Gate: only these records' pixels, rectangle and TySh changed, and (stories) only these Txt2 stories and the faces
    added at the end of its FontSet (fonts), else the document is the old bytes but its merged image (stripped:
    psd_edit.changed_part); raises Refused("bytes")."""
    old_records, new_records = psd_edit.layer_records(path), psd_edit.layer_records(out_path)
    if len(old_records) != len(new_records):
        raise Refused("bytes", "layer count changed")
    for i, (a, b) in enumerate(zip(old_records, new_records)):
        if i not in records:
            if a != b:
                raise Refused("bytes", "a layer record changed beyond the redrawn text")
        elif {k for k in a if a[k] != b[k]} - {"channels", "rect", "blocks"} or \
                [k for (k, x), (_, y) in zip(a["blocks"], b["blocks"]) if x != y and k != b"TySh"]:
            raise Refused("bytes", f"{a['name']}: changed beyond its pixels")
    before, new = PSDImage.open(path), PSDImage.open(out_path)
    part = psd_edit.changed_part(before, new, ("header", "color_mode_data", "image_resources"))
    if part:
        raise Refused("bytes", f"{part} changed")
    if stories is None:
        if psd_edit._document_blocks(before) != psd_edit._document_blocks(new):
            raise Refused("bytes", "document blocks changed")
    else:
        psd_edit.gate_stories(before, new, stories, fonts=fonts)


def _by_glyph(layer, report, whole: bool = False) -> None:
    """Refuses a redraw the engine did not splice by glyph (its whole-layer fallback), unless whole allows it."""
    spliced = getattr(report, "splice", None)
    if not whole and (spliced is None or spliced.fallback):
        raise Refused("unsupported-card", f"{layer.name.strip()}: not redrawn by glyph"
                                          + (f" ({spliced.fallback})" if spliced is not None else ""))


def straighten(path: str, out_path: str, fonts: Optional[Callable[[list], Optional[str]]]) -> list:
    """
    Apostrophes (owner, 2026-10-25): every U+2019 in the type layers set as ', one at a time (each keeps its own style), in
    TySh and the Txt2 story; the changed glyphs redrawn (a glyph splice), the stored preview patched; gated
    (_only_pixels, the stories by gate_stories).

    Returns:
        The layer names changed; raises Refused or the engine's errors
    """
    from psd_tools.composite.text import capture
    psd = PSDImage.open(path)
    layers = curly_layers(psd)
    if not layers:
        return []
    stories = []
    for layer in layers:
        found = _fonts_of(layer, fonts)
        keep = capture(layer, found)
        if psd_edit.story_index(layer) is not None:
            stories.append(psd_edit.story_index(layer))
        text = layer.text
        for at in reversed([i for i, c in enumerate(text) if c == APOSTROPHE]):
            text = text[:at] + "'" + text[at + 1:]
            layer.set_text(text, keep="runs")
        _by_glyph(layer, layer.rasterize(fonts=found, keep=keep, unit=psd_edit.SPLICE_UNIT))
    psd_edit.save_psd(psd, out_path)
    _only_pixels(path, out_path, {psd_edit._record_index(layer) for layer in layers}, stories)
    if curly_layers(PSDImage.open(out_path)):
        raise Refused("bytes", "a curly apostrophe is left")
    return [layer.name.strip() for layer in layers]


def redraw_old_glyphs(path: str, out_path: str, fonts: Optional[Callable[[list], Optional[str]]],
                      seen: Optional[list] = None) -> list:
    """
    Icon font glyphs Photoshop drew with an older font (old_glyphs) drawn again with the current one, those glyphs alone
    (a splice; refused when it would redraw the layer whole), the stored preview patched; gated (_only_pixels). Refused
    when one of them has another advance in the layout it may have been laid out with (laid_advances: Photoshop's view,
    an earlier icon font's, fonts.earlier): the splice would set it at its new width in the old layout (badge widths,
    first, re-lays those).

    Args:
        seen: Gets each layer whose old glyphs show another build of the icon font (_another_build), before any
            refusal: the guards tell one they leave

    Returns:
        The layer names redrawn; raises Refused or the engine's errors
    """
    from dataclasses import replace
    from psd_tools.composite.text import capture
    earlier = tuple(getattr(fonts, "earlier", ()))
    psd = PSDImage.open(path)
    records, stories, names = set(), [], []
    for layer in icon_layers(psd):
        found = _fonts_of(layer, fonts)
        drawn = _drawn_old(layer, found)
        stale = drawn[4] if drawn else []
        if not stale:
            continue
        psd_edit.follow_kerning(psd, layer)     # B12: drawn as Photoshop kerns it (the story)
        keep = capture(layer, found)
        moved = {gid for way in laid_advances(layer, found, earlier) for gid in way}
        relaid = any(keep.layout.glyphs[i].gid in moved for i in stale)
        if seen is not None and _another_build(layer, found, earlier, drawn, relaid):
            seen.append(layer.name.strip())
        if relaid:
            raise Refused("unsupported-card", f"{layer.name.strip()}: an old glyph was laid out at another width "
                                              "(badge widths')")
        glyphs = tuple(replace(g, fill=STALE_FILL) if i in stale else g for i, g in enumerate(keep.layout.glyphs))
        _by_glyph(layer, layer.rasterize(fonts=found, keep=replace(keep, layout=replace(keep.layout, glyphs=glyphs)),
                                         unit="glyph"))
        records.add(psd_edit._record_index(layer))
        stories.append(psd_edit.story_index(layer))
        names.append(layer.name.strip())
    if not names:
        return []
    psd_edit.save_psd(psd, out_path)
    _only_pixels(path, out_path, records, stories)
    return names


def _another_build(layer, faces, earlier: tuple, drawn: tuple, relaid: bool) -> bool:
    """Whether a layer's old glyphs (drawn: _drawn_old's) show another build of the icon font, not a layout the fork can't
    make (Photoshop 2023's own saves: Wigfrid's line that fits its box by 0.05 pt, broken there; Michael's Blessing's
    tracking re-saved 65536 times larger): their advances moved, an earlier layout is its pixels (laid_plan); else,
    beside them, the current font's layout is (_off_beside within LAID_OFF)."""
    if not relaid:
        return _off_beside(drawn, set(drawn[4])) <= LAID_OFF
    try:
        return laid_plan(layer, faces, earlier) is not None
    except Refused:
        return False


def badge_plan(psd: PSDImage, fonts: Optional[Callable[[list], Optional[str]]]) -> list:
    """
    The badges and icons of the visible text layers that sit off their text (owner 2026-10-25): a badge more than
    BADGE_RESIZE off its computed size resized, a badge or icon whose middle is past FIT_PX and FIT_PCT shifted
    (psd_edit.badge_wants). A layer whose faces can't be had is not measured.

    Returns:
        [(layer, [(psd_edit.badge_fits entry, style keys)])]; raises MissingFontError
    """
    out = []
    for layer in icon_layers(psd):
        folder = fonts(layer_fonts(layer)) if fonts else ""
        if not folder:
            continue
        fits = psd_edit.badge_fits(layer, psd_edit.font_map(folder, layer))
        wants = [(fit, keys) for fit in fits for keys in [psd_edit.badge_wants(fit)] if keys]
        if wants:
            out.append((layer, wants))
    return out


def _placed(layer, fonts) -> dict:
    """Where the layout puts each character outside the icon font: {index into the engine's text: (x, y)}."""
    from psd_tools.composite.text.layout import layout
    from psd_tools.composite.text.model import TextModel
    return {glyph.cluster: (round(glyph.x, 2), round(glyph.y, 2)) for glyph in layout(TextModel.from_layer(layer), fonts).glyphs
            if not glyph.face.key[0].lower().endswith(".ttf") or ICON_FONT_PREFIX not in glyph.face.key[0].lower()}


def _set(layer, wants: list) -> None:
    for fit, keys in wants:
        layer.set_span_style(fit["start"], fit["end"], **keys)


def _recut_fx(psd: PSDImage, layer, moved: set, faces) -> list:
    """
    The layer's FX layers whose words moved (moved: _placed indices), cut again from its new raster where the words are
    now, as the bake's Update all cuts them (psd_edit._phrase_mask, _cut): only their pixels change.

    Returns:
        [(FX layer, its old bbox)]; raises Refused for a phrase on a badge or icon, or one with no pixels
    """
    fxs = psd_edit.fx_layers(psd, layer)
    if not fxs or not moved:
        return []
    engine = psd_edit._fx_engine()
    model, text = engine[0].from_layer(layer), layer.text
    offset = model.text.find(text)
    if offset < 0:
        raise Refused("unsupported-card", f"{layer.name.strip()}: the engine's text differs from the layer's")
    icons, glyphs, out = psd_edit._char_styles(layer)[1], None, []
    color, shape = layer.numpy("color"), layer.numpy("shape")
    for fx in fxs:
        ranges = psd_edit.phrase_ranges(text, psd_edit.fx_phrases(layer, fx))
        if not any(i + offset in moved for a, b in ranges for i in range(a, b)):
            continue
        if any(icons[i] for a, b in ranges for i in range(a, b)):
            raise Refused("unsupported-card", f"{fx.name.strip()}: its words fall in a badge or token")
        if any(icons) and glyphs is None:       # the colour glyphs' ink out of the key (as _rebuild_fx)
            glyphs = psd_edit._key_ink(model, lambda run: [(run.start, run.end, psd_edit.KEY_BLACK)], layer, faces, engine)
        mask = psd_edit._phrase_mask(layer, model, [(a + offset, b + offset) for a, b in ranges], glyphs, faces, engine)
        cut = psd_edit._cut(layer, mask, color, shape)
        if cut is None:
            raise Refused("unsupported-card", f"{fx.name.strip()}: its words have no pixels")
        old = tuple(fx.bbox)
        engine[6](fx, cut[0], cut[1])
        fx._invalidate_bbox()
        out.append((fx, old))
    return out


def _ink_off(layer, faces) -> float:
    """The share of the layer's ink (stored or drawn) off the fork's render of it as it stands, unshifted: how far the
    fork's layout is from Photoshop's (1.0 when either has none)."""
    import numpy as np
    from psd_tools.composite.text import capture, render
    stored, drawn = capture(layer, faces), render(layer, fonts=faces)
    if stored.rgba is None or stored.bbox is None or drawn.rgba is None or drawn.bbox is None:
        return 1.0
    boxes = (stored.bbox, drawn.bbox)
    window = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
    a, b = _premultiplied(stored.rgba, stored.bbox, window), _premultiplied(drawn.rgba, drawn.bbox, window)
    ink = (a[..., 3] > OLD_GLYPH_INK) | (b[..., 3] > OLD_GLYPH_INK)
    off = (np.abs(a - b).max(axis=2) > OLD_GLYPH_LEVEL) & ink
    return float(off.sum()) / int(ink.sum()) if ink.any() else 1.0


def _lines(layer, faces) -> list:
    """The layout's lines: [(start, end, drawn)] into the engine's text."""
    from psd_tools.composite.text.layout import layout
    from psd_tools.composite.text.model import TextModel
    return [(line.start, line.end, line.drawn) for line in layout(TextModel.from_layer(layer), faces).lines]


def fit_badges(path: str, out_path: str, fonts: Optional[Callable[[list], Optional[str]]], whole: bool = False) -> list:
    """
    Badges (owner 2026-10-25), the PIXEL_FIXES kind: badge_plan's badges and icons set (FontSize, BaselineShift; TySh
    and the Txt2 story), their layers' changed glyphs redrawn (a splice), the stored preview patched. A resize is made
    even when it rewraps the text or moves other words (owner 2026-10-25, the 58 shifted only that night): the moved
    glyphs are redrawn and an FX layer on moved words is cut again (_recut_fx). Refused when a line no longer fits the
    box (a new last line below it too: psd_edit.below_box) or ink runs off the canvas. Gated (_only_pixels, the stories
    by gate_stories), then checked: each fitted one within FIT_CHECK px of its place and at its computed size. whole (a
    sweep's --whole, never the guards): a layer the splice can't take is redrawn whole when the fork draws it as stored
    (_ink_off within WHOLE_OFF) and the fit keeps its lines.

    Returns:
        The layer names fitted; raises Refused or the engine's errors
    """
    from psd_tools.composite.text import capture
    psd = PSDImage.open(path)
    done, cuts = [], set()                      # done: (record, layer name, story, faces, spans)
    for layer, wants in badge_plan(psd, fonts):
        name, faces = layer.name.strip(), psd_edit.font_map(fonts(layer_fonts(layer)), layer)
        psd_edit.follow_kerning(psd, layer)
        whole_ok = whole and _ink_off(layer, faces) <= WHOLE_OFF
        lines = _lines(layer, faces) if whole_ok else None
        keep, placed = capture(layer, faces), _placed(layer, faces)
        off = psd_edit._ink_off_canvas(layer, psd.width, psd.height)
        _set(layer, wants)
        now = _placed(layer, faces)
        report = layer.rasterize(fonts=faces, keep=keep, unit=psd_edit.SPLICE_UNIT)
        _by_glyph(layer, report, whole_ok and _lines(layer, faces) == lines)
        if "overflow" in report.warnings or psd_edit.below_box(keep, report.lines) or \
                (not off and psd_edit._ink_off_canvas(layer, psd.width, psd.height)):
            raise Refused("overflow", f"{name}: resizing its badges pushes text out of its box")
        for fx, _ in _recut_fx(psd, layer, {i for i in placed if now.get(i) != placed[i]}, faces):
            cuts.add(psd_edit._record_index(fx))
        done.append((psd_edit._record_index(layer), name, psd_edit.story_index(layer), faces,
                     {(fit["start"], fit["end"]): "FontSize" in keys for fit, keys in wants}))
    if not done:
        return []
    psd_edit.save_psd(psd, out_path)
    _only_pixels(path, out_path, {record for record, *_ in done} | cuts,
                 [story for _, _, story, *_ in done if story is not None])
    layers = {psd_edit._record_index(layer): layer for layer in PSDImage.open(out_path).descendants() if layer.kind == "type"}
    for record, name, _, faces, spans in done:
        for fit in psd_edit.badge_fits(layers[record], faces, list(spans)):
            if abs(fit["off_px"]) > FIT_CHECK or (spans[(fit["start"], fit["end"])] and abs(fit["size_off"]) > 0.001):
                raise Refused("bytes", f"{name} {fit['token']}: {fit['off_px']:+.1f} px off after the fit")
    return [name for _, name, *_ in done]


def _salt_spans(layer) -> list:
    """The layer's badges and :stickers in the icon font when an overlay paints over it (psd_edit.one_colour)."""
    if not psd_edit.one_colour(layer):
        return []
    text = layer.text
    return psd_edit.alternate_spans(text, psd_edit.icon_text(text, None), psd_edit._char_styles(layer)[1])


def _unsalted(layer) -> bool:
    spans = _salt_spans(layer)
    if not spans:
        return False
    flagged = {i for a, b in psd_edit.stylistic_alternates(layer) or () for i in range(a, b)}
    return any(i not in flagged for a, b in spans for i in range(a, b))


def salt_layers(psd: PSDImage) -> list:
    """The visible type layers under an overlay with a badge or :sticker not in the icon font's Stylistic Alternates:
    printed as a blob (Butterfly's Section layers, re-saved in Photoshop; owner 2026-10-30)."""
    out = []
    for layer in icon_layers(psd):
        try:
            if _unsalted(layer):
                out.append(layer)
        except Exception:                       # unreadable runs, or a fork without the alternates: left alone
            continue
    return out


def one_colour_badges(path: str, out_path: str, fonts: Optional[Callable[[list], Optional[str]]]) -> list:
    """
    One-colour badges (owner 2026-10-30): salt_layers' badges and :stickers set in the icon font's Stylistic Alternates
    (psd_edit.set_alternates, the Txt2 story alone, as the bake sets them), those glyphs redrawn (a splice), the stored
    preview patched; gated (_only_pixels, the stories by gate_stories).

    Returns:
        The layer names changed; raises Refused or the engine's errors
    """
    from psd_tools.composite.text import capture
    psd = PSDImage.open(path)
    records, stories, names = set(), [], []
    for layer in salt_layers(psd):
        found = _fonts_of(layer, fonts)
        psd_edit.follow_kerning(psd, layer)
        keep = capture(layer, found)
        psd_edit.set_alternates(layer, _salt_spans(layer), [])
        _by_glyph(layer, layer.rasterize(fonts=found, keep=keep, unit=psd_edit.SPLICE_UNIT))
        records.add(psd_edit._record_index(layer))
        if psd_edit.story_index(layer) is not None:
            stories.append(psd_edit.story_index(layer))
        names.append(layer.name.strip())
    if not names:
        return []
    psd_edit.save_psd(psd, out_path)
    _only_pixels(path, out_path, records, stories)
    if salt_layers(PSDImage.open(out_path)):
        raise Refused("bytes", "a badge is left without its one-colour shape")
    return names


def tags_to_badges(path: str, out_path: str, fonts: Optional[Callable[[list], Optional[str]]], whole: bool = False) -> list:
    """
    R1's fix (owner 2026-11-03: "make sure everything uses the badges"; the guards on every push too), the PIXEL_FIXES
    kind: tag_layers' plain timing tags and costs set in the icon font at the layer's own badge style
    (psd_edit.icon_style; with none, at the text's own style, ICON_FONT added as the bake adds it), ligatures on, then
    fitted as a typed badge (badge_fits: size and middle), in Stylistic Alternates under an overlay (psd_edit.one_colour);
    their layers' changed glyphs redrawn (a splice; text a badge moves is redrawn, an FX layer on moved words cut
    again), the stored preview patched. Refused when a line no longer fits the box (a new last line below it too:
    psd_edit.below_box) or ink runs off the canvas. Gated (_only_pixels, the stories and the added face by
    gate_stories), then checked: no plain tag left. whole (a sweep's --whole, never the guards): a layer the splice
    can't take is redrawn whole when the fork draws it as stored (_ink_off within WHOLE_OFF).

    Returns:
        The layer names changed; raises Refused or the engine's errors
    """
    from psd_tools.composite.text import capture
    psd = PSDImage.open(path)
    records, stories, names, added = set(), [], [], []
    for layer in tag_layers(psd):
        spans, name = plain_tags(layer), layer.name.strip()
        faces = psd_edit.font_map(fonts(layer_fonts(layer)), layer)
        whole_ok = whole and _ink_off(layer, faces) <= WHOLE_OFF
        psd_edit.follow_kerning(psd, layer)
        keep, placed = capture(layer, faces), _placed(layer, faces)
        off = psd_edit._ink_off_canvas(layer, psd.width, psd.height)
        look = psd_edit.icon_style(layer, True, None)
        if look is None:                        # no badge or token on the layer: at the text's own style
            if ICON_FONT not in layer_fonts(layer):
                added.append(ICON_FONT)
            layer.add_font(ICON_FONT)
            look = {"Font": ICON_FONT}
        for a, b in spans:
            layer.set_span_style(a, b, **look)
            layer.set_span_style(a, b, Ligatures=True)
        faces = psd_edit.font_map(fonts(layer_fonts(layer)), layer)
        for fit in psd_edit.badge_fits(layer, faces, spans):
            layer.set_span_style(fit["start"], fit["end"], FontSize=fit["fit_size"], BaselineShift=fit["fit_shift"])
        if psd_edit.one_colour(layer):          # under an overlay: in one colour, as its other badges (Morpho Knight)
            psd_edit.set_alternates(layer, _salt_spans(layer), [])
        now = _placed(layer, faces)
        report = layer.rasterize(fonts=faces, keep=keep, unit=psd_edit.SPLICE_UNIT)
        _by_glyph(layer, report, whole_ok)
        if "overflow" in report.warnings or psd_edit.below_box(keep, report.lines) or \
                (not off and psd_edit._ink_off_canvas(layer, psd.width, psd.height)):
            raise Refused("overflow", f"{name}: its badges push text out of its box")
        for fx, _ in _recut_fx(psd, layer, {i for i in placed if now.get(i) != placed[i]}, faces):
            records.add(psd_edit._record_index(fx))
        records.add(psd_edit._record_index(layer))
        if psd_edit.story_index(layer) is not None:
            stories.append(psd_edit.story_index(layer))
        names.append(name)
    if not names:
        return []
    psd_edit.save_psd(psd, out_path)
    _only_pixels(path, out_path, records, stories, tuple(dict.fromkeys(added)))
    if tag_layers(PSDImage.open(out_path)):
        raise Refused("bytes", "a timing tag is left outside the badge font")
    return names


def _view_steps(layer) -> dict:
    """Photoshop's layout of the layer's story when it last laid it out (Txt2's view, its glyph strikes; Photoshop CC's
    numbered keys): {index into the text: [each glyph's pen step, pt]} per glyph run; {} without one."""
    from psd_tools.api import text_edit
    from psd_tools.psd import engine_data
    try:
        _, keys, stories = text_edit._txt2(layer._psd)
        story = text_edit._nth_story(stories, psd_edit.story_index(layer), keys)
    except (TypeError, ValueError, KeyError):
        return {}
    if not keys.numbered:
        return {}
    out, pending = {}, [(story[keys.view].get(keys.strikes), 0)]
    while pending:
        node, start = pending.pop()
        if isinstance(node, engine_data.List):
            pending.extend((item, start) for item in node)
        elif isinstance(node, engine_data.Dict):
            tag = str(_value(node.get("99", "")))
            if tag == "S":                      # a segment: its first character
                start = int(_value(node.get("20", 0)))
            if tag != "G":
                pending.extend((item, start) for item in node.values())
                continue
            try:                                # a glyph strike: runs of glyphs, pen positions
                pen, k = [float(_value(x)) for x in node["15"]], 0
                for run, n in zip(node["10"]["0"], node["10"]["1"]):
                    n = int(_value(n))
                    out[start + int(_value(run.get("0", 0)))] = [b - a for a, b in zip(pen[k:k + n], pen[k + 1:k + n + 1])]
                    k += n
            except (KeyError, TypeError, ValueError):
                continue
    return out


@lru_cache(maxsize=4)
def _hb_font(path: str):
    import uharfbuzz as hb
    return hb.Font(hb.Face(hb.Blob.from_file_path(path), 0))


def laid_advances(layer, faces, earlier: tuple = ()) -> list:
    """
    The advances the layer's icon font glyphs may have been laid out with where they differ from the current font's:
    Photoshop's (its Txt2 view, _view_steps: exact when Photoshop drew the layer last), each earlier icon font's (by
    glyph name: a raster the fork drew), and the view's over each.

    Args:
        faces: A FontMap (psd_edit.font_map)
        earlier: Earlier icon font files

    Returns:
        [{glyph id in the current font: advance in font units}], one per way; [] when none differs
    """
    from psd_tools.composite.text.layout import layout
    from psd_tools.composite.text.model import TextModel
    from psd_tools.composite.text.shape import rendered_em
    model = TextModel.from_layer(layer)
    glyphs = layout(model, faces).glyphs
    icons = [g for g in glyphs if _icon_face(g.face)]
    if not icons:
        return []
    face, first, view = icons[0].face, {}, {}
    for i, g in enumerate(glyphs):
        first.setdefault(g.cluster, i)
    for at, steps in _view_steps(layer).items():
        run = glyphs[first[at]:first[at] + len(steps)] if at in first else ()
        if len(run) != len(steps) or not all(_icon_face(g.face) for g in run):
            continue
        for g, step in zip(run, steps):
            style = model.runs[g.run].style
            # Photoshop's tracking is scaled by the run's HorizontalScale (Wigfrid's strike: 9.54 pt, not 16.17)
            off = step - g.advance - style.tracking / 1000 * rendered_em(style, model.sub_size) * style.horizontal_scale
            if abs(off) > LAID_TOL and g.face is face:
                units = round(face.hb.get_glyph_h_advance(g.gid) + off * face.metrics.upem / g.em_w)
                if view.setdefault(g.gid, units) != units:
                    view = {None: None}         # two sizes in the view: not a layout of this text
    ways = [{} if None in view else view]
    for path in earlier:
        font, way = _hb_font(path), {}
        for g in icons:
            found = font.get_glyph_from_name(face.hb.get_glyph_name(g.gid) or "")
            if found is not None and font.get_glyph_h_advance(found) != face.hb.get_glyph_h_advance(g.gid):
                way[g.gid] = font.get_glyph_h_advance(found)
        ways += [way, {**way, **ways[0]}]
    return [way for i, way in enumerate(ways) if way and way not in ways[:i]]


class _Swapped:
    """A FontMap whose icon font is another file (an earlier version): its own glyphs, advances and drawings."""

    def __init__(self, faces, path: str):
        self.faces, self.path, self._face = faces, path, None

    def resolve(self, name: str):
        from psd_tools.composite.text.fonts import Resolved, load_face
        found = self.faces.resolve(name)
        if found.face is None or not _icon_face(found.face):
            return found
        self._face = self._face or load_face(self.path)
        return Resolved(self._face, found.how)

    def resolve_all(self, names) -> dict:
        return {name: self.resolve(name) for name in dict.fromkeys(names)}


def _glyph_key(g) -> tuple:
    """A laid glyph as two icon font versions' layouts compare: its cluster and, in the icon font, its name (ids shift
    when icons are added), else its id."""
    return g.cluster, g.face.hb.get_glyph_name(g.gid) if _icon_face(g.face) else g.gid


def _drawn_earlier(layer, faces, earlier: tuple, now: tuple) -> set:
    """The icon glyphs (indices into now's layout: _glyph_offs at the current font) an earlier icon font draws as the
    layer's pixels show them: that font's own render at its own layout fits the glyph within EARLIER_FIT (not old to it)
    and better by EARLIER_GAIN than the current font's (a badge whose letters changed, its ring or pill alike). Only a
    layout of the same glyphs counts (_glyph_key; an older font that sets the text otherwise is not compared)."""
    found, keys = set(), [_glyph_key(g) for g in now[0].layout.glyphs]
    for path in earlier:
        try:
            then = _glyph_offs(layer, _Swapped(faces, path))
        except Exception:                       # an unreadable earlier font: no evidence
            continue
        if then is None or [_glyph_key(g) for g in then[0].layout.glyphs] != keys:
            continue
        found |= {i for i, share in then[4].items()
                  if i in now[4] and share <= EARLIER_FIT and now[4][i] - share >= EARLIER_GAIN}
    return found


class _Laid:
    """A FontMap whose icon font glyphs take other advances (a HarfBuzz sub-font): the same file, outlines and face key, so
    the splice and the glyph cache take its glyphs for the current font's."""

    def __init__(self, faces, advances: dict):
        self.faces, self.advances, self._made = faces, dict(advances), {}

    def resolve(self, name: str):
        import uharfbuzz as hb
        from psd_tools.composite.text.fonts import Face, Resolved
        found = self.faces.resolve(name)
        face = found.face
        if face is None or not _icon_face(face):
            return found
        if face.key not in self._made:
            sub, funcs, parent = hb.Font(face.hb), hb.FontFuncs(), face.hb.get_glyph_h_advance
            funcs.set_glyph_h_advance_func(lambda _, gid, wants: wants[gid] if gid in wants else parent(gid), self.advances)
            sub.funcs = funcs
            made = Face(hb=sub, ft=face.ft, metrics=face.metrics, key=face.key, lock=face.lock,
                        _instructed=face._instructed)
            made.__dict__["svg"] = face.svg     # its parsed SVG table shared
            self._made[face.key] = made
        return Resolved(self._made[face.key], found.how)

    def resolve_all(self, names) -> dict:
        return {name: self.resolve(name) for name in dict.fromkeys(names)}


def _off_beside(drawn: tuple, marks: set) -> float:
    """_drawn_old's share of the layer's ink (stored or drawn) off the render, unshifted, outside the marked glyphs' boxes
    and pen cells (1.0 with no ink)."""
    from math import ceil, floor
    import numpy as np
    from psd_tools.composite.text.geometry import _cell, _map_box
    from psd_tools.composite.text.splice import _glyph_box
    before, window, stored, mine, _ = drawn
    ink = (stored[..., 3] > OLD_GLYPH_INK) | (mine[..., 3] > OLD_GLYPH_INK)
    off = (np.abs(stored - mine).max(axis=2) > OLD_GLYPH_LEVEL) & ink
    for i in marks:
        g = before.layout.glyphs[i]
        cell = _map_box(before.model.transform, _cell(g))
        for box in (_glyph_box(g, before.model.transform, OLD_GLYPH_SHIFT),
                    (floor(cell[0]) - OLD_GLYPH_SHIFT, floor(cell[1]), ceil(cell[2]) + OLD_GLYPH_SHIFT, ceil(cell[3]))):
            if box is not None:
                x0, y0 = max(box[0] - window[0], 0), max(box[1] - window[1], 0)
                ink[y0:max(box[3] - window[1], 0), x0:max(box[2] - window[0], 0)] = False
    off &= ink
    return float(off.sum()) / int(ink.sum()) if ink.any() else 1.0


def laid_plan(layer, faces, earlier: tuple = ()) -> Optional[tuple]:
    """
    Whether the layer's stored pixels show its icon font glyphs laid out at other advances: some glyph laid_advances
    finds is drawn old (old_glyphs, or in an earlier icon font: _drawn_earlier) and the layer drawn at one of its ways is
    its pixels, those glyphs and the ones drawn old there aside (_off_beside within LAID_OFF; the nearest way). None when the pixels show the current font's
    (re-laid already, or drawn in it) or no advance differs; raises Refused when no way is (the fork lays the layer out
    off Photoshop's pixels).

    Returns:
        (_Laid faces, {glyph index: redrawn}, that share)
    """
    ways = laid_advances(layer, faces, earlier)
    now = _glyph_offs(layer, faces) if ways else None
    if now is None:
        return None
    moved = lambda found: any(now[0].layout.glyphs[i].gid in way for i in found for way in ways)
    drawn_then = _drawn_earlier(layer, faces, earlier, now)
    if not moved({i for i, share in now[4].items() if share > OLD_GLYPH_BAD}) and not moved(drawn_then):
        odd = {i for i, share in now[4].items() if share > DRAWN_ODD and i not in drawn_then}
        if moved(odd):
            raise Refused("unsupported-card", f"{layer.name.strip()}: a badge drawn in no known icon font (its pixels "
                                              f"{max(now[4][i] for i in odd):.0%} off this font's); export it in Photoshop")
        return None
    best = None
    for way in ways:
        laid = _Laid(faces, way)
        drawn = _drawn_old(layer, laid)
        marks = set(drawn[4]) | {i for i, g in enumerate(drawn[0].layout.glyphs) if g.gid in way and _icon_face(g.face)}
        share = _off_beside(drawn, marks)
        if best is None or share < best[2]:
            best = (laid, marks, share)
    if best[2] > LAID_OFF:
        raise Refused("unsupported-card", f"{layer.name.strip()}: at its old badge widths the fork's layout is "
                                          f"{best[2]:.0%} off its pixels")
    return best


def relay_badges(path: str, out_path: str, fonts: Optional[Callable[[list], Optional[str]]]) -> list:
    """
    Badge widths (owner 2026-11-04: the cooldown ring widens every cooldown badge), the PIXEL_FIXES kind: a layer whose
    icon font glyphs were laid out at other advances than the current font's (laid_plan: Photoshop's, from its Txt2
    view; an earlier icon font's, fonts.earlier: the guards' and a sweep's) laid out again at the current ones, its lines
    rewrapped in its box as they now fall: a splice from that old layout (the glyphs that moved or were drawn old
    redrawn, an FX layer on moved words cut again), the stored preview patched; the text, its styles and its story untouched. Refused when a
    line no longer fits the box (its baseline below it too: Photoshop hides it) or ink runs off the canvas; noted (the
    sweep plan) when a layer gains a line, a line holds one word it didn't, or ink comes near the canvas edge. The splice
    may fall back to a whole redraw only when the fork draws the layer as stored at its old advances (WHOLE_OFF beside
    the redrawn glyphs), in the guards too. Gated (_only_pixels), then checked: nothing left to re-lay.

    Returns:
        The layer names re-laid (their notes in .notes); raises Refused or the engine's errors
    """
    from dataclasses import replace
    from psd_tools.composite.text import capture
    earlier = tuple(getattr(fonts, "earlier", ()))
    psd = PSDImage.open(path)
    records, stories, names = set(), [], Relaid()
    for record, faces, plan in _laid_plans(path, fonts, earlier):
        layer = next(layer for layer in icon_layers(psd) if psd_edit._record_index(layer) == record)
        psd_edit.follow_kerning(psd, layer)
        laid, marks, share = plan
        name, keep = layer.name.strip(), capture(layer, laid)
        glyphs = tuple(replace(g, fill=STALE_FILL) if i in marks else g for i, g in enumerate(keep.layout.glyphs))
        placed, now = _placed(layer, laid), _placed(layer, faces)
        off, inked = psd_edit._ink_off_canvas(layer, psd.width, psd.height), _ink_box(layer)
        report = layer.rasterize(fonts=faces, keep=replace(keep, layout=replace(keep.layout, glyphs=glyphs)),
                                 unit=psd_edit.SPLICE_UNIT)
        _by_glyph(layer, report, share <= WHOLE_OFF)
        if "overflow" in report.warnings or psd_edit.below_box(keep, report.lines):
            raise Refused("overflow", f"{name}: its badges at their new widths push text out of its box")
        reach, sides = psd_edit._stroke_reach(layer), _sides(_ink_box(layer), psd.width, psd.height)
        close = [now for now, was in zip(sides, _sides(inked, psd.width, psd.height)) if now <= max(reach, 1) and now < was]
        if (not off and psd_edit._ink_off_canvas(layer, psd.width, psd.height)) or close:   # its outline at the edge too, a side each
            raise Refused("overflow", f"{name}: its badges at their new widths take its ink {min(close or sides)} px from "
                                      f"the canvas edge (its outline {reach} px)")
        alone = _badges_alone(keep.layout.lines, report.lines)
        if alone:
            raise Refused("overflow", f"{name}: its badges at their new widths leave {alone!r} alone on a line")
        hair = _razor(layer, faces)
        if hair == "overset":
            raise Refused("overflow", f"{name}: a line fits its box by a hair (its breaks change {RAZOR:.1%} narrower and "
                                      "its last line drops below the box): Photoshop may hide it")
        names.notes += [f"{name}: its line breaks hang on a hair (they change in a box {RAZOR:.1%} narrower or wider)"] * bool(hair)
        names.notes += _reflow_notes(name, keep.layout.lines, report.lines, inked, _ink_box(layer), psd.width, psd.height,
                                     keep.model, (reach, _glow_reach(layer)), getattr(fonts, "cards", None))
        for fx, _ in _recut_fx(psd, layer, {i for i in placed if now.get(i) != placed[i]}, faces):
            records.add(psd_edit._record_index(fx))
        records.add(psd_edit._record_index(layer))
        stories.append(psd_edit.story_index(layer))
        names.append(name)
    if not names:
        return []
    psd_edit.save_psd(psd, out_path)
    _only_pixels(path, out_path, records, stories)
    try:
        left = _laid_plans(out_path, fonts, earlier)
    except Refused as error:
        left = [error]
    if left:
        raise Refused("bytes", "a badge is left at its old width")
    return names


class Relaid(list):
    """relay_badges' layer names, with notes: what the re-layout did that a reader should look at."""
    notes: list

    def __init__(self, *args):
        super().__init__(*args)
        self.notes = []


EDGE_NEAR = 24                                  # px: ink this near the canvas edge is noted (nearer than before)


def _ink_box(layer) -> tuple:
    """The layer's ink (alpha past psd_edit.INK_ALPHA) as a canvas bbox: a faint fringe (a halo's tail) left out."""
    import numpy as np
    bbox, alpha = tuple(layer.bbox), psd_edit._alpha(layer)
    if alpha is None or not (alpha > psd_edit.INK_ALPHA).any():
        return bbox
    rows, cols = np.nonzero((alpha > psd_edit.INK_ALPHA).any(axis=1))[0], np.nonzero((alpha > psd_edit.INK_ALPHA).any(axis=0))[0]
    return bbox[0] + int(cols[0]), bbox[1] + int(rows[0]), bbox[0] + int(cols[-1]) + 1, bbox[1] + int(rows[-1]) + 1


def _sides(bbox, width: int, height: int) -> tuple:
    """px from ink (a bbox) to each canvas edge: left, top, right, bottom."""
    return bbox[0], bbox[1], width - bbox[2], height - bbox[3]


def _glow_reach(layer) -> int:
    """Pixels the outer glows and drop shadows of the layer and its groups draw past its ink (the farthest)."""
    out = 0
    for owner in (layer, *psd_edit._ancestors(layer)):
        try:
            effects = list(owner.effects)
        except Exception:                       # no effects block, or one psd-tools can't read
            effects = []
        for effect in effects:
            kind = type(effect).__name__
            if effect.enabled and kind in ("OuterGlow", "DropShadow"):
                out = max(out, int(float(getattr(effect, "size", 0) or 0)
                                   + (float(getattr(effect, "distance", 0) or 0) if kind == "DropShadow" else 0) + 0.5))
    return out


BADGE_LINE_REST = re_compile(r"[\s.,;:!?]*")      # what a line of badges alone may hold beside them: (ENDT).


def _badges_only(text: str) -> bool:
    """Whether a line holds badges (psd_edit.badge_spans: timing tags and costs the font draws) and nothing else but
    spaces and punctuation: (1CD), (1CD)(3 points), (ENDT).; not plain bracketed text ('(cannot ignore usage conditions).')."""
    spans = psd_edit.badge_spans(text)
    rest = "".join(text[b:a] for (_, b), (a, _) in zip([(0, 0), *spans], [*spans, (len(text), len(text))]))
    return bool(spans) and BADGE_LINE_REST.fullmatch(rest) is not None


def _badges_alone(was, now) -> str:
    """A new line of the re-layout that holds only badges (Wigfrid's '(1CD)', cut off from its cost), or ""."""
    texts = lambda ls: [line.text.strip() for line in ls if line.drawn and line.text.strip()]
    old = texts(was)
    return next((t for t in texts(now) if _badges_only(t) and t not in old), "")


def _razor(layer, faces) -> str:
    """Whether the layer's breaks hang on a hair: "overset" when a box RAZOR narrower breaks them so that a line drops
    below the box (Photoshop hides it), "breaks" when it or one RAZOR wider breaks them otherwise, else ""."""
    from dataclasses import replace
    from psd_tools.composite.text.layout import frame, layout
    from psd_tools.composite.text.model import TextModel
    model = TextModel.from_layer(layer)
    box = frame(model)
    if box is None:
        return ""
    lines = lambda m: layout(m, faces).lines
    below = lambda ls: sum(1 for line in ls if line.text.strip() and line.y > box[1] + box[3] + 1e-6)
    base = lines(model)
    left, top, right, bottom = model.box
    found = ""
    for sign in (-1, 1):
        now = lines(replace(model, box=(left, top, right + sign * RAZOR * (right - left), bottom)))
        if [(line.start, line.end) for line in now] == [(line.start, line.end) for line in base]:
            continue
        if sign < 0 and below(now) > below(base):
            return "overset"
        found = "breaks"
    return found


COUNT_END = re_compile(r"[+\-\u2212]?\d+(?:\.\d+)?|\d+T\d*|lvl?|x\d+", IGNORECASE)   # a count cut from its word: '4' / 'HP',
                                                # '+2' / 'ATK', '1T' / 'Insane', 'lv' / '1-3', 'x2' / 'DMG'
TERM_WORDS = 4                                  # a coloured term ('Fire Totem', 'Frozen In Time'): a coloured paragraph isn't one


def _split_terms(model, was, now, cards: Optional[tuple] = None) -> list:
    """New breaks of a re-layout (not in its old layout) that cut a term: right after a count, a word next; inside
    a run of up to TERM_WORDS words coloured unlike the layer's main colour ('Fire' / 'Totem'); or inside a card's
    name (cards: Context.cards; 'Mini' / 'Drill.', however coloured): [(before, after)]. model: its text and runs."""
    lines = [line for line in now if line.drawn and line.text.strip()]
    old, out = {line.start for line in was if line.drawn}, []
    names = _name_index(cards)[0] if cards and model is not None else None
    spans = [m.span(1) for m in names.finditer(model.text)] if names else []
    fill, main = {}, None
    if model is not None:
        fill = {i: tuple(run.style.fill) for run in model.runs for i in range(run.start, run.end)}
        counts = Counter(fill[i] for i in fill if i < len(model.text) and not model.text[i].isspace())
        main = counts.most_common(1)[0][0] if counts else None
    for a, b in zip(lines, lines[1:]):
        head, tail = a.text.split(), b.text.split()
        if b.start in old or not head or not tail:
            continue
        cut = (COUNT_END.fullmatch(head[-1]) and tail[0][:1].isalnum()) or any(a0 < b.start < b1 for a0, b1 in spans)
        if not cut and main is not None:
            text = model.text
            last = next((i for i in range(min(a.end, len(text)) - 1, a.start - 1, -1) if not text[i].isspace()), None)
            first = next((i for i in range(b.start, min(b.end, len(text))) if not text[i].isspace()), None)
            cut = None not in (last, first) and fill.get(last) == fill.get(first) != main and fill.get(last) is not None
            if cut:                                     # the coloured run's words, spaces inside it included
                same = lambda i: fill.get(i) == fill[last] or (text[i].isspace() and fill.get(i + 1) == fill[last])
                a0, b1 = last, first
                while a0 > 0 and same(a0 - 1) and text[a0 - 1] != "\r":
                    a0 -= 1
                while b1 + 1 < len(text) and same(b1 + 1) and text[b1 + 1] != "\r":
                    b1 += 1
                cut = len(text[a0:b1 + 1].split()) <= TERM_WORDS
        if cut:
            out.append((head[-1], tail[0]))
    return out


def _reflow_notes(name: str, was, now, bbox_was, bbox_now, width: int, height: int, model=None, reach=(0, 0),
                  cards: Optional[tuple] = None) -> list:
    """A re-layout's notes: more lines, a new line of one word (a widow), a new break that cuts a term (_split_terms),
    ink within EDGE_NEAR of a canvas edge and nearer than before there (each side; reach: its outline's px and its
    glow's or shadow's, how near they come too)."""
    texts = lambda ls: [line.text.strip() for line in ls if line.drawn]
    old, new = texts(was), texts(now)
    notes = [f"{name}: {len(old)} line{'s' * (len(old) != 1)} became {len(new)}"] if len(new) > len(old) else []
    notes += [f"{name}: {t!r} alone on a line" for t in new if len(t.split()) == 1 and t not in old]
    notes += [f"{name}: {a!r} / {b!r} split across lines" for a, b in _split_terms(model, was, now, cards)]
    near = [(now, was) for now, was in zip(_sides(bbox_now, width, height), _sides(bbox_was, width, height))
            if now < EDGE_NEAR and now < was]
    if near:
        now, was = min(near)
        stroke, glow = reach
        notes.append(f"{name}: ink {now} px from the canvas edge (was {was})" + (f", its outline {now - stroke} px" if stroke else "")
                     + (f", its glow {now - glow} px" if glow > stroke else ""))
    return notes


def _laid_plans(path: str, fonts, earlier: tuple) -> list:
    """laid_plan over the PSD's icon layers, each measured as Photoshop kerns it (B12, follow_kerning: on a copy, so a
    layer left alone keeps its bytes): [(record index, faces, plan)]."""
    psd, out = PSDImage.open(path), []
    for layer in icon_layers(psd):
        faces = _fonts_of(layer, fonts)
        psd_edit.follow_kerning(psd, layer)
        plan = laid_plan(layer, faces, earlier)
        if plan is not None:
            out.append((psd_edit._record_index(layer), faces, plan))
    return out


# ========================
# Template bars (owner 2026-11-06): the creature template's layer-comp stat bars in place of the exposure digit rows
# ========================
ALT_TEMPLATE = "MDW/Templates/Creature_Template.psd"
ALT_ROW = (813, 84)         # a template bar row's size; a card's row within ALT_ROW_TOL of it takes the template's bar
ALT_ROW_TOL = 4
ALT_FILL = (0.96, 0.99)    # the share of its box a bar shaped like the template's fills (0.977; Little Guy's squares 0.84)
ALT_SIZES = ((300, 40), (1400, 160))   # a card's own bar this size takes a resized bar (Little Guy 920x78, Sonic's 816x99)
ALT_PLAIN = 171             # a bar's dark: the template's 67%
ALT_PLAIN_NEAR = 0.1        # a plain digit passing this near what ALT_PLAIN passes stays at it
ALT_COVER = 99.5            # an overlay's % that hides the cells: its bar stays solid
ALT_GAMMA = 2.2             # an exposure e passes 2 ** (e / ALT_GAMMA) of the art


def _bar_art(psd, stat: str) -> Optional[tuple]:
    """
    The card's own bar for a stat: a visible layer named like it ("hp", "hp bar") in a group named like *bar*, else a
    visible group so named there (its drawn leaves but lines and darks: Sonic's "spd top" and "spd bottom"), else its
    top and bottom halves there (Royal Eradicator's "atk top" and "atk bottom"); its bbox.
    """
    named = lambda layer: fullmatch(rf"\s*({STAT_PATTERNS[stat]})\s*(bar)?\s*", layer.name, IGNORECASE)
    under_bars = lambda layer: (layer.parent is not None and not isinstance(layer.parent, PSDImage)
                                and "bar" in layer.parent.name.lower())
    for layer in psd.descendants():
        if not layer.is_group() and layer.is_visible() and under_bars(layer) and named(layer):
            image = layer.topil()               # the template's rounded bar fills ALT_FILL of its box (squares or arrows don't)
            filled = (np.asarray(image.convert("RGBA"))[..., 3] > 128).mean() if image is not None else ALT_FILL[0]
            return tuple(layer.bbox) if ALT_FILL[0] <= filled <= ALT_FILL[1] else None
    for group in psd.descendants():
        if group.is_group() and group.is_visible() and under_bars(group) and named(group):
            art = [l.bbox for l in group.descendants() if not l.is_group() and l.is_visible() and l.bbox[2] > l.bbox[0]
                   and not search(r"line|dark", l.name, IGNORECASE)]
            if art:
                return union_box(art)
    return union_box([layer.bbox for layer in psd.descendants() if not layer.is_group() and layer.is_visible() and under_bars(layer)
                      and fullmatch(rf"\s*({STAT_PATTERNS[stat]})\s+(top|bottom)\s*", layer.name, IGNORECASE)])


def _alt_parts(psd):
    """
    (the Darks group, {stat: (value, its bar's box, its shown digit or None)}, the empty dark groups to drop) when the
    card's bars are the template's exposure digit rows: per stat one "<Stat> Dark" group of them, or of its half rows (a
    stat past 10: Top and Bottom, the parser's sum), all four in one visible Darks group holding nothing else that draws;
    another dark group of a stat showing no digit draws nothing and goes (Sonic's). A bar takes the card's own bar's box
    (_bar_art, resized ones too) when it is on the digits' row and spans it, else the art on that row (a name can lie:
    the Pandas' def and atk), else its digit masks' at the template's size; else None.
    """
    if any(comp_bars(psd).values()):
        return None
    parts, parents, drop = {}, {}, []
    for stat, found in stat_rows(psd).items():
        if not found:
            return None
        owners = {}
        for g, shown, _ in found:
            owner = next((a for a in [g, *_ancestors_of(g)]
                          if fullmatch(rf".*\b({STAT_PATTERNS[stat]})\b.*dark.*", a.name.strip(), IGNORECASE)), None)
            if owner is None:
                return None
            owners.setdefault(id(owner), [owner, False])[1] |= bool(shown)
        drawn = [o for o, any_shown in owners.values() if any_shown] or [next(iter(owners.values()))[0]]
        if len(drawn) != 1:
            return None
        owner = drawn[0]
        drop += [o for o, _ in owners.values() if o is not owner]
        rows = [(g, shown, box) for g, shown, box in found if g is owner or any(a is owner for a in _ancestors_of(g))]
        leaves = [d for d in owner.descendants() if not d.is_group() and (d.is_visible() or d.name.strip().isdigit())
                  and not (d.kind == "type" and not (d.text or "").strip())]
        if not leaves or any(d.kind != "exposure" or not d.name.strip().isdigit() for d in leaves):
            return None
        filled = [min(shown) for _, shown, _ in rows if shown]
        value = sum(filled) if filled else 10
        row = bar_box(psd, stat, rows[0][2]) if len(rows) == 1 and rows[0][2] else None
        on_row = lambda box: box and row[1] <= (box[1] + box[3]) / 2 <= row[3] \
            and box[2] - box[0] >= min(row[2] - row[0], ALT_ROW[0]) - ALT_ROW_TOL       # Monkey D Letion's icons don't span it
        art = _bar_art(psd, stat)
        if art and row and not on_row(art):
            art = next((box for box in (_bar_art(psd, other) for other in STAT_PATTERNS) if on_row(box)), None)
        (w0, h0), (w1, h1) = ALT_SIZES
        if art and w0 <= art[2] - art[0] <= w1 and h0 <= art[3] - art[1] <= h1:
            box = tuple(art)
        elif row:
            x0, y0, x1, y1 = row
            if abs(y1 - y0 - ALT_ROW[1]) > ALT_ROW_TOL or abs(x1 - x0 - ALT_ROW[0]) > ALT_ROW_TOL:
                return None
            box = (x0, y0, x0 + ALT_ROW[0], y0 + ALT_ROW[1])
        else:                                   # half rows and no bar of its own: placed by the other rows (_spaced)
            box = None
        shown = [d for d in leaves if d.is_visible()]
        parts[stat] = (value, box, next((d for d in shown if _styled(d)), next(iter(shown), None)))
        parents[id(owner.parent)] = owner.parent
    if len(parts) != 4 or len(parents) != 1 or not _spaced(parts):
        return None
    darks = next(iter(parents.values()))
    if (isinstance(darks, PSDImage) or len(darks) != 4 or not darks.visible or darks.opacity != 255 or darks.has_mask()
            or darks.has_effects()):
        return None
    return darks, parts, drop


def _spaced(parts: dict) -> bool:
    """A bar with no box of its own takes the others' size and x, and their evenly spaced y in stat order (the
    template's 126 px); False when the others are not so spaced or two bars lack one."""
    order = list(STAT_PATTERNS)
    placed = {stat: box for stat, (_, box, _) in parts.items() if box}
    missing = [stat for stat in parts if parts[stat][1] is None]
    if not missing:
        return True
    if len(missing) > 1 or len(placed) < 2 or len({(b[0], b[2] - b[0], b[3] - b[1]) for b in placed.values()}) != 1:
        return False
    known = sorted((order.index(stat), box[1]) for stat, box in placed.items())
    steps = {(y1 - y0) / (i1 - i0) for (i0, y0), (i1, y1) in zip(known, known[1:])}
    if len(steps) != 1:
        return False
    step, (i0, y0), like = steps.pop(), known[0], next(iter(placed.values()))
    stat = missing[0]
    top = round(y0 + (order.index(stat) - i0) * step)
    value, _, digit = parts[stat]
    parts[stat] = (value, (like[0], top, like[2], top + like[3] - like[1]), digit)
    return True


def _ancestors_of(layer) -> list:
    out, parent = [], layer.parent
    while parent is not None and not isinstance(parent, PSDImage):
        out.append(parent)
        parent = parent.parent
    return out


def alt_bar_rows(psd: PSDImage) -> list:
    """The Darks group the template's bars would replace (alt_template), else []."""
    found = _alt_parts(psd)
    return [found[0]] if found else []


def _styled(digit) -> bool:
    """A digit draws a style (the shown one alone counts: a hidden one's can be stale, set_stat's rule)."""
    return digit.has_effects() and any(e.enabled for e in digit.effects)


def alt_dark(digit) -> tuple:
    """
    (a bar's opacity, its overlay's % or None) that draw like the dark digit: passing T = (1 - p) * 2 ** (e / ALT_GAMMA)
    of the art under a Normal overlay at p, the bar at 1 - T, the overlay at p / (1 - T) to land its colour as before.
    A plain digit: the template's dark when near it, else 1 - T. Any other style: solid.
    """
    exposure = digit.exposure if digit is not None and digit.kind == "exposure" else None
    if exposure is None or exposure > 0:
        return ALT_PLAIN, None
    passes = 2 ** (exposure / ALT_GAMMA)
    drawn = [e for e in digit.effects if e.enabled] if _styled(digit) else []
    if not drawn:
        near = abs(passes - (1 - ALT_PLAIN / 255)) <= ALT_PLAIN_NEAR
        return (ALT_PLAIN if near else round((1 - passes) * 255)), None
    over = drawn[0]
    if len(drawn) > 1 or type(over).__name__ not in ("GradientOverlay", "ColorOverlay") or over.blend_mode != BlendMode.NORMAL \
            or over.opacity >= ALT_COVER:
        return 255, None
    p = over.opacity / 100
    solid = 1 - (1 - p) * passes
    return round(solid * 255), p / solid * 100


def set_overlay(bar, percent: float) -> None:
    """A bar's enabled overlay at percent."""
    for effect in bar.effects:
        if type(effect).__name__ in ("GradientOverlay", "ColorOverlay") and effect.enabled:
            effect.descriptor[b"Opct"].value = percent
    bar._psd.mark_updated()


def _alt_darks(psd: PSDImage, template: PSDImage, found: tuple) -> list:
    """The card's Darks group of exposure digit rows replaced by the template's layer-comp bars, each moved onto the
    card's row and set to its value (psd_edit.set_comp), a styled dark digit's effects on its bar, each bar drawn like
    its digit (alt_dark)."""
    darks, parts, drop = found
    source = next(group for group in template.descendants()
                  if group.is_group() and group.name.strip() == "Darks" and any(comp_value(l) is not None for l in group))
    parent = darks.parent
    new = source.copy_to(parent, list(parent).index(darks) + 1)
    bars = {comp_stat(layer): layer for layer in new if comp_value(layer) is not None}
    if set(bars) != set(parts):
        raise Refused("unsupported-card", "the template's Darks group is not one bar per stat")
    looks = {stat: alt_dark(digit) for stat, (_, _, digit) in parts.items()}
    own = any(opacity != ALT_PLAIN for opacity, _ in looks.values())
    if own:                                     # the darks per bar
        new.opacity = 255
    for stat, (value, box, digit) in parts.items():
        bar = bars[stat]
        psd_edit.place_bar(bar, box)
        psd_edit.set_comp(psd, {"card": {key: comp_value(b) for key, b in bars.items()}}, stat, value)
        opacity, overlay = looks[stat]
        if own:
            bar.opacity = opacity
        if digit is not None and _styled(digit):
            bar.copy_effects_from(digit)
        if overlay is not None:
            set_overlay(bar, overlay)
    parent.remove(darks)
    for group in drop:
        group.parent.remove(group)
    return [f"{bars[stat].name} {value}" for stat, (value, _, _) in parts.items()]


ALT_BAR_AT = {"hp": (360, 1200), "def": (360, 1326), "atk": (360, 1452), "spd": (360, 1578)}   # the old template's bars
ALT_SHAPE = {"hp": "Health", "def": "Defense", "atk": "Attack", "spd": "Speed"}                  # the template's shapes


def _alt_shapes(psd: PSDImage, template: PSDImage) -> list:
    """The old template's bar art (a pixel bar at its place, coloured by its effects) replaced by the template's vector
    bar shapes, the card's effects on them."""
    bars = next((g for g in psd.descendants() if g.is_group() and g.name.strip() == "Bars"
                 and g.parent is not None and not isinstance(g.parent, PSDImage) and g.parent.name.strip() == "Stats Bars"), None)
    shapes = {layer.name.strip(): layer for group in template.descendants() if group.is_group() and group.name.strip() == "Bars"
              for layer in group if layer.kind == "shape"}
    done = []
    for layer in list(bars or []):
        stat = next((k for k, pattern in STAT_PATTERNS.items() if fullmatch(rf"\s*({pattern})\s*", layer.name, IGNORECASE)), None)
        if (stat is None or layer.kind != "pixel" or not layer.has_effects() or tuple(layer.bbox[:2]) != ALT_BAR_AT[stat]
                or abs(layer.bbox[2] - layer.bbox[0] - ALT_ROW[0]) > 3 or abs(layer.bbox[3] - layer.bbox[1] - ALT_ROW[1]) > 2
                or ALT_SHAPE[stat] not in shapes):
            continue
        shape = shapes[ALT_SHAPE[stat]].copy_to(bars, list(bars).index(layer) + 1)
        shape.copy_effects_from(layer)
        shape.visible = layer.visible
        bars.remove(layer)
        done.append(ALT_SHAPE[stat])
    return [f"bar shapes: {', '.join(done)}"] if done else []


def _alt_text(psd: PSDImage, template: PSDImage = None) -> list:
    """The old template's text layout as the template's: "Stat Txt" named "Stat Text and Name", the Name layers moved
    into it."""
    text = next((l for l in psd if l.is_group() and l.name.strip() == "Text"), None)
    stats = next((g for g in text if g.is_group() and g.name.strip() == "Stat Txt"), None) if text is not None else None
    names = [l for l in text if l.kind == "type" and base_name(l).strip() == "Name"] if text is not None else []
    if stats is None or not names:
        return []
    stats.name = "Stat Text and Name"
    for name in names:                          # a split title's parts together, in their order
        name.move_to(stats)
    return ["text layout"]


ALT_CLIPS = {"Clip Ability Backround to this": ((0, 1721, 1200, 2400), "Clip Ability Backround to this"),
             "Clip Main image to this": ((0, 0, 1200, 1731), "Clip Main Image to this")}   # old base: (its box, the template's shape)


def _alt_clips(psd: PSDImage, template: PSDImage) -> list:
    """The old template's clip bases in images (pixel layers at their place, LINE beside them) swapped for the
    template's vector shapes, each where the old one was (the art clipped to it stays clipped)."""
    images = next((l for l in psd if l.is_group() and l.name.strip() == "images"), None)
    kids = {l.name.strip(): l for l in images} if images is not None else {}
    if "LINE" not in kids or not all(n in kids and kids[n].kind == "pixel" and tuple(kids[n].bbox) == box
                                     for n, (box, _) in ALT_CLIPS.items()):
        return []
    shapes = {l.name: l for g in template if g.is_group() and g.name == "images" for l in g if l.kind == "shape"}
    for old_name, (_, shape_name) in ALT_CLIPS.items():
        old = kids[old_name]
        shapes[shape_name].copy_to(images, list(images).index(old))
        images.remove(old)
    return ["clip shapes"]


ALT_STEP_OFF = 24           # a step past the dark cells is kept only where no pixel of the card moves past this


def alt_template(path: str, out_path: str, fonts) -> list:
    """
    A card onto the template (fonts.template, its PSD) in one rewrite: its dark cells as layer-comp bars (_alt_darks),
    and where the card renders the same with them (each tried alone, ALT_STEP_OFF), its bar art as the template's bar
    shapes (_alt_shapes), its text layout (_alt_text) and its clip bases as the template's shapes (_alt_clips). Gate:
    every layer none of them touched as it was (_alt_gate).

    Returns:
        what it did; [] when the card's bars are not the template's rows
    """
    template = getattr(fonts, "template", None)
    psd = PSDImage.open(path)
    found = _alt_parts(psd)
    if not found or not template:
        return []
    tmpl = PSDImage.open(template)
    base = np.asarray(PSDImage.open(path).composite().convert("RGB"), np.int16)
    steps = []
    for step in (_alt_shapes, _alt_text, _alt_clips):
        trial = PSDImage.open(path)
        if step(trial, tmpl) and (np.abs(np.asarray(trial.composite().convert("RGB"), np.int16) - base).max() <= ALT_STEP_OFF):
            steps.append(step)
    names = _alt_darks(psd, tmpl, found)
    for step in steps:
        names += step(psd, tmpl)
    psd_edit.save_psd(psd, out_path)
    _alt_gate(path, out_path)
    return names


def _orphan_links(psd: PSDImage) -> set:
    """The smart-object documents, embedded or linked (lnkE), no layer of psd uses."""
    used = {str(layer.smart_object.unique_id).strip("\x00") for layer in psd.descendants() if layer.kind == "smartobject"}
    links = set()
    for tag in (Tag.LINKED_LAYER1, Tag.LINKED_LAYER2, Tag.LINKED_LAYER3, Tag.LINKED_LAYER_EXTERNAL):
        links |= {str(item.uuid).strip("\x00") for item in (psd.tagged_blocks.get_data(tag) or [])} \
            if psd.tagged_blocks and tag in psd.tagged_blocks else set()
    return links - used


def _alt_gate(path: str, out_path: str) -> None:
    """Gate: the only layers gone are the ones the conversion replaces (the Darks rows, the old bar art, the old clip
    bases), and every other layer's record is as it was (by layer id; "Stat Txt" renamed); raises Refused("bytes")."""
    def by_id(psd: PSDImage) -> dict:
        records = psd_edit.layer_records(psd)
        return {layer.layer_id: (records[psd_edit._record_index(layer)], layer) for layer in psd.descendants()}
    before, after = PSDImage.open(path), PSDImage.open(out_path)
    if _orphan_links(after) - _orphan_links(before):    # Photoshop 2023 won't open a file holding them
        raise Refused("bytes", "a smart object's embedded document is left with no layer using it")
    old, new = by_id(before), by_id(after)
    found = _alt_parts(before)
    replaced = {id(layer) for group in ([found[0], *found[2]] if found else []) for layer in [group, *group.descendants()]}
    for layer in before.descendants():
        parent = layer.parent
        where = parent.name.strip() if parent is not None and not isinstance(parent, PSDImage) else ""
        if (where == "Bars" and layer.kind == "pixel") or (where == "images" and layer.name.strip() in ALT_CLIPS):
            replaced.add(id(layer))
    for lid, (a, layer) in old.items():
        if lid not in new:
            if id(layer) not in replaced:
                raise Refused("bytes", f"{layer.name!r} went missing")
            continue
        b = new[lid][0]
        if a != b and not (layer.name.strip() == "Stat Txt" and {k for k in a if a[k] != b[k]} <= {"name", "blocks"}):
            raise Refused("bytes", f"{layer.name!r} changed beyond the conversion")


# ========================
# Normalise (the big pass, owner 2026-11-07): sub and superscript text to normal, the pictures linked, no merged image
# ========================
SCRIPT_RATIOS = {1: ("SuperscriptSize", "SuperscriptPosition"),   # FontBaseline: the ResourceDict's size and position
                 2: ("SubscriptSize", "SubscriptPosition")}
SCRIPT_DEFAULT = (0.583, 0.333)  # a ResourceDict without them: Photoshop's size and position
AUTO_LEADING = 1.2          # a paragraph's AutoLeading when its sheets set none
BOX_LO, BOX_HI, BOX_TIGHT = 0.003, 0.018, 1.02   # the box-width rule (a redo): room past the widest line and before
                                                 # the next word; a box this tight keeps its width


def _v(value):
    return getattr(value, "value", value)


def _runs(layer):
    from psd_tools.composite.text.model import TextModel
    return TextModel.from_layer(layer)


def script_layers(psd: PSDImage) -> list:
    """The type layers holding a sub or superscript run."""
    out = []
    for layer in psd.descendants():
        if layer.kind != "type":
            continue
        try:
            if any(run.end > run.start and run.style.baseline in SCRIPT_RATIOS for run in _runs(layer).runs):
                out.append(layer)
        except Exception:                       # unreadable runs: left alone
            continue
    return out


def _leading_ratio(layer) -> Callable[[int], float]:
    """A character index -> its paragraph's AutoLeading ratio."""
    from bisect import bisect_right
    from psd_tools.api import text_edit
    engine, resources = layer.engine_dict, layer.resource_dict
    try:
        sheet = resources["ParagraphSheetSet"][int(_v(resources.get("TheNormalParagraphSheet", 0)))]["Properties"]
        base = float(_v(sheet.get("AutoLeading", AUTO_LEADING)))
    except Exception:
        base = AUTO_LEADING
    runs = engine.get("ParagraphRun", {})
    ends, values, unit = [], [], 0
    for item, n in zip(runs.get("RunArray", []), runs.get("RunLengthArray", [])):
        unit += int(_v(n))
        ends.append(unit)
        values.append(float(_v(item.get("ParagraphSheet", {}).get("Properties", {}).get("AutoLeading", base))))
    units = text_edit.utf16_prefix(text_edit._editor_text(layer))
    return lambda i: values[min(bisect_right(ends, units[i]), len(values) - 1)] if values else base


def script_plan(layer) -> list:
    """
    [(start, end, style keys)] that set each sub or superscript run normal at the same size on screen and the same
    place (proofs/subscript.py): FontSize S*k, FontBaseline 0, Leading fixed at the line pitch it had, BaselineShift
    less sign*p*S*VerticalScale (sign +1 sub, -1 super), k and p the layer's Sub- or SuperscriptSize and Position.
    Tracking, kerning and HorizontalScale are per drawn em: unchanged.
    """
    resources, ratio, out = layer.resource_dict, _leading_ratio(layer), []
    for run in _runs(layer).runs:
        style = run.style
        if run.end <= run.start or style.baseline not in SCRIPT_RATIOS:
            continue
        k, p = (float(_v(resources.get(key, default)))
                for key, default in zip(SCRIPT_RATIOS[style.baseline], SCRIPT_DEFAULT))
        sign = 1.0 if style.baseline == 2 else -1.0
        leading = ratio(run.start) * style.size if style.auto_leading else style.leading
        shift = style.baseline_shift - sign * p * style.size * style.vertical_scale
        out.append((run.start, run.end, {"FontBaseline": 0, "FontSize": style.size * k, "AutoLeading": False,
                                         "Leading": leading, "BaselineShift": shift}))
    return out


def to_normal(layer, fonts, widen: bool = False) -> Optional[str]:
    """
    One layer's sub and superscript runs set normal in place (TySh and the Txt2 story, its stale story synced first;
    the stored pixels stay). Box text also rises by d, how far its first baseline moves when those runs shrink (it sits
    one ascent of the line's largest run below the box top), on every run; widen: the box-width rule too (a redo, when
    Photoshop re-breaks the converted box). Checked: the fork lays every glyph where it was (psd_edit.GLYPH_TOL; proofs
    P1: 225 of 225 layers).

    Args:
        fonts: The fork's FontMap (psd_edit.layout_fonts)

    Returns:
        Why the conversion fails (the layer is left converted: the caller opens the PSD again without it), or None
    """
    from psd_tools.composite.text import font_names, layout
    edits = script_plan(layer)
    if not edits:
        return None
    model = _runs(layer)
    lost = [name for name in font_names(layer) if model.shape == 1 and fonts.find(name) is None]
    if lost:                                    # d would be a stand-in face's
        return f"its font {lost[0]} is missing"
    layer.sync_story()
    model = _runs(layer)
    before, was = psd_edit.glyph_places(layer, fonts), layout.layout(model, fonts)
    for start, end, keys in edits:
        layer.set_span_style(start, end, **keys)
    if model.shape == 1:
        now = layout.layout(_runs(layer), fonts)
        rise = was.lines[0].y - now.lines[0].y if was.lines and now.lines else 0.0
        if abs(rise) > 1e-9:
            for run in _runs(layer).runs:
                if run.end > run.start:
                    layer.set_span_style(run.start, run.end, BaselineShift=run.style.baseline_shift - rise)
        if widen:
            _fit_box(layer, fonts)
    moved = not psd_edit.same_places(before, psd_edit.glyph_places(layer, fonts), psd_edit.GLYPH_TOL)
    return "its glyphs would move" if moved else None


def _slack(layer, fonts) -> Optional[tuple]:
    """(lo, hi): the fork breaks the box's lines the same for any width in [lo, hi) (lo the widest line, hi the
    narrowest line plus its next word); None for a line it compresses or breaks inside a word, or no box."""
    from psd_tools.composite.text.compose import break_lines
    from psd_tools.composite.text.layout import frame, pieces
    from psd_tools.composite.text.shape import shape_piece
    model = _runs(layer)
    box = frame(model)
    if model.shape != 1 or box is None:
        return None
    text, lo, hi, aligned = model.text, 0.0, float("inf"), set()
    for piece in pieces(model):
        paragraph = model.paragraphs[piece.paragraph]
        shaped = shape_piece(text, piece.start, piece.end, model.runs, fonts, sub_size=model.sub_size,
                             tab_width=paragraph.tab_width)
        lines = break_lines(text, piece.start, piece.end, shaped.measure, sorted(shaped.space_adv), box[2],
                            paragraph.justification, None)
        for k, line in enumerate(lines):
            if line.decision in ("compress", "break"):
                return None
            if line.end > line.start:
                aligned.add(paragraph.justification)
                lo = max(lo, shaped.measure(line.start, line.end))
            if line.decision == "fit" and k + 1 < len(lines):
                j = lines[k + 1].start
                while j < lines[k + 1].end and text[j] not in " \t":
                    j += 1
                hi = min(hi, shaped.measure(line.start, j))
    return (lo, hi, aligned) if lo and hi != float("inf") else None


def _fit_box(layer, fonts) -> None:
    """The box-width rule (proofs P1, a redo only): a box width W' in the slack, max(lo (1 + BOX_LO), min(W,
    hi / (1 + BOX_HI))), W kept when hi / lo < BOX_TIGHT; whole px, the side the paragraphs align to kept. Rotated or
    offset boxes and mixed alignment keep theirs."""
    found = _slack(layer, fonts)
    model = _runs(layer)
    xx, xy, yx, _, _, _ = model.transform
    if found is None or abs(xy) > 1e-6 or abs(yx) > 1e-6 or any(abs(v) > 1e-6 for v in model.box[:2]):
        return
    lo, hi, aligned = found
    if len(aligned) != 1 or not aligned <= {0, 1, 2} or hi / lo < BOX_TIGHT:
        return
    _, _, width, height = model.box
    side = next(iter(aligned))
    step = 2 if side == 2 else 1                 # centred: a px each side
    n = round((width - max(lo * (1 + BOX_LO), min(width, hi / (1 + BOX_HI)))) * xx / step)
    if n:
        x0, y0 = model.transform[4] + {0: 0, 1: n * step, 2: n}[side], model.transform[5]
        psd_edit.set_text_box(layer, (x0, y0, x0 + (width - n * step / xx) * xx, y0 + height * model.transform[3]),
                              pixels=False)


NORMAL_NOTE = "The {name} text still has subscript or superscript in it: set it to normal text in Photoshop."   # note
LINK_BLOCKS = frozenset(bytes(tag) for tag in (Tag.LINKED_LAYER1, Tag.LINKED_LAYER2, Tag.LINKED_LAYER3,
                                               Tag.LINKED_LAYER_EXTERNAL))   # the smart objects' documents


class Normalised(list):
    """normalise()'s names, and what else it did: files (repo path -> bytes; None deletes it), the layers it drew
    again (a link's raster redrawn: the PNG follows), what it left and why (plan.json), hand notes (one plain sentence
    each)."""

    def __init__(self, *args):
        super().__init__(*args)
        self.files, self.redraws, self.left, self.notes = {}, [], [], []


def _links():
    """commands.psd_links, the link step (big pass A4): needs(psd) -> [what to link or repair]; link(psd, rel, fonts)
    -> Normalised, linking in place. None until it is in the tree: normalise sets text normal and strips alone."""
    try:
        from commands import psd_links
    except ImportError:
        return None
    return psd_links if hasattr(psd_links, "needs") and hasattr(psd_links, "link") else None


def needs_normalise(psd: PSDImage) -> list:
    """What normalise() would change: the sub and superscript type layers, the PSD itself for its merged image, and
    the link step's finds (an embedded picture, a link needing repair, a linked stat bar)."""
    links = _links()
    return [*script_layers(psd), *([psd] if psd.has_preview() else []), *(links.needs(psd) if links else [])]


def _layout_fonts(fonts, layer):
    """psd_edit.layout_fonts over the layer's faces' folder (fonts gives it), else the environment's."""
    found = fonts(layer_fonts(layer)) if callable(fonts) else ""
    return psd_edit.layout_fonts(found or "")


def normalise(path: str, out_path: str, fonts, rel: str = "") -> Normalised:
    """
    The big pass's rewrite, in one save (owner 2026-11-07): the held cards' bars lined up (fonts.held(psd, rel), the
    sweep's alone), sub and superscript text set normal (to_normal; widened boxes: fonts.widen, layer ids; none with
    fonts.keep_text, the reason: the sweep's fallback), the pictures linked (the link step, _links()), the merged image
    stripped (psd_edit.save_psd). Gated: the converted layers' records changed in TySh alone, the stories by
    gate_stories (a widened box's own frames too), every other layer as it was (but the link step's), no smart-object
    document left without a layer.

    Returns:
        Normalised; empty when there was nothing to do (nothing written)
    """
    widen, kept = set(getattr(fonts, "widen", ())), getattr(fonts, "keep_text", "")
    psd, failed = PSDImage.open(path), {}
    texts = script_layers(psd)
    if kept:                                    # 5.2 step 9: the text left as it is (kept: why)
        failed = {layer.layer_id: (layer.name.strip(), kept) for layer in texts}
        texts = []
    for layer in texts:
        why = to_normal(layer, _layout_fonts(fonts, layer), layer.layer_id in widen)
        if why:
            failed[layer.layer_id] = (layer.name.strip(), why)
    if failed:                                  # opened again: the layers that would move stay as they are
        psd = PSDImage.open(path)
        texts = [layer for layer in script_layers(psd) if layer.layer_id not in failed]
        for layer in texts:
            to_normal(layer, _layout_fonts(fonts, layer), layer.layer_id in widen)
    texts = [layer for layer in texts if layer.layer_id not in failed]
    done = Normalised(layer.name.strip() for layer in texts)
    done.left = [f"{name}: {why}" for name, why in failed.values()]
    names = list(dict.fromkeys(name for name, _ in failed.values()))
    done.notes = [NORMAL_NOTE.format(name=", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0])
                  ] if names else []                # one sentence a card (owner 11-07)
    held = getattr(fonts, "held", None)
    bars = held(psd, rel) if held is not None else []
    done += bars
    done.notes += getattr(bars, "notes", [])
    links, linked = _links(), Normalised()
    if links is not None:
        linked = links.link(psd, rel, fonts)
        done += linked
        done.files.update(linked.files)
        done.redraws += linked.redraws
        done.left += linked.left
        done.notes += linked.notes
    if psd.has_preview():
        done.append("merged image")
    if not done:
        return done
    psd_edit.save_psd(psd, out_path)
    stories = {layer.layer_id: psd_edit.story_index(layer) for layer in texts}
    _normal_gate(path, out_path, {psd_edit._record_index(layer) for layer in texts},
                 [i for i in stories.values() if i is not None], bool(linked or bars),
                 [i for lid, i in stories.items() if lid in widen and i is not None])
    return done


def _normal_gate(path: str, out_path: str, texts: set, stories: list, others: bool, boxes: list = ()) -> None:
    """normalise()'s gate: the text records changed in TySh alone, every other record as it was unless the link or
    held-bar steps ran (others: theirs are checked by the pass's own links check), the document but its merged image,
    the stories (gate_stories, the link blocks free when others; boxes: the widened stories, their frames free), no
    orphaned smart-object document; Refused("bytes")."""
    old_records, new_records = psd_edit.layer_records(path), psd_edit.layer_records(out_path)
    if len(old_records) != len(new_records):
        raise Refused("bytes", "layer count changed")
    for i, (a, b) in enumerate(zip(old_records, new_records)):
        if a == b:
            continue
        if i not in texts:
            if others:
                continue
            raise Refused("bytes", f"{a['name']}: changed beyond normal text")
        if {k for k in a if a[k] != b[k]} - {"blocks"} or \
                [k for (k, x), (_, y) in zip(a["blocks"], b["blocks"]) if x != y and k != b"TySh"]:
            raise Refused("bytes", f"{a['name']}: changed beyond its text's style")
    before, after = PSDImage.open(path), PSDImage.open(out_path)
    part = psd_edit.changed_part(before, after, ("header", "color_mode_data", "image_resources"))
    if part:
        raise Refused("bytes", f"{part} changed")
    psd_edit.gate_stories(before, after, stories, also=LINK_BLOCKS if others else frozenset(), frames=boxes)
    if _orphan_links(after) - _orphan_links(before):    # Photoshop 2023 won't open a file holding them
        raise Refused("bytes", "a smart object's document is left with no layer using it")


# ========================
# Status words (owner 2026-11-07: the chart's colour and FX layer, silent, on every push)
# ========================
NOTE_LAYER = re_compile(r"notes?(?: copy(?: \d+)?)?", IGNORECASE)    # a note text layer's name (MDW's "Note copy 3")
STATUS_LEVELS = 1           # a status word's fill this many levels (of 255) off the chart's still matches it
BLANKS = " \t\r\x03"


def status_texts(psd: PSDImage) -> list:
    """The text layers status words are set in: the Ability's (ability_layers: every picture) and the visible note
    layers; never a title or stat label."""
    out = ability_layers(psd)
    pictures = variants_of(psd)
    for n in (pictures if len(pictures) > 1 else [None]):
        with variant_view(psd, n):
            out += [layer for layer in psd.descendants() if psd_edit._is_text_layer(layer) and layer.is_visible()
                    and EFFECT_LAYER_TAG not in layer.name and NOTE_LAYER.fullmatch(base_name(layer).strip())
                    and layer not in out]
    return out


def _chart_fill(fill, want) -> bool:
    return fill is not None and max(abs(round(a * 255) - round(b * 255)) for a, b in zip(fill, want)) <= STATUS_LEVELS


def status_plan(psd: PSDImage, chart: Optional[dict]) -> list:
    """
    [(layer, spans, fresh, restyle)] per status_texts layer with work: spans, its chart status words (the bake's
    psd_edit.status_word_spans: capitalised, longest name first) whose letters are off the chart's fill (STATUS_LEVELS);
    fresh, the styled ones no FX layer holds whole (psd_edit.new_fx, the bake's names); restyle, [(FX layer, status)]
    on a chart status word whose style draws otherwise (psd_edit.chart_style, owner 2026-11-07). Left alone: a word an
    overlay word layer prints (R5's), a layer with an FX layer that is no pixel layer (the bake refuses it), a word
    partly under another FX layer (new_fx's told case). No chart: [].
    """
    if not chart:
        return []
    out, printed = [], None
    for layer in status_texts(psd):
        text = layer.text or ""
        spans = psd_edit.status_word_spans(text, chart, psd_edit.icon_spans(layer))
        if not spans:
            continue
        if printed is None:
            printed = {psd_edit.collapse(word.text).lower() for word in overlay_words(psd)}
        spans = [span for span in spans if span[2].name.lower() not in printed]
        fxs = psd_edit.fx_layers(psd, layer)
        if not spans or any(fx.kind != "pixel" for fx in fxs):
            continue
        fills = psd_edit._char_styles(layer)[0]
        off = [(a, b, status) for a, b, status in spans
               if not all(_chart_fill(fills[i], status.fill) for i in range(a, b) if text[i] not in BLANKS)]
        held = [span for fx in fxs for span in psd_edit.phrase_ranges(text, psd_edit.fx_phrases(layer, fx))]
        fresh = psd_edit.new_fx(text, [span for span in spans if span[2].styled], held, lambda code, why: None)
        restyle = [(fx, status) for fx in fxs for status in [psd_edit.chart_style(layer, fx, text, spans)]
                   if status and psd_edit._style_key(fx.tagged_blocks) != psd_edit._style_key(status.blocks)]
        if off or fresh or restyle:
            out.append((layer, off, fresh, restyle))
    return out


def status_word_layers(psd: PSDImage) -> list:
    """status_plan's layers, the chart from psd.status_chart (pixel_kinds sets it)."""
    return [layer for layer, *_ in status_plan(psd, getattr(psd, "status_chart", None))]


def status_words(path: str, out_path: str, fonts) -> list:
    """
    Status words (owner 2026-11-07), the PIXEL_FIXES kind, by the bake's own ability-edit steps (set_text_layer): each
    status_plan word off the chart set in the chart's fill (the chart always wins), its glyphs redrawn (a splice), an FX
    layer on it cut again in place (_recut_fx: Update all); each styled one no FX layer holds gets one
    (psd_edit._regenerate_fx: Effect Layers.jsx's name and place, the chart's effect blocks); an FX layer on one takes
    the chart's style (psd_edit.restyle_fx, owner 2026-11-07; another phrase's keeps its own, q6). fonts.chart:
    psd_edit.status_fills of the chart. Gated (_status_gate), then checked: nothing left to do, every status FX layer
    on its own words (gate_status_words).

    Returns:
        The text layers' names, then the FX layers made or restyled; raises Refused or the engine's errors
    """
    from psd_tools.composite.text import capture
    chart = getattr(fonts, "chart", None)
    psd = PSDImage.open(path)
    plan = status_plan(psd, chart)
    if not plan:
        return []
    engine, done, stories = psd_edit._fx_engine(), [], []
    for layer, spans, fresh, restyle in plan:
        faces, cut, made = _fonts_of(layer, fonts), [], []
        if spans:
            psd_edit.follow_kerning(psd, layer)
            keep = capture(layer, faces)
            for a, b, status in spans:
                layer.set_span_style(a, b, fill=status.fill)
            _by_glyph(layer, layer.rasterize(fonts=faces, keep=keep, unit=psd_edit.SPLICE_UNIT))
            offset = engine[0].from_layer(layer).text.find(layer.text)
            cut = [fx for fx, _ in _recut_fx(psd, layer, {i + offset for a, b, _ in spans for i in range(a, b)}, faces)]
            if psd_edit.story_index(layer) is not None:
                stories.append(psd_edit.story_index(layer))
        if fresh:
            made = psd_edit._regenerate_fx(psd, layer, layer.text, fresh, faces, engine)
        styled = [(fx, psd_edit.restyle_fx(fx, status)) for fx, status in restyle]
        done.append((layer, bool(spans), cut, made, styled))
    edits = [psd_edit.Edit("ability", layer.text, layer.text, [layer], [psd_edit._record_index(layer)] if set_ else [],
                           None, fx=[(fx.name, {bytes(key): fx.tagged_blocks[key].tobytes()
                                                for key in psd_edit.EFFECT_BLOCKS if key in fx.tagged_blocks})
                                     for fx in made],
                           rebuilt=[(psd_edit._record_index(fx), fx.name, fx.name) for fx in cut],
                           restyled=[(psd_edit._record_index(fx), blocks) for fx, blocks in styled])
             for layer, set_, cut, made, styled in done]
    psd_edit.save_psd(psd, out_path)
    _status_gate(path, out_path, edits, stories)
    after = PSDImage.open(out_path)
    if status_plan(after, chart):
        raise Refused("bytes", "a status word is left off the chart")
    for layer in status_texts(after):
        psd_edit.gate_status_words(after, chart, layer)
    return [layer.name.strip() for layer, *_ in done] + [fx.name for *_, made, styled in done
                                                         for fx in made + [fx for fx, _ in styled]]


def status_font_layers(psd: PSDImage) -> list:
    """status_font.plan's layers, the repo path and the chart from psd.repo_path and psd.status_chart (pixel_kinds)."""
    from commands import status_font as sf
    return sf.plan(psd, getattr(psd, "repo_path", ""), getattr(psd, "status_chart", None))


def restyle_status(path: str, out_path: str, fonts, rel: str) -> list:
    """Status font (owner 2026-11-09), the PIXEL_FIXES kind: status_font.fix (fonts.chart: the status chart's fills).

    Returns:
        The text layers' names; raises Refused or the engine's errors
    """
    from commands import status_font as sf
    folder = fonts([sf.KP, ICON_FONT]) if fonts else ""
    if not folder:
        raise Refused("unsupported-card", "no fonts to draw Krabby Patty")
    return sf.fix(path, out_path, folder, getattr(fonts, "chart", None), rel)


def _status_gate(path: str, out_path: str, edits: list, stories: list) -> None:
    """status_words' gate: the records as the bake's gate_bytes allows its Edits (the FX layers made or cut again), a
    recoloured text record changed in its pixels and TySh alone, its story alone (gate_stories), the document as it was
    but what a made FX layer moves (ADDED_RESOURCES); Refused("bytes")."""
    old, new = psd_edit.layer_records(path), psd_edit.layer_records(out_path)
    psd_edit.gate_bytes(old, new, edits)
    made, pairs = Counter(name for edit in edits for name, _ in edit.fx), {}
    for j, record in enumerate(new):            # new index -> old, the made FX records taken out
        if made[record["name"]]:
            made[record["name"]] -= 1
        else:
            pairs[j] = len(pairs)
    for index in {index for edit in edits for index in edit.records}:
        a, b = old[pairs[index]], new[index]
        tysh = bytes(Tag.TYPE_TOOL_OBJECT_SETTING)
        if a["name"] != b["name"] or {k for k in a if a[k] != b[k]} - {"channels", "rect", "blocks"} or \
                [k for (k, x), (_, y) in zip(a["blocks"], b["blocks"]) if x != y and k != tysh]:
            raise Refused("bytes", f"{a['name']}: changed beyond its text's fill")
    before, after = PSDImage.open(path), PSDImage.open(out_path)
    part = psd_edit.changed_part(before, after, ("header", "color_mode_data"))
    added = any(edit.fx for edit in edits)
    if part or _resources(before, after.has_preview(), added) != _resources(after, after.has_preview(), added):
        raise Refused("bytes", f"{part or 'image_resources'} changed")
    psd_edit.gate_stories(before, after, stories)


ADDED_RESOURCES = (Resource.IDS_SEED_NUMBER, Resource.LAYER_STATE_INFO, Resource.LAYER_GROUP_INFO,
                   Resource.LAYER_GROUPS_ENABLED_ID)    # a layer added moves the id seed and the per-record lists (the fork)


def _resources(psd: PSDImage, preview: bool, added: bool) -> list:
    """The image resources, with no preview the merged image flag off (save_psd strips it); added: an FX layer made,
    ADDED_RESOURCES left out."""
    out = []
    for key, value in psd._record.image_resources.items():
        if added and key in ADDED_RESOURCES:
            continue
        if key == Resource.VERSION_INFO and not preview:
            value = deepcopy(value)
            value.data.has_composite = False
        out.append((key, psd_edit._raw(value)))
    return out


# ========================
# Clean (owner 2026-11-09): hidden leftovers and data that draws nothing, on cards; every picture and the parse the same
# ========================
CLEAN_ICON = 0.03           # share of the card: a star or type icon is smaller (a 100 px icon is 0.5%)
STAR_GROUP = re_compile(r"\bstars?\b", IGNORECASE)
TYPE_GROUP = re_compile(r"types", IGNORECASE)
STAT_GROUP = re_compile(r"\b(hp|health|def|defen[cs]e|atk|attack|spd|speed|darks?|lit|stats?|bars?)\b", IGNORECASE)
ANCESTORS = re_compile(rb"(?<!\s)\s*<photoshop:DocumentAncestors>.*?</photoshop:DocumentAncestors>", DOTALL)  # (?<!\s): linear
PATTERN_TAGS = (Tag.PATTERNS1, Tag.PATTERNS2, Tag.PATTERNS3)
FX_TAGS = (Tag.FILTER_EFFECTS1, Tag.FILTER_EFFECTS2, Tag.FILTER_EFFECTS3)
LEFT_FILLS = frozenset(("solidcolorfill", "gradientfill", "patternfill"))
LEFT_LINES = frozenset(("line if needed", "line"))     # hidden shapes that go, by name (owner 2026-11-09)
LEFT_STAT = re_compile(r"\d{1,2}|hp|health|def|defen[cs]e|atk|attack|spd|speed|darks?|lit|stats?( bars?)?|bars?|stat txt"
                       r"|.*\b(dark|lit)\b.*", IGNORECASE)     # stat bars and digits: kept (the parser reads them)
LEFT_WORDS = 60             # characters: a hidden text layer longer is an Ability copy (kept)
LEFT_TITLE = 0.12           # share of the card's height: hidden text above it is a title copy


@dataclass(frozen=True)
class Hidden:
    """clean's leftovers (owner 2026-11-09, the guards'): the linked files (lowercased repo paths) another PSD in scope
    shows, and dups: the card has picture-only duplicates, drawn from its hidden layers (no leftover goes)."""
    elsewhere: frozenset = frozenset()
    dups: bool = False


def _shown_anywhere(psd: PSDImage) -> set:
    """ids of the nodes some picture shows (the saved one or a variant's)."""
    found = {id(layer) for layer in psd.descendants() if layer.is_visible()}
    for n in variants_of(psd):
        with variant_view(psd, n):
            found |= {id(layer) for layer in psd.descendants() if layer.is_visible()}
    return found


def _groups(layer) -> list:
    out, parent = [], layer.parent
    while parent is not None and not isinstance(parent, PSDImage):
        out.append(parent.name)
        parent = parent.parent
    return out


def _link_kind(layer) -> Optional[str]:
    """"star" or "type" for a smart object linked to Types/Stars or Types (judged from its document's repo_path)."""
    if layer.is_group() or layer.kind != "smartobject":
        return None
    try:
        target = layer.smart_object.link_target(getattr(layer._psd, "repo_path", "") or None) or ""
    except (ValueError, KeyError, AttributeError):
        return None
    return "star" if target.startswith("Types/Stars/") else "type" if target.startswith("Types/") else None


def _clean_kind(layer, psd: PSDImage) -> Optional[str]:
    """"star" or "type" for a star or type icon leaf: linked to Types/Stars or Types, or an icon-sized picture in a Stars
    or Types group or named a star; never a stat bar or a layer in a stat group (Star Bars' cells)."""
    if layer.is_group() or layer.kind not in ("pixel", "smartobject", "shape") or comp_value(layer) is not None:
        return None
    groups = _groups(layer)
    if any(STAT_GROUP.search(name) for name in groups):
        return None
    linked = _link_kind(layer)
    if linked:
        return linked
    box = layer.bbox
    if box[2] <= box[0] or (box[2] - box[0]) * (box[3] - box[1]) > CLEAN_ICON * psd.width * psd.height:
        return None
    if any(TYPE_GROUP.search(name) for name in groups):
        return "type"
    if any(STAR_GROUP.search(name) for name in groups) or "star" in sub(r"[^a-z]+", " ", base_name(layer).lower()).split():
        return "star"
    return None


def _clipped_on(layer) -> list:
    """The clipping layers right above a base layer in its group (they draw nothing once it is hidden)."""
    if layer.is_group() or layer.clipping:
        return []
    siblings = list(layer.parent)
    out = []
    for above in siblings[siblings.index(layer) + 1:]:
        if not above.clipping:
            break
        out.append(above)
    return out


def _layer_ink(layer):
    """psd_links.ink_key of a layer's own pixels (clean's copies), None without any."""
    rgba = layer.numpy()
    if rgba is None or not rgba.size:
        return None
    if rgba.shape[2] == 3:
        rgba = np.dstack([rgba, np.ones(rgba.shape[:2], rgba.dtype)])
    return _links().ink_key(np.clip(np.round(rgba * 255), 0, 255).astype(np.uint8))


def _art_key(layer, rel: str) -> Optional[str]:
    """A picture's content: its linked file (lowercased repo path), its embedded document's blob or its pixel channels'
    sha1; None without one."""
    if layer.kind == "smartobject":
        try:
            found = layer.smart_object
            if found.kind == "external":
                target = found.link_target(rel or None)
                return "file:" + target.lower() if target else None
            data = found.data if found.kind == "data" else None
        except (ValueError, KeyError, AttributeError, NotImplementedError):
            return None
        return "blob:" + sha1(b"blob %d\0" % len(data) + data).hexdigest() if data else None
    if layer.kind != "pixel" or layer.bbox[2] <= layer.bbox[0]:
        return None
    digest = sha1(repr(tuple(layer.size)).encode())
    for info, channel in zip(layer._record.channel_info, layer._channels):
        if int(info.id) in (-1, 0, 1, 2):
            digest.update(b"%d:%d:" % (int(info.id), int(channel.compression)) + (channel.data or b""))
    return "px:" + digest.hexdigest()


def _kept(layer) -> bool:
    """A hidden leaf no leftover rule takes: a stat bar, digit or stat group's layer, an FX layer, a star or type icon."""
    return (comp_value(layer) is not None or bool(LEFT_STAT.fullmatch(_key(layer.name))) or EFFECT_LAYER_TAG in layer.name
            or bool(search(r"\bFX\b", layer.name)) or any(STAT_GROUP.search(group) for group in _groups(layer))
            or bool(_link_kind(layer)))


def hidden_links(psd: PSDImage, rel: str = "") -> set:
    """The linked files (lowercased repo paths, Types/ aside) of the hidden pictures a leftover rule may take: where the
    guards look for them on other PSDs (Hidden.elsewhere)."""
    shown = _shown_anywhere(psd)
    keys = {_art_key(layer, rel or getattr(psd, "repo_path", "")) for layer in psd.descendants()
            if layer.kind == "smartobject" and id(layer) not in shown and not _kept(layer)}
    return {key[5:] for key in keys if key and key.startswith("file:") and not key.startswith("file:types/")}


def shown_links(psd: PSDImage, rel: str) -> set:
    """The linked files (lowercased repo paths) some picture of the PSD at rel shows."""
    shown = _shown_anywhere(psd)
    keys = {_art_key(layer, rel) for layer in psd.descendants() if layer.kind == "smartobject" and id(layer) in shown}
    return {key[5:] for key in keys if key and key.startswith("file:")}


def _leftover(layer, psd: PSDImage, keys: set, hidden: Hidden) -> Optional[str]:
    """Why a hidden leaf goes (owner 2026-11-09), None to keep it: an adjustment or fill, a 'Line if needed' or 'LINE'
    shape, a name or title copy or old words (never Ability text, a note or digits), an empty layer, a picture shown on
    the card (keys) or a linked one another PSD shows; never a stat bar, digit, FX layer or star or type icon."""
    name = _key(layer.name)
    if _kept(layer):
        return None
    if layer.kind == "type":
        try:
            text = sub(r"\s+", " ", str(layer.text or "").replace("\x03", " ")).strip()
        except Exception:                       # unreadable text: kept
            return None
        if "abilit" in name or NOTE_LAYER.fullmatch(base_name(layer).strip()) or len(text) > LEFT_WORDS \
                or fullmatch(r"[\d\s]+", text):
            return None
        if not text:
            return "an empty text layer"
        return "a title copy" if search(r"name|title", name) or layer.bbox[1] < LEFT_TITLE * psd.height else "old words"
    if layer.kind in ADJUSTMENTS or layer.kind in LEFT_FILLS:
        return "an adjustment or fill"
    if layer.kind == "shape":
        return "a line shape" if name in LEFT_LINES else None
    if layer.kind not in ("pixel", "smartobject"):
        return None
    if layer.bbox[2] <= layer.bbox[0] or layer.bbox[3] <= layer.bbox[1]:
        return "an empty layer"
    key = _art_key(layer, getattr(psd, "repo_path", "") or "")
    if key is None:
        return None
    if key in keys:
        return "a copy shown on the card"
    return "a copy shown on another PSD" if key.startswith("file:") and key[5:] in hidden.elsewhere else None


def _clip_base(layer):
    """The layer a clipping layer clips onto (None for an orphan clip)."""
    siblings = list(layer.parent)
    return next((one for one in reversed(siblings[:siblings.index(layer)]) if not one.clipping), None)


def _clips(layer) -> list:
    """The clipping layers right above a base, a group's too."""
    if layer.clipping:
        return []
    siblings = list(layer.parent)
    out = []
    for above in siblings[siblings.index(layer) + 1:]:
        if not above.clipping:
            break
        out.append(above)
    return out


def _isolating(layer) -> bool:
    """A clipping layer on a pass-through group: never deleted, hidden too (Photoshop draws a pass-through group anything
    clips onto as a styled one: without the clip it draws differently; Matterling, The Puppet)."""
    if not layer.clipping:
        return False
    base = _clip_base(layer)
    return base is not None and base.is_group() and base.blend_mode == BlendMode.PASS_THROUGH


def _leftovers(psd: PSDImage, shown: set, taken: set, hidden: Hidden) -> list:
    """
    The hidden leftovers' deletions [(layer, why)] (owner 2026-11-09): each top hidden node whose leaves all go (a
    group: an empty one too); a clipping layer alone (never on a pass-through group: _isolating); a base only with
    everything clipped onto it hidden and going too, never one something shown or kept clips onto.
    """
    if hidden.dups:
        return []
    rel = getattr(psd, "repo_path", "") or ""
    tops = [layer for layer in psd.descendants() if id(layer) not in shown and id(layer) not in taken
            and (isinstance(layer.parent, PSDImage) or id(layer.parent) in shown)]
    if not tops:
        return []
    pictures = any(one.kind in ("pixel", "smartobject") for layer in tops
                   for one in ([layer] + (list(layer.descendants()) if layer.is_group() else [])))
    keys = {key for key in (_art_key(layer, rel) for layer in psd.descendants()
                            if id(layer) in shown and layer.kind in ("pixel", "smartobject")) if key} if pictures else set()
    reasons: dict = {}

    def why(node) -> Optional[str]:
        if id(node) not in reasons:
            if id(node) in taken:
                reasons[id(node)] = "taken"
            elif node.is_group():
                leaves = [one for one in node.descendants() if not one.is_group()]
                reasons[id(node)] = "an empty group" if not leaves else "hidden leftovers" if all(
                    id(one) in taken or _leftover(one, psd, keys, hidden) for one in leaves) else None
            else:
                reasons[id(node)] = _leftover(node, psd, keys, hidden)
        return reasons[id(node)]
    out, going = [], set()
    for layer in [one for one in tops if not one.clipping] + [one for one in tops if one.clipping]:
        if id(layer) in going or not why(layer):
            continue
        if layer.clipping:
            base = _clip_base(layer)
            if not _isolating(layer) and (base is None or id(base) not in going):
                out.append((layer, why(layer)))
                going.add(id(layer))
            continue
        clips = _clips(layer)
        if any(id(one) in shown or _isolating(one) or not why(one) for one in clips):
            continue
        out += [(layer, why(layer))] + [(one, "clipped onto it") for one in clips if id(one) not in taken]
        going |= {id(layer)} | {id(one) for one in clips}
    return out


def clean_plan(psd: PSDImage, hidden: Optional[Hidden] = None) -> list:
    """
    clean()'s deletions on a card: [(layer, why)]. A node no picture shows: a group of star or type icons alone, a star
    or type icon leaf (with the clipping layers on it), a picture that is a visible layer's exact copy (the same pixels
    anywhere on the card, owner 2026-11-09), with hidden the leftovers (_leftovers); never a stat bar or a layer some
    variant shows, nor a layer clipped onto a pass-through group (_isolating).
    """
    shown = _shown_anywhere(psd)
    unseen = [layer for layer in psd.descendants() if id(layer) not in shown]
    out, taken = [], set()
    for layer in unseen:
        if id(layer) in taken:
            continue
        if layer.is_group():
            leaves = [one for one in layer.descendants() if not one.is_group()]
            kinds = [_clean_kind(one, psd) for one in leaves]
            if leaves and any(kinds) and all(kind or (one.clipping and one.kind != "type") for kind, one in zip(kinds, leaves)):
                out.append((layer, "hidden stars or type icons"))
                taken |= {id(layer)} | {id(one) for one in layer.descendants()}
            continue
        kind = _clean_kind(layer, psd)
        if kind:
            clipped = _clipped_on(layer)
            out += [(layer, f"hidden {kind}")] + [(one, "clipped onto it") for one in clipped]
            taken |= {id(layer)} | {id(one) for one in clipped}
    visible = [layer for layer in psd.descendants() if layer.kind in ("pixel", "smartobject") and layer.is_visible()]
    keys: dict = {}
    for layer in unseen:
        if id(layer) in taken or layer.kind not in ("pixel", "smartobject") or _clipped_on(layer) \
                or comp_value(layer) is not None or any(STAT_GROUP.search(name) for name in _groups(layer)):
            continue
        ink = _layer_ink(layer)
        if ink is None:
            continue
        for other in visible:
            if (other.bbox[2] - other.bbox[0]) < ink[0] or (other.bbox[3] - other.bbox[1]) < ink[1]:
                continue
            if id(other) not in keys:
                keys[id(other)] = _layer_ink(other)
            if keys[id(other)] == ink:
                out.append((layer, f"a copy of {other.name!r}"))
                taken.add(id(layer))
                break
    if hidden is not None:
        out += _leftovers(psd, shown, taken, hidden)
    out = [(layer, why) for layer, why in out if not _isolating(layer)]
    going = {id(layer) for layer, _ in out}
    out = [(layer, why) for layer, why in out if all(id(one) in going for one in _clips(layer))]  # a base with its clips
    planned = {id(layer) for layer, _ in out if layer.is_group()}
    return [(layer, why) for layer, why in out if not any(id(group) in planned for group in _ancestors_of(layer))]


def _dead(psd: PSDImage, data: bytes, apply: bool = False) -> dict:
    """Data that draws nothing, {kind: bytes}: pixel sources (PxSD, a layer's PxSc: video frames), XMP
    DocumentAncestors, and (data: the file's bytes) patterns and filter-effect items nothing else in the file names;
    apply: removed."""
    def written(block) -> bytes:
        out = BytesIO()
        block.write(out)
        return out.getvalue()

    def named(blob: bytes, key: str) -> int:
        return blob.count(key.encode("ascii", "ignore")) + blob.count(key.encode("utf-16-be"))
    found: dict = {}
    blocks = psd._record.layer_and_mask_information.tagged_blocks
    if blocks is not None and Tag.PIXEL_SOURCE_DATA2 in blocks:
        found["pixel source"] = len(written(blocks[Tag.PIXEL_SOURCE_DATA2]))
        if apply:
            del blocks[Tag.PIXEL_SOURCE_DATA2]
    for layer in psd.descendants():
        tagged = layer._record.tagged_blocks
        if tagged is not None and Tag.PIXEL_SOURCE_DATA1 in tagged:
            found["layer pixel sources"] = found.get("layer pixel sources", 0) + len(written(tagged[Tag.PIXEL_SOURCE_DATA1]))
            if apply:
                del tagged[Tag.PIXEL_SOURCE_DATA1]
    for tags, kind, ident in ((PATTERN_TAGS, "patterns", "pattern_id"), (FX_TAGS, "filter effects", "uuid")):
        for tag in tags if data else ():
            items = blocks.get_data(tag) if blocks is not None and tag in blocks else None
            unused = []
            for item in items or []:
                key, own = str(getattr(item, ident, "") or "").strip("\x00"), written(item)
                if key and named(data, key) <= named(own, key):
                    unused.append(item)
                    found[kind] = found.get(kind, 0) + len(own)
            if apply and unused:
                items[:] = [item for item in items if item not in unused]
    resources = psd._record.image_resources
    if resources is not None and Resource.XMP_METADATA in resources:
        xmp = resources[Resource.XMP_METADATA]
        trimmed = ANCESTORS.sub(b"", xmp.data)
        if len(trimmed) < len(xmp.data):
            found["XMP ancestors"] = len(xmp.data) - len(trimmed)
            if apply:
                xmp.data = trimmed
    if apply and found:
        psd.mark_updated()
    return found


def clean_layers(psd: PSDImage) -> list:
    """clean()'s finds on a card (psd.repo_path; templates and other files never; psd.clean_hidden: the leftovers'
    Hidden): its deletions, and the PSD itself when it holds pixel sources (XMP ancestors go only with something else)."""
    from commands.font_check import is_card_psd
    if not is_card_psd(getattr(psd, "repo_path", "") or ""):
        return []
    plan = clean_plan(psd, getattr(psd, "clean_hidden", None))
    return [layer for layer, _ in plan] + ([psd] if set(_dead(psd, b"")) - {"XMP ancestors"} else [])


def _remove(psd: PSDImage, layer) -> None:
    """A layer deleted, a group's text layers first (their Txt2 stories go with them)."""
    if layer.is_group():
        for one in [one for one in layer.descendants() if one.kind == "type"]:
            psd_edit.remove_layer(psd, one)
    psd_edit.remove_layer(psd, layer)


def _cleaned(data: bytes, out_path: str, rel: str, hidden: Optional[Hidden]) -> list:
    """clean's one try: clean_plan(hidden) and _dead in one save, gated (Refused)."""
    psd = PSDImage.open(BytesIO(data))
    psd.repo_path = rel
    plan = clean_plan(psd, hidden)
    orphans = _orphan_links(psd)
    for layer, _ in plan:
        _remove(psd, layer)
    gone = _orphan_links(psd) - orphans
    for tag in (Tag.LINKED_LAYER1, Tag.LINKED_LAYER2, Tag.LINKED_LAYER3, Tag.LINKED_LAYER_EXTERNAL):
        items = psd.tagged_blocks.get_data(tag) if psd.tagged_blocks and tag in psd.tagged_blocks else None
        if items:
            items[:] = [item for item in items if str(item.uuid).strip("\x00") not in gone]
    dead = _dead(psd, data, apply=True)
    if not plan and not set(dead) - {"XMP ancestors"}:
        return []
    psd.mark_updated()
    psd_edit.save_psd(psd, out_path)
    old, new = psd_edit.use_record(PSDImage.open(BytesIO(data)), rel), psd_edit.use_record(PSDImage.open(out_path), rel)
    if _orphan_links(new):
        raise Refused("bytes", "a smart object's document is left with no layer using it")
    pictures = variants_of(old)
    for n in [None] + (pictures if len(pictures) > 1 else []):
        with variant_view(old, n), variant_view(new, n):
            box = (0, 0, old.width, old.height)
            a = np.asarray(psd_edit.render_region(old, box, False).convert("RGBA"))
            b = np.asarray(psd_edit.render_region(new, box, False).convert("RGBA"))
        if not np.array_equal(a, b):
            raise Refused("pixels", f"picture {n or 1} would change")
    return [layer.name for layer, _ in plan] + list(dead)


def clean(path: str, out_path: str, fonts=None, rel: str = "") -> list:
    """
    Clean (owner 2026-11-09): clean_plan's layers deleted (no smart-object document left without a layer) and _dead's
    data removed, in one save; every picture (the saved one and each variant's) drawn the same to the pixel, else
    Refused("pixels"). fonts.hidden (Hidden, the guards'): the leftovers too, left out again when they refuse. Returns
    what went: layer names, then the data's kinds ([]: nothing written).
    """
    with open(path, "rb") as fh:
        data = fh.read()
    hidden = getattr(fonts, "hidden", None)
    try:
        return _cleaned(data, out_path, rel, hidden)
    except Refused:
        if hidden is None:
            raise
    return _cleaned(data, out_path, rel, None)


PIXEL_FIXES = {     # kind -> (finds(psd) -> [layer], fix(path, out_path, fonts) -> [layer name], the parse it makes or None)
    "clean": (clean_layers, clean, None),      # first: the other fixes work on what stays
    "badge widths": (icon_layers, relay_badges, None),   # first: straighten and old glyphs lay out at the current widths
    "apostrophes": (curly_layers, straighten, straight_parse),
    "old glyphs": (icon_layers, redraw_old_glyphs, None),
    "badges": (icon_layers, fit_badges, None),
    "overlay badges": (salt_layers, one_colour_badges, None),
    "tags": (tag_layers, tags_to_badges, None),
    "status font": (status_font_layers, restyle_status, None),  # before status words: they cut FX from its text
    "status words": (status_word_layers, status_words, None),   # after the layout fixes: FX cut from the final text
    "alt template": (alt_bar_rows, alt_template, None),
    "normalise": (needs_normalise, normalise, None),     # last: linked pictures, normal text, no merged image (silent)
}
SWEEP_ONLY = {"alt template", "normalise"}   # run by card_sweep alone, never by the push guards (owner's call,
#   2026-11-06); normalise until the guards land its files silently (big pass G1: then out of this set)
PIXEL_LINES = {"clean": "hidden stars, type icons, copies and leftovers deleted, data that draws nothing removed",
               "badge widths": "badges laid out again at their new widths", "apostrophes": "straight apostrophes",
               "old glyphs": "old icon font glyphs redrawn",
               "badges": "badges and icons fitted to their text",
               "overlay badges": "badges under an overlay in one colour", "tags": "timing tags and costs set as badges",
               "status font": "status card text in Krabby Patty, its title in the chart's colour and effects",
               "status words": "status words in the chart's colours and effects",
               "alt template": "the template's stat bars, bar shapes, text layout and clip shapes",
               "normalise": "pictures linked, text set normal, no merged image"}


def pixel_kinds(psd, kinds: tuple = (), rel: str = "", chart: Optional[dict] = None, links=None,
                hidden: Optional[Hidden] = None) -> tuple:
    """The pixel fix kinds (all but SWEEP_ONLY by default) whose layers the PSD (open, or its bytes) has; rel: its repo
    path (the link step judges its links from there); chart: the status chart's fills (status words; none: no such
    kind); links: the link step's context (fonts.links: its read judges each link's Rslt); hidden: clean's leftovers
    (fonts.hidden)."""
    try:
        psd = PSDImage.open(BytesIO(psd)) if isinstance(psd, bytes) else psd
        if rel:
            psd.repo_path = rel
        if links is not None and hasattr(_links(), "ppis_of"):
            psd.link_ppi = _links().ppis_of(links.read, getattr(links, "ppis", None))
        psd.status_chart, psd.clean_hidden = chart, hidden
        default = tuple(kind for kind in PIXEL_FIXES if kind not in SWEEP_ONLY)
        return tuple(kind for kind in kinds or default if PIXEL_FIXES[kind][0](psd))
    except Exception:                           # unreadable: nothing to fix
        return ()


def _pixel_fixes(folder: str, base: str, current: str, rel: str, parser: Optional[PSDParser], result: Fixed,
                 fonts: Optional[Callable[[list], Optional[str]]], kinds: tuple, whole: bool = False) -> None:
    """fix()'s pixel fixes over current (the other fixes' result), each gated on its own; plain becomes the PSD without
    them (the pushed one when they are all) unless R8 redrew."""
    plain = result.data
    for kind in kinds:
        _, apply, reads = PIXEL_FIXES[kind]
        out = join(folder, kind.replace(" ", "-") + ".psd")
        try:
            if whole and apply in (fit_badges, tags_to_badges):
                names = apply(current, out, fonts, whole=True)
            elif apply is redraw_old_glyphs:    # its stale layers kept before any refusal (told)
                names = apply(current, out, fonts, seen=result.stale_glyphs)
            elif apply in (normalise, clean, restyle_status):
                names = apply(current, out, fonts, rel)
            else:
                names = apply(current, out, fonts)
            if not names:
                if apply is normalise:          # nothing written: what it left still says so
                    result.left, result.notes = list(names.left), result.notes + list(names.notes)
                continue
            data = _gated(current, out, rel, parser, reads, sized=False)
        except Exception as error:              # best effort, never told: plan.json says why
            result.pixel_left.append((kind, f"{type(error).__name__}: {error}"))
            continue
        if apply is normalise:                # the PNG follows its redrawn layers alone (a link's new raster)
            result.normalised, result.files, result.left = list(names), dict(names.files), list(names.left)
            if names.redraws:
                result.redrawn[kind] = list(names.redraws)
        else:
            result.redrawn[kind] = names
        result.data, current = data, out
        result.notes += getattr(names, "notes", [])
    if result.redrawn and not result.rewrapped:
        if not plain:
            with open(base, "rb") as fh:
                plain = fh.read()
        result.plain = plain


# ========================
# Folders (owner 2026-11-09): every card's and template's folders as the templates' (names, colours, order; loose
# layers and extra folders in their folder by role; text kept off a styled Text or Stat Text and Name in Overlay Text,
# or under a styled Text in a plain Stat Text and Name around the styled one; art drawn over other folders in Overlay;
# hidden Curves gone; nested star and type folders flattened; adjustments clipped to a folder moved into it, the folder
# isolated), only where every picture renders and the bot reads the card as before (folders_gate); silent, as the
# composer
# ========================
_C = SheetColorType
FOLDER_TYPES = {"creature": "Creature Types", "aux creature": "Creature Types", "minion": "Creature Types",
                "item": "Item Types", "aux item": "Item Types", "field": "Item Types", "nme": "Item Types",
                "status": "Status Effect Symbols"}  # the templates' type folder per class
FOLDER_TEMPLATES = {"Creature_Template": "Creature Types", "2_Creature_Template": "Creature Types",
                    "3_Creature_Template": "Creature Types", "Raid_Boss_Template": "Creature Types",
                    "Item_Template": "Item Types", "ActiveItem_Template": "Item Types",
                    "Status_Effect_Template": "Status Effect Symbols"}   # MDW/Templates' own (owner 2026-11-09)
FOLDER_ORDER = ("images", "stars", "types", "bars", "text", "overlay text", "overlay")   # bottom to top (Overlay: where
                                                # its art keeps the picture, above what it was drawn over)
FOLDER_NAMES = {"images": "images", "stars": "Stars", "bars": "Stats Bars", "text": "Text", "overlay text": "Overlay Text",
                "overlay": "Overlay"}
FOLDER_COLOURS = {"images": _C.BLUE, "stars": _C.YELLOW, "types": _C.GREEN, "bars": _C.ORANGE, "text": _C.RED,
                  "overlay text": _C.RED, "overlay": _C.BLUE}
SUB_COLOURS = {"stars": _C.YELLOW, "types": _C.GREEN}           # their subfolders'
BAR_PARTS = (("bars", "Bars", _C.SEAFOAM), ("darks", "Darks", _C.GRAY), ("lines", "Lines", _C.VIOLET))  # bottom to top
STAT_TEXT = ("Stat Text and Name", _C.BLUE)
FOLDER_ROLES = (("bars", re_compile(r"(stats?|star) ?bars?( blocks)?|bars?|darks?|lines?")),   # Star Bars: Minecraft Panda's
                ("overlay text", re_compile(r"overlay text")), ("overlay", re_compile(r"overlays?")),
                ("text", re_compile(r"text|stat ?txt|stat text and name|names?|ability txt")),
                ("types", re_compile(r".*\btypes?|tyoes|status effect symbols|field")),
                ("stars", re_compile(r".*\bstars?\b.*|\d+ ?stars?")),
                ("images", re_compile(r"images?")))
MAIN_FOLDER = {"images": re_compile(r"images?"), "stars": re_compile(r"stars?"),
               "types": re_compile(r"(creature |item )?types?|tyoes|status effect symbols"),
               "bars": re_compile(r"(stats?|star) ?bars?"), "text": re_compile(r"text"),
               "overlay text": re_compile(r"overlay text"), "overlay": re_compile(r"overlays?")}
GENERIC = re_compile(r"group( \d+)?|new group|layer \d+")   # Photoshop's own names: named by what they hold
NESTED = {"stars": re_compile(r"stars?"), "types": re_compile(r"(creature |item )?types?")}   # a redundant nest
BAR_LEAF = re_compile(r"(hp|health|def|defen[cs]e|atk|attack|spd|speed)( ?(bars?|top|bottom|dark|lines?))?")
STAT_WORDS = {"health", "defense", "attack", "speed"}
ADJUSTMENTS = frozenset(("curves", "exposure", "levels", "huesaturation", "colorbalance", "vibrance", "gradientmap",
                         "brightnesscontrast", "blackandwhite", "channelmixer", "invert", "photofilter", "posterize",
                         "threshold", "selectivecolor", "colorlookup"))
FOLDER_TOL = 2              # levels (of 255) a picture's pixel may move in the fork's render
FOLDER_LEVELS = ("template", "plain", "kept", "names", "bare", "colours")   # tried in order (arrange)


def _key(name: str) -> str:
    return sub(r"\s+", " ", sub(r"\s+copy(\s+\d+)?$", "", layer_tag(name)[0].strip(), flags=IGNORECASE)).lower()


def folder_kind(rel: str) -> Optional[str]:
    """The card's or template's type folder name (FOLDER_TYPES, FOLDER_TEMPLATES); None for any other PSD."""
    if rel.startswith("MDW/Templates/") and rel.count("/") == 2:
        return FOLDER_TEMPLATES.get(basename(rel)[:len(basename(rel)) - 4])
    return FOLDER_TYPES.get(kind_of(rel))


def _plain(layer) -> bool:
    """A folder its layers draw the same without: shown, pass-through, 100%, no mask, effects, clip or tag."""
    return (layer.is_group() and layer.visible and layer.blend_mode == BlendMode.PASS_THROUGH and layer.opacity == 255
            and layer.fill_opacity == 255 and not layer.has_mask() and not layer.has_vector_mask()
            and not layer.has_effects() and not layer.clipping and layer_tag(layer.name)[1] is None)


def _leaf_role(layer, types, stars, bars: bool = False) -> str:
    """text, adjust, bars, types, stars or art (bars: inside Stats Bars, where a bar's names come first)."""
    name = _key(layer.name)
    if layer.kind == "type" or EFFECT_LAYER_TAG in layer.name:
        return "text"
    if layer.kind in ADJUSTMENTS:
        return "adjust"
    if bars and (comp_value(layer) is not None or BAR_LEAF.fullmatch(name) or name.isdigit()
                 or any(word in name for word in ("line", "dark", "light"))):
        return "bars"
    if name in types:
        return "types"
    if sub(r"[^a-z0-9]+", "", name) in stars or "star" in sub(r"[^a-z]+", " ", name).split():
        return "stars"
    linked = _link_kind(layer)                  # linked to Types/Stars or Types, whatever its name (Joel Miller's)
    if linked:
        return linked + "s"
    if comp_value(layer) is not None or BAR_LEAF.fullmatch(name):
        return "bars"
    return "art"


def _content_role(group, types, stars, bars: bool = False) -> Optional[str]:
    """A folder's role by its layers: text, bars, types or stars (art beside them), images (art), adjust or empty;
    None when mixed."""
    roles = {_leaf_role(layer, types, stars, bars) for layer in group.descendants() if not layer.is_group()}
    if not roles:
        return "empty"
    named = roles - {"adjust", "art"}
    if not named:
        return "images" if "art" in roles else "adjust"
    if len(named) > 1 or (named == {"text"} and "art" in roles):
        return None
    return named.pop()


def _group_role(group, types, stars) -> Optional[str]:
    """A top-level folder's role: its name's (FOLDER_ROLES) when its layers agree, else its layers'."""
    key, found = _key(group.name), _content_role(group, types, stars)
    named = next((role for role, pattern in FOLDER_ROLES if pattern.fullmatch(key)), None)
    if named is None:
        return found
    if MAIN_FOLDER[named].fullmatch(key):       # a role's own folder, whatever it holds
        return named
    if found in (named, "images", "adjust", "empty") or (named == "bars" and _content_role(group, types, stars, True) == "bars"):
        return named
    return None


def _units(group) -> list:
    """The folder's layers bottom to top as units: a base and the layers clipped to it (an orphan clip alone)."""
    out = []
    for layer in group:
        if layer.clipping and out:
            out[len(out) - 1].append(layer)
        else:
            out.append([layer])
    return out


def _is_empty(layer) -> bool:
    return layer.is_group() and not any(not child.is_group() for child in layer.descendants())


def _new_folder(parent, name: str, colour, index: int):
    """A pass-through folder made in parent at index, with a layer id of its own."""
    from psd_tools.api.layers import Group, _issue_layer_ids
    psd = parent if isinstance(parent, PSDImage) else parent._psd
    group = Group.new(parent=parent, name=name)
    group.blend_mode = BlendMode.PASS_THROUGH
    psd.image_resources[Resource.IDS_SEED_NUMBER] = _issue_layer_ids(
        psd, [layer for layer in psd.descendants() if layer is not group], [group])
    group.sheet_color = colour
    group.move_to(parent, index=index)
    return group


def _place(units: list, parent) -> bool:
    """parent's layers as units say, bottom to top (each unit's layers in the document already); whether one moved."""
    i, moved = 0, False
    for unit in units:
        for layer in unit:
            if not (layer.parent is parent and list(parent).index(layer) == i):
                layer.move_to(parent, index=i)
                moved = True
            i += 1
    return moved


def _name_colour(layer, name: Optional[str], colour, done: list, counts: Counter) -> None:
    if name is not None and layer.name != name and layer_tag(layer.name)[1] is None:
        done.append(f"{layer.name!r} named {name!r}")
        counts["types renamed" if name in FOLDER_TYPES.values() else "renamed"] += 1
        layer.name = name
    if colour is not None and layer.sheet_color != colour:
        layer.sheet_color = colour
        done.append(f"{layer.name!r} {colour.name.lower()}")
        counts["coloured"] += 1


def hidden_curves(psd: PSDImage, on_groups: bool = True) -> list:
    """Curves layers turned off and shown in no picture, that no layer clips to (owner 2026-11-09); on_groups False:
    not one clipped to a folder (Photoshop composites a folder something clips to on its own, hidden or not: Matterling's
    header moved 16 levels without its hidden Curves; only a Photoshop-proven sweep takes those)."""
    pictures = variants_of(psd)
    out = []
    for layer in psd.descendants():
        if layer.kind != "curves" or layer.visible or any(shown_in(layer, n) for n in pictures):
            continue
        above = layer.next_sibling()
        if not layer.clipping and above is not None and above.clipping:
            continue
        if layer.clipping and not on_groups:
            kids = list(layer.parent)
            i = kids.index(layer)
            base = next((kids[j] for j in range(i - 1, -1, -1) if not kids[j].clipping), None)
            if base is not None and base.is_group():
                continue
        out.append(layer)
    return out


def _unwrap(folder, child, done: list, counts: Counter, what: str) -> bool:
    """child's layers into folder in its place, child gone; False where a clip would change."""
    kids, inner = list(folder), list(child)
    i = kids.index(child)
    above = kids[i + 1] if i + 1 < len(kids) else None
    if (inner and inner[0].clipping) or (above is not None and above.clipping):
        return False
    for j, layer in enumerate(inner):
        layer.move_to(folder, index=i + j)
    folder.remove(child)
    done.append(f"{what} {child.name!r} flattened")
    counts[what] += 1
    return True


def _draws(layer) -> bool:
    return layer.visible and layer.kind not in ADJUSTMENTS and (layer.is_group() or layer.has_pixels())


def _flatten(folder, role: str, types, stars, done: list, counts: Counter) -> None:
    """A redundant nest into folder, in its place, and again inside: a plain folder named after the role or by
    Photoshop (Stars/stars, Stars/Group 1, Types/types); in Stars only one that is all Stars draws (one star in a group
    of its own is a unit the bake copies whole: kept)."""
    changed = True
    while changed:
        changed = False
        for child in list(folder):
            key = _key(child.name)
            if (child.is_group() and _plain(child) and (NESTED[role].fullmatch(key) or GENERIC.fullmatch(key))
                    and (role == "types" or not any(_draws(other) for other in folder if other is not child))
                    and _content_role(child, types, stars) in (role, "images", "adjust")
                    and _unwrap(folder, child, done, counts, f"nested {role} folder")):
                changed = True
                break


def _bar_part(unit) -> Optional[str]:
    """A Stats Bars unit's subfolder: bars, darks, lines or None."""
    base = unit[0]
    key = _key(base.name)
    if base.is_group():
        for part in ("bars", "darks", "lines"):
            if key in (part, part[:len(part) - 1]):
                return part
        leaves = [layer for layer in base.descendants() if not layer.is_group()]
        if leaves and all("line" in _key(layer.name) for layer in leaves):
            return "lines"
        if any(comp_value(l) is not None or _key(l.name).isdigit() or l.kind == "exposure" for l in base.descendants()):
            return "darks" if BAR_LEAF.fullmatch(key) or "dark" in key else None
        if BAR_LEAF.fullmatch(key):
            return "bars"
        return "bars" if base.visible and _content_role(base, (), (), True) in ("bars", "images") else None
    if "line" in key:
        return "lines"
    if comp_value(base) is not None or key.isdigit() or base.kind == "exposure":
        return "darks"
    return "bars" if base.kind in ("pixel", "shape", "smartobject") else None     # the bars' own art


def _inside_bars(folder, level: str, done: list, counts: Counter) -> None:
    """Stats Bars' layers into Bars, Darks and Lines (template level), named and coloured."""
    if level not in ("bare", "colours"):
        units = _units(folder)
        parts = [_bar_part(unit) for unit in units]
        made = {}
        for part, name, colour in BAR_PARTS:
            members = [unit for unit, found in zip(units, parts) if found == part]
            if not members:
                continue
            k = next((i for i, u in enumerate(members) if len(u) == 1 and u[0].is_group()
                      and _key(u[0].name) in (part, part[:len(part) - 1])), None)
            if k is not None and level != "template" and not _plain(members[k][0]) and len(members) > 1:
                continue                        # into a part with its own look: kept apart
            if k is None:
                made[part] = [_new_folder(folder, name, colour, list(folder).index(members[0][0]))]
                content = members
                done.append(f"{name} folder made")
                counts["folders made"] += 1
            else:
                made[part] = members[k]
                content = members[:k] + [[layer] for layer in members[k][0]] + members[k + 1:]
            moved = [u for u in members if u is not made[part]]
            for u in moved:
                counts["extra folders" if u[0].is_group() else "loose layers"] += 1
                done.append(f"{u[0].name!r} into {name}")
            _place(content, made[part][0])
        final, last, rest = [], None, []
        for unit, part in zip(units, parts):    # a unit of none stays above the part under it
            if part is None:
                rest.append((last, unit))
            else:
                last = part
        for part in (None, "bars", "darks", "lines"):
            if part in made:
                final.append(made[part])
            final += [unit for after, unit in rest if after == part]
        if _place(final, folder):               # an order alone is a change too
            done.append(f"{folder.name!r} parts in the templates' order")
            counts["reordered"] += 1
    for child in folder:
        part = next((p for p in BAR_PARTS if child.is_group() and _key(child.name) in (p[0], p[0][:len(p[0]) - 1])), None)
        if part is not None:
            _name_colour(child, part[1] if level != "colours" else None, part[2], done, counts)


def _stat_folder(folder, types, stars):
    """Text's Stat Text and Name (by its names, else a Photoshop-named folder of labels), or None."""
    groups = [group for group in folder if group.is_group()]
    stat = next((g for g in groups if _key(g.name) in ("stat text and name", "stat txt", "stat text")), None)
    if stat is None and any(l.kind == "type" and _key(l.name) in STAT_WORDS for l in folder.descendants()):
        stat = next((g for g in groups if GENERIC.fullmatch(_key(g.name)) and _content_role(g, types, stars) == "text"
                     and any(l.kind == "type" and _key(l.name) in STAT_WORDS for l in g.descendants())), None)
    return stat


def _stat_units(folder) -> list:
    """Text's loose labels and Names, each with what clips to it (_inside_text moves these into Stat Text and Name)."""
    if not any(l.kind == "type" and _key(l.name) in STAT_WORDS for l in folder.descendants()):
        return []
    return [unit for unit in _units(folder)
            if unit[0].kind == "type" and not unit[0].clipping and _key(unit[0].name) in STAT_WORDS | {"name"}]


def _inside_text(folder, level: str, types, stars, done: list, counts: Counter) -> None:
    """Text's own: the stat labels and Name in Stat Text and Name (template level), Photoshop-named text folders
    flattened."""
    labels = [layer for layer in folder.descendants() if layer.kind == "type" and _key(layer.name) in STAT_WORDS]
    groups = [group for group in folder if group.is_group()]
    stat = _stat_folder(folder, types, stars)
    for group in groups if level != "colours" else ():
        if (group is not stat and GENERIC.fullmatch(_key(group.name)) and _plain(group)
                and _content_role(group, types, stars) == "text"):
            _unwrap(folder, group, done, counts, "text folder")
    if stat is None and labels and level not in ("bare", "colours"):
        first = next((l for l in folder if l.kind == "type" and _key(l.name) in STAT_WORDS | {"name"}), None)
        if first is not None:
            stat = _new_folder(folder, STAT_TEXT[0], STAT_TEXT[1], list(folder).index(first))
            done.append(f"{STAT_TEXT[0]} folder made")
            counts["folders made"] += 1
    if stat is None:
        return
    _name_colour(stat, STAT_TEXT[0] if level != "colours" else None, STAT_TEXT[1], done, counts)
    if level in ("bare", "colours") or not labels or (level != "template" and not _plain(stat)):
        return
    for unit in _stat_units(folder):            # a label or Name with what clips to it
        for layer in unit:
            layer.move_to(stat)
        done.append(f"{unit[0].name!r} into {STAT_TEXT[0]}")
        counts["loose layers"] += 1


def _inside(folder, role: str, level: str, types, stars, done: list, counts: Counter) -> None:
    """A role folder's own standard: its subfolders' names, colours and nests."""
    if role in NESTED and level != "colours":
        _flatten(folder, role, types, stars, done, counts)
    if role == "bars":
        _inside_bars(folder, level, done, counts)
    elif role == "text":
        _inside_text(folder, level, types, stars, done, counts)
    for child in list(folder):
        if not child.is_group() or layer_tag(child.name)[1] is not None:
            continue
        key = _key(child.name)
        if level == "colours":                  # colours alone
            if role in SUB_COLOURS and _content_role(child, types, stars) in (role, "images", "adjust", "empty"):
                _name_colour(child, None, SUB_COLOURS[role], done, counts)
        elif _is_empty(child) and (GENERIC.fullmatch(key) or (role in NESTED and NESTED[role].fullmatch(key))):
            folder.remove(child)
            done.append(f"empty folder {child.name!r} deleted")
            counts["empty folders"] += 1
        elif role in SUB_COLOURS and _content_role(child, types, stars) in (role, "images", "adjust", "empty"):
            drawn = [l for l in child.descendants() if not l.is_group() and _draws(l)]
            name = None
            if GENERIC.fullmatch(key) and drawn:    # one icon's folder: named after it (a type's never as the type)
                name = sub(r"\s+copy(\s+\d+)?$", "", layer_tag(drawn[len(drawn) - 1].name)[0].strip(), flags=IGNORECASE)
                name = f"{name} icon" if role == "types" else name
            _name_colour(child, name, SUB_COLOURS[role], done, counts)
        elif role == "images" and GENERIC.fullmatch(key):
            _name_colour(child, "Adjustments" if _content_role(child, types, stars) == "adjust" else "Art", None, done, counts)


def _mains(units: list, roles: list, names: dict) -> dict:
    """role -> the unit of its own folder (named as the template's, else after the role)."""
    mains = {}
    for role in FOLDER_ORDER:
        mine = [u for u, r in zip(units, roles) if r == role and u[0].is_group()]
        main = next((u for u in mine if u[0].name == names[role]), None) or \
            next((u for u in mine if MAIN_FOLDER[role].fullmatch(_key(u[0].name))), None)
        if main is not None:
            mains[role] = main
    return mains


def _extract(psd: PSDImage, types, stars, done: list, counts: Counter, art: frozenset = frozenset()) -> None:
    """A folder of another kind, or a star or type icon (_leaf_role: Joel Miller's linked stars in images), at the top
    or bottom of a plain role folder (a card's Text inside Item Types) out beside it, in the same place in the stack."""
    for folder in [layer for layer in psd if layer.is_group()]:
        named = next((role for role, pattern in MAIN_FOLDER.items() if pattern.fullmatch(_key(folder.name))), None)
        if named is None or not _plain(folder):
            continue
        for top in (True, False):
            while len(folder) > 1:
                kids = list(folder)
                child = kids[len(kids) - 1] if top else kids[0]
                found = _content_role(child, types, stars) if child.is_group() else \
                    next((role for role in ("stars", "types") if _leaf_role(child, types, stars) == role
                          and child.layer_id not in art), None)
                if (found in (None, named, "images", "adjust", "empty") or child.clipping
                        or (named == "overlay text" and found == "text")    # its own: no out and back in
                        or (not top and kids[1].clipping)):
                    break
                at = list(psd).index(folder)
                child.move_to(psd, index=at + 1 if top else at)
                done.append(f"{child.name!r} out of {folder.name!r}")
                counts["folders out of another kind's"] += 1


def _boxes(unit) -> list:
    """The drawn boxes of a unit's shown layers, their effects' reach added; None for an adjustment (it changes what
    lies under it everywhere)."""
    from commands.psd_regions import fx_pad
    out = []
    for layer in unit:
        leaves = [layer] if not layer.is_group() else [l for l in layer.descendants() if not l.is_group()]
        for leaf in leaves:
            if not leaf.is_visible():
                continue
            if leaf.kind in ADJUSTMENTS:
                return None
            x0, y0, x1, y1 = leaf.bbox
            pad = fx_pad(leaf)
            if x1 > x0 and y1 > y0:
                out.append((x0 - pad, y0 - pad, x1 + pad, y1 + pad))
    return out


def _crosses(unit, others: list) -> bool:
    """Whether moving unit past others can change the picture: their drawings meet, or one is an adjustment."""
    mine = _boxes(unit)
    if mine is None:
        return True
    for other in others:
        theirs = _boxes(other)
        if theirs is None:
            return True
        if any(a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3] for a in mine for b in theirs):
            return True
    return False


def _overlays(units: list, roles: list, mains: dict) -> None:
    """Text when Text has a look of its own (effects, a mask, another blend) to Overlay Text; art drawn over another
    role's layers to Overlay (owner 2026-11-09): roles changed in place."""
    text = mains.get("text")
    if text is not None and not _plain(text[0]):
        for i, unit in enumerate(units):
            if roles[i] == "text" and unit is not text:
                roles[i] = "overlay text"
    images = mains.get("images")
    at = next((i for i, unit in enumerate(units) if unit is images), None)
    if at is None:
        at = next((i for i, role in enumerate(roles) if role == "images"), None)
    if at is None:
        return
    lo = hi = at
    while lo > 0 and roles[lo - 1] in ("images", "adjust", "empty"):
        lo -= 1
    while hi + 1 < len(roles) and roles[hi + 1] in ("images", "adjust", "empty"):
        hi += 1
    for i, role in enumerate(roles):
        if role == "images" and not lo <= i <= hi and units[i] is not images:
            crossed = units[hi + 1:i] if i > hi else units[i + 1:lo]
            if _crosses(units[i], crossed):
                roles[i] = "overlay"


def _off_stat(psd: PSDImage, units: list, roles: list, mains: dict, types, stars, done: list, counts: Counter) -> None:
    """Text's loose labels and Names, with what clips to them, out over a plain Text into Overlay Text when Stat Text
    and Name has a look of its own (owner 2026-11-09: their look stays); units and roles grow. Under a Text with a look
    of its own (theirs too): _around_stat."""
    text = mains.get("text")
    stat = _stat_folder(text[0], types, stars) if text is not None else None
    moving = _stat_units(text[0]) if stat is not None and not _plain(stat) else []
    if not moving:
        return
    if not _plain(text[0]) or len(text) > 1:
        _around_stat(text[0], stat, moving, done, counts)
        return
    at = next(i for i, unit in enumerate(units) if unit is text) + 1
    top = text[0]
    for unit in moving:
        for layer in unit:                      # over Text, in their order
            layer.move_to(psd, index=list(psd).index(top) + 1)
            top = layer
    units[at:at] = moving
    roles[at:at] = ["overlay text"] * len(moving)


def _around_stat(text, stat, moving: list, done: list, counts: Counter) -> None:
    """A plain Stat Text and Name made around Text's styled one (named Stat Text), the labels and Names beside it, in
    their order."""
    kids = list(text)
    own = next(unit for unit in _units(text) if unit[0] is stat)
    below = [unit for unit in moving if kids.index(unit[0]) < kids.index(stat)]
    above = [unit for unit in moving if kids.index(unit[0]) > kids.index(stat)]
    _place(below + [own] + above, _new_folder(text, STAT_TEXT[0], STAT_TEXT[1], kids.index(stat)))
    done.append(f"{STAT_TEXT[0]} folder made")
    done += [f"{unit[0].name!r} into {STAT_TEXT[0]}" for unit in moving]
    done.append(f"{stat.name!r} named 'Stat Text' in it")
    counts["folders made"] += 1
    counts["loose layers"] += len(moving)
    stat.name = "Stat Text"


def arrange(psd: PSDImage, rel: str, types, stars, level: str = "template", proven: bool = True,
            art: frozenset = frozenset()) -> tuple:
    """
    The card's folders as the templates' (FOLDER_ORDER, FOLDER_NAMES, FOLDER_COLOURS), in place.

    Args:
        psd: The open card (changed)
        rel: Its repo path (its class names the type folder)
        types: Types/ names (lowercase); stars: PSDParser.star_names()
        level: template (every role's layers in its folder, the folders in the templates' order); plain (only a
            role's layers next to its folder folded in, none into a folder with effects, a mask or another blend, the
            folders in the templates' order); kept (as plain, the folders where they are); names (no layer out of its
            folder); bare (names, colours, deletions and nests alone); colours (colours and hidden Curves alone)
        proven: hidden Curves clipped to a folder go too (hidden_curves' on_groups: a sweep proven in Photoshop)
        art: layer ids of star icons that are art (art_icons): they stay where they are

    Returns:
        (what it did, Counter of the owner's counts)
    """
    done, counts = [], Counter()
    type_name = folder_kind(rel)
    if type_name is None:
        return done, counts
    names = {**FOLDER_NAMES, "types": type_name}
    for layer in hidden_curves(psd, proven):
        layer.parent.remove(layer)
        done.append(f"hidden {layer.name!r} deleted")
        counts["hidden curves"] += 1
    if level in ("template", "plain"):
        _extract(psd, types, stars, done, counts, art)
    units, roles = [], []
    for unit in _units(psd):
        base = unit[0]
        if base.clipping:                       # an orphan clip stays
            role = None
        elif base.is_group():
            role = "empty" if _is_empty(base) else _group_role(base, types, stars)
        else:
            role = {"art": "images"}.get(_leaf_role(base, types, stars), _leaf_role(base, types, stars))
        key = _key(base.name)
        if (role == "empty" and len(unit) == 1 and level != "colours"
                and (GENERIC.fullmatch(key) or any(p.fullmatch(key) for _, p in FOLDER_ROLES))):
            psd.remove(base)                    # an empty folder named by Photoshop or after a role
            done.append(f"empty folder {base.name!r} deleted")
            counts["empty folders"] += 1
            continue
        units.append(unit)
        roles.append(role)
    mains = _mains(units, roles, names)
    if level not in ("names", "bare", "colours"):
        _overlays(units, roles, mains)
    if level == "template":
        _off_stat(psd, units, roles, mains, types, stars, done, counts)
    owner = []                                  # a unit of no role goes with the one under it (else over it; an
                                                # orphan clip stays at the bottom)
    for i, role in enumerate(roles):
        below = next((roles[j] for j in range(i - 1, -1, -1) if roles[j] in FOLDER_ORDER), None)
        above = next((roles[j] for j in range(i + 1, len(roles)) if roles[j] in FOLDER_ORDER), None)
        owner.append(role if role in FOLDER_ORDER else None if units[i][0].clipping else below or above)
    folds = [role in FOLDER_ORDER or role in ("adjust", "empty") for role in roles]
    mains = _mains(units, roles, names)
    folders = {}
    if level not in ("names", "bare", "colours"):
        plan = {role: [] for role in FOLDER_ORDER}
        loose = []                              # [index, owner, unit] at the top level
        for i, (unit, role) in enumerate(zip(units, owner)):
            main = mains.get(role)
            if role is None or not folds[i] or (level != "template" and main is not None and unit is not main
                                                  and not _plain(main[0])):
                loose.append((i, role, unit))
            else:
                plan[role].append((i, unit))
        if level != "template":                 # a role's run next to its folder (or its first unit) alone
            for role, members in plan.items():
                if not members:
                    continue
                at = next((k for k, (_, u) in enumerate(members) if u is mains.get(role)), 0)
                lo = hi = at
                while lo > 0 and members[lo - 1][0] == members[lo][0] - 1:
                    lo -= 1
                while hi + 1 < len(members) and members[hi + 1][0] == members[hi][0] + 1:
                    hi += 1
                loose += [(i, role, u) for k, (i, u) in enumerate(members) if not lo <= k <= hi]
                plan[role] = members[lo:hi + 1]
        for role in FOLDER_ORDER:
            members = plan[role]
            if not members:
                continue
            main = mains.get(role)
            k = next((k for k, (_, u) in enumerate(members) if u is main), None)
            if k is None:
                folder = _new_folder(psd, names[role], FOLDER_COLOURS[role], list(psd).index(members[0][1][0]))
                done.append(f"{names[role]} folder made")
                counts["folders made"] += 1
                content = [u for _, u in members]
                folders[role] = (members[0][0], [folder])
            else:
                folder = main[0]
                content = [u for _, u in members[:k]] + [[layer] for layer in folder] + [u for _, u in members[k + 1:]]
                folders[role] = (members[0][0], [folder, *main[1:]])    # what clips to the folder stays on it
            for _, u in members:
                if u is not main:
                    done.append(f"{u[0].name!r} into {names[role]}")
                    counts["extra folders" if u[0].is_group() else "loose layers"] += 1
            _place(content, folder)
        if level in ("template", "plain"):
            order = [None, *(role for role in FOLDER_ORDER if role != "overlay")]
            if "overlay" in folders:            # over the highest role its layers lay over
                first = folders["overlay"][0]
                under = [owner[j] for j in range(first) if owner[j] not in (None, "overlay")]
                order.insert(order.index(max(under, key=FOLDER_ORDER.index)) + 1 if under else 1, "overlay")
            final = []
            for role in order:
                if role in folders:
                    final.append(folders[role][1])
                final += [u for _, o, u in sorted(loose, key=lambda t: t[0]) if o == role]
        else:                                   # every folder where its run began
            items = [(at, unit) for at, unit in folders.values()] + [(i, u) for i, _, u in loose]
            final = [unit for _, unit in sorted(items, key=lambda t: t[0])]
        if _place(final, psd):                  # an order alone is a change too (The Farlands)
            done.append("folders in the templates' order")
            counts["reordered"] += 1
    for role in FOLDER_ORDER:
        folder = folders[role][1][0] if role in folders else mains[role][0] if role in mains else None
        if folder is None:
            continue
        _name_colour(folder, names[role] if level != "colours" else None, FOLDER_COLOURS[role], done, counts)
        _inside(folder, role, level, types, stars, done, counts)
    return done, counts


def folder_issues(psd: PSDImage, rel: str, parser: Optional[PSDParser] = None, art: Optional[frozenset] = None) -> list:
    """What keeps the card's folders off the templates', in plain words ([]: standard); a star icon that is art (art:
    their layer ids, else art_icons with a parser) is none."""
    type_name = folder_kind(rel)
    if type_name is None:
        return []
    if not getattr(psd, "repo_path", None):
        psd.repo_path = rel
    if art is None:
        art = art_icons(psd, rel, parser) if parser is not None else frozenset()
    names = {**FOLDER_NAMES, "types": type_name}
    want = {names[role]: role for role in FOLDER_ORDER}
    out, seen = [], []
    for layer in psd:
        role = want.get(layer.name) if layer.is_group() else None
        if role is None:
            out.append(f"{layer.name!r} at the top level")
            continue
        if layer.sheet_color != FOLDER_COLOURS[role]:
            out.append(f"{layer.name!r} not {FOLDER_COLOURS[role].name.lower()}")
        if role in seen:
            out.append(f"two {layer.name!r} folders")
        seen.append(role)
        out += [f"{child.name!r} in {layer.name!r}" for child in _odd_children(layer, role, art)]
    ordered = [role for role in seen if role != "overlay"]  # Overlay: over what its art lies over
    if ordered != sorted(ordered, key=FOLDER_ORDER.index):
        out.append("folders out of the templates' order")
    out += [f"hidden {layer.name!r}" for layer in hidden_curves(psd)]
    return out


def _odd_children(folder, role: str, art: frozenset = frozenset()) -> list:
    """A role folder's layers off its template (art: layer ids of star icons that are art, none)."""
    kids = list(folder)
    if role == "bars":
        parts = {name: colour for _, name, colour in BAR_PARTS}
        found = [k.name for k in kids if k.is_group() and k.name in parts]
        out = [k for k in kids if not (k.is_group() and k.name in parts) or k.sheet_color != parts[k.name]]
        return out + ([folder] if found != sorted(found, key=list(parts).index) else [])
    if role == "text":
        out = [k for k in kids if k.is_group() and (GENERIC.fullmatch(_key(k.name))
                                                     or (k.name == STAT_TEXT[0] and k.sheet_color != STAT_TEXT[1]))]
        if any(layer.kind == "type" and _key(layer.name) in STAT_WORDS for layer in folder.descendants()):
            out += [k for k in kids if k.kind == "type" and _key(k.name) in STAT_WORDS | {"name"}]
        return out
    if role in SUB_COLOURS:
        return [k for k in kids if k.is_group() and (k.sheet_color != SUB_COLOURS[role] or GENERIC.fullmatch(_key(k.name))
                or (_plain(k) and NESTED[role].fullmatch(_key(k.name)) and not any(_draws(o) for o in kids if o is not k)))]
    return [k for k in kids if (k.is_group() and GENERIC.fullmatch(_key(k.name)))
            or (role == "images" and not k.clipping and _link_kind(k) and k.layer_id not in art)]   # a linked star
                                                # or type icon in images


class Foldered(list):
    """standard_folders' lines, with the level that held (FOLDER_LEVELS), the owner's counts, what the folders still
    lack (folder_issues) and each level refused and why."""

    def __init__(self, lines=(), level: str = "", counts=None, left=(), tried=()):
        super().__init__(lines)
        self.level, self.counts, self.left, self.tried = level, Counter(counts or {}), list(left), list(tried)


def _sorted_lists(value):
    """A read with its lists of names or records in one order (folders reorder the walk)."""
    if isinstance(value, dict):
        return {k: _sorted_lists(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_sorted_lists(v) for v in value]
        if items and all(isinstance(v, (dict, str)) for v in items):
            return sorted(items, key=repr)
        return items
    return value


def folder_reads(psd: PSDImage, rel: str, parser: PSDParser) -> dict:
    """What folders_gate compares: every picture's render (the fork's) and parse and the details record
    (psd_details.gather, its time and tree aside, each layer path read as its layer's id); the regions (unpadded) as
    a note alone (owner 2026-11-09: the website's regions may change where nothing else does)."""
    import commands.psd_details as psd_details
    import commands.psd_parts as psd_parts
    from commands.psd_regions import regions
    tags = variants_of(psd)
    renders, parses = [], []
    for n in tags if len(tags) > 1 else [None]:
        with psd_edit.variant(psd, n):
            drawn = psd.composite(ignore_preview=True, apply_icc=False, text_preview=False)
            renders.append(np.asarray(drawn.convert("RGBA"), np.int16))
        card = parser.parse(psd, rel, variant=n)
        parses.append({"card": card.to_dict(), "problems": CardValidator.validate(card)})
    stem, paths = basename(rel), (psd_details.layer_path, psd_parts.layer_path)
    psd_details.layer_path = psd_parts.layer_path = lambda layer: f"#{layer.layer_id}"     # a folder moves no id
    try:
        record = psd_details.gather(psd, rel, stem[:len(stem) - 4], None, parser.all_types, parser)
    finally:
        psd_details.layer_path, psd_parts.layer_path = paths
    record = {k: v for k, v in record.items() if k not in ("at", "tree")}
    return {"renders": renders, "parses": parses, "regions": _sorted_lists(regions(psd, parser.all_types, pad=False)),
            "details": _sorted_lists(record)}


def _by_id(psd: PSDImage) -> dict:
    records = psd_edit.layer_records(psd)
    out = {}
    for layer in psd.descendants():
        out.setdefault(layer.layer_id, []).append((records[psd_edit._record_index(layer)], layer))
    return out


def folders_gate(path: str, out_path: str, rel: str, parser: PSDParser, base: Optional[dict] = None,
                 allow: Optional[dict] = None, tol: int = FOLDER_TOL, spots: bool = False,
                 memo: Optional[dict] = None) -> dict:
    """
    Gate: every layer's record as it was (a folder's name and colour aside; allow: layer id -> the record parts that
    may change too, an isolated folder's blend and the clips moved into it), gone only a hidden Curves layer or a
    folder whose layers stay, new only plain folders, no smart object's data left unused; every picture renders
    within tol and the bot reads the card the same (folder_reads: the parse and the details; the website's regions
    may change; spots: the bake's star and type spots too, which follow the stars filed in Stars). Raises Refused;
    returns base's reads (made when None; memo["base"] keeps them, a refusal too).
    """
    allow = allow or {}
    before, after = PSDImage.open(path), PSDImage.open(out_path)
    if _orphan_links(after) - _orphan_links(before):
        raise Refused("bytes", "a smart object's embedded document is left with no layer using it")
    old, new = _by_id(before), _by_id(after)
    for lid, items in old.items():
        if lid not in new:
            for _, layer in items:
                if not ((layer.kind == "curves" and not layer.visible) or layer.is_group()):
                    raise Refused("bytes", f"{layer.name!r} went missing")
            continue
        if len(items) != len(new[lid]):
            raise Refused("bytes", f"layer id {lid} changed count")
        for (a, layer), (b, _) in zip(items, new[lid]):
            if a == b:
                continue
            keys = {k for k in a if a[k] != b[k]}
            blocks = {key for key, raw in set(a["blocks"]) ^ set(b["blocks"])}
            more = allow.get(lid, set())
            names = {bytes(Tag.UNICODE_LAYER_NAME), bytes(Tag.SHEET_COLOR_SETTING)} if layer.is_group() else set()
            if not (keys <= ({"name", "blocks"} if layer.is_group() else set()) | more
                    and blocks <= names | ({bytes(Tag.SECTION_DIVIDER_SETTING)} if "blocks" in more else set())):
                raise Refused("bytes", f"{layer.name!r} changed beyond its folder")
    for lid, items in new.items():
        if lid not in old and not all(_plain(layer) for _, layer in items):
            raise Refused("bytes", f"{items[0][1].name!r} is new and not a plain folder")
    gone = [layer for lid, items in old.items() if lid not in new for _, layer in items if layer.is_group()]
    for group in gone:
        if any(child.layer_id not in new for child in group.descendants() if not child.is_group() and child.kind != "curves"):
            raise Refused("bytes", f"{group.name!r} went missing with its layers")
    base = base if base is not None else folder_reads(before, rel, parser)
    if memo is not None:
        memo["base"] = base
    now = folder_reads(after, rel, parser)
    if len(base["renders"]) != len(now["renders"]):
        raise Refused("render", "the pictures changed")
    for n, (a, b) in enumerate(zip(base["renders"], now["renders"]), 1):
        if a.shape != b.shape or int(np.abs(a - b).max()) > tol:
            off = int((np.abs(a - b).max(axis=2) > tol).sum()) if a.shape == b.shape else -1
            raise Refused("render", f"picture {n} draws differently ({off} px)")
    gone_ids = {str(lid) for lid in old if lid not in new}
    renamed = {_icon_key(items[0][1].name): _icon_key(new[lid][0][1].name) for lid, items in old.items()
               if lid in new and items[0][1].is_group() and items[0][1].name != new[lid][0][1].name}
    for key in ("parses", "details"):
        was = _folder_details(base[key], now[key], gone_ids, renamed) if key == "details" else base[key]
        found = now[key]
        if spots and key == "details":
            was, found = _no_spots(was), _no_spots(found)
        if was != found:
            raise Refused("reparse", f"the bot reads the card differently ({key})")
    return base


def _no_spots(details):
    """A details read without the header's star and type spots (where the bake would add them)."""
    header = details.get("header") if isinstance(details, dict) else None
    if not isinstance(header, dict):
        return details
    header = {k: ({kk: vv for kk, vv in v.items() if kk not in ("at", "spots")} if isinstance(v, dict) else v)
              for k, v in header.items()}
    return {**details, "header": header}


def _icon_key(name: str) -> str:
    return sub(r"\s+copy(\s+\d+)?$", "", name).strip().lower()     # psd_details._copied's


def _without(value, ids: set):
    """A details read without the layers gone (ids: layer ids as text; a read's "#1,2" keeps the others)."""
    def kept(text):
        if not (isinstance(text, str) and fullmatch(r"#\d+(,\d+)*", text)):
            return text
        left = [n for n in text[1:].split(",") if n not in ids]
        return "#" + ",".join(left) if left else None
    if isinstance(value, dict):
        return {kept(k): _without(v, ids) for k, v in value.items() if kept(k) is not None}
    if isinstance(value, list):
        return [_without(kept(v), ids) for v in value if kept(v) is not None]
    return value


def _folder_details(was: dict, now: dict, ids: set, renamed: dict) -> dict:
    """The details read before, as the folders may change it: layers gone, the bake's star icon named by its renamed
    folder, stars the bake found none of before now found (filed in Stars: Joel Miller's), the Ability layer no longer
    outside Text."""
    was = _without(was, ids)
    header = was.get("header")
    if isinstance(header, dict) and isinstance(header.get("stars"), dict) and header["stars"].get("icon") in renamed:
        header["stars"] = {**header["stars"], "icon": renamed[header["stars"]["icon"]]}
    found = (now.get("header") or {}).get("stars") if isinstance(now.get("header"), dict) else None
    if isinstance(header, dict) and isinstance(header.get("stars"), dict) and not header["stars"].get("units") \
            and isinstance(found, dict) and found.get("units"):
        header["stars"] = found
    template, after = was.get("template"), now.get("template")
    if isinstance(template, dict) and isinstance(after, dict):
        gone = set(template.get("deviations", ())) - set(after.get("deviations", ()))
        if gone <= {"ability layer outside Text"}:
            template["deviations"] = after.get("deviations")
    return was


def standard_folders(path: str, out_path: str, fonts=None, rel: str = "", parser: Optional[PSDParser] = None,
                     base: Optional[dict] = None, isolated: bool = True) -> Foldered:
    """
    The card's folders as the templates' (arrange), at the first FOLDER_LEVELS level folders_gate lets through, then
    the adjustments clipped to a folder moved into it (isolate, gated on its own; isolated False: not, nor hidden Curves
    clipped to a folder deleted, the guards' call: the owner wants each proven in Photoshop), written to out_path; nothing ([]) where the folders are standard,
    the card is no card or nothing holds (Foldered.tried says why). fonts is unused. base: the card's folder_reads.
    """
    result = Foldered()
    if parser is None or folder_kind(rel) is None:
        return result
    types, stars = frozenset(parser.all_types), parser.star_names()
    card = PSDImage.open(path)
    card.repo_path = rel
    art = art_icons(card, rel, parser)
    left = folder_issues(card, rel, parser, art)
    if not left:
        return result
    found, memo, refused = None, {"base": base}, {}     # the card read once; a refused file's sha1 -> why
    for level in FOLDER_LEVELS:
        psd = PSDImage.open(path)
        done, counts = arrange(psd, rel, types, stars, level, isolated, art)
        if not done:
            result.tried.append((level, "nothing to change"))
            continue
        psd_edit.save_psd(psd, out_path)
        with open(out_path, "rb") as fh:
            made = sha1(fh.read()).digest()
        if made in refused:                     # the same file as a level refused: the same refusal
            result.tried.append((level, refused[made]))
            continue
        try:
            folders_gate(path, out_path, rel, parser, memo["base"], memo=memo)
        except Refused as error:
            result.tried.append((level, refused.setdefault(made, str(error))))
            continue
        except Exception as error:              # a read the fork can't make: refused the same
            result.tried.append((level, refused.setdefault(made, f"{type(error).__name__}: {error}")))
            continue
        found = Foldered(done, level, counts, [], result.tried)
        break
    found, base = found or Foldered(tried=result.tried), memo["base"]
    if _loose_icons(card)[1]:
        _icons(path, out_path, rel, parser, base, found, art)
    if isolated:
        _isolated(path, out_path, rel, parser, base, found)
    if not found:
        return Foldered(left=left, tried=found.tried)
    found.left = folder_issues(PSDImage.open(out_path), rel, parser, art)
    return found


def _alpha(unit, box: tuple) -> np.ndarray:
    """A unit drawn alone inside box (x0, y0, x1, y1): its alpha (0-255), its base's effects in, its clips inside it."""
    out = np.zeros((box[3] - box[1], box[2] - box[0]), np.uint16)
    for layer in unit[:1]:
        if layer.is_visible():
            image = layer.composite(viewport=box, apply_icc=False)
            if image is not None:
                out = np.maximum(out, np.asarray(image.convert("RGBA"))[..., 3])
    return out


def _meets(unit, others: list):
    """The first unit of others whose ink meets unit's (an adjustment always does), else None."""
    mine = _boxes(unit[:1])
    for other in others:
        theirs = _boxes(other[:1])
        if mine is None or theirs is None:
            return other
        for a in mine:
            for b in theirs:
                box = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
                if box[0] < box[2] and box[1] < box[3] and ((_alpha(unit, box) > 0) & (_alpha(other, box) > 0)).any():
                    return other
    return None


def _icon_folder(psd: PSDImage, role: str, name: str, images) -> tuple:
    """(role's top-level folder or None, the index a new one goes at: over the highest lower role's folder)."""
    units = _units(psd)
    folder = next((u[0] for u in units if u[0].is_group() and u[0].name == name), None)
    if folder is not None:
        return folder, None
    names = {FOLDER_NAMES.get(r) for r in FOLDER_ORDER[:FOLDER_ORDER.index(role)]} | {images.name}
    at = 0
    for unit in units:
        if unit[0].is_group() and unit[0].name in names:
            at = list(psd).index(unit[len(unit) - 1]) + 1
    return None, at


def _file_icon(psd: PSDImage, unit: list, kind: str, images, filled: dict, done: list, counts: Counter) -> None:
    """An icon's unit to the bottom of Stars or the type folder kind (made when missing), over the ones filed before
    (filled: folder id -> how many)."""
    role = _link_kind(unit[0]) + "s"
    name = FOLDER_NAMES["stars"] if role == "stars" else kind
    folder, at = _icon_folder(psd, role, name, images)
    if folder is None:
        folder = _new_folder(psd, name, FOLDER_COLOURS[role], at)
        done.append(f"{name} folder made")
        counts["folders made"] += 1
    n = filled.get(id(folder), 0)
    for layer in unit:
        layer.move_to(folder, index=n)
        n += 1
    filled[id(folder)] = n
    done.append(f"{unit[0].name!r} out of {images.name!r} into {name}")
    counts["icons out of images"] += 1


def _loose_icons(psd: PSDImage) -> tuple:
    """(images, its linked star and type icons' units) at the top level ((None, []) without images)."""
    images = next((layer for layer in psd if layer.is_group() and layer.name == FOLDER_NAMES["images"]), None)
    if images is None:
        return None, []
    return images, [u for u in _units(images) if not u[0].is_group() and not u[0].clipping and _link_kind(u[0])]


def art_icons(psd: PSDImage, rel: str, parser: PSDParser) -> frozenset:
    """Layer ids of the linked star icons in images that are the card's art: filed in Stars alone (on a copy), the
    parser reads the card otherwise (Hero Cap's two 'Mother' stars: 3 stars, not 1; owner 2026-11-09). They stay put,
    untold."""
    kind = folder_kind(rel)
    images, units = _loose_icons(psd)
    stars = [u for u in units if _link_kind(u[0]) == "star"]
    if kind is None or not stars:
        return frozenset()
    buffer = BytesIO()
    psd.save(buffer)
    data = buffer.getvalue()

    def read(card) -> tuple:
        found = parser.parse(card, rel)
        return found.to_dict(), CardValidator.validate(found)
    was, out = read(PSDImage.open(BytesIO(data))), set()
    for unit in stars:
        trial = PSDImage.open(BytesIO(data))
        trial.repo_path = rel
        ids = [layer.layer_id for layer in unit]
        layers = {layer.layer_id: layer for layer in trial.descendants()}
        where = next(layer for layer in trial if layer.is_group() and layer.name == images.name)
        _file_icon(trial, [layers[i] for i in ids], kind, where, {}, [], Counter())
        if read(trial) != was:
            out.add(unit[0].layer_id)
    return frozenset(out)


def mid_icons(psd: PSDImage, rel: str, types=(), stars=None, done: Optional[list] = None,
              counts: Optional[Counter] = None, art: frozenset = frozenset()) -> list:
    """
    Linked star and type icons (Types/Stars, Types) loose in images, with what clips to them, to the bottom of Stars or
    the card's type folder (made as the templates' when missing), bottom to top (owner 2026-11-09), in place. One stays
    when it is art (art: their layer ids), its folder or images is not plain, or ink it crosses meets its own (or an
    adjustment). Returns why each one stayed; done and counts grow.
    """
    done, counts = done if done is not None else [], counts if counts is not None else Counter()
    kind = folder_kind(rel)
    images, picks = _loose_icons(psd)
    if kind is None or not picks:
        return []
    if not _plain(images) or len(next(u for u in _units(psd) if u[0] is images)) > 1:
        return [f"icons left in {images.name!r}: the folder is not plain"]
    left, filled = [], {}
    for unit in picks:
        icon = unit[0]
        if icon.layer_id in art:
            continue
        role = _link_kind(icon) + "s"
        name = FOLDER_NAMES["stars"] if role == "stars" else kind
        folder, at = _icon_folder(psd, role, name, images)
        if folder is not None and (not _plain(folder) or len(next(u for u in _units(psd) if u[0] is folder)) > 1):
            left.append(f"{icon.name!r} left in {images.name!r}: {name!r} is not plain")
            continue
        above = _units(images)
        above = above[next(i for i, u in enumerate(above) if u[0] is icon) + 1:]
        top = _units(psd)
        first = next(i for i, u in enumerate(top) if u[0] is images) + 1
        last = next(i for i, u in enumerate(top) if u[0] is folder) if folder is not None else \
            next((i for i, u in enumerate(top) if list(psd).index(u[0]) >= at), len(top))
        blocker = _meets(unit, above + top[first:last])
        if blocker is not None:
            left.append(f"{icon.name!r} left in {images.name!r}: {blocker[0].name!r} draws over it")
            continue
        _file_icon(psd, unit, kind, images, filled, done, counts)
    return left


def _icons(path: str, out_path: str, rel: str, parser: PSDParser, base: Optional[dict], found: Foldered,
           art: frozenset) -> None:
    """standard_folders' icon step: mid_icons over its result (or the card), gated on its own; found grows."""
    psd = PSDImage.open(out_path if found else path)
    psd.repo_path = rel
    done, counts = [], Counter()
    mid_icons(psd, rel, frozenset(parser.all_types), parser.star_names(), done, counts, art)
    if not done:
        return
    trial = out_path + ".icons"
    psd_edit.save_psd(psd, trial)
    try:
        folders_gate(path, trial, rel, parser, base, spots=True)
    except Exception as error:                  # Refused or a read the fork can't make: the icons stay
        found.tried.append(("icons", f"{error}"))
        remove(trial)
        return
    move_file(trial, out_path)
    found.extend(done)
    found.counts.update(counts)
    found.level = found.level or "icons"


def isolate(psd: PSDImage, done: list, counts: Counter) -> dict:
    """The adjustments clipped to a top-level folder moved into it, on top, and the folder set to Normal (its layers
    alone adjusted, as the clip did; owner 2026-11-09); {layer id: the record parts it changes} for folders_gate."""
    allow = {}
    standard = set(FOLDER_NAMES.values()) | set(FOLDER_TYPES.values())
    for unit in _units(psd):
        folder, clips = unit[0], unit[1:]
        if (not clips or not folder.is_group() or folder.name not in standard or not all(c.kind in ADJUSTMENTS for c in clips)
                or folder.blend_mode not in (BlendMode.PASS_THROUGH, BlendMode.NORMAL)):
            continue
        for layer in clips:
            layer.move_to(folder)
            layer.clipping = False
            allow[layer.layer_id] = {"blend"}
            done.append(f"{layer.name!r} into {folder.name} (was clipped to it)")
        if folder.blend_mode == BlendMode.PASS_THROUGH:
            folder.blend_mode = BlendMode.NORMAL
            allow[folder.layer_id] = {"blend", "blocks"}
        counts["clipped adjustments in"] += len(clips)
        counts["folders isolated"] += 1
    for folder in [layer for layer in psd if layer.is_group() and layer.name in ("Stars", *FOLDER_TYPES.values())]:
        _nest_isolated(folder, allow, done, counts)
    return allow


def _nest_isolated(folder, allow: dict, done: list, counts: Counter) -> None:
    """A redundant nest that adjustments clip to (Stars/stars under a Color Balance), all the folder draws: the
    adjustments onto the folder's top, the folder set to Normal, the nest flattened (isolate's)."""
    if folder.blend_mode not in (BlendMode.PASS_THROUGH, BlendMode.NORMAL):
        return
    for unit in _units(folder):
        nest, clips = unit[0], unit[1:]
        role = "stars" if folder.name == "Stars" else "types"
        if (not clips or not nest.is_group() or not _plain(nest) or not all(c.kind in ADJUSTMENTS for c in clips)
                or not (NESTED[role].fullmatch(_key(nest.name)) or GENERIC.fullmatch(_key(nest.name)))
                or any(_draws(other) for other in folder if other is not nest and other not in clips)
                or (len(nest) and list(nest)[0].clipping)):
            continue
        for layer in clips:
            layer.move_to(folder)
            layer.clipping = False
            allow[layer.layer_id] = {"blend"}
        if folder.blend_mode == BlendMode.PASS_THROUGH:
            folder.blend_mode = BlendMode.NORMAL
            allow[folder.layer_id] = {"blend", "blocks"}
        i = list(folder).index(nest)
        for j, layer in enumerate(list(nest)):
            layer.move_to(folder, index=i + j)
        folder.remove(nest)
        done.append(f"nested {role} folder {nest.name!r} flattened, {len(clips)} adjustment(s) on it onto {folder.name}")
        counts[f"nested {role} folder"] += 1
        counts["folders isolated"] += 1
        return


def _isolated(path: str, out_path: str, rel: str, parser: PSDParser, base: Optional[dict], found: Foldered) -> None:
    """standard_folders' second step: isolate() over its result (or the card), gated on its own; found grows."""
    current = out_path if found else path
    psd = PSDImage.open(current)
    done, counts = [], Counter()
    allow = isolate(psd, done, counts)
    if not allow:
        return
    trial = out_path + ".isolated"
    psd_edit.save_psd(psd, trial)
    try:
        folders_gate(path, trial, rel, parser, base, allow)
    except Exception as error:                  # Refused or a read the fork can't make: the clips stay
        found.tried.append(("isolate", f"{error}"))
        remove(trial)
        return
    move_file(trial, out_path)
    found.extend(done)
    found.counts.update(counts)
    found.level = found.level or "isolate"


# ========================
# Module Exports
# ========================
__all__ = [
    'AUTO_FIX',
    'BADGE_KEYS',
    'BADGE_TAGS',
    'Context',
    'FIXABLE',
    'Fixed',
    'Issue',
    'SCOPE',
    'STARS_KINDS',
    'WAIVED',
    'badge_tags',
    'card_key',
    'check',
    'clean',
    'clean_layers',
    'clean_plan',
    'badge_plan',
    'composer_layers',
    'curly_layers',
    'every_line',
    'fit_badges',
    'fix',
    'folder_issues',
    'folder_kind',
    'Foldered',
    'folders_gate',
    'icon_layers',
    'inexact_names',
    'kind_of',
    'laid_advances',
    'laid_plan',
    'mark_new',
    'needs_normalise',
    'normalise',
    'Normalised',
    'miscased_names',
    'old_glyphs',
    'one_colour_badges',
    'plain_tags',
    'PIXEL_FIXES',
    'PIXEL_LINES',
    'pixel_kinds',
    'salt_layers',
    'script_layers',
    'script_plan',
    'tag_layers',
    'tags_to_badges',
    'arrange',
    'hidden_curves',
    'rename',
    'rename_titles',
    'reported',
    'redraw',
    'redraw_old_glyphs',
    'relay_badges',
    'reflow',
    'run_fonts',
    'single_line',
    'standard_folders',
    'status_plan',
    'status_texts',
    'status_word_layers',
    'status_words',
    'straighten',
    'title_problem',
    'title_renames',
    'to_normal',
]
