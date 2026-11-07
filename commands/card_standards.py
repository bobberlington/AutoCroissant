"""
Guard 2 of the push guards (docs/Campaigns/PUSH_GUARDS.md section 6): a pushed card PSD against the card standards of
docs/Campaigns/CARD_STANDARDIZATION_CAMPAIGN.md. Rules R1-R6, R9 and R11 run over an open PSD (R10, the PNG against
the PSD, is png_export.png_check; R7, the fonts, is the font check's alone since 2026-10-24, owner). R8, the Single-line
Composer (L10), is always set and never told (owner, 2026-10-25): fix() sets it with the other fixes; box text that
rewraps is redrawn (the guards re-export its PNG). R3 and R4 keep only
the real issues (several visible Ability layers, or a hidden one with words the parser reads nowhere; a hidden or
rasterized Name layer the title can't be read from): the template layout they also checked was dropped by the owner on
2026-10-23.
Since the parts editor (PARTS_EDITOR.md §7) R3's several Ability layers and R4's several title layers are notes ("not
standardized yet"), R11 also suggests names for the parts not named Name or Ability, R12 checks variant tags and R13
duplicate layer ids. R14 (owner 2026-10-30) reports card names typed in another letter case (report only); R15 the
ability's or title's text run off the card's edge (report only; existing cards waived as R4).
R9 (Txt2 stories) is fixed by the fork with byte gates, the rest is reported. R11's rename (exact layer names) is built
and gated but reported until the owner says yes (AUTO_FIX). The pixel fixes (PIXEL_FIXES,
silent, owner 2026-10-25) run on every card, in this order: badges laid out at older widths laid out again
(relay_badges), straight apostrophes, icon font glyphs Photoshop drew with an older font drawn again (never one whose
advance moved: relay_badges'), badges and icons fitted to their text (fit_badges: a badge's pill 1.35 times
the text's capital height, its middle on the capitals' middle, even when that rewraps the text; an FX layer on words
that moved is cut again) and badges under an overlay set one-colour (one_colour_badges); each redraws only the changed
glyphs and the guards re-export the PNG. Every save strips the merged image (Maximize Compatibility off, owner
2026-11-07: psd_edit.save_psd). The big pass's normalise (card_sweep alone until the guards land its files, G1): sub and
superscript text set normal at the same size and place (to_normal), the pictures linked (the link step, psd_links),
the merged image stripped, in one gated save.

    check(psd, rel, context, old=None, added=False) -> [Issue]
    fix(data, rel, issues, work, parser=None, fonts=None, only=None) -> Fixed   the fixes, gated (one PSD's new bytes)
    normalise(path, out_path, fonts, rel) -> Normalised           the big pass's rewrite (fix(only="normalise"))
    composer_layers(psd) -> [layer]                               the card text layers set Every-line (R8)
    redraw(path, out_path, names, fonts) -> [name]                the rewrapped layers drawn again (R8)
    BADGE_TAGS, WAIVED                                            the badge vocabulary and the owner's class waivers

No Discord, network or config.
"""
from dataclasses import dataclass, field
from functools import lru_cache
from io import BytesIO
from os import makedirs
from os.path import basename, getsize, join
from re import IGNORECASE, compile as re_compile, escape, fullmatch, search
from shutil import rmtree
from tempfile import mkdtemp
from typing import Callable, Optional

import numpy as np
from psd_tools import PSDImage
from psd_tools.constants import BlendMode, Tag

import commands.psd_edit as psd_edit
from commands.card_edits import CardAuthors
from commands.card_models import APOSTROPHE, straight
from commands.font_check import _get, _value, layer_fonts
from commands.psd_edit import BADGE_KEYS, BADGE_TAGS, ICON_FONT, Refused, badge_spans, badge_tags
from commands.psd_parser import (EFFECT_LAYER_TAG, ICON_FONT_PREFIX, TAG, CardValidator, PSDParser, base_name, layer_tag,
                                 variant_view, variants_of)
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
TXT2_COMPOSER = "29"                            # Txt2's EveryLineComposer (Photoshop CC's numbered keys)
LINE_IOU = 0.9                                  # R8 box text: a line rewraps under it (each line's ink, the fork's
                                                # Single-line layout vs the stored; matches 0.94-1.0, a moved word 0.04-0.34)
LINE_PAD = 2                                    # px around a line's box
NAME_BLOCK = bytes(Tag.UNICODE_LAYER_NAME)      # luni
WAIVED = {                                      # the owner's class waivers (section 6.2): rule -> what and why
    "R2": {"cards": {"tainted lazarus"}, "why": "its header icon (2026-10-22)"},
    "R3": {"cards": {"lit bob-omb", "tainted lazarus", "miracle matter", "twin emperors", "warcraftian druid", "pix"},
           "why": "several-Ability and multi-creature cards stay deviations (question 2, 2026-10-21)"},
    "R3 excluded": {"why": "in the parser's ABILITY_EXCLUSIONS: prints no ability (2026-10-23)"},
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


def _story_props(psd: PSDImage, index: Optional[int]) -> list:
    """The paragraph properties of Txt2 story index (Photoshop CC's numbered keys); [] without it."""
    root = psd_edit._txt2(psd)
    if root is None or index is None:
        return []
    _, _, items, model, _, spelling = psd_edit._story_list(root)
    if spelling:
        raise Refused("unsupported-card", "a Txt2 block with named keys")
    if not 0 <= index < len(items):
        return []
    return [run["0"]["0"]["5"] for run in items[index][model]["5"]["0"]]


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
    """R8: the visible card text layers (COMPOSER_LAYERS) set with the Every-line Composer, in TySh or their Txt2 story."""
    out = []
    for layer in _visible_texts(psd):
        if layer.name.strip().lower() not in COMPOSER_LAYERS:
            continue
        try:
            story = _story_props(psd, psd_edit.story_index(layer))
        except (Refused, KeyError, TypeError, AttributeError):
            story = []
        if every_line(layer) or any(_on(props.get(TXT2_COMPOSER, False)) for props in story):
            out.append(layer)
    return out


def _point_text(layer) -> bool:
    """No box: a line ends only at a return, whatever the composer."""
    shapes = layer._engine_data["EngineDict"]["Rendered"]["Shapes"]["Children"]
    return all(int(_on(shape["Cookie"]["Photoshop"]["ShapeType"])) == 0 for shape in shapes)


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
    R8: set every composer_layers Single-line (TySh and the Txt2 story) and save without a new preview; gated: set back,
    every layer record and document block is the old bytes. Each box text layer's lines are measured (reflow) for
    redraw().

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
    psd_edit.save_psd(psd, out_path)
    new = PSDImage.open(out_path)
    if any(psd_edit._record_index(layer) in records for layer in composer_layers(new)):
        raise Refused("bytes", "a layer is still Every-line")
    values = _composer_values(new, records)
    if set(flipped) - set(values):
        raise Refused("bytes", "a composer value went missing")
    for key in flipped:
        props, name = values[key]
        props[name].value = True
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
    """R4: the title is not left in a hidden or rasterized Name layer (no title read); a title over several layers is a
    note (each a box on the website)."""
    why = title_problem(psd, card)
    if why is None:
        texts = [layer for layer in psd.descendants() if psd_edit._is_text_layer(layer)]
        titles = PSDParser.title_parts(psd, texts, card.replace("_", " "))[0]
        if len(titles) > 1:
            return Issue("R4", f"not standardized yet: {len(titles)} title layers (a box each on the website)",
                         (f"{len(titles)} title layers",))
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
    pairs = tuple(f"{layer.name!r} -> {target!r}" for layers, target in ((titles, "Name"), (abilities, "Ability"))
                  for layer in layers if base_name(layer).strip().lower() != target.lower())
    if not pairs:
        return None
    return Issue("R11", f"{'a part is' if len(pairs) == 1 else 'parts are'} not named so (the website labels "
                        f"{'it' if len(pairs) == 1 else 'them'}): rename {', '.join(pairs)}", pairs, discord=False)


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
        ability = out[len(out) - 1]
        if ability is not None and ability.code == "R3" and key in {CardAuthors.key(n) for n in CardValidator.ABILITY_EXCLUSIONS}:
            ability.waived = WAIVED["R3 excluded"]["why"]
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


def _every_line_in(data: bytes) -> bool:
    try:
        return bool(composer_layers(PSDImage.open(BytesIO(data))))
    except Exception:                           # unreadable: nothing to set
        return False


def _fix_in(folder: str, data: bytes, rel: str, wanted: list, parser: Optional[PSDParser], result: Fixed,
            fonts: Optional[Callable[[list], Optional[str]]] = None, composer: bool = True, pixels: tuple = (),
            whole: bool = False) -> None:
    """fix()'s work in its own temporary folder."""
    from commands.txt2_repair import repair
    base = join(folder, "pushed.psd")
    with open(base, "wb") as fh:
        fh.write(data)
    current, codes = base, []
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
    out, moves = join(folder, "R8.psd"), []
    if composer:
        try:
            result.composer, moves, result.composer_left = single_line(current, out, fonts)
        except Exception as error:              # best effort: never told, the other fixes stand
            result.composer_left = [("", f"{type(error).__name__}: {error}")]
    if result.composer:
        current = out
        codes.append("R8")
    if current != base:
        try:
            result.data = _gated(base, current, rel, parser)
        except Refused as refused:
            result.refused += [(code, str(refused)) for code in codes if code != "R8"]
            result.composer_left += [(name, str(refused)) for name in result.composer]
            result.applied, result.composer, result.data = [], [], b""
            current, moves = base, []           # the pixel fixes go on the pushed PSD
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
        if (before["card"], before["alt_names"], before["validator"]) != (after["card"], after["alt_names"], after["validator"]):
            raise Refused("reparse", "the bot reads the fixed PSD differently")
    with open(path, "rb") as fh:
        return fh.read()


def fix(data: bytes, rel: str, issues: list, work: str, parser: Optional[PSDParser] = None,
        fonts: Optional[Callable[[list], Optional[str]]] = None, only: Optional[str] = None, whole: bool = False,
        skip: tuple = ()) -> Fixed:
    """
    The fixable issues of one PSD (R11 renames, then R9 story syncs), each gated, then R8's Single-line Composer
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
        fonts: Gives MDW/Fonts for R8's box text (single_line) and the pixel fixes
        only: One PIXEL_FIXES kind alone (a sweep: card_sweep), no other fix
        whole: fit_badges may redraw a layer whole (a sweep's --whole)
        skip: PIXEL_FIXES kinds left out (the guards convert in a step of their own)

    Returns:
        Fixed
    """
    result = Fixed()
    wanted = [] if only else [issue for issue in issues if issue.fix and not issue.waived]
    composer = not only and _every_line_in(data)
    pixels = tuple(kind for kind in pixel_kinds(data, (only,) if only else (), rel) if kind not in skip)
    if not (wanted or composer or pixels):
        return result
    makedirs(work, exist_ok=True)
    folder = mkdtemp(prefix="fix-", dir=work)   # one per call: cards of one key in two folders never share a file
    try:
        _fix_in(folder, data, rel, wanted, parser, result, fonts, composer, pixels, whole)
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
    out = {key: _straightened(parsed[key]) for key in ("card", "alt_names", "validator")}
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
    from collections import Counter
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


PIXEL_FIXES = {     # kind -> (finds(psd) -> [layer], fix(path, out_path, fonts) -> [layer name], the parse it makes or None)
    "badge widths": (icon_layers, relay_badges, None),   # first: straighten and old glyphs lay out at the current widths
    "apostrophes": (curly_layers, straighten, straight_parse),
    "old glyphs": (icon_layers, redraw_old_glyphs, None),
    "badges": (icon_layers, fit_badges, None),
    "overlay badges": (salt_layers, one_colour_badges, None),
    "tags": (tag_layers, tags_to_badges, None),
    "alt template": (alt_bar_rows, alt_template, None),
    "normalise": (needs_normalise, normalise, None),     # last: linked pictures, normal text, no merged image (silent)
}
SWEEP_ONLY = {"alt template", "normalise"}   # run by card_sweep alone, never by the push guards (owner's call,
#   2026-11-06); normalise until the guards land its files silently (big pass G1: then out of this set)
PIXEL_LINES = {"badge widths": "badges laid out again at their new widths", "apostrophes": "straight apostrophes",
               "old glyphs": "old icon font glyphs redrawn",
               "badges": "badges and icons fitted to their text",
               "overlay badges": "badges under an overlay in one colour", "tags": "timing tags and costs set as badges",
               "alt template": "the template's stat bars, bar shapes, text layout and clip shapes",
               "normalise": "pictures linked, text set normal, no merged image"}


def pixel_kinds(psd, kinds: tuple = (), rel: str = "") -> tuple:
    """The pixel fix kinds (all but SWEEP_ONLY by default) whose layers the PSD (open, or its bytes) has; rel: its repo
    path (the link step judges its links from there)."""
    try:
        psd = PSDImage.open(BytesIO(psd)) if isinstance(psd, bytes) else psd
        if rel:
            psd.repo_path = rel
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
            elif apply is normalise:
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
        if apply is normalise:                  # the PNG follows its redrawn layers alone (a link's new raster)
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
    'badge_plan',
    'composer_layers',
    'curly_layers',
    'every_line',
    'fit_badges',
    'fix',
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
    'rename',
    'reported',
    'redraw',
    'redraw_old_glyphs',
    'relay_badges',
    'reflow',
    'run_fonts',
    'single_line',
    'straighten',
    'title_problem',
    'to_normal',
]
