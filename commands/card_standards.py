"""
Guard 2 of the push guards (docs/Campaigns/PUSH_GUARDS.md section 6): a pushed card PSD against the card standards of
docs/Campaigns/CARD_STANDARDIZATION_CAMPAIGN.md. Rules R1-R6, R9 and R11 run over an open PSD (R10, the PNG against
the PSD, is png_export.png_check; R7, the fonts, is the font check's alone since 2026-10-24, owner). R8, the Single-line
Composer (L10), is always set and never told (owner, 2026-10-25): fix() sets it with the other fixes; box text that
rewraps is redrawn (the guards re-export its PNG). R3 and R4 keep only
the real issues (several visible Ability layers, or a hidden one with words the parser reads nowhere; a hidden or
rasterized Name layer the title can't be read from): the template layout they also checked was dropped by the owner on
2026-10-23.
R9 (Txt2 stories) is fixed by the fork with byte gates, the rest is reported. R11's rename (exact layer names) is built
and gated but reported until the owner says yes (AUTO_FIX). Rule pages are never checked. The pixel fixes (PIXEL_FIXES,
silent, owner 2026-10-25) run on every card and rule page: straight apostrophes, and icon font glyphs Photoshop drew with
an older font drawn again; on cards only, badges and icons fitted to their text (fit_badges: a badge's pill 1.35 times
the text's capital height, its middle on the capitals' middle); each redraws only the changed glyphs and the guards
re-export the PNG.

    check(psd, rel, context, old=None, added=False) -> [Issue]
    fix(data, rel, issues, work, parser=None, fonts=None, only=None) -> Fixed   the fixes, gated (one PSD's new bytes)
    composer_layers(psd) -> [layer]                               the card text layers set Every-line (R8)
    redraw(path, out_path, names, fonts) -> [name]                the rewrapped layers drawn again (R8)
    BADGE_TAGS, WAIVED                                            the badge vocabulary and the owner's class waivers

No Discord, network or config.
"""
from dataclasses import dataclass, field
from io import BytesIO
from os import makedirs
from os.path import basename, getsize, join
from shutil import rmtree
from tempfile import mkdtemp
from typing import Callable, Optional

from psd_tools import PSDImage
from psd_tools.constants import Tag

import commands.psd_edit as psd_edit
from commands.card_edits import CardAuthors
from commands.card_models import APOSTROPHE, straight
from commands.font_check import _get, _value, layer_fonts
from commands.psd_edit import BADGE_KEYS, BADGE_TAGS, PAREN, Refused, badge_tags
from commands.psd_parser import EFFECT_LAYER_TAG, ICON_FONT_PREFIX, CardValidator, PSDParser
from commands.psd_regions import clip, fx_pad, overlay_words, union_box
from global_config import RULEBOOK_DIR

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
}


# ========================
# Data
# ========================
@dataclass
class Issue:
    """One standards finding on one PSD."""
    code: str                                   # R1..R11
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


# ========================
# Helper Functions
# ========================
def kind_of(rel: str) -> str:
    """The card's class from its repo path."""
    if RULEBOOK_DIR in rel.split("/"):
        return "rule"
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
    fonts = psd_edit.font_map(fonts_dir)
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
    psd_edit.save_psd(psd, out_path, update_preview=False)
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
    R8: the named text layers (lines that rewrap Single-line) drawn again whole by the fork, and the stored preview
    patched over them (merge_preview), so the guards' export can follow (png_export.export); gated: only those records'
    pixels, rectangle and TySh and the preview change.

    Args:
        fonts: single_line()'s

    Returns:
        The layer names redrawn; raises Refused("bytes") or ("unsupported-card": no preview to patch)
    """
    import numpy as np
    before, psd = PSDImage.open(path), PSDImage.open(path)
    preview = psd_edit.stored_preview(before)
    if preview is None:
        raise Refused("unsupported-card", "no stored preview to patch")
    layers = [layer for layer in _visible_texts(psd) if layer.name.strip() in names]
    if not layers:
        return []
    boxes = []
    for layer in layers:
        old = layer.bbox
        layer.rasterize(fonts=psd_edit.font_map(fonts(layer_fonts(layer))))
        pad = fx_pad(layer) + LINE_PAD
        boxes += [(x0 - pad, y0 - pad, x1 + pad, y1 + pad) for x0, y0, x1, y1 in (old, layer.bbox) if x1 > x0 and y1 > y0]
    region = clip((min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes),
                   max(b[3] for b in boxes)), psd.width, psd.height)
    was, now = psd_edit.render_pair(before, region, False)[1], psd_edit.render_pair(psd, region, False)[1]
    moved = np.asarray(was.convert("RGBA")) != np.asarray(now.convert("RGBA"))
    psd_edit.save_psd(psd, out_path, preview=psd_edit.merge_preview(preview, [(moved.any(axis=2), now, region)]))
    records = {psd_edit._record_index(layer) for layer in layers}
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
    new = PSDImage.open(out_path)
    for part in ("header", "color_mode_data", "image_resources"):
        if psd_edit._raw(getattr(before._record, part)) != psd_edit._raw(getattr(new._record, part)):
            raise Refused("bytes", f"{part} changed")
    if psd_edit._document_blocks(before) != psd_edit._document_blocks(new):
        raise Refused("bytes", "document blocks changed")
    return [layer.name.strip() for layer in layers]


def _unique(items) -> tuple:
    return tuple(dict.fromkeys(items))


def _list(items, quote: bool = False) -> str:
    items = [f'"{item}"' if quote else item for item in items]
    return items[0] if len(items) == 1 else ", ".join(items[:len(items) - 1]) + " and " + items[len(items) - 1]


# ========================
# The rules
# ========================
def rule_tags(psd: PSDImage) -> Optional[Issue]:
    """R1: timing tags in a visible Ability layer set in the stoogedings badge font (L1)."""
    bad = []
    for layer in _visible_texts(psd):
        if layer.name.strip().lower() != "ability":
            continue
        try:
            text, runs = run_fonts(layer)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            continue
        for match in PAREN.finditer(text):
            if match.group(1).lower() not in BADGE_KEYS:
                continue
            fonts = {font for start, end, font in runs if start < match.end() and end > match.start()}
            if not fonts or any(not font.lower().startswith(ICON_FONT_PREFIX) for font in fonts):
                bad.append(match.group(0))
    if not bad:
        return None
    tags = _unique(bad)
    return Issue("R1", f"timing tag{'s' if len(tags) > 1 else ''} {_list(tags)} not set in the stoogedings badge font", tags)


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
             if psd_edit._is_text_layer(layer) and layer.name.strip().lower() == "ability"]
    count = sum(1 for layer in named if layer.is_visible())
    if count > 1:
        return Issue("R3", f"{count} visible Ability layers (one is the standard)", (f"{count} visible Ability layers",))
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
    named = [layer for layer in psd.descendants() if not layer.is_group() and layer.name.strip().lower() == "name"]
    if not named or any(psd_edit._is_text_layer(layer) and layer.is_visible() for layer in named):
        return None
    if any(not psd_edit._is_text_layer(layer) for layer in named):
        return "rasterized name layer"
    return "hidden name layer"


def rule_name(psd: PSDImage, card: str) -> Optional[Issue]:
    """R4: the title is not left in a hidden or rasterized Name layer (no title read)."""
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
        target = EXACT.get(layer.name.strip().lower())
        if layer.kind == "type" and target and layer.name != target:
            out.append((layer, target))
            bases[layer.name] = target
    for layer in psd.descendants():
        base, tag, rest = layer.name.partition(EFFECT_LAYER_TAG)
        if tag and base in bases:
            out.append((layer, bases[base] + tag + rest))
    return out


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
def check(psd: PSDImage, rel: str, context: Context, old: Optional[PSDImage] = None, added: bool = False) -> list:
    """
    The standards issues of one card PSD.

    Args:
        psd: The pushed PSD
        rel: Its repo path (the class: scope, waivers)
        context: Types/, the chart, the story sync
        old: The PSD before the push (R4 regressions; "(new)" marks)
        added: A new card (R4 applies in full)

    Returns:
        [Issue] in rule order, waived ones included (waived set); [] for a rule page
    """
    kind = kind_of(rel)
    if kind == "rule":
        return []
    key = card_key(rel)
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
        stem = basename(rel)
        card = stem[:len(stem) - 4]
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
    """The part of the document other than the layer records that changed, or None."""
    for part in ("header", "color_mode_data", "image_resources", "image_data"):
        if psd_edit._raw(getattr(old._record, part)) != psd_edit._raw(getattr(new._record, part)):
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
    renames = inexact_names(psd)
    if not renames:
        return []
    indices = set()
    records = psd._record.layer_and_mask_information.layer_info.layer_records
    for layer, target in renames:
        indices.add(next(i for i, record in enumerate(records) if record is layer._record))
        layer.name = target
    psd_edit.save_psd(psd, out_path, update_preview=False)
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
            fonts: Optional[Callable[[list], Optional[str]]] = None, composer: bool = True, pixels: tuple = ()) -> None:
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
        _pixel_fixes(folder, base, current, rel, parser, result, fonts, pixels)


def _gated(base: str, path: str, rel: str, parser: Optional[PSDParser], reads: Optional[Callable] = None,
           sized: bool = True) -> bytes:
    """A fixed PSD's bytes through the size and the reparse gates (reads: what the fix makes of the old parse; sized
    False: a pixel fix, its bytes proven by _only_pixels, the preview re-encoded); raises Refused."""
    if sized and abs(getsize(path) - getsize(base)) > SIZE_MAX * getsize(base):
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
        fonts: Optional[Callable[[list], Optional[str]]] = None, only: Optional[str] = None) -> Fixed:
    """
    The fixable issues of one PSD (R11 renames, then R9 story syncs), each gated, then R8's Single-line Composer
    (single_line: never told, a refusal left in composer_left); then the size and the reparse gates over the result. A
    refusal leaves that fix out (and reports it); a rule page takes the pixel fixes alone. Box text that rewraps is
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

    Returns:
        Fixed
    """
    result = Fixed()
    rule = kind_of(rel) == "rule"               # a rule page takes the pixel fixes alone
    wanted = [] if rule or only else [issue for issue in issues if issue.fix and not issue.waived]
    composer = not (rule or only) and _every_line_in(data)
    pixels = tuple(kind for kind in pixel_kinds(data, (only,) if only else ()) if not (rule and kind in CARD_PIXELS))
    if not (wanted or composer or pixels):
        return result
    makedirs(work, exist_ok=True)
    folder = mkdtemp(prefix="fix-", dir=work)   # one per call: cards of one key in two folders never share a file
    try:
        _fix_in(folder, data, rel, wanted, parser, result, fonts, composer, pixels)
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
    return psd_edit.font_map(folder)


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


def old_glyphs(layer, fonts) -> list:
    """
    The icon font glyphs Photoshop drew with an older version of the font: their stored pixels against the current
    font's render (_off_share past OLD_GLYPH_BAD).

    Args:
        fonts: A FontMap (psd_edit.font_map)

    Returns:
        Indices into the layout's glyphs (capture(layer, fonts).layout)
    """
    from psd_tools.composite.text import capture, render
    from psd_tools.composite.text.splice import _glyph_box
    before = capture(layer, fonts)
    if before.rgba is None or before.bbox is None:
        return []
    drawn = render(layer, fonts=fonts)
    pad = OLD_GLYPH_SHIFT + 2
    boxes = [b for b in (before.bbox, drawn.bbox) if b is not None]
    window = (min(b[0] for b in boxes) - pad, min(b[1] for b in boxes) - pad,
              max(b[2] for b in boxes) + pad, max(b[3] for b in boxes) + pad)
    stored, mine = _premultiplied(before.rgba, before.bbox, window), _premultiplied(drawn.rgba, drawn.bbox, window)
    out = []
    for index, glyph in enumerate(before.layout.glyphs):
        name = glyph.face.ft.postscript_name
        name = name.decode("latin-1") if isinstance(name, bytes) else str(name or "")
        if not name.lower().startswith(ICON_FONT_PREFIX):
            continue
        box = _glyph_box(glyph, before.model.transform, 1)
        if box is None:
            continue
        x0, y0, x1, y1 = (box[0] - window[0], box[1] - window[1], box[2] - window[0], box[3] - window[1])
        x0, y0 = max(x0, pad), max(y0, pad)
        x1, y1 = min(x1, window[2] - window[0] - pad), min(y1, window[3] - window[1] - pad)
        if x1 > x0 and y1 > y0 and _off_share(stored, mine, (x0, y0, x1, y1)) > OLD_GLYPH_BAD:
            out.append(index)
    return out


def _save_patched(before: PSDImage, psd: PSDImage, boxes: list, out_path: str) -> None:
    """psd saved with its stored preview patched where the boxes' render changed (redraw's way)."""
    import numpy as np
    preview = psd_edit.stored_preview(before)
    if preview is None:
        raise Refused("unsupported-card", "no stored preview to patch")
    region = clip(union_box(boxes), psd.width, psd.height)
    was, now = psd_edit.render_pair(before, region, False)[1], psd_edit.render_pair(psd, region, False)[1]
    moved = np.asarray(was.convert("RGBA")) != np.asarray(now.convert("RGBA"))
    psd_edit.save_psd(psd, out_path, preview=psd_edit.merge_preview(preview, [(moved.any(axis=2), now, region)]))


def _only_pixels(path: str, out_path: str, records: set, stories: Optional[list] = None) -> None:
    """Gate: only these records' pixels, rectangle and TySh changed, and (stories) only these Txt2 stories, else the
    document is the old bytes but its preview; raises Refused("bytes")."""
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
    for part in ("header", "color_mode_data", "image_resources"):
        if psd_edit._raw(getattr(before._record, part)) != psd_edit._raw(getattr(new._record, part)):
            raise Refused("bytes", f"{part} changed")
    if stories is None:
        if psd_edit._document_blocks(before) != psd_edit._document_blocks(new):
            raise Refused("bytes", "document blocks changed")
    else:
        psd_edit.gate_stories(before, new, stories)


def _padded(layer, old: tuple) -> list:
    pad = fx_pad(layer) + LINE_PAD
    return [(x0 - pad, y0 - pad, x1 + pad, y1 + pad) for x0, y0, x1, y1 in (old, tuple(layer.bbox)) if x1 > x0 and y1 > y0]


def _by_glyph(layer, report) -> None:
    """Refuses a redraw the engine did not splice by glyph (its whole-layer fallback)."""
    spliced = getattr(report, "splice", None)
    if spliced is None or spliced.fallback:
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
    before, psd = PSDImage.open(path), PSDImage.open(path)
    layers = curly_layers(psd)
    if not layers:
        return []
    boxes, stories = [], []
    for layer in layers:
        found = _fonts_of(layer, fonts)
        old, keep = tuple(layer.bbox), capture(layer, found)
        if psd_edit.story_index(layer) is not None:
            stories.append(psd_edit.story_index(layer))
        text = layer.text
        for at in reversed([i for i, c in enumerate(text) if c == APOSTROPHE]):
            text = text[:at] + "'" + text[at + 1:]
            layer.set_text(text, keep="runs")
        _by_glyph(layer, layer.rasterize(fonts=found, keep=keep, unit=psd_edit.SPLICE_UNIT))
        boxes += _padded(layer, old)
    _save_patched(before, psd, boxes or [(0, 0, 1, 1)], out_path)
    _only_pixels(path, out_path, {psd_edit._record_index(layer) for layer in layers}, stories)
    if curly_layers(PSDImage.open(out_path)):
        raise Refused("bytes", "a curly apostrophe is left")
    return [layer.name.strip() for layer in layers]


def redraw_old_glyphs(path: str, out_path: str, fonts: Optional[Callable[[list], Optional[str]]]) -> list:
    """
    Icon font glyphs Photoshop drew with an older font (old_glyphs) drawn again with the current one, those glyphs alone
    (a splice; refused when it would redraw the layer whole), the stored preview patched; gated (_only_pixels).

    Returns:
        The layer names redrawn; raises Refused or the engine's errors
    """
    from dataclasses import replace
    from psd_tools.composite.text import capture
    before, psd = PSDImage.open(path), PSDImage.open(path)
    boxes, records, names = [], set(), []
    for layer in icon_layers(psd):
        found = _fonts_of(layer, fonts)
        stale = old_glyphs(layer, found)
        if not stale:
            continue
        keep, old = capture(layer, found), tuple(layer.bbox)
        glyphs = tuple(replace(g, fill=STALE_FILL) if i in stale else g for i, g in enumerate(keep.layout.glyphs))
        _by_glyph(layer, layer.rasterize(fonts=found, keep=replace(keep, layout=replace(keep.layout, glyphs=glyphs)),
                                         unit="glyph"))
        boxes += _padded(layer, old)
        records.add(psd_edit._record_index(layer))
        names.append(layer.name.strip())
    if not names:
        return []
    _save_patched(before, psd, boxes, out_path)
    _only_pixels(path, out_path, records)
    return names


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
        fits = psd_edit.badge_fits(layer, psd_edit.font_map(folder))
        wants = [(fit, keys) for fit in fits for keys in [psd_edit.badge_wants(fit)] if keys]
        if wants:
            out.append((layer, wants))
    return out


def _lines(layer, fonts, words: bool = False) -> list:
    """The layout's lines (start, end, baseline); words: and every glyph outside the icon font where it sits."""
    from psd_tools.composite.text.layout import layout
    from psd_tools.composite.text.model import TextModel
    placed = layout(TextModel.from_layer(layer), fonts)
    out = [(line.start, line.end, round(line.y, 2)) for line in placed.lines]
    if words:
        out += [(glyph.cluster, round(glyph.x, 2), round(glyph.y, 2)) for glyph in placed.glyphs
                if not glyph.face.key[0].lower().endswith(".ttf") or ICON_FONT_PREFIX not in glyph.face.key[0].lower()]
    return out


def _set(layer, wants: list) -> None:
    for fit, keys in wants:
        layer.set_span_style(fit["start"], fit["end"], **keys)


def _settle(path: str, record: int, wants: list, faces) -> tuple:
    """
    The style keys a layer takes, tried on a fresh copy: all of them, else (a resize would move its lines, or its words
    under an FX layer cut from them) the shifts alone, else none. Returns (wants, why some were left out or "").
    """
    def moves(option: list) -> bool:
        psd = PSDImage.open(path)
        layer = next(layer for layer in psd.descendants() if layer.kind == "type" and psd_edit._record_index(layer) == record)
        words = bool(psd_edit.fx_layers(psd, layer))    # an FX layer holds its words' pixels where they are
        lines = _lines(layer, faces, words)
        _set(layer, option)
        return _lines(layer, faces, words) != lines

    if not moves(wants):
        return wants, ""
    shifts = [(fit, {"BaselineShift": fit["own_shift"]}) for fit, keys in wants
              if abs(fit["off_px"]) > psd_edit.FIT_PX and abs(fit["off_pct"]) > psd_edit.FIT_PCT]
    if any("FontSize" in keys for _, keys in wants) and not (shifts and moves(shifts)):
        return shifts, "resizing its badges would move its lines or words" + (": shifted only" if shifts else "")
    return [], "fitting its badges would move its lines or words"


def fit_badges(path: str, out_path: str, fonts: Optional[Callable[[list], Optional[str]]],
               left: Optional[list] = None) -> list:
    """
    Badges (owner 2026-10-25), the PIXEL_FIXES kind: badge_plan's badges and icons set (FontSize, BaselineShift; TySh
    and the Txt2 story), their layers' changed glyphs redrawn (a splice), the stored preview patched. A resize that
    would move the layer's lines falls back to the shift alone (left gets why). Gated (_only_pixels, the stories by
    gate_stories), then checked: each fitted one within FIT_CHECK px of its place and at its computed size.

    Returns:
        The layer names fitted; raises Refused or the engine's errors
    """
    from psd_tools.composite.text import capture
    before, psd = PSDImage.open(path), PSDImage.open(path)
    boxes, done = [], []                        # (record, layer name, story, faces, spans)
    for layer, wants in badge_plan(psd, fonts):
        name, faces = layer.name.strip(), psd_edit.font_map(fonts(layer_fonts(layer)))
        wants, why = _settle(path, psd_edit._record_index(layer), wants, faces)
        if why and left is not None:
            left.append((name, why))
        if not wants:
            continue
        old, keep = tuple(layer.bbox), capture(layer, faces)
        _set(layer, wants)
        _by_glyph(layer, layer.rasterize(fonts=faces, keep=keep, unit=psd_edit.SPLICE_UNIT))
        boxes += _padded(layer, old)
        done.append((psd_edit._record_index(layer), name, psd_edit.story_index(layer), faces,
                     {(fit["start"], fit["end"]): "FontSize" in keys for fit, keys in wants}))
    if not done:
        return []
    _save_patched(before, psd, boxes, out_path)
    _only_pixels(path, out_path, {record for record, *_ in done}, [story for _, _, story, *_ in done if story is not None])
    layers = {psd_edit._record_index(layer): layer for layer in PSDImage.open(out_path).descendants() if layer.kind == "type"}
    for record, name, _, faces, spans in done:
        for fit in psd_edit.badge_fits(layers[record], faces, list(spans)):
            if abs(fit["off_px"]) > FIT_CHECK or (spans[(fit["start"], fit["end"])] and abs(fit["size_off"]) > 0.001):
                raise Refused("bytes", f"{name} {fit['token']}: {fit['off_px']:+.1f} px off after the fit")
    return [name for _, name, *_ in done]


def badges_left(path: str, fonts: Optional[Callable[[list], Optional[str]]]) -> list:
    """What is still off after fit_badges: [(layer name, token, px off, size off)] (badge_plan)."""
    return [(layer.name.strip(), fit["token"], fit["off_px"], fit["size_off"])
            for layer, wants in badge_plan(PSDImage.open(path), fonts) for fit, _ in wants]


PIXEL_FIXES = {     # kind -> (finds(psd) -> [layer], fix(path, out_path, fonts) -> [layer name], the parse it makes or None)
    "apostrophes": (curly_layers, straighten, straight_parse),
    "old glyphs": (icon_layers, redraw_old_glyphs, None),
    "badges": (icon_layers, fit_badges, None),
}
PIXEL_LINES = {"apostrophes": "straight apostrophes", "old glyphs": "old icon font glyphs redrawn",
               "badges": "badges and icons fitted to their text"}
CARD_PIXELS = ("badges",)   # not on rule pages: their badges are laid out by hand (legends, tables)


def pixel_kinds(psd, kinds: tuple = ()) -> tuple:
    """The pixel fix kinds (all by default) whose layers the PSD (open, or its bytes) has."""
    try:
        psd = PSDImage.open(BytesIO(psd)) if isinstance(psd, bytes) else psd
        return tuple(kind for kind in kinds or tuple(PIXEL_FIXES) if PIXEL_FIXES[kind][0](psd))
    except Exception:                           # unreadable: nothing to fix
        return ()


def _pixel_fixes(folder: str, base: str, current: str, rel: str, parser: Optional[PSDParser], result: Fixed,
                 fonts: Optional[Callable[[list], Optional[str]]], kinds: tuple) -> None:
    """fix()'s pixel fixes over current (the other fixes' result), each gated on its own; plain becomes the PSD without
    them (the pushed one when they are all) unless R8 redrew."""
    plain = result.data
    for kind in kinds:
        _, apply, reads = PIXEL_FIXES[kind]
        out = join(folder, kind.replace(" ", "-") + ".psd")
        try:
            names = apply(current, out, fonts)
            if not names:
                continue
            data = _gated(current, out, rel, parser, reads, sized=False)
        except Exception as error:              # best effort, never told: plan.json says why
            result.pixel_left.append((kind, f"{type(error).__name__}: {error}"))
            continue
        result.redrawn[kind], result.data, current = names, data, out
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
    'badges_left',
    'composer_layers',
    'curly_layers',
    'every_line',
    'fit_badges',
    'fix',
    'icon_layers',
    'inexact_names',
    'kind_of',
    'mark_new',
    'old_glyphs',
    'PIXEL_FIXES',
    'PIXEL_LINES',
    'pixel_kinds',
    'rename',
    'reported',
    'redraw',
    'redraw_old_glyphs',
    'reflow',
    'run_fonts',
    'single_line',
    'straighten',
    'title_problem',
]
