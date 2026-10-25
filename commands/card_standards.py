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
duplicate layer ids.
R9 (Txt2 stories) is fixed by the fork with byte gates, the rest is reported. R11's rename (exact layer names) is built
and gated but reported until the owner says yes (AUTO_FIX). Rule pages are never checked.

    check(psd, rel, context, old=None, added=False) -> [Issue]
    fix(data, rel, issues, work, parser=None, fonts=None) -> Fixed   the 0-pixel fixes, gated (one PSD's new bytes)
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
from commands.font_check import _get, _value, layer_fonts
from commands.psd_edit import BADGE_KEYS, BADGE_TAGS, PAREN, Refused, badge_tags
from commands.psd_parser import (EFFECT_LAYER_TAG, ICON_FONT_PREFIX, TAG, CardValidator, PSDParser, base_name, layer_tag,
                                 variants_of)
from commands.psd_regions import clip, fx_pad, overlay_words
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
    pngs: Optional[frozenset] = None            # the tree's PNG paths, lowercased (R12's pictures); None: not checked


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
        run("R11", rule_part_names, psd, card)
    run("R12", rule_variants, psd, rel, context.pngs)
    run("R13", rule_layer_ids, psd)
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
            fonts: Optional[Callable[[list], Optional[str]]] = None) -> None:
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
    try:
        result.composer, moves, result.composer_left = single_line(current, out, fonts)
    except Exception as error:                  # best effort: never told, the other fixes stand
        result.composer_left = [("", f"{type(error).__name__}: {error}")]
    if result.composer:
        current = out
        codes.append("R8")
    if current == base:
        return
    try:
        result.data = _gated(base, current, rel, parser)
    except Refused as refused:
        result.refused += [(code, str(refused)) for code in codes if code != "R8"]
        result.composer_left += [(name, str(refused)) for name in result.composer]
        result.applied, result.composer, result.data = [], [], b""
        return
    if not moves:
        return
    drawn = join(folder, "R8-redrawn.psd")
    try:                                        # the lines that rewrap drawn again: the guards re-export the PNG
        names = redraw(current, drawn, [name for name, _ in moves], fonts)
        data = _gated(base, drawn, rel, parser)
    except Exception as error:                  # best effort: set, not redrawn
        result.composer_left += [(name, f"rewraps ({why}), not redrawn: {error}") for name, why in moves]
        return
    result.plain, result.data, result.rewrapped = result.data, data, names


def _gated(base: str, path: str, rel: str, parser: Optional[PSDParser]) -> bytes:
    """A fixed PSD's bytes through the size and the reparse gates; raises Refused."""
    if abs(getsize(path) - getsize(base)) > SIZE_MAX * getsize(base):
        raise Refused("size", f"PSD {getsize(base)} -> {getsize(path)} bytes")
    if parser is not None:
        before, after = psd_edit.parse_card(parser, base, rel), psd_edit.parse_card(parser, path, rel)
        if (before["card"], before["alt_names"], before["validator"]) != (after["card"], after["alt_names"], after["validator"]):
            raise Refused("reparse", "the bot reads the fixed PSD differently")
    with open(path, "rb") as fh:
        return fh.read()


def fix(data: bytes, rel: str, issues: list, work: str, parser: Optional[PSDParser] = None,
        fonts: Optional[Callable[[list], Optional[str]]] = None) -> Fixed:
    """
    The fixable issues of one PSD (R11 renames, then R9 story syncs), each gated, then R8's Single-line Composer
    (single_line: never told, a refusal left in composer_left); then the size and the reparse gates over the result. A
    refusal leaves that fix out (and reports it); nothing is fixed on a rule page. Box text that rewraps is redrawn
    (redraw, gated again: data, the result without it in plain).

    Args:
        data: The pushed PSD
        rel: Its repo path
        issues: check()'s issues
        work: A folder for the temporary files
        parser: The offline parser for gate_reparse (skipped without)
        fonts: Gives MDW/Fonts for R8's box text (single_line)

    Returns:
        Fixed
    """
    result = Fixed()
    wanted = [issue for issue in issues if issue.fix and not issue.waived]
    if kind_of(rel) == "rule" or not (wanted or _every_line_in(data)):
        return result
    makedirs(work, exist_ok=True)
    folder = mkdtemp(prefix="fix-", dir=work)   # one per call: cards of one key in two folders never share a file
    try:
        _fix_in(folder, data, rel, wanted, parser, result, fonts)
    finally:
        rmtree(folder, ignore_errors=True)
    return result


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
    'composer_layers',
    'every_line',
    'fix',
    'inexact_names',
    'kind_of',
    'mark_new',
    'rename',
    'reported',
    'redraw',
    'reflow',
    'run_fonts',
    'single_line',
    'title_problem',
]
