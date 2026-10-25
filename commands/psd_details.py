"""
The PSD facts behind the website's card "Details" dropdown: fonts, text layers, effects, smart objects, canvas, type-icon
style, stat-row encoding, status-word layers, template deviations, the header's star and type icon boxes (where the bake
would put them: the website mock-up's, psd_edit's geometry) and a compact layer tree. One record per card PSD,
keyed by its git blob sha, in the card data's db/psd_details.json (the Card sync Action's build).

gather(psd, rel, name, index, vocab) -> the record: pure, never raises for a card (a failed section is a short
"error: ..." string). load, text_of (the file's text), prune(cards, shas) to the tree. The build
gathers every re-parsed card's record, its "supplied" flags from font_index() (font_check.local_fonts:
PSD_TOOLS_FONT_DIRS, else the checkout's MDW/Fonts), then refills every record's flags from that index
(records gathered without one, or before MDW/Fonts changed). No Discord or config imports; stock psd-tools. Pillow's ImageCms
names the ICC profile when it is built in.
"""
from datetime import datetime, timezone
from io import BytesIO
from json import dumps, load as json_load
from os import PathLike
from os.path import isdir
from re import IGNORECASE, fullmatch, search, sub
from typing import Optional

from psd_tools import PSDImage
from psd_tools.constants import Resource

from commands.font_check import FontIndex, dir_fonts, layer_fonts, local_fonts, tree_fonts
from commands.psd_edit import (
    STAR_FOLDER, Refused, _Unit, _core, _icon_reason, _icon_spots, _median_size, _moved_box, _reading, _star_count,
    _star_layouts, _star_units, _stars_total, _texts, offline_parser, plan_stars, type_row,
)
from commands.psd_parser import EFFECT_LAYER_TAG, ICON_FONT_PREFIX, PARSER_VERSION, PSDParser
from commands.psd_regions import STATS, badge_layers, overlay_words, stat_rows
from global_config import RULEBOOK_DIR

# ========================
# Configuration
# ========================
VERSION = 1                                 # the file's "version"
TEXT_CHARS = 200                            # text kept per type layer
TREE_CHILDREN = 80                          # children kept per group
TREE_DEPTH = 8                              # groups nested deeper show one "more" entry
MORE = ["…", "more", True]             # the tree's cut marker
DRAWN = ("pixel", "smartobject", "shape")   # digit layers of these kinds are drawn bar cells
HEADER_STARS = 10                           # star counts the header section lays out (1..10)
HEADER_ICONS = 8                            # type icons it keeps spots for
_indexes: dict = {}                         # (fonts dir, its files) -> FontIndex: built once per MDW/Fonts change
_parsers: dict = {}                         # vocab -> offline_parser: the header's type rule


# ========================
# Helper Functions
# ========================
def _value(item):
    return getattr(item, "value", item)


def _ancestors(layer):
    """The enclosing groups, innermost first."""
    parent = layer.parent
    while parent is not None and not isinstance(parent, PSDImage):
        yield parent
        parent = parent.parent


def layer_path(layer) -> str:
    """"Group/Sub/Layer" by name."""
    return "/".join(reversed([layer.name, *(owner.name for owner in _ancestors(layer))]))


def _clean(text: str) -> str:
    """Photoshop's line ends and byte-order mark out, clipped to TEXT_CHARS."""
    return (text or "").replace("﻿", "").replace("\r", "\n").replace("\x03", "\n")[:TEXT_CHARS]


def _colour(values) -> Optional[str]:
    """"#rrggbb" from a FillColor's Values ([a, r, g, b], or [a, grey]); None otherwise."""
    if values is None:
        return None
    parts = [max(0, min(255, int(round(float(_value(v)) * 255)))) for v in values]
    if len(parts) == 4:
        return "#%02x%02x%02x" % (parts[1], parts[2], parts[3])
    if len(parts) == 2:
        return "#%02x%02x%02x" % (parts[1], parts[1], parts[1])
    return None


def _style(layer) -> tuple:
    """(font, size, colour) of a type layer: its first run's, the Normal sheet's where the run says nothing."""
    resources, engine = layer.resource_dict, layer.engine_dict
    font_set = [str(_value(f.get("Name", ""))) for f in resources.get("FontSet") or ()]
    sheets = resources.get("StyleSheetSet") or []
    normal_index = int(_value(resources.get("TheNormalStyleSheet", 0)))
    normal = sheets[normal_index].get("StyleSheetData") or {} if 0 <= normal_index < len(sheets) else {}
    runs = (engine.get("StyleRun") or {}).get("RunArray") or []
    run = (runs[0].get("StyleSheet") or {}).get("StyleSheetData") or {} if runs else {}

    def setting(key):
        return run.get(key) if run.get(key) is not None else normal.get(key)
    used = layer_fonts(layer)
    index = int(_value(setting("Font") or 0))
    font = used[0] if used else (font_set[index] if 0 <= index < len(font_set) else None)
    size = setting("FontSize")
    colour = (setting("FillColor") or {}).get("Values")
    return font, (round(float(_value(size)), 2) if size is not None else None), _colour(colour)


def _guard(func, *args):
    """A section's value, or a short error string: one odd layer never loses the card."""
    try:
        return func(*args)
    except Exception as e:                      # psd-tools raises on blocks it can't read
        return f"error: {type(e).__name__}: {e}"[:200]


# ========================
# Sections
# ========================
def _canvas(psd: PSDImage) -> dict:
    icc = psd.image_resources.get_data(Resource.ICC_PROFILE)
    description = None
    if icc:
        try:
            from PIL import ImageCms           # an optional Pillow feature (littlecms)
            description = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(BytesIO(icc))).strip()
        except Exception:
            description = None
    return {"w": psd.width, "h": psd.height, "mode": psd.color_mode.name, "depth": psd.depth, "icc": description}


def _fonts(texts: list, index: Optional[FontIndex]) -> dict:
    """{font: {"layers": [paths], "supplied": bool | None}} over the visible type layers, as font_check.psd_fonts reads them."""
    out: dict = {}
    for layer in texts:
        if layer.is_visible():
            for name in layer_fonts(layer):
                out.setdefault(name, {"layers": [], "supplied": None})["layers"].append(layer_path(layer))
    if index is not None:
        for name, entry in out.items():
            entry["supplied"] = index.supplies(name)
    return out


def _text(texts: list) -> list:
    """Every type layer, visible or not, in tree order; a style psd-tools can't read leaves font, size and colour null."""
    out = []
    for layer in texts:
        try:
            text = _clean(layer.text)
        except Exception:                       # text data psd-tools can't read
            text = None
        try:
            font, size, colour = _style(layer)
        except Exception:                       # engine data psd-tools can't read
            font, size, colour = None, None, None
        out.append({"layer": layer_path(layer), "text": text, "font": font, "size": size,
                    "color": colour, "visible": layer.is_visible()})
    return out


def _effects(layers: list) -> dict:
    out: dict = {}
    for layer in layers:
        try:
            on = layer.effects.enabled              # the layer's Effects switch off: every effect off (Springtrap's)
            names = [type(e).__name__ + ("" if on and e.enabled else " (off)") for e in layer.effects]
        except Exception:                       # an effects block psd-tools can't read
            names = ["unreadable"]
        if names:
            out[layer_path(layer)] = names
    return out


def _smart_objects(layers: list) -> list:
    return [layer_path(layer) for layer in layers if layer.kind == "smartobject"]


def _icons(psd: PSDImage, texts: list, vocab) -> str:
    """Type icons as image layers, stoogedings tokens in the text, both, or none."""
    _, icons, inline = badge_layers(psd, vocab)
    images = bool(icons or inline)
    tokens = any(font.lower().startswith(ICON_FONT_PREFIX) for layer in texts if layer.is_visible() for font in layer_fonts(layer))
    return "mixed" if images and tokens else "images" if images else "tokens" if tokens else "none"


def _drawn_bar(psd: PSDImage, key: str) -> bool:
    """A visible drawn layer named like the stat, or inside a group so named, under a *bar*/*stat* group."""
    pattern = rf"\b({STATS[key]})\b"
    for layer in psd.descendants():
        if layer.is_group() or layer.kind == "type" or not layer.is_visible() or not layer.has_pixels():
            continue
        chain = [layer, *_ancestors(layer)]
        if (any(search(pattern, owner.name, IGNORECASE) for owner in chain)
                and any(search(r"bar|stat", owner.name, IGNORECASE) for owner in chain[1:])):
            return True
    return False


def _stats(psd: PSDImage) -> dict:
    """Per stat: "template" (one digit row of exposure layers), "two-row" (several), "pixel" (drawn cells or bars), "other", "none"."""
    rows = stat_rows(psd)
    out = {}
    for key in STATS:
        found = rows.get(key) or []
        if not found:
            out[key] = "pixel" if _drawn_bar(psd, key) else "none"
            continue
        kinds = {child.kind for group, _, _ in found for child in group if not child.is_group() and child.name.strip().isdigit()}
        drawn = {kind for kind in kinds if kind in DRAWN}
        if not drawn and "type" not in kinds:
            out[key] = "two-row" if len(found) > 1 else "template"
        elif drawn == kinds:
            out[key] = "pixel"
        else:
            out[key] = "other"
    return out


def _fx(layers: list) -> list:
    return [layer_path(layer) for layer in layers if EFFECT_LAYER_TAG in layer.name and layer.kind != "group"]


def _overlays(psd: PSDImage, texts: list) -> list:
    shown = [layer for layer in texts if layer.is_visible()]
    return [layer_path(layer) for layer in overlay_words(psd, None, shown)]


def _template(psd: PSDImage, texts: list, name: str, stats, icons, overlays, rulebook: bool) -> dict:
    """
    The title layer the parser reads (Text/Name, else a root Name, one named after the card or after its own text: no
    rename needed, L6), the Ability layers and the ledger's cheap rules (L1, L2, L5, L6, L7); rule pages get no
    deviations.
    """
    text_group = next((layer for layer in psd if layer.is_group() and layer.name.strip() == "Text"), None)
    titles = PSDParser.title_layers(psd, texts, name, own_text=not rulebook)
    abilities = [layer for layer in texts if layer.is_visible() and layer.name.strip().lower() == "ability"]
    deviations = []
    if not rulebook:
        if not titles:
            deviations.append("no title layer")
        if not abilities:
            deviations.append("no Ability layer")
        elif len(abilities) > 1:
            deviations.append("several visible Ability layers")
        if any(layer.parent is not text_group for layer in abilities):
            deviations.append("ability layer outside Text")
        if isinstance(stats, dict) and any(value in ("pixel", "other") for value in stats.values()):
            deviations.append("stat bars drawn")
        if icons in ("images", "mixed"):
            deviations.append("icons as images")
        if isinstance(overlays, list) and overlays:
            deviations.append("overlay word layers")
    return {"name_layer": layer_path(titles[0]) if titles else None,
            "ability_layers": [layer_path(layer) for layer in abilities], "deviations": deviations}


def _ints(box) -> list:
    return [int(round(v)) for v in box]


def _icon_parser(vocab) -> PSDParser:
    key = tuple(vocab)
    if key not in _parsers:
        _parsers[key] = offline_parser(list(key), [], None)
    return _parsers[key]


def _star_container(group, types):
    """The group plan_stars lays the stars out in: the one shown "N stars" folder, or the one star group drawing them all."""
    while True:
        folders = [child for child in group if child.is_group() and fullmatch(STAR_FOLDER, child.name, IGNORECASE)]
        if folders:
            shown = [folder for folder in folders if folder.visible]
            if len(shown) != 1:
                return group
            group = shown[0]
            continue
        shown, hidden = _star_units(group, types)
        if len(shown) == 1 and shown[0].count > 1 and not hidden:
            group = shown[0].layers[0]
            continue
        return group


def _plan_centres(plan, container, types, size) -> list:
    """The star centres once a plan_stars plan is applied, in reading order."""
    if plan.how.startswith("folder"):
        units = _star_units(plan.show[0] if plan.show else container, types)[0]
    else:
        shown, hidden = _star_units(container, types)
        hide, show = {id(layer) for layer in plan.hide}, {id(layer) for layer in plan.show}
        moved = {id(layer): (dx, dy) for layer, dx, dy in plan.moves}
        units = []
        for unit in shown + hidden:
            ids = [id(layer) for layer in unit.layers]
            if any(i in hide for i in ids) or (unit in hidden and not any(i in show for i in ids)):
                continue
            dx, dy = next((moved[i] for i in ids if i in moved), (0, 0))
            units.append(_Unit([], _moved_box(unit.core, dx, dy)))
        units += [_Unit([], _moved_box(unit.core, dx, dy)) for unit, dx, dy, _ in plan.copies]
    return [_ints(unit.centre) for unit in _reading(units, size[1], True)]


def _star_header(psd: PSDImage, rel: str, vocab) -> Optional[dict]:
    """
    The shown stars ({core, box}, reading order), the star layer the bake copies, each count's centres (the bake's plan,
    else the layout's: first n, then _star_layouts) and the counts the bake refuses ("*": all, for one reason).
    None without exactly one Stars group drawing them.
    """
    types = _icon_parser(vocab).all_types
    groups = [group for group in PSDParser._star_groups(psd) if _star_count(group, {}, {}, {})]
    if len(groups) != 1:
        return None
    total, container = _stars_total(psd), _star_container(groups[0], types)
    shown = _star_units(container, types)[0]
    if not shown:
        return None
    size = _median_size(shown)
    order = _reading(shown, size[1], True)
    plain = len(order) == total and all(unit.count == 1 for unit in order)     # one unit per star: the layout's spots hold
    source = max(order, key=lambda unit: max(container.index(layer) for layer in unit.layers))
    parsed = {"card": {"stars": total, "path": rel, "type": ""}, "info": None}
    at, refused = {}, {}
    for n in range(1, HEADER_STARS + 1):
        try:
            at[str(n)] = _plan_centres(plan_stars(psd, parsed, n, _icon_parser(vocab)), container, types, size)
            continue
        except Refused as e:
            refused[str(n)] = sub(r"^stars: |\s*\([^)]*\bledger [A-Z]\d+\)", "", e.why)   # no ledger codes
        if plain and n <= len(order):
            at[str(n)] = [_ints(unit.centre) for unit in order[:n]]
        elif plain:
            layouts = _star_layouts(order, size, n)
            if layouts:
                at[str(n)] = [_ints(spot) for spot in layouts[0][1]]
    if len(refused) == HEADER_STARS and len(set(refused.values())) == 1:
        refused = {"*": refused["1"]}
    return {"units": [{"core": _ints(unit.core), "box": _ints(unit.box)} for unit in order],
            "icon": sub(r"\s+copy(\s+\d+)?$", "", source.layers[len(source.layers) - 1].name).strip().lower() or None,
            "at": at, "refused": refused}


def _type_header(psd: PSDImage, rel: str, name: str, vocab) -> Optional[dict]:
    """
    The header type icons as plan_types reads them (a title's tokens are the name's): the row ({name, core, box}, slot
    order), each longer row's spots past it ({count: boxes}), icons the bake can't take off ([name, why]) and a token
    row's refusal. None without icons.
    """
    images, tokens = type_row(psd, _icon_parser(vocab), name, RULEBOOK_DIR not in rel)
    if not images and not tokens:
        return None
    units, names, fixed = [], {}, []
    for name, layer in images:
        why = _icon_reason(psd, layer)
        if why:
            fixed.append([name, f"{layer.name!r} is {why}"])
        else:
            unit = _Unit([layer], _core(layer) or tuple(layer.bbox))
            units.append(unit)
            names[id(unit)] = name
    slots = _reading(units, _median_size(units)[1], False) if units else []
    clear = _texts(psd) + badge_layers(psd)[0]
    spots = {str(n): [_ints(box) for box in _icon_spots(psd, slots, n - len(slots), clear)]
             for n in range(len(slots) + 1, HEADER_ICONS + 1)} if slots else {}
    return {"row": [{"name": names[id(unit)], "core": _ints(unit.core), "box": _ints(unit.box)} for unit in slots],
            "spots": spots, "fixed": fixed, "tokens": bool(tokens),
            "refused": f"printed as icon tokens in {tokens[0].name!r}: edit them in Photoshop" if tokens else None}


def _header(psd: PSDImage, rel: str, name: str, vocab) -> dict:
    """Where the header's stars and type icons are and where the bake would put them (canvas px): the mock-up's."""
    return {"stars": _guard(_star_header, psd, rel, vocab), "types": _guard(_type_header, psd, rel, name, vocab)}


def _tree(group, depth: int = 0) -> list:
    """[name, kind, visible, children?] per layer, bottom to top as psd-tools iterates; cut at TREE_CHILDREN and TREE_DEPTH."""
    if depth >= TREE_DEPTH:
        return [list(MORE)]
    children = list(group)
    out = []
    for layer in children[:TREE_CHILDREN]:
        entry = [layer.name, layer.kind, bool(layer.visible)]
        if layer.is_group():
            entry.append(_tree(layer, depth + 1))
        out.append(entry)
    if len(children) > TREE_CHILDREN:
        out.append(list(MORE))
    return out


# ========================
# Public API Functions
# ========================
def gather(psd, rel: str, name: str, index: Optional[FontIndex] = None, vocab=()) -> dict:
    """
    One card's record.

    Args:
        psd: A PSDImage, PSD bytes or a path
        rel: The card's repo path
        name: The card's name
        index: The fonts folder's faces (None: "supplied" stays null)
        vocab: Type icon names (Types/ stems) for the image-icon rule; the *types* group rule applies without them

    Returns:
        The record (module docstring); a section psd-tools can't read is an "error: ..." string
    """
    if isinstance(psd, (bytes, bytearray)):
        psd = PSDImage.open(BytesIO(psd))
    elif isinstance(psd, (str, PathLike)):
        psd = PSDImage.open(psd)
    record = {"path": rel, "name": name, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "parser": PARSER_VERSION}
    try:
        layers = list(psd.descendants())
    except Exception as e:                      # a layer tree psd-tools can't walk: every section says so
        layers = None
        walk_error = f"error: {type(e).__name__}: {e}"[:200]
    texts = [layer for layer in layers if layer.kind == "type"] if layers is not None else []
    record["canvas"] = _guard(_canvas, psd)
    if layers is None:
        for key in ("fonts", "text", "effects", "smart_objects", "icons", "stats", "fx", "overlays", "template", "header", "tree"):
            record[key] = walk_error
        return record
    record["fonts"] = _guard(_fonts, texts, index)
    record["text"] = _guard(_text, texts)
    record["effects"] = _guard(_effects, layers)
    record["smart_objects"] = _guard(_smart_objects, layers)
    record["icons"] = _guard(_icons, psd, texts, vocab)
    record["stats"] = _guard(_stats, psd)
    record["fx"] = _guard(_fx, layers)
    record["overlays"] = _guard(_overlays, psd, texts)
    record["template"] = _guard(_template, psd, texts, name, record["stats"], record["icons"], record["overlays"],
                                RULEBOOK_DIR in rel)
    record["header"] = _guard(_header, psd, rel, name, vocab)
    record["tree"] = _guard(_tree, psd)
    return record


def mark_supplied(record: dict, index: FontIndex) -> bool:
    """Set each font's "supplied" from the index (records gathered without one, as the CLI's pool workers do). True when one changed."""
    fonts = record.get("fonts")
    changed = False
    if isinstance(fonts, dict):
        for name, entry in fonts.items():
            supplied = index.supplies(name)
            if isinstance(entry, dict) and entry.get("supplied") is not supplied:
                entry["supplied"] = supplied
                changed = True
    return changed


def refill(cards: dict, index: FontIndex) -> int:
    """mark_supplied over every record. Returns how many records changed."""
    return sum(mark_supplied(record, index) for record in cards.values() if isinstance(record, dict))


def font_index(tree: Optional[dict] = None) -> Optional[FontIndex]:
    """
    The faces the "supplied" flags are judged by: font_check.local_fonts (PSD_TOOLS_FONT_DIRS, else the checkout's
    MDW/Fonts), counted only when its font files are the tree's, by blob sha.

    Args:
        tree: The recursive tree the caller fetched; None or truncated: the folder as it is

    Returns:
        FontIndex, or None (one console line) when the fonts can't be had: "supplied" stays null then, never a wrong False
    """
    files = None
    if tree is not None and not tree.get("truncated"):
        files = tree_fonts({e.get("path", ""): e.get("sha", "") for e in tree.get("tree", []) if e.get("type") == "blob"})
    directory = local_fonts()[0]
    try:
        if not directory or not isdir(directory):
            print("Fonts for the details: no PSD_TOOLS_FONT_DIRS or MDW/Fonts checkout; supplied stays null")
            return None
        if files and dir_fonts(directory) != files:
            print(f"Fonts for the details: {directory} is not the branch's MDW/Fonts; supplied stays null")
            return None
    except OSError as e:                        # never fails a build
        print(f"Fonts for the details: {e}")
        return None
    key = (directory, tuple(sorted(files.items()))) if files else None
    if key in _indexes:
        return _indexes[key]
    index = FontIndex(directory)
    index = index if index.files else None
    if key is not None and index is not None:
        _indexes.clear()
        _indexes[key] = index
    return index


def load(path: str) -> dict:
    """
    The file's records.

    Args:
        path: The details file

    Returns:
        {blob sha: record}; {} when the file is missing or not a details file (one console line then)
    """
    try:
        with open(path, encoding="utf-8") as fh:
            data = json_load(fh)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        print(f"{path}: unreadable ({e}); starting empty")
        return {}
    cards = data.get("cards") if isinstance(data, dict) else None
    if not isinstance(cards, dict):
        print(f"{path}: not a details file; starting empty")
        return {}
    return cards


def text_of(cards: dict) -> str:
    """The file's text: records sorted by sha, one per line (diffs stay per card), nothing else indented."""
    lines = ",\n".join(f"{dumps(sha)}: {dumps(cards[sha], ensure_ascii=False, separators=(',', ':'))}" for sha in sorted(cards))
    return '{"version": %d, "cards": {\n%s\n}}\n' % (VERSION, lines)


def prune(cards: dict, shas) -> int:
    """Drop the records whose blob is not in shas (the tree's). Returns how many went."""
    gone = [sha for sha in cards if sha not in shas]
    for sha in gone:
        del cards[sha]
    return len(gone)


# ========================
# Module Exports
# ========================
__all__ = [
    'DRAWN',
    'HEADER_ICONS',
    'HEADER_STARS',
    'MORE',
    'TEXT_CHARS',
    'TREE_CHILDREN',
    'TREE_DEPTH',
    'VERSION',
    'font_index',
    'gather',
    'layer_path',
    'load',
    'mark_supplied',
    'prune',
    'refill',
    'text_of',
]
