"""
The PSD facts behind the website's card "Details" dropdown: fonts, text layers, effects, smart objects, canvas, type-icon
style, stat-row encoding, status-word layers, template deviations and a compact layer tree. One record per card PSD,
keyed by its git blob sha, in psd_details.json beside the pickles (committed; /push carries it with them).

    python -m commands.psd_details --clone <dir> [--out psd_details.json] [--fonts <dir>]
    python -m commands.psd_details --refill [--out psd_details.json] [--fonts <dir>]

gather(psd, rel, name, index, vocab) -> the record: pure, never raises for a card (a failed section is a short
"error: ..." string). load/save (tmp + fsync + replace under one lock), prune(cards, shas) to the tree. The traverser
gathers every re-parsed card's record in /update_stats, its "supplied" flags from font_index() (font_check.find_fonts:
the checkout's MDW/Fonts, else the fonts cache brought to the update's tree), then refills every record's flags from
that index (records gathered without one, or before MDW/Fonts changed). The CLI backfills a clone in a process pool (the
open is the whole cost, like font_check --all), or --refill sets the flags by hand. Run --refill from the bot's
directory (psd_details.json and fonts_cache/ are cwd-relative) while the bot runs no /update_stats (the daily 5AM
reminder too), /bake_cards or /fonts: the details file's temp and the fonts cache are shared, and their locks are per
process. No Discord or config imports; stock psd-tools. Pillow's ImageCms names the ICC profile when it is built in.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from io import BytesIO
from json import dumps, load as json_load
from os import PathLike, cpu_count, fsync, listdir, remove, replace, walk
from os.path import isdir, join, relpath, splitext
from re import IGNORECASE, search
from statistics import median
from threading import Lock
from time import perf_counter
from typing import Optional

from psd_tools import PSDImage
from psd_tools.constants import Resource

from commands.card_models import card_name
from commands.font_check import (
    EXCLUDE_FOLDERS, FONTS_DIR, POOL_CHUNK, POOL_MIN, FontIndex, dir_fonts, find_fonts, is_card_psd, layer_fonts, tree_fonts,
)
from commands.github_api import blob_sha, token_from_env
from commands.psd_parser import EFFECT_LAYER_TAG, ICON_FONT_PREFIX, PARSER_VERSION, PSDParser
from commands.psd_regions import STATS, badge_layers, overlay_words, stat_rows
from global_config import CARD_REPO, DETAILS_JSON, FONTS_CACHE, RULEBOOK_DIR, TYPES_DIR

# ========================
# Configuration
# ========================
VERSION = 1                                 # the file's "version"
TEXT_CHARS = 200                            # text kept per type layer
TREE_CHILDREN = 80                          # children kept per group
TREE_DEPTH = 8                              # groups nested deeper show one "more" entry
MORE = ["…", "more", True]             # the tree's cut marker
DRAWN = ("pixel", "smartobject", "shape")   # digit layers of these kinds are drawn bar cells
_lock = Lock()                              # one writer of the file
_indexes: dict = {}                         # (cache dir, its files) -> FontIndex: built once per MDW/Fonts change


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
            entry["supplied"] = index.resolve(name) is not None
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
            names = [type(e).__name__ + ("" if e.enabled else " (off)") for e in layer.effects]
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
        for key in ("fonts", "text", "effects", "smart_objects", "icons", "stats", "fx", "overlays", "template", "tree"):
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
    record["tree"] = _guard(_tree, psd)
    return record


def mark_supplied(record: dict, index: FontIndex) -> bool:
    """Set each font's "supplied" from the index (records gathered without one, as the CLI's pool workers do). True when one changed."""
    fonts = record.get("fonts")
    changed = False
    if isinstance(fonts, dict):
        for name, entry in fonts.items():
            supplied = index.resolve(name) is not None
            if isinstance(entry, dict) and entry.get("supplied") is not supplied:
                entry["supplied"] = supplied
                changed = True
    return changed


def refill(cards: dict, index: FontIndex) -> int:
    """mark_supplied over every record. Returns how many records changed."""
    return sum(mark_supplied(record, index) for record in cards.values() if isinstance(record, dict))


def font_index(repository: str = CARD_REPO, cache: str = FONTS_CACHE, tree: Optional[dict] = None) -> Optional[FontIndex]:
    """
    The faces the "supplied" flags are judged by: font_check.find_fonts (PSD_TOOLS_FONT_DIRS, the checkout's MDW/Fonts,
    else the fonts cache brought to the tree: no listing, only the changed font blobs; no tree: 1 conditional listing).
    A local dir counts only when its font files are the tree's, by blob sha.

    Args:
        repository: owner/name of the card repo
        cache: The fonts cache
        tree: The recursive tree the caller fetched (/update_stats); None or truncated lists MDW/Fonts

    Returns:
        FontIndex, or None (one console line) when the fonts can't be had or a file failed to download: "supplied" stays
        null then, never a wrong False. A failed listing uses the cache as the last refresh left it, if that one finished
    """
    files, sub = None, ""
    if tree is not None and not tree.get("truncated"):
        items = tree.get("tree", [])
        files = tree_fonts({e.get("path", ""): e.get("sha", "") for e in items if e.get("type") == "blob"})
        sub = next((e.get("sha", "") for e in items if e.get("path") == FONTS_DIR and e.get("type") == "tree"), "")
    try:
        fonts = find_fonts(repository, files, cache, sub)
        if fonts.source != "cache" and files and dir_fonts(fonts.directory) != files:
            print(f"Fonts for the details: {fonts.directory} is not the branch's MDW/Fonts; supplied stays null")
            return None
    except Exception as e:                      # never fails an update
        print(f"Fonts for the details: {e}")
        return None
    if fonts.failed:
        print(f"Fonts for the details: {len(fonts.failed)} MDW/Fonts files failed to download; supplied stays null")
        return None
    if fonts.error:
        print(f"Fonts for the details: {fonts.error}")
    if not fonts.complete:                      # a failed listing over a cache the last refresh left short
        print("Fonts for the details: the fonts cache is incomplete or not this tree's; supplied stays null")
        return None
    held = fonts.files if fonts.source == "cache" else files
    key = (fonts.directory, tuple(sorted(held.items()))) if held else None
    if key in _indexes:
        return _indexes[key]
    index = FontIndex(fonts.directory) if isdir(fonts.directory) else None
    index = index if index is not None and index.files else None
    if key is not None and index is not None:
        _indexes.clear()
        _indexes[key] = index
    return index


def load(path: str = DETAILS_JSON) -> dict:
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


def save(cards: dict, path: str = DETAILS_JSON) -> None:
    """Write the file to <path>.tmp, fsync, then replace (card_models.save_pickle's way), under the module lock."""
    text = text_of(cards)
    with _lock:
        temp_path = path + ".tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                fsync(fh.fileno())
            replace(temp_path, path)
        except BaseException:
            try:
                remove(temp_path)
            except OSError:
                pass
            raise


def prune(cards: dict, shas) -> int:
    """Drop the records whose blob is not in shas (the tree's). Returns how many went."""
    gone = [sha for sha in cards if sha not in shas]
    for sha in gone:
        del cards[sha]
    return len(gone)


# ========================
# A clone (the backfill)
# ========================
def card_psds(root: str) -> list:
    """(repo path, file path) of every card PSD under a clone, in path order; MDW/ and Markers/ left out like the traverser."""
    found = []
    for folder, subdirs, files in walk(root):
        subdirs[:] = sorted(d for d in subdirs if d not in EXCLUDE_FOLDERS and d != ".git")
        for name in sorted(files):
            rel = relpath(join(folder, name), root).replace("\\", "/")
            if is_card_psd(rel):
                found.append((rel, join(folder, name)))
    return found


def clone_types(root: str) -> list:
    """The Types/ icon names of a clone (the traverser's all_types: file stems, lowercased)."""
    folder = join(root, TYPES_DIR)
    if not isdir(folder):
        return []
    return sorted(splitext(f)[0].lower() for f in listdir(folder) if not f.startswith(".") and "." in f)


def _gather_one(item: tuple) -> tuple:
    """(rel, file path, vocab) -> (blob sha, record, None) or ("", None, the error) in a pool worker."""
    rel, path, vocab = item
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        return blob_sha(data), gather(data, rel, card_name(rel), None, vocab), None
    except Exception as e:                      # one unreadable PSD never stops the backfill
        return "", None, f"{rel}: {e}"


def gather_clone(root: str, index: Optional[FontIndex] = None) -> tuple:
    """
    Every card PSD of a clone, keyed by blob sha.

    Args:
        root: The clone
        index: The fonts folder's faces, applied after the gathers (None: "supplied" stays null)

    Returns:
        ({blob sha: record}, [error lines]); from POOL_MIN PSDs the gathers run in a process pool of half the cores
    """
    found = card_psds(root)
    vocab = tuple(clone_types(root))
    items = [(rel, path, vocab) for rel, path in found]
    cards, errors = {}, []

    def collect(results) -> None:
        for sha, record, error in results:
            if error is not None:
                errors.append(error)
            else:
                cards[sha] = record
    if len(items) < POOL_MIN:
        collect(map(_gather_one, items))
    else:
        workers = max(1, min((cpu_count() or 2) // 2, -(-len(items) // POOL_CHUNK)))
        with ProcessPoolExecutor(workers) as pool:
            collect(pool.map(_gather_one, items, chunksize=POOL_CHUNK))
    if index is not None:
        for record in cards.values():
            mark_supplied(record, index)
    return cards, errors


def record_sizes(cards: dict) -> list:
    """Each record's size in bytes as the file holds it, sorted."""
    return sorted(len(dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) for record in cards.values())


# ========================
# Command line
# ========================
def main(args: Optional[list] = None) -> int:
    """The backfill (every card PSD of a clone into the details file) or the refill (the supplied flags); one summary line."""
    parser = ArgumentParser(prog="python -m commands.psd_details",
                            description="Gather every card PSD's details (fonts, text layers, effects, stats encoding, layer tree...) "
                                        "from a local clone into psd_details.json, keyed by git blob sha; or, with --refill, set "
                                        "every record's supplied flags from the fonts.")
    parser.add_argument("--clone", default="", metavar="DIR", help="the card repo checkout")
    parser.add_argument("--refill", action="store_true", help="set the supplied flags of the records already in --out (no PSD is read)")
    parser.add_argument("--out", default=DETAILS_JSON, metavar="FILE", help=f"the details file (default: {DETAILS_JSON})")
    parser.add_argument("--fonts", default="", metavar="DIR", help="the fonts folder for the supplied flags (default: the clone's "
                                                                   "MDW/Fonts when present; --refill: the checkout's, else the fonts cache)")
    options = parser.parse_args(args)
    if options.refill == bool(options.clone):
        parser.error("give --clone or --refill")
    if options.refill:
        return _refill_file(options.out, options.fonts)
    fonts_dir = options.fonts or (join(options.clone, FONTS_DIR) if isdir(join(options.clone, FONTS_DIR)) else "")
    index = FontIndex(fonts_dir) if fonts_dir else None
    started = perf_counter()
    cards, errors = gather_clone(options.clone, index)
    save(cards, options.out)
    sizes = record_sizes(cards)
    stats = (f"records {sizes[0]}-{sizes[len(sizes) - 1]} B, median {int(median(sizes))} B, {sum(sizes) / 1e6:.1f} MB in all"
             if sizes else "no records")
    print(f"{len(cards)} cards into {options.out} in {perf_counter() - started:.1f} s ({len(errors)} errors); {stats}")
    for line in errors:
        print("  " + line)
    return 0


def _refill_file(path: str, fonts_dir: str) -> int:
    """--refill: load, mark, save only when a flag changed (atomic, so the bot never reads half a file)."""
    if fonts_dir:
        index = FontIndex(fonts_dir)
        index = index if index.files else None
    else:
        token_from_env()
        index = font_index()
    if index is None:
        print(f"No fonts {'in ' + fonts_dir if fonts_dir else 'to judge by'}: {path} left as it is")
        return 1
    cards = load(path)
    changed = refill(cards, index)
    if changed:
        save(cards, path)
    print(f"{changed} of {len(cards)} records' supplied flags changed in {path} ({index.files} font files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'DRAWN',
    'MORE',
    'TEXT_CHARS',
    'TREE_CHILDREN',
    'TREE_DEPTH',
    'VERSION',
    'card_psds',
    'clone_types',
    'font_index',
    'gather',
    'gather_clone',
    'layer_path',
    'load',
    'main',
    'mark_supplied',
    'prune',
    'record_sizes',
    'refill',
    'save',
    'text_of',
]
