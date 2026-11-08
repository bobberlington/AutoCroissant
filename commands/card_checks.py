"""
Automatic card flags (owner 2026-11-07): what a card needs from a developer, read off its PSD and its picture, kept by PSD
blob in the card data's db/card_checks.json (TTSCardMaker's card-checks.yml, off Card sync's path; the website shows them).
hand: the picture says other than the data, old-style bars, a website edit that cannot bake, a linked picture the repo
lacks, one that looks the same as another file, or one Card sync could not draw again (db/links.json, worded at each
save). bars: special bars that do not read like the template. A note is one short sentence a developer can act on.
Nothing here blocks or tells anyone.

    python -m commands.card_checks list --clone <checkout> --db <db dir> [--head <sha>] [--all] [--shard i/n]
    python -m commands.card_checks run  --clone <checkout> --db <db dir> --out <file> [--head <sha>] [--all] [--shard i/n]
                                        [--cards a,b] [--workers n] [--minutes m]
    python -m commands.card_checks save --clone <checkout> --db <db dir> [--head <sha>] [results.json ...]

list prints the blobs to check (PSDs without a result, their PNGs; a PNG pushed alone counts); run checks them; save puts
the results in db/, drops PSDs gone from the tree, words every record's notes against the data (a website-only fix clears
without a push) and marks the people's flags that went. Exit 0, also for what it flags. No Discord or config imports.

Which edits are tried (a real bake is 25 s; the whole list of 13 kinds of edit, 4 minutes a card, would be 50 hours a
tree): the plans alone for titles, stars and types, a made icon on a copy; a real bake only where a plan cannot tell:
stars on a card with variant pictures, the ability on a card with FX layers, the stats on digit rows or new bars under
art (their pixels too), a type on a very small PSD. A bake that breaks leaves no record: the next run tries again.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from io import BytesIO
from json import dumps, load as json_load, loads
from os import environ
from os.path import basename, exists, join
from re import IGNORECASE, finditer, fullmatch, search, sub
from shutil import rmtree
from sys import argv
from tempfile import mkdtemp
from time import perf_counter
from typing import Callable, Optional

import numpy as np
from PIL import Image
from psd_tools import PSDImage

import commands.card_bake as card_bake
import commands.png_export as png_export
import commands.psd_edit as psd_edit
from commands.card_bake import Card, Snapshot, local_snapshot, psd_fields, vocabulary
from commands.card_db import CHECKS_JSON, STATS_JSON
from commands.card_edits import CardAuthors, CardEdits
from commands.card_models import RAID_STARS, card_name
from commands.card_preview import outcome, resolve
from commands.card_standards import _bar_art
from commands.github_api import token_from_env
from commands.psd_parser import EFFECT_LAYER_TAG, variants_of
from commands.psd_regions import STATS, bar_box, comp_bars, linked_items, placed_data, stat_rows
from global_config import CARD_REPO, EDITS_PATH, ICON_MANIFEST_PATH, TYPES_DIR

# ========================
# Configuration
# ========================
SPARSE_FILES = ("commands/card_checks.py", "commands/card_bake.py", "commands/card_db.py", "commands/card_edits.py",
                "commands/card_models.py", "commands/card_preview.py", "commands/card_preview_branch.py",
                "commands/card_standards.py", "commands/font_check.py", "commands/git_commit.py", "commands/github_api.py",
                "commands/png_export.py", "commands/psd_edit.py", "commands/psd_parser.py", "commands/psd_parts.py",
                "commands/psd_regions.py", "commands/repo_history.py", "commands/rulebook.py", "commands/text_align.py",
                "commands/txt2_repair.py", "commands/ability_place.py", "commands/ability_index.json", "global_config.py",
                "requirements-cards.txt",
                "commands/psd_links.py")         # card-checks.yml's bot checkout
VERSION = 1                                 # the file's "version"
STAT_ORDER = tuple(STATS)                   # hp, def, atk, spd: the template's rows, top to bottom
STAT_NAMES = {"hp": "HP", "def": "DEF", "atk": "ATK", "spd": "SPD"}
DATA_STATS = {"hp": "hp", "def": "defense", "atk": "attack", "spd": "speed"}    # stats.json's names
# the picture
TEMPLATE_CELLS = (0, 75, 158, 245, 326, 404, 487, 568, 649, 726, 813)   # the template bar's cell edges, px
TOP_BAND, BOTTOM_BAND = (0.12, 0.42), (0.58, 0.88)      # a row's two halves; past 10 the bottom one counts on
EDGE_GAP, EDGE_WIDTH = 5, 24                # px beside a cell edge that are compared, clear of its line
FIRST_EDGE = 1.5                            # the template's shading shows this much more at its first edge
STEP_MIN = 12.0                             # colour distance (Lab) a filled end shows at the least (true steps: 15 and up)
STEP_CLEAR = 1.8                            # a step reads when it beats every other edge by this much
STEP_NEAR = 0.6                             # an edge this close to the strongest could be the value too
HALF_ROW = 0.4                              # a step in the bottom half alone: the top half shows under this share of it
LIT_GAP = 10.0                              # lit and empty cells this far apart (L) tell a flat row's state
ART_FILL = 0.9                              # bar art filling less of its box is separate icons (the template's fills 0.977)
PIXEL_MOVED = 24                            # a pixel counts as changed past this many levels (the census's)
# the edits
TYPE_TRIES = ("fire", "water", "ice")       # a type to add: the first a card lacks
MAX_STARS = 7                               # the census's: a star added up to here
SMALL_PSD = 3_000_000                       # bytes: under this a type edit bakes for real (3 cards)
WHOLE_FIELD = r"edit each box on the website|^\d+ title layers|title is split over several layers|title is printed again by"
WHAT = {"name": "A new title", "adding a star": "Adding a star", "removing a star": "Taking off a star",
        "adding a type": "Adding a type", "removing a type": "Taking off a type", "stats": "Stat edits"}
# the notes
TITLE_NOTE = "The title is part of the picture: a new title has to be drawn in Photoshop."
OLD_NOTE = "Its stat bars are old-style digit rows: put them on the new bars in Photoshop."
CREATURES_NOTE = "Several creatures: the website changes the first one's bars, the others are changed in Photoshop."
NO_CHANGE_NOTE = "{} {} bake but change nothing on the card: change them in Photoshop."
SWAP_NOTE = "The {} bars are on each other's rows: a stat edit moves the wrong bar, put them back in Photoshop."
BRIGHT_NOTE = "Its shown digits brighten the cells, so a stat edit turns the bars dark: change its stats in Photoshop."
LINKS_JSON = "links.json"                   # the card data's db/: every PSD's links (card_data.gather_links)
DUP_NOTE = "{name} is drawn apart from this PSD: when the card changes, export it in Photoshop."
LINK_MISSING = "Links {name}, which is not in the repo: push that file."
LINK_ALIKE = "Links {name}, which looks the same as {kept}: link {kept} instead."
LINK_STUCK = "{layer} links {name}, changed since: Card sync could not draw it again ({why}); update it in Photoshop."
CUSTOM_NOTES = {"icons": "Its stat bars are separate icons with gaps, not the template's cells.",
                "pictures": "Its stat bars are pictures, not the template's cells.",
                "shape": "Its stat bars have their own shape, not the template's cells.",
                "creatures": "Its stat bars hold several creatures, not the template's four rows."}


# ========================
# Reading a bar off the picture
# ========================
def _lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB 0-255 to Lab."""
    c = np.asarray(rgb, np.float64) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    f = xyz / np.array([0.9505, 1.0, 1.089])
    f = np.where(f > 0.008856, np.cbrt(f), 7.787 * f + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _steps(row: np.ndarray, band: tuple) -> np.ndarray:
    """The colour change at each of a row's 9 inner cell edges, in one band of it."""
    height, width = row.shape[:2]
    part = row[int(height * band[0]):max(int(height * band[1]), int(height * band[0]) + 1)]
    scale = width / TEMPLATE_CELLS[-1]
    gap, reach = max(1, int(EDGE_GAP * scale)), max(2, int(EDGE_WIDTH * scale))
    out = []
    for edge in TEMPLATE_CELLS[1:-1]:
        x = int(edge * scale)
        left, right = part[:, max(x - reach, 0):x - gap], part[:, x + gap:x + reach]
        out.append(float(np.linalg.norm(_lab(left.reshape(-1, 3).mean(0)) - _lab(right.reshape(-1, 3).mean(0))))
                   if left.size and right.size else 0.0)
    out[0] /= FIRST_EDGE
    return np.array(out)


def _clear(steps: np.ndarray) -> Optional[int]:
    """The filled cells a row's steps read: its one clear step; None for none."""
    top = float(steps.max())
    return int(steps.argmax()) + 1 if top >= STEP_MIN and top >= STEP_CLEAR * float(np.sort(steps)[-2]) else None


def read_row(row: np.ndarray) -> dict:
    """
    One bar row as the picture shows it: {"best": the value it reads or None, "could": the values it does not rule out}.
    A step in both halves is that many cells; one in the bottom half alone is past 10; a flat row ("flat": True) is 0, 10
    or 20 (read_picture tells which); a busy one has no say (could None).
    """
    top, bottom = _steps(row, TOP_BAND), _steps(row, BOTTOM_BAND)
    both = (top + bottom) / 2
    strongest = float(both.max())
    near = [k + 1 for k, step in enumerate(both) if step >= STEP_NEAR * strongest]
    low = _clear(bottom)
    if low is not None and top[low - 1] < HALF_ROW * bottom[low - 1]:
        return {"best": 10 + low, "could": sorted({low, 10 + low, *near, *(10 + k for k in near)})}   # own art can hide a step
    if strongest < STEP_MIN:
        return {"best": None, "could": [0, 10, 20], "flat": True}
    best = _clear(both)
    return {"best": best, "could": near} if best is not None else {"best": None, "could": None}


def _lightness(row: np.ndarray, first: int = 0, last: int = 10) -> float:
    """The mean L of a row's cells first..last-1, its middle band."""
    height, width = row.shape[:2]
    scale = width / TEMPLATE_CELLS[-1]
    part = row[int(height * TOP_BAND[0]):int(height * BOTTOM_BAND[1]),
               int(TEMPLATE_CELLS[first] * scale):int(TEMPLATE_CELLS[last] * scale)]
    return float(_lab(part.reshape(-1, 3).mean(0))[0])


def _tell_flats(rows: list) -> None:
    """A flat row is full or empty by the card's other rows' lit and empty cells; left open when they don't tell."""
    stepped = [(row, read) for row, read in rows if read["best"] and read["best"] < 10]
    if not stepped:
        return
    lit = float(np.median([_lightness(row, 0, read["best"]) for row, read in stepped]))
    empty = float(np.median([_lightness(row, read["best"]) for row, read in stepped]))
    if lit - empty < LIT_GAP:
        return
    for row, read in rows:
        if read.get("flat"):
            full = abs(_lightness(row) - lit) < abs(_lightness(row) - empty)
            read.update(best=10 if full else 0, could=[10, 20] if full else [0])


def bar_boxes(psd: PSDImage) -> dict:
    """{stat: [each bar's box, top to bottom]}: the layer-comp bars, else the digit rows."""
    found = {stat: [tuple(box) for _, _, box in bars] for stat, bars in comp_bars(psd).items()}
    if not any(found.values()):
        found = {stat: [bar_box(psd, stat, box) for _, _, box in rows if box] for stat, rows in stat_rows(psd).items()}
    return {stat: sorted(boxes, key=lambda box: box[1]) for stat, boxes in found.items()}


def by_label(psd: PSDImage, boxes: dict) -> dict:
    """The bars by the labels printed beside them: a Defense bar on the Attack row is that row's. No four labels and four
    bars: by their layers' names."""
    if any(len(boxes[stat]) != 1 for stat in STAT_ORDER):
        return boxes
    rows = sorted((boxes[stat][0] for stat in STAT_ORDER), key=lambda box: box[1])
    labels = {}
    for layer in psd.descendants():
        word = (layer.text or "").strip() if layer.kind == "type" and layer.is_visible() else ""
        stat = next((key for key, pattern in STATS.items() if fullmatch(pattern, word, IGNORECASE)), None)
        if stat and rows[0][1] - 40 <= (layer.bbox[1] + layer.bbox[3]) / 2 <= rows[-1][3] + 40:
            labels[stat] = (layer.bbox[1] + layer.bbox[3]) / 2
    if len(labels) != len(STAT_ORDER):
        return boxes
    return {stat: [rows[n]] for n, stat in enumerate(sorted(STAT_ORDER, key=labels.get))}


def swapped(psd: PSDImage) -> list:
    """The stats whose bar lies on another's row by the printed labels: equal values show nothing, an edit moves the wrong bar."""
    boxes = bar_boxes(psd)
    rows = by_label(psd, boxes)
    return [stat for stat in STAT_ORDER if rows[stat] != boxes[stat]]


def _cut(pixels: np.ndarray, box: tuple, width: int) -> Optional[np.ndarray]:
    """A bar's part of a picture of a PSD this wide; None past the canvas."""
    scale = pixels.shape[1] / width
    x0, y0, x1, y1 = (int(v * scale) for v in box)
    return pixels[y0:y1, x0:x1] if 0 <= x0 < x1 <= pixels.shape[1] and 0 <= y0 < y1 <= pixels.shape[0] else None


def read_picture(psd: PSDImage, image: Image.Image) -> dict:
    """{stat: [{"best", "could"} of each bar, top to bottom]} off the card's picture; a bar past the canvas has no say."""
    pixels = np.asarray(image.convert("RGB"))
    out, rows = {}, []
    for stat, boxes in by_label(psd, bar_boxes(psd)).items():
        out[stat] = []
        for box in boxes:
            row = _cut(pixels, box, psd.width)
            if row is None:
                out[stat].append({"best": None, "could": None})
                continue
            out[stat].append(read_row(row))
            rows.append((row, out[stat][-1]))
    _tell_flats(rows)
    return out


# ========================
# The picture against the data
# ========================
def _pairs(rows: list, values: list) -> list:
    """(read, value) per bar of a stat: a bar per creature, one bar for one value, two for one (the rows add up), else none."""
    if len(rows) == len(values):
        return list(zip(rows, values))
    if len(rows) == 2 and len(values) == 1 and all(row["best"] is not None for row in rows):
        total = rows[0]["best"] + rows[1]["best"]
        return [({"best": total, "could": [total]}, values[0])]
    return []


def mismatches(record: dict, data: dict, site=()) -> list:
    """
    What a card's picture says against its data, a line per bar: the card's own bars speak bar by bar, custom ones only
    when every bar reads (a picture that merely looks like cells must stay silent).

    Args:
        record: check_card()'s
        data: The card in stats.json
        site: Its Card Edits fields: a website-only value differs from the picture on purpose, a pending one until it bakes
    """
    reads = record.get("reads") or {}
    if record.get("style") == "custom" and any(row["best"] is None for rows in reads.values() for row in rows):
        return []
    if (data.get("stars") or 0) >= RAID_STARS:  # its bars run to 40 in rows of their own
        return []
    stats, creatures = data.get("stats") or {}, data.get("creatures") or []
    out = []
    for stat in STAT_ORDER:
        values = [one.get(DATA_STATS[stat]) for one in creatures] if creatures else [stats.get(DATA_STATS[stat])]
        if stat in site or "creatures" in site or not reads.get(stat):
            continue
        for n, (row, value) in enumerate(_pairs(reads[stat], values)):
            if value is None or value < 0 or row["could"] is None or value in row["could"]:
                continue
            whose = f"Creature {n + 1}'s {STAT_NAMES[stat]} bar" if creatures else f"The {STAT_NAMES[stat]} bar"
            shows = f"shows {row['best']}" if row["best"] is not None else "shows no step"
            out.append(f"{whose} {shows} but the data says {value}.")
    return out


# ========================
# The bars' style
# ========================
def _art_fill(psd: PSDImage, box: tuple) -> Optional[float]:
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


def _digits(rows: dict) -> list:
    """Every digit layer of the rows."""
    return [layer for found in rows.values() for group, _, _ in found for layer in group
            if not layer.is_group() and layer.name.strip().isdigit()]


def _template_rows(psd: PSDImage, rows: dict) -> bool:
    """Digit rows as the old template made them: one or two rows of darkening exposures a stat, over bars named so."""
    digits = _digits(rows)
    return all(1 <= len(rows[stat]) <= 2 for stat in STAT_ORDER) \
        and len(digits) >= 10 * len(STAT_ORDER) and all(layer.kind == "exposure" and layer.exposure <= 0 for layer in digits) \
        and all(_bar_art(psd, stat) for stat in STAT_ORDER)


def bar_style(psd: PSDImage, creatures: Optional[list]) -> tuple:
    """
    (style, why): "" for a card without bars, "few" for an item's digit rows, "new" for layer-comp bars, "old" for the old
    template's digit rows, "custom" for anything else or new bars over separate icons; why: CUSTOM_NOTES' key.
    """
    comps = comp_bars(psd)
    if any(comps.values()):
        fills = [_art_fill(psd, tuple(box)) for bars in comps.values() for _, _, box in bars]
        return ("custom", "icons") if any(fill is not None and fill < ART_FILL for fill in fills) else ("new", "")
    rows = stat_rows(psd)
    if not any(rows.values()):
        return "", ""
    if not all(rows.values()):                  # an item's HP and DEF: read, nothing to convert
        return "few", ""
    if creatures:
        return "custom", "creatures"
    if _template_rows(psd, rows):
        return "old", ""
    return "custom", "shape" if all(layer.kind == "exposure" for layer in _digits(rows)) else "pictures"


# ========================
# Website edits
# ========================
@dataclass
class Context:
    """What every card's check shares: the tree, the parser, the fonts and the status chart."""
    snapshot: Snapshot
    parser: object
    fonts: str
    colors: list = field(default_factory=list)

    def chart(self) -> Optional[dict]:
        """status_fills() of the status chart, once."""
        if not self.colors:
            path = psd_edit.STATUS_CHART
            self.colors.append(psd_edit.status_fills(PSDImage.open(BytesIO(self.snapshot.fetch(self.snapshot.sha(path)))))
                               if path in self.snapshot.tree else None)
        return self.colors[0]


def context(repository: str, clone: str, head: str = "") -> Context:
    """A Context on a checkout (blobless is enough)."""
    snapshot = local_snapshot(repository, clone, head)
    return Context(snapshot, vocabulary(snapshot), card_bake._fonts_dir(snapshot))


def _why(reason: str) -> str:
    """A refusal as a note's words: no code, field, coordinates or Photoshop advice."""
    if reason.startswith("size: "):
        return "the PSD would grow past the bake's limit"
    text = sub(r"^[a-z0-9>-]+: ", "", reason.split(";")[0].strip(), count=1)
    text = sub(r"^(website only: )?(stars|types|hp|def|atk|spd): ", "", text)
    text = text.replace(" at (", " (")
    for _ in range(2):                          # coordinates, nested in a sentence's brackets
        text = sub(r" ?\([^()]*\d[^()]*\)", "", text)
    text = sub(r"(?<![A-Za-z])'([^']{33})[^']+'(?![A-Za-z])", r"'\1…'", text)
    return sub(r":? ?((edit|set) (it|them) in Photoshop|a (change|toggle) would).*$", "", text).strip().rstrip(".,:")


def _note(what: str, reason: str) -> str:
    """What cannot bake, as a sentence."""
    if what == "ability":
        return f"Some ability edits fail: {_why(reason)}: recut it in Photoshop."
    return f"{WHAT[what]} {'need' if what == 'stats' else 'needs'} Photoshop: {_why(reason)}."


def _real_bake(ctx: Context, rel: str, fields: dict, png: Optional[np.ndarray] = None) -> tuple:
    """A real bake on a temp copy: (state, reason, the pixels changed against png). One that broke raises: never clean,
    the next run checks the card again."""
    work = mkdtemp(prefix="card-checks_", dir=environ.get(card_bake.WORK_ENV) or None)
    try:
        card = resolve(ctx.snapshot, rel, fields)
        card_bake.bake_card(ctx.snapshot, card, ctx.parser, ctx.fonts, ctx.chart, work)
        state, _, reason, _ = outcome(card)
        if state == "error":
            raise RuntimeError(reason)
        moved = None
        if png is not None and card.baked is not None:
            baked = np.asarray(Image.open(card.baked.png_path).convert("RGB"), np.int16)
            moved = np.abs(baked - png).max(-1) > PIXEL_MOVED if baked.shape == png.shape else None
        return state, reason, moved
    finally:
        rmtree(work, ignore_errors=True)


def _ability_edit(text: str) -> Optional[str]:
    """An ability with its last long word changed (the census's edit)."""
    words = list(finditer(r"\b[a-z]{4,}\b", text or ""))
    if not words:
        return None
    word = words[len(words) - 1]
    return text[:word.start()] + ("test" if word.group() != "test" else "card") + text[word.end():]


def _art_over_darks(psd: PSDImage) -> bool:
    """Layers but Lines above the new bars' Darks group: they can hide it (Sans, The Thin Man: their bars never change)."""
    darks = next(layer.parent for bars in comp_bars(psd).values() for layer, _, _ in bars)
    siblings = list(darks.parent)
    above = siblings[next(n for n, layer in enumerate(siblings) if layer is darks) + 1:]
    return any(layer.is_visible() and layer.name.strip().lower() != "lines" for layer in above)


def _still(psd: PSDImage, moved: np.ndarray, style: str) -> list:
    """The stats a bake left as they were: every one when nothing moved, else bar by bar on new bars alone (a digit row's
    box need not be where its picture changes: Zigg); a bar past the canvas has no say."""
    if not moved.any():
        return list(STAT_ORDER)
    if style != "new":
        return []
    rows = {stat: [_cut(moved, box, psd.width) for box in boxes] for stat, boxes in by_label(psd, bar_boxes(psd)).items()}
    return [stat for stat in STAT_ORDER if rows[stat] and all(row is not None and not row.any() for row in rows[stat])]


def _stat_notes(ctx: Context, rel: str, psd: PSDImage, parsed: dict, style: str, image: Optional[Image.Image]) -> list:
    """Stat edits that cannot bake: new bars by their state (their bakes draw: measured on every card) unless art lies
    over them; those and digit rows by a real bake of all four, one down and one up, which must move pixels (new bars: each)."""
    card = parsed["card"]
    have = {stat: card.get(stat) for stat in STAT_ORDER if isinstance(card.get(stat), int) and card.get(stat) >= 0}
    if len(have) < len(STAT_ORDER) or not style:
        return []
    if style == "new":
        for stat, bars in comp_bars(psd).items():
            if len(bars) == 1 and (bars[0][1] is None or not bars[0][0].is_visible()):
                return [_note("stats", f"{stat}: its bar shows no layer comp")]
        if not _art_over_darks(psd):
            return []
    exposures = [layer for layer in _digits(stat_rows(psd)) if layer.kind == "exposure"]
    if any(one.is_visible() and one.exposure > 0 for one in exposures) and any(not one.is_visible() and one.exposure < 0 for one in exposures):
        return [BRIGHT_NOTE]
    png = np.asarray(image.convert("RGB"), np.int16) if image is not None else None
    for what, fields in (("edits", {stat: value - 1 if value > 0 else value + 1 for stat, value in have.items()}),
                         ("raises", {stat: value + 1 if value < 10 else value - 1 for stat, value in have.items()})):
        state, reason, moved = _real_bake(ctx, rel, fields, png)
        if state == "refused":
            return [_note("stats", reason)]
        still = _still(psd, moved, style) if state == "baked" and moved is not None else []
        if still:
            names = "Stat" if len(still) == len(STAT_ORDER) else " and ".join(STAT_NAMES[stat] for stat in still)
            return [NO_CHANGE_NOTE.format(names, what)]
    return []


def edit_notes(ctx: Context, rel: str, psd: PSDImage, parsed: dict, style: str, image: Optional[Image.Image]) -> list:
    """The website edits a card cannot take, a sentence each (the module's docstring: which are tried)."""
    card = parsed["card"]
    one = Card(card_name(rel), CardAuthors.key(card_name(rel)), {}, rel, ctx.snapshot.sha(rel))
    work = mkdtemp(prefix="card-checks_", dir=environ.get(card_bake.WORK_ENV) or None)
    pictures = len(variants_of(psd)) > 1
    notes = []

    def refused(what: str, refusal) -> None:
        if not search(WHOLE_FIELD, refusal.why, IGNORECASE):
            notes.append(_note(what, f"{refusal.code}: {refusal.why}"))

    def plan(what: str, fields: dict, template=None) -> bool:
        found = psd_edit.classify(parsed, fields, psd, ctx.parser, template)
        for refusal in found.refusals:
            refused(what, refusal)
        if psd_edit.PICTURE_TITLE in found.site.values():
            notes.append(TITLE_NOTE)
        return bool(found.jobs)

    def types(what: str, value: list, template) -> None:
        if not plan(what, {"types": value}, template):
            return
        raw = ctx.snapshot.fetch(ctx.snapshot.sha(rel))
        if len(raw) < SMALL_PSD:                # a made icon can pass gate_size's share: only the bake's save tells
            state, reason, _ = _real_bake(ctx, rel, {"types": value})
            if state == "refused":
                notes.append(_note(what, reason))
            return
        try:                                    # a made icon's ink shows once it is made: on a copy, no render
            icons = card_bake.types_icons(ctx.snapshot, work, ctx.parser, value, card.get("types"))
            copy = psd_edit.edit_copy(psd, raw)
            psd_edit.set_icons(copy, psd_edit.plan_types(copy, parsed, value, ctx.parser, template, icons), icons)
        except psd_edit.Refused as refusal:
            refused(what, refusal.refusal)

    def stars(what: str, value: int) -> None:
        if pictures:                            # each picture bakes: their layers must match
            state, reason, _ = _real_bake(ctx, rel, {"stars": value})
            if state == "refused":
                notes.append(_note(what, sub(r"picture \d+'s bake made other layers than picture \d+'s",
                                             "its pictures bake different layers", reason)))
        elif plan(what, {"stars": value}):
            try:
                psd_edit.plan_stars(psd, parsed, value, ctx.parser, moving=True)
            except psd_edit.Refused as refusal:
                if not psd_edit.star_folder(rel):           # in a star folder the card moves all the same
                    refused(what, refusal.refusal)
    try:
        plan("name", {"name": (psd_edit.shown_name(parsed) or one.name) + " II"})
        count, printed = card.get("stars"), list(card.get("types") or [])
        if isinstance(count, int):
            if count < MAX_STARS:
                stars("adding a star", count + 1)
            if count > 1:
                stars("removing a star", count - 1)
        template = card_bake.first_type_spot(ctx.snapshot, work, one, parsed, psd, ctx.parser)
        types("adding a type", printed + [next(t for t in TYPE_TRIES if t not in printed)], template)
        if printed:
            types("removing a type", printed[1:], template)
        if any(EFFECT_LAYER_TAG in layer.name for layer in psd.descendants()):
            edited = _ability_edit(card.get("ability"))
            state, reason, _ = _real_bake(ctx, rel, {"ability": edited}) if edited else ("", "", None)
            if state == "refused" and not search(WHOLE_FIELD, reason, IGNORECASE):
                notes.append(_note("ability", reason))
        notes += _stat_notes(ctx, rel, psd, parsed, style, image)
    finally:
        rmtree(work, ignore_errors=True)
    return list(dict.fromkeys(notes))


# ========================
# One card
# ========================
def check_card(ctx: Context, rel: str) -> dict:
    """
    A card PSD's record: {"card", "png", "style", "reads", "notes", "bars"}; mismatches() and flags() word it.

    Args:
        ctx: context()
        rel: Its repo path
    """
    snapshot = ctx.snapshot
    png_rel = snapshot.png_for(rel)[0]
    psd = PSDImage.open(BytesIO(snapshot.fetch(snapshot.sha(rel))))
    image = Image.open(BytesIO(snapshot.fetch(snapshot.sha(png_rel)))) if png_rel else None
    if image is not None:
        image.load()
    parsed = psd_edit.parse_card(ctx.parser, rel, rel, psd)
    creatures = parsed["card"].get("creatures")
    style, why = bar_style(psd, creatures)
    notes = [OLD_NOTE] if style == "old" else []
    if style == "custom" and creatures:
        notes.append(CREATURES_NOTE)
    wrong = swapped(psd) if style else []
    if wrong:
        notes.append(SWAP_NOTE.format(" and ".join(STAT_NAMES[stat] for stat in wrong)))
    reads = read_picture(psd, image) if style and image is not None and len(variants_of(psd)) < 2 else {}
    notes += edit_notes(ctx, rel, psd, parsed, style, image)
    for path in snapshot.dup_pictures().get(rel, []) if image is not None else ():   # nothing redraws these (owner 11-07)
        if png_export.drawn_apart(psd, image, png_export.open_png(snapshot.fetch(snapshot.sha(path)))):
            notes.append(DUP_NOTE.format(name=basename(path)))
    return {"card": card_name(rel), "png": png_key(snapshot, rel), "style": style, "reads": reads,
            "notes": notes, "bars": CUSTOM_NOTES[why] if style == "custom" else None,
            "unredrawable": unredrawable(psd, rel, snapshot.tree)}


def unredrawable(psd: PSDImage, rel: str, tree) -> dict:
    """{linked path: [layer, why]} of the linked layers Card sync cannot draw again when their file changes (a warp, smart
    filters: psd_links.why_not_redrawn); a fact of the blob, worded only once its file changed (link_notes)."""
    try:
        from commands.psd_links import why_not_redrawn
    except ImportError:                         # the links module not checked out: none known
        return {}
    paths = {item["uuid"]: item["path"] for item in linked_items(psd, rel, tree) if item["path"]}
    out: dict = {}
    for layer in psd.descendants():
        placed = placed_data(layer)
        key = str(getattr(placed.get(b"Idnt"), "value", placed.get(b"Idnt")) or "").strip("\x00") if placed else ""
        if paths.get(key) and paths[key] not in out:
            why = why_not_redrawn(layer)
            if why:
                out[paths[key]] = [layer.name, why]
    return out


def look_alikes(links: dict, sha: Callable[[str], str], paths) -> dict:
    """{MDW/Linked path: the file it looks the same as}: equal decoded pixels (db/links.json's pixels), other bytes; the
    one most PSDs link is kept (the older, Q6: no dates at depth 1), then the first by path (G3 merges byte-equal
    copies; these only get the note)."""
    groups: dict = {}
    for path in sorted(paths):
        found = (links.get("pixels") or {}).get(sha(path)) if path.startswith("MDW/Linked/") else None
        if found:
            groups.setdefault(found, []).append(path)
    users: dict = {}
    for rel, record in links.items():
        if rel not in ("files", "pixels") and isinstance(record, dict):
            for link in {link.get("path") for link in record.get("links") or []}:
                users[link] = users.get(link, 0) + 1
    out = {}
    for group in groups.values():
        kept = min(group, key=lambda p: (-users.get(p, 0), p))
        out.update({path: kept for path in group if sha(path) != sha(kept)})
    return out


def link_notes(rel: str, blob: str, links: dict, sha: Callable[[str], str], alike: dict, record: dict) -> list:
    """The hand notes of a PSD's links (db/links.json at its blob) against the tree now: a file the repo lacks, a
    look-alike, a changed file Card sync could not draw again (record's unredrawable). [] without a record at the blob."""
    found = links.get(rel)
    if not isinstance(found, dict) or found.get("blob") != blob:
        return []
    out = []
    for link in found.get("links") or []:
        path, name = link.get("path") or "", link.get("name") or ""
        if not path or not sha(path):
            out.append(LINK_MISSING.format(name=name or basename(path)))
            continue
        if path in alike:
            out.append(LINK_ALIKE.format(name=basename(path), kept=basename(alike[path])))
        size = ((links.get("files") or {}).get(path) or {}).get(sha(path))
        due = size is not None and (link.get("size") != size or bool(link.get("drawn") and link["drawn"] != sha(path)))
        stuck = (record.get("unredrawable") or {}).get(path)
        if due and stuck:
            out.append(LINK_STUCK.format(layer=stuck[0], name=basename(path), why=stuck[1]))
    return list(dict.fromkeys(out))


def flags(record: dict, data: Optional[dict], site=(), linked=()) -> dict:
    """{"hand": note or None, "bars": note or None} of a record against the card's data (linked: link_notes)."""
    hand = (mismatches(record, data, site) if data else []) + list(record.get("notes") or []) + list(linked)
    return {"hand": " ".join(hand) or None, "bars": record.get("bars")}


# ========================
# The file
# ========================
def load(path: str) -> dict:
    """db/card_checks.json, {"version", "cards": {PSD blob: record}, "cleared": {PSD path: flags' psd}}; empty when new."""
    if not exists(path):
        return {"version": VERSION, "cards": {}, "cleared": {}}
    with open(path, encoding="utf-8") as fh:
        return json_load(fh)


def text_of(checks: dict) -> str:
    """The file's text, one record a line."""
    cards = ",\n".join(f"{dumps(sha)}: {dumps(checks['cards'][sha], ensure_ascii=False, sort_keys=True)}" for sha in sorted(checks["cards"]))
    return (f'{{"version": {VERSION},\n"cleared": {dumps(checks.get("cleared") or {}, ensure_ascii=False, sort_keys=True)},\n'
            f'"cards": {{\n{cards}\n}}}}\n')


def png_key(snapshot: Snapshot, rel: str) -> str:
    """A record's "png": its PNG's blob (8 hex digits), its picture-only duplicates' after it; a change checks it again."""
    png = snapshot.png_for(rel)[0]
    return ",".join([snapshot.sha(png)[:8] if png else ""]
                    + [snapshot.sha(path)[:8] for path in snapshot.dup_pictures().get(rel, [])])


def to_check(snapshot: Snapshot, checks: dict, everything: bool = False, only=(), shard: tuple = (0, 1)) -> list:
    """The card PSDs to check: no result for the blob, or for its PNG (and duplicates); only: card keys."""
    out = []
    for key, rel in sorted(snapshot.psds.items(), key=lambda item: item[1]):
        sha = snapshot.sha(rel)
        record = checks["cards"].get(sha)
        stale = record is None or record.get("png") != png_key(snapshot, rel)
        if (key in only if only else everything or stale) and int(sha[:8], 16) % shard[1] == shard[0]:
            out.append(rel)
    return out


def blobs_of(snapshot: Snapshot, rels: list) -> list:
    """The blobs check_card reads: the PSDs and PNGs, and what every card shares (one fetch, no REST read a card)."""
    pngs = [snapshot.png_for(rel)[0] for rel in rels] + [p for rel in rels for p in snapshot.dup_pictures().get(rel, [])]
    shared = [path for path in snapshot.tree if path in (psd_edit.STATUS_CHART, ICON_MANIFEST_PATH)
              or path.startswith(psd_edit.TEMPLATES_DIR + "/") or path.lower() in {f"{TYPES_DIR}/{t}.png".lower() for t in TYPE_TRIES}]
    return [snapshot.sha(path) for path in rels + [png for png in pngs if png] + (shared if rels else [])]


def refresh(checks: dict, snapshot: Snapshot, stats: dict, edits: CardEdits, links: Optional[dict] = None) -> None:
    """Every record's "hand" and "bars" against the data and the links (db/links.json: a pushed file clears its note),
    the records of PSDs gone from the tree out, and the people's flags that went: set on another PSD than the card's,
    whose checks are clean (build_site leaves them out)."""
    cards = {card.get("blob_sha"): card for card in stats.values() if card.get("blob_sha")}
    blobs = {snapshot.sha(rel): key for key, rel in snapshot.psds.items()}
    checks["cards"] = {sha: record for sha, record in checks["cards"].items() if sha in blobs}
    def sha(path: str) -> str:
        return snapshot.sha(path) if path in snapshot.tree else ""
    alike = look_alikes(links, sha, snapshot.tree) if links else {}
    for blob, record in checks["cards"].items():
        linked = link_notes(snapshot.psds[blobs[blob]], blob, links, sha, alike, record) if links else []
        record.update(flags(record, cards.get(blob), edits.cards.get(blobs[blob]) or {}, linked))
    was, cleared = checks.get("cleared") or {}, {}
    for key, entry in edits.cards.items():
        mark, rel = (entry.get("flags") or {}).get("psd"), snapshot.psds.get(key)
        if not mark or not rel:
            continue
        record = checks["cards"].get(snapshot.sha(rel))
        clean = record is not None and not record["hand"] and not record["bars"] and not snapshot.sha(rel).startswith(mark)
        if clean or was.get(rel) == mark:
            cleared[rel] = mark                 # gone stays gone though the checks speak again; a new flag starts over
    checks["cleared"] = cleared


# ========================
# Public API Functions
# ========================
_worker: list = []


def _start(repository: str, clone: str, head: str) -> None:
    token_from_env()                            # a blob the clone lacks comes by REST
    _worker[:] = [context(repository, clone, head)]


def _one(rel: str) -> tuple:
    """(rel, record, seconds), or the error line in the record's place: one card never loses the others."""
    started = perf_counter()
    try:
        return rel, check_card(_worker[0], rel), perf_counter() - started
    except Exception as error:                  # noqa: BLE001 - it is checked again by the next run
        return rel, f"{type(error).__name__}: {error}", perf_counter() - started


def run(rels: list, repository: str, clone: str, head: str, workers: int, minutes: float, out: str) -> dict:
    """check_card over rels in worker processes, for at most this long; {blob: record}, written to out as it grows."""
    snapshot = local_snapshot(repository, clone, head)
    found, started = {}, perf_counter()
    with ProcessPoolExecutor(workers, initializer=_start, initargs=(repository, clone, head)) as pool:
        jobs = [pool.submit(_one, rel) for rel in rels]
        for n, job in enumerate(as_completed(jobs), 1):
            rel, record, seconds = job.result()
            print(f"{n}/{len(rels)} {rel}: {record if isinstance(record, str) else 'done'} ({seconds:.1f} s)", flush=True)
            if isinstance(record, dict):
                found[snapshot.sha(rel)] = {**record, "seconds": round(seconds, 1)}
                with open(out, "w", encoding="utf-8") as fh:
                    fh.write(dumps(found, ensure_ascii=False))
            if perf_counter() - started > minutes * 60:
                print(f"out of time: {len(rels) - n} cards wait for the next run", flush=True)
                for left in jobs:
                    left.cancel()
                break
    return found


def main(args: Optional[list] = None) -> int:
    """The command line (module docstring)."""
    parser = ArgumentParser(prog="python -m commands.card_checks", description="Automatic card flags")
    parser.add_argument("action", choices=("list", "run", "save"))
    parser.add_argument("results", nargs="*", help="save: run's files")
    parser.add_argument("--clone", required=True, help="a checkout of the card repo (blobless is enough)")
    parser.add_argument("--db", required=True, help="the card data's db/")
    parser.add_argument("--head", default="", help="the commit (default: the clone's HEAD)")
    parser.add_argument("--repo", default=CARD_REPO)
    parser.add_argument("--all", action="store_true", help="every card, not only the PSDs without a result")
    parser.add_argument("--cards", default="", help="only these cards (comma separated)")
    parser.add_argument("--shard", default="0/1", help="i/n: this share of the cards")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--minutes", type=float, default=50)
    parser.add_argument("--out", default="", help="run: where its results go")
    options = parser.parse_args(list(argv[1:] if args is None else args))
    if options.action == "run" and not options.out:
        parser.error("run needs --out")
    token_from_env()
    snapshot = local_snapshot(options.repo, options.clone, options.head)
    path = join(options.db, CHECKS_JSON)
    checks = load(path)
    shard = tuple(int(part) for part in options.shard.split("/"))
    only = {CardAuthors.key(name) for name in options.cards.split(",") if name.strip()}
    if options.action == "save":
        for result in options.results:
            with open(result, encoding="utf-8") as fh:
                checks["cards"].update({sha: {k: v for k, v in record.items() if k != "seconds"}
                                        for sha, record in loads(fh.read()).items()})
        with open(join(options.db, STATS_JSON), encoding="utf-8") as fh:
            stats = json_load(fh)["cards"]
        text = snapshot.fetch(snapshot.sha(EDITS_PATH)).decode("utf-8") if EDITS_PATH in snapshot.tree else "{}"
        links = {}
        if exists(join(options.db, LINKS_JSON)):
            with open(join(options.db, LINKS_JSON), encoding="utf-8") as fh:
                links = json_load(fh)
        refresh(checks, snapshot, stats, CardEdits.parse(text), links)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text_of(checks))
        print(f"{len(checks['cards'])} records, {sum(1 for r in checks['cards'].values() if r.get('hand'))} hand, "
              f"{sum(1 for r in checks['cards'].values() if r.get('bars'))} bars")
        return 0
    rels = to_check(snapshot, checks, options.all, only, shard)
    if options.action == "list":
        print("\n".join(blobs_of(snapshot, rels)))
    else:
        run(rels, options.repo, options.clone, options.head, options.workers, options.minutes, options.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'SPARSE_FILES',
    'VERSION',
    'blobs_of',
    'check_card',
    'context',
    'edit_notes',
    'flags',
    'link_notes',
    'load',
    'look_alikes',
    'main',
    'mismatches',
    'png_key',
    'read_picture',
    'read_row',
    'refresh',
    'run',
    'swapped',
    'text_of',
    'to_check',
    'unredrawable',
]
