"""
Template eras and stat bars of card pictures that have no PSD, for the card timelines (docs/Future/CARD_TIMELINE.md,
step 3 Read): the Discord archive's card pictures and the old pre-wipe repo's PNG versions. Run from the repo root
with a Python that has Pillow and numpy (OCR of old-repo pictures needs pyobjc's Vision, as commands/discord_match.py):

    python -m commands.era_bars truth     [--root <dir>] [--old-git <dir>] [--workers <n>]
    python -m commands.era_bars read      [--root <dir>] [--old-git <dir>] [--workers <n>] [--only discord|oldgit]
    python -m commands.era_bars calibrate [--root <dir>]
    python -m commands.era_bars add       [--root <dir>] [--workers <n>]
    python -m commands.era_bars versions  [--root <dir>] [--old-git <dir>] [--cards-repo <clone>] [--workers <n>]

truth      truth/psd.json: each old-repo creature or minion PNG version's stats as its same-commit PSD gives them
           (today's parser on digit rows and comp bars; the rows' own layers on the alt template and masked rows).
read       Each distinct card picture of the card channels (the eras pass's derived/eras/discord_feats.jsonl: look
           features and Vision OCR lines, its first post) gets an era (era_of; by-eye classes in truth/eras.json win)
           and, when it shows stat bars, its stats by that era's readers (the matcher's card, derived/match/
           verdicts.jsonl when there, names two-creature, icon, three-form and busy-bar cards beside the OCR'd title);
           each old-repo PNG version (derived/oldgit/versions.jsonl, blobs from the old .git, read only) the same, plus
           a 400 px JPEG copy of it (as Discord shows cards). Every stat carries its method ('picture'), reader,
           confidence and how its row was found. -> reads.jsonl
calibrate  Each era's truth (truth/*.json: old-repo PSD stats where its digit rows give them, by-eye readings
           elsewhere) against reads.jsonl: per era the lowest confidence bar whose reads at or above it are SURE right,
           with per stat n, accuracy at the bar and the share above it (and on Discord pictures alone, with a Wilson
           bound) -> calibration.json; reads.jsonl's 'sure' marks follow it (never on a held row or a NEVER_SURE group).
add        The card-sure Discord pictures (derived/match/verdicts.jsonl) read nowhere above (other channels, posts after
           2023-11-06): their era from the text readers (derived/readers/text/reads.jsonl), else era_of; their OCR lines
           the text readers' (ocr.jsonl), else Vision's; read as above and made sure by calibration.json as it stands
           (its bars are not fitted again) -> appended to reads.jsonl.
versions   The git versions the look checks compare a picture with: the card repo's PNG versions the matcher knows
           (derived/matcher/match/pairs.json) whose blob the clone stores (never a lazy fetch) and the old repo's item
           versions (read only), era by size; read as 'read' does (the PNG and its 400 px JPEG) and made sure by
           calibration.json as it stands (a 1:2 card repo version: the stricter of D2's and D3's bars, its
           construction unknown) -> reads.jsonl (src 'tc', and 'oldgit' rows marked 'item'), replacing earlier ones.
Outputs go to the archive root's derived/readers/bars/ (never a repo). Nothing here talks to Discord.
"""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from json import dumps, load, loads
from multiprocessing import get_context
from os import listdir, makedirs, replace
from os.path import exists, expanduser, join
from re import compile as re_compile, fullmatch, search, sub
from sys import stderr
from typing import Optional

import numpy as np
from PIL import Image

from commands import discord_match as dm

# ========================
# Configuration
# ========================
OUT = ("derived", "readers", "bars")                # under the archive root
ERAS = ("derived", "eras")                          # the eras pass's outputs
OLDGIT = ("derived", "oldgit")
VERDICTS = ("derived", "match", "verdicts.jsonl")   # the matcher's card per Discord picture (when there)
TEXT = ("derived", "readers", "text")               # the text readers: eras (reads.jsonl) and OCR lines (ocr.jsonl)
CARD_SURE = ("version_sure", "version_unsure", "draft")
ORDER = ("hp", "def", "atk", "spd")                 # the bar rows, top to bottom
LABELS = {"hp": r"health|heaith|healt|hp", "def": r"defen[cs]e|defemse|deffense|def",
          "atk": r"attack|atack|attak|attach|atk", "spd": r"speed|spead|spee|spd"}
TEMPLATE_CELLS = (0, 75, 158, 245, 326, 404, 487, 568, 649, 726, 813)    # card_checks.TEMPLATE_CELLS
D_BOX = (360 / 1200, 1173 / 1200)                   # the template's bars (eras D1-D3), x of the card's width
D_ROWS, D_H = (1200, 1326, 1452, 1578), 84          # their rows' tops and height, of 2400
RAID_BOX = (451 / 2400, 2317 / 2400)                # the raid template (2400x2400): 20 cells a row
RAID_ROWS, RAID_H, RAID_CELLS = (1139, 1286, 1433, 1580), 97, 20
TWINS = re_compile(r"tw[il1][nm]\s*(princes?|emperors?)|g[o0]d\s*tamer")   # two creatures' bars a row (top, bottom); OCR slips
WORK = 1600                                         # pictures read at this long side (small ones scaled up)
SPLIT_MIN = 8.0                                     # a row's best split scores this (Lab) or it is one colour
TEMPLATE_PRIOR = 1.15                               # a 1:2 card: its fixed box unless a fit's cells are this much more even
SURE = 0.98                                         # 'sure' reads are at least this share right on their era's truth
MARGIN_FULL = 30.0                                  # a split this far (Lab) ahead of the runner-up count: a sure row
TELL_LOW, TELL_FULL = 0.4, 0.8                      # a one-colour row's tell: worthless under, sure from
FLAT_REF = 25.0                                     # a one-colour row needs 2 split rows or one this clear (margin)
AGREE_ALONE = 0.4                                   # the agreement a read gets when no other reader ran
MIN_TRUTH = 30                                      # stats an era's bar needs under it (else nothing there is sure)
WINDOW = 50                                         # the least sure reads above a bar that must be SURE right too
JPEG = (400, 70)                                    # the Discord-like copy of an old-repo PNG: width, quality
CREATURE_ERAS = ("A", "B", "C", "D1", "D2", "D3", "R")
TEMPLATE_FROM = "2022-06"                            # the 400x800 template's first post (era D1)
ICONS = re_compile(r"satou|tainted\s*(keeper|lost)|hive\s*knight|minecraft\s*panda|tyler\s*zao")   # icon bars at times
MULTI = re_compile(r"\bpix\b|computer\s*virus")      # three forms' colours in each row: one stat set can't say
BUSY = re_compile(r"\W*omori\W*")                    # bars drawn as arms: never sure
BUSY_TEMPLATE = re_compile(r"\W*(paint\s*warrior|frog)\W*")   # heavy texture bars on the template (D1-D3): never sure
NEVER_SURE = ("multi", "busy")                       # groups no bar applies to
TEMPLATE_ERAS = ("D1", "D2", "D3")
PAIRS = ("derived", "matcher", "match", "pairs.json")    # the matcher's references: the card repo's versions
ITEM_W, ITEM_ASP = 1150, (1.45, 1.55)                # the item template (1200x1800): bars there are an item's


def log(text: str):
    print(text, file=stderr, flush=True)


# ========================
# Pieces
# ========================
def _lab(rgb) -> np.ndarray:
    """sRGB 0-255 to Lab (card_checks._lab)."""
    c = np.asarray(rgb, np.float64) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    f = xyz / np.array([0.9505, 1.0, 1.089])
    f = np.where(f > 0.008856, np.cbrt(f), 7.787 * f + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def uniform(n: int = 10) -> tuple:
    return tuple(range(n + 1))


def is_item_size(w: int, h: int) -> bool:
    return w >= ITEM_W and ITEM_ASP[0] <= h / w <= ITEM_ASP[1]


def labels(lines, strict: bool = False) -> dict:
    """{stat: (x0, y0, x1, y1) fractions} of the printed stat labels among OCR lines (strict: no line with a digit
    or '+', as an item's '+2 HP' text)."""
    out = {}
    for text, _conf, x, y, w, h in lines or []:
        if strict and search(r"[0-9+]", text):
            continue
        word = sub(r"[^a-z]", "", text.lower())
        for stat, pattern in LABELS.items():
            if stat not in out and fullmatch(pattern, word):
                out[stat] = (x, y, x + w, y + h)
    return out


def fill_rows(found: dict):
    """({stat: row middle y}, pitch) by the labels' even spacing (a missed label placed by the others); pitch None
    under two labels."""
    ys = {s: (b[1] + b[3]) / 2 for s, b in found.items()}
    if len(ys) < 2:
        return ys, None
    idx = np.array([ORDER.index(s) for s in ys])
    slope, at = np.polyfit(idx, np.array(list(ys.values())), 1)
    return {s: ys.get(s, at + slope * ORDER.index(s)) for s in ORDER}, float(slope)


def cell_colours(row: np.ndarray, cells: tuple) -> np.ndarray:
    """Each cell's mean Lab on its middle (60% of its width and of the row's height)."""
    h, w = row.shape[:2]
    edges = [int(round(e * w / cells[-1])) for e in cells]
    out = []
    for a, b in zip(edges, edges[1:]):
        cut = (b - a) * 0.2
        part = row[int(h * 0.2):max(int(h * 0.8), int(h * 0.2) + 1), int(a + cut):max(int(b - cut), int(a + cut) + 1)]
        out.append(_lab(part.reshape(-1, 3).mean(0)) if part.size else np.zeros(3))
    return np.array(out)


def split_read(row: np.ndarray, cells: tuple) -> dict:
    """
    A two-tone row: the count k whose first k (lit) and other (empty) cells are farthest apart against their own
    spread. {'best', 'score', 'margin' (over the runner-up count), 'flat'}: flat when no count scores SPLIT_MIN (one
    colour: none or all lit, tell_flats decides).
    """
    c = cell_colours(row, cells)
    n = len(c)
    scores = []
    for k in range(1, n):
        a, b = c[:k], c[k:]
        gap = np.linalg.norm(a.mean(0) - b.mean(0))
        spread = max(np.linalg.norm(a - a.mean(0), axis=1).max(), np.linalg.norm(b - b.mean(0), axis=1).max())
        scores.append(gap - 1.5 * spread)
    order = np.argsort(scores)[::-1]
    best, score = int(order[0]) + 1, float(scores[order[0]])
    second = float(scores[order[1]]) if n > 2 else -1e9
    if score < SPLIT_MIN:
        return {"best": None, "flat": True, "mean": c.mean(0).tolist(), "score": score, "margin": 0.0, "n": n}
    return {"best": best, "flat": False, "score": score, "margin": score - max(second, 0.0), "n": n,
            "lit": c[:best].mean(0).tolist(), "empty": c[best:].mean(0).tolist()}


def tell_flats(reads: dict):
    """A one-colour row is all lit when nearer the card's lit cells than its empty ones, else none; its 'tell' is
    how clearly (0..1)."""
    lit = [r["lit"] for r in reads.values() if r and not r["flat"]]
    emp = [r["empty"] for r in reads.values() if r and not r["flat"]]
    for r in reads.values():
        if not r or not r["flat"]:
            continue
        if not lit:
            r["tell"] = 0.0
            continue
        m = np.array(r["mean"])
        to_lit, to_empty = np.linalg.norm(m - np.mean(lit, 0)), np.linalg.norm(m - np.mean(emp, 0))
        r["best"] = r["n"] if to_lit < to_empty else 0
        r["tell"] = float(abs(to_empty - to_lit) / max(to_empty + to_lit, 1e-6))


def grid_fit(img: np.ndarray, ys: list, half: float, x_from: float, cells: tuple, mode: str = "median") -> tuple:
    """
    The bars' cell grid the rows share: the offset and width (px) whose cell edges sit on the strongest vertical edges
    of the rows' middles (median row: edges every row has; sum: all of them). Bars span 50-85% of a card's width.
    """
    H, W = img.shape[:2]
    g = img.astype(np.float32).mean(-1)
    rows = []
    for y in ys:
        y0, y1 = max(0, int(y - 0.5 * half)), min(H, int(y + 0.5 * half))
        band = g[y0:max(y1, y0 + 1)]
        if band.size:
            p = np.pad(np.abs(band[:, 2:] - band[:, :-2]).mean(0), (1, 1))
            rows.append(np.maximum.reduce([p, np.roll(p, 1), np.roll(p, -1)]))
    prof = (np.median(rows, 0) if mode == "median" else np.sum(rows, 0)) if rows else np.zeros(W, np.float32)
    rel = np.array(cells, np.float32) / cells[-1]
    x_from = int(max(0, x_from))
    best, span = -1.0, (x_from, W)
    for width in range(int(W * 0.5), int(W * 0.85), max(1, W // 400)):
        xs = (rel * width).astype(int)
        n = W - 1 - width - x_from
        if n <= 0:
            continue
        s = sum(prof[x_from + k:x_from + k + n] for k in xs)
        i = int(s.argmax())
        if s[i] > best:
            best, span = float(s[i]), (x_from + i, x_from + i + width)
    return span


def cell_spread(img: np.ndarray, yp: list, half: float, span: tuple, cells: tuple) -> float:
    """Mean colour spread inside the cells' middles of every row: low when the grid sits on the cells."""
    x0, x1 = span
    out = []
    for y in yp:
        row = img[int(max(0, y - half)):int(min(img.shape[0], y + half)), x0:x1]
        if row.size == 0 or row.shape[0] < 3:
            continue
        h, w = row.shape[:2]
        edges = [int(round(e * w / cells[-1])) for e in cells]
        for a, b in zip(edges, edges[1:]):
            cut = (b - a) * 0.2
            part = row[int(h * 0.2):max(int(h * 0.8), int(h * 0.2) + 1), int(a + cut):max(int(b - cut), int(a + cut) + 1)]
            if part.size:
                out.append(float(np.linalg.norm(_lab(part.reshape(-1, 3)).std(0))))
    return float(np.mean(out)) if out else 1e9


# ========================
# Readers: each returns {stat: split_read result} (rows as read) and how
# ========================
def _label_rows(H: int, found: dict, only: bool):
    """(row middles px, half height px) by the labels; only: the labels' own rows (an item's), none placed past them."""
    ys, pitch = fill_rows(found)
    if pitch is None:
        return None, None
    return {s: ys[s] * H for s in ORDER if not only or s in found}, abs(pitch) * H * 0.34


def bar_extent(img: np.ndarray, y: float, half: float, span: tuple, cells: tuple) -> Optional[tuple]:
    """
    The bar's own rows (y0, y1 px) about its label's middle y (which can sit off the bar's): the pixel rows where most
    cell dividers are darker than the cells beside them, lit or empty; None when no clear run.
    """
    H = img.shape[0]
    x0, x1 = span
    a, b = int(max(0, y - 1.4 * half)), int(min(H, y + 1.4 * half))
    w = x1 - x0
    if b - a < 8 or w < 20:
        return None
    g = img[a:b, x0:x1].astype(np.float32).mean(-1)
    pad = max(2, w // 120)
    edge = [int(round(e * w / cells[-1])) for e in cells]
    mid = [g[:, max(0, (e0 + e1) // 2 - pad):(e0 + e1) // 2 + pad + 1].mean(1) for e0, e1 in zip(edge, edge[1:])]
    hits = [(np.minimum(mid[k], mid[k + 1]) - g[:, max(0, e - pad):e + pad + 1].min(1)) > 12
            for k, e in enumerate(edge[1:-1])]
    on = np.mean(hits, 0) >= 0.6
    runs, start = [], None
    for i, v in enumerate(list(on) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if runs and start - runs[-1][1] <= max(4, 0.3 * min(i - start, runs[-1][1] - runs[-1][0])):
                runs[-1] = (runs[-1][0], i)              # a line between the halves
            else:
                runs.append((start, i))
            start = None
    if not runs:
        return None
    at = y - a
    r0, r1 = min(runs, key=lambda r: 0 if r[0] <= at < r[1] else min(abs(at - r[0]), abs(at - r[1])))
    return (a + r0, a + r1) if r1 - r0 >= 6 else None


def _rows(img: np.ndarray, yp: dict, half: float, span: tuple, cells: tuple, halves: bool = False) -> dict:
    """Each stat's row (or its top and bottom halves: two creatures) read across span."""
    H = img.shape[0]
    x0, x1 = span
    out = {}
    for s in ORDER:
        y = yp.get(s)
        if y is None:
            out[s] = None
            continue
        y0, y1 = int(max(0, y - half)), int(min(H, y + half))
        if halves:                                       # two creatures: split the bar itself, not the label's window
            y0, y1 = bar_extent(img, y, half, span, cells) or (y0, y1)
        row = img[y0:y1, x0:x1]
        if row.size == 0 or row.shape[0] < 3:
            out[s] = None
        elif halves:
            mid = row.shape[0] // 2
            out[s] = (split_read(row[:mid], cells), split_read(row[mid:], cells))
        else:
            out[s] = split_read(row, cells)
    if halves:
        for k in (0, 1):
            tell_flats({s: (r[k] if r else None) for s, r in out.items()})
    else:
        tell_flats(out)
    return out


def read_uniform(img: np.ndarray, found: dict, halves: bool = False, only: bool = False) -> tuple:
    """Label rows and one grid of 10 equal cells fitted right of the labels (the white, black and full-art eras);
    halves: each row's top and bottom halves (two creatures)."""
    H, W = img.shape[:2]
    yp, half = _label_rows(H, found, only)
    if yp is None:
        return None, None
    span = grid_fit(img, list(yp.values()), half, max(b[2] for b in found.values()) * W + 2, uniform(), "sum")
    return _rows(img, yp, half, span, uniform(), halves), dict(span=(span[0] / W, span[1] / W), cells=10)


def read_best(img: np.ndarray, found: dict, halves: bool = False, only: bool = False) -> tuple:
    """
    Label rows and the most even of several grids (uniform or template cells, fitted from the labels' right with the
    summed rows or from the shortest label with the median row; on a 1:2 card the template's own box, kept unless a
    fit's cells are TEMPLATE_PRIOR times more even).
    """
    H, W = img.shape[:2]
    yp, half = _label_rows(H, found, only)
    if yp is None:
        return None, None
    starts = {"sum": max(b[2] for b in found.values()) * W + 2, "median": (min(b[2] for b in found.values()) - 0.02) * W}
    cands = [(name, cells, mode, grid_fit(img, list(yp.values()), half, x_from, cells, mode))
             for name, cells in (("uniform", uniform()), ("template", TEMPLATE_CELLS)) for mode, x_from in starts.items()]
    if 1.8 <= H / W <= 2.2:
        cands.append(("template", TEMPLATE_CELLS, "fixed", (int(D_BOX[0] * W), int(D_BOX[1] * W))))

    def flat_rows(span, cells):
        x0, x1 = span
        k = 0
        for y in yp.values():
            row = img[int(max(0, y - half)):int(min(H, y + half)), x0:x1]
            k += row.size == 0 or row.shape[0] < 3 or split_read(row, cells)["flat"]
        return k
    cands = [c for c in cands if flat_rows(c[3], c[1]) < 3] or cands      # a grid on plain background reads no step
    scored = sorted(((cell_spread(img, list(yp.values()), half, span, cells), name, mode, cells, span)
                     for name, cells, mode, span in cands), key=lambda c: c[0])
    fixed = [c for c in scored if c[2] == "fixed"]
    if fixed and fixed[0][0] <= TEMPLATE_PRIOR * scored[0][0]:
        scored = fixed
    spread, name, mode, cells, span = scored[0]
    return _rows(img, yp, half, span, cells, halves), dict(cells=name, fit=mode, span=(span[0] / W, span[1] / W),
                                                          spread=round(spread, 2))


def read_fixed(img: np.ndarray, kind: str = "template", halves: bool = False) -> tuple:
    """A template's own rows, no OCR: 'template' (1:2 cards, eras D1-D3) or 'raid' (2400x2400, 20 cells)."""
    H, W = img.shape[:2]
    box, tops, h, cells = ((D_BOX, D_ROWS, D_H, TEMPLATE_CELLS) if kind == "template"
                           else (RAID_BOX, RAID_ROWS, RAID_H, uniform(RAID_CELLS)))
    side = 2400
    yp = {s: (t + h / 2) * H / side for s, t in zip(ORDER, tops)}
    span = (int(box[0] * W), int(box[1] * W))
    return _rows(img, yp, h / 2 * H / side, span, cells, halves), dict(cells=kind, fit="fixed", span=box)


def read_raid(img: np.ndarray, found: dict) -> tuple:
    """A raid boss: label rows and a 20-cell grid (the fixed raid box when it fits as well)."""
    H, W = img.shape[:2]
    ys, pitch = fill_rows(found)
    if pitch is None:
        return read_fixed(img, "raid")
    half = abs(pitch) * H * 0.34
    yp = {s: ys[s] * H for s in ORDER}
    cells = uniform(RAID_CELLS)
    fit = grid_fit(img, list(yp.values()), half, max(b[2] for b in found.values()) * W + 2, cells, "sum")
    box = (int(RAID_BOX[0] * W), int(RAID_BOX[1] * W))
    pts = list(yp.values())
    span = box if cell_spread(img, pts, half, box, cells) <= TEMPLATE_PRIOR * cell_spread(img, pts, half, fit, cells) else fit
    return _rows(img, yp, half, span, cells), dict(cells="raid", fit="fixed" if span == box else "sum",
                                                  span=(span[0] / W, span[1] / W))


# ========================
# Eras
# ========================
def look(img: Image.Image, ts: str) -> dict:
    """The look features era_of reads (the eras pass's): the white share of the frame and of the bottom 30%, the black
    share of the bottom 45%, at 200 px wide."""
    a = np.asarray(img.convert("RGB").resize((200, max(1, round(200 * img.height / img.width)))), np.int16)
    h, w = a.shape[:2]
    b = max(2, w // 40)
    frame = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3), a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])
    white = lambda x: float((x > 225).all(-1).mean())
    return {"w": img.width, "h": img.height, "ts": ts, "frame_white": white(frame), "low_white": white(a[int(h * .7):]),
            "low_black": float((a[int(h * .55):] < 35).all(-1).mean())}


def era_of(rec: dict, span: Optional[tuple], lines=None) -> tuple:
    """
    (era, why) of a card picture by its look (rec: w, h, ts, frame_white, low_white, low_black) and its bar span (a
    fraction pair; None: no stat labels read). Creature eras by where the bars sit; the rest by shape, look and text.
    """
    w, h = rec["w"], rec["h"]
    asp = h / w
    text = " ".join(line[0] for line in lines or []).lower()
    if span:
        x0, x1 = span
        if is_item_size(w, h):
            return "D-item", "item template size, bars"
        if asp < 1.2:
            return ("R", "square, bars") if w >= 2000 else ("sheet", "square, bars")
        if 0.285 <= x0 <= 0.335 and x1 >= 0.955:
            return ("D1", "template box") if w < 900 else ("D2", "template box")
        if 0.25 <= x0 < 0.31 and 0.83 <= x1 <= 0.91:
            return "B", "black panel box"
        if 0.165 <= x0 < 0.25 and x1 >= 0.90:
            return ("A", "white frame") if rec["frame_white"] > 0.45 else ("C", "full art")
        if 1.8 <= asp <= 2.2:
            return ("D1" if w < 900 else "D2", "1:2, template time") if rec["ts"] >= TEMPLATE_FROM else ("C", "1:2, before the template")
        return "bars-other", "bars elsewhere"
    if 0.9 <= asp <= 1.1 and w <= 1300:
        titled = any(line[3] < 0.15 and len(sub(r"[^a-z]", "", line[0].lower())) >= 3 for line in lines or [])
        return ("S", "small square, a title") if titled else ("not-card", "square, no title")
    if 1.4 <= asp <= 1.6 and w >= 1000:
        return "D-item", "item template size"
    if search(r"\bquest\b|costs? \d+(\.\d+)? less per|less per .{0,12} on (your|the) field", text) and asp < 1.75:
        return "F", "field text"
    if rec["frame_white"] > 0.45 and rec["low_white"] > 0.4:
        return "A-item", "white frame"
    if w <= 420 and 1.25 <= asp <= 1.55 and rec["low_black"] > 0.2:
        return "B-item", "small, Arial panel"
    if 1.8 <= asp <= 2.2:
        return ("D2" if w >= 900 else "D1", "1:2, no labels") if rec["ts"] >= TEMPLATE_FROM else ("C", "1:2, no labels")
    if 1.1 <= asp <= 1.65 and rec["ts"] < TEMPLATE_FROM:
        return "C-item", "full-art item"
    return "other", "no rule"


# ========================
# A picture's stats
# ========================
def work_array(img: Image.Image) -> np.ndarray:
    """RGB pixels at WORK long side when smaller (thin cells survive the grid fits)."""
    im = img.convert("RGB")
    s = WORK / max(im.size)
    if s > 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return np.asarray(im)


def _row_of(rows: Optional[dict], stat: str, half: Optional[int]):
    r = rows.get(stat) if rows else None
    return r[half] if (r is not None and half is not None) else r


def confidence(st: dict) -> float:
    """
    How sure one stat's read is (0..1): a split row half by its margin over the runner-up count (full from MARGIN_FULL),
    half by the other readers' agreement (one that can't tell disagrees; AGREE_ALONE when no other reader ran); a
    one-colour row by how clearly it sides with the card's lit or empty cells (nothing under TELL_LOW), times 0.5 to 1
    by agreement.
    """
    if st.get("v") is None:
        return 0.0
    agree = st.get("agree")
    agree = AGREE_ALONE if agree is None else agree
    if st.get("flat"):                                   # the readers tell one-colour rows alike: agreeing proves less
        return round(min(1.0, max(0.0, (st.get("tell", 0.0) - TELL_LOW) / (TELL_FULL - TELL_LOW))) * (0.5 + 0.5 * agree), 3)
    return round(0.5 * min(1.0, max(0.0, st.get("margin", 0.0) / MARGIN_FULL)) + 0.5 * agree, 3)


def flat_holds(stats: dict):
    """One-colour rows hold (never sure) when the card's other rows give no clear lit/empty colours to tell them by:
    under two split rows and none with a FLAT_REF margin (a picture in every cell reads so)."""
    split = [st for st in stats.values() if st.get("v") is not None and not st.get("flat") and st.get("row") != "none"]
    if len(split) >= 2 or any(st.get("margin", 0.0) >= FLAT_REF for st in split):
        return
    for st in stats.values():
        if st.get("flat") and st.get("v") is not None:
            st.setdefault("hold", "one colour, no clear lit/empty rows to tell it by")


def _names(names) -> list:
    return [sub(r"[-_\s]+", " ", n.lower()).strip() for n in ([names] if isinstance(names, str) else names) if n]


def is_twin(names) -> bool:
    """Two creatures' bars in each row, by any of the card's names (OCR'd title, the matcher's card id, old name)."""
    return any(TWINS.search(n) for n in _names(names))


def flags(names, era: Optional[str] = None) -> dict:
    """
    What a card's names (its OCR'd title, the matcher's card id, an old-repo card's own) say about its bars: icon rows,
    three forms, bars drawn as pictures or heavy texture (busy; Paint Warrior and Frog only on the template).
    """
    names = _names(names)
    busy = any(BUSY.fullmatch(n) or (era in TEMPLATE_ERAS and BUSY_TEMPLATE.fullmatch(n)) for n in names)
    return {**({"icons": True} if any(ICONS.search(n) for n in names) else {}),
            **({"multi": True} if any(MULTI.search(n) for n in names) else {}), **({"busy": True} if busy else {})}


def read_card(img: Image.Image, lines, rec: Optional[dict] = None, era: Optional[str] = None, title: str = "",
              card: str = "") -> dict:
    """
    A card picture's era (era_of with rec's look, unless given) and the stats it prints by that era's readers:
    {'era', 'era_how', 'labels', 'reader', 'cells', 'stats': {stat: {v, method, reader, conf, row, ...the row's split}},
    'stats2' (two creatures: their bottom halves), 'readers': every reader's values}. card: the matcher's card id.
    Readers: 'uniform' (label rows, 10 equal cells: eras A-C), 'best' (label rows, the most even grid: D1-D3 and an
    item's labelled rows), 'template' (a 1:2 card's fixed template rows, no OCR), 'raid' / 'raid_fixed' (20 cells).
    A row its label OCR missed ('placed' by the others) holds ('hold': never sure) unless the template reader agrees.
    """
    arr = work_array(img)
    H, W = arr.shape[:2]
    item = is_item_size(*img.size)
    found = labels(lines, strict=item)
    runs, infos = {}, {}
    if len(found) >= 2:
        runs["uniform"], infos["uniform"] = read_uniform(arr, found, only=item)
    out = {}
    if rec is not None:
        rule, why = era_of(rec, infos["uniform"]["span"] if "uniform" in infos else None, lines)
        out.update(era_rule=rule, era_why=why)
        era, how = (era, "given") if era else (rule, "rule")
    else:
        how = "given"
    twin = is_twin((title, card))
    out.update(era=era, era_how=how, labels=len(found), **flags((title, card), era))
    if era == "R":
        runs = {}
        runs["raid"], infos["raid"] = read_raid(arr, found)
        runs["raid_fixed"], infos["raid_fixed"] = read_fixed(arr, "raid")
        main, n = ("raid_fixed" if 0.95 <= W / H <= 1.05 else "raid"), RAID_CELLS     # the template's own box when square
    elif era == "D-item" and len(found) >= 2:            # an item's bars: only the rows it labels
        n, twin = 10, False
        runs["uniform"], infos["uniform"] = read_uniform(arr, found, only=True)
        runs["best"], infos["best"] = read_best(arr, found, only=True)
        main = "best"
    elif era in CREATURE_ERAS or era in ("bars-other", "other"):
        n = 10
        if len(found) >= 2:
            runs["best"], infos["best"] = read_best(arr, found, halves=twin, only=item)
            if twin:
                runs["uniform"], infos["uniform"] = read_uniform(arr, found, halves=True, only=item)
        if 1.8 <= H / W <= 2.2 and era in TEMPLATE_ERAS:
            runs["template"], infos["template"] = read_fixed(arr, "template", halves=twin)
        main = ("uniform" if (era in ("A", "B", "C") or twin) and "uniform" in runs else "best" if "best" in runs
                else "template" if "template" in runs else None)
    else:
        return {**out, "reader": None, "stats": None}
    if main is None:
        return {**out, "reader": None, "stats": None}
    halves = (0, 1) if twin else (None,)
    out.update(reader=main, cells=n, info=infos.get(main), **({"two": True} if twin else {}),
               readers={k: {f"{s}{'' if h in (None, 0) else '2'}": (_row_of(v, s, h) or {}).get("best")
                            for h in halves for s in ORDER} for k, v in runs.items()})
    for key, h in zip(("stats", "stats2"), halves):
        vals = {}
        for s in ORDER:
            if era == "D-item" and s not in found:       # the item prints no such row
                vals[s] = {"v": None, "method": "picture", "reader": main, "row": "none", "conf": 0.0}
                continue
            r = _row_of(runs[main], s, h)
            st = {"v": None if r is None else r.get("best"), "method": "picture", "reader": main}
            if r is not None:
                st.update(score=round(r["score"], 1), margin=round(r["margin"], 1))
                if r["flat"]:
                    st.update(flat=True, tell=round(r.get("tell", 0.0), 3))
            others = [(_row_of(runs[o], s, h) or {}).get("best") for o in runs if o != main]      # None: can't tell
            if others:
                st["agree"] = round(sum(o == st["v"] for o in others) / len(others), 3)
            if main in ("template", "raid_fixed"):
                st["row"] = "template"
            elif s in found:
                st["row"] = "label"
            else:
                st["row"] = "placed"
                if "template" not in runs or (_row_of(runs["template"], s, h) or {}).get("best") != st["v"]:
                    st["hold"] = "row placed by the other labels"
            st["conf"] = confidence(st)
            vals[s] = st
        flat_holds(vals)
        out[key] = vals
    return out


# ========================
# Truth: old-repo PSDs
# ========================
def _cells_of(right: float, x0: float, width: float) -> int:
    """The template cell a bar's right edge closes (0..10)."""
    scale = width / TEMPLATE_CELLS[-1]
    return int(np.argmin([abs(e * scale - (right - x0)) for e in TEMPLATE_CELLS]))


def psd_style(psd) -> tuple:
    """
    (construction, {stat: value} when the structure itself gives it): 'alt' (each row's Light layer slid in a clipped
    row: the alt template, era D3), 'mask' (one '<STAT> Dark' exposure whose mask darkens the empty cells), 'digit'
    (the template's 0-9 rows: the parser's), 'comp' (layer-comp bars), 'partial', 'none' (flattened bars).
    """
    from commands.psd_regions import comp_bars, stat_rows
    alt, mask = {}, {}
    for g in psd.descendants():
        name = g.name.strip().lower()
        if g.is_group() and g.is_visible() and name in ("hp", "def", "atk", "spd"):
            kids = {c.name.strip().lower(): c for c in g}
            clip = kids.get("clipping layer") or kids.get("dark")
            if "light" in kids and clip is not None:
                light = kids["light"]
                alt[name] = _cells_of(light.bbox[2], clip.bbox[0], clip.bbox[2] - clip.bbox[0]) if light.is_visible() else 0
        if (g.kind == "exposure" and g.is_visible() and g.has_mask() and name.endswith("dark")
                and name.split()[0] in ("hp", "def", "atk", "spd")):
            box, scale = g.mask.bbox, psd.width / 1200
            mask[name.split()[0]] = (_cells_of(box[0], 360 * scale, 813 * scale) if box[2] > box[0] else 10)
    if len(alt) >= 4:
        return "alt", alt
    if len(mask) >= 4:
        return "mask", mask
    rows = stat_rows(psd)
    if all(rows.values()):
        return "digit", None
    if any(comp_bars(psd).values()):
        return "comp", None
    return ("partial" if any(rows.values()) else "none"), None


def _truth_task(job: tuple) -> dict:
    png, psd_blob, git_dir = job
    from psd_tools import PSDImage
    data = dm.git_out(git_dir, "cat-file", "blob", psd_blob)
    if data is None:
        return {"png": png, "error": "no blob"}
    from io import BytesIO
    try:
        style, values = psd_style(PSDImage.open(BytesIO(data)))
    except Exception as e:
        return {"png": png, "error": f"{type(e).__name__}: {str(e)[:80]}"}
    return {"png": png, "style": style, "values": values}


def old_pairs(root: str) -> list:
    """Old-repo PNG versions of creatures and minions with the PSD of the same commit: (png row, psd row, its read)."""
    base = join(root, *OLDGIT)
    reads = {}
    for line in open(join(base, "reads.jsonl")):
        r = loads(line)
        reads[r["blob"]] = r
    by = defaultdict(dict)
    for line in open(join(base, "versions.jsonl")):
        v = loads(line)
        by[(v["commit"], v["path"].rsplit(".", 1)[0])][v["ext"]] = v
    out, seen = [], set()
    for (_commit, stem), d in sorted(by.items(), key=lambda kv: kv[1].get("png", {}).get("ts", 0)):
        png = d.get("png")
        if not png or png["blob"] in seen or not stem.startswith(("Creatures/", "Auxillary/Minions/")):
            continue
        seen.add(png["blob"])
        psd = d.get("psd")
        out.append((png, psd, reads.get(psd["blob"]) if psd else None, reads.get(png["blob"])))
    return out


def build_truth(root: str, old_git: str, workers: int):
    """truth/psd.json: each old-repo creature or minion PNG's stats as its same-commit PSD gives them."""
    pairs = old_pairs(root)
    jobs = [(png["blob"], psd["blob"], old_git) for png, psd, _r, _p in pairs if psd]
    styles = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        for i, fut in enumerate(as_completed([pool.submit(_truth_task, j) for j in jobs]), 1):
            r = fut.result()
            styles[r["png"]] = r
            if i % 200 == 0:
                log(f"truth: {i}/{len(jobs)}")
    out = {}
    for png, psd, read, pread in pairs:
        if not psd:
            continue
        size = tuple((pread or {}).get("size") or ())
        s = styles.get(png["blob"], {})
        stats = ((read or {}).get("read") or {}).get("stats") or {}
        kind = ("raid" if size and size[0] >= 2000 else "two" if TWINS.search(png["old_id"].lower()) else "one")
        rec = {"card": png["old_id"], "path": png["path"], "date": png["date"], "psd": psd["blob"], "size": list(size),
               "style": s.get("style"), "kind": kind}
        vals = None
        if kind == "one":
            if s.get("values"):
                vals = s["values"]
            elif s.get("style") in ("digit", "comp") and all(0 <= stats.get(k, -1) <= 10 for k in ORDER):
                vals = {k: stats[k] for k in ORDER}
        rec["stats"] = vals
        rec["era"] = "D3" if s.get("style") == "alt" else ("R" if kind == "raid" else "D2" if size == (1200, 2400) else None)
        out[png["blob"]] = rec
    path = join(root, *OUT, "truth", "psd.json")
    makedirs(join(root, *OUT, "truth"), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        f.write(dumps({"about": "Old-repo creature/minion PNG versions: stats as the same-commit PSD's structure gives "
                                "them (digit rows/comp: today's parser; alt/mask: the rows' layers). None: by eye.",
                       "versions": out}, indent=1))
    replace(path + ".tmp", path)
    log(f"truth: {len(out)} versions, {sum(1 for r in out.values() if r['stats'])} with PSD stats; "
        f"styles {Counter(r['style'] for r in out.values())}")


# ========================
# read
# ========================
def title_of(lines, top: float = 0.13) -> str:
    """The OCR lines starting in a card's top band, in reading order."""
    got = sorted((line for line in lines or [] if line[3] < top), key=lambda line: (round(line[3] * 20), line[2]))
    return " ".join(line[0] for line in got)


def _ocr(img: Image.Image) -> list:
    """Vision lines of a picture at WORK long side (as the eras pass read Discord's)."""
    s = WORK / max(img.size)
    return dm.ocr_lines(img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS) if s > 1 else img)


def read_task(job: dict) -> list:
    """A worker: one Discord picture (its OCR lines given) or one old-repo PNG (and its JPEG copy, OCR'd here)."""
    try:
        if job["src"] == "discord":
            img = dm.load_image(job["path"])
            rec = job["rec"] or look(img, job["row"]["ts"])
            lines = job["lines"] if job["lines"] is not None else _ocr(img)
            got = read_card(img, lines, rec, job.get("era"), title_of(lines), job["row"].get("match_card"))
            if job.get("era"):
                got["era_how"] = job.get("era_how") or "eye"
            return [{**job["row"], **got}]
        img = dm.load_image(dm.open_source(("git", job["git_dir"], job["row"]["blob"])))
        rows = []
        for variant, pic in (("png", img), (f"jpeg{JPEG[0]}", dm.degrade(img, *JPEG))):
            lines = _ocr(pic)
            rows.append({**job["row"], "variant": variant,
                         **read_card(pic, lines, era=job["era"], title=job["row"]["card"])})
        return rows
    except Exception as e:
        return [{**job["row"], "error": f"{type(e).__name__}: {str(e)[:160]}"}]


def match_cards(root: str) -> dict:
    """{picture file: card id} where the matcher is sure of the card (derived/match/verdicts.jsonl; {} without it)."""
    path = join(root, *VERDICTS)
    if not exists(path):
        return {}
    out = {}
    for line in open(path):
        r = loads(line)
        if r.get("verdict") in CARD_SURE and r.get("card") and not r.get("cell"):
            out[r["file"]] = r["card"]
    return out


def discord_jobs(root: str) -> list:
    """Each distinct picture of the card channels (the eras pass's) at its first post, with its by-eye class and the
    matcher's card if any."""
    cards = match_cards(root)
    first = {}
    for line in open(join(root, *ERAS, "discord_feats.jsonl")):
        r = loads(line)
        if r["file"] not in first or r["ts"] < first[r["file"]]["ts"]:
            first[r["file"]] = r
    eye_path = join(root, *OUT, "truth", "eras.json")
    eyes = load(open(eye_path)) if exists(eye_path) else {}
    eye = {**eyes.get("sample", {}), **eyes.get("files", {})}
    jobs = []
    for f, r in sorted(first.items(), key=lambda kv: kv[1]["ts"]):
        keys = ("w", "h", "ts", "frame_white", "low_white", "low_black")
        rec = {k: r[k] for k in keys} if all(k in r for k in keys) else None      # else measured from the picture
        jobs.append({"src": "discord", "path": join(root, f), "lines": r.get("lines"), "rec": rec,
                     "era": eye.get(f, {}).get("era"),
                     "row": {"src": "discord", "file": f, "ts": r["ts"], "ch": r["ch"], "w": r["w"], "h": r["h"],
                             **({"match_card": cards[f]} if f in cards else {})}})
    return jobs


def old_jobs(root: str, old_git: str) -> list:
    """Each old-repo creature or minion PNG version (distinct blobs), its era by its same-commit PSD (truth/psd.json)."""
    truth_path = join(root, *OUT, "truth", "psd.json")
    psd = load(open(truth_path))["versions"] if exists(truth_path) else {}
    jobs = []
    for png, _psd, _read, pread in old_pairs(root):
        size = tuple((pread or {}).get("size") or ())
        t = psd.get(png["blob"], {})
        era = t.get("era") or ("R" if size and size[0] >= 2000 else "D2" if size == (1200, 2400)
                               else "C" if size and 1.8 <= size[1] / size[0] <= 2.2 else None)
        if era is None:
            continue
        jobs.append({"src": "oldgit", "git_dir": old_git, "era": era,
                     "row": {"src": "oldgit", "blob": png["blob"], "card": png["old_id"], "path": png["path"],
                             "commit": png["commit"], "date": png["date"], "w": size[0] if size else None,
                             "h": size[1] if size else None, "era_from": "psd" if t.get("era") else "size"}})
    return jobs


def read_all(root: str, old_git: str, workers: int, only: Optional[str] = None):
    """reads.jsonl: every picture (only: 'discord' or 'oldgit' read again, the other's rows kept)."""
    out_dir = join(root, *OUT)
    makedirs(out_dir, exist_ok=True)
    path = join(out_dir, "reads.jsonl")
    jobs = (discord_jobs(root) if only != "oldgit" else []) + (old_jobs(root, old_git) if only != "discord" else [])
    log(f"read: {len(jobs)} pictures")
    rows = [r for r in map(loads, open(path)) if r["src"] != only] if only and exists(path) else []
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(read_task, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.extend(fut.result())
            if i % 250 == 0 or i == len(jobs):
                log(f"read: {i}/{len(jobs)}")
    rows.sort(key=lambda r: (r["src"], r.get("ts") or r.get("date") or "", r.get("file") or r.get("blob"),
                             r.get("variant", "")))
    with open(path + ".tmp", "w") as f:
        for r in rows:
            f.write(dumps(r) + "\n")
    replace(path + ".tmp", path)
    log(f"read: {len(rows)} rows; eras {Counter(r.get('era') for r in rows)}")


def add_jobs(root: str) -> list:
    """The card-sure Discord pictures (whole, not a sheet's cell) reads.jsonl lacks, at their first post: era and OCR
    lines from the text readers when they read them."""
    done = {r.get("file") for r in map(loads, open(join(root, *OUT, "reads.jsonl"))) if r.get("src") == "discord"}
    eras = {r["file"]: r.get("era") for r in map(loads, open(join(root, *TEXT, "reads.jsonl")))
            if r.get("src") == "discord" and r.get("file")}
    ocr = {}
    for line in open(join(root, *TEXT, "ocr.jsonl")):
        r = loads(line)
        if str(r.get("id", "")).startswith("d:"):
            ocr[r["id"][2:]] = [x[:6] for x in r.get("fix") or []]
    jobs, seen = [], set()
    for line in open(join(root, *VERDICTS)):
        r = loads(line)
        f = r.get("file")
        if r.get("verdict") not in CARD_SURE or r.get("cell") or not f or f in done or f in seen or not r.get("first"):
            continue
        seen.add(f)
        sha = r["sha256"]
        era = eras.get(f) if eras.get(f) not in (None, "other", "not-card") else None
        jobs.append({"src": "discord", "path": join(root, f), "lines": ocr.get(sha), "rec": None, "era": era,
                     "era_how": "text" if era else None,
                     "row": {"src": "discord", "file": f, "ts": r["first"]["date"][:19], "ch": r["first"]["channel"],
                             "w": r.get("w"), "h": r.get("h"), "match_card": r.get("card"), "added": True}})
    return jobs


def add_reads(root: str, workers: int):
    """add: read add_jobs' pictures, mark them sure by calibration.json as it stands, append to reads.jsonl."""
    base = join(root, *OUT)
    jobs = add_jobs(root)
    log(f"add: {len(jobs)} pictures ({sum(1 for j in jobs if j['lines'] is None)} need OCR)")
    cal = load(open(join(base, "calibration.json")))
    sample = cal["classes"]["rule_on_random_sample"]
    rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(read_task, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.extend(fut.result())
            if i % 50 == 0 or i == len(jobs):
                log(f"add: {i}/{len(jobs)}")
    for r in rows:
        bar = cal["groups"].get(group_of(r), {}).get("discord_truth", {}).get("bar")
        for key in ("stats", "stats2"):
            for st in (r.get(key) or {}).values():
                st["sure"] = bar is not None and st["conf"] >= bar and not st.get("hold")
        got = sample.get(r.get("era_rule") if r.get("era_how") == "rule" else None, {})
        r["era_conf"] = round(got["right"] / got["n"], 3) if got.get("n", 0) >= 10 else None
    old = [loads(line) for line in open(join(base, "reads.jsonl"))]
    every = old + rows
    every.sort(key=lambda r: (r["src"], r.get("ts") or r.get("date") or "", r.get("file") or r.get("blob"),
                              r.get("variant", "")))
    with open(join(base, "reads.jsonl.tmp"), "w") as f:
        for r in every:
            f.write(dumps(r) + "\n")
    replace(join(base, "reads.jsonl.tmp"), join(base, "reads.jsonl"))
    log(f"add: {len(rows)} rows; eras {Counter(r.get('era') for r in rows)}; with stats "
        f"{sum(1 for r in rows if r.get('stats'))}")


def stored(git_dir: str, specs: list) -> dict:
    """{'commit:path': blob sha} of the specs whose blob this store holds (no lazy fetch)."""
    data = dm.git_out(git_dir, "cat-file", "--batch-check", data=("\n".join(specs) + "\n").encode()) if specs else None
    out = {}
    for spec, line in zip(specs, (data or b"").decode().splitlines()):
        parts = line.split()
        if len(parts) == 3 and parts[1] == "blob":
            out[spec] = parts[0]
    return out


def version_era(w: int, h: int) -> Optional[str]:
    """A git version's era by its size: the item template, a 1:2 card (D2; D1 under 900 px), the raid square."""
    if is_item_size(w, h):
        return "D-item"
    if 1.9 <= h / w <= 2.1:
        return "D2" if w >= 900 else "D1"
    if 0.95 <= h / w <= 1.05 and w >= 2000:
        return "R"
    return None


def version_jobs(root: str, old_git: str, cards_repo: str) -> list:
    """The card repo's stored PNG versions (the matcher's references) and the old repo's item PNG versions."""
    pairs = dm.read_json(join(root, *PAIRS), {}) or {}
    cards = pairs.get("cards") or {}
    tc = [v for v in pairs.get("versions") or [] if not v.get("old")]
    git_dir = dm.git_dir_of(cards_repo)
    have = stored(git_dir, [f"{v['commit']}:{v['path']}" for v in tc]) if git_dir else {}
    jobs, seen = [], set()
    for v in tc:
        sha = have.get(f"{v['commit']}:{v['path']}")
        if not sha or sha in seen:
            continue
        seen.add(sha)
        jobs.append({"git_dir": git_dir, "blob": sha,
                     "row": {"src": "tc", "blob": sha, "card": v["card"], "path": v["path"], "commit": v["commit"],
                             "date": dm.iso(v["ts"]), "title": (cards.get(v["card"]) or {}).get("name") or v["card"]}})
    for line in open(join(root, *OLDGIT, "versions.jsonl")):
        v = loads(line)
        if v["ext"] != "png" or not v["path"].startswith(("Items/", "Auxillary/Items/")) or v["blob"] in seen:
            continue
        seen.add(v["blob"])
        jobs.append({"git_dir": old_git, "blob": v["blob"],
                     "row": {"src": "oldgit", "blob": v["blob"], "card": v["old_id"], "path": v["path"],
                             "commit": v["commit"], "date": v["date"], "item": True, "title": v["old_id"]}})
    return jobs


def version_task(job: dict) -> list:
    """A worker: one git version, its era by size, read as stored and as its 400 px JPEG."""
    try:
        img = dm.load_image(dm.open_source(("git", job["git_dir"], job["blob"])))
        era = version_era(*img.size)
        base = {**job["row"], "w": img.width, "h": img.height}
        if era is None:
            return [{**base, "era": None, "skipped": "size"}]
        rows = []
        for variant, pic in (("png", img), (f"jpeg{JPEG[0]}", dm.degrade(img, *JPEG))):
            got = read_card(pic, _ocr(pic), era=era, title=job["row"]["title"], card=job["row"]["card"])
            rows.append({**base, "variant": variant, "era_from": "size", **got})
        return rows
    except Exception as e:
        return [{**job["row"], "error": f"{type(e).__name__}: {str(e)[:160]}"}]


def read_versions(root: str, old_git: str, cards_repo: str, workers: int):
    """versions: read version_jobs, mark them sure by calibration.json as it stands, replace their rows."""
    base = join(root, *OUT)
    jobs = version_jobs(root, old_git, cards_repo)
    log(f"versions: {len(jobs)} ({sum(1 for j in jobs if j['row']['src'] == 'tc')} card repo, "
        f"{sum(1 for j in jobs if j['row']['src'] == 'oldgit')} old repo items)")
    cal = load(open(join(base, "calibration.json")))
    rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(version_task, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.extend(fut.result())
            if i % 100 == 0 or i == len(jobs):
                log(f"versions: {i}/{len(jobs)}")
    for r in rows:
        bar = bar_of(cal, r)
        for key in ("stats", "stats2"):
            for st in (r.get(key) or {}).values():
                st["sure"] = bar is not None and st["conf"] >= bar and not st.get("hold")
    keep = [r for r in map(loads, open(join(base, "reads.jsonl")))
            if r.get("src") != "tc" and not (r.get("src") == "oldgit" and r.get("item"))]
    every = keep + rows
    every.sort(key=lambda r: (r["src"], r.get("ts") or r.get("date") or "", r.get("file") or r.get("blob"),
                              r.get("variant", "")))
    with open(join(base, "reads.jsonl.tmp"), "w") as f:
        for r in every:
            f.write(dumps(r) + "\n")
    replace(join(base, "reads.jsonl.tmp"), join(base, "reads.jsonl"))
    log(f"versions: {len(rows)} rows, {sum(1 for r in rows if r.get('error'))} errors; eras "
        f"{Counter(r.get('era') for r in rows)}; with a sure stat "
        f"{sum(1 for r in rows if any((s or {}).get('sure') for s in (r.get('stats') or {}).values()))}")


# ========================
# calibrate
# ========================
def group_of(row: dict) -> Optional[str]:
    """The calibration group of a read: three forms and busy bars (NEVER_SURE), two creatures, icon rows, else its era."""
    return ("multi" if row.get("multi") else "two" if row.get("two") else "busy" if row.get("busy")
            else "icons" if row.get("icons") else row.get("era"))


def row_key(row: dict) -> str:
    return row.get("file") or row.get("blob")


def load_truth(root: str) -> dict:
    """{picture key: {'stats': {stat: v}, 'stats2', 'from'}}: old-repo PSD stats, then by-eye readings (win; stats
    None, e.g. three forms or a card upside down, drops the PSD's: one picture, no single stat set)."""
    base = join(root, *OUT, "truth")
    out = {}
    psd = join(base, "psd.json")
    if exists(psd):
        for blob, t in load(open(psd))["versions"].items():
            if t.get("stats"):
                out[blob] = {"stats": t["stats"], "from": "psd"}
    for name in sorted(n for n in listdir(base) if n.startswith("eye_") and n.endswith(".json")):
        for key, t in load(open(join(base, name)))["items"].items():
            if t.get("stats") is None:
                out.pop(key, None)                           # no truth: the PSD's would publish wrong stats
            else:
                out[key] = {**t, "from": "eye"}
    return out


def _wilson(k: int, n: int, z: float = 1.96) -> float:
    if not n:
        return 0.0
    p = k / n
    return (p + z * z / (2 * n) - z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5) / (1 + z * z / n)


def bar_for(points: list) -> Optional[float]:
    """
    The lowest confidence whose reads at or above it are SURE right, and so are the WINDOW least sure of them (a bar
    the bulk of easy reads can't carry); points: (conf, right). None: too little truth, or no such bar.
    """
    if len(points) < MIN_TRUTH:
        return None
    pts = sorted(points, key=lambda p: -p[0])
    right = np.cumsum([p[1] for p in pts])
    best = None
    for i in range(len(pts)):
        if pts[i][0] <= 0 or (i + 1 < len(pts) and pts[i + 1][0] == pts[i][0]):
            continue                                         # every read at this confidence counted together
        k = min(WINDOW, i + 1)
        low = (right[i] - (right[i - k] if i >= k else 0)) / k
        if right[i] / (i + 1) >= SURE and low >= SURE:
            best = pts[i][0]
    return best


def scored(rows: list, truth: dict) -> list:
    """(group, variant, stat, conf, right, key) of every read stat with a truth."""
    out = []
    for r in rows:
        t = truth.get(row_key(r))
        if not t or not r.get("stats"):
            continue
        for key, tkey in (("stats", "stats"), ("stats2", "stats2")):
            if not r.get(key) or not t.get(tkey):
                continue
            for s in ORDER:
                want, st = t[tkey].get(s), r[key][s]
                if want is None or (want == "none" and st.get("v") is None):
                    continue                                 # no truth; an unprinted row left unread
                conf = 0.0 if st.get("hold") else st.get("conf", 0.0)     # a held row is never sure
                out.append((group_of(r), r.get("variant", "discord"), s, conf, st.get("v") == want, row_key(r)))
    return out


def _at_bar(pts: list, bar: Optional[float]) -> dict:
    """n, accuracy at or above the bar and the share there, of (.., conf, right, ..) points."""
    above = [p for p in pts if bar is not None and p[3] >= bar]
    return {"n": len(pts), "accuracy_at_bar": round(sum(p[4] for p in above) / len(above), 4) if above else None,
            "share_above": round(len(above) / len(pts), 4) if pts else None}


def classes(rows: list, root: str) -> dict:
    """The era rules against the by-eye classes (truth/eras.json), and every Discord picture's class."""
    path = join(root, *OUT, "truth", "eras.json")
    eyes = load(open(path)) if exists(path) else {}
    eye = {**eyes.get("sample", {}), **eyes.get("files", {})}
    disc = [r for r in rows if r["src"] == "discord"]
    per, sample = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for r in disc:
        if r.get("file") in eye and r.get("era_rule"):
            want = eye[r["file"]]["era"]
            per[want][0] += r["era_rule"] == want
            per[want][1] += 1
            if r["file"] in eyes.get("sample", {}):              # random per class: what the rule-classed ones are worth
                sample[r["era_rule"]][0] += r["era_rule"] == want
                sample[r["era_rule"]][1] += 1
    stage1 = defaultdict(Counter)
    for f, e in eye.items():
        stage1[e.get("stage1")][e["era"]] += 1
    return {"pictures": len(disc), "eras": dict(Counter(r.get("era") for r in disc).most_common()),
            "by": dict(Counter(r.get("era_how") for r in disc)),
            "rule_vs_eye": {k: {"n": n, "right": ok, "accuracy": round(ok / n, 3)} for k, (ok, n) in sorted(per.items())},
            "rule_accuracy": round(sum(v[0] for v in per.values()) / max(1, sum(v[1] for v in per.values())), 4),
            "rule_on_random_sample": {k: {"n": n, "right": ok} for k, (ok, n) in sorted(sample.items())},
            "stage1_reclassified": {k: dict(v.most_common()) for k, v in stage1.items() if k in ("other", "B-item", "bars-other")},
            "field_2021_12_to_2023_05": sum(1 for r in disc if r.get("era") == "F" and "2021-12" <= r["ts"][:7] <= "2023-05")}


def bar_of(cal: dict, row: dict) -> Optional[float]:
    """The bar a read's sure marks follow: Discord pictures their own truth's; a card repo version on the template
    (D2/D3 by size) the stricter of both groups' (its construction unknown); else its group's."""
    g = group_of(row)
    if row.get("src") == "discord":
        return cal["groups"].get(g, {}).get("discord_truth", {}).get("bar")
    groups = ("D2", "D3") if row.get("src") == "tc" and g in ("D2", "D3") else (g,)
    bars = [cal["groups"].get(x, {}).get("bar") for x in groups]
    return None if any(b is None for b in bars) else max(bars)


def calibrate(root: str):
    base = join(root, *OUT)
    rows = [loads(line) for line in open(join(base, "reads.jsonl"))]
    for r in rows:                                           # confidence and an old-repo card's flags by today's rules
        if r["src"] == "oldgit":
            for k in ("icons", "multi", "busy"):
                r.pop(k, None)
            r.update(flags(r["card"], r.get("era")))
        for key in ("stats", "stats2"):
            for st in (r.get(key) or {}).values():
                st["conf"] = confidence(st)
            if r.get(key):
                flat_holds(r[key])
    truth = load_truth(root)
    for r in rows:                                           # a picture's truth rides along (eye or psd: use it first)
        t = truth.get(row_key(r))
        r.pop("truth", None)
        if t:
            r["truth"] = {**{k: t[k] for k in ("stats", "stats2") if t.get(k)}, "from": t["from"]}
    pts = scored(rows, truth)
    cal = {"about": f"Per era (two: two-creature cards, icons: icon rows; {' and '.join(NEVER_SURE)}: never sure) the "
                    f"confidence bar whose reads at or above it are at least {SURE:.0%} right on that era's truth, the "
                    f"{WINDOW} least sure of them too; per stat n, accuracy at the bar, share above it; 'discord_truth': "
                    f"the group's Discord pictures alone: their bar (the higher of the group's and their own truth's "
                    f"when old-repo PNGs share the group; it marks Discord reads), n, accuracy and share at it, the Wilson "
                    f"95% lower bound; "
                    f"'discord': the share of every Discord read above it. A row placed by the other labels (hold) is "
                    f"never sure.", "sure": SURE, "groups": {}}
    for g in sorted({p[0] for p in pts if p[0]}):
        gp = [p for p in pts if p[0] == g]
        bar = None if g in NEVER_SURE else bar_for([(p[3], p[4]) for p in gp])
        dp = [p for p in gp if p[1] == "discord"]
        bar_d = bar                                          # Discord reads: their own truth's bar too, when mixed
        if bar is not None and len(dp) < len(gp) and len(dp) >= MIN_TRUTH:
            own = bar_for([(p[3], p[4]) for p in dp])
            bar_d = None if own is None else max(bar, own)
        entry = {"bar": bar, **_at_bar(gp, bar), "accuracy_all": round(sum(p[4] for p in gp) / len(gp), 4),
                 "pictures": len({p[5] for p in gp}), "truth": dict(Counter(truth[p[5]]["from"] for p in gp))}
        above = [p for p in gp if bar is not None and p[3] >= bar]
        entry["wilson_low"] = round(_wilson(sum(p[4] for p in above), len(above)), 4) if above else None
        entry["fields"] = {s: _at_bar([p for p in gp if p[2] == s], bar) for s in ORDER}
        entry["variants"] = {v: _at_bar([p for p in gp if p[1] == v], bar) for v in sorted({p[1] for p in gp})}
        dabove = [p for p in dp if bar_d is not None and p[3] >= bar_d]
        entry["discord_truth"] = {"bar": bar_d, **_at_bar(dp, bar_d), "pictures": len({p[5] for p in dp}),
                                  "wilson_low": round(_wilson(sum(p[4] for p in dabove), len(dabove)), 4) if dabove else None,
                                  "fields": {s: _at_bar([p for p in dp if p[2] == s], bar_d) for s in ORDER}}
        cal["groups"][g] = entry
    reads = defaultdict(lambda: [0, 0])
    for r in rows:
        bar = bar_of(cal, r)
        for key in ("stats", "stats2"):
            for st in (r.get(key) or {}).values():
                st["sure"] = bar is not None and st["conf"] >= bar and not st.get("hold")
                if r["src"] == "discord" and group_of(r) in cal["groups"] and st.get("row") != "none":
                    reads[group_of(r)][0] += st["sure"]
                    reads[group_of(r)][1] += 1
    for g, (k, n) in reads.items():
        cal["groups"][g]["discord"] = {"stats": n, "share_sure": round(k / n, 4)}
    cal["classes"] = classes(rows, root)
    sample = cal["classes"]["rule_on_random_sample"]
    for r in rows:                                           # the class's confidence: by eye, else its rule's sample
        if r["src"] == "discord":
            got = sample.get(r.get("era"), {})
            r["era_conf"] = (1.0 if r.get("era_how") == "eye" else
                             round(got["right"] / got["n"], 3) if got.get("n", 0) >= 10 else None)
    with open(join(base, "calibration.json.tmp"), "w") as f:
        f.write(dumps(cal, indent=1))
    replace(join(base, "calibration.json.tmp"), join(base, "calibration.json"))
    with open(join(base, "reads.jsonl.tmp"), "w") as f:
        for r in rows:
            f.write(dumps(r) + "\n")
    replace(join(base, "reads.jsonl.tmp"), join(base, "reads.jsonl"))
    for g, e in cal["groups"].items():
        log(f"{g}: bar {e['bar']} n {e['n']} acc {e['accuracy_all']} at bar {e['accuracy_at_bar']} above {e['share_above']}")


def main(argv=None):
    ap = ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=("truth", "read", "calibrate", "add", "versions"))
    ap.add_argument("--root", default=dm.DEFAULT_ROOT)
    ap.add_argument("--old-git", default=dm.OLD_GIT)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", choices=("discord", "oldgit"), help="read: these pictures again, the others' rows kept")
    ap.add_argument("--cards-repo", default=dm.LOCAL_DIR_LOC, help="versions: the card repo clone (stored blobs only)")
    a = ap.parse_args(argv)
    root, old_git = expanduser(a.root), expanduser(a.old_git)
    if a.step == "truth":
        build_truth(root, old_git, a.workers)
    elif a.step == "read":
        read_all(root, old_git, a.workers, a.only)
    elif a.step == "add":
        add_reads(root, a.workers)
    elif a.step == "versions":
        read_versions(root, old_git, expanduser(a.cards_repo), a.workers)
    else:
        calibrate(root)


if __name__ == "__main__":
    main()
