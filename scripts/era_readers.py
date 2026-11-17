"""The eras' readers for card pictures without a PSD (timelines stage 1, s1110/timelines/eras/readers.py; owner 11-13):
bars by label-anchored rows, title and ability off whole-card OCR lines. scripts/oldgit_rows.py reads them (its
OLDROWS_READERS setting may name another readers.py's folder)."""
import os
import re
import sys
from difflib import SequenceMatcher

import numpy as np

AC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if AC not in sys.path:
    sys.path.insert(0, AC)
from commands.card_checks import TEMPLATE_CELLS, _lab, read_row  # noqa: E402

LABELS = {'hp': r'health|heaith|healt|hp', 'def': r'defen[cs]e|defemse|defence|deffense|def', 'atk': r'attack|atack|attak|atk', 'spd': r'speed|spead|spee|spd'}
ORDER = ('hp', 'def', 'atk', 'spd')
# template bar (era D): x 360..1173 of 1200, rows 84 tall at y 1200/1326/1452/1578 of 2400
D_X = (360 / 1200, 1173 / 1200)
D_ROWS = (1200, 1326, 1452, 1578)
D_H = 84
UNIFORM = tuple(round(813 * k / 10) for k in range(11))
TEMPLATE_PRIOR = 1.15             # the fixed template box wins within this much of the best fit's cell spread


def labels(lines):
    """{stat: (x0, y0, x1, y1) fractions} of the printed stat labels in whole-card OCR lines."""
    out = {}
    for text, conf, x, y, w, h in lines or []:
        word = re.sub(r'[^a-z]', '', text.lower())
        for stat, pat in LABELS.items():
            if stat not in out and re.fullmatch(pat, word):
                out[stat] = (x, y, x + w, y + h)
    return out


def row_bands(found, height_px):
    """Each found label's row band (y0, y1) in px: centred on the label, ~0.7 of the row pitch tall."""
    ys = {s: (b[1] + b[3]) / 2 for s, b in found.items()}
    order = [s for s in ORDER if s in ys]
    if len(order) >= 2:
        pitch = np.median(np.diff(sorted(ys[s] for s in order))) / max(1, np.median(np.diff([ORDER.index(s) for s in order])))
    else:
        pitch = 1.6 * np.mean([b[3] - b[1] for b in found.values()]) if found else 0
    half = 0.34 * pitch
    return {s: (int((ys[s] - half) * height_px), int((ys[s] + half) * height_px)) for s in order}, pitch


def edges_of(width, cells):
    return [int(round(e * width / cells[-1])) for e in cells]


def cell_colours(row, cells):
    """Mean Lab of each of the 10 cells' middles (60% of width, middle 60% of height)."""
    h, w = row.shape[:2]
    e = edges_of(w, cells)
    out = []
    for k in range(10):
        a, b = e[k], e[k + 1]
        cut = (b - a) * 0.2
        part = row[int(h * 0.2):max(int(h * 0.8), int(h * 0.2) + 1), int(a + cut):max(int(b - cut), int(a + cut) + 1)]
        out.append(_lab(part.reshape(-1, 3).mean(0)))
    return np.array(out)


def split_read(row, cells):
    """
    A two-tone row of 10 cells: the k whose lit (first k) and empty cells are farthest apart against their own spread.
    {'best', 'margin', 'flat'}: flat when no split beats SPLIT (one colour: 0 or 10 by the other rows).
    """
    c = cell_colours(row, cells)
    best, score, second = None, -1e9, -1e9
    for k in range(1, 10):
        a, b = c[:k], c[k:]
        gap = np.linalg.norm(a.mean(0) - b.mean(0))
        spread = max(np.linalg.norm(a - a.mean(0), axis=1).max(), np.linalg.norm(b - b.mean(0), axis=1).max())
        s = gap - 1.5 * spread
        if s > score:
            best, score, second = k, s, score
        elif s > second:
            second = s
    if score < 8:
        return {'best': None, 'flat': True, 'mean': c.mean(0).tolist(), 'score': float(score)}
    return {'best': best, 'flat': False, 'score': float(score), 'lit': c[:best].mean(0).tolist(), 'empty': c[best:].mean(0).tolist()}


def tell_flats(reads):
    """A flat row is 10 when its colour is nearer the card's lit cells than its empty ones, else 0."""
    lit = [r['lit'] for r in reads.values() if r and not r['flat']]
    emp = [r['empty'] for r in reads.values() if r and not r['flat']]
    if not lit:
        return
    L, E = np.mean(lit, 0), np.mean(emp, 0)
    for r in reads.values():
        if r and r['flat']:
            m = np.array(r['mean'])
            r['best'] = 10 if np.linalg.norm(m - L) < np.linalg.norm(m - E) else 0


def fill_rows(found):
    """Labels OCR missed, placed by the others' even spacing (a line through label index vs y): {stat: y centre}."""
    ys = {s: (b[1] + b[3]) / 2 for s, b in found.items()}
    if len(ys) < 2:
        return ys, None
    idx = np.array([ORDER.index(s) for s in ys]); y = np.array(list(ys.values()))
    b, a = np.polyfit(idx, y, 1)
    out = {s: ys.get(s, a + b * ORDER.index(s)) for s in ORDER}
    return out, b


def grid_fit(img, ys, half, x_from, cells=UNIFORM, mode='median'):
    """
    The bars' cell grid shared by the rows: offset x0 and scale whose cell edges (cells, scaled to 10 cells' width)
    sit on the strongest vertical edges summed over the rows' middles. Returns (x0, x1) px.
    """
    H, W = img.shape[:2]
    g = img.astype(np.float32).mean(-1)
    rows = []
    for y in ys:
        y0, y1 = int(y - 0.5 * half), int(y + 0.5 * half)
        y0, y1 = max(0, y0), min(H, max(y1, y0 + 1))
        band = g[y0:y1]
        if band.size:
            p = np.pad(np.abs(band[:, 2:] - band[:, :-2]).mean(0), (1, 1))
            rows.append(np.maximum.reduce([p, np.roll(p, 1), np.roll(p, -1)]))     # +-1 px
    # an edge the rows share (cell lines) beats one row's (a label's letters): the median row
    prof = (np.median(np.array(rows), 0) if mode == 'median' else np.sum(np.array(rows), 0)) if rows else np.zeros(W, np.float32)
    rel = np.array(cells, np.float32) / cells[-1]
    x_from = int(max(0, x_from))
    best, bx = -1, (x_from, W)
    for width in range(int(W * 0.5), int(W * 0.85), max(1, W // 400)):     # every era's bars span 55-80% of the card
        xs = (rel * width).astype(int)
        n = W - 1 - width - x_from
        if n <= 0:
            continue
        s = sum(prof[x_from + k:x_from + k + n] for k in xs)
        i = int(s.argmax())
        if s[i] > best:
            best, bx = float(s[i]), (x_from + i, x_from + i + width)
    return bx


def read_grid(img, found, cells=UNIFORM, how='split', x_from=None, start='min', mode='median'):
    """Label-anchored rows (missing labels filled by spacing) + a fitted cell grid: the four stats."""
    H, W = img.shape[:2]
    ys, pitch = fill_rows(found)
    if pitch is None:
        return {s: None for s in ORDER}, None
    half = abs(pitch) * H * 0.34
    yp = {s: ys[s] * H for s in ORDER}
    if x_from is not None:
        right = x_from * W
    elif start == 'min':
        right = (min(b[2] for b in found.values()) - 0.02) * W     # a big label may run into its bar
    else:
        right = max(b[2] for b in found.values()) * W
    x0, x1 = grid_fit(img, list(yp.values()), half, right + 2, cells, mode)
    reads, rows = {}, []
    for s in ORDER:
        y0, y1 = int(max(0, yp[s] - half)), int(min(H, yp[s] + half))
        row = img[y0:y1, x0:x1]
        if row.size == 0 or y1 - y0 < 3:
            reads[s] = None; continue
        if how == 'step':
            r = read_row(row); rows.append((row, r)); reads[s] = r
        else:
            reads[s] = split_read(row, cells)
    if how == 'step':
        from commands.card_checks import _tell_flats
        _tell_flats(rows)
    else:
        tell_flats(reads)
    return {s: (r['best'] if r else None) for s, r in reads.items()}, (x0 / W, x1 / W)


def read_bars(img, found, xspan, cells=UNIFORM, how='split'):
    """
    The four stats off a picture: rows at the labels (found: labels()), bars over xspan (fractions); how: 'split'
    (two-tone cells) or 'step' (card_checks.read_row's edge steps, template cells).
    """
    H, W = img.shape[:2]
    bands, pitch = row_bands(found, H)
    x0, x1 = int(xspan[0] * W), int(xspan[1] * W)
    reads = {}
    for s in ORDER:
        if s not in bands:
            reads[s] = None
            continue
        y0, y1 = max(0, bands[s][0]), min(H, bands[s][1])
        row = img[y0:y1, x0:x1]
        if row.size == 0:
            reads[s] = None
            continue
        if how == 'step':
            r = read_row(row)
            reads[s] = {'best': r['best'], 'flat': bool(r.get('flat')), 'could': r.get('could'), 'row': row}
        else:
            reads[s] = split_read(row, cells)
    if how == 'step':
        stepped = [(v['row'], v) for v in reads.values() if v]
        from commands.card_checks import _tell_flats
        _tell_flats(stepped)
        for v in reads.values():
            if v:
                v.pop('row', None)
    else:
        tell_flats(reads)
    return {s: (r['best'] if r else None) for s, r in reads.items()}


def template_bars(img, how='step'):
    """Era D's fixed template rows (no OCR)."""
    H, W = img.shape[:2]
    sx, sy = W / 1200, H / 2400
    reads = {}
    rows = []
    for s, y in zip(ORDER, D_ROWS):
        row = img[int(y * sy):int((y + D_H) * sy), int(360 * sx):int(1173 * sx)]
        if how == 'step':
            r = read_row(row); rows.append((row, r)); reads[s] = r
        else:
            reads[s] = split_read(row, TEMPLATE_CELLS)
    if how == 'step':
        from commands.card_checks import _tell_flats
        _tell_flats(rows)
    else:
        tell_flats(reads)
    return {s: r['best'] for s, r in reads.items()}


# ---------- text ----------
STARS = re.compile(r'[*★☆✦✧✩✪✫✬✭✮✯✰⭐]+')


def norm(t):
    return re.sub(r'[^a-z0-9]+', '', (t or '').lower())


def words(t):
    return re.findall(r'[a-z0-9]+', (t or '').lower())


def title_of(lines, top=0.13, labels_y=None):
    """The title off whole-card OCR lines: lines starting in the top band, left to right, star glyphs dropped."""
    got = [l for l in lines or [] if l[3] < top and len(norm(l[0])) >= 1]
    got.sort(key=lambda l: (round(l[3] * 20), l[2]))
    text = ' '.join(STARS.sub(' ', l[0]) for l in got)
    return re.sub(r'\s+', ' ', text).strip()


def ability_of(lines, y_from, y_to=1.0, drop_class=True):
    """The ability text: lines between y_from and y_to (fractions), reading order; the bottom line dropped when it is
    the class line (short, centred, the lowest)."""
    got = [l for l in lines or [] if y_from <= l[3] + l[5] / 2 <= y_to]
    got.sort(key=lambda l: (l[3], l[2]))
    if drop_class and got:
        last = got[-1]
        if len(words(last[0])) <= 4 and abs(last[2] + last[4] / 2 - 0.5) < 0.12 and last[3] > 0.85:
            got = got[:-1]
    return ' '.join(l[0] for l in got)


def strip_tokens(t):
    return re.sub(r'\[[^\]]*\]', ' ', t or '')


def word_score(truth, got):
    """Share of the truth's words OCR recovered in order (SequenceMatcher on word lists)."""
    a, b = words(truth), words(got)
    if not a:
        return None
    m = SequenceMatcher(None, a, b, autojunk=False)
    return sum(x.size for x in m.get_matching_blocks()) / len(a)


def char_ratio(truth, got):
    return SequenceMatcher(None, norm(truth), norm(got), autojunk=False).ratio()


def cell_spread(img, yp, half, span, cells):
    """Mean colour spread (Lab std) inside the cells' middles of every row: low when the grid sits on the cells."""
    x0, x1 = span
    out = []
    for y in yp:
        row = img[int(max(0, y - half)):int(min(img.shape[0], y + half)), x0:x1]
        if row.size == 0 or row.shape[0] < 3:
            continue
        h, w = row.shape[:2]
        e = edges_of(w, cells)
        for k in range(10):
            a, b = e[k], e[k + 1]
            cut = (b - a) * 0.2
            part = row[int(h * 0.2):max(int(h * 0.8), int(h * 0.2) + 1), int(a + cut):max(int(b - cut), int(a + cut) + 1)]
            out.append(float(np.linalg.norm(_lab(part.reshape(-1, 3)).std(0))))
    return float(np.mean(out)) if out else 1e9


def read_best(img, found):
    """
    The era-free bar reader: candidate grids (uniform or template cells; fitted with the label-right start and summed
    rows, or the shortest label's start and the median row; the template's own box on a 1:2 card) and the one whose
    cells are most even inside wins; then the two-tone split. Returns (stats, info).
    """
    H, W = img.shape[:2]
    ys, pitch = fill_rows(found)
    if pitch is None:
        if 1.8 <= H / W <= 2.2:              # labels unread on a 1:2 card: the template's own rows
            return template_bars(img, 'split'), dict(cells='template', fit='fixed rows', span=D_X)
        return {s: None for s in ORDER}, None
    half = abs(pitch) * H * 0.34
    yp = [ys[s] * H for s in ORDER]
    starts = {'sum': max(b[2] for b in found.values()) * W + 2, 'median': (min(b[2] for b in found.values()) - 0.02) * W}
    cands = []
    for cname, cells in (('uniform', UNIFORM), ('template', TEMPLATE_CELLS)):
        for mode, x_from in starts.items():
            cands.append((cname, cells, mode, grid_fit(img, yp, half, x_from, cells, mode)))
    if 1.8 <= H / W <= 2.2:
        cands.append(('template', TEMPLATE_CELLS, 'fixed', (int(D_X[0] * W), int(D_X[1] * W))))
    def flat_rows(span, cells):
        x0, x1 = span
        n = 0
        for y in yp:
            row = img[int(max(0, y - half)):int(min(H, y + half)), x0:x1]
            n += row.size == 0 or row.shape[0] < 3 or split_read(row, cells)['flat']
        return n
    cands = [c for c in cands if flat_rows(c[3], c[1]) < 3] or cands      # a grid on plain background reads no step
    scored = sorted((cell_spread(img, yp, half, span, cells), cname, mode, cells, span) for cname, cells, mode, span in cands)
    fixed = [c for c in scored if c[2] == 'fixed']
    if fixed and fixed[0][0] <= TEMPLATE_PRIOR * scored[0][0]:
        scored = fixed                      # a 1:2 card: the template's own box unless a fit is clearly better
    spread, cname, mode, cells, (x0, x1) = scored[0]
    reads = {}
    for s in ORDER:
        y = ys[s] * H
        row = img[int(max(0, y - half)):int(min(H, y + half)), x0:x1]
        reads[s] = split_read(row, cells) if row.size and row.shape[0] >= 3 else None
    tell_flats(reads)
    return {s: (r['best'] if r else None) for s, r in reads.items()}, dict(cells=cname, fit=mode, span=(x0 / W, x1 / W), spread=round(spread, 2))
