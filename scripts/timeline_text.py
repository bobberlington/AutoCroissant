#!/usr/bin/env python3
"""
Card timelines' text readers (docs/Future/CARD_TIMELINE.md, stage 2 S4): the title, ability, stars and types off card
pictures with no PSD (Discord posts, the old repo's PNGs), each read with its method and confidence; a per-era bar
makes a read "sure" (>= 98% right on that era's truth). By hand, on the Mac (macOS Vision through a small Swift CLI).

    python timeline_text.py icons          Types/ (+ the old repo's old-only types) and Types/Stars icons -> icons/
    python timeline_text.py ocr [-j N]     Vision OCR of every input (whole card twice: language correction on and
                                           off; the title band) -> ocr.jsonl (cached, resumable)
    python timeline_text.py read [-j N]    raw reads (era, title, ability, stars, types + their scores) -> raw.jsonl
                                           (eye files: eye/*.json {id: {title, stars, types, ability}} by hand)
    python timeline_text.py truth          truth.jsonl: old-repo PNGs with their PSD's read, today's cards, eye files
    python timeline_text.py calibrate      calibration.json (per era per field: bar, n, accuracy at the bar, share
                                           above it) and reads.jsonl (each read with method, confidence, sure)
    python timeline_text.py stars [-j N]   the stars read again (stars.jsonl; calibrate takes them over raw.jsonl's)
    python timeline_text.py edges [-j N]   the star reads' doubts (edges.jsonl: the title's words, a star cut by the
                                           right edge; calibrate never takes such a count, nor shurikens, nor
                                           a blob count under its strong copies, as sure)
    python timeline_text.py types [-j N]   the title and the types read again, every bank type's places kept
                                           (types.jsonl; calibrate takes them over raw.jsonl's)
    python timeline_text.py abilities [-j N]   the abilities in raw.jsonl read again from the stored OCR (in place)

Inputs (shared with the bars readers): every card-shaped picture of the catalogue (pictures.jsonl, size classes
CARD_CLASSES) and every PNG version of the old repo (oldgit/versions.jsonl, read only, up to the cut); today's card
PNGs are read for calibration only. Output under ~/Documents/Tools/discord-archive/derived/readers/text (never a repo):
no message text is read or written. Fork Python from AC's scripts/ (cv2, numpy, PIL).
"""
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from difflib import SequenceMatcher
from io import BytesIO
from os.path import basename, dirname, exists, expanduser, join, splitext

HOME = expanduser("~")
HERE = dirname(os.path.abspath(__file__))
AC = dirname(HERE)
if AC not in sys.path:
    sys.path.insert(0, AC)
ROOT = expanduser(os.environ.get("DISCORD_ARCHIVE") or "~/Documents/Tools/discord-archive")   # the archive
OUT = join(ROOT, "derived/readers/text")
from commands.discord_match import FIRST_GIT, first_store  # noqa: E402  (the one store path)
OLD_GIT = expanduser(FIRST_GIT)                  # both old repos' objects (read only; first_store() before a read)
TC = join(HOME, "Desktop/TTSCardMaker")
OLD_ICONS = ("86cd22b2d86586d51464d7cfe4e59ca3dc0fca46", "268c2b12")   # the old repo's first and last Types/ trees
CARD_CLASSES = ("ps1200", "gen400", "gen350", "shot", "cardshape")      # the catalogue's card-shaped pictures
CELL_SURE = ("version_sure", "version_unsure", "draft")                 # sheet cells read: the card sure
SWIFT = join(HERE, "timeline_text_ocr.swift")
OCR_SIDE = 1600                                  # whole card OCR'd at this long side
WORK_W = 360                                     # icons matched on the card at this width
BAND = {"1:2": 0.135, "2:3": 0.2, "1:1": 0.24}   # title band: share of the card's height
ICON_BAND = {"1:2": 0.2, "2:3": 0.25, "1:1": 0.3}   # types (stacks of icons reach lower)
STAR_BAND = {"1:2": 0.28, "2:3": 0.3, "1:1": 0.35}  # stars (stacks of five reach lower still)
BLOB_BAND = {"1:2": 0.11, "2:3": 0.11, "1:1": 0.2}  # yellow star blobs (a status card's 2 x 2 stack reaches 0.2)
ROW_MAX_W = 0.15                                 # a yellow blob this wide and 3/4 as tall (card widths): no row of stars
ROW_FILL = (0.33, 0.8)                           # a star fills this share of its box (eye truths: 1%-99% 0.36-0.83)
LABELS = {"hp": r"health|heaith|healt|hp", "def": r"defen[cs]e|defemse|deffense|def",
          "atk": r"attack|atack|attak|atk", "spd": r"speed|spead|spee|spd"}
STAR_SIZES = [0.045 * 1.14 ** k for k in range(11)]    # star icon width / card width
TYPE_SIZES = [0.05 * 1.13 ** k for k in range(12)]     # type icon width / card width
MIN_STAR = 0.04                                  # a star at least this share of the card's width
COPY_FLOOR, MAX_COPIES = 0.2, 12                 # star copies kept: correlation at least this, at most this many
KNOWN_STARS_ONLY = ("A", "C")                    # eras whose star counts are sure only when a known count agrees
ONE_STAR_UNSURE = KNOWN_STARS_ONLY + ("B-item", "F")   # copies read 1 for overlapped (B-item) or unlike (F) stars
SHURIKENS = ("ninja star",)                      # stars turned and staggered over the art (PoD): copies miss them
EDGE_FROM, EDGE_SEEN = 0.3, 0.5                  # a star cut by the right edge: this share shows; copies this alike seen
LETTER_SIZE = 0.6                                # a copy under this share of a title word's height inside it: a letter
STRONG_COPY, STRONG_BANK = (0.95, 10), 0.9       # a copy this alike (ncc, mad) of a star the bank knows this well: a star
NMS_STAR, NMS_TYPE = 0.45, 0.6                   # peaks nearer than this share of the icon are one
INLINE_FLOOR = 0.6                               # an icon inside the ability's lines
CLASS_Y = 0.96                                   # a creature's class line sits under this (share of height)
NAME_MIN = 0.6                                   # a title this like a name takes the name
BAR_TARGET = 0.985                               # bars are picked at this share right ("sure" >= 98%, with margin)
BAR_TARGETS = (BAR_TARGET, 0.99, 0.995, 0.999)  # stricter fits while the 2-fold cross-check is under SURE_MIN
SURE_MIN = 0.98
EYE_POOL = 50                                    # eye-truth eras with fewer reads also meet the pooled eye bar
EYE_GROUP = {"D2": "inline icons", "D3": "inline icons", "D-item": "inline icons", "F": "inline icons"}
# ... pooled apart: OCR drops their icons and misreads their styled digits (gap round 3: at the 0.81 floor 24 of 33
# right, the plain-text eras 88 of 88)
UNSORTED = ("other", "bars-other")                # pictures no era fits (not all cards): no borrowed bars
MIN_N = 8                                        # an era with less truth borrows its group's bar
STAR_MIN_ABOVE = MIN_N                           # a star bar resting on fewer truths at or above it: none (F had 3)
FIELDS = ("title", "ability", "stars", "types")
TOKEN = re.compile(r"\[([^\]:]+)(?::[^\]]*)?\]")
WORD = re.compile(r"[a-z0-9]+")
ANY_WORD = re.compile(r"[^\W_]+")              # letters of any script kept (OCR's Cyrillic 'А' is not 'A')
CONFUSE = str.maketrans({"0": "o", "1": "l", "i": "l", "5": "s", "|": "l"})


# ========================
# Helpers
# ========================
def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, data):
    os.makedirs(dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    os.replace(path + ".tmp", path)


def jsonl(path):
    if not exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path + ".tmp", "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(path + ".tmp", path)


def git(git_dir, *args, data=None):
    """git output, None on failure: read only (no lazy fetch, fsmonitor off, no pager)."""
    done = subprocess.run(["git", "--git-dir", git_dir, "-c", "core.fsmonitor=false", "--no-pager", *args],
                          input=data, capture_output=True, env={**os.environ, "GIT_NO_LAZY_FETCH": "1"})
    return done.stdout if done.returncode == 0 else None


def norm(text):
    """Lower case letters and digits only (titles)."""
    return "".join(WORD.findall((text or "").lower()))


def loose(text):
    return norm(text).translate(CONFUSE)


def ratio(a, b):
    return SequenceMatcher(None, a, b, autojunk=False).ratio() if a and b else 0.0


def ability_words(text, word=WORD):
    """An ability as words: tokens ([fire]) as '<fire>', the rest lower-case letters and digits."""
    out = []
    for part in re.split(r"(\[[^\]]*\])", text or ""):
        m = TOKEN.fullmatch(part)
        if m:
            out.append("<%s>" % m.group(1).strip().lower())
        else:
            out += word.findall(part.lower())
    return out


def passes_agree(a, b):
    """The two OCR passes give the same words, letters of any script kept: a letter one pass gave in another
    script never drops out of the check."""
    return ability_words(a, ANY_WORD) == ability_words(b, ANY_WORD)


def type_name(stem):
    return stem.strip().lower().replace("_", " ")


# ========================
# Inputs
# ========================
def discord_inputs():
    """The catalogue's card-shaped pictures: one row per stored picture (its first post; reposts share the file); and
    each card-sure sheet cell (the match's verdicts) as a picture of its own: id and file end '#c<n>', cut by its box."""
    rows = {}
    for r in sorted(jsonl(join(ROOT, "derived/catalogue/pictures.jsonl")), key=lambda r: r["at"]):
        if r.get("size_class") in CARD_CLASSES or bar_eras().get(r["file"]) == "F":     # field cards of any size
            sha = splitext(basename(r["file"]))[0]
            rows.setdefault("d:" + sha, {"id": "d:" + sha, "src": "discord", "file": r["file"], "at": r["at"],
                                         "channel": r["channel"], "w": r["w"], "h": r["h"]})
    for v in jsonl(join(ROOT, "derived/match/verdicts.jsonl")):
        if v.get("cell") is not None and v.get("verdict") in CELL_SURE and v.get("first") and v.get("box"):
            cell = f"#c{v['cell']}"
            rows.setdefault("d:" + v["sha256"] + cell, {
                "id": "d:" + v["sha256"] + cell, "src": "discord", "file": v["file"] + cell, "box": v["box"],
                "at": v["first"]["date"], "channel": v["first"]["channel"], "w": v.get("w"), "h": v.get("h")})
    return list(rows.values())


def old_inputs():
    """The old repo's PNG versions up to the cut: one row per blob (its first version)."""
    rows, seen = [], set()
    for v in sorted(jsonl(join(ROOT, "derived/oldgit/versions.jsonl")), key=lambda v: v["ts"]):
        if v.get("ext") != "png" or v["blob"] in seen:
            continue
        seen.add(v["blob"])
        rows.append({"id": "o:" + v["blob"], "src": "old", "blob": v["blob"], "at": v["date"], "ts": v["ts"],
                     "path": v["path"], "old_id": v.get("old_id")})
    return rows


def today_inputs():
    """Today's card PNGs (calibration only): the data branch's cards with their main-tree PNG."""
    from commands.discord_match import load_cards
    rows = []
    for cid, card in sorted(load_cards(TC).items()):
        if card.get("png") and exists(join(TC, card["png"])):
            rows.append({"id": "t:" + cid, "src": "today", "path": card["png"], "key": card.get("key")})
    return rows


def all_inputs(which=("discord", "old", "today")):
    out = []
    if "old" in which:
        out += old_inputs()
    if "today" in which:
        out += today_inputs()
    if "discord" in which:
        out += discord_inputs()
    return out


class Blobs:
    """The old repo's blobs through one cat-file --batch (read only)."""
    def __init__(self):
        self.p = None

    def get(self, sha):
        if self.p is None:
            first_store()                        # a missing store stops: its blobs never read as missing ones
            self.p = subprocess.Popen(["git", "--git-dir", OLD_GIT, "-c", "core.fsmonitor=false", "cat-file",
                                       "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      env={**os.environ, "GIT_NO_LAZY_FETCH": "1"})
        self.p.stdin.write(sha.encode() + b"\n")
        self.p.stdin.flush()
        head = self.p.stdout.readline().split()
        if len(head) < 3 or head[1] != b"blob":
            raise FileNotFoundError(sha)
        data = self.p.stdout.read(int(head[2]))
        self.p.stdout.read(1)
        return data


_BLOBS = Blobs()


def load(item):
    """The item's picture: RGB, borders trimmed (discord_match's own load and trim)."""
    from commands.discord_match import load_image, trim
    if item["src"] == "discord":
        source = join(ROOT, item["file"].split("#")[0])
    elif item["src"] == "old":
        source = BytesIO(_BLOBS.get(item["blob"]))
    else:
        source = join(TC, item["path"])
    img = load_image(source)
    if item.get("box"):                          # a sheet's cell
        w, h = img.size
        x0, y0, x1, y1 = item["box"]
        img = img.crop((round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)))
    return trim(img)


def shape(w, h):
    r = h / w
    return "1:2" if r > 1.75 else "2:3" if r > 1.25 else "1:1"


# ========================
# OCR
# ========================
def ocr_bin():
    path = join(OUT, "bin", "ocr")
    if not exists(path) or os.path.getmtime(path) < os.path.getmtime(SWIFT):
        os.makedirs(dirname(path), exist_ok=True)
        subprocess.run(["swiftc", "-O", SWIFT, "-o", path], check=True)
    return path


class OCR:
    """The Swift Vision CLI as a pipe."""
    def __init__(self):
        self.p = subprocess.Popen([ocr_bin()], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def __call__(self, path, crop=None, scale=1.0, fix=True):
        job = {"path": path, "scale": float(scale), "fix": bool(fix)}
        if crop:
            job["crop"] = [float(c) for c in crop]
        self.p.stdin.write(json.dumps(job) + "\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def rounded(lines):
    return [[l[0], round(l[1], 2), *(round(x, 4) for x in l[2:6]),
             [[w[0], *(round(x, 4) for x in w[1:5])] for w in (l[6] if len(l) > 6 else [])]] for l in lines or []]


_OCR = None


def ocr_task(item):
    """One input's OCR: whole card (fix on, off) at OCR_SIDE and the title band at ~1600 px wide."""
    global _OCR
    if _OCR is None:
        _OCR = OCR()
    try:
        img = load(item)
    except Exception as e:
        return {"id": item["id"], "error": f"{type(e).__name__}: {str(e)[:120]}"}
    from PIL import Image
    w, h = img.size
    s = OCR_SIDE / max(w, h)
    if s != 1:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    fd, tmp = tempfile.mkstemp(suffix=".png", dir=join(OUT, "tmp"))
    os.close(fd)
    try:
        img.save(tmp, "PNG", compress_level=1)
        band = BAND[shape(w, h)]
        out = {"id": item["id"], "w": w, "h": h, "shape": shape(w, h)}
        out["fix"] = rounded(_OCR(tmp, fix=True).get("lines"))
        out["raw"] = rounded(_OCR(tmp, fix=False).get("lines"))
        out["band"] = rounded(_OCR(tmp, crop=(0, 0, 1, band), scale=max(1.0, 1600 / img.size[0]), fix=False).get("lines"))
        return out
    finally:
        os.remove(tmp)


def run_pool(task, items, workers, path, label):
    done = {r["id"] for r in jsonl(path)}
    todo = [i for i in items if i["id"] not in done]
    print(f"{label}: {len(done)} done, {len(todo)} to go", flush=True)
    start, n = time.time(), 0
    with open(path, "a") as f, ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(task, i) for i in todo]
        for fut in as_completed(futs):
            f.write(json.dumps(fut.result()) + "\n")
            f.flush()
            n += 1
            if n % 200 == 0:
                print(f"{label}: {n}/{len(todo)} {time.time() - start:.0f}s", flush=True)


# ========================
# Icons
# ========================
def build_icons():
    """Types/ and Types/Stars icons: today's (TC main's tree) and the old repo's first and last trees (read only)."""
    import hashlib
    from PIL import Image
    seen, names, today = {}, {}, set()
    for kind in ("types", "stars"):
        os.makedirs(join(OUT, "icons", kind), exist_ok=True)

    def keep(kind, stem, data):
        digest = hashlib.sha1(data).hexdigest()[:10]
        names.setdefault(digest, [])
        if stem not in names[digest]:
            names[digest].append(stem)
        if digest in seen:
            return
        try:
            im = Image.open(BytesIO(data)).convert("RGBA")
        except Exception:
            return
        seen[digest] = stem
        im.thumbnail((128, 128), Image.LANCZOS)
        im.save(join(OUT, "icons", kind, f"{stem}~{digest}.png"))
    tc_git = join(TC, ".git")
    first_store()
    for tree, store in [("origin/main", tc_git)] + [(c, OLD_GIT) for c in OLD_ICONS]:
        listing = git(store, "ls-tree", "-r", tree, "--", "Types") or b""
        for line in listing.decode().splitlines():
            meta, path = line.split("\t", 1)
            sha = meta.split()[2]
            stem, ext = splitext(basename(path))
            if ext.lower() not in (".png", ".webp"):
                continue
            data = git(store, "cat-file", "blob", sha)
            if data:
                if store == tc_git:
                    today.add(type_name(stem))
                keep("stars" if "/Stars/" in path else "types", type_name(stem), data)
    same = {}
    for n in names.values():
        if len(n) > 1:                  # byte-identical icons: today's name is the type's name
            main = next((x for x in n if x in today), n[0])
            same[main] = [x for x in n if x != main]
    write_json(join(OUT, "icons", "same.json"), same)
    print(len(os.listdir(join(OUT, "icons", "types"))), "type icons,",
          len(os.listdir(join(OUT, "icons", "stars"))), "star icons")


_BANK = {}


def bank(kind):
    """[(name, rgb float32, mask uint8)] for 'types' or 'stars' (+ eye-cut extras in icons/<kind>-extra)."""
    if kind in _BANK:
        return _BANK[kind]
    from PIL import Image
    import numpy as np
    out = []
    for folder in (kind, kind + "-extra"):
        d = join(OUT, "icons", folder)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            im = Image.open(join(d, f)).convert("RGBA")
            box = im.getchannel("A").getbbox()
            if box:
                im = im.crop(box)
            a = np.asarray(im, np.float32)
            out.append((f.split("~")[0], a[..., :3], (a[..., 3] > 127).astype(np.uint8) * 255))
    _BANK[kind] = out
    return out


_SCALED = {}


def scaled_icon(icon, width):
    """(rgb, mask, coarse): the icon at width px; coarse: transparent pixels at the icon's mean (no weight in a
    plain correlation)."""
    import cv2
    key = (id(icon), int(round(width)))
    if key not in _SCALED:
        name, rgb, mask = icon
        w = max(4, int(round(width)))
        h = max(4, round(rgb.shape[0] * w / rgb.shape[1]))
        t = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0
        g = gray(t)
        coarse = g.copy()
        if m.any():
            coarse[~m] = g[m].mean()
        _SCALED[key] = (g, m, coarse)
        alpha = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)
        _SCALED[key + ("rgb",)] = (t, alpha >= 230)
    return _SCALED[key]


def gray(a):
    """Luminance (float32): icons are matched by shape, since cards tint them."""
    import numpy as np
    a = np.asarray(a, np.float32)
    return a @ np.array([0.299, 0.587, 0.114], np.float32) if a.ndim == 3 else a


def match(region, templ):
    """Plain normalized correlation map (nan-free), None when the template doesn't fit."""
    import cv2
    import numpy as np
    if templ.shape[0] >= region.shape[0] or templ.shape[1] >= region.shape[1] or templ.std() < 1:
        return None
    r = cv2.matchTemplate(region, templ, cv2.TM_CCOEFF_NORMED)
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def verify(region, x, y, t, m):
    """Masked correlation and mean absolute difference (luminance) of the icon at (x, y): (ncc, mad)."""
    import numpy as np
    patch = region[y:y + t.shape[0], x:x + t.shape[1]]
    if patch.shape[:2] != t.shape[:2] or m.sum() < 12:
        return 0.0, 255.0
    p, q = patch[m], t[m]
    mad = float(np.abs(p - q).mean())
    p = p - p.mean()
    q = q - q.mean()
    den = float(np.sqrt((p * p).sum() * (q * q).sum()))
    return (float((p * q).sum() / den) if den > 0 else 0.0), mad


def peaks(score, size, radius, floor, most=12):
    """Local maxima >= floor, at least radius*size apart: [(score, x, y)]."""
    s = score.copy()
    out = []
    r = max(1, int(radius * size))
    while len(out) < most:
        i = int(s.argmax())
        y, x = divmod(i, s.shape[1])
        v = float(s[y, x])
        if v < floor:
            break
        out.append((v, x, y))
        s[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = -1
    return out


def colour_fit(A, x, y, icon, width):
    """The icon's opaque pixels against the card's at (x, y), in colour: (ncc, mad)."""
    import numpy as np
    scaled_icon(icon, width)
    t, m = _SCALED[(id(icon), int(round(width)), "rgb")]
    patch = A[y:y + t.shape[0], x:x + t.shape[1]]
    if patch.shape[:2] != t.shape[:2] or m.sum() < 12:
        return 0.0, 255.0
    p, q = patch[m].astype(np.float32), t[m]
    mad = float(np.abs(p - q).mean())
    p, q = p - p.mean(0), q - q.mean(0)
    den = float(np.sqrt((p * p).sum() * (q * q).sum()))
    return (float((p * q).sum() / den) if den > 0 else 0.0), mad


def icon_score(ncc, mad):
    """One number per hit: the shape's masked correlation (tinted icons keep their shape)."""
    return ncc


def refine(region, icon, x, y, width, steps=(0.92, 0.96, 1.0, 1.04, 1.08), shift=2):
    """The best masked correlation near (x, y) over nearby sizes and shifts: (ncc, mad, x, y, w, h)."""
    best = (-1.0, 255.0, x, y, 0, 0)
    for k in steps:
        t, m, _ = scaled_icon(icon, width * k)
        cx, cy = x + (round(width) - t.shape[1]) // 2, y + (round(width) - t.shape[1]) // 2
        for dy in range(-shift, shift + 1):
            for dx in range(-shift, shift + 1):
                px, py = cx + dx, cy + dy
                if px < 0 or py < 0:
                    continue
                ncc, mad = verify(region, px, py, t, m)
                if ncc > best[0]:
                    best = (ncc, mad, px, py, t.shape[1], t.shape[0])
    return best


def detect(region, kind, sizes, card_w, floor, radius, coarse_floor=0.3, per=3, top=30):
    """Every bank icon at every size over region (luminance): a plain correlation finds places, the masked
    correlation scores them, the best are refined over nearby sizes and shifts. [(score, name, x, y, w, h, ncc,
    mad, icon)] best first, overlaps dropped."""
    region = gray(region)
    cands = []
    for icon in bank(kind):
        for size in sizes:
            t, m, coarse = scaled_icon(icon, size * card_w)
            s = match(region, coarse)
            if s is None:
                continue
            for v, x, y in peaks(s, min(t.shape[:2]), radius, coarse_floor, per):
                ncc, mad = verify(region, x, y, t, m)
                cands.append((ncc, icon, x, y, t.shape[1]))
    cands.sort(key=lambda c: -c[0])
    found = []
    for ncc, icon, x, y, w in cands[:top]:
        ncc, mad, x, y, w, h = refine(region, icon, x, y, w)
        sc = icon_score(ncc, mad)
        if sc >= floor:
            found.append((sc, icon[0], x, y, w, h, ncc, mad, icon))
    found.sort(key=lambda c: -c[0])
    kept = []
    for c in found:
        if all(overlap(c, k) < 0.3 for k in kept):
            kept.append(c)
    return kept


def overlap(a, b):
    ax0, ay0, ax1, ay1 = a[2], a[3], a[2] + a[4], a[3] + a[5]
    bx0, by0, bx1, by1 = b[2], b[3], b[2] + b[4], b[3] + b[5]
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    return iw * ih / max(1, min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0)))


# ========================
# Readers
# ========================
_NAMES = None
_STAGE1 = None
_BAR_ERAS = None
BAR_ERAS = ("A", "B", "C", "D1", "D2", "D-item", "A-item", "B-item", "C-item", "F")   # a bar reader's era over 'other'
F_TOP = 0.4                                      # a field card's (era F) text: every line under this (share of height)


def names_index():
    """Every name a title may print: today's cards (key, website name, printed title) and the old repo's cards and
    titles (old names are matched, never made aliases)."""
    global _NAMES
    if _NAMES is None:
        found = read_json(join(OUT, "names.json"))
        if found is None:
            found = {}
            from commands.discord_match import load_cards
            for card in load_cards(TC).values():
                for n in (card.get("name"), card.get("key")):
                    if n:
                        found.setdefault(n, "live")
            for r in jsonl(join(ROOT, "derived/oldgit/reads.jsonl")):
                t = (r.get("read") or {}).get("title")
                if t:
                    found.setdefault(t, "old")
            for k in (read_json(join(ROOT, "derived/oldgit/map.json"), {}) or {}).get("cards", {}):
                found.setdefault(k, "old")
            write_json(join(OUT, "names.json"), found)
        _NAMES = [(n, loose(n)) for n in sorted(found) if len(loose(n)) >= 2]
    return _NAMES


def stage1_eras():
    global _STAGE1
    if _STAGE1 is None:
        _STAGE1 = {r["file"]: r["era"] for r in jsonl(join(ROOT, "derived/eras/discord_eras.jsonl"))}
    return _STAGE1


def bar_eras():
    """{file: era} of the bar readers' Discord reads (derived/readers/bars/reads.jsonl) in a card era (BAR_ERAS)."""
    global _BAR_ERAS
    if _BAR_ERAS is None:
        _BAR_ERAS = {r["file"]: r["era"] for r in jsonl(join(ROOT, "derived/readers/bars/reads.jsonl"))
                     if r.get("src") == "discord" and r.get("era") in BAR_ERAS}
    return _BAR_ERAS


def stat_labels(lines):
    """{stat: (x0, y0, x1, y1)} of the printed stat labels in whole-card OCR lines (fractions)."""
    out = {}
    for line in lines or []:
        word = re.sub(r"[^a-z]", "", line[0].lower())
        for stat, pat in LABELS.items():
            if stat not in out and re.fullmatch(pat, word):
                out[stat] = (line[2], line[3], line[2] + line[4], line[3] + line[5])
    return out


def look_of(a):
    """Frame/panel shares of a card (the stage-1 classifier's look features)."""
    import cv2
    import numpy as np
    small = cv2.resize(a, (200, max(2, round(200 * a.shape[0] / a.shape[1]))), interpolation=cv2.INTER_AREA)
    small = small.astype(np.int16)
    h, w = small.shape[:2]
    b = max(2, w // 40)
    frame = np.concatenate([small[:b].reshape(-1, 3), small[-b:].reshape(-1, 3), small[:, :b].reshape(-1, 3),
                            small[:, -b:].reshape(-1, 3)])
    white = lambda x: float((x > 225).all(-1).mean())
    black = lambda x: float((x < 35).all(-1).mean())
    return {"frame_white": white(frame), "low_black": black(small[int(h * .55):]), "low_white": white(small[int(h * .7):])}


def era_of(item, w, h, labels, look):
    """The card's template era: stage 1's for the card channels' pictures; else by size (old repo, today) or by the
    stat labels' place and the look (the stage-1 rules); a Discord picture no rule fits takes the bar readers' era."""
    era, how = era_rules(item, w, h, labels, look)
    if item["src"] == "discord" and bar_eras().get(item["file"]) == "F":
        return "F", "bars"                           # a field card (2021-12..2023-05: by eye or its quest text)
    if era in UNSORTED and item["src"] == "discord" and bar_eras().get(item["file"]):
        return bar_eras()[item["file"]], "bars"      # no text era fits: the bar readers' (labels, span, look)
    if era == "bars-other" and w >= 1100 and 1.9 <= h / w <= 2.1:
        return "D3", "size"                          # a 1200 px export whose labels sit off the template's: the alt one
    if era == "bars-other" and w >= 1100 and 1.4 <= h / w <= 1.6:
        return "D-item", "size"
    return era, how


def era_rules(item, w, h, labels, look):
    """era_of's own rules (stage 1's era first)."""
    if item["src"] == "discord":
        known = stage1_eras().get(item["file"])
        if known:
            return known, "stage1"
    asp = h / w
    if item["src"] != "discord":
        if asp < 1.2:
            return ("R" if w >= 2000 else "S"), "size"
        return ("D-item" if asp < 1.75 else "D2"), "size"
    if len(labels) >= 3:
        x1 = max(b[2] for b in labels.values())
        if asp < 1.2:
            return ("R" if w >= 2000 else "bars-other"), "labels"
        if 1.4 <= asp <= 1.6 and 0.23 <= x1 <= 0.265:
            return "B", "labels"
        if 1.9 <= asp <= 2.1 and 0.25 <= x1 <= 0.315:
            return ("D1" if w < 900 else "D2"), "labels"
        if 1.9 <= asp <= 2.1 and 0.16 <= x1 <= 0.225:
            return ("A" if look["frame_white"] > 0.45 else "C"), "labels"
        return "bars-other", "labels"
    if 0.9 <= asp <= 1.1 and w <= 1300:
        return "S", "look"
    if look["frame_white"] > 0.45 and look["low_white"] > 0.4:
        return "A-item", "look"
    if 1.4 <= asp <= 1.6 and w >= 1000:
        return "D-item", "look"
    if 1.9 <= asp <= 2.1 and w >= 1100:
        return "D2", "look"              # the 1200x2400 template, labels unread
    if look["low_black"] > 0.45 and asp < 1.7:
        return "B-item", "look"
    return "other", "look"


STAR_GLYPHS = re.compile(r"[*★☆✦✧✩✪✫✬✭✮✯✰⭐☀]+")


def name_ranks(text, drop=()):
    """Names near a text, best first: [(ratio, name)]; ties go to the longer name. drop: names (norm) left out."""
    lt = loose(text)
    best = {}
    for name, ln in names_index():
        if drop and norm(name) in drop:
            continue
        if abs(len(ln) - len(lt)) > max(len(ln), len(lt)) * 0.6:
            continue
        sm = SequenceMatcher(None, lt, ln, autojunk=False)
        if sm.real_quick_ratio() < 0.5 or sm.quick_ratio() < 0.5:
            continue
        r = sm.ratio()
        key = norm(name)
        if r > best.get(key, (0, None))[0]:
            best[key] = (r, name)
    return sorted(best.values(), key=lambda v: (v[0], len(loose(v[1])), v[1]), reverse=True)


def read_title(band, drop=()):
    """The title off the title band's lines: the tallest lines joined (two-line titles), then the nearest name.
    A single line naming another card makes the read unsure, unless its name sits inside the joined text's name
    (Blooper / Tentacle). {text, name, s1, s2, conf, clash?}."""
    cands = [l for l in band or [] if len(norm(STAR_GLYPHS.sub("", l[0]))) >= 2]
    if not cands:
        return {"text": None, "name": None, "s1": 0.0, "s2": 0.0, "conf": 0.0}
    hmax = max(l[5] for l in cands)
    big = [l for l in cands if l[5] >= 0.6 * hmax]
    big.sort(key=lambda l: (round((l[3] + l[5] / 2) / max(hmax, 1e-3)), l[2]))
    joined = " ".join(STAR_GLYPHS.sub(" ", l[0]).strip() for l in big)
    ranked = name_ranks(joined, drop)
    s1, name = ranked[0] if ranked else (0.0, None)
    s2 = ranked[1][0] if len(ranked) > 1 else 0.0
    conf = s1 - 0.5 * max(0.0, s2 - (s1 - 0.15))
    clash = None
    for l in big if len(big) > 1 else []:
        lr = name_ranks(STAR_GLYPHS.sub(" ", l[0]).strip(), drop)
        if lr and lr[0][0] >= max(NAME_MIN, s1 - 0.05) and norm(lr[0][1]) != norm(name or "") \
                and not (s1 >= NAME_MIN and norm(lr[0][1]) in norm(name)):
            if lr[0][0] > s1 + 0.1:        # the joined text is noise around a name: that name, unsure
                s1, name = lr[0]
            clash = lr[0][1]
            conf = min(conf, 0.5)
            break
    out = {"text": re.sub(r"\s+", " ", joined).strip(), "name": name if s1 >= NAME_MIN else None,
           "s1": round(s1, 4), "s2": round(s2, 4), "conf": round(conf, 4)}
    if clash:
        out["clash"] = clash
    return out


def visual_lines(lines):
    """OCR lines grouped into printed lines (segments sharing a row), left to right: [[segments]]."""
    rows = []
    for l in sorted(lines, key=lambda l: l[3] + l[5] / 2):
        cy = l[3] + l[5] / 2
        for row in rows:
            r0 = row[0]
            if abs((r0[3] + r0[5] / 2) - cy) < 0.5 * min(r0[5], l[5]):
                row.append(l)
                break
        else:
            rows.append([l])
    return [sorted(r, key=lambda l: l[2]) for r in rows]


def latin(text):
    return len("".join(WORD.findall(text.lower()))) >= 2


def class_like(row):
    """A creature's class line: low, centred, short, no digit or final period; never a line with a timing tag,
    2+ lower-case words, or a final ','/';' (unless all capitals: 'ROYALTY/MAGE/HIGHER BEING,')."""
    text = " ".join(l[0] for l in row)
    x0, x1 = min(l[2] for l in row), max(l[2] + l[4] for l in row)
    cy = row[0][3] + row[0][5] / 2
    if cy < CLASS_Y or abs((x0 + x1) / 2 - 0.5) >= 0.13 or len(text.split()) > 5 or re.search(r"[0-9]|\.\s*$", text):
        return False
    lower = sum(1 for w in text.split() if re.fullmatch(r"[^A-Za-z]*[a-z][a-z'’\-]*[^A-Za-z]*", w))
    return not (re.search(r"\([^)]*\)", text) or lower >= 2
                or (re.search(r"[,;]\s*$", text) and re.search(r"[a-z]", text)))


def ability_block(lines, era, labels, band, kind):
    """The ability's printed lines (visual lines) with the class line dropped, and the lines in the block OCR gave
    in another script (a word lost): (rows, class_row, odd). A lone class line is the class (no ability)."""
    lines = [l for l in lines or [] if len(l[0].strip()) > 0]
    top = band + 0.01
    if len(labels) >= 2:
        ys = sorted(((b[1] + b[3]) / 2) for b in labels.values())
        pitch = (ys[-1] - ys[0]) / max(1, len(ys) - 1)
        last = max(labels.values(), key=lambda b: b[3])
        top = max(top, last[3])
        if "spd" not in labels:
            top = max(top, last[3] + 0.6 * pitch)
    elif era in ("D1", "D2", "D3"):
        top = max(top, 0.7)
    below = [l for l in lines if l[3] + l[5] / 2 >= top]
    lines = [l for l in below if latin(l[0])]
    others = [l for l in below if not latin(l[0]) and sum(c.isalpha() for c in l[0]) >= 2]
    rows = visual_lines(lines)
    if not rows:
        return [], None, 0
    if era == "F":                     # a field card: its paragraphs under the art, gaps and icons between
        block = [r for r in rows if min(l[3] + l[5] / 2 for l in r) >= F_TOP]
        if not block:
            return [], None, 0
    elif kind == "card":               # a creature or minion: everything under the bars
        block = rows
    else:                              # items, fields, statuses: the bottom block of evenly spaced lines
        block = [rows[-1]]
        for row in reversed(rows[:-1]):
            prev = block[0]
            hgt, rh = max(l[5] for l in prev), max(l[5] for l in row)
            gap = min(l[3] for l in prev) - max(l[3] + l[5] for l in row)
            if gap > 1.3 * max(hgt, rh) or not 0.5 <= rh / max(hgt, 1e-4) <= 2.0:
                break                  # a short line ('same amount', '3 uses') is lower: the taller row's gap
            block.insert(0, row)
    cls = None
    if kind == "card" and class_like(block[-1]):
        cls, block = block[-1], block[:-1]
    if not block:
        return [], cls, 0
    # other-script lines among the ability's lines or right under them (a class line in capitals excepted)
    hgt = max(l[5] for r in block for l in r)
    end = max(l[3] + l[5] for l in block[-1])
    y0 = top if kind == "card" else F_TOP if era == "F" else min(l[3] for l in block[0]) - 1.3 * hgt
    odd = sum(1 for l in others if y0 <= l[3] + l[5] / 2 <= end + 1.3 * hgt
              and not (kind == "card" and cls is None and l[3] >= end and l[0].isupper() and class_like([l])))
    return block, cls, odd


def ability_text(rows):
    return " ".join(" ".join(l[0] for l in row) for row in rows).strip()


_VOCAB = None


def vocab():
    """Words an ability may print: every word of every known ability (today's, the old repo's). An OCR slip into
    another English word ('DEE', 'O') is not a game word."""
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = {w for v in known_abilities().values() for t, src in v for w in ability_words(t)}
    return _VOCAB


_KNOWN = None


def known_abilities():
    """{loose name: [[ability text, [sources]]]} from today's cards (t:<id>) and the old repo's PSD reads (psd:<blob>)."""
    global _KNOWN
    if _KNOWN is None:
        found = read_json(join(OUT, "known_abilities.json"))
        if found is None:
            acc = defaultdict(dict)
            from commands.discord_match import load_cards
            for cid, card in load_cards(TC).items():
                if card.get("ability"):
                    for n in {card.get("name"), card.get("key")} - {None}:
                        acc[loose(n)].setdefault(card["ability"], []).append("t:" + cid)
            for r in jsonl(join(ROOT, "derived/oldgit/reads.jsonl")):
                rd = r.get("read") or {}
                if r.get("kind") == "psd" and rd.get("ability") and rd.get("title"):
                    acc[loose(rd["title"])].setdefault(rd["ability"], []).append("psd:" + r["blob"])
            found = {k: [[t, sorted(set(src))] for t, src in v.items()] for k, v in acc.items()}
            write_json(join(OUT, "known_abilities.json"), found)
        _KNOWN = found
    return _KNOWN


def ability_known(name, text, exclude=()):
    """True when a known version of the named card prints these words (sources in exclude left out)."""
    ws = ability_words(text)
    return bool(ws) and any(ability_words(t) == ws and not set(src) <= set(exclude)
                            for t, src in known_abilities().get(loose(name or ""), []))


ODD = re.compile(r"[^A-Za-z0-9\s.,:;'\"()/+\-%!?&*#\[\]’‘“”—–]")
TAG = re.compile(r"\(([A-Z0-9/]{2,6})\)")
_TAGS = None


def known_tags():
    """Timing tags printed in known abilities ((OYT), (1CD), ...): an OCR'd tag outside them is a slip."""
    global _TAGS
    if _TAGS is None:
        _TAGS = {m.upper() for v in known_abilities().values() for t, src in v
                 for m in re.findall(r"\(([A-Za-z0-9/]{1,6})\)", t)}
    return _TAGS


def odd_marks(text):
    """Odd characters (accents, OCR's brackets) plus timing tags no known text prints."""
    return len(ODD.findall(text)) + sum(1 for m in TAG.findall(text) if not m.isdigit() and m not in known_tags())


def ability_scores(text, other, title):
    """Confidence parts kept with a read: game-word share, the two OCR passes' agreement, a known text of the card."""
    ws = ability_words(text)
    alpha = [w for w in ws if not w.isdigit() and not w.startswith("<")]
    voc = vocab()
    share = sum(w in voc and (len(w) > 1 or w == "a") for w in alpha) / len(alpha) if alpha else 0.0
    agree = passes_agree(text, other) if other is not None else False
    return {"vocab": round(share, 4), "agree": agree, "known": ability_known((title or {}).get("name"), text),
            "words": len(ws)}


def inline_icons(a, rows, card_w):
    """Type icons inside the ability's lines: [(row index, x fraction, name, score)] (score >= INLINE_FLOOR)."""
    import numpy as np
    H, W = a.shape[:2]
    found = []
    for i, row in enumerate(rows):
        y0 = min(l[3] for l in row)
        y1 = max(l[3] + l[5] for l in row)
        hgt = (y1 - y0) * H
        if hgt < 6:
            continue
        pad = 0.35 * hgt
        ya, yb = int(max(0, y0 * H - pad)), int(min(H, y1 * H + pad))
        region = a[ya:yb].astype(np.float32)
        if region.shape[0] < 8:
            continue
        sizes = [hgt * k / card_w for k in (0.9, 1.1, 1.35)]
        for c in detect(region, "types", sizes, card_w, 0.7, NMS_TYPE):
            cn, cm = colour_fit(region, c[2], c[3], c[8], c[4])
            mix = 0.5 * c[6] + 0.5 * cn - cm / 255.0
            if mix >= INLINE_FLOOR:
                found.append((i, (c[2] + c[4] / 2) / W, same_icons().get(c[1], c[1]), round(mix, 3)))
    return found


def place_icons(rows, icons):
    """The ability text with the icons put among the words by x (a word the icon covers is dropped)."""
    out = []
    for i, row in enumerate(rows):
        words = []
        for l in row:
            ws = l[6] if len(l) > 6 and l[6] else [[l[0], l[2], l[3], l[4], l[5]]]
            words += [(w[1], w[1] + w[3], w[0]) for w in ws]
        mine = sorted((x, n) for r, x, n, s in icons if r == i)
        merged = []
        for x0, x1, t in words:
            if any(x0 <= x <= x1 and len(re.sub(r"[A-Za-z0-9]", "", t)) >= len(t) / 2 for x, n in mine):
                continue
            merged.append((x0, t))
        merged += [(x, "[%s]" % n) for x, n in mine]
        out.append(" ".join(t for _, t in sorted(merged)))
    return " ".join(out).strip()


def rotated(patch, angle):
    """A patch turned about its centre, and its inscribed circle as the mask."""
    import cv2
    import numpy as np
    h, w = patch.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    turned = cv2.warpAffine(patch, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    yy, xx = np.mgrid[:h, :w]
    circle = ((xx - w / 2 + 0.5) ** 2 + (yy - h / 2 + 0.5) ** 2 <= (min(w, h) / 2) ** 2).astype(np.uint8) * 255
    return turned, circle


def read_stars(a, band_h, title_box):
    """Stars: the bank's best star icon right of the title, then that very star cut from the card (a circle, turned
    in 10 degree steps: cards turn their stars) matched over the band; its copies' scores and colour differences
    are kept, the count is decided in calibrate (count_stars)."""
    import cv2
    import numpy as np
    H, W = a.shape[:2]
    x_from = star_x_from(W, title_box)
    region = a[:max(8, int(band_h * H)), x_from:].astype(np.float32)
    if region.shape[1] < 16:
        return {"icon": None}
    cands = detect(region, "stars", STAR_SIZES, W, 0.0, NMS_STAR)
    if not cands:
        return {"icon": None}
    v, name, x, y, tw, th, ncc, mad = cands[0][:8]
    g = gray(region)
    side = min(tw, th)
    cx, cy = x + tw // 2, y + th // 2
    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
    templ = g[y0:y0 + side, x0:x0 + side]
    colour = region[y0:y0 + side, x0:x0 + side]
    if templ.shape[0] < 6 or templ.shape[1] < 6 or templ.shape[0] >= g.shape[0] or templ.shape[1] >= g.shape[1]:
        return {"icon": None}
    best = None
    for angle in range(0, 90, 10):
        t, m = rotated(templ, angle)
        if t.std() < 1:
            continue
        r = np.nan_to_num(cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED, mask=m), nan=0.0, posinf=0.0, neginf=0.0)
        best = r if best is None else np.maximum(best, r)
    if best is None:
        return {"icon": None}
    _, circle = rotated(templ, 0)
    inside = circle > 0
    ref = colour[inside].mean(0)
    copies = []
    for c, px, py in peaks(best, side, 0.4, COPY_FLOOR, MAX_COPIES):
        patch = region[py:py + side, px:px + side]
        diff = float(np.abs(patch[inside].mean(0) - ref).mean()) if patch.shape[:2] == inside.shape else 255.0
        copies.append([round(c, 3), round(diff, 1), px, py])
    return {"icon": name, "bank": round(v, 3), "ncc": round(ncc, 3), "mad": round(mad, 1),
            "size": round(side / W, 3), "copies": copies}


def star_x_from(W, title_box):
    """Where read_stars' region starts (px): right of the title, a little into its end."""
    x_from = int(W * 0.45)
    if title_box:
        x_from = max(int(W * 0.3), min(x_from, int(title_box[2] * W) - int(0.03 * W)))
    return x_from


def title_words(rec, title, W, H, shp):
    """The title's words of 3+ letters as boxes on the work image (px): a star copy inside one is maybe a letter."""
    bh = BAND[shp] * H
    return [[round(w[1] * W, 1), round(w[2] * bh, 1), round((w[1] + w[3]) * W, 1), round((w[2] + w[4]) * bh, 1)]
            for l in rec.get("band") or [] if title["text"] and l[0].strip() and l[0].strip() in title["text"]
            for w in (l[6] if len(l) > 6 else []) if len(re.findall(r"[A-Za-z]", w[0])) >= 3]


def read_edge(a, st, band_h, x_from):
    """Stars running off the right edge: read_stars' template (at its own first copy), turned in 10 degree steps, its
    left part (EDGE_FROM of it or more) matched against the region's last columns at each copy's row, away from the
    copies seen (>= EDGE_SEEN). [[ncc, colour difference, share shown, y, x], ...] best first, one per row."""
    import cv2
    import numpy as np
    copies = (st or {}).get("copies") or []
    if not copies or copies[0][0] < 0.999:
        return []
    H, W = a.shape[:2]
    region = a[:max(8, int(band_h * H)), x_from:].astype(np.float32)
    g = gray(region)
    side = int(round(st["size"] * W))
    x0, y0 = copies[0][2], copies[0][3]
    templ, colour = g[y0:y0 + side, x0:x0 + side], region[y0:y0 + side, x0:x0 + side]
    if side < 6 or templ.shape != (side, side):
        return []
    _, circle = rotated(templ, 0)
    inside = circle > 0
    ref = colour[inside].mean(0)
    turns = [rotated(templ, angle) for angle in range(0, 90, 10)]
    wr, hr = g.shape[1], g.shape[0]
    seen = [c for c in copies if c[0] >= EDGE_SEEN]
    rows, out = [], []
    for c in copies:
        if all(abs(c[3] - y) > side / 2 for y in rows):
            rows.append(c[3])
    for y in rows:
        lo, hi = max(0, y - 2), min(hr - side, y + 2)
        if hi < lo:
            continue
        best = None
        for v in range(int(np.ceil(EDGE_FROM * side)), side):
            x = wr - v
            if x < 0 or any(abs(s[2] - x) < side / 2 and abs(s[3] - y) < side / 2 for s in seen):
                continue
            strip = np.ascontiguousarray(g[lo:hi + side, x:wr])
            for t, m in turns:
                tp, mp = np.ascontiguousarray(t[:, :v]), np.ascontiguousarray(m[:, :v])
                if tp.std() < 1 or not mp.any():
                    continue
                r = np.nan_to_num(cv2.matchTemplate(strip, tp, cv2.TM_CCOEFF_NORMED, mask=mp), nan=0.0, posinf=0.0,
                                  neginf=0.0)
                k = int(r.argmax())
                if best is None or r.flat[k] > best[0]:
                    yy = lo + k // r.shape[1]
                    patch = region[yy:yy + side, x:wr]
                    diff = float(np.abs(patch[inside[:, :v]].mean(0) - ref).mean())
                    best = (float(r.flat[k]), diff, v / side, yy, x)
        if best:
            out.append([round(best[0], 3), round(best[1], 1), round(best[2], 2), best[3], best[4]])
    return sorted(out, key=lambda e: -e[0])


def strong_copies(rd):
    """The stars a blob count can't overlook: STRONG_COPY copies, off the title's words, not in a column of three
    (a frame's edge), one per star (centres 3/4 of a star apart), at most two stars below the top one."""
    st = rd.get("stars") or {}
    if (st.get("bank") or 0) < STRONG_BANK:
        return []
    geo = rd.get("star_edge") or {}
    side, x_from = (st.get("size") or 0) * WORK_W, geo.get("x_from") or 0
    good = [c for c in st.get("copies") or [] if c[0] >= STRONG_COPY[0] and c[1] <= STRONG_COPY[1]]
    kept = []
    for c in good:
        x, y = c[2], c[3]
        if sum(abs(x - d[2]) <= 2 for d in good) >= 3 or any(
                x0 - 1 <= x_from + x and x_from + x + side <= x1 + 1 and y0 - 1 <= y and y + side <= y1 + 1
                for x0, y0, x1, y1 in geo.get("words") or []):
            continue
        if all(math.hypot(x - a, y - b) >= 0.75 * side for a, b in kept):
            kept.append((x, y))
    top = min((b for _, b in kept), default=0)
    return [k for k in kept if k[1] <= top + 2 * side]


def star_doubt(rd, rule, n=None):
    """Why an icon star count can't be sure: SHURIKENS (turned and staggered over the art: the copies aren't the
    stars), a counted copy on a title letter (an 'o' taken for a star), or a star cut by the picture's right edge in
    a counted copy's row (one more may print); a blob count n under its strong copies (a star in a second row,
    Spice). None: no doubt."""
    st = rd.get("stars") or {}
    if rule.get("kind"):
        return "more stars copied than counted" if n is not None and len(strong_copies(rd)) > n else None
    if st.get("icon") in SHURIKENS:
        return "shuriken stars"
    geo = rd.get("star_edge") or {}
    side = (st.get("size") or 0) * WORK_W
    good = [c for c in st.get("copies") or [] if c[0] >= rule["ncc"] and c[1] <= rule["mad"]]
    x_from = geo.get("x_from") or 0
    for x0, y0, x1, y1 in geo.get("words") or []:
        if side < LETTER_SIZE * (y1 - y0) and any(x0 - 1 <= x_from + c[2] and x_from + c[2] + side <= x1 + 1 and
                                                  y0 - 1 <= c[3] and c[3] + side <= y1 + 1 for c in good):
            return "a title letter counted"
    for ncc, mad, shown, y, x in geo.get("edge") or []:
        if ncc >= rule["ncc"] and mad <= rule["mad"] and any(abs(c[3] - y) <= side / 2 for c in good):
            return "a star cut by the edge"
    return None


def read_star_blobs(a, title_box, band=0.11):
    """Yellow and gold stars (eras A and B: suns in a row, an overlapped cluster) as colour blobs right of the title:
    {blobs: [[area, x, y, w, h]] largest first, cluster: [w, h] of the largest} (fractions of the card's width)."""
    import cv2
    import numpy as np
    H, W = a.shape[:2]
    x_from = int(W * 0.55)
    if title_box:
        x_from = max(int(W * 0.35), min(x_from, int(title_box[2] * W)))
    region = np.ascontiguousarray(a[:max(8, int(band * H)), x_from:])
    hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 0] >= 10) & (hsv[..., 0] <= 40) & (hsv[..., 1] >= 90) & (hsv[..., 2] >= 140)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    blobs = sorted(([int(st[4]), round(st[0] / W, 4), round(st[1] / W, 4), round(st[2] / W, 4), round(st[3] / W, 4)]
                    for st in stats[1:] if st[4] >= 6), reverse=True)[:12]
    return {"blobs": blobs, "cluster": blobs[0][3:5] if blobs else None}


def count_blobs(sb, rule):
    """(n, conf) from read_star_blobs. 'row': the largest group of like blobs (area within rule['tol'], squarish,
    filling their box as a star does, one row); sure when no other sizeable blob shares the row; none behind a
    yellow wall. 'cluster': an overlapped cluster's width over its
    height, each star after the first adding rule['step'] (the cluster must fill its box like stars do)."""
    if not sb or not sb.get("blobs"):
        return None, 0.0
    blobs = [b for b in sb["blobs"] if b[0] >= 12]
    if rule["kind"] == "row" and any(b[3] > ROW_MAX_W and b[4] > 0.75 * ROW_MAX_W for b in blobs):
        return None, 0.0                    # a yellow backdrop (a status card's wall): stars merge into it
    if rule["kind"] == "row":
        best = []
        stars = [b for b in blobs if ROW_FILL[0] <= b[0] / max(1.0, b[3] * WORK_W * b[4] * WORK_W) <= ROW_FILL[1]]
        for b in stars:
            cy = b[2] + b[4] / 2
            group = [c for c in stars if 1 / rule["tol"] <= c[0] / b[0] <= rule["tol"]
                     and 0.6 <= c[3] / max(c[4], 1e-4) <= 1.6 and abs(c[2] + c[4] / 2 - cy) < rule.get("rows", 0.6) * b[4]]
            if len(group) > len(best):
                best = group
        if not best:
            return None, 0.0
        mean = sum(c[0] for c in best) / len(best)
        cy = sum(c[2] + c[4] / 2 for c in best) / len(best)
        h = max(c[4] for c in best)
        others = [c for c in blobs if c not in best and c[0] >= 0.4 * mean and abs(c[2] + c[4] / 2 - cy) < h]
        return len(best), (1.0 if not others else 0.3)
    big = blobs[0] if blobs else None
    if not big or big[4] <= 0:
        return None, 0.0
    fill = big[0] / max(1.0, big[3] * WORK_W * big[4] * WORK_W)
    x = (big[3] / big[4] - rule.get("base", 1.05)) / rule["step"] + 1
    n = int(round(x))
    if n < 1 or not 0.3 <= fill <= 0.8:
        return None, 0.0
    return n, round(max(0.0, 1 - 2 * abs(x - n)), 4)


def count_stars(st, rule):
    """(n, conf) from read_stars' copies under a rule {ncc, mad}: copies at least that alike (correlation, mean
    colour difference) are stars; the confidence is the margin between the weakest star and the strongest
    non-star. A star under MIN_STAR of the card's width is no read."""
    if not st or not st.get("icon") or st.get("bank", 0) < rule.get("bank", 0.0) or st.get("size", 0) < MIN_STAR:
        return None, 0.0
    good, bad = [], []
    for c, mad, x, y in st["copies"]:
        (good if c >= rule["ncc"] and mad <= rule["mad"] else bad).append(c - mad / 160.0)
    if not good or (not bad and len(good) >= MAX_COPIES):      # every copy kept is a star: can't count
        return None, 0.0
    margin = min(good) - (max(bad) if bad else COPY_FLOOR)     # no copy kept but stars: the next is under the floor
    return len(good), round(min(1.0, st["bank"] + 0.2) * max(0.0, min(1.0, 0.5 + margin)), 4)


_KNOWN_TYPES = None
_SAME = None


def same_icons():
    """{name: canonical name} for type icons whose files are byte-identical (undread = undead, ...)."""
    global _SAME
    if _SAME is None:
        _SAME = {o: k for k, v in (read_json(join(OUT, "icons", "same.json"), {}) or {}).items() for o in v}
    return _SAME


def canon_types(types):
    m = same_icons()
    return sorted(m.get(type_name(t), type_name(t)) for t in types or [])


def known_types():
    """{loose name: [[types, [sources]]]}: each card's type sets in today's data and the old repo's PSD reads."""
    global _KNOWN_TYPES
    if _KNOWN_TYPES is None:
        found = read_json(join(OUT, "known_types.json"))
        if found is None:
            acc = defaultdict(dict)
            stats = json.loads(git(join(TC, ".git"), "show", "origin/data:db/stats.json") or b"{}").get("cards") or {}
            from commands.discord_match import load_cards
            for cid, card in load_cards(TC).items():
                st = stats.get(card.get("key")) or {}
                if st.get("types") is None:
                    continue
                key = json.dumps(canon_types(st["types"]))
                for n in {card.get("name"), card.get("key")} - {None}:
                    acc[loose(n)].setdefault(key, []).append("t:" + cid)
            for r in jsonl(join(ROOT, "derived/oldgit/reads.jsonl")):
                rd = r.get("read") or {}
                if r.get("kind") == "psd" and rd.get("title") and rd.get("types") is not None:
                    acc[loose(rd["title"])].setdefault(json.dumps(canon_types(rd["types"])), []).append(
                        "psd:" + r["blob"])
            found = {k: [[json.loads(t), sorted(set(src))] for t, src in v.items()] for k, v in acc.items()}
            write_json(join(OUT, "known_types.json"), found)
        _KNOWN_TYPES = found
    return _KNOWN_TYPES


_KNOWN_STARS = None


def known_stars():
    """{loose name: [[stars, [sources]]]}: each card's star counts in today's data and the old repo's PSD reads."""
    global _KNOWN_STARS
    if _KNOWN_STARS is None:
        found = read_json(join(OUT, "known_stars.json"))
        if found is None:
            acc = defaultdict(dict)
            stats = json.loads(git(join(TC, ".git"), "show", "origin/data:db/stats.json") or b"{}").get("cards") or {}
            from commands.discord_match import load_cards
            for cid, card in load_cards(TC).items():
                st = stats.get(card.get("key")) or {}
                if isinstance(st.get("stars"), int):
                    for n in {card.get("name"), card.get("key")} - {None}:
                        acc[loose(n)].setdefault(str(st["stars"]), []).append("t:" + cid)
            for r in jsonl(join(ROOT, "derived/oldgit/reads.jsonl")):
                rd = r.get("read") or {}
                if r.get("kind") == "psd" and rd.get("title") and isinstance(rd.get("stars"), int):
                    acc[loose(rd["title"])].setdefault(str(rd["stars"]), []).append("psd:" + r["blob"])
            found = {k: [[int(n), sorted(set(src))] for n, src in v.items()] for k, v in acc.items()}
            write_json(join(OUT, "known_stars.json"), found)
        _KNOWN_STARS = found
    return _KNOWN_STARS


def stars_known(name, n, exclude=()):
    """True when a known version of the named card has n stars (sources in exclude left out)."""
    return any(c == n and not set(src) <= set(exclude) for c, src in known_stars().get(loose(name or ""), []))


def type_presence(a, band_h, names, top=3):
    """Each named type's best places in the icon band (left 60%), apart from each other: {name: [[gray ncc, colour
    mix, x, y, w], ...]} best first, at most top."""
    H, W = a.shape[:2]
    A = a[:max(8, int(band_h * H))].astype("float32")[:, :int(0.6 * W)]
    G = gray(A)
    canon = same_icons()
    out = {}
    for name in names:
        icons = [i for i in bank("types") if canon.get(i[0], i[0]) == name]
        cands = []
        for icon in icons:
            for size in TYPE_SIZES:
                t, m, coarse = scaled_icon(icon, size * W)
                s = match(G, coarse)
                if s is None:
                    continue
                for v, x, y in peaks(s, min(t.shape[:2]), NMS_TYPE, 0.0, 3):
                    ncc, mad = verify(G, x, y, t, m)
                    cands.append((ncc, icon, x, y, t.shape[1]))
        cands.sort(key=lambda c: -c[0])
        places = []
        for ncc, icon, x, y, w in cands[:6 + 2 * top]:
            r = refine(G, icon, x, y, w)
            cn, cm = colour_fit(A, r[2], r[3], icon, r[4])
            places.append([round(r[0], 3), round(0.5 * r[0] + 0.5 * cn - cm / 255.0, 3), r[2], r[3], r[4]])
        places.sort(key=lambda p: -p[0])
        kept = []
        for p in places:
            if all(place_overlap(p, k) < 0.3 for k in kept):
                kept.append(p)
        out[name] = kept[:top] or [[0.0, 0.0, 0, 0, 0]]
    return out


def place_overlap(p, q):
    """Overlap of two icon places [score, mix, x, y, w] (square boxes), as a share of the smaller."""
    return overlap((0, 0, p[2], p[3], p[4], p[4]), (0, 0, q[2], q[3], q[4], q[4]))


def places_of(v):
    """A presence entry as a list of places (old reads kept one flat place)."""
    return v if v and isinstance(v[0], list) else [v]


def verified_types(rd, rule, exclude=(), alarms=None):
    """(types, conf) from the card's known type sets checked on the picture (types_known): the one set whose icons
    are all present (gray ncc >= rule['on']) while the other known icons are absent (< rule['off']). An empty set
    is never offered (absence proves nothing). alarms {type: level}: any other bank icon at or over its level (and
    rule['on']) away from the set's icons makes the read unsure (conf 0): a Discord version with an icon git never
    had."""
    tk = rd.get("types_known")
    if not tk or not tk.get("sets"):
        return None, 0.0
    card = rd.get("kind") == "card"
    pres = {}
    for t, v in tk["presence"].items():
        c = canon_types([t])[0]
        v = places_of(v)
        if c not in pres or v[0][0] > pres[c][0][0]:
            pres[c] = v
    sets = [sorted(t for t in canon_types(ts) if not (card and t == "equip")) for ts, src in tk["sets"]
            if not set(src) <= set(exclude)]
    sets = [json.loads(x) for x in dict.fromkeys(json.dumps(s) for s in sets) if x != "[]"]   # no icon: unprovable
    if not sets:
        return None, 0.0
    union = sorted({t for s in sets for t in s})
    fits = []
    for ts in sets:
        on = [pres[t][0][0] for t in ts if t in pres]
        off = [pres[t][0][0] for t in union if t not in ts and t in pres]
        if len(on) == len(ts) and all(v >= rule["on"] for v in on) and all(v < rule["off"] for v in off):
            fits.append((min([v - rule["on"] for v in on] + [rule["off"] - v for v in off] + [0.3]), ts))
    if len(fits) != 1:
        return None, 0.0
    conf, ts = fits[0]
    if alarms is not None and extra_icons(pres, ts, rule, alarms, card):
        return ts, 0.0
    return ts, round(conf, 4)


def extra_icons(pres, ts, rule, alarms, card):
    """Bank types other than the set's at or over their alarm level away from the set's icons."""
    mine = [pres[t][0] for t in ts if t in pres]
    out = []
    for t, places in pres.items():
        if t in ts or (card and t == "equip") or t not in alarms:
            continue
        level = max(alarms[t], rule["on"])
        if any(p[0] >= level and all(place_overlap(p, m) < 0.3 for m in mine) for p in places):
            out.append(t)
    return out


def title_box(rec, title, band):
    """The title's box in the band (x0, 0, x1, band) from the band OCR's lines that make the title, or None."""
    big = [l for l in rec.get("band") or [] if title["text"] and l[0].strip() and l[0].strip() in title["text"]]
    return (min(l[2] for l in big), 0, max(l[2] + l[4] for l in big), band) if big else None


def star_task(args):
    """Stars only (a second pass with a better star reader)."""
    import cv2
    import numpy as np
    item, rec = args
    try:
        img = load(item)
        a = np.asarray(img.convert("RGB"))
        W0, H0 = img.size
        work = cv2.resize(a, (WORK_W, max(2, round(WORK_W * H0 / W0))), interpolation=cv2.INTER_AREA)
        shp = rec.get("shape") or shape(W0, H0)
        tb = title_box(rec, read_title(rec.get("band")), BAND[shp])
        return {"id": item["id"], "stars": read_stars(work, STAR_BAND[shp], tb),
                "star_blobs": read_star_blobs(work, tb, BLOB_BAND[shp])}
    except Exception as e:
        return {"id": item["id"], "error": f"{type(e).__name__}: {str(e)[:160]}"}


_STARS = None


def edge_task(args):
    """The star read's doubts (edges.jsonl; calibrate takes them): read_stars' region start, the title's words and a
    star cut by the right edge (read_edge), from stars.jsonl's read of the same picture."""
    import cv2
    import numpy as np
    global _STARS
    item, rec = args
    try:
        if _STARS is None:
            _STARS = {r["id"]: r.get("stars") for r in jsonl(join(OUT, "stars.jsonl")) if not r.get("error")}
        st = _STARS.get(item["id"]) or {}
        if not st.get("icon"):
            return {"id": item["id"], "edge": []}
        img = load(item)
        a = np.asarray(img.convert("RGB"))
        W0, H0 = img.size
        work = cv2.resize(a, (WORK_W, max(2, round(WORK_W * H0 / W0))), interpolation=cv2.INTER_AREA)
        shp = rec.get("shape") or shape(W0, H0)
        title = read_title(rec.get("band"))
        x_from = star_x_from(WORK_W, title_box(rec, title, BAND[shp]))
        return {"id": item["id"], "x_from": x_from, "words": title_words(rec, title, WORK_W, work.shape[0], shp),
                "edge": read_edge(work, st, STAR_BAND[shp], x_from)}
    except Exception as e:
        return {"id": item["id"], "error": f"{type(e).__name__}: {str(e)[:160]}"}


def read_one(item, rec, parts=None):
    """Every text field of one input from its OCR record and its picture (parts: only the title, era and these of
    'ability', 'stars', 'blobs', 'types'; the preview check reads only what was published: types' presence is over
    half the time, the stars a quarter)."""
    want = lambda part: parts is None or part in parts
    import numpy as np
    import cv2
    img = load(item)
    a = np.asarray(img.convert("RGB"))
    W0, H0 = img.size
    work = cv2.resize(a, (WORK_W, max(2, round(WORK_W * H0 / W0))), interpolation=cv2.INTER_AREA)
    labels = stat_labels(rec.get("fix"))
    look = look_of(a)
    era, how = era_of(item, W0, H0, labels, look)
    shp = rec.get("shape") or shape(W0, H0)
    band = BAND[shp]
    title = read_title(rec.get("band"))
    tb = title_box(rec, title, band)
    kind = "card" if (len(labels) >= 2 or era in ("A", "B", "C", "D1", "D2", "D3", "R", "bars-other")) else "other"
    out = {"id": item["id"], "src": item["src"], "era": era, "era_how": how, "w": W0, "h": H0, "kind": kind,
           "title": title}
    if want("ability"):
        ability_icons(out, work, ability_reads(out, rec, labels, band, title), title)
    if want("stars"):
        out["stars"] = read_stars(work, STAR_BAND[shp], tb)
    if want("blobs"):
        out["star_blobs"] = read_star_blobs(work, tb, BLOB_BAND[shp])
    tk = types_known_of(work, shp, title) if want("types") else None
    if tk:
        out["types_known"] = tk
    return out


def ability_reads(out, rec, labels, band, title):
    """out's ability_fix / ability_raw (both OCR passes: text, rows, class, minconf, odd, scores); the fix rows."""
    fix_rows = []
    for which in ("fix", "raw"):
        rows, cls, odd = ability_block(rec.get(which), out["era"], labels, band, out["kind"])
        out["ability_" + which] = {"text": ability_text(rows), "rows": len(rows),
                                   "class": " ".join(l[0] for l in cls) if cls else None,
                                   "minconf": min((l[1] for r in rows for l in r), default=0.0), "odd": odd}
        if which == "fix":
            fix_rows = rows
    out["ability_fix"].update(ability_scores(out["ability_fix"]["text"], out["ability_raw"]["text"], title))
    out["ability_raw"].update(ability_scores(out["ability_raw"]["text"], out["ability_fix"]["text"], title))
    return fix_rows


def ability_icons(out, work, fix_rows, title):
    """out's ability_icons: the fix text with the type icons in its lines placed (none found: no entry)."""
    out.pop("ability_icons", None)
    icons = inline_icons(work, fix_rows, WORK_W) if fix_rows else []
    if icons:
        out["ability_icons"] = {"text": place_icons(fix_rows, icons), "odd": out["ability_fix"]["odd"],
                                "icons": [[r, round(x, 3), n, s] for r, x, n, s in icons]}
        out["ability_icons"].update(ability_scores(out["ability_icons"]["text"], None, title))


def ability_task(args):
    """The ability read again from the stored OCR (raw.jsonl's other fields kept); the picture is loaded only when
    the fix text changed, for its icons."""
    import cv2
    import numpy as np
    item, rec, rd = args
    try:
        out = dict(rd)
        shp = rec.get("shape") or shape(rd["w"], rd["h"])
        title = read_title(rec.get("band"))
        before = (rd.get("ability_fix") or {}).get("text")
        fix_rows = ability_reads(out, rec, stat_labels(rec.get("fix")), BAND[shp], title)
        if out["ability_fix"]["text"] != before:
            img = load(item)
            a = np.asarray(img.convert("RGB"))
            W0, H0 = img.size
            work = cv2.resize(a, (WORK_W, max(2, round(WORK_W * H0 / W0))), interpolation=cv2.INTER_AREA)
            ability_icons(out, work, fix_rows, title)
        elif out.get("ability_icons"):
            out["ability_icons"]["odd"] = out["ability_fix"]["odd"]
        return out
    except Exception as e:
        return {"id": item["id"], "error": f"{type(e).__name__}: {str(e)[:160]}"}


def types_known_of(work, shp, title):
    """The read card's known type sets and every bank type's places on the picture (other icons raise alarms)."""
    sets = known_types().get(loose(title["name"])) if title.get("name") else None
    if not sets:
        return None
    canon = same_icons()
    names = sorted({t for ts, src in sets for t in ts} | {canon.get(i[0], i[0]) for i in bank("types")})
    return {"sets": sets, "presence": type_presence(work, ICON_BAND[shp], names)}


def types_task(args):
    """The title and the types read again (types.jsonl; calibrate takes them over raw.jsonl's)."""
    import cv2
    import numpy as np
    item, rec = args
    try:
        title = read_title(rec.get("band"))
        out = {"id": item["id"], "title": title}
        if known_types().get(loose(title["name"])) if title.get("name") else None:
            img = load(item)
            a = np.asarray(img.convert("RGB"))
            W0, H0 = img.size
            work = cv2.resize(a, (WORK_W, max(2, round(WORK_W * H0 / W0))), interpolation=cv2.INTER_AREA)
            out["types_known"] = types_known_of(work, rec.get("shape") or shape(W0, H0), title)
        return out
    except Exception as e:
        return {"id": item["id"], "error": f"{type(e).__name__}: {str(e)[:160]}"}


def read_task(args):
    item, rec = args
    if rec.get("error"):
        return {"id": item["id"], "error": rec["error"]}
    try:
        return read_one(item, rec)
    except Exception as e:
        return {"id": item["id"], "error": f"{type(e).__name__}: {str(e)[:160]}"}


def run_reads(which, workers, path=None, ids=None, task=None):
    """raw.jsonl (task read_task), or stars.jsonl (task star_task: the stars read again; calibrate takes them)."""
    task = task or read_task
    path = path or join(OUT, "raw.jsonl")
    ocr = {r["id"]: r for r in jsonl(join(OUT, "ocr.jsonl"))}
    names_index()
    known_abilities()
    known_types()
    items = [i for i in all_inputs(which) if i["id"] in ocr and (ids is None or i["id"] in ids)]
    done = {r["id"] for r in jsonl(path)}
    todo = [(i, ocr[i["id"]]) for i in items if i["id"] not in done]
    print(f"read: {len(done)} done, {len(todo)} to go", flush=True)
    start, n = time.time(), 0
    with open(path, "a") as f, ProcessPoolExecutor(workers) as ex:
        for res in ex.map(task, todo, chunksize=4):
            f.write(json.dumps(res) + "\n")
            f.flush()
            n += 1
            if n % 200 == 0:
                print(f"read: {n}/{len(todo)} {time.time() - start:.0f}s", flush=True)


def rerun_abilities(workers):
    """raw.jsonl's abilities read again (ability_task), written in place."""
    path = join(OUT, "raw.jsonl")
    ocr = {r["id"]: r for r in jsonl(join(OUT, "ocr.jsonl"))}
    items = {i["id"]: i for i in all_inputs()}
    names_index()
    known_abilities()
    rows = list(jsonl(path))
    todo = [(items[r["id"]], ocr[r["id"]], r) for r in rows if not r.get("error")]
    keep = [r for r in rows if r.get("error")]
    start, out, failed = time.time(), [], 0
    with ProcessPoolExecutor(workers) as ex:
        for (item, rec, rd), res in zip(todo, ex.map(ability_task, todo, chunksize=16)):
            failed += bool(res.get("error"))
            out.append(rd if res.get("error") else res)      # a failed re-read keeps the row as it was
            if len(out) % 1000 == 0:
                print(f"abilities: {len(out)}/{len(todo)} {time.time() - start:.0f}s", flush=True)
    write_jsonl(path, keep + out)
    print(f"abilities: {len(out)} read, {failed} kept as they were (errors)")


# ========================
# Truth
# ========================
PAIR_BITS = 3.0                                  # an old PNG and a PSD of its card this near (hash bits): one look


def build_truth():
    """truth.jsonl: {id, src, era?, fields: {title, stars, types, ability}, how}: the old repo's PNGs with the PSD of
    the same look (its read), today's cards with today's data, and the eye files (eye/*.json, {id: {field: value}})."""
    from commands.discord_match import hash_distance, load_cards
    rows = []
    reads = {}
    for r in jsonl(join(ROOT, "derived/oldgit/reads.jsonl")):
        reads.setdefault((r["kind"], r["blob"]), r)
    by_card = defaultdict(list)
    for v in jsonl(join(ROOT, "derived/oldgit/versions.jsonl")):
        by_card[v.get("old_id")].append(v)
    for v in old_inputs():
        png = reads.get(("png", v["blob"]))
        if not png or not png.get("look"):
            continue
        best, near = None, set()
        for p in by_card.get(v["old_id"], []):
            if p["ext"] != "psd":
                continue
            psd = reads.get(("psd", p["blob"]))
            if not psd or not psd.get("ok") or not isinstance(psd.get("composite"), dict) or not psd.get("read"):
                continue                # no composite ('blank', an error)
            d = hash_distance(png["look"]["whole"], psd["composite"]["whole"])
            if d <= 2 * PAIR_BITS:
                near.add("psd:" + p["blob"])
            if d <= PAIR_BITS:          # the PSD as it was when the PNG was made (text moves no hash bits)
                when = (p["ts"] > v["ts"] + 60, abs(p["ts"] - v["ts"]))
                if best is None or when < best[1]:
                    best = (d, when, psd)
        if not best:
            continue
        best = (best[0], best[2])
        rd = best[1]["read"]
        problems = " ".join(rd.get("problems") or []) + " " + " ".join(rd.get("validator") or [])
        fields = {}
        if rd.get("title"):
            fields["title"] = rd.get("printed_title") or rd["title"]
        if isinstance(rd.get("stars"), int) and "STARS" not in problems:
            fields["stars"] = rd["stars"]
        if rd.get("types") is not None and rd.get("type") in ("creature", "minion", "item", "field", "aux item"):
            fields["types"] = rd["types"]
        if rd.get("ability") and "ABILITY" not in problems:
            fields["ability"] = rd["ability"]
        rows.append({"id": v["id"], "src": "old", "fields": fields, "how": "psd", "bits": round(best[0], 2),
                     "kind": rd.get("type"), "own": sorted(near)})
    stats = json.loads(git(join(TC, ".git"), "show", "origin/data:db/stats.json") or b"{}").get("cards") or {}
    for cid, card in load_cards(TC).items():
        st = stats.get(card.get("key")) or {}
        if not card.get("png") or not st:
            continue
        fields = {"title": (st.get("parsed_values") or {}).get("name") or st.get("display_name") or card["key"],
                  "stars": st.get("stars")}
        if st.get("types") is not None:
            fields["types"] = st["types"]
        if st.get("ability"):
            fields["ability"] = st["ability"]
        rows.append({"id": "t:" + cid, "src": "today", "fields": fields, "how": "today", "kind": st.get("card_type"),
                     "own": ["t:" + cid]})
    eye_dir = join(OUT, "eye")
    for f in sorted(os.listdir(eye_dir)) if os.path.isdir(eye_dir) else []:
        for k, fields in (read_json(join(eye_dir, f), {}) or {}).items():
            if not k.startswith("_"):
                rows.append({"id": k, "src": "discord" if k.startswith("d:") else k.split(":")[0], "fields": fields,
                             "how": "eye:" + f})
    write_jsonl(join(OUT, "truth.jsonl"), rows)
    print(Counter(r["how"] for r in rows), Counter(f for r in rows for f in r["fields"]))


# ========================
# Calibration
# ========================
STAR_RULES = ([{"ncc": n, "mad": m} for n in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85) for m in (20, 30, 45, 60, 255)]
              + [{"kind": "row", "tol": t, "rows": r} for t in (1.25, 1.43, 1.7) for r in (0.6, 1.6)]
              + [{"kind": "cluster", "step": s, "base": b} for s in (0.33, 0.37, 0.41, 0.45) for b in (1.0, 1.05, 1.1)])
TYPE_RULES = [{"on": a, "off": b} for a in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9) for b in (0.6, 0.7, 0.8, 0.9)
              if b <= a + 0.05]
VARIANTS = ("fix", "raw", "icons")
TRUTH_OF = {"discord": "eye", "old": "psd", "today": "today"}
TITLE_FLOOR = 0.9286          # names no list has: with its own name left out, 2.3% of truth titles still reach this
EYE_ABILITY_BAR = 0.81        # eye eras' plain OCR: both passes agree, Vision sure, every word a game word
KNOWN_STARS_BAR = 0.55        # A/C star counts: a known count, with some margin of their own
NO_STAR_BAR = ("R",)          # raid bosses: the known/truth count is the folder's 6, not the print (Oryx 7, Giegue 9)
NOT_CARD_LOOKS = ("S",)       # squares by look alone (screenshots too): never sure
ALARM_Q = 0.995               # an other type icon at its 99.5th percentile on pictures without it: an extra icon
_ALARMS = None


def titles_of(t):
    return {norm(x) for x in (t if isinstance(t, list) else [t]) if x}


def ability_conf(text, agree, minconf, known, odd=0):
    """An ability read's confidence: a known text of the card read exactly is 1; else at most 0.81, from the two
    OCR passes agreeing, Vision's line confidence, the share of game words (one-letter words other than 'a' are
    not) and no odd characters (accents, OCR's brackets). Each line of the block OCR gave in another script
    (odd: a word lost) halves it, a known text's too."""
    if not text:
        return 0.0
    if known:
        return round(0.5 ** odd, 4)
    ws = [w for w in ability_words(text) if not w.isdigit() and not w.startswith("<")]
    voc = vocab()
    share = sum(w in voc and (len(w) > 1 or w == "a") for w in ws) / len(ws) if ws else 0.0
    base = 0.5 + 0.3 * bool(agree) + 0.1 * (minconf >= 0.99)
    return round(0.9 * base * share ** 4 * 0.5 ** (odd_marks(text) + odd), 4)


def ability_variant(rd, variant):
    """The ability read of one variant (fix / raw / icons: the fix text with the type icons placed), or None."""
    if variant == "icons":
        return rd.get("ability_icons") or rd.get("ability_fix")
    return rd.get("ability_" + variant)


def field_read(rd, field, how, exclude=()):
    """(value, method, conf) of one field of a raw read under how (the era's calibrated choices)."""
    if field == "title":
        t = rd.get("title") or {}
        if t.get("name"):
            return t["name"], "ocr+name", t.get("conf", 0.0)
        return t.get("text"), "ocr", 0.0
    if field == "ability":
        ab = ability_variant(rd, how.get("variant", "fix")) or {}
        text = ab.get("text") or None
        method = "ocr+icons" if how.get("variant") == "icons" and rd.get("ability_icons") else "ocr"
        agree = passes_agree((rd.get("ability_fix") or {}).get("text"), (rd.get("ability_raw") or {}).get("text"))
        known = ability_known((rd.get("title") or {}).get("name"), text, exclude)
        minconf = min((rd.get("ability_fix") or {}).get("minconf") or 0.0, 1.0)
        return text, method + ("+known" if known else ""), ability_conf(text, agree, minconf, known, ab.get("odd", 0))
    if field == "stars":
        rule = how.get("rule") or STAR_RULES[0]
        n, conf = (count_blobs(rd.get("star_blobs"), rule) if rule.get("kind") else
                   count_stars(rd.get("stars"), rule))
        if n == 1 and not rule.get("kind") and rd.get("era") in ONE_STAR_UNSURE:
            return n, "icon", 0.0              # the cut star alone: overlapped stars read 1 too
        if n is not None and stars_known((rd.get("title") or {}).get("name"), n, exclude):
            return n, "icon+known", round(0.5 + 0.5 * conf, 4)
        if rd.get("era") in KNOWN_STARS_ONLY:      # loose stars, clusters and stacks: a count alone is never sure
            return n, "icon", 0.0
        return n, "icon", round(0.5 * conf, 4)
    if field == "types":
        ts, conf = verified_types(rd, how.get("rule") or TYPE_RULES[0], exclude, _ALARMS)
        return ts, "icon+known", conf
    return None, None, 0.0


def is_right(field, value, truth, card=False):
    if value is None:
        return False
    if field == "title":
        return norm(value) in titles_of(truth)
    if field == "ability":
        canon = same_icons()
        fix = lambda ws: [("<%s>" % canon.get(w[1:-1], w[1:-1])) if w.startswith("<") else w for w in ws]
        return fix(ability_words(value)) == fix(ability_words(truth))
    if field == "stars":
        return int(value) == int(truth)
    if field == "types":
        return set(value) == {t for t in canon_types(truth) if not (card and t == "equip")}
    return False


def pick_bar(rows, target=BAR_TARGET):
    """rows [(conf, right)]: the lowest bar whose reads at or above it are >= target right (errors allowed:
    floor((1 - target) * k)); (bar, k, right). No such bar: (None, 0, 0)."""
    rows = sorted(rows, key=lambda r: -r[0])
    best = (None, 0, 0)
    ok = 0
    for k, (conf, right) in enumerate(rows, 1):
        ok += bool(right)
        if conf <= 0:
            break
        if k - ok <= int((1 - target) * k) and (k == len(rows) or rows[k][0] < conf):
            best = (conf, k, ok)
    return best


def truth_rows():
    out = defaultdict(list)
    for t in jsonl(join(OUT, "truth.jsonl")):
        out[t["id"]].append(t)
    return out


def usable_truth(field, t):
    """A truth row's field, when it is trusted: the old repo's PSD reads give no empty type lists (the 2023 parser
    missed item icons) and no abilities of cards whose text sits in a layer named after it."""
    v = t["fields"].get(field)
    if v is None:
        return None
    if field == "types" and t["how"] == "psd" and not v:
        return None
    return v


def alarm_levels(raw, truth, q=None):
    """{type: level}: each bank type's best score on truth pictures known not to print it (away from their own
    icons), at quantile q. Types that score that high on look-alike shapes (light, equip) get a level they never
    reach on a card that lacks them only rarely; a type with too few such pictures gets none."""
    q = ALARM_Q if q is None else q
    scores = defaultdict(list)
    for rid, rd in raw.items():
        tk = rd.get("types_known")
        if not tk or "light" not in tk.get("presence", {}):        # every bank type's places (the types step)
            continue
        for t in truth.get(rid, []):
            v = usable_truth("types", t)
            if v is None or t["how"].startswith("eye:val_"):
                continue
            card = rd.get("kind") == "card"
            ts = {x for x in canon_types(v) if not (card and x == "equip")}
            pres = {}
            for name, p in tk["presence"].items():
                c = canon_types([name])[0]
                p = places_of(p)
                if c not in pres or p[0][0] > pres[c][0][0]:
                    pres[c] = p
            mine = [pres[x][0] for x in ts if x in pres]
            for name, places in pres.items():
                if name in ts or (card and name == "equip"):
                    continue
                scores[name].append(max((p[0] for p in places if all(place_overlap(p, m) < 0.3 for m in mine)),
                                        default=0.0))
            break
    out = {}
    for name, v in scores.items():
        v.sort()
        if len(v) >= 50:
            out[name] = round(v[min(len(v) - 1, int(q * len(v)))] + 0.001, 3)
    return out


def sheet_ids():
    """Pictures of several cards (no sure read until cut into cells): the match's checked sheets and the
    catalogue's multi-card sheets and before/after pairs."""
    out = set()
    for v in jsonl(join(ROOT, "derived/match/verdicts.jsonl")):
        if v.get("sheet") and v.get("sha256"):
            out.add("d:" + v["sha256"])
    for r in jsonl(join(ROOT, "derived/catalogue/pictures.jsonl")):
        if r.get("sheet") in ("multi", "pair"):
            out.add("d:" + splitext(basename(r["file"]))[0])
    return out


def no_sure(rd, sheets):
    """Why a picture gets no sure read: a sheet of cards, or a square classed by look alone (screenshots too)."""
    if rd["id"] in sheets:
        return "sheet"
    if rd.get("era_how") == "look" and rd["era"] in NOT_CARD_LOOKS:
        return "look"
    return None


def confirmed(rd):
    """A card by stage 1's era, its stat labels or its file (not by look alone): its OCR text is a card's."""
    return rd.get("era_how") in ("stage1", "labels", "size", "bars") and rd["era"] not in UNSORTED


def floored(era, src, field, n, bar):
    """The bar after the floors: every title bar >= TITLE_FLOOR; small eye eras' abilities >= EYE_ABILITY_BAR;
    eras whose star counts need a known count > 0.5; no raid-boss stars; no Discord types (a pre-git icon outside
    every Types/ tree, or stacked under another, goes unseen: 4 of 48 sure reads wrong by eye)."""
    if bar is None or (field == "types" and src == "discord") or (field == "stars" and era in NO_STAR_BAR):
        return None
    if field == "title":
        return max(bar, TITLE_FLOOR)
    if field == "ability" and src == "discord" and 0 < n < EYE_POOL:
        return max(bar, EYE_ABILITY_BAR)
    if field == "stars" and era in KNOWN_STARS_ONLY:
        return max(bar, KNOWN_STARS_BAR)
    return bar


def calibrate():
    """Per era and field: the rule/variant with the most right reads on the era's truth, then the bar (BAR_TARGET),
    2-fold cross-checked, then the floors; reads.jsonl (Discord, old repo) and today.jsonl with each field's value,
    method, confidence and sure."""
    global _ALARMS
    raw = {r["id"]: r for r in jsonl(join(OUT, "raw.jsonl")) if not r.get("error")}
    for r in jsonl(join(OUT, "stars.jsonl")):
        if r["id"] in raw and not r.get("error"):
            raw[r["id"]].update(stars=r["stars"], star_blobs=r.get("star_blobs"))
    for r in jsonl(join(OUT, "edges.jsonl")):
        if r["id"] in raw and not r.get("error"):
            raw[r["id"]]["star_edge"] = r
    for r in jsonl(join(OUT, "types.jsonl")):
        if r["id"] in raw and not r.get("error"):
            raw[r["id"]]["title"] = r["title"]
            raw[r["id"]].pop("types_known", None)
            if r.get("types_known"):
                raw[r["id"]]["types_known"] = r["types_known"]
    truth = truth_rows()
    sheets = sheet_ids()
    _ALARMS = alarm_levels(raw, truth)
    by_era = defaultdict(list)
    for rid, rd in raw.items():
        by_era[(rd["era"], rd["src"])].append(rd)
    cal = {"made": time.strftime("%Y-%m-%d %H:%M"), "target": BAR_TARGET, "eras": {},
           "type_alarms": {"q": ALARM_Q, "levels": _ALARMS}, "title_floor": TITLE_FLOOR,
           "eye_ability_bar": EYE_ABILITY_BAR, "known_stars_bar": KNOWN_STARS_BAR,
           "no_sure": dict(Counter(no_sure(rd, sheets) for rd in raw.values() if no_sure(rd, sheets)))}
    choices, labelled_of = {}, {}
    eye_rows = defaultdict(list)

    def score(field, how, rows):
        res = []
        for rd, v, own in rows:
            val, method, conf = field_read(rd, field, how, own)
            res.append((conf if not no_sure(rd, sheets) else 0.0,
                        is_right(field, val, v, rd.get("kind") == "card")))
        return res

    def n_sure(rds, field, how, bar):
        return sum(1 for rd in rds if bar is not None and not no_sure(rd, sheets)
                   and field_read(rd, field, how)[0] is not None and field_read(rd, field, how)[2] >= bar)
    for (era, src), rds in sorted(by_era.items()):
        key = f"{era}/{src}"
        cal["eras"][key] = {"population": len(rds)}
        for field in FIELDS:
            labelled = []
            for rd in rds:
                for t in truth.get(rd["id"], []):
                    v = usable_truth(field, t)
                    if v is not None and not t["how"].startswith("eye:val_"):
                        labelled.append((rd, v, t.get("own") or []))
                        break
            labelled_of[(era, src, field)] = labelled
            options = [{}]
            if field == "stars":
                options = [{"rule": r} for r in STAR_RULES]
            elif field == "types":
                options = [{"rule": r} for r in TYPE_RULES]
            elif field == "ability":
                options = [{"variant": v} for v in VARIANTS]
            best = None
            for how in options:
                res = score(field, how, labelled)
                bar, k, ok = pick_bar(res)
                rank = (k, sum(r[1] for r in res), -len(json.dumps(how)))
                if best is None or rank > best[0]:
                    best = (rank, how, res, bar, k, ok)
            _, how, res, bar, k, ok = best

            def crosscheck(target):
                """Bars fitted on one half, measured on the other (n >= 40)."""
                if len(labelled) < 40:
                    return None
                half = [labelled[0::2], labelled[1::2]]
                hits = n_above = 0
                for a, b in ((0, 1), (1, 0)):
                    ba, _, _ = pick_bar(score(field, how, half[a]), target)
                    ba = floored(era, src, field, len(labelled), ba)
                    if ba is not None:
                        sb = score(field, how, half[b])
                        hits += sum(r[1] for r in sb if r[0] >= ba)
                        n_above += sum(1 for r in sb if r[0] >= ba)
                return {"right": hits, "above": n_above, "accuracy": round(hits / n_above, 4) if n_above else None,
                        "target": target}
            for target in BAR_TARGETS:      # a stricter fit while the cross-check falls under SURE_MIN
                bar = pick_bar(res, target)[0]
                cv = crosscheck(target)
                if not cv or cv["accuracy"] is None or cv["accuracy"] >= SURE_MIN:
                    break
            else:
                bar = None
            thin = pick_bar(res, target)[1] if field == "stars" and bar is not None else None
            if thin is not None and thin >= STAR_MIN_ABOVE:
                thin = None
            cal["eras"][key][field] = {"truth": TRUTH_OF.get(src), "n": len(labelled), "how": how,
                                       "fitted_bar": None if thin is not None else bar, "target": target,
                                       "accuracy_all": round(sum(r[1] for r in res) / len(res), 4) if res else None,
                                       "cv": cv}
            if thin is not None:
                cal["eras"][key][field].update(thin_bar=bar, thin_above=thin)
                bar = None
            choices[(era, src, field)] = (how, bar)
            eye_rows[(field, EYE_GROUP.get(era, ""))] += [(r[0], r[1], key) for r in res] if src == "discord" else []
    # small eye samples: a bar under the pooled eye bar (every eye era's reads of its group, each under its era's
    # choice) is raised to it (stars excepted: each era counts by its own method)
    for field in ("title", "ability", "types"):
        for group in sorted({g for f, g in eye_rows if f == field}):
            rows = eye_rows[(field, group)]
            pbar, pk, pok = pick_bar([(c, r) for c, r, k in rows])
            cal["eye_pooled_" + field + (f" ({group})" if group else "")] = {"n": len(rows), "bar": pbar, "above": pk,
                                                                            "right_above": pok}
            for (era, src, f), (how, bar) in list(choices.items()):
                entry = cal["eras"][f"{era}/{src}"][f]
                if f == field and src == "discord" and EYE_GROUP.get(era, "") == group and 0 < entry["n"] < EYE_POOL \
                        and bar is not None and pbar is not None and pbar > bar:
                    entry["raised_from"] = bar
                    choices[(era, src, f)] = (how, pbar)
    # pooled bars (every era's truth): a title's name likeness and an ability read equal to a known text mean
    # the same in every era; eras with no truth of their own take these
    pooled = {}
    for field, how in (("title", {}), ("ability", {"variant": "fix"})):
        rows = []
        for rd in raw.values():
            for t in truth.get(rd["id"], []):
                v = usable_truth(field, t)
                if v is not None and not t["how"].startswith("eye:val_"):
                    rows += score(field, how, [(rd, v, t.get("own") or [])])
                    break
        bar, k, ok = pick_bar(rows)
        pooled[field] = (how, bar, "pooled")
        cal["pooled_" + field] = {"n": len(rows), "bar": bar, "above": k, "right_above": ok}
    # eras with too little truth borrow: Discord template eras take the old repo's bar for the same era
    for (era, src, field), (how, bar) in list(choices.items()):
        n = cal["eras"][f"{era}/{src}"][field]["n"]
        if n >= MIN_N:
            continue
        donor = next((choices[(era, s, field)] + (f"{era}/{s}",) for s in ("old", "discord", "today")
                      if s != src and (era, s, field) in choices and cal["eras"][f"{era}/{s}"][field]["n"] >= MIN_N),
                     None) or (pooled.get(field) if era not in UNSORTED else None)
        if donor:
            cal["eras"][f"{era}/{src}"][field].update(borrowed=donor[2], how=donor[0])
        choices[(era, src, field)] = donor[:2] if donor else (how, None)
    # the floors, then each era's numbers at its final bar
    for (era, src, field), (how, bar) in list(choices.items()):
        entry = cal["eras"][f"{era}/{src}"][field]
        fbar = floored(era, src, field, entry["n"], bar)
        if fbar != bar:
            entry["floored_from"] = bar
        choices[(era, src, field)] = (how, fbar)
        res = score(field, how, labelled_of[(era, src, field)])
        above = [r for r in res if fbar is not None and r[0] >= fbar]
        rds = by_era[(era, src)]
        entry.update(bar=fbar, above=len(above), right_above=sum(r[1] for r in above),
                     accuracy_at_bar=round(sum(r[1] for r in above) / len(above), 4) if above else None,
                     share_above_truth=round(len(above) / len(res), 4) if res else None,
                     share_above_all=round(n_sure(rds, field, how, fbar) / len(rds), 4) if rds else None)
    for (era, src), rds in by_era.items():          # held-out eye checks (eye/val_*.json) of the final bars
        for field in FIELDS:
            how, bar = choices.get((era, src, field), ({}, None))
            rows = []
            for rd in rds:
                for t in truth.get(rd["id"], []):
                    v = usable_truth(field, t)
                    if v is not None and t["how"].startswith("eye:val_"):
                        val, method, conf = field_read(rd, field, how)
                        rows.append((bar is not None and val is not None and conf >= bar and not no_sure(rd, sheets),
                                     is_right(field, val, v, rd.get("kind") == "card")))
            if rows:
                cal["eras"][f"{era}/{src}"][field]["validation"] = {
                    "n": len(rows), "sure": sum(r[0] for r in rows), "right_sure": sum(r[0] and r[1] for r in rows),
                    "right_all": sum(r[1] for r in rows)}
    write_json(join(OUT, "calibration.json"), cal)
    outs = {"reads": [], "today": []}
    where = {i["id"]: {k: i[k] for k in ("file", "blob", "path") if k in i} for i in all_inputs(("discord", "old"))}
    for rid, rd in raw.items():
        row = {"id": rid, "src": rd["src"], "era": rd["era"], "era_how": rd.get("era_how"), "w": rd["w"], "h": rd["h"],
               **where.get(rid, {})}
        why = no_sure(rd, sheets)
        if why:
            row["no_sure"] = why
        card = confirmed(rd)
        for field in FIELDS:
            how, bar = choices.get((rd["era"], rd["src"], field), ({}, None))
            val, method, conf = field_read(rd, field, how)
            sure = bool(bar is not None and val is not None and conf >= bar and not why)
            if not card and not sure and field in ("title", "ability") and method in ("ocr", "ocr+icons"):
                val = None              # OCR of a picture no era confirms (screenshots): never kept
            doubt = star_doubt(rd, how.get("rule") or STAR_RULES[0], val) if field == "stars" and val is not None \
                else None
            row[field] = {"value": val, "method": method, "conf": conf, "sure": sure and not doubt}
            if doubt:
                row[field]["doubt"] = doubt     # never sure (the bars stay as fitted)
            if field == "title" and card and rd.get("title", {}).get("text"):
                row[field]["ocr"] = rd["title"]["text"]
            if field == "title" and rd.get("title", {}).get("clash"):
                row[field]["clash"] = rd["title"]["clash"]
            if field == "types" and val is not None and _ALARMS and rd.get("types_known"):
                extra = extra_icons({canon_types([t])[0]: places_of(p) for t, p in rd["types_known"]["presence"].items()},
                                    val, how.get("rule") or TYPE_RULES[0], _ALARMS, rd.get("kind") == "card")
                if extra:
                    row[field]["extra"] = extra
        outs["today" if rd["src"] == "today" else "reads"].append(row)
    write_jsonl(join(OUT, "reads.jsonl"), sorted(outs["reads"], key=lambda r: r["id"]))
    write_jsonl(join(OUT, "today.jsonl"), sorted(outs["today"], key=lambda r: r["id"]))
    for key, v in sorted(cal["eras"].items()):
        print(key, v["population"], "  ".join(
            f"{f}: n={v[f]['n']} bar={v[f]['bar']} acc={v[f]['accuracy_at_bar']} above={v[f]['share_above_truth']}"
            f"/{v[f]['share_above_all']}" for f in FIELDS if f in v))


# ========================
# CLI
# ========================
def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=("icons", "ocr", "read", "stars", "edges", "types", "abilities", "truth",
                                     "calibrate"))
    ap.add_argument("-j", "--workers", type=int, default=2)
    ap.add_argument("--only", choices=("discord", "old", "today"), action="append")
    a = ap.parse_args(argv)
    os.makedirs(join(OUT, "tmp"), exist_ok=True)
    which = tuple(a.only or ("discord", "old", "today"))
    if a.step == "icons":
        build_icons()
    elif a.step == "ocr":
        ocr_bin()
        run_pool(ocr_task, all_inputs(which), a.workers, join(OUT, "ocr.jsonl"), "ocr")
    elif a.step == "read":
        run_reads(which, a.workers)
    elif a.step == "stars":
        run_reads(which, a.workers, join(OUT, "stars.jsonl"), task=star_task)
    elif a.step == "edges":
        run_reads(which, a.workers, join(OUT, "edges.jsonl"), task=edge_task)
    elif a.step == "types":
        run_reads(which, a.workers, join(OUT, "types.jsonl"), task=types_task)
    elif a.step == "abilities":
        rerun_abilities(a.workers)
    elif a.step == "truth":
        build_truth()
    elif a.step == "calibrate":
        calibrate()


if __name__ == "__main__":
    main(sys.argv[1:])
