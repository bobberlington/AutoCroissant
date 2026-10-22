"""
Discord archive matcher: ties the posts the miner stored (commands/discord_archive.py) to cards and pulls card
metadata out of them. Run from the repo root after the miner, with a Python that has Pillow and numpy; OCR needs
pyobjc's Vision (macOS: pip install pyobjc-framework-Vision pyobjc-framework-Quartz), else pass --no-ocr:

    python -m commands.discord_match score    [--root <dir>] [--cards-repo <dir>] [--site <url>] [--refresh-site]
                                              [--versions none|oldest|all] [--no-ocr] [--workers <n>] [--limit <n>]
    python -m commands.discord_match build    [--root <dir>]
    python -m commands.discord_match selftest [--root <dir>] [--cards-repo <dir>] [--workers <n>] [--limit <n>]

score    Cards: every card PNG in the checkout, plus older versions from the site's versions.json (fetched from GitHub
         by commit), each reduced to perceptual hashes (dHash, pHash 64 and 256 bit) of the whole card and of its art,
         its OCR'd title band and its whole OCR'd text; versions alike in text and hash are one look. Posts: every
         stored image (attachments, embed images, PSD composites) gets the same (and, in screenshots, art hashes under
         each title-like line). Each image's candidate cards, per-method scores (hash, title OCR, OCR'd lines, OCR'd
         text vs the versions' and vs the ability, file name, chat text) and the version it shows (by text, else by
         hash) go to match/pairs.json. A sheet of several cards is cut into its cells, each scored as a card; card-
         shaped images alike (twins) are paired. Features are cached by content: a rerun after more mining is quick.
build    One confidence per (image, card), methods that disagree settled by kind of evidence, the owner's rulings
         (match/rulings.json, from apply_rulings.py) applied, a twin of a sure post made sure, then matches.json
         (per card: first post, the look it shows, each later look's first post, its art's first post; suspect when
         after fs), ambiguous.json and ambiguous.html (stable ids A1, A2...; first posts to confirm too; bot posts
         left out), REPORT.md and proposal/ (created dates and aliases as Card Edits.json fields, author fixes as
         Card Authors.txt lines).
         Standard library only: apply_rulings.py reruns it with any Python 3.
selftest The image methods on the cards themselves (degraded copies of every PNG), and the version pick on degraded
         copies of every version of the cards with more than one look: accuracy into match/selftest.json.
Everything stays under the archive root (never inside a git repo: the messages are other people's words); nothing
here talks to Discord. Bot and webhook posts never count as creation evidence.
"""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from difflib import SequenceMatcher
from fcntl import LOCK_EX, LOCK_NB, flock
from glob import glob
from hashlib import sha1, sha256
from html import escape
from io import BytesIO
from json import dumps, load, loads
from math import log
from multiprocessing import get_context
from os import cpu_count, environ, makedirs, replace
from os.path import abspath, basename, dirname, exists, expanduser, getmtime, isdir, isfile, join, realpath, splitext
from re import IGNORECASE, compile as re_compile, findall, finditer, fullmatch, split as re_split, sub
from sys import stderr, stdin
from time import time
from typing import Callable, Optional
from unicodedata import normalize
from urllib.parse import quote

from global_config import CARD_REPO, LOCAL_DIR_LOC, RAW_BASE, SITE_URL

# ========================
# Configuration
# ========================
DEFAULT_ROOT = "~/Documents/Tools/discord-archive"
MATCH = "match"                                   # the matcher's folder in the root
CARD_FEATURES, POST_FEATURES, CELL_FEATURES = 5, 3, 1   # bump when card, post or sheet cell features change
PAIRS_VERSION = 3
SITE_FILES = ("cards.json", "versions.json")
SITE_MAX_AGE = 12 * 3600                          # seconds before the site's data is fetched again
USER_AGENT = "AutoCroissant discord_match"
IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "webp", "bmp", "tif", "tiff", "psd"}
MAX_PIXELS = 120_000_000
WORK_SIDE = 2400                                  # images shrunk to this long side first
OCR_SIDE = 2000                                   # whole-image OCR size
REREADS = ((900, 95), (600, 85))                  # card versions read again: width, JPEG quality
SHAPES = {"2:3": 2 / 3, "1:2": 0.5, "1:1": 1.0}   # card width / height
SHAPE_TOL = 0.04
TITLE_BAND = 0.28                                 # title band height, in card widths (two-line titles)
ART = (0.06, 0.17, 0.94, 0.62)                    # art box in card widths: art on every shape
TRIM_TOL, TRIM_MAX = 16, 0.35                     # border trim: color tolerance, most of a side cut
TRIM_CORNERS = 6                                  # padding's four corners agree this closely
TITLE_TEXT = (0.065, 0.08, 0.095, 0.11)           # title text height / card width: screenshot anchor guesses
TITLE_ROWS = (0.075, 0.05, 0.15)                  # a title line's middle (one line, first or second of two)
TITLE_LOW, TITLE_LINE = 0.7, 0.9                  # a title under the art: above this share of the height, this
                                                  # near a card's name
STAT_WORDS = {"health", "healthy", "defense", "defence", "attack", "speed", "hp", "def", "atk", "spd"}
ANCHOR_LINES = 8
THUMB_POST, THUMB_CARD = 480, 260
BACKGROUND = (0, 0, 0, 255)                       # transparency flattened onto this
DISCORD_GRAY = (49, 51, 56)

# scoring (calibrated by selftest; see REPORT.md)
HASH_FULL, HASH_ZERO = 7.0, 15.0                  # combined hash distance (of 64 bits): score 1 at or under, 0 at or over
TITLE_FLOOR, TITLE_FULL = 0.62, 0.92              # title OCR ratio: score 0 .. 1
LINE_FLOOR, LINE_FULL = 0.75, 0.97                # an OCR'd line in a screenshot
TALL, SMALL_LINE = 1.4, 0.95                      # a title line: this much taller than the median; near misses
                                                  # on other lines scaled
BODY_FLOOR, BODY_FULL, BODY_WORDS = 0.3, 0.75, 4  # OCR'd text vs ability: weighted share of the card's words
BODY_SURE = 8                                     # shared rare words a body score needs to count in full
OCR_FLOOR, OCR_FULL, OCR_GRAMS = 0.5, 0.8, 30     # OCR'd text vs a version's: share of the image's 4-grams found
LINE_ALONE, LIST_NAMES = 0.4, 3                   # screenshots: a name read with nothing else behind it counts
                                                  # this much when the image isn't card-like or names more cards
SWAP_WORD = 0.5                                   # a title word this unlike every word of a name: another name
PART_OF = {"of", "from", "s"}                     # a two-line title with these names a thing of a card (Cape of
                                                  # the Old King), not the card under an older, longer name
SHEET_ROW, SHEET_GAP, SHEET_LINE = 0.04, 0.3, 0.45  # a sheet: card names on one row (this near in height), this
                                                  # far apart across, each line under this share of the width
TWIN_BITS = 3.0                                   # card-shaped images this near (whole) are one image
NO_SUPPORT = 0.25                                 # a kind of evidence under this backs a card not at all
COMMON_GRAMS = 0.06                               # 4-grams on more than this share of cards: left out there
WEIGHTS = {"hash": 0.98, "anchor": 0.9, "title": 0.93, "line": 0.8, "ocr": 0.9, "body": 0.7, "file": 0.85,
           "text": 0.6}
IMAGE_METHODS = ("hash", "anchor", "title", "line", "ocr", "body", "file")
TEXT_METHODS = ("title", "line", "anchor", "ocr", "body")    # the image's own text backs the card
EVIDENCE = {"look": ("hash", "anchor"), "content": ("ocr", "body"), "name": ("title", "line", "file", "text")}
WHOLE_SURE = 0.9                                  # hash score of the whole image that needs no text behind it
ART_TEXT = 25                                     # characters read: fewer (and no stat label), and only art or
                                                  # the file name backs it: art
CONFIDENT, PLAUSIBLE, STRONG = 0.85, 0.35, 0.6
ODDS = 3.0                                        # the runner-up's risk vs the best's, for a confident pick
SHORT_TITLE = 8                                   # a title this short needs more than OCR alone
HIGH = 0.98
REVIEW = 0.95                                     # a card's first post under this goes to the owner to confirm
# versions (calibrated by selftest's version check; see REPORT.md)
TEXT_MIN = 20                                     # OCR'd characters an image needs to pick a version by text
LOOK_BITS, LOOK_GRAMS = 3.0, 2                    # versions one look: whole-card hash this near, and at most this
                                                  # many 4-grams in all of one's reads and none of the other's
BEAT_HITS, BEAT_SHARE = 2, 0.4                    # a look beats another: this many of its own 4-grams read, a
                                                  # share this much over the other's
ABSENT_GRAMS, ABSENT_SHARE = 6, 0.15              # uncropped: a look whose own 4-grams (this many) are this absent
HASH_TIE = 1.5                                    # bits: looks left tied, the nearest by hash within this
DRAFT_RUN = 18                                    # a look not in git: an uncropped image reads this many
DRAFT_BITS = 6.0                                  # characters in a row its look lacks; no text: art matches, the
                                                  # whole card this many bits off
CANDIDATES = 8
GIT_EARLIER_DAYS = 14                             # Discord first post this much after fs: an anomaly
NOTES_PER_CARD = 12
PORT = re_compile(r"(^|-)port$")                  # port channels: cards re-uploaded by whoever ported them
MEME = re_compile(r"meme")                        # meme channels: a card's image under a joke name
MENTION_ROUND = 600                               # seconds: a poster's message this close to an image is its caption

CREATION = re_compile(r"\b(new (card|creature|item|field|minion|status|buff|debuff)s?|introducing|made (a|this|another|my)|"
                      r"here'?s (my|a|the)|wip|concept|draft|finished|first (draft|version))\b", IGNORECASE)
BALANCE = re_compile(r"\b(buff(ed|s)?|nerf(ed|s)?|op|overpowered|underpowered|broken|balanc(e|ed|ing)|"
                     r"too (strong|weak|good|bad)|rework(ed)?|ban(ned)?)\b", IGNORECASE)
LORE = re_compile(r"\b(lore|story|backstory|based on|reference|inspired|canon|universe)\b", IGNORECASE)
GENERIC_FILE = {"image", "images", "unknown", "img", "screenshot", "screen", "shot", "capture", "photo", "download",
                "untitled", "message", "clipboard", "snip", "discord", "attachment", "file", "card", "cards", "new",
                "test", "temp", "tmp", "ezgif", "giphy", "tenor", "maxresdefault", "hqdefault", "default", "unnamed",
                "export", "at", "am", "pm", "pxl", "dsc", "dcim", "image0", "image1", "image2", "unknown0", "edit",
                "final", "copy", "version", "png", "jpg", "large", "medium", "small", "frame", "gif", "titre",
                "tumblr", "hq720", "sddefault", "mqdefault", "spoiler"}
VERSION_WORD = re_compile(r"v\d+|final|new|old|updated?|edit(ed)?|wip|copy|fixed|draft|rework(ed)?|\d{1,4}")
CHAT_STOP = {"lol", "lmao", "thoughts", "here", "this", "new", "card", "cards", "done", "finished", "wip", "what",
             "do", "you", "think", "it", "is", "the", "a", "an", "my", "i", "made", "ok", "okay", "yes", "no", "idk",
             "and", "or", "of", "for", "to", "in", "on", "with", "updated", "update", "fixed", "v2", "now", "also"}
CONFUSABLE = str.maketrans("01i5|", "olls" + "l")  # OCR look-alikes folded together
LOOKALIKE = str.maketrans("АВЕКМНОРСТХУЯИЗЬаеорсухкмтнвΑΒΕΗΙΚΜΝΟΡΤΧΥ", "ABEKMHOPCTXYRN3baeopcyxkmthbABEHIKMNOPTXY")


# ========================
# Helpers
# ========================
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def epoch(stamp) -> Optional[float]:
    """Seconds since 1970 from an ISO time or a number (None if neither)."""
    if isinstance(stamp, (int, float)):
        return float(stamp)
    try:
        return datetime.fromisoformat(str(stamp)).timestamp()
    except (TypeError, ValueError):
        return None


def iso(seconds) -> Optional[str]:
    return None if seconds is None else datetime.fromtimestamp(seconds, timezone.utc).isoformat(timespec="seconds")


def snowflake_seconds(snowflake) -> Optional[float]:
    try:
        return ((int(snowflake) >> 22) + 1420070400000) / 1000
    except (TypeError, ValueError):
        return None


def fold(text) -> str:
    """Lowercase ASCII: accents dropped, symbols gone."""
    return normalize("NFKD", str(text or "").translate(LOOKALIKE)).encode("ascii", "ignore").decode().lower()


def words(text) -> str:
    return " ".join(findall(r"[a-z0-9]+", fold(text)))


def alnum(text) -> str:
    return "".join(findall(r"[a-z0-9]+", fold(text)))


def squash(text) -> str:
    """alnum with OCR look-alikes folded (0/o, 1/i/l, 5/s)."""
    return alnum(text).translate(CONFUSABLE)


def ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b, autojunk=False).ratio() if a and b else 0.0


def window_ratio(long: str, short: str) -> float:
    """short against long's best same-length window."""
    if len(long) <= len(short):
        return ratio(long, short)
    matcher, best = SequenceMatcher(None, autojunk=False), 0.0
    matcher.set_seq2(short)
    for i in range(len(long) - len(short) + 1):
        matcher.set_seq1(long[i:i + len(short)])
        best = max(best, matcher.ratio())
    return best


def trigrams(form: str) -> set:
    padded = f"^{form}$"
    return {padded[i:i + 3] for i in range(len(padded) - 2)}


def ramp(value: float, low: float, high: float) -> float:
    return max(0.0, min(1.0, (value - low) / (high - low)))


def bits(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def hash_distance(x: dict, y: dict) -> float:
    """Two hash sets' distance in bits of 64 (pHash, dHash and the 256-bit pHash / 4, averaged)."""
    return (bits(x["p"], y["p"]) + bits(x["d"], y["d"]) + bits(x["q"], y["q"]) / 4) / 3


def hash_score(distance) -> float:
    return 0.0 if distance is None else ramp(HASH_ZERO - distance, 0, HASH_ZERO - HASH_FULL)


def read_text(lines) -> str:
    """An image's OCR'd lines (text, confidence, ...) as one squashed string."""
    return squash(" ".join(line[0] for line in lines or [] if line[1] >= 0.3))


def grams(text: str, n: int = 4) -> set:
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def novel_run(text: str, known: set) -> int:
    """The longest run of text's 4-grams not in known, in characters past the first gram."""
    best = run = 0
    for i in range(len(text) - 3):
        run = 0 if text[i:i + 4] in known else run + 1
        best = max(best, run)
    return best


def excerpt(text, limit: int = 240) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def read_json(path: str, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return load(f)
    except (OSError, ValueError):
        return default


def write_text(path: str, text: str):
    """Atomic: a temp file renamed over path."""
    makedirs(dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        f.write(text)
    replace(path + ".tmp", path)


def write_json(path: str, data, indent: Optional[int] = 1):
    write_text(path, dumps(data, ensure_ascii=False, indent=indent) + "\n")


def git_root(path: str) -> Optional[str]:
    """The git work tree holding path, if any."""
    current = realpath(expanduser(path))
    while True:
        if exists(join(current, ".git")):
            return current
        parent = dirname(current)
        if parent == current:
            return None
        current = parent


def check_root(root: str) -> str:
    """The archive root, refused inside a git repo."""
    root = realpath(expanduser(root))
    if not isdir(root):
        raise SystemExit(f"no archive at {root}: run the miner first (python -m commands.discord_archive)")
    repo = git_root(root)
    if repo:
        raise SystemExit(f"{root} is inside the git repo {repo}: the archive must stay outside every repo")
    return root


class Paths:
    """Where things go under the archive root."""
    def __init__(self, root: str):
        self.root, self.match = root, join(root, MATCH)
        self.site, self.thumbs, self.cache = join(self.match, "site"), join(self.match, "thumbs"), join(self.match, "cache")
        self.card_cache, self.image_cache = join(self.cache, "cards.jsonl"), join(self.cache, "images.jsonl")
        self.pairs, self.rulings = join(self.match, "pairs.json"), join(self.match, "rulings.json")
        self.ids, self.selftest = join(self.match, "ambiguous_ids.json"), join(self.match, "selftest.json")
        self.lock = join(self.match, ".lock")
        self.matches, self.ambiguous = join(root, "matches.json"), join(root, "ambiguous.json")
        self.html, self.report, self.script = join(root, "ambiguous.html"), join(root, "REPORT.md"), join(root, "apply_rulings.py")
        self.proposal = join(root, "proposal")

    def rel(self, path: str) -> str:
        return path[len(self.root) + 1:] if path.startswith(self.root + "/") else path


def hold(path: str):
    """One matcher run at a time per root."""
    makedirs(dirname(path), exist_ok=True)
    handle = open(path, "a")
    try:
        flock(handle.fileno(), LOCK_EX | LOCK_NB)
    except OSError:
        handle.close()
        raise SystemExit(f"another matcher run holds {path}") from None
    return handle


class Cache:
    """A JSONL file of {key, ...}: the last line of a key wins; a torn last line is dropped."""
    def __init__(self, path: str):
        self.path, self.entries = path, {}
        if not exists(path):
            return
        with open(path, "rb") as f:
            data = f.read()
        if data and not data.endswith(b"\n"):
            data = data[:data.rfind(b"\n") + 1]
            with open(path, "wb") as f:
                f.write(data)
        for line in data.decode("utf-8", "replace").splitlines():
            try:
                entry = loads(line)
                self.entries[entry["key"]] = entry
            except (ValueError, KeyError, TypeError):
                continue

    def fresh(self, key: str, ocr: bool, version: int) -> bool:
        entry = self.entries.get(key)
        if not entry or entry.get("v") != version or entry.get("retry"):
            return False
        return bool(entry.get("ocr")) or not ocr or bool(entry.get("error"))

    def add(self, entry: dict):
        makedirs(dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(dumps(entry, ensure_ascii=False) + "\n")
        self.entries[entry["key"]] = entry


# ========================
# Images (Pillow and numpy, imported when used)
# ========================
_DCT: dict = {}
_VISION = None


def _pil():
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    return Image


def _np():
    import numpy
    return numpy


def load_image(source):
    """An RGB image: the first frame, transparency flattened, the long side at most WORK_SIDE."""
    Image = _pil()
    with Image.open(source) as im:                # opens on the first frame (a PSD: its composite)
        im.draft("RGB", (WORK_SIDE, WORK_SIDE))
        im = im.convert("RGBA")
    if max(im.size) > WORK_SIDE:
        im.thumbnail((WORK_SIDE, WORK_SIDE), Image.LANCZOS)
    flat = Image.new("RGBA", im.size, BACKGROUND)
    flat.alpha_composite(im)
    return flat.convert("RGB")


def trim(img):
    """img without uniform borders (the corners' color), at most TRIM_MAX of a side."""
    np = _np()
    a = np.asarray(img, dtype=np.int16)
    h, w = a.shape[0], a.shape[1]
    if h < 8 or w < 8:
        return img
    corners = np.array([a[0, 0], a[0, w - 1], a[h - 1, 0], a[h - 1, w - 1]])
    border = np.median(corners, axis=0)
    if np.abs(corners - border).max() > TRIM_CORNERS:     # corners that differ: no padding around it
        return img
    unlike = np.abs(a - border).max(axis=2) > TRIM_TOL
    rows, cols = np.flatnonzero(unlike.mean(axis=1) > 0.01), np.flatnonzero(unlike.mean(axis=0) > 0.01)
    if not rows.size or not cols.size:
        return img
    top, bottom = min(int(rows[0]), int(h * TRIM_MAX)), max(int(rows[len(rows) - 1]) + 1, int(h * (1 - TRIM_MAX)))
    left, right = min(int(cols[0]), int(w * TRIM_MAX)), max(int(cols[len(cols) - 1]) + 1, int(w * (1 - TRIM_MAX)))
    return img if (left, top, right, bottom) == (0, 0, w, h) else img.crop((left, top, right, bottom))


def shape_of(w: int, h: int, force: bool = False) -> Optional[str]:
    """The card shape an image's aspect fits (force: the nearest)."""
    if not w or not h:
        return None
    r = w / h
    best = min(SHAPES, key=lambda s: abs(SHAPES[s] - r))
    return best if force or abs(SHAPES[best] - r) <= SHAPE_TOL else None


def _dct(n: int):
    np = _np()
    if n not in _DCT:
        k, i = np.arange(n)[:, None], np.arange(n)[None, :]
        m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2 / n)
        m[0] /= np.sqrt(2)
        _DCT[n] = m
    return _DCT[n]


def _hex(bitmap) -> str:
    np = _np()
    return np.packbits(bitmap.astype(np.uint8).ravel()).tobytes().hex()


def hashes(img) -> dict:
    """pHash (64 bit), dHash (64) and pHash (256) of an image, hex."""
    np, Image = _np(), _pil()
    gray = img.convert("L")
    found = {}
    for key, size in (("p", 8), ("q", 16)):
        n = size * 4
        m = _dct(n)
        low = (m @ np.asarray(gray.resize((n, n), Image.LANCZOS), dtype=np.float64) @ m.T)[:size, :size].ravel()
        found[key] = _hex(low > np.median(low[1:]))
    a = np.asarray(gray.resize((9, 8), Image.LANCZOS), dtype=np.int16)
    found["d"] = _hex(a[:, 1:] > a[:, :8])
    return found


def card_box(w: int, h: int, box: tuple) -> tuple:
    return (round(box[0] * w), min(h, round(box[1] * w)), round(box[2] * w), min(h, round(box[3] * w)))


def _vision():
    """(Vision, NSData, autorelease_pool), or False without pyobjc."""
    global _VISION
    if _VISION is None:
        try:
            import Vision
            from Foundation import NSData
            from objc import autorelease_pool
            _VISION = (Vision, NSData, autorelease_pool)
        except ImportError:
            _VISION = False
    return _VISION


def ocr_lines(img) -> list:
    """macOS Vision's accurate text lines: [text, confidence, x, y, w, h] (fractions, origin top left)."""
    kit = _vision()
    if not kit:
        return []
    Vision, NSData, pool = kit
    buf = BytesIO()
    img.save(buf, "PNG", compress_level=1)
    data = buf.getvalue()
    found = []
    with pool():
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
            NSData.dataWithBytes_length_(data, len(data)), None)
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setRecognitionLanguages_(["en-US"])
        request.setUsesLanguageCorrection_(False)
        ok, _ = handler.performRequests_error_([request], None)
        for result in (request.results() or []) if ok else []:
            top = result.topCandidates_(1)
            if not top:
                continue
            box = result.boundingBox()
            found.append([str(top[0].string()), round(float(top[0].confidence()), 3), round(box.origin.x, 4),
                          round(1 - box.origin.y - box.size.height, 4), round(box.size.width, 4),
                          round(box.size.height, 4)])
    return sorted(found, key=lambda line: (line[3], line[2]))


def anchors(img, lines: list) -> list:
    """Art hashes where a card would sit under each title-like line (screenshots)."""
    w, h = img.size
    found, used = [], set()
    for i, (text, conf, x, y, lw, lh) in enumerate(lines):
        if conf < 0.3 or not 4 <= len(alnum(text)) <= 40:
            continue
        height, centre, top = lh * h, (x + lw / 2) * w, y * h
        for share in TITLE_TEXT:
            for row in TITLE_ROWS:
                cw = height / share
                left, ctop = centre - cw / 2, top + height / 2 - row * cw
                x0, y0, x1, y1 = left + ART[0] * cw, ctop + ART[1] * cw, left + ART[2] * cw, ctop + ART[3] * cw
                if x0 < -0.05 * cw or y0 < 0 or x1 > w + 0.05 * cw or y1 > h or x1 - x0 < 48:
                    continue
                found.append({"line": i, "art": hashes(img.crop((max(0, round(x0)), round(y0), min(w, round(x1)),
                                                                 round(y1))))})
                used.add(i)
        if len(used) >= ANCHOR_LINES:
            break
    return found


def features(img, ocr: bool = True, body: bool = True, force_shape: bool = False, reread: bool = False) -> dict:
    """An image's hashes (whole, art), title band and whole-image OCR, and screenshot anchors; reread: the text
    read again at a post's size (card versions: misreads differ by size)."""
    img = trim(img)
    w, h = img.size
    shape = shape_of(w, h, force_shape)
    found = {"w": w, "h": h, "shape": shape, "whole": hashes(img)}
    if shape:
        found["art"] = hashes(img.crop(card_box(w, h, ART)))
    if ocr and _vision():
        found["ocr"] = True
        if shape:
            band = img.crop((0, 0, w, min(h, round(w * TITLE_BAND))))
            if w < 900:
                band = band.resize((w * 2, band.height * 2))
            found["title"] = [[text, conf] for text, conf, *_ in ocr_lines(band)]
        if body:
            small = img
            if max(w, h) > OCR_SIDE:
                small = img.copy()
                small.thumbnail((OCR_SIDE, OCR_SIDE))
            lines = ocr_lines(small)
            found["lines"] = lines[:80]
            if not shape:
                found["anchors"] = anchors(img, lines)
            if reread:
                found["rereads"] = [[line[:2] for line in ocr_lines(degrade(img, width, quality))][:80]
                                    for width, quality in REREADS]
    return found


def save_thumb(img, path: str, width: int):
    thumb = img.copy()
    thumb.thumbnail((width, width * 3))
    makedirs(dirname(path), exist_ok=True)
    thumb.save(path + ".tmp", "JPEG", quality=82)
    replace(path + ".tmp", path)


def post_task(args: tuple) -> dict:
    """A worker: one stored image's features (and its thumbnail)."""
    key, path, thumb, ocr = args
    try:
        img = load_image(path)
        save_thumb(img, thumb, THUMB_POST)
        return {"key": key, "v": POST_FEATURES, **features(img, ocr)}
    except Exception as e:                      # a bad file is noted, never fatal
        return {"key": key, "v": POST_FEATURES, "error": f"{type(e).__name__}: {str(e)[:160]}"}


class Missing(Exception):
    """A card version's file: absent at that commit, or not fetched (retry)."""
    def __init__(self, text: str, retry: bool = False):
        super().__init__(text)
        self.retry = retry


def open_source(source: str):
    """A card version's image: a local path, or its bytes fetched from GitHub by commit."""
    if not source.startswith("https://"):
        return source
    from requests import get
    for _ in range(3):
        try:
            reply = get(source, timeout=90, headers={"User-Agent": USER_AGENT})
        except Exception:
            continue
        if reply.status_code == 200:
            return BytesIO(reply.content)
        if reply.status_code == 404:
            raise Missing("404 at that commit")
    raise Missing("download failed", retry=True)


def card_task(args: tuple) -> dict:
    """A worker: one card version's features (a local PNG, or one fetched from GitHub by commit)."""
    key, source, thumb, ocr = args
    try:
        img = load_image(open_source(source))
        save_thumb(img, thumb, THUMB_CARD)
        return {"key": key, "v": CARD_FEATURES, **features(img, ocr, force_shape=True, reread=True)}
    except Missing as e:
        return {"key": key, "v": CARD_FEATURES, "error": str(e), **({"retry": True} if e.retry else {})}
    except Exception as e:
        return {"key": key, "v": CARD_FEATURES, "error": f"{type(e).__name__}: {str(e)[:160]}"}


def run_tasks(task_name: str, jobs: list, workers: int, done: Callable, log: Callable, label: str):
    """jobs through a spawn pool (Vision isn't fork-safe); done(result) as each finishes."""
    if not jobs:
        return
    from commands import discord_match as module          # importable by name: never __main__
    task, started = getattr(module, task_name), time()
    if workers <= 1:
        for i, job in enumerate(jobs, 1):
            done(task(job))
            if i % 200 == 0:
                log(f"{label}: {i}/{len(jobs)}")
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(task, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            done(future.result())
            if i % 200 == 0 or i == len(jobs):
                log(f"{label}: {i}/{len(jobs)} ({time() - started:.0f} s)")


# ========================
# Cards
# ========================
def fetch_site(paths: Paths, site: str, cards_repo: str, refresh: bool, log: Callable) -> str:
    """The site's cards.json and versions.json in match/site/ (fetched; else the checkout's site/dist/data)."""
    makedirs(paths.site, exist_ok=True)
    stale = refresh or any(not exists(join(paths.site, n)) or time() - getmtime(join(paths.site, n)) > SITE_MAX_AGE
                           for n in SITE_FILES)
    if stale and site:
        try:
            from requests import get
            for name in SITE_FILES:
                reply = get(f"{site.rstrip('/')}/data/{name}", timeout=60, headers={"User-Agent": USER_AGENT})
                reply.raise_for_status()
                reply.json()
                write_text(join(paths.site, name), reply.text)
            log(f"site data fetched from {site}")
        except Exception as e:
            log(f"site data not fetched ({type(e).__name__}); using what is there")
    for name in SITE_FILES:
        local = join(expanduser(cards_repo), "site", "dist", "data", name)
        if not exists(join(paths.site, name)) and exists(local):
            with open(local, encoding="utf-8") as f:
                write_text(join(paths.site, name), f.read())
    return paths.site


CARD_FIELDS = ("id", "name", "key", "aliases", "png", "psd", "author", "authorListed", "fs", "series", "subseries",
               "type", "ability", "dup", "dups", "stars")


def load_cards(site_dir: str) -> dict:
    data = read_json(join(site_dir, "cards.json"), {}) or {}
    return {c["id"]: {k: c.get(k) for k in CARD_FIELDS if c.get(k) not in (None, [], "")}
            for c in data.get("cards") or [] if isinstance(c, dict) and c.get("id")}


def file_name(path: str) -> str:
    return splitext(basename(path or ""))[0].replace("_", " ")


def card_names(card: dict) -> list:
    """(text, kind): printed title, key, aliases, file names."""
    found = [(card.get("name"), "title"), (card.get("key"), "key")]
    found += [(a, "alias") for a in card.get("aliases") or []]
    found += [(file_name(card.get(k)), "file") for k in ("png", "psd") if card.get(k)]
    return [(text, kind) for text, kind in found if text]


def plan_versions(cards: dict, site_dir: str, cards_repo: str, mode: str) -> list:
    """Card versions to featurize: {key, card, source, commit, ts, current, path}."""
    repo, plan = expanduser(cards_repo), []
    versions = read_json(join(site_dir, "versions.json"), {}) or {}
    commits = versions.get("commits") or []
    for cid, card in sorted(cards.items()):
        local = join(repo, card["png"]) if card.get("png") else None
        rows = (versions.get("cards") or {}).get(cid) or []
        if local and isfile(local):
            with open(local, "rb") as f:
                digest = sha256(f.read()).hexdigest()[:20]
            newest = commits[rows[0][0]] if rows and rows[0][0] < len(commits) else None
            plan.append({"key": "f" + digest, "card": cid, "source": local, "path": card["png"], "current": True,
                         "commit": newest[0] if newest else None, "ts": newest[1] if newest else card.get("fs")})
        old = rows[1:] if mode == "all" else rows[len(rows) - 1:] if mode == "oldest" and len(rows) > 1 else []
        for row in old:
            if not row or row[0] >= len(commits):
                continue
            sha, ts = commits[row[0]][0], commits[row[0]][1]
            path = row[1] if len(row) > 1 else card.get("png")
            if not path:
                continue
            plan.append({"key": "g" + sha1(f"{sha}:{path}".encode()).hexdigest()[:20], "card": cid, "path": path,
                         "source": f"{RAW_BASE}{CARD_REPO}/{sha}/{quote(path)}", "current": False, "commit": sha,
                         "ts": ts})
    return plan


class Names:
    """Every card's names, for exact, fuzzy and in-text lookups (stdlib only)."""
    def __init__(self, cards: dict, extra: Optional[dict] = None):
        self.cards = cards
        self.forms = defaultdict(dict)          # squashed form -> {card: kind}
        self.phrases = defaultdict(dict)        # words form -> {card: kind}
        self.grams = defaultdict(set)
        self.card_forms = defaultdict(set)
        self.form_words = {}                    # squashed form -> its words (alnum)
        for cid, card in cards.items():
            for text, kind in card_names(card):
                self._add(cid, text, kind)
        for cid, texts in (extra or {}).items():   # OCR'd titles of the card's versions
            for text in texts:
                if cid in cards:
                    self._add(cid, text, "ocr", phrase=False)
        for form in self.forms:
            for gram in trigrams(form):
                self.grams[gram].add(form)
        self.longest = min(8, max((len(p.split()) for p in self.phrases), default=1))

    def _add(self, cid: str, text: str, kind: str, phrase: bool = True):
        form = squash(text)
        if len(form) < 2:
            return
        self.forms[form].setdefault(cid, kind)
        self.card_forms[cid].add(form)
        self.form_words.setdefault(form, words(text).split())
        if phrase and words(text):
            self.phrases[words(text)].setdefault(cid, kind)

    def exact(self, text: str) -> dict:
        return dict(self.forms.get(squash(text), {}))

    def fuzzy(self, text: str, floor: float, limit: int = 5, partial: bool = False) -> dict:
        """{card: (ratio, form)} for names like text."""
        q = squash(text)
        if len(q) < 3:
            return {}
        if q in self.forms:
            return {cid: (1.0, q) for cid in self.forms[q]}
        grams = trigrams(q)
        counts = Counter(form for gram in grams for form in self.grams.get(gram, ()))
        best: dict = {}
        for form, n in counts.most_common(40):
            if n < max(1, len(grams) // 4):
                break
            r = ratio(q, form)
            if partial and len(form) >= 6 and len(q) > len(form) + 2:
                r = max(r, 0.95 * window_ratio(q, form))
            if r >= floor:
                for cid in self.forms[form]:
                    if r > best.get(cid, (0, ""))[0]:
                        best[cid] = (r, form)
        return dict(sorted(best.items(), key=lambda kv: -kv[1][0])[:limit])

    def best_ratio(self, cid: str, texts: list, swaps: bool = False) -> tuple:
        """(ratio, text, form) of texts against one card's names; ties go to the longer text. swaps: a text with a
        word of its own where the name has another (Cape of the... vs Heir of the...) isn't the name."""
        found = (0.0, "", "")
        for text in texts:
            q = squash(text)
            for form in self.card_forms.get(cid, ()):
                if swaps and swapped(text, self.form_words.get(form) or []):
                    continue
                r = ratio(q, form)
                if r > found[0] or (r == found[0] and r and len(q) > len(squash(found[1]))):
                    found = (r, text, form)
        return found

    def part_ratio(self, cid: str, texts: list, swaps: bool = False) -> tuple:
        """(credit, text, form) for a text that is a piece of one card's name ("SWORD" of Dark Sword): its best
        window, scaled by the share of the name it covers."""
        found = (0.0, "", "")
        for text in texts:
            q = squash(text)
            if len(q) < 4:
                continue
            for form in self.card_forms.get(cid, ()):
                if swaps and swapped(text, self.form_words.get(form) or []):
                    continue
                if len(form) > len(q):
                    credit = window_ratio(form, q) * (len(q) / len(form)) ** 0.5
                    if credit > found[0]:
                        found = (credit, text, form)
        return found

    def mentions(self, text: str) -> dict:
        """{card: phrase} for names written word for word in text (one-word names of 4+ letters)."""
        tokens = words(text).split()
        found: dict = {}
        taken = [False] * len(tokens)               # words a longer name already used
        for n in range(min(self.longest, len(tokens)), 0, -1):
            for i in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[i:i + n])
                if (n == 1 and len(phrase) < 4) or all(taken[i:i + n]) or phrase not in self.phrases:
                    continue
                for cid in self.phrases[phrase]:
                    found.setdefault(cid, phrase)
                for j in range(i, i + n):
                    taken[j] = True
        return found


def swapped(text: str, name_words: list) -> bool:
    """A read with a word unlike every word of a name, and the name with one unlike every word read (Cape of the
    Old King vs Heir of the Old King, Fire Lance vs Flame Lance): another name, not a misread of it. Words both
    have aren't compared; words unfolded (squash's i -> l would make fire flame-like)."""
    shared = set(words(text).split()) & set(name_words)        # words both have aren't compared
    mine = [w for w in words(text).split() if len(w) >= 3 and w not in shared]
    theirs = [w for w in name_words if len(w) >= 3 and w not in shared]

    def lone(xs: list, ys: list) -> bool:
        return any(all(ratio(x, y) < SWAP_WORD for y in ys) for x in xs)
    return bool(mine and theirs) and lone(mine, theirs) and lone(theirs, mine)


def wordy(text: str) -> bool:
    """Reads as words: two letters or more, nearly all of it letters."""
    chars = sub(r"\s", "", str(text or ""))
    letters = sum(c.isalpha() for c in chars)
    return letters >= 2 and letters >= 0.7 * len(chars)


def card_like(texts: list) -> bool:
    """OCR'd lines that read like a card: enough text, or a stat label (HEALTH, +8 ATK)."""
    return len(alnum(" ".join(texts))) >= ART_TEXT or any(set(words(t).split()) & STAT_WORDS for t in texts)


def strip_tokens(text: str) -> str:
    return sub(r"\[[^\]]*\]", " ", str(text or ""))


class Bodies:
    """Ability text by card: OCR'd words scored by their rare-word share of a card's ability."""
    def __init__(self, cards: dict):
        docs = {cid: {w for w in words(strip_tokens(c.get("ability"))).split() if len(w) >= 3 and not w.isdigit()}
                for cid, c in cards.items()}
        df = Counter(w for doc in docs.values() for w in doc)
        n = max(1, len(docs))
        common = max(3, n * 0.08)
        self.idf = {w: log(n / k) for w, k in df.items()}
        self.docs = {cid: {w for w in doc if df[w] <= common} for cid, doc in docs.items()}
        self.weight = {cid: sum(self.idf[w] for w in doc) for cid, doc in self.docs.items()}
        self.post = defaultdict(set)
        for cid, doc in self.docs.items():
            for w in doc:
                self.post[w].add(cid)

    def scores(self, text: str) -> dict:
        """{card: (score, shared words)}."""
        found, shared = defaultdict(float), Counter()
        for w in set(words(text).split()):
            for cid in self.post.get(w, ()):
                found[cid] += self.idf[w]
                shared[cid] += 1
        return {cid: (ramp(v / self.weight[cid], BODY_FLOOR, BODY_FULL), shared[cid]) for cid, v in found.items()
                if shared[cid] >= BODY_WORDS and self.weight[cid] and ramp(v / self.weight[cid], BODY_FLOOR, BODY_FULL)}


def few_words(candidate: dict) -> float:
    """The share a body score counts for alone or as a vote: under BODY_SURE shared words, less (a short ability's
    few rare words turn up in another card's longer text)."""
    return min(1.0, ((candidate.get("why") or {}).get("body_words") or BODY_SURE) / BODY_SURE)


class CardIndex:
    """The cards, their versions' hashes (numpy), OCR'd text, looks, names and ability words, for scoring images."""
    def __init__(self, cards: dict, versions: list):
        np = _np()
        self.cards = cards
        self.versions = [v for v in versions if v.get("whole") and v["card"] in cards]
        self.ids = sorted(cards)
        self.slot = {cid: i for i, cid in enumerate(self.ids)}
        self.owner = np.array([self.slot[v["card"]] for v in self.versions], dtype=np.int64)
        self.arrays = {}
        for part in ("whole", "art"):
            for key, n in (("p", 8), ("d", 8), ("q", 32)):
                rows = [bytes.fromhex(v[part][key]) if v.get(part) else bytes(n) for v in self.versions]
                self.arrays[part, key] = np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(len(rows), n) \
                    if rows else np.zeros((0, n), dtype=np.uint8)
        self.has_art = np.array([bool(v.get("art")) for v in self.versions], dtype=bool)
        self.by_card = defaultdict(list)                # card -> its version indexes, current first, then newest
        for i, v in enumerate(self.versions):
            self.by_card[v["card"]].append(i)
        for vis in self.by_card.values():
            vis.sort(key=lambda i: (not self.versions[i].get("current"), -(self.versions[i].get("ts") or 0)))
        self.reads = [[t for t in [read_text(v.get("lines"))] + [read_text(r) for r in v.get("rereads") or []] if t]
                      for v in self.versions]               # each version read at several sizes: misreads differ
        self.union = [set().union(*(grams(t) for t in r)) for r in self.reads]     # in any read
        self.stable = [set.intersection(*(grams(t) for t in r)) if r else set() for r in self.reads]   # in all
        card_grams = defaultdict(set)
        for i, found in enumerate(self.union):
            card_grams[self.versions[i]["card"]] |= found
        df = Counter(g for found in card_grams.values() for g in found)
        self.common = {g for g, k in df.items() if k > max(3, COMMON_GRAMS * len(card_grams))}
        self.grams = [found - self.common for found in self.union]
        self.gram_cards = defaultdict(set)
        for cid, found in card_grams.items():
            for g in found - self.common:
                self.gram_cards[g].add(cid)
        self.look_of = self._looks()
        self.groups = defaultdict(dict)                 # card -> look -> (stable grams, any-read grams, versions)
        for cid, vis in self.by_card.items():
            for i in vis:
                stable, union, members = self.groups[cid].setdefault(self.look_of[i], (set(), set(), []))
                stable |= self.stable[i]
                union |= self.union[i]
                members.append(i)
        titles = defaultdict(list)
        for v in self.versions:
            text = " ".join(t for t, c in v.get("title") or [] if c >= 0.3)
            if len(alnum(text)) >= 3:
                titles[v["card"]].append(text)
        self.names = Names(cards, titles)
        self.bodies = Bodies(cards)

    def _looks(self) -> dict:
        """{version index: look}: versions no image could tell apart share a look (the whole-card hash within
        LOOK_BITS, and at most LOOK_GRAMS 4-grams in all of one's reads and none of the other's); 0 is the current
        version's, then by newest."""
        found = {}
        for vis in self.by_card.values():
            group = list(range(len(vis)))

            def top(a):
                while group[a] != a:
                    a = group[a]
                return a
            for a in range(len(vis)):
                for b in range(a + 1, len(vis)):
                    x, y = vis[a], vis[b]
                    if hash_distance(self.versions[x]["whole"], self.versions[y]["whole"]) > LOOK_BITS:
                        continue
                    if len(self.stable[x] - self.union[y]) > LOOK_GRAMS or \
                            len(self.stable[y] - self.union[x]) > LOOK_GRAMS:
                        continue
                    group[top(b)] = top(a)
            numbers: dict = {}
            for a, i in enumerate(vis):
                found[i] = numbers.setdefault(top(a), len(numbers))
        return found

    def looks(self) -> dict:
        """{version key: look}."""
        return {self.versions[i]["key"]: look for i, look in self.look_of.items()}

    def distances(self, hashset: dict, part: str):
        """Every version's distance to a hash set (bits of 64)."""
        np = _np()
        total = 0
        for key, scale in (("p", 1), ("d", 1), ("q", 4)):
            mine = np.frombuffer(bytes.fromhex(hashset[key]), dtype=np.uint8)
            total = total + np.bitwise_count(self.arrays[part, key] ^ mine).sum(axis=1) / scale
        return total / 3

    def scan(self, feat: dict) -> dict:
        """{card: (distance, version index, part)}: each card's nearest version, whole or art (ties: by whole)."""
        np = _np()
        if not self.versions or not feat.get("whole"):
            return {}
        whole = self.distances(feat["whole"], "whole")
        best = whole
        part = np.zeros(len(whole), dtype=bool)
        if feat.get("art"):
            art = np.where(self.has_art, self.distances(feat["art"], "art"), np.inf)
            part = art < whole
            best = np.minimum(whole, art)
        found: dict = {}
        order = np.lexsort((whole, best))
        for i in order[:400]:
            cid = self.versions[i]["card"]
            if cid not in found:
                found[cid] = (float(best[i]), int(i), "art" if part[i] else "whole")
        return found

    def anchor_distance(self, cid: str, art: dict) -> Optional[float]:
        found = [hash_distance(art, self.versions[i]["art"]) for i in self.by_card.get(cid, ())
                 if self.versions[i].get("art")]
        return min(found) if found else None

    def text_candidates(self, found: set, limit: int = 3) -> list:
        """Cards whose versions' text has most of an image's 4-grams."""
        counts = Counter(cid for g in found for cid in self.gram_cards.get(g, ()))
        return [cid for cid, n in counts.most_common(limit) if n >= OCR_FLOOR * len(found)]

    def text_fit(self, cid: str, found: set) -> float:
        """The share of an image's 4-grams in the card's nearest version."""
        return max((len(found & self.grams[i]) / len(found) for i in self.by_card.get(cid, ())), default=0.0) \
            if found else 0.0

    @staticmethod
    def beats(a: tuple, b: tuple, seen: set, whole_card: bool) -> int:
        """1 if an image's 4-grams say look a, not b; -1 the other way; 0 can't tell. Each look's own grams are
        those in all reads of it and in no read of the other; an uncropped image lacking most of one's counts too."""
        own_a, own_b = a[0] - b[1], b[0] - a[1]
        hit_a, hit_b = len(own_a & seen), len(own_b & seen)
        share_a, share_b = hit_a / len(own_a) if own_a else 0.0, hit_b / len(own_b) if own_b else 0.0
        if hit_a >= BEAT_HITS and hit_a > hit_b and share_a >= share_b + BEAT_SHARE:
            return 1
        if hit_b >= BEAT_HITS and hit_b > hit_a and share_b >= share_a + BEAT_SHARE:
            return -1
        if whole_card:
            if len(own_b) >= ABSENT_GRAMS and share_b < ABSENT_SHARE and share_a >= share_b and (
                    not own_a or share_a >= 0.5):
                return 1
            if len(own_a) >= ABSENT_GRAMS and share_a < ABSENT_SHARE and share_b >= share_a and (
                    not own_b or share_b >= 0.5):
                return -1
        return 0

    def pick_version(self, cid: str, feat: dict, text: str, skip: Optional[int] = None) -> dict:
        """Which of a card's looks a card-shaped image shows: by its OCR'd text when it has enough (the looks no
        other look beats), else by the whole-card hash. {keys: the versions of the looks left, nearest first; by:
        text, hash, draft (a look not in git) or None (can't tell); fit, novel, bits}. skip: leave a look out."""
        groups = {look: g for look, g in self.groups.get(cid, {}).items() if look != skip}
        members = [i for look, g in groups.items() for i in g[2]]
        whole = {i: hash_distance(feat["whole"], self.versions[i]["whole"]) for i in members} \
            if feat.get("whole") else {}
        art = {i: hash_distance(feat["art"], self.versions[i]["art"]) for i in members
               if feat.get("art") and self.versions[i].get("art")}
        near_whole, near_art = min(whole.values(), default=99.0), min(art.values(), default=99.0)
        whole_card = near_whole <= HASH_FULL or near_art <= LOOK_BITS     # not a crop: the hashes mean something
        found = {"keys": [], "by": None, "bits": round(near_whole, 2) if whole else None}
        readable = [look for look, g in groups.items() if g[1]]
        won: list = []
        if len(text) >= TEXT_MIN and readable:
            seen = grams(text)
            beaten = set()
            for n, a in enumerate(readable):
                for b in readable[n + 1:]:
                    result = self.beats(groups[a], groups[b], seen, whole_card)
                    if result:
                        beaten.add(b if result > 0 else a)
            won = [look for look in readable if look not in beaten]
            if whole_card and whole and len(won) > 1:                     # same text: the art may differ
                near = {look: min(whole[i] for i in groups[look][2]) for look in won}
                won = [look for look in won if near[look] <= min(near.values()) + HASH_TIE]
            known = set().union(*(groups[look][1] for look in won))
            found.update(fit=round(len(seen & known) / max(1, len(seen)), 3), novel=novel_run(text, known))
            if whole_card and found["novel"] >= DRAFT_RUN:
                won, found["by"] = [], "draft"
            else:
                found["by"] = "text"
        elif whole and near_whole <= LOOK_BITS:
            won = list({self.look_of[i] for i in members if whole[i] <= near_whole + HASH_TIE})
            found["by"] = "hash"
        elif near_art <= LOOK_BITS and near_whole >= DRAFT_BITS:
            found["by"] = "draft"
        keys = [i for look in won for i in groups[look][2]]
        keys.sort(key=lambda i: (whole.get(i, 99.0), not self.versions[i].get("current")))
        found["keys"] = [self.versions[i]["key"] for i in keys]
        return found


# ========================
# Posts
# ========================
def media_entries(record: dict) -> list:
    """A message's stored images: (media id, entry, kind)."""
    found = [(a.get("id"), a, "attachment") for a in record.get("attachments") or []]
    for embed in record.get("embeds") or []:
        found += [(m.get("id"), m, "embed") for m in embed.get("media") or []]
    for forwarded in record.get("forwarded") or []:
        found += [(a.get("id"), a, "forwarded") for a in forwarded.get("attachments") or []]
        for embed in forwarded.get("embeds") or []:
            found += [(m.get("id"), m, "forwarded") for m in embed.get("media") or []]
    return [(mid, entry, kind) for mid, entry, kind in found if isinstance(entry, dict) and mid]


def is_image(stored: str, entry: dict, kind: str) -> bool:
    """An image by its extension, its content type, or (stored as .bin) an embed's image or thumbnail."""
    return splitext(stored)[1][1:].lower() in IMAGE_EXTS or (entry.get("content_type") or "").startswith("image/") \
        or (kind != "attachment" and entry.get("kind") in ("image", "thumbnail"))


def load_posts(root: str) -> tuple:
    """(records by id, images): every stored message, and every stored image with its message."""
    records, images = {}, []
    for path in sorted(glob(join(root, "messages", "*.jsonl"))):
        container = basename(path)[:-len(".jsonl")]
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.endswith("\n"):
                    break
                try:
                    record = loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict) and record.get("id"):
                    record["_container"] = container
                    records[record["id"]] = record
    for record in sorted(records.values(), key=lambda r: int(r["id"])):
        for mid, entry, kind in media_entries(record):
            stored = entry.get("file")
            if not stored or not entry.get("sha256") or not is_image(stored, entry, kind):
                continue
            images.append({"key": f"{record['id']}:{mid}", "message": record["id"], "media": mid,
                           "file": stored, "sha256": entry["sha256"], "filename": entry.get("filename"),
                           "kind": kind})
    return records, images


def is_bot(record: dict) -> bool:
    return bool((record.get("author") or {}).get("bot") or record.get("webhook_id"))


def content_of(record: dict) -> str:
    return "\n".join([record.get("content") or ""] + [f.get("content") or "" for f in record.get("forwarded") or []])


class Context:
    """Messages around an image: thread names, replies, the poster's nearby messages."""
    def __init__(self, records: dict):
        self.records = records
        self.replies = defaultdict(list)
        self.threads = {}                       # thread id -> name
        self.thread_members = defaultdict(list)
        containers = defaultdict(list)
        for record in records.values():
            if record.get("referenced_message_id"):
                self.replies[record["referenced_message_id"]].append(record)
            thread = record.get("thread") or {}
            if thread.get("id"):
                self.threads[thread["id"]] = thread.get("name")
                self.thread_members[thread["id"]].append(record)
            containers[record["_container"]].append(record)
        self.place = {}
        self.containers = {}
        for container, items in containers.items():
            items.sort(key=lambda r: int(r["id"]))
            self.containers[container] = items
            for i, record in enumerate(items):
                self.place[record["id"]] = (container, i)

    def thread_name(self, record: dict) -> Optional[str]:
        """The thread it's in, or the thread started from it."""
        return (record.get("thread") or {}).get("name") or self.threads.get(record["id"])

    def nearby(self, record: dict, seconds: float = MENTION_ROUND) -> list:
        """The poster's messages just before and after it."""
        container, i = self.place.get(record["id"], (None, 0))
        items, found = self.containers.get(container, []), []
        mine, when = (record.get("author") or {}).get("id"), epoch(record.get("timestamp"))
        for j in list(range(i - 1, max(-1, i - 4), -1)) + list(range(i + 1, min(len(items), i + 4))):
            other = items[j]
            if (other.get("author") or {}).get("id") == mine and when is not None \
                    and abs((epoch(other.get("timestamp")) or 0) - when) <= seconds:
                found.append(other)
        return found

    def discussion(self, record: dict) -> list:
        """Replies to it and its thread's messages."""
        return self.replies.get(record["id"], []) + [r for r in self.thread_members.get(record["id"], [])
                                                     if r["id"] != record["id"]]


def text_scores(names: Names, record: dict, ctx: Context) -> dict:
    """{card: (score, where, phrase)}: names in the message, its thread, the poster's nearby messages, replies."""
    found: dict = {}

    def add(cid, score, where, phrase):
        if score > found.get(cid, (0,))[0]:
            found[cid] = (round(score, 3), where, phrase)

    content = content_of(record)
    count = max(1, len(words(content).split()))
    for cid, phrase in names.mentions(content).items():
        n = len(phrase.split())
        add(cid, 0.6 if n >= 0.6 * count and len(phrase) >= 5 else 0.5 if n > 1 else 0.25, "message", phrase)
    thread = ctx.thread_name(record)
    if thread:
        for cid, (r, form) in names.fuzzy(thread, 0.86, limit=3).items():
            add(cid, 0.5, "thread", thread)
        for cid, phrase in names.mentions(thread).items():
            add(cid, 0.4 if " " in phrase else 0.2, "thread", phrase)
    for other in ctx.nearby(record):
        text = content_of(other)
        size = max(1, len(words(text).split()))
        for cid, phrase in names.mentions(text).items():
            n = len(phrase.split())
            add(cid, 0.4 if n >= 0.6 * size and len(phrase) >= 5 else 0.3 if n > 1 else 0.1, "nearby", phrase)
    for other in ctx.replies.get(record["id"], []):
        for cid, phrase in names.mentions(content_of(other)).items():
            add(cid, 0.3 if " " in phrase else 0.1, "reply", phrase)
    if CREATION.search(content) or (thread and CREATION.search(thread)):
        for cid, (score, where, phrase) in list(found.items()):
            found[cid] = (round(min(0.65, score + 0.05), 3), where, phrase)
    return found


def file_stem(filename: str) -> str:
    """A file name without its extension and Discord's SPOILER_ prefixes."""
    return sub(r"^(?:SPOILER_)+", "", splitext(basename(filename or ""))[0], flags=IGNORECASE)


def random_id(token: str) -> bool:
    """An upload id like q7Rx2Tb or hvkdz41 (not a series code like PoD11ZOMG, nor Golem2)."""
    if fullmatch(r"(?i)(pod|btd)\d+[a-z]*", token):
        return False
    letters = sub(r"[^A-Za-z]", "", token)
    return bool(findall(r"\d[A-Za-z]|[A-Za-z]\d+[A-Za-z]", token)) or len(findall(r"[a-z][A-Z]", token)) >= 2 or (
        len(letters) >= 4 and not findall(r"[aeiouyAEIOUY]", letters))


def name_tokens(filename: str, numbered: bool = False) -> list:
    """A file name's words, version words dropped (numbered: numbers ending a word too, Golem2); [] for a generic
    name (image.png, IMG_2041, unknown-1, an upload id...)."""
    tokens = []
    for t in re_split(r"[\W_]+", file_stem(filename)):
        if t and not random_id(t):
            t = sub(r"(?<=[A-Za-z]{3})\d{1,2}$", "", t) if numbered else t
            tokens += [w for w in words(t).split() if not VERSION_WORD.fullmatch(w)]
    if all(t in GENERIC_FILE or t.isdigit() or fullmatch(r"[0-9a-f]{8,}|img\d+|dsc\d+|image\d+", t) for t in tokens):
        return []
    return tokens


def file_scores(names: Names, filename: str) -> dict:
    """{card: (score, stem)} from an attachment's file name (generic names give nothing)."""
    stem = splitext(basename(filename or ""))[0]
    found = {}
    for numbered in (False, True):              # Kit2: a name's own number first, else a version number
        tokens = name_tokens(filename, numbered)
        if not tokens or any(score >= 1.0 for score, _ in found.values()):
            break
        for cid, (r, form) in names.fuzzy("".join(tokens), 0.82, limit=3).items():
            score = 1.0 if r >= 0.999 else round(ramp(r, 0.8, 1.0), 3)
            if score > found.get(cid, (0.0, ""))[0]:
                found[cid] = (score, stem)
    return found


def title_blocks(feat: dict, parts: bool = False) -> list:
    """A card-shaped image's title printed on two or three lines (Cape of the / Old King): lines in the title
    band of one height, one under the other and centred alike, each reading as words, 40 characters at most in
    all; joined (parts: the lines)."""
    return [[r[0] for r in run] if parts else " ".join(r[0] for r in run) for run in title_runs(feat)]


def title_runs(feat: dict) -> list:
    """title_blocks' lines, with their geometry."""
    if not feat.get("shape"):
        return []
    band = TITLE_BAND * (feat.get("w") or 0) / max(1, feat.get("h") or 1)
    lines = [line for line in feat.get("lines") or [] if line[3] < band and line[1] >= 0.5 and wordy(line[0])]
    found, run = [], []
    for line in lines + [None]:
        last = run[len(run) - 1] if run else None
        if line and last and abs(line[2] + line[4] / 2 - last[2] - last[4] / 2) <= 0.1 and \
                min(line[5], last[5]) >= 0.7 * max(line[5], last[5]) and \
                -0.3 * last[5] <= line[3] - last[3] - last[5] <= last[5]:
            run.append(line)
            continue
        if 1 < len(run) <= 3 and len(alnum(" ".join(r[0] for r in run))) <= 40:
            found.append(run)
        run = [line] if line else []
    return found


def title_texts(feat: dict, names: Optional["Names"] = None) -> list:
    """A card-like image's title: the band's OCR and the whole image's lines in the band (either may misread);
    with names, also a line under the art that is nearly a card's name (2020's cards print the title there)."""
    found = [text for text, conf in feat.get("title") or [] if conf >= 0.3 and len(alnum(text)) >= 2]
    lines = feat.get("lines") or [] if feat.get("shape") else []
    band = TITLE_BAND * (feat.get("w") or 0) / max(1, feat.get("h") or 1)
    found += [line[0] for line in lines if line[3] < band and line[1] >= 0.3 and len(alnum(line[0])) >= 2
              and line[0] not in found]
    lone: list = []                             # one read per line (the band's and the whole image's), the fullest
    for text in found:
        q = squash(text)
        same = next((i for i, other in enumerate(lone) if ratio(q, squash(other)) >= 0.8 or q in squash(other)
                     or squash(other) in q), None)
        if same is None:
            lone.append(text)
        elif len(q) > len(squash(lone[same])):
            lone[same] = text
    lone = [t for i, t in enumerate(lone) if not any(j != i and squash(t) in squash(u) for j, u in enumerate(lone))]
    if len(lone) > 1:                           # a title on two lines, or noise around it (one line read
        found += [f"{a} {b}" for a, b in zip(lone, lone[1:])]                  # twice is no two-line title)
        found += [" ".join(lone)] if len(lone) > 2 else []
    found += [block for block in title_blocks(feat) if block not in found]
    if names is not None:                       # one to three lines in a row: titles wrap
        middle = [line[0] for line in lines if band <= line[3] < TITLE_LOW and line[1] >= 0.3]
        for i in range(len(middle)):
            for n in range(1, 4):
                part = middle[i:i + n]
                text = " ".join(part)
                if len(part) < n or text in found or not 4 <= len(alnum(text)) <= 40 or any(
                        squash(t) in STAT_WORDS or t.rstrip().endswith(":") for t in part):   # labels ("Cursed:")
                    continue
                if names.fuzzy(text, TITLE_LINE):
                    found.append(text)
    return found


def score_image(index: CardIndex, feat: dict, filename: str = "", record: Optional[dict] = None,
                ctx: Optional[Context] = None) -> list:
    """An image's candidate cards, each with every method's score (0..1) and why."""
    feat = feat or {}
    names, shape = index.names, feat.get("shape")
    nearest = index.scan(feat) if not feat.get("error") else {}
    pool = set(cid for cid, (d, _, _) in sorted(nearest.items(), key=lambda kv: kv[1][0])[:5] if d < HASH_ZERO)
    titles = title_texts(feat, names) if shape else []
    for text in titles:
        pool.update(names.fuzzy(text, TITLE_FLOOR))
    lines: dict = {}                            # card -> (ratio, text, line indexes)
    if not shape:
        heights = sorted(line[5] for line in feat.get("lines") or [])
        middle = heights[len(heights) // 2] if heights else 0
        for i, line in enumerate(feat.get("lines") or []):
            text, conf = line[0], line[1]
            if conf < 0.3 or not 4 <= len(alnum(text)) <= 48:
                continue
            tall = line[5] >= TALL * middle         # title type is bigger than ability text
            for cid, (r, form) in names.fuzzy(text, LINE_FLOOR, limit=3, partial=tall).items():
                r = r if tall or r >= 0.95 else r * SMALL_LINE
                old = lines.get(cid, (0.0, "", []))
                lines[cid] = (max(r, old[0]), text if r > old[0] else old[1], old[2] + [i])
        pool.update(c for c, (r, _, _) in lines.items() if r >= LINE_FLOOR)
    body_text = " ".join(line[0] for line in feat.get("lines") or [] if line[0] not in titles)
    bodies = index.bodies.scores(body_text) if body_text else {}
    post_text = read_text(feat.get("lines")) if shape else ""
    post_grams = grams(post_text) - index.common if post_text else set()
    if len(post_grams) < OCR_GRAMS:
        post_grams = set()
    if shape:
        pool.update(c for c, _ in sorted(bodies.items(), key=lambda kv: -kv[1][0])[:3])
        pool.update(index.text_candidates(post_grams) if post_grams else ())
    files = file_scores(names, filename)
    pool.update(files)
    texts = text_scores(names, record, ctx) if record is not None and ctx is not None else {}
    pool.update(texts)
    reads = {}                                  # card -> (title ratio, text read, its name form)
    members = {t for block in title_blocks(feat, parts=True) for t in block}
    pieces = [t for t in titles if t not in members]        # a line of a two-line title isn't a piece of a name
    for cid in pool if titles else ():
        if cid in index.cards:
            reads[cid] = max(names.best_ratio(cid, titles, swaps=True), names.part_ratio(cid, pieces, swaps=True))
    whole = [(squash(text), form) for r, text, form in reads.values() if r >= TITLE_LINE]
    whole += [(squash(block), "") for block in title_blocks(feat) if set(words(block).split()) & PART_OF]
    for cid, (r, text, form) in list(reads.items()):
        for read, other in whole:               # a name that is only part of a read naming another card whole, or
            longer = len(form) < len(other) if other else len(form) + 2 < len(read)    # of a title on two lines
            if form and form != other and longer and form in read:
                r = min(r, r * len(form) / len(read))
        reads[cid] = (r, text, form)
    found = []
    for cid in pool:
        if cid not in index.cards:
            continue
        s, why = {}, {}
        if cid in nearest and nearest[cid][0] < HASH_ZERO:
            d, vi, part = nearest[cid]
            s["hash"] = hash_score(d)
            why.update(hash_bits=round(d, 2), hash_part=part, version=index.versions[vi]["key"])
        if cid in reads and reads[cid][0] >= TITLE_FLOOR:
            s["title"], why["title"] = ramp(reads[cid][0], TITLE_FLOOR, TITLE_FULL), reads[cid][1]
        if cid in lines:
            r, text, at = lines[cid]
            s["line"], why["line"] = ramp(r, LINE_FLOOR, LINE_FULL), text
            near = [index.anchor_distance(cid, a["art"]) for a in feat.get("anchors") or [] if a["line"] in at]
            near = [d for d in near if d is not None]
            if near and hash_score(min(near)):
                s["anchor"], why["anchor_bits"] = hash_score(min(near)), round(min(near), 2)
        if cid in bodies and (shape or s.get("line") or s.get("anchor")):   # several cards' text mixes
            s["body"], why["body_words"] = bodies[cid][0], bodies[cid][1]
        if post_grams:
            fit = index.text_fit(cid, post_grams)
            if fit >= OCR_FLOOR:
                s["ocr"], why["ocr_fit"] = ramp(fit, OCR_FLOOR, OCR_FULL), round(fit, 3)
        if cid in files:
            s["file"], why["file"] = files[cid]
        if cid in texts:
            s["text"], why["text_where"], why["text"] = texts[cid]
        s = {m: round(v, 3) for m, v in s.items() if v > 0}
        if "body" in s and set(s) <= {"body", "text"}:     # the ability's words alone: few count less
            s["body"] = round(s["body"] * few_words({"why": why}), 3)
        if s:
            found.append({"card": cid, "s": s, "why": why})
    alone = [c for c in found if c["s"].get("line") and all(v < 0.3 for m, v in c["s"].items() if m != "line")]
    read_lines = [line[0] for line in feat.get("lines") or []]
    if not shape and alone and (len(alone) > LIST_NAMES or not card_like(read_lines)):
        for c in alone:                         # names read, nothing shown: a list, a meme's caption
            c["s"]["line"], c["why"]["line_alone"] = round(c["s"]["line"] * LINE_ALONE, 3), True
    found = [c for c in found if max(WEIGHTS[m] * v for m, v in c["s"].items()) >= 0.12]
    found.sort(key=lambda c: -confidence(c["s"]))
    found = found[:CANDIDATES]
    for c in found if shape else ():            # the version shown, for the cards it may be
        if confidence(c["s"]) >= PLAUSIBLE:
            c["why"]["pick"] = index.pick_version(c["card"], feat, post_text)
            if c["why"]["pick"]["keys"]:
                c["why"]["version"] = c["why"]["pick"]["keys"][0]
    return found


def clusters(values: list, gap: float = 0.15) -> list:
    """Values grouped where neighbours are within gap."""
    found: list = []
    for v in sorted(values):
        if found and v - found[len(found) - 1][len(found[len(found) - 1]) - 1] <= gap:
            found[len(found) - 1].append(v)
        else:
            found.append([v])
    return found


def sheet_grid(feat: dict, names: Names) -> Optional[tuple]:
    """A card-shaped image showing several cards (a 2x2 sheet): two lines on one row, far apart, each
    just a card's name (different cards). (rows, columns) of the grid its card-shaped cells make, (0, 0) for
    cards in no grid; None for one card (its own title, and names in its text, are one column)."""
    if not feat.get("shape") or not feat.get("w") or not feat.get("h"):
        return None
    hits = []
    runs = [[" ".join(r[0] for r in run), min(r[1] for r in run), min(r[2] for r in run), run[0][3],
             max(r[2] + r[4] for r in run) - min(r[2] for r in run), run[0][5]] for run in title_runs(feat)]
    for text, conf, x, y, w, h in (feat.get("lines") or []) + runs:     # names on two lines too (The Old / King)
        if conf >= 0.5 and w <= SHEET_LINE and len(squash(text)) >= 4:
            found = names.fuzzy(text, TITLE_LINE, limit=1)
            if found:
                hits.append((x + w / 2, y + h / 2, next(iter(found))))
    pairs = [(a, b) for i, a in enumerate(hits) for b in hits[i + 1:] if abs(a[1] - b[1]) <= SHEET_ROW and
             abs(a[0] - b[0]) >= SHEET_GAP and not related(names.cards, a[2], b[2])]
    if not pairs:
        return None
    rows_seen = clusters([y for _, y, _ in hits])
    cols = max(len(clusters([x for x, y, _ in hits if y in row])) for row in rows_seen)
    fits = [r for r in range(1, 5) if 2 <= cols <= 4 and shape_of(feat["w"] / cols, feat["h"] / r)]
    if not fits:
        return (0, 0)
    return (min(fits, key=lambda r: (abs(r - len(rows_seen)), r)), cols)


def sheet_cells(path: str, grid: tuple) -> list:
    """A sheet's cells as cards: box (fractions of the trimmed image, as its OCR'd lines), size, shape, hashes."""
    img = trim(load_image(path))
    w, h = img.size
    rows, cols = grid
    found = []
    for i in range(rows):
        for j in range(cols):
            box = [round(j / cols, 4), round(i / rows, 4), round((j + 1) / cols, 4), round((i + 1) / rows, 4)]
            cell = trim(img.crop((round(box[0] * w), round(box[1] * h), round(box[2] * w), round(box[3] * h))))
            cw, ch = cell.size
            found.append({"box": box, "w": cw, "h": ch, "shape": shape_of(cw, ch, force=True), "whole": hashes(cell),
                          "art": hashes(cell.crop(card_box(cw, ch, ART)))})
    return found


def cell_feat(feat: dict, cell: dict) -> dict:
    """A sheet cell's features as a card's: its hashes, and the sheet's OCR'd lines inside it (cell coordinates)."""
    x0, y0, x1, y1 = cell["box"]
    lines = [[t, c, round((x - x0) / (x1 - x0), 4), round((y - y0) / (y1 - y0), 4), round(w / (x1 - x0), 4),
              round(h / (y1 - y0), 4)] for t, c, x, y, w, h in feat.get("lines") or []
             if x0 <= x + w / 2 < x1 and y0 <= y + h / 2 < y1]
    return {**{k: cell[k] for k in ("w", "h", "shape", "whole", "art")}, "ocr": feat.get("ocr"), "title": [],
            "lines": lines}


def twins(out: dict) -> dict:
    """{image key: [[key, whole bits, art bits]...]}: card-shaped images alike, the nearest first: the whole within
    HASH_FULL bits (TWIN_BITS: one image, posted again or re-saved) or the art within TWIN_BITS."""
    np = _np()
    keys = [k for k, i in out.items() if i.get("shape") and i.get("whole") and i.get("art") and not i.get("cells")]
    if len(keys) < 2:
        return {}
    arrays = {(part, k): np.frombuffer(b"".join(bytes.fromhex(out[key][part][k]) for key in keys), dtype=np.uint8)
              .reshape(len(keys), n) for part in ("whole", "art") for k, n in (("p", 8), ("d", 8), ("q", 32))}

    def bits(part, i):
        return (np.bitwise_count(arrays[part, "p"] ^ arrays[part, "p"][i]).sum(axis=1)
                + np.bitwise_count(arrays[part, "d"] ^ arrays[part, "d"][i]).sum(axis=1)
                + np.bitwise_count(arrays[part, "q"] ^ arrays[part, "q"][i]).sum(axis=1) / 4) / 3
    found = {}
    for i, key in enumerate(keys):
        whole, art = bits("whole", i), bits("art", i)
        near = np.flatnonzero((whole <= HASH_FULL) | (art <= TWIN_BITS))
        near = sorted((j for j in near if j != i), key=lambda j: (whole[j], art[j]))[:40]
        if near:
            found[key] = [[keys[j], round(float(whole[j]), 2), round(float(art[j]), 2)] for j in near]
    return found


def score(root: str, cards_repo: str = LOCAL_DIR_LOC, site: str = SITE_URL, versions: str = "all",
          ocr: bool = True, workers: int = 0, refresh: bool = False, limit: Optional[int] = None,
          log: Callable = print) -> dict:
    """Features for every card version and stored image, then each image's scored candidates (match/pairs.json)."""
    root = check_root(root)
    paths = Paths(root)
    lock = hold(paths.lock)
    try:
        workers = workers or max(1, (cpu_count() or 2) - 2)
        if ocr and not _vision():
            log("no Vision (pyobjc): running without OCR")
            ocr = False
        site_dir = fetch_site(paths, site, cards_repo, refresh, log)
        cards = load_cards(site_dir)
        if not cards:
            raise SystemExit(f"no cards in {site_dir}/cards.json")
        plan = plan_versions(cards, site_dir, cards_repo, versions)
        card_cache = Cache(paths.card_cache)
        jobs = [(v["key"], v["source"], join(paths.thumbs, f"c-{v['key']}.jpg"), ocr) for v in plan
                if not card_cache.fresh(v["key"], ocr, CARD_FEATURES)]
        log(f"{len(cards)} cards, {len(plan)} versions ({len(jobs)} to featurize)")
        run_tasks("card_task", jobs, workers, card_cache.add, log, "card versions")
        records, images = load_posts(root)
        if limit:
            images = images[:limit]
        image_cache = Cache(paths.image_cache)
        todo = {}
        for image in images:
            if not image_cache.fresh(image["sha256"], ocr, POST_FEATURES) and image["sha256"] not in todo:
                todo[image["sha256"]] = (image["sha256"], join(root, image["file"]),
                                         join(paths.thumbs, f"p-{image['sha256'][:24]}.jpg"), ocr)
        log(f"{len(records)} messages, {len(images)} images ({len(todo)} to featurize)")
        run_tasks("post_task", list(todo.values()), workers, image_cache.add, log, "images")
        versions_found = []
        for v in plan:
            entry = card_cache.entries.get(v["key"]) or {}
            if entry.get("whole"):
                versions_found.append({**{k: v[k] for k in ("key", "card", "commit", "ts", "current", "path")},
                                       **{k: entry.get(k) for k in ("whole", "art", "title", "shape", "lines",
                                                                    "rereads")}})
        index = CardIndex(cards, versions_found)
        looks = index.looks()
        ctx = Context(records)
        cell_cache, grids = Cache(join(paths.cache, "cells.jsonl")), {}
        for image in images:                    # sheets of several cards: each cell scored as a card
            feat = image_cache.entries.get(image["sha256"]) or {}
            grid = sheet_grid(feat, index.names) if not feat.get("error") else None
            if not grid or image["sha256"] in grids:
                continue
            grids[image["sha256"]] = grid
            entry = cell_cache.entries.get(image["sha256"]) or {}
            if grid != (0, 0) and (entry.get("v") != CELL_FEATURES or entry.get("grid") != list(grid)):
                try:
                    entry = {"cells": sheet_cells(join(root, image["file"]), grid)}
                except Exception as e:
                    entry = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
                cell_cache.add({"key": image["sha256"], "v": CELL_FEATURES, "grid": list(grid), **entry})
        log(f"sheets: {sum(1 for g in grids.values() if g != (0, 0))} in a grid, "
            f"{sum(1 for g in grids.values() if g == (0, 0))} of several cards in no grid")
        out = {}
        for n, image in enumerate(images, 1):
            feat = image_cache.entries.get(image["sha256"]) or {}
            record = records[image["message"]]
            grid = grids.get(image["sha256"])
            cells = (cell_cache.entries.get(image["sha256"]) or {}).get("cells") if grid and grid != (0, 0) else None
            if grid == (0, 0) or (grid and not cells):          # several cards, no grid: a screenshot's methods
                feat = {**feat, "shape": None}
            entry = out[image["key"]] = {
                **image, "thumb": f"{MATCH}/thumbs/p-{image['sha256'][:24]}.jpg", "error": feat.get("error"),
                **{k: feat.get(k) for k in ("w", "h", "shape", "whole", "art")},
                "ocr": bool(feat.get("ocr")), "title": title_texts(feat, index.names) if feat.get("shape") else [],
                "title_block": next(iter(title_blocks(feat)), None),
                "lines": [line[0] for line in feat.get("lines") or []][:40],
                "text": read_text(feat.get("lines")) if feat.get("shape") else "", "multi": bool(grid)}
            if cells:
                entry["sheet"], entry["cells"], merged = list(grid), [], {}
                for cell in cells:
                    part = cell_feat(feat, cell)
                    found = score_image(index, part)
                    entry["cells"].append({**{k: part[k] for k in ("w", "h", "shape", "whole", "art")},
                                           "box": cell["box"], "title": title_texts(part, index.names),
                                           "lines": [line[0] for line in part["lines"]][:20],
                                           "text": read_text(part["lines"]), "candidates": found})
                    for c in found:
                        if confidence(c["s"]) > confidence(merged.get(c["card"], {"s": {}})["s"]):
                            merged[c["card"]] = c
                entry["candidates"] = sorted(merged.values(), key=lambda c: -confidence(c["s"]))[:CANDIDATES]
            else:
                entry["candidates"] = score_image(index, feat, image.get("filename") or "", record, ctx)
            if n % 2000 == 0:
                log(f"scored {n}/{len(images)}")
        for key, near in twins(out).items():
            out[key]["twins"] = near
        thumbs = {v["key"]: f"{MATCH}/thumbs/c-{v['key']}.jpg" for v in versions_found}
        pairs = {"version": PAIRS_VERSION, "generated": now_iso(), "ocr": ocr, "versions_mode": versions,
                 "cards": cards, "versions": [{**{k: v[k] for k in ("key", "card", "commit", "ts", "current", "path",
                                                                     "whole", "art", "shape")},
                                               "title": " ".join(t for t, c in v.get("title") or [] if c >= 0.3),
                                               "look": looks.get(v["key"], 0),
                                               "thumb": thumbs[v["key"]]} for v in versions_found],
                 "images": out}
        write_json(paths.pairs, pairs, indent=None)
        log(f"pairs: {sum(1 for i in out.values() if i['candidates'])} of {len(out)} images have candidates")
    finally:
        lock.close()
    return build(root, cards_repo, log=log)


# ========================
# Verdicts
# ========================
def confidence(s: dict) -> float:
    """Methods combined as independent evidence: 1 - product of (1 - weight x score)."""
    miss = 1.0
    for method, value in s.items():
        miss *= 1 - WEIGHTS.get(method, 0) * min(1.0, value)
    return 1 - miss


def image_evidence(s: dict) -> float:
    return max([WEIGHTS[m] * s[m] for m in IMAGE_METHODS if s.get(m)] or [0.0])


def related(cards: dict, a: str, b: str) -> bool:
    """Duplicates of one card (the site's dup links)."""
    if a == b:
        return True
    da, db = (cards.get(a) or {}).get("dup"), (cards.get(b) or {}).get("dup")
    return da == b or db == a or (da is not None and da == db)


def title_only(candidate: dict) -> bool:
    """A short title read with nothing else from the image behind it."""
    s, why = candidate["s"], candidate.get("why") or {}
    return (s.get("title", 0) > 0 and s.get("hash", 0) < 0.3 and s.get("body", 0) < 0.3 and s.get("file", 0) < 0.5
            and s.get("text", 0) < 0.3 and len(alnum(why.get("title", ""))) < SHORT_TITLE)


def art_only(candidate: dict) -> bool:
    """No text on the image backs the card and the whole image doesn't hash as it: only its art's hash or the file
    name does (a post of the art, or of a draft under another title)."""
    s, why = candidate["s"], candidate.get("why") or {}
    if any(s.get(m, 0) >= 0.3 for m in TEXT_METHODS):
        return False
    return not (s.get("hash", 0) >= WHOLE_SURE and why.get("hash_part", "whole") == "whole")


def card_text(image: dict) -> bool:
    """The image reads like a card: enough text, or a stat label (HEALTH, +8 ATK)."""
    return card_like(image.get("lines") or [])


def support(candidate: dict) -> dict:
    """A candidate's evidence by kind (EVIDENCE), each kind's methods combined; an art-only hash isn't the look."""
    s, why = candidate["s"], candidate.get("why") or {}
    return {kind: confidence({m: s[m] for m in methods if s.get(m) and not (m == "hash" and why.get("hash_part") ==
                                                                              "art")})
            for kind, methods in EVIDENCE.items()}


def settle(a: dict, b: dict) -> Optional[str]:
    """Which of two cards whose methods disagree an image shows, or None. A card wins a kind of evidence it is
    strong in (STRONG) and the other has none of (NO_SUPPORT); only one card winning decides. Both winning: the
    card's own look (the whole card's hash) or its versions' text (ocr) outweighs a name alone (the file name or
    title: an old name on the card's own image). An ability's words alone don't beat a card's name and versions'
    text (an ability moved to another card, the first card's old versions reading the same)."""
    sa, sb = support(a), support(b)
    wins = {a["card"]: {k for k in EVIDENCE if sa[k] >= STRONG and sb[k] < NO_SUPPORT},
            b["card"]: {k for k in EVIDENCE if sb[k] >= STRONG and sa[k] < NO_SUPPORT}}
    (x, wx), (y, wy) = wins.items()
    if wx and not wy:
        return x
    if wy and not wx:
        return y
    if wx and wy:
        hard = {c["card"]: "look" in wins[c["card"]] or ("content" in wins[c["card"]] and c["s"].get("ocr", 0) >= 0.5)
                for c in (a, b)}
        if hard[x] and not hard[y] and wy <= {"name"}:
            return x
        if hard[y] and not hard[x] and wx <= {"name"}:
            return y
    return None


def titled_otherwise(image: dict, scored: list) -> bool:
    """A card-shaped image whose title reads as a name (5+ letters) that no candidate's name is like."""
    return any(wordy(t) and len(alnum(t)) >= 5 for t in image.get("title") or []) and \
        not any(c["s"].get("title", 0) >= 0.3 for c in scored)


def judge(image: dict, cards: dict) -> dict:
    """An image's verdict: status (confident, ambiguous, art, mention, none), its cards, and maybe-cards; guess:
    the likeliest card. art: the post shows a card's art (or a file named for it), not the card: never creation
    evidence. A sheet: each cell judged as a card."""
    if image.get("cells"):
        return judge_sheet(image, cards)
    scored = sorted(({**c, "conf": round(confidence(c["s"]), 4), "img": round(image_evidence(c["s"]), 4)}
                     for c in image.get("candidates") or []), key=lambda c: -c["conf"])
    verdict = {"status": "none", "cards": [], "maybe": [], "scored": scored, "reason": ""}
    if not scored:
        return verdict
    best = scored[0]
    verdict["guess"] = best["card"]
    if image.get("ocr") and best["conf"] >= PLAUSIBLE and art_only(best) and not card_text(image) and \
            not any(c["conf"] >= PLAUSIBLE and not art_only(c) for c in scored):
        verdict.update(status="art", art=[c["card"] for c in scored if c["conf"] >= PLAUSIBLE][:3],
                       reason="the art's hash or the file name only, no card text")
        return verdict
    if image.get("shape"):                     # a card-like image shows one card
        votes = {}
        for method in IMAGE_METHODS:
            def value(c):
                return c["s"].get(method, 0) * (few_words(c) if method == "body" else 1.0)
            top = max(scored, key=value)
            if WEIGHTS[method] * value(top) >= STRONG:
                votes[method] = top["card"]
        by_card, clashes = {c["card"]: c for c in scored}, []
        items = sorted(votes.items())
        for i, (m1, x) in enumerate(items):
            for m2, y in items[i + 1:]:
                if x == y or ("title" in (m1, m2) and related(cards, x, y)):     # the title tells duplicates apart
                    continue
                if by_card[x]["s"].get(m2, 0) < 0.3 and by_card[y]["s"].get(m1, 0) < 0.3:
                    clashes.append((m1, x, m2, y))
        unsettled, won, lost = [], set(), set()
        for m1, x, m2, y in clashes:
            winner = settle(by_card[x], by_card[y])
            if winner is None:
                unsettled.append(f"{m1} says {x}, {m2} says {y}")
            else:
                won.add(winner)
                lost.add(y if winner == x else x)
        pick = best
        if won and not unsettled:
            if len(won) == 1 and not won & lost:
                pick = by_card[next(iter(won))]
                verdict["settled"] = [f"{m1} says {x}, {m2} says {y}: {pick['card']}" for m1, x, m2, y in clashes]
            else:
                unsettled.append("settled both ways: " + ", ".join(sorted(won | lost)))
        verdict["guess"] = pick["card"]
        rival = next((c for c in scored if c["card"] != pick["card"] and c["card"] not in lost
                      and not related(cards, c["card"], pick["card"])), None)       # a duplicate is no rival
        risk, rival_risk = 1 - pick["conf"], (1 - rival["conf"]) if rival else 1.0
        if unsettled:
            verdict["reason"] = "methods disagree: " + "; ".join(unsettled)
        elif image.get("ocr") and art_only(pick):
            if not pick["s"].get("hash") and titled_otherwise(image, scored):
                verdict["reason"] = "its title names no card; only the file name or chat names one"
                return verdict
            verdict["reason"] = "only the art's hash or the file name; its text names no card (a draft?)"
        elif pick["conf"] < CONFIDENT:
            verdict["reason"] = f"low confidence ({pick['conf']:.3f})"
        elif rival_risk < ODDS * max(risk, 1e-4):
            verdict["reason"] = f"close second: {rival['card']} ({rival['conf']:.2f})"
        elif title_only(pick):
            verdict["reason"] = "only a short OCR'd title"
        else:
            verdict.update(status="confident", cards=[pick["card"]])
            return verdict
        plausible = [c["card"] for c in scored if c["conf"] >= PLAUSIBLE]
        if plausible and max(c["img"] for c in scored if c["conf"] >= PLAUSIBLE) >= 0.2:
            verdict.update(status="ambiguous", maybe=plausible)
        elif plausible:
            verdict.update(status="mention", maybe=plausible, reason="named in chat only")
        return verdict
    sure = [c["card"] for c in scored if c["conf"] >= CONFIDENT and not (image.get("ocr") and art_only(c)) and max(
        [c["s"].get(m, 0) for m in ("line", "anchor", "hash", "file")]) >= 0.5]
    maybe = [c["card"] for c in scored if c["card"] not in sure and c["conf"] >= PLAUSIBLE and c["img"] >= 0.2]
    if sure:
        verdict.update(status="confident", cards=sure, maybe=maybe)
    elif maybe:
        verdict.update(status="ambiguous", maybe=maybe, reason=f"screenshot, best {best['conf']:.3f}")
    elif best["conf"] >= PLAUSIBLE:
        verdict.update(status="mention", maybe=[c["card"] for c in scored if c["conf"] >= PLAUSIBLE],
                       reason="named in chat only")
    return verdict


def judge_sheet(image: dict, cards: dict) -> dict:
    """A sheet's verdict: its cells' cards (each cell judged as a card-shaped image)."""
    parts = [judge({**cell, "ocr": image.get("ocr")}, cards) for cell in image["cells"]]
    best: dict = {}
    for part in parts:
        for c in part["scored"]:
            if c["conf"] > best.get(c["card"], {"conf": -1})["conf"]:
                best[c["card"]] = c
    sure = list(dict.fromkeys(c for part in parts for c in part["cards"]))
    maybe = list(dict.fromkeys(c for part in parts for c in part["maybe"] if c not in sure))
    unsure = [f"cell {n}: {part['reason']}" for n, part in enumerate(parts, 1) if part["status"] != "confident"
              and part["reason"]]
    verdict = {"status": "confident" if sure else "ambiguous" if maybe else "none", "cards": sure, "maybe": maybe,
               "scored": sorted(best.values(), key=lambda c: -c["conf"]), "guess": next(iter(sure + maybe), None),
               "reason": f"a sheet of {len(parts)} cards" + (": " + "; ".join(unsure) if unsure else ""),
               "cells": [{k: part.get(k) for k in ("status", "cards", "maybe", "reason")} for part in parts]}
    art = [c for part in parts for c in part.get("art") or [] if c not in sure]
    if art:
        verdict["art"] = art
    return verdict


def same_image(images: dict, verdicts: dict, cards: dict):
    """An image that isn't sure but is a twin of a sure one-card post (confident or ruled) shows that card: the
    same image (whole within TWIN_BITS), or one alike (whole within HASH_FULL, or the art within TWIN_BITS) whose
    own title reads the card (look and name agree). Not when its own title or text names another card strongly.
    Its verdict becomes confident, at most at the twin's confidence (a card posted twice before the post it was
    dated by, its short title alone not enough)."""
    for key, image in images.items():
        v = verdicts.get(key)
        if not v or v["status"] in ("confident", "ruled") or not image.get("twins"):
            continue
        titled = {c["card"] for c in v["scored"] if c["s"].get("title", 0) >= 0.9}
        sure: dict = {}
        for other, whole, art in image["twins"]:
            twin = verdicts.get(other)
            if not twin or twin["status"] not in ("confident", "ruled") or len(twin["cards"]) != 1 or twin.get("twin"):
                continue
            cid = twin["cards"][0]
            if whole <= TWIN_BITS or cid in titled:
                sure.setdefault(cid, (other, whole, art, twin))
        if len(sure) != 1:
            continue
        cid, (other, whole, art, twin) = next(iter(sure.items()))
        if any(not related(cards, c["card"], cid) and (c["s"].get("title", 0) >= 0.9 or c["s"].get("ocr", 0) >= 0.9)
               for c in v["scored"]):
            continue
        conf = 1.0 if twin["status"] == "ruled" else next((c["conf"] for c in twin["scored"] if c["card"] == cid), 0)
        own = next((c for c in v["scored"] if c["card"] == cid), None)
        if own is None:
            own = {"card": cid, "s": {}, "why": {}, "conf": 0.0, "img": 0.0}
            v["scored"].append(own)
        alike = confidence({**own["s"], "hash": hash_score(whole) if whole <= HASH_FULL else WHOLE_SURE})
        own.update(conf=round(max(own["conf"], conf if whole <= TWIN_BITS else min(conf, alike)), 4),
                   why={**(own.get("why") or {}), "twin": other, "twin_bits": [whole, art]})
        v["scored"].sort(key=lambda c: -c["conf"])
        v.update(status="confident", cards=[cid], maybe=[], guess=cid, twin={"key": other, "bits": [whole, art]},
                 reason=f"the same image as a {twin['status']} post of it ({whole} bits)" if whole <= TWIN_BITS else
                 f"its title and a {twin['status']} post's look (whole {whole}, art {art} bits) agree")


# ========================
# Rulings
# ========================
NONE_WORDS = {"none", "no", "nothing", "-", "no card", "not a card", "x", "n/a", "na"}
UNDO_WORDS = {"?", "skip", "unsure", "undo", "clear", "withdraw"}


def parse_rulings(text: str) -> list:
    """[(id, value)] from text like "A12 = Isma, A13 = none" (= or :, any separators)."""
    marks = list(finditer(r"(?<![A-Za-z0-9])([Aa]\d+)\s*[=:]\s*", text or ""))
    found = []
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        value = text[mark.end():end].strip().rstrip(",;").strip()
        found.append((mark.group(1).upper(), value))
    return found


def resolve(value: str, entry: Optional[dict], names: Names) -> tuple:
    """(cards, note): the cards a ruling names; ([], note) for none; (None, note) to withdraw; note 'error: ...'."""
    plain = " ".join(value.lower().split())
    if plain in NONE_WORDS:
        return [], "none"
    if plain in UNDO_WORDS:
        return None, "withdrawn"
    candidates = [c["card"] for c in (entry or {}).get("candidates") or []]
    number = fullmatch(r"#?(\d{1,2})", plain)
    if number:
        k = int(number.group(1))
        if 1 <= k <= len(candidates):
            return [candidates[k - 1]], f"candidate {k}"
        return None, f"error: no candidate {k}"

    def one(text: str):
        text = text.strip().strip("\"'")
        if text in names.cards:
            return [text]
        hits = names.exact(text)
        if len(hits) > 1:
            preferred = [c for c in hits if c in candidates]
            hits = preferred if len(preferred) == 1 else hits
        if len(hits) == 1:
            return list(hits)
        if len(hits) > 1:
            return "error: " + f"{text} could be " + ", ".join(sorted(hits))
        close = names.fuzzy(text, 0.9, limit=2)
        if len(close) == 1 or (len(close) > 1 and list(close.values())[0][0] - list(close.values())[1][0] >= 0.05):
            return [next(iter(close))]
        guesses = names.fuzzy(text, 0.6, limit=3)
        return f"error: no card {text!r}" + (f" (maybe {', '.join(guesses)})" if guesses else "")

    found = one(value)
    if isinstance(found, list):
        return found, "card"
    parts = [p for p in re_split(r"\s*(?:\+|&|,|\band\b)\s*", value) if p.strip()]
    if len(parts) > 1:
        many, errors = [], []
        for part in parts:
            got = one(part)
            (many.extend(got) if isinstance(got, list) else errors.append(got))
        if not errors:
            return list(dict.fromkeys(many)), "cards"
        return None, "; ".join(errors)
    return None, found


def rulings_main(root: str, argv: list) -> int:
    """apply_rulings.py: parse, resolve and store rulings, then rebuild."""
    parser = ArgumentParser(prog="apply_rulings.py", description="Rule over ambiguous.html's entries, then rebuild.")
    parser.add_argument("rulings", nargs="*", help='e.g. "A12 = Isma, A13 = none"')
    parser.add_argument("--file", help="read the rulings from a file")
    parser.add_argument("--list", action="store_true", help="show the rulings so far")
    parser.add_argument("--dry-run", action="store_true", help="resolve only: store nothing")
    args = parser.parse_args(argv)
    root = check_root(root)
    paths = Paths(root)
    stored = read_json(paths.rulings, {}) or {}
    if args.list:
        for aid in sorted(stored, key=lambda a: int(a[1:])):
            print(f"{aid} = {' + '.join(stored[aid]['cards']) or 'none'}  ({stored[aid].get('text')})")
        return 0
    text = " ".join(args.rulings)
    if args.file:
        text += "\n" + open(args.file, encoding="utf-8").read()
    if not text.strip() and not stdin.isatty():
        text = stdin.read()
    parsed = parse_rulings(text)
    if not parsed:
        print('no rulings found: write them like "A12 = Isma, A13 = none"')
        return 2
    entries = {e["id"]: e for e in read_json(paths.ambiguous, {}).get("entries", [])}
    pairs = read_json(paths.pairs, {}) or {}
    names = Names(pairs.get("cards") or {})
    errors, done = [], []
    for aid, value in parsed:
        if aid not in entries:
            errors.append(f"{aid}: no such entry in ambiguous.json")
            continue
        cards, note = resolve(value, entries[aid], names)
        if note.startswith("error"):
            errors.append(f"{aid} = {value}: {note[7:]}")
        elif cards is None:
            stored.pop(aid, None)
            done.append(f"{aid}: ruling withdrawn")
        else:
            stored[aid] = {"cards": cards, "text": value, "key": entries[aid]["key"], "at": now_iso()}
            done.append(f"{aid} = {' + '.join(cards) or 'none'}")
    if errors:                                  # all or nothing: fix the line and send it all again
        print("\n".join(errors + [f"nothing stored: {len(errors)} to fix, then send the whole line again"]))
        return 1
    print("\n".join(done))
    if args.dry_run:
        return 0
    write_json(paths.rulings, stored)
    build(root, log=print)
    return 0


APPLY_SCRIPT = '''#!/usr/bin/env python3
"""Rule over ambiguous.html's entries, then matches.json, ambiguous.json/html and REPORT.md are rebuilt (no OCR,
no hashing, a few seconds):

    python3 -I apply_rulings.py "A12 = Isma, A13 = none"     # or --file rulings.txt, or the rulings on stdin
    python3 -I apply_rulings.py --list                        # the rulings so far

A ruling names what the post image shows: a card (its name, an alias, its file name or its site id), several
joined with +, none, or a candidate's number on the page (#2). "A12 = ?" withdraws one; a later ruling of an id
wins. Rulings are kept in match/rulings.json; a line with any error stores nothing (fix it, send it all again). Written
by AutoCroissant's commands/discord_match.py, which does the work: the checkout that wrote this file first, else
~/Desktop/AutoCroissant (set DISCORD_MATCH_REPO for another), if its matcher reads these files."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPOS = [os.environ.get("DISCORD_MATCH_REPO", ""), %(repo)r, "~/Desktop/AutoCroissant"]   # the writer's first
NEEDS = ("def rulings_main(", "PAIRS_VERSION = %(version)d")                              # one that reads these files
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]     # nothing imported from the archive
for repo in REPOS:
    repo = os.path.expanduser(repo)
    try:
        with open(os.path.join(repo, "commands", "discord_match.py"), encoding="utf-8") as f:
            source = f.read()
    except OSError:
        continue
    if repo and all(need in source for need in NEEDS):
        sys.path.insert(0, repo)
        break
else:
    sys.exit("can't find AutoCroissant's commands/discord_match.py (PAIRS_VERSION %(version)d): set DISCORD_MATCH_REPO")
from commands.discord_match import rulings_main

sys.exit(rulings_main(HERE, sys.argv[1:]))
'''


# ========================
# Build
# ========================
def post_info(record: dict, channels: dict) -> dict:
    channel = channels.get(record.get("channel_id")) or {}
    category = (channels.get(channel.get("parent_id")) or {}).get("name") if channel.get("parent_id") else None
    author = record.get("author") or {}
    return {"message_id": record["id"], "date": record.get("timestamp"), "channel": record.get("channel_name"),
            "category": category, "thread": (record.get("thread") or {}).get("name"),
            "jump_url": record.get("jump_url"), "bot": is_bot(record),
            "poster": {k: author.get(k) for k in ("id", "username", "global_name")},
            "excerpt": excerpt(content_of(record))}


def assign_ids(keys: list, registry: dict) -> dict:
    """Stable ids: a key keeps its A-number forever; new keys get the next ones."""
    ids, last = registry.setdefault("ids", {}), registry.get("next", 1)
    for key in keys:
        if key not in ids:
            ids[key] = f"A{last}"
            last += 1
    registry["next"] = last
    return ids


def poster_name(post: dict) -> str:
    poster = post.get("poster") or {}
    return poster.get("global_name") or poster.get("username") or "?"


def is_port(post: dict) -> bool:
    """A port channel's post: cards re-uploaded to be ported, by whoever ported them."""
    return bool(PORT.search(post.get("channel") or ""))


def people_map(first_posts: list, people: list) -> dict:
    """Discord poster -> credited person: by name likeness, else by the credits of the cards they posted first
    outside the port channels (most of them; or the most, for a person no poster's name is like)."""
    by_poster = defaultdict(list)
    for post, credited in first_posts:
        if (post.get("poster") or {}).get("id"):
            by_poster[post["poster"]["id"]].append((post, credited))
    found, claimed = {}, set()
    for pid, items in by_poster.items():
        poster = items[0][0]["poster"]
        handles = [h for h in (poster.get("username"), poster.get("global_name")) if h]
        like = max(((ratio(alnum(h), alnum(p)), p) for h in handles for p in people), default=(0, None))
        counts = Counter(c for post, c in items if c and not is_port(post))
        top = counts.most_common(1)[0] if counts else (None, 0)
        total = sum(counts.values())
        if like[0] >= 0.8:
            person, basis = like[1], f"name like {like[1]}"
            claimed.add(person)
        elif top[0] and top[1] >= 3 and top[1] >= 0.6 * total:
            person, basis = top[0], f"credited on {top[1]} of the {total} cards they posted first"
        else:
            person, basis = None, "unknown"
        found[pid] = {"username": poster.get("username"), "global_name": poster.get("global_name"), "person": person,
                      "basis": basis, "first_posts": len(items), "port_first_posts": sum(map(is_port, (
                          post for post, _ in items))), "credits": dict(counts)}
    for m in found.values():                    # the most credits, for a person no poster's name is like
        counts = Counter(m["credits"])
        top = counts.most_common(1)[0] if counts else (None, 0)
        if m["person"] is None and top[0] and top[0] not in claimed and top[1] >= 10 and \
                top[1] >= 0.5 * sum(counts.values()):
            m["person"], m["basis"] = top[0], (f"credited on {top[1]} of the {sum(counts.values())} cards they "
                                               "posted first (the most; no poster's name is like it): a guess")
    return found


def parse_authors(path: str) -> tuple:
    """site/Card Authors.txt: ({git name: shown}, {card key: person})."""
    people, credited, section = {}, {}, None
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError:
        return people, credited
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].lower()
            continue
        if "=" in line:
            left, right = (part.strip() for part in line.split("=", 1))
            (people if section == "people" else credited)[left if section == "people" else alnum(left)] = right
    return people, credited


def build(root: str, cards_repo: str = LOCAL_DIR_LOC, log: Callable = print) -> dict:
    """matches.json, ambiguous.json/html and REPORT.md from match/pairs.json and the rulings."""
    root = check_root(root)
    paths = Paths(root)
    lock = hold(paths.lock)
    try:
        return _build(root, paths, cards_repo, log)
    finally:
        lock.close()


def _build(root: str, paths: Paths, cards_repo: str, log: Callable) -> dict:
    pairs = read_json(paths.pairs, {}) or {}
    cards = pairs.get("cards") or load_cards(paths.site)
    images = pairs.get("images") or {}
    versions = {v["key"]: v for v in pairs.get("versions") or []}
    rulings = read_json(paths.rulings, {}) or {}
    registry = read_json(paths.ids, {}) or {}
    records, _ = load_posts(root)
    channels = {c["id"]: c for c in read_json(join(root, "channels.json"), []) or [] if isinstance(c, dict)}
    summary = read_json(join(root, "summary.json"), {}) or {}
    names = Names(cards)
    ctx = Context(records)
    people, credited = parse_authors(join(expanduser(cards_repo), "site", "Card Authors.txt"))

    # verdicts
    verdicts, posts, art = {}, defaultdict(list), defaultdict(list)
    for key, image in images.items():
        record = records.get(image["message"])
        if record is None:
            continue
        verdicts[key] = judge(image, cards)
    keys = sorted(verdicts, key=lambda k: (int(images[k]["message"]), k))
    id_of = registry.get("ids", {})
    by_id = {aid: key for key, aid in id_of.items()}
    for aid, ruling in rulings.items():
        key = by_id.get(aid) or ruling.get("key")
        if key in verdicts:
            verdicts[key].update(status="ruled", cards=[c for c in ruling["cards"] if c in cards], maybe=[],
                                 reason=f"ruled {aid}: {ruling.get('text')}")
    same_image(images, verdicts, cards)
    first_sure, first_key = {}, {}
    for key in keys:
        v, record = verdicts[key], records[images[key]["message"]]
        if v["status"] in ("confident", "ruled") and not is_bot(record):
            for cid in v["cards"]:
                first_sure.setdefault(cid, epoch(record.get("timestamp")))
                first_key.setdefault(cid, key)
    for cid, key in first_key.items():           # a first post the date rests on, not sure enough: the owner's
        v = verdicts[key]
        best = next((c for c in v["scored"] if c["card"] == cid), None)
        if v["status"] == "confident" and best and best["conf"] < REVIEW:
            v.setdefault("confirm", []).append(cid)
    flagged = []
    for key in keys:
        v, record = verdicts[key], records[images[key]["message"]]
        when = epoch(record.get("timestamp"))
        earlier = [c for c in v["maybe"] if not is_bot(record) and (first_sure.get(c) is None or when < first_sure[c])]
        v["affects_earliest"] = earlier
        v.setdefault("confirm", [])
        if v["status"] == "confident" and (earlier or v["confirm"]):
            v["reason"] = "; ".join(([f"confirm: the first post of {', '.join(v['confirm'])} at " + ", ".join(
                f"{c['conf']:.3f}" for c in v["scored"] if c["card"] in v["confirm"])] if v["confirm"] else [])
                + (["also maybe " + ", ".join(earlier)] if earlier else []))
        if is_bot(record):                      # never a date: nothing to rule on
            continue
        if v["status"] in ("ambiguous", "ruled") or (v["status"] == "confident" and (earlier or v["confirm"])):
            flagged.append(key)
    id_of = assign_ids(flagged, registry)       # an id once given stays with its post, shown or not
    write_json(paths.ids, registry)

    # entries
    entries = []
    for key in flagged:
        image, v = images[key], verdicts[key]
        record = records[image["message"]]
        aid = id_of[key]
        shown = [c for c in v["scored"] if c["conf"] >= 0.15][:6] or v["scored"][:3]
        guess = v["cards"] or [c for c in [v.get("guess")] if c] or [c["card"] for c in shown[:1]]
        entries.append({
            "id": aid, "key": key, "status": v["status"], "reason": v["reason"], "post": post_info(record, channels),
            "image": image["file"], "thumb": image.get("thumb"), "filename": image.get("filename"),
            "shape": image.get("shape"), "multi": bool(image.get("multi")), "ocr_title": image.get("title") or [],
            "ocr_lines": image.get("lines") or [], "best_guess": guess, "affects_earliest": v["affects_earliest"],
            "confirm": v["confirm"], "ruling": rulings.get(aid), "settled": v.get("settled") or [],
            "candidates": [{"card": c["card"], "name": cards[c["card"]].get("name"), "conf": c["conf"], "s": c["s"],
                            "why": c.get("why") or {},
                            "thumb": (versions.get((c.get("why") or {}).get("version")) or {}).get("thumb")
                            or card_thumb(versions, c["card"])} for c in shown]})
    entries.sort(key=lambda e: (e["ruling"] is not None, not (e["affects_earliest"] or e["confirm"]),
                                int(e["id"][1:])))
    by_entry = {e["id"]: e for e in entries}

    # per card
    for key in keys:
        v = verdicts[key]
        if v["status"] in ("confident", "ruled"):
            record = records[images[key]["message"]]
            for cid in v["cards"]:
                scored = next((c for c in v["scored"] if c["card"] == cid), None)
                posts[cid].append((record, images[key], scored, v["status"]))
        elif v["status"] == "art" and not is_bot(records[images[key]["message"]]):
            for cid in v["art"]:
                art[cid].append((records[images[key]["message"]], images[key]))
    mention_first, named = mention_index(names, records)
    out_cards, first_posts = {}, []
    for cid in sorted(cards):
        card = cards[cid]
        listed = credited.get(alnum(card.get("key"))) or credited.get(alnum(card.get("name")))
        entry = {"name": card.get("name"), "key": card.get("key"), "type": card.get("type"),
                 "series": card.get("series"), "credited": card.get("author") or listed,
                 "credited_in_file": listed, "fs": iso(card["fs"]) if card.get("fs") else None, "discord": None}
        mine = [p for p in posts.get(cid, []) if not is_bot(p[0])]
        pending = [e["id"] for e in entries if cid in e["affects_earliest"] and e["ruling"] is None]
        if mine or pending or cid in mention_first or cid in art:
            entry["discord"] = card_discord(cid, card, mine, posts.get(cid, []), pending, mention_first.get(cid),
                                            named.get(cid, []), versions, channels, names, ctx)
            if art.get(cid):                     # its art posted (not the card): context, never its date
                record, image = art[cid][0]
                entry["discord"]["art_first"] = {**post_info(record, channels), "image": image["file"],
                                                 "thumb": image.get("thumb"), "filename": image.get("filename")}
                entry["discord"]["art_posts"] = len(art[cid])
                if not mine and not pending:
                    entry["discord"]["tier"] = "art"
            entry["discord"]["to_confirm"] = [e["id"] for e in entries if cid in e["confirm"] and e["ruling"] is None]
            if pending:
                aid = min(pending, key=lambda a: (by_entry[a]["post"]["date"] or "", int(a[1:])))
                entry["discord"]["earliest_open"] = {"id": aid, "date": by_entry[aid]["post"]["date"]}
            if mine:
                first_posts.append((entry["discord"]["first"], entry["credited"]))
                if entry["discord"]["fs_check"].get("verdict") == "git_earlier":     # shown after the upload
                    entry["discord"]["suspect"] = "first sure post after fs: an earlier post is likely missed"
        out_cards[cid] = entry
    suspects = {cid for cid, e in out_cards.items() if (e["discord"] or {}).get("suspect")}
    for e in entries:
        e["suspect"] = sorted(set(e["affects_earliest"]) & suspects)
    entries.sort(key=lambda e: (e["ruling"] is not None, not (e["affects_earliest"] or e["confirm"]),
                                not e["suspect"], int(e["id"][1:])))
    mapping = people_map(first_posts, sorted(set(people.values()) | {c.get("author") for c in cards.values()
                                                                       if c.get("author")}))
    for cid, entry in out_cards.items():
        first = (entry["discord"] or {}).get("first")
        if first:
            person = (mapping.get((first.get("poster") or {}).get("id")) or {}).get("person")
            entry["discord"]["poster_vs_author"] = {
                "poster": poster_name(first), "person": person, "credited": entry["credited"],
                "verdict": "ported" if is_port(first) else "unmapped" if not person
                else "same" if person == entry["credited"] else "different"}
    unmatched = unmatched_cardlike(images, verdicts, records, channels)
    anomalies = find_anomalies(out_cards, snowflake_seconds(summary.get("guild_id")))
    counts = tally(out_cards, images, verdicts, entries, records)
    result = {"generated": now_iso(), "root": root, "mining": {k: summary.get(k) for k in
                                                                ("guild_id", "status", "started", "finished",
                                                                 "totals")},
              "pairs_generated": pairs.get("generated"), "ocr": pairs.get("ocr"),
              "versions_mode": pairs.get("versions_mode"), "counts": counts, "thresholds": thresholds(),
              "people_map": mapping, "anomalies": anomalies, "unmatched_cardlike": unmatched, "cards": out_cards}
    result["proposal"] = proposals(paths, result, cards, names, cards_repo)
    write_json(paths.matches, result)
    write_json(paths.ambiguous, {"generated": result["generated"], "how": "reply with rulings like "
                                 "\"A12 = Isma, A13 = none\" (python3 -I apply_rulings.py ...)", "entries": entries})
    write_text(paths.html, render_html(entries, cards, result))
    write_text(paths.report, render_report(result, entries, read_json(paths.selftest), cards))
    write_text(paths.script, APPLY_SCRIPT % {"repo": dirname(dirname(abspath(__file__))), "version": PAIRS_VERSION})
    log(f"build: {counts['cards_dated']} of {counts['cards']} cards dated, {len(entries)} ambiguous entries "
        f"({counts['ambiguous_open']} open), {len(anomalies)} anomalies")
    return result


SURE_TIERS = ("ruled", "high", "confident")
EDITS_FILE, AUTHORS_FILE = "Card Edits - created and aliases.json", "Card Authors - corrections.txt"


def proposals(paths: Paths, result: dict, cards: dict, names: Names, cards_repo: str) -> dict:
    """What the matches would change on the site, for the owner to review (nothing is applied): created dates and
    aliases as Card Edits.json fields, author corrections as Card Authors.txt lines. Card names, dates and the
    site's people only: no message text, links or Discord names."""
    edits = read_json(join(expanduser(cards_repo), "site", "Card Edits.json"), {}) or {}
    current = {alnum(k): v for k, v in (edits.get("cards") or {}).items() if isinstance(v, dict)}
    ready, pending, authors = {}, {}, []
    for cid, entry in sorted(result["cards"].items(), key=lambda kv: (kv[1].get("key") or kv[0]).lower()):
        d, key = entry.get("discord") or {}, entry.get("key") or entry.get("name") or cid
        first, fields = d.get("first"), {}
        if first and d.get("tier") in SURE_TIERS and (d.get("fs_check") or {}).get("verdict") in (
                "discord_earlier", "no_fs"):
            fields["created"] = first["date"][:10]
        aliases = list((current.get(alnum(key)) or {}).get("aliases") or [])
        known = set(names.card_forms.get(cid, ())) | {squash(a) for a in aliases}
        for alias in aliases:
            names.form_words.setdefault(squash(alias), words(alias).split())
        new = []                                  # a former title once (backed), a file name on two days
        rows = [(r["text"], True) for r in (d.get("names") or {}).get("former_titles", [])]
        rows += [(r["text"], False) for r in (d.get("names") or {}).get("file_names", []) if r.get("days", 0) >= 2]
        for text, printed in rows:
            if alias_ok(text, cid, names, known, printed):
                known.add(squash(text))
                new.append(text)
        if new and d.get("tier") in SURE_TIERS:
            fields["aliases"] = list((current.get(alnum(key)) or {}).get("aliases") or []) + new
        waits = sorted(set(d.get("earlier_ambiguous") or []) | set(d.get("to_confirm") or []), key=lambda a: int(a[1:]))
        if d.get("suspect") and d.get("earliest_open"):     # its first sure post is after fs: one is likely missed
            fields["created_if_ruled"] = {"id": d["earliest_open"]["id"], "date": d["earliest_open"]["date"][:10]}
        if fields:
            (pending if waits else ready)[key] = {**fields, **({"waits_on": waits} if waits else {})}
        pva = d.get("poster_vs_author") or {}
        if pva.get("verdict") == "different" and pva.get("person") and d.get("tier") in SURE_TIERS:
            authors.append((pva["person"], entry.get("credited") or "?", key, first["date"][:10], first.get("channel"),
                            first.get("creation_wording"), waits))
    makedirs(paths.proposal, exist_ok=True)
    write_json(join(paths.proposal, EDITS_FILE), {
        "about": "Proposed from the Discord archive: merge into TTSCardMaker's site/Card Edits.json after review "
                 "(created: a new database-only field, the date of the card's first Discord post; aliases: the "
                 "edit's current aliases plus names used on Discord). pending: waits on rulings in ambiguous.html; "
                 "created_if_ruled (not a field): a card whose first sure post is after fs gets this date if that "
                 "entry is ruled to show it.",
        "generated": result["generated"], "cards": ready, "pending": pending})
    lines = ["# Author corrections proposed from the Discord archive. Nothing is applied: review first.",
             "# Each line is a [cards] line for TTSCardMaker's site/Card Authors.txt: it replaces the card's line.",
             "# Evidence: the card's first Discord post was by this person (mapped from their Discord name; see",
             "# REPORT.md). The first poster isn't always the maker (a remake, a post for someone else): check.",
             f"# Generated {result['generated']}.", ""]
    for (person, credited), group in sorted(Counter((a[0], a[1]) for a in authors).items(), key=lambda kv: -kv[1]):
        lines.append(f"# --- first posted by {person}, credited to {credited} ({group}) ---")
        for _, _, key, date, channel, wording, waits in (a for a in authors if (a[0], a[1]) == (person, credited)):
            lines.append(f"{key} = {person}    # now {credited}; first post {date} #{channel}"
                         + ("; new-card wording" if wording else "") + (f"; waits on {', '.join(waits)}" if waits
                                                                         else ""))
        lines.append("")
    write_text(join(paths.proposal, AUTHORS_FILE), "\n".join(lines))
    return {"created_ready": sum(1 for f in ready.values() if "created" in f),
            "created_pending": sum(1 for f in pending.values() if "created" in f),
            "created_if_ruled": sum(1 for f in pending.values() if "created_if_ruled" in f),
            "aliases": sum(len(f.get("aliases", [])) - len((current.get(alnum(k)) or {}).get("aliases") or [])
                           for k, f in list(ready.items()) + list(pending.items())),
            "alias_cards": sum(1 for f in list(ready.values()) + list(pending.values()) if "aliases" in f),
            "authors": len(authors), "authors_by": {f"{p} (credited {c})": n for (p, c), n in Counter(
                (a[0], a[1]) for a in authors).most_common()},
            "files": [paths.rel(join(paths.proposal, EDITS_FILE)), paths.rel(join(paths.proposal, AUTHORS_FILE))]}


def card_thumb(versions: dict, cid: str) -> Optional[str]:
    found = [v for v in versions.values() if v["card"] == cid]
    found.sort(key=lambda v: not v.get("current"))
    return found[0].get("thumb") if found else None


def mention_index(names: Names, records: dict) -> tuple:
    """({card: first message naming it}, {card: messages naming it}): word for word, multi-word names or a message
    that is just the name."""
    first, named = {}, defaultdict(list)
    for record in sorted(records.values(), key=lambda r: int(r["id"])):
        text = content_of(record)
        if not text or is_bot(record):
            continue
        size = len(words(text).split())
        for cid, phrase in names.mentions(text).items():
            if " " not in phrase and size > 3:
                continue
            named[cid].append(record)
            first.setdefault(cid, {"date": record.get("timestamp"), "jump_url": record.get("jump_url"),
                                   "phrase": phrase, "poster": (record.get("author") or {}).get("username")})
    return first, named


def shown_look(scored: Optional[dict], image: dict, versions: dict) -> tuple:
    """(look, pick): ("git", n) for one look of the card's versions, ("draft",) for one not in git, or None when
    the image can't tell (a screenshot, a crop without text, versions of several looks tied)."""
    pick = ((scored or {}).get("why") or {}).get("pick") or {}
    if not image.get("shape") or not pick:
        return None, pick
    if pick.get("by") == "draft":
        return ("draft",), pick
    looks = {versions[k].get("look", 0) for k in pick.get("keys") or [] if k in versions}
    return (("git", looks.pop()) if pick.get("by") in ("text", "hash") and len(looks) == 1 else None), pick


def same_draft(a: dict, b: dict) -> bool:
    """Two posts of a look not in git show the same one."""
    if a.get("whole") and b.get("whole") and hash_distance(a["whole"], b["whole"]) <= LOOK_BITS:
        return True
    ta, tb = a.get("text") or "", b.get("text") or ""
    return len(ta) >= TEXT_MIN and len(tb) >= TEXT_MIN and max(novel_run(ta, grams(tb)),
                                                               novel_run(tb, grams(ta))) < DRAFT_RUN


def version_note(look: Optional[tuple], pick: dict, versions: dict) -> dict:
    """What a post shows: the current look, an older one (its git versions), a draft, or unknown."""
    tied = [versions[k] for k in pick.get("keys") or [] if k in versions]
    shows = None if look is None else "draft" if look[0] == "draft" else "current" if look[1] == 0 else "older"
    note = {"shows": shows, "by": pick.get("by"), "commits": [v.get("commit") for v in tied if v.get("commit")],
            "git_dates": sorted({iso(v["ts"])[:10] for v in tied if v.get("ts")})}
    note.update({k: pick[k] for k in ("fit", "novel", "bits") if pick.get(k) is not None})
    if look and look[0] == "git":
        note["look"] = look[1]
    return note


def card_discord(cid: str, card: dict, mine: list, every: list, pending: list, mention: Optional[dict],
                 named: list, versions: dict, channels: dict, names: Names, ctx: Context) -> dict:
    """A card's Discord evidence: first post, the first post of each look (revisions), names, notes, the date
    check."""
    found = {"posts": len(mine), "bot_posts": len(every) - len(mine), "earlier_ambiguous": pending,
             "first_mention": mention}
    seen, drafts, undecided = {}, [], 0        # look -> (record, image, note); drafts: [(look, image)]
    for n, (record, image, scored, status) in enumerate(mine):
        look, pick = shown_look(scored, image, versions)
        if look == ("draft",):
            match = next((d for d, other in drafts if same_draft(image, other)), None)
            look = match or ("draft", len(drafts) + 1)
            if match is None:
                drafts.append((look, image))
        note = version_note(look, pick, versions)
        if n == 0:
            first_note = note
        if look is None:
            undecided += int(bool(image.get("shape")))
        elif look not in seen:
            seen[look] = (record, image, note)
    if mine:
        record, image, scored, status = mine[0]
        conf = 1.0 if status == "ruled" else (scored or {}).get("conf", 0)
        tier = "ruled" if status == "ruled" else "high" if conf >= HIGH and image_evidence(
            (scored or {}).get("s") or {}) >= 0.85 else "confident"
        found["first"] = {**post_info(record, channels), "image": image["file"], "thumb": image.get("thumb"),
                          "filename": image.get("filename"), "conf": round(conf, 4), "tier": tier,
                          "methods": (scored or {}).get("s"), "creation_wording": bool(CREATION.search(
                              content_of(record))), "version": first_note}
        found["tier"] = tier
        found["last"] = mine[len(mine) - 1][0].get("timestamp")
    else:
        found["tier"] = "ambiguous" if pending else "mention"
    order = sorted(seen.values(), key=lambda t: int(t[0]["id"]))   # each look's first post; the earliest isn't
    revisions = [{"date": record.get("timestamp"), "jump_url": record.get("jump_url"), "image": image["file"],
                  **note} for record, image, note in order[1:]]                       # a revision
    current = seen.get(("git", 0))
    found["revisions"], found["current_since"] = revisions, current[0].get("timestamp") if current else None
    found["looks_untold"] = undecided
    found["names"] = chat_names(cid, card, mine, names, ctx)
    found["notes"] = card_notes(cid, mine, named, ctx)
    found["series_hints"] = series_hints(card, mine)
    found["posters"] = dict(Counter(poster_name(post_info(record, channels)) for record, *_ in mine).most_common(5))
    found["fs_check"] = fs_check(card.get("fs"), epoch(found["first"]["date"]) if mine else None)
    return found


LABEL_WORDS = {"color", "colour", "colored", "coloured", "template", "base", "art", "bg", "background", "icon",
               "render", "sprite", "pic", "alt", "blank", "proto", "concept", "idea", "ideas", "sketch", "done"}


def typed_name(filename: str) -> str:
    """A file name as a name: words split at _, - and camel case; upload ids, version, generic and label words
    dropped (Pale_Wizard2.png -> Pale Wizard)."""
    found = []
    for t in re_split(r"[\W_]+", file_stem(filename)):
        if not t or random_id(t):
            continue
        for w in sub(r"(?<=[a-z])(?=[A-Z])", " ", sub(r"(?<=[A-Za-z]{3})\d{1,2}$", "", t)).split():
            if not (VERSION_WORD.fullmatch(w.lower()) or w.lower() in GENERIC_FILE or w.lower() in LABEL_WORDS):
                found.append(w)
    return " ".join(found)


def chat_names(cid: str, card: dict, mine: list, names: Names, ctx: Context) -> dict:
    """Names the site lacks: titles read off its posted images (as typed in the chat or file name that backs
    them), file names of card-shaped posts (typed, with the days they were posted), short captions."""
    known = names.card_forms.get(cid, set())

    def new(text: str, floor: float = 0.85) -> bool:    # not a known name, nor a piece of one, nor one plus noise
        form = squash(text)
        return len(form) >= 3 and all(ratio(form, k) < floor and form not in k and k not in form for k in known)

    titles, files, captions = defaultdict(list), defaultdict(list), defaultdict(list)
    said = {}                                   # squashed -> as typed: captions, thread names, file names
    for record, image, *_ in mine:
        for text in (content_of(record).strip(), ctx.thread_name(record) or "", typed_name(image.get("filename"))):
            if squash(text):
                said.setdefault(squash(text), " ".join(text.split()))

    def backing(text: str) -> tuple:            # OCR alone misreads stylized fonts: chat or a file name agrees
        q = squash(text)
        return max(((ratio(q, t), typed) for t, typed in said.items()), default=(0.0, "")) if len(q) >= 4 \
            else (0.0, "")
    for record, image, scored, status in mine:
        if MEME.search(record.get("channel_name") or ""):     # a joke name, not the card's
            continue
        found = [(backing(t), t) for t in image.get("title") or [] if new(t)]
        found = [(b, typed) for (b, typed), t in found if b >= 0.9 and new(typed)]
        if found:
            titles[max(found)[1]].append(record.get("timestamp"))
        typed = typed_name(image.get("filename"))
        if typed and image.get("shape") and not image.get("multi") and new(typed, 0.8):
            files[typed].append(record.get("timestamp"))
        text = content_of(record).strip()
        tokens = words(text).split()
        if 1 <= len(tokens) <= 5 and "http" not in text and not set(tokens) <= CHAT_STOP and new(text, 0.8) \
                and text[:1].isupper():
            captions[text].append(record.get("timestamp"))

    def rows(found: dict) -> list:
        return sorted(({"text": k, "count": len(v), "days": len({d[:10] for d in v if d}), "first": min(v)}
                       for k, v in found.items()), key=lambda r: (-r["count"], r["first"]))[:8]
    return {"former_titles": rows(titles), "file_names": rows(files), "captions": rows(captions)}


def alias_ok(text: str, cid: str, names: Names, known: set, printed: bool) -> bool:
    """A name worth adding as an alias: a word the card's names lack (not a misspelling or a piece of them: a
    fuzzy lookup finds those), no one-letter words unless it is the name's initials (GOK), naming no other card;
    a file name (not printed on the card) shares a word with the card's names (a swapped file name doesn't)."""
    form = squash(text)
    if len(form) < 3 or form in known or [c for c in names.fuzzy(text, 0.9) if c != cid]:
        return False
    initials = {squash("".join(w[0] for w in names.form_words.get(k) or [] if w)) for k in known}
    if form in {i for i in initials if len(i) >= 3}:
        return True
    if any(ratio(form, k) >= 0.75 or form in k or k in form for k in known):
        return False
    mine = [squash(w) for w in words(text).split()]
    theirs = {squash(w) for k in known for w in names.form_words.get(k) or [] if len(w) >= 3}
    if any(len(w) == 1 for w in mine):
        return False
    fresh = [w for w in mine if len(w) >= 3 and all(ratio(w, k) < 0.75 and w not in k and k not in w for k in theirs)]
    shared = [w for w in mine if w in theirs]
    return bool(fresh) and (printed or bool(shared))


def card_notes(cid: str, mine: list, named: list, ctx: Context) -> list:
    """Balance and lore remarks in replies, threads and messages naming the card."""
    seen, found = set(), []
    pool = [r for record, *_ in mine for r in ctx.discussion(record)] + named
    for record in sorted(pool, key=lambda r: int(r["id"])):
        if record["id"] in seen or is_bot(record):
            continue
        seen.add(record["id"])
        text = content_of(record)
        kinds = [k for k, rx in (("balance", BALANCE), ("lore", LORE)) if rx.search(text)]
        if kinds:
            found.append({"kind": "+".join(kinds), "date": record.get("timestamp"), "jump_url": record.get("jump_url"),
                          "poster": (record.get("author") or {}).get("username"), "excerpt": excerpt(text, 280)})
    if len(found) > NOTES_PER_CARD:
        found = found[:NOTES_PER_CARD // 2] + found[len(found) - NOTES_PER_CARD // 2:]
    return found


def series_hints(card: dict, mine: list) -> list:
    """Channel and thread names of its posts (series or set context)."""
    places = Counter()
    for record, *_ in mine:
        for place in (record.get("channel_name"), (record.get("thread") or {}).get("name")):
            if place:
                places[place] += 1
    return [{"place": p, "posts": n} for p, n in places.most_common(4)]


def fs_check(fs, first: Optional[float]) -> dict:
    if first is None:
        return {"verdict": "no_discord"}
    if not fs:
        return {"verdict": "no_fs", "discord_first": iso(first)}
    days = (fs - first) / 86400
    verdict = "discord_earlier" if days > 2 else "same_time" if days >= -2 else "git_earlier"
    return {"verdict": verdict, "discord_first": iso(first), "fs": iso(fs), "days_discord_before_git": round(days, 1)}


def unmatched_cardlike(images: dict, verdicts: dict, records: dict, channels: dict) -> list:
    """Card-like images whose OCR'd title matches no card: retired or never uploaded cards."""
    groups = defaultdict(list)
    for key, image in images.items():
        v = verdicts.get(key)
        if not v or v["status"] != "none" or not image.get("shape") or not image.get("title"):
            continue
        title = image.get("title_block") or image["title"][0]
        if len(alnum(title)) >= 4:
            groups[squash(title)].append((records[image["message"]], image, title))
    found = []
    for items in groups.values():
        items.sort(key=lambda t: int(t[0]["id"]))
        record, image, title = items[0]
        found.append({"title": title, "count": len(items), "first": record.get("timestamp"),
                      "jump_url": record.get("jump_url"), "poster": (record.get("author") or {}).get("username"),
                      "image": image["file"], "thumb": image.get("thumb")})
    return sorted(found, key=lambda f: (-f["count"], f["first"] or ""))


def find_anomalies(cards: dict, server_born: Optional[float]) -> list:
    found = []
    for cid, card in cards.items():
        discord = card.get("discord") or {}
        first, check = discord.get("first"), discord.get("fs_check") or {}
        if not first:
            continue
        when = epoch(first["date"])
        if server_born and when and when < server_born - 86400:
            found.append({"card": cid, "kind": "before_server", "detail": f"{first['date']} predates the server"})
        if check.get("verdict") == "git_earlier" and check["days_discord_before_git"] < -GIT_EARLIER_DAYS:
            found.append({"card": cid, "kind": "git_earlier", "detail": f"first Discord post {first['date'][:10]} is "
                          f"{-check['days_discord_before_git']:.0f} days after the git upload {check['fs'][:10]}"})
        pva = discord.get("poster_vs_author") or {}
        if pva.get("verdict") == "different":
            found.append({"card": cid, "kind": "poster_not_author", "detail": f"first posted by {pva['poster']} "
                          f"({pva['person']}), credited to {pva['credited']}"})
        mention = discord.get("first_mention")
        if mention and when and epoch(mention["date"]) and when - epoch(mention["date"]) > 90 * 86400:
            found.append({"card": cid, "kind": "named_before_shown", "detail": f"named in chat {mention['date'][:10]}, "
                          f"first shown {first['date'][:10]}"})
        if discord.get("earlier_ambiguous"):
            found.append({"card": cid, "kind": "earlier_ambiguous", "detail": "an earlier post may show it: "
                          + ", ".join(discord["earlier_ambiguous"][:6])})
        if discord.get("to_confirm"):
            found.append({"card": cid, "kind": "first_to_confirm", "detail": f"first post at {first['conf']:.3f}: "
                          + ", ".join(discord["to_confirm"])})
    return found


def tally(cards: dict, images: dict, verdicts: dict, entries: list, records: dict) -> dict:
    tiers = Counter((c.get("discord") or {}).get("tier") or "none" for c in cards.values())
    checks = Counter(((c.get("discord") or {}).get("fs_check") or {}).get("verdict") for c in cards.values()
                     if (c.get("discord") or {}).get("first"))
    return {"cards": len(cards), "cards_dated": sum(1 for c in cards.values() if (c.get("discord") or {}).get("first")),
            "tiers": dict(tiers), "fs_checks": dict(checks), "messages": len(records), "images": len(images),
            "images_by_status": dict(Counter(v["status"] for v in verdicts.values())),
            "ambiguous_entries": len(entries), "ambiguous_open": sum(1 for e in entries if e["ruling"] is None),
            "ambiguous_affecting_dates": sum(1 for e in entries if e["affects_earliest"] and e["ruling"] is None),
            "confirm_open": sum(1 for e in entries if e["confirm"] and e["ruling"] is None),
            "suspect_entries": sum(1 for e in entries if e.get("suspect") and e["ruling"] is None),
            "suspect_cards": sum(1 for c in cards.values() if (c.get("discord") or {}).get("suspect")),
            "images_by_kind": {"sheets": sum(1 for i in images.values() if i.get("cells")),
                               "several_no_grid": sum(1 for i in images.values() if i.get("multi")
                                                      and not i.get("cells")),
                               "twin_verdicts": sum(1 for v in verdicts.values() if v.get("twin")),
                               "settled_clashes": sum(1 for v in verdicts.values() if v.get("settled"))},
            "rulings": sum(1 for e in entries if e["ruling"] is not None)}


def thresholds() -> dict:
    return {"hash_full_bits": HASH_FULL, "hash_zero_bits": HASH_ZERO, "title_ratio": [TITLE_FLOOR, TITLE_FULL],
            "line_ratio": [LINE_FLOOR, LINE_FULL], "body_share": [BODY_FLOOR, BODY_FULL], "weights": WEIGHTS,
            "confident": CONFIDENT, "plausible": PLAUSIBLE, "strong_vote": STRONG, "odds": ODDS,
            "ocr_fit": [OCR_FLOOR, OCR_FULL], "text_min": TEXT_MIN, "look": {"bits": LOOK_BITS, "grams": LOOK_GRAMS},
            "beat": {"hits": BEAT_HITS, "share": BEAT_SHARE, "absent": [ABSENT_GRAMS, ABSENT_SHARE]},
            "hash_tie_bits": HASH_TIE, "draft": {"run": DRAFT_RUN, "bits": DRAFT_BITS}, "review": REVIEW}


# ========================
# Pages
# ========================
HTML_HEAD = """<!doctype html><meta charset="utf-8"><title>Discord matches: ambiguous posts</title>
<style>
body{font:14px/1.4 system-ui,sans-serif;margin:0 0 9rem;background:#1e1f22;color:#dbdee1}
header{padding:1rem 1.5rem;background:#2b2d31}h1{margin:0 0 .3rem;font-size:1.3rem}
section{margin:1rem 1.5rem;padding:1rem;background:#2b2d31;border-radius:8px}
section.ruled{opacity:.55}
h2.group{margin:1.6rem 1.5rem .4rem;font-size:1.15rem;color:#f0b232}h2{margin:0;font-size:1.1rem}h2 small{font-weight:normal;color:#949ba4}
.meta{color:#b5bac1;margin:.3rem 0 .6rem}.row{display:flex;gap:.8rem;flex-wrap:wrap;align-items:flex-start}
figure{margin:0;width:200px}figure.post{width:300px}figure.shot{width:480px}img{max-width:100%;border-radius:4px;background:#111}
figcaption{font-size:12px;color:#b5bac1;margin-top:.3rem}.best{outline:2px solid #23a55a;outline-offset:3px}
button{margin-top:.3rem;background:#5865f2;color:#fff;border:0;border-radius:4px;padding:.25rem .6rem;cursor:pointer}
button.none{background:#4e5058}.chosen{background:#23a55a!important}pre{white-space:pre-wrap;color:#b5bac1}
.flag{color:#f0b232}a{color:#00a8fc}footer{position:fixed;bottom:0;left:0;right:0;background:#111214;padding:.6rem 1.5rem}
footer textarea{width:calc(100% - 9rem);height:3.2rem;background:#2b2d31;color:#dbdee1;border:0;border-radius:4px}
</style>"""

HTML_SCRIPT = """<script>
const picks = {};                                   // id -> chosen values (screenshots: several)
function pick(btn) {
  const id = btn.dataset.id, v = btn.dataset.v, many = btn.closest("section").dataset.multi === "1";
  let chosen = picks[id] || [];
  if (v === "none" || !many) chosen = chosen.length === 1 && chosen[0] === v ? [] : [v];
  else chosen = chosen.includes(v) ? chosen.filter(x => x !== v) : chosen.filter(x => x !== "none").concat([v]);
  picks[id] = chosen;
  document.querySelectorAll(`button[data-id="${id}"]`).forEach(b => b.classList.toggle("chosen", chosen.includes(b.dataset.v)));
  document.getElementById("out").value = Object.entries(picks).filter(([k, c]) => c.length)
    .map(([k, c]) => `${k} = ${c.join(" + ")}`).join(", ");
}
function copyOut() { const o = document.getElementById("out"); o.select(); navigator.clipboard.writeText(o.value); }
</script>"""


def _ruling_value(cid: str, cards: dict) -> str:
    """The name to write in a ruling: the printed title when it names one card, else the id."""
    name = cards[cid].get("name") or cid
    same = [c for c, card in cards.items() if squash(card.get("name")) == squash(name)]
    return name if len(same) == 1 and "+" not in name and "," not in name else cid


GROUPS = {0: "Could change a card's first date, or a first post to confirm", 1: "The rest: a later post, or no card "
          "yet dated by it (revisions and counts only)", 2: "Ruled"}


def entry_group(entry: dict) -> int:
    """The page's sections, in the order entries are sorted."""
    return 2 if entry.get("ruling") is not None else 0 if entry["affects_earliest"] or entry["confirm"] else 1


def render_html(entries: list, cards: dict, result: dict) -> str:
    parts = [HTML_HEAD, HTML_SCRIPT, "<header><h1>Ambiguous posts</h1>"]
    counts = result["counts"]
    parts.append(f"<div>{counts['ambiguous_open']} open of {len(entries)} entries "
                 f"({counts['ambiguous_affecting_dates']} of the open could change a card's first date, "
                 f"{counts['confirm_open']} are a first post to confirm; they come first, those of cards now "
                 f"dated after their git upload ({counts.get('suspect_entries', 0)}) at the top). Bot posts never "
                 f"date a card and aren't listed. Pick a card (or none) per entry, then copy the line below and reply with it, or run "
                 f"<code>python3 -I apply_rulings.py \"A12 = Isma, A13 = none\"</code>. Several cards in one image: "
                 f"<code>A7 = Isma + Abbi</code>. Generated {escape(result['generated'])}.</div>")
    status = (result.get("mining") or {}).get("status")
    if status and status != "done":
        parts.append(f"<div class=flag>Mining: {escape(str(status))}</div>")
    parts.append("</header>")
    if not entries:
        parts.append("<section>Nothing to rule on.</section>")
    groups = Counter(entry_group(e) for e in entries)
    group = None
    for e in entries:
        if entry_group(e) != group:
            group = entry_group(e)
            parts.append(f"<h2 class=group>{escape(GROUPS[group])} ({groups[group]})</h2>")
        post, aid = e["post"], e["id"]
        ruled = e.get("ruling")
        many = not e.get("shape") or e.get("multi")
        parts.append(f"<section id={aid} data-multi={int(many)} class='{'ruled' if ruled else ''}'><h2>{aid} <small>"
                     f"{escape(e['reason'] or e['status'])}"
                     + (f" · <span class=flag>may be the first post of "
                        f"{escape(', '.join(cards[c].get('name', c) for c in e['affects_earliest']))}</span>"
                        if e["affects_earliest"] else "")
                     + (f" · <span class=flag>dated after its git upload now: "
                        f"{escape(', '.join(cards[c].get('name', c) for c in e['suspect']))}</span>"
                        if e.get("suspect") else "")
                     + (f" · <span class=flag>dates {escape(', '.join(cards[c].get('name', c) for c in e['confirm']))}"
                        f": confirm (pick every card it shows, or none)</span>" if e.get("confirm") else "")
                     + (f" · ruled: {escape(' + '.join(ruled['cards']) or 'none')}" if ruled else "")
                     + "</small></h2>")
        where = " › ".join(escape(p) for p in (post.get("category"), "#" + str(post.get("channel")),
                                                post.get("thread")) if p and p != "#None")
        parts.append(f"<div class=meta>{escape(str(post.get('date'))[:16].replace('T', ' '))} · {where} · "
                     f"{escape(poster_name(post))}{' (bot)' if post.get('bot') else ''} · "
                     f"<a href='{escape(post.get('jump_url') or '')}'>jump</a> · best guess: "
                     f"{escape(', '.join(cards[c].get('name', c) for c in e['best_guess'] if c in cards) or '-')}"
                     + (" · <b>several cards: pick every card it shows</b>" if many else "") + "</div><div class=row>")
        parts.append(f"<figure class='post{' shot' if many else ''}'><a href='{escape(e['image'])}'><img loading=lazy "
                     f"src='{escape(e.get('thumb') or e['image'])}'></a><figcaption>posted image "
                     f"{escape(e.get('filename') or '')}</figcaption></figure>")
        for i, c in enumerate(e["candidates"], 1):
            methods = ", ".join(f"{m} {v:.2f}" for m, v in sorted(c["s"].items(), key=lambda kv: -kv[1]))
            why = c.get("why") or {}
            extra = []
            if why.get("hash_bits") is not None:
                extra.append(f"{why['hash_bits']} bits ({why.get('hash_part')})")
            if why.get("title"):
                extra.append(f"read “{escape(why['title'])}”")
            if why.get("text"):
                extra.append(f"chat: “{escape(why['text'])}” ({why.get('text_where')})")
            value = _ruling_value(c["card"], cards)
            parts.append(f"<figure class='{'best' if c['card'] in e['best_guess'] else ''}'>"
                         + (f"<img loading=lazy src='{escape(c['thumb'])}'>" if c.get("thumb") else "")
                         + f"<figcaption>#{i} <b>{escape(c['name'] or c['card'])}</b> ({escape(c['card'])}) · "
                         f"{c['conf']:.2f}<br>{escape(methods)}<br>{'; '.join(extra)}<br>"
                         f"<button data-id={aid} data-v=\"{escape(value)}\" onclick=pick(this)>{aid} = "
                         f"{escape(value)}</button></figcaption></figure>")
        parts.append(f"<figure><button class=none data-id={aid} data-v=none onclick=pick(this)>{aid} = none</button>"
                     f"</figure></div>")
        ocr = "\n".join(["title: " + " | ".join(e["ocr_title"])] if e["ocr_title"] else [])
        ocr += ("\n" if ocr else "") + "\n".join(e["ocr_lines"][:25])
        if ocr:
            parts.append(f"<details><summary>OCR</summary><pre>{escape(ocr)}</pre></details>")
        if e.get("settled"):
            parts.append(f"<div class=meta>settled: {escape('; '.join(e['settled']))}</div>")
        if post.get("excerpt"):
            parts.append(f"<pre>{escape(post['excerpt'])}</pre>")
        parts.append("</section>")
    parts.append("<footer><textarea id=out placeholder='your rulings appear here'></textarea> "
                 "<button onclick=copyOut()>Copy</button></footer>")
    return "\n".join(parts)


def _pct(values: list, q: float):
    if not values:
        return None
    values = sorted(values)
    return round(values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))], 2)


def _method_lines(selftest: Optional[dict]) -> list:
    """REPORT.md's method and calibration sections."""
    out = []
    out += ["## Method", "",
            "Each stored image is compared with every card version (the PNG in the checkout, plus versions from the "
            "site's git history) and with every card name; the methods' scores combine into one confidence "
            "(1 - product of (1 - weight x score)):", "",
            "| method | what | weight |", "|---|---|---|",
            f"| hash | pHash 64 + dHash 64 + pHash 256/4 of the whole card or its art; score 1 at <= {HASH_FULL} bits, "
            f"0 at >= {HASH_ZERO} | {WEIGHTS['hash']} |",
            f"| title | Vision OCR of the title band (card-shaped images; and one to three lines above {TITLE_LOW:.0%} "
            f"of the height within {TITLE_LINE} of a name: 2020's cards print the title under the art) vs printed "
            f"titles, keys, aliases, file names and the versions' OCR'd titles; ratio {TITLE_FLOOR}..{TITLE_FULL} | "
            f"{WEIGHTS['title']} |",
            f"| line / anchor | screenshots: each OCR'd line vs the names, and the art under that line vs the card's "
            f"art | {WEIGHTS['line']} / {WEIGHTS['anchor']} |",
            f"| ocr | card-shaped images: their OCR'd text vs each card version's OCR'd text (share of the image's "
            f"4-grams found, template grams dropped; {OCR_FLOOR}..{OCR_FULL}): finds crops and old versions "
            f"| {WEIGHTS['ocr']} |",
            f"| body | OCR'd text vs the card's ability (rare-word share {BODY_FLOOR}..{BODY_FULL}, {BODY_WORDS}+ shared "
            f"words, scaled down under {BODY_SURE}: a short ability's words turn up in longer texts) | "
            f"{WEIGHTS['body']} |",
            f"| file | attachment file name vs names (generic names ignored) | {WEIGHTS['file']} |",
            f"| text | names in the message, its thread, the poster's nearby messages, replies; \"new card\" wording "
            f"+0.05 | {WEIGHTS['text']} |", "",
            "Art is not the card: when no text on the image backs a card (no title, OCR'd text, ability or line "
            f"score of 0.3) and the whole image doesn't hash as it (hash score {WHOLE_SURE}+ on the whole), the art's "
            "hash, the file name or chat alone say the post shows its art (status art, kept as `art_first`, never a "
            f"date) when the image reads under {ART_TEXT} characters and no stat label, else a draft under another "
            "title (ambiguous).", "",
            "Titles: a read with a word of its own where the name has another (Cape of the Old King vs Heir of the "
            f"Old King: under {SWAP_WORD} alike) isn't that name; a name that is only part of a title printed on two "
            "lines (one height, centred alike) naming a thing of it (Cape of the Old King) counts by the share it "
            "covers; a line of such a title isn't a piece of another name. Each line read once (the fullest read) "
            "before lines are joined. Screenshots: "
            f"a name read with nothing else behind it counts {LINE_ALONE:g}x when the image isn't card-like or "
            f"names more than {LIST_NAMES} cards so (a list, a meme's caption).", "",
            f"A card-shaped image shows one card: confident when the best reaches {CONFIDENT}, the runner-up's risk is "
            f"{ODDS:g}x the best's, no two methods strongly ({STRONG}) name different cards unsettled, and a short "
            "title isn't the only evidence. Methods that disagree are settled by kind (look: the whole card's hash; "
            "content: OCR'd text vs the versions and the ability; name: title, file name, chat): a card strong in a "
            f"kind the other has nothing of (under {NO_SUPPORT}) wins it; one card winning decides, and when both do, "
            "the look or the versions' text beats a name alone (an old name on the card's own image). "
            f"A sheet (two card names side by side on one row, {SHEET_GAP:g}+ apart) is cut into its card-shaped "
            "cells, each judged as a card; several cards in no grid are read like a screenshot. A card-shaped post "
            f"within {TWIN_BITS:g} bits (whole) of a sure one-card post is the same image: sure as it, unless its "
            "own title or text names another card. Screenshots may show several. Everything from "
            f"{PLAUSIBLE} with some image evidence "
            "that isn't confident is an ambiguous entry, as is any plausible post earlier than a card's first "
            f"confident one, and a first post under {REVIEW} is one to confirm. Bot and webhook posts never date a "
            "card. Rulings are the owner's only.", "",
            "Which version a post shows: each card version is OCR'd three times (full size, 900 and 600 wide: "
            "misreads differ by size); a look is the versions no image could tell apart (whole-card hash within "
            f"{LOOK_BITS} bits, at most {LOOK_GRAMS} 4-grams in all reads of one and no read of the other). A "
            f"card-shaped post with {TEXT_MIN}+ OCR'd characters keeps the looks no other look beats: look A beats "
            f"B when the post reads {BEAT_HITS}+ of A's own 4-grams (in all of A's reads, none of B's), a share "
            f"{BEAT_SHARE} over B's; an uncropped post lacking B's own grams ({ABSENT_GRAMS}+, under "
            f"{ABSENT_SHARE:.0%} read) counts too. Looks left with the same text: the whole-card hash within "
            f"{HASH_TIE} bits. Without text the hash decides only when near (<= {LOOK_BITS} bits): text-only edits "
            "move it 1-3 bits, a crop 8-14, so a crop without text tells nothing. A post is told when one look "
            f"is left; an uncropped one reading {DRAFT_RUN}+ characters in a row that no read of its look has (or, "
            f"without text, its art matching and the card {DRAFT_BITS}+ bits off) shows a look not in git (a "
            "draft). Revisions are the first post of each look "
            "after the earliest; current_since is the first post of the current look.", ""]
    if selftest:
        out += ["## Calibration (self-test on the cards)", "",
                f"Every current card PNG ({selftest.get('cards')}) degraded like a Discord post and matched against all "
                f"{selftest.get('versions')} card versions (no chat text, no file names):", "",
                "| variant | n | hash top-1 | title OCR top-1 | confident right | confident wrong | ambiguous | none | "
                "true bits p50/p95 | nearest other p5/p50 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for name, v in (selftest.get("variants") or {}).items():
            out.append(f"| {name} | {v['n']} | {v['hash_top1']:.1%} | "
                       f"{'-' if v.get('title_top1') is None else format(v['title_top1'], '.1%')} | "
                       f"{v['confident_right']:.1%} | {v['confident_wrong']:.1%} | {v['ambiguous']:.1%} | "
                       f"{v['none']:.1%} | {v['true_bits_p50']}/{v['true_bits_p95']} | "
                       f"{v['other_bits_p5']}/{v['other_bits_p50']} |")
        old = selftest.get("old_versions") or {}
        if old.get("n"):
            out += ["", f"Older versions: {old['n']} cards' oldest git version vs today's PNG: median "
                        f"{old['bits_p50']} bits, {old['within_zero']:.0%} within {HASH_ZERO} bits of the current one "
                        f"(the rest would be missed by hashing today's PNG alone; their old versions are indexed).",
                    f"Older versions whose OCR'd title isn't today's: {old.get('renamed', 0)}, nearly all stylized fonts "
                    "misread (Vision rates them 1.0 too), so a title read off an image becomes a former-name "
                    "candidate only when the post's text or file name agrees."]
        check = selftest.get("version_check") or {}
        if check.get("variants"):
            out += ["", f"Version pick: every version ({check['versions']}) of the {check['cards']} cards with more "
                        f"than one look ({check['looks']} looks), degraded, then its version picked among the card's "
                        "own (right: its look alone; tied: looks tied, untold; draft: taken for a look not in git):",
                    "", "| variant | right | wrong | tied | draft | none |", "|---|---|---|---|---|---|"]
            for name, c in check["variants"].items():
                n = max(1, sum(c.values()))
                out.append(f"| {name} | " + " | ".join(f"{c.get(k, 0) / n:.1%}" for k in
                                                       ("right", "wrong", "tied", "draft", "none")) + " |")
            held = check.get("held_out") or {}
            if held:
                far = held.get("draft_far", 0) + held.get("taken_far", 0)
                near = held.get("draft_near", 0) + held.get("taken_near", 0)
                out.append(f"\nDrafts (looks not in git): each version's own look left out, as if never committed "
                           f"(discord variant): called a draft {held.get('draft_far', 0) / max(1, far):.0%} of the "
                           f"{far} whose look has 12+ 4-grams of its own, "
                           f"{held.get('draft_near', 0) / max(1, near):.0%} of the {near} nearer ones (those are "
                           "taken for the nearest look). Drafts are a lower bound.")
            found = check.get("old_found") or {}
            if found:
                n = max(1, sum(found.values()))
                out.append(f"\nOlder versions as posts (discord variant, {n}): confident right "
                           f"{found.get('confident_right', 0) / n:.1%}, wrong "
                           f"{found.get('confident_wrong', 0) / n:.1%}, ambiguous {found.get('ambiguous', 0) / n:.1%}.")
        for line in selftest.get("notes") or []:
            out.append(f"- {line}")
        out.append("")
    return out


TIER_TEXT = {"ruled": "the owner ruled on its first post", "high": "confidence 0.98+ with strong image evidence (the "
             "card's hash or OCR'd title, plus another method)", "confident": "confidence 0.85+, no close rival, "
             "methods agree", "ambiguous": "no sure post; an ambiguous one may show it", "art": "only its art (or a "
             "file named for it) was posted", "mention": "named in chat only", "none": "nothing found"}


def _dated(result: dict) -> list:
    return [(cid, c) for cid, c in sorted(result["cards"].items()) if (c.get("discord") or {}).get("first")]


def _summary_lines(result: dict, entries: list) -> list:
    """REPORT.md's findings: coverage, Discord vs fs, posters vs credits, names, revisions, metadata, proposal."""
    counts, cards, proposal = result["counts"], result["cards"], result.get("proposal") or {}
    dated = _dated(result)
    tiers = counts["tiers"]
    waiting = [cid for cid, c in dated if c["discord"].get("earlier_ambiguous") or c["discord"].get("to_confirm")]
    checks = counts["fs_checks"]
    earlier = sorted((c["discord"]["fs_check"]["days_discord_before_git"], cid, c) for cid, c in dated
                     if c["discord"]["fs_check"].get("verdict") == "discord_earlier")
    fs_months = Counter((c.get("fs") or "")[:7] for c in cards.values() if c.get("fs"))
    bulk = fs_months.most_common(1)[0] if fs_months else ("", 0)
    pvas = [c["discord"]["poster_vs_author"] for _, c in dated if c["discord"].get("poster_vs_author")]
    verdicts = Counter(p["verdict"] for p in pvas)
    out = ["## Summary", "",
           f"- **{counts['cards_dated']} of {counts['cards']} cards** have a creation date from Discord (their "
           f"earliest post showing the card): {tiers.get('high', 0)} high, {tiers.get('confident', 0)} confident, "
           f"{tiers.get('ruled', 0)} ruled. {len(waiting)} of these could still move: an earlier ambiguous post or a "
           f"first post to confirm waits on a ruling. {counts['cards'] - counts['cards_dated']} have no date: "
           + ", ".join(f"{tiers[k]} {TIER_TEXT[k].split(';')[0]}" for k in ("ambiguous", "art", "mention", "none")
                       if tiers.get(k)) + ".",
           f"- The site's created sort uses fs, the first git upload: {bulk[1]} cards share {bulk[0]} (the repo's "
           f"first bulk upload), so fs says little before then. Discord is earlier for "
           f"{checks.get('discord_earlier', 0)} cards (median {_pct([d for d, _, _ in earlier], 0.5) or 0:.0f} days), "
           f"about the same (within 2 days) for {checks.get('same_time', 0)}, later for "
           f"{checks.get('git_earlier', 0)}.",
           f"- First posters vs credited authors: {verdicts.get('same', 0)} same person, "
           f"{verdicts.get('different', 0)} different, {verdicts.get('ported', 0)} first seen in a port channel "
           f"(re-uploads, not authorship), {verdicts.get('unmapped', 0)} with a poster not mapped to a person.",
           f"- Ambiguous posts: {counts['ambiguous_open']} open in `ambiguous.html`; "
           f"{counts['ambiguous_affecting_dates']} could change a first date and {counts['confirm_open']} are first "
           f"posts to confirm: those come first on the page, led by the {counts.get('suspect_entries', 0)} entries "
           f"of the {counts.get('suspect_cards', 0)} cards dated after their git upload (suspect, below). Bot posts "
           "aren't listed: they never date a card.",
           f"- Proposals (in `proposal/`, nothing applied): {proposal.get('created_ready', 0)} created dates ready "
           f"and {proposal.get('created_pending', 0)} after rulings ({proposal.get('created_if_ruled', 0)} more if a "
           f"suspect card's earliest open post is ruled for it), {proposal.get('aliases', 0)} aliases on "
           f"{proposal.get('alias_cards', 0)} cards, {proposal.get('authors', 0)} author corrections.", ""]

    out += ["## Coverage", "", "| tier | meaning | cards |", "|---|---|---|"]
    out += [f"| {k} | {TIER_TEXT[k]} | {tiers[k]} |" for k in TIER_TEXT if tiers.get(k)]
    years = Counter(c["discord"]["first"]["date"][:4] for _, c in dated)
    places = Counter(c["discord"]["first"]["channel"] for _, c in dated)
    shows = Counter((c["discord"]["first"].get("version") or {}).get("shows") or "untold" for _, c in dated)
    out += ["", "- First posts by year: " + ", ".join(f"{y} {n}" for y, n in sorted(years.items())) + ".",
            "- By channel: " + ", ".join(f"#{p} {n}" for p, n in places.most_common(10)) + ".",
            f"- What the first post shows: an older git version {shows.get('older', 0)}, the current one "
            f"{shows.get('current', 0)}, a draft not in git {shows.get('draft', 0)}, can't tell "
            f"{shows.get('untold', 0)} (screenshots, crops, versions alike).",
            "- Images: " + ", ".join(f"{k} {v}" for k, v in sorted(counts["images_by_status"].items())) + " (art: a "
            "card's art or a file named for it, never creation evidence).",
            f"- Sheets of several cards read cell by cell: {counts['images_by_kind']['sheets']}; several cards in no "
            f"grid (read like screenshots): {counts['images_by_kind']['several_no_grid']}. Posts sure as the twin "
            f"of a sure post (the same image again): {counts['images_by_kind']['twin_verdicts']}. Card-shaped posts "
            f"whose methods disagreed but one card's evidence settled it: "
            f"{counts['images_by_kind']['settled_clashes']}.", ""]

    out += ["## Discord date vs fs (the site's first git upload)", "",
            f"- {checks.get('discord_earlier', 0)} earlier on Discord, {checks.get('same_time', 0)} the same, "
            f"{checks.get('git_earlier', 0)} later (each with `days_discord_before_git` in matches.json)."]
    if earlier:
        out.append("- The largest gaps:")
        for days, cid, c in sorted(earlier, reverse=True)[:12]:
            out.append(f"  - {c['name']}: {c['discord']['first']['date'][:10]} vs fs {c['fs'][:10]} ({days:.0f} days; "
                       f"{c['discord']['tier']}, #{c['discord']['first']['channel']})")
    later = [(cid, c) for cid, c in dated if c["discord"].get("suspect")]
    if later:
        out.append(f"- **Suspect: {len(later)} cards are dated after their git upload.** A card is made before it "
                   "is uploaded, so the date is likely wrong: an earlier post was missed (left ambiguous, or not "
                   "matched). Their open earlier posts come first in `ambiguous.html`; a ruling on the earliest "
                   "gives `created_if_ruled` in the proposal:")
        for _, c in later:
            d = c["discord"]
            out.append(f"  - {c['name']}: {d['first']['date'][:10]} vs fs {c['fs'][:10]}; "
                       + (f"open earlier posts {', '.join(d['earlier_ambiguous'][:8])}"
                          + (f" (earliest {d['earliest_open']['id']}, {d['earliest_open']['date'][:10]})"
                             if d.get("earliest_open") else "") if d.get("earlier_ambiguous")
                          else "no earlier post found: check by hand"))
    out.append("")

    mapping = result.get("people_map") or {}
    out += ["## First posters vs credited authors", "",
            "Discord posters mapped to the site's people (`[people]` of Card Authors.txt) by name, else by the "
            "credits of the cards they posted first outside the port channels:", ""]
    for m in sorted(mapping.values(), key=lambda m: -m["first_posts"]):
        out.append(f"- {m.get('global_name') or m.get('username')} (@{m.get('username')}): **{m['person'] or '?'}** "
                   f"({m['basis']}); first poster of {m['first_posts']} cards ({m.get('port_first_posts', 0)} in "
                   "port channels)")
    by = Counter((p["person"], p["credited"]) for p in pvas if p["verdict"] == "different")
    if by:
        out += ["", "Different person (first poster -> credited now): " + ", ".join(
            f"{a} -> {b} {n}" for (a, b), n in by.most_common()) + ". Each is a line in "
            f"`proposal/{AUTHORS_FILE}`; most come from batches one person posted (a remake or a post for "
            "someone else is possible), so check before copying."]
    out.append("")

    kinds = {"former_titles": "Former printed titles (read off its posted images, backed by the post's text or "
                              "file name)", "file_names": "File names that aren't a known name",
             "captions": "Short captions posted with it"}
    out += ["## Names used on Discord that the site lacks", ""]
    shared = Counter(squash(r["text"]) for _, c in dated for r in (c["discord"].get("names") or {}).get("captions", []))
    roles = Counter(r["text"] for _, c in dated for r in (c["discord"].get("names") or {}).get("captions", [])
                    if shared[squash(r["text"])] >= 3)
    for kind, label in kinds.items():
        rows = [(c["name"], r) for _, c in dated for r in (c["discord"].get("names") or {}).get(kind, [])
                if kind != "captions" or shared[squash(r["text"])] < 3]
        out.append(f"- {label}: {len(rows)}" + (": " + "; ".join(f"{a}: “{r['text']}”" + (f" x{r['count']}" if
                                                                                         r['count'] > 1 else "")
                                                         for a, r in rows[:14]) if rows else "") + ".")
    if roles:
        out.append("- Captions used on 3+ cards are roles or remarks, not names (classification posts): "
                   + ", ".join(f"“{t}” x{n}" for t, n in roles.most_common(12)) + ".")
    out += [f"- Proposed as aliases (former titles and file names naming no other card): {proposal.get('aliases', 0)} "
            f"on {proposal.get('alias_cards', 0)} cards, in `proposal/{EDITS_FILE}`. Captions are left out (often "
            "not names); they are in matches.json per card.", ""]

    revised = [(c["name"], c["discord"]) for _, c in dated if c["discord"].get("revisions")]
    since = sum(1 for _, c in dated if c["discord"].get("current_since"))
    out += ["## Revisions", "",
            f"- {len(revised)} cards show more than one look on Discord "
            f"({sum(len(d['revisions']) for _, d in revised)} revision posts: the first post of each later look); "
            f"{since} have the date their current look was first posted (current_since).", "- Examples:"]
    for name, d in sorted(revised, key=lambda t: -len(t[1]["revisions"]))[:12]:
        out.append(f"  - {name}: first {d['first']['date'][:10]} ({d['first']['version'].get('shows') or 'untold'}), "
                   + ", ".join(f"{r['date'][:10]} {r['shows']}" for r in d["revisions"])
                   + f"; current since {(d.get('current_since') or '-')[:10]}")
    out.append("")

    art = [(c["name"], c["discord"]) for _, c in dated if c["discord"].get("art_first")]
    art_only = [c["name"] for c in cards.values() if (c.get("discord") or {}).get("tier") == "art"]
    leads = [(epoch(d["first"]["date"]) - epoch(d["art_first"]["date"])) / 86400 for _, d in art
             if epoch(d["art_first"]["date"]) < epoch(d["first"]["date"])]
    notes = sum(len((c.get("discord") or {}).get("notes") or []) for c in cards.values())
    named = [a for a in result["anomalies"] if a["kind"] == "named_before_shown"]
    unmatched = result.get("unmatched_cardlike") or []
    out += ["## Other metadata kept (matches.json)", "",
            f"- Art before the card: {len(art)} dated cards had their art posted first (median "
            f"{_pct(leads, 0.5) or 0:.0f} days earlier, `art_first`); {len(art_only)} cards only ever as art"
            + (f" ({', '.join(art_only[:8])})" if art_only else "") + ".",
            f"- Balance and lore remarks (replies, threads, messages naming the card): {notes}, per card in `notes`.",
            f"- First chat mention: {sum(1 for c in cards.values() if (c.get('discord') or {}).get('first_mention'))} "
            f"cards; {len(named)} named 90+ days before first shown (many names are the games' own, so a mention isn't "
            "creation evidence).",
            "- Per card also: every poster of it (`posters`), the channels and threads it was posted in "
            "(`series_hints`), the first and last post, links to each.",
            "- Card-shaped images whose title matches no card (retired, renamed past recognition, or never "
            f"uploaded): {len(unmatched)}" + (": " + "; ".join(
                f"“{u['title']}” x{u['count']} ({(u['first'] or '')[:10]})" for u in unmatched[:12])
                if unmatched else "") + ".", ""]

    example = next(((c.get("key"), c) for _, c in dated if c["discord"].get("tier") in SURE_TIERS
                    and c["discord"]["fs_check"].get("verdict") == "discord_earlier"), (None, None))
    out += ["## Proposal (nothing is applied or published)", "",
            "1. **created**, for the site's created sort: the date (day only) of the card's first Discord post, "
            "when its tier is high, confident or ruled and it is earlier than fs. Written as a new database-only "
            "field in TTSCardMaker's `site/Card Edits.json`, keyed like the other edits:", "",
            "   ```json", f"   \"{example[0] or 'Card Name'}\": {{\"created\": "
            f"\"{(example[1] or {}).get('discord', {}).get('first', {}).get('date', '2020-07-22')[:10]}\", \"_by\": "
            "\"MichaelJSr\", \"_at\": \"<now>\"}", "   ```", "",
            "   The site needs `created` added where the edit fields are listed (build_site.py EDIT_FIELDS, edit.js "
            "FIELDS, AutoCroissant's CardEdits.TYPES with KIND `database`), and the created sort to use created, "
            "else fs. Only dates leave this folder: no message text, links or Discord names. "
            f"{proposal.get('created_ready', 0)} are ready, {proposal.get('created_pending', 0)} wait on rulings "
            f"(`pending`, with their ids): `proposal/{EDITS_FILE}`.",
            "2. **Author corrections**: `[cards]` lines for `site/Card Authors.txt` (`Card Name = person`, replacing "
            f"the card's current line), {proposal.get('authors', 0)} in `proposal/{AUTHORS_FILE}` with the evidence "
            "in a comment. Confirm the Discord -> person map above first.",
            "3. **Aliases**: Card Edits.json `aliases` (database-only; the card's current edit aliases plus the new "
            f"ones), in the same file as created: {proposal.get('aliases', 0)} names on "
            f"{proposal.get('alias_cards', 0)} cards.",
            "4. **History**: each look's first post (revisions, current_since) could date the card view's image "
            "history, as dates only; the images are other people's posts and stay here.",
            "5. **Retired cards**: the unmatched card-shaped titles are candidates for the archive of retired cards, "
            "to check by hand.", "",
            "## Rulings", "",
            "Open `ambiguous.html`, pick a card (or none) per entry, then reply with the line it builds "
            "(`A12 = Isma, A13 = none`) or run `python3 -I apply_rulings.py \"...\"`: rulings are kept in "
            "`match/rulings.json`, and matches.json, ambiguous.*, `proposal/` and this report are rebuilt in "
            "seconds. Ids never change.", ""]
    return out


def _anomaly_lines(result: dict, cards: dict) -> list:
    out = ["## Anomalies (all in matches.json)", ""]
    kinds = Counter(a["kind"] for a in result["anomalies"])
    out.append(f"- {', '.join(f'{k} {v}' for k, v in sorted(kinds.items())) or 'none'}.")
    for kind in ("git_earlier", "first_to_confirm", "earlier_ambiguous"):
        rows = [a for a in result["anomalies"] if a["kind"] == kind]
        if rows:
            out.append(f"- {kind}: " + "; ".join(f"{cards.get(a['card'], {}).get('name', a['card'])} ({a['detail']})"
                                                 for a in rows[:12]) + ("; ..." if len(rows) > 12 else ""))
    return out + [""]


def render_report(result: dict, entries: list, selftest: Optional[dict], cards: dict) -> str:
    counts, mining = result["counts"], result.get("mining") or {}
    out = ["# Discord archive: cards matched to posts", "",
           f"Generated {result['generated']} by `commands/discord_match.py` (AutoCroissant). Raw data and this "
           "report stay in this folder: nothing here is published.", ""]
    out += ["## Status", ""]
    totals = mining.get("totals") or {}
    out.append(f"- Mining (`summary.json`): **{mining.get('status') or 'no run yet'}**; "
               f"{totals.get('messages', 0)} messages, {totals.get('images', 0)} images stored at its end "
               f"({totals.get('download_failures', 0)} downloads failed, {totals.get('too_big', 0)} too big).")
    if mining.get("status") and mining["status"] != "done":
        out.append("- Until the miner finishes, everything below counts only what was stored. Once the bot can read "
                   "the server: `python -m commands.discord_archive --guild <id> --channels <ids> --all`, then "
                   "`python -m commands.discord_match score` (rerunnable; caches by content).")
    out.append(f"- Scored {counts['images']} stored images (retried downloads included) from {counts['messages']} "
               "messages; OCR "
               f"{'on' if result.get('ocr') else 'off'}; older card versions: {result.get('versions_mode') or '-'}.")
    out += [""] + _summary_lines(result, entries)
    out += _method_lines(selftest)
    out += _anomaly_lines(result, cards)
    return "\n".join(out) + "\n"


# ========================
# Self-test
# ========================
VARIANTS = {"discord": {"width": 800, "quality": 80}, "small": {"width": 360, "quality": 60},
            "crop": {"width": 600, "quality": 85, "crop": 0.03}, "padded": {"width": 700, "quality": 85, "pad": 0.06},
            "cut": {"width": 700, "quality": 80, "crop": (0.05, 0.07, 0.0, 0.03)}}
OCR_VARIANTS = ("discord", "small", "crop", "cut")
CARD_VARIANTS = ("discord", "small", "crop", "padded")    # finding the card: today's PNGs
VERSION_VARIANTS = ("discord", "small", "cut")            # picking the version: every version of cards with 2+ looks
SELFTEST_FEATURES = 1


def degrade(img, width: int, quality: int, crop=0, pad: float = 0):
    """A card as it might be posted: resized, JPEG'd, cropped (each side, or left, top, right, bottom) or on
    Discord's gray."""
    Image = _pil()
    w, h = img.size
    if crop:
        left, top, right, bottom = crop if isinstance(crop, tuple) else (crop,) * 4
        img = img.crop((round(w * left), round(h * top), round(w * (1 - right)), round(h * (1 - bottom))))
    img = img.resize((width, round(width * img.height / img.width)), Image.LANCZOS)
    if pad:
        p = round(width * pad)
        canvas = Image.new("RGB", (img.width + 2 * p, img.height + 2 * p), DISCORD_GRAY)
        canvas.paste(img, (p, p))
        img = canvas
    buf = BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def as_post(feat: dict, candidates: list) -> dict:
    """A degraded card's features as judge() sees a post."""
    return {"shape": feat.get("shape"), "ocr": feat.get("ocr"), "lines": [line[0] for line in feat.get("lines") or []],
            "candidates": candidates}


def selftest_task(args: tuple) -> dict:
    key, source, ocr, names = args
    try:
        img = load_image(open_source(source))
        return {"key": key, "v": SELFTEST_FEATURES, "ocr": ocr, "variants": {
            name: features(degrade(img, **VARIANTS[name]), ocr and name in OCR_VARIANTS) for name in names}}
    except Exception as e:
        return {"key": key, "v": SELFTEST_FEATURES, "error": f"{type(e).__name__}: {str(e)[:160]}",
                **({"retry": True} if getattr(e, "retry", False) else {})}


def version_outcome(index: CardIndex, looks: dict, cid: str, key: str, feat: dict, skip: Optional[int] = None) \
        -> tuple:
    """(outcome, pick) of picking the version of one degraded card version: right, wrong, tied (looks tied),
    draft (taken for a look not in git) or none; skip: its own look left out (a draft's stand-in)."""
    pick = index.pick_version(cid, feat, read_text(feat.get("lines")), skip)
    tied = {looks[k] for k in pick["keys"]}
    if pick["by"] == "draft":
        return "draft", pick
    if not tied:
        return "none", pick
    if len(tied) > 1:
        return "tied", pick
    return ("right" if looks[key] in tied else "wrong"), pick


def selftest(root: str, cards_repo: str = LOCAL_DIR_LOC, workers: int = 0, limit: Optional[int] = None,
             log: Callable = print) -> dict:
    """The image methods on degraded copies of every current card PNG, and the version pick on degraded copies of
    every version of the cards with more than one look (needs a prior score run's card cache)."""
    root = check_root(root)
    paths = Paths(root)
    workers = workers or max(1, (cpu_count() or 2) - 2)
    cards = load_cards(paths.site)
    cache = Cache(paths.card_cache)
    plan = plan_versions(cards, paths.site, cards_repo, "all")
    versions = []
    for v in plan:
        entry = cache.entries.get(v["key"]) or {}
        if entry.get("whole"):
            versions.append({**v, **{k: entry.get(k) for k in ("whole", "art", "title", "shape", "lines",
                                                                "rereads")}})
    if not versions:
        raise SystemExit("no card features cached: run score first")
    index = CardIndex(cards, versions)
    looks = index.looks()
    current = [v for v in versions if v["current"]][:limit] if limit else [v for v in versions if v["current"]]
    many = sorted(cid for cid, vis in index.by_card.items() if len({looks[index.versions[i]["key"]] for i in vis}) > 1)
    many = set(many[:limit] if limit else many)
    wanted: dict = {}
    for v in current:
        wanted[v["key"]] = (v, CARD_VARIANTS)
    for v in versions:
        if v["card"] in many:
            names = wanted.get(v["key"], (v, ()))[1]
            wanted[v["key"]] = (v, tuple(dict.fromkeys(names + VERSION_VARIANTS)))
    ocr = bool(_vision())
    tests = Cache(join(paths.cache, "selftest.jsonl"))

    def ready(key: str, names: tuple) -> bool:
        entry = tests.entries.get(key) or {}
        return entry.get("v") == SELFTEST_FEATURES and not entry.get("retry") and (
            bool(entry.get("error")) or (set(names) <= set(entry.get("variants") or {}) and entry.get("ocr") == ocr))
    run_tasks("selftest_task", [(key, v["source"], ocr, names) for key, (v, names) in wanted.items()
                                if not ready(key, names)], workers, tests.add, log, "self-test")
    results = {key: tests.entries.get(key) or {} for key in wanted}
    stats = {name: defaultdict(list) for name in CARD_VARIANTS}
    confusions, unsure = Counter(), defaultdict(list)
    for v in current:
        variants = results[v["key"]].get("variants") or {}
        for name in CARD_VARIANTS:
            feat = variants.get(name)
            if not feat:
                continue
            st, truth = stats[name], v["card"]
            nearest = index.scan(feat)
            own = nearest.get(truth, (99, 0, ""))[0]
            others = [d for c, (d, _, _) in nearest.items() if not related(cards, c, truth)]
            st["true_bits"].append(own)
            st["other_bits"].append(min(others) if others else 99)
            ranked = sorted(nearest.items(), key=lambda kv: kv[1][0])
            st["hash_top1"].append(bool(ranked) and related(cards, ranked[0][0], truth))
            if feat.get("title") is not None:
                titles = title_texts(feat)
                scores = {c: index.names.best_ratio(c, titles)[0] for c in index.names.fuzzy(
                    titles[0], TITLE_FLOOR)} if titles else {}
                top = max(scores, key=scores.get) if scores else None
                st["title_top1"].append(top is not None and related(cards, top, truth))
            verdict = judge(as_post(feat, score_image(index, feat)), cards)
            if verdict["status"] == "confident":
                right = any(related(cards, c, truth) for c in verdict["cards"])
                st["confident_right" if right else "confident_wrong"].append(1)
                if not right:
                    confusions[(truth, verdict["cards"][0])] += 1
            else:
                st[verdict["status"] if verdict["status"] in ("ambiguous", "none") else "none"].append(1)
                if len(unsure[name]) < 25:
                    unsure[name].append({"card": truth, "status": verdict["status"], "reason": verdict["reason"],
                                         "maybe": verdict["maybe"][:3]})
    report = {"generated": now_iso(), "cards": len(current), "versions": len(versions), "variants": {}, "notes": []}
    for name, st in stats.items():
        n = len(st["true_bits"])
        if not n:
            continue
        report["variants"][name] = {
            "n": n, "hash_top1": sum(st["hash_top1"]) / n,
            "title_top1": (sum(st["title_top1"]) / len(st["title_top1"])) if st["title_top1"] else None,
            "confident_right": len(st["confident_right"]) / n, "confident_wrong": len(st["confident_wrong"]) / n,
            "ambiguous": len(st["ambiguous"]) / n, "none": len(st["none"]) / n,
            "true_bits_p50": _pct(st["true_bits"], 0.5), "true_bits_p95": _pct(st["true_bits"], 0.95),
            "true_bits_p99": _pct(st["true_bits"], 0.99), "other_bits_p1": _pct(st["other_bits"], 0.01),
            "other_bits_p5": _pct(st["other_bits"], 0.05), "other_bits_p50": _pct(st["other_bits"], 0.5)}
    report["confusions"] = [{"card": a, "taken_for": b, "times": n} for (a, b), n in confusions.most_common(30)]
    report["unsure"] = unsure

    # the version pick: every version of the cards with more than one look
    picks = {name: Counter() for name in VERSION_VARIANTS}
    held, old_found, wrong = Counter(), Counter(), []
    for key, (v, names) in sorted(wanted.items()):
        if v["card"] not in many:
            continue
        variants = results[key].get("variants") or {}
        for name in VERSION_VARIANTS:
            feat = variants.get(name)
            if not feat:
                picks[name]["unread"] += 1
                continue
            outcome, pick = version_outcome(index, looks, v["card"], key, feat)
            picks[name][outcome] += 1
            if outcome in ("wrong", "draft") and len(wrong) < 40:
                wrong.append({"card": v["card"], "variant": name, "commit": v["commit"], "outcome": outcome,
                              "look": looks[key], "pick": pick})
            if name != "discord":
                continue
            groups, look = index.groups[v["card"]], looks[key]       # its look left out: stands for a draft
            own = groups[look][0] - set().union(*(g[1] for other, g in groups.items() if other != look))
            held["draft" if version_outcome(index, looks, v["card"], key, feat, look)[0] == "draft" else "taken",
                 "far" if len(own) >= 12 else "near"] += 1
            if not v["current"]:
                verdict = judge(as_post(feat, score_image(index, feat)), cards)
                ok = verdict["status"] == "confident" and any(related(cards, c, v["card"]) for c in verdict["cards"])
                old_found["confident_right" if ok else "confident_wrong" if verdict["status"] == "confident"
                          else verdict["status"]] += 1
    report["version_check"] = {"cards": len(many), "versions": sum(1 for v, _ in wanted.values() if v["card"] in many),
                               "looks": sum(len(index.groups[c]) for c in many),
                               "variants": {name: dict(c) for name, c in picks.items()},
                               "held_out": {f"{a}_{b}": n for (a, b), n in sorted(held.items())},
                               "old_found": dict(old_found), "misses": wrong}
    olds, renamed, plain = [], [], Names(cards)
    by_card = defaultdict(list)
    for v in versions:
        by_card[v["card"]].append(v)
    for cid, vs in by_card.items():
        now = next((v for v in vs if v["current"]), None)
        old = sorted((v for v in vs if not v["current"]), key=lambda v: v["ts"] or 0)
        if now and old:
            olds.append(min(hash_distance(old[0]["whole"], now["whole"]),
                            hash_distance(old[0]["art"], now["art"]) if old[0].get("art") and now.get("art") else 99))
        for v in old:
            text = " ".join(t for t, c in v.get("title") or [] if c >= 0.3)
            if len(alnum(text)) >= 4 and plain.best_ratio(cid, title_texts({"title": v.get("title")}))[0] < 0.7:
                renamed.append({"card": cid, "name": cards[cid].get("name"), "read": text, "commit": v["commit"],
                                "ts": iso(v["ts"]) if v.get("ts") else None})
                break
    report["old_versions"] = {"n": len(olds), "bits_p50": _pct(olds, 0.5),
                              "within_zero": (sum(1 for d in olds if d < HASH_ZERO) / len(olds)) if olds else None,
                              "renamed": len(renamed), "renamed_examples": renamed[:40]}
    write_json(paths.selftest, report)
    log(dumps({"variants": report["variants"], "version_check": {k: v for k, v in report["version_check"].items()
                                                                   if k != "misses"}}, indent=1))
    return report


# ========================
# Command line
# ========================
def main(argv: Optional[list] = None) -> int:
    parser = ArgumentParser(prog="python -m commands.discord_match", description="Match mined Discord posts to cards.")
    sub_parsers = parser.add_subparsers(dest="command", required=True)
    for name in ("score", "build", "selftest"):
        p = sub_parsers.add_parser(name)
        p.add_argument("--root", default=environ.get("DISCORD_ARCHIVE", DEFAULT_ROOT))
        p.add_argument("--cards-repo", default=LOCAL_DIR_LOC)
        if name in ("score", "selftest"):
            p.add_argument("--workers", type=int, default=0)
            p.add_argument("--limit", type=int)
        if name == "score":
            p.add_argument("--site", default=SITE_URL)
            p.add_argument("--refresh-site", action="store_true")
            p.add_argument("--versions", choices=("none", "oldest", "all"), default="all")
            p.add_argument("--no-ocr", action="store_true")
    args = parser.parse_args(argv)

    def log(text):
        print(text, file=stderr, flush=True)
    if args.command == "score":
        score(args.root, args.cards_repo, args.site, args.versions, not args.no_ocr, args.workers, args.refresh_site,
              args.limit, log)
    elif args.command == "build":
        build(args.root, args.cards_repo, log)
    else:
        selftest(args.root, args.cards_repo, args.workers, args.limit, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
