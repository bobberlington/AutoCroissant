#!/usr/bin/env python3
"""
The old repo's version rows for the card timelines (STAGE1 S3; docs/Future/CARD_TIMELINE.md; owner 11-13). By hand.
Each PNG version of a card's old file (oldgit map.json) is a row: date, mapped author, commit subject, badge "old repo",
no link, its thumb, and its fields from the PSD read of the same commit (else the PSD nearest it in the commit range),
each checked against the picture: stats read off the PNG (the printed value wins; a disagreement goes to review), name
and ability against the PNG's OCR both ways (the matcher's card cache), stars the PSD draws against the 'N Stars'
folder, types by their header icon layers' pictures (Types/; today's names), and the PSD's composite against the PNG.
Two-creature cards are flagged, never summed.
Minor looks (under png_changes.VISIBLE of the pixels, every field sure on both rows and the same) fold under "minor
changes"; formatting-only text changes merge (owner 11-09). A line ending 'can't affect' (an icon the PSD text lacks)
goes to review. A renamed card's rows show the old name (old_name, old_file); never an alias.

    python oldgit_rows.py reread [-j N]     PSDs whose read has no ability, read again with the text layers named
                                            after their own text as the Ability (rereads.jsonl)
    python oldgit_rows.py icons [-j N]      each PSD's header icon layers matched to Types/ pictures (icons.jsonl)
    python oldgit_rows.py stars [-j N]      the stars each PSD draws (stars.jsonl)
    python oldgit_rows.py pictures [-j N]   each PNG version: stats off the picture (two bar readers, each row's two
                                            halves) and the change share from the version before it (pictures.jsonl)
    python oldgit_rows.py rows              rows.jsonl (live cards, sure maps), review.jsonl (what is not sure, the
                                            unsure maps, lost and Morgue cards), archived.jsonl, summary.json
    python oldgit_rows.py sample [N] [--seed S]
                                            N random rows (seed 1113): contact sheets for an eye check (sample/)

Reads derived/oldgit and derived/matcher; writes derived/oldrows (never a repo). Bar readers: the eras part's
readers.py (OLDROWS_READERS: its folder; default the stage-1 scratch copy). Stats keys: hp, def, atk, spd.
"""
import hashlib
import importlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from os.path import basename, dirname, exists, join, splitext

sys.path.insert(0, dirname(os.path.abspath(__file__)))
import oldgit_index as oi                                      # noqa: E402  (AC on sys.path, the old .git, paths)

DERIVED = dirname(oi.OUT)
OUT = join(DERIVED, "oldrows")
CARD_CACHE = join(DERIVED, "matcher", "match", "cache", "cards.jsonl")
BAR_TRUTH = join(DERIVED, "readers", "bars", "truth")   # the bar readers' by-eye stats (eye_*.json, by PNG blob)
BAR_READS = join(DERIVED, "readers", "bars", "reads.jsonl")   # the era bar readers' calibrated reads (era_bars)
VERDICTS = join(DERIVED, "match", "verdicts.jsonl")     # Discord pictures told as a git look (version_sure)
TEXT_RAW = (join(DERIVED, "readers", "text", "raw.jsonl"),       # the text readers' raw reads (o:<blob>: old PNGs),
            join(DERIVED, "readers", "text", "raw_ability.jsonl"))   # and their ability-alone reads (read-ability)
TEXT_CAL = join(DERIVED, "readers", "text", "calibration.json")  # their eras' bars (never fitted here)
HEADER_ROWS = join(DERIVED, "readers", "types", "header_rows.jsonl")   # header_types rows: types by a git header
READERS = os.environ.get("OLDROWS_READERS") or join(oi.HOME, "Documents/Tools/wt/scratch/s1110/timelines/eras")
ORDER = ("hp", "def", "atk", "spd")
BADGE = "old repo"
FIRST_BADGE = "first repo"  # the first repo's versions (oldgit_index: 2022-08-24..2023-04-21, refound on GitHub)
STAT_KINDS = ("creature", "minion")
ITEM_KINDS = ("item", "aux item", "nme")      # some print HP/DEF bars (Blockade)
ABILITY_KINDS = ("creature", "minion", "item", "aux item", "field", "buff", "debuff", "nme")
STAT_WORDS = {"health", "defense", "defence", "attack", "speed", "hp", "def", "atk", "spd"}
NOT_ABILITY = {"name", "title", "attribute", "class", "type", "types", "stars"}    # layer names that are no ability
SAME_PICTURE = 0.005        # composite vs PNG: share of 8 px thumb blocks off by more than BLOCK_DIFF (0: identical)
BLOCK_DIFF = 20
NAME_RATIO = 0.8            # OCR'd title vs a name (SequenceMatcher on letters and digits)
RECALL = 0.75               # OCR word recall of an ability: it is printed ...
LETTERS = 0.8               # ... or this share of its letters and digits in order (OCR joins words: '1or 2')
CLASS_ZONE = 0.935          # a 1:2 card's class line below this (y of a line's middle, card heights)
GAP = re.compile(r"\S {5,}\S")  # an icon the parser left as spaces ('not          take 2 damage')
DANGLING = {"affect", "per", "against", "vs", "to", "a", "an", "the", "of", "on", "by", "with", "from", "any", "all",
            "each", "every", "is", "are"}       # a line ending so wants an icon after it ("can't affect [fire]")
TOKEN = re.compile(r"\[[^\]]*\]")
LEADING = ":;,.)/"                # a line starting so: a word or icon before it not read
TEXT_SEEN = 2               # OCR words under the bars/title: the card prints text ('-25% SPD')
HEADER, ICON_MAX = 0.32, 0.3      # type icon layers: bottom (a type's picture: middle) and largest side, card widths
ICONS_V = 2                 # icons.jsonl rows older than this are read again (2: low type icons)
ICON_SIDE = 32              # type icons compared at this size; one matches a type's picture within
ICON_SAME = 15              # mean RGBA difference (0-255) of one icon (a recolour or an older drawing: ~10) ...
ICON_GAP = 4                # ... nearer than the next type's by this much or half as far (or its layer named so)
ICON_NEAR = 6               # nearest types kept per icon
TWO_ROWS = 2                # rows whose top and bottom halves read apart: two creatures
SPLITS = {"Twin Princes": ("Lothric", "Lorian")}    # one old card, two live ones (STAGE1 Q3, provisional A)
SPLIT_EYE = {"Twin Princes": "the bar colours: Lorian's orange top rows, Lothric's blue bottom ones, as on today's "
                             "cards (by eye)"}
STAR_BAND = 1 / 6           # the card's top, where its stars are: the composite there = the PNG's
MISSING_WORDS = 4           # card words in OCR'd ability lines the PSD text lacks: the card prints more (Pix)
OLD_TYPES = {"undread": "undead", "tornado": "wind"}    # renamed types (same pictures)
OLD_TOKENS = re.compile(r"\[(" + "|".join(OLD_TYPES) + r")(?=[\]:])")   # their [tokens] in an ability
MINION_ICONS = {"equip": "offensive_minion", "protective": "defensive_minion"}   # 2023 minion role icons
SEPARATORS = re.compile(r"[^a-z0-9\-' ]")              # what a printed name's form keeps besides letters
SEED = 1113


def log(*a):
    oi.log(*a)


def stable_id(*parts) -> str:
    return hashlib.sha1("\x1f".join(str(p) for p in parts).encode()).hexdigest()[:12]


def readers():
    import commands.card_checks  # noqa: F401  (the readers import it: this worktree's, not theirs)
    if READERS not in sys.path:
        sys.path.insert(0, READERS)
    return importlib.import_module("readers")


def text_key(text) -> str:
    from commands.changelog import text_key as key
    return key(text)


def norm(text) -> str:
    return re.sub(r"[^0-9a-z]", "", oi.alnum(text or ""))


def card_words(text) -> set:
    """Words of 3+ letters (no numbers, [tokens] dropped): an ability's vocabulary."""
    return set(words3(text))


def folder_stars(path: str):
    found = re.search(r"(?:^|/)(\d+) Stars/", path)
    return int(found.group(1)) if found else None


def kind_of(path: str) -> str:
    """The card kind its old path classifies (when no PSD read says)."""
    p = path.lower()
    for part, kind in (("/minions/", "minion"), ("debuffs/", "debuff"), ("buffs/", "buff"), ("auxillary/items/", "aux item"),
                       ("auxiliary/items/", "aux item"), ("creatures/", "creature"), ("items/", "item"), ("field/", "field"),
                       ("n.m.e/", "nme"), ("cards/", "creature")):     # Cards/: the first repo's creatures
        if part in p or p.startswith(part):
            return kind
    return "unknown"


# ========================
# The PSD again: text layers named after their own text as the Ability
# ========================
def ability_layers(psd) -> list:
    """Visible text layers in the card's lower part that print no title, stat label, class line or placeholder: the
    ability an old card set in a layer named after its text (no 'Ability' layer)."""
    from commands.psd_parser import base_name
    w, h = psd.width, psd.height
    found = []
    for layer in psd.descendants():
        if layer.kind != "type" or not layer.is_visible() or base_name(layer).strip().lower() == "ability":
            continue
        try:
            text = layer.text or ""
        except Exception:
            continue
        letters = norm(text)
        x0, y0, x1, y1 = layer.bbox
        if len(letters) < 2 or letters.startswith("loremipsum") or letters in STAT_WORDS \
                or base_name(layer).strip().lower() in NOT_ABILITY:
            continue
        if x1 <= max(0, x0) or y1 <= max(0, y0) or x0 >= w or y0 >= h or y0 < 0.45 * h:
            continue
        if h >= 1.8 * w and y0 > 0.93 * h and len(text.strip()) <= 40:
            continue                                           # a creature's class line ('Nature/Brute')
        found.append((layer, letters))
    return [layer for layer, letters in found          # a word laid over another layer's text (Undyne's 'Condemned')
            if not any(letters != other and letters in other for _, other in found)]


def w_reread(job: tuple) -> dict:
    from psd_tools import PSDImage
    from commands.psd_parser import CardValidator, variants_of
    sha, path = job
    row = {"blob": sha, "path": path}
    try:
        psd = PSDImage.open(BytesIO(oi.w_blob(sha)))
        layers = ability_layers(psd)
        row["layers"] = [layer.name[:60] for layer in layers]
        if not layers:
            return row
        for layer in layers:
            layer.name = "Ability"
        rel = oi.today_rel(path)
        parts = rel.split("/")
        if parts[0] in ("Creatures", "Items", "Field") and not re.match(r"\d", parts[len(parts) - 2]):
            rel = "/".join(parts[:-1] + ["0 Stars", parts[len(parts) - 1]])     # no 'N Stars' folder (as the reads)
        card = oi._W["parser"].parse(psd, rel, **({"variant": 1} if len(variants_of(psd)) > 1 else {}))
        row["read"] = oi.card_fields(card, CardValidator.validate(card))
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return row


def reread_main(workers: int):
    reads = oi.read_jsonl(join(oi.OUT, "reads.jsonl"))
    todo = sorted({(r["blob"], r["path"]) for r in reads if r["kind"] == "psd" and r.get("read")
                   and r["read"]["type"] in ABILITY_KINDS and not (r["read"].get("ability") or "").strip()})
    os.makedirs(OUT, exist_ok=True)
    target = join(OUT, "rereads.jsonl")
    done = {(r["blob"], r["path"]) for r in oi.read_jsonl(target)}
    todo = [t for t in todo if t not in done]
    vocab = json.load(open(join(oi.OUT, "vocabulary.json")))
    from commands.card_check import clone_vocabulary
    manifest = clone_vocabulary(oi.TC).manifest
    log(f"reread: {len(todo)} PSD reads with no ability")
    n, found = 0, 0
    with open(target, "a", encoding="utf-8") as out, ProcessPoolExecutor(
            workers, initializer=oi.w_init, initargs=(vocab["types"], vocab["stars"], manifest)) as pool:
        for row in pool.map(w_reread, todo, chunksize=4):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            found += bool((row.get("read") or {}).get("ability"))
            if n % 50 == 0 or n == len(todo):
                out.flush()
                log(f"{n}/{len(todo)} reread, {found} with an ability")


# ========================
# The header's type icons: each icon layer matched to Types/ pictures
# ========================
def icon_vector(img):
    """An icon as 32 x 32 RGBA on black (its alpha box, centred square), float: what type icons are compared by."""
    import numpy as np
    from PIL import Image
    img = img.convert("RGBA")
    box = img.getchannel("A").point(lambda a: 255 if a > 32 else 0).getbbox()
    if not box:
        return None
    img = img.crop(box)
    side = max(img.size)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    a = np.asarray(square.resize((ICON_SIDE, ICON_SIDE), Image.LANCZOS), dtype=np.float32)
    return np.concatenate([a[..., :3] * a[..., 3:] / 255, a[..., 3:]], axis=2)


def type_icons() -> list:
    """[(type name, icon vector)]: every Types/ picture of the old repo's trees and today's (Stars left out), named as
    card_check.vocabulary names them."""
    from PIL import Image
    from commands.card_check import vocabulary
    found, seen, trees = [], set(), set()
    for commit in json.load(open(join(oi.OUT, "index.json")))["order"]:
        try:
            tree = oi.git(oi.OG, "rev-parse", f"{commit}:Types").decode().strip()
        except Exception:
            continue
        if tree in trees:
            continue
        trees.add(tree)
        for entry in oi.git(oi.OG, "ls-tree", "-z", tree).decode("utf-8", "replace").split("\0"):
            if entry and entry.split()[1] == "blob" and entry.lower().endswith(".png"):
                sha, name = entry.split()[2], entry.split("\t", 1)[1]
                if sha not in seen:
                    seen.add(sha)
                    with Image.open(BytesIO(oi.git(oi.OG, "cat-file", "blob", sha))) as im:
                        found.append((vocabulary([f"Types/{name}"]).all_types[0], icon_vector(im)))
    folder = join(oi.TC, "Types")
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(".png"):
            with Image.open(join(folder, name)) as im:
                found.append((vocabulary([f"Types/{name}"]).all_types[0], icon_vector(im)))
    return [(name, vec) for name, vec in found if vec is not None]


def header_icons(psd, right: bool = False) -> list:
    """Visible picture layers in the header's left half or a 'Types' group (middle within HEADER card widths of the
    top; one whose bottom is past it is low: header_icon_kept), icon-sized, no star in their name: a card's type icons
    (its stars sit right of the title; right: the right half too, a buff's arrow where other cards have stars, 2022-23)."""
    w = psd.width
    found = []
    for layer in psd.descendants():
        if layer.is_group() or layer.kind not in ("pixel", "smartobject", "shape") or not layer.is_visible():
            continue
        x0, y0, x1, y1 = layer.bbox
        typed = layer.parent is not None and "type" in getattr(layer.parent, "name", "").lower()   # a 'Types' group
        if y0 + y1 > 2 * HEADER * w or (x0 + x1) / 2 > 0.55 * w and not typed and not right or \
                not (0.04 * w <= x1 - x0 <= ICON_MAX * w and 0.04 * w <= y1 - y0 <= ICON_MAX * w):
            continue
        if "star" in layer.name.lower():
            continue
        found.append(layer)
    return sorted(found, key=lambda layer: (layer.bbox[0], layer.bbox[1]))


def header_icon_kept(icon: dict, names: set) -> bool:
    """A low header layer (its bottom past HEADER card widths, its middle above) counts only when its picture or name
    is a type's (Nailsage Sly's tornado, 2022-23: one row lower, under the title); else it is art, left out as before."""
    return not icon.get("low") or icon_type(icon, names) is not None


def w_icons(job) -> dict:
    import numpy as np
    from psd_tools import PSDImage
    sha, buff = job
    row = {"blob": sha, "icons": [], "v": ICONS_V, **({"buff": True} if buff else {})}
    try:
        psd = PSDImage.open(BytesIO(oi.w_blob(sha)))
        refs = _W["icons"]
        for layer in header_icons(psd, right=buff):
            try:
                vec = icon_vector(layer.topil())
            except Exception:
                vec = None
            found = {"layer": layer.name[:60], "bbox": list(layer.bbox)}
            if layer.bbox[3] > HEADER * psd.width:
                found["low"] = True
            if vec is not None:
                best = {}
                for name, ref in refs:
                    d = float(np.abs(vec - ref).mean())
                    best[name] = min(d, best.get(name, d))
                found["near"] = [[name, round(d, 1)] for name, d in sorted(best.items(), key=lambda kv: kv[1])[:ICON_NEAR]]
            if header_icon_kept(found, _W["names"]):
                row["icons"].append(found)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return row


def w_icons_init(refs, names):
    w_picture_init()
    _W["icons"], _W["names"] = refs, names


def icons_main(workers: int):
    reads = [r for r in oi.read_jsonl(join(oi.OUT, "reads.jsonl")) if r["kind"] == "psd"]
    buffs = {r["blob"] for r in reads if (r.get("read") or {}).get("type") == "buff"}    # the right half too
    blobs = sorted({r["blob"] for r in reads})
    target = join(OUT, "icons.jsonl")
    done = {r["blob"] for r in oi.read_jsonl(target)
            if r.get("v", 1) >= ICONS_V and (r["blob"] not in buffs or r.get("buff"))}   # a later row wins
    blobs = [(b, b in buffs) for b in blobs if b not in done]
    refs = type_icons()
    names = set(json.load(open(join(oi.OUT, "vocabulary.json")))["types"])
    log(f"icons: {len(blobs)} PSDs, {len(refs)} type pictures ({len({n for n, _ in refs})} types)")
    os.makedirs(OUT, exist_ok=True)
    n = 0
    with open(target, "a", encoding="utf-8") as out, ProcessPoolExecutor(
            workers, initializer=w_icons_init, initargs=(refs, names)) as pool:
        for row in pool.map(w_icons, blobs, chunksize=4):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if n % 100 == 0 or n == len(blobs):
                out.flush()
                log(f"{n}/{len(blobs)} PSDs")


# ========================
# The stars each PSD draws (the parser takes most kinds' stars from the 'N Stars' folder)
# ========================
def w_stars(sha: str) -> dict:
    from psd_tools import PSDImage
    row = {"blob": sha}
    try:
        psd = PSDImage.open(BytesIO(oi.w_blob(sha)))
        row["drawn"] = oi._W["parser"]._count_stars(psd)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return row


def stars_main(workers: int):
    blobs = sorted({r["blob"] for r in oi.read_jsonl(join(oi.OUT, "reads.jsonl")) if r["kind"] == "psd"})
    target = join(OUT, "stars.jsonl")
    done = {r["blob"] for r in oi.read_jsonl(target) if "error" not in r}
    blobs = [b for b in blobs if b not in done]
    vocab = json.load(open(join(oi.OUT, "vocabulary.json")))
    from commands.card_check import clone_vocabulary
    manifest = clone_vocabulary(oi.TC).manifest
    log(f"stars: {len(blobs)} PSDs")
    os.makedirs(OUT, exist_ok=True)
    n = 0
    with open(target, "a", encoding="utf-8") as out, ProcessPoolExecutor(
            workers, initializer=oi.w_init, initargs=(vocab["types"], vocab["stars"], manifest)) as pool:
        for row in pool.map(w_stars, blobs, chunksize=4):
            out.write(json.dumps(row) + "\n")
            n += 1
            if n % 200 == 0 or n == len(blobs):
                out.flush()
                log(f"{n}/{len(blobs)} PSDs")


# ========================
# The pictures: stats off the PNG, the change from the version before
# ========================
def halves(R, arr, found: dict, info) -> dict:
    """Each stat row's top and bottom half read apart (read_best's grid): a two-creature card's rows split in two."""
    if not info:
        return {}
    from commands.card_checks import TEMPLATE_CELLS
    h, w = arr.shape[:2]
    x0, x1 = int(info["span"][0] * w), int(info["span"][1] * w)
    cells = TEMPLATE_CELLS if info["cells"] == "template" else R.UNIFORM
    if info.get("fit") == "fixed rows":
        sy = h / 2400
        bands = [(int(y * sy), int((y + R.D_H) * sy)) for y in R.D_ROWS]
    else:
        ys, pitch = R.fill_rows(found)
        half = abs(pitch) * h * 0.34
        bands = [(int(max(0, ys[s] * h - half)), int(min(h, ys[s] * h + half))) for s in ORDER]
    tops, bottoms = {}, {}
    for s, (y0, y1) in zip(ORDER, bands):
        row = arr[y0:y1, x0:x1]
        if row.size == 0 or row.shape[0] < 6:
            tops[s] = bottoms[s] = None
            continue
        mid = row.shape[0] // 2
        tops[s], bottoms[s] = R.split_read(row[:mid], cells), R.split_read(row[mid:], cells)
    apart = sum(1 for s in ORDER if tops[s] and bottoms[s] and not tops[s]["flat"] and not bottoms[s]["flat"]
                and tops[s]["best"] != bottoms[s]["best"])          # a one-colour half is told by the others: no evidence
    R.tell_flats(tops)
    R.tell_flats(bottoms)
    top = {s: (r["best"] if r else None) for s, r in tops.items()}
    bottom = {s: (r["best"] if r else None) for s, r in bottoms.items()}
    return {"top": top, "bottom": bottom, "apart": apart}


def w_picture(job: tuple) -> dict:
    import numpy as np
    import commands.discord_match as dm
    from commands.png_changes import changed_share
    blob, prevs, lines, stats = job
    out = {"blob": blob}
    try:
        data = oi.w_blob(blob)
        if stats:
            R = _W.get("readers") or _W.setdefault("readers", readers())
            full = dm.load_image(BytesIO(data))
            arr = np.asarray(dm.trim(full))
            found = R.labels(lines)
            best, info = R.read_best(arr, found)
            out["best"], out["labels"] = best, sorted(found)
            out["info"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in (info or {}).items()} or None
            whole = np.asarray(full)
            if 1.8 <= whole.shape[0] / whole.shape[1] <= 2.2:
                out["template"] = R.template_bars(whole, "split")
            out["halves"] = halves(R, arr, found, info)
        if prevs:
            out["shares"] = [[prev, changed_share(oi.w_blob(prev), data)] for prev in prevs]
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return out


_W: dict = {}


def w_picture_init():
    import subprocess
    oi._W["cat"] = subprocess.Popen(oi.STORE + ["cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)


# ========================
# Inputs
# ========================
class Data:
    """Everything the rows read, loaded once."""
    def __init__(self):
        self.index = json.load(open(join(oi.OUT, "index.json")))
        self.map = json.load(open(join(oi.OUT, "map.json")))
        self.lost = json.load(open(join(oi.OUT, "lost.json")))
        self.authors = {k: v for k, v in json.load(open(oi.AUTHORS)).items() if not k.startswith("_")}
        self.versions, self.by_commit = defaultdict(list), defaultdict(list)
        for v in oi.read_jsonl(join(oi.OUT, "versions.jsonl")):
            self.versions[v["old_id"]].append(v)
            if v["ext"] == "psd":
                self.by_commit[v["commit"]].append(v)
        self.psd, self.png = {}, {}
        for r in oi.read_jsonl(join(oi.OUT, "reads.jsonl")):
            if r["kind"] == "psd":
                self.psd[(r["blob"], r["path"])] = r
            else:
                self.png[r["blob"]] = r
        self.rereads = {}
        for r in oi.read_jsonl(join(OUT, "rereads.jsonl")):
            self.rereads[(r["blob"], r["path"])] = r
            if (r.get("read") or {}).get("ability"):
                self.rereads.setdefault(r["blob"], r)          # the same PSD under another path (Unused/)
        self.icons = {r["blob"]: r["icons"] for r in oi.read_jsonl(join(OUT, "icons.jsonl")) if "error" not in r}
        self.drawn = {r["blob"]: r["drawn"] for r in oi.read_jsonl(join(OUT, "stars.jsonl")) if "drawn" in r}
        self.eye_bars, self.eye_bars2 = {}, {}                 # PNG blob -> stats by eye (None: not readable)
        for name in sorted(os.listdir(BAR_TRUTH)) if exists(BAR_TRUTH) else ():
            if name.startswith("eye_") and name.endswith(".json"):
                for blob, t in json.load(open(join(BAR_TRUTH, name)))["items"].items():
                    self.eye_bars[blob] = t.get("stats")
                    if t.get("stats2"):                        # two creatures: the bottom half's rows
                        self.eye_bars2[blob] = t["stats2"]
        self.era_reads, self.look_reads = {}, defaultdict(list)    # PNG blob -> its era read; look key -> Discord's
        discord = {}
        for r in oi.read_jsonl(BAR_READS) if exists(BAR_READS) else ():
            if r.get("src") == "oldgit" and r.get("variant") == "png" and r.get("stats"):
                self.era_reads[r["blob"]] = r
            elif r.get("src") == "discord" and r.get("file"):
                truth = r.get("truth") or {}
                stats = r.get("stats") or {}
                if truth.get("from") == "eye" and truth.get("stats"):
                    discord[r["file"]] = ("eye", truth["stats"])
                elif stats and all((s or {}).get("sure") for s in stats.values()):
                    discord[r["file"]] = ("read", {k: s.get("v") for k, s in stats.items()})
        for v in oi.read_jsonl(VERDICTS) if exists(VERDICTS) else ():
            file = (v.get("file") or "") + (f"#c{v['cell']}" if v.get("cell") else "")
            if v.get("verdict") == "version_sure" and not v.get("joke") and file in discord:
                for key in (v.get("version") or {}).get("keys") or ():
                    self.look_reads[key].append(discord[file])
        self.text_raw = {r["id"][2:]: r for path in TEXT_RAW for r in oi.read_jsonl(path)
                         if str(r.get("id", "")).startswith("o:") and not r.get("error")}
        self.text_cal = json.load(open(TEXT_CAL)).get("eras") or {} if exists(TEXT_CAL) else {}
        self.header = {r["blob"]: r for r in oi.read_jsonl(HEADER_ROWS)} if exists(HEADER_ROWS) else {}
        self.type_names = set(json.load(open(join(oi.OUT, "vocabulary.json")))["types"])
        self.vocab = set()                                     # card words: every ability's (old PSDs, today's)
        for r in list(self.psd.values()) + list(self.rereads.values()):
            self.vocab.update(card_words(((r.get("read") or {}).get("ability") or "")))
        self.ocr = {}
        if exists(CARD_CACHE):
            for line in open(CARD_CACHE, encoding="utf-8"):
                e = json.loads(line)
                if e.get("lines") is not None or e.get("title") is not None:
                    self.ocr[e["key"]] = {"title": e.get("title") or [], "lines": e.get("lines") or []}
        self.pictures = {r["blob"]: r for r in oi.read_jsonl(join(OUT, "pictures.jsonl"))}
        self.shares = {(prev, r["blob"]): share for r in self.pictures.values() for prev, share in r.get("shares") or []}
        self.today = {}
        self.thumbs = {}
        self.psd_firsts = {}                                   # card -> its first record when a PSD (psd_first)

    def load_today(self):
        stats, _, _, _ = oi.today_cards()
        edits = {}
        try:
            edits = json.loads(oi.git(oi.TG, "show", "origin/main:site/Card Edits.json", env=oi.TENV)).get("cards", {})
        except Exception:
            pass
        for key, card in stats.items():
            names = {key, card.get("display_name") or key, (edits.get(key) or {}).get("name") or key}
            s = card.get("stats") or {}
            self.today[key] = {"names": names, "creatures": len(card.get("creatures") or ()), "type": card.get("card_type"),
                               "stars": card.get("stars"), "types": list(card.get("types") or ()),
                               "ability": card.get("ability"), "stats": valid_stats(
                                   {k: s.get(w) for k, w in zip(ORDER, ("hp", "defense", "attack", "speed"))})}
            self.vocab.update(card_words(card.get("ability") or ""))

    def text_bar(self, era, name: str) -> tuple:
        """(how, bar) of the text readers' era for an old PNG (its old-repo era, else the Discord era's: never
        lower than calibrate would borrow); none: (None, None)."""
        for src in ("old", "discord"):
            entry = (self.text_cal.get(f"{era}/{src}") or {}).get(name)
            if entry and "how" in entry:
                return entry["how"], entry.get("bar")
        return None, None

    def ocr_of(self, blob: str) -> dict:
        return self.ocr.get("b" + blob[:20]) or {}

    def thumb(self, blob: str):
        """A thumb as an RGB int16 array on black (cached), or None."""
        if blob not in self.thumbs:
            path = join(oi.OUT, "thumbs", f"{blob}.webp")
            arr = None
            if exists(path):
                import numpy as np
                from PIL import Image
                with Image.open(path) as im:
                    flat = Image.new("RGBA", im.size, (0, 0, 0, 255))
                    flat.alpha_composite(im.convert("RGBA"))
                    arr = np.asarray(flat.convert("RGB"), dtype=np.int16)
            self.thumbs[blob] = arr
        return self.thumbs[blob]


def picture_off(data: Data, psd_blob: str, png_blob: str, top: float = 1.0):
    """Share of 8 px blocks of the PSD's composite thumb off the PNG's (top: only that share of the card's height, from
    its top; None: no composite or sizes differ)."""
    import numpy as np
    a, b = data.thumb(psd_blob), data.thumb(png_blob)
    if a is None or b is None or a.shape != b.shape:
        return None
    d = np.abs(a - b).max(axis=2)
    d = d[:max(8, int(d.shape[0] * top))]
    h, w = d.shape[0] // 8 * 8, d.shape[1] // 8 * 8
    blocks = d[:h, :w].reshape(h // 8, 8, w // 8, 8).mean(axis=(1, 3))
    return round(float((blocks > BLOCK_DIFF).mean()), 4)


# ========================
# Rows
# ========================
def pick_psd(data: Data, png: dict, psds: list, until: float) -> tuple:
    """(PSD version, how, composite off) for a PNG version: the same commit's, else the nearest composite among the
    PSD as it was then and those committed before the next PNG version."""
    same = [s for s in psds if s["commit"] == png["commit"]]
    if same:
        return same[0], "same commit", picture_off(data, same[0]["blob"], png["blob"])
    if not psds:                                               # a second picture of another card's PSD (A_Snack2.png)
        stem = oi.stem_key(png["path"])
        for s in data.by_commit.get(png["commit"], []):
            variants = (data.psd.get((s["blob"], s["path"])) or {}).get("variants") or []
            if any(oi.alnum(v.get("name") or "") == stem for v in variants):
                return s, "same commit, its second picture", picture_off(data, s["blob"], png["blob"])
    before = [s for s in psds if s["ts"] <= png["ts"]][-1:]
    later = [s for s in psds if png["ts"] < s["ts"] < until]
    best = None
    for s in before + later:
        off = picture_off(data, s["blob"], png["blob"])
        rank = (off is None, off if off is not None else 1.0, s not in before)
        if best is None or rank < best[0]:
            best = (rank, s, "earlier" if s in before else "later in range", off)
    return (best[1], best[2], best[3]) if best else (None, "none", None)


def read_of(data: Data, psd: dict, ocr_title: str, png_path: str) -> tuple:
    """(fields read, how): the PSD read (its second picture's when the OCR'd title names that one); the reread's
    ability when the read has none."""
    row = data.psd.get((psd["blob"], psd["path"])) or {}
    read, how = row.get("read"), "psd"
    if not read:
        return None, "unread"
    for variant in row.get("variants") or []:
        named = oi.alnum(variant.get("name") or "") == oi.stem_key(png_path)
        if named or title_match(ocr_title, variant.get("printed_title") or variant.get("title")) and \
                not title_match(ocr_title, read.get("printed_title") or read.get("title")):
            read, how = variant, f"psd picture {variant['n']}"
            break
    again = data.rereads.get((psd["blob"], psd["path"])) or data.rereads.get(psd["blob"]) or {}
    if not (read.get("ability") or "").strip() and (again.get("read") or {}).get("ability"):
        read = {**read, "ability": again["read"]["ability"], "ability_layers": len(again.get("layers") or ())}
        how += "; ability from its text layer" + ("s" if read["ability_layers"] > 1 else "")
    return read, how


def title_match(ocr: str, name) -> bool:
    from difflib import SequenceMatcher
    a, b = norm(ocr), norm(name)
    if not a or not b:
        return False
    return a == b or (len(b) >= 4 and b in a) or SequenceMatcher(None, a, b, autojunk=False).ratio() >= NAME_RATIO


def name_form(text) -> str:
    """A name's letters and separators (hyphens, apostrophes, spaces), lower case, trimmed: 'WX-78' -> 'wx-78'."""
    text = re.sub(r"\s+", " ", SEPARATORS.sub("", (text or "").lower().replace("’", "'")))
    return re.sub(r"\s*-\s*", "-", text).strip(" -'")


def shape(text) -> str:
    return re.sub(r"[a-z0-9]+", "a", name_form(text))


def with_separators(name: str, ocr: str) -> str:
    """name (same letters as ocr) with the OCR'd separators: 'All Purpose Medical Kit' -> 'All-Purpose Medical Kit'."""
    letters = iter(c for c in name if c.isalnum())
    return "".join(next(letters) if c.isalnum() else c for c in name_form(ocr))


def check_name(ocr_title: str, printed, stem: str, today, same_look: bool) -> dict:
    """The printed name: of the PSD's title, the file name (a trailing picture number dropped too) and today's names,
    one the OCR'd title spells letter for letter (one with its separators first; else it takes the OCR'd ones), else
    the nearest of those near it."""
    names = [(printed, "psd"), (stem, "file name")]
    bare = re.sub(r"\s*\d+$", "", stem)
    if bare and bare != stem and not re.search(r"\d$", norm(ocr_title)):
        names.append((bare, "file name"))                      # a second picture's file ('..._Kit2')
    names = [(n, how) for n, how in names + [(n, "today") for n in sorted(today)] if n]
    want = norm(ocr_title)
    exact = [(n, how) for n, how in names if want and norm(n) == want]
    if exact:
        n, how = next(((n, how) for n, how in exact if shape(n) == shape(ocr_title)), exact[0])
        if how != "psd" and shape(n) != shape(ocr_title):
            n = with_separators(n, ocr_title)
        return field(n, how, True, "ocr")
    near = [(n, how) for n, how in names if title_match(ocr_title, n)]
    if near:                                                   # the nearest ('Himphrey's belly': today's, not the file's)
        from difflib import SequenceMatcher
        hit = max(near, key=lambda nh: SequenceMatcher(None, want, norm(nh[0]), autojunk=False).ratio())
        return field(hit[0], hit[1], True, "ocr")
    if printed and same_look and text_key(printed) == text_key(stem):
        return field(printed, "psd", True, "composite, file name")
    return field(printed or stem, "psd" if printed else "file name", False,
                 why="the OCR'd title names neither the PSD's title nor the file")


def check_stars(folder, drawn, star_look: bool, kind: str):
    """The printed stars: those the PSD draws (its composite = the PNG where the stars are), checked with the 'N Stars'
    folder (the parser's count for most kinds: no check alone); with no drawn stars the folder (card_rows may confirm
    it); None for a buff with neither."""
    if drawn and (folder is None or drawn == folder):
        out = field(drawn, "psd", star_look, "drawn" + (", folder" if folder is not None else "") +
                    (", composite" if star_look else ""), "" if star_look else "the PSD's look is not the picture's")
    elif drawn:
        out = field(drawn if star_look else folder, "psd" if star_look else "folder", False,
                    why=f"the PSD draws {drawn}, the folder says {folder}")
    elif folder is not None:
        out = field(folder, "folder", False, why="the PSD draws no stars (custom icons or no Stars group)")
    elif kind == "buff":
        return None                                            # buffs have no stars
    else:
        return field(None, "none", False, why="no stars read")
    out["reads"] = {"folder": folder, "drawn": drawn}
    return out


def confirm_stars(rows: list, today: dict):
    """Stars across a card's rows: one picture filed under two star folders -> review; a folder no PSD draws stars
    for is sure when it never changes and today's stars are the same. A PSD of another look drawing the folder's
    count stays in review: the picture may be older than both (Nosk 2023-03-19 prints 3, PSD and folder say 4)."""
    folders = defaultdict(set)
    for row in rows:
        folders[row["blob"]].add(folder_stars(row["path"]))
    every = {folder_stars(row["path"]) for row in rows} - {None}
    for row in rows:
        f = row["fields"].get("stars")
        if not f:
            continue
        moved = sorted(folders[row["blob"]] - {None})
        if len(moved) > 1:
            f.update(confidence="review", why=f"the same picture is filed under {' and '.join(map(str, moved))} Stars")
            f.pop("check", None)
        elif f["method"] == "folder" and not (f.get("reads") or {}).get("drawn") and len(every) == 1 \
                and today.get("stars") in every and f["value"] in every:
            f.update(confidence="sure", check="today, never changed")
            f.pop("why", None)


def valid_stats(stats, high: bool = False):
    top = 40 if high else 10
    if not stats or any(not isinstance(stats.get(s), int) or not 0 <= stats[s] <= top for s in ORDER):
        return None
    return {s: stats[s] for s in ORDER}


def complete(read):
    return {s: read[s] for s in ORDER} if read and all(isinstance(read.get(s), int) for s in ORDER) else None


def field(value, method: str, sure: bool, check: str = "", why: str = "") -> dict:
    out = {"value": value, "method": method, "confidence": "sure" if sure else "review"}
    if check:
        out["check"] = check
    if why:
        out["why"] = why
    return out


def check_stats(read, pic, two, square: bool, keys: tuple = ORDER, eye=False, eye2=None, first=False, era=None,
                looks=(), creature=False) -> dict:
    """The printed stats (keys: an item's HP/DEF bars): the bar readers' eye truth for the PNG first (eye: its
    stats, None when not readable by eye, False: none), else the picture's read (two readers), the PSD's digit rows
    where they agree (no PSD stats: both readers and a sure era read on every stat); two: a multi-creature card (True
    or its creature count) has none, its creatures kept apart (two creatures read by eye, eye2 the bottom half's
    rows: both halves sure, never summed). A first-repo row with
    no PSD stats (first) is sure only by the same look's Discord eye read, its era read (era: era_bars' row, sure by
    its era's calibration) or a reader agreeing with the same look's sure Discord read (looks: [(how, stats)]); the
    two readers alone share a bias on the 2020-22 templates. A creature's picture read with HP 0 is never sure."""
    out = _check_stats(read, pic, two, square, keys, eye, eye2, first, era, looks)
    v = out.get("value")
    if creature and out["confidence"] == "sure" and out["method"] != "eye" and isinstance(v, dict) and v.get("hp") == 0:
        out.update(confidence="review", why="HP 0 on a creature: a reader's miss")
        out.pop("check", None)
    return out


def _check_stats(read, pic, two, square, keys, eye, eye2, first, era, looks) -> dict:
    def only(stats):
        return {k: stats[k] for k in keys} if stats and all(isinstance(stats.get(k), int) for k in keys) else None
    if two and int(two) <= 2 and eye and only(eye) and only(eye2):
        halves = [only(eye), only(eye2)]
        out = field({"creatures": halves}, "eye", True, "bar truth")
        out["creatures"] = {"count": 2, "psd": (read or {}).get("creatures"), "picture": halves, "eye": True}
        return out
    if eye is not False and not two:
        seen = only(eye)
        return field(seen, "eye", True, "bar truth") if seen else \
            field(None, "none", False, why="the bars are not readable by eye")
    psd = only((read or {}).get("stats"))
    psd = psd if psd and all(0 <= v <= (40 if square else 10) for v in psd.values()) else None
    if two:                                                    # True or the creature count
        creatures = (read or {}).get("creatures")
        if int(two) >= 3:                                      # Pix: shared HP, a DEF/ATK/SPD row each
            out = field(None, "picture", False, why=f"{int(two)} creatures: read each creature's bars by eye "
                                                    "(the halves reader splits a row in two only), never summed")
            out["creatures"] = {"count": int(two), "psd": creatures, "picture": None}
            return out
        out = field(None, "picture", False, why="two creatures: each half read apart, never summed")
        out["creatures"] = {"count": 2, "psd": creatures,
                            "picture": [(pic.get("halves") or {}).get(k) for k in ("top", "bottom")]}
        return out
    if square:
        return field(psd, "psd", False, why="raid rows (40 cells): no picture reader")
    a, b = only(complete(pic.get("best"))), only(complete(pic.get("template")))
    if psd and a == psd:
        return field(psd, "picture", True, "psd")
    if first and psd is None:
        return first_stats(a, b, only, era, looks)
    if psd is None and a and a == b:
        if era_agrees(a, era):
            return field(a, "picture", True, "second reader")
        out = field(a, "picture", False, why="the picture readers agree, but its era read is not sure on every stat "
                                             "or says otherwise")
        out["reads"] = {"psd": None, "picture": a, "second": b,
                        "era": {k: (s or {}).get("v") for k, s in ((era or {}).get("stats") or {}).items()} or None}
        return out
    if psd and a:
        votes = Counter(json.dumps(x, sort_keys=True) for x in (a, b, psd) if x)
        top, n = votes.most_common(1)[0]
        out = field({k: json.loads(top)[k] for k in keys} if n > 1 else a, "picture", False,
                    why="the picture and the PSD's rows disagree")
    elif a:
        out = field(a, "picture", False, why="one picture reader only" if not b else "the picture readers disagree")
    elif psd:
        out = field(psd, "psd", False, why="the picture's bars unread")
    else:
        out = field(None, "none", False, why="no stats read")
    out["reads"] = {"psd": psd, "picture": a, "second": b}
    return out


def era_agrees(stats: dict, era) -> bool:
    """The era read (era_bars) is sure on every one of these stats and reads the same (Dogma 2023-04-21: its static
    bars fool both readers, the era read too, but not surely)."""
    got = (era or {}).get("stats") or {}
    return all((got.get(k) or {}).get("sure") and got[k].get("v") == v for k, v in stats.items())


def first_stats(a, b, only, era, looks) -> dict:
    """A first-repo row's stats with no PSD stats (check_stats): the same look's Discord eye read unless its own era
    read is sure on a stat and says otherwise (a one-cell bar change keeps the hash: False Knight 2023-04-04), else its
    era read when sure on every stat, else a reader that agrees with the same look's sure Discord read; else review."""
    stats = (era or {}).get("stats") or {}
    seen = only({k: (s or {}).get("v") for k, s in stats.items()})
    era_sure = seen and all(stats.get(k, {}).get("sure") for k in seen)
    own = {k: s["v"] for k, s in stats.items() if (s or {}).get("sure") and isinstance(s.get("v"), int)}

    def fits(stats):
        return not any(k in own and own[k] != v for k, v in stats.items())
    eyes = [only(s) for how, s in looks if how == "eye" and only(s)]
    told = [t for t in eyes if fits(t)]
    disc = [only(s) for how, s in looks if how == "read" and only(s)]
    reads = {"picture": a, "second": b, "era": seen, **({"era_name": era.get("era")} if era else {}),
             **({"discord": eyes + disc} if eyes or disc else {})}
    if told:
        out = field(told[0], "eye", True, "the same look's Discord picture, by eye")
    elif era_sure:
        out = field(seen, "picture", True, f"era reader ({era.get('era')})" +
                    (", over the same look's Discord picture" if eyes else ""))
    elif any(x in disc and fits(x) for x in (a, b, seen) if x):
        out = field(next(x for x in (a, b, seen) if x and x in disc and fits(x)), "picture", True,
                    "the same look's Discord read")
    else:
        out = field(a or seen or b, "picture" if a or seen or b else "none", False,
                    why="first repo, no PSD: its sure era read and the same look's Discord eye read disagree" if eyes
                    else "first repo, no PSD: no eye read, calibrated era read or same look's Discord read agrees")
    out["reads"] = reads
    return out


def check_fields(data: Data, png: dict, psd, pick: str, off, today: dict) -> tuple:
    """(fields {name: {value, method, confidence, check?, why?}}, card kind, two creatures, psd note)."""
    ocr = data.ocr_of(png["blob"])
    ocr_title = " ".join(t for t, *_ in ocr.get("title") or [])
    read, how = read_of(data, psd, ocr_title, png["path"]) if psd else (None, "no psd")
    kind = (read or {}).get("type") or kind_of(png["path"])
    if kind == "unused" and today.get("type"):
        kind = today["type"]                                   # a status card kept in Unused then
    same_look = off is not None and off <= SAME_PICTURE
    fields = {}
    stem = splitext(basename(png["path"]))[0].replace("_", " ")
    printed = (read or {}).get("printed_title") or (read or {}).get("title") or (read or {}).get("name")
    fields["name"] = check_name(ocr_title, printed, stem, today.get("names") or (), same_look)
    star_look = same_look
    if psd and not same_look:                                  # the stars' band alone
        top = picture_off(data, psd["blob"], png["blob"], STAR_BAND)
        star_look = top is not None and top <= SAME_PICTURE
    stars = check_stars(folder_stars(png["path"]), data.drawn.get(psd["blob"]) if psd else None, star_look, kind)
    if stars:
        fields["stars"] = stars
    if read is not None:
        fields["types"] = check_types(list(read.get("types") or []), data.icons.get(psd["blob"]), same_look,
                                      data.type_names, kind, today.get("types"), (stars or {}).get("value"))
    pic = data.pictures.get(png["blob"]) or {}
    count = max(len((read or {}).get("creatures") or ()), today.get("creatures") or 0)
    two = kind in STAT_KINDS and bool(count or png["old_id"] in SPLITS or (pic.get("halves") or {}).get("apart", 0) >= TWO_ROWS)
    two = max(count, 2) if two else 0                          # how many creatures
    keys = ORDER if kind in STAT_KINDS else tuple(k for k in ORDER if kind in ITEM_KINDS and (
        k in (pic.get("labels") or ()) or ((read or {}).get("stats") or {}).get(k, -1) >= 0))
    if kind in STAT_KINDS or len(keys) >= 2:                   # an item's HP/DEF bars too
        size = (data.png.get(png["blob"]) or {}).get("size") or [0, 0]
        fields["stats"] = check_stats(read, pic, two, size[0] == size[1] and size[0] >= 2000, keys,
                                      data.eye_bars.get(png["blob"], False), data.eye_bars2.get(png["blob"]),
                                      png.get("repo") == "first", data.era_reads.get(png["blob"]),
                                      data.look_reads.get("b" + png["blob"][:20], ()), kind in STAT_KINDS)
    if kind in ABILITY_KINDS:
        fields["ability"] = check_ability(read, ocr.get("lines") or [], kind, same_look, data.vocab, today.get("ability"))
        if fields["ability"]["method"] == "none" or read is None:     # no PSD text: the text readers' read
            fields["ability"] = text_ability(data, png["blob"], today.get("known") or ()) or fields["ability"]
    got = data.header.get(png["blob"]) if read is None else None
    if got:                                                    # no PSD: its header against a git version's
        fields["types"] = field(sorted(got["types"]), "git header", got["sure"], f"{got['version']} ncc {got['ncc']}",
                                "" if got["sure"] else "its header under the bar, or near a version with other types")
    note = {"blob": psd["blob"], "commit": psd["commit"], "how": pick, "read": how,
            **({"off": off} if off is not None else {})} if psd else None
    return fields, kind, two, note


def icon_type(icon: dict, names: set):
    """The type an icon layer's picture is: its layer's own type name when that picture is near enough, else the
    nearest Types/ picture when near and clearly nearer than the next; None: not identified."""
    near = icon.get("near") or []
    if not near:
        return None
    by = dict(near)
    own = re.sub(r"( copy( \d+)?)+$", "", icon["layer"].strip().lower())
    if own in names and (by.get(own, ICON_SAME + 1) <= ICON_SAME or near[0][0] == own):
        return own                                             # named so, and its picture is that or nearest it
    best, d = near[0]
    if d <= ICON_SAME and (len(near) < 2 or near[1][1] - d >= ICON_GAP or near[1][1] >= 2 * d):
        return best
    return None


def today_types(types: list, kind: str) -> list:
    """Types under today's names: renamed types (undread, tornado) and, on a minion, the 2023 role icons (the Equip
    sword marked an offensive minion, the Protective shield a defensive one)."""
    types = [OLD_TYPES.get(t, t) for t in types]
    return [MINION_ICONS.get(t, t) for t in types] if kind == "minion" else types


def check_types(read: list, icons, same_look: bool, names: set, kind: str = "", today=None, stars=None) -> dict:
    """The types (check_icons) under today's names; the icons' own names kept as raw when they differ. A type twice
    that today's card (today: its types; None: no card today) has fewer times goes to review: one icon drawn as two
    layers (Smoldering's stacked 'debuff' pair is one double arrow, Mr Jawsum's two drops print one)."""
    out = check_icons(read, icons, same_look, names, kind, stars)
    value = out["value"]
    if value is not None and today_types(value, kind) != value:
        out["raw"], out["value"] = value, today_types(value, kind)
    twice = [t for t, n in Counter(out["value"] or ()).items() if n > 1 and n > list(today or ()).count(t)]
    if twice and out["confidence"] == "sure":
        out.update(confidence="review", why=f"'{twice[0]}' twice" + (", today's card has fewer" if today is not None
                                                                      else "") + ": one icon drawn as two layers?")
        out.pop("check", None)
    return out


def check_icons(read: list, icons, same_look: bool, names: set, kind: str = "", stars=None) -> dict:
    """The PSD's types, checked by its header icons' pictures: an icon layer's picture names its type (where the
    parser missed the layer's name, or the name is another type's); a read type no icon layer explains stays, but on
    an item it goes to review (Paintbrush's art has layers named after types). A repeated layer no picture names is
    art beside a type icon, or the stars when drawn as many times (stars: the row's count; Zote's flowers); else
    review (Kyle's two glitch cubes)."""
    look = "composite" if same_look else ""
    if icons is None:
        return field(read, "psd", same_look, look, "" if same_look else "the PSD's look is not the picture's")
    base = lambda icon: re.sub(r"( copy( \d+)?)+$", "", icon["layer"].strip().lower())
    twice = Counter(base(icon) for icon in icons)
    found = [(base(icon), icon_type(icon, names)) for icon in icons]
    typed = any(name for _, name in found)
    spare = lambda own: twice[own] >= 2 and (typed or twice[own] == stars)
    unknown = [icon["layer"] for icon, (own, name) in zip(icons, found) if name is None and not spare(own)]
    value, claimed, left = [name for _, name in found if name], set(), []     # left to right, each by its picture
    for t in read:                                             # the parser's: its layer's picture replaces it ...
        hit = next((i for i, (own, name) in enumerate(found) if name and i not in claimed and t in (own, name)), None)
        if hit is None:
            value.append(t)                                    # ... or it stays (no icon layer of it found)
            left.append(t)
        else:
            claimed.add(hit)
    if unknown:
        out = field(read, "psd", False, why="header icons not identified: " + ", ".join(unknown))
        out["icons"] = [name for _, name in found if name]
        return out
    if left and kind in ITEM_KINDS:
        out = field(value, "psd", False, why="read types no header icon shows (art?): " + ", ".join(left))
        out["read"] = read
        return out
    out = field(value, "psd" if value == read else "icon pictures", same_look, "icons" + (", " + look if look else ""),
                "" if same_look else "the PSD's look is not the picture's")
    if value != read:
        out["read"] = read
    return out


def ability_top(lines: list) -> float:
    """Where a 1:2 card's ability starts: under its stat labels."""
    R = _W.get("readers") or _W.setdefault("readers", readers())
    ys = [b[3] for b in R.labels(lines).values()]
    return (max(ys) + 0.01) if ys else 0.69


def ability_zone(lines: list, kind: str) -> list:
    """The OCR'd lines where the ability is printed, in reading order: under the bars above a 1:2 card's class line
    zone; under the title on other cards."""
    if kind not in STAT_KINDS:
        return sorted((line for line in lines if line[3] + line[5] / 2 >= 0.13), key=lambda line: (line[3], line[2]))
    top = ability_top(lines)
    return sorted((line for line in lines if top <= line[3] + line[5] / 2 <= CLASS_ZONE), key=lambda line: (line[3], line[2]))


def ability_lines(lines: list, kind: str) -> list:
    """The OCR'd ability two ways: the eras reader's (lines under the bars, its class line dropped), and every line
    under the bars above a 1:2 card's class line zone (an ability's last short line kept)."""
    R = _W.get("readers") or _W.setdefault("readers", readers())
    if kind not in STAT_KINDS:
        return [R.ability_of(lines, 0.13, 1.0, drop_class=False)]
    return [R.ability_of(lines, ability_top(lines)), " ".join(line[0] for line in ability_zone(lines, kind))]


def words3(text) -> list:
    return [w for w in re.findall(r"[a-z0-9]+", re.sub(r"\[[^\]]*\]", " ", text or "").lower())
            if len(w) >= 3 and not w.isdigit()]


def printed_back(text: str, zone: list, vocab: set) -> dict:
    """The check the other way round: of the card words (vocab) OCR reads where the ability is printed, the share the
    PSD text has (near spellings count: OCR), the lines it lacks (3+ card words, at most a third of them in the text)
    and whether the digits are the same."""
    from difflib import SequenceMatcher
    truth = set(words3(text))
    near = lambda w: w in truth or any(abs(len(w) - len(t)) <= 2 and SequenceMatcher(None, w, t).ratio() >= NAME_RATIO
                                       for t in truth)
    real = found = 0
    lacking = []
    for line in zone:
        ws = [w for w in words3(line[0]) if w in vocab]
        hit = sum(1 for w in ws if near(w))
        real, found = real + len(ws), found + hit
        if len(ws) >= 3 and hit <= len(ws) / 3:
            lacking.append((line[0][:120], len(ws)))
    digits = sorted(re.findall(r"\d", re.sub(r"\[[^\]]*\]", " ", text))) == \
        sorted(d for line in zone for d in re.findall(r"\d", line[0]))
    return {"share": round(found / real, 2) if real else None, "digits": digits,
            "lacking": [t for t, _ in lacking], "lacking_words": sum(n for _, n in lacking)}


def recall(R, text: str, seen: list):
    """How much of the ability the picture prints, the best of the readings: {"words": its words in order, "letters":
    its letters and digits in order (OCR joins words: '1or 2')}; None without words."""
    from difflib import SequenceMatcher
    truth = R.strip_tokens(text)
    if not R.words(truth):
        return None
    words = letters = 0.0
    for got in seen:
        a = norm(truth)
        words = max(words, R.word_score(truth, got) or 0.0)
        letters = max(letters, sum(x.size for x in SequenceMatcher(None, a, norm(got), autojunk=False).get_matching_blocks()) / len(a))
    return {"words": round(words, 2), "letters": round(letters, 2)}


def dangling(text: str) -> list:
    """Words a line ends in that want an icon after them ('can't affect', 'per') where the text ends or the next line
    starts a new sentence: an icon printed at a line's end the PSD text lacks (status cards)."""
    lines = [line.strip() for line in re.split(r"[\r\n]+", text or "") if line.strip()]
    out = []
    for i, line in enumerate(lines):
        last = re.search(r"([A-Za-z]+)$", line)
        after = lines[i + 1][0] if i + 1 < len(lines) else None
        if last and last.group(1).lower() in DANGLING and (after is None or after.isupper() or after == "("):
            out.append(last.group(1))
    return out


def leading(text: str):
    """The punctuation a line of the text starts with (': top stats', ', [ice]: +2 DEF'), or None."""
    lines = (line.strip() for line in re.split(r"[\r\n]+", text or ""))
    return next((line[0] for line in lines if line and line[0] in LEADING), None)


def with_icons(text: str, today) -> bool:
    """Today's ability is this text's words with more icons."""
    bare = lambda t: text_key(TOKEN.sub(" ", t or ""))
    return bool(today) and bare(today) == bare(text) and len(TOKEN.findall(today)) > len(TOKEN.findall(text))


def check_ability(read, lines: list, kind: str, same_look: bool = True, vocab=frozenset(), today=None) -> dict:
    """The PSD's ability checked against the OCR both ways: its words are printed (recall), and the printed card words
    are in it (printed_back: no lines it lacks; with the composite off the PNG, every card word and digit). A line
    starting with punctuation, or ending in a word that wants an icon, goes to review (the latter: today's text proposed
    when it is these words with icons)."""
    R = _W.get("readers") or _W.setdefault("readers", readers())
    text = OLD_TOKENS.sub(lambda m: "[" + OLD_TYPES[m.group(1)], ((read or {}).get("ability") or "").strip())
    text = re.sub(r"%{2,}", "%", text)       # a % in a font with no % glyph prints nothing (The Thin Man's '25%%')
    seen = ability_lines(lines, kind)
    if text:
        found = recall(R, text, seen)
        back = printed_back(text, ability_zone(lines, kind), vocab)
        ends, lead = dangling(text), leading(text)
        if GAP.search(text):
            why = "a gap in the text: an icon not read"
        elif lead:
            why = f"a line starts with '{lead}': a word or icon before it not read"
        elif ends:
            why = f"a line ends in '{ends[0]}': an icon there not read"
            if with_icons(text, today):
                out = field(today, "psd + today's icons", False, why=why + "; proposed: today's text (the same words)")
                out["psd"], out["recall"], out["back"] = text, found, back
                return out
        elif (read or {}).get("ability_layers", 1) > 1:
            why = f"joined from {read['ability_layers']} text layers"
        elif not (found is None or found["words"] >= RECALL or found["letters"] >= LETTERS):
            why = "the OCR'd text has too little of it"
        elif back["lacking_words"] >= MISSING_WORDS:
            why = "the card prints ability lines the PSD text lacks"
        elif not same_look and not (back["share"] == 1 and back["digits"]):
            why = "the PSD's look is not the picture's, and the printed words or digits differ from its text"
        else:
            return field(text, "psd", True, "ocr" if same_look else "ocr both ways")
        out = field(text, "psd", False, why=why)
        out["recall"], out["back"] = found, back
        return out
    if max(len(R.words(got)) for got in seen) >= TEXT_SEEN:
        out = field(None, "none", False, why="the card prints text the PSD read has not")
        out["ocr"] = max(seen, key=len)[:300]
        return out
    return field("", "psd", read is not None, "ocr", "" if read is not None else "no PSD")


def text_readers():
    if "text" not in _W:
        import timeline_text
        _W["text"] = timeline_text
    return _W["text"]


def text_ability(data: Data, blob: str, known) -> dict:
    """A PNG whose PSD gives no text: the text readers' read (timeline_text read; its era's stored bar, no refit): sure
    when its words are one of the card's own texts ('ocr+known': today's, its PSDs'), or at its era's OCR bar; else
    proposed for review. None: not read."""
    rd = data.text_raw.get(blob)
    if not rd:
        return None
    tt = text_readers()
    words = {tuple(tt.ability_words(k)) for k in known if k}
    for variant in ("icons", "fix", "raw"):
        text = (tt.ability_variant(rd, variant) or {}).get("text")
        if text and tuple(tt.ability_words(text)) in words:
            return {**field(text, "ocr+known", True, "a known text of the card"), "conf": 1.0}
    how, bar = data.text_bar(rd.get("era"), "ability")
    if how is None:
        return None
    val, method, conf = tt.field_read(rd, "ability", how)
    if not val:
        return None
    sure = bar is not None and conf >= bar and not tt.no_sure(rd, ())
    return {**field(val, method, sure, why="" if sure else "the text readers' read, under its era's bar"),
            "conf": conf}


def known_texts(data: Data, versions: list, today: dict) -> list:
    """The card's own ability texts: today's and every read of its PSDs (an old PNG's OCR equal to one is sure)."""
    found = {today.get("ability")}
    for v in versions:
        if v["ext"] == "psd":
            for r in (data.psd.get((v["blob"], v["path"])), data.rereads.get((v["blob"], v["path"]))):
                found.add(((r or {}).get("read") or {}).get("ability"))
    return sorted(t for t in found if t)


def creature_reads(f: dict):
    """A multi-creature card's stats as read (each creature's picture half, the PSD's rows); None unless every half
    is read in full."""
    reads = (f or {}).get("creatures") or {}
    halves = reads.get("picture")
    if not halves or any(not half or None in half.values() for half in halves):
        return None
    return json.dumps([halves, reads.get("psd")], sort_keys=True)


def changed(a: dict, b: dict) -> list:
    """Fields b changed from a, sure or not (a field either could not read is no change here: settled() keeps such
    rows from folding; text beyond formatting only; a multi-creature card's stats by its creatures' reads)."""
    out = []
    for name in ("name", "stars", "types", "stats", "ability"):
        x, y = (a.get(name) or {}).get("value"), (b.get(name) or {}).get("value")
        if name == "stats" and x is None and y is None:
            x, y = creature_reads(a.get(name)), creature_reads(b.get(name))
        if x is None or y is None:
            continue
        if name in ("name", "ability"):
            if text_key(x) != text_key(y):
                out.append(name)
        elif name == "types":
            if Counter(x) != Counter(y):
                out.append(name)
        elif x != y:
            out.append(name)
    return out


def settled(a: dict, b: dict) -> bool:
    """Every field either row has is sure in both and the same (a multi-creature card's stats: the same full reads of
    each creature): only then may a small look change fold as minor (owner 11-09: a data change is always listed)."""
    for name in set(a) | set(b):
        x, y = a.get(name), b.get(name)
        if not x or not y:
            return False
        if name == "stats" and (x.get("creatures") or y.get("creatures")):
            if creature_reads(x) is None or creature_reads(x) != creature_reads(y):
                return False
        elif any(f.get("confidence") != "sure" or f.get("value") is None for f in (x, y)):
            return False
    return not changed(a, b)


def ability_change(prev: dict, row: dict) -> str:
    """same / formatting / changed between two sure texts; unknown when either is missing or not sure (a text run never
    joins across it); same for one picture."""
    a, b = prev["fields"].get("ability") or {}, row["fields"].get("ability") or {}
    if prev["blob"] == row["blob"]:
        return "same"                                          # one picture prints one card
    if any(f.get("value") is None or f.get("confidence") != "sure" for f in (a, b)):
        return "unknown"
    from commands.changelog import formatting_only
    return "same" if a["value"] == b["value"] else "formatting" if formatting_only(a["value"], b["value"]) else "changed"


def psd_first(data: Data, key, versions: list, pngs: list) -> tuple:
    """(the card's first old repo record when it is a PSD added before any PNG of it: {ts, date, commit, blob, path,
    author, subject}, that PSD's version when its composite is a look of its own: a row) or (None, None). A PSD
    whose composite is the first PNG's picture only dates the card (S6: 'old repo (PSD)')."""
    psds = [v for v in versions if v["ext"] == "psd"]
    if not psds or not pngs or psds[0]["ts"] >= pngs[0]["ts"]:
        return None, None
    first = psds[0]
    off = picture_off(data, first["blob"], pngs[0]["blob"])
    found = {"ts": first["ts"], "date": first["date"], "commit": first["commit"], "blob": first["blob"],
             "path": first["path"], "author": first.get("author_mapped") or data.authors.get(first["author"], first["author"]),
             "subject": first["subject"], "off": off, "repo": first.get("repo") or "old"}
    return found, first if off is not None and off > SAME_PICTURE else None


def psd_alone(data: Data, old_id: str) -> bool:
    """An old card of PSDs alone (no PNG version: 'sumo outside', Items/Freeze_Spell.psd)."""
    found = data.versions.get(old_id) or []
    return bool(found) and all(v["ext"] == "psd" for v in found)


def card_rows(data: Data, key, old_ids: list, today: dict, psd_from=frozenset()) -> list:
    """The rows of one card's old files' PNG versions, oldest first; a PSD added before any PNG whose composite is a
    look of its own first (psd_only: its composite and read; Tainted Lazarus 2023-09-10). psd_from: old cards of
    PSDs alone the owner gave this card: each PSD the card's PSD of its commit when its composite is that commit's
    PNG (and the PNG's own PSD is not), else a psd_only row when a look of its own."""
    from commands.png_changes import VISIBLE
    versions = sorted((v for o in old_ids for v in data.versions.get(o, [])), key=lambda v: (v["ts"], v["path"]))
    pngs = [v for v in versions if v["ext"] == "png"]
    names = {norm(n) for n in today.get("names") or ()}
    today = {**today, "known": known_texts(data, versions, today)}
    rows = []
    first, own = psd_first(data, key, versions, pngs)
    if first:
        data.psd_firsts[key] = {k: v for k, v in first.items() if v is not None}
    extra, seen = [], set()
    for v in versions:                             # the owner's PSDs alone: one per blob, its first version
        if v["ext"] == "psd" and v["old_id"] in psd_from and v["blob"] not in seen:
            seen.add(v["blob"])
            extra.append(v)
    picked, used = {}, set()
    for png in pngs:
        psds = [v for v in versions if v["ext"] == "psd" and v["old_id"] == png["old_id"]]
        until = next((p["ts"] for p in pngs[pngs.index(png) + 1:] if p["old_id"] == png["old_id"] and
                      p["ts"] > png["ts"]), float("inf"))
        psd, pick, off = pick_psd(data, png, psds, until)
        if off is None or off > SAME_PICTURE:      # an owner's PSD of the same commit that is this picture
            alt = [(picture_off(data, s["blob"], png["blob"]), s) for s in extra if s["commit"] == png["commit"]]
            alt = sorted((o, s["path"], s) for o, s in alt if o is not None and o <= SAME_PICTURE)
            if alt:
                off, _, psd = alt[0]
                pick = "same commit, the owner's PSD"
                used.add(psd["blob"])
        picked[png["id"]] = (psd, pick, off)
    alone = [s for s in extra if s is not own and s["blob"] not in used and not any(
        (o := picture_off(data, s["blob"], p["blob"])) is not None and o <= SAME_PICTURE for p in pngs)]
    alone_ids = {s["id"] for s in alone}
    for png in sorted(([own] if own else []) + alone + pngs, key=lambda v: (v["ts"], v["ext"] == "png", v["path"])):
        if png is own or png["id"] in alone_ids and png["ext"] == "psd":    # the composite is the picture
            psd, pick, off = png, "its own composite", 0.0
        else:
            psd, pick, off = picked[png["id"]]
        fields, kind, two, note = check_fields(data, png, psd, pick, off, today)
        row = {"id": png["id"], "card": key, "old_id": png["old_id"], "date": png["date"], "ts": png["ts"],
               "author": png.get("author_mapped") or data.authors.get(png["author"], png["author"]), "subject": png["subject"],
               "badge": FIRST_BADGE if png.get("repo") == "first" else BADGE, "repo": png.get("repo") or "old",
               "link": None, "commit": png["commit"], "path": png["path"], "blob": png["blob"], "kind": png["kind"],
               "new_look": png["new_look"], "thumb": f"oldgit/thumbs/{png['blob']}.webp", "card_kind": kind,
               "fields": fields, "psd": note}
        if png is own or png["ext"] == "psd":
            row["psd_only"] = True                 # no PNG of it: the PSD's composite (its thumb) is the picture
        if png.get("lost"):
            row["lost"] = True                     # the old repo's lost 2023 history (oldgit_index LOST_GLOB)
        name, stem = fields["name"]["value"], splitext(basename(png["path"]))[0].replace("_", " ")
        if name and names and norm(name) not in names:
            row["old_name"] = name                             # printed then: shown on the row, never an alias
        if names and norm(stem) not in names:
            row["old_file"] = stem                             # a renamed file (oldgit map's renames)
        if two:
            row["two_creatures"], row["creatures"] = True, two
        rows.append(row)
    confirm_stars(rows, today)                                 # before the fold: it moves stars' confidence
    run = 0
    for prev, row in zip([None] + rows, rows):
        if prev is None:
            row["minor"], row["data_change"], row["ability_change"] = False, [], "first"
        else:
            same = prev["blob"] == row["blob"]                 # one picture prints one card
            share = 0.0 if same else data.shares.get((prev["blob"], row["blob"]))
            row["share"] = share
            row["data_change"] = [] if same else changed(prev["fields"], row["fields"])
            row["minor"] = share is not None and share < VISIBLE and (same or settled(prev["fields"], row["fields"]))
            row["ability_change"] = ability_change(prev, row)
            run += row["ability_change"] in ("changed", "unknown")
        row["text_run"] = run                                  # one text, formatting aside: its last form, dated by its first
    return rows


def psd_rows(data: Data, key, old_id: str, today: dict, ids: list, base: list) -> list:
    """An unsure map's old card of PSDs alone (its rows wait for the owner): the card's rows with its PSDs
    (card_rows psd_from) that differ from base, the card's own rows: a row it gave its PSD ('replaces' that row,
    psd_file) and its own looks (psd_only)."""
    kept = data.psd_firsts.get(key)
    found = card_rows(data, key, list(ids) + [old_id], today, psd_from={old_id})
    if kept is None:
        data.psd_firsts.pop(key, None)
    else:
        data.psd_firsts[key] = kept
    before = {r["id"]: r for r in base}
    path = {v["blob"]: v["path"] for v in data.versions.get(old_id) or ()}
    out = []
    for r in found:
        if r["id"] not in before:
            out.append(r)
        elif (r.get("psd") or {}).get("blob") in path and (r["psd"] or {}).get("blob") != \
                (before[r["id"]].get("psd") or {}).get("blob"):
            out.append({**r, "replaces": r["id"], "psd_file": path[r["psd"]["blob"]]})
    return out


def unsure(row: dict) -> list:
    return [name for name, f in row["fields"].items() if f["confidence"] != "sure"]


def review_items(rows: list, kind: str = "field") -> list:
    """One item per row and field that is not sure (a first-repo row with no PSD: only fields its picture proposes)."""
    items = []
    for row in rows:
        for name in unsure(row):
            f = row["fields"][name]
            if row.get("repo") == "first" and (row.get("psd") or {}).get("read", "unread") == "unread" and \
                    f.get("value") in (None, ""):
                continue                                       # nothing read to confirm (its PSD not fetched)
            items.append({"id": "oldrows-" + stable_id(kind, row["id"], name), "kind": f"{name} unsure", "card": row["card"],
                          "row": row["id"], "date": row["date"], "thumb": row["thumb"], "field": name,
                          "proposed": f.get("value"), "why": f.get("why"),
                          **{k: f[k] for k in ("reads", "creatures", "recall", "back", "ocr", "raw", "psd") if k in f}})
    return items


def split_rows(data: Data, old_id: str, rows: list) -> dict:
    """STAGE1 Q3 (owner A): each half's rows 'as <old name>', its half's stats; the halves matched to the live cards
    by today's stats (else their last old versions'). Every reading is kept for the owner: the first versions after
    the split read the other way round (Twin Princes 10-16), but the bar colours follow today's (Lorian's orange
    top rows, Lothric's blue bottom ones: gap round 3 by eye)."""
    cards = SPLITS[old_id]
    refs = {"today": {}, "last old version": {}, "first old version": {}}
    days = defaultdict(set)
    for key in cards:
        if (data.today.get(key) or {}).get("stats"):
            refs["today"][key] = data.today[key]["stats"]
        ids = [o for o, c in data.map["cards"].items() if c.get("key") == key and c.get("confidence") == "sure"]
        found = [(v, s) for v, s in ((v, valid_stats(((data.psd.get((v["blob"], v["path"])) or {}).get("read") or {})
                                                      .get("stats")))
                                     for v in sorted((v for o in ids for v in data.versions.get(o, [])
                                                      if v["ext"] == "psd"), key=lambda v: v["ts"])) if s]
        if found:
            refs["first old version"][key], refs["last old version"][key] = found[0][1], found[-1][1]
            days["first old version"].add(found[0][0]["date"][:10])
            days["last old version"].add(found[-1][0]["date"][:10])
    last = (rows[-1]["fields"].get("stats") or {}).get("creatures", {}).get("picture") or [None, None]
    dist = lambda a, b: sum(abs(a[s] - b[s]) for s in ORDER) if a and b and None not in a.values() else 99
    readings = {}
    for name, ref in refs.items():
        if len(ref) == len(cards):
            ways = {"top": cards[0], "bottom": cards[1]}, {"top": cards[1], "bottom": cards[0]}
            costs = [dist(last[0], ref[w["top"]]) + dist(last[1], ref[w["bottom"]]) for w in ways]
            readings[name] = {**ways[costs.index(min(costs))], "distance": min(costs), "other way": max(costs),
                              "stats": ref, **({"days": sorted(days[name])} if days.get(name) else {})}
    by = next((n for n in refs if n in readings), None)
    order = (readings[by]["top"], readings[by]["bottom"]) if by else cards
    out = {}
    for half, key in enumerate(order):
        out[key] = [{**r, "card": key, "old_name": old_id, "split_from": old_id,
                     "fields": {**r["fields"], "stats": half_stats(r["fields"].get("stats") or {}, half, old_id)}}
                    for r in rows]
    return {"halves": {k: ("top" if i == 0 else "bottom") for i, k in enumerate(order)}, "matched_by": by,
            "readings": readings, "rows": out, **({"eye": SPLIT_EYE[old_id]} if old_id in SPLIT_EYE else {})}


def half_stats(stats: dict, half: int, old_id: str) -> dict:
    """A split card's row: one half's stats (owner Q3), sure when both halves were read by eye."""
    creatures = stats.get("creatures") or {}
    value = (creatures.get("picture") or [None, None])[half]
    where = f"{'top' if half == 0 else 'bottom'} half of {old_id}'s rows"
    if creatures.get("eye") and value:
        out = field(value, "eye", True, "bar truth")
        out["half"] = where
        return out
    return field(value, "picture", False, why=where)


def rows_main():
    data = Data()
    data.load_today()
    log(f"{sum(len(v) for v in data.versions.values())} versions, {len(data.pictures)} pictures read, "
        f"{len(data.rereads)} rereads, {len(data.ocr)} OCR'd versions")
    live_rows, archived_rows, review = [], [], []
    found = groups(data)
    live_ids = {key: ids for what, key, ids in found if what == "live"}
    of_card = defaultdict(list)
    for what, key, old_ids in found:
        today = data.today.get(key) or {"names": {key or old_ids[0]}}
        owners = {o for o in old_ids if what == "live" and psd_alone(data, o) and
                  "owner" in (data.map["cards"][o].get("how") or ())}
        if what == "unsure map" and psd_alone(data, old_ids[0]) and key in live_ids:
            rows = psd_rows(data, key, old_ids[0], today, live_ids[key], of_card[key])
        else:
            rows = card_rows(data, key, old_ids, today if what in ("live", "unsure map") else
                             {"names": {key or old_ids[0]}}, owners)
        if what == "live":
            of_card[key] = rows
        if what == "live":
            live_rows += rows
            review += review_items(rows)
        elif what == "archived":
            archived_rows += rows
        elif what == "unsure map":                             # the rows wait for the owner
            old_id = old_ids[0]
            card = data.map["cards"][old_id]
            item = {"id": "oldrows-" + stable_id("map", old_id), "kind": "unsure map", "old_id": old_id, "card": key,
                    "status": card["status"], "how": card.get("how"), "why": card.get("why_unsure"), "rows": rows}
            if old_id in SPLITS:
                item["split"] = split_rows(data, old_id, rows)
                item["decided_without_you"] = (f"Q3 A: both {' and '.join(SPLITS[old_id])} start with {old_id}'s "
                                               f"versions ('as {old_id}'), each with its own half's stats")
            review.append(item)
        elif what == "lost":                                   # Morgue candidates (Q2 A: a file in the old repo)
            card = next(c for c in data.lost["cards"] if c["old_id"] == old_ids[0])
            review.append({"id": "oldrows-" + stable_id("lost", card["old_id"]), "kind": "lost", "old_id": card["old_id"],
                           "first": card["first"], "deleted": card.get("deleted"), "note": card.get("note"),
                           "nearest": card.get("nearest"), "rows": rows})
        else:
            review.append({"id": "oldrows-" + stable_id("morgue", old_ids[0]), "kind": "morgue", "old_id": old_ids[0],
                           "card": key, "rows": rows})
    init = {sha for path, (sha, _) in oi.ls_tree(oi.OG, oi.TC_INIT).items() if oi.ext_of(path) == "png"}
    for row in live_rows:                                      # the review items a row has
        row["review"] = [i["id"] for i in review if i.get("row") == row["id"]]
        row["in_tc_init"] = row["blob"] in init                # TTSCardMaker's first version is this picture
    os.makedirs(OUT, exist_ok=True)
    for name, items in (("rows.jsonl", live_rows), ("archived.jsonl", archived_rows), ("review.jsonl", review)):
        with open(join(OUT, name + ".tmp"), "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        os.replace(join(OUT, name + ".tmp"), join(OUT, name))
    live = {r["card"] for r in live_rows}
    oi.write_json(join(OUT, "psd_first.json"), {
        "about": "Live cards whose first old repo record is a PSD added before any PNG of them (S6 dates them by it: "
                 "'old repo (PSD)'); row: its composite is a look of its own, a psd_only row in rows.jsonl.",
        "cards": {k: {**v, "row": any(r.get("psd_only") and r["card"] == k for r in live_rows)}
                  for k, v in sorted(data.psd_firsts.items()) if k in live}})
    summary = summarize(data, live_rows, archived_rows, review)
    oi.write_json(join(OUT, "summary.json"), summary)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


def summarize(data: Data, rows: list, archived: list, review: list) -> dict:
    fields = defaultdict(Counter)
    for row in rows:
        for name, f in row["fields"].items():
            fields[name][f["confidence"]] += 1
            fields[name][f"method {f['method']}" + (f" + {f['check']}" if f.get("check") else "")] += 1
    renamed = sorted({r["card"] for r in rows if r.get("old_name")})
    files = sorted({r["card"] for r in rows if r.get("old_file")})
    return {
        "rows": len(rows), "cards": len({r["card"] for r in rows}), "archived rows": len(archived),
        "minor": sum(r["minor"] for r in rows), "shown": sum(not r["minor"] for r in rows),
        "in tc init": sum(r.get("in_tc_init", False) for r in rows),
        "all sure": sum(not unsure(r) for r in rows), "two creatures": sorted({r["card"] for r in rows if r.get("two_creatures")}),
        "psd": dict(Counter((r["psd"] or {}).get("how", "none") for r in rows)),
        "fields": {k: dict(v) for k, v in fields.items()},
        "ability change": dict(Counter(r["ability_change"] for r in rows)),
        "old names": {"cards": len(renamed), "rows": sum(1 for r in rows if r.get("old_name")), "list": renamed},
        "old files": {"cards": len(files), "rows": sum(1 for r in rows if r.get("old_file")), "list": files},
        "review": dict(Counter(i["kind"] for i in review)),
    }


def groups(data: Data) -> list:
    """(what, card key, old ids) of every row set: live and archived cards (sure maps, by key), then each unsure map,
    lost card and Morgue card alone."""
    sure = defaultdict(list)
    for old_id, card in data.map["cards"].items():
        if card["confidence"] == "sure" and card["status"] in ("live", "archived"):
            sure[(card["status"], card["key"])].append(old_id)
    found = [(what, key, sorted(ids)) for (what, key), ids in sorted(sure.items())]
    found += [("unsure map", data.map["cards"][o].get("key"), [o]) for o in data.map["review"]]
    found += [("lost", None, [c["old_id"]]) for c in data.lost["cards"]]
    found += [("morgue", c["key"], [o]) for o, c in sorted(data.map["cards"].items()) if c["status"] == "morgue"]
    return found


def pictures_main(workers: int):
    data = Data()
    prevs, order, seen = defaultdict(set), [], set()
    for _, _, old_ids in groups(data):
        pngs = sorted((v for o in old_ids for v in data.versions.get(o, []) if v["ext"] == "png"),
                      key=lambda v: (v["ts"], v["path"]))
        prev = None
        for png in pngs:
            if png["blob"] not in seen:
                seen.add(png["blob"])
                order.append(png)
            if prev and prev != png["blob"]:
                prevs[png["blob"]].add(prev)
            prev = png["blob"]
    jobs = []
    R = readers()
    for png in order:
        lines = data.ocr_of(png["blob"]).get("lines") or []
        bars = kind_of(png["path"]) in STAT_KINDS + ("unknown",) or len(R.labels(lines)) >= 2    # items with HP/DEF bars
        done = data.pictures.get(png["blob"])
        if done and ("best" in done or not bars):
            continue                                           # a later line of a blob replaces its earlier one
        jobs.append((png["blob"], sorted(prevs[png["blob"]]), lines, bars))
    log(f"pictures: {len(jobs)} PNG versions to read ({sum(len(j[1]) for j in jobs)} changes to measure)")
    os.makedirs(OUT, exist_ok=True)
    n = 0
    with open(join(OUT, "pictures.jsonl"), "a", encoding="utf-8") as out, \
            ProcessPoolExecutor(workers, initializer=w_picture_init) as pool:
        for row in pool.map(w_picture, jobs, chunksize=4):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if n % 100 == 0 or n == len(jobs):
                out.flush()
                log(f"{n}/{len(jobs)} pictures")


# ========================
# The eye check
# ========================
def sample_main(n: int, seed: int = SEED):
    """n random rows: each card at 450 x 900 beside its fields, two to a sheet (sample/sample_N.webp; another seed:
    sample-<seed>/), and sample.json for the tally."""
    from PIL import Image, ImageDraw, ImageFont
    rows = oi.read_jsonl(join(OUT, "rows.jsonl"))
    picked = random.Random(seed).sample(rows, min(n, len(rows)))
    folder = join(OUT, "sample" if seed == SEED else f"sample-{seed}")
    os.makedirs(folder, exist_ok=True)
    font = ImageFont.load_default(size=17)
    tiles = []
    for i, row in enumerate(picked):
        with Image.open(BytesIO(oi.git(oi.STORE, "cat-file", "blob", row["blob"]))) as im:
            card = im.convert("RGBA")
        flat = Image.new("RGBA", card.size, (0, 0, 0, 255))
        flat.alpha_composite(card)
        card = flat.convert("RGB")
        card.thumbnail((450, 900), Image.LANCZOS)
        tile = Image.new("RGB", (900, 920), "white")
        tile.paste(card, (0, 10))
        lines = [f"#{i + 1} {row['card']}", f"{row['date'][:10]} {row['card_kind']}" + (f" (as {row['old_name']})" if row.get("old_name") else "")]
        for name, f in row["fields"].items():
            value = f["value"]
            if name == "stats" and value:
                value = " ".join(f"{s.upper()} {value[s]}" for s in ORDER if s in value)
            lines.append(f"{name} [{f['method']}{'' if f['confidence'] == 'sure' else ', REVIEW'}]:")
            text = str(value).replace("\n", " | ") if value is not None else "-"
            lines += [text[k:k + 44] for k in range(0, min(len(text), 440), 44)] or [""]
        ImageDraw.Draw(tile).multiline_text((460, 10), "\n".join(lines), fill="black", font=font, spacing=3)
        tiles.append(tile)
    for k in range(0, len(tiles), 2):
        sheet = Image.new("RGB", (1800, 920), "white")
        for j, tile in enumerate(tiles[k:k + 2]):
            sheet.paste(tile, (j * 900, 0))
        sheet.save(join(folder, f"sample_{k // 2 + 1:02d}.webp"), "WEBP", quality=80)
    with open(join(folder, "sample.json"), "w", encoding="utf-8") as f:
        json.dump([{"n": i + 1, "id": r["id"], "card": r["card"], "fields": {k: v["value"] for k, v in r["fields"].items()},
                    "sure": {k: v["confidence"] == "sure" for k, v in r["fields"].items()}} for i, r in enumerate(picked)],
                  f, ensure_ascii=False, indent=1)
    log(f"sample: {len(picked)} rows in {folder}")


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    workers = min(4, int(argv[argv.index("-j") + 1])) if "-j" in argv else 2
    if argv[0] == "reread":
        reread_main(workers)
    elif argv[0] == "icons":
        icons_main(workers)
    elif argv[0] == "stars":
        stars_main(workers)
    elif argv[0] == "pictures":
        pictures_main(workers)
    elif argv[0] == "rows":
        rows_main()
    elif argv[0] == "sample":
        sample_main(int(argv[1]) if len(argv) > 1 and argv[1].isdigit() else 30,
                    int(argv[argv.index("--seed") + 1]) if "--seed" in argv else SEED)
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
