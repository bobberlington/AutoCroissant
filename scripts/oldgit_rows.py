#!/usr/bin/env python3
"""
The old repo's version rows for the card timelines (STAGE1 S3; docs/Future/CARD_TIMELINE.md; owner 11-13). By hand.
Each PNG version of a card's old file (oldgit map.json) is a row: date, mapped author, commit subject, badge "old repo",
no link, its thumb, and its fields from the PSD read of the same commit (else the PSD nearest it in the commit range),
each checked against the picture: stats read off the PNG (the printed value wins; a disagreement goes to review), name
and ability against the PNG's OCR (the matcher's card cache), stars against the 'N Stars' folder, types by their header
icon layers' pictures (Types/), and the PSD's composite against the PNG. Two-creature cards are flagged, never summed.
Minor looks (under png_changes.VISIBLE of the pixels, no data change) fold under "minor changes"; formatting-only text
changes merge (owner 11-09). A renamed card's rows show the old name (old_name, old_file); never an alias.

    python oldgit_rows.py reread [-j N]     PSDs whose read has no ability, read again with the text layers named
                                            after their own text as the Ability (rereads.jsonl)
    python oldgit_rows.py icons [-j N]      each PSD's header icon layers matched to Types/ pictures (icons.jsonl)
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
READERS = os.environ.get("OLDROWS_READERS") or join(oi.HOME, "Documents/Tools/wt/scratch/s1110/timelines/eras")
ORDER = ("hp", "def", "atk", "spd")
BADGE = "old repo"
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
TEXT_SEEN = 2               # OCR words under the bars/title: the card prints text ('-25% SPD')
HEADER, ICON_MAX = 0.32, 0.3      # type icon layers: bottom and largest side, in card widths
ICON_SIDE = 32              # type icons compared at this size; one matches a type's picture within
ICON_SAME = 15              # mean RGBA difference (0-255) of one icon (a recolour or an older drawing: ~10) ...
ICON_GAP = 4                # ... nearer than the next type's by this much or half as far (or its layer named so)
ICON_NEAR = 6               # nearest types kept per icon
TWO_ROWS = 2                # rows whose top and bottom halves read apart: two creatures
SPLITS = {"Twin Princes": ("Lothric", "Lorian")}    # one old card, two live ones (STAGE1 Q3, provisional A)
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


def folder_stars(path: str):
    found = re.search(r"(?:^|/)(\d+) Stars/", path)
    return int(found.group(1)) if found else None


def kind_of(path: str) -> str:
    """The card kind its old path classifies (when no PSD read says)."""
    p = path.lower()
    for part, kind in (("/minions/", "minion"), ("debuffs/", "debuff"), ("buffs/", "buff"), ("auxillary/items/", "aux item"),
                       ("auxiliary/items/", "aux item"), ("creatures/", "creature"), ("items/", "item"), ("field/", "field"),
                       ("n.m.e/", "nme")):
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


def header_icons(psd) -> list:
    """Visible picture layers in the header's left half or a 'Types' group (bottom within HEADER card widths of the
    top), icon-sized, no star in their name: a card's type icons (its stars sit right of the title)."""
    w = psd.width
    found = []
    for layer in psd.descendants():
        if layer.is_group() or layer.kind not in ("pixel", "smartobject", "shape") or not layer.is_visible():
            continue
        x0, y0, x1, y1 = layer.bbox
        typed = layer.parent is not None and "type" in getattr(layer.parent, "name", "").lower()   # a 'Types' group
        if y1 > HEADER * w or (x0 + x1) / 2 > 0.55 * w and not typed or \
                not (0.04 * w <= x1 - x0 <= ICON_MAX * w and 0.04 * w <= y1 - y0 <= ICON_MAX * w):
            continue
        if "star" in layer.name.lower():
            continue
        found.append(layer)
    return sorted(found, key=lambda layer: (layer.bbox[0], layer.bbox[1]))


def w_icons(sha: str) -> dict:
    import numpy as np
    from psd_tools import PSDImage
    row = {"blob": sha, "icons": []}
    try:
        psd = PSDImage.open(BytesIO(oi.w_blob(sha)))
        refs = _W["icons"]
        for layer in header_icons(psd):
            try:
                vec = icon_vector(layer.topil())
            except Exception:
                vec = None
            found = {"layer": layer.name[:60], "bbox": list(layer.bbox)}
            if vec is not None:
                best = {}
                for name, ref in refs:
                    d = float(np.abs(vec - ref).mean())
                    best[name] = min(d, best.get(name, d))
                found["near"] = [[name, round(d, 1)] for name, d in sorted(best.items(), key=lambda kv: kv[1])[:ICON_NEAR]]
            row["icons"].append(found)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return row


def w_icons_init(refs):
    w_picture_init()
    _W["icons"] = refs


def icons_main(workers: int):
    blobs = sorted({r["blob"] for r in oi.read_jsonl(join(oi.OUT, "reads.jsonl")) if r["kind"] == "psd"})
    target = join(OUT, "icons.jsonl")
    done = {r["blob"] for r in oi.read_jsonl(target)}
    blobs = [b for b in blobs if b not in done]
    refs = type_icons()
    log(f"icons: {len(blobs)} PSDs, {len(refs)} type pictures ({len({n for n, _ in refs})} types)")
    os.makedirs(OUT, exist_ok=True)
    n = 0
    with open(target, "a", encoding="utf-8") as out, ProcessPoolExecutor(
            workers, initializer=w_icons_init, initargs=(refs,)) as pool:
        for row in pool.map(w_icons, blobs, chunksize=4):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if n % 100 == 0 or n == len(blobs):
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
    oi._W["cat"] = subprocess.Popen(oi.OG + ["cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)


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
        self.type_names = set(json.load(open(join(oi.OUT, "vocabulary.json")))["types"])
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

    def load_today(self):
        stats, _, _, _ = oi.today_cards()
        edits = {}
        try:
            edits = json.loads(oi.git(oi.TG, "show", "origin/main:site/Card Edits.json", env=oi.TENV)).get("cards", {})
        except Exception:
            pass
        for key, card in stats.items():
            names = {key, card.get("display_name") or key, (edits.get(key) or {}).get("name") or key}
            self.today[key] = {"names": names, "creatures": bool(card.get("creatures")), "type": card.get("card_type")}

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


def picture_off(data: Data, psd_blob: str, png_blob: str):
    """Share of 8 px blocks of the PSD's composite thumb off the PNG's (None: no composite or sizes differ)."""
    import numpy as np
    a, b = data.thumb(psd_blob), data.thumb(png_blob)
    if a is None or b is None or a.shape != b.shape:
        return None
    d = np.abs(a - b).max(axis=2)
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


def check_stats(read, pic, two: bool, square: bool, keys: tuple = ORDER) -> dict:
    """The printed stats (keys: an item's HP/DEF bars): the picture's read (two readers), the PSD's digit rows where
    they agree."""
    def only(stats):
        return {k: stats[k] for k in keys} if stats and all(isinstance(stats.get(k), int) for k in keys) else None
    psd = only((read or {}).get("stats"))
    psd = psd if psd and all(0 <= v <= (40 if square else 10) for v in psd.values()) else None
    if two:
        creatures = (read or {}).get("creatures")
        out = field(None, "picture", False, why="two creatures: each half read apart, never summed")
        out["creatures"] = {"psd": creatures, "picture": [(pic.get("halves") or {}).get(k) for k in ("top", "bottom")]}
        return out
    if square:
        return field(psd, "psd", False, why="raid rows (40 cells): no picture reader")
    a, b = only(complete(pic.get("best"))), only(complete(pic.get("template")))
    if psd and a == psd:
        return field(psd, "picture", True, "psd")
    if psd is None and a and a == b:
        return field(a, "picture", True, "second reader")
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
    named = [n for n in (printed, stem, *sorted(today.get("names") or ())) if n]
    hit = next((n for n in named if title_match(ocr_title, n)), None)
    if hit:
        fields["name"] = field(hit, "psd" if hit == printed else "file name" if hit == stem else "today", True, "ocr")
    elif printed and same_look and text_key(printed) == text_key(stem):
        fields["name"] = field(printed, "psd", True, "composite, file name")
    else:
        fields["name"] = field(printed or stem, "psd" if printed else "file name", False,
                               why="the OCR'd title names neither the PSD's title nor the file")
    folder = folder_stars(png["path"])
    stars = (read or {}).get("stars")
    if stars is not None and stars == folder:
        fields["stars"] = field(stars, "psd", True, "folder")
    elif stars is not None and folder is None:
        fields["stars"] = field(stars, "psd", same_look, "composite" if same_look else "",
                                "" if same_look else "the PSD's look is not the picture's")
    elif stars is not None:
        fields["stars"] = field(stars, "psd", False, why=f"the folder says {folder}")
    elif folder is not None or kind != "buff":                 # buffs have no stars
        fields["stars"] = field(folder, "folder", False, why="the PSD read no stars")
    if read is not None:
        fields["types"] = check_types(list(read.get("types") or []), data.icons.get(psd["blob"]), same_look, data.type_names)
    pic = data.pictures.get(png["blob"]) or {}
    two = kind in STAT_KINDS and bool((read or {}).get("creatures") or today.get("creatures") or png["old_id"] in SPLITS
                                      or (pic.get("halves") or {}).get("apart", 0) >= TWO_ROWS)
    keys = ORDER if kind in STAT_KINDS else tuple(k for k in ORDER if kind in ITEM_KINDS and (
        k in (pic.get("labels") or ()) or ((read or {}).get("stats") or {}).get(k, -1) >= 0))
    if kind in STAT_KINDS or len(keys) >= 2:                   # an item's HP/DEF bars too
        size = (data.png.get(png["blob"]) or {}).get("size") or [0, 0]
        fields["stats"] = check_stats(read, pic, two, size[0] == size[1] and size[0] >= 2000, keys)
    if kind in ABILITY_KINDS:
        fields["ability"] = check_ability(read, ocr.get("lines") or [], kind)
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


def check_types(read: list, icons, same_look: bool, names: set) -> dict:
    """The PSD's types, checked by its header icons' pictures: an icon layer's picture names its type (where the
    parser missed the layer's name, or the name is another type's); a read type no icon layer explains stays."""
    look = "composite" if same_look else ""
    if icons is None:
        return field(read, "psd", same_look, look, "" if same_look else "the PSD's look is not the picture's")
    base = lambda icon: re.sub(r"( copy( \d+)?)+$", "", icon["layer"].strip().lower())
    twice = Counter(base(icon) for icon in icons)
    found = [(base(icon), icon_type(icon, names)) for icon in icons]
    unknown = [icon["layer"] for icon, (_, name) in zip(icons, found) if name is None and twice[base(icon)] < 2]   # repeated: stars
    value, claimed = [name for _, name in found if name], set()     # left to right, each by its picture
    for t in read:                                             # the parser's: its layer's picture replaces it ...
        hit = next((i for i, (own, name) in enumerate(found) if name and i not in claimed and t in (own, name)), None)
        if hit is None:
            value.append(t)                                    # ... or it stays (no icon layer of it found)
        else:
            claimed.add(hit)
    if unknown:
        out = field(read, "psd", False, why="header icons not identified: " + ", ".join(unknown))
        out["icons"] = [name for _, name in found if name]
        return out
    out = field(value, "psd" if value == read else "icon pictures", same_look, "icons" + (", " + look if look else ""),
                "" if same_look else "the PSD's look is not the picture's")
    if value != read:
        out["read"] = read
    return out


def ability_lines(lines: list, kind: str) -> list:
    """The OCR'd ability two ways: the eras reader's (lines under the bars, its class line dropped), and every line
    under the bars above a 1:2 card's class line zone (an ability's last short line kept)."""
    R = _W.get("readers") or _W.setdefault("readers", readers())
    if kind not in STAT_KINDS:
        return [R.ability_of(lines, 0.13, 1.0, drop_class=False)]
    ys = [b[3] for b in R.labels(lines).values()]
    top = (max(ys) + 0.01) if ys else 0.69
    kept = sorted((line for line in lines if top <= line[3] + line[5] / 2 <= CLASS_ZONE), key=lambda line: (line[3], line[2]))
    return [R.ability_of(lines, top), " ".join(line[0] for line in kept)]


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


def check_ability(read, lines: list, kind: str) -> dict:
    R = _W.get("readers") or _W.setdefault("readers", readers())
    text = ((read or {}).get("ability") or "").strip()
    seen = ability_lines(lines, kind)
    if text:
        found = recall(R, text, seen)
        if GAP.search(text):
            out = field(text, "psd", False, why="a gap in the text: an icon not read")
            out["recall"] = found
            return out
        if (read or {}).get("ability_layers", 1) > 1:
            out = field(text, "psd", False, why=f"joined from {read['ability_layers']} text layers")
            out["recall"] = found
            return out
        if found is None or found["words"] >= RECALL or found["letters"] >= LETTERS:
            return field(text, "psd", True, "ocr")
        out = field(text, "psd", False, why="the OCR'd text has too little of it")
        out["recall"] = found
        return out
    if max(len(R.words(got)) for got in seen) >= TEXT_SEEN:
        out = field(None, "none", False, why="the card prints text the PSD read has not")
        out["ocr"] = max(seen, key=len)[:300]
        return out
    return field("", "psd", read is not None, "ocr", "" if read is not None else "no PSD")


def changed(a: dict, b: dict) -> list:
    """Fields b changed from a (a field either could not know is no change; text beyond formatting only)."""
    out = []
    for name in ("name", "stars", "types", "stats", "ability"):
        x, y = (a.get(name) or {}).get("value"), (b.get(name) or {}).get("value")
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


def card_rows(data: Data, key, old_ids: list, today: dict) -> list:
    """The rows of one card's old files' PNG versions, oldest first."""
    from commands.png_changes import VISIBLE
    from commands.changelog import formatting_only
    versions = sorted((v for o in old_ids for v in data.versions.get(o, [])), key=lambda v: (v["ts"], v["path"]))
    pngs = [v for v in versions if v["ext"] == "png"]
    names = {norm(n) for n in today.get("names") or ()}
    rows, prev, run = [], None, 0
    for i, png in enumerate(pngs):
        psds = [v for v in versions if v["ext"] == "psd" and v["old_id"] == png["old_id"]]
        until = next((p["ts"] for p in pngs[i + 1:] if p["old_id"] == png["old_id"] and p["ts"] > png["ts"]), float("inf"))
        psd, pick, off = pick_psd(data, png, psds, until)
        fields, kind, two, note = check_fields(data, png, psd, pick, off, today)
        row = {"id": png["id"], "card": key, "old_id": png["old_id"], "date": png["date"], "ts": png["ts"],
               "author": data.authors.get(png["author"], png["author"]), "subject": png["subject"], "badge": BADGE,
               "link": None, "commit": png["commit"], "path": png["path"], "blob": png["blob"], "kind": png["kind"],
               "new_look": png["new_look"], "thumb": f"oldgit/thumbs/{png['blob']}.webp", "card_kind": kind,
               "fields": fields, "psd": note}
        name, stem = fields["name"]["value"], splitext(basename(png["path"]))[0].replace("_", " ")
        if name and names and norm(name) not in names:
            row["old_name"] = name                             # printed then: shown on the row, never an alias
        if names and norm(stem) not in names:
            row["old_file"] = stem                             # a renamed file (oldgit map's renames)
        if two:
            row["two_creatures"] = True
        if prev is None:
            row["minor"], row["data_change"], row["ability_change"] = False, [], "first"
        else:
            share = 0.0 if prev["blob"] == png["blob"] else data.shares.get((prev["blob"], png["blob"]))
            diff = changed(prev["fields"], fields)
            row["share"] = share
            row["data_change"] = diff
            row["minor"] = not diff and share is not None and share < VISIBLE
            a, b = (prev["fields"].get("ability") or {}).get("value"), (fields.get("ability") or {}).get("value")
            row["ability_change"] = "unknown" if a is None or b is None else "same" if a == b else \
                "formatting" if formatting_only(a, b) else "changed"
            run += row["ability_change"] == "changed"
        row["text_run"] = run                                  # one text, formatting aside: its last form, dated by its first
        rows.append(row)
        prev = row
    return rows


def unsure(row: dict) -> list:
    return [name for name, f in row["fields"].items() if f["confidence"] != "sure"]


def review_items(rows: list, kind: str = "field") -> list:
    """One item per row and field that is not sure."""
    items = []
    for row in rows:
        for name in unsure(row):
            f = row["fields"][name]
            items.append({"id": "oldrows-" + stable_id(kind, row["id"], name), "kind": f"{name} unsure", "card": row["card"],
                          "row": row["id"], "date": row["date"], "thumb": row["thumb"], "field": name,
                          "proposed": f.get("value"), "why": f.get("why"),
                          **{k: f[k] for k in ("reads", "creatures", "recall", "ocr") if k in f}})
    return items


def split_rows(data: Data, old_id: str, rows: list) -> dict:
    """STAGE1 Q3 (provisional A): each half's rows 'as <old name>', its half's stats; the halves matched to the
    live cards by their first versions' stats."""
    cards = SPLITS[old_id]
    first = {}
    for key in cards:
        ids = [o for o, c in data.map["cards"].items() if c.get("key") == key and c.get("confidence") == "sure"]
        for v in sorted((v for o in ids for v in data.versions.get(o, []) if v["ext"] == "psd"), key=lambda v: v["ts"]):
            stats = valid_stats(((data.psd.get((v["blob"], v["path"])) or {}).get("read") or {}).get("stats"))
            if stats:
                first[key] = stats
                break
    last = (rows[-1]["fields"].get("stats") or {}).get("creatures", {}).get("picture") or [None, None]
    dist = lambda a, b: sum(abs(a[s] - b[s]) for s in ORDER) if a and b and None not in a.values() else 99
    order = cards if dist(last[0], first.get(cards[0])) + dist(last[1], first.get(cards[1])) <= \
        dist(last[0], first.get(cards[1])) + dist(last[1], first.get(cards[0])) else cards[::-1]
    out = {}
    for half, key in enumerate(order):
        out[key] = [{**r, "card": key, "old_name": old_id, "split_from": old_id,
                     "fields": {**r["fields"], "stats": field(((r["fields"].get("stats") or {}).get("creatures") or {})
                                                              .get("picture", [None, None])[half], "picture", False,
                                                              why=f"{'top' if half == 0 else 'bottom'} half of {old_id}'s rows")}}
                    for r in rows]
    return {"halves": {k: ("top" if i == 0 else "bottom") for i, k in enumerate(order)}, "first_stats": first, "rows": out}


def rows_main():
    data = Data()
    data.load_today()
    log(f"{sum(len(v) for v in data.versions.values())} versions, {len(data.pictures)} pictures read, "
        f"{len(data.rereads)} rereads, {len(data.ocr)} OCR'd versions")
    live_rows, archived_rows, review = [], [], []
    for what, key, old_ids in groups(data):
        today = data.today.get(key) or {"names": {key or old_ids[0]}}
        rows = card_rows(data, key, old_ids, today if what in ("live", "unsure map") else {"names": {key or old_ids[0]}})
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
    for row in live_rows:                                      # the review items a row has
        row["review"] = [i["id"] for i in review if i.get("row") == row["id"]]
    os.makedirs(OUT, exist_ok=True)
    for name, items in (("rows.jsonl", live_rows), ("archived.jsonl", archived_rows), ("review.jsonl", review)):
        with open(join(OUT, name + ".tmp"), "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        os.replace(join(OUT, name + ".tmp"), join(OUT, name))
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
        with Image.open(BytesIO(oi.git(oi.OG, "cat-file", "blob", row["blob"]))) as im:
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
