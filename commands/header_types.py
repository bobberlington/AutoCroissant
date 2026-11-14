"""
Discord pictures' types by the card's own git versions' header (docs/Future/CARD_TIMELINE.md, step 3 Read): on the
card template (eras D1-D3) a picture's type icons sit where its git version's do, so the header's icon band compared
with the band of each git version of the card whose types are known tells the types when they match. Icons a
version's PSD places outside the band (a two-line title: Warcraftian Priest) are compared in their own boxes too, and
for a version with no icons, the boxes where the card's other versions have them; a version with types whose icon
boxes are unknown is never a sure match.

    python -m commands.header_types calibrate [--root <dir>] [--old-git <dir>] [--n 400]
    python -m commands.header_types read      [--root <dir>]

calibrate  Old repo PNG versions (the old .git, read only) as Discord shows them (a 400 px JPEG, discord_match.degrade)
           against the matcher's thumbs: its own version (the same icons) and the card's other versions whose sure
           types differ (other icons; versions whose own bands are DATA_ONLY alike differ in the data alone: left out).
           The NCC bar: MARGIN over every pair with other icons -> derived/readers/types/calibration.json.
read       Each card-sure Discord picture (derived/match/verdicts.jsonl, not a sheet's cell) of era D1-D3 (the bar or
           text readers'): its band against each git version of its card with known types (old repo rows' sure types;
           the current version: the card data's) -> derived/readers/types/header.jsonl {picture id, types, version,
           ncc, boxes} sure when the best is over the bar, no version with other types is within MARGIN of it, its
           data says no 'unknown' and its icon boxes are known when it has types.
Nothing here talks to Discord; outputs stay under the archive's derived/.
"""
from argparse import ArgumentParser
from collections import defaultdict
from io import BytesIO
from json import loads
from os import environ, makedirs
from os.path import expanduser, join
from random import Random
from subprocess import PIPE, Popen
from sys import stderr
from typing import Optional

from commands import discord_match as dm
from global_config import DATA_BRANCH

# ========================
# Configuration
# ========================
OUT = ("derived", "readers", "types")
BAND = (0.0, 0.0, 0.26, 0.115)                      # x0, y0, x1, y1 of the card: the icon grid left of the title
SIDE = (60, 52)                                     # bands compared at this size
TEMPLATE_ERAS = ("D1", "D2", "D3")
MARGIN = 0.02                                       # the bar's lead over every other-types pair, and a match's over
                                                    # a version with other types
FLAT = 6.0                                          # a cell this even (std, 0-255) on both bands: no icon there
DATA_ONLY = 0.97                                    # two versions' bands this alike: their types differ in data only
PAD = -0.15                                         # an icon box shrunk by this share of its side each way (its core)
BOX_SIDE = (24, 24)                                 # icon boxes compared at this size
DETAILS_FILE = "db/psd_details.json"                # the data branch's PSD records (header type icon boxes)
JPEG = (400, 70)                                    # a Discord-like copy: width, quality (as era_bars)
SURE = ("version_sure", "version_unsure", "draft")


def log(text: str):
    print(text, file=stderr, flush=True)


def band(img):
    """The header's icon band, trimmed card, as a float array SIDE (RGB)."""
    return head(img)[0]


def head(img) -> tuple:
    """(the icon band as band() gives it, the trimmed card): what a picture is compared by."""
    import numpy as np
    from PIL import Image
    img = dm.trim(img.convert("RGB"))
    w, h = img.size
    crop = img.crop((round(BAND[0] * w), round(BAND[1] * h), round(BAND[2] * w), round(BAND[3] * h)))
    return np.asarray(crop.resize(SIDE, Image.LANCZOS), dtype=np.float32), img


def outside(box) -> bool:
    """An icon box (card fractions) whose centre is not in the band."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return not (BAND[0] <= cx <= BAND[2] and BAND[1] <= cy <= BAND[3])


def box_crop(img, box):
    """A trimmed card's icon box, PAD shrunk (a layer box holds its glow and the title's edge), as a float array BOX_SIDE."""
    import numpy as np
    from PIL import Image
    w, h = img.size
    px, py = (box[2] - box[0]) * PAD, (box[3] - box[1]) * PAD
    x0, y0 = max(0.0, box[0] - px) * w, max(0.0, box[1] - py) * h
    x1, y1 = min(1.0, box[2] + px) * w, min(1.0, box[3] + py) * h
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    crop = img.crop((round(x0), round(y0), round(x1), round(y1)))
    return np.asarray(crop.resize(BOX_SIDE, Image.LANCZOS), dtype=np.float32)


def score(mine: tuple, theirs: tuple, boxes: list) -> float:
    """ncc of the bands, and the lowest with each box outside the band (an icon there on one side only shows; a box
    even on both sides counts 1)."""
    scores = [ncc(mine[0], theirs[0])]
    for box in boxes:
        a, b = box_crop(mine[1], box), box_crop(theirs[1], box)
        if a is None or b is None:
            continue
        scores.append(_ncc(a, b) if max(float(a.std()), float(b.std())) >= FLAT else 1.0)
    return round(min(scores), 4)


def _ncc(a, b) -> float:
    import numpy as np
    out = []
    for c in range(3):
        x, y = a[..., c].ravel() - a[..., c].mean(), b[..., c].ravel() - b[..., c].mean()
        d = float(np.sqrt((x * x).sum() * (y * y).sum()))
        out.append(float((x * y).sum()) / d if d else 0.0)
    return sum(out) / 3


def ncc(a, b) -> float:
    """The bands' normalized cross-correlation (each channel's, averaged), the lowest of the whole band's and of each
    icon cell's (a 2 x 2 grid: one icon missing or changed shows there); a cell flat on both sides (FLAT) counts 1."""
    h, w = a.shape[:2]
    scores = [_ncc(a, b)]
    for y0, y1 in ((0, h // 2), (h // 2, h)):
        for x0, x1 in ((0, w // 2), (w // 2, w)):
            ca, cb = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
            if max(float(ca.std()), float(cb.std())) >= FLAT:
                scores.append(_ncc(ca, cb))
    return round(min(scores), 4)


def thumb_band(root: str, version: dict):
    return thumb_head(root, version)[0]


def thumb_head(root: str, version: dict) -> tuple:
    from PIL import Image
    with Image.open(join(root, *dm.OUT, version["thumb"])) as im:
        return head(im)


def icon_boxes(root: str, cards_repo: str, cards: dict, versions: list) -> dict:
    """{version key: [icon boxes, card fractions]} where the version's PSD is known: the old repo's (oldrows' PSD,
    its header icon layers in icons.jsonl, its size in oldgit reads.jsonl), the current version's (the card data's
    psd_details record: header.types.row; none there: no icons)."""
    psd_of = {"b" + r["blob"][:20]: (r.get("psd") or {}).get("blob")
              for r in dm.read_jsonl(join(root, "derived", "oldrows", "rows.jsonl")) if r.get("blob")}
    icons = {r["blob"]: r["icons"] for r in dm.read_jsonl(join(root, "derived", "oldrows", "icons.jsonl"))
             if "error" not in r}
    size = {r["blob"]: r.get("size") for r in dm.read_jsonl(join(root, "derived", "oldgit", "reads.jsonl"))
            if r.get("kind") == "psd" and r.get("size")}
    git_dir = dm.git_dir_of(cards_repo)
    stats = (loads(dm.git_out(git_dir, "show", f"origin/{DATA_BRANCH}:{dm.STATS_FILE}") or b"{}") or {}).get(
        "cards") or {}
    details = (loads(dm.git_out(git_dir, "show", f"origin/{DATA_BRANCH}:{DETAILS_FILE}") or b"{}") or {}).get(
        "cards") or {}
    clip = lambda x: min(1.0, max(0.0, x))
    out = {}
    for v in versions:
        if v.get("old"):
            blob = psd_of.get(v["key"])
            if blob in icons and size.get(blob):
                w, h = size[blob]
                out[v["key"]] = [[clip(b[0] / w), clip(b[1] / h), clip(b[2] / w), clip(b[3] / h)]
                                 for b in (i["bbox"] for i in icons[blob])]
        elif v.get("current"):
            rec = details.get((stats.get((cards.get(v["card"]) or {}).get("key")) or {}).get("blob_sha"))
            if rec and rec.get("canvas"):
                w, h = rec["canvas"]["w"], rec["canvas"]["h"]
                row = ((rec.get("header") or {}).get("types") or {}).get("row") or []
                out[v["key"]] = [[clip(b[0] / w), clip(b[1] / h), clip(b[2] / w), clip(b[3] / h)]
                                 for b in (i["box"] for i in row)]
    return out


def card_slots(versions: list, boxes: dict) -> dict:
    """{card id: the boxes outside the band any version of the card has an icon in} (near-equal boxes once)."""
    out = defaultdict(list)
    for v in versions:
        for b in boxes.get(v["key"]) or ():
            if outside(b) and all(max(abs(p - q) for p, q in zip(b, c)) > 0.02 for c in out[v["card"]]):
                out[v["card"]].append(b)
    return out


def compared(key: str, card: str, types: dict, boxes: dict, slots: dict) -> Optional[list]:
    """The boxes outside the band a version is compared in: its own icons' (None: it has types and no known icon
    boxes, never a sure match); for a version with no icons, the card's slots (card_slots)."""
    if types.get(key):
        mine = boxes.get(key)
        return [b for b in mine if outside(b)] if mine else None
    return list(slots.get(card) or ())


def known_types(root: str, cards_repo: str, cards: dict, versions: list) -> dict:
    """{version key: sorted types} for git versions whose types are known: old repo rows' sure types, the current
    version's from the card data (parser 2 on)."""
    found = {}
    for r in dm.read_jsonl(join(root, "derived", "oldrows", "rows.jsonl")):
        t = (r.get("fields") or {}).get("types") or {}
        if t.get("confidence") == "sure" and isinstance(t.get("value"), list) and r.get("blob"):
            found["b" + r["blob"][:20]] = sorted(t["value"])
    git_dir = dm.git_dir_of(cards_repo)
    stats = (loads(dm.git_out(git_dir, "show", f"origin/{DATA_BRANCH}:{dm.STATS_FILE}") or b"{}") or {}).get(
        "cards") or {}
    for v in versions:
        entry = stats.get((cards.get(v["card"]) or {}).get("key"))
        if v.get("current") and v["key"] not in found and entry and entry.get("parser_version") is not None:
            found[v["key"]] = sorted(entry.get("types") or [])
    return found


class Blobs:
    """The old .git's blobs, read only (one cat-file --batch)."""
    def __init__(self, git_dir: str):
        self.proc = Popen(["git", "--git-dir", git_dir, "-c", "core.fsmonitor=false", "--no-pager", "cat-file",
                           "--batch"], stdin=PIPE, stdout=PIPE)

    def get(self, sha: str) -> bytes:
        self.proc.stdin.write(sha.encode() + b"\n")
        self.proc.stdin.flush()
        head = self.proc.stdout.readline().split()
        if len(head) < 3 or head[1] != b"blob":
            raise KeyError(sha)
        data = self.proc.stdout.read(int(head[2]))
        self.proc.stdout.read(1)
        return data

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def calibrate(root: str, old_git: str, n: int = 400, seed: int = 1113) -> dict:
    from PIL import Image
    pairs = dm.read_json(dm.Paths(root).pairs, {}) or {}
    cards, versions = pairs.get("cards") or {}, pairs.get("versions") or []
    types = known_types(root, dm.LOCAL_DIR_LOC, cards, versions)
    by_card = defaultdict(list)
    for v in versions:
        if v["key"] in types and v.get("thumb") and v.get("shape") == "1:2":
            by_card[v["card"]].append(v)
    olds = [v for vs in by_card.values() for v in vs if v.get("old")]
    rng = Random(seed)
    sample = rng.sample(olds, min(n, len(olds)))
    blob_of = {"b" + r["blob"][:20]: r["blob"] for r in dm.read_jsonl(join(root, "derived", "oldrows", "rows.jsonl"))
               if r.get("blob")}
    boxes = icon_boxes(root, dm.LOCAL_DIR_LOC, cards, versions)
    slots = card_slots(versions, boxes)
    blobs = Blobs(expanduser(old_git))
    same, other, quirks = [], [], 0
    try:
        for v in sample:
            if v["key"] not in blob_of:
                continue
            with Image.open(BytesIO(blobs.get(blob_of[v["key"]]))) as im:
                copy = dm.degrade(im.convert("RGB"), *JPEG)
            mine = head(copy)
            own = compared(v["key"], v["card"], types, boxes, slots)
            same.append(score(mine, thumb_head(root, v), own or []))
            for w in by_card[v["card"]]:
                if w["key"] != v["key"] and types[w["key"]] != types[v["key"]]:
                    if ncc(thumb_band(root, v), thumb_band(root, w)) >= DATA_ONLY:
                        quirks += 1             # the bands alike: types differ in the data, not the icons
                        continue
                    other.append(score(mine, thumb_head(root, w),
                                       compared(w["key"], w["card"], types, boxes, slots) or []))
    finally:
        blobs.close()
    top = max(other, default=0.0)
    bar = round(top + MARGIN, 4)
    cal = {"about": "Header band NCC (and icon boxes outside the band): an old repo PNG's 400 px JPEG copy against the "
                    "matcher's thumbs; same: its own version; other: the card's versions with other sure types. bar = "
                    "the highest other + MARGIN.",
           "band": BAND, "side": SIDE, "box_side": BOX_SIDE, "pad": PAD, "margin": MARGIN, "same": len(same), "other": len(other),
           "other_left_out_data_only": quirks, "bar": bar,
           "same_over_bar": round(sum(s >= bar for s in same) / len(same), 4) if same else None,
           "same_quantiles": quantiles(same), "other_quantiles": quantiles(other)}
    makedirs(join(root, *OUT), exist_ok=True)
    dm.write_json(join(root, *OUT, "calibration.json"), cal)
    log(dm.dumps(cal, indent=1))
    return cal


def quantiles(xs: list) -> list:
    xs = sorted(xs)
    return [round(xs[min(len(xs) - 1, int(q * len(xs)))], 4) for q in (0, 0.05, 0.25, 0.5, 0.75, 0.95)] + \
        [round(xs[-1], 4)] if xs else []


def read(root: str, cards_repo: str = dm.LOCAL_DIR_LOC) -> list:
    pairs = dm.read_json(dm.Paths(root).pairs, {}) or {}
    cards, versions = pairs.get("cards") or {}, pairs.get("versions") or []
    cal = dm.read_json(join(root, *OUT, "calibration.json"), {}) or {}
    if not cal.get("bar"):
        raise SystemExit("no calibration: run python -m commands.header_types calibrate first")
    types = known_types(root, cards_repo, cards, versions)
    boxes = icon_boxes(root, cards_repo, cards, versions)
    slots = card_slots(versions, boxes)
    by_card = defaultdict(list)
    for v in versions:
        if v["key"] in types and v.get("thumb"):
            by_card[v["card"]].append(v)
    eras = {}
    for r in dm.read_jsonl(join(root, "derived", "readers", "text", "reads.jsonl")):
        if r.get("src") == "discord":
            eras[r["file"]] = r.get("era")
    for r in dm.read_jsonl(join(root, "derived", "readers", "bars", "reads.jsonl")):
        if r.get("src") == "discord" and r.get("era"):
            eras[r["file"]] = r["era"]
    bands, out = {}, []
    for row in dm.read_jsonl(join(root, "derived", "match", "verdicts.jsonl")):
        if row["verdict"] not in SURE or row.get("cell") or eras.get(row["file"]) not in TEMPLATE_ERAS or \
                not by_card.get(row.get("card")):
            continue
        try:
            mine = head(dm.load_image(join(root, row["file"])))
        except Exception:
            continue
        scores = []
        for v in by_card[row["card"]]:
            if v["key"] not in bands:
                bands[v["key"]] = thumb_head(root, v)
            own = compared(v["key"], v["card"], types, boxes, slots)
            scores.append((score(mine, bands[v["key"]], own or []), v["key"], own))
        scores.sort(key=lambda t: t[:2], reverse=True)
        best, key, own = scores[0]
        rival = max((s for s, k, _ in scores if types[k] != types[key]), default=-1.0)
        found = {"id": row["id"], "card": row["card"], "era": eras.get(row["file"]), "ncc": best, "version": key,
                 "types": types[key], "rival": round(rival, 4), "boxes": None if own is None else len(own)}
        found["sure"] = best >= cal["bar"] and best - rival >= MARGIN and "unknown" not in types[key] and \
            own is not None
        out.append(found)
    dm.write_text(join(root, *OUT, "header.jsonl"), "".join(dm.dumps(r) + "\n" for r in out))
    log(f"read: {len(out)} pictures, {sum(r['sure'] for r in out)} over the bar {cal['bar']}")
    return out


def main(argv: Optional[list] = None) -> int:
    ap = ArgumentParser(prog="python -m commands.header_types", description="Types by the git versions' header.")
    ap.add_argument("step", choices=("calibrate", "read"))
    ap.add_argument("--root", default=environ.get("DISCORD_ARCHIVE", dm.DEFAULT_ROOT))
    ap.add_argument("--old-git", default=dm.OLD_GIT)
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args(argv)
    root = dm.check_root(a.root)
    if a.step == "calibrate":
        calibrate(root, a.old_git, a.n)
    else:
        read(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
