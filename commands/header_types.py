"""
Discord pictures' types by the card's own git versions' header (docs/Future/CARD_TIMELINE.md, step 3 Read): on the
card template (eras D1-D3) a picture's type icons sit where its git version's do, so the header's icon band compared
with the band of each git version of the card whose types are known tells the types when they match.

    python -m commands.header_types calibrate [--root <dir>] [--old-git <dir>] [--n 400]
    python -m commands.header_types read      [--root <dir>]

calibrate  Old repo PNG versions (the old .git, read only) as Discord shows them (a 400 px JPEG, discord_match.degrade)
           against the matcher's thumbs: its own version (the same icons) and the card's other versions whose sure
           types differ (other icons; versions whose own bands are DATA_ONLY alike differ in the data alone: left out).
           The NCC bar: MARGIN over every pair with other icons -> derived/readers/types/calibration.json.
read       Each card-sure Discord picture (derived/match/verdicts.jsonl, not a sheet's cell) of era D1-D3 (the bar or
           text readers'): its band against each git version of its card with known types (old repo rows' sure types;
           the current version: the card data's) -> derived/readers/types/header.jsonl {picture id, types, version,
           ncc} when the best is over the bar and no version with other types is within MARGIN of it (never a
           version whose data says 'unknown').
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
JPEG = (400, 70)                                    # a Discord-like copy: width, quality (as era_bars)
SURE = ("version_sure", "version_unsure", "draft")


def log(text: str):
    print(text, file=stderr, flush=True)


def band(img):
    """The header's icon band, trimmed card, as a float array SIDE (RGB)."""
    import numpy as np
    from PIL import Image
    img = dm.trim(img.convert("RGB"))
    w, h = img.size
    crop = img.crop((round(BAND[0] * w), round(BAND[1] * h), round(BAND[2] * w), round(BAND[3] * h)))
    return np.asarray(crop.resize(SIDE, Image.LANCZOS), dtype=np.float32)


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
    from PIL import Image
    with Image.open(join(root, *dm.OUT, version["thumb"])) as im:
        return band(im)


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
    blobs = Blobs(expanduser(old_git))
    same, other, quirks = [], [], 0
    try:
        for v in sample:
            if v["key"] not in blob_of:
                continue
            with Image.open(BytesIO(blobs.get(blob_of[v["key"]]))) as im:
                copy = dm.degrade(im.convert("RGB"), *JPEG)
            mine = band(copy)
            same.append(ncc(mine, thumb_band(root, v)))
            for w in by_card[v["card"]]:
                if w["key"] != v["key"] and types[w["key"]] != types[v["key"]]:
                    if ncc(thumb_band(root, v), thumb_band(root, w)) >= DATA_ONLY:
                        quirks += 1             # the bands alike: types differ in the data, not the icons
                        continue
                    other.append(ncc(mine, thumb_band(root, w)))
    finally:
        blobs.close()
    top = max(other, default=0.0)
    bar = round(top + MARGIN, 4)
    cal = {"about": "Header band NCC: an old repo PNG's 400 px JPEG copy against the matcher's thumbs; same: its own "
                    "version; other: the card's versions with other sure types. bar = the highest other + MARGIN.",
           "band": BAND, "side": SIDE, "margin": MARGIN, "same": len(same), "other": len(other),
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
            mine = band(dm.load_image(join(root, row["file"])))
        except Exception:
            continue
        scores = []
        for v in by_card[row["card"]]:
            if v["key"] not in bands:
                bands[v["key"]] = thumb_band(root, v)
            scores.append((ncc(mine, bands[v["key"]]), v["key"]))
        scores.sort(reverse=True)
        best, key = scores[0]
        rival = max((s for s, k in scores if types[k] != types[key]), default=-1.0)
        found = {"id": row["id"], "card": row["card"], "era": eras.get(row["file"]), "ncc": best, "version": key,
                 "types": types[key], "rival": round(rival, 4)}
        found["sure"] = best >= cal["bar"] and best - rival >= MARGIN and "unknown" not in types[key]
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
