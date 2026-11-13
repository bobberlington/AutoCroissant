"""
Card timelines' local preview (docs/Future/CARD_TIMELINE.md, steps 6 and 7; STAGE1 S9): the history data the website
reads, under the archive's derived/preview/history/ (never a repo; the layout of the history branch). Run from the
repo root with the fork Python (Pillow with WebP; check also needs numpy, cv2 and scripts/timeline_text.py's Vision CLI):

    python -m commands.discord_preview build [--root <dir>] [--old-git <dir>] [--workers <n>] [--force]
    python -m commands.discord_preview check [--root <dir>] [--workers <n>]

build   From the chain (derived/chain/timelines, created.json) and the Morgue candidates (derived/morgue/candidates.json):
          index.json             every card with a history (its folder, counts), the Morgue's new entries, the MDW
                                 Morgue cards' Discord pictures, and the Created dates the chain proposes (earlier or
                                 new only; "Before 2020" for a paper card, owner Q6)
          <card id>/timeline.json  its entries oldest first: paper, discord, first repo (owner Q5), text (owner Q4: data
                                 only) and old repo; each picture's version names its webp
          <card id>/<picture>.webp one per version, at most WEBP_W wide (never a full-size copy)
          _morgue/<slug>/        the same for each Morgue candidate (owner Q1, Q2) and each MDW Morgue card
        Only sure fields go in (value, method, confidence); review items, notes, posters, message ids and links never.
        Old-repo versions come from the old .git (read only, cat-file). Webps already there are kept (--force: made
        again); webps no entry names any more are deleted.
check   Every published version's fields read again from its webp (title OCR and its nearest name, the stars, the
        bars, the ability's words, the types' icons) against its timeline.json -> derived/preview/check.jsonl and
        check.txt. A field the readers can't read again is counted apart from a disagreement.
"""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from io import BytesIO
from json import loads
from os import environ, listdir, makedirs, remove, replace, rmdir, walk
from os.path import exists, expanduser, getsize, isdir, join
from re import sub
from subprocess import PIPE, Popen
from typing import Optional

from commands import discord_match as dm

# ========================
# Configuration
# ========================
OUT = ("derived", "preview", "history")
CHAIN = ("derived", "chain", "timelines")
CREATED = ("derived", "chain", "created.json")
MORGUE = ("derived", "morgue", "candidates.json")
MORGUE_DIR = ("derived", "morgue")
VERDICTS = ("derived", "match", "verdicts.jsonl")
MORGUE_FOLDER = "_morgue"                  # card ids never start with _ (build_site's history.json _shas)
PICTURE_KINDS = ("discord", "old repo", "paper")
TEXT_KINDS = ("text", "first repo")
BADGE = {"discord": "Discord", "old repo": "old repo", "paper": "paper card", "first repo": "first repo",
         "text": "changelog"}
WEBP_W, WEBP_Q = 720, 80                   # the website's card view width; quality
MAX_PIXELS = 40_000_000                    # a picture past this is shrunk on open (phone photos)
CHANGE_KEYS = ("name", "stars", "types", "stats", "bal", "buffs", "nerfs", "ability_change", "unknown")
FACT_KEYS = ("field", "from", "to", "by", "op", "unit")
PAPER_DAY = "Before 2020"                  # owner Q6 (discord_chain.PAPER_DAY)
ABOUT = ("Card timelines before the card repo (preview): per card its versions from Discord (date and channel), the old "
         "pre-wipe repo (date, mapped author, commit subject), paper cards (Before 2020), the first repo's push "
         "notices (text only) and changelog facts (data only). Only sure fields; no message text, posters or links.")


def log(text: str):
    print(f"[{datetime.now():%H:%M:%S}] {text}", flush=True)


# ========================
# Data
# ========================
def published(fields: Optional[dict]) -> dict:
    """The sure fields only: {field: {value, method, confidence}} (a review or unread field never goes in)."""
    out = {}
    for name, f in (fields or {}).items():
        if isinstance(f, dict) and f.get("confidence") == "sure" and "value" in f:
            out[name] = {"value": f["value"], "method": f.get("method"), "confidence": "sure"}
    return out


def changes_of(e: dict) -> dict:
    return {k: e["changes"][k] for k in CHANGE_KEYS if k in (e.get("changes") or {})}


def picture_name(e: dict) -> str:
    """A version's webp: its first picture's id (a Discord or paper post), or the old repo's blob."""
    if e["kind"] == "old repo":
        return f"O-{e['blob'][:12]}.webp"
    return sub(r"[^A-Za-z0-9_-]", "_", e["pictures"][0]["id"]) + ".webp"


def picture_job(e: dict, root: str) -> tuple:
    """(source kind, source, box) of a version's picture: a file of the archive (a sheet's cell box), or an old blob."""
    if e["kind"] == "old repo":
        return ("blob", e["blob"], None)
    pic = e["pictures"][0]
    return ("file", join(root, pic["file"]), tuple(pic["box"]) if pic.get("box") else None)


def entry_out(e: dict, image: Optional[str]) -> dict:
    """One timeline entry as published."""
    kind = e["kind"]
    out = {"id": e["id"], "kind": kind, "date": e.get("date"), "day": e.get("day"), "badge": BADGE[kind]}
    if kind == "paper":
        out.update(label=e.get("label") or PAPER_DAY, sort=e.get("sort"))
    if kind in ("discord", "text"):
        out["channel"] = e.get("channel")
    if kind in ("old repo", "first repo"):
        out.update(author=e.get("author"), subject=e.get("subject"))
    if e.get("as"):
        out["as"] = e["as"]
    if kind in PICTURE_KINDS:
        out.update(image=image, fields=published(e.get("fields")), changes=changes_of(e), minor=bool(e.get("minor")))
        if kind == "discord":
            out["posts"] = len(e.get("pictures") or ())
            if (e.get("look") or {}).get("git"):
                out["git_look"] = True      # the same look as a git version (posted before it)
            if e.get("era"):
                out["era"] = e["era"]
    if kind == "text":
        out["change"] = {k: v for k, v in (e.get("change") or {}).items() if k in FACT_KEYS + ("ability",)}
        if e.get("confirmed_by"):
            out["confirmed"] = True
    return out


def timeline_out(t: dict, created: Optional[dict]) -> tuple:
    """(the published timeline, [(webp name, picture job)])."""
    entries, jobs = [], []
    for e in t["entries"]:
        image = None
        if e["kind"] in PICTURE_KINDS:
            image = picture_name(e)
            jobs.append((image, e))
        entries.append(entry_out(e, image))
    stop = t.get("stop") or {}
    out = {"card": t["card"], "key": t["key"], "name": t["name"], "kind": t.get("kind"),
           "stop": {"day": stop.get("day"), "source": stop.get("source")} if stop else None, "entries": entries}
    first = t.get("git_first") or {}
    if first.get("date") and first.get("changes"):
        out["git_first"] = {"date": first["date"], "changes": changes_of(first)}   # its first recorded version's change
    if created:
        out["created"] = created
    return out, jobs


def created_proposals(created: dict) -> dict:
    """{card key: {date, sort, source, status, was}}: the chain's earlier or new Created dates (never later; hand
    entries are the chain's 'hand' status, never proposed)."""
    out = {}
    for key, c in sorted((created.get("cards") or {}).items()):
        if not c.get("proposed") or c.get("status") not in ("earlier", "new"):
            continue
        ours = c.get("ours") or {}
        out[key] = {"date": c["date"], "sort": c.get("sort") or c["date"], "source": ours.get("source"),
                    "status": c["status"], "was": (c.get("file") or {}).get("date")}
        if ours.get("as"):
            out[key]["as"] = ours["as"]
    return out


def morgue_out(entry: dict, files: dict, kind: str) -> tuple:
    """(a Morgue entry's published timeline, [(webp name, job)]): its pictures as versions, oldest first (a paper photo
    first, a printed card's photo undated)."""
    entries, jobs = [], []
    for p in entry.get("pictures") or ():
        if p.get("source") == "old repo":
            source = ("thumb", p["thumb"], None)
            kind_of = "old repo"
        else:
            row = files.get(p["id"])
            source = ("file", row[0], row[1]) if row else ("thumb", p["thumb"], None)
            kind_of = "paper" if p.get("paper") else "discord"
        name = sub(r"[^A-Za-z0-9_-]", "_", p["id"]) + ".webp"
        jobs.append((name, source))
        e = {"id": p["id"], "kind": kind_of, "date": p.get("date"),
             "day": (p.get("day") if p.get("paper") else (p.get("date") or "")[:10] or None),
             "badge": BADGE[kind_of], "image": name, "fields": {}, "changes": {}, "minor": False}
        if kind_of == "paper":
            e.update(date=None, day=None, label=PAPER_DAY, sort=p.get("sort") or "0000-00-00")
        elif kind_of == "discord":
            e["channel"] = p.get("channel")
            if p.get("printed"):
                e.update(date=None, day=None, printed=True)
        entries.append(e)
    entries.sort(key=lambda e: (e.get("sort") or e.get("date") or "9999"))
    first = entry.get("first") or {}
    out = {"card": entry.get("slug") or entry["id"], "key": entry["name"], "name": entry["name"], "morgue": kind,
           "first": {"date": first.get("date"), "day": first.get("day") or (first.get("date") or "")[:10] or None,
                     "channel": first.get("channel") if first.get("source") == "discord" else None,
                     "source": first.get("source")},
           "entries": entries}
    if entry.get("kind"):
        out["kind_of"] = entry["kind"]             # joke | played
    return out, jobs


# ========================
# Pictures
# ========================
class OldBlobs:
    """The old repo's blobs through one cat-file --batch (read only, no lazy fetch, fsmonitor off)."""
    def __init__(self, git_dir: str):
        self.git_dir, self.p = git_dir, None

    def get(self, sha: str) -> bytes:
        if self.p is None:
            self.p = Popen(["git", "--git-dir", self.git_dir, "-c", "core.fsmonitor=false", "--no-pager", "cat-file",
                            "--batch"], stdin=PIPE, stdout=PIPE, env={**environ, "GIT_NO_LAZY_FETCH": "1"})
        self.p.stdin.write(sha.encode() + b"\n")
        self.p.stdin.flush()
        head = self.p.stdout.readline().split()
        if len(head) < 3 or head[1] != b"blob":
            raise FileNotFoundError(sha)
        data = self.p.stdout.read(int(head[2]))
        self.p.stdout.read(1)
        return data


_BLOBS: Optional[OldBlobs] = None


def _init_worker(old_git: str):
    global _BLOBS
    _BLOBS = OldBlobs(old_git)


def open_picture(kind: str, source: str, box: Optional[tuple]):
    """A version's picture, upright, RGBA (a sheet's cell: its box of the trimmed sheet, trimmed, as the matcher cut it)."""
    from PIL import Image, ImageOps
    Image.MAX_IMAGE_PIXELS = None
    if box:
        img = dm.trim(dm.load_image(source))
        w, h = img.size
        return dm.trim(img.crop((round(box[0] * w), round(box[1] * h), round(box[2] * w), round(box[3] * h)))).convert("RGBA")
    data = _BLOBS.get(source) if kind == "blob" else None
    with Image.open(BytesIO(data) if data is not None else source) as im:
        if im.width * im.height > MAX_PIXELS:
            im.draft("RGB", (WEBP_W * 2, WEBP_W * 4))
        im = ImageOps.exif_transpose(im)
        return im.convert("RGBA")


def make_webp(job: tuple) -> tuple:
    """(path, width, height, error): one picture as a webp at most WEBP_W wide (no alpha kept when it is all opaque)."""
    from PIL import Image
    path, kind, source, box = job
    try:
        img = open_picture(kind, source, box)
        if img.width > WEBP_W:
            img = img.resize((WEBP_W, max(1, round(WEBP_W * img.height / img.width))), Image.LANCZOS)
        if img.getchannel("A").getextrema()[0] == 255:
            img = img.convert("RGB")
        img.save(path + ".tmp", "WEBP", quality=WEBP_Q, method=5)
        replace(path + ".tmp", path)
        return path, img.width, img.height, None
    except Exception as e:                       # one bad picture never stops the build: its version has no image
        return path, None, None, f"{type(e).__name__}: {str(e)[:160]}"


def picture_size(path: str) -> tuple:
    from PIL import Image
    with Image.open(path) as im:
        return im.size


# ========================
# build
# ========================
def build(root: str, old_git: str, workers: int = 3, force: bool = False):
    root = dm.check_root(root)
    old_git = expanduser(old_git)
    out_dir = join(root, *OUT)
    makedirs(out_dir, exist_ok=True)
    created = dm.read_json(join(root, *CREATED), {}) or {}
    by_key = (created.get("cards") or {})
    index = {"version": 1, "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "about": ABOUT,
             "cards": {}, "morgue": [], "mdw": [], "created": created_proposals(created)}
    want, jobs = {}, []                          # folder -> {webp names}; (path, kind, source, box)
    counts = Counter()
    chain_dir = join(root, *CHAIN)
    for name in sorted(listdir(chain_dir)):
        if not name.endswith(".json"):
            continue
        t = dm.read_json(join(chain_dir, name))
        if not t or not t.get("entries"):
            continue
        c = by_key.get(t["key"]) or {}
        mine = {"date": c.get("date"), "sort": c.get("sort"), "source": (c.get("ours") or {}).get("source"),
                "status": c.get("status")} if c else None
        out, pics = timeline_out(t, mine)
        folder = t["card"]
        want[folder] = set()
        for image, e in pics:
            want[folder].add(image)
            jobs.append((join(out_dir, folder, image), *picture_job(e, root)))
        counts.update(e["kind"] for e in out["entries"])
        index["cards"][folder] = {"key": t["key"], "name": t["name"], "dir": folder,
                                  "versions": sum(e["kind"] in PICTURE_KINDS for e in out["entries"]),
                                  "texts": sum(e["kind"] in TEXT_KINDS for e in out["entries"])}
        dm.write_json(join(out_dir, folder, "timeline.json"), out)
    morgue = dm.read_json(join(root, *MORGUE), {}) or {}
    files = {}
    for line in open(join(root, *VERDICTS)):
        r = loads(line)
        files[r["id"]] = (join(root, r["file"]), tuple(r["box"]) if r.get("box") else None)
    entries = [(e, "candidate") for e in morgue.get("candidates") or ()]
    entries += [(e, "mdw") for e in morgue.get("not_candidates") or () if e.get("status") == "morgue"]
    for entry, kind in entries:
        out, pics = morgue_out(entry, files, kind)
        if not out["entries"]:
            continue
        folder = f"{MORGUE_FOLDER}/{out['card']}"
        want[folder] = set()
        for image, (src_kind, source, box) in pics:
            want[folder].add(image)
            if src_kind == "thumb":
                jobs.append((join(out_dir, folder, image), "file", join(root, *MORGUE_DIR, source), None))
            else:
                jobs.append((join(out_dir, folder, image), src_kind, source, box))
        dm.write_json(join(out_dir, folder, "timeline.json"), out)
        row = {"dir": folder, "name": out["name"], "first": out["first"], "versions": len(out["entries"])}
        if kind == "candidate":
            index["morgue"].append({**row, "kind": entry.get("kind"), "id": entry["id"]})
        else:
            index["mdw"].append({**row, "card": entry.get("card") or entry["name"]})
    # webps: made where missing (or all with --force); none left that no entry names
    todo = [j for j in jobs if force or not exists(j[0])]
    for path, *_ in todo:
        makedirs(path.rsplit("/", 1)[0], exist_ok=True)
    failed = []
    log(f"build: {len(jobs)} pictures, {len(todo)} to make ({workers} workers)")
    if todo:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(old_git,)) as pool:
            for i, (path, w, h, err) in enumerate(pool.map(make_webp, todo, chunksize=8), 1):
                if err:
                    failed.append((path, err))
                if i % 250 == 0:
                    log(f"build: {i}/{len(todo)} pictures")
    gone = 0
    for folder in [d for d in listdir(out_dir) if isdir(join(out_dir, d))]:
        for sub_dir in ([join(folder, s) for s in listdir(join(out_dir, folder)) if isdir(join(out_dir, folder, s))]
                        if folder == MORGUE_FOLDER else [folder]):
            for f in listdir(join(out_dir, sub_dir)):
                if f.endswith(".webp") and f not in want.get(sub_dir, ()):
                    remove(join(out_dir, sub_dir, f))
                    gone += 1
            if sub_dir not in want and not listdir(join(out_dir, sub_dir)):
                rmdir(join(out_dir, sub_dir))
    # each published picture's size (the website sizes its <img> by it)
    sizes = {}
    for path, *_ in jobs:
        if exists(path):
            sizes[path] = picture_size(path)
    missing = 0
    for folder in want:
        path = join(out_dir, folder, "timeline.json")
        t = dm.read_json(path)
        for e in t["entries"]:
            if e.get("image"):
                size = sizes.get(join(out_dir, folder, e["image"]))
                if size:
                    e["w"], e["h"] = size
                else:
                    e["image"] = None
                    missing += 1
        dm.write_json(path, t)
    dm.write_json(join(out_dir, "index.json"), index)
    total = sum(getsize(join(d, f)) for d, _, fs in walk(out_dir) for f in fs)
    log(f"build: {len(index['cards'])} cards, entries {dict(counts)}; Morgue {len(index['morgue'])} candidates, "
        f"{len(index['mdw'])} MDW cards; Created proposals {len(index['created'])}; pictures {len(sizes)} "
        f"({len(failed)} failed, {missing} versions without one, {gone} old webps deleted); {total / 1e6:.1f} MB")
    for path, err in failed[:20]:
        log(f"build: failed {path[len(out_dir) + 1:]}: {err}")
    return index


def main(argv: Optional[list] = None) -> int:
    parser = ArgumentParser(prog="python -m commands.discord_preview", description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("build", "check"))
    parser.add_argument("--root", default=environ.get("DISCORD_ARCHIVE", dm.DEFAULT_ROOT))
    parser.add_argument("--old-git", default=dm.OLD_GIT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "build":
        build(args.root, args.old_git, args.workers, args.force)
    else:
        from commands.discord_preview_check import check
        check(args.root, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
