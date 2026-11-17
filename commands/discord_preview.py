"""
Card timelines' local preview (docs/Future/CARD_TIMELINE.md, steps 6 and 7; STAGE1 S9): the history data the website
reads, under the archive's derived/preview/history/ (never a repo; the layout of the history branch). Run from the
repo root with the fork Python (Pillow with WebP; check also needs numpy, cv2 and scripts/timeline_text.py's Vision
CLI):

    python -m commands.discord_preview build [--root <dir>] [--old-git <dir>] [--cards-repo <dir>] [--workers <n>] [--force]
    python -m commands.discord_preview check [--root <dir>] [--workers <n>] [--redo]

build   From the chain (derived/chain/timelines, created.json) and the Morgue candidates (derived/morgue/
        candidates.json, with the review page's answers: derived/review/rulings.json):
          index.json             every card with a history (its folder, counts), the Morgue's new entries, the MDW
                                 Morgue cards' Discord pictures, and the Created dates the chain proposes (earlier or
                                 new only; "Before 2020" from a physical card photo for the three the owner named, Q6)
          <card id>/timeline.json  its entries oldest first: discord, text (owner Q4: data only) and old repo
                                 (the first repo's badged 'first repo', owner Q5); each picture's version names its webp
          <card id>/<picture>.webp one per version, at most WEBP_W wide (never a full-size copy)
          _morgue/<slug>/        the same for each Morgue candidate (owner Q1, Q2) and each MDW Morgue card
        Only sure fields go in (value, method, confidence); review items, notes, posters, message ids and links never.
        A title snapped to an old PSD layer's casing takes today's name when the picture's own OCR spells it so
        (printed_name). Stop at git: a card's entries end at its first record in the card repo (--cards-repo, read
        only: its PNG's moves followed as the website follows them, lineage); its git_first is dated when its card-data
        version was committed, approx when the card had versions before it; an MDW Morgue card's Discord pictures end
        at its first record in the old repo (oldgit map.json) or the card repo (mdw_firsts). A confirmed changelog fact
        names the version that confirms it (confirmed_by). The eye's leave_out (derived/preview/eye.json: pictures
        that are no version of the card, e.g. AI restyles) are left out, the next version diffed again (leave_out).
        Old-repo versions come from the old .git (read only, cat-file). Webps already there are kept (--force: made
        again); webps and timelines no entry names any more are deleted.
check   Every published version's fields read again from its webp (title OCR and its nearest name, the stars, the
        bars, the ability's words, the types' icons) against its timeline.json -> derived/preview/check.jsonl and
        check.txt. A field the readers can't read again is counted apart from a disagreement.
"""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from hashlib import sha1
from io import BytesIO
from json import loads
from os import environ, listdir, makedirs, remove, replace, rmdir, walk
from os.path import basename, exists, expanduser, getsize, isdir, join, splitext
from re import sub
from subprocess import PIPE, Popen
from typing import Optional

from commands import discord_chain as dc
from commands import discord_match as dm
from commands.card_history import LOG_FORMAT, NO_BLOB
from commands.card_models import rename_trailer
from global_config import CARD_BRANCH, LOCAL_DIR_LOC

# ========================
# Configuration
# ========================
OUT = ("derived", "preview", "history")
CHAIN = ("derived", "chain", "timelines")
CREATED = ("derived", "chain", "created.json")
MORGUE = ("derived", "morgue", "candidates.json")
MORGUE_DIR = ("derived", "morgue")
IDEAS = ("derived", "morgue", "ideas.json")
RULINGS = ("derived", "review", "rulings.json")             # the review page's answers (the Morgue list's)
FACTS = ("derived", "facts", "facts.jsonl")                  # a Morgue entry's changelog marks (card 'morgue:<id>')
EYE = ("derived", "preview", "eye.json")     # the check's eye rulings: {"drop": {"<card>|<entry id>|<field>": why},
#                                             "leave_out": {"<card>|<entry id>": why}: no version of the card}
VERDICTS = ("derived", "match", "verdicts.jsonl")
TEXT_READS = ("derived", "readers", "text", "reads.jsonl")   # the title reads' own OCR (printed_name)
OLD_MAP = ("derived", "oldgit", "map.json")                  # old cards -> today's (an MDW card's first old record)
OLD_ROWS = ("derived", "oldrows")             # archived.jsonl (MDW Morgue cards' rows), review.jsonl (lost, morgue)
OLD_VERSIONS = ("derived", "oldgit", "versions.jsonl")       # a first-repo file's versions (Morgue picture rows)
AUTHORS = join(expanduser("~"), "Documents/Tools/wt/scratch/s1110/timelines/authors.json")   # old commits' authors
MDW_FOLDER = "MDW/The Morgue/"
BEFORE_GIT = ("discord", "text")                             # dated entries that must come before the card's first git record
MORGUE_FOLDER = "_morgue"                  # card ids never start with _ (build_site's history.json _shas)
PICTURE_KINDS = ("discord", "old repo")
BADGE = {"discord": "Discord", "old repo": "old repo", "first repo": "first repo",
         "text": "changelog"}                       # first repo: an old repo row's (repo 'first')
WEBP_W, WEBP_Q = 720, 80                   # the website's card view width; quality
MAX_PIXELS = 40_000_000                    # a picture past this is shrunk on open (phone photos)
KINDS_OF = {"name": str, "stars": int, "types": list, "stats": dict, "ability": str}   # another kind: left out
CHANGE_KEYS = ("name", "stars", "types", "stats", "bal", "buffs", "nerfs", "ability_change", "unknown")
FACT_KEYS = ("field", "from", "to", "by", "op", "unit")
ABOUT = ("Card timelines before the card repo (preview): per card its versions from Discord (date and channel), the "
         "old pre-wipe repo and the first repo (date, mapped author, commit subject) and changelog facts (data "
         "only). Only sure fields; no message text, posters or links.")


def log(text: str):
    print(f"[{datetime.now():%H:%M:%S}] {text}", flush=True)


# ========================
# Data
# ========================
def published(fields: Optional[dict], drop=()) -> dict:
    """The sure fields only: {field: {value, method, confidence}} (a review or unread field never goes in, nor a value
    of the wrong kind, nor one the eye ruled wrong: drop)."""
    out = {}
    for name, f in (fields or {}).items():
        if isinstance(f, dict) and f.get("confidence") == "sure" and "value" in f and name not in drop \
                and isinstance(f["value"], KINDS_OF.get(name, object)) and not isinstance(f["value"], bool):
            out[name] = {"value": f["value"], "method": f.get("method"), "confidence": "sure"}
            if f.get("half"):                      # a split card's half (owner Q3): which rows it took
                out[name]["half"] = f["half"]
    return out


def changes_of(e: dict) -> dict:
    return {k: e["changes"][k] for k in CHANGE_KEYS if k in (e.get("changes") or {})}


def picture_name(e: dict) -> str:
    """A version's webp: its first picture's id (a Discord post; a crop by eye: its box too, so a new box makes it
    again), or the old repo's blob."""
    if e["kind"] == "old repo":
        return f"O-{e['blob'][:12]}.webp"
    p = e["pictures"][0]
    crop = "-k" + sha1(str(p["box"]).encode()).hexdigest()[:6] if p.get("crop") and p.get("box") else ""
    return sub(r"[^A-Za-z0-9_-]", "_", p["id"]) + crop + ".webp"


def picture_job(e: dict, root: str) -> tuple:
    """(source kind, source, box) of a version's picture: a file of the archive (a sheet's cell box), or an old blob
    (a PSD added before any PNG: its stored composite, oldgit's thumb)."""
    if e["kind"] == "old repo" and e.get("psd_only"):
        return ("file", join(root, "derived", e["thumb"]), None)
    if e["kind"] == "old repo":
        return ("blob", e["blob"], None)
    pic = e["pictures"][0]
    return ("file", join(root, pic["file"]), tuple(pic["box"]) if pic.get("box") else None)


def entry_out(e: dict, image: Optional[str], drop=()) -> dict:
    """One timeline entry as published."""
    kind = e["kind"]
    out = {"id": e["id"], "kind": kind, "date": e.get("date"), "day": e.get("day"), "badge": BADGE[kind]}
    if kind in ("discord", "text"):
        out["channel"] = e.get("channel")
    if kind == "old repo":
        out.update(author=e.get("author"), subject=e.get("subject"))
        if e.get("old_file"):
            out["file"] = e["old_file"]                       # a renamed or MDW draft file: its own name shown
    if kind == "old repo" and e.get("repo") == "first":
        out.update(badge=BADGE["first repo"], repo="first")      # the first repo's file (2022-08..2023-04)
    if e.get("as"):
        out["as"] = e["as"]
    if kind in PICTURE_KINDS:
        out.update(image=image, fields=published(e.get("fields"), drop), changes=changes_of(e),
                   minor=bool(e.get("minor")))
        if kind == "discord":
            out["posts"] = len(e.get("pictures") or ())
            if (e.get("look") or {}).get("git"):
                out["git_look"] = True      # the same look as a git version (posted before it)
            if e.get("era"):
                out["era"] = e["era"]
    if kind == "text":
        out["change"] = {k: v for k, v in (e.get("change") or {}).items() if k in FACT_KEYS + ("ability",)}
        if e.get("confirmed_by"):
            out.update(confirmed=True, confirmed_by=e["confirmed_by"])   # the version whose picture agrees with it
    return out


def printed_name(value, files: list, reads: dict, today: str):
    """A title read snapped to a known name (ocr+name) whose nearest name was an old PSD layer's casing ('MEtal
    General') takes today's name when it is today's name but for case and one of the version's pictures' own title
    OCR spells it that way (letters only: the picture prints 'Metal General'); otherwise it stays as read (a picture
    printing 'DooPliss' or 'FAWFUL' keeps it)."""
    letters = lambda text: "".join(ch for ch in text or "" if ch.isalnum())
    if not isinstance(value, str) or not today or value == today or value.casefold() != today.casefold():
        return value
    want = letters(today)
    for f in files:
        got = letters(reads.get(f))
        at = got.casefold().find(want.casefold())
        if at >= 0 and got[at:at + len(want)] == want:
            return today
    return value


CHANGE_OF = {"stats": ("stats", "bal", "buffs", "nerfs"), "stars": ("stars",), "types": ("types",), "name": ("name",)}


def timeline_out(t: dict, created: Optional[dict], eye: Optional[dict] = None, reads: Optional[dict] = None) -> tuple:
    """(the published timeline, [(webp name, picture job)]). eye: the eye's drops for this card {entry id: {field}}:
    the field is left out, and so are the changes it made (its own against the version before, the next version's
    against it). reads: the title OCR by file (printed_name)."""
    entries, jobs = [], []
    for e in t["entries"]:
        image = None
        if e["kind"] in PICTURE_KINDS:
            image = picture_name(e)
            jobs.append((image, e))
        out = entry_out(e, image, (eye or {}).get(e["id"], ()))
        name = (out.get("fields") or {}).get("name")
        if e["kind"] == "discord" and name and str(name["method"]).startswith("ocr+name"):
            name["value"] = printed_name(name["value"], [p["file"] for p in e.get("pictures") or () if not p.get("box")],
                                         reads or {}, t.get("name"))
        entries.append(out)
    pictures = [e for e in entries if e["kind"] in PICTURE_KINDS]
    for i, e in enumerate(pictures):
        for field in (eye or {}).get(e["id"], ()):
            for other in pictures[i:i + 2]:
                ch = other["changes"]
                for key in CHANGE_OF.get(field, ()):
                    ch.pop(key, None)
                if field == "ability" and ch.get("ability_change") in ("changed", "same", "formatting"):
                    ch["ability_change"] = "unknown"
    stop = t.get("stop") or {}
    out = {"card": t["card"], "key": t["key"], "name": t["name"], "kind": t.get("kind"),
           "stop": {"day": stop.get("day"), "source": stop.get("source")} if stop else None, "entries": entries}
    first = t.get("git_first") or {}
    if first.get("date") and first.get("changes"):
        out["git_first"] = {"date": first["date"], "changes": changes_of(first)}   # its first recorded version's change
        if first.get("approx"):
            out["git_first"]["approx"] = True     # the card had versions before that one: its change is in one of them
        for field in (eye or {}).get(pictures[-1]["id"] if pictures else None, ()):
            for key in CHANGE_OF.get(field, ()):
                out["git_first"]["changes"].pop(key, None)
    if created:
        out["created"] = created
    return out, jobs


def leave_out(t: dict, ids, versions: list = (), root: Optional[str] = None) -> list:
    """A chain timeline without the entries the eye left out (ids: pictures that are no version of the card): the version
    after them is diffed again against the one before them (as the chain's diff; its fold measured from root's
    pictures), git_first too when the last ones went, and a fact they confirmed loses confirmed_by. Returns the entries
    taken out."""
    gone = [e for e in t["entries"] if e["id"] in ids]
    if not gone:
        return []
    kept, before, redo = [], None, False
    for e in t["entries"]:
        if e["id"] in ids:
            redo = True
            continue
        kept.append(e)
        if e["kind"] == "text":
            if e.get("confirmed_by") in ids:
                e.pop("confirmed_by")
            continue
        if redo:
            e["changes"] = dc.compare(before["fields"], e["fields"], t.get("kind")) if before else \
                ({"ability_change": "first"} if e["fields"].get("ability") else {})
            if e["kind"] == "discord":
                e["minor"], share = dc.fold(root, before, e)
                e.pop("share", None)
                if share is not None:
                    e["share"] = round(share, 4)
            redo = False
        before = e
    if redo and t.get("git_first"):
        t["git_first"]["changes"] = git_first_changes(t, before, versions)
    t["entries"] = kept
    return gone


def created_proposals(created: dict) -> dict:
    """{card key: {date, sort, source, status, was}}: the chain's earlier or new Created dates, and the review page's
    picks (status 'owner': its source, 'owner' for a typed date) when earlier than the website's (never later; hand
    entries are the chain's 'hand' status, never proposed)."""
    out = {}
    for key, c in sorted((created.get("cards") or {}).items()):
        if not c.get("proposed") or c.get("status") not in ("earlier", "new", "owner"):
            continue
        was = (c.get("file") or {}).get("date")
        ours = (c.get("owner") or {}).get("source") or {} if c["status"] == "owner" else c.get("ours") or {}
        if c["status"] == "owner" and (ours.get("source") == "website" or was and (c.get("sort") or c["date"]) >= was):
            continue                               # the website's date kept, or a pick later than it
        source = ours.get("source")
        out[key] = {"date": c["date"], "sort": c.get("sort") or c["date"],
                    "source": "old repo" if source == "old repo (PSD)" else source,
                    "status": c["status"], "was": (c.get("file") or {}).get("date")}
        if source == "old repo (PSD)" or ours.get("psd"):
            out[key]["psd"] = True                 # a PSD (old or first repo), added before any PNG of the card
        if ours.get("as"):
            out[key]["as"] = ours["as"]
    return out


def morgue_entries(morgue: dict, ideas: list, rulings: dict) -> list:
    """[(entry, 'candidate' | 'mdw')]: the Morgue list with the review page's answers (timeline_review.morgue_list):
    the candidates not answered 'no', the ideas and the rest answered 'yes'; the MDW Morgue cards' Discord pictures."""
    said = lambda e: (rulings.get(f"morgue:{e['id']}") or {}).get("choice")
    out = [(e, "candidate") for e in morgue.get("candidates") or () if said(e) != "no"]
    out += [(e, "candidate") for e in [*ideas, *(e for e in morgue.get("not_candidates") or ()
                                                if e.get("status") != "morgue")] if said(e) == "yes"]
    return out + [(e, "mdw") for e in morgue.get("not_candidates") or () if e.get("status") == "morgue"]


def morgue_marks(facts: list, entry: dict) -> list:
    """A Morgue entry's changelog marks (owner Q4: data only): the sure changelog facts given to it ('morgue:<id>',
    e.g. the owner's D9: 2019 'Computer virus 2.0', never the live Computer Virus)."""
    out = []
    for f in facts:
        if f.get("card") != f"morgue:{entry['id']}" or f.get("source") != "changelog" or f.get("confidence") != "sure":
            continue
        change = {"ability": "changed"} if f.get("kind") == "ability changed" else \
            {k: f[k] for k in FACT_KEYS + ("stated",) if f.get(k) is not None}
        out.append({"id": "F-" + f["id"], "kind": "text", "date": f.get("date"), "day": (f.get("date") or "")[:10],
                    "badge": BADGE["text"], "channel": f.get("channel"),
                    "change": {k: v for k, v in change.items() if k in FACT_KEYS + ("ability",)}})
    return out


def morgue_rows(root: str) -> dict:
    """The old and first repos' versions of Morgue entries: {"card": {MDW card (alnum): rows}} (oldrows archived.jsonl,
    review.jsonl 'morgue'), {"lost": {old id: rows}} (review.jsonl 'lost': the old repo's lost cards), {"first": {old
    id: picture rows}} (a first-repo file's PNG versions, one per blob: date, mapped author, subject; no fields)."""
    out = {"card": defaultdict(list), "lost": {}, "first": defaultdict(list)}
    for r in dm.read_jsonl(join(root, *OLD_ROWS, "archived.jsonl")):
        out["card"][dm.alnum(r.get("card"))].append(r)
    for item in dm.read_jsonl(join(root, *OLD_ROWS, "review.jsonl")):
        if item.get("kind") == "morgue":
            out["card"][dm.alnum(item.get("card"))] += item.get("rows") or []
        elif item.get("kind") == "lost":
            out["lost"][item.get("old_id")] = item.get("rows") or []
    authors = {k: v for k, v in (dm.read_json(AUTHORS, {}) or {}).items() if not k.startswith("_")}
    seen = set()
    for v in dm.read_jsonl(join(root, *OLD_VERSIONS)):
        if v.get("repo") == "first" and v.get("ext") == "png" and (v["old_id"], v["blob"]) not in seen:
            seen.add((v["old_id"], v["blob"]))
            out["first"][v["old_id"]].append({
                "id": v["id"], "ts": v["ts"], "date": v["date"], "repo": "first", "blob": v["blob"], "path": v["path"],
                "author": v.get("author_mapped") or authors.get(v.get("author"), v.get("author")),
                "subject": v.get("subject"), "fields": {}, "minor": False})
    return out


def entry_rows(entry: dict, kind: str, old: dict) -> list:
    """A Morgue entry's old and first repo rows, oldest first: an MDW card's own (by its name), a lost old-repo
    card's (old_repo), the first repo's files linked to it (first_files); one row a picture."""
    if kind == "mdw":
        rows = old["card"].get(dm.alnum(entry.get("card") or entry["name"])) or []
    else:
        rows = list(old["lost"].get((entry.get("old_repo") or {}).get("old_id")) or [])
        for f in entry.get("first_files") or ():
            rows += old["first"].get(f) or []
    found = {}
    for r in sorted(rows, key=lambda r: (r["ts"], r["id"])):
        found.setdefault(r["id"], r)
    return list(found.values())


def row_entry(r: dict) -> tuple:
    """(an old or first repo row as a Morgue timeline's entry, its webp name and job)."""
    name = f"O-{r['blob'][:12]}.webp"
    job = ("file", r["thumb"], None) if r.get("psd_only") else ("blob", r["blob"], None)
    fields = published(r.get("fields"))
    if (fields.get("ability") or {}).get("value") == "":
        fields.pop("ability")                      # no text read: a Morgue card's PNG has no OCR to tell 'none'
    if r.get("card_kind") in dc.ITEM_CARDS and "types" in fields:     # an item's kinds are never collected
        left = [t for t in fields["types"]["value"] if str(t).lower() not in dc.ITEM_KINDS]
        if left:
            fields["types"] = {**fields["types"], "value": left}
        else:
            fields.pop("types")
    e = {"id": "O-" + r["id"], "kind": "old repo", "date": dm.iso(r["ts"]), "day": dc.day(r["ts"]),
         "badge": BADGE["first repo" if r.get("repo") == "first" else "old repo"], "author": r.get("author"),
         "subject": r.get("subject"), "image": name, "fields": fields, "changes": {}, "minor": bool(r.get("minor"))}
    if r.get("repo") == "first":
        e["repo"] = "first"
    return e, (name, job)


def morgue_out(entry: dict, files: dict, kind: str, facts: list = (), rows: list = ()) -> tuple:
    """(a Morgue entry's published timeline, [(webp name, job)]): its pictures as versions, its old and first repo
    rows (entry_rows; they take the place of its old-repo picture) and its changelog marks, oldest first; its first
    (M'uru's "Before 2020" from a physical card photo, owner Q6, with no day); git_file: the repo that had its
    file."""
    entries, jobs = [], []
    for r in rows:
        e, job = row_entry(r)
        entries.append(e)
        jobs.append(job)
    for p in entry.get("pictures") or ():
        if p.get("source") == "old repo" and rows:
            continue                               # its rows show the old repo's file
        if p.get("source") == "old repo":
            source = ("thumb", p["thumb"], None)
            kind_of = "old repo"
        else:
            row = files.get(p["id"])
            source = ("file", row[0], row[1]) if row else ("thumb", p["thumb"], None)
            kind_of = "discord"
        name = sub(r"[^A-Za-z0-9_-]", "_", p["id"]) + ".webp"
        jobs.append((name, source))
        e = {"id": p["id"], "kind": kind_of, "date": p.get("date"), "day": (p.get("date") or "")[:10] or None,
             "badge": BADGE[kind_of], "image": name, "fields": {}, "changes": {}, "minor": False}
        if kind_of == "discord":
            e["channel"] = p.get("channel")
        entries.append(e)
    entries += morgue_marks(facts, entry)
    entries.sort(key=lambda e: e.get("date") or "9999")
    first = entry.get("first") or {}
    out = {"card": entry.get("slug") or entry["id"], "key": entry["name"], "name": entry["name"], "morgue": kind,
           "first": {"date": first.get("date"), "day": first.get("day") or (first.get("date") or "")[:10] or None,
                     "channel": first.get("channel") if first.get("source") == "discord" else None,
                     "source": first.get("source")},
           "entries": entries}
    if entry.get("kind"):
        out["kind_of"] = entry["kind"]             # joke | played
    git = "old repo" if entry.get("old_repo") else "first repo" if entry.get("first_repo") else None
    if git:
        out["git_file"] = git                      # a file in an older repo: lost before this repo, not before git
    return out, jobs


# ========================
# The card repo's first records
# ========================
def repo_log(cards_repo: str) -> list:
    """[(sha, time, {new path: old path} the bake told, [(raw meta, path)])] newest first: one `git log --raw` of the
    card repo's main over its PNGs and PSDs (trees only, never a blob)."""
    git_dir = dm.git_dir_of(cards_repo) if cards_repo else None
    raw = dm.git_out(git_dir, "log", f"origin/{CARD_BRANCH}", f"--format={LOG_FORMAT}", "--raw", "--no-abbrev",
                     "--no-renames", "-z", "--", "*.png", "*.psd") if git_dir else None
    if raw is None:
        return []
    commits, meta = [], None
    for token in raw.decode("utf-8", "replace").split("\0"):
        token = token.lstrip("\n")
        if meta is not None:
            commits[-1][3].append((meta, token))
            meta = None
        elif token.startswith(":"):
            meta = token
        elif token.startswith("C\t"):
            _, sha, at, _, trailers = (token.split("\t", 4) + ["", ""])[:5]
            told = {new: old for old, new in filter(None, map(rename_trailer, trailers.split("\x1f")))}
            commits.append((sha, float(at), told, []))
    return commits


def lineage(commits: list, paths: dict, kind: str = ".png") -> dict:
    """{card id: [(time, sha, path then, blob)] newest first}: every record of each card's file (paths: {card id: its
    path today}, kind: its suffix) in the card repo, followed back as the website's Image versions follow a PNG
    (build_site image_versions): an exact rename; a move that changed the file too, when that commit deleted exactly one
    file of the same name and added only this one; a rename the bake told. The matcher's stop follows exact and told
    renames only, so a card moved and redrawn at once started there (Elemental World, 4 Stars -> 3 Stars in 2025)."""
    name_of = lambda p: dm.alnum(splitext(basename(p or ""))[0])
    tracked = defaultdict(list)                  # path at the commit being read -> card ids
    for cid, path in paths.items():
        if path:
            tracked[path].append(cid)
    found = defaultdict(list)
    for sha, at, told, entries in commits:       # newest first
        adds, dels, mods = [], defaultdict(list), []
        for meta, path in entries:
            src, dst, status = (meta[1:].split() + [""] * 5)[2:5]
            if not path.lower().endswith(kind):
                continue
            if status.startswith("A"):
                adds.append((path, dst))
            elif status.startswith("D"):
                dels[src].append(path)
            elif dst != NO_BLOB:
                mods.append((path, dst))
        for path, blob in mods:
            for cid in tracked.get(path, ()):
                found[cid].append((at, sha, path, blob))
        gone, changed = defaultdict(list), defaultdict(list)
        for p in (p for ps in dels.values() for p in ps):
            gone[name_of(p)].append(p)
        for path, blob in sorted(adds, key=lambda a: a[0] not in tracked):     # tracked ones claim exact renames first
            for cid in tracked.get(path, ()):
                found[cid].append((at, sha, path, blob))
            if dels.get(blob):
                old = dels[blob].pop(0)
                if path in tracked:
                    tracked[old] = tracked.pop(path)
            else:
                changed[name_of(path)].append(path)
        for name, added in changed.items():
            for path in added:
                if path not in tracked:
                    continue
                ids, old = tracked.pop(path), told.get(path)
                if old and old in gone.get(name_of(old), ()):
                    tracked[old] = ids
                elif len(added) == 1 and len(gone.get(name, ())) == 1:
                    tracked[gone[name][0]] = ids
    return dict(found)


def git_first_changes(t: dict, last: Optional[dict], versions: list) -> dict:
    """git_first diffed again against the last version left (the chain's rule: discord_chain.first_changes)."""
    changes = dc.first_changes(last, t["git_first"].get("fields") or {}, t.get("kind"), versions, t.get("stop"),
                               dm.epoch(t["git_first"].get("date")))
    changes.pop("same_title", None)
    return changes


def stop_at_git(t: dict, pngs: list, psds: list, data: Optional[dict], versions: list = ()) -> Optional[dict]:
    """A chain timeline cut at its card's first record in the card repo (pngs: its PNG's lineage), when that is earlier
    than the chain's stop: its Discord and changelog entries from then on are left out, its stop moves back
    and its git_first is diffed again against the last version left. git_first is dated when its card-data version
    (data: discord_chain.first_versions) was first committed in the card's PSD lineage (psds), approx when the card had
    PSD versions before that one (its change is in one of them) or that version isn't there. Returns what changed:
    {stop, was, dropped, approx} or None."""
    stop, out = t.get("stop") or {}, None
    when, first = (dm.epoch(stop.get("date")) if stop else None), (pngs[-1] if pngs else None)
    if first and (when is None or first[0] < when - 1):
        dropped = [e["id"] for e in t["entries"]
                   if e["kind"] in BEFORE_GIT and (dm.epoch(e.get("date")) or 0) >= first[0]]
        t["entries"] = [e for e in t["entries"] if e["id"] not in dropped]
        t["stop"] = {"date": dm.iso(first[0]), "day": dm.iso(first[0])[:10], "source": "git", "commit": first[1],
                     "path": first[2]}
        out = {"stop": t["stop"]["day"], "was": stop.get("day"), "dropped": dropped}
        if t.get("git_first") and dropped:
            last = next((e for e in reversed(t["entries"]) if e["kind"] in ("discord", "old repo")), None)
            t["git_first"]["changes"] = git_first_changes(t, last, versions)
    gf = t.get("git_first")
    if gf and (t.get("stop") or {}).get("source") == "git" and data:
        at = min((r[0] for r in psds if r[3] == data.get("blob_sha")), default=None)
        if at:
            gf["date"] = dm.iso(at)
        gf["approx"] = not at or at > psds[-1][0] + 1
        out = {**(out or {}), "approx": gf["approx"]}
    return out


def mdw_firsts(root: str, cards_repo: Optional[str], commits: list) -> dict:
    """{MDW card (alnum): epoch of its first git record}: the earlier of the old repo's first commit of the old cards
    mapped to it (oldgit map.json, archived or Morgue) and its PNG's first record in the card repo (moves followed)."""
    out = {}
    for c in ((dm.read_json(join(root, *OLD_MAP), {}) or {}).get("cards") or {}).values():
        when = dm.epoch((c.get("first") or {}).get("date"))
        if c.get("status") in ("archived", "morgue") and when:
            key = dm.alnum(c.get("key"))
            out[key] = min(when, out.get(key, when))
    git_dir = dm.git_dir_of(cards_repo) if cards_repo and commits else None
    listed = dm.git_out(git_dir, "ls-tree", "--name-only", "-z", f"origin/{CARD_BRANCH}", MDW_FOLDER) \
        if git_dir else None
    pngs = [p for p in (listed or b"").decode("utf-8", "replace").split("\0") if p.lower().endswith(".png")]
    for key, rows in lineage(commits, {dm.alnum(splitext(basename(p))[0]): p for p in pngs}).items():
        out[key] = min(rows[-1][0], out.get(key, rows[-1][0]))
    return out


def mdw_names(cards_repo: Optional[str]) -> list:
    """The MDW Morgue cards' names: each PSD in MDW/The Morgue with its PNG (build_site's morgue_cards)."""
    listed = dm.git_out(dm.git_dir_of(cards_repo), "ls-tree", "--name-only", "-z", f"origin/{CARD_BRANCH}",
                        MDW_FOLDER) if cards_repo else None
    paths = set((listed or b"").decode("utf-8", "replace").split("\0"))
    return [splitext(basename(p))[0].replace("_", " ") for p in sorted(paths)
            if p.lower().endswith(".psd") and p[:-4] + ".png" in paths]


def mdw_cut(entry: dict, first: Optional[float]) -> list:
    """Stop at git for an MDW Morgue card: its Discord pictures posted at or after its first git record (first) are left
    out; returns their ids."""
    if first is None:
        return []
    gone = [p["id"] for p in entry.get("pictures") or () if p.get("source") == "discord" and p.get("date")
            and (dm.epoch(p["date"]) or 0) >= first]
    entry["pictures"] = [p for p in entry.get("pictures") or () if p["id"] not in gone]
    return gone


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
    """A version's picture, upright, RGBA (a sheet's cell: its box of the trimmed sheet, trimmed, as the matcher cut
    it)."""
    from PIL import Image, ImageOps
    Image.MAX_IMAGE_PIXELS = None
    if box:
        img = dm.trim(dm.load_image(source))
        w, h = img.size
        cell = img.crop((round(box[0] * w), round(box[1] * h), round(box[2] * w), round(box[3] * h)))
        return dm.trim(cell).convert("RGBA")
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
def title_reads(root: str) -> dict:
    """{archive file: the title band's own OCR text} from the text readers (whole pictures only)."""
    out = {}
    for r in dm.read_jsonl(join(root, *TEXT_READS)):
        ocr = (r.get("title") or {}).get("ocr")
        if r.get("src") == "discord" and isinstance(ocr, str) and ocr:
            out[r["file"]] = ocr
    return out


def build(root: str, old_git: str, workers: int = 3, force: bool = False, cards_repo: Optional[str] = LOCAL_DIR_LOC):
    root = dm.check_root(root)
    old_git = expanduser(old_git)
    out_dir = join(root, *OUT)
    makedirs(out_dir, exist_ok=True)
    created = dm.read_json(join(root, *CREATED), {}) or {}
    rulings = dm.read_json(join(root, *EYE), {}) or {}
    eye = {}                                     # {card: {entry id: {field}}}
    for key in rulings.get("drop") or {}:
        card, entry, field = key.split("|")
        eye.setdefault(card, {}).setdefault(entry, set()).add(field)
    out_of = defaultdict(set)                    # {card: {entry id}}: the eye's leave_out
    for key in rulings.get("leave_out") or {}:
        card, entry = key.split("|")
        out_of[card].add(entry)
    reads = title_reads(root)
    # stop at git: each card's first record in the card repo, its moves followed as the website follows them
    commits = repo_log(expanduser(cards_repo)) if cards_repo else []
    pairs = dm.read_json(dm.Paths(root).pairs, {}) or {}
    repo_cards = pairs.get("cards") or {} if commits else {}
    versions = defaultdict(list)                 # the matcher's card versions (their printed titles: git_first)
    for v in pairs.get("versions") or []:
        versions[v["card"]].append(v)
    del pairs
    pngs = lineage(commits, {cid: c.get("png") for cid, c in repo_cards.items()})
    psds = lineage(commits, {cid: c.get("psd") for cid, c in repo_cards.items()}, ".psd")
    data = dc.first_versions(dm.git_dir_of(expanduser(cards_repo))) if commits else {}
    if not commits:
        log("build: no card repo history: the chain's stops as they are")
    cut, left = [], {}
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
        taken = leave_out(t, out_of.get(t["card"]) or (), versions[t["card"]], root)
        if taken:
            left[t["card"]] = len(taken)
            if (c.get("ours") or {}).get("picture") in {p["id"] for e in taken for p in e.get("pictures") or ()}:
                index["created"].pop(t["key"], None)      # its Created came from a picture left out: no proposal
                c = {}
                log(f"build: {t['card']}: its Created source was left out by the eye: no proposal")
            if not t["entries"]:
                continue
        if commits:
            moved = stop_at_git(t, pngs.get(t["card"]) or [], psds.get(t["card"]) or [], data.get(t["key"]),
                                versions[t["card"]])
            if moved:
                cut.append((t["card"], moved))
            if not t["entries"]:
                continue
        mine = {"date": c.get("date"), "sort": c.get("sort"), "source": (c.get("ours") or {}).get("source"),
                "status": c.get("status")} if c else None
        out, pics = timeline_out(t, mine, eye.get(t["card"]), reads)
        folder = t["card"]
        want[folder] = set()
        for image, e in pics:
            want[folder].add(image)
            jobs.append((join(out_dir, folder, image), *picture_job(e, root)))
        counts.update(e["kind"] for e in out["entries"])
        index["cards"][folder] = {"key": t["key"], "name": t["name"], "dir": folder,
                                  "versions": sum(e["kind"] in PICTURE_KINDS for e in out["entries"]),
                                  "texts": sum(e["kind"] == "text" for e in out["entries"])}
        dm.write_json(join(out_dir, folder, "timeline.json"), out)
    morgue = dm.read_json(join(root, *MORGUE), {}) or {}
    files = {}
    for line in open(join(root, *VERDICTS)):
        r = loads(line)
        files[r["id"]] = (join(root, r["file"]), tuple(r["box"]) if r.get("box") else None)
    entries = morgue_entries(morgue, (dm.read_json(join(root, *IDEAS), {}) or {}).get("ideas") or [],
                             (dm.read_json(join(root, *RULINGS), {}) or {}).get("rulings") or {})
    facts = [f for f in dm.read_jsonl(join(root, *FACTS)) if str(f.get("card") or "").startswith("morgue:")]
    firsts, mdw_gone = mdw_firsts(root, expanduser(cards_repo) if cards_repo else None, commits), {}
    old_rows = morgue_rows(root)
    have = {dm.alnum(e.get("card") or e["name"]) for e, kind in entries if kind == "mdw"}
    for name in mdw_names(expanduser(cards_repo) if cards_repo and commits else None):
        if dm.alnum(name) not in have and old_rows["card"].get(dm.alnum(name)):   # an MDW card with no Discord picture
            entries.append(({"id": f"MDW-{dm.slugify(name)}", "name": name, "card": name, "pictures": [],
                             "first": {}}, "mdw"))
    for entry, kind in entries:
        if kind == "mdw":
            gone = mdw_cut(entry, firsts.get(dm.alnum(entry.get("card") or entry["name"])))
            if gone:
                mdw_gone[entry.get("card") or entry["name"]] = len(gone)
        mine = entry_rows(entry, kind, old_rows)
        out, pics = morgue_out(entry, files, kind, facts, mine)
        counts["morgue old repo"] += len(mine)
        if not any(e["kind"] in PICTURE_KINDS for e in out["entries"]):
            continue
        folder = f"{MORGUE_FOLDER}/{out['card']}"
        want[folder] = set()
        for image, (src_kind, source, box) in pics:
            want[folder].add(image)
            if src_kind == "thumb":
                jobs.append((join(out_dir, folder, image), "file", join(root, *MORGUE_DIR, source), None))
            elif src_kind == "file" and not source.startswith("/"):   # an old row's PSD composite: oldgit's thumb
                jobs.append((join(out_dir, folder, image), "file", join(root, "derived", source), None))
            else:
                jobs.append((join(out_dir, folder, image), src_kind, source, box))
        dm.write_json(join(out_dir, folder, "timeline.json"), out)
        row = {"dir": folder, "name": out["name"], "first": out["first"],
               "versions": sum(e["kind"] in PICTURE_KINDS for e in out["entries"]),
               **({"texts": n} if (n := sum(e["kind"] == "text" for e in out["entries"])) else {})}
        if kind == "candidate":
            index["morgue"].append({**row, "kind": entry.get("kind"), "id": entry["id"]})
        else:
            index["mdw"].append({**row, "card": entry.get("card") or entry["name"]})
    # webps: made where missing (or all with --force); none left that no entry names
    todo = list({j[0]: j for j in jobs if force or not exists(j[0])}.values())   # one job a webp (a blob twice)
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
            if sub_dir not in want and exists(join(out_dir, sub_dir, "timeline.json")):
                remove(join(out_dir, sub_dir, "timeline.json"))   # no history any more
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
    log(f"build: the eye's leave_out: {sum(left.values())} entries left out "
        f"({', '.join(f'{k} {n}' for k, n in sorted(left.items())) or 'none'})")
    log(f"build: MDW Morgue cards, stop at git: {sum(mdw_gone.values())} Discord pictures at or after the card's first "
        f"git record left out ({', '.join(f'{k} {n}' for k, n in sorted(mdw_gone.items())) or 'none'})")
    if commits:
        moved = [(c, m) for c, m in cut if m.get("stop")]
        left = ", ".join(f"{c} {len(m['dropped'])}" for c, m in moved if m["dropped"]) or "none"
        log(f"build: stop at git: {len(moved)} stops moved back to the card's first record (entries after it left "
            f"out: {left}); git_first approximate on {sum(1 for _, m in cut if m.get('approx'))} cards")
    return index


def main(argv: Optional[list] = None) -> int:
    parser = ArgumentParser(prog="python -m commands.discord_preview", description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("build", "check"))
    parser.add_argument("--root", default=environ.get("DISCORD_ARCHIVE", dm.DEFAULT_ROOT))
    parser.add_argument("--old-git", default=dm.old_store())
    parser.add_argument("--cards-repo", default=LOCAL_DIR_LOC, help="build: the card repo clone (its history, read only)")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--redo", action="store_true", help="check: read again only what last read otherwise")
    args = parser.parse_args(argv)
    if args.command == "build":
        build(args.root, args.old_git, args.workers, args.force, args.cards_repo)
    else:
        from commands.discord_preview_check import check
        check(args.root, args.workers, args.redo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
