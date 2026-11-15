"""
The Morgue candidates and the ideas never ported, from the card timelines' lost cards (docs/Future/CARD_TIMELINE.md,
stage 2, STAGE1 S7; the owner's answers Q1 and Q2, confirmed 11-13). Run after `discord_verdicts build`:

    python -m commands.discord_morgue build  [--root <dir>] [--cards-repo <dir>]
    python -m commands.discord_morgue sheets [--root <dir>]

build   Each lost card (derived/match/lost_groups.json) and each lost card of the old repo (derived/oldgit/lost.json) is
        a Morgue candidate (a joke card, or a card really played: a port or update picture, a classification channel's,
        an update changelog line, the old repo's file), an idea never ported (review only), or neither: already in
        MDW/The Morgue or MDW/Unfinished, an old name of a live card, a split, no card. The check by eye
        (derived/morgue/eye.json; the owner's rulings over it: derived/review/owner.json 'morgue') decides: name,
        status, the card it is, nearest live card, splits, merges, pictures added or dropped. A lost card with no eye
        entry keeps the automatic status (MDW names, archived renames, the lost group's) and is marked unchecked. A
        candidate or idea whose picture is the same picture as a row given to a live card or the unsure review (whole
        and art within TWIN bits) keeps that row only when the eye adds it (then listed in taken: the review page drops
        it there) or rules it another picture (not_twin); else it goes to review (status twin), so no picture is both a
        live card's and the Morgue's. The kin pass lists rows no entry holds that may be a lost card's (a none, art or
        unsure row whose title fits a candidate's or idea's name better than any live card's; a none row that is a joke
        picture, or a played channel's with card evidence) until the eye adds each to an entry, makes an entry of it
        (eye.json keyed by its picture id) or rules it not a lost card's (not_kin). It also lists the rows that may be
        an MDW/The Morgue or MDW/Unfinished card's (the whole within MDW_BITS of its PNG or the art within TWIN, a title
        like its name, the same picture as such an entry's): an MDW entry takes them as a candidate does (taken), and an
        entry whose picture is an MDW card's PNG but is ruled otherwise is listed in mdw_clash. M'uru's first is
        "Before 2020" (discord_chain.PHOTO_MORGUE, owner Q6 11-14: a physical card photo, itself never read), sorting
        first. Outputs under the archive's derived/morgue/ (never a repo):
        candidates.json (candidates, then the lost cards that are not, with why; taken; created_hints: the renamed and
        live ones' first dates for S6), ideas.json, thumbs/ (webp, every entry's pictures). Candidates go only to the
        website's Morgue list (cards.json morgue): never cards[], decks, Unused, the default search or Mentioned by. No
        message text, posters or authors.
sheets  Contact sheets for the check by eye: each candidate's and idea's thumbs beside its nearest live card.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from json import loads
from os import environ, makedirs
from os.path import exists, join
from shutil import copyfile
from sys import stderr
from typing import Callable, Optional

from commands import discord_chain as dc
from commands import discord_match as dm
from commands import discord_verdicts as dv

# ========================
# Configuration
# ========================
OUT = ("derived", "morgue")
LOST = ("derived", "match", "lost_groups.json")
VERDICTS = ("derived", "match", "verdicts.jsonl")
OLD_LOST = ("derived", "oldgit", "lost.json")
OLD_DIR = ("derived", "oldgit")                 # old lost cards' thumbs are relative to it
PAIRS = ("derived", "matcher", "match", "pairs.json")
OWNER = ("derived", "review", "owner.json")     # the owner's rulings outside the review page: morgue, entries as eye's
MORGUE_DIR = dv.MORGUE_DIR
UNFINISHED_DIR = "MDW/Unfinished/"
OLD_STATS = "db/old_stats.json"                 # the data branch's archived cards (renamed_to: a rename)
MDW_BITS = 5.0                                  # a row's whole this near an MDW card's PNG (or art within TWIN): kin
SAME_NAME = 0.9                                 # a title this near an MDW card's or an archived old name: that card
NAME_NEAR = 0.75                                # nearest live card by name at least this, else by look
FAR_BITS = 12.0                                 # a nearest look farther than this: far (discord_verdicts' far bits)
THUMB_W = 320
TWIN = dm.TWIN_BITS                             # whole and art this near: the same picture
LISTED = ("version_sure", "version_unsure", "draft", "unsure")     # rows a live card or the unsure review lists
KIN_NAME = dv.GROUP_TITLE                       # a row's title this near a candidate's or idea's name: kin (eye)
KIN_VERDICTS = ("none", "art", "unsure")        # rows no live card holds for sure
CANDIDATE, IDEA, TWIN_STATUS = "candidate", "idea", "twin"
MDW = {"morgue": MORGUE_DIR, "unfinished": UNFINISHED_DIR}      # statuses of cards already in MDW: their folder
TAKES = (CANDIDATE, IDEA, *MDW)                 # entries whose pictures leave a live card's or the unsure review
NOT = {"morgue": "already in MDW/The Morgue", "unfinished": "in MDW/Unfinished (kept, not retired)",
       "renamed": "an old name or draft of a live card (review: its timeline)", "live": "a live card's own "
       "picture (review: its timeline)", "split": "an old card that became "
       "two (owner Q3=A, review)", "not a card": "not a card of ours", "backup": "a backup copy of another card",
       "merged": "the same card as another lost card", "review": "no card evidence (review)",
       TWIN_STATUS: "the same picture as a live card's or an unsure row (review, never the Morgue unless the eye "
       "takes it)"}
ABOUT = ("Morgue candidates (owner's answers Q1, Q2, confirmed 11-13): joke cards (meme-cards, joke-ideas, cursed, "
         "idea-graveyard, or the same picture) and lost cards really played (a port or update picture, a "
         "classification channel's, an update changelog line, the old repo's file). Only for the website's Morgue "
         "list (cards.json morgue): never cards[], decks, Unused, the default search or Mentioned by. Each checked by "
         "eye (eye.json). not_candidates: lost cards that are not, with why. taken: rows listed for a live card or "
         "the unsure review that the eye gave to a candidate, an idea or a card already in MDW (the review page "
         "drops them there). M'uru (owner Q6, 11-14): first 'Before 2020' from a physical card photo (dated: its "
         "first post), sorting first; the photos themselves are never read. "
         "created_hints: renamed and live entries' first dates (S6: the live card's Created; weak ones only by the "
         "owner; meme: only meme channels date it, so never a live card's Created). counts.kin_open: rows that may be "
         "a lost card's or an MDW card's, not yet ruled by eye; counts.mdw_clash: pictures that are an MDW card's "
         "PNG in an entry ruled otherwise. why and eye notes: the review page's, never published. No message "
         "text, posters or authors.")
IDEAS_ABOUT = ("Ideas never ported (owner's answer Q2): card drafts never played, for the review page only (never "
               "published). Each checked by eye (eye.json). No message text, posters or authors.")


# ========================
# Helpers
# ========================
def plain(text: str) -> str:
    """A name squashed, a leading 'the' dropped."""
    form = dm.squash(text or "")
    return form[3:] if form.startswith("the") and len(form) > 6 else form


def near(a: str, b: str) -> float:
    return dm.ratio(plain(a), plain(b))


def utc(stamp: Optional[str]) -> Optional[str]:
    """An ISO time in UTC (old repo dates carry their offset)."""
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(stamp).astimezone(timezone.utc).isoformat()
    except ValueError:
        return stamp


def mdw_looks(git_dir: str, tree: dict, log: Callable = print) -> list:
    """[{place, name, path, whole, art}]: MDW/The Morgue's and MDW/Unfinished's PNGs (cards in a folder of MDW),
    hashed as the matcher hashes a card version; a blob not in the clone is skipped (never fetched)."""
    from io import BytesIO
    found = []
    for path in sorted(tree):
        place = next((k for k, d in MDW.items() if path.startswith(d)), None)
        if not place or not path.lower().endswith(".png") or place == "morgue" and \
                path[:-4] + ".psd" not in tree:     # the Morgue's cards: a PSD with its PNG (build_site)
            continue
        data = dm.git_out(git_dir, "cat-file", "blob", tree[path])
        if data is None:
            log(f"mdw look {path}: not in the clone, skipped")
            continue
        feat = dm.features(dm.load_image(BytesIO(data)), ocr=False, body=False, force_shape=True)
        found.append({"place": place, "name": dm.file_name(path), "path": path, "whole": feat["whole"],
                      "art": feat.get("art")})
    return found


def repo_known(cards_repo: str, log: Callable = print) -> dict:
    """{'morgue': {name: path}, 'unfinished': {name: path}, 'renamed': {archived name: today's name}, 'looks':
    mdw_looks}: the card repo's MDW/The Morgue (a PSD with its PNG, as build_site), MDW/Unfinished and the data
    branch's archived renames (renamed_to followed to the end)."""
    git_dir = dm.git_dir_of(cards_repo) if cards_repo else None
    if not git_dir:
        return {"morgue": {}, "unfinished": {}, "renamed": {}, "looks": []}
    tree = dm.repo_tree(git_dir, f"origin/{dm.CARD_BRANCH}")
    morgue = {dm.file_name(p): p for p in tree if p.startswith(MORGUE_DIR) and "/" not in p[len(MORGUE_DIR):]
              and p.lower().endswith(".psd") and p[:-4] + ".png" in tree}
    unfinished = {dm.file_name(p): p for p in sorted(tree) if p.startswith(UNFINISHED_DIR)
                  and p.lower().endswith((".png", ".psd"))}
    raw = dm.git_out(git_dir, "show", f"origin/{dm.DATA_BRANCH}:{OLD_STATS}")
    old = ((loads(raw) or {}).get("cards") or {}) if raw else {}
    step = {}
    for name, versions in old.items():
        tag = (versions[len(versions) - 1] or {}).get("archived") if versions else None
        if isinstance(tag, dict) and tag.get("renamed_to"):
            step[name] = tag["renamed_to"]

    def last(name: str) -> str:
        seen = {name}
        while name in step and step[name] not in seen:
            name = step[name]
            seen.add(name)
        return name
    return {"morgue": morgue, "unfinished": unfinished, "renamed": {n: last(n) for n in step},
            "looks": mdw_looks(git_dir, tree, log)}


def auto_status(group: dict, known: dict, live: dict) -> tuple:
    """(status, card) from the lost group and the repo's names; (None, None) when it is lost indeed."""
    titles = [t for t in [group.get("title")] if t]
    for place in ("morgue", "unfinished"):
        hit = next((n for n in known.get(place) or {} for t in titles if near(t, n) >= SAME_NAME), None)
        if hit:
            return place, hit
    if group.get("morgue_card"):
        return "morgue", group["morgue_card"]
    if group.get("split_of"):
        return "split", ",".join(group["split_of"].get("into") or [])
    if group.get("old_name_of"):
        return "renamed", group["old_name_of"]
    for old, new in (known.get("renamed") or {}).items():
        if any(near(t, old) >= SAME_NAME for t in titles) and not any(near(old, n) >= 1.0 for n in live.values()):
            return "renamed", live_id(new, live) or new
    return None, None


def live_id(name: str, live: dict) -> Optional[str]:
    """A live card's id by its name or key."""
    return next((cid for cid, n in live.items() if plain(n) == plain(name)), None)


def played_posts(posts: list, kinds: dict) -> list:
    """[(channel, day)]: the posts in port, update or classification channels (owner Q2=A), people's only."""
    found = set()
    for p in posts:
        thread = (p.get("thread") or "").lower()
        if not p.get("bot") and (kinds.get(p.get("channel_id")) in dv.PLAYED_KINDS or
                                 p.get("channel") in dv.PLAYED_NAMES or thread in dv.PLAYED_NAMES):
            found.add((p.get("thread") or p.get("channel"), (p.get("date") or "")[:10]))
    return sorted(found, key=lambda x: (x[1], x[0]))


def first_post(posts: list) -> Optional[dict]:
    """The earliest post a person made (no bot's or embed's), else any."""
    people = [p for p in posts if not (p.get("bot") or p.get("embed"))]
    return min(people or posts, key=lambda p: int(p["message"])) if posts else None


def picture_of(row: dict, takes: bool = False) -> dict:
    """One picture; taken (a candidate's, idea's or MDW card's): the list it leaves (a row a live card or the unsure
    review lists, added by eye)."""
    post = first_post(row.get("posts") or []) or {}
    taken = {"verdict": row["verdict"], "card": row_card(row)} if takes and row.get("verdict") in LISTED else None
    return {"id": row["id"], "date": post.get("date"), "channel": post.get("channel"),
            "thread": post.get("thread"), "cell": row.get("cell"), "w": row.get("w"), "h": row.get("h"),
            "thumb": f"thumbs/{row['id']}.webp", "source": "discord", **({"taken": taken} if taken else {})}


def image_of(row: dict, images: dict, by_sha: dict) -> dict:
    """A verdict row's matcher image (pairs.json images), a sheet cell's own."""
    keys = by_sha.get(row["sha256"]) or []
    image = (images.get(keys[0]) or {}) if keys else {}
    cells = image.get("cells") or []
    return cells[row["cell"] - 1] if row.get("cell") and len(cells) >= row["cell"] else image


def ocr_titles(image: dict) -> list:
    """An image's OCR'd titles that read as words (discord_verdicts' lost title rule)."""
    return [t for t in [image.get("title_block"), *(image.get("title") or [])]
            if t and dm.wordy(t) and len(dm.alnum(t)) >= dv.LOST_TITLE]


def kin_rows(rows: dict, held: set, names: list, live: dict, images: dict, by_sha: dict, eras: dict,
             kinds: dict) -> dict:
    """{picture id: why}: rows no entry holds that may be a lost card's, for the check by eye. (a) A none, art or
    unsure row whose title fits a candidate's, idea's or MDW card's name (names: [(label, name)]) at KIN_NAME, better
    than any live card's name; (b) a none row that is a joke picture (Q1: the flag follows the picture), or a played
    channel's (Q2) with card evidence (stat labels, a template's size, a card era, a card shape)."""
    found = {}
    for pid, row in rows.items():
        if pid in held or row.get("verdict") not in KIN_VERDICTS:
            continue
        image = image_of(row, images, by_sha)
        for title in ocr_titles(image) if names else []:
            fit, gid = max((near(title, n), gid) for gid, n in names)
            if fit >= KIN_NAME and fit > max((near(title, n) for n in live.values()), default=0.0):
                found[pid] = f"titled like {gid}"
                break
        if pid in found or row["verdict"] != "none":
            continue
        if row.get("joke"):
            found[pid] = "a joke picture"
        elif played_posts(row.get("posts") or [], kinds):
            evidence = dv.card_evidence(image, row.get("size_class"), eras.get(row["sha256"])) or (
                "a card shape" if image.get("shape") else None)
            if evidence:
                found[pid] = f"a played channel's picture ({evidence})"
    return found


class Hashes:
    """Rows' whole and art hashes as arrays: each row's distance in bits to one hash set (dm.hash_distance)."""
    def __init__(self, rows: list):
        np = self.np = dm._np()

        def table(part):
            return {k: np.frombuffer(b"".join(bytes.fromhex(r[part][k]) if r.get(part) else bytes(n) for r in rows),
                                     dtype=np.uint8).reshape(len(rows), n) for k, n in (("p", 8), ("d", 8), ("q", 32))}
        self.parts = {"whole": table("whole"), "art": table("art")}
        self.has_art = np.array([bool(r.get("art")) for r in rows])

    def dist(self, part: str, h: dict):
        np, arrays = self.np, self.parts[part]
        return sum(np.bitwise_count(arrays[k] ^ np.frombuffer(bytes.fromhex(h[k]), dtype=np.uint8)).sum(axis=1) /
                   (4 if k == "q" else 1) for k in ("p", "d", "q")) / 3


def twin_finder(rows: dict) -> Callable:
    """picture id -> the LISTED rows that are the same picture (whole and art within TWIN bits; the whole alone when
    either has no art)."""
    listed = [r for r in rows.values() if r.get("verdict") in LISTED and r.get("whole")]
    if not listed:
        return lambda pid: []
    table = Hashes(listed)

    def find(pid: str) -> list:
        row = rows.get(pid) or {}
        if not row.get("whole"):
            return []
        near = table.dist("whole", row["whole"]) <= TWIN
        if row.get("art"):
            near &= ~table.has_art | (table.dist("art", row["art"]) <= TWIN)
        return [listed[i]["id"] for i in table.np.flatnonzero(near) if listed[i]["id"] != pid]
    return find


def mdw_near(rows: list, looks: list) -> dict:
    """{picture id: (bits, look)}: each row's nearest MDW card PNG (mdw_looks) within MDW_BITS (whole), or with the
    art within TWIN (another text or look of that card)."""
    rows = [r for r in rows if r.get("whole")]
    if not rows or not looks:
        return {}
    table, found = Hashes(rows), {}
    for look in looks:
        whole = table.dist("whole", look["whole"])
        near = whole <= MDW_BITS
        if look.get("art"):
            near |= table.has_art & (table.dist("art", look["art"]) <= TWIN)
        for i in table.np.flatnonzero(near):
            pid, bits = rows[i]["id"], round(float(whole[i]), 1)
            if pid not in found or bits < found[pid][0]:
                found[pid] = (bits, look)
    return found


def mdw_ruled(x: dict, look: dict) -> bool:
    """An entry ruled the MDW card that look is (its status the card's folder, its card that name or a picture of it:
    Satou for Satou Matsustaka's entry, The One Who Rhymes2 for The One Who Rhymes)."""
    a, b = plain(x.get("card") or ""), plain(look["name"])
    return x["status"] == look["place"] and bool(a and b) and (a.startswith(b) or b.startswith(a))


def row_card(row: dict) -> Optional[str]:
    """The live card a listed row gives, or its first maybe."""
    return row.get("card") or next(iter(row.get("maybe") or []), None)


def unique_slug(name: str, used: set) -> str:
    """build_site's slug of the name, never one a live card, a Morgue card or another candidate has."""
    base = found = dm.slugify(name)
    if found in used:
        base = found = f"{base}-morgue"
    n = 2
    while found in used:
        found, n = f"{base}-{n}", n + 1
    used.add(found)
    return found


# ========================
# Build
# ========================
class Paths:
    def __init__(self, root: str):
        self.root = dm.check_root(root)
        self.out = join(self.root, *OUT)
        self.eye = join(self.out, "eye.json")
        self.thumbs = join(self.out, "thumbs")
        self.sheets = join(self.out, "sheets")


def build(root: str, log: Callable = print, cards_repo: str = dm.LOCAL_DIR_LOC) -> dict:
    return _build(Paths(root), repo_known(cards_repo, log), log)


def _build(paths: Paths, known: dict, log: Callable = print) -> dict:
    root = paths.root
    groups = (dm.read_json(join(root, *LOST), {}) or {}).get("groups") or []
    if not groups:
        raise SystemExit(f"no {join(root, *LOST)}: run python -m commands.discord_verdicts build first")
    rows = {r["id"]: r for r in dm.read_jsonl(join(root, *VERDICTS))}
    old_lost = (dm.read_json(join(root, *OLD_LOST), {}) or {}).get("cards") or []
    pairs = dm.read_json(join(root, *PAIRS), {}) or {}
    cards, images = pairs.get("cards") or {}, pairs.get("images") or {}
    by_sha = {}
    for key in sorted(images, key=lambda k: (int(images[k].get("message") or 0), k)):
        by_sha.setdefault(images[key].get("sha256"), []).append(key)
    live = {cid: c.get("name") or c.get("key") or cid for cid, c in cards.items()}
    eye = dm.read_json(paths.eye, {}) or {}
    owner = (dm.read_json(join(root, *OWNER), {}) or {}).get("morgue") or {}
    eye = {**eye, **{k: dict(v) for k, v in owner.items() if isinstance(v, dict)}}   # the owner's over the eye's
    made = {k: dict(v) for k, v in eye.items() if k.startswith("P-") and isinstance(v, dict)}   # keyed by a picture
    kinds = dv.channel_kinds(root)
    eras = dv.bar_reads(root)[2]
    used = set(cards) | {dm.slugify(n) for n in known.get("morgue") or {}}
    entries = {}
    for g in groups:
        e = dict(eye.get(g["id"]) or {})
        mine = next((p for p in g["pictures"] if p in made), None) if not e else None
        e = made.pop(mine) if mine else e       # a picture's ruling, once its row joined a lost group
        status, card = auto_status(g, known, live)
        if not status:
            status = CANDIDATE if g["morgue_candidate"] else IDEA if g["ideas_never_ported"] else "review"
        entries[g["id"]] = {"group": g, "eye": e, "status": e.get("status") or status,
                            "card": e.get("card") or card, "pictures": list(g["pictures"]), "old": None}
    for old in old_lost:                        # the old repo's lost cards: on their Discord group, else their own
        joined = next((x for x in entries.values() if (x["group"] or {}).get("old_lost") == old["id"] or
                       (x["eye"].get("old") == old["id"])), None)
        if joined:
            joined["old"] = old
            continue
        e = dict(eye.get(old["id"]) or {})
        entries[old["id"]] = {"group": None, "eye": e, "status": e.get("status") or CANDIDATE, "card": e.get("card"),
                              "pictures": [], "old": old}
    held = {p for x in entries.values() for p in x["pictures"] + list(x["eye"].get("add") or [])}
    mades = [rows[p] for p in made if p in rows and p not in held]
    looks = dv.nearest_looks(mades, pairs.get("versions") or [], cards) if mades else {}
    names = dm.Names(cards) if mades else None
    for p in sorted(set(made) - {r["id"] for r in mades}):
        log(f"eye {p}: {'no such row' if p not in rows else 'held by another entry'}, skipped")
    for row in mades:                           # lost cards the eye found among other rows (kin): keyed by a picture
        e, image = made[row["id"]], image_of(row, images, by_sha)
        title = next(iter(ocr_titles(image)), None)
        evidence = dv.card_evidence(image, row.get("size_class"), eras.get(row["sha256"])) or "by eye"
        by_name = [{"card": c, "name": (cards.get(c) or {}).get("name"), "ratio": round(r, 3)}
                   for c, (r, _) in names.fuzzy(title, 0.6, limit=3).items()] if title else []
        gid = dv.group_id(row["sha256"], row.get("cell"))
        assert gid not in entries, f"{gid} made twice"
        entries[gid] = {"group": None, "eye": e, "status": e.get("status") or "review", "card": e.get("card"),
                        "pictures": [row["id"]], "old": None, "title": title,
                        "made": {"title": title, "card_evidence": evidence,
                                 "nearest": {"look": looks.get(row["id"]) or [], "name": by_name}}}
    for gid, x in list(entries.items()):         # splits (by eye): one lost card each, its own title, the group kept
        parts = x["eye"].get("split") or []
        base = {k: v for k, v in x["eye"].items() if k != "split"}     # the first part keeps the group's rulings
        shared = {k: base[k] for k in ("status", "kind") if k in base}  # the others only its status and kind
        status, card = x["status"], x["card"]
        for n, part in enumerate(parts):
            own = {k: v for k, v in part.items() if k != "pictures"}
            row = rows[part["pictures"][0]]
            split = {"eye": {**(shared if n else base), **own}, "status": part.get("status") or status,
                     "card": part.get("card") or card, "pictures": list(part["pictures"]),
                     "title": row.get("title_read") or part.get("name")}
            if n == 0:
                x.update(split)
            else:
                entries[dv.group_id(row["sha256"], row.get("cell"))] = {"group": x["group"], "old": None, **split}
    for gid, x in entries.items():              # merges (by eye) and pictures added or dropped
        into = x["eye"].get("merge")
        if into in entries:
            entries[into]["pictures"] += x["pictures"]
            x.update(status="merged", card=into)
    for x in entries.values():
        drop = set(x["eye"].get("drop") or [])
        x["pictures"] = [p for p in dict.fromkeys(x["pictures"] + list(x["eye"].get("add") or [])) if p not in drop]
    first_files = {plain(f["name"]): f for f in (dm.read_json(join(root, *OLD_LOST), {}) or {}).get("first_repo") or []}
    for x in entries.values():                  # a file of that name in the first repo: played (owner Q2: a repo file)
        if x["status"] not in (CANDIDATE, IDEA):
            continue
        names = {x["eye"].get("name"), x.get("title"), (x["group"] or {}).get("title"), (x["old"] or {}).get("old_id")}
        hit = next((first_files[plain(n)] for n in sorted(names - {None, ""}) if plain(n) in first_files), None)
        if hit:
            x["first_repo"] = hit
            if x["status"] == IDEA and not hit.get("mdw"):     # MDW/: unfinished then, no proof it was played
                x.update(status=CANDIDATE, was_idea=True)
                if any("/Meme/" in p for p in hit["paths"]):     # the repo's meme folder: a joke card (Q1)
                    x["eye"] = {**x["eye"], "kind": x["eye"].get("kind") or "joke"}
    twins = twin_finder(rows)
    for x in entries.values():                  # twins of listed rows: taken by eye, else review
        if x["status"] not in (CANDIDATE, IDEA):
            continue
        unlike = set(x["eye"].get("not_twin") or [])
        x["twins"] = sorted({t for p in x["pictures"] for t in twins(p)} - set(x["pictures"]) - unlike)
        if x["twins"]:
            x.update(status=TWIN_STATUS, card=row_card(rows[x["twins"][0]]))
    held = {p for x in entries.values() for p in x["pictures"] + (x.get("twins") or [])}
    named = [(gid, n) for gid, x in entries.items() if x["status"] in TAKES
             for n in sorted({x["eye"].get("name"), x.get("title"), (x["group"] or {}).get("title")} - {None, ""})]
    named += [(f"{MDW[place]}{n}", n) for place in MDW for n in sorted(known.get(place) or {})]
    kin = kin_rows(rows, held, named, live, images, by_sha, eras, kinds)
    mdw = known.get("looks") or []
    for p, (bits, look) in mdw_near([r for k, r in rows.items() if k not in held], mdw).items():
        kin[p] = f"like {look['path']} ({bits} bits)"
    for gid, x in entries.items():              # MDW cards' entries: their twins are kin (eye), never a demotion
        if x["status"] in MDW:
            for t in sorted({t for p in x["pictures"] for t in twins(p)} - held):
                kin.setdefault(t, f"the same picture as {gid}'s")
    kin_open = sorted(set(kin) - set(eye.get("not_kin") or {}))     # each ruled by eye: added, made, or not_kin
    for p in kin_open:
        log(f"kin open {p}: {kin[p]}")
    near_mdw = mdw_near([rows[p] for p in held if p in rows], mdw)
    clash = sorted(f"{p} in {gid}: {look['path']} ({bits} bits)" for gid, x in entries.items()
                   if x["status"] != "merged" for p in x["pictures"] if p in near_mdw
                   for bits, look in [near_mdw[p]] if bits <= MDW_BITS and not mdw_ruled(x, look))
    for c in clash:                             # an MDW card's PNG in an entry ruled another card: fix the eye
        log(f"mdw clash {c}")
    candidates, ideas, others = [], [], []
    for gid, x in entries.items():
        entry = describe(gid, x, rows, kinds, live, used if x["status"] == CANDIDATE else None)
        (candidates if x["status"] == CANDIDATE else ideas if x["status"] == IDEA else others).append(entry)
    for found in (candidates, ideas, others):
        found.sort(key=lambda e: (dc.BEFORE_SORT if e["first"].get("day") else e["first"]["date"] or "9", e["id"]))
    thumbs(paths, candidates + ideas + others, rows)
    taken = [{"picture": p["id"], "entry": e["id"], **p["taken"]} for e in candidates + ideas + others
             for p in e["pictures"] if p.get("taken")]
    assert len({t["picture"] for t in taken}) == len(taken), "a picture taken twice"
    hints = [created_hint(e, rows) for e in others
             if e["status"] in ("renamed", "live", TWIN_STATUS) and e.get("card") in live]
    unchecked = [e["id"] for e in candidates + ideas + others if not e["eye"]["checked"]]
    counts = {"candidates": len(candidates), "jokes": sum(e["kind"] == "joke" for e in candidates),
              "played": sum(e["kind"] == "played" for e in candidates), "ideas": len(ideas),
              "not_candidates": dict(Counter(e["status"] for e in others)),
              "pictures": sum(len(e["pictures"]) for e in candidates), "taken": len(taken),
              "made": sum(bool(x.get("made")) for x in entries.values()), "kin": len(kin),
              "twins_open": [e["id"] for e in others if e["status"] == TWIN_STATUS], "kin_open": kin_open,
              "mdw_clash": clash, "unchecked": unchecked}
    stamp = dm.now_iso()
    dm.write_json(join(paths.out, "candidates.json"), {"generated": stamp, "about": ABOUT, "counts": counts,
                                                       "candidates": candidates, "not_candidates": others,
                                                       "taken": taken, "created_hints": hints})
    dm.write_json(join(paths.out, "ideas.json"), {"generated": stamp, "about": IDEAS_ABOUT, "ideas": ideas})
    log(dm.dumps(counts, indent=1))
    return counts


def created_hint(e: dict, rows: dict) -> dict:
    """A renamed or live entry's first date for S6: its earliest post that may date a live card (discord_verdicts
    dating: no meme channel's, bot's or embed's) or old repo file; meme: none, its date is a meme channel's (never a
    live card's Created)."""
    posts = [p for pic in e["pictures"] for p in (rows.get(pic["id"]) or {}).get("posts") or [] if dv.dating(p)]
    found = [(utc(p["date"]), p.get("channel")) for p in posts] + [
        (pic["date"], "old repo") for pic in e["pictures"] if pic.get("source") == "old repo" and pic.get("date")]
    date, channel = min(found) if found else (e["first"]["date"], e["first"]["channel"])
    return {"entry": e["id"], "name": e["name"], "card": e["card"], "date": date, "channel": channel,
            "weak": bool(e.get("weak")), "meme": not found}


def describe(gid: str, x: dict, rows: dict, kinds: dict, live: dict, used: Optional[set]) -> dict:
    """One lost card's entry: name, kind, first date and channel, pictures, nearest live card, why."""
    g, e, old = x["group"] or x.get("made") or {}, x["eye"], x["old"]
    pics = [picture_of(rows[p], x["status"] in TAKES) for p in x["pictures"] if p in rows]
    posts = [p for pid in x["pictures"] if pid in rows for p in rows[pid].get("posts") or []]
    if old:
        pics.append({"id": old["id"], "date": utc(old["first"]["date"]), "commit": old["first"]["commit"],
                     "thumb": f"thumbs/{old['id']}.webp", "source": "old repo"})
    dated = sorted((p for p in pics if p.get("date")), key=lambda p: utc(p["date"]))
    first = dated[0] if dated else {}
    played = played_posts(posts, kinds)
    joke = "joke" in (g.get("kinds") or [])
    kind = e.get("kind") or ("joke" if joke else "played")
    why = []
    if x["status"] == CANDIDATE:
        if kind == "joke":
            meme = sorted({p["channel"] for p in posts if p.get("morgue")})
            why.append(f"a joke card: posted in {', '.join(meme)}" if meme else
                       "a joke card: the same picture as one posted in a meme channel")
        why += [f"{'an' if c[:1] in 'aeiou' else 'a'} {c} picture ({d})" for c, d in played]
        if kind == "played" and g.get("changelog_days"):
            why.append(f"named in update changelogs on {', '.join(g['changelog_days'][:6])}")
        if x.get("first_repo"):
            f = x["first_repo"]
            why.append(f"the first repo's file {f['paths'][0]} ({f['first']['date'][:10]})" +
                       (", an idea until this was found" if x.get("was_idea") else ""))
        if old:
            end = (old.get("deleted") or {}).get("date") or (old.get("last") or {}).get("date")
            why.append(f"the old repo's file {old['paths'][0]} ({old['first']['date'][:10]} .. {(end or '')[:10]}"
                       f"{', deleted' if old.get('deleted') else ', at its end'})")
    else:
        why.append(NOT.get(x["status"], x["status"]) + (f": {x['card']}" if x.get("card") else ""))
    why += [f"{p['id']}: by {'the owner' if e.get('answer') else 'eye'} this card's; taken from "
            f"{p['taken']['card'] or 'no card'} ({p['taken']['verdict']})" for p in pics if p.get("taken")]
    for t in x.get("twins") or []:
        twin, post = rows[t], first_post(rows[t].get("posts") or []) or {}
        why.append(f"the same picture as {t} ({twin['verdict']} {row_card(twin)}, {post.get('channel')} "
                   f"{(post.get('date') or '')[:10]})")
    if e.get("weak"):
        why.append(f"a weak link, not picture-backed: {e['weak']}")
    if e.get("note"):
        why.append(e["note"])
    near_name = (g.get("nearest") or {}).get("name") or []
    near_look = (g.get("nearest") or {}).get("look") or []
    if e.get("nearest") or x["status"] in ("renamed", "live", TWIN_STATUS) and x.get("card") in live:
        cid = e.get("nearest") or x["card"]
        nearest = {"card": cid, "name": live.get(cid), "by": "eye" if e.get("nearest") or x["status"] !=
                   TWIN_STATUS else "twin"}
    elif near_name and near_name[0]["ratio"] >= NAME_NEAR:
        nearest = {**near_name[0], "by": "name"}
    elif near_look:
        nearest = {**near_look[0], "by": "look"}
    elif old and old.get("nearest"):
        n = old["nearest"][0]
        nearest = {"card": live_id(n["key"], live), "name": n["key"], "bits": n["d"], "by": "look"}
    else:
        nearest = None
    if nearest and nearest.get("bits", 0) > FAR_BITS:
        nearest["far"] = True                   # no look alike: a hint only
    name = e.get("name") or g.get("title") or (old or {}).get("old_id") or gid
    entry = {"id": gid, "name": name, "title_read": x.get("title") or g.get("title") or None,
             "status": x["status"],
             "kind": kind if x["status"] == CANDIDATE else None,
             "card": x.get("card") if x["status"] != CANDIDATE else None,
             "first": {"date": utc(first.get("date")), "channel": first.get("channel") or (
                 "old repo" if first.get("source") == "old repo" else None), "thread": first.get("thread"),
                 "source": first.get("source")},
             "last": max((utc(p["date"]) for p in dated), default=None),
             "channels": dict(Counter(p["channel"] for p in posts if p.get("channel"))),
             "pictures": pics, "thumb": pics[0]["thumb"] if pics else None,
             "nearest": nearest, "nearest_look": near_look, "nearest_name": near_name,
             "played": [{"channel": c, "date": d} for c, d in played], "changelog_days": g.get("changelog_days") or [],
             "card_evidence": g.get("card_evidence") or ("the old repo's file" if old else None) or
             ("the first repo's file" if x.get("first_repo") else None),
             "first_repo": {"paths": x["first_repo"]["paths"][:3], "date": x["first_repo"]["first"]["date"],
                            "mdw": bool(x["first_repo"].get("mdw"))} if x.get("first_repo") else None,
             "old_repo": {k: old.get(k) for k in ("id", "old_id", "paths", "first", "last", "deleted", "versions")}
             if old else None, "group": (x["group"] or {}).get("id") if x["group"] else None,
             "weak": True if e.get("weak") else None, "twins": x.get("twins"), "made": bool(x.get("made")) or None,
             "why": why, "eye": {"checked": bool(e), "note": e.get("note")}, "owner": e.get("answer")}
    if gid in dc.PHOTO_MORGUE:                   # owner Q6 (11-14 22:20): 'Before 2020', its photo never read
        entry["first"] = {"date": None, "channel": None, "thread": None, "source": dc.PHOTO_SOURCE,
                          "day": dc.BEFORE_DAY, "dated": entry["first"]}
        why.append(f"Created '{dc.BEFORE_DAY}': the owner's ruling (Q6, 11-14 22:20), from a photo of its physical card")
    if used is not None:
        entry["slug"] = unique_slug(name, used)
    return {k: v for k, v in entry.items() if v not in (None, [], {}, "")} | {"eye": entry["eye"],
                                                                              "first": entry["first"]}


def thumbs(paths: Paths, entries: list, rows: dict):
    """thumbs/<picture id>.webp, THUMB_W wide at most: a Discord picture (its cell cut out), an old repo card's own."""
    makedirs(paths.thumbs, exist_ok=True)
    old_dir = join(paths.root, *OLD_DIR)
    olds = {o["id"]: o for o in (dm.read_json(join(paths.root, *OLD_LOST), {}) or {}).get("cards") or []}
    for e in entries:
        for p in e["pictures"]:
            path = join(paths.out, p["thumb"])
            if exists(path):
                continue
            if p["source"] == "old repo":
                src = join(old_dir, olds[p["id"]]["thumb"])
                if exists(src):
                    copyfile(src, path)
                continue
            row = rows[p["id"]]
            img = dm.trim(dm.load_image(join(paths.root, row["file"])))
            if row.get("box"):
                w, h = img.size
                x0, y0, x1, y1 = row["box"]
                img = dm.trim(img.crop((round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h))))
            dm.save_thumb(img, path, THUMB_W)


# ========================
# Sheets (the check by eye)
# ========================
def sheets(root: str, log: Callable = print, per_sheet: int = 4) -> int:
    from PIL import Image, ImageDraw
    paths = Paths(root)
    pairs = dm.read_json(join(paths.root, *PAIRS), {}) or {}
    versions = {v["key"]: v for v in pairs.get("versions") or []}
    data = dm.read_json(join(paths.out, "candidates.json"), {}) or {}
    entries = (data.get("candidates") or []) + ((dm.read_json(join(paths.out, "ideas.json"), {}) or {}).get("ideas")
                                                or [])
    makedirs(paths.sheets, exist_ok=True)
    side, n = 200, 0
    blocks = []
    for e in entries:
        tiles = [join(paths.out, p["thumb"]) for p in e["pictures"]]
        ref = dm.card_thumb(versions, (e.get("nearest") or {}).get("card")) if e.get("nearest") else None
        tiles += [join(paths.root, *dm.OUT, ref)] if ref else []
        block = Image.new("RGB", (side * 8, 24 + ((len(tiles) + 7) // 8) * (side * 2 + 4)), (235, 235, 235))
        draw = ImageDraw.Draw(block)
        draw.text((4, 4), f"{e['id']}  {e['name']}  {e.get('kind') or e['status']}  {(e['first']['date'] or '')[:10]}"
                  f"  {e['first']['channel'] or ''}  | ref: {(e.get('nearest') or {}).get('card')}",
                  fill=(160, 0, 0), font=dv._font(15))
        for i, t in enumerate(tiles):
            try:
                img = Image.open(t).convert("RGB")
            except OSError:
                continue
            img.thumbnail((side - 4, side * 2))
            block.paste(img, ((i % 8) * side, 24 + (i // 8) * (side * 2 + 4)))
        blocks.append(block)
    for start in range(0, len(blocks), per_sheet):
        part = blocks[start:start + per_sheet]
        sheet = Image.new("RGB", (side * 8, sum(b.height for b in part)), "white")
        y = 0
        for b in part:
            sheet.paste(b, (0, y))
            y += b.height
        n += 1
        sheet.save(join(paths.sheets, f"sheet-{n:02d}.jpg"), quality=80)
    log(f"{n} sheets, {len(entries)} entries: {paths.sheets}")
    return n


def main(argv: Optional[list] = None) -> int:
    parser = ArgumentParser(prog="python -m commands.discord_morgue", description="Morgue candidates, ideas.")
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "sheets"):
        p = subs.add_parser(name)
        p.add_argument("--root", default=environ.get("DISCORD_ARCHIVE", dm.DEFAULT_ROOT))
        if name == "build":
            p.add_argument("--cards-repo", default=dm.LOCAL_DIR_LOC)
    args = parser.parse_args(argv)

    def log(text):
        print(text, file=stderr, flush=True)
    if args.command == "build":
        build(args.root, log, args.cards_repo)
    else:
        sheets(args.root, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
