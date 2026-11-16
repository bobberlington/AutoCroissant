#!/usr/bin/env python3
"""
The old card repo's version index for the card timelines (docs/Future/CARD_TIMELINE.md; owner 11-13). By hand.
Read-only on the old pre-wipe .git: only its history before the current repo was populated counts (owner 11-13: up to
TC 36d62e53, the last 'init', 2023-11-05 23:15 PST), minus those init commits themselves (the current repo's own).

    python oldgit_index.py index           versions.jsonl: each card PSD/PNG blob version per path (renames, moves and
                                           case changes followed; old_id: one old card); index.json (commits, old cards)
    python oldgit_index.py reads [-j N]    reads.jsonl: each PSD version through today's parser (card_data's path:
                                           variants, validator) + its composite's hashes; each PNG's pHash/dHash (whole,
                                           art: discord_match.features) + thumbs/<blob>.webp; refs.jsonl: the same
                                           hashes of today's card PNGs and the init commits' PNGs (first TC versions)
    python oldgit_index.py prune [--retry-failed]
                                           reads.jsonl down to the rows versions.jsonl uses; orphan thumbs deleted
    python oldgit_index.py map             map.json (old card -> today's card key + evidence; unsure ones for review),
                                           lost.json (no counterpart, live or archived)
    python oldgit_index.py report          the parse rate by era, what fails, the map's counts

Runs with the fork's Python from AC's scripts/ (else AC = the timelines worktree); oldgit_diag.py explains empty reads. Output under
~/Documents/Tools/discord-archive/derived/oldgit (never a repo). Card repo: GIT_NO_LAZY_FETCH=1, trees only.

The first repo (FIRST_GIT: discord_match.FIRST_GIT, env TIMELINES_FIRST_GIT; missing: an error): the same GitHub repo's history before the old .git, cut by the force-pushes of
2023-04-02 and 2023-04-21 (2022-08-24..2023-04-21), fetched read only into a bare repo of its own (blobless, its card PNGs
fetched; its objects/info/alternates reads the old .git's objects). Its main commits come first, as one history with the
old repo's (rows: repo 'first'); 'Initial No PSD's' (2023-04-02) re-made the tree before it, so that commit is its
parent here. The side branch the_restoration_project re-committed 2020 port looks in 2022 (no history of them): left out.
Authors not in authors.json map by their commit e-mail when another name with that e-mail maps (Kittycoon,
MichaelSrouji: panda2525252525's e-mail).

The lost 2023 history (LOST_GLOB in the first repo's store; missing: an error): main heads pushed 2023-10-26..11-05 and lost
by the force-pushes of 11-03 and 11-05 (GitHub's activity log), fetched by id. The old repo's own history (repo
'old', lost: true): 78 commits continue its main 268c2b1, then an unrelated history re-uploads a43fa582's tree in
batches ('1'..'8', UPLOADS: files not re-uploaded yet are carried, not deleted) and goes on to the wipe's c76646e2.
Output root: DISCORD_ARCHIVE (default ~/Documents/Tools/discord-archive).
"""
import json
import os
import re
import signal
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from os.path import basename, dirname, exists, expanduser, join, splitext

HOME = expanduser("~")
HERE = dirname(os.path.abspath(__file__))
AC = dirname(HERE) if exists(join(dirname(HERE), "commands", "psd_parser.py")) else join(HOME, "Documents/Tools/wt/timelines-1113")
OLD_GIT = join(HOME, "Downloads/Old TTSCardMaker/.git")
FIRST_REFS = ("refs/first/main-0402", "refs/first/main-0421")   # main before each force-push, oldest first
FIRST_LINKS = {"2056706e9ac92dfc024c3fc267154d13c78f972c": "7d95f1eb94d5b4e848028ab01b7c2286312b89d0"}
LOST_GLOB = "refs/lost2023/*"                       # the old repo's lost 2023 heads, in the first repo's store
# '1' (a root) follows a43fa582, the tree it re-uploads
LOST_LINKS = {"4ddc820971da13633fd79d9a64d34ddfb96b5563": "a43fa5827a238b43dedc8ad897b6548153c10032"}
UPLOADS = frozenset((                               # '1'..'8': a43fa582's tree re-uploaded in batches
    "4ddc820971da13633fd79d9a64d34ddfb96b5563", "eb29218e783be5c3fca42398b029526bbdb73dd9",
    "35750bbc76496b19ea5af84189bd85f1b13b6f8e", "1295bbf82a16b8adcf5fac846af13524f1939b8d",
    "da9fc176c31ce9ac172a387ac8fa8cb50957cb52", "bad43704fa849eb0bbb5f7b643f3fec16f8bdc9f",
    "fbbf72d77f6d13ff93d09c6da2313fe9d0ca07bf", "e63c5231ac66cf97b563274030e168a123bae43d"))
TC = join(HOME, "Desktop/TTSCardMaker")
OUT = join(expanduser(os.environ.get("DISCORD_ARCHIVE") or "~/Documents/Tools/discord-archive"), "derived", "oldgit")
AUTHORS = join(HOME, "Documents/Tools/wt/scratch/s1110/timelines/authors.json")
TC_INIT = "36d62e5336de5134f1471bb18bcca3c74cae0ef3"    # the current repo's last 'init' (2023-11-05 23:15:22 -08:00)
CUT = "2023-11-06T07:15:30Z"
CARD_DIRS = ("Creatures", "Auxillary", "Auxiliary", "Items", "Field", "N.M.E", "Cards")   # Cards: the first repo's
                                                                                           # creatures (to 2023-04-20)
EXTS = ("psd", "png")
MEME = "/meme/"                                             # the first repo's joke folders (Items/Meme, Cards/Meme)
NOT_CARDS = ("Rules", "Rulebook", "Markers")               # rule pages and markers: no cards (build_site, discord_match)
OLD_FOLDERS = (("Auxillary/", "Auxiliary/"), ("Cards/", "Creatures/"))   # old spellings: today's classifier reads today's
THUMB = 480
TASK_SECONDS = 600

sys.path.insert(0, AC)
os.environ["GIT_NO_LAZY_FETCH"] = "1"                     # the first repo is blobless: a missing blob is never fetched
from commands.discord_match import FIRST_GIT as FIRST_PATH, first_store  # noqa: E402  (the one store path)
FIRST_GIT = expanduser(FIRST_PATH)                        # the first repo (read only; it holds the old .git's objects)
OG = ["git", "--git-dir", OLD_GIT, "-c", "core.fsmonitor=false", "-c", "gc.auto=0", "--no-pager"]
FG = ["git", "--git-dir", FIRST_GIT, "-c", "core.fsmonitor=false", "-c", "gc.auto=0", "--no-pager"]
STORE = FG                                                # both repos' objects (old_commits stops when it is missing)
TG = ["git", "-C", TC, "-c", "core.fsmonitor=false", "-c", "gc.auto=0", "-c", "core.quotePath=false", "--no-pager"]
TENV = {**os.environ, "GIT_NO_LAZY_FETCH": "1"}


def git(base, *args, env=None, data=None) -> bytes:
    return subprocess.run(base + list(args), input=data, capture_output=True, check=True, env=env).stdout


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ext_of(path: str) -> str:
    return path.rsplit(".", 1)[-1].lower() if "." in basename(path) else ""


def alnum(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^0-9a-z]", "", text)


def stem_key(path: str) -> str:
    return alnum(splitext(basename(path))[0])


def folder_stem(path: str) -> tuple:
    return (dirname(path).lower(), splitext(basename(path))[0].lower())


def today_rel(path: str) -> str:
    for old, new in OLD_FOLDERS:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def is_card(path: str) -> bool:
    parts = path.split("/")
    return parts[0] in CARD_DIRS and ext_of(path) in EXTS and not any(p in NOT_CARDS for p in parts[1:-1])


def read_jsonl(path: str) -> list:
    if not exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: str, data, indent=1):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    os.replace(path + ".tmp", path)


def ls_tree(base, commit: str, env=None) -> dict:
    """{path: (blob, size)} of the commit's card PSDs/PNGs."""
    out = git(base, "ls-tree", "-r", "-l", "-z", commit, "--", *CARD_DIRS, env=env).decode("utf-8", "surrogateescape")
    found = {}
    for entry in out.split("\0"):
        if not entry:
            continue
        meta, path = entry.split("\t", 1)
        mode, kind, sha, size = meta.split()
        if kind == "blob" and is_card(path):
            found[path] = (sha, int(size) if size.isdigit() else None)
    return found


# ========================
# index
# ========================
def log_commits(base, *args) -> list:
    fmt = "%H%x1f%P%x1f%aI%x1f%cI%x1f%ct%x1f%an%x1f%ae%x1f%s%x1e"
    out = git(base, "log", "--topo-order", "--reverse", "--format=" + fmt, *args).decode("utf-8", "replace")
    found = []
    for rec in out.split("\x1e"):
        rec = rec.strip("\n")
        if rec:
            sha, parents, authored, committed, ct, author, email, subject = rec.split("\x1f")
            found.append((sha, {"parents": parents.split(), "authored": authored, "date": committed, "ts": int(ct),
                                "author": author, "email": email, "subject": subject}))
    return found


def parents_first(found: list, known: set) -> list:
    """found [(sha, meta)] reordered so each comes after its parents among them (a linked root after its link)."""
    mine, out, placed = {sha for sha, _ in found}, [], set(known)
    while found:
        rest = [(sha, m) for sha, m in found if any(p in mine and p not in placed for p in m["parents"])]
        out += [(sha, m) for sha, m in found if (sha, m) not in rest]
        placed |= {sha for sha, _ in out}
        if len(rest) == len(found):
            raise ValueError(f"no order for {[sha[:8] for sha, _ in rest]}")
        found = rest
    return out


def old_commits() -> tuple:
    """(commits parents first, {sha: meta}) of the old history before CUT, the current repo's init commits left out;
    the first repo's main commits before them (repo 'first'; FIRST_LINKS joins its two histories); the old repo's lost
    2023 commits after them (repo 'old', lost; LOST_LINKS)."""
    first_store()                                              # no store, no index: never the old .git alone
    own = set(git(OG, "rev-list", TC_INIT).decode().split())
    order, meta = [], {}
    if STORE is FG:
        for ref in FIRST_REFS:                                 # each ref's history, oldest first, once
            for sha, m in log_commits(FG, ref, *[f"^{r}" for r in FIRST_REFS[:FIRST_REFS.index(ref)]]):
                if sha not in meta:
                    order.append(sha)
                    meta[sha] = {**m, "repo": "first"}
        for sha, parent in FIRST_LINKS.items():
            if sha in meta and parent in meta:
                meta[sha]["parents"] = [parent]
                meta[sha]["linked"] = True
    for sha, m in log_commits(OG, "--all", "--before=" + CUT):
        if sha not in own and sha not in meta:
            order.append(sha)
            meta[sha] = {**m, "repo": "old"}
    if not git(FG, "for-each-ref", "--count=1", LOST_GLOB).strip():
        raise SystemExit(f"no {LOST_GLOB} in {FIRST_GIT}: the lost 2023 history is missing")
    if STORE is FG:
        lost = [(sha, {**m, "repo": "old", "lost": True}) for sha, m in log_commits(FG, "--glob=" + LOST_GLOB,
                                                                                   "--before=" + CUT)
                if sha not in own and sha not in meta]
        for sha, m in lost:
            if sha in LOST_LINKS:
                m.update(parents=[LOST_LINKS[sha]], linked=True)
        for sha, m in parents_first(lost, set(meta)):
            order.append(sha)
            meta[sha] = m
    return order, meta


def mapped_authors(meta: dict, authors: dict) -> dict:
    """{author name: mapped name}: authors.json, plus names it lacks whose commit e-mail a mapped name shares."""
    by_email = {}
    for m in meta.values():
        if m["author"] in authors and not m["author"].startswith("_"):
            by_email.setdefault(m.get("email"), authors[m["author"]])
    found = {k: v for k, v in authors.items() if not k.startswith("_")}
    for m in meta.values():
        if m["author"] not in found and by_email.get(m.get("email")):
            found[m["author"]] = by_email[m["email"]]
    return found


def index_main():
    os.makedirs(OUT, exist_ok=True)
    order, meta = old_commits()
    authors = mapped_authors(meta, json.load(open(AUTHORS)))
    log(f"{len(order)} old commits ({meta[order[0]]['date']} .. {max(m['date'] for m in meta.values())}), "
        f"{sum(m['repo'] == 'first' for m in meta.values())} of them the first repo's")
    trees = {c: ls_tree(STORE, c) for c in order}
    sizes = {sha: size for t in trees.values() for sha, size in t.values()}
    trees = {c: {p: v[0] for p, v in t.items()} for c, t in trees.items()}
    state, lineages, events, deletes = walk(order, meta, trees)
    finish(order, meta, authors, sizes, state, lineages, events, deletes)


def walk(order: list, meta: dict, trees: dict, carry=UPLOADS) -> tuple:
    """(state {commit: {path: (blob, lineage)}}, lineages, events, deletes) of the history, parents first: edits,
    moves (a case change, a moved and changed file of one name), copies, adds and deletions. A commit in carry keeps
    the files it hasn't re-uploaded yet (a batch of a re-upload: no deletion, no new lineage)."""
    state, lineages, events, deletes = {}, {}, [], []
    counter = Counter()

    def new_lineage(path, repo):
        prefix = "F" if repo == "first" else "L"               # the first repo's lineages: F1.. (L1.. as before)
        counter[prefix] += 1
        lid = f"{prefix}{counter[prefix]}"
        lineages[lid] = {"ext": ext_of(path), "paths": [path]}
        return lid

    for c in order:
        tree = trees[c]
        parents = [p for p in meta[c]["parents"] if p in state]
        had = defaultdict(list)
        for p in parents:
            for path, value in state[p].items():
                had[path].append(value)
        cur = {}

        def event(path, kind, lid, **extra):
            events.append({"commit": c, "path": path, "blob": tree[path], "kind": kind, "lineage": lid,
                           **{k: v for k, v in extra.items() if v is not None}})

        for path, blob in tree.items():                        # kept paths: an edit when no parent had this blob
            if path in had:
                lid = had[path][0][1]
                if all(b != blob for b, _ in had[path]):
                    event(path, "edit", lid)
                cur[path] = (blob, lid)
        gone = {path: values[0] for path, values in had.items() if path not in tree}
        by_blob, by_stem = defaultdict(list), defaultdict(list)
        for path, (blob, _) in sorted(gone.items()):
            by_blob[(blob, ext_of(path))].append(path)
            by_stem[(stem_key(path), ext_of(path))].append(path)
        used, moved = set(), {}

        def follow(path, src, kind):
            used.add(src)
            moved[path] = src
            lid = gone[src][1]
            cur[path] = (tree[path], lid)
            lineages[lid]["paths"].append(path)
            event(path, kind, lid, from_path=src)

        new = sorted(p for p in tree if p not in had)
        left = []
        for path in new:                                       # moved as is (a case change too)
            src = next((s for s in by_blob.get((tree[path], ext_of(path)), ()) if s not in used), None)
            if src:
                follow(path, src, "move")
            else:
                left.append(path)
        rest = []
        for path in left:                                      # moved and changed: a deleted file of this name
            cands = [s for s in by_stem.get((stem_key(path), ext_of(path)), ()) if s not in used]
            if cands:
                near = max(cands, key=lambda s: len(os.path.commonprefix([s.lower(), path.lower()])))
                follow(path, near, "move+edit")
            else:
                rest.append(path)
        last = []
        for path in rest:                                      # renamed and changed beside its partner's move
            src = None
            for partner, origin in moved.items():
                if folder_stem(partner) == folder_stem(path) and ext_of(partner) != ext_of(path):
                    want = folder_stem(origin)
                    src = next((s for s in gone if s not in used and folder_stem(s) == want
                                and ext_of(s) == ext_of(path)), None)
                    if src:
                        break
            if src:
                follow(path, src, "move+edit")
            else:
                last.append(path)
        here = defaultdict(list)                               # blobs still in this tree or a parent's
        for path, blob in tree.items():
            here[blob].append(path)
        for path, values in had.items():
            for blob, _ in values:
                here[blob].append(path)
        for path in last:
            copy = next((p for p in here.get(tree[path], ()) if p != path), None)
            lid = new_lineage(path, meta[c]["repo"])
            cur[path] = (tree[path], lid)
            event(path, "copy" if copy else "add", lid, copy_of=copy)
        for path, (blob, lid) in gone.items():                 # deleted here (a merge: gone from every side)
            if path in used:
                continue
            if c in carry:                                     # not re-uploaded yet: still there
                cur[path] = (blob, lid)
            elif len(had[path]) == len(parents):
                deletes.append({"commit": c, "path": path, "blob": blob, "lineage": lid})
        state[c] = cur
    return state, lineages, events, deletes


def finish(order: list, meta: dict, authors: dict, sizes: dict, state: dict, lineages: dict, events: list,
           deletes: list):
    """Old cards from the walk's lineages, then versions.jsonl and index.json."""
    # old cards: a lineage's PSD and PNG (same name), one name's lineages over time; none two at once
    parent = {lid: lid for lid in lineages}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    together = defaultdict(set)                                # (name, ext) -> lineages at once in one tree
    clash = set()
    for c in order:
        seen = defaultdict(set)
        for path, (blob, lid) in state[c].items():
            seen[(stem_key(path), ext_of(path))].add(lid)
        for key, lids in seen.items():
            if len(lids) > 1:
                clash.add(key[0])
                together[key] |= lids
    top = lambda path: today_rel(path).split("/", 1)[0].lower()
    for repo in ("L", "F"):                                    # each repo's lineages by name, as the old repo's were
        by_name, memes = defaultdict(list), []                 # (the first repo's by their last name: a misnamed file
        for lid, lin in lineages.items():                      # renamed must not join two cards: Metal Shield)
            if lid[0] == repo:
                if repo == "F" and MEME in lin["paths"][-1].lower():
                    memes.append(lid)
                    continue
                for path in lin["paths"] if repo == "L" else lin["paths"][-1:]:
                    by_name[stem_key(path)].append(lid)
        alone = defaultdict(list)                              # a Meme file joins its name in its top folder alone
        for lid in memes:                                      # (Alex YIIK's Cards/Meme -> Cards/Other), never another
            path = lineages[lid]["paths"][-1]                  # kind's: the joke item pain.png is not the debuff Pain
            same = [o for o in by_name.get(stem_key(path), ()) if top(lineages[o]["paths"][-1]) == top(path)]
            if same and stem_key(path) not in clash:
                parent[find(lid)] = find(same[0])
            else:
                alone[folder_stem(path) if stem_key(path) in clash else (stem_key(path), top(path))].append(lid)
        for lids in alone.values():
            for lid in lids[1:]:
                parent[find(lid)] = find(lids[0])
        for name, lids in by_name.items():
            if name in clash:                                  # two files of one name at once: by folder only
                groups = defaultdict(list)
                for lid in lids:
                    for path in lineages[lid]["paths"]:
                        groups[folder_stem(path)].append(lid)
                for same in groups.values():
                    for lid in same[1:]:
                        parent[find(lid)] = find(same[0])
                continue
            for lid in lids[1:]:
                parent[find(lid)] = find(lids[0])
    joined = join_first(lineages, find, events, meta)
    for f, o in joined.items():                                # a first-repo card into the old repo card it became
        parent[find(f)] = find(o)

    position = {c: i for i, c in enumerate(order)}
    lost = {c for c in order if meta[c].get("lost")}

    def newest(commits: list) -> str:                          # the newest commit no other of them has as a parent
        tip = commits[len(commits) - 1]
        for c in commits:
            if meta[c]["ts"] > meta[tip]["ts"] and not any(c in meta[o]["parents"] for o in commits):
                tip = c
        return tip
    tip = newest(order)                                        # the old history's newest commit
    own_tip = newest([c for c in order if c not in lost]) if lost else tip   # before the lost 2023 history
    groups = defaultdict(list)
    for lid in lineages:
        groups[find(lid)].append(lid)
    old_cards, of_lineage = {}, {}
    names_used = Counter()
    first_at = {}                          # names: the old repo's cards in their old order first, the first repo's,
    for e in events:                       # then those first seen in the lost 2023 history (ids kept: HR-H, the meme Tyler)
        root = find(e["lineage"])
        rank = (2 if e["commit"] in lost else int(e["lineage"][0] == "F"), position[e["commit"]])
        first_at[root] = min(first_at.get(root, rank), rank)
    for root, lids in sorted(groups.items(), key=lambda kv: first_at[kv[0]]):
        mine = [e for e in events if e["lineage"] in lids]
        latest = max(mine, key=lambda e: position[e["commit"]])
        named = max((e for e in mine if e["commit"] not in lost), key=lambda e: position[e["commit"]], default=latest)
        name = splitext(basename(named["path"]))[0].replace("_", " ")
        names_used[name] += 1
        oid = name if names_used[name] == 1 else f"{name} ({dirname(named['path'])})"
        while oid in old_cards:
            oid += "'"
        for lid in lids:
            of_lineage[lid] = oid
        end = {p: b for p, (b, lid) in state[tip].items() if lid in lids}
        own_end = {p: b for p, (b, lid) in state[own_tip].items() if lid in lids}
        old_cards[oid] = {"lineages": sorted(lids, key=lambda x: (x[0] == "L", int(x[1:]))),
                          "paths": sorted({p for lid in lids for p in lineages[lid]["paths"]}),
                          "first": min(mine, key=lambda e: position[e["commit"]])["commit"],
                          "last": latest["commit"],
                          "at_end": end, **({"at_own_end": own_end} if own_tip != tip and own_end != end else {}),
                          "deleted": [d for d in deletes if d["lineage"] in lids]}

    seen_blobs, first_seen = defaultdict(dict), defaultdict(dict)   # per lineage; per old card in the first repo
    rows = []
    for e in events:
        m = meta[e["commit"]]
        vid = f"{e['commit'][:12]}:{e['path']}"
        earlier = seen_blobs[e["lineage"]].get(e["blob"]) or first_seen[of_lineage[e["lineage"]]].get(e["blob"])
        row = {"id": vid, "old_id": of_lineage[e["lineage"]], "lineage": e["lineage"], "commit": e["commit"],
               "date": m["date"], "authored": m["authored"], "ts": m["ts"], "author": m["author"],
               "author_mapped": authors.get(m["author"]), "subject": m["subject"], "merge": len(m["parents"]) > 1,
               "path": e["path"], "ext": ext_of(e["path"]), "blob": e["blob"], "size": sizes.get(e["blob"]),
               "kind": e["kind"], "era": m["date"][:7], "repo": m["repo"], **({"lost": True} if m.get("lost") else {})}
        for key in ("from_path", "copy_of"):
            if e.get(key):
                row[key] = e[key]
        if e["kind"] == "move":
            row["new_look"] = False
        elif earlier:
            row["repeat_of"] = earlier
            row["new_look"] = False
        else:
            row["new_look"] = True
        seen_blobs[e["lineage"]].setdefault(e["blob"], vid)
        if m["repo"] == "first":                               # a blob the first repo had: no new look again
            first_seen[of_lineage[e["lineage"]]].setdefault(e["blob"], vid)
        rows.append(row)
    with open(join(OUT, "versions.jsonl.tmp"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(join(OUT, "versions.jsonl.tmp"), join(OUT, "versions.jsonl"))
    write_json(join(OUT, "index.json"), {
        "about": "Old pre-wipe card repo (read-only), commits before the current repo's population (TC "
                 f"{TC_INIT[:8]}), its init commits left out; the first repo's main commits before them (commits' "
                 "repo 'first'). versions.jsonl: one row per card file version.",
        "cut": CUT, "tc_init": TC_INIT, "tip": tip, "own_tip": own_tip,
        "commits": {c: {**meta[c], "author_mapped": authors.get(meta[c]["author"])} for c in order},
        "order": order, "old_cards": old_cards,
        "unmapped_authors": sorted({m["author"] for m in meta.values() if m["author"] not in authors})})
    kinds = Counter(r["kind"] for r in rows)
    log(f"versions: {len(rows)} ({dict(kinds)}); {sum(r['new_look'] for r in rows)} new looks; "
        f"{len(old_cards)} old cards from {len(lineages)} lineages; {len(deletes)} deletions; "
        f"clashing names: {sorted(clash)}")


def join_first(lineages: dict, find, events: list, meta: dict) -> dict:
    """{first-repo group root: old repo group root}: the old repo card each first-repo card became, by the file name
    and folder of its last first-repo path against the old cards' first paths. A first-repo card never joins two old
    ones (a rename inside the first repo keeps its last name only)."""
    last_path, first_paths = {}, defaultdict(list)
    for e in events:                                           # events are in commit order
        root = find(e["lineage"])
        if e["lineage"][0] == "F":
            last_path[root] = e["path"]
        elif e["path"] not in first_paths[root]:
            first_paths[root].append(e["path"])
    by_stem = defaultdict(set)
    for root, paths in first_paths.items():
        start = [p for p in paths if stem_key(p) == stem_key(paths[0])]   # the old card's first name
        for p in start:
            by_stem[stem_key(p)].add((root, dirname(today_rel(p)).lower()))
    joined = {}
    for root, path in last_path.items():                       # the same name in the same folder (Cards/ read as
        here = dirname(today_rel(path)).lower()                # Creatures/): a meme 'Something' is not Omori's
        roots = {r for r, folder in by_stem.get(stem_key(path), set()) if folder == here}
        if len(roots) == 1:
            joined[root] = next(iter(roots))
    return joined


# ========================
# reads (worker side)
# ========================
_W: dict = {}


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout(f"over {TASK_SECONDS} s")


def w_init(types, stars, manifest):
    os.chdir(AC)
    from commands.card_models import StatsDatabase
    from commands.psd_parser import IconVocabulary, PSDParser
    db = StatsDatabase()
    db.all_types = list(types)
    _W["parser"] = PSDParser(db.all_types, db, IconVocabulary.build(db.all_types, stars, manifest))
    _W["cat"] = subprocess.Popen(STORE + ["cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    signal.signal(signal.SIGALRM, _alarm)


def w_blob(sha: str) -> bytes:
    cat = _W["cat"]
    cat.stdin.write(sha.encode() + b"\n")
    cat.stdin.flush()
    head = cat.stdout.readline().split()
    if len(head) < 3 or head[1] != b"blob":
        raise KeyError(f"{sha} not in the old repo")
    data = cat.stdout.read(int(head[2]))
    cat.stdout.read(1)
    return data


def look_of(img_rgba) -> dict:
    """discord_match's hashes (pHash 64/256, dHash) of the whole card and its art, on the matcher's flattening."""
    from PIL import Image
    import commands.discord_match as dm
    im = img_rgba.convert("RGBA")
    if max(im.size) > dm.WORK_SIDE:
        im = im.copy()
        im.thumbnail((dm.WORK_SIDE, dm.WORK_SIDE), Image.LANCZOS)
    flat = Image.new("RGBA", im.size, dm.BACKGROUND)
    flat.alpha_composite(im)
    feat = dm.features(flat.convert("RGB"), ocr=False)
    return {k: feat[k] for k in ("w", "h", "shape", "whole", "art") if k in feat}


def save_thumb(img_rgba, sha: str) -> str:
    from PIL import Image
    name = f"thumbs/{sha}.webp"
    path = join(OUT, name)
    if not exists(path):
        thumb = img_rgba.convert("RGBA")
        thumb.thumbnail((THUMB, THUMB), Image.LANCZOS)
        thumb.save(path + ".tmp", "WEBP", quality=80, method=4)
        os.replace(path + ".tmp", path)
    return name


def blank(img) -> bool:
    lo_hi = img.convert("RGB").getextrema()
    return all(hi - lo <= 2 for lo, hi in lo_hi)


def card_fields(card, notes) -> dict:
    s = card.stats
    found = {"name": card.name, "title": card.display_name or card.name, "printed_title": card.display_name,
             "type": card.card_type, "stars": card.stars, "types": list(card.types or []),
             "stats": {"hp": s.hp, "def": s.defense, "atk": s.attack, "spd": s.speed},
             "ability": card.ability, "subtype": card.subtype, "series": card.series,
             "problems": list(card.problems or []), "validator": list(notes or [])}
    if card.creatures:
        found["creatures"] = [{"hp": c.hp, "def": c.defense, "atk": c.attack, "spd": c.speed} for c in card.creatures]
    return found


def w_psd(sha: str, paths: list) -> list:
    from psd_tools import PSDImage
    from commands.psd_parser import CardValidator, variants_of
    t0 = time.time()
    base = {"kind": "psd", "blob": sha}
    signal.alarm(TASK_SECONDS)
    try:
        data = w_blob(sha)
        base["bytes"] = len(data)
        try:
            psd = PSDImage.open(BytesIO(data))
        except Exception as e:
            return [{**base, "path": p, "ok": False, "error": f"open: {type(e).__name__}: {str(e)[:200]}"} for p in paths]
        base["size"] = [psd.width, psd.height]
        base["depth"], base["color_mode"] = psd.depth, str(psd.color_mode).split(".")[-1]
        try:
            comp = psd.topil()
            if comp is None:
                base["composite"] = None
            elif blank(comp):
                base["composite"] = "blank"
            else:
                base["composite"] = look_of(comp)
                base["thumb"] = save_thumb(comp, sha)
        except Exception as e:
            import traceback
            base["composite"] = f"error: {type(e).__name__}: {str(e)[:120]}"
            base["composite_trace"] = traceback.format_exc()[-600:]
        rows = []
        parser = _W["parser"]
        for path in paths:
            rel = today_rel(path)
            row = {**base, "path": path, "rel": rel}
            t1 = time.time()
            def read(rel):
                pictures = variants_of(psd)
                card = parser.parse(psd, rel) if len(pictures) < 2 else parser.parse(psd, rel, variant=1)
                row["read"] = card_fields(card, CardValidator.validate(card))
                if len(pictures) > 1:
                    row["variants"] = []
                    for n in pictures[1:]:
                        other = parser.parse(psd, rel, variant=n)
                        other.name = f"{card.name}{n}"
                        row["variants"].append({"n": n, **card_fields(other, CardValidator.validate(other))})
            try:
                read(rel)
                row["ok"] = True
            except Timeout:
                raise
            except Exception as e:
                row["ok"] = False
                row["error"] = f"parse: {type(e).__name__}: {str(e)[:200]}"
                parts = rel.split("/")
                if isinstance(e, ValueError) and parts[0] in ("Creatures", "Items", "Field") and \
                        not re.match(r"\d", parts[len(parts) - 2]):
                    try:                                       # an old path with no 'N Stars' folder: the rest read
                        read("/".join(parts[:-1] + ["0 Stars", parts[len(parts) - 1]]))
                        row["read"]["stars"] = None
                        row["fallback"] = "no 'N Stars' folder in the old path: read as '0 Stars', stars unknown"
                    except Timeout:
                        raise
                    except Exception as e2:
                        row["fallback_error"] = f"{type(e2).__name__}: {str(e2)[:200]}"
            row["secs"] = round(time.time() - t1, 2)
            rows.append(row)
        return rows
    except Timeout as e:
        return [{**base, "path": p, "ok": False, "error": f"timeout: {e}"} for p in paths]
    finally:
        signal.alarm(0)
        base["secs_total"] = round(time.time() - t0, 2)


def w_png(sha: str) -> list:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 200_000_000
    row = {"kind": "png", "blob": sha}
    signal.alarm(TASK_SECONDS)
    try:
        data = w_blob(sha)
        row["bytes"] = len(data)
        with Image.open(BytesIO(data)) as im:
            im.load()
            row["size"], row["mode"] = list(im.size), im.mode
            rgba = im.convert("RGBA")
        row["look"] = look_of(rgba)
        row["thumb"] = save_thumb(rgba, sha)
        row["ok"] = True
    except Exception as e:
        row["ok"] = False
        row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        signal.alarm(0)
    return [row]


def w_ref(ref: dict) -> list:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 200_000_000
    row = dict(ref)
    signal.alarm(TASK_SECONDS)
    try:
        if ref.get("file"):
            with open(ref["file"], "rb") as f:
                data = f.read()
        else:
            data = w_blob(ref["blob"])
        with Image.open(BytesIO(data)) as im:
            im.load()
            row["size"] = list(im.size)
            rgba = im.convert("RGBA")
        row["look"] = look_of(rgba)
        row["ok"] = True
    except Exception as e:
        row["ok"] = False
        row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        signal.alarm(0)
    return [row]


def w_task(task: tuple) -> tuple:
    kind = task[0]
    if kind == "psd":
        return task, w_psd(task[1], task[2])
    if kind == "png":
        return task, w_png(task[1])
    return task, w_ref(task[1])


# ========================
# reads (main side)
# ========================
def vocabularies() -> tuple:
    """Today's Types/ (+ Stars, manifest) as card_check.clone_vocabulary reads them, plus every old tree's Types/ names
    (offense, protective, undread...), names that are stars anywhere counted as stars."""
    from commands.card_check import clone_vocabulary, vocabulary
    today = clone_vocabulary(TC)
    types, stars = set(today.all_types), set(today.stars)
    order = json.load(open(join(OUT, "index.json")))["order"]
    seen_trees = set()
    for c in order:
        try:
            tree = git(OG, "rev-parse", f"{c}:Types").decode().strip()
        except subprocess.CalledProcessError:
            continue                                           # no Types/ yet
        if tree in seen_trees:
            continue
        seen_trees.add(tree)
        paths = [p for p in git(OG, "ls-tree", "-r", "--name-only", "-z", c, "--", "Types").decode().split("\0") if p]
        old = vocabulary(paths)
        types |= set(old.all_types)
        stars |= set(old.stars)
    old_types = sorted(types - set(today.all_types) - stars)
    return sorted(types - stars), sorted(stars), today.manifest, old_types


def today_cards() -> tuple:
    """(stats.json cards, {key: today's PNG path}, origin/main sha) from the card repo's data branch and main tree."""
    stats = json.loads(git(TG, "show", "origin/data:db/stats.json", env=TENV))["cards"]
    head = git(TG, "rev-parse", "origin/main", env=TENV).decode().strip()
    tree = ls_tree(TG, head, env=TENV)
    lower = {p.lower(): p for p in tree}
    pngs = {}
    for key, card in stats.items():
        folder = dirname(card["path"])
        for stem in (key.replace(" ", "_"), splitext(basename(card["path"]))[0]):
            hit = lower.get(f"{folder}/{stem}.png".lower())
            if hit:
                pngs[key] = hit
                break
    return stats, pngs, head, tree


def reads_main(workers: int, limit: int = 0):
    index = json.load(open(join(OUT, "index.json")))
    versions = read_jsonl(join(OUT, "versions.jsonl"))
    os.makedirs(join(OUT, "thumbs"), exist_ok=True)
    types, stars, manifest, old_types = vocabularies()
    log(f"vocabulary: {len(types)} types ({len(old_types)} only in the old repo: {old_types}), {len(stars)} stars")
    psd_paths = defaultdict(list)
    png_blobs = []
    for v in versions:                                         # oldest first: one file's versions near each other
        if v["ext"] == "psd":
            if v["path"] not in psd_paths[v["blob"]]:
                psd_paths[v["blob"]].append(v["path"])
        elif v["blob"] not in png_blobs:
            png_blobs.append(v["blob"])
    checked = git(STORE, "cat-file", "--batch-check=%(objectname) %(objecttype)", data=(
        "\n".join([*psd_paths, *png_blobs]) + "\n").encode()).decode().split("\n")
    held = {line.split()[0] for line in checked if line.endswith(" blob")}
    missing = [sha for sha in psd_paths if sha not in held]    # the first repo's PSDs are fetched only when needed
    psd_paths = {sha: paths for sha, paths in psd_paths.items() if sha in held}
    png_blobs = [sha for sha in png_blobs if sha in held]
    log(f"{len(missing)} PSD blobs not held (left unread)")
    done_reads = {(r["blob"], r.get("path")) for r in read_jsonl(join(OUT, "reads.jsonl"))}
    done_refs = {r["ref"] for r in read_jsonl(join(OUT, "refs.jsonl"))}
    tasks = []
    for sha, paths in psd_paths.items():
        todo = [p for p in paths if (sha, p) not in done_reads]
        if todo:
            tasks.append(("psd", sha, todo))
    tasks += [("png", sha) for sha in png_blobs if (sha, None) not in done_reads]
    stats, pngs, head, tree = today_cards()
    refs = [{"ref": f"today:{key}", "key": key, "path": path, "commit": head, "blob": tree[path][0],
             "file": join(TC, path)} for key, path in sorted(pngs.items())]
    init = ls_tree(OG, TC_INIT)
    refs += [{"ref": f"init:{path}", "path": path, "commit": TC_INIT, "blob": sha} for path, (sha, _) in sorted(init.items())
             if ext_of(path) == "png"]
    for ref in refs:                                           # today's file must be main's blob, else skipped
        if ref.get("file") and (not exists(ref["file"]) or git(["git"], "hash-object", ref["file"]).decode().strip() != ref["blob"]):
            ref["file"] = None
            ref["skipped"] = "working tree file is not origin/main's blob"
    tasks += [("ref", ref) for ref in refs if ref["ref"] not in done_refs and not ref.get("skipped")]
    if limit:                                                  # a trial: a few of each kind
        tasks = [t for k in ("psd", "png", "ref") for t in [t for t in tasks if t[0] == k][:limit]]
    log(f"{len(tasks)} tasks ({sum(t[0] == 'psd' for t in tasks)} PSD blobs, {sum(t[0] == 'png' for t in tasks)} PNG "
        f"blobs, {sum(t[0] == 'ref' for t in tasks)} refs); {len(done_reads)} reads already done; {workers} workers")
    skipped = [r for r in refs if r.get("skipped")]
    if skipped:
        log(f"{len(skipped)} today's PNGs skipped (not main's blob): {[r['key'] for r in skipped][:10]}")
    write_json(join(OUT, "vocabulary.json"), {"types": types, "stars": stars, "old_only_types": old_types})
    t0, n = time.time(), 0
    with open(join(OUT, "reads.jsonl"), "a", encoding="utf-8") as out_reads, \
            open(join(OUT, "refs.jsonl"), "a", encoding="utf-8") as out_refs, \
            ProcessPoolExecutor(max_workers=workers, initializer=w_init, initargs=(types, stars, manifest)) as pool:
        futures = [pool.submit(w_task, t) for t in tasks]
        for fut in as_completed(futures):
            task, rows = fut.result()
            target = out_refs if task[0] == "ref" else out_reads
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
            target.flush()
            n += 1
            if n % 50 == 0 or n == len(tasks):
                rate = n / max(1e-9, time.time() - t0)
                log(f"{n}/{len(tasks)} done, {rate:.2f}/s, ~{(len(tasks) - n) / max(rate, 1e-9) / 60:.0f} min left")
    log("reads done")


def prune_main(retry_failed: bool):
    """reads.jsonl down to the rows versions.jsonl uses (retry_failed: failed PSD reads dropped too, so the next
    reads run redoes them); thumbs no kept row names deleted."""
    versions = read_jsonl(join(OUT, "versions.jsonl"))
    want_psd = {(v["blob"], v["path"]) for v in versions if v["ext"] == "psd"}
    want_png = {v["blob"] for v in versions if v["ext"] == "png"}
    rows = read_jsonl(join(OUT, "reads.jsonl"))
    keep, seen = [], set()
    for r in rows:
        key = (r["blob"], r.get("path"))
        if key in seen:
            continue
        if r["kind"] == "psd" and key not in want_psd or r["kind"] == "png" and r["blob"] not in want_png:
            continue
        if retry_failed and r["kind"] == "psd" and not r.get("ok"):
            continue
        seen.add(key)
        keep.append(r)
    with open(join(OUT, "reads.jsonl.tmp"), "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(join(OUT, "reads.jsonl.tmp"), join(OUT, "reads.jsonl"))
    named = {r["thumb"] for r in keep if r.get("thumb")}
    gone = 0
    for name in os.listdir(join(OUT, "thumbs")):
        if f"thumbs/{name}" not in named:
            os.remove(join(OUT, "thumbs", name))
            gone += 1
    log(f"prune: {len(rows)} -> {len(keep)} reads; {gone} thumbs deleted")


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    if argv[0] == "index":
        index_main()
    elif argv[0] == "reads":
        workers = int(argv[argv.index("-j") + 1]) if "-j" in argv else 2
        reads_main(min(4, workers), int(argv[argv.index("--limit") + 1]) if "--limit" in argv else 0)
    elif argv[0] == "prune":
        prune_main("--retry-failed" in argv)
    elif argv[0] == "map":
        from oldgit_map import map_main
        map_main()
    elif argv[0] == "report":
        from oldgit_map import report_main
        report_main()
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
