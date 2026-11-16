"""
oldgit_index.py's map and report steps: each old card (index.json old_cards) to today's card key, and the text report.

Evidence, strongest first:
  path     the old card's files at the old history's end -> the same file in the current repo's init tree (36d62e53:
           same path, case, blob, or name) -> followed through the current repo's renames and moves (build_site's
           image_versions rules: exact renames, a moved-and-changed file of one name, the bake's Renamed trailers) to
           today's file (live) or to the commit that deleted it (archived: old_stats.json, renamed_to followed)
  name     a file name of the old card (any version) = today's key, printed title, file name or alias (Card Edits.json),
           or an archived name (renamed_to followed); '_Old'/'Old_' copies by their name without 'old'
  title    the old PSDs' printed titles (today's parser) = one of those names
  picture  pHash/dHash distance (discord_match.hash_distance, bits of 64) of the old PNGs and PSD composites vs today's
           PNG and the init tree's PNG (the first current-repo version) of each card, whole card and art
"""
import json
import os
import re
from collections import Counter, defaultdict
from os.path import basename, dirname, join, splitext

from oldgit_index import (FG, OG, OUT, STORE, TC_INIT, TENV, TG, alnum, ext_of, folder_stem, git, log, ls_tree,
                          read_jsonl, stem_key, today_cards, write_json)

REPORT = os.path.expanduser("~/Documents/Tools/wt/scratch/s1110/timelines/oldgit.txt")
CARD_EXTS = ("*.png", "*.PNG", "*.psd", "*.PSD")
ABILITY_KINDS = ("creature", "minion", "item", "aux item", "field")
MORGUE = "MDW/The Morgue/"   # retired cards (build_site MORGUE): status morgue, key = file name
PICTURE_SAME = 4.0          # bits: the same picture (a re-save, a small fix)
PICTURE_NEAR = 9.0          # bits: near enough to back a name or a path
PICTURE_GAP = 4.0           # bits: a picture match with nothing else behind it (an unsure one) beats the runner-up by
OLD_WORD = re.compile(r"(^|_| )old(_| |$)", re.IGNORECASE)
BACKUP = re.compile(r"(^|[_ -])(copy|bak|backup)([_ -]|$)", re.IGNORECASE)
SECOND = re.compile(r"^(.*?\D)[ _]?([2-9])$")   # 'Grapes2', 'Bag_of_Good 2': a card's second picture file
EYE = join(OUT, "eye.json")  # by eye: {ties: {old id: {key, status, why}}}: a lost old or first-repo file's live card


# ========================
# The current repo: init paths -> today's
# ========================
def tc_history(head: str) -> list:
    fmt = "%x01%H%x09%ct%x09%aN%x09%s%n%x04%(trailers:key=Renamed,valueonly,separator=%x1f)"
    out = git(TG, "log", "--raw", "--no-renames", "--no-abbrev", f"--format={fmt}", f"{TC_INIT}..{head}", "--",
              *CARD_EXTS, env=TENV).decode("utf-8", "replace")
    commits = []
    for chunk in out.split("\x01")[1:]:
        head_line, _, body = chunk.partition("\n")
        sha, ts, author, subject = (head_line.split("\t", 3) + [""] * 4)[:4]
        entries, told = [], {}
        for line in body.splitlines():
            if line.startswith("\x04"):
                if author.lower() == "autocroissant":
                    for value in line[1:].split("\x1f"):
                        old, sep, new = value.strip().partition(" -> ")
                        if sep and old and new:
                            told[new] = old
                continue
            if not line.startswith(":"):
                continue
            meta, _, path = line.partition("\t")
            parts = meta.split()
            if path.startswith('"'):
                path = path[1:-1].encode("latin-1", "backslashreplace").decode("unicode_escape").encode("latin-1") \
                    .decode("utf-8", "replace")
            if len(parts) >= 5:
                entries.append((parts[4], parts[2], parts[3], path))
        commits.append((sha, int(ts or 0), entries, told))
    return commits


def follow_tc(today_paths: list, commits: list) -> dict:
    """{path at the init tree: today's path, or 'archived:<path deleted>'} (newest commit first, as image_versions)."""
    tracked = {p: p for p in today_paths}
    for sha, ts, entries, told in commits:
        for ext in ("png", "psd"):
            adds, dels, gone_blob = [], defaultdict(list), {}
            for status, old, new, path in entries:
                if ext_of(path) != ext:
                    continue
                if status == "A":
                    adds.append((path, new))
                elif status == "D":
                    dels[old].append(path)
                    gone_blob[path] = old
            changed, used = defaultdict(list), set()
            for path, blob in sorted(adds, key=lambda a: a[0] not in tracked):
                sources = [s for s in dels.get(blob, []) if s not in used]
                if sources:
                    used.add(sources[0])
                    if path in tracked:
                        tracked[sources[0]] = tracked.pop(path)
                    elif sources[0] not in tracked:            # moved out of the card folders (MDW/Unfinished)
                        tracked[sources[0]] = f"archived:{sources[0]}"
                else:
                    changed[stem_key(path)].append(path)
            gone = defaultdict(list)
            for p in (p for ps in dels.values() for p in ps if p not in used):
                gone[stem_key(p)].append(p)
            for name, added in changed.items():
                for path in added:
                    if path not in tracked:
                        continue
                    key = tracked.pop(path)
                    if told.get(path) in gone.get(stem_key(told.get(path) or ""), ()):
                        tracked[told[path]] = key
                        used.add(told[path])
                    elif len(added) == 1 and len(gone.get(name, ())) == 1:
                        tracked[gone[name][0]] = key
                        used.add(gone[name][0])
            for path in gone_blob:                             # deleted here for good: an archived card's file
                if path not in used and path not in tracked:
                    tracked[path] = f"archived:{path}"
    return tracked


# ========================
# Names
# ========================
def resolve_archived(name: str, stats: dict, archived: dict, hops: int = 0) -> tuple:
    if name in stats:
        return ("live", name)
    tag = archived.get(name) or {}
    if tag.get("renamed_to") and hops < 10:
        return resolve_archived(tag["renamed_to"], stats, archived, hops + 1)
    return ("archived", name)


def name_index(stats: dict, pngs: dict, aliases: dict, archived: dict) -> dict:
    """{alnum text: {(status, key)}}."""
    found = defaultdict(set)
    for key, card in stats.items():
        texts = [key, card.get("display_name"), splitext(basename(card["path"]))[0], *aliases.get(key, [])]
        if pngs.get(key):
            texts.append(splitext(basename(pngs[key]))[0])
        for text in texts:
            if text and alnum(text):
                found[alnum(text)].add(("live", key))
    for name in archived:
        target = resolve_archived(name, stats, archived)
        if alnum(name):
            found[alnum(name)].add(target)
        path = (archived[name] or {}).get("path")
        if path and stem_key(path):
            found[stem_key(path)].add(target)
    return found


# ========================
# Pictures
# ========================
def _ints(look: dict, part: str):
    h = (look or {}).get(part)
    return (int(h["p"], 16), int(h["d"], 16), int(h["q"], 16)) if h else None


def distance(a, b) -> float:
    return ((a[0] ^ b[0]).bit_count() + (a[1] ^ b[1]).bit_count() + (a[2] ^ b[2]).bit_count() / 4) / 3


def picture_table(refs: list) -> dict:
    """{(status, key): [(whole ints, art ints, ref id)]}."""
    table = defaultdict(list)
    for ref in refs:
        if not ref.get("ok") or not ref.get("target"):
            continue
        table[ref["target"]].append((_ints(ref["look"], "whole"), _ints(ref["look"], "art"), ref["ref"]))
    return table


def nearest(looks: list, table: dict, limit: int = 4) -> list:
    """The cards nearest the old card's pictures: [{status, key, d (min of whole and art), whole, art}]."""
    best = []
    for target, refs in table.items():
        w = a = 99.0
        for lw, la in looks:
            for rw, ra, _ in refs:
                if lw and rw:
                    w = min(w, distance(lw, rw))
                if la and ra:
                    a = min(a, distance(la, ra))
        best.append({"status": target[0], "key": target[1], "d": round(min(w, a), 2), "whole": round(w, 2),
                     "art": round(a, 2)})
    best.sort(key=lambda x: (x["d"], x["whole"]))
    return best[:limit]


def picture_id(prefix: str, versions: list) -> str:
    """A stable id from the oldest picture's blob (a PNG, else any): prefix-<10 hex>."""
    pictures = [v for v in versions if v.get("ext") == "png"] or versions
    return f"{prefix}-{pictures[0]['blob'][:10]}" if pictures else f"{prefix}-none"


def to_target(looks: list, refs: list) -> dict:
    w = a = 99.0
    for lw, la in looks:
        for rw, ra, _ in refs:
            if lw and rw:
                w = min(w, distance(lw, rw))
            if la and ra:
                a = min(a, distance(la, ra))
    return {"d": round(min(w, a), 2), "whole": round(w, 2), "art": round(a, 2)}


# ========================
# map
# ========================
def map_main():
    index = json.load(open(join(OUT, "index.json")))
    versions = read_jsonl(join(OUT, "versions.jsonl"))
    reads = read_jsonl(join(OUT, "reads.jsonl"))
    refs = read_jsonl(join(OUT, "refs.jsonl"))
    stats, pngs, head, today_tree = today_cards()
    old_stats = json.loads(git(TG, "show", "origin/data:db/old_stats.json", env=TENV))["cards"]
    archived = {name: hist[len(hist) - 1].get("archived") for name, hist in old_stats.items()
                if hist and hist[len(hist) - 1].get("archived") and name not in stats}
    edits = json.loads(git(TG, "show", "origin/main:site/Card Edits.json", env=TENV))
    aliases = {k: v.get("aliases") or [] for k, v in (edits.get("cards") or {}).items() if isinstance(v, dict)}
    names = name_index(stats, pngs, aliases, archived)
    family = defaultdict(set)                                  # a card's variants and duplicates: one picture family
    for key, card in stats.items():
        root = card.get("variant_of") or card.get("duplicate_of") or key
        family[root].add(key)
    family_of = {k: frozenset(v) for root, v in family.items() for k in v}

    def related(a, b) -> bool:
        return a == b or (a[0] == b[0] == "live" and b[1] in family_of.get(a[1], ()))
    arch_by_path, arch_by_moved = {}, defaultdict(set)
    for name, tag in archived.items():
        if tag.get("path"):
            arch_by_path.setdefault(tag["path"], name)
        if tag.get("moved_to"):                                # a record named before a rename (Pandiety)
            arch_by_moved[basename(tag["moved_to"])].add(name)

    # today's file -> key (the PSD's and the PNG's)
    key_of_path = {}
    for dup in (False, True):                                  # a duplicate shares its original's PSD
        for key, card in stats.items():
            if bool(card.get("duplicate_of") or card.get("variant_of")) == dup:
                key_of_path.setdefault(card["path"], key)
    for key, path in pngs.items():
        key_of_path[path] = key

    def kind_of(path: str) -> str:
        parts = path.replace("Auxillary/", "Auxiliary/").split("/")
        if parts[0] == "N.M.E":                                # N.M.E. shop items came from Items/ (Double Pain)
            return "Items"
        return "/".join(parts[:2]) if parts[0] == "Auxiliary" else parts[0]

    def target_of(today_path: str) -> tuple:
        """(target, how): how 'tracked' (today's file), 'archived path' (old_stats' record of the deleted file) or
        'archived name' (the deleted file's name only: weak)."""
        if today_path.startswith("archived:"):
            path = today_path[len("archived:"):]
            if path in arch_by_path:
                return resolve_archived(arch_by_path[path], stats, archived), "archived path"
            name = splitext(basename(path))[0].replace("_", " ")
            moved = arch_by_moved.get(basename(path), ())
            if len(moved) == 1 and name not in archived and name not in stats:
                return resolve_archived(next(iter(moved)), stats, archived), "archived path"
            target = resolve_archived(name, stats, archived) if name in archived or name in stats else ("archived", name)
            if target[0] == "live" and kind_of(stats[target[1]]["path"]) != kind_of(path):
                target = ("archived", name)                    # a live card of that name, another kind (Pizza)
            return target, "archived name"
        key = key_of_path.get(today_path)
        if key:
            return ("live", key), "tracked"
        sibling = join(dirname(today_path), splitext(basename(today_path))[0])
        for p, k in key_of_path.items():
            if splitext(p)[0] == sibling:
                return ("live", k), "tracked"
        return None, None

    ties = (json.load(open(EYE)) if os.path.exists(EYE) else {}).get("ties") or {}
    commits = tc_history(head)
    tracked = follow_tc(list(today_tree), commits)
    init = {p: v[0] for p, v in ls_tree(OG, TC_INIT).items()}
    init_lower = defaultdict(list)
    init_blob = defaultdict(list)
    init_name = defaultdict(list)
    for p, b in init.items():
        init_lower[p.lower()].append(p)
        init_blob[b].append(p)
        init_name[(stem_key(p), ext_of(p))].append(p)
    init_found = {p: target_of(tracked[p]) if p in tracked else (None, None) for p in init}
    init_target = {}
    for p, (target, how) in init_found.items():
        if how == "archived name":                             # weak: its PSD/PNG partner's stronger target wins
            for q in init:
                if q != p and folder_stem(q) == folder_stem(p) and init_found[q][1] in ("tracked", "archived path"):
                    target = init_found[q][0]
                    break
        init_target[p] = target
    lost_init = sorted(p for p, t in init_target.items() if t is None)
    morgue_old = defaultdict(list)                             # blob -> MDW/The Morgue path (old tip, init)
    for commit in (index["tip"], TC_INIT):
        for entry in git(STORE, "ls-tree", "-z", commit, "--", MORGUE).decode("utf-8", "surrogateescape").split("\0"):
            if "\t" in entry:
                meta, path = entry.split("\t", 1)
                morgue_old[meta.split()[2]].append(path)
    morgue_today = set(git(TG, "ls-tree", "-z", "--name-only", head, "--", MORGUE, env=TENV)
                       .decode("utf-8", "surrogateescape").split("\0"))
    log(f"init tree: {len(init)} card files; {sum(1 for t in init_target.values() if t and t[0] == 'live')} reach a "
        f"live card, {sum(1 for t in init_target.values() if t and t[0] == 'archived')} an archived one, "
        f"{len(lost_init)} untracked")

    # refs -> targets: today's PNG by key, init PNG by its tracked target
    for ref in refs:
        if ref["ref"].startswith("today:"):
            ref["target"] = ("live", ref["key"])
        else:
            ref["target"] = init_target.get(ref["path"])
    table = picture_table(refs)

    psd_reads = defaultdict(list)
    png_looks = {}
    for r in reads:
        if r["kind"] == "psd":
            psd_reads[r["blob"]].append(r)
        elif r.get("ok"):
            png_looks[r["blob"]] = r["look"]
    by_old = defaultdict(list)
    for v in versions:
        by_old[v["old_id"]].append(v)
    commits_meta = index["commits"]
    position = {c: i for i, c in enumerate(index["order"])}

    cards, by_key, review = {}, defaultdict(list), []
    for oid, old in index["old_cards"].items():
        mine = sorted(by_old[oid], key=lambda v: position[v["commit"]])
        first_only = all(v.get("repo") == "first" for v in mine)
        if not first_only:                                     # the old repo's evidence alone, as before the first repo
            mine = [v for v in mine if v.get("repo") != "first"]
        stems = sorted({splitext(basename(p))[0] for p in old["paths"]})
        looks, thumbs = [], []
        for v in reversed(mine):
            if v["ext"] == "png" and v["blob"] in png_looks:
                looks.append((_ints(png_looks[v["blob"]], "whole"), _ints(png_looks[v["blob"]], "art")))
                thumbs.append(f"thumbs/{v['blob']}.webp")
            elif v["ext"] == "psd":
                for r in psd_reads.get(v["blob"], ()):
                    if isinstance(r.get("composite"), dict):
                        looks.append((_ints(r["composite"], "whole"), _ints(r["composite"], "art")))
                        thumbs.append(r.get("thumb"))
                        break
        titles, last_read = [], None
        for v in reversed(mine):
            if v["ext"] != "psd":
                continue
            for r in psd_reads.get(v["blob"], ()):
                if r.get("path") == v["path"] and r.get("read"):
                    last_read = last_read or r["read"]
                    t = r["read"].get("printed_title")
                    if t and t not in titles:
                        titles.append(t)
        evidence, how = {}, []

        # path: the files at the old history's end, else (deleted in the lost 2023 history) at its own end before it
        path_target = None
        for end, label in ((old["at_end"], None), (old.get("at_own_end") or {}, index.get("own_tip"))):
            for p in sorted(end):
                via, q = None, None
                if p in init:
                    via, q = "same path", p
                elif init_lower.get(p.lower()):
                    via, q = "same path (case)", init_lower[p.lower()][0]
                elif init_blob.get(end[p]):
                    via, q = "same blob", init_blob[end[p]][0]
                elif len(init_name.get((stem_key(p), ext_of(p)), ())) == 1:
                    via, q = "same file name", init_name[(stem_key(p), ext_of(p))][0]
                if q and init_target.get(q):
                    path_target = init_target[q]
                    evidence["path"] = {"end_path": p, "via": via, "init_path": q, "today": tracked.get(q),
                                        "target": list(path_target), **({"end": label[:8]} if label else {})}
                    if ext_of(p) == "psd":
                        break
            if path_target:
                break
        if path_target:
            how.append("path")

        # name, title
        def lookup(texts):
            hits = Counter()
            for t in texts:
                for target in names.get(alnum(t), ()):
                    hits[target] += 1
            return hits
        name_hits = lookup(stems)
        old_copy = False
        if not name_hits:
            plain = [OLD_WORD.sub(" ", s).strip(" _") for s in stems if OLD_WORD.search(s)]
            name_hits = lookup(plain)
            old_copy = bool(name_hits)
        title_hits = lookup(titles)
        if name_hits:
            evidence["name"] = {"texts": stems, "targets": [list(t) for t in name_hits], **({"old_copy": True} if old_copy else {})}
        if title_hits:
            evidence["title"] = {"texts": titles, "targets": [list(t) for t in title_hits]}

        # picture
        near = nearest(looks, table) if looks else []
        if near:
            evidence["picture"] = {"nearest": near, "looks": len(looks)}

        # decide
        candidates = Counter()
        if path_target:
            candidates[path_target] += 3
        for t in name_hits:
            candidates[t] += 2
        for t in title_hits:
            candidates[t] += 1
        if near and (near[0]["d"] <= PICTURE_NEAR and (len(near) < 2 or near[1]["d"] - near[0]["d"] >= PICTURE_GAP)):
            candidates[(near[0]["status"], near[0]["key"])] += 1
        target, confidence, why = None, None, []
        if candidates:
            ranked = candidates.most_common()
            target = ranked[0][0]
            pic = to_target(looks, table.get(target, [])) if looks else None
            if pic:
                evidence["picture_to_target"] = pic
            agree_pic = pic is not None and pic["d"] <= PICTURE_NEAR
            other_pic = near and not related(target, (near[0]["status"], near[0]["key"])) \
                and near[0]["d"] <= PICTURE_SAME and (pic is None or pic["d"] > PICTURE_NEAR)
            tied = len(ranked) > 1 and ranked[1][1] == ranked[0][1]
            if "picture" in evidence and near and (near[0]["status"], near[0]["key"]) == target:
                how.append("picture")
            if name_hits.get(target):
                how.append("name")
            if title_hits.get(target):
                how.append("title")
            if tied:
                why.append("evidence names several cards: " + ", ".join(f"{t[1]} ({t[0]})" for t, _ in ranked[:3]))
            if other_pic:
                why.append(f"the picture is another card's: {near[0]['key']} ({near[0]['d']} bits)")
            if path_target and path_target != target:
                why.append(f"the path leads to {path_target[1]}")
            if not path_target and not name_hits.get(target):
                if title_hits.get(target) and agree_pic:
                    pass                                       # printed title + picture: a rename
                else:
                    why.append("no path or file name backs it" + ("" if agree_pic else ", nor a near picture"))
            if path_target == target or name_hits.get(target):
                if not agree_pic and pic is not None and not path_target:
                    why.append(f"file name only; picture {pic['d']} bits off")
            if old_copy and not path_target:
                why.append("an old copy ('Old' in the file name)")
            confidence = "unsure" if why else "sure"
        if first_only and confidence == "unsure" and any("/Meme/" in p for p in old["paths"]):
            target, confidence, why = None, None, []           # a meme folder's copy (a joke on a live card): evidence
            evidence["meme_copy"] = True                       # for the Morgue's joke cards, never a live card's map
        if not target:                                         # moved into MDW/The Morgue: today's Morgue card
            for blob in [d["blob"] for d in old["deleted"]] + list(old["at_end"].values()):
                for p in morgue_old.get(blob, ()):
                    psd = splitext(p)[0] + ".psd"
                    if psd in morgue_today and splitext(psd)[0] + ".png" in morgue_today:
                        target = ("morgue", splitext(basename(psd))[0].replace("_", " "))
                        evidence["path"] = {"via": "moved to MDW/The Morgue", "today": psd, "target": list(target)}
            if target:
                confidence = "sure"
                how.append("path")
        record = {"key": target[1] if target else None, "status": target[0] if target else "lost",
                  "confidence": confidence, "how": how, "why_unsure": why or None,
                  "first": {"commit": old["first"], "date": commits_meta[old["first"]]["date"]},
                  "last": {"commit": old["last"], "date": commits_meta[old["last"]]["date"]},
                  "at_end": sorted(old["at_end"]), "paths": old["paths"],
                  "versions": {"psd": sum(v["ext"] == "psd" for v in mine), "png": sum(v["ext"] == "png" for v in mine),
                               "new_looks": sum(bool(v.get("new_look")) for v in mine)},
                  "authors": sorted({v["author_mapped"] or v["author"] for v in mine}),
                  "thumb": next((t for t in thumbs if t), None), "evidence": evidence}
        if old_copy:
            record["old_copy"] = True
        if any(BACKUP.search(s) for s in stems):
            record["note"] = "a backup or copy file (name)"
        if old["deleted"] and not old["at_end"]:
            record["deleted"] = {"commit": old["deleted"][-1]["commit"],
                                 "date": commits_meta[old["deleted"][-1]["commit"]]["date"]}
        if last_read:
            record["read"] = {k: last_read.get(k) for k in ("title", "type", "stars", "types", "stats")}
        cards[oid] = record
        meme = any("/Meme/" in p for p in old["paths"])
        if not target and first_only:                         # the first repo's 'Grapes2': a second picture of a card
            base = SECOND.match(stems[0]) if len(stems) == 1 else None
            hit = names.get(alnum(base.group(1))) if base else None
            if hit and len(hit) == 1 and next(iter(hit))[0] == "live":
                if meme:                                       # a meme folder's copy: a joke on it, evidence only
                    evidence["meme_copy"] = True
                else:
                    record.update(key=next(iter(hit))[1], status="second picture", confidence="sure", how=["name"])
        tie = ties.get(oid) if not target and not meme else None
        if tie and tie.get("key") in stats:                    # by eye: the picture is that live card's (Blooper, Eldman)
            record.update(key=tie["key"], status=tie.get("status") or "live", confidence="sure", how=["picture", "eye"],
                          eye=tie.get("why"))
            if record["status"] == "live":
                by_key[tie["key"]].append(oid)
        if first_only:
            record["repo"] = "first"
        if target:
            by_key[f"{target[1]}" if target[0] == "live" else f"{target[0]}:{target[1]}"].append(oid)
        if confidence == "unsure":
            review.append(oid)

    shared = {k: v for k, v in by_key.items() if len(v) > 1}
    for key, oids in shared.items():                           # several old cards on one card: say so
        for oid in oids:
            cards[oid]["shares_key_with"] = [o for o in oids if o != oid]
    status = Counter((c["status"], c["confidence"]) for c in cards.values())
    summary = {"old_cards": len(cards), "live_sure": status[("live", "sure")], "live_unsure": status[("live", "unsure")],
               "archived_sure": status[("archived", "sure")], "archived_unsure": status[("archived", "unsure")],
               "morgue": status[("morgue", "sure")], "lost": status[("lost", None)], "review": len(review),
               "live_cards_reached": len({c["key"] for c in cards.values() if c["status"] == "live"}),
               "keys_with_several_old_cards": len(shared),
               "by_how": dict(Counter("+".join(c["how"]) or "-" for c in cards.values()))}
    write_json(join(OUT, "map.json"), {
        "about": "Each old card (oldgit index.json old_cards: one file name's PSD+PNG over the old history, renames "
                 "followed) -> today's card key (stats.json; status live) or an archived card (old_stats.json; status "
                 "archived), a Morgue card (MDW/The Morgue; status morgue), or none (lost). confidence sure/unsure; "
                 "unsure ones are listed in review with why_unsure. "
                 "Picture distances in bits of 64 (discord_match.hash_distance).",
        "today": {"main": head, "cards": len(stats)}, "summary": summary, "review": sorted(review),
        "by_key": {k: sorted(v) for k, v in sorted(by_key.items())}, "cards": cards})
    lost, first_lost = [], []
    for oid, c in sorted(cards.items(), key=lambda kv: kv[1]["first"]["date"]):
        if c["status"] != "lost":
            continue
        if c.get("repo") == "first":                          # evidence only (played files): no candidate of its own
            first_lost.append({"old_id": oid, "name": oid, "paths": c["paths"], "first": c["first"],
                               "last": c["last"], "note": c.get("note"), "thumb": c["thumb"]})
            continue
        mine = sorted(by_old[oid], key=lambda v: position[v["commit"]])
        lost.append({"id": picture_id("OL", mine), "old_id": oid, "paths": c["paths"], "first": c["first"],
                     "last": c["last"], "deleted": c.get("deleted"), "at_end": c["at_end"], "authors": c["authors"],
                     "versions": c["versions"], "read": c.get("read"), "thumb": c["thumb"], "note": c.get("note"),
                     "nearest": (c["evidence"].get("picture") or {}).get("nearest", [])[:3]})
    write_json(join(OUT, "lost.json"), {
        "about": "Old cards with no counterpart among today's, archived or Morgue cards: Morgue candidates "
                 "(owner 11-13). id: stable, from the oldest picture's blob (OL-<10 hex>). "
                 "nearest: the closest pictures, for the owner's check. first_repo: the first repo's card files "
                 "(and its MDW/ files) of no live card, by name: evidence that a Morgue candidate of that name was "
                 "played (a file in the game's repo), never a candidate of its own.",
        "cards": lost, "first_repo": first_lost + first_mdw(index, {c["old_id"] for c in first_lost}, names)})
    log(f"map: {json.dumps(summary)}")
    return cards


def first_mdw(index: dict, seen: set, names: dict) -> list:
    """The first repo's MDW/ card files (pictures and PSDs) by name, a live card's name left out (an old copy of it):
    {name, paths, first}, oldest first."""
    if STORE is not FG:
        return []
    found = {}
    for c in index["order"]:
        meta = index["commits"][c]
        if meta.get("repo") != "first":
            continue
        out = git(STORE, "ls-tree", "-r", "--name-only", "-z", c, "--", "MDW").decode("utf-8", "surrogateescape")
        for path in out.split("\0"):
            if ext_of(path) in ("png", "psd", "jpg") and "/" in path:
                name = splitext(basename(path))[0].replace("_", " ")
                got = found.setdefault(name, {"name": name, "paths": [], "first": {"commit": c, "date": meta["date"]},
                                              "mdw": True})
                if path not in got["paths"]:
                    got["paths"].append(path)
    return [f for n, f in found.items() if n not in seen and not any(t[0] == "live" for t in names.get(alnum(n), ()))]


# ========================
# report
# ========================
def problem_kind(text: str) -> str:
    text = re.sub(r"\(.*", "", text)
    return re.split(r"[:;,]", text)[0].strip()[:60]


def report_main():
    index = json.load(open(join(OUT, "index.json")))
    versions = read_jsonl(join(OUT, "versions.jsonl"))
    reads = read_jsonl(join(OUT, "reads.jsonl"))
    refs = read_jsonl(join(OUT, "refs.jsonl"))
    mapped = json.load(open(join(OUT, "map.json")))
    lost = json.load(open(join(OUT, "lost.json")))["cards"]
    vocab = json.load(open(join(OUT, "vocabulary.json")))
    first_of = {}
    for v in versions:
        if v["ext"] == "psd":
            first_of.setdefault((v["blob"], v["path"]), v)
    lines = []
    out = lines.append
    out("OLDGIT: the old pre-wipe card repo's version index (card timelines, stage 1)")
    out(f"Data: {OUT}/ (versions.jsonl, reads.jsonl, refs.jsonl, map.json, lost.json, index.json, vocabulary.json, thumbs/)")
    out("")
    order = index["order"]
    out(f"Commits: {len(order)} old commits {index['commits'][order[0]]['date']} .. "
        f"{index['commits'][index['tip']]['date']} (tip {index['tip'][:8]}); the current repo's 10 init commits "
        f"(e9b568a1..36d62e53, 2023-11-05) are its own history and left out; cut {index['cut']}.")
    authors = Counter((index["commits"][c]["author"], index["commits"][c]["author_mapped"]) for c in order)
    out("Authors (raw -> mapped): " + ", ".join(f"{a} -> {m} ({n})" for (a, m), n in authors.most_common()))
    if index["unmapped_authors"]:
        out(f"UNMAPPED authors: {index['unmapped_authors']}")
    kinds = Counter(v["kind"] for v in versions)
    out(f"Versions: {len(versions)} card file versions ({sum(v['ext'] == 'psd' for v in versions)} PSD, "
        f"{sum(v['ext'] == 'png' for v in versions)} PNG); kinds {dict(kinds)}; new looks "
        f"{sum(bool(v.get('new_look')) for v in versions)} (moves and blobs a file had before are not).")
    out(f"Old cards: {len(index['old_cards'])} (one file name's PSD+PNG, renames/moves/case changes followed); "
        f"{sum(1 for o in index['old_cards'].values() if o['at_end'])} still in the tree at the end.")
    out("Scope: card PSD/PNG under Creatures/, Auxillary/, Items/, Field/ (rule pages and markers left out, as the "
        "website does); a version = a blob new at its path (add, edit, move+edit, copy; a pure move or a blob the file "
        "had before is kept with new_look false). Dates: ts = committer time (as the website dates versions), date = "
        "its ISO form, authored = author time. old_id joins versions.jsonl to map.json.")
    out("Hashes: discord_match.features(ocr=False) (pHash 64 'p', pHash 256 'q', dHash 'd'; whole card and art box "
        "ART, after trim, flattened on black), the matcher's own recipe; thumbs/<blob>.webp at most 480 px "
        "(PNG versions and PSD composites).")
    out("")
    psd = [r for r in reads if r["kind"] == "psd"]
    out(f"PSD READS (today's parser, PARSER_VERSION as in AC; vocabulary: today's Types/ + old-only {vocab['old_only_types']})")
    by_era = defaultdict(Counter)
    fails = Counter()
    problems = Counter()
    examples = defaultdict(list)
    by_type = defaultdict(Counter)
    by_size = defaultdict(Counter)
    fields = defaultdict(Counter)
    for r in psd:
        v = first_of.get((r["blob"], r["path"]))
        era = v["era"] if v else "?"
        kind = (r.get("read") or {}).get("type", "?")
        if not r.get("ok"):
            state = "fallback" if r.get("read") else "failed"
            fails["a path with no 'N Stars' folder: CardClassifier raises ValueError (read again under '0 Stars')"
                  if r.get("read") else r["error"][:100]] += 1
            examples[state].append(f"{r['path']} @{v['commit'][:8] if v else '?'}: {r['error'][:120]}")
        else:
            notes = (r["read"].get("problems") or []) + (r["read"].get("validator") or [])
            state = "clean" if not notes else "noted"
            for n in set(problem_kind(x) for x in notes):
                problems[n] += 1
                if len(examples[n]) < 4:
                    examples[n].append(f"{r['path']} @{v['commit'][:8] if v else '?'}")
        by_era[era][state] += 1
        by_type[kind][state] += 1
        read = r.get("read")
        if read:                                               # each field read where the card kind carries it
            stats = read["stats"]
            got = {"stars": bool(read.get("stars")) if r.get("ok") else None,
                   "types": bool(read["types"]) if kind in ("creature", "minion") else None,
                   "stats": all(x >= 0 for x in stats.values()) if kind in ("creature", "minion") else None,
                   "ability": bool((read["ability"] or "").strip()) if kind in ABILITY_KINDS else None}
            fields[era]["reads"] += 1
            fields["all"]["reads"] += 1
            for f, ok in got.items():
                if ok is not None:
                    for key in (era, "all"):
                        fields[key][f + "_n"] += 1
                        fields[key][f] += ok
            whole = all(ok is not False for ok in got.values()) and state == "clean"
            fields[era]["complete"] += whole
            fields["all"]["complete"] += whole
        by_size["x".join(map(str, r.get("size") or ["?"]))][state] += 1
    out(f"{len(psd)} reads ({len({r['blob'] for r in psd})} PSD blobs; a moved blob is read once per path).")
    out("parsed = today's parser read it as is; clean = parsed with no parser or validator note; fallback = the path "
        "raised (no 'N Stars' folder), read again under '0 Stars' (stars unknown)")
    out("era (commit month)  reads  parsed  clean   noted  fallback  failed")

    def era_line(label, c):
        n = sum(c.values())
        parsed = c["clean"] + c["noted"]
        out(f"  {label:17s} {n:6d}  {100 * parsed / n:5.1f}%  {100 * c['clean'] / n:5.1f}%  {c['noted']:5d}  "
            f"{c['fallback']:8d}  {c['failed']:6d}")
    for era in sorted(by_era):
        era_line(era, by_era[era])
    tot = Counter()
    for c in by_era.values():
        tot.update(c)
    era_line("all", tot)
    out("Fields read (share of the reads whose card kind carries the field: stars all; types and stats creatures and "
        "minions; ability creatures, minions, items, aux items, fields); complete = clean and every such field read")
    out("era (commit month)  reads  complete  stars   types   stats   ability")
    pct = lambda c, f: f"{100 * c[f] / c[f + '_n']:5.1f}%" if c[f + "_n"] else "    - "
    for era in sorted(k for k in fields if k != "all") + ["all"]:
        c = fields[era]
        out(f"  {era:17s} {c['reads']:6d}  {100 * c['complete'] / c['reads']:6.1f}%  {pct(c, 'stars')}  "
            f"{pct(c, 'types')}  {pct(c, 'stats')}  {pct(c, 'ability')}")
    out("by canvas size: " + "; ".join(f"{k} {sum(c.values())} ({100 * c['clean'] / sum(c.values()):.0f}% clean)"
                                       for k, c in sorted(by_size.items(), key=lambda kv: -sum(kv[1].values()))))
    out("by card kind (as the path classifies it): " + "; ".join(
        f"{k} {sum(c.values())} ({100 * c['clean'] / sum(c.values()):.0f}% clean, {c['failed']} failed)"
        for k, c in sorted(by_type.items(), key=lambda kv: -sum(kv[1].values()))))
    out("What fails (reads that raised):" if fails else "No read raised.")
    for k, n in fails.most_common():
        out(f"  {n:4d}  {k}")
    for e in (examples["failed"] + examples["fallback"])[:10]:
        out(f"        e.g. {e}")
    out("Notes on parsed reads (one count per read):")
    for k, n in problems.most_common(20):
        out(f"  {n:4d}  {k}   e.g. {'; '.join(examples[k][:2])}")
    diag_path = join(OUT, "diag_empty_reads.json")
    if os.path.exists(diag_path):                              # oldgit_diag.py: the layers behind empty reads
        diag = json.load(open(diag_path))
        count = lambda field: Counter(re.sub(r"in '.*'", "in the bars group", v[field]) for v in diag.values()
                                      if field in v)
        out("Why creatures/minions read no ability or no stats (oldgit_diag.py, per PSD blob):")
        for field in ("ability", "stats"):
            for k, n in count(field).most_common():
                out(f"  {field}: {n:4d}  {k}")
    comp = Counter("hashed" if isinstance(r.get("composite"), dict) else str(r.get("composite"))[:40]
                   for r in {r['blob']: r for r in psd}.values())
    out(f"PSD composites (Maximize Compatibility image): {dict(comp)}")
    png = [r for r in reads if r["kind"] == "png"]
    out(f"PNG versions: {len(png)} blobs, {sum(1 for r in png if r.get('ok'))} hashed + thumbnailed"
        + (f"; failed: {[r['blob'][:8] + ' ' + r.get('error', '')[:60] for r in png if not r.get('ok')][:5]}" if any(not r.get('ok') for r in png) else ""))
    out(f"Refs: {sum(1 for r in refs if r['ref'].startswith('today:'))} of today's card PNGs, "
        f"{sum(1 for r in refs if r['ref'].startswith('init:'))} init-tree PNGs hashed "
        f"({sum(1 for r in refs if not r.get('ok'))} failed)")
    out("")
    s = mapped["summary"]
    out("MAP (old card -> today's card key)")
    out(f"  {s['old_cards']} old cards: live sure {s['live_sure']}, live unsure {s['live_unsure']}, archived sure "
        f"{s['archived_sure']}, archived unsure {s['archived_unsure']}, Morgue {s.get('morgue', 0)}, lost {s['lost']}; "
        f"review list {s['review']}")
    out(f"  {s['live_cards_reached']} of today's {mapped['today']['cards']} cards get old-repo versions; "
        f"{s['keys_with_several_old_cards']} cards get several old cards")
    out(f"  evidence: {s['by_how']}")
    out("  Unsure (for review):")
    for oid in mapped["review"]:
        c = mapped["cards"][oid]
        out(f"    {oid} -> {c['key']} ({c['status']}): {'; '.join(c['why_unsure'] or [])}")
    out("  Several old cards on one card:")
    for key, oids in mapped["by_key"].items():
        if len(oids) > 1:
            out(f"    {key}: {oids}")
    out("")
    for oid, c in mapped["cards"].items():
        if c["status"] == "morgue":
            out(f"  Morgue: {oid} -> {c['evidence']['path']['today']} (moved there, still in today's Morgue)")
    out("")
    out(f"LOST (no live, archived or Morgue counterpart; Morgue candidates): {len(lost)}")
    for c in lost:
        near = c["nearest"][0] if c["nearest"] else None
        out(f"  {c['old_id']}: {', '.join(c['paths'][:3])}; {c['first']['date'][:10]}..{c['last']['date'][:10]}"
            f"{' deleted ' + c['deleted']['date'][:10] if c.get('deleted') else ''}; by {', '.join(c['authors'])}; "
            f"read {json.dumps({k: v for k, v in (c.get('read') or {}).items() if v not in (None, [], -1)})}"
            + (f"; nearest picture {near['key']} ({near['d']} bits)" if near else ""))
    text = "\n".join(lines) + "\n"
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
